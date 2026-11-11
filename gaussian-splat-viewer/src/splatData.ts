import * as THREE from 'three';
import type { SplatMesh } from '@sparkjsdev/spark';

/** Entpackte Kopie aller Splats eines Meshes (Objektkoordinaten). Dient als unveränderliche
 *  Quelle, aus der Segmentierung und Darstellungsmodi jederzeit neu geschrieben werden. */
export interface SplatData {
  count: number;
  centers: Float32Array;
  scales: Float32Array;
  quats: Float32Array;
  opacities: Float32Array;
  colors: Float32Array;
}

export type ViewMode = 'color' | 'clay' | 'normals';

export interface ViewOptions {
  mask: Uint8Array | null;
  scale: number;
  minOpacity: number;
  mode: ViewMode;
}

export function extractSplats(mesh: SplatMesh): SplatData {
  const count = mesh.packedSplats?.numSplats ?? 0;
  const data: SplatData = {
    count,
    centers: new Float32Array(count * 3),
    scales: new Float32Array(count * 3),
    quats: new Float32Array(count * 4),
    opacities: new Float32Array(count),
    colors: new Float32Array(count * 3),
  };
  mesh.forEachSplat((i, c, s, q, o, col) => {
    data.centers.set([c.x, c.y, c.z], i * 3);
    data.scales.set([s.x, s.y, s.z], i * 3);
    data.quats.set([q.x, q.y, q.z, q.w], i * 4);
    data.opacities[i] = o;
    data.colors.set([col.r, col.g, col.b], i * 3);
  });
  return data;
}

/** Normale eines Splats = Richtung seiner kürzesten Achse (Gaussians liegen flach auf Oberflächen). */
function splatNormal(data: SplatData, i: number, q: THREE.Quaternion, out: THREE.Vector3) {
  const sx = data.scales[i * 3];
  const sy = data.scales[i * 3 + 1];
  const sz = data.scales[i * 3 + 2];
  if (sx <= sy && sx <= sz) out.set(1, 0, 0);
  else if (sy <= sz) out.set(0, 1, 0);
  else out.set(0, 0, 1);
  return out.applyQuaternion(q);
}

/** Schreibt die Splats mit Maske, Größenfaktor, Deckkraft-Filter und Darstellungsmodus ins Mesh. */
export function writeSplats(mesh: SplatMesh, data: SplatData, opts: ViewOptions) {
  const packed = mesh.packedSplats;
  if (!packed) return;

  // Zwei Lichter in Weltkoordinaten, ins Objektsystem gedreht (Mesh kann geflippt sein).
  const toObject = mesh.quaternion.clone().invert();
  const key = new THREE.Vector3(0.5, 1, 0.7).normalize().applyQuaternion(toObject);
  const fill = new THREE.Vector3(-0.8, 0.2, -0.4).normalize().applyQuaternion(toObject);

  const c = new THREE.Vector3();
  const s = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const n = new THREE.Vector3();
  const col = new THREE.Color();

  for (let i = 0; i < data.count; i++) {
    const opacity = data.opacities[i];
    const visible = (!opts.mask || opts.mask[i] === 1) && opacity >= opts.minOpacity;
    c.fromArray(data.centers, i * 3);
    s.fromArray(data.scales, i * 3).multiplyScalar(opts.scale);
    q.fromArray(data.quats, i * 4);

    if (opts.mode === 'color') {
      col.fromArray(data.colors, i * 3);
    } else {
      splatNormal(data, i, q, n);
      if (opts.mode === 'normals') {
        col.setRGB(Math.abs(n.x), Math.abs(n.y), Math.abs(n.z));
      } else {
        // Zweiseitig beleuchteter "Ton"-Look: das Vorzeichen der Splat-Normale ist unbestimmt.
        const shade = 0.12 + 0.6 * Math.abs(n.dot(key)) + 0.28 * Math.abs(n.dot(fill));
        col.setRGB(0.9 * shade, 0.86 * shade, 0.8 * shade);
      }
    }
    packed.setSplat(i, c, s, q, visible ? opacity : 0, col);
  }
  packed.needsUpdate = true;
}

/** Robustes Zentrum (Median) und Radius (Distanz-Perzentil) der Splat-Zentren in Weltkoordinaten.
 *  Ausreißer wie weit entfernte Hintergrund-Splats (z. B. Scaniverse-Hülle) zählen kaum. */
export function robustBounds(
  data: SplatData,
  matrixWorld: THREE.Matrix4,
  mask: Uint8Array | null = null,
  radiusQuantile = 0.6,
): { center: THREE.Vector3; radius: number } {
  const step = Math.max(1, Math.floor(data.count / 100_000));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const p = new THREE.Vector3();
  for (let i = 0; i < data.count; i += mask ? 1 : step) {
    if (mask && mask[i] !== 1) continue;
    p.fromArray(data.centers, i * 3).applyMatrix4(matrixWorld);
    xs.push(p.x);
    ys.push(p.y);
    zs.push(p.z);
  }
  if (xs.length === 0) return { center: new THREE.Vector3(), radius: 1 };

  const quantile = (v: number[], qu: number) => {
    const sorted = Float64Array.from(v).sort();
    return sorted[Math.min(sorted.length - 1, Math.floor(qu * sorted.length))];
  };
  const center = new THREE.Vector3(quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5));
  const dists = xs.map((x, i) => Math.hypot(x - center.x, ys[i] - center.y, zs[i] - center.z));
  return { center, radius: Math.max(quantile(dists, radiusQuantile), 1e-4) };
}

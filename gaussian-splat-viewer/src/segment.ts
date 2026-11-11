import * as THREE from 'three';
import type { SplatData } from './splatData';

export interface SegmentParams {
  /** Angeklickter Punkt auf dem Objekt (Objektkoordinaten). */
  seed: THREE.Vector3;
  /** Richtung "oben" (Objektkoordinaten), um die Tischebene von Objektwänden zu unterscheiden. */
  up: THREE.Vector3;
  /** Typische Szenengröße; Suchradius und alle relativen Parameter beziehen sich darauf. */
  radius: number;
  /** Mindestabstand über der Tischebene, relativ zu `radius`. */
  planeOffset: number;
  /** Voxelgröße für das Region-Growing, relativ zu `radius`. */
  voxel: number;
  minOpacity: number;
}

export interface SegmentResult {
  mask: Uint8Array;
  selected: number;
  planeFound: boolean;
}

const MAX_TILT = Math.cos(THREE.MathUtils.degToRad(25));

/** Findet den ersten "echten" Oberflächenpunkt entlang eines Klickstrahls (Objektkoordinaten).
 *  Anders als ein reiner Raycast überspringt das einzelne, kaum sichtbare Floater vor dem Objekt:
 *  Ein Treffer zählt erst, wenn in seiner Umgebung genügend Splats liegen. */
export function pickSurface(
  data: SplatData,
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  coneTan: number,
  radius: number,
): THREE.Vector3 | null {
  const { centers, opacities, count } = data;
  const cell = 0.01 * radius;
  const keyOf = (x: number, y: number, z: number) =>
    `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;

  const density = new Map<string, number>();
  const candidates: { t: number; i: number }[] = [];
  for (let i = 0; i < count; i++) {
    if (opacities[i] < 0.1) continue;
    const x = centers[i * 3];
    const y = centers[i * 3 + 1];
    const z = centers[i * 3 + 2];
    const k = keyOf(x, y, z);
    density.set(k, (density.get(k) ?? 0) + 1);

    if (opacities[i] < 0.3) continue;
    const vx = x - origin.x;
    const vy = y - origin.y;
    const vz = z - origin.z;
    const t = vx * dir.x + vy * dir.y + vz * dir.z;
    if (t <= 0) continue;
    const perp = Math.hypot(vx - t * dir.x, vy - t * dir.y, vz - t * dir.z);
    if (perp < t * coneTan + 0.5 * cell) candidates.push({ t, i });
  }

  candidates.sort((a, b) => a.t - b.t);
  for (const { i } of candidates) {
    const x = centers[i * 3];
    const y = centers[i * 3 + 1];
    const z = centers[i * 3 + 2];
    if ((density.get(keyOf(x, y, z)) ?? 0) >= 8) return new THREE.Vector3(x, y, z);
  }
  return null;
}

/** Stellt das angeklickte Objekt frei:
 *  1. RANSAC sucht um den Klickpunkt die dominante, annähernd waagerechte Ebene (Tisch).
 *  2. Alles auf oder unter der Ebene fällt weg.
 *  3. Region-Growing über ein Voxelgitter behält nur die mit dem Klickpunkt verbundenen Splats. */
export function segmentObject(data: SplatData, p: SegmentParams): SegmentResult {
  const { centers, opacities, count } = data;
  const R = p.radius;
  const seed = p.seed;

  // --- Nachbarschaft um den Klickpunkt sammeln ----------------------------------
  const near: number[] = [];
  for (let i = 0; i < count; i++) {
    if (opacities[i] < p.minOpacity) continue;
    const dx = centers[i * 3] - seed.x;
    const dy = centers[i * 3 + 1] - seed.y;
    const dz = centers[i * 3 + 2] - seed.z;
    if (dx * dx + dy * dy + dz * dz < R * R) near.push(i);
  }

  // --- RANSAC: Tischebene --------------------------------------------------------
  const scoreSet = sample(near, 6000);
  const threshold = 0.012 * R;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const n = new THREE.Vector3();
  let bestCount = 0;
  const bestNormal = new THREE.Vector3();
  let bestD = 0;

  const random = mulberry32(1); // deterministisch: gleicher Klick → gleiches Ergebnis
  for (let iter = 0; iter < 600 && near.length >= 3; iter++) {
    a.fromArray(centers, near[(random() * near.length) | 0] * 3);
    b.fromArray(centers, near[(random() * near.length) | 0] * 3);
    c.fromArray(centers, near[(random() * near.length) | 0] * 3);
    n.subVectors(b, a).cross(c.sub(a));
    const len = n.length();
    if (len < 1e-12) continue;
    n.divideScalar(len);
    if (n.dot(p.up) < 0) n.negate();
    if (n.dot(p.up) < MAX_TILT) continue;
    const d = -n.dot(a);
    // Der Tisch liegt unter dem angeklickten Objekt.
    if (n.dot(seed) + d < threshold) continue;
    let inliers = 0;
    for (const i of scoreSet) {
      const dist = n.x * centers[i * 3] + n.y * centers[i * 3 + 1] + n.z * centers[i * 3 + 2] + d;
      if (Math.abs(dist) < threshold) inliers++;
    }
    if (inliers > bestCount) {
      bestCount = inliers;
      bestNormal.copy(n);
      bestD = d;
    }
  }

  // Eine Ebene zählt nur, wenn sie einen nennenswerten Teil der Nachbarschaft trägt.
  const planeFound = bestCount > scoreSet.length * 0.05;
  const minHeight = p.planeOffset * R;

  // --- Kandidaten oberhalb der Ebene, Voxelgitter um den Klickpunkt ---------------
  const h = p.voxel * R;
  const extent = 3 * R;
  const cells = Math.ceil((2 * extent) / h) + 2;
  const voxelKey = (i: number) => {
    const x = Math.floor((centers[i * 3] - seed.x + extent) / h);
    const y = Math.floor((centers[i * 3 + 1] - seed.y + extent) / h);
    const z = Math.floor((centers[i * 3 + 2] - seed.z + extent) / h);
    if (x < 0 || y < 0 || z < 0 || x >= cells || y >= cells || z >= cells) return -1;
    return (x * cells + y) * cells + z;
  };

  const grid = new Map<number, number[]>();
  let seedVoxel = -1;
  let seedDist = Infinity;
  for (let i = 0; i < count; i++) {
    if (opacities[i] < p.minOpacity) continue;
    if (planeFound) {
      const height =
        bestNormal.x * centers[i * 3] + bestNormal.y * centers[i * 3 + 1] + bestNormal.z * centers[i * 3 + 2] + bestD;
      if (height < minHeight) continue;
    }
    const key = voxelKey(i);
    if (key < 0) continue;
    let list = grid.get(key);
    if (!list) grid.set(key, (list = []));
    list.push(i);
    const dx = centers[i * 3] - seed.x;
    const dy = centers[i * 3 + 1] - seed.y;
    const dz = centers[i * 3 + 2] - seed.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < seedDist) {
      seedDist = d2;
      seedVoxel = key;
    }
  }

  // --- Region-Growing (26er-Nachbarschaft) -----------------------------------------
  // Voxel mit nur einem Splat leiten nicht weiter, damit einzelne Floater keine Brücken bauen.
  const mask = new Uint8Array(count);
  let selected = 0;
  if (seedVoxel >= 0) {
    const visited = new Set<number>([seedVoxel]);
    const queue = [seedVoxel];
    while (queue.length) {
      const key = queue.pop()!;
      const list = grid.get(key)!;
      for (const i of list) mask[i] = 1;
      selected += list.length;
      if (list.length < 2 && key !== seedVoxel) continue;
      const z = key % cells;
      const y = Math.floor(key / cells) % cells;
      const x = Math.floor(key / (cells * cells));
      for (let ox = -1; ox <= 1; ox++)
        for (let oy = -1; oy <= 1; oy++)
          for (let oz = -1; oz <= 1; oz++) {
            const nk = ((x + ox) * cells + (y + oy)) * cells + (z + oz);
            if (visited.has(nk) || !grid.has(nk)) continue;
            visited.add(nk);
            queue.push(nk);
          }
    }
  }
  if (planeFound) selected -= trimGroundSkirt(data, mask, bestNormal, bestD, h, GROUND_BAND * R);
  return { mask, selected, planeFound };
}

/** Höhe über der Tischebene, bis zu der raue Tischoberfläche noch als "Boden" gilt (relativ zu R). */
const GROUND_BAND = 0.08;

/** Entfernt bodennahe Splats, die seitlich neben dem Objekt liegen (Reste einer rauen Tischfläche).
 *  Der Umriss des Objekts wird aus den Splats oberhalb der Bodenzone auf die Ebene projiziert;
 *  in der Bodenzone bleibt nur, was unter diesem Umriss liegt. */
function trimGroundSkirt(
  data: SplatData,
  mask: Uint8Array,
  normal: THREE.Vector3,
  d: number,
  cell: number,
  band: number,
): number {
  const { centers, count } = data;
  const u = new THREE.Vector3().crossVectors(normal, Math.abs(normal.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)).normalize();
  const v = new THREE.Vector3().crossVectors(normal, u);
  const p = new THREE.Vector3();
  const cellOf = (i: number) => {
    p.fromArray(centers, i * 3);
    return `${Math.floor(p.dot(u) / cell)},${Math.floor(p.dot(v) / cell)}`;
  };
  const heightOf = (i: number) => normal.dot(p.fromArray(centers, i * 3)) + d;

  const footprint = new Set<string>();
  for (let i = 0; i < count; i++) {
    if (!mask[i] || heightOf(i) <= band) continue;
    const [cu, cv] = cellOf(i).split(',').map(Number);
    for (let ou = -1; ou <= 1; ou++) for (let ov = -1; ov <= 1; ov++) footprint.add(`${cu + ou},${cv + ov}`);
  }
  // Kein Objekt oberhalb der Bodenzone: nichts beschneiden.
  if (footprint.size === 0) return 0;

  let removed = 0;
  for (let i = 0; i < count; i++) {
    if (!mask[i] || heightOf(i) > band) continue;
    if (!footprint.has(cellOf(i))) {
      mask[i] = 0;
      removed++;
    }
  }
  return removed;
}

function sample(arr: number[], max: number): number[] {
  if (arr.length <= max) return arr;
  const random = mulberry32(2);
  const out: number[] = [];
  for (let k = 0; k < max; k++) out.push(arr[(random() * arr.length) | 0]);
  return out;
}

/** Kleiner, deterministischer Zufallsgenerator. */
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

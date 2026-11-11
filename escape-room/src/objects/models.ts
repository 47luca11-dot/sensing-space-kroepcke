import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatFileType, SplatMesh } from '@sparkjsdev/spark';
import type { ColliderConfig, ModelSource } from '../config/types';
import { assetUrl, fetchBytes, isEncoded } from './assets';
import { buildPlaceholder } from './placeholders';

/** Einheitliche Sicht auf ein geladenes Modell, egal ob Platzhalter, Splat oder glTF. */
export interface LoadedModel {
  /** Wird unter das Objekt gehängt, das der Physik folgt. Maßstab ist bereits angewendet. */
  object: THREE.Object3D;
  /** Grenzen im lokalen Objektsystem (skaliert). Grundlage für automatische Kollisionskörper. */
  bounds: THREE.Box3;
  /** Punktwolke für die konvexe Hülle (lokal, skaliert, ausgedünnt). */
  hullPoints?: Float32Array;
  /** Kollisionskörper, den das Modell selbst vorschlägt (nur Platzhalter). */
  defaultCollider?: ColliderConfig;
  setHighlight(on: boolean): void;
}

const MAX_HULL_POINTS = 1500;


export async function loadModel(source: ModelSource, scale: number): Promise<LoadedModel> {
  switch (source.type) {
    case 'placeholder':
      return loadPlaceholder(source, scale);
    case 'splat':
      return loadSplat(source, scale);
    case 'gltf':
      return loadGltf(source, scale);
  }
}

function loadPlaceholder(source: Extract<ModelSource, { type: 'placeholder' }>, scale: number): LoadedModel {
  const { object, collider } = buildPlaceholder(source.shape, source.size, source.color);
  object.scale.setScalar(scale);
  object.updateMatrixWorld(true);
  return {
    object,
    bounds: new THREE.Box3().setFromObject(object),
    defaultCollider: collider,
    setHighlight: emissiveHighlighter(object),
  };
}

/** .ply und .spz erkennt Spark auch am Inhalt, .splat/.ksplat nur über Endung oder fileType. */
function splatFileType(name: string): SplatFileType | undefined {
  const ext = name.split(/[?#]/)[0].split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'ply':
      return SplatFileType.PLY;
    case 'spz':
      return SplatFileType.SPZ;
    case 'splat':
      return SplatFileType.SPLAT;
    case 'ksplat':
      return SplatFileType.KSPLAT;
    default:
      return undefined;
  }
}

async function loadSplat(source: Extract<ModelSource, { type: 'splat' }>, scale: number): Promise<LoadedModel> {
  const fileType = splatFileType(source.url);
  const url = assetUrl(source.url);
  const mesh = source.file
    ? new SplatMesh({ fileBytes: await source.file.arrayBuffer(), fileName: source.file.name, fileType })
    : isEncoded(url)
      ? new SplatMesh({ fileBytes: await fetchBytes(url), fileName: source.url.split('/').pop(), fileType })
      : new SplatMesh({ url, fileType });
  await mesh.initialized;
  // Scaniverse und viele 3DGS-Trainer speichern mit Y nach unten.
  if (source.flip ?? true) mesh.quaternion.set(1, 0, 0, 0);
  mesh.scale.setScalar(scale);
  mesh.updateMatrix();

  const minOpacity = source.minOpacity ?? 0.2;
  const raw: number[] = [];
  const p = new THREE.Vector3();
  mesh.forEachSplat((_i, center, _scales, _quat, opacity) => {
    if (opacity < minOpacity) return;
    p.copy(center).applyMatrix4(mesh.matrix);
    raw.push(p.x, p.y, p.z);
  });

  const { bounds, hullPoints } = robustBounds(Float32Array.from(raw));
  // Modell auf die Mitte seiner Grenzen zentrieren, damit Drehpunkt und Kollisionskörper passen.
  const center = bounds.getCenter(new THREE.Vector3());
  mesh.position.sub(center);
  bounds.translate(center.clone().negate());
  for (let i = 0; i < hullPoints.length; i += 3) {
    hullPoints[i] -= center.x;
    hullPoints[i + 1] -= center.y;
    hullPoints[i + 2] -= center.z;
  }

  return {
    object: mesh,
    bounds,
    hullPoints,
    setHighlight: (on) => {
      // recolor multipliziert die Splat-Farbe; > 1 hellt auf.
      mesh.recolor.setScalar(on ? 1.6 : 1);
    },
  };
}

async function loadGltf(source: Extract<ModelSource, { type: 'gltf' }>, scale: number): Promise<LoadedModel> {
  const loader = new GLTFLoader();
  const gltf = source.file
    ? await loader.parseAsync(await source.file.arrayBuffer(), '')
    : await loader.parseAsync(await fetchBytes(assetUrl(source.url)), '');
  const object = gltf.scene;
  object.scale.setScalar(scale);
  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) child.castShadow = child.receiveShadow = true;
  });
  object.updateMatrixWorld(true);

  const raw: number[] = [];
  const p = new THREE.Vector3();
  object.traverse((child) => {
    const m = child as THREE.Mesh;
    if (!m.isMesh) return;
    const pos = m.geometry.getAttribute('position');
    const step = Math.max(1, Math.floor(pos.count / 20_000));
    for (let i = 0; i < pos.count; i += step) {
      p.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
      raw.push(p.x, p.y, p.z);
    }
  });

  const bounds = new THREE.Box3().setFromObject(object);
  const center = bounds.getCenter(new THREE.Vector3());
  object.position.sub(center);
  bounds.translate(center.clone().negate());
  const { hullPoints } = robustBounds(Float32Array.from(raw), 0);
  for (let i = 0; i < hullPoints.length; i += 3) {
    hullPoints[i] -= center.x;
    hullPoints[i + 1] -= center.y;
    hullPoints[i + 2] -= center.z;
  }

  return { object, bounds, hullPoints, setHighlight: emissiveHighlighter(object) };
}

/**
 * Grenzen über Perzentile je Achse, damit vereinzelte Ausreißer-Splats (Scan-Rauschen) den
 * Kollisionskörper nicht aufblähen. Liefert zusätzlich eine ausgedünnte Punktwolke innerhalb der Grenzen.
 */
function robustBounds(points: Float32Array, trim = 0.005) {
  const n = points.length / 3;
  if (n === 0) return { bounds: new THREE.Box3(new THREE.Vector3(-0.05, -0.05, -0.05), new THREE.Vector3(0.05, 0.05, 0.05)), hullPoints: new Float32Array() };

  const axis = (k: number) => {
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) values[i] = points[i * 3 + k];
    values.sort();
    const at = (q: number) => values[Math.min(n - 1, Math.max(0, Math.round(q * (n - 1))))];
    return [at(trim), at(1 - trim)];
  };
  const [x0, x1] = axis(0);
  const [y0, y1] = axis(1);
  const [z0, z1] = axis(2);
  const bounds = new THREE.Box3(new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1));

  const inside: number[] = [];
  const step = Math.max(1, Math.floor(n / MAX_HULL_POINTS));
  const p = new THREE.Vector3();
  for (let i = 0; i < n; i += step) {
    p.fromArray(points, i * 3);
    if (bounds.containsPoint(p)) inside.push(p.x, p.y, p.z);
  }
  return { bounds, hullPoints: Float32Array.from(inside) };
}

/** Hebt Standard-Materialien über ihre Eigenleuchtfarbe hervor. Materialien werden pro Objekt geklont. */
function emissiveHighlighter(root: THREE.Object3D) {
  const materials: THREE.MeshStandardMaterial[] = [];
  root.traverse((child) => {
    const m = child as THREE.Mesh;
    if (!m.isMesh) return;
    const list = Array.isArray(m.material) ? m.material : [m.material];
    const cloned = list.map((mat) => mat.clone());
    m.material = Array.isArray(m.material) ? cloned : cloned[0];
    for (const mat of cloned) {
      if ((mat as THREE.MeshStandardMaterial).emissive) materials.push(mat as THREE.MeshStandardMaterial);
    }
  });
  const original = materials.map((m) => m.emissive.clone());
  const glow = new THREE.Color(0x3a3a3a);
  return (on: boolean) => {
    materials.forEach((m, i) => m.emissive.copy(on ? glow : original[i]));
  };
}

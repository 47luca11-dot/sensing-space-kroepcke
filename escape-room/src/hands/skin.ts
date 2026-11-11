import * as THREE from 'three';
import { HANDS } from '../config/hands';

/** Kantenlänge einer Porenkachel auf der Haut in Metern. */
const PORE_TILE_METERS = 0.025;
const TEXTURE_SIZE = 512;

/**
 * Macht aus dem einfarbigen Handmodell eine glaubwürdigere Haut:
 * Poren als Normal Map, unregelmäßige Rauheit, Sheen für den weichen Glanz und
 * stärker durchblutete Fingerkuppen und Knöchel über Vertexfarben.
 */
export function applySkin(mesh: THREE.SkinnedMesh) {
  const { normalMap, roughnessMap, colorMap } = skinTextures();
  const repeat = 1 / (PORE_TILE_METERS * uvPerMeter(mesh.geometry));
  for (const t of [normalMap, roughnessMap, colorMap]) t.repeat.set(repeat, repeat);

  addBloodFlowColors(mesh);

  mesh.material = new THREE.MeshPhysicalMaterial({
    color: HANDS.skinColor,
    map: colorMap,
    vertexColors: true,
    roughness: 0.62,
    roughnessMap,
    normalMap,
    normalScale: new THREE.Vector2(0.35, 0.35),
    sheen: 0.35,
    sheenRoughness: 0.55,
    sheenColor: new THREE.Color('#ffb59e'),
    clearcoat: 0.06,
    clearcoatRoughness: 0.45,
    side: THREE.DoubleSide,
  });
}

/** Wie viele UV-Einheiten auf einen Meter Oberfläche kommen (Median über die Dreiecke). */
function uvPerMeter(geometry: THREE.BufferGeometry) {
  const pos = geometry.getAttribute('position');
  const uv = geometry.getAttribute('uv') as THREE.BufferAttribute | undefined;
  const index = geometry.getIndex();
  if (!uv) return 1;
  const ratios: number[] = [];
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const ua = new THREE.Vector2();
  const ub = new THREE.Vector2();
  const count = index ? index.count : pos.count;
  for (let i = 0; i < count; i += 3 * 7) {
    const i0 = index ? index.getX(i) : i;
    const i1 = index ? index.getX(i + 1) : i + 1;
    const world = a.fromBufferAttribute(pos, i0).distanceTo(b.fromBufferAttribute(pos, i1));
    const tex = ua.fromBufferAttribute(uv, i0).distanceTo(ub.fromBufferAttribute(uv, i1));
    if (world > 1e-6 && tex > 1e-6) ratios.push(tex / world);
  }
  ratios.sort((x, y) => x - y);
  return ratios[Math.floor(ratios.length / 2)] ?? 1;
}

/** Rötung an Fingerkuppen und etwas an den Knöcheln, berechnet in der Ruhepose des Rigs. */
function addBloodFlowColors(mesh: THREE.SkinnedMesh) {
  const geometry = mesh.geometry;
  const pos = geometry.getAttribute('position');
  const bindPos = (name: string) => {
    const i = mesh.skeleton.bones.findIndex((b) => b.name === name);
    // Knochenposition im selben Raum wie die Vertexdaten (Bind-Space des Meshes)
    const bind = mesh.bindMatrixInverse.clone().multiply(mesh.skeleton.boneInverses[i].clone().invert());
    return new THREE.Vector3().setFromMatrixPosition(bind);
  };

  const tips: THREE.Vector3[] = [];
  const knuckles: THREE.Vector3[] = [];
  for (const bones of Object.values(HANDS.fingers)) {
    const [b1, b2, b3] = bones.map(bindPos);
    tips.push(b3.clone().add(b3.clone().sub(b2).multiplyScalar(0.7)));
    knuckles.push(b1, b2);
  }

  const base = new THREE.Color(1, 1, 1);
  const flush = new THREE.Color(1.0, 0.78, 0.74);
  const knuckle = new THREE.Color(0.93, 0.8, 0.76);
  const c = new THREE.Color();
  const p = new THREE.Vector3();
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    const tipDist = Math.min(...tips.map((t) => t.distanceTo(p)));
    const knuckleDist = Math.min(...knuckles.map((k) => k.distanceTo(p)));
    c.copy(base);
    c.lerp(knuckle, THREE.MathUtils.smoothstep(0.016, 0.006, knuckleDist) * 0.6);
    c.lerp(flush, THREE.MathUtils.smoothstep(0.024, 0.008, tipDist));
    c.toArray(colors, i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

/** Kachelbare Porenstruktur: Höhenfeld → Normal Map, dazu Rauheit und leichte Farbvariation. */
function skinTextures() {
  const n = TEXTURE_SIZE;
  const height = new Float32Array(n * n);
  const rand = mulberry32(7);

  // Viele kleine Vertiefungen (Poren) und feine Hautfalten
  for (let k = 0; k < 2600; k++) {
    const cx = rand() * n;
    const cy = rand() * n;
    const r = 1.2 + rand() * 1.8;
    const depth = 0.4 + rand() * 0.6;
    for (let dy = -4; dy <= 4; dy++) {
      for (let dx = -4; dx <= 4; dx++) {
        const d2 = (dx * dx + dy * dy) / (r * r);
        if (d2 > 1) continue;
        const x = (Math.floor(cx) + dx + n) % n;
        const y = (Math.floor(cy) + dy + n) % n;
        height[y * n + x] -= depth * (1 - d2);
      }
    }
  }
  for (let k = 0; k < 220; k++) {
    let x = rand() * n;
    let y = rand() * n;
    const angle = rand() * Math.PI;
    for (let s = 0; s < 40; s++) {
      x += Math.cos(angle);
      y += Math.sin(angle) * 0.6;
      const i = ((Math.floor(y) + n) % n) * n + ((Math.floor(x) + n) % n);
      height[i] -= 0.35;
    }
  }

  const normalData = new Uint8Array(n * n * 4);
  const roughData = new Uint8Array(n * n * 4);
  const colorData = new Uint8Array(n * n * 4);
  const h = (x: number, y: number) => height[((y + n) % n) * n + ((x + n) % n)];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = (y * n + x) * 4;
      const sx = h(x + 1, y) - h(x - 1, y);
      const sy = h(x, y + 1) - h(x, y - 1);
      const v = new THREE.Vector3(-sx, -sy, 2).normalize();
      normalData[i] = (v.x * 0.5 + 0.5) * 255;
      normalData[i + 1] = (v.y * 0.5 + 0.5) * 255;
      normalData[i + 2] = (v.z * 0.5 + 0.5) * 255;
      normalData[i + 3] = 255;
      // Poren sind etwas rauer, glatte Stellen glänzen leicht
      const rough = THREE.MathUtils.clamp(0.75 - h(x, y) * 0.25 + (rand() - 0.5) * 0.1, 0.4, 1);
      roughData[i] = roughData[i + 1] = roughData[i + 2] = rough * 255;
      roughData[i + 3] = 255;
      const tone = 1 + h(x, y) * 0.04 + (rand() - 0.5) * 0.03;
      colorData[i] = THREE.MathUtils.clamp(tone, 0, 1) * 255;
      colorData[i + 1] = THREE.MathUtils.clamp(tone * 0.985, 0, 1) * 255;
      colorData[i + 2] = THREE.MathUtils.clamp(tone * 0.98, 0, 1) * 255;
      colorData[i + 3] = 255;
    }
  }

  const make = (data: Uint8Array, srgb = false) => {
    const t = new THREE.DataTexture(data, n, n, THREE.RGBAFormat);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.generateMipmaps = true;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    t.needsUpdate = true;
    return t;
  };
  return { normalMap: make(normalData), roughnessMap: make(roughData), colorMap: make(colorData, true) };
}

/** Kleiner deterministischer Zufallsgenerator, damit die Haut bei jedem Start gleich aussieht. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

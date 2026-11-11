// Platzhalter-Testgelände (wird erzeugt, solange assets/city/hannover.glb fehlt).
// Gleiches Koordinatensystem wie die Stadt: Meter, Y oben, Ursprung = Kröpcke-Uhr, +X Osten, -Z Norden.
// Enthält: sanfte Hügel, Platz mit Uhr, Straße mit Bordsteinen, Rampe/Plattform/Treppe/Absturzkante,
// Steilhang (Rutschen), Geröllfeld mit Steinen (Gleichgewicht), Kolonnade (Kamera-Kollision), niedrige Mauern.
import * as THREE from 'three';

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const rectMask = (x, z, x0, z0, x1, z1, f) =>      // 1 innerhalb, weicher Rand der Breite f
  smooth(x0 - f, x0, x) * (1 - smooth(x1, x1 + f, x)) * smooth(z0 - f, z0, z) * (1 - smooth(z1, z1 + f, z));

export const ROUGH = { x0: 58, z0: -22, x1: 94, z1: 14 };         // Geröllfeld
const FLAT = [
  { x0: -96, z0: -74, x1: -34, z1: -30, f: 8 },                  // Testparcours
  { x0: -140, z0: 20, x1: 140, z1: 42, f: 6 },                   // Straße + Gehwege
  { x0: -44, z0: -16, x1: -24, z1: 16, f: 6 },                   // Kolonnade
];

function hills(x, z) {
  return 1.7 * Math.sin(x * 0.021 + 0.3) * Math.cos(z * 0.017) + 0.9 * Math.sin(x * 0.047 + z * 0.031)
    + 0.45 * Math.cos(z * 0.06 - x * 0.013) + 0.2 * Math.sin(x * 0.11) * Math.sin(z * 0.13);
}
export function baseHeight(x, z) {
  let k = smooth(24, 46, Math.hypot(x, z));                     // Platz am Kröpcke eben
  for (const r of FLAT) k *= 1 - rectMask(x, z, r.x0, r.z0, r.x1, r.z1, r.f);
  return hills(x, z) * k;
}
function bumps(x, z) {
  return 0.11 * Math.sin(x * 3.1 + 0.7) * Math.cos(z * 2.7) + 0.07 * Math.sin(x * 5.3 + z * 4.1)
    + 0.06 * Math.cos(x * 1.7 - z * 2.2) + 0.04 * Math.sin(x * 8.9 - z * 7.3);
}

function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function facadeTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 256; const g = c.getContext('2d');
  g.fillStyle = '#d8d4cc'; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 1400; i++) { g.fillStyle = `rgba(${90 + Math.random() * 60},${90 + Math.random() * 50},${80 + Math.random() * 40},${Math.random() * 0.08})`; g.fillRect(Math.random() * 256, Math.random() * 256, 3 + Math.random() * 10, 2 + Math.random() * 6); }
  g.fillStyle = 'rgba(0,0,0,.18)'; g.fillRect(0, 0, 256, 10);                       // Gesims
  for (let wx = 0; wx < 2; wx++) {
    const x = 34 + wx * 128, y = 64;
    g.fillStyle = '#e9e6e0'; g.fillRect(x - 6, y - 6, 72, 116);                       // Fensterlaibung
    g.fillStyle = '#3c4650'; g.fillRect(x, y, 60, 104);
    g.fillStyle = 'rgba(190,210,225,.35)'; g.fillRect(x + 3, y + 3, 25, 46); g.fillRect(x + 32, y + 3, 25, 46);
    g.fillStyle = '#e9e6e0'; g.fillRect(x + 28, y, 4, 104); g.fillRect(x, y + 50, 60, 4);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4;
  return t;
}

// Box mit Metern als UV-Einheit (eine Textur-Kachel = tile m), damit Fenster überall gleich groß sind
function worldBox(w, h, d, tile = 3) {
  const g = new THREE.BoxGeometry(w, h, d);
  const uv = g.attributes.uv;
  const dims = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];          // +x -x +y -y +z -z
  for (let f = 0; f < 6; f++) for (let v = 0; v < 4; v++) {
    const i = f * 4 + v; uv.setXY(i, uv.getX(i) * dims[f][0] / tile, uv.getY(i) * dims[f][1] / tile);
  }
  return g;
}

export function buildPlaceholder() {
  const root = new THREE.Group(); root.name = 'Platzhalter';
  const rand = rng(1234);
  const add = (geo, mat, name, x = 0, y = 0, z = 0, ry = 0) => {
    const m = new THREE.Mesh(geo, mat); m.name = name; m.position.set(x, y, z); m.rotation.y = ry; root.add(m); return m;
  };

  // ---------- Gelände ----------
  const SIZE = 280, SEG = 280;
  const tg = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG); tg.rotateX(-Math.PI / 2);
  const tp = tg.attributes.position, col = new Float32Array(tp.count * 3);
  const cGrass = new THREE.Color(0x6e7558), cPave = new THREE.Color(0x8d8a83), cRoad = new THREE.Color(0x4b4c4f), cDirt = new THREE.Color(0x77695a), c = new THREE.Color();
  for (let i = 0; i < tp.count; i++) {
    const x = tp.getX(i), z = tp.getZ(i);
    let y = baseHeight(x, z);
    const inRough = x > ROUGH.x0 && x < ROUGH.x1 && z > ROUGH.z0 && z < ROUGH.z1 ? 1 : 0;
    y -= inRough * 0.35;                                              // liegt unter dem fein aufgelösten Geröllfeld
    tp.setY(i, y);
    const r = Math.hypot(x, z);
    c.copy(cGrass).lerp(cPave, 1 - smooth(20, 26, r));
    if (z > 26 && z < 36) c.copy(cRoad);
    c.lerp(cDirt, rectMask(x, z, ROUGH.x0, ROUGH.z0, ROUGH.x1, ROUGH.z1, 4) * 0.8);
    const n = 0.88 + rand() * 0.18;
    col[i * 3] = c.r * n; col[i * 3 + 1] = c.g * n; col[i * 3 + 2] = c.b * n;
  }
  tg.setAttribute('color', new THREE.BufferAttribute(col, 3));
  tg.computeVertexNormals();
  const groundMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.96, metalness: 0 });
  add(tg, groundMat, 'Gelaende');

  // ---------- Geröllfeld (0.25 m Raster, kleine Unebenheiten) ----------
  const rw = ROUGH.x1 - ROUGH.x0, rd = ROUGH.z1 - ROUGH.z0;
  const rg = new THREE.PlaneGeometry(rw, rd, rw * 4, rd * 4); rg.rotateX(-Math.PI / 2); rg.translate((ROUGH.x0 + ROUGH.x1) / 2, 0, (ROUGH.z0 + ROUGH.z1) / 2);
  const rp = rg.attributes.position, rcol = new Float32Array(rp.count * 3);
  for (let i = 0; i < rp.count; i++) {
    const x = rp.getX(i), z = rp.getZ(i);
    const edge = rectMask(x, z, ROUGH.x0 + 3, ROUGH.z0 + 3, ROUGH.x1 - 3, ROUGH.z1 - 3, 3);
    rp.setY(i, baseHeight(x, z) + bumps(x, z) * edge);
    const n = 0.8 + rand() * 0.35;
    c.copy(cDirt).lerp(cGrass, 0.25 * rand());
    rcol[i * 3] = c.r * n; rcol[i * 3 + 1] = c.g * n; rcol[i * 3 + 2] = c.b * n;
  }
  rg.setAttribute('color', new THREE.BufferAttribute(rcol, 3));
  rg.computeVertexNormals();
  add(rg, groundMat, 'Geroellfeld');

  // ---------- Materialien ----------
  const tex = facadeTexture();
  const roof = new THREE.MeshStandardMaterial({ color: 0x5d5f63, roughness: 0.9 });
  const facades = [0xe8dcc8, 0xd6c3a5, 0xc9b9a8, 0xb9a89a, 0xdedad2].map((hex) => new THREE.MeshStandardMaterial({ map: tex, color: hex, roughness: 0.88 }));
  const stone = new THREE.MeshStandardMaterial({ color: 0xa9a49a, roughness: 0.92 });
  const concrete = new THREE.MeshStandardMaterial({ color: 0x9c9a95, roughness: 0.95 });
  const curbMat = new THREE.MeshStandardMaterial({ color: 0xb4b1aa, roughness: 0.9 });
  const metal = new THREE.MeshStandardMaterial({ color: 0x2f3438, roughness: 0.45, metalness: 0.6 });
  const clockFace = new THREE.MeshStandardMaterial({ color: 0xf2efe6, roughness: 0.6, emissive: 0x222018 });
  const rockMat = new THREE.MeshStandardMaterial({ color: 0x8a847a, roughness: 0.95, flatShading: true });

  const minUnder = (x0, z0, x1, z1) => { let m = Infinity; for (let i = 0; i <= 4; i++) for (let j = 0; j <= 4; j++) m = Math.min(m, baseHeight(x0 + (x1 - x0) * i / 4, z0 + (z1 - z0) * j / 4)); return m; };

  // ---------- Gebäude ----------
  const B = [
    [-38, -66, -14, -44, 18], [18, -68, 46, -46, 23], [-72, -24, -50, 6, 15], [16, 44, 42, 66, 16], [-48, 44, -14, 62, 21],
    [50, -70, 74, -50, 13], [-112, 50, -80, 74, 26], [100, 46, 128, 70, 19], [-30, -110, 10, -86, 28], [104, -64, 126, -36, 17],
    [46, 70, 70, 96, 12], [-124, -20, -104, 14, 14],
  ];
  B.forEach(([x0, z0, x1, z1, h], i) => {
    const base = minUnder(x0, z0, x1, z1) - 0.6, w = x1 - x0, d = z1 - z0;
    const m = add(worldBox(w, h + 0.6, d), [facades[i % 5], facades[i % 5], roof, roof, facades[i % 5], facades[i % 5]], 'Gebaeude_' + (i + 1),
      (x0 + x1) / 2, base + (h + 0.6) / 2, (z0 + z1) / 2);
    m.userData.kind = 'building';
  });

  // ---------- Kröpcke-Uhr ----------
  add(new THREE.CylinderGeometry(0.13, 0.17, 3.1, 12), metal, 'Uhr_Saeule', 0, 1.55, 0);
  add(new THREE.CylinderGeometry(0.5, 0.5, 0.95, 20), clockFace, 'Uhr', 0, 3.55, 0);
  add(new THREE.ConeGeometry(0.58, 0.45, 20), metal, 'Uhr_Dach', 0, 4.25, 0);
  add(new THREE.CylinderGeometry(1.4, 1.6, 0.22, 24), stone, 'Uhr_Sockel', 0, 0.11, 0);

  // Poller, Bänke und niedrige Mauern am Platz (Sprung / Kollision)
  for (let i = 0; i < 14; i++) {
    const a = (i / 14) * Math.PI * 2;
    if (i % 7 === 3) continue;
    add(new THREE.CylinderGeometry(0.11, 0.13, 0.85, 10), metal, 'Poller', Math.cos(a) * 19, 0.42, Math.sin(a) * 19);
  }
  for (const [x, z, ry] of [[7, -9, 0.3], [-8, -7, -0.4], [-6, 10, 0.2], [11, 6, 1.4]]) add(new THREE.BoxGeometry(2.0, 0.45, 0.6), stone, 'Bank', x, 0.225, z, ry);
  add(new THREE.BoxGeometry(6, 0.45, 0.5), stone, 'Mauer_niedrig', 10, 0.225, -18);
  add(new THREE.BoxGeometry(5, 0.75, 0.5), stone, 'Mauer_mittel', -10, 0.375, -18);

  // ---------- Straße mit Bordsteinen (15 cm) ----------
  for (const [z0, z1] of [[21.5, 26], [36, 40.5]]) {
    add(new THREE.BoxGeometry(260, 0.15, z1 - z0), concrete, 'Gehweg', 0, 0.075, (z0 + z1) / 2);
    add(new THREE.BoxGeometry(260, 0.16, 0.3), curbMat, 'Bordstein', 0, 0.08, z0 === 21.5 ? 26 - 0.15 : 36 + 0.15);
  }

  // ---------- Kolonnade (Dach auf Säulen; Kamera-Kollision) ----------
  for (let i = 0; i < 6; i++) add(new THREE.CylinderGeometry(0.28, 0.32, 4.2, 14), stone, 'Saeule', -30, 2.1, -10 + i * 4);
  for (let i = 0; i < 6; i++) add(new THREE.CylinderGeometry(0.28, 0.32, 4.2, 14), stone, 'Saeule', -36, 2.1, -10 + i * 4);
  add(new THREE.BoxGeometry(7.4, 0.5, 22.6), stone, 'Kolonnade_Dach', -33, 4.45, 0);
  add(new THREE.BoxGeometry(0.6, 6, 26), facades[2], 'Kolonnade_Wand', -39, 3.0, 0);

  // ---------- Testparcours: Plattform (3 m), Rampe 15°, Treppe, Absturzkante ----------
  const PX0 = -62, PX1 = -48, PZ0 = -64, PZ1 = -50, PH = 3.0;
  add(new THREE.BoxGeometry(PX1 - PX0, PH + 0.5, PZ1 - PZ0), concrete, 'Plattform', (PX0 + PX1) / 2, (PH - 0.5) / 2, (PZ0 + PZ1) / 2);
  const rampLen = PH / Math.tan(THREE.MathUtils.degToRad(15)), rampW = 4;
  const ramp = add(new THREE.BoxGeometry(rampW, 0.4, Math.hypot(rampLen, PH)), concrete, 'Rampe_15', -58, PH / 2 - 0.2 * Math.cos(0.26), PZ1 + rampLen / 2);
  ramp.rotation.x = Math.atan2(PH, rampLen);   // steigt nach Norden (-Z) zur Plattform an
  const STEP_H = PH / 18, STEP_D = 0.3;
  for (let s = 0; s < 18; s++) {
    const top = (s + 1) * STEP_H;
    add(new THREE.BoxGeometry(STEP_D, top, 3.2), concrete, 'Treppe', PX1 + 0.15 + (17 - s) * STEP_D, top / 2, -58);
  }
  // Brüstung an der Nordkante fehlt absichtlich -> 3 m Absturz testen; an der Westkante eine Mauer
  add(new THREE.BoxGeometry(0.3, 1.1, PZ1 - PZ0), stone, 'Bruestung', PX0 + 0.15, PH + 0.55, (PZ0 + PZ1) / 2);

  // ---------- Steilhang 40° (Rutschen) ----------
  const slopeH = 5, slopeLen = slopeH / Math.tan(THREE.MathUtils.degToRad(40));
  const steep = add(new THREE.BoxGeometry(8, 0.5, Math.hypot(slopeLen, slopeH)), stone, 'Steilhang_40', -84, slopeH / 2 - 0.25, -46 - slopeLen / 2);
  steep.rotation.x = -Math.atan2(slopeH, slopeLen);                   // steigt nach Süden (+Z) an
  add(new THREE.BoxGeometry(8, slopeH, 6), stone, 'Steilhang_Kopf', -84, slopeH / 2, -46 + 3 - 0.1);

  // Mittelsteile Rampe 25° (gelb im Scan)
  const midLen = 2.5 / Math.tan(THREE.MathUtils.degToRad(25));
  const mid = add(new THREE.BoxGeometry(5, 0.4, Math.hypot(midLen, 2.5)), concrete, 'Rampe_25', -72, 2.5 / 2 - 0.2, -36 - midLen / 2);
  mid.rotation.x = -Math.atan2(2.5, midLen);
  add(new THREE.BoxGeometry(5, 2.5, 4), concrete, 'Rampe_25_Kopf', -72, 1.25, -36 + 2 - 0.05);

  // ---------- Steine im Geröllfeld ----------
  const hash = (x, y, z, sd) => { const h = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719 + sd) * 43758.5453; return h - Math.floor(h); };
  const rockGeo = (s) => {                 // Ikosaeder ist nicht indiziert -> Verformung als Funktion der Position (keine Risse)
    const g = new THREE.IcosahedronGeometry(1, 1), p = g.attributes.position, sd = rand() * 100;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i), k = 0.75 + 0.5 * hash(Math.round(x * 1e3), Math.round(y * 1e3), Math.round(z * 1e3), sd);
      p.setXYZ(i, x * k, y * k * 0.7, z * k);
    }
    g.computeVertexNormals(); g.scale(s, s, s); return g;
  };
  for (let i = 0; i < 46; i++) {
    const x = ROUGH.x0 + 3 + rand() * (rw - 6), z = ROUGH.z0 + 3 + rand() * (rd - 6), s = 0.15 + rand() * rand() * 1.1;
    const r = add(rockGeo(s), rockMat, 'Stein', x, baseHeight(x, z) + bumps(x, z) - s * 0.35, z, rand() * 6);
    r.rotation.x = (rand() - 0.5) * 0.4;
  }
  for (let i = 0; i < 30; i++) {
    const a = rand() * Math.PI * 2, d = 50 + rand() * 70, x = Math.cos(a) * d, z = Math.sin(a) * d, s = 0.2 + rand() * 0.6;
    if (z > 18 && z < 44) continue;
    add(rockGeo(s), rockMat, 'Stein', x, baseHeight(x, z) - s * 0.3, z, rand() * 6);
  }

  root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  root.userData.meta = {
    spawn: [0, 0, 7], spawnYaw: 0, bounds: [-138, -138, 138, 138],
    labels: [
      { name: 'Kröpcke', pos: [0, 0], r: 24 },
      { name: 'Testparcours · Rampe & Treppe', pos: [-58, -54], r: 16 },
      { name: 'Steilhang', pos: [-84, -46], r: 10 },
      { name: 'Geröllfeld', pos: [76, -4], r: 20 },
      { name: 'Kolonnade', pos: [-33, 0], r: 7 },
      { name: 'Straße', pos: [0, 31], r: 7 },
    ],
  };
  return root;
}

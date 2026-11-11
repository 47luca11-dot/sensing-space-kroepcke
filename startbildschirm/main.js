import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SparkRenderer, SplatMesh, SplatSkinning } from '@sparkjsdev/spark';

const FW = 1588, FH = 991;                         // Design-Rahmen (aus der PSD)
const $ = (id) => document.getElementById(id);
const stage = $('stage'), world = $('world'), bgEl = $('bg'), treesEl = $('trees'), flashEl = $('flash');
const glCanvas = $('gl'), backCanvas = $('fxBack'), frontCanvas = $('fxFront');
const back = backCanvas.getContext('2d'), front = frontCanvas.getContext('2d');
const rand = (a, b) => a + Math.random() * (b - a);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));

// Hintergrundbild (1376 x 768) -> Rahmen: background-size cover + scale(1.04)
const IMG_W = 1376, IMG_H = 768, BG_ZOOM = 1.04;
const COVER = Math.max(FW / IMG_W, FH / IMG_H);
function imgToFrame(ix, iy) {
  const x = ix * COVER - (IMG_W * COVER - FW) / 2, y = iy * COVER - (IMG_H * COVER - FH) / 2;
  return [(x - FW / 2) * BG_ZOOM + FW / 2, (y - FH / 2) * BG_ZOOM + FH / 2];
}

// ---------------- Charaktere (Texte/Stats hier anpassen) ----------------
const CHARACTERS = [
  { id: 'person1', name: 'CARLOS', role: 'PLÜNDERER', x: -0.70, timeOffset: 0.0,
    stats: { AUSDAUER: 6, GESUNDHEIT: 7, GESCHWINDIGKEIT: 4, WILLE: 9 },
    bio: 'Kennt jeden Keller zwischen Bothfeld und Vahrenheide. Bleibt ruhig, wenn alles brennt, und hört selten auf Befehle.' },
  { id: 'andis', name: 'ANDIS', role: 'ÜBERLEBENDER', x: 0.10, timeOffset: 1.7,
    stats: { AUSDAUER: 8, GESUNDHEIT: 5, GESCHWINDIGKEIT: 7, WILLE: 6 },
    bio: 'Schreckhaft, aber zäh. Dreht sich bei jedem Knall um – und läuft trotzdem als Erster los.' },
  { id: 'luca', name: 'LUCA', role: 'STRATEGE', x: 0.90, timeOffset: 0.0, lookScale: 0.3,
    stats: { AUSDAUER: 5, GESUNDHEIT: 4, GESCHWINDIGKEIT: 6, WILLE: 8 },
    bio: 'Hat immer Tabak und Papers dabei. Dreht sich in jeder Feuerpause in aller Ruhe eine – und plant dabei schon den nächsten Zug.' },
  { id: 'mohammad', name: 'MOHAMMAD', role: 'KUNDSCHAFTER', x: 1.70, timeOffset: 0.9,
    stats: { AUSDAUER: 7, GESUNDHEIT: 6, GESCHWINDIGKEIT: 6, WILLE: 7 },
    bio: 'Die Hände bleiben in den Taschen, egal was passiert. Läuft die Wege ab, bevor die anderen losgehen – und weiß immer, wo es weitergeht.' },
  { id: 'daunenjacke', name: 'DAUNENJACKE', role: 'NACHTWACHE', x: 2.50, timeOffset: 0.4, mesh: true,     // Name vorläufig
    stats: { AUSDAUER: 8, GESUNDHEIT: 8, GESCHWINDIGKEIT: 5, WILLE: 6 },
    bio: 'Kapuze auf, Jacke zu bis oben. Friert nie, redet wenig – und ist immer da, wenn es dunkel wird.' },
];
const SEGMENTS = 10;

// ---------------- Layout + adaptive Qualität ----------------
let fxScale = 1, quality = 1;
function layout() {
  const s = Math.max(innerWidth / FW, innerHeight / FH);
  stage.style.transform = `translate(-50%, -50%) scale(${s})`;
  stage.style.setProperty('--oy', Math.max(0, (FH - innerHeight / s) / 2) + 'px');   // vertikal abgeschnittener Rand (16:9)
  const pr = Math.min(1.75, devicePixelRatio * s) * quality;
  renderer.setPixelRatio(pr);
  renderer.setSize(FW, FH, false);
  fxScale = Math.min(1, pr);
  for (const c of [backCanvas, frontCanvas]) { c.width = Math.round(FW * fxScale); c.height = Math.round(FH * fxScale); }
}
const perf = { acc: 0, n: 0 };
function adaptQuality(dt) {
  if (dt > 0.25) return;                       // Hänger (Tab-Wechsel) nicht mitzählen
  perf.acc += dt; perf.n++;
  if (perf.acc < 1.5) return;
  const avg = perf.acc / perf.n; perf.acc = perf.n = 0;
  if (avg > 0.024 && quality > 0.5) { quality = Math.max(0.5, quality - 0.2); layout(); }
  else if (avg < 0.019 && quality < 1) { quality = Math.min(1, quality + 0.1); layout(); }   // erholt sich auch bei 60 Hz
}

// ---------------- three.js + Spark ----------------
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, alpha: true, antialias: false });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.autoClear = false;
const scene = new THREE.Scene();
const spark = new SparkRenderer({ renderer });
spark.layers.set(1);                               // eigener Render-Durchgang (siehe tick)
scene.add(spark);

// Kamera: Füße bei ~y 935 px, Köpfe bei ~y 235 px; schwenkt seitlich zur gewählten Figur
// Perspektive passend zum Hintergrundfoto: dessen Horizont (Fluchtpunkt der Seitenflügel) liegt bei
// Bild-y 520 -> Rahmen-y ~678, das Foto ist also ~4.5° nach oben geneigt. Die 3D-Kamera bekommt exakt diese
// Neigung; Höhe/Abstand so, dass Köpfe bei ~225 px stehen und das Bild unter den Knien endet (Füße unsichtbar).
const FOV = 24, CAM_D = 4.4, CAM_Y = 1.08, HORIZON_Y = 678;
const PITCH_TAN = (HORIZON_Y - FH / 2) / ((FH / 2) / Math.tan(THREE.MathUtils.degToRad(FOV / 2)));
const LOOK_Y = CAM_Y + CAM_D * PITCH_TAN;
const camera = new THREE.PerspectiveCamera(FOV, FW / FH, 0.1, 60);
const HALF_W = CAM_D * Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * (FW / FH);   // halbe Bildbreite in m (z = 0)
const SEL_NDC_X = -0.24;                       // gewählte Figur steht bei ~x 600 px, links vom Info-Panel
const cam = { x: 0 };
const lookTarget = new THREE.Vector3(0, LOOK_Y, 0);

const hemi = new THREE.HemisphereLight(0xc2ccd6, 0x3a3b3e, 1.25); scene.add(hemi);
const key = new THREE.DirectionalLight(0xdde6f0, 1.3);
key.position.set(-1.2, 9, 2.6); key.castShadow = true;   // steil von oben (bedeckt): kurze, weiche Schatten
key.shadow.mapSize.set(1024, 1024);
Object.assign(key.shadow.camera, { left: -2.5, right: 2, top: 2.5, bottom: -0.5, near: 0.5, far: 15 });
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02;
scene.add(key);
const rim = new THREE.DirectionalLight(0xb0c0d0, 1.2); rim.position.set(1.5, 3, -4); scene.add(rim);
const fireLight = new THREE.PointLight(0xff7a2a, 2.5, 7, 1.6); fireLight.position.set(-2.6, 0.5, -1.0); scene.add(fireLight);
const boomLight = new THREE.PointLight(0xff9a4a, 0, 30, 1.2); boomLight.position.set(0.5, 3.5, -8); scene.add(boomLight);
for (const l of [hemi, key, rim, fireLight, boomLight]) l.layers.enable(1);   // gleicher Licht-Zustand in beiden Durchgängen

const ground = new THREE.Mesh(new THREE.PlaneGeometry(10, 8), new THREE.ShadowMaterial({ opacity: 0.28 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
function radialTex(inner, outer, ring) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d'), gr = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  if (ring) { gr.addColorStop(0, 'rgba(95,211,196,0)'); gr.addColorStop(0.72, 'rgba(95,211,196,0)'); gr.addColorStop(0.8, 'rgba(160,240,230,.9)'); gr.addColorStop(0.86, 'rgba(95,211,196,.25)'); gr.addColorStop(1, 'rgba(95,211,196,0)'); }
  else { gr.addColorStop(0, inner); gr.addColorStop(1, outer); }
  g.fillStyle = gr; g.fillRect(0, 0, 256, 256); return new THREE.CanvasTexture(c);
}
const blobTex = radialTex('rgba(10,12,16,.7)', 'rgba(10,12,16,0)');
function addBlob(x) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.45), new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2; m.position.set(x, 0.002, 0.02); scene.add(m); return m;
}
// Auswahl-Ring am Boden unter der aktiven Figur
const selRing = new THREE.Mesh(new THREE.PlaneGeometry(1.15, 1.15), new THREE.MeshBasicMaterial({ map: radialTex(0, 0, true), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
selRing.rotation.x = -Math.PI / 2; selRing.position.y = 0.004; scene.add(selRing);

// ---------------- Figuren: Mesh (Füllung) + Original-Gaussian-Splats, gemeinsam gerigged ----------------
// Mesh = geschlossene Füllung knapp unter der Scan-Oberfläche (deckt Scan-Lücken), Splats = sichtbares Detail.
// Jeder Splat folgt per Dual-Quaternion-Skinning (Spark) denselben Knochen wie das Mesh.
const figures = {};
const loader = new GLTFLoader();
let loadedCount = 0;
const BASE_TINT = new THREE.Color(1, 1, 1);            // Originalfarben der Scans unverändert

async function loadFigure(c, index) {
  // c.mesh: Figur ist ein texturiertes Mesh ohne Splats (nur <id>.glb)
  const [gltf, skinBuf, splat] = await Promise.all([
    loader.loadAsync(`assets/${c.id}.glb`),
    c.mesh ? null : fetch(`assets/${c.id}_skin.bin`).then((r) => r.arrayBuffer()),
    c.mesh ? null : (async () => { const s = new SplatMesh({ url: `assets/${c.id}_splats.ply` }); await s.initialized; s.maxSh = 1; return s; })(),   // SH Grad 1: Glanz/Blickwinkel; Grad 2-3 rauschen an selten gesehenen Seiten
  ]);
  const root = gltf.scene;
  root.position.x = c.x;
  let skinned = null;
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true; o.receiveShadow = false; o.frustumCulled = false;
    // Füllung wie die Splats: unbeleuchtet, ohne Tonemapping, gleiche Tönung -> keine Farbsprünge in Lücken
    o.material = c.mesh
      ? new THREE.MeshBasicMaterial({ map: o.material.map, toneMapped: false })      // Scan-Textur (Licht ist eingebacken)
      : new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false });
    o.material.color.copy(BASE_TINT);
    if (o.isSkinnedMesh) skinned = o;
  });
  scene.add(root); addBlob(c.x);

  let skin = null;
  if (!c.mesh) {
    // Splats als Geschwister des SkinnedMesh -> gleicher lokaler Frame wie die Mesh-Vertices
    skinned.parent.add(splat);
    splat.position.copy(skinned.position); splat.quaternion.copy(skinned.quaternion); splat.scale.copy(skinned.scale);
    const n = splat.numSplats, bones = skinned.skeleton.bones;
    // Dual-Quaternion-Skinning: Spark's LBS braucht Kovarianz-Splats, die sichtbar unschärfer rendern. Die kleine
    // Abweichung zur LBS-Füllung ist unsichtbar, weil die Füllung ~2 cm unter den Splats liegt und Splats obenauf gezeichnet werden.
    skin = new SplatSkinning({ mesh: splat, numSplats: n, numBones: bones.length });
    const raw = new Uint8Array(skinBuf);
    if (raw.length !== n * 8) console.warn(c.id, 'skin/splat count mismatch', raw.length / 8, n);
    for (let i = 0; i < n; i++) {                     // 4 Knochen-Indizes + 4 Gewichte (Summe 255) je Splat
      const o = i * 8, d = i * 4;
      for (let k = 0; k < 4; k++) skin.skinData[d + k] = raw[o + 4 + k] + (raw[o + k] << 8);
    }
    skin.skinTexture.needsUpdate = true;
    const I = new THREE.Quaternion(), Z0 = new THREE.Vector3();
    for (let b = 0; b < bones.length; b++) skin.setRestQuatPos(b, I, Z0);   // Ruhe = Identität, wir liefern Delta-Matrizen
    splat.skinning = skin;
    splat.updateGenerator();
    splat.recolor = BASE_TINT.clone();
    splat.layers.set(1);                              // Splat-Durchgang (Spark prüft die Layer der SplatMeshes)
  }

  // Clips aus Blender: <id>_Idle_Breathing (Loop), bei Andis zusätzlich <id>_LookBack_L / _R (einmalig)
  const mixer = new THREE.AnimationMixer(root);
  const clip = (suffix) => gltf.animations.find((a) => a.name === `${c.id}_${suffix}`);
  const idle = mixer.clipAction(clip('Idle_Breathing') || gltf.animations[0]);
  idle.play(); mixer.setTime(c.timeOffset);
  const look = {};
  for (const side of ['L', 'R']) {
    const cl = clip('LookBack_' + side); if (!cl) continue;
    const a = mixer.clipAction(cl); a.setLoop(THREE.LoopOnce, 1); a.clampWhenFinished = true; look[side] = a;
  }
  const fidgets = ['Pockets', 'HeadTilt_L', 'HeadTilt_R'].map(clip).filter(Boolean).map((cl) => {
    const a = mixer.clipAction(cl); a.setLoop(THREE.LoopOnce, 1); a.clampWhenFinished = true; return a;
  });
  const byName = {};
  root.traverse((o) => { if (o.isBone) byName[o.name.replace(/\./g, '')] = o; });
  figures[c.id] = { yaw: 0, spin: 0, rest: 0, c, index, root, mixer, bones: byName, idle, look, fidgets, skinned, splat, skin, dim: 1, meshMat: skinned.material };
  if (++loadedCount === CHARACTERS.length) { $('loading').style.opacity = 0; portraitsDue = T + 0.6; }
}
CHARACTERS.forEach((c, i) => loadFigure(c, i).catch((e) => { console.error(e); $('loading').querySelector('span').textContent = 'FEHLER BEIM LADEN: ' + c.name; }));

// Knochen-Deltas für die Splats: D_k = M^-1 · Bone_k · BoneInverse_k · BindMatrix (wie three.js-Skinning)
const _m = new THREE.Matrix4(), _inv = new THREE.Matrix4(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3();
function updateSplatSkin(f) {
  if (!f.skin) return;                                 // Mesh-Figur: three.js skinnt selbst
  const sk = f.skinned.skeleton;
  _inv.copy(f.skinned.matrixWorld).invert();
  for (let b = 0; b < sk.bones.length; b++) {
    _m.multiplyMatrices(sk.bones[b].matrixWorld, sk.boneInverses[b]).premultiply(_inv).multiply(f.skinned.bindMatrix);
    _m.decompose(_p, _q, _s);
    f.skin.setBoneQuatPos(b, _q, _p);
  }
  f.skin.updateBones();
}

// ---------------- Auswahl / UI ----------------
let selected = 1;                                   // Start: Andis
const panel = $('panel');
function renderPanel() {
  const c = CHARACTERS[selected];
  $('c-name').textContent = c.name; $('c-role').textContent = c.role; $('c-bio').textContent = c.bio;
  $('c-name').classList.toggle('long', c.name.length > 6); $('c-name').classList.toggle('xlong', c.name.length > 9);          // lange Namen (MOHAMMAD) kleiner, damit sie ins Panel passen
  $('c-index').textContent = `${String(selected + 1).padStart(2, '0')} / ${String(CHARACTERS.length).padStart(2, '0')}`;
  $('stats').innerHTML = Object.entries(c.stats).map(([k, v]) =>
    `<li><span class="lbl">${k[0] + k.slice(1).toLowerCase()}</span><span class="bar">${Array.from({ length: SEGMENTS }, (_, i) => `<i class="seg${i < v ? ' on' : ''}"></i>`).join('')}</span></li>`).join('');
  document.querySelectorAll('.card').forEach((el, i) => el.classList.toggle('active', i === selected));
}
function select(i) {
  const n = CHARACTERS.length; i = ((i % n) + n) % n;
  if (i === selected) return;
  selected = i;
  panel.classList.add('swap');
  setTimeout(() => { renderPanel(); panel.classList.remove('swap'); }, 230);
}
$('roster').innerHTML = CHARACTERS.map((c, i) => `<button class="card" data-i="${i}" aria-label="${c.name}"><canvas width="216" height="216"></canvas><span class="nm">${c.name}</span></button>`).join('');
document.querySelectorAll('.card').forEach((el) => el.addEventListener('click', () => select(+el.dataset.i)));
$('prev').addEventListener('click', () => select(selected - 1));
$('next').addEventListener('click', () => select(selected + 1));
addEventListener('keydown', (e) => { if (e.key === 'ArrowLeft') select(selected - 1); if (e.key === 'ArrowRight') select(selected + 1); });
renderPanel();

// Porträts für die Roster-Karten: einmalig aus dem gerenderten Bild ausschneiden (Kopf-Region)
let portraitsDue = Infinity;
const _hp = new THREE.Vector3();
function capturePortraits() {
  // einmal mit voller Helligkeit rendern (die nicht gewählte Figur ist sonst abgedunkelt)
  const saved = [];
  for (const f of Object.values(figures)) {
    saved.push([f, f.splat?.recolor.clone(), f.meshMat.color.clone()]);
    f.splat?.recolor.copy(BASE_TINT).multiplyScalar(1.1); f.meshMat.color.copy(BASE_TINT).multiplyScalar(1.1);
  }
  renderScene();
  document.querySelectorAll('.card canvas').forEach((cv, i) => {
    const f = figures[CHARACTERS[i].id]; if (!f || !f.bones.spine006) return;
    f.bones.spine006.getWorldPosition(_hp); _hp.y += 0.09; _hp.project(camera);   // ~Gesichtsmitte
    const px = (_hp.x * 0.5 + 0.5) * glCanvas.width, py = (-_hp.y * 0.5 + 0.5) * glCanvas.height;
    const pxPerM = (FH / 2) / (CAM_D * Math.tan(THREE.MathUtils.degToRad(FOV / 2)));
    const size = 0.46 * (glCanvas.height / FH) * pxPerM;   // ~0.46 m Ausschnitt (Kopf + Schulteransatz)
    const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height);
    g.filter = 'brightness(1.5) contrast(1.06) saturate(1.05)';
    g.drawImage(glCanvas, px - size / 2, py - size * 0.36, size, size, 0, 0, cv.width, cv.height);   // Gesicht im oberen Drittel, über dem Namensschild
  });
  // Farben zurücksetzen und das normale Bild erneut rendern, damit kein heller Zwischenframe sichtbar wird
  for (const [f, rc, mc] of saved) { if (f.splat) f.splat.recolor.copy(rc); f.meshMat.color.copy(mc); }
  renderScene();
}

// Einmal-Animationen aus Blender, überblendet mit dem Atem-Loop:
//  - Andis: LookBack_L/_R (über die Schulter schauen) bei jeder Explosion
//  - Carlos: Pockets / HeadTilt_L / HeadTilt_R zufällig alle paar Sekunden (natürliche Idle-Gesten)
const FADE_IN = 0.25, FADE_OUT = 0.45;
const shots = Object.fromEntries(CHARACTERS.map((c) => [c.id, { state: 'idle', action: null, weight: 0 }]));
const turn = shots.andis;                                    // (Test-Hook: __demo.turn)
function playShot(id, act) {
  const f = figures[id], s = shots[id];
  if (!f || !act || s.action) return false;                 // läuft schon -> nicht neu starten
  act.reset().setEffectiveWeight(1).fadeIn(FADE_IN).play();
  f.idle.fadeOut(FADE_IN);
  s.action = act; s.state = 'playing';
  return true;
}
function updateShot(id) {
  const f = figures[id], s = shots[id], act = s && s.action;
  if (!f || !act) { if (s) s.weight = 0; return; }
  if (s.state === 'playing' && act.time >= act.getClip().duration - FADE_OUT) {
    // Basisgewicht 1, fadeIn() blendet dann 0 -> 1 (Gewicht 0 hier würde den Fade auf 0 festnageln)
    f.idle.enabled = true; f.idle.setEffectiveWeight(1); f.idle.fadeIn(FADE_OUT);
    act.fadeOut(FADE_OUT);
    s.state = 'returning';
  }
  s.weight = act.enabled ? act.getEffectiveWeight() : 0;
  if (s.state === 'returning' && (!act.enabled || s.weight <= 0.001)) {
    act.stop(); s.action = null; s.state = 'idle'; s.weight = 0;
  }
}
const _sp = new THREE.Vector3();
function triggerTurn(explosionX) {
  const a = figures.andis; if (!a) return;
  a.root.getWorldPosition(_sp).project(camera);
  const andisScreenX = (_sp.x * 0.5 + 0.5) * FW;
  playShot('andis', explosionX > andisScreenX ? a.look.R : a.look.L);   // über die Schulter zur Explosion
}
let nextFidget = 5;
function scheduleFidgets(dt) {
  const c = figures.person1; if (!c || !c.fidgets.length) return;
  nextFidget -= dt;
  if (nextFidget > 0 || shots.person1.action) return;
  const pool = c.fidgets.flatMap((a) => (a.getClip().name.endsWith('Pockets') ? [a, a] : [a]));
  const act = pool[Math.floor(Math.random() * pool.length)];
  if (playShot('person1', act)) nextFidget = act.getClip().duration + rand(7, 13);
}

// ---------------- Partikel-Sprites ----------------
function sprite(size, stops) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'), gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [o, col] of stops) gr.addColorStop(o, col);
  g.fillStyle = gr; g.fillRect(0, 0, size, size); return c;
}
function puff([r, g, b], a) {      // wolkiger, unregelmäßiger Ballen in einer Farbe
  const s = 128, c = document.createElement('canvas'); c.width = c.height = s; const ctx = c.getContext('2d');
  for (let i = 0; i < 18; i++) {
    const x = s / 2 + rand(-24, 24), y = s / 2 + rand(-24, 24), rr = rand(16, 40);
    const gr = ctx.createRadialGradient(x, y, 0, x, y, rr);
    gr.addColorStop(0, `rgba(${r},${g},${b},${a})`); gr.addColorStop(1, `rgba(${r},${g},${b},0)`);
    ctx.fillStyle = gr; ctx.fillRect(0, 0, s, s);
  }
  return c;
}
const SPR = {
  core: sprite(128, [[0, 'rgba(255,255,240,1)'], [0.25, 'rgba(255,230,160,.9)'], [0.6, 'rgba(255,150,50,.35)'], [1, 'rgba(255,90,20,0)']]),
  hot: sprite(64, [[0, 'rgba(255,250,220,1)'], [0.35, 'rgba(255,210,110,.85)'], [1, 'rgba(255,140,40,0)']]),
  flame: sprite(64, [[0, 'rgba(255,190,80,.95)'], [0.45, 'rgba(255,110,30,.6)'], [1, 'rgba(200,50,10,0)']]),
  ember: sprite(64, [[0, 'rgba(230,80,20,.8)'], [0.5, 'rgba(150,40,10,.35)'], [1, 'rgba(90,20,5,0)']]),
  smoke: (() => {   // weicher, leicht unregelmäßiger Rauchballen
    const s = 128, c = document.createElement('canvas'); c.width = c.height = s; const g = c.getContext('2d');
    for (let i = 0; i < 14; i++) {
      const x = s / 2 + rand(-22, 22), y = s / 2 + rand(-22, 22), r = rand(22, 44);
      const gr = g.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, 'rgba(60,62,66,.22)'); gr.addColorStop(1, 'rgba(60,62,66,0)');
      g.fillStyle = gr; g.fillRect(0, 0, s, s);
    }
    return c;
  })(),
  puffHot: puff([255, 214, 120], 0.55), puffMid: puff([235, 110, 35], 0.5), puffDark: puff([70, 48, 38], 0.45),
  snow: sprite(32, [[0, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,.7)'], [1, 'rgba(255,255,255,0)']]),
};

// ---------------- Feuer ----------------
const FIRE_SRC = [[302, 566, 0.55], [434, 549, 0.5], [818, 579, 0.6], [996, 586, 0.5], [1101, 312, 0.45], [66, 612, 1.0]]
  .map(([x, y, s]) => { const [fx, fy] = imgToFrame(x, y); return { x: fx, y: fy, s, life: Infinity }; })
  .filter((f) => f.x > -40);
const flames = [], smokes = [], sparks = [];
function emitFire(F, dt) {
  const s = F.s * (F.life === Infinity ? 1 : clamp01(F.life / 3));
  if (s <= 0.02) return;
  const n = 140 * s * dt + Math.random();
  for (let i = 0; i < n; i++) {
    const off = rand(-1, 1);
    flames.push({ x: F.x + off * 11 * s, y: F.y + rand(-2, 3), vx: -off * 10 * s, vy: -rand(40, 90) * (0.45 + s),
      life: 0, max: rand(0.3, 0.75) * (0.6 + s * 0.6), size: rand(6, 12) * (0.5 + s), flick: rand(0, 6.28) });
  }
  if (Math.random() < 7 * s * dt) smokes.push({ x: F.x + rand(-6, 6), y: F.y - 18 * s, vx: rand(4, 12), vy: -rand(14, 26), life: 0, max: rand(3.5, 6), size: 10 * s, grow: rand(14, 26) * s, a: 0.55 });
  if (Math.random() < 5 * s * dt) sparks.push({ x: F.x, y: F.y - 6, vx: rand(-20, 20), vy: -rand(40, 110), g: -10, life: 0, max: rand(1, 2.4), size: rand(1, 1.8) });
}

// ---------------- Explosionen ----------------
// sichtbare Gebäudeteile (Bildkoordinaten), nicht hinter den Figuren
const BOOM_ZONES = [[230, 450, 345, 555, 3], [665, 712, 345, 520, 1], [925, 1040, 340, 555, 2], [1050, 1250, 430, 540, 2], [90, 205, 450, 540, 1]];
function boomSpot() {
  const tot = BOOM_ZONES.reduce((s, z) => s + z[4], 0); let r = Math.random() * tot;
  for (const z of BOOM_ZONES) { if ((r -= z[4]) <= 0) return [rand(z[0], z[1]), rand(z[2], z[3])]; }
  return [300, 450];
}
let nextBoom = 4, shake = 0, flash = 0, boomLightV = 0;
const pendingTurns = [], subBursts = [];
function fireball(ex, ey, k, sub) {
  flames.push({ x: ex, y: ey, vx: 0, vy: 0, life: 0, max: sub ? 0.12 : 0.18, size: (sub ? 45 : 80) * k, core: true });
  for (let i = 0, n = sub ? 22 : 55; i < n; i++) {
    const a = rand(0, Math.PI * 2), v = rand(40, 230) * k * Math.sqrt(Math.random());
    flames.push({ x: ex + rand(-8, 8), y: ey + rand(-8, 8), vx: Math.cos(a) * v, vy: Math.sin(a) * v * 0.7 - 50, drag: 3.2,
      life: 0, max: rand(0.45, 1.1), size: rand(30, 75) * k, flick: rand(0, 6.28), av: rand(0.45, 0.9) });
  }
}
function explode(spot) {                     // spot: optional [x, y] in Bildkoordinaten (Test)
  const [ex, ey] = imgToFrame(...(spot || boomSpot()));
  const k = rand(1.1, 1.6);
  fireball(ex, ey, k, 0);
  // zwei, drei versetzte Nachdetonationen lassen den Feuerball unregelmäßig aufquellen
  for (let i = 0, n = 2 + (Math.random() < 0.5); i < n; i++) subBursts.push({ at: T + rand(0.06, 0.3), x: ex + rand(-40, 40) * k, y: ey + rand(-30, 10) * k, k: k * rand(0.45, 0.7) });
  for (let i = 0; i < 70; i++) {
    const a = rand(-Math.PI, 0), v = rand(120, 420) * k;
    sparks.push({ x: ex, y: ey, vx: Math.cos(a) * v, vy: Math.sin(a) * v, g: 260, life: 0, max: rand(0.8, 2.0), size: rand(1, 2.4) });
  }
  for (let i = 0; i < 26; i++) {
    smokes.push({ x: ex + rand(-30, 30) * k, y: ey + rand(-20, 15) * k, vx: rand(-25, 25) + 8, vy: -rand(10, 40), life: -rand(0, 0.4),
      max: rand(6, 10), size: rand(30, 60) * k, grow: rand(14, 30) * k, a: 0.95 });
  }
  FIRE_SRC.push({ x: ex + rand(-10, 10), y: ey + 14, s: 0.45 * k, life: rand(7, 12) });   // Brand im Fenster
  shake = Math.min(1.4, shake + 1.0 * k);
  flash = 1; boomLightV = 1;
  flashEl.style.setProperty('--fx', (ex / FW * 100) + '%'); flashEl.style.setProperty('--fy', (ey / FH * 100) + '%');
  pendingTurns.push({ at: T + 0.35, x: ex });                                      // Reaktionszeit
  nextBoom = rand(7, 13);
}

// ---------------- Schnee ----------------
const snowBack = Array.from({ length: 320 }, () => newFlake(true, true));
const snowFront = Array.from({ length: 110 }, () => newFlake(false, true));
function newFlake(far, anyY) {
  const z = far ? rand(0.25, 0.6) : rand(0.7, 1.6);
  return { x: rand(-50, FW + 50), y: anyY ? rand(-20, FH) : rand(-60, -10), z, ph: rand(0, 6.28), sp: rand(0.6, 1.4) };
}
function drawSnow(ctx, flakes, dt, t, far) {
  const wind = 26 + 18 * Math.sin(t * 0.21) + 8 * Math.sin(t * 0.83);
  for (let i = 0; i < flakes.length; i++) {
    const f = flakes[i];
    f.y += (far ? 34 : 75) * f.z * f.sp * dt;
    f.x += (wind * f.z + Math.sin(t * 1.3 + f.ph) * 14 * f.z) * dt;
    if (f.y > FH + 20 || f.x > FW + 60) { flakes[i] = newFlake(far, false); continue; }
    const r = (far ? 2.2 : 3.4) * f.z + 0.6;
    ctx.globalAlpha = far ? 0.55 * f.z + 0.25 : 0.75 / (0.6 + f.z * 0.5);
    ctx.drawImage(SPR.snow, f.x - r, f.y - r, r * 2, r * 2);
  }
  ctx.globalAlpha = 1;
}

// ---------------- 2D-Ebene hinter den Figuren ----------------
function drawBack(dt, t) {
  back.setTransform(fxScale, 0, 0, fxScale, 0, 0);
  back.clearRect(0, 0, FW, FH);
  drawSnow(back, snowBack, dt, t, true);
  for (let i = FIRE_SRC.length - 1; i >= 0; i--) {
    const F = FIRE_SRC[i];
    if (F.life !== Infinity) { F.life -= dt; if (F.life <= 0) { FIRE_SRC.splice(i, 1); continue; } }
    emitFire(F, dt);
    const fl = 0.75 + 0.15 * Math.sin(t * 13 + F.x) + 0.1 * Math.sin(t * 29 + F.y);
    const s = F.s * (F.life === Infinity ? 1 : clamp01(F.life / 3)), R = 55 * s + 6;
    back.globalCompositeOperation = 'lighter'; back.globalAlpha = 0.16 * fl;     // flacher Lichtschein am Boden
    back.drawImage(SPR.flame, F.x - R * 1.6, F.y - R * 0.35, R * 3.2, R * 0.8);
  }
  // Rauch (normal gemischt)
  back.globalCompositeOperation = 'source-over';
  for (let i = smokes.length - 1; i >= 0; i--) {
    const p = smokes[i]; p.life += dt;
    if (p.life > p.max) { smokes.splice(i, 1); continue; }
    if (p.life < 0) continue;
    const u = p.life / p.max;
    p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= 1 - 0.3 * dt; p.vy *= 1 - 0.15 * dt;
    const r = p.size + p.grow * p.life;
    back.globalAlpha = p.a * Math.min(1, p.life * 3) * (1 - u) * (1 - u);
    back.drawImage(SPR.smoke, p.x - r, p.y - r, r * 2, r * 2);
  }
  // Flammen + Feuerbälle (additiv), Farbe nach Alter
  back.globalCompositeOperation = 'lighter';
  for (let i = flames.length - 1; i >= 0; i--) {
    const p = flames[i]; p.life += dt;
    if (p.life > p.max) { flames.splice(i, 1); continue; }
    const u = p.life / p.max;
    if (p.core) {
      const r = p.size * (0.35 + 0.65 * Math.min(1, u * 5));
      back.globalAlpha = Math.pow(1 - u, 1.6);
      back.drawImage(SPR.core, p.x - r, p.y - r, r * 2, r * 2);
      continue;
    }
    if (p.drag) { p.vx *= 1 - p.drag * dt; p.vy *= 1 - p.drag * dt; p.vy -= 30 * dt; }
    p.x += (p.vx + Math.sin(t * 14 + (p.flick || i)) * 12 * u) * dt; p.y += p.vy * dt;
    if (!p.drag) p.vx *= 1 - 3 * dt;                                     // Flammen laufen oben spitz zusammen
    const r = p.size * (p.drag ? 0.6 + u * 0.9 : (1 - 0.7 * u) * (0.8 + 0.4 * Math.sin(u * 3.14)));
    if (p.drag) {
      // Feuerball: kurz additiv glühend, dann deckende Puffs von gelb über orange zu Ruß
      if (u < 0.18) {
        back.globalAlpha = 0.22 * (1 - u / 0.18);
        back.drawImage(SPR.hot, p.x - r * 0.7, p.y - r * 0.7, r * 1.4, r * 1.4);
      }
      back.globalCompositeOperation = 'source-over';
      const img = u < 0.22 ? SPR.puffHot : u < 0.5 ? SPR.puffMid : SPR.puffDark;
      back.globalAlpha = p.av * (u < 0.5 ? 0.95 : 0.95 * ((1 - u) / 0.5) ** 2);   // eigene Deckkraft je Puff, weiches Ausblenden
      back.drawImage(img, p.x - r, p.y - r, r * 2, r * 2);
      back.globalCompositeOperation = 'lighter';
      continue;
    }
    const img = u < 0.25 ? SPR.hot : u < 0.65 ? SPR.flame : SPR.ember;
    back.globalAlpha = Math.pow(1 - u, 1.2) * 0.7;
    back.drawImage(img, p.x - r * 0.75, p.y - r * 2.2, r * 1.5, r * 3.0);
  }
  for (let i = sparks.length - 1; i >= 0; i--) {
    const p = sparks[i]; p.life += dt;
    if (p.life > p.max) { sparks.splice(i, 1); continue; }
    p.vy += p.g * dt; p.vx *= 1 - 0.6 * dt; p.x += p.vx * dt; p.y += p.vy * dt;
    const a = 1 - p.life / p.max;
    back.globalAlpha = a;
    back.fillStyle = `rgb(255,${160 + 80 * a | 0},${70 * a | 0})`;
    back.beginPath(); back.arc(p.x, p.y, p.size * (0.6 + a * 0.6), 0, Math.PI * 2); back.fill();
  }
  back.globalCompositeOperation = 'source-over'; back.globalAlpha = 1;
}
function drawFront(dt, t) {
  front.setTransform(fxScale, 0, 0, fxScale, 0, 0);
  front.clearRect(0, 0, FW, FH);
  drawSnow(front, snowFront, dt, t, false);
}

// ---------------- Maus ----------------
const mouse = { x: 0, y: 0, sx: 0, sy: 0 };
addEventListener('pointermove', (e) => { mouse.x = (e.clientX / innerWidth) * 2 - 1; mouse.y = (e.clientY / innerHeight) * 2 - 1; });

// ---------------- Figuren mit der Maus drehen ----------------
// ziehen = drehen (mit Nachschwung), Klick auf die andere Figur = auswählen, Doppelklick = nach vorn
const ROT_PER_PX = 0.011;
const drag = { fig: null, lastX: 0, startX: 0, vel: 0, moved: 0, t: 0 };
const _pp = new THREE.Vector3();
function toFrame(e) { const r = stage.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * FW, (e.clientY - r.top) / r.height * FH]; }
function pickFigure(e) {
  const [fx, fy] = toFrame(e); let best = null, bd = Infinity;
  for (const f of Object.values(figures)) {
    f.root.getWorldPosition(_pp); _pp.y = 0.9; _pp.project(camera);
    const sx = (_pp.x * 0.5 + 0.5) * FW, d = Math.abs(fx - sx);
    if (d < 170 && fy > 150 && fy < FH - 140 && d < bd) { bd = d; best = f; }
  }
  return best;
}
const uiHit = (e) => e.target.closest && e.target.closest('button, #panel, #topbar, #roster, .arrow');
stage.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || uiHit(e)) return;
  const f = pickFigure(e); if (!f) return;
  Object.assign(drag, { fig: f, lastX: e.clientX, startX: e.clientX, vel: 0, moved: 0, t: performance.now() });
  stage.setPointerCapture(e.pointerId); stage.style.cursor = 'grabbing'; e.preventDefault();
});
stage.addEventListener('pointermove', (e) => {
  if (!drag.fig) { stage.style.cursor = !uiHit(e) && pickFigure(e) ? 'grab' : ''; return; }
  const now = performance.now(), dx = e.clientX - drag.lastX, dt = Math.max(1, now - drag.t) / 1000;
  const r = stage.getBoundingClientRect(), k = FW / r.width;          // Bildschirm-px -> Rahmen-px
  drag.fig.yaw += dx * k * ROT_PER_PX;
  drag.vel = damp(drag.vel, dx * k * ROT_PER_PX / dt, 18, dt);
  drag.moved += Math.abs(dx); drag.lastX = e.clientX; drag.t = now; drag.fig.rest = 0;
});
function endDrag(e) {
  if (!drag.fig) return;
  if (drag.moved < 5) select(drag.fig.index);                        // Klick ohne Ziehen = auswählen
  else drag.fig.spin = performance.now() - drag.t < 80 ? drag.vel : 0;   // Nachschwung nur bei Loslassen in Bewegung
  drag.fig = null; stage.style.cursor = '';
}
stage.addEventListener('pointerup', endDrag); stage.addEventListener('pointercancel', endDrag);
stage.addEventListener('dblclick', (e) => { const f = !uiHit(e) && pickFigure(e); if (f) { f.spin = 0; f.rest = 99; } });
function updateYaw(f, dt) {
  if (drag.fig === f) return;
  f.yaw += f.spin * dt; f.spin *= Math.exp(-2.6 * dt);
  if (Math.abs(f.spin) < 0.05) {
    f.rest += dt;
    if (f.rest > 3.5) {                                                // nach einer Pause langsam wieder nach vorn
      const target = Math.round(f.yaw / (2 * Math.PI)) * 2 * Math.PI;
      f.yaw = damp(f.yaw, target, f.rest > 50 ? 8 : 1.6, dt);
    }
  } else f.rest = 0;
}

// ---------------- Menü ----------------
const toast = $('toast');
function showToast(text) { toast.textContent = text; toast.style.opacity = 1; clearTimeout(showToast.t); showToast.t = setTimeout(() => (toast.style.opacity = 0), 1100); }
document.querySelectorAll('[data-nav]').forEach((el) => el.addEventListener('click', () => {
  const nav = el.dataset.nav;
  if (nav !== 'play') return showToast({ locker: 'SPIND', quests: 'AUFTRÄGE', options: 'OPTIONEN' }[nav] || nav.toUpperCase());
  startGame();
}));
$('btn-ready').addEventListener('click', startGame);

// Spiel starten: spielbar sind Figuren mit Geh-/Lauf-Animationen (Carlos, Mohammad, Daunenjacke).
// Läuft der Startbildschirm über den Spiel-Server (/startbildschirm/), geht es relativ zurück, sonst zum Spiel-Port 5177.
const PLAYABLE = ['person1', 'mohammad', 'daunenjacke'];
const GAME_URL = location.pathname.startsWith('/startbildschirm') ? '/' : `${location.protocol}//${location.hostname}:5177/`;
function startGame() {
  const c = CHARACTERS[selected];
  if (!PLAYABLE.includes(c.id)) { showToast(`${c.name} NOCH NICHT SPIELBAR`); return; }
  $('fade').style.opacity = 1;
  showToast(`${c.name} · EINSATZ`);
  setTimeout(() => { location.href = `${GAME_URL}?figur=${c.id}`; }, 900);
}
$('btn-custom').addEventListener('click', () => showToast('ANPASSEN'));

// ---------------- Loop ----------------
const clock = new THREE.Clock();
const timing = {};
const q = new THREE.Quaternion(), Y = new THREE.Vector3(0, 1, 0), X = new THREE.Vector3(1, 0, 0);
const _c = new THREE.Color(), WARM = new THREE.Color(1.0, 0.55, 0.25);
let T = 0;
function tick(rawDt, scripted = false, render = true) {
  const dt = Math.min(rawDt, 0.05); T += dt; const t = T;
  if (!scripted) adaptQuality(rawDt);              // gescriptete Schritte (Tests) nicht in die Qualität einrechnen
  mouse.sx += (mouse.x - mouse.sx) * Math.min(1, dt * 3);
  mouse.sy += (mouse.y - mouse.sy) * Math.min(1, dt * 3);

  nextBoom -= dt; if (nextBoom <= 0) explode();
  shake *= Math.exp(-dt * 2.6); flash *= Math.exp(-dt * 5); boomLightV *= Math.exp(-dt * 3.2);
  const sh = shake * 16;
  const sx = sh * (Math.sin(t * 61) * 0.6 + Math.sin(t * 37.3) * 0.4), sy = sh * (Math.sin(t * 53.7) * 0.6 + Math.sin(t * 29.1) * 0.4);
  world.style.transform = `translate(${sx}px, ${sy}px) rotate(${shake * 0.35 * Math.sin(t * 41)}deg) scale(1.025)`;
  flashEl.style.opacity = flash.toFixed(3);
  const par = `translate(${-mouse.sx * 14}px, ${-mouse.sy * 8}px)`;
  bgEl.style.transform = treesEl.style.transform = `${par} scale(${BG_ZOOM})`;
  backCanvas.style.transform = par;
  frontCanvas.style.transform = `translate(${-mouse.sx * 26}px, ${-mouse.sy * 14}px)`;

  // Kamera schwenkt weich zur gewählten Figur
  const selX = CHARACTERS[selected].x;
  cam.x = damp(cam.x, selX - SEL_NDC_X * HALF_W, 3.2, dt);
  camera.position.set(cam.x + mouse.sx * 0.08, CAM_Y - mouse.sy * 0.04, CAM_D);
  lookTarget.set(cam.x + mouse.sx * 0.03, LOOK_Y, 0); camera.lookAt(lookTarget);
  selRing.position.x = damp(selRing.position.x, selX, 6, dt);
  selRing.material.opacity = 0.75 + 0.25 * Math.sin(t * 2.4);
  fireLight.intensity = 2.2 + Math.sin(t * 11) * 0.5 + Math.sin(t * 23.7) * 0.3;
  boomLight.intensity = boomLightV * 60;

  while (pendingTurns.length && pendingTurns[0].at <= T) triggerTurn(pendingTurns.shift().x);
  scheduleFidgets(dt);
  for (let i = subBursts.length - 1; i >= 0; i--) if (subBursts[i].at <= T) { const b = subBursts.splice(i, 1)[0]; fireball(b.x, b.y, b.k, 1); shake = Math.min(1.5, shake + 0.25); }
  for (const [id, f] of Object.entries(figures)) {
    f.mixer.update(dt);
    updateShot(id);
    const head = f.bones.spine006, chest = f.bones.spine003;
    const look = 1 - shots[id].weight;                     // Maus-Blick nur ohne laufende Geste
    if (id !== 'andis' && shake > 0.05 && chest) { q.setFromAxisAngle(X, -shake * 0.05); chest.quaternion.premultiply(q); }   // Zusammenzucken
    if (head && look > 0) {
      const ls = look * (f.c.lookScale ?? 1);              // Luca schaut auf seine Hände -> nur wenig Maus-Blick
      q.setFromAxisAngle(Y, (mouse.sx * 0.22 + (id === 'andis' ? -0.04 : 0.04)) * ls); head.quaternion.premultiply(q);
      q.setFromAxisAngle(X, mouse.sy * 0.10 * ls); head.quaternion.premultiply(q);
    }
    updateYaw(f, dt); f.root.rotation.y = f.yaw;
    f.root.updateMatrixWorld(true);
    updateSplatSkin(f);
    // nicht gewählte Figur abdunkeln; Explosionen färben die Splats kurz warm
    f.dim = damp(f.dim, f.index === selected ? 1 : (f.index > selected ? 0.35 : 0.55), 4, dt);   // rechts = hinter dem Panel
    _c.copy(BASE_TINT).multiplyScalar(f.dim).lerp(WARM, boomLightV * 0.18);
    if (f.splat) f.splat.recolor.copy(_c);
    f.meshMat.color.copy(_c);
  }
  const p0 = performance.now();
  drawBack(dt, t);
  const p1 = performance.now();
  if (render) renderScene();             // Spark sortiert asynchron: gescriptete Zwischenschritte nur simulieren
  if (render && T >= portraitsDue) { capturePortraits(); portraitsDue = Infinity; }   // direkt nach dem Rendern auslesen
  const p2 = performance.now();
  drawFront(dt, t);
  const p3 = performance.now();
  timing.back = p1 - p0; timing.gl = p2 - p1; timing.front = p3 - p2; timing.frame = rawDt * 1000;
}
// Durchgang 1: Mesh-Füllung, Boden, Schatten. Durchgang 2: Splats immer obenauf (Mesh füllt nur Lücken)
function renderScene() {
  renderer.clear();
  camera.layers.set(0); renderer.render(scene, camera);
  renderer.clearDepth();
  renderer.shadowMap.autoUpdate = false;
  camera.layers.set(1); renderer.render(scene, camera);
  renderer.shadowMap.autoUpdate = true;
  camera.layers.set(0);
}
function frame() { const d = clock.getDelta(); if (!window.__demo?.paused) tick(d); requestAnimationFrame(frame); }
addEventListener('resize', layout);
layout();
frame();
window.__demo = {
  advance: (sec, step = 1 / 30) => { const n = Math.max(1, Math.round(sec / step)); for (let i = 0; i < n; i++) tick(step, true, i === n - 1); },
  scene, camera, figures, explode, turn, shots, playShot, renderer, timing, select, capturePortraits,
  get selected() { return selected; },
  particles: () => ({ flames: flames.length, smokes: smokes.length, sparks: sparks.length }),
};

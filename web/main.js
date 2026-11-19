// Hannover – Third-Person-Spaziergang mit Carlos (Gaussian Splats + Füll-Mesh), Bewegungsgefühl à la Death Stranding.
import * as THREE from 'three';
import { SparkRenderer } from '@sparkjsdev/spark';
import { World } from './js/world.js';
import { loadCarlos, fillerPush } from './js/carlos.js';
import { Player, P } from './js/player.js';
import { CameraRig } from './js/camera.js';
import { Input } from './js/input.js';
import { Hud } from './js/hud.js';
import { Scan, scanUniforms } from './js/scan.js';
import { Dust, Sfx } from './js/fx.js';
import { loadScans } from './js/scans.js';
import { PointCloud, makeOccluderMaterial, PC, SCAN_HOLD } from './js/pointcloud.js';
import { PostFX } from './js/post.js';
import { Karte } from './js/karte.js';
import { Gelaender } from './js/gelaender.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Einstieg über Lucas Startbildschirm (Figurenwahl); von dort kommt man mit ?figur=… zurück. Test-Parameter umgehen das.
if (!location.search) location.replace('startbildschirm/');
const canvas = $('gl');

// ---------------- Renderer ----------------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: params.has('aa'), powerPreference: 'high-performance', stencil: false });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.autoClear = false;
const FOG = new THREE.Color(0xbfc6cc);
// Look: Stadt als Punktwolke in schwarzem Raum (?mesh = alte Mesh-Ansicht; Taste P schaltet um)
let POINTS = !params.has('mesh');
let hudNote = '';
const VOID = new THREE.Color(0x030405);
renderer.setClearColor(POINTS ? VOID : FOG, 1);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(FOG, 0.0032);
const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.05, 2500);
scene.add(camera);

// Spark: eigener Splat-Durchgang auf Layer 1
const spark = new SparkRenderer({ renderer });
spark.layers.set(1);
spark.minSortIntervalMs = 60;                // Sortieren max. ~16x/s (spart GPU-Readbacks)
spark.maxStdDev = 2.4;                       // Splats bei 2.4 Sigma abschneiden (weniger Füllrate, optisch gleich)
scene.add(spark);

// ---------------- Licht: bedeckter Tag ----------------
const hemi = new THREE.HemisphereLight(0xdfe6ee, 0x5d574f, 1.25);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff3e6, 1.9);
const SUN_DIR = new THREE.Vector3(-0.45, 0.8, 0.38).normalize();
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const SH = 30;                                                   // ±30 m um den Spieler
Object.assign(sun.shadow.camera, { left: -SH, right: SH, top: SH, bottom: -SH, near: 20, far: 190 });
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.035; sun.shadow.radius = 2.5;
sun.shadow.intensity = 0.72;                                     // diffus: Schatten nicht ganz schwarz
scene.add(sun, sun.target);
for (const l of [hemi, sun]) l.layers.enable(1);                 // gleicher Licht-Zustand in beiden Durchgängen

// Himmel (Verlauf, bewölkt) + Umgebungslicht daraus
function makeSky() {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { uTop: { value: new THREE.Color(0x8e9aa6) }, uHor: { value: FOG.clone() }, uBot: { value: new THREE.Color(0x9a9890) }, uSun: { value: SUN_DIR } },
    vertexShader: 'varying vec3 vD; void main(){ vD = normalize(position); vec4 p = projectionMatrix * modelViewMatrix * vec4(position,1.); gl_Position = p.xyww; }',
    fragmentShader: `varying vec3 vD; uniform vec3 uTop, uHor, uBot, uSun;
      void main(){ float h = vD.y; vec3 c = h > 0. ? mix(uHor, uTop, pow(clamp(h,0.,1.), 0.6)) : mix(uHor, uBot, clamp(-h*4.,0.,1.));
        c += vec3(1.,.97,.9) * pow(max(dot(normalize(vD), uSun), 0.), 6.) * 0.22;
        gl_FragColor = vec4(c, 1.); }`,
  });
  const m = new THREE.Mesh(new THREE.SphereGeometry(1000, 32, 16), mat);
  m.frustumCulled = false; m.renderOrder = -10; m.name = 'Himmel';
  return m;
}
const sky = makeSky();
scene.add(sky);
// Grafikstufe: auf integrierten/mobilen GPUs Lambert statt PBR und ohne Umgebungs-Map (spart ~40 % Pixelkosten)
const gpu = (() => { try { const gl = renderer.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info'); return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : ''; } catch { return ''; } })();
const LOW_POWER = params.has('lambert') || (!params.has('pbr') && /Intel|UHD|Iris|Mali|Adreno|PowerVR|SwiftShader|llvmpipe|Basic Render/i.test(gpu));
if (!LOW_POWER) {
  const pm = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene(); envScene.add(makeSky());
  scene.environment = pm.fromScene(envScene, 0.04).texture;
  scene.environmentIntensity = 0.45;
  pm.dispose();
}

// ---------------- Spielobjekte ----------------
const world = new World(scene, renderer, camera);
const input = new Input(canvas);
const hud = new Hud();
const scan = new Scan();
const pc = new PointCloud(scene, renderer, camera);
const occMat = makeOccluderMaterial();
const post = new PostFX(renderer);   // Glühen + digitales Rauschen (nur Punktwolken-Ansicht)
let OCCLUDE = !params.has('noocc');
const dust = new Dust(scene);
const sfx = new Sfx();
let player = null, carlos = null, rig = null, karte = null, gel = null;
const state = { ready: false, playing: false, test: false, simPaused: false, ui: false };
let quality = 1, maxPR = Math.min(devicePixelRatio || 1, 1.5);

const statusEl = $('status'), statusText = statusEl.querySelector('span');
const setStatus = (t) => { statusText.textContent = t; };

function layout() {
  renderer.setPixelRatio(maxPR * quality);
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', layout);
layout();

async function boot() {
  try {
    const carlosP = loadCarlos(scene);
    setStatus('Lade Stadt …');
    // Punktwolke zuerst prüfen: fehlt sie (z. B. frischer Klon ohne erzeugte Kacheln), Mesh-Ansicht mit allen Sicht-Meshes laden
    if (POINTS) {
      try { setStatus('Lade Punktwolke …'); await pc.init(); }
      catch (e) { console.warn(e); POINTS = false; hudNote = 'Punktwolke fehlt – Mesh-Ansicht (erzeugen: pipeline/pointcloud/run_headless.py)'; }
    }
    await world.load(setStatus, { simplify: LOW_POWER, base: params.get('city') || 'assets/city/', collisionOnly: POINTS });
    if (POINTS) { pc.setOccluders(world.colliders, occMat); const s = world.meta.spawn || [0, 0, 0]; pc.prewarmScan(new THREE.Vector3(s[0], (s[1] || 0) + 1.5, s[2])); }
    applyLook();
    world.updateStreaming(new THREE.Vector3(...(world.meta.spawn || [0, 0, 0])));
    const scansP = loadScans(scene, params.get('city') || 'assets/city/');   // eigene Splat-Scans aus dem Manifest
    setStatus('Lade Carlos …');
    carlos = await carlosP;
    window.__scans = await scansP;
    player = new Player(world);
    setStatus('Lade Karte …');
    try { karte = new Karte({ world, hud, base: params.get('city') || 'assets/city/' }); await karte.load(); setupKarte(); }
    catch (e) { console.warn('Karte fehlt', e); karte = null; }
    try { gel = new Gelaender({ world, scene, camera, hud, base: params.get('city') || 'assets/city/' }); await gel.load(); setupGelaender(); }
    catch (e) { console.warn('Geländer fehlen', e); gel = null; }
    initAdmin();
    rig = new CameraRig(camera, world);
    respawn();
    carlos.onStep = (side, v) => {
      sfx.step(v, player.careful);
      if (v > 1.2 || player.rough > 0.06) { const p = carlos.footWorld(side, _v); p.y = player.pos.y; dust.emit(p, v > 3 ? 4 : 2, Math.min(1.5, v / 3), player.vel.x * 0.1, player.vel.z * 0.1); }
    };
    state.ready = true;
    statusEl.classList.add('done');
    setStatus(world.source === 'placeholder' ? 'Bereit – Testgelände (Stadtmodell fehlt noch)' : 'Bereit');
    const srcText = { manifest: `Stadtmodell: ${world.chunks.length} Dateien`, glb: 'Stadtmodell: hannover.glb', placeholder: 'Platzhalter-Gelände · kein Stadtmodell gefunden' }[world.source];
    $('source').textContent = `${srcText} · ${Math.round(world.triCount / 1000)}k Kollisionsdreiecke · BVH ${Math.round(world.bvhMs)} ms` + (world.hf ? ' · Höhenraster' : '');
    $('attrib').textContent = world.attribution || '';
    $('go').disabled = false;
    if (hudNote) setStatus(hudNote);
    if (params.has('debug')) hud.toggleDebug();
    if (params.has('autostart')) startTest();
  } catch (e) {
    console.error(e);
    statusEl.classList.add('error'); setStatus('Fehler beim Laden: ' + (e.message || e));
  }
}
const _v = new THREE.Vector3();

function respawn() {
  const m = world.meta, s = m.spawn || [0, 0, 6];
  // spawnYaw: Blickrichtung wie bei einer three.js-Kamera (rotation.y; 0 = Blick nach Norden/-Z)
  const yaw = (typeof m.spawnYaw === 'number' ? m.spawnYaw : 0) + Math.PI;
  player.spawn(s[0], s[1] ?? 0, s[2], yaw);
  rig.snap(player.pos, player.yaw);
  carlos.setTransform(player.pos.x, player.visualY, player.pos.z, player.yaw);
}

// ---------------- Look umschalten ----------------
function applyLook() {
  renderer.setClearColor(POINTS ? VOID : FOG, 1);
  sky.visible = !POINTS;
  pc.group.visible = POINTS;
  sun.castShadow = !POINTS;
  dust.points.layers.set(0);
}
addEventListener('keydown', (e) => {
  if (e.repeat || !state.ready) return;
  if (e.code === 'KeyP') toggleMeshView();
  if (e.code === 'KeyO' && POINTS) { OCCLUDE = !OCCLUDE; hud.toast(OCCLUDE ? 'VERDECKUNG AN' : 'VERDECKUNG AUS (DURCHSICHT)', 1.6); }
  if (e.code === 'KeyB' && POINTS) { pc.u.uCull.value = 1 - pc.u.uCull.value; hud.toast(pc.u.uCull.value ? 'RÜCKSEITEN AUS' : 'RÜCKSEITEN AN', 1.6); }
});

// Mesh-Ansicht braucht die Sicht-Meshes, die im Punktmodus nicht geladen werden -> Seite mit/ohne ?mesh neu laden
function toggleMeshView() {
  const p = new URLSearchParams(location.search);
  if (p.has('mesh')) p.delete('mesh'); else p.set('mesh', '1');
  location.search = p.toString();
}

// ---------------- Escape-Menü (Optionen) ----------------
const OPT_KEY = 'hannover.optionen';
const OPTS = [
  { id: 'opt-view', label: 'Sichtweite', min: 6, max: 20, step: 0.5, unit: ' m', def: PC.end,
    apply: (v) => { pc.u.uEnd.value = v; pc.u.uFar.value = pc.u.uNear.value = v * 0.65; } },
  { id: 'opt-dup', label: 'Punktdichte (Verdichtung)', min: 1, max: 4, step: 1, unit: '×', def: PC.dup, apply: (v) => { PC.dup = v; } },
  { id: 'opt-size', label: 'Punktgröße', min: 1, max: 4, step: 0.5, unit: ' px', def: PC.maxPx, apply: (v) => { pc.u.uMaxPx.value = v; } },
  { id: 'opt-expo', label: 'Belichtung', min: 0.8, max: 2.8, step: 0.1, unit: '', def: 1.7, apply: (v) => { pc.u.uExpo.value = v; } },
  { id: 'opt-glow', label: 'Glühen', min: 0, max: 2.5, step: 0.1, unit: '', def: 1, apply: (v) => { post.comp.uniforms.uGlow.value = v; } },
  { id: 'opt-noise', label: 'Digitales Rauschen', min: 0, max: 2.5, step: 0.1, unit: '', def: 1, apply: (v) => { post.comp.uniforms.uNoise.value = v; } },
  { id: 'opt-streak', label: 'Lichtschlieren', min: 0, max: 2.5, step: 0.1, unit: '', def: 1, apply: (v) => { post.comp.uniforms.uStreak.value = v; } },
  { id: 'opt-res', label: 'Auflösung', min: 50, max: 100, step: 5, unit: ' %', def: 100,
    apply: (v) => { maxPR = Math.min(devicePixelRatio || 1, 1.5) * v / 100; layout(); } },
];
function loadOpts() { try { return JSON.parse(localStorage.getItem(OPT_KEY) || '{}'); } catch { return {}; } }
function saveOpts(o) { try { localStorage.setItem(OPT_KEY, JSON.stringify(o)); } catch { /* ohne Speicher weiter */ } }
function setupMenu() {
  const box = $('opts'), saved = loadOpts();
  for (const o of OPTS) {
    const v = saved[o.id] ?? o.def;
    const row = document.createElement('label'); row.className = 'opt';
    const name = document.createElement('span'); name.textContent = o.label;
    const inp = document.createElement('input'); Object.assign(inp, { type: 'range', id: o.id, min: o.min, max: o.max, step: o.step, value: v });
    const out = document.createElement('output'); out.textContent = v + o.unit;
    row.append(name, inp, out); box.appendChild(row);
    inp.addEventListener('input', () => { const x = +inp.value; out.textContent = x + o.unit; o.apply(x); const s = loadOpts(); s[o.id] = x; saveOpts(s); });
    o.apply(+v);
  }
  $('opt-admin').addEventListener('change', (e) => setAdmin(e.target.checked));
  const hintBox = $('opt-hint');
  hintBox.checked = !$('hint').classList.contains('off');
  hintBox.addEventListener('change', () => { if (hintBox.checked === $('hint').classList.contains('off')) hud.toggleHint(); });
  $('opt-mesh').textContent = params.has('mesh') ? 'Punktwolken-Ansicht (lädt neu)' : 'Mesh-Ansicht (lädt neu)';
  $('opt-mesh').addEventListener('click', toggleMeshView);
  $('opt-reset').addEventListener('click', () => { saveOpts({}); for (const o of OPTS) { const i = $(o.id); i.value = o.def; i.dispatchEvent(new Event('input')); } });
  $('to-start').addEventListener('click', () => { location.href = 'startbildschirm/'; });
}

// ---------------- Karte (Tab), Reisen mit der Bahn ----------------
function karteAuf(modus, zone) {
  if (!karte || !karte.ready || karte.offen) return;
  state.ui = true; karte.zoneBeimOeffnen = zone || null;
  input.keys.clear(); if (document.pointerLockElement) document.exitPointerLock();
  karte.oeffnen(modus, zone ? zone.station : null);
}
function reiseAn(x, z, yawDeg) {
  const y = (world.groundAt(x, z, 500, 2000) || { y: 0 }).y;
  player.spawn(x, y, z, THREE.MathUtils.degToRad(yawDeg));
  rig.snap(player.pos, player.yaw); carlos.setTransform(player.pos.x, player.visualY, player.pos.z, player.yaw);
  world.updateStreaming(player.pos);
}
function setupKarte() {
  karte.onTravel = (st) => { const a = st.ankunft; reiseAn(a.x, a.z, a.yaw); karte.sperre = 4; hud.toast(st.name.toUpperCase(), 2.4); };
  karte.onClose = (grund, modus) => {
    state.ui = false; input.edges.clear();
    const z = karte.zoneBeimOeffnen; karte.zoneBeimOeffnen = null;
    if (grund !== 'reise' && modus === 'reise' && z) {          // abgebrochen: zurück nach oben bzw. ein Stück zurück
      if (z.art === 'eingang') { const e = z.e; reiseAn(e.top[0] - e.dir[0] * 2.2, e.top[1] - e.dir[1] * 2.2, Math.atan2(-e.dir[0], -e.dir[1]) * 180 / Math.PI); }
      else { const P = karte.stat.passerelle, s = P.s_ende - 15; reiseAn(P.origin[0] + P.axis[0] * s, P.origin[1] + P.axis[1] * s, Math.atan2(-P.axis[0], -P.axis[1]) * 180 / Math.PI); }
      karte.sperre = 2;
    }
    if (state.playing && !state.test) enter();
  };
  addEventListener('keydown', (e) => {
    if (!state.playing || !karte) return;
    if (karte.offen && (e.code === 'Escape' || (e.code === 'KeyM' && karte.modus === 'karte'))) { e.preventDefault(); karte.schliessen(); return; }
  });
}

// ---------------- Geländer-Rutschen (Passerelle-Treppen) ----------------
let gelZiel = null, rutschVorher = false;
function setupGelaender() {
  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || !state.playing || state.ui || !gel || gel.aktiv) return;
    if (gelZiel && gel.starten(gelZiel)) { hud.toast('GELÄNDER!', 1.0); sfx.thud(0.3); }
  });
}
// Rückgabe true = Figur wird vom Rutschen gesteuert (normale Bewegung aussetzen)
function rutschen(dt) {
  if (!gel) return false;
  const r = gel.update(dt, T);
  if (!r) {
    if (rutschVorher) { rutschVorher = false; carlos.sitz = 0; }
    gelZiel = state.ui ? null : gel.zielen(player); gel.hervorheben(gelZiel);
    hud.setPrompt && hud.setPrompt(gelZiel ? 'KLICK · GELÄNDER RUNTERRUTSCHEN' : '');
    return false;
  }
  hud.setPrompt && hud.setPrompt('');
  rutschVorher = true;
  if (r.ende) {
    const y = (world.groundAt(r.pos.x, r.pos.z, r.pos.y + 2, 10) || { y: r.pos.y }).y;
    player.spawn(r.pos.x, y, r.pos.z, r.yaw); player.speed = Math.min(3, r.v * 0.5);
    carlos.sitz = 0; rig.shake = Math.max(rig.shake, 0.35); sfx.thud(0.6); dust.emit(player.pos, 6, 1.0);
    return false;
  }
  player.pos.copy(r.pos); player.prev.copy(r.pos); player.visualY = r.pos.y; player.yaw = r.yaw; player.speed = 0; player.vy = 0;
  carlos.sitz = r.sitz; carlos.sitzWack = r.wack || 0;
  return true;
}

// ---------------- Start / Pause ----------------
function enter() {
  if (!state.ready) return;
  sfx.init();
  input.lock();
  // falls der Browser den Pointer Lock verweigert: trotzdem spielen, Maus bewegt die Kamera direkt
  clearTimeout(enter.t);
  enter.t = setTimeout(() => { if (!input.locked) startFree(); }, 600);
}
function startFree() {
  input.free = true; state.playing = true;
  $('start').classList.add('off'); $('pause').classList.add('off'); hud.show(true);
  hud.toast('KAMERA: MAUS ODER ← →', 3);
}
document.addEventListener('pointerlockerror', () => { if (state.ready && !input.locked) startFree(); });
addEventListener('keydown', (e) => {
  if (e.code === 'Escape' && input.free && state.playing && !state.test && !state.ui) { input.free = false; input.keys.clear(); $('pause').classList.remove('off'); }
});
$('go').addEventListener('click', enter);
$('resume').addEventListener('click', enter);
setupMenu();
document.addEventListener('pointerlockchange', () => {
  const locked = document.pointerLockElement === canvas;
  if (locked) { input.free = false; state.playing = true; $('start').classList.add('off'); $('pause').classList.add('off'); hud.show(true); }
  else if (state.playing && !state.test && !state.ui) { $('pause').classList.remove('off'); }
});
function startTest() {                           // Testmodus: ohne Pointer Lock spielbar (Skript-Eingabe)
  state.test = true; state.playing = true; input.script.__enabled = true;
  $('start').classList.add('off'); $('pause').classList.add('off'); hud.show(true);
}

// ---------------- Admin-Modus (Taste M): keine Ausdauer, kein Stolpern, sehr schnell ----------------
function setAdmin(on, quiet = false) {
  if (!player) return;
  player.admin = !!on;
  document.body.classList.toggle('admin', player.admin);
  const ab = $('opt-admin'); if (ab) ab.checked = player.admin;
  try { localStorage.setItem('hannover.admin', player.admin ? '1' : '0'); } catch { /* ohne Speicher weiter */ }
  if (!quiet) hud.toast(player.admin ? 'ADMIN-MODUS AN' : 'ADMIN-MODUS AUS', 2.2);
}
function initAdmin() {
  let on = params.has('admin');
  try { if (!on) on = localStorage.getItem('hannover.admin') === '1'; } catch { /* egal */ }
  setAdmin(on, true);
}

// ---------------- Simulation ----------------
let T = 0;
const stepped = { inp: null };
const NO_INPUT = Object.freeze({ forward: false, back: false, left: false, right: false, run: false, careful: false, balL: false, balR: false, jump: false, scan: false, swap: false, help: false, debug: false });
function tick(dt) {
  T += dt;
  const active = state.playing && (input.active || state.test);
  const inp = active ? input.state() : NO_INPUT;
  if (!active) input.edges.clear();
  const look = input.consumeLook(dt, rig.sens);
  if (!active) { look.dx = look.dy = look.wheel = 0; }
  stepped.inp = inp;
  if (inp.help) hud.toggleHint();
  if (inp.debug) hud.toggleDebug();
  if (inp.swap) rig.side *= -1;
  if (inp.admin) setAdmin(!player.admin);
  if (inp.karte && karte && !state.ui) karteAuf('karte');

  const wasStumbling = player.stumbling;
  const gleitet = rutschen(dt);
  if (!gleitet) player.update(dt, inp, rig.yaw);
  for (const e of player.events) {
    if (e.type === 'stumble') { hud.toast(e.reason === 'fall' ? 'HARTE LANDUNG' : 'GESTOLPERT'); rig.shake = 1; sfx.thud(1); dust.emit(player.pos, 10, 1.4); }
    if (e.type === 'land' && e.impact > 2.5) { rig.shake = Math.max(rig.shake, Math.min(1, e.impact / 8)); sfx.thud(e.impact / 8); dust.emit(player.pos, 6, Math.min(1.5, e.impact / 5)); }
    if (e.type === 'respawn') { respawn(); hud.toast('ZURÜCKGESETZT'); }
  }
  player.events.length = 0;
  if (!player.admin && inp.run && player.exhausted && !wasStumbling && T - (tick.lastExh || -9) > 4) { hud.toast('ERSCHÖPFT', 1.4); tick.lastExh = T; }

  if (inp.scan) {
    if (POINTS && pc.ready) { if (!pc.scanning || pc.scanT > 1.5) { pc.scan(_v.set(player.pos.x, player.pos.y + 1.55, player.pos.z)); if (karte) karte.scan(_v); if (gel) gel.gescannt(_v.clone(), T); hud.scanFlash(); sfx.scan(); hud.toast('SCAN · ' + SCAN_HOLD + ' S IM GEDÄCHTNIS', 1.6); } }
    else if (!scan.active) { scan.trigger(player.pos); hud.scanFlash(); sfx.scan(); }
  }

  if (karte && !state.ui) {
    const z = karte.update(dt, player, rig.yaw);
    if (z) { hud.toast(z.art === 'passerelle' ? 'HAUPTBAHNHOF' : z.station.toUpperCase(), 1.4); karteAuf('reise', z); }
  }
  carlos.setTransform(player.pos.x, player.visualY, player.pos.z, player.yaw);
  carlos.update(dt, {
    speed: player.actualSpeed, careful: player.careful, running: player.running, stumbling: player.stumbling,
    idleTime: player.idleTime, grounded: player.grounded, bal: player.bal, yawRate: player.yawRate,
    longAccel: player.longAccel, grade: player.grade, stabilize: player.stabilize,
  });
  rig.update(dt, look, player);
  scan.update(dt, camera);
  if (POINTS) pc.update(dt, player.pos, camera, renderer);
  dust.update(dt, renderer, camera);
  sfx.setWind(player.actualSpeed);
  followShadow();
  if ((tick.stream = (tick.stream || 0) - dt) <= 0) { tick.stream = 0.5; world.updateStreaming(player.pos); }
  sky.position.copy(camera.position);
  world.updateSpecial(T);

  hud.update(dt, {
    camYaw: rig.yaw, bal: player.bal, stamina: player.stamina, exhausted: player.exhausted, mode: player.mode, admin: player.admin, speed: player.actualSpeed,
    stabilize: player.stabilize, inp, place: world.labelAt(player.pos.x, player.pos.z), debugText: debugText(), stream: world.stream,
  });
}

// Schattenkamera folgt dem Spieler, auf Texel gerastert (kein Flimmern)
const _lm = new THREE.Matrix4(), _li = new THREE.Matrix4(), _c = new THREE.Vector3();
function followShadow() {
  _lm.lookAt(new THREE.Vector3(), SUN_DIR.clone().negate(), new THREE.Vector3(0, 1, 0));
  _li.copy(_lm).invert();
  _c.copy(player.pos).applyMatrix4(_li);
  const texel = (2 * SH) / sun.shadow.mapSize.x;
  _c.x = Math.round(_c.x / texel) * texel; _c.y = Math.round(_c.y / texel) * texel;
  _c.applyMatrix4(_lm);
  sun.target.position.copy(_c);
  sun.position.copy(_c).addScaledVector(SUN_DIR, 100);
  sun.target.updateMatrixWorld(); sun.updateMatrixWorld();
}

// ---------------- Rendern ----------------
// Durchgang 1: Stadt + Füll-Mesh (schreibt Tiefe, ~3 cm nach hinten versetzt) + Schatten.
// Durchgang 2: Splats mit Tiefentest gegen Durchgang 1 (kein clearDepth) -> Gebäude verdecken Carlos korrekt.
// Schattenkarte nur jedes zweite Bild neu (statische Stadt; nur Carlos' Schatten hinkt max. ein Bild nach)
let frameNo = 0;
renderer.shadowMap.autoUpdate = false;
function renderPoints() {
  renderer.info.reset();
  if (post.enabled) post.begin(); else renderer.clear();
  camera.updateMatrixWorld();
  // 1) unsichtbare Stadt-Meshes nur als Tiefe (leicht nach hinten versetzt) -> Punkte hinter Wänden verdeckt
  if (OCCLUDE) renderer.render(pc.occScene, camera);
  // 2) Punktwolke
  camera.layers.set(3); pc.render(renderer, scene, camera);
  // 3) Figur (Füll-Mesh / Mesh-Figur), Staub – ohne Stadt-Meshes
  world.root.visible = false; camera.layers.set(0); renderer.render(scene, camera); world.root.visible = true;
  // 4) Splats
  camera.layers.set(1); renderer.render(scene, camera);
  camera.layers.set(0);
  if (post.enabled) post.end(performance.now() / 1000);
}
function render() {
  if (POINTS) return renderPoints();
  if ((frameNo++ & 1) === 0 || params.has('shadow60')) renderer.shadowMap.needsUpdate = true;
  renderer.info.reset();
  renderer.clear();
  camera.layers.set(0); renderer.render(scene, camera);
  renderer.shadowMap.needsUpdate = false;
  camera.layers.set(1); renderer.render(scene, camera);
  camera.layers.set(0);
}
renderer.info.autoReset = false;

// ---------------- Leistung ----------------
const perf = { acc: 0, n: 0, fps: 0, ms: 0, fAcc: 0, fN: 0, lastT: performance.now() };
function adapt(rawDt) {
  if (rawDt > 0.25 || params.has('fixedq')) return;
  perf.acc += rawDt; perf.n++;
  if (perf.acc < 2) return;
  const avg = perf.acc / perf.n; perf.acc = perf.n = 0;
  if (avg > 0.021 && quality > 0.6) { quality = Math.max(0.6, quality - 0.1); layout(); }
  else if (avg < 0.0145 && quality < 1) { quality = Math.min(1, quality + 0.05); layout(); }
}
function debugText() {
  if (!hud.debug) return '';
  const i = renderer.info, p = player;
  return `${perf.fps.toFixed(0)} fps  ${perf.ms.toFixed(1)} ms  q ${quality.toFixed(2)} pr ${renderer.getPixelRatio().toFixed(2)}\n`
    + `draw ${i.render.calls}  tris ${(i.render.triangles / 1000).toFixed(0)}k  splats ${carlos.numSplats}\n`
    + (POINTS ? `punkte ${(pc.drawnPts / 1e6).toFixed(2)}M gez. / ${(pc.loadedPts / 1e6).toFixed(1)}M geladen  occ ${OCCLUDE ? 'an' : 'aus'}\n` : '')
    + `pos ${p.pos.x.toFixed(1)} ${p.pos.y.toFixed(2)} ${p.pos.z.toFixed(1)}  v ${p.actualSpeed.toFixed(2)}  ${p.mode}\n`
    + `bal ${p.bal.x.toFixed(2)} ${p.bal.y.toFixed(2)}  grade ${p.grade.toFixed(2)}  rough ${p.rough.toFixed(3)}`;
}

const clock = new THREE.Timer();
function frame(t) {
  requestAnimationFrame(frame);
  clock.update(t);
  const raw = clock.getDelta();
  if (state.ready) {
    adapt(raw);
    const now = performance.now(); perf.fAcc += now - perf.lastT; perf.fN++; perf.lastT = now;
    if (perf.fAcc > 500) { perf.ms = perf.fAcc / perf.fN; perf.fps = 1000 / perf.ms; perf.fAcc = perf.fN = 0; }
    if (!state.simPaused) {
      let dt = Math.min(raw, 0.1);
      const maxStep = player && player.admin ? 1 / 60 : 1 / 30;            // Admin-Tempo: kleinere Schritte gegen Durchtunneln
      while (dt > 1e-6) { const s = Math.min(dt, maxStep); tick(s); dt -= s; }   // große Hänger in Teilschritten
    }
    render();
  } else {
    renderer.clear();
  }
}
requestAnimationFrame(frame);
boot();

// ---------------- Test-Schnittstelle ----------------
window.__game = {
  get world() { return world; },
  get ready() { return state.ready; },
  setAdmin(on) { setAdmin(on); return player && player.admin; },
  start: startTest,
  setInput(o) { if (!state.test) startTest(); Object.assign(input.script, o); },
  clearInput() { input.script = { __enabled: true }; },
  teleport(x, z, yawDeg) {
    const y = (world.groundAt(x, z, 500, 2000) || { y: 0 }).y;
    player.spawn(x, y, z, yawDeg === undefined ? player.yaw : THREE.MathUtils.degToRad(yawDeg));
    rig.snap(player.pos, player.yaw); carlos.setTransform(player.pos.x, player.visualY, player.pos.z, player.yaw);
  },
  // deterministisch simulieren (Echtzeit pausiert), danach einmal rendern
  step(seconds, dt = 1 / 60) {
    if (!state.test) startTest();
    state.simPaused = true;
    const n = Math.max(1, Math.round(seconds / dt));
    for (let i = 0; i < n; i++) tick(dt);
    render();
    return this.state();
  },
  pause(b = true) { state.simPaused = b; },
  render() { render(); },
  // Kamera frei setzen (nur Tests): Simulation pausiert, ein Bild rendern
  lookFrom(px, py, pz, tx, ty, tz) { state.simPaused = true; camera.position.set(px, py, pz); camera.lookAt(tx, ty, tz); camera.updateMatrixWorld(); render(); },
  state() {
    const p = player, c = carlos;
    return {
      pos: [+p.pos.x.toFixed(3), +p.pos.y.toFixed(3), +p.pos.z.toFixed(3)], yawDeg: +THREE.MathUtils.radToDeg(p.yaw).toFixed(1),
      speed: +p.actualSpeed.toFixed(3), cmdSpeed: +p.speed.toFixed(3), mode: p.mode, grounded: p.grounded, sliding: !!p.sliding,
      stamina: +p.stamina.toFixed(3), exhausted: p.exhausted, bal: { x: +p.bal.x.toFixed(3), y: +p.bal.y.toFixed(3) },
      stumbles: p.stumbles, stumbling: p.stumbling, grade: +p.grade.toFixed(3), rough: +p.rough.toFixed(4),
      anim: Object.fromEntries(Object.entries(c.w).map(([k, v]) => [k, +v.toFixed(3)])), phase: +c.phase.toFixed(3),
      cam: { pos: camera.position.toArray().map((v) => +v.toFixed(2)), dist: +rig.cur.toFixed(2), yawDeg: +THREE.MathUtils.radToDeg(rig.yaw).toFixed(1) },
      place: world.labelAt(p.pos.x, p.pos.z)?.name || null, source: world.source,
    };
  },
  scan() { input.script.scan = true; },
  stumble() { player.triggerStumble('test'); },
  setCamera({ yaw, pitch, dist } = {}) {
    if (yaw !== undefined) rig.yaw = THREE.MathUtils.degToRad(yaw);
    if (pitch !== undefined) rig.pitch = THREE.MathUtils.degToRad(pitch);
    if (dist !== undefined) { rig.dist = dist; rig.cur = dist; }
  },
  setQuality(q) { quality = q; layout(); },
  perf: () => ({ fps: perf.fps, ms: perf.ms, quality, calls: renderer.info.render.calls, tris: renderer.info.render.triangles }),
  get player() { return player; }, get carlos() { return carlos; }, get rig() { return rig; }, get karte() { return karte; }, get gel() { return gel; },
  karteAuf, reiseAn,
  world, scene, camera, renderer, spark, P, fillerPush, scanUniforms, hud, input, pc, PC, post,
  setLook(p) { POINTS = !!p; applyLook(); }, setOcclude(o) { OCCLUDE = !!o; },
};

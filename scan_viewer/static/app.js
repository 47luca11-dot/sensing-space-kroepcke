import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

const $ = (id) => document.getElementById(id);
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0e1116);
const scene = new THREE.Scene();
scene.add(new SparkRenderer({ renderer }));
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 2000);
camera.position.set(0, 0, 3);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.12; controls.screenSpacePanning = true;
const grid = new THREE.GridHelper(4, 40, 0x2c3542, 0x1c232c); grid.material.transparent = true; grid.material.opacity = 0.6; scene.add(grid);
// 3DGS/COLMAP-Szenen haben y nach unten: Inhalte um 180° um x drehen, damit oben oben ist
const content = new THREE.Group(); content.rotation.x = Math.PI; scene.add(content);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404850, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(2, 4, 3); scene.add(sun);
let meshObj = null;
const plyLoader = new PLYLoader();
async function loadSurface() {
  if (meshObj) { content.remove(meshObj); meshObj.geometry.dispose(); meshObj = null; }
  if (!current?.mesh) return;
  const g = await plyLoader.loadAsync(`/api/scan/${current.id}/mesh.ply?t=${Date.now()}`);
  g.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0 });
  meshObj = new THREE.Mesh(g, mat);
  meshObj.add(new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: 0x5fd3c4, transparent: true, opacity: 0.35 })));
  meshObj.children[0].visible = false;
  content.add(meshObj);
}

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// ---------------- Zustand ----------------
let scans = [], current = null, view = 'original';
const meshes = { original: null, extracted: null };
const status = (t, err) => { $('status').textContent = t; $('status').classList.toggle('err', !!err); };
const bar = (f) => { $('bar').style.transform = `scaleX(${f})`; if (f >= 1) setTimeout(() => ($('bar').style.transform = 'scaleX(0)'), 400); };
const fmt = (b) => (b > 1e9 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(1) + ' MB');

async function refreshList(selectId) {
  scans = await (await fetch('/api/scans')).json();
  $('list').innerHTML = scans.map((s) => `<div class="item${current && s.id === current.id ? ' active' : ''}" data-id="${s.id}">
    <div class="n">${s.name.replace(/</g, '&lt;')}${s.extracted ? '<span class="tag">extrahiert</span>' : ''}</div>
    <div class="m">${fmt(s.size)}${s.info ? ` · ${s.info.selected.toLocaleString('de')} von ${s.info.total.toLocaleString('de')} Splats` : ''}</div></div>`).join('');
  document.querySelectorAll('.item').forEach((el) => el.addEventListener('click', () => open(el.dataset.id)));
  if (selectId) open(selectId);
}

// ---------------- Upload ----------------
async function upload(file) {
  if (!/\.ply$/i.test(file.name)) { status(`${file.name}: keine .ply-Datei`, true); return null; }
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload?name=' + encodeURIComponent(file.name));
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { bar(e.loaded / e.total); status(`Lade ${file.name} hoch … ${Math.round(100 * e.loaded / e.total)} %`); } };
    xhr.onload = () => {
      bar(1);
      try { const r = JSON.parse(xhr.responseText); if (xhr.status !== 200) throw new Error(r.error); resolve(r); }
      catch (e) { status(`Upload fehlgeschlagen: ${e.message}`, true); resolve(null); }
    };
    xhr.onerror = () => { status('Upload fehlgeschlagen', true); resolve(null); };
    xhr.send(file);
  });
}
async function handleFiles(files) {
  let last = null;
  for (const f of files) { const r = await upload(f); if (r) last = r; }
  if (last) { await refreshList(last.id); if ($('auto').checked) extract(); }
}
const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') $('file').click(); });
$('file').addEventListener('change', (e) => handleFiles([...e.target.files]));
for (const ev of ['dragenter', 'dragover']) addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
for (const ev of ['dragleave', 'drop']) addEventListener(ev, (e) => { e.preventDefault(); if (ev === 'drop' || e.target === document.documentElement) drop.classList.remove('over'); });
addEventListener('drop', (e) => handleFiles([...e.dataTransfer.files]));

// ---------------- Anzeige ----------------
async function loadMesh(kind) {
  if (meshes[kind]) { content.remove(meshes[kind]); meshes[kind].dispose?.(); meshes[kind] = null; }
  const m = new SplatMesh({ url: `/api/scan/${current.id}/${kind}.ply?t=${Date.now()}` });
  await m.initialized;
  meshes[kind] = m; content.add(m);
  return m;
}
function applyView() {
  document.querySelectorAll('#seg button').forEach((b) => b.classList.toggle('on', b.dataset.v === view));
  const isMesh = view === 'mesh' || view === 'wire';
  if (meshObj) { meshObj.visible = isMesh; meshObj.children[0].visible = view === 'wire'; }
  if (meshes.original) {
    meshes.original.visible = !isMesh && view !== 'extracted';
    meshes.original.opacity = view === 'both' ? 0.45 : 1;
  }
  if (meshes.extracted) meshes.extracted.visible = !isMesh && view !== 'original';
}
document.querySelectorAll('#seg button').forEach((b) => b.addEventListener('click', () => { if (!b.disabled) { view = b.dataset.v; applyView(); } }));

function frame(target) {
  // Kamera auf den Extrakt (oder die dichte Mitte des Originals) einpassen
  const info = current?.info;
  let c, size;
  if (info && target !== 'original') {
    const a = new THREE.Vector3(...info.bbox_min), b = new THREE.Vector3(...info.bbox_max);
    c = a.clone().add(b).multiplyScalar(0.5); size = a.distanceTo(b);
  } else {
    const ce = info ? info.center : [0, 0, 0];
    c = new THREE.Vector3(...ce); size = (info ? info.scale : 0.6) * 4;
  }
  c.applyMatrix4(content.matrixWorld);
  const up = info ? new THREE.Vector3(...info.up).applyQuaternion(content.quaternion) : new THREE.Vector3(0, 1, 0);
  // Boden-Gitter auf die erkannte Bodenebene legen
  if (info) {
    const n = new THREE.Vector3(...info.up), p0 = n.clone().multiplyScalar(-info.plane_d).applyMatrix4(content.matrixWorld);
    grid.position.copy(p0); grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up.clone().normalize());
    grid.scale.setScalar(Math.max(size, 0.1) / 2);
  }
  controls.target.copy(c);
  const dir = new THREE.Vector3(0, 0, 1).projectOnPlane(up).normalize().multiplyScalar(size * 1.6).add(up.clone().multiplyScalar(size * 0.35));
  camera.position.copy(c).add(dir); camera.up.copy(up);
  camera.near = size / 200; camera.far = size * 400; camera.updateProjectionMatrix();
}
addEventListener('keydown', (e) => { if (e.key === 'f' || e.key === 'F') frame(view); });

function showInfo() {
  const i = current?.info;
  $('info').style.display = i ? 'block' : 'none';
  if (!i) return;
  $('info').innerHTML = `<b>${i.selected.toLocaleString('de')}</b> von ${i.total.toLocaleString('de')} Splats (${(100 * i.selected / i.total).toFixed(1)} %)<br>
    Höhe: <b>${i.height.toFixed(2)}</b> Einheiten · ${i.taken} von ${i.clusters} Clustern<br>
    ${i.mode === 'surface' ? 'Flächenmodus (Wand/Boden-Scan)<br>' : ''}Boden: ${i.ground_inliers.toLocaleString('de')} Punkte${i.seconds ? ` · ${i.seconds} s` : ''}
    ${i.mesh ? `<br>Bereinigt: <b>${i.mesh.removed_splats.toLocaleString('de')}</b> Splats entfernt<br>Mesh: <b>${i.mesh.faces.toLocaleString('de')}</b> Dreiecke · ${i.mesh.watertight ? '<b>geschlossen</b>' : 'offen'} · ${i.mesh.components} Teil${i.mesh.components > 1 ? 'e' : ''}` : ''}`;
}

async function open(id) {
  current = scans.find((s) => s.id === id); if (!current) return;
  document.querySelectorAll('.item').forEach((el) => el.classList.toggle('active', el.dataset.id === id));
  $('empty').style.display = 'none';
  for (const k of ['original', 'extracted']) if (meshes[k]) { content.remove(meshes[k]); meshes[k].dispose?.(); meshes[k] = null; }
  $('run').disabled = false; $('dl-or').disabled = false;
  const hasEx = !!current.extracted;
  $('dl-ex').disabled = !hasEx; $('dl-mesh').disabled = !current.mesh;
  document.querySelectorAll('#seg button[data-v="extracted"], #seg button[data-v="both"]').forEach((b) => (b.disabled = !hasEx));
  document.querySelectorAll('#seg button[data-v="mesh"], #seg button[data-v="wire"]').forEach((b) => (b.disabled = !current.mesh));
  status(`Lade ${current.name} …`); bar(0.3);
  try {
    await loadMesh('original');
    if (hasEx) await loadMesh('extracted');
    await loadSurface();
    view = hasEx ? 'extracted' : 'original'; applyView(); frame(view); showInfo();
    bar(1); status(`${current.name} · ${meshes.original.numSplats.toLocaleString('de')} Splats`);
  } catch (e) { bar(1); status(`Konnte ${current.name} nicht laden: ${e.message || e}`, true); }
}

async function extract() {
  if (!current) return;
  $('run').disabled = true; status('Extrahiere, bereinige und meshe … (kann bis zu einer Minute dauern)'); bar(0.5);
  try {
    const gv = $('group').value;
    const body = { radius: +$('radius').value, min_opacity: +$('minop').value, group: gv === 'auto' ? 'auto' : gv === '1', keep_soft: $('soft').checked, clean: $('clean').checked, mesh: $('mkmesh').checked };
    const r = await fetch(`/api/scan/${current.id}/extract`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const info = await r.json(); if (!r.ok) throw new Error(info.error);
    current.info = info; current.extracted = true; current.mesh = body.mesh;
    await loadMesh('extracted'); await loadSurface();
    document.querySelectorAll('#seg button').forEach((b) => (b.disabled = !body.mesh && (b.dataset.v === 'mesh' || b.dataset.v === 'wire'))); $('dl-ex').disabled = false; $('dl-mesh').disabled = !body.mesh;
    view = 'extracted'; applyView(); frame(view); showInfo();
    bar(1); status(`Extrahiert in ${info.seconds} s`);
    refreshList();
  } catch (e) { bar(1); status(`Extraktion fehlgeschlagen: ${e.message}`, true); }
  $('run').disabled = false;
}
$('run').addEventListener('click', extract);
$('dl-ex').addEventListener('click', () => current && (location.href = `/api/scan/${current.id}/extracted.ply?download=1`));
$('dl-mesh').addEventListener('click', () => current && (location.href = `/api/scan/${current.id}/mesh.ply?download=1`));
$('dl-or').addEventListener('click', () => current && (location.href = `/api/scan/${current.id}/original.ply?download=1`));
for (const [id, d] of [['radius', 1], ['minop', 2]]) $(id).addEventListener('input', () => ($(id + '-o').textContent = (+$(id).value).toFixed(d)));

// automatisch extrahieren nach dem Upload (Option im Upload-Feld)
drop.insertAdjacentHTML('afterend', '<label style="display:flex;gap:8px;align-items:center;margin:-6px 18px 10px;color:var(--dim);font-size:12px"><input id="auto" type="checkbox" checked> nach dem Hochladen automatisch extrahieren</label>');
refreshList().then(() => { if (scans.length) open(scans[0].id); });
window.__viewer = { scene, camera, meshes, get current() { return current; }, open, extract };

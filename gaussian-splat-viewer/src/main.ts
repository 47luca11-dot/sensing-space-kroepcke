import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatFileType, SplatMesh } from '@sparkjsdev/spark';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { FloorComponent, TableComponent, type Placement, type SceneComponent } from './components';
import { exportPly } from './plyExport';
import { pickSurface, segmentObject } from './segment';
import { extractSplats, robustBounds, writeSplats, type SplatData, type ViewMode } from './splatData';
import './style.css';

// Kleine öffentliche Beispiel-Szene aus den Spark-Beispielen (~4 MB, CORS-freigegeben).
const DEFAULT_URL = 'https://sparkjs.dev/assets/splats/butterfly.spz';

const $ = <T extends Element>(sel: string) => document.querySelector<T>(sel)!;
const canvas = $<HTMLCanvasElement>('#canvas');
const urlForm = $<HTMLFormElement>('#url-form');
const urlInput = $<HTMLInputElement>('#url-input');
const fileInput = $<HTMLInputElement>('#file-input');
const flipInput = $<HTMLInputElement>('#flip-input');
const resetButton = $<HTMLButtonElement>('#reset-button');
const statusEl = $<HTMLParagraphElement>('#status');
const pickButton = $<HTMLButtonElement>('#pick-button');
const showAllButton = $<HTMLButtonElement>('#show-all-button');
const exportButton = $<HTMLButtonElement>('#export-button');
const offsetInput = $<HTMLInputElement>('#offset-input');
const voxelInput = $<HTMLInputElement>('#voxel-input');
const sizeInput = $<HTMLInputElement>('#size-input');
const opacityInput = $<HTMLInputElement>('#opacity-input');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x111216);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(0, 0, 3);

const spark = new SparkRenderer({ renderer });
scene.add(spark);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

// Licht und Umgebungsreflexionen für die Komponenten (Splats bringen ihr Licht selbst mit).
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.6;
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(2, 4, 3);
scene.add(sun, new THREE.HemisphereLight(0xffffff, 0x3a3a40, 0.5));

const components: SceneComponent[] = [new TableComponent(), new FloorComponent()];
for (const c of components) {
  c.object.visible = c.enabled;
  scene.add(c.object);
}

/** Zustand der aktuell geladenen Szene. */
let current: SplatMesh | null = null;
let data: SplatData | null = null;
let sourceBytes: Uint8Array | null = null;
let sourceName = '';
let mask: Uint8Array | null = null;
let lastSeed: THREE.Vector3 | null = null;
let loadToken = 0;

function setStatus(text: string, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

/** Leitet den Dateityp aus der Endung ab. .ply und .spz erkennt Spark auch am Inhalt,
 *  .splat/.ksplat aber nur über Endung oder explizites fileType. */
function fileTypeFromName(name: string): SplatFileType | undefined {
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

/** Viele 3DGS-Dateien nutzen die OpenCV-Konvention (Y nach unten); 180° um X korrigiert das. */
function applyOrientation(mesh: SplatMesh) {
  if (flipInput.checked) mesh.quaternion.set(1, 0, 0, 0);
  else mesh.quaternion.identity();
  mesh.updateMatrixWorld(true);
}

/** Richtet die Orbit-Kamera auf den Kern der Szene bzw. auf die freigestellte Auswahl aus. */
function frameMesh() {
  if (!current || !data) return;
  // 95 %-Radius passt für einzelne Objekte; bei Szenen mit weit entfernter Hintergrund-Hülle
  // begrenzt der 60 %-Kern den Abstand, damit die Kamera innerhalb der Hülle bleibt.
  const core = robustBounds(data, current.matrixWorld, mask, 0.6);
  const outer = robustBounds(data, current.matrixWorld, mask, 0.95);
  const center = core.center;
  const radius = Math.min(outer.radius, 1.6 * core.radius);
  const distance = (1.5 * radius) / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  camera.near = distance / 1000;
  camera.far = distance * 1000;
  camera.updateProjectionMatrix();
  // Leicht von oben: bei Scans liegt vor der Kamera oft unscharfer Boden-Vordergrund.
  const elevation = THREE.MathUtils.degToRad(30);
  camera.position
    .copy(center)
    .add(new THREE.Vector3(0, Math.sin(elevation), Math.cos(elevation)).multiplyScalar(distance));
  controls.target.copy(center);
  controls.update();
}

/** Standfläche des sichtbaren Objekts (Weltkoordinaten), für Tisch und Boden. */
function computePlacement(): Placement | null {
  if (!current || !data) return null;
  const minOpacity = Math.max(0.05, Number(opacityInput.value));
  const step = mask ? 1 : Math.max(1, Math.floor(data.count / 100_000));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const p = new THREE.Vector3();
  for (let i = 0; i < data.count; i += step) {
    if ((mask && !mask[i]) || data.opacities[i] < minOpacity) continue;
    p.fromArray(data.centers, i * 3).applyMatrix4(current.matrixWorld);
    xs.push(p.x);
    ys.push(p.y);
    zs.push(p.z);
  }
  if (!xs.length) return null;
  const q = (v: number[], f: number) => Float64Array.from(v).sort()[Math.floor(f * (v.length - 1))];
  const base = new THREE.Vector3(q(xs, 0.5), q(ys, 0.005), q(zs, 0.5));
  const radial = xs.map((x, i) => Math.hypot(x - base.x, zs[i] - base.z));
  return { base, footprint: q(radial, 0.95), height: q(ys, 0.995) - base.y };
}

function updateComponents() {
  const placement = computePlacement();
  for (const c of components) {
    c.object.visible = c.enabled && !!placement;
    if (c.object.visible) c.place(placement!, components);
  }
}

function buildComponentsUI() {
  const list = $<HTMLDivElement>('#components-list');
  for (const c of components) {
    const item = document.createElement('div');
    item.className = 'component';
    const toggle = document.createElement('input');
    toggle.type = 'checkbox';
    toggle.checked = c.enabled;
    const head = document.createElement('label');
    head.className = 'component-head';
    head.append(toggle, ` ${c.label}`);
    const settings = document.createElement('div');
    settings.className = 'component-settings';
    settings.hidden = !c.enabled;
    c.mount(settings, updateComponents);
    toggle.addEventListener('change', () => {
      c.enabled = toggle.checked;
      settings.hidden = !c.enabled;
      updateComponents();
    });
    item.append(head, settings);
    list.append(item);
  }
}

function viewMode(): ViewMode {
  return ($<HTMLInputElement>('input[name="mode"]:checked').value as ViewMode) ?? 'color';
}

/** Schreibt Maske und Darstellungsoptionen ins Mesh (immer ausgehend von den Originaldaten). */
function applyView() {
  if (!current || !data) return;
  const mode = viewMode();
  writeSplats(current, data, {
    mask,
    scale: Number(sizeInput.value) / 100,
    minOpacity: Number(opacityInput.value),
    mode,
  });
  // Spherical Harmonics würden die Schattierung mit blickabhängiger Farbe überlagern.
  current.maxSh = mode === 'color' ? 3 : 0;
  current.updateGenerator();
  // Maske und Deckkraft-Filter verändern die Standfläche.
  updateComponents();
}

function updateLabels() {
  $('#offset-value').textContent = `${offsetInput.value} %`;
  $('#voxel-value').textContent = `${voxelInput.value} %`;
  $('#size-value').textContent = `${sizeInput.value} %`;
  $('#opacity-value').textContent = opacityInput.value;
  showAllButton.disabled = !mask;
  exportButton.disabled = !mask;
}

async function load(source: { url: string } | { file: File }) {
  const token = ++loadToken;
  const name = 'url' in source ? source.url : source.file.name;
  setStatus(`Lade ${name} …`);
  setPicking(false);

  if (current) {
    scene.remove(current);
    current.dispose();
    current = null;
  }
  data = null;
  mask = null;
  lastSeed = null;
  updateLabels();
  updateComponents();

  try {
    // Bytes selbst laden, damit der Export später die Original-PLY-Zeilen übernehmen kann.
    let bytes: ArrayBuffer;
    if ('url' in source) {
      const res = await fetch(source.url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      bytes = await res.arrayBuffer();
    } else {
      bytes = await source.file.arrayBuffer();
    }
    if (token !== loadToken) return;

    const fileName = name.split(/[?#]/)[0].split('/').pop() ?? name;
    const mesh = new SplatMesh({
      fileBytes: bytes.slice(0),
      fileName,
      fileType: fileTypeFromName(name),
    });
    await mesh.initialized;

    if (token !== loadToken) {
      mesh.dispose();
      return;
    }
    current = mesh;
    data = extractSplats(mesh);
    sourceBytes = new Uint8Array(bytes);
    sourceName = fileName;
    applyOrientation(mesh);
    scene.add(mesh);
    applyView();
    frameMesh();
    setStatus(`${fileName} – ${data.count.toLocaleString('de-DE')} Splats`);
  } catch (err) {
    if (token !== loadToken) return;
    console.error(err);
    setStatus(`Fehler beim Laden: ${err instanceof Error ? err.message : String(err)}`, true);
  }
}

// --- Freistellen ------------------------------------------------------------------

let picking = false;
function setPicking(on: boolean) {
  picking = on;
  document.body.classList.toggle('picking', on);
  pickButton.classList.toggle('active', on);
}

function runSegmentation(seedObject: THREE.Vector3) {
  if (!current || !data) return;
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(current.quaternion.clone().invert());
  const { radius } = robustBounds(data, new THREE.Matrix4());
  const result = segmentObject(data, {
    seed: seedObject,
    up,
    radius,
    planeOffset: Number(offsetInput.value) / 100,
    voxel: Number(voxelInput.value) / 100,
    minOpacity: Math.max(0.05, Number(opacityInput.value)),
  });
  lastSeed = seedObject.clone();
  mask = result.mask;
  applyView();
  updateLabels();
  setStatus(
    `${result.selected.toLocaleString('de-DE')} von ${data.count.toLocaleString('de-DE')} Splats freigestellt` +
      (result.planeFound ? '' : ' (keine Tischebene gefunden)'),
  );
}

function pickAt(clientX: number, clientY: number) {
  if (!current || !data) return;
  const mesh = current;
  const splats = data;
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(ndc, camera);
  const toObject = mesh.matrixWorld.clone().invert();
  const origin = raycaster.ray.origin.clone().applyMatrix4(toObject);
  const dir = raycaster.ray.direction.clone().transformDirection(toObject);
  // Toleranzkegel von ca. 6 Pixeln um den Klickpunkt.
  const coneTan = (6 / (rect.height / 2)) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));

  setPicking(false);
  setStatus('Stelle frei …');
  // Erst neu zeichnen lassen, dann rechnen.
  requestAnimationFrame(() =>
    setTimeout(() => {
      const { radius } = robustBounds(splats, new THREE.Matrix4());
      const seed = pickSurface(splats, origin, dir, coneTan, radius);
      if (!seed) {
        setPicking(true);
        setStatus('Keine Oberfläche getroffen – bitte direkt auf das Objekt klicken.', true);
        return;
      }
      runSegmentation(seed);
      frameMesh();
    }),
  );
}

let downPos: { x: number; y: number } | null = null;
canvas.addEventListener('pointerdown', (e) => (downPos = { x: e.clientX, y: e.clientY }));
canvas.addEventListener('pointerup', (e) => {
  if (!picking || !downPos || e.button !== 0) return;
  // Nur echte Klicks auswerten, kein Drehen der Kamera.
  if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) < 5) pickAt(e.clientX, e.clientY);
});

pickButton.addEventListener('click', () => {
  setPicking(!picking);
  if (picking) setStatus('Jetzt auf das Objekt in der Szene klicken.');
});

showAllButton.addEventListener('click', () => {
  mask = null;
  lastSeed = null;
  applyView();
  updateLabels();
  frameMesh();
  if (data) setStatus(`${sourceName} – ${data.count.toLocaleString('de-DE')} Splats`);
});

exportButton.addEventListener('click', () => {
  if (!data || !mask) return;
  const blob = exportPly(sourceBytes, data, mask);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${sourceName.replace(/\.[^.]+$/, '')}-freigestellt.ply`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

for (const input of [offsetInput, voxelInput]) {
  input.addEventListener('input', updateLabels);
  input.addEventListener('change', () => lastSeed && runSegmentation(lastSeed));
}

// --- Darstellung ------------------------------------------------------------------

for (const input of [sizeInput, opacityInput]) {
  input.addEventListener('input', updateLabels);
  input.addEventListener('change', applyView);
}
document.querySelectorAll('input[name="mode"]').forEach((el) => el.addEventListener('change', applyView));

// --- Laden ------------------------------------------------------------------------

urlForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const url = urlInput.value.trim();
  if (!url) return;
  const params = new URLSearchParams(location.search);
  params.set('url', url);
  history.replaceState(null, '', `?${params}`);
  load({ url });
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) load({ file });
  fileInput.value = '';
});

flipInput.addEventListener('change', () => {
  if (!current) return;
  applyOrientation(current);
  applyView();
  frameMesh();
});

resetButton.addEventListener('click', frameMesh);

let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth++;
  document.body.classList.add('dragging');
});
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) document.body.classList.remove('dragging');
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  const file = e.dataTransfer?.files[0];
  if (file) load({ file });
});

// --- Bedienfeld ein-/ausblenden (Button oder Taste H) ------------------------------

function toggleUI() {
  document.body.classList.toggle('ui-hidden');
}
$<HTMLButtonElement>('#ui-toggle').addEventListener('click', toggleUI);
window.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement;
  if (target.matches('input[type="url"], input[type="text"], textarea')) return;
  if (e.key === 'h' || e.key === 'H') toggleUI();
});

// --- Render-Loop --------------------------------------------------------------------

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

// Nur im Dev-Server: Zugriff für Debugging in der Browser-Konsole.
if (import.meta.env.DEV) {
  Object.assign(window, {
    viewer: { THREE, camera, controls, segmentObject, robustBounds, get mesh() { return current; }, get data() { return data; }, get mask() { return mask; }, get sourceBytes() { return sourceBytes; }, exportPly },
  });
}

buildComponentsUI();
updateLabels();
const initialUrl = new URLSearchParams(location.search).get('url') ?? DEFAULT_URL;
urlInput.value = initialUrl;
load({ url: initialUrl });

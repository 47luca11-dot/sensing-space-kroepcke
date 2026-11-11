// Stadt laden und abfragen.
// Reihenfolge der Quellen (unter assets/city/, per ?city=<ordner>/ überschreibbar):
//   1. manifest.json  – mehrere GLBs (visual / collision / both), werden nach Entfernung zum Spawn geladen,
//                      optional heightfield (float32-Raster) und labels
//   2. hannover.glb   – einzelne Datei (+ optional collision.glb, city_meta.json)
//   3. Platzhalter-Testgelände (js/placeholder.js)
// Kollision: eine MeshBVH pro Kollisionsdatei (Positionen in Weltkoordinaten), Abfragen prüfen zuerst die Hüllquader.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { MeshBVH } from 'three-mesh-bvh';
import { buildPlaceholder } from './placeholder.js';
import { patchScanMaterial } from './scan.js';

const JSDELIVR = 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/libs/';
// Knoten-Namenskonventionen (Groß-/Kleinschreibung egal):
//   COL_* / *_collision  -> nur Kollision, unsichtbar
//   *nocol*              -> sichtbar, aber ohne Kollision (Laub, Kabel …)
//   *noshadow*           -> wirft keinen Schatten
const RX_COLONLY = /(^col_|_collision$|^collision)/i, RX_NOCOL = /nocol/i, RX_NOSHADOW = /noshadow/i;
const RX_GLASS = /glass|glas\b|glas_|fenster|window|scheibe/i;
export const CULL_DIST = 700, CULL_DIST_LANDMARK = 1100, SHADOW_DIST = 36, SHADOW_DIST_DETAIL = 14;
const TILE = 64;
const DETAIL_NEAR = 90, DETAIL_FAR = 150;   // Sichtweite sehr dichter / dichter Detail-Meshes                      // Kachelgröße (m) für das Zerlegen großer Meshes
const ESSENTIAL_DIST = 260;            // sichtbare Dateien näher als das werden vor dem Start geladen

async function exists(files) {
  try {
    const r = await fetch('__exists?f=' + files.map(encodeURIComponent).join(','), { cache: 'no-store' });
    if (r.ok && (r.headers.get('content-type') || '').includes('json')) return await r.json();
  } catch { /* anderer Server: direkt versuchen */ }
  return null;
}
const boxDistXZ = (b, x, z) => Math.hypot(Math.max(b.min.x - x, 0, x - b.max.x), Math.max(b.min.z - z, 0, z - b.max.z));

// PBR -> Lambert (deutlich günstiger pro Pixel auf integrierten GPUs; bei bedecktem Himmel kaum sichtbar)
const LAMBERT_KEYS = ['name', 'color', 'map', 'vertexColors', 'flatShading', 'transparent', 'opacity', 'alphaTest', 'alphaMap', 'side',
  'aoMap', 'aoMapIntensity', 'emissive', 'emissiveMap', 'emissiveIntensity', 'normalMap', 'normalMapType', 'normalScale',
  'lightMap', 'lightMapIntensity', 'fog', 'depthWrite', 'depthTest', 'polygonOffset', 'polygonOffsetFactor', 'polygonOffsetUnits', 'visible'];
function toLambert(m) {
  const l = new THREE.MeshLambertMaterial();
  for (const k of LAMBERT_KEYS) if (m[k] !== undefined) { if (m[k] && (m[k].isColor || m[k].isVector2)) l[k].copy(m[k]); else l[k] = m[k]; }
  if (m.metalness > 0.5) l.color.multiplyScalar(0.75);           // Metall ohne Spiegelung etwas dunkler
  l.userData = { ...m.userData };
  return l;
}

export class World {
  constructor(scene, renderer, camera) {
    this.scene = scene; this.renderer = renderer; this.camera = camera;
    this.root = new THREE.Group(); this.root.name = 'Stadt'; scene.add(this.root);
    this.meta = {}; this.source = 'placeholder'; this.attribution = '';
    this.colliders = [];          // { bvh, box, name }
    this.chunks = [];             // { root, box, group, meshes:[{mesh, sphere, cast}] }
    this.hf = null;               // Höhenraster
    this.triCount = 0; this.bvhMs = 0;
    this.labels = [];
    this.stream = { total: 0, loaded: 0, failed: 0, active: false };
    this.simplify = false;
    this._ray = new THREE.Ray(); this._seg = new THREE.Line3(); this._box = new THREE.Box3();
    this._tp = new THREE.Vector3(); this._cp = new THREE.Vector3(); this._d = new THREE.Vector3();
    this._matCache = new Map();
  }

  loader() {
    if (this._loader) return this._loader;
    const l = new GLTFLoader();
    const draco = new DRACOLoader().setDecoderPath(JSDELIVR + 'draco/gltf/');
    const ktx2 = new KTX2Loader().setTranscoderPath(JSDELIVR + 'basis/').detectSupport(this.renderer);
    l.setDRACOLoader(draco); l.setKTX2Loader(ktx2); l.setMeshoptDecoder(MeshoptDecoder);
    return (this._loader = l);
  }

  async load(status = () => {}, { simplify = false, base = 'assets/city/' } = {}) {
    this.simplify = simplify; this.base = base;
    const MAN = base + 'manifest.json', CITY = base + 'hannover.glb', COLL = base + 'collision.glb', META = base + 'city_meta.json';
    const ex = await exists([MAN, CITY, COLL, META]);
    const has = (f) => (ex ? !!ex[f] : true);
    let done = false;
    if (has(MAN)) {
      try {
        const r = await fetch(MAN, { cache: 'no-store' });
        if (r.ok) { await this.loadManifest(await r.json(), status); done = true; }
      } catch (e) { console.warn('manifest.json konnte nicht geladen werden:', e); }
    }
    if (!done && has(CITY)) done = await this.loadSingle(CITY, has(COLL) ? COLL : null, has(META) ? META : null, status);
    if (!done) {
      status('Erzeuge Testgelände …');
      const city = buildPlaceholder(); this.source = 'placeholder';
      this.meta = { ...city.userData.meta };
      this.prepareVisual(city);
      this.addVisual(city, 'placeholder');
      this.addColliderFrom(city, { instanced: false });
    }
    this.finish();
    return this;
  }

  // ---------------- Quelle 1: Manifest ----------------
  async loadManifest(man, status) {
    this.source = 'manifest';
    this.meta = { spawn: man.spawn, spawnYaw: man.spawnYaw, labels: man.labels, bounds: man.bounds };
    this.attribution = man.attribution || '';
    if (man.heightfield && man.heightfield.bin && man.heightfield.json) {
      status('Lade Höhenraster …');
      try {
        const [j, b] = await Promise.all([
          fetch(this.base + man.heightfield.json).then((r) => r.json()),
          fetch(this.base + man.heightfield.bin).then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }),
        ]);
        const data = new Float32Array(b);
        if (data.length >= j.w * j.h) this.hf = { data, x0: j.x0, z0: j.z0, res: j.res ?? j.dx ?? 1, w: j.w, h: j.h };
        else console.warn('heightfield: Größe passt nicht', data.length, j.w * j.h);
      } catch (e) { console.warn('heightfield:', e); }
    }
    const sp = man.spawn || [0, 0, 0];
    const files = (man.files || []).map((f) => {
      const b = f.bounds && f.bounds.length === 6 ? new THREE.Box3(new THREE.Vector3(f.bounds[0], f.bounds[1], f.bounds[2]), new THREE.Vector3(f.bounds[3], f.bounds[4], f.bounds[5])) : null;
      return { ...f, kind: f.kind || 'visual', box: b, dist: b ? boxDistXZ(b, sp[0], sp[2]) : 0 };
    }).sort((a, b) => a.dist - b.dist);
    const anyCol = files.some((f) => f.kind === 'collision' || f.kind === 'both');
    this.visualCollision = !anyCol;            // keine Kollisionsdateien -> sichtbare Dateien dienen als Kollision
    const essential = files.filter((f) => f.kind !== 'visual' || f.dist < ESSENTIAL_DIST || (!anyCol && f.dist < ESSENTIAL_DIST + 140));
    const later = files.filter((f) => !essential.includes(f));
    this.stream.total = files.length;
    let n = 0;
    await this.loadQueue(essential, 3, () => status(`Lade Stadt … ${++n}/${essential.length}`));
    if (later.length) {
      this.stream.active = true;
      this.streamDone = this.loadQueue(later, 2).then(() => { this.stream.active = false; });
    }
  }

  async loadQueue(list, concurrency, onEach = () => {}) {
    const q = list.slice();
    const worker = async () => {
      while (q.length) {
        const f = q.shift();
        try { await this.loadFile(f); this.stream.loaded++; }
        catch (e) { this.stream.failed++; console.warn('Stadtdatei fehlgeschlagen:', f.url, e); }
        onEach(f);
      }
    };
    await Promise.all(Array.from({ length: Math.min(concurrency, q.length) }, worker));
  }

  async loadFile(f) {
    const gltf = await this.loader().loadAsync(this.base + f.url);
    const root = gltf.scene; root.name = f.url;
    root.updateMatrixWorld(true);
    const vis = f.kind !== 'collision', col = f.kind !== 'visual' || this.visualCollision;
    if (col) {
      const t0 = performance.now();
      // instanzierte Meshes nur aus reinen Kollisionsdateien (sonst z. B. Bäume/Poller als Hindernisse)
      this.addColliderFrom(root, { instanced: f.kind === 'collision', name: f.url });
      this.bvhMs += performance.now() - t0;
    }
    if (vis) {
      this.prepareVisual(root);
      try { await this.renderer.compileAsync(root, this.camera, this.scene); } catch { /* optional */ }
      this.addVisual(root, f.group || 'city', f.box, f.cullDist);
      this.findSpecial(root);
    } else {
      root.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
    }
  }

  // ---------------- Quelle 2: einzelne hannover.glb ----------------
  async loadSingle(CITY, COLL, META, status) {
    let meta = {};
    if (META) { try { const r = await fetch(META, { cache: 'no-store' }); if (r.ok) meta = await r.json(); } catch (e) { console.warn('city_meta.json:', e); } }
    status('Lade Stadtmodell …');
    let city;
    try {
      city = (await this.loader().loadAsync(CITY, (e) => e.total && status(`Lade Stadtmodell … ${Math.round(e.loaded / e.total * 100)} %`))).scene;
    } catch (e) { console.warn('hannover.glb konnte nicht geladen werden:', e); return false; }
    this.source = 'glb'; this.meta = meta; this.attribution = meta.attribution || '';
    city.updateMatrixWorld(true);
    status('Berechne Kollision …');
    await new Promise((r) => setTimeout(r, 30));
    const t0 = performance.now();
    let colDone = false;
    if (COLL) {
      try { const g = await this.loader().loadAsync(COLL); g.scene.updateMatrixWorld(true); this.addColliderFrom(g.scene, { instanced: true, all: true }); colDone = true; }
      catch (e) { console.warn('collision.glb:', e); }
    }
    if (!colDone) {
      const colOnly = []; city.traverse((o) => { if (o.isMesh && RX_COLONLY.test(o.name || '')) colOnly.push(o); });
      if (colOnly.length) this.addColliderFromMeshes(colOnly, true);
      else this.addColliderFrom(city, { instanced: false });
    }
    this.bvhMs = performance.now() - t0;
    this.prepareVisual(city);
    this.addVisual(city, 'city');
    this.findSpecial(city);
    return true;
  }

  // ---------------- Materialien / Sichtbares ----------------
  roomEnv() {
    if (!this._room) {
      const pm = new THREE.PMREMGenerator(this.renderer);
      this._room = pm.fromScene(new RoomEnvironment(), 0.04).texture; pm.dispose();
    }
    return this._room;
  }
  fixMaterial(m) {
    if (!m) return m;
    if (this._matCache.has(m)) return this._matCache.get(m);
    // gleichnamige Materialien aus verschiedenen Kacheln teilen (Generatoren referenzieren dieselben Texturen):
    // spart Textur-Uploads und Shader-Varianten
    this._byName = this._byName || new Map();
    const key = m.name ? m.name + '|' + (m.vertexColors ? 'vc' : '') : '';
    if (key && this._byName.has(key)) { const shared = this._byName.get(key); this._matCache.set(m, shared); return shared; }
    let out = m;
    const glass = (m.transmission > 0) || (RX_GLASS.test(m.name || '') && (m.transparent || m.opacity < 1 || m.transmission > 0));
    if (glass) {
      // günstiges Glas: kein Transmissions-Durchgang, nur transparent + spiegelnd
      out = new THREE.MeshStandardMaterial({
        name: m.name, color: m.color ? m.color.clone().multiplyScalar(0.6) : new THREE.Color(0x26303a), transparent: true, opacity: 0.35,
        metalness: 0.1, roughness: 0.05, envMap: this.roomEnv(), envMapIntensity: 0.9, depthWrite: false, side: m.side,
      });
      out.userData.glass = true;
    } else {
      if (m.transparent && (m.map || m.alphaMap)) {     // Blattkarten & Co: Alpha-Test statt Sortierung
        m.alphaTest = Math.max(m.alphaTest || 0, 0.5); m.transparent = false; m.depthWrite = true;
      }
      if (this.simplify && m.isMeshStandardMaterial) out = toLambert(m);
    }
    if (out.map) out.map.anisotropy = Math.min(4, this.renderer.capabilities.getMaxAnisotropy());
    patchScanMaterial(out);
    this._matCache.set(m, out);
    if (key) this._byName.set(key, out);
    return out;
  }
  prepareVisual(root) {
    root.traverse((o) => {
      if (!o.isMesh) return;
      if (RX_COLONLY.test(o.name || '')) { o.visible = false; return; }
      const n = (o.name || '') + ' ' + (o.parent?.name || '');
      o.castShadow = !RX_NOSHADOW.test(n); o.receiveShadow = true;
      o.material = Array.isArray(o.material) ? o.material.map((m) => this.fixMaterial(m)) : this.fixMaterial(o.material);
      const glass = [].concat(o.material).every((m) => m.userData && m.userData.glass);
      if (glass) o.castShadow = false;
      o.userData.castShadow = o.castShadow;
    });
  }
  // Große, stadtweite Meshes in Kacheln (TILE m) zerlegen: erst dann greifen Frustum-Culling und Schatten-Culling.
  // Die Kacheln teilen sich die Vertex-Attribute des Originals (nur der Index wird aufgeteilt).
  splitLarge(root) {
    root.updateMatrixWorld(true);
    const list = [];
    root.traverse((o) => {
      if (!o.isMesh || o.isInstancedMesh || o.isSkinnedMesh || !o.visible || o.morphTargetInfluences) return;
      const g = o.geometry, tris = (g.index ? g.index.count : g.attributes.position.count) / 3;
      if (tris < 6000) return;
      if (!g.boundingSphere) g.computeBoundingSphere();
      const r = g.boundingSphere.radius * o.matrixWorld.getMaxScaleOnAxis();
      if (r > TILE * 0.6) list.push(o);
    });
    let made = 0;
    for (const o of list) made += this.splitMesh(o);
    return made;
  }
  splitMesh(mesh) {
    const g = mesh.geometry, pos = g.attributes.position, n = pos.count, idx = g.index ? g.index.array : null;
    const e = mesh.matrixWorld.elements, cx = new Float32Array(n), cz = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      cx[i] = e[0] * x + e[4] * y + e[8] * z + e[12]; cz[i] = e[2] * x + e[6] * y + e[10] * z + e[14];
    }
    const groups = g.groups.length ? g.groups : [{ start: 0, count: idx ? idx.length : n, materialIndex: 0 }];
    const cells = new Map();
    for (const gr of groups) {
      const end = Math.min(gr.start + gr.count, idx ? idx.length : n);
      for (let t = gr.start; t + 2 < end; t += 3) {
        const a = idx ? idx[t] : t, b = idx ? idx[t + 1] : t + 1, c = idx ? idx[t + 2] : t + 2;
        const key = Math.floor((cx[a] + cx[b] + cx[c]) / (3 * TILE)) * 4096 + Math.floor((cz[a] + cz[b] + cz[c]) / (3 * TILE));
        let cell = cells.get(key); if (!cell) cells.set(key, (cell = new Map()));
        let l = cell.get(gr.materialIndex || 0); if (!l) cell.set(gr.materialIndex || 0, (l = []));
        l.push(a, b, c);
      }
    }
    if (cells.size <= 1) return 0;
    const parent = mesh.parent, box = new THREE.Box3(), v = new THREE.Vector3();
    for (const [key, cell] of cells) {
      let total = 0; for (const l of cell.values()) total += l.length;
      const arr = n < 65536 ? new Uint16Array(total) : new Uint32Array(total);
      const ng = new THREE.BufferGeometry();
      for (const name in g.attributes) ng.setAttribute(name, g.attributes[name]);
      let off = 0; box.makeEmpty();
      for (const [mi, l] of cell) {
        arr.set(l, off); ng.addGroup(off, l.length, mi); off += l.length;
        for (let i = 0; i < l.length; i++) box.expandByPoint(v.fromBufferAttribute(pos, l[i]));
      }
      ng.setIndex(new THREE.BufferAttribute(arr, 1));
      if (!Array.isArray(mesh.material)) ng.clearGroups();      // ein Material: keine Gruppen nötig
      ng.boundingBox = box.clone(); ng.boundingSphere = box.getBoundingSphere(new THREE.Sphere());
      const t = new THREE.Mesh(ng, mesh.material);
      t.name = `${mesh.name}#${key}`;
      t.position.copy(mesh.position); t.quaternion.copy(mesh.quaternion); t.scale.copy(mesh.scale);
      t.castShadow = mesh.castShadow; t.receiveShadow = mesh.receiveShadow; t.renderOrder = mesh.renderOrder;
      t.userData = { ...mesh.userData, tile: true }; t.layers.mask = mesh.layers.mask;
      parent.add(t);
    }
    parent.remove(mesh);
    if (g.index) g.setIndex(null);                               // Original-Index freigeben (Attribute bleiben geteilt)
    return cells.size;
  }

  addVisual(root, group, box, cullDist) {
    this.tiles = (this.tiles || 0) + this.splitLarge(root);
    this.root.add(root);
    root.updateMatrixWorld(true);
    const meshes = [];
    root.traverse((o) => {
      o.matrixAutoUpdate = false; o.matrixWorldAutoUpdate = false;               // statisch
      if (!o.isMesh || !o.visible) return;
      if (!o.geometry.boundingSphere) o.geometry.computeBoundingSphere();
      if (o.isInstancedMesh && !o.boundingSphere) o.computeBoundingSphere();
      const s = (o.isInstancedMesh ? o.boundingSphere : o.geometry.boundingSphere).clone().applyMatrix4(o.matrixWorld);
      if (o.userData.castShadow === undefined) o.userData.castShadow = o.castShadow;
      // Detaildichte (Dreiecke pro m² Grundfläche): sehr feine Teile (Fensterrahmen, Möblierung, Kiosk) nur im Nahbereich
      const tris = ((o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count) / 3) * (o.isInstancedMesh ? o.count : 1);
      const dens = tris / Math.max(64, Math.PI * s.radius * s.radius * 0.6);
      // Gebäudehüllen (Wände/Dächer) nie ausblenden -- sonst schweben Fenster ohne Wand; Fenster-Dateien werden dateiweise (cullDist) ausgeblendet
      const mn = Array.isArray(o.material) ? '' : (o.material?.name || '');
      // Öffnungen schließende Teile (Glas, dunkle Innenflächen, Wände) dürfen nie verschwinden, sonst sieht man durch Fassaden
      const shell = /_shell/.test(o.name) || /_shell/.test(o.parent?.name || '') || /glass|glas|interior|innen|plaster|wall|wand/i.test(mn);
      const maxDist = group === 'landmark' || group === 'placeholder' || group === 'ground' || group === 'windows' || shell ? Infinity
        : group === 'buildings' ? (dens > 6 ? DETAIL_FAR : Infinity)
        : dens > 6 ? DETAIL_NEAR : dens > 1 ? DETAIL_FAR : Infinity;
      meshes.push({ mesh: o, sphere: s, maxDist, shadowDist: dens > 6 && group !== 'landmark' ? SHADOW_DIST_DETAIL : SHADOW_DIST });
    });
    const b = box ? box.clone() : new THREE.Box3().setFromObject(root);
    this.chunks.push({ root, box: b, group, meshes, cull: cullDist });
  }

  // ---------------- Kollision ----------------
  addColliderFrom(root, { instanced = false, all = false, name = '' } = {}) {
    const list = [];
    root.traverse((o) => {
      if (!o.isMesh) return;
      if (o.isInstancedMesh && !instanced) return;
      if (!all && RX_NOCOL.test((o.name || '') + ' ' + (o.parent?.name || ''))) return;
      if (!all && [].concat(o.material).every((m) => m && (m.transmission > 0 || m.userData?.glass))) return;   // Glas: begehbar? nein, aber keine Kollision nötig
      list.push(o);
    });
    if (list.length) this.addColliderFromMeshes(list, instanced, name);
  }
  // alle Dreiecke einer Datei in Weltkoordinaten in eine Geometrie (nur Positionen) -> eine BVH
  addColliderFromMeshes(meshes, instanced, name = '') {
    let nv = 0, ni = 0;
    const items = [], m4 = new THREE.Matrix4();
    for (const o of meshes) {
      const g = o.geometry; if (!g || !g.attributes.position) continue;
      const count = o.isInstancedMesh ? (instanced ? o.count : 0) : 1;
      const vc = g.attributes.position.count, ic = g.index ? g.index.count : vc;
      for (let k = 0; k < count; k++) {
        const mat = new THREE.Matrix4().copy(o.matrixWorld);
        if (o.isInstancedMesh) { o.getMatrixAt(k, m4); mat.multiply(m4); }
        items.push({ g, mat }); nv += vc; ni += ic;
      }
    }
    if (!ni) return;
    const pos = new Float32Array(nv * 3), idx = new Uint32Array(ni), v = new THREE.Vector3();
    let vo = 0, io = 0;
    for (const { g, mat } of items) {
      const p = g.attributes.position;
      for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i).applyMatrix4(mat); pos[(vo + i) * 3] = v.x; pos[(vo + i) * 3 + 1] = v.y; pos[(vo + i) * 3 + 2] = v.z; }
      if (g.index) { const a = g.index.array, s = g.drawRange.start || 0; for (let i = 0; i < g.index.count; i++) idx[io + i] = a[i] + vo; io += g.index.count; void s; }
      else { for (let i = 0; i < p.count; i++) idx[io + i] = vo + i; io += p.count; }
      vo += p.count;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    const bvh = new MeshBVH(geo, { maxLeafTris: 8 });
    geo.computeBoundingBox();
    this.colliders.push({ bvh, box: geo.boundingBox.clone().expandByScalar(0.01), name, tris: ni / 3 });
    this.triCount += ni / 3;
  }

  finish() {
    const m = this.meta;
    this.labels = (m.labels || m.places || []).map((l) => {
      const p = l.p || l.pos || l.position || [l.x ?? 0, l.z ?? 0];
      return { name: l.name || l.label || '', x: p[0], z: p.length === 3 ? p[2] : p[1], r: l.r ?? l.radius ?? 35 };
    }).filter((l) => l.name);
    const b = m.bounds;
    if (Array.isArray(b) && b.length === 4) this.bounds = b;
    else if (Array.isArray(b) && b.length === 6) this.bounds = [b[0], b[2], b[3], b[5]];
    else {
      const bb = new THREE.Box3();
      for (const c of this.colliders) bb.union(c.box);
      if (this.hf) { const h = this.hf; bb.expandByPoint(new THREE.Vector3(h.x0, 0, h.z0)).expandByPoint(new THREE.Vector3(h.x0 + (h.w - 1) * h.res, 0, h.z0 + (h.h - 1) * h.res)); }
      this.bounds = bb.isEmpty() ? null : [bb.min.x + 2, bb.min.z + 2, bb.max.x - 2, bb.max.z - 2];
    }
    let minY = Infinity;
    for (const c of this.colliders) minY = Math.min(minY, c.box.min.y);
    if (this.hf) { const d = this.hf.data; for (let i = 0; i < d.length; i += 97) if (Number.isFinite(d[i])) minY = Math.min(minY, d[i]); }
    this.minY = Number.isFinite(minY) ? minY - 30 : -100;
    // Sicherheitsboden (solange kein Höhenraster existiert): Kollision y = 0 als letzte Rückfallebene,
    // sichtbar nur außerhalb der Detailfläche (dezent, damit er nichts verdeckt)
    if (this.source !== 'placeholder' && !this.hf) {
      this.safetyY = 0;
      this.minY = Math.min(this.minY, -40);
      this.addSafetyPlane();
    }
  }

  // Sichtbarkeit ganzer Dateien nach Entfernung, Schattenwurf nur im Nahbereich (alle ~0.5 s aufrufen)
  updateStreaming(pos) {
    for (const c of this.chunks) {
      const d = boxDistXZ(c.box, pos.x, pos.z);
      const vis = c.group === 'placeholder' || d < (c.cull || (c.group === 'landmark' ? CULL_DIST_LANDMARK : CULL_DIST));
      if (c.root.visible !== vis) c.root.visible = vis;
      if (!vis) continue;
      for (const m of c.meshes) {
        const s = m.sphere, dd = Math.hypot(s.center.x - pos.x, s.center.z - pos.z) - s.radius;
        const show = dd < m.maxDist;
        if (m.mesh.visible !== show) m.mesh.visible = show;
        m.mesh.castShadow = m.mesh.userData.castShadow && dd < m.shadowDist;
      }
    }
  }

  // ---------------- Abfragen ----------------
  heightAt(x, z) {
    const h = this.hf; if (!h) return null;
    const fx = (x - h.x0) / h.res, fz = (z - h.z0) / h.res;
    if (fx < 0 || fz < 0 || fx > h.w - 1 || fz > h.h - 1) return null;
    const i = Math.min(h.w - 2, Math.floor(fx)), j = Math.min(h.h - 2, Math.floor(fz)), u = fx - i, v = fz - j, d = h.data, w = h.w;
    const a = d[j * w + i], b = d[j * w + i + 1], c = d[(j + 1) * w + i], e = d[(j + 1) * w + i + 1];
    if (!(Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c) && Number.isFinite(e))) return null;
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + e * u) * v;
  }

  // höchster Boden unter (x, yFrom, z) innerhalb maxDist: Kollisions-BVHs + Höhenraster; {y, nx, ny, nz} oder null
  groundAt(x, z, yFrom, maxDist = 60, out = {}) {
    const r = this._ray; r.origin.set(x, yFrom, z); r.direction.set(0, -1, 0);
    let best = null;
    for (const c of this.colliders) {
      const b = c.box;
      if (x < b.min.x || x > b.max.x || z < b.min.z || z > b.max.z || b.min.y > yFrom || b.max.y < yFrom - maxDist) continue;
      const hit = c.bvh.raycastFirst(r, THREE.DoubleSide, 0, maxDist);
      if (hit && (!best || hit.distance < best.distance)) best = hit;
    }
    const hy = this.heightAt(x, z);
    if (hy !== null && hy <= yFrom + 1e-4 && yFrom - hy <= maxDist && (!best || hy > best.point.y + 0.15)) {
      const e = this.hf.res, hx0 = this.heightAt(x - e, z) ?? hy, hx1 = this.heightAt(x + e, z) ?? hy, hz0 = this.heightAt(x, z - e) ?? hy, hz1 = this.heightAt(x, z + e) ?? hy;
      const nx = -(hx1 - hx0) / (2 * e), nz = -(hz1 - hz0) / (2 * e), l = Math.hypot(nx, 1, nz);
      out.y = hy; out.nx = nx / l; out.ny = 1 / l; out.nz = nz / l;
      return out;
    }
    if (!best) {
      if (this.safetyY === undefined || this.safetyY > yFrom || yFrom - this.safetyY > maxDist) return null;
      out.y = this.safetyY; out.nx = 0; out.ny = 1; out.nz = 0; return out;
    }
    const n = best.face.normal, s = n.y < 0 ? -1 : 1;
    out.y = best.point.y; out.nx = n.x * s; out.ny = n.y * s; out.nz = n.z * s;
    return out;
  }

  // Abstand bis zum ersten Treffer entlang dir (normiert) oder Infinity
  raycast(origin, dir, far) {
    const r = this._ray; r.origin.copy(origin); r.direction.copy(dir);
    let best = Infinity;
    for (const c of this.colliders) {
      if (!r.intersectsBox(c.box)) continue;
      const hit = c.bvh.raycastFirst(r, THREE.DoubleSide, 0, Math.min(far, best));
      if (hit && hit.distance < best) best = hit.distance;
    }
    return best;
  }

  // Kapsel (vertikal, von pos.y+y0 bis pos.y+y1, Radius r) horizontal aus der Geometrie schieben.
  // Unterkante liegt über der Stufenhöhe -> Bordsteine/Treppen blockieren nicht, die übernimmt der Boden-Raycast.
  collideCapsule(pos, r, y0, y1) {
    const seg = this._seg, box = this._box, tp = this._tp, cp = this._cp, d = this._d;
    let pushed = false;
    for (let iter = 0; iter < 2; iter++) {
      seg.start.set(pos.x, pos.y + y0, pos.z); seg.end.set(pos.x, pos.y + y1, pos.z);
      box.makeEmpty(); box.expandByPoint(seg.start); box.expandByPoint(seg.end);
      box.min.addScalar(-r); box.max.addScalar(r);
      let hitAny = false;
      const cb = {
        intersectsBounds: (b) => b.intersectsBox(box),
        intersectsTriangle: (tri) => {
          const dist = tri.closestPointToSegment(seg, tp, cp);
          if (dist >= r) return false;
          d.subVectors(cp, tp);
          if (d.lengthSq() < 1e-10) tri.getNormal(d);                     // Segment berührt das Dreieck
          const fy = d.y; d.y = 0;
          const len = d.length();
          if (len < 0.25 * Math.hypot(len, fy) || len < 1e-6) return false;   // Boden/Decke: nicht seitlich schieben
          // horizontale Verschiebung, die den Abstand um (r - dist) vergrößert (schräge Kontakte brauchen mehr)
          const h = Math.min((r - dist) * Math.hypot(len, fy) / len, r);
          d.multiplyScalar(h / len);
          seg.start.add(d); seg.end.add(d);
          box.min.x += d.x; box.max.x += d.x; box.min.z += d.z; box.max.z += d.z;
          hitAny = true;
          return false;
        },
      };
      for (const c of this.colliders) if (c.box.intersectsBox(box)) c.bvh.shapecast(cb);
      if (!hitAny) break;
      pushed = true;
      pos.x = seg.start.x; pos.z = seg.start.z;
    }
    return pushed;
  }

  addSafetyPlane() {
    // Loch = Vereinigung der xz-Grenzen bodennaher Dateien (Straße, Grün, Boden …)
    const hole = new THREE.Box2();
    for (const c of this.chunks) if (c.box && c.box.min.y < 0.5 && c.box.max.y < 9 && !c.box.isEmpty()) hole.expandByPoint(new THREE.Vector2(c.box.min.x, c.box.min.z)).expandByPoint(new THREE.Vector2(c.box.max.x, c.box.max.z));
    const R = 2500, shape = new THREE.Shape();
    shape.moveTo(-R, -R); shape.lineTo(R, -R); shape.lineTo(R, R); shape.lineTo(-R, R); shape.lineTo(-R, -R);
    if (!hole.isEmpty()) {
      const h = new THREE.Path();                    // Shape liegt in xy; nach rotateX(-90°) wird y zu -z
      h.moveTo(hole.min.x, -hole.min.y); h.lineTo(hole.min.x, -hole.max.y); h.lineTo(hole.max.x, -hole.max.y); h.lineTo(hole.max.x, -hole.min.y); h.lineTo(hole.min.x, -hole.min.y);
      shape.holes.push(h);
    }
    const g = new THREE.ShapeGeometry(shape); g.rotateX(-Math.PI / 2); g.translate(0, (this.safetyY || 0) - 0.02, 0);
    const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0x8b8984 }));
    patchScanMaterial(m.material);
    m.name = 'Sicherheitsboden'; m.receiveShadow = true;
    this.root.add(m); m.updateMatrixWorld(true);
    this.safetyMesh = m;
  }

  // Sonderobjekte: Uhrzeiger der Kröpcke-Uhr (zeigen echte Ortszeit), pulsierende Lichtwolke
  findSpecial(root) {
    this.hands = this.hands || []; this.pulse = this.pulse || [];
    const seenMat = new Set(this.pulse);
    root.traverse((o) => {
      const ud = { ...(o.userData || {}) };
      if (o.isMesh && o.parent) Object.assign(ud, o.parent.userData || {}, o.userData || {});
      const kind = ud.clock_hand || (/^Zeiger_hour/i.test(o.name || '') ? 'hour' : /^Zeiger_minute/i.test(o.name || '') ? 'minute' : null);
      if (kind && !this.hands.some((h) => h.node === o || h.node === o.parent && /^Zeiger_/i.test(o.parent.name || ''))) {
        const n = ud.face_normal_game;
        const axis = Array.isArray(n) && n.length === 3 ? new THREE.Vector3(n[0], n[1], n[2]).normalize() : null;
        if (axis) {
          o.traverse((c) => { c.matrixAutoUpdate = true; c.matrixWorldAutoUpdate = true; });
          const pq = new THREE.Quaternion(); if (o.parent) o.parent.getWorldQuaternion(pq);
          this.hands.push({ node: o, kind: kind === 'hour' ? 'hour' : 'minute', axis, q0: o.quaternion.clone(), pq, pqi: pq.clone().invert() });
        }
      }
      if (o.isMesh) for (const m of [].concat(o.material)) if (m && /lichtwolke_led/i.test(m.name || '') && !seenMat.has(m)) { seenMat.add(m); this.pulse.push({ mat: m, base: m.emissiveIntensity || 1 }); }
    });
  }
  updateSpecial(t) {
    if (this.hands && this.hands.length) {
      const d = new Date(), min = d.getMinutes() + d.getSeconds() / 60, hr = (d.getHours() % 12) + min / 60;
      const q = this._hq || (this._hq = new THREE.Quaternion());
      for (const h of this.hands) {
        const ang = h.kind === 'minute' ? 2 * Math.PI * (min / 60) : 2 * Math.PI * (hr / 12);
        q.setFromAxisAngle(h.axis, -ang);                                   // Weltraum: q = R(n, -a) · q0
        h.node.quaternion.copy(h.pqi).multiply(q).multiply(h.pq).multiply(h.q0);
      }
    }
    if (this.pulse) for (const p of this.pulse) p.mat.emissiveIntensity = 5.5 + 1.5 * Math.sin(t * (2 * Math.PI / 7));   // sanft 4…7
  }

  labelAt(x, z) {
    let best = null, bd = Infinity;
    for (const l of this.labels) { const d = Math.hypot(x - l.x, z - l.z); if (d < l.r && d / l.r < bd) { bd = d / l.r; best = l; } }
    return best;
  }
}

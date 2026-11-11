// Carlos: geriggte Füll-Mesh + fotoreale Gaussian Splats (Spark), gemeinsam per Skelett animiert.
// Übernommen/angepasst aus startbildschirm_demo/main.js (loadFigure, updateSplatSkin, SplatSkinning-Setup).
// Unterschied hier: die Füllung schreibt Tiefe, aber ~3 cm entlang des Sehstrahls nach hinten versetzt, damit
// Gebäude Carlos korrekt verdecken, seine eigene Füllung aber nie seine Splats verdeckt.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatMesh, SplatSkinning } from '@sparkjsdev/spark';

const BASE = 'assets/carlos/', ID = 'person1';
export const fillerPush = { value: 0.03 };
const BASE_TINT = new THREE.Color(1, 1, 1);

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));
const lerp = THREE.MathUtils.lerp;

function fillerMaterial() {
  const m = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false });
  m.color.copy(BASE_TINT);
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uPush = fillerPush;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uPush;')
      .replace('#include <project_vertex>', '#include <project_vertex>\n\tmvPosition.xyz += normalize(mvPosition.xyz) * uPush;\n\tgl_Position = projectionMatrix * mvPosition;');
  };
  m.customProgramCacheKey = () => 'carlos-filler-push';
  return m;
}

export async function loadCarlos(scene) {
  const loader = new GLTFLoader();
  const [gltf, skinBuf, splat] = await Promise.all([
    loader.loadAsync(BASE + ID + '.glb'),
    fetch(BASE + ID + '_skin.bin').then((r) => { if (!r.ok) throw new Error('skin.bin ' + r.status); return r.arrayBuffer(); }),
    (async () => { const s = new SplatMesh({ url: BASE + ID + '_splats.ply' }); await s.initialized; s.maxSh = 1; return s; })(),
  ]);
  return new Character(scene, gltf, skinBuf, splat);
}

class Character {
  constructor(scene, gltf, skinBuf, splat) {
    this.root = new THREE.Group(); this.root.name = 'Carlos';
    this.tilt = new THREE.Group(); this.root.add(this.tilt);
    this.model = gltf.scene; this.tilt.add(this.model);
    let skinned = null;
    this.model.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true; o.receiveShadow = false; o.frustumCulled = false;
      o.material = fillerMaterial();
      if (o.isSkinnedMesh) skinned = o;
    });
    this.skinned = skinned; this.meshMat = skinned.material;
    scene.add(this.root);

    // Splats als Geschwister des SkinnedMesh -> gleicher lokaler Frame wie die Mesh-Vertices
    skinned.parent.add(splat);
    splat.position.copy(skinned.position); splat.quaternion.copy(skinned.quaternion); splat.scale.copy(skinned.scale);
    const n = splat.numSplats, bones = skinned.skeleton.bones;
    const skin = new SplatSkinning({ mesh: splat, numSplats: n, numBones: bones.length });
    const raw = new Uint8Array(skinBuf);
    if (raw.length !== n * 8) console.warn('Carlos: skin/splat count mismatch', raw.length / 8, n);
    for (let i = 0; i < n; i++) {
      const o = i * 8, d = i * 4;
      for (let k = 0; k < 4; k++) skin.skinData[d + k] = raw[o + 4 + k] + (raw[o + k] << 8);
    }
    skin.skinTexture.needsUpdate = true;
    const I = new THREE.Quaternion(), Z0 = new THREE.Vector3();
    for (let b = 0; b < bones.length; b++) skin.setRestQuatPos(b, I, Z0);
    splat.skinning = skin;
    splat.updateGenerator();
    splat.recolor = BASE_TINT.clone();
    splat.layers.set(1);                       // eigener Render-Durchgang (Spark prüft die Layer der SplatMeshes)
    this.splat = splat; this.skin = skin; this.numSplats = n;

    this.bones = {};
    this.model.traverse((o) => { if (o.isBone) this.bones[o.name.replace(/\./g, '')] = o; });

    // ---------- Animation: alle Clips laufen, Zeiten und Gewichte setzen wir selbst ----------
    this.mixer = new THREE.AnimationMixer(this.model);
    const clip = (s) => gltf.animations.find((a) => a.name === `${ID}_${s}`);
    const mk = (c) => { if (!c) return null; const a = this.mixer.clipAction(c); a.setLoop(THREE.LoopRepeat, Infinity); a.play(); a.setEffectiveWeight(0); a.timeScale = 0; return a; };
    this.idle = mk(clip('Idle_Breathing') || gltf.animations[0]);
    this.walk = mk(clip('Walk')); this.run = mk(clip('Run')); this.stumbleA = mk(clip('Stumble'));
    this.fidgets = ['Pockets', 'HeadTilt_L', 'HeadTilt_R'].map(clip).filter(Boolean).map(mk);
    this.loco = { walk: this.analyze(this.walk), run: this.analyze(this.run) };
    this.phase = 0; this.idleT = 0;
    this.stumbleT = -1;
    this.fidget = null; this.fidgetT = 0; this.fidgetW = 0; this.nextFidget = 6;
    this.w = { idle: 1, walk: 0, run: 0, stumble: 0, fidget: 0 };
    this.lean = { x: 0, y: 0, arms: 0, bank: 0, pitch: 0 };
    this.onStep = null;
    this._q = new THREE.Quaternion(); this._pq = new THREE.Quaternion(); this._ax = new THREE.Vector3();
    this._fwd = new THREE.Vector3(); this._left = new THREE.Vector3(); this._tq = new THREE.Quaternion();
    this._m = new THREE.Matrix4(); this._inv = new THREE.Matrix4(); this._p = new THREE.Vector3(); this._s = new THREE.Vector3(); this._qq = new THREE.Quaternion();
  }

  // Lauf-Clip vermessen: Bodengeschwindigkeit des Standfußes (-> Strecke pro Zyklus) und Phase "linker Fuß vorn"
  analyze(action) {
    if (!action) return null;
    const clip = action.getClip(), dur = clip.duration, N = 60;
    const fl = this.bones.footL, fr = this.bones.footR, hips = this.bones.spine;
    for (const a of this.mixer._actions) a.setEffectiveWeight(0);
    action.setEffectiveWeight(1);
    const L = [], R = [], H = [];
    const v = new THREE.Vector3();
    this.root.updateMatrixWorld(true);
    const inv = new THREE.Matrix4().copy(this.model.matrixWorld).invert();
    for (let i = 0; i <= N; i++) {
      action.time = Math.min(dur - 1e-4, (i / N) * dur);
      this.mixer.update(0);
      this.model.updateMatrixWorld(true);
      L.push(v.setFromMatrixPosition(fl.matrixWorld).applyMatrix4(inv).clone());
      R.push(v.setFromMatrixPosition(fr.matrixWorld).applyMatrix4(inv).clone());
      H.push(v.setFromMatrixPosition(hips.matrixWorld).applyMatrix4(inv).clone());
    }
    const speeds = [];
    let best = -Infinity, bestI = 0, minY = Infinity;
    for (let i = 0; i <= N; i++) minY = Math.min(minY, L[i].y, R[i].y);
    for (let i = 0; i < N; i++) {
      const dt = dur / N;
      for (const F of [L, R]) {
        if (F[i].y < minY + 0.035 && F[i + 1].y < minY + 0.035) speeds.push((F[i].z - F[i + 1].z) / dt);   // Standfuß gleitet nach hinten
      }
      const d = L[i].z - R[i].z; if (d > best) { best = d; bestI = i; }
    }
    speeds.sort((a, b) => a - b);
    const med = speeds.length ? speeds[speeds.length >> 1] : 0;
    const rootMotion = H[N].z - H[0].z;
    action.setEffectiveWeight(0); action.time = 0;
    const res = { dur, speed: med, cycle: Math.max(0.3, med * dur), offset: bestI / N, rootMotion, stride: best };
    return res;
  }

  // st: { speed, mode, careful, running, stumbling, idleTime, grounded, bal:{x,y}, yawRate, longAccel, grade, stabilize }
  update(dt, st) {
    const v = st.speed;
    const W = this.loco.walk, R = this.loco.run;
    const walkRef = W ? W.speed : 1.4, runRef = R ? R.speed : 4.0;

    // --- Fortbewegungs-Mischung nach Tempo ---
    let wI, wW, wR, u = 0, t = 0;
    const vWalkFull = Math.min(walkRef, 1.25);
    if (v <= vWalkFull) { t = smooth(0.02, vWalkFull, v); wI = 1 - t; wW = t; wR = 0; }
    else { u = smooth(Math.max(vWalkFull, 1.6), Math.max(runRef * 0.92, 3.3), v); wI = 0; wW = 1 - u; wR = u; }
    if (!st.grounded) { wI *= 0.5; }

    // gemeinsame Phase (Walk/Run synchron), Schrittlänge bei langsamem Tempo kürzer
    const strideK = v < walkRef ? lerp(0.55, 1, clamp(v / walkRef, 0, 1)) * (st.careful ? 0.85 : 1) : 1;
    const fast = v > runRef * 1.15 ? Math.pow(v / (runRef * 1.15), 0.62) : 1;   // Admin-Tempo: laengere Schritte statt Zappeln
    const cycle = lerp(W ? W.cycle : 1.6, R ? R.cycle : 3.0, u) * strideK * fast;
    const prevPhase = this.phase;
    if (st.grounded) this.phase = (this.phase + (v * dt) / cycle) % 1;
    // Fußkontakte bei Phase 0 (links) und 0.5 (rechts)
    if (this.onStep && v > 0.3 && st.grounded) {
      if (prevPhase > this.phase) this.onStep('L', v);
      else if (prevPhase < 0.5 && this.phase >= 0.5) this.onStep('R', v);
    }
    if (this.walk) this.walk.time = ((this.phase + W.offset) % 1) * W.dur;
    if (this.run) this.run.time = ((this.phase + R.offset) % 1) * R.dur;
    this.idleT = (this.idleT + dt) % this.idle.getClip().duration;
    this.idle.time = this.idleT;

    // --- Stolpern (einmalig) ---
    if (st.stumbling && this.stumbleT < 0 && this.stumbleA) this.stumbleT = 0;
    let wS = 0;
    if (this.stumbleT >= 0) {
      const d = this.stumbleA.getClip().duration;
      this.stumbleT += dt;
      wS = smooth(0, 0.12, this.stumbleT) * (1 - smooth(d - 0.38, d, this.stumbleT));
      this.stumbleA.time = Math.min(this.stumbleT, d - 1e-3);
      if (this.stumbleT >= d) this.stumbleT = -1;
    }

    // --- Idle-Gesten nach längerem Stehen ---
    if (this.fidget) {
      const d = this.fidget.getClip().duration;
      this.fidgetT += dt;
      const target = st.idleTime > 0.05 && !st.stumbling ? 1 - smooth(d - 0.5, d, this.fidgetT) : 0;
      this.fidgetW = damp(this.fidgetW, target, target > this.fidgetW ? 6 : 9, dt);
      this.fidget.time = Math.min(this.fidgetT, d - 1e-3);
      if (this.fidgetT >= d || (target === 0 && this.fidgetW < 0.01)) { this.fidget.setEffectiveWeight(0); this.fidget = null; this.fidgetW = 0; this.nextFidget = 7 + Math.random() * 7; }
    } else if (st.idleTime > this.nextFidget && this.fidgets.length && !st.stumbling) {
      const pool = this.fidgets.flatMap((a) => (a.getClip().name.endsWith('Pockets') ? [a, a] : [a]));
      this.fidget = pool[Math.floor(Math.random() * pool.length)];
      this.fidgetT = 0; this.fidgetW = 0;
    }
    if (st.idleTime === 0 && !this.fidget) this.nextFidget = Math.max(this.nextFidget, 6);
    const wF = this.fidget ? this.fidgetW : 0;

    // --- Gewichte normiert setzen (Summe 1 -> keine Bindpose-Beimischung) ---
    const rest = (1 - wS);
    const w = this.w;
    w.stumble = wS; w.fidget = wF * rest;
    w.idle = wI * rest * (1 - wF); w.walk = wW * rest * (1 - wF); w.run = wR * rest * (1 - wF);
    this.idle.setEffectiveWeight(w.idle);
    if (this.walk) this.walk.setEffectiveWeight(w.walk);
    if (this.run) this.run.setEffectiveWeight(w.run);
    if (this.stumbleA) this.stumbleA.setEffectiveWeight(w.stumble);
    for (const f of this.fidgets) f.setEffectiveWeight(f === this.fidget ? w.fidget : 0);
    this.mixer.update(dt);                       // timeScale 0: wertet nur unsere gesetzten Zeiten aus

    // --- prozedurale Ebene: Gleichgewicht (Oberkörper kippt, Arme gehen raus), Kurvenlage ---
    const L = this.lean, k = 1 - wS;
    const bx = st.bal.x, by = st.bal.y, bm = Math.max(Math.abs(bx), Math.abs(by));
    L.x = damp(L.x, bx * 0.34 * k, 14, dt);
    L.y = damp(L.y, (by * 0.26 + (st.careful ? 0.07 : 0)) * k, 14, dt);
    const armTarget = (smooth(0.18, 0.85, bm) * 0.85 + (st.careful ? 0.38 : 0) + (st.stabilize ? 0.12 : 0)) * k;
    L.arms = damp(L.arms, Math.min(0.95, armTarget), 8, dt);
    L.bank = damp(L.bank, clamp(-v * st.yawRate * 0.035, -0.22, 0.22) * k * (st.grounded ? 1 : 0), 6, dt);
    L.pitch = damp(L.pitch, clamp(st.grade * 0.18 + st.longAccel * 0.012, -0.12, 0.16) * k, 5, dt);
    this.tilt.rotation.set(L.pitch, 0, L.bank);
    this.applyLean();
    this.updateSplatSkin();
  }

  // Drehung um eine Welt-Achse auf einen Knochen anwenden: q_local' = P^-1 · R · P · q_local
  rotWorld(bone, axis, angle) {
    if (!bone || Math.abs(angle) < 1e-5) return;
    bone.parent.getWorldQuaternion(this._pq);
    this._q.setFromAxisAngle(axis, angle);
    this._tq.copy(this._pq).invert().multiply(this._q).multiply(this._pq);
    bone.quaternion.premultiply(this._tq);
  }

  applyLean() {
    const b = this.bones, L = this.lean;
    this.root.updateMatrixWorld(true);
    this.tilt.getWorldQuaternion(this._qq);
    const fwd = this._fwd.set(0, 0, 1).applyQuaternion(this._qq);    // Blickrichtung
    const left = this._left.set(1, 0, 0).applyQuaternion(this._qq);  // +X lokal = links
    for (const name of ['spine001', 'spine002', 'spine003']) {
      this.rotWorld(b[name], fwd, L.x / 3);                          // + um Vorwärtsachse = nach rechts kippen
      this.rotWorld(b[name], left, L.y / 3);                         // + um Links-Achse = nach vorn
    }
    this.rotWorld(b.spine004, fwd, -L.x * 0.55);                     // Kopf hält gegen
    this.rotWorld(b.upper_armL, fwd, L.arms + Math.max(0, -L.x) * 0.5);
    this.rotWorld(b.upper_armR, fwd, -(L.arms + Math.max(0, L.x) * 0.5));
    this.rotWorld(b.forearmL, fwd, L.arms * 0.25);
    this.rotWorld(b.forearmR, fwd, -L.arms * 0.25);
    this.root.updateMatrixWorld(true);
  }

  // Knochen-Deltas für die Splats: D_k = M^-1 · Bone_k · BoneInverse_k · BindMatrix (wie three.js-Skinning)
  updateSplatSkin() {
    const sk = this.skinned.skeleton;
    this._inv.copy(this.skinned.matrixWorld).invert();
    for (let i = 0; i < sk.bones.length; i++) {
      this._m.multiplyMatrices(sk.bones[i].matrixWorld, sk.boneInverses[i]).premultiply(this._inv).multiply(this.skinned.bindMatrix);
      this._m.decompose(this._p, this._q, this._s);
      this.skin.setBoneQuatPos(i, this._q, this._p);
    }
    this.skin.updateBones();
  }

  setTransform(x, y, z, yaw) { this.root.position.set(x, y, z); this.root.rotation.set(0, yaw, 0); }
  footWorld(side, out) { return (side === 'L' ? this.bones.footL : this.bones.footR).getWorldPosition(out); }
}

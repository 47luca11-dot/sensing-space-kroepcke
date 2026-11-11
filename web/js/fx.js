// Kleine Effekte: Staubwölkchen bei Schritten/Landungen und synthetische Geräusche (WebAudio, keine Dateien).
import * as THREE from 'three';

export class Dust {
  constructor(scene, max = 160) {
    this.max = max; this.i = 0;
    this.p = new Float32Array(max * 3); this.v = new Float32Array(max * 3);
    this.life = new Float32Array(max); this.dur = new Float32Array(max).fill(1); this.size = new Float32Array(max);
    const g = new THREE.BufferGeometry();
    this.aPos = new THREE.BufferAttribute(this.p, 3).setUsage(THREE.DynamicDrawUsage);
    this.aA = new THREE.BufferAttribute(new Float32Array(max), 1).setUsage(THREE.DynamicDrawUsage);
    this.aS = new THREE.BufferAttribute(new Float32Array(max), 1).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this.aPos); g.setAttribute('alpha', this.aA); g.setAttribute('psize', this.aS);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
    this.uniforms = { uScale: { value: 500 }, uColor: { value: new THREE.Color(0.62, 0.58, 0.52) } };
    const m = new THREE.ShaderMaterial({
      uniforms: this.uniforms, transparent: true, depthWrite: false,
      vertexShader: `attribute float alpha; attribute float psize; varying float vA; uniform float uScale;
        void main(){ vA = alpha; vec4 mv = modelViewMatrix * vec4(position,1.); gl_PointSize = psize * uScale / -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `varying float vA; uniform vec3 uColor;
        void main(){ vec2 c = gl_PointCoord - .5; float d = dot(c,c)*4.; if (d > 1.) discard; gl_FragColor = vec4(uColor, vA * (1.-d) * (1.-d)); }`,
    });
    this.points = new THREE.Points(g, m); this.points.frustumCulled = false; this.points.renderOrder = 5;
    scene.add(this.points);
    this.rand = Math.random;
  }
  emit(pos, n, strength = 1, dirX = 0, dirZ = 0) {
    for (let k = 0; k < n; k++) {
      const i = this.i; this.i = (this.i + 1) % this.max;
      const a = this.rand() * Math.PI * 2, sp = (0.25 + this.rand() * 0.45) * strength;
      this.p[i * 3] = pos.x + Math.cos(a) * 0.08; this.p[i * 3 + 1] = pos.y + 0.03; this.p[i * 3 + 2] = pos.z + Math.sin(a) * 0.08;
      this.v[i * 3] = Math.cos(a) * sp + dirX * 0.3; this.v[i * 3 + 1] = 0.15 + this.rand() * 0.35 * strength; this.v[i * 3 + 2] = Math.sin(a) * sp + dirZ * 0.3;
      this.life[i] = 0; this.dur[i] = 0.7 + this.rand() * 0.7; this.size[i] = (0.1 + this.rand() * 0.12) * (0.7 + strength * 0.4);
    }
  }
  update(dt, renderer, camera) {
    this.uniforms.uScale.value = renderer.domElement.height * 0.5 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const A = this.aA.array, S = this.aS.array;
    for (let i = 0; i < this.max; i++) {
      if (this.life[i] >= this.dur[i]) { A[i] = 0; continue; }
      this.life[i] += dt;
      const u = this.life[i] / this.dur[i], drag = Math.exp(-2.5 * dt);
      this.v[i * 3] *= drag; this.v[i * 3 + 1] = this.v[i * 3 + 1] * drag - 0.05 * dt; this.v[i * 3 + 2] *= drag;
      this.p[i * 3] += this.v[i * 3] * dt; this.p[i * 3 + 1] += this.v[i * 3 + 1] * dt; this.p[i * 3 + 2] += this.v[i * 3 + 2] * dt;
      A[i] = 0.32 * Math.min(1, u * 8) * (1 - u);
      S[i] = this.size[i] * (1 + u * 2.2);
    }
    this.aPos.needsUpdate = true; this.aA.needsUpdate = true; this.aS.needsUpdate = true;
  }
}

export class Sfx {
  constructor() { this.ctx = null; this.vol = 0.5; }
  init() {
    if (this.ctx) { if (this.ctx.state === 'suspended') this.ctx.resume(); return; }
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.out = this.ctx.createGain(); this.out.gain.value = this.vol; this.out.connect(this.ctx.destination);
      const len = this.ctx.sampleRate * 0.5, b = this.ctx.createBuffer(1, len, this.ctx.sampleRate), d = b.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      this.noise = b;
      // leiser Wind
      const wind = this.ctx.createBufferSource(); wind.buffer = b; wind.loop = true;
      const wf = this.ctx.createBiquadFilter(); wf.type = 'lowpass'; wf.frequency.value = 380;
      this.windGain = this.ctx.createGain(); this.windGain.gain.value = 0.05;
      wind.connect(wf).connect(this.windGain).connect(this.out); wind.start();
    } catch (e) { console.warn('Audio nicht verfügbar', e); this.ctx = null; }
  }
  step(speed, careful) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t = this.ctx.currentTime, s = this.ctx.createBufferSource(); s.buffer = this.noise;
    s.playbackRate.value = 0.8 + Math.random() * 0.4;
    const f = this.ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 700 + speed * 260 + Math.random() * 200; f.Q.value = 0.7;
    const g = this.ctx.createGain(), v = (careful ? 0.12 : 0.22) + Math.min(0.4, speed * 0.07);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(v, t + 0.008); g.gain.exponentialRampToValueAtTime(0.001, t + 0.13 + speed * 0.01);
    s.connect(f).connect(g).connect(this.out); s.start(t, Math.random() * 0.3); s.stop(t + 0.2);
  }
  thud(strength = 1) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t = this.ctx.currentTime, o = this.ctx.createOscillator(), g = this.ctx.createGain();
    o.type = 'sine'; o.frequency.setValueAtTime(120, t); o.frequency.exponentialRampToValueAtTime(45, t + 0.18);
    g.gain.setValueAtTime(0.5 * Math.min(1, strength), t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
    o.connect(g).connect(this.out); o.start(t); o.stop(t + 0.3);
    this.step(3, false);
  }
  scan() {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t = this.ctx.currentTime;
    for (const [f0, f1, d, v] of [[520, 1400, 0.9, 0.12], [260, 700, 1.2, 0.08]]) {
      const o = this.ctx.createOscillator(), g = this.ctx.createGain();
      o.type = 'sine'; o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f1, t + d * 0.6);
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(v, t + 0.03); g.gain.exponentialRampToValueAtTime(0.001, t + d);
      o.connect(g).connect(this.out); o.start(t); o.stop(t + d + 0.05);
    }
  }
  setWind(speed) { if (this.windGain) this.windGain.gain.value = 0.04 + Math.min(0.05, speed * 0.008); }
}

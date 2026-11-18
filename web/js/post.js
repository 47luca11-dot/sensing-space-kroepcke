// Nachbearbeitung für den Punktwolken-Look (Referenz: "Interitum – Surreal VR Point Cloud Game"):
// weiches Glühen der Punkte (Bloom aus zwei Unschärfe-Stufen), digitales Rauschen (Korn, aufblitzende Pixel,
// gelegentliche Störstreifen, feine Zeilen), senkrechte Lichtschlieren, Vignette.
// Ablauf: begin() -> alle Durchgänge rendern in rtMain (linear, HalfFloat) -> end() setzt das Bild auf den Canvas.
import * as THREE from 'three';

const VERT = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }';

const BLUR = /* glsl */`
varying vec2 vUv; uniform sampler2D tSrc; uniform vec2 uDir; uniform float uGain;
void main(){
  vec3 c = texture2D(tSrc, vUv).rgb * 0.2270;
  c += (texture2D(tSrc, vUv + uDir * 1.3846).rgb + texture2D(tSrc, vUv - uDir * 1.3846).rgb) * 0.3162;
  c += (texture2D(tSrc, vUv + uDir * 3.2308).rgb + texture2D(tSrc, vUv - uDir * 3.2308).rgb) * 0.0703;
  gl_FragColor = vec4(c * uGain, 1.0);
}`;

const COMPOSITE = /* glsl */`
varying vec2 vUv;
uniform sampler2D tBase, tGlow1, tGlow2;
uniform vec2 uRes; uniform float uTime, uGlow, uNoise, uStreak;
float hash(vec2 p){ vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
void main(){
  vec2 uv = vUv;
  float tf = floor(uTime * 14.0);
  // Störstreifen: selten einzelne waagerechte Bänder kurz versetzen
  float band = floor(uv.y * 70.0);
  float glitch = step(0.992, hash(vec2(band, tf))) * uNoise;
  uv.x += glitch * (hash(vec2(band + 3.0, tf)) - 0.5) * 0.02;
  vec3 c = texture2D(tBase, uv).rgb;
  if (glitch > 0.0) c = vec3(texture2D(tBase, uv + vec2(0.003, 0.0)).r, c.g, texture2D(tBase, uv - vec2(0.003, 0.0)).b);
  // Glühen
  vec3 g1 = texture2D(tGlow1, vUv).rgb, g2 = texture2D(tGlow2, vUv).rgb;
  c += (g1 * 0.55 + g2 * 0.75) * uGlow;
  // senkrechte Lichtschlieren (wie im Video: helle Punkte ziehen dünne Fäden nach unten)
  if (uStreak > 0.0) {
    float s = 0.0;
    for (int k = 1; k <= 6; k++) s += luma(texture2D(tGlow1, vUv + vec2(0.0, float(k) * 0.012)).rgb) * (1.0 - float(k) / 7.0);
    float col = hash(vec2(floor(vUv.x * uRes.x / 2.0), 5.0));
    c += vec3(0.85, 0.9, 1.0) * s * 0.10 * uStreak * smoothstep(0.55, 1.0, col);
  }
  // digitales Rauschen: Korn (stärker in hellen Bereichen), aufblitzende Pixel, feine Zeilen
  vec2 px = floor(vUv * uRes);
  float n = hash(px + fract(uTime * 37.0) * vec2(113.0, 71.0)) - 0.5;
  c += n * uNoise * (0.025 + 0.22 * luma(c));
  float spark = step(0.99935, hash(px * 0.5 + vec2(tf * 17.31, tf * 11.73)));
  c += spark * vec3(0.5) * uNoise;
  c *= 1.0 - 0.035 * uNoise * step(0.5, fract(px.y * 0.5));
  // Vignette
  float v = length(vUv - 0.5) * 1.35;
  c *= mix(1.0, smoothstep(1.15, 0.35, v), 0.55);
  gl_FragColor = vec4(max(c, 0.0), 1.0);
}`;

export class PostFX {
  constructor(renderer) {
    this.r = renderer;
    const opt = { type: THREE.HalfFloatType, depthBuffer: true, generateMipmaps: false };
    this.rtMain = new THREE.WebGLRenderTarget(4, 4, opt);
    this.rtMain.texture.colorSpace = THREE.SRGBColorSpace;   // wie der Canvas: Szene landet fertig in sRGB, Effekte arbeiten darauf
    const small = { type: THREE.HalfFloatType, depthBuffer: false, generateMipmaps: false };
    this.rtA = new THREE.WebGLRenderTarget(2, 2, small); this.rtB = new THREE.WebGLRenderTarget(2, 2, small);
    this.rtC = new THREE.WebGLRenderTarget(1, 1, small); this.rtD = new THREE.WebGLRenderTarget(1, 1, small);
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.blur = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: BLUR, depthTest: false, depthWrite: false, toneMapped: false,
      uniforms: { tSrc: { value: null }, uDir: { value: new THREE.Vector2() }, uGain: { value: 1 } } });
    this.comp = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: COMPOSITE, depthTest: false, depthWrite: false, toneMapped: false,
      uniforms: { tBase: { value: this.rtMain.texture }, tGlow1: { value: this.rtB.texture }, tGlow2: { value: this.rtD.texture },
        uRes: { value: new THREE.Vector2(1, 1) }, uTime: { value: 0 }, uGlow: { value: 1 }, uNoise: { value: 1 }, uStreak: { value: 1 } } });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.blur); this.quad.frustumCulled = false;
    this.enabled = true;
  }
  setSize(w, h) {
    this.rtMain.setSize(w, h);
    const hw = Math.max(1, w >> 1), hh = Math.max(1, h >> 1), qw = Math.max(1, w >> 2), qh = Math.max(1, h >> 2);
    this.rtA.setSize(hw, hh); this.rtB.setSize(hw, hh); this.rtC.setSize(qw, qh); this.rtD.setSize(qw, qh);
    this.comp.uniforms.uRes.value.set(w, h);
  }
  begin() {
    const r = this.r, s = r.getDrawingBufferSize(new THREE.Vector2());
    if (s.x !== this.rtMain.width || s.y !== this.rtMain.height) this.setSize(s.x, s.y);
    r.setRenderTarget(this.rtMain); r.clear();
  }
  _pass(mat, target) { this.quad.material = mat; this.r.setRenderTarget(target); this.r.render(this.quad, this.cam); }
  end(time) {
    const b = this.blur.uniforms;
    b.tSrc.value = this.rtMain.texture; b.uDir.value.set(1 / this.rtA.width, 0); b.uGain.value = 1; this._pass(this.blur, this.rtA);
    b.tSrc.value = this.rtA.texture; b.uDir.value.set(0, 1 / this.rtA.height); this._pass(this.blur, this.rtB);
    b.tSrc.value = this.rtB.texture; b.uDir.value.set(1 / this.rtC.width, 0); b.uGain.value = 1.15; this._pass(this.blur, this.rtC);
    b.tSrc.value = this.rtC.texture; b.uDir.value.set(0, 1 / this.rtC.height); this._pass(this.blur, this.rtD);
    this.comp.uniforms.uTime.value = time;
    this._pass(this.comp, null);
  }
}

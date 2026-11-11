// "Odradek"-Scan: eine Ringwelle läuft über das Gelände und färbt es kurz nach Begehbarkeit
// (blau = eben, gelb = uneben/schräg, rot = steil). Umsetzung als onBeforeCompile-Patch der Stadtmaterialien
// (Weltposition aus der View-Position, Neigung aus Ableitungen) -> kein zusätzlicher Render-Durchgang.
import * as THREE from 'three';

export const scanUniforms = {
  uScanOrigin: { value: new THREE.Vector3() },
  uScanRadius: { value: 0 },
  uScanFade: { value: 0 },
  uScanMax: { value: 75 },
  uScanInvView: { value: new THREE.Matrix4() },
};

const LIT = (m) => m.isMeshStandardMaterial || m.isMeshLambertMaterial || m.isMeshPhongMaterial || m.isMeshToonMaterial;
const patched = new WeakSet();

const VERT_HEAD = /* glsl */`
uniform mat4 uScanInvView;
varying vec3 vScanW;
#ifdef SCAN_SMOOTH_N
varying vec3 vScanN;
#endif
`;
const VERT_BODY = /* glsl */`
vScanW = (uScanInvView * mvPosition).xyz;
#ifdef SCAN_SMOOTH_N
vScanN = mat3(uScanInvView) * transformedNormal;
#endif
`;
const FRAG_HEAD = /* glsl */`
uniform vec3 uScanOrigin;
uniform float uScanRadius, uScanFade, uScanMax;
varying vec3 vScanW;
#ifdef SCAN_SMOOTH_N
varying vec3 vScanN;
#endif
`;
const FRAG_BODY = /* glsl */`
if (uScanFade > 0.001) {
  vec3 sdx = dFdx(vScanW), sdy = dFdy(vScanW);
  vec3 sN = normalize(cross(sdx, sdy));
  float ny = abs(sN.y);
  float sd = length(vScanW.xz - uScanOrigin.xz);
  float inside = 1.0 - smoothstep(uScanRadius - 1.5, uScanRadius, sd);
  float reach = 1.0 - smoothstep(uScanMax * 0.8, uScanMax, sd);
  float slope = degrees(acos(clamp(ny, 0.0, 1.0)));
  float rough = 0.0;
#ifdef SCAN_SMOOTH_N
  float px = max(length(fwidth(vScanW)), 1e-4);
  rough = smoothstep(0.35, 1.4, length(fwidth(normalize(vScanN))) / px);   // Normalenänderung pro Meter
#endif
  vec3 cBlue = vec3(0.10, 0.45, 1.25), cYel = vec3(1.30, 0.95, 0.12), cRed = vec3(1.45, 0.12, 0.08);
  float k = max(smoothstep(9.0, 15.0, slope), rough * 0.9);
  vec3 sc = mix(cBlue, cYel, k);
  sc = mix(sc, cRed, smoothstep(26.0, 33.0, slope));
  float ground = smoothstep(0.25, 0.4, ny);                       // Wände nicht einfärben, nur Gelände
  vec2 g = abs(fract(vScanW.xz * 0.5) - 0.5) / max(fwidth(vScanW.xz * 0.5), vec2(1e-4));
  float grid = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
  float a = inside * reach * uScanFade * ground * (0.26 + 0.5 * grid);
  gl_FragColor.rgb = mix(gl_FragColor.rgb, sc * (0.55 + 0.45 * grid), a);
  float ring = exp(-pow((sd - uScanRadius) / 0.7, 2.0)) * reach * smoothstep(0.0, 0.25, uScanFade);
  gl_FragColor.rgb += vec3(0.35, 0.85, 1.4) * ring * (uScanRadius < uScanMax ? 1.0 : 0.0);
}
`;

export function patchScanMaterial(mat) {
  if (!mat || patched.has(mat) || mat.isShaderMaterial || mat.isPointsMaterial || mat.isLineBasicMaterial || mat.isSpriteMaterial) return;
  patched.add(mat);
  const prev = mat.onBeforeCompile, prevKey = mat.customProgramCacheKey;
  const smooth = LIT(mat);
  mat.onBeforeCompile = function (sh, r) {
    if (prev) prev.call(this, sh, r);
    if (!sh.fragmentShader.includes('#include <opaque_fragment>') || !sh.vertexShader.includes('#include <project_vertex>')) return;
    Object.assign(sh.uniforms, scanUniforms);
    const def = smooth ? '#define SCAN_SMOOTH_N\n' : '';
    sh.vertexShader = def + sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + VERT_HEAD)
      .replace('#include <project_vertex>', '#include <project_vertex>\n' + VERT_BODY);
    sh.fragmentShader = def + sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + FRAG_HEAD)
      .replace('#include <opaque_fragment>', '#include <opaque_fragment>\n' + FRAG_BODY);
  };
  mat.customProgramCacheKey = function () { return (prevKey ? prevKey.call(this) : '') + '|scan' + (smooth ? 'S' : ''); };
  mat.needsUpdate = true;
}

export class Scan {
  constructor() { this.t = -1; this.speed = 38; this.hold = 3.2; this.fadeTime = 1.6; }
  get active() { return this.t >= 0; }
  trigger(origin) {
    scanUniforms.uScanOrigin.value.copy(origin);
    this.t = 0;
  }
  update(dt, camera) {
    scanUniforms.uScanInvView.value.copy(camera.matrixWorld);
    if (this.t < 0) { scanUniforms.uScanFade.value = 0; return; }
    this.t += dt;
    const max = scanUniforms.uScanMax.value, travel = max / this.speed;
    scanUniforms.uScanRadius.value = Math.min(this.t * this.speed, max + 2);
    const end = travel + this.hold;
    scanUniforms.uScanFade.value = this.t < end ? 1 : Math.max(0, 1 - (this.t - end) / this.fadeTime);
    if (this.t > end + this.fadeTime) this.t = -1;
  }
}

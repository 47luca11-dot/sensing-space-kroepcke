import * as THREE from 'three';
import { ROOM } from '../config/room';

/**
 * Sichtbeton aus dem Scan, zusätzlich im Shader verschmutzt und gegen sichtbare Wiederholung gemischt:
 *  - Anti-Tiling: zwei waagerecht versetzte Abtastungen der Kachel, per Weltrauschen überblendet
 *    (nur waagerecht, damit die Schalungsfugen in einer Linie bleiben)
 *  - Schmutzwolken, Laufspuren von oben, Dreck am Boden, Ruß unter der Decke, alles aus der Weltposition
 *  - feuchte Stellen glänzen etwas mehr (geringere Rauheit)
 */
export function grungeWallMaterial() {
  const loader = new THREE.TextureLoader();
  const url = (path: string) => import.meta.env.BASE_URL + path.replace(/^\//, '');
  const { color, normal, roughness } = ROOM.wallTexture;
  const map = loader.load(url(color));
  map.colorSpace = THREE.SRGBColorSpace;
  const normalMap = loader.load(url(normal));
  const roughnessMap = loader.load(url(roughness));
  for (const t of [map, normalMap, roughnessMap]) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
  }

  const material = new THREE.MeshStandardMaterial({
    map,
    normalMap,
    roughnessMap,
    normalScale: new THREE.Vector2(0.9, 0.9),
    roughness: 1,
  });

  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWallPos;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvWallPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');

    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${GLSL_HELPERS}`)
      .replace('#include <map_fragment>', MAP_FRAGMENT.replaceAll('WALL_HEIGHT', ROOM.height.toFixed(3)))
      .replace('texture2D( normalMap, vNormalMapUv )', 'wallSample( normalMap, vNormalMapUv )')
      .replace('texture2D( roughnessMap, vRoughnessMapUv )', 'wallSample( roughnessMap, vRoughnessMapUv )')
      .replace(
        '#include <roughnessmap_fragment>',
        '#include <roughnessmap_fragment>\nroughnessFactor = clamp(roughnessFactor - wallWet * 0.3, 0.35, 1.0);',
      );
  };
  // Eigener Shader-Cache-Schlüssel, damit andere Standard-Materialien unverändert bleiben
  material.customProgramCacheKey = () => 'grunge-wall';
  return material;
}

const GLSL_HELPERS = /* glsl */ `
varying vec3 vWallPos;
float wallBlend = 0.0;
float wallWet = 0.0;

float wHash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float wNoise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(wHash(i), wHash(i + vec3(1, 0, 0)), f.x), mix(wHash(i + vec3(0, 1, 0)), wHash(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(wHash(i + vec3(0, 0, 1)), wHash(i + vec3(1, 0, 1)), f.x), mix(wHash(i + vec3(0, 1, 1)), wHash(i + vec3(1, 1, 1)), f.x), f.y),
    f.z);
}
float wFbm(vec3 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { s += a * wNoise(p); p *= 2.03; a *= 0.5; }
  return s / 0.9375;
}
// Zwei waagerecht versetzte Abtastungen der Kachel, überblendet nach Weltrauschen
vec4 wallSample(sampler2D tex, vec2 uv) {
  vec4 a = texture2D(tex, uv);
  if (wallBlend <= 0.001) return a;
  vec4 b = texture2D(tex, uv + vec2(0.4173, 0.0));
  return mix(a, b, wallBlend);
}
`;

const MAP_FRAGMENT = /* glsl */ `
vec3 wp = vWallPos;
wallBlend = smoothstep(0.42, 0.58, wFbm(wp * 0.45 + 3.1));
#ifdef USE_MAP
  diffuseColor *= wallSample(map, vMapUv);
#endif
{
  float along = wp.x + wp.z;             // waagerechte Richtung, egal welche Wand
  float blotch = wFbm(wp * vec3(0.8, 0.6, 0.8) + 11.0);
  float fine = wFbm(wp * 7.0);
  // Dreck und Spritzwasser am Boden
  float floorDirt = 1.0 - smoothstep(0.0, 0.45 + 0.35 * fine, wp.y);
  // Ruß und Wasserränder unter der Decke
  float soot = smoothstep(WALL_HEIGHT - 0.8, WALL_HEIGHT, wp.y) * (0.4 + 0.6 * blotch);
  // Laufspuren von oben: schmale, lange Rauschbänder
  float streak = smoothstep(0.58, 0.82, wFbm(vec3(along * 6.0, wp.y * 0.3, 1.7)))
               * smoothstep(0.0, 1.2, WALL_HEIGHT - wp.y + fine * 0.6);
  vec3 g = vec3(1.0);
  g *= mix(1.0, 0.68, smoothstep(0.5, 0.8, blotch) * (0.5 + 0.7 * fine));
  g *= mix(vec3(1.0), vec3(0.58, 0.52, 0.44), floorDirt * 0.85);
  g *= mix(1.0, 0.62, soot);
  g *= 1.0 - streak * 0.28;
  diffuseColor.rgb *= g;
  wallWet = floorDirt * 0.5 + streak * 0.6;
}
`;

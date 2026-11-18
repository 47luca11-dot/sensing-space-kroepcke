// Punktwolken-Look: die ganze Stadt als Punktwolke (erzeugt in Blender: pipeline/pointcloud/bl_sample.py + bl_tile.py).
// - Kacheln à 32 m, je Kachel zufällig gemischt -> der Rang eines Punkts (Index / Anzahl) ist eine gleichmäßige Stichprobe.
// - Dichte nach Entfernung zum Spieler: im Radius NEAR alle Punkte, danach fällt der sichtbare Anteil f(d) ab;
//   ein Punkt wird gezeichnet, wenn rang < f(d). Pro Kachel wird nur das nötige Präfix gezeichnet (drawRange),
//   die feine Stufe (_b) wird nur in der Nähe geladen.
// - F-Scan: vom Scan-Ursprung wird eine Distanz-Cubemap der (unsichtbaren) Stadt-Meshes gerendert. Alles, was von dort
//   aus frei sichtbar ist, wird von der Scanwelle erfasst und für SCAN_HOLD Sekunden in voller Dichte "erinnert".
import * as THREE from 'three';

// Sichtweite wie ein Scanner mit kurzer Reichweite: bis NEAR volle Dichte, bei END ist alles schwarz.
// Weiter sieht man nur, was man mit F gescannt hat (10 s Gedächtnis).
export const PC = {
  near: 6.5,         // m: bis hier volle Dichte
  pow: 1.0,          // (ungenutzt bei harter Sichtgrenze, bleibt für ?far-Tests)
  floor: 0.0,        // kein Mindestanteil: dahinter schwarz
  far: 6.5, end: 10, // Sichtgrenze: ab far ausdünnen, bei end nichts mehr
  sizeK: 0.5,       // Punktgröße relativ zum mittleren Punktabstand (klein, scan-artig)
  maxPx: 2.0, 
  dup: 4,            // Nahbereich so oft (versetzt) zeichnen
        // Obergrenze Punktgröße in Pixeln
  fineDist: 26,      // feine Stufe (_b) laden bis hier (m, Kachelrand)
  coarseDist: 170,   // grobe Stufe (_a) laden bis hier (für den Scan bis 150 m)
};
export const SCAN_HOLD = 10, SCAN_SPEED = 42, SCAN_MAX = 150;

const VERT = /* glsl */`
attribute vec3 pos; attribute vec4 col; attribute vec2 oct;
uniform vec3 uOrigin, uQuant, uPlayer, uCam;
uniform float uOff, uN, uSpacing, uPx;
uniform float uNear, uPow, uFloor, uFar, uEnd, uSizeK, uCull, uBaked, uMaxPx, uExpo, uSat, uGamma, uNoiseLo, uNoiseAmp, uDup;
uniform vec3 uScanO; uniform float uScanR, uScanAge, uScanHold, uScanMax, uScanOn;
uniform samplerCube uScanCube;
varying vec3 vCol; varying float vA; varying float vRing; varying float vSize;
vec3 octDec(vec2 e){ e = e * 2.0 - 1.0; vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y)); if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * sign(n.xy); return normalize(n); }
float frac(float d){
  float f = d <= uNear ? 1.0 : pow(uNear / d, uPow);
  f = max(f, uFloor);
  return f * (1.0 - smoothstep(uFar, uEnd, d));
}
void main(){
  vec3 wp = uOrigin + pos * uQuant;
  // Spielkoordinaten: x,z relativ zur Kachel, y absolut
  float flags = col.a * 255.0;
  bool two = mod(flags, 2.0) >= 1.0;
  bool emis = mod(floor(flags / 2.0), 2.0) >= 1.0;
  bool glass = mod(floor(flags / 4.0), 2.0) >= 1.0;
  bool fine = mod(floor(flags / 8.0), 2.0) >= 1.0;
  vec3 n = octDec(oct);
  // Verdichtung im Nahbereich: Durchgang uDup>0 zeichnet jeden Punkt versetzt in seiner Tangentialebene nochmal
  if (uDup > 0.5) {
    float hd = fract(sin((uOff + float(gl_VertexID)) * 78.233 + uDup * 19.19) * 43758.5453);
    float hd2 = fract(hd * 57.13 + uDup * 0.37);
    vec3 t1 = normalize(abs(n.y) < 0.9 ? cross(n, vec3(0.0, 1.0, 0.0)) : cross(n, vec3(1.0, 0.0, 0.0)));
    vec3 t2 = cross(n, t1);
    wp += (t1 * (hd - 0.5) + t2 * (hd2 - 0.5)) * uSpacing * 0.95;
    if (length(wp.xz - uPlayer.xz) > uEnd + 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  }
  vec3 V = uCam - wp; float dc = length(V); V /= max(dc, 1e-4);
  if (two && dot(n, V) < 0.0) n = -n;
  float rank = (uOff + float(gl_VertexID) + 0.5) / uN;
  float d = length(wp.xz - uPlayer.xz) + 0.35 * abs(wp.y - uPlayer.y - 1.0);
  float f = frac(d);
  // ---- Scan-Gedächtnis
  float scanK = 0.0; vRing = 0.0;
  if (uScanOn > 0.5) {
    vec3 sd = wp - uScanO; float ds = length(sd);
    float vis = 0.0;
    if (ds < uScanMax) {
      float occ = texture(uScanCube, sd / max(ds, 1e-4)).r;
      vis = ds <= occ + 0.25 + ds * 0.012 ? 1.0 : 0.0;
    }
    float reached = 1.0 - smoothstep(uScanR - 1.0, uScanR, ds);
    float life = 1.0 - smoothstep(uScanHold - 2.5, uScanHold, uScanAge);
    scanK = vis * reached * life * (1.0 - smoothstep(uScanMax * 0.85, uScanMax, ds));
    vRing = vis * exp(-pow((ds - uScanR) / 1.1, 2.0)) * (uScanR < uScanMax ? 1.0 : 0.0);
    f = max(f, scanK);
  }
  bool back = !two && dot(n, V) < -0.05;
  if (rank > f || (back && uCull > 0.5 && scanK < 0.5)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  // ---- Farbe: in Blender gebackenes Licht (Sonne + Schlagschatten + AO + Himmelssicht) steckt schon in col.rgb
  vec3 c = pow(col.rgb, vec3(2.2));
  if (uBaked < 0.5) {
    if (!emis) {
      vec3 L = normalize(vec3(-0.45, 0.8, 0.38));
      c *= (0.66 + 0.40 * max(dot(n, L), 0.0) + 0.14 * (n.y * 0.5 + 0.5)) * 1.15;
    } else c *= 1.6;
  } else if (emis) c *= 1.4;
  c *= uExpo;
  if (glass) c = mix(c, vec3(0.20, 0.26, 0.30), 0.5);
  // Sensorrauschen wie bei einem echten Scan: Helligkeit/Farbe pro Punkt leicht streuen
  float h = fract(sin((uOff + float(gl_VertexID) + uDup * 7.31) * 12.9898 + uOrigin.x * 0.137 + uOrigin.z * 0.071) * 43758.5453);
  float h2 = fract(h * 91.7 + 0.31);
  c *= uNoiseLo + uNoiseAmp * h * h * 1.6;
  c *= vec3(1.0 + 0.06 * (h2 - 0.5), 1.0, 1.0 - 0.06 * (h2 - 0.5));
  // Scan: begehbarer Boden in Blau, schräg Gelb, steil Rot (wie der alte Gelände-Scan), Rest heller
  if (scanK > 0.0) {
    float life = scanK;
    float slope = degrees(acos(clamp(n.y, 0.0, 1.0)));
    vec3 sc = mix(vec3(0.15, 0.55, 1.3), vec3(1.3, 0.95, 0.15), smoothstep(9.0, 16.0, slope));
    sc = mix(sc, vec3(1.4, 0.15, 0.1), smoothstep(26.0, 34.0, slope));
    float ground = smoothstep(0.55, 0.75, n.y);
    float fresh = 1.0 - smoothstep(0.0, 3.5, uScanAge - length(wp - uScanO) / ${SCAN_SPEED.toFixed(1)});
    c = mix(c, sc * 0.45, ground * life * 0.28);
    c += vec3(0.25, 0.6, 0.9) * fresh * life * 0.25;
  }
  // Licht fällt mit der Entfernung ab (Look: schwarzer Raum, nur das Nahe ist hell); Gescanntes bleibt etwas heller
  float fall = mix(1.0, 0.6, smoothstep(2.0, uEnd, d));
  if (scanK > 0.0) fall = max(fall, mix(0.85, 0.35, smoothstep(10.0, uScanMax, length(wp - uScanO))) * scanK);
  c *= fall;
  // Tonkurve wie bei der Referenz (Scan-Foto): Sättigung runter, Kontrast hoch, weiche Schulter, kein ACES
  float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(vec3(lum), c, uSat);
  c = pow(max(c, 0.0), vec3(uGamma));
  c = c / (1.0 + 0.18 * c);
  vCol = c;
  vec4 mv = viewMatrix * vec4(wp, 1.0);
  gl_Position = projectionMatrix * mv;
  // Punktgröße: Weltgröße ~ Punktabstand; weiter weg (dünner) etwas größer; nahe der Schwelle schrumpfen (kein Ploppen)
  float wsize = uSpacing * uSizeK;
  float edge = clamp((f - rank) / (0.18 * f + 1e-5), 0.0, 1.0);
  float px = wsize * uPx / max(-mv.z, 0.05) * (0.45 + 0.55 * edge);
  if (fine) px *= 0.7;
  if (emis) px *= 1.5;
  gl_PointSize = clamp(px, 1.0, emis ? uMaxPx * 1.5 : uMaxPx);
  vSize = gl_PointSize;
  vA = 1.0;
}`;

const FRAG = /* glsl */`
varying vec3 vCol; varying float vA; varying float vRing; varying float vSize;
void main(){
  vec2 q = gl_PointCoord * 2.0 - 1.0; float r2 = dot(q, q);
  if (vSize > 2.5 && r2 > 0.85) discard;   // winzige Punkte (1–2 px) bleiben volle Pixel
  vec3 c = vCol;
  c += vec3(0.5, 0.95, 1.4) * vRing;
  gl_FragColor = vec4(c, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

// Distanz-Material für die Scan-Cubemap (schreibt Abstand zum Scan-Ursprung)
function makeDistMat(origin) {
  // Verdecker sind einfache Meshes in Weltkoordinaten (Kollisionsgeometrie) -> kein Instancing/Skinning nötig
  return new THREE.ShaderMaterial({
    side: THREE.DoubleSide, toneMapped: false, fog: false,
    uniforms: { uO: { value: origin } },
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: 'uniform vec3 uO; varying vec3 vW; void main(){ gl_FragColor = vec4(length(vW - uO)); }',
  });
}

export class PointCloud {
  constructor(scene, renderer, camera, base = 'assets/city/pointcloud/') {
    this.scene = scene; this.renderer = renderer; this.camera = camera; this.base = base;
    this.group = new THREE.Group(); this.group.name = 'Punktwolke';
    this.group.layers.set(3);
    scene.add(this.group);
    this.tiles = []; this.ready = false; this.enabled = true;
    this.loading = 0; this.maxLoads = 6; this.loadedPts = 0; this.drawnPts = 0;
    this.u = {
      uPlayer: { value: new THREE.Vector3() }, uCam: { value: new THREE.Vector3() }, uPx: { value: 800 },
      uNear: { value: PC.near }, uPow: { value: PC.pow }, uFloor: { value: PC.floor }, uFar: { value: PC.far }, uEnd: { value: PC.end },
      uSizeK: { value: PC.sizeK }, uCull: { value: 1 }, uBaked: { value: 0 }, uMaxPx: { value: PC.maxPx }, uExpo: { value: 1.7 }, uSat: { value: 0.32 }, uGamma: { value: 1.3 }, uNoiseLo: { value: 0.45 }, uNoiseAmp: { value: 0.8 }, uDup: { value: 0 },   // per Histogramm-Abgleich mit der Referenz
      uScanO: { value: new THREE.Vector3() }, uScanR: { value: 0 }, uScanAge: { value: 99 }, uScanHold: { value: SCAN_HOLD },
      uScanMax: { value: SCAN_MAX }, uScanOn: { value: 0 }, uScanCube: { value: null },
    };
    // Scan-Cubemap
    this.cubeRT = new THREE.WebGLCubeRenderTarget(512, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.cubeCam = new THREE.CubeCamera(0.3, SCAN_MAX + 20, this.cubeRT);
    this.cubeCam.layers.set(0);
    this.scanOrigin = new THREE.Vector3();
    this.distMat = makeDistMat(this.scanOrigin);
    this.u.uScanCube.value = this.cubeRT.texture;
    this.scanT = -1;
    this.occScene = new THREE.Scene(); this.occScene.name = 'Verdecker';
  }

  // Kollisions-Geometrien (Weltkoordinaten, wenige große Meshes) als Verdecker und für die Scan-Sichtprüfung
  setOccluders(colliders, mat) {
    this.occScene.clear();
    this.occScene.overrideMaterial = mat;
    for (const c of colliders) {
      const g = c.bvh && c.bvh.geometry; if (!g) continue;
      if (/furniture|green|tree|fix.glb|kiosk.glb|mahnmal|pavillon/i.test(c.name)) continue;   // Kleinteile verdecken sonst ihre eigenen Punkte
      if (!g.boundingSphere) g.computeBoundingSphere();
      const m = new THREE.Mesh(g, mat); m.matrixAutoUpdate = false; m.name = 'occ_' + c.name;
      this.occScene.add(m);
    }
    this.occMat = mat;
  }

  async init() {
    const r = await fetch(this.base + 'index.json', { cache: 'no-cache' });
    if (!r.ok) throw new Error('Punktwolke fehlt (' + this.base + 'index.json)');
    const idx = await r.json();
    this.idx = idx;
    this.u.uBaked.value = idx.baked ? 1 : 0;
    const [qx, qy, qz] = idx.quant;
    for (const t of idx.tiles) {
      const T = idx.tile, h = T / 2;
      const tile = {
        ...t, box: new THREE.Box3(new THREE.Vector3(t.cx - h, t.y0, t.cz - h), new THREE.Vector3(t.cx + h, t.y1, t.cz + h)),
        parts: { a: null, b: null }, want: { a: false, b: false }, spacing: 1 / Math.sqrt(Math.max(t.dens, 0.5)),
      };
      tile.uOrigin = new THREE.Vector3(t.cx, 0, t.cz);
      // Zellen (8 m) – ältere Kacheln ohne "cells" gelten als eine Zelle
      const C = idx.cell || T, nc = Math.max(1, Math.round(T / C));
      const cells = t.cells || [[t.na, t.n - t.na]];
      tile.cellList = cells.map(([na, nb], k) => {
        const cu = k % nc, cv = Math.floor(k / nc), s = T / nc;
        const box = new THREE.Box3(new THREE.Vector3(t.cx - h + cu * s, t.y0, t.cz - h + cv * s), new THREE.Vector3(t.cx - h + (cu + 1) * s, t.y1, t.cz - h + (cv + 1) * s));
        return { k, na, nb, box, sphere: box.getBoundingSphere(new THREE.Sphere()) };
      });
      this.tiles.push(tile);
    }
    this.quant = new THREE.Vector3(qx, qy, qz);
    this.ready = true;
    return idx;
  }

  // Eine Kachel-Datei (_a oder _b) ist nach 8-m-Zellen geordnet (bl_tile.cellify). Je Zelle ein eigenes Points-Objekt,
  // alle teilen dieselben Attribut-Puffer (nur ein Upload) und zeichnen per drawRange nur ihren Abschnitt -> es werden nur
  // Zellen im Sichtradius verarbeitet statt ganzer 32-m-Kacheln.
  _makePart(tile, which, buf) {
    const n = which === 'a' ? tile.na : tile.n - tile.na;
    const attrs = {
      pos: new THREE.BufferAttribute(new Int16Array(buf, 0, n * 3), 3, false),
      col: new THREE.BufferAttribute(new Uint8Array(buf, n * 6, n * 4), 4, true),
      oct: new THREE.BufferAttribute(new Uint8Array(buf, n * 10, n * 2), 2, true),
    };
    const objs = [];
    let start = 0;
    for (const c of tile.cellList) {
      const cnt = which === 'a' ? c.na : c.nb;
      if (cnt > 0) {
        const g = new THREE.BufferGeometry();
        for (const k in attrs) g.setAttribute(k, attrs[k]);
        g.setDrawRange(start, 0);
        g.boundingBox = c.box; g.boundingSphere = c.sphere;
        const m = new THREE.ShaderMaterial({
          vertexShader: VERT, fragmentShader: FRAG, toneMapped: false,
          uniforms: { ...this.u, uOrigin: { value: tile.uOrigin }, uQuant: { value: this.quant },
            uOff: { value: (which === 'a' ? 0 : c.na) - start }, uN: { value: c.na + c.nb }, uSpacing: { value: tile.spacing } },
        });
        const p = new THREE.Points(g, m);
        p.layers.set(3); p.matrixAutoUpdate = false; p.visible = false; p.name = `pc_${tile.i}_${tile.j}_${which}_${c.k}`;
        this.group.add(p);
        objs.push({ obj: p, cell: c, start, n: cnt });
      }
      start += cnt;
    }
    tile.parts[which] = { objs, n };
    this.loadedPts += n;
  }

  _load(tile, which) {
    if (tile.parts[which] || tile['_ld' + which]) return;
    tile['_ld' + which] = true; this.loading++;
    fetch(`${this.base}t_${tile.i}_${tile.j}_${which}.bin`).then((r) => r.ok ? r.arrayBuffer() : Promise.reject(r.status))
      .then((buf) => { if (tile.want[which]) this._makePart(tile, which, buf); })
      .catch(() => {}).finally(() => { this.loading--; tile['_ld' + which] = false; });
  }
  _unload(tile, which) {
    const p = tile.parts[which]; if (!p) return;
    for (const o of p.objs) { this.group.remove(o.obj); o.obj.geometry.dispose(); o.obj.material.dispose(); }
    this.loadedPts -= p.n; tile.parts[which] = null;
  }

  frac(d) {
    const u = this.u;
    let f = d <= u.uNear.value ? 1 : Math.pow(u.uNear.value / d, u.uPow.value);
    f = Math.max(f, u.uFloor.value);
    const s = THREE.MathUtils.smoothstep(d, u.uFar.value, u.uEnd.value);
    return f * (1 - s);
  }

  // Streaming + drawRange (jeden Frame; Laden wird gedrosselt)
  update(dt, playerPos, camera, renderer) {
    if (!this.ready) return;
    const u = this.u;
    u.uPlayer.value.copy(playerPos); u.uCam.value.setFromMatrixPosition(camera.matrixWorld);
    u.uPx.value = renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    // Scan-Zeit
    if (this.scanT >= 0) {
      this.scanT += dt;
      u.uScanAge.value = this.scanT; u.uScanR.value = Math.min(this.scanT * SCAN_SPEED, SCAN_MAX + 2); u.uScanOn.value = 1;
      if (this.scanT > SCAN_HOLD) { this.scanT = -1; u.uScanOn.value = 0; }
    }
    const scanning = this.scanT >= 0;
    const so = this.scanOrigin;
    const cand = [];
    let drawn = 0;
    for (const t of this.tiles) {
      const dx = Math.max(t.box.min.x - playerPos.x, 0, playerPos.x - t.box.max.x);
      const dz = Math.max(t.box.min.z - playerPos.z, 0, playerPos.z - t.box.max.z);
      const d = Math.hypot(dx, dz);
      let f = this.frac(d);
      let ds = Infinity;
      if (scanning) {
        const sx = Math.max(t.box.min.x - so.x, 0, so.x - t.box.max.x), sz = Math.max(t.box.min.z - so.z, 0, so.z - t.box.max.z);
        ds = Math.hypot(sx, sz);
        if (ds < SCAN_MAX) f = 1;
      }
      t.want.a = d < PC.coarseDist || ds < SCAN_MAX;
      t.want.b = d < PC.fineDist || ds < 45;
      for (const w of ['a', 'b']) {
        if (t.want[w] && !t.parts[w]) cand.push([w === 'a' ? d : d * 0.8 - 20, t, w]);
        if (!t.want[w] && t.parts[w] && d > (w === 'a' ? PC.coarseDist + 60 : PC.fineDist + 40)) this._unload(t, w);
      }
      const pa = t.parts.a, pb = t.parts.b;
      if (!pa && !pb) continue;
      if (f <= 0) {                                   // ganze Kachel außer Sicht
        for (const p of [pa, pb]) if (p) for (const o of p.objs) o.obj.visible = false;
        continue;
      }
      // je Zelle: Anteil nach Zellabstand, a = die ersten na Punkte, b = Rest
      for (const p of [pa, pb]) if (p) for (const o of p.objs) {
        const c = o.cell, b = c.box;
        let fc = this.frac(Math.hypot(Math.max(b.min.x - playerPos.x, 0, playerPos.x - b.max.x), Math.max(b.min.z - playerPos.z, 0, playerPos.z - b.max.z)));
        if (scanning && fc < 1) {
          const ds = Math.hypot(Math.max(b.min.x - so.x, 0, so.x - b.max.x), Math.max(b.min.z - so.z, 0, so.z - b.max.z));
          if (ds < SCAN_MAX) fc = 1;
        }
        const need = Math.ceil((c.na + c.nb) * Math.min(1, fc * 1.02));
        const cnt = p === pa ? Math.min(c.na, need) : Math.max(0, Math.min(c.nb, need - c.na));
        o.obj.geometry.drawRange.count = cnt; o.obj.visible = cnt > 0; drawn += cnt;
      }
    }
    this.drawnPts = drawn;
    if (cand.length && this.loading < this.maxLoads) {
      cand.sort((a, b) => a[0] - b[0]);
      for (const [, t, w] of cand) { if (this.loading >= this.maxLoads) break; this._load(t, w); }
    }
  }

  // Scan auslösen: Distanz-Cubemap der Stadt-Meshes vom Ursprung aus rendern
  scan(origin) {
    this.scanOrigin.copy(origin);
    this.u.uScanO.value.copy(origin);
    const sc = this.occScene, prevOv = sc.overrideMaterial;
    sc.overrideMaterial = this.distMat;
    const r = this.renderer, prevClear = r.getClearColor(new THREE.Color()), prevA = r.getClearAlpha(), prevAuto = r.autoClear;
    r.autoClear = true;
    // "unendlich" weit = großer Wert
    r.setClearColor(new THREE.Color(1e4, 1e4, 1e4), 1);
    this.cubeCam.position.copy(origin);
    this.cubeCam.updateMatrixWorld(true);
    this.cubeCam.update(r, sc);
    r.setClearColor(prevClear, prevA); r.autoClear = prevAuto;
    sc.overrideMaterial = prevOv;
    this.scanT = 0;
    this.u.uScanOn.value = 1; this.u.uScanAge.value = 0; this.u.uScanR.value = 0;
  }
  get scanning() { return this.scanT >= 0; }

  // Nahbereich mehrfach zeichnen (DUP Durchgänge, versetzt) -> dichte, winzige Punkte wie in der Referenz
  render(renderer, scene, camera, passes = PC.dup) {
    for (let k = 0; k < passes; k++) {
      this.u.uDup.value = k;
      for (const o of this.group.children) if (o.visible) o.material.uniformsNeedUpdate = true;
      renderer.render(scene, camera);
    }
    this.u.uDup.value = 0;
  }
}

// Tiefen-Vorpass der unsichtbaren Stadt-Meshes, leicht nach hinten versetzt: verdeckt Punkte hinter massiven Wänden
export function makeOccluderMaterial() {
  const m = new THREE.MeshDepthMaterial({ side: THREE.FrontSide });
  m.colorWrite = false;
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace('#include <project_vertex>', `#include <project_vertex>
      { float dz = max(0.22, -mvPosition.z * 0.008); mvPosition.xyz += normalize(mvPosition.xyz) * dz; gl_Position = projectionMatrix * mvPosition; }`);
  };
  m.customProgramCacheKey = () => 'pc-occluder';
  return m;
}

// Macht aus dem gerenderten Wandbild eine kachelbare Textur (Farbe + Normal Map).
// Aufruf: node make-seamless.mjs <render.rgba> <render.json> <ausgabe-ordner>
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { writePng } from './splat-lib.mjs';

const [rawFile, metaFile, outDir, cropArg] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const meta = JSON.parse(readFileSync(metaFile, 'utf8'));
const { W, H, pix } = meta;
const src = new Uint8Array(readFileSync(rawFile));
/** Ab dieser Deckkraft (0–255) gilt ein Pixel als erfasst. */
const COVERED = 15;

// 1) Größtes Rechteck mit fast voller Abdeckung (Summed-Area-Table über „Loch“-Pixel)
const holes = new Int32Array((W + 1) * (H + 1));
for (let y = 0; y < H; y++)
  for (let x = 0; x < W; x++) {
    const hole = src[(y * W + x) * 4 + 3] < COVERED ? 1 : 0;
    holes[(y + 1) * (W + 1) + x + 1] = hole + holes[y * (W + 1) + x + 1] + holes[(y + 1) * (W + 1) + x] - holes[y * (W + 1) + x];
  }
const holeCount = (x0, y0, x1, y1) =>
  holes[y1 * (W + 1) + x1] - holes[y0 * (W + 1) + x1] - holes[y1 * (W + 1) + x0] + holes[y0 * (W + 1) + x0];
let best = null;
const STEP = 8;
for (let h = 160; h <= H; h += STEP)
  for (let w = 160; w <= W; w += STEP) {
    if (best && w * h <= best.w * best.h) continue;
    for (let y = 0; y + h <= H; y += STEP)
      for (let x = 0; x + w <= W; x += STEP) {
        if (holeCount(x, y, x + w, y + h) <= w * h * 0.04) {
          best = { x, y, w, h };
          x = W;
          y = H;
        }
      }
  }
if (cropArg) { const [x, y, w, h] = cropArg.split(',').map(Number); best = { x, y, w, h }; console.log('Löcher im Ausschnitt', (holeCount(x, y, x + w, y + h) / (w * h) * 100).toFixed(1) + ' %'); }
console.log('Bester Ausschnitt', best, `= ${(best.w * pix).toFixed(2)} × ${(best.h * pix).toFixed(2)} m`);

// In Float-Kanäle kopieren, kleine Löcher per Diffusion füllen
let { x: cx, y: cy, w: cw, h: ch } = best;
const img = new Float32Array(cw * ch * 3);
const known = new Uint8Array(cw * ch);
for (let y = 0; y < ch; y++)
  for (let x = 0; x < cw; x++) {
    const s = ((cy + y) * W + cx + x) * 4;
    const d = y * cw + x;
    known[d] = src[s + 3] >= COVERED ? 1 : 0;
    for (let k = 0; k < 3; k++) img[d * 3 + k] = src[s + k] / 255;
  }
for (let it = 0; it < 200; it++)
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      const d = y * cw + x;
      if (known[d]) continue;
      for (let k = 0; k < 3; k++) {
        let sum = 0, n = 0;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= cw || yy >= ch) continue;
          sum += img[(yy * cw + xx) * 3 + k];
          n++;
        }
        img[d * 3 + k] = sum / n;
      }
    }

// 2) Beleuchtung ausgleichen: durch stark weichgezeichnete Helligkeit teilen
const lum = new Float32Array(cw * ch);
for (let i = 0; i < cw * ch; i++) lum[i] = 0.2126 * img[i * 3] + 0.7152 * img[i * 3 + 1] + 0.0722 * img[i * 3 + 2];
const blur = boxBlur(boxBlur(boxBlur(lum, cw, ch, 40), cw, ch, 40), cw, ch, 40);
const mean = lum.reduce((a, b) => a + b, 0) / lum.length;
void blur;
void mean;
for (let k = 0; k < 3; k++) {
  const ch_ = new Float32Array(cw * ch);
  for (let i = 0; i < cw * ch; i++) ch_[i] = img[i * 3 + k];
  const b = boxBlur(boxBlur(boxBlur(ch_, cw, ch, 40), cw, ch, 40), cw, ch, 40);
  const m = ch_.reduce((a, c) => a + c, 0) / ch_.length;
  for (let i = 0; i < cw * ch; i++) img[i * 3 + k] = Math.min(1, ch_[i] * (m / Math.max(0.05, b[i])));
}
for (let i = 0; i < cw * ch; i++) lum[i] = 0.2126 * img[i * 3] + 0.7152 * img[i * 3 + 1] + 0.0722 * img[i * 3 + 2];

// 3) Höhe so wählen, dass die Schalungsfugen beim vertikalen Kacheln weiterlaufen:
//    Periode der Zeilenhelligkeit per Autokorrelation
const rows = new Float32Array(ch);
for (let y = 0; y < ch; y++) {
  let s = 0;
  for (let x = 0; x < cw; x++) s += lum[y * cw + x];
  rows[y] = s / cw;
}
const rm = rows.reduce((a, b) => a + b, 0) / ch;
let period = ch, bestCorr = -Infinity;
for (let p = 30; p < ch * 0.75; p++) {
  let c = 0;
  for (let y = 0; y + p < ch; y++) c += (rows[y] - rm) * (rows[y + p] - rm);
  c /= ch - p;
  if (c > bestCorr) {
    bestCorr = c;
    period = p;
  }
}
// Ränder: je Achse ein Überblendbereich m; Kachel = Ausschnitt minus m.
const marginY = Math.max(8, Math.round(period * 0.4));
let tileH = Math.floor((ch - marginY) / period) * period;
if (tileH < period) tileH = ch - marginY; // keine Periode gefunden: einfach überblenden
const marginX = Math.round(cw * 0.15);
const tileW = cw - marginX;
console.log('Fugenperiode', period, 'px =', (period * pix).toFixed(3), 'm · Kachel', tileW, '×', tileH);

// 4) Kachelbar machen: Anfangsbereich mit dem Bild genau eine Kachel weiter überblenden.
//    Dann schließt die letzte Spalte/Zeile nahtlos an die erste an.
const at = (x, y, k) => img[(y * cw + x) * 3 + k];
const vert = new Float32Array(cw * tileH * 3);
for (let y = 0; y < tileH; y++)
  for (let x = 0; x < cw; x++)
    for (let k = 0; k < 3; k++) {
      const t = y < marginY ? smooth(y / marginY) : 1;
      vert[(y * cw + x) * 3 + k] = t === 1 ? at(x, y, k) : at(x, y, k) * t + at(x, y + tileH, k) * (1 - t);
    }
const tile = new Float32Array(tileW * tileH * 3);
for (let y = 0; y < tileH; y++)
  for (let x = 0; x < tileW; x++)
    for (let k = 0; k < 3; k++) {
      const t = x < marginX ? smooth(x / marginX) : 1;
      tile[(y * tileW + x) * 3 + k] = t === 1 ? vert[(y * cw + x) * 3 + k] : vert[(y * cw + x) * 3 + k] * t + vert[(y * cw + x + tileW) * 3 + k] * (1 - t);
    }

// 5) Normal Map aus der Helligkeit (Fugen und Poren als Vertiefung)
const height = new Float32Array(tileW * tileH);
for (let i = 0; i < tileW * tileH; i++) height[i] = 0.2126 * tile[i * 3] + 0.7152 * tile[i * 3 + 1] + 0.0722 * tile[i * 3 + 2];
const hb = boxBlur(height, tileW, tileH, 1);
const normal = new Uint8Array(tileW * tileH * 4);
const color = new Uint8Array(tileW * tileH * 4);
const H_ = (x, y) => hb[((y + tileH) % tileH) * tileW + ((x + tileW) % tileW)];
for (let y = 0; y < tileH; y++)
  for (let x = 0; x < tileW; x++) {
    const i = y * tileW + x;
    const sx = (H_(x + 1, y) - H_(x - 1, y)) * 6;
    const sy = (H_(x, y + 1) - H_(x, y - 1)) * 6;
    const l = Math.hypot(sx, sy, 1);
    normal[i * 4] = Math.round(((-sx / l) * 0.5 + 0.5) * 255);
    normal[i * 4 + 1] = Math.round(((sy / l) * 0.5 + 0.5) * 255);
    normal[i * 4 + 2] = Math.round(((1 / l) * 0.5 + 0.5) * 255);
    normal[i * 4 + 3] = 255;
    for (let k = 0; k < 3; k++) color[i * 4 + k] = Math.round(tile[i * 3 + k] * 255);
    color[i * 4 + 3] = 255;
  }
writePng(`${outDir}/concrete-color.png`, tileW, tileH, color);
writePng(`${outDir}/concrete-normal.png`, tileW, tileH, normal);
// Vorschau 3×2 gekachelt, um Nähte zu prüfen
const PW = tileW * 3, PH = tileH * 2;
const prev = new Uint8Array(PW * PH * 4);
for (let y = 0; y < PH; y++)
  for (let x = 0; x < PW; x++) prev.set(color.subarray(((y % tileH) * tileW + (x % tileW)) * 4, ((y % tileH) * tileW + (x % tileW)) * 4 + 4), (y * PW + x) * 4);
writePng(`${outDir}/preview-tiled.png`, PW, PH, prev);
writeFileSync(`${outDir}/concrete.json`, JSON.stringify({ widthMeters: +(tileW * pix).toFixed(3), heightMeters: +(tileH * pix).toFixed(3), px: [tileW, tileH] }));
console.log('Kachel', tileW, '×', tileH, 'px =', (tileW * pix).toFixed(2), '×', (tileH * pix).toFixed(2), 'm');

function smooth(t) {
  t = Math.min(1, Math.max(0, t));
  return t * t * (3 - 2 * t);
}

function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s / (2 * r + 1);
      s += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / (2 * r + 1);
      s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

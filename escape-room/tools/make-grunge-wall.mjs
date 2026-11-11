// Erzeugt aus dem gerenderten Wandscan eine große, kachelbare und „räudige“ Betontextur.
// 1. Licht ausgleichen  2. Lücken mit Flicken aus erfassten Stellen füllen (Fugen bleiben in Reihe)
// 3. kachelbar machen  4. Schmutz, Laufspuren, Rost, Risse, Ausblühungen einbacken
// 5. Farbe, Normal Map und Rauheit schreiben
//
// Aufruf: node make-grunge-wall.mjs <render.rgba> <render.json> <ausgabe-ordner> [seed]
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { writePng } from './splat-lib.mjs';

const [rawFile, metaFile, outDir, seedArg] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const { W, H, pix } = JSON.parse(readFileSync(metaFile, 'utf8'));
const src = new Uint8Array(readFileSync(rawFile));
let seed = Number(seedArg ?? 7) || 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const px = (m) => Math.round(m / pix); // Meter → Pixel

// ── 1) Einlesen, Licht ausgleichen (normierte Weichzeichnung ignoriert Lücken) ──
const COVERED = 15;
const known = new Uint8Array(W * H);
const img = new Float32Array(W * H * 3);
for (let i = 0; i < W * H; i++) {
  known[i] = src[i * 4 + 3] >= COVERED ? 1 : 0;
  for (let k = 0; k < 3; k++) img[i * 3 + k] = src[i * 4 + k] / 255;
}
{
  const mask = Float32Array.from(known);
  const mb = blur3(mask, W, H, 45);
  for (let k = 0; k < 3; k++) {
    const c = new Float32Array(W * H);
    let sum = 0, n = 0;
    for (let i = 0; i < W * H; i++) if (known[i]) { c[i] = img[i * 3 + k]; sum += c[i]; n++; }
    const mean = sum / n;
    const cb = blur3(c, W, H, 45);
    for (let i = 0; i < W * H; i++) {
      const local = mb[i] > 0.02 ? cb[i] / mb[i] : mean;
      img[i * 3 + k] = known[i] ? clamp01(img[i * 3 + k] * (mean / Math.max(0.05, local))) : 0;
    }
  }
}

// Fugenabstand aus den Zeilen erfasster Pixel
const rows = new Float32Array(H);
for (let y = 0; y < H; y++) {
  let s = 0, n = 0;
  for (let x = 0; x < W; x++) if (known[y * W + x]) { s += lum(img, y * W + x); n++; }
  rows[y] = n > 20 ? s / n : NaN;
}
const period = findPeriod(rows);
console.log('Fugenabstand', period, 'px =', (period * pix).toFixed(3), 'm · erfasst', ((known.reduce((a, b) => a + b, 0) / (W * H)) * 100).toFixed(0), '%');

// ── 2) Lücken füllen: Flicken aus erfassten Stellen, nur waagerecht oder um ganze Fugenabstände versetzt ──
const valid = Uint8Array.from(known); // erfasst oder schon gefüllt
const filled = new Uint8Array(W * H);
const B = 72, O = 18;
for (let pass = 0; pass < 3; pass++) {
  for (let by = 0; by < H; by += B - O) {
    for (let bx = 0; bx < W; bx += B - O) {
      const bw = Math.min(B, W - bx), bh = Math.min(B, H - by);
      let missing = 0;
      for (let y = by; y < by + bh; y++) for (let x = bx; x < bx + bw; x++) if (!valid[y * W + x]) missing++;
      if (!missing) continue;
      let best = null;
      for (let c = 0; c < 120; c++) {
        const dy = Math.round((rand() * 6 - 3)) * period;
        const sx = Math.floor(rand() * (W - bw));
        const sy = by + dy;
        if (sy < 0 || sy + bh > H) continue;
        // Quelle muss fast vollständig original erfasst sein
        let srcKnown = 0;
        for (let y = 0; y < bh; y += 3) for (let x = 0; x < bw; x += 3) srcKnown += known[(sy + y) * W + sx + x];
        if (srcKnown < Math.ceil(bh / 3) * Math.ceil(bw / 3) * 0.97) continue;
        // Passt die Quelle zu den schon vorhandenen Pixeln im Block?
        let err = 0, cnt = 0;
        for (let y = 0; y < bh; y += 2)
          for (let x = 0; x < bw; x += 2) {
            const t = (by + y) * W + bx + x;
            if (!valid[t]) continue;
            const s = (sy + y) * W + sx + x;
            for (let k = 0; k < 3; k++) err += (img[t * 3 + k] - img[s * 3 + k]) ** 2;
            cnt++;
          }
        const score = cnt ? err / cnt : rand();
        if (!best || score < best.score) best = { sx, sy, score };
      }
      if (!best) continue;
      // Einsetzen; am Rand zu vorhandenen Pixeln weich überblenden
      for (let y = 0; y < bh; y++)
        for (let x = 0; x < bw; x++) {
          const t = (by + y) * W + bx + x;
          if (valid[t]) continue;
          const s = (best.sy + y) * W + best.sx + x;
          for (let k = 0; k < 3; k++) img[t * 3 + k] = img[s * 3 + k];
          valid[t] = 1;
          filled[t] = 1;
        }
    }
  }
}
// Übergänge zwischen Original und Flicken glätten
{
  const edge = new Uint8Array(W * H);
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (filled[i] !== filled[i + 1] || filled[i] !== filled[i + W]) for (let d = -3; d <= 3; d++) for (let e = -3; e <= 3; e++) {
        const j = (y + d) * W + x + e;
        if (j >= 0 && j < W * H) edge[j] = 1;
      }
    }
  const copy = Float32Array.from(img);
  for (let y = 2; y < H - 2; y++)
    for (let x = 2; x < W - 2; x++) {
      const i = y * W + x;
      if (!edge[i]) continue;
      for (let k = 0; k < 3; k++) {
        let s = 0;
        for (let d = -2; d <= 2; d++) for (let e = -2; e <= 2; e++) s += copy[((y + d) * W + x + e) * 3 + k];
        img[i * 3 + k] = s / 25;
      }
    }
}
console.log('gefüllt', ((filled.reduce((a, b) => a + b, 0) / (W * H)) * 100).toFixed(0), '%');

// ── 3) Kachelbar: Anfang jeder Achse mit dem Bild eine Kachel weiter überblenden ──
const marginY = Math.max(10, Math.round(period * 0.45));
let TH = Math.floor((H - marginY) / period) * period;
if (TH < period) TH = H - marginY;
const marginX = Math.round(W * 0.12);
const TW = W - marginX;
const tile = new Float32Array(TW * TH * 3);
for (let y = 0; y < TH; y++)
  for (let x = 0; x < TW; x++)
    for (let k = 0; k < 3; k++) {
      const v = (xx, yy) => img[(yy * W + xx) * 3 + k];
      const ty = y < marginY ? smooth(y / marginY) : 1;
      const tx = x < marginX ? smooth(x / marginX) : 1;
      const a = ty === 1 ? v(x, y) : v(x, y) * ty + v(x, y + TH) * (1 - ty);
      const b = ty === 1 ? v(x + TW, y) : v(x + TW, y) * ty + v(x + TW, y + TH) * (1 - ty);
      tile[(y * TW + x) * 3 + k] = tx === 1 ? a : a * tx + b * (1 - tx);
    }
console.log('Kachel', TW, '×', TH, 'px =', (TW * pix).toFixed(2), '×', (TH * pix).toFixed(2), 'm');

// ── 4) Verschmutzung einbacken (alles kachelbar, Koordinaten laufen über den Rand) ──
const height = new Float32Array(TW * TH);
for (let i = 0; i < TW * TH; i++) height[i] = lum(tile, i);
const rough = new Float32Array(TW * TH).fill(0.9);
const wrap = (x, y) => ((((y % TH) + TH) % TH) * TW + (((x % TW) + TW) % TW));
const tint = (i, r, g, b, amount) => {
  tile[i * 3] = mixf(tile[i * 3], tile[i * 3] * r, amount);
  tile[i * 3 + 1] = mixf(tile[i * 3 + 1], tile[i * 3 + 1] * g, amount);
  tile[i * 3 + 2] = mixf(tile[i * 3 + 2], tile[i * 3 + 2] * b, amount);
};

// 4a) Großflächige Schmutzwolken und grünlich-braune Verfärbungen
const cellsX = Math.max(2, Math.round(TW / px(0.35)));
const cellsY = Math.max(2, Math.round(TH / px(0.35)));
for (let y = 0; y < TH; y++)
  for (let x = 0; x < TW; x++) {
    const i = y * TW + x;
    const n = fbm(x / TW, y / TH, cellsX, cellsY, 5, 11);
    const m = fbm(x / TW, y / TH, cellsX * 2, cellsY * 2, 4, 23);
    const grain = fbm(x / TW, y / TH, cellsX * 16, cellsY * 16, 3, 5);
    const dirt = smooth((n - 0.45) / 0.35) * (0.55 + 0.9 * grain);
    tint(i, 0.62, 0.6, 0.52, dirt * 0.75);
    const moss = smooth((m - 0.62) / 0.2);
    tint(i, 0.78, 0.86, 0.66, moss * 0.45);
    rough[i] -= dirt * 0.08;
  }

// 4b) Laufspuren: weiche, unregelmäßige Wasserfahnen von den Fugen nach unten
const stain = new Float32Array(TW * TH);
const rust = new Float32Array(TW * TH);
for (let s = 0; s < Math.round((TW * TH) / 9000); s++) {
  const x0 = Math.floor(rand() * TW);
  const y0 = Math.floor(rand() * TH / period) * period + Math.floor(rand() * 6);
  const len = px(0.15 + rand() * 0.9);
  const w0 = 3 + rand() * 10;
  const dark = 0.25 + rand() * 0.5;
  let x = x0;
  for (let d = 0; d < len; d++) {
    x += (rand() - 0.5) * 0.6;
    const fade = (1 - d / len) ** 1.5;
    const w = w0 * (0.6 + 0.4 * Math.sin(d * 0.07 + s));
    for (let e = -Math.ceil(w); e <= Math.ceil(w); e++) {
      const i = wrap(Math.round(x) + e, y0 + d);
      const f = Math.max(0, 1 - Math.abs(e) / (w + 0.5)) ** 0.7 * fade * dark;
      stain[i] = Math.max(stain[i], f);
    }
  }
}

// 4c) Ankerlöcher der Schalung mit Rostfahnen
const tiesX = Math.max(1, Math.round(TW / px(0.75)));
const tieRows = Math.max(1, Math.round(TH / (period * 2)));
for (let ty = 0; ty < tieRows; ty++)
  for (let tx = 0; tx < tiesX; tx++) {
    const cx = Math.round(((tx + 0.5 + (rand() - 0.5) * 0.2) * TW) / tiesX);
    const cy = Math.round((ty + 0.5) * (TH / tieRows)) + Math.round((rand() - 0.5) * 6);
    const r = px(0.012 + rand() * 0.006);
    for (let y = -r - 2; y <= r + 2; y++)
      for (let x = -r - 2; x <= r + 2; x++) {
        const d = Math.hypot(x, y) / r;
        const i = wrap(cx + x, cy + y);
        if (d < 1) {
          tint(i, 0.35, 0.33, 0.3, 0.85);
          height[i] -= 0.35 * (1 - d * d);
        } else if (d < 1.4) tint(i, 0.8, 0.78, 0.74, (1.4 - d) * 2);
      }
    // Rostfahne
    const len = px(0.12 + rand() * 0.45);
    let x = cx;
    for (let d = r; d < len; d++) {
      x += (rand() - 0.5) * 0.5;
      const fade = (1 - d / len) ** 1.2;
      const w = 3 + (d / len) * 7;
      for (let e = -Math.ceil(w); e <= Math.ceil(w); e++) {
        const f = Math.max(0, 1 - Math.abs(e) / w) * fade;
        const i = wrap(Math.round(x) + e, cy + d);
        rust[i] = Math.max(rust[i], f * 0.8);
      }
    }
  }

{
  const sb = blur3(stain, TW, TH, 2, true);
  const rb = blur3(rust, TW, TH, 2, true);
  for (let y = 0; y < TH; y++)
    for (let x = 0; x < TW; x++) {
      const i = y * TW + x;
      const breakup = 0.45 + 0.9 * fbm(x / TW, y / TH, cellsX * 10, cellsY * 4, 3, 77);
      const st = clamp01(sb[i] * breakup);
      tint(i, 0.62, 0.6, 0.55, st);
      rough[i] -= st * 0.2;
      tint(i, 1.08, 0.7, 0.45, clamp01(rb[i] * breakup));
    }
}

// 4d) Risse: verzweigte Zufallspfade
function crack(x, y, angle, len, depth) {
  for (let d = 0; d < len; d++) {
    angle += (rand() - 0.5) * 0.5;
    x += Math.cos(angle);
    y += Math.sin(angle);
    const i = wrap(Math.round(x), Math.round(y));
    tint(i, 0.55, 0.53, 0.5, 0.6 * depth);
    height[i] -= 0.4 * depth;
    for (const [dx, dy] of [[1, 0], [0, 1]]) {
      const j = wrap(Math.round(x) + dx, Math.round(y) + dy);
      tint(j, 0.85, 0.83, 0.8, 0.3 * depth);
      height[j] -= 0.12 * depth;
    }
    if (rand() < 0.012 && depth > 0.4) crack(x, y, angle + (rand() < 0.5 ? 0.8 : -0.8), len * 0.4, depth * 0.7);
  }
}
for (let c = 0; c < Math.round((TW * TH) / 60000); c++) {
  crack(rand() * TW, rand() * TH, Math.PI / 2 + (rand() - 0.5) * 1.6, px(0.15 + rand() * 0.5), 0.7 + rand() * 0.3);
}

// 4e) Abplatzungen und Lunker (Luftporen)
for (let p = 0; p < Math.round((TW * TH) / 700); p++) {
  const cx = rand() * TW, cy = rand() * TH;
  const r = 0.8 + rand() ** 3 * 4;
  for (let y = -Math.ceil(r); y <= Math.ceil(r); y++)
    for (let x = -Math.ceil(r); x <= Math.ceil(r); x++) {
      const d = Math.hypot(x, y) / r;
      if (d > 1) continue;
      const i = wrap(Math.round(cx) + x, Math.round(cy) + y);
      tint(i, 0.5, 0.48, 0.45, (1 - d) * 0.8);
      height[i] -= 0.25 * (1 - d * d);
    }
}
for (let p = 0; p < 7; p++) {
  // größere Abplatzungen mit hellem Bruch
  const cx = rand() * TW, cy = rand() * TH, r = px(0.02 + rand() * 0.04);
  for (let y = -r; y <= r; y++)
    for (let x = -r; x <= r; x++) {
      const wob = 1 + 0.35 * Math.sin(Math.atan2(y, x) * 5 + p);
      const d = Math.hypot(x, y) / (r * wob);
      if (d > 1) continue;
      const i = wrap(Math.round(cx) + x, Math.round(cy) + y);
      tint(i, 1.12, 1.1, 1.06, 0.6);
      height[i] -= 0.3;
      rough[i] = 1;
    }
}

// 4f) Ausblühungen: kalkig-weiße Schleier an den Fugen
for (let y = 0; y < TH; y++)
  for (let x = 0; x < TW; x++) {
    const i = y * TW + x;
    const nearJoint = 1 - Math.min(1, (y % period) / 10);
    const n = fbm(x / TW, y / TH, cellsX * 3, cellsY * 3, 3, 41);
    const salt = smooth((n - 0.6) / 0.15) * (0.25 + nearJoint * 0.75);
    for (let k = 0; k < 3; k++) tile[i * 3 + k] = mixf(tile[i * 3 + k], 0.86, salt * 0.45);
    rough[i] += salt * 0.08;
  }

// ── 5) Ausgabe ──
const hb = blur3(height, TW, TH, 1, true);
const color = new Uint8Array(TW * TH * 4);
const normal = new Uint8Array(TW * TH * 4);
const roughness = new Uint8Array(TW * TH * 4);
const Hs = (x, y) => hb[wrap(x, y)];
for (let y = 0; y < TH; y++)
  for (let x = 0; x < TW; x++) {
    const i = y * TW + x;
    const sx = (Hs(x + 1, y) - Hs(x - 1, y)) * 7;
    const sy = (Hs(x, y + 1) - Hs(x, y - 1)) * 7;
    const l = Math.hypot(sx, sy, 1);
    normal.set([((-sx / l) * 0.5 + 0.5) * 255, ((sy / l) * 0.5 + 0.5) * 255, ((1 / l) * 0.5 + 0.5) * 255, 255], i * 4);
    for (let k = 0; k < 3; k++) color[i * 4 + k] = Math.round(clamp01(tile[i * 3 + k]) * 255);
    color[i * 4 + 3] = 255;
    const r = Math.round(clamp01(rough[i]) * 255);
    roughness.set([r, r, r, 255], i * 4);
  }
writePng(`${outDir}/concrete-color.png`, TW, TH, color);
writePng(`${outDir}/concrete-normal.png`, TW, TH, normal);
writePng(`${outDir}/concrete-roughness.png`, TW, TH, roughness);
const PW = Math.min(TW * 2, 2400), PH = Math.min(TH * 2, 2400);
const prev = new Uint8Array(PW * PH * 4);
for (let y = 0; y < PH; y++) for (let x = 0; x < PW; x++) prev.set(color.subarray(wrap(x, y) * 4, wrap(x, y) * 4 + 4), (y * PW + x) * 4);
writePng(`${outDir}/preview-tiled.png`, PW, PH, prev);
writeFileSync(`${outDir}/concrete.json`, JSON.stringify({ widthMeters: +(TW * pix).toFixed(3), heightMeters: +(TH * pix).toFixed(3), px: [TW, TH] }));

// ── Hilfsfunktionen ──
function lum(a, i) {
  return 0.2126 * a[i * 3] + 0.7152 * a[i * 3 + 1] + 0.0722 * a[i * 3 + 2];
}
function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}
function mixf(a, b, t) {
  return a + (b - a) * clamp01(t);
}
function smooth(t) {
  t = clamp01(t);
  return t * t * (3 - 2 * t);
}
function findPeriod(rows) {
  const vals = [...rows].filter((v) => !Number.isNaN(v));
  const m = vals.reduce((a, b) => a + b, 0) / vals.length;
  let best = rows.length, bestC = -Infinity;
  for (let p = 40; p < rows.length * 0.6; p++) {
    let c = 0, n = 0;
    for (let y = 0; y + p < rows.length; y++) {
      if (Number.isNaN(rows[y]) || Number.isNaN(rows[y + p])) continue;
      c += (rows[y] - m) * (rows[y + p] - m);
      n++;
    }
    if (n > 30 && c / n > bestC) {
      bestC = c / n;
      best = p;
    }
  }
  return best;
}
/** Kachelbares Value-Noise-fBm: Gitter wiederholt sich nach (cx, cy) Zellen. */
function fbm(u, v, cx, cy, octaves, salt) {
  let sum = 0, amp = 0.5, norm = 0;
  for (let o = 0; o < octaves; o++) {
    const gx = cx << o, gy = cy << o;
    sum += amp * vnoise(u * gx, v * gy, gx, gy, salt + o * 17);
    norm += amp;
    amp *= 0.5;
  }
  return sum / norm;
}
function vnoise(x, y, px_, py_, salt) {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  const h = (i, j) => {
    let n = ((((i % px_) + px_) % px_) * 374761393 + (((j % py_) + py_) % py_) * 668265263 + salt * 2147483647) | 0;
    n = (n ^ (n >>> 13)) * 1274126177;
    return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
  };
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  return mixf(mixf(h(x0, y0), h(x0 + 1, y0), sx), mixf(h(x0, y0 + 1), h(x0 + 1, y0 + 1), sx), sy);
}
function blur3(a, w, h, r, wrapEdges = false) {
  let out = a;
  for (let pass = 0; pass < 3; pass++) out = boxBlur(out, w, h, r, wrapEdges);
  return out;
}
function boxBlur(src, w, h, r, wrapEdges) {
  const at = (x, n) => (wrapEdges ? ((x % n) + n) % n : Math.min(n - 1, Math.max(0, x)));
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[y * w + at(x, w)];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s / (2 * r + 1);
      s += src[y * w + at(x + r + 1, w)] - src[y * w + at(x - r, w)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[at(y, h) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / (2 * r + 1);
      s += tmp[at(y + r + 1, h) * w + x] - tmp[at(y - r, h) * w + x];
    }
  }
  return out;
}

// Findet die größten ebenen Flächen im Scan (RANSAC) und rendert jede als Draufsicht-PNG.
// Aufruf: node find-planes.mjs <scan.ply> <ausgabeordner>
import { mkdirSync, writeFileSync } from 'node:fs';
import { readSplats, sub, dot, cross, norm, quantile, writePng } from './splat-lib.mjs';

const [file, outDir] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const s = readSplats(file);

// Nur deckende, nicht riesige Splats
const cand = [];
for (let i = 0; i < s.count; i++) {
  const maxScale = Math.max(s.scale[i * 3], s.scale[i * 3 + 1], s.scale[i * 3 + 2]);
  if (s.op[i] > 0.5 && maxScale < 0.1) cand.push(i);
}
const P = (i) => [s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]];

// Robuste Gesamtausdehnung
const ax = [0, 1, 2].map((k) => [quantile(cand.map((i) => s.pos[i * 3 + k]), 0.02), quantile(cand.map((i) => s.pos[i * 3 + k]), 0.98)]);
console.log('Splats', s.count, 'Kandidaten', cand.length, 'Ausdehnung 2–98 %', ax.map(([a, b]) => `${a.toFixed(2)}..${b.toFixed(2)}`).join('  '));

let remaining = cand.slice();
const planes = [];
const TOL = 0.02;
let seed = 1;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
for (let p = 0; p < 6 && remaining.length > 500; p++) {
  let best = null;
  for (let it = 0; it < 4000; it++) {
    const a = P(remaining[Math.floor(rand() * remaining.length)]);
    const b = P(remaining[Math.floor(rand() * remaining.length)]);
    const c = P(remaining[Math.floor(rand() * remaining.length)]);
    const n = norm(cross(sub(b, a), sub(c, a)));
    if (!isFinite(n[0]) || Math.hypot(...cross(sub(b, a), sub(c, a))) < 1e-6) continue;
    const d = dot(n, a);
    let count = 0;
    for (let j = 0; j < remaining.length; j += 4) if (Math.abs(dot(n, P(remaining[j])) - d) < TOL) count++;
    if (!best || count > best.count) best = { n, d, count };
  }
  const inliers = remaining.filter((i) => Math.abs(dot(best.n, P(i)) - best.d) < TOL);
  remaining = remaining.filter((i) => Math.abs(dot(best.n, P(i)) - best.d) >= TOL);
  planes.push({ ...best, inliers });
}

// Draufsicht je Ebene
const PIX = 0.004; // 4 mm pro Pixel
const summary = [];
planes.forEach((pl, k) => {
  const n = pl.n;
  const helper = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u = norm(cross(helper, n));
  const v = cross(n, u);
  const us = pl.inliers.map((i) => dot(P(i), u));
  const vs = pl.inliers.map((i) => dot(P(i), v));
  const u0 = quantile(us, 0.01), u1 = quantile(us, 0.99), v0 = quantile(vs, 0.01), v1 = quantile(vs, 0.99);
  const W = Math.min(1600, Math.ceil((u1 - u0) / PIX));
  const H = Math.min(1600, Math.ceil((v1 - v0) / PIX));
  const pix = Math.max((u1 - u0) / W, (v1 - v0) / H);
  const acc = new Float32Array(W * H * 4);
  // alle Splats nahe der Ebene (nicht nur strenge Inlier), weich gezeichnet
  for (let i = 0; i < s.count; i++) {
    const p = P(i);
    if (Math.abs(dot(n, p) - pl.d) > 0.03 || s.op[i] < 0.2) continue;
    const pu = (dot(p, u) - u0) / pix, pv = (v1 - dot(p, v)) / pix;
    const r = Math.min(8, Math.max(1, (Math.max(s.scale[i * 3], s.scale[i * 3 + 1], s.scale[i * 3 + 2]) / pix) * 0.8));
    for (let y = Math.floor(pv - r); y <= Math.ceil(pv + r); y++) {
      if (y < 0 || y >= H) continue;
      for (let x = Math.floor(pu - r); x <= Math.ceil(pu + r); x++) {
        if (x < 0 || x >= W) continue;
        const d2 = ((x - pu) ** 2 + (y - pv) ** 2) / (r * r);
        if (d2 > 1) continue;
        const w = s.op[i] * Math.exp(-3 * d2);
        const o = (y * W + x) * 4;
        acc[o] += s.col[i * 3] * w;
        acc[o + 1] += s.col[i * 3 + 1] * w;
        acc[o + 2] += s.col[i * 3 + 2] * w;
        acc[o + 3] += w;
      }
    }
  }
  const rgba = new Uint8Array(W * H * 4);
  let covered = 0;
  for (let j = 0; j < W * H; j++) {
    const w = acc[j * 4 + 3];
    if (w > 0.05) covered++;
    for (let c = 0; c < 3; c++) rgba[j * 4 + c] = w > 0 ? Math.round((acc[j * 4 + c] / w) * 255) : 0;
    rgba[j * 4 + 3] = Math.min(255, Math.round(w * 400));
  }
  writePng(`${outDir}/plane-${k}.png`, W, H, rgba);
  writeFileSync(`${outDir}/plane-${k}.json`, JSON.stringify({ n, d: pl.d, u, v, u0, u1, v0, v1, pix, W, H }));
  summary.push({
    plane: k,
    normal: n.map((x) => +x.toFixed(2)),
    inliers: pl.inliers.length,
    size_m: [(u1 - u0).toFixed(2), (v1 - v0).toFixed(2)],
    px: [W, H],
    coverage: (covered / (W * H)).toFixed(2),
  });
});
console.table(summary);

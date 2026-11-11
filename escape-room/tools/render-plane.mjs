// Rendert die Splats nahe einer Ebene orthografisch als elliptische Gaussians (Front-to-Back-Alpha).
// Aufruf: node render-plane.mjs <scan.ply> <plane.json> <ausgabe.png> [pixelgröße_m] [band_m]
import { readFileSync, writeFileSync } from 'node:fs';
import { readSplats, dot, writePng } from './splat-lib.mjs';

const [file, planeFile, out, pixArg, bandArg] = process.argv.slice(2);
const pl = JSON.parse(readFileSync(planeFile, 'utf8'));
const PIX = Number(pixArg ?? 0.003);
const BAND = Number(bandArg ?? 0.08);
const s = readSplats(file);
const { n, u, v, u0, u1, v0, v1 } = pl;

// Von welcher Seite schaut man auf die Wand? Dort liegen mehr Splats knapp vor der Ebene.
let front = 0, back = 0;
for (let i = 0; i < s.count; i++) {
  const d = dot(n, [s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]]) - pl.d;
  if (d > 0.005 && d < 0.1) front++;
  if (d < -0.005 && d > -0.1) back++;
}
const side = front >= back ? 1 : -1;

const W = Math.ceil((u1 - u0) / PIX);
const H = Math.ceil((v1 - v0) / PIX);
const list = [];
for (let i = 0; i < s.count; i++) {
  const p = [s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]];
  const d = (dot(n, p) - pl.d) * side; // > 0 = Betrachterseite
  if (d < -BAND * 0.5 || d > BAND || s.op[i] < 0.05) continue;
  list.push({ i, d });
}
// Front-to-Back: zuerst die, die dem Betrachter am nächsten sind
list.sort((a, b) => b.d - a.d);

const color = new Float32Array(W * H * 3);
const trans = new Float32Array(W * H).fill(1);

for (const { i } of list) {
  // Kovarianz Σ = R S² Rᵀ, projiziert auf (u, v)
  const [w, x, y, z] = [s.rot[i * 4], s.rot[i * 4 + 1], s.rot[i * 4 + 2], s.rot[i * 4 + 3]];
  const ql = Math.hypot(w, x, y, z) || 1;
  const qw = w / ql, qx = x / ql, qy = y / ql, qz = z / ql;
  const R = [
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
    [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
    [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
  ];
  const sc = [s.scale[i * 3], s.scale[i * 3 + 1], s.scale[i * 3 + 2]];
  // Achsen des Gaussians in (u, v) projiziert
  let a = 0, b = 0, c = 0;
  for (let k = 0; k < 3; k++) {
    const axis = [R[0][k], R[1][k], R[2][k]];
    const pu = dot(axis, u) * sc[k], pv = dot(axis, v) * sc[k];
    a += pu * pu;
    b += pu * pv;
    c += pv * pv;
  }
  // in Pixel², plus Mindestgröße gegen Aliasing
  a = a / (PIX * PIX) + 0.3;
  b = b / (PIX * PIX);
  c = c / (PIX * PIX) + 0.3;
  const det = a * c - b * b;
  if (det <= 0) continue;
  const ia = c / det, ib = -b / det, ic = a / det;
  const rx = Math.min(60, Math.ceil(3 * Math.sqrt(a)));
  const ry = Math.min(60, Math.ceil(3 * Math.sqrt(c)));
  const p = [s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]];
  const cu = (dot(p, u) - u0) / PIX, cv = (v1 - dot(p, v)) / PIX;
  for (let py = Math.floor(cv - ry); py <= Math.ceil(cv + ry); py++) {
    if (py < 0 || py >= H) continue;
    const dy = py + 0.5 - cv;
    for (let px = Math.floor(cu - rx); px <= Math.ceil(cu + rx); px++) {
      if (px < 0 || px >= W) continue;
      const j = py * W + px;
      if (trans[j] < 0.004) continue;
      const dx = px + 0.5 - cu;
      // v zeigt im Bild nach unten: Vorzeichen von b umkehren
      const power = -0.5 * (ia * dx * dx - 2 * ib * dx * dy + ic * dy * dy);
      if (power < -4.5) continue;
      const alpha = Math.min(0.99, s.op[i] * Math.exp(power));
      if (alpha < 1 / 255) continue;
      const wgt = alpha * trans[j];
      color[j * 3] += s.col[i * 3] * wgt;
      color[j * 3 + 1] += s.col[i * 3 + 1] * wgt;
      color[j * 3 + 2] += s.col[i * 3 + 2] * wgt;
      trans[j] *= 1 - alpha;
    }
  }
}

const rgba = new Uint8Array(W * H * 4);
let covered = 0;
for (let j = 0; j < W * H; j++) {
  const cov = 1 - trans[j];
  if (cov > 0.9) covered++;
  for (let k = 0; k < 3; k++) rgba[j * 4 + k] = cov > 0.01 ? Math.round(Math.min(1, color[j * 3 + k] / cov) * 255) : 0;
  rgba[j * 4 + 3] = Math.round(cov * 255);
}
writePng(out, W, H, rgba);
writeFileSync(out.replace(/.png$/, '.rgba'), rgba);
writeFileSync(out.replace(/\.png$/, '.json'), JSON.stringify({ ...pl, pix: PIX, W, H, side }));
console.log({ W, H, splats: list.length, side, coverage90: (covered / (W * H)).toFixed(2) });

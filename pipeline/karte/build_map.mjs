// Draufsicht der ganzen Punktwolke als Kartenbild (Look: Punkte im Schwarz, wie ein Luft-Scan).
// Jedes Pixel = oberster Punkt der Spalte (Dächer, Bäume, Straßen), Farbe = gebackenes Licht aus den Kacheln.
//   node pipeline/karte/build_map.mjs [m_pro_pixel=0.5]
// Liest web/assets/city/pointcloud/{index.json,t_*_{a,b}.bin}, schreibt web/assets/city/karte/karte.png + karte.json
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { fileURLToPath } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const PC = path.join(ROOT, 'web/assets/city/pointcloud');
const OUT = path.join(ROOT, 'web/assets/city/karte');
const MPP = +(process.argv[2] || 0.5);
const idx = JSON.parse(fs.readFileSync(path.join(PC, 'index.json'), 'utf8'));
const T = idx.tile, H = T / 2, [qx, qy, qz] = idx.quant;
let x0 = 1e9, x1 = -1e9, z0 = 1e9, z1 = -1e9;
for (const t of idx.tiles) { x0 = Math.min(x0, t.cx - H); x1 = Math.max(x1, t.cx + H); z0 = Math.min(z0, t.cz - H); z1 = Math.max(z1, t.cz + H); }
const W = Math.ceil((x1 - x0) / MPP), Hh = Math.ceil((z1 - z0) / MPP);
console.log(`Karte ${W}x${Hh} px, ${MPP} m/px, x ${x0}..${x1}, z ${z0}..${z1}`);
const top = new Float32Array(W * Hh).fill(-1e9);
const rgb = new Uint8Array(W * Hh * 3);
let pts = 0;
for (const t of idx.tiles) {
  for (const which of ['a', 'b']) {
    const f = path.join(PC, `t_${t.i}_${t.j}_${which}.bin`);
    if (!fs.existsSync(f)) continue;
    const n = which === 'a' ? t.na : t.n - t.na;
    if (n <= 0) continue;
    const buf = fs.readFileSync(f);
    const P = new Int16Array(buf.buffer, buf.byteOffset, n * 3), C = new Uint8Array(buf.buffer, buf.byteOffset + n * 6, n * 4);
    for (let k = 0; k < n; k++) {
      const x = t.cx + P[k * 3] * qx, y = P[k * 3 + 1] * qy, z = t.cz + P[k * 3 + 2] * qz;
      if (C[k * 4 + 3] & 4) continue;                                   // Glas weglassen (sonst Schleier über Dächern)
      const u = ((x - x0) / MPP) | 0, v = ((z - z0) / MPP) | 0;
      if (u < 0 || v < 0 || u >= W || v >= Hh) continue;
      const p = v * W + u;
      if (y > top[p]) { top[p] = y; rgb[p * 3] = C[k * 4]; rgb[p * 3 + 1] = C[k * 4 + 1]; rgb[p * 3 + 2] = C[k * 4 + 2]; }
    }
    pts += n;
  }
}
console.log(`${(pts / 1e6).toFixed(1)} M Punkte gelesen`);
// Tonwert: Belichtung + etwas Sättigung (wie der Spiel-Look), Höhe leicht aufhellen (Dächer heller als Straßenschluchten)
const img = Buffer.alloc((W * 3 + 1) * Hh);
for (let v = 0; v < Hh; v++) {
  img[v * (W * 3 + 1)] = 0;
  for (let u = 0; u < W; u++) {
    const p = v * W + u, o = v * (W * 3 + 1) + 1 + u * 3;
    if (top[p] < -1e8) { img[o] = 3; img[o + 1] = 4; img[o + 2] = 6; continue; }
    const hk = 1 + Math.max(-0.25, Math.min(0.35, top[p] / 60));
    let r = rgb[p * 3] / 255, g = rgb[p * 3 + 1] / 255, b = rgb[p * 3 + 2] / 255;
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b, sat = 1.35;
    r = l + (r - l) * sat; g = l + (g - l) * sat; b = l + (b - l) * sat;
    // dunkel wie ein Luft-Scan bei Nacht: Belichtung runter, Kontrast hoch, je Pixel etwas Sensorrauschen (Punkt-Look)
    const hsh = Math.sin((u * 12.9898 + v * 78.233)) * 43758.5453, nz = 0.55 + 0.75 * (hsh - Math.floor(hsh));
    const e = 0.58 * hk * nz;
    img[o] = Math.max(0, Math.min(255, Math.pow(Math.max(r, 0) * e, 1.35) * 255));
    img[o + 1] = Math.max(0, Math.min(255, Math.pow(Math.max(g, 0) * e, 1.35) * 255));
    img[o + 2] = Math.max(0, Math.min(255, Math.pow(Math.max(b, 0) * e, 1.35) * 255 + 4));
  }
}
// PNG schreiben (RGB, zlib)
const crcT = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
const crc = (b) => { let c = -1; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; };
const chunk = (type, data) => { const l = Buffer.alloc(4); l.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([l, td, c]); };
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(Hh, 4); ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
const png = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(img, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'karte.png'), png);
fs.writeFileSync(path.join(OUT, 'karte.json'), JSON.stringify({ x0, z0, x1, z1, mpp: MPP, w: W, h: Hh, quelle: 'pipeline/karte/build_map.mjs', punkte: pts }, null, 1));
console.log(`geschrieben: karte.png ${(png.length / 1e6).toFixed(1)} MB`);

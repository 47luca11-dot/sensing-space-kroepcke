// Hilfsfunktionen: 3DGS-PLY lesen, PNG schreiben, einfache Vektorrechnung.
import { readFileSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

const SH_C0 = 0.28209479177387814;

export function readSplats(path) {
  const buf = readFileSync(path);
  const headerEnd = buf.indexOf('end_header\n') + 'end_header\n'.length;
  const header = buf.subarray(0, headerEnd).toString('latin1');
  const count = Number(/element vertex (\d+)/.exec(header)[1]);
  const props = [...header.matchAll(/property float (\w+)/g)].map((m) => m[1]);
  const stride = props.length * 4;
  const idx = Object.fromEntries(props.map((p, i) => [p, i]));
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  const op = new Float32Array(count);
  const scale = new Float32Array(count * 3);
  const rot = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const o = headerEnd + i * stride;
    const f = (name) => buf.readFloatLE(o + idx[name] * 4);
    pos[i * 3] = f('x');
    pos[i * 3 + 1] = f('y');
    pos[i * 3 + 2] = f('z');
    for (let c = 0; c < 3; c++) col[i * 3 + c] = Math.min(1, Math.max(0, 0.5 + SH_C0 * f(`f_dc_${c}`)));
    op[i] = 1 / (1 + Math.exp(-f('opacity')));
    for (let c = 0; c < 3; c++) scale[i * 3 + c] = Math.exp(f(`scale_${c}`));
    for (let c = 0; c < 4; c++) rot[i * 4 + c] = f(`rot_${c}`);
  }
  return { count, pos, col, op, scale, rot };
}

// ── Vektoren ──
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => {
  const l = Math.hypot(...a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const quantile = (arr, q) => {
  const s = Float64Array.from(arr).sort();
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))];
};

// ── PNG (RGBA 8 bit) ──
const crcTable = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
export function writePng(path, width, height, rgba) {
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  writeFileSync(
    path,
    Buffer.concat([
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
      chunk('IHDR', ihdr),
      chunk('IDAT', deflateSync(raw, { level: 9 })),
      chunk('IEND', Buffer.alloc(0)),
    ]),
  );
}

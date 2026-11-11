import type { SplatData } from './splatData';

const TYPE_SIZES: Record<string, number> = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

interface PlyLayout {
  header: string;
  headerLength: number;
  vertexCount: number;
  stride: number;
}

/** Erkennt binäre Little-Endian-PLYs, deren erstes Element die Vertices mit festen Zeilen sind. */
function parseLayout(bytes: Uint8Array): PlyLayout | null {
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 64 * 1024)));
  const end = head.indexOf('end_header\n');
  if (!head.startsWith('ply') || end < 0) return null;
  const header = head.slice(0, end + 'end_header\n'.length);
  const lines = header.split('\n').map((l) => l.trim());
  if (!lines.includes('format binary_little_endian 1.0')) return null;

  let vertexCount = -1;
  let stride = 0;
  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts[0] === 'element') {
      if (vertexCount >= 0) break; // Nur das erste Element (Vertices) zählt.
      if (parts[1] !== 'vertex') return null;
      vertexCount = Number(parts[2]);
    } else if (parts[0] === 'property' && vertexCount >= 0) {
      if (parts[1] === 'list' || !(parts[1] in TYPE_SIZES)) return null;
      stride += TYPE_SIZES[parts[1]];
    }
  }
  if (vertexCount < 0 || stride === 0) return null;
  return { header, headerLength: new TextEncoder().encode(header).length, vertexCount, stride };
}

/** Kopiert die ausgewählten Zeilen 1:1 aus der Original-PLY (inkl. Spherical Harmonics).
 *  Gibt null zurück, wenn die Datei nicht passt oder die Reihenfolge nicht mit Spark übereinstimmt. */
function exportFromSource(source: Uint8Array, data: SplatData, mask: Uint8Array): Blob | null {
  const layout = parseLayout(source);
  if (!layout || layout.vertexCount !== data.count) return null;

  // Stichprobe: Position aus der Datei muss zur von Spark geladenen Position passen.
  // Spark speichert Positionen als float16, daher relative Toleranz.
  const view = new DataView(source.buffer, source.byteOffset, source.byteLength);
  for (const i of [0, data.count >> 1, data.count - 1]) {
    const off = layout.headerLength + i * layout.stride;
    for (let k = 0; k < 3; k++) {
      const fromFile = view.getFloat32(off + k * 4, true);
      if (Math.abs(fromFile - data.centers[i * 3 + k]) > 2e-3 * Math.abs(fromFile) + 1e-4) return null;
    }
  }

  const selected = mask.reduce((sum, v) => sum + v, 0);
  const header = layout.header.replace(/element vertex \d+/, `element vertex ${selected}`);
  const headerBytes = new TextEncoder().encode(header);
  const out = new Uint8Array(headerBytes.length + selected * layout.stride);
  out.set(headerBytes);
  let pos = headerBytes.length;
  for (let i = 0; i < data.count; i++) {
    if (!mask[i]) continue;
    const off = layout.headerLength + i * layout.stride;
    out.set(source.subarray(off, off + layout.stride), pos);
    pos += layout.stride;
  }
  return new Blob([out], { type: 'application/octet-stream' });
}

/** Schreibt eine Standard-3DGS-PLY aus den entpackten Daten (ohne höhere SH-Bänder). */
function exportFromData(data: SplatData, mask: Uint8Array): Blob {
  const props = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const selected = mask.reduce((sum, v) => sum + v, 0);
  const header =
    `ply\nformat binary_little_endian 1.0\nelement vertex ${selected}\n` +
    props.map((p) => `property float ${p}\n`).join('') +
    'end_header\n';
  const headerBytes = new TextEncoder().encode(header);
  const buf = new ArrayBuffer(headerBytes.length + selected * props.length * 4);
  new Uint8Array(buf).set(headerBytes);
  const f = new DataView(buf, headerBytes.length);
  const SH_C0 = 0.28209479177387814;
  let o = 0;
  const put = (v: number) => {
    f.setFloat32(o, v, true);
    o += 4;
  };
  for (let i = 0; i < data.count; i++) {
    if (!mask[i]) continue;
    for (let k = 0; k < 3; k++) put(data.centers[i * 3 + k]);
    for (let k = 0; k < 3; k++) put((data.colors[i * 3 + k] - 0.5) / SH_C0);
    const a = Math.min(Math.max(data.opacities[i], 1e-6), 1 - 1e-6);
    put(Math.log(a / (1 - a)));
    for (let k = 0; k < 3; k++) put(Math.log(Math.max(data.scales[i * 3 + k], 1e-12)));
    put(data.quats[i * 4 + 3]); // rot_0 = w
    for (let k = 0; k < 3; k++) put(data.quats[i * 4 + k]);
  }
  return new Blob([buf], { type: 'application/octet-stream' });
}

export function exportPly(source: Uint8Array | null, data: SplatData, mask: Uint8Array): Blob {
  return (source && exportFromSource(source, data, mask)) ?? exportFromData(data, mask);
}

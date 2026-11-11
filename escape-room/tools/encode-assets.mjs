// Bereitet einen Build für die Veröffentlichung als claude.ai-Artifact vor:
// Binärdateien (.ply, .spz, .glb …) werden zu <name>.b64.txt, weil Artifacts nur Web-Standardtypen ausliefern.
// Gaussian-Splat-PLYs werden dabei verschlankt: Die blickwinkelabhängigen Farbanteile (f_rest_*) fallen weg,
// das spart rund drei Viertel der Größe.
//
// Aufruf: node tools/encode-assets.mjs <build-ordner>
import { readFileSync, writeFileSync, unlinkSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const root = process.argv[2];
const BINARY = /\.(ply|spz|splat|ksplat|glb)$/i;

for (const file of walk(root)) {
  if (!BINARY.test(file)) continue;
  let bytes = readFileSync(file);
  if (file.toLowerCase().endsWith('.ply')) bytes = stripSphericalHarmonics(bytes);
  writeFileSync(`${file}.b64.txt`, bytes.toString('base64'));
  unlinkSync(file);
  console.log(`${file} → .b64.txt (${(bytes.length / 1e6).toFixed(1)} MB binär)`);
}

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else yield p;
  }
}

/** Schreibt eine 3DGS-PLY ohne f_rest_*-Eigenschaften neu (nur float-Eigenschaften, little endian). */
function stripSphericalHarmonics(buf) {
  const end = buf.indexOf('end_header\n') + 'end_header\n'.length;
  const header = buf.subarray(0, end).toString('latin1');
  if (!header.includes('binary_little_endian')) return buf;
  const count = Number(/element vertex (\d+)/.exec(header)[1]);
  const props = [...header.matchAll(/property (\w+) (\w+)/g)].map((m) => ({ type: m[1], name: m[2] }));
  if (props.some((p) => p.type !== 'float') || !props.some((p) => p.name.startsWith('f_rest_'))) return buf;
  const keep = props.map((p, i) => ({ ...p, i })).filter((p) => !p.name.startsWith('f_rest_'));
  const stride = props.length * 4;
  const out = Buffer.alloc(count * keep.length * 4);
  for (let v = 0; v < count; v++) {
    for (let k = 0; k < keep.length; k++) out.writeFloatLE(buf.readFloatLE(end + v * stride + keep[k].i * 4), (v * keep.length + k) * 4);
  }
  const newHeader = ['ply', 'format binary_little_endian 1.0', `element vertex ${count}`, ...keep.map((p) => `property float ${p.name}`), 'end_header', ''].join('\n');
  return Buffer.concat([Buffer.from(newHeader, 'latin1'), out]);
}

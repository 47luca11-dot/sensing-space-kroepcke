// Baut eine statische Demo des Spiels für ein claude.ai-Artifact: node pipeline/build_artifact_demo.mjs [radius_m]
// Artifact-Grenzen: 256 MB pro Version, 16 MB pro Textdatei, nur Web-Dateitypen (kein .glb/.bin/.ply).
//  - nur Stadtdateien im Umkreis <radius> um den Spawn
//  - Binärdateien (.glb/.bin/.ply) als Base64-Text in Teilen (<name>.<i>.txt); ein fetch-Shim in index.html
//    setzt sie im Browser wieder zusammen (b64index.json: Pfad -> Teilezahl), der Spielcode lädt unverändert
//  - kein /__exists-Endpunkt nötig, index.html als Seitenfragment (das Artifact liefert <head>/<body>)
// Ausgabe: output/hannover/artifact_demo/
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, rmSync, statSync, readdirSync } from 'node:fs';
import { dirname, join, normalize, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const WEB = join(ROOT, 'web'), OUT = join(ROOT, 'artifact_demo');
const RADIUS = Number(process.argv[2] || 300);
const PART = 11 * 1024 * 1024;               // Rohbytes pro Teil -> ~15 MB Base64
const BINARY = new Set(['.glb', '.bin', '.ply']);
const REPO = 'https://github.com/47luca11-dot/sensing-space-kroepcke';

rmSync(OUT, { recursive: true, force: true });
const put = (relOut, data) => { const d = join(OUT, relOut); mkdirSync(dirname(d), { recursive: true }); writeFileSync(d, data); return statSync(d).size; };
const index = {};
let total = 0, nFiles = 0;
// kopiert eine Datei aus web/; Binärdateien werden zu Base64-Teilen
const cp = (rel, relOut = rel) => {
  const buf = readFileSync(join(WEB, rel));
  if (!BINARY.has(extname(rel).toLowerCase())) { nFiles++; return put(relOut, buf); }
  let n = 0, s = 0;
  for (let o = 0; o < buf.length || n === 0; o += PART, n++) s += put(`${relOut}.${n}.txt`, buf.subarray(o, o + PART).toString('base64'));
  index[relOut] = n; nFiles += n;
  return s;
};

// ---------- Code ----------
for (const f of readdirSync(join(WEB, 'js'))) total += cp('js/' + f);
total += cp('main.js'); total += cp('vendor/spark/spark.module.js'); total += cp('vendor/spark/LICENSE', 'vendor/spark/LICENSE.txt');

// world.js: ohne Node-Server kein /__exists -> direkt null zurückgeben (vermeidet 404 in der Konsole)
let world = readFileSync(join(WEB, 'js/world.js'), 'utf8');
world = world.replace(/async function exists\(files\) \{\r?\n/, (m) => m + '  if (globalThis.__STATIC_DEMO) return null;\n');
if (!world.includes('__STATIC_DEMO')) throw new Error('world.js: exists() nicht gefunden');
total += put('js/world.js', world);

// carlos.js: Splats per fetch (läuft durch den Shim) statt über Sparks eigenen Loader
let carlos = readFileSync(join(WEB, 'js/carlos.js'), 'utf8');
const oldSplat = "new SplatMesh({ url: BASE + ID + '_splats.ply' })";
if (!carlos.includes(oldSplat)) throw new Error('carlos.js: SplatMesh-Aufruf nicht gefunden');
carlos = carlos.replace(oldSplat, "new SplatMesh({ fileBytes: new Uint8Array(await (await fetch(BASE + ID + '_splats.ply')).arrayBuffer()), fileName: ID + '_splats.ply' })");
total += put('js/carlos.js', carlos);
for (const f of ['person1.glb', 'person1_skin.bin', 'person1_splats.ply']) total += cp('assets/carlos/' + f);

// ---------- Stadt (Radius um den Spawn) ----------
const man = JSON.parse(readFileSync(join(WEB, 'assets/city/manifest.json'), 'utf8'));
const sp = man.spawn || [0, 0, 0];
const dist = (b) => (!b ? 0 : Math.hypot(Math.max(b[0] - sp[0], 0, sp[0] - b[3]), Math.max(b[2] - sp[2], 0, sp[2] - b[5])));
const keep = man.files.filter((f) => dist(f.bounds) <= RADIUS);
// externe Ressourcen (Texturen/Buffer) aus dem JSON-Chunk einer GLB
const glbUris = (path) => {
  const b = readFileSync(path);
  if (b.readUInt32LE(0) !== 0x46546c67) return [];
  const json = JSON.parse(b.subarray(20, 20 + b.readUInt32LE(12)).toString('utf8'));
  return [...(json.images || []), ...(json.buffers || [])].map((x) => x.uri).filter((u) => u && !u.startsWith('data:'));
};
const copied = new Set();
const cpCity = (rel) => { if (copied.has(rel)) return 0; copied.add(rel); return cp('assets/city/' + rel); };
for (const f of keep) {
  total += cpCity(f.url);
  const dir = f.url.includes('/') ? f.url.slice(0, f.url.lastIndexOf('/') + 1) : '';
  for (const u of glbUris(join(WEB, 'assets/city', f.url))) total += cpCity(normalize(dir + decodeURIComponent(u)).replace(/\\/g, '/'));
}
for (const k of ['bin', 'json']) if (man.heightfield?.[k]) total += cpCity(man.heightfield[k]);
const outMan = { ...man, files: keep, scans: [], demo: { radius: RADIUS, of: man.files.length } };
if (man.clock && !keep.some((f) => f.url === man.clock.url)) delete outMan.clock;
total += put('assets/city/manifest.json', JSON.stringify(outMan));
total += put('b64index.json', JSON.stringify(index));
nFiles += 2;

// ---------- Seite ----------
const SHIM = `<script>
// Binärdateien liegen als Base64-Teile (<pfad>.<i>.txt) neben der Seite; dieser Shim liefert sie dem Spiel als normale Antworten.
globalThis.__STATIC_DEMO = true;
(() => {
  const nativeFetch = globalThis.fetch.bind(globalThis);
  const base = new URL('.', location.href);
  let idx = null;
  const getIdx = () => (idx ??= nativeFetch(new URL('b64index.json', base)).then((r) => r.json()));
  const decode = (s) => {
    if (Uint8Array.fromBase64) return Uint8Array.fromBase64(s);
    const bin = atob(s), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  };
  globalThis.fetch = async (input, init) => {
    const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url, location.href);
    if (url.origin === base.origin && url.pathname.startsWith(base.pathname)) {
      const rel = decodeURIComponent(url.pathname.slice(base.pathname.length));
      const n = (await getIdx())[rel];
      if (n) {
        const parts = await Promise.all(Array.from({ length: n }, (_, i) =>
          nativeFetch(new URL(rel + '.' + i + '.txt', base)).then((r) => { if (!r.ok) throw new Error(rel + ' ' + r.status); return r.text(); }).then(decode)));
        const size = parts.reduce((a, p) => a + p.length, 0), out = new Uint8Array(size);
        let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
        return new Response(out, { status: 200, headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(size) } });
      }
    }
    return nativeFetch(input, init);
  };
})();
</script>`;
const html = readFileSync(join(WEB, 'index.html'), 'utf8');
const css = readFileSync(join(WEB, 'styles.css'), 'utf8');
let body = html.slice(html.indexOf('<body>') + 6, html.indexOf('</body>'));
body = body.replace('<div id="source"></div>', `<div id="source"></div>
    <div class="demo-note">Demo: Innenstadt im Umkreis von ${RADIUS} m um die Kröpcke-Uhr · am besten am Desktop mit Tastatur und Maus</div>
    <a class="repo" href="${REPO}" target="_blank" rel="noopener">Quellcode auf GitHub ↗</a>`);
body = body.replace('<div id="attrib"></div>', `<div id="attrib"></div>\n  <a id="repo-hud" href="${REPO}" target="_blank" rel="noopener">GitHub ↗</a>`);
body = body.replace('<script type="module" src="main.js"></script>', `${SHIM}\n<script type="module" src="main.js"></script>`);
const page = `<title>Hannover-Spaziergang</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
<style>
${css}
/* Artifact-Demo */
html, body { background: #0b1116; }
.demo-note { margin-top: 10px; font-size: 13px; letter-spacing: .08em; color: var(--dim); max-width: 520px; margin-inline: auto; }
.repo { display: inline-block; margin-top: 14px; font-size: 14px; letter-spacing: .18em; color: var(--accent); text-decoration: none; border-bottom: 1px solid currentColor; padding-bottom: 2px; }
.repo:hover, .repo:focus-visible { color: var(--fg); outline: none; }
#repo-hud { position: absolute; left: 10px; bottom: 6px; font-size: 11px; letter-spacing: .14em; color: rgba(242, 245, 247, .7); text-decoration: none; pointer-events: auto; }
#repo-hud:hover { color: #fff; }
@media (max-width: 600px) { .panel { padding: 24px 16px; } .keys { display: none; } }
</style>
${body.trim()}
`;
total += put('index.html', page); nFiles++;
console.log(`Radius ${RADIUS} m: ${keep.length}/${man.files.length} Stadtdateien, ${nFiles} Dateien, gesamt ${(total / 1e6).toFixed(1)} MB`);

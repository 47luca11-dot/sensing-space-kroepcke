// Kleiner Lese-Server für die Projekt-Listen (Begehungsliste, offene Punkte, Treppenbericht) auf localhost.
// Aufruf: node pipeline/tools/md_server.mjs 5178   ->  http://localhost:5178/  (Auswahl oben, ?f=<datei>)
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const PORT = +process.argv[2] || 5178;
const DOCS = [
  ['BEGEHUNGSLISTE.md', 'Begehungsliste'],
  ['pipeline/landmarks/gottfried/OFFENE_PUNKTE.md', 'Gottfrieds Feinkiosk – offene Punkte'],
  ['pipeline/ground/TREPPEN_PASSERELLE.md', 'Passerelle-Treppen – Bericht'],
];

const PAGE = (sel) => `<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sensing Space – Listen</title>
<style>
:root{--bg:#0d0f11;--fg:#e6e6e3;--mut:#9aa0a6;--acc:#7cc4ff;--line:#262a2f;--card:#14171a}
@media (prefers-color-scheme: light){:root{--bg:#fafaf8;--fg:#1d1f21;--mut:#5f6368;--acc:#0b63c4;--line:#e3e3df;--card:#fff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,Segoe UI,sans-serif}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:8px;flex-wrap:wrap;z-index:2}
nav a{color:var(--mut);text-decoration:none;padding:6px 12px;border-radius:999px;border:1px solid var(--line);font-size:14px}
nav a.on{color:var(--fg);border-color:var(--acc)}
main{max-width:860px;margin:0 auto;padding:16px 16px 80px}
h1{font-size:28px;line-height:1.25}h2{margin-top:2.2em;padding-top:.6em;border-top:1px solid var(--line)}h3{margin-top:1.8em;color:var(--acc)}
a{color:var(--acc)}code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em}
table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}td,th{border:1px solid var(--line);padding:6px 8px;text-align:left}
li{margin:.25em 0}strong{color:var(--fg)}
</style></head><body>
<nav>${DOCS.map(([f, t]) => `<a href="?f=${encodeURIComponent(f)}" class="${f === sel ? 'on' : ''}">${t}</a>`).join('')}</nav>
<main id="doc">Lade …</main>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script>
fetch('/raw?f=${encodeURIComponent(sel)}').then(r => r.ok ? r.text() : Promise.reject(r.status))
  .then(t => { document.getElementById('doc').innerHTML = marked.parse(t); document.title = (document.querySelector('h1')||{}).textContent || document.title; })
  .catch(e => { document.getElementById('doc').textContent = 'Datei nicht gefunden (' + e + ')'; });
</script></body></html>`;

http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  const f = u.searchParams.get('f') || DOCS[0][0];
  if (!DOCS.some(([d]) => d === f)) { res.writeHead(404); return res.end('unbekannt'); }
  if (u.pathname === '/raw') {
    fs.readFile(path.join(ROOT, f), 'utf8', (err, t) => {
      if (err) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { 'content-type': 'text/markdown; charset=utf-8', 'cache-control': 'no-store' }); res.end(t);
    });
    return;
  }
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end(PAGE(f));
}).listen(PORT, () => console.log(`Listen-Server: http://localhost:${PORT}/`));

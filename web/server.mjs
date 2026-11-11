// Minimaler statischer Server für den Hannover-Spaziergang: node server.mjs [port]
//  GET /__exists?f=assets/city/hannover.glb,assets/city/collision.glb  -> {"assets/city/hannover.glb": true, ...}
//  (damit der Client fehlende Stadt-Dateien ohne 404-Meldungen in der Konsole erkennt)
import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.argv[2] || 5177);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.svg': 'image/svg+xml', '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json', '.json': 'application/json', '.wasm': 'application/wasm', '.ktx2': 'image/ktx2',
  '.ply': 'application/octet-stream', '.bin': 'application/octet-stream', '.woff2': 'font/woff2' };

const safe = (p) => {
  const n = normalize(decodeURIComponent(p)).replace(/^([\\/])+/, '');
  return n.startsWith('..') ? null : n;
};

http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/__exists') {
    const out = {};
    for (const f of (url.searchParams.get('f') || '').split(',').filter(Boolean)) {
      const p = safe(f);
      try { out[f] = !!p && (await stat(join(root, p))).isFile(); } catch { out[f] = false; }
    }
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' }).end(JSON.stringify(out));
    return;
  }
  const path = safe(url.pathname);
  if (path === null) { res.writeHead(403).end(); return; }
  try {
    const file = join(root, path || 'index.html');
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': types[extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch { res.writeHead(404).end('not found'); }
}).listen(port, '127.0.0.1', () => console.log(`Hannover-Spaziergang: http://localhost:${port}`));

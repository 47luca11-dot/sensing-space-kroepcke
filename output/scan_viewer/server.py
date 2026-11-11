"""Lokaler Scan-Viewer: PLY hochladen, ansehen, Person/Objekt automatisch extrahieren, Ergebnis herunterladen.

Start:  .venv/Scripts/python server.py [port]      (Standard 5176, nur 127.0.0.1)
Uploads und Extrakte liegen in ./uploads und ./extracted.
"""
import io, json, os, re, sys, time, threading, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import extract as X  # noqa: E402
import process as Pr  # noqa: E402

UP, EX, ST = (os.path.join(ROOT, d) for d in ('uploads', 'extracted', 'static'))
for d in (UP, EX): os.makedirs(d, exist_ok=True)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5176
MAX_UPLOAD = 2 * 1024 ** 3
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
         '.ply': 'application/octet-stream', '.png': 'image/png', '.svg': 'image/svg+xml'}
SCANS = {}            # id -> {name, path, size, info?, extracted?}
LOCK = threading.Lock()
CACHE = {}            # id -> loaded splat data (last 2)


def safe_name(n):
    n = os.path.basename(n or 'scan.ply')
    n = re.sub(r'[^\w\-. ()äöüÄÖÜß]', '_', n)
    return n if n.lower().endswith('.ply') else n + '.ply'


def register(path, name):
    sid = uuid.uuid4().hex[:10]
    SCANS[sid] = dict(id=sid, name=name, path=path, size=os.path.getsize(path), added=time.time())
    return sid


def get_loaded(sid):
    with LOCK:
        if sid in CACHE: return CACHE[sid]
    S = X.load(SCANS[sid]['path'])
    with LOCK:
        CACHE[sid] = S
        while len(CACHE) > 2: CACHE.pop(next(iter(CACHE)))
    return S


class H(BaseHTTPRequestHandler):
    server_version = 'ScanViewer/1.0'

    def log_message(self, fmt, *a):
        sys.stderr.write('%s %s\n' % (self.command, self.path))

    def send(self, code, body=b'', ctype='application/json', extra=None):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def send_file(self, path, ctype, download=None):
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-cache')
        if download: self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(download))
        self.end_headers()
        with open(path, 'rb') as f:
            while True:
                b = f.read(1 << 20)
                if not b: break
                self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); p = u.path; q = parse_qs(u.query)
        try:
            if p == '/api/scans':
                return self.send(200, sorted([{k: v for k, v in s.items() if k != 'path'} for s in SCANS.values()], key=lambda s: -s['added']))
            m = re.fullmatch(r'/api/scan/(\w+)/(original|extracted|mesh)\.ply', p)
            if m:
                s = SCANS.get(m[1])
                if not s: return self.send(404, {'error': 'unbekannt'})
                path = {'original': s['path'], 'extracted': s.get('extracted'), 'mesh': s.get('mesh')}[m[2]]
                if not path or not os.path.isfile(path): return self.send(404, {'error': 'noch nicht vorhanden'})
                dl = None
                if 'download' in q:
                    dl = {'original': s['name'], 'extracted': s['name'][:-4] + '_extrahiert.ply', 'mesh': s['name'][:-4] + '_mesh.ply'}[m[2]]
                return self.send_file(path, 'application/octet-stream', dl)
            f = os.path.normpath(os.path.join(ST, p.lstrip('/') or 'index.html'))
            if not f.startswith(ST) or not os.path.isfile(f): return self.send(404, 'not found', 'text/plain')
            return self.send_file(f, TYPES.get(os.path.splitext(f)[1], 'application/octet-stream'))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        u = urlparse(self.path); p = u.path; q = parse_qs(u.query)
        try:
            if p == '/api/upload':
                n = int(self.headers.get('Content-Length', 0))
                if n <= 0 or n > MAX_UPLOAD: return self.send(413, {'error': 'Datei zu groß oder leer'})
                name = safe_name(q.get('name', ['scan.ply'])[0])
                path = os.path.join(UP, uuid.uuid4().hex[:6] + '_' + name)
                left = n
                with open(path, 'wb') as f:
                    while left > 0:
                        b = self.rfile.read(min(left, 1 << 20))
                        if not b: break
                        f.write(b); left -= len(b)
                with open(path, 'rb') as f:
                    head = f.read(4)
                if head != b'ply\n' and head[:3] != b'ply':
                    os.remove(path); return self.send(400, {'error': 'keine PLY-Datei'})
                sid = register(path, name)
                return self.send(200, {k: v for k, v in SCANS[sid].items() if k != 'path'})  # extracted/mesh sind Pfade -> nur als Flag genutzt
            m = re.fullmatch(r'/api/scan/(\w+)/extract', p)
            if m:
                s = SCANS.get(m[1])
                if not s: return self.send(404, {'error': 'unbekannt'})
                n = int(self.headers.get('Content-Length', 0) or 0)
                opts = json.loads(self.rfile.read(n) or b'{}') if n else {}
                t0 = time.time()
                S = get_loaded(m[1])
                g = opts.get('group', 'auto'); g = None if g == 'auto' else bool(g)
                pre = os.path.join(EX, os.path.basename(s['path'])[:-4])
                do_mesh = bool(opts.get('mesh', True))
                info = Pr.run(S, pre, radius=float(opts.get('radius', 2.5)), group=g, do_clean=bool(opts.get('clean', True)),
                              do_mesh=do_mesh, min_opacity=float(opts.get('min_opacity', 0.1)), keep_soft=bool(opts.get('keep_soft', True)))
                s['extracted'] = pre + '_extrahiert.ply'; s['info'] = info
                if do_mesh: s['mesh'] = pre + '_mesh.ply'
                elif 'mesh' in s and os.path.isfile(s['mesh']): os.remove(s['mesh']); s.pop('mesh')
                return self.send(200, info)
            return self.send(404, {'error': 'unbekannt'})
        except Exception as e:
            traceback.print_exc()
            return self.send(500, {'error': str(e)})


if __name__ == '__main__':
    # Scans, die schon im Ordner liegen (z. B. per Kommandozeile extrahiert), wieder anbieten
    for fn in sorted(os.listdir(UP)):
        if fn.lower().endswith('.ply'):
            sid = register(os.path.join(UP, fn), fn.split('_', 1)[1] if '_' in fn else fn)
            ex = os.path.join(EX, fn[:-4] + '_extrahiert.ply')
            if os.path.isfile(ex):
                SCANS[sid]['extracted'] = ex
                if os.path.isfile(ex.replace('_extrahiert.ply', '_mesh.ply')): SCANS[sid]['mesh'] = ex.replace('_extrahiert.ply', '_mesh.ply')
                j = ex[:-4] + '.json'
                if os.path.isfile(j): SCANS[sid]['info'] = json.load(open(j))
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    print(f'Scan-Viewer: http://localhost:{PORT}', flush=True)
    srv.serve_forever()

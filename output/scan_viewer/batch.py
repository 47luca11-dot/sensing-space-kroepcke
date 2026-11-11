"""Alle übergebenen Scans in den Viewer übernehmen und komplett verarbeiten (parallel)."""
import sys, os, json, shutil, hashlib, traceback
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)

def job(path):
    import extract as X, process as Pr
    name = os.path.basename(path)
    sid = hashlib.md5(name.encode()).hexdigest()[:6]
    dst = os.path.join(ROOT, 'uploads', f'{sid}_{name}')
    if not os.path.isfile(dst): shutil.copyfile(path, dst)
    pre = os.path.join(ROOT, 'extracted', f'{sid}_{name[:-4]}')
    try:
        info = Pr.run(X.load(dst), pre)
        m = info['mesh']
        return f"{name}: {info['selected']}/{info['total']} Splats, Höhe {info['height']:.2f}, Mesh {m['faces']} F, dicht={m['watertight']}, Rand={m['boundary_edges']}, Teile={m['components']}, {info['seconds']} s"
    except Exception as e:
        return f"{name}: FEHLER {e}\n{traceback.format_exc()[-600:]}"

if __name__ == '__main__':
    with ProcessPoolExecutor(3) as ex:
        for r in ex.map(job, sys.argv[1:]): print(r, flush=True)

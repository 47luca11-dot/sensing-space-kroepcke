"""Nur fuer Vorschaubilder: LoD2-Gebaeude (ausser den im Detailmodell enthaltenen) als einfache graue Koerper."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, mapbox_earcut as earcut
from common import *
from glb import GLB
B = json.load(open(DATA + 'lod2_buildings.json'))['buildings']
cov = set(json.load(open(DATA + 'detail_covered.json'))['ids'])
P = {'wall': [], 'roof': []}
def tri3(ring):
    R = np.array(ring, float)
    if len(R) < 3: return None
    R = np.c_[R[:, 0], R[:, 2] - H0, -R[:, 1]]          # Spiel: x=dE, y=Z-56.16, z=-dN
    c = R.mean(0); U, S, Vt = np.linalg.svd(R - c); n = Vt[2]
    a = Vt[0]; b = Vt[1]
    Q = np.c_[(R - c) @ a, (R - c) @ b]
    t = earcut.triangulate_float64(Q, np.array([len(Q)], np.uint32)).reshape(-1, 3)
    return R, t
for b in B:
    if b['id'] in cov: continue
    for k, key in (('walls', 'wall'), ('roofs', 'roof')):
        for poly in b[k]:
            r = tri3(poly['ext'])
            if r is None or len(r[1]) == 0: continue
            P[key].append(r)
g = GLB()
prims = []
for key, col in (('wall', (0.62, 0.60, 0.56, 1)), ('roof', (0.30, 0.30, 0.32, 1))):
    off = 0; V = []; I = []
    for R, t in P[key]:
        V.append(R); I.append(t + off); off += len(R)
    prims.append({'pos': np.concatenate(V), 'nrm': None, 'uv': None, 'idx': np.concatenate(I), 'mat': 'lod2_' + key})
mats = {'lod2_wall': {'color': (0.62, 0.60, 0.56, 1), 'rough': 0.9, 'double': True}, 'lod2_roof': {'color': (0.25, 0.25, 0.27, 1), 'rough': 0.9, 'double': True}}
mi = g.mesh('lod2_preview', prims, mats); g.node('lod2_preview', mi)
print('saved', g.save(CACHE + 'lod2_preview.glb'), 'tris', sum(len(p['idx']) for p in prims))

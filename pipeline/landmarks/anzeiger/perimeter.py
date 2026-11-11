"""Perimeter closure test (game renders FrontSide only): horizontal rays from 1.5 m outside the replaced footprints
(LoD2 grounds + 0.25 m) toward the building, every 0.25 m along the outline, at ground + 0.02 / 0.06 / 0.12 / 0.3 /
0.6 / 1 / 1.5 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 10 / 12 m (only heights at least 0.3 m under the model top at that spot).
A ray must first hit a FRONT face of the landmark (or a neighbouring LoD2 building) within 6 m.
Misses = slits / gaps under walls, back-face hits = walls facing the wrong way.  Exit code 1 if any fail.
Run: python perimeter.py   (reads build/anzeiger_parts.npz)"""
import os, sys, json, collections
import numpy as np, trimesh
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
meta = json.load(open(os.path.join(HERE, 'build', 'anzeiger_meta.json')))
npz = np.load(os.path.join(HERE, 'build', 'anzeiger_parts.npz'))
V = np.concatenate([npz['V_' + o].astype(np.float64) for o in meta['objects']])
NL = len(V) // 3
rep = json.load(open(ROOT + 'data/landmarks_replace/anzeiger.json'))
L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))
occ, nfp = [], []
for b in L['buildings']:
    g = np.array(b['grounds'][0]['ext'])
    if np.hypot(g[:, 0].mean() + 510, -g[:, 1].mean() + 310) > 120 or b['id'] in rep['lod2_ids']: continue
    nfp.append(Polygon([(p[0], -p[1]) for p in g]).buffer(0))
    for part in b['walls'] + b['roofs']:
        P = np.array([[p[0], p[2] - 56.16, -p[1]] for p in part['ext']])
        for i in range(1, len(P) - 1): occ.append([P[0], P[i], P[i + 1]])
NEI = unary_union(nfp)
Vall = np.concatenate([V, np.array(occ).reshape(-1, 3)])
mesh = trimesh.Trimesh(Vall, np.arange(len(Vall)).reshape(-1, 3), process=False)
N = mesh.face_normals
from trimesh.ray.ray_pyembree import RayMeshIntersector
ri = RayMeshIntersector(mesh)
hj = json.load(open(ROOT + 'web/assets/city/ground/heightfield.json'))
hf = np.fromfile(ROOT + 'web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(hj['h'], hj['w'])
def H(x, z):
    fx = (x - hj['x0']) / hj['dx']; fz = (z - hj['z0']) / hj['dz']; i = int(np.floor(fx)); j = int(np.floor(fz)); a = fx - i; b = fz - j
    return float((1 - a) * (1 - b) * hf[j, i] + a * (1 - b) * hf[j, i + 1] + (1 - a) * b * hf[j + 1, i] + a * b * hf[j + 1, i + 1])
FP = unary_union([Polygon(f) for f in rep['footprints']])
SITE = json.load(open(os.path.join(HERE, 'site_lod2.json')))
CORE = unary_union([Polygon(np.array(SITE[k]['grounds'][0])[:, [0, 2]]).buffer(0) for k in ('anz', 'bad')]).buffer(-0.2)
from shapely.geometry import LineString
OFF = 3.0
HS = [0.02, 0.06, 0.12, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
O, D, info = [], [], []
for ring in (FP.geoms if hasattr(FP, 'geoms') else [FP]):
    ext = ring.exterior
    for s in np.arange(0, ext.length, 0.25):
        p = ext.interpolate(s); p2 = ext.interpolate(s + 0.05)
        t = np.array([p2.x - p.x, p2.y - p.y]); t /= np.linalg.norm(t) + 1e-12
        nr = np.array([t[1], -t[0]])
        if ring.contains(Point(p.x + nr[0] * 0.3, p.y + nr[1] * 0.3)): nr = -nr
        for off in (OFF, 2.0, 1.2, 0.6):               # origin must lie outside every footprint (concave corners)
            o = np.array([p.x, p.y]) + nr * off
            if not FP.contains(Point(*o)): break
        else:
            continue
        if NEI.buffer(0.3).contains(Point(*o)): continue          # origin inside a neighbour -> wall there is hidden anyway
        if not LineString([o, o - nr * (OFF + 8.0)]).intersects(CORE): continue     # grazing ray along a wall: not aimed at the building
        g = H(o[0], o[1])
        # model top at 0.8 m inside the outline
        q = np.array([p.x, p.y]) - nr * 0.8
        loc, ir, it = ri.intersects_location([[q[0], 200.0, q[1]]], [[0, -1.0, 0]], multiple_hits=False)
        top = loc[0][1] if len(loc) else -10
        for h in HS:
            if g + h > top - 0.3: continue
            O.append([o[0], g + h, o[1]]); D.append([-nr[0], 0.0, -nr[1]]); info.append((p.x, p.y, h))
O = np.array(O); D = np.array(D)
loc, ir, it = ri.intersects_location(O, D, multiple_hits=False)
ok = np.zeros(len(O), bool); back = np.zeros(len(O), bool)
for k, r in enumerate(ir):
    d = np.linalg.norm(loc[k] - O[r])
    if d > OFF + 8.0: continue
    if np.dot(D[r], N[it[k]]) < 0: ok[r] = True
    else: back[r] = True
bad = ~ok
print('PERIMETER rays %d, failing %d (misses %d, back-face first %d)' % (len(O), bad.sum(), (bad & ~back).sum(), back.sum()))
cl = collections.defaultdict(list)
for k in np.where(bad)[0]:
    x, z, h = info[k]
    cl[(int(np.floor(x / 2)) * 2, int(np.floor(z / 2)) * 2, "back" if back[k] else "miss")].append((h, (np.round(O[k], 2).tolist(), np.round(D[k], 2).tolist())))
for k, v in sorted(cl.items()):
    print("   x %d..%d z %d..%d %s: %d rays, heights %s" % (k[0], k[0] + 2, k[1], k[1] + 2, k[2], len(v), sorted(set(h for h, _ in v))), "e.g. from", v[0][1])
sys.exit(1 if bad.sum() else 0)

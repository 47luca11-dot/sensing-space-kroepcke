"""Marktkirche -- Pruefung des exportierten GLB (Spielframe): Strahlen von aussen (Kugelschale + Fussgaengerhoehe) auf das Modell;
Treffer auf Rueckseiten (Normale zeigt vom Betrachter weg) = Loch/verdrehte Normale -> Liste mit Ort + Objekt.
Aufruf: python mk_check.py <model.glb> [n_rays]"""
import sys, json, math
import numpy as np, trimesh

path = sys.argv[1]; NR = int(sys.argv[2]) if len(sys.argv) > 2 else 400000
sc = trimesh.load(path, force='scene')
meshes, names = [], []
for nm, g in sc.geometry.items():
    T = sc.graph.get(nm) if False else None
dump = sc.dump(concatenate=False) if hasattr(sc, 'dump') else list(sc.geometry.values())
tri_off = [0]; allm = []
for m in dump:
    allm.append(m); names.append(m.metadata.get('name', '?') if hasattr(m, 'metadata') else '?'); tri_off.append(tri_off[-1] + len(m.faces))
M = trimesh.util.concatenate(allm)
print('tris', len(M.faces), 'bounds', M.bounds.round(2).tolist())
lo, hi = M.bounds; c = (lo + hi) / 2
rng = np.random.default_rng(1)
# Strahlenquellen: (a) Halbkugel um das Modell, (b) Augenhoehe-Ring im Abstand 15-40 m
n1 = NR // 2
d = rng.standard_normal((n1, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True); d[:, 1] = np.abs(d[:, 1]) * 0.9 + 0.05
d /= np.linalg.norm(d, axis=1, keepdims=True)
R = np.linalg.norm(hi - lo) * 0.6
o1 = c + d * R
tgt = lo + rng.random((n1, 3)) * (hi - lo)
d1 = tgt - o1; d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
n2 = NR - n1
ang = rng.random(n2) * 2 * math.pi; rr = 15 + rng.random(n2) * 25
o2 = np.c_[c[0] + np.cos(ang) * (hi[0] - lo[0]) / 2 * 1.0 + np.cos(ang) * rr, 1.0 + rng.random(n2) * 2.0, c[2] + np.sin(ang) * (hi[2] - lo[2]) / 2 + np.sin(ang) * rr]
tgt2 = np.c_[lo[0] + rng.random(n2) * (hi[0] - lo[0]), rng.random(n2) * 40, lo[2] + rng.random(n2) * (hi[2] - lo[2])]
d2 = tgt2 - o2; d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
O = np.vstack([o1, o2]); D = np.vstack([d1, d2])
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    inter = RayMeshIntersector(M)
except Exception:
    inter = M.ray
idx_tri, idx_ray, locs = inter.intersects_id(O, D, multiple_hits=False, return_locations=True)
fn = M.face_normals[idx_tri]
dots = np.einsum('ij,ij->i', fn, D[idx_ray])
back = dots > 0.02
print('rays', len(O), 'hits', len(idx_tri), 'backface hits', int(back.sum()), '(%.3f%%)' % (100 * back.mean() if len(back) else 0))
obj = np.searchsorted(np.array(tri_off), idx_tri, side='right') - 1
from collections import Counter
cnt = Counter(names[o] for o in obj[back])
print('by object:', cnt.most_common(12))
bl = locs[back]
if len(bl):
    # Cluster grob auf 1-m-Raster
    key = Counter(tuple(np.round(p).astype(int)) for p in bl)
    print('hot spots (x,y,z game):', key.most_common(25))
json.dump(dict(backface=int(back.sum()), hits=int(len(idx_tri))), open(path + '.check.json', 'w'))

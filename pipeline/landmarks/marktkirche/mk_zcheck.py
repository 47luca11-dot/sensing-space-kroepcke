"""Marktkirche -- Z-Fighting-Suche: Strahlen von aussen, alle Treffer; erste zwei Treffer < 2 mm auseinander, beide Vorderseiten,
gleiche Normale, verschiedene Objekte/Materialien -> Kandidat. Aufruf: python mk_zcheck.py <model.glb> [n]"""
import sys, math
import numpy as np, trimesh
from collections import Counter
path = sys.argv[1]; NR = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002          # Abstand fuer "gestapelt" (m)
sc = trimesh.load(path, force='scene')
dump = sc.dump(concatenate=False)
names = [m.metadata.get('name', '?') for m in dump]
off = np.cumsum([0] + [len(m.faces) for m in dump])
M = trimesh.util.concatenate(dump)
lo, hi = M.bounds; c = (lo + hi) / 2
rng = np.random.default_rng(7)
d = rng.standard_normal((NR, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True); d[:, 1] = np.abs(d[:, 1]) * 0.6 + 0.02
d /= np.linalg.norm(d, axis=1, keepdims=True)
O = c + d * np.linalg.norm(hi - lo) * 0.6
T = lo + rng.random((NR, 3)) * (hi - lo); T[:, 1] = rng.random(NR) ** 2 * (hi[1] - lo[1])
D = T - O; D /= np.linalg.norm(D, axis=1, keepdims=True)
if len(sys.argv) > 4:        # gezielt: Ziele in Kugeln um Spielframe-Punkte 'x,y,z,r;x,y,z,r', Quellen 3-12 m entfernt (obere Halbkugel)
    spots_ = [tuple(map(float, s.split(','))) for s in sys.argv[4].split(';')]
    k_ = rng.integers(0, len(spots_), NR)
    C_ = np.array([s[:3] for s in spots_])[k_]; R_ = np.array([s[3] for s in spots_])[k_]
    T = C_ + (rng.random((NR, 3)) * 2 - 1) * R_[:, None]
    dd_ = rng.standard_normal((NR, 3)); dd_[:, 1] = np.abs(dd_[:, 1]) * 0.8 + 0.05; dd_ /= np.linalg.norm(dd_, axis=1, keepdims=True)
    O = T + dd_ * (3 + 9 * rng.random(NR))[:, None]
    D = T - O; D /= np.linalg.norm(D, axis=1, keepdims=True)
from trimesh.ray.ray_pyembree import RayMeshIntersector
it = RayMeshIntersector(M)
tri, ray, loc = it.intersects_id(O, D, multiple_hits=True, max_hits=3, return_locations=True)
dist = np.linalg.norm(loc - O[ray], axis=1)
order = np.lexsort((dist, ray)); tri, ray, loc, dist = tri[order], ray[order], loc[order], dist[order]
cand = Counter(); spots = []
first = np.r_[True, ray[1:] != ray[:-1]]
idx = np.where(first)[0]
for i in idx:
    if i + 1 >= len(ray) or ray[i + 1] != ray[i]: continue
    a, b = tri[i], tri[i + 1]
    if dist[i + 1] - dist[i] > THR: continue
    na, nb = M.face_normals[a], M.face_normals[b]
    dd = D[ray[i]]
    if na @ dd > 0 or nb @ dd > 0: continue
    if na @ nb < 0.995: continue
    if names[np.searchsorted(off, a, side="right") - 1] == names[np.searchsorted(off, b, side="right") - 1]: continue
    oa = np.searchsorted(off, a, side='right') - 1; ob = np.searchsorted(off, b, side='right') - 1
    cand[(names[oa], names[ob])] += 1; spots.append(np.round(loc[i], 1))
print('rays', NR, 'thr', THR, 'z-fight candidates', sum(cand.values()))
for k, v in cand.most_common(15): print(' ', v, k)
for p in spots[:20]: print('  at', p.tolist())

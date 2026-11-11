"""Marktkirche -- Diagnose: Strahlen von aussen auf Kugeln um Punkte; Rueckseiten-Ersttreffer -> zugehoerige
Zusammenhangskomponente (Objekt, Bounds, Flaechenzahl, wasserdicht?, Volumen-Vorzeichen).
Aufruf: python mk_backdiag.py <model.glb> 'x,y,z,r;x,y,z,r' [rays]"""
import sys
import numpy as np, trimesh
from collections import Counter
from trimesh.ray.ray_pyembree import RayMeshIntersector

path = sys.argv[1]
spots = [tuple(map(float, s.split(','))) for s in sys.argv[2].split(';')]
NR = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
sc = trimesh.load(path, force='scene')
dump = sc.dump(concatenate=False)
names = [m.metadata.get('name', '?') for m in dump]
off = np.cumsum([0] + [len(m.faces) for m in dump])
M = trimesh.util.concatenate(dump)
it = RayMeshIntersector(M)
import json, os
from shapely.geometry import Polygon
from shapely import contains_xy
HERE = os.path.dirname(os.path.abspath(__file__))
_rep = json.load(open(os.path.join(HERE, '..', '..', '..', 'data', 'landmarks_replace', 'marktkirche.json')))
FP = Polygon(_rep['footprints'][0]).buffer(0.3)
rng = np.random.default_rng(11)
hits = Counter()
for (x, y, z, r) in spots:
    C = np.array([x, y, z])
    T = C + (rng.random((NR, 3)) * 2 - 1) * r
    d = rng.standard_normal((NR, 3)); d[:, 1] = np.abs(d[:, 1]) * 0.8 + 0.05; d /= np.linalg.norm(d, axis=1, keepdims=True)
    O = T + d * (3 + 9 * rng.random(NR))[:, None]
    keep = ~contains_xy(FP, O[:, 0], O[:, 2]); O = O[keep]; T = T[keep]
    D = T - O; D /= np.linalg.norm(D, axis=1, keepdims=True)
    tri, ray, loc = it.intersects_id(O, D, multiple_hits=False, return_locations=True)
    back = (M.face_normals[tri] * D[ray]).sum(1) > 0.02
    for t in tri[back]: hits[int(t)] += 1
print('backface first hits', sum(hits.values()), 'distinct tris', len(hits))
comp_cache = {}
seen = set()
for t, c in hits.most_common():
    oi = int(np.searchsorted(off, t, side='right') - 1); lt = t - off[oi]
    m = dump[oi]
    if oi not in comp_cache:
        mm = m.copy(); mm.merge_vertices(merge_tex=True, merge_norm=True)
        comps = trimesh.graph.connected_components(mm.face_adjacency, nodes=np.arange(len(mm.faces)), min_len=1)
        lab = np.full(len(mm.faces), -1)
        for k, cc in enumerate(comps): lab[cc] = k
        comp_cache[oi] = (mm, comps, lab)
    mm, comps, lab = comp_cache[oi]
    k = lab[lt]
    if (oi, k) in seen: continue
    seen.add((oi, k))
    sub = mm.submesh([comps[k]], append=True)
    print('%-28s hits %3d comp %5d faces %6d watertight %s vol %.4f bounds %s / %s  tri %s' % (
        names[oi], c, k, len(sub.faces), sub.is_watertight, sub.volume if sub.is_watertight else float('nan'),
        np.round(sub.bounds[0], 2).tolist(), np.round(sub.bounds[1], 2).tolist(), np.round(m.triangles_center[lt], 2).tolist()))

"""Marktkirche -- gezielte Pruefung kritischer Stellen (Spielframe): Strahlen von aussen (obere Halbkugel, 3-12 m) auf Kugeln um
Pruefpunkte. Meldet je Punkt (a) Rueckseiten als ersten Treffer (Loch / verdrehte Normale) und (b) gestapelte Flaechen
verschiedener Objekte (erste zwei Treffer < thr, beide Vorderseiten, parallel) = Z-Fighting-Gefahr.
Aufruf: python mk_spotcheck.py <model.glb> [rays_je_punkt] [thr_m]"""
import sys, json
import numpy as np, trimesh
from collections import Counter
from trimesh.ray.ray_pyembree import RayMeshIntersector

path = sys.argv[1]; NR = int(sys.argv[2]) if len(sys.argv) > 2 else 40000; THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.012
SPOTS = {   # Name: (x, y, z, Radius) im Spielframe
    'Schwelle Suedportal': (-220.8, 0.9, 305.6, 1.6), 'Schwelle Nordportal': (-254.5, 1.05, 283.9, 1.6),
    'Suedportal Bogen': (-220.8, 4.5, 305.9, 2.2), 'Westportal Vorhalle': (-269.6, 2.5, 300.4, 2.6),
    'Westportal Figuren': (-272.1, 6.6, 299.5, 3.6), 'Laterne': (-262.4, 72.5, 299.2, 2.6),
    'Turmecke SW Gurt': (-269.8, 30.1, 308.3, 1.2), 'Turmecke NO Gurt': (-254.3, 40.3, 290.3, 1.2),
    'Turmecke NW Gurt': (-270.0, 19.8, 292.0, 1.2), 'Sohlbank Nordfenster': (-245.0, 6.0, 286.0, 2.0),
    'Sohlbank Suedfenster': (-235.0, 6.0, 311.4, 2.0), 'Lueftungsziegel': (-252.0, 24.0, 287.5, 3.0),
    'Uhr Sued': (-261.0, 56.8, 307.3, 2.8), 'Lutherdenkmal': (-258.25, 3.5, 311.06, 1.8),
    'Reliefs Nord': (-237.0, 3.4, 284.5, 1.2), 'Barock-Epitaph': (-270.0, 5.0, 304.3, 1.6),
}
sc = trimesh.load(path, force='scene')
dump = sc.dump(concatenate=False)
names = [m.metadata.get('name', '?') for m in dump]
off = np.cumsum([0] + [len(m.faces) for m in dump])
M = trimesh.util.concatenate(dump)
it = RayMeshIntersector(M)
import os
from shapely.geometry import Polygon
from shapely import contains_xy as contains
HERE = os.path.dirname(os.path.abspath(__file__))
_rep = json.load(open(os.path.join(HERE, '..', '..', '..', 'data', 'landmarks_replace', 'marktkirche.json')))
FP = Polygon(_rep['footprints'][0]).buffer(0.3)
PL = json.load(open(os.path.join(HERE, 'build', 'plan_local.json'))); GG = PL['ground_grid']; GY = np.array(GG['y'])
_O, _U, _V = np.array(PL['origin']), np.array(PL['U']), np.array(PL['V'])


def ground(x, z):
    q = np.c_[x - _O[0], z - _O[1]]; u = q @ _U; v = q @ _V      # U, V orthonormal (Spiel x/z)
    fu = np.clip((u - GG['u0']) / GG['d'], 0, GG['nu'] - 1.001); fv = np.clip((v - GG['v0']) / GG['d'], 0, GG['nv'] - 1.001)
    j = np.floor(fu).astype(int); i = np.floor(fv).astype(int); b = fu - j; a = fv - i
    return (1 - a) * (1 - b) * GY[i, j] + (1 - a) * b * GY[i, j + 1] + a * (1 - b) * GY[i + 1, j] + a * b * GY[i + 1, j + 1]


rng = np.random.default_rng(3)
res = {}
tot_b = tot_z = 0
for name, (x, y, z, r) in SPOTS.items():
    C = np.array([x, y, z])
    T = C + (rng.random((NR, 3)) * 2 - 1) * r
    d = rng.standard_normal((NR, 3)); d[:, 1] = np.abs(d[:, 1]) * 0.8 + 0.05; d /= np.linalg.norm(d, axis=1, keepdims=True)
    O = T + d * (3 + 17 * rng.random(NR))[:, None]
    # nur Quellen im Freien (ausserhalb des Kirchengrundrisses + 0.3 m) und Ziele ueber dem Gelaende
    keep = ~contains(FP, O[:, 0], O[:, 2]) & (T[:, 1] > ground(T[:, 0], T[:, 2]) + 0.03)
    O = O[keep]; T = T[keep]
    D = T - O; D /= np.linalg.norm(D, axis=1, keepdims=True)
    tri, ray, loc = it.intersects_id(O, D, multiple_hits=True, max_hits=3, return_locations=True)
    dist = np.linalg.norm(loc - O[ray], axis=1)
    o = np.lexsort((dist, ray)); tri, ray, loc, dist = tri[o], ray[o], loc[o], dist[o]
    first = np.where(np.r_[True, ray[1:] != ray[:-1]])[0]
    fn = M.face_normals
    above = loc[:, 1] > ground(loc[:, 0], loc[:, 2]) + 0.02
    first = first[above[first]]
    back = [i for i in first if fn[tri[i]] @ D[ray[i]] > 0.02]
    zc = Counter(); spots = []
    for i in first:
        if i + 1 >= len(ray) or ray[i + 1] != ray[i] or dist[i + 1] - dist[i] > THR: continue
        a, b = tri[i], tri[i + 1]
        if fn[a] @ D[ray[i]] > 0 or fn[b] @ D[ray[i]] > 0 or fn[a] @ fn[b] < 0.995: continue
        na = names[np.searchsorted(off, a, side='right') - 1]; nb = names[np.searchsorted(off, b, side='right') - 1]
        if na == nb: continue
        zc[(na, nb)] += 1; spots.append(np.round(loc[i], 2).tolist())
    bo = Counter(names[np.searchsorted(off, tri[i], side='right') - 1] for i in back)
    res[name] = dict(hits=len(first), backface=len(back), backface_by=dict(bo), stacked=sum(zc.values()),
                     stacked_by={' / '.join(k): v for k, v in zc.items()}, stacked_at=spots[:5],
                     back_at=[np.round(loc[i], 2).tolist() for i in back[:5]])
    tot_b += len(back); tot_z += sum(zc.values())
    print('%-22s hits %6d  backface %4d  stacked(<%.0f mm) %4d  %s %s' % (name, len(first), len(back), THR * 1000, sum(zc.values()),
                                                                     dict(zc.most_common(3)), res[name]['back_at'][:2]))
print('TOTAL backface', tot_b, 'stacked', tot_z)
json.dump(res, open(path + '.spotcheck.json', 'w'), indent=1)

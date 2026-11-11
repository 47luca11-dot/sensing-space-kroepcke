"""z-fighting finder: coplanar, same-facing, overlapping triangles (overlap area > 2 cm^2) across the whole landmark.
Coarse buckets (normal ~1 deg, plane offset 1 cm in a local frame, incl. the neighbouring offset bin), then every
candidate pair is verified locally: all vertices of one triangle within 2 mm of the other's plane (this removes the
false positives a global n.p offset gives for long thin float32 triangles 500 m from the origin).
Run: python zfight.py"""
import os, json, collections
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, 'build', 'anzeiger_meta.json')))
npz = np.load(os.path.join(HERE, 'build', 'anzeiger_parts.npz'))
T, tag = [], []
for o, od in meta['objects'].items():
    V = npz['V_' + o].astype(np.float64).reshape(-1, 3, 3); M = npz['M_' + o]
    T.append(V); tag += [(o, od['materials'][m]) for m in M]
T = np.concatenate(T)
C0 = T.reshape(-1, 3).mean(0)
TL = T - C0
n = np.cross(TL[:, 1] - TL[:, 0], TL[:, 2] - TL[:, 0]); A = np.linalg.norm(n, axis=1) / 2; n /= np.maximum(A * 2, 1e-12)[:, None]
cen = TL.mean(1)
d = np.einsum('ij,ij->i', n, cen)
nk = np.round(n * 60).astype(np.int64)
dk = np.floor(d / 0.01).astype(np.int64)
buckets = collections.defaultdict(list)
for i in np.where(A > 2e-5)[0]:
    buckets[(nk[i, 0], nk[i, 1], nk[i, 2], dk[i])].append(i)
seen = set(); issues = []
for k, idx in list(buckets.items()):
    cand = list(idx) + buckets.get(k[:3] + (k[3] + 1,), [])
    if len(cand) < 2: continue
    nn = n[idx[0]]
    a = np.cross(nn, [0, 1, 0] if abs(nn[1]) < 0.9 else [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(nn, a)
    polys = [Polygon([(p @ a, p @ b) for p in TL[i]]) for i in cand]
    tr = STRtree(polys)
    for ii, p in enumerate(polys):
        for jj in tr.query(p):
            if jj <= ii: continue
            I, J = cand[ii], cand[jj]
            if (I, J) in seen: continue
            seen.add((I, J))
            if n[I] @ n[J] < 0.9995: continue
            big, sm = (I, J) if A[I] >= A[J] else (J, I)
            dist = np.abs((TL[sm] - TL[big][0]) @ n[big]).max()
            if dist > 0.002: continue
            inter = p.intersection(polys[jj]).area
            if inter > 2e-4:
                c3 = TL[big].mean(0) + C0
                issues.append((inter, c3, tag[I], tag[J], n[I]))
agg = collections.defaultdict(lambda: [0.0, None, None])
for inter, c3, t1, t2, nn in issues:
    key = (tuple(np.round(c3 / 2.0).astype(int)), t1, t2)
    agg[key][0] += inter; agg[key][1] = c3; agg[key][2] = nn
rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
print('z-fight pairs %d, total overlap m2 %.3f' % (len(issues), sum(i[0] for i in issues)))
for (cell, t1, t2), (ov, c3, nn) in rows[:30]:
    print('  %.4f m2 at' % ov, np.round(c3, 2).tolist(), 'n', np.round(nn, 2).tolist(), t1, t2)

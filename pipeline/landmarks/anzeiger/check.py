"""Visibility / winding check for the Anzeiger landmark (game renders FrontSide only).
Casts rays from exterior viewpoints (street level ring, elevated ring, aerial) to sampled surface points; the first hit
must be a front face (dot(ray, normal) < 0). Back-face first hits = visible holes in the game.
Also reports open boundary edges per object (informational: shells are open toward the ground / walls behind piers).
Run: python check.py   (reads build/anzeiger_parts.npz)"""
import os, sys, json
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, 'build', 'anzeiger_meta.json')))
npz = np.load(os.path.join(HERE, 'build', 'anzeiger_parts.npz'))

V_all, F_all, tag = [], [], []
off = 0
names = []
for oi, (o, od) in enumerate(meta['objects'].items()):
    V = npz['V_' + o].astype(np.float64); M = npz['M_' + o]
    n = len(V) // 3
    V_all.append(V); F_all.append(np.arange(off, off + 3 * n).reshape(-1, 3)); off += 3 * n
    tag.append(np.stack([np.full(n, oi), M], 1)); names.append((o, od['materials']))
V = np.concatenate(V_all); F = np.concatenate(F_all); TAG = np.concatenate(tag)
NL = len(F)                       # landmark faces come first; LoD2 neighbours appended as occluders
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
rep = json.load(open(ROOT + 'data/landmarks_replace/anzeiger.json'))
L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))
fps = [Polygon(f) for f in rep['footprints']]
occ = []
for bld in L['buildings']:
    g = np.array(bld['grounds'][0]['ext'])
    if np.hypot(g[:, 0].mean() + 510, -g[:, 1].mean() + 310) > 160: continue
    fps.append(Polygon([(p[0], -p[1]) for p in g]).buffer(0))
    if bld['id'] in rep['lod2_ids']: continue
    for part in bld['walls'] + bld['roofs']:
        P = np.array([[p[0], p[2] - 56.16, -p[1]] for p in part['ext']])
        if len(P) < 3: continue
        tri = trimesh.creation.triangulate_polygon if False else None
        # fan triangulation is fine for occlusion (LoD2 parts are planar, mostly convex)
        for i in range(1, len(P) - 1): occ.append([P[0], P[i], P[i + 1]])
occ = np.array(occ)
FOOT = unary_union(fps).buffer(0.5)
if len(occ):
    Vo = occ.reshape(-1, 3); Fo = np.arange(len(Vo)).reshape(-1, 3) + len(V)
    V = np.concatenate([V, Vo]); F = np.concatenate([F, Fo])
mesh = trimesh.Trimesh(V, F, process=False)
N = mesh.face_normals
print('faces', len(F))
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    ri = RayMeshIntersector(mesh)
except Exception:
    ri = mesh.ray

rng = np.random.default_rng(3)
lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2
cams = []
for r, h in ((25, 1.7), (45, 1.7), (70, 15), (90, 45), (60, 80), (140, 30)):
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        cams.append([c[0] + r * np.cos(a), h, c[2] + r * np.sin(a)])
cams = [q for q in cams if not FOOT.contains(Point(q[0], q[2])) or q[1] > 60]
# explicit QA viewpoints: Goseriede walking south (NE corner slit), light well from SW, Stiftstrasse parking, portals
cams += [[-490.0, 1.7, -378.0], [-484.0, 1.7, -352.0], [-532.0, 28.0, -282.0], [-545.0, 10.0, -275.0], [-511.0, 45.0, -286.0],
         [-541.0, 0.9, -322.0], [-535.0, 1.0, -327.0], [-481.5, 1.6, -285.5], [-479.0, 1.6, -268.0], [-479.5, 1.6, -297.5],
         [-489.0, 0.2, -346.0], [-565.0, 30.0, -245.0], [-560.0, 2.0, -390.0],
         # grazing views along the Goseriede front (setback returns: Kestner glass <-> north wing / right wing, recess)
         [-489.5, 1.7, -306.0], [-492.0, 1.7, -362.0], [-492.5, 3.0, -322.0], [-484.0, 1.7, -312.0], [-486.5, 1.7, -280.0]]
cams = np.array(cams)
print('cameras', len(cams))
# drop cameras inside the footprints (rough: inside bbox and below roof)
area = mesh.area_faces[:NL]
pts_idx = rng.choice(NL, 40000, p=area / area.sum())
bary = rng.random((40000, 2)); m = bary.sum(1) > 1; bary[m] = 1 - bary[m]
P = V[F[pts_idx, 0]] + (V[F[pts_idx, 1]] - V[F[pts_idx, 0]]) * bary[:, :1] + (V[F[pts_idx, 2]] - V[F[pts_idx, 0]]) * bary[:, 1:]
bad = {}
CAMS = {}
SEE = []
tot = 0
for cam in cams:
    sel = rng.choice(len(P), 4000, replace=False)
    D = P[sel] - cam; D /= np.linalg.norm(D, axis=1, keepdims=True)
    O = np.repeat(cam[None], len(D), 0)
    loc, ir, it = ri.intersects_location(O, D, multiple_hits=False)
    if len(it) == 0: continue
    keep = it < NL
    ir, it = ir[keep], it[keep]
    dots = np.einsum('ij,ij->i', D[ir], N[it])
    b = dots > 1e-4
    tot += len(it)
    lb = loc[keep][b] if len(loc) else loc
    for (f, r_), l_ in zip(zip(it[b], ir[b]), lb):
        k = (names[TAG[f, 0]][0], names[TAG[f, 0]][1][TAG[f, 1]])
        bad.setdefault(k, []).append(f)
        # real see-through? (no front face of the landmark within 12 cm behind the back face; a front face right
        # behind = covered sliver, e.g. a roof edge tucked under a soffit)
        o2 = l_ + D[r_] * 1e-3
        L2, I2, T2 = ri.intersects_location([o2], [D[r_]], multiple_hits=True)
        cov = any(t3 < NL and N[t3] @ D[r_] < 0 and np.linalg.norm(l3 - o2) < 0.12 for l3, t3 in zip(L2, T2))
        if not cov: SEE.append((k, tuple(np.round(cam, 1)), tuple(np.round(l_, 2))))
        CAMS.setdefault(k, []).append((tuple(np.round(cam, 1)), tuple(np.round(P[sel][r_], 2))))
nb = sum(len(v) for v in bad.values())
print('rays hit %d, back-face first hits %d (%.3f%%), real see-through (no front face within 12 cm behind) %d' % (tot, nb, 100.0 * nb / max(tot, 1), len(SEE)))
for k_, c_, p_ in SEE[:30]: print('   SEE', k_, 'cam', c_, 'at', p_)
for k, v in sorted(bad.items(), key=lambda kv: -len(kv[1])):
    fs = np.unique(v)
    cen = V[F[fs]].mean(1)
    print('  ', k, 'hits', len(v), 'faces', len(fs), 'e.g. at', np.round(cen[:3], 2).tolist())

if '--debug' in sys.argv:
    import anz, bad as badm
    for k, v in sorted(bad.items(), key=lambda kv: -len(kv[1]))[:6]:
        fs, cnt = np.unique(v, return_counts=True)
        order = np.argsort(-cnt)[:5]
        print(k)
        for f in fs[order]:
            cen = V[F[f]].mean(0)
            pass
        import collections
        cc = collections.Counter([c_ for c_, _ in CAMS[k]])
        print('    cams', cc.most_common(4))
        print('    pairs', CAMS[k][:4])
        for f in fs[order]:
            cen = V[F[f]].mean(0)
            print('    face', f, 'cen', np.round(cen, 2), 'n', np.round(N[f], 2), 'u/e', np.round(anz.FE.W(cen) if hasattr(anz.FE, 'W') else 0, 2) if False else '', 'area', round(mesh.area_faces[f], 3))

# ---------------------------------------------------------------- buried letters: every front face of the sign / facade
# letters must see free space in front of it (a ray along its normal must not hit a back face = inside a wall/pier)
LET = [i for i, (o, mats) in enumerate(names) if 'letters' in o]
sel = np.where(np.isin(TAG[:NL, 0], LET))[0]
Nl = N[sel]; front = sel[(np.abs(Nl[:, 1]) < 0.3) & (mesh.area_faces[sel] > 2e-4)]      # skip earcut slivers (unreliable normals)
Fv = V[F[front]]; cen = Fv.mean(1) + N[front] * 0.002
loc, ir, it = ri.intersects_location(cen, N[front], multiple_hits=False)
buried = 0; where = []
for k, r in enumerate(ir):
    if it[k] < NL and np.dot(N[front[r]], N[it[k]]) > 0 and np.linalg.norm(loc[k] - cen[r]) < 3.0:
        buried += 1; where.append((np.round(cen[r], 2).tolist(), np.round(N[front[r]], 2).tolist(), names[TAG[it[k], 0]][0], names[TAG[it[k], 0]][1][TAG[it[k], 1]], round(float(np.linalg.norm(loc[k] - cen[r])), 3)))
print('letters: %d front faces tested, %d buried (inside a wall)' % (len(front), buried), where[:5])
if buried: sys.exit(2)

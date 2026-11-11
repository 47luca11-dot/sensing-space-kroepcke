"""Marktkirche -- Mesh-Hygiene + koplanare Flaechen verschiedener Objekte (geometrisch, nicht strahlbasiert).
Meldet je Objekt: Dreiecke, degenerierte Dreiecke (<1e-7 m2), Null-UV-Flaechen (Flaeche > 0.01 m2, UV-Flaeche ~0),
nicht-mannigfaltige Kanten (>2 Flaechen); danach koplanare, gleichgerichtete, ueberlappende Flaechen verschiedener
Objekte (Abstand < THR) oberhalb des Gelaendes. Aufruf: python mk_hygiene.py <model.glb> [thr_m=0.004] [--verbose]"""
import sys, json, os
import numpy as np, trimesh
from collections import Counter, defaultdict
from shapely.geometry import Polygon
from shapely.strtree import STRtree

path = sys.argv[1]
THR = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 0.004
VERB = '--verbose' in sys.argv
sc = trimesh.load(path, force='scene')
dump = sc.dump(concatenate=False)
names = [m.metadata.get('name', '?') for m in dump]
tot = 0
rep = {}
for m, n in zip(dump, names):
    a = m.area_faces
    deg = int((a < 1e-7).sum())
    uv = getattr(m.visual, 'uv', None)
    zuv = 0; zuv_area = 0.0
    if uv is not None and len(uv) == len(m.vertices):
        t = uv[m.faces]
        ua = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
        bad = (ua < 1e-9) & (a > 0.01)
        zuv = int(bad.sum()); zuv_area = float(a[bad].sum())
    # nicht-mannigfaltige Kanten (nach Zusammenfuehren gleicher Positionen)
    mm = m.copy(); mm.merge_vertices(merge_tex=True, merge_norm=True)
    e = np.sort(mm.edges, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    nm = int((cnt > 2).sum())
    rep[n] = dict(tris=len(m.faces), degenerate=deg, zero_uv=zuv, zero_uv_m2=round(zuv_area, 2), nonmanifold=nm)
    tot += len(m.faces)
    if deg or zuv or nm:
        print('%-34s tris %7d  degen %4d  zeroUV %5d (%.1f m2)  nonmanifold %4d' % (n, len(m.faces), deg, zuv, zuv_area, nm))
print('TOTAL tris', tot, 'degenerate', sum(r['degenerate'] for r in rep.values()), 'zeroUV', sum(r['zero_uv'] for r in rep.values()),
      'nonmanifold', sum(r['nonmanifold'] for r in rep.values()))

# ---------------------------------------------------------------- koplanare Flaechen verschiedener Objekte
V = []; F = []; O = []; off = 0
for k, m in enumerate(dump):
    V.append(m.vertices); F.append(m.faces + off); O.append(np.full(len(m.faces), k)); off += len(m.vertices)
V = np.vstack(V).astype(np.float64); F = np.vstack(F); O = np.concatenate(O)
V0 = V.mean(axis=0); VL = V - V0          # lokaler Ursprung: Ebenenabstaende gut konditioniert
T = VL[F]
N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); A = np.linalg.norm(N, axis=1); ok = A > 2e-6
N[ok] /= A[ok][:, None]
D = np.einsum('ij,ij->i', N, T[:, 0])
ymin = T[:, :, 1].min(axis=1)
idx = np.where(ok & (ymin > 0.15))[0]          # Gelaende-/Fundamentflaechen ignorieren
# Bins: Normalenrichtung (0.02) und Ebenenabstand (THR); Nachbarbins in d mitpruefen
kn = np.round(N[idx] * 50).astype(np.int64)
DB = 0.03                                  # grobe Bins (Normalenrauschen kleiner Dreiecke), Feinpruefung lokal
kd = np.floor(D[idx] / DB).astype(np.int64)
bins = defaultdict(list)
for j, i in enumerate(idx):
    bins[(kn[j, 0], kn[j, 1], kn[j, 2], kd[j])].append(i)
from trimesh.ray.ray_pyembree import RayMeshIntersector
MALL = trimesh.Trimesh(V, F, process=False); RI = RayMeshIntersector(MALL)
_rng = np.random.default_rng(5)
def visible(p, n):
    # Halbkugel-Strahlen ab p + n*1cm; sichtbar, wenn mind. einer frei entkommt
    # (Strahlen nach unten zaehlen nicht: das Innere ist unten offen, dort liegt im Spiel das Gelaende)
    d = _rng.standard_normal((96, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[d @ n < 0] *= -1; d = 0.5 * d + 0.5 * n; d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.repeat((p + V0 + n * 0.01)[None], len(d), 0)
    h = RI.intersects_any(o, d)
    return int(((~h) & (d[:, 1] > 0.02)).sum())
hits = Counter(); where = defaultdict(list); seen = set(); hidden = Counter(); hwhere = defaultdict(list)
for key, lst in bins.items():
    cand = list(lst)
    nb = bins.get((key[0], key[1], key[2], key[3] + 1))
    if nb: cand = cand + nb
    objs = set(O[cand])
    if len(objs) < 2: continue
    cand = np.array(cand)
    # 2D-Projektion in die Ebene
    n0 = N[cand[0]]; ax = np.cross(n0, [0, 1, 0] if abs(n0[1]) < 0.9 else [1, 0, 0]); ax /= np.linalg.norm(ax); ay = np.cross(n0, ax)
    polys = [Polygon(np.c_[T[i] @ ax, T[i] @ ay]) for i in cand]
    tree = STRtree(polys)
    for a_, pa in enumerate(polys):
        ia = cand[a_]
        for b_ in tree.query(pa):
            ib = cand[b_]
            if ib <= ia or O[ia] == O[ib]: continue
            if N[ia] @ N[ib] < 0.999: continue
            ca, cb = T[ia].mean(axis=0), T[ib].mean(axis=0)
            sep = 0.5 * (abs(N[ia] @ (cb - T[ia, 0])) + abs(N[ib] @ (ca - T[ib, 0])))
            if sep > THR: continue
            pr = (ia, ib)
            if pr in seen: continue
            seen.add(pr)
            try:
                ar = pa.intersection(polys[b_]).area
            except Exception:
                continue
            if ar < 2e-4: continue
            k2 = tuple(sorted((names[O[ia]], names[O[ib]])))
            ip = pa.intersection(polys[b_]).representative_point()
            p3 = ax * ip.x + ay * ip.y + n0 * (T[ia, 0] @ n0)
            if visible(p3, N[ia]) == 0:
                hidden[k2] += ar
                if len(hwhere[k2]) < 3: hwhere[k2].append([round(float(x), 2) for x in p3 + V0] + [round(float(sep) * 1000, 1)])
                continue
            hits[k2] += ar
            if len(where[k2]) < 6: where[k2].append([round(float(x), 2) for x in T[ia].mean(axis=0) + V0] + [round(float(sep) * 1000, 1)])
print('hidden coplanar (occluded) m2 %.3f' % sum(hidden.values()))
if VERB:
    for k, v in hidden.most_common(40): print('  hidden %.3f m2  %s / %s  at %s' % (v, k[0], k[1], hwhere[k]))
print('VISIBLE coplanar (<%.1f mm, same facing, overlap) pairs:' % (THR * 1000), len(hits), 'area m2 %.3f' % sum(hits.values()))
for k, v in hits.most_common(30):
    print('  %.3f m2  %s / %s  at %s' % (v, k[0], k[1], where[k][:3]))
json.dump(dict(objects=rep, coplanar={' / '.join(k): dict(m2=v, at=where[k]) for k, v in hits.items()}),
          open(path + '.hygiene.json', 'w'), indent=1)

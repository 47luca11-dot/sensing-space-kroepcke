"""Klassen-Polygone -> gemeinsame, lueckenlose Triangulation (Triangle, CDT + Flaechen-/Fehlerverfeinerung),
Hoehen je Klasse (Bordsteine, Fasen, abgesenkte Borde an Uebergaengen, Naht zum Detailmodell),
senkrechte Stufenwaende an Hoehenspruengen, Schuerzen an Raendern.
Ausgabe: data/ground_cache/terrain.npz + terrain_parts.pkl"""
import sys, json, time, pickle, math
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, MultiLineString, Point, box
from shapely.ops import unary_union
import triangle as tr
from scipy.spatial import cKDTree
from common import *
import geom as G

T0 = time.time()
def log(*a): print('[tmesh %6.1fs]' % (time.time() - T0), *a, flush=True)

L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
CL = L['classes']; HOLES = L['holes']
bz = np.load(CACHE + 'base.npz'); BG = Grid(bz['B'].astype(np.float64), float(bz['x0']), float(bz['z0']), 1.0)

# ---------------------------------------------------------------- Klassen
#            name        Niveau   Material        UV-Massstab  Richtung  max. Flaeche
CLS = [('seam_road', 0.0, 'asphalt', 1, 0, 0.30), ('seam_setts', 0.0, 'setts_road', 1, 0, 0.30), ('seam_grass', 0.0, 'grass', 1, 0, 0.30),
       ('seam', 0.0, 'slabs_ped', 1, 0, 0.30),
       ('coping', 0.0, 'granite', 1.5, 1, 2.0), ('rim', 0.45, 'granite', 1.5, 0, 1.5), ('water', 0.30, 'water', 4.0, 0, 6.0),
       ('hbs', 0.72, 'concrete', 2.0, 1, 3.0), ('pitframe', 0.0, 'granite', 1.0, 1, 1.0), ('treepit', -0.03, 'grate', 1.0, 1, 1.0),
       ('kerb_ch', 0.0, 'kerb', 2.0, 2, 3.0), ('kerb', 0.0, 'kerb', 2.0, 2, 3.0), ('gutter', -0.14, 'setts_small', 2.5, 2, 3.0),
       ('setts_road', -0.14, 'setts_road', 4.0, 1, 3.0), ('track', -0.14, 'setts_road', 4.0, 1, 3.0), ('parking', -0.14, 'asphalt2', 2.0, 0, 4.0),
       ('road', -0.14, 'asphalt', 2.0, 0, 3.0), ('bed', -0.04, 'soil', 2.0, 0, 4.0), ('gravel', -0.02, 'gravel', 2.0, 0, 4.0),
       ('path', 0.0, 'pavers', 3.0, 1, 3.0), ('edge', 0.0, 'granite', 1.0, 2, 3.0), ('grass', -0.03, 'grass', 2.0, 0, 5.0),
       ('cycle', 0.0, 'asphalt_red', 2.0, 0, 3.0), ('island', 0.0, 'pavers', 3.0, 1, 3.0), ('railplat', 0.0, 'concrete', 2.0, 1, 8.0),
       ('ballast', 0.0, 'ballast', 2.5, 0, 8.0), ('ped', 0.0, 'slabs_ped', 4.8, 1, 3.0), ('sidewalk', 0.0, 'pavers', 3.0, 1, 3.0),
       ('river', 0.0, 'river', 6.0, 0, 40.0), ('alt_setts', 0.0, 'setts_arc', 4.0, 1, 3.0), ('clinker', 0.0, 'clinker', 3.0, 1, 3.0)]
NAMES = [c[0] for c in CLS]; CID = {n: i for i, n in enumerate(NAMES)}
LEVEL = np.array([c[1] for c in CLS]); CMAT = [c[2] for c in CLS]; CUV = np.array([c[3] for c in CLS], float)
CDIR = np.array([c[4] for c in CLS]); CAREA = np.array([c[5] for c in CLS])
LOWC = np.array([CID[n] for n in ('road', 'gutter', 'parking', 'setts_road', 'track')])
RAISED = np.array([CID[n] for n in ('kerb', 'kerb_ch', 'sidewalk', 'ped', 'cycle', 'island', 'edge', 'pitframe', 'treepit', 'path', 'grass', 'bed', 'gravel', 'alt_setts', 'clinker')])
SEAMC = np.array([CID[n] for n in ('seam_road', 'seam_setts', 'seam_grass', 'seam')])
SEAMGRP = {'seam_road': (1, 2), 'seam_setts': (1, 2), 'seam_grass': (3,), 'seam': (0, 4)}
WALLMAT = {'rim': 'granite', 'hbs': 'concrete', 'coping': 'granite', 'water': 'granite'}
RIVER_C = CID['river']
_rj = json.load(open(CACHE + 'river.json'))['rivers']
_wl = np.array([r['wl'] for r in _rj]); _wmed = float(np.median(_wl)) if len(_wl) else -7.4
RIVER_LV = [(Polygon(r['poly']), min(r['wl'], _wmed + 0.4)) for r in _rj]

# ---------------------------------------------------------------- PSLG
def rings_of(g):
    out = []
    for p in getattr(g, 'geoms', [g]):
        if p.is_empty or p.geom_type != 'Polygon': continue
        out.append(p.exterior); out += list(p.interiors)
    return out

lines = []
for n, g in CL.items(): lines += rings_of(g)
for h in HOLES.values(): lines.append(h['poly'].exterior)
lines += rings_of(L['region'])
# Gebaeude-Inneres (1 m eingerueckt) als eigene Flaechen fuer grobe Vermaschung
BLDIN = L['bld'].buffer(-1.0).simplify(0.3).intersection(L['region'].buffer(-0.5))
BLDIN = BLDIN.difference(unary_union([h['poly'].buffer(1) for h in HOLES.values()]))
BLDIN = shapely.make_valid(BLDIN)
BLDIN = unary_union([p for p in getattr(BLDIN, 'geoms', [BLDIN]) if p.geom_type == 'Polygon' and p.area > 4])
lines += rings_of(BLDIN)
log('rings', len(lines))
ML = MultiLineString([np.asarray(r.coords) for r in lines])
noded = shapely.unary_union(ML, grid_size=0.002)          # Snap-Rounding: robust, alle Knoten auf 2-mm-Raster
noded = shapely.segmentize(noded, 2.0)
log('noded')
segs = []
for ls in getattr(noded, 'geoms', [noded]):
    c = np.asarray(ls.coords)
    if len(c) >= 2: segs.append(np.stack([c[:-1], c[1:]], 1))
segs = np.concatenate(segs)
P = segs.reshape(-1, 2)
key = np.round(P * 2000).astype(np.int64)
uk, inv = np.unique(key, axis=0, return_inverse=True)
V0 = uk / 2000.0
S0 = inv.reshape(-1, 2)
S0 = S0[S0[:, 0] != S0[:, 1]]
S0 = np.unique(np.sort(S0, 1), axis=0)
log('pslg vertices', len(V0), 'segments', len(S0))
hole_pts = [list(h['poly'].representative_point().coords[0]) for h in HOLES.values()]
hole_pts.append(list(L['sq_in'].representative_point().coords[0]))

# ---------------------------------------------------------------- Klassen-Zuordnung je Dreieck
prep = {}
for n in NAMES:
    if n in CL and not CL[n].is_empty:
        g = CL[n]; shapely.prepare(g); prep[n] = g
shapely.prepare(BLDIN)

def label(V, T):
    c = V[T].mean(1)
    lab = np.full(len(T), -1)
    for n in NAMES:
        if n not in prep: continue
        todo = np.nonzero(lab < 0)[0]
        if len(todo) == 0: break
        ins = shapely.contains_xy(prep[n], c[todo, 0], c[todo, 1])
        lab[todo[ins]] = CID[n]
    miss = np.nonzero(lab < 0)[0]
    if len(miss):
        # Nachbar-Label uebernehmen
        good = lab >= 0
        kd = cKDTree(c[good]); _, ii = kd.query(c[miss]); lab[miss] = lab[good][ii]
    return lab, c

def tri_area(V, T):
    a = V[T[:, 1]] - V[T[:, 0]]; b = V[T[:, 2]] - V[T[:, 0]]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

np.savez(CACHE + 'pslg.npz', V0=V0, S0=S0, H=np.array(hole_pts))
if '--pslg-only' in sys.argv: sys.exit(0)
t1 = tr.triangulate({'vertices': V0, 'segments': S0, 'holes': np.array(hole_pts)}, 'pYQ')
V, T = t1['vertices'], t1['triangles']
log('pass1 CDT', len(V), len(T))
for it in range(2):
    lab, cen = label(V, T)
    amax = CAREA[lab].copy()
    inb = shapely.contains_xy(BLDIN, cen[:, 0], cen[:, 1])
    amax[inb & ~np.isin(lab, SEAMC)] = 60.0
    if it == 1:
        # Fehlergetrieben: Abweichung Mittelpunkt/Kantenmitten gegen B
        hv = BG(V[:, 0], V[:, 1])
        err = np.abs(hv[T].mean(1) - BG(cen[:, 0], cen[:, 1]))
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            m = (V[T[:, a_]] + V[T[:, b_]]) / 2
            err = np.maximum(err, np.abs((hv[T[:, a_]] + hv[T[:, b_]]) / 2 - BG(m[:, 0], m[:, 1])))
        ar = tri_area(V, T)
        big = err > 0.025
        amax[big] = np.minimum(amax[big], ar[big] / 3)
        log('error refine', int(big.sum()))
    t2 = tr.triangulate({'vertices': V, 'segments': t1['segments'] if it == 0 else t2['segments'], 'triangles': T, 'triangle_max_area': amax}, 'rpYaQ')
    V, T = t2['vertices'], t2['triangles']
    log('pass', it + 2, len(V), len(T))
SEGS = t2['segments']
lab, cen = label(V, T)
nT = len(T)
log('labels', {NAMES[i]: int((lab == i).sum()) for i in np.unique(lab)})

# ---------------------------------------------------------------- Kanten / Nachbarschaft
E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
Etri = np.tile(np.arange(nT), 3); Ecor = np.repeat([0, 1, 2], nT)          # Ecke des Startpunkts
Es = np.sort(E, 1)
ek = Es[:, 0].astype(np.int64) * len(V) + Es[:, 1]
order = np.argsort(ek); eks = ek[order]
same = np.r_[eks[1:] == eks[:-1], False]
iA = order[np.nonzero(same)[0]]; iB = order[np.nonzero(same)[0] + 1]           # innere Kanten (Paar)
cnt = np.bincount(np.searchsorted(np.unique(eks), eks), minlength=0)
uniq_first = np.r_[True, eks[1:] != eks[:-1]]
single = np.ones(len(eks), bool)
single[np.nonzero(same)[0]] = False; single[np.nonzero(same)[0] + 1] = False
iBd = order[single]                                                               # Randkanten
log('inner edges', len(iA), 'boundary edges', len(iBd))

# ---------------------------------------------------------------- Hoehen je (Dreieck, Ecke)
Vx, Vz = V[:, 0], V[:, 1]
Bv = BG(Vx, Vz)
touch_low = np.zeros(len(V), bool)
lowtri = np.isin(lab, LOWC)
touch_low[np.unique(T[lowtri])] = True

# abgesenkte Borde an Fussgaengerueberwegen
osm = load_osm()
LOWG = L['low']; shapely.prepare(LOWG)
cross = []
road_lines = [(np.array(c), w, t) for c, w, t in L['road_lines']]
from shapely.strtree import STRtree
RLs = [LineString(c) for c, w, t in road_lines]; rtree = STRtree(RLs)
for n in osm['nodes']:
    t = n['tags']
    if t.get('highway') not in ('crossing', 'traffic_signals') and t.get('crossing') is None: continue
    if t.get('highway') == 'traffic_signals' and t.get('crossing') is None: continue
    p = np.array(n['p'])
    if np.hypot(*p) > RADIUS + 15: continue
    pt = Point(p)
    i = rtree.nearest(pt)
    Ls = RLs[i]
    if Ls.distance(pt) > 1.0: continue
    s = Ls.project(pt); a = np.array(Ls.interpolate(max(0, s - 1.5)).coords[0]); b = np.array(Ls.interpolate(min(Ls.length, s + 1.5)).coords[0])
    d = (b - a) / (np.linalg.norm(b - a) + 1e-9); nrm = np.array([-d[1], d[0]])
    # Ausdehnung der Fahrbahn quer
    ray = LineString([p - nrm * 25, p + nrm * 25]).intersection(LOWG)
    if ray.is_empty: continue
    segs_ = [np.array(g.coords) for g in getattr(ray, 'geoms', [ray]) if g.geom_type == 'LineString']
    best = None
    for sg in segs_:
        ss = (sg - p) @ nrm
        if ss.min() <= 0.5 and ss.max() >= -0.5: best = (ss.min(), ss.max())
    if best is None: continue
    cross.append((p, d, nrm, best[0], best[1], t.get('crossing', 'traffic_signals' if t.get('highway') == 'traffic_signals' else 'uncontrolled')))
log('crossings', len(cross))
lower = np.zeros(len(V))
if cross:
    cp = np.array([c[0] for c in cross]); kdc = cKDTree(cp)
    cand = kdc.query_ball_point(V, r=16.0)
    for vi, lst in enumerate(cand):
        if not lst: continue
        for ci in lst:
            p, d, nrm, e0, e1, _ = cross[ci]
            q = V[vi] - p; al = abs(q @ d); s = q @ nrm
            de = (e0 - s) if s < e0 else ((s - e1) if s > e1 else -1)
            if de < 0: continue
            f = np.clip((3.2 - al) / 1.0, 0, 1) * np.clip((1.8 - de) / 1.3, 0, 1)
            lower[vi] = min(lower[vi], -0.105 * f)
log('lowered kerb vertices', int((lower < -0.01).sum()))

# Naht: Detailhoehen (Streifen-Raster)
st = np.load(CACHE + 'strip.npz')
okst = ~np.isnan(st['y'])
SP = np.c_[st['x'][okst], st['z'][okst]]; SY = st['y'][okst]; SK = st['k'][okst]
kds = cKDTree(SP)
# je Nahtklasse: Hoehe der naechsten Detailprobe gleicher Belagsgruppe; unter Detailboden Minimum (bleibt darunter)
seam_hc = {}
dcov, _ = kds.query(V, k=1, distance_upper_bound=0.3)
for nm in SEAMGRP:
    c_ = CID[nm]
    vv = np.unique(T[lab == c_])
    if len(vv) == 0: continue
    msk = np.isin(SK, SEAMGRP[nm])
    if msk.sum() < 3: msk = np.ones(len(SK), bool)
    kdg = cKDTree(SP[msk]); YG = SY[msk]
    q = V[vv]
    d1, i1 = kdg.query(q, k=1, distance_upper_bound=6.0)
    near = np.where(np.isfinite(d1), YG[np.minimum(i1, len(YG) - 1)], np.nan)
    dd, ii = kds.query(q, k=24, distance_upper_bound=0.45)
    ymin = np.where(np.isfinite(dd), SY[np.minimum(ii, len(SY) - 1)], np.inf).min(1)
    covered = np.isfinite(dcov[vv])
    y = np.where(covered & np.isfinite(ymin), np.minimum(near, ymin), near)
    fb = np.isnan(y); y[fb] = Bv[vv][fb] + (-0.14 if nm in ('seam_road', 'seam_setts') else 0)
    arr = np.full(len(V), np.nan); arr[vv] = y - 0.02
    seam_hc[c_] = arr
river_h = np.full(len(V), np.nan)
rv = np.unique(T[lab == RIVER_C])
if len(rv):
    best = np.full(len(rv), 1e9); hh = np.full(len(rv), _wmed)
    for poly, wl in RIVER_LV:
        dd = shapely.distance(poly, shapely.points(V[rv]))
        m_ = dd < best; best[m_] = dd[m_]; hh[m_] = wl
    river_h[rv] = hh
log('river vertices', len(rv), 'levels', np.unique(np.round(hh, 2)) if len(rv) else None)
HC = np.zeros((nT, 3))
for k in range(3):
    v = T[:, k]
    h = Bv[v] + LEVEL[lab]
    ch = lab == CID['kerb_ch']
    h[ch] = Bv[v[ch]] + np.where(touch_low[v[ch]], -0.025, 0.0)
    r = np.isin(lab, RAISED)
    h[r] = h[r] + lower[v[r]]
    # Fase darf nach Absenkung nicht unter Fahrbahn+1.5 cm
    h[ch] = np.maximum(h[ch], Bv[v[ch]] - 0.125)
    for c_, arr in seam_hc.items():
        sm = lab == c_
        h[sm] = arr[v[sm]]
    rm = lab == RIVER_C
    h[rm] = river_h[v[rm]]
    HC[:, k] = h
log('heights done')

# ---------------------------------------------------------------- Stufenwaende zwischen Klassen
tA, tB = Etri[iA], Etri[iB]; cA, cB = Ecor[iA], Ecor[iB]
u = E[iA, 0]; w = E[iA, 1]
def corner_h(t, vtx):
    hh = HC[t]; m = T[t] == vtx[:, None]
    return (hh * m).sum(1)
hAu, hAw = corner_h(tA, u), corner_h(tA, w)
hBu, hBw = corner_h(tB, u), corner_h(tB, w)
du = hAu - hBu; dw = hAw - hBw
need = (np.maximum(np.abs(du), np.abs(dw)) > 0.002)
log('wall edges', int(need.sum()))
# alle an einem Vertex vorkommenden Hoehen (fuer T-Knoten-freie senkrechte Wandkanten)
_cv = T.ravel(); _ch = np.round(HC.ravel(), 4)
_o = np.lexsort((_ch, _cv)); _cv, _ch = _cv[_o], _ch[_o]
_keep = np.r_[True, (_cv[1:] != _cv[:-1]) | (_ch[1:] != _ch[:-1])]
_cv, _ch = _cv[_keep], _ch[_keep]
HS_start = np.searchsorted(_cv, np.arange(len(V))); HS_end = np.searchsorted(_cv, np.arange(len(V)), side='right')
def hset(v, lo, hi):
    h = _ch[HS_start[v]:HS_end[v]]
    h = h[(h > lo + 1e-4) & (h < hi - 1e-4)]
    return [lo] + h.tolist() + [hi]
wall_tris = {}
def zipper(mat, pa, pb, ca, cb, look=None, away=None):
    """Wandstreifen zwischen senkrechten Ketten ca (bei pa) und cb (bei pb), aufsteigend. Normale zu 'look' hin bzw. weg von 'away'."""
    A = [np.array([pa[0], y, pa[2]]) for y in ca]; Bq = [np.array([pb[0], y, pb[2]]) for y in cb]
    i = j = 0; tris = []
    while i < len(A) - 1 or j < len(Bq) - 1:
        if j == len(Bq) - 1 or (i < len(A) - 1 and ca[i + 1] <= cb[j + 1]):
            tris.append((A[i], Bq[j], A[i + 1])); i += 1
        else:
            tris.append((A[i], Bq[j], Bq[j + 1])); j += 1
    d = np.array([pb[0] - pa[0], 0, pb[2] - pa[2]]); n = np.array([-d[2], 0, d[0]])     # horizontale Wandnormale (eine Seite)
    mid = (np.array(pa) + np.array(pb)) / 2
    if look is not None: flip = np.dot(n[[0, 2]], np.array(look) - mid[[0, 2]]) < 0
    else: flip = np.dot(n[[0, 2]], np.array(away) - mid[[0, 2]]) > 0
    if flip: n = -n
    n = n / (np.linalg.norm(n) + 1e-12)
    L_ = np.linalg.norm(d)
    for tr_ in tris:
        P3 = np.array(tr_)
        fn = np.cross(P3[1] - P3[0], P3[2] - P3[0])
        if np.linalg.norm(fn) < 1e-9: continue
        if np.dot(fn, n) < 0: P3 = P3[[0, 2, 1]]
        s = np.array([np.linalg.norm((p - np.array(pa))[[0, 2]]) for p in P3]) + (pa[0] + pa[2])
        wall_tris.setdefault(mat, []).append((P3, n, s))
for j in np.nonzero(need)[0]:
    pu = np.array([Vx[u[j]], 0, Vz[u[j]]]); pw = np.array([Vx[w[j]], 0, Vz[w[j]]])
    hiA = (du[j] + dw[j]) > 0
    hic = lab[tA[j]] if hiA else lab[tB[j]]
    loc_ = lab[tB[j]] if hiA else lab[tA[j]]
    mat = 'sandstone' if (loc_ == RIVER_C or hic == RIVER_C) else WALLMAT.get(NAMES[hic], 'kerb')
    cen_lo = cen[tB[j]] if hiA else cen[tA[j]]
    if du[j] * dw[j] >= 0:
        lo_u, hi_u = sorted((hAu[j], hBu[j])); lo_w, hi_w = sorted((hAw[j], hBw[j]))
        zipper(mat, pu, pw, hset(u[j], lo_u, hi_u), hset(w[j], lo_w, hi_w), look=cen_lo)
    else:
        # Vorzeichenwechsel der Hoehendifferenz: zwei Dreiecke ueber den Kreuzungspunkt
        tt = du[j] / (du[j] - dw[j]); pm = pu + (pw - pu) * tt
        hm = hAu[j] + (hAw[j] - hAu[j]) * tt
        for (a_, vv, ha1, ha2, look) in ((pu, u[j], hAu[j], hBu[j], cen[tB[j]] if du[j] > 0 else cen[tA[j]]), (pw, w[j], hAw[j], hBw[j], cen[tB[j]] if dw[j] > 0 else cen[tA[j]])):
            lo, hi = sorted((ha1, ha2))
            zipper(mat, a_, pm, hset(vv, lo, hi), [hm], look=look)

# ---------------------------------------------------------------- Raender: Loch-Kanten, Schuerzen
bt = Etri[iBd]; bu = E[iBd, 0]; bw = E[iBd, 1]
bhu = corner_h(bt, bu); bhw = corner_h(bt, bw)
bm = (V[bu] + V[bw]) / 2
hole_edges = {k: [] for k in HOLES}
hp = {k: h['poly'].buffer(0.05) for k, h in HOLES.items()}
sq_in = L['sq_in'].buffer(0.05)
for k in hp: shapely.prepare(hp[k])
shapely.prepare(sq_in)
assigned = np.zeros(len(bt), bool)
for k, g in hp.items():
    m = shapely.contains_xy(g, bm[:, 0], bm[:, 1]) & ~assigned
    for j in np.nonzero(m)[0]:
        hole_edges[k].append((V[bu[j]].tolist(), V[bw[j]].tolist(), float(bhu[j]), float(bhw[j]), cen[bt[j]].tolist()))
    assigned |= m
insq = shapely.contains_xy(sq_in, bm[:, 0], bm[:, 1]) & ~assigned
outer = ~assigned & ~insq
skirts = 0
for j in np.nonzero(insq | outer)[0]:
    depth = 0.6 if insq[j] else 1.5
    a_ = np.array([Vx[bu[j]], 0, Vz[bu[j]]]); b_ = np.array([Vx[bw[j]], 0, Vz[bw[j]]])
    bot_u = _ch[HS_start[bu[j]]] - depth; bot_w = _ch[HS_start[bw[j]]] - depth
    zipper('soil', a_, b_, hset(bu[j], bot_u, bhu[j]), hset(bw[j], bot_w, bhw[j]), away=cen[bt[j]])
    skirts += 1
log('hole edges', {k: len(v) for k, v in hole_edges.items()}, 'skirts', skirts)

# ---------------------------------------------------------------- Richtungsfeld fuer Belaege
def dir_samples(geoms_lines, step=1.0):
    P = []; A = []
    for c in geoms_lines:
        c = np.asarray(c)
        if len(c) < 2: continue
        d = np.diff(c, axis=0); l = np.linalg.norm(d, axis=1)
        for i in range(len(d)):
            if l[i] < 1e-6: continue
            n = max(1, int(l[i] / step)); t = (np.arange(n) + 0.5) / n
            P.append(c[i] + d[i] * t[:, None]); A.append(np.full(n, math.atan2(d[i][1], d[i][0])))
    return np.concatenate(P), np.concatenate(A)
axis_lines = [c for c, w, t in L['road_lines']] + [c for c, t in L['ped_lines']] + [c for c, t in L['foot_lines']] + [c for c, t in L['side_lines']]
PA, AA = dir_samples(axis_lines, 1.5)
kdA = cKDTree(PA)
edge_lines = []
for g in [L['low'].boundary] + [CL[n].boundary for n in ('grass', 'bed') if n in CL]:
    for ls in getattr(g, 'geoms', [g]):
        edge_lines.append(np.asarray(ls.coords))
PE, AE = dir_samples(edge_lines, 0.5)
kdE = cKDTree(PE)
ang = np.zeros(nT)
d1 = CDIR[lab] == 1
_, ii = kdA.query(cen[d1]); ang[d1] = AA[ii]
d2 = CDIR[lab] == 2
_, ii = kdE.query(cen[d2]); ang[d2] = AE[ii]
BINDEG = 7.5
abin = np.round(np.degrees(np.mod(ang, np.pi)) / BINDEG).astype(int) % int(180 / BINDEG)
abin[CDIR[lab] == 0] = 0

# Detail-UV-Abbildungen (wie im Detailmodell) fuer den Nahtstreifen
import trimesh
DUV = {}
for g, key, matname in (('street', 'asphalt 61feabfb', 'asphalt'), ('street', 'slabs f833e41b', 'slabs_ped'), ('street', 'setts 84d7cf54', 'setts_road'), ('green', 'grass 69e1b23b', 'grass')):
    sc = trimesh.load(DATA + 'rhino_export/' + g + '.glb', force='scene')
    for name, geo in sc.geometry.items():
        if key in str(getattr(geo.visual.material, 'name', '')) and getattr(geo.visual, 'uv', None) is not None:
            Tm = sc.graph.get(sc.graph.geometry_nodes[name][0])[0]
            v = trimesh.transform_points(geo.vertices, Tm)
            X = np.c_[v[:, 0], v[:, 2], np.ones(len(v))]
            A_, *_ = np.linalg.lstsq(X, geo.visual.uv, rcond=None)
            DUV[matname] = A_; break
log('detail uv maps', list(DUV))
SEAM_MAT = {CID['seam_road']: 'asphalt', CID['seam_setts']: 'setts_road', CID['seam_grass']: 'grass', CID['seam']: 'slabs_ped'}
SEAM_LIST = list(SEAM_MAT)
tmat = np.array([CMAT[c] if CMAT[c] else '' for c in lab], dtype=object)
sm = np.isin(lab, SEAMC)
abin[sm] = 100 + np.array([SEAM_LIST.index(c) for c in lab[sm]])

# ---------------------------------------------------------------- Ausgabe-Vertices (Schluessel: Vertex, Klasse, Winkel-Bin)
cv = T.ravel(); cc = np.repeat(lab, 3); cb = np.repeat(abin, 3); chh = HC.ravel()
key = (cv.astype(np.int64) * 64 + cc) * 256 + cb
uk, oinv = np.unique(key, return_inverse=True)
nO = len(uk)
OV = np.zeros((nO, 3)); OV[oinv, 0] = Vx[cv]; OV[oinv, 2] = Vz[cv]; OV[oinv, 1] = chh
OC = np.zeros(nO, int); OC[oinv] = cc; OB = np.zeros(nO, int); OB[oinv] = cb
OT = oinv.reshape(-1, 3)
# Orientierung: Normalen nach oben
fn = np.cross(OV[OT[:, 1]] - OV[OT[:, 0]], OV[OT[:, 2]] - OV[OT[:, 0]])
flip = fn[:, 1] < 0; OT[flip] = OT[flip][:, ::-1]; fn[flip] *= -1
ON = np.zeros((nO, 3))
for k in range(3): np.add.at(ON, OT[:, k], fn)
ON /= np.linalg.norm(ON, axis=1, keepdims=True) + 1e-12
# UV
OUV = np.zeros((nO, 2))
x, z = OV[:, 0], OV[:, 2]
reg = OB < 100
a = np.radians(OB[reg] * BINDEG); sc_ = CUV[OC[reg]]
ca, sa = np.cos(a), np.sin(a)
OUV[reg, 0] = (x[reg] * ca + z[reg] * sa) / sc_
OUV[reg, 1] = (-x[reg] * sa + z[reg] * ca) / sc_
kerbm = reg & np.isin(OC, [CID['kerb'], CID['kerb_ch']])
OUV[kerbm, 1] *= 2.0 / 0.25            # Bordstein-Textur: 0.25 m quer
for b_ in np.unique(OB[~reg]):
    m = OB == b_
    matname = SEAM_MAT[SEAM_LIST[b_ - 100]]
    A_ = DUV.get(matname)
    xr = x[m] - RHINO_OFF[0]; zr = z[m] - RHINO_OFF[2]
    if A_ is not None:
        uvd = np.c_[xr, zr, np.ones(m.sum())] @ A_
        OUV[m, 0] = uvd[:, 0]; OUV[m, 1] = 1 - uvd[:, 1]
    else:
        OUV[m] = np.c_[x[m], z[m]] / 4.8
log('output vertices', nO, 'triangles', len(OT))

# Waende/Schuerzen als Parts
parts = []
for mat, lst in wall_tris.items():
    P_ = np.concatenate([x[0] for x in lst]); N_ = np.repeat(np.array([x[1] for x in lst]), 3, 0); S_ = np.concatenate([x[2] for x in lst])
    uvs = np.c_[S_, -P_[:, 1]]
    if mat == 'kerb': uvs[:, 0] /= 2.0; uvs[:, 1] *= 4.0          # Bordstein: 2 m entlang, 0.25 m hoch
    elif mat == 'soil': uvs /= 2.0
    elif mat == 'sandstone': uvs /= 2.4
    parts.append(G.part(mat, P_, np.arange(len(P_)).reshape(-1, 3), N_, uvs))
log('wall/skirt triangles', G.tri_count(parts))

np.savez_compressed(CACHE + 'terrain.npz', V=V, T=T, lab=lab, HC=HC, OV=OV, ON=ON, OUV=OUV, OT=OT, tmat=tmat.astype(str), names=np.array(NAMES))
pickle.dump({'parts': parts, 'hole_edges': hole_edges, 'crossings': [(c[0].tolist(), c[1].tolist(), c[2].tolist(), c[3], c[4], c[5]) for c in cross]},
            open(CACHE + 'terrain_parts.pkl', 'wb'))
log('saved')

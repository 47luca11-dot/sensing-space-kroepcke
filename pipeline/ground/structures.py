"""Bauwerke am Boden: Hecken, Mauern, Zaeune, Stadtbahn-Schienen (Rillenschienen), Fahrbahnmarkierungen.
(U-Bahn-Zugaenge: ubahn.py, Passerelle: promenade.py, Haltestellen/Oberleitung: transit.py, Altstadt/Leine: altstadt.py) Ausgabe: data/ground_cache/structures.pkl (Parts + Kollisions-Parts)"""
import sys, json, time, pickle, math
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point, box, MultiLineString
from shapely.ops import unary_union
import mapbox_earcut as earcut
import triangle as tr
from matplotlib.tri import Triangulation
from common import *
import geom as G

T0 = time.time()
def log(*a): print('[struct %6.1fs]' % (time.time() - T0), *a, flush=True)

L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
TP = pickle.load(open(CACHE + 'terrain_parts.pkl', 'rb'))
TM = np.load(CACHE + 'terrain.npz', allow_pickle=False)
V, T, lab, HC = TM['V'], TM['T'], TM['lab'], TM['HC']
NAMES = list(TM['names'])
bz = np.load(CACHE + 'base.npz'); BG = Grid(bz['B'].astype(np.float64), float(bz['x0']), float(bz['z0']), 1.0)
LOWC = [NAMES.index(n) for n in ('road', 'gutter', 'parking', 'setts_road', 'track')]

# ---------------------------------------------------------------- Oberflaechen-Abfrage (Dreiecks-Suche)
triang = Triangulation(V[:, 0], V[:, 1], T)
finder = triang.get_trifinder()
def surface(x, z, low_only=False):
    x = np.asarray(x, float); z = np.asarray(z, float)
    ti = finder(x, z)
    h = BG(x, z)
    ok = ti >= 0
    if ok.any():
        t = T[ti[ok]]; a = V[t[:, 0]]; b = V[t[:, 1]]; c = V[t[:, 2]]
        p = np.c_[x[ok], z[ok]]
        v0 = b - a; v1 = c - a; v2 = p - a
        d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01; den[den == 0] = 1
        wv = (d11 * d20 - d01 * d21) / den; ww = (d00 * d21 - d01 * d20) / den; wu = 1 - wv - ww
        hc = HC[ti[ok]]
        h[ok] = wu * hc[:, 0] + wv * hc[:, 1] + ww * hc[:, 2]
    cls = np.full(len(x), -1); cls[ok] = lab[ti[ok]]
    return h, cls

parts = []; coll = []
def add(p, collide=False):
    if p is None: return
    if isinstance(p, list):
        for q in p: add(q, collide)
        return
    parts.append(p)
    if collide: coll.append(p)

# ---------------------------------------------------------------- Helfer
def wall_from_edges(edges, bottom_fn, mat, uvscale=1.0):
    """Waende unter Lochkanten: edges=(pu,pw,hu,hw,terrain_centroid); bottom_fn(x,z)->y; Normale weg vom Gelaende"""
    Q = []
    for pu, pw, hu, hw, c in edges:
        bu = bottom_fn(*pu); bw = bottom_fn(*pw)
        if hu - bu < 0.01 and hw - bw < 0.01: continue
        q = np.array([[pu[0], bu, pu[1]], [pw[0], bw, pw[1]], [pw[0], hw, pw[1]], [pu[0], hu, pu[1]]])
        n = np.cross(q[1] - q[0], q[3] - q[0])
        if np.dot(n[[0, 2]], np.array(c) - q.mean(0)[[0, 2]]) > 0: q = q[[1, 0, 3, 2]]
        Q.append(q)
    return G.flat_quads(mat, np.array(Q), uvscale) if Q else None

def poly_mesh(mat, ring, yfn, extra_pts=(), up=True, maxa=4.0, uvscale=1.0, quality=True):
    """Ebene/geneigte Flaeche aus Polygon (xz) mit festen Randpunkten (Y: keine Steinerpunkte am Rand)."""
    P = np.asarray(ring, float)
    if np.allclose(P[0], P[-1]): P = P[:-1]
    # zusaetzliche Randpunkte (auf den Kanten liegend) einfuegen
    pts = []
    ex = np.asarray(extra_pts, float).reshape(-1, 2)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        pts.append(a)
        if len(ex):
            d = b - a; l2 = d @ d
            if l2 < 1e-12: continue
            t = ((ex - a) @ d) / l2
            dist = np.abs((ex - a) @ np.array([-d[1], d[0]])) / math.sqrt(l2)
            m = (t > 1e-6) & (t < 1 - 1e-6) & (dist < 2e-3)
            for tt in sorted(t[m]): pts.append(a + d * tt)
    pts = np.array(pts)
    _, ui = np.unique(np.round(pts, 4), axis=0, return_index=True); ui = np.sort(ui); pts = pts[ui]
    n = len(pts); seg = np.c_[np.arange(n), (np.arange(n) + 1) % n]
    r = tr.triangulate({'vertices': pts, 'segments': seg}, ('pq20Ya%.2fQ' % maxa) if quality else 'pYQ')
    Vv, Tt = r['vertices'], r['triangles']
    y = np.array([yfn(x, z) for x, z in Vv])
    pos = np.c_[Vv[:, 0], y, Vv[:, 1]]
    p = G.part(mat, pos, Tt, None, Vv / uvscale)
    fn = np.cross(pos[Tt[:, 1]] - pos[Tt[:, 0]], pos[Tt[:, 2]] - pos[Tt[:, 0]])
    if (fn[:, 1].sum() < 0) == up: p['idx'] = p['idx'][:, ::-1]
    p['nrm'] = G.vertex_normals(p['pos'], p['idx'])
    return p

def stair(mat, origin, d, nrm, t0, t1, y_top, n_steps, rise, tread, y_floor, side_mat=None):
    """Treppe als geschlossener Koerper: absteigend in Richtung d ab origin. Querschnitt t0..t1."""
    o = np.asarray(origin, float); d = np.asarray(d, float); nr = np.asarray(nrm, float)
    def W(s, t, y): return [o[0] + d[0] * s + nr[0] * t, y, o[1] + d[1] * s + nr[1] * t]
    Q = []
    for i in range(n_steps):
        s0, s1 = i * tread, (i + 1) * tread
        yt = y_top - (i + 1) * rise
        Q.append([W(s0, t0, yt), W(s1, t0, yt), W(s1, t1, yt), W(s0, t1, yt)])               # Trittflaeche
        Q.append([W(s0, t0, yt + rise), W(s0, t0, yt), W(s0, t1, yt), W(s0, t1, yt + rise)])   # Setzstufe (Blick gegen d)
    tq = G.flat_quads(mat, np.array(Q), 1.0)
    # Orientierung: Tritte nach oben, Setzstufen gegen d
    for k in range(len(Q)):
        pass
    pos = tq['pos']; idx = tq['idx']
    fn = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    cen = pos[idx].mean(1)
    want = np.where(np.abs(fn[:, 1]) > 1e-9, np.sign(fn[:, 1]), 0)
    # Tritt: fn.y muss > 0; Setzstufe: fn . d3 < 0
    d3 = np.array([d[0], 0, d[1]])
    bad = ((np.abs(fn[:, 1]) > 1e-9) & (fn[:, 1] < 0)) | ((np.abs(fn[:, 1]) <= 1e-9) & ((fn @ d3) > 0))
    idx[bad] = idx[bad][:, ::-1]
    tq['nrm'] = G.vertex_normals(pos, idx)
    # Wangen (Seiten) als Profil-Polygone
    prof = [(0, y_floor)]
    for i in range(n_steps):
        prof += [(i * tread, y_top - i * rise - rise), ((i + 1) * tread, y_top - (i + 1) * rise)]
    prof = np.array(prof)
    tri = earcut.triangulate_float64(prof, np.array([len(prof)], np.uint32)).reshape(-1, 3)
    sides = []
    for t, sgn in ((t0, -1), (t1, 1)):
        pos2 = np.array([W(s, t, y) for s, y in prof])
        p = G.part(side_mat or mat, pos2, tri, None, prof)
        fn2 = np.cross(pos2[tri[:, 1]] - pos2[tri[:, 0]], pos2[tri[:, 2]] - pos2[tri[:, 0]])
        n3 = np.array([nr[0], 0, nr[1]]) * sgn
        if len(fn2) and (fn2 @ n3).sum() < 0: p['idx'] = p['idx'][:, ::-1]
        p['nrm'] = np.tile(n3, (len(pos2), 1)); sides.append(p)
    return [tq] + sides

def railing(pts_xz, base_fn, height=1.05, post=1.4, mat='steel'):
    """Gelaender entlang Polyline (xz): Pfosten, Handlauf, Knieleiste"""
    P = np.asarray(pts_xz, float)
    if len(P) < 2: return []
    seg = LineString(P)
    n = max(2, int(math.ceil(seg.length / post)) + 1)
    S = np.array([seg.interpolate(i / (n - 1), normalized=True).coords[0] for i in range(n)])
    yb = np.array([base_fn(x, z) for x, z in S])
    out = []
    for (x, z), y in zip(S, yb):
        out.append(G.box(mat, (x, y, z), (0.05, height, 0.05)))
    out.append(G.tube(mat, np.c_[S[:, 0], yb + height, S[:, 1]], 0.028, 6))
    out.append(G.tube(mat, np.c_[S[:, 0], yb + 0.12, S[:, 1]], 0.015, 5))
    out.append(G.tube(mat, np.c_[S[:, 0], yb + height * 0.55, S[:, 1]], 0.015, 5))
    return out

def sign_board(tex_mat, c, size, a, frame_mat='steel'):
    """quadratisches Schild beidseitig texturiert, Mittelpunkt c=(x,y,z), Drehung a um Y"""
    w = size / 2; R = G.rot_y(a)
    corners = np.array([[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]])
    out = []
    for sgn in (1, -1):
        q = corners.copy(); q[:, 2] = 0.022 * sgn
        if sgn < 0: q = q[[1, 0, 3, 2]]
        pos = q @ R.T + c
        uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
        n = np.cross(pos[1] - pos[0], pos[3] - pos[0]); n /= np.linalg.norm(n)
        out.append(G.part(tex_mat, pos, [(0, 1, 2), (0, 2, 3)], np.tile(n, (4, 1)), uv))
    out.append(G.box(frame_mat, c - [0, w, 0], (size + 0.02, size + 0.02, 0.04), a))
    # Rahmen-Box liegt hinter den Bildflaechen (0.04 dick, Bildflaechen bei +-0.022)
    return out

# U-Bahn-Zugaenge -> ubahn.py, Niki-de-Saint-Phalle-Promenade -> promenade.py (Ausbau 2)
ub_info = []

# ---------------------------------------------------------------- Hecken, Mauern, Zaeune
osm = load_osm()
REG = L['region']; BLD = L['bld']; shapely.prepare(REG)
nhed = 0
for c in L['hedges']:
    Ls = LineString(c).intersection(REG)
    for g in getattr(Ls, 'geoms', [Ls]):
        if g.is_empty or g.geom_type != 'LineString' or g.length < 0.8: continue
        poly = g.buffer(0.42, cap_style=2, join_style=2)
        if poly.geom_type != 'Polygon': continue
        ring = np.array(poly.exterior.coords)
        yb = surface(ring[:, 0], ring[:, 1])[0]
        hh = 1.05 + 0.1 * math.sin(len(ring))
        add(G.extrude('hedge', ring, yb - 0.08, yb + hh, uvscale=1.0, top=True), True)
        nhed += 1
nwall = 0; nfence = 0; nskip_hole = 0
# Mauern/Zaeune nicht in Gelaende-Loechern (Passerelle-Oeffnungen, U-Bahn-Troege): dort gibt es kein Strassenniveau
# (QA 2026-11-10: OSM-Mauern 435784547/139432718/435784546 auf der Trogmitte schwebten 3.5 m ueber dem Boden; BEGEHUNGSLISTE QG-1)
HOLES_ALL = unary_union([h['poly'] for h in L['holes'].values()]).buffer(0.35)
REG_WALL = REG.difference(BLD).difference(HOLES_ALL)
for w in osm['ways']:
    t = w['tags']; b = t.get('barrier')
    if b not in ('wall', 'fence', 'city_wall', 'retaining_wall') or len(w['pts']) < 2: continue
    if str(t.get('layer', '0')).startswith('-') or t.get('indoor') == 'yes': continue
    L0 = LineString(w['pts'])
    if L0.intersects(HOLES_ALL):
        nskip_hole += 1
        log('  barrier', w.get('id'), b, 'clipped at hole (len %.1f m in hole)' % L0.intersection(HOLES_ALL).length)
    Ls = L0.intersection(REG_WALL)
    for g in getattr(Ls, 'geoms', [Ls]):
        if g.is_empty or g.geom_type != 'LineString' or g.length < 1.0: continue
        if b in ('wall', 'city_wall', 'retaining_wall'):
            hgt = min(float(str(t.get('height', '1.1')).split()[0].replace(',', '.')) if t.get('height') else (1.1 if b != 'retaining_wall' else 0.6), 3.0)
            poly = g.buffer(0.15, cap_style=2, join_style=2)
            if poly.geom_type != 'Polygon': continue
            ring = np.array(poly.exterior.coords); yb = surface(ring[:, 0], ring[:, 1])[0]
            add(G.extrude('wallstone', ring, yb - 0.1, yb + hgt, uvscale=1.0), True); nwall += 1
        else:
            seg = shapely.segmentize(g, 2.0); c2 = np.array(seg.coords)
            yb = surface(c2[:, 0], c2[:, 1])[0]
            Q = []
            for j in range(len(c2) - 1):
                Q.append([[c2[j, 0], yb[j], c2[j, 1]], [c2[j + 1, 0], yb[j + 1], c2[j + 1, 1]], [c2[j + 1, 0], yb[j + 1] + 1.2, c2[j + 1, 1]], [c2[j, 0], yb[j] + 1.2, c2[j, 1]]])
            fq = G.flat_quads('fence', np.array(Q), 1.0)
            fq['uv'][:, 1] = np.where(fq['pos'][:, 1] > np.repeat(yb[:-1], 4)[:len(fq['pos'])] + 0.6, 0.0, 1.0) if False else fq['uv'][:, 1]
            # u entlang, v 0..1 ueber 1.2 m
            add(fq, True); nfence += 1
log('hedges', nhed, 'walls', nwall, 'fences', nfence, 'clipped at holes', nskip_hole)

# ---------------------------------------------------------------- Stadtbahn-Schienen (in der Fahrbahn eingelassen)
def strip_along(line_xz, offset, width, y_off, mat, step=1.0):
    Ls = LineString(line_xz)
    if Ls.length < 0.5: return None
    n = max(2, int(Ls.length / step) + 1)
    S = np.array([Ls.interpolate(i / (n - 1), normalized=True).coords[0] for i in range(n)])
    Tg = np.gradient(S, axis=0); Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-12
    N = np.c_[-Tg[:, 1], Tg[:, 0]]
    A = S + N * (offset - width / 2); B = S + N * (offset + width / 2)
    ya = surface(A[:, 0], A[:, 1])[0] + y_off; yb = surface(B[:, 0], B[:, 1])[0] + y_off
    pos = np.concatenate([np.c_[A[:, 0], ya, A[:, 1]], np.c_[B[:, 0], yb, B[:, 1]]])
    idx = []
    for i in range(n - 1): idx += [(i, n + i, n + i + 1), (i, n + i + 1, i + 1)]
    L_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]
    uv = np.concatenate([np.c_[L_, np.zeros(n)], np.c_[L_, np.ones(n)]])
    p = G.part(mat, pos, idx, np.tile([0, 1, 0], (2 * n, 1)), uv)
    return G.orient_up(p)
nr_ = 0
for c, t in L['lr']:
    Ls = LineString(c).intersection(REG)
    for g in getattr(Ls, 'geoms', [Ls]):
        if g.is_empty or g.geom_type != 'LineString': continue
        cc = np.array(g.coords)
        for off in (-0.7175, 0.7175):
            add(strip_along(cc, off, 0.07, 0.006, 'rail'))
            add(strip_along(cc, off - np.sign(off) * 0.055, 0.04, 0.004, 'groove'))
        nr_ += 1
log('rail tracks', nr_)

# ---------------------------------------------------------------- Markierungen
LOWG = L['low']
LOWS = LOWG.buffer(-0.35)
shapely.prepare(LOWS)
marks = []
def dash_line(Ls, dash, gap, width):
    out = []; s = gap / 2
    while s + dash <= Ls.length:
        seg = shapely.ops.substring(Ls, s, s + dash)
        out.append(seg.buffer(width / 2, cap_style=2)); s += dash + gap
    return out
import shapely.ops
cross = TP['crossings']
cross_zone = unary_union([Point(c[0]).buffer(9.0) for c in cross]) if cross else Polygon()
rl = [(LineString(c), w, t) for c, w, t in L['road_lines']]
# Knoten (Einmuendungen): Endpunkte und Kreuzungen der Strassenachsen
ends = []
for Ls, w, t in rl:
    ends += [Ls.coords[0], Ls.coords[-1]]
junction = unary_union([Point(p).buffer(11.0) for p in ends])
for Ls, w, t in rl:
    hw = t['highway']
    if hw not in ('primary', 'secondary', 'tertiary', 'secondary_link', 'tertiary_link', 'residential', 'unclassified'): continue
    if t.get('surface') in ('sett', 'cobblestone', 'paving_stones'): continue
    lanes = int(float(t.get('lanes', '0') or 0)) if str(t.get('lanes', '0')).replace('.', '', 1).isdigit() else 0
    oneway = t.get('oneway') == 'yes'
    core = Ls.difference(junction).difference(cross_zone)
    for g in getattr(core, 'geoms', [core]):
        if g.is_empty or g.geom_type != 'LineString' or g.length < 6: continue
        if not oneway and w >= 6.0 and hw != 'residential':
            marks += dash_line(g, 3.0, 6.0, 0.12)
        elif oneway and lanes >= 2:
            lw = w / lanes
            for k in range(1, lanes):
                off = -w / 2 + k * lw
                go = g.offset_curve(off)
                if go.is_empty or go.geom_type != 'LineString': continue
                marks += dash_line(go, 3.0, 6.0, 0.12)
# Ueberwege
for p, d, nrm, e0, e1, kind in cross:
    p = np.array(p); d = np.array(d); nrm = np.array(nrm)
    if kind in ('zebra', 'marked', 'uncontrolled'):
        k = e0 + 0.4
        while k + 0.5 <= e1 - 0.3:
            c0 = p + nrm * (k + 0.25)
            marks.append(LineString([c0 - d * 2.0, c0 + d * 2.0]).buffer(0.25, cap_style=2)); k += 1.0
    elif kind in ('traffic_signals',):
        for sgn in (-1, 1):
            Lf = LineString([p + d * 2.2 * sgn + nrm * e0, p + d * 2.2 * sgn + nrm * e1])
            marks += dash_line(Lf, 0.5, 0.25, 0.12)
        # Haltlinien je Fahrtrichtung (rechte Haelfte)
        for sgn in (-1, 1):
            a0 = p + d * (3.6 * sgn);
            half = (e0, (e0 + e1) / 2) if sgn > 0 else ((e0 + e1) / 2, e1)
            marks.append(LineString([a0 + nrm * half[0], a0 + nrm * half[1]]).buffer(0.25, cap_style=2))
M = unary_union(marks).intersection(LOWS) if marks else Polygon()
mparts = []
for poly in getattr(M, 'geoms', [M]):
    if poly.is_empty or poly.geom_type != 'Polygon' or poly.area < 0.01: continue
    poly = shapely.segmentize(poly, 0.8)
    ring = np.array(poly.exterior.coords)[:-1]
    holes_ = [np.array(r.coords)[:-1] for r in poly.interiors]
    allp = np.concatenate([ring] + holes_) if holes_ else ring
    ends_ = np.cumsum([len(ring)] + [len(h) for h in holes_]).astype(np.uint32)
    tri = earcut.triangulate_float64(allp, ends_).reshape(-1, 3)
    if len(tri) == 0: continue
    y = surface(allp[:, 0], allp[:, 1])[0] + 0.012
    pp = G.part('marking', np.c_[allp[:, 0], y, allp[:, 1]], tri, np.tile([0, 1, 0], (len(allp), 1)), allp)
    mparts.append(G.orient_up(pp))
add(mparts)
log('markings', len(mparts), 'area', round(M.area, 1) if not M.is_empty else 0)

pickle.dump({'parts': parts, 'coll': coll, 'ubahn': ub_info}, open(CACHE + 'structures.pkl', 'wb'))
log('total structure tris', G.tri_count(parts), 'collision', G.tri_count(coll))

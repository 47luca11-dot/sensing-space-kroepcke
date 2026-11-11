"""Gemeinsame Bausteine fuer bh_build.py: Kontext (Mesh-Builder, Instanzen, Prototypen), Gelaendehoehe, Waende mit
Oeffnungen, Fachwerk-Generator, Steinsockel/Eckquader, Fallrohre, Efeu, Schilder, Innenwaende aus dem Dachmodell."""
import json, os, math
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box as sbox
from shapely.ops import unary_union
from bh_mesh import MB, Frame, WORLD, unit, area2, ccw, offset_poly, tri_poly, UP
from bh_roof import as_polys
import bh_parts as BP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
_hj = json.load(open(ROOT + 'web/assets/city/ground/heightfield.json'))
_HF = np.fromfile(ROOT + 'web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(_hj['h'], _hj['w'])


def H(x, z):
    fx = (np.asarray(x, float) - _hj['x0']) / _hj['dx']; fz = (np.asarray(z, float) - _hj['z0']) / _hj['dz']
    i = np.floor(fx).astype(int); j = np.floor(fz).astype(int); a = fx - i; b = fz - j
    return (1 - a) * (1 - b) * _HF[j, i] + a * (1 - b) * _HF[j, i + 1] + (1 - a) * b * _HF[j + 1, i] + a * b * _HF[j + 1, i + 1]


class Ctx:
    def __init__(self):
        self.mb = MB()
        self.inst = {}          # proto -> [M4]
        self.protos = {}        # proto -> MB
        self.col = MB()         # Kollision
        self.rng = np.random.default_rng(5)

    def proto(self, key, fn):
        if key not in self.protos:
            self.protos[key] = fn()
        return key

    def place(self, key, M):
        self.inst.setdefault(key, []).append(np.asarray(M, float))


# ============================================================================================== Wand-Stueck
class Wall:
    """Ebenes Wandstueck: Frame F (a nach rechts, b = absolute Hoehe y, c aussen), Polygon (a,b), Material.
    ref: Frame der Gebaeude-Aussenkante (ohne Rueckversatz) fuer Positionsangaben (a_ref)."""

    def __init__(self, ctx, F, poly, mat, ref=None, L=None, edge=None, zone=None, bld=None):
        self.ctx, self.F, self.mat = ctx, F, mat
        self.poly = Polygon(poly).buffer(0)
        self.ref = ref or F
        self.L = L; self.edge = edge; self.zone = zone; self.bld = bld
        self.holes = []
        self.members = []         # (Polygon, grain, mat, c0, c1, bevel)
        self.extra_clip = []

    def A(self, a_ref):
        """a im Referenzframe -> a in diesem Wandframe"""
        return float(self.F.local(self.ref.P([a_ref, 0.0, 0.0]))[0])

    @property
    def amin(self):
        return self.poly.bounds[0]

    @property
    def amax(self):
        return self.poly.bounds[2]

    def ytop(self, a):
        """Oberkante des Wandpolygons an Stelle a"""
        ln = LineString([(a, -50), (a, 80)]).intersection(self.poly)
        return ln.bounds[3] if not ln.is_empty else None

    def hole(self, pts):
        self.holes.append(Polygon(pts))

    def emit(self, uvo=(0.0, 0.0)):
        g = self.poly
        if self.holes:
            g = g.difference(unary_union(self.holes))
        for p in as_polys(g):
            if p.area < 1e-4: continue
            ex = np.array(p.exterior.coords)[:-1]
            hs = [np.array(r.coords)[:-1] for r in p.interiors]
            self.ctx.mb.poly(self.mat, self.F, ex, hs, c=0.0, uv=lambda V: V + np.array(uvo))
        # Holzwerk / Gewaende
        self.emit_members()

    def member(self, poly, grain=(1, 0), mat='Holz_dunkel', c0=-0.02, c1=0.035, bevel=0.012, prio=0):
        self.members.append((Polygon(poly).buffer(0) if not isinstance(poly, Polygon) else poly, grain, mat, c0, c1, bevel, prio))

    def emit_members(self):
        if not self.members: return
        cut = unary_union(self.holes) if self.holes else None
        used = None
        clip = self.poly.buffer(-0.002, join_style=2)
        for poly, grain, mat, c0, c1, bev, prio in sorted(self.members, key=lambda m: m[6]):
            g = poly.intersection(clip)
            if cut is not None: g = g.difference(cut)
            if used is not None and c0 < 0.0: g = g.difference(used)
            if g.is_empty: continue
            for p in as_polys(g):
                if p.area < 2e-4: continue
                ex = np.array(p.exterior.coords)[:-1]
                if len(p.interiors):
                    # Loecher: ohne Fase als Polygon mit Loechern (selten)
                    self.ctx.mb.poly(mat, self.F, ex, [np.array(r.coords)[:-1] for r in p.interiors], c=c1, uv=lambda V: V @ np.array([[grain[0], -grain[1]], [grain[1], grain[0]]]).T if False else V)
                    continue
                ex = _simplify_ring(ex)
                if len(ex) < 3: continue
                self.ctx.mb.prism(mat, self.F, ex, c0, c1, bevel=bev, grain=grain)
            if c0 < 0.0:
                used = g if used is None else used.union(g)
        self.members = []


def _simplify_ring(ex, tol=1e-4):
    out = []
    n = len(ex)
    for i in range(n):
        a, b, c = ex[i - 1], ex[i], ex[(i + 1) % n]
        if np.linalg.norm(b - a) < 0.004: continue
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < tol * max(np.linalg.norm(b - a), 1e-3): continue
        out.append(b)
    return np.array(out)


def rect(a0, b0, a1, b1):
    return [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]


def strut(p, q, w):
    """Strebe als Parallelogramm/Rechteck von p nach q mit Breite w (senkrecht zur Achse)"""
    p = np.asarray(p, float); q = np.asarray(q, float)
    d = unit(q - p); n = np.array([-d[1], d[0]]) * w / 2
    return [tuple(p - n), tuple(q - n), tuple(q + n), tuple(p + n)]


def hstrut(p, q, w):
    """Strebe zwischen zwei Hoehen mit waagrechten Schnitten (Breite w waagrecht gemessen)"""
    (x0, y0), (x1, y1) = p, q
    return [(x0 - w / 2, y0), (x0 + w / 2, y0), (x1 + w / 2, y1), (x1 - w / 2, y1)]


# ============================================================================================== Fenster
def add_window(wall, a_ref, b0, w, h, key, mk, reveal=None):
    """Fenster mittig bei a_ref (Referenzframe), Unterkante b0; Prototyp key (mk erzeugt ihn)"""
    a = wall.A(a_ref) - w / 2
    wall.hole(rect(a, b0, a + w, b0 + h))
    wall.ctx.proto(key, mk)
    wall.ctx.place(key, wall.F.sub(a, b0, 0.0).mat4())
    return a


# ============================================================================================== Fachwerk-Geschoss
def fachwerk(wall, y0, y1, posts, wins, sill, lintel, opts=None):
    """posts: Stielpositionen (a_ref); wins: [(a_ref_mitte, breite)] ; sill/lintel: Unter-/Oberkante der Fenster (y).
    opts: hs (Schwelle), hr (Rahm), wp (Stiel), hb (Riegel), brace ('V','A','X','FB','none'), mat, top_riegel (y|None),
    no_schwelle, no_raehm, full_brace ('K','/','none') fuer Felder ohne Fenster"""
    o = dict(hs=0.22, hr=0.20, wp=0.20, hb=0.15, brace='V', mat='Holz_dunkel', bw=0.16, top_riegel=None, full_brace='none',
             no_schwelle=False, no_raehm=False, mid_riegel=True)
    o.update(opts or {})
    A = wall.A
    a_lo, a_hi = wall.amin - 0.05, wall.amax + 0.05
    m = o['mat']
    yS = y0 + (0 if o['no_schwelle'] else o['hs'])
    yR = y1 - (0 if o['no_raehm'] else o['hr'])
    if not o['no_schwelle']:
        wall.member(rect(a_lo, y0, a_hi, y0 + o['hs']), (1, 0), m, prio=0)
    if not o['no_raehm']:
        wall.member(rect(a_lo, y1 - o['hr'], a_hi, y1), (1, 0), m, prio=0)
    P = sorted(A(p) for p in posts)
    for p in P:
        wall.member(rect(p - o['wp'] / 2, yS, p + o['wp'] / 2, yR), (0, 1), m, prio=1)
    W = [(A(c) - w / 2, A(c) + w / 2) for c, w in wins]
    for i in range(len(P) - 1):
        pa, pb = P[i] + o['wp'] / 2, P[i + 1] - o['wp'] / 2
        if pb - pa < 0.15: continue
        win = [(wa, wb) for (wa, wb) in W if wa > pa - 0.05 and wb < pb + 0.05]
        if win:
            # Brustriegel, Sturzriegel, Fensterstiele
            wall.member(rect(pa, sill - o['hb'], pb, sill), (1, 0), m, prio=2)
            if lintel + o['hb'] < yR - 0.05:
                wall.member(rect(pa, lintel, pb, lintel + o['hb']), (1, 0), m, prio=2)
            for (wa, wb) in win:
                if wa - pa > 0.12: wall.member(rect(wa - 0.12, sill, wa, lintel), (0, 1), m, prio=3)
                if pb - wb > 0.12: wall.member(rect(wb, sill, wb + 0.12, lintel), (0, 1), m, prio=3)
            # Brustungsfeld: Streben
            yb0, yb1 = yS, sill - o['hb']
            mid = (pa + pb) / 2; bw = o['bw']
            if yb1 - yb0 > 0.25:
                if o['brace'] == 'V':
                    wall.member(strut((pa + bw * 0.3, yb1), (mid, yb0), bw), (unit([mid - pa, yb0 - yb1])), m, prio=4)
                    wall.member(strut((pb - bw * 0.3, yb1), (mid, yb0), bw), (unit([mid - pb, yb0 - yb1])), m, prio=4)
                elif o['brace'] == 'A':
                    wall.member(strut((pa + bw * 0.3, yb0), (mid, yb1), bw), unit([mid - pa, yb1 - yb0]), m, prio=4)
                    wall.member(strut((pb - bw * 0.3, yb0), (mid, yb1), bw), unit([mid - pb, yb1 - yb0]), m, prio=4)
                elif o['brace'] == 'X':
                    wall.member(strut((pa, yb0), (pb, yb1), bw), unit([pb - pa, yb1 - yb0]), m, prio=4)
                    wall.member(strut((pb, yb0), (pa, yb1), bw), unit([pa - pb, yb1 - yb0]), m, prio=5)
                elif o['brace'] == 'FB':
                    # Feuerbock: kurze geschweifte Fussstreben von den Stielen
                    for s0, s1 in ((pa, 1), (pb, -1)):
                        pts = []
                        for t in np.linspace(0, 1, 6):
                            x = s0 + s1 * (0.05 + 0.42 * (pb - pa) / 2 * t)
                            y = yb0 + (yb1 - yb0) * (0.15 + 0.85 * math.sin(t * math.pi / 2) ** 1.4)
                            pts.append((x, y))
                        for k in range(len(pts) - 1):
                            wall.member(strut(pts[k], pts[k + 1], bw * 0.9), unit(np.subtract(pts[k + 1], pts[k])), m, prio=4)
            if o['top_riegel'] and o['top_riegel'] < yR - 0.1 and o['top_riegel'] > lintel + o['hb'] + 0.1:
                wall.member(rect(pa, o['top_riegel'] - o['hb'] / 2, pb, o['top_riegel'] + o['hb'] / 2), (1, 0), m, prio=2)
        else:
            # Feld ohne Fenster: Riegel auf Brustungs- und Sturzhoehe, ggf. Streben
            if o['mid_riegel']:
                wall.member(rect(pa, sill - o['hb'], pb, sill), (1, 0), m, prio=2)
                if lintel + o['hb'] < yR - 0.05:
                    wall.member(rect(pa, lintel, pb, lintel + o['hb']), (1, 0), m, prio=2)
            bw = o['bw']
            if o['full_brace'] == '/':
                wall.member(strut((pa, yS), (pb, sill - o['hb']), bw), unit([pb - pa, sill - yS]), m, prio=4)
                wall.member(strut((pa, sill), (pb, lintel), bw), unit([pb - pa, lintel - sill]), m, prio=4)
            elif o['full_brace'] == 'V':
                mid = (pa + pb) / 2
                wall.member(strut((pa + bw * 0.3, sill - o['hb']), (mid, yS), bw), unit([mid - pa, yS - sill]), m, prio=4)
                wall.member(strut((pb - bw * 0.3, sill - o['hb']), (mid, yS), bw), unit([mid - pb, yS - sill]), m, prio=4)
            elif o['full_brace'] == 'K':
                wall.member(strut((pa, yS), (pb, yR), bw), unit([pb - pa, yR - yS]), m, prio=4)
    return W


def gable_framing(wall, ybase, levels, posts_step, wins, opts=None):
    """Fachwerk im Giebeldreieck oberhalb ybase: Kehlbalkenlagen 'levels' (y), Stiele im Raster, Ortgang-Sparren entlang
    der schraegen Oberkanten, Fenster wins [(a_ref, w, b0, h)]"""
    o = dict(mat='Holz_dunkel', wp=0.18, hb=0.18, raf=0.20)
    o.update(opts or {})
    m = o['mat']
    a0, a1 = wall.amin, wall.amax
    for y in levels:
        wall.member(rect(a0 - 0.05, y, a1 + 0.05, y + o['hb']), (1, 0), m, prio=0)
    # Ortgang: schraege Oberkanten des Polygons
    ext = np.array(wall.poly.exterior.coords)
    for k in range(len(ext) - 1):
        p, q = ext[k], ext[k + 1]
        if p[1] > ybase + 0.2 and q[1] > ybase + 0.2 and abs(p[0] - q[0]) > 0.05 and abs(p[1] - q[1]) > 0.05:
            d = unit(q - p); n = np.array([-d[1], d[0]])
            if n[1] > 0: n = -n
            wall.member([tuple(p), tuple(q), tuple(q + n * o['raf']), tuple(p + n * o['raf'])], tuple(d), m, prio=0)
    n = int((a1 - a0) / posts_step)
    for i in range(1, n + 1):
        a = a0 + (a1 - a0) * i / (n + 1)
        wall.member(rect(a - o['wp'] / 2, ybase, a + o['wp'] / 2, 60), (0, 1), m, prio=1)
    W = []
    for (ar, w, b0, h) in wins:
        a = wall.A(ar)
        wall.member(rect(a - w / 2 - 0.12, b0 - 0.15, a + w / 2 + 0.12, b0), (1, 0), m, prio=2)
        wall.member(rect(a - w / 2 - 0.12, b0 + h, a + w / 2 + 0.12, b0 + h + 0.15), (1, 0), m, prio=2)
        wall.member(rect(a - w / 2 - 0.12, b0, a - w / 2, b0 + h), (0, 1), m, prio=2)
        wall.member(rect(a + w / 2, b0, a + w / 2 + 0.12, b0 + h), (0, 1), m, prio=2)
        W.append((ar, w, b0, h))
    return W


# ============================================================================================== Stein
def quoins(wall, a, y0, y1, side, mat='Sandstein', course=0.34, long=0.62, short=0.36, prot=0.025):
    """Eckquader an der Wandkante a (side=+1: Quader nach rechts, -1: nach links)"""
    y = y0; k = 0
    while y < y1 - 0.05:
        h = min(course, y1 - y)
        L = long if k % 2 == 0 else short
        wall.member(rect(a if side > 0 else a - L, y + 0.006, a + L if side > 0 else a, y + h - 0.006), (1, 0), mat, c0=-0.02, c1=prot, bevel=0.012, prio=0)
        y += h; k += 1


def surround(wall, a0, b0, a1, b1, w=0.16, mat='Sandstein', prot=0.03, sill=0.10, keystone=False):
    """Stein-Gewaende um eine Oeffnung (lokale a)"""
    wall.member(rect(a0 - w, b0, a0, b1), (0, 1), mat, c0=-0.02, c1=prot, bevel=0.012, prio=0)
    wall.member(rect(a1, b0, a1 + w, b1), (0, 1), mat, c0=-0.02, c1=prot, bevel=0.012, prio=0)
    wall.member(rect(a0 - w, b1, a1 + w, b1 + w), (1, 0), mat, c0=-0.02, c1=prot, bevel=0.012, prio=0)
    if sill:
        wall.member(rect(a0 - w - 0.04, b0 - sill, a1 + w + 0.04, b0), (1, 0), mat, c0=-0.02, c1=prot + 0.035, bevel=0.012, prio=0)
    if keystone:
        c = (a0 + a1) / 2
        wall.member([(c - 0.11, b1 - 0.02), (c + 0.11, b1 - 0.02), (c + 0.14, b1 + w + 0.06), (c - 0.14, b1 + w + 0.06)], (0, 1), mat, c0=-0.02, c1=prot + 0.02, bevel=0.012, prio=-1)


def plinth(wall, y_top, mat='Sandstein', prot=0.04, h=0.40):
    wall.member(rect(wall.amin - 0.02, -20, wall.amax + 0.02, y_top), (1, 0), mat, c0=-0.02, c1=prot, bevel=0.015, prio=-2)


# ============================================================================================== Fallrohr
def downpipe(ctx, top3, wall_pt3, wall_n, y_ground, r=0.05, mat='Zink'):
    """Fallrohr: von Rinnenpunkt top3 mit Schwanenhals an die Wand (wall_pt3, Normale wall_n) und senkrecht zum Boden"""
    top3 = np.asarray(top3, float); w = np.asarray(wall_pt3, float); n = unit(wall_n)
    x = w + n * 0.09
    y1 = top3[1] - 0.25
    path = [top3 - np.array([0, 0.04, 0]), np.array([top3[0], y1, top3[2]]),
            np.array([x[0], y1 - 0.35, x[2]]), np.array([x[0], y_ground + 0.25, x[2]]),
            np.array([x[0], y_ground + 0.05, x[2]]) + n * 0.12]
    ctx.mb.tube(mat, path, r, n=8)
    # Rohrschellen
    for y in np.arange(y_ground + 0.9, y1 - 0.5, 1.6):
        p = np.array([x[0], y, x[2]])
        ctx.mb.tube(mat, [p - np.array([0, 0.025, 0]), p + np.array([0, 0.025, 0])], r + 0.012, n=8)


# ============================================================================================== Efeu
def ivy(ctx, wall, region, holes=(), seed=1, cell=0.45, depth=(0.05, 0.22), layers=2):
    """Efeu-Huelle vor der Wand: Gitter ueber region (Polygon in a,b), ohne holes (Fenster + Rand)"""
    rng = np.random.default_rng(seed)
    R = Polygon(region).buffer(0)
    if holes:
        R = R.difference(unary_union([Polygon(h).buffer(0.06) for h in holes]))
    a0, b0, a1, b1 = R.bounds
    na = int((a1 - a0) / cell) + 2; nb = int((b1 - b0) / cell) + 2
    for L in range(layers):
        D = rng.uniform(*depth, size=(nb + 1, na + 1)) + L * 0.05
        J = rng.uniform(-0.12, 0.12, size=(nb + 1, na + 1, 2)) * cell
        for j in range(nb):
            for i in range(na):
                ca, cb = a0 + (i + 0.5) * cell, b0 + (j + 0.5) * cell
                if not R.contains(Point(ca, cb)): continue
                if rng.random() < 0.06 * (L + 1): continue
                q = []
                for (di, dj) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    a = a0 + (i + di) * cell + J[j + dj, i + di, 0]
                    b = b0 + (j + dj) * cell + J[j + dj, i + di, 1]
                    q.append(wall.F.P([a, b, D[j + dj, i + di]]))
                uvo = rng.uniform(0, 4, 2)
                uv = np.array([[0, 0], [cell, 0], [cell, cell], [0, cell]]) / 1.6 + uvo
                ctx.mb.quad('Efeu', *q, uv=uv)


# ============================================================================================== Schild
SIGNS = None


def sign(ctx, F, a0, b0, w, h, key, c=0.02, depth=0.04, mat='Schilder', frame_mat='Holz_dunkel'):
    global SIGNS
    if SIGNS is None:
        SIGNS = json.load(open(os.path.join(HERE, 'build', 'tex', 'signs.json')))
    u0, v0, u1, v1 = SIGNS[key]['uv']
    P = [F.P([a0, b0, c + depth]), F.P([a0 + w, b0, c + depth]), F.P([a0 + w, b0 + h, c + depth]), F.P([a0, b0 + h, c + depth])]
    ctx.mb.quad(mat, *P, uv=np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]))
    if depth > 0.005:
        ctx.mb.box(frame_mat, F, a0, a0 + w, b0, b0 + h, c, c + depth - 0.0005, skip=('-c', '+c'))


# ============================================================================================== Innenwaende (Dachspruenge, Gauben)
def internal_walls(ctx, rs, outline, mat_fn, step=0.25, decor=None):
    """Senkrechte Wandflaechen an Teilgrenzen, wo ein Dachteil hoeher liegt als das Nachbardach (Gaubenfronten,
    -wangen, Zwerchhaeuser, Dachspruenge). mat_fn(part_index, seg) -> Material; decor(ctx, wall, part_index, seg_index)"""
    OL = Polygon(outline)
    obound = OL.exterior
    for k, (reg, planes) in enumerate(rs.parts):
        ring = np.array(reg.exterior.coords)
        if area2(ring[:-1]) < 0: ring = ring[::-1]
        for s in range(len(ring) - 1):
            p0, p1 = ring[s], ring[s + 1]
            seg = LineString([p0, p1])
            if seg.length < 0.05: continue
            if obound.distance(Point(seg.interpolate(0.5, normalized=True))) < 0.08 and \
               obound.distance(Point(p0)) < 0.1 and obound.distance(Point(p1)) < 0.1:
                continue                                       # Aussenwand -> Fassade
            d = unit(p1 - p0); nout = np.array([d[1], -d[0]])  # nach aussen (CCW)
            # Bruchstellen: Schnitte mit allen Ebenenpolygon-Raendern
            ts = set([0.0, seg.length])
            for g in rs.pp.values():
                for pg in as_polys(g):
                    it = seg.intersection(pg.boundary)
                    for q in getattr(it, 'geoms', [it]):
                        if q.is_empty: continue
                        for c in (q.coords if hasattr(q, 'coords') else []):
                            ts.add(float(seg.project(Point(c))))
            for t in np.arange(0, seg.length, step): ts.add(float(t))
            ts = sorted(ts)
            top, bot = [], []
            for t in ts:
                pt = p0 + d * t
                pin = pt - nout * 0.03; pout = pt + nout * 0.03
                hin = min(pl.h(pin[0], pin[1]) for pl in planes)
                hg = rs.h(pin[0], pin[1])
                if hg is not None and hg > hin + 0.02: hin = None     # anderer Teil liegt hier hoeher
                hout = rs.h(pout[0], pout[1]) if OL.buffer(-0.01).contains(Point(pout)) else None
                top.append(hin); bot.append(hout)
            # zusammenhaengende Abschnitte mit hin > hout
            run = []
            def flush(run):
                if len(run) < 2: return
                F = Frame.facade(p1, p0)          # von aussen: links = p1
                Ls = seg.length
                pts = [(Ls - t, yb - 0.25) for (t, yt, yb) in run] + [(Ls - t, yt - 0.03) for (t, yt, yb) in run[::-1]]
                poly = Polygon(pts).buffer(0)
                if poly.area < 0.02: return
                w = Wall(ctx, F, np.array(as_polys(poly)[0].exterior.coords)[:-1], mat_fn(k, s), L=Ls)
                if decor: decor(ctx, w, k, s)
                w.emit()
            for t, yt, yb in zip(ts, top, bot):
                if yt is not None and yb is not None and yt > yb + 0.03:
                    run.append((t, yt, yb))
                else:
                    flush(run); run = []
            flush(run)

"""Bauteile des Alten Rathauses (Backsteingotik): Profile, Waende mit Oeffnungen, Fenster, Nischen, Fialen,
Staffelgiebel, Zwerchhaeuser, Dachflaechen mit Ziegelschichten, Friese, Rautenbaender, Rinnen/Fallrohre, Lauben.
Alle Masse in Metern; Fassadenframe: a entlang (rechts), b oben, c aussen."""
import math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, box as sbox
from shapely.ops import unary_union
from ar_geom import Frame, MB, arch_outline, ring_between, side_band, unit, area2, trs_matrix, frame_matrix, UP, arc_param

# Materialnamen
BR = 'AR_Backstein'; BK = 'AR_Glasur'; BB = 'AR_Backstein_Fiale'; B19 = 'AR_Backstein19'
SS = 'AR_Sandstein'; QD = 'AR_Quader'; RF = 'AR_Dachziegel'; RG = 'AR_First'; SL = 'AR_Schiefer'
PL = 'AR_Putz'; GR = 'AR_Granit'; FR = 'AR_Fries'; QF = 'AR_Vierpass'; RS = 'AR_Rosette'; TR = 'AR_Masswerk_gruen'
TRR = 'AR_Masswerk_rot'; TRB = 'AR_Masswerk_ziegel'
FRM = 'AR_Fensterrahmen'; GL = 'AR_Fensterglas'; SH = 'AR_Schaufenster'; ZN = 'AR_Zink'; FE = 'AR_Eisen'
CU = 'AR_Kupfer'; DR = 'AR_Tuer'; DRG = 'AR_Tuer_gruen'; DK = 'AR_Dunkel'; ST = 'AR_Stahl'; GD = 'AR_Glasdach'
BIT = 'AR_Dachbahn'; SPL = 'AR_Fialhelm'; WE = 'AR_Wechsel'; BG = 'AR_Backstein_Giebel'


# ============================================================================ Profil-Sweep entlang Kette
def chain_frames(pts, closed=False):
    """Planpunkte (x,z) -> Tangenten, Aussennormalen (n = (-tz, tx)), Gehrungsvektoren je Punkt"""
    P = np.asarray(pts, float)
    k = len(P)
    seg = []
    rng_ = range(k) if closed else range(k - 1)
    for i in rng_:
        d = P[(i + 1) % k] - P[i]; d = d / np.linalg.norm(d)
        seg.append(d)
    seg = np.array(seg)
    nrm = np.stack([-seg[:, 1], seg[:, 0]], 1)
    miter = np.zeros((k, 2))
    for i in range(k):
        if closed:
            n0 = nrm[i - 1]; n1 = nrm[i % len(nrm)]
        else:
            n0 = nrm[max(0, i - 1)]; n1 = nrm[min(i, len(nrm) - 1)]
        m = n0 + n1
        den = 1 + float(np.dot(n0, n1))
        miter[i] = m / den if den > 1e-6 else n1
    return P, seg, nrm, miter


def sweep(mb, mat, pts, profile, closed=False, caps=True, y0=0.0, mats=None, uv_along=True):
    """profile: Liste (o, y) (o = Abstand nach aussen, y absolut); Reihenfolge Wand unten -> aussen -> oben -> Wand.
    pts: Planpunkte, von aussen gesehen links -> rechts (Normale = links der Laufrichtung in x/z ... n = (-tz, tx))."""
    P, seg, nrm, miter = chain_frames(pts, closed)
    prof = np.asarray(profile, float)
    k = len(P); m = len(prof)
    # 3D-Gitter (k, m, 3)
    G = np.zeros((k, m, 3))
    for i in range(k):
        for j in range(m):
            q = P[i] + miter[i] * prof[j, 0]
            G[i, j] = (q[0], prof[j, 1] + y0, q[1])
    # Laengen fuer UV
    sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    idfr = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    rngs = range(k) if closed else range(k - 1)
    for i in rngs:
        i2 = (i + 1) % k
        t3 = np.array([seg[i][0], 0, seg[i][1]]); n3 = np.array([nrm[i][0], 0, nrm[i][1]])
        for j in range(m - 1):
            A, B_, C, D = G[i, j], G[i2, j], G[i2, j + 1], G[i, j + 1]
            mt = mats[j] if mats else mat
            if mt is None: continue
            # UV: entlang (a) und Profil-Bogenlaenge / Hoehe
            pa = [float(np.dot(np.array([x[0], 0, x[2]]), t3)) for x in (A, B_, C, D)]
            pv = [x[1] if abs(prof[j + 1, 1] - prof[j, 1]) > abs(prof[j + 1, 0] - prof[j, 0]) else float(np.dot(np.array([x[0], 0, x[2]]), n3)) for x in (A, B_, C, D)]
            uv = np.array([[[pa[0], pv[0]], [pa[1], pv[1]], [pa[2], pv[2]]], [[pa[0], pv[0]], [pa[2], pv[2]], [pa[3], pv[3]]]])
            mb.tri_local(mt, idfr, np.array([[A, B_, C], [A, C, D]]), uv=uv)
    if caps and not closed:
        for (i, sgn) in ((0, -1), (k - 1, 1)):
            ring = [G[i, j] for j in range(m)]
            t = np.array([seg[min(i, len(seg) - 1)][0], 0, seg[min(i, len(seg) - 1)][1]]) * sgn
            mb.poly(mats[0] if mats else mat, idfr, ring, hint=t)
    return G


# ============================================================================ Wandflaeche mit Ausschnitten
def wall_face(mb, fr, outline, cut_polys=(), mat=BR, c=0.0):
    """outline (k,2) in (a,b); cut_polys: Liste von (k,2) Umrissen (Oeffnungen, Baender) -> Frontflaeche bei c"""
    P = Polygon(outline).buffer(0)
    cuts = [Polygon(cp).buffer(0) for cp in cut_polys if len(cp) >= 3]
    if cuts:
        P = P.difference(unary_union(cuts))
    geoms = list(P.geoms) if isinstance(P, MultiPolygon) else [P]
    for g in geoms:
        if g.is_empty or g.area < 1e-6: continue
        ext = np.array(g.exterior.coords)[:-1]
        holes = [np.array(h.coords)[:-1] for h in g.interiors]
        mb.poly(mat, fr, [(x, y, c) for x, y in ext], holes=[[(x, y, c) for x, y in h] for h in holes], hint=(0, 0, 1))


# ============================================================================ Oeffnungen
class Opening:
    """Oeffnung in einer Wand (Fassadenframe), Mitte a, Unterkante y0.
    kind: rect|seg|point|round; w Breite; hs Kaempferhoehe ueber y0; rise Pfeil; orders [(inset, tiefe), ...] (erste = (0, d1));
    ring: dict(w=Breite, proj=Vorsprung, alt=bool) Keilsteinring; content: 'window'|'niche'|'dark'|'none'|'door'|'shop'
    """

    def __init__(self, a, y0, w, hs, kind='seg', rise=None, orders=((0, 0.12), (0.10, 0.30)), ring=None, content='window',
                 lights=2, bars_h=0.42, bars_v=1, tymp=None, sill=True, n=18, nj=4, mat_reveal=BR, back_mat=BR, frame_mat=FRM,
                 glass=GL, transom=None, mullion_mat=SS, skip_bottom=False, door_mat=DR):
        self.a = a; self.y0 = y0; self.w = w; self.hs = hs; self.kind = kind; self.rise = rise
        self.orders = list(orders); self.ring = ring; self.content = content; self.lights = lights; self.bars_h = bars_h
        self.bars_v = bars_v; self.tymp = tymp; self.sill = sill; self.n = n; self.nj = nj; self.mat_reveal = mat_reveal
        self.back_mat = back_mat; self.frame_mat = frame_mat; self.glass = glass; self.transom = transom
        self.mullion_mat = mullion_mat; self.skip_bottom = skip_bottom; self.door_mat = door_mat

    def outline(self, inset=0.0):
        k = self.kind
        w = self.w - 2 * inset
        if k == 'point':
            r0 = self.rise if self.rise else self.w * 0.85
            rise = max(0.05, r0 - inset * 1.1)
        elif k == 'seg':
            r0 = self.rise if self.rise else self.w * 0.18
            rise = max(0.02, r0 - inset * 0.4)
        else:
            rise = self.rise
        y0 = self.y0 + (0 if (self.skip_bottom or not self.sill) else 0)
        o = _outline(k, w, self.hs, rise, self.n, self.nj, y0)
        o[:, 0] += self.a
        return o

    def top(self):
        o = self.outline(0); return float(o[:, 1].max())


def _outline(kind, w, hs, rise, n, nj, y0):
    """Umriss mit fester Punktzahl: unten links, unten rechts, rechte Laibung (nj), Bogen (n-1 innen), linke Laibung (nj)"""
    hw = w / 2
    ys = y0 + hs
    pts = [(-hw, y0), (hw, y0)]
    for j in range(1, nj + 1):
        pts.append((hw, y0 + hs * j / nj))
    if kind in ('rect', 'flat'):
        top = [(hw - w * i / n, ys) for i in range(1, n)]
    elif kind == 'round':
        top = [(hw * math.cos(math.pi * i / n), ys + hw * math.sin(math.pi * i / n)) for i in range(1, n)]
    elif kind == 'seg':
        r = rise
        R = (hw * hw + r * r) / (2 * r); cy = ys + r - R
        a0 = math.atan2(ys - cy, hw)
        top = [(R * math.cos(a0 + (math.pi - 2 * a0) * i / n), cy + R * math.sin(a0 + (math.pi - 2 * a0) * i / n)) for i in range(1, n)]
    elif kind == 'point':
        r = rise
        R = (hw * hw + r * r) / (2 * hw); cx = hw - R
        amax = math.atan2(r, -cx)
        half = n // 2
        right = [(cx + R * math.cos(amax * i / half), ys + R * math.sin(amax * i / half)) for i in range(1, half)]
        top = right + [(0.0, ys + r)] + [(-x, y) for x, y in right[::-1]]
    else:
        raise ValueError(kind)
    pts += top
    for j in range(nj, 0, -1):
        pts.append((-hw, y0 + hs * j / nj))
    return np.array(pts, float)


def ring_outline(op, rw):
    """aeusserer Keilsteinring: Umriss um rw nach aussen (gleiche Punktzahl)"""
    k = op.kind
    w = op.w + 2 * rw
    if k == 'point':
        r0 = op.rise if op.rise else op.w * 0.85; rise = r0 + rw * 1.1
    elif k == 'seg':
        r0 = op.rise if op.rise else op.w * 0.18; rise = r0 + rw * 0.4
    else:
        rise = op.rise
    o = _outline(k, w, op.hs, rise, op.n, op.nj, op.y0)
    o[:, 0] += op.a
    return o


def build_opening(mb, fr, op, protos=None):
    """Laibungen, Keilsteinring, Inhalt. Gibt den Ausschnitt-Umriss fuer die Wandflaeche zurueck."""
    ords = op.orders
    outs = [op.outline(ins) for ins, _ in ords]
    depth = 0.0
    alt = lambda i: BK if (i % 2) else BR
    ring = op.ring
    cut = outs[0]
    s_ring = None
    if ring:
        rw = ring.get('w', 0.14); proj = ring.get('proj', 0.0)
        ro = ring_outline(op, rw)
        cut = ro
        nseg = len(ro)
        if ring.get('alt', True) and 'm1' not in ring and 'm2' not in ring:
            # Keilsteine + Gewaende im Wechsel rot / schwarz glasiert: eine Flaeche mit Wechsel-Textur (u = Bogenlaenge)
            s_ring = arc_param(ro, outs[0])
            ring_between(mb, WE, fr, ro, outs[0], proj, skip_bottom=True, s=s_ring)
            if proj > 0:
                side_band(mb, WE, fr, ro, 0.0, proj, skip_bottom=True, inward=False, s=s_ring)
                side_band(mb, WE, fr, outs[0], 0.0, proj, skip_bottom=True, inward=True, s=s_ring)
        else:
            if ring.get('alt', True):
                mats_fn = lambda i: (ring.get('m2', BK) if (i % 2) else ring.get('m1', BR))
            else:
                mats_fn = lambda i: ring.get('m1', BR)
            ring_between(mb, ring.get('m1', BR), fr, ro, outs[0], proj, skip_bottom=True, mats_fn=mats_fn)
            if proj > 0:
                # Aussenkante des Rings (Normalen nach aussen) und Innenkante bis Wandebene
                side_band(mb, BR, fr, ro, 0.0, proj, skip_bottom=True, inward=False, mats_fn=mats_fn)
                side_band(mb, op.mat_reveal, fr, outs[0], 0.0, proj, skip_bottom=True, inward=True, mats_fn=mats_fn)
    for k, (ins, dep) in enumerate(ords):
        d0 = depth; d1 = dep
        if k == 0 and s_ring is not None and op.mat_reveal == BR:
            # erste Laibung: Wechselsteine laufen um die Kante (Kopfseiten der Keilsteine)
            side_band(mb, WE, fr, outs[0], -d1, -d0, skip_bottom=True, s=s_ring)
            if not op.skip_bottom:      # Bruestungskante (Unterkante der Oeffnung), Normale nach oben
                p, q = outs[0][0], outs[0][1]
                mb.quad(op.mat_reveal, fr, (p[0], p[1], -d0), (q[0], q[1], -d0), (q[0], q[1], -d1), (p[0], p[1], -d1))
            depth = d1
            if k + 1 < len(ords):
                ring_between(mb, op.mat_reveal, fr, outs[k], outs[k + 1], -d1, skip_bottom=op.skip_bottom)
            continue
        side_band(mb, op.mat_reveal, fr, outs[k], -d1, -d0, skip_bottom=op.skip_bottom, mats_fn=(lambda i: op.mat_reveal) if k else None)
        if k + 1 < len(ords):
            ring_between(mb, op.mat_reveal, fr, outs[k], outs[k + 1], -d1, skip_bottom=op.skip_bottom)
        depth = d1
    inner = outs[-1]; D = depth
    # Sohlbank
    if op.sill and not op.skip_bottom and op.content in ('window', 'niche', 'dark'):
        x0, x1 = outs[0][0, 0], outs[0][1, 0]
        sw = 0.05
        y = op.y0
        mb.box(SS, fr, x0 - sw, x1 + sw, y - 0.11, y, 0.0, 0.07, faces='XxYyZ')
    c = op.content
    if c == 'niche':
        mb.poly(op.back_mat, fr, [(p[0], p[1], -D) for p in inner], hint=(0, 0, 1))
    elif c == 'dark':
        mb.poly(DK, fr, [(p[0], p[1], -D) for p in inner], hint=(0, 0, 1))
    elif c == 'window':
        window_fill(mb, fr, op, inner, D, protos)
    elif c == 'door':
        door_fill(mb, fr, op, inner, D)
    elif c == 'shop':
        shop_fill(mb, fr, op, inner, D)
    return cut


def window_fill(mb, fr, op, inner, D, protos=None):
    """Rahmen, Pfosten, Kaempfer, Sprossen, Glas, ggf. Tympanon mit Rosetten"""
    fw = 0.055; fd = 0.06
    ys = op.y0 + op.hs   # Kaempfer (ohne Einzug)
    xs0, xs1 = inner[0, 0], inner[1, 0]
    W = xs1 - xs0
    top = float(inner[:, 1].max())
    tymp = op.tymp
    if tymp:
        # Tympanon: Feld oberhalb des Kaempfers, Lichter enden an ys - eps
        yt = ys
        # Tympanonflaeche (Backstein / Terrakotta), leicht zurueck
        tpts = [p for p in inner if p[1] >= yt - 1e-6]
        tpts = np.array([(xs1, yt)] + [tuple(p) for p in inner[2 + op.nj:len(inner) - op.nj]] + [(xs0, yt)])
        mb.poly(tymp.get('mat', BR), fr, [(p[0], p[1], -D - 0.02) for p in tpts], hint=(0, 0, 1))
        # Laibungsrest zwischen Laibungsende (-D) und dem 2 cm zurueckliegenden Tympanon: sonst offener Schlitz entlang
        # des Bogens, durch den man schraeg ins leere Innere sieht (Strahltest 10.11.)
        arch_ol = np.array([(xs0, yt), (xs1, yt)] + [tuple(p) for p in inner[2 + op.nj:len(inner) - op.nj]])
        side_band(mb, tymp.get('mat', BR), fr, arch_ol, -D - 0.02, -D, skip_bottom=True)
        # Kaempfer-Steinbalken
        mb.box(SS, fr, xs0, xs1, yt - 0.10, yt, -D - 0.10, -D + 0.04, faces='XxYyZ')
        # Rosetten
        nr = tymp.get('n', 2); rr = tymp.get('r', min(W / (2 * nr + 0.6), (top - yt) * 0.38))
        for i in range(nr):
            cx = xs0 + W * (i + 0.5) / nr
            cy = yt + (top - yt) * 0.45
            if protos is not None:
                M = frame_matrix(fr.local(cx, cy, -D - 0.02), s=rr)
                mb.instance(tymp.get('proto', 'rosette'), M)
        light_top = yt - 0.10
        light_outline = None
    else:
        light_top = None
    # Lichter
    nl = max(1, op.lights)
    mw = 0.12 if nl > 1 else 0
    lw = (W - mw * (nl - 1)) / nl
    for i in range(nl):
        x0 = xs0 + i * (lw + mw); x1 = x0 + lw
        if light_top is not None:
            lo = np.array([(x0, op.y0), (x1, op.y0), (x1, light_top), (x0, light_top)])
        elif nl == 1:
            lo = inner
        else:
            # Licht = innerer Umriss geschnitten mit Streifen [x0, x1]
            Pp = Polygon(inner).intersection(sbox(x0, op.y0 - 1, x1, 100))
            if Pp.is_empty: continue
            if isinstance(Pp, MultiPolygon): Pp = max(Pp.geoms, key=lambda g: g.area)
            lo = np.array(Pp.exterior.coords)[:-1]
            if area2(lo) < 0: lo = lo[::-1]
        _light(mb, fr, lo, D, fw, fd, op)
        if i > 0 and mw > 0:
            # Pfosten (Sandstein) zwischen den Lichtern
            xm0 = x0 - mw; xm1 = x0
            ytop = light_top if light_top is not None else _y_at(inner, (xm0 + xm1) / 2) - 0.0
            mb.box(op.mullion_mat, fr, xm0, xm1, op.y0, ytop, -D - 0.06, -D + 0.06, faces='XxYZ')


def _y_at(outline, x):
    """obere Kante eines Umrisses bei x (max y)"""
    L = LineString([(x, -100), (x, 100)]).intersection(Polygon(outline))
    if L.is_empty: return float(outline[:, 1].max())
    return float(np.array(L.coords)[:, 1].max()) if hasattr(L, 'coords') else float(max(np.array(g.coords)[:, 1].max() for g in L.geoms))


def _light(mb, fr, lo, D, fw, fd, op):
    """ein Fensterlicht: Rahmenring, Glas, Sprossen"""
    P = Polygon(lo)
    inn = P.buffer(-fw, join_style=2)
    if inn.is_empty:
        mb.poly(op.glass, fr, [(p[0], p[1], -D - fd) for p in lo], hint=(0, 0, 1)); return
    if isinstance(inn, MultiPolygon): inn = max(inn.geoms, key=lambda g: g.area)
    ii = np.array(inn.exterior.coords)[:-1]
    if area2(ii) < 0: ii = ii[::-1]
    # Rahmen-Frontflaeche (Ring) bei c = -D
    mb.poly(op.frame_mat, fr, [(p[0], p[1], -D) for p in lo], holes=[[(p[0], p[1], -D) for p in ii]], hint=(0, 0, 1))
    # innere Rahmenkante bis Glas
    for i in range(len(ii)):
        p, q = ii[i], ii[(i + 1) % len(ii)]
        mb.quad(op.frame_mat, fr, (p[0], p[1], -D), (q[0], q[1], -D), (q[0], q[1], -D - fd), (p[0], p[1], -D - fd))
    mb.poly(op.glass, fr, [(p[0], p[1], -D - fd) for p in ii], hint=(0, 0, 1))
    # Sprossen
    x0, y0, x1, y1 = inn.bounds
    bw = 0.024; bd = 0.025
    if op.bars_h:
        nb = int((y1 - y0) / op.bars_h)
        for k in range(1, nb + 1):
            y = y0 + (y1 - y0) * k / (nb + 1)
            L = LineString([(x0 - 1, y), (x1 + 1, y)]).intersection(inn)
            if L.is_empty or not hasattr(L, 'coords'): continue
            c_ = np.array(L.coords)
            mb.box(op.frame_mat, fr, c_[:, 0].min(), c_[:, 0].max(), y - bw / 2, y + bw / 2, -D - fd, -D - fd + bd, faces='YyZ')
    if op.transom:
        y = op.transom
        L = LineString([(x0 - 1, y), (x1 + 1, y)]).intersection(inn)
        if not L.is_empty and hasattr(L, 'coords'):
            c_ = np.array(L.coords)
            mb.box(op.frame_mat, fr, c_[:, 0].min(), c_[:, 0].max(), y - 0.035, y + 0.035, -D - fd, -D - 0.005, faces='YyZ')
    for k in range(1, op.bars_v + 1):
        x = x0 + (x1 - x0) * k / (op.bars_v + 1)
        L = LineString([(x, y0 - 1), (x, y1 + 1)]).intersection(inn)
        if L.is_empty or not hasattr(L, 'coords'): continue
        c_ = np.array(L.coords)
        mb.box(op.frame_mat, fr, x - bw / 2, x + bw / 2, c_[:, 1].min(), c_[:, 1].max(), -D - fd, -D - fd + bd * 1.2, faces='XxZ')


def door_fill(mb, fr, op, inner, D):
    """zweifluegelige Holztuer mit Fuellungen, Oberlicht im Bogen"""
    ys = op.y0 + op.hs
    x0, x1 = inner[0, 0], inner[1, 0]
    # Zarge: schliesst den Spalt zwischen Laibungsende (-D) und Tuerblatt/Oberlicht (-D-0.06), inkl. Schwelle
    side_band(mb, op.door_mat, fr, inner, -D - 0.065, -D, skip_bottom=False)
    # Oberlicht (Bogenfeld) als Glas mit Rahmen
    top = np.array([p for p in inner[2 + op.nj:len(inner) - op.nj]])
    if len(top) and float(inner[:, 1].max()) - ys > 0.15:
        tp = np.array([(x1, ys)] + [tuple(p) for p in top] + [(x0, ys)])
        mb.poly(op.glass, fr, [(p[0], p[1], -D - 0.05) for p in tp], hint=(0, 0, 1))
        mb.box(op.frame_mat, fr, x0, x1, ys - 0.08, ys, -D - 0.06, -D + 0.0, faces='YyZ')
    dm = op.door_mat
    mb.poly(dm, fr, [(x0, op.y0, -D - 0.06), (x1, op.y0, -D - 0.06), (x1, ys - 0.08, -D - 0.06), (x0, ys - 0.08, -D - 0.06)], hint=(0, 0, 1))
    # Fuellungen + Mittelstoss
    xm = (x0 + x1) / 2
    mb.box(dm, fr, xm - 0.03, xm + 0.03, op.y0, ys - 0.08, -D - 0.06, -D - 0.02, faces='XxZ')
    for (a0, a1) in ((x0 + 0.09, xm - 0.08), (xm + 0.08, x1 - 0.09)):
        for (b0, b1) in ((op.y0 + 0.25, op.y0 + (ys - op.y0) * 0.45), (op.y0 + (ys - op.y0) * 0.52, ys - 0.25)):
            if b1 - b0 > 0.2 and a1 - a0 > 0.1:
                mb.box(dm, fr, a0, a1, b0, b1, -D - 0.06, -D - 0.035, faces='XxYyZ')
    # Griffe
    for sx in (-1, 1):
        mb.box(FE, fr, xm + sx * 0.06 - 0.012, xm + sx * 0.06 + 0.012, op.y0 + 0.95, op.y0 + 1.35, -D - 0.06, -D, faces='XxYyZ')


def shop_fill(mb, fr, op, inner, D):
    """Ladenfront (Stahl-Glas mit Tuer, geschlossene Laibung, beleuchteter Raum, Auslagen): siehe ar_shops.shop_front"""
    from ar_shops import shop_front
    shop_front(mb, fr, op, inner, D)


# ============================================================================ Prototypen (instanziert)
def proto_rosette():
    """Sechspass-Rosette, Radius 1 (skaliert), Front +c, Ruecken bei c=0"""
    mb = MB()
    fr = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    n = 36
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.stack([np.cos(ang), np.sin(ang)], 1) * 1.0
    # Innenkontur: 6 Paesse
    r_in = 0.78 - 0.16 * np.abs(np.cos(3 * ang)) ** 0.6
    inner = np.stack([np.cos(ang) * r_in, np.sin(ang) * r_in], 1)
    d = 0.09
    mb.extrude(TR if False else 'AR_Formstein', fr, outer, 0.0, d, front=True, back=False, holes=[inner[::-1]])
    # Nabe + Speichen (Sternrosette)
    hub = np.stack([np.cos(ang[::3]), np.sin(ang[::3])], 1) * 0.16
    mb.extrude('AR_Formstein', fr, hub, 0.0, d * 0.9, front=True, back=False)
    for k in range(6):
        a = k * np.pi / 3 + np.pi / 6
        ca, sa = math.cos(a), math.sin(a)
        sp = np.array([[0.12, -0.035], [0.70, -0.025], [0.70, 0.025], [0.12, 0.035]])
        rot = np.array([[ca, -sa], [sa, ca]])
        mb.extrude('AR_Formstein', fr, sp @ rot.T, 0.0, d * 0.7, front=True, back=False)
    return mb


def proto_crown(kind='std'):
    """Fialenbekroenung ueber einem uebereck gestellten Quadratschaft (Halbdiagonale 1.0 -> skaliert).
    Ursprung = Schaftoberkante Mitte. Wasserschlag, vier Wimperge, Pyramidenhelm, Knauf, Eisenstab, Wetterfahne."""
    mb = MB()
    fr = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    r = 1.0   # Halbdiagonale des Schafts
    dia = lambda rr: np.array([[rr, 0], [0, -rr], [-rr, 0], [0, rr]])    # Raute in (a,c)
    # Gesims (vorkragend)
    mb.prism(BK, fr, dia(r * 1.18), 0.0, 0.14, top=False)
    mb.prism(BK, fr, dia(r * 1.18), 0.14, 0.18, top=False)
    # Schraege von 1.18 auf 1.0
    _frustum(mb, BK, fr, dia(r * 1.18), dia(r * 0.98), 0.18, 0.30)
    # kurzer Schaftaufsatz mit Wimpergen (je Seite ein Dreieck-Giebel)
    mb.prism(BB, fr, dia(r * 0.98), 0.30, 1.00, top=False)
    corners = dia(r * 0.98)
    for i in range(4):
        p = corners[i]; q = corners[(i + 1) % 4]
        mid = (p + q) / 2
        nrm = mid / np.linalg.norm(mid)
        apex = mid + nrm * 0.06
        # Giebeldreieck vor der Schaftseite
        A = (p[0] + nrm[0] * 0.06, 1.00, p[1] + nrm[1] * 0.06); B = (q[0] + nrm[0] * 0.06, 1.00, q[1] + nrm[1] * 0.06)
        T = (apex[0], 1.55, apex[1])
        mb.poly(BK, fr, [A, B, T], hint=(nrm[0], 0, nrm[1]))
        # Unterkante des Giebels (sichtbar von unten) + Dachschraegen der Wimperge
        mb.poly(BK, fr, [(p[0], 1.00, p[1]), (q[0], 1.00, q[1]), B, A], hint=(0, -1, 0))
        mb.poly(SPL, fr, [A, T, (0, 1.75, 0)], hint=(nrm[0] - nrm[1] * 0, 0.6, nrm[1]))
        mb.poly(SPL, fr, [T, B, (0, 1.75, 0)], hint=(nrm[0], 0.6, nrm[1]))
    # Helm (vierseitige Pyramide, gestreckt)
    base = dia(r * 0.98)
    top_y = 4.2
    for i in range(4):
        p = base[i]; q = base[(i + 1) % 4]
        mb.poly(SPL, fr, [(p[0], 1.00, p[1]), (q[0], 1.00, q[1]), (0, top_y, 0)], hint=((p[0] + q[0]), 0.3, (p[1] + q[1])))
    # Knauf (Oktaeder-artig) + Eisenstab + Fahne
    kn = 0.22
    pts = [(kn, top_y + 0.05, 0), (0, top_y + 0.05, -kn), (-kn, top_y + 0.05, 0), (0, top_y + 0.05, kn)]
    for i in range(4):
        p = pts[i]; q = pts[(i + 1) % 4]
        mb.poly(FE, fr, [p, q, (0, top_y + 0.05 + kn, 0)], hint=(p[0] + q[0], 0.5, p[2] + q[2]))
        mb.poly(FE, fr, [q, p, (0, top_y + 0.05 - kn * 0.6, 0)], hint=(p[0] + q[0], -0.5, p[2] + q[2]))
    mb.prism(FE, fr, dia(0.05), top_y + 0.2, top_y + 1.9, top=True)
    # Kreuzblume (vier Blaetter)
    for i in range(4):
        a = i * math.pi / 2
        ca, sa = math.cos(a), math.sin(a)
        mb.poly(FE, fr, [(0.04 * ca, top_y + 0.55, 0.04 * sa), (0.26 * ca, top_y + 0.62, 0.26 * sa), (0.04 * ca, top_y + 0.85, 0.04 * sa)], hint=(-sa, 0, ca))
        mb.poly(FE, fr, [(0.04 * ca, top_y + 0.85, 0.04 * sa), (0.26 * ca, top_y + 0.62, 0.26 * sa), (0.04 * ca, top_y + 0.55, 0.04 * sa)], hint=(sa, 0, -ca))
    # Wetterfahne (Kupfer, zweiseitig als duenne Platte)
    mb.box(CU, fr, 0.05, 0.75, top_y + 1.35, top_y + 1.72, -0.012, 0.012)
    return mb


def _frustum(mb, mat, fr, lo, hi, y0, y1):
    k = len(lo)
    for i in range(k):
        p, q = lo[i], lo[(i + 1) % k]; p2, q2 = hi[i], hi[(i + 1) % k]
        mid = (p + q) / 2
        mb.poly(mat, fr, [(p[0], y0, p[1]), (q[0], y0, q[1]), (q2[0], y1, q2[1]), (p2[0], y1, p2[1])], hint=(mid[0], 0.5, mid[1]))
    mb.poly(mat, fr, [(p[0], y0, p[1]) for p in lo], hint=(0, -1, 0))


def proto_frieze(hmap, L=2.4, H=0.55, nu=80, nv=16, depth=0.05):
    """Terrakotta-Fries-Modul (L x H) mit Reliefverschiebung aus Hoehenkarte; Front +c, Ruecken c=0"""
    mb = MB(tiles={FR: (L, H)})
    fr = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    hh, ww = hmap.shape
    us = np.linspace(0, 1, nu + 1); vs = np.linspace(0, 1, nv + 1)
    G = np.zeros((nu + 1, nv + 1, 3)); UVg = np.zeros((nu + 1, nv + 1, 2))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            px = min(ww - 1, int(u * (ww - 1))); py = min(hh - 1, int((1 - v) * (hh - 1)))
            # gemittelte Hoehe (Tiefpass) gegen Treppeneffekte
            y0_ = max(0, py - 2); y1_ = min(hh, py + 3); x0_ = max(0, px - 2); x1_ = min(ww, px + 3)
            h = float(hmap[y0_:y1_, x0_:x1_].mean())
            edge = 0 if (j == 0 or j == nv) else 1
            G[i, j] = (u * L, v * H, depth * (0.15 + 0.85 * h) * edge + 0.004)
            UVg[i, j] = (u * L, v * H)
    mb.grid(FR, fr, G, smooth=True, uvG=UVg)
    # Ober-/Unterkante schliessen (Streifen zur Rueckebene)
    for j, sgn in ((0, -1), (nv, 1)):
        for i in range(nu):
            p, q = G[i, j], G[i + 1, j]
            if sgn < 0:
                mb.quad(FR, fr, (q[0], q[1], 0), (p[0], p[1], 0), tuple(p), tuple(q))
            else:
                mb.quad(FR, fr, (p[0], p[1], 0), (q[0], q[1], 0), tuple(q), tuple(p))
    return mb


# ============================================================================ Fialenschaft (Raute, uebereck)
def pinnacle_shaft(mb, fr, a, y0, y1, r=0.36, c=0.0, mat=BB, corbel=0.45, crown_scale=None, proto='crown', back=True, square=True):
    """Schaft (Halbdiagonale r) an Fassadenposition a, Mitte bei c; unten Konsolspitze.
    square=True: quadratischer Schaft mit Front parallel zur Fassade (wie an den Rathausgiebeln), sonst uebereck (Raute)."""
    f = fr.local(a, 0, c)
    if square:
        s_ = r / math.sqrt(2.0)
        dia = np.array([[s_, s_], [s_, -s_], [-s_, -s_], [-s_, s_]])     # rechts-vorn, rechts-hinten, links-hinten, links-vorn
    else:
        dia = np.array([[r, 0], [0, -r], [-r, 0], [0, r]])    # (a, c): rechts, hinten, links, vorne
    for i in range(4):
        p, q = dia[i], dia[(i + 1) % 4]
        mb.poly(mat, f, [(p[0], y0, p[1]), (q[0], y0, q[1]), (q[0], y1, q[1]), (p[0], y1, p[1])], hint=((p[0] + q[0]), 0, (p[1] + q[1])))
    # Konsolspitze unten (umgekehrte Pyramide; nur die vorderen zwei Flaechen)
    if corbel:
        tip = (0, y0 - corbel, -0.02)
        for i in range(4):
            p, q = dia[i], dia[(i + 1) % 4]
            if p[1] + q[1] < -1e-6: continue
            mb.poly(mat, f, [(q[0], y0, q[1]), (p[0], y0, p[1]), tip], hint=((p[0] + q[0]), -0.6, (p[1] + q[1])))
    if crown_scale:
        cf = Frame(f.P([0, y1, 0]), fr.ex, fr.ey, fr.ez)
        if square: cf = cf.rot_y(45.0)
        M = frame_matrix(cf, s=crown_scale)
        mb.instance(proto, M)
    else:
        mb.poly(BK, f, [(p[0], y1, p[1]) for p in dia], hint=(0, 1, 0))


# ============================================================================ Rauten-/Gitterband
def recess_walls(mb, mat, fr, a0, a1, y0, y1, recess, ends=True):
    """Leibungen eines rechteckigen Wandruecksprungs [a0,a1]x[y0,y1] (Tiefe recess): Normalen in den Ruecksprung
    (oben nach unten, unten nach oben, Enden zur Mitte) -- wie side_band(inward) eines CCW-Rechtecks."""
    R_ = np.array([(a0, y0), (a1, y0), (a1, y1), (a0, y1)])
    for i in range(4):
        if not ends and i in (1, 3): continue
        p, q = R_[i], R_[(i + 1) % 4]
        mb.quad(mat, fr, (p[0], p[1], 0.0), (q[0], q[1], 0.0), (q[0], q[1], -recess), (p[0], p[1], -recess))


def lattice_band(mb, fr, a0, a1, y0, y1, recess=0.035, proj=0.02, pitch=0.30, sw=0.075, mat=BR, bg=BK, ends=True):
    """Rautenfries: Rueckplatte (glasiert) bei -recess, diagonale Ziegelstege ueber Kreuz bis +proj.
    Der Wandausschnitt [a0,a1]x[y0,y1] muss in der Wandflaeche ausgespart sein."""
    mb.poly(bg, fr, [(a0, y0, -recess), (a1, y0, -recess), (a1, y1, -recess), (a0, y1, -recess)], hint=(0, 0, 1))
    # Ober-/Unterleibung + Enden (Normalen in den Ruecksprung)
    recess_walls(mb, mat, fr, a0, a1, y0, y1, recess, ends)
    H = y1 - y0
    band = sbox(a0, y0, a1, y1)
    for di, (dep0, dep1) in ((1, (-recess, proj)), (-1, (-recess, proj * 0.6))):
        x = a0 - H - pitch
        while x < a1 + pitch:
            # Steg als Parallelogramm von (x, y0) nach (x + di*H, y1)
            xb = x if di > 0 else x + H
            pg = Polygon([(xb, y0), (xb + sw, y0), (xb + sw + di * H, y1), (xb + di * H, y1)]).intersection(band)
            x += pitch
            if pg.is_empty or pg.area < 1e-5: continue
            if isinstance(pg, MultiPolygon): pg = max(pg.geoms, key=lambda g: g.area)
            o = np.array(pg.exterior.coords)[:-1]
            mb.extrude(mat if di > 0 else 'AR_Backstein', fr, o, dep0, dep1, front=True, back=False)


def recessed_band(mb, fr, a0, a1, y0, y1, mat, recess=0.03, proj_rails=0.03, rail_mat=BR, ends=True):
    """Texturband (Vierpass, Rosetten) zurueckgesetzt mit Leibungen"""
    mb.poly(mat, fr, [(a0, y0, -recess), (a1, y0, -recess), (a1, y1, -recess), (a0, y1, -recess)], hint=(0, 0, 1))
    recess_walls(mb, rail_mat, fr, a0, a1, y0, y1, recess, ends)


def frieze_band(mb, fr, a0, a1, y0, h=0.55, recess=0.02, L=2.4, proto='frieze'):
    """Terrakottafries: Ausschnitt [a0,a1]x[y0,y0+h]; Rueckplatte + instanzierte Reliefmodule (gestreckt auf ganze Zahl)"""
    y1 = y0 + h
    mb.poly(FR, fr, [(a0, y0, -recess), (a1, y0, -recess), (a1, y1, -recess), (a0, y1, -recess)], hint=(0, 0, 1))
    recess_walls(mb, BR, fr, a0, a1, y0, y1, recess, True)
    n = max(1, int(round((a1 - a0) / L)))
    sx = (a1 - a0) / (n * L)
    for i in range(n):
        f = fr.local(a0 + i * L * sx, y0, -recess)
        M = np.eye(4); M[:3, :3] = f.M @ np.diag([sx, 1.0, 1.0]); M[:3, 3] = f.o
        mb.instance(proto, M)


# ============================================================================ Dachflaechen
def roof_plane(mb, region_plan, eave_pt, eave_dir, slope_deg, y_eave, course=0.30, step=0.028, mat=RF, under=DK,
               region_is_local=False, clip_y=None, walls=None, soffit=PL):
    """Dachflaeche: region_plan = shapely Polygon im Plan (x, z) (projizierte Flaeche); Traufe durch eave_pt (x,z) mit
    Richtung eave_dir (x,z), Hoehe y_eave; steigt mit slope_deg nach innen (links der Traufrichtung von aussen gesehen ...).
    Die Steigrichtung ist senkrecht zur Traufe, zur Region hin."""
    e = unit([eave_dir[0], 0, eave_dir[1]])
    # Steigrichtung (Plan) = Richtung zur Region
    cen = np.array(region_plan.representative_point().coords[0])
    nplan = np.array([-e[2], e[0]])
    if np.dot(cen - np.asarray(eave_pt), nplan) < 0: nplan = -nplan
    s = math.radians(slope_deg)
    up_dir = unit([nplan[0] * math.cos(s), math.sin(s), nplan[1] * math.cos(s)])
    nrm = np.cross(up_dir, e)
    if nrm[1] < 0:
        nrm = -nrm
    # Rechtssystem (a = e, b = up_dir, c = nrm) pruefen
    if np.dot(np.cross(e, up_dir), nrm) < 0:
        e = -e
    fr = Frame([eave_pt[0], y_eave, eave_pt[1]], e, up_dir, nrm)
    # Region in lokale (a, b): Planpunkt -> a = (p-o)·e_plan, b = Abstand in Steigrichtung / cos(s)
    def to_local(xz):
        d = np.array([xz[0] - eave_pt[0], xz[1] - eave_pt[1]])
        a = d[0] * e[0] + d[1] * e[2]
        b = (d @ nplan) / math.cos(s)
        return (a, b)
    def conv(g):
        ext = [to_local(p) for p in np.array(g.exterior.coords)[:-1]]
        ints = [[to_local(p) for p in np.array(h.coords)[:-1]] for h in g.interiors]
        return Polygon(ext, ints)
    geoms = list(region_plan.geoms) if isinstance(region_plan, MultiPolygon) else [region_plan]
    for g in geoms:
        if g.is_empty or g.area < 1e-4: continue
        L = conv(g).buffer(0)
        if L.is_empty: continue
        # Unterlage (schliesst Spalten)
        for gg in (list(L.geoms) if isinstance(L, MultiPolygon) else [L]):
            ext = np.array(gg.exterior.coords)[:-1]
            mb.poly(under, fr, [(p[0], p[1], -0.004) for p in ext], holes=[[(p[0], p[1], -0.004) for p in np.array(h.coords)[:-1]] for h in gg.interiors], hint=(0, 0, 1))
        # Unterseite der Dachflaeche innerhalb der Mauerflucht (dunkel, Normale nach unten): wer durch eine Fuge oder ein
        # Dachbodenfenster nach oben schaut, sieht eine geschlossene Dachschalung statt (FrontSide-gecullt) den Himmel
        inner_reg = g if walls is None else g.intersection(walls)
        for gi in (list(inner_reg.geoms) if hasattr(inner_reg, 'geoms') else [inner_reg]):
            if gi.geom_type != 'Polygon' or gi.area < 1e-4: continue
            Li = conv(gi).buffer(0)
            for gg in (list(Li.geoms) if isinstance(Li, MultiPolygon) else [Li]):
                if gg.is_empty or gg.area < 1e-4: continue
                ext = np.array(gg.exterior.coords)[:-1]
                mb.poly(under, fr, [(p[0], p[1], -0.03) for p in ext],
                        holes=[[(p[0], p[1], -0.03) for p in np.array(h.coords)[:-1]] for h in gg.interiors], hint=(0, 0, -1))
        # Untersicht des Dachueberstands (Teil ausserhalb der Mauerflucht): nach unten gerichtete Schalung
        if walls is not None:
            ov = g.difference(walls)
            for go in (list(ov.geoms) if hasattr(ov, 'geoms') else [ov]):
                if go.geom_type != 'Polygon' or go.area < 1e-4: continue
                Lo = conv(go).buffer(0)
                for gg in (list(Lo.geoms) if isinstance(Lo, MultiPolygon) else [Lo]):
                    if gg.is_empty or gg.area < 1e-4: continue
                    ext = np.array(gg.exterior.coords)[:-1]
                    mb.poly(soffit, fr, [(p[0], p[1], -0.03) for p in ext],
                            holes=[[(p[0], p[1], -0.03) for p in np.array(h.coords)[:-1]] for h in gg.interiors], hint=(0, 0, -1))
        b0, b1 = L.bounds[1], L.bounds[3]
        k0 = int(math.floor(b0 / course)); k1 = int(math.ceil(b1 / course))
        for k in range(k0, k1):
            lo, hi = k * course, (k + 1) * course
            strip = L.intersection(sbox(L.bounds[0] - 1, lo, L.bounds[2] + 1, hi))
            if strip.is_empty: continue
            for sg in (list(strip.geoms) if hasattr(strip, 'geoms') else [strip]):
                if sg.geom_type != 'Polygon' or sg.area < 1e-5: continue
                ext = np.array(sg.exterior.coords)[:-1]
                cfun = lambda b: step * (1 - (b - lo) / course)
                pts = [(p[0], p[1], cfun(p[1])) for p in ext]
                holes = [[(p[0], p[1], cfun(p[1])) for p in np.array(h.coords)[:-1]] for h in sg.interiors]
                mb.poly(mat, fr, pts, holes=holes, hint=(0, 0, 1))
            # Stossflaeche an der Unterkante der Schicht (Hoehe step), Normale -b (zur Traufe)
            ln = L.intersection(LineString([(L.bounds[0] - 1, lo), (L.bounds[2] + 1, lo)]))
            segs = []
            if ln.is_empty: continue
            for gl in (list(ln.geoms) if hasattr(ln, 'geoms') else [ln]):
                if gl.geom_type == 'LineString': segs.append(np.array(gl.coords))
            for sgm in segs:
                xa, xb = sgm[:, 0].min(), sgm[:, 0].max()
                if xb - xa < 1e-3: continue
                mb.quad(mat, fr, (xb, lo, step), (xa, lo, step), (xa, lo, 0.0), (xb, lo, 0.0))
    return fr


# ============================================================================ Rinne / Fallrohr
def gutter(mb, pts, y, out=0.18, r=0.075):
    """halbrunde Hängerinne entlang Planpunkten (aussen bei out), Oberkante y"""
    prof = []
    for i in range(9):
        t = math.pi * i / 8
        prof.append((out + r - r * math.cos(t), y - r * math.sin(t)))
    prof = prof[::-1]
    # Profil: von innen oben ueber Boden nach aussen oben, dann Wulst
    pr = [(out, y)] + [(out + r - r * math.cos(t), y - r * math.sin(t)) for t in np.linspace(0, math.pi, 9)] + [(out + 2 * r + 0.01, y + 0.01)]
    sweep(mb, ZN, pts, pr, closed=False, caps=False)
    # Innenseite (sichtbar von oben): umgekehrtes Profil leicht innen
    pr2 = [(out + 2 * r - 0.004, y)] + [(out + r - (r - 0.004) * math.cos(t), y - (r - 0.004) * math.sin(t)) for t in np.linspace(math.pi, 0, 9)] + [(out + 0.004, y)]
    sweep(mb, ZN, pts, pr2, closed=False, caps=False)


def downpipe(mb, fr, a, y0, y1, c=0.12, r=0.05, mat=ZN):
    """Fallrohr (8-eckig) an Fassadenposition a, Achse bei c vor der Wand"""
    f = fr.local(a, 0, c)
    n = 8
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(ang) * r, np.zeros(n), np.sin(ang) * r], 1)
    G = np.stack([ring + [0, y0, 0], ring + [0, y1, 0]], 1)   # (n, 2, 3)
    mb.grid(mat, f, G, closed_u=True, smooth=True, flip=True)
    # Schellen
    for y in np.arange(y0 + 1.0, y1 - 0.3, 2.0):
        mb.box(FE, f, -0.012, 0.012, y - 0.03, y + 0.03, -c, -r * 0.6, faces='XxYyZ')


def proto_figure():
    """Standfigur (Sandstein) fuer die Zwerchhausnischen: Drehkoerper (Gewandfigur mit Kopf), Hoehe 1, Fuss bei y=0,
    Mitte bei a=c=0; ungleichmaessig skaliert instanziert (Breite a, Tiefe c)."""
    mb = MB()
    fr = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    prof = [(0.17, 0.0), (0.17, 0.05), (0.14, 0.07), (0.150, 0.20), (0.150, 0.45), (0.13, 0.66), (0.15, 0.74), (0.13, 0.79),
            (0.07, 0.82), (0.05, 0.845), (0.075, 0.87), (0.083, 0.92), (0.07, 0.97), (0.035, 0.995), (0.0, 1.0)]
    n = 12
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    G = np.zeros((n, len(prof), 3))
    for i, t in enumerate(ang):
        # Gewand vorn etwas flacher, Ruecken gerade (Nischenfigur)
        k = 0.85 if math.sin(t) < 0 else 1.0
        for j, (r, y) in enumerate(prof):
            G[i, j] = (r * math.cos(t), y, r * math.sin(t) * k)
    mb.grid(SS, fr, G, closed_u=True, smooth=True, flip=True)
    mb.poly(SS, fr, [(0.17 * math.cos(t), 0.0, 0.17 * math.sin(t) * (0.85 if math.sin(t) < 0 else 1.0)) for t in ang], hint=(0, -1, 0))
    return mb

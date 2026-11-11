"""Fassaden: Wandanalyse (Strasse/Hof/Brandwand), Geschossraster, Achsenraster, echte Oeffnungen (Hülle: Laibung,
Innenlaibung, dunkle Rueckwand), instanzierte Fenster-/Tuer-Baugruppen, Baender/Gesimse, Werbebaender, Markisen."""
import math
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box as sbox
from shapely.ops import unary_union
from geom import Frame, wall_frame, sweep, mitre, UP, tri2d, EdgeResolver
from style import RETAIL_STREETS
import protos as PR
from common import jitter_hex

COMMERCIAL = {'2010', '2310', '1120', '2000', '2050', '2052', '2070', '2071', '2081', '1123', '3036'}
FABRIC = ['#F4F1EA', '#EDE6D6', '#E6DCC8', '#D8D4CC', '#C9CDD0', '#B9A88E', '#8E3B35', '#5A6B7A', '#F7F7F2', '#D6CBB5']
WHITE_FRAMES = ('#F2F2EE', '#ECECE6', '#F5F3EC')


class WallInfo:
    pass


def _geoms(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    return [x for x in getattr(g, 'geoms', []) if x.geom_type == 'Polygon' and not x.is_empty]


# ------------------------------------------------------------------ Wandanalyse
def analyse_walls(b, ctx):
    vol = ctx['vol']
    out = []
    for i, s in enumerate(b.walls):
        w = WallInfo(); w.s = s; w.idx = i
        n = s.n.copy()
        w.vertical = abs(n[1]) < 0.05
        if w.vertical:
            n[1] = 0; n /= np.linalg.norm(n)
            P0 = s.P[0]
            fr = wall_frame(n, [P0[0], 0.0, P0[2]])
        else:
            fr = wall_frame(n, s.P[0])
        w.fr = fr; w.n = n
        L = fr.loc(s.P)
        if w.vertical:
            fr = fr.shifted(dd=float(L[:, 2].mean())); w.fr = fr
            L = fr.loc(s.P)
        uv = L[:, :2].copy()
        holes = [fr.loc(h)[:, :2] for h in s.holes]
        P3 = s.P.copy()
        if w.vertical:
            low = uv[:, 1] < b.y_ground + 0.05
            uv[low, 1] = np.minimum(uv[low, 1], b.y_base)
            P3[low, 1] = uv[low, 1]
        w.uv = uv
        try:
            poly = Polygon(uv, holes).buffer(0)
        except Exception:
            poly = Polygon()
        if poly.geom_type != 'Polygon':
            # nach dem Verschweissen selbstberuehrender Ring -> groesste Teilflaeche (Rest sind mm-Zipfel)
            parts = [g for g in getattr(poly, 'geoms', []) if g.geom_type == 'Polygon']
            poly = max(parts, key=lambda g: g.area) if parts else Polygon()
        w.poly = poly
        w.keep3d = EdgeResolver([uv] + holes, [P3] + list(s.holes))
        w.openings = []; w.retail = False; w.street = None; w.street_dist = 99; w.street_hw = None; w.shops = []
        w.party_wall = False; w.bays = None; w.units = []; w.jetty = None
        if not poly.is_empty and w.vertical and ctx.get('wall_tree') is not None:
            poly = _trim_coplanar(w, b, s, poly, ctx)
            w.poly = poly
        if poly.is_empty or poly.area < 1e-3:
            w.drop = True; out.append(w); continue
        w.umin, w.vmin, w.umax, w.vmax = poly.bounds
        w.L = w.umax - w.umin
        tl = uv[np.abs(uv[:, 0] - w.umin) < 0.05, 1]; tr = uv[np.abs(uv[:, 0] - w.umax) < 0.05, 1]
        w.vcorner = float(min(tl.max() if len(tl) else w.vmax, tr.max() if len(tr) else w.vmax))
        w.drop = False
        out.append(w)
    for w in out:
        w.party_u = np.zeros(0); w.party = np.zeros(0, bool)
        if w.drop or not w.vertical:
            continue
        us = np.arange(w.umin + 0.25, w.umax - 0.1, 1.0) if w.L > 0.6 else np.array([(w.umin + w.umax) / 2])
        if len(us) == 0:
            us = np.array([(w.umin + w.umax) / 2])
        y0 = max(w.vmin, b.y_ground) + 0.3
        vs = np.arange(y0, w.vmax - 0.05, 1.5)
        if len(vs) == 0:
            vs = np.array([(max(w.vmin, b.y_ground) + w.vmax) / 2])
        UU, VV = np.meshgrid(us, vs)
        pts = np.c_[UU.ravel(), VV.ravel()]
        from shapely import contains_xy
        ins = contains_xy(w.poly, pts[:, 0], pts[:, 1])
        pts = pts[ins] if ins.any() else pts
        P3 = w.fr.w(np.c_[pts, np.full(len(pts), 0.35)])
        bo, bs = vol.blocked(P3, b.idx)
        w.own_hidden = bool(bs.all() and not bo.any())
        w.blocked_frac = float((bo | bs).mean())
        if w.vmin < b.y_ground + 2.0:
            pu = np.arange(w.umin + 0.25, w.umax, 0.5)
            if len(pu) == 0:
                pu = np.array([(w.umin + w.umax) / 2])
            pv = np.full(len(pu), b.y_ground + 1.3)
            Q = np.concatenate([w.fr.w(np.c_[pu, pv, np.full(len(pu), d)]) for d in (0.05, 0.4, 1.4)])
            bo2, bs2 = vol.blocked(Q, b.idx)
            w.party_u = pu; w.party = bo2.reshape(3, -1).any(0)
            w.party_wall = bool(w.party.mean() > 0.6)
        if w.own_hidden:
            # nur weglassen, wenn auch der obere Rand verdeckt ist (sonst bleibt ein schmaler Streifen ueber
            # einem niedrigeren Bauteil offen): Proben entlang der Oberkante, 5 cm unter dem Rand
            ring = np.asarray(w.poly.exterior.coords)
            seg = []
            for i in range(len(ring) - 1):
                a, c = ring[i], ring[i + 1]
                k = max(1, int(np.linalg.norm(c - a) / 0.5))
                seg.append(a + (c - a) * (np.arange(k) / k)[:, None])
            S = np.concatenate(seg)
            S = S[S[:, 1] > w.vmin + 0.3]
            if len(S):
                ctr = np.array(w.poly.representative_point().coords[0])
                S = S + (ctr - S) / np.maximum(np.linalg.norm(ctr - S, axis=1, keepdims=True), 1e-9) * 0.06
                bo3, bs3 = vol.blocked(w.fr.w(np.c_[S, np.full(len(S), 0.2)]), b.idx)
                if not (bs3 | bo3).all():
                    w.own_hidden = False
        if w.own_hidden:
            w.drop = True
    return out


def _trim_coplanar(w, b, s, poly, ctx):
    """Flaechen abziehen, die von einer anderen, koplanaren und gleich ausgerichteten Wand ueberdeckt werden
    (LoD2-Doppelungen zwischen Bauteilen/Nachbarn/Detailmodell) -> kein Z-Fighting, keine verdeckten Fenster."""
    from shapely.geometry import LineString as _LS
    seg = s.P[:, [0, 2]]
    q = _LS(seg).buffer(0.06)
    WI = ctx['wall_info']
    for j in ctx['wall_tree'].query(q):
        bi, wi, n2, p0, area, covered = WI[j]
        if bi == b.idx and wi == w.idx:
            continue
        if float(np.dot(n2, w.n)) < 0.998:
            continue
        delta = float(np.dot(p0 - s.P[0], w.n))          # >0: andere Wand liegt vor meiner
        if delta < -0.04 or delta > 0.6:
            continue
        mine = s.area
        if delta > 0.04:
            wins = True                                     # meine Wand steckt hinter einer fremden Fassade
        else:
            wins = covered or (bi != b.idx and bi < b.idx) or (bi == b.idx and (area > mine + 1e-6 or (abs(area - mine) < 1e-6 and wi < w.idx)))
        if not wins:
            continue
        P2 = ctx['bldgs'][bi].walls[wi].P
        uv2 = w.fr.loc(P2)[:, :2]
        try:
            o = Polygon(uv2).buffer(0)
        except Exception:
            continue
        if o.is_empty or not o.intersects(poly):
            continue
        ov = o.intersection(poly)
        if ov.area < 0.05 or 2 * ov.area / max(ov.length, 1e-9) < 0.04:
            continue                                        # nur ein Streifen von wenigen mm: nicht ausschneiden (sonst Spalt)
        poly = poly.difference(o)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda g: g.area) if poly.area > 0 else Polygon()
        if poly.geom_type != 'Polygon':
            poly = Polygon()
        if poly.is_empty:
            break
    return poly


def is_party(w, u):
    u = np.asarray(u, float)
    if len(w.party_u) == 0:
        return np.zeros(np.shape(u), bool)
    k = np.clip(np.searchsorted(w.party_u, u), 0, len(w.party_u) - 1)
    k2 = np.clip(k - 1, 0, len(w.party_u) - 1)
    near = np.where(np.abs(w.party_u[k] - u) < np.abs(w.party_u[k2] - u), k, k2)
    return w.party[near]


def open_at(w, b, vol, u, v, deltas=(0.05, 0.45, 1.4), behind=True):
    """True, wenn vor der Wand bei (u,v) kein Gebaeudevolumen liegt und die Wand dort nicht in einem fremden
    Gebaeude steckt (Probe 3 cm hinter der Wandflaeche, nur fremde Volumen)."""
    u = np.atleast_1d(np.asarray(u, float)); v = np.broadcast_to(np.asarray(v, float), u.shape)
    Q = np.concatenate([w.fr.w(np.c_[u, v, np.full(len(u), d)]) for d in deltas])
    bo, bs = vol.blocked(Q, b.idx)
    ok = ~((bo | bs).reshape(len(deltas), -1).any(0))
    if behind:
        bo2, _ = vol.blocked(w.fr.w(np.c_[u, v, np.full(len(u), -0.03)]), b.idx)
        ok &= ~bo2
    return ok


# ------------------------------------------------------------------ Strassenniveau
def street_reference(b, walls, ctx):
    """Gehwegniveau vor jeder Erdgeschosswand (w.yl, Median aus dem Hoehenfeld) und Bezugsniveau des Gebaeudes:
    laengengewichteter Median ueber die freien Aussenwaende (Strassenwaende mit Laeden zaehlen mehr).
    Nie tiefer als der Wandfuss (y_base), nie so hoch, dass weniger als 4 m Fassade bleiben."""
    import level as LV
    vol = ctx['vol']
    cand = []
    for w in walls:
        w.yl = None
        if w.drop or not w.vertical or w.L < 1.0 or w.vmin > b.y_ground + 1.0:
            continue
        w.yl = LV.front(w, w.umin + 0.2, w.umax - 0.2, vol=vol, own=b.idx)
        if w.yl is None or w.party_wall or w.L < 2.0:
            continue
        wt = w.L * (3.0 if w.shops else 1.0) * (2.0 if w.street_dist < 26 else 1.0)
        cand.append((w.yl, wt))
    if not cand:
        return b.y_ground
    ys, wt = np.array(cand, float).T
    o = np.argsort(ys); cw = np.cumsum(wt[o])
    yr = float(ys[o][min(len(o) - 1, int(np.searchsorted(cw, cw[-1] / 2)))]) - 0.02
    if abs(yr - b.y_ground) < 0.12:
        return b.y_ground                                  # kaum Unterschied: LoD2-Niveau behalten
    yr = max(yr, getattr(b, 'y_base', b.y_ground - 0.3) + 0.1)
    yr = min(yr, max(b.y_ground, b.y_top - 4.0))
    return float(yr)


def wall_params(w, P):
    """Wandweise Parameter (style.OVERRIDES 'wall_P': {wand_idx: {...}}) ueber die Gebaeudeparameter legen."""
    import collections
    ov = (P.get('wall_P') or {}).get(w.idx)
    w.P = collections.ChainMap(dict(ov), P) if ov else P


def local_level(w, b, ctx, u0, u1, lo=None, hi=None):
    """Gehwegniveau vor der Wand zwischen u0 und u1 (Hoehenfeld), begrenzt auf den Wandfuss und [lo, hi]."""
    import level as LV
    y = LV.front(w, u0, u1, vol=ctx['vol'], own=b.idx)
    if y is None:
        y = w.yl if getattr(w, 'yl', None) is not None else b.y_ground
    y -= 0.02
    y = max(y, w.vmin + 0.05)
    if lo is not None:
        y = max(y, lo)
    if hi is not None:
        y = min(y, hi)
    return float(y)


# ------------------------------------------------------------------ Strassenbezug / Laeden
def street_info(w, b, ctx):
    if w.drop or not w.vertical or w.L < 2.5:
        return
    sidx = ctx['street_tree']; streets = ctx['streets']
    best = None
    for uf in (0.25, 0.5, 0.75):
        u = w.umin + w.L * uf
        p = w.fr.w([u, b.y_ground + 1.0, 0.2]); q = w.fr.w([u, b.y_ground + 1.0, 32.0])
        ray = LineString([(p[0], p[2]), (q[0], q[2])])
        for j in sidx.query(ray):
            g = ctx['street_geoms'][j]
            if not g.intersects(ray):
                continue
            dd = Point(p[0], p[2]).distance(g.intersection(ray))
            st = streets[j]
            if st['name'] is None and st['hw'] in ('service', 'footway'):
                continue
            if best is None or dd < best[0]:
                best = (dd, st['name'], st['hw'])
    if best:
        w.street_dist, w.street, w.street_hw = best
    c0 = w.fr.w([w.umin, b.y_ground, 0]); c1 = w.fr.w([w.umax, b.y_ground, 0])
    seg = LineString([(c0[0], c0[2]), (c1[0], c1[2])])
    for j in ctx['shop_tree'].query(seg.buffer(14)):
        sp = ctx['shops'][j]['p']
        L = w.fr.loc([sp[0], b.y_ground, sp[1]])
        if w.umin - 1 < L[0] < w.umax + 1 and -6 < L[2] < 14:
            w.shops.append(ctx['shops'][j])


def decide_retail(w, b, kind, fcode):
    if w.drop or not w.vertical or w.vmin > b.y_ground + 1.0:
        return False
    if getattr(w, 'units', None) and w.L >= 2.2:
        return True                                    # bekannter Laden (shops.json) auch auf kurzem Frontsegment
    if w.L < 3.0:
        return False
    if w.party_wall:
        return False
    if kind in ('gothic', 'parking', 'canopy', 'shed', 'fachwerk') and not w.shops:
        return False
    if kind in ('gothic', 'canopy', 'tower'):
        return False
    st = w.street if w.street_dist < 26 else None
    if w.shops:
        return True
    if st in RETAIL_STREETS and (fcode in COMMERCIAL or fcode is None or kind in ('kaufhaus', 'nachkrieg', 'modern', 'curtain', 'altbau')):
        return True
    if w.street_hw == 'pedestrian' and w.street_dist < 20 and (fcode in COMMERCIAL or kind == 'kaufhaus'):
        return True
    if kind == 'kaufhaus' and w.street_dist < 20:
        return True
    return False


# ------------------------------------------------------------------ Geschossraster
def storey_grid(b, P, walls, pitched):
    vw = [(w.vcorner, w.L) for w in walls if not w.drop and w.vertical and w.L > 2 and w.vmin < b.y_ground + 1.0]
    if not vw:
        vw = [(w.vcorner, max(w.L, 0.1)) for w in walls if not w.drop and w.vertical] or [(b.y_top, 1.0)]
    vw.sort()
    tot = sum(l for _, l in vw); acc = 0; He = vw[-1][0]
    for v, l in vw:
        acc += l
        if acc >= 0.6 * tot:
            He = v; break
    He = He - b.y_ground
    extra = 0.6 if pitched else max(0.8, P.get('parapet_h', 0.6) + 0.3)
    g = P['gf']; s0 = P['storey']
    gmin = P.get('gf_min', 4.2 if P['kind'] == 'kaufhaus' else 3.2)
    if He < g + 1.8:
        return {'He': He, 'g': max(2.6, He - extra * 0.5), 's': s0, 'n': 1}
    cand = []
    a = b.attrs
    if a.get('osm_levels'):
        cand.append(int(round(a['osm_levels'])))
    if b.storeys:
        cand.append(int(b.storeys))
    n = None
    for c in cand:
        if c >= 2:
            s = (He - g - extra) / (c - 1)
            if 2.75 <= s <= 5.2:
                n = c; break
            g2 = min(max(gmin, He - extra - (c - 1) * s0), 5.6)
            s = (He - g2 - extra) / (c - 1)
            if 2.75 <= s <= 5.2:
                n = c; g = g2; break
    if n is None:
        n = max(2, int(round((He - g - extra) / s0)) + 1)
    s = (He - g - extra) / max(1, n - 1)
    s = float(np.clip(s, 2.7, 5.4))
    return {'He': He, 'g': g, 's': s, 'n': n}


def floor_y(b, G, k):
    return b.y_ground + (0.0 if k == 0 else G['g'] + (k - 1) * G['s'])


# ------------------------------------------------------------------ Achsen / Oeffnungen
def bays(w, P, axis=None, margin=None, win_w=None):
    axis = axis or P['axis']; m = P['margin'] if margin is None else margin
    usable = w.L - 2 * m
    ww0 = win_w if win_w is not None else P['win_w']
    if usable < max(ww0, 0.6):
        if w.L > ww0 + 0.9 and w.L > 1.8:
            return [((w.umin + w.umax) / 2, min(ww0, w.L - 0.9))]
        return []
    gp = P.get('win_gap', 0.55) if win_w is None else 0.55
    nb = max(1, int(math.floor(usable / axis + 0.5)))
    a = usable / nb
    ww = min(ww0, a - gp)
    while ww < 0.55 and nb > 1:
        nb -= 1; a = usable / nb; ww = min(ww0, a - gp)
    if ww < 0.5:
        return []
    return [(w.umin + m + a * (i + 0.5), ww) for i in range(nb)]


def rect_fits(w, u0, u1, v0, v1, mside=0.3, mtop=0.4, mbot=0.15):
    return w.poly.contains(sbox(u0 - mside, v0 - mbot, u1 + mside, v1 + mtop))


class Opening:
    def __init__(self, u0, u1, v0, v1, r, ri=0.35, zone='up', kind='win', poly=None, back='#1E1D1C', inner='#A39C92'):
        self.u0, self.u1, self.v0, self.v1 = u0, u1, v0, v1
        self.r = r; self.ri = ri; self.zone = zone; self.kind = kind; self.poly = poly
        self.back = back; self.inner = inner
        self.insts = []          # (key, matmap, origin, uc, vb, sx, sy, color)

    def add(self, key, origin, uc, vb, w, h, color, matmap=None, unit=False, sy=None):
        p = PR.PROTOS.get(key)
        if p is None:
            from inst import get_proto
            p = get_proto(key)
        sx = w if unit else w / p.w0
        syy = (h if unit else h / p.h0) if sy is None else sy
        self.insts.append((key, matmap, origin, uc, vb, sx, syy, color))


def _back(rng):
    return ['#161514', '#1C1A18', '#201E1B', '#141618', '#24211D', '#1A1917'][int(rng.integers(6))]


def window_assembly(op, P, frame_kind, rng, interior_mix, surround=None, sill='stone', rollo=0.0, balcony=False):
    uc = (op.u0 + op.u1) / 2; w = op.u1 - op.u0; h = op.v1 - op.v0
    op.add(frame_kind, 'frame', uc, op.v0, w, h, P['frame_col'])
    op.add('glass', 'frame', uc, op.v0, w, h, jitter_hex('#C4CED4', rng, 0.05), unit=True)
    it = _choose(interior_mix, rng)
    if it:
        fc = FABRIC[int(rng.integers(len(FABRIC)))] if it == 'curtains' else ['#F2F0EA', '#E8E4DA', '#D8D8D4', '#C8C2B4'][int(rng.integers(4))]
        op.add('int_' + it, 'frame', uc, op.v0, w, h, fc, unit=True)
    if sill:
        op.add('sill_' + sill, 'wall', uc, op.v0, w, 1.0, P.get('sill_col', '#B8B0A2'), sy=1.0)
    if surround:
        tm = P['trim_mat'] if P['trim_mat'] in ('plaster', 'stone', 'panel', 'brick') else 'plaster'
        op.add('sur_' + surround, 'wall', uc, op.v0, w, h, P['trim_col'], matmap=(('surround', tm),))
    if rollo > 0:
        f = 25 if rollo < 0.35 else (50 if rollo < 0.65 else 80)
        op.add('rollo_%d' % f, 'frame', uc, op.v0, w, h, ['#D8D8D4', '#B4B8BA', '#E8E4DA', '#8C969C'][int(rng.integers(4))])
    if balcony:
        op.add('fr_bal', 'wall', uc, op.v0, w, 1.0, '#2F3337', sy=1.0)


def _choose(mix, rng):
    r = rng.random(); acc = 0
    for k, p in mix:
        acc += p
        if r < acc:
            return k
    return None


INT_MIX = {
    'altbau': [('curtains', 0.38), ('net', 0.25), ('roller', 0.07), ('blind', 0.05)],
    'nachkrieg': [('net', 0.3), ('curtains', 0.28), ('blind', 0.1), ('roller', 0.08)],
    'fachwerk': [('curtains', 0.45), ('net', 0.3)],
    'modern': [('blind', 0.35), ('blind_low', 0.15), ('roller', 0.1)],
    'curtain': [('blind', 0.25), ('blind_low', 0.12), ('roller', 0.08)],
    'kaufhaus': [('blind', 0.2)],
    'shed': [('blind', 0.2)],
    'parking': [], 'gothic': [], 'tower': []}


def layout_wall(w, b, P, G, ctx, rng, flags):
    P = getattr(w, 'P', P)
    if P['kind'] == 'fachwerk' and not w.drop and w.vertical and not w.party_wall and w.L >= 0.3 and G['n'] >= 2:
        # auch kurze Fasen-/Eckwaende kragen mit aus (sonst bleibt an der Ecke ein zurueckspringender, unverzimmerter Putzstreifen)
        w.jetty = jetty_bands(w, b, P, G)
    _layout_wall(w, b, P, G, ctx, rng, flags)
    if getattr(w, 'yl', None) is not None and w.yl > b.y_ground + 0.4 and w.openings:
        # hangseitig hoeherer Gehweg: Obergeschossfenster, die ins Gelaende reichen wuerden, entfallen
        keep = []
        for o in w.openings:
            if o.zone != 'gf' and o.v0 < w.yl + 0.6 and o.v0 < local_level(w, b, ctx, o.u0, o.u1) + 0.45:
                continue
            keep.append(o)
        w.openings = keep
    if w.jetty:
        keep = []
        for o in w.openings:
            if any(o.v0 - 0.05 < y < o.v1 + 0.05 for (y, _, _) in w.jetty[1:]):
                continue                                          # Oeffnung ueber einer Auskragungslinie: weglassen
            o.dz = jetty_dz(w, (o.v0 + o.v1) / 2)
            keep.append(o)
        w.openings = keep


def jetty_bands(w, b, P, G):
    """Fachwerk-Auskragung: Baender [(y0, y1, dz)] (Erdgeschoss dz=0, jedes Obergeschoss + Giebel weiter vor)."""
    jd = P.get('jetty', 0.18)
    ys = [floor_y(b, G, k) for k in range(1, G['n'])]
    He = b.y_ground + G['He']
    gable = w.vmax - w.vcorner > 1.2
    if gable and w.vcorner - 0.3 > (ys[-1] if ys else b.y_ground) + 1.5 and w.vmax - w.vcorner > 1.6:
        ys.append(w.vcorner - 0.25)                            # Giebelschwelle
    if not ys:
        return None
    cap = 0.62 if gable else max(0.0, P.get('overhang', 0.5) - 0.16)
    out = [(-1e6, ys[0], 0.0)]
    for k, y in enumerate(ys):
        y1 = ys[k + 1] if k + 1 < len(ys) else 1e6
        out.append((y, y1, round(min(cap, jd * (k + 1)), 3)))
    if all(dz <= 0.0 for _, _, dz in out):
        return None
    return out


def jetty_dz(w, v):
    if not w.jetty:
        return 0.0
    for y0, y1, dz in w.jetty:
        if y0 <= v < y1:
            return dz
    return 0.0


def _layout_wall(w, b, P, G, ctx, rng, flags):
    kind = P['kind']; vol = ctx['vol']
    if w.idx in (P.get('glazed_walls') or ()) and not w.drop and w.vertical and not w.party_wall:
        w.glazed = True                                  # auch kurze Segmente (gerundeter Glasturm) als Glas
    if w.drop or not w.vertical or w.L < 1.2 or w.party_wall:
        return
    if w.idx in (P.get('glazed_walls') or ()) and kind != 'curtain':
        # verglaster Eckturm/Treppenhaus (Override): Obergeschosse als Pfosten-Riegel-Fassade, Wand dahinter Bruestungsglas
        import collections
        w.glazed = True
        P = collections.ChainMap({'kind': 'curtain', 'axis': 1.35, 'frame_col': '#8C969C', 'glass': '#2A3A44'}, P)
        kind = 'curtain'
    if kind == 'tower':
        _tower_slits(w, b, P, G, ctx, rng, flags)
        return
    n = G['n']
    imix = INT_MIX.get(kind, [])
    r = P['reveal']
    # ---------------- Obergeschosse
    if kind == 'curtain':
        BB = bays(w, P, axis=P['axis'], margin=0.3, win_w=P['axis'])
        if not BB:
            return
        a = (w.L - 0.6) / len(BB)
        for k in range(1, 60):
            fy = floor_y(b, G, k)
            if fy + G['s'] > w.vmax + 0.01 - 0.4:
                break
            v0 = fy + 0.45; v1 = fy + G['s'] - 0.45
            ok = [bool(open_at(w, b, vol, uc, (v0 + v1) / 2)[0]) and not is_party(w, uc) for uc, _ in BB]
            for i0, i1 in _runs(ok):
                u0 = BB[i0][0] - a / 2 + 0.03; u1 = BB[i1][0] + a / 2 - 0.03
                if not rect_fits(w, u0, u1, v0, v1, 0.2, 0.25, 0.2):
                    continue
                op = Opening(u0, u1, v0, v1, 0.06)
                _modules(op, P, 'cw_1', i1 - i0 + 1, rng, imix)
                w.openings.append(op)
        flags.setdefault('fins', []).append((w, BB, a))
    elif kind in ('altbau', 'nachkrieg', 'fachwerk', 'shed', 'modern', 'kaufhaus', 'parking'):
        if kind == 'modern' and P.get('ribbon'):
            BB = bays(w, P, axis=P['axis'], margin=0.6, win_w=P['axis'] - 0.12)
        elif kind == 'kaufhaus' and P['upper'] == 'bands':
            BB = bays(w, P, axis=1.5, margin=1.2, win_w=1.4)
        elif kind == 'parking':
            BB = bays(w, P, axis=P['axis'], margin=0.5, win_w=P['axis'] - 0.5)
        else:
            BB = bays(w, P)
        w.bays = BB
        for k in range(1, 80):
            fy = floor_y(b, G, k)
            if fy > w.vmax - 0.8:
                break
            above_main = k >= n
            if kind in ('shed',) or (kind == 'kaufhaus' and P['upper'] == 'closed'):
                break
            sill_h = P['sill']; wh = P['win_h']; head_clear = 0.45
            if kind == 'altbau':
                wh = P['win_h'] + (0.1 if k == 1 else 0) - 0.07 * max(0, k - 2)
                head_clear = 1.1 if (k == 1 and P['head'] == 'pediment') else (0.8 if k <= 2 else 0.5)
            if kind == 'kaufhaus' and P['upper'] == 'slits':
                wh = G['s'] - 1.0; sill_h = 0.5
            if kind == 'kaufhaus' and P['upper'] == 'bands':
                wh = 0.95; sill_h = 1.3
            if kind == 'parking':
                sill_h = 1.05; wh = G['s'] - 1.05 - 0.35
            if above_main:
                sill_h = 0.75; wh = min(1.25, wh); head_clear = 0.5
            wh = min(wh, G['s'] - sill_h - head_clear) if not above_main else wh
            if wh < 0.6:
                continue
            v0 = fy + sill_h; v1 = v0 + wh
            if kind == 'modern' and P.get('ribbon') and not above_main and len(BB) > 0:
                ok = [bool(open_at(w, b, vol, uc, (v0 + v1) / 2)[0]) and not is_party(w, uc) for uc, _ in BB]
                a = (BB[1][0] - BB[0][0]) if len(BB) > 1 else P['axis']
                for i0, i1 in _runs(ok):
                    u0 = BB[i0][0] - a / 2 + 0.06; u1 = BB[i1][0] + a / 2 - 0.06
                    if not rect_fits(w, u0, u1, v0, v1, 0.25, 0.35):
                        continue
                    op = Opening(u0, u1, v0, v1, r)
                    _modules(op, P, 'rib_1', i1 - i0 + 1, rng, imix)
                    op.add('sill_metal', 'wall', (u0 + u1) / 2, v0, u1 - u0, 1.0, P.get('sill_col', '#8C969C'), sy=1.0)
                    w.openings.append(op)
                continue
            for uc, ww in BB:
                if kind == 'kaufhaus' and P['upper'] == 'slits':
                    ww = 0.55
                if kind == 'kaufhaus' and P['upper'] == 'grid':
                    ww = min(ww, 1.0)
                bal = False
                vv0, vv1 = v0, v1
                if kind == 'altbau' and (not above_main) and k in (1, 2) and rng.random() < P['balcony_p']:
                    bal = True; vv0 = fy + 0.08
                if kind == 'nachkrieg' and (not above_main) and rng.random() < P['balcony_p']:
                    bal = True; vv0 = fy + 0.08
                u0, u1 = uc - ww / 2, uc + ww / 2
                if not rect_fits(w, u0, u1, vv0, vv1, 0.32, head_clear if not above_main else 0.5):
                    continue
                if is_party(w, uc) or not open_at(w, b, vol, uc, (vv0 + vv1) / 2)[0] or not open_at(w, b, vol, [u0 + 0.1, u1 - 0.1], vv1 - 0.1).all():
                    continue
                if kind == 'parking':
                    ri = _park_depth(w, b, vol, u0, u1, v0, v1, r, flags)
                    if ri is None:
                        continue
                    op = Opening(u0, u1, v0, v1, r, ri=ri, back='#141414', inner='#5A5856')
                    op.add('park', 'frame', uc, v0, ww, wh, '#3A3D40')
                    w.openings.append(op); continue
                op = Opening(u0, u1, vv0, vv1, r, back=_back(rng))
                if kind == 'altbau':
                    fk = ['alt_T', 'alt_K'][int(P['alt_variant'])] + ('_spr' if P['sprossen'] else '')
                    if above_main:
                        fk = 'nk_2'
                    sur = None
                    if not above_main:
                        if k == 1:
                            sur = {'cornice': 'fasche_cornice', 'pediment': 'fasche_pediment'}.get(P['head'], 'fasche_key')
                        elif k == 2:
                            sur = 'fasche_key'
                        else:
                            sur = 'fasche'
                    window_assembly(op, P, fk, rng, imix, surround=sur, sill='stone', balcony=bal)
                elif kind == 'fachwerk':
                    window_assembly(op, P, 'fw_S', rng, imix, None, None)
                else:
                    if kind == 'nachkrieg':
                        fk = 'nk_3' if ww > 1.45 else 'nk_2'
                        if above_main:
                            fk = 'nk_2'
                        ro = float(rng.uniform(0.1, 0.9)) if rng.random() < P['blinds'] else 0.0
                        window_assembly(op, P, fk, rng, imix, surround=P.get('surround') if not above_main else None,
                                        sill='metal' if P.get('sill_col') == '#8C969C' else 'stone', rollo=ro, balcony=bal)
                    elif kind == 'kaufhaus':
                        window_assembly(op, P, 'mod_1', rng, imix, None, 'metal')
                    else:
                        window_assembly(op, P, 'mod_2' if ww > 1.3 else 'mod_1', rng, imix, None, 'metal')
                w.openings.append(op)
    elif kind == 'gothic':
        BB = bays(w, P, axis=P['axis'], margin=1.8, win_w=P['win_w'])
        tall = w.vmax - b.y_ground
        rows = []
        if w.L < 14 and tall > 30:
            y = b.y_ground + 8.0
            while y + 5.5 < w.vcorner - 2:
                rows.append((y, 4.5)); y += 11.0
        else:
            hh = min(w.vcorner - b.y_ground - 5.5, 12.0)
            if hh > 3:
                rows.append((b.y_ground + 3.2, hh))
        for y0, hh in rows:
            for uc, ww in BB:
                ww = round(min(ww, 2.4 if hh > 6 else 1.3), 1); hh = round(hh, 1)
                if not rect_fits(w, uc - ww / 2, uc + ww / 2, y0, y0 + hh, 0.4, 0.6):
                    continue
                if is_party(w, uc) or not open_at(w, b, vol, uc, y0 + hh / 2)[0]:
                    continue
                poly = PR.lancet_poly(ww, hh) + [uc, y0]
                op = Opening(uc - ww / 2, uc + ww / 2, y0, y0 + hh, P['reveal'], ri=0.3, kind='lancet', poly=poly, back='#101012')
                op.add('lancet_%d_%d' % (int(ww * 100), int(hh * 100)), 'frame', uc, y0, ww, hh, '#4A4E52')
                w.openings.append(op)
    # ---------------- Erdgeschoss
    if w.vmin > b.y_ground + 1.0:
        return
    fy1 = floor_y(b, G, 1) if G['n'] > 1 else b.y_ground + G['g']
    if w.retail:
        import shops as SH
        SH.layout_retail(w, b, P, G, ctx, rng, flags)
    else:
        if kind in ('gothic', 'canopy'):
            return
        BB = w.bays if w.bays is not None else bays(w, P)
        if kind == 'curtain':
            BB = bays(w, P, axis=P['axis'], margin=0.3, win_w=P['axis'])
            a = (w.L - 0.6) / max(1, len(BB))
            v0 = b.y_ground + 0.3; v1 = fy1 - 0.45
            ok = [bool(open_at(w, b, vol, uc, b.y_ground + 1.5)[0]) and not is_party(w, uc) for uc, _ in BB]
            for i0, i1 in _runs(ok):
                u0 = BB[i0][0] - a / 2 + 0.03; u1 = BB[i1][0] + a / 2 - 0.03
                if rect_fits(w, u0, u1, v0, v1, 0.2, 0.25, 0.1):
                    op = Opening(u0, u1, v0, v1, 0.08, zone='gf')
                    _modules(op, P, 'cw_1', i1 - i0 + 1, rng, imix)
                    w.openings.append(op)
            return
        door_bay = None
        if flags.get('house_door_done') is None and w.street_dist < 22 and BB and w.L > 4 and kind not in ('parking', 'tower'):
            door_bay = len(BB) // 2
        sill_gf = max(1.15, P['sill'] + 0.3)
        for i, (uc, ww) in enumerate(BB):
            if is_party(w, uc) or not open_at(w, b, vol, uc, b.y_ground + 1.6)[0]:
                continue
            if i == door_bay:
                dk = 'alt2' if kind == 'altbau' else ('nk' if kind in ('nachkrieg', 'fachwerk', 'shed') else 'glass')
                dp = PR.door_proto(dk, 0.3 if dk == 'alt2' else 0.2)
                dw, dh = dp.w0, dp.h0
                yd = local_level(w, b, ctx, uc - dw / 2 - 0.3, uc + dw / 2 + 0.3, hi=fy1 - 2.4)
                dh = min(dh, fy1 - yd - 0.5)
                lift = 0.32 if kind in ('altbau', 'nachkrieg', 'fachwerk') else 0.0
                if lift and not rect_fits(w, uc - dw / 2, uc + dw / 2, yd + lift, yd + lift + dh, 0.45, 0.3, 0.0):
                    lift = 0.0
                if dh < 1.95 or not rect_fits(w, uc - dw / 2, uc + dw / 2, yd + lift, yd + lift + dh, 0.45, 0.3, 0.0):
                    continue
                rr = 0.3 if dk == 'alt2' else 0.2
                op = Opening(uc - dw / 2, uc + dw / 2, yd + lift, yd + lift + dh, rr, ri=0.12, zone='gf', kind='door')
                dcol = ['#5A3B28', '#4B3528', '#294738', '#682F32', '#2F3337', '#3D2B22'][int(rng.integers(6))]
                if dk == 'glass':
                    dcol = P['frame_col'] if P['frame_col'] not in WHITE_FRAMES else '#3A3D40'
                    op.add('glass', 'frame', uc, op.v0, dw, dh, '#C4CED4', unit=True)
                op.add('door_%s_%d' % (dk, int(rr * 100)), 'frame', uc, op.v0, dw, dh, dcol)
                if kind == 'altbau':
                    tm = P['trim_mat'] if P['trim_mat'] in ('plaster', 'stone') else 'stone'
                    op.add('sur_fasche_cornice', 'wall', uc, op.v0, dw, dh, P['trim_col'], matmap=(('surround', tm),))
                op.steps = 2 if lift else 0
                w.openings.append(op)
                flags['house_door_done'] = True
                flags.setdefault('steps', []).append((w, uc, dw, lift, op.v0))
                continue
            if kind in ('kaufhaus', 'tower'):
                continue
            yw = max(b.y_ground, local_level(w, b, ctx, uc - ww / 2, uc + ww / 2))     # hangseitig hoeher: Fenster mit anheben
            wh = min(P['win_h'], fy1 - yw - sill_gf - 0.5)
            if kind == 'parking':
                wh = fy1 - yw - 1.05 - 0.4; sill_gf = 1.05
            if kind == 'shed':
                wh = min(1.2, w.vcorner - yw - sill_gf - 0.4)
            if wh < 0.6:
                continue
            v0 = yw + sill_gf; v1 = v0 + wh
            if not rect_fits(w, uc - ww / 2, uc + ww / 2, v0, v1, 0.32, 0.4):
                continue
            if kind == 'parking':
                ri = _park_depth(w, b, vol, uc - ww / 2, uc + ww / 2, v0, v1, r, flags)
                if ri is None:
                    continue
                op = Opening(uc - ww / 2, uc + ww / 2, v0, v1, r, ri=ri, zone='gf', back='#141414', inner='#5A5856')
                op.add('park', 'frame', uc, v0, ww, wh, '#3A3D40')
                w.openings.append(op); continue
            op = Opening(uc - ww / 2, uc + ww / 2, v0, v1, r, zone='gf', back=_back(rng))
            if kind == 'altbau':
                fk = ['alt_T', 'alt_K'][int(P['alt_variant'])] + ('_spr' if P['sprossen'] else '')
                window_assembly(op, P, fk, rng, imix, surround=None if P.get('rustic') else 'fasche', sill='stone')
            elif kind == 'fachwerk':
                window_assembly(op, P, 'fw_S', rng, imix, None, 'stone')
            elif kind == 'nachkrieg':
                window_assembly(op, P, 'nk_2', rng, imix, None, 'metal' if P.get('sill_col') == '#8C969C' else 'stone',
                                rollo=float(rng.uniform(0.1, 0.9)) if rng.random() < P['blinds'] else 0.0)
            else:
                window_assembly(op, P, 'mod_1', rng, imix, None, 'metal')
            w.openings.append(op)


def _park_depth(w, b, vol, u0, u1, v0, v1, r, flags, dmax=5.0):
    """Tiefe der offenen Parkdeck-Oeffnung: nur so weit das eigene Volumen reicht und ohne die Tunnel anderer Waende
    (gleiche Hoehe) zu schneiden - sonst koplanare Laibungen (Z-Fighting) und Laibungen, die durch die Nachbarfassade stossen."""
    us = np.linspace(u0 + 0.1, u1 - 0.1, 3)
    D = 0.0
    for k in np.arange(0.4, dmax + 0.01, 0.3):
        Q = w.fr.w(np.c_[us, np.full(len(us), (v0 + v1) / 2), np.full(len(us), -r - k)])
        bo, bs = vol.blocked(Q, b.idx)
        if bo.any() or not bs.all():
            break
        D = k
    D -= 0.35
    if D < 0.6:
        return None
    tun = flags.setdefault('park_tunnels', [])

    def rect(Dd):
        c = [w.fr.w([u, 0, d]) for u, d in ((u0 - 0.05, 0.0), (u1 + 0.05, 0.0), (u1 + 0.05, -r - Dd), (u0 - 0.05, -r - Dd))]
        return Polygon([(q[0], q[2]) for q in c])
    pg = rect(D)
    while D > 0.6 and any(t[1] < v1 + 0.05 and t[2] > v0 - 0.05 and pg.intersects(t[0]) for t in tun):
        D -= 0.3; pg = rect(D)
    if D < 0.6:
        return None
    tun.append((pg, v0, v1))
    return float(D)


def _tower_slits(w, b, P, G, ctx, rng, flags):
    """Wehrturm: wenige schmale Schlitzfenster (tiefe Laibung, dunkel), verteilt ueber die Hoehe, nie im Erdgeschoss."""
    from common import rng_of
    if w.L < 1.0:
        return
    rr = rng_of('slit', b.id, w.idx)
    done = flags.setdefault('slits', 0)
    if done >= 5 or rr.random() > 0.3:
        return
    H = w.vcorner - b.y_ground
    for f in (0.32, 0.55, 0.8):
        if flags['slits'] >= 5:
            break
        if rr.random() < 0.55:
            continue
        uc = (w.umin + w.umax) / 2 + float(rr.uniform(-0.2, 0.2)) * max(0.0, w.L - 0.8)
        v0 = b.y_ground + H * f; v1 = v0 + 0.95
        if not rect_fits(w, uc - 0.12, uc + 0.12, v0, v1, 0.25, 0.4) or not open_at(w, b, ctx['vol'], uc, (v0 + v1) / 2)[0]:
            continue
        poly = PR.lancet_poly(0.24, 0.95) + [uc, v0]
        op = Opening(uc - 0.12, uc + 0.12, v0, v1, 0.45, ri=0.25, kind='lancet', poly=poly, back='#121212', inner='#8E8676')
        w.openings.append(op)
        flags['slits'] += 1


def _modules(op, P, key, nm, rng, imix):
    """Band-/Pfosten-Riegel-Fenster: je Feld eine Modul-Instanz (Rahmen, Glas, Innenansicht)."""
    a = (op.u1 - op.u0) / nm; h = op.v1 - op.v0
    for i in range(nm):
        uc = op.u0 + a * (i + 0.5)
        op.add(key, 'frame', uc, op.v0, a, h, P['frame_col'])
        op.add('glass', 'frame', uc, op.v0, a, h, jitter_hex('#C4CED4', rng, 0.05), unit=True)
        it = _choose(imix, rng)
        if it:
            op.add('int_' + it, 'frame', uc, op.v0, a, h, ['#F2F0EA', '#E8E4DA', '#D8D8D4', '#C8C2B4'][int(rng.integers(4))], unit=True)


def _runs(ok):
    runs = []; i = 0; n = len(ok)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    return runs


# ------------------------------------------------------------------ Wand + Oeffnungen ausgeben
def shell_opening(acc, fr, op, wall_mat, wall_col):
    """Laibung (Wandmaterial bis Rahmenebene), Innenlaibung (hell) und dunkle Rueckwand: schliesst das Loch."""
    r, ri = op.r, op.ri
    if op.poly is not None:
        P2 = np.asarray(op.poly, float)
    else:
        P2 = np.array([[op.u0, op.v0], [op.u1, op.v0], [op.u1, op.v1], [op.u0, op.v1]])
    c2 = P2.mean(0); n = len(P2)
    Vo = []; Fo = []; Wo = []; Vi = []; Fi = []; Wi = []
    for i in range(n):
        a, bq = P2[i], P2[(i + 1) % n]
        e = bq - a; nrm = np.array([-e[1], e[0]])
        if np.dot(nrm, c2 - (a + bq) / 2) < 0:
            nrm = -nrm
        want = nrm[0] * fr.t + nrm[1] * fr.b
        for (d0, d1, V, F, Wl) in ((0.0, -r, Vo, Fo, Wo), (-r, -r - ri, Vi, Fi, Wi))[:1 if ri <= 0 else 2]:
            k = len(V)
            V += [fr.w([a[0], a[1], d0]), fr.w([bq[0], bq[1], d0]), fr.w([bq[0], bq[1], d1]), fr.w([a[0], a[1], d1])]
            F += [[k, k + 1, k + 2], [k, k + 2, k + 3]]; Wl += [want, want]
    _oriented(acc, 'shell', wall_mat, Vo, Fo, Wo, wall_col)
    if ri <= 0:                                  # Ladenraum dahinter schliesst die Oeffnung (shops.emit_rooms)
        return
    _oriented(acc, 'shell', 'interior', Vi, Fi, Wi, op.inner)
    V2, F2 = tri2d(P2)
    Vb = fr.w(np.c_[V2, np.full(len(V2), -r - ri)])
    _oriented(acc, 'shell', 'interior', Vb, F2, [fr.n] * len(F2), op.back)


def _jetty_sides(acc, w, part, dz, mat, c):
    """Seitenflaechen eines vorgezogenen Wandbands: zur Nachbarzone (Untersicht der Auskragung) bzw. zur Originalwand-Ebene
    (Kanten an Ecken/Dach, Ebene d=0 exakt auf den LoD2-Kanten)."""
    from shapely.geometry.polygon import orient
    ring = np.asarray(orient(part, 1.0).exterior.coords)[:-1]
    n = len(ring)
    for i in range(n):
        a, bq = ring[i], ring[(i + 1) % n]
        e = bq - a; L = np.linalg.norm(e)
        if L < 1e-4:
            continue
        out = np.array([e[1], -e[0]]) / L
        q = (a + bq) / 2 + out * 0.02
        base = jetty_dz(w, q[1]) if w.poly.contains(Point(q)) else 0.0
        if base >= dz - 1e-6:
            continue
        V = np.array([[a[0], a[1], base], [bq[0], bq[1], base], [bq[0], bq[1], dz], [a[0], a[1], dz]])
        W = w.fr.w(V)
        if base == 0.0:
            W[:2] = w.keep3d(np.array([a, bq]), W[:2])
        want = out[0] * w.fr.t + out[1] * w.fr.b
        _oriented(acc, 'shell', mat, W, [[0, 1, 2], [0, 2, 3]], [want, want], c)


def _oriented(acc, layer, mat, V, F, Wn, c):
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    if len(F) == 0:
        return
    tri = V[F]; nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum('ij,ij->i', nn, np.asarray(Wn, float)) < 0
    F = F.copy(); F[flip] = F[flip][:, [0, 2, 1]]
    acc.add(layer, mat, V, F, c)


def emit_wall(acc, inst, w, b, P, G):
    if w.drop or w.poly.is_empty:
        return
    wall_mat, wall_col = P['wall_mat'], P['wall_col']
    if getattr(w, 'glazed', False):
        wall_mat, wall_col = 'spandrel', '#2B3940'
    if wall_mat == 'glass':
        wall_mat = 'spandrel'
    gf_mat, gf_col = P.get('gf_mat_eff', wall_mat), P.get('gf_col_eff', wall_col)
    if gf_mat == 'glass':
        gf_mat = 'spandrel'
    if w.party_wall:
        pm = wall_mat if wall_mat in ('brick', 'brick_y', 'stone', 'plaster', 'rubble') else 'plaster'
        pc = wall_col if wall_mat in ('brick', 'brick_y', 'stone', 'plaster', 'rubble') else P.get('party_col', '#C8C2B8')
        zones = [(w.poly, pm, pc)]
    elif not w.vertical:
        zones = [(w.poly, wall_mat, wall_col)]
    else:
        ysplit = (floor_y(b, G, 1) if G['n'] > 1 else b.y_ground + G['g']) - 0.12
        if (gf_mat, gf_col) != (wall_mat, wall_col) and w.vmin < ysplit - 0.05 and w.vmax > ysplit + 0.05:
            lo = w.poly.intersection(sbox(w.umin - 1, w.vmin - 1, w.umax + 1, ysplit))
            hi = w.poly.intersection(sbox(w.umin - 1, ysplit, w.umax + 1, w.vmax + 1))
            zones = [(lo, gf_mat, gf_col), (hi, wall_mat, wall_col)]
        else:
            m, c = (gf_mat, gf_col) if w.vmax < ysplit + 0.05 else (wall_mat, wall_col)
            zones = [(w.poly, m, c)]
    hole_polys = [(o, sbox(o.u0, o.v0, o.u1, o.v1) if o.poly is None else Polygon(o.poly)) for o in w.openings]
    # nur Oeffnungen, die vollstaendig in der Wand liegen (sonst kein Fenster ohne Loch und kein Loch ohne Fenster)
    hole_polys = [(o, h) for o, h in hole_polys if w.poly.buffer(1e-6).contains(h)]
    allh = unary_union([h for _, h in hole_polys]) if hole_polys else None
    for zp, mat, c in zones:
        g = zp.difference(allh) if allh is not None and zp.intersects(allh) else zp
        bands = w.jetty if (w.jetty and w.vertical and not w.party_wall) else [(-1e6, 1e6, 0.0)]
        for (y0, y1, dz) in bands:
            gb = g if len(bands) == 1 else g.intersection(sbox(w.umin - 1, max(y0, w.vmin - 1), w.umax + 1, min(y1, w.vmax + 1)))
            for part in _geoms(gb):
                ext = np.array(part.exterior.coords)[:-1]
                ints = [np.array(rr.coords)[:-1] for rr in part.interiors]
                if dz <= 0:
                    acc.poly('shell', mat, w.fr, ext, c, ints, keep3d=w.keep3d)
                else:
                    acc.poly('shell', mat, w.fr, ext, c, ints, d=dz)
                    _jetty_sides(acc, w, part, dz, mat, c)
    for o, h in hole_polys:
        cpt = h.representative_point()
        mat, c = zones[0][1], zones[0][2]
        for zp, zm, zc in zones:
            if zp.contains(cpt):
                mat, c = zm, zc; break
        dz = getattr(o, 'dz', 0.0)
        fro = w.fr.shifted(dd=dz) if dz else w.fr
        shell_opening(acc, fro, o, mat, c)
        for (key, mm, origin, uc, vb, sx, sy, colr) in o.insts:
            d = -o.r if origin == 'frame' else 0.0
            inst.add(key, mm, w.fr, uc, vb, d + dz, sx, sy, 1.0, colr)
    if any(getattr(o, 'room', None) is not None for o, _ in hole_polys):
        import shops as SH
        SH.emit_rooms(acc, inst, w, b, [o for o, _ in hole_polys])


# ------------------------------------------------------------------ Baender / Gesimse
def band_runs(walls, b, ctx, y0, y1, need_party_free=True, min_len=0.03, deltas=(0.3,)):
    vol = ctx['vol']
    runs = []
    for w in walls:
        if w.drop or not w.vertical or w.vmin > y0 + 0.01 or w.vmax < y1 - 0.01:
            continue
        if need_party_free and w.party_wall:
            continue
        for ua, ub in _span_at(w, y0, y1):
            if ub - ua < min_len:
                continue
            e_ = min(0.05, (ub - ua) / 4); us = np.linspace(ua + e_, ub - e_, max(2, int((ub - ua) / 0.5) + 1))
            ok = open_at(w, b, vol, us, (y0 + y1) / 2, deltas=deltas)
            if need_party_free:
                ok &= ~is_party(w, us)
            for i0, i1 in _runs(list(ok)):
                a = ua if i0 == 0 else (us[i0 - 1] + us[i0]) / 2
                c = ub if i1 == len(us) - 1 else (us[i1] + us[i1 + 1]) / 2
                if c - a >= min_len:
                    runs.append((w, a, c))
    return runs


def _span_at(w, y0, y1):
    l0 = w.poly.intersection(LineString([(w.umin - 1, y0 + 0.002), (w.umax + 1, y0 + 0.002)]))
    l1 = w.poly.intersection(LineString([(w.umin - 1, y1 - 0.002), (w.umax + 1, y1 - 0.002)]))
    return _isect(_intervals(l0), _intervals(l1))


def _intervals(g):
    res = []
    if g.is_empty:
        return res
    for s in ([g] if g.geom_type == 'LineString' else list(getattr(g, 'geoms', []))):
        if s.geom_type != 'LineString':
            continue
        xs = [c[0] for c in s.coords]
        res.append((min(xs), max(xs)))
    res.sort()
    return res


def _isect(A, B):
    out = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > 1e-3:
                out.append((lo, hi))
    return out


def emit_band(acc, runs, b, y0, profile, mat, c, layer='detail'):
    if not runs:
        return
    prof = np.asarray(profile, float)
    ends = [(w.fr.w([ua, y0, 0]), w.fr.w([ub, y0, 0])) for (w, ua, ub) in runs]
    starts = np.array([e[0] for e in ends]); stops = np.array([e[1] for e in ends])
    nxt = {}; prv = {}
    for k in range(len(runs)):
        d = np.linalg.norm(starts - stops[k], axis=1); d[k] = 9
        j = int(np.argmin(d))
        if d[j] < 0.03 and j not in prv:
            nxt[k] = j; prv[j] = k
    dmax = prof[:, 0].max()
    for k, (w, ua, ub) in enumerate(runs):
        n = w.n
        sm = mitre(runs[prv[k]][0].n, n) if k in prv else None
        em = mitre(n, runs[nxt[k]][0].n) if k in nxt else None
        L = ub - ua
        es = float(sm @ w.fr.t) * dmax if sm is not None else 0.0
        ee = float(em @ w.fr.t) * dmax if em is not None else 0.0
        if L + ee - es < 0.02:
            sm = em = None
        pa, pb = ends[k]
        sweep(acc, layer, mat, np.array([pa, pb]), np.array([n]), prof, c, y0=y0, start_mitre=sm, end_mitre=em)


PROFILES = {
    'altbau_cornice': [(0, 0), (0.05, 0), (0.05, 0.1), (0.1, 0.15), (0.1, 0.23), (0.2, 0.28), (0.32, 0.36), (0.42, 0.42),
                       (0.42, 0.54), (0.38, 0.58), (0, 0.58)],
    'altbau_frieze': [(0, 0), (0.025, 0), (0.025, 0.42), (0, 0.42)],
    'altbau_gurt': [(0, 0), (0.08, 0), (0.12, 0.05), (0.12, 0.2), (0.17, 0.24), (0.17, 0.3), (0, 0.3)],
    'altbau_band': [(0, 0), (0.05, 0), (0.07, 0.035), (0.07, 0.12), (0.04, 0.15), (0, 0.15)],
    'slab': [(0, 0), (0.035, 0), (0.045, 0.01), (0.045, 0.19), (0.035, 0.2), (0, 0.2)],
    'slab_cornice': [(0, 0), (0.28, 0.0), (0.3, 0.02), (0.3, 0.15), (0.28, 0.17), (0, 0.17)],
    'attic': [(0, 0), (0.03, 0), (0.03, 0.55), (0, 0.55)],
    'attic_kh': [(0, 0), (0.06, 0), (0.06, 0.08), (0.1, 0.12), (0.1, 0.85), (0, 0.85)],
    'thin': [(0, 0), (0.08, 0), (0.1, 0.06), (0.1, 0.14), (0, 0.14)],
}


def plinth_profile(ph, dp=0.04):
    return [(0, 0), (dp, 0), (dp, ph - 0.01), (dp - 0.01, ph), (0, ph + dp * 0.6)]


def emit_bands(acc, walls, b, P, G, ctx, pitched, eaves_off):
    kind = P['kind']
    tm, tc = P['trim_mat'], P['trim_col']
    if tm == 'glass':
        tm = 'metal'
    if tm not in ('plaster', 'stone', 'brick', 'brick_y', 'rubble', 'panel', 'metal', 'trim'):
        tm = 'plaster'
    if kind != 'canopy':
        ph = (b.y_ground + (0.55 if kind != 'altbau' else 0.75)) - b.y_base
        runs = band_runs([w for w in walls if not w.retail], b, ctx, b.y_base + 0.31, b.y_ground + 0.5, need_party_free=False, deltas=(0.25,))
        pm = 'stone' if kind in ('altbau', 'gothic') else 'plaster'
        emit_band(acc, runs, b, b.y_base, plinth_profile(ph, 0.04), pm, P['plinth_col'] if pm == 'plaster' else '#B8B0A2')
    n = G['n']
    for f in P.get('rings') or ():
        yr = b.y_ground + (max(w.vcorner for w in walls if not w.drop and w.vertical) - b.y_ground) * f if any(
            not w.drop and w.vertical for w in walls) else None
        if yr is not None:
            emit_band(acc, band_runs(walls, b, ctx, yr, yr + 0.3), b, yr, PROFILES['altbau_gurt'], tm, tc)
    if P.get('bands') and n > 1:
        for k in range(1, n):
            fy = floor_y(b, G, k)
            if P['bands'] == 'altbau':
                prof = PROFILES['altbau_gurt'] if k == 1 else PROFILES['altbau_band']
                y0 = fy - (0.15 if k == 1 else 0.07)
            elif P['bands'] == 'fachwerk':
                continue
            else:
                prof = PROFILES['slab']; y0 = fy - 0.1
            h = max(p[1] for p in prof)
            emit_band(acc, band_runs(walls, b, ctx, y0, y0 + h), b, y0, prof, tm, tc)
    ck = P.get('cornice')
    if not ck:
        return
    tops = {}
    for w in walls:
        if w.drop or not w.vertical or w.party_wall:
            continue
        top = round(w.vcorner, 2)
        if top - b.y_ground < 2.5:
            continue
        tops.setdefault(top, []).append(w)
    for top, ws in tops.items():
        off = eaves_off if pitched else 0.06
        if ck == 'altbau':
            y0 = top - off - 0.58
            emit_band(acc, band_runs(ws, b, ctx, y0, top - off), b, y0, PROFILES['altbau_cornice'], tm, tc)
            yf = y0 - 0.42
            if yf > b.y_ground + 3:
                emit_band(acc, band_runs(ws, b, ctx, yf, y0), b, yf, PROFILES['altbau_frieze'], tm, tc)
        elif ck == 'slab':
            y0 = top - off - (0.17 if pitched else 0.25)
            emit_band(acc, band_runs(ws, b, ctx, y0, y0 + 0.17), b, y0, PROFILES['slab_cornice'], tm, tc)
        elif ck == 'attic':
            prof = PROFILES['attic_kh'] if kind == 'kaufhaus' else PROFILES['attic']
            h = max(p[1] for p in prof)
            y0 = top - off - h - 0.02
            emit_band(acc, band_runs(ws, b, ctx, y0, y0 + h), b, y0, prof, tm, tc)
        elif ck == 'thin':
            y0 = top - off - 0.16
            emit_band(acc, band_runs(ws, b, ctx, y0, y0 + 0.14), b, y0, PROFILES['thin'], tm, tc)


def emit_signs(acc, flags, b, P, rng):
    """Werbe-/Blendband ueber den Schaufenstern mit Relief-Buchstaben, Ausleger, Markisen mit Konsolen."""
    for w, runs, v1, fy1 in flags.get('signs', []):
        if not runs:
            continue
        groups = []
        for u0, u1, _ in sorted(runs):
            if groups and u0 - groups[-1][1] < 1.2:
                groups[-1][1] = u1
            else:
                groups.append([u0, u1])
        ys = v1 + 0.14; ye = min(fy1 - 0.14, ys + 0.62)
        if ye - ys < 0.25:
            continue
        fr = w.fr
        for g0, g1 in groups:
            u0, u1 = g0 - 0.15, g1 + 0.15
            col = P['sign_col'] if rng.random() < 0.7 else ['#2F3337', '#7A1F24', '#1F2A44', '#F2F2EE'][int(rng.integers(4))]
            acc.box('detail', 'trim', fr, [u0, ys, 0], [u1, ye, 0.07], col, skip=('-d',))
            acc.box('detail', 'metal', fr, [u0 - 0.01, ys - 0.03, 0], [u1 + 0.01, ys, 0.09], '#8C969C', skip=('-d',))
            acc.box('detail', 'metal', fr, [u0 - 0.01, ye, 0], [u1 + 0.01, ye + 0.03, 0.09], '#8C969C', skip=('-d',))
            lc = '#F2F2EE' if col not in ('#F2F2EE', '#B8B0A2') else '#22201F'
            if rng.random() < 0.5 and col != '#F2F2EE':
                lc = ['#E8C468', '#F2F2EE', '#D94A3A'][int(rng.integers(3))]
            # mehrere "Woerter" aus Relief-Buchstaben je Laden (Breite ~ Ladenbreite)
            L = u1 - u0
            nl = int(min(16, max(3, L / 0.4)) * rng.uniform(0.45, 0.85))
            hh = min(0.34, (ye - ys) * 0.62); lw = hh * 0.62; gap = hh * 0.16
            start = (u0 + u1) / 2 - nl * (lw + gap) / 2
            yc = (ys + ye) / 2
            for i in range(nl):
                if rng.random() < 0.1:
                    continue
                x = start + i * (lw + gap)
                shape = rng.integers(4)
                if shape == 0:                                       # Block
                    acc.box('detail', 'trim', fr, [x, yc - hh / 2, 0.07], [x + lw, yc + hh / 2, 0.095], lc, skip=('-d',))
                elif shape == 1:                                     # zwei Stege (H/N)
                    acc.box('detail', 'trim', fr, [x, yc - hh / 2, 0.07], [x + lw * 0.3, yc + hh / 2, 0.095], lc, skip=('-d',))
                    acc.box('detail', 'trim', fr, [x + lw * 0.7, yc - hh / 2, 0.07], [x + lw, yc + hh / 2, 0.095], lc, skip=('-d',))
                    acc.box('detail', 'trim', fr, [x + lw * 0.3, yc - hh * 0.1, 0.07], [x + lw * 0.7, yc + hh * 0.1, 0.09], lc, skip=('-d',))
                elif shape == 2:                                     # E/F
                    acc.box('detail', 'trim', fr, [x, yc - hh / 2, 0.07], [x + lw * 0.3, yc + hh / 2, 0.095], lc, skip=('-d',))
                    for yy in (yc + hh / 2 - hh * 0.2, yc - hh * 0.08, yc - hh / 2):
                        acc.box('detail', 'trim', fr, [x + lw * 0.3, yy, 0.07], [x + lw, yy + hh * 0.18, 0.09], lc, skip=('-d',))
                else:                                                # O
                    acc.box('detail', 'trim', fr, [x, yc - hh / 2, 0.07], [x + lw, yc - hh / 2 + hh * 0.2, 0.095], lc, skip=('-d',))
                    acc.box('detail', 'trim', fr, [x, yc + hh / 2 - hh * 0.2, 0.07], [x + lw, yc + hh / 2, 0.095], lc, skip=('-d',))
                    acc.box('detail', 'trim', fr, [x, yc - hh * 0.3, 0.07], [x + lw * 0.25, yc + hh * 0.3, 0.095], lc, skip=('-d',))
                    acc.box('detail', 'trim', fr, [x + lw * 0.75, yc - hh * 0.3, 0.07], [x + lw, yc + hh * 0.3, 0.095], lc, skip=('-d',))
            if rng.random() < 0.3 and fy1 + 1.4 < w.vmax:
                ux = u0 + 0.35
                acc.box('detail', 'metal', fr, [ux - 0.02, fy1 + 0.9, 0], [ux + 0.02, fy1 + 0.95, 0.8], '#2F3337', skip=('-d',))
                acc.box('detail', 'metal', fr, [ux - 0.06, fy1 + 0.82, 0], [ux + 0.06, fy1 + 1.02, 0.03], '#2F3337', skip=('-d',))
                acc.box('detail', 'trim', fr, [ux - 0.035, fy1 + 0.2, 0.14], [ux + 0.035, fy1 + 0.88, 0.76], lc, skip=())
        if P['awnings']:
            for u0, u1, vv in runs:
                if rng.random() < 0.5:
                    continue
                d1 = float(rng.uniform(1.0, 1.5)); drop = float(rng.uniform(0.4, 0.6))
                ya = vv + 0.1
                cc = P['awning_col']
                pts = np.array([[u0, ya, 0.0], [u1, ya, 0.0], [u1, ya - drop, d1], [u0, ya - drop, d1]])
                acc.add('detail', 'trim', fr.w(pts), [[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]], cc)
                val = np.array([[u0, ya - drop, d1], [u1, ya - drop, d1], [u1, ya - drop - 0.22, d1], [u0, ya - drop - 0.22, d1]])
                acc.add('detail', 'trim', fr.w(val), [[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]], cc)
                for uu in (u0, u1):
                    tri = np.array([[uu, ya, 0.0], [uu, ya - drop, d1], [uu, ya - drop - 0.22, d1], [uu, ya - 0.05, 0.0]])
                    acc.add('detail', 'trim', fr.w(tri), [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], cc)
                # Kasten + Gelenkarme (Konsolen)
                acc.box('detail', 'metal', fr, [u0 - 0.03, ya - 0.02, 0], [u1 + 0.03, ya + 0.14, 0.16], '#B4B8BA', skip=('-d',))
                for uu in (u0 + 0.15, u1 - 0.15):
                    p0 = fr.w([uu, ya - 0.6, 0.02]); p1 = fr.w([uu, ya - drop + 0.02, d1 - 0.05])
                    from geom import beam
                    beam(acc, 'detail', 'metal', p0, p1, 0.035, 0.035, '#8C969C')
                    acc.box('detail', 'metal', fr, [uu - 0.05, ya - 0.7, 0], [uu + 0.05, ya - 0.5, 0.04], '#8C969C', skip=('-d',))


def emit_fins(acc, flags, b, P, G):
    for w, BB, a in flags.get('fins', []):
        if len(BB) < 1:
            continue
        y0 = floor_y(b, G, 1) - 0.45 if G['n'] > 1 else b.y_ground + G['g']
        top = w.vcorner - 0.7
        if top - y0 < 2:
            continue
        us = [BB[0][0] - a / 2] + [c + a / 2 for c, _ in BB]
        for u in us:
            if u < w.umin + 0.1 or u > w.umax - 0.1 or is_party(w, u):
                continue
            if not w.poly.contains(sbox(u - 0.05, y0, u + 0.05, top)):
                continue
            acc.box('detail', 'metal', w.fr, [u - 0.035, y0, 0], [u + 0.035, top, 0.11], P['trim_col'], skip=('-d',))


def emit_steps(acc, flags, b):
    for w, uc, dw, lift, vb in flags.get('steps', []):
        if lift <= 0:
            continue
        n = 2
        for s in range(n):
            dd = 0.34 * (n - s)
            top = vb - s * (lift / n)
            acc.box('detail', 'stone', w.fr, [uc - dw / 2 - 0.3 - 0.08 * (n - s), b.y_base, 0], [uc + dw / 2 + 0.3 + 0.08 * (n - s), top, dd],
                    '#C8C0B4', skip=('-d',))


def emblem(acc, walls, b, P, G, clear=False):
    """Grosses Relief-Emblem auf der Natursteinfassade (Override 'emblem_wall'): stilisiertes Kreuz aus vier Balken und ein
    Kranz quadratischer Noppen - freie geometrische Form, keine Nachbildung eines Firmenlogos. clear=True: Fenster im
    Emblemfeld vorher entfernen."""
    ws = [w for w in walls if w.idx == P.get('emblem_wall') and not w.drop and w.vertical]
    if not ws:
        return
    w = ws[0]
    R = 2.1
    uc = (w.umin + w.umax) / 2
    yc = floor_y(b, G, 1) + 3.6 + R
    if yc + R + 0.6 > w.vcorner or w.L < 2 * R + 2:
        return
    if clear:
        w.openings = [o for o in w.openings if o.u1 < uc - R - 0.5 or o.u0 > uc + R + 0.5 or o.v1 < yc - R - 0.5 or o.v0 > yc + R + 0.5]
        return
    c = '#D2D0CA'
    for ang in (35, 145, 215, 325):
        a = math.radians(ang)
        _diag(acc, w.fr, uc + 0.35 * math.cos(a), yc + 0.35 * math.sin(a), uc + 1.55 * math.cos(a), yc + 1.55 * math.sin(a), c, 0.28, 0.12)
    for k in range(12):
        a = 2 * math.pi * k / 12
        x, y = uc + R * math.cos(a), yc + R * math.sin(a)
        acc.box('detail', 'stone', w.fr, [x - 0.16, y - 0.16, 0], [x + 0.16, y + 0.16, 0.1], c, skip=('-d',))
    acc.box('detail', 'stone', w.fr, [uc - 0.22, yc - 0.22, 0], [uc + 0.22, yc + 0.22, 0.14], c, skip=('-d',))


def _vspan(w, u, y0, y1):
    """Senkrechte Ausdehnung der Wand bei u innerhalb [y0, y1] -> (lo, hi) oder None."""
    g = w.poly.intersection(LineString([(u, y0), (u, y1)]))
    if g.is_empty:
        return None
    ys = []
    for seg in ([g] if g.geom_type == 'LineString' else [x for x in getattr(g, 'geoms', []) if x.geom_type == 'LineString']):
        ys += [c[1] for c in seg.coords]
    return (min(ys), max(ys)) if ys else None


def _tbox(acc, fr, u0, v0, u1, v1, d0, d1, c, skip=('-d',)):
    if u1 - u0 > 0.01 and v1 - v0 > 0.01:
        acc.box('detail', 'trim', fr, [u0, v0, d0], [u1, v1, d1], c, skip=skip)


def _wedge(acc, fr, u0, u1, tri_dv, c, mat='trim'):
    """Dreiecksprisma (Knagge/Konsole): Dreieck in der (d, v)-Ebene, extrudiert von u0 bis u1."""
    T = np.asarray(tri_dv, float)
    V = np.array([[u, v, d] for u in (u0, u1) for d, v in T])
    ctr = V.mean(0)
    F = [[0, 1, 2], [3, 5, 4]]
    for i in range(3):
        j = (i + 1) % 3
        F += [[i, j, 3 + j], [i, 3 + j, 3 + i]]
    F = np.array(F)
    Wl = V[F].mean(1) - ctr
    Wl[0] = [-1, 0, 0]; Wl[1] = [1, 0, 0]
    Ww = Wl @ fr.M
    _oriented(acc, 'detail', mat, fr.w(V), F, Ww, c)


def _prism_dv(acc, fr, u0, u1, poly_dv, c, mat='trim'):
    """Prisma aus einem konvexen Polygon in der (d, v)-Ebene, extrudiert von u0 bis u1 (Konsole, Kragstein)."""
    T = np.asarray(poly_dv, float); m = len(T)
    V = np.array([[u, v, d] for u in (u0, u1) for d, v in T])
    ctr = V.mean(0)
    F = [[0, i, i + 1] for i in range(1, m - 1)] + [[m, m + i + 1, m + i] for i in range(1, m - 1)]
    for i in range(m):
        j = (i + 1) % m
        F += [[i, j, m + j], [i, m + j, m + i]]
    F = np.array(F)
    Wl = V[F].mean(1) - ctr
    Wl[:m - 2] = [-1, 0, 0]; Wl[m - 2:2 * (m - 2)] = [1, 0, 0]
    _oriented(acc, 'detail', mat, fr.w(V), F, Wl @ fr.M, c)


def _diag(acc, fr, u0, v0, u1, v1, c, wdt=0.14, dp=0.04):
    from geom import beam
    beam(acc, 'detail', 'trim', fr.w([u0, v0, dp / 2]), fr.w([u1, v1, dp / 2]), wdt, dp, c, up_hint=fr.n)


def _fan(acc, fr, uc, vb, r, c, dp=0.05):
    """Faecherrosette (Halbscheibe mit Strahlen) in einem Brustungsfeld."""
    if r < 0.12:
        return
    a = np.linspace(0, np.pi, 11)
    ring = np.c_[uc + r * np.cos(a), vb + r * np.sin(a)]
    acc.poly('detail', 'trim', fr, ring, c, d=dp * 0.6)
    for k in range(1, 10, 2):
        _diag(acc, fr, uc, vb, uc + r * 0.95 * np.cos(a[k]), vb + r * 0.95 * np.sin(a[k]), '#EDE3CF', wdt=0.03, dp=dp * 0.9)


def _hits_sign(P, p, ya, yb, mu=0.15):
    """Liegt ein Bauteil (Weltpunkt p, Hoehe ya..yb) vor einer Werbeanlage/Markise des Gebaeudes?"""
    for fr, u0, u1, y0, y1 in P.get('_sign_spans', ()):
        q = fr.loc(p)
        if abs(q[2]) < 0.7 and u0 - mu < q[0] < u1 + mu and ya < y1 and yb > y0:
            return True
    return False


def emit_fachwerk(acc, walls, b, P, G):
    """Fachwerk je Geschossband (mit Auskragung): Schwelle, Balkenkoepfe + Fuellholz, Knaggen, Staender (Eck-, Fenster-,
    Bundstaender), Brust-/Sturzriegel, Andreaskreuze/Faecherrosetten, Streben ('Mann'), Giebel mit Kehlbalken und Ortgangbrettern."""
    from common import rng_of
    dp = 0.045; bw = 0.17
    for w in walls:
        if w.drop or not w.vertical or w.party_wall or w.L < 0.3:
            continue
        Pw = getattr(w, 'P', P)
        col = Pw.get('timber_col', P['trim_col'])
        brace = Pw.get('brace', 'cross')
        bands = w.jetty
        if not bands:
            ys = [floor_y(b, G, k) for k in range(1, G['n'])]
            if not ys:
                continue
            bands = [(-1e6, ys[0], 0.0)] + [(y, ys[k + 1] if k + 1 < len(ys) else 1e6, 0.0) for k, y in enumerate(ys)]
        ops = [o for o in w.openings if o.zone != 'gf']
        for k in range(1, len(bands)):
            y0, y1, dz = bands[k]
            dprev = bands[k - 1][2]
            fr = w.fr.shifted(dd=dz)
            top_band = k == len(bands) - 1
            yt = min(y1, w.vmax)
            spans = _span_at(w, y0 + 0.01, y0 + 0.22)
            bops = [o for o in ops if y0 <= (o.v0 + o.v1) / 2 < y1]
            for (ua, ub) in spans:
                if ub - ua < 0.25:
                    continue
                _tbox(acc, fr, ua, y0, ub, y0 + 0.2, 0.0, dp + 0.015, col)                 # Schwelle
                if ub - ua < 0.75:
                    # Fase / schmale Ecke: ein Eckstaender ueber die ganze Breite, Rähm darueber
                    yt2 = min(y1, w.vmax)
                    vs = _vspan(w, (ua + ub) / 2, y0 + 0.2, yt2 if top_band else yt2 - 0.24)
                    if vs is not None and vs[1] - vs[0] > 0.3:
                        _tbox(acc, fr, ua + 0.01, y0 + 0.2, ub - 0.01, vs[1] - (0.04 if top_band else 0.0), 0.0, dp, col)
                    if dz - dprev > 0.03:
                        _tbox(acc, fr, ua, y0 - 0.2, ub, y0, -(dz - dprev), -(dz - dprev) + 0.03, col, skip=('-d', '+v'))  # Fuellholz
                    continue
                if dz - dprev > 0.03:
                    D = dz - dprev
                    _tbox(acc, fr, ua, y0 - 0.2, ub, y0, -D, -D + 0.03, col, skip=('-d', '+v'))      # Fuellholz
                    for u in np.arange(ua + 0.12, ub - 0.1, 0.78):                                    # Balkenkoepfe
                        _tbox(acc, fr, u - 0.075, y0 - 0.24, u + 0.075, y0, -D, 0.03, col, skip=('-d', '+v'))
                    kn = [ua + 0.1, ub - 0.1]
                    if ub - ua > 6 and k > 1:                      # ueber dem Erdgeschoss nur an den Ecken (Platz fuer Ladenschilder)
                        kn += list(np.linspace(ua + 0.1, ub - 0.1, int((ub - ua) / 3) + 1)[1:-1])
                    for u in kn:
                        if k > 1:
                            # Knagge: ruht auf dem Eck-/Bundstaender des Geschosses darunter
                            if _hits_sign(P, fr.w([u, y0 - 0.5, -D]), y0 - 0.79, y0 - 0.24):
                                continue
                            _wedge(acc, fr, u - 0.07, u + 0.07, [(-D, y0 - 0.24 - 0.55), (-D, y0 - 0.24), (0.02, y0 - 0.24)], col)
                        else:
                            # ueber massivem Erdgeschoss (kein Staender darunter): kurze stumpfe Konsole statt spitzer Knagge
                            if _hits_sign(P, fr.w([u, y0 - 0.4, -D]), y0 - 0.56, y0 - 0.24):
                                continue
                            _prism_dv(acc, fr, u - 0.08, u + 0.08, [(-D, y0 - 0.24 - 0.3), (-D + 0.06, y0 - 0.24 - 0.3),
                                                                    (0.02, y0 - 0.24 - 0.08), (0.02, y0 - 0.24), (-D, y0 - 0.24)], col)
                posts = [(ua + 0.11, 0.22), (ub - 0.11, 0.22)]
                for o in bops:
                    posts += [(o.u0 - bw / 2 - 0.01, bw), (o.u1 + bw / 2 + 0.01, bw)]
                posts.sort()
                fill = []
                for (pa, _), (pb, _) in zip(posts, posts[1:]):
                    gap = pb - pa
                    if gap > 1.35 and not any(o.u0 - 0.2 < (pa + pb) / 2 < o.u1 + 0.2 for o in bops):
                        n = int(gap / 1.0)
                        fill += [(pa + gap * i / n, bw) for i in range(1, n)]
                posts = sorted(posts + fill)
                mp = []                                    # dicht stehende Fensterstaender zu einem Staender zusammenfassen
                for pu, pw in posts:
                    if mp and pu - mp[-1][0] < 0.3:
                        a0 = min(mp[-1][0] - mp[-1][1] / 2, pu - pw / 2); a1 = max(mp[-1][0] + mp[-1][1] / 2, pu + pw / 2)
                        mp[-1] = ((a0 + a1) / 2, a1 - a0)
                    else:
                        mp.append((pu, pw))
                posts = mp
                ytop = (yt - 0.42) if not top_band else yt
                drawn = []
                for (pu, pw) in posts:
                    if pu - pw / 2 < ua - 0.01 or pu + pw / 2 > ub + 0.01:
                        continue
                    vs = _vspan(w, pu, y0 + 0.2, ytop)
                    if vs is None or vs[1] - vs[0] < 0.4:
                        continue
                    if any(abs(pu - q) < 0.12 for q in drawn):
                        continue
                    _tbox(acc, fr, pu - pw / 2, y0 + 0.2, pu + pw / 2, vs[1] - (0.04 if top_band else 0.0), 0.0, dp, col)
                    drawn.append(pu)
                if not top_band and yt - y0 > 1.2:                                                     # Rähm
                    _tbox(acc, fr, ua, yt - 0.42, ub, yt - 0.24, 0.0, dp, col)
                for o in bops:                                                                        # Brust-/Sturzriegel
                    _tbox(acc, fr, o.u0 - bw, o.v0 - 0.15, o.u1 + bw, o.v0, 0.0, dp, col)
                    if o.v1 + 0.3 < ytop:
                        _tbox(acc, fr, o.u0 - bw, o.v1, o.u1 + bw, o.v1 + 0.14, 0.0, dp, col)
                    fa, fb, va, vb = o.u0 - 0.01, o.u1 + 0.01, y0 + 0.2, o.v0 - 0.15
                    if vb - va > 0.3 and fb - fa > 0.4:
                        if brace == 'foot':
                            _fan(acc, fr, (fa + fb) / 2, va, min((fb - fa) / 2 - 0.05, vb - va - 0.04), col)
                        else:
                            _diag(acc, fr, fa + 0.05, va + 0.04, fb - 0.05, vb - 0.04, col, 0.12, dp)
                            _diag(acc, fr, fa + 0.05, vb - 0.04, fb - 0.05, va + 0.04, col, 0.12, dp)
                dd = sorted(drawn)
                for pa, pb in zip(dd, dd[1:]):                                                        # geschlossene Felder
                    a_, b_ = pa + bw / 2, pb - bw / 2
                    if b_ - a_ < 0.45 or any(o.u0 - 0.25 < (a_ + b_) / 2 < o.u1 + 0.25 for o in bops):
                        continue
                    vs = _vspan(w, (a_ + b_) / 2, y0 + 0.2, ytop)
                    if vs is None or vs[1] - vs[0] < 0.8:
                        continue
                    lo, hi = vs[0], vs[1] - (0.05 if top_band else 0.0)
                    if not w.poly.contains(sbox(a_, lo, b_, hi)):
                        hi = min(hi, lo + 0.9)
                        if not w.poly.contains(sbox(a_, lo, b_, hi)):
                            continue
                    mid = (lo + hi) / 2
                    _tbox(acc, fr, a_, mid - 0.075, b_, mid + 0.075, 0.0, dp, col)
                    left = abs(pa - (ua + 0.11)) < 0.05
                    near_corner = left or abs(pb - (ub - 0.11)) < 0.05
                    if near_corner or brace == 'mann':
                        if left:
                            _diag(acc, fr, a_ + 0.06, lo + 0.05, b_ - 0.06, mid - 0.06, col, 0.15, dp)
                            _diag(acc, fr, a_ + 0.06, hi - 0.05, b_ - 0.06, mid + 0.06, col, 0.15, dp)
                        else:
                            _diag(acc, fr, b_ - 0.06, lo + 0.05, a_ + 0.06, mid - 0.06, col, 0.15, dp)
                            _diag(acc, fr, b_ - 0.06, hi - 0.05, a_ + 0.06, mid + 0.06, col, 0.15, dp)
                    elif brace == 'cross':
                        _diag(acc, fr, a_ + 0.05, lo + 0.04, b_ - 0.05, mid - 0.08, col, 0.12, dp)
                        _diag(acc, fr, a_ + 0.05, mid - 0.08, b_ - 0.05, lo + 0.04, col, 0.12, dp)
            if top_band and w.vmax - w.vcorner > 1.2:                                                 # Giebel
                yk = y0 + (w.vmax - y0) * 0.5
                for (ua, ub) in _span_at(w, yk - 0.08, yk + 0.08):
                    _tbox(acc, fr, ua + 0.05, yk - 0.08, ub - 0.05, yk + 0.08, 0.0, dp, col)
                ring = np.asarray(w.poly.exterior.coords)
                for a, c2 in zip(ring[:-1], ring[1:]):
                    e = c2 - a; L = np.linalg.norm(e)
                    if L < 0.8 or abs(e[0]) < 0.05 or abs(e[1]) < 0.3 or max(a[1], c2[1]) < y0 + 0.5:
                        continue
                    nin = np.array([-e[1], e[0]]) / L
                    if not w.poly.contains(Point((a + c2) / 2 + nin * 0.1)):
                        nin = -nin
                    p0 = a + nin * 0.11; p1 = c2 + nin * 0.11
                    if p0[1] < y0:
                        t = (y0 - p0[1]) / (p1[1] - p0[1]); p0 = p0 + (p1 - p0) * t
                    if p1[1] < y0:
                        t = (y0 - p1[1]) / (p0[1] - p1[1]); p1 = p1 + (p0 - p1) * t
                    _diag(acc, fr, p0[0], p0[1], p1[0], p1[1], col, 0.2, dp + 0.02)


def emit_brick_gable(acc, walls, b, P, G):
    """Backsteingotik: Staffelgiebel (Stufen ueber der Dachlinie mit Sandsteinabdeckung), spitzbogige Blendnischen mit hellem
    Putzgrund im Giebeldreieck, Strebepfeiler an Kirchen-Langseiten."""
    import text3d as TX3
    from shapely.geometry.polygon import orient
    brick, bc = P['wall_mat'], P['wall_col']
    if brick not in ('brick', 'brick_y', 'stone', 'rubble'):
        brick, bc = 'brick', '#E8D6CC'
    stone = P.get('trim_col', '#E6DDCB')
    if not isinstance(stone, str) or not stone.startswith('#'):
        stone = '#E6DDCB'
    for w in walls:
        if w.drop or not w.vertical or w.party_wall or w.L < 4:
            continue
        fr = w.fr
        gable = w.vmax - w.vcorner > 2.0
        if gable and P.get('stepgable', True):
            H0 = w.vcorner
            n = max(3, int(w.L / 1.6)) | 1
            us = np.linspace(w.umin, w.umax, n + 1)
            for i in range(n):
                ua, ub = us[i] + 0.02, us[i + 1] - 0.02
                sa = _vspan(w, ua + 0.05, H0 - 1, w.vmax + 1); sb = _vspan(w, ub - 0.05, H0 - 1, w.vmax + 1)
                if sa is None or sb is None:
                    continue
                t0 = min(sa[1], sb[1]); t1 = max(sa[1], sb[1])
                if t1 < H0 + 0.3:
                    continue
                top = t1 + 0.55
                acc.box('detail', brick, fr, [ua, t0 - 0.35, -0.45], [ub, top, 0.06], bc, skip=('-v',))
                acc.box('detail', 'stone', fr, [ua - 0.04, top, -0.5], [ub + 0.04, top + 0.08, 0.1], stone, skip=())
                nw = min(0.7, (ub - ua) * 0.55); nh = (t0 - 0.6) - (H0 + 0.4)
                if nh > 1.2 and nw > 0.35 and P.get('gable_niches', True):                           # Blendnische
                    uc = (ua + ub) / 2
                    lp = Polygon(PR.lancet_poly(nw, nh) + [uc, H0 + 0.4])
                    if w.poly.contains(lp.buffer(0.15)):
                        acc.poly('detail', 'plaster', fr, np.asarray(orient(lp, 1.0).exterior.coords)[:-1], '#ECE6DA', d=0.014)
                        ring = lp.buffer(0.1, join_style=2).difference(lp)
                        if ring.geom_type == 'Polygon':
                            ring = orient(ring, 1.0)
                            ext = np.asarray(ring.exterior.coords)[:-1]; ints = [np.asarray(r.coords)[:-1] for r in ring.interiors]
                            V, F, Wn = TX3.extrude_local([(ext, ints)], 0.0, 0.06)
                            if len(F):
                                _oriented(acc, 'detail', brick, fr.w(V), F, Wn @ fr.M, bc)
        elif P['kind'] == 'gothic' and w.L > 10:
            BB = bays(w, P, axis=P['axis'], margin=1.8, win_w=P['win_w'])                            # Strebepfeiler
            if len(BB) < 2:
                continue
            a = BB[1][0] - BB[0][0]
            for uc in [BB[0][0] - a / 2] + [c + a / 2 for c, _ in BB]:
                if uc < w.umin + 0.6 or uc > w.umax - 0.6 or is_party(w, uc):
                    continue
                vs = _vspan(w, uc, b.y_ground - 1, w.vmax)
                if vs is None:
                    continue
                top = min(vs[1], w.vcorner) - 1.2
                if top - b.y_ground < 4:
                    continue
                y1 = b.y_ground + (top - b.y_ground) * 0.55
                acc.box('detail', 'stone', fr, [uc - 0.5, b.y_base, 0], [uc + 0.5, b.y_ground + 0.9, 1.0], '#8A8278', skip=('-d', '-v'))
                acc.box('detail', brick, fr, [uc - 0.45, b.y_ground + 0.9, 0], [uc + 0.45, y1, 0.9], bc, skip=('-d', '-v'))
                _wedge(acc, fr, uc - 0.47, uc + 0.47, [(0.0, y1), (0.95, y1), (0.0, y1 + 0.8)], stone, mat='stone')
                acc.box('detail', brick, fr, [uc - 0.4, y1, 0], [uc + 0.4, top, 0.55], bc, skip=('-d', '-v'))
                _wedge(acc, fr, uc - 0.42, uc + 0.42, [(0.0, top), (0.6, top), (0.0, top + 0.9)], '#6E9C86', mat='metal')

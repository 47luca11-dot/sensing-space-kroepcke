"""Altstadt + Leine (Ausbau 2): Holzmarktbrunnen (Oskar-Winter-Brunnen, 1896, OSM 348132278) mit achteckigem Sockel, Treppen,
Eisengelaender, Steinbecken, schmiedeeisernem Gitter mit vergoldeten Spitzen und neugotischer Brunnensaeule (vergoldete Fiale);
Ballhofbrunnen (OSM 503561080, ovales Betonbecken mit dunklen Bronzeformen); Leine-Ufer (Hohes Ufer, Leibnizufer): Abdeckplatten
+ schmiedeeisernes Gelaender auf den Kaimauern, wo der begehbare Uferweg > 1 m ueber dem Wasser liegt.
(Pflaster: layout.py-Klassen alt_setts / clinker; Kaimauern: terrain_mesh.py)   Ausgabe: data/ground_cache/altstadt.pkl"""
import json, time, pickle, math
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from common import *
import geom as G
import geom2 as G2
from geom2 import Frame
from surface import Surface
from kit import _strip

T0 = time.time()
def log(*a): print('[altstadt %6.1fs]' % (time.time() - T0), *a, flush=True)
L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
SURF = Surface()
osm = load_osm()
parts = []; coll = []; walk = []
def add(p, collide=False, walkable=False):
    if p is None: return
    if isinstance(p, (list, tuple)):
        for q in p: add(q, collide, walkable)
        return
    if len(p['idx']) == 0: return
    parts.append(p)
    if collide: coll.append(p)
    if walkable: walk.append(p)
def gy(x, z): return float(SURF([x], [z])[0][0])
def node(nid, default):
    for n in osm['nodes']:
        if n['id'] == nid: return np.array(n['p'], float)
    return np.array(default, float)

def octagon(r, rot=math.pi / 8):
    a = rot + np.arange(8) * math.pi / 4
    return np.c_[r * np.cos(a), r * np.sin(a)]

def prism(mat, ring, y0, y1, c, uvscale=1.0, top=True, bottom=False, top_mat=None):
    return G.extrude(mat, ring + c, y0, y1, uvscale=uvscale, top=top, bottom=bottom, top_mat=top_mat)

def ring_wall(mat, r_out, r_in, y0, y1, c, uvscale=1.0, n=8, rot=math.pi / 8):
    """achteckige Ringmauer (aussen + innen + Krone), Normalen nach aussen bzw. zum Beckeninneren"""
    out = []
    Po = octagon(r_out, rot) + c; Pi = octagon(r_in, rot) + c
    Qo = []; Qi = []; Qt = []
    for i in range(n):
        j = (i + 1) % n
        Qo.append([[Po[i, 0], y0, Po[i, 1]], [Po[j, 0], y0, Po[j, 1]], [Po[j, 0], y1, Po[j, 1]], [Po[i, 0], y1, Po[i, 1]]])
        Qi.append([[Pi[j, 0], y0, Pi[j, 1]], [Pi[i, 0], y0, Pi[i, 1]], [Pi[i, 0], y1, Pi[i, 1]], [Pi[j, 0], y1, Pi[j, 1]]])
        Qt.append([[Po[i, 0], y1, Po[i, 1]], [Po[j, 0], y1, Po[j, 1]], [Pi[j, 0], y1, Pi[j, 1]], [Pi[i, 0], y1, Pi[i, 1]]])
    cc = np.array([c[0], (y0 + y1) / 2, c[1]])
    for Q, sgn in ((Qo, 1), (Qi, -1)):
        p = G.flat_quads(mat, np.array(Q), uvscale)
        P_, I_ = p['pos'], p['idx']; fn = np.cross(P_[I_[:, 1]] - P_[I_[:, 0]], P_[I_[:, 2]] - P_[I_[:, 0]])
        rad = P_[I_].mean(1) - cc; rad[:, 1] = 0
        bad = (fn * rad).sum(1) * sgn < 0; I_[bad] = I_[bad][:, ::-1]; p['nrm'] = G.vertex_normals(P_, I_)
        out.append(p)
    t = G.flat_quads(mat, np.array(Qt), uvscale); G.orient_up(t); t['nrm'] = np.tile([0, 1.0, 0], (len(t['pos']), 1)); out.append(t)
    return out

# ================================================================= Holzmarktbrunnen (Oskar-Winter-Brunnen)
def holzmarkt_brunnen():
    c = node(348132278, (-431.6, 332.2))
    hs = [gy(*(c + q)) for q in octagon(4.0)] + [gy(*c)]
    yb = max(hs)                       # Sockel-Unterkante auf hoechstem Gelaendepunkt
    R_step, R_pl, R_bo, R_bi = 4.15, 3.75, 2.55, 2.25
    y_step = yb + 0.16; y_pl = yb + 0.95; y_rim = y_pl + 0.78; y_w = y_rim - 0.16
    # unterste Stufe (umlaufend) + Sockel
    add(prism('sandstone', octagon(R_step), min(hs) - 0.1, y_step, c, 1.0), True, True)
    add(prism('sandstone', octagon(R_pl), y_step - 0.01, y_pl, c, 1.0, top=False), True)
    # Sockel-Oberflaeche (Plattenring zwischen Becken und Kante)
    Po = octagon(R_pl) + c; Pi = octagon(R_bo) + c; Q = []
    for i in range(8):
        j = (i + 1) % 8
        Q.append([[Po[i, 0], y_pl, Po[i, 1]], [Po[j, 0], y_pl, Po[j, 1]], [Pi[j, 0], y_pl, Pi[j, 1]], [Pi[i, 0], y_pl, Pi[i, 1]]])
    t = G.flat_quads('stone_grey', np.array(Q), 1.0); G.orient_up(t); t['nrm'] = np.tile([0, 1.0, 0], (len(t['pos']), 1)); add(t, True, True)
    # Treppen auf zwei gegenueberliegenden Seiten (Ost/West), 3 Stufen
    for ang in (0.0, math.pi):
        d = np.array([math.cos(ang), math.sin(ang)])
        fr = Frame(c + d * (R_pl * math.cos(math.pi / 8)), d)          # s nach aussen, ab Sockelkante
        nst = 3; rise = (y_pl - y_step) / nst
        for k in range(nst):
            add(fr.box('sandstone', 0.32 * k, 0.32 * (k + 1), -0.85, 0.85, y_step - 0.01, y_pl - k * rise, 1.0), True, True)
    # Eisengelaender auf der Sockelkante (Luecken an den Treppen)
    Pr = octagon(R_pl - 0.12) + c
    for i in range(8):
        j = (i + 1) % 8
        a, b = Pr[i], Pr[j]; m = (a + b) / 2; ang = math.atan2(*(m - c)[::-1])
        if min(abs(((ang - 0.0) + math.pi) % (2 * math.pi) - math.pi), abs(((ang - math.pi) + math.pi) % (2 * math.pi) - math.pi)) < 0.3:
            continue
        P3 = np.array([[a[0], y_pl, a[1]], [b[0], y_pl, b[1]]])
        S = np.linspace(0, 1, 4)
        Q3 = P3[0] + (P3[1] - P3[0]) * S[:, None]
        for q in Q3: add(G2.cyl_between('iron_dark', q, q + [0, 1.0, 0], 0.025, 6))
        add(G.tube('iron_dark', Q3 + [0, 1.0, 0], 0.028, 6)); add(G.tube('iron_dark', Q3 + [0, 0.5, 0], 0.014, 5))
        coll.append(_strip('collision', Q3, Q3 + [0, 1.0, 0], look=np.r_[c[0], y_pl + 0.5, c[1]]))
    # Becken: achteckige Steinmauer, Wasser
    add(ring_wall('sandstone', R_bo, R_bi, y_pl - 0.01, y_rim, c, 1.0), True)
    add(prism('water_basin', octagon(R_bi + 0.01), y_w - 0.3, y_w, c, 2.0, top=True))
    # Schmiedeeisernes Gitter auf dem Beckenrand mit vergoldeten Spitzen
    Pg = octagon(R_bo - 0.15) + c; yg0, yg1 = y_rim, y_rim + 1.45
    for i in range(8):
        j = (i + 1) % 8; a, b = Pg[i], Pg[j]
        add(G2.cyl_between('iron_dark', [a[0], yg0, a[1]], [a[0], yg1 + 0.15, a[1]], 0.045, 8))
        add(G.cylinder('gold', (a[0], yg1 + 0.15, a[1]), 0.07, 0.0, 0.28, 8, cap=False))
        nb = 13
        for k in range(1, nb):
            q = a + (b - a) * k / nb
            add(G2.cyl_between('iron_dark', [q[0], yg0, q[1]], [q[0], yg1 - 0.05, q[1]], 0.011, 4, caps=False))
            if k % 2 == 0: add(G.cylinder('gold', (q[0], yg1 - 0.05, q[1]), 0.025, 0.0, 0.12, 4, cap=False))
        for hh in (yg0 + 0.12, yg0 + 0.95, yg1 - 0.08):
            add(G.tube('iron_dark', np.array([[a[0], hh, a[1]], [b[0], hh, b[1]]]), 0.018, 5, caps=False))
        # Rankenwerk: Ring je Feld
        m = (a + b) / 2; u = np.r_[(b - a)[0], 0, (b - a)[1]]; u /= np.linalg.norm(u)
        add(G2.ring_tube('iron_dark', [m[0], yg0 + 0.53, m[1]], 0.33, 0.014, 14, 4, axis_u=u, axis_v=(0, 1, 0)))
        add(G2.ring_tube('gold', [m[0], yg0 + 0.53, m[1]], 0.10, 0.012, 8, 4, axis_u=u, axis_v=(0, 1, 0)))
        coll.append(_strip('collision', np.array([[a[0], yg0, a[1]], [b[0], yg0, b[1]]]), np.array([[a[0], yg1, a[1]], [b[0], yg1, b[1]]]), look=np.r_[c[0], yg0 + 0.5, c[1]] * 2 - np.r_[m[0], yg0 + 0.5, m[1]]))
    # Brunnensaeule: Steinsockel, dunkle Eisensaeule, Tabernakel mit Figur, Fialen, vergoldete Spitze
    yf = y_w - 0.3
    add(prism('sandstone', octagon(0.72), yf, y_w + 0.45, c, 1.0), True)
    add(prism('stone_grey', octagon(0.55), y_w + 0.45, y_w + 1.55, c, 1.0), True)
    add(prism('iron_dark', octagon(0.62), y_w + 1.55, y_w + 1.75, c, 1.0))
    y0 = y_w + 1.75
    add(G.cylinder('iron_dark', (c[0], y0, c[1]), 0.20, 0.16, 4.2, 12, cap=False))
    # Wasserspeier (4) aus dem Steinsockel
    for k in range(4):
        a_ = k * math.pi / 2 + math.pi / 4; d = np.array([math.cos(a_), math.sin(a_)])
        p0 = c + d * 0.5; p1 = c + d * 0.95
        add(G.tube('bronze', np.array([[p0[0], y_w + 1.1, p0[1]], [p1[0], y_w + 1.0, p1[1]], [(c + d * 1.05)[0], y_w + 0.92, (c + d * 1.05)[1]]]), 0.035, 6))
    # Tabernakel-Geschoss (y0+0.6 .. y0+2.2): 8 Saeulchen, Spitzbogen-Wimperge (Kegel), Figur in der Mitte
    yt0, yt1 = y0 + 0.6, y0 + 2.2
    add(prism('iron_dark', octagon(0.75), yt0 - 0.12, yt0, c, 1.0, bottom=True))
    for q in octagon(0.62):
        p = c + q
        add(G.cylinder('iron_dark', (p[0], yt0, p[1]), 0.035, 0.035, yt1 - yt0, 6, cap=False))
        add(G.cylinder('gold', (p[0], yt1, p[1]), 0.07, 0.0, 0.45, 6, cap=False))
    add(G.cylinder('bronze', (c[0], yt0, c[1]), 0.17, 0.12, 1.15, 10, cap=True))           # Figur (vereinfacht)
    add(G.cylinder('bronze', (c[0], yt0 + 1.15, c[1]), 0.10, 0.08, 0.22, 8, cap=True))
    add(prism('iron_dark', octagon(0.78), yt1, yt1 + 0.14, c, 1.0, bottom=True))
    for k, q in enumerate(octagon(0.70)):
        p = c + q
        add(G.cylinder('iron_dark', (p[0], yt1 + 0.14, p[1]), 0.10, 0.0, 0.85, 4, cap=False))
        if k % 2 == 0: add(G.cylinder('gold', (p[0], yt1 + 0.99, p[1]), 0.04, 0.0, 0.18, 4, cap=False))
    # Spitze mit Krabben + vergoldete Kreuzblume
    ys = yt1 + 0.14
    add(G.cylinder('iron_dark', (c[0], ys, c[1]), 0.36, 0.05, 2.6, 8, cap=False))
    for k in range(6):
        hh = ys + 0.45 + k * 0.35; rr = 0.36 * (1 - (hh - ys) / 2.6) + 0.05
        for q in octagon(rr + 0.04)[::2]:
            p = c + q; add(G.cylinder('gold', (p[0], hh, p[1]), 0.035, 0.0, 0.12, 4, cap=False))
    yfin = ys + 2.6
    add(G.cylinder('gold', (c[0], yfin - 0.05, c[1]), 0.05, 0.05, 0.55, 6, cap=False))
    for ang in (0, math.pi / 2):
        d = np.array([math.cos(ang), math.sin(ang)]) * 0.18
        add(G.tube('gold', np.array([[c[0] - d[0], yfin + 0.30, c[1] - d[1]], [c[0] + d[0], yfin + 0.30, c[1] + d[1]]]), 0.025, 6))
    add(G2.ring_tube('gold', [c[0], yfin + 0.30, c[1]], 0.12, 0.02, 10, 4, axis_u=(1, 0, 0), axis_v=(0, 1, 0)))
    add(G.cylinder('gold', (c[0], yfin + 0.5, c[1]), 0.06, 0.0, 0.25, 6, cap=False))
    log('Holzmarktbrunnen at', np.round(c, 1), 'top y %.1f' % (yfin + 0.75), 'height %.1f m' % (yfin + 0.75 - yb))
    return {'name': 'Holzmarktbrunnen (Oskar-Winter-Brunnen)', 'p': c.round(2).tolist(), 'y': round(yb, 2)}

# ================================================================= Ballhofbrunnen
def ballhof_brunnen():
    c = node(503561080, (-410.5, 213.4))
    a_, b_ = 3.2, 2.1; n = 32
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring_o = np.c_[a_ * np.cos(t), b_ * np.sin(t)]; ring_i = np.c_[(a_ - 0.3) * np.cos(t), (b_ - 0.3) * np.sin(t)]
    hs = [gy(*(c + q)) for q in ring_o]; yb = max(hs)
    yr = yb + 0.42
    Q_o, Q_i, Q_t = [], [], []
    for i in range(n):
        j = (i + 1) % n; po, pj, qi, qj = ring_o[i] + c, ring_o[j] + c, ring_i[i] + c, ring_i[j] + c
        Q_o.append([[po[0], min(hs) - 0.1, po[1]], [pj[0], min(hs) - 0.1, pj[1]], [pj[0], yr, pj[1]], [po[0], yr, po[1]]])
        Q_i.append([[qj[0], yb, qj[1]], [qi[0], yb, qi[1]], [qi[0], yr, qi[1]], [qj[0], yr, qj[1]]])
        Q_t.append([[po[0], yr, po[1]], [pj[0], yr, pj[1]], [qj[0], yr, qj[1]], [qi[0], yr, qi[1]]])
    cc = np.r_[c[0], yr, c[1]]
    for Q, sgn in ((Q_o, 1), (Q_i, -1)):
        p = G.flat_quads('concrete', np.array(Q), 1.0)
        P_, I_ = p['pos'], p['idx']; fn = np.cross(P_[I_[:, 1]] - P_[I_[:, 0]], P_[I_[:, 2]] - P_[I_[:, 0]])
        rad = P_[I_].mean(1) - cc; rad[:, 1] = 0
        bad = (fn * rad).sum(1) * sgn < 0; I_[bad] = I_[bad][:, ::-1]; p['nrm'] = G.vertex_normals(P_, I_); add(p, True)
    tq = G.flat_quads('concrete', np.array(Q_t), 1.0); G.orient_up(tq); tq['nrm'] = np.tile([0, 1.0, 0], (len(tq['pos']), 1)); add(tq, True)
    add(G.extrude('water_basin', ring_i + c, yb - 0.2, yr - 0.12, uvscale=2.0, top=True))
    # Bronzeformen (abgerundete Kegelstuempfe, unterschiedlich hoch)
    rng = np.random.default_rng(3)
    for k, (dx, dz, h, r) in enumerate(((-1.2, 0.3, 1.6, 0.42), (-0.2, -0.5, 2.3, 0.50), (0.9, 0.4, 1.3, 0.38), (1.7, -0.2, 0.9, 0.30))):
        p = c + [dx, dz]
        add(G.cylinder('bronze', (p[0], yr - 0.3, p[1]), r, r * 0.55, h, 14, cap=True), True)
        add(G.cylinder('bronze', (p[0], yr - 0.3 + h, p[1]), r * 0.55, 0.05, r * 0.6, 14, cap=False))
    log('Ballhofbrunnen at', np.round(c, 1))
    return {'name': 'Ballhofbrunnen', 'p': c.round(2).tolist(), 'y': round(yb, 2)}

# ================================================================= Leine: Abdeckung + Gelaender auf den Kaimauern
def leine():
    CL = L['classes']; RIV = CL.get('river')
    if RIV is None or RIV.is_empty: return 0
    rj = json.load(open(CACHE + 'river.json'))['rivers']
    wl = float(np.median([r['wl'] for r in rj])) if rj else -7.4
    BR = unary_union([LineString(b['line']).buffer(float(b.get('width') or 8) / 2 + 1.5, cap_style=2) for b in L.get('bridges', []) if len(b['line']) >= 2]) if L.get('bridges') else Polygon()
    WALKC = {'ped', 'sidewalk', 'path', 'alt_setts', 'clinker', 'cycle', 'island', 'gravel', 'edge', 'seam'}
    n_rail = 0; tot_len = 0.0
    for poly in getattr(RIV, 'geoms', [RIV]):
        for ring in [poly.exterior] + list(poly.interiors):
            Ls = LineString(ring.coords)
            S = np.arange(0, Ls.length, 1.5)
            if len(S) < 3: continue
            P = np.array([Ls.interpolate(s).coords[0] for s in S])
            Tg = np.gradient(P, axis=0); Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-12
            N = np.c_[-Tg[:, 1], Tg[:, 0]]
            inside = shapely.contains_xy(poly, *(P + N * 0.3).T)
            N[inside] *= -1                                         # N zeigt vom Wasser weg (ans Ufer)
            B = P + N * 0.45
            hb, cb = SURF(B[:, 0], B[:, 1])
            ok = np.array([c_ in WALKC for c_ in cb]) & (hb - wl > 1.0) & ~shapely.contains_xy(BR, P[:, 0], P[:, 1]) if not BR.is_empty else \
                 np.array([c_ in WALKC for c_ in cb]) & (hb - wl > 1.0)
            # Laeufe zusammenhaengender Punkte
            runs = []; cur = []
            for k in range(len(P)):
                if ok[k]: cur.append(k)
                elif cur: runs.append(cur); cur = []
            if cur: runs.append(cur)
            for r in runs:
                if len(r) < 3: continue
                Pr, Nr, Hr = P[r], N[r], hb[r]
                A = np.c_[Pr[:, 0] + Nr[:, 0] * 0.02, Hr + 0.10, Pr[:, 1] + Nr[:, 1] * 0.02]
                Bq = np.c_[Pr[:, 0] + Nr[:, 0] * 0.42, Hr + 0.10, Pr[:, 1] + Nr[:, 1] * 0.42]
                add(_strip('sandstone', A, Bq, look=A.mean(0) + [0, 5, 0]), True)               # Abdeckplatte oben
                A2 = np.c_[Pr[:, 0] + Nr[:, 0] * 0.02, Hr - 0.25, Pr[:, 1] + Nr[:, 1] * 0.02]
                add(_strip('sandstone', A2, A, look=A.mean(0) - np.r_[Nr.mean(0)[0], 0, Nr.mean(0)[1]] * 5))   # Stirn zum Wasser
                A3 = np.c_[Pr[:, 0] + Nr[:, 0] * 0.42, Hr - 0.05, Pr[:, 1] + Nr[:, 1] * 0.42]
                add(_strip('sandstone', A3, Bq, look=Bq.mean(0) + np.r_[Nr.mean(0)[0], 0, Nr.mean(0)[1]] * 5))  # Innenkante
                rp = np.c_[Pr[:, 0] + Nr[:, 0] * 0.22, Hr + 0.10, Pr[:, 1] + Nr[:, 1] * 0.22]
                for q in rp[::2]: add(G2.cyl_between('iron_dark', q, q + [0, 1.0, 0], 0.022, 6))
                add(G.tube('iron_dark', rp + [0, 1.0, 0], 0.03, 6)); add(G.tube('iron_dark', rp + [0, 0.12, 0], 0.016, 5))
                for q in rp[1::2]: add(G2.cyl_between('iron_dark', q + [0, 0.12, 0], q + [0, 1.0, 0], 0.010, 4, caps=False))
                coll.append(_strip('collision', rp, rp + [0, 1.1, 0], look=rp.mean(0) + np.r_[Nr.mean(0)[0], 0, Nr.mean(0)[1]] * 3))
                n_rail += 1; tot_len += 1.5 * (len(r) - 1)
    log('Leine railings', n_rail, 'length %.0f m' % tot_len, 'water level %.2f' % wl)
    return round(tot_len)

# ================================================================= Leine-Bruecken (Fahrbahnplatte ueber dem Wasser)
def leine_bridges():
    """Die Leine (Klasse river, Wasserspiegel) laeuft im Gelaende unter den Bruecken durch -> Fahrbahnplatte von Ufer zu Ufer:
    Oberseite an den Enden exakt auf Uferhoehe (Proben 4 cm landseitig), Asphalt bzw. Pflaster, Brueckenkappen mit
    Eisengelaender, Seitenflaechen Sandstein, Untersicht Beton.  Kollision + begehbar."""
    import triangle as tr_
    RIV = L['classes'].get('river')
    if RIV is None or RIV.is_empty: return []
    out_info = []
    shapely.prepare(RIV)
    def bank_y(p, P):
        inward = np.array(P.centroid.coords[0]) - p; inward /= np.linalg.norm(inward) + 1e-9
        q = p - inward * 0.04
        if RIV.contains(Point(q)): return None
        return float(SURF([q[0]], [q[1]])[0][0])
    for b in L.get('bridges', []):
        Lb = LineString(b['line']); w = float(b.get('width') or 6.0); hw = b['tags'].get('highway')
        road = (hw in ('primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'service', 'living_street')) or bool(b['tags'].get('railway'))
        half = w / 2 + (2.2 if road else 0.35)
        c = np.asarray(Lb.coords); d = c[-1] - c[0]
        if np.linalg.norm(d) < 2: continue
        d = d / np.linalg.norm(d); fr = Frame(c[0], d); Ltot = float(np.dot(c[-1] - c[0], d))
        S_ = np.arange(0.0, Ltot + 0.01, 0.5)
        Pax = np.array([fr.xz(x, 0) for x in S_])
        inr = shapely.contains_xy(RIV.buffer(0.3), Pax[:, 0], Pax[:, 1])
        if not inr.any() or inr.all(): continue
        ter = SURF(Pax[:, 0], Pax[:, 1])[0]
        # Endhoehen: erster/letzter Achspunkt ausserhalb des Flusses (OSM-Brueckenenden liegen manchmal schon ueber dem Wasser)
        ka = int(np.nonzero(~inr)[0][0]); kb = int(np.nonzero(~inr)[0][-1])
        sA0, sB0, yA0, yB0 = S_[ka], S_[kb], float(ter[ka]), float(ter[kb])
        def ydeck(s_): return yA0 + (yB0 - yA0) * np.clip((np.asarray(s_, float) - sA0) / max(sB0 - sA0, 1e-6), 0, 1)
        inr = shapely.contains_xy(RIV, Pax[:, 0], Pax[:, 1])
        i0, i1 = np.nonzero(inr)[0][[0, -1]]
        ia = i0
        while ia > 0 and not (ter[ia] >= ydeck(S_[ia]) - 0.10 and not inr[ia]): ia -= 1
        ib = i1
        while ib < len(S_) - 1 and not (ter[ib] >= ydeck(S_[ib]) - 0.10 and not inr[ib]): ib += 1
        sa_, sb_ = S_[ia], S_[ib]
        if True:
            if True:
                P = Polygon([fr.xz(sa_, -half), fr.xz(sb_, -half), fr.xz(sb_, half), fr.xz(sa_, half)])
                ring = np.array(shapely.segmentize(P.exterior, 1.0).coords)[:-1]
                n = len(ring)
                r = tr_.triangulate({'vertices': ring, 'segments': np.c_[np.arange(n), (np.arange(n) + 1) % n]}, 'pq25a2.0YQ')
                V, T = r['vertices'], r['triangles']
                cen = V[T].mean(1); T = T[shapely.contains_xy(P, cen[:, 0], cen[:, 1])]
                fr = Frame(c[0], d)
                ST_ = np.array([fr.st(*p) for p in V]); sv, tv = ST_[:, 0], ST_[:, 1]
                Y = ydeck(sv)
                for i in np.nonzero((np.abs(sv - sa_) < 0.01) | (np.abs(sv - sb_) < 0.01))[0]:
                    q = V[i] + d * (0.04 if sv[i] > (sa_ + sb_) / 2 else -0.04)
                    if not RIV.contains(Point(q)): Y[i] = float(SURF([q[0]], [q[1]])[0][0])
                yA, yB = float(ydeck(sa_)), float(ydeck(sb_))
                pos = np.c_[V[:, 0], Y, V[:, 1]]
                mat = 'asphalt' if road else 'pavers'
                top = G.part(mat, pos, T, None, V / (2.0 if road else 3.0)); G.orient_up(top); top['nrm'] = G.vertex_normals(top['pos'], top['idx'])
                add(top, True, True)
                E_ = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
                ek, cnt = np.unique(E_, axis=0, return_counts=True)
                Q = []
                for u_, w_ in ek[cnt == 1]:
                    a, bb = V[u_], V[w_]; m_ = (a + bb) / 2; dd = bb - a; ln = np.linalg.norm(dd)
                    if ln < 1e-6: continue
                    nrm = np.array([-dd[1], dd[0]]) / ln
                    if P.contains(Point(m_ + nrm * 0.05)): nrm = -nrm
                    q = np.array([[a[0], Y[u_] - 0.9, a[1]], [bb[0], Y[w_] - 0.9, bb[1]], [bb[0], Y[w_], bb[1]], [a[0], Y[u_], a[1]]])
                    if np.dot(np.cross(q[1] - q[0], q[3] - q[0])[[0, 2]], nrm) < 0: q = q[[1, 0, 3, 2]]
                    Q.append(q)
                if Q: add(G.flat_quads('sandstone', np.array(Q), 1.0), True)
                und = G.part('concrete', pos - [0, 0.9, 0], T, None, V / 2.0)
                fn = np.cross(und['pos'][T[:, 1]] - und['pos'][T[:, 0]], und['pos'][T[:, 2]] - und['pos'][T[:, 0]])
                und['idx'] = np.where((fn[:, 1] > 0)[:, None], T[:, ::-1], T); und['nrm'] = G.vertex_normals(und['pos'], und['idx'])
                add(und)
                for sg in (-1, 1):
                    edge = LineString([fr.xz(-500, sg * (half - 0.15)), fr.xz(500, sg * (half - 0.15))]).intersection(P.buffer(-0.02))
                    for e in getattr(edge, 'geoms', [edge]):
                        if e.is_empty or e.geom_type != 'LineString' or e.length < 1: continue
                        ce = np.array(shapely.segmentize(e, 1.5).coords)
                        ys = ydeck([fr.st(*p)[0] for p in ce])
                        # Kappe (Sandstein, 0.32 m breit, 0.20 m hoch) als Band entlang der Kante
                        In = np.c_[[fr.xz(fr.st(*p)[0], sg * (half - 0.32)) for p in ce]]
                        Out = np.c_[[fr.xz(fr.st(*p)[0], sg * half) for p in ce]]
                        Ti = np.c_[In[:, 0], ys + 0.20, In[:, 1]]; To = np.c_[Out[:, 0], ys + 0.20, Out[:, 1]]
                        add(_strip('sandstone', Ti, To, look=Ti.mean(0) + [0, 5, 0]), True)
                        Bi = np.c_[In[:, 0], ys - 0.02, In[:, 1]]
                        add(_strip('sandstone', Bi, Ti, look=Ti.mean(0) - np.r_[fr.vec(0, sg)] * 5), True)
                        Bo = np.c_[Out[:, 0], ys - 0.02, Out[:, 1]]
                        add(_strip('sandstone', Bo, To, look=To.mean(0) + np.r_[fr.vec(0, sg)] * 5), True)
                        A3 = np.c_[ce[:, 0], ys + 0.20, ce[:, 1]]
                        for q in A3[::2]: add(G2.cyl_between('iron_dark', q, q + [0, 0.95, 0], 0.025, 6))
                        add(G.tube('iron_dark', A3 + [0, 0.95, 0], 0.035, 6)); add(G.tube('iron_dark', A3 + [0, 0.15, 0], 0.016, 5))
                        for q in A3[1::2]: add(G2.cyl_between('iron_dark', q + [0, 0.15, 0], q + [0, 0.95, 0], 0.012, 4, caps=False))
                        coll.append(_strip('collision', A3, A3 + [0, 1.0, 0], look=np.r_[P.centroid.coords[0][0], A3[:, 1].mean(), P.centroid.coords[0][1]]))
                out_info.append({'id': b['id'], 'name': b['tags'].get('name'), 'width': round(2 * half, 1), 'y': [round(float(yA), 2), round(float(yB), 2)]})
                log('bridge', b['id'], b['tags'].get('name'), 'deck %.0f m2' % P.area, 'y %.2f..%.2f' % (yA, yB))
    return out_info

# ================================================================= Tunnelportale (Leine im Durchlass, OSM waterway tunnel=yes)
def leine_portals():
    """Wo die Leine in einen Durchlass geht (OSM waterway=river, tunnel=yes; z. B. unter dem Friederikenplatz), endet die
    Wasserflaeche an einer Uferlinie quer zum Fluss. Dort: Sandstein-Stirnmauer vom Grund (0.6 m unter Wasser) bis zur
    Gelaendeoberkante mit dunkler Bogenoeffnung (Durchlass) und Abdeckplatte. Schliesst die Wasserkante sichtbar ab."""
    RIV = L['classes'].get('river')
    if RIV is None or RIV.is_empty: return []
    rj = json.load(open(CACHE + 'river.json'))['rivers']
    out = []
    tun = [np.array(w['pts'], float) for w in osm['ways'] if w['tags'].get('waterway') in ('river', 'canal') and w['tags'].get('tunnel') in ('yes', 'culvert') and len(w['pts']) >= 2]
    ends = []
    for P in tun:
        for e, nb in ((P[0], P[1]), (P[-1], P[-2])): ends.append((e, (e - nb) / (np.linalg.norm(e - nb) + 1e-9)))
    for poly in getattr(RIV, 'geoms', [RIV]):
        wl = min([r['wl'] for r in rj if Polygon(r['poly']).buffer(1).intersects(poly)] or [-7.4])
        ring = np.asarray(poly.exterior.coords)
        for e, wd in ends:
            if poly.exterior.distance(Point(e)) > 12: continue
            # Randkanten nahe am Tunnelende, quer zum Fluss
            sel = []
            for i in range(len(ring) - 1):
                a, b = ring[i], ring[i + 1]; L_ = np.linalg.norm(b - a)
                if L_ < 0.3: continue
                m = (a + b) / 2
                if np.linalg.norm(m - e) > 30: continue
                if abs(np.dot((b - a) / L_, wd)) > 0.55: continue
                sel.append((a, b))
            if not sel: continue
            for a, b in sel:
                d = (b - a) / np.linalg.norm(b - a); n = np.array([-d[1], d[0]])
                if poly.contains(Point((a + b) / 2 + n * 0.5)) is False: n = -n          # n zeigt ins Wasser
                L_ = np.linalg.norm(b - a)
                k = max(1, int(math.ceil(L_ / 2.0))); S = np.linspace(0, 1, k + 1)
                Pq = a[None] + (b - a)[None] * S[:, None]
                land = Pq - n * 0.6
                hl, _ = SURF(land[:, 0], land[:, 1])
                ytop = np.maximum(hl, wl + 2.2) + 0.15
                y0 = wl - 0.6
                # Stirnflaeche zum Wasser (Sandstein), Oeffnung: dunkle Nische 0.35 m tief bis wl + 1.8 (Bogenstich angedeutet)
                F = np.c_[Pq[:, 0] + n[0] * 0.02, np.full(len(Pq), y0), Pq[:, 1] + n[1] * 0.02]
                yo = wl + np.clip(ytop - wl - 0.6, 0.9, 1.8)
                mid_ = (S > 0.06) & (S < 0.94)
                Fo = F.copy(); Fo[:, 1] = np.where(mid_, yo, y0)
                Ft = F.copy(); Ft[:, 1] = ytop
                add(_strip('sandstone', Fo, Ft, look=F.mean(0) + np.r_[n[0], 0, n[1]] * 5 + [0, 1.0, 0]), True)
                # Nische (dunkel): Rueckwand + Laibung oben
                ii = np.nonzero(mid_)[0]
                if len(ii) >= 2:
                    R0 = np.c_[Pq[ii, 0] - n[0] * 0.35, np.full(len(ii), y0), Pq[ii, 1] - n[1] * 0.35]
                    R1 = R0.copy(); R1[:, 1] = yo[ii]
                    add(_strip('interior_dark', R0, R1, look=R0.mean(0) + np.r_[n[0], 0, n[1]] * 5))
                    T0 = np.c_[Pq[ii, 0] + n[0] * 0.02, yo[ii], Pq[ii, 1] + n[1] * 0.02]; T1 = R1.copy()
                    add(_strip('sandstone', T1, T0, look=T0.mean(0) - [0, 3.0, 0]))
                    for j, sg in ((ii[0], 1), (ii[-1], -1)):
                        q0 = np.array([Pq[j, 0] + n[0] * 0.02, y0, Pq[j, 1] + n[1] * 0.02]); q1 = np.array([Pq[j, 0] - n[0] * 0.35, y0, Pq[j, 1] - n[1] * 0.35])
                        qq = G2.quad('sandstone', q0, q1, q1 + [0, yo[j] - y0, 0], q0 + [0, yo[j] - y0, 0])
                        if qq is not None and np.dot(qq['nrm'][0][[0, 2]], d * sg) < 0: qq['idx'] = qq['idx'][:, ::-1]; qq['nrm'] *= -1
                        add(qq)
                    W0 = np.c_[Pq[ii, 0] + n[0] * 0.03, np.full(len(ii), wl), Pq[ii, 1] + n[1] * 0.03]
                    W1 = np.c_[Pq[ii, 0] - n[0] * 0.36, np.full(len(ii), wl), Pq[ii, 1] - n[1] * 0.36]
                    add(_strip('river', W0, W1, look=W0.mean(0) + [0, 5, 0]))
                # Abdeckplatte + Eisengelaender
                A = np.c_[Pq[:, 0] + n[0] * 0.08, ytop + 0.10, Pq[:, 1] + n[1] * 0.08]; Bq = np.c_[Pq[:, 0] - n[0] * 0.45, ytop + 0.10, Pq[:, 1] - n[1] * 0.45]
                add(_strip('sandstone', A, Bq, look=A.mean(0) + [0, 5, 0]), True)
                A2 = A.copy(); A2[:, 1] = ytop - 0.02
                add(_strip('sandstone', A2, A, look=A.mean(0) + np.r_[n[0], 0, n[1]] * 5))
                B2 = Bq.copy(); B2[:, 1] = np.minimum(hl, ytop) - 0.35
                add(_strip('sandstone', B2, Bq, look=Bq.mean(0) - np.r_[n[0], 0, n[1]] * 5))
                rp = np.c_[Pq[:, 0] - n[0] * 0.18, ytop + 0.10, Pq[:, 1] - n[1] * 0.18]
                for q in rp: add(G2.cyl_between('iron_dark', q, q + [0, 1.0, 0], 0.022, 6))
                add(G.tube('iron_dark', rp + [0, 1.0, 0], 0.03, 6)); add(G.tube('iron_dark', rp + [0, 0.12, 0], 0.016, 5))
                coll.append(_strip('collision', rp, rp + [0, 1.1, 0], look=rp.mean(0) + np.r_[n[0], 0, n[1]] * 3))
                out.append({'a': a.round(2).tolist(), 'b': b.round(2).tolist(), 'y_water': round(wl, 2), 'y_top': round(float(ytop.max()), 2)})
    log('Leine tunnel portals', len(out))
    return out

INFO = {'fountains': [holzmarkt_brunnen(), ballhof_brunnen()], 'leine_railing_m': leine(), 'leine_bridges': leine_bridges(), 'leine_portals': leine_portals()}
coll = [c for c in coll if c is not None]
pickle.dump({'parts': parts, 'coll': coll, 'walk': walk, 'mats': {}, 'info_extra': INFO}, open(CACHE + 'altstadt.pkl', 'wb'))
log('done: tris', G.tri_count(parts), 'collision', G.tri_count(coll))

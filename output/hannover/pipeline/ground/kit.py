"""Gemeinsame Bausteine (Ausbau 2) fuer ubahn.py und promenade.py: Schilderkaesten, Gelaender, Waende aus Rim-Kanten,
U-Mast, Treppen (Granit, Trittkantenstreifen), Rolltreppen, Deckenstreifen."""
import math
import numpy as np
from shapely.geometry import LineString
import geom as G
import geom2 as G2
from geom2 import Frame
import ubahn_spec as US

def stair_geom(D, top_landing=0.6, rise_max=0.165, tread=0.30, landing=1.5, land_over=18):
    """Treppen-Kennwerte wie ubahn_spec.lane_geom (s_riser je Stufe ab dem oberen Podest)"""
    n = int(math.ceil(D / rise_max - 1e-6)); rise = D / n; land = n > land_over; n1 = n // 2 if land else n
    s = [top_landing + (k - 1) * tread + (landing - tread if (land and k > n1) else 0.0) for k in range(1, n + 1)]
    return {'type': 'stair', 'n': n, 'rise': rise, 'tread': tread, 'n1': n1, 'landing': landing if land else 0.0, 's_riser': s, 's_end': s[-1]}

def esc_geom(D, flat=1.7):
    L_inc = D / math.tan(math.radians(30))
    return {'type': 'esc', 's_inc0': flat, 's_inc1': flat + L_inc, 's_end': flat + L_inc + flat}

def panel_box(fr, mat_front, mat_side, s0, s1, t0, t1, y0, y1, front, top=True, bottom=True):
    """Quader; Seiten in front (str oder Menge aus 's0','s1','t0','t1') bekommen mat_front mit UV 0..1 (Text lesbar)."""
    fronts = {front} if isinstance(front, str) else set(front)
    out = []
    sc, tc = (s0 + s1) / 2, (t0 + t1) / 2
    faces = {'s0': ((s0, t0), (s0, t1), (s0 - 1, tc)), 's1': ((s1, t0), (s1, t1), (s1 + 1, tc)),
             't0': ((s0, t0), (s1, t0), (sc, t0 - 1)), 't1': ((s0, t1), (s1, t1), (sc, t1 + 1))}
    for k, (a, b, far) in faces.items():
        isf = k in fronts
        q = fr.vquad(mat_front if isf else mat_side, a, b, y0, y1, far)
        if q is None: continue
        if isf:
            P = q['pos']; n = q['nrm'][0]
            right = np.cross([0, 1.0, 0], n); right /= np.linalg.norm(right) + 1e-12
            u = P @ right; u = (u - u.min()) / (u.max() - u.min() + 1e-9)
            v = 1 - (P[:, 1] - y0) / (y1 - y0)
            q['uv'] = np.c_[u, v]
        out.append(q)
    if top: out.append(fr.hquad(mat_side, s0, s1, t0, t1, y1, up=True))
    if bottom: out.append(fr.hquad(mat_side, s0, s1, t0, t1, y0, up=False))
    return out


def railing_line(pts, posts_every=1.4, h=1.0, r_top=0.022, knee=True, mat='stainless', base=None):
    """Gelaender auf 3D-Polyline (Fusspunkte). Pfosten, Handlauf, Knieleiste."""
    P = np.asarray(pts, float)
    Ls = LineString(P[:, [0, 2]]); n = max(2, int(math.ceil(Ls.length / posts_every)) + 1)
    out = []
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1))]
    S = np.linspace(0, cum[-1], n)
    Q = np.c_[np.interp(S, cum, P[:, 0]), np.interp(S, cum, P[:, 1]), np.interp(S, cum, P[:, 2])]
    for q in Q: out.append(G2.cyl_between(mat, q, q + [0, h, 0], 0.02, 6))
    out.append(G.tube(mat, Q + [0, h, 0], r_top, 8))
    if knee: out.append(G.tube(mat, Q + [0, h * 0.45, 0], 0.012, 6))
    return out


def wall_from_edges(edges, bottom_fn, mat, center, uvscale=2.4):
    Q = []
    for pu, pw, hu, hw, c in edges:
        bu = bottom_fn(*pu); bw = bottom_fn(*pw)
        if hu - bu < 0.005 and hw - bw < 0.005: continue
        q = np.array([[pu[0], bu, pu[1]], [pw[0], bw, pw[1]], [pw[0], hw, pw[1]], [pu[0], hu, pu[1]]])
        n = np.cross(q[1] - q[0], q[3] - q[0])
        if np.dot(n[[0, 2]], np.asarray(center) - q.mean(0)[[0, 2]]) < 0: q = q[[1, 0, 3, 2]]
        Q.append(q)
    return G.flat_quads(mat, np.array(Q), uvscale) if Q else None


def u_pylon(fr, s, t, yb, sign_mat, double=True):
    """Mast + beleuchteter U-Wuerfel (4 Seiten) + Stationsname"""
    out = []
    out.append(fr.box('paint_dark', s - 0.07, s + 0.07, t - 0.07, t + 0.07, yb - 0.05, yb + 2.75))
    a = 0.31; y0 = yb + 2.75; y1 = y0 + 2 * a
    out += panel_box(fr, 'sign_u2', 'paint_dark', s - a, s + a, t - a, t + a, y0, y1, ('s0', 's1', 't0', 't1'))
    # Namensschild unter dem Wuerfel (beidseitig, quer zur Gehrichtung)
    out += panel_box(fr, sign_mat, 'paint_dark', s - 0.10, s + 0.10, t - 0.45, t + 0.45, yb + 2.40, yb + 2.62, ('s0', 's1'))
    return out


def build_stair(fr, g, y_top, y_floor, t0, t1, sty):
    out = []; col = []
    n, rise, tread, sr = g['n'], g['rise'], g['tread'], g['s_riser']
    # oberes Aufmerksamkeitsfeld wird mit dem Podest (Rim) gebaut; hier Stufen
    for k in range(1, n + 1):
        s_k = sr[k - 1]
        y_hi = y_top - (k - 1) * rise; y_lo = y_top - k * rise
        # Setzstufe: sichtbar von unten (Blick bergauf) -> Normale +d (zum unteren Podest; QA 2026-11-10)
        q = fr.vquad('granite_step', (s_k, t0), (s_k, t1), y_lo, y_hi, (s_k + 1, (t0 + t1) / 2), uvscale=1.0)
        out.append(q)
        if k < n:
            s_n = sr[k]
            # Tritt mit dunklem Trittkantenstreifen (5 cm) an der Vorderkante
            out.append(fr.hquad('granite_step', s_k, s_n - 0.05, t0, t1, y_lo, up=True, uvscale=1.0))
            out.append(fr.hquad('nosing', s_n - 0.05, s_n, t0, t1, y_lo, up=True, uvscale=1.0))
    # unteres Aufmerksamkeitsfeld (0.6 m) folgt im Boden
    return out


def stair_nose_line(g, y_top):
    sr = g['s_riser']; n = g['n']; rise = g['rise']
    pts = [(sr[0] - 0.3, y_top)] + [(sr[k - 1], y_top - (k - 1) * rise) for k in range(1, n + 1)] + [(sr[-1] + 0.3, y_top - n * rise)]
    return np.array(pts)


ESC_PLATE = 0.85; ESC_COMB = 0.15
def esc_profile(s, y_top, y_floor, s_i0, s_i1):
    return np.where(s <= s_i0, y_top, np.where(s >= s_i1, y_floor, y_top - (s - s_i0) * (y_top - y_floor) / (s_i1 - s_i0)))

def build_escalator(fr, g, y_top, y_floor, t0, t1, edge_wall_l, edge_wall_r):
    out = []; col = []
    D = y_top - y_floor
    s_i0, s_i1 = g['s_inc0'], g['s_inc1']; s_end = g['s_end']
    tc = (g['t0'] + g['t1']) / 2; hb = 0.5
    prof = lambda s: esc_profile(np.asarray(s, float), y_top, y_floor, s_i0, s_i1)
    # Podestplatten + Kammplatten oben/unten (volle Spurbreite)
    out.append(fr.hquad('esc_plate', 0.0, ESC_PLATE, t0, t1, y_top, uvscale=1.0))
    out.append(fr.hquad('stainless', ESC_PLATE, ESC_PLATE + ESC_COMB, t0, t1, y_top, uvscale=1.0))
    out.append(fr.hquad('stainless', s_end - ESC_PLATE - ESC_COMB, s_end - ESC_PLATE, t0, t1, y_floor, uvscale=1.0))
    out.append(fr.hquad('esc_plate', s_end - ESC_PLATE, s_end, t0, t1, y_floor, uvscale=1.0))
    # Stufenband: flache Stufen + geneigte Stufen
    sa = ESC_PLATE + ESC_COMB; sb = s_end - ESC_PLATE - ESC_COMB
    n_e = max(4, int(round(D / 0.21))); rise = D / n_e; dep = (s_i1 - s_i0) / n_e
    def tread(s0, s1, y):
        q = fr.hquad('esc_step', s0, s1, tc - hb, tc + hb, y, up=True, world_uv=False)
        # UV: u quer 0..1, v entlang 0..1
        P = q['pos']; st = np.array([fr.st(p[0], p[2]) for p in P])
        q['uv'] = np.c_[(st[:, 1] - (tc - hb)) / (2 * hb), (s1 - st[:, 0]) / max(s1 - s0, 1e-6)]
        return q
    s = sa
    while s < s_i0 - 1e-6:
        e = min(s + 0.40, s_i0); out.append(tread(s, e, y_top)); s = e
    for k in range(1, n_e + 1):
        a = s_i0 + (k - 1) * dep; b = s_i0 + k * dep
        out.append(fr.vquad('steel_dark', (a, tc - hb), (a, tc + hb), y_top - k * rise, y_top - (k - 1) * rise, (a + 1, tc)))   # Normale +d (von unten sichtbar)
        out.append(tread(a, b, y_top - k * rise))
    s = s_i1
    while s < sb - 1e-6:
        e = min(s + 0.40, sb); out.append(tread(s, e, y_floor)); s = e
    # Kollision: Rampe ueber das Stufenband
    # Seitenteile entlang des Profils
    S = np.r_[np.linspace(0.45, s_i0, 3), np.linspace(s_i0, s_i1, 12)[1:], np.linspace(s_i1, s_end - 0.45, 3)[1:]]
    Y = prof(S)
    for side in (-1, 1):
        tsk = tc + side * (hb + 0.005); tgl = tc + side * 0.60; tedge = (t1 if side > 0 else t0)
        # Sockelblech (Skirt) zum Stufenband hin
        A = np.array([fr.P(s_, tsk, y - 0.05) for s_, y in zip(S, Y)]); B = np.array([fr.P(s_, tsk, y + 0.25) for s_, y in zip(S, Y)])
        out.append(_strip('stainless', A, B, look=fr.P(S.mean(), tc, Y.mean())))
        # innere Abdeckung (schraeg) Skirt -> Balustrade
        A = np.array([fr.P(s_, tsk, y + 0.25) for s_, y in zip(S, Y)]); B = np.array([fr.P(s_, tgl, y + 0.33) for s_, y in zip(S, Y)])
        out.append(_strip('stainless', A, B, look=fr.P(S.mean(), tc, Y.mean() + 2)))
        # Glasbalustrade
        S2 = S[(S > 0.75) & (S < s_end - 0.75)]; Y2 = prof(S2)
        A = np.array([fr.P(s_, tgl, y + 0.33) for s_, y in zip(S2, Y2)]); B = np.array([fr.P(s_, tgl, y + 0.95) for s_, y in zip(S2, Y2)])
        out.append(_strip('glass', A, B, look=fr.P(S.mean(), tc, Y.mean())))
        # Handlauf (schwarz): Halbkreis-Umlenkung (r = 0.45 m) an beiden Enden, dazwischen parallel zum Stufenprofil
        RN = 0.45; s_a = 0.80; s_b = s_end - 0.80
        hp = [fr.P(s_a - RN * math.sin(a), tgl, y_top + 0.55 - RN * math.cos(a)) for a in np.linspace(0, math.pi, 9)]
        Sm = S2[(S2 > s_a + 0.05) & (S2 < s_b - 0.05)]
        hp += [fr.P(s_, tgl, y + 1.0) for s_, y in zip(Sm, prof(Sm))]
        hp += [fr.P(s_b + RN * math.sin(a), tgl, y_floor + 0.55 + RN * math.cos(a)) for a in np.linspace(0, math.pi, 9)]
        out.append(G.tube('rubber', np.array(hp), 0.035, 8, caps=True))
        # Glas-Endstuecke (Viertelkreis-Form angenaehert) unter den Umlenkungen
        for sc_, yy, sg in ((s_a, y_top, -1), (s_b, y_floor, 1)):
            arc = [fr.P(sc_ + sg * RN * math.sin(a), tgl, yy + 0.55 + RN * math.cos(a) * 0.9) for a in np.linspace(0, math.pi / 2, 5)]
            base = [fr.P(sc_ + sg * RN * math.sin(a), tgl, yy + 0.33) for a in np.linspace(0, math.pi / 2, 5)]
            out.append(_strip('glass', base, arc, look=fr.P(sc_, tc, yy + 0.6)))
        # Balustradenfuss (Edelstahl-Sockelleiste, 12 cm) unter dem Glas-Endstueck; kein massiver Kopf (Glas bis zur Umlenkung)
        for sx_, yy, sg in ((s_a, y_top, -1), (s_b, y_floor, 1)):
            a0, a1 = sorted((sx_, sx_ + sg * RN))
            out.append(fr.box('stainless', a0, a1, tgl - 0.04, tgl + 0.04, yy, yy + 0.33))
        # aeussere Abdeckung Balustrade -> Spurkante
        A = np.array([fr.P(s_, tgl, y + 0.33) for s_, y in zip(S, Y)]); B = np.array([fr.P(s_, tedge, y + 0.25) for s_, y in zip(S, Y)])
        out.append(_strip('stainless', A, B, look=fr.P(S.mean(), tgl, Y.mean() + 2)))
        # Aussenverkleidung an der Spurkante (nicht an der Trogwand)
        if not (edge_wall_l if side < 0 else edge_wall_r):
            A = np.array([fr.P(s_, tedge, y_floor - 0.3) for s_ in S]); B = np.array([fr.P(s_, tedge, y + 0.25) for s_, y in zip(S, Y)])
            out.append(_strip('stainless', A, B, look=fr.P(S.mean(), tedge + side * 2, Y.mean())))
            col.append(_strip('collision', A, B, look=fr.P(S.mean(), tedge + side * 2, Y.mean())))
    # Kollision Stufen (glatte Rampe)
    Sx = np.array([sa, s_i0, s_i1, sb]); Yx = prof(Sx)
    A = np.array([fr.P(s_, tc - hb, y) for s_, y in zip(Sx, Yx)]); B = np.array([fr.P(s_, tc + hb, y) for s_, y in zip(Sx, Yx)])
    col.append(_strip('collision', A, B, look=fr.P(5, tc, 50)))
    return out, col

def _strip(mat, A, B, look):
    p = G2.tri_strip_quads(mat, A, B)
    P, I = p['pos'], p['idx']
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if (fn * (np.asarray(look) - P[I].mean(1))).sum() < 0: p['idx'] = I[:, ::-1]
    p['nrm'] = G.vertex_normals(P, p['idx'])
    return p


def _ceiling_strips(fr, s0, s1, t0, t1, y, sty):
    """Decke in Laengsstreifen: Lamellen + Lichtbaender (keine ueberlagerten Flaechen)"""
    out = []
    W = t1 - t0; n = max(1, int(round(W / 1.8)))
    edges = np.linspace(t0, t1, n + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (a + b) / 2
        out.append(fr.hquad(sty['ceil'], s0, s1, a, m - 0.06, y, up=False, uvscale=1.0))
        out.append(fr.hquad('lightline', s0, s1, m - 0.06, m + 0.06, y, up=False, uvscale=1.0))
        out.append(fr.hquad(sty['ceil'], s0, s1, m + 0.06, b, y, up=False, uvscale=1.0))
    return out


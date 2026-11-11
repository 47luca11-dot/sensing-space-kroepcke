"""Kroepcke-Center (Hannover) -- Grundriss aus LoD2 (Stadt Hannover / LGLN) + TrueDOP10 2025.

Plan-Koordinaten: x = Ost (= Spiel-x), y = Nord (= -Spiel-z), Meter relativ zur Kroepcke-Uhr.
Die LoD2-Grundflaeche wird in Geraden (Least-Squares) und tangentiale Boegen (Kreisfit) zerlegt und
als CCW-Pfad (Gebaeude links) beschrieben. Jeder Abschnitt traegt Fassadentyp und Traufhoehe.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, 'kc_lod2.json')))
Z0 = 56.16
GROUND = np.array([p[:2] for p in B['grounds'][0]['ext']])          # CW in LoD2
ROOFS = [(np.array([p[:2] for p in r['ext']]), [np.array([q[:2] for q in h]) for h in r['holes']], max(p[2] for p in r['ext']) - Z0) for r in B['roofs']]


def roof_at(z, tol=0.05):
    return [r for r in ROOFS if abs(r[2] - z) < tol]


def fit_line(pts):
    pts = np.asarray(pts, float); c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c); d = vt[0]
    return c, d / np.linalg.norm(d)


def fit_circle(pts):
    pts = np.asarray(pts, float); x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]; b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([cx, cy]), math.sqrt(c + cx * cx + cy * cy)


def line_through(p, d):
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    return np.asarray(p, float), d


def orient(line, heading):
    """Richtung der Geraden an gewuenschte Laufrichtung (CCW) anpassen."""
    p, d = line
    if np.dot(d, heading) < 0: d = -d
    return p, d


def left(d): return np.array([-d[1], d[0]])


def intersect(l1, l2):
    p1, d1 = l1; p2, d2 = l2
    A = np.array([d1, -d2]).T
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def offset(line, dist):
    """Gerade parallel nach links (ins Gebaeude bei CCW) verschieben."""
    p, d = line
    return p + left(d) * dist, d


def proj(line, q):
    p, d = line
    return float(np.dot(np.asarray(q) - p, d))


def fillet(l_in, l_out, r):
    """Tangentialer Bogen (Linkskurve, CCW) zwischen einlaufender und auslaufender Geraden."""
    c = intersect(offset(l_in, r), offset(l_out, r))
    p_in, d_in = l_in; p_out, d_out = l_out
    t1 = p_in + d_in * proj(l_in, c)
    t2 = p_out + d_out * proj(l_out, c)
    a0 = math.atan2(*(t1 - c)[::-1]); a1 = math.atan2(*(t2 - c)[::-1])
    while a1 <= a0: a1 += 2 * math.pi
    return dict(c=c, r=r, a0=a0, a1=a1, p0=t1, p1=t2)


def arc_from(c, r, a0, a1):
    while a1 <= a0: a1 += 2 * math.pi
    return dict(c=np.asarray(c, float), r=r, a0=a0, a1=a1,
                p0=np.asarray(c) + r * np.array([math.cos(a0), math.sin(a0)]),
                p1=np.asarray(c) + r * np.array([math.cos(a1), math.sin(a1)]))


def ang(c, p): return math.atan2(p[1] - c[1], p[0] - c[0])


def circle_line_hit(c, r, line, a_start):
    """erster Schnittpunkt Kreis/Gerade CCW nach a_start (Winkel)."""
    p, d = line
    f = p - c; b = 2 * np.dot(f, d); cc = np.dot(f, f) - r * r
    disc = b * b - 4 * cc
    disc = max(disc, 0.0)
    best = None
    for t in ((-b - math.sqrt(disc)) / 2, (-b + math.sqrt(disc)) / 2):
        q = p + d * t; a = ang(c, q)
        while a <= a_start + 1e-6: a += 2 * math.pi
        if best is None or a < best[0]: best = (a, q)
    return best


def P(x, y):
    """LoD2-Grundpunkt nachschlagen (Kontrolle, dass die Zahlen stimmen)."""
    d = np.hypot(GROUND[:, 0] - x, GROUND[:, 1] - y); i = int(np.argmin(d))
    if d[i] > 0.02: raise KeyError((x, y, d[i]))
    return GROUND[i].copy()


def build_plan():
    # ---------------- Geraden (Ausgleich) -----------------------------------------------------------
    L_NE = orient(fit_line([P(-20.55, -19.33), P(7.63, -69.76), P(10.73, -75.32), P(16.32, -85.31)]), (-0.488, 0.873))
    L_W = orient(fit_line([P(-29.72, -20.43), P(-32.79, -31.69), P(-35.45, -41.41), P(-45.10, -76.83), P(-47.70, -86.35), P(-58.10, -124.29)]), (-0.264, -0.965))
    L_S = orient(fit_line([P(13.71, -94.94), P(13.55, -95.04), P(-6.11, -106.03), P(-27.36, -117.84), P(-44.96, -127.72), P(-48.16, -129.50), P(-49.92, -130.50)]), (0.873, 0.488))
    L_S2 = (P(-25.54, -114.15), L_S[1].copy())          # Mittelteil, parallel zu L_S
    L_S2 = (L_S2[0] + L_S2[1] * 0.0, L_S2[1])
    # Ruecksprung-Tiefe Mittelteil (gemittelt aus beiden Punkten)
    dS2 = np.mean([np.dot(P(-16.00, -108.89) - L_S[0], left(L_S[1])), np.dot(P(-25.54, -114.15) - L_S[0], left(L_S[1]))])
    L_S2 = offset(L_S, dS2)
    dSb = np.mean([np.dot(P(-10.85, -104.68) - L_S[0], left(L_S[1])), np.dot(P(-13.61, -106.20) - L_S[0], left(L_S[1]))])
    L_Sb = offset(L_S, dSb)                             # Rueckwand Fuge Sued
    dWb = np.mean([np.dot(P(-42.48, -81.17) - L_W[0], left(L_W[1])), np.dot(P(-43.27, -84.06) - L_W[0], left(L_W[1]))])
    L_Wb = offset(L_W, dWb)                             # Rueckwand Fuge West (= Flucht Staffelgeschoss)

    segs = []
    def add(kind, geo, **kw):
        s = dict(kind=kind, **geo); s.update(kw); segs.append(s)

    def line_seg(p0, p1):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        d = p1 - p0; L = float(np.linalg.norm(d))
        return dict(type='line', p0=p0, p1=p1, t=d / L, L=L)

    def arc_seg(a):
        A = a['r'] * (a['a1'] - a['a0'])
        return dict(type='arc', c=a['c'], r=a['r'], a0=a['a0'], a1=a['a1'], p0=a['p0'], p1=a['p1'], L=A)

    # ---------------- Boegen --------------------------------------------------------------------------
    prow_pts = [P(-22.28, -16.25), P(-22.92, -15.45), P(-23.76, -14.86), P(-24.73, -14.54), P(-25.75, -14.51), P(-26.75, -14.77), P(-27.62, -15.29), P(-28.32, -16.05), P(-28.77, -16.97)]
    c, r = fit_circle(prow_pts)
    # Radius so waehlen, dass der Bogen tangential an beide Geraden passt (Mittel der Abstaende)
    r_prow = 0.5 * (abs(np.dot(c - L_NE[0], left(L_NE[1]))) + abs(np.dot(c - L_W[0], left(L_W[1]))))
    A_prow = fillet(L_NE, L_W, r_prow)
    e_pts = [P(18.01, -88.36), P(18.34, -89.25), P(18.43, -90.19), P(18.29, -91.12), P(17.92, -92.00), P(17.34, -92.75), P(16.59, -93.32)]
    c, r = fit_circle(e_pts)
    r_e = 0.5 * (abs(np.dot(c - L_S[0], left(L_S[1]))) + abs(np.dot(c - L_NE[0], left(L_NE[1]))))
    A_E = fillet(L_S, L_NE, r_e)
    sw_pts = [P(-53.02, -132.22), P(-53.58, -132.46), P(-54.36, -132.66), P(-55.16, -132.70), P(-55.96, -132.58), P(-56.71, -132.31), P(-57.41, -131.90), P(-58.01, -131.37), P(-58.50, -130.73), P(-58.85, -130.01), P(-59.06, -129.24), P(-59.12, -128.43), P(-59.02, -127.66)]
    c, r = fit_circle(sw_pts)
    r_sw = 0.5 * (abs(np.dot(c - L_W[0], left(L_W[1]))) + abs(np.dot(c - L_S[0], left(L_S[1]))))
    A_SW = fillet(L_W, L_S, r_sw)

    # Fuge West: Nordteil-Ecke (Viertelkreis von L_W in Richtung Osten) bis Rueckwand-Flucht, Radius = Fugentiefe
    rW = abs(dWb)
    tN = P(-45.10, -76.83); tN = L_W[0] + L_W[1] * proj(L_W, tN)
    cN = tN + left(L_W[1]) * rW
    A_WN = arc_from(cN, rW, ang(cN, tN), ang(cN, tN) + math.pi / 2)
    tS = P(-47.70, -86.35); tS = L_W[0] + L_W[1] * proj(L_W, tS)
    cS = tS + left(L_W[1]) * rW
    A_WS = arc_from(cS, rW, ang(cS, tS) - math.pi / 2, ang(cS, tS))
    # Fuge Sued: kleine Rundungen
    # Ostteil-Westende: Bogen von L_S (Richtung ONO laufend? nein: CCW laeuft L_S nach ONO) --
    # CCW-Reihenfolge Sued: SW-Spitze -> L_S (West-Fluegel) -> Stufe -> L_S2 (Mittelteil) -> Rundung -> Fuge -> Rundung -> L_S (Ost)
    st_pts = [P(-25.67, -116.89), P(-25.35, -116.67), P(-25.10, -116.37), P(-24.95, -116.01), P(-24.89, -115.63), P(-24.95, -115.24), P(-25.11, -114.89)]
    c_st, r_st = fit_circle(st_pts)
    # tangential an L_S anschliessen: Mittelpunkt auf Parallele im Abstand r
    r_st = abs(np.dot(c_st - L_S[0], left(L_S[1])))
    t_st = L_S[0] + L_S[1] * proj(L_S, c_st)
    hit = circle_line_hit(c_st, r_st, L_S2, ang(c_st, t_st))
    A_ST = arc_from(c_st, r_st, ang(c_st, t_st), hit[0])
    mw_pts = [P(-14.08, -107.77), P(-13.82, -107.60), P(-13.62, -107.37), P(-13.49, -107.09), P(-13.45, -106.79), P(-13.49, -106.48)]
    c_mw, r_mw = fit_circle(mw_pts)
    r_mw = abs(np.dot(c_mw - L_S2[0], left(L_S2[1])))
    t_mw = L_S2[0] + L_S2[1] * proj(L_S2, c_mw)
    hit = circle_line_hit(c_mw, r_mw, L_Sb, ang(c_mw, t_mw))
    A_MW = arc_from(c_mw, r_mw, ang(c_mw, t_mw), hit[0])
    eo_pts = [P(-9.65, -106.72), P(-9.36, -106.97), P(-9.00, -107.12), P(-8.62, -107.18), P(-8.23, -107.12), P(-7.88, -106.96)]
    c_eo, r_eo = fit_circle(eo_pts)
    r_eo = abs(np.dot(c_eo - L_S[0], left(L_S[1])))
    t_eo = L_S[0] + L_S[1] * proj(L_S, c_eo)
    # Bogen endet tangential an L_S (bei t_eo); Start = Schnitt mit Fugen-Seitenwand (senkrecht zu L_S)
    # Seitenwand-Gerade durch (-10.85,-104.68) -> (-9.87,-106.40)
    side = fit_line([P(-10.85, -104.68), P(-9.87, -106.40)])
    side = orient(side, -left(L_S[1]))             # laeuft nach aussen (Sued)
    # Startwinkel: Schnitt Kreis / Seitenwand, CCW-Bogen bis t_eo
    a_end = ang(c_eo, t_eo)
    best = None
    for a in np.linspace(a_end - math.pi, a_end, 2000):
        q = c_eo + r_eo * np.array([math.cos(a), math.sin(a)])
        dd = abs(np.dot(q - side[0], left(side[1])))
        if best is None or dd < best[0]: best = (dd, a)
    A_EO = arc_from(c_eo, r_eo, best[1], a_end)

    # ---------------- Pfad (CCW, Start Bug) -------------------------------------------------------------
    H_N, H_SW = 22.18, 22.22
    H_HI = 26.06
    add('prow', arc_seg(A_prow), top=H_N, part='N')
    add('std', line_seg(A_prow['p1'], A_WN['p0']), top=H_N, part='N', name='W_nord')
    add('curve', arc_seg(A_WN), top=H_N, part='N')
    # Rueckwand Fuge West
    pK1 = A_WN['p1']; pK2 = A_WS['p0']
    add('joint', line_seg(pK1, pK2), top=H_HI, part='SW')
    add('curve', arc_seg(A_WS), top=H_SW, part='SW')
    add('std', line_seg(A_WS['p1'], A_SW['p0']), top=H_SW, part='SW', name='W_sued')
    add('curve', arc_seg(A_SW), top=H_SW, part='SW')
    add('std', line_seg(A_SW['p1'], A_ST['p0']), top=H_SW, part='SW', name='S_west')
    add('round', arc_seg(A_ST), top=H_SW, part='SW')
    pM0 = A_ST['p1']
    add('std', line_seg(pM0, A_MW['p0']), top=H_HI, part='M', name='S_mitte')
    add('round', arc_seg(A_MW), top=H_HI, part='M')
    add('joint', line_seg(A_MW['p1'], intersect(L_Sb, side)), top=H_N, part='N')
    add('joint', line_seg(intersect(L_Sb, side), A_EO['p0']), top=H_N, part='N')
    add('round', arc_seg(A_EO), top=H_N, part='N')
    add('std', line_seg(A_EO['p1'], A_E['p0']), top=H_N, part='N', name='S_ost')
    add('curve', arc_seg(A_E), top=H_N, part='N')
    add('std', line_seg(A_E['p1'], A_prow['p0']), top=H_N, part='N', name='NE')

    # Kontrolle: Pfad geschlossen
    for a, b in zip(segs, segs[1:] + segs[:1]):
        gap = np.linalg.norm(np.asarray(a['p1']) - np.asarray(b['p0']))
        if gap > 0.02: print('WARN gap', a['kind'], b['kind'], round(gap, 3))
    info = dict(r_prow=r_prow, r_e=r_e, r_sw=r_sw, rW=rW, r_st=r_st, r_mw=r_mw, r_eo=r_eo, dS2=dS2, dSb=dSb, dWb=dWb)
    return segs, info, dict(L_NE=L_NE, L_W=L_W, L_S=L_S, L_S2=L_S2, L_Sb=L_Sb, L_Wb=L_Wb)


def sample_path(segs, da=math.radians(3), dl=None):
    pts = []
    for s in segs:
        if s['type'] == 'line':
            pts.append(np.asarray(s['p0']))
        else:
            n = max(2, int(math.ceil((s['a1'] - s['a0']) / da)))
            for a in np.linspace(s['a0'], s['a1'], n + 1)[:-1]:
                pts.append(s['c'] + s['r'] * np.array([math.cos(a), math.sin(a)]))
    return np.array(pts)


if __name__ == '__main__':
    segs, info, lines = build_plan()
    print({k: round(float(v), 3) for k, v in info.items()})
    for s in segs:
        print(s['kind'], s['type'], 'L=%.2f' % s['L'], 'top', s['top'], s.get('name', ''), np.round(s['p0'], 2), np.round(s['p1'], 2))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    pts = sample_path(segs)
    fig, ax = plt.subplots(figsize=(12, 16))
    g = np.vstack([GROUND, GROUND[:1]]); ax.plot(g[:, 0], g[:, 1], 'r-', lw=0.8)
    p = np.vstack([pts, pts[:1]]); ax.plot(p[:, 0], p[:, 1], 'k-', lw=1.2)
    for s in segs:
        m = (np.asarray(s['p0']) + np.asarray(s['p1'])) / 2; ax.text(m[0], m[1], s['kind'], fontsize=7)
    ax.set_aspect('equal'); ax.grid(True)
    fig.savefig(os.path.join(os.environ.get('KC_TMP', HERE), 'plan_check.png'), dpi=90)

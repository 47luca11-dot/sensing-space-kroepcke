"""Leibnizhaus (Holzmarkt 5): rekonstruierte Renaissance-Sandsteinfassade (1652 / Rekonstruktion 1983) mit dreigeschossiger
Utlucht, Rundbogenportal, fuenfstufigem Volutengiebel mit Obelisken und Kriegerfigur; steiles Dach (dunkle Biberschwanzziegel).
Hoehen aus Ref.-Foto leibnizhaus/01 (Pixelmasse, Perspektivkorrektur) + LoD2 (Traufe/First). Gibt MB zurueck."""
import math
import numpy as np
from shapely.geometry import Polygon, box as sbox
import lh_common as C
import lh_geom as G
import lh_parts as LP

# ------------------------------------------------------------------ Rahmen: Fassade (u nach Norden, w nach Osten)
P_S = np.array([-438.78, 341.02]); P_N = np.array([-439.96, 329.99])
W = float(np.linalg.norm(P_N - P_S))                     # 11.09
_u = (P_N - P_S) / W
FR = G.Frame((P_S[0], 0.0, P_S[1]), (_u[0], 0.0, _u[1]))
Y0 = -1.65                                              # Fassadenfuss (Gelaende -1.64..-1.70)
UG = 5.31                                               # Giebelachse (Foto + LoD2-First)
C_PERSP = 0.00636


def Yp(orig_y):
    """Foto-Zeile (leibnizhaus/01, Originalpixel) -> Spiel-y (Fuss 1700 px, 64.5 px/m, projektive Korrektur)."""
    yp = (1700.0 - orig_y) / 64.5
    return Y0 + yp / (1.0 - C_PERSP * yp)


def Up(orig_x):
    return (orig_x - 278.0) / 64.5 * (W / (715.0 / 64.5))


# Geschossgrenzen (Oberkante Gesims)
L = dict(
    plinth=Yp(1672), eg_w0=Yp(1580), eg_w1=Yp(1470), eg_c=Yp(1445),
    og1_w0=Yp(1395), og1_w1=Yp(1290), og1_c=Yp(1250),
    og2_w0=Yp(1210), og2_w1=Yp(1100), og2_c=Yp(1040),
    og3_w0=Yp(1020), og3_w1=Yp(965), og3_f0=Yp(960), og3_f1=Yp(895), og3_c=Yp(865),
    og4_w0=Yp(840), og4_w1=Yp(775), eave=Yp(758),
    st1_w0=Yp(720), st1_w1=Yp(655), st1_c=Yp(632),
    st2_w0=Yp(605), st2_w1=Yp(545), st2_c=Yp(530),
    st3_w0=Yp(480), st3_w1=Yp(425), st3_c=Yp(405),
    crown=Yp(282), statue=Yp(205),
)
CT = 0.31            # Hauptgesims-Hoehe
STAGES = [  # (u0, u1, y_unten, y_gesims_oben)
    (1.37, 9.25, L['eave'], L['st1_c']),
    (2.45, 8.15, L['st1_c'], L['st2_c']),
    (3.45, 7.15, L['st2_c'], L['st3_c']),
    (4.15, 6.45, L['st3_c'], L['st3_c'] + 0.95),
]
T_WALL = 0.45
ROOF_EAVE = 14.30
ROOF_RIDGE = 22.9
DEPTH = 19.5          # Hausfigur bis Rueckwand
HIP = 3.2


def oriel_dims():
    return dict(u0=5.22, u1=9.30, d=1.0)


# ------------------------------------------------------------------ Fassadenwand mit Oeffnungen
def window_list():
    """(u0, u1, y0, y1, kind) der Fenster in der Hauptwand (ohne Utlucht)."""
    Wn = []
    pairs_left = [(0.70, 1.45), (1.65, 2.40), (3.45, 4.20), (4.40, 5.12)]
    Wn += [(a, b, L['eg_w0'], L['eg_w1'], 'eg') for (a, b) in pairs_left[:2]]
    for lv in ('og1', 'og2'):
        Wn += [(a, b, L[lv + '_w0'], L[lv + '_w1'], lv) for (a, b) in pairs_left]
    Wn += [(a, b, L['og3_w0'], L['og3_w1'], 'og3') for (a, b) in [(0.70, 1.40), (1.65, 2.36), (3.50, 4.26), (4.42, 5.12), (8.0, 8.72)]]
    Wn += [(a, b, L['og4_w0'], L['og4_w1'], 'og4') for (a, b) in [(1.66, 2.38), (3.62, 4.40), (4.70, 5.50), (6.80, 7.60), (7.90, 8.72)]]
    Wn += [(a, b, L['st1_w0'], L['st1_w1'], 'st1') for (a, b) in [(3.70, 4.45), (4.78, 5.55), (6.88, 7.66)]]
    Wn += [(4.75, 5.55, L['st2_w0'], L['st2_w1'], 'st2'), (4.78, 5.55, L['st3_w0'], L['st3_w1'], 'st3')]
    return Wn


PORTAL = dict(u0=3.62, u1=5.02, y0=Y0 + 0.18, spring=Yp(1530))


def portal_ring(n=16):
    u0, u1, y0, ys = PORTAL['u0'], PORTAL['u1'], PORTAL['y0'], PORTAL['spring']
    r = (u1 - u0) / 2; cx = (u0 + u1) / 2
    a = np.linspace(0, math.pi, n + 1)
    return np.r_[[[u0, y0], [u1, y0]], np.c_[cx + r * np.cos(a), ys + r * np.sin(a)]]


def outline():
    E = L['eave'] + CT
    tops = [s[3] for s in STAGES]; bots = [E] + tops[:-1]
    right = [(W, E)]
    for i, s in enumerate(STAGES): right += [(s[1], bots[i]), (s[1], tops[i])]
    left = []
    for i in range(len(STAGES) - 1, -1, -1): left += [(STAGES[i][0], tops[i]), (STAGES[i][0], bots[i])]
    pts = [(0, -2.1), (W, -2.1)] + right + left + [(0, E)]
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6: out.append(p)
    return np.array(out, float)


def build_wall(mb):
    holes = [G.rect(a, y0, b, y1) for (a, b, y0, y1, k) in window_list()]
    holes.append(portal_ring())
    poly = outline()
    G.extrude(mb, FR, 'ashlar', poly, -T_WALL, 0.0, holes=holes, uvs=0.5, side_mat='sand_fine')
    return poly


def build_windows(mb):
    for (a, b, y0, y1, k) in window_list():
        tr = (y1 - y0) > 1.25
        LP.window_cross(mb, FR, a, b, y0, y1, reveal=0.24, transom=tr, mullion=True, glass='leaded', stone='sand_fine')
    # Reveal-Ersatz: extrude erzeugt die Laibung ueber die ganze Wandstaerke; Fenster sitzen bei -0.15


# ------------------------------------------------------------------ Gesimse, Friese, Pilaster
def build_cornices(mb, rng):
    # Sockel
    LP.cornice(mb, FR, 'sand_plinth', -0.0, W, Y0 - 0.35, 'plinth', 1.0)
    # Geschossgesimse (volle Breite, Rueckkroepfung an den Ecken)
    for key, prof, sc in (('eg_c', 'cornice_s', 1.35), ('og1_c', 'cornice_s', 1.35), ('og2_c', 'cornice_s', 1.35), ('og3_c', 'cornice_s', 1.35)):
        y = L[key] - 0.17 * sc
        LP.cornice_path(mb, FR, 'sand_fine', [(0.0, -0.6), (0.0, 0.0), (W, 0.0), (W, -2.9)], y, prof, sc)
    LP.cornice_path(mb, FR, 'sand_fine', [(0.0, -0.6), (0.0, 0.0), (W, 0.0), (W, -2.9)], L['eave'], 'cornice', 1.15)
    # Sohlbankgesimse unter den Fenstern (Bandgesimse)
    for key in ('og1_w0', 'og2_w0', 'og3_w0', 'og4_w0'):
        LP.cornice_path(mb, FR, 'sand_fine', [(0.0, -0.6), (0.0, 0.0), (W, 0.0), (W, -2.9)], L[key] - 0.16, 'band', 1.0)
    # Giebelstufen
    for (u0, u1, y0, y1) in STAGES[:3]:
        LP.cornice_path(mb, FR, 'sand_fine', [(u0, -T_WALL), (u0, 0.0), (u1, 0.0), (u1, -T_WALL)], y1 - 0.255, 'cornice_s', 1.5)
        LP.cornice_path(mb, FR, 'sand_fine', [(u0, -T_WALL), (u0, 0.0), (u1, 0.0), (u1, -T_WALL)], y0 + 0.35 + 0.0, 'band', 0.8)


def frieze(mb, u0, u1, y0, y1, kind, seed, res=36, z=0.0, depth=0.035):
    w = u1 - u0; h = y1 - y0
    if w < 0.1 or h < 0.05: return
    if kind == 'scroll': H = LP.rel_scroll(w, h, res, depth, seed)
    elif kind == 'medal': H = LP.rel_medallions(w, h, res, depth, seed)
    elif kind == 'strap': H = LP.rel_strap(w, h, res, depth, seed)
    elif kind == 'cand': H = LP.rel_candelabra(w, h, res, depth, seed)
    elif kind == 'cart': H = LP.rel_cartouche(w, h, res, depth, seed)
    elif kind == 'cherub': H = LP.rel_cherub(w, h, res, depth, seed)
    elif kind == 'scene': H = LP.rel_scene(w, h, res, depth, seed)
    else: raise ValueError(kind)
    G.relief(mb, FR, 'sand_relief', H, u0, y0, w, h, z0=z, uvs=1.0)


def build_ornament(mb, rng):
    s = 0
    # Friese unter den Geschossgesimsen (Hauptwand links + rechts der Utlucht)
    for key, h in (('eg_c', 0.2), ('og1_c', 0.32), ('og2_c', 0.30)):
        y1 = L[key] - 0.18; y0 = y1 - h
        frieze(mb, 0.08, 5.12, y0, y1, 'scroll' if key != 'og1_c' else 'medal', s); s += 1
        frieze(mb, 9.36, W - 0.08, y0, y1, 'scroll', s); s += 1
    frieze(mb, 0.08, W - 0.08, L['og3_f0'] + 0.06, L['og3_f1'] - 0.04, 'medal', s, res=30); s += 1
    for (u0, u1, y0, y1), wt in zip(STAGES[:3], (L['st1_w1'], L['st2_w1'], L['st3_w1'])):
        if y1 - 0.44 > wt + 0.03:
            frieze(mb, u0 + 0.1, u1 - 0.1, y1 - 0.44, y1 - 0.24, 'scroll', s, res=24); s += 1
    # Brustungsfelder unter den Fenstern (Beschlagwerk)
    for (a, b, y0, y1, k) in window_list():
        if k in ('og1', 'og2'):
            frieze(mb, a, b, y0 - 0.62, y0 - 0.2, 'strap', s, res=34); s += 1
    # Pilaster mit Kandelaberreliefs: Ecken + zwischen den Fensterpaaren; kleine Saeulen zwischen Paarfenstern
    lv = [('eg', Y0 + 0.5, L['eg_c'] - 0.48), ('og1', L['eg_c'] + 0.05, L['og1_c'] - 0.52), ('og2', L['og1_c'] + 0.05, L['og2_c'] - 0.5),
          ('og3', L['og2_c'] + 0.05, L['og3_f0']), ('og4', L['og3_c'] + 0.05, L['eave'] - 0.38)]
    for (name, y0, y1) in lv:
        for (pu0, pu1) in ((0.08, 0.55), (2.52, 3.32) if name != 'eg' else (None, None), (9.45, 10.15), (10.55, W - 0.08)):
            if pu0 is None: continue
            if name == 'eg' and pu0 > 9: pass
            G.box(mb, FR, 'sand_fine', pu0, pu1, y0, y1, 0.0, 0.035, bevel=0.01)
            frieze(mb, pu0 + 0.03, pu1 - 0.03, y0 + 0.06, y1 - 0.06, 'cand', s, res=32, z=0.035, depth=0.03); s += 1
        if name in ('og1', 'og2'):
            for uc in (1.55, 4.30):
                LP.column(mb, FR, 'sand_fine', uc, 0.07, L[name + '_w0'] - 0.05, L[name + '_w1'] + 0.08, 0.065, segs=10)
        if name == 'og3':
            for uc in (1.52, 4.34):
                LP.column(mb, FR, 'sand_fine', uc, 0.06, L['og3_w0'] - 0.03, L['og3_w1'] + 0.05, 0.055, segs=10)
    # Giebel: Pilasterstreifen mit Kandelaber zwischen Fenstern
    for (u0, u1, y0, y1), (wy0, wy1) in zip(STAGES[:3], ((L['st1_w0'], L['st1_w1']), (L['st2_w0'], L['st2_w1']), (L['st3_w0'], L['st3_w1']))):
        for pu in (u0 + 0.25, u1 - 0.45):
            G.box(mb, FR, 'sand_fine', pu, pu + 0.2, wy0 - 0.1, wy1 + 0.1, 0.0, 0.03, bevel=0.008)
            frieze(mb, pu + 0.02, pu + 0.18, wy0 - 0.05, wy1 + 0.05, 'cand', s, res=40, z=0.03, depth=0.025); s += 1
    # Rechteckige Relief-Kartuschen am rechten Streifen (Bild: Schild mit Glocke) auf 4. OG
    frieze(mb, 10.1, 10.85, L['og4_w0'] - 0.1, L['og4_w1'] + 0.15, 'cart', s, res=40, depth=0.05); s += 1


# ------------------------------------------------------------------ Portal
def build_portal(mb, rng):
    u0, u1, y0, ys = PORTAL['u0'], PORTAL['u1'], PORTAL['y0'], PORTAL['spring']
    r = (u1 - u0) / 2; cx = (u0 + u1) / 2
    # Stufen vor dem Portal (2 Stufen, aus Bodenhoehe)
    G.box(mb, FR, 'sand_plinth', u0 - 0.35, u1 + 0.25, Y0 - 0.4, Y0 + 0.0, 0.0, 0.62, bevel=0.015)
    G.box(mb, FR, 'sand_plinth', u0 - 0.2, u1 + 0.15, Y0 - 0.4, y0, -T_WALL, 0.32, bevel=0.015)
    # Keilsteine (Rustika-Bogen)
    n = 13
    for k in range(n):
        a0 = math.pi * k / n; a1 = math.pi * (k + 1) / n
        am = (a0 + a1) / 2
        ri, ro = r, r + 0.34 + (0.08 if k == n // 2 else 0.0)
        pts = np.array([[cx + ri * math.cos(a0), ys + ri * math.sin(a0)], [cx + ro * math.cos(a0), ys + ro * math.sin(a0)],
                        [cx + ro * math.cos(a1), ys + ro * math.sin(a1)], [cx + ri * math.cos(a1), ys + ri * math.sin(a1)]])
        pts = pts[::-1] if np.cross(pts[1] - pts[0], pts[2] - pts[0]) < 0 else pts
        G.extrude(mb, FR, 'sand_fine', pts, 0.0, 0.07 + (0.06 if k == n // 2 else 0.0), uvs=1.0)
    # Kaempfer + Gewaende (rustiziert)
    for (a, b) in ((u0 - 0.3, u0), (u1, u1 + 0.3)):
        yy = y0
        k = 0
        while yy < ys - 0.05:
            h = min(0.36, ys - yy)
            G.box(mb, FR, 'sand_fine', a, b, yy + 0.01, yy + h - 0.01, 0.0, 0.06 if k % 2 == 0 else 0.04, bevel=0.012)
            yy += h; k += 1
        G.box(mb, FR, 'sand_fine', a - 0.04, b + 0.04, ys - 0.06, ys + 0.06, 0.0, 0.1, bevel=0.01)
    # Schlussstein mit Maske
    H = LP.rel_cherub(0.42, 0.27, 50, 0.06, 3)
    G.relief(mb, FR, 'sand_relief', H, cx - 0.21, ys + r + 0.02, 0.42, 0.27, z0=0.13)
    # Saeule links auf Postament
    G.box(mb, FR, 'sand_fine', 2.88, 3.36, Y0 - 0.1, Y0 + 0.9, 0.0, 0.42, bevel=0.02)
    LP.column(mb, FR, 'sand_fine', 3.12, 0.22, Y0 + 0.9, L['eg_c'] - 0.45, 0.15, segs=14, capital='corinth')
    G.box(mb, FR, 'sand_fine', 2.85, 3.39, L['eg_c'] - 0.45, L['eg_c'] - 0.3, 0.0, 0.42, bevel=0.01)
    # Tuer: zwei Fluegel mit Kassetten (dunkles Holz), Oberlicht mit schmiedeeisernem Gitter
    zd = -0.3
    P = np.array([[u0, y0, zd], [u1, y0, zd], [u1, ys, zd], [u0, ys, zd]])
    mb.add_local(FR, 'wood_door', P, np.tile([0, 0, 1.0], (4, 1)), P[:, :2], np.array([[0, 1, 2], [0, 2, 3]]))
    for (a, b) in ((u0 + 0.06, cx - 0.03), (cx + 0.03, u1 - 0.06)):
        for (c, d) in ((y0 + 0.15, y0 + 1.0), (y0 + 1.15, ys - 0.12)):
            G.box(mb, FR, 'wood_door', a + 0.06, b - 0.06, c, d, zd, zd + 0.04, bevel=0.015)
            G.box(mb, FR, 'wood_door', a + 0.12, b - 0.12, c + 0.08, d - 0.08, zd + 0.04, zd + 0.06, bevel=0.01)
    G.box(mb, FR, 'wood_door', cx - 0.03, cx + 0.03, y0, ys, zd, zd + 0.05)
    # Lunette: Glas + Gitter
    a = np.linspace(0, math.pi, 17)
    semi = np.c_[cx + r * np.cos(a), ys + r * np.sin(a)]
    poly = np.r_[semi, [[u0, ys]]]
    tri = G.Frame(FR.p([0, 0, zd - 0.02]), FR.u, FR.v)
    G.face_with_holes(mb, tri, 'glass', semi, z=0.0)
    for k in range(1, 8):
        aa = math.pi * k / 8
        G.tube(mb, FR, 'iron', np.array([[cx, ys + 0.05, zd + 0.03], [cx + r * 0.95 * math.cos(aa), ys + r * 0.95 * math.sin(aa), zd + 0.03]]), 0.012, 5)
    for rr in (0.28, 0.5, 0.68):
        t = np.linspace(0.05, math.pi - 0.05, 24)
        G.tube(mb, FR, 'iron', np.c_[cx + rr * np.cos(t), ys + rr * np.sin(t), np.full(24, zd + 0.035)], 0.012, 5)
    for k in range(6):
        aa = math.pi * (k + 0.5) / 6
        LP_spiral_iron(mb, cx + 0.6 * r * math.cos(aa), ys + 0.6 * r * math.sin(aa), zd + 0.04, 0.09, aa)
    G.box(mb, FR, 'iron', u0, u1, ys - 0.03, ys + 0.02, zd, zd + 0.05)


def LP_spiral_iron(mb, cx, cy, z, r0, a0):
    t = np.linspace(0, 1, 24)
    rad = r0 * (1 - 0.8 * t); a = a0 + t * 1.6 * math.tau
    G.tube(mb, FR, 'iron', np.c_[cx + rad * np.cos(a), cy + rad * np.sin(a), np.full(24, z)], 0.009, 4)


# ------------------------------------------------------------------ Utlucht (Erker)
def build_oriel(mb, rng):
    od = oriel_dims(); u0, u1, d = od['u0'], od['u1'], od['d']
    ytop = L['og2_c']
    yb = Y0 - 0.3
    # Fensteroeffnungen: Vorderseite 3, Seiten je 1
    piers = 0.30
    wfr = (u1 - u0 - 4 * piers) / 3
    levels = [('eg', L['eg_w0'], L['eg_w1']), ('og1', Yp(1388), Yp(1272)), ('og2', Yp(1185), Yp(1058))]
    front_holes = []; side_holes = []
    for (nm, a, b) in levels:
        for k in range(3):
            x0 = u0 + piers + k * (wfr + piers)
            front_holes.append((x0, x0 + wfr, a, b))
        side_holes.append((0.28, d - 0.22, a, b))
    # Front (Rahmen: u wie Fassade, Ebene w = d)
    FF = G.Frame(FR.p([0, 0, d]), FR.u, FR.v)
    G.face_with_holes(mb, FF, 'sand_fine', G.rect(u0, yb, u1, ytop), [G.rect(h[0], h[2], h[1], h[3]) for h in front_holes], uvs=1.0)
    for (a, b, c, e) in front_holes:
        G.reveal(mb, FF, 'sand_fine', G.rect(a, c, b, e), 0.0, -0.22)
        LP.window_cross(mb, FF, a, b, c, e, reveal=0.22, transom=True, frame=True, sill=False)
    # Seiten: Rahmen mit u = Tiefe
    for side, uu in (('S', u0), ('N', u1)):
        SF = side_frame(side, u0, u1, d)
        holes = [G.rect(a, c, b, e) for (a, b, c, e) in side_holes]
        G.face_with_holes(mb, SF, 'sand_fine', G.rect(0, yb, d, ytop), holes)
        for (a, b, c, e) in side_holes:
            G.reveal(mb, SF, 'sand_fine', G.rect(a, c, b, e), 0.0, -0.2)
            LP.window_cross(mb, SF, a, b, c, e, reveal=0.2, transom=True, mullion=False, frame=True, sill=False)
        # Ecksaeulen / Pilaster je Geschoss an der Seite (hinten an der Wand)
    # Deckel (Oberseite unter der Bekroenung)
    G.box(mb, FR, 'sand_fine', u0, u1, ytop - 0.02, ytop, 0.0, d, skip=('b',))
    # Saeulen vor den Pfeilern, Gebaelke, Reliefbaender
    col_u = [u0 + piers / 2 + k * (wfr + piers) for k in range(4)]
    for li, (nm, a, b) in enumerate(levels):
        ytopc = levels[li + 1][1] - 0.62 if li < 2 else ytop - 0.55
        for cu in col_u:
            G.box(mb, FR, 'sand_fine', cu - piers / 2, cu + piers / 2, a - 0.12, a, d, d + 0.18, bevel=0.012)      # Postament-Deckplatte
            LP.column(mb, FR, 'sand_fine', cu, d + 0.11, a, b + 0.06, 0.095 if li == 0 else 0.085, segs=12)
        # Gebaelk: Architrav + Fries mit Engelskoepfen + Gesims
        ya = b + 0.06
        LP.cornice_path(mb, FR, 'sand_fine', [(u0 - 0.0, 0.0), (u0, d), (u1, d), (u1, 0.0)], ya, 'architrave', 1.15)
        fy0 = ya + 0.17; fy1 = ytopc - 0.02
        if fy1 - fy0 > 0.15:
            # Fries (Ranken) mit Engelskoepfen ueber den Saeulen
            for k in range(3):
                x0 = col_u[k] + 0.12; x1 = col_u[k + 1] - 0.12
                frieze_oriel(mb, x0, x1, fy0, fy1, d, 'scroll', 100 + li * 10 + k)
            for cu in col_u:
                H = LP.rel_cherub(0.42, fy1 - fy0, 44, 0.07, 7 + li)
                G.relief(mb, FR_oriel(d), 'sand_relief', H, cu - 0.21, fy0, 0.42, fy1 - fy0, z0=0.0)
        LP.cornice_path(mb, FR, 'sand_fine', [(u0 - 0.0, 0.0), (u0, d), (u1, d), (u1, 0.0)], ytopc, 'cornice_s', 1.05)
        # Bruestungs-Figurenreliefs der naechsten Etage (zwischen Gesims und Fenstern)
        if li < 2:
            by0 = ytopc + 0.18; by1 = levels[li + 1][1] - 0.12
            for k in range(3):
                x0 = col_u[k] + 0.13; x1 = col_u[k + 1] - 0.13
                H = LP.rel_scene(x1 - x0, by1 - by0, 44, 0.08, 200 + li * 10 + k)
                G.relief(mb, FR_oriel(d), 'sand_relief', H, x0, by0, x1 - x0, by1 - by0, z0=0.0)
            for cu in col_u:
                G.box(mb, FR, 'sand_fine', cu - 0.14, cu + 0.14, by0, by1, d, d + 0.06, bevel=0.01)
                H = LP.rel_scene(0.26, by1 - by0, 60, 0.07, li * 7 + int(cu * 10), n=1, animal=False, frame=False)
                G.relief(mb, FR_oriel(d + 0.06), 'sand_relief', H, cu - 0.13, by0, 0.26, by1 - by0, z0=0.0)
    # Sockelzone EG: Reliefplatten (Samson & Loewe usw.) zwischen Hermenpilastern
    by0 = Y0 + 0.28; by1 = L['eg_w0'] - 0.18
    G.box(mb, FR, 'sand_plinth', u0 - 0.06, u1 + 0.06, Y0 - 0.35, Y0 + 0.28, 0.0, d + 0.12, bevel=0.02)
    for k in range(3):
        x0 = col_u[k] + 0.16; x1 = col_u[k + 1] - 0.16
        H = LP.rel_scene(x1 - x0, by1 - by0, 46, 0.09, 300 + k, animal=(k == 2))
        G.relief(mb, FR_oriel(d), 'sand_relief', H, x0, by0, x1 - x0, by1 - by0, z0=0.0)
    for cu in col_u:
        G.box(mb, FR, 'sand_fine', cu - 0.15, cu + 0.15, by0, by1, d, d + 0.08, bevel=0.012)
        H = LP.rel_scene(0.28, by1 - by0 - 0.04, 60, 0.08, int(cu * 13), n=1, animal=False, frame=False)
        G.relief(mb, FR_oriel(d + 0.08), 'sand_relief', H, cu - 0.14, by0 + 0.02, 0.28, by1 - by0 - 0.04, z0=0.0)
    LP.cornice_path(mb, FR, 'sand_fine', [(u0, 0.0), (u0, d), (u1, d), (u1, 0.0)], by1, 'sill', 1.3)
    # Seiten-Reliefs Sockel
    for side in ('S', 'N'):
        SF = side_frame(side, u0, u1, d)
        H = LP.rel_scene(d - 0.3, by1 - by0, 40, 0.07, 400 + (side == 'N'), n=1)
        G.relief(mb, SF, 'sand_relief', H, 0.15, by0, d - 0.3, by1 - by0, z0=0.0)
    # Bekroenung (Aufsatz) ueber der Utlucht: Sockel, Mittelaedikula mit Figurenrelief, Seitenfiguren, Voluten, Obelisken
    yc = ytop
    G.box(mb, FR, 'sand_fine', u0 + 0.1, u1 - 0.1, yc, yc + 0.35, 0.15, d - 0.1, bevel=0.015)
    cxm = (u0 + u1) / 2
    G.box(mb, FR, 'sand_fine', cxm - 0.95, cxm + 0.95, yc + 0.35, yc + 1.55, 0.25, d - 0.25, bevel=0.015)
    H = LP.rel_scene(1.7, 1.05, 46, 0.1, 501, n=3, frame=True)
    G.relief(mb, FR, 'sand_relief', H, cxm - 0.85, yc + 0.42, 1.7, 1.05, z0=d - 0.25)
    LP.cornice_path(mb, FR, 'sand_fine', [(cxm - 0.95, 0.25), (cxm - 0.95, d - 0.25), (cxm + 0.95, d - 0.25), (cxm + 0.95, 0.25)], yc + 1.55, 'cornice_s', 1.0)
    # Giebelchen mit Rollwerk + Spitze
    for sg in (-1, 1):
        LP.volute(mb, FR, 'sand_fine', cxm + sg * 0.25, cxm + sg * 0.95, yc + 1.72, 0.75, thick=0.2, w0=d - 0.3)
        LP.volute(mb, FR, 'sand_fine', cxm + sg * 0.95, cxm + sg * 1.75, yc + 0.35, 0.95, thick=0.2, w0=d - 0.3)
        LP.statue(mb, FR, 'sand_relief', cxm + sg * 1.45, d - 0.4, yc + 0.35, 0.95, seed=600 + sg, spear=False)
        LP.obelisk(mb, FR, 'sand_fine', u0 + 0.25 if sg < 0 else u1 - 0.25, d - 0.25, yc + 0.35, 1.1, 0.22)
    G.box(mb, FR, 'sand_fine', cxm - 0.25, cxm + 0.25, yc + 1.72, yc + 2.35, 0.4, d - 0.35, bevel=0.02)
    LP.statue(mb, FR, 'sand_relief', cxm, d - 0.5, yc + 2.35, 0.85, seed=777, spear=False)
    return col_u


def side_frame(side, u0, u1, d):
    """Seitenflaeche der Utlucht: lokal x 0..d (Sued: von der Wand nach vorn; Nord: von vorn zur Wand), z nach aussen."""
    if side == 'S':
        return G.Frame(FR.p([u0, 0, 0]), FR.w, FR.v)
    return G.Frame(FR.p([u1, 0, d]), -FR.w, FR.v)


def FR_oriel(d):
    return G.Frame(FR.p([0, 0, d]), FR.u, FR.v)


def frieze_oriel(mb, x0, x1, y0, y1, d, kind, seed):
    H = LP.rel_scroll(x1 - x0, y1 - y0, 40, 0.04, seed)
    G.relief(mb, FR_oriel(d), 'sand_relief', H, x0, y0, x1 - x0, y1 - y0, z0=0.0)


# ------------------------------------------------------------------ Giebel: Voluten, Obelisken, Krone, Figur
def build_gable(mb, rng):
    prev = (0.0, W)
    for i, (u0, u1, y0, y1) in enumerate(STAGES):
        if i == 0: y0 = L['eave'] + CT
        Hh = (y1 - y0) + (0.0 if i < 3 else 0.2)
        LP.volute(mb, FR, 'sand_fine', u0, prev[0] + 0.32, y0, Hh - 0.05, thick=0.24, w0=0.0)
        LP.volute(mb, FR, 'sand_fine', u1, prev[1] - 0.32, y0, Hh - 0.05, thick=0.24, w0=0.0)
        # Obelisken an den Stufenecken (aussen auf dem Gesims)
        hob = 1.35 if i < 3 else 1.0
        LP.obelisk(mb, FR, 'sand_fine', prev[0] + 0.17, -0.13, y0, hob, 0.26)
        LP.obelisk(mb, FR, 'sand_fine', prev[1] - 0.17, -0.13, y0, hob, 0.26)
        prev = (u0, u1)
    # Krone: Volutenpaar (Rollwerkgiebel) mit Mittelplatte + Figur
    u0, u1, y0, y1 = STAGES[-1]
    cx = (u0 + u1) / 2
    yt = L['crown']
    LP.cornice_path(mb, FR, 'sand_fine', [(u0, -T_WALL), (u0, 0.0), (u1, 0.0), (u1, -T_WALL)], y1 - 0.14, 'cornice_s', 0.85)
    G.box(mb, FR, 'sand_fine', cx - 0.4, cx + 0.4, y1, yt - 0.15, -0.38, 0.06, bevel=0.02)
    H = LP.rel_cartouche(0.66, yt - 0.15 - y1 - 0.2, 44, 0.05, 9)
    G.relief(mb, FR, 'sand_relief', H, cx - 0.33, y1 + 0.1, 0.66, yt - 0.15 - y1 - 0.2, z0=0.06)
    for sg in (-1, 1):
        LP.volute(mb, FR, 'sand_fine', cx + sg * 0.4, cx + sg * 1.12, y1, yt - y1 - 0.1, thick=0.22, w0=0.0)
    G.box(mb, FR, 'sand_fine', cx - 0.3, cx + 0.3, yt - 0.15, yt + 0.05, -0.3, 0.12, bevel=0.015)
    # Kriegerfigur mit Lanze (Bild: Figur mit Wetterfahne)
    LP.statue(mb, FR, 'sand_relief', cx, -0.08, yt + 0.05, L['statue'] - yt - 0.05, seed=1, spear=False)
    # Wetterfahne (Eisen) ueber der Figur
    yv = L['statue']
    G.cyl(mb, FR, 'iron', (cx - 0.12, yt + 0.5, -0.05), (cx - 0.12, yv + 0.85, -0.05), 0.016, 0.016, 6)
    flag = np.array([[cx - 0.12, yv + 0.45], [cx + 0.45, yv + 0.5], [cx + 0.35, yv + 0.62], [cx + 0.45, yv + 0.72], [cx - 0.12, yv + 0.7]])
    G.extrude(mb, FR, 'iron', flag, -0.06, -0.04)
    G.tube(mb, FR, 'iron', np.array([[cx - 0.42, yv + 0.3, -0.05], [cx + 0.18, yv + 0.3, -0.05]]), 0.008, 4)
    G.sphere(mb, FR, 'gold', (cx - 0.12, yv + 0.88, -0.05), 0.04, 8, 5)


# ------------------------------------------------------------------ Dach + Baukoerper
def roof_slope(mb, fr, mat, L_slope, width, rows_h=1 / 6.0, step=0.018, w0=0.0, uv_off=0.0):
    """Ziegelflaeche mit echten Reihen-Stufen: lokale x = quer (0..width), y = hangaufwaerts (0..L_slope), z = Normal.
    Jede Reihe: geneigte Ziegelflaeche + kleine Stossflaeche (Ziegelende)."""
    n = max(1, int(L_slope / rows_h))
    rh = L_slope / n
    P = []; N = []; U = []; I = []; off = 0
    for k in range(n):
        y0 = k * rh; y1 = (k + 1) * rh
        # Ziegelflaeche: unten (y0) erhaben um step, oben (y1) auf 0
        q = np.array([[0, y0, step], [width, y0, step], [width, y1, 0.0], [0, y1, 0.0]])
        nn = G.nrm(np.cross(q[1] - q[0], q[3] - q[0]))
        P.append(q); N.append(np.tile(nn, (4, 1))); U.append(np.c_[q[:, 0], q[:, 1] + uv_off]); I.append(np.array([[0, 1, 2], [0, 2, 3]]) + off); off += 4
        # Stossflaeche (Ziegelunterkante) nach unten blickend
        q = np.array([[0, y0, 0.0], [width, y0, 0.0], [width, y0, step], [0, y0, step]])
        P.append(q); N.append(np.tile([0, -1.0, 0], (4, 1))); U.append(np.c_[q[:, 0], q[:, 2] + y0 + uv_off]); I.append(np.array([[0, 2, 1], [0, 3, 2]]) + off); off += 4
    P = np.concatenate(P); N = np.concatenate(N); U = np.concatenate(U); I = np.concatenate(I)
    I = G.orient(P, I, N[I[:, 0]])
    mb.add_local(fr, mat, P, N, U, I)


def slope_frame(p_low_a, p_low_b, p_high_a):
    """Rahmen einer Dachflaeche: Ursprung p_low_a, x entlang Traufe (a->b), y hangaufwaerts (->p_high_a)."""
    p0 = np.asarray(p_low_a, float); x = G.nrm(np.asarray(p_low_b) - p0); yv = np.asarray(p_high_a) - p0
    yv = yv - x * np.dot(yv, x); Ls = np.linalg.norm(yv); yv = yv / Ls
    return G.Frame(p0, x, yv, np.cross(x, yv)), Ls


def build_roof(mb):
    """Satteldach mit Walm hinten: First bei u=UG; Traufen u=0 (Sued, ohne Ueberstand, Brandwand) und u=W (Nord, 0.35 m)."""
    eS, eN, ri = ROOF_EAVE, ROOF_EAVE, ROOF_RIDGE
    wf = -T_WALL + 0.05; wr = -DEPTH; wh = -DEPTH + HIP
    uS = 0.0; uN = W + 0.35
    eN2 = eN - 0.35 * (ri - eN) / (W - UG)
    # Weltpunkte
    def P(u, y, w): return FR.p([u, y, w])
    # Suedflaeche (Blick von Sueden: Traufe von hinten nach vorne)
    fr, Ls = slope_frame(P(uS, eS, wr), P(uS, eS, wf), P(UG, ri, wr))
    # Breite entlang Traufe: von wf bis wr; Walm schneidet hinten -> Trapez: einfach Rechteck bis wh und Dreieck-Rest ueber Walm
    roof_slope(mb, fr, 'tile_dark', Ls, abs(wr - wf))
    fr, Ls = slope_frame(P(uN, eN2, wf), P(uN, eN2, wr), P(UG, ri, wf))
    roof_slope(mb, fr, 'tile_dark', Ls, abs(wr - wf))
    # Giebeldreieck vorne (hinter der Wand verborgen) und Walmflaeche hinten
    tri = np.array([P(uS, eS, wf), P(UG, ri, wf), P(uN, eN2, wf)])
    I = G.orient(tri, np.array([[0, 1, 2]]), FR.w)
    PP, NN, II = G.flat_tris(tri, I); mb.add(  'tile_dark', PP, NN, PP[:, [0, 2]], II)
    tri = np.array([P(uS, eS, wr), P(UG, ri, wr), P(uN, eN2, wr)])
    I = G.orient(tri, np.array([[0, 1, 2]]), -FR.w)
    PP, NN, II = G.flat_tris(tri, I); mb.add('plaster', PP, NN, PP[:, [0, 2]], II)
    # Untersicht Nordtraufe (Ueberstand) + Boden
    q = np.array([P(W - 0.01, eN - 0.01, wf), P(uN, eN2 - 0.01, wf), P(uN, eN2 - 0.01, wr), P(W - 0.01, eN - 0.01, wr)])
    I = G.orient(q, np.array([[0, 1, 2], [0, 2, 3]]), [0, -1, 0]); PP, NN, II = G.flat_tris(q, I); mb.add('wood_dark', PP, NN, PP[:, [0, 2]], II)
    # Firstziegel (Halbrund) + Rinne + Fallrohr Nord
    a = P(UG, ri + 0.02, wf); b = P(UG, ri + 0.02, wr)
    G.tube(mb, G.WORLD, 'tile_dark', np.array([a, b]), 0.11, 10)
    gut = np.array([P(uN + 0.08, eN2 - 0.06, wf - 0.05), P(uN + 0.08, eN2 - 0.06, wr + 0.05)])
    G.tube(mb, G.WORLD, 'zinc', gut, 0.075, 10)
    dp = np.array([P(uN + 0.1, eN2 - 0.1, -0.6), P(W + 0.12, eN2 - 0.6, -0.25), P(W + 0.12, Y0 + 0.15, -0.25)])
    G.tube(mb, G.WORLD, 'zinc', dp, 0.05, 8)
    # Schneefanggitter / Dachfenster (Sued) - 2 Gauben-Dachfenster
    return


def build_body(mb):
    """Seiten-/Rueckwaende des Haupthauses (bis Traufe) + Nebenbaukoerper aus LoD2-Rest."""
    yb = -2.2
    # Nordwand (sichtbar ueber Noltehaus und im 3-m-Streifen am Holzmarkt): Quader-Sandstein, Gesimse laufen um
    NF = G.Frame(FR.p([W, 0, -T_WALL]), -FR.w, FR.v)     # Blick von Norden: u nach Westen
    G.face_with_holes(mb, NF, 'ashlar', G.rect(0, yb, DEPTH - T_WALL, ROOF_EAVE), [], uvs=0.5)
    SF = G.Frame(FR.p([0, 0, -DEPTH]), FR.w, FR.v)        # Sued (Brandwand): u nach Osten
    G.face_with_holes(mb, SF, 'plaster', G.rect(0, yb, DEPTH - T_WALL, ROOF_EAVE), [], uvs=0.5)
    RF = G.Frame(FR.p([W, 0, -DEPTH]), -FR.u, FR.v)       # Rueckwand (Blick von Westen)
    G.face_with_holes(mb, RF, 'plaster', G.rect(0, yb, W, ROOF_EAVE), [], uvs=0.5)
    # Fenster in der Nordwand oberhalb des Noltehaus-Dachs (wenige)
    for k, dd in enumerate((3.6, 7.0, 10.4, 13.8)):
        for yy in (L['og3_w0'] + 0.3,):
            pass
    # Nebenbaukoerper: LoD2-Grundriss minus Haupthaus
    fp = Polygon(C.lod2_footprint(C.full_ids()['leibnizhaus'])).buffer(0)
    corners = [FR.p([0, 0, 0.3]), FR.p([W + 0.2, 0, 0.3]), FR.p([W + 0.2, 0, -DEPTH - 0.02]), FR.p([0, 0, -DEPTH - 0.02])]
    main = Polygon([(c[0], c[2]) for c in corners])
    rest = fp.difference(main)
    geoms = getattr(rest, 'geoms', [rest])
    for g in geoms:
        if g.area < 1.0: continue
        g = g.buffer(-0.02, join_style=2).simplify(0.05)
        if g.is_empty or g.geom_type != 'Polygon': continue
        ring = np.asarray(g.exterior.coords)[:-1]
        cen = np.asarray(g.centroid.coords[0])
        hgt = 18.4 if np.dot(cen - P_S, _u) < 0.5 else 13.2
        # Grundriss in Weltkoordinaten (x, z): Extrusion lokal: Rahmen mit u=x, v=-z?  -> eigener Rahmen: Polygon in (x, -z), extrudiert entlang +y
        poly = np.c_[ring[:, 0], -ring[:, 1]]
        if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0: poly = poly[::-1]
        TF = G.Frame((0, 0, 0), (1, 0, 0), (0, 0, -1))     # lokal x=x, y=-z, z=+y (oben)
        G.extrude(mb, TF, 'roof_membrane', poly, yb, hgt, uvs=0.5, side_mat='plaster')


def build(seed=5):
    rng = np.random.default_rng(seed)
    mb = G.MB()
    build_wall(mb)
    build_windows(mb)
    build_cornices(mb, rng)
    build_ornament(mb, rng)
    build_portal(mb, rng)
    build_oriel(mb, rng)
    build_gable(mb, rng)
    build_roof(mb)
    build_body(mb)
    return mb


if __name__ == '__main__':
    for k, v in L.items(): print('%-8s %.2f' % (k, v))
    mb = build()
    print('tris', mb.ntris(), {k: sum(len(i) for (_, _, _, i) in v) for k, v in mb.parts.items()})

"""Bauteil-Prototypen (lokale Frames) fuer das Ballhof-Ensemble: Fenster, Bogentueren, Knaggen, Balkenkoepfe,
Firstziegel, Rinnenhaken, Wandlaternen.  Lokaler Fassadenframe: a nach rechts, b nach oben, c nach aussen (Wandflaeche c=0)."""
import math
import numpy as np
from bh_mesh import MB, Frame, WORLD, unit

FR = 'Fensterrahmen'; GL = 'Fensterglas'


def window(w, h, rows=3, cas=2, transom=None, top_rows=1, dr=0.12, reveal='Putz_weiss', sill='wood', sill_mat=None,
           frame_mat=FR, bars_v=False, glass=GL, single_top=False):
    """Fenster-Prototyp: Oeffnung [0,w]x[0,h], Laibung bis c=-dr. rows: Scheiben uebereinander je Fluegel;
    transom: Hoehe des Kaempfers ueber Unterkante (None: kein Oberlicht)."""
    m = MB()
    W = WORLD
    # Laibung
    m.quad(reveal, [0, 0, 0], [0, 0, -dr], [0, h, -dr], [0, h, 0])
    m.quad(reveal, [w, 0, -dr], [w, 0, 0], [w, h, 0], [w, h, -dr])
    m.quad(reveal, [0, h, -dr], [w, h, -dr], [w, h, 0], [0, h, 0])
    if sill is None:
        m.quad(reveal, [0, 0, 0], [w, 0, 0], [w, 0, -dr], [0, 0, -dr])
    fw, fd = 0.055, 0.065
    c0, c1 = -dr, -dr + fd
    m.box(frame_mat, W, 0, fw, 0, h, c0, c1, skip=('-c', '-a'))
    m.box(frame_mat, W, w - fw, w, 0, h, c0, c1, skip=('-c', '+a'))
    m.box(frame_mat, W, fw, w - fw, h - fw, h, c0, c1, skip=('-c', '+b', '-a', '+a'))
    m.box(frame_mat, W, fw, w - fw, 0, fw * 1.2, c0, c1, skip=('-c', '-b', '-a', '+a'))
    # Glas
    m.quad(glass, [fw, fw, c0 + 0.022], [w - fw, fw, c0 + 0.022], [w - fw, h - fw, c0 + 0.022], [fw, h - fw, c0 + 0.022])
    sc0, sc1 = c0 + 0.03, c1 + 0.012
    bw = 0.022
    def sash(a0, a1, b0, b1, nr, nc=1):
        sw = 0.05
        m.box(frame_mat, W, a0, a0 + sw, b0, b1, sc0, sc1, skip=('-c',))
        m.box(frame_mat, W, a1 - sw, a1, b0, b1, sc0, sc1, skip=('-c',))
        m.box(frame_mat, W, a0 + sw, a1 - sw, b0, b0 + sw * 1.3, sc0, sc1, skip=('-c', '-a', '+a'))
        m.box(frame_mat, W, a0 + sw, a1 - sw, b1 - sw, b1, sc0, sc1, skip=('-c', '-a', '+a'))
        ia0, ia1, ib0, ib1 = a0 + sw, a1 - sw, b0 + sw * 1.3, b1 - sw
        for k in range(1, nr):
            y = ib0 + (ib1 - ib0) * k / nr
            m.box(frame_mat, W, ia0, ia1, y - bw / 2, y + bw / 2, c0 + 0.03, sc1 - 0.006, skip=('-c', '-a', '+a'))
        for k in range(1, nc):
            x = ia0 + (ia1 - ia0) * k / nc
            m.box(frame_mat, W, x - bw / 2, x + bw / 2, ib0, ib1, c0 + 0.03, sc1 - 0.008, skip=('-c', '-b', '+b'))
    ia0, ia1 = fw, w - fw
    ib0 = fw * 1.2
    ib1 = (transom - 0.035) if transom else h - fw
    for k in range(cas):
        sash(ia0 + (ia1 - ia0) * k / cas, ia0 + (ia1 - ia0) * (k + 1) / cas, ib0, ib1, rows, 2 if bars_v else 1)
    if cas == 2:
        mid = (ia0 + ia1) / 2
        m.box(frame_mat, W, mid - 0.022, mid + 0.022, ib0, ib1, sc1 - 0.004, sc1 + 0.012, skip=('-c',))
    if transom:
        m.box(frame_mat, W, fw, w - fw, transom - 0.035, transom + 0.035, c0, c1 + 0.004, skip=('-c', '-a', '+a'))
        tb0, tb1 = transom + 0.035, h - fw
        if single_top:
            sash(ia0, ia1, tb0, tb1, top_rows, 1)
        else:
            for k in range(cas):
                sash(ia0 + (ia1 - ia0) * k / cas, ia0 + (ia1 - ia0) * (k + 1) / cas, tb0, tb1, top_rows, 1)
    # Fensterbank
    if sill == 'wood':
        m.box(sill_mat or frame_mat, W, -0.03, w + 0.03, -0.045, 0.0, -dr, 0.05, skip=('-c',))
        m.quad(reveal, [0, 0, 0], [w, 0, 0], [w, 0, -dr + 0.0], [0, 0, -dr + 0.0]) if False else None
    elif sill == 'stone':
        m.box(sill_mat or 'Sandstein', W, -0.08, w + 0.08, -0.12, 0.0, -dr, 0.07, skip=('-c',))
    elif sill == 'zinc':
        m.box('Zink', W, -0.02, w + 0.02, -0.02, 0.0, -dr, 0.05, skip=('-c',))
    if sill is not None:
        m.quad(reveal, [0, 0.0005, 0], [w, 0.0005, 0], [w, 0.0005, -dr], [0, 0.0005, -dr]) if False else None
        # Innere Bank (sichtbar zwischen Wand und Rahmen)
        m.quad(sill_mat or (frame_mat if sill == 'wood' else 'Sandstein'), [0, 0, 0], [w, 0, 0], [w, 0, -dr], [0, 0, -dr])
    return m


def arch_points(w, hs, n=14, seg=None):
    """Umriss Rundbogen (Breite w, Kaempferhoehe hs) als CCW-Polygon (a,b)"""
    pts = [(0.0, 0.0), (w, 0.0), (w, hs)]
    r = w / 2
    for k in range(1, n):
        t = math.pi * k / n
        pts.append((r + r * math.cos(t), hs + r * math.sin(t)))
    pts.append((0.0, hs))
    return np.array(pts)


def arched_door(w, hs, dr=0.35, reveal='Quader', leaf='Tuer_holz', stone='Sandstein', ring=0.30, n=14, straps=True):
    """zweifluegelige Bogentuer: Oeffnung arch_points(w,hs); Laibung, Bogensteine (Keilsteine), Bohlenfluegel mit Baendern"""
    m = MB(); W = WORLD
    P = arch_points(w, hs, n)
    # Laibung (nach innen gerichtete Flaechen)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        if abs(a[1]) < 1e-6 and abs(b[1]) < 1e-6:
            m.quad('Sandstein', [a[0], 0, 0], [b[0], 0, 0], [b[0], 0, -dr], [a[0], 0, -dr])      # Schwelle
            continue
        m.quad(reveal, [a[0], a[1], 0], [a[0], a[1], -dr], [b[0], b[1], -dr], [b[0], b[1], 0])
    # Fluegel (Bohlen)
    cl = -dr + 0.06
    m.poly(leaf, WORLD, P, c=cl, uv=lambda V: V.copy())
    # Rahmen/Mittelstoss/Baender
    m.box('Holz_dunkel', W, w / 2 - 0.03, w / 2 + 0.03, 0, hs + w / 2 - 0.02, cl, cl + 0.03, skip=('-c',))
    if straps:
        for y in (0.35, hs * 0.55, hs - 0.15):
            for (a0, a1) in ((0.04, w / 2 - 0.08), (w / 2 + 0.08, w - 0.04)):
                m.box('Eisen', W, a0, a1, y - 0.03, y + 0.03, cl, cl + 0.012, skip=('-c',))
        # Ringe
        for a in (w / 2 - 0.18, w / 2 + 0.18):
            m.tube('Eisen', [[a, hs * 0.48, cl + 0.04], [a + 0.001, hs * 0.48 - 0.001, cl + 0.04]], 0.01, n=4)
            m.lathe('Eisen', Frame([a, hs * 0.5, cl + 0.01], [1, 0, 0], [0, 0, 1], [0, -1, 0]), [(0.0, 0.0), (0.035, 0.0), (0.03, 0.02), (0.0, 0.025)], n=8)
    # Diagonale Bohlenfugen im Bogenfeld (Sonnenmuster)
    r = w / 2
    for k in range(1, 6):
        t = math.pi * k / 6
        p0 = np.array([r, hs]); p1 = p0 + 0.96 * r * np.array([math.cos(t), math.sin(t)])
        d = unit(p1 - p0); nn = np.array([-d[1], d[0]]) * 0.012
        m.quad('Holz_dunkel', [*(p0 - nn), cl + 0.004], [*(p1 - nn), cl + 0.004], [*(p1 + nn), cl + 0.004], [*(p0 + nn), cl + 0.004])
    # Bogensteine
    if ring > 0:
        nst = 9
        for k in range(nst):
            t0, t1 = math.pi * k / nst, math.pi * (k + 1) / nst
            g = 0.006
            ang = np.linspace(t0 + g / r, t1 - g / r, 3)
            inner = [(r + r * math.cos(t), hs + r * math.sin(t)) for t in ang]
            outer = [(r + (r + ring) * math.cos(t), hs + (r + ring) * math.sin(t)) for t in ang[::-1]]
            poly = np.array(inner + outer)
            m.prism(stone, WORLD, poly, -0.02, 0.035, bevel=0.012, grain=(math.cos((t0 + t1) / 2), math.sin((t0 + t1) / 2)))
        # Kaempfersteine
        for a0 in (-ring, w):
            m.prism(stone, WORLD, np.array([[a0, hs - 0.30], [a0 + ring, hs - 0.30], [a0 + ring, hs + 0.0], [a0, hs + 0.0]]), -0.02, 0.03, bevel=0.01)
    return m


def knagge(H=0.75, D=0.40, w=0.17, red=True):
    """Knagge (geschnitzte Kopfstrebe) unter dem Balkenkopf; lokal: a entlang Wand, b nach oben (0 = Unterkante Balken),
    c nach aussen (0 = untere Wandflaeche)."""
    m = MB()
    prof = [(0.0, -H), (0.07, -H), (0.08, -H + 0.06), (0.05, -H + 0.11), (0.09, -H + 0.16)]
    # konkave Kehle
    for k in range(1, 9):
        t = k / 9
        d = 0.09 + (D - 0.16) * (1 - math.cos(t * math.pi / 2))
        y = -H + 0.16 + (H - 0.30) * math.sin(t * math.pi / 2)
        prof.append((d, y))
    prof += [(D - 0.04, -0.14), (D - 0.06, -0.10), (D, -0.07), (D, 0.0), (0.0, 0.0)]
    F = Frame([0, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0])      # Profil in (c, b), Extrusion entlang -a
    m.prism('Holz_dunkel', F, np.array(prof), -w / 2, w / 2, bevel=0.012, back=True, grain=(0.3, 1.0))
    if red:
        # rot gefasste Rolle und Band (vgl. Nachtfoto)
        Fr = Frame([0, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0])
        for (yy, dd) in ((-H + 0.13, 0.085), (-0.09, D - 0.035)):
            m.box('Farbe_rot', WORLD, -w / 2 - 0.004, w / 2 + 0.004, yy - 0.025, yy + 0.025, dd - 0.05, dd + 0.004, skip=('-c',))
    return m


def joist_head(D=0.30, w=0.20, hgt=0.22, mat='Holz_dunkel', red=False):
    m = MB()
    prof = [(0.0, -hgt), (D - 0.05, -hgt), (D, -hgt + 0.06), (D, 0.0), (0.0, 0.0)]
    F = Frame([0, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0])
    m.prism(mat, F, np.array(prof), -w / 2, w / 2, bevel=0.01, back=True)
    if red:
        m.box('Farbe_rot', WORLD, -w / 2 - 0.003, w / 2 + 0.003, -hgt - 0.003, -hgt + 0.035, D - 0.12, D - 0.04, skip=('+b',))
    return m


def ridge_tile(L=0.40, r0=0.125, r1=0.14, mat='Ziegel', n=7):
    """Firstziegel: halber Kegelstumpf entlang +a (0..L), oben offen nach -b; Lippe am weiten Ende"""
    m = MB()
    P, N, UV = [], [], []
    angs = np.linspace(0, math.pi, n + 1)
    for j in range(n):
        q = []; nn = []
        for (a, r, t) in ((0, r0, angs[j]), (L, r1, angs[j]), (L, r1, angs[j + 1]), (0, r0, angs[j + 1])):
            q.append([a, r * math.sin(t) - 0.04, r * math.cos(t)]); nn.append([0, math.sin(t), math.cos(t)])
        uv = [[0, j / n * 0.4], [L, j / n * 0.4], [L, (j + 1) / n * 0.4], [0, (j + 1) / n * 0.4]]
        for k in (0, 1, 2, 0, 2, 3):
            P.append(q[k]); N.append(nn[k]); UV.append(uv[k])
    m.add(mat, P, N, UV)
    # Lippe (Stirnring) am Ende a=L
    for j in range(n):
        t0, t1 = angs[j], angs[j + 1]
        o0 = [L, r1 * math.sin(t0) - 0.04, r1 * math.cos(t0)]; o1 = [L, r1 * math.sin(t1) - 0.04, r1 * math.cos(t1)]
        i0 = [L, (r1 - 0.022) * math.sin(t0) - 0.04, (r1 - 0.022) * math.cos(t0)]; i1 = [L, (r1 - 0.022) * math.sin(t1) - 0.04, (r1 - 0.022) * math.cos(t1)]
        m.quad(mat, o1, o0, i0, i1, np.array([[0, 0], [0.02, 0], [0.02, 0.02], [0, 0.02]]))
    # Flankenabschluss unten (beidseitig, gegen Durchsicht)
    for s in (1, -1):
        a0 = [0, -0.04, s * r0]; a1 = [L, -0.04, s * r1]
        m.quad(mat, a0, a1, [L, -0.10, s * (r1 - 0.01)], [0, -0.10, s * (r0 - 0.01)]) if s < 0 else \
            m.quad(mat, a1, a0, [0, -0.10, s * (r0 - 0.01)], [L, -0.10, s * (r1 - 0.01)])
    return m


def gutter_hook(mat='Zink'):
    m = MB()
    m.box(mat, WORLD, -0.012, 0.012, -0.08, -0.07, -0.09, 0.09, skip=())
    return m


def wall_lantern(mat='Laterne_gruen', glass='Laternenglas'):
    """Altstadt-Wandlaterne: Konsolarm aus der Wand (c), Laterne darunter. Ursprung = Wandbefestigung, b nach oben."""
    m = MB()
    arm = [[0, 0.0, 0.0], [0, 0.05, 0.25], [0, 0.04, 0.50], [0, -0.02, 0.62]]
    m.tube(mat, arm, 0.018, n=6)
    m.tube(mat, [[0, -0.30, 0.0], [0, -0.22, 0.12], [0, -0.06, 0.32], [0, 0.03, 0.45]], 0.012, n=5)
    m.box(mat, WORLD, -0.06, 0.06, -0.38, 0.08, -0.02, 0.02, skip=('-c',))
    cx = 0.62
    F = Frame([0, -0.80, cx], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    # Unterteil, Glaskoerper (6-eckig, nach oben weiter), Dach, Spitze
    m.lathe(mat, F, [(0.0, 0.0), (0.06, 0.0), (0.09, 0.08), (0.11, 0.10)], n=6, smooth=False)
    m.lathe(glass, F, [(0.105, 0.10), (0.15, 0.48)], n=6, smooth=False)
    for k in range(6):
        t = 2 * math.pi * k / 6
        p0 = F.P([0.11 * math.cos(t), 0.10, -0.11 * math.sin(t)]); p1 = F.P([0.155 * math.cos(t), 0.48, -0.155 * math.sin(t)])
        m.tube(mat, [p0, p1], 0.008, n=4, caps=False)
    m.lathe(mat, F, [(0.15, 0.47), (0.19, 0.50), (0.18, 0.53), (0.10, 0.62), (0.03, 0.70), (0.03, 0.74), (0.045, 0.76), (0.0, 0.80)], n=6, smooth=False)
    return m


def bench_plaque(w=0.5, h=0.7, mat='Bronze'):
    m = MB()
    m.box(mat, WORLD, 0, w, 0, h, 0.0, 0.025, skip=('-c',))
    return m


PROTOS = dict(knagge=knagge, ridge=ridge_tile, gutter_hook=gutter_hook, lantern=wall_lantern)

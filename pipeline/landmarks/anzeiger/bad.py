"""Goseriedebad / Kestnergesellschaft (Goseriede 11, municipal bath 1905) - hand-built front + hall, LoD2-derived rear wing.
Front frame FB: u along the street (south -> north) from the Anzeiger NE corner, e outward (east)."""
import json, os
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import geo, parts, font, anz
from geo import MB, Frame, UP, lbox, prism, lathe, nrm
from parts import place, wall_grid, ring_faces

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'site_lod2.json')))

O_SH, O_DET, O_WIN, O_ROOF, O_ROOFD, O_LET = 'bad_shell', 'bad_detail_nocol', 'bad_windows_nocol', 'bad_roof', 'bad_roof_detail_nocol', 'bad_letters_nocol'
YG, Y0 = -2.0, -0.68
Y_LB = 10.06                      # low block roof
Y_DRUM, Y_COR, Y_EAVE, Y_RIDGE = 10.06, 14.6, 15.15, 23.37
P0 = np.array([-489.93, 0.0, -300.29])
DB = nrm(np.array([-0.1444, 0.0, -0.9895]))
FB = Frame(P0, DB)
AE = np.array([-499.0, 0.0, -309.12]); AW = np.array([-514.95, 0.0, -308.14])
AX = nrm(AW - AE)                 # hall axis (westward)
BX = np.cross(AX, UP)             # across (outward right of walking west = north? computed)
if BX[2] > 0: BX = -BX            # BX points north (z negative)
RE, RW, W2 = 4.75, 4.5, 8.15
EAVE_OH = 0.45

U_STONE = parts.window_unit(0.72, 1.9, 'grid', bars=(2, 3))


def F_at(u0, e0, t2):
    return Frame(FB.w(u0, 0, e0), FB.s * t2[0] + FB.n * t2[1])


# ------------------------------------------------------------------ helpers
def sweep3(mb, obj, mat, pts, outs, profile, closed=False, caps=True):
    """profile [(a,b)] (bottom->top, outward normal (db,-da)) applied at 3d points with individual horizontal outward dirs"""
    pts = np.asarray(pts, float); outs = np.asarray(outs, float); pr = np.asarray(profile, float)
    n = len(pts); k = len(pr)
    rings = pts[:, None, :] + outs[:, None, :] * pr[None, :, 0:1] + UP[None, None, :] * pr[None, :, 1:2]
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        o = nrm(outs[i] + outs[j])
        for q in range(k - 1):
            da, db = pr[q + 1] - pr[q]
            mb.quad(obj, mat, rings[i, q], rings[j, q], rings[j, q + 1], rings[i, q + 1], o * db + UP * (-da))
    if caps and not closed:
        t0 = nrm(pts[0] - pts[1]); t1 = nrm(pts[-1] - pts[-2])
        mb.poly(obj, mat, rings[0], t0); mb.poly(obj, mat, rings[-1], t1)


CORNICE = [(-0.35, 0.0), (0.0, 0.0), (0.0, 0.12), (0.12, 0.18), (0.12, 0.26), (0.32, 0.42), (0.42, 0.44), (0.42, 0.56), (-0.35, 0.56)]
STRING = [(-0.1, 0.0), (0.0, 0.0), (0.08, 0.06), (0.08, 0.16), (0.0, 0.2), (-0.1, 0.2)]


def dentils(mb, F, s0, s1, y, e, w=0.08, gap=0.1, h=0.1, d=0.08, mat='stone'):
    s = s0
    while s + w <= s1:
        lbox(mb, O_DET, mat, F, s, s + w, y - h, y, e, e + d, ('e-', 'y+'))
        s += w + gap


def ornament(mb, F, s, y, e, kind='leaf', scale=1.0, mat='stone', depth=0.08):
    """carved relief ornament from shapely shapes (leaf fan / cartouche / mask), extruded from the wall"""
    from shapely.geometry import Point as P, Polygon as Pg
    from shapely import affinity
    shapes = []
    if kind == 'leaf':
        for a in (-50, -25, 0, 25, 50):
            leaf = affinity.scale(P(0, 0).buffer(0.5, 12), 0.22, 1.0)
            leaf = affinity.translate(leaf, 0, 0.5); leaf = affinity.rotate(leaf, a, origin=(0, 0))
            shapes.append(leaf)
        shapes.append(P(0, 0).buffer(0.16, 10))
    elif kind == 'cartouche':
        body = affinity.scale(P(0, 0).buffer(0.5, 16), 0.7, 1.0)
        shapes.append(body.difference(affinity.scale(P(0, 0).buffer(0.5, 16), 0.45, 0.72)))
        shapes.append(affinity.scale(P(0, 0).buffer(0.5, 16), 0.3, 0.5))
        for sx in (-1, 1):
            shapes.append(affinity.translate(P(0, 0).buffer(0.13, 10).difference(P(0, 0).buffer(0.06, 8)), sx * 0.45, 0.35))
            shapes.append(affinity.translate(P(0, 0).buffer(0.13, 10).difference(P(0, 0).buffer(0.06, 8)), sx * 0.42, -0.38))
    elif kind == 'mask':
        shapes.append(affinity.scale(P(0, 0).buffer(0.5, 16), 0.55, 0.7))
    elif kind == 'rosette':
        for a in range(0, 360, 45):
            shapes.append(affinity.rotate(affinity.translate(affinity.scale(P(0, 0).buffer(0.5, 8), 0.12, 0.3), 0, 0.25), a, origin=(0, 0)))
        shapes.append(P(0, 0).buffer(0.12, 10))
    elif kind == 'scroll':
        for sx in (-1, 1):
            r = P(0, 0).buffer(0.2, 12).difference(P(0, 0).buffer(0.1, 10))
            shapes.append(affinity.translate(r, sx * 0.35, 0))
        shapes.append(Pg([(-0.35, -0.2), (0.35, -0.2), (0.25, 0.2), (-0.25, 0.2)]))
    shp = unary_union(shapes)
    shp = affinity.scale(shp, scale, scale, origin=(0, 0))
    shp = affinity.translate(shp, s, y)
    geoms = list(shp.geoms) if shp.geom_type == 'MultiPolygon' else [shp]
    for g in geoms:
        ext = np.array(g.exterior.coords)[:-1]; holes = [np.array(h.coords)[:-1] for h in g.interiors]
        Wp = lambda pts, ee: [F.w(p[0], p[1], ee) for p in pts]
        # two-level relief: base slab + slightly smaller raised core for depth cues
        mb.poly(O_DET, mat, Wp(ext, e + depth), F.n, holes=[Wp(h, e + depth) for h in holes])
        for ring, hole in [(ext, False)] + [(h, True) for h in holes]:
            r = np.asarray(ring)
            area = 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
            for i in range(len(r)):
                a, b = r[i], r[(i + 1) % len(r)]
                d = b - a
                if np.hypot(*d) < 1e-6: continue
                o2 = np.array([d[1], -d[0]])
                if (area < 0) != hole: o2 = -o2
                mb.quad(O_DET, mat, F.w(a[0], a[1], e), F.w(b[0], b[1], e), F.w(b[0], b[1], e + depth), F.w(a[0], a[1], e + depth), F.s * o2[0] + UP * o2[1])


def round_hole_wall(mb, F, s0, s1, y0, y1, circles, mat='ashlar', depth=0.3, rects=()):
    """planar wall rect with circular/elliptic holes [(sc, yc, rx, ry)] and rect holes; reveals; round window units"""
    outer = [F.w(s0, y0, 0), F.w(s1, y0, 0), F.w(s1, y1, 0), F.w(s0, y1, 0)]
    holes = []
    for (sc, yc, rx, ry) in circles:
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([sc + rx * np.cos(a), yc + ry * np.sin(a)], -1)
        holes.append([F.w(p[0], p[1], 0) for p in ring])
        # reveal (cylinder inward)
        for i in range(24):
            p, q = ring[i], ring[(i + 1) % 24]
            mid = (p + q) / 2 - np.array([sc, yc])
            mb.quad(O_SH, mat, F.w(p[0], p[1], 0), F.w(q[0], q[1], 0), F.w(q[0], q[1], -depth), F.w(p[0], p[1], -depth), -(F.s * mid[0] + UP * mid[1]))
        # stone frame ring (raised) with ears
        ro = np.stack([sc + (rx + 0.12) * np.cos(a), yc + (ry + 0.12) * np.sin(a)], -1)
        mb.poly(O_DET, 'stone', [F.w(p[0], p[1], 0.06) for p in ro], F.n, holes=[[F.w(p[0], p[1], 0.06) for p in ring]])
        ring_faces_world(mb, F, ro, 0.0, 0.06)
        for i in range(24):              # inner side of the raised frame (facing the opening) - closes grazing-view slits
            p, q = ring[i], ring[(i + 1) % 24]
            mid = (p + q) / 2 - np.array([sc, yc])
            mb.quad(O_DET, 'stone', F.w(p[0], p[1], 0.0), F.w(q[0], q[1], 0.0), F.w(q[0], q[1], 0.06), F.w(p[0], p[1], 0.06), -(F.s * mid[0] + UP * mid[1]))
        for sx in (-1, 1):
            lbox(mb, O_DET, 'stone', F, sc + sx * (rx + 0.1) - 0.09, sc + sx * (rx + 0.1) + 0.09, yc - 0.18, yc + 0.18, 0.0, 0.07, ('e-',))
        lbox(mb, O_DET, 'stone', F, sc - rx - 0.15, sc + rx + 0.15, yc - ry - 0.2, yc - ry - 0.1, 0.0, 0.1, ('e-',))
        # glass + cross bars
        mb.poly(O_WIN, 'glass', [F.w(p[0], p[1], -depth + 0.05) for p in ring], F.n)
        lbox(mb, O_WIN, 'metal_dark', F, sc - 0.02, sc + 0.02, yc - ry, yc + ry, -depth + 0.05, -depth + 0.09, ('e-',))
        lbox(mb, O_WIN, 'metal_dark', F, sc - rx, sc + rx, yc - 0.02, yc + 0.02, -depth + 0.05, -depth + 0.09, ('e-',))
    for (sa, sb, ya, yb) in rects:
        holes.append([F.w(sa, ya, 0), F.w(sb, ya, 0), F.w(sb, yb, 0), F.w(sa, yb, 0)])
    mb.poly(O_SH, mat, outer, F.n, holes=holes)


def ring_faces_world(mb, F, ring, e0, e1):
    r = np.asarray(ring)
    area = 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
    for i in range(len(r)):
        a, b = r[i], r[(i + 1) % len(r)]
        d = b - a; o2 = np.array([d[1], -d[0]])
        if area < 0: o2 = -o2
        mb.quad(O_DET, 'stone', F.w(a[0], a[1], e0), F.w(b[0], b[1], e0), F.w(b[0], b[1], e1), F.w(a[0], a[1], e1), F.s * o2[0] + UP * o2[1])


def rect_panel(mb, F, s0, s1, y0, y1, depth=0.05, mat='stone', motif='leaf'):
    """recessed relief panel with carved motif"""
    # frame (raised border)
    for (a, b, c, d) in ((s0, s1, y0, y0 + 0.08), (s0, s1, y1 - 0.08, y1), (s0, s0 + 0.08, y0 + 0.08, y1 - 0.08), (s1 - 0.08, s1, y0 + 0.08, y1 - 0.08)):
        lbox(mb, O_DET, mat, F, a, b, c, d, 0.0, 0.05, ('e-',))
    ornament(mb, F, (s0 + s1) / 2, (y0 + y1) / 2, 0.0, motif, scale=min(s1 - s0, y1 - y0) * 0.8, depth=0.07)


def grille(mb, F, s0, s1, y0, y1, e):
    n = max(3, int((s1 - s0) / 0.1))
    for s in np.linspace(s0 + 0.05, s1 - 0.05, n):
        lbox(mb, O_DET, 'metal_dark', F, s - 0.012, s + 0.012, y0, y1, e, e + 0.025, ('e-',))
    for y in (y0 + 0.05, y1 - 0.05):
        lbox(mb, O_DET, 'metal_dark', F, s0, s1, y - 0.015, y + 0.015, e + 0.025, e + 0.04, ('e-',))


# ------------------------------------------------------------------ front: left wing, recess, bay, right wing
def left_wing(mb):
    F = F_at(0.0, 0.0, (1, 0)); L = 3.49; top = 13.7
    # south side strip: the wing front stands ~0.55 m proud of the Anzeiger's east wall -> close that slot (faces south)
    Fs_ = F_at(0.0, -0.7, (0, 1))
    mb.quad(O_SH, 'ashlar', Fs_.w(0, YG, 0), Fs_.w(0.7, YG, 0), Fs_.w(0.7, top - 0.56, 0), Fs_.w(0, top - 0.56, 0), Fs_.n)
    mb.quad(O_SH, 'ashlar', Fs_.w(0, top - 0.56, 0), Fs_.w(0.35, top - 0.56, 0), Fs_.w(0.35, top, 0), Fs_.w(0, top, 0), Fs_.n)   # rest = cornice cap
    holes = []
    for sc in (1.65, 2.62):
        holes += [(sc - 0.29, sc + 0.29, 3.9, 6.3), (sc - 0.29, sc + 0.29, 7.95, 10.1)]
    # basement windows
    holes += [(0.55, 1.25, -0.6, -0.1), (2.25, 2.95, -0.6, -0.1)]
    wall_grid(mb, F, 0.0, L, YG, 12.9, holes, O_SH, 'ashlar', None, 0.25, 'ashlar', 'stone', sill=False)
    for sc in (1.65, 2.62):
        for (ya, yb) in ((3.9, 6.3), (7.95, 10.1)):
            place(mb, O_WIN, parts.window_unit(0.58, yb - ya, 'grid', bars=(1, 3)), F, sc, ya, -0.25)
        rect_panel(mb, F, sc - 0.3, sc + 0.3, 6.45, 7.8, motif='leaf')
    for sa in (0.55, 2.25):
        grille(mb, F, sa, sa + 0.7, -0.6, -0.1, -0.15)
        mb.quad(O_WIN, 'glass', F.w(sa, -0.6, -0.25), F.w(sa + 0.7, -0.6, -0.25), F.w(sa + 0.7, -0.1, -0.25), F.w(sa, -0.1, -0.25), F.n)   # dark basement pane
    # window surround (raised stone frame) + cartouche hood
    lbox(mb, O_DET, 'stone', F, 1.2, 3.07, 3.75, 3.9, 0.0, 0.1, ('e-',))
    lbox(mb, O_DET, 'stone', F, 1.2, 3.07, 10.1, 10.3, 0.0, 0.12, ('e-',))
    for x in (1.2, 3.07 - 0.12):
        lbox(mb, O_DET, 'stone', F, x, x + 0.12, 3.9, 10.1, 0.0, 0.06, ('e-',))
    ornament(mb, F, 2.13, 10.85, 0.0, 'cartouche', 0.9, depth=0.14)
    lbox(mb, O_DET, 'stone', F, 1.25, 3.02, 10.3, 10.42, 0.0, 0.18, ('e-',))
    # plinth course
    sweep3(mb, O_DET, 'stone', [F.w(0, 0.3, 0), F.w(L, 0.3, 0)], [F.n, F.n], [(0.0, 0.0), (0.06, 0.04), (0.06, 0.12), (0.0, 0.12)], caps=True)      # starts at the wall plane (caps do not overlap the side strips)
    # cornice + attic band on front and north return, flat roof
    mb.quad(O_SH, 'ashlar', F.w(0, 12.9, 0), F.w(L, 12.9, 0), F.w(L, top - 0.56, 0), F.w(0, top - 0.56, 0), F.n)
    pts = [F.w(-0.0, top - 0.56, 0), F.w(L, top - 0.56, 0), F.w(L, top - 0.56, -2.3)]
    outs = [F.n, nrm(F.n + F.s), F.s]
    sweep3(mb, O_SH, 'stone', pts, outs, CORNICE)
    dentils(mb, F, 0.05, L - 0.05, top - 0.44, 0.0)
    # north return (u = L) down to the recess, and back face above the low block
    Fr = F_at(L, 0.0, (0, -1))
    mb.quad(O_SH, 'ashlar', Fr.w(0, YG, 0), Fr.w(1.73, YG, 0), Fr.w(1.73, Y_LB, 0), Fr.w(0, Y_LB, 0), Fr.n)
    mb.quad(O_SH, 'ashlar', Fr.w(0, Y_LB, 0), Fr.w(2.3, Y_LB, 0), Fr.w(2.3, top - 0.56, 0), Fr.w(0, top - 0.56, 0), Fr.n)
    Fbk = F_at(L, -2.3, (-1, 0))
    mb.quad(O_SH, 'ashlar', Fbk.w(0, Y_LB, 0), Fbk.w(L, Y_LB, 0), Fbk.w(L, top - 0.56, 0), Fbk.w(0, top - 0.56, 0), Fbk.n)
    mb.quad(O_SH, 'ashlar', Fbk.w(0.35, top - 0.56, 0), Fbk.w(L, top - 0.56, 0), Fbk.w(L, top - 0.02, 0), Fbk.w(0.35, top - 0.02, 0), Fbk.n)   # cornice end cap covers s < 0.35
    P = [FB.w(0, top, 0.42), FB.w(L + 0.42, top, 0.42), FB.w(L + 0.42, top, -2.3), FB.w(0, top, -2.3)]
    mb.poly(O_SH, 'roof_flat', [FB.w(0, top - 0.02, 0), FB.w(L, top - 0.02, 0), FB.w(L, top - 0.02, -2.3), FB.w(0, top - 0.02, -2.3)], UP)


def bay_geometry():
    """circle through the LoD2 bay points (local u,e)"""
    A = np.array([5.49, -1.665]); C = np.array([13.40, -1.148]); B = np.array([9.45, 0.05])
    # circumcircle
    ax, ay = A; bx, by = B; cx, cy = C
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    c = np.array([ux, uy]); r = np.linalg.norm(A - c)
    a0 = np.arctan2(*(A - c)[::-1]); a1 = np.arctan2(*(C - c)[::-1])
    return c, r, a0, a1


def bay(mb):
    c, r, a0, a1 = bay_geometry()
    # 9 facets: 1 side, 7 lights, 1 side ; angles from a0 (south end) to a1 (north end)
    # (u,e) param: p = c + r*(cos a, sin a) with u = x, e = y of the 2d circle
    side = 0.16
    angs = [a0, a0 + (a1 - a0) * side] + list(np.linspace(a0 + (a1 - a0) * side, a1 - (a1 - a0) * side, 8))[1:] + [a1]
    pts = [c + r * np.array([np.cos(a), np.sin(a)]) for a in angs]
    for i in range(len(pts) - 1):
        p, q = pts[i], pts[i + 1]
        t2 = (q - p); L = np.linalg.norm(t2); t2 = t2 / L
        F = F_at(p[0], p[1], t2)
        if F.n @ FB.n < -0.2: F = F_at(q[0], q[1], -t2)       # keep outward
        light = 1 <= i <= 7
        if not light:
            wall_grid(mb, F, 0.0, L, YG, Y_LB, [], O_SH, 'ashlar')
            continue
        k = i                         # 1..7
        # band window light (segmental head: centre taller)
        tnorm = (k - 4) / 3.0
        head = 7.75 + 0.9 * np.cos(tnorm * np.pi / 2.2)
        lw = L - 0.22
        holes = [(0.11, L - 0.11, 5.7, head)]
        rects = []
        circ = []
        if k % 2 == 1:
            circ = [(L / 2, 2.64, 0.4, 0.5)]
        wall_parts_bay(mb, F, L, holes, circ, k)
        place(mb, O_WIN, parts.window_unit(lw, head - 5.7, 'grid', bars=(1, 2 if head - 5.7 < 2.6 else 3)), F, L / 2, 5.7, -0.3)
        # stone mullion fronts between lights
        lbox(mb, O_DET, 'stone', F, -0.11, 0.11, 5.55, head + 0.05, 0.0, 0.06, ('e-',))
        if k == 7:
            lbox(mb, O_DET, 'stone', F, L - 0.11, L + 0.11, 5.55, head + 0.05, 0.0, 0.06, ('e-',))
        if k % 2 == 1:
            rect_panel(mb, F, 0.1, L - 0.1, 3.8, 5.35, motif='leaf')
            lbox(mb, O_DET, 'metal_dark', F, L / 2 - 0.32, L / 2 + 0.32, -0.62, -0.12, -0.25, -0.2, ('e-',))
            grille(mb, F, L / 2 - 0.32, L / 2 + 0.32, -0.62, -0.12, -0.12)
        else:
            ornament(mb, F, L / 2, 5.05, 0.0, 'scroll', 0.45, depth=0.16)
            ornament(mb, F, L / 2, 2.64, 0.0, 'rosette', 0.42, depth=0.08)
    # sill band (light facets) and plinth course (all facets) as mitred sweeps (no overlapping boxes at the facet joints)
    def miter_outs(P2):
        W = [FB.w(p[0], 0, p[1]) for p in P2]
        nf = []
        for i in range(len(W) - 1):
            t = nrm(W[i + 1] - W[i]); o = np.cross(t, UP)
            rad = (P2[i] + P2[i + 1]) / 2 - c                # radial direction of the facet midpoint (u, e)
            if o @ (FB.s * rad[0] + FB.n * rad[1]) < 0: o = -o
            nf.append(o)
        outs = []
        for i in range(len(W)):
            if i == 0: outs.append(nf[0])
            elif i == len(W) - 1: outs.append(nf[-1])
            else:
                m = nrm(nf[i - 1] + nf[i]); outs.append(m / (m @ nf[i]))
        return W, outs
    W, O = miter_outs(pts[1:9])
    sweep3(mb, O_DET, 'stone', [w + UP * 5.45 for w in W], O, [(-0.3, 0.0), (0.14, 0.0), (0.14, 0.27), (-0.3, 0.27)])
    W, O = miter_outs(pts)
    sweep3(mb, O_DET, 'stone', [w + UP * 0.22 for w in W], O, [(-0.05, 0.0), (0.07, 0.0), (0.07, 0.16), (-0.05, 0.16)])
    # hood cornice following the arc, rising to the centre (segmental bow) + keystone mask
    n = 40
    aa = np.linspace(angs[1], angs[-2], n)
    P3, O3 = [], []
    for a in aa:
        t = (a - (angs[1] + angs[-2]) / 2) / ((angs[-2] - angs[1]) / 2)
        y = 8.75 + 1.05 * np.clip(np.cos(t * np.pi / 2.0), 0, 1) ** 1.2
        p2 = c + r * np.array([np.cos(a), np.sin(a)])
        P3.append(FB.w(p2[0], y, p2[1])); o = FB.s * np.cos(a) + FB.n * np.sin(a); O3.append(nrm(o))
    sweep3(mb, O_SH, 'stone', P3, O3, [(-0.3, 0.0), (0.0, 0.0), (0.12, 0.08), (0.2, 0.2), (0.3, 0.32), (0.3, 0.42), (-0.3, 0.42)])
    # bay top: wall above hood to the flat roof
    for i in range(1, n):
        a, b = aa[i - 1], aa[i]
    pm = c + r * np.array([np.cos((angs[1] + angs[-2]) / 2), np.sin((angs[1] + angs[-2]) / 2)])
    tt = np.array([-np.sin((angs[1] + angs[-2]) / 2), np.cos((angs[1] + angs[-2]) / 2)])
    Fm = F_at(pm[0], pm[1], tt)
    if Fm.n @ FB.n < 0: Fm = F_at(pm[0], pm[1], -tt)
    ornament(mb, Fm, 0.0, 9.55, 0.3, 'mask', 0.55, depth=0.2)
    lbox(mb, O_DET, 'stone', Fm, -0.18, 0.18, 8.55, 9.3, 0.0, 0.22, ('e-',))
    # flat roof of the bay at Y_LB (closes the top between front arc and drum)
    ring = [FB.w(p[0], Y_LB, p[1]) for p in pts]
    return pts


def wall_parts_bay(mb, F, L, holes, circ, k):
    """bay light facet wall: lower part with optional porthole (polygon with hole), upper part with window hole"""
    # lower: YG..5.45 with porthole
    round_hole_wall(mb, F, 0.0, L, YG, 5.45, circ if circ else [], rects=[(L / 2 - 0.32, L / 2 + 0.32, -0.62, -0.12)] if k % 2 == 1 else [])
    if k % 2 == 1:
        sa, sb, ya, yb = L / 2 - 0.32, L / 2 + 0.32, -0.62, -0.12
        for (p, q, hint) in (((sa, ya), (sa, yb), F.s), ((sb, yb), (sb, ya), -F.s)):
            mb.quad(O_SH, 'ashlar', F.w(p[0], p[1], 0), F.w(p[0], p[1], -0.2), F.w(q[0], q[1], -0.2), F.w(q[0], q[1], 0), hint)
        mb.quad(O_SH, 'ashlar', F.w(sa, yb, 0), F.w(sa, yb, -0.2), F.w(sb, yb, -0.2), F.w(sb, yb, 0), -UP)
        mb.quad(O_SH, 'ashlar', F.w(sa, ya, -0.2), F.w(sa, ya, 0), F.w(sb, ya, 0), F.w(sb, ya, -0.2), UP)
    # upper: 5.45..Y_LB with window hole
    wall_grid(mb, F, 0.0, L, 5.45, Y_LB, [(h[0], h[1], h[2], h[3]) for h in holes], O_SH, 'ashlar', None, 0.3, 'ashlar', 'stone', sill=False)


def recess(mb):
    F = F_at(3.49, -1.73, (1, 0))
    L = 5.49 - 3.49
    wall_grid(mb, F, 0.0, L + 0.05, YG, Y_LB, [], O_SH, 'ashlar')
    lbox(mb, O_DET, 'stone', F, 0.0, L, 0.22, 0.38, 0.0, 0.07, ('e-',))


def right_wing(mb):
    F = F_at(13.40, -1.0, (1, 0)); L = 18.6 - 13.40
    top = 8.45
    # south return of the wing between the bay's last facet (ends at e = -1.148) and the wing front (e = -1.0)
    _c, _r, _a0, _a1 = bay_geometry()
    e_end = (_c + _r * np.array([np.cos(_a1), np.sin(_a1)]))[1]
    Fsr = F_at(13.40, e_end - 0.03, (0, 1))
    wsr = -1.0 - e_end + 0.03
    mb.quad(O_SH, 'ashlar', Fsr.w(0, YG, 0), Fsr.w(wsr, YG, 0), Fsr.w(wsr, 8.45, 0), Fsr.w(0, 8.45, 0), Fsr.n)
    mb.quad(O_SH, 'ashlar', Fsr.w(0, 8.9, 0), Fsr.w(wsr, 8.9, 0), Fsr.w(wsr, Y_LB, 0), Fsr.w(0, Y_LB, 0), Fsr.n)
    if wsr > 0.3: mb.quad(O_SH, 'ashlar', Fsr.w(0, 8.45, 0), Fsr.w(wsr - 0.3, 8.45, 0), Fsr.w(wsr - 0.3, 8.9, 0), Fsr.w(0, 8.9, 0), Fsr.n)
    sc = 3.6
    holes = [(sc - 0.38, sc + 0.38, 5.4, 7.2)]
    wall_grid(mb, F, 0.0, L, 3.4, top, holes, O_SH, 'ashlar', None, 0.25, 'ashlar', 'stone')
    place(mb, O_WIN, parts.window_unit(0.76, 1.8, 'grid', bars=(1, 3)), F, sc, 5.4, -0.25)
    round_hole_wall(mb, F, 0.0, L, YG, 3.4, [(sc, 2.48, 0.4, 0.55)], rects=[(sc - 0.33, sc + 0.33, -0.62, -0.12)])
    sa, sb, ya, yb = sc - 0.33, sc + 0.33, -0.62, -0.12
    mb.quad(O_SH, 'ashlar', F.w(sa, ya, 0), F.w(sa, ya, -0.2), F.w(sa, yb, -0.2), F.w(sa, yb, 0), F.s)
    mb.quad(O_SH, 'ashlar', F.w(sb, yb, 0), F.w(sb, yb, -0.2), F.w(sb, ya, -0.2), F.w(sb, ya, 0), -F.s)
    mb.quad(O_SH, 'ashlar', F.w(sa, yb, 0), F.w(sa, yb, -0.2), F.w(sb, yb, -0.2), F.w(sb, yb, 0), -UP)
    mb.quad(O_SH, 'ashlar', F.w(sa, ya, -0.2), F.w(sa, ya, 0), F.w(sb, ya, 0), F.w(sb, ya, -0.2), UP)
    grille(mb, F, sa, sb, ya, yb, -0.12)
    mb.quad(O_WIN, 'glass', F.w(sa, ya, -0.2), F.w(sb, ya, -0.2), F.w(sb, yb, -0.2), F.w(sa, yb, -0.2), F.n)        # dark basement pane
    rect_panel(mb, F, sc - 0.36, sc + 0.36, 3.55, 5.05, motif='leaf')
    ornament(mb, F, sc, 7.65, 0.0, 'scroll', 0.75, depth=0.15)
    lbox(mb, O_DET, 'stone', F, sc - 0.5, sc + 0.5, 7.25, 7.38, 0.0, 0.14, ('e-',))
    ornament(mb, F, 1.25, 5.85, 0.0, 'cartouche', 0.32, depth=0.06)
    lbox(mb, O_DET, 'stone', F, 0.0, L, 0.22, 0.38, 0.0, 0.07, ('e-',))
    # side return to the Kestner glass front (north face) and cornice + slate pavilion roof
    A_r = FB.w(18.6, 0, -1.0)
    Lr = float(np.linalg.norm(K0 - A_r))
    Fr = Frame(A_r, nrm(K0 - A_r))
    wall_grid(mb, Fr, 0.0, Lr, YG, top, [], O_SH, 'ashlar')
    pts = [F.w(-0.0, top, 0), F.w(L, top, 0), K0 + UP * top]
    sweep3(mb, O_SH, 'stone', pts, [F.n, nrm(F.n + F.s), F.s], [(-0.3, 0.0), (0.0, 0.0), (0.0, 0.1), (0.12, 0.16), (0.3, 0.32), (0.38, 0.34), (0.38, 0.45), (-0.3, 0.45)])
    # slate hipped roof (pavilion) up against the drum
    y0 = top + 0.45
    a = [FB.w(13.40 - 0.05, y0, -1.0 + 0.38), FB.w(18.6 + 0.38, y0, -1.0 + 0.38), FB.w(18.6 + 0.38, y0, -3.6), FB.w(13.40 - 0.05, y0, -3.6)]
    rdg = [FB.w(14.5, y0 + 1.2, -2.3), FB.w(17.6, y0 + 1.2, -2.3)]
    mb.poly(O_ROOF, 'slate', [a[0], a[1], rdg[1], rdg[0]], UP + FB.n)
    mb.poly(O_ROOF, 'slate', [a[1], a[2], rdg[1]], UP + FB.s)
    mb.poly(O_ROOF, 'slate', [a[2], a[3], rdg[0], rdg[1]], UP - FB.n)
    mb.poly(O_ROOF, 'slate', [a[3], a[0], rdg[0]], UP - FB.s)
    lathe_ridge(mb, rdg[0], rdg[1], 0.06, 'copper_dark')
    # soffit closing the underside of the pavilion roof (overhang beyond the chamfered cornice corner showed the roof's back faces)
    mb.poly(O_ROOF, 'stone', [p for p in (a[3], a[2], a[1], a[0])], -UP)     # at eave level: no 1.5 cm slit under the slate edge
    # finial
    for p in rdg:
        lathe(mb, O_ROOFD, 'stone', (p[0], p[2]), [(0.1, p[1]), (0.12, p[1] + 0.25), (0.05, p[1] + 0.45), (0.0, p[1] + 0.55)], segs=10)
    # wall above cornice up to the low block roof (inside) is hidden: close the gap between wing top and Y_LB at the back
    Fn_ = F_at(13.40, -1.148, (0, -1))      # bay side wall above the right wing (faces north over the pavilion's south hip)
    mb.quad(O_SH, 'ashlar', Fn_.w(0, top, 0), Fn_.w(3.6 - 1.148, top, 0), Fn_.w(3.6 - 1.148, Y_LB, 0), Fn_.w(0, Y_LB, 0), Fn_.n)
    Fb = F_at(13.40, -3.6, (1, 0))          # faces east (visible step between the slate roof's back edge and the low-block roof)
    # runs 0.45 m on into the Kestner fascia box (closes the slit between the pavilion's back wall and the fascia end)
    mb.quad(O_SH, 'ashlar', Fb.w(0, top, 0), Fb.w(L + 0.45, top, 0), Fb.w(L + 0.45, Y_LB, 0), Fb.w(0, Y_LB, 0), Fb.n)


def lathe_ridge(mb, a, b, r, mat, obj=None, skirt=0.0, caps=False):
    """half-round ridge / hip tile from a to b; skirt: vertical side flanks reaching `skirt` m down into the roof
    (no grazing view under the tile edges); caps: closed half-disc ends"""
    d = nrm(b - a); s = nrm(np.cross(d, UP)); n = 8
    ang = np.linspace(0, np.pi, n + 1)
    prof = [(np.cos(t), np.sin(t)) for t in ang]
    if skirt > 0: prof = [(1.0, -skirt / r)] + prof + [(-1.0, -skirt / r)]
    ring = lambda p: [p + s * r * c + UP * r * q for c, q in prof]
    A = ring(a); B = ring(b)
    for i in range(len(prof) - 1):
        mid = (A[i] + A[i + 1]) / 2 - a
        if skirt > 0 and i == 0: mid = s
        if skirt > 0 and i == len(prof) - 2: mid = -s
        mb.quad(obj or O_ROOFD, mat, A[i], B[i], B[i + 1], A[i + 1], mid)
    if caps:
        mb.poly(obj or O_ROOFD, mat, A, -d); mb.poly(obj or O_ROOFD, mat, B, d)


# ------------------------------------------------------------------ hall (drum walls + tiled roof with apsidal ends + lantern)
# west end: NOT an apse. DOP10 2025 shows planar, sharply lit hip facets at the west end (only the east end is a round
# apse with the lantern); the LoD2 eave outline bulges to x -519.45 -> polygonal hipped end (straight walls):
# corners (a along AX west of AW, b along BX north) at wall line, symmetric about the hall axis
A1W, B1W = 2.4, 3.6
W_CORNERS = [(0.0, W2), (A1W, W2), (RW, B1W), (RW, -B1W), (A1W, -W2), (0.0, -W2)]


def west_outline(offset=0.0, step=0.45):
    """polygonal west end (north -> south) offset outward by `offset`; dense points every <= step m.
    outs: edge normals, mitre vectors (|o| >= 1) at the corners so swept profiles stay parallel to every facet."""
    C = [np.array(c, float) for c in W_CORNERS]
    nrm2 = []
    for i in range(len(C) - 1):
        d = C[i + 1] - C[i]; n = np.array([-d[1], d[0]]); nrm2.append(n / np.linalg.norm(n))
    cor = [C[0] + nrm2[0] * offset]; cor_o = [nrm2[0]]
    for k in range(1, len(C) - 1):
        m = (nrm2[k - 1] + nrm2[k]) / (1.0 + nrm2[k - 1] @ nrm2[k])
        cor.append(C[k] + m * offset); cor_o.append(m)
    cor.append(C[-1] + nrm2[-1] * offset); cor_o.append(nrm2[-1])
    P2, O2, corner = [], [], []
    for i in range(len(cor) - 1):
        L = np.linalg.norm(cor[i + 1] - cor[i]); ns = max(1, int(np.ceil(L / step)))
        for j in range(ns):
            P2.append(cor[i] + (cor[i + 1] - cor[i]) * j / ns); O2.append(cor_o[i] if j == 0 else nrm2[i]); corner.append(j == 0)
    P2.append(cor[-1]); O2.append(cor_o[-1]); corner.append(True)
    W = lambda ab: AW + AX * ab[0] + BX * ab[1]
    pts = np.array([W(p) for p in P2]); outs = np.array([AX * o[0] + BX * o[1] for o in O2])
    return pts, outs, np.array(corner), [W(c) for c in cor]


def hall_outline(n_end=40, offset=0.0, with_corners=False):
    """closed outline (world xz points) of the hall at eaves: east apse (south->north), north side, polygonal west end
    (north->south), south side"""
    pts = []; outs = []; corner = []
    phis = np.linspace(-np.pi / 2, np.pi / 2, n_end + 1)
    R = RE + offset
    for ph in phis:
        p = AE + AX * (-R * np.cos(ph)) + BX * ((W2 + offset) * np.sin(ph))
        o = nrm(-AX * np.cos(ph) / R + BX * np.sin(ph) / (W2 + offset))
        pts.append(p); outs.append(o); corner.append(False)
    wp, wo, wc, _ = west_outline(offset)
    pts = np.concatenate([np.array(pts), wp]); outs = np.concatenate([np.array(outs), wo]); corner = np.concatenate([np.array(corner), wc])
    corner[n_end + 1] = False; corner[-1] = False          # ends of the straight sides are not facet corners
    if with_corners: return pts, outs, corner
    return pts, outs


def hall(mb):
    # walls (ashlar) from inside the low block to the cornice, with arched windows on the east drum, west apse, north side
    pts, outs, corner = hall_outline(40, with_corners=True)
    n = len(pts)
    win_e = [-25, 0, 25]
    for i in range(n):
        j = (i + 1) % n
        a, b = pts[i], pts[j]
        if np.linalg.norm(b - a) < 1e-6: continue
        o = nrm(outs[i] + outs[j])
        mb.quad(O_SH, 'ashlar', a + UP * (Y_DRUM - 0.2), b + UP * (Y_DRUM - 0.2), b + UP * Y_COR, a + UP * Y_COR, o)
    # cornice with dentils around
    sweep3(mb, O_SH, 'stone', pts + UP * Y_COR, outs, [(-0.4, 0.0), (0.0, 0.0), (0.0, 0.1), (0.1, 0.16), (0.1, 0.24), (0.34, 0.44), (0.5, 0.48), (0.5, 0.55), (-0.4, 0.55)], closed=True)
    for i in range(0, n, 1):
        if corner[i]: continue                    # no dentil on a mitred facet corner of the west end
        p = pts[i]; o = nrm(outs[i])
        t = nrm(np.cross(UP, o))
        c = p + o * 0.05 + UP * (Y_COR + 0.05)
        box_world(mb, O_DET, 'stone', c, t, o, 0.05, 0.06, 0.07)
    # arched windows (set in front as framed units; the drum wall behind acts as dark backing via glass)
    hall_windows(mb, pts, outs)
    roof(mb)
    lantern(mb)
    dormer(mb)


def box_world(mb, obj, mat, c, t, o, hw, hh, d):
    """small box centred at c (on a wall), t tangent, o outward; half-width hw, half-height hh, depth d outward"""
    P = lambda a, b, e: c + t * a + UP * b + o * e
    F = [((-hw, -hh, d), (hw, -hh, d), (hw, hh, d), (-hw, hh, d), o), ((hw, -hh, 0), (hw, -hh, d), (hw, hh, d), (hw, hh, 0), t),
         ((-hw, -hh, d), (-hw, -hh, 0), (-hw, hh, 0), (-hw, hh, d), -t), ((-hw, -hh, 0), (hw, -hh, 0), (hw, -hh, d), (-hw, -hh, d), -UP)]
    for a, b, cc, dd, h in F:
        mb.quad(obj, mat, P(*a), P(*b), P(*cc), P(*dd), h)


def hall_windows(mb, pts, outs):
    """3 triple round-arched windows on the east drum, 1 triple + 2 single on the polygonal west end, 4 clerestory windows on the north side"""
    def ell_pt(C, R, ph, east=True):
        if east:
            return C + AX * (-R * np.cos(ph)) + BX * (W2 * np.sin(ph)), nrm(-AX * np.cos(ph) / R + BX * np.sin(ph) / W2)
        return C + AX * (R * np.cos(ph)) - BX * (W2 * np.sin(ph)), nrm(AX * np.cos(ph) / R - BX * np.sin(ph) / W2)
    U3 = parts.arched_window_unit(2.1, 3.2, 1.05, mullions=2, transom=2.15)
    U1 = parts.arched_window_unit(1.3, 2.6, 0.65, mullions=1)
    items = [(ell_pt(AE, RE, np.radians(a), True), U3, 2.1, 3.2) for a in (-27, 0, 27)]
    # polygonal west end: one triple arched window on the west face, single arched lights on the two diagonal facets
    wc = [np.array(c, float) for c in W_CORNERS]
    items.append(((AW + AX * RW, AX), U3, 2.1, 3.2))
    for k in (1, 3):
        m2 = (wc[k] + wc[k + 1]) / 2; d = wc[k + 1] - wc[k]; n2 = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        items.append(((AW + AX * m2[0] + BX * m2[1], nrm(AX * n2[0] + BX * n2[1])), U1, 1.3, 2.6))
    for ((p, o), U, w, h) in items:
        t = nrm(np.cross(UP, o))
        Fw = Frame(p - t * 0 + o * 0.0, t)
        if Fw.n @ o < 0: Fw = Frame(p, -t)
        y0 = 10.55
        # recess niche: dark backing + stone arch surround
        place(mb, O_WIN, U, Fw, 0.0, y0, 0.12)
        arch_surround(mb, Fw, w + 0.36, h + 0.18, y0 - 0.12, 0.0, 0.22)
        lbox(mb, O_DET, 'stone', Fw, -w / 2 - 0.25, w / 2 + 0.25, y0 - 0.2, y0, 0.0, 0.2, ())
    for k in range(4):
        p = AE + AX * (2.6 + 3.2 * k) + BX * W2
        Fw = Frame(p, AX)
        if Fw.n @ BX < 0: Fw = Frame(p, -AX)
        place(mb, O_WIN, U1, Fw, 0.0, 11.3, 0.12)
        arch_surround(mb, Fw, 1.6, 2.75, 11.2, 0.0, 0.18)
    # cartouches between east drum windows
    for a in (-13.5, 13.5, -40, 40):
        p, o = ell_pt(AE, RE, np.radians(a), True)
        t = nrm(np.cross(UP, o))
        Fw = Frame(p, t)
        if Fw.n @ o < 0: Fw = Frame(p, -t)
        ornament(mb, Fw, 0.0, 14.1, 0.02, 'leaf', 1.0 if abs(a) < 20 else 0.75, depth=0.14)


def arch_surround(mb, F, w, h, y0, e0, depth):
    """raised stone band around an arched opening (outline only)"""
    rise = w / 2
    r = w / 2
    cy = y0 + h - r
    ang = np.linspace(np.pi, 0, 17)
    outer = [(-w / 2 - 0.16, y0)] + [(-(r + 0.16) * np.cos(np.pi - a) * -1 if False else (r + 0.16) * np.cos(a), cy + (r + 0.16) * np.sin(a)) for a in ang] + [(w / 2 + 0.16, y0)]
    inner = [(w / 2, y0)] + [(r * np.cos(a), cy + r * np.sin(a)) for a in ang[::-1]] + [(-w / 2, y0)]
    poly = outer + inner
    from shapely.geometry import Polygon as Pg
    g = Pg(poly).buffer(0)
    if g.geom_type == 'MultiPolygon': g = max(g.geoms, key=lambda q: q.area)
    ext = np.array(g.exterior.coords)[:-1]
    mb.poly(O_DET, 'stone', [F.w(p[0], p[1], e0 + depth) for p in ext], F.n)
    ring_faces_world2(mb, F, ext, e0, e0 + depth)


def ring_faces_world2(mb, F, r, e0, e1, mat='stone'):
    r = np.asarray(r)
    area = 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
    for i in range(len(r)):
        a, b = r[i], r[(i + 1) % len(r)]
        d = b - a
        if np.hypot(*d) < 1e-7: continue
        o2 = np.array([d[1], -d[0]])
        if area < 0: o2 = -o2
        mb.quad(O_DET, mat, F.w(a[0], a[1], e0), F.w(b[0], b[1], e0), F.w(b[0], b[1], e1), F.w(a[0], a[1], e1), F.s * o2[0] + UP * o2[1])


def roof(mb):
    """plain-tile roof: course-stepped surface (real tile butt relief) over the hall incl. both half-cone ends"""
    pts, outs = hall_outline(48, EAVE_OH)
    n = len(pts)
    ridge_e = AE + UP * Y_RIDGE; ridge_w = AW + UP * Y_RIDGE
    tops = []
    for p in pts:
        # project onto the ridge segment
        t = np.clip((p - AE) @ AX / np.linalg.norm(AW - AE), 0, 1)
        q = AE + (AW - AE) * t
        tops.append(q + UP * Y_RIDGE)
    tops = np.array(tops)
    E = pts + UP * Y_EAVE
    nc = 46                          # courses
    ts = np.linspace(0, 1, nc + 1)
    delta = 0.028
    T = []; UVs = []
    def S(i, t): return E[i] + (tops[i] - E[i]) * t
    for i in range(n):
        j = (i + 1) % n
        if np.linalg.norm(tops[i] - tops[j]) < 1e-6 and np.linalg.norm(E[i] - E[j]) < 1e-6: continue
        for c in range(nc):
            t0, t1 = ts[c], ts[c + 1]
            a0, b0 = S(i, t0), S(j, t0); a1, b1 = S(i, t1), S(j, t1)
            nrm_ = np.cross(b0 - a0, a1 - a0)
            if np.linalg.norm(nrm_) < 1e-12:
                nrm_ = np.cross(b1 - a1, a1 - a0)
            nn = nrm(nrm_)
            if nn[1] < 0: nn = -nn
            lift = nn * delta
            # sloped course face (butt proud at bottom)
            quad = [a0 + lift, b0 + lift, b1, a1]
            for tri in ([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]):
                tri = np.array(tri)
                tn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                if np.linalg.norm(tn) < 1e-12: continue
                if tn @ nn < 0: tri = tri[::-1]
                T.append(tri)
            # butt riser at t1 (from surface up to the next course's proud edge), facing down-slope
            if c + 1 < nc:
                down = nrm(E[i] - tops[i]) * 0.5 + nrm(E[j] - tops[j]) * 0.5
                quadr = [a1, b1, b1 + lift, a1 + lift]
                for tri in ([quadr[0], quadr[1], quadr[2]], [quadr[0], quadr[2], quadr[3]]):
                    tri = np.array(tri)
                    tn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                    if np.linalg.norm(tn) < 1e-12: continue
                    if tn @ down < 0: tri = tri[::-1]
                    T.append(tri)
    T = np.array(T)
    # uv: u along eaves arc length, v along slope
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1))])
    UV = np.zeros((len(T), 3, 2))
    # approximate uv: planar by tangent of nearest outline point
    cen = (AE + AW) / 2
    for k in range(len(T)):
        tri = T[k]
        c3 = tri.mean(0)
        d = c3 - cen; d[1] = 0
        ang = np.arctan2(d @ BX, d @ AX)
        tan = nrm(np.cross(UP, nrm(d))) if np.linalg.norm(d) > 1e-6 else AX
        UV[k, :, 0] = (tri @ tan) / 1.8
        UV[k, :, 1] = (tri[:, 1] * 1.35) / 1.8
    mb.tris(O_ROOF, 'tiles', T, UV)
    # ridge tiles
    lathe_ridge(mb, ridge_e + UP * 0.0, ridge_w + UP * 0.0, 0.13, 'tiles', O_ROOFD, skirt=0.2, caps=True)
    # hip tiles along the 4 hip lines of the polygonal west end (cover the course steps where the facets meet)
    _, _, _, wcor = west_outline(EAVE_OH)
    for q in wcor[1:-1]:
        lathe_ridge(mb, q + UP * (Y_EAVE + 0.03), ridge_w + UP * 0.02, 0.12, 'tiles', O_ROOFD, skirt=0.15, caps=True)
    # copper gutter along the eaves + soffit closing between cornice and eaves
    gp, go = hall_outline(48, EAVE_OH + 0.02)
    sweep3(mb, O_ROOFD, 'copper_dark', gp + UP * (Y_EAVE - 0.02), go, [(-0.02, 0.0), (0.06, -0.08), (0.14, -0.06), (0.16, 0.06), (0.12, 0.06), (0.11, -0.02), (0.06, -0.04), (0.0, 0.04)], closed=True)
    sp, so = hall_outline(48, 0.0)
    for i in range(len(sp)):
        j = (i + 1) % len(sp)
        a, b = sp[i], sp[j]
        if np.linalg.norm(b - a) < 1e-6: continue
        o = nrm(so[i] + so[j])
        # (eave band at Y_EAVE is already the top of the drum cornice sweep (0.5 m > EAVE_OH) - no duplicate quad here)
    # downpipes at the front corners of the drum
    for ph in (-62, 62):
        a = np.radians(ph)
        p = AE + AX * (-(RE + EAVE_OH) * np.cos(a)) + BX * ((W2 + EAVE_OH) * np.sin(a))
        lathe(mb, O_ROOFD, 'copper_dark', (p[0], p[2]), [(0.055, Y_LB - 0.1), (0.055, Y_EAVE - 0.1), (0.0, Y_EAVE - 0.05)], segs=8)


def lantern(mb):
    c = np.array([-498.6, 0, -309.15])
    cx, cz = c[0], c[2]
    # tiled octagonal base (tapered)
    lathe(mb, O_ROOF, 'tiles', (cx, cz), [(0.95, 21.6), (0.9, 23.2), (0.82, 24.25)], segs=8, smooth=False, uv_scale=(1.8, 1.8))
    lathe(mb, O_ROOFD, 'copper_dark', (cx, cz), [(0.82, 24.2), (0.95, 24.32), (0.95, 24.42), (0.0, 24.43)], segs=8, smooth=False)
    # copper lantern with louvred openings
    lathe(mb, O_ROOF, 'copper_dark', (cx, cz), [(0.62, 24.4), (0.62, 25.55)], segs=8, smooth=False)
    for k in range(8):
        a = 2 * np.pi * (k + 0.5) / 8
        d = np.array([np.cos(a), 0, np.sin(a)]); t = np.array([-np.sin(a), 0, np.cos(a)])
        p = c + d * 0.6
        Fw = Frame(p, t)
        if Fw.n @ d < 0: Fw = Frame(p, -t)
        lbox(mb, O_ROOFD, 'metal_dark', Fw, -0.16, 0.16, 24.6, 25.35, -0.02, 0.0, ('e-',))
        for y in np.arange(24.66, 25.3, 0.11):
            lbox(mb, O_ROOFD, 'copper_dark', Fw, -0.16, 0.16, y, y + 0.03, 0.0, 0.05, ('e-',))
        lbox(mb, O_ROOFD, 'copper_dark', Fw, -0.04, 0.04, 24.4, 25.55, 0.0, 0.09, ('e-',))
    lathe(mb, O_ROOF, 'copper_dark', (cx, cz), [(0.6, 25.5), (0.7, 25.5), (0.78, 25.6), (0.78, 25.68), (0.6, 25.72), (0.5, 25.95), (0.32, 26.2), (0.12, 26.32), (0.05, 26.38)], segs=16, smooth=True)
    lathe(mb, O_ROOFD, 'gold', (cx, cz), [(0.0, 26.35), (0.08, 26.38), (0.1, 26.46), (0.08, 26.54), (0.0, 26.57)], segs=10, smooth=True)
    lathe(mb, O_ROOFD, 'metal_dark', (cx, cz), [(0.02, 26.55), (0.015, 27.3), (0.0, 27.35)], segs=6)


def dormer(mb):
    """stone aedicule dormer on the east half-cone with a round-headed window and scroll volutes"""
    base = AE + AX * (-(RE - 0.15))
    o = -AX; t = nrm(np.cross(UP, o))
    F = Frame(base, t)
    if F.n @ o < 0: F = Frame(base, -t)
    y0, y1 = Y_EAVE - 0.1, 18.4
    w = 0.85
    # body: front, sides, back into the roof
    lbox(mb, O_SH, 'ashlar', F, -w, w, y0, y1 - 0.2, -2.4, 0.0, ('e-', 'y-'))
    hole_win = parts.arched_window_unit(0.7, 1.3, 0.35, mullions=1)
    place(mb, O_WIN, hole_win, F, 0.0, y0 + 0.75, 0.02)
    lbox(mb, O_DET, 'stone', F, -0.5, 0.5, y0 + 0.62, y0 + 0.75, 0.0, 0.12, ('e-',))
    for sx in (-1, 1):
        lbox(mb, O_DET, 'stone', F, sx * w - 0.12, sx * w + 0.12, y0, y1 - 0.2, 0.0, 0.12, ('e-',))
        ornament(mb, F, sx * (w + 0.18), y0 + 0.35, 0.0, 'scroll', 0.42, depth=0.2)
    # curved pediment
    lbox(mb, O_DET, 'stone', F, -w - 0.15, w + 0.15, y1 - 0.2, y1, -2.4, 0.15, ('e-',))
    ang = np.linspace(np.pi, 0, 13)
    arc = [(np.cos(a) * (w + 0.15), y1 + 0.55 * np.sin(a)) for a in ang]
    prof = [F.w(p[0], p[1], 0.15) for p in arc]
    mb.poly(O_DET, 'stone', prof, F.n)
    for i in range(len(arc) - 1):
        p, q = arc[i], arc[i + 1]
        mb.quad(O_DET, 'stone', F.w(p[0], p[1], 0.15), F.w(p[0], p[1], -2.4), F.w(q[0], q[1], -2.4), F.w(q[0], q[1], 0.15), F.s * (p[0] + q[0]) + UP * 2)
    ornament(mb, F, 0.0, y1 + 0.25, 0.15, 'rosette', 0.35, depth=0.06)
    lathe(mb, O_DET, 'stone', tuple(F.w(0, 0, 0.0)[[0, 2]]), [(0.12, y1 + 0.55), (0.14, y1 + 0.75), (0.06, y1 + 0.95), (0.0, y1 + 1.0)], segs=10)


# ------------------------------------------------------------------ Kestner glass front + low block
K0 = np.array([-496.17, 0.0, -318.38]); K1 = np.array([-497.43, 0.0, -340.64])
FK = Frame(K0, nrm(K1 - K0))


def kestner(mb):
    L = np.linalg.norm(K1 - K0)
    F = FK
    floor = -0.08
    ent0, ent1 = 8.6, 13.6
    rd = 11.1
    # plinth wall (sandstone) left/right of the entrance steps
    for a, b in ((0.0, ent0), (ent1, L)):
        lbox(mb, O_SH, 'ashlar', F, a, b, YG, 0.5, -0.4, 0.0, ('e-', 'y-', 'y+'))
        lbox(mb, O_SH, 'stone', F, a - 0.02, b + 0.02, 0.5, 0.58, -0.47, 0.04, ('e-',))        # cap incl. its soffit (overhang); runs 2 cm under the glass (no slit at the glass foot)
    font.extrude_text(mb, O_LET, F, 'KESTNER GESELLSCHAFT', 0.6, -0.05, 0.0, 0.12, 0.004, 'stone_dark', 'stone_dark', stroke=0.12, spacing=0.4)
    font.extrude_text(mb, O_LET, F, 'GOSERIEDE 11', L - 0.6, -0.05, 0.0, 0.12, 0.004, 'stone_dark', 'stone_dark', stroke=0.12, spacing=0.4, align='right')
    # steps: the pavement rises toward the building (heightfield), so every tread is placed >= 4 cm above the highest
    # ground point under its visible part; treads that would end up buried / too close to the next one are merged
    nst = 5
    HGT = anz.HGT
    gmax = lambda e0, e1: max(HGT(*F.w(s, 0, e)[[0, 2]]) for s in np.linspace(ent0, ent1, 9) for e in np.linspace(e0, e1, 4))
    fronts = [2.2 - k * 0.38 for k in range(nst)]
    tops = [-0.85 + (floor + 0.85) * (k + 1) / nst for k in range(nst)]
    steps = []
    y_next = None
    for k in reversed(range(nst)):
        e_back = fronts[k + 1] if k + 1 < nst else -0.4
        y = max(tops[k], gmax(e_back, fronts[k] + 0.05) + 0.04)
        if y_next is not None and y > y_next - 0.06: continue        # would be (nearly) level with the tread above -> pavement
        steps.insert(0, (fronts[k], y)); y_next = y
    yprev = YG
    for j, (e1, y) in enumerate(steps):    # stacked step blocks (no coplanar side faces); top face = visible tread only
        lbox(mb, O_SH, 'stone', F, ent0, ent1, yprev, y, -0.4, e1, ('e-', 'y-', 'y+'))
        eb = steps[j + 1][0] if j + 1 < len(steps) else -0.4
        mb.quad(O_SH, 'stone', F.w(ent0, y, e1), F.w(ent1, y, e1), F.w(ent1, y, eb), F.w(ent0, y, eb), UP)
        yprev = y
    y_low = steps[0][1]; e_low = steps[0][0]
    for x in (ent0 + 0.25, ent1 - 0.25):
        lbox(mb, O_DET, 'steel', F, x - 0.02, x + 0.02, y_low, y_low + 0.9, e_low - 0.15, e_low - 0.1)
        lbox(mb, O_DET, 'steel', F, x - 0.02, x + 0.02, floor, floor + 0.9, 0.1, 0.15)
        a = F.w(x, y_low + 0.9, e_low - 0.12); b = F.w(x, floor + 0.9, 0.12)
        tube(mb, a, b, 0.025, 'steel')
    # glass curtain wall
    y_g0, y_g1, y_f = 0.58, 8.55, Y_LB
    mb.quad(O_SH, 'glass_green', F.w(0, y_g0, -0.45), F.w(L, y_g0, -0.45), F.w(L, y_g1, -0.45), F.w(0, y_g1, -0.45), F.n)
    # entrance part: glass down to floor
    mb.quad(O_SH, 'glass_green', F.w(ent0, floor, -0.44), F.w(ent1, floor, -0.44), F.w(ent1, y_g0, -0.44), F.w(ent0, y_g0, -0.44), F.n)
    ncol = int(L / 1.45)
    for k in range(ncol + 1):
        s = L * k / ncol
        lbox(mb, O_DET, 'alu', F, s - 0.04, s + 0.04, floor if ent0 <= s <= ent1 else y_g0, y_g1, -0.45, -0.3, ('e-',))
    for y in (2.95, 3.6, 4.3, 6.4):
        lbox(mb, O_DET, 'alu', F, 0.0, L, y - 0.035, y + 0.035, -0.45, -0.32, ('e-',))
    # printed glass band (ref goseriede/06): silver-grey artwork print over the full width between the 2.95 and 4.3
    # transoms (behind the mullions, 2 cm in front of the glass)
    mb.quad(O_DET, 'print_band', F.w(0.02, 2.985, -0.43), F.w(L - 0.02, 2.985, -0.43), F.w(L - 0.02, 4.265, -0.43), F.w(0.02, 4.265, -0.43), F.n)
    # fascia with letters
    lbox(mb, O_SH, 'metal_dark', F, -0.05, L + 0.05, y_g1, y_f, -0.45, 0.05, ('e-', 'y-', 'y+'))     # top hidden under the coping
    lbox(mb, O_SH, 'copper_dark', F, -0.08, L + 0.08, y_f, y_f + 0.06, -0.6, 0.1, ())
    mb.quad(O_SH, 'metal_dark', F.w(-0.05, y_g1, 0.05), F.w(L + 0.05, y_g1, 0.05), F.w(L + 0.05, y_g1, -0.45), F.w(-0.05, y_g1, -0.45), -UP)
    font.extrude_text(mb, O_LET, F, 'KESTNER GESELLSCHAFT', L / 2, 9.05, 0.05, 0.5, 0.03, 'steel', 'steel', stroke=0.11, spacing=0.55, cond=1.0, align='center')
    # small white lettering on the glass left of the entrance (ref goseriede/06): CAFE / BUECHER / CINEMATHEQUE,
    # placed inside the last full glass panel before the entrance (clear of the mullions)
    pw = L / ncol
    k0 = int(np.floor((ent0 - 0.05) / pw)) - 1
    s_txt = pw * k0 + 0.12
    for i, line in enumerate(('CAFÉ', 'BÜCHER', 'CINÉMATHÈQUE')):
        w = font.extrude_text(mb, O_LET, F, line, s_txt, 1.95 - i * 0.17, -0.448, 0.1, 0.006, 'paint_white', 'paint_white', stroke=0.14, spacing=0.3, cond=0.9)
        if s_txt + w > pw * (k0 + 1) - 0.08: raise SystemExit('Kestner glass lettering does not fit its panel (%.2f m)' % w)
    # revolving door drum
    c = F.w(rd, 0, -0.45)
    # dark bronze drum (ref goseriede/06): dark glass between bronze posts, bronze canopy + floor ring
    lathe(mb, O_SH, 'glass_dark', (c[0], c[2]), [(1.05, floor), (1.05, 2.4)], segs=24, a0=0, a1=2 * np.pi)
    lathe(mb, O_DET, 'bronze', (c[0], c[2]), [(1.0, 2.4), (1.1, 2.4), (1.1, 2.75), (0.0, 2.76)], segs=24)     # incl. soffit ring over the drum glass
    lathe(mb, O_DET, 'bronze', (c[0], c[2]), [(0.0, floor + 0.01), (1.1, floor + 0.01)][::-1], segs=24)
    for k in range(8):
        a = 2 * np.pi * (k + 0.5) / 8
        d = np.array([np.cos(a), 0, np.sin(a)]); t = np.array([-np.sin(a), 0, np.cos(a)])
        q = np.array([c[0], 0, c[2]]) + d * 1.075
        box_world(mb, O_DET, 'bronze', q + UP * ((floor + 2.4) / 2), t, d, 0.035, (2.4 - floor) / 2, 0.035)
    lathe(mb, O_DET, 'bronze', (c[0], c[2]), [(1.075, 0.98), (1.085, 1.0), (1.085, 1.06), (1.075, 1.08)], segs=24)   # push-bar ring
    # side walls of the setback: south return next to the right wing (faces north) and north return between the glass plane (e -0.45) and the north wing's street face (e 0),
    # plinth top up to the fascia (was open: see-through slit when walking north along the Goseriede)
    mb.quad(O_SH, 'ashlar', F.w(L, 0.5, -0.46), F.w(L, 0.5, 0.0), F.w(L, y_g1, 0.0), F.w(L, y_g1, -0.46), -F.s)
    mb.quad(O_SH, 'ashlar', F.w(0.0, 0.5, 0.0), F.w(0.0, 0.5, -0.46), F.w(0.0, y_g1, -0.46), F.w(0.0, y_g1, 0.0), F.s)
    return L


def tube(mb, a, b, r, mat, obj=None):
    d = nrm(b - a); s = nrm(np.cross(d, UP if abs(d[1]) < 0.95 else np.array([1.0, 0, 0]))); u = np.cross(s, d)
    n = 8
    ang = np.linspace(0, 2 * np.pi, n + 1)
    A = [a + (s * np.cos(t) + u * np.sin(t)) * r for t in ang]; B = [b + (s * np.cos(t) + u * np.sin(t)) * r for t in ang]
    for i in range(n):
        mid = (A[i] + A[i + 1]) / 2 - a
        mb.quad(obj or O_DET, mat, A[i], B[i], B[i + 1], A[i + 1], mid)


def low_block(mb):
    """flat roof at Y_LB over the LoD2 low-block polygon; west walls with windows; rooftop boxes"""
    big = [r for r in S['bad']['roofs'] if abs(np.array(r)[:, 1].max() - 10.06) < 0.02][0]
    P = np.array(big)[:, [0, 2]]
    # cut out the right wing's slate pavilion roof (eave 8.9 < 10.06) so it is not hidden under the flat roof
    cut = Polygon([FB.w(u, 0, e)[[0, 2]] for u, e in ((13.40, 1.5), (18.98, 1.5), (18.98, -3.6), (13.40, -3.6))])
    R = Polygon(P).buffer(0).difference(cut)
    for g in (getattr(R, 'geoms', None) or [R]):
        if g.is_empty or g.area < 0.05: continue
        ext = np.array(g.exterior.coords)[:-1]
        hol = [[np.array([q[0], Y_LB, q[1]]) for q in np.array(h.coords)[:-1]] for h in g.interiors]
        mb.poly(O_SH, 'roof_flat', [np.array([q[0], Y_LB, q[1]]) for q in ext], UP, holes=hol)
    n = len(P)
    lb = Polygon(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        mid = (a + b) / 2
        d = b - a; L = np.linalg.norm(d)
        if L < 0.05: continue
        o2 = np.array([d[1], -d[0]]) / L
        if lb.contains(Point(mid + o2 * 0.2)): o2 = -o2
        # west-facing edges (parking side) and the small north edge near x -505.9 (exposed above the north wing? no)
        if mid[0] < -521.0 and o2[0] < -0.5:
            A = np.array([a[0], 0, a[1]]); B = np.array([b[0], 0, b[1]])
            t = nrm(B - A)
            F = Frame(A, t)
            if F.n @ np.array([o2[0], 0, o2[1]]) < 0:
                F = Frame(B, -t)
            nax = max(1, int(L / 3.2))
            axes = [L * (k + 0.5) / nax for k in range(nax)]
            holes = []
            for sc in axes:
                for (ya, yb) in ((1.2, 3.4), (5.4, 7.8)):
                    holes.append((sc - 0.6, sc + 0.6, ya, yb))
                    place(mb, O_WIN, parts.window_unit(1.2, yb - ya, 'grid', bars=(2, 3)), F, sc, ya, -0.22)
            wall_grid(mb, F, 0.0, L, YG, Y_LB, holes, O_SH, 'ashlar', None, 0.22, 'ashlar', 'stone')
            lbox(mb, O_DET, 'stone', F, 0.0, L, 0.22, 0.38, 0.0, 0.06, ('e-',))
        elif mid[0] < -512.0 and not (o2[1] < -0.9 and mid[1] < -339.0):
            # short jogs of the west wall + exposed south edge next to the Anzeiger annex: plain ashlar (closes see-through slits)
            A = np.array([a[0], 0, a[1]]); B = np.array([b[0], 0, b[1]])
            on = np.array([o2[0], 0, o2[1]])
            mb.quad(O_SH, 'ashlar', A + UP * YG, B + UP * YG, B + UP * Y_LB, A + UP * Y_LB, on)
        elif o2[1] > 0.9 and mid[0] < -503.0 and mid[1] > -299.0:
            # light well between the Anzeiger tower and the Goseriedebad: south wall of the low block above the
            # Anzeiger NW annex roof (y 4.0) up to the roof, coping + flashing (was open: QA 'light well' hole)
            A = np.array([a[0], 0, a[1]]); B = np.array([b[0], 0, b[1]])
            on = np.array([o2[0], 0, o2[1]])
            F = Frame(A, nrm(B - A))
            if F.n @ on < 0: F = Frame(B, nrm(A - B))
            nax = max(1, int(L / 3.2))
            axes = [L * (k + 0.5) / nax for k in range(nax)]
            holes = []
            for sc in axes:
                holes.append((sc - 0.5, sc + 0.5, 5.6, 7.8))
                place(mb, O_WIN, parts.window_unit(1.0, 2.2, 'grid', bars=(2, 3)), F, sc, 5.6, -0.22)
            wall_grid(mb, F, 0.0, L, 3.9, Y_LB, holes, O_SH, 'ashlar', None, 0.22, 'ashlar', 'stone')
            lbox(mb, O_DET, 'copper_dark', F, 0.0, L, 3.98, 4.12, 0.0, 0.05, ('e-',))         # flashing on the annex roof
    # copper coping along the free low-block edges (west walls with their small jogs + south edges up to the Anzeiger)
    # as ONE mitred sweep -> no overlapping coping boxes at the corners (z-fighting)
    k0 = int(np.argmin(np.hypot(P[:, 0] + 524.23, P[:, 1] + 340.86))); k1 = int(np.argmin(np.hypot(P[:, 0] + 503.69, P[:, 1] + 297.92)))
    chain = [P[(k0 + j) % n] for j in range(((k1 - k0) % n) + 1)]
    W3 = [np.array([q[0], Y_LB, q[1]]) for q in chain]
    segn = []
    for j in range(len(W3) - 1):
        t = nrm(W3[j + 1] - W3[j]); o = np.cross(t, UP)
        m2 = (chain[j] + chain[j + 1]) / 2
        if lb.contains(Point(m2[0] + o[0] * 0.2, m2[1] + o[2] * 0.2)): o = -o
        segn.append(o)
    outs = [segn[0]] + [nrm(segn[j - 1] + segn[j]) / (nrm(segn[j - 1] + segn[j]) @ segn[j]) for j in range(1, len(segn))] + [segn[-1]]
    sweep3(mb, O_SH, 'copper_dark', W3, outs, [(-0.08, 0.06), (-0.08, -0.02), (0.06, -0.02), (0.06, 0.06), (-0.08, 0.06)])
    # street-side coping on the front parts at Y_LB (bay top & recess) and parapet of the bay roof
    # rooftop boxes from LoD2 (12.42, 10.35)
    for r in S['bad']['roofs']:
        p = np.array(r); h = p[:, 1].max()
        if abs(h - 12.42) < 0.02 or abs(h - 10.35) < 0.02 or abs(h - 13.09) < 0.02:
            Q = p[:, [0, 2]]
            pg = Polygon(Q).buffer(0)
            Qe = np.array(pg.exterior.coords)[:-1]
            cen = Qe.mean(0)
            ac = abs(h - 13.09) < 0.02
            hb = Y_LB + 1.35 if ac else h            # 13.09 = LoD2 top of the AC units -> plant box + fans (DOP)
            for i in range(len(Qe)):
                a, b = Qe[i], Qe[(i + 1) % len(Qe)]
                mid = (a + b) / 2 - cen
                mb.quad(O_SH, 'plaster' if not ac else 'alu', [a[0], Y_LB - 0.1, a[1]], [b[0], Y_LB - 0.1, b[1]], [b[0], hb, b[1]], [a[0], hb, a[1]], [mid[0], 0, mid[1]])
            mb.poly(O_SH, 'roof_flat' if not ac else 'alu', [[q[0], hb, q[1]] for q in Qe], UP)
            if ac:
                # 2 x 2 fans with guards on top (DOP: 4 round fans)
                ax_ = nrm(np.array([Qe[1][0] - Qe[0][0], 0, Qe[1][1] - Qe[0][1]])); bx_ = nrm(np.array([Qe[3][0] - Qe[0][0], 0, Qe[3][1] - Qe[0][1]]))
                la = np.hypot(*(Qe[1] - Qe[0])); lb_ = np.hypot(*(Qe[3] - Qe[0]))
                o = np.array([Qe[0][0], 0, Qe[0][1]])
                for fa in (0.25, 0.75):
                    for fb in (0.25, 0.75):
                        cc = o + ax_ * la * fa + bx_ * lb_ * fb
                        lathe(mb, O_DET, 'metal_dark', (cc[0], cc[2]), [(0.5, hb), (0.5, hb + 0.35), (0.44, hb + 0.35), (0.0, hb + 0.25)], segs=20, smooth=False)
            elif abs(h - 12.42) < 0.02:
                kestner_roof_glazing(mb, Qe, h)
            continue


def kestner_roof_glazing(mb, Qe, h, width=3.0):
    """DOP 2025: long dark glazed strips (3 m wide) along the west and east sides of the raised Kestner hall roof
    -> lean-to skylights from the box top down to an upstand on the low roof, aluminium ends, glazing bars every 1.2 m"""
    cen = Qe.mean(0)
    yb = Y_LB + 0.45
    for i in range(4):
        a2, b2 = Qe[i], Qe[(i + 1) % 4]
        d = b2 - a2; L = np.hypot(*d)
        if L < 15.0: continue                   # long sides only (west / east)
        t = np.array([d[0], 0, d[1]]) / L
        o = np.cross(t, UP)
        if o @ np.array([(a2 + b2)[0] / 2 - cen[0], 0, (a2 + b2)[1] / 2 - cen[1]]) < 0: o = -o
        A = np.array([a2[0], 0, a2[1]]); B = np.array([b2[0], 0, b2[1]])
        Ao, Bo = A + o * width, B + o * width
        top_a, top_b = A + UP * (h - 0.05), B + UP * (h - 0.05)
        bot_a, bot_b = Ao + UP * yb, Bo + UP * yb
        sv = bot_a - top_a
        ng = nrm(np.cross(t, sv))
        if ng[1] < 0: ng = -ng
        mb.quad(O_SH, 'glass', top_a, top_b, bot_b, bot_a, ng)
        # upstand + ends (alu)
        mb.quad(O_SH, 'alu', Ao + UP * (Y_LB - 0.05), Bo + UP * (Y_LB - 0.05), bot_b, bot_a, o)
        for P0, P1, sg in ((A, Ao, -1), (B, Bo, 1)):
            mb.poly(O_SH, 'alu', [P0 + UP * (Y_LB - 0.05), P1 + UP * (Y_LB - 0.05), P1 + UP * yb, P0 + UP * (h - 0.05)], t * sg)
        # glazing bars (raised 4 cm) incl. both edge bars, ridge flashing
        nb = int(L / 1.2)
        for k in range(nb + 1):
            c = min(max(L * k / nb, 0.04), L - 0.04)
            p0 = top_a + t * c; p1 = bot_a + t * c
            for sg in (-1, 1):
                q0 = p0 + t * 0.03 * sg; q1 = p1 + t * 0.03 * sg
                mb.quad(O_DET, 'metal_dark', q0, q1, q1 + ng * 0.04, q0 + ng * 0.04, t * sg)
            mb.quad(O_DET, 'metal_dark', p0 - t * 0.03 + ng * 0.04, p0 + t * 0.03 + ng * 0.04, p1 + t * 0.03 + ng * 0.04, p1 - t * 0.03 + ng * 0.04, ng)
        Fr = Frame(A, t)
        if Fr.n @ o < 0: Fr = Frame(B, -t)
        lbox(mb, O_DET, 'alu', Fr, -0.02, L + 0.02, h - 0.12, h + 0.05, -0.01, 0.25, ())


def north_roof_windows(mb):
    """DOP 2025: 4 large roof windows on each slope of the north wing + 2 small ones on the west hip.
    Flush roof windows: white frame raised 7 cm above the LoD2 roof plane, dark glass 4 cm above it, 3 mullions."""
    roofs = []
    for r in S['bad']['roofs']:
        p = np.array(r); n = geo.newell(p)
        if np.linalg.norm(n) < 1e-9: continue
        n = nrm(n)
        if n[1] < 0: n = -n
        if 0.2 < n[1] < 0.97: roofs.append((Polygon(p[:, [0, 2]]).buffer(0), n, p[0]))
    wins = [((-538.25, -532.75), (-346.75, -344.25)), ((-529.5, -524.25), (-346.75, -344.25)), ((-521.0, -515.6), (-346.75, -344.25)), ((-512.5, -507.1), (-346.75, -344.25)),
            ((-537.75, -532.25), (-361.0, -358.5)), ((-529.1, -523.6), (-361.0, -358.5)), ((-520.5, -515.0), (-361.0, -358.5)), ((-511.9, -506.5), (-361.0, -358.5)),
            ((-543.0, -542.0), (-359.7, -358.3)), ((-543.0, -542.0), (-351.7, -350.3))]
    for (x0, x1), (z0, z1) in wins:
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        hit = [r for r in roofs if r[0].contains(Point(cx, cz))]
        if not hit: continue
        poly, n, p0 = hit[0]
        yat = lambda x, z: p0[1] - (n[0] * (x - p0[0]) + n[2] * (z - p0[2])) / n[1]
        # local frame on the roof plane: t horizontal along the eave, sdir up-slope
        hz = np.array([n[0], 0, n[2]]); hz = nrm(hz)
        t = nrm(np.cross(UP, hz))
        corners = []
        for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
            corners.append(np.array([x, yat(x, z), z]))
        C = np.array(corners)
        # rectangle in roof plane coords
        sdir = nrm(np.cross(n, t))
        if sdir[1] < 0: sdir = -sdir
        cc = C.mean(0)
        uu = (C - cc) @ t; vv = (C - cc) @ sdir
        u0, u1, v0, v1 = uu.min(), uu.max(), vv.min(), vv.max()
        Pw = lambda u, v, d: cc + t * u + sdir * v + n * d
        fw = 0.09
        # frame ring (top + outer + inner sides), glass, mullions
        ring_o = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        ring_i = [(u0 + fw, v0 + fw), (u1 - fw, v0 + fw), (u1 - fw, v1 - fw), (u0 + fw, v1 - fw)]
        mb.poly(O_ROOFD, 'paint_white', [Pw(u, v, 0.07) for u, v in ring_o], n, holes=[[Pw(u, v, 0.07) for u, v in ring_i]])
        for ring, sgn in ((ring_o, 1), (ring_i, -1)):
            for k in range(4):
                (ua, va), (ub, vb) = ring[k], ring[(k + 1) % 4]
                mid = np.array([(ua + ub) / 2, (va + vb) / 2])
                hint = (t * mid[0] + sdir * mid[1]) * sgn
                mb.quad(O_ROOFD, 'paint_white', Pw(ua, va, -0.02), Pw(ub, vb, -0.02), Pw(ub, vb, 0.07), Pw(ua, va, 0.07), hint)
        mb.poly(O_ROOFD, 'glass', [Pw(u, v, 0.04) for u, v in ring_i], n)
        nm = max(1, int(round((u1 - u0) / 1.4)))
        for k in range(1, nm):
            um = u0 + (u1 - u0) * k / nm
            for sg in (-1, 1):
                mb.quad(O_ROOFD, 'paint_white', Pw(um + sg * 0.03, v0 + fw, 0.04), Pw(um + sg * 0.03, v1 - fw, 0.04), Pw(um + sg * 0.03, v1 - fw, 0.065), Pw(um + sg * 0.03, v0 + fw, 0.065), t * sg)
            mb.quad(O_ROOFD, 'paint_white', Pw(um - 0.03, v0 + fw, 0.065), Pw(um + 0.03, v0 + fw, 0.065), Pw(um + 0.03, v1 - fw, 0.065), Pw(um - 0.03, v1 - fw, 0.065), n)


def north_wing(mb):
    """LoD2-derived north wing (radio ffn / former bath halls) + 4.7 m annex; windows on street faces"""
    ring = Polygon(np.array(S['bad']['grounds'][0])[:, [0, 2]]).buffer(0)
    keep_roofs, keep_walls = [], []
    for r in S['bad']['roofs']:
        p = np.array(r); c = p.mean(0); h = p[:, 1].max()
        if abs(h - 10.06) < 0.02: continue
        if c[2] < -340.0 or (c[0] < -529 and c[2] < -336.5):
            keep_roofs.append(p)
    for w in S['bad']['walls']:
        p = np.array(w); c = p.mean(0)
        if c[2] < -340.4 or (c[0] < -529.5 and c[2] < -336.8 and c[2] > -342):
            keep_walls.append(p)
    for p in keep_roofs:
        n = geo.newell(p)
        if np.linalg.norm(n) < 1e-9: continue
        slope = abs(n[1]) / np.linalg.norm(n) < 0.97
        mb.poly(O_ROOF if slope else O_SH, 'tiles' if slope else 'roof_flat', p, UP)
    nwin = 0
    for p in keep_walls:
        n = geo.newell(p)
        if np.linalg.norm(n) < 1e-9: continue
        n = nrm(n); n[1] = 0; n = nrm(n)
        c = p.mean(0)
        y0, y1 = p[:, 1].min(), p[:, 1].max()
        on = ring_normal(p) if y0 < 1.0 else None
        if on is not None:
            n = on                                 # footprint wall: outward normal of the footprint edge (robust for 0.2 m jogs)
        elif ring.buffer(0.05).contains(Point(c[0] + n[0] * 0.4, c[2] + n[2] * 0.4)) and not ring.buffer(-0.3).contains(Point(c[0] - n[0] * 0.4, c[2] - n[2] * 0.4)):
            n = -n
        # horizontal extent
        xz = p[:, [0, 2]]
        t = nrm(np.array([n[2], 0, -n[0]]))
        s = (p - p[0]) @ t
        L = s.max() - s.min()
        street = (n[0] > 0.7 and c[0] > -500) or (n[2] < -0.7 and c[2] < -360) or (n[0] < -0.7 and c[0] < -540)
        if street and L > 4.0 and y1 - y0 > 6.0 and y0 < 0:
            A = p[np.argmin(s)]; A = np.array([A[0], 0, A[2]])
            F = Frame(A, t)
            if F.n @ n < 0: F = Frame(A + t * L, -t)
            nax = max(1, int(L / 3.0))
            axes = [L * (k + 0.5) / nax for k in range(nax)]
            # gable walls: the windowed rectangle stops at the eave; the gable triangle above follows the LoD2 outline
            sl = (p - F.O) @ F.s
            sy = np.stack([sl, p[:, 1]], 1)
            y_e = min(max((b for a, b in sy if abs(a) < 0.3), default=y1), max((b for a, b in sy if abs(a - L) < 0.3), default=y1))
            if y1 - y_e > 0.3:
                from shapely.geometry import box as sbox
                gp = Polygon(sy).buffer(0).intersection(sbox(-1.0, y_e, L + 1.0, y1 + 1.0))
                for g in (getattr(gp, 'geoms', None) or [gp]):
                    if g.is_empty or g.area < 1e-3: continue
                    ring_ = np.array(g.exterior.coords)[:-1]
                    mb.poly(O_SH, 'ashlar', [F.w(a, b, 0.0) for a, b in ring_], F.n)
                y1 = y_e
            holes = []
            rows = [(1.2, 3.3), (5.0, 7.4)] + ([(8.9, 11.0)] if y1 > 12.5 else [])
            for sc in axes:
                for (ya, yb) in rows:
                    holes.append((sc - 0.6, sc + 0.6, ya, yb)); nwin += 1
                    place(mb, O_WIN, parts.window_unit(1.2, yb - ya, 'grid', bars=(2, 3)), F, sc, ya, -0.22)
            wall_grid(mb, F, 0.0, L, min(YG, y0), y1, holes, O_SH, 'ashlar', None, 0.22, 'ashlar', 'stone')
            lbox(mb, O_DET, 'stone', F, 0.0, L, 0.22, 0.38, 0.0, 0.06, ('e-',))
            lbox(mb, O_SH, 'stone', F, -0.02, L + 0.02, y1 - 0.35, y1 + 0.03, 0.0, 0.18, ('e-',))
            # inner face of the cornice where it stands 3 cm above the flat / tiled roof (closes a grazing slit)
            mb.quad(O_SH, 'stone', F.w(L + 0.02, y1, 0.0), F.w(-0.02, y1, 0.0), F.w(-0.02, y1 + 0.03, 0.0), F.w(L + 0.02, y1 + 0.03, 0.0), -F.n)
        else:
            P2 = p.copy()
            if y0 < 1.0: P2[P2[:, 1] < y0 + 1e-6, 1] = YG          # every footprint wall reaches 1.3 m below the LoD2 ground
            mb.poly(O_SH, 'ashlar' if c[2] > -364 else 'plaster', P2, n)
    return nwin


def ring_normal(p, tol=0.12):
    """outward normal (x,0,z) of the Goseriedebad footprint edge under the wall polygon p (None if p is not on the footprint)"""
    from shapely.geometry.polygon import orient
    E = np.array(orient(Polygon(np.array(S['bad']['grounds'][0])[:, [0, 2]]).buffer(0), 1.0).exterior.coords)
    b = p[p[:, 1] < p[:, 1].min() + 1e-3][:, [0, 2]]
    if len(b) < 2: return None
    m = b.mean(0)
    best = None
    for a, q in zip(E[:-1], E[1:]):
        d = q - a; L = np.hypot(*d)
        if L < 1e-6: continue
        t = np.clip((m - a) @ d / L ** 2, 0, 1)
        dist = np.hypot(*(a + d * t - m))
        if best is None or dist < best[0]:
            best = (dist, np.array([d[1], 0.0, -d[0]]) / L)       # CCW ring in (x, z): outward = (dz, -dx)
    return best[1] if best[0] < tol else None


def build(mb):
    left_wing(mb)
    recess(mb)
    bay(mb)
    right_wing(mb)
    hall(mb)
    kestner(mb)
    low_block(mb)
    north_wing(mb)
    north_roof_windows(mb)


if __name__ == '__main__':
    mb = MB(); build(mb)
    print(mb.total())
    for o, d in mb.stats().items(): print(o, d)

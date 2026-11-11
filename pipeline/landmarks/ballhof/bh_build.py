"""Ballhof-Ensemble (Ballhofplatz, Hannover) -> build/ballhof.glb (Spielframe, unkomprimiert) + build/ballhof_stats.json
Gebaeude: Ballhof eins (Ballhaus 1649-64) mit Foyer-Pavillon und Ostfluegel, Hinterhaus, Spittahaus (Ballhofplatz 7 /
Burgstrasse 23), Ballhofplatz 1 (Burgstrasse 21A, [hof:gefluester]), Ballhofplatz 2 (Teestuebchen, Efeu), Ballhofplatz 3.
Masse: city-LoD2 (Grundriss, Trauf-/Firsthoehen), Details: Fotos (refs.json, landmark_notes.md).
Aufruf: python bh_build.py [out.glb]"""
import json, os, sys, math, time
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from bh_mesh import MB, Frame, WORLD, unit, area2, ccw, offset_poly, tri_poly, UP
from bh_roof import Plane, RoofSet, RoofBuilder, as_polys
import bh_parts as BP
import bh_core as C
from bh_core import Wall, rect, fachwerk, gable_framing, quoins, surround, plinth, add_window, H
import bh_glb

T0 = time.time()
HERE = C.HERE
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'build', 'ballhof.glb')
TEX = os.path.join(HERE, 'build', 'tex')
OUTL = json.load(open(os.path.join(HERE, 'build', 'outlines.json')))
S8 = lambda *c: tuple(v / 255.0 for v in c)

MATS = {
    'Putz_weiss': dict(color=S8(240, 237, 228), tex='plaster', rough=0.92),
    'Putz_creme': dict(color=S8(243, 226, 210), tex='plaster', rough=0.92),
    'Putz_rosa': dict(color=S8(200, 142, 114), tex='plaster', rough=0.92),
    'Putz_bordeaux': dict(color=S8(146, 60, 74), tex='plaster', rough=0.9),
    'Putz_hell': dict(color=S8(222, 216, 203), tex='plaster', rough=0.93),
    'Holz_dunkel': dict(color=S8(66, 47, 36), tex='wood', rough=0.75),
    'Holz_braun': dict(color=S8(92, 52, 37), tex='wood', rough=0.75),
    'Holz_rotbraun': dict(color=S8(104, 50, 36), tex='wood', rough=0.75),
    'Holz_schwarz': dict(color=S8(44, 34, 30), tex='wood', rough=0.75),
    'Quader': dict(color=(1, 1, 1), tex='ashlar', rough=0.93, nscale=1.0),
    'Quader_hell': dict(color=(1, 1, 1), tex='ashlar_light', rough=0.93),
    'Sandstein': dict(color=(1, 1, 1), tex='dressed', rough=0.85),
    'Ziegel': dict(color=(1, 1, 1), tex='tiles', rough=0.8),
    'Ziegel_alt': dict(color=(1, 1, 1), tex='tiles_old', rough=0.82),
    'Ziegel_braun': dict(color=(1, 1, 1), tex='tiles_brown', rough=0.82),
    'Schiefer': dict(color=(1, 1, 1), tex='slate', rough=0.6),
    'Blech_tuerkis': dict(color=S8(104, 168, 156), tex='seam', rough=0.45, metal=0.35),
    'Zink': dict(color=S8(160, 166, 170), tex='zinc', rough=0.45, metal=0.6),
    'Fensterrahmen': dict(color=S8(238, 238, 232), rough=0.5),
    'Fensterglas': dict(color=S8(30, 38, 44), rough=0.06, metal=0.1),
    'Schaufenster': dict(color=S8(44, 54, 60), rough=0.06, metal=0.1),
    'Tuer_holz': dict(color=S8(112, 76, 50), tex='planks', rough=0.7),
    'Tuer_gruen': dict(color=S8(46, 98, 80), tex='planks', rough=0.6),
    'Tuer_dunkel': dict(color=S8(62, 46, 38), tex='planks', rough=0.7),
    'Eisen': dict(color=S8(38, 38, 38), rough=0.5, metal=0.6),
    'Laterne_gruen': dict(color=S8(30, 72, 54), rough=0.45, metal=0.4),
    'Laternenglas': dict(color=S8(236, 228, 196), rough=0.1, emit=S8(40, 34, 20)),
    'Stahl_dunkel': dict(color=S8(50, 58, 60), rough=0.45, metal=0.5),
    'Dachbahn': dict(color=(1, 1, 1), tex='roofing', rough=0.85),
    'Kies': dict(color=(1, 1, 1), tex='gravel', rough=0.95),
    'Efeu': dict(color=(1, 1, 1), tex='ivy', rough=0.7, alpha_mode='MASK', cutoff=0.5),
    'Farbe_rot': dict(color=S8(156, 44, 36), rough=0.6),
    'Schilder': dict(color=(1, 1, 1), tex='signs', rough=0.6),
    'Markise': dict(color=S8(236, 232, 220), tex='plaster', rough=0.85),
    'Traufe': dict(color=S8(74, 56, 44), tex='planks', rough=0.8),
    'Bronze': dict(color=S8(112, 86, 52), rough=0.4, metal=0.7),
    'Glasdach': dict(color=S8(120, 140, 146), rough=0.12, metal=0.2),
    'col': dict(color=(1, 0, 1)),
}
TILES = {'Putz_weiss': 2.0, 'Putz_creme': 2.0, 'Putz_rosa': 2.0, 'Putz_bordeaux': 2.0, 'Putz_hell': 2.0, 'Markise': 2.0,
         'Holz_dunkel': 1.0, 'Holz_braun': 1.0, 'Holz_rotbraun': 1.0, 'Holz_schwarz': 1.0,
         'Quader': 4.0, 'Quader_hell': 3.0, 'Sandstein': 1.0, 'Ziegel': (1.05, 1.2), 'Ziegel_alt': (1.05, 1.2),
         'Ziegel_braun': (1.05, 1.2), 'Schiefer': 1.0, 'Blech_tuerkis': 1.0, 'Zink': 1.0, 'Tuer_holz': 1.0, 'Tuer_gruen': 1.0,
         'Tuer_dunkel': 1.0, 'Dachbahn': 2.0, 'Kies': 2.0, 'Efeu': 1.0, 'Schilder': 1.0, 'Traufe': 1.0}

ctx = C.Ctx()
TOP = [0.0]
STATS = {}


def P2(p):
    return np.asarray(p, float)


def lerp(a, b, t):
    return P2(a) + (P2(b) - P2(a)) * t


def line_x(p0, p1, q0, q1):
    """Schnittpunkt der Geraden p0p1 und q0q1 (Plan)"""
    p0, p1, q0, q1 = map(P2, (p0, p1, q0, q1))
    d1 = p1 - p0; d2 = q1 - q0
    den = d1[0] * d2[1] - d1[1] * d2[0]
    t = ((q0[0] - p0[0]) * d2[1] - (q0[1] - p0[1]) * d2[0]) / den
    return p0 + d1 * t



def halfspace(p, q, side, L=400.0):
    """grosses Polygon auf der Seite 'side' der Geraden p-q"""
    p, q, side = P2(p), P2(q), P2(side)
    d = unit(q - p); n = np.array([-d[1], d[0]])
    if (side - p) @ n < 0: n = -n
    return Polygon([p - d * L, p + d * L, p + d * L + n * L, p - d * L + n * L])


def inward(pts, p0, p1, d):
    """Punkt um d nach innen von der Kantenmitte"""
    p0, p1 = P2(p0), P2(p1)
    t = unit(p1 - p0); n = np.array([-t[1], t[0]])
    m = (p0 + p1) / 2
    if not Polygon(pts).contains(Point(m + n * 0.2)): n = -n
    return m + n * d


# ============================================================================================== Profil / Gebaeude
def profile(rs, pL, pR, inset=0.05):
    F = Frame.facade(pL, pR)
    L = float(np.linalg.norm(P2(pR) - P2(pL)))
    a0p = F.P([0, 0, -inset]); a1p = F.P([L, 0, -inset])
    ln = LineString([(a0p[0], a0p[2]), (a1p[0], a1p[2])])
    ts = {0.0, L}
    geoms = [g for g in rs.pp.values()] + [r for r, _ in rs.parts]
    for g in geoms:
        for pg in as_polys(g):
            it = ln.intersection(pg.boundary)
            for q in getattr(it, 'geoms', [it]):
                if q.is_empty: continue
                for c in q.coords:
                    ts.add(float(ln.project(Point(c))))
    out = []
    for t in sorted(ts):
        for tt in ((t - 0.003, t + 0.003) if 0 < t < L else (t,)):
            tt = min(max(tt, 0.0), L)
            p = F.P([tt, 0, -inset])
            out.append((tt, rs.h_near(p[0], p[2])))
    # kollineare Punkte entfernen
    res = [out[0]]
    for i in range(1, len(out) - 1):
        a, b, c = res[-1], out[i], out[i + 1]
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-4 and b[0] - a[0] > 1e-4: continue
        res.append(b)
    res.append(out[-1])
    return res


def building(B):
    """B: key, pts, rs, zones [dict(top, rec, mat, rec_e={})], party {e: mat}, decor(wall, e, k), roof opts, ...
    Baut Waende (je Zone/ Kante), Auskragungs-Untersichten, Dach, Innenwaende, Kollision."""
    pts = np.array(B['pts'], float)
    n = len(pts)
    rs = B['rs']
    zones = B['zones']
    party = B.get('party', {})
    yg = []
    for e in range(n):
        p0, p1 = pts[e], pts[(e + 1) % n]
        s = np.linspace(0, 1, 12)
        xs = p0[0] + (p1[0] - p0[0]) * s; zs = p0[1] + (p1[1] - p0[1]) * s
        yg.append(float(np.min(H(xs, zs))) - 0.35)
    B['yg'] = yg
    outl = []
    for k, z in enumerate(zones):
        rec = np.array([0.0 if e in party else z.get('rec_e', {}).get(e, z['rec']) for e in range(n)])
        outl.append(offset_poly(pts, -rec) if np.any(rec) else pts.copy())
    walls = {}
    for e in range(n):
        pL, pR = pts[(e + 1) % n], pts[e]
        ref = Frame.facade(pL, pR)
        Lref = float(np.linalg.norm(pR - pL))
        if e in party:
            prof = profile(rs, pL, pR)
            poly = [(0, yg[e]), (Lref, yg[e])] + [(t, h - 0.03) for t, h in prof[::-1]]
            w = Wall(ctx, ref, poly, party[e], ref=ref, L=Lref, edge=e, zone='party', bld=B)
            if B.get('decor'): B['decor'](w, e, 'party')
            w.emit()
            continue
        for k, z in enumerate(zones):
            qL, qR = outl[k][(e + 1) % n], outl[k][e]
            F = Frame.facade(qL, qR)
            L = float(np.linalg.norm(qR - qL))
            yb = yg[e] if k == 0 else zones[k - 1]['top']
            if z['top'] is None:
                prof = profile(rs, qL, qR)
                poly = [(0, yb), (L, yb)] + [(t, h - 0.03) for t, h in prof[::-1]]
            else:
                yt = z['top']
                poly = rect(0, yb, L, yt)
            w = Wall(ctx, F, poly, z['mat'], ref=ref, L=L, edge=e, zone=k, bld=B)
            if B.get('decor'): B['decor'](w, e, k)
            w.emit()
            walls[(e, k)] = w
            # Auskragung: Untersicht zwischen Zone k (zurueckgesetzt) und k+1
            if k + 1 < len(zones):
                a0, a1 = outl[k][e], outl[k][(e + 1) % n]
                b0, b1 = outl[k + 1][e], outl[k + 1][(e + 1) % n]
                if np.linalg.norm(a0 - b0) > 0.01 or np.linalg.norm(a1 - b1) > 0.01:
                    y = z['top']
                    Q = [np.array([p[0], y, p[1]]) for p in (b0, b1, a1, a0)]
                    nrm = np.cross(Q[1] - Q[0], Q[2] - Q[0])
                    if nrm[1] > 0: Q = Q[::-1]
                    ctx.mb.quad(B.get('soffit_mat', 'Holz_dunkel'), *Q)
    # Dach
    ro = dict(B.get('roof', {}))
    ro.setdefault('party_lines', [[tuple(pts[e]), tuple(pts[(e + 1) % n])] for e in party if B.get('party_roof', True)])
    rb = RoofBuilder(rs, ctx.mb, ro, ctx.inst).build()
    OLb = Polygon(pts).exterior
    rb.gutters(skip=lambda A, Bp: OLb.distance(Point((A[0] + Bp[0]) / 2, (A[2] + Bp[2]) / 2)) > 0.9 or B.get('no_gutter', False))
    B['rb'] = rb
    # Innenwaende
    C.internal_walls(ctx, rs, pts, B.get('int_mat', lambda k, s: B['zones'][-1]['mat']), decor=B.get('int_decor'))
    # Kollision: Prisma bis Traufe + Dachflaechen
    yb = min(yg)
    eave = min(rs.h_near(p[0], p[1]) for p in pts)
    for e in range(n):
        p0, p1 = pts[e], pts[(e + 1) % n]
        ctx.col.quad('col', [p1[0], yb, p1[1]], [p0[0], yb, p0[1]], [p0[0], eave, p0[1]], [p1[0], eave, p1[1]])
    for p, g in rs.pp.items():
        for pg in as_polys(g):
            ex = np.array(pg.exterior.coords)[:-1]
            if area2(ex) < 0: ex = ex[::-1]
            V, I = tri_poly(ex)
            P = np.c_[V[:, 0], p.h(V[:, 0], V[:, 1]), V[:, 1]]
            # Plan-CCW (x,z) -> Normale nach unten; umdrehen
            ctx.col.tris('col', P, I[:, ::-1], V)
    TOP[0] = max(TOP[0], max(pl.h(*np.array(g.representative_point().coords[0])) for pl, g in rs.pp.items()))
    return walls


def zwerch(pts, e, a_c, w, depth, y_e, slope, ov=None):
    """Zwerchhaus/Gaube als Dachteil an Kante e: Front im Abstand 'set' hinter der Aussenwand. Rueckgabe (region, planes, meta)"""
    raise NotImplementedError


def gable_part(pL, pR, a_c, w, setback, depth, y_e, slope, ov_e=0.15, ov_v=0.12):
    """Satteldach-Gaube / Zwerchhaus an der Fassade pL->pR (von aussen gesehen), Mitte bei a_c, Breite w,
    Front 'setback' hinter der Wandflucht, Tiefe 'depth', Traufhoehe y_e, Neigung slope."""
    F = Frame.facade(pL, pR)
    c0, c1 = -setback, -setback - depth
    q = lambda a, c: F.P([a, 0, c])[[0, 2]]
    FL, FR = q(a_c - w / 2, c0), q(a_c + w / 2, c0)
    BL, BR = q(a_c - w / 2, c1), q(a_c + w / 2, c1)
    region = Polygon([FL, FR, BR, BL])
    if area2(np.array(region.exterior.coords)[:-1]) < 0: region = Polygon([FL, BL, BR, FR])
    mid = q(a_c, c0)
    pl = Plane.eave(FL, BL, y_e, slope, mid, 'zw_l'); pl.ov_e, pl.ov_v = ov_e, ov_v
    pr = Plane.eave(FR, BR, y_e, slope, mid, 'zw_r'); pr.ov_e, pr.ov_v = ov_e, ov_v
    return region, [pl, pr], dict(F=F, a_c=a_c, w=w, setback=setback, y_e=y_e, slope=slope)


# ============================================================================================== Prototypen
def win_key(name, **kw):
    key = name
    ctx.proto(key, lambda: BP.window(**kw))
    return key


def put_window(wall, a_ref, b0, key, w, h):
    return add_window(wall, a_ref, b0, w, h, key, lambda: None)


def lantern(wall, a_ref, y):
    ctx.proto('lantern', BP.wall_lantern)
    a = wall.A(a_ref)
    ctx.place('lantern', wall.F.sub(a, y, 0.0).mat4())


def knaggen(wall, posts, y_top, depth, H_=0.75, red=True):
    key = 'knagge_%d_%d' % (round(depth * 100), round(H_ * 100))
    ctx.proto(key, lambda: BP.knagge(H=H_, D=depth + 0.02, red=red))
    for p in posts:
        a = wall.A(p)
        if wall.amin + 0.1 < a < wall.amax - 0.1:
            ctx.place(key, wall.F.sub(a, y_top, -depth).mat4())


def joists(wall, posts, y_top, depth, mat='Holz_dunkel', red=False, D_out=0.10):
    key = 'joist_%s_%d_%d' % (mat, round(depth * 100), int(red))
    ctx.proto(key, lambda: BP.joist_head(D=depth + D_out, mat=mat, red=red))
    for p in posts:
        a = wall.A(p)
        if wall.amin + 0.1 < a < wall.amax - 0.1:
            ctx.place(key, wall.F.sub(a, y_top, -depth).mat4())


def rhythm(L, bay, margin=0.0):
    n = max(1, int(round((L - 2 * margin) / bay)))
    return [margin + (L - 2 * margin) * i / n for i in range(n + 1)]


def arched_door(wall, a_ref, y0, w, hs, key, **kw):
    ctx.proto(key, lambda: BP.arched_door(w, hs, **kw))
    a = wall.A(a_ref) - w / 2
    wall.hole(BP.arch_points(w, hs) + np.array([a, y0]))
    ctx.place(key, wall.F.sub(a, y0, 0.0).mat4())
    return a


def rect_door(wall, a_ref, y0, w, h, leaf='Tuer_dunkel', dr=0.2, reveal='Sandstein', glass_top=False):
    a = wall.A(a_ref) - w / 2
    wall.hole(rect(a, y0, a + w, y0 + h))
    F = wall.F
    m = ctx.mb
    m.quad(reveal, F.P([a, y0, 0]), F.P([a, y0, -dr]), F.P([a, y0 + h, -dr]), F.P([a, y0 + h, 0]))
    m.quad(reveal, F.P([a + w, y0, -dr]), F.P([a + w, y0, 0]), F.P([a + w, y0 + h, 0]), F.P([a + w, y0 + h, -dr]))
    m.quad(reveal, F.P([a, y0 + h, -dr]), F.P([a + w, y0 + h, -dr]), F.P([a + w, y0 + h, 0]), F.P([a, y0 + h, 0]))
    m.quad('Sandstein', F.P([a, y0, 0]), F.P([a + w, y0, 0]), F.P([a + w, y0, -dr]), F.P([a, y0, -dr]))
    if glass_top:
        m.poly(leaf, F, rect(a, y0, a + w, y0 + h * 0.62), c=-dr + 0.04)
        m.poly('Fensterglas', F, rect(a, y0 + h * 0.62, a + w, y0 + h), c=-dr + 0.03)
        m.box('Fensterrahmen', F, a, a + w, y0 + h * 0.62, y0 + h * 0.62 + 0.06, -dr + 0.03, -dr + 0.08, skip=('-c',))
    else:
        m.poly(leaf, F, rect(a, y0, a + w, y0 + h), c=-dr + 0.04)
        if w > 1.0:
            m.box('Holz_dunkel', F, a + w / 2 - 0.025, a + w / 2 + 0.025, y0, y0 + h, -dr + 0.04, -dr + 0.07, skip=('-c',))
        for yy in (y0 + 0.35 * h, y0 + 0.75 * h):
            m.box('Holz_dunkel', F, a + 0.08, a + w - 0.08, yy, yy + 0.06, -dr + 0.04, -dr + 0.06, skip=('-c',))
    # Griff
    m.box('Eisen', F, a + w * 0.5 + 0.06, a + w * 0.5 + 0.09, y0 + 1.0, y0 + 1.16, -dr + 0.04, -dr + 0.10, skip=('-c',))
    return a


def steps(wall, a_ref, w, y_ground, y_top, n=None, depth=0.32, mat='Sandstein'):
    """Stufen vor einer Tuer (lokal vor der Wand, c > 0)"""
    if n is None: n = max(1, int(round((y_top - y_ground) / 0.17)))
    a = wall.A(a_ref)
    hstep = (y_top - y_ground) / n
    for i in range(n):
        yt = y_top - i * hstep
        c1 = depth * (i + 1)
        wall.ctx.mb.box(mat, wall.F, a - w / 2 - 0.05 * i, a + w / 2 + 0.05 * i, y_ground - 0.25, yt, 0.0, c1, skip=('-c',))
    ctx.col.box('col', wall.F, a - w / 2, a + w / 2, y_ground - 0.2, y_top, 0.0, depth * n, skip=())


def awning(wall, a0, a1, y_top, proj, drop, key=None, mat='Markise', valance=0.28):
    """Markise: schraeges Tuch von der Wand (y_top) nach vorn/unten, Volant mit Schrift"""
    F = wall.F
    p0 = F.P([a0, y_top, 0.02]); p1 = F.P([a1, y_top, 0.02])
    q0 = F.P([a0, y_top - drop, proj]); q1 = F.P([a1, y_top - drop, proj])
    ctx.mb.quad(mat, q0, q1, p1, p0)                    # Oberseite
    ctx.mb.quad(mat, p0, p1, q1, q0)                    # Unterseite
    v0 = F.P([a0, y_top - drop - valance, proj]); v1 = F.P([a1, y_top - drop - valance, proj])
    if key:
        C.sign(ctx, F, a0, y_top - drop - valance, a1 - a0, valance, key, c=proj, depth=0.0)
        ctx.mb.quad(mat, q0, q1, v1, v0) if False else ctx.mb.quad(mat, v1, v0, q0, q1)
    else:
        ctx.mb.quad(mat, v0, v1, q1, q0); ctx.mb.quad(mat, v1, v0, q0, q1)
    for a in (a0, a1):
        ctx.mb.tube('Eisen', [F.P([a, y_top - drop * 0.9 - 0.4, 0.02]), F.P([a, y_top - drop, proj - 0.02])], 0.015, n=4)


def shopfront(wall, a0, a1, y0, h, mullion=1.6, frame='Stahl_dunkel', glass='Schaufenster', door_at=None, dr=0.18):
    """Schaufensterfront: Oeffnung [a0,a1]x[y0,y0+h] (lokal), Pfosten-Raster, Sockel, Tuer"""
    F = wall.F; m = ctx.mb
    wall.hole(rect(a0, y0, a1, y0 + h))
    m.quad('Putz_hell', F.P([a0, y0, 0]), F.P([a0, y0, -dr]), F.P([a0, y0 + h, -dr]), F.P([a0, y0 + h, 0]))
    m.quad('Putz_hell', F.P([a1, y0, -dr]), F.P([a1, y0, 0]), F.P([a1, y0 + h, 0]), F.P([a1, y0 + h, -dr]))
    m.quad('Putz_hell', F.P([a0, y0 + h, -dr]), F.P([a1, y0 + h, -dr]), F.P([a1, y0 + h, 0]), F.P([a0, y0 + h, 0]))
    m.quad('Sandstein', F.P([a0, y0, 0]), F.P([a1, y0, 0]), F.P([a1, y0, -dr]), F.P([a0, y0, -dr]))
    m.poly(glass, F, rect(a0, y0, a1, y0 + h), c=-dr + 0.03)
    n = max(1, int(round((a1 - a0) / mullion)))
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        m.box(frame, F, max(a0, a - 0.04), min(a1, a + 0.04), y0, y0 + h, -dr, -dr + 0.09, skip=('-c',))
    m.box(frame, F, a0, a1, y0, y0 + 0.40, -dr, -dr + 0.10, skip=('-c',))
    m.box(frame, F, a0, a1, y0 + h - 0.08, y0 + h, -dr, -dr + 0.09, skip=('-c',))
    if door_at is not None:
        m.box(frame, F, door_at - 0.5, door_at + 0.5, y0 + 2.15, y0 + 2.22, -dr, -dr + 0.1, skip=('-c',))
        m.box('Eisen', F, door_at + 0.3, door_at + 0.33, y0 + 0.9, y0 + 1.3, -dr + 0.09, -dr + 0.14, skip=('-c',))


# ============================================================================================== 1) Ballhof eins
def build_ballhof():
    pts = [(-410.206, 246.158), (-414.384, 233.029), (-383.79, 222.484), (-379.212, 237.028)]
    EAVE = 7.8
    rA, rB = (-388.539, 231.973), (-405.762, 237.496)        # First (LoD2) 17.83
    RY = 17.83
    rmid = lerp(rA, rB, 0.5)
    N = Plane.p3((pts[1][0], EAVE, pts[1][1]), (pts[2][0], EAVE, pts[2][1]), (rmid[0], RY, rmid[1]), 'N')
    S = Plane.p3((pts[3][0], EAVE, pts[3][1]), (pts[0][0], EAVE, pts[0][1]), (rmid[0], RY, rmid[1]), 'S')
    W = Plane.p3((pts[0][0], EAVE, pts[0][1]), (pts[1][0], EAVE, pts[1][1]), (rB[0], RY, rB[1]), 'W')
    E = Plane.p3((pts[2][0], EAVE, pts[2][1]), (pts[3][0], EAVE, pts[3][1]), (rA[0], RY, rA[1]), 'E')
    rs = RoofSet([(Polygon(pts), [N, S, W, E])], outline=pts)
    YS = 4.20          # Oberkante Steinwand / Unterkante Balkenlage
    REC = 0.40
    gy_n = -0.78
    bay = 1.80
    up = dict(y0=4.42, y1=7.30, sill=5.25, lintel=6.62)

    def decor(w, e, k):
        L = w.L
        posts = rhythm(np.linalg.norm(P2(pts[(e + 1) % 4]) - P2(pts[e])), bay)
        Lref = posts[-1]
        if k == 0:
            # Steinwand
            plinth(w, H(*w.F.P([w.amin, 0, 0])[[0, 2]]) + 0.35, mat='Sandstein')
            if e == 1:
                for dc in (14.35, 18.9, 23.45):
                    a = arched_door(w, dc, float(H(*w.ref.P([dc, 0, 0])[[0, 2]])) + 0.02, 2.0, 1.85, 'door_bh', dr=0.45, reveal='Quader')
                for lc in (12.05, 16.65, 21.15, 25.75):
                    lantern(w, lc, 2.95)
                # Gedenktafel
                a = w.A(28.6)
                ctx.mb.box('Bronze', w.F, a - 0.25, a + 0.25, 1.55, 2.25, 0.0, 0.025, skip=('-c',))
                C.sign(ctx, w.F, a - 0.21, 1.6, 0.42, 0.6, 'plaque', c=0.025, depth=0.0)
                quoins(w, w.amax, gy_n, YS, -1, mat='Quader_hell', course=0.42, long=0.75, short=0.45)
            if e == 0:
                quoins(w, w.amin, gy_n, YS, +1, mat='Quader_hell', course=0.42, long=0.75, short=0.45)
                # Nebentuer am Westende (vom Pavillon aus)
            # Gesims auf der Steinwand
            w.member(rect(w.amin - 0.03, YS - 0.16, w.amax + 0.03, YS), (1, 0), 'Sandstein', c0=-0.02, c1=0.08, bevel=0.02, prio=-1)
        else:
            # Obergeschoss: Balkenlage + Schwelle + Fensterband
            mat = 'Holz_dunkel'
            w.member(rect(w.amin - 0.05, YS, w.amax + 0.05, up['y0']), (1, 0), mat, c0=-0.02, c1=0.03, prio=0)
            wins = [((posts[i] + posts[i + 1]) / 2, 1.05) for i in range(len(posts) - 1)]
            fachwerk(w, up['y0'], up['y1'], posts, wins, up['sill'], up['lintel'],
                     dict(mat=mat, brace='none', hs=0.24, hr=0.25, wp=0.22, hb=0.15))
            # profilierte Schwelle: roter Begleitstrich + Wulst
            w.member(rect(w.amin, up['y0'] + 0.09, w.amax, up['y0'] + 0.125), (1, 0), 'Farbe_rot', c0=0.03, c1=0.042, bevel=0.0, prio=9)
            w.member(rect(w.amin, up['y1'] - 0.06, w.amax, up['y1'] - 0.025), (1, 0), 'Farbe_rot', c0=0.03, c1=0.042, bevel=0.0, prio=9)
            key = win_key('w_bh', w=1.05, h=up['lintel'] - up['sill'], rows=3, cas=2, dr=0.10, reveal='Putz_weiss', sill='wood')
            for c, ww in wins:
                put_window(w, c, up['sill'], key, ww, up['lintel'] - up['sill'])
            knaggen(w, posts, YS, REC, H_=0.78)
            joists(w, posts, up['y0'], 0.0, mat=mat, D_out=0.07)
    B = dict(key='BH', pts=pts, rs=rs, zones=[dict(top=YS, rec=REC, mat='Quader'), dict(top=None, rec=0.0, mat='Putz_weiss')],
             decor=decor, roof=dict(tile_mat='Ziegel_alt', eave=0.38, verge=0.25, course=0.30, ridge_proto='ridge_alt'),
             soffit_mat='Holz_dunkel')
    ctx.proto('ridge_alt', lambda: BP.ridge_tile(mat='Ziegel_alt'))
    building(B)
    # Firstspitzen (Wetterstangen) an beiden Firstenden
    for r in (rA, rB):
        Fz = Frame([r[0], RY + 0.12, r[1]], [1, 0, 0], [0, 1, 0], [0, 0, 1])
        ctx.mb.lathe('Eisen', Fz, [(0.06, 0.0), (0.05, 0.25), (0.035, 0.30), (0.02, 1.2), (0.045, 1.25), (0.045, 1.32), (0.012, 1.36), (0.008, 2.1), (0.0, 2.15)], n=8)
    TOP[0] = max(TOP[0], RY + 2.3)
    return B


# ============================================================================================== 2) Ostfluegel (+ Vorbau)
def build_ost():
    O = OUTL['OST']
    pts = [tuple(p) for p in O['pts']]
    # Kanten: e0 Kopfbau-West, e2 Hauptfassade West, e3 Nord, e4..e7 Ost (Brandwand/Hinterhaus), e8 Ballhof
    low = [(-393.921, 220.925), (-392.17, 225.897), (-383.79, 222.484), (-385.349, 217.777)]
    main_poly = Polygon(pts).difference(Polygon(low).buffer(0.0)).buffer(0)
    main_poly = max(as_polys(main_poly), key=lambda g: g.area)
    RY = 12.08
    rN, rS = (-400.117, 190.559), (-391.078, 216.151)
    rmid = lerp(rN, rS, 0.5)
    W = Plane.p3((-394.444, 6.85, 220.341), (-405.175, 6.85, 190.062), (rmid[0], RY, rmid[1]), 'W')
    E = Plane.p3((-390.837, 6.40, 202.072), (-383.79, 6.40, 222.484), (rmid[0], RY, rmid[1]), 'E')
    Sh = Plane.p3((-393.921, 5.90, 220.925), (-385.349, 5.90, 217.777), (rS[0], RY, rS[1]), 'S')
    Nh = Plane.p3((-405.175, 6.85, 190.062), (-398.084, 6.85, 183.899), (rN[0], RY, rN[1]), 'N')
    NE = Plane.eave((-398.084, 183.899), (-393.156, 190.446), 6.40, 49.0, rmid, 'NE')
    lw_r0, lw_r1 = (-389.826, 219.421), (-388.214, 224.286)
    LW = Plane.p3((-393.921, 3.20, 220.925), (-392.17, 3.20, 225.897), (lw_r0[0], 5.50, lw_r0[1]), 'LW')
    LE = Plane.p3((-385.349, 2.98, 217.777), (-383.79, 2.98, 222.484), (lw_r0[0], 5.50, lw_r0[1]), 'LE')
    parts = [(main_poly, [W, E, Sh, Nh, NE]), (Polygon(low), [LW, LE])]
    # Gauben auf der Westseite (Fassade e2: pL = Nordende)
    e2L, e2R = P2((-405.175, 190.062)), P2((-394.444, 220.341))
    dormers = []
    for a in (5.0, 10.5, 16.0, 21.5, 27.0):
        reg, pls, meta = gable_part(e2L, e2R, a, 1.5, 0.95, 2.6, 8.75, 48.0, ov_e=0.14, ov_v=0.12)
        parts.append((reg, pls)); dormers.append(meta)
    rs = RoofSet(parts, outline=pts)
    YS = 2.40; REC = 0.30
    bay = 1.60
    up = dict(y0=2.62, y1=5.77, sill=3.55, lintel=5.0)
    gy = -1.0

    def decor(w, e, k):
        Lref = float(np.linalg.norm(P2(pts[(e + 1) % len(pts)]) - P2(pts[e])))
        if e == 0 or e == 1:
            # niedriger Zwischenbau: Bruchstein bis unters Dach, vier kleine Fenster
            w.mat = 'Quader'
            if e == 0 and k == 0:
                plinth(w, -0.6)
                key = win_key('w_ost_klein', w=0.62, h=0.95, rows=2, cas=2, dr=0.22, reveal='Quader', sill='stone')
                for a in (1.1, 2.0, 3.9, 4.8):
                    a_ = put_window(w, a, 0.15, key, 0.62, 0.95)
                    surround(w, a_, 0.15, a_ + 0.62, 1.10, w=0.12, sill=0.0)
            return
        if e == 2:
            posts = rhythm(Lref, bay)
            pat = ['P' if i % 3 == 0 else 'W' for i in range(len(posts) - 1)]
            if k == 0:
                plinth(w, -0.55)
                key = win_key('w_ost_eg', w=1.0, h=1.45, rows=2, cas=2, transom=1.05, top_rows=1, dr=0.24, reveal='Quader', sill='stone')
                for i, p in enumerate(pat):
                    if p == 'W':
                        c = (posts[i] + posts[i + 1]) / 2
                        a_ = put_window(w, c, 0.0, key, 1.0, 1.45)
                        surround(w, a_, 0.0, a_ + 1.0, 1.45, w=0.14, sill=0.0)
                w.member(rect(w.amin - 0.03, YS - 0.14, w.amax + 0.03, YS), (1, 0), 'Sandstein', c0=-0.02, c1=0.06, bevel=0.015, prio=-1)
                for a in (8.8, 24.0):
                    lantern(w, a, 2.0)
            else:
                mat = 'Holz_braun'
                wins = [((posts[i] + posts[i + 1]) / 2, 0.98) for i, p in enumerate(pat) if p == 'W']
                fachwerk(w, up['y0'], up['y1'], posts, wins, up['sill'], up['lintel'],
                         dict(mat=mat, brace='V', hs=0.22, hr=0.22, wp=0.19, hb=0.15, bw=0.15, full_brace='V'))
                w.member(rect(w.amin - 0.05, YS, w.amax + 0.05, up['y0']), (1, 0), mat, c0=-0.02, c1=0.03, prio=0)
                key = win_key('w_ost_og', w=0.98, h=up['lintel'] - up['sill'], rows=3, cas=2, dr=0.10, reveal='Putz_weiss', sill='wood')
                for c, ww in wins:
                    put_window(w, c, up['sill'], key, ww, up['lintel'] - up['sill'])
                joists(w, posts, YS + 0.0, REC, mat=mat, red=True, D_out=0.12)
            return
        if e == 3:
            posts = rhythm(Lref, 1.55)
            if k == 0:
                plinth(w, -0.6)
                quoins(w, w.amax, -1.2, YS, -1, mat='Quader_hell')
                a_ = rect_door(w, 3.0, float(H(*w.ref.P([3.0, 0, 0])[[0, 2]])) + 0.03, 1.2, 2.3, leaf='Tuer_dunkel', dr=0.3, reveal='Quader')
                surround(w, a_, -0.9, a_ + 1.2, 1.43, w=0.16, sill=0.0)
                key = win_key('w_ost_eg', w=1.0, h=1.45, rows=2, cas=2, transom=1.05, top_rows=1, dr=0.24, reveal='Quader', sill='stone')
                a_ = put_window(w, 6.6, 0.0, key, 1.0, 1.45); surround(w, a_, 0.0, a_ + 1.0, 1.45, w=0.14, sill=0.0)
            else:
                mat = 'Holz_braun'
                wins = [((posts[i] + posts[i + 1]) / 2, 0.9) for i in range(len(posts) - 1) if i % 2 == 1]
                fachwerk(w, up['y0'], up['y1'], posts, wins, up['sill'], up['lintel'],
                         dict(mat=mat, brace='V', hs=0.22, hr=0.22, wp=0.19, hb=0.15, bw=0.15, full_brace='V'))
                w.member(rect(w.amin - 0.05, YS, w.amax + 0.05, up['y0']), (1, 0), mat, c0=-0.02, c1=0.03, prio=0)
                key = win_key('w_ost_og2', w=0.9, h=up['lintel'] - up['sill'], rows=3, cas=2, dr=0.10, reveal='Putz_weiss', sill='wood')
                for c, ww in wins:
                    put_window(w, c, up['sill'], key, ww, up['lintel'] - up['sill'])
                joists(w, posts, YS, REC, mat=mat, red=True, D_out=0.12)
                # Giebeldreieck (Walm -> nur schmale Wand); nichts weiter
            return
        if w.zone != 'party' and k == 1:
            w.mat = 'Putz_weiss'

    def int_mat(k, s):
        return 'Schiefer' if k >= 2 else 'Quader'

    def int_decor(ctx_, w, k, s):
        if k >= 2:
            m = dormers[k - 2]
            if abs(w.F.ez @ m['F'].ez) > 0.9:          # Gaubenfront
                a = (w.amin + w.amax) / 2
                key = win_key('w_gaube_ost', w=0.62, h=0.78, rows=2, cas=2, dr=0.08, reveal='Fensterrahmen', sill='zinc')
                ybot = w.poly.bounds[1] + 0.25
                add_window(w, w.ref.local(w.F.P([a, 0, 0]))[0], ybot + 0.12, 0.62, 0.78, key, lambda: None)
        else:
            # Wand zwischen Kopfbau und Hauptfluegel: Fachwerk-Brandwand, schlicht verputzt
            w.mat = 'Putz_weiss'
    B = dict(key='OST', pts=pts, rs=rs,
             zones=[dict(top=YS, rec=REC, mat='Quader', rec_e={0: 0.0, 1: 0.0}), dict(top=None, rec=0.0, mat='Putz_weiss')],
             party={4: 'Putz_hell', 5: 'Putz_hell', 6: 'Putz_hell', 7: 'Putz_hell', 8: 'Quader'}, decor=decor,
             roof=dict(tile_mat='Ziegel_braun', eave=0.45, verge=0.22, course=0.30, ridge_proto='ridge_braun'),
             int_mat=int_mat, int_decor=int_decor, soffit_mat='Holz_braun')
    ctx.proto('ridge_braun', lambda: BP.ridge_tile(mat='Ziegel_braun'))
    building(B)
    # Fallrohr an der Suedecke der Westfassade
    rb = B['rb']
    Fw = Frame.facade(e2L, e2R)
    top = Fw.P([31.6, 6.85 - 0.45 * 1.17 - 0.32, 0.45 + 0.08])
    C.downpipe(ctx, top, Fw.P([31.6, 0, 0]), Fw.ez, -1.05)
    # Vorbau Nordspitze (kleiner Windfang)
    V = OUTL['VOR']['pts']
    vp = np.array(V)
    yb = float(np.min(H(vp[:, 0], vp[:, 1]))) - 0.3
    for i in range(4):
        p0, p1 = vp[i], vp[(i + 1) % 4]
        ctx.mb.quad('Putz_hell', [p1[0], yb, p1[1]], [p0[0], yb, p0[1]], [p0[0], 2.1, p0[1]], [p1[0], 2.1, p1[1]])
    Vo = offset_poly(ccw(vp), 0.08)
    Vt, It = tri_poly(Vo)
    ctx.mb.tris('Zink', np.c_[Vt[:, 0], np.full(len(Vt), 2.12), Vt[:, 1]], It[:, ::-1], Vt)
    for i in range(4):
        p0, p1 = Vo[i], Vo[(i + 1) % 4]
        ctx.mb.quad('Zink', [p1[0], 1.98, p1[1]], [p0[0], 1.98, p0[1]], [p0[0], 2.12, p0[1]], [p1[0], 2.12, p1[1]])
    return B


# ============================================================================================== 3) Pavillon + Anbauten (98a969)
def build_pav():
    O = OUTL['PAV']
    raw = [tuple(p) for p in O['pts']]
    # Front zum Platz begradigen (e12..e18 -> eine Gerade)
    i0 = raw.index((-436.337, 242.451)); i1 = raw.index((-413.715, 235.136))
    pts = raw[:i0 + 1] + raw[i1:] if i0 < i1 else raw
    pts = [p for p in pts]
    fL, fR = P2((-413.715, 235.136)), P2((-436.337, 242.451))      # von aussen (Norden) gesehen: links = Ost
    Ff = Frame.facade(fL, fR)
    DEP = 5.6
    bk = lambda a: Ff.P([a, 0, -DEP])[[0, 2]]
    # Rueckwand-Linie schneidet Ballhof-West (e0) und Spittahaus-Ost
    pb_e = line_x(bk(0), bk(1), (-414.384, 233.029), (-410.206, 246.158))
    pb_w = line_x(bk(0), bk(1), (-438.177, 239.314), (-429.852, 253.597))
    pav_reg = Polygon([fL, fR, pb_w, pb_e]).buffer(0)
    PV = Plane.eave(fL, fR, 3.95, 10.0, pb_e, 'pav'); PV.ov_e = 0.55; PV.ov_v = 0.30
    glass_reg = Polygon([(-390.159, 243.123), (-386.075, 249.82), (-377.371, 246.982), (-372.379, 244.451), (-371.388, 244.15),
                         (-374.574, 234.349), (-379.212, 237.028)]).buffer(0)
    GD = Plane.p3((-390.159, 4.18, 243.123), (-386.075, 4.18, 249.82), (-374.574, 5.14, 234.349), 'glas')
    full = Polygon(pts).buffer(0)
    flat_reg = full.difference(pav_reg).difference(glass_reg).buffer(0)
    FL = Plane(0, 0, 3.38, 'flach')
    parts = [(pav_reg, [PV]), (glass_reg.intersection(full).buffer(0), [GD])]
    for g in as_polys(flat_reg):
        if g.area > 1.0: parts.append((g, [FL]))
    rs = RoofSet(parts, outline=pts)
    n = len(pts)
    # Kanten bestimmen
    def eidx(p0):
        return pts.index(p0)
    e_front = eidx((-436.337, 242.451))
    party = {}
    for e in range(n):
        p0, p1 = P2(pts[e]), P2(pts[(e + 1) % n])
        mid = (p0 + p1) / 2
        # an Ballhof / Spittahaus / Ostfluegel / Hinterhaus / Burgstr. 23A
        for key in ('BH', 'SP', 'OST', 'HINT'):
            if LineString(OUTL[key]['pts'] + [OUTL[key]['pts'][0]]).distance(Point(mid)) < 0.35:
                party[e] = 'Putz_hell'
        if e == eidx((-426.203, 251.434)): party[e] = 'Putz_hell'

    def decor(w, e, k):
        if e == e_front:
            # Glasfront Foyer "Ballhof eins / junges schauspiel"
            F = w.F; L = w.L
            y0 = float(np.min(H(*np.array([F.P([a, 0, 0])[[0, 2]] for a in np.linspace(0, L, 8)]).T))) + 0.02
            yt = 2.95
            w.mat = 'Stahl_dunkel'
            shopfront(w, 0.25, L - 0.25, y0, yt - y0, mullion=1.95, frame='Stahl_dunkel', glass='Schaufenster', door_at=11.0, dr=0.20)
            # Plakatkaesten
            for a, key in ((1.4, 'poster1'), (4.4, 'poster2'), (17.0, 'poster1')):
                C.sign(ctx, F, a, y0 + 0.45, 1.6, 2.2, key, c=-0.12, depth=0.0)
            # Blende mit Schrift
            C.sign(ctx, F, 1.0, 3.2, 8.0, 0.62, 'junges_schauspiel', c=0.0, depth=0.03)
            C.sign(ctx, F, 12.5, 3.2, 8.4, 0.62, 'junge_oper', c=0.0, depth=0.03)
            # senkrechtes Fahnenschild
            Fb = Frame(F.P([10.6, 0.6, 0.05]), F.ez, UP, -F.ex)
            ctx.mb.box('Stahl_dunkel', Fb, 0.0, 0.34, 0.0, 2.2, -0.06, 0.06, skip=())
            for s in (1, -1):
                Fs = Frame(Fb.P([0.02, 0.05, 0.061 * s]), Fb.ex * s, UP, Fb.ez * s) if s > 0 else Frame(Fb.P([0.32, 0.05, -0.061]), -Fb.ex, UP, -Fb.ez)
                C.sign(ctx, Fs, 0.0, 0.0, 0.30, 2.1, 'ballhof_eins_v', c=0.0, depth=0.0)
            plinth(w, y0 + 0.02, mat='Stahl_dunkel', prot=0.02)
            return
        if e in party: return
        # Rueckseiten: schlichte Putzflaechen mit einzelnen Fenstern
        Lr = w.L
        if Lr > 5.0 and k == 0:
            key = win_key('w_pav', w=1.0, h=1.2, rows=2, cas=2, dr=0.14, reveal='Putz_hell', sill='zinc')
            for c in np.arange(2.0, Lr - 1.5, 3.2):
                put_window(w, c, 1.0, key, 1.0, 1.2)
    B = dict(key='PAV', pts=pts, rs=rs, zones=[dict(top=None, rec=0.0, mat='Putz_hell')], party=party, decor=decor,
             roof=dict(tile_mat='Blech_tuerkis', eave=0.55, verge=0.3, course=None, flat_mat='Kies', wall_mat='Putz_hell',
                       smooth={'pav': 'Blech_tuerkis', 'glas': 'Glasdach'}), soffit_mat='Stahl_dunkel', no_gutter=True,
             int_mat=lambda k, s: 'Stahl_dunkel' if k == 0 else 'Putz_hell')
    building(B)
    return B


# ============================================================================================== 4) Hinterhaus (ff0b07)
def build_hint():
    O = OUTL['HINT']
    pts = [tuple(p) for p in O['pts']]
    full = Polygon(pts).buffer(0)
    cut = Polygon([(-420, 203.4), (-360, 222.6), (-360, 260), (-420, 260)])
    south = full.intersection(cut).buffer(0); north = full.difference(cut).buffer(0)
    parts = [(max(as_polys(north), key=lambda g: g.area), [Plane(0, 0, 7.04, 'n')]), (max(as_polys(south), key=lambda g: g.area), [Plane(0, 0, 2.62, 's')])]
    rs = RoofSet(parts, outline=pts)
    party = {e: 'Putz_hell' for e in range(1, len(pts))}

    def decor(w, e, k):
        if e == 0 and w.L > 10:
            key = win_key('w_pav', w=1.0, h=1.2, rows=2, cas=2, dr=0.14, reveal='Putz_hell', sill='zinc')
            for c in (3.0, 6.5, 10.0):
                put_window(w, c, 3.9, key, 1.0, 1.2)
    B = dict(key='HINT', pts=pts, rs=rs, zones=[dict(top=None, rec=0.0, mat='Putz_hell')], party=party, decor=decor,
             roof=dict(flat_mat='Dachbahn', wall_mat='Putz_hell', course=0.3), no_gutter=True, party_roof=False)
    building(B)
    return B


# ============================================================================================== 5) Spittahaus (Ballhofplatz 7)
def build_sp():
    pts = [tuple(p) for p in OUTL['SP']['pts']]       # e0 Burgstr., e1 Platz, e2 Pavillon, e3 Burgstr. 23A
    rN, rS = (-441.182, 243.734), (-433.567, 255.549)
    RY = 17.25
    rmid = lerp(rN, rS, 0.5)
    W = Plane.p3((pts[0][0], 11.80, pts[0][1]), (pts[1][0], 11.80, pts[1][1]), (rmid[0], RY, rmid[1]), 'W')
    E = Plane.p3((pts[2][0], 11.90, pts[2][1]), (pts[3][0], 11.90, pts[3][1]), (rmid[0], RY, rmid[1]), 'E')
    Nn = Plane.p3((pts[1][0], 12.60, pts[1][1]), (pts[2][0], 12.60, pts[2][1]), (rN[0], RY, rN[1]), 'N')
    parts = [(Polygon(pts), [W, E, Nn])]
    e1L, e1R = P2(pts[2]), P2(pts[1])           # Platzseite: links = Ost (Pavillon)
    reg, pls, zmeta = gable_part(e1L, e1R, 4.83, 5.4, 0.0, 4.2, 12.55, 50.0, ov_e=0.30, ov_v=0.22)
    parts.append((reg, pls))
    e0L, e0R = P2(pts[1]), P2(pts[0])           # Burgstrasse: links = Nord
    dorm = []
    for a in (4.6, 8.0, 11.4, 14.6):
        reg, pls, meta = gable_part(e0L, e0R, a, 1.3, 0.8, 2.4, 13.55, 50.0, ov_e=0.14, ov_v=0.12)
        parts.append((reg, pls)); dorm.append(meta)
    rs = RoofSet(parts, outline=pts)
    YS = 3.15
    z1 = dict(y0=3.15, y1=7.0, sill=4.35, lintel=5.95)
    z2 = dict(y0=7.0, y1=10.55, sill=8.0, lintel=9.55)
    mat = 'Holz_rotbraun'

    def storey(w, posts, wins, z, schwelle_h=0.30, key='w_sp', brace='V'):
        fachwerk(w, z['y0'], z['y1'], posts, wins, z['sill'], z['lintel'],
                 dict(mat=mat, brace=brace, hs=schwelle_h, hr=0.22, wp=0.20, hb=0.15, bw=0.16, full_brace='V'))
        kk = win_key(key, w=0.95, h=z['lintel'] - z['sill'], rows=3, cas=2, transom=None, dr=0.10, reveal='Putz_weiss', sill='wood')
        for c, ww in wins:
            put_window(w, c, z['sill'], kk, 0.95, z['lintel'] - z['sill'])

    def decor(w, e, k):
        Lref = float(np.linalg.norm(P2(pts[(e + 1) % 4]) - P2(pts[e])))
        if e == 3 or w.zone == 'party': return
        if e == 2:
            # zum Pavillon: oberhalb des Pavillondachs Fachwerk mit wenigen Fenstern
            if k == 0: return
            posts = rhythm(Lref, 1.65)
            wins = [((posts[i] + posts[i + 1]) / 2, 0.95) for i in range(len(posts) - 1) if i % 2 == 1]
            if k == 1: storey(w, posts, wins, z1, schwelle_h=0.24, key='w_sp1')
            if k == 2: storey(w, posts, wins, z2, schwelle_h=0.24, key='w_sp2')
            return
        if e == 1:
            posts = rhythm(Lref, Lref / 5)
            wins = [((posts[i] + posts[i + 1]) / 2, 0.95) for i in range(len(posts) - 1) if i != 0]
            if k == 0:
                plinth(w, -0.55)
                yg = float(H(*w.ref.P([1.55, 0, 0.6])[[0, 2]]))
                a = arched_door(w, 1.55, yg + 0.52, 1.15, 1.75, 'door_sp', dr=0.40, reveal='Quader_hell', ring=0.22, straps=False, leaf='Tuer_dunkel')
                steps(w, 1.55, 1.7, yg, yg + 0.52, n=3)
                for c in (5.55, 7.45):
                    kk = win_key('w_sp_eg', w=1.0, h=1.5, rows=2, cas=2, transom=1.05, dr=0.28, reveal='Quader_hell', sill='stone')
                    a_ = put_window(w, c, 0.75, kk, 1.0, 1.5)
                    surround(w, a_, 0.75, a_ + 1.0, 2.25, w=0.16, sill=0.0)
                C.sign(ctx, w.F, w.A(3.45) - 0.33, 1.25, 0.66, 0.66, 'craftbier', c=0.0, depth=0.03)
                lantern(w, 0.6, 2.4)
                quoins(w, w.amax, -1.3, YS, -1, mat='Quader_hell')
            elif k == 1:
                storey(w, posts, wins, z1, key='w_sp1')
                # Inschrift-Schwelle (rote Antiqua auf dunklem Balken)
                C.sign(ctx, w.F, w.amin + 0.05, z1['y0'] + 0.03, w.amax - w.amin - 0.1, 0.24, 'spitta', c=0.035, depth=0.0)
            else:
                storey(w, posts, wins, z2, key='w_sp2')
                # Zwerchhaus-Front (oberhalb z2): Giebelfachwerk + Fenster
                Wz = gable_framing(w, z2['y1'], [z2['y1'], 12.45, 13.9], 1.3,
                                   [(3.4, 0.8, 10.95, 1.15), (4.83, 0.8, 10.95, 1.15), (6.26, 0.8, 10.95, 1.15),
                                    (4.13, 0.7, 12.75, 0.95), (5.53, 0.7, 12.75, 0.95), (4.83, 0.55, 14.2, 0.75)], dict(mat=mat))
                for (ar, ww, b0, hh) in Wz:
                    kk = win_key('w_spg_%d' % round(ww * 100), w=ww, h=hh, rows=2, cas=2, dr=0.08, reveal='Putz_weiss', sill='wood')
                    put_window(w, ar, b0, kk, ww, hh)
                w.member(rect(w.amin - 0.05, z2['y1'], w.amax + 0.05, z2['y1'] + 0.2), (1, 0), mat, prio=0)
            return
        if e == 0:
            posts = rhythm(Lref, Lref / 10)
            wins = [((posts[i] + posts[i + 1]) / 2, 0.95) for i in range(len(posts) - 1) if i not in (0, 4, 9)]
            if k == 0:
                plinth(w, -0.6)
                quoins(w, w.amin, -1.3, YS, +1, mat='Quader_hell')
                for c in (4.0, 6.0, 10.5, 12.5, 15.0):
                    kk = win_key('w_sp_eg', w=1.0, h=1.5, rows=2, cas=2, transom=1.05, dr=0.28, reveal='Quader_hell', sill='stone')
                    a_ = put_window(w, c, 0.75, kk, 1.0, 1.5)
                    surround(w, a_, 0.75, a_ + 1.0, 2.25, w=0.16, sill=0.0)
            elif k == 1:
                storey(w, posts, wins, z1, key='w_sp1')
            else:
                storey(w, posts, wins, z2, key='w_sp2')

    def int_decor(ctx_, w, k, s):
        if k >= 2:
            m = dorm[k - 2]
            if abs(w.F.ez @ m['F'].ez) > 0.9:
                a = (w.amin + w.amax) / 2
                key = win_key('w_gaube_sp', w=0.7, h=0.9, rows=2, cas=2, dr=0.08, reveal='Putz_weiss', sill='zinc')
                ybot = w.poly.bounds[1] + 0.25
                add_window(w, w.ref.local(w.F.P([a, 0, 0]))[0], ybot + 0.1, 0.7, 0.9, key, lambda: None)
                gable_framing(w, ybot + 0.1 + 0.9 + 0.15, [], 9.0, [], dict(mat=mat, raf=0.16))
        elif k == 1:
            gable_framing(w, w.poly.bounds[1] + 0.3, [], 1.4, [], dict(mat=mat))
    B = dict(key='SP', pts=pts, rs=rs,
             zones=[dict(top=YS, rec=0.30, mat='Quader_hell'), dict(top=z1['y1'], rec=0.12, mat='Putz_weiss'), dict(top=None, rec=0.0, mat='Putz_weiss')],
             party={3: 'Putz_hell'}, decor=decor, roof=dict(tile_mat='Ziegel', eave=0.42, verge=0.25, course=0.30),
             int_mat=lambda k, s: 'Putz_weiss', int_decor=int_decor, soffit_mat='Holz_rotbraun')
    building(B)
    return B


# ============================================================================================== 6) Ballhofplatz 1 / Burgstrasse 21A
def build_bp1():
    pts = [tuple(p) for p in OUTL['BP1']['pts']]
    # e0 Brandwand Burgstr. 21 (Nord), e1/e2 Hof, e3 Brandwand BP2, e4 Platz, e5 Burgstrasse
    p = [P2(q) for q in pts]
    EA = 10.1; SL = 47.0
    Wp = Plane.eave(p[5], p[0], EA, SL, (p[2] + p[0]) / 2, 'W')
    EAp = Plane.eave(p[1], p[2], EA, SL, (p[0] + p[5]) / 2, 'EA')
    Sp = Plane.eave(p[4], p[5], EA, SL, (p[3] + p[2]) / 2, 'S')
    NBp = Plane.eave(p[2], p[3], EA, SL, (p[4] + p[5]) / 2, 'NB')
    c_A = line_x(p[1], p[2], p[4], p[5])
    c_B = line_x(p[2], p[3], p[5], p[0])
    regA = Polygon([p[0], p[1], c_A, p[5]]).buffer(0)
    regB = Polygon([c_B, p[3], p[4], p[5]]).buffer(0)
    parts = [(regA, [Wp, EAp, Sp]), (regB, [Sp, NBp, Wp])]
    e4L, e4R = p[5], p[4]          # Platz: links = Ecke Burgstrasse
    e5L, e5R = p[0], p[5]          # Burgstrasse: links = Nord
    reg, pls, zmeta = gable_part(e4L, e4R, 8.2, 6.0, 0.0, 6.0, 12.45, 45.0, ov_e=0.32, ov_v=0.22)
    parts.append((reg, pls))
    dorm = []
    for a in (2.9, 13.1, 14.95, 16.8):
        reg, pls, meta = gable_part(e4L, e4R, a, 1.25, 0.55, 2.4, 11.75, 50.0, ov_e=0.14, ov_v=0.12)
        parts.append((reg, pls)); dorm.append(meta)
    for a in (4.4, 7.9, 11.4, 14.9, 18.4):
        reg, pls, meta = gable_part(e5L, e5R, a, 1.25, 0.55, 2.4, 11.75, 50.0, ov_e=0.14, ov_v=0.12)
        parts.append((reg, pls)); dorm.append(meta)
    rs = RoofSet(parts, outline=pts)
    YS = 3.2
    z1 = dict(y0=3.2, y1=6.45, sill=4.15, lintel=5.6)
    z2 = dict(y0=6.45, y1=9.65, sill=7.35, lintel=8.8)
    mat = 'Holz_schwarz'

    def storey(w, posts, wins, z, key, hs=0.24):
        fachwerk(w, z['y0'], z['y1'], posts, wins, z['sill'], z['lintel'],
                 dict(mat=mat, brace='FB', hs=hs, hr=0.22, wp=0.19, hb=0.15, bw=0.14, full_brace='K'))
        kk = win_key(key, w=0.92, h=z['lintel'] - z['sill'], rows=3, cas=2, transom=z['lintel'] - z['sill'] - 0.36, top_rows=1, dr=0.10, reveal='Putz_creme', sill='wood')
        for c, ww in wins:
            put_window(w, c, z['sill'], kk, 0.92, z['lintel'] - z['sill'])
        # gruene Begleitlinie an der Schwelle (Foto 14)
        w.member(rect(w.amin, z['y0'] + 0.03, w.amax, z['y0'] + 0.06), (1, 0), 'Tuer_gruen', c0=0.03, c1=0.041, bevel=0.0, prio=9)

    def eg_window(w, c, y0=0.55, ww=0.95, hh=2.05, key='w_bp1_eg'):
        kk = win_key(key, w=ww, h=hh, rows=3, cas=2, transom=hh - 0.55, top_rows=1, dr=0.26, reveal='Putz_rosa', sill='stone')
        a_ = put_window(w, c, y0, kk, ww, hh)
        surround(w, a_, y0, a_ + ww, y0 + hh, w=0.15, sill=0.0)

    def decor(w, e, k):
        Lref = float(np.linalg.norm(p[(e + 1) % 6] - p[e]))
        if e == 3: return
        if e == 0:
            if w.zone == 'party':
                # Nordgiebel ueber Burgstr. 21: Fachwerkgiebel mit zwei Fenstern
                w.mat = 'Putz_creme'
                gw = gable_framing(w, 10.0, [10.0, 12.2], 1.35, [(Lref / 2 - 1.0, 0.8, 10.6, 1.1), (Lref / 2 + 1.0, 0.8, 10.6, 1.1), (Lref / 2, 0.6, 12.6, 0.85)], dict(mat=mat))
                for (ar, ww, b0, hh) in gw:
                    kk = win_key('w_bp1_g%d' % round(ww * 100), w=ww, h=hh, rows=2, cas=2, dr=0.08, reveal='Putz_creme', sill='wood')
                    put_window(w, ar, b0, kk, ww, hh)
            return
        if e in (1, 2):
            # Hofseiten: schlichteres Fachwerk
            posts = rhythm(Lref, 1.7)
            wins = [((posts[i] + posts[i + 1]) / 2, 0.92) for i in range(len(posts) - 1) if i % 2 == 0]
            if k == 0:
                for c, _ in wins: eg_window(w, c, y0=0.7, hh=1.6, key='w_bp1_hof')
            elif k == 1: storey(w, posts, wins, z1, 'w_bp1_1')
            else: storey(w, posts, wins, z2, 'w_bp1_2')
            return
        if e == 5:
            posts = rhythm(Lref, Lref / 13)
            wins = [((posts[i] + posts[i + 1]) / 2, 0.92) for i in range(len(posts) - 1)]
            if k == 0:
                plinth(w, -0.5, mat='Sandstein', prot=0.05)
                yg = float(H(*w.ref.P([2.6, 0, 0.5])[[0, 2]]))
                a = arched_door(w, 2.6, yg + 0.05, 1.9, 1.95, 'door_bp1', dr=0.35, reveal='Putz_rosa', ring=0.24, straps=False, leaf='Tuer_gruen')
                for c, _ in wins[3:]:
                    eg_window(w, c)
                quoins(w, w.amax, -1.3, YS, -1, mat='Sandstein', course=0.38, long=0.55, short=0.32, prot=0.03)
                C.sign(ctx, w.F, w.A(6.6) - 0.3, 2.4, 0.3, 0.3, 'craftbier', c=0.0, depth=0.0) if False else None
            elif k == 1: storey(w, posts, wins, z1, 'w_bp1_1')
            else: storey(w, posts, wins, z2, 'w_bp1_2')
            return
        if e == 4:
            posts = rhythm(Lref, Lref / 10)
            wins = [((posts[i] + posts[i + 1]) / 2, 0.92) for i in range(len(posts) - 1)]
            if k == 0:
                plinth(w, -0.5, mat='Sandstein', prot=0.05)
                quoins(w, w.amin, -1.3, YS, +1, mat='Sandstein', course=0.38, long=0.55, short=0.32, prot=0.03)
                # grosses Sprossenfenster
                kk = win_key('w_bp1_gross', w=1.55, h=2.15, rows=5, cas=2, bars_v=True, dr=0.26, reveal='Putz_rosa', sill='stone')
                a_ = put_window(w, 2.7, 0.45, kk, 1.55, 2.15)
                surround(w, a_, 0.45, a_ + 1.55, 2.6, w=0.17, sill=0.0)
                yg = float(H(*w.ref.P([4.95, 0, 0.5])[[0, 2]]))
                a_ = rect_door(w, 4.95, yg + 0.04, 1.05, 2.45, leaf='Tuer_dunkel', dr=0.28, reveal='Sandstein', glass_top=True)
                surround(w, a_, yg + 0.04, a_ + 1.05, yg + 2.49, w=0.18, sill=0.0)
                C.sign(ctx, w.F, a_ - 0.05, yg + 2.55, 1.15, 0.29, 'hofgefluester', c=0.03, depth=0.03)
                for c in (7.4, 9.3, 11.2, 13.4, 15.3):
                    eg_window(w, c)
                lantern(w, 3.9, 2.55); lantern(w, 6.15, 2.55)
            elif k == 1: storey(w, posts, wins, z1, 'w_bp1_1')
            else:
                storey(w, posts, wins, z2, 'w_bp1_2')
                gw = gable_framing(w, z2['y1'], [z2['y1'], 12.35], 1.25,
                                   [(6.6, 0.85, 10.15, 1.3), (8.2, 0.85, 10.15, 1.3), (9.8, 0.85, 10.15, 1.3), (8.2, 0.75, 12.8, 1.05)], dict(mat=mat))
                for (ar, ww, b0, hh) in gw:
                    kk = win_key('w_bp1_z%d' % round(ww * 100), w=ww, h=hh, rows=3, cas=2, dr=0.08, reveal='Putz_creme', sill='wood')
                    put_window(w, ar, b0, kk, ww, hh)

    def int_decor(ctx_, w, k, s):
        if k >= 3:
            m = dorm[k - 3]
            if abs(w.F.ez @ m['F'].ez) > 0.9:
                a = (w.amin + w.amax) / 2
                key = win_key('w_gaube_bp1', w=0.66, h=0.92, rows=3, cas=2, dr=0.08, reveal='Putz_creme', sill='zinc')
                ybot = w.poly.bounds[1] + 0.25
                add_window(w, w.ref.local(w.F.P([a, 0, 0]))[0], ybot + 0.08, 0.66, 0.92, key, lambda: None)
                gable_framing(w, ybot + 0.08 + 0.92 + 0.1, [ybot + 0.08 + 0.92 + 0.1], 9.0, [], dict(mat=mat, raf=0.15))
                w.member(rect(w.amin, ybot - 0.3, w.amin + 0.16, ybot + 1.2), (0, 1), mat, prio=1)
                w.member(rect(w.amax - 0.16, ybot - 0.3, w.amax, ybot + 1.2), (0, 1), mat, prio=1)
        elif k == 2:
            gable_framing(w, w.poly.bounds[1] + 0.3, [], 1.5, [], dict(mat=mat))
    B = dict(key='BP1', pts=pts, rs=rs,
             zones=[dict(top=YS, rec=0.28, mat='Putz_rosa'), dict(top=z1['y1'], rec=0.12, mat='Putz_creme'), dict(top=None, rec=0.0, mat='Putz_creme')],
             party={0: 'Putz_hell', 3: 'Putz_hell'}, decor=decor, roof=dict(tile_mat='Ziegel', eave=0.40, verge=0.24, course=0.30),
             int_mat=lambda k, s: 'Putz_creme', int_decor=int_decor, soffit_mat='Holz_schwarz')
    building(B)
    # Fallrohre (Foto 14: an der Ecke und an der Burgstrasse)
    for (L_, R_, a) in ((e5L, e5R, 22.4), (e4L, e4R, 17.5), (e5L, e5R, 9.7)):
        F = Frame.facade(L_, R_)
        top = F.P([a, EA - 0.40 * 1.07 - 0.30, 0.40 + 0.08])
        C.downpipe(ctx, top, F.P([a, 0, 0]), F.ez, float(H(*F.P([a, 0, 0.3])[[0, 2]])))
    return B


# ============================================================================================== 7) Ballhofplatz 2 (Teestuebchen)
def build_bp2():
    pts = [tuple(p) for p in OUTL['BP2']['pts']]       # e0 Hof, e1 Brandwand BP3, e2 Platz, e3 Brandwand BP1
    p = [P2(q) for q in pts]
    Sp = Plane.eave(p[2], p[3], 11.0, 50.2, (p[0] + p[1]) / 2, 'S')
    Np = Plane.eave(p[0], p[1], 11.04, 50.2, (p[2] + p[3]) / 2, 'N')
    parts = [(Polygon(pts), [Sp, Np])]
    e2L, e2R = p[3], p[2]           # Platz: links = West (BP1)
    reg, pls, zmeta = gable_part(e2L, e2R, 7.6, 6.4, 0.0, 6.5, 11.0, 58.3, ov_e=0.3, ov_v=0.22)
    parts.append((reg, pls))
    e0L, e0R = p[1], p[0]
    dorm = []
    for a in (3.5, 7.6, 11.7):
        reg, pls, meta = gable_part(e0L, e0R, a, 1.3, 0.6, 2.4, 12.6, 50.0, ov_e=0.14, ov_v=0.12)
        parts.append((reg, pls)); dorm.append(meta)
    rs = RoofSet(parts, outline=pts)
    rows = [(3.95, 1.45), (7.25, 1.45)]

    def decor(w, e, k):
        Lref = float(np.linalg.norm(p[(e + 1) % 4] - p[e]))
        if w.zone == 'party': return
        key = win_key('w_bp2', w=0.86, h=1.45, rows=3, cas=2, dr=0.12, reveal='Putz_hell', sill='wood')
        holes = []
        if e == 2:
            cs = list(np.linspace(0.9, Lref - 0.9, 11))
            for (b0, hh) in rows:
                for c in cs:
                    a_ = put_window(w, c, b0, key, 0.86, hh); holes.append(rect(a_, b0, a_ + 0.86, b0 + hh))
            # Zwerchgiebel-Fenster
            kz = win_key('w_bp2_z', w=0.8, h=1.2, rows=3, cas=2, dr=0.10, reveal='Putz_hell', sill='wood')
            for c, b0 in ((6.2, 11.35), (7.6, 11.35), (9.0, 11.35), (6.9, 13.3), (8.3, 13.3)):
                a_ = put_window(w, c, b0, kz, 0.8, 1.2); holes.append(rect(a_, b0, a_ + 0.8, b0 + 1.2))
            # Erdgeschoss: Goldschmiede (links), Tuer, Teestuebchen (rechts)
            yg = float(np.min(H(*np.array([w.ref.P([a, 0, 0.4])[[0, 2]] for a in (1, 8, 15)]).T)))
            a0, a1 = w.A(0.7), w.A(6.3)
            shopfront(w, a0, a1, yg + 0.05, 2.45, mullion=1.4, frame='Holz_dunkel', door_at=a0 + 4.9)
            holes.append(rect(a0 - 0.3, -5, a1 + 0.3, yg + 3.3))
            C.sign(ctx, w.F, a0 - 0.2, yg + 2.62, a1 - a0 + 0.4, 0.42, 'silvergold', c=0.03, depth=0.04)
            awning(w, a0, a1, yg + 3.45, 1.3, 0.55, key=None)
            ad = rect_door(w, 7.35, yg + 0.05, 1.35, 2.5, leaf='Tuer_dunkel', dr=0.3, reveal='Putz_hell')
            holes.append(rect(ad - 0.2, -5, ad + 1.55, yg + 2.8))
            b0_, b1_ = w.A(8.6), w.A(15.4)
            shopfront(w, b0_, b1_, yg + 0.05, 2.5, mullion=1.15, frame='Holz_dunkel', door_at=b0_ + 3.4)
            awning(w, b0_ - 0.2, b1_ + 0.2, yg + 3.35, 3.2, 0.9, key='teestuebchen', valance=0.30)
            holes.append(rect(b0_ - 0.5, -5, b1_ + 0.5, yg + 3.5))
            lantern(w, 6.75, yg + 2.9)
            # Efeu ueber die ganze Front
            C.ivy(ctx, w, w.poly.buffer(-0.15).exterior.coords, holes, seed=3)
        elif e == 0:
            cs = list(np.linspace(1.2, Lref - 1.2, 8))
            for (b0, hh) in [(0.9, 1.45)] + rows:
                for c in cs:
                    put_window(w, c, b0, key, 0.86, hh)

    def int_decor(ctx_, w, k, s):
        if k >= 2:
            m = dorm[k - 2]
            if abs(w.F.ez @ m['F'].ez) > 0.9:
                a = (w.amin + w.amax) / 2
                key = win_key('w_gaube_sp', w=0.7, h=0.9, rows=2, cas=2, dr=0.08, reveal='Putz_weiss', sill='zinc')
                ybot = w.poly.bounds[1] + 0.25
                add_window(w, w.ref.local(w.F.P([a, 0, 0]))[0], ybot + 0.1, 0.7, 0.9, key, lambda: None)
    B = dict(key='BP2', pts=pts, rs=rs, zones=[dict(top=None, rec=0.0, mat='Putz_hell')],
             party={1: 'Putz_hell', 3: 'Putz_hell'}, decor=decor, roof=dict(tile_mat='Ziegel', eave=0.40, verge=0.24, course=0.30),
             int_mat=lambda k, s: 'Putz_hell', int_decor=int_decor, soffit_mat='Traufe')
    building(B)
    return B


# ============================================================================================== 8) Ballhofplatz 3
def build_bp3():
    pts = [tuple(p) for p in OUTL['BP3']['pts']]
    p = [P2(q) for q in pts]
    # e0 Hof NW, e1 Hof WSW, e2 Brandwand Kreuzstr. 11, e3/e5 Ost (Kreuzstrasse), e6 Platz, e7 Brandwand BP2
    EA = 10.9; SL = 47.0
    Sf = Plane.eave(p[6], p[7], EA, SL, p[0], 'S')
    Nf = Plane.eave(p[0], p[1], EA, SL, p[6], 'N')
    Ef = Plane.eave(p[5], p[6], EA, SL, p[0], 'E')
    regA = Polygon([p[6], p[7], p[0], line_x(p[0], p[1], p[5], p[6])]).buffer(0)
    # Nordfluegel: Ebenen an e1 (West) und e5/e3 (Ost); Region = noerdlich der um 5.2 m nach vorn verschobenen Rueckfront
    Wb = Plane.eave(p[1], p[2], EA, SL, p[4], 'Wb')
    Eb = Plane.eave(p[5], p[4], EA, SL, p[1], 'Eb')
    fwd = unit(p[7] - p[0]) if False else None
    t0 = unit(p[1] - p[0]); nfront = np.array([t0[1], -t0[0]])
    if (p[6] - p[0]) @ nfront < 0: nfront = -nfront
    regB = Polygon(pts).buffer(0).intersection(halfspace(p[0] + nfront * 5.2, p[1] + nfront * 5.2, p[3]))         .intersection(halfspace(p[1], p[2], p[4])).buffer(0)
    parts = [(regA, [Sf, Nf, Ef]), (max(as_polys(regB), key=lambda g: g.area), [Wb, Eb])]
    e6L, e6R = p[7], p[6]
    dorm = []
    for a in (4.5, 9.1, 13.7):
        reg, pls, meta = gable_part(e6L, e6R, a, 1.3, 0.6, 2.4, 12.45, 50.0, ov_e=0.14, ov_v=0.12)
        parts.append((reg, pls)); dorm.append(meta)
    rs = RoofSet(parts, outline=pts)

    def decor(w, e, k):
        Lref = float(np.linalg.norm(p[(e + 1) % len(p)] - p[e]))
        if w.zone == 'party': return
        key = win_key('w_bp3', w=0.95, h=1.45, rows=3, cas=2, dr=0.14, reveal='Putz_weiss', sill='wood')
        rows = [(3.7, 1.45), (7.0, 1.45)]
        if e == 6:
            cs = list(np.linspace(1.3, Lref - 1.3, 7))
            yg = float(np.min(H(*np.array([w.ref.P([a, 0, 0.4])[[0, 2]] for a in (1, 9, 17)]).T)))
            for (b0, hh) in rows:
                for c in cs:
                    a_ = put_window(w, c, b0, key, 0.95, hh)
                    surround(w, a_, b0, a_ + 0.95, b0 + hh, w=0.13, mat='Putz_weiss', prot=0.025, sill=0.08)
            # EG: Wohnungstuer links, Laden "marie jo" rechts
            ad = rect_door(w, 2.0, yg + 0.05, 1.1, 2.35, leaf='Tuer_dunkel', dr=0.25, reveal='Putz_weiss', glass_top=True)
            surround(w, ad, yg + 0.05, ad + 1.1, yg + 2.4, w=0.15, mat='Putz_weiss', prot=0.025, sill=0.0)
            kk = win_key('w_bp3_eg', w=1.1, h=1.7, rows=3, cas=2, dr=0.2, reveal='Putz_weiss', sill='stone')
            for c in (4.6, 6.6):
                a_ = put_window(w, c, 0.4, kk, 1.1, 1.7)
                surround(w, a_, 0.4, a_ + 1.1, 2.1, w=0.13, mat='Putz_weiss', prot=0.025, sill=0.0)
            a0, a1 = w.A(9.0), w.A(17.4)
            shopfront(w, a0, a1, yg + 0.05, 2.5, mullion=2.1, frame='Stahl_dunkel', door_at=a0 + 6.2)
            C.sign(ctx, w.F, a0 + 2.6, yg + 2.75, 2.6, 0.65, 'mariejo', c=0.0, depth=0.04)
            w.member(rect(w.amin - 0.02, 2.9, w.amax + 0.02, 3.08), (1, 0), 'Putz_weiss', c0=-0.02, c1=0.05, bevel=0.015, prio=-1)
            plinth(w, -0.6, mat='Sandstein')
            # Efeu am Westrand (Uebergang zu BP2)
            C.ivy(ctx, w, rect(w.A(0.0), -0.5, w.A(2.6), 9.6), [rect(w.A(c) - 0.475, b0, w.A(c) + 0.475, b0 + hh) for (b0, hh) in rows for c in cs[:1]] + [rect(ad, -5, ad + 1.1, yg + 2.4)], seed=7)
        elif e in (5, 3):
            cs = list(np.linspace(1.2, Lref - 1.2, 4)) if Lref > 6 else [Lref / 2]
            for (b0, hh) in [(0.6, 1.45)] + rows:
                for c in cs:
                    a_ = put_window(w, c, b0, key, 0.95, hh)
                    surround(w, a_, b0, a_ + 0.95, b0 + hh, w=0.13, mat='Putz_weiss', prot=0.025, sill=0.08)
            plinth(w, -0.6, mat='Sandstein')
        elif e in (0, 1):
            cs = list(np.linspace(1.2, Lref - 1.2, max(1, int(Lref / 2.5))))
            for (b0, hh) in [(0.8, 1.45)] + rows:
                for c in cs:
                    put_window(w, c, b0, key, 0.95, hh)

    def int_decor(ctx_, w, k, s):
        if k >= 2:
            m = dorm[k - 2]
            if abs(w.F.ez @ m['F'].ez) > 0.9:
                a = (w.amin + w.amax) / 2
                kk = win_key('w_gaube_sp', w=0.7, h=0.9, rows=2, cas=2, dr=0.08, reveal='Putz_weiss', sill='zinc')
                ybot = w.poly.bounds[1] + 0.25
                add_window(w, w.ref.local(w.F.P([a, 0, 0]))[0], ybot + 0.1, 0.7, 0.9, kk, lambda: None)
    B = dict(key='BP3', pts=pts, rs=rs, zones=[dict(top=None, rec=0.0, mat='Putz_bordeaux')],
             party={2: 'Putz_hell', 7: 'Putz_hell'}, decor=decor, roof=dict(tile_mat='Ziegel', eave=0.40, verge=0.24, course=0.30),
             int_mat=lambda k, s: 'Putz_bordeaux', int_decor=int_decor, soffit_mat='Traufe')
    building(B)
    return B


# ============================================================================================== Export
def export(out):
    ctx.proto('gutter_hook', BP.gutter_hook)
    ctx.proto('ridge', lambda: BP.ridge_tile(mat='Ziegel'))
    nodes = []
    fin = ctx.mb.finish(TILES)
    tris = {}
    for mat, (Pv, Nv, UVv, I) in sorted(fin.items()):
        nodes.append({'name': 'BH_%s_nocol' % mat, 'prims': [(mat, Pv, Nv, UVv, I)]})
        tris[mat] = int(len(I))
    inst_tris = 0
    for key, Ms in sorted(ctx.inst.items()):
        if key not in ctx.protos or not Ms: continue
        pf = ctx.protos[key].finish(TILES)
        prims = [(mat, Pv, Nv, UVv, I) for mat, (Pv, Nv, UVv, I) in sorted(pf.items())]
        T, R, S = [], [], []
        for M in Ms:
            t, q, s = bh_glb.mat_to_trs(M)
            T.append(t); R.append(q); S.append(s)
        nodes.append({'name': '%s_inst_nocol' % key, 'prims': prims, 'inst': (np.array(T), np.array(R), np.array(S))})
        pt = sum(len(p[4]) for p in prims)
        inst_tris += pt * len(Ms)
        tris['inst:' + key] = [pt, len(Ms)]
    cf = ctx.col.finish({})
    if 'col' in cf:
        Pv, Nv, UVv, I = cf['col']
        nodes.append({'name': 'col_ballhof', 'prims': [('col', Pv, Nv, UVv, I)]})
    base = sum(v for k, v in tris.items() if not k.startswith('inst:'))
    size = bh_glb.write_glb(out, nodes, MATS, TEX, extras={'name': 'Ballhof-Ensemble', 'source': 'pipeline/landmarks/ballhof'})
    STATS.update(dict(tris_merged=base, tris_instanced=inst_tris, tris_total=base + inst_tris, bytes=size, per_mat=tris,
                      top=TOP[0], seconds=round(time.time() - T0, 1)))
    json.dump(STATS, open(os.path.join(HERE, 'build', 'ballhof_stats.json'), 'w'), indent=1)
    json.dump({'top': round(TOP[0], 2)}, open(os.path.join(HERE, 'build', 'model_top.json'), 'w'))
    print('GLB', out, '%.1f MB' % (size / 1e6), 'tris merged', base, 'instanced', inst_tris, 'total', base + inst_tris, 'top %.2f' % TOP[0])


if __name__ == '__main__':
    only = os.environ.get('BH_ONLY', '')
    for name, fn in (('BH', build_ballhof), ('OST', build_ost), ('PAV', build_pav), ('HINT', build_hint), ('SP', build_sp),
                     ('BP1', build_bp1), ('BP2', build_bp2), ('BP3', build_bp3)):
        if only and name not in only.split(','): continue
        t = time.time(); n0 = ctx.mb.ntri
        fn()
        print('%-5s %6d tris  %.1fs' % (name, ctx.mb.ntri - n0, time.time() - t), flush=True)
    export(OUT)

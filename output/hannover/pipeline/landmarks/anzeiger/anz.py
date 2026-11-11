"""Anzeiger-Hochhaus (Goseriede 9, Fritz Hoeger 1927/28) - hand-built geometry in game frame.
Building frame: u along the east facade (south -> north), e outward (east); world = FE.w(u, y, e)."""
import json, os
import numpy as np
import geo, parts, font
from geo import MB, Frame, UP, lbox, prism, sweep, lathe, nrm
from parts import place, v_plan, corner_plan, pier, crown, teeth, courses, wall_grid, chevron_relief

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'

E0 = np.array([-485.82, 0.0, -268.61])
D = np.array([-0.12865665, 0.0, -0.9916892])
FE = Frame(E0, D)

O_SHELL, O_DET, O_WIN, O_DOME, O_DOMED, O_LET = 'anz_shell', 'anz_detail_nocol', 'anz_windows_nocol', 'anz_dome', 'anz_dome_detail_nocol', 'anz_letters_nocol'

# ------------------------------------------------------------------ heights (game y)
YG, YB = -2.0, -0.64
Y_EGT = 6.21
ROWS = [7.52 + 3.65 * k for k in range(7)]          # R1..R7 centres
WW, WH = 1.24, 2.25
Y_BT = 20.85                                        # top of the full-width base block
Y_WD, Y_WP = 27.85, 28.2                            # wing deck, wing parapet
SLIT_C, SLIT_H, SLIT_W = 33.0, 3.0, 0.36
LET_Y0, LET_H = 35.35, 1.5
Y_DECK, Y_PAR = 36.6, 37.45
DOME_U, DOME_E, DOME_R, DOME_SPRING = 15.73, -11.64, 10.6, 39.4

EW = -0.55
U_S, U_N = -0.6, 31.9                               # south / north wall planes of the base block
U_TS, U_TN = 4.636, 26.82                           # tower south / north wall planes
E_TW = -23.1                                        # tower / base west wall plane
E_WS, E_WN = -13.6, -13.95                          # wing west faces
MAJ = [0.8 + 4.436 * k for k in range(1, 7)]        # east-face local s of major piers (s = u - U_S)
WING_MIN = (MAJ[0] - 2.4, MAJ[-1] + 2.4)


def face(u0, e0, t2):
    """frame on a wall plane: origin at building (u0,e0), walking direction t2 in (u,e); outward = right side"""
    return Frame(FE.w(u0, 0, e0), FE.s * t2[0] + FE.n * t2[1])


def hf_sampler():
    hj = json.load(open(ROOT + 'web/assets/city/ground/heightfield.json'))
    hf = np.fromfile(ROOT + 'web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(hj['h'], hj['w'])
    def h(x, z):
        fx = (x - hj['x0']) / hj['dx']; fz = (z - hj['z0']) / hj['dz']
        i = int(np.floor(fx)); j = int(np.floor(fz)); a = fx - i; b = fz - j
        v = hf[j:j + 2, i:i + 2]
        return float((v[0, 0] * (1 - a) + v[0, 1] * a) * (1 - b) + (v[1, 0] * (1 - a) + v[1, 1] * a) * b)
    return h


HGT = hf_sampler()


def neighbour_top_sampler():
    """roof height (game y) of the LoD2 neighbour west of the tower (Goseriede 5-7 block / Medienzentrum) at a world
    point just outside the Anzeiger west wall; -2 where there is no neighbour roof. Used to continue the window grid
    on the west face down to the neighbour's roofs (light court: 3.5 / 11.5 m parts, QA: blank brick below 20 m)."""
    from shapely.geometry import Polygon, Point
    L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))
    polys = []
    for b in L['buildings']:
        if not b['id'].startswith('ID_2c0e0ff2'): continue
        for r in b['roofs']:
            P = np.array(r['ext']); pg = Polygon([(p[0], -p[1]) for p in P]).buffer(0)
            if pg.area > 0.5: polys.append((pg, float(P[:, 2].max() - 56.16)))
    def top(x, z, r=0.9):
        pt = Point(x, z).buffer(r)
        hs = [h for pg, h in polys if pg.intersects(pt)]
        return max(hs) if hs else -2.0
    return top


NTOP = neighbour_top_sampler()

# window units
U_STD = parts.window_unit(WW, WH, 'std')
U_SLIT = parts.window_unit(SLIT_W, SLIT_H, 'slit', tl=0)
U_SMALL = parts.window_unit(1.0, 1.6, 'std', tl=0.45, bars=1)


def put_window(mb, F, sc, yc, unit=U_STD, w=WW, h=WH, depth=0.2):
    place(mb, O_WIN, unit, F, sc, yc - h / 2, -depth)


def axis_wall(mb, F, s0, s1, y0, y1, axes, rows, half, depth=0.2, spandrel=True, unit=U_STD, w=WW, h=WH, extra_holes=()):
    """wall with windows at axes x rows; spandrel chevrons between rows within +-half of each axis"""
    holes = []
    for sc in axes:
        for yc in rows:
            holes.append((sc - w / 2, sc + w / 2, yc - h / 2, yc + h / 2))
            put_window(mb, F, sc, yc, unit, w, h, depth)
    holes += list(extra_holes)
    sp = []
    if spandrel and len(rows) > 1:
        rs = sorted(rows)
        for sc in axes:
            for a, b in zip(rs[:-1], rs[1:]):
                sp.append((max(s0, sc - half), min(s1, sc + half), a + h / 2, b - h / 2, sc))
    wall_grid(mb, F, s0, s1, y0, y1, holes, O_SHELL, 'klinker', sp, depth)
    for z in sp:
        chevron_relief(mb, O_DET, F, *z)
    return holes


def pier_set(mb, F, specs):
    """specs: list of dict(s, kind, y0, y1, crown=h or None, teeth=(ya,yb) or None)"""
    for p in specs:
        k = p['kind']
        if k == 'major': plan = v_plan(p['s'], 0.80, 0.50)
        elif k == 'minor': plan = v_plan(p['s'], 0.66, 0.40)
        elif k == 'eg': plan = v_plan(p['s'], 1.05, 0.65)
        elif k == 'cl': plan = [(x + p['s'], e) for x, e in corner_plan(-1)]
        elif k == 'cr': plan = [(x + p['s'], e) for x, e in corner_plan(+1)]
        pier(mb, F, plan, p['y0'], p['y1'], O_SHELL, 'klinker', bottom=p.get('bottom', False))
        if p.get('top'):
            mb.poly(O_SHELL, 'klinker', [F.w(a, p['y1'], b) for a, b in plan], UP)
        if p.get('crown'):
            crown(mb, F, plan, p['y1'], p['crown'], back=p.get('back', 0.42), obj=O_SHELL)
        if p.get('teeth'):
            teeth(mb, F, plan, p['teeth'][0], p['teeth'][1], O_DET, 'klinker')
        if p.get('courses'):
            courses(mb, F, plan, p['courses'][0], p['courses'][1], obj=O_DET)
        if p.get('bracket'):
            # stepped corbel at the bottom of a minor pier (lamp hanger)
            y = p['y0']
            for i, (bw, pr) in enumerate(((0.30, 0.26), (0.44, 0.33), (0.56, 0.38))):
                prism(mb, O_SHELL, 'klinker', F, v_plan(p['s'], bw, pr), y - 0.24 + i * 0.08, y - 0.16 + i * 0.08 + 0.08 * (i == 2), top=False, bottom=True, skip_back=0.0)


# ------------------------------------------------------------------ ground floor portals (east facade)
# Stepped (corbelled) portals, refs 03_20230320 (frontal), goseriede/06 (left edge) and the historic 09 (looking up):
# a deep opening that narrows upward in 4 steps above the door zone; every step is carried by a light cast-stone
# corbel slab (front flush + 4 cm lip, soffit visible from the street); doors / shop window ~1 m back, the stepped
# zone above is glazed dark ~1.4 m back; the hexagonal lantern hangs from the V-shaped pier end into the top notch.
# (bottom, top, half-width) relative to YB, measured on ref 03 (75 px/m)
LEVELS = [(0.0, 2.05, 1.60), (2.05, 3.10, 1.52), (3.10, 4.10, 1.26), (4.10, 5.10, 0.96), (5.10, 6.05, 0.62)]
PD = 1.40         # depth of the stepped (glazed) zone behind the bay wall plane
DD = 1.00         # depth of the door / shop-window screen
SLAB = 0.22       # corbel slab thickness
LIP = 0.04        # slab lip in front of the wall plane
LET_COL = 1.47    # vertical ANZEIGER letter columns at sc +- LET_COL (wall strip between eg pier and opening)


def ground_max(F, s0, s1, e0, e1, step=0.2):
    """highest heightfield value over a rectangle given in facade coords"""
    best = -1e9
    for s in np.arange(s0, s1 + 1e-6, step):
        for e in np.arange(e0, e1 + 1e-6, step):
            p = F.w(s, 0, e)
            best = max(best, HGT(p[0], p[2]))
    return best


def slab(mb, F, sa, sb, y0, inner=None, top_from=None):
    """cast-stone corbel slab [sa,sb] x [y0, y0+SLAB] x [-PD, LIP]; soffit + front + lip top/ends.
    inner: 's-' / 's+' -> also the end face toward the opening (full depth); top_from: (a, b) -> top face over [a,b] (full depth)"""
    y1 = y0 + SLAB
    m = 'concrete'
    mb.quad(O_SHELL, m, F.w(sa, y0, -PD), F.w(sb, y0, -PD), F.w(sb, y0, LIP), F.w(sa, y0, LIP), -UP)          # soffit
    mb.quad(O_SHELL, m, F.w(sa, y0, LIP), F.w(sb, y0, LIP), F.w(sb, y1, LIP), F.w(sa, y1, LIP), F.n)          # front
    mb.quad(O_SHELL, m, F.w(sa, y1, LIP), F.w(sb, y1, LIP), F.w(sb, y1, 0.0), F.w(sa, y1, 0.0), UP)           # lip top
    mb.quad(O_SHELL, m, F.w(sa, y0, 0.0), F.w(sa, y0, LIP), F.w(sa, y1, LIP), F.w(sa, y1, 0.0), -F.s)         # lip ends
    mb.quad(O_SHELL, m, F.w(sb, y0, LIP), F.w(sb, y0, 0.0), F.w(sb, y1, 0.0), F.w(sb, y1, LIP), F.s)
    if inner == 's+':
        mb.quad(O_SHELL, m, F.w(sb, y0, 0.0), F.w(sb, y0, -PD), F.w(sb, y1, -PD), F.w(sb, y1, 0.0), F.s)
    elif inner == 's-':
        mb.quad(O_SHELL, m, F.w(sa, y0, -PD), F.w(sa, y0, 0.0), F.w(sa, y1, 0.0), F.w(sa, y1, -PD), -F.s)
    if top_from is not None:
        a, b = top_from
        mb.quad(O_SHELL, m, F.w(a, y1, 0.0), F.w(b, y1, 0.0), F.w(b, y1, -PD), F.w(a, y1, -PD), UP)


def glazed_screen(mb, F, s0, s1, y0, y1, e, pitch=0.78, bottom=True):
    """dark set-back glazing with a dark metal frame and mullions (front of the frame at e + 0.07)"""
    fw = 0.07
    lbox(mb, O_DET, 'metal_dark', F, s0, s1, y0, y0 + fw, e, e + 0.07, ('e-', 's-', 's+') + (() if bottom else ('y-',)))
    lbox(mb, O_DET, 'metal_dark', F, s0, s1, y1 - fw, y1, e, e + 0.07, ('e-', 's-', 's+'))
    lbox(mb, O_DET, 'metal_dark', F, s0, s0 + fw, y0 + fw, y1 - fw, e, e + 0.07, ('e-', 's-', 'y-', 'y+'))
    lbox(mb, O_DET, 'metal_dark', F, s1 - fw, s1, y0 + fw, y1 - fw, e, e + 0.07, ('e-', 's+', 'y-', 'y+'))
    mb.quad(O_DET, 'glass_dark', F.w(s0 + fw, y0 + fw, e + 0.025), F.w(s1 - fw, y0 + fw, e + 0.025), F.w(s1 - fw, y1 - fw, e + 0.025), F.w(s0 + fw, y1 - fw, e + 0.025), F.n)
    n = max(1, int(round((s1 - s0) / pitch)))
    for k in range(1, n):
        x = s0 + (s1 - s0) * k / n
        lbox(mb, O_DET, 'metal_dark', F, x - 0.03, x + 0.03, y0 + fw, y1 - fw, e + 0.025, e + 0.06, ('e-', 'y-', 'y+'))


def fit_text(text, max_w, max_h, stroke=0.16, spacing=0.28, cond=1.0):
    """largest cap height <= max_h whose rendered width <= max_w"""
    _, w1 = font.text_shapes(text, 1.0, stroke, spacing, cond)
    return min(max_h, max_w / w1)


def sign_panel(mb, F, sc, hw, y0, y1, lines, e_back=-0.14):
    """light fascia panel filling the level-1 opening, dark letters fitted inside (5 cm margins), centred"""
    e0, e1 = e_back, e_back + 0.08
    lbox(mb, O_DET, 'sign_panel', F, sc - hw, sc + hw, y0, y1, e0, e1, ('e-', 'y-', 's-', 's+'))
    mx, my = 0.10, 0.08
    W, Hh = 2 * hw - 2 * mx, (y1 - y0) - 2 * my
    gap = 0.30
    nl = len(lines)
    hmax = (Hh - gap * 0.6 * (nl - 1)) / nl
    h = min(fit_text(t, W, hmax) for t in lines)
    tot = nl * h + (nl - 1) * h * gap
    yb = (y0 + y1) / 2 + tot / 2 - h
    for t in lines:
        w = font.extrude_text(mb, O_LET, F, t, sc, yb, e1, h, 0.02, 'metal_dark', 'metal_dark', stroke=0.16, spacing=0.28, align='center')
        if w > W + 1e-6: raise SystemExit('sign text %r does not fit (%.2f > %.2f)' % (t, w, W))
        yb -= h * (1 + gap)


def portal(mb, F, sc, bay_l, bay_r, kind, sign=None):
    lv = [(YB + a, YB + b, hw) for a, b, hw in LEVELS]
    hw0 = lv[0][2]
    # threshold: 4 cm above the highest ground point under it (no flicker against the ground mesh)
    floor_y = ground_max(F, sc - hw0, sc + hw0, -DD, 0.3) + 0.04
    tops = [b for a, b, hw in lv]
    y_lo = [floor_y] + [tops[i - 1] + SLAB for i in range(1, len(lv))]
    # ---- bay wall plane (herringbone 'chevron' brick as on the photos) with the stepped opening as a hole
    # stepped outline (the slab fronts at e=+LIP cover the wall plane above each level top)
    left, right = [], []
    for i, (ya, yb, hw) in enumerate(lv):
        y0 = floor_y if i == 0 else tops[i - 1]
        left += [(sc - hw, y0), (sc - hw, yb)]
        right += [(sc + hw, y0), (sc + hw, yb)]
    hole = left + right[::-1]
    outer = [(bay_l, YG), (bay_r, YG), (bay_r, Y_EGT), (bay_l, Y_EGT)]
    W3 = lambda pts: [F.w(a, b, 0.0) for a, b in pts]
    mb.poly(O_SHELL, 'chevron', W3(outer), F.n, holes=[W3(hole)], uvmode=(F.w(sc - 0.7, 0, 0), F.s, UP))
    # ---- risers (brick reveals) and corbel slabs
    for i, (ya, yb, hw) in enumerate(lv):
        dep = DD if i == 0 else PD
        y0 = y_lo[i]
        mb.quad(O_SHELL, 'klinker', F.w(sc - hw, y0, 0), F.w(sc - hw, y0, -dep), F.w(sc - hw, yb, -dep), F.w(sc - hw, yb, 0), F.s)
        mb.quad(O_SHELL, 'klinker', F.w(sc + hw, y0, -dep), F.w(sc + hw, y0, 0), F.w(sc + hw, yb, 0), F.w(sc + hw, yb, -dep), -F.s)
        if i == 0:
            # full-width lintel slab over the door zone (soffit = ceiling of the door recess), its top is the level-1 floor
            hn = lv[1][2]
            slab(mb, F, sc - hw, sc + hw, yb, top_from=(sc - hn, sc + hn))
        elif i + 1 < len(lv):
            hn = lv[i + 1][2]
            slab(mb, F, sc - hw, sc - hn, yb, inner='s+')
            slab(mb, F, sc + hn, sc + hw, yb, inner='s-')
        else:
            slab(mb, F, sc - hw, sc + hw, yb)
    # ---- set-back screens
    shopfront(mb, F, sc, hw0, floor_y, tops[0], kind)
    for i in range(1, len(lv)):
        ya, yb, hw = lv[i]
        if i == 1 and sign:
            # fascia sign in front; the dark glazed screen behind it stays (rays entering the level-2 opening and
            # dropping behind the sign must not see into the empty interior)
            sign_panel(mb, F, sc, hw, y_lo[1], yb - 0.01, sign['lines'])
        # in the middle the openings of consecutive levels are continuous -> screens meet at the level tops
        glazed_screen(mb, F, sc - hw, sc + hw, y_lo[1] if i == 1 else tops[i - 1], yb, -PD, bottom=(i == 1))
    # ---- threshold (dark granite) from the door screen to 0.3 m in front of the wall plane
    lbox(mb, O_SHELL, 'stone_dark', F, sc - hw0, sc + hw0, YG, floor_y, -DD, 0.30, ('e-', 'y-'))
    # ---- lantern hanging from the V end of the minor pier into the top notch
    lamp(mb, F, sc, Y_EGT - 0.24, 0.20)


def shopfront(mb, F, sc, hw, y0, y1, kind):
    e = -DD
    m = 'metal_dark'
    lbox(mb, O_DET, m, F, sc - hw, sc + hw, y1 - 0.12, y1, e, e + 0.10, ('e-', 's-', 's+'))     # head
    lbox(mb, O_DET, m, F, sc - hw, sc - hw + 0.09, y0, y1 - 0.12, e, e + 0.10, ('e-',))
    lbox(mb, O_DET, m, F, sc + hw - 0.09, sc + hw, y0, y1 - 0.12, e, e + 0.10, ('e-',))
    lbox(mb, O_DET, m, F, sc - hw + 0.09, sc + hw - 0.09, y0, y0 + 0.08, e, e + 0.10, ('e-', 's-', 's+'))
    gl = 'glass_shop' if kind == 'shop' else 'glass_dark'
    mb.quad(O_DET, gl, F.w(sc - hw + 0.09, y0 + 0.08, e + 0.04), F.w(sc + hw - 0.09, y0 + 0.08, e + 0.04), F.w(sc + hw - 0.09, y1 - 0.12, e + 0.04), F.w(sc - hw + 0.09, y1 - 0.12, e + 0.04), F.n)
    if kind == 'shop':
        for x in (sc - hw / 3, sc + hw / 3):
            lbox(mb, O_DET, m, F, x - 0.04, x + 0.04, y0 + 0.08, y1 - 0.12, e + 0.02, e + 0.09, ('e-', 'y-', 'y+'))
    else:
        # two door leaves + side lights
        for x in (sc - 0.95, sc, sc + 0.95):
            lbox(mb, O_DET, m, F, x - 0.04, x + 0.04, y0 + 0.08, y1 - 0.12, e + 0.02, e + 0.10, ('e-', 'y-', 'y+'))
        for xa, xb in ((sc - 0.91, sc - 0.04), (sc + 0.04, sc + 0.91)):
            lbox(mb, O_DET, m, F, xa, xb, y1 - 0.42, y1 - 0.36, e + 0.02, e + 0.10, ('e-', 's-', 's+'))
        for x in (sc - 0.12, sc + 0.12):          # handles
            lbox(mb, O_DET, 'steel', F, x - 0.015, x + 0.015, YB + 0.75, YB + 1.45, e + 0.10, e + 0.16, ('e-',))


def lamp(mb, F, sc, ytop, e):
    """prismatic glass lantern (hexagonal) with dark bronze cap and bottom"""
    # ref 09: elongated hexagonal crystal (~1.5 m) hanging directly under the V end of the minor pier
    cx = F.w(sc, 0, e)
    r = 0.27
    hexa = geo.ring_pts(cx[0], cx[2], r, 6, np.pi / 6)
    def pr(pts, y0, y1, mat, top=True, bottom=True):
        P = np.array(pts)
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            mid = (a + b) / 2 - cx[[0, 2]]
            mb.quad(O_DET, mat, [a[0], y0, a[1]], [b[0], y0, b[1]], [b[0], y1, b[1]], [a[0], y1, a[1]], [mid[0], 0, mid[1]])
        if top: mb.poly(O_DET, mat, [[p[0], y1, p[1]] for p in P], UP)
        if bottom: mb.poly(O_DET, mat, [[p[0], y0, p[1]] for p in P], -UP)
    pr(geo.ring_pts(cx[0], cx[2], 0.05, 6), ytop - 0.14, ytop, 'metal_dark', top=False)
    pr(hexa, ytop - 0.26, ytop - 0.14, 'metal_dark')
    P = geo.ring_pts(cx[0], cx[2], r * 0.92, 6, np.pi / 6)
    yb = ytop - 1.10
    pr(P, yb, ytop - 0.26, 'lamp_glass', top=False, bottom=False)
    # tapered bottom (pyramid)
    tip = np.array([cx[0], ytop - 1.55, cx[2]])
    for i in range(6):
        a, b = P[i], P[(i + 1) % 6]
        mb.poly(O_DET, 'lamp_glass', [[b[0], yb, b[1]], [a[0], yb, a[1]], tip], [((a + b) / 2 - cx[[0, 2]])[0], -0.4, ((a + b) / 2 - cx[[0, 2]])[1]])
    # frame ribs (corners) + a horizontal band at mid height
    for p in P:
        d = nrm(np.array([p[0] - cx[0], 0, p[1] - cx[2]]))
        q = np.array([p[0], 0, p[1]]) + d * 0.005
        t = np.cross(UP, d) * 0.014
        mb.quad(O_DET, 'metal_dark', q - t + UP * yb, q + t + UP * yb, q + t + UP * (ytop - 0.26), q - t + UP * (ytop - 0.26), d)
    pr(geo.ring_pts(cx[0], cx[2], r * 0.95, 6, np.pi / 6), yb + 0.40, yb + 0.45, 'metal_dark')


# ------------------------------------------------------------------ build
def build(mb):
    Fe = face(U_S, EW, (1, 0))                       # east face, s = u - U_S
    L_E = U_N - U_S                                  # 32.5
    sM = MAJ
    s_tower = (sM[0], sM[-1])
    # ---- east piers
    specs = [dict(s=0.0, kind='cl', y0=YG, y1=Y_BT, crown=0.9, teeth=(YB + 0.5, Y_BT - 0.1), back=0.0),
             dict(s=L_E, kind='cr', y0=YG, y1=Y_BT, crown=0.9, teeth=(YB + 0.5, Y_BT - 0.1), back=0.0)]
    for i, s in enumerate(sM):
        specs.append(dict(s=s, kind='eg', y0=YG, y1=Y_EGT, top=True, courses=(YB + 0.45, Y_EGT - 0.1)))
        if i in (0, 5):
            specs.append(dict(s=s, kind='major', y0=Y_EGT, y1=Y_WP, top=True, teeth=(Y_EGT + 0.3, Y_WP - 0.1)))
        else:
            specs.append(dict(s=s, kind='major', y0=Y_EGT, y1=Y_PAR, crown=1.0, teeth=(Y_EGT + 0.3, Y_PAR - 0.1)))
    minors = [WING_MIN[0]] + [s + 2.218 for s in sM[:-1]] + [WING_MIN[1]]
    for j, s in enumerate(minors):
        top = Y_BT if j in (0, len(minors) - 1) else Y_PAR
        specs.append(dict(s=s, kind='minor', y0=Y_EGT, y1=top, crown=0.75 if top == Y_BT else 0.85, teeth=(Y_EGT + 0.3, top - 0.1), bracket=True, bottom=True))
    pier_set(mb, Fe, specs)
    # tower corner piers above the wings (SE / NE) - diagonal
    Fs = face(U_TS, E_TW, (0, 1))                     # tower south face, s from west corner
    Fn = face(U_TN, EW, (0, -1))                      # tower north face, s from east corner
    Fw = face(U_TN, E_TW, (-1, 0))                    # tower west face, s from north corner
    L_SN = EW - E_TW                                   # 22.55
    L_EW = U_TN - U_TS                                 # 22.18
    # ---- east walls
    axes_low = [0.836, 3.236] + [s + d for s in sM[:-1] for d in (1.109, 3.327)] + [sM[-1] + 1.2, sM[-1] + 3.6]
    for k, sc in enumerate(axes_low):
        pass
    # EG portals
    bays = [(0.0, sM[0], WING_MIN[0])] + [(sM[i], sM[i + 1], (sM[i] + sM[i + 1]) / 2) for i in range(5)] + [(sM[-1], L_E, WING_MIN[1])]
    kinds = ['door', 'door', 'door', 'door', 'door', 'door', 'shop']
    signs = [dict(lines=['HOCHHAUS-', 'LICHTSPIELE']), None, None, None, None, None, dict(lines=['RADIO 21'])]
    for (bl, br, sc), kd, sg in zip(bays, kinds, signs):
        portal(mb, Fe, sc, bl, br, kd, sg)
        # vertical ANZEIGER letter columns in the wall strip between the eg pier and the stepped opening (ref 03 / 06)
        for side in (-1, 1):
            col = sc + side * LET_COL
            font.extrude_text(mb, O_LET, Fe, 'ANZEIGER', col, YB + 6.75, 0.0, 0.27, 0.04, 'gold', 'gold', stroke=0.17, vertical=True, pitch=0.44)
        # plinth (dark granite strip, stops at the reveals)
        for sa, sb in ((bl, sc - LEVELS[0][2]), (sc + LEVELS[0][2], br)):
            lbox(mb, O_DET, 'stone_dark', Fe, sa, sb, YG + 1.0, YB + 0.32, 0.0, 0.05, ('e-', 'y-'))
    # upper base wall R1..R4 (all axes), y Y_EGT..Y_BT
    axes_t = [s + d for s in sM[:-1] for d in (1.109, 3.327)]
    axes_w0 = [WING_MIN[0] - 1.2, WING_MIN[0] + 1.2]
    axes_w6 = [WING_MIN[1] - 1.2, WING_MIN[1] + 1.2]
    all_axes = axes_w0 + axes_t + axes_w6
    axis_wall(mb, Fe, 0.0, L_E, Y_EGT, Y_BT, all_axes, ROWS[:4], 1.109)
    # tower east wall R5..R7 + slits, y Y_BT..Y_PAR
    holes = []
    for sc in axes_t:
        holes.append((sc - SLIT_W / 2, sc + SLIT_W / 2, SLIT_C - SLIT_H / 2, SLIT_C + SLIT_H / 2))
        put_window(mb, Fe, sc, SLIT_C, U_SLIT, SLIT_W, SLIT_H, 0.16)
    axis_wall(mb, Fe, s_tower[0], s_tower[1], Y_BT, Y_PAR, axes_t, ROWS[4:7], 1.109, extra_holes=holes)
    # zone R4->R5 spandrel chevrons across Y_BT (both walls meet at Y_BT, chevron zones split there - acceptable)
    # ---- wing upper volumes (white frames)
    wing_upper(mb, south=True)
    wing_upper(mb, south=False)
    # base top ledges in front of the setback (1 m strips) and roofs
    ledge_strip(mb, U_S, U_S + 1.0, E_WS, EW)
    ledge_strip(mb, U_N - 1.0, U_N, E_WN, EW)
    # ---- base block side walls (south: attached to Goseriede 7; north: attached to Goseriedebad; west: neighbour)
    Fbs = face(U_S, E_WS, (0, 1))                     # south face of south wing (s along +e)
    mb.quad(O_SHELL, 'klinker', Fbs.w(0, YG, 0), Fbs.w(E_WS * -1 + EW, YG, 0), Fbs.w(EW - E_WS, Y_BT, 0), Fbs.w(0, Y_BT, 0), Fbs.n)
    Fbn = face(U_N, EW, (0, -1))                      # north face of north wing
    LN = EW - E_WN
    # Goseriedebad covers to ~15.3 -> windows R4 on the exposed part
    axis_wall(mb, Fbn, 0.0, LN, YG, Y_BT, [1.35 + 2.25 * k for k in range(6)], [ROWS[3]], 1.1, spandrel=False)
    # wing west faces (light well / annex side)
    Fws = face(U_TS, E_WS, (-1, 0))                   # south wing west face, s from tower (north) to south
    axis_wall(mb, Fws, 0.0, U_TS - U_S, YG, Y_BT, [1.35, 3.55], ROWS[:4], 1.1, unit=U_STD)       # above Y_BT: wing_upper west face
    Fwn = face(U_N, E_WN, (-1, 0))                    # north wing west face, s from north end
    axis_wall(mb, Fwn, 0.0, U_N - U_TN, 4.0, Y_BT, [1.4, 3.65], ROWS[:4], 1.1)
    mb.quad(O_SHELL, 'klinker', Fwn.w(0, YG, 0), Fwn.w(U_N - U_TN, YG, 0), Fwn.w(U_N - U_TN, 4.0, 0), Fwn.w(0, 4.0, 0), Fwn.n)
    # ---- tower side faces
    tower_side(mb, Fs, L_SN, exposed=lambda s: YG if s < (E_WS - E_TW) + 0.1 else Y_WP, corner_l=True)
    tower_side(mb, Fn, L_SN, exposed=lambda s: Y_WP if s < (EW - E_WN) + 0.1 else 4.0, corner_r=True)
    # west face: covered only up to the neighbour roofs right outside (3.5 / 11.5 m, LoD2) -> window grid continues
    # down to them (light court toward the Medienzentrum block)
    def west_exposed(s):
        hs = [NTOP(*(Fw.w(si, 0, 0.6)[[0, 2]])) for si in (s - 0.7, s, s + 0.7)]
        return max(max(hs), 3.5) + 0.3
    tower_side(mb, Fw, L_EW, exposed=west_exposed, corner_l=False, corner_r=False, west=True)
    # SE / NE tower corners above the wings
    for s, kind in ((sM[0], 'cl'), (sM[-1], 'cr')):
        plan = [(x + s, e) for x, e in corner_plan(-1 if kind == 'cl' else 1)]
        pier(mb, Fe, plan, Y_WP, Y_PAR, O_SHELL, 'klinker')
        crown(mb, Fe, plan, Y_PAR, 1.7, back=0.0, obj=O_SHELL)
        teeth(mb, Fe, plan, Y_WP + 0.2, Y_PAR - 0.1, O_DET, 'klinker')
        rod(mb, Fe.w(s + (-0.25 if kind == 'cl' else 0.25), Y_PAR + 1.3, 0.25))
    # ---- NW annex (4 m, flat roof)
    annex = [(U_TN, E_WN), (31.37, E_WN), (33.12, -23.72), (26.32, -23.53)]
    P = [FE.w(u, 0, e) for u, e in annex]
    prism_world(mb, O_SHELL, 'klinker', P, YG, 4.0)
    mb.poly(O_SHELL, 'roof_flat', [p + UP * 4.0 for p in P], UP)
    # ---- parapet ring (inner faces + copper coping) and roofs
    parapet_ring(mb)
    roof_deck(mb)
    dome(mb)
    top_letters(mb, Fe, axes_t)


def rod(mb, p, h=1.6, r=0.025):
    lathe(mb, O_DET, 'steel', (p[0], p[2]), [(r, p[1] - 1.4), (r, p[1] + h - 1.4), (0.0, p[1] + h - 1.35)], segs=8, smooth=True)


def prism_world(mb, obj, mat, P, y0, y1):
    P = np.asarray(P, float)
    cen = P.mean(0)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        mid = (a + b) / 2 - cen
        mb.quad(obj, mat, a + UP * y0, b + UP * y0, b + UP * y1, a + UP * y1, mid * np.array([1, 0, 1]))


def ledge_strip(mb, u0, u1, e0, e1):
    P = [FE.w(u0, Y_BT, e0), FE.w(u1, Y_BT, e0), FE.w(u1, Y_BT, e1), FE.w(u0, Y_BT, e1)]
    mb.poly(O_SHELL, 'roof_flat', P, UP)
    # copper coping along the outer edge
    uo = u0 if u0 == U_S else u1
    sgn = -1 if u0 == U_S else 1
    Fx = face(uo, e1, (0, -1)) if sgn < 0 else face(uo, e0, (0, 1))
    Fx = face(uo, e0 if sgn > 0 else e1, (0, 1) if sgn < 0 else (0, -1))
    # simple coping box in building frame
    a = FE.w(uo - sgn * 0.3, Y_BT, e0); b = FE.w(uo + sgn * 0.05, Y_BT, e1)
    lbox(mb, O_SHELL, 'copper_dark', face(min(uo - sgn * 0.3, uo + sgn * 0.05), e0, (1, 0)), 0, 0.35, Y_BT, Y_BT + 0.06, 0, e1 - e0 + 0.05)


def wing_upper(mb, south=True):
    if south:
        u_out, u_in = U_S + 1.0, U_TS
        ew = E_WS
        Ff = face(u_out, EW, (1, 0))           # east face of upper wing, s from outer edge
        Fo = face(u_out, ew, (0, 1))           # outer (south) face, s from west end
        Fw_ = face(u_in, ew, (-1, 0))          # west face, s from tower
        L_f = u_in - u_out
    else:
        u_out, u_in = U_N - 1.0, U_TN
        ew = E_WN
        Ff = face(u_in, EW, (1, 0))
        Fo = face(u_out, EW, (0, -1))          # north face, s from east end
        Fw_ = face(u_out, ew, (-1, 0))
        L_f = u_out - u_in
    L_o = EW - ew
    # east face: white frames around window pairs R5, R6
    # window pair centred on the (1 m set back, ~4.1-4.2 m wide) wing face so window + light frame stay inside it
    axes = [L_f / 2 - 1.15, L_f / 2 + 1.15]
    white_band(mb, Ff, 0.0, L_f, axes)
    # outer face
    ax_o = [1.75, 3.9, 6.05, 8.2, 10.35, 12.0] if L_o > 12.5 else [1.75, 3.9, 6.05, 8.2, 10.35]
    ax_o = [a for a in ax_o if a < L_o - 0.8]
    white_band(mb, Fo, 0.0, L_o, ax_o)
    # west face (plain, with windows)
    axis_wall(mb, Fw_, 0.0, L_f, Y_BT, Y_WP, [L_f / 2 - 1.1, L_f / 2 + 1.1], ROWS[4:6], 1.1)
    # deck + coping
    P = [FE.w(u_out, Y_WD, ew), FE.w(u_in, Y_WD, ew), FE.w(u_in, Y_WD, EW), FE.w(u_out, Y_WD, EW)]
    mb.poly(O_SHELL, 'roof_flat', P, UP)
    # parapet: inner faces & coping (outer, east and west edges)
    # copings: the outer-face coping runs over both corners, the east / west ones stop at it (no overlapping tops)
    rng_f = (0.3, L_f + 0.05) if south else (-0.05, L_f - 0.3)
    rng_w = (-0.05, L_f - 0.3) if south else (0.3, L_f + 0.05)
    for Fp, L, (c0, c1) in ((Ff, L_f, rng_f), (Fo, L_o, (-0.05, L_o + 0.05)), (Fw_, L_f, rng_w)):
        lbox(mb, O_SHELL, 'copper_dark', Fp, c0, c1, Y_WP, Y_WP + 0.05, -0.3, 0.04, ())
        mb.quad(O_SHELL, 'klinker', Fp.w(L, Y_WD, -0.3), Fp.w(0, Y_WD, -0.3), Fp.w(0, Y_WP, -0.3), Fp.w(L, Y_WP, -0.3), -Fp.n)
    # railing along east and outer edges
    railing(mb, Ff, 0.2, L_f - 0.2, Y_WP + 0.05, -0.15)
    railing(mb, Fo, 0.2 if south else 0.21, L_o - (0.21 if south else 0.2), Y_WP + 0.05, -0.15)      # no overlap with the east rail at the corner


def white_band(mb, F, s0, s1, axes):
    """setback wing face: brick with light concrete frames around window pairs, rows R5/R6"""
    holes = []
    for yc in ROWS[4:6]:
        for sc in axes:
            holes.append((sc - WW / 2, sc + WW / 2, yc - WH / 2, yc + WH / 2))
            put_window(mb, F, sc, yc)
        # frame (raised 4 cm) around the pair
        fa, fb = min(axes) - WW / 2 - 0.18, max(axes) + WW / 2 + 0.18
        y0, y1 = yc - WH / 2 - 0.16, yc + WH / 2 + 0.2
        fr = [(fa, fb, y0, yc - WH / 2), (fa, fb, yc + WH / 2, y1)]
        xs = sorted([fa] + [v for sc in axes for v in (sc - WW / 2, sc + WW / 2)] + [fb])
        for a, b in zip(xs[0::2], xs[1::2]):
            fr.append((a, b, yc - WH / 2, yc + WH / 2))
        for (a, b, c, d) in fr:
            lbox(mb, O_SHELL, 'concrete', F, a, b, c, d, 0.0, 0.04, ('e-',))
    wall_grid(mb, F, s0, s1, Y_BT, Y_WP, holes, O_SHELL, 'klinker', None, 0.2, sill=False)     # the light frame is the sill


def railing(mb, F, s0, s1, y, e, h=1.0):
    n = max(2, int((s1 - s0) / 1.6) + 1)
    for s in np.linspace(s0, s1, n):
        lbox(mb, O_DET, 'metal_dark', F, s - 0.025, s + 0.025, y, y + h - 0.05, e - 0.025, e + 0.025, ('y+',))      # posts end under the top rail
    lbox(mb, O_DET, 'metal_dark', F, s0 - 0.03, s1 + 0.03, y + h - 0.05, y + h, e - 0.03, e + 0.03)
    lbox(mb, O_DET, 'metal_dark', F, s0, s1, y + h * 0.5 - 0.015, y + h * 0.5 + 0.015, e - 0.012, e + 0.012)


def tower_side(mb, F, L, exposed, corner_l=False, corner_r=False, west=False):
    """pier-and-window face of the tower (south / north / west). exposed(s) -> y below which the face is covered."""
    n = 10
    ax = L / n
    axes = [ax * (k + 0.5) for k in range(n)]
    # wall below/above per axis column
    holes = []
    for sc in axes:
        ylo = exposed(sc)
        rows = [r for r in ROWS if r - WH / 2 > ylo + 0.05]
        for yc in rows:
            holes.append((sc - WW / 2, sc + WW / 2, yc - WH / 2, yc + WH / 2)); put_window(mb, F, sc, yc)
        if ylo < 2.0:      # light well: ground floor windows
            holes.append((sc - 0.5, sc + 0.5, YB + 1.2, YB + 2.8)); put_window(mb, F, sc, YB + 2.0, U_SMALL, 1.0, 1.6)
        holes.append((sc - SLIT_W / 2, sc + SLIT_W / 2, SLIT_C - SLIT_H / 2, SLIT_C + SLIT_H / 2))
        put_window(mb, F, sc, SLIT_C, U_SLIT, SLIT_W, SLIT_H, 0.16)
    sp = []
    for sc in axes:
        ylo = exposed(sc)
        rs = [r for r in ROWS if r - WH / 2 > ylo + 0.05]
        for a, b in zip(rs[:-1], rs[1:]):
            sp.append((sc - ax / 2, sc + ax / 2, a + WH / 2, b - WH / 2, sc))
    wall_grid(mb, F, 0.0, L, YG, Y_PAR, holes, O_SHELL, 'klinker', sp, 0.2)
    for z in sp:
        chevron_relief(mb, O_DET, F, *z)
    # piers
    specs = []
    for k in range(1, n):
        s = ax * k
        y0 = max(exposed(s - 0.01), exposed(s + 0.01))
        y0 = YB if y0 < 2 else y0
        kind = 'major' if k % 2 == 0 else 'minor'
        specs.append(dict(s=s, kind=kind, y0=y0, y1=Y_PAR, crown=1.0 if kind == 'major' else 0.85, teeth=(y0 + 0.3, Y_PAR - 0.1), bottom=True))
    if corner_l:
        y0 = exposed(0.01); y0 = YG if y0 < 2 else y0
        specs.append(dict(s=0.0, kind='cl', y0=y0, y1=Y_PAR, crown=1.7, back=0.0, teeth=(max(y0, YB) + 0.3, Y_PAR - 0.1), bottom=True))
    if corner_r:
        y0 = exposed(L - 0.01); y0 = YG if y0 < 2 else y0
        specs.append(dict(s=L, kind='cr', y0=y0, y1=Y_PAR, crown=1.7, back=0.0, teeth=(max(y0, YB) + 0.3, Y_PAR - 0.1), bottom=True))
    pier_set(mb, F, specs)
    for s, sgn in ((0.0, -1), (L, 1)):
        if (s == 0.0 and corner_l) or (s == L and corner_r):
            rod(mb, F.w(s + sgn * 0.25, Y_PAR + 1.3, 0.25))


def parapet_ring(mb):
    """inner parapet faces and copper coping around the tower top"""
    corners = [(U_TS, EW), (U_TN, EW), (U_TN, E_TW), (U_TS, E_TW)]       # walk: east face north, ...
    # path with outward on the right: east face walking +u
    path = [FE.w(u, Y_PAR, e) for u, e in corners]
    th = 0.42
    prof_cop = [(-th - 0.04, -0.0), (-th - 0.04, 0.06), (0.06, 0.06), (0.06, 0.0)]
    # coping as closed sweep (profile ordered so normals point outward/up)
    sweep(mb, O_SHELL, 'copper_dark', path, [(0.0, -0.02), (0.06, -0.02), (0.06, 0.06), (-th - 0.04, 0.06), (-th - 0.04, -0.02)], closed=True, caps=False)
    # inner parapet face (facing inward, toward the dome)
    inner = [FE.w(u, 0, e) for u, e in corners]
    cen = np.mean(inner, 0)
    for i in range(4):
        a, b = inner[i], inner[(i + 1) % 4]
        da = nrm(cen - a) * th * 1.414; db = nrm(cen - b) * th * 1.414
        A = a + da; B = b + db
        mb.quad(O_SHELL, 'klinker', A + UP * Y_DECK, B + UP * Y_DECK, B + UP * Y_PAR, A + UP * Y_PAR, (cen - (a + b) / 2) * np.array([1, 0, 1]))


def roof_deck(mb):
    th = 0.42 * 1.414
    corners = [(U_TS, EW), (U_TN, EW), (U_TN, E_TW), (U_TS, E_TW)]
    P = [FE.w(u, Y_DECK, e) for u, e in corners]
    cen = np.mean(P, 0)
    P = [p + nrm(cen - p) * th for p in P]
    mb.poly(O_SHELL, 'roof_flat', P, UP)


def dome(mb):
    c = FE.w(DOME_U, 0, DOME_E)
    cx, cz = c[0], c[2]
    R = DOME_R
    # drum + hemisphere profile (bottom -> top)
    prof = [(R + 0.06, Y_DECK), (R + 0.06, Y_DECK + 0.35), (R, Y_DECK + 0.45), (R, DOME_SPRING)]
    na = 30
    for i in range(1, na + 1):
        a = np.pi / 2 * i / na
        if a > np.radians(81): break
        prof.append((R * np.cos(a), DOME_SPRING + R * np.sin(a)))
    a_top = np.radians(81)
    r_top = R * np.cos(a_top); y_top = DOME_SPRING + R * np.sin(a_top)
    prof += [(r_top, y_top + 0.32), (r_top + 0.06, y_top + 0.32), (r_top + 0.06, y_top + 0.40), (0.0, y_top + 0.62)]
    lathe(mb, O_DOME, 'copper', (cx, cz), prof, segs=96, smooth=True)
    # standing seams (ribs) along the meridians
    nrib = 48
    for k in range(nrib):
        a = 2 * np.pi * k / nrib
        d = np.array([np.cos(a), 0, np.sin(a)]); t = np.array([-np.sin(a), 0, np.cos(a)])
        pts = []
        for i in range(0, 41):
            th = np.radians(78) * i / 40
            r = R * np.cos(th); y = DOME_SPRING + R * np.sin(th)
            pts.append((r, y, np.array([np.cos(th), np.sin(th)])))
        # include drum part
        pts = [(R, Y_DECK + 0.45, np.array([1.0, 0.0])), (R, DOME_SPRING, np.array([1.0, 0.0]))] + pts[1:]
        T = []
        for (r0, y0, n0), (r1, y1, n1) in zip(pts[:-1], pts[1:]):
            p0 = np.array([cx, y0, cz]) + d * r0; p1 = np.array([cx, y1, cz]) + d * r1
            q0 = p0 + (d * n0[0] + UP * n0[1]) * 0.035; q1 = p1 + (d * n1[0] + UP * n1[1]) * 0.035
            for sgn in (-1, 1):
                b0 = p0 + t * sgn * 0.022; b1 = p1 + t * sgn * 0.022
                tri1 = [b0, b1, q1]; tri2 = [b0, q1, q0]
                for tri in (tri1, tri2):
                    tn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                    if tn @ (t * sgn) < 0: tri = tri[::-1]
                    T.append(tri)
        T = np.array(T)
        UV = np.zeros((len(T), 3, 2)); UV[..., 0] = 0.5; UV[..., 1] = T[..., 1] / 2.0
        mb.tris(O_DOMED, 'copper', T, UV)
    # horizontal seam rings (slightly raised bands every ~1.5 m of arc)
    for th_deg in (12, 24, 36, 48, 60, 70):
        th = np.radians(th_deg)
        r = R * np.cos(th); y = DOME_SPRING + R * np.sin(th)
        nn = np.array([np.cos(th), np.sin(th)])
        prof = [(r + nn[0] * 0.0 + np.sin(th) * 0.05, y - np.cos(th) * 0.05), (r + nn[0] * 0.025, y + nn[1] * 0.025), (r - np.sin(th) * 0.05, y + np.cos(th) * 0.05)]
        lathe(mb, O_DOMED, 'copper', (cx, cz), prof, segs=96, smooth=False)
    # 8 hatches (small triangular dormers) near the top
    for k in range(8):
        a = 2 * np.pi * (k + 0.5) / 8
        th = np.radians(66)
        r = R * np.cos(th); y = DOME_SPRING + R * np.sin(th)
        d = np.array([np.cos(a), 0, np.sin(a)]); t = np.array([-np.sin(a), 0, np.cos(a)])
        base = np.array([cx, 0, cz]) + d * r + UP * y
        w, hh, depth = 0.45, 0.55, 1.1
        # front triangle (vertical, facing outward)
        f0 = base + d * 0.15 - t * w; f1 = base + d * 0.15 + t * w; f2 = base + d * 0.15 + UP * hh
        # back apex up the dome
        th2 = th + depth / R
        back = np.array([cx, 0, cz]) + d * R * np.cos(th2) + UP * (DOME_SPRING + R * np.sin(th2))
        mb.poly(O_DOMED, 'metal_dark', [f0, f1, f2], d)
        mb.poly(O_DOMED, 'copper_dark', [f1, back + UP * 0.05, f2], t + UP * 0.6)
        mb.poly(O_DOMED, 'copper_dark', [f0, f2, back + UP * 0.05], -t + UP * 0.6)
        # small louvre frame
        lb = base + d * 0.16
        mb.poly(O_DOMED, 'copper_dark', [lb - t * w * 0.9 + UP * 0.02, lb + t * w * 0.9 + UP * 0.02, lb + t * w * 0.9 + UP * 0.07, lb - t * w * 0.9 + UP * 0.07], d)


def top_letters(mb, Fe, axes_t):
    """ANZEIGER: flat gilded letters fixed directly on the brick between the piers, tops just under the crown bases
    (refs 03 / 04: one letter per bay in bays 2..9, no rail)"""
    word = 'ANZEIGER'
    y0 = Y_PAR - 0.14 - LET_H
    for i, ch in enumerate(word):
        sc = MAJ[0] + 2.218 * (i + 1.5)
        w = font.extrude_text(mb, O_LET, Fe, ch, sc, y0, 0.0, LET_H, 0.06, 'gold', 'gold', stroke=0.12, cond=0.8, align='center')
        if w > 2.218 - 0.80 - 0.10: raise SystemExit('crown letter %s too wide (%.2f)' % (ch, w))
    floodlights(mb)


def floodlights(mb):
    """a few dark-grey floodlights on the tower corners only (ref 04: small twin heads on the corner piers)"""
    corners = [(face(U_TS, EW, (1, 0)), 0.0, 1), (face(U_TS, EW, (1, 0)), U_TN - U_TS, -1),
               (face(U_TN, E_TW, (-1, 0)), 0.0, 1), (face(U_TN, E_TW, (-1, 0)), U_TN - U_TS, -1)]
    y = Y_PAR + 0.06
    for F, s, sg in corners:
        s0 = s + sg * 1.2                          # just inside the big corner crown, on the coping
        e = -0.24
        lbox(mb, O_DET, 'metal_dark', F, s0 - 0.03, s0 + 0.03, y, y + 0.38, e - 0.03, e + 0.03, ('y-',))       # post
        lbox(mb, O_DET, 'metal_dark', F, s0 - 0.28, s0 + 0.28, y + 0.38, y + 0.43, e - 0.04, e + 0.04, ())      # cross bar
        for dx in (-0.2, 0.2):                     # twin heads tilted up toward the dome/facade
            lbox(mb, O_DET, 'metal_dark', F, s0 + dx - 0.11, s0 + dx + 0.11, y + 0.43, y + 0.62, e - 0.10, e + 0.08, ())
            lbox(mb, O_DET, 'lamp_glass', F, s0 + dx - 0.09, s0 + dx + 0.09, y + 0.45, y + 0.60, e + 0.08, e + 0.09, ('e-',))


if __name__ == '__main__':
    mb = MB()
    build(mb)
    print(mb.total())
    for o, d in mb.stats().items(): print(o, d)

"""Reusable architectural parts: windows (instanced), piers with saw-tooth headers, pier-and-window wall faces."""
import numpy as np
import geo
from geo import MB, Frame, UP, lbox, prism, nrm

F0 = Frame((0, 0, 0), (1, 0, 0))     # local unit frame: s = x, y up, e = +z


# ------------------------------------------------------------------ window units (built once, placed many times)
def window_unit(w, h, kind='std', frame_mat='paint_white', glass_mat='glass', tl=0.55, bars=2):
    m = MB(); fw = 0.07; fd = 0.09
    o = 'u'
    x0, x1 = -w / 2, w / 2
    sk = ('e-',)
    lbox(m, o, frame_mat, F0, x0, x0 + fw, 0, h, -fd, 0, sk)
    lbox(m, o, frame_mat, F0, x1 - fw, x1, 0, h, -fd, 0, sk)
    lbox(m, o, frame_mat, F0, x0 + fw, x1 - fw, 0, fw, -fd, 0, sk + ('s-', 's+'))
    lbox(m, o, frame_mat, F0, x0 + fw, x1 - fw, h - fw, h, -fd, 0, sk + ('s-', 's+'))
    gz = -0.065
    m.quad(o, glass_mat, F0.w(x0 + fw, fw, gz), F0.w(x1 - fw, fw, gz), F0.w(x1 - fw, h - fw, gz), F0.w(x0 + fw, h - fw, gz), F0.n)
    bd0, bd1 = -0.055, -0.012
    if kind == 'std':
        yt = h - tl
        lbox(m, o, frame_mat, F0, -0.04, 0.04, fw, h - fw, -0.075, -0.005, sk + ('y-', 'y+'))
        lbox(m, o, frame_mat, F0, x0 + fw, -0.04, yt - 0.04, yt + 0.04, -0.075, -0.005, sk + ('s-', 's+'))
        lbox(m, o, frame_mat, F0, 0.04, x1 - fw, yt - 0.04, yt + 0.04, -0.075, -0.005, sk + ('s-', 's+'))
        # casement sash frames (slightly proud of glass)
        sw = 0.045
        for a, b in ((x0 + fw, -0.04), (0.04, x1 - fw)):
            lbox(m, o, frame_mat, F0, a, a + sw, fw, yt - 0.04, -0.06, -0.02, sk + ('y-', 'y+'))
            lbox(m, o, frame_mat, F0, b - sw, b, fw, yt - 0.04, -0.06, -0.02, sk + ('y-', 'y+'))
            lbox(m, o, frame_mat, F0, a + sw, b - sw, fw, fw + sw, -0.06, -0.02, sk + ('s-', 's+'))
            lbox(m, o, frame_mat, F0, a + sw, b - sw, yt - 0.04 - sw, yt - 0.04, -0.06, -0.02, sk + ('s-', 's+'))
            for k in range(1, bars + 1):
                yb = fw + sw + (yt - 0.04 - sw - fw - sw) * k / (bars + 1)
                lbox(m, o, frame_mat, F0, a + sw, b - sw, yb - 0.013, yb + 0.013, bd0, bd1, sk + ('s-', 's+'))
    elif kind == 'slit':
        n = max(2, int(round(h / 0.55)))
        for k in range(1, n):
            yb = fw + (h - 2 * fw) * k / n
            lbox(m, o, frame_mat, F0, x0 + fw, x1 - fw, yb - 0.015, yb + 0.015, bd0, bd1, sk + ('s-', 's+'))
    elif kind == 'grid':                  # n x m small panes (Goseriedebad)
        nx, ny = bars if isinstance(bars, tuple) else (2, 3)
        for k in range(1, nx):
            xb = x0 + fw + (w - 2 * fw) * k / nx
            lbox(m, o, frame_mat, F0, xb - 0.02, xb + 0.02, fw, h - fw, bd0, bd1, sk + ('y-', 'y+'))
        xs = [x0 + fw] + [v for k in range(1, nx) for v in (x0 + fw + (w - 2 * fw) * k / nx - 0.02, x0 + fw + (w - 2 * fw) * k / nx + 0.02)] + [x1 - fw]
        for k in range(1, ny):          # horizontal bars in pieces between the verticals (no crossing coplanar fronts)
            yb = fw + (h - 2 * fw) * k / ny
            for xa, xb in zip(xs[0::2], xs[1::2]):
                lbox(m, o, frame_mat, F0, xa, xb, yb - 0.02, yb + 0.02, bd0, bd1, sk + ('s-', 's+'))
    return m.arrays()


def arched_window_unit(w, h, rise, frame_mat='paint_white', glass_mat='glass', mullions=0, transom=None, segs=12):
    """round/segmental-headed window: rectangle w x (h - rise) + arch of given rise. local coords like window_unit."""
    m = MB(); o = 'u'; fw = 0.07
    r = (rise ** 2 + (w / 2) ** 2) / (2 * rise)
    cy = h - r
    a0 = np.arcsin(min(1.0, (w / 2) / r))
    angs = np.linspace(np.pi / 2 + a0, np.pi / 2 - a0, segs + 1)
    outer = [(-w / 2, 0)] + [(r * np.cos(a), cy + r * np.sin(a)) for a in angs] + [(w / 2, 0)]
    outer = [(-w / 2, 0), (-w / 2, h - rise)] + [(r * np.cos(a), cy + r * np.sin(a)) for a in angs[1:-1]] + [(w / 2, h - rise), (w / 2, 0)]
    ri = r - fw
    a0i = np.arcsin(min(1.0, (w / 2 - fw) / ri))
    angi = np.linspace(np.pi / 2 + a0i, np.pi / 2 - a0i, segs + 1)
    yi = cy + ri * np.cos(a0i)
    inner = [(-w / 2 + fw, fw), (-w / 2 + fw, yi)] + [(ri * np.cos(a), cy + ri * np.sin(a)) for a in angi[1:-1]] + [(w / 2 - fw, yi), (w / 2 - fw, fw)]
    P = lambda pts, e: [F0.w(p[0], p[1], e) for p in pts]
    m.poly(o, frame_mat, P(outer, 0), F0.n, holes=[P(inner, 0)])
    # frame inner reveal + outer edge
    ring_faces(m, o, frame_mat, inner, 0, -0.09, inward=True)
    ring_faces(m, o, frame_mat, outer, 0, -0.09, inward=False)
    m.poly(o, glass_mat, P(inner, -0.065), F0.n)
    for k in range(1, mullions + 1):
        xb = -w / 2 + w * k / (mullions + 1)
        ytop = cy + np.sqrt(max(ri ** 2 - xb ** 2, 0)) if abs(xb) < ri else yi
        lbox(m, o, frame_mat, F0, xb - 0.035, xb + 0.035, fw, ytop, -0.07, -0.005, ('e-', 'y-'))
    if transom:          # transom pieces between the mullions (no crossing boxes with coplanar fronts)
        xs = [-w / 2 + fw] + [v for k in range(1, mullions + 1) for v in (-w / 2 + w * k / (mullions + 1) - 0.035, -w / 2 + w * k / (mullions + 1) + 0.035)] + [w / 2 - fw]
        for xa, xb in zip(xs[0::2], xs[1::2]):
            lbox(m, o, frame_mat, F0, xa, xb, transom - 0.035, transom + 0.035, -0.07, -0.005, ('e-', 's-', 's+'))
    return m.arrays()


def ring_faces(m, obj, mat, ring, e0, e1, inward):
    r = np.asarray(ring, float)
    area = 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
    for i in range(len(r)):
        a, b = r[i], r[(i + 1) % len(r)]
        d = b - a
        if np.hypot(*d) < 1e-7: continue
        o2 = np.array([d[1], -d[0]])
        if area < 0: o2 = -o2               # now o2 points out of the ring
        if inward: o2 = -o2
        hint = F0.s * o2[0] + UP * o2[1]
        m.quad(obj, mat, F0.w(a[0], a[1], e0), F0.w(b[0], b[1], e0), F0.w(b[0], b[1], e1), F0.w(a[0], a[1], e1), hint)


def place(mb, obj, unit, F, s, y, e, mat_map=None, scale=(1, 1, 1), flip=False):
    """place a unit (dict (o,mat)->(P,UV)) into frame F at local (s,y,e)"""
    R = F.M * np.array(scale)[None, :]
    t = F.w(s, y, e)
    for (o, mat), (P, UV) in unit.items():
        T = P.reshape(-1, 3, 3).astype(float)
        if flip:     # mirror s
            T = T * np.array([-1, 1, 1]); T = T[:, ::-1]; UV2 = UV.reshape(-1, 3, 2)[:, ::-1]
        else:
            UV2 = UV.reshape(-1, 3, 2)
        mm = mat if mat_map is None else mat_map.get(mat, mat)
        mb.tris(obj, mm, T @ R.T + t, UV2)


# ------------------------------------------------------------------ spandrel relief (projecting header chevrons, Hoeger)
_CHEV_CACHE = {}


def chevron_unit(half, height, mat='klinker', lines=(0.42, 0.86), slope=0.8, bw=0.115, bh=0.068, proj=0.042, pitch=0.125):
    """'^' rows of projecting brick headers inside a spandrel panel, centred on s=0, bottom at y=0, wall plane e=0.
    Follows the course geometry of the 'chevron' texture (courses rise with slope 0.8 toward the centre).
    Each header = box with the back face omitted (sits on the wall plane -> no coplanar overlap)."""
    key = (round(half, 2), round(height, 2), mat)
    if key in _CHEV_CACHE: return _CHEV_CACHE[key]
    m = MB(); o = 'u'
    for f in lines:
        apex = height * f + 0.04
        ds = np.arange(0.0, half - bw / 2 + 1e-6, pitch)
        for d in ds:
            yc = apex - slope * d
            if yc - bh / 2 < 0.06: continue
            for sg in ((0,) if d < 1e-6 else (-1, 1)):
                c = sg * d
                lbox(m, o, mat, F0, c - bw / 2, c + bw / 2, yc - bh / 2, yc + bh / 2, 0.0, proj, ('e-',))
    _CHEV_CACHE[key] = m.arrays()
    return _CHEV_CACHE[key]


def chevron_relief(mb, obj, F, sa, sb, ya, yb, sc, pier_clear=0.42, max_half=0.72):
    half = min(min(sc - sa, sb - sc) - pier_clear, max_half)
    if half < 0.3 or yb - ya < 0.5: return
    place(mb, obj, chevron_unit(half, yb - ya), F, sc, ya, 0.0)


# ------------------------------------------------------------------ piers
def v_plan(s, bw, pr):
    return [(s - bw / 2, 0.0), (s, pr), (s + bw / 2, 0.0)]


def corner_plan(side, a=0.75, t=0.32):
    """diagonal corner pier at local s=0 (side=-1: building extends to +s, the other wall is the plane s=0 going to -e)
    side=+1: mirror (corner at s=0, building extends to -s)"""
    if side < 0:
        return [(a, 0.0), (-t, t), (0.0, -a), (0.0, 0.0)]
    return [(0.0, 0.0), (0.0, -a), (t, t), (-a, 0.0)]


def pier(mb, F, plan, y0, y1, obj='shell', mat='klinker', bottom=False):
    prism(mb, obj, mat, F, plan, y0, y1, top=False, bottom=bottom, skip_back=0.0)


def crown(mb, F, plan, y0, h, back=0.42, obj='shell', mat='klinker', apex_e=None):
    """pointed brick crown on a pier plan (plan points with e>=0) + parapet thickness behind"""
    P = [tuple(p) for p in plan]
    ss = [p[0] for p in P]
    base = list(P)
    # add back points behind the wall plane
    sl, sr = min(ss), max(ss)
    if all(p[1] >= -1e-9 for p in P):
        base = [(sr, -back), (sl, -back)] + [p for p in P]
        # order: go around: left-back -> ... ensure polygon order (sl,0)->(tip)->(sr,0)->(sr,-back)->(sl,-back)
        base = [p for p in P] + [(sr, -back), (sl, -back)]
    cen = np.mean(np.array(base), 0)
    ae = cen[1] if apex_e is None else apex_e
    apex = F.w(cen[0], y0 + h, ae)
    n = len(base)
    B2 = np.array(base, float)
    area = 0.5 * np.sum(B2[:, 0] * np.roll(B2[:, 1], -1) - np.roll(B2[:, 0], -1) * B2[:, 1])
    for i in range(n):
        a, b = base[i], base[(i + 1) % n]
        A = F.w(a[0], y0, a[1]); B = F.w(b[0], y0, b[1])
        d = np.array(b, float) - np.array(a, float)
        if np.hypot(*d) < 1e-9: continue
        o2 = np.array([d[1], -d[0]]) if area > 0 else np.array([-d[1], d[0]])      # true outward edge normal (works for the concave corner plans)
        out = F.s * o2[0] + F.n * o2[1]
        mb.poly(obj, mat, [A, B, apex], nrm(out) + UP * 0.3)


def teeth(mb, F, plan, y0, y1, obj='detail_nocol', mat='klinker', row=0.25, frac=((0.28, 0.72), (0.5,)), size=0.13, out=0.07, th=0.075, skip_wall=True):
    """saw-tooth (dog-tooth) projecting headers on the outward faces of a pier plan"""
    P = np.asarray(plan, float)
    n = len(P)
    faces = []
    cen = P.mean(0)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        if skip_wall and abs(a[1]) < 1e-9 and abs(b[1]) < 1e-9: continue
        if skip_wall and (a[1] < -1e-6 and b[1] < 1e-6) and abs(a[0] - b[0]) < 1e-6: continue    # edge along a side wall
        d = b - a; L = np.hypot(*d)
        if L < 0.25: continue
        t2 = d / L; o2 = np.array([t2[1], -t2[0]])
        if o2 @ ((a + b) / 2 - cen) < 0: o2 = -o2
        if o2[1] < -0.95: continue          # faces pointing into the building
        faces.append((a, t2, o2, L))
    T_all = []; C_all = []
    k = 0
    y = y0
    W = lambda p, yy: F.w(p[0], yy, p[1])
    while y + th < y1:
        fr = frac[k % len(frac)]
        for (a, t2, o2, L) in faces:
            for f in fr:
                c = a + t2 * (L * f)
                b0 = c - t2 * size / 2; b1 = c + t2 * size / 2; ap = c + o2 * out
                ct = W((b0 + b1 + ap) / 3, y + th / 2)
                for p, q in ((b0, ap), (ap, b1)):
                    T_all.append([W(p, y), W(q, y), W(q, y + th)]); T_all.append([W(p, y), W(q, y + th), W(p, y + th)])
                T_all.append([W(b0, y + th), W(ap, y + th), W(b1, y + th)])
                T_all.append([W(b0, y), W(b1, y), W(ap, y)])
                C_all += [ct] * 6
        y += row; k += 1
    if not T_all: return
    T = np.array(T_all); C = np.array(C_all)
    tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = np.einsum('ij,ij->i', tn, T.mean(1) - C) < 0
    T[flip] = T[flip][:, ::-1]
    UV = np.zeros((len(T), 3, 2))
    UV[..., 0] = (T[..., 0] + T[..., 2]) / 2.0; UV[..., 1] = T[..., 1] / 2.0
    mb.tris(obj, mat, T, UV)


def courses(mb, F, plan, y0, y1, step=0.33, th=0.075, out=0.035, obj='detail_nocol', mat='klinker'):
    """horizontal projecting brick courses around the outward faces of a pier plan"""
    P = np.asarray(plan, float)
    # offset polygon outward (only for points with e>0 or on faces) - simple: scale around the wall-base midpoint
    cen = P.mean(0)
    y = y0
    while y + th < y1:
        Q = []
        for p in P:
            if abs(p[1]) < 1e-9: Q.append((p[0] + np.sign(p[0] - cen[0]) * out, 0.0))
            else:
                d = p - np.array([cen[0], 0.0]); Q.append(tuple(p + nrm(d) * out * 1.4))
        prism(mb, obj, mat, F, Q, y, y + th, top=True, bottom=True, skip_back=0.0)
        y += step


# ------------------------------------------------------------------ wall with rectangular openings, reveals and spandrels
def wall_grid(mb, F, s0, s1, y0, y1, holes, obj='shell', mat='klinker', spandrel=None, depth=0.2, reveal_mat='klinker',
              sill_mat='concrete', sill=True, hide_reveal_sides=False):
    """planar wall rectangle [s0,s1]x[y0,y1] at e=0 with rectangular holes [(sa,sb,ya,yb)];
    spandrel: list of (sa, sb, ya, yb, s_centre) zones that get the 'chevron' material centred on s_centre."""
    sb_ = sorted(set([s0, s1] + [h[0] for h in holes] + [h[1] for h in holes] + ([z[0] for z in spandrel] + [z[1] for z in spandrel] if spandrel else [])))
    yb_ = sorted(set([y0, y1] + [h[2] for h in holes] + [h[3] for h in holes] + ([z[2] for z in spandrel] + [z[3] for z in spandrel] if spandrel else [])))
    sb_ = [s for s in sb_ if s0 - 1e-9 <= s <= s1 + 1e-9]; yb_ = [y for y in yb_ if y0 - 1e-9 <= y <= y1 + 1e-9]
    # merge cells row-wise with identical material
    for j in range(len(yb_) - 1):
        ya, yb = yb_[j], yb_[j + 1]
        if yb - ya < 1e-6: continue
        ym = (ya + yb) / 2
        run = None
        cells = []
        for i in range(len(sb_) - 1):
            sa, sb = sb_[i], sb_[i + 1]
            if sb - sa < 1e-6: continue
            sm = (sa + sb) / 2
            if any(h[0] - 1e-9 <= sm <= h[1] + 1e-9 and h[2] - 1e-9 <= ym <= h[3] + 1e-9 for h in holes):
                cells.append(None); continue
            z = None
            if spandrel:
                for zz in spandrel:
                    if zz[0] - 1e-9 <= sm <= zz[1] + 1e-9 and zz[2] - 1e-9 <= ym <= zz[3] + 1e-9: z = zz; break
            cells.append((sa, sb, z))
        # merge consecutive same-zone cells
        merged = []
        for c in cells:
            if c is None: merged.append(None); continue
            if merged and merged[-1] is not None and merged[-1][2] is c[2] and abs(merged[-1][1] - c[0]) < 1e-9:
                merged[-1] = (merged[-1][0], c[1], c[2])
            else: merged.append(c)
        for c in merged:
            if c is None: continue
            sa, sb, z = c
            if z is None:
                mb.quad(obj, mat, F.w(sa, ya, 0), F.w(sb, ya, 0), F.w(sb, yb, 0), F.w(sa, yb, 0), F.n)
            else:
                o = F.w(z[4] - 0.7, 0, 0)
                mb.quad(obj, 'chevron', F.w(sa, ya, 0), F.w(sb, ya, 0), F.w(sb, yb, 0), F.w(sa, yb, 0), F.n, uvmode=(o, F.s, UP))
    for (sa, sb, ya, yb) in holes:
        if depth <= 0: continue
        # jambs
        mb.quad(obj, reveal_mat, F.w(sa, ya, 0), F.w(sa, ya, -depth), F.w(sa, yb, -depth), F.w(sa, yb, 0), F.s)
        mb.quad(obj, reveal_mat, F.w(sb, ya, -depth), F.w(sb, ya, 0), F.w(sb, yb, 0), F.w(sb, yb, -depth), -F.s)
        mb.quad(obj, reveal_mat, F.w(sa, yb, 0), F.w(sa, yb, -depth), F.w(sb, yb, -depth), F.w(sb, yb, 0), -UP)
        if sill:
            lbox(mb, obj, sill_mat, F, sa - 0.04, sb + 0.04, ya - 0.06, ya, -depth, 0.05, ('e-', 'y-'))
            mb.quad(obj, sill_mat, F.w(sa - 0.04, ya - 0.06, 0.05), F.w(sb + 0.04, ya - 0.06, 0.05), F.w(sb + 0.04, ya - 0.06, 0.0), F.w(sa - 0.04, ya - 0.06, 0.0), -UP)
        else:
            mb.quad(obj, reveal_mat, F.w(sa, ya, -depth), F.w(sa, ya, 0), F.w(sb, ya, 0), F.w(sb, ya, -depth), UP)

"""Detailed window assemblies in a local frame: x right, y up, z out of the wall (wall face z=0),
opening = [0,w] x [0,h]. Everything is flat-shaded, closed where it can be seen from outside.

Builder collects triangles per material key; world transform applied at the end."""
import math
import numpy as np
import mapbox_earcut as earcut


# ----------------------------------------------------------------------------- builder
class Builder:
    def __init__(self):
        self.parts = {}     # mat -> list of (V (k,3), N (k,3), F (m,3))

    def _add(self, mat, V, N, F):
        self.parts.setdefault(mat, []).append((np.asarray(V, float), np.asarray(N, float), np.asarray(F, np.int64)))

    def tri_count(self):
        return sum(len(F) for L in self.parts.values() for (_, _, F) in L)

    # quad a,b,c,d (counter-clockwise seen from the side the normal points to)
    def quad(self, mat, a, b, c, d, expect=None):
        a, b, c, d = map(np.asarray, (a, b, c, d))
        n = np.cross(b - a, c - a) + np.cross(c - a, d - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return
        n = n / ln
        if expect is not None and np.dot(n, expect) < 0:
            a, b, c, d = a, d, c, b
            n = -n
        self._add(mat, [a, b, c, d], [n] * 4, [[0, 1, 2], [0, 2, 3]])

    def tri(self, mat, a, b, c, expect=None):
        a, b, c = map(np.asarray, (a, b, c))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-14:
            return
        n = n / ln
        if expect is not None and np.dot(n, expect) < 0:
            b, c = c, b
            n = -n
        self._add(mat, [a, b, c], [n] * 3, [[0, 1, 2]])

    def poly(self, mat, P3, normal):
        """planar polygon (possibly concave), P3 (k,3) ordered; normal = desired facing."""
        P3 = np.asarray(P3, float)
        normal = np.asarray(normal, float) / np.linalg.norm(normal)
        # 2D basis
        a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(normal, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        P2 = np.stack([P3 @ e1, P3 @ e2], 1)
        idx = earcut.triangulate_float64(P2, np.array([len(P2)], np.uint32)).reshape(-1, 3)
        if len(idx) == 0:
            return
        F = []
        for t in idx:
            nn = np.cross(P3[t[1]] - P3[t[0]], P3[t[2]] - P3[t[0]])
            F.append(t if np.dot(nn, normal) >= 0 else t[[0, 2, 1]])
        self._add(mat, P3, [normal] * len(P3), F)

    def box(self, mat, lo, hi, skip=()):
        """axis-aligned box in local coords; skip: faces to omit ('-x','+x','-y','+y','-z','+z')."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        if x1 - x0 < 1e-6 or y1 - y0 < 1e-6 or z1 - z0 < 1e-6:
            return
        C = lambda i, j, k: (x1 if i else x0, y1 if j else y0, z1 if k else z0)
        faces = {
            "-x": ([C(0, 0, 0), C(0, 0, 1), C(0, 1, 1), C(0, 1, 0)], (-1, 0, 0)),
            "+x": ([C(1, 0, 0), C(1, 1, 0), C(1, 1, 1), C(1, 0, 1)], (1, 0, 0)),
            "-y": ([C(0, 0, 0), C(1, 0, 0), C(1, 0, 1), C(0, 0, 1)], (0, -1, 0)),
            "+y": ([C(0, 1, 0), C(0, 1, 1), C(1, 1, 1), C(1, 1, 0)], (0, 1, 0)),
            "-z": ([C(0, 0, 0), C(0, 1, 0), C(1, 1, 0), C(1, 0, 0)], (0, 0, -1)),
            "+z": ([C(0, 0, 1), C(1, 0, 1), C(1, 1, 1), C(0, 1, 1)], (0, 0, 1)),
        }
        for k, (q, e) in faces.items():
            if k in skip:
                continue
            self.quad(mat, *q, expect=np.array(e, float))

    def extrude(self, mat, prof, axis, a0, a1, caps=True, place=None):
        """Extrude closed 2D profile prof [(p,q)] along an axis from a0 to a1.
        axis 'x': profile in (y,z); axis 'y': profile in (x,z); axis 'z': profile in (x,y)."""
        prof = np.asarray(prof, float)
        if _signed_area(prof) < 0:
            prof = prof[::-1]
        def P(pq, t):
            p, q = pq
            if axis == "x":
                return np.array([t, p, q])
            if axis == "y":
                return np.array([p, t, q])
            return np.array([p, q, t])
        k = len(prof)
        # outward direction of an edge (p0->p1) for a CCW polygon in (p,q): (dq, -dp)
        for i in range(k):
            p0, p1 = prof[i], prof[(i + 1) % k]
            d = p1 - p0
            o2 = np.array([d[1], -d[0]])
            exp = P(o2, 0.0) - P((0.0, 0.0), 0.0)
            self.quad(mat, P(p0, a0), P(p1, a0), P(p1, a1), P(p0, a1), expect=exp)
        if caps:
            ax = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}[axis]
            self.poly(mat, [P(p, a1) for p in prof], ax if a1 > a0 else -ax)
            self.poly(mat, [P(p, a0) for p in prof], -ax if a1 > a0 else ax)

    def ring(self, mat, x0, y0, x1, y1, prof, zref=0.0):
        """Profiled rectangular ring, mitred corners. prof: closed [(s,z)], s = inset from outer
        edge (>=0), z relative to zref. Outer rect [x0,x1]x[y0,y1]."""
        prof = np.asarray(prof, float)
        # orientation: in (s,z) with s pointing inward; want polygon CCW in (s,z)
        if _signed_area(prof) < 0:
            prof = prof[::-1]
        k = len(prof)

        def corner(s, z):
            return [np.array([x0 + s, y0 + s, zref + z]), np.array([x1 - s, y0 + s, zref + z]),
                    np.array([x1 - s, y1 - s, zref + z]), np.array([x0 + s, y1 - s, zref + z])]
        # side i: from corner i to corner i+1; inward direction per side
        inward = [np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([1.0, 0, 0])]
        for i in range(k):
            s0, z0 = prof[i]
            s1, z1 = prof[(i + 1) % k]
            C0 = corner(s0, z0)
            C1 = corner(s1, z1)
            ds, dz = s1 - s0, z1 - z0
            for sd in range(4):
                # outward 2D normal in (s,z) = (dz, -ds) ; s maps to inward[sd], z to +z
                exp = inward[sd] * dz + np.array([0, 0, 1.0]) * (-ds)
                a, b = C0[sd], C0[(sd + 1) % 4]
                c, d = C1[(sd + 1) % 4], C1[sd]
                self.quad(mat, a, b, c, d, expect=exp)

    def transformed(self, O, r, u, n):
        R = np.stack([r, u, n], 1)   # local -> world: R @ p
        out = {}
        for mat, L in self.parts.items():
            Vs, Ns, Fs, o = [], [], [], 0
            for V, N, F in L:
                Vs.append(V @ R.T + O)
                Ns.append(N @ R.T)
                Fs.append(F + o)
                o += len(V)
            out[mat] = (np.concatenate(Vs), np.concatenate(Ns), np.concatenate(Fs))
        return out


def _signed_area(P):
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ----------------------------------------------------------------------------- profiles
def frame_profile(fw, fd):
    """Blendrahmen: outer face chamfered, inner rebate. (s, z) with z=0 front face."""
    c = 0.006
    return [(0.0, -fd), (0.0, -c), (c, 0.0), (fw - 0.018, 0.0), (fw - 0.012, -0.006),
            (fw - 0.012, -0.020), (fw, -0.026), (fw, -fd)]


def sash_profile(sw, sd, glass_z, ov=0.012):
    """Fluegel incl. glazing bead. s=0 outer edge (overlaps frame by ov), front at z=0."""
    c = 0.005
    b = 0.02          # bead width
    return [(0.0, -sd), (0.0, -c), (c, 0.0), (sw - b - 0.004, 0.0), (sw - b, -0.004),
            (sw - b, -0.008), (sw - 0.004, glass_z + 0.006), (sw, glass_z + 0.002),
            (sw, glass_z - 0.012), (sw - 0.008, glass_z - 0.016), (sw - 0.008, -sd)]


def shop_profile(fw, fd):
    """Aluminium shopfront frame: crisp, slim face with shadow groove."""
    return [(0.0, -fd), (0.0, -0.003), (0.003, 0.0), (fw * 0.45, 0.0), (fw * 0.45, -0.006),
            (fw * 0.55, -0.006), (fw * 0.55, 0.0), (fw - 0.003, 0.0), (fw, -0.003), (fw, -fd)]


def bar_section(wd, dp, ch=0.006):
    """cross-section of mullion/transom bars (p across, q depth), front at q=0."""
    h = wd / 2
    return [(-h, -dp), (h, -dp), (h, -ch), (h - ch, 0.0), (-h + ch, 0.0), (-h, -ch)]


def sill_section(depth_in, proj, thick, drop_front, rise_back, metal=False):
    """Fensterbank cross-section in (y, z): from behind frame (z=-depth_in) to nose (z=proj).
    Top slopes outward; nose with drip groove (stone) or folded lip (metal)."""
    if metal:
        t = 0.004
        lip = 0.035
        return [(rise_back, -depth_in), (drop_front + 0.004, proj - 0.004), (drop_front - 0.002, proj + 0.002),
                (drop_front - lip, proj + 0.002), (drop_front - lip, proj - 0.004), (drop_front - 0.012, proj - 0.004),
                (drop_front - 0.012 - t, proj - 0.012), (rise_back - 0.02, -depth_in)]
    g = 0.01   # drip groove
    return [(rise_back, -depth_in), (drop_front + 0.003, proj - 0.006), (drop_front, proj),
            (drop_front - thick + 0.004, proj), (drop_front - thick, proj - 0.004),
            (drop_front - thick, proj - 0.016), (drop_front - thick + g, proj - 0.020),
            (drop_front - thick + g, proj - 0.030), (drop_front - thick, proj - 0.034),
            (drop_front - thick, -depth_in)]


# ----------------------------------------------------------------------------- pieces
def fixed_light(B, M, x0, y0, x1, y1, zf, glass_z):
    """glass held directly in the frame with a glazing bead ring."""
    bead = [(0.0, -0.012), (0.0, 0.0), (0.004, 0.0), (0.018, -0.010), (0.018, -0.012)]
    B.ring(M["joinery"], x0, y0, x1, y1, bead, zref=zf + glass_z + 0.012)
    B.quad(M["glass"], (x0, y0, zf + glass_z), (x1, y0, zf + glass_z), (x1, y1, zf + glass_z), (x0, y1, zf + glass_z),
           expect=np.array([0, 0, 1.0]))


def handle(B, M, x, y, z, side, inside=True):
    """Fenstergriff (rosette + lever) on the room side of the sash (z = sash back face)."""
    B.box(M["metal"], (x - 0.016, y - 0.034, z - 0.010), (x + 0.016, y + 0.034, z), skip=("+z",))
    B.box(M["metal"], (x - 0.010, y - 0.13, z - 0.030), (x + 0.010, y + 0.012, z - 0.010), skip=("+z",))


def sash(B, M, x0, y0, x1, y1, zf, P, bars=0, handle_side=None, tilt=0.0, vbar=False):
    """one opening sash in the opening rect [x0,x1]x[y0,y1] of the frame (zf = frame front z).
    The sash overlaps the frame opening by P['ov']."""
    ov = P["ov"]
    sw, sd = P["sw"], P["sd"]
    gz = P["glass_z"]
    X0, Y0, X1, Y1 = x0 - ov, y0 - ov, x1 + ov, y1 + ov
    zs = zf + P["sash_proud"]
    sub = Builder()
    sub.ring(M["joinery"], X0, Y0, X1, Y1, sash_profile(sw, sd, gz), zref=zs)
    gx0, gy0, gx1, gy1 = X0 + sw - 0.012, Y0 + sw - 0.012, X1 - sw + 0.012, Y1 - sw + 0.012
    sub.quad(M["glass"], (gx0, gy0, zs + gz), (gx1, gy0, zs + gz), (gx1, gy1, zs + gz), (gx0, gy1, zs + gz),
             expect=np.array([0, 0, 1.0]))
    # glazing bars (Sprossen) across the glass, on both faces
    ix0, ix1 = X0 + sw - 0.004, X1 - sw + 0.004
    iy0, iy1 = Y0 + sw - 0.004, Y1 - sw + 0.004
    for k in range(bars):
        yy = iy0 + (iy1 - iy0) * (k + 1) / (bars + 1)
        sec = [(yy - 0.014, zs + gz - 0.012), (yy + 0.014, zs + gz - 0.012), (yy + 0.014, zs + gz + 0.010),
               (yy + 0.008, zs + gz + 0.016), (yy - 0.008, zs + gz + 0.016), (yy - 0.014, zs + gz + 0.010)]
        sub.extrude(M["joinery"], sec, "x", ix0, ix1, caps=False)
    if vbar:
        xx = (ix0 + ix1) / 2
        sec = [(xx - 0.014, zs + gz - 0.012), (xx + 0.014, zs + gz - 0.012), (xx + 0.014, zs + gz + 0.010),
               (xx + 0.008, zs + gz + 0.016), (xx - 0.008, zs + gz + 0.016), (xx - 0.014, zs + gz + 0.010)]
        sub.extrude(M["joinery"], sec, "y", iy0, iy1, caps=False)
    if handle_side is not None:
        hx = X1 - sw / 2 if handle_side == "R" else X0 + sw / 2
        hy = Y0 + min((Y1 - Y0) * 0.5, 1.05)
        handle(sub, M, hx, hy, zs - sd, handle_side)
    if tilt:
        # Kippstellung: rotate about the bottom edge, top moves into the room
        a = -min(tilt, math.atan(0.09 / max(Y1 - Y0, 0.1)))     # top moves at most 9 cm into the room
        ca, sa = math.cos(a), math.sin(a)
        for mat, L in sub.parts.items():
            for V, N, F in L:
                y = V[:, 1] - Y0
                z = V[:, 2] - zs
                V[:, 1], V[:, 2] = Y0 + y * ca - z * sa, zs + y * sa + z * ca
                ny, nz = N[:, 1].copy(), N[:, 2].copy()
                N[:, 1], N[:, 2] = ny * ca - nz * sa, ny * sa + nz * ca
    for mat, L in sub.parts.items():
        for V, N, F in L:
            B._add(mat, V, N, F)


def bar_v(B, M, x, y0, y1, zf, wd, dp):
    B.extrude(M["joinery"], [(x + p, zf + q) for p, q in bar_section(wd, dp)], "y", y0 - 0.01, y1 + 0.01)


def bar_h(B, M, y, x0, x1, zf, wd, dp):
    B.extrude(M["joinery"], [(y + p, zf + q) for p, q in bar_section(wd, dp)], "x", x0 - 0.01, x1 + 0.01)


def curtains(B, M, rng, x0, y0, x1, y1, z, kind):
    """interior textiles behind the glass. kind: 'side', 'net', 'blind', 'none'."""
    w = x1 - x0
    h = y1 - y0
    mat = M["curtain_a"] if rng.random() < 0.6 else M["curtain_b"]
    def pleat(xa, xb, ya, yb, zz, n, amp):
        xs = np.linspace(xa, xb, n + 1)
        for i in range(n):
            za = zz + (amp if i % 2 else 0.0)
            zb = zz + (0.0 if i % 2 else amp)
            B.quad(mat, (xs[i], ya, za), (xs[i + 1], ya, zb), (xs[i + 1], yb, zb), (xs[i], yb, za),
                   expect=np.array([0, 0, 1.0]))
    if kind == "side":
        cw = min(0.38, w * rng.uniform(0.18, 0.3))
        n = max(3, int(cw / 0.05))
        pleat(x0, x0 + cw, y0, y1, z, n, -0.035)
        pleat(x1 - cw, x1, y0, y1, z, n, -0.035)
        # valance / rod
        B.box(M["metal"], (x0, y1 - 0.03, z - 0.05), (x1, y1 - 0.015, z - 0.035))
    elif kind == "net":
        top = y1
        bot = y0 + h * rng.choice([0.0, 0.0, 0.35, 0.5])
        n = max(4, int(w / 0.06))
        pleat(x0, x1, bot, top, z + 0.01, n, -0.02)
    elif kind == "blind":
        # interior venetian blind, partly lowered
        frac = rng.uniform(0.25, 0.7)
        yb = y1 - h * frac
        mt = M["blind"]
        k = max(2, int((y1 - yb) / 0.05))
        ys = np.linspace(y1 - 0.04, yb, k)
        for yy in ys:
            B.quad(mt, (x0, yy - 0.012, z + 0.012), (x1, yy - 0.012, z + 0.012), (x1, yy + 0.012, z - 0.004),
                   (x0, yy + 0.012, z - 0.004), expect=np.array([0, -0.3, 1.0]))
        B.box(mt, (x0, y1 - 0.04, z - 0.02), (x1, y1, z + 0.02))
        B.box(mt, (x0, yb - 0.02, z - 0.008), (x1, yb, z + 0.008))


def interior_box(B, M, x0, y0, x1, y1, z_front, depth, mat="interior"):
    """open box seen through the glass, faces point into the box opening (+z side)."""
    zb = z_front - depth
    m = M[mat]
    B.quad(m, (x0, y0, zb), (x1, y0, zb), (x1, y1, zb), (x0, y1, zb), expect=np.array([0, 0, 1.0]))
    B.quad(m, (x0, y0, zb), (x0, y0, z_front), (x0, y1, z_front), (x0, y1, zb), expect=np.array([1.0, 0, 0]))
    B.quad(m, (x1, y0, zb), (x1, y1, zb), (x1, y1, z_front), (x1, y0, z_front), expect=np.array([-1.0, 0, 0]))
    B.quad(M.get(mat + "_ceil", m), (x0, y1, zb), (x0, y1, z_front), (x1, y1, z_front), (x1, y1, zb), expect=np.array([0, -1.0, 0]))
    B.quad(m, (x0, y0, zb), (x1, y0, zb), (x1, y0, z_front), (x0, y0, z_front), expect=np.array([0, 1.0, 0]))


def roller_shutter(B, M, x0, x1, ytop, ybot_open, frac, z):
    """Rollladenpanzer between guide rails: lowered from ytop by frac of (ytop-ybot_open)."""
    if frac <= 0.01:
        return
    yb = ytop - (ytop - ybot_open) * frac
    m = M["shutter"]
    pitch = 0.045
    n = max(1, int(round((ytop - yb) / pitch)))
    ys = np.linspace(ytop, yb, n + 1)
    for i in range(n):
        ya, ybb = ys[i], ys[i + 1]
        ym = ya + (ybb - ya) * 0.82
        # slat: slightly convex: two quads per slat
        B.quad(m, (x0, ym, z + 0.008), (x1, ym, z + 0.008), (x1, ya, z), (x0, ya, z), expect=np.array([0, 0.25, 1.0]))
        B.quad(m, (x0, ybb, z), (x1, ybb, z), (x1, ym, z + 0.008), (x0, ym, z + 0.008), expect=np.array([0, -0.6, 1.0]))
    # bottom bar (Endleiste)
    B.box(M["metal"], (x0, yb - 0.02, z - 0.008), (x1, yb + 0.005, z + 0.014))

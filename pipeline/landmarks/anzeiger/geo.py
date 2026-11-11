"""Small mesh-building library (game frame: x east, y up, z south; metres).
All faces are generated with an outward normal hint so the winding is CCW seen from outside (three.js FrontSide).
Geometry is accumulated non-indexed per (object, material); UVs are planar projections in metres / material tile size."""
import numpy as np
import mapbox_earcut as earcut

UP = np.array([0.0, 1.0, 0.0])

# texture tile size in metres (u, v) per material; None = no texture (uv still written)
TILE = {
    'klinker': (2.0, 2.0), 'klinker_dark': (2.0, 2.0), 'chevron': (1.4, 2.0), 'ashlar': (3.2, 3.2), 'stone': (2.0, 2.0),
    'tiles': (1.8, 1.8), 'copper': (2.0, 2.0), 'copper_dark': (2.0, 2.0), 'slate': (1.2, 1.2), 'roof_flat': (4.0, 4.0),
    'plaster': (3.0, 3.0), 'concrete': (2.0, 2.0), 'stone_dark': (2.0, 2.0), 'brick_red': (2.0, 2.0),
}


def nrm(v):
    v = np.asarray(v, float); l = np.linalg.norm(v)
    return v / l if l > 1e-12 else v


def newell(P):
    P = np.asarray(P, float); n = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2]); n[1] += (a[2] - b[2]) * (a[0] + b[0]); n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return n


def plane_uv(P, n, mat, uvmode=None):
    """planar uv (metres / tile). walls: u along horizontal tangent, v = y; roofs: v up-slope; flat: x / -z"""
    P = np.asarray(P, float)
    tu, tv = TILE.get(mat, (1.0, 1.0))
    if uvmode is not None:
        o, a, b = uvmode            # explicit: origin, u-axis, v-axis (metres)
        d = P - o
        return np.stack([d @ a / tu, d @ b / tv], -1)
    ny = abs(n[1])
    if ny < 0.12:
        t = nrm(np.cross(UP, n))
        return np.stack([P @ t / tu, P[:, 1] / tv], -1)
    if ny < 0.97:
        t = nrm(np.cross(UP, n)); s = nrm(UP - n * n[1])
        return np.stack([P @ t / tu, P @ s / tv], -1)
    return np.stack([P[:, 0] / tu, -P[:, 2] / tv], -1) * (1 if n[1] > 0 else np.array([1, 1]))


class MB:
    def __init__(self):
        self.parts = {}          # (obj, mat) -> [list of pos arrays, list of uv arrays]
        self.smooth = set()

    def _acc(self, obj, mat):
        k = (obj, mat)
        if k not in self.parts: self.parts[k] = [[], []]
        return self.parts[k]

    def tris(self, obj, mat, T, UV):
        """T: (m,3,3) triangles (already CCW outward), UV: (m,3,2)"""
        a = self._acc(obj, mat)
        a[0].append(np.asarray(T, np.float32).reshape(-1, 3)); a[1].append(np.asarray(UV, np.float32).reshape(-1, 2))

    def poly(self, obj, mat, P, hint=None, holes=(), uvmode=None):
        P = np.asarray(P, float)
        if len(P) < 3: return
        n = newell(P)
        if np.linalg.norm(n) < 1e-10: return
        n = nrm(n)
        if hint is not None and n @ np.asarray(hint, float) < 0:
            P = P[::-1]; n = -n; holes = [np.asarray(h, float)[::-1] for h in holes]
        if len(P) == 3 and not len(holes):
            T = P[None]
        elif len(P) == 4 and not len(holes) and _convex_quad(P, n):
            T = np.array([[P[0], P[1], P[2]], [P[0], P[2], P[3]]])
        else:
            u = nrm(P[1] - P[0] if np.linalg.norm(P[1] - P[0]) > 1e-9 else P[2] - P[0])
            u = nrm(u - n * (u @ n)); v = np.cross(n, u)
            rings = [P] + [np.asarray(h, float) for h in holes]
            allp = np.concatenate(rings)
            p2 = np.stack([allp @ u, allp @ v], -1)
            ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
            idx = earcut.triangulate_float64(p2, ends).reshape(-1, 3)
            if not len(idx): return
            T = allp[idx]
            # earcut returns CW or CCW depending on input; enforce outward
            tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            flip = tn @ n < 0
            T[flip] = T[flip][:, ::-1]
        UV = plane_uv(T.reshape(-1, 3), n, mat, uvmode).reshape(-1, 3, 2)
        self.tris(obj, mat, T, UV)

    def quad(self, obj, mat, a, b, c, d, hint=None, uvmode=None):
        self.poly(obj, mat, [a, b, c, d], hint, uvmode=uvmode)

    def mesh(self, obj, mat, V, F, UV=None, smooth=False):
        """indexed mesh: V (n,3), F (m,3) CCW outward, UV (n,2) per vertex (or None -> planar)"""
        V = np.asarray(V, float); F = np.asarray(F, int)
        T = V[F]
        if UV is None:
            UVt = np.zeros((len(F), 3, 2))
            tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            for i in range(len(F)):
                UVt[i] = plane_uv(T[i], nrm(tn[i]), mat)
        else:
            UVt = np.asarray(UV, float)[F]
        self.tris(obj, mat, T, UVt)
        if smooth: self.smooth.add(obj)

    def stats(self):
        out = {}
        for (o, m), (P, U) in self.parts.items():
            out.setdefault(o, {})[m] = int(sum(len(p) for p in P) // 3)
        return out

    def total(self):
        return sum(sum(len(p) for p in P) // 3 for P, U in self.parts.values())

    def arrays(self):
        res = {}
        for (o, m), (P, U) in self.parts.items():
            if not P: continue
            res[(o, m)] = (np.concatenate(P), np.concatenate(U))
        return res


def _convex_quad(P, n):
    for i in range(4):
        a, b, c = P[i], P[(i + 1) % 4], P[(i + 2) % 4]
        if np.cross(b - a, c - b) @ n < -1e-12: return False
    return True


# ------------------------------------------------------------------ frames / transforms
class Frame:
    """local facade frame: origin O (3d), s = horizontal tangent, n = outward normal (horizontal), y up.
    local (s, y, e) -> world"""
    def __init__(self, O, s_dir):
        self.O = np.array(O, float)
        s = np.array([s_dir[0], 0.0, s_dir[-1]], float) if len(s_dir) == 2 else np.array(s_dir, float)
        self.s = nrm(s)
        self.n = nrm(np.cross(self.s, UP))     # right-hand: s x up -> outward when walking with building on the left... checked by caller
        self.M = np.stack([self.s, UP, self.n], 1)

    def flip(self):
        self.n = -self.n; self.M = np.stack([self.s, UP, self.n], 1); return self

    def w(self, s, y, e):
        return self.O + self.s * s + UP * y + self.n * e

    def W(self, L):
        L = np.asarray(L, float)
        return self.O + L @ self.M.T

    def dirw(self, d):
        return self.M @ np.asarray(d, float)


def transform_tris(T, R, t):
    """T (m,3,3) local -> world with rotation matrix R (3x3, columns = local axes) and translation t"""
    return T @ R.T + t


# ------------------------------------------------------------------ primitives in a local frame (s, y, e)
def lbox(mb, obj, mat, F, s0, s1, y0, y1, e0, e1, skip=(), uvmode=None):
    """axis-aligned box in frame F. skip: subset of {'s-','s+','y-','y+','e-','e+'}"""
    c = lambda s, y, e: F.w(s, y, e)
    faces = {
        'e+': ([c(s0, y0, e1), c(s1, y0, e1), c(s1, y1, e1), c(s0, y1, e1)], F.n),
        'e-': ([c(s1, y0, e0), c(s0, y0, e0), c(s0, y1, e0), c(s1, y1, e0)], -F.n),
        's+': ([c(s1, y0, e1), c(s1, y0, e0), c(s1, y1, e0), c(s1, y1, e1)], F.s),
        's-': ([c(s0, y0, e0), c(s0, y0, e1), c(s0, y1, e1), c(s0, y1, e0)], -F.s),
        'y+': ([c(s0, y1, e1), c(s1, y1, e1), c(s1, y1, e0), c(s0, y1, e0)], UP),
        'y-': ([c(s0, y0, e0), c(s1, y0, e0), c(s1, y0, e1), c(s0, y0, e1)], -UP),
    }
    for k, (P, h) in faces.items():
        if k in skip: continue
        if min(abs(s1 - s0), abs(y1 - y0), abs(e1 - e0)) <= 0: continue
        mb.poly(obj, mat, P, h, uvmode=uvmode)


def prism(mb, obj, mat, F, poly_se, y0, y1, top=True, bottom=True, skip_back=None):
    """vertical prism over a plan polygon given in (s, e) local coords (any orientation).
    skip_back: e-value -> side faces lying on e == skip_back (wall plane) are skipped."""
    P = np.asarray(poly_se, float)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    # plan coords (s,e) with s x up = n  -> (s,e) is a left-handed pair seen from above; compute outward per edge
    cen = P.mean(0)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        if skip_back is not None and abs(a[1] - skip_back) < 1e-6 and abs(b[1] - skip_back) < 1e-6: continue
        d = b - a
        if np.hypot(*d) < 1e-9: continue
        o2 = np.array([d[1], -d[0]]) if area > 0 else np.array([-d[1], d[0]])   # true outward normal (also for concave plans)
        hint = F.s * o2[0] + F.n * o2[1]
        mb.quad(obj, mat, F.w(a[0], y0, a[1]), F.w(b[0], y0, b[1]), F.w(b[0], y1, b[1]), F.w(a[0], y1, a[1]), hint)
    if top: mb.poly(obj, mat, [F.w(p[0], y1, p[1]) for p in P], UP)
    if bottom: mb.poly(obj, mat, [F.w(p[0], y0, p[1]) for p in P], -UP)


def sweep(mb, obj, mat, path, profile, closed=False, caps=True, smooth=False):
    """sweep a profile [(a, b)] (a = outward offset, b = vertical offset; ordered bottom->top so that the
    outward normal of a segment is (db, -da)) along a horizontal polyline path [(x, y, z)] whose outward side
    is to the right of the walking direction seen from above (o = t x up ... computed as cross(t, UP))."""
    path = np.asarray(path, float); pr = np.asarray(profile, float)
    n = len(path)
    # per-vertex miter normals
    segs = []
    for i in range(n - (0 if closed else 1)):
        t = nrm(path[(i + 1) % n] - path[i]); t[1] = 0; t = nrm(t)
        segs.append(t)
    outs = []
    for i in range(n):
        if closed:
            t0 = segs[i - 1]; t1 = segs[i % len(segs)]
        else:
            t0 = segs[max(i - 1, 0)]; t1 = segs[min(i, len(segs) - 1)]
        o0 = np.cross(t0, UP); o1 = np.cross(t1, UP)
        m = nrm(o0 + o1); cosh = max(m @ o0, 0.3)
        outs.append(m / cosh)
    outs = np.array(outs)
    rings = np.array([[path[i] + outs[i] * a + UP * b for a, b in pr] for i in range(n)])   # (n, k, 3)
    k = len(pr)
    rng_i = range(n if closed else n - 1)
    for i in rng_i:
        j = (i + 1) % n
        o_seg = np.cross(segs[i], UP)
        for q in range(k - 1):
            da, db = pr[q + 1] - pr[q]
            hint = o_seg * db + UP * (-da)
            mb.quad(obj, mat, rings[i, q], rings[j, q], rings[j, q + 1], rings[i, q + 1], hint)
    if caps and not closed:
        for i, sgn in ((0, -1), (n - 1, 1)):
            P = rings[i]
            if len(P) >= 3:
                mb.poly(obj, mat, P, segs[0] * sgn if i == 0 else segs[-1] * sgn)


def lathe(mb, obj, mat, center, profile, segs=48, a0=0.0, a1=2 * np.pi, smooth=True, uv_scale=None, caps=False):
    """surface of revolution around vertical axis through center (x, z); profile [(r, y)] ordered so that
    outward normal of a segment is (dy, -dr) (i.e. bottom -> top for an outward-facing surface)."""
    cx, cz = center
    pr = np.asarray(profile, float); k = len(pr)
    full = abs(a1 - a0 - 2 * np.pi) < 1e-9
    na = segs if full else segs + 1
    ang = a0 + (a1 - a0) * np.arange(na) / segs
    V = np.zeros((na, k, 3))
    V[..., 0] = cx + pr[None, :, 0] * np.cos(ang)[:, None]
    V[..., 2] = cz + pr[None, :, 0] * np.sin(ang)[:, None]
    V[..., 1] = pr[None, :, 1]
    # arc-length uv
    L = np.concatenate([[0], np.cumsum(np.hypot(np.diff(pr[:, 0]), np.diff(pr[:, 1])))])
    tu, tv = TILE.get(mat, (1, 1)) if uv_scale is None else uv_scale
    rmax = pr[:, 0].max()
    F = []; UVv = np.zeros((na, k, 2))
    UVv[..., 0] = (ang[:, None] * rmax) / tu
    UVv[..., 1] = L[None, :] / tv
    # for a full revolution duplicate the seam column so uv does not wrap
    if full:
        V = np.concatenate([V, V[:1]], 0); UVv = np.concatenate([UVv, UVv[:1]], 0); UVv[-1, :, 0] = (a1 * rmax) / tu
        na += 1
    idx = np.arange(na * k).reshape(na, k)
    for i in range(na - 1):
        for q in range(k - 1):
            a, b, c, d = idx[i, q], idx[i + 1, q], idx[i + 1, q + 1], idx[i, q + 1]
            # orientation: angle increases x->z (counter-clockwise seen from below = clockwise from above in x,z with z south)
            F.append([a, c, b]); F.append([a, d, c])
    F = np.array(F)
    Vf = V.reshape(-1, 3); UVf = UVv.reshape(-1, 2)
    # verify orientation with the first non-degenerate tri against intended outward normal
    T = Vf[F]
    tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    cen = T.mean(1)
    radial = cen - np.array([cx, 0, cz]); radial[:, 1] = 0
    # intended normal from profile: (dy, -dr) in (r, y)
    qidx = (F[:, 0] % k)
    q2 = np.minimum(qidx, k - 2)
    dr = pr[q2 + 1, 0] - pr[q2, 0]; dy = pr[q2 + 1, 1] - pr[q2, 1]
    want = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9) * dy[:, None] + UP[None] * (-dr)[:, None]
    good = np.einsum('ij,ij->i', tn, want)
    valid = np.linalg.norm(tn, axis=1) > 1e-12
    if np.sum(good[valid] < 0) > np.sum(good[valid] > 0):
        F = F[:, ::-1]
    F = F[valid]
    mb.mesh(obj, mat, Vf, F, UVf, smooth=smooth)


def ring_pts(cx, cz, r, n, a0=0.0):
    a = a0 + 2 * np.pi * np.arange(n) / n
    return np.stack([cx + r * np.cos(a), cz + r * np.sin(a)], -1)

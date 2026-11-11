"""Geometry builder for the signage overlay.

All shapes are authored in a local frame (x right, y up, z out of the facade) and transformed into the
game frame on commit.  Every triangle carries one palette colour (constant UV = palette cell centre), so the
whole overlay needs only a handful of materials (matte / gloss / metal / lit / glow / vitrine glass).
Faces are flat-shaded, outward (or viewer-) facing: the game renders FrontSide only."""
import math
import os

import mapbox_earcut as earcut
import numpy as np

from common import hex2rgb, rgb2hex

# ----------------------------------------------------------------------------- palette
PAL_N = 32            # 32 x 32 cells
PAL_CELL = 8          # px per cell -> 256 x 256 texture


class Palette:
    def __init__(self):
        self.cols = []
        self.idx = {}

    def get(self, c):
        if isinstance(c, str):
            key = c.lower()
            rgb = hex2rgb(c)
        else:
            rgb = np.clip(np.asarray(c, float), 0, 1)
            key = rgb2hex(rgb)
        if key in self.idx:
            return self.idx[key]
        if len(self.cols) >= PAL_N * PAL_N:
            # palette full: nearest existing colour
            d = np.linalg.norm(np.array(self.cols) - rgb[None], axis=1)
            k = int(np.argmin(d))
            self.idx[key] = k
            return k
        self.cols.append(rgb)
        self.idx[key] = len(self.cols) - 1
        return self.idx[key]

    def uv(self, k):
        i, j = k % PAL_N, k // PAL_N
        return ((i + 0.5) / PAL_N, (j + 0.5) / PAL_N)

    def image(self):
        S = PAL_N * PAL_CELL
        img = np.zeros((S, S, 3), np.uint8)
        img[:] = 128
        for k, c in enumerate(self.cols):
            i, j = k % PAL_N, k // PAL_N
            img[j * PAL_CELL:(j + 1) * PAL_CELL, i * PAL_CELL:(i + 1) * PAL_CELL] = np.round(np.asarray(c) * 255)
        return img


PAL = Palette()

# ----------------------------------------------------------------------------- builder


class Builder:
    """pending (local) geometry -> commit(frame) -> per-material game-frame triangle soups."""

    def __init__(self):
        self.out = {}         # mat -> [V list, N list, C list]
        self.pend = []        # (mat, colidx, V (k,3), N (k,3), F (m,3))
        self.dropped = 0
        self.items = 0

    # -------- low level
    def _add(self, mat, col, V, N, F):
        self.pend.append((mat, PAL.get(col) if not isinstance(col, (int, np.integer)) else int(col),
                          np.asarray(V, float), np.asarray(N, float), np.asarray(F, np.int64)))

    def quad(self, mat, col, a, b, c, d, expect=None):
        a, b, c, d = (np.asarray(p, float) for p in (a, b, c, d))
        n = np.cross(b - a, c - a) + np.cross(c - a, d - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return
        n /= ln
        if expect is not None and np.dot(n, expect) < 0:
            b, d = d, b
            n = -n
        self._add(mat, col, [a, b, c, d], [n] * 4, [[0, 1, 2], [0, 2, 3]])

    def tri(self, mat, col, a, b, c, expect=None):
        a, b, c = (np.asarray(p, float) for p in (a, b, c))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-14:
            return
        n /= ln
        if expect is not None and np.dot(n, expect) < 0:
            b, c = c, b
            n = -n
        self._add(mat, col, [a, b, c], [n] * 3, [[0, 1, 2]])

    def rect_z(self, mat, col, x0, y0, x1, y1, z, facing=1):
        """axis-aligned rectangle in a plane z=const, facing +z (1) or -z (-1)."""
        self.quad(mat, col, (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), expect=np.array([0, 0, facing]))

    def poly(self, mat, col, P3, normal, holes=()):
        P3 = np.asarray(P3, float)
        normal = np.asarray(normal, float)
        normal = normal / np.linalg.norm(normal)
        a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(normal, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        rings = [P3] + [np.asarray(h, float) for h in holes]
        allp = np.concatenate(rings)
        P2 = np.stack([allp @ e1, allp @ e2], 1)
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        idx = earcut.triangulate_float64(P2, ends).reshape(-1, 3)
        if len(idx) == 0:
            return
        F = []
        for t in idx:
            nn = np.cross(allp[t[1]] - allp[t[0]], allp[t[2]] - allp[t[0]])
            F.append(t if np.dot(nn, normal) >= 0 else t[[0, 2, 1]])
        self._add(mat, col, allp, [normal] * len(allp), F)

    def box(self, mat, col, lo, hi, skip=(), cols=None):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        if x1 - x0 < 1e-5 or y1 - y0 < 1e-5 or z1 - z0 < 1e-5:
            return
        C = lambda i, j, k: (x1 if i else x0, y1 if j else y0, z1 if k else z0)  # noqa: E731
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
            m, c = mat, col
            if cols and k in cols:
                m, c = cols[k]
            self.quad(m, c, *q, expect=np.array(e, float))

    def loft(self, mat, col, rings, cap_bottom=False, cap_top=True, cols=None, inner=None):
        """rings: list of (k,3) closed loops (same k, ordered CCW seen from +y); side quads between them.
        inner=colour: also the inside of the shell (same quads facing inwards) - for open shells seen from
        below/inside, e.g. a lamp shade (FrontSide rendering would show its outer faces' backs as a hole)."""
        rings = [np.asarray(r, float) for r in rings]
        k = len(rings[0])
        for ri in range(len(rings) - 1):
            A, B = rings[ri], rings[ri + 1]
            cc = cols[ri] if cols else col
            cen = (A.mean(0) + B.mean(0)) / 2
            for i in range(k):
                j = (i + 1) % k
                out = (A[i] + A[j] + B[i] + B[j]) / 4 - cen
                out[1] = 0
                self.quad(mat, cc, A[i], A[j], B[j], B[i], expect=out if np.linalg.norm(out) > 1e-9 else None)
                if inner is not None and np.linalg.norm(out) > 1e-9:
                    self.quad(mat, inner, A[i], A[j], B[j], B[i], expect=-out)
        if cap_top:
            self.poly(mat, cols[-1] if cols else col, rings[-1], (0, 1, 0))
            if inner is not None:
                self.poly(mat, inner, rings[-1], (0, -1, 0))
        if cap_bottom:
            self.poly(mat, cols[0] if cols else col, rings[0], (0, -1, 0))

    def ellipse_ring(self, cx, y, cz, rx, rz, k=10, rot=0.0):
        t = np.linspace(0, 2 * np.pi, k, endpoint=False) + rot
        return np.stack([cx + rx * np.cos(t), np.full(k, y), cz - rz * np.sin(t)], 1)

    def cyl(self, mat, col, cx, cz, r, y0, y1, k=8, cap_top=True, cap_bottom=False, rz=None):
        rz = r if rz is None else rz
        self.loft(mat, col, [self.ellipse_ring(cx, y0, cz, r, rz, k), self.ellipse_ring(cx, y1, cz, r, rz, k)],
                  cap_bottom=cap_bottom, cap_top=cap_top)

    def frustum(self, mat, col, cx, cz, r0, r1, y0, y1, k=8, cap_top=True, rz0=None, rz1=None, inner=None):
        self.loft(mat, col, [self.ellipse_ring(cx, y0, cz, r0, rz0 or r0, k),
                             self.ellipse_ring(cx, y1, cz, r1, rz1 or r1, k)], cap_top=cap_top, inner=inner)

    def sphere(self, mat, col, c, r, k=8, n=5, ry=None, rz=None):
        ry = r if ry is None else ry
        rz = r if rz is None else rz
        cx, cy, cz = c
        th = np.linspace(-np.pi / 2, np.pi / 2, n + 1)
        rings = []
        for t in th[1:-1]:
            rings.append(self.ellipse_ring(cx, cy + ry * math.sin(t), cz, r * math.cos(t), rz * math.cos(t), k))
        top = np.array([cx, cy + ry, cz])
        bot = np.array([cx, cy - ry, cz])
        for ri in range(len(rings) - 1):
            A, B = rings[ri], rings[ri + 1]
            for i in range(k):
                j = (i + 1) % k
                out = (A[i] + A[j] + B[i] + B[j]) / 4 - np.array(c)
                self.quad(mat, col, A[i], A[j], B[j], B[i], expect=out)
        for i in range(k):
            j = (i + 1) % k
            self.tri(mat, col, rings[-1][i], rings[-1][j], top, expect=(rings[-1][i] + rings[-1][j]) / 2 + top - 2 * np.array(c))
            self.tri(mat, col, rings[0][j], rings[0][i], bot, expect=(rings[0][i] + rings[0][j]) / 2 + bot - 2 * np.array(c))

    def disc(self, mat, col, c, r, normal, k=12):
        normal = np.asarray(normal, float)
        a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(normal, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        t = np.linspace(0, 2 * np.pi, k, endpoint=False)
        P = np.asarray(c, float)[None] + r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
        self.poly(mat, col, P, normal)

    def extrude2d(self, mat, col, polys, z0, z1, side_mat=None, side_col=None, front=True, sides=True):
        """shapely polygons in the xy plane -> prism between z0 and z1 (front face at z1 facing +z, no back face).
        sides=False: front face only (thin vinyl / print a few mm proud of its backing)."""
        from shapely.geometry.polygon import orient
        side_mat = side_mat or mat
        side_col = side_col if side_col is not None else col
        for pg in polys:
            pg = orient(pg, 1.0)
            ext = np.asarray(pg.exterior.coords)[:-1]
            holes = [np.asarray(h.coords)[:-1] for h in pg.interiors]
            if front:
                self.poly(mat, col, np.c_[ext, np.full(len(ext), z1)], (0, 0, 1),
                          holes=[np.c_[h, np.full(len(h), z1)] for h in holes])
            if sides and z1 - z0 > 1e-5:
                for ring in [ext] + holes:
                    k = len(ring)
                    for i in range(k):
                        p, q = ring[i], ring[(i + 1) % k]
                        d = q - p
                        ln = np.hypot(*d)
                        if ln < 1e-6:
                            continue
                        out = np.array([d[1], -d[0], 0.0]) / ln
                        self.quad(side_mat, side_col, (p[0], p[1], z0), (q[0], q[1], z0), (q[0], q[1], z1),
                                  (p[0], p[1], z1), expect=out)

    def mark(self):
        return len(self.pend)

    def xform_since(self, mk, A, t):
        """affine map of the pending geometry added after mark mk: p -> A p + t (mirrors keep the winding right)."""
        A = np.asarray(A, float)
        t = np.asarray(t, float)
        Ninv = np.linalg.inv(A).T
        flip = np.linalg.det(A) < 0
        for i in range(mk, len(self.pend)):
            mat, ci, V, N, F = self.pend[i]
            N2 = N @ Ninv.T
            N2 /= np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-12)
            self.pend[i] = (mat, ci, V @ A.T + t, N2, F[:, ::-1].copy() if flip else F)

    # -------- items
    def commit(self, O, R, bounds=None, tol=1e-4, planes=None):
        """transform pending geometry into the game frame. bounds=(lo, hi) local box: drop the item if any
        vertex lies outside (returns False).  planes: local half-planes (cx, cz, k): cx*x + cz*z <= k
        (mitre-clipped corner boxes)."""
        pend, self.pend = self.pend, []
        if not pend:
            return True
        if bounds is not None:
            lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
            for (_, _, V, _, _) in pend:
                if (V < lo - tol).any() or (V > hi + tol).any():
                    self.dropped += 1
                    return False
        if planes:
            for (_, _, V, _, _) in pend:
                for cx_, cz_, k in planes:
                    if (V[:, 0] * cx_ + V[:, 2] * cz_ > k + tol).any():
                        self.dropped += 1
                        return False
        self.items += 1
        O = np.asarray(O, float)
        R = np.asarray(R, float)
        for mat, ci, V, N, F in pend:
            L = self.out.setdefault(mat, [[], [], [], []])
            L[0].append(V @ R.T + O)
            L[1].append(N @ R.T)
            L[2].append(np.full(len(V), ci, np.int32))
            L[3].append(F)
        return True

    def discard(self):
        self.pend = []

    def tri_count(self):
        return sum(sum(len(f) for f in L[3]) for L in self.out.values())


# ----------------------------------------------------------------------------- text
_FONT_DIR = None
FONTS = {
    "bold-sans": "DejaVuSans-Bold.ttf",
    "sans": "DejaVuSans.ttf",
    "serif": "DejaVuSerif-Bold.ttf",
    "serif-light": "DejaVuSerif.ttf",
    "script": "STIXGeneralBolIta.ttf",
    "italic": "DejaVuSans-BoldOblique.ttf",
    "mono": "DejaVuSansMono-Bold.ttf",
}
_TXT = {}


def font_path(key):
    global _FONT_DIR
    if _FONT_DIR is None:
        import matplotlib
        _FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return os.path.join(_FONT_DIR, FONTS.get(key, FONTS["sans"]))


def text_shapes(s, font="sans", cap=1.0, tol=0.012, squeeze=1.0):
    """shapely polygons of the string, baseline y=0, x from 0, scaled to cap height `cap`.
    returns (polys, width, cap, descent)."""
    key = (s, font, squeeze)
    if key not in _TXT:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        fp = FontProperties(fname=font_path(font))
        Hp = TextPath((0, 0), "H", size=1.0, prop=fp)
        capH = Hp.vertices[:, 1].max()
        tp = TextPath((0, 0), s, size=1.0, prop=fp)
        rings = [np.asarray(r) for r in tp.to_polygons(closed_only=True) if len(r) >= 4]
        polys = []
        for r in rings:
            r = r.copy()
            r[:, 0] *= squeeze
            pg = Polygon(r / capH)
            if not pg.is_valid:
                pg = pg.buffer(0)
            if pg.is_empty or pg.area < 1e-5:
                continue
            polys.append(pg)
        # even-odd nesting
        polys.sort(key=lambda p: -p.area)
        depth = []
        for i, p in enumerate(polys):
            d = 0
            rp = p.representative_point()
            for j in range(i):
                if polys[j].contains(rp):
                    d += 1
            depth.append(d)
        solids = []
        for i, p in enumerate(polys):
            if depth[i] % 2 == 0:
                g = p
                for j in range(i + 1, len(polys)):
                    if depth[j] == depth[i] + 1 and p.contains(polys[j].representative_point()):
                        g = g.difference(polys[j])
                solids.append(g)
        geo = unary_union(solids) if solids else None
        out = []
        if geo is not None and not geo.is_empty:
            geo = geo.simplify(tol, preserve_topology=True)
            for g in getattr(geo, "geoms", [geo]):
                if g.geom_type == "Polygon" and g.area > 1e-5:
                    out.append(g)
        if out:
            b = np.array([g.bounds for g in out])
            minx, miny, maxx = b[:, 0].min(), b[:, 1].min(), b[:, 2].max()
        else:
            minx = miny = maxx = 0.0
        _TXT[key] = (out, minx, maxx, miny)
    out, minx, maxx, miny = _TXT[key]
    from shapely import affinity
    polys = [affinity.scale(affinity.translate(g, -minx, 0), cap, cap, origin=(0, 0)) for g in out]
    return polys, (maxx - minx) * cap, cap, -min(miny, 0) * cap


def text_width(s, font="sans", cap=1.0, squeeze=1.0):
    return text_shapes(s, font, cap, squeeze=squeeze)[1]

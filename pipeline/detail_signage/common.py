"""Shared helpers for the detail-model signage overlay (game frame: metres, Y-up, +X east, -Z north).

Local opening frame used everywhere: x = right as seen from the street, y = up, z = out of the facade
(towards the viewer). R = [r, u, n] (columns), P_game = O + R @ p_local.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glbio  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(HERE, "cache")
RAW_OUT = os.path.join(DATA, "detail_raw", "signage.glb")
WEB_OUT = os.path.join(ROOT, "web", "assets", "city", "detail", "signage.glb")
PREV_DIR = os.path.join(ROOT, "blender", "previews")
PY = sys.executable
BLENDER = r"C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
os.makedirs(CACHE, exist_ok=True)

UP = np.array([0.0, 1.0, 0.0])

# files of the detail model that can hold geometry next to / behind shopfronts
BUILDING_FILES = ["bldg_walls", "bldg_openings", "bldg_base", "bldg_roof", "d0_windows", "fix", "kiosk"]
OTHER_FILES = ["street", "street_furniture", "ground", "green", "landmark_opernhaus", "landmark_tiefgarage_pavillon",
               "landmark_mahnmal"]


def detail_path(name):
    """the version the game shows: detail_final (landmark replacement applied) if present, else detail_raw."""
    p = os.path.join(DATA, "detail_final", name + ".glb")
    if os.path.isfile(p):
        return p
    return os.path.join(DATA, "detail_raw", name + ".glb")


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


# ----------------------------------------------------------------------------- geometry loading
def load_tris(path, keep=None, mat_filter=None):
    """-> T (N,3,3) float64 game frame, mats (N,) int, matnames list.
    keep(centroids (N,3)) -> bool mask (applied per primitive, cheap pre-filter)."""
    g = glbio.read_glb(path)
    Ts, Ms, names = [], [], []
    for nd in g["nodes"]:
        for p in nd["prims"]:
            mat = (p["material"] or {}).get("name", "")
            if mat_filter is not None and not mat_filter(mat):
                continue
            T = p["pos"][p["idx"]]
            if keep is not None and len(T):
                T = T[keep(T.mean(1))]
            if not len(T):
                continue
            if mat not in names:
                names.append(mat)
            Ts.append(T.astype(np.float64))
            Ms.append(np.full(len(T), names.index(mat), np.int32))
    if not Ts:
        return np.zeros((0, 3, 3)), np.zeros(0, np.int32), names
    return np.concatenate(Ts), np.concatenate(Ms), names


class TriSet:
    """triangles of several files + xz grid index for box queries."""

    def __init__(self, cell=4.0):
        self.T = []
        self.mat = []
        self.src = []
        self.matnames = []
        self.cell = cell

    def add(self, T, mats, names, src):
        off = len(self.matnames)
        self.matnames += [src + "::" + n for n in names]
        self.T.append(T)
        self.mat.append(mats + off)
        self.src.append(np.full(len(T), 0, np.int8))

    def finish(self):
        self.T = np.concatenate(self.T) if self.T else np.zeros((0, 3, 3))
        self.mat = np.concatenate(self.mat) if self.mat else np.zeros(0, np.int32)
        lo = self.T.min(1)
        hi = self.T.max(1)
        self.lo, self.hi = lo, hi
        c = self.cell
        i0 = np.floor(lo[:, 0] / c).astype(np.int64)
        i1 = np.floor(hi[:, 0] / c).astype(np.int64)
        k0 = np.floor(lo[:, 2] / c).astype(np.int64)
        k1 = np.floor(hi[:, 2] / c).astype(np.int64)
        grid = {}
        big = []
        single = (i0 == i1) & (k0 == k1)
        ids = np.where(single)[0]
        key = i0[ids] * 100003 + k0[ids]
        order = np.argsort(key, kind="stable")
        ks, starts = np.unique(key[order], return_index=True)
        ends = np.r_[starts[1:], len(order)]
        for kk, s0, s1 in zip(ks, starts, ends):
            t = ids[order[s0]]
            grid[(int(i0[t]), int(k0[t]))] = [ids[order[s0:s1]]]
        for t in np.where(~single)[0]:
            if (i1[t] - i0[t] + 1) * (k1[t] - k0[t] + 1) > 64:
                big.append(t)
                continue
            for i in range(i0[t], i1[t] + 1):
                for k in range(k0[t], k1[t] + 1):
                    grid.setdefault((int(i), int(k)), []).append(np.array([t]))
        self.grid = {k: np.concatenate(v).astype(np.int64) for k, v in grid.items()}
        self.big = np.array(big, np.int64)
        return self

    def query(self, lo, hi):
        """indices of triangles whose bbox overlaps the game-frame box [lo, hi]."""
        c = self.cell
        cand = [self.big]
        for i in range(int(np.floor(lo[0] / c)), int(np.floor(hi[0] / c)) + 1):
            for k in range(int(np.floor(lo[2] / c)), int(np.floor(hi[2] / c)) + 1):
                v = self.grid.get((i, k))
                if v is not None:
                    cand.append(v)
        idx = np.unique(np.concatenate(cand)) if len(cand) > 1 else self.big
        if not len(idx):
            return idx
        m = np.all(self.lo[idx] <= np.asarray(hi) + 1e-9, 1) & np.all(self.hi[idx] >= np.asarray(lo) - 1e-9, 1)
        return idx[m]

    def matname(self, i):
        return self.matnames[self.mat[i]]


# ----------------------------------------------------------------------------- frames
def frame_from_normal(n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    r = np.cross(UP, n)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(n, r)
    return np.stack([r, u, n], 1)


def to_local(P, O, R):
    return (np.asarray(P) - O) @ R


def to_game(p, O, R):
    return np.asarray(p) @ R.T + O


def box_game_bounds(O, R, lo, hi):
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    G = to_game(C, O, R)
    return G.min(0), G.max(0)


def prism_zrange(TL, a, b, c, d, eps=1e-9):
    """Triangles TL (N,3,3) in local coords. Intersection with the infinite prism x in [a,b], y in [c,d]:
    returns hit mask, zmin, zmax of the intersection (exact: extremes lie on clipped-polygon vertices)."""
    N = len(TL)
    if N == 0:
        return np.zeros(0, bool), np.zeros(0), np.zeros(0)
    cand_z = []
    cand_ok = []
    X, Y, Z = TL[:, :, 0], TL[:, :, 1], TL[:, :, 2]
    # 1) vertices inside
    ins = (X >= a - eps) & (X <= b + eps) & (Y >= c - eps) & (Y <= d + eps)
    cand_z.append(Z)
    cand_ok.append(ins)
    # 2) edge / boundary-plane intersections
    for i, j in ((0, 1), (1, 2), (2, 0)):
        Px, Py, Pz = X[:, i], Y[:, i], Z[:, i]
        Qx, Qy, Qz = X[:, j], Y[:, j], Z[:, j]
        for axis, val in ((0, a), (0, b), (1, c), (1, d)):
            P0 = Px if axis == 0 else Py
            Q0 = Qx if axis == 0 else Qy
            den = Q0 - P0
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (val - P0) / den
            ok = (np.abs(den) > 1e-12) & (t >= -eps) & (t <= 1 + eps)
            t = np.clip(np.nan_to_num(t), 0, 1)
            ox = Px + (Qx - Px) * t
            oy = Py + (Qy - Py) * t
            oz = Pz + (Qz - Pz) * t
            if axis == 0:
                ok &= (oy >= c - eps) & (oy <= d + eps)
            else:
                ok &= (ox >= a - eps) & (ox <= b + eps)
            cand_z.append(oz[:, None])
            cand_ok.append(ok[:, None])
    # 3) prism corner lines through the triangle interior
    x0, y0 = X[:, 0], Y[:, 0]
    e1x, e1y = X[:, 1] - x0, Y[:, 1] - y0
    e2x, e2y = X[:, 2] - x0, Y[:, 2] - y0
    det = e1x * e2y - e1y * e2x
    good = np.abs(det) > 1e-12
    sdet = np.where(good, det, 1.0)
    for cx, cy in ((a, c), (a, d), (b, c), (b, d)):
        px, py = cx - x0, cy - y0
        u = (px * e2y - py * e2x) / sdet
        v = (e1x * py - e1y * px) / sdet
        ok = good & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        z = Z[:, 0] + u * (Z[:, 1] - Z[:, 0]) + v * (Z[:, 2] - Z[:, 0])
        cand_z.append(z[:, None])
        cand_ok.append(ok[:, None])
    CZ = np.concatenate(cand_z, 1)
    OK = np.concatenate(cand_ok, 1)
    hit = OK.any(1)
    zmin = np.where(OK, CZ, np.inf).min(1)
    zmax = np.where(OK, CZ, -np.inf).max(1)
    return hit, zmin, zmax


# ----------------------------------------------------------------------------- colours
def hex2rgb(hx):
    hx = (hx or "#808080").strip().lstrip("#")
    if len(hx) == 3:
        hx = "".join(ch * 2 for ch in hx)
    try:
        return np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
    except ValueError:
        return np.array([0.5, 0.5, 0.5])


def rgb2hex(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


def luminance(c):
    c = np.asarray(c, float)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


_HF = None


def heightfield_y(x, z):
    """walkable ground y from web/assets/city/ground/heightfield (1 m grid, bilinear; nan outside / missing)."""
    global _HF
    if _HF is None:
        d = os.path.join(ROOT, "web", "assets", "city", "ground")
        J = jload(os.path.join(d, "heightfield.json"))
        if not J or not os.path.isfile(os.path.join(d, "heightfield.bin")):
            _HF = False
        else:
            A = np.fromfile(os.path.join(d, "heightfield.bin"), np.float32).reshape(J["h"], J["w"])
            _HF = (J, A)
    if not _HF:
        return float("nan")
    J, A = _HF
    fx = (x - J["x0"]) / J["dx"]
    fz = (z - J["z0"]) / J["dz"]
    i0, j0 = int(np.floor(fz)), int(np.floor(fx))
    if not (0 <= i0 < J["h"] - 1 and 0 <= j0 < J["w"] - 1):
        return float("nan")
    tz, tx = fz - i0, fx - j0
    q = A[i0:i0 + 2, j0:j0 + 2].astype(float)
    return float(q[0, 0] * (1 - tx) * (1 - tz) + q[0, 1] * tx * (1 - tz) + q[1, 0] * (1 - tx) * tz + q[1, 1] * tx * tz)


def load_shops():
    """data/shops.json shops with the sign corrections of sign_overrides.json (Street View notes) applied."""
    S = jload(os.path.join(DATA, "shops.json"))["shops"]
    ov = (jload(os.path.join(HERE, "sign_overrides.json"), {}) or {}).get("overrides", [])
    for s in S:
        for o in ov:
            if s.get("name") == o.get("name") and (s.get("address") or "") == o.get("address", s.get("address") or ""):
                sg = dict(s.get("sign") or {})
                sg.update(o.get("sign") or {})
                s["sign"] = sg
                if o.get("display"):
                    dp = dict(s.get("display") or {})
                    dp.update(o["display"])
                    s["display"] = dp
    return S


def jload(p, default=None):
    if not os.path.isfile(p):
        return default
    return json.load(open(p, encoding="utf-8"))


def jdump(obj, p):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)

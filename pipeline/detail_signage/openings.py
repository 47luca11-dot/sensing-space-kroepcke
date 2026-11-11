"""Detect the ground-floor shopfront / door openings of the Rhino detail model and the free space behind
their glass.  -> cache/openings.json (game frame)

Sources (all read-only):
  cache/rhino_openings.json        block placements of layers Bldg::Openings::Shopfronts / Doors (extract_rhino.py)
  pipeline/detail_d0/cache_windows.json   D0 windows rebuilt by the d0 pipeline (kind shop / door)
  data/detail_final/*.glb          the geometry the game shows (glass panes = 'Facade Glass v4')

For every opening: local frame (x right seen from the street, y up, z out of the facade), glass rectangle,
inner glass face zg, and the largest empty box behind the glass (exact triangle/prism clipping against every
building triangle) -> box rect + z_front/z_back.  Nothing is ever placed outside that box.
"""
import os
import re
import time

import numpy as np

from common import (CACHE, DATA, ROOT, BUILDING_FILES, TriSet, detail_path, frame_from_normal, jdump, jload,
                    load_tris, log, prism_zrange, to_local, box_game_bounds)

GLASS_MATS = ("Facade Glass v4",)
MAX_DEPTH = 1.6           # deepest display box (m)
MARGIN = 0.008            # clearance to any existing triangle (no z-fighting)


def load_landmark_polys():
    """footprints whose detail-model building geometry is removed at integration (blender/apply_landmark_replace.py:
    'files' lists the filtered detail files, default = the building files). Replacements that only touch street /
    ground (e.g. U-Bahn stair shafts) do not remove shopfronts and are ignored here."""
    polys = []
    for sub in ("landmarks_replace", "detail_replace"):
        p2 = os.path.join(DATA, sub)
        if not os.path.isdir(p2):
            continue
        for fn in sorted(os.listdir(p2)):
            if not fn.endswith(".json"):
                continue
            J = jload(os.path.join(p2, fn))
            files = set(J.get("files") or [])
            if files and not files & set(BUILDING_FILES):
                continue
            for fp in J.get("footprints", []):
                polys.append((np.array(fp, float), float(J.get("ymin", -2)), float(J.get("ymax", 60)), fn))
    return polys


def inside_poly(poly, x, z):
    c = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, zi = poly[i]
        xj, zj = poly[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi:
            c = not c
        j = i
    return c


def in_landmark(polys, P, buf=1.0):
    for poly, y0, y1, fn in polys:
        lo = poly.min(0) - buf
        hi = poly.max(0) + buf
        if not (lo[0] <= P[0] <= hi[0] and lo[1] <= P[2] <= hi[1]):
            continue
        if inside_poly(poly, P[0], P[2]):
            return fn
        # within buf of the outline
        d = np.min([_seg_dist(P[[0, 2]], poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))])
        if d < buf:
            return fn
    return None


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + ab * t)))


# ----------------------------------------------------------------------------- candidates
def rhino_candidates():
    J = jload(os.path.join(CACHE, "rhino_openings.json"))
    out = []
    for b in J["blocks"]:
        M = np.array(b["M"], float)
        into = M[:3, 1] / np.linalg.norm(M[:3, 1])
        up = M[:3, 2] / np.linalg.norm(M[:3, 2])
        if abs(up[1]) < 0.99:
            continue
        n = -into
        r = np.cross(into, up)
        R = np.stack([r, up, n], 1)
        O = M[:3, 3].copy()
        # frame box in block coords -> local
        fr = b["parts"].get("Frame") or []
        if not fr:
            continue
        lo = np.min([f[0] for f in fr], 0)
        hi = np.max([f[1] for f in fr], 0)
        C = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        L = to_local((C @ M.T)[:, :3], O, R)
        m = re.search(r"sb([0-9.]+)", b["defname"])
        sb = float(m.group(1)) if m else 0.1
        dn = b["defname"]
        if b["layer"] == "door":
            kind = "entrance"
        elif "shopdoor" in dn:
            kind = "shopdoor"
        else:
            kind = "window"
        hw = b["parts"].get("Hardware") or []
        HW = []
        for h in hw:
            Ch = np.array([[x, y, z, 1.0] for x in (h[0][0], h[1][0]) for y in (h[0][1], h[1][1]) for z in (h[0][2], h[1][2])])
            Lh = to_local((Ch @ M.T)[:, :3], O, R)
            HW.append([Lh.min(0).tolist(), Lh.max(0).tolist()])
        out.append(dict(src="blk", sid=b["id"], defname=dn, kind=kind, O=O, R=R,
                        frame=[L[:, 0].min(), L[:, 0].max(), L[:, 1].min(), L[:, 1].max()],
                        zrange=[-0.45, sb + 0.1], wall_z=sb, hardware=HW))
    loose = J["loose"]
    return out, loose


def d0_candidates():
    W = jload(os.path.join(ROOT, "pipeline", "detail_d0", "cache_windows.json"))
    out = []
    for w in W:
        if w["kind"] not in ("shop", "door"):
            continue
        n = np.array(w["n"], float)
        R = frame_from_normal(n)
        c = np.array(w["c"], float)
        O = c - R[:, 0] * w["w"] / 2 - R[:, 1] * w["h"] / 2
        out.append(dict(src="d0", sid=str(w["id"]), defname="d0 %s %.2fx%.2f" % (w["kind"], w["w"], w["h"]),
                        kind="window" if w["kind"] == "shop" else "shopdoor", O=O, R=R,
                        frame=[0.0, w["w"], 0.0, w["h"]], zrange=[-0.6, 0.05], wall_z=0.0, hardware=[],
                        d0_mode=w["mode"]))
    return out


def loose_candidates(loose, glass, obst):
    """custom shopfront meshes: group their glass panes by plane, outward = side with more free space."""
    out = []
    for lo_ in loose:
        if not lo_["mat"].startswith("Facade Glass v4"):
            continue
        V = np.array(lo_["V"], float)
        F = np.array(lo_["F"], int)
        if not len(F):
            continue
        # panes = connected components (shared rounded vertex positions)
        key = np.round(V / 1e-3).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        Fi = inv.reshape(-1)[F]
        par = np.arange(inv.max() + 1)

        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for a, b_, c_ in Fi:
            ra, rb, rc = find(a), find(b_), find(c_)
            par[rb] = ra
            par[find(rc)] = ra
        comp = np.array([find(f[0]) for f in Fi])
        panes = []
        for cid in np.unique(comp):
            T = V[F[comp == cid]]
            nr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            ar = np.linalg.norm(nr, axis=1)
            k = np.argmax(ar)
            n0 = nr[k] / ar[k]
            if abs(n0[1]) > 0.2:
                continue
            n0[1] = 0
            n0 /= np.linalg.norm(n0)
            if n0[0] < -0.7 or (abs(n0[0]) <= 0.7 and n0[2] < 0):
                n0 = -n0          # canonical sign; orientation decided later
            R = frame_from_normal(n0)
            P = T.reshape(-1, 3)
            L = to_local(P, np.zeros(3), R)
            panes.append(dict(n=n0, off=float(L[:, 2].mean()), x0=L[:, 0].min(), x1=L[:, 0].max(),
                              y0=L[:, 1].min(), y1=L[:, 1].max(), P=P))
        # group panes: same plane, touching along x, overlapping in y
        used = [False] * len(panes)
        groups = []
        for i, p in enumerate(panes):
            if used[i]:
                continue
            g = [i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, q in enumerate(panes):
                    if used[j]:
                        continue
                    for k in g:
                        a = panes[k]
                        same = np.dot(a["n"], q["n"]) > 0.995 and abs(a["off"] - q["off"]) < 0.06
                        side = q["x0"] < a["x1"] + 0.25 and q["x1"] > a["x0"] - 0.25 and \
                            min(a["y1"], q["y1"]) - max(a["y0"], q["y0"]) > 0.3 * min(a["y1"] - a["y0"], q["y1"] - q["y0"])
                        stack = q["y0"] < a["y1"] + 0.3 and q["y1"] > a["y0"] - 0.3 and \
                            min(a["x1"], q["x1"]) - max(a["x0"], q["x0"]) > 0.5 * min(a["x1"] - a["x0"], q["x1"] - q["x0"])
                        if same and (side or stack):
                            g.append(j)
                            used[j] = True
                            changed = True
                            break
            groups.append(g)
        for gi, g in enumerate(groups):
            Q = np.concatenate([panes[k]["P"] for k in g])
            n0 = panes[g[0]]["n"]
            for sgn in (1, -1):
                nrm = n0 * sgn
                R = frame_from_normal(nrm)
                O = Q.mean(0)
                L = to_local(Q, O, R)
                x0, x1, y0, y1 = L[:, 0].min(), L[:, 0].max(), L[:, 1].min(), L[:, 1].max()
                O = O + R @ np.array([x0, y0, 0.0])
                x1, y1, x0, y0 = x1 - x0, y1 - y0, 0.0, 0.0
                if x1 < 0.4 or y1 < 0.4:
                    continue
                zg = to_local(Q, O, R)[:, 2].min()
                free = probe_depth(obst, O, R, [x0 + 0.1, x1 - 0.1, y0 + 0.1, y1 - 0.1], zg, 1.5)
                out.append(dict(src="loose", sid="%s:%d:%d" % (lo_["id"], gi, sgn), defname="loose " + lo_["layer"],
                                kind="window" if lo_["layer"] == "shop" else "entrance", O=O, R=R,
                                frame=[x0 - 0.06, x1 + 0.06, y0 - 0.06, y1 + 0.06], zrange=[-0.3, 0.3],
                                wall_z=0.15, hardware=[], _free=free, _pair="%s:%d" % (lo_["id"], gi)))
    # orientation: the point 1.2 m behind the glass must lie inside a LoD2 building, the point in front outside;
    # fallback: the side with the smaller free depth behind is the inside
    best = {}
    for c in out:
        k = c["_pair"]
        ctr = c["O"] + c["R"] @ np.array([(c["frame"][0] + c["frame"][1]) / 2, 1.0, 0.0])
        n = c["R"][:, 2]
        bi = LOD2.building_at(ctr - n * 1.2)
        fo = LOD2.building_at(ctr + n * 1.2)
        score = (2 if bi is not None else 0) + (1 if fo is None else 0) - c["_free"] * 0.1
        c["_score"] = score
        if k not in best or score > best[k]["_score"]:
            best[k] = c
    res = []
    for c in best.values():
        if c["_free"] >= 1.49 and c["_score"] < 2.5:
            continue      # ambiguous: no interior anywhere
        res.append(c)
    return res


class Lod2:
    def __init__(self):
        from shapely.geometry import Polygon, Point
        from shapely.strtree import STRtree
        self.Point = Point
        J = jload(os.path.join(DATA, "lod2_buildings.json"))
        polys, meta = [], []
        for b in J["buildings"]:
            c = b.get("centre") or [0, 0]
            if abs(c[0]) > 450 or abs(c[1]) > 450:
                continue
            for g in b.get("grounds", []):
                ext = [(p[0], -p[1]) for p in g["ext"]]
                if len(ext) < 3:
                    continue
                pg = Polygon(ext, [[(p[0], -p[1]) for p in h] for h in g.get("holes", []) if len(h) >= 3])
                if not pg.is_valid:
                    pg = pg.buffer(0)
                if pg.area < 1:
                    continue
                polys.append(pg)
                meta.append(dict(id=b["id"], street=b.get("street"), number=b.get("number"),
                                 storeys=b.get("storeys"), height=b.get("height"), function=b.get("function")))
        self.polys = polys
        self.meta = meta
        self.tree = STRtree(polys)

    def building_at(self, P, buf=0.0):
        pt = self.Point(P[0], P[2])
        for i in self.tree.query(pt.buffer(buf) if buf else pt):
            if self.polys[i].buffer(buf).contains(pt) if buf else self.polys[i].contains(pt):
                return i
        return None

    def nearest(self, P, maxd=3.0):
        pt = self.Point(P[0], P[2])
        best, bd = None, maxd
        for i in self.tree.query(pt.buffer(maxd)):
            d = self.polys[i].exterior.distance(pt)
            if d < bd:
                best, bd = i, d
        return best, bd


LOD2 = None


def probe_depth(obst, O, R, rect, zg, maxd):
    a, b, c, d = rect
    lo, hi = box_game_bounds(O, R, [a, c, zg - maxd], [b, d, zg])
    idx = obst.query(lo - 0.01, hi + 0.01)
    if not len(idx):
        return maxd
    TL = to_local(obst.T[idx].reshape(-1, 3), O, R).reshape(-1, 3, 3)
    hit, zmin, zmax = prism_zrange(TL, a, b, c, d)
    m = hit & (zmin < zg - 1e-3) & (zmax > zg - maxd)
    if not m.any():
        return maxd
    return float(zg - min(zg, zmax[m].max()))


# ----------------------------------------------------------------------------- glass + free box
def glass_of(c, glass):
    O, R = c["O"], c["R"]
    fx0, fx1, fy0, fy1 = c["frame"]
    z0, z1 = c["zrange"]
    lo, hi = box_game_bounds(O, R, [fx0 - 0.02, fy0 - 0.02, z0], [fx1 + 0.02, fy1 + 0.02, z1])
    idx = glass.query(lo, hi)
    if not len(idx):
        return None
    TL = to_local(glass.T[idx].reshape(-1, 3), O, R).reshape(-1, 3, 3)
    cen = TL.mean(1)
    m = (cen[:, 0] > fx0 - 0.01) & (cen[:, 0] < fx1 + 0.01) & (cen[:, 1] > fy0 - 0.01) & (cen[:, 1] < fy1 + 0.01) & \
        (cen[:, 2] > z0) & (cen[:, 2] < z1)
    # only panes parallel to the facade
    nr = np.cross(TL[:, 1] - TL[:, 0], TL[:, 2] - TL[:, 0])
    ln = np.linalg.norm(nr, axis=1)
    par = np.abs(nr[:, 2]) > 0.95 * np.maximum(ln, 1e-12)
    m &= par & (ln > 1e-8)
    if m.sum() < 2:
        return None
    P = TL[m].reshape(-1, 3)
    # panes: connected by bbox overlap in xy (group triangles of the same pane)
    T2 = TL[m]
    panes = []
    bb = np.stack([T2[:, :, 0].min(1), T2[:, :, 0].max(1), T2[:, :, 1].min(1), T2[:, :, 1].max(1)], 1)
    order = np.argsort(bb[:, 0])
    for t in order:
        x0, x1, y0, y1 = bb[t]
        for p in panes:
            if x0 < p[1] - 0.005 and x1 > p[0] + 0.005 and y0 < p[3] - 0.005 and y1 > p[2] + 0.005:
                p[0], p[1], p[2], p[3] = min(p[0], x0), max(p[1], x1), min(p[2], y0), max(p[3], y1)
                break
        else:
            panes.append([x0, x1, y0, y1])
    # merge again (transitive)
    changed = True
    while changed:
        changed = False
        for i in range(len(panes)):
            for j in range(i + 1, len(panes)):
                p, q = panes[i], panes[j]
                if p[0] < q[1] - 0.005 and p[1] > q[0] + 0.005 and p[2] < q[3] - 0.005 and p[3] > q[2] + 0.005:
                    panes[i] = [min(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), max(p[3], q[3])]
                    panes.pop(j)
                    changed = True
                    break
            if changed:
                break
    panes = [p for p in panes if (p[1] - p[0]) > 0.12 and (p[3] - p[2]) > 0.12]
    if not panes:
        return None
    return dict(rect=[P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()], zg=float(P[:, 2].min()),
                zgo=float(P[:, 2].max()), panes=panes)


def coverage(T2, a, b, c, d, panes=None):
    """fraction of the glass (pane rects within [a,b]x[c,d], or the whole rect) covered by the xy projections of
    the triangles T2 (k,3,2) (exact polygon union).  Triangles seen edge-on (perpendicular slabs, reveals) project
    to nothing and do not count; a solid door panel beside / below the glass does not count either."""
    if not len(T2) or b <= a or d <= c:
        return 0.0
    e1 = T2[:, 1] - T2[:, 0]
    e2 = T2[:, 2] - T2[:, 0]
    ar = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
    # cheap necessary condition first (sum of the clipped bbox areas): wide fronts with many thin mullions
    bx0 = np.clip(T2[:, :, 0].min(1), a, b)
    bx1 = np.clip(T2[:, :, 0].max(1), a, b)
    by0 = np.clip(T2[:, :, 1].min(1), c, d)
    by1 = np.clip(T2[:, :, 1].max(1), c, d)
    if float(((bx1 - bx0) * (by1 - by0))[ar > 1e-8].sum()) < 0.5 * (b - a) * (d - c) * (0.2 if panes else 1.0):
        return 0.0
    from shapely.geometry import Polygon, box as sbox
    from shapely.ops import unary_union
    ref = sbox(a, c, b, d)
    if panes:
        ref = unary_union([sbox(p[0], p[2], p[1], p[3]) for p in panes]).intersection(ref)
        if ref.area < 1e-6:
            return 0.0
    e1 = T2[:, 1] - T2[:, 0]
    e2 = T2[:, 2] - T2[:, 0]
    ar = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
    polys = [Polygon(t) for t in T2[ar > 1e-8]]
    if not polys:
        return 0.0
    U = unary_union([p if p.is_valid else p.buffer(0) for p in polys]).intersection(ref)
    return float(U.area / ref.area)


def free_box(c, g, obst, rect=None, min_w=0.3):
    """largest box behind the glass: lateral rect = glass rect (shrunk if the frame reaches behind the glass),
    front just behind the deepest glass face (or behind thin frame parts), back in front of the first obstacle.
    rect: lateral sub-range of the glass (segments between posts, see free_segments).
    An opaque panel right behind the glass (> 50 % of the window, within the sweep range, e.g. the d0 'Facade
    Interior Dark' board 7 mm behind the pane) stops the sweep: the model already closes that window and a box
    behind the panel would be invisible -> depth 0, covered=True."""
    O, R = c["O"], c["R"]
    gx0, gx1, gy0, gy1 = rect or g["rect"]
    zg = g["zg"]
    lo, hi = box_game_bounds(O, R, [gx0 - 0.05, gy0 - 0.05, zg - MAX_DEPTH - 0.05], [gx1 + 0.05, gy1 + 0.05, zg + 0.01])
    idx = obst.query(lo, hi)
    TL = to_local(obst.T[idx].reshape(-1, 3), O, R).reshape(-1, 3, 3) if len(idx) else np.zeros((0, 3, 3))
    best = None
    for s in (0.0, 0.01, 0.02, 0.035, 0.05, 0.07, 0.1, 0.13):
        a, b, cc, d = gx0 + s, gx1 - s, gy0 + s, gy1 - s
        if b - a < min_w or d - cc < 0.3:
            break
        hit, zmin, zmax = prism_zrange(TL, a - MARGIN, b + MARGIN, cc - MARGIN, d + MARGIN)
        m = hit & (zmin < zg - 1e-4)
        mi = np.where(m)[0]
        zl, zh = zmin[m], np.minimum(zmax[m], zg)
        # front: sweep down through thin parts within 9 cm of the glass
        zf = zg - 0.012
        ok = True
        for _ in range(50):
            bl = zh > zf - MARGIN
            if not bl.any():
                break
            Tb = TL[mi[bl]]
            nb = np.cross(Tb[:, 1] - Tb[:, 0], Tb[:, 2] - Tb[:, 0])
            par = np.abs(nb[:, 2]) >= 0.85 * np.maximum(np.linalg.norm(nb, axis=1), 1e-12)   # panels parallel to the pane
            if par.any() and coverage(Tb[par][:, :, :2], a, b, cc, d, g.get("panes")) > 0.5:
                k = mi[bl][par][int(np.argmax(zh[bl][par]))]
                return dict(rect=[a + 0.0025, b - 0.0025, cc + 0.0025, d - 0.0025], z_front=float(zf), z_back=float(zf),
                            depth=0.0, shrink=s, blocker=obst.matname(idx[k]), covered=True)
            zf = zl[bl].min() - MARGIN
            if zg - zf > 0.16:
                ok = False
                break
        if not ok:
            continue
        behind = zh <= zf - MARGIN + 1e-9
        zb = max(zh[behind].max() + MARGIN, zg - MAX_DEPTH) if behind.any() else zg - MAX_DEPTH
        D = zf - zb
        # 2.5 mm inset: the shell never lies in the plane of a frame's inner face (no coplanar contact)
        cand = dict(rect=[a + 0.0025, b - 0.0025, cc + 0.0025, d - 0.0025], z_front=float(zf), z_back=float(zb),
                    depth=float(D), shrink=s, blocker=None)
        if behind.any():
            k = np.argmax(np.where(behind, zh, -np.inf))
            cand["blocker"] = obst.matname(idx[np.where(m)[0][k]])
        if best is None or D > best["depth"] + 0.05:
            best = cand
        if D >= 0.30:
            break
    return best


def free_segments(c, g, obst):
    """glass units whose prism is crossed close behind the pane by a deep element (the other facade's interior
    slab at a building corner, a column): 3 cm strips across the glass, front sweep per strip; runs of strips with
    a free depth >= 12 cm become separate boxes (left of the post, and the corner pocket right of it)."""
    O, R = c["O"], c["R"]
    gx0, gx1, gy0, gy1 = g["rect"]
    zg = g["zg"]
    lo, hi = box_game_bounds(O, R, [gx0 - 0.05, gy0 - 0.05, zg - MAX_DEPTH - 0.05], [gx1 + 0.05, gy1 + 0.05, zg + 0.01])
    idx = obst.query(lo, hi)
    if not len(idx):
        return []
    TL = to_local(obst.T[idx].reshape(-1, 3), O, R).reshape(-1, 3, 3)
    n = max(1, int(round((gx1 - gx0) / 0.03)))
    xs = np.linspace(gx0, gx1, n + 1)
    okc = np.zeros(n, bool)
    for i in range(n):
        hit, zmin, zmax = prism_zrange(TL, xs[i] - 0.002, xs[i + 1] + 0.002, gy0 + 0.06, gy1 - 0.06)
        m = hit & (zmin < zg - 1e-4)
        zl, zh = zmin[m], np.minimum(zmax[m], zg)
        zf = zg - 0.012
        ok = True
        for _ in range(50):
            bl = zh > zf - MARGIN
            if not bl.any():
                break
            zf = zl[bl].min() - MARGIN
            if zg - zf > 0.16:
                ok = False
                break
        if ok:
            behind = zh <= zf - MARGIN + 1e-9
            zb = max(zh[behind].max() + MARGIN, zg - MAX_DEPTH) if behind.any() else zg - MAX_DEPTH
            okc[i] = zf - zb >= 0.12
    segs = []
    i = 0
    while i < n:
        if not okc[i]:
            i += 1
            continue
        j = i
        while j < n and okc[j]:
            j += 1
        segs.append((xs[i], xs[j]))
        i = j
    out = []
    for xa, xb in segs:
        if xb - xa < 0.14:
            continue
        fb = free_box(c, g, obst, rect=[xa, xb, gy0, gy1], min_w=0.1)
        if fb is not None and fb["depth"] >= 0.1 and not fb.get("covered"):
            fb["segment"] = [float(xa), float(xb)]
            out.append(fb)
    return out


def clip_for(rx, ry):
    """mitre half-plane for box X against box Y (both dicts with O, R, zf): the points of X's box that lie deeper
    behind X's glass than behind Y's glass go to Y.  Local X coords: cx*x + cz*z <= k."""
    OX, RX, zfX = rx
    OY, RY, zfY = ry
    nX, rX, nY = RX[:, 2], RX[:, 0], RY[:, 2]
    gap = 0.006
    return [float(rX @ nY), float(nX @ nY - 1.0), float(-gap - zfX + zfY - (OX - OY) @ nY)]


def in_clips(L, clips, tol=0.0):
    ok = np.ones(len(L), bool)
    for cx_, cz_, k in clips:
        ok &= L[:, 0] * cx_ + L[:, 2] * cz_ <= k + tol
    return ok


def resolve_overlaps(recs):
    """boxes of different openings that share volume (building corners, acute corners): each keeps the part that
    is closer to its own glass than to the other's (mitre plane = bisector of the two glass planes), stored as
    box['clip'] half-planes; the shell gets a mitre wall there (displays.shell)."""
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    items = []
    for r in recs:
        b = r["box"]
        if b["depth"] < 0.03:
            continue
        O, R = np.array(r["O"]), np.array(r["R"])
        a, bb, c, d = b["rect"]
        P = np.array([[a, 0, b["z_front"]], [bb, 0, b["z_front"]], [bb, 0, b["z_back"]], [a, 0, b["z_back"]]]) @ R.T + O
        pg = Polygon(P[:, [0, 2]])
        if not pg.is_valid or pg.area < 1e-6:
            continue
        items.append((r, pg, O[1] + c, O[1] + d, O, R))
    if not items:
        return 0
    tree = STRtree([it[1] for it in items])
    n_pairs = 0
    rng = np.random.default_rng(5)
    for i, (r, pg, y0, y1, O, R) in enumerate(items):
        for j in tree.query(pg):
            j = int(j)
            if j <= i:
                continue
            r2, pg2, y2, y3, O2, R2 = items[j]
            if min(y1, y3) - max(y0, y2) < 0.02:
                continue
            if pg.intersection(pg2).area < 2e-4:
                continue
            if abs(R[:, 2] @ R2[:, 2]) > 0.95 and R[:, 2] @ R2[:, 2] > 0:
                log("  overlap of parallel boxes (not resolved):", r["id"], r2["id"])
                continue
            cA = clip_for((O, R, r["box"]["z_front"]), (O2, R2, r2["box"]["z_front"]))
            cB = clip_for((O2, R2, r2["box"]["z_front"]), (O, R, r["box"]["z_front"]))
            r["box"].setdefault("clip", []).append(cA)
            r2["box"].setdefault("clip", []).append(cB)
            n_pairs += 1
            # verify: sample A's clipped box, none of it inside B's clipped box
            a, bb, c, d = r["box"]["rect"]
            Ls = rng.uniform([a, c, r["box"]["z_back"]], [bb, d, r["box"]["z_front"]], (6000, 3))
            Ls = Ls[in_clips(Ls, r["box"]["clip"])]
            G = O + Ls @ R.T
            L2 = (G - O2) @ R2
            a2, b2, c2, d2 = r2["box"]["rect"]
            ins = (L2[:, 0] > a2) & (L2[:, 0] < b2) & (L2[:, 1] > c2) & (L2[:, 1] < d2) & \
                (L2[:, 2] > r2["box"]["z_back"]) & (L2[:, 2] < r2["box"]["z_front"]) & in_clips(L2, r2["box"]["clip"])
            # the open fronts must stay on their own side
            fr = np.array([[a, (c + d) / 2, r["box"]["z_front"]], [bb, (c + d) / 2, r["box"]["z_front"]]])
            log("  mitre", r["id"], "|", r2["id"], "overlap left %d samples, A front corners kept %s" % (
                int(ins.sum()), in_clips(fr, r["box"]["clip"]).tolist()))
    return n_pairs


def load_scene(lo_xz, hi_xz, ymax, t0=None):
    """glass panes + every building triangle of the detail model (as shown in the game) + adjacent landmarks."""
    t0 = t0 or time.time()

    def keep(C):
        return (C[:, 0] > lo_xz[0]) & (C[:, 0] < hi_xz[0]) & (C[:, 2] > lo_xz[2]) & (C[:, 2] < hi_xz[2]) & (C[:, 1] < ymax)

    glass = TriSet(cell=2.0)
    obst = TriSet(cell=2.0)
    for name in BUILDING_FILES + ["street_furniture", "landmark_opernhaus", "landmark_tiefgarage_pavillon",
                                  "landmark_mahnmal", "green_trees"]:
        p = detail_path(name)
        if not os.path.isfile(p):
            continue
        T, M, names = load_tris(p, keep=keep)
        if not len(T):
            continue
        gm = np.array([any(n.startswith(g) for g in GLASS_MATS) for n in names])
        isg = gm[M]
        if name in ("bldg_openings", "d0_windows") and isg.any():
            glass.add(T[isg], M[isg], names, name)
        obst.add(T, M, names, name)
        log("loaded", name, len(T), "tris (%.1fs)" % (time.time() - t0))
    for lm in ("cafe_kroepcke", "kroepcke_center"):
        p = os.path.join(DATA, "landmarks_raw", lm + ".glb")
        if os.path.isfile(p):
            T, M, names = load_tris(p, keep=keep)
            obst.add(T, M, names, "landmark_" + lm)
            log("loaded landmark", lm, len(T))
    glass.finish()
    obst.finish()
    log("glass tris", len(glass.T), "obstacle tris", len(obst.T), "(%.1fs)" % (time.time() - t0))
    return glass, obst


def glass_candidates(cands, glass, obst):
    """ground-floor display windows that are not on the Shopfronts layer (Rhino 'Windows' blocks with a dark
    interior panel, e.g. Georgstrasse 42): every vertical glass pane of the detail model near street level that
    no other candidate covers -> grouped into window rectangles, outward side from LoD2 + free depth."""
    T = glass.T
    nr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(nr, axis=1)
    vert = (ln > 1e-8) & (np.abs(nr[:, 1]) < 0.2 * np.maximum(ln, 1e-12))
    covered = np.zeros(len(T), bool)
    for c in cands:
        O, R = c["O"], c["R"]
        fx0, fx1, fy0, fy1 = c["frame"]
        lo, hi = box_game_bounds(O, R, [fx0 - 0.1, fy0 - 0.1, -0.7], [fx1 + 0.1, fy1 + 0.1, 0.7])
        idx = glass.query(lo, hi)
        if not len(idx):
            continue
        L = to_local(T[idx].mean(1), O, R)
        m = (L[:, 0] > fx0 - 0.1) & (L[:, 0] < fx1 + 0.1) & (L[:, 1] > fy0 - 0.1) & (L[:, 1] < fy1 + 0.1) & \
            (np.abs(L[:, 2]) < 0.7)
        covered[idx[m]] = True
    sel = np.where(vert & ~covered)[0]
    log("glass tris: vertical", int(vert.sum()), "covered", int((vert & covered).sum()), "free", len(sel))
    if not len(sel):
        return []
    # panes = connected components (shared rounded vertices)
    V = T[sel].reshape(-1, 3)
    key = np.round(V / 2e-3).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    Fi = inv.reshape(-1, 3)
    par = np.arange(inv.max() + 1)

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b_, c_ in Fi:
        ra, rb, rc_ = find(a), find(b_), find(c_)
        par[rb] = ra
        par[find(rc_)] = ra
    comp = np.array([find(f[0]) for f in Fi])
    panes = []
    for cid in np.unique(comp):
        Tc = T[sel[comp == cid]]
        nn_ = nr[sel[comp == cid]]
        n0 = nn_[np.argmax(np.linalg.norm(nn_, axis=1))].copy()     # (panes are thin boxes: sums cancel)
        n0[1] = 0
        if np.linalg.norm(n0) < 1e-9:
            continue
        n0 = n0 / np.linalg.norm(n0)
        if n0[0] < -0.7 or (abs(n0[0]) <= 0.7 and n0[2] < 0):
            n0 = -n0
        R = frame_from_normal(n0)
        P = Tc.reshape(-1, 3)
        L = to_local(P, np.zeros(3), R)
        if np.ptp(L[:, 2]) > 0.08:
            continue          # not planar (bay / curved glazing)
        panes.append(dict(n=n0, off=float(L[:, 2].mean()), x0=L[:, 0].min(), x1=L[:, 0].max(), y0=L[:, 1].min(),
                          y1=L[:, 1].max(), P=P))
    # group: same plane, stacked (transoms) only - side-by-side panes stay separate openings, because the
    # mullions between them reach behind the glass and would block a common display box
    used = [False] * len(panes)
    out = []
    dbg = {}
    for i in range(len(panes)):
        if used[i]:
            continue
        g = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, q in enumerate(panes):
                if used[j]:
                    continue
                for k in g:
                    a = panes[k]
                    same = np.dot(a["n"], q["n"]) > 0.995 and abs(a["off"] - q["off"]) < 0.06
                    side = q["x0"] < a["x1"] + 0.2 and q["x1"] > a["x0"] - 0.2 and \
                        min(a["y1"], q["y1"]) - max(a["y0"], q["y0"]) > 0.5 * min(a["y1"] - a["y0"], q["y1"] - q["y0"])
                    stack = q["y0"] < a["y1"] + 0.25 and q["y1"] > a["y0"] - 0.25 and \
                        min(a["x1"], q["x1"]) - max(a["x0"], q["x0"]) > 0.6 * min(a["x1"] - a["x0"], q["x1"] - q["x0"])
                    if same and stack and not side:
                        g.append(j)
                        used[j] = True
                        changed = True
                        break
        Q = np.concatenate([panes[k]["P"] for k in g])
        n0 = panes[g[0]]["n"]
        # street level at the window: lowest building triangle within 1.5 m (walls / plinth reach the ground)
        cxz = Q.mean(0)
        idx = obst.query(np.array([cxz[0] - 1.5, -50, cxz[2] - 1.5]), np.array([cxz[0] + 1.5, Q[:, 1].min() + 0.5, cxz[2] + 1.5]))
        ground = float(obst.lo[idx, 1].min()) if len(idx) else Q[:, 1].min()
        ybot, ytop = Q[:, 1].min(), Q[:, 1].max()
        if ybot - ground > 0.9 or ytop - ybot < 1.5:
            dbg["level"] = dbg.get("level", 0) + 1
            continue
        best = None
        for sgn in (1, -1):
            nrm = n0 * sgn
            R = frame_from_normal(nrm)
            O = Q.mean(0)
            L = to_local(Q, O, R)
            x0, x1, y0, y1 = L[:, 0].min(), L[:, 0].max(), L[:, 1].min(), L[:, 1].max()
            O = O + R @ np.array([x0, y0, 0.0])
            W, H = x1 - x0, y1 - y0
            if W < 0.45:
                break
            zg = to_local(Q, O, R)[:, 2].min()
            ctr = O + R @ np.array([W / 2, min(1.0, H / 2), 0.0])
            bi = LOD2.building_at(ctr - nrm * 1.2)
            fo = LOD2.building_at(ctr + nrm * 1.5)
            free = probe_depth(obst, O, R, [0.1, W - 0.1, 0.1, min(H, 3.0) - 0.1], zg, 1.0)
            score = (2 if bi is not None else 0) + (1 if fo is None else 0) - free * 0.3
            c = dict(src="gl", sid="%.2f,%.2f,%.2f" % tuple(O), defname="glass %.2fx%.2f" % (W, H), kind="window", O=O, R=R,
                     frame=[-0.06, W + 0.06, -0.06, H + 0.06], zrange=[-0.3, 0.3], wall_z=0.15, hardware=[], _score=score)
            if best is None or score > best["_score"]:
                best = c
        if best is not None and best["_score"] >= 2.0:
            best.pop("_score")
            out.append(best)
        else:
            dbg["orient"] = dbg.get("orient", 0) + 1
    log("glass groups", len(panes), "panes;", dbg)
    return out


def main():
    global LOD2
    t0 = time.time()
    LOD2 = Lod2()
    polys = load_landmark_polys()
    rc, loose = rhino_candidates()
    dc = d0_candidates()
    cands = rc + dc
    allP = np.array([c["O"] for c in cands] + [np.array(lo_["V"]).mean(0) for lo_ in loose if lo_["V"]])
    glass, obst = load_scene(allP.min(0) - 6, allP.max(0) + 6, allP[:, 1].max() + 8, t0)
    cands += loose_candidates(loose, glass, obst)
    gc = glass_candidates(cands, glass, obst)
    log("glass-only ground-floor windows", len(gc))
    cands += gc
    log("candidates", len(cands), {k: sum(c["src"] == k for c in cands) for k in ("blk", "d0", "loose", "gl")})
    res = []
    skipped = {}
    for i, c in enumerate(cands):
        Pc = c["O"] + c["R"] @ np.array([(c["frame"][0] + c["frame"][1]) / 2, 1.0, 0.0])
        lm = in_landmark(polys, Pc, buf=0.8)
        if lm:
            skipped["landmark:" + lm] = skipped.get("landmark:" + lm, 0) + 1
            continue
        g = glass_of(c, glass)
        if g is None:
            skipped["no glass"] = skipped.get("no glass", 0) + 1
            continue
        fb = free_box(c, g, obst)
        boxes = [fb]
        if fb is None or (fb["depth"] < 0.03 and not fb.get("covered")):
            # corner units: the other facade's interior slab crosses the prism right behind the pane
            segs = free_segments(c, g, obst)
            if segs:
                boxes = segs
                skipped["segmented"] = skipped.get("segmented", 0) + 1
                log("  segmented", "%s:%s" % (c["src"], c["sid"]), [b_["segment"] for b_ in segs])
        if boxes[0] is None or max(b_["depth"] for b_ in boxes) < 0.10:
            skipped["no depth"] = skipped.get("no depth", 0) + 1
            if fb is not None and fb.get("covered"):
                skipped["covered by panel"] = skipped.get("covered by panel", 0) + 1
            boxes = [fb or dict(rect=g["rect"], z_front=g["zg"] - 0.012, z_back=g["zg"] - 0.012, depth=0.0, shrink=0)]
        gc = c["O"] + c["R"] @ np.array([(g["rect"][0] + g["rect"][1]) / 2, (g["rect"][2] + g["rect"][3]) / 2, g["zg"]])
        n = c["R"][:, 2]
        bi = LOD2.building_at(gc - n * 1.5)
        if bi is None:
            bi, _ = LOD2.nearest(gc - n * 0.5, 2.5)
        outside_ok = LOD2.building_at(gc + n * 1.0) is None
        bmeta = LOD2.meta[bi] if bi is not None else {}
        base_id = "%s:%s" % (c["src"], c["sid"])
        for k, bx in enumerate(boxes):
            frame = [float(v) for v in c["frame"]]
            rid = base_id
            if len(boxes) > 1:
                # sub-opening: one box per free segment of the glass (shop assignment by the base id)
                rid = "%s#s%d" % (base_id, k)
                frame = [max(frame[0], bx["segment"][0] - 0.06), min(frame[1], bx["segment"][1] + 0.06), frame[2], frame[3]]
            rec = dict(id=rid, base=base_id, src=c["src"], defname=c["defname"], kind=c["kind"],
                       O=[float(v) for v in c["O"]], R=c["R"].tolist(), frame=frame,
                       wall_z=float(c["wall_z"]), glass=g, box=bx, hardware=c["hardware"],
                       lod2=bmeta, outside_ok=bool(outside_ok))
            if "d0_mode" in c:
                rec["d0_mode"] = c["d0_mode"]
            res.append(rec)
        if i % 100 == 0:
            log(i, "/", len(cands), "(%.1fs)" % (time.time() - t0))
    # duplicates (same glass detected twice, e.g. loose + block) -> keep the first
    keep_ = []
    for r in res:
        O1, R1 = np.array(r["O"]), np.array(r["R"])
        g1 = r["glass"]["rect"]
        dup = False
        for q in keep_:
            if q["base"] == r["base"]:
                continue          # sub-openings of one glass unit
            O2, R2 = np.array(q["O"]), np.array(q["R"])
            if R1[:, 2] @ R2[:, 2] < 0.95:
                continue
            g2 = q["glass"]["rect"]
            C = O2 + np.array([[g2[0], g2[2], q["glass"]["zg"]], [g2[1], g2[3], q["glass"]["zg"]]]) @ R2.T
            L = (C - O1) @ R1
            if abs(L[:, 2].mean() - r["glass"]["zg"]) > 0.25 or np.linalg.norm(O1 - O2) > 40:
                continue
            ox = min(g1[1], L[:, 0].max()) - max(g1[0], L[:, 0].min())
            oy = min(g1[3], L[:, 1].max()) - max(g1[2], L[:, 1].min())
            if ox > 0 and oy > 0 and ox * oy > 0.15 * min((g1[1] - g1[0]) * (g1[3] - g1[2]), (g2[1] - g2[0]) * (g2[3] - g2[2])):
                dup = True
                break
        if dup:
            skipped["duplicate"] = skipped.get("duplicate", 0) + 1
            continue
        keep_.append(r)
    log("box overlaps resolved by mitre planes:", resolve_overlaps(keep_))
    D = np.array([r["box"]["depth"] for r in keep_])
    log("openings", len(keep_), "skipped", skipped)
    log("depth: <0.1 %d, 0.1-0.3 %d, 0.3-0.6 %d, 0.6-1.0 %d, >1.0 %d" % (
        (D < 0.1).sum(), ((D >= 0.1) & (D < 0.3)).sum(), ((D >= 0.3) & (D < 0.6)).sum(), ((D >= 0.6) & (D < 1.0)).sum(),
        (D >= 1.0).sum()))
    jdump(dict(openings=keep_, skipped=skipped), os.path.join(CACHE, "openings.json"))
    log("done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

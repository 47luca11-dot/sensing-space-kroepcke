"""Extract D0 window rectangles from bldg_openings, match them to coplanar wall triangles,
decide cut vs flush, ground height, building clusters and style."""
import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from rtree import index as rindex

UP = np.array([0.0, 1.0, 0.0])


def frame_from_normal(n):
    n = n / np.linalg.norm(n)
    r = np.cross(UP, n)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(n, r)
    return r, u, n


class Win:
    pass


def extract_d0(openings_glb):
    node = [n for n in openings_glb["nodes"] if n["name"].endswith("Facade Glass D0")][0]
    p = node["prims"][0]
    T = p["pos"][p["idx"]]
    # cluster triangles into quads via shared (rounded) vertices
    V = T.reshape(-1, 3)
    key = np.round(V / 0.001).astype(np.int64)
    u, inv = np.unique(key, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    m = trimesh.Trimesh(u * 0.001, F, process=False)
    comps = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(F)), min_len=1)
    wins = []
    for ci, comp in enumerate(comps):
        tri = T[comp]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n = nrm.sum(0)
        if abs(n[1]) < 1e-3 * np.linalg.norm(n):
            n[1] = 0.0
        n /= np.linalg.norm(n)
        r, up, n = frame_from_normal(n)
        P = tri.reshape(-1, 3)
        c0 = P.mean(0)
        lx = (P - c0) @ r
        ly = (P - c0) @ up
        lz = (P - c0) @ n
        w = Win()
        w.id = len(wins)
        w.r, w.u, w.n = r, up, n
        x0, x1, y0, y1 = lx.min(), lx.max(), ly.min(), ly.max()
        w.width = x1 - x0
        w.height = y1 - y0
        w.origin = c0 + r * x0 + up * y0 + n * lz.mean()   # bottom-left corner on the D0 plane
        w.center = w.origin + r * w.width / 2 + up * w.height / 2
        w.planar_err = float(np.abs(lz - lz.mean()).max())
        w.area_ratio = float((np.linalg.norm(nrm, axis=1).sum() / 2) / (w.width * w.height))
        wins.append(w)
    return wins


def corners(w):
    o, r, u = w.origin, w.r, w.u
    return np.array([o, o + r * w.width, o + r * w.width + u * w.height, o + u * w.height])


def match_walls(wins, ws, plane_tol=0.02, log=print):
    """For each window: coplanar wall tris (same orientation) overlapping the rect, coverage."""
    idx = rindex.Index(properties=rindex.Property(dimension=3))
    lo = ws.T.min(1)
    hi = ws.T.max(1)
    for i in range(len(lo)):
        idx.insert(i, (*lo[i], *hi[i]))
    for w in wins:
        C = corners(w)
        bl = C.min(0) - 0.05
        bh = C.max(0) + 0.05
        cand = np.array(list(idx.intersection((*bl, *bh))), dtype=np.int64)
        w.tris = np.zeros(0, np.int64)
        w.coverage = 0.0
        w.wall_offset = np.nan
        if len(cand) == 0:
            continue
        par = ws.fn[cand] @ w.n > 0.9995
        d = np.einsum("ijk,k->ij", ws.T[cand] - w.origin, w.n)
        close = np.abs(d).max(1) < plane_tol
        cand = cand[par & close]
        if len(cand) == 0:
            continue
        rect = box(0, 0, w.width, w.height)
        polys = []
        keep = []
        for t in cand:
            P = ws.T[t] - w.origin
            q = Polygon(np.stack([P @ w.r, P @ w.u], 1))
            if not q.is_valid or q.area < 1e-9:
                continue
            inter = q.intersection(rect)
            if inter.area > 1e-7:
                keep.append(t)
                polys.append(inter)
        if not keep:
            continue
        w.tris = np.array(keep)
        w.coverage = unary_union(polys).area / rect.area
        dd = np.einsum("ijk,k->ij", ws.T[w.tris] - w.origin, w.n)
        w.wall_offset = float(np.average(dd.mean(1), weights=ws.area[w.tris]))
        nodes, cnt = np.unique(ws.NI[w.tris], return_counts=True)
        ar = np.array([sum(ws.area[t] for t in w.tris if ws.NI[t] == k) for k in nodes])
        w.wall_node = int(nodes[np.argmax(ar)])
    return wins


def ground_heights(wins, ground_ray):
    """Ray down from 0.8 m in front of the window bottom centre onto ground/street/base."""
    O = np.array([w.origin + w.r * w.width / 2 + w.n * 0.8 for w in wins])
    O[:, 1] += 30.0
    D = np.tile([0.0, -1.0, 0.0], (len(wins), 1))
    locs, ri, _ = ground_ray.intersects_location(O, D, multiple_hits=False)
    gy = np.full(len(wins), np.nan)
    gy[ri] = locs[:, 1]
    for w, g in zip(wins, gy):
        w.ground = float(g)
    return gy


def d0_orientation_votes(wins, ws, plane_tol=0.02, weight=25.0):
    """Extra orientation votes for wall tris from the (reliably outward) D0 quads."""
    idx = rindex.Index(properties=rindex.Property(dimension=3))
    lo = ws.T.min(1)
    hi = ws.T.max(1)
    for i in range(len(lo)):
        idx.insert(i, (*lo[i], *hi[i]))
    votes = np.zeros(len(ws.T))
    for w in wins:
        C = corners(w)
        cand = np.array(list(idx.intersection((*(C.min(0) - 0.05), *(C.max(0) + 0.05)))), dtype=np.int64)
        if len(cand) == 0:
            continue
        dot = ws.fn[cand] @ w.n
        d = np.einsum("ijk,k->ij", ws.T[cand] - w.origin, w.n)
        sel = (np.abs(dot) > 0.9995) & (np.abs(d).max(1) < plane_tol)
        for t, s in zip(cand[sel], np.sign(dot[sel])):
            P = ws.T[t] - w.origin
            q = Polygon(np.stack([P @ w.r, P @ w.u], 1))
            if not q.is_valid:
                continue
            a = q.intersection(box(0, 0, w.width, w.height)).area
            votes[t] += s * weight * a
    return votes


def wall_bottom(wins, ws, radius=5.0):
    """Lowest wall vertex near the window (fallback ground height)."""
    from scipy.spatial import cKDTree
    V = ws.T.reshape(-1, 3)
    tree = cKDTree(V[:, [0, 2]])
    out = []
    for w in wins:
        p = w.origin + w.r * w.width / 2
        ids = tree.query_ball_point([p[0], p[2]], radius)
        out.append(V[ids, 1].min() if ids else w.origin[1])
    return np.array(out)


def facade_bottom(wins, ws, plane_tol=0.03):
    """Lowest y of the coplanar facade directly below each window (walks down the union of the
    vertical intervals where the line x = window centre crosses coplanar wall triangles)."""
    idx = rindex.Index(properties=rindex.Property(dimension=3))
    lo = ws.T.min(1)
    hi = ws.T.max(1)
    for i in range(len(lo)):
        idx.insert(i, (*lo[i], *hi[i]))
    out = []
    for w in wins:
        o = w.origin + w.r * (w.width / 2)
        bl = np.minimum(o - 0.05, o - 0.05) ; bh = o + 0.05
        bl = bl.copy(); bh = bh.copy()
        bl[1] -= 60; bh[1] += w.height
        cand = np.array(list(idx.intersection((*bl, *bh))), dtype=np.int64)
        best = w.origin[1]
        if len(cand):
            dot = ws.fn[cand] @ w.n
            d = np.einsum("ijk,k->ij", ws.T[cand] - w.origin, w.n)
            cand = cand[(np.abs(dot) > 0.999) & (np.abs(d).max(1) < plane_tol)]
            iv = []
            for t in cand:
                P = ws.T[t] - o
                x = P @ w.r
                y = P @ w.u
                ys = []
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    if (x[a] - 0) * (x[b] - 0) <= 0 and abs(x[a] - x[b]) > 1e-9:
                        tt = (0 - x[a]) / (x[b] - x[a])
                        ys.append(y[a] + tt * (y[b] - y[a]))
                    elif abs(x[a]) < 1e-9 and abs(x[b]) < 1e-9:
                        ys += [y[a], y[b]]
                if ys:
                    iv.append((min(ys), max(ys)))
            iv.sort(key=lambda v: -v[1])
            cur = 0.0      # window bottom relative
            changed = True
            while changed:
                changed = False
                for a, b in iv:
                    if a < cur - 1e-4 and b >= cur - 0.02:
                        cur = a
                        changed = True
            best = w.origin[1] + cur
        out.append(best)
    return np.array(out)


def merge_touching(wins, gap=0.12, log=print):
    """Merge D0 rects that sit side by side in the same plane with (almost) no wall between them."""
    from scipy.spatial import cKDTree
    C = np.array([w.center for w in wins])
    parent = list(range(len(wins)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in cKDTree(C).query_pairs(6.0):
        wa, wb = wins[a], wins[b]
        if wa.n @ wb.n < 0.9995 or abs((wb.center - wa.center) @ wa.n) > 0.01:
            continue
        if abs(wa.origin[1] - wb.origin[1]) > 0.01 or abs(wa.height - wb.height) > 0.01:
            continue
        dx = abs((wb.center - wa.center) @ wa.r) - (wa.width + wb.width) / 2
        if -1e-3 < dx < gap:
            parent[find(b)] = find(a)
    groups = {}
    for i in range(len(wins)):
        groups.setdefault(find(i), []).append(i)
    out = []
    nm = 0
    for g in groups.values():
        if len(g) == 1:
            out.append(wins[g[0]])
            continue
        nm += len(g)
        w0 = wins[g[0]]
        xs = []
        for i in g:
            w = wins[i]
            x0 = (w.origin - w0.origin) @ w0.r
            xs += [x0, x0 + w.width]
        m = Win()
        m.r, m.u, m.n = w0.r, w0.u, w0.n
        m.width = max(xs) - min(xs)
        m.height = w0.height
        m.origin = w0.origin + w0.r * min(xs)
        m.center = m.origin + m.r * m.width / 2 + m.u * m.height / 2
        m.planar_err = 0.0
        m.area_ratio = 1.0
        m.merged = len(g)
        out.append(m)
    for i, w in enumerate(out):
        w.id = i
    log("merged %d touching D0 rects into %d openings" % (nm, sum(1 for g in groups.values() if len(g) > 1)))
    return out


def match_layer2(wins, ws, max_depth=0.6, margin=0.012):
    """Second wall shell(s) right behind the window (overlapping building shells, possibly slightly
    rotated, sometimes duplicated coplanar). Per window: list of non-overlapping layers of the nearest
    plane behind, each with its coverage of the rect expanded by margin."""
    idx = rindex.Index(properties=rindex.Property(dimension=3))
    lo = ws.T.min(1)
    hi = ws.T.max(1)
    for i in range(len(lo)):
        idx.insert(i, (*lo[i], *hi[i]))
    for w in wins:
        w.layer2 = None
        C = np.array([w.wall_origin, w.wall_origin + w.r * w.width, w.wall_origin + w.r * w.width + w.u * w.height,
                      w.wall_origin + w.u * w.height])
        C2 = np.concatenate([C, C - w.n * max_depth])
        cand = np.array(list(idx.intersection((*(C2.min(0) - 0.05), *(C2.max(0) + 0.05)))), dtype=np.int64)
        if len(cand) == 0:
            continue
        cand = cand[ws.fn[cand] @ w.n > 0.995]
        if len(cand) == 0:
            continue
        rect = box(-margin, -margin, w.width + margin, w.height + margin)
        info = []
        for t in cand:
            f = ws.fn[t]
            o = ws.T[t, 0] @ f
            dep = (C @ f - o) / (w.n @ f)
            if dep.min() < 0.03 or dep.max() > max_depth:
                continue
            P = ws.T[t] - w.wall_origin
            q = Polygon(np.stack([P @ w.r, P @ w.u], 1))
            if not q.is_valid or q.area < 1e-9:
                continue
            inter = q.intersection(rect)
            if inter.area <= 1e-7:
                continue
            info.append((t, f, dep, inter))
        if not info:
            continue
        # plane clusters (depth at the corners within 1.5 cm, parallel)
        planes = []
        for it in info:
            for pl in planes:
                if it[1] @ pl[0][1] > 0.9995 and np.abs(it[2] - pl[0][2]).max() < 0.015:
                    pl.append(it)
                    break
            else:
                planes.append([it])
        planes.sort(key=lambda pl: pl[0][2].mean())
        pl = planes[0]
        # split coincident duplicates into non-overlapping layers
        layers = []
        for it in sorted(pl, key=lambda x: -x[3].area):
            for L in layers:
                if L["union"].intersection(it[3]).area < 1e-5:
                    L["items"].append(it)
                    L["union"] = L["union"].union(it[3])
                    break
            else:
                layers.append(dict(items=[it], union=it[3]))
        deps = np.concatenate([[x[2] for x in L["items"]] for L in layers])
        w.layer2 = dict(layers=[np.array([x[0] for x in L["items"]]) for L in layers],
                        coverages=[float(L["union"].area / rect.area) for L in layers],
                        coverage=float(min(L["union"].area / rect.area for L in layers)),
                        depth=float(deps.max()), mean=float(deps.mean()), dmin=float(deps.min()), margin=margin,
                        n_planes_behind=len(planes))
    return wins

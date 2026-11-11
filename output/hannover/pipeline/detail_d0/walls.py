"""Wall soup utilities: load bldg_walls as a triangle soup, fix inverted orientation
(per connected component, parity propagation + up-ray-under-roof vote), coplanar lookups."""
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, breadth_first_order

import glbio


class WallSoup:
    def __init__(self, glb):
        self.nodes = glb["nodes"]
        T, N, U, NI = [], [], [], []
        for i, n in enumerate(self.nodes):
            p = n["prims"][0]
            T.append(p["pos"][p["idx"]])
            N.append(p["nrm"][p["idx"]] if p["nrm"] is not None else np.zeros((len(p["idx"]), 3, 3)))
            U.append(p["uv"][p["idx"]] if p["uv"] is not None else np.zeros((len(p["idx"]), 3, 2)))
            NI.append(np.full(len(p["idx"]), i))
        self.T = np.concatenate(T)          # (F,3,3)
        self.N = np.concatenate(N)          # per-corner normals
        self.U = np.concatenate(U)          # per-corner uv
        self.NI = np.concatenate(NI)        # node index per tri
        self.has_uv = [n["prims"][0]["uv"] is not None for n in self.nodes]
        self.recompute()

    def recompute(self):
        T = self.T
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        a = np.linalg.norm(n, axis=1)
        self.area = a / 2
        self.fn = n / np.maximum(a, 1e-15)[:, None]
        self.c = T.mean(1)

    def flip(self, mask):
        self.T[mask] = self.T[mask][:, [0, 2, 1]]
        self.N[mask] = -self.N[mask][:, [0, 2, 1]]
        self.U[mask] = self.U[mask][:, [0, 2, 1]]
        self.recompute()


def roof_intersector(roof_glb, extra=()):
    V, F, o = [], [], 0
    for g in [roof_glb] + list(extra):
        for n in g["nodes"]:
            for p in n["prims"]:
                V.append(p["pos"]); F.append(p["idx"] + o); o += len(p["pos"])
    m = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=False)
    return m.ray


def orientation_votes(ws, ray, offsets=(0.6, 1.2)):
    """+area: normal points outside (point in front not covered by roof, point behind covered).
    -area: inverted. 0: ambiguous / not vertical."""
    vert = np.abs(ws.fn[:, 1]) < 0.5
    up = np.tile([0.0, 1.0, 0.0], (len(ws.c), 1))
    score = np.zeros(len(ws.c))
    for d in offsets:
        po = ws.c + ws.fn * d
        pi = ws.c - ws.fn * d
        ho = ray.intersects_any(po, up)
        hi = ray.intersects_any(pi, up)
        s = np.where(~ho & hi, 1.0, np.where(ho & ~hi, -1.0, 0.0))
        score += s
    score[~vert] = 0
    return score * ws.area


def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def escape_votes(ws, occl_ray, sel, off=0.25, ndir=40, far=250.0):
    """For tris sel: fraction of hemisphere rays escaping from c+n*off minus from c-n*off.
    Positive -> normal points to the open side (outside)."""
    D = fib_dirs(ndir)
    c = ws.c[sel]
    n = ws.fn[sel]
    out = np.zeros(len(sel))
    for sgn, acc in ((1.0, 1.0), (-1.0, -1.0)):
        P = c + sgn * n * off
        O, Dr, owner = [], [], []
        for k in range(len(sel)):
            m = (D @ (sgn * n[k])) > 0.15
            O.append(np.repeat(P[k:k + 1], m.sum(), 0)); Dr.append(D[m]); owner.append(np.full(m.sum(), k))
        O = np.concatenate(O); Dr = np.concatenate(Dr); owner = np.concatenate(owner)
        hit = occl_ray.intersects_any(O, Dr)
        tot = np.bincount(owner, minlength=len(sel))
        esc = np.bincount(owner, weights=~hit, minlength=len(sel))
        out += acc * esc / np.maximum(tot, 1)
    return out


def viewpoints_from_ground(ground_glbs, n=4000, seed=0, extra_heights=(1.7,)):
    """sample public-space viewpoints on upward-facing ground/street faces + an aerial grid."""
    T = []
    for g in ground_glbs:
        for nd in g["nodes"]:
            for p in nd["prims"]:
                T.append(p["pos"][p["idx"]])
    T = np.concatenate(T)
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    a = np.linalg.norm(nrm, axis=1)
    up = (nrm[:, 1] / np.maximum(a, 1e-12)) > 0.7
    T, a = T[up], a[up]
    rng = np.random.default_rng(seed)
    k = rng.choice(len(T), n, p=a / a.sum())
    r1 = np.sqrt(rng.random(n))[:, None]
    r2 = rng.random(n)[:, None]
    P = (1 - r1) * T[k, 0] + r1 * (1 - r2) * T[k, 1] + r1 * r2 * T[k, 2]
    V = [P + np.array([0, h, 0]) for h in extra_heights]
    lo, hi = P.min(0), P.max(0)
    gx, gz = np.meshgrid(np.arange(lo[0] - 60, hi[0] + 60, 30.0), np.arange(lo[2] - 60, hi[2] + 60, 30.0))
    A = np.stack([gx.ravel(), np.full(gx.size, 70.0), gz.ravel()], 1)
    return np.concatenate(V + [A])


def visibility_votes(ws, occl_mesh, sel, views, k=48, radius=220.0, seed=1):
    """+1 per viewpoint that sees the front of the face, -1 per viewpoint that sees its back."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    tree = cKDTree(views[:, [0, 2]])
    O, D, own, dist = [], [], [], []
    for j, t in enumerate(sel):
        c = ws.c[t]
        ids = tree.query_ball_point([c[0], c[2]], radius)
        if not ids:
            continue
        ids = np.array(ids)
        if len(ids) > k:
            ids = rng.choice(ids, k, replace=False)
        d = views[ids] - c
        L = np.linalg.norm(d, axis=1)
        d = d / L[:, None]
        O.append(c + d * 0.02); D.append(d); own.append(np.full(len(ids), j)); dist.append(L - 0.02)
    O = np.concatenate(O); D = np.concatenate(D); own = np.concatenate(own); dist = np.concatenate(dist)
    _, ri, locs = occl_mesh.ray.intersects_id(O, D, multiple_hits=False, return_locations=True)
    hd = np.full(len(O), np.inf)
    hd[ri] = np.linalg.norm(locs - O[ri], axis=1)
    vis = hd >= dist
    side = np.sign(np.einsum("ij,ij->i", D, ws.fn[sel][own]))
    return np.bincount(own, weights=vis * side, minlength=len(sel)), np.bincount(own, weights=vis, minlength=len(sel))


def fix_orientation(ws, ray, extra_votes=None, log=print, occl_ray=None, occl_mesh=None, views=None):
    """Make winding consistent within connected components (parity BFS), then orient every
    component so that the area-weighted vote says 'outward'. Returns flip mask applied."""
    T = ws.T
    V = T.reshape(-1, 3)
    key = np.round(V / 0.002).astype(np.int64)
    u, inv = np.unique(key, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    nF = len(F)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(nF), 3)
    und = np.sort(E, axis=1)
    ek = und[:, 0] * (len(u) + 1) + und[:, 1]
    order = np.argsort(ek, kind="stable")
    eks = ek[order]
    _, start, cnt = np.unique(eks, return_index=True, return_counts=True)
    m2 = cnt == 2
    i1 = order[start[m2]]
    i2 = order[start[m2] + 1]
    f1, f2 = fid[i1], fid[i2]
    ok = f1 != f2
    f1, f2, i1, i2 = f1[ok], f2[ok], i1[ok], i2[ok]
    # also require near-continuation (not folded back-to-back): dihedral normal dot > -0.9
    cont = np.einsum("ij,ij->i", ws.fn[f1], ws.fn[f2]) > -0.9
    f1, f2, i1, i2 = f1[cont], f2[cont], i1[cont], i2[cont]
    incons = (E[i1, 0] == E[i2, 0]).astype(np.int8)
    A = coo_matrix((np.ones(len(f1)), (f1, f2)), shape=(nF, nF)).tocsr()
    A = A + A.T
    ncomp, lab = connected_components(A, directed=False)
    # parity BFS
    par = np.full(nF, -1, np.int8)
    adj_inc = {}
    # build adjacency lists with inconsistency flags
    rows = np.concatenate([f1, f2]); cols = np.concatenate([f2, f1]); fl = np.concatenate([incons, incons])
    o2 = np.argsort(rows, kind="stable")
    rows, cols, fl = rows[o2], cols[o2], fl[o2]
    ptr = np.searchsorted(rows, np.arange(nF + 1))
    conflicts = 0
    for s in range(nF):
        if par[s] >= 0:
            continue
        par[s] = 0
        stack = [s]
        while stack:
            f = stack.pop()
            for k in range(ptr[f], ptr[f + 1]):
                g = cols[k]
                want = par[f] ^ fl[k]
                if par[g] < 0:
                    par[g] = want
                    stack.append(g)
                elif par[g] != want:
                    conflicts += 1
    # votes with parity applied
    flip0 = par.astype(bool)
    sgn0 = np.where(flip0, -1.0, 1.0)
    comp_area = np.bincount(lab, weights=ws.area, minlength=ncomp)
    fl_area = np.bincount(lab, weights=ws.area * flip0, minlength=ncomp)
    # (a) roof-cover vote: point in front not under a roof, point behind under a roof
    v_up = np.bincount(lab, weights=orientation_votes(ws, ray) * sgn0, minlength=ncomp)
    # (b) visibility from public space (street points + aerial grid): who sees which side
    v_esc = np.zeros(ncomp)
    if occl_mesh is not None and views is not None:
        sel = []
        for k in range(ncomp):
            m = np.where(lab == k)[0]
            if len(m) > 60:
                m = m[np.argsort(-ws.area[m])[:60]]
            sel.append(m)
        sel = np.concatenate(sel)
        vv, nvis = visibility_votes(ws, occl_mesh, sel, views)
        frac = vv / np.maximum(nvis, 1)
        e = np.where((nvis >= 2) & (np.abs(frac) > 0.3), np.sign(frac), 0.0) * ws.area[sel] * sgn0[sel]
        v_esc = np.bincount(lab[sel], weights=e, minlength=ncomp)
        ws.vis = (sel, vv, nvis)
    # (c) D0 votes (reliable where present)
    v_d0 = np.zeros(ncomp)
    if extra_votes is not None:
        v_d0 = np.bincount(lab, weights=extra_votes * sgn0, minlength=ncomp)
    thr = 0.02 * comp_area + 0.05
    comp_flip = fl_area > 0.5 * comp_area                      # default: keep the majority winding
    up_says = np.where(np.abs(v_up) > thr, np.sign(v_up), 0)
    vis_says = np.where(np.abs(v_esc) > thr, np.sign(v_esc), 0)
    # visibility from public space is the primary signal, the roof test only breaks ties
    decided_vis = vis_says != 0
    comp_flip = np.where(decided_vis, vis_says < 0, np.where(up_says != 0, up_says < 0, comp_flip))
    d0 = np.abs(v_d0) > 0.5
    comp_flip = np.where(d0, v_d0 < 0, comp_flip)
    self_check = d0 & decided_vis
    log("orientation votes: %d comps by D0 (vis agrees on %d/%d), %d by visibility, %d by roof test, %d kept" % (
        d0.sum(), ((vis_says < 0) == (v_d0 < 0))[self_check].sum(), self_check.sum(),
        (decided_vis & ~d0).sum(), (~decided_vis & ~d0 & (up_says != 0)).sum(), (~decided_vis & ~d0 & (up_says == 0)).sum()))
    final = flip0 ^ comp_flip[lab]
    log("orientation: %d comps, %d inconsistent edges, %d parity conflicts, flipping %d/%d tris (%.0f m2)" % (
        ncomp, incons.sum(), conflicts // 2, final.sum(), nF, ws.area[final].sum()))
    ws.flip(final)
    return final

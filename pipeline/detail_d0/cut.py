"""Cut rectangular openings into coplanar wall triangles (constrained triangulation, exact
re-use of the original triangle planes) and report the hole boundary vertices per side."""
import numpy as np
import triangle as tr

import extract


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def _bary(P, A, B, C):
    """barycentric coords of 2D points P (k,2) in triangle ABC."""
    v0, v1 = B - A, C - A
    v2 = P - A
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], -1)


def cut(ws, wins, log=print):
    """wins: windows to cut (have .tris, .r, .u, .n, .wall_origin, .width, .height).
    Returns (removed_tri_mask, new_tris dict) where new_tris has arrays T (k,3,3), N (k,3,3), U (k,3,2), NI (k,)
    and sets w.hole = {'bottom': [...3D...], 'right': [...], 'top': [...], 'left': [...]} (ordered CCW from outside),
    or w.cut_ok = False when the triangulation failed."""
    uf = UF(len(wins))
    owner = {}
    for i, w in enumerate(wins):
        for t in w.tris:
            if t in owner:
                uf.union(owner[t], i)
            else:
                owner[t] = i
    groups = {}
    for i in range(len(wins)):
        groups.setdefault(uf.find(i), []).append(i)
    removed = np.zeros(len(ws.T), bool)
    outT, outN, outU, outNI = [], [], [], []
    nfail = 0
    for gi, members in groups.items():
        G = [wins[i] for i in members]
        S = np.unique(np.concatenate([w.tris for w in G]))
        res = _cut_group(ws, G, S)
        if res is None:
            nfail += 1
            for w in G:
                w.cut_ok = False
            continue
        T, N, U, NI = res
        removed[S] = True
        outT.append(T); outN.append(N); outU.append(U); outNI.append(NI)
        for w in G:
            w.cut_ok = True
    log("cut: %d groups, %d failed" % (len(groups), nfail))
    if outT:
        new = dict(T=np.concatenate(outT), N=np.concatenate(outN), U=np.concatenate(outU), NI=np.concatenate(outNI))
    else:
        new = dict(T=np.zeros((0, 3, 3)), N=np.zeros((0, 3, 3)), U=np.zeros((0, 3, 2)), NI=np.zeros(0, int))
    return removed, new


def _cut_group(ws, G, S):
    w0 = G[0]
    n = np.average(ws.fn[S], axis=0, weights=ws.area[S])
    n /= np.linalg.norm(n)
    r, u, n = extract.frame_from_normal(n)
    O = ws.T[S[0], 0].copy()
    to2 = lambda P: np.stack([(P - O) @ r, (P - O) @ u], -1)
    T2 = to2(ws.T[S])                      # (s,3,2)
    # vertices / segments of the region
    pts = T2.reshape(-1, 2)
    key = np.round(pts / 1e-7).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    V = uk * 1e-7
    tv = inv.reshape(-1, 3)
    E = np.concatenate([tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]])
    E = np.unique(np.sort(E, 1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    Vl = [V]
    segs = [E]
    holes = []
    rects = []
    nv = len(V)
    for w in G:
        C = np.array([w.wall_origin, w.wall_origin + w.r * w.width,
                      w.wall_origin + w.r * w.width + w.u * w.height, w.wall_origin + w.u * w.height])
        c2 = to2(C)
        # snap rect corners onto existing vertices if extremely close
        for k in range(4):
            d = np.linalg.norm(V - c2[k], axis=1)
            j = np.argmin(d)
            if d[j] < 2e-4:
                c2[k] = V[j]
        rects.append(c2)
        Vl.append(c2)
        segs.append(np.array([[nv, nv + 1], [nv + 1, nv + 2], [nv + 2, nv + 3], [nv + 3, nv]]))
        holes.append(c2.mean(0))
        nv += 4
    A = dict(vertices=np.concatenate(Vl), segments=np.concatenate(segs), holes=np.array(holes))
    try:
        B = tr.triangulate(A, "p")
    except Exception as e:
        print("triangle failed", e)
        return None
    BV = B["vertices"]
    BT = B["triangles"]
    if len(BT) == 0:
        return None
    cen = BV[BT].mean(1)
    # parent triangle for each output triangle (point in triangle)
    parent = np.full(len(BT), -1)
    for si in range(len(S)):
        a, b, c = T2[si]
        lo = np.minimum(np.minimum(a, b), c) - 1e-6
        hi = np.maximum(np.maximum(a, b), c) + 1e-6
        cand = np.where((parent < 0) & np.all(cen >= lo, 1) & np.all(cen <= hi, 1))[0]
        if len(cand) == 0:
            continue
        bc = _bary(cen[cand], a, b, c)
        ok = np.all(bc > -1e-7, 1)
        parent[cand[ok]] = si
    # drop triangles inside rects (should already be holes) or outside region
    inside_rect = np.zeros(len(BT), bool)
    for c2 in rects:
        x0, y0 = c2.min(0)
        x1, y1 = c2.max(0)
        # rect may be slightly rotated in this frame: use its own axes
        e1 = c2[1] - c2[0]; e2 = c2[3] - c2[0]
        L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
        p = cen - c2[0]
        a1 = p @ e1 / L1
        a2 = p @ e2 / L2
        inside_rect |= (a1 > 1e-6) & (a1 < L1 - 1e-6) & (a2 > 1e-6) & (a2 < L2 - 1e-6)
    keep = (parent >= 0) & ~inside_rect
    BT = BT[keep]
    parent = parent[keep]
    # area check
    a_out = 0.5 * np.abs(np.cross(BV[BT[:, 1]] - BV[BT[:, 0]], BV[BT[:, 2]] - BV[BT[:, 0]])).sum()
    a_in = ws.area[S].sum()
    a_holes = sum(w.coverage * w.width * w.height for w in G)
    if abs(a_in - a_holes - a_out) > 1e-3 * max(1.0, a_in):
        print("area mismatch group", w0.id, a_in, a_holes, a_out)
        return None
    # 3D positions via parent barycentrics
    T = np.zeros((len(BT), 3, 3))
    N = np.zeros((len(BT), 3, 3))
    U = np.zeros((len(BT), 3, 2))
    for k in range(3):
        P = BV[BT[:, k]]
        for si in np.unique(parent):
            m = parent == si
            a, b, c = T2[si]
            bc = _bary(P[m], a, b, c)
            t = S[si]
            T[m, k] = bc @ ws.T[t]
            N[m, k] = bc @ ws.N[t]
            U[m, k] = bc @ ws.U[t]
    NI = ws.NI[S[parent]]
    # orientation: keep parent's facing
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = np.einsum("ij,ij->i", fn, ws.fn[S[parent]]) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    N[flip] = N[flip][:, [0, 2, 1]]
    U[flip] = U[flip][:, [0, 2, 1]]
    # hole boundaries: output vertices on rect sides, 3D from parents
    v3 = {}
    for k in range(3):
        P = BV[BT[:, k]]
        for ti in range(len(BT)):
            vi = BT[ti, k]
            if vi in v3:
                continue
            si = parent[ti]
            a, b, c = T2[si]
            bc = _bary(P[ti:ti + 1], a, b, c)[0]
            v3[vi] = bc @ ws.T[S[si]]
    used = np.array(sorted(v3.keys()))
    for w, c2 in zip(G, rects):
        sides = {}
        names = ["bottom", "right", "top", "left"]
        for s in range(4):
            a, b = c2[s], c2[(s + 1) % 4]
            d = b - a
            L = np.linalg.norm(d)
            dn = d / L
            p = BV[used] - a
            t = p @ dn
            off = np.abs(p[:, 0] * dn[1] - p[:, 1] * dn[0])
            m = (off < 1e-5) & (t > -1e-6) & (t < L + 1e-6)
            ids = used[m][np.argsort(t[m])]
            sides[names[s]] = np.array([v3[i] for i in ids])
        w.hole = sides
    return T, N, U, NI

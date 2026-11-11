"""Kleine Geometrie-Bibliothek (numpy) fuer das Landmark Leibnizhaus.
Weltrahmen = Spielrahmen (x Ost, y oben, z Sued), glTF-Front = CCW von aussen gesehen.
Alle Primitive werden in lokalen Koordinaten (u rechts, v oben, w nach aussen) gebaut und ueber Frame in die Welt gelegt.
MB = Mesh-Sammler je Material: (P, N, UV, I)."""
import math
from collections import defaultdict
import numpy as np
import mapbox_earcut as earcut

TAU = 2 * math.pi


def nrm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1, n)


class Frame:
    """o + a*u + b*v + c*w; (u, v, w) rechtshaendig orthonormal."""

    def __init__(self, o, u, v=(0, 1, 0), w=None):
        self.o = np.asarray(o, float)
        u = nrm(u); v = np.asarray(v, float)
        v = nrm(v - u * np.dot(u, v))
        w = np.cross(u, v) if w is None else nrm(w)
        self.u, self.v, self.w = u, v, w
        self.M = np.c_[u, v, w]          # Spalten

    def p(self, L):
        L = np.asarray(L, float)
        return self.o + L @ self.M.T

    def n(self, L):
        return np.asarray(L, float) @ self.M.T

    def sub(self, a=0.0, b=0.0, c=0.0, u=None, v=None):
        """Unterrahmen: verschoben (lokal a,b,c), optional um v gedreht (u neu in lokaler uw-Ebene)."""
        o = self.p([a, b, c])
        if u is None: return Frame(o, self.u, self.v, self.w)
        uu = self.n(u); vv = self.n(v) if v is not None else self.v
        return Frame(o, uu, vv)

    def rot(self, ang, a=0.0, b=0.0, c=0.0):
        """Rahmen um lokale v-Achse gedreht (ang rad, positiv: u -> -w Richtung, d.h. Blick von oben gegen Uhrzeiger)."""
        ca, sa = math.cos(ang), math.sin(ang)
        return Frame(self.p([a, b, c]), self.n([ca, 0, -sa]), self.v)

    def mat4(self):
        M = np.eye(4); M[:3, :3] = self.M; M[:3, 3] = self.o
        return M


WORLD = Frame((0, 0, 0), (1, 0, 0), (0, 1, 0))


class MB:
    def __init__(self):
        self.parts = defaultdict(list)

    def add(self, mat, P, N, UV, I):
        P = np.asarray(P, np.float64); I = np.asarray(I, np.int64).reshape(-1, 3)
        if len(I) == 0: return
        self.parts[mat].append((P, np.asarray(N, np.float64), np.asarray(UV, np.float64), I))

    def add_local(self, fr, mat, P, N, UV, I):
        self.add(mat, fr.p(P), nrm(fr.n(N)), UV, I)

    def extend(self, other, fr=None):
        for m, L in other.parts.items():
            for (P, N, UV, I) in L:
                if fr is None: self.parts[m].append((P, N, UV, I))
                else: self.parts[m].append((fr.p(P), nrm(fr.n(N)), UV, I))

    def transformed(self, M4):
        o = MB(); A = M4[:3, :3]; t = M4[:3, 3]
        Ninv = np.linalg.inv(A).T
        for m, L in self.parts.items():
            for (P, N, UV, I) in L:
                Q = P @ A.T + t; NN = nrm(N @ Ninv.T)
                if np.linalg.det(A) < 0: I = I[:, ::-1]
                o.parts[m].append((Q, NN, UV, I))
        return o

    def arrays(self):
        out = {}
        for m, L in self.parts.items():
            P = []; N = []; U = []; I = []; off = 0
            for (p, n, uv, i) in L:
                P.append(p); N.append(n); U.append(uv); I.append(i + off); off += len(p)
            out[m] = (np.concatenate(P), np.concatenate(N), np.concatenate(U), np.concatenate(I))
        return out

    def ntris(self):
        return int(sum(len(i) for L in self.parts.values() for (_, _, _, i) in L))

    def bounds(self):
        P = np.concatenate([p for L in self.parts.values() for (p, _, _, _) in L])
        return P.min(0), P.max(0)

    def rename(self, src, dst):
        if src in self.parts:
            self.parts[dst].extend(self.parts.pop(src))


# ------------------------------------------------------------------ Grundbausteine (lokal)
def flat_tris(P, I):
    """Dreiecke -> flache Normalen (Vertices je Dreieck dupliziert)."""
    P = np.asarray(P, float); I = np.asarray(I).reshape(-1, 3)
    T = P[I]                                     # (n,3,3)
    fn = nrm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]))
    PP = T.reshape(-1, 3); NN = np.repeat(fn, 3, axis=0)
    II = np.arange(len(PP)).reshape(-1, 3)
    return PP, NN, II


def box_uv(P, N, s=1.0):
    """Box-Projektion in lokalen Koordinaten (Meter * s)."""
    P = np.asarray(P); N = np.asarray(N)
    a = np.abs(N)
    uv = np.where((a[:, 1] >= a[:, 0]) & (a[:, 1] >= a[:, 2]), 0, np.where(a[:, 0] >= a[:, 2], 1, 2))
    U = np.empty((len(P), 2))
    m0 = uv == 0; m1 = uv == 1; m2 = uv == 2
    U[m0] = np.c_[P[m0, 0], P[m0, 2]]
    U[m1] = np.c_[P[m1, 2] * np.sign(N[m1, 0] + 1e-9), P[m1, 1]]
    U[m2] = np.c_[P[m2, 0] * np.sign(N[m2, 2] + 1e-9), P[m2, 1]]
    return U * s


def orient(P, I, want):
    """Dreiecke so drehen, dass die Flaechennormale mit want (n,3 oder 3) uebereinstimmt."""
    I = np.asarray(I).reshape(-1, 3).copy()
    T = P[I]
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    w = np.broadcast_to(np.asarray(want, float), fn.shape)
    bad = (fn * w).sum(1) < 0
    I[bad] = I[bad][:, ::-1]
    return I


def quads_to_tris(Q):
    """Q (n,4) Indexquads -> (2n,3)"""
    Q = np.asarray(Q)
    return np.r_[Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]] if len(Q) else np.zeros((0, 3), int)


def box(mb, fr, mat, x0, x1, y0, y1, z0, z1, bevel=0.0, uvs=1.0, skip=()):
    """Quader in lokalen Koordinaten, optional mit Fase (bevel) an allen Kanten. skip: Seiten 'b','t','l','r','k','f' weglassen
    (nur ohne Fase)."""
    x0, x1 = min(x0, x1), max(x0, x1); y0, y1 = min(y0, y1), max(y0, y1); z0, z1 = min(z0, z1), max(z0, z1)
    b = min(bevel, (x1 - x0) * 0.45, (y1 - y0) * 0.45, (z1 - z0) * 0.45)
    if b <= 1e-4:
        C = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
        F = {'k': [0, 3, 2, 1], 'f': [4, 5, 6, 7], 'b': [0, 1, 5, 4], 't': [3, 7, 6, 2], 'l': [0, 4, 7, 3], 'r': [1, 2, 6, 5]}
        Q = [F[k] for k in F if k not in skip]
        if not Q: return
        P, N, I = flat_tris(C, quads_to_tris(np.array(Q)))
    else:
        pts = []
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    X = (x0, x1)[sx]; Y = (y0, y1)[sy]; Z = (z0, z1)[sz]
                    dx = b if sx == 0 else -b; dy = b if sy == 0 else -b; dz = b if sz == 0 else -b
                    pts += [[X + dx, Y + dy, Z], [X + dx, Y, Z + dz], [X, Y + dy, Z + dz]]
        pts = np.array(pts)
        from scipy.spatial import ConvexHull
        h = ConvexHull(pts)
        I = h.simplices
        c = pts.mean(0)
        I = orient(pts, I, pts[I].mean(1) - c)
        P, N, I = flat_tris(pts, I)
        # nach Hull: nahezu koplanare Dreiecke koennen Normalenrauschen haben -> ok (flach)
    mb.add_local(fr, mat, P, N, box_uv(P, N, uvs), I)


def extrude(mb, fr, mat, poly, d0, d1, holes=(), uvs=1.0, caps=(True, True), smooth=False, side_mat=None):
    """2D-Polygon (lokal x,y; CCW gesehen von +z) von z=d0 bis z=d1 extrudieren (Prisma); Loecher moeglich."""
    poly = np.asarray(poly, float)
    rings = [poly] + [np.asarray(h, float) for h in holes]
    # Ringe orientieren: aussen CCW, Loecher CW
    def area(r): return 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
    rr = []
    for k, r in enumerate(rings):
        a = area(r)
        if (k == 0 and a < 0) or (k > 0 and a > 0): r = r[::-1]
        rr.append(r)
    sm = side_mat or mat
    for r in rr:
        n = len(r)
        P = []; N = []; U = []; I = []
        if smooth:
            # glatte Seitenwand (Kreise usw.)
            e = np.roll(r, -1, 0) - r
            en = nrm(np.c_[e[:, 1], -e[:, 0]])
            vn = nrm(en + np.roll(en, 1, 0))
            s = np.r_[0, np.cumsum(np.linalg.norm(e, axis=1))]
            rc = np.r_[r, r[:1]]; vc = np.r_[vn, vn[:1]]
            P = np.r_[np.c_[rc, np.full(n + 1, d0)], np.c_[rc, np.full(n + 1, d1)]]
            N = np.r_[np.c_[vc, np.zeros(n + 1)], np.c_[vc, np.zeros(n + 1)]]
            U = np.r_[np.c_[s, np.full(n + 1, d0)], np.c_[s, np.full(n + 1, d1)]] * uvs
            k = np.arange(n)
            Q = np.c_[k, k + 1, k + 1 + n + 1, k + n + 1]
            I = quads_to_tris(Q)
            I = orient(P, I, N[I[:, 0]] + N[I[:, 1]] + N[I[:, 2]])
            mb.add_local(fr, sm, P, N, U, I)
        else:
            a = r; b = np.roll(r, -1, 0)
            Q = np.stack([np.c_[a, np.full(n, d0)], np.c_[b, np.full(n, d0)], np.c_[b, np.full(n, d1)], np.c_[a, np.full(n, d1)]], 1)
            e = b - a; en = np.c_[e[:, 1], -e[:, 0], np.zeros(n)]
            PP = Q.reshape(-1, 3)
            II = quads_to_tris(np.arange(4 * n).reshape(-1, 4))
            II = orient(PP, II, np.repeat(en, 2, axis=0) if False else en[np.r_[np.arange(n), np.arange(n)]])
            PP, NN, II = flat_tris(PP, II)
            L = np.linalg.norm(e, axis=1)
            s0 = np.r_[0, np.cumsum(L)[:-1]]
            # UV: Abwicklung entlang Umfang
            UU = box_uv(PP, NN, uvs)
            mb.add_local(fr, sm, PP, NN, UU, II)
    if caps[0] or caps[1]:
        verts = np.concatenate(rr)
        ends = np.cumsum([len(r) for r in rr]).astype(np.uint32)
        tri = np.asarray(earcut.triangulate_float64(verts, ends), np.int64).reshape(-1, 3)
        for k, (d, sgn) in enumerate(((d0, -1.0), (d1, 1.0))):
            if not caps[k]: continue
            P = np.c_[verts, np.full(len(verts), d)]
            I = orient(P, tri, [0, 0, sgn * (1 if d1 > d0 else -1)])
            N = np.tile([0, 0, sgn * (1 if d1 > d0 else -1)], (len(P), 1))
            mb.add_local(fr, mat, P, N, verts * uvs, I)


def face_with_holes(mb, fr, mat, outer, holes=(), z=0.0, uvs=1.0, uv_off=(0, 0)):
    """ebene Flaeche (lokal x,y bei z) mit Loechern, Normale +z."""
    rings = [np.asarray(outer, float)] + [np.asarray(h, float) for h in holes]
    verts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = np.asarray(earcut.triangulate_float64(verts, ends), np.int64).reshape(-1, 3)
    P = np.c_[verts, np.full(len(verts), z)]
    I = orient(P, tri, [0, 0, 1])
    N = np.tile([0, 0, 1.0], (len(P), 1))
    mb.add_local(fr, mat, P, N, (verts + uv_off) * uvs, I)


def reveal(mb, fr, mat, ring, z0, z1, uvs=1.0):
    """Laibung eines Lochs (ring lokal x,y) von z0 (Wandflaeche) nach z1 (tiefer, z1<z0); Normalen zur Lochmitte."""
    r = np.asarray(ring, float); n = len(r)
    a = r; b = np.roll(r, -1, 0)
    c = r.mean(0)
    Q = np.stack([np.c_[a, np.full(n, z1)], np.c_[b, np.full(n, z1)], np.c_[b, np.full(n, z0)], np.c_[a, np.full(n, z0)]], 1)
    PP = Q.reshape(-1, 3)
    II = quads_to_tris(np.arange(4 * n).reshape(-1, 4))
    m = (a + b) / 2 - c
    e = b - a
    en = np.c_[e[:, 1], -e[:, 0]]
    en = en * np.sign((en * -m).sum(1, keepdims=True) + 1e-12)   # zur Mitte zeigend
    want = np.c_[en, np.zeros(n)]
    II = orient(PP, II, want[np.r_[np.arange(n), np.arange(n)]])
    PP, NN, II = flat_tris(PP, II)
    mb.add_local(fr, mat, PP, NN, box_uv(PP, NN, uvs), II)


def sweep(mb, fr, mat, prof, path, up=(0, 1, 0), closed=False, caps=True, smooth=False, uvs=1.0, miter=True):
    """Profil (m,2) [(aussen, oben)] entlang Pfad (n,3, lokal) ziehen. Profil von unten-Wand ueber aussen nach oben-Wand
    (Koerper links der Laufrichtung). Seitenrichtung s = t x up."""
    prof = np.asarray(prof, float); path = np.asarray(path, float); up = nrm(np.asarray(up, float))
    n = len(path); m = len(prof)
    if closed:
        d = np.roll(path, -1, 0) - path
        dprev = np.roll(d, 1, 0)
    else:
        d = np.r_[path[1:] - path[:-1], (path[-1] - path[-2])[None]]
        dprev = np.r_[(path[1] - path[0])[None], path[1:] - path[:-1]]
    tn = nrm(d); tp = nrm(dprev)
    t = nrm(tn + tp)
    s = nrm(np.cross(t, up))
    sn = nrm(np.cross(tn, up))
    sc = 1.0 / np.clip((s * sn).sum(1), 0.3, 1.0) if miter else np.ones(n)
    upv = np.tile(up, (n, 1))
    # Punkte
    P = path[:, None, :] + prof[None, :, 0:1] * (s * sc[:, None])[:, None, :] + prof[None, :, 1:2] * upv[:, None, :]
    # Profil-Normalen 2D
    dp = np.roll(prof, -1, 0) - prof
    pn2 = nrm(np.c_[dp[:, 1], -dp[:, 0]])          # (m,2) je Segment k->k+1
    segs = m - 1                                   # offenes Profil
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if closed: L = np.r_[L, L[-1] + np.linalg.norm(path[0] - path[-1])]
    pl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
    nseg = n if closed else n - 1
    outP = []; outN = []; outU = []; outI = []; off = 0
    for i in range(nseg):
        j = (i + 1) % n
        for k in range(segs):
            q = np.array([P[i, k], P[j, k], P[j, k + 1], P[i, k + 1]])
            sv = sn[i]
            nn = nrm(pn2[k, 0] * sv + pn2[k, 1] * up)
            if smooth:
                kp = max(k - 1, 0); kn = min(k + 1, segs - 1)
                n0 = nrm(pn2[kp] + pn2[k]); n1 = nrm(pn2[k] + pn2[kn])
                N4 = np.array([n0[0] * sn[i] + n0[1] * up, n0[0] * sn[i] + n0[1] * up, n1[0] * sn[i] + n1[1] * up, n1[0] * sn[i] + n1[1] * up])
                N4 = nrm(N4)
            else:
                N4 = np.tile(nn, (4, 1))
            uv = np.array([[L[i], pl[k]], [L[i + 1], pl[k]], [L[i + 1], pl[k + 1]], [L[i], pl[k + 1]]]) * uvs
            I = np.array([[0, 1, 2], [0, 2, 3]])
            I = orient(q, I, nn)
            outP.append(q); outN.append(N4); outU.append(uv); outI.append(I + off); off += 4
    if outP:
        mb.add_local(fr, mat, np.concatenate(outP), np.concatenate(outN), np.concatenate(outU), np.concatenate(outI))
    if caps and not closed:
        poly = prof
        ends = np.array([len(poly)], np.uint32)
        try:
            tri = np.asarray(earcut.triangulate_float64(poly, ends), np.int64).reshape(-1, 3)
        except Exception:
            tri = np.zeros((0, 3), int)
        if len(tri):
            for idx, sgn in ((0, -1), (n - 1, 1)):
                Pc = path[idx] + poly[:, 0:1] * (s[idx] * sc[idx]) + poly[:, 1:2] * up
                want = sgn * t[idx]
                I = orient(Pc, tri, want)
                mb.add_local(fr, mat, Pc, np.tile(want, (len(Pc), 1)), poly * uvs, I)


def lathe(mb, fr, mat, prof, segs=16, a0=0.0, a1=TAU, smooth=True, uvs=1.0, c=(0, 0, 0)):
    """Rotationskoerper um lokale y-Achse durch c. prof (m,2) [(r, y)] von unten-Achse nach aussen/oben (Koerper links)."""
    prof = np.asarray(prof, float); m = len(prof)
    full = abs(a1 - a0 - TAU) < 1e-9
    na = segs + 1
    ang = np.linspace(a0, a1, na)
    ca, sa = np.cos(ang), np.sin(ang)
    dp = np.diff(prof, axis=0)
    pn = nrm(np.c_[dp[:, 1], -dp[:, 0]])          # Segment-Normalen (r,y)
    c = np.asarray(c, float)
    outP = []; outN = []; outU = []; outI = []; off = 0
    pl = np.r_[0, np.cumsum(np.linalg.norm(dp, axis=1))]
    for k in range(m - 1):
        r0, y0 = prof[k]; r1, y1 = prof[k + 1]
        if abs(r0) < 1e-9 and abs(r1) < 1e-9: continue
        if smooth:
            n0 = nrm(pn[max(k - 1, 0)] + pn[k]) if k > 0 and np.dot(pn[k - 1], pn[k]) > 0.5 else pn[k]
            n1 = nrm(pn[k] + pn[min(k + 1, m - 2)]) if k < m - 2 and np.dot(pn[k + 1], pn[k]) > 0.5 else pn[k]
        else:
            n0 = n1 = pn[k]
        P0 = np.c_[r0 * ca, np.full(na, y0), -r0 * sa]
        P1 = np.c_[r1 * ca, np.full(na, y1), -r1 * sa]
        N0 = np.c_[n0[0] * ca, np.full(na, n0[1]), -n0[0] * sa]
        N1 = np.c_[n1[0] * ca, np.full(na, n1[1]), -n1[0] * sa]
        P = np.r_[P0, P1] + c; N = np.r_[N0, N1]
        rr = max(abs(r0), abs(r1), 0.05)
        U = np.r_[np.c_[ang * rr, np.full(na, pl[k])], np.c_[ang * rr, np.full(na, pl[k + 1])]] * uvs
        i = np.arange(segs)
        Q = np.c_[i, i + 1, i + 1 + na, i + na]
        I = quads_to_tris(Q)
        # Orientierung je Dreieck pruefen
        I = orient(P, I, N[I].sum(1))
        # entartete Dreiecke (r=0) entfernen
        T = P[I]; ar = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        I = I[ar > 1e-12]
        outP.append(P); outN.append(N); outU.append(U); outI.append(I + off); off += len(P)
    if outP:
        mb.add_local(fr, mat, np.concatenate(outP), np.concatenate(outN), np.concatenate(outU), np.concatenate(outI))


def tube(mb, fr, mat, path, r, segs=8, caps=True, closed=False, uvs=1.0):
    """Rohr entlang Pfad (Kreisprofil, glatte Normalen)."""
    path = np.asarray(path, float); n = len(path)
    if n < 2: return
    rr = np.full(n, r) if np.isscalar(r) else np.asarray(r, float)
    if closed:
        d = nrm(np.roll(path, -1, 0) - np.roll(path, 1, 0))
    else:
        d = np.gradient(path, axis=0); d = nrm(d)
    # paralleler Transport
    ref = np.array([0, 1.0, 0]) if abs(d[0][1]) < 0.9 else np.array([1.0, 0, 0])
    a = nrm(np.cross(d[0], ref)); b = np.cross(d[0], a)
    A = [a]; B = [b]
    for i in range(1, n):
        a = a - d[i] * np.dot(a, d[i]); a = nrm(a); b = np.cross(d[i], a)
        A.append(a); B.append(b)
    A = np.array(A); B = np.array(B)
    ang = np.linspace(0, TAU, segs + 1)
    ca, sa = np.cos(ang), np.sin(ang)
    Nn = ca[None, :, None] * A[:, None, :] + sa[None, :, None] * B[:, None, :]      # (n, segs+1, 3)
    P = path[:, None, :] + rr[:, None, None] * Nn
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    U = np.stack(np.broadcast_arrays(ang[None, :] * max(r if np.isscalar(r) else rr.max(), 0.01), L[:, None]), -1)
    W = segs + 1
    nseg = n if closed else n - 1
    idx = []
    for i in range(nseg):
        j = (i + 1) % n
        k = np.arange(segs)
        idx.append(np.c_[i * W + k, j * W + k, j * W + k + 1, i * W + k + 1])
    I = quads_to_tris(np.concatenate(idx))
    PP = P.reshape(-1, 3); NN = Nn.reshape(-1, 3)
    I = orient(PP, I, NN[I].sum(1))
    mb.add_local(fr, mat, PP, NN, U.reshape(-1, 2) * uvs, I)
    if caps and not closed:
        for idx_, sgn in ((0, -1), (n - 1, 1)):
            ring = P[idx_, :-1]
            c = path[idx_]
            Pc = np.r_[c[None], ring]
            k = np.arange(segs)
            I = np.c_[np.zeros(segs, int), 1 + k, 1 + (k + 1) % segs]
            want = sgn * d[idx_]
            I = orient(Pc, I, want)
            mb.add_local(fr, mat, Pc, np.tile(want, (len(Pc), 1)), np.zeros((len(Pc), 2)), I)


def cyl(mb, fr, mat, p0, p1, r0, r1=None, segs=8, caps=True, smooth=True, uvs=1.0):
    """Kegelstumpf/Zylinder zwischen zwei lokalen Punkten."""
    r1 = r0 if r1 is None else r1
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ax = p1 - p0; h = np.linalg.norm(ax)
    if h < 1e-9: return
    ax /= h
    ref = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
    a = nrm(np.cross(ax, ref)); b = np.cross(ax, a)
    sub = Frame(p0, a, ax, None)    # u=a, v=ax, w=a x ax
    prof = [(0, 0), (r0, 0), (r1, h), (0, h)] if caps else [(r0, 0), (r1, h)]
    tmp = MB()
    lathe(tmp, WORLD, mat, prof, segs=segs, smooth=smooth, uvs=uvs)
    # tmp in sub-Rahmen, dann in fr
    for m, L in tmp.parts.items():
        for (P, N, U, I) in L:
            mb.add_local(fr, m, sub.p(P), sub.n(N), U, I)


def sphere(mb, fr, mat, c, r, segs=10, rings=6, scale=(1, 1, 1)):
    t = np.linspace(-math.pi / 2, math.pi / 2, rings + 1)
    prof = np.c_[r * np.cos(t), r * np.sin(t)]
    prof[0, 0] = 0; prof[-1, 0] = 0
    tmp = MB()
    lathe(tmp, WORLD, mat, prof, segs=segs, smooth=True)
    S = np.asarray(scale, float)
    for m, L in tmp.parts.items():
        for (P, N, U, I) in L:
            mb.add_local(fr, m, P * S + np.asarray(c, float), nrm(N / S), U, I)


def relief(mb, fr, mat, H, x0, y0, w, h, z0=0.0, uvs=1.0, skirt=True, edge_mat=None):
    """Hoehenfeld-Relief H (rows=y von unten nach oben, cols=x) [m] als Netz auf Ebene z=z0 (lokal), Normale +z.
    Rand-Schuerze zur Wand (z0) schliesst Luecken."""
    H = np.asarray(H, float); ny, nx = H.shape
    xs = np.linspace(x0, x0 + w, nx); ys = np.linspace(y0, y0 + h, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = z0 + H
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    gy, gx = np.gradient(H, ys, xs)
    N = nrm(np.c_[-gx.ravel(), -gy.ravel(), np.ones(nx * ny)])
    U = np.c_[X.ravel(), Y.ravel()] * uvs
    j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
    a = (j * nx + i).ravel()
    Q = np.c_[a, a + 1, a + 1 + nx, a + nx]
    I = quads_to_tris(Q)
    I = orient(P, I, [0, 0, 1])
    mb.add_local(fr, mat, P, N, U, I)
    if skirt:
        em = edge_mat or mat
        border = [(np.arange(nx), np.zeros(nx, int)), (np.full(ny, nx - 1), np.arange(ny)), (np.arange(nx)[::-1], np.full(nx, ny - 1)), (np.zeros(ny, int), np.arange(ny)[::-1])]
        for ii, jj in border:
            top = np.c_[xs[ii], ys[jj], z0 + H[jj, ii]]
            bot = np.c_[xs[ii], ys[jj], np.full(len(ii), z0 - 0.002)]
            k = len(ii)
            PP = np.r_[top, bot]
            Q = np.c_[np.arange(k - 1), np.arange(1, k), k + np.arange(1, k), k + np.arange(k - 1)]
            I = quads_to_tris(Q)
            out = top.mean(0) - np.r_[x0 + w / 2, y0 + h / 2, top[:, 2].mean()]
            out[2] = 0
            I = orient(PP, I, out)
            PP2, NN2, II2 = flat_tris(PP, I)
            mb.add_local(fr, em, PP2, NN2, box_uv(PP2, NN2, uvs), II2)


def poly_circle(r, n, a0=0.0, c=(0, 0)):
    a = a0 + np.arange(n) * TAU / n
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


def arch_ring(x0, x1, ys, n=12):
    """Halbkreisbogen-Loch: Rechteck x0..x1, unten y=0 bis Kaempfer ys, darueber Halbkreis. CCW."""
    r = (x1 - x0) / 2; cx = (x0 + x1) / 2
    a = np.linspace(0, math.pi, n + 1)
    top = np.c_[cx + r * np.cos(a), ys + r * np.sin(a)]
    return np.r_[[[x0, 0.0], [x1, 0.0]], top[:, :]][[0, 1] + list(range(2, n + 3))]


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

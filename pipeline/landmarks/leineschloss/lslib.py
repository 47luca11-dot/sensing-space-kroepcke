"""Leineschloss -- Geometrie-Bibliothek (numpy, ohne Blender).
Alles im Gebaeude-Lokalrahmen (u = laengs Leinstrasse nach SO, v = nach NO aus dem Portikus, h = Spiel-y), rechtshaendig.
Geometrie = Dreiecks-'Suppe' je (Material, Gruppe): Dreiecke (k,3,3) + UV in Metern (k,3,2).
Geschlossene Koerper werden ueber das Vorzeichen des Volumens nach aussen orientiert, offene Flaechen ueber 'expect'.
Wiederholte Teile (Kapitelle, Vasen ...) koennen als Instanzen (ein Netz, viele 4x4-Matrizen) abgelegt werden."""
import math
import numpy as np
import mapbox_earcut as earcut
from scipy.spatial import ConvexHull

UP = np.array([0.0, 0.0, 1.0])
EPS = 1e-9


def nrm(a):
    a = np.asarray(a, float)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, EPS)


def newell(P):
    P = np.asarray(P, float); Q = np.roll(P, -1, axis=0)
    return np.array([np.sum((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])),
                     np.sum((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])),
                     np.sum((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1]))])


def tri_normals(T):
    return np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])


def auto_uv(T):
    """UV in Metern: steile Flaechen (horizontal entlang der Flaeche, Hoehe), flache (u, v)."""
    if len(T) == 0: return np.zeros((0, 3, 2))
    n = nrm(tri_normals(T))
    steep = np.abs(n[:, 2]) < 0.72
    t = np.cross(UP[None, :], n)
    tl = np.linalg.norm(t, axis=1)
    t = np.where((tl > 1e-6)[:, None], t / np.maximum(tl, 1e-9)[:, None], np.array([1.0, 0, 0]))
    b = np.cross(n, t)
    tf = np.tile(np.array([1.0, 0, 0]), (len(T), 1))
    bf = np.where((n[:, 2] >= 0)[:, None], np.array([0, 1.0, 0]), np.array([0, -1.0, 0]))
    t = np.where(steep[:, None], t, tf); b = np.where(steep[:, None], b, bf)
    return np.stack([np.einsum('kij,kj->ki', T, t), np.einsum('kij,kj->ki', T, b)], axis=-1)


def fix_orient(T, nn):
    if len(T) == 0: return T
    n = tri_normals(T)
    nn = np.asarray(nn, float)
    d = n @ nn if nn.ndim == 1 else np.einsum('ij,ij->i', n, nn)
    flip = d < 0
    T = T.copy(); T[flip] = T[flip][:, ::-1]
    return T


def tri_poly(P, normal=None):
    P = np.asarray(P, float)
    if len(P) < 3: return np.zeros((0, 3, 3))
    nn = newell(P) if normal is None else np.asarray(normal, float)
    if np.linalg.norm(nn) < 1e-12: return np.zeros((0, 3, 3))
    nn = nrm(nn)
    if len(P) == 3: return fix_orient(P[None], nn)
    a = nrm(np.cross(nn, [1, 0, 0] if abs(nn[0]) < 0.9 else [0, 1, 0])); b = np.cross(nn, a)
    q = np.c_[P @ a, P @ b]
    idx = earcut.triangulate_float64(q, np.array([len(q)], dtype=np.uint32)).reshape(-1, 3)
    return fix_orient(P[idx], nn)


def tri_poly_holes(outer, holes, normal):
    nn = nrm(np.asarray(normal, float))
    a = nrm(np.cross(nn, [1, 0, 0] if abs(nn[0]) < 0.9 else [0, 1, 0])); b = np.cross(nn, a)
    rings = [np.asarray(outer, float)] + [np.asarray(h, float) for h in holes]
    allp = np.vstack(rings)
    q = np.c_[allp @ a, allp @ b]
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = earcut.triangulate_float64(q, ends).reshape(-1, 3)
    return fix_orient(allp[idx], nn)


def quads(Q):
    Q = np.asarray(Q, float).reshape(-1, 4, 3)
    return np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]], axis=0)


def grid_tris(G, closed_u=False, closed_v=False):
    """G (n,m,3) Punktgitter -> Dreiecke (Orientierung: Reihenfolge i,j)."""
    G = np.asarray(G, float)
    if closed_u: G = np.concatenate([G, G[:1]], axis=0)
    if closed_v: G = np.concatenate([G, G[:, :1]], axis=1)
    a = G[:-1, :-1]; b = G[1:, :-1]; c = G[1:, 1:]; d = G[:-1, 1:]
    Q = np.stack([a, b, c, d], axis=2).reshape(-1, 4, 3)
    return quads(Q)


def drop_degenerate(T, uv=None, eps=1e-10):
    if len(T) == 0: return (T, uv) if uv is not None else T
    a = np.linalg.norm(tri_normals(T), axis=1)
    k = a > eps
    return (T[k], uv[k]) if uv is not None else T[k]


def signed_volume(T):
    if len(T) == 0: return 0.0
    c = T.reshape(-1, 3).mean(axis=0); A = T - c
    return float(np.sum(np.einsum('ij,ij->i', A[:, 0], np.cross(A[:, 1], A[:, 2]))) / 6.0)


def xform(T, M):
    """T (...,3) mit 4x4-Matrix transformieren."""
    T = np.asarray(T, float)
    s = T.shape
    P = T.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    return P.reshape(s)


def mat_frame(origin, X, Y, Z):
    M = np.eye(4); M[:3, 0] = X; M[:3, 1] = Y; M[:3, 2] = Z; M[:3, 3] = origin
    return M


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4); M[0, 0] = c; M[0, 1] = -s; M[1, 0] = s; M[1, 1] = c
    return M


def trans(x, y, z):
    M = np.eye(4); M[:3, 3] = (x, y, z); return M


def scal(sx, sy=None, sz=None):
    M = np.eye(4); M[0, 0] = sx; M[1, 1] = sx if sy is None else sy; M[2, 2] = sx if sz is None else sz; return M


class Builder:
    def __init__(self):
        self.G = {}          # (mat, grp) -> [[T...], [UV...]]
        self.inst = {}       # name -> {'parts': {(mat,grp): (T,UV)}, 'mats': [...], 'M': [4x4...]}

    def add(self, mat, T, grp='main', uv=None, closed=False, expect=None):
        T = np.asarray(T, float).reshape(-1, 3, 3)
        if len(T) == 0: return
        if uv is not None: uv = np.asarray(uv, float).reshape(-1, 3, 2)
        keep = np.linalg.norm(tri_normals(T), axis=1) > 1e-11
        if not keep.all():
            T = T[keep]
            if uv is not None: uv = uv[keep]
            if expect is not None and np.asarray(expect).ndim == 2: expect = np.asarray(expect)[keep]
        if len(T) == 0: return
        if closed and signed_volume(T) < 0:
            T = T[:, ::-1]
            if uv is not None: uv = uv[:, ::-1]
        if expect is not None:
            n = tri_normals(T); e = np.asarray(expect, float)
            d = n @ e if e.ndim == 1 else np.einsum('ij,ij->i', n, e)
            if np.sum(d < 0) > np.sum(d > 0):
                T = T[:, ::-1]
                if uv is not None: uv = uv[:, ::-1]
        if uv is None: uv = auto_uv(T)
        g = self.G.setdefault((mat, grp), [[], []])
        g[0].append(T); g[1].append(uv)

    def merge(self, other, M=None):
        for (mat, grp), (Ts, UVs) in other.G.items():
            for T, uv in zip(Ts, UVs):
                if M is not None:
                    T = xform(T, M)
                    if np.linalg.det(M[:3, :3]) < 0: T = T[:, ::-1]; uv = uv[:, ::-1]
                    uv = auto_uv(T) if getattr(other, 'reuv', False) else uv
                g = self.G.setdefault((mat, grp), [[], []])
                g[0].append(T); g[1].append(uv)

    def add_instances(self, name, part, Ms):
        """part: Builder (lokal); Ms: Liste 4x4 -> als verknuepfte Duplikate exportiert."""
        e = self.inst.setdefault(name, {'part': part, 'M': []})
        e['M'].extend([np.asarray(M, float) for M in Ms])

    def tri_count(self):
        n = sum(sum(len(t) for t in v[0]) for v in self.G.values())
        ni = sum(sum(sum(len(t) for t in v[0]) for v in e['part'].G.values()) * len(e['M']) for e in self.inst.values())
        return n, ni

    def arrays(self):
        out = {}
        for k, (Ts, UVs) in self.G.items():
            out[k] = (np.concatenate(Ts), np.concatenate(UVs))
        return out


# ---------------------------------------------------------------- Grundkoerper
def box_tris(u0, u1, v0, v1, h0, h1, skip=()):
    """Achsparalleler Quader; skip: Teilmenge von {'b','t','u0','u1','v0','v1'}."""
    P = np.array([[u0, v0, h0], [u1, v0, h0], [u1, v1, h0], [u0, v1, h0],
                  [u0, v0, h1], [u1, v0, h1], [u1, v1, h1], [u0, v1, h1]], float)
    F = {'b': [0, 3, 2, 1], 't': [4, 5, 6, 7], 'v0': [0, 1, 5, 4], 'u1': [1, 2, 6, 5], 'v1': [2, 3, 7, 6], 'u0': [3, 0, 4, 7]}
    Q = [P[F[k]] for k in F if k not in skip]
    return quads(np.array(Q)) if Q else np.zeros((0, 3, 3))


def box(b, mat, u0, u1, v0, v1, h0, h1, skip=(), grp='main', uv=None):
    if u1 < u0: u0, u1 = u1, u0
    if v1 < v0: v0, v1 = v1, v0
    if h1 < h0: h0, h1 = h1, h0
    T = box_tris(u0, u1, v0, v1, h0, h1, skip)
    b.add(mat, T, grp)   # Reihenfolge der Ecken ist nach aussen orientiert


def obox_tris(c, X, Y, Z, hx, hy, hz):
    """Orientierter Quader: Mitte c, Achsen X,Y,Z (Einheitsvektoren), Halbmasse."""
    T = box_tris(-hx, hx, -hy, hy, -hz, hz)
    M = mat_frame(np.asarray(c, float), nrm(X), nrm(Y), nrm(Z))
    T = xform(T, M)
    if np.linalg.det(M[:3, :3]) < 0: T = T[:, ::-1]
    return T


def hull_tris(P):
    P = np.asarray(P, float)
    h = ConvexHull(P)
    T = P[h.simplices]
    c = P.mean(axis=0)
    n = tri_normals(T); d = np.einsum('ij,ij->i', n, T.mean(axis=1) - c)
    T[d < 0] = T[d < 0][:, ::-1]
    return T


def bevel_box_tris(u0, u1, v0, v1, h0, h1, c=0.02, top_only=False):
    """Quader mit 45-Grad-Fasen (Breite c) an allen Kanten (konvexe Huelle)."""
    pts = []
    for (a0, a1, b0, b1, z0, z1) in [(u0 + c, u1 - c, v0 + c, v1 - c, h0, h1), (u0, u1, v0 + c, v1 - c, h0 + c, h1 - c), (u0 + c, u1 - c, v0, v1, h0 + c, h1 - c)]:
        for a in (a0, a1):
            for bb in (b0, b1):
                for z in (z0, z1):
                    pts.append((a, bb, z))
    if top_only:
        pts = [(p[0], p[1], h0) if p[2] < h0 + c + 1e-9 else p for p in pts]
    return hull_tris(np.array(pts))


def prism_tris(poly, h0, h1, top=True, bottom=False, sides=True):
    """Vertikales Prisma ueber 2D-Polygon (u,v)."""
    poly = np.asarray(poly, float)
    if signed_area(poly) < 0: poly = poly[::-1]
    n = len(poly); out = []
    if sides:
        a = poly; b = np.roll(poly, -1, axis=0)
        Q = np.stack([np.c_[a, np.full(n, h0)], np.c_[b, np.full(n, h0)], np.c_[b, np.full(n, h1)], np.c_[a, np.full(n, h1)]], axis=1)
        out.append(quads(Q))
    if top: out.append(tri_poly(np.c_[poly, np.full(n, h1)], [0, 0, 1]))
    if bottom: out.append(tri_poly(np.c_[poly, np.full(n, h0)], [0, 0, -1]))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def signed_area(p):
    p = np.asarray(p, float)
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


# ---------------------------------------------------------------- Profile ziehen
def sweep_tris(path, profile, bv=UP, closed=False, caps=True, side_flip=False):
    """Profil (M,2: x = seitlich 'aussen', y = entlang bv) entlang Pfad (N,3) ziehen; Gehrung an den Ecken.
    'aussen' = tangente x bv (bei gegen den Uhrzeigersinn umlaufenem Grundriss und bv=oben: nach aussen)."""
    P = np.asarray(path, float); prof = np.asarray(profile, float); bv = nrm(bv)
    N = len(P)
    if closed:
        tin = nrm(P - np.roll(P, 1, axis=0)); tout = nrm(np.roll(P, -1, axis=0) - P)
    else:
        d = nrm(np.diff(P, axis=0))
        tin = np.vstack([d[:1], d]); tout = np.vstack([d, d[-1:]])
    sin_ = nrm(np.cross(tin, bv)); sout = nrm(np.cross(tout, bv))
    s = nrm(sin_ + sout)
    cosh = np.maximum(np.einsum('ij,ij->i', s, sout), 0.2)
    s = s / cosh[:, None]
    if side_flip: s = -s
    R = P[:, None, :] + prof[None, :, 0, None] * s[:, None, :] + prof[None, :, 1, None] * bv[None, None, :]
    T = grid_tris(R, closed_u=closed)
    out = [T]
    if caps and not closed and len(prof) >= 3:
        c0 = tri_poly(R[0], -tout[0]); c1 = tri_poly(R[-1], tout[-1])
        out += [c0, c1]
    T = np.concatenate(out)
    return T


def sweep(b, mat, path, profile, bv=UP, closed=False, caps=True, grp='main', side_flip=False):
    T = sweep_tris(path, profile, bv, closed, caps, side_flip)
    b.add(mat, T, grp, closed=caps or closed)


def lathe_tris(profile, segs=24, center=(0, 0, 0), a0=0.0, a1=2 * math.pi, cap_bottom=False, cap_top=False):
    """Rotationsflaeche: profile (M,2) = (radius, hoehe) von unten nach oben."""
    prof = np.asarray(profile, float); c = np.asarray(center, float)
    full = abs(a1 - a0 - 2 * math.pi) < 1e-9
    n = segs if full else segs + 1
    ang = a0 + (a1 - a0) * np.arange(n) / segs
    ca, sa = np.cos(ang), np.sin(ang)
    G = np.stack([c[0] + prof[None, :, 0] * ca[:, None], c[1] + prof[None, :, 0] * sa[:, None], c[2] + np.broadcast_to(prof[None, :, 1], (n, len(prof)))], axis=-1)
    T = grid_tris(G, closed_u=full)
    out = [T]
    if cap_bottom and prof[0, 0] > 1e-6:
        out.append(tri_poly(G[:, 0][::-1] if full else G[:, 0], [0, 0, -1]))
    if cap_top and prof[-1, 0] > 1e-6:
        out.append(tri_poly(G[:, -1], [0, 0, 1]))
    return np.concatenate(out)


def lathe(b, mat, profile, segs=24, center=(0, 0, 0), grp='main', caps=True):
    T = lathe_tris(profile, segs, center, cap_bottom=caps, cap_top=caps)
    b.add(mat, T, grp, closed=caps)


# ---------------------------------------------------------------- Wand mit Oeffnungen
class Wall:
    """Ebene Wandflaeche: Ursprung o (Fusspunkt links aussen), Richtung t (horizontal), Normale n (aussen).
    Lokale 2D-Koordinaten (s, h): s entlang t ab o, h = absolute Hoehe."""
    def __init__(self, p0, p1, n=None):
        p0 = np.asarray(p0, float)[:2]; p1 = np.asarray(p1, float)[:2]
        d = p1 - p0; L = float(np.hypot(*d))
        self.o = p0; self.t = np.array([d[0] / L, d[1] / L, 0.0]); self.L = L
        nn = np.array([self.t[1], -self.t[0], 0.0]) if n is None else nrm(np.asarray(n, float))
        self.n = nn

    def P(self, s, h, d=0.0):
        """Weltpunkt(e) fuer (s, h) und Abstand d vor (positiv) / hinter (negativ) der Wand."""
        s = np.asarray(s, float); h = np.asarray(h, float); d = np.asarray(d, float)
        base = np.array([self.o[0], self.o[1], 0.0])
        return base + s[..., None] * self.t + d[..., None] * self.n + h[..., None] * UP

    def poly(self, pts2d, d=0.0):
        pts2d = np.asarray(pts2d, float)
        return self.P(pts2d[:, 0], pts2d[:, 1], np.full(len(pts2d), d))


def rect2d(s0, s1, h0, h1):
    return np.array([[s0, h0], [s1, h0], [s1, h1], [s0, h1]], float)


def arch2d(s0, s1, h0, hs, segs=12, rise=None):
    """Rundbogen- (rise=None: Halbkreis) bzw. Segmentbogen-Oeffnung: unten h0, Kaempfer hs."""
    w = s1 - s0; r = w / 2; c = (s0 + s1) / 2
    pts = [[s0, h0], [s1, h0], [s1, hs]]
    if rise is None or rise >= r - 1e-6:
        for i in range(1, segs):
            a = math.pi * i / segs
            pts.append([c + r * math.cos(a), hs + r * math.sin(a)])
    else:
        R = (r * r + rise * rise) / (2 * rise); a0 = math.asin(r / R)
        for i in range(1, segs):
            a = math.pi / 2 - a0 + 2 * a0 * i / segs
            pts.append([c + R * math.cos(a), hs - (R - rise) + R * math.sin(a)])
    pts.append([s0, hs])
    return np.array(pts, float)


def wall_face(b, mat, W, outline2d, holes2d=(), d=0.0, grp='main'):
    T = tri_poly_holes(W.poly(outline2d, d), [W.poly(h, d) for h in holes2d], W.n)
    b.add(mat, T, grp)


def reveal(b, mat, W, hole2d, d0, d1, grp='main', skip_bottom=False):
    """Laibung: Lochumriss von Tiefe d0 (Wandflaeche) bis d1 (dahinter) extrudieren, Flaechen zeigen ins Loch."""
    h = np.asarray(hole2d, float)
    n = len(h)
    A = W.poly(h, d0); B = W.poly(h, d1)
    Q = np.stack([A, np.roll(A, -1, axis=0), np.roll(B, -1, axis=0), B], axis=1)
    if skip_bottom:
        keep = ~((np.abs(h[:, 1] - h[:, 1].min()) < 1e-6) & (np.abs(np.roll(h[:, 1], -1) - h[:, 1].min()) < 1e-6))
        Q = Q[keep]
    T = quads(Q)
    c2 = h.mean(axis=0)
    cen = W.P(c2[0], c2[1], (d0 + d1) / 2)
    # nach innen ins Loch zeigend
    n_ = tri_normals(T); m = T.mean(axis=1)
    flip = np.einsum('ij,ij->i', n_, cen - m) < 0
    T = T.copy(); T[flip] = T[flip][:, ::-1]
    b.add(mat, T, grp)


def frame_profile_ring(b, mat, W, ring2d, profile, grp='main', closed=True):
    """Gewaende/Rahmen: Profil (x = von der Oeffnung weg in der Wandebene, y = aus der Wand) um ring2d ziehen."""
    P = W.poly(ring2d, 0.0)
    T = sweep_tris(P, profile, bv=W.n, closed=closed, caps=not closed)
    # Orientierung: Profilpunkte mit x>0 sollen von der Oeffnung weg liegen -> Umlaufsinn pruefen
    b.add(mat, T, grp, closed=True)


def relief_tris(H, W, s0, h0, sx, sy, depth, d0=0.0, thr=0.04, sides=True):
    """Hoehenbild H (ny,nx, 0..1, Zeile 0 = oben) als Relief vor der Wand W: Rechteck (s0,h0)-(s0+sx,h0+sy),
    Tiefe 'depth'*H vor der Ebene d0. Zellen mit H<thr werden weggelassen; Kanten erhalten Seitenflaechen."""
    H = np.asarray(H, float); ny, nx = H.shape
    ss = s0 + sx * (np.arange(nx) + 0.5) / nx
    hh = h0 + sy * (1 - (np.arange(ny) + 0.5) / ny)
    S, HH = np.meshgrid(ss, hh)
    D = d0 + depth * H
    G = W.P(S, HH, D)
    on = H >= thr
    cell = on[:-1, :-1] & on[1:, :-1] & on[1:, 1:] & on[:-1, 1:]
    a = G[:-1, :-1]; bq = G[:-1, 1:]; c = G[1:, 1:]; d = G[1:, :-1]
    Q = np.stack([a, d, c, bq], axis=2)[cell]
    T = quads(Q) if len(Q) else np.zeros((0, 3, 3))
    T = fix_orient(T, W.n) if len(T) else T
    out = [T]
    if sides:
        # Raender: Kante zwischen belegter und leerer Zelle -> senkrechte Flaeche zur Ebene d0
        cy, cx = cell.shape
        pad = np.zeros((cy + 2, cx + 2), bool); pad[1:-1, 1:-1] = cell
        side = []; exp = []
        base0 = W.P(0, 0, d0)
        for (dy, dx) in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = pad[1 + dy:1 + dy + cy, 1 + dx:1 + dx + cx]
            m = cell & ~nb
            ys, xs = np.nonzero(m)
            ov = dx * W.t - dy * UP
            for y, x in zip(ys, xs):
                if dx == 1: e = (G[y, x + 1], G[y + 1, x + 1])
                elif dx == -1: e = (G[y + 1, x], G[y, x])
                elif dy == 1: e = (G[y + 1, x + 1], G[y + 1, x])
                else: e = (G[y, x], G[y, x + 1])
                p0, p1 = e
                q0 = p0 - W.n * (np.dot(p0 - base0, W.n) + 0.003)
                q1 = p1 - W.n * (np.dot(p1 - base0, W.n) + 0.003)
                side.append([p0, p1, q1, q0]); exp.append(ov); exp.append(ov)
        if side:
            ST = quads(np.array(side))
            E = np.array(exp)
            E = np.concatenate([E[0::2], E[1::2]])   # quads(): erst alle ersten, dann alle zweiten Dreiecke
            out.append(fix_orient(ST, E))
    T = np.concatenate(out)
    return T, on


def orient_away(T, center):
    n = tri_normals(T); m = T.mean(axis=1)
    flip = np.einsum('ij,ij->i', n, m - np.asarray(center, float)) < 0
    T = T.copy(); T[flip] = T[flip][:, ::-1]
    return T

"""Geometrie-Kern fuer das Ballhof-Ensemble (reines numpy, ausserhalb von Blender).

Spielframe: x Ost, y oben, z Sued (Meter, Ursprung Kroepcke-Uhr).
Frame(o, ex, ey, ez): lokale (a, b, c) -> o + ex*a + ey*b + ez*c.  Fassadenframe: ex = von aussen gesehen nach rechts,
ey = oben, ez = Aussennormale.  2D-Polygone in (a, b) gegen den Uhrzeigersinn (von +c gesehen) -> Normale +c.
MB sammelt Dreiecke je Material (Position, Normale, UV in Metern); finish() schweisst und teilt UV durch die Kachelgroesse.
"""
import math
import numpy as np
import mapbox_earcut as earcut

UP = np.array([0.0, 1.0, 0.0])


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def area2(p):
    p = np.asarray(p, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def ccw(p):
    p = np.asarray(p, float)
    return p if area2(p) > 0 else p[::-1].copy()


def offset_poly(pts, d):
    """Kantenweiser Mitre-Versatz eines CCW-Polygons (2D). d: Skalar oder je Kante (Kante i = pts[i] -> pts[i+1]);
    positiv = nach aussen. Gibt neue Ecken (gleiche Anzahl) zurueck."""
    p = np.asarray(pts, float)
    n = len(p)
    d = np.broadcast_to(np.asarray(d, float), (n,))
    out = np.zeros_like(p)
    for i in range(n):
        # Ecke i liegt zwischen Kante i-1 (p[i-1]->p[i]) und Kante i (p[i]->p[i+1])
        a0, a1 = p[i - 1], p[i]
        b0, b1 = p[i], p[(i + 1) % n]
        ta = unit(a1 - a0); tb = unit(b1 - b0)
        na = np.array([ta[1], -ta[0]]); nb = np.array([tb[1], -tb[0]])
        pa = a1 + na * d[i - 1]; pb = b0 + nb * d[i]
        den = ta[0] * tb[1] - ta[1] * tb[0]
        if abs(den) < 1e-6:
            out[i] = (pa + pb) / 2
        else:
            t = ((pb[0] - pa[0]) * tb[1] - (pb[1] - pa[1]) * tb[0]) / den
            q = pa + ta * t
            # Begrenzung der Gehrung
            if np.linalg.norm(q - p[i]) > 4 * max(abs(d[i - 1]), abs(d[i]), 1e-3):
                q = (pa + pb) / 2
            out[i] = q
    return out


class Frame:
    def __init__(self, o, ex, ey, ez):
        self.o = np.asarray(o, float); self.ex = unit(ex); self.ey = unit(ey); self.ez = unit(ez)
        self.M = np.stack([self.ex, self.ey, self.ez], 1)

    @staticmethod
    def facade(pL, pR, y0=0.0):
        """pL -> pR (Plan x,z), von aussen gesehen links -> rechts; ez = Aussennormale"""
        t = unit([pR[0] - pL[0], 0.0, pR[1] - pL[1]])
        n = np.array([-t[2], 0.0, t[0]])
        return Frame([pL[0], y0, pL[1]], t, UP, n)

    def P(self, pts):
        return self.o + np.asarray(pts, float) @ self.M.T

    def D(self, d):
        return np.asarray(d, float) @ self.M.T

    def sub(self, a=0.0, b=0.0, c=0.0):
        return Frame(self.P([a, b, c]), self.ex, self.ey, self.ez)

    def rot_b(self, ang, a=0.0, b=0.0, c=0.0):
        """um die lokale b-Achse gedreht (Grad, positiv: ex -> -ez)"""
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        return Frame(self.P([a, b, c]), self.ex * ca - self.ez * sa, self.ey, self.ex * sa + self.ez * ca)

    def rot_a(self, ang, a=0.0, b=0.0, c=0.0):
        """um die lokale a-Achse gedreht (Grad, positiv: ey -> ez)"""
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        return Frame(self.P([a, b, c]), self.ex, self.ey * ca + self.ez * sa, -self.ey * sa + self.ez * ca)

    def local(self, p):
        return (np.asarray(p, float) - self.o) @ self.M

    def mat4(self):
        M = np.eye(4); M[:3, :3] = self.M; M[:3, 3] = self.o
        return M


WORLD = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])


def tri_poly(outer, holes=()):
    """earcut: outer (n,2) + holes -> Dreiecke (Indexliste) in gemeinsamer Punktliste"""
    rings = [np.asarray(outer, float)] + [np.asarray(h, float) for h in holes]
    V = np.concatenate(rings, 0)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    I = np.asarray(earcut.triangulate_float64(V, ends), np.int64).reshape(-1, 3)
    if len(I):
        T = V[I]
        s = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 1, 1] - T[:, 0, 1]) * (T[:, 2, 0] - T[:, 0, 0])
        I = np.where((s < 0)[:, None], I[:, ::-1], I)        # immer gegen den Uhrzeigersinn
    return V, I


class MB:
    """Mesh-Builder: je Material Dreiecksecken (P, N, UV) als Listen von Arrays."""

    def __init__(self):
        self.d = {}
        self.ntri = 0

    def add(self, mat, P, N, UV):
        P = np.asarray(P, float).reshape(-1, 3); N = np.asarray(N, float).reshape(-1, 3); UV = np.asarray(UV, float).reshape(-1, 2)
        if not len(P): return
        assert len(P) % 3 == 0 and len(N) == len(P) and len(UV) == len(P)
        L = self.d.setdefault(mat, [[], [], []])
        L[0].append(P); L[1].append(N); L[2].append(UV)
        self.ntri += len(P) // 3

    def tris(self, mat, P, I, UV, N=None):
        """indizierte Dreiecke: P (n,3), I (m,3), UV (n,2); N None -> Flaechennormalen"""
        P = np.asarray(P, float); I = np.asarray(I, np.int64).reshape(-1, 3); UV = np.asarray(UV, float)
        if not len(I): return
        T = P[I]
        if N is None:
            fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            NN = np.repeat(fn[:, None, :], 3, 1)
        else:
            NN = np.asarray(N, float)[I]
        self.add(mat, T.reshape(-1, 3), NN.reshape(-1, 3), UV[I].reshape(-1, 2))

    # ---------------------------------------------------------------- Grundformen
    def poly(self, mat, F, outer, holes=(), c=0.0, uv=None, flip=False):
        """ebenes Polygon im Frame F bei Tiefe c, Normale +ez (flip: -ez). uv(a,b)->(u,v) optional (Meter)."""
        outer = ccw(outer)
        hs = [np.asarray(h, float)[::-1] if area2(h) > 0 else np.asarray(h, float) for h in holes]
        V, I = tri_poly(outer, hs)
        if not len(I): return
        if flip: I = I[:, ::-1]
        P = F.P(np.c_[V, np.full(len(V), c)])
        UV = uv(V) if uv is not None else V.copy()
        if flip and uv is None: UV[:, 0] = -UV[:, 0]
        n = -F.ez if flip else F.ez
        T = P[I]
        self.add(mat, T.reshape(-1, 3), np.tile(n, (I.size, 1)), UV[I].reshape(-1, 2))

    def quad(self, mat, p0, p1, p2, p3, uv=None):
        """p0..p3 gegen den Uhrzeigersinn von vorne; uv: (4,2) oder None (planar, Meter)"""
        P = np.array([p0, p1, p2, p3], float)
        n = unit(np.cross(P[1] - P[0], P[2] - P[0]) + np.cross(P[2] - P[0], P[3] - P[0]))
        if uv is None:
            e1 = unit(P[1] - P[0]); e2 = np.cross(n, e1)
            uv = np.c_[(P - P[0]) @ e1, (P - P[0]) @ e2]
        uv = np.asarray(uv, float)
        I = np.array([0, 1, 2, 0, 2, 3])
        self.add(mat, P[I], np.tile(n, (6, 1)), uv[I])

    def box(self, mat, F, a0, a1, b0, b1, c0, c1, skip=(), uvs=1.0):
        """achsparalleler Quader im Frame F; skip: Teilmenge von {'-a','+a','-b','+b','-c','+c'}"""
        A = lambda a, b, c: F.P([a, b, c])
        if '+c' not in skip:
            self.quad(mat, A(a0, b0, c1), A(a1, b0, c1), A(a1, b1, c1), A(a0, b1, c1), np.array([[a0, b0], [a1, b0], [a1, b1], [a0, b1]]) * uvs)
        if '-c' not in skip:
            self.quad(mat, A(a1, b0, c0), A(a0, b0, c0), A(a0, b1, c0), A(a1, b1, c0), np.array([[-a1, b0], [-a0, b0], [-a0, b1], [-a1, b1]]) * uvs)
        if '+a' not in skip:
            self.quad(mat, A(a1, b0, c1), A(a1, b0, c0), A(a1, b1, c0), A(a1, b1, c1), np.array([[-c1, b0], [-c0, b0], [-c0, b1], [-c1, b1]]) * uvs)
        if '-a' not in skip:
            self.quad(mat, A(a0, b0, c0), A(a0, b0, c1), A(a0, b1, c1), A(a0, b1, c0), np.array([[c0, b0], [c1, b0], [c1, b1], [c0, b1]]) * uvs)
        if '+b' not in skip:
            self.quad(mat, A(a0, b1, c1), A(a1, b1, c1), A(a1, b1, c0), A(a0, b1, c0), np.array([[a0, -c1], [a1, -c1], [a1, -c0], [a0, -c0]]) * uvs)
        if '-b' not in skip:
            self.quad(mat, A(a0, b0, c0), A(a1, b0, c0), A(a1, b0, c1), A(a0, b0, c1), np.array([[a0, c0], [a1, c0], [a1, c1], [a0, c1]]) * uvs)

    def prism(self, mat, F, outer, c0, c1, bevel=0.0, back=False, grain=None, front=True):
        """Polygon (a,b) im Frame F von c0 bis c1 extrudiert; Vorderseite bei c1 mit Fase 'bevel'.
        grain: Einheitsvektor (2D) der Maserrichtung fuer die UV der Vorderseite (u entlang grain)."""
        P2 = ccw(outer)
        n = len(P2)
        if n < 3 or abs(area2(P2)) < 1e-5: return
        g = np.array([1.0, 0.0]) if grain is None else unit(grain)
        gp = np.array([-g[1], g[0]])
        uvf = lambda V: np.c_[V @ g, V @ gp]
        bev = bevel
        inner = P2
        if bev > 0:
            inner = offset_poly(P2, -bev)
            # Fase nur, wenn das Innenpolygon nicht umklappt
            if area2(inner) <= 0.2 * area2(P2) or np.any(np.linalg.norm(inner - P2, axis=1) > 3 * bev):
                inner = P2; bev = 0.0
        if front:
            self.poly(mat, F, inner, c=c1, uv=uvf)
        if back:
            self.poly(mat, F, P2, c=c0, uv=uvf, flip=True)
        cs = c1 - bev
        for i in range(n):
            p, q = P2[i], P2[(i + 1) % n]
            L = float(np.linalg.norm(q - p))
            if L < 1e-6: continue
            t = (q - p) / L
            u0 = float(p @ g) if abs(t @ g) > 0.7 else float(p @ gp)
            u1 = u0 + L
            if cs - c0 > 1e-6:
                self.quad(mat, F.P([p[0], p[1], c0]), F.P([q[0], q[1], c0]), F.P([q[0], q[1], cs]), F.P([p[0], p[1], cs]),
                          np.array([[u0, c0], [u1, c0], [u1, cs], [u0, cs]]))
            if bev > 0:
                pi, qi = inner[i], inner[(i + 1) % n]
                self.quad(mat, F.P([p[0], p[1], cs]), F.P([q[0], q[1], cs]), F.P([qi[0], qi[1], c1]), F.P([pi[0], pi[1], c1]),
                          np.array([[u0, cs], [u1, cs], [u1, c1 + 0.02], [u0, c1 + 0.02]]))

    def tube(self, mat, path, r, n=8, caps=True, half=False, smooth=True, up=None):
        """Rohr entlang Polylinie (k,3). half: nur untere Haelfte (Rinne, offen nach oben)"""
        path = np.asarray(path, float)
        k = len(path)
        if half:
            angs = np.linspace(math.pi, 2 * math.pi, n + 1)
        else:
            angs = np.linspace(0, 2 * math.pi, n + 1)
        rings = []
        for i in range(k):
            if i == 0: t = unit(path[1] - path[0])
            elif i == k - 1: t = unit(path[-1] - path[-2])
            else: t = unit(unit(path[i] - path[i - 1]) + unit(path[i + 1] - path[i]))
            ref = np.array([0, 1.0, 0]) if up is None else np.asarray(up, float)
            if abs(t @ ref) > 0.95: ref = np.array([1.0, 0, 0])
            e1 = unit(np.cross(t, ref)); e2 = np.cross(e1, t)     # e2 ~ "oben"
            rings.append([path[i] + r * (math.cos(a) * e1 + math.sin(a) * e2) for a in angs])
        rings = np.array(rings)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        P, N, UV = [], [], []
        for i in range(k - 1):
            for j in range(n):
                q = [rings[i, j], rings[i + 1, j], rings[i + 1, j + 1], rings[i, j + 1]]
                c = [path[i], path[i + 1], path[i + 1], path[i]]
                nn = [unit(q[m] - c[m]) for m in range(4)]
                if not smooth:
                    f = unit(np.cross(q[1] - q[0], q[2] - q[0])); nn = [f] * 4
                uv = [[s[i], j * r * 2 * math.pi / n], [s[i + 1], j * r * 2 * math.pi / n], [s[i + 1], (j + 1) * r * 2 * math.pi / n], [s[i], (j + 1) * r * 2 * math.pi / n]]
                for m in (0, 1, 2, 0, 2, 3):
                    P.append(q[m]); N.append(nn[m]); UV.append(uv[m])
        # Winding pruefen (Normale nach aussen)
        P = np.array(P); N = np.array(N); UV = np.array(UV)
        T = P.reshape(-1, 3, 3)
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        if len(fn) and np.sum(np.einsum('ij,ij->i', fn, N.reshape(-1, 3, 3).mean(1))) < 0:
            T = T[:, ::-1]; N = N.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3); UV = UV.reshape(-1, 3, 2)[:, ::-1].reshape(-1, 2)
            P = T.reshape(-1, 3)
        if half:
            # Innenseite (Rinne von oben sichtbar): gleiche Flaeche, umgekehrt, leicht nach innen
            Pi = P.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
            Ni = -N.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
            Pi = Pi + Ni * 0.004
            UVi = UV.reshape(-1, 3, 2)[:, ::-1].reshape(-1, 2)
            P = np.r_[P, Pi]; N = np.r_[N, Ni]; UV = np.r_[UV, UVi]
        self.add(mat, P, N, UV)
        if caps and not half:
            for i, sgn in ((0, -1), (k - 1, 1)):
                t = unit(path[1] - path[0]) if i == 0 else unit(path[-1] - path[-2])
                ring = rings[i, :-1]
                c = path[i]
                for j in range(n):
                    a, b = ring[j], ring[(j + 1) % n]
                    tri = [c, a, b] if sgn > 0 else [c, b, a]
                    f = unit(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
                    if f @ (t * sgn) < 0: tri = tri[::-1]
                    self.add(mat, tri, [t * sgn] * 3, [[0, 0], [a[0] - c[0], a[1] - c[1]], [b[0] - c[0], b[1] - c[1]]])

    def lathe(self, mat, F, prof, n=12, smooth=True, a0=0.0, a1=2 * math.pi):
        """Rotationskoerper um die lokale b-Achse von F; prof [(r, b), ...] von unten nach oben"""
        prof = np.asarray(prof, float)
        angs = np.linspace(a0, a1, n + 1)
        full = abs(a1 - a0 - 2 * math.pi) < 1e-6
        P, N, UV = [], [], []
        s = np.r_[0, np.cumsum(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])))]
        for i in range(len(prof) - 1):
            r0, b0 = prof[i]; r1, b1 = prof[i + 1]
            dr, db = r1 - r0, b1 - b0
            ln = math.hypot(dr, db) or 1.0
            nr, nb = db / ln, -dr / ln          # Profilnormale (nach aussen, wenn Profil aufwaerts)
            for j in range(n):
                q = []; nn = []
                for (r, b, aa) in ((r0, b0, angs[j]), (r1, b1, angs[j]), (r1, b1, angs[j + 1]), (r0, b0, angs[j + 1])):
                    q.append(F.P([r * math.cos(aa), b, -r * math.sin(aa)]))
                for aa in (angs[j], angs[j], angs[j + 1], angs[j + 1]):
                    nn.append(F.D([nr * math.cos(aa), nb, -nr * math.sin(aa)]))
                if not smooth:
                    am = (angs[j] + angs[j + 1]) / 2; nn = [F.D([nr * math.cos(am), nb, -nr * math.sin(am)])] * 4
                uv = [[j / n * 2, s[i]], [j / n * 2, s[i + 1]], [(j + 1) / n * 2, s[i + 1]], [(j + 1) / n * 2, s[i]]]
                for m in (0, 1, 2, 0, 2, 3):
                    P.append(q[m]); N.append(nn[m]); UV.append(uv[m])
        P = np.array(P); N = np.array(N); UV = np.array(UV)
        T = P.reshape(-1, 3, 3)
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ok = np.linalg.norm(fn, axis=1) > 1e-12
        if np.sum(np.einsum('ij,ij->i', fn[ok], N.reshape(-1, 3, 3).mean(1)[ok])) < 0:
            P = T[:, ::-1].reshape(-1, 3); N = N.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3); UV = UV.reshape(-1, 3, 2)[:, ::-1].reshape(-1, 2)
            T = P.reshape(-1, 3, 3)
        keep = np.repeat(ok, 3)
        self.add(mat, P[keep], N[keep], UV[keep])

    def merge(self, other, M4=None, matmap=None):
        M4 = np.eye(4) if M4 is None else np.asarray(M4, float)
        R = M4[:3, :3]; Rn = np.linalg.inv(R).T
        for mat, (Ps, Ns, UVs) in other.d.items():
            m2 = matmap.get(mat, mat) if matmap else mat
            for P, N, UV in zip(Ps, Ns, UVs):
                P2 = P @ R.T + M4[:3, 3]
                N2 = N @ Rn.T; N2 /= np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-12)
                if np.linalg.det(R) < 0:
                    P2 = P2.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3); N2 = N2.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3); UV = UV.reshape(-1, 3, 2)[:, ::-1].reshape(-1, 2)
                self.add(m2, P2, N2, UV)

    def arrays(self, mat):
        Ps, Ns, UVs = self.d[mat]
        return np.concatenate(Ps), np.concatenate(Ns), np.concatenate(UVs)

    def finish(self, tiles=None, mats=None):
        """-> {mat: (P, N, UV, I)} geschweisst; UV / Kachelgroesse"""
        out = {}
        tiles = tiles or {}
        for mat in (mats or list(self.d.keys())):
            if mat not in self.d: continue
            P, N, UV = self.arrays(mat)
            ts = tiles.get(mat, 1.0)
            ts = np.broadcast_to(np.asarray(ts, float), (2,))
            UV = UV / ts
            key = np.c_[np.round(P * 2000), np.round(N * 500), np.round(UV * 2000)].astype(np.int64)
            _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            I = inv.reshape(-1, 3)
            # degenerierte Dreiecke entfernen
            good = (I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 0] != I[:, 2])
            Pv = P[first]
            T = Pv[I[good]]
            ar = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
            gi = np.where(good)[0][ar > 1e-10]
            out[mat] = (Pv, N[first], UV[first], I[gi])
        return out


def mirror_x():
    M = np.eye(4); M[0, 0] = -1
    return M

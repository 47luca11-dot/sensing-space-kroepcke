"""Zusatz-Geometrie (Ausbau 2): lokale Rahmen (s entlang, t quer/rechts, y oben), Quader, Innenraeume mit nach innen
gerichteten Flaechen, orientierte Quads, Profil-Extrusionen. Alle Flaechen einseitig (FrontSide) korrekt orientiert."""
import math
import numpy as np
import geom as G

def quad(mat, p0, p1, p2, p3, uv=None, uvscale=1.0):
    """Quad p0..p3 gegen den Uhrzeigersinn von der Sichtseite gesehen (Normale = (p1-p0)x(p3-p0))."""
    P = np.array([p0, p1, p2, p3], float)
    n = np.cross(P[1] - P[0], P[3] - P[0]); ln = np.linalg.norm(n)
    if ln < 1e-12: return None
    n /= ln
    if uv is None:
        e = np.linalg.norm(P[1] - P[0]); h = np.linalg.norm(P[3] - P[0])
        uv = np.array([[0, 0], [e, 0], [e, h], [0, h]]) / uvscale
        uv[:, 1] *= -1
    return G.part(mat, P, [(0, 1, 2), (0, 2, 3)], np.tile(n, (4, 1)), np.asarray(uv, float))

def tri_strip_quads(mat, A, B, uvA=None, uvB=None, flip=False):
    """Band zwischen Polylinien A und B (je k Punkte, 3D). Normale = (A[i+1]-A[i]) x (B[i]-A[i]) (flip kehrt um)."""
    A = np.asarray(A, float); B = np.asarray(B, float); k = len(A)
    pos = np.concatenate([A, B])
    if uvA is None:
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))]
        w = np.linalg.norm(B - A, axis=1)
        uvA = np.c_[L, np.zeros(k)]; uvB = np.c_[L, -w]
    uv = np.concatenate([uvA, uvB])
    idx = []
    for i in range(k - 1):
        a, b, c, d = i, i + 1, k + i + 1, k + i
        idx += [(a, b, c), (a, c, d)] if not flip else [(a, c, b), (a, d, c)]
    p = G.part(mat, pos, idx, None, uv)
    return p

class Frame:
    """Lokaler Rahmen: Ursprung o (x,z), Richtung d (Einheitsvektor xz). right = (-dz, dx)."""
    def __init__(self, o, d):
        self.o = np.asarray(o, float); self.d = np.asarray(d, float) / np.linalg.norm(d)
        self.r = np.array([-self.d[1], self.d[0]])
        self.yaw = math.atan2(-self.d[1], self.d[0])     # G.box/rot_y: lokales +x -> d, lokales +z -> r
    def xz(self, s, t):
        return self.o + self.d * s + self.r * t
    def P(self, s, t, y):
        q = self.xz(s, t); return np.array([q[0], y, q[1]])
    def vec(self, ds, dt, dy=0.0):
        v = self.d * ds + self.r * dt; return np.array([v[0], dy, v[1]])
    def st(self, x, z):
        q = np.array([x, z]) - self.o; return q @ self.d, q @ self.r
    def box(self, mat, s0, s1, t0, t1, y0, y1, uvscale=1.0):
        c = self.xz((s0 + s1) / 2, (t0 + t1) / 2)
        return G.box(mat, (c[0], y0, c[1]), (abs(s1 - s0), abs(y1 - y0), abs(t1 - t0)), self.yaw, uvscale)
    def quad(self, mat, pts, uvscale=1.0, uv=None):
        """pts: 4 lokale (s,t,y) gegen den Uhrzeigersinn von der Sichtseite"""
        return quad(mat, *[self.P(*p) for p in pts], uv=uv, uvscale=uvscale)
    def hquad(self, mat, s0, s1, t0, t1, y, up=True, uvscale=1.0, world_uv=True, y_s=None):
        """horizontale Flaeche (oben sichtbar wenn up). y_s: optional (y bei s0, y bei s1) fuer geneigte Flaechen"""
        ya, yb = (y, y) if y_s is None else y_s
        P = [self.P(s0, t0, ya), self.P(s0, t1, ya), self.P(s1, t1, yb), self.P(s1, t0, yb)]
        # Reihenfolge so, dass Normale nach oben (up) zeigt
        n = np.cross(P[1] - P[0], P[3] - P[0])
        if (n[1] > 0) != up: P = P[::-1]
        P = np.array(P)
        uv = P[:, [0, 2]] / uvscale if world_uv else None
        return quad(mat, *P, uv=uv, uvscale=uvscale)
    def vquad(self, mat, a, b, y0, y1, look, uvscale=1.0, ya=None, yb=None):
        """senkrechte Wand zwischen lokalen Punkten a=(s,t), b=(s,t), von y0 bis y1; Normale zeigt zum lokalen Punkt look=(s,t).
        ya/yb: optionale Oberkanten je Ende (statt y1)."""
        y1a = y1 if ya is None else ya; y1b = y1 if yb is None else yb
        A0 = self.P(a[0], a[1], y0); B0 = self.P(b[0], b[1], y0); B1 = self.P(b[0], b[1], y1b); A1 = self.P(a[0], a[1], y1a)
        n = np.cross(B0 - A0, A1 - A0)
        lk = self.P(look[0], look[1], (y0 + y1) / 2)
        if np.dot(n, lk - (A0 + B1) / 2) < 0: A0, B0, B1, A1 = B0, A0, A1, B1
        L = np.linalg.norm(B0 - A0)
        u0 = 0.0
        uv = np.array([[u0, -A0[1]], [u0 + L, -B0[1]], [u0 + L, -B1[1]], [u0, -A1[1]]]) / uvscale
        return quad(mat, A0, B0, B1, A1, uv=uv)

def room(fr, mat_floor, mat_wall, mat_ceil, s0, s1, t0, t1, y0, y1, open_sides=(), uv_floor=1.2, uv_wall=1.0, uv_ceil=1.0, wall_mats=None):
    """Innenraum (Normalen nach innen). open_sides: Teilmenge von {'s0','s1','t0','t1'}; wall_mats: {'s0': mat, ...}"""
    out = []
    sc, tc = (s0 + s1) / 2, (t0 + t1) / 2
    if mat_floor: out.append(fr.hquad(mat_floor, s0, s1, t0, t1, y0, up=True, uvscale=uv_floor))
    if mat_ceil: out.append(fr.hquad(mat_ceil, s0, s1, t0, t1, y1, up=False, uvscale=uv_ceil))
    wm = wall_mats or {}
    for side, a, b in (('s0', (s0, t0), (s0, t1)), ('s1', (s1, t0), (s1, t1)), ('t0', (s0, t0), (s1, t0)), ('t1', (s0, t1), (s1, t1))):
        if side in open_sides: continue
        out.append(fr.vquad(wm.get(side, mat_wall), a, b, y0, y1, (sc, tc), uvscale=uv_wall))
    return [p for p in out if p is not None]

def ring_tube(mat, center, radius, r_tube, n=16, seg=6, axis_u=(1, 0, 0), axis_v=(0, 1, 0), a0=0.0, a1=2 * math.pi):
    c = np.asarray(center, float); u = np.asarray(axis_u, float); v = np.asarray(axis_v, float)
    a = np.linspace(a0, a1, n + 1)
    pts = c + radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
    return G.tube(mat, pts, r_tube, seg)

def cyl_between(mat, p0, p1, r, seg=8, caps=True):
    return G.tube(mat, np.array([p0, p1], float), r, seg, caps=caps)

def oval_column(mat, cx, cz, y0, y1, rx, rz, yaw=0.0, seg=20, uvscale=1.0, cap=False):
    """elliptische Stuetze (Mantel nach aussen)"""
    a = np.linspace(0, 2 * math.pi, seg + 1)
    ca, sa = math.cos(yaw), math.sin(yaw)
    ex = rx * np.cos(a); ez = rz * np.sin(a)
    x = cx + ex * ca - ez * sa; z = cz + ex * sa + ez * ca
    bot = np.c_[x, np.full(seg + 1, y0), z]; top = np.c_[x, np.full(seg + 1, y1), z]
    pos = np.concatenate([bot, top])
    nr = np.c_[np.cos(a) / rx * ca - np.sin(a) / rz * sa, np.zeros(seg + 1), np.cos(a) / rx * sa + np.sin(a) / rz * ca]
    nr /= np.linalg.norm(nr, axis=1, keepdims=True)
    per = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
    uv = np.concatenate([np.c_[per, np.zeros(seg + 1)], np.c_[per, -np.full(seg + 1, y1 - y0)]]) / uvscale
    idx = []
    n = seg + 1
    for i in range(seg): idx += [(i, n + i + 1, i + 1), (i, n + i, n + i + 1)]
    p = G.part(mat, pos, idx, np.concatenate([nr, nr]), uv)
    # Orientierung pruefen
    P = p['pos']; I = p['idx']; fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    cen = P[I].mean(1); out = cen[:, [0, 2]] - [cx, cz]
    if (fn[:, [0, 2]] * out).sum() < 0: p['idx'] = I[:, ::-1]
    if cap:
        # Deckel (Faecher, Normale nach oben)
        tc = np.concatenate([[[cx, y1, cz]], top[:-1]])
        ti = np.array([(0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)])
        fn2 = np.cross(tc[ti[:, 1]] - tc[ti[:, 0]], tc[ti[:, 2]] - tc[ti[:, 0]])
        if fn2[:, 1].sum() < 0: ti = ti[:, ::-1]
        c_ = G.part(mat, tc, ti, np.tile([0, 1.0, 0], (seg + 1, 1)), tc[:, [0, 2]] / uvscale)
        p = G.merge([p, c_])[0]
    return p

def fix_outward(p, center):
    """Dreiecke so drehen, dass Normalen von center weg zeigen (geschlossene konvexe Koerper)"""
    P, I = p['pos'], p['idx']
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    bad = (fn * (P[I].mean(1) - np.asarray(center))).sum(1) < 0
    I[bad] = I[bad][:, ::-1]
    p['nrm'] = G.vertex_normals(P, I)
    return p

def flat_normals(p):
    """Duplikat-Vertices je Dreieck mit Flaechennormale (fuer harte Kanten)"""
    P, I, U = p['pos'], p['idx'], p['uv']
    pos = P[I].reshape(-1, 3); uv = U[I].reshape(-1, 2)
    fn = np.cross(pos[1::3] - pos[0::3], pos[2::3] - pos[0::3]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    return G.part(p['mat'], pos, np.arange(len(pos)).reshape(-1, 3), np.repeat(fn, 3, 0), uv)

"""Geometrie-Grundbausteine fuer den Kroepcke-Center-Generator (reines numpy).

Lokales Fassaden-System: x = entlang der Fassade, y = Tiefe nach innen (0 = Vorderkante Pilaster = LoD2-Linie),
z = Hoehe (Spielhoehe, 0 = Kroepcke-Strassenniveau). (x, y, z) ist rechtshaendig.
Abbildung in den Plan (Blender-Welt: X Ost, Y Nord, Z oben):
  Gerade:  P = p0 + x*t + y*links(t)
  Bogen:   theta = a0 + x/r,  P = c + (r - y)*(cos, sin)   (CCW, Gebaeude innen)
Alle Flaechen zeigen nach aussen (Normalen aus Wicklung, CCW von aussen gesehen).
UVs werden vor dem Biegen im lokalen System per Box-Projektion bestimmt (stetig ueber Boegen).
"""
import math
from collections import defaultdict
import numpy as np
import mapbox_earcut as earcut

DXMAX = None            # Unterteilung entlang x (nur fuer Boegen gesetzt)
TILE = {}               # Material -> Texturkachel in m (Default 2.0)


def set_dxmax(v):
    global DXMAX
    DXMAX = v


def nseg(x0, x1):
    if DXMAX is None: return 1
    return max(1, int(math.ceil(abs(x1 - x0) / DXMAX - 1e-9)))


class Piece:
    """Dreiecke je Material (lokal)."""
    def __init__(self):
        self.d = defaultdict(lambda: [[], [], [], 0])   # mat -> [V, F, UV, nverts]

    def add(self, mat, V, F, UV):
        if len(F) == 0: return
        e = self.d[mat]
        F = np.asarray(F, np.int64) + e[3]
        e[0].append(np.asarray(V, float)); e[1].append(F); e[2].append(np.asarray(UV, float)); e[3] += len(V)

    def extend(self, other, fn=None):
        for m, (Vs, Fs, UVs, n) in other.d.items():
            if not Vs: continue
            V = np.concatenate(Vs); F = np.concatenate(Fs); UV = np.concatenate(UVs)
            if fn is not None: V = fn(V)
            self.add(m, V, F, UV)

    def arrays(self):
        out = {}
        for m, (Vs, Fs, UVs, n) in self.d.items():
            if not Vs: continue
            out[m] = (np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs))
        return out

    def tris(self):
        return sum(sum(len(f) for f in e[1]) for e in self.d.values())

    def transformed(self, fn):
        p = Piece(); p.extend(self, fn); return p


# ------------------------------------------------------------------ UV
def face_uv(V, F, mat, normals=None):
    """Box-Projektion je Dreieck (lokal). Rueckgabe (m,3,2)."""
    T = TILE.get(mat, 2.0)
    P = V[F]                                   # m,3,3
    if normals is None:
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    else:
        n = normals
    a = np.abs(n)
    ax = np.argmax(a, axis=1)
    uv = np.zeros(P.shape[:2] + (2,))
    # Normale ~x: (y, z); ~y: (x, z); ~z: (x, y)
    m0 = ax == 0; m1 = ax == 1; m2 = ax == 2
    uv[m0] = P[m0][:, :, [1, 2]]
    uv[m1] = P[m1][:, :, [0, 2]]
    uv[m2] = P[m2][:, :, [0, 1]]
    # Spiegelung je nach Richtung, damit Texturen nicht seitenverkehrt laufen (kosmetisch)
    s = np.where(n[np.arange(len(n)), ax] < 0, -1.0, 1.0)
    flip = (ax == 1)
    uv[flip, :, 0] *= -s[flip][:, None]
    flip0 = (ax == 0)
    uv[flip0, :, 0] *= s[flip0][:, None]
    return uv / T


def emit(piece, mat, V, F):
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    if len(F) == 0: return
    piece.add(mat, V, F, face_uv(V, F, mat))


# ------------------------------------------------------------------ Polygone
def poly_area(P):
    P = np.asarray(P, float)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def triangulate(rings):
    """rings: [outer, hole1, ...] (je (n,2)); Ergebnis: Punkte (N,2), Dreiecke (m,3) CCW."""
    pts = np.concatenate([np.asarray(r, float) for r in rings])
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = earcut.triangulate_float64(pts, ends).reshape(-1, 3).astype(np.int64)
    if len(idx) == 0: return pts, idx
    a = pts[idx]
    ar = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])
    idx[ar < 0] = idx[ar < 0][:, [0, 2, 1]]
    idx = idx[np.abs(ar) > 1e-12]
    return pts, idx


def resample_x(P, dmax):
    """Kanten eines geschlossenen (x,y)-Polygons unterteilen, wenn |dx| > dmax."""
    if dmax is None: return np.asarray(P, float)
    P = np.asarray(P, float); out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(1, int(math.ceil(abs(b[0] - a[0]) / dmax - 1e-9)))
        for k in range(n): out.append(a + (b - a) * k / n)
    return np.array(out)


# ------------------------------------------------------------------ Extrusionen
def ext_x(piece, mat, prof, x0, x1, caps=(True, True), skip=()):
    """Profil (y,z) entlang x extrudieren. skip: Indizes der Profilkanten ohne Flaeche."""
    prof = np.asarray(prof, float)
    if poly_area(prof) < 0:
        prof = prof[::-1]
        n0 = len(prof); skip = tuple((n0 - 2 - i) % n0 for i in skip)
    n = len(prof); nx = nseg(x0, x1); xs = np.linspace(x0, x1, nx + 1)
    V = []; F = []
    for i in range(n):
        if i in skip: continue
        a, b = prof[i], prof[(i + 1) % n]
        if np.allclose(a, b): continue
        base = len(V)
        for x in xs:
            V.append((x, a[0], a[1])); V.append((x, b[0], b[1]))
        for k in range(nx):
            A = base + 2 * k; Dv = A + 1; B_ = A + 2; C = A + 3
            F.append((A, C, B_)); F.append((A, Dv, C))
    emit(piece, mat, V, F)
    if caps[0] or caps[1]:
        pts, tri = triangulate([prof])
        if caps[0]:
            Vc = np.c_[np.full(len(pts), x0), pts]; emit(piece, mat, Vc, tri[:, [0, 2, 1]])
        if caps[1]:
            Vc = np.c_[np.full(len(pts), x1), pts]; emit(piece, mat, Vc, tri)


def ext_z(piece, mat, prof, z0, z1, caps=(True, True), skip=(), nz=1):
    """Profil (x,y) entlang z extrudieren."""
    prof = np.asarray(prof, float)
    flipped = poly_area(prof) < 0
    if flipped:
        prof = prof[::-1]; n0 = len(prof); skip = tuple((n0 - 2 - i) % n0 for i in skip)
    # Unterteilung fuer Boegen (Kanten in x), skip-Indizes mitfuehren
    if DXMAX is not None:
        newp = []; newskip = []
        for i in range(len(prof)):
            a, b = prof[i], prof[(i + 1) % len(prof)]
            m = max(1, int(math.ceil(abs(b[0] - a[0]) / DXMAX - 1e-9)))
            for k in range(m):
                if i in skip: newskip.append(len(newp))
                newp.append(a + (b - a) * k / m)
        prof = np.array(newp); skip = tuple(newskip)
    n = len(prof); zs = np.linspace(z0, z1, nz + 1)
    V = []; F = []
    for i in range(n):
        if i in skip: continue
        a, b = prof[i], prof[(i + 1) % n]
        if np.allclose(a, b): continue
        base = len(V)
        for z in zs:
            V.append((a[0], a[1], z)); V.append((b[0], b[1], z))
        for k in range(nz):
            A = base + 2 * k; Bv = A + 1; D = A + 2; C = A + 3
            F.append((A, Bv, C)); F.append((A, C, D))
    emit(piece, mat, V, F)
    if caps[0] or caps[1]:
        pts, tri = triangulate([prof])
        if caps[1]: emit(piece, mat, np.c_[pts, np.full(len(pts), z1)], tri)
        if caps[0]: emit(piece, mat, np.c_[pts, np.full(len(pts), z0)], tri[:, [0, 2, 1]])


def box(piece, mat, x0, x1, y0, y1, z0, z1, skip=''):
    """Quader; skip: Zeichenkette aus 'xXyYzZ' (x = -x-Seite, X = +x-Seite ...)."""
    prof = [(y0, z0), (y1, z0), (y1, z1), (y0, z1)]          # CCW in (y,z)
    # Kanten: 0 unten(z0, Normale -z), 1 hinten(y1, +y), 2 oben(z1, +z), 3 vorne(y0, -y)
    sk = []
    if 'z' in skip: sk.append(0)
    if 'Y' in skip: sk.append(1)
    if 'Z' in skip: sk.append(2)
    if 'y' in skip: sk.append(3)
    ext_x(piece, mat, prof, x0, x1, caps=('x' not in skip, 'X' not in skip), skip=tuple(sk))


def rect_y(piece, mat, x0, x1, z0, z1, y, out=True, nz=1):
    """Rechteck in Ebene y=const; out=True -> Normale -y (nach aussen)."""
    nx = nseg(x0, x1); xs = np.linspace(x0, x1, nx + 1); zs = np.linspace(z0, z1, nz + 1)
    V = [(x, y, z) for z in zs for x in xs]; F = []
    w = nx + 1
    for j in range(nz):
        for i in range(nx):
            a = j * w + i; b = a + 1; c = a + w + 1; d = a + w
            if out: F += [(a, b, c), (a, c, d)]
            else: F += [(a, c, b), (a, d, c)]
    emit(piece, mat, V, F)


def rect_z(piece, mat, x0, x1, y0, y1, z, up=True):
    nx = nseg(x0, x1); xs = np.linspace(x0, x1, nx + 1)
    V = [(x, y, z) for y in (y0, y1) for x in xs]; F = []
    w = nx + 1
    for i in range(nx):
        a = i; b = i + 1; c = w + i + 1; d = w + i
        if up: F += [(a, b, c), (a, c, d)]
        else: F += [(a, c, b), (a, d, c)]
    emit(piece, mat, V, F)


def rect_x(piece, mat, y0, y1, z0, z1, x, plus=True):
    V = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    F = [(0, 1, 2), (0, 2, 3)] if plus else [(0, 2, 1), (0, 3, 2)]
    emit(piece, mat, V, F)


def rod(piece, mat, p0, p1, r, n=6, caps=True):
    """Stab (Zylinder) zwischen zwei Punkten."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); d = p1 - p0; L = np.linalg.norm(d)
    if L < 1e-6: return
    d /= L
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u); v = np.cross(d, u)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.array([math.cos(t) * u + math.sin(t) * v for t in ang]) * r
    V = np.concatenate([p0 + ring, p1 + ring]); F = []
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, n + j), (i, n + j, n + i)]
    emit(piece, mat, V, F)
    if caps:
        c0 = len(V); V2 = np.concatenate([V, [p0, p1]]); F2 = []
        for i in range(n):
            j = (i + 1) % n
            F2 += [(c0, j, i), (c0 + 1, n + i, n + j)]
        emit(piece, mat, V2, F2)


def uvsphere(piece, mat, c, rx, ry, rz, nu=8, nv=5):
    c = np.asarray(c, float); V = []; F = []
    for j in range(nv + 1):
        t = math.pi * j / nv
        for i in range(nu):
            p = 2 * math.pi * i / nu
            V.append(c + (rx * math.sin(t) * math.cos(p), ry * math.sin(t) * math.sin(p), rz * math.cos(t)))
    for j in range(nv):
        for i in range(nu):
            a = j * nu + i; b = j * nu + (i + 1) % nu; cc = (j + 1) * nu + (i + 1) % nu; d = (j + 1) * nu + i
            if j > 0: F.append((a, d, b))
            if j < nv - 1: F.append((b, d, cc))
    # obere/untere Reihe sind Pole (mehrfache Punkte) -> Dreiecke oben ok
    emit(piece, mat, V, F)


def lathe(piece, mat, c, prof_rz, n=10):
    """Rotationskoerper um die z-Achse durch c; prof_rz: [(r,z)...] von unten nach oben."""
    c = np.asarray(c, float); V = []; F = []
    m = len(prof_rz)
    for (r, z) in prof_rz:
        for i in range(n):
            t = 2 * math.pi * i / n
            V.append((c[0] + r * math.cos(t), c[1] + r * math.sin(t), c[2] + z))
    for j in range(m - 1):
        for i in range(n):
            a = j * n + i; b = j * n + (i + 1) % n; cc = (j + 1) * n + (i + 1) % n; d = (j + 1) * n + i
            F += [(a, b, cc), (a, cc, d)]
    # Deckel
    V.append((c[0], c[1], c[2] + prof_rz[0][1])); V.append((c[0], c[1], c[2] + prof_rz[-1][1]))
    b0 = len(V) - 2; b1 = len(V) - 1
    for i in range(n):
        j = (i + 1) % n
        if prof_rz[0][0] > 0: F.append((b0, j, i))
        if prof_rz[-1][0] > 0: F.append((b1, (m - 1) * n + i, (m - 1) * n + j))
    emit(piece, mat, V, F)


# ------------------------------------------------------------------ Abbildungen
def map_line(p0, t):
    p0 = np.asarray(p0, float); t = np.asarray(t, float); l = np.array([-t[1], t[0]])
    def fn(V):
        V = np.asarray(V, float)
        out = np.empty_like(V)
        out[:, 0] = p0[0] + V[:, 0] * t[0] + V[:, 1] * l[0]
        out[:, 1] = p0[1] + V[:, 0] * t[1] + V[:, 1] * l[1]
        out[:, 2] = V[:, 2]
        return out
    return fn


def map_arc(c, r, a0, x_off=0.0):
    c = np.asarray(c, float)
    def fn(V):
        V = np.asarray(V, float)
        th = a0 + (V[:, 0] + x_off) / r
        rr = r - V[:, 1]
        out = np.empty_like(V)
        out[:, 0] = c[0] + rr * np.cos(th); out[:, 1] = c[1] + rr * np.sin(th); out[:, 2] = V[:, 2]
        return out
    return fn


def seg_map(seg, x_off=0.0):
    if seg['type'] == 'line':
        return map_line(np.asarray(seg['p0']) + np.asarray(seg['t']) * x_off, seg['t'])
    return map_arc(seg['c'], seg['r'], seg['a0'], x_off)


def seg_point(seg, x, y=0.0):
    return seg_map(seg)(np.array([[x, y, 0.0]]))[0, :2]


def seg_tangent(seg, x):
    if seg['type'] == 'line': return np.asarray(seg['t'], float)
    th = seg['a0'] + x / seg['r']
    return np.array([-math.sin(th), math.cos(th)])

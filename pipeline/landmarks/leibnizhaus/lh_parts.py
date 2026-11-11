"""Bauteile: Relief-Hoehenfelder (Ranken, Engelskoepfe, Figurenszenen, Kandelaber, Medaillons, Beschlagwerk),
Fenster (Kreuzstock), Gesimse, Saeulen, Obelisken, Voluten, Figuren. Alles in lokalen Fassadenkoordinaten (u, y, w)."""
import math
import numpy as np
from scipy.spatial import cKDTree
import lh_geom as G

TAU = 2 * math.pi


# ============================================================================ Relief-Hoehenfelder
class Relief:
    """Sammelt Striche (Polylinien mit Radius/Hoehe) und Ellipsen-Blobs; rendert H(y, x) in Metern."""

    def __init__(self, w, h, res=40, depth=0.06):
        self.w, self.h, self.res, self.depth = w, h, res, depth
        self.strokes = []      # (pts (n,2), r, hz)
        self.blobs = []        # (cx, cy, rx, ry, ang, hz)
        self.base = None

    def stroke(self, pts, r, hz=None):
        pts = np.asarray(pts, float)
        if len(pts) < 2: return
        hz = self.depth * 0.6 if hz is None else hz
        # gleichmaessig nachabtasten
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        n = max(2, int(d[-1] / (0.4 / self.res)) + 1)
        t = np.linspace(0, d[-1], n)
        P = np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])]
        rr = np.interp(t, d, np.broadcast_to(np.asarray(r, float), len(pts)))
        self.strokes.append((P, rr, hz))

    def blob(self, cx, cy, rx, ry, hz=None, ang=0.0):
        self.blobs.append((cx, cy, rx, ry, ang, self.depth * 0.7 if hz is None else hz))

    def spiral(self, cx, cy, r0, turns=1.6, a0=0.0, sgn=1, r=0.012, hz=None, n=60, shrink=0.85):
        t = np.linspace(0, 1, n)
        rad = r0 * (1 - shrink * t)
        a = a0 + sgn * t * turns * TAU
        self.stroke(np.c_[cx + rad * np.cos(a), cy + rad * np.sin(a)], np.linspace(r, r * 0.6, n), hz)

    def leaf(self, x0, y0, ang, L, wd, hz=None):
        a = np.array([math.cos(ang), math.sin(ang)]); p = np.array([-a[1], a[0]])
        t = np.linspace(0, 1, 9)
        mid = np.array([x0, y0]) + np.outer(t, a) * L
        self.blob(*(np.array([x0, y0]) + a * L / 2), L / 2, wd / 2, hz, ang)
        self.stroke(mid, wd * 0.12, (hz or self.depth * 0.5) * 1.15)

    def frame(self, m=0.02, hz=None):
        w, h = self.w, self.h
        self.stroke([[m, m], [w - m, m], [w - m, h - m], [m, h - m], [m, m]], m * 0.9, hz or self.depth * 0.45)

    def render(self):
        w, h, res = self.w, self.h, self.res
        nx = max(4, int(round(w * res)) + 1); ny = max(4, int(round(h * res)) + 1)
        X, Y = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, h, ny))
        Q = np.c_[X.ravel(), Y.ravel()]
        H = np.zeros(len(Q))
        if self.strokes:
            P = np.concatenate([s[0] for s in self.strokes])
            R = np.concatenate([s[1] for s in self.strokes])
            Z = np.concatenate([np.full(len(s[0]), s[2]) for s in self.strokes])
            tree = cKDTree(P)
            rmax = R.max()
            d, i = tree.query(Q, k=6, distance_upper_bound=rmax * 1.01)
            ok = np.isfinite(d)
            ii = np.where(ok, i, 0)
            rr = R[ii]; zz = Z[ii]
            prof = np.where(ok & (d < rr), np.sqrt(np.clip(1 - (np.where(ok, d, 0) / rr) ** 2, 0, 1)), 0) * zz
            H = np.maximum(H, prof.max(1))
        for (cx, cy, rx, ry, ang, hz) in self.blobs:
            ca, sa = math.cos(ang), math.sin(ang)
            dx = Q[:, 0] - cx; dy = Q[:, 1] - cy
            a = (dx * ca + dy * sa) / rx; b = (-dx * sa + dy * ca) / ry
            q = a * a + b * b
            H = np.maximum(H, np.sqrt(np.clip(1 - q, 0, 1)) * hz)
        H = H.reshape(ny, nx)
        if self.base is not None: H = np.maximum(H, self.base)
        # Rand auf 0 ziehen (Schuerze schliesst an Wand)
        H[0, :] = H[-1, :] = H[:, 0] = H[:, -1] = np.minimum(H[0, 0], 0.0)
        return H


def rel_scroll(w, h, res=40, depth=0.05, seed=0, frame=True):
    """Rankenfries: Wellenranke mit Spiralen und Blaettern."""
    rng = np.random.default_rng(seed)
    R = Relief(w, h, res, depth)
    if frame: R.frame(min(w, h) * 0.06, depth * 0.5)
    m = h * 0.18
    xs = np.linspace(m, w - m, 80)
    per = max(1, round((w - 2 * m) / (h * 1.6)))
    ys = h / 2 + (h / 2 - m) * 0.55 * np.sin((xs - m) / (w - 2 * m) * per * TAU)
    R.stroke(np.c_[xs, ys], h * 0.035, depth * 0.55)
    for k in range(per * 2):
        x = m + (k + 0.5) * (w - 2 * m) / (per * 2)
        sgn = 1 if k % 2 == 0 else -1
        cy = h / 2 - sgn * (h / 2 - m) * 0.25
        R.spiral(x, cy, h * 0.22, 1.3, a0=(0 if sgn > 0 else math.pi), sgn=sgn, r=h * 0.03, hz=depth * 0.7)
        R.leaf(x - h * 0.15, cy + sgn * h * 0.15, math.pi / 2 + sgn * 0.6 + rng.normal(0, 0.2), h * 0.28, h * 0.12, depth * 0.5)
    return R.render()


def rel_cherub(w, h, res=40, depth=0.08, seed=0):
    """Gefluegelter Engelskopf (Kopf + Fluegel aus Federn)."""
    R = Relief(w, h, res, depth)
    cx, cy = w / 2, h * 0.5
    hr = min(w * 0.18, h * 0.32)
    R.blob(cx, cy, hr, hr * 1.08, depth)
    R.blob(cx, cy + hr * 0.75, hr * 0.9, hr * 0.45, depth * 0.85)          # Locken
    for sgn in (-1, 1):
        for k in range(5):
            a = math.pi / 2 - (0.35 + k * 0.28)
            if sgn < 0: a = math.pi - a
            L = w * 0.42 * (1 - k * 0.1)
            x0, y0 = cx + sgn * hr * 0.7, cy + hr * 0.1
            R.leaf(x0, y0, a, L, hr * 0.45, depth * (0.55 - k * 0.05))
    return R.render()


def _figure(R, x, y0, fh, rng, depth, robe=None):
    """Stehende/bewegte Figur (Relief) mit Fuessen bei y0, Hoehe fh."""
    lean = rng.normal(0, 0.12)
    hipx = x; hipy = y0 + fh * 0.47
    shx = x + lean * fh * 0.25; shy = y0 + fh * 0.8
    robe = rng.random() < 0.5 if robe is None else robe
    R.blob(x + lean * fh * 0.12, y0 + fh * 0.64, fh * 0.12, fh * 0.2, depth * 0.85, ang=lean)
    R.blob(shx + lean * 0.05, y0 + fh * 0.92, fh * 0.065, fh * 0.075, depth * 0.8)
    if robe:
        R.blob(x, y0 + fh * 0.25, fh * 0.17, fh * 0.26, depth * 0.7)
    else:
        for s in (-1, 1):
            k = rng.uniform(0.05, 0.22)
            R.stroke([[hipx + s * fh * 0.05, hipy], [x + s * fh * k * 0.6, y0 + fh * 0.24], [x + s * fh * k, y0 + fh * 0.02]], fh * 0.045, depth * 0.6)
    for s in (-1, 1):
        a = rng.uniform(-1.2, 1.4)
        ex = shx + s * fh * 0.13 + s * math.cos(a) * fh * 0.16; ey = shy - 0.02 * fh + math.sin(a) * fh * 0.16
        hx = ex + s * math.cos(a + rng.uniform(-0.8, 0.8)) * fh * 0.15; hy = ey + math.sin(a) * fh * 0.14
        R.stroke([[shx + s * fh * 0.1, shy], [ex, ey], [hx, hy]], fh * 0.035, depth * 0.62)


def rel_scene(w, h, res=40, depth=0.08, seed=0, n=None, animal=None, frame=True):
    """Figurenrelief (Bibelszene/Allegorie): Figuren, ggf. Loewe, Baum, Architektur."""
    rng = np.random.default_rng(seed)
    R = Relief(w, h, res, depth)
    if frame: R.frame(min(w, h) * 0.05, depth * 0.55)
    m = min(w, h) * 0.1
    n = n or max(1, int(round(w / (h * 0.45))))
    if animal is None: animal = rng.random() < 0.4
    xs = np.linspace(m + w * 0.12, w - m - w * 0.12, n) if n > 1 else [w / 2]
    for k, x in enumerate(xs):
        if animal and k == n - 1 and n > 1:
            # Loewe
            by = m + h * 0.25
            R.blob(x, by, h * 0.22, h * 0.12, depth * 0.75, ang=0.15)
            R.blob(x - h * 0.22, by + h * 0.1, h * 0.12, h * 0.13, depth * 0.85)
            R.blob(x - h * 0.27, by + h * 0.12, h * 0.06, h * 0.06, depth)
            for dx in (-0.14, -0.05, 0.08, 0.16):
                R.stroke([[x + dx * h, by], [x + dx * h + 0.02 * h, m + 0.01]], h * 0.03, depth * 0.55)
            R.stroke([[x + h * 0.2, by + 0.02], [x + h * 0.32, by + h * 0.12], [x + h * 0.3, by + h * 0.22]], h * 0.018, depth * 0.5)
        else:
            _figure(R, x + rng.normal(0, w * 0.02), m, (h - 2 * m) * rng.uniform(0.82, 0.95), rng, depth)
    if rng.random() < 0.6:
        tx = rng.choice([m + w * 0.05, w - m - w * 0.05])
        R.stroke([[tx, m], [tx, h * 0.6]], h * 0.025, depth * 0.35)
        for _ in range(5):
            R.blob(tx + rng.normal(0, h * 0.08), h * 0.68 + rng.normal(0, h * 0.06), h * 0.1, h * 0.08, depth * 0.35)
    H = R.render()
    H = np.maximum(H, depth * 0.08)            # Reliefgrund leicht erhaben
    H[0, :] = H[-1, :] = H[:, 0] = H[:, -1] = 0
    return H


def rel_candelabra(w, h, res=40, depth=0.05, seed=0):
    """senkrechtes Kandelaber-/Groteskenornament (Vasen, Blattwerk, Masken)."""
    rng = np.random.default_rng(seed)
    R = Relief(w, h, res, depth)
    R.frame(w * 0.06, depth * 0.45)
    cx = w / 2
    y = h * 0.05
    R.stroke([[cx, h * 0.05], [cx, h * 0.95]], w * 0.05, depth * 0.5)
    while y < h * 0.9:
        kind = rng.integers(0, 3)
        s = w * rng.uniform(0.28, 0.38)
        if kind == 0:     # Vase
            R.blob(cx, y + s * 0.6, s * 0.75, s * 0.55, depth)
            R.blob(cx, y + s * 1.2, s * 0.35, s * 0.15, depth * 0.8)
        elif kind == 1:   # Maske
            R.blob(cx, y + s * 0.7, s * 0.55, s * 0.65, depth)
            R.blob(cx, y + s * 0.4, s * 0.3, s * 0.15, depth * 0.85)
        else:             # Blattpaar + Schnecken
            for sg in (-1, 1):
                R.spiral(cx + sg * s * 0.55, y + s * 0.7, s * 0.4, 1.2, a0=math.pi / 2, sgn=-sg, r=w * 0.04, hz=depth * 0.7)
        y += s * 1.6
    return R.render()


def rel_medallions(w, h, res=40, depth=0.04, seed=0):
    """Fries mit Medaillons (Rosetten) und verbindenden Ranken."""
    R = Relief(w, h, res, depth)
    R.frame(h * 0.08, depth * 0.5)
    n = max(1, int(round(w / (h * 2.2))))
    for k in range(n):
        x = (k + 0.5) * w / n
        R.stroke(np.c_[x + h * 0.32 * np.cos(np.linspace(0, TAU, 40)), h / 2 + h * 0.32 * np.sin(np.linspace(0, TAU, 40))], h * 0.05, depth * 0.7)
        for j in range(8):
            a = j * TAU / 8
            R.leaf(x, h / 2, a, h * 0.25, h * 0.11, depth * 0.6)
        if k < n - 1:
            xs = np.linspace(x + h * 0.4, x + w / n - h * 0.4, 20)
            R.stroke(np.c_[xs, h / 2 + h * 0.15 * np.sin((xs - xs[0]) / (xs[-1] - xs[0] + 1e-9) * TAU)], h * 0.035, depth * 0.5)
    return R.render()


def rel_strap(w, h, res=40, depth=0.035, seed=0):
    """Beschlagwerk-Fuellung (Brustungsfelder): Rahmen, Raute, Rollwerk an den Ecken."""
    R = Relief(w, h, res, depth)
    R.frame(min(w, h) * 0.07, depth * 0.6)
    cx, cy = w / 2, h / 2
    R.stroke([[cx, h * 0.15], [w * 0.85, cy], [cx, h * 0.85], [w * 0.15, cy], [cx, h * 0.15]], min(w, h) * 0.035, depth * 0.55)
    R.blob(cx, cy, min(w, h) * 0.14, min(w, h) * 0.14, depth)
    for sx in (0.18, 0.82):
        for sy in (0.2, 0.8):
            R.spiral(w * sx, h * sy, min(w, h) * 0.08, 1.0, sgn=1 if sx < 0.5 else -1, r=min(w, h) * 0.025, hz=depth * 0.6)
    return R.render()


def rel_cartouche(w, h, res=40, depth=0.06, seed=0):
    R = Relief(w, h, res, depth)
    t = np.linspace(0, TAU, 60)
    R.stroke(np.c_[w / 2 + w * 0.32 * np.cos(t), h / 2 + h * 0.36 * np.sin(t)], min(w, h) * 0.06, depth * 0.8)
    R.blob(w / 2, h / 2, w * 0.26, h * 0.3, depth * 0.45)
    for sg in (-1, 1):
        R.spiral(w / 2 + sg * w * 0.38, h * 0.2, min(w, h) * 0.12, 1.3, sgn=sg, r=min(w, h) * 0.035, hz=depth * 0.8)
        R.spiral(w / 2 + sg * w * 0.38, h * 0.8, min(w, h) * 0.12, 1.3, sgn=-sg, r=min(w, h) * 0.035, hz=depth * 0.8)
    return R.render()


def relief_panel(mb, fr, mat, H, u0, y0, w, h, z0=0.0, res=None):
    G.relief(mb, fr, mat, H, u0, y0, w, h, z0=z0, uvs=1.0)


# ============================================================================ Profile
PROF = {
    # (aussen, oben) von unten-Wand -> aussen -> oben-Wand
    'cornice': [(0, 0), (0.03, 0.0), (0.03, 0.04), (0.07, 0.07), (0.09, 0.11), (0.2, 0.13), (0.2, 0.19), (0.22, 0.2), (0.24, 0.25),
                (0.24, 0.27), (0.0, 0.27)],
    'cornice_s': [(0, 0), (0.02, 0.0), (0.02, 0.03), (0.06, 0.06), (0.14, 0.08), (0.14, 0.13), (0.16, 0.15), (0.16, 0.17), (0.0, 0.17)],
    'band': [(0, 0), (0.035, 0.0), (0.05, 0.02), (0.05, 0.06), (0.035, 0.08), (0.0, 0.08)],
    'sill': [(0, 0), (0.05, 0.0), (0.07, 0.03), (0.07, 0.06), (0.0, 0.07)],
    'plinth': [(0, 0), (0.12, 0.0), (0.12, 0.38), (0.08, 0.44), (0.0, 0.46)],
    'architrave': [(0, 0), (0.02, 0.0), (0.02, 0.05), (0.035, 0.06), (0.035, 0.11), (0.05, 0.12), (0.05, 0.15), (0.0, 0.15)],
}


def cornice(mb, fr, mat, u0, u1, y, prof='cornice', scale=1.0, ret=(True, True)):
    """Gesims entlang u von u0..u1 bei Hoehe y (Wand bei w=0); Enden mit Rueckkroepfung (gerade Kappe)."""
    p = np.asarray(PROF[prof], float) * scale
    path = np.array([[u0, y, 0.0], [u1, y, 0.0]])
    G.sweep(mb, fr, mat, p, path, up=(0, 1, 0), caps=True)


def cornice_path(mb, fr, mat, path_uw, y, prof='cornice', scale=1.0, closed=False):
    """Gesims entlang eines Pfads in der Ebene (u, w) bei Hoehe y (Laufrichtung: Aussenseite rechts)."""
    p = np.asarray(PROF[prof], float) * scale
    path = np.c_[np.asarray(path_uw)[:, 0], np.full(len(path_uw), y), np.asarray(path_uw)[:, 1]]
    G.sweep(mb, fr, mat, p, path, up=(0, 1, 0), caps=not closed, closed=closed)


# ============================================================================ Fenster
def window_cross(mb, fr, u0, u1, y0, y1, reveal=0.24, frame_w=0.11, transom=True, mullion=True, glass='leaded', stone='sand_fine',
                 wood='wood_dark', sill=True, frame=True):
    """Kreuzstockfenster in Wandloch (u0..u1, y0..y1) der Wand w=0. Loch selbst baut die Wand."""
    if frame:
        fw = frame_w
        prof = [(fw, 0.0), (fw, 0.025), (fw * 0.75, 0.04), (0.02, 0.045), (0.0, 0.03), (0.0, -0.03)]
        path = np.array([[u0, y0, 0], [u1, y0, 0], [u1, y1, 0], [u0, y1, 0]])
        G.sweep(mb, fr, stone, prof, path, up=(0, 0, 1), closed=True)
    if sill:
        G.box(mb, fr, stone, u0 - 0.05, u1 + 0.05, y0 - 0.075, y0 + 0.0, -0.02, 0.07, bevel=0.012)
    W = u1 - u0; Hh = y1 - y0
    um = (u0 + u1) / 2; yt = y0 + Hh * 0.68
    mw = 0.085
    if mullion:
        G.box(mb, fr, stone, um - mw / 2, um + mw / 2, y0, y1, -reveal * 0.55, 0.0, bevel=0.012)
    if transom:
        G.box(mb, fr, stone, u0, u1, yt - mw / 2, yt + mw / 2, -reveal * 0.55, -0.005, bevel=0.012)
    # Fluegel: Holzrahmen + Bleiglas
    zg = -reveal * 0.62
    cols = [(u0, um - mw / 2), (um + mw / 2, u1)] if mullion else [(u0, u1)]
    rows = [(y0, yt - mw / 2), (yt + mw / 2, y1)] if transom else [(y0, y1)]
    for (a, b) in cols:
        for (c, d) in rows:
            sash(mb, fr, a, b, c, d, zg, wood, glass)


def sash(mb, fr, a, b, c, d, z, wood='wood_dark', glass='leaded', fw=0.05, depth=0.05, bars=0):
    """Fensterfluegel: Holzrahmen (Profil) + Glasflaeche bei z (lokal w)."""
    path = np.array([[a, c, z], [b, c, z], [b, d, z], [a, d, z]])
    prof = [(fw, -depth), (fw, 0.0), (fw * 0.4, 0.012), (0.0, 0.012), (0.0, -depth)]
    G.sweep(mb, fr, wood, prof, path, up=(0, 0, 1), closed=True)
    P = np.array([[a, c, z - 0.02], [b, c, z - 0.02], [b, d, z - 0.02], [a, d, z - 0.02]])
    N = np.tile([0, 0, 1.0], (4, 1))
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]) if glass == 'leaded' else P[:, :2]
    mb.add_local(fr, glass, P, N, UV, np.array([[0, 1, 2], [0, 2, 3]]))
    for k in range(bars):
        x = a + (b - a) * (k + 1) / (bars + 1)
        G.box(mb, fr, wood, x - 0.012, x + 0.012, c, d, z - 0.02, z + 0.005)


# ============================================================================ Saeulen, Obelisken, Kugeln
def column(mb, fr, mat, u, w, y0, y1, r, segs=14, band=True, capital='ionic'):
    """Rundsaeule (Basis, Schaft mit Entasis, Schaftring, Kapitell), Achse lokal (u, *, w)."""
    H = y1 - y0
    hb = min(0.12, H * 0.08); hc = min(0.18, H * 0.11)
    prof = [(0, 0), (r * 1.45, 0), (r * 1.45, hb * 0.35), (r * 1.3, hb * 0.5), (r * 1.38, hb * 0.7), (r * 1.15, hb), (r, hb * 1.15)]
    ys = np.linspace(hb * 1.15, H - hc, 8)
    for k, yy in enumerate(ys[1:]):
        t = (yy - hb) / (H - hc - hb)
        rr = r * (1 - 0.13 * t ** 1.5)
        if band and abs(t - 0.38) < 0.08:
            prof += [(rr * 1.18, yy - 0.04), (rr * 1.18, yy + 0.04)]
        prof.append((rr, yy))
    rt = r * 0.87
    if capital == 'ionic':
        prof += [(rt * 1.1, H - hc + 0.02), (rt * 1.1, H - hc + 0.04), (rt, H - hc + 0.05), (rt * 1.35, H - hc * 0.45), (r * 1.55, H - hc * 0.3),
                 (r * 1.55, H), (0, H)]
    else:
        prof += [(rt * 1.1, H - hc), (r * 1.5, H - 0.04), (r * 1.6, H - 0.04), (r * 1.6, H), (0, H)]
    sub = G.Frame(fr.p([u, y0, w]), fr.u, fr.v)
    G.lathe(mb, sub, mat, prof, segs=segs, smooth=True, uvs=1.0)
    if capital == 'ionic':
        for s in (-1, 1):
            G.tube(mb, sub, mat, np.c_[s * r * 1.35 + 0.0 * np.zeros(2), [H - hc * 0.35] * 2, [-r * 1.2, r * 1.2]], r * 0.28, 8)


def obelisk(mb, fr, mat, u, w, y0, h, b=0.25, ball=True, ball_mat=None):
    """Obelisk-Fiale auf Sockel: Sockelwuerfel, Pyramidenstumpf, Kugel."""
    s = b
    G.box(mb, fr, mat, u - s / 2, u + s / 2, y0, y0 + s * 0.8, w - s / 2, w + s / 2, bevel=0.015)
    G.box(mb, fr, mat, u - s * 0.6, u + s * 0.6, y0 + s * 0.8, y0 + s * 0.9, w - s * 0.6, w + s * 0.6, bevel=0.01)
    hh = h - s * 0.9 - (s * 0.35 if ball else 0)
    a = s * 0.36; c = s * 0.08
    base = y0 + s * 0.9
    P = np.array([[u - a, base, w - a], [u + a, base, w - a], [u + a, base, w + a], [u - a, base, w + a],
                  [u - c, base + hh, w - c], [u + c, base + hh, w - c], [u + c, base + hh, w + c], [u - c, base + hh, w + c]])
    Q = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])
    I = G.quads_to_tris(Q)
    cen = P.mean(0)
    I = G.orient(P, I, P[I].mean(1) - cen)
    PP, NN, II = G.flat_tris(P, I)
    mb.add_local(fr, mat, PP, NN, G.box_uv(PP, NN), II)
    if ball:
        G.sphere(mb, fr, ball_mat or mat, (u, base + hh + s * 0.15, w), s * 0.17, 10, 6)


# ============================================================================ Voluten
def _bez(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def volute_shape(L, Hh):
    """S-Volute: grosse Schnecke aussen unten, konkave Flanke, kleine Schnecke innen oben. Rueckgabe (Polygon (n,2) CCW,
    Kreise [(cx,cy,r)], Flankenkurve (m,2)). x: 0 = Stufenwand (innen) .. L = aussen."""
    from shapely.geometry import Point, Polygon as SP
    from shapely.ops import unary_union
    r1 = min(0.30 * Hh, 0.40 * L); c1 = np.array([L - r1, r1])
    r2 = max(0.42 * r1, 0.07); c2 = np.array([r2 * 1.05, Hh - r2])
    p0 = np.array([c1[0] - r1 * 0.2, 2 * r1 * 0.97]); p3 = np.array([c2[0] + r2 * 0.85, c2[1] - r2 * 0.2])
    p1 = np.array([L * 0.38, 2 * r1 * 1.02]); p2 = np.array([c2[0] + r2 * 1.1, Hh * 0.52 + r1 * 0.5])
    curve = _bez(p0, p1, p2, p3)
    body = SP(np.r_[[[0.0, 0.0], [c1[0], 0.0]], curve, [[0.0, p3[1]]]]).buffer(0)
    shp = unary_union([body, Point(*c1).buffer(r1, 10), Point(*c2).buffer(r2, 8), SP([(0, 0), (0.02, 0), (0.02, Hh), (0, Hh)])])
    shp = shp.intersection(SP([(0, 0), (L, 0), (L, Hh), (0, Hh)])).simplify(0.004)
    if shp.geom_type != 'Polygon': shp = max(shp.geoms, key=lambda g: g.area)
    poly = np.asarray(shp.exterior.coords)[:-1]
    if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0: poly = poly[::-1]
    return poly, [(c1[0], c1[1], r1), (c2[0], c2[1], r2)], curve


def volute(mb, fr, mat, u_in, u_out, y0, Hh, thick=0.22, w0=0.0, deco_mat=None):
    """Volute zwischen u_in (Stufenwand) und u_out (aussen) ab Hoehe y0, Hoehe Hh; Platte von w0-thick..w0."""
    L = abs(u_out - u_in); sg = 1 if u_out > u_in else -1
    if L < 0.15 or Hh < 0.2: return
    poly, circ, curve = volute_shape(L, Hh)
    poly2 = np.c_[u_in + sg * poly[:, 0], y0 + poly[:, 1]]
    if sg < 0: poly2 = poly2[::-1]
    G.extrude(mb, fr, mat, poly2, w0 - thick, w0, uvs=1.0)
    dm = deco_mat or mat
    sc = max(0.6, min(1.4, L / 1.1))
    for k, (cx, cy, rr) in enumerate(circ):
        t = np.linspace(0, 1, 56)
        turns = 2.2 if k == 0 else 1.8
        a0 = (-math.pi / 2 if k == 0 else math.pi / 2)
        sgn = -sg if k == 0 else sg
        rad = rr * 0.88 * (1 - 0.8 * t); a = a0 + sgn * t * turns * TAU
        path = np.c_[u_in + sg * cx + rad * np.cos(a), y0 + cy + rad * np.sin(a), np.full(56, w0 + 0.02)]
        G.tube(mb, fr, dm, path, np.linspace(0.045, 0.02, 56) * sc, 6)
        G.sphere(mb, fr, dm, (u_in + sg * cx, y0 + cy, w0 + 0.03), rr * 0.16, 8, 5)
    cpath = np.c_[u_in + sg * curve[:, 0], y0 + curve[:, 1] - 0.035 * sc, np.full(len(curve), w0 + 0.015)]
    G.tube(mb, fr, dm, cpath, 0.03 * sc, 6)


# ============================================================================ Figur (Statue)
def statue(mb, fr, mat, u, w, y0, h, pose=0, seed=0, spear=True):
    """Stehende Figur (Ritter/Krieger) aus Rotationskoerpern und Rohren, Hoehe h, Blick +w."""
    s = h / 1.8
    sub = G.Frame(fr.p([u, y0, w]), fr.u, fr.v)
    # Plinthe
    G.box(mb, sub, mat, -0.22 * s, 0.22 * s, 0, 0.08 * s, -0.2 * s, 0.2 * s, bevel=0.01)
    # Beine
    for sx in (-1, 1):
        G.tube(mb, sub, mat, np.array([[sx * 0.08 * s, 0.08 * s, 0.0], [sx * 0.09 * s, 0.5 * s, 0.01 * s], [sx * 0.09 * s, 0.9 * s, 0]]),
               np.array([0.06, 0.07, 0.085]) * s, 8)
    # Rumpf (Harnisch) + Rock
    G.lathe(mb, sub, mat, [(0, 0.82 * s), (0.2 * s, 0.82 * s), (0.17 * s, 1.0 * s), (0.19 * s, 1.25 * s), (0.22 * s, 1.42 * s),
                           (0.12 * s, 1.5 * s), (0.06 * s, 1.52 * s), (0, 1.53 * s)], segs=10)
    G.sphere(mb, sub, mat, (0, 1.64 * s, 0.01 * s), 0.1 * s, 10, 6, scale=(1, 1.15, 1))
    # Helm
    G.lathe(mb, sub, mat, [(0, 1.66 * s), (0.115 * s, 1.68 * s), (0.1 * s, 1.76 * s), (0.02 * s, 1.8 * s), (0, 1.8 * s)], segs=10)
    # Arme
    G.tube(mb, sub, mat, np.array([[0.2 * s, 1.4 * s, 0], [0.3 * s, 1.18 * s, 0.05 * s], [0.32 * s, 1.0 * s, 0.15 * s]]), 0.055 * s, 7)
    G.tube(mb, sub, mat, np.array([[-0.2 * s, 1.4 * s, 0], [-0.3 * s, 1.55 * s, 0.05 * s], [-0.28 * s, 1.75 * s, 0.05 * s]]), 0.055 * s, 7)
    if spear:
        G.cyl(mb, sub, mat, (-0.28 * s, 0.1 * s, 0.05 * s), (-0.28 * s, 2.15 * s, 0.05 * s), 0.018 * s, 0.018 * s, 6)
        G.cyl(mb, sub, mat, (-0.28 * s, 2.15 * s, 0.05 * s), (-0.28 * s, 2.35 * s, 0.05 * s), 0.04 * s, 0.0, 4)
    # Schild
    G.cyl(mb, sub, mat, (0.34 * s, 1.0 * s, 0.12 * s), (0.34 * s, 1.0 * s, 0.17 * s), 0.17 * s, 0.17 * s, 12)


def small_figure(mb, fr, mat, u, w, y0, h, seed=0):
    """vereinfachte Sitz-/Standfigur (Putto/Allegorie) als Vollplastik."""
    rng = np.random.default_rng(seed)
    s = h
    sub = G.Frame(fr.p([u, y0, w]), fr.u, fr.v)
    G.lathe(mb, sub, mat, [(0, 0), (0.2 * s, 0), (0.22 * s, 0.25 * s), (0.17 * s, 0.55 * s), (0.2 * s, 0.68 * s), (0.08 * s, 0.75 * s), (0, 0.76 * s)], segs=9)
    G.sphere(mb, sub, mat, (0, 0.86 * s, 0.02 * s), 0.11 * s, 9, 5)
    for sx in (-1, 1):
        a = rng.uniform(-0.6, 1.2)
        G.tube(mb, sub, mat, np.array([[sx * 0.17 * s, 0.66 * s, 0], [sx * 0.27 * s, 0.66 * s + math.sin(a) * 0.15 * s, 0.06 * s],
                                       [sx * 0.3 * s, 0.66 * s + math.sin(a) * 0.3 * s, 0.1 * s]]), 0.045 * s, 6)


# ============================================================================ Schrift (3D-Buchstaben)
def text3d(mb, fr, mat, text, x, y, size, depth=0.02, z0=0.0, weight='bold', family='DejaVu Sans', align='left', stretch=1.0):
    """3D-Schrift (TextPath -> Polygone mit Loechern -> Extrusion) in der lokalen xy-Ebene ab z0 nach +z."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from shapely.geometry import Polygon as SP
    tp = TextPath((0, 0), text, size=1.0, prop=FontProperties(family=family, weight=weight))
    polys = [p for p in tp.to_polygons() if len(p) >= 3]
    if not polys: return 0.0
    allp = np.concatenate(polys)
    w = (allp[:, 0].max() - allp[:, 0].min()) * size * stretch
    ox = x - (w / 2 if align == 'center' else (w if align == 'right' else 0.0)) - allp[:, 0].min() * size * stretch
    sp = [SP(p).buffer(0) for p in polys]
    depthc = [sum(1 for j, q in enumerate(sp) if j != i and q.contains(sp[i].representative_point())) for i in range(len(sp))]
    for i, p in enumerate(polys):
        if depthc[i] % 2 == 1: continue
        holes = [polys[j] for j in range(len(polys)) if depthc[j] == depthc[i] + 1 and sp[i].contains(sp[j].representative_point())]
        T = lambda q: np.c_[ox + np.asarray(q)[:, 0] * size * stretch, y + np.asarray(q)[:, 1] * size]
        G.extrude(mb, fr, mat, T(p), z0, z0 + depth, holes=[T(h) for h in holes], uvs=1.0)
    return w


def win_modern(mb, fr, u0, u1, y0, y1, stone='sand_white', frame='shopframe', glass='glass', reveal=0.22, transom=True, mullion=True,
               surround=0.14, ears=True, sill=True):
    """Gruenderzeit-Fenster: Steinlaibung + Faschen (Ohren), Sohlbank, zweifluegelig mit Oberlicht (Rahmen dunkel)."""
    G.reveal(mb, fr, stone, G.rect(u0, y0, u1, y1), 0.0, -reveal)
    s = surround
    prof = [(s, 0.0), (s, 0.03), (s * 0.7, 0.045), (0.02, 0.05), (0.0, 0.03), (0.0, -0.02)]
    G.sweep(mb, fr, stone, prof, np.array([[u0, y0, 0], [u1, y0, 0], [u1, y1, 0], [u0, y1, 0]]), up=(0, 0, 1), closed=True)
    if ears:
        for xx in (u0 - s - 0.06, u1 + s - 0.0):
            G.box(mb, fr, stone, xx, xx + 0.06, y1 - 0.12, y1 + s, 0.0, 0.045, bevel=0.008)
    if sill:
        G.box(mb, fr, stone, u0 - s - 0.05, u1 + s + 0.05, y0 - 0.1, y0 + 0.0, -0.02, 0.09, bevel=0.012)
    z = -reveal * 0.55
    yt = y0 + (y1 - y0) * 0.74 if transom else y1
    um = (u0 + u1) / 2
    fw = 0.06
    # Blendrahmen
    G.sweep(mb, fr, frame, [(fw, -0.06), (fw, 0.0), (0.0, 0.0), (0.0, -0.06)], np.array([[u0, y0, z], [u1, y0, z], [u1, y1, z], [u0, y1, z]]), up=(0, 0, 1), closed=True)
    if transom: G.box(mb, fr, frame, u0, u1, yt - 0.035, yt + 0.035, z - 0.06, z)
    if mullion: G.box(mb, fr, frame, um - 0.03, um + 0.03, y0, yt, z - 0.06, z)
    P = np.array([[u0, y0, z - 0.04], [u1, y0, z - 0.04], [u1, y1, z - 0.04], [u0, y1, z - 0.04]])
    mb.add_local(fr, glass, P, np.tile([0, 0, 1.0], (4, 1)), P[:, :2], np.array([[0, 1, 2], [0, 2, 3]]))
    if transom:
        for k in range(1, 4):
            xx = u0 + (u1 - u0) * k / 4
            G.box(mb, fr, frame, xx - 0.012, xx + 0.012, yt, y1, z - 0.05, z - 0.02)

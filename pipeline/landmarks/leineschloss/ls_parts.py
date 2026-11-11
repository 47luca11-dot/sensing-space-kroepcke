"""Leineschloss -- Bauteile: korinthisches Kapitell, kannelierte Saeule/Halbsaeule, attische Basis, toskanische Saeule,
Baluster, Vase, Fenster/Tuer-Einheiten, Gaube, Mansarddach, Konsolen, Reliefs (Hoehenbilder).
Teile, die oft vorkommen, werden lokal (Ursprung = Fusspunkt-Mitte, h nach oben) erzeugt und spaeter instanziert/transformiert."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
import lslib as L
from lslib import Builder, nrm, UP

S_ = 'LS_Sandstein'


# ============================================================== Saeulen
def attic_base(b, D, mat=S_, segs=40, plinth=True, half=False):
    """Attische Basis fuer Schaftdurchmesser D (unten): Plinthe 1.0*D*1.25 quadratisch, Torus, Trochilus, Torus."""
    r = D / 2
    if plinth:
        w = r * 1.3
        T = L.bevel_box_tris(-w, w, -w if not half else 0, w, 0, 0.16 * D, c=0.012)
        b.add(mat, T, closed=True)
    h0 = 0.16 * D if plinth else 0.0
    prof = [(r * 1.22, h0), (r * 1.27, h0 + 0.03 * D), (r * 1.28, h0 + 0.07 * D), (r * 1.24, h0 + 0.11 * D), (r * 1.14, h0 + 0.13 * D),
            (r * 1.12, h0 + 0.14 * D), (r * 1.04, h0 + 0.17 * D), (r * 1.02, h0 + 0.2 * D), (r * 1.06, h0 + 0.22 * D),
            (r * 1.10, h0 + 0.245 * D), (r * 1.10, h0 + 0.27 * D), (r * 1.05, h0 + 0.3 * D), (r * 1.01, h0 + 0.31 * D), (r * 1.0, h0 + 0.33 * D)]
    a0, a1 = (0, math.pi) if half else (0, 2 * math.pi)
    T = L.lathe_tris(prof, segs, (0, 0, 0), a0=a0, a1=a1, cap_bottom=False, cap_top=False)
    b.add(mat, T, expect=None)
    return h0 + 0.33 * D


def fluted_shaft(b, r0, r1, H, h0=0.0, nfl=24, mat=S_, rows=16, half=False, entasis=0.012):
    """Kannelierter Schaft (Hohlkehlen mit Stegen), Radius r0 unten -> r1 oben, Entasis.
    half=True: Halbsaeule (nur 0..pi, Rueckseite offen -> an Wand)."""
    k = 6                           # Punkte je Kannelur (Steg + Hohlkehle)
    nf = nfl // 2 if half else nfl
    n = nf * k + (1 if half else 0)
    ang = np.arange(n) * (2 * math.pi / (nfl * k))
    phase = (np.arange(n) % k) / k
    # Profil einer Kannelur: Steg (phase 0..0.18) flach, dann Halbkreis-Kehle
    flute = np.where(phase < 0.18, 0.0, np.sin(math.pi * (phase - 0.18) / 0.82))
    zs = np.linspace(0, 1, rows + 1)
    # Kanneluren laufen oben/unten rund aus
    endfade = np.clip(np.minimum(zs / 0.035, (1 - zs) / 0.035), 0, 1) ** 0.5
    rad = r0 + (r1 - r0) * zs + entasis * r0 * np.sin(math.pi * np.clip(zs, 0, 1)) * (zs > 0.0)
    depth = 0.07 * r0
    R = rad[None, :] - depth * flute[:, None] * endfade[None, :]
    G = np.stack([R * np.cos(ang)[:, None], R * np.sin(ang)[:, None], np.broadcast_to(h0 + zs * H, R.shape)], axis=-1)
    T = L.grid_tris(G, closed_u=not half)
    b.add(mat, T, expect=None)
    return T


def plain_shaft(b, r0, r1, H, h0=0.0, segs=32, mat=S_, half=False):
    a0, a1 = (0, math.pi) if half else (0, 2 * math.pi)
    T = L.lathe_tris([(r0, h0), (r0 + (r1 - r0) * 0.33 + 0.01 * r0, h0 + H * 0.33), (r1, h0 + H)], segs, a0=a0, a1=a1)
    b.add(mat, T)


def _leaf(b, mat, ang, r_in, h0, hgt, width, lean, curl, lobes=5, nu=9, nv=10, thick=0.012):
    """Akanthusblatt als duenne geschlossene Schale auf Radius r_in, Mitte bei Winkel ang.
    Ueber die Hoehe: Breite nimmt mit gezacktem Rand ab, Blatt neigt nach aussen (lean) und rollt oben ueber (curl)."""
    t = np.linspace(0, 1, nv)
    x = np.linspace(-1, 1, nu)
    # Umriss: unten schmal, oben breit, Spitze schmal; tiefe Saegezahn-Lappen
    grow = np.clip(t / 0.62, 0, 1)
    wprof = width * (0.38 + 0.62 * np.sin(grow * math.pi / 2)) * (1 - 0.75 * (np.clip(t - 0.8, 0, 1) / 0.2) ** 1.3)
    fr = (t * lobes) % 1.0
    lob = 1 - 0.32 * fr ** 2 * (t > 0.12)
    W_ = (wprof * lob)[:, None] * x[None, :]
    # Neigung nach aussen + Ueberrollen der Spitze (Kreisbogen)
    out = lean * t ** 2
    hh = h0 + hgt * np.minimum(t, 0.8) / 0.8 * 0.86
    ph = np.clip((t - 0.8) / 0.2, 0, 1) * math.radians(120)
    out = out + curl * np.sin(ph)
    hh = hh + curl * np.sin(ph) * 0.4 - curl * (1 - np.cos(ph)) * 0.9 + np.where(t > 0.8, hgt * 0.14 * np.sin(ph / 2), 0)
    ridge = 0.18 * width * (1 - np.abs(x)) ** 2             # Mittelrippe
    edge = 0.10 * width * np.abs(x) ** 3                     # Blattraender nach vorn gewellt
    dent = 0.05 * width * np.cos(x * math.pi * 3) ** 2       # Nebenrippen
    R = r_in + out[:, None] + (ridge + edge)[None, :] - dent[None, :] * 0.4
    ca, sa = math.cos(ang), math.sin(ang)
    tx, ty = -sa, ca
    Pf = np.stack([R * ca + W_ * tx, R * sa + W_ * ty, np.broadcast_to(hh[:, None], R.shape)], -1)
    Pb = np.stack([(R - thick) * ca + W_ * 0.96 * tx, (R - thick) * sa + W_ * 0.96 * ty, np.broadcast_to(hh[:, None] - thick * 0.3, R.shape)], -1)
    front = L.grid_tris(Pf.transpose(1, 0, 2))
    back = L.grid_tris(Pb.transpose(1, 0, 2))
    # Raender schliessen
    rim = []
    for P1, P2 in ((Pf[:, 0], Pb[:, 0]), (Pf[:, -1], Pb[:, -1]), (Pf[-1], Pb[-1])):
        Q = np.stack([P1[:-1], P1[1:], P2[1:], P2[:-1]], axis=1)
        rim.append(L.quads(Q))
    T = np.concatenate([front, back] + rim)
    c = np.array([ (r_in - 0.05) * ca, (r_in - 0.05) * sa, h0 + hgt * 0.3])
    # Orientierung: Vorderflaeche nach aussen (radial), Rueckseite nach innen
    nf = L.tri_normals(front); radial = np.c_[front.mean(1)[:, :2], np.zeros(len(front))]
    if np.sum(np.einsum('ij,ij->i', nf, radial) < 0) > len(front) / 2: front = front[:, ::-1]
    nb = L.tri_normals(back); radb = np.c_[back.mean(1)[:, :2], np.zeros(len(back))]
    if np.sum(np.einsum('ij,ij->i', nb, radb) > 0) > len(back) / 2: back = back[:, ::-1]
    rimT = np.concatenate(rim)
    m = rimT.mean(1); cen = np.c_[m[:, 0] * 0 + c[0], m[:, 1] * 0 + c[1], m[:, 2]]
    rimT = L.fix_orient(rimT, m - cen)
    b.add(mat, np.concatenate([front, back, rimT]))


def _spiral_tube(b, mat, origin, X, Y, r_start, turns, r_end, prof_r, nseg=36, ring=6, z_up=0.0):
    """Volute: Rohr mit Radius prof_r entlang einer Spirale in der Ebene (X,Y) um origin."""
    th = np.linspace(0, turns * 2 * math.pi, nseg)
    rr = r_start * (r_end / r_start) ** (th / th[-1])
    P = origin[None, :] + (rr * np.cos(th))[:, None] * X[None, :] + (rr * np.sin(th))[:, None] * Y[None, :]
    Z = nrm(np.cross(X, Y))
    pr = prof_r * np.linspace(1.0, 0.55, nseg)
    tan = np.gradient(P, axis=0); tan = nrm(tan)
    ang = np.linspace(0, 2 * math.pi, ring, endpoint=False)
    n1 = nrm(np.cross(tan, Z[None, :])); n2 = np.cross(tan, n1)
    G = P[:, None, :] + pr[:, None, None] * (np.cos(ang)[None, :, None] * n1[:, None, :] + np.sin(ang)[None, :, None] * n2[:, None, :])
    T = L.grid_tris(G, closed_v=True)
    # nach aussen (weg von der Rohrachse)
    m = T.mean(1)
    idx = np.clip(np.round(np.linspace(0, nseg - 1, len(T))).astype(int), 0, nseg - 1)
    # naechster Achspunkt
    d = np.linalg.norm(m[:, None, :] - P[None, :, :], axis=2); k = d.argmin(1)
    T = L.fix_orient(T, m - P[k])
    caps = [L.tri_poly(G[0]), L.tri_poly(G[-1])]
    caps[0] = L.fix_orient(caps[0], -tan[0]); caps[1] = L.fix_orient(caps[1], tan[-1])
    b.add(mat, np.concatenate([T] + caps))
    return P


def corinthian_capital(D, mat=S_, segs=40, detail=1.0, half=False):
    """Korinthisches Kapitell fuer oberen Schaftdurchmesser D; Fuss h=0 (Oberkante Astragal), Hoehe 1.17*D.
    Rueckgabe: Builder (lokal), Hoehe."""
    b = Builder()
    r = D / 2; H = 1.17 * D
    nl = 8
    sel = (lambda a: True) if not half else (lambda a: -0.05 <= (a % (2 * math.pi)) <= math.pi + 0.05)
    a0, a1 = (0, math.pi) if half else (0, 2 * math.pi)
    # Astragal (Rundstab unter dem Kapitell, h<0)
    prof = [(r * 1.0, -0.09 * D), (r * 1.06, -0.075 * D), (r * 1.08, -0.05 * D), (r * 1.06, -0.025 * D), (r * 1.0, -0.012 * D), (r * 1.0, 0.0)]
    b.add(mat, L.lathe_tris(prof, segs, a0=a0, a1=a1))
    # Kelch (Kalathos)
    hb = 0.86 * H
    prof = [(r * 0.98, 0.0), (r * 0.99, hb * 0.5), (r * 1.06, hb * 0.85), (r * 1.13, hb * 0.97), (r * 1.16, hb), (r * 1.1, hb + 0.01 * D)]
    b.add(mat, L.lathe_tris(prof, segs, a0=a0, a1=a1, cap_top=not half))
    nu = max(int(9 * detail), 5); nv = max(int(13 * detail), 7)
    # untere Blattreihe (8), obere Reihe (8, versetzt)
    for i in range(nl):
        ang = 2 * math.pi * i / nl
        if sel(ang): _leaf(b, mat, ang, r * 0.98, 0.0, H * 0.4, r * 0.4, r * 0.2, r * 0.13, nu=nu, nv=nv, thick=0.008)
        ang2 = ang + math.pi / nl
        if sel(ang2): _leaf(b, mat, ang2, r * 1.0, H * 0.25, H * 0.42, r * 0.4, r * 0.24, r * 0.15, nu=nu, nv=nv, thick=0.008)
    # Abakus: konkav geschwungenes Quadrat mit abgefasten Ecken
    ha0 = hb; ha1 = H
    A = r * 1.42      # halbe Diagonal-Ausdehnung an der Ecke (Fase)
    pts = []
    for q in range(4):
        cang = q * math.pi / 2 + math.pi / 4      # Ecken diagonal
        # Ecke -> naechste Ecke mit Einwoelbung
        e0 = np.array([math.cos(cang), math.sin(cang)]) * A
        e1 = np.array([math.cos(cang + math.pi / 2), math.sin(cang + math.pi / 2)]) * A
        mid = (e0 + e1) / 2
        inward = -nrm(np.r_[mid, 0])[:2]
        for j in range(8):
            t = j / 8
            p = e0 + (e1 - e0) * t + inward * (math.sin(math.pi * t) * r * 0.16)
            # Fase an der Ecke: erste/letzte Punkte etwas zurueckgezogen
            if j == 0: p = p - nrm(np.r_[e0, 0])[:2] * r * 0.05
            pts.append(p)
    pts = np.array(pts)
    if half:
        pts = pts[(pts[:, 1] >= -1e-6) | True]
    aprof = [(0.0, 0.0), (0.02 * D, 0.0), (0.04 * D, 0.03 * D), (0.035 * D, 0.06 * D), (0.05 * D, 0.08 * D), (0.05 * D, ha1 - ha0)]
    path = np.c_[pts, np.full(len(pts), ha0)]
    aprof2 = [(x - 0.05 * D, y) for x, y in aprof]
    T = L.sweep_tris(path[::-1] if L.signed_area(pts) < 0 else path, aprof2, closed=True)
    top = L.tri_poly(np.c_[pts + nrm(np.c_[pts, np.zeros(len(pts))])[:, :2] * 0.0, np.full(len(pts), ha1)], [0, 0, 1])
    bot = L.tri_poly(np.c_[pts, np.full(len(pts), ha0)], [0, 0, -1])
    AB = np.concatenate([T, top, bot])
    if half:
        AB = AB[AB.mean(1)[:, 1] > -0.02 * D]
    b.add(mat, AB)
    # Voluten (Ecken) + Helices (Mitte der Seiten)
    for q in range(4):
        cang = q * math.pi / 2 + math.pi / 4
        if not sel(cang): continue
        dirc = np.array([math.cos(cang), math.sin(cang), 0.0])
        side = np.array([-math.sin(cang), math.cos(cang), 0.0])
        o = dirc * r * 1.28 + UP * (ha0 - 0.1 * D)
        _spiral_tube(b, mat, o, -UP, dirc * 0.6 + UP * 0.0 + 0 * side, 0.16 * D, 1.7, 0.03 * D, 0.04 * D, nseg=int(30 * detail) + 6)
        # Stengel (Caulicoli) von der oberen Blattreihe zur Volute
        st = np.array([dirc * r * 1.05 + UP * H * 0.5, dirc * r * 1.18 + UP * H * 0.68, o + UP * 0.1 * D - dirc * 0.02 * D])
        b.add(mat, L.sweep_tris(st, [(0.025 * D * math.cos(a), 0.025 * D * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)], bv=side, caps=True), closed=True)
    for q in range(4):
        cang = q * math.pi / 2
        if not sel(cang): continue
        dirc = np.array([math.cos(cang), math.sin(cang), 0.0])
        side = np.array([-math.sin(cang), math.cos(cang), 0.0])
        for sgn in (-1, 1):
            o = dirc * r * 1.08 + side * sgn * r * 0.2 + UP * (ha0 - 0.08 * D)
            _spiral_tube(b, mat, o, -UP, side * sgn * 0.7, 0.07 * D, 1.3, 0.015 * D, 0.018 * D, nseg=int(16 * detail) + 6)
        # Abakusblume (Fleuron)
        fl = [(0.0, 0.0), (0.06 * D, 0.01 * D), (0.07 * D, 0.04 * D), (0.04 * D, 0.07 * D), (0.0, 0.075 * D)]
        T = L.lathe_tris(fl, 10)
        R3 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)    # Achse nach aussen drehen
        Mrot = np.eye(4); Mrot[:3, :3] = np.c_[side, UP, dirc] @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Mrot[:3, :3] = np.c_[side, -np.cross(side, dirc), dirc]
        Mrot[:3, 3] = dirc * (r * 1.42 * 0.71 - r * 0.16 + 0.05 * D) + UP * (ha0 + 0.5 * (ha1 - ha0))
        b.add(mat, L.xform(T, Mrot), closed=True)
    return b, H


def tuscan_column(b, D, H, mat=S_, segs=28, h0=0.0, cx=0.0, cy=0.0):
    """Toskanische Saeule (glatt): Plinthe, Torus, Schaft mit Verjuengung, Echinus, Abakus."""
    r = D / 2
    T = L.bevel_box_tris(cx - r * 1.2, cx + r * 1.2, cy - r * 1.2, cy + r * 1.2, h0, h0 + 0.12 * D, c=0.01)
    b.add(mat, T, closed=True)
    prof = [(r * 1.15, h0 + 0.12 * D), (r * 1.18, h0 + 0.18 * D), (r * 1.12, h0 + 0.24 * D), (r * 1.02, h0 + 0.27 * D), (r, h0 + 0.3 * D),
            (r * 0.99, h0 + H * 0.4), (r * 0.86, h0 + H - 0.42 * D), (r * 0.92, h0 + H - 0.4 * D), (r * 0.92, h0 + H - 0.36 * D), (r * 0.86, h0 + H - 0.34 * D),
            (r * 0.86, h0 + H - 0.3 * D), (r * 1.02, h0 + H - 0.22 * D), (r * 1.12, h0 + H - 0.16 * D)]
    b.add(mat, L.lathe_tris(prof, segs, (cx, cy, 0)))
    T = L.bevel_box_tris(cx - r * 1.2, cx + r * 1.2, cy - r * 1.2, cy + r * 1.2, h0 + H - 0.16 * D, h0 + H, c=0.01)
    b.add(mat, T, closed=True)


def baluster(D=0.2, H=0.7, mat=S_, segs=12):
    b = Builder(); r = D / 2
    prof = [(r * 0.95, 0.0), (r * 0.95, 0.08 * H), (r * 0.6, 0.12 * H), (r * 0.55, 0.16 * H), (r * 0.9, 0.3 * H), (r * 1.0, 0.42 * H), (r * 0.8, 0.58 * H),
            (r * 0.42, 0.72 * H), (r * 0.38, 0.8 * H), (r * 0.6, 0.84 * H), (r * 0.6, 0.88 * H), (r * 0.9, 0.92 * H), (r * 0.9, H)]
    b.add(mat, L.lathe_tris(prof, segs, cap_top=True, cap_bottom=True), closed=True)
    return b


def urn(D=0.5, H=0.9, mat=S_, segs=16):
    """Deckelvase (Balustradenaufsatz)."""
    b = Builder(); r = D / 2
    prof = [(r * 0.55, 0.0), (r * 0.55, 0.08 * H), (r * 0.35, 0.12 * H), (r * 0.3, 0.18 * H), (r * 0.7, 0.26 * H), (r * 1.0, 0.42 * H), (r * 0.95, 0.56 * H),
            (r * 0.7, 0.66 * H), (r * 0.75, 0.7 * H), (r * 0.8, 0.72 * H), (r * 0.55, 0.78 * H), (r * 0.4, 0.86 * H), (r * 0.25, 0.92 * H), (r * 0.12, 0.97 * H), (0.0, H)]
    b.add(mat, L.lathe_tris(prof, segs, cap_bottom=True), closed=True)
    T = L.bevel_box_tris(-r * 0.62, r * 0.62, -r * 0.62, r * 0.62, -0.1 * H, 0.0, c=0.008)
    b.add(mat, T, closed=True)
    return b


def modillion(w=0.16, d=0.34, h=0.2, mat=S_):
    """Konsole (Sparrenkopf mit Volute) unter der Hängeplatte: lokales x = nach vorn, y = quer, z = oben (Oberkante = 0)."""
    b = Builder()
    n = 10
    t = np.linspace(0, 1, n)
    # Seitenprofil: oben gerade, vorne Volute, unten S-Schwung
    top = np.c_[t * d, np.zeros(n)]
    xs = np.linspace(d, 0, n)
    bot = np.c_[xs, -h * (0.35 + 0.65 * (1 - xs / d) ** 1.5) + 0.06 * h * np.sin(xs / d * 2 * math.pi)]
    prof = np.vstack([top, [[d, -h * 0.35]], bot[1:]])
    poly = np.c_[prof[:, 0], prof[:, 1]]
    # Extrusion quer (y)
    P0 = np.c_[poly[:, 0], np.full(len(poly), -w / 2), poly[:, 1]]
    P1 = np.c_[poly[:, 0], np.full(len(poly), w / 2), poly[:, 1]]
    nP = len(poly)
    Q = np.stack([P0, np.roll(P0, -1, 0), np.roll(P1, -1, 0), P1], axis=1)
    T = np.concatenate([L.quads(Q), L.tri_poly(P0, [0, -1, 0]), L.tri_poly(P1, [0, 1, 0])])
    b.add(mat, T, closed=True)
    return b


def console_s(w=0.18, d=0.22, h=0.42, mat=S_):
    """Senkrechte Volutenkonsole (unter Fensterverdachung): lokales x = aus der Wand, y = quer, z oben (Oberkante 0)."""
    b = Builder()
    n = 12
    z = np.linspace(0, -h, n)
    x = d * (0.25 + 0.75 * (1 - np.abs(z / h)) ** 2)
    poly = np.vstack([np.c_[np.zeros(n), z][::-1], np.c_[x, z]])
    poly = np.vstack([np.c_[np.zeros(1), [-h]], np.c_[x[::-1], z[::-1]], [[0, 0]]])
    P0 = np.c_[poly[:, 0], np.full(len(poly), -w / 2), poly[:, 1]]
    P1 = np.c_[poly[:, 0], np.full(len(poly), w / 2), poly[:, 1]]
    Q = np.stack([P0, np.roll(P0, -1, 0), np.roll(P1, -1, 0), P1], axis=1)
    T = np.concatenate([L.quads(Q), L.tri_poly(P0, [0, -1, 0]), L.tri_poly(P1, [0, 1, 0])])
    b.add(mat, T, closed=True)
    return b


# ============================================================== Profile (x = nach aussen, y = nach oben), Nullpunkt = Wandflaeche
def prof_cornice(proj=0.9, H=1.0):
    """Hauptgesims: Kehle, Platte (Hängeplatte), Karnies (Sima). proj = Ausladung, H = Hoehe."""
    p = [(0.0, 0.0), (0.06, 0.0), (0.06, 0.06), (0.12, 0.10), (0.18, 0.18), (0.22, 0.22), (0.24, 0.30), (proj * 0.72, 0.36), (proj * 0.74, 0.38),
         (proj * 0.74, 0.62), (proj * 0.78, 0.64), (proj * 0.82, 0.70), (proj * 0.95, 0.80), (proj, 0.92), (proj, 1.0), (0.0, 1.0)]
    return [(x, y * H) for x, y in p]


def prof_string(proj=0.12, H=0.3):
    p = [(0.0, 0.0), (0.03, 0.0), (0.05, 0.15), (proj * 0.7, 0.3), (proj, 0.55), (proj, 0.8), (proj * 0.8, 0.9), (proj * 0.8, 1.0), (0.0, 1.0)]
    return [(x, y * H) for x, y in p]


def prof_sill(proj=0.1, H=0.12):
    return [(0.0, 0.0), (proj * 0.7, 0.0), (proj, H * 0.45), (proj, H), (0.0, H)]


def prof_plinth(proj=0.12, H=0.6):
    p = [(0.0, 0.0), (proj, 0.0), (proj, 0.75), (proj * 0.6, 0.85), (proj * 0.3, 0.95), (0.0, 1.0)]
    return [(x, y * H) for x, y in p]


def prof_architrave(w=0.18, proj=0.06):
    """Fenstergewaende: x = von der Oeffnung weg, y = aus der Wand."""
    return [(0.0, 0.0), (0.0, proj * 0.6), (0.03, proj * 0.6), (0.04, proj), (w * 0.7, proj), (w * 0.75, proj * 0.75), (w, proj * 0.6), (w, 0.0)]


# ============================================================== Fenster / Tueren
FR = 'LS_Rahmen'; GL = 'LS_Fensterglas'


def window_unit(b, W, s0, s1, h0, h1, shape='rect', depth=0.22, bars=(2, 3), frame_mat=FR, glass=GL, reveal_mat=S_, arch_segs=12,
                grp='main', fw=0.075, bw=0.035, rise=None, sill=True, sill_mat=S_, transom=None):
    """Oeffnung in Wand W (Loch muss in der Wandflaeche ausgespart sein): Laibung, Blendrahmen, Fluegel mit Sprossen, Scheibe.
    shape: 'rect' | 'arch' (Rundbogen, Kaempfer = h1 - Breite/2) | 'seg' (Stichbogen mit 'rise')."""
    w = s1 - s0
    if shape == 'rect': hole = L.rect2d(s0, s1, h0, h1)
    elif shape == 'arch': hole = L.arch2d(s0, s1, h0, h1 - w / 2, arch_segs)
    else: hole = L.arch2d(s0, s1, h0, h1 - rise, arch_segs, rise=rise)
    L.reveal(b, reveal_mat, W, hole, 0.0, -depth, grp)
    # Scheibe (dunkles Glas) knapp hinter dem Rahmen
    dg = -depth - 0.01
    b.add(glass, L.tri_poly(W.poly(hole, dg), W.n), 'main')
    # Blendrahmen als Ring (Profil in der Ebene, nach innen ins Loch)
    ring = hole
    A = W.poly(ring, -depth + 0.06)
    # innerer Ring: Loch um fw verkleinert
    inner = shrink2d(ring, fw)
    B = W.poly(inner, -depth + 0.06)
    A2 = W.poly(ring, dg + 0.002); B2 = W.poly(inner, dg + 0.002)
    n = len(ring)
    Q = np.stack([A, np.roll(A, -1, 0), np.roll(B, -1, 0), B], axis=1)        # Vorderseite Rahmen
    Qi = np.stack([B, np.roll(B, -1, 0), np.roll(B2, -1, 0), B2], axis=1)     # innere Kante
    T = L.quads(np.concatenate([Q]))
    b.add(frame_mat, L.fix_orient(T, W.n), grp)
    Ti = L.quads(Qi)
    cen = W.P(*(ring.mean(0)), dg)
    b.add(frame_mat, _inward(Ti, cen), grp)
    # Sprossen (Kreuzstock + Sprossen) als flache Leisten vor der Scheibe
    nx, ny = bars
    sx0, sx1 = inner[:, 0].min(), inner[:, 0].max(); hy0, hy1 = inner[:, 1].min(), inner[:, 1].max()
    if shape == 'rect':
        hb = hy1
    else:
        hb = (h1 - w / 2) if shape == 'arch' else (h1 - rise)
    d0 = dg + 0.002; d1 = -depth + 0.045
    for i in range(1, nx):
        s = sx0 + (sx1 - sx0) * i / nx
        top = hy1 if shape == 'rect' else _arch_top(inner, s)
        _bar(b, frame_mat, W, s - bw / 2, s + bw / 2, hy0, top, d0, d1, grp)
    rows = ny
    hs = [hy0 + (hb - hy0) * j / rows for j in range(1, rows)]
    if transom is not None: hs = [transom]
    for hh in hs:
        _bar(b, frame_mat, W, sx0, sx1, hh - bw / 2, hh + bw / 2, d0, d1, grp)
    if shape != 'rect':
        _bar(b, frame_mat, W, sx0, sx1, hb - bw / 2, hb + bw / 2, d0, d1, grp)
        # radiale Sprossen im Bogenfeld
        c = ((s0 + s1) / 2, hb)
        rr = (sx1 - sx0) / 2
        for a in (math.pi / 4, math.pi / 2, 3 * math.pi / 4):
            p0 = np.array([c[0], hb]); p1 = np.array([c[0] + rr * math.cos(a), hb + (rr if shape == 'arch' else rise) * math.sin(a)])
            _bar_line(b, frame_mat, W, p0, p1, bw, d0, d1, grp)
    if sill:
        # Sohlbank (Stein), steht vor die Wand
        sw = 0.08
        T = L.box_tris(0, 1, 0, 1, 0, 1)
        P = W.P(np.array([s0 - sw, s1 + sw]), np.array([h0 - 0.12, h0]), np.array([-0.05, 0.09]))
        _wbox(b, sill_mat, W, s0 - sw, s1 + sw, h0 - 0.13, h0 + 0.005, -0.06, 0.09, grp)
    return hole


def _inward(T, cen):
    n = L.tri_normals(T); m = T.mean(1)
    flip = np.einsum('ij,ij->i', n, cen - m) < 0
    T = T.copy(); T[flip] = T[flip][:, ::-1]
    return T


def _arch_top(ring, s):
    # hoechster Punkt des Rings bei s (lineare Interpolation am oberen Rand)
    pts = ring[ring[:, 1] > ring[:, 1].min() + 1e-6]
    order = np.argsort(pts[:, 0])
    return float(np.interp(s, pts[order, 0], pts[order, 1]))


def shrink2d(ring, d):
    """Polygon (gegen Uhrzeigersinn) um d nach innen versetzen (einfacher Kantenversatz mit Gehrung)."""
    P = np.asarray(ring, float)
    if L.signed_area(P) < 0: P = P[::-1]; rev = True
    else: rev = False
    n = len(P)
    e_in = nrm(np.c_[P - np.roll(P, 1, 0), np.zeros(n)])[:, :2]
    e_out = nrm(np.c_[np.roll(P, -1, 0) - P, np.zeros(n)])[:, :2]
    n_in = np.c_[-e_in[:, 1], e_in[:, 0]]; n_out = np.c_[-e_out[:, 1], e_out[:, 0]]
    m = n_in + n_out; ml = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.maximum(ml, 1e-9)
    c = np.maximum(np.einsum('ij,ij->i', m, n_out), 0.3)
    Q = P + m * (d / c)[:, None]
    return Q[::-1] if rev else Q


def _wbox(b, mat, W, s0, s1, h0, h1, d0, d1, grp='main'):
    """Quader in Wandkoordinaten (s, h, d)."""
    c = W.P((s0 + s1) / 2, (h0 + h1) / 2, (d0 + d1) / 2)
    T = L.obox_tris(c, W.t, UP, W.n, (s1 - s0) / 2, (h1 - h0) / 2, (d1 - d0) / 2)
    # obox: Achsen X=t, Y=UP, Z=n -> Rechtssystem? t x UP = -n bei n = (t1,-t0)... Orientierung per Volumen sichern
    b.add(mat, T, grp, closed=True)


def _bar(b, mat, W, s0, s1, h0, h1, d0, d1, grp):
    if s1 - s0 < 1e-4 or h1 - h0 < 1e-4: return
    _wbox(b, mat, W, s0, s1, h0, h1, d0, d1, grp)


def _bar_line(b, mat, W, p0, p1, bw, d0, d1, grp):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    dd = p1 - p0; Lg = np.hypot(*dd)
    if Lg < 1e-4: return
    tt = dd / Lg; nn = np.array([-tt[1], tt[0]])
    q = [p0 + nn * bw / 2, p1 + nn * bw / 2, p1 - nn * bw / 2, p0 - nn * bw / 2]
    q = np.array(q)
    A = W.poly(q, d1); B = W.poly(q, d0)
    T = np.concatenate([L.quads(np.stack([A, np.roll(A, -1, 0), np.roll(B, -1, 0), B], 1)), L.tri_poly(A, W.n), L.tri_poly(B, -W.n)])
    b.add(mat, T, grp, closed=True)


def door_unit(b, W, s0, s1, h0, h1, depth=0.35, mat='LS_Holz', panels=True, glass_top=None, grp='main', reveal_mat=S_, shape='rect'):
    """Tuer: Laibung, Holztuer zweifluegelig mit Fuellungen (Kassetten als Relief)."""
    w = s1 - s0
    hole = L.rect2d(s0, s1, h0, h1) if shape == 'rect' else L.arch2d(s0, s1, h0, h1 - w / 2, 12)
    L.reveal(b, reveal_mat, W, hole, 0.0, -depth, grp, skip_bottom=True)
    dd = -depth
    b.add(mat, L.tri_poly(W.poly(hole, dd), W.n), 'main')
    if panels:
        hb = h1 if shape == 'rect' else h1 - w / 2
        for k in range(2):
            a0 = s0 + w * k / 2 + 0.1; a1 = s0 + w * (k + 1) / 2 - 0.1
            for (q0, q1) in ((h0 + 0.15, h0 + (hb - h0) * 0.45), (h0 + (hb - h0) * 0.5, hb - 0.15)):
                _wbox(b, mat, W, a0, a1, q0, q1, dd, dd + 0.035, grp)
        _wbox(b, mat, W, s0 + w / 2 - 0.02, s0 + w / 2 + 0.02, h0, hb, dd, dd + 0.05, grp)
    return hole


# ============================================================== Daecher
def mansard_tris(u0, u1, v0, v1, he, inset, rise1, rise2, hip=True):
    """Mansard-Walmdach ueber Rechteck: unterer steiler Teil (Einzug 'inset', Hoehe rise1), oberer flacher Walm (rise2).
    Rueckgabe: (Dreiecke unten, Dreiecke oben, Knickkante-Hoehe, First-Endpunkte)."""
    a = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)
    bpts = np.array([[u0 + inset, v0 + inset], [u1 - inset, v0 + inset], [u1 - inset, v1 - inset], [u0 + inset, v1 - inset]], float)
    hk = he + rise1
    lower = []
    for i in range(4):
        j = (i + 1) % 4
        Q = np.array([np.r_[a[i], he], np.r_[a[j], he], np.r_[bpts[j], hk], np.r_[bpts[i], hk]])
        lower.append(Q)
    TL = L.quads(np.array(lower))
    TL = L.fix_orient(TL, _out_normals(TL, (u0 + u1) / 2, (v0 + v1) / 2))
    # oberer Teil: Walm ueber bpts
    bu0, bv0, bu1, bv1 = u0 + inset, v0 + inset, u1 - inset, v1 - inset
    W_ = bu1 - bu0; D_ = bv1 - bv0
    hr = hk + rise2
    if W_ >= D_:
        r0 = np.array([bu0 + D_ / 2, (bv0 + bv1) / 2, hr]); r1 = np.array([bu1 - D_ / 2, (bv0 + bv1) / 2, hr])
        if not hip: r0[0] = bu0; r1[0] = bu1
        c = [np.r_[bpts[k], hk] for k in range(4)]
        faces = [np.array([c[0], c[1], r1, r0]), np.array([c[2], c[3], r0, r1])]
        tris = [np.array([c[1], c[2], r1]), np.array([c[3], c[0], r0])]
    else:
        r0 = np.array([(bu0 + bu1) / 2, bv0 + W_ / 2, hr]); r1 = np.array([(bu0 + bu1) / 2, bv1 - W_ / 2, hr])
        if not hip: r0[1] = bv0; r1[1] = bv1
        c = [np.r_[bpts[k], hk] for k in range(4)]
        faces = [np.array([c[1], c[2], r1, r0]), np.array([c[3], c[0], r0, r1])]
        tris = [np.array([c[0], c[1], r0]), np.array([c[2], c[3], r1])]
    TU = np.concatenate([L.quads(np.array(faces))] + [t[None] for t in tris])
    TU = L.drop_degenerate(TU)
    TU = L.fix_orient(TU, np.tile([0, 0, 1.0], (len(TU), 1)))
    return TL, TU, hk, (r0, r1)


def _out_normals(T, cu, cv):
    m = T.mean(1)
    d = np.c_[m[:, 0] - cu, m[:, 1] - cv, np.zeros(len(m))]
    return d


# ============================================================== Reliefs (Hoehenbilder 0..1)
def _poly(d, pts, sx, sy, ox, oy, val):
    d.polygon([(ox + x * sx, oy + y * sy) for x, y in pts], fill=val)


LION = [(0.30, 1.00), (0.26, 0.86), (0.18, 0.80), (0.12, 0.70), (0.16, 0.60), (0.10, 0.52), (0.20, 0.50), (0.26, 0.44), (0.22, 0.36), (0.28, 0.22),
        (0.40, 0.18), (0.46, 0.08), (0.52, 0.02), (0.62, 0.0), (0.70, 0.06), (0.74, 0.0), (0.82, 0.04), (0.84, 0.14), (0.76, 0.18), (0.80, 0.26),
        (0.92, 0.22), (1.0, 0.28), (0.94, 0.34), (0.82, 0.38), (0.86, 0.46), (0.96, 0.44), (1.0, 0.52), (0.90, 0.56), (0.76, 0.54), (0.70, 0.62),
        (0.74, 0.76), (0.86, 0.86), (0.84, 0.96), (0.70, 0.94), (0.58, 0.84), (0.52, 0.92), (0.48, 1.0)]
UNICORN = [(0.30, 1.00), (0.26, 0.84), (0.18, 0.78), (0.14, 0.66), (0.20, 0.56), (0.14, 0.48), (0.24, 0.46), (0.28, 0.36), (0.26, 0.24),
           (0.34, 0.14), (0.40, 0.10), (0.46, 0.0), (0.50, 0.10), (0.56, 0.08), (0.64, 0.04), (0.72, 0.10), (0.74, 0.18), (0.70, 0.22),
           (0.76, 0.28), (0.92, 0.22), (1.0, 0.28), (0.92, 0.34), (0.80, 0.38), (0.84, 0.46), (0.94, 0.46), (0.98, 0.54), (0.86, 0.56), (0.72, 0.56),
           (0.68, 0.66), (0.74, 0.80), (0.84, 0.88), (0.82, 0.98), (0.68, 0.94), (0.56, 0.84), (0.50, 0.94), (0.46, 1.0)]


def _ell(d, cx, cy, rx, ry, ang, val, n=28):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ca, sa = math.cos(ang), math.sin(ang)
    pts = [(cx + rx * math.cos(a) * ca - ry * math.sin(a) * sa, cy + rx * math.cos(a) * sa + ry * math.sin(a) * ca) for a in t]
    d.polygon(pts, fill=val)


def _limb(d, pts, w, val):
    d.line(pts, fill=val, width=int(w))
    for x, y in (pts[0], pts[-1]):
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=val)


def _beast(d, ox, oy, s, flip, kind, val):
    """Steigender Loewe bzw. Einhorn (Schildhalter), Fuesse bei (ox, oy), Masstab s (Pixel je Meter), blickt nach +x (flip=-1: nach -x)."""
    f = flip
    X = lambda x: ox + f * x * s
    Y = lambda y: oy - y * s
    # Hinterbeine
    _limb(d, [(X(-0.25), Y(0.0)), (X(-0.35), Y(0.45)), (X(-0.15), Y(0.85))], 0.22 * s, val - 0.05)
    _limb(d, [(X(0.15), Y(0.0)), (X(0.1), Y(0.4)), (X(0.0), Y(0.85))], 0.2 * s, val - 0.1)
    # Rumpf (schraeg aufgerichtet)
    _ell(d, X(0.1), Y(1.12), 0.42 * s, 0.74 * s, f * 0.5, val)
    _ell(d, X(0.3), Y(1.62), 0.3 * s, 0.32 * s, 0, val + 0.03)
    # Schweif
    _limb(d, [(X(-0.3), Y(0.75)), (X(-0.75), Y(0.6)), (X(-0.85), Y(1.0)), (X(-0.65), Y(1.35))], 0.07 * s, val - 0.1)
    _ell(d, X(-0.62), Y(1.42), 0.1 * s, 0.14 * s, 0, val - 0.05)
    # Vorderbeine zum Schild
    _limb(d, [(X(0.3), Y(1.45)), (X(0.62), Y(1.35)), (X(0.78), Y(1.6))], 0.16 * s, val)
    _limb(d, [(X(0.28), Y(1.6)), (X(0.6), Y(1.85)), (X(0.72), Y(2.05))], 0.14 * s, val - 0.05)
    if kind == 'lion':
        # Maehne (gezackt) + Kopf mit Krone
        pts = []
        for k in range(18):
            a = 2 * math.pi * k / 18; r = (0.42 if k % 2 == 0 else 0.3) * s
            pts.append((X(0.32) + r * math.cos(a), Y(1.88) + r * math.sin(a)))
        d.polygon(pts, fill=val + 0.05)
        _ell(d, X(0.42), Y(1.95), 0.2 * s, 0.18 * s, 0, val + 0.15)
        _ell(d, X(0.6), Y(1.9), 0.11 * s, 0.08 * s, 0, val + 0.12)     # Schnauze
        d.polygon([(X(0.3), Y(2.12)), (X(0.36), Y(2.3)), (X(0.42), Y(2.16)), (X(0.48), Y(2.32)), (X(0.54), Y(2.12))], fill=val + 0.1)
    else:
        # Hals + Pferdekopf + Horn + Maehne
        _limb(d, [(X(0.25), Y(1.55)), (X(0.38), Y(2.05))], 0.26 * s, val)
        _ell(d, X(0.52), Y(2.1), 0.24 * s, 0.12 * s, -f * 0.5, val + 0.12)
        _limb(d, [(X(0.5), Y(2.2)), (X(0.78), Y(2.62))], 0.05 * s, val + 0.2)
        for k in range(5):
            _ell(d, X(0.2 - 0.02 * k), Y(1.6 + 0.12 * k), 0.08 * s, 0.05 * s, 0.4, val - 0.05)


def tympanum_heightmap(px_per_m, width, height):
    """Koenigliches Wappen (Hannover) mit Loewe und Einhorn als Schildhalter, Krone, Akanthusranken zu den Ecken.
    Bild: width x height Meter (Giebeldreieck-Grundlinie), Zeile 0 = oben (Giebelspitze)."""
    W = int(width * px_per_m); H = int(height * px_per_m)
    S = 4
    im = Image.new('F', (W * S, H * S), 0.0); d = ImageDraw.Draw(im)
    cx = W * S / 2
    s = px_per_m * S
    base_y = H * S * 0.985
    # Schild (oval) mit Ordensband, auf Postament
    sw, sh = 0.5 * s, 0.66 * s
    sy = base_y - 1.6 * s
    d.ellipse([cx - sw * 1.18, sy - sh * 0.12, cx + sw * 1.18, sy + sh * 2.12], fill=0.55)
    d.ellipse([cx - sw, sy, cx + sw, sy + sh * 2.0], fill=0.78)
    d.line([cx, sy + sh * 0.15, cx, sy + sh * 1.85], fill=0.66, width=int(S * 2))
    d.line([cx - sw * 0.92, sy + sh * 0.95, cx + sw * 0.92, sy + sh * 0.95], fill=0.66, width=int(S * 2))
    _ell(d, cx, sy + sh, sw * 0.3, sh * 0.32, 0, 0.88)                   # Herzschild
    # Krone
    kw = sw * 0.85; ky = sy - 0.36 * s
    d.rectangle([cx - kw, ky + 0.28 * s, cx + kw, ky + 0.42 * s], fill=0.82)
    for k in range(-2, 3):
        x = cx + k * kw * 0.48
        d.polygon([(x - kw * 0.2, ky + 0.3 * s), (x, ky + (0.0 if k else -0.12 * s)), (x + kw * 0.2, ky + 0.3 * s)], fill=0.78)
        d.ellipse([x - 0.05 * s, ky - 0.06 * s - (0.12 * s if k == 0 else 0), x + 0.05 * s, ky + 0.04 * s - (0.12 * s if k == 0 else 0)], fill=0.85)
    # Schildhalter
    _beast(d, cx - 1.75 * s, base_y - 0.05 * s, s * 0.78, 1, 'lion', 0.68)
    _beast(d, cx + 1.75 * s, base_y - 0.05 * s, s * 0.78, -1, 'unicorn', 0.68)
    # Postament / Bodenleiste
    d.rectangle([cx - 3.0 * s, base_y - 0.08 * s, cx + 3.0 * s, H * S], fill=0.4)
    # Akanthusranken zu den Ecken (liegende Spiralen mit Blaettern)
    for sgn in (-1, 1):
        x0 = cx + sgn * 2.9 * s
        pts = []
        for t in np.linspace(0, 1, 40):
            pts.append((x0 + sgn * t * 6.2 * s, base_y - 0.32 * s * (1 - t) - 0.12 * s * math.sin(t * 9)))
        d.line(pts, fill=0.5, width=int(0.12 * s))
        for k in range(5):
            xc = x0 + sgn * (0.6 + k * 1.15) * s; yc = base_y - (0.38 - 0.05 * k) * s
            rad = (0.36 - 0.05 * k) * s
            sp = []
            for t in np.linspace(0, 1.7 * math.pi, 24):
                rr = rad * (1 - 0.6 * t / (1.7 * math.pi))
                sp.append((xc + sgn * rr * math.cos(t + math.pi / 2), yc + rr * math.sin(t + math.pi / 2) * 0.85 - rad * 0.2))
            d.line(sp, fill=0.55, width=int(0.07 * s))
            _ell(d, xc - sgn * rad * 0.9, yc - rad * 0.55, rad * 0.55, rad * 0.2, -sgn * 0.5, 0.45)
            _ell(d, xc + sgn * rad * 0.5, yc - rad * 1.1, rad * 0.4, rad * 0.16, sgn * 0.3, 0.45)
    a = np.array(im.resize((W, H), Image.BILINEAR))
    a = ndimage.gaussian_filter(a, 0.7)
    dist = ndimage.distance_transform_edt(a > 0.05)
    a = np.where(a > 0.05, a * np.clip(0.5 + dist / (0.1 * px_per_m), 0, 1), 0)
    return np.clip(a, 0, 1)


def anthemion_heightmap(px_per_m, length, height, period=0.85):
    """Fries mit Palmetten und Lotusblueten, verbunden durch S-Ranken."""
    W = int(length * px_per_m); H = int(height * px_per_m)
    S = 4
    im = Image.new('F', (W * S, H * S), 0.0); d = ImageDraw.Draw(im)
    p = period * px_per_m * S
    n = int(W * S / p) + 2
    hh = H * S
    for k in range(n):
        x = k * p
        # Palmette (Faecher aus 7 Blaettern)
        cxp = x; by = hh * 0.86
        for j in range(-3, 4):
            ang = math.pi / 2 + j * 0.28
            L_ = hh * (0.62 - 0.05 * abs(j))
            ex = cxp + math.cos(ang) * L_ * 0.55; ey = by - math.sin(ang) * L_
            d.line([cxp, by, ex, ey], fill=0.75, width=int(S * 2.4))
            d.ellipse([ex - S * 2.4, ey - S * 2.4, ex + S * 2.4, ey + S * 2.4], fill=0.8)
        d.ellipse([cxp - hh * 0.07, by - hh * 0.07, cxp + hh * 0.07, by + hh * 0.07], fill=0.85)
        # Lotus (geschlossen) dazwischen
        lx = x + p / 2
        d.polygon([(lx, hh * 0.18), (lx - hh * 0.12, hh * 0.55), (lx - hh * 0.05, hh * 0.86), (lx + hh * 0.05, hh * 0.86), (lx + hh * 0.12, hh * 0.55)], fill=0.6)
        d.line([lx - hh * 0.18, hh * 0.5, lx - hh * 0.06, hh * 0.84], fill=0.55, width=int(S * 2))
        d.line([lx + hh * 0.18, hh * 0.5, lx + hh * 0.06, hh * 0.84], fill=0.55, width=int(S * 2))
        # S-Ranke unten verbindet
        pts = []
        for t in np.linspace(0, 1, 16):
            pts.append((x + t * p, hh * 0.9 - math.sin(t * math.pi) * hh * 0.08))
        d.line(pts, fill=0.5, width=int(S * 2))
    a = np.array(im.resize((W, H), Image.BILINEAR))
    a = ndimage.gaussian_filter(a, 0.6)
    return np.clip(a, 0, 1)


def garland_heightmap(px_per_m, length, height):
    """Fuellungsrelief (Feston mit Schleifen) fuer Seitenfelder."""
    W = int(length * px_per_m); H = int(height * px_per_m); S = 4
    im = Image.new('F', (W * S, H * S), 0.0); d = ImageDraw.Draw(im)
    ww, hh = W * S, H * S
    d.rectangle([0, 0, ww - 1, hh - 1], outline=0.5, width=int(S * 2))
    pts = [(ww * 0.08 + t * ww * 0.84, hh * 0.3 + math.sin(t * math.pi) * hh * 0.4) for t in np.linspace(0, 1, 40)]
    for k, (x, y) in enumerate(pts):
        r = hh * (0.07 + 0.08 * math.sin(k / 39 * math.pi))
        d.ellipse([x - r, y - r, x + r, y + r], fill=0.6 + 0.2 * ((k % 3) == 0))
    for x in (ww * 0.08, ww * 0.92, ww * 0.5):
        d.ellipse([x - hh * 0.1, hh * 0.15, x + hh * 0.1, hh * 0.35], fill=0.75)
    a = np.array(im.resize((W, H), Image.BILINEAR))
    return np.clip(ndimage.gaussian_filter(a, 0.6), 0, 1)


def cartouche_heightmap(px_per_m, w, h):
    """Kartusche / Muschelornament ueber den Fenstern des Leinefluegels."""
    W = max(int(w * px_per_m), 4); H = max(int(h * px_per_m), 4); S = 4
    im = Image.new('F', (W * S, H * S), 0.0); d = ImageDraw.Draw(im)
    ww, hh = W * S, H * S
    d.ellipse([ww * 0.3, hh * 0.05, ww * 0.7, hh * 0.95], fill=0.8)
    for j in range(-3, 4):
        ang = math.pi / 2 + j * 0.35
        d.line([ww / 2, hh * 0.95, ww / 2 + math.cos(ang) * ww * 0.45, hh * 0.95 - math.sin(ang) * hh * 0.85], fill=0.6, width=int(S * 1.5))
    for sgn in (-1, 1):
        d.ellipse([ww / 2 + sgn * ww * 0.35 - hh * 0.2, hh * 0.5, ww / 2 + sgn * ww * 0.35 + hh * 0.2, hh * 0.9], fill=0.5)
    a = np.array(im.resize((W, H), Image.BILINEAR))
    return np.clip(ndimage.gaussian_filter(a, 0.5), 0, 1)

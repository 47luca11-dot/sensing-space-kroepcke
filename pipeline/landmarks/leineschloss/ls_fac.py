"""Leineschloss -- Fassaden-Helfer: Wandbaender mit Oeffnungen, Fenster mit Schmuck (Gewaende, Verdachung, Giebel,
Schlussstein, Kartusche), Gurtgesimse, Hauptgesims mit Konsolen/Zahnschnitt, Gauben, Fallrohre, Lisenen, Mansarddach."""
import math
import numpy as np
import lslib as L
import ls_parts as P
from lslib import Builder, nrm, UP

# Materialnamen (Praefix LS_ -> keine Kollision mit gleichnamigen Materialien anderer Dateien im Spiel)
SAND = 'LS_Sandstein'; QUAD = 'LS_Sandstein_Quader'; RUST = 'LS_Rustika'; PUTZ = 'LS_Putz'; SCH = 'LS_Schiefer'
ZINK = 'LS_Zink'; KUP = 'LS_Kupfer'; MARM = 'LS_Marmor'; GRAN = 'LS_Granit'; QUAI = 'LS_Quai'; FLD = 'LS_Flachdach'
HOLZ = 'LS_Holz'; PW = 'LS_Putz_weiss'; PLAT = 'LS_Platten'; FR = 'LS_Rahmen'; FRD = 'LS_Rahmen_dunkel'; GL = 'LS_Fensterglas'
EIS = 'LS_Eisen'; EDS = 'LS_Edelstahl'; BRZ = 'LS_Bronze'; DUNK = 'LS_Dunkel'


# ------------------------------------------------------------------ Grundbausteine in Wandkoordinaten
def wprism(b, mat, W, poly2d, d0, d1, grp='det', back=False):
    """Prisma: Umriss (s,h) in Wandebene, von Tiefe d0 bis d1 (d1 > d0 = weiter vor der Wand)."""
    poly2d = np.asarray(poly2d, float)
    if L.signed_area(poly2d) < 0: poly2d = poly2d[::-1]
    A = W.poly(poly2d, d0); Bp = W.poly(poly2d, d1)
    Q = np.stack([A, np.roll(A, -1, 0), np.roll(Bp, -1, 0), Bp], axis=1)
    parts = [L.quads(Q), L.tri_poly(Bp, W.n)]
    if back: parts.append(L.tri_poly(A, -W.n))
    T = np.concatenate(parts)
    c = W.P(*poly2d.mean(0), (d0 + d1) / 2)
    T = L.orient_away(T, c) if not back else T
    b.add(mat, T, grp, closed=back)


def wbox(b, mat, W, s0, s1, h0, h1, d0, d1, grp='det'):
    P._wbox(b, mat, W, s0, s1, h0, h1, d0, d1, grp)


def band_wall(b, W, s0, s1, h0, h1, mat, holes, grp='main'):
    """Wandflaeche (s0..s1, h0..h1) mit allen Loechern, die vollstaendig im Band liegen."""
    hs = [h for h in holes if h[:, 1].min() >= h0 - 1e-6 and h[:, 1].max() <= h1 + 1e-6 and h[:, 0].min() >= s0 - 1e-6 and h[:, 0].max() <= s1 + 1e-6]
    L.wall_face(b, mat, W, L.rect2d(s0, s1, h0, h1), hs, 0.0, grp)


def hsweep(b, mat, pts_uv, h, prof, closed=False, grp='det'):
    """Horizontales Profil (x nach aussen, y nach oben) entlang Grundrisslinie (gegen Uhrzeigersinn -> aussen rechts)."""
    pts = np.c_[np.asarray(pts_uv, float), np.full(len(pts_uv), h)]
    prof = list(prof)
    prof = [(-0.02, prof[0][1])] + prof + [(-0.02, prof[-1][1])]
    T = L.sweep_tris(pts, prof, UP, closed=closed, caps=not closed)
    b.add(mat, T, grp, closed=True)


def wall_hsweep(b, mat, W, s0, s1, h, prof, grp='det', caps=True):
    """Gesimsstueck vor einer Wand W von s0 bis s1 (Hoehe h)."""
    pts = np.array([W.P(s0, h), W.P(s1, h)])
    prof = [(-0.02, prof[0][1])] + list(prof) + [(-0.02, prof[-1][1])]
    T = L.sweep_tris(pts, prof, UP, closed=False, caps=caps)
    # Seite = Tangente x oben = W.n (Wand gegen den Uhrzeigersinn) -> passt
    if np.dot(np.cross(W.t, UP), W.n) < 0:
        T = L.sweep_tris(pts, prof, UP, closed=False, caps=caps, side_flip=True)
    b.add(mat, T, grp, closed=True)


# ------------------------------------------------------------------ Fenster mit Schmuck
class WSpec:
    def __init__(self, w, h0, h1, shape='rect', bars=(2, 3), depth=0.24, frame=None, hood=None, pediment=False, keystone=False,
                 cartouche=False, sill=True, arch_frame=None, rise=None, glass=GL, fmat=FR, reveal=SAND, door=False, deco_mat=SAND,
                 transom=None, panel_below=False, ears=False):
        self.__dict__.update(locals()); del self.__dict__['self']


def place_window(b, W, sc, sp, grp='det', ctx=None):
    """Fenster (Mitte sc) in Wand W nach Spezifikation sp. Rueckgabe: Lochumriss (s,h) fuer die Wandflaeche."""
    s0, s1 = sc - sp.w / 2, sc + sp.w / 2
    if sp.door:
        hole = P.door_unit(b, W, s0, s1, sp.h0, sp.h1, depth=sp.depth, grp=grp, reveal_mat=sp.reveal, shape='rect' if sp.shape == 'rect' else 'arch')
    else:
        hole = P.window_unit(b, W, s0, s1, sp.h0, sp.h1, sp.shape, depth=sp.depth, bars=sp.bars, frame_mat=sp.fmat, glass=sp.glass,
                             reveal_mat=sp.reveal, grp=grp, rise=sp.rise, sill=sp.sill, transom=sp.transom)
    dm = sp.deco_mat
    top = sp.h1
    if sp.frame:                       # Gewaende (Breite, Ausladung)
        fw, fp = sp.frame
        prof = [(0.0, -0.01)] + [(x, y) for x, y in P.prof_architrave(fw, fp)[1:-1]] + [(fw, -0.01)]
        T = L.sweep_tris(W.poly(hole, 0.0), prof, bv=W.n, closed=True, caps=False)
        b.add(dm, T, grp, closed=True)
        top = sp.h1 + fw
        if sp.ears:                    # Ohren (Verkroepfung) an den oberen Ecken
            for sg in (-1, 1):
                x = sc + sg * (sp.w / 2 + fw * 0.5)
                wbox(b, dm, W, x - fw * 0.75, x + fw * 0.75, sp.h1 + fw - 0.18, sp.h1 + fw + 0.02, -0.01, fp + 0.01, grp)
    if sp.arch_frame:                  # Archivolte um Rundbogen-Oeffnung
        fw, fp = sp.arch_frame
        prof = [(0.0, -0.01)] + [(x, y) for x, y in P.prof_architrave(fw, fp)[1:-1]] + [(fw, -0.01)]
        ring = hole[2:] if sp.shape != 'rect' else hole
        T = L.sweep_tris(W.poly(ring, 0.0), prof, bv=W.n, closed=(sp.shape == 'rect'), caps=(sp.shape != 'rect'))
        b.add(dm, T, grp, closed=True)
    if sp.keystone:
        kw, kh = 0.28, 0.5
        y0 = sp.h1 - 0.12
        poly = np.array([[sc - kw * 0.38, y0], [sc + kw * 0.38, y0], [sc + kw / 2, y0 + kh], [sc - kw / 2, y0 + kh]])
        wprism(b, dm, W, poly, -0.01, 0.1, grp)
        top = max(top, y0 + kh)
    if sp.hood:                        # gerade Verdachung auf Konsolen
        hh = top + 0.02
        ext = 0.18 if sp.frame is None else sp.frame[0] + 0.12
        wall_hsweep(b, dm, W, s0 - ext, s1 + ext, hh, P.prof_string(0.2, 0.26), grp)
        con = P.console_s(0.14, 0.18, 0.42)
        for sg in (-1, 1):
            x = sc + sg * (sp.w / 2 + ext - 0.1)
            M = L.mat_frame(W.P(x, hh), W.n, W.t, UP)
            if np.linalg.det(M[:3, :3]) < 0: M = L.mat_frame(W.P(x, hh), W.n, -W.t, UP)
            for (mat, g2), (Ts, UVs) in con.G.items():
                for T_ in Ts: b.add(dm, L.xform(T_, M), grp, closed=True)
        top = hh + 0.26
    if sp.pediment:                    # Dreiecksgiebel
        ext = 0.2 if sp.frame is None else sp.frame[0] + 0.12
        hb = top + 0.02
        a0, a1 = s0 - ext - 0.05, s1 + ext + 0.05
        wall_hsweep(b, dm, W, a0, a1, hb, P.prof_string(0.18, 0.2), grp)
        rise = (a1 - a0) / 2 * math.tan(math.radians(20))
        tri = np.array([[a0, hb + 0.2], [a1, hb + 0.2], [sc, hb + 0.2 + rise]])
        wprism(b, dm, W, tri, -0.01, 0.12, grp)
        # Schraeggesims
        for sg in (-1, 1):
            p0 = np.array([sc, hb + 0.2 + rise]); p1 = np.array([sc + sg * (a1 - a0) / 2, hb + 0.2])
            dd = p1 - p0; ln = np.hypot(*dd); t2 = dd / ln; n2 = np.array([-t2[1], t2[0]]) * (-sg)
            quad = np.array([p0 + n2 * 0.0, p1, p1 + n2 * 0.14, p0 + n2 * 0.14 + t2 * 0.0])
            quad = np.array([p0 - np.array([0, 0.0]), p1 + t2 * 0.08, p1 + t2 * 0.08 + n2 * 0.15, p0 + n2 * 0.15])
            wprism(b, dm, W, quad, -0.01, 0.2, grp)
        top = hb + 0.35 + rise
    if sp.cartouche and ctx is not None:
        cw, ch = 0.75, 0.42
        Hm = ctx.cartouche(cw, ch)
        T, on = L.relief_tris(Hm, W, sc - cw / 2, top + 0.02, cw, ch, 0.09, 0.0)
        b.add(dm, T, grp)
    if sp.panel_below:                 # Brüstungsfeld
        wbox(b, dm, W, s0 + 0.05, s1 - 0.05, sp.h0 - 0.95, sp.h0 - 0.25, -0.01, 0.03, grp)
    return hole


# ------------------------------------------------------------------ Hauptgesims mit Konsolen + Zahnschnitt
def main_cornice(b, pts_uv, h, H=1.0, proj=0.85, closed=True, modillions=0.6, dentils=True, mat=SAND, grp='det', skip=None):
    """Kranzgesims entlang Grundriss (gegen Uhrzeigersinn). Konsolen alle 'modillions' m unter der Haengeplatte."""
    hsweep(b, mat, pts_uv, h, P.prof_cornice(proj, H), closed=closed, grp=grp)
    pts = np.asarray(pts_uv, float)
    segs = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts) if closed else len(pts) - 1)]
    mod = P.modillion(0.14, proj * 0.55, H * 0.2) if modillions else None
    for k, (a, c) in enumerate(segs):
        if skip is not None and skip(k, a, c): continue
        d = c - a; Lg = np.hypot(*d)
        if Lg < 0.5: continue
        t = np.r_[d / Lg, 0]; n = np.array([t[1], -t[0], 0.0])
        if modillions:
            n_m = max(int((Lg - 0.4) / modillions), 1)
            for i in range(n_m + 1):
                s = 0.2 + (Lg - 0.4) * i / n_m
                o = np.r_[a, 0] + t * s + UP * (h + H * 0.36)
                M = L.mat_frame(o, n, t, UP)
                if np.linalg.det(M[:3, :3]) < 0: M = L.mat_frame(o, n, -t, UP)
                for (m_, g_), (Ts, UVs) in mod.G.items():
                    for T_ in Ts: b.add(mat, L.xform(T_, M), grp, closed=True)
        if dentils:
            nd = int(Lg / 0.17)
            Q = []
            for i in range(nd):
                s = (i + 0.5) * Lg / nd
                c0 = np.r_[a, 0] + t * s + n * 0.13 + UP * (h + 0.12 * H + 0.05 * H)
                Q.append(L.obox_tris(c0, t, n, UP, 0.045, 0.06, 0.05 * H))
            if Q: b.add(mat, np.concatenate(Q), grp)


# ------------------------------------------------------------------ Mansarddach mit Gauben
def mansard(b, u0, u1, v0, v1, he, inset, rise1, rise2, dormers=(), hip=True, ridge_mat=ZINK, slate=SCH, grp='main', dormer_spec=None):
    """Mansarddach ueber Rechteck; dormers: Liste (seite, position) mit seite in {'v0','v1','u0','u1'}."""
    TL, TU, hk, (r0, r1) = P.mansard_tris(u0, u1, v0, v1, he, inset, rise1, rise2, hip)
    b.add(slate, TL, grp); b.add(slate, TU, grp)
    # Knickleiste (Zink) + Gratkappen
    ring = np.array([[u0 + inset, v0 + inset], [u1 - inset, v0 + inset], [u1 - inset, v1 - inset], [u0 + inset, v1 - inset]])
    hsweep(b, ridge_mat, ring, hk - 0.06, [(0.0, 0.0), (0.06, 0.02), (0.06, 0.1), (0.0, 0.12)], closed=True, grp='det')
    roll = [(0.045 * math.cos(a), 0.045 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    lines = [(r0, r1)] if np.linalg.norm(r1 - r0) > 0.05 else []
    c = [np.r_[ring[k], hk] for k in range(4)]
    if hip:
        if abs((u1 - u0) - (v1 - v0)) >= 0 and (u1 - u0) >= (v1 - v0):
            lines += [(c[0], r0), (c[3], r0), (c[1], r1), (c[2], r1)]
        else:
            lines += [(c[0], r0), (c[1], r0), (c[2], r1), (c[3], r1)]
    for p0, p1 in lines:
        d = p1 - p0
        if np.linalg.norm(d) < 0.05: continue
        side = nrm(np.cross(d, UP)) if np.linalg.norm(np.cross(d, UP)) > 1e-6 else np.array([1.0, 0, 0])
        bv = nrm(np.cross(side, d))
        T = L.sweep_tris(np.array([p0, p1]) + UP * 0.03, roll, bv=bv, caps=True)
        b.add(ridge_mat, T, 'det', closed=True)
    # Eckgrate des unteren Teils
    a = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    for k in range(4):
        p0 = np.r_[a[k], he]; p1 = c[k]
        d = p1 - p0; side = nrm(np.cross(d, UP)); bv = nrm(np.cross(side, d))
        b.add(ridge_mat, L.sweep_tris(np.array([p0, p1]), roll, bv=bv, caps=True), 'det', closed=True)
    for (side, pos) in dormers:
        dormer(b, side, pos, u0, u1, v0, v1, he, inset, rise1, dormer_spec or {})
    return hk, (r0, r1)


def dormer(b, side, pos, u0, u1, v0, v1, he, inset, rise1, spec):
    """Gaube auf dem steilen Mansardteil. side: welche Rechteckseite, pos: Lage entlang der Seite (u bzw. v)."""
    w = spec.get('w', 1.15); hd = spec.get('h', 1.95); top = spec.get('top', 'seg')
    a_front = spec.get('front', 0.18)                      # Einzug der Gaubenfront ab Traufe
    slope = rise1 / inset
    a = a_front
    if side == 'v0':   W = L.Wall((pos - w / 2, v0 + a), (pos + w / 2, v0 + a))
    elif side == 'v1': W = L.Wall((pos + w / 2, v1 - a), (pos - w / 2, v1 - a))
    elif side == 'u1': W = L.Wall((u1 - a, pos - w / 2), (u1 - a, pos + w / 2))
    else:              W = L.Wall((u0 + a, pos + w / 2), (u0 + a, pos - w / 2))
    h_base = he + slope * a_front - 0.12
    h_top = h_base + hd
    depth = 1.9
    rise = 0.32 if top == 'seg' else (w / 2 if top == 'arch' else 0.0)
    if top == 'tri':
        outline = np.array([[0, h_base], [w, h_base], [w, h_top], [w / 2, h_top + 0.45], [0, h_top]])
    else:
        outline = L.arch2d(0, w, h_base, h_top, 10, rise=rise if top == 'seg' else None)
    # Fenster
    win_h0 = h_base + 0.3; win_h1 = h_top + (rise if top != 'tri' else 0) - 0.16
    ww = w - 0.3
    sp = WSpec(ww, win_h0, win_h1, 'seg' if top == 'seg' else ('arch' if top == 'arch' else 'rect'), bars=(2, 2), depth=0.14,
               rise=(rise * 0.85 if top == 'seg' else None), sill=False, reveal=ZINK)
    hole = place_window(b, W, w / 2, sp, grp='det')
    # Front mit Loch
    L.wall_face(b, spec.get('mat', ZINK), W, outline, [hole], 0.0, 'det')
    # Seitenwangen und Dach: Umriss nach hinten extrudieren
    A = W.poly(outline, 0.0); Bk = W.poly(outline, -depth)
    n = len(outline)
    Q = np.stack([A, np.roll(A, -1, 0), np.roll(Bk, -1, 0), Bk], axis=1)
    # untere Kante (Brüstung) weglassen: liegt im Dach
    keep = np.ones(n, bool); keep[0] = False
    T = L.quads(Q[keep])
    T = L.orient_away(T, W.P(w / 2, h_base + hd / 2, -depth / 2))
    sides = T[np.abs(L.nrm(L.tri_normals(T)) @ UP) < 0.35]
    tops = T[np.abs(L.nrm(L.tri_normals(T)) @ UP) >= 0.35]
    b.add(spec.get('mat', ZINK), sides, 'det')
    b.add(ZINK, tops, 'det')
    # Gesimsleiste ueber der Front
    if top == 'tri':
        wbox(b, SAND, W, -0.05, w + 0.05, h_top - 0.05, h_top + 0.05, -0.02, 0.1, 'det')


# ------------------------------------------------------------------ Kleinteile
def downpipe(b, p_top, h_bot, r=0.055, mat=ZINK, n=8, bracket_every=2.0):
    p = np.asarray(p_top, float)
    prof = [(r, h_bot), (r, p[2])]
    T = L.lathe_tris(prof, n, (p[0], p[1], 0), cap_top=True)
    b.add(mat, T, 'det')
    # Schelle + Kniestueck oben
    for h in np.arange(h_bot + 1.0, p[2] - 0.3, bracket_every):
        b.add(mat, L.lathe_tris([(r * 1.35, h), (r * 1.35, h + 0.05)], n, (p[0], p[1], 0), cap_top=True, cap_bottom=True), 'det', closed=True)


def pilaster_strip(b, mat, W, s0, s1, h0, h1, proj=0.06, grp='det'):
    wbox(b, mat, W, s0, s1, h0, h1, -0.02, proj, grp)


def quoins(b, mat, W, s_edge, h0, h1, sgn=1, course=0.5, long=0.9, short=0.55, proj=0.04, grp='det'):
    """Eckquader (abwechselnd lang/kurz) an der Ecke s_edge; sgn: Richtung ins Wandfeld."""
    h = h0; k = 0
    while h < h1 - 1e-3:
        h2 = min(h + course, h1)
        ln = long if k % 2 == 0 else short
        a, c = (s_edge, s_edge + sgn * ln)
        wbox(b, mat, W, min(a, c), max(a, c), h + 0.012, h2 - 0.012, -0.02, proj, grp)
        h = h2; k += 1


def railing(b, p0, p1, h0, height=1.0, post=1.2, mat=EIS, bar=0.018, top_d=0.045, infill='bars', grp='det'):
    """Gelaender zwischen Grundrisspunkten p0, p1 (lokal u,v) auf Hoehe h0."""
    p0 = np.r_[np.asarray(p0, float)[:2], h0]; p1 = np.r_[np.asarray(p1, float)[:2], h0]
    d = p1 - p0; Lg = np.linalg.norm(d[:2])
    if Lg < 0.05: return
    t = d / np.linalg.norm(d); side = nrm(np.cross(t, UP))
    sq = lambda r: [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
    b.add(mat, L.sweep_tris(np.array([p0 + UP * height, p1 + UP * height]), sq(top_d / 2), bv=UP if abs(t[2]) < 0.9 else side, caps=True), grp, closed=True)
    b.add(mat, L.sweep_tris(np.array([p0 + UP * 0.12, p1 + UP * 0.12]), sq(bar), bv=UP if abs(t[2]) < 0.9 else side, caps=True), grp, closed=True)
    n_p = max(int(Lg / post), 1)
    for i in range(n_p + 1):
        q = p0 + d * i / n_p
        b.add(mat, L.lathe_tris([(0.025, q[2]), (0.025, q[2] + height)], 6, (q[0], q[1], 0), cap_top=True), grp)
    if infill == 'bars':
        n_b = int(Lg / 0.12)
        for i in range(1, n_b):
            q = p0 + d * i / n_b
            b.add(mat, L.lathe_tris([(bar * 0.7, q[2] + 0.12), (bar * 0.7, q[2] + height)], 4, (q[0], q[1], 0)), grp)
    elif infill == 'x':
        for i in range(n_p):
            qa = p0 + d * i / n_p; qb = p0 + d * (i + 1) / n_p
            for (x0, x1) in ((qa + UP * 0.12, qb + UP * (height - 0.03)), (qa + UP * (height - 0.03), qb + UP * 0.12)):
                dd = x1 - x0; sd = nrm(np.cross(dd, UP)) if np.linalg.norm(np.cross(dd, UP)) > 1e-6 else side
                b.add(mat, L.sweep_tris(np.array([x0, x1]), sq(bar * 0.8), bv=nrm(np.cross(sd, dd)), caps=True), grp, closed=True)


def balustrade(b, pts_uv, h0, H=0.95, closed=False, mat=SAND, spacing=0.3, grp='det', pedestals=()):
    """Steinbalustrade: Sockel, Baluster, Handlauf entlang Linienzug; pedestals: Liste Punkte (u,v) fuer Postamente."""
    pts = np.asarray(pts_uv, float)
    hsweep(b, mat, pts, h0, [(0.0, 0.0), (0.2, 0.0), (0.2, 0.16), (0.0, 0.16)], closed=closed, grp=grp)
    hsweep(b, mat, pts, h0 + H - 0.16, [(0.0, 0.0), (0.24, 0.0), (0.26, 0.06), (0.26, 0.12), (0.22, 0.16), (0.0, 0.16)], closed=closed, grp=grp)
    bal = P.baluster(0.18, H - 0.32, mat, 8)
    segs = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts) if closed else len(pts) - 1)]
    for a, c in segs:
        d = c - a; Lg = np.hypot(*d); t = d / Lg; n = np.array([t[1], -t[0]])
        nb = int(Lg / spacing)
        for i in range(nb):
            s = (i + 0.5) * Lg / nb
            q = a + t * s + n * 0.1
            if any(np.hypot(*(q - np.asarray(pp))) < 0.35 for pp in pedestals): continue
            M = L.trans(q[0], q[1], h0 + 0.16)
            for (m_, g_), (Ts, UVs) in bal.G.items():
                for T_ in Ts: b.add(mat, L.xform(T_, M), grp, closed=True)
    for pp in pedestals:
        pp = np.asarray(pp, float)
        b.add(mat, L.bevel_box_tris(pp[0] - 0.3, pp[0] + 0.3, pp[1] - 0.3, pp[1] + 0.3, h0, h0 + H + 0.05, c=0.02), grp, closed=True)

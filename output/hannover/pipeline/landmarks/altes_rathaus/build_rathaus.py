"""Altes Rathaus Hannover -- handmodelliertes Wahrzeichen (Spielframe, Meter).
Fluegel: Marktfluegel (NW, Am Markte), Schmiedestrassenfluegel (NO), Karmarschstrassenfluegel (SO, Hase 1880er; NICHT die Ratsapotheke -- die steht gegenueber, Foto 17),
Gerichtsfluegel 'Dogenpalast'/Standesamt (SW, Andreae 1845-50), Verbindungsbau, Atrium im Hof.
Masse: Grundriss city-LoD2 ID_dedae90c (Schmiedestrasse 1), Hoehen aus LGLN-LoD2/DOM1 2016, Gliederung aus Fotos.
Aufruf: python build_rathaus.py <out.glb> [--nofrieze]"""
import sys, os, json, math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box as sbox
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ar_geom import Frame, MB, unit, area2, frame_matrix, UP
from ar_parts import *
from ar_export import export, tiles_for_mats, MATS
from ar_shops import fascia, blade_sign, emit_rooms, display_protos, STATE as SHOPSTATE

ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else os.path.join(HERE, 'build', 'altes_rathaus.glb')
TEX = os.path.join(HERE, 'build', 'tex')
T = tiles_for_mats()
mb = MB(T)          # Hauptgeometrie
col = MB()          # Kollisionsersatz
LOG = []

# ------------------------------------------------------------------------------------------- Grundriss (LoD2)
N = np.array([-193.86, 323.41]); W = np.array([-220.08, 347.51]); E = np.array([-161.79, 360.26]); S = np.array([-184.68, 389.84])
Wb = np.array([-211.15, 357.28])                  # Ende Marktfluegel-SW-Giebel
GsLod = np.array([-209.38, 360.47])               # Beginn Gerichtsfluegel (springt 0.7 m vor)


def U(v): v = np.asarray(v, float); return v / np.linalg.norm(v)


def inward_of(p0, p1):
    t = U(p1 - p0); n = np.array([-t[1], t[0]]); return -n          # n = aussen


dNE = U(N - E)                       # F2 Richtung E -> N
dSE = U(E - S)                       # F3 Richtung S -> E
dSW = U(S - GsLod)                   # F4 Richtung Gs -> S
N2 = N - dNE * 13.40                 # Marktfluegel-Hofseite auf F2
E2 = E + dNE * 8.16                  # Karmarschstrassenfluegel-Hofseite auf F2
Q = S - dSW * 8.18                   # Karmarschstrassenfluegel-Hofseite auf F4
inF4 = inward_of(GsLod, S)
C1 = Wb + U(Wb - W) * 3.52 * 1.0     # Ende Verbindungsbau (Flucht Marktfluegelgiebel)
Gs = C1 + (-inF4) * 0.70             # Vorsprung Gerichtsfluegel
inF1 = inward_of(N, W); inF2 = inward_of(E, N); inF3 = inward_of(S, E)

# Bodenhoehe (Heightfield) -----------------------------------------------------------------
_HJ = json.load(open(ROOT + '/web/assets/city/ground/heightfield.json'))       # Raster-Metadaten (Groesse aendert sich mit dem Bodenradius)
HF = np.fromfile(ROOT + '/web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(_HJ['h'], _HJ['w'])


def ground(x, z):
    j = (x - _HJ['x0']) / _HJ['dx']; i = (z - _HJ['z0']) / _HJ['dz']; j0 = int(math.floor(j)); i0 = int(math.floor(i)); fx = j - j0; fz = i - i0
    return float(HF[i0, j0] * (1 - fx) * (1 - fz) + HF[i0, j0 + 1] * fx * (1 - fz) + HF[i0 + 1, j0] * (1 - fx) * fz + HF[i0 + 1, j0 + 1] * fx * fz)


def ground_along(p0, p1, a, out=1.2):
    t = U(p1 - p0); n = np.array([-t[1], t[0]])
    q = p0 + t * a + n * out
    return ground(q[0], q[1])


YB = -0.8            # Unterkante aller Waende (unter Terrain)
LOG.append('ground N %.2f W %.2f E %.2f S %.2f' % (ground(*N), ground(*W), ground(*E), ground(*S)))

# Hoehen -----------------------------------------------------------------------------------
MF_EAVE = 9.60; MF_RIDGE = 21.20; MF_ROOFLINE = 9.95
SF_EAVE = 9.90; SF_RIDGE = 19.10; SF_ROOFLINE = 10.25
AF_EAVE = 15.80; AF_RIDGE = 22.40; AF_ROOFLINE = 16.15
GF_TOP = 22.95; GF_RIDGE = 23.75
CN_TOP = 11.00
YARD = 11.10; YARD_BLOCK = 17.60


# =========================================================================================== Fassadenhelfer
class Fac:
    def __init__(self, name, p0, p1, ytop, mat=BR, ybot=YB, outline=None):
        self.name = name; self.p0 = np.asarray(p0, float); self.p1 = np.asarray(p1, float)
        self.fr = Frame.facade(self.p0, self.p1)
        self.L = float(np.linalg.norm(self.p1 - self.p0))
        self.mat = mat; self.cuts = []
        self.outline = outline if outline is not None else [(0, ybot), (self.L, ybot), (self.L, ytop), (0, ytop)]
        self.low = []          # (a0, a1) Bereiche ohne Sockel
        self.g = np.mean([ground_along(self.p0, self.p1, a) for a in np.linspace(0, self.L, 7)])
        self.gmax = max(ground_along(self.p0, self.p1, a) for a in np.linspace(0, self.L, 9))

    def op(self, o, low=False):
        cut = build_opening(mb, self.fr, o, protos=True)
        self.cuts.append(cut)
        if low or o.y0 < self.gmax + 0.7:
            rw = o.ring['w'] if o.ring else 0.0
            self.low.append((o.a - o.w / 2 - rw - 0.02, o.a + o.w / 2 + rw + 0.02))
        return o

    def cut(self, poly):
        self.cuts.append(np.asarray(poly, float))

    def lattice(self, y0, y1, a0=0.0, a1=None, **kw):
        a1 = self.L if a1 is None else a1
        lattice_band(mb, self.fr, a0, a1, y0, y1, **kw)
        self.cut([(a0, y0), (a1, y0), (a1, y1), (a0, y1)])

    def frieze(self, y0, h=0.55, a0=0.0, a1=None):
        a1 = self.L if a1 is None else a1
        frieze_band(mb, self.fr, a0, a1, y0, h)
        self.cut([(a0, y0), (a1, y0), (a1, y0 + h), (a0, y0 + h)])

    def texband(self, mat, y0, y1, a0=0.0, a1=None, recess=0.03):
        a1 = self.L if a1 is None else a1
        recessed_band(mb, self.fr, a0, a1, y0, y1, mat, recess=recess)
        self.cut([(a0, y0), (a1, y0), (a1, y1), (a0, y1)])

    def course(self, y, h=0.10, proj=0.05, a0=0.0, a1=None, mat=BR):
        """Gesimsband (vorspringend) -- Box mit Endkappen"""
        a1 = self.L if a1 is None else a1
        mb.box(mat, self.fr, a0, a1, y, y + h, 0.0, proj, faces='xXyYZ')

    def finish(self):
        wall_face(mb, self.fr, self.outline, self.cuts, self.mat)


def P2(fac, a, c=0.0):
    return fac.p0 + U(fac.p1 - fac.p0) * a + np.array([-U(fac.p1 - fac.p0)[1], U(fac.p1 - fac.p0)[0]]) * c


# =========================================================================================== Fensterbausteine (Rathaus)
def win_up(a, y0=5.55, w=1.18, hs=2.05, rise=0.80, lights=2, kind='point', tymp=True, ring_w=0.13, alt=True, bars_h=0.42):
    return Opening(a, y0, w, hs, kind, rise=rise, ring={'w': ring_w, 'proj': 0.015, 'alt': alt}, tymp={'n': lights if lights < 3 else 2} if tymp else None,
                   lights=lights, bars_h=bars_h, bars_v=1, orders=((0, 0.10), (0.09, 0.30)), nj=6)


def win_ground(a, y0=2.05, w=1.55, hs=1.75, rise=0.28, lights=3):
    return Opening(a, y0, w, hs, 'seg', rise=rise, ring={'w': 0.13, 'proj': 0.015, 'alt': True}, lights=lights, bars_h=0.36,
                   bars_v=1, orders=((0, 0.10), (0.09, 0.30)), nj=5)


def cellar(a, w=1.80, y0=None, h=0.50, rise=0.72, g=0.0):
    """breite, tiefe Kellerbogennische am Sockel (Fotos Marktseite: ~1.8 m breit, Scheitel ~1.3 m), Rueckwand Backstein
    mit kleinem vergittertem Kellerfenster"""
    return Opening(a, g - 0.05, w, h, 'seg', rise=rise, ring={'w': 0.13, 'proj': 0.0, 'alt': True}, content='niche', back_mat=BK,
                   lights=1, orders=((0, 0.12), (0.08, 0.45)), sill=False, nj=2)


def cellar_window(fac, o, g):
    """Kellerfenster in der Nischenrueckwand (c = -0.45): dunkle Oeffnung, Sandsteinrahmen, Eisengitter"""
    fr = fac.fr
    ww = min(0.95, o.w * 0.5); y0 = g + 0.30; y1 = g + o.hs + 0.25
    c = -0.45 + 0.004
    mb.poly(DK, fr, [(o.a - ww / 2, y0, c), (o.a + ww / 2, y0, c), (o.a + ww / 2, y1, c), (o.a - ww / 2, y1, c)], hint=(0, 0, 1))
    for (x0_, x1_, y0_, y1_) in ((o.a - ww / 2 - 0.08, o.a - ww / 2, y0 - 0.08, y1 + 0.08), (o.a + ww / 2, o.a + ww / 2 + 0.08, y0 - 0.08, y1 + 0.08),
                                 (o.a - ww / 2, o.a + ww / 2, y0 - 0.08, y0), (o.a - ww / 2, o.a + ww / 2, y1, y1 + 0.08)):
        mb.box(SS, fr, x0_, x1_, y0_, y1_, c, c + 0.05, faces='xXyYZ')
    for x in np.linspace(o.a - ww / 2, o.a + ww / 2, 6)[1:-1]:
        mb.box(FE, fr, x - 0.012, x + 0.012, y0, y1, c + 0.02, c + 0.045, faces='xXZ')


# =========================================================================================== Staffelgiebel
def stepped_gable(fac, a0, a1, yb, pins, tops, field_rows=None, T=0.55, r=0.40, c0=-0.14, crown=0.40, rows_h=None,
                  windows=None, mat=BG, corner_pins=True, base_ledge=True, niche_mat='AR_Backstein_Nische', dark_fields=(), dark_from=1, shaft_over=0.55):
    """Staffelgiebel nach Foto 01 (Marktgiebel): Lisenen dreieckig vortretend (uebereck gestellte Quadratschaefte, Spitze
    nach aussen), die ueber die Stufen hinaus als Fialen weiterlaufen; je Stufe (Geschoss des Giebels) und Feld EINE Reihe:
    unten zwei spitzbogige Blendnischen, darueber zwei hohe gruen glasierte Masswerktafeln, oben ein Gurt.
    pins: Lisenen-/Fialenpositionen (a), tops: Feldoberkanten zwischen den Lisenen (len(pins)-1).
    dark_fields: Feldindizes, deren Nischen ab Stufe dark_from echte (dunkle) Dachbodenfenster sind.
    windows: dict feld -> {stufe: Fabrik(ca, cb, y0, y1) -> [Opening]} | 'skip' (eigene Fenster statt Nischen)."""
    fr = fac.fr
    # Treppenumriss (a, b)
    pts = [(a0, yb - 0.02), (a1, yb - 0.02)]
    for k in range(len(tops) - 1, -1, -1):
        pa, pb = pins[k], pins[k + 1]
        xa = a0 if k == 0 else pa; xb = a1 if k == len(tops) - 1 else pb
        pts.append((xb, tops[k])); pts.append((xa, tops[k]))
    outline = np.array(pts)
    levels = [yb] + sorted(set(round(t, 3) for t in tops))
    # Feldinhalte -> Ausschnitte
    cuts = []
    sx = r * 0.70                      # halbe sichtbare Lisenenbreite in der Wandebene (Raute schneidet c=0)
    for k in range(len(tops)):
        fa = (a0 if k == 0 else pins[k]) ; fb = (a1 if k == len(tops) - 1 else pins[k + 1])
        ca = pins[k] + sx + 0.06 if k > 0 or corner_pins else fa + 0.1
        cb = pins[k + 1] - sx - 0.06 if k < len(tops) - 1 or corner_pins else fb - 0.1
        cw = cb - ca
        if cw < 0.35: continue
        tiers = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1) if levels[i + 1] <= tops[k] + 1e-6]
        for row, (t0, t1) in enumerate(tiers):
            ht = t1 - t0
            spec = (windows or {}).get(k, {}).get(row)
            if spec == 'skip':
                pass
            elif spec is not None:
                for o in spec(ca, cb, t0, t1):
                    cuts.append(build_opening(mb, fr, o, protos=True))
            else:
                nw = 2 if cw > 0.62 else 1
                nwid = min(0.30, (cw - 0.12 * (nw + 1)) / nw)
                gap = (cw - nw * nwid) / (nw + 1)
                y = t0 + (0.30 if row == 0 else 0.18)
                hn = min(1.30, 0.50 * ht)                     # Nische inkl. Bogen
                dark = k in dark_fields and row >= dark_from
                for i in range(nw):
                    ax = ca + gap * (i + 1) + nwid * (i + 0.5)
                    o = Opening(ax, y, nwid, hn - nwid * 0.62, 'point', rise=nwid * 0.62, orders=((0, 0.20 if dark else 0.14),),
                                content='dark' if dark else 'niche', sill=False, n=8, nj=1, back_mat=niche_mat)
                    cuts.append(build_opening(mb, fr, o))
                    if dark:   # Gitter im Dachbodenfenster
                        mb.box(FE, fr, ax - 0.008, ax + 0.008, y, y + hn - nwid * 0.30, -0.19, -0.17, faces='xXZ')
                    else:      # Dreipass-Nasen im Nischenkopf (Foto 01)
                        trefoil_head(fr, o, -0.11, -0.015)
                    # gruenes Masswerkpaneel darueber (hoch, schmal: Foto 01)
                    py0 = y + hn + 0.14; py1 = t1 - 0.30
                    if py1 - py0 > 0.22:
                        x0, x1 = ax - nwid * 0.46, ax + nwid * 0.46
                        uv = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
                        mb.tri_local(TR, fr, np.array([[(x0, py0, -0.035), (x1, py0, -0.035), (x1, py1, -0.035)], [(x0, py0, -0.035), (x1, py1, -0.035), (x0, py1, -0.035)]]), uv=uv)
                        side_band(mb, BK, fr, np.array([(x0, py0), (x1, py0), (x1, py1), (x0, py1)]), -0.035, 0.0, skip_bottom=False)
                        cuts.append(np.array([(x0, py0), (x1, py0), (x1, py1), (x0, py1)]))
                        # Spitzbogige Abdeckung (Formstein-Ueberschlag) ueber der Tafel
                        mb.box(BK, fr, x0 - 0.03, x1 + 0.03, py1, py1 + 0.06, 0.0, 0.03, faces='xXyYZ')
            # Gurt (Stufengesims) an der Stufenoberkante, wenn das Feld hoeher weitergeht
            if t1 < tops[k] - 0.1:
                mb.box(BK, fr, ca - 0.12, cb + 0.12, t1 - 0.10, t1 - 0.02, 0.0, 0.06, faces='YyZ')
                mb.box(BR, fr, ca - 0.12, cb + 0.12, t1 - 0.02, t1 + 0.06, 0.0, 0.03, faces='YZ')
    # Scheibe: Front mit Ausschnitten, Rueckseite, Oberseiten
    wall_face(mb, fr, outline, cuts, mat)
    mb.extrude(mat, fr, outline, -T, 0.0, front=False, back=True, sides=True)
    # Abdeckungen (Schiefer, Satteldach-Profil) auf den Feldoberkanten
    for k in range(len(tops)):
        xa = a0 if k == 0 else pins[k]; xb = a1 if k == len(tops) - 1 else pins[k + 1]
        h = tops[k]
        prof = [(-T - 0.06, h - 0.03), (0.06, h - 0.03), (0.06, h + 0.03), (-T / 2, h + 0.17), (-T - 0.06, h + 0.03)]
        _cope(fr, xa, xb, prof)
    # Lisenen/Fialenschaefte (Raute, Spitze nach aussen) + Kronen; Gesimsringe auf Stufenhoehe
    for i, a in enumerate(pins):
        adj = [tops[j] for j in (i - 1, i) if 0 <= j < len(tops)]
        ytop = max(adj) + shaft_over
        pinnacle_shaft(mb, fr, a, yb, ytop, r=r, c=c0, mat=BB, corbel=0.55, crown_scale=crown, square=False)
        for lv in levels[1:]:
            if lv < ytop - 0.3:
                f_ = fr.local(a, 0, c0)
                rr = r + 0.05
                mb.prism(BK, f_, np.array([[rr, 0], [0, rr], [-rr, 0]]), lv - 0.13, lv + 0.01, top=True, bottom=True)   # nur vordere Haelfte (Rest steckt in der Wand)
    # Sockelgesims (Schiefer-Wasserschlag) an der Giebelbasis
    if base_ledge:
        prof = [(0.0, yb - 0.10), (0.16, yb - 0.10), (0.16, yb - 0.04), (0.0, yb + 0.10)]
        _cope(fr, a0, a1, prof, mat=SL)
    return max(tops)


def trefoil_head(fr, o, c_back, c_front, mat=BR):
    """Dreipass im Spitzbogenkopf einer Oeffnung o: Restflaeche zwischen Bogen und drei Paessen (zwei seitliche, ein
    oberer) als duenne Formsteinplatte (c_back..c_front) -> die typischen 'Nasen' der Blendnischen."""
    hw = o.w / 2; ys = o.y0 + o.hs; r = o.rise
    out = o.outline(0)
    head = Polygon([tuple(p) for p in out]).intersection(sbox(o.a - hw - 0.01, ys - 0.06, o.a + hw + 0.01, ys + r + 0.05))
    lobes = unary_union([Point(o.a - hw * 0.42, ys).buffer(hw * 0.58, quad_segs=5),
                         Point(o.a + hw * 0.42, ys).buffer(hw * 0.58, quad_segs=5),
                         Point(o.a, ys + r * 0.50).buffer(hw * 0.46, quad_segs=5),
                         sbox(o.a - hw * 0.9, ys - 0.2, o.a + hw * 0.9, ys)])
    rest = head.difference(lobes).buffer(0)
    for g in (rest.geoms if hasattr(rest, 'geoms') else [rest]):
        if g.geom_type != 'Polygon' or g.area < 4e-4: continue
        g = g.simplify(0.004)
        pts = np.array(g.exterior.coords)[:-1]
        if len(pts) < 3: continue
        mb.extrude(mat, fr, pts, c_back, c_front, front=True, back=False, sides=True)


def _cope(fr, xa, xb, prof, mat=SL):
    """Profil in (c, y) entlang a von xa bis xb (Fassadenframe) -- Kappen an beiden Enden"""
    prof = np.asarray(prof, float)
    k = len(prof)
    ring_a = [(xa, p[1], p[0]) for p in prof]; ring_b = [(xb, p[1], p[0]) for p in prof]
    for i in range(k):
        j = (i + 1) % k
        mb.poly(mat, fr, [ring_a[i], ring_b[i], ring_b[j], ring_a[j]], hint=_pnorm(prof, i))
    mb.poly(mat, fr, ring_a, hint=(-1, 0, 0)); mb.poly(mat, fr, ring_b, hint=(1, 0, 0))


def _pnorm(prof, i):
    p, q = prof[i], prof[(i + 1) % len(prof)]
    d = q - p
    # Profil im Uhrzeigersinn? Normale = rechts der Laufrichtung in (c, y) -> (dy, -dc)
    cen = prof.mean(0)
    n = np.array([d[1], -d[0]])
    mid = (p + q) / 2
    if np.dot(n, mid - cen) < 0: n = -n
    return (0, n[1], n[0])


# =========================================================================================== Zwerchhaus (3 Fialen)
def dormer(fac, a, ye, w=3.0, ytop=13.4, slope_main=59.2, roofline=None, wins=2, mat=BR, pin_r=0.30, crown=0.30, fig=True):
    """Zwerchhaus in der Fassadenflucht: Front von ye bis ytop, drei Fialen, kleines Satteldach nach hinten."""
    fr = fac.fr
    hw = w / 2
    a0, a1 = a - hw, a + hw
    T = 0.45
    yg = ytop + 0.75       # Giebelspitzen der Wimperge
    # Frontumriss: Rechteck + zwei Wimperge zwischen den Fialen
    yo = max(ye, max(p[1] for p in fac.outline))      # Fassadenwand reicht bis yo -> Zwerchhausfront beginnt dort
    outline = [(a0, yo), (a1, yo), (a1, ytop), (a + hw / 2, yg), (a, ytop), (a - hw / 2, yg), (a0, ytop)]
    cuts = []
    # Fenster: zwei zweibahnige Spitzbogenfenster (Foto 05: ~10.7-11.8 m)
    wy = ye + 0.85
    for i in range(wins):
        ax = a + (i - (wins - 1) / 2) * min(1.05, w * 0.34)
        o = Opening(ax, wy, 0.70, 0.95, 'point', rise=0.34, ring={'w': 0.09, 'proj': 0.012, 'alt': True}, lights=2, bars_h=0.33,
                    bars_v=0, orders=((0, 0.08), (0.06, 0.24)), nj=3)
        cuts.append(build_opening(mb, fr, o))
    # Figurennischen (zwei Standfiguren) und Masswerktafeln unter der Krone
    ny = wy + 0.95 + 0.34 + 0.45
    nh = ytop - 0.95 - ny
    if fig and nh > 0.9:
        for i in range(2):
            ax = a + (i - 0.5) * min(1.05, w * 0.34)
            nw_ = 0.58
            o = Opening(ax, ny, nw_, nh - 0.30, 'point', rise=0.30, orders=((0, 0.26),), content='niche', sill=False, n=8, nj=2, back_mat=BK)
            cuts.append(build_opening(mb, fr, o))
            # Konsole + Figur (instanziert)
            mb.box(SS, fr, ax - 0.20, ax + 0.20, ny, ny + 0.10, -0.26, -0.02, faces='xXYZ')
            Hf = min(1.75, nh - 0.30 - 0.05)
            f_ = fr.local(ax, ny + 0.10, -0.13)
            M = frame_matrix(f_, 1.0); M[:3, :3] = f_.M @ np.diag([0.80 * Hf, Hf, 0.62 * Hf])
            mb.instance('figure', M)
        # Masswerktafeln (rot/weiss bzw. gruen) im obersten Feld
        for i, tm in enumerate((TRR, TR)):
            ax = a + (i - 0.5) * min(1.05, w * 0.34)
            x0, x1 = ax - 0.30, ax + 0.30; py0, py1 = ytop - 0.80, ytop - 0.22
            uv = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
            mb.tri_local(tm, fr, np.array([[(x0, py0, -0.04), (x1, py0, -0.04), (x1, py1, -0.04)], [(x0, py0, -0.04), (x1, py1, -0.04), (x0, py1, -0.04)]]), uv=uv)
            side_band(mb, BK, fr, np.array([(x0, py0), (x1, py0), (x1, py1), (x0, py1)]), -0.04, 0.0, skip_bottom=False)
            cuts.append(np.array([(x0, py0), (x1, py0), (x1, py1), (x0, py1)]))
    wall_face(mb, fr, outline, cuts, mat)
    mb.extrude(mat, fr, outline, -T, 0.0, front=False, back=True, sides=True)
    # Wimpergabdeckung
    for (xa, xb, up) in ((a - hw + 0.2, a - hw / 2, True), (a - hw / 2, a - 0.2, False), (a + 0.2, a + hw / 2, True), (a + hw / 2, a + hw - 0.2, False)):
        pass
    # Fialen
    for ax, yt in ((a0 + 0.05, ytop + 0.55), (a, ytop + 0.95), (a1 - 0.05, ytop + 0.55)):
        pinnacle_shaft(mb, fr, ax, ye + 0.2, yt, r=pin_r, c=-0.12, mat=BB, corbel=0.35, crown_scale=crown)
    # Dach: Wangen + Satteldach nach hinten (durchdringt das Hauptdach)
    rl = roofline if roofline is not None else ye + 0.35
    tanm = math.tan(math.radians(slope_main))
    yr = ytop - 0.05                     # Firsthoehe Zwerchhausdach
    yc = ytop - 1.30                     # Traufe des Zwerchhausdachs (Wangenoberkante)
    depth = (yr - rl) / tanm + 0.6       # bis hinter den Schnitt mit dem Hauptdach
    for sx in (-1, 1):
        x = a + sx * (hw - 0.02)
        # Wange (Backstein), Innenseite unsichtbar
        q = [(0.0 if sx < 0 else 0.0)]
        pts = [(x, ye - 0.2, -T), (x, ye - 0.2, -depth), (x, yc, -depth), (x, yc, -T)]
        if sx > 0:
            mb.poly(mat, fr, pts, hint=(1, 0, 0))
        else:
            mb.poly(mat, fr, pts, hint=(-1, 0, 0))
    # Dachflaechen (Ziegel), leichter Ueberstand seitlich
    for sx in (-1, 1):
        xe = a + sx * (hw + 0.12); ye_ = yc - 0.10
        quad = [(xe, ye_, -T + 0.02), (a, yr, -T + 0.02), (a, yr, -depth), (xe, ye_, -depth)]
        hint = (sx * 0.8, 0.6, 0)
        mb.poly(RF, fr, quad, hint=hint)
        # Untersicht
        mb.poly(DK, fr, [(xe, ye_ - 0.02, -T + 0.02), (a + sx * (hw - 0.02), yc - 0.02, -T + 0.02), (a + sx * (hw - 0.02), yc - 0.02, -depth), (xe, ye_ - 0.02, -depth)], hint=(0, -1, 0))
        # Traufkante (Stirn zwischen Ziegelkante und Untersicht, 3 cm): sonst offener Spalt, durch den man flach auf die
        # Rueckseite der Untersicht sieht (Strahltest)
        mb.poly(DK, fr, [(xe, ye_ - 0.03, -T + 0.02), (xe, ye_ + 0.01, -T + 0.02), (xe, ye_ + 0.01, -depth), (xe, ye_ - 0.03, -depth)], hint=(sx, 0, 0))
    # Firstziegel
    mb.box(RG, fr, a - 0.09, a + 0.09, yr - 0.02, yr + 0.09, -depth, -T + 0.02, faces='xXYZ')
    return yg


# =========================================================================================== Dachflaechen-Solver
class RoofSet:
    def __init__(self):
        self.planes = []      # dict(region, eave_pt, eave_dir, slope, y, name, ridge)

    def add(self, name, region, eave_p0, eave_p1, yline, slope, wing):
        """Dachebene: Traufkante eave_p0->eave_p1 (Planpunkte an der Wandflucht) mit Hoehe yline dort; steigt nach innen"""
        e = U(np.asarray(eave_p1) - np.asarray(eave_p0))
        n_in = np.array([e[1], -e[0]])
        cen = np.array(region.representative_point().coords[0])
        if np.dot(cen - eave_p0, n_in) < 0: n_in = -n_in
        tn = math.tan(math.radians(slope))
        # h(x,z) = yline + tn * (p - p0)·n_in
        A, B = tn * n_in[0], tn * n_in[1]; C = yline - A * eave_p0[0] - B * eave_p0[1]
        self.planes.append(dict(name=name, region=region, p0=np.asarray(eave_p0, float), dir=e, y=yline, slope=slope, abc=(A, B, C), wing=wing))

    def solve(self):
        BIG = 400.0
        for i, pi in enumerate(self.planes):
            reg = pi['region']
            for j, pj in enumerate(self.planes):
                if i == j or not reg.intersects(pj['region']): continue
                if pi['wing'] == pj['wing']: continue
                Ai, Bi, Ci = pi['abc']; Aj, Bj, Cj = pj['abc']
                a, b, c = Aj - Ai, Bj - Bi, Cj - Ci          # >0 wo j hoeher
                hp = halfplane(a, b, c + 0.002, BIG)
                if hp is None: continue
                reg = reg.difference(pj['region'].intersection(hp))
            # gleicher Fluegel: Ebenen schneiden sich am First (Regionen sind schon getrennt)
            pi['vis'] = reg.buffer(0)

    def build(self):
        for p in self.planes:
            v = p['vis']
            if v.is_empty: continue
            roof_plane(mb, v, p['p0'], p['dir'], p['slope'], p['y'])

    def height(self, x, z, wing=None):
        best = -1e9
        for p in self.planes:
            if wing and p['wing'] != wing: continue
            if p['region'].contains(Point(x, z)):
                A, B, C = p['abc']; best = max(best, A * x + B * z + C)
        return best


def halfplane(a, b, c, BIG):
    """Polygon {a x + b z + c > 0} (grosses Quadrat)"""
    g = math.hypot(a, b)
    if g < 1e-9:
        return sbox(-1e4, -1e4, 1e4, 1e4) if c > 0 else None
    n = np.array([a, b]) / g
    p0 = -c / g * n
    t = np.array([-n[1], n[0]])
    BIGL = 1e4
    pts = [p0 - t * BIGL, p0 + t * BIGL, p0 + t * BIGL + n * BIGL, p0 - t * BIGL + n * BIGL]
    return Polygon(pts)


def strip(p0, p1, d0, d1):
    """Planviereck entlang p0->p1 zwischen Abstaenden d0..d1 nach aussen (n = links der Richtung)"""
    t = U(np.asarray(p1) - np.asarray(p0)); n = np.array([-t[1], t[0]])
    return Polygon([p0 + n * d0, p1 + n * d0, p1 + n * d1, p0 + n * d1])


def eave_detail(p0, p1, y_wall_top, roofline, slope, oh=0.38, gut=True, name=''):
    """Traufe: Untersicht, Stirnbrett, Rinne entlang p0->p1 (Wandflucht, von aussen links->rechts)"""
    t = U(np.asarray(p1) - np.asarray(p0)); n = np.array([-t[1], t[0]])
    tn = math.tan(math.radians(slope))
    y_edge = roofline - oh * tn                 # Dachflaeche an der Traufkante
    y_sof = y_edge - 0.10
    fr = Frame.facade(p0, p1)
    L = float(np.linalg.norm(np.asarray(p1) - p0))
    # Untersicht (Normale nach unten), Stirnbrett (aussen), Kappen
    mb.poly(PL, fr, [(0, y_sof, -0.05), (L, y_sof, -0.05), (L, y_sof, oh), (0, y_sof, oh)], hint=(0, -1, 0))
    mb.box(DR, fr, 0, L, y_sof, y_edge + 0.01, oh, oh + 0.03, faces='xXyYZ')
    for a_, sg in ((0.0, -1), (L, 1)):
        mb.poly(PL, fr, [(a_, y_sof, -0.05), (a_, y_sof, oh), (a_, y_edge + 0.01, oh), (a_, roofline + 0.05, -0.05)], hint=(sg, 0, 0))
    if gut:
        gutter(mb, [tuple(np.asarray(p0) + 0 * n), tuple(np.asarray(p1))], y_edge - 0.02, out=oh + 0.03, r=0.08)
    return y_edge


# =========================================================================================== Boden-/Sockel
def plinth(fac, y_top=0.55, proj=0.08, a0=0.0, a1=None, ext0=0.0, ext1=0.0, mat=SS):
    a1 = fac.L if a1 is None else a1
    lo = sorted(fac.low)
    segs = []; cur = a0 - ext0
    for (l0, l1) in lo:
        if l1 <= cur or l0 >= a1 + ext1: continue
        if l0 > cur + 0.05: segs.append((cur, l0))
        cur = max(cur, l1)
    if a1 + ext1 > cur + 0.05: segs.append((cur, a1 + ext1))
    for (s0, s1) in segs:
        fr = fac.fr
        yt = max(fac.gmax + 0.35, y_top)
        mb.box(mat, fr, s0, s1, YB, yt - 0.07, 0.0, proj, faces='xXYZ')
        # Fase
        mb.quad(mat, fr, (s0, yt - 0.07, proj), (s1, yt - 0.07, proj), (s1, yt, 0.0), (s0, yt, 0.0))
        mb.poly(mat, fr, [(s0, yt - 0.07, 0.0), (s0, yt - 0.07, proj), (s0, yt, 0.0)], hint=(-1, 0, 0))
        mb.poly(mat, fr, [(s1, yt - 0.07, proj), (s1, yt - 0.07, 0.0), (s1, yt, 0.0)], hint=(1, 0, 0))


def colbox_wall(fac, a0, a1, y0, y1, c0=-0.9, c1=0.0):
    col.box('AR_col', fac.fr, a0, a1, y0, y1, c0, c1)


# ###########################################################################################
# 1) MARKTFLUEGEL  (F1 NW-Fassade N->W, F2c NO-Giebel N2->N, F4a SW-Giebel W->Wb)
# ###########################################################################################
F1 = Fac('F1_markt', N, W, MF_EAVE + 0.35)
g1 = F1.g
# Erdgeschoss (Hochparterre) -- Fenster, Kellerboegen, Eingangsbereich mit Treppen
for a, li, w in ((2.40, 2, 1.15), (6.85, 3, 1.70), (22.45, 2, 1.15), (24.95, 3, 1.70), (27.85, 3, 1.70), (30.35, 3, 1.70), (33.05, 3, 1.70)):
    F1.op(win_ground(a, w=w, lights=li))
for a in (6.85, 10.40, 24.95, 27.85, 30.35, 33.05):
    cellar_window(F1, F1.op(cellar(a, w=1.70 if a in (10.40, 16.85) else 1.80, g=g1), low=True), g1)
# Foto 05: zwischen den beiden Treppenlaeufen ein niedriger rundbogiger Kellereingang mit dunkler Holztuer
F1.op(Opening(16.85, g1 - 0.02, 1.10, 0.92, 'round', ring={'w': 0.13, 'proj': 0.0, 'alt': True}, content='door', door_mat=DR,
              orders=((0, 0.12), (0.08, 0.40)), skip_bottom=True, nj=3, n=14), low=True)
# Eingangstueren (erhoehtes EG, Treppen davor)
DOORS_F1 = (14.30, 19.40)
for a in DOORS_F1:
    F1.op(Opening(a, 1.35 + g1, 1.45, 2.05, 'point', rise=0.75, ring={'w': 0.15, 'proj': 0.015}, content='door', orders=((0, 0.12), (0.10, 0.45)), nj=4, door_mat=DR), low=True)
F1.frieze(4.55, 0.55)
# Obergeschoss: 15 Fenster mit Masswerk-Tympanon
UP_A = (2.40, 4.42, 6.45, 8.50, 10.55, 12.58, 14.62, 17.05, 19.48, 21.52, 23.57, 25.62, 27.66, 29.71, 31.78)
for a in UP_A:
    F1.op(win_up(a, w=1.70 if abs(a - 17.05) < 0.1 else 1.18, lights=3 if abs(a - 17.05) < 0.1 else 2))
F1.lattice(9.08, 9.52)
F1.finish()
plinth(F1, y_top=0.55 + g1 * 0, ext1=0.08)

# Treppenanlage + Vordach (Glas/Stahl) vor den Tueren
def stair(fac, a, y_land, width=1.9, run=0.30, depth_land=0.7, g=0.0):
    fr = fac.fr
    nst = max(1, int(round((y_land - g) / 0.155)))
    rise = (y_land - g) / nst
    for i in range(nst):
        y1 = g + rise * (i + 1)
        c1 = depth_land + run * (nst - i)            # Vorderkante dieser Stufe
        mb.box(GR, fr, a - width / 2, a + width / 2, YB, y1, 0.0, c1, faces='YZ')      # Seiten liegen in den Wangen
    # Wangen + Gelaender
    for sx in (-1, 1):
        x = a + sx * (width / 2 + 0.06)
        mb.box(SS, fr, x - 0.12, x + 0.12, YB, y_land + 0.15, 0.0, depth_land + run * nst + 0.05, faces='xXYZ')
        top = [(x, y_land + 1.05, 0.15), (x, g + 1.0, depth_land + run * nst)]
        for (p, q) in ((top[0], top[1]),):
            # Handlauf (Edelstahl) als duenner Quader entlang der Schraege
            v = np.array(q) - np.array(p); L = np.linalg.norm(v)
            f2 = Frame(fr.P(p), fr.D([0, v[1] / L, v[2] / L]) if False else fr.ez, fr.ey, fr.ex)
        for cc in np.linspace(0.3, depth_land + run * nst - 0.1, 4):
            yb_ = y_land + 0.15 if cc < depth_land else g + (y_land - g) * (1 - (cc - depth_land) / (run * nst)) + 0.05
            mb.box(ZN, fr, x - 0.02, x + 0.02, yb_, yb_ + 0.95, cc - 0.02, cc + 0.02, faces='xXzZ')
        # Handlauf in Stuecken
        pts = [(0.25, y_land + 1.10)] + [(cc, (g + (y_land - g) * (1 - (cc - depth_land) / (run * nst)) if cc > depth_land else y_land) + 1.00) for cc in np.linspace(depth_land, depth_land + run * nst - 0.1, 5)]
        for (c_a, y_a), (c_b, y_b) in zip(pts[:-1], pts[1:]):
            mb.poly(ZN, fr, [(x - 0.025, y_a, c_a), (x + 0.025, y_a, c_a), (x + 0.025, y_b, c_b), (x - 0.025, y_b, c_b)], hint=(0, 1, 0))
            mb.poly(ZN, fr, [(x + 0.025, y_a - 0.05, c_a), (x - 0.025, y_a - 0.05, c_a), (x - 0.025, y_b - 0.05, c_b), (x + 0.025, y_b - 0.05, c_b)], hint=(0, -1, 0))
            mb.poly(ZN, fr, [(x + sgn * 0.025, y_a - 0.05, c_a) for sgn in (1,)] + [(x + 0.025, y_b - 0.05, c_b), (x + 0.025, y_b, c_b), (x + 0.025, y_a, c_a)], hint=(1, 0, 0))
            mb.poly(ZN, fr, [(x - 0.025, y_a, c_a), (x - 0.025, y_b, c_b), (x - 0.025, y_b - 0.05, c_b), (x - 0.025, y_a - 0.05, c_a)], hint=(-1, 0, 0))
    # Kollision: Rampe als schraeger Quader
    L = depth_land + run * nst
    col.poly('AR_col', fr, [(a - width / 2, y_land, 0.0), (a + width / 2, y_land, 0.0), (a + width / 2, y_land, depth_land), (a - width / 2, y_land, depth_land)], hint=(0, 1, 0))
    col.poly('AR_col', fr, [(a - width / 2, y_land, depth_land), (a + width / 2, y_land, depth_land), (a + width / 2, g, L), (a - width / 2, g, L)], hint=(0, 1, 0))


for a in DOORS_F1:
    stair(F1, a, 1.35 + g1, g=g1)


def canopy(fac, a0, a1, y, depth, posts=True):
    fr = fac.fr
    # Glasscheibe (durchsichtig, Spiel: RX_GLASS) 2 mm ueber den Traegern (sonst koplanar mit deren Oberseite)
    mb.box('AR_Vordach_Glas', fr, a0 + 0.055, a1 - 0.055, y + 0.002, y + 0.03, 0.05, depth, faces='xXyYzZ')
    for c_ in (0.05, depth - 0.12):
        mb.box(ST, fr, a0, a1, y - 0.18, y, c_, c_ + 0.12, faces='xXyYzZ')
    for a_ in (a0, a1):
        mb.box(ST, fr, a_ - 0.06, a_ + 0.06, y - 0.18, y + 0.02, 0.05, depth, faces='xXyYzZ')
        if posts:
            mb.box(ST, fr, a_ - 0.06, a_ + 0.06, YB + 0.5, y - 0.18, depth - 0.18, depth - 0.06, faces='xXzZ')
            col.box('AR_col', fr, a_ - 0.06, a_ + 0.06, -0.5, y, depth - 0.18, depth - 0.06)


canopy(F1, 12.75, 20.95, 4.25 + g1, 3.4)

# NO-Giebel des Marktfluegels (F2c) --------------------------------------------------------
F2c = Fac('F2c_marktgiebel', N2, N, MF_EAVE - 0.02)          # endet am Giebelfuss
g2c = F2c.g
F2c.op(Opening(6.70, 1.80 + g2c, 2.15, 1.95, 'point', rise=1.25, ring={'w': 0.34, 'proj': 0.02}, lights=4, bars_h=0.40,
               orders=((0, 0.10), (0.10, 0.22), (0.20, 0.38)), nj=6))
for a in (1.75, 11.65):
    F2c.op(win_ground(a, y0=2.25 + g2c, w=1.55, hs=1.45))
    F2c.op(Opening(a, g2c - 0.05, 2.05, 1.30, 'seg', rise=0.55, ring={'w': 0.14, 'proj': 0.0}, content='niche', back_mat=BR,
                   orders=((0, 0.16),), sill=False, nj=2), low=True)
F2c.frieze(4.30, 0.52)
for a in (1.78, 4.30, 6.70, 9.10, 11.62):
    F2c.op(win_up(a, y0=5.20, hs=1.75, rise=0.72, w=1.15))
F2c.lattice(9.08, 9.52)
F2c.finish()
plinth(F2c, ext1=0.08)
# Staffelgiebel (10 Fialen, 9 Felder)
L2c = F2c.L
pins_mf = list(np.linspace(0.30, L2c - 0.30, 10))
pins_mf[4] = L2c / 2 - 0.50; pins_mf[5] = L2c / 2 + 0.50
tops_mf = [13.0, 15.6, 18.2, 20.8, 23.6, 20.8, 18.2, 15.6, 13.0]
gt1 = stepped_gable(F2c, 0.0, L2c, MF_EAVE, pins_mf, tops_mf, dark_fields=(3, 4, 5))

# SW-Giebel des Marktfluegels (F4a) mit Laube an der Westecke -----------------------------
F4a = Fac('F4a_sw_giebel', W, Wb, MF_EAVE - 0.02)
g4a = F4a.g
LAUBE = (0.85, 4.70)       # Laube (a-Bereich auf F4a), springt 3.5 m vor
F4a.op(win_ground(7.10, y0=2.05 + g4a, w=1.70))
F4a.op(win_ground(10.60, y0=2.05 + g4a, w=1.70))
cellar_window(F4a, F4a.op(cellar(7.10, g=g4a), low=True), g4a); cellar_window(F4a, F4a.op(cellar(10.60, g=g4a), low=True), g4a)
# Durchgang in der Laube (Tuer hinter dem Bogen)
F4a.op(Opening(2.80, 0.05 + g4a, 1.30, 2.10, 'point', rise=0.65, ring={'w': 0.12, 'proj': 0.0}, content='door', orders=((0, 0.15), (0.10, 0.40)), nj=3), low=True)
F4a.frieze(4.55, 0.55, a0=LAUBE[1] + 0.05)
for a in (2.05, 4.55, 7.05, 9.55, 12.00):
    F4a.op(win_up(a, w=1.15))
F4a.lattice(9.08, 9.52)
F4a.finish()
plinth(F4a, ext1=0.0)
L4a = F4a.L
pins_sw = list(np.linspace(0.30, L4a - 0.30, 10))
pins_sw[4] = L4a / 2 - 0.50; pins_sw[5] = L4a / 2 + 0.50
stepped_gable(F4a, 0.0, L4a, MF_EAVE, pins_sw, tops_mf, dark_fields=(3, 4, 5))


def laube(fac, a0, a1, proj=3.50, h=6.10):
    """offene Laube (zwei Spitzboegen) vor dem SW-Giebel an der Westecke, Masswerkbruestung oben"""
    fr = fac.fr
    w = a1 - a0
    g = fac.g
    # Seitenflaechen als eigene Fassaden (lokale Frames)
    pA = P2(fac, a0, 0.0); pB = P2(fac, a1, 0.0); pA2 = P2(fac, a0, proj); pB2 = P2(fac, a1, proj)
    # Front (parallel zu F4a), links (Richtung Markt, parallel zu F1), rechts
    sides = [Fac('laube_front', pA2, pB2, h), Fac('laube_l', pA, pA2, h), Fac('laube_r', pB2, pB, h)]
    for k, s in enumerate(sides):
        Ls = s.L
        s.inner_holes = []
        if k == 0:
            o_ = s.op(Opening(Ls / 2, g - 0.02, Ls - 1.25, 1.85, 'point', rise=1.55, ring={'w': 0.30, 'proj': 0.02},
                              content='none', orders=((0, 0.20), (0.12, 0.55)), skip_bottom=True, nj=4), low=True)
            s.inner_holes.append(o_.outline(0.12))
        elif k == 1:
            o_ = s.op(Opening(Ls / 2 + 0.20, g - 0.02, Ls - 1.45, 1.95, 'point', rise=1.45, ring={'w': 0.30, 'proj': 0.02},
                              content='none', orders=((0, 0.20), (0.12, 0.55)), skip_bottom=True, nj=4), low=True)
            s.inner_holes.append(o_.outline(0.12))
        else:
            s.op(Opening(Ls / 2 - 0.25, g + 0.9, 1.10, 1.5, 'point', rise=0.7, ring={'w': 0.12, 'proj': 0.015}, lights=2, bars_h=0.4,
                         orders=((0, 0.12), (0.10, 0.4)), nj=3))
        s.course(4.30, h=0.12, proj=0.06)
        # Bruestung (Foto 06/07): Reihe von Masswerktafeln zwischen Lisenen, aussen je ein Rautenfeld
        py0, py1 = h - 1.30, h - 0.30
        npan = max(3, int(round((Ls - 0.14) / 0.62)))
        pw = (Ls - 0.14 * (npan + 1)) / npan
        for i in range(npan):
            x0 = 0.14 + i * (pw + 0.14); x1 = x0 + pw
            if i in (0, npan - 1) and npan > 2:
                s.lattice(py0, py1, a0=x0, a1=x1, recess=0.06, proj=0.02, pitch=0.22, sw=0.05)
            else:
                uv = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
                mb.tri_local(TRB, s.fr, np.array([[(x0, py0, -0.05), (x1, py0, -0.05), (x1, py1, -0.05)], [(x0, py0, -0.05), (x1, py1, -0.05), (x0, py1, -0.05)]]), uv=uv)
                side_band(mb, BR, s.fr, np.array([(x0, py0), (x1, py0), (x1, py1), (x0, py1)]), -0.05, 0.0, skip_bottom=False)
                s.cut([(x0, py0), (x1, py0), (x1, py1), (x0, py1)])
        s.course(py1 + 0.02, h=0.08, proj=0.04)
        s.finish()
    # Innenraum der Laube: Gewoelbedecke (flach, Putz), Boden, Innenwaende
    inner = Frame(fr.P([a0, 0, 0]), fr.ex, fr.ey, fr.ez)
    mb.poly('AR_Putz_innen', inner, [(0.55, 4.15, 0.0), (w - 0.55, 4.15, 0.0), (w - 0.55, 4.15, proj - 0.55), (0.55, 4.15, proj - 0.55)], hint=(0, -1, 0))
    gl_ = max(g, max(ground(*P2(fac, a0 + w * u, proj * v)) for u in (0.15, 0.5, 0.85) for v in (0.1, 0.5, 0.85))) + 0.02   # ueber dem Gelaende
    mb.poly(GR, inner, [(0.55, gl_, 0.0), (w - 0.55, gl_, 0.0), (w - 0.55, gl_, proj - 0.55), (0.55, gl_, proj - 0.55)], hint=(0, 1, 0))
    # Innenseiten der drei Laubenwaende (Ebene c = -0.55 im jeweiligen Seitenframe, Normale in die Laube), mit Bogenausschnitt
    for k, s in enumerate(sides):
        Ls = s.L
        a_lo, a_hi = (0.55, Ls - 0.55) if k == 0 else ((0.0, proj - 0.55) if k == 1 else (0.55, Ls))
        P_ = Polygon([(a_lo, YB), (a_hi, YB), (a_hi, 4.15), (a_lo, 4.15)])
        for cut in s.inner_holes:
            P_ = P_.difference(Polygon(cut).buffer(0))
        for gg in (P_.geoms if hasattr(P_, 'geoms') else [P_]):
            if gg.is_empty or gg.area < 1e-4: continue
            mb.poly('AR_Backstein_innen', s.fr, [(x, y, -0.55) for x, y in np.array(gg.exterior.coords)[:-1]],
                    holes=[[(x, y, -0.55) for x, y in np.array(h.coords)[:-1]] for h in gg.interiors], hint=(0, 0, -1))
    # Dach (flach, Blei) + Abdeckung
    mb.poly(BIT, inner, [(0, h - 0.25, 0), (w, h - 0.25, 0), (w, h - 0.25, proj), (0, h - 0.25, proj)], hint=(0, 1, 0))
    mb.box(SS, inner, -0.05, w + 0.05, h - 0.02, h + 0.10, proj - 0.30, proj + 0.05, faces='xXyYzZ')
    mb.box(SS, inner, -0.05, 0.25, h - 0.02, h + 0.10, -0.1, proj - 0.30, faces='xXyYZ')
    mb.box(SS, inner, w - 0.25, w + 0.05, h - 0.02, h + 0.10, -0.1, proj - 0.30, faces='xXyYZ')
    # Innenseiten der Bruestung (von oben sichtbar)
    for (q0, q1, q2, q3) in (((0.25, h - 0.25, proj - 0.30), (w - 0.25, h - 0.25, proj - 0.30), (w - 0.25, h - 0.02, proj - 0.30), (0.25, h - 0.02, proj - 0.30)),):
        mb.poly(BR, inner, [q0, q1, q2, q3], hint=(0, 0, -1))
    for x0, sg in ((0.25, 1), (w - 0.25, -1)):
        mb.poly(BR, inner, [(x0, h - 0.25, 0.0), (x0, h - 0.25, proj - 0.30), (x0, h - 0.02, proj - 0.30), (x0, h - 0.02, 0.0)], hint=(sg, 0, 0))
    for s in sides:
        plinth(s, ext1=0.0)
    # Kollision: Pfeiler
    col.box('AR_col', inner, 0, 0.55, -0.5, h, 0, proj); col.box('AR_col', inner, w - 0.55, w, -0.5, h, 0, proj)
    col.box('AR_col', inner, 0, w, 4.15, h, proj - 0.55, proj)


laube(F4a, *LAUBE)

# Dachtraufe Marktfluegel (NW) + Hofseite
tan_mf = (MF_RIDGE - MF_ROOFLINE) / 6.70
SLOPE_MF = math.degrees(math.atan(tan_mf))
eave_detail(N + dNE * 0.0, W, MF_EAVE, MF_ROOFLINE, SLOPE_MF)
# Zwerchhaeuser (3) auf der Marktseite
for a in (7.75, 17.90, 28.05):
    dormer(F1, a, MF_EAVE - 0.02, w=3.25, ytop=15.25, slope_main=SLOPE_MF, roofline=MF_ROOFLINE)   # Foto 05: Koerper bis ~15.7 m, Fialen ~16.6 m

# Fallrohre
for a in (0.55, 11.6, 24.0, 35.0):
    downpipe(mb, F1.fr, a, F1.g - 0.1, MF_EAVE - 0.35)

LOG.append('Marktfluegel: slope %.1f, gable top %.1f' % (SLOPE_MF, gt1))
exec(open(os.path.join(HERE, 'build_wings.py'), encoding='utf8').read())

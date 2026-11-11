"""Marktkirche -- Figuren am Westportal (Foto 16: links St. Georg mit dem Drachen, rechts Tod (Skelett) und Mann)
und Reliefplatten (Hoehenfeld-Netze) fuer Epitaphe / Wandreliefs.
Alles geschlossene Koerper im Kirchen-Lokalframe (u, v, h). Masse in Figur-Einheiten (Scheitelhoehe ~1) * S."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from mklib import UP, nrm, loft, lathe, sweep, tube, box, chamfer_box, extrude_profile, rect_prof, tri_poly, quads_to_tris, newell
from mk_luther import blob


def ring_prof(n=10, ax=1.0, ay=1.0):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[ax * np.cos(t), ay * np.sin(t)]


class Fig:
    """Lokales Figurensystem: foot, f = vorn (Blickrichtung), r = rechte Koerperseite, S = Massstab (m je Einheit)."""
    def __init__(self, B, foot, facing, S, mat, grp='det', turn=0.0):
        self.B = B; self.foot = np.asarray(foot, float)
        f = nrm([facing[0], facing[1], 0.0]); r = np.cross(f, UP)
        c, s = math.cos(turn), math.sin(turn)
        self.f = nrm(c * f + s * r); self.r = np.cross(self.f, UP)
        self.S = S; self.mat = mat; self.grp = grp

    def P(self, rr, ff, hh):
        return self.foot + self.S * (rr * self.r + ff * self.f + hh * UP)

    def V(self, rr, ff, hh):
        return nrm(rr * self.r + ff * self.f + hh * UP)

    def limb(self, pts, rad, seg=10, nref=None, mat=None, prof=None, grp=None):
        P = np.array([self.P(*p) for p in pts])
        rad = np.asarray(rad, float) * self.S
        if prof is None: prof = ring_prof(seg)
        if nref is None:
            d = nrm(P[-1] - P[0]); nref = UP if abs(d[2]) < 0.85 else self.f
        self.B.add(mat or self.mat, sweep(P, prof, nref, scale=list(rad)), grp or self.grp, closed=True)

    def ball(self, c, rx, ry, rz, nu=12, nv=8, disp=None, mat=None, axes=None, grp=None):
        ax, ay, az = axes if axes is not None else (self.r, self.f, UP)
        self.B.add(mat or self.mat, blob(self.P(*c), ax, ay, az, rx * self.S, ry * self.S, rz * self.S, nu, nv, disp), grp or self.grp, closed=True)

    def lathe_at(self, c, prof, seg=12, az=None, mat=None):
        prof = [(r * self.S, h * self.S) for r, h in prof]
        self.B.add(mat or self.mat, lathe(self.P(*c), prof, seg, az=az), self.grp, closed=True)

    def robe(self, prof, nseg=40, seed=0, folds=9, sharp=0.7, h_open=None, grp=None):
        """Gewand: Loft elliptischer Ringe (h, a, b, Faltenamplitude) mit Laengsfalten."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
        fph = rng.uniform(0, 2 * math.pi, folds); fk = rng.integers(5, 14, folds); fa = rng.uniform(0.4, 1.0, folds)
        rings = []
        for (h, a, b, amp) in prof:
            fold = sum(fa[j] * np.cos(fk[j] * t + fph[j] + 0.5 * h) for j in range(folds)) / folds * 2.2
            fold = np.sign(fold) * np.abs(fold) ** sharp
            rad = 1.0 + amp * fold
            rings.append(np.array([self.P(a * math.cos(q) * k, b * math.sin(q) * k, h) for q, k in zip(t, rad)]))
        self.B.add(self.mat, loft(rings), grp or self.grp, closed=True)

    def head(self, c, look, size, hair=0.07, nu=28, nv=18, beard=False, skull=False):
        """Kopf mit Gesicht (Nase, Brauen, Augenhoehlen, Mund, Kinn, Wangen, Ohren); skull=True -> Totenschaedel."""
        look = nrm(look); hr = nrm(np.cross(look, UP)); hu = np.cross(hr, look)

        def disp(th, ph):
            x, y, z = math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)
            fr = max(y, 0.0); d = 1.0
            if skull:
                d -= 0.26 * math.exp(-(((abs(x) - 0.36) / 0.17) ** 2 + ((z - 0.08) / 0.15) ** 2)) * fr       # Augenhoehlen tief
                d -= 0.18 * math.exp(-((x / 0.09) ** 2 + ((z + 0.18) / 0.1) ** 2)) * fr                      # Nasenloch
                d += 0.08 * math.exp(-(((abs(x) - 0.55) / 0.15) ** 2 + ((z + 0.12) / 0.12) ** 2))            # Jochbein
                d -= 0.12 * math.exp(-(((abs(x) - 0.62) / 0.2) ** 2 + ((z + 0.42) / 0.2) ** 2))              # Schlaefen eingefallen
                d -= 0.10 * max(-z - 0.45, 0) * 2.0                                                          # schmal nach unten
                return d
            d += 0.30 * math.exp(-((x / 0.11) ** 2 + ((z + 0.05) / 0.2) ** 2)) * fr ** 5                 # Nase
            d += 0.08 * math.exp(-((x / 0.2) ** 2 + ((z - 0.24) / 0.07) ** 2)) * fr ** 2                 # Brauenwulst Mitte
            d += 0.06 * math.exp(-(((abs(x) - 0.33) / 0.15) ** 2 + ((z - 0.25) / 0.06) ** 2)) * fr       # Brauen
            d -= 0.09 * math.exp(-(((abs(x) - 0.33) / 0.12) ** 2 + ((z - 0.12) / 0.08) ** 2)) * fr       # Augenhoehlen
            d += 0.03 * math.exp(-(((abs(x) - 0.33) / 0.06) ** 2 + ((z - 0.12) / 0.04) ** 2)) * fr       # Augapfel
            d -= 0.05 * math.exp(-((x / 0.22) ** 2 + ((z + 0.33) / 0.025) ** 2)) * fr                    # Mundspalte
            d += 0.04 * math.exp(-((x / 0.18) ** 2 + ((z + 0.28) / 0.05) ** 2)) * fr                     # Oberlippe
            d += 0.08 * math.exp(-((x / 0.25) ** 2 + ((z + 0.55) / 0.12) ** 2)) * (0.5 + 0.5 * fr)        # Kinn
            d += 0.05 * math.exp(-(((abs(x) - 0.52) / 0.18) ** 2 + ((z + 0.12) / 0.2) ** 2))             # Wangen
            d += 0.12 * math.exp(-((abs(x) - 0.97) / 0.08) ** 2 - ((z - 0.02) / 0.17) ** 2) * (y < 0.25)  # Ohren
            if beard and z < -0.15 and y > -0.2:
                d += 0.10 * (min(-z - 0.15, 0.4) / 0.4) * (0.8 + 0.2 * math.sin(9 * th) * math.sin(7 * ph))
            hr_ = ((z > -0.2) and (y < 0.15)) or (z > 0.55) or (z > 0.38 and y < 0.55)
            if hr_:
                d += hair * (1.0 + 0.45 * math.sin(8 * th + 2 * ph) * math.sin(6 * ph + 3 * th))
            return d
        self.B.add(self.mat, blob(self.P(*c), hr, look, hu, size * 0.8 * self.S, size * 0.92 * self.S, size * self.S, nu, nv, disp), self.grp, closed=True)
        return hr, hu


# ------------------------------------------------------------------ St. Georg mit dem Drachen
def george_dragon(B, foot, facing, S, mat, grp='det'):
    """Ritter in Plattenruestung, Lanze schraeg nach unten in den Rachen des Drachen zu seinen Fuessen (Foto 16 links)."""
    g = Fig(B, foot, facing, S, mat, grp, turn=math.radians(12))
    # Felssockel
    g.ball((0.0, 0.03, 0.02), 0.27, 0.21, 0.07, 16, 6, disp=lambda th, ph: 1.0 + 0.08 * math.sin(5 * th) * math.sin(3 * ph))
    # Beine (breite Stellung), Kniekacheln, Schuhe
    for sg, kn, an in ((1, (0.13, 0.05, 0.28), (0.15, 0.02, 0.06)), (-1, (-0.15, 0.07, 0.29), (-0.2, 0.05, 0.06))):
        hip = (sg * 0.085, 0.0, 0.5)
        g.limb([hip, kn, an], [0.06, 0.045, 0.036], 12)
        g.ball(kn, 0.05, 0.055, 0.045, 10, 6)
        g.ball((an[0], an[1] + 0.045, 0.035), 0.04, 0.085, 0.032, 10, 6)
    # Schurz (Bauchreifen) + Brustharnisch mit Grat
    lames = [(0.44, 0.17, 0.13), (0.475, 0.165, 0.125), (0.478, 0.155, 0.118), (0.515, 0.152, 0.115), (0.518, 0.142, 0.108),
             (0.555, 0.14, 0.105), (0.558, 0.13, 0.098), (0.6, 0.13, 0.098)]
    t = np.linspace(0, 2 * math.pi, 28, endpoint=False)
    g.B.add(mat, loft([np.array([g.P(a * math.cos(q), b * math.sin(q), h) for q in t]) for h, a, b in lames]), grp, closed=True)
    torso = [(0.59, 0.128, 0.096, 0.0), (0.65, 0.145, 0.11, 0.02), (0.72, 0.16, 0.115, 0.035), (0.78, 0.17, 0.105, 0.03),
             (0.82, 0.15, 0.09, 0.0), (0.845, 0.09, 0.07, 0.0), (0.86, 0.045, 0.045, 0.0)]
    rings = []
    for h, a, b, ridge in torso:
        rr = [g.P(a * math.cos(q), b * math.sin(q) + ridge * max(math.sin(q), 0) ** 8, h) for q in t]
        rings.append(np.array(rr))
    g.B.add(mat, loft(rings), grp, closed=True)
    g.limb([(-0.15, 0.0, 0.585), (0.0, 0.1, 0.578), (0.15, 0.0, 0.585)], [0.012, 0.014, 0.012], 6)       # Guertel vorn
    # Schulterstuecke
    for sg in (1, -1):
        g.ball((sg * 0.175, -0.005, 0.795), 0.075, 0.07, 0.055, 12, 7)
    # Lanze: oben rechts (ueber der Schulter) -> unten links in den Drachenrachen
    L0 = np.array([0.30, -0.06, 1.18]); L1 = np.array([-0.33, 0.33, 0.17])
    lp = lambda s: tuple(L0 + (L1 - L0) * s)
    g.limb([lp(0.0), lp(1.0)], [0.016, 0.016], 8)
    g.limb([lp(1.0), tuple(L0 + (L1 - L0) * 1.09)], [0.026, 0.0005], 6)
    g.limb([lp(0.33), lp(0.36)], [0.03, 0.03], 8)                                                          # Brechscheibe
    # Arme mit Panzerhandschuhen an der Lanze
    hR = lp(0.25); hL = lp(0.47)
    g.limb([(0.175, 0.0, 0.79), (0.25, 0.08, 0.68), hR], [0.048, 0.04, 0.034], 10)
    g.limb([(-0.175, 0.0, 0.79), (-0.15, 0.17, 0.64), hL], [0.048, 0.04, 0.034], 10)
    for h_ in (hR, hL):
        g.ball(h_, 0.04, 0.035, 0.035, 10, 6)
    g.ball((0.25, 0.08, 0.68), 0.04, 0.04, 0.04, 8, 6)
    g.ball((-0.15, 0.17, 0.64), 0.04, 0.04, 0.04, 8, 6)
    # Hals + Helm mit Visier und Kamm
    g.lathe_at((0.0, 0.0, 0.84), [(0.045, 0.0), (0.04, 0.05), (0.0, 0.06)], 10)
    hc = (0.0, 0.01, 0.95)
    g.ball(hc, 0.075, 0.085, 0.1, 18, 12, disp=lambda th, ph: 1.0 + 0.08 * max(math.sin(th), 0) * math.exp(-((math.cos(ph) + 0.1) / 0.3) ** 2))
    g.limb([(0.0, -0.07, 0.93), (0.0, -0.02, 1.06), (0.0, 0.07, 1.02)], [0.008, 0.016, 0.006], 6)          # Kamm
    for dz in (0.955, 0.925):                                                                                 # Sehschlitze (Grat)
        g.limb([(-0.055, 0.07, dz), (0.0, 0.098, dz + 0.005), (0.055, 0.07, dz)], [0.006, 0.007, 0.006], 6)
    # Schwert an der linken Huefte
    g.limb([(-0.15, -0.03, 0.6), (-0.2, -0.1, 0.24)], [0.022, 0.016], 8)
    g.limb([(-0.13, 0.0, 0.66), (-0.15, -0.03, 0.6)], [0.01, 0.01], 6)
    g.limb([(-0.19, 0.03, 0.62), (-0.11, -0.09, 0.6)], [0.009, 0.009], 6)
    # Drache: Schwanz hinten rechts -> Leib um die Fuesse -> Hals hoch -> Kopf mit offenem Rachen (Lanze darin)
    body = [(0.33, -0.2, 0.035), (0.2, -0.24, 0.05), (0.02, -0.22, 0.07), (-0.18, -0.14, 0.085), (-0.3, 0.0, 0.095),
            (-0.34, 0.14, 0.11), (-0.33, 0.25, 0.15), (-0.3, 0.31, 0.2)]
    rad = [0.008, 0.03, 0.06, 0.08, 0.085, 0.07, 0.055, 0.048]
    g.limb(body, rad, 12)
    for i in range(1, 7):                                                                                     # Rueckenzacken
        p = np.array(body[i]); q = np.array(body[i + 1]); m = (p + q) / 2
        g.limb([tuple(m + np.array([0, 0, rad[i] * 0.8])), tuple(m + np.array([0, 0, rad[i] * 0.8 + 0.05]))], [0.02, 0.001], 5)
    # Kopf: Oberkiefer (nach oben aufgerissen) + Unterkiefer, Augenwuelste, Hoerner
    hd = np.array(body[-1])
    g.limb([tuple(hd), tuple(hd + np.array([0.05, 0.07, 0.07])), tuple(hd + np.array([0.08, 0.1, 0.13]))], [0.045, 0.035, 0.012], 10)
    g.limb([tuple(hd), tuple(hd + np.array([0.06, 0.1, 0.0])), tuple(hd + np.array([0.09, 0.14, 0.02]))], [0.038, 0.028, 0.01], 10)
    for sg in (-1, 1):
        g.ball(tuple(hd + np.array([sg * 0.03 + 0.02, 0.03, 0.06])), 0.016, 0.016, 0.014, 6, 4)
        g.limb([tuple(hd + np.array([sg * 0.03, -0.02, 0.05])), tuple(hd + np.array([sg * 0.05, -0.08, 0.12]))], [0.012, 0.002], 5)
    # Vorderbeine mit Krallen
    for p0, p1 in (((-0.28, 0.2, 0.1), (-0.2, 0.3, 0.02)), ((-0.36, 0.08, 0.09), (-0.42, 0.18, 0.02))):
        g.limb([p0, p1], [0.03, 0.02], 8)
        for k in (-1, 0, 1):
            q = np.array(p1)
            g.limb([tuple(q), tuple(q + np.array([0.025 * k, 0.04, -0.01]))], [0.008, 0.001], 5)
    # Fluegel (Haut zwischen 4 Fingerknochen), rechts hinter dem Ritter aufsteigend
    root = np.array([0.1, -0.2, 0.09])
    tips = [np.array([0.36, -0.12, 0.52]), np.array([0.42, -0.02, 0.36]), np.array([0.4, 0.06, 0.22]), np.array([0.33, 0.1, 0.1])]
    for tp in tips:
        mid = root + (tp - root) * 0.55 + np.array([0.04, -0.03, 0.03])
        g.limb([tuple(root), tuple(mid), tuple(tp)], [0.016, 0.011, 0.003], 6)
    for a_, b_ in zip(tips[:-1], tips[1:]):
        sag = (a_ + b_) / 2 * 0.86 + root * 0.14
        poly = [g.P(*root), g.P(*a_), g.P(*sag), g.P(*b_)]
        nn = nrm(np.cross(poly[1] - poly[0], poly[3] - poly[0]))
        P0 = np.array(poly); th = 0.006 * S
        if float(newell(P0) @ nn) < 0: P0 = P0[::-1]          # Umlauf CCW um nn -> Seitenflaechen zeigen nach aussen
        T = np.concatenate([tri_poly(P0 + nn * th, normal=nn), tri_poly(P0 - nn * th, normal=-nn),
                            quads_to_tris(np.stack([P0 + nn * th, P0 - nn * th, np.roll(P0, -1, 0) - nn * th, np.roll(P0, -1, 0) + nn * th], axis=1))])
        g.B.add(mat, T, grp)


# ------------------------------------------------------------------ Tod (Skelett mit Leichentuch) und Mann
def skeleton(g, look_r=0.0):
    """Skelett in Figurkoordinaten von g (Hoehe ~1)."""
    m = g.mat
    # Schaedel + Unterkiefer
    look = g.V(look_r, 1.0, 0.05)
    g.head((0.0, 0.02, 0.935), look, 0.072, skull=True, nu=24, nv=16)
    g.ball((0.0, 0.05, 0.87), 0.045, 0.04, 0.022, 10, 6)
    # Wirbelsaeule (Wirbel als Perlen)
    sp = [(0.0, -0.035, 0.5), (0.0, -0.05, 0.6), (0.0, -0.045, 0.7), (0.0, -0.03, 0.8), (0.0, 0.0, 0.875)]
    g.limb(sp, [0.016, 0.015, 0.014, 0.013, 0.012], 8)
    for h in np.arange(0.52, 0.86, 0.035):
        k = (h - 0.5) / 0.375
        y = -0.035 - 0.015 * math.sin(k * math.pi) + 0.03 * max(k - 0.8, 0) * 5
        g.ball((0.0, y - 0.006, h), 0.022, 0.02, 0.011, 8, 4)
    # Rippen (6 Paare) + Brustbein
    for i in range(6):
        h = 0.79 - i * 0.04; a = 0.1 + 0.012 * min(i, 3); b = 0.075
        for sg in (1, -1):
            pts = [(0.0, -0.05, h), (sg * a * 0.75, -0.035, h - 0.008), (sg * a, 0.0, h - 0.02), (sg * a * 0.8, 0.045, h - 0.035), (sg * 0.03, b, h - 0.05)]
            g.limb(pts, [0.008, 0.009, 0.009, 0.008, 0.006], 6)
    g.limb([(0.0, 0.077, 0.79), (0.0, 0.075, 0.62)], [0.012, 0.009], 6)
    g.ball((0.0, 0.0, 0.5), 0.11, 0.06, 0.055, 12, 6)                          # Becken
    for sg in (1, -1):
        g.limb([(sg * 0.03, 0.0, 0.83), (sg * 0.13, -0.01, 0.82)], [0.009, 0.008], 6)    # Schluesselbein
    # Beine
    for sg in (1, -1):
        hip = (sg * 0.07, 0.01, 0.48); kn = (sg * 0.085, 0.04 + 0.03 * (sg < 0), 0.27); an = (sg * 0.09, 0.0, 0.04)
        g.limb([hip, kn], [0.018, 0.015], 8); g.limb([kn, an], [0.015, 0.012], 8)
        g.ball(kn, 0.026, 0.024, 0.024, 8, 5); g.ball(hip, 0.024, 0.024, 0.024, 8, 5)
        g.ball((an[0], an[1] + 0.04, 0.018), 0.025, 0.06, 0.015, 8, 5)


def death_and_man(B, foot, facing, S, mat, grp='det'):
    """Gruppe nach Foto 16 rechts: links das Skelett (Tod) mit Leichentuch und Sanduhr, die Knochenhand auf der Schulter
    des Mannes; rechts der Mann im weiten Mantel mit Kappe und Geldbeutel."""
    f0 = nrm([facing[0], facing[1], 0.0]); r0 = np.cross(f0, UP)
    # Sockelplatte
    g0 = Fig(B, foot, facing, S, mat, grp)
    g0.ball((0.0, 0.02, 0.02), 0.3, 0.2, 0.065, 16, 6, disp=lambda th, ph: 1.0 + 0.05 * math.sin(4 * th))
    # --- Mann (rechts im Bild = linke Koerperseite der Gruppe: -r), leicht zum Tod gedreht
    gm = Fig(B, foot - r0 * S * 0.17 + f0 * 0.02 * S, facing, S, mat, grp, turn=math.radians(-28))
    prof = [(0.0, 0.17, 0.14, 0.1), (0.03, 0.172, 0.142, 0.1), (0.15, 0.16, 0.13, 0.09), (0.3, 0.15, 0.115, 0.08),
            (0.45, 0.145, 0.105, 0.06), (0.58, 0.145, 0.1, 0.05), (0.68, 0.152, 0.1, 0.04), (0.76, 0.16, 0.098, 0.03),
            (0.8, 0.15, 0.09, 0.02), (0.825, 0.11, 0.075, 0.0), (0.84, 0.05, 0.05, 0.0)]
    gm.robe(prof, nseg=40, seed=23, folds=8)
    for sg in (1, -1):
        gm.ball((sg * 0.07, 0.09, 0.03), 0.032, 0.06, 0.025, 8, 5)
    gm.lathe_at((0.0, 0.0, 0.82), [(0.04, 0.0), (0.036, 0.06), (0.0, 0.07)], 10)
    look = gm.V(0.35, 1.0, 0.0)
    gm.head((0.0, 0.01, 0.925), look, 0.068, hair=0.05, beard=True, nu=26, nv=16)
    gm.lathe_at((0.0, 0.0, 0.975), [(0.0, 0.0), (0.075, 0.0), (0.085, 0.02), (0.07, 0.05), (0.0, 0.06)], 14)        # Kappe
    # Mantelkragen
    gm.limb([(0.12, -0.03, 0.8), (0.0, -0.08, 0.815), (-0.12, -0.03, 0.8)], [0.025, 0.03, 0.025], 8)
    # rechter Arm (zum Tod hin) angewinkelt vor der Brust, linker Arm haelt den Beutel
    gm.limb([(0.15, 0.0, 0.78), (0.2, 0.06, 0.6), (0.06, 0.13, 0.62)], [0.05, 0.048, 0.035], 10)
    gm.ball((0.05, 0.14, 0.625), 0.028, 0.02, 0.03, 8, 6)
    gm.limb([(-0.15, 0.0, 0.78), (-0.19, 0.02, 0.6), (-0.2, 0.06, 0.48)], [0.05, 0.05, 0.04], 10)
    gm.ball((-0.2, 0.07, 0.44), 0.045, 0.04, 0.06, 10, 6)                                                       # Beutel
    gm.limb([(-0.2, 0.07, 0.5), (-0.2, 0.065, 0.47)], [0.012, 0.02], 6)
    # Mantel faellt vom Arm herab
    gm.limb([(-0.17, 0.03, 0.62), (-0.2, 0.0, 0.4), (-0.18, -0.02, 0.2)], [0.03, 0.04, 0.025], 8, prof=ring_prof(10, 1.0, 1.8))
    # --- Tod (Skelett) links, etwas zurueck, zum Mann gedreht, mit Leichentuch
    gs = Fig(B, foot + r0 * S * 0.2 - f0 * 0.04 * S, facing, S * 0.98, mat, grp, turn=math.radians(30))
    skeleton(gs, look_r=-0.5)
    # Arme: rechter (zum Mann) auf dessen Schulter, linker haelt die Sanduhr hoch
    sh_m = gm.P(0.14, 0.02, 0.8)
    rel = sh_m - gs.foot
    mr = (float(rel @ gs.r) / gs.S, float(rel @ gs.f) / gs.S, float(rel[2]) / gs.S)
    for sg, hand in ((-1, mr), (1, (0.2, 0.08, 0.95))):
        sh = (sg * 0.13, -0.01, 0.82)
        el = ((sh[0] + hand[0]) / 2 + sg * 0.03, (sh[1] + hand[1]) / 2 + 0.02, min(sh[2], hand[2]) - 0.1)
        gs.limb([sh, el], [0.014, 0.012], 7); gs.limb([el, hand], [0.012, 0.01], 7)
        gs.ball(el, 0.018, 0.018, 0.018, 6, 4); gs.ball(sh, 0.022, 0.022, 0.022, 6, 4)
        for k in range(4):
            hp = np.array(hand)
            gs.limb([tuple(hp), tuple(hp + np.array([0.012 * (k - 1.5), 0.03, -0.03]))], [0.005, 0.003], 5)
    hg = np.array([0.2, 0.09, 0.97])
    gs.lathe_at(tuple(hg), [(0.03, 0.0), (0.03, 0.008), (0.024, 0.012), (0.004, 0.045), (0.024, 0.078), (0.03, 0.082), (0.03, 0.09), (0.0, 0.09)], 10)
    # Leichentuch: vom Kopf ueber den Ruecken bis zum Boden (dicke, offene Schale, nach hinten gerafft)
    t = np.linspace(math.radians(200), math.radians(340), 15)
    rings = []
    for h, a, b, k in ((1.0, 0.09, 0.09, 0.0), (0.92, 0.12, 0.11, 0.0), (0.82, 0.19, 0.12, 0.2), (0.62, 0.2, 0.12, 0.6),
                       (0.4, 0.21, 0.13, 0.9), (0.18, 0.22, 0.14, 1.0), (0.02, 0.24, 0.15, 1.0)):
        outer = [gs.P(a * math.cos(q) * (1 + 0.06 * k * math.sin(7 * q + h * 4)), b * math.sin(q) - 0.02, h) for q in t]
        inner = [gs.P(a * 0.88 * math.cos(q), b * 0.85 * math.sin(q) - 0.02, h) for q in t[::-1]]
        rings.append(np.array(outer + inner))
    gs.B.add(mat, loft(rings), grp, closed=True)


# ------------------------------------------------------------------ Reliefplatte aus Hoehenfeld
def _field(shapes, nx, ny):
    """shapes: Liste (typ, params, hoehe) in Platten-Einheitskoordinaten (x 0..1, y 0..1, y nach oben)."""
    X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
    H = np.zeros((ny, nx))
    for s in shapes:
        kind = s[0]
        if kind == 'ell':
            _, cx, cy, rx, ry, h = s
            d = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
            H = np.maximum(H, h * np.sqrt(np.clip(1 - d, 0, 1)))
        elif kind == 'cap':
            _, x0, y0, x1, y1, r, h = s
            px, py = X - x0, Y - y0; dx, dy = x1 - x0, y1 - y0
            tt = np.clip((px * dx + py * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
            d = np.hypot(px - tt * dx, py - tt * dy) / r
            H = np.maximum(H, h * np.sqrt(np.clip(1 - d * d, 0, 1)))
        elif kind == 'rect':
            _, x0, y0, x1, y1, h = s
            H = np.maximum(H, np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1), h, 0.0))
        elif kind == 'trap':                          # Gewand: Trapez (unten breit)
            _, cx, yb, yt, wb, wt, h = s
            k = np.clip((Y - yb) / max(yt - yb, 1e-6), 0, 1)
            hw = wb + (wt - wb) * k
            inside = (Y >= yb) & (Y <= yt) & (np.abs(X - cx) <= hw)
            H = np.maximum(H, np.where(inside, h * (0.7 + 0.3 * np.sqrt(np.clip(1 - ((X - cx) / np.maximum(hw, 1e-6)) ** 2, 0, 1))), 0.0))
    return H


def person(cx, cy, hgt, kneel=False, arms='pray', h=1.0, face=0):
    """Stehende/kniende Figur als Relief-Primitive (Unterkante cy, Hoehe hgt)."""
    s = hgt
    out = []
    if kneel:
        out += [('trap', cx, cy, cy + 0.55 * s, 0.11 * s, 0.07 * s, h * 0.85), ('ell', cx + face * 0.06 * s, cy + 0.12 * s, 0.14 * s, 0.1 * s, h * 0.8)]
        top = cy + 0.55 * s
    else:
        out += [('trap', cx, cy, cy + 0.8 * s, 0.1 * s, 0.06 * s, h * 0.85)]
        top = cy + 0.8 * s
    out += [('ell', cx + face * 0.012 * s, top + 0.085 * s, 0.065 * s, 0.085 * s, h)]                       # Kopf
    out += [('ell', cx, top - 0.02 * s, 0.1 * s, 0.05 * s, h * 0.8)]                                          # Schultern
    if arms == 'pray':
        out += [('cap', cx - 0.06 * s, top - 0.06 * s, cx + face * 0.09 * s, top - 0.16 * s, 0.03 * s, h * 0.95)]
        out += [('cap', cx + face * 0.09 * s, top - 0.16 * s, cx + face * 0.12 * s, top - 0.06 * s, 0.025 * s, h)]
    elif arms == 'open':
        out += [('cap', cx - 0.08 * s, top - 0.05 * s, cx - 0.2 * s, top + 0.12 * s, 0.03 * s, h * 0.9),
                ('cap', cx + 0.08 * s, top - 0.05 * s, cx + 0.2 * s, top + 0.12 * s, 0.03 * s, h * 0.9)]
    elif arms == 'down':
        out += [('cap', cx - 0.09 * s, top - 0.04 * s, cx - 0.1 * s, top - 0.4 * s, 0.03 * s, h * 0.9),
                ('cap', cx + 0.09 * s, top - 0.04 * s, cx + 0.1 * s, top - 0.4 * s, 0.03 * s, h * 0.9)]
    return out


def scene_shapes(kind, seed=0):
    rng = np.random.default_rng(seed)
    S = []
    if kind == 'crucifixion':       # Kreuz mit Christus, Maria + Johannes, darunter kniende Stifterfamilie
        S += [('rect', 0.485, 0.3, 0.515, 0.97, 0.7), ('rect', 0.36, 0.8, 0.64, 0.83, 0.7)]
        S += [('cap', 0.5, 0.86, 0.38, 0.81, 0.022, 1.0), ('cap', 0.5, 0.86, 0.62, 0.81, 0.022, 1.0),
              ('trap', 0.5, 0.55, 0.85, 0.035, 0.05, 1.0), ('ell', 0.5, 0.89, 0.03, 0.035, 1.0)]
        S += person(0.27, 0.32, 0.55, arms='pray', face=1) + person(0.73, 0.32, 0.55, arms='down', face=-1)
        for i, x in enumerate((0.1, 0.2, 0.3)):
            S += person(x, 0.04, 0.32, kneel=True, face=1, h=0.9)
        for i, x in enumerate((0.7, 0.8, 0.9)):
            S += person(x, 0.04, 0.32 - 0.03 * i, kneel=True, face=-1, h=0.9)
        S += [('rect', 0.0, 0.0, 1.0, 0.025, 0.5)]
    elif kind == 'resurrection':    # Christus ueber dem Grab, Waechter
        S += [('rect', 0.2, 0.05, 0.8, 0.25, 0.6)] + person(0.5, 0.3, 0.6, arms='open')
        S += [('ell', 0.5, 0.62, 0.22, 0.3, 0.25)]
        S += person(0.15, 0.03, 0.4, kneel=True, face=1, h=0.9) + person(0.85, 0.03, 0.4, kneel=True, face=-1, h=0.9)
    elif kind == 'arms':            # Wappen: Schild + Helm + Helmdecke
        S += [('trap', 0.5, 0.15, 0.6, 0.2, 0.28, 0.9), ('ell', 0.5, 0.2, 0.22, 0.15, 0.9), ('ell', 0.5, 0.72, 0.12, 0.13, 1.0)]
        for k in range(7):
            a = math.pi * (0.1 + 0.8 * k / 6)
            S += [('cap', 0.5, 0.7, 0.5 + 0.38 * math.cos(a), 0.62 + 0.3 * math.sin(a) - 0.1, 0.04, 0.6)]
        S += [('rect', 0.42, 0.3, 0.58, 0.5, 1.0)]
    elif kind == 'donors':          # Stifter kniend vor einem Kruzifix
        S += [('rect', 0.49, 0.1, 0.51, 0.8, 0.8), ('rect', 0.42, 0.62, 0.58, 0.65, 0.8), ('ell', 0.5, 0.08, 0.1, 0.06, 0.6)]
        for x in (0.12, 0.26, 0.38):
            S += person(x, 0.05, 0.62, kneel=True, face=1, h=0.9)
        for x in (0.62, 0.74, 0.88):
            S += person(x, 0.05, 0.6, kneel=True, face=-1, h=0.9)
    elif kind == 'ornament':        # Kandelaber-Groteske (Pilasterfuellung)
        for y in np.linspace(0.08, 0.92, 6):
            S += [('ell', 0.5, y, 0.18, 0.05, 0.8), ('cap', 0.5, y, 0.15, y + 0.06, 0.035, 0.7), ('cap', 0.5, y, 0.85, y + 0.06, 0.035, 0.7)]
        S += [('rect', 0.47, 0.02, 0.53, 0.98, 0.6)]
    elif kind == 'skull':           # Totenkopf mit gekreuzten Knochen + Kartusche
        S += [('ell', 0.5, 0.55, 0.22, 0.24, 1.0), ('cap', 0.2, 0.2, 0.8, 0.4, 0.05, 0.8), ('cap', 0.2, 0.4, 0.8, 0.2, 0.05, 0.8)]
    return S


def relief_panel(B, F, x0, y0, w, h, d0, depth, kind, mat, back=0.08, res=0.022, seed=0, frame=0.06, grp='det', mat_bg=None):
    """Reliefplatte in Wandebene F: linke untere Ecke (x0, y0), Groesse w x h, Grund bei d0 (vor der Wand), Relief bis d0+depth.
    Geschlossener Koerper: Hoehenfeld vorn, Seiten bis d0-back, Rueckwand. Rahmenleiste (frame) am Rand erhaben."""
    nx = max(8, int(round(w / res))); ny = max(8, int(round(h / res)))
    S = scene_shapes(kind, seed)
    # Bildfeld innerhalb des Rahmens
    fx = frame / w; fy = frame / h
    Hin = _field([(s[0],) + _remap(s, fx, fy) for s in S], nx, ny) if S else np.zeros((ny, nx))
    Hin = gaussian_filter(Hin, 0.7)
    X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
    fr = np.minimum(np.minimum(X, 1 - X) / fx, np.minimum(Y, 1 - Y) / fy)
    Hf = np.where(fr < 1.0, 0.55, 0.0)
    Hf = gaussian_filter(Hf, 0.6)
    rng = np.random.default_rng(seed + 5)
    H = np.maximum(Hin, Hf) + 0.04 * gaussian_filter(rng.standard_normal((ny, nx)), 1.2)
    # Gitter (Knoten an Zellraendern): Rand auf 0
    Hn = np.zeros((ny + 1, nx + 1))
    Hn[1:-1, 1:-1] = 0.25 * (H[:-1, :-1] + H[1:, :-1] + H[:-1, 1:] + H[1:, 1:])
    Hn = np.clip(Hn, 0, None)
    xs = x0 + np.linspace(0, w, nx + 1); ys = y0 + np.linspace(0, h, ny + 1)
    XX, YY = np.meshgrid(xs, ys)
    Pf = F.p(XX, YY, d0 + depth * Hn)                         # (ny+1, nx+1, 3)
    Q = np.stack([Pf[:-1, :-1], Pf[:-1, 1:], Pf[1:, 1:], Pf[1:, :-1]], axis=2).reshape(-1, 4, 3)
    front = quads_to_tris(Q)
    # Randring (CCW in x/y) vorn bei d0 und hinten bei d0-back
    ring_idx = [(0, i) for i in range(nx + 1)] + [(j, nx) for j in range(1, ny + 1)] + [(ny, i) for i in range(nx - 1, -1, -1)] + [(j, 0) for j in range(ny - 1, 0, -1)]
    R0 = np.array([Pf[j, i] for j, i in ring_idx])
    R1 = np.array([F.p(xs[i], ys[j], d0 - back) for j, i in ring_idx])
    n = len(R0); k = np.arange(n); k1 = (k + 1) % n
    # Orientierung: Vorderseite (+n) wie das Gitter; Seiten nach aussen, Rueckwand nach -n (sonst Rueckseiten von aussen sichtbar)
    sides = quads_to_tris(np.stack([R0[k], R1[k], R1[k1], R0[k1]], axis=1))
    backf = tri_poly(np.array([F.p(x0, y0, d0 - back), F.p(x0 + w, y0, d0 - back), F.p(x0 + w, y0 + h, d0 - back), F.p(x0, y0 + h, d0 - back)]), normal=-np.cross(F.t, UP))
    nf_ = np.cross(front[:, 1] - front[:, 0], front[:, 2] - front[:, 0]).sum(axis=0)
    flip = float(nf_ @ F.n) < 0                                # Gitter-Umlaufsinn je nach Wandrahmen
    # Reliefgrund (tiefe Stellen) dunkel verwittert (Sandsteinkruste) -> Figuren lesbar auch bei Frontallicht
    Hq = 0.25 * (Hn[:-1, :-1] + Hn[:-1, 1:] + Hn[1:, 1:] + Hn[1:, :-1]).reshape(-1)
    low = np.concatenate([Hq < 0.12, Hq < 0.12])
    parts = [(mat, np.concatenate([front[~low] if mat_bg else front, sides, backf]))]
    if mat_bg: parts.append((mat_bg, front[low]))
    for m_, T_ in parts:
        B.add(m_, T_[:, ::-1] if flip else T_, grp)


def _remap(s, fx, fy):
    """Formparameter aus 0..1 in das Bildfeld innerhalb des Rahmens abbilden."""
    kind = s[0]; p = list(s[1:])
    mx = lambda v: fx + v * (1 - 2 * fx); my = lambda v: fy + v * (1 - 2 * fy)
    sx = 1 - 2 * fx; sy = 1 - 2 * fy
    if kind == 'ell':
        return (mx(p[0]), my(p[1]), p[2] * sx, p[3] * sy, p[4])
    if kind == 'cap':
        return (mx(p[0]), my(p[1]), mx(p[2]), my(p[3]), p[4] * min(sx, sy), p[5])
    if kind == 'rect':
        return (mx(p[0]), my(p[1]), mx(p[2]), my(p[3]), p[4])
    if kind == 'trap':
        return (mx(p[0]), my(p[1]), my(p[2]), p[3] * sx, p[4] * sx, p[5])
    return tuple(p)

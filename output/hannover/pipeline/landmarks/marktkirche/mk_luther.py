"""Marktkirche -- Lutherdenkmal (Ferdinand Hartzer, 1900): Bronzefigur im Talar auf hohem Steinsockel.
Nach Foto ref_images/marktkirche/13_Hannover_Lutherdenkmal_452.jpg: rechter Arm erhoben (Zeigefinger nach oben),
linke Hand haelt die Bibel vor der Brust, weiter Talar mit tiefen Laengsfalten, linker Fuss vorgesetzt, Bronze-Standplatte,
Sockel = schlichter dunkler Steinblock (Inschrift MARTIN LUTHER / 1483 - 1546 wird im Blender-Schritt gesetzt).
Alle Masse normiert auf Scheitelhoehe 1.0 und mit Hh skaliert. Erzeugt nur geschlossene Koerper (Builder.add closed=True)."""
import math
import numpy as np
from mklib import UP, nrm, loft, lathe, sweep, tube, chamfer_box, box


def blob(c, ax, ay, az, rx, ry, rz, nu=20, nv=12, disp=None):
    """Geschlossenes, verformtes Ellipsoid. disp(theta, phi) -> radialer Faktor (theta um az, phi von +az nach -az)."""
    c = np.asarray(c, float); ax, ay, az = (np.asarray(a, float) for a in (ax, ay, az))
    th = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    ph = np.linspace(0, math.pi, nv + 1)[1:-1]
    rings = []
    for p in ph:
        f = np.ones(nu) if disp is None else np.array([disp(t, p) for t in th])
        P = c[None] + (f * rx * np.sin(p) * np.cos(th))[:, None] * ax + (f * ry * np.sin(p) * np.sin(th))[:, None] * ay \
            + (f * rz * math.cos(p))[:, None] * az
        rings.append(P)
    ft = 1.0 if disp is None else disp(0.0, 0.0)
    fb = 1.0 if disp is None else disp(0.0, math.pi)
    top = c + az * rz * ft; bot = c - az * rz * fb
    out = []
    k = np.arange(nu); k1 = (k + 1) % nu
    out.append(np.stack([np.repeat(top[None], nu, 0), rings[0][k], rings[0][k1]], axis=1))
    for a, b in zip(rings[:-1], rings[1:]):
        out.append(np.stack([a[k], b[k], b[k1]], axis=1)); out.append(np.stack([a[k], b[k1], a[k1]], axis=1))
    out.append(np.stack([np.repeat(bot[None], nu, 0), rings[-1][k1], rings[-1][k]], axis=1))
    return np.concatenate(out)


def luther(B, foot, facing, Hh, mat, grp='det', seed=1900):
    """foot = Standpunkt (Oberkante Standplatte), facing = Blickrichtung (horizontal), Hh = Scheitelhoehe in m."""
    foot = np.asarray(foot, float)
    f = nrm([facing[0], facing[1], 0.0]); r = np.cross(f, UP)        # r = rechte Koerperseite der Figur
    S = Hh
    rng = np.random.default_rng(seed)

    def P(rr, ff, hh):
        rr, ff, hh = (np.asarray(q, float)[..., None] for q in (rr, ff, hh))
        return foot + S * (rr * r + ff * f + hh * UP)

    # ---------------------------------------------------------- Talar (Loft elliptischer Ringe mit Laengsfalten)
    #       h      a(rechts/links) b(vorn/hinten) Faltenamplitude  Versatz vorn (Schritt)
    prof = [(0.000, 0.205, 0.175, 0.13, 0.035), (0.020, 0.208, 0.178, 0.13, 0.035), (0.080, 0.196, 0.165, 0.12, 0.030),
            (0.180, 0.182, 0.148, 0.11, 0.022), (0.300, 0.172, 0.132, 0.10, 0.014), (0.420, 0.164, 0.120, 0.08, 0.008),
            (0.520, 0.158, 0.112, 0.06, 0.004), (0.600, 0.156, 0.110, 0.045, 0.0), (0.680, 0.162, 0.114, 0.035, 0.0),
            (0.740, 0.168, 0.112, 0.03, 0.0), (0.780, 0.165, 0.102, 0.02, 0.0), (0.805, 0.140, 0.088, 0.01, 0.0),
            (0.822, 0.090, 0.070, 0.0, 0.0), (0.835, 0.050, 0.050, 0.0, 0.0)]
    nseg = 120
    t = np.linspace(0, 2 * math.pi, nseg, endpoint=False)          # t=0 -> rechts, pi/2 -> vorn
    nf = 14
    fph = rng.uniform(0, 2 * math.pi, nf); fk = rng.integers(4, 17, nf); fa = rng.uniform(0.4, 1.0, nf)
    # dichtere Ringe (Profil linear verfeinert) -> weich fallende Laengsfalten statt Facetten
    P_ = np.array(prof)
    hs_ = np.unique(np.concatenate([np.linspace(P_[i, 0], P_[i + 1, 0], 4)[:-1] for i in range(len(P_) - 1)] + [P_[-1:, 0]]))
    prof = [tuple(float(np.interp(h_, P_[:, 0], P_[:, k])) for k in range(5)) for h_ in hs_]
    rings = []
    for i, (h, a, b, amp, step) in enumerate(prof):
        fold = np.zeros(nseg)
        for j in range(nf):
            fold += fa[j] * np.cos(fk[j] * t + fph[j] + 0.6 * h + 0.25 * math.sin(7 * h + j))
        fold = fold / nf * 2.4
        fold = np.sign(fold) * np.abs(fold) ** 0.55                  # schaerfere Faltengrate, breite Taeler
        rad = 1.0 + 1.3 * amp * fold
        # Schaube vorn offen: Kante links der Mitte (Ueberlappung), leichte Stufe
        front = np.exp(-((t - math.pi / 2 - 0.25) / 0.35) ** 2)
        rad = rad + 0.025 * front * (1 if h < 0.8 else 0)
        x = a * np.cos(t) * rad
        y = b * np.sin(t) * rad
        # Schritt: linkes Bein vor -> vordere linke Haelfte nach vorn
        y = y + step * np.clip(np.sin(t), 0, 1) * np.clip(-np.cos(t) + 0.3, 0, 1.3)
        # Ruecken etwas flacher, Saum hinten haengt tiefer (Schleppe)
        rings.append(P(x, y, np.full(nseg, h)))
    B.add(mat, loft(rings), grp, closed=True)
    # Saumumschlag (Wulst) unten
    hem = np.array(rings[1]) + UP * 0.0
    # ---------------------------------------------------------- Schuhe (linker Fuss vor, rechter halb verdeckt)
    for rr, ff, sc in ((-0.075, 0.175, 1.0), (0.085, 0.120, 0.85)):
        B.add(mat, blob(P(rr, ff, 0.018), f, r, UP, S * 0.06 * sc, S * 0.034 * sc, S * 0.022, 12, 6,
                        disp=lambda th, ph: 1.0 - 0.15 * (math.cos(ph) < -0.3)), grp, closed=True)
    # ---------------------------------------------------------- Kragen / Revers der Schaube (breite Aufschlaege bis zur Brust)
    for sg in (-1, 1):
        path = [P(sg * 0.10, -0.03, 0.812), P(sg * 0.105, 0.055, 0.80), P(sg * 0.075, 0.105, 0.75), P(sg * 0.045, 0.12, 0.66), P(sg * 0.03, 0.13, 0.56)]
        prof2 = np.array([(-0.032, -0.006), (0.032, -0.006), (0.030, 0.008), (-0.030, 0.008)]) * S
        B.add(mat, sweep(np.array(path), prof2, UP, scale=[1.1, 1.25, 1.1, 0.8, 0.5]), grp, closed=True)
    # Kragen hinten (liegender Wulst um den Hals)
    cpts = [P(0.085 * math.cos(q), 0.065 * math.sin(q) - 0.005, 0.815 + 0.012 * math.sin(q)) for q in np.linspace(math.radians(110), math.radians(430), 18)]
    B.add(mat, sweep(np.array(cpts), np.array([(0.022 * math.cos(q), 0.014 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 9)[:-1]]) * S, UP), grp, closed=True)
    # Beffchen (zwei kleine Baender unter dem Kinn)
    for sg in (-1, 1):
        B.add(mat, chamfer_box(P(sg * 0.011, 0.062, 0.80), r, f, UP, S * 0.009, S * 0.004, S * 0.02, 0.002), grp, closed=True)
    # ---------------------------------------------------------- Hals + Kopf
    B.add(mat, lathe(P(0, 0.0, 0.81), [(0.052 * S, 0), (0.048 * S, 0.03 * S), (0.044 * S, 0.06 * S), (0.0, 0.07 * S)], 16), grp, closed=True)   # kraeftiger Hals
    hc = P(0.0, 0.01, 0.912)
    look = nrm(f * 0.93 + UP * 0.22 - r * 0.12)                       # Blick leicht nach oben, zur erhobenen Hand hin
    hr = nrm(np.cross(look, UP)); hu = np.cross(hr, look)

    def head_disp(th, ph):
        # th: 0 = rechts, pi/2 = vorn (Gesicht); ph: 0 = Scheitel
        x, y, z = math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)
        d = 1.0
        front = max(y, 0.0)
        d += 0.38 * math.exp(-((x / 0.11) ** 2 + ((z + 0.04) / 0.2) ** 2)) * front ** 5        # Nase
        d += 0.06 * math.exp(-((x / 0.07) ** 2 + ((z + 0.2) / 0.05) ** 2)) * front ** 4        # Nasenspitze/-fluegel
        d += 0.08 * math.exp(-((x / 0.22) ** 2 + ((z - 0.24) / 0.07) ** 2)) * front ** 2       # Brauenwulst
        d += 0.05 * math.exp(-(((abs(x) - 0.33) / 0.15) ** 2 + ((z - 0.26) / 0.06) ** 2)) * front   # Brauen
        d -= 0.14 * math.exp(-(((abs(x) - 0.33) / 0.12) ** 2 + ((z - 0.12) / 0.08) ** 2)) * front   # Augenhoehlen
        d += 0.035 * math.exp(-(((abs(x) - 0.33) / 0.06) ** 2 + ((z - 0.12) / 0.04) ** 2)) * front  # Augapfel/Lider
        d -= 0.07 * math.exp(-((x / 0.24) ** 2 + ((z + 0.34) / 0.025) ** 2)) * front           # Mundspalte
        d += 0.04 * math.exp(-((x / 0.2) ** 2 + ((z + 0.29) / 0.05) ** 2)) * front             # Oberlippe
        d += 0.035 * math.exp(-((x / 0.18) ** 2 + ((z + 0.4) / 0.05) ** 2)) * front            # Unterlippe
        d += 0.12 * math.exp(-((z + 0.58) / 0.16) ** 2) * (0.6 + 0.4 * front)                  # schweres Kinn / Doppelkinn
        d += 0.07 * math.exp(-(((abs(x) - 0.52) / 0.2) ** 2 + ((z + 0.18) / 0.22) ** 2))       # volle Wangen
        d += 0.12 * math.exp(-((abs(x) - 0.96) / 0.08) ** 2 - ((z + 0.0) / 0.17) ** 2) * (y < 0.2)   # Ohren
        # Haar: kurze kraeftige Locken am Hinterkopf, an den Seiten und oben, Stirn frei
        hair = ((z > -0.25) and (y < 0.2)) or (z > 0.55) or (z > 0.35 and y < 0.6)
        if hair:
            d += 0.08 + 0.05 * math.sin(11 * th + 4 * ph) * math.sin(9 * ph + 3 * th) + 0.02 * math.sin(23 * th) * math.sin(19 * ph)
        return d
    B.add(mat, blob(hc, hr, look, hu, S * 0.06, S * 0.068, S * 0.072, 56, 40, disp=head_disp), grp, closed=True)
    # ---------------------------------------------------------- rechter Arm: erhoben, Zeigefinger nach oben
    sh = P(0.150, 0.0, 0.790); el = P(0.235, 0.035, 0.865); wr = P(0.215, 0.07, 1.005)
    # weiter Aermel von der Schulter bis ueber den Ellbogen, gebauscht
    B.add(mat, sweep(np.array([sh - UP * 0.02 * S, P(0.205, 0.02, 0.83), el, P(0.228, 0.05, 0.92)]),
                     np.array([(0.05 * math.cos(q), 0.05 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 21)[:-1]]) * S, f,
                     scale=[1.25, 1.45, 1.75, 1.5]), grp, closed=True)
    # herabfallender Aermelsack unter dem Ellbogen (Stoff haengt nach unten)
    bag = [P(0.235, 0.04, 0.90), P(0.25, 0.03, 0.83), P(0.255, 0.02, 0.76), P(0.245, 0.01, 0.70)]
    B.add(mat, sweep(np.array(bag), np.array([(0.035 * math.cos(q), 0.06 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 11)[:-1]]) * S, r,
                     scale=[1.0, 1.15, 1.0, 0.45]), grp, closed=True)
    # Unterarm (Unterkleid-Aermel)
    B.add(mat, sweep(np.array([P(0.228, 0.05, 0.92), wr]), np.array([(0.03 * math.cos(q), 0.03 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 11)[:-1]]) * S, f,
                     scale=[1.15, 0.85]), grp, closed=True)
    # Hand: Faust + Zeigefinger + Daumen
    hd = P(0.212, 0.075, 1.035)
    B.add(mat, blob(hd, r, f, UP, S * 0.022, S * 0.017, S * 0.03, 12, 8), grp, closed=True)
    B.add(mat, tube(np.array([P(0.206, 0.082, 1.055), P(0.203, 0.084, 1.085), P(0.201, 0.085, 1.108)]), S * 0.0065, 6), grp, closed=True)
    B.add(mat, tube(np.array([P(0.196, 0.083, 1.035), P(0.193, 0.088, 1.052)]), S * 0.0065, 6), grp, closed=True)
    # ---------------------------------------------------------- linker Arm: angewinkelt, Bibel vor der Brust
    shl = P(-0.150, 0.0, 0.790); ell = P(-0.190, 0.04, 0.615); hl = P(-0.055, 0.135, 0.655)
    B.add(mat, sweep(np.array([shl - UP * 0.02 * S, P(-0.18, 0.02, 0.72), ell]),
                     np.array([(0.05 * math.cos(q), 0.05 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 21)[:-1]]) * S, r,
                     scale=[1.2, 1.35, 1.5]), grp, closed=True)
    B.add(mat, sweep(np.array([ell, P(-0.13, 0.10, 0.63), hl]),
                     np.array([(0.05 * math.cos(q), 0.05 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 21)[:-1]]) * S, UP,
                     scale=[1.5, 1.45, 1.1]), grp, closed=True)
    # langer weiter Aermel haengt vom Unterarm bis zum Knie herab (Foto: rechte Bildseite)
    drape = [P(-0.16, 0.08, 0.62), P(-0.185, 0.07, 0.52), P(-0.20, 0.06, 0.42), P(-0.205, 0.05, 0.33), P(-0.20, 0.04, 0.27)]
    B.add(mat, sweep(np.array(drape), np.array([(0.03 * math.cos(q), 0.075 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 21)[:-1]]) * S, r,
                     scale=[1.2, 1.25, 1.15, 0.95, 0.4]), grp, closed=True)
    # Bibel (aufrecht gegen die Brust, Buchschnitt + Deckel)
    bk = P(-0.02, 0.145, 0.705)
    bz = nrm(f * 0.95 + UP * 0.1); bx = nrm(np.cross(UP, bz)) * -1; by = np.cross(bz, bx)
    B.add(mat, chamfer_box(bk, bx, by, bz, S * 0.052, S * 0.07, S * 0.018, 0.012), grp, closed=True)
    B.add(mat, chamfer_box(bk + bz * S * 0.019, bx, by, bz, S * 0.054, S * 0.072, S * 0.003, 0.002), grp, closed=True)
    # Hand ueber dem Buch (Finger auf dem Deckel)
    B.add(mat, blob(hl + f * S * 0.012 + UP * S * 0.015, r, f, UP, S * 0.03, S * 0.016, S * 0.024, 12, 8), grp, closed=True)
    for k in range(3):
        B.add(mat, tube(np.array([hl + f * S * 0.02 + UP * S * (0.0 + 0.012 * k) + r * S * 0.02,
                                  hl + f * S * 0.03 + UP * S * (0.004 + 0.012 * k) + r * S * 0.05]), S * 0.006, 6), grp, closed=True)


def standplatte(B, c, facing, mat, grp='det', S=1.0):
    """Unregelmaessige Bronze-Standplatte (Plinthe) mit herabfallendem Gewandzipfel."""
    c = np.asarray(c, float); f = nrm([facing[0], facing[1], 0.0]); r = np.cross(f, UP)
    t = np.linspace(0, 2 * math.pi, 22, endpoint=False)
    rad = 1.0 + 0.04 * np.sin(3 * t + 0.5) + 0.03 * np.sin(5 * t + 1.3)
    ring = lambda h, k: c[None] + (k * 0.6 * S * np.cos(t) * rad)[:, None] * r + (k * 0.5 * S * np.sin(t) * rad)[:, None] * f + UP * h
    B.add(mat, loft([ring(0.0, 1.0), ring(0.11, 0.99), ring(0.15, 0.95)]), grp, closed=True)

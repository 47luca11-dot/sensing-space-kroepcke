# Teil 2 des Altes-Rathaus-Bauskripts (wird von build_rathaus.py per exec eingebunden; teilt dessen Namensraum).
# Schmiedestrassenfluegel, Karmarschstrassenfluegel (frueher hier "Apothekenfluegel" genannt; die Ratsapotheke ist ein anderes Haus), Gerichtsfluegel, Verbindungsbau, Hof, Daecher, Export.

# ###########################################################################################
# 2) SCHMIEDESTRASSENFLUEGEL (F2b, E2 -> N2)
# ###########################################################################################
SF_DEPTH = 12.6
F2b = Fac('F2b_schmiede', E2, N2, SF_EAVE + 0.35)
g2b = F2b.g
L2b = F2b.L
bays = [1.05 + 2.1 * k for k in range(13)]
GROUND_SF = {0: 'w', 1: 'arch', 3: 'door', 4: 'w', 5: 'w', 6: 'arch', 8: 'w', 9: 'w', 10: 'door', 11: 'w', 12: 'w'}
for k, kind in GROUND_SF.items():
    a = bays[k]
    if kind == 'w':
        F2b.op(win_ground(a, y0=1.95 + g2b, w=1.15, hs=1.65, lights=2))
    elif kind == 'arch':
        a2 = a + 1.05
        o = Opening(a2, g2b + 0.0, 2.35, 2.05, 'point', rise=1.45, ring={'w': 0.28, 'proj': 0.02}, content='shop',
                    orders=((0, 0.15), (0.12, 0.45)), skip_bottom=True, nj=4)
        o.shop = dict(kind='generic', door=1, prio=1, depth=3.0)
        F2b.op(o, low=True)
    elif kind == 'door':
        F2b.op(Opening(a, g2b + 0.0, 1.30, 2.25, 'point', rise=0.75, ring={'w': 0.15, 'proj': 0.015}, content='door',
                       orders=((0, 0.12), (0.10, 0.40)), nj=4), low=True)
F2b.frieze(4.60, 0.52)
for a in bays:
    F2b.op(win_up(a, y0=5.55, hs=2.05, rise=0.78, w=1.12))
F2b.lattice(9.40, 9.84)
F2b.finish()
plinth(F2b)
tan_sf = (SF_RIDGE - SF_ROOFLINE) / (SF_DEPTH / 2)
SLOPE_SF = math.degrees(math.atan(tan_sf))
eave_detail(E2, N2, SF_EAVE, SF_ROOFLINE, SLOPE_SF)
for a in (2.70, 14.70, 22.70):
    dormer(F2b, a, SF_EAVE - 0.02, w=2.80, ytop=14.60, slope_main=SLOPE_SF, roofline=SF_ROOFLINE, pin_r=0.27, crown=0.27)
for a in (0.45, 12.0, 26.9):
    downpipe(mb, F2b.fr, a, g2b - 0.1, SF_EAVE - 0.35)
canopy(F2b, 18.4, 25.1, 3.55 + g2b, 3.6)

# ###########################################################################################
# 3) KARMARSCHSTRASSENFLUEGEL (F3 S->E mit Laubengang und Risalit, F2a NO-Giebel, F4d SW-Giebel)
# ###########################################################################################
AF_DEPTH = 8.0
tan_af = (AF_RIDGE - AF_ROOFLINE) / (AF_DEPTH / 2)
SLOPE_AF = math.degrees(math.atan(tan_af))
RIS = (13.27, 24.01, 0.25)                  # Risalit (a0, a1, Vorsprung)
ARC_D = 3.50                                 # Tiefe Laubengang (Wandflucht bis Rueckwand)
ARC_C = 4.70                                 # Deckenhoehe Laubengang
g3 = np.mean([ground_along(S, E, a) for a in np.linspace(0, 37.4, 9)])


def af_upper(fac, centers, w1=1.42, w2=1.42, a_off=0.0, lights=2):
    # Foto 10: 1. OG hohe Fenster unter Segmentbogen mit Kaempfer, 2. OG Spitzbogenfenster mit Rosetten-Tympanon
    for a in centers:
        fac.op(Opening(a, 6.05, w1, 2.95, 'seg', rise=0.30, ring={'w': 0.15, 'proj': 0.015}, lights=lights, bars_h=0.55, bars_v=1,
                       orders=((0, 0.10), (0.09, 0.30)), nj=6, transom=8.15))
        fac.op(Opening(a, 11.00, w2, 2.40, 'point', rise=1.05, ring={'w': 0.14, 'proj': 0.015}, tymp={'n': lights}, lights=lights,
                       bars_h=0.45, orders=((0, 0.10), (0.09, 0.30)), nj=6))


def arcade_arches(fac, centers, span, g, depth=0.90):
    for a in centers:
        fac.op(Opening(a, g - 0.02, span, 2.75, 'point', rise=1.55, ring={'w': 0.32, 'proj': 0.02}, content='none',
                       orders=((0, 0.18), (0.14, depth)), skip_bottom=True, nj=5), low=True)


# Hauptflucht S-Teil und E-Teil
F3s = Fac('F3s_karmarsch', S, S + dSE * RIS[0], AF_EAVE + 0.35)
F3e = Fac('F3e_karmarsch', S + dSE * RIS[1], E, AF_EAVE + 0.35)
cs = [1.66 + 3.32 * k for k in range(4)]
ce = [1.67 + 3.35 * k for k in range(4)]
arcade_arches(F3s, cs, 2.35, g3); arcade_arches(F3e, ce, 2.35, g3)
for f in (F3s, F3e):
    f.frieze(5.20, 0.55)
    f.lattice(10.25, 10.70)
    f.lattice(15.25, 15.70)
af_upper(F3s, cs, w1=1.85, w2=1.85, lights=3); af_upper(F3e, ce, w1=1.75, w2=1.75, lights=2)
F3s.finish(); F3e.finish()
# Risalit (vorspringend) mit grossem Staffelgiebel
pr0 = S + dSE * RIS[0] - inF3 * RIS[2]; pr1 = S + dSE * RIS[1] - inF3 * RIS[2]
F3m = Fac('F3m_risalit', pr0, pr1, AF_EAVE - 0.02)
cm = [1.79 + 3.58 * k for k in range(3)]
arcade_arches(F3m, cm, 2.60, g3, depth=0.90 + RIS[2])      # Leibung bis zur Rueckseite der Hauptflucht
F3m.frieze(5.20, 0.55); F3m.lattice(10.25, 10.70); F3m.lattice(15.25, 15.70)
for a in cm:
    F3m.op(Opening(a, 6.05, 1.55, 2.95, 'seg', rise=0.32, ring={'w': 0.15, 'proj': 0.015}, lights=2, bars_h=0.55, orders=((0, 0.10), (0.09, 0.30)), nj=6, transom=8.15))
for a in np.linspace(1.25, F3m.L - 1.25, 5):
    F3m.op(Opening(a, 11.00, 0.78, 2.55, 'point', rise=0.62, ring={'w': 0.12, 'proj': 0.015}, lights=1, bars_h=0.45, bars_v=1, orders=((0, 0.09), (0.07, 0.28)), nj=6))
F3m.finish()
# Seitenwangen des Risalits
for (pa, pb) in ((S + dSE * RIS[0], pr0), (pr1, S + dSE * RIS[1])):
    fw = Fac('ris_side', pa, pb, AF_EAVE + 0.35)
    fw.finish()
    plinth(fw)
Lm = F3m.L
pins_r = list(np.linspace(0.30, Lm - 0.30, 8))
tops_r = [18.30, 20.50, 22.70, 25.40, 22.70, 20.50, 18.30]


def ris_win(ca, cb, t0, t1):
    w = min(0.62, (cb - ca) - 0.30)
    hs = max(0.6, (t1 - t0) - 0.55 - w * 0.6 - 0.45)
    return [Opening((ca + cb) / 2, t0 + 0.40, w, hs, 'point', rise=w * 0.6, ring={'w': 0.10, 'proj': 0.012}, lights=1, bars_h=0.4, bars_v=0,
                    orders=((0, 0.08), (0.06, 0.24)), nj=3)]


WIN_R = {k: {0: ris_win, 1: ris_win} for k in range(1, 6)}
WIN_R[3][2] = ris_win
gt_r = stepped_gable(F3m, 0.0, Lm, AF_EAVE, pins_r, tops_r, windows=WIN_R, rows_h=1.55, r=0.44, crown=0.42)

# Laubengang (Geometrie folgt nach den Giebeln F2a/F4d, deren Eckboegen den Gang abschliessen)
arcL = 37.4
farc = Frame.facade(S, E)
L3 = float(np.linalg.norm(E - S))
n3_ = -inF3
g3f = max(g3, max(ground(*(S + dSE * a_ + inF3 * d_)) for a_ in np.linspace(0.6, 36.8, 25) for d_ in (1.0, 2.2, 3.4)) + 0.02)

# NO-Giebel (F2a, E -> E2) -- 'Balzac'-Giebel
F2a = Fac('F2a_af_nogiebel', E, E2, AF_EAVE - 0.02)        # endet am Giebelfuss (stepped_gable ab AF_EAVE - 0.02)
g2a = F2a.g
for a in (2.05, 6.11):
    o = Opening(a, g2a - 0.02, 2.70, 2.55, 'point', rise=1.55, ring={'w': 0.32, 'proj': 0.02}, content='shop',
                orders=((0, 0.16), (0.13, 0.30 if a < 4 else 0.50)), skip_bottom=True, nj=4)
    if a < 4:       # Eckbogen vor dem Laubengang: Glaswand mit Tuer
        o.shop = dict(mode='screen', door=1, floor=g3f); OP_CORNER_E = o
    else:           # Laden (Mieter unbekannt, BEGEHUNGSLISTE AR-3)
        o.shop = dict(kind='lobby', door=1, prio=2.5, depth=1.3, side=0.25)
    F2a.op(o, low=True)
F2a.frieze(5.20, 0.55)
for a in (1.40, 4.08, 6.76):
    F2a.op(Opening(a, 6.10, 1.32, 2.90, 'seg', rise=0.30, ring={'w': 0.15, 'proj': 0.015}, lights=2, bars_h=0.55, orders=((0, 0.10), (0.09, 0.30)), nj=6, transom=8.2))
    F2a.op(Opening(a, 10.75, 1.32, 2.55, 'seg', rise=0.30, ring={'w': 0.15, 'proj': 0.015}, lights=2, bars_h=0.55, orders=((0, 0.10), (0.09, 0.30)), nj=6, transom=12.6))
F2a.texband(QF, 9.70, 10.10)
F2a.texband(RS, 14.55, 14.85)
F2a.lattice(15.25, 15.70)
F2a.finish()
plinth(F2a)
L2a = F2a.L
pins_a = list(np.linspace(0.30, L2a - 0.30, 8))
tops_a = [18.0, 20.2, 22.4, 24.6, 22.4, 20.2, 18.0]


def two_win(ca, cb, t0, t1):
    cw = cb - ca; w = min(0.30, cw / 2 - 0.12)
    hs = max(0.6, (t1 - t0) - 0.55 - w * 0.6 - 0.40)
    return [Opening(ca + cw * 0.3, t0 + 0.40, w, hs, 'point', rise=w * 0.6, ring=None, lights=1, bars_h=0.35, bars_v=0, orders=((0, 0.08), (0.04, 0.22)), nj=3),
            Opening(ca + cw * 0.7, t0 + 0.40, w, hs, 'point', rise=w * 0.6, ring=None, lights=1, bars_h=0.35, bars_v=0, orders=((0, 0.08), (0.04, 0.22)), nj=3)]


WIN_A = {k: {0: two_win, 1: two_win} for k in (2, 3, 4)}
WIN_A[1] = {0: two_win}; WIN_A[5] = {0: two_win}
stepped_gable(F2a, 0.0, L2a, AF_EAVE, pins_a, tops_a, windows=WIN_A, rows_h=1.50, r=0.40, crown=0.38)

# SW-Giebel (F4d, Q -> S)
F4d = Fac('F4d_af_swgiebel', Q, S, AF_EAVE - 0.02)
g4d = F4d.g
for a in (2.15, 6.05):
    o = Opening(a, g4d - 0.02, 2.65, 2.55, 'point', rise=1.55, ring={'w': 0.32, 'proj': 0.02}, content='shop',
                orders=((0, 0.16), (0.13, 0.30 if a > 4 else 0.50)), skip_bottom=True, nj=4)
    if a > 4:       # Eckbogen vor dem Laubengang: Glaswand mit Tuer
        o.shop = dict(mode='screen', door=1, floor=g3f); OP_CORNER_S = o
    else:
        o.shop = dict(kind='generic', door=1, prio=2.3, depth=1.2, side=0.0)       # vor Joch 0 des Laubengangs zuteilen, ohne seitlichen Ueberstand (Platz fuer beide)
    F4d.op(o, low=True)
F4d.frieze(5.20, 0.55)
for a in (2.15, 6.05):
    F4d.op(Opening(a, 6.05, 1.50, 2.95, 'seg', rise=0.30, ring={'w': 0.15, 'proj': 0.015}, lights=2, bars_h=0.55, orders=((0, 0.10), (0.09, 0.30)), nj=6, transom=8.15))
    F4d.op(Opening(a, 11.00, 1.50, 2.40, 'point', rise=0.95, ring={'w': 0.14, 'proj': 0.015}, tymp={'n': 2}, lights=2, bars_h=0.45, orders=((0, 0.10), (0.09, 0.30)), nj=6))
F4d.lattice(10.25, 10.70); F4d.lattice(15.25, 15.70)
F4d.finish()
plinth(F4d)
L4d = F4d.L
pins_d = list(np.linspace(0.30, L4d - 0.30, 8))
stepped_gable(F4d, 0.0, L4d, AF_EAVE, pins_d, tops_a, windows=WIN_A, rows_h=1.50, r=0.40, crown=0.38)

# ###########################################################################################
# LAUBENGANG Karmarschstrasse: Boden, Decke, Rueckwand mit Ladenfronten, Rueckseite der Bogenwand, Stirnseiten
# Die Eckboegen der Giebel (F4d a=6.05 an der Suedecke, F2a a=2.05 an der Ostecke) liegen genau vor dem Gangquerschnitt
# (c = -0.9 ... -3.5 hinter der Karmarschstrassenflucht): der Gang reicht bis zur Innenflucht der Giebelwand (Ebene
# c = -DEND im Giebelframe = Ende der Bogenleibung) und wird dort von einer Glaswand mit Tuer geschlossen (ar_shops
# mode 'screen'). Die Stirnflaeche (Innenseite der Giebelwand mit dem Bogenausschnitt) zeigt in den Gang.
# ###########################################################################################
DEND = 0.30


def _arc_a_on_plane(fr_g, cval):
    """farc-a des Punktes auf der farc-Linie c = cval, der in der Ebene c_g = -DEND des Giebelframes fr_g liegt"""
    def cg(a):
        p = S + dSE * a + n3_ * cval
        return float(fr_g.from_world([p[0], 0.0, p[1]])[2])
    a0, a1 = 0.0, 30.0
    c0, c1 = cg(a0), cg(a1)
    return a0 + (-DEND - c0) * (a1 - a0) / (c1 - c0)


aSf = _arc_a_on_plane(F4d.fr, -0.90); aSb = _arc_a_on_plane(F4d.fr, -ARC_D)
aEf = _arc_a_on_plane(F2a.fr, -0.90); aEb = _arc_a_on_plane(F2a.fr, -ARC_D)
JS_a, JE_a, KS_a, KE_a = aSf, aEf, aSb, aEb
LOG.append('Laubengang Stirnebenen: S %.2f/%.2f  E %.2f/%.2f (farc a bei c=-0.9 / -3.5)' % (aSf, aSb, aEf, aEb))
P_ARC = lambda a, c: S + dSE * a + n3_ * c
arcade_poly = Polygon([P_ARC(aSf, -0.90), P_ARC(aEf, -0.90), P_ARC(aEb, -ARC_D), P_ARC(aSb, -ARC_D)])
# Rueckseite der Bogenwand (zum Gang): Wandflaechen mit den inneren Bogenumrissen
for f, cen, span in ((F3s, cs, 2.35), (F3e, ce, 2.35), (F3m, cm, 2.60)):
    holes = []
    for a in cen:
        o = Opening(a, g3 - 0.02, span, 2.75, 'point', rise=1.55, orders=((0, 0.18), (0.14, 0.90)), skip_bottom=True, nj=5)
        holes.append(o.outline(0.14))
    d_back = 0.90 + (RIS[2] if f is F3m else 0.0)
    fb = Frame(f.fr.P([0, 0, -d_back]), f.fr.ex, f.fr.ey, f.fr.ez)
    a_s = aSf if f is F3s else 0.0                       # Suedende: bis zur Stirnebene
    a_e = (aEf - RIS[1]) if f is F3e else f.L            # Ostende: bis zur Stirnebene
    outline = [(a_s, g3 - 0.02), (a_e, g3 - 0.02), (a_e, ARC_C), (a_s, ARC_C)]
    P_ = Polygon(outline).difference(unary_union([Polygon(h) for h in holes]))
    for gg in (P_.geoms if hasattr(P_, 'geoms') else [P_]):
        ext = np.array(gg.exterior.coords)[:-1]
        mb.poly(BR, fb, [(x, y, 0) for x, y in ext], holes=[[(x, y, 0) for x, y in np.array(h.coords)[:-1]] for h in gg.interiors], hint=(0, 0, -1))
# Boden, Decke
dep0 = 0.90
g3f = max(g3, max(ground(*(S + dSE * a_ + inF3 * d_)) for a_ in np.linspace(aSf, aEf, 25) for d_ in (1.0, 2.2, 3.4)) + 0.02)
LOG.append('Laubengang Boden %.2f (Mittel Gelaende %.2f)' % (g3f, g3))
mb.poly(GR, farc, [(aSf, g3f, -dep0 + 0.01), (aEf, g3f, -dep0 + 0.01), (aEb, g3f, -ARC_D), (aSb, g3f, -ARC_D)], hint=(0, 1, 0))
mb.poly(GR, farc, [(aSf, g3 - 0.3, -dep0 + 0.01), (aEf, g3 - 0.3, -dep0 + 0.01), (aEf, g3f, -dep0 + 0.01), (aSf, g3f, -dep0 + 0.01)], hint=(0, 0, 1))
mb.poly(PL, farc, [(aSf, ARC_C, -dep0), (aEf, ARC_C, -dep0), (aEb, ARC_C, -ARC_D), (aSb, ARC_C, -ARC_D)], hint=(0, -1, 0))
# Pflaster unter den Strassenboegen (Flucht bis Setzstufe), dem Gelaende folgend (+1.5 cm): schliesst den Blick vom Gang
# unter die Bogenleibungen
gl_lo, gl_hi = 9.0, -9.0
for f, cen, span in ((F3s, cs, 2.35), (F3e, ce, 2.35), (F3m, cm, 2.60)):
    d_back = 0.90 + (RIS[2] if f is F3m else 0.0)
    for a in cen:
        xs_ = np.linspace(a - span / 2, a + span / 2, 7); cs_ = np.linspace(0.0, -d_back + 0.01, 3)
        Gp = np.zeros((len(xs_), len(cs_), 3))
        for i_, x_ in enumerate(xs_):
            for j_, c_ in enumerate(cs_):
                q_ = f.fr.P([x_, 0, c_]); gy = ground(q_[0], q_[2]) + 0.015
                gl_lo = min(gl_lo, gy); gl_hi = max(gl_hi, gy)
                Gp[i_, j_] = (x_, min(gy, g3f - 0.005), c_)
        mb.grid(GR, f.fr, Gp, smooth=False, flip=False)        # Normale nach oben (u = a, v = -c)
LOG.append('Gelaende unter den Laubengangboegen %.2f .. %.2f (Bogenfuss %.2f)' % (gl_lo, gl_hi, g3 - 0.02))
# Stirnseiten: Innenseite der Giebelwand (Ebene c = -DEND) mit dem Eckbogen-Ausschnitt, Schwelle, Glaswand
for fac, o_c, a_front_arc, a_back_arc in ((F4d, OP_CORNER_S, aSf, aSb), (F2a, OP_CORNER_E, aEf, aEb)):
    fg = fac.fr
    pf = P_ARC(a_front_arc, -0.90); pb = P_ARC(a_back_arc, -ARC_D)
    af_ = float(fg.from_world([pf[0], 0, pf[1]])[0]); ab_ = float(fg.from_world([pb[0], 0, pb[1]])[0])
    lo_, hi_ = min(af_, ab_), max(af_, ab_)
    hole = Polygon(o_c.outline(0.13)).buffer(0)
    region = sbox(lo_, g3 - 0.3, hi_, ARC_C).difference(hole)
    for gg in (region.geoms if hasattr(region, 'geoms') else [region]):
        if gg.area < 1e-4: continue
        ext = np.array(gg.exterior.coords)[:-1]
        mb.poly(BR, fg, [(x, y, -DEND) for x, y in ext], holes=[[(x, y, -DEND) for x, y in np.array(h.coords)[:-1]] for h in gg.interiors], hint=(0, 0, -1))
    # Teil des Bogenausschnitts ausserhalb des Gangquerschnitts (im Eckpfeiler): Rueckwand der Leibung, zur Strasse
    rest = hole.difference(sbox(lo_, -5, hi_, 50))
    for gg in (rest.geoms if hasattr(rest, 'geoms') else [rest]):
        if gg.geom_type != 'Polygon' or gg.area < 1e-5: continue
        ext = np.array(gg.exterior.coords)[:-1]
        mb.poly(BR, fg, [(x, y, -DEND) for x, y in ext], hint=(0, 0, 1))
    # Granitschwelle im Bogen (Strasse -> Gangboden), Seiten stecken in den Leibungen
    xi = o_c.outline(0.13); x0_, x1_ = float(xi[0, 0]), float(xi[1, 0])
    mb.box(GR, fg, x0_, x1_, min(o_c.y0, g3f - 0.05), g3f, -DEND - 0.02, 0.0, faces='YZ')      # Front nur im Bogenausschnitt (darunter Wand)
    # Kollision: Glaswand
    col.box('AR_col', fg, lo_ - 0.2, hi_ + 0.2, -1.0, ARC_C, -DEND - 0.05, 0.0)
# Gurtboegen (Querrippen) an der Decke zwischen den Jochen
rib_a = []
for f, cen, span in ((F3s, cs, 2.35), (F3e, ce, 2.35), (F3m, cm, 2.60)):
    for a in cen:
        for sx in (-1, 1):
            ax = a + sx * (span / 2 + 0.45)
            q = f.fr.P([ax, 0, 0]); aa = float(np.dot(q - farc.o, farc.ex))
            if max(aSf, aSb) + 0.4 < aa < min(aEf, aEb) - 0.4:
                rib_a.append(aa)
# benachbarte Boegen liefern fast dieselbe Rippenlage (7 cm Versatz) -> zusammenfassen, sonst doppelte, koplanare Rippen
rib_a.sort(); ribs = []
for aa in rib_a:
    if ribs and aa - ribs[-1][-1] < 0.5: ribs[-1].append(aa)
    else: ribs.append([aa])
for grp in ribs:
    aa = float(np.mean(grp))
    mb.box(BR, farc, aa - 0.20, aa + 0.20, ARC_C - 0.35, ARC_C, -ARC_D, -dep0, faces='xXy')
LOG.append('Gurtrippen Laubengang: %d (aus %d Bogenkanten)' % (len(ribs), len(rib_a)))
# Rueckwand mit Ladenfronten (9 Joche, gleichmaessig), Ladeneinheiten nach data/shops.json (Abstand ab Suedecke S)
back = Fac('arc_back', S + inF3 * ARC_D + dSE * aSb, S + inF3 * ARC_D + dSE * aEb, ARC_C + 0.05, ybot=g3f - 0.3)
NB = 9; BSP = 3.95; BW = 2.90
b_off = (back.L - (NB - 1) * BSP) / 2
bay_a = [b_off + k * BSP for k in range(NB)]                  # im Rueckwandframe
bay_farc = [aSb + a for a in bay_a]
# (Name, Ladenart, Ziel-a ab S, Joche, verifiziert) -- Sparkasse: OSM 2025 verifiziert; Viviry/Stichnoth/Atrium: OSM, unverifiziert
UNITS = [('sparkasse', 'bank', 19.0, 3), ('viviry', 'fashion', 8.3, 2), ('stichnoth', 'jewellery', 33.1, 2), ('atrium', 'cafe', 26.2, 1)]
bay_unit = [None] * NB
for nm, kind, ta, nbays in UNITS:
    free = sorted([k for k in range(NB) if bay_unit[k] is None], key=lambda k: abs(bay_farc[k] - ta))
    for k in free[:nbays]: bay_unit[k] = (nm, kind)
unit_bays = {}
for k, u in enumerate(bay_unit):
    if u: unit_bays.setdefault(u[0], []).append(k)
for k in range(NB):
    u = bay_unit[k]
    nm, kind = u if u else (None, 'generic')
    ks = unit_bays.get(nm, [k])
    door_bay = min(ks, key=lambda j: abs(bay_farc[j] - dict((x[0], x[2]) for x in UNITS).get(nm, bay_farc[k]))) if nm else k
    o = Opening(bay_a[k], g3f, BW, 2.85, 'rect', ring=None, content='shop', orders=((0, 0.06),), skip_bottom=True, n=6, nj=2)
    o.shop = dict(kind=kind, name=nm, unit=nm or ('bay%d' % k), door=1 if k == door_bay else None, prio=2 if nm else 2.4,
                  decal=nm if (nm and k != door_bay) else None, depth=3.2 if nm else 1.6, side=0.45 if nm else 0.05)
    back.op(o, low=True)
back.finish()
# Schilder: Blende ueber der Einheit (Rueckwand), Ausleger am naechsten Strassenpfeiler, beidseitig
piers = []
for f, cen, span in ((F3s, cs, 2.35), (F3e, ce, 2.35), (F3m, cm, 2.60)):
    for i in range(len(cen) - 1):
        piers.append((f, (cen[i] + cen[i + 1]) / 2, float(np.dot(f.fr.P([(cen[i] + cen[i + 1]) / 2, 0, 0]) - farc.o, farc.ex))))
for nm, ks in unit_bays.items():
    a_u = float(np.mean([bay_a[k] for k in ks]))
    fascia(mb, back.fr, a_u, g3f + 3.12, nm, h=0.42 if len(ks) > 1 else 0.36)
    fa = aSb + a_u
    f_p, a_p, _ = min(piers, key=lambda t: abs(t[2] - fa))
    blade_sign(mb, f_p.fr, a_p, g3 + 2.95, nm, size=0.60)
    LOG.append('Laden %s: Joche %s, Ausleger am Pfeiler %s a=%.2f' % (nm, ks, f_p.name, a_p))

# Traufe SO + Zwerchhaeuser + Fallrohre
eave_detail(S, S + dSE * RIS[0], AF_EAVE, AF_ROOFLINE, SLOPE_AF)
eave_detail(S + dSE * RIS[1], E, AF_EAVE, AF_ROOFLINE, SLOPE_AF)
for a in (3.32, 9.95):
    dormer(F3s, a, AF_EAVE - 0.02, w=2.90, ytop=20.30, slope_main=SLOPE_AF, roofline=AF_ROOFLINE, pin_r=0.30, crown=0.30)
dormer(F3e, 3.35, AF_EAVE - 0.02, w=2.90, ytop=20.30, slope_main=SLOPE_AF, roofline=AF_ROOFLINE, pin_r=0.30, crown=0.30)
for f, a in ((F3s, 0.35), (F3e, F3e.L - 0.35)):
    downpipe(mb, f.fr, a, ARC_C + 0.2, AF_EAVE - 0.35)
# Glasvordach vor dem Laubengang rechts vom Risalit (Foto 10)
canopy(F3e, 0.55, 12.85, ARC_C - 0.55 + g3, 3.0)
# Hofseite Karmarschstrassenfluegel (oberhalb des Atriumdachs sichtbar)
FAh = Fac('AF_hof', E2, Q, AF_EAVE + 0.35, ybot=YARD - 0.3)      # nur ueber dem Atriumdach sichtbar
for a in np.arange(2.0, FAh.L - 1.0, 3.3):
    FAh.op(Opening(a, 12.20, 1.20, 2.20, 'seg', rise=0.25, ring={'w': 0.12, 'proj': 0.0, 'alt': False}, lights=2, bars_h=0.6, orders=((0, 0.10), (0.08, 0.28)), nj=4))
FAh.lattice(15.25, 15.70)
FAh.finish()
eave_detail(E2, Q, AF_EAVE, AF_ROOFLINE, SLOPE_AF)

# ###########################################################################################
# 4) GERICHTSFLUEGEL 'Dogenpalast' / Standesamt (F4c Gs -> Q), Verbindungsbau (F4b Wb -> C1)
# ###########################################################################################
GF_DEPTH = 12.5
G2 = Gs + inF4 * GF_DEPTH


def line_isect(p, d, q, e):
    den = d[0] * e[1] - d[1] * e[0]
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + d * t


Q2 = line_isect(G2, U(Q - Gs), Q, U(E2 - Q))
F4c_lo = Fac('F4c_dogen_unten', Gs, Q, 10.45, mat=QD)        # endet an der Gesimsoberkante, wo F4c_hi (Backstein) beginnt -> keine Doppelflaeche
F4c_hi = Fac('F4c_dogen_oben', Gs, Q, GF_TOP, mat=B19, ybot=10.45)
gc = F4c_lo.g
Lc = F4c_lo.L
PAV = 5.20
bays_c = [PAV + 1.55 + 3.10 * k for k in range(8)]
round_win = lambda a, y0, w, hs, mat=QD, ring=0.16, lights=1, bars=0.45: Opening(a, y0, w, hs, 'round', ring={'w': ring, 'proj': 0.04, 'alt': False, 'm1': mat},
                                                                                  lights=lights, bars_h=bars, bars_v=1, orders=((0, 0.12), (0.10, 0.34)), nj=4, mat_reveal=mat, n=14)
# Erdgeschoss (Rustika)
F4c_lo.op(Opening(2.60, gc - 0.02, 2.05, 2.90, 'round', ring={'w': 0.30, 'proj': 0.05, 'alt': False, 'm1': QD}, content='door', door_mat=DRG,
                  orders=((0, 0.16), (0.12, 0.55)), nj=4, mat_reveal=QD, n=14), low=True)
for k, a in enumerate(bays_c):
    if k == 2:
        F4c_lo.op(Opening(a, gc + 0.45, 1.55, 2.55, 'round', ring={'w': 0.22, 'proj': 0.05, 'alt': False, 'm1': QD}, content='door', door_mat=DR,
                          orders=((0, 0.14), (0.10, 0.45)), nj=4, mat_reveal=QD, n=14), low=True)
        stair_c = a
    else:
        F4c_lo.op(round_win(a, gc + 1.35, 1.10, 2.35))
        F4c_lo.op(Opening(a, gc + 0.25, 0.70, 0.45, 'rect', ring=None, content='dark', orders=((0, 0.20),), sill=False, n=4, nj=1, mat_reveal=QD), low=True)
F4c_lo.course(5.00, h=0.30, proj=0.12, mat=QD)
# 1. OG: Bogengalerie (venezianisch)
gal = [6.0 + 1.16 * k for k in range(21)]
for a in gal:
    F4c_lo.op(Opening(a, 6.15, 0.74, 1.55, 'round', ring=None, lights=1, bars_h=0.5, bars_v=0, orders=((0, 0.10), (0.04, 0.38)), nj=3, mat_reveal=QD, n=10))
for a in (1.80, 3.40):
    F4c_lo.op(Opening(a, 6.15, 0.86, 1.70, 'round', ring=None, lights=1, bars_h=0.5, bars_v=0, orders=((0, 0.10), (0.04, 0.38)), nj=3, mat_reveal=QD, n=10))
F4c_lo.course(9.95, h=0.50, proj=0.10, mat=QD)
F4c_lo.finish()
# Saeulchen zwischen den Galeriebogen
for a in [g + 0.58 for g in gal[:-1]] + [2.60]:
    f = F4c_lo.fr.local(a, 0, 0.11)
    n_ = 8; ang = np.linspace(0, 2 * np.pi, n_, endpoint=False)
    ring_ = np.stack([np.cos(ang) * 0.085, np.zeros(n_), np.sin(ang) * 0.085], 1)
    G_ = np.stack([ring_ + [0, 6.25, 0], ring_ + [0, 7.55, 0]], 1)
    mb.grid(QD, f, G_, closed_u=True, smooth=True, flip=True)
    mb.box(QD, f, -0.15, 0.15, 7.55, 7.75, -0.15, 0.13, faces='xXyYZ')
    mb.box(QD, f, -0.13, 0.13, 6.05, 6.25, -0.13, 0.11, faces='xXyYZ')      # Kaempferblock (mit Unterseite)
# 2./3. OG (Backstein) mit Sandsteingewaenden
F4c_hi.op(round_win(2.60, 11.30, 1.45, 2.30))
for a in bays_c:
    F4c_hi.op(round_win(a, 11.30, 1.40, 2.30))
    F4c_hi.op(round_win(a, 17.40, 0.90, 1.55, ring=0.12))
F4c_hi.op(round_win(2.60, 17.40, 0.90, 1.55, ring=0.12))
F4c_hi.course(16.55, h=0.18, proj=0.07, mat=QD)
F4c_hi.finish()
# Lisenen am Pavillon
for a in (0.0, PAV):
    mb.box(QD, F4c_lo.fr, a - 0.0, a + 0.42, 10.45, GF_TOP - 1.4, 0.0, 0.12, faces='xXYZ')
plinth(F4c_lo, y_top=0.65, mat=QD)

# Kranzgesims Gerichtsfluegel (umlaufend NW, SW, SO, Hof) + Rundbogenfries
gf_chain = [G2, Gs, Q, Q2]
sweep(mb, QD, gf_chain + [G2], [(0.0, GF_TOP - 1.40), (0.10, GF_TOP - 1.40), (0.10, GF_TOP - 0.55), (0.32, GF_TOP - 0.40), (0.38, GF_TOP - 0.25),
                                 (0.38, GF_TOP + 0.02), (0.0, GF_TOP + 0.02)], closed=False, caps=True)
# Rundbogenfries (kleine Konsolboegen) unter dem Gesims auf der Strassenseite
for k in range(int(Lc / 0.60)):
    a = 0.30 + k * 0.60
    o = Opening(a, GF_TOP - 1.90, 0.42, 0.20, 'round', ring=None, content='niche', orders=((0, 0.05),), sill=False, n=6, nj=1, back_mat=B19)
# (Rundbogenfries als Relief: kleine Konsolen)
for k in range(int(Lc / 0.60) + 1):
    a = 0.0 + k * Lc / int(Lc / 0.60)
    mb.box(QD, F4c_hi.fr, a - 0.06, a + 0.06, GF_TOP - 1.95, GF_TOP - 1.40, 0.0, 0.18, faces='xXyZ')     # Konsole bis an die Wand

# NW-Stirnwand (ueber dem Verbindungsbau sichtbar) + Hofwand + SO-Wand (ueber dem Karmarschstrassenfluegel)
FGnw = Fac('GF_nw', G2, Gs, GF_TOP, mat=B19)
for a in (2.5, 6.2, 9.9):
    FGnw.op(round_win(a, 12.40, 1.20, 2.20, mat=QD))
    FGnw.op(round_win(a, 17.60, 0.85, 1.45, mat=QD, ring=0.12))
FGnw.finish()
FGh = Fac('GF_hof', Q2, G2, GF_TOP, mat=B19)
for a in np.arange(2.0, FGh.L - 1.0, 3.1):
    FGh.op(round_win(a, 12.40, 1.10, 2.10, mat=QD))
    FGh.op(round_win(a, 17.60, 0.85, 1.45, mat=QD, ring=0.12))
FGh.finish()
FGse = Fac('GF_se', Q, Q2, GF_TOP, mat=B19)
for a in (3.8, 7.8):
    FGse.op(round_win(a, 17.60, 0.85, 1.45, mat=QD, ring=0.12))
FGse.finish()

# Verbindungsbau (F4b Wb -> C1) und Ruecksprung (C1 -> Gs gehoert zu GF_nw)
F4b = Fac('F4b_verbindung', Wb, C1, CN_TOP + 0.70)
gb = F4b.g
F4b.op(win_ground(1.76, y0=2.05 + gb, w=1.25, lights=2))
F4b.op(win_up(1.76, w=1.10))
F4b.frieze(4.55, 0.55)
F4b.lattice(9.10, 9.54)
F4b.finish()
plinth(F4b)
mb.box(SS, F4b.fr, -0.05, F4b.L + 0.05, CN_TOP + 0.70, CN_TOP + 0.82, -0.40, 0.06, faces='xXyYZ')

# ###########################################################################################
# 5) HOF (Atrium) + DAECHER
# ###########################################################################################
block = Polygon([N, W, Wb, C1, Gs, Q, S, E]).buffer(0)
w_MF = Polygon([N, W, Wb, N2])
w_SF = Polygon([N2, E2, E2 + inF2 * SF_DEPTH, N2 + inF2 * SF_DEPTH])
w_AF = Polygon([S, E, E2, Q])
w_GF = Polygon([Gs, Q, Q2, G2])
yard = block.difference(unary_union([w_MF, w_SF, w_AF, w_GF])).buffer(-0.01).buffer(0.01)
LOG.append('yard area %.1f' % yard.area)

R = RoofSet()
OH = 0.38
# Marktfluegel: Region = Fluegel + Traufueberstand (NW) + Hof-Erweiterung
mf_reg = unary_union([w_MF, strip(N, W, 0.0, OH), strip(Wb, N2, -0.2, 1.2)]).buffer(0)
for (gp0, gp1) in ((W, Wb), (N2, N)):          # nicht vor die Staffelgiebel F4a / F2c (Hofueberstand-Ecken)
    tg_ = U(gp1 - gp0)
    mf_reg = mf_reg.difference(strip(gp0 - tg_ * 6.0, gp1 + tg_ * 6.0, 0.0, 6.0)).buffer(0)
R.add('MF_nw', mf_reg, N, W, MF_ROOFLINE, SLOPE_MF, 'MF')
R.add('MF_hof', mf_reg, Wb, N2, MF_ROOFLINE, SLOPE_MF, 'MF')
# Schmiedestrassenfluegel: Erweiterung bis in die Nachbarfirste
sf_core = Polygon([N2 + dNE * 7.0, E2 - dNE * 4.3, E2 - dNE * 4.3 + inF2 * SF_DEPTH, N2 + dNE * 7.0 + inF2 * SF_DEPTH])
sf_reg = unary_union([sf_core.intersection(block), strip(E2, N2, 0.0, OH), strip(N2 + inF2 * SF_DEPTH, E2 + inF2 * SF_DEPTH, -0.2, 1.2)]).buffer(0)
R.add('SF_no', sf_reg, E2, N2, SF_ROOFLINE, SLOPE_SF, 'SF')
R.add('SF_hof', sf_reg, N2 + inF2 * SF_DEPTH, E2 + inF2 * SF_DEPTH, SF_ROOFLINE, SLOPE_SF, 'SF')
# Karmarschstrassenfluegel
af_reg = unary_union([w_AF, strip(S, E, 0.0, OH), strip(Q, E2, -0.42, 1.2)]).buffer(0)
# vor dem Risalit (Staffelgiebel steht 0.25 m vor der Flucht) keinen Traufueberstand des Hauptdachs: sonst sticht der
# Ueberstand 13 cm aus der Giebelfront
af_reg = af_reg.difference(strip(S + dSE * (RIS[0] + 0.02), S + dSE * (RIS[1] - 0.02), -0.05, OH + 0.5)).buffer(0)
# nicht ueber die Giebelfronten F2a (E->E2) / F4d (Q->S) hinaus: der Hof-Ueberstand stach an der Ecke E2 bis 0.15 m vor
# die Giebelflucht, dort sah man von unten auf die Dachkante
for (gp0, gp1) in ((E, E2), (Q, S)):
    tg_ = U(gp1 - gp0)
    af_reg = af_reg.difference(strip(gp0 - tg_ * 6.0, gp1 + tg_ * 6.0, 0.0, 6.0)).buffer(0)
R.add('AF_so', af_reg, S, E, AF_ROOFLINE, SLOPE_AF, 'AF')
R.add('AF_hof', af_reg, Q, E2, AF_ROOFLINE, SLOPE_AF, 'AF')
# Querdach des Risalits (Firstlinie senkrecht zur Flucht)
rp0 = S + dSE * RIS[0]; rp1 = S + dSE * RIS[1]
ris_reg = Polygon([rp0 - inF3 * RIS[2], rp1 - inF3 * RIS[2], rp1 + inF3 * (AF_DEPTH / 2), rp0 + inF3 * (AF_DEPTH / 2)])
half_w = RIS[1] - RIS[0]
tan_x = (AF_RIDGE - AF_ROOFLINE) / (half_w / 2)
SLOPE_X = math.degrees(math.atan(tan_x))
R.add('AFx_s', ris_reg, rp0 + inF3 * 3.0, rp0, AF_ROOFLINE, SLOPE_X, 'AFx')
R.add('AFx_e', ris_reg, rp1, rp1 + inF3 * 3.0, AF_ROOFLINE, SLOPE_X, 'AFx')
# Gerichtsfluegel: flaches Walmdach (Schiefer)
gf_reg = w_GF.buffer(0.0, join_style=2)
gfc = list(gf_reg.exterior.coords)[:-1]
if area2(np.array(gfc)) < 0: gfc = gfc[::-1]
for i in range(len(gfc)):
    p0_ = np.array(gfc[i]); p1_ = np.array(gfc[(i + 1) % len(gfc)])
    R.add('GF_%d' % i, gf_reg, p0_, p1_, GF_TOP + 0.02, 7.0, 'GF')
for p in R.planes:
    if p['wing'] == 'GF': p['mat'] = SL
R.solve_min = True


def solve_all(Rs):
    BIG = 400.0
    # 1) min innerhalb eines Fluegels
    for i, pi in enumerate(Rs.planes):
        reg = pi['region']
        for j, pj in enumerate(Rs.planes):
            if i == j or pj['wing'] != pi['wing']: continue
            Ai, Bi, Ci = pi['abc']; Aj, Bj, Cj = pj['abc']
            hp = halfplane(Ai - Aj, Bi - Bj, Ci - Cj - 1e-4, BIG)      # wo i hoeher als j -> weg
            if hp is not None: reg = reg.difference(hp)
        pi['reg1'] = reg.buffer(0)
    # 2) max ueber Fluegel
    for i, pi in enumerate(Rs.planes):
        reg = pi['reg1']
        for j, pj in enumerate(Rs.planes):
            if pj['wing'] == pi['wing'] or not reg.intersects(pj['reg1']): continue
            Ai, Bi, Ci = pi['abc']; Aj, Bj, Cj = pj['abc']
            hp = halfplane(Aj - Ai, Bj - Bi, Cj - Ci + 1e-4, BIG)
            if hp is None: continue
            reg = reg.difference(pj['reg1'].intersection(hp))
        # Hof-Flachdach (y = YARD) schneidet ebenfalls
        hp = halfplane(-pi['abc'][0], -pi['abc'][1], YARD - pi['abc'][2], BIG)
        if hp is not None and pi['wing'] in ('MF', 'SF'):
            reg = reg.difference(yard_reg.intersection(hp))
        pi['vis'] = reg.buffer(0)


yard_reg = yard.union(Polygon([Wb, C1, C1 + inF4 * GF_DEPTH, Wb + inF4 * GF_DEPTH]).intersection(block)).buffer(0)
# Mauerflucht fuer die Dachuntersichten: Fluegelgrundrisse (nicht der ganze Block) -> auch die Hofueberstaende ueber dem
# Atrium-/Verbindungsbaudach bekommen eine helle Untersicht, alles innerhalb der Fluegel eine dunkle Dachschalung
wings_walls = unary_union([w_MF, w_SF, w_AF, w_GF]).buffer(0.01, join_style=2)
solve_all(R)
for p in R.planes:
    v = p['vis']
    if v.is_empty: continue
    roof_plane(mb, v, p['p0'], p['dir'], p['slope'], p['y'], mat=p.get('mat', RF), course=0.30 if p['wing'] != 'GF' else 0.24,
               step=0.028 if p['wing'] != 'GF' else 0.012, walls=wings_walls)
# First-/Gratziegel
def ridge_cap(p_a, p_b, y, r=0.13, mat=RG):
    t = U(np.asarray(p_b) - np.asarray(p_a)); L = float(np.linalg.norm(np.asarray(p_b) - p_a))
    fr = Frame([p_a[0], y, p_a[1]], [t[0], 0, t[1]], [0, 1, 0], np.cross([t[0], 0, t[1]], [0, 1, 0]))
    n = 10
    ang = np.linspace(0, np.pi, n + 1)
    # Profil: Halbrund + senkrechte Moertelwangen bis unter die Dachflaeche (schliesst den Spalt zwischen First und Dach)
    prof = [(r, -0.26)] + [(math.cos(aa) * r, math.sin(aa) * r * 0.8) for aa in ang] + [(-r, -0.26)]
    G_ = np.zeros((len(prof), 2, 3))
    for i, (cc, ss) in enumerate(prof):
        G_[i, 0] = (0, ss, cc); G_[i, 1] = (L, ss, cc)
    mb.grid(mat, fr, G_, smooth=True, flip=True)
    for a_, sg in ((0, -1), (L, 1)):
        mb.poly(mat, fr, [(a_, ss, cc) for cc, ss in prof], hint=(sg, 0, 0))


def ridge_of(wing, pa_name, pb_name):
    A = [p for p in R.planes if p['name'] == pa_name][0]; B_ = [p for p in R.planes if p['name'] == pb_name][0]
    seg = A['vis'].buffer(0.05).intersection(B_['vis'].buffer(0.05))
    if seg.is_empty: return
    # Firstlinie = Schnitt der Ebenen
    Ai, Bi, Ci = A['abc']; Aj, Bj, Cj = B_['abc']
    a, b, c = Ai - Aj, Bi - Bj, Ci - Cj
    g = math.hypot(a, b); n = np.array([a, b]) / g; p0 = -c / g * n; t = np.array([-n[1], n[0]])
    ln = LineString([p0 - t * 500, p0 + t * 500]).intersection(seg)
    if ln.is_empty: return
    for gl in (ln.geoms if hasattr(ln, 'geoms') else [ln]):
        if gl.geom_type != 'LineString' or gl.length < 0.3: continue
        cs_ = np.array(gl.coords)
        ya = Ai * cs_[0, 0] + Bi * cs_[0, 1] + Ci
        ridge_cap(cs_[0], cs_[-1], ya + 0.01)


for pa_, pb_ in (('MF_nw', 'MF_hof'), ('SF_no', 'SF_hof'), ('AF_so', 'AF_hof'), ('AFx_s', 'AFx_e')):
    ridge_of(None, pa_, pb_)

# Hof-Flachdach + Glasdach (Atrium)
yv = yard_reg
for p in R.planes:
    if p['wing'] in ('MF', 'SF'):
        pass
yflat = yv
for gg in (yflat.geoms if hasattr(yflat, 'geoms') else [yflat]):
    ext = np.array(gg.exterior.coords)[:-1]
    idf = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    mb.poly(BIT, idf, [(x, YARD, z) for x, z in ext], holes=[[(x, YARD, z) for x, z in np.array(h.coords)[:-1]] for h in gg.interiors], hint=(0, 1, 0))
# Attika des Verbindungsbaus zur Strasse ist oben; Glasdach ueber dem Hofkern
core = yard.buffer(-1.6, join_style=2)
if not core.is_empty:
    core = max(core.geoms, key=lambda g: g.area) if hasattr(core, 'geoms') else core
    rect = core.minimum_rotated_rectangle
    rc = np.array(rect.exterior.coords)[:-1]
    e0 = rc[1] - rc[0]; e1 = rc[2] - rc[1]
    if np.linalg.norm(e0) < np.linalg.norm(e1):
        rc = np.roll(rc, -1, 0); e0 = rc[1] - rc[0]; e1 = rc[2] - rc[1]
    Lr = np.linalg.norm(e0); Wr = np.linalg.norm(e1)
    fr_g = Frame([rc[0][0], YARD, rc[0][1]], [e0[0] / Lr, 0, e0[1] / Lr], [0, 1, 0], np.cross([e0[0] / Lr, 0, e0[1] / Lr], [0, 1, 0]))
    sgn = 1 if np.dot(fr_g.ez[[0, 2]], e1) > 0 else -1
    # Aufkantung
    zz = sgn * Wr
    mb.box(ST, fr_g, 0, Lr, 0, 0.35, min(0, zz), max(0, zz), faces='xXYzZ')
    hr = 1.4
    a_ = (0, 0.35, 0); b_ = (Lr, 0.35, 0); c_ = (Lr, 0.35 + hr, zz / 2); d_ = (0, 0.35 + hr, zz / 2)
    e_ = (Lr, 0.35, zz); f_ = (0, 0.35, zz)
    mb.poly(GD, fr_g, [a_, b_, c_, d_], hint=(0, 1, -sgn)); mb.poly(GD, fr_g, [f_, d_, c_, e_], hint=(0, 1, sgn))
    mb.poly(GD, fr_g, [a_, d_, f_], hint=(-1, 0, 0)); mb.poly(GD, fr_g, [b_, e_, c_], hint=(1, 0, 0))
    for x in np.arange(0, Lr + 0.01, 1.2):
        for (p_, q_) in ((a_, d_), (f_, d_)):
            pp = (x, p_[1], p_[2]); qq = (x, q_[1], q_[2])
            v = np.array(qq) - np.array(pp); Lv = np.linalg.norm(v)
            mb.box(ST, Frame(fr_g.P(pp), fr_g.ex, fr_g.D(v / Lv), np.cross(fr_g.ex, fr_g.D(v / Lv))), -0.04, 0.04, 0, Lv, 0.0, 0.07, faces='xXZ')
    mb.box(ST, fr_g, 0, Lr, 0.35 + hr - 0.03, 0.35 + hr + 0.08, zz / 2 - 0.08, zz / 2 + 0.08, faces='xXYzZ')

# ###########################################################################################
# 5b) INNENSCHALE: dunkle Waende 0.8 m hinter den Fassaden, Normalen nach innen (von aussen gecullt, unsichtbar).
# Wer durch eine Haarfuge zwischen zwei Bauteilen (Quantisierung im Web-GLB) oder eine Oeffnung ins leere Innere blickt,
# sieht dunkles Mauerwerk statt (FrontSide-gecullt) durch das Gebaeude hindurch. Liegt hinter allen Laibungen
# (tiefste Laibung + Fuellung < 0.65 m) und wird von den geschlossenen Ladenraeumen verdeckt.
# ###########################################################################################
idf_ = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])


def inner_shell(poly, y0, y1, inset=0.8, mat='AR_Innen'):
    P = poly.buffer(-inset, join_style=2)
    n = 0
    for g in (P.geoms if hasattr(P, 'geoms') else [P]):
        if g.geom_type != 'Polygon' or g.area < 2.0: continue
        o = np.array(g.simplify(0.02).exterior.coords)[:-1]
        if area2(o) > 0: o = o[::-1]
        for i in range(len(o)):
            p, q = o[i], o[(i + 1) % len(o)]
            mb.quad(mat, idf_, (q[0], y0, q[1]), (p[0], y0, p[1]), (p[0], y1, p[1]), (q[0], y1, q[1]))     # nach innen
            n += 1
    return n


conn_poly = Polygon([Wb, C1, C1 + inF4 * 12.0, Wb + inF4 * 12.0]).intersection(block).difference(unary_union([w_MF, w_SF, w_AF, w_GF])).buffer(0)


def build_inner_shell(rooms):
    # Ladenraeume (Planpolygone aus ar_shops.emit_rooms) aussparen: die Schale darf keinen Raum durchschneiden
    holes_ = unary_union([arcade_poly] + list(rooms)).buffer(0.08, join_style=2)
    nsh = 0
    for poly_, ytop_ in ((w_MF, MF_EAVE - 0.25), (w_SF, SF_EAVE - 0.25), (w_AF, AF_EAVE - 0.25), (w_GF, GF_TOP - 0.40), (conn_poly, CN_TOP - 0.30)):
        nsh += inner_shell(poly_.difference(holes_), YB, ytop_)
    LOG.append('Innenschale: %d Wandflaechen' % nsh)

# ###########################################################################################
# 6) KOLLISION (vereinfachte Koerper)
# ###########################################################################################
idf = Frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
def colprism(poly, y0, y1):
    ext = np.array(poly.exterior.coords)[:-1]
    col.prism('AR_col', idf, ext, y0, y1, top=True, bottom=False)
colprism(w_MF, -1.0, MF_EAVE); colprism(w_SF, -1.0, SF_EAVE); colprism(w_GF, -1.0, GF_TOP)
colprism(yard_reg.convex_hull.intersection(block).difference(unary_union([w_MF, w_SF, w_GF, w_AF])).buffer(-0.05) if False else yard, -1.0, YARD)
af_back = Polygon([S + inF3 * ARC_D, E + inF3 * ARC_D, E2, Q]).intersection(w_AF)
colprism(af_back if af_back.geom_type == 'Polygon' else max(af_back.geoms, key=lambda g: g.area), -1.0, AF_EAVE)
colprism(Polygon([S, E, E + inF3 * 0.9, S + inF3 * 0.9]), ARC_C - 0.2, AF_EAVE)
for f, cen, span in ((F3s, cs, 2.35), (F3e, ce, 2.35), (F3m, cm, 2.60)):
    edges = [0.0] + [x for a in cen for x in (a - span / 2, a + span / 2)] + [f.L]
    for k in range(0, len(edges), 2):
        if edges[k + 1] - edges[k] > 0.05:
            col.box('AR_col', f.fr, edges[k], edges[k + 1], -1.0, ARC_C, -0.9, 0.0)
colprism(Polygon([Wb, C1, C1 + inF4 * 12.0, Wb + inF4 * 12.0]), -1.0, CN_TOP)

# ###########################################################################################
# Export
# ###########################################################################################
# Ladenraeume zuteilen (nach allen Ladenfronten) + Auslagen-Prototypen
emit_rooms(mb, allowed=block.buffer(-0.35, join_style=2), obstacles=[arcade_poly])
LOG.extend(SHOPSTATE.log)
build_inner_shell(SHOPSTATE.rooms)
protos = {'rosette': proto_rosette(), 'crown': proto_crown(), 'figure': proto_figure(), 'frieze': proto_frieze(np.load(os.path.join(TEX, 'frieze_height.npy')), nu=44, nv=9)}
protos.update(display_protos())
for p in protos.values(): p.tiles = T
st = {}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ntri = export(OUT, mb, protos, colmb=col, tex_dir=TEX, stats=st,
              extras={'name': 'Altes Rathaus Hannover', 'source': 'handmodelliert (pipeline/landmarks/altes_rathaus)',
                      'lod2_id': 'ID_dedae90c-48f4-3b9e-9176-f04541ffbcb2'})
# hoechster Punkt
allY = []
for m, d in mb.data.items():
    for P in d[0]: allY.append(P[:, 1].max())
top = float(max(allY)) + 2.3
json.dump({'top': round(top, 2), 'tris': ntri, 'groups': st['groups'], 'bytes': st['bytes'], 'log': LOG},
          open(os.path.join(HERE, 'build', 'model_top.json'), 'w'), indent=1)
print('\n'.join(LOG))
print('RATHAUS tris %d bytes %d top %.1f' % (ntri, st['bytes'], top))
for k, v in sorted(st['groups'].items(), key=lambda x: -x[1])[:30]:
    print('  %-28s %d' % (k, v))

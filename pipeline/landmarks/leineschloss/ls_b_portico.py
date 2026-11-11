"""Leineschloss -- Portikus (Laves): Freitreppe, Podest, 6 kannelierte korinthische Saeulen, Gebaelk mit Anthemionfries,
Konsolgesims, Giebel mit Wappenrelief, Stehfalzdach, Pronaos-Rueckwand mit Halbsaeulen, Tueren, Lunetten, Kassettendecke,
Seitenfelder (Nischen, Inschrift), Gelaender."""
import math
import numpy as np
import lslib as L
import ls_parts as P
import ls_fac as F
from lslib import Builder, UP
from ls_fac import SAND, QUAD, RUST, GRAN, ZINK, PW, PLAT, FR, FRD, GL, EIS, EDS, BRZ, DUNK

HG = -1.45           # Platz vor dem Portikus
HP = 0.35            # Podest (Stylobat)
D, DT = 1.25, 1.06   # Saeulendurchmesser unten/oben
COLU = [-11.75, -7.05, -2.35, 2.35, 7.05, 11.75]
COLV = -1.15
H_CAPTOP = 12.2      # Oberkante Kapitell
H_ARCH = 13.05       # Oberkante Architrav
H_FRIEZE = 13.95     # Oberkante Fries
H_CT = 15.05         # Oberkante Kranzgesims
EF = -0.62           # Gebaelk-Vorderflaeche (v)
EU = 12.28           # Gebaelk-Seitenflaeche (|u|)
VB = -7.0            # Pronaos-Rueckwand
VR = -13.5           # Ende Giebeldach
T15 = math.tan(math.radians(15.2))
RISE = 0.15; NSTEP = 12; TREAD = 0.40; V_POD = -0.1; HW0 = 7.9; HW_STEP = 0.62
BLOCK_U = 13.4
SIDE_U = 17.6        # Aussenkante Seitenfelder


def build(B, ctx):
    n0 = B.tri_count()[0]
    stairs(B)
    podium(B)
    columns(B, ctx)
    entablature(B, ctx)
    pediment_roof(B, ctx)
    pronaos(B, ctx)
    side_bays(B, ctx)
    ctx.log('portico', B.tri_count()[0] - n0)


# ------------------------------------------------------------------ Freitreppe (12 Stufen, dreiseitig, pyramidal verbreitert)
def stairs(B):
    for k in range(1, NSTEP + 1):
        top = HP - RISE * (k - 1)            # Oberkante Stufe k ... (k=1: Podest-Absatz)
        top = HP - RISE * k
        hw = HW0 + HW_STEP * k; vf = V_POD + TREAD * k
        bot = top - RISE if k < NSTEP else HG - 0.35
        hw_up = HW0 + HW_STEP * (k - 1); vf_up = V_POD + TREAD * (k - 1)
        # Trittflaeche: Rechteck k minus Rechteck k-1 (U-Form, hinten offen bei V_POD)
        if k == 1:
            ring = np.array([[-hw, V_POD], [hw, V_POD], [hw, vf], [-hw, vf]])
            # hinter Stufe 1 liegt das Podest zwischen den Bloecken -> volle Flaeche
            B.add(GRAN, L.tri_poly(np.c_[ring, np.full(4, top)], [0, 0, 1]), 'main')
        else:
            ring = np.array([[-hw, V_POD], [-hw_up, V_POD], [-hw_up, vf_up], [hw_up, vf_up], [hw_up, V_POD], [hw, V_POD], [hw, vf], [-hw, vf]])
            B.add(GRAN, L.tri_poly(np.c_[ring, np.full(len(ring), top)], [0, 0, 1]), 'main')
        # Setzstufe vorn + Seiten
        T = L.box_tris(-hw, hw, V_POD, vf, bot, top, skip=('b', 't', 'v0'))
        B.add(GRAN, T, 'main')
        # Rueckseite nur ausserhalb der Podestbloecke sichtbar
        if hw > BLOCK_U:
            for sg in (-1, 1):
                a, c = sorted([sg * BLOCK_U, sg * hw])
                Q = np.array([[[a, V_POD, bot], [a, V_POD, top], [c, V_POD, top], [c, V_POD, bot]]])
                B.add(GRAN, L.fix_orient(L.quads(Q), [0, -1, 0]), 'main')
        # Stufenkante (Fase) als schmale Leiste
        B.add(GRAN, L.box_tris(-hw, hw, vf - 0.03, vf + 0.012, top - 0.035, top - 0.005, skip=('b',)), 'det')
    # Edelstahl-Handlaeufe entlang der Stufenenden (beidseitig) und Gelaender auf den Podestbloecken
    for sg in (-1, 1):
        pts = []
        for k in (1, NSTEP):
            top = HP - RISE * k
            pts.append([sg * (HW0 + HW_STEP * k - 0.35), V_POD + TREAD * k - 0.22, top])
        p0, p1 = np.array(pts[0]), np.array(pts[1])
        handrail(B, p0, p1)
        F.railing(B, (sg * (HW0 + 0.15), V_POD - 0.18), (sg * (BLOCK_U - 0.15), V_POD - 0.18), HP, 1.0, 1.35, EDS, infill='x')
        # kurze Handlaeufe vor den Seitentueren (Podest)
        for uu in (sg * 4.7 - 0.9, sg * 4.7 + 0.9):
            F.railing(B, (uu, VB + 0.6), (uu, VB + 2.2), HP, 0.95, 1.6, EDS, infill='none')


def handrail(B, p0, p1, height=0.95, mat=EDS):
    d = p1 - p0
    side = L.nrm(np.cross(d, UP)); bv = L.nrm(np.cross(side, d))
    ring = [(0.024 * math.cos(a), 0.024 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    B.add(mat, L.sweep_tris(np.array([p0 + UP * height, p1 + UP * height]), ring, bv=bv, caps=True), 'det', closed=True)
    B.add(mat, L.sweep_tris(np.array([p0 + UP * 0.45, p1 + UP * 0.45]), [(r * 0.6, s * 0.6) for r, s in ring], bv=bv, caps=True), 'det', closed=True)
    n = int(np.linalg.norm(d[:2]) / 1.3) + 1
    for i in range(n + 1):
        q = p0 + d * i / n
        B.add(mat, L.lathe_tris([(0.022, q[2] - 0.05), (0.022, q[2] + height)], 8, (q[0], q[1], 0), cap_top=True), 'det')


# ------------------------------------------------------------------ Podest + Bossenbloecke
def podium(B):
    # Podest-Oberflaeche (Granitplatten) zwischen Vorderkante und Rueckwand
    ring = np.array([[-BLOCK_U, V_POD], [BLOCK_U, V_POD], [BLOCK_U, VB], [-BLOCK_U, VB]])
    B.add(GRAN, L.tri_poly(np.c_[ring, np.full(4, HP)], [0, 0, 1]), 'main')
    for sg in (-1, 1):
        a, c = sorted([sg * HW0, sg * BLOCK_U])
        # Block: Vorderseite (Bossenquader als echte Geometrie) + Seite
        W = L.Wall((a, V_POD), (c, V_POD)) if sg < 0 else L.Wall((a, V_POD), (c, V_POD))
        W = L.Wall((c, V_POD), (a, V_POD))   # Normale +v ... t=(-1,0) -> n=(0,1)
        Lw = W.L
        # Rustika-Quader: 3 Schichten je 0.6 m, versetzt, Fugen 3 cm, Bossen 4 cm vor
        h0 = HG - 0.35; rows = [(h0, -1.05), (-1.05, -0.45), (-0.45, 0.15)]
        F.wbox(B, RUST, W, 0, Lw, h0, HP, -0.06, -0.0, 'main')        # Kern (bleibt hinter den Quadern)
        for r, (z0, z1) in enumerate(rows):
            n = 4 if r % 2 == 0 else 5
            for i in range(n):
                s0 = Lw * i / n; s1 = Lw * (i + 1) / n
                T = L.bevel_box_tris(s0 + 0.015, s1 - 0.015, z0 + 0.015, z1 - 0.015, -0.02, 0.05, c=0.025)
                M = L.mat_frame(W.P(0, 0, 0) * np.array([1, 1, 0]), W.t, UP, W.n)
                T = L.xform(T, M)
                if np.linalg.det(M[:3, :3]) < 0: T = T[:, ::-1]
                B.add(SAND, T, 'main', closed=True)
        # Deckplatte (Abschlussgesims) unter den Saeulen
        F.wall_hsweep(B, SAND, W, -0.0, Lw, 0.15, [(0.0, 0.0), (0.08, 0.0), (0.08, 0.12), (0.12, 0.16), (0.12, 0.2), (0.0, 0.2)], 'det')
        # Aussenseite des Blocks
        Ws = L.Wall((sg * BLOCK_U, V_POD), (sg * BLOCK_U, VB)) if sg > 0 else L.Wall((sg * BLOCK_U, VB), (sg * BLOCK_U, V_POD))
        F.band_wall(B, Ws, 0, Ws.L, h0, HP, RUST, [], 'main')
    # Podest-Front zwischen den Bloecken (hinter Stufe 1 verdeckt) -> nicht noetig


# ------------------------------------------------------------------ Saeulen
def columns(B, ctx):
    cap, HC = P.corinthian_capital(DT, SAND, segs=40, detail=1.0)
    base = Builder(); hb = P.attic_base(base, D, SAND, segs=40)
    shaft = Builder(); H_SH = H_CAPTOP - HC - (HP + hb)
    P.fluted_shaft(shaft, D / 2, DT / 2, H_SH, 0.0, 24, SAND, rows=18)
    Ms = []
    for u in COLU:
        Ms.append(L.trans(u, COLV, HP))
    B.add_instances('Saeule_Basis', base, Ms)
    B.add_instances('Saeule_Schaft', shaft, [M @ L.trans(0, 0, hb) for M in Ms])
    B.add_instances('Saeule_Kapitell', cap, [M @ L.trans(0, 0, hb + H_SH) for M in Ms])
    # Halbsaeulen an der Rueckwand (gleiche Ordnung), Achse in der Wandflaeche
    capH, _ = P.corinthian_capital(DT, SAND, segs=40, detail=0.8, half=True)
    baseH = Builder(); P.attic_base(baseH, D, SAND, segs=24, half=True)
    shaftH = Builder(); P.fluted_shaft(shaftH, D / 2, DT / 2, H_SH, 0.0, 24, SAND, rows=12, half=True)
    MsH = [L.trans(u, VB, HP) for u in COLU]
    B.add_instances('Halbsaeule_Basis', baseH, MsH)
    B.add_instances('Halbsaeule_Schaft', shaftH, [M @ L.trans(0, 0, hb) for M in MsH])
    B.add_instances('Halbsaeule_Kapitell', capH, [M @ L.trans(0, 0, hb + H_SH) for M in MsH])
    ctx.col_h = (hb, H_SH, HC)


# ------------------------------------------------------------------ Gebaelk
def entablature(B, ctx):
    # Pfad gegen Uhrzeigersinn um den Portikus: rechts hinten -> vorn -> links hinten
    path = np.array([[EU, VR], [EU, EF], [-EU, EF], [-EU, VR]])
    path_arch = np.array([[EU, VB], [EU, EF], [-EU, EF], [-EU, VB]])
    # Architrav: drei Faszien + Abschlussprofil
    prof = [(0.0, 0.0), (0.0, 0.25), (0.03, 0.25), (0.03, 0.52), (0.06, 0.52), (0.06, 0.75), (0.1, 0.8), (0.12, 0.85), (0.0, 0.85)]
    F.hsweep(B, SAND, path_arch, H_CAPTOP, prof, closed=False, grp='main')
    # Fries (Flaeche) + Anthemion-Relief
    F.hsweep(B, SAND, path_arch, H_ARCH, [(0.0, 0.0), (0.0, H_FRIEZE - H_ARCH)], closed=False, grp='main')
    for W, s0, s1 in fronts(path_arch):
        Lg = s1 - s0
        Hm = ctx.anthemion(Lg, H_FRIEZE - H_ARCH - 0.12)
        T, on = L.relief_tris(Hm, W, s0, H_ARCH + 0.06, Lg, H_FRIEZE - H_ARCH - 0.12, 0.07, 0.0, thr=0.08)
        B.add(SAND, T, 'det')
    # Kranzgesims mit Konsolen und Zahnschnitt (laeuft an den Flanken bis zum Dachende)
    F.main_cornice(B, path, H_FRIEZE, H_CT - H_FRIEZE, 0.64, closed=False, modillions=0.62, dentils=True, mat=SAND, grp='det')
    # Unterzuege + Decke des Pronaos
    B.add(SAND, L.box_tris(-EU, EU, EF - 1.06, EF, H_CAPTOP, H_FRIEZE, skip=('t', 'v1')), 'main')
    for sg in (-1, 1):
        a, c = sorted([sg * EU, sg * (EU - 1.06)])
        B.add(SAND, L.box_tris(a, c, VB, EF - 1.06, H_CAPTOP, H_FRIEZE, skip=('t', 'v1', 'v0', 'u1' if sg > 0 else 'u0')), 'main')
    for u in COLU[1:-1]:
        B.add(SAND, L.box_tris(u - 0.45, u + 0.45, VB, EF - 1.06, H_CAPTOP + 0.4, 13.45, skip=('t', 'v0', 'v1')), 'det')
    # Kassettendecke (weiss) zwischen den Unterzuegen, mit Rippen
    B.add(PW, L.tri_poly(np.array([[-EU + 1.06, VB, 13.45], [EU - 1.06, VB, 13.45], [EU - 1.06, EF - 1.06, 13.45], [-EU + 1.06, EF - 1.06, 13.45]]), [0, 0, -1]), 'det')
    for i in range(len(COLU) - 1):
        a, c = COLU[i] + (0.45 if 0 < i else 0.55), COLU[i + 1] - (0.45 if i + 1 < len(COLU) - 1 else 0.55)
        for vv in np.linspace(VB, EF - 1.06, 4)[1:-1]:
            B.add(SAND, L.box_tris(a, c, vv - 0.12, vv + 0.12, 13.25, 13.45, skip=('t',)), 'det')
    # Rueckwand oberhalb der Pronaos-Decke bis unter das Dach (verdeckt) -> Abschluss oben durch Dach


def fronts(path):
    """Wandsegmente (W, s0, s1) entlang eines offenen Linienzugs gegen Uhrzeigersinn."""
    out = []
    for i in range(len(path) - 1):
        W = L.Wall(path[i], path[i + 1])
        out.append((W, 0.0, W.L))
    return out


# ------------------------------------------------------------------ Giebel + Dach
def pediment_roof(B, ctx):
    Wf = L.Wall((EU, EF), (-EU, EF))            # Giebelflaeche, Normale +v
    he = lambda u: H_FRIEZE + (EU - abs(u)) * T15
    apex = he(0.0)
    # Tympanon (zurueckliegend, Quader)
    tri = np.array([[0.0, H_CT], [2 * EU, H_CT], [EU, apex]])          # s: 0..2EU (Wf.t = -u)
    tri_s = np.array([[Wf.L - 0.0, H_CT], [0.0, H_CT], [EU, apex]])
    L.wall_face(B, QUAD, Wf, np.array([[0.0, H_CT], [Wf.L, H_CT], [EU, apex]]), [], -0.05, 'main')
    # Wappenrelief
    tw, th = 2 * EU - 1.0, apex - H_CT - 0.05
    Hm = ctx.tympanum(tw, th)
    ny, nx = Hm.shape
    ss = (np.arange(nx) + 0.5) / nx * tw + 0.5; hh = H_CT + th * (1 - (np.arange(ny) + 0.5) / ny)
    S_, H_ = np.meshgrid(ss, hh)
    inside = H_ < (H_CT + (EU - np.abs(S_ - EU)) * T15 - 0.12)
    Hm = Hm * inside
    T, on = L.relief_tris(Hm, Wf, 0.5, H_CT, tw, th, 0.32, -0.05, thr=0.06)
    B.add(SAND, T, 'det')
    # Schraeggesims mit Konsolen
    ext = 0.66
    pl = np.array([[-EU - ext, he(EU) - ext * T15], [0.0, apex], [EU + ext, he(EU) - ext * T15]])
    path = np.c_[pl[:, 0], np.full(3, EF), pl[:, 1]]
    hp = [(y, x) for (x, y) in P.prof_cornice(0.64, H_CT - H_FRIEZE)]
    hp = [(hp[0][0], -0.02)] + hp + [(hp[-1][0], -0.02)]
    T = L.sweep_tris(path, hp, bv=np.array([0, 1.0, 0]), closed=False, caps=True)
    B.add(SAND, T, 'det', closed=True)
    mod = P.modillion(0.14, 0.64 * 0.55, (H_CT - H_FRIEZE) * 0.2)
    for sg in (-1, 1):
        p0 = np.array([0.0, EF, apex]); p1 = np.array([sg * (EU + 0.2), EF, he(EU) - 0.2 * T15])
        d = p1 - p0; Lg = np.linalg.norm(d); t = d / Lg
        perp = L.nrm(np.cross(np.array([0, 1.0, 0]), t)) * (-sg)
        if perp[2] < 0: perp = -perp
        n_m = int(Lg / 0.62)
        for i in range(1, n_m + 1):
            o = p0 + t * (Lg * i / (n_m + 0.5)) + perp * ((H_CT - H_FRIEZE) * 0.36)
            M = L.mat_frame(o, np.array([0, 1.0, 0]), t, perp)
            if np.linalg.det(M[:3, :3]) < 0: M = L.mat_frame(o, np.array([0, 1.0, 0]), -t, perp)
            for (m_, g_), (Ts, UVs) in mod.G.items():
                for T_ in Ts: B.add(SAND, L.xform(T_, M), 'det', closed=True)
        nd = int(Lg / 0.17); Q = []
        for i in range(1, nd):
            c0 = p0 + t * (Lg * i / nd) + perp * ((H_CT - H_FRIEZE) * 0.17) + np.array([0, 0.13, 0])
            Q.append(L.obox_tris(c0, t, np.array([0, 1.0, 0]), perp, 0.045, 0.06, 0.05))
        B.add(SAND, np.concatenate(Q), 'det')
    # Dachflaechen (Stehfalz, Zink) von der Giebelkante bis VR
    top = lambda u: he(u) + (H_CT - H_FRIEZE) / math.cos(math.radians(15.2)) - 0.04
    ue = EU + 0.68
    for sg in (-1, 1):
        q = np.array([[0.0, -0.45, top(0)], [sg * ue, -0.45, top(ue)], [sg * ue, VR, top(ue)], [0.0, VR, top(0)]])
        B.add(ZINK, L.fix_orient(L.quads(q[None]), [sg * 0.25, 0, 1]), 'main')
        # Falze alle 0.52 m (von First zur Traufe)
        k = 0
        for vv in np.arange(-0.9, VR, -0.52):
            p0 = np.array([0.0, vv, top(0)]); p1 = np.array([sg * ue, vv, top(ue)])
            d = p1 - p0; t = d / np.linalg.norm(d); nn = L.nrm(np.cross(t, np.array([0, 1.0, 0]))) * (-1)
            if nn[2] < 0: nn = -nn
            c = (p0 + p1) / 2 + nn * 0.025
            B.add(ZINK, L.obox_tris(c, t, np.array([0, 1.0, 0]), nn, np.linalg.norm(d) / 2, 0.012, 0.025), 'det', closed=True)
        # Traufkante / Rinne
        B.add(ZINK, L.box_tris(min(sg * ue, sg * (ue + 0.12)), max(sg * ue, sg * (ue + 0.12)), VR, -0.5, top(ue) - 0.16, top(ue) + 0.02), 'det')
    # Firstkappe
    roll = [(0.06 * math.cos(a), 0.06 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    B.add(ZINK, L.sweep_tris(np.array([[0, -0.4, top(0) + 0.02], [0, VR, top(0) + 0.02]]), roll, bv=UP, caps=True), 'det', closed=True)
    # Rueckgiebel (Quader) ueber dem Hallendach
    Wb = L.Wall((-EU, VR), (EU, VR))           # Normale -v
    poly = np.array([[0.0, 12.6], [2 * EU, 12.6], [2 * EU, top(EU) - 0.05], [EU, top(0) - 0.05], [0.0, top(EU) - 0.05]])
    L.wall_face(B, QUAD, Wb, poly, [], 0.0, 'main')
    # Flanken unter dem Gesims (zwischen Pronaos-Rueckwand und VR) bis Seitenfeld-Dach
    for sg in (-1, 1):
        W = L.Wall((sg * EU, VB), (sg * EU, VR)) if sg > 0 else L.Wall((sg * EU, VR), (sg * EU, VB))
        F.band_wall(B, W, 0, W.L, 13.4, H_FRIEZE, QUAD, [], 'main')
        # Ortgang-Abdeckung der Rueckseite: Dachrand
    ctx.portico_top = top(0)


# ------------------------------------------------------------------ Pronaos-Rueckwand
def pronaos(B, ctx):
    W = L.Wall((-EU, VB), (EU, VB))     # t=+u -> n = (0,-1)?  -> umdrehen: Normale soll +v sein
    W = L.Wall((EU, VB), (-EU, VB))     # t = -u, n = +v ; s = EU - u
    s_of = lambda u: EU - u
    holes = []
    # Mittelfeld: Tuer + Lunette; Felder 2/4: Tueren + Lunetten; Felder 1/5: hohe Fenster + Lunetten
    door_c = [0.0, -4.7, 4.7]
    for uc in door_c:
        w = 2.0 if uc == 0 else 1.5
        sp = F.WSpec(w, HP, 3.95, 'rect', bars=(2, 2), depth=0.45, frame=(0.32, 0.08), fmat=FRD, sill=False, transom=3.2, reveal=SAND)
        holes.append(F.place_window(B, W, s_of(uc), sp, 'det', ctx))
        r = 1.25 if uc == 0 else 0.95
        sp = F.WSpec(2 * r, 6.0 - 0.05, 6.0 + r, 'arch', bars=(2, 1), depth=0.35, fmat=FRD, sill=False, arch_frame=(0.22, 0.06), keystone=True)
        holes.append(F.place_window(B, W, s_of(uc), sp, 'det', ctx))
    for uc in (-9.4, 9.4):
        sp = F.WSpec(1.9, 1.4, 5.0, 'rect', bars=(2, 4), depth=0.35, frame=(0.25, 0.07), fmat=FRD, sill=True)
        holes.append(F.place_window(B, W, s_of(uc), sp, 'det', ctx))
        sp = F.WSpec(2.3, 6.15, 6.15 + 1.15, 'arch', bars=(2, 1), depth=0.35, fmat=FRD, sill=False, arch_frame=(0.22, 0.06), keystone=True)
        holes.append(F.place_window(B, W, s_of(uc), sp, 'det', ctx))
    F.band_wall(B, W, 0, W.L, HP, 13.45, QUAD, holes, 'main')
    # Kaempfergesims zwischen den Halbsaeulen
    for i in range(len(COLU) - 1):
        a, c = COLU[i] + 0.62, COLU[i + 1] - 0.62
        F.wall_hsweep(B, SAND, W, s_of(c), s_of(a), 5.55, P.prof_string(0.1, 0.25), 'det')
    # Inschrift ueber der Mitteltuer
    ctx.texts.append(dict(text='NIEDERSÄCHSISCHER LANDTAG', size=0.2, extrude=0.012, mat=BRZ,
                          pos=list(W.P(s_of(0.0), 4.42, 0.04)), x=list(W.t), z=list(W.n), grp='fine'))


# ------------------------------------------------------------------ Seitenfelder + Verbindungsbau Nord
def side_bays(B, ctx):
    for sg in (-1, 1):
        u_in, u_out = sg * EU, sg * SIDE_U
        # Front (v = VB), Normale +v
        W = L.Wall((max(u_in, u_out), VB), (min(u_in, u_out), VB))
        s_of = lambda u, W=W: np.dot(np.array([u, VB]) - W.o, W.t[:2])
        sc = s_of(sg * 15.0)
        holes = []
        # Nische (Rundbogen), Tiefe 0.6
        nh = L.arch2d(sc - 1.3, sc + 1.3, HP, 2.75, 14)
        L.reveal(B, QUAD, W, nh, 0.0, -0.6, 'main', skip_bottom=True)
        B.add(QUAD, L.tri_poly(W.poly(nh, -0.6), W.n), 'main')
        B.add(GRAN, L.tri_poly(np.c_[np.array([W.P(sc - 1.3, HP, 0)[:2], W.P(sc + 1.3, HP, 0)[:2], W.P(sc + 1.3, HP, -0.6)[:2], W.P(sc - 1.3, HP, -0.6)[:2]]), np.full(4, HP + 0.005)], [0, 0, 1]), 'main')
        F.place_window(B, W, sc, F.WSpec(2.6, HP, 4.05, 'arch', door=False, sill=False, arch_frame=(0.28, 0.07), keystone=True,
                                          depth=0.0, glass=QUAD, fmat=QUAD, bars=(1, 1)), 'det', ctx) if False else None
        T = L.sweep_tris(W.poly(nh[2:], 0.0), [(0.0, -0.01)] + P.prof_architrave(0.28, 0.07)[1:-1] + [(0.28, -0.01)], bv=W.n, closed=False, caps=True)
        B.add(SAND, T, 'det', closed=True)
        holes.append(nh)
        if sg < 0:
            # "Einigkeit / Recht / Freiheit" (dunkle Metallbuchstaben) + Bronzetafel
            for k, word in enumerate(('Einigkeit', 'Recht', 'Freiheit')):
                ctx.texts.append(dict(text=word, size=0.42, extrude=0.03, mat=DUNK, pos=list(W.P(sc, 3.1 - k * 0.48, -0.56)),
                                      x=list(W.t), z=list(W.n), grp='fine'))
            F.wbox(B, BRZ, W, sc - 0.3, sc + 0.3, 0.75, 1.55, -0.6, -0.56, 'fine')
        # Reliefplatte (Feston)
        Hm = ctx.garland(2.6, 0.7)
        F.wbox(B, SAND, W, sc - 1.35, sc + 1.35, 4.55, 5.3, -0.01, 0.04, 'det')
        T, on = L.relief_tris(Hm, W, sc - 1.3, 4.58, 2.6, 0.68, 0.05, 0.04)
        B.add(SAND, T, 'det')
        # Fenster OG
        holes.append(F.place_window(B, W, sc, F.WSpec(1.3, 6.3, 8.7, 'rect', bars=(2, 3), frame=(0.2, 0.06), hood=True), 'det', ctx))
        holes.append(F.place_window(B, W, sc, F.WSpec(1.3, 9.9, 11.8, 'rect', bars=(2, 3), frame=(0.18, 0.05)), 'det', ctx))
        F.band_wall(B, W, 0, W.L, HP, 13.2, QUAD, holes, 'main')
        # Sockel unter dem Seitenfeld (Platz bis Podest)
        F.band_wall(B, W, 0, W.L, HG - 0.4, HP, RUST, [], 'main')
        F.wall_hsweep(B, SAND, W, 0, W.L, HP - 0.05, [(0.0, 0.0), (0.1, 0.0), (0.1, 0.15), (0.0, 0.2)], 'det')
        # Eckpilaster aussen
        F.pilaster_strip(B, SAND, W, s_of(sg * (SIDE_U - 0.75)) if sg > 0 else s_of(sg * SIDE_U), s_of(sg * SIDE_U) if sg > 0 else s_of(sg * (SIDE_U - 0.75)), HP, 13.2, 0.1)
        # Seitenwand aussen (u = +-SIDE_U), bis VR
        Ws = L.Wall((u_out, VB), (u_out, VR)) if sg > 0 else L.Wall((u_out, VR), (u_out, VB))
        F.band_wall(B, Ws, 0, Ws.L, HG - 0.4, 13.2, QUAD, [], 'main')
        # Gesims + Dach (flach mit Attika-Abdeckung)
        ring = np.array([[min(u_in, u_out), VB], [max(u_in, u_out), VB], [max(u_in, u_out), VR], [min(u_in, u_out), VR]])
        ring = np.array([[max(u_in, u_out), VR], [max(u_in, u_out), VB], [min(u_in, u_out), VB], [min(u_in, u_out), VR]])
        if sg > 0: F.hsweep(B, SAND, ring[:3], 13.2, P.prof_cornice(0.45, 0.8), closed=False, grp='det')
        else: F.hsweep(B, SAND, ring[1:], 13.2, P.prof_cornice(0.45, 0.8), closed=False, grp='det')
        B.add('LS_Flachdach', L.tri_poly(np.c_[ring, np.full(4, 13.95)], [0, 0, 1]), 'main')
    # Verbindungsbau Nord (u -17.6 .. -20.55, Front v = -7.8)
    W = L.Wall((-SIDE_U, -7.8), (-20.55, -7.8))
    holes = []
    sc = W.L / 2
    holes.append(F.place_window(B, W, sc, F.WSpec(1.2, 0.4, 2.4, 'rect', bars=(2, 2), frame=(0.15, 0.05)), 'det', ctx))
    holes.append(F.place_window(B, W, sc, F.WSpec(1.25, 5.0, 7.7, 'rect', bars=(2, 3), frame=(0.18, 0.06), hood=True), 'det', ctx))
    holes.append(F.place_window(B, W, sc, F.WSpec(1.2, 9.7, 12.0, 'rect', bars=(2, 3), frame=(0.18, 0.06), pediment=True), 'det', ctx))
    F.band_wall(B, W, 0, W.L, -2.3, 13.2, QUAD, holes, 'main')
    Wsd = L.Wall((-SIDE_U, VB), (-SIDE_U, -7.8))
    F.band_wall(B, Wsd, 0, Wsd.L, -2.3, 13.2, QUAD, [], 'main')
    F.hsweep(B, SAND, np.array([[-SIDE_U, -7.8], [-20.55, -7.8]]), 13.2, P.prof_cornice(0.45, 0.8), closed=False, grp='det')
    B.add('LS_Flachdach', L.tri_poly(np.array([[-20.55, VR, 13.95], [-SIDE_U, VR, 13.95], [-SIDE_U, -7.8, 13.95], [-20.55, -7.8, 13.95]]), [0, 0, 1]), 'main')

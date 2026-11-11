"""Leineschloss -- Laves-Bau an der Leinstrasse (noerdlich des Portikus): Sandsteinquader, rustiziertes Erdgeschoss,
Beletage mit Verdachungen auf Konsolen, 2. OG mit Dreiecksgiebeln, Konsolgesims, zurueckgesetzte Attika,
Eckrisalit mit offener Saeulenhalle (toskanisch) an der Nordecke, Hofseite."""
import numpy as np
import lslib as L
import ls_parts as P
import ls_fac as F
from lslib import Builder, UP
from ls_fac import SAND, QUAD, RUST, GRAN, ZINK, FLD, EIS, PUTZ

U_E, U_W = -20.55, -54.85          # Ost-/Westkante
V_N, V_S = -2.1, -22.3             # Front Leinstrasse / Hofseite
RIS = 0.35                         # Risalit-Vorsprung
U_RIS = -45.85; V_RIS = -11.1      # Risalit-Grenzen (Nordecke)
HB = -2.6; H_PL = -0.9; H_GF = 4.1; H_1 = 8.9; H_2 = 12.9; H_CT = 13.85; H_AT = 16.1


def axis_set(B, W, sc, ctx, kinds=('gf', '1', '2'), big=False):
    holes = []
    if 'gf' in kinds:
        holes.append(F.place_window(B, W, sc, F.WSpec(1.2, 0.3, 2.3, bars=(2, 3), frame=(0.15, 0.05)), 'det', ctx))
    if '1' in kinds:
        w = 1.7 if big else 1.3
        holes.append(F.place_window(B, W, sc, F.WSpec(w, 5.0, 8.0 if big else 7.85, bars=(2, 4), frame=(0.2, 0.07), hood=not big, pediment=big, ears=True), 'det', ctx))
    if '2' in kinds:
        holes.append(F.place_window(B, W, sc, F.WSpec(1.25, 9.7, 12.05, bars=(2, 3), frame=(0.18, 0.06), pediment=True), 'det', ctx))
    return holes


def facade(B, W, holes, ctx, s0=0.0, s1=None, gf=True, mat_up=QUAD):
    s1 = W.L if s1 is None else s1
    if gf:
        F.band_wall(B, W, s0, s1, HB, H_PL, GRAN, holes, 'main')
        F.band_wall(B, W, s0, s1, H_PL, H_GF, RUST, holes, 'main')
    F.band_wall(B, W, s0, s1, H_GF, H_1, mat_up, holes, 'main')
    F.band_wall(B, W, s0, s1, H_1, H_2, mat_up, holes, 'main')


def build(B, ctx):
    n0 = B.tri_count()[0]
    # ---------------- Nordostfront (Leinstrasse), Hauptteil
    W = L.Wall((U_E, V_N), (U_RIS, V_N))
    holes = []
    sp = (U_E - U_RIS) / 6
    for k in range(6):
        holes += axis_set(B, W, sp * (k + 0.5), ctx)
    facade(B, W, holes, ctx)
    # Risalit-Rueckspruenge (0.35 m) an der Ostseite des Risalits
    Wr = L.Wall((U_RIS, V_N), (U_RIS, V_N + RIS))
    facade(B, Wr, [], ctx)
    # ---------------- Eckrisalit: Front NO (v = V_N + RIS) und NW (u = U_W - RIS)
    vf = V_N + RIS; uf = U_W - RIS
    Wf = L.Wall((U_RIS, vf), (uf, vf)); Wn = L.Wall((uf, vf), (uf, V_RIS))
    for Wx in (Wf, Wn):
        sc = Wx.L / 2
        holes = axis_set(B, Wx, sc, ctx, kinds=('1', '2'), big=True)
        facade(B, Wx, holes, ctx, gf=False)
        # Pilaster um das grosse Fenster + Eckquader
        for sg in (-1, 1):
            F.pilaster_strip(B, SAND, Wx, sc + sg * 1.45 - 0.25, sc + sg * 1.45 + 0.25, H_GF + 0.35, H_1 - 0.15, 0.1)
            F.wbox(B, SAND, Wx, sc + sg * 1.45 - 0.33, sc + sg * 1.45 + 0.33, H_1 - 0.32, H_1 - 0.05, -0.01, 0.16, 'det')
        F.quoins(B, SAND, Wx, 0.0, H_GF, H_2, 1, 0.5, 0.75, 0.45, 0.05)
        F.quoins(B, SAND, Wx, Wx.L, H_GF, H_2, -1, 0.5, 0.75, 0.45, 0.05)
    # Rueckwand der NW-Risalitflanke (v = V_RIS) bis zur Hauptflucht
    Wx = L.Wall((uf, V_RIS), (U_W, V_RIS))
    facade(B, Wx, [], ctx)
    # ---------------- offene Saeulenhalle im Erdgeschoss des Eckrisalits
    porch(B, ctx, vf, uf)
    # ---------------- Nordwestfront (Parkplatzseite), Hauptteil
    W = L.Wall((U_W, V_RIS), (U_W, V_S))
    holes = []
    for k in range(2):
        holes += axis_set(B, W, W.L * (k + 0.5) / 2, ctx)
    facade(B, W, holes, ctx)
    # ---------------- Suedostflanke (zum Verbindungsbau) sichtbarer Teil
    W = L.Wall((U_E, -7.8), (U_E, V_N))
    holes = axis_set(B, W, W.L / 2, ctx)
    facade(B, W, holes, ctx)
    # ---------------- Hofseite (v = V_S), sichtbarer Teil u -40.05 .. U_E
    W = L.Wall((-40.05, V_S), (U_E, V_S))
    holes = []
    for k in range(4):
        holes += axis_set(B, W, W.L * (k + 0.5) / 4, ctx)
    F.band_wall(B, W, 0, W.L, -2.8, H_GF, QUAD, holes, 'main')
    F.band_wall(B, W, 0, W.L, H_GF, H_2, QUAD, holes, 'main')
    # ---------------- Gurtgesimse + Kranzgesims (umlaufend)
    outline = np.array([[U_E, V_S], [U_E, V_N], [U_RIS, V_N], [U_RIS, vf], [uf, vf], [uf, V_RIS], [U_W, V_RIS], [U_W, V_S]])
    F.hsweep(B, SAND, outline, H_PL - 0.15, P.prof_plinth(0.1, 0.3), closed=True, grp='det')
    F.hsweep(B, SAND, outline, H_GF, P.prof_string(0.16, 0.36), closed=True, grp='det')
    F.hsweep(B, SAND, outline, H_1, P.prof_string(0.1, 0.25), closed=True, grp='det')
    F.main_cornice(B, outline, H_2, H_CT - H_2, 0.85, closed=True, modillions=0.62, dentils=True, mat=SAND, grp='det',
                   skip=lambda k, a, c: abs(a[1] - V_S) < 0.01 and abs(c[1] - V_S) < 0.01)
    # ---------------- Attika (zurueckgesetzt)
    at = np.array([[U_E - 1.2, V_S + 1.2], [U_E - 1.2, V_N - 1.2], [U_W + 1.2, V_N - 1.2], [U_W + 1.2, V_S + 1.2]])
    B.add(QUAD, L.prism_tris(at, H_CT - 0.05, H_AT, top=False), 'main')
    F.hsweep(B, SAND, at, H_AT - 0.35, [(0.0, 0.0), (0.06, 0.05), (0.14, 0.2), (0.16, 0.35), (0.0, 0.35)], closed=True, grp='det')
    F.hsweep(B, SAND, at, H_CT - 0.05, [(0.0, 0.0), (0.12, 0.0), (0.12, 0.3), (0.0, 0.4)], closed=True, grp='det')
    B.add(FLD, L.tri_poly(np.c_[at, np.full(4, H_AT)], [0, 0, 1]), 'main')
    # Umgang zwischen Gesims und Attika (Zinkabdeckung)
    ring_out = outline
    T = L.tri_poly_holes(np.c_[ring_out, np.full(len(ring_out), H_CT - 0.02)], [np.c_[at[::-1], np.full(4, H_CT - 0.02)]], [0, 0, 1])
    B.add(ZINK, T, 'main')
    # Haustechnik auf der Attika (Luftbild: Nordwestende)
    for (a0, a1, c0, c1, hh) in ((-53.2, -48.6, -4.2, -8.0, 1.6), (-53.2, -48.6, -9.0, -12.8, 1.6), (-47.8, -45.6, -4.2, -7.2, 1.1)):
        B.add('LS_Zink', L.box_tris(a0, a1, c1, c0, H_AT, H_AT + hh, skip=('b',)), 'main')
        for k in range(2):
            uc = a0 + (a1 - a0) * (k + 0.5) / 2; vc = (c0 + c1) / 2
            B.add(EIS, L.lathe_tris([(0.7, H_AT + hh), (0.7, H_AT + hh + 0.12)], 16, (uc, vc, 0), cap_top=True), 'det')
    ctx.log('laves', B.tri_count()[0] - n0)


def porch(B, ctx, vf, uf):
    """Offene Halle an der Nordecke: Innenwaende, Decke, toskanische Saeulen, Gebaelk, Gitter."""
    d = 2.8; hf = -1.55
    ui = uf + d; vi = vf - d
    # Innenwaende (Quader) mit Tuer
    Wa = L.Wall((U_RIS, vi), (ui, vi))           # Normale +v
    hole = F.place_window(B, Wa, Wa.L / 2, F.WSpec(1.6, hf + 0.15, 3.0, door=True, depth=0.3), 'main', ctx)
    F.band_wall(B, Wa, 0, Wa.L, hf - 0.6, H_GF, RUST, [hole], 'main')
    Wb = L.Wall((ui, vi), (ui, V_RIS))           # Normale -u
    F.band_wall(B, Wb, 0, Wb.L, hf - 0.6, H_GF, RUST, [], 'main')
    We = L.Wall((U_RIS, vf), (U_RIS, vi))        # Stirnwand SO (Normale -u)
    F.band_wall(B, We, 0, We.L, hf - 0.6, H_GF, RUST, [], 'main')
    Ws = L.Wall((uf, V_RIS), (ui, V_RIS))        # Stirnwand SW (Normale +v)
    F.band_wall(B, Ws, 0, Ws.L, hf - 0.6, H_GF, RUST, [], 'main')
    # Boden + Decke (L-Form)
    Lp = np.array([[U_RIS, vf], [uf, vf], [uf, V_RIS], [ui, V_RIS], [ui, vi], [U_RIS, vi]])
    B.add(GRAN, L.tri_poly(np.c_[Lp, np.full(len(Lp), hf)], [0, 0, 1]), 'main')
    B.add(GRAN, L.prism_tris(Lp, hf - 0.6, hf, top=False), 'main')
    B.add('LS_Putz_weiss', L.tri_poly(np.c_[Lp, np.full(len(Lp), H_GF - 0.55)], [0, 0, -1]), 'det')
    # Gebaelk ueber den Saeulen (aussen buendig mit Risalit)
    B.add(SAND, L.box_tris(uf, U_RIS, vf - 0.9, vf, H_GF - 0.55, H_GF + 0.02, skip=('t',)), 'main')
    B.add(SAND, L.box_tris(uf, uf + 0.9, V_RIS, vf - 0.9, H_GF - 0.55, H_GF + 0.02, skip=('t',)), 'main')
    # Saeulen
    Dc = 0.66; Hc = H_GF - 0.55 - hf
    pos = [(uf + 0.45, vf - 0.45), (-51.6, vf - 0.45), (-48.6, vf - 0.45), (uf + 0.45, -5.3), (uf + 0.45, -8.3)]
    for (u, v) in pos:
        P.tuscan_column(B, Dc, Hc, SAND, 24, hf, u, v)
    # Pfeiler an den Enden
    B.add(SAND, L.bevel_box_tris(U_RIS - 0.7, U_RIS, vf - 0.7, vf, hf, H_GF - 0.55, c=0.02), 'main', closed=True)
    B.add(SAND, L.bevel_box_tris(uf, uf + 0.7, V_RIS, V_RIS + 0.7, hf, H_GF - 0.55, c=0.02), 'main', closed=True)
    # schmiedeeiserne Gitter zwischen den Saeulen der NW-Seite
    for (v0, v1) in ((vf - 0.85, -4.95), (-5.65, -7.95), (-8.65, V_RIS + 0.7)):
        F.railing(B, (uf + 0.45, v0), (uf + 0.45, v1), hf, 2.6, 1.2, EIS, bar=0.016)

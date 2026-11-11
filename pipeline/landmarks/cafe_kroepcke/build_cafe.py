"""Cafe Kroepcke / Moevenpick-Pavillon (Georgstrasse 35, Hannover; 1976, Matthaei et al.) -- Detail-Landmarke.

Quellen der Masse
  * Grundriss + Tonnenachsen: LGLN LoD2 (Gebaeude DENILD43...9TB7, 13 Teile), siehe lgln_parts.json / analyse_parts.py
  * Hoehen: LGLN-DOM1 (2016) -- Firste der niedrigen Tonnen (P1, P4) 7.45, der hohen Tonnen (P2, P3, P6, D3, P5) 8.85,
    kleine Tonnen 5.8, Trommel 13.1 (LoD2 unterschaetzt die Tonnen: 6.35 / 4.52). Kaempfer = First - Halbbreite.
  * Fassaden/Details: Commons-Fotos 2016-2022 (Panorama, Lockdown 2021, Rueckseite Opernplatz, Seite, Flaggen 2022).
Frames: Spiel (x Ost, y oben, z Sued); Blender X=x, Y=-z, Z=y. Export GLB Y-up.
Aufruf: blender -b --factory-startup -P build_cafe.py -- <out.glb> <out.blend>
"""
import bpy, bmesh, math, sys, os, random, json
from mathutils import Vector, Matrix, noise
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib, cafe_lib
importlib.reload(cafe_lib)
from cafe_lib import *

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT_GLB = argv[0] if argv else os.path.join(HERE, 'cafe_kroepcke.glb')
OUT_BLEND = argv[1] if len(argv) > 1 else None
DETAIL = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/detail_raw/"
FONT_CAFE = "C:/Windows/Fonts/COOPBL.TTF"
FONT_MP = "C:/Windows/Fonts/georgiab.ttf"
random.seed(1976)
TEXTS = []

bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
COLL = bpy.data.collections.new('Cafe_Kroepcke'); SC.collection.children.link(COLL)
INST = bpy.data.collections.new('Cafe_Kroepcke_Instanzen'); SC.collection.children.link(INST)

# ======================================================================= Gelaende (Pflaster des Detailmodells)
_bvh = []
def _load_ground():
    tmp = bpy.data.collections.new('tmp_ground'); SC.collection.children.link(tmp)
    for n in ('street', 'ground', 'fix'):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=DETAIL + n + '.glb')
        for o in set(bpy.data.objects) - before:
            if o.type == 'MESH':
                mw = o.matrix_world
                vs = [mw @ v.co for v in o.data.vertices]
                _bvh.append(BVHTree.FromPolygons(vs, [list(p.vertices) for p in o.data.polygons]))
    for o in list(bpy.data.objects):
        if o.name not in COLL.objects and o.name not in INST.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    for coll_ in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for m in list(coll_):
            if m.users == 0: coll_.remove(m)
    bpy.data.collections.remove(tmp)
_load_ground()


def ground_y(p, default=-0.29):
    """Pflasterhoehe (Spiel-y) am Planpunkt p (x, z)"""
    best = None
    for b in _bvh:
        hit = b.ray_cast(Vector((p[0], -p[1], 40.0)), Vector((0, 0, -1)))
        if hit[0] is not None and (best is None or hit[0].z > best): best = hit[0].z
    return best if best is not None else default


# ======================================================================= Materialien
M = {}
M['kupfer'] = make_mat('Cafe_Kupfer_Patina', (110, 150, 134), metal=0.25, rough=0.36, vcol=True, coat=0.3)
M['falz'] = make_mat('Cafe_Kupfer_Falz', (78, 120, 108), metal=0.45, rough=0.38)
M['rippe'] = make_mat('Cafe_Stahl_Deckleiste', (40, 52, 50), metal=0.55, rough=0.42)
M['decke'] = make_mat('Cafe_Innen_Decke', (132, 106, 80), rough=0.75, emit=(255, 210, 150), emit_s=0.10)
M['stahl'] = make_mat('Cafe_Stahl_Dunkel', (36, 41, 41), metal=0.6, rough=0.42)
M['paneel'] = make_mat('Cafe_Fassade_Paneel', (24, 27, 28), metal=0.3, rough=0.5)
M['glas'] = make_mat('Cafe_Glas', (40, 52, 51), metal=0.1, rough=0.03, alpha=0.38, double=True)
M['satin'] = make_mat('Cafe_Satinierung', (160, 168, 166), rough=0.45, emit=(255, 236, 210), emit_s=0.03)
M['rot'] = make_mat('Cafe_Portal_Rot', (196, 30, 32), metal=0.15, rough=0.38)
M['edel'] = make_mat('Cafe_Edelstahl', (205, 207, 208), metal=1.0, rough=0.22)
M['granit'] = make_mat('Cafe_Granit', (104, 102, 99), rough=0.72)
M['dach'] = make_mat('Cafe_Dach_Bitumen', (66, 66, 68), rough=0.9)
M['trommel'] = make_mat('Cafe_Trommel_Lamellen', (44, 58, 52), metal=0.55, rough=0.4)
M['trommel_kern'] = make_mat('Cafe_Trommel_Kern', (26, 30, 29), metal=0.3, rough=0.6)
M['trommel_band'] = make_mat('Cafe_Trommel_Abschluss', (128, 132, 128), metal=0.7, rough=0.35)
M['globus'] = make_mat('Cafe_Globusleuchte', (248, 244, 235), rough=0.25, emit=(255, 236, 205), emit_s=2.5)
M['schrift_w'] = make_mat('Cafe_Schrift_Weiss', (250, 248, 240), rough=0.3, emit=(255, 250, 240), emit_s=1.6)
M['schrift_r'] = make_mat('Cafe_Schrift_Rot', (215, 28, 36), rough=0.35, emit=(255, 40, 40), emit_s=1.4)
M['markise'] = make_mat('Cafe_Markise_Stoff', (238, 236, 230), rough=0.9, double=True)
M['holz'] = make_mat('Cafe_Holz', (132, 86, 52), rough=0.62)
M['stuhl'] = make_mat('Cafe_Stuhl_Rot', (178, 42, 38), rough=0.5)
M['innen_boden'] = make_mat('Cafe_Innen_Boden', (112, 78, 52), rough=0.6)
M['innen_dunkel'] = make_mat('Cafe_Innen_Moebel', (58, 42, 34), rough=0.6)
M['innen_wand'] = make_mat('Cafe_Innen_Wand', (112, 80, 56), rough=0.7, emit=(255, 190, 130), emit_s=0.10)
M['licht'] = make_mat('Cafe_Innen_Licht', (255, 230, 190), rough=0.4, emit=(255, 214, 160), emit_s=6.0)
M['pflanze'] = make_mat('Cafe_Pflanze', (58, 92, 44), rough=0.8)
M['led'] = make_mat('Cafe_LED_Stufe', (255, 240, 220), emit=(255, 236, 200), emit_s=4.0)
M['col'] = make_mat('Cafe_Kollision', (128, 128, 128))

B = {}
def BL(key, mat, vcol=False, smooth=35.0):
    if key not in B: B[key] = Builder('Cafe_Kroepcke_' + key, M[mat], smooth_angle=smooth, vcol=vcol)
    return B[key]

# Exportgruppen (Builder-Schluessel) -> Namenszusaetze fuer die Spiel-Engine (nocol / noshadow)
FLAGS = {}

# ======================================================================= Grundriss (LGLN LoD2, Spielframe x/z)
RAW = {
    'N1': (23.70, 4.53), 'N2': (29.96, 6.70), 'N3': (36.47, 8.96), 'N4': (35.25, 12.31), 'N5': (42.54, 14.85),
    'N6': (48.61, 16.97), 'N7': (47.32, 20.58), 'N8': (56.14, 36.13), 'N9': (50.20, 39.40), 'N10': (43.88, 42.88),
    'N11': (38.26, 45.99), 'N12': (33.14, 37.08), 'N13': (26.80, 40.56), 'N15': (17.25, 23.03),
    'J1': (24.71, 22.39), 'J2': (31.80, 21.82), 'J3': (40.34, 21.14),
    'S0': (19.23, 6.92), 'S1': (18.01, 10.33), 'S2': (16.82, 13.64), 'S3': (15.63, 16.95), 'S4': (14.45, 20.26),
    'W1': (22.34, 8.06), 'W2': (21.16, 11.51), 'W3': (20.03, 14.80), 'W4': (18.91, 18.06), 'W5': (17.80, 21.31),
}
# Anschlusspunkte der kleinen Tonnen auf die (gerade) Westtraufe von P1 projizieren
_d = norm2(sub2(RAW['N15'], RAW['N1']))
for k in ('W1', 'W2', 'W3', 'W4', 'W5'):
    t = dot2(sub2(RAW[k], RAW['N1']), _d); RAW[k] = add2(RAW['N1'], _d, t)
P = RAW

FLOOR = 0.0                              # Fussboden / OK Sockel (Pflaster ~ -0.29)
LOW_SP, LOW_RIDGE = 4.03, 7.45
HIGH_SP, HIGH_RIDGE = 5.40, 8.85
SM_SP, SM_B = 4.03, 1.70
INS, TH = 0.18, 0.12                     # Traufeinzug der Tonne, Schalendicke
SM_INS, SM_TH = 0.12, 0.10
FASCIA0 = 2.95                           # UK Blende / OK Verglasung

VDEF = {
    'P1': dict(L=('N1', 'N15'), R=('N2', 'J1'), A=('gable', 'N1', 'N2'), B=('mitre', 'N15', 'J1'), lvl='low'),
    'P4': dict(L=('N15', 'N13'), R=('J1', 'N12'), A=('mitre', 'N15', 'J1'), B=('gable', 'N13', 'N12'), lvl='low'),
    'P2': dict(L=('N2', 'J1'), R=('N3', 'J2'), A=('gable', 'N2', 'N3'), B=('mitre', 'J1', 'J2'), lvl='high'),
    'P6': dict(L=('J1', 'N11'), R=('J2', 'N10'), A=('mitre', 'J1', 'J2'), B=('gable', 'N11', 'N10'), lvl='high'),
    'D3': dict(L=('J2', 'N10'), R=('J3', 'N9'), A=('cap', 'J2', 'J3'), B=('gable', 'N10', 'N9'), lvl='high'),
    'P3': dict(L=('N5', 'J3'), R=('N6', 'N7'), A=('cap', 'N5', 'N6'), B=('mitre', 'J3', 'N7'), lvl='high'),   # Nordgiebel = Brandwand zum Nachbarn
    'P5': dict(L=('J3', 'N9'), R=('N7', 'N8'), A=('mitre', 'J3', 'N7'), B=('gable', 'N9', 'N8'), lvl='high'),
}
SMALL = [('S8', 'S4', 'S3', 'W5', 'W4'), ('S9', 'S3', 'S2', 'W4', 'W3'), ('S10', 'S2', 'S1', 'W3', 'W2'), ('S11', 'S1', 'S0', 'W2', 'W1')]


class Vault:
    pass


def make_vdef(name, Lraw, Rraw, ends, sp, b, ins, th):
    """Linien (aussen/innen) + Endpunkte einer Tonne bestimmen."""
    V = Vault(); V.name = name; V.sp = sp; V.b = b; V.bi = b - th; V.th = th
    Lp, Lq = Lraw; Rp, Rq = Rraw
    mL = lerp2(Lp, Lq, 0.5); mR = lerp2(Rp, Rq, 0.5)
    V.Lo = offset_line(Lp, Lq, mR, ins); V.Ro = offset_line(Rp, Rq, mL, ins)
    V.Li = offset_line(Lp, Lq, mR, ins + th); V.Ri = offset_line(Rp, Rq, mL, ins + th)
    V.ends = {}
    for key, (typ, a, c) in ends.items():
        e = {'type': typ, 'raw': (a, c)}
        if typ in ('gable', 'mitre', 'cap'):
            pa, pc = P[a], P[c]; ed = sub2(pc, pa)
            for nm, ln in (('Lo', V.Lo), ('Ro', V.Ro), ('Li', V.Li), ('Ri', V.Ri)):
                e[nm] = line_isect(ln[0], sub2(ln[1], ln[0]), pa, ed)
        V.ends[key] = e
    return V


VAULTS = {}
for nm, d in VDEF.items():
    sp, rid = (LOW_SP, LOW_RIDGE) if d['lvl'] == 'low' else (HIGH_SP, HIGH_RIDGE)
    VAULTS[nm] = make_vdef(nm, (P[d['L'][0]], P[d['L'][1]]), (P[d['R'][0]], P[d['R'][1]]), {'A': d['A'], 'B': d['B']}, sp, rid - sp, INS, TH)
# Gehrungen: gemeinsame Eckpunkte mitteln
for a, b in (('P1', 'P4'), ('P2', 'P6'), ('P3', 'P5')):
    ea = VAULTS[a].ends['B']; eb = VAULTS[b].ends['A']
    for k in ('Lo', 'Ro', 'Li', 'Ri'):
        m = lerp2(ea[k], eb[k], 0.5); ea[k] = m; eb[k] = m


def vault_frame(V):
    A, Bn = V.ends['A'], V.ends['B']
    O = lerp2(A['Lo'], A['Ro'], 0.5)
    u = norm2(sub2(lerp2(Bn['Lo'], Bn['Ro'], 0.5), O))
    v = (-u[1], u[0])
    if dot2(sub2(A['Lo'], O), v) < 0: v = (-v[0], -v[1])
    V.O, V.u, V.v = O, u, v
    def sw(p): return (dot2(sub2(p, O), u), dot2(sub2(p, O), v))
    V.sw = sw
    def lin(p, q):
        (s0, w0), (s1, w1) = sw(p), sw(q)
        k = (w1 - w0) / (s1 - s0)
        return lambda s: w0 + (s - s0) * k
    V.wLo = lin(A['Lo'], Bn['Lo']); V.wRo = lin(A['Ro'], Bn['Ro'])
    V.wLi = lin(A['Li'], Bn['Li']); V.wRi = lin(A['Ri'], Bn['Ri'])
    V.pos = lambda s, w: (O[0] + u[0] * s + v[0] * w, O[1] + u[1] * s + v[1] * w)


for V in VAULTS.values(): vault_frame(V)


def host_height(V, p):
    """Hoehe der Aussenflaeche der Tonne V ueber Planpunkt p (None ausserhalb)"""
    s, w = V.sw(p)
    wl, wr = V.wLo(s), V.wRo(s)
    c, a = (wl + wr) / 2, (wl - wr) / 2
    x = (w - c) / a
    if abs(x) >= 1: return None
    return V.sp + V.b * math.sqrt(1 - x * x)


# kleine Tonnen (Westseite, quer zu P1): Giebel aussen, innen in P1 eingeschnitten
for nm, g0, g1, w0, w1 in SMALL:
    pa, pb = P[g0], P[g1]; qa, qb = P[w0], P[w1]
    u = norm2(sub2(lerp2(qa, qb, 0.5), lerp2(pa, pb, 0.5)))
    ext_a = add2(qa, u, 2.6); ext_b = add2(qb, u, 2.6)
    V = make_vdef(nm, (pa, ext_a), (pb, ext_b), {'A': ('gable', g0, g1), 'B': ('host', None, None)}, SM_SP, SM_B, SM_INS, SM_TH)
    # Endpunkte B (nur fuer Rahmen): Verlaengerung
    e = V.ends['B']
    for k, ln in (('Lo', V.Lo), ('Ro', V.Ro), ('Li', V.Li), ('Ri', V.Ri)):
        e[k] = ln[1]
    e['host'] = 'P1'
    VAULTS[nm] = V
    vault_frame(V)

# ======================================================================= Tonnen bauen
NTH_BIG, NTH_SMALL = 48, 32


def patina(p):
    """Vertexfarbe der Kupferpatina (linear) aus Weltposition"""
    x, y, z = p
    n1 = noise.fractal(Vector((x * 0.22, y * 0.22, z * 0.5)), 0.6, 2.0, 4)
    n2 = noise.noise(Vector((x * 1.3, y * 1.3, z * 0.25)))
    base = Vector(srgb2lin((110, 150, 134)))
    dark = Vector(srgb2lin((78, 112, 100))); light = Vector(srgb2lin((150, 184, 166))); blue = Vector(srgb2lin((102, 142, 140)))
    t = max(-1, min(1, n1 * 1.3))
    c = base.lerp(light, t) if t > 0 else base.lerp(dark, -t)
    c = c.lerp(blue, 0.25 + 0.25 * n2)
    return (c.x, c.y, c.z, 1.0)


def build_vault(V, nth, row_ds, seam_ds, rib_every, long_th, groin=False):
    Bo = BL('Kupfer', 'kupfer', vcol=True); Bi = BL('Decke_innen', 'decke'); Bs = BL('Kupfer_Falz', 'falz'); Br = BL('Rippen', 'rippe')
    th = [math.pi * j / nth for j in range(nth + 1)]
    cs = [(1 - math.cos(t)) / 2 for t in th]
    sp, b, bi = V.sp, V.b, V.bi

    def col_w(outer, j, s):
        wl, wr = (V.wLo(s), V.wRo(s)) if outer else (V.wLi(s), V.wRi(s))
        return wl * (1 - cs[j]) + wr * cs[j]

    def col_y(outer, j): return sp + (b if outer else bi) * math.sin(th[j])

    def endverts(key, outer):
        e = V.ends[key]
        L, R = (e['Lo'], e['Ro']) if outer else (e['Li'], e['Ri'])
        out = []
        for j in range(nth + 1):
            if e['type'] == 'host':
                H = VAULTS[e['host']]; y = col_y(outer, j)
                s_lo, s_hi = V.sw(V.ends['A']['Lo'])[0] + 0.3, V.sw(e['Lo'])[0]
                def inside(s):
                    hh = host_height(H, V.pos(s, col_w(outer, j, s)))
                    return hh is not None and hh > y
                lo, hi = s_lo, s_hi
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if inside(mid): hi = mid
                    else: lo = mid
                s = hi + 0.05
                p = V.pos(s, col_w(outer, j, s))
                out.append((s, p, y))
            else:
                p = lerp2(L, R, cs[j]); out.append((V.sw(p)[0], p, col_y(outer, j)))
        return out

    EA = {o: endverts('A', o) for o in (True, False)}
    EB = {o: endverts('B', o) for o in (True, False)}
    smin = min(e[0] for e in EA[True]); smax = max(e[0] for e in EB[True])
    rows = [smin + row_ds * k for k in range(1, int((smax - smin) / row_ds) + 1)]
    rows_in = [smin + row_ds * 3 * k for k in range(1, int((smax - smin) / (row_ds * 3)) + 1)]

    def surface(Bx, outer, rws):
        start = len(Bx.f)
        cols = []
        for j in range(nth + 1):
            a = EA[outer][j]; c = EB[outer][j]
            lst = [(a[0], Bx.vert(W(a[1], a[2]), patina(W(a[1], a[2])) if Bx.vcol else None))]
            for s in rws:
                if a[0] + 0.12 < s < c[0] - 0.12:
                    p = V.pos(s, col_w(outer, j, s)); q = W(p, col_y(outer, j))
                    lst.append((s, Bx.vert(q, patina(q) if Bx.vcol else None)))
            lst.append((c[0], Bx.vert(W(c[1], c[2]), patina(W(c[1], c[2])) if Bx.vcol else None)))
            cols.append(lst)
        for j in range(nth):
            zipper(Bx, cols[j], cols[j + 1])
        mid = len(Bx.f) - 1 - (len(Bx.f) - start) // 2
        # Orientierung: Firstflaeche aussen nach oben, innen nach unten
        orient_faces(Bx, start, (0, 0, 1) if outer else (0, 0, -1), sample=start + (len(Bx.f) - start) // 2)
        return cols

    def zipper(Bx, a, c):
        i = k = 0
        while i < len(a) - 1 or k < len(c) - 1:
            if k == len(c) - 1 or (i < len(a) - 1 and a[i + 1][0] <= c[k + 1][0]):
                Bx.face([a[i][1], a[i + 1][1], c[k][1]]); i += 1
            else:
                Bx.face([a[i][1], c[k + 1][1], c[k][1]]); k += 1

    co = surface(Bo, True, rows)
    ci = surface(Bi, False, rows_in)
    # Traufstreifen + Endstreifen (Schale schliessen) -> in den Innen-Builder (Material Decke, verdeckt)
    for j, sgn in ((0, 1), (nth, -1)):
        a = [(s, Bi.vert(Bo.v[idx])) for s, idx in co[j]]
        c = [(s, Bi.vert(Bi.v[idx])) for s, idx in ci[j]]
        st = len(Bi.f); zipper(Bi, a, c); orient_faces(Bi, st, (0, 0, -1))
    for key in ('A', 'B'):
        e = V.ends[key]
        if e['type'] == 'mitre': continue
        st = len(Bi.f)
        oo = [Bi.vert(W(p, y)) for s, p, y in (EA if key == 'A' else EB)[True]]
        ii = [Bi.vert(W(p, y)) for s, p, y in (EA if key == 'A' else EB)[False]]
        for j in range(nth):
            Bi.face([oo[j], oo[j + 1], ii[j + 1], ii[j]])
        d = D3(V.u) * (-1 if key == 'A' else 1)
        orient_faces(Bi, st, d, sample=st + nth // 2)
        if e['type'] == 'cap':
            # Abschlusswand (separat, zwei Seiten, leicht ins Innere versetzt -> Schale bleibt mannigfaltig)
            Bp_ = BL('Paneel', 'paneel')
            for off, dd in ((0.006, d), (0.03, -d)):
                st = len(Bp_.f)
                Bp_.face([Bp_.vert(Vector(Bi.v[k]) - d * off) for k in ii]); orient_faces(Bp_, st, dd)
    V.EA, V.EB, V.th, V.cs = EA, EB, th, cs

    # ---- Stehfalze (Kupfer) + Deckleisten (Stahl) quer zur Achse
    k = 0; s = smin + seam_ds * 0.5
    while s < smax - 0.05:
        rib = (k % rib_every == rib_every // 2)
        Bx = Br if rib else Bs
        hw0, hw1, hh = (0.05, 0.03, 0.06) if rib else (0.013, 0.006, 0.03)
        if nth == NTH_SMALL and rib: hw0, hw1, hh = 0.04, 0.025, 0.05
        run = []
        for j in range(nth + 1):
            if EA[True][j][0] + 0.06 < s < EB[True][j][0] - 0.06: run.append(j)
            else:
                if len(run) > 1: seam(Bx, V, run, s, hw0, hw1, hh, col_w, col_y, th)
                run = []
        if len(run) > 1: seam(Bx, V, run, s, hw0, hw1, hh, col_w, col_y, th)
        s += seam_ds; k += 1
    # ---- Laengsstoesse
    for thl in long_th:
        j = int(round(thl / 180 * nth))
        a, c = EA[True][j][0] + 0.03, EB[True][j][0] - 0.03
        nn = normal_at(V, j, th, (a + c) / 2)
        p0 = W(V.pos(a, col_w(True, j, a)), col_y(True, j)); p1 = W(V.pos(c, col_w(True, j, c)), col_y(True, j))
        tg = (p1 - p0).normalized(); side = tg.cross(nn).normalized()
        st = [(p0 - nn * 0.004, side, nn), (p1 - nn * 0.004, side, nn)]
        sweep(Bs, st, [(-0.02, 0), (0.02, 0), (0.012, 0.022), (-0.012, 0.022)], caps=True)
    # ---- Gratleiste an der Gehrung
    if groin:
        e = EB[True]
        stn = []
        for j in range(nth + 1):
            s, p, y = e[j]; q = W(p, y)
            nn = normal_at(V, j, th, s)
            tj = (W(e[min(j + 1, nth)][1], e[min(j + 1, nth)][2]) - W(e[max(j - 1, 0)][1], e[max(j - 1, 0)][2])).normalized()
            side = tj.cross(nn).normalized()
            stn.append((q - nn * 0.01, side, nn))
        sweep(Br, stn, [(-0.07, 0), (0.07, 0), (0.045, 0.055), (-0.045, 0.055)])


def normal_at(V, j, th, s):
    """Aussennormale (Blender) der Tonne in Spalte j"""
    wl, wr = V.wLo(s), V.wRo(s); a = (wl - wr) / 2
    nw = V.b * math.cos(th[j]); ny = a * math.sin(th[j])
    vv = D3(V.v); n = vv * nw + UP * ny
    return n.normalized()


def seam(Bx, V, run, s, hw0, hw1, hh, col_w, col_y, th):
    uu = D3(V.u)
    stn = []
    for j in run:
        q = W(V.pos(s, col_w(True, j, s)), col_y(True, j))
        nn = normal_at(V, j, th, s)
        stn.append((q - nn * 0.006, uu, nn))
    sweep(Bx, stn, [(-hw0, 0), (hw0, 0), (hw1, hh), (-hw1, hh)], caps=True)


for nm in ('P1', 'P4', 'P2', 'P6', 'D3', 'P3', 'P5'):
    build_vault(VAULTS[nm], NTH_BIG, 1.0, 0.5, 3, (45, 90, 135), groin=nm in ('P1', 'P2', 'P3'))
for nm, *_ in SMALL:
    build_vault(VAULTS[nm], NTH_SMALL, 0.6, 0.42, 3, (90,))
FLAGS.update({'Kupfer_Falz': 'nocol', 'Rippen': 'nocol'})

# ======================================================================= Umriss, Normalen, Segmente
PERIM = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N15', 'W5', 'S4', 'S3', 'S2', 'S1', 'S0', 'W1']
POLY = [P[k] for k in PERIM]
NP = len(PERIM)
SEG = {}
SPEC = {
    ('N1', 'N2'): dict(kind='gable', vault='P1', doors=[('mid', 1.9, 'double', 'dark')]),
    ('N2', 'N3'): dict(kind='gable', vault='P2'),
    # Ost-/Nordostseite: Brandwaende zum 10.4 m hohen Nachbarbau des Detailmodells (Dach y=10.39)
    ('N3', 'N4'): dict(kind='party', top=HIGH_SP),
    ('N4', 'N5'): dict(kind='party', top=HIGH_SP),
    ('N5', 'N6'): dict(kind='party', top=HIGH_SP),
    ('N6', 'N7'): dict(kind='party', top=HIGH_SP),
    ('N7', 'N8'): dict(kind='party', top=HIGH_SP),
    ('N8', 'N9'): dict(kind='gable', vault='P5'),
    ('N9', 'N10'): dict(kind='gable', vault='D3', doors=[('mid', 1.9, 'double', 'red')]),
    ('N10', 'N11'): dict(kind='gable', vault='P6'),
    ('N11', 'N12'): dict(kind='eave', top=HIGH_SP),
    ('N12', 'N13'): dict(kind='gable', vault='P4', doors=[('mid', 1.9, 'double', 'dark')]),
    ('N13', 'N15'): dict(kind='eave', top=LOW_SP, portal=5.7),
    ('N15', 'W5'): dict(kind='eave', top=LOW_SP, doors=[('mid', 1.05, 'single', 'dark')]),
    ('W5', 'S4'): dict(kind='eave', top=SM_SP, small=True),
    ('S4', 'S3'): dict(kind='gable', vault='S8', small=True, doors=[('mid', 1.7, 'double', 'dark')]),
    ('S3', 'S2'): dict(kind='gable', vault='S9', small=True),
    ('S2', 'S1'): dict(kind='gable', vault='S10', small=True, doors=[('mid', 1.7, 'double', 'dark')]),
    ('S1', 'S0'): dict(kind='gable', vault='S11', small=True),
    ('S0', 'W1'): dict(kind='eave', top=SM_SP, small=True),
    ('W1', 'N1'): dict(kind='eave', top=LOW_SP),
}
for i in range(NP):
    a, c = PERIM[i], PERIM[(i + 1) % NP]
    sp = dict(SPEC[(a, c)]); A, C = P[a], P[c]
    t = norm2(sub2(C, A)); n = (-t[1], t[0])
    mid = lerp2(A, C, 0.5)
    if point_in_poly(add2(mid, n, 0.2), POLY): n = (-n[0], -n[1])
    sp.update(A=A, C=C, t=t, n=n, L=dist2(A, C), a=a, c=c, i=i)
    if sp['kind'] == 'gable':
        sp['top'] = VAULTS[sp['vault']].sp
    SEG[i] = sp


def offset_poly(pts, nrm, o, closed=True):
    """Gehrungsversatz einer Polylinie (Plan) um o entlang der Kantennormalen nrm[i] (Kante i: pts[i]->pts[i+1])"""
    n = len(pts); out = []
    for i in range(n):
        if not closed and i == 0: out.append(add2(pts[0], nrm[0], o)); continue
        if not closed and i == n - 1: out.append(add2(pts[-1], nrm[-1], o)); continue
        n0 = nrm[(i - 1) % len(nrm)]; n1 = nrm[i % len(nrm)]
        p0 = add2(pts[i], n0, o); p1 = add2(pts[i], n1, o)
        d0 = sub2(pts[i], pts[(i - 1) % n]); d1 = sub2(pts[(i + 1) % n], pts[i])
        q = line_isect(p0, d0, p1, d1)
        out.append(q if q is not None else p1)
    return out


def band(Bx, pts, nrm, profile, closed=True, caps=True):
    """Profil (o, y) entlang Polylinie mit Gehrung extrudieren"""
    offs = {}
    for o, y in profile:
        if o not in offs: offs[o] = offset_poly(pts, nrm, o, closed)
    stations = len(pts)
    rings = [[Bx.vert(W(offs[o][i], y)) for o, y in profile] for i in range(stations)]
    m = len(profile); st = len(Bx.f)
    seq = list(range(stations)) + ([0] if closed else [])
    for a, c in zip(seq[:-1], seq[1:]):
        for k in range(m):
            l = (k + 1) % m
            Bx.face([rings[a][k], rings[a][l], rings[c][l], rings[c][k]])
    if not closed and caps:
        Bx.face(rings[0][::-1]); Bx.face(rings[-1])
    return st


NRM = [SEG[i]['n'] for i in range(NP)]


def prof_rect(o0, o1, y0, y1, ch=0.0):
    if ch <= 0: return [(o0, y0), (o1, y0), (o1, y1), (o0, y1)]
    return [(o0, y0), (o1, y0), (o1, y1 - ch), (o1 - ch, y1), (o0, y1)]


def oriented(Bx, st, probe_pt, probe_dir):
    pass


def fix_band_orientation(Bx, st, pts, nrm, o_out):
    """Bandflaechen nach aussen orientieren: Flaeche, deren Mittelpunkt naeher am Aussenversatz liegt, zeigt nach aussen"""
    pass


# Sockel (Granit), Blende (Paneel) rundum
st = band(BL('Granit', 'granit'), POLY, NRM, [(-0.05, -0.9), (0.07, -0.9), (0.07, FLOOR - 0.02), (0.05, FLOOR), (-0.05, FLOOR)])
orient_faces(B['Granit'], st, D3(NRM[0]), sample=st + 1)
st = band(BL('Paneel', 'paneel'), POLY, NRM, [(-0.35, FASCIA0), (0.10, FASCIA0), (0.10, LOW_SP - 0.03), (0.07, LOW_SP), (-0.35, LOW_SP)])
orient_faces(B['Paneel'], st, D3(NRM[0]), sample=st + 1)
# Abschlussprofil unter der Blende (Stahl)
st = band(BL('Stahl', 'stahl'), POLY, NRM, [(-0.06, FASCIA0 - 0.10), (0.12, FASCIA0 - 0.10), (0.12, FASCIA0 + 0.02), (-0.06, FASCIA0 + 0.02)])
orient_faces(B['Stahl'], st, D3(NRM[0]), sample=st + 1)
# Fussprofil der Verglasung
st = band(B['Stahl'], POLY, NRM, [(-0.07, FLOOR), (0.06, FLOOR), (0.06, 0.12), (-0.07, 0.12)])
orient_faces(B['Stahl'], st, D3(NRM[0]), sample=st + 1)

# Boden innen
Bf = BL('Innen_Boden', 'innen_boden')
Bf.poly([W(p, FLOOR + 0.001) for p in offset_poly(POLY, NRM, -0.05)])
orient_faces(Bf, len(Bf.f) - 1, (0, 0, 1))
FLAGS['Innen_Boden'] = 'nocol_noshadow'


# ======================================================================= Fassadenelemente
def F(sg, s, o, y):
    p = add2(add2(sg['A'], sg['t'], s), sg['n'], o)
    return W(p, y)


def fbox(Bx, sg, s0, s1, o0, o1, y0, y1, ch=0.0):
    c = add2(add2(sg['A'], sg['t'], (s0 + s1) / 2), sg['n'], (o0 + o1) / 2)
    obox(Bx, W(c), D3(sg['t']), D3(sg['n']), abs(s1 - s0), abs(o1 - o0), y0, y1, ch)


def fquad(Bx, sg, s0, s1, o, y0, y1):
    pts = [F(sg, s0, o, y0), F(sg, s1, o, y0), F(sg, s1, o, y1), F(sg, s0, o, y1)]
    ids = [Bx.vert(p) for p in pts]
    n = newell(pts)
    Bx.face(ids if n.dot(D3(sg['n'])) >= 0 else ids[::-1])


def post(sg, s, y0, y1, big=False):
    Bs = B['Stahl']
    if big:
        fbox(Bs, sg, s - 0.11, s + 0.11, -0.16, 0.10, y0, y1, ch=0.02)
    else:
        fbox(Bs, sg, s - 0.035, s + 0.035, -0.01, 0.075, y0, y1, ch=0.012)      # Deckschale aussen
        fbox(Bs, sg, s - 0.025, s + 0.025, -0.15, -0.01, y0, y1)               # Steg innen


def glass_bay(sg, s0, s1, y_bot, y_top, frost=True, solid=False):
    Bg = BL('Glas', 'glas'); Bt = BL('Satinierung', 'satin'); Bp = B['Paneel']; Bs = B['Stahl']
    a, b = s0 + 0.03, s1 - 0.03
    if solid:
        fbox(Bp, sg, a, b, -0.06, 0.0, y_bot, y_top)
        # Fugen der Paneele
        h = y_bot + 1.0
        while h < y_top - 0.4:
            fbox(Bs, sg, a, b, -0.01, 0.015, h - 0.015, h + 0.015); h += 1.0
        return
    if frost and y_bot < 0.5:
        fquad(Bt, sg, a, b, 0.0, y_bot, 0.98)
        fbox(Bs, sg, s0, s1, -0.05, 0.06, 0.98, 1.04)                  # Kaempferprofil
        fquad(Bg, sg, a, b, 0.0, 1.04, y_top)
        bead(sg, a, b, y_bot, 0.98); bead(sg, a, b, 1.04, y_top)
    else:
        fquad(Bg, sg, a, b, 0.0, y_bot, y_top)
        bead(sg, a, b, y_bot, y_top)


def bead(sg, a, b, y0, y1, w=0.028):
    """Glashalteleisten (aussen, 3 cm) rund um eine Scheibe"""
    Bb = B['Stahl']
    fbox(Bb, sg, a, a + w, 0.0, 0.032, y0, y1); fbox(Bb, sg, b - w, b, 0.0, 0.032, y0, y1)
    fbox(Bb, sg, a + w, b - w, 0.0, 0.032, y0, y0 + w); fbox(Bb, sg, a + w, b - w, 0.0, 0.032, y1 - w, y1)


def door(sg, s0, s1, kind, frame, y_top=2.85, recess=0.0):
    """Tuer(en) zwischen s0 und s1: Fluegelrahmen, Glas, Stossgriffe, Trittschutzbleche, Oberlicht"""
    Bs = B['Stahl']; Be = BL('Edelstahl', 'edel'); Bg = BL('Glas', 'glas')
    Bfr = BL('Portal_Rot', 'rot') if frame == 'red' else Bs
    o = -recess
    h_door = 2.32
    fbox(Bs, sg, s0, s1, o - 0.06, o + 0.06, h_door, h_door + 0.07)               # Kaempfer
    fquad(Bg, sg, s0 + 0.03, s1 - 0.03, o, h_door + 0.07, y_top)                 # Oberlicht
    if kind == 'service':
        fbox(B['Paneel'], sg, s0 + 0.03, s1 - 0.03, o - 0.05, o + 0.01, FLOOR + 0.01, h_door)
        fbox(Bs, sg, s0 + 0.03, s1 - 0.03, o + 0.01, o + 0.03, 0.9, 1.0)
        cylinder(Be, F(sg, s1 - 0.12, o + 0.02, 1.05), F(sg, s1 - 0.12, o + 0.09, 1.05), 0.012, 8)
        box3(Be, F(sg, s1 - 0.12, o + 0.09, 1.05), F(sg, s1 - 0.27, o + 0.09, 1.05), 0.025, 0.02)
        return
    leaves = [(s0, s1)] if kind == 'single' else [(s0, (s0 + s1) / 2), ((s0 + s1) / 2, s1)]
    for li, (a, b) in enumerate(leaves):
        a += 0.005; b -= 0.005; w = 0.065
        fbox(Bfr, sg, a, a + w, o - 0.03, o + 0.03, FLOOR + 0.01, h_door - 0.005)
        fbox(Bfr, sg, b - w, b, o - 0.03, o + 0.03, FLOOR + 0.01, h_door - 0.005)
        fbox(Bfr, sg, a + w, b - w, o - 0.03, o + 0.03, h_door - 0.075, h_door - 0.005)
        fbox(Bfr, sg, a + w, b - w, o - 0.03, o + 0.03, FLOOR + 0.01, FLOOR + 0.12)
        fquad(Bg, sg, a + w, b - w, o, FLOOR + 0.12, h_door - 0.075)
        # Trittschutzblech (Edelstahl) aussen + innen
        for oo in (o + 0.032, o - 0.036):
            fbox(Be, sg, a + w, b - w, oo, oo + 0.004, FLOOR + 0.12, FLOOR + 0.36)
        # Stossgriff: senkrechte Stange auf zwei Stuetzen, nahe der Schliesskante
        sh = b - 0.16 if (kind == 'single' or li == 0) else a + 0.16
        for yy in (0.62, 1.68):
            cylinder(Be, F(sg, sh, o + 0.03, yy), F(sg, sh, o + 0.085, yy), 0.012, 8)
            cylinder(Be, F(sg, sh, o - 0.03, yy), F(sg, sh, o - 0.085, yy), 0.012, 8)
        cylinder(Be, F(sg, sh, o + 0.09, 0.52), F(sg, sh, o + 0.09, 1.78), 0.017, 10)
        cylinder(Be, F(sg, sh, o - 0.09, 0.52), F(sg, sh, o - 0.09, 1.78), 0.017, 10)
    # Schwelle
    fbox(Be, sg, s0, s1, o - 0.10, o + 0.10, FLOOR - 0.005, FLOOR + 0.012)


def layout(sg, maxbay=2.25):
    """Feldteilung eines Segments: Liste (s0, s1, typ, info) zwischen Eckpfeilern"""
    L = sg['L']; cuts = []
    items = []
    for d in sg.get('doors', []):
        sc = L / 2 if d[0] == 'mid' else d[0]
        items.append((sc - d[1] / 2, sc + d[1] / 2, 'door', d))
    if sg.get('portal'):
        sc = L - sg['portal']
        items.append((sc - 1.75, sc + 1.75, 'portal', None))
    items.sort()
    out = []; cur = 0.12
    for a, b, typ, d in items + [(L - 0.12, L - 0.12, 'end', None)]:
        gap = a - cur
        if gap > 0.15:
            n = max(1, int(math.ceil(gap / maxbay - 0.08)))
            for k in range(n):
                out.append((cur + gap * k / n, cur + gap * (k + 1) / n, 'glass', None))
        if typ != 'end': out.append((a, b, typ, d))
        cur = b
    return out


def facade(sg):
    small = sg.get('small', False)
    if sg['kind'] == 'party':
        fbox(B['Paneel'], sg, 0.0, sg['L'], -0.16, -0.05, FLOOR, sg['top'] + 0.02)
        sg['layout'] = []
        return
    top = sg['top']
    lay = layout(sg, 1.75 if small else 2.25)
    sg['layout'] = lay
    solid = sg.get('solid', False)
    for k, (s0, s1, typ, d) in enumerate(lay):
        # Pfosten an den Feldgrenzen (ausser an Eckpfeilern)
        if k > 0:
            post(sg, s0, 0.12, FASCIA0 - 0.10)
        if typ == 'glass':
            glass_bay(sg, s0, s1, 0.12, FASCIA0 - 0.10, frost=not small, solid=solid)
        elif typ == 'door':
            door(sg, s0, s1, d[2], d[3])
            if k == len(lay) - 1: post(sg, s1, 0.12, FASCIA0 - 0.10)
        elif typ == 'portal':
            portal(sg, s0, s1)
    # Hohe Zone (zwischen Blende und Kaempfer der hohen Tonnen)
    if top > LOW_SP + 0.5:
        y0, y1 = LOW_SP + 0.05, top - 0.08
        n = max(1, int(round((sg['L'] - 0.24) / 1.6)))
        ss = [0.12 + (sg['L'] - 0.24) * k / n for k in range(n + 1)]
        for k in range(n):
            if 0 < k: post(sg, ss[k], y0, y1)
            if sg['kind'] == 'flat' or solid:
                glass_bay(sg, ss[k], ss[k + 1], y0, y1, frost=False, solid=True)
            else:
                glass_bay(sg, ss[k], ss[k + 1], y0, y1, frost=False)
        fbox(B['Stahl'], sg, 0.0, sg['L'], -0.06, 0.08, LOW_SP, LOW_SP + 0.05)
        if sg['kind'] != 'gable':
            fbox(B['Paneel'], sg, 0.0, sg['L'], -0.30, 0.10, top - 0.08, top)
            if sg['kind'] == 'flat':   # Attika mit Abdeckblech
                fbox(B['Paneel'], sg, 0.0, sg['L'], -0.30, 0.10, top, top + 0.22)
                fbox(B['Stahl'], sg, -0.02, sg['L'] + 0.02, -0.32, 0.13, top + 0.22, top + 0.26)


def portal(sg, s0, s1):
    """Rotes Moevenpick-Portal (P4 West): breiter roter U-Rahmen, zurueckgesetzte Doppeltuer"""
    Br = BL('Portal_Rot', 'rot')
    fbox(Br, sg, s0, s0 + 0.48, -0.25, 0.22, FLOOR, FASCIA0 - 0.10, ch=0.03)
    fbox(Br, sg, s1 - 0.48, s1, -0.25, 0.22, FLOOR, FASCIA0 - 0.10, ch=0.03)
    fbox(Br, sg, s0, s1, -0.25, 0.22, 2.52, FASCIA0 - 0.10)
    # Laibung innen (dunkel) + Tuer zurueckgesetzt
    fbox(B['Stahl'], sg, s0 + 0.48, s1 - 0.48, -0.30, -0.24, 2.40, 2.52)
    fbox(BL('Innen_Licht', 'licht'), sg, s0 + 0.9, s1 - 0.9, -0.20, -0.05, 2.505, 2.515)        # Downlight im Windfang
    for x in (s0 + 0.24, s1 - 0.24):                                                         # Bodenstrahler vor den roten Laibungen
        cylinder(BL('LED', 'led'), F(sg, x, 0.42, FLOOR - 0.004), F(sg, x, 0.42, FLOOR + 0.006), 0.06, 12)
    a, b = s0 + 0.48, s1 - 0.48
    door(sg, a + 0.28, b - 0.28, 'double', 'dark', y_top=2.40, recess=0.24)
    for x0, x1 in ((a, a + 0.28), (b - 0.28, b)):
        fquad(B['Glas'], sg, x0 + 0.02, x1 - 0.02, -0.24, FLOOR + 0.1, 2.40)
        fbox(B['Stahl'], sg, x0, x0 + 0.05, -0.30, -0.18, FLOOR, 2.40); fbox(B['Stahl'], sg, x1 - 0.05, x1, -0.30, -0.18, FLOOR, 2.40)
    # Bodenmatte
    fbox(BL('Innen_Dunkel', 'innen_dunkel'), sg, a + 0.05, b - 0.05, -0.22, 0.20, FLOOR - 0.002, FLOOR + 0.004)
    sg['portal_s'] = ((s0 + s1) / 2)


for i in range(NP): facade(SEG[i])
FLAGS['Innen_Dunkel'] = 'nocol_noshadow'

# Eckpfeiler
for i in range(NP):
    sa = SEG[(i - 1) % NP]; sb = SEG[i]
    pnt = P[PERIM[i]]
    d = norm2(add2(sa['n'], sb['n']))
    tt = (-d[1], d[0])
    hi = FASCIA0 - 0.10
    obox(B['Stahl'], W(add2(pnt, d, 0.02)), D3(tt), D3(d), 0.24, 0.24, 0.12, hi, ch=0.03)
    if sa['top'] > LOW_SP + 0.5 and sb['top'] > LOW_SP + 0.5:
        obox(B['Stahl'], W(add2(pnt, d, 0.02)), D3(tt), D3(d), 0.2, 0.2, LOW_SP, HIGH_SP - 0.02, ch=0.03)


# ======================================================================= Rinnen, Kehlen, Lichtband
def gutter_profile(sp):
    return [(-0.26, sp - 0.02), (0.24, sp - 0.02), (0.26, sp + 0.03), (0.26, sp + 0.15), (0.21, sp + 0.15), (0.21, sp + 0.04),
            (-0.21, sp + 0.04), (-0.21, sp + 0.07), (-0.26, sp + 0.07)]


def run_band(names, profile, Bx):
    pts = [P[k] for k in names]
    nrm = []
    for a, c in zip(names[:-1], names[1:]):
        i = PERIM.index(a)
        nrm.append(SEG[i]['n'])
    st = band(Bx, pts, nrm, profile, closed=False)
    orient_faces(Bx, st, D3(nrm[0]), sample=st + 1)


Bst = B['Stahl']
for names, sp in ((['N13', 'N15', 'W5', 'S4'], LOW_SP), (['S0', 'W1', 'N1'], LOW_SP), (['N11', 'N12'], HIGH_SP)):
    run_band(names, gutter_profile(sp), Bst)


def line_band(Bx, pa, pb, toward, profile):
    """Profil (o, y) entlang Strecke pa->pb; +o zeigt weg von 'toward'"""
    t = norm2(sub2(pb, pa)); n = (-t[1], t[0])
    if dot2(n, sub2(toward, pa)) > 0: n = (-n[0], -n[1])
    st = band(Bx, [pa, pb], [n], profile, closed=False)
    orient_faces(Bx, st, D3(n), sample=st + 1)


def poly_band(Bx, names, toward_pt, profile):
    pts = [P[k] for k in names]; nrm = []
    for a, c in zip(pts[:-1], pts[1:]):
        t = norm2(sub2(c, a)); n = (-t[1], t[0])
        if dot2(n, sub2(toward_pt, a)) > 0: n = (-n[0], -n[1])
        nrm.append(n)
    st = band(Bx, pts, nrm, profile, closed=False)
    orient_faces(Bx, st, D3(nrm[0]), sample=st + 1)
    return pts, nrm


# Kehlrinnen zwischen gleich hohen Tonnen (+ Unterzug innen)
VALLEYS = [(('J2', 'N10'), HIGH_SP), (('J3', 'N9'), HIGH_SP), (('W4', 'S3'), SM_SP), (('W3', 'S2'), SM_SP), (('W2', 'S1'), SM_SP)]
for (a, c), sp in VALLEYS:
    pa, pc = P[a], P[c]
    if a.startswith('W'):     # kleine Tonnen: Rinne endet am Giebel, beginnt in P1
        pa = add2(pa, norm2(sub2(pa, pc)), 0.8)
    t = norm2(sub2(pc, pa)); n = (-t[1], t[0])
    prof = [(-0.30, sp - 0.02), (0.30, sp - 0.02), (0.30, sp + 0.05), (0.26, sp + 0.05), (0.26, sp + 0.03), (-0.26, sp + 0.03), (-0.26, sp + 0.05), (-0.30, sp + 0.05)]
    st = band(Bst, [pa, pc], [n], prof, closed=False); orient_faces(Bst, st, (0, 0, -1), sample=st)
    Bu = BL('Innen_Unterzug', 'paneel')
    st = band(Bu, [pa, pc], [n], prof_rect(-0.36, 0.36, sp - 0.5, sp - 0.02), closed=False); orient_faces(Bu, st, D3(n), sample=st + 1)
FLAGS['Innen_Unterzug'] = 'nocol'

# Lichtband (Obergaden) zwischen niedriger (P1/P4, West) und hoher Tonne (P2/P6, Ost): N2 -> J1 -> N12
LOWSIDE = P['N15']
pts, nrm = poly_band(Bst, ['N2', 'J1', 'N12'], (100, 22), [(-0.05, LOW_SP - 0.02), (0.30, LOW_SP - 0.02), (0.30, LOW_SP + 0.10), (0.26, LOW_SP + 0.10),
                                                             (0.26, LOW_SP + 0.04), (0.02, LOW_SP + 0.04), (0.02, LOW_SP + 0.18), (-0.05, LOW_SP + 0.18)])
poly_band(Bst, ['N2', 'J1', 'N12'], (100, 22), [(-0.30, HIGH_SP - 0.10), (0.06, HIGH_SP - 0.10), (0.06, HIGH_SP + 0.02), (0.30, HIGH_SP + 0.02),
                                                 (0.33, HIGH_SP + 0.06), (0.33, HIGH_SP + 0.16), (-0.30, HIGH_SP + 0.16)])
poly_band(BL('Innen_Unterzug', 'paneel'), ['N2', 'J1', 'N12'], (100, 22), prof_rect(-0.10, 0.36, LOW_SP - 0.5, LOW_SP - 0.02))
for a, c in (('N2', 'J1'), ('J1', 'N12')):
    pa, pc = P[a], P[c]; L = dist2(pa, pc)
    t = norm2(sub2(pc, pa)); n = (-t[1], t[0])
    if dot2(n, sub2((100, 22), pa)) > 0: n = (-n[0], -n[1])
    sg = dict(A=pa, t=t, n=n, L=L)
    m = max(1, int(round(L / 1.5)))
    for k in range(m + 1):
        s = L * k / m
        if not (a == 'J1' and k == 0):        # J1 nur einmal
            fbox(Bst, sg, s - 0.035, s + 0.035, -0.08, 0.06, LOW_SP + 0.04, HIGH_SP - 0.08)
        if k < m:
            fquad(BL('Glas', 'glas'), sg, s + 0.04, s + L / m - 0.04, 0.0, LOW_SP + 0.18, HIGH_SP - 0.10)

# Flachdach P13 (um die Trommel) + Attika-Rinne an P2/P3
P13 = [P['N4'], P['N5'], P['J3'], P['J2']]
prism(BL('Dach', 'dach'), [W(p) for p in P13], HIGH_SP - 0.45, HIGH_SP - 0.02)
FLAGS['Dach'] = ''

# ======================================================================= Giebelboegen mit Faecherfenster
GABLES = []
for i in range(NP):
    sg = SEG[i]
    if sg['kind'] != 'gable': continue
    V = VAULTS[sg['vault']]
    e = V.ends['A'] if V.ends['A']['type'] == 'gable' else V.ends['B']
    GABLES.append((sg, V, e))


def arch(sg, V, e):
    small = V.name.startswith('S')
    Lo, Ro = e['Lo'], e['Ro']
    M2 = lerp2(Lo, Ro, 0.5); ag = dist2(Lo, Ro) / 2; tg = norm2(sub2(Ro, Lo)); n = sg['n']
    sp, b = V.sp, V.b
    tgB, nB = D3(tg), D3(n)
    Bs = B['Stahl']; Bg = BL('Glas', 'glas')
    nseg = 40 if not small else 28
    # Bogenrahmen: Profil (delta radial, o aus der Ebene)
    if small:
        prof = [(-0.20, -0.10), (0.05, -0.10), (0.05, 0.09), (0.02, 0.12), (-0.17, 0.12), (-0.20, 0.09)]
        leg0 = FLOOR + 0.0
    else:
        prof = [(-0.30, -0.12), (0.06, -0.12), (0.06, 0.10), (0.02, 0.14), (-0.26, 0.14), (-0.30, 0.10)]
        leg0 = LOW_SP
    def stn(theta, h):
        org = W(add2(M2, tg, -ag * math.cos(theta)), h + b * math.sin(theta))
        ax1 = tgB * (-math.cos(theta)) + UP * math.sin(theta)
        return (org, ax1, nB)
    stations = []
    if sp - leg0 > 0.05:
        for h in (leg0, sp): stations.append(stn(0.0, h))
    for k in range(1 if stations else 0, nseg + 1):
        stations.append(stn(math.pi * k / nseg, sp))
    if sp - leg0 > 0.05:
        stations.append(stn(math.pi, leg0))
    st = len(Bs.f)
    sweep(Bs, stations, prof)
    # Lichtbogen: Glas, Kaempferriegel, Nabe, Ring, Speichen
    di = prof[0][0]
    ai, bi = ag + di, b + di
    def pt(theta, k, o):
        return W(add2(add2(M2, tg, -ai * k * math.cos(theta)), n, o), sp + bi * k * math.sin(theta))
    poly = [pt(math.pi * k / nseg, 1.0, -0.05) for k in range(nseg + 1)]
    polygon_cap(Bg, poly, nB)
    fbox(Bs, sg if False else dict(A=Lo, t=tg, n=n), 0.0, 2 * ag, -0.09, 0.08, sp - 0.07, sp + 0.07)
    hub, ring = (0.15, 0.50) if small else (0.11, 0.44)
    spokes = [math.pi * k / 6 for k in range(1, 6)] if small else [math.pi * k / 8 for k in range(1, 8)]
    for kr, w in ((hub, 0.05), (ring, 0.045)):
        stl = []
        for k in range(nseg + 1):
            th = math.pi * k / nseg
            org = pt(th, kr, 0.0); ax1 = tgB * (-math.cos(th)) + UP * math.sin(th)
            stl.append((org, ax1, nB))
        sweep(Bs, stl, [(-w / 2, -0.04), (w / 2, -0.04), (w / 2, 0.06), (-w / 2, 0.06)])
    for th in spokes:
        p0 = pt(th, hub, 0.01); p1 = pt(th, 1.0, 0.01)
        box3(Bs, p0, p1, 0.05, 0.09, upv=nB)


for sg, V, e in GABLES: arch(sg, V, e)

# ======================================================================= Kleine Giebel: Markisen
FONT_OK = os.path.exists(FONT_MP)
for sg, V, e in GABLES:
    if not V.name.startswith('S'): continue
    Bm = BL('Markise', 'markise'); Bs = B['Stahl']
    L = sg['L']; a, b = 0.25, L - 0.25
    fbox(Bs, sg, a, b, 0.02, 0.24, 3.28, 3.46, ch=0.03)                       # Kassette
    y_hi, y_lo, o_out = 3.32, 2.78, 1.55
    pts = [F(sg, a + 0.02, 0.2, y_hi), F(sg, b - 0.02, 0.2, y_hi), F(sg, b - 0.02, o_out, y_lo), F(sg, a + 0.02, o_out, y_lo)]
    polygon_cap(Bm, pts, UP)
    # Volant
    fquad(Bm, sg, a + 0.02, b - 0.02, o_out, y_lo - 0.26, y_lo)
    fbox(Bs, sg, a + 0.02, b - 0.02, o_out - 0.03, o_out + 0.02, y_lo - 0.03, y_lo + 0.02)  # Fallstange
    for s in (a + 0.05, b - 0.05):
        box3(Bs, F(sg, s, 0.2, 3.30), F(sg, s, o_out - 0.02, y_lo + 0.02), 0.03, 0.03)
    if FONT_OK:
        TEXTS.append(text_obj(COLL, 'Txt_Markise', 'MÖVENPICK', FONT_MP, 0.17, 0.004, M['schrift_r'],
                              F(sg, L / 2, o_out + 0.006, y_lo - 0.2), None, D3(sg['n'])))

# ======================================================================= Trommel (Moevenpick-Turm)
u_dr = norm2(sub2(P['J3'], P['N5']))           # Achse (nach Sueden), parallel P3-Westtraufe
e_dr = (-u_dr[1], u_dr[0])                      # nach Osten
if e_dr[0] < 0: e_dr = (-e_dr[0], -e_dr[1])
R_DR = 3.62
east_line_pt = add2(P['N5'], e_dr, -0.24)       # Ostkante 0.24 westlich der P3-Westtraufe
nd = norm2(sub2(P['N5'], P['N4'])); nn = (-nd[1], nd[0])
if nn[1] < 0: nn = (-nn[0], -nn[1])             # zeigt nach Sueden (ins Gebaeude)
cand = add2(east_line_pt, e_dr, -R_DR)
def north_dist(p): return dot2(sub2(p, P['N4']), nn)
du = dot2(u_dr, nn)
C_DR = add2(cand, u_dr, (0.30 + R_DR * du - north_dist(cand)) / du)   # Nordscheitel 0.30 innerhalb der Nordkante
jd = norm2(sub2(P['J3'], P['J2'])); jn = (-jd[1], jd[0])
if jn[1] > 0: jn = (-jn[0], -jn[1])
J2s = add2(P['J2'], jn, 0.04)                   # Suedseite: Linie J2-J3, 4 cm nach Norden versetzt


def drum_outline(r_off=0.0, nsemi=36, rc=0.9):
    """D-foermiger Grundriss (DOP 2025 / DOM 2016): Halbkreis im Norden, gerade Seiten, Suedseite entlang J2-J3"""
    R = R_DR + r_off
    pts = []
    for k in range(nsemi + 1):                  # von Ost (a=0) ueber Nord nach West (a=pi)
        a = math.pi * k / nsemi
        pts.append((C_DR[0] + R * (e_dr[0] * math.cos(a) - u_dr[0] * math.sin(a)), C_DR[1] + R * (e_dr[1] * math.cos(a) - u_dr[1] * math.sin(a))))
    pw = add2(C_DR, e_dr, -R); pe = add2(C_DR, e_dr, R)
    sl = add2(J2s, jn, r_off)
    sw_c = line_isect(pw, u_dr, sl, jd); se_c = line_isect(pe, u_dr, sl, jd)
    def round_corner(cn, d_in, d_out, r, nseg=6):
        a0 = add2(cn, d_in, -r); a1 = add2(cn, d_out, r); out = []
        for k in range(nseg + 1):
            t = k / nseg; out.append(lerp2(lerp2(a0, cn, t), lerp2(cn, a1, t), t))
        return out
    pts += round_corner(sw_c, u_dr, jd, rc)
    pts += round_corner(se_c, jd, (-u_dr[0], -u_dr[1]), rc)
    return pts


DR_Y0, DR_Y1 = HIGH_SP - 0.05, 13.30
dr_core = drum_outline(-0.10)
prism(BL('Trommel_Kern', 'trommel_kern'), [W(p) for p in dr_core], DR_Y0, DR_Y1 - 0.45)
outl = drum_outline(-0.10, nsemi=72, rc=0.9)
segs = list(zip(outl, outl[1:] + [outl[0]]))
per = [0.0]
for a_, c_ in segs: per.append(per[-1] + dist2(a_, c_))
Ltot = per[-1]; nfin = int(Ltot / 0.30)
Bt = BL('Trommel', 'trommel')
import bisect
for k in range(nfin):
    d = Ltot * k / nfin
    i = min(len(segs) - 1, bisect.bisect_right(per, d) - 1)
    a_, c_ = segs[i]; L = dist2(a_, c_)
    if L < 1e-6: continue
    p = lerp2(a_, c_, (d - per[i]) / L); tt = norm2(sub2(c_, a_))
    nrm2 = (tt[1], -tt[0])
    if point_in_poly(add2(p, nrm2, 0.05), dr_core): nrm2 = (-nrm2[0], -nrm2[1])
    obox(Bt, W(add2(p, nrm2, 0.07)), D3(tt), D3(nrm2), 0.05, 0.16, DR_Y0 + 0.3, DR_Y1 - 0.48)
prism(BL('Trommel_Band', 'trommel_band'), [W(p) for p in drum_outline(0.08, nsemi=72)], DR_Y1 - 0.48, DR_Y1)
prism(B['Trommel_Band'], [W(p) for p in drum_outline(0.04, nsemi=72)], DR_Y0, DR_Y0 + 0.3)
prism(B['Dach'], [W(p) for p in drum_outline(-0.12, nsemi=72)], DR_Y1 - 0.5, DR_Y1 - 0.12)
for (ds, de, sx, sy, h) in ((-1.2, -1.4, 1.6, 1.0, 0.9), (1.0, -1.6, 1.2, 1.2, 0.7), (0.6, 1.2, 2.0, 0.8, 0.55)):
    c = add2(add2(C_DR, u_dr, ds), e_dr, de)
    obox(BL('Dachtechnik', 'trommel_band'), W(c), D3(e_dr), D3(u_dr), sx, sy, DR_Y1 - 0.12, DR_Y1 - 0.12 + h, ch=0.03)
FLAGS['Dachtechnik'] = 'nocol'
# Gelaender auf dem Trommeldach
rail_pts = drum_outline(-0.35, nsemi=24)
tube(BL('Edelstahl', 'edel'), [W(p, DR_Y1 + 0.95) for p in rail_pts], 0.025, seg=8, closed=True, cap=False)
for p in rail_pts[::3]:
    cylinder(B['Edelstahl'], W(p, DR_Y1 - 0.12), W(p, DR_Y1 + 0.95), 0.02, 6)


def curved_text(body, size, font, mat, y, r_off, aim):
    """Schriftzug Buchstabe fuer Buchstabe dem Trommelgrundriss folgend, mittig auf die Richtung 'aim' ausgerichtet"""
    path = drum_outline(r_off, nsemi=96, rc=0.9)
    seg_ = list(zip(path, path[1:] + [path[0]]))
    cum = [0.0]
    for a_, c_ in seg_: cum.append(cum[-1] + dist2(a_, c_))
    def outn(a_, c_):
        tt = norm2(sub2(c_, a_)); nrm = (tt[1], -tt[0])
        if point_in_poly(add2(lerp2(a_, c_, 0.5), nrm, 0.05), path): nrm = (-nrm[0], -nrm[1])
        return tt, nrm
    best, bi = -9, 0
    for i, (a_, c_) in enumerate(seg_):
        tt, nrm = outn(a_, c_)
        m = lerp2(a_, c_, 0.5); sc_ = dot2(nrm, norm2(sub2(aim, m)))
        if sc_ > best: best, bi = sc_, i
    s_mid = (cum[bi] + cum[bi + 1]) / 2
    objs_ = []
    for ch in body:
        cu = bpy.data.curves.new('Txt_' + ch, 'FONT'); cu.body = ch; cu.font = bpy.data.fonts.load(font, check_existing=True)
        cu.size = size; cu.extrude = 0.06; cu.align_x = 'CENTER'; cu.align_y = 'BOTTOM_BASELINE'; cu.resolution_u = 3
        o = bpy.data.objects.new('Txt_Trommel_' + ch, cu); COLL.objects.link(o); o.data.materials.append(mat); objs_.append(o)
    bpy.context.view_layer.update()
    widths = [max(o.dimensions.x, size * 0.3) + size * 0.14 for o in objs_]
    total = sum(widths)
    tt, nrm = outn(*seg_[bi])
    right = UP.cross(D3(nrm)); sgn = 1 if right.dot(D3(tt)) > 0 else -1
    pos = s_mid - sgn * total / 2
    for o, w in zip(objs_, widths):
        sc_ = (pos + sgn * w / 2) % cum[-1]
        i = min(len(seg_) - 1, bisect.bisect_right(cum, sc_) - 1)
        a_, c_ = seg_[i]; L_ = max(1e-6, dist2(a_, c_)); p = lerp2(a_, c_, (sc_ - cum[i]) / L_)
        t2, n2 = outn(a_, c_)
        Z = D3(n2).normalized(); X = UP.cross(Z).normalized(); Y = Z.cross(X)
        Mx = Matrix((X, Y, Z)).transposed().to_4x4(); Mx.translation = W(p, y); o.matrix_world = Mx
        TEXTS.append(o); pos += sgn * w
    # Montageschienen + Abstandshalter (Buchstaben sitzen auf zwei Rohrschienen vor den Lamellen)
    s_a, s_b = sorted((s_mid - total / 2 - 0.15, s_mid + total / 2 + 0.15))
    rails = {0: [], 1: []}
    k = 0; sv = s_a
    while sv <= s_b + 1e-6:
        q = sv % cum[-1]
        i = min(len(seg_) - 1, bisect.bisect_right(cum, q) - 1)
        a_, c_ = seg_[i]; L_ = max(1e-6, dist2(a_, c_)); p = lerp2(a_, c_, (q - cum[i]) / L_)
        t2, n2 = outn(a_, c_)
        for r, yy in ((0, y + size * 0.18), (1, y + size * 0.58)):
            pr = add2(p, n2, -0.09); rails[r].append(W(pr, yy))
            if k % 3 == 0:
                cylinder(B['Stahl'], W(add2(p, n2, -0.21), yy), W(add2(p, n2, -0.09), yy), 0.015, 6)
        sv += 0.25; k += 1
    for r in rails.values():
        tube(B['Stahl'], r, 0.028, seg=8)
    return objs_


if FONT_OK:
    curved_text('MÖVENPICK', 1.05, FONT_MP, M['schrift_r'], 10.35, 0.24, (0.0, 0.0))

# ======================================================================= Hauptportal: Podest, Stufen, Rampe, Gelaender, Kuebel
sgP = next(SEG[i] for i in range(NP) if SEG[i].get('portal'))
sc = sgP['portal_s']
def SP(s, o): return add2(add2(sgP['A'], sgP['t'], s), sgP['n'], o)
y_pv = ground_y(SP(sc, 3.0))
print('CAFE pavement at portal', round(y_pv, 3))
Bgr = B['Granit']; Bled = BL('LED', 'led'); FLAGS['LED'] = 'nocol_noshadow'
land = (sc - 2.4, sc + 2.4, 0.22, 2.25)              # s0, s1, o0, o1
fbox(Bgr, sgP, land[0], land[1], land[2], land[3], y_pv - 0.3, FLOOR, ch=0.0)
nsteps = max(1, int(round((FLOOR - y_pv) / 0.15)))
rise = (FLOOR - y_pv) / nsteps
for k in range(nsteps - 1):
    top = FLOOR - rise * (k + 1)
    o0 = land[3] + 0.33 * k; o1 = o0 + 0.33
    fbox(Bgr, sgP, land[0] - 0.33 * (k + 1), land[1] - 0.0, o0 - 0.33 * 0, o1, y_pv - 0.3, top)
    fbox(Bgr, sgP, land[0] - 0.33 * (k + 1), land[0] - 0.33 * k, land[2], o0, y_pv - 0.3, top)
    # LED-Stufenlichter
    for s in [land[0] + 0.6 + 0.9 * q for q in range(5)]:
        fbox(Bled, sgP, s - 0.06, s + 0.06, o1 + 0.001, o1 + 0.006, top - 0.07, top - 0.03)
# Rampe suedlich (entlang der Fassade), Gelaender
r0, r1 = land[1], land[1] + 4.6
def ramp_solid(Bx, r0, r1, o0, o1, ya, yb, ybot):
    top = [F(sgP, r0, o0, ya), F(sgP, r1, o0, yb), F(sgP, r1, o1, yb), F(sgP, r0, o1, ya)]
    bot = [F(sgP, r0, o0, ybot), F(sgP, r1, o0, ybot), F(sgP, r1, o1, ybot), F(sgP, r0, o1, ybot)]
    vt = [Bx.vert(p) for p in top]; vb = [Bx.vert(p) for p in bot]
    st = len(Bx.f)
    Bx.face(vt)
    orient_faces(Bx, st, (0, 0, 1))
    flip = Bx.f[st] != vt
    faces = [vb[::-1], [vb[0], vb[1], vt[1], vt[0]], [vb[1], vb[2], vt[2], vt[1]], [vb[2], vb[3], vt[3], vt[2]], [vb[3], vb[0], vt[0], vt[3]]]
    for f in faces: Bx.face(f[::-1] if flip else f)
ramp_solid(B['Granit'], r0, r1, land[2], land[2] + 1.3, FLOOR, y_pv + 0.005, y_pv - 0.3)
Be = BL('Edelstahl', 'edel')
def railing(p_list, h=0.95):
    for i, (pp, yy) in enumerate(p_list):
        cylinder(Be, W(pp, yy), W(pp, yy + h), 0.022, 10)
    tube(Be, [W(pp, yy + h) for pp, yy in p_list], 0.024, seg=10)
    tube(Be, [W(pp, yy + h * 0.5) for pp, yy in p_list], 0.012, seg=8)
railing([(SP(r1 - 0.1, land[2] + 1.25), y_pv), (SP((r0 + r1) / 2, land[2] + 1.25), (y_pv + FLOOR) / 2), (SP(r0 + 0.05, land[2] + 1.25), FLOOR),
         (SP(r0 + 0.05, land[3] - 0.05), FLOOR)])
railing([(SP(land[0] + 0.05, land[2] + 0.1), FLOOR), (SP(land[0] + 0.05, land[3] - 0.4), FLOOR)])
# Pflanzkuebel mit Buchskugeln
Bk = BL('Kuebel', 'paneel'); Bpl = BL('Pflanze', 'pflanze'); FLAGS['Pflanze'] = 'nocol'
for s in (sc - 2.0, sc + 2.0):
    c = SP(s, 1.75)
    obox(Bk, W(c), D3(sgP['t']), D3(sgP['n']), 0.5, 0.5, FLOOR, FLOOR + 0.75, ch=0.02)
    for q in range(5):
        off = (random.uniform(-0.12, 0.12), random.uniform(-0.12, 0.12))
        sphere(Bpl, W(add2(c, off), FLOOR + 0.95 + random.uniform(-0.05, 0.12)), random.uniform(0.2, 0.28), 10, 7)

# ======================================================================= Schriftzuege + Leuchten an der Westblende
FASC_O = 0.10
def fascia_text(sg, s, body, size, font, mat, y=3.12, shear=0.0, ext=0.06):
    TEXTS.append(text_obj(COLL, 'Txt_' + body[:6], body, font, size, ext, mat, F(sg, s, FASC_O + 0.025, y), None, D3(sg['n']), shear=shear))
if os.path.exists(FONT_CAFE):
    fascia_text(sgP, sgP['L'] - 12.0, 'Café Kröpcke', 1.0, FONT_CAFE, M['schrift_w'], y=3.20, shear=0.22)
    sgS = next(SEG[i] for i in range(NP) if SEG[i]['a'] == 'N9')
    fascia_text(sgS, sgS['L'] / 2, 'Café Kröpcke', 0.78, FONT_CAFE, M['schrift_w'], y=3.25, shear=0.22)
    sgN = next(SEG[i] for i in range(NP) if SEG[i]['a'] == 'N1')
    fascia_text(sgN, sgN['L'] / 2, 'Café Kröpcke', 0.72, FONT_CAFE, M['schrift_w'], y=3.27, shear=0.22)
if FONT_OK:
    fascia_text(sgP, sc, 'MÖVENPICK', 0.40, FONT_MP, M['schrift_w'], y=3.30)


# Globusleuchten (Instanzen): Wandarm + 1/2/3 Kugeln
def make_mesh(name, fill, mat):
    tmp = Builder(name, mat); fill(tmp)
    me = bpy.data.meshes.new(name); me.from_pydata(tmp.v, [], tmp.f); me.validate()
    for p in me.polygons: p.use_smooth = True
    try: me.set_sharp_from_angle(angle=math.radians(40))
    except Exception: pass
    me.materials.append(mat)
    return me


def lamp_arm(n):
    def fill(Bx):
        obox(Bx, (0, 0.02, 0), (1, 0, 0), (0, 1, 0), 0.10, 0.04, -0.12, 0.12, ch=0.01)
        if n == 1:
            tube(Bx, [Vector((0, 0.03, 0)), Vector((0, 0.25, 0.02)), Vector((0, 0.36, 0.10)), Vector((0, 0.38, 0.22))], 0.018, 8)
            cylinder(Bx, (0, 0.38, 0.20), (0, 0.38, 0.27), 0.045, 12)
        else:
            tube(Bx, [Vector((0, 0.03, 0)), Vector((0, 0.32, 0.0))], 0.022, 8)
            xs = [-0.28, 0.28] if n == 2 else [-0.30, 0.0, 0.30]
            tube(Bx, [Vector((xs[0], 0.34, 0.12)), Vector((xs[0], 0.34, 0.03)), Vector((xs[0] * 0.6, 0.34, 0.0)), Vector((0, 0.34, 0.0)),
                      Vector((xs[-1] * 0.6, 0.34, 0.0)), Vector((xs[-1], 0.34, 0.03)), Vector((xs[-1], 0.34, 0.12))], 0.018, 8)
            for x in xs:
                hz = 0.30 if (n == 3 and x == 0.0) else 0.12
                if n == 3 and x == 0.0:
                    cylinder(Bx, (0, 0.34, 0.0), (0, 0.34, hz), 0.018, 8)
                cylinder(Bx, (x, 0.34, hz - 0.01), (x, 0.34, hz + 0.06), 0.045, 12)
    return fill


def lamp_globes(n):
    def fill(Bx):
        if n == 1: sphere(Bx, (0, 0.38, 0.45), 0.19, 18, 12)
        else:
            xs = [-0.28, 0.28] if n == 2 else [-0.30, 0.0, 0.30]
            for x in xs:
                hz = 0.30 if (n == 3 and x == 0.0) else 0.12
                sphere(Bx, (x, 0.34, hz + 0.24), 0.19, 18, 12)
    return fill


LAMP_ME = {n: (make_mesh(f'Leuchte_Arm_{n}', lamp_arm(n), M['stahl']), make_mesh(f'Leuchte_Kugel_{n}', lamp_globes(n), M['globus'])) for n in (1, 2, 3)}
INSTANCES = []   # (Objekt, Exportgruppe)


def place(me, loc, xdir, ydir, group):
    o = bpy.data.objects.new(me.name + '_i', me); INST.objects.link(o)
    X = Vector(xdir).normalized(); Y = Vector(ydir).normalized(); Z = X.cross(Y).normalized()
    if Z.z < 0: X = -X; Z = -Z
    Mx = Matrix((X, Y, Z)).transposed().to_4x4(); Mx.translation = Vector(loc)
    o.matrix_world = Mx; INSTANCES.append((o, group)); return o


def lamp(sg, s, y, n, o=FASC_O):
    arm, glb = LAMP_ME[n]
    loc = F(sg, s, o, y); xd = D3(sg['t']); yd = D3(sg['n'])
    place(arm, loc, xd, yd, 'Leuchten_Arm'); place(glb, loc, xd, yd, 'Globusleuchten')


L_P = sgP['L']
for s, n in ((sc - 2.15, 3), (sc + 2.15, 3), (L_P - 12.0 - 4.1, 2), (L_P - 0.9, 2), (1.2, 2)):
    lamp(sgP, s, 3.45, n)
# kleine Giebel: Einzelkugeln an den Pfeilern
for nm in ('S4', 'S3', 'S2', 'S1', 'S0'):
    i = PERIM.index(nm); sgx = SEG[i] if nm != 'S0' else SEG[PERIM.index('S1')]
    s = 0.0 if nm != 'S0' else sgx['L']
    lamp(sgx, s, 3.52, 1, o=0.13)
for nm in ('N1', 'N9', 'N10', 'N11', 'N8'):
    i = PERIM.index(nm); sgx = SEG[i]
    for s in (0.9, sgx['L'] - 0.9):
        lamp(sgx, s, 3.45, 2)
FLAGS['Globusleuchten'] = 'nocol_noshadow'; FLAGS['Leuchten_Arm'] = 'nocol'

# Heizstrahler + Strahler unter der Westblende (Foto 2021)
Bh = BL('Heizstrahler', 'stahl'); FLAGS['Heizstrahler'] = 'nocol'
for s in (sc + 3.2, sc + 6.2, sc + 9.2, sc - 4.0):
    if 0.5 < s < L_P - 0.5:
        fbox(Bh, sgP, s - 0.45, s + 0.45, FASC_O + 0.05, FASC_O + 0.22, FASCIA0 - 0.05, FASCIA0 + 0.07, ch=0.02)
        fbox(BL('LED', 'led'), sgP, s - 0.40, s + 0.40, FASC_O + 0.07, FASC_O + 0.20, FASCIA0 - 0.055, FASCIA0 - 0.05)

# ======================================================================= Aussengastronomie (Instanzen)
def chair_frame(Bx):
    for x in (-0.21, 0.21):
        tube(Bx, [Vector((x, -0.22, 0.0)), Vector((x, -0.20, 0.45)), Vector((x, 0.20, 0.45)), Vector((x, 0.22, 0.0))], 0.012, 6, cap=True)
        tube(Bx, [Vector((x, 0.19, 0.45)), Vector((x, 0.24, 0.86))], 0.012, 6)
    tube(Bx, [Vector((-0.21, -0.20, 0.43)), Vector((0.21, -0.20, 0.43))], 0.01, 6)


def chair_seat(Bx):
    for k in range(5):
        y = -0.19 + k * 0.085
        obox(Bx, (0, y, 0), (1, 0, 0), (0, 1, 0), 0.44, 0.07, 0.45, 0.47, ch=0.0)
    for k in range(3):
        z = 0.58 + k * 0.1
        c = Vector((0, 0.205 + (z - 0.45) * 0.1, 0))
        obox(Bx, c, (1, 0, 0), (0, 1, 0), 0.44, 0.025, z, z + 0.07)


def table_top(Bx):
    obox(Bx, (0, 0, 0), (1, 0, 0), (0, 1, 0), 0.72, 0.72, 0.72, 0.75, ch=0.02)


def table_base(Bx):
    cylinder(Bx, (0, 0, 0.0), (0, 0, 0.72), 0.03, 10)
    for a in (0.785, 2.356):
        box3(Bx, (math.cos(a) * 0.3, math.sin(a) * 0.3, 0.02), (-math.cos(a) * 0.3, -math.sin(a) * 0.3, 0.02), 0.04, 0.03)


def umbrella_pole(Bx):
    cylinder(Bx, (0, 0, 0.0), (0, 0, 2.62), 0.028, 10)
    obox(Bx, (0, 0, 0), (1, 0, 0), (0, 1, 0), 0.55, 0.55, 0.0, 0.10, ch=0.03)
    for k in range(8):
        a = math.pi / 4 * k + math.pi / 8
        r = 1.55 / math.cos(math.pi / 8) if False else 1.55
        tube(Bx, [Vector((0, 0, 2.55)), Vector((math.cos(a) * r * 0.98, math.sin(a) * r * 0.98, 2.25))], 0.008, 4, cap=True)


def umbrella_cloth(Bx):
    apex = Vector((0, 0, 2.62)); ring = []
    hs = 1.55
    corners = [Vector((hs, hs, 2.24)), Vector((-hs, hs, 2.24)), Vector((-hs, -hs, 2.24)), Vector((hs, -hs, 2.24))]
    for i in range(4):
        a, c = corners[i], corners[(i + 1) % 4]
        m = (a + c) / 2; m.z = 2.30
        Bx.poly([apex, a, m]); Bx.poly([apex, m, c])
        # Volant
        Bx.poly([a, m, m - Vector((0, 0, 0.16)), a - Vector((0, 0, 0.16))])
        Bx.poly([m, c, c - Vector((0, 0, 0.16)), m - Vector((0, 0, 0.16))])
    Bx.poly([apex + Vector((0, 0, 0.08)), apex + Vector((0.12, 0, -0.03)), apex + Vector((0, 0.12, -0.03))])
    Bx.poly([apex + Vector((0, 0, 0.08)), apex + Vector((0, 0.12, -0.03)), apex + Vector((-0.12, 0, -0.03))])
    Bx.poly([apex + Vector((0, 0, 0.08)), apex + Vector((-0.12, 0, -0.03)), apex + Vector((0, -0.12, -0.03))])
    Bx.poly([apex + Vector((0, 0, 0.08)), apex + Vector((0, -0.12, -0.03)), apex + Vector((0.12, 0, -0.03))])


def windscreen(Bx):
    obox(Bx, (0, 0, 0), (1, 0, 0), (0, 1, 0), 1.78, 0.04, 0.08, 1.05, ch=0.0)


def windscreen_post(Bx):
    obox(Bx, (0, 0, 0), (1, 0, 0), (0, 1, 0), 0.06, 0.06, 0.0, 1.12, ch=0.01)
    obox(Bx, (0, 0, 0), (1, 0, 0), (0, 1, 0), 0.3, 0.3, 0.0, 0.03)


ME = {
    'chair_frame': make_mesh('Stuhl_Gestell', chair_frame, M['stahl']),
    'chair_seat': make_mesh('Stuhl_Sitz', chair_seat, M['stuhl']),
    'table_top': make_mesh('Tisch_Platte', table_top, M['holz']),
    'table_base': make_mesh('Tisch_Fuss', table_base, M['stahl']),
    'umb_pole': make_mesh('Schirm_Mast', umbrella_pole, M['stahl']),
    'umb_cloth': make_mesh('Schirm_Bespannung', umbrella_cloth, M['markise']),
    'ws': make_mesh('Windschutz_Paneel', windscreen, M['holz']),
    'ws_post': make_mesh('Windschutz_Pfosten', windscreen_post, M['stahl']),
}


def place_plan(me, p, yaw_dir, y, group):
    """Instanz am Planpunkt p (Hoehe y), lokale +Y-Achse zeigt in Planrichtung yaw_dir"""
    yd = D3(yaw_dir).normalized(); xd = yd.cross(UP).normalized()
    return place(me, W(p, y), xd, yd, group)


# Terrasse vor den kleinen Giebeln: Linie S0 -> S4, Normale nach Westen
g0, g1 = P['S0'], P['S4']
tline = norm2(sub2(g1, g0)); tn = (-tline[1], tline[0])
if point_in_poly(add2(lerp2(g0, g1, 0.5), tn, 0.3), POLY): tn = (-tn[0], -tn[1])
Lt = dist2(g0, g1)
tables = []
for row, o in enumerate((1.85, 3.75)):
    n = 7
    for k in range(n):
        s = 0.9 + (Lt - 1.8) * k / (n - 1) + (0.4 if row else 0.0)
        tables.append(add2(add2(g0, tline, s), tn, o))
for p in tables:
    y = ground_y(p)
    place_plan(ME['table_top'], p, tline, y, 'Moebel_Holz'); place_plan(ME['table_base'], p, tline, y, 'Moebel_Stahl')
    for side in (-1, 1):
        q = add2(p, tn, 0.52 * side)
        rot = (tn[0] * side, tn[1] * side)
        jit = random.uniform(-0.25, 0.25)
        rot = norm2(add2(rot, tline, jit))
        place_plan(ME['chair_frame'], q, rot, ground_y(q), 'Moebel_Stahl'); place_plan(ME['chair_seat'], q, rot, ground_y(q), 'Stuhl_Rot')
for k in range(4):
    s = Lt * (k + 0.5) / 4
    p = add2(add2(g0, tline, s), tn, 2.8)
    y = ground_y(p)
    place_plan(ME['umb_pole'], p, tline, y, 'Moebel_Stahl'); place_plan(ME['umb_cloth'], p, tline, y, 'Schirme')
# Windschutz entlang der Terrassenkante
o_ws = 5.0
n_ws = int(Lt / 1.85)
for k in range(n_ws + 1):
    s = 0.3 + 1.85 * k
    p = add2(add2(g0, tline, s), tn, o_ws); place_plan(ME['ws_post'], p, tn, ground_y(p), 'Moebel_Stahl')
    if k < n_ws:
        q = add2(add2(g0, tline, s + 0.925), tn, o_ws); place_plan(ME['ws'], q, tn, ground_y(q), 'Moebel_Holz')
# Terrasse entlang der Westfassade P4 (suedlich des Portals)
for k in range(4):
    s = sc + 4.0 + 2.6 * k
    if s > L_P - 1.5: break
    p = SP(s, 2.1); y = ground_y(p)
    place_plan(ME['table_top'], p, sgP['t'], y, 'Moebel_Holz'); place_plan(ME['table_base'], p, sgP['t'], y, 'Moebel_Stahl')
    for side in (-1, 1):
        q = SP(s + 0.52 * side, 2.1)
        rot = (sgP['t'][0] * side, sgP['t'][1] * side)
        place_plan(ME['chair_frame'], q, rot, ground_y(q), 'Moebel_Stahl'); place_plan(ME['chair_seat'], q, rot, ground_y(q), 'Stuhl_Rot')
for k in range(7):
    s = sc + 2.9 + 1.85 * k
    if s > L_P - 0.8: break
    p = SP(s, 3.3); place_plan(ME['ws_post'], p, sgP['n'], ground_y(p), 'Moebel_Stahl')
    if k < 6 and s + 1.85 < L_P - 0.8:
        q = SP(s + 0.925, 3.3); place_plan(ME['ws'], q, sgP['n'], ground_y(q), 'Moebel_Holz')
FLAGS['Schirme'] = 'nocol'

# ======================================================================= Innenraum (Silhouetten, warmes Licht)
def ichair(Bx):
    obox(Bx, (0, 0, 0), (1, 0, 0), (0, 1, 0), 0.42, 0.42, 0.42, 0.47)
    obox(Bx, (0, 0.19, 0), (1, 0, 0), (0, 1, 0), 0.42, 0.04, 0.47, 0.9)
    for x in (-0.18, 0.18):
        for y in (-0.18, 0.18):
            obox(Bx, (x, y, 0), (1, 0, 0), (0, 1, 0), 0.035, 0.035, 0.0, 0.42)


def itable(Bx):
    cylinder(Bx, (0, 0, 0.73), (0, 0, 0.76), 0.38, 16)
    cylinder(Bx, (0, 0, 0.0), (0, 0, 0.73), 0.04, 8)
    cylinder(Bx, (0, 0, 0.0), (0, 0, 0.03), 0.24, 12)


def pendant(Bx):
    sphere(Bx, (0, 0, 0), 0.17, 14, 9)


def pendant_cord(Bx):
    cylinder(Bx, (0, 0, 0.15), (0, 0, 2.2), 0.006, 4)


ME['ichair'] = make_mesh('Innen_Stuhl', ichair, M['innen_dunkel'])
ME['itable'] = make_mesh('Innen_Tisch', itable, M['innen_dunkel'])
ME['pend'] = make_mesh('Innen_Pendel', pendant, M['licht'])
ME['cord'] = make_mesh('Innen_Pendel_Kabel', pendant_cord, M['stahl'])
ME['cord_s'] = make_mesh('Innen_Pendel_Kabel_kurz', lambda Bx: cylinder(Bx, (0, 0, 0.15), (0, 0, 0.95), 0.006, 4), M['stahl'])


def inside_ok(p, margin=0.9):
    if not point_in_poly(p, POLY): return False
    for k in range(8):
        a = math.pi * k / 4
        if not point_in_poly((p[0] + math.cos(a) * margin, p[1] + math.sin(a) * margin), POLY): return False
    return True


P13_poly = [P['N4'], P['N5'], P['J3'], P['J2']]
for nm in ('P1', 'P4', 'P2', 'P6', 'D3', 'P3', 'P5'):
    V = VAULTS[nm]
    s0 = V.sw(V.ends['A']['Lo'])[0]; s1 = V.sw(V.ends['B']['Lo'])[0]
    s = s0 + 1.6
    while s < s1 - 1.4:
        wl, wr = V.wLo(s), V.wRo(s); c = (wl + wr) / 2; a = (wl - wr) / 2
        for w in (c + a * 0.52, c - a * 0.52):
            p = V.pos(s, w)
            if inside_ok(p, 1.0) and not point_in_poly(p, P13_poly):
                place_plan(ME['itable'], p, V.u, FLOOR, 'Innen')
                for side in (-1, 1):
                    q = add2(p, V.u, 0.62 * side)
                    place_plan(ME['ichair'], q, (V.u[0] * side, V.u[1] * side), FLOOR, 'Innen')
        # Pendelleuchte auf der Mittellinie
        pc = V.pos(s, c)
        if inside_ok(pc, 0.5) and not point_in_poly(pc, P13_poly):
            hy = V.sp + V.b - 2.6
            place_plan(ME['pend'], pc, V.u, hy, 'Innen_Licht'); place_plan(ME['cord'], pc, V.u, hy, 'Innen_Licht_Kabel')
        s += 2.6
for nm, *_ in SMALL:
    V = VAULTS[nm]
    s0 = V.sw(V.ends['A']['Lo'])[0]
    p = V.pos(s0 + 1.8, (V.wLo(s0 + 1.8) + V.wRo(s0 + 1.8)) / 2)
    place_plan(ME['itable'], p, V.u, FLOOR, 'Innen')
    for side in (-1, 1):
        q = add2(p, V.v, 0.6 * side); place_plan(ME['ichair'], q, (V.v[0] * side, V.v[1] * side), FLOOR, 'Innen')
    place_plan(ME['pend'], p, V.u, SM_SP + 0.6, 'Innen_Licht'); place_plan(ME['cord_s'], p, V.u, SM_SP + 0.6, 'Innen_Licht_Kabel')
# Kern unter der Trommel (Kueche/Nebenraeume): warm beleuchtete Holzwand
Bw = BL('Innen_Wand', 'innen_wand'); FLAGS['Innen_Wand'] = 'nocol_noshadow'
_nr = []
for a_, b_ in zip(P13_poly, P13_poly[1:] + P13_poly[:1]):
    t_ = norm2(sub2(b_, a_)); n_ = (-t_[1], t_[0])
    if point_in_poly(add2(lerp2(a_, b_, 0.5), n_, 0.1), P13_poly): n_ = (-n_[0], -n_[1])
    _nr.append(n_)
core = offset_poly(P13_poly, _nr, -0.45)
prism(Bw, [W(p) for p in core], FLOOR, HIGH_SP - 0.46)
# Theke "Delikatessen" hinter der Westfassade P4 (Foto 2021)
Bth = BL('Innen_Theke', 'innen_dunkel'); FLAGS['Innen_Theke'] = 'nocol_noshadow'
Bli = BL('Innen_Licht', 'licht'); FLAGS['Innen_Licht'] = 'nocol_noshadow'
s_a, s_b = sc + 3.0, min(L_P - 1.5, sc + 10.0)
fbox(Bth, sgP, s_a, s_b, -2.6, -1.9, FLOOR, 0.95)
fbox(Bli, sgP, s_a + 0.05, s_b - 0.05, -2.55, -1.95, 0.95, 1.25)
fbox(Bth, sgP, s_a, s_b, -2.6, -1.9, 1.25, 1.30)
fbox(Bth, sgP, s_a - 0.5, s_b + 0.5, -4.4, -4.2, FLOOR, 2.6)
for h in (1.2, 1.6, 2.0):
    fbox(Bli, sgP, s_a - 0.4, s_b + 0.4, -4.2, -3.9, h, h + 0.04)
FLAGS['Innen'] = 'nocol_noshadow'; FLAGS['Innen_Licht_Kabel'] = 'nocol_noshadow'

# Rueckwaende der kleinen Tonnen (verdecken die Aussenhaut von P1 beim Blick durch die Faecherfenster)
Bw2 = BL('Innen_Wand', 'innen_wand')
for nm, g0, g1, w0, w1 in SMALL:
    V = VAULTS[nm]
    pa, pb = P[w0], P[w1]
    s_w = (V.sw(add2(pa, V.u, -0.06))[0] + V.sw(add2(pb, V.u, -0.06))[0]) / 2
    pts = []
    for k in range(25):
        th = math.pi * k / 24; c = (1 - math.cos(th)) / 2
        w = V.wLi(s_w) * (1 - c) + V.wRi(s_w) * c
        pts.append(W(V.pos(s_w, w), V.sp + V.bi * math.sin(th) - 0.01))
    wl, wr = V.wLi(s_w), V.wRi(s_w)
    pts = pts + [W(V.pos(s_w, wr), FASCIA0), W(V.pos(s_w, wl), FASCIA0)]       # nur oberhalb des Durchgangs zu P1
    polygon_cap(Bw2, pts, D3((-V.u[0], -V.u[1])))
# Fallrohre an den Enden der Aussenrinnen
Bdp = B['Stahl']
for nm, sgi, at_end in (('N13', 'N13', False), ('S4', 'W5', True), ('S0', 'S0', False), ('N1', 'W1', True), ('N11', 'N11', False)):
    sg = SEG[PERIM.index(sgi)]
    s = sg['L'] - 0.45 if at_end else 0.45
    top = sg['top']
    gy = ground_y(add2(add2(sg['A'], sg['t'], s), sg['n'], 0.35))
    o = 0.17
    tube(Bdp, [F(sg, s, 0.10, top - 0.02), F(sg, s, o, top - 0.25), F(sg, s, o, gy + 0.25), F(sg, s, o + 0.12, gy + 0.08)], 0.05, seg=10)
    for yy in [gy + 0.6 + 1.1 * k for k in range(int((top - gy - 0.8) / 1.1))]:
        cylinder(Bdp, F(sg, s, 0.09, yy), F(sg, s, o, yy), 0.012, 6)
        cylinder(Bdp, F(sg, s, o, yy - 0.03), F(sg, s, o, yy + 0.03), 0.058, 10)

# ======================================================================= Kollisionshuelle (unsichtbar im Spiel)
Bc = Builder('col_Cafe_Kroepcke_Huelle', M['col'])
prism(Bc, [W(p) for p in POLY], -0.6, LOW_SP)

# ======================================================================= Objekte erzeugen
objs = {}
for key, Bx in B.items():
    o = Bx.build(COLL, weld=0.0005 if key in ('Kupfer', 'Decke_innen') else 0.0)
    if o: objs[key] = o
oc = Bc.build(COLL)

# Fasen (Bevel) an harten Kanten der groben Bauteile
for key in ('Paneel', 'Granit', 'Portal_Rot', 'Trommel_Band', 'Kuebel'):
    o = objs.get(key)
    if not o: continue
    m = o.modifiers.new('Fase', 'BEVEL'); m.width = 0.006; m.segments = 1; m.limit_method = 'ANGLE'; m.angle_limit = math.radians(50)
    m.harden_normals = False

# Text -> Mesh
for t in TEXTS:
    pass

# ---- Statistik: Dreiecke (inkl. Instanzen, Modifikatoren)
dg = bpy.context.evaluated_depsgraph_get()
def tri_count(o):
    if o.type == 'MESH':
        ev = o.evaluated_get(dg); me = ev.to_mesh(); n = sum(len(p.vertices) - 2 for p in me.polygons); ev.to_mesh_clear(); return n
    if o.type == 'FONT':
        ev = o.evaluated_get(dg); me = ev.to_mesh(); n = sum(len(p.vertices) - 2 for p in me.polygons) if me else 0; ev.to_mesh_clear(); return n
    return 0
stats = {}
tot = 0
for o in list(COLL.objects) + list(INST.objects):
    if o.name.startswith('col_'): continue
    n = tri_count(o); tot += n
    k = o.name.split('_i')[0] if o.name.endswith('_i') or '_i.' in o.name else o.name
    stats[k] = stats.get(k, 0) + n
print('CAFE tris total', tot)
for k, v in sorted(stats.items(), key=lambda x: -x[1])[:40]: print('   ', k, v)

# Watertight-Check der Tonnenschalen
for key in ('Kupfer', 'Decke_innen'):
    pass
bm = bmesh.new()
for key in ('Kupfer', 'Decke_innen'):
    me = objs[key].data; bm.from_mesh(me)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005)
nm_edges = [e for e in bm.edges if not e.is_manifold]
print('CAFE vault shells: verts', len(bm.verts), 'faces', len(bm.faces), 'non-manifold edges', len(nm_edges))
bm.free()

json.dump({'tris': tot, 'groups': stats}, open(os.path.join(HERE, 'build_stats.json'), 'w'), indent=1)
SC['cafe_tris'] = tot

# ======================================================================= Speichern (Arbeitsstand, Instanzen bleiben verlinkt)
if OUT_BLEND:
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
    print('saved', OUT_BLEND)

# ======================================================================= Export: Instanzen + Texte realisieren, je Material/Gruppe zusammenfuehren
from collections import defaultdict
bpy.ops.object.select_all(action='DESELECT')
for o in list(COLL.objects) + list(INST.objects):
    o.select_set(True)
bpy.context.view_layer.objects.active = next(iter(COLL.objects))
bpy.ops.object.make_single_user(object=True, obdata=True)
bpy.ops.object.convert(target='MESH')
groups = defaultdict(list)
for o in list(COLL.objects) + list(INST.objects):
    if o.type != 'MESH': continue
    if o.name.startswith('col_'): groups['col'].append(o); continue
    if o.name.startswith('Txt_'):
        mat = o.data.materials[0].name if o.data.materials else 'x'
        groups['Schrift_' + ('Rot' if 'Rot' in mat else 'Weiss') + '|nocol_noshadow'].append(o); continue
    g = None
    for io, grp in INSTANCES:
        if io == o: g = grp
    if g is None:
        g = o.name.replace('Cafe_Kroepcke_', '').split('.')[0]
    f = FLAGS.get(g, '')
    if g.startswith('Innen'): f = f or 'nocol_noshadow'
    groups[g + ('|' + f if f else '')].append(o)
for g, objs_ in groups.items():
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs_: o.select_set(True)
    bpy.context.view_layer.objects.active = objs_[0]
    if len(objs_) > 1: bpy.ops.object.join()
    j = bpy.context.view_layer.objects.active
    if g == 'col':
        j.name = 'col_Cafe_Kroepcke_Huelle'
    else:
        base, _, f = g.partition('|')
        j.name = 'Cafe_Kroepcke_' + base + ('_' + f if f else '')
    # Material je Objekt vereinheitlichen (Join kann mehrere Slots erzeugen)
    if j.parent is not None:
        mw = j.matrix_world.copy(); j.parent = None; j.matrix_world = mw
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
final = [o for o in SC.objects if o.type == 'MESH']
tot2 = 0
for o in final:
    tot2 += sum(len(p.vertices) - 2 for p in o.data.polygons)
print('CAFE export nodes', len(final), 'tris', tot2)
os.makedirs(os.path.dirname(OUT_GLB), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=OUT_GLB, export_format='GLB', export_yup=True, export_apply=True, export_materials='EXPORT',
                          export_vertex_color='MATERIAL', export_extras=False, use_selection=False)
print('exported', OUT_GLB)

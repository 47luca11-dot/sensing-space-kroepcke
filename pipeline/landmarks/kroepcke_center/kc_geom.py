"""Kroepcke-Center Hannover (Kleihues + Kleihues, 2009-2014) -- Geometrie-Generator (Stufe 1, numpy/shapely).

Erzeugt alle Dreiecke (Plan-/Blender-Weltkoordinaten: X Ost, Y Nord, Z oben, Ursprung Kroepcke-Uhr)
und schreibt  kc_geo.npz  (Netze je Gruppe/Material, UV je Ecke) + kc_geo.json (Materialien, Instanzen,
Schrift-Objekte, Statistik). Stufe 2 (kc_blender.py) baut daraus die Blender-Szene und exportiert das GLB.

Hoehen (aus Foto-Vermessung Commons 2014 + LoD2): Kolonnade/Sockelzone 0-7.25 (Bug 7.54),
Fensterreihen 8.90-11.80 / 13.15-16.10 / 17.50-20.05 (Dachgeschoss tiefer liegend), Attika bis 22.20,
Staffel-/Mittelteil bis 26.06, Technik-Einhausung bis 28.64. Achsraster 3.60 m (DOP: Vordach-Rhythmus 3.575 m).
Aufruf: python kc_geom.py <outdir>
"""
import json, math, os, sys
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point, box as sbox
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, linemerge
import kc_plan
import kc_prims as K
from kc_prims import Piece, box, ext_x, ext_z, rect_y, rect_z, rect_x, rod, uvsphere, lathe, emit

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2014)

# ------------------------------------------------------------------ Materialien
M_STONE = 'KC Limestone'
M_COPE = 'KC Limestone Coping'
M_GRAN = 'KC Granite'
M_FRAME = 'KC Metal Bronze'
M_GLASS = 'KC Glass'
M_CGLASS = 'KC Canopy Glass'
M_STEEL = 'KC Steel'
M_STEELD = 'KC Steel Dark'
M_IFLOOR = 'KC Interior Floor'
M_ICEIL = 'KC Interior Ceiling'
M_IWALL = 'KC Interior Wall'
M_ILIGHT = 'KC Interior Light'
M_RACK = 'KC Shop Fixture'
M_TEX_A = 'KC Textile Navy'
M_TEX_B = 'KC Textile Sand'
M_TEX_C = 'KC Textile Red'
M_MANN = 'KC Mannequin'
M_SEDUM = 'KC Green Roof'
M_GRAVEL = 'KC Gravel'
M_PAVER = 'KC Pavers'
M_MEMB = 'KC Roof Membrane'
M_LAWN = 'KC Lawn'
M_PLANT = 'KC Plant'
M_BARK = 'KC Bark'
M_HVAC = 'KC Plant Room Metal'
M_FAN = 'KC Fan Dark'
M_SBLUE = 'KC Sign Blue'
M_SRED = 'KC Sign Red'
M_SWHITE = 'KC Sign White'
M_SBLACK = 'KC Sign Black'
M_CLIGHT = 'KC Canopy Light'

MATS = {
    M_STONE: dict(color=[0.84, 0.81, 0.74], rough=0.78, tex='kc_limestone.png', normal='kc_limestone_n.png', tile=2.0),
    M_COPE: dict(color=[0.86, 0.84, 0.78], rough=0.62, tex='kc_limestone.png', normal='kc_limestone_n.png', tile=2.0),
    M_GRAN: dict(color=[0.40, 0.39, 0.38], rough=0.45),
    M_FRAME: dict(color=[0.20, 0.185, 0.17], metal=0.75, rough=0.38),
    M_GLASS: dict(color=[0.11, 0.15, 0.17], rough=0.04, alpha=0.30, glass=True),
    M_CGLASS: dict(color=[0.70, 0.76, 0.76], rough=0.15, alpha=0.45, glass=True),
    M_STEEL: dict(color=[0.58, 0.60, 0.61], metal=0.85, rough=0.32),
    M_STEELD: dict(color=[0.24, 0.25, 0.26], metal=0.6, rough=0.45),
    M_IFLOOR: dict(color=[0.60, 0.58, 0.55], rough=0.35),
    M_ICEIL: dict(color=[0.86, 0.85, 0.82], rough=0.9),
    M_IWALL: dict(color=[0.40, 0.38, 0.36], rough=0.85),
    M_ILIGHT: dict(color=[1.0, 0.96, 0.88], rough=0.5, emissive=[1.0, 0.95, 0.85], emissive_strength=2.0),
    M_RACK: dict(color=[0.72, 0.72, 0.70], metal=0.8, rough=0.3),
    M_TEX_A: dict(color=[0.10, 0.14, 0.26], rough=0.9),
    M_TEX_B: dict(color=[0.72, 0.64, 0.50], rough=0.9),
    M_TEX_C: dict(color=[0.55, 0.10, 0.12], rough=0.9),
    M_MANN: dict(color=[0.90, 0.89, 0.86], rough=0.4),
    M_SEDUM: dict(color=[1, 1, 1], rough=0.95, tex='kc_sedum.png', tile=4.0),
    M_GRAVEL: dict(color=[1, 1, 1], rough=0.95, tex='kc_gravel.png', normal='kc_gravel_n.png', tile=1.0),
    M_PAVER: dict(color=[1, 1, 1], rough=0.8, tex='kc_pavers.png', tile=2.0),
    M_MEMB: dict(color=[0.20, 0.20, 0.21], rough=0.7),
    M_LAWN: dict(color=[1, 1, 1], rough=0.95, tex='kc_lawn.png', tile=2.0),
    M_PLANT: dict(color=[0.17, 0.30, 0.10], rough=0.85),
    M_BARK: dict(color=[0.28, 0.21, 0.15], rough=0.9),
    M_HVAC: dict(color=[0.76, 0.77, 0.78], metal=0.4, rough=0.45),
    M_FAN: dict(color=[0.12, 0.12, 0.13], metal=0.3, rough=0.6),
    M_SBLUE: dict(color=[0.04, 0.15, 0.48], rough=0.35, emissive=[0.04, 0.15, 0.48], emissive_strength=0.6),
    M_SRED: dict(color=[0.78, 0.05, 0.08], rough=0.35, emissive=[0.78, 0.05, 0.08], emissive_strength=0.6),
    M_SWHITE: dict(color=[0.96, 0.96, 0.96], rough=0.3, emissive=[1, 1, 1], emissive_strength=0.8),
    M_SBLACK: dict(color=[0.025, 0.025, 0.03], rough=0.35),
    M_CLIGHT: dict(color=[1.0, 0.95, 0.85], rough=0.5, emissive=[1.0, 0.93, 0.80], emissive_strength=2.5),
}
for m, d in MATS.items():
    if 'tile' in d: K.TILE[m] = d['tile']

# ------------------------------------------------------------------ Masse
B = 3.60          # Achsmass
HPU = 0.33        # halbe Pilasterbreite Obergeschosse
HPG = 0.43        # halbe Pfeilerbreite Sockelzone
YB = 0.70         # Tiefe Fassadenschicht (Innenkante)
Z_BOT = -1.0      # Fundament (unter Gelaende)
Z_FL = -0.10      # Fertigfussboden EG / Schwelle (Pflaster -0.135)
Z_PL = 0.30       # Sockelstein-Oberkante
Z_GFT = 7.25      # Sturz Sockelzone (lange Fassaden)
Z_PROW = 7.54     # Unterkante Bugsturz (Foto)


def rows_for(top):
    if top > 24:
        return [(8.90, 11.80, False), (13.15, 16.10, False), (17.50, 20.05, False), (21.40, 23.95, True)]
    return [(8.90, 11.80, False), (13.15, 16.10, False), (17.50, 20.05, True)]


def cope_z(top):
    return top - 0.20, top


# ------------------------------------------------------------------ Profile
def band_prof(zb, zt, sill=True):
    P = [(0.06, zb), (YB, zb), (YB, zt)]
    if sill:
        P += [(0.10, zt), (0.035, zt - 0.012), (0.035, zt - 0.09), (0.045, zt - 0.09), (0.045, zt - 0.078),
              (0.053, zt - 0.078), (0.053, zt - 0.09), (0.06, zt - 0.09)]
        ztop = zt - 0.09
    else:
        P += [(0.06, zt)]
        ztop = zt
    if ztop - zb > 0.45:
        zm = 0.5 * (zb + ztop)
        P += [(0.06, zm + 0.012), (0.074, zm), (0.06, zm - 0.012)]
    return P


def cope_prof(cb, ct, yb=YB):
    return [(-0.045, cb), (-0.03, cb), (-0.03, cb + 0.012), (-0.018, cb + 0.012), (-0.018, cb), (yb + 0.04, cb),
            (yb + 0.04, ct - 0.035), (0.0, ct), (-0.045, ct - 0.012)]



# ------------------------------------------------------------------ Natursteinfugen
JOINT = 0.012     # Fugenbreite/-tiefe


def stack_z(p, mat, prof, z0, z1, joints, skip=(), caps=(False, False)):
    """Pfeiler/Pilaster als Plattenstapel: zwischen den Platten 12-mm-Lagerfugen (Front zurueckgesetzt)."""
    js = sorted(z for z in joints if z0 + 0.05 < z < z1 - 0.05)
    jprof = [(x, y + JOINT if y < 0.10 else y) for (x, y) in prof]
    zc = z0
    for k, zj in enumerate(js):
        ext_z(p, mat, prof, zc, zj - JOINT / 2, caps=(caps[0] if k == 0 else True, True), skip=skip)
        ext_z(p, mat, jprof, zj - JOINT / 2, zj + JOINT / 2, caps=(False, False), skip=skip)
        zc = zj + JOINT / 2
    ext_z(p, mat, prof, zc, z1, caps=(caps[0] if not js else True, caps[1]), skip=skip)


def upper_joints(top):
    rows = rows_for(top); cb, ct = cope_z(top)
    j = []
    for (zs, zh, d) in rows:
        j += [zs, zh, 0.5 * (zs + zh)]
    j += [0.5 * (rows[0][0] + Z_GFT)] + [0.5 * (rows[i][1] + rows[i + 1][0]) for i in range(len(rows) - 1)] + [0.5 * (rows[-1][1] + cb)]
    return j


GF_JOINTS = [1.45, 2.60, 3.80, 4.95, 6.10]

# ------------------------------------------------------------------ Fenster
def window(p, a, b, zs, zh, deep, openable=False):
    yf = 0.56 if deep else 0.34
    fw = 0.06
    box(p, M_FRAME, a, a + fw, yf, yf + 0.08, zs, zh, skip='xzZ')
    box(p, M_FRAME, b - fw, b, yf, yf + 0.08, zs, zh, skip='XzZ')
    box(p, M_FRAME, a + fw, b - fw, yf, yf + 0.08, zs, zs + fw, skip='xXz')
    box(p, M_FRAME, a + fw, b - fw, yf, yf + 0.08, zh - fw, zh, skip='xXZ')
    gx0, gx1, gz0, gz1 = a + fw, b - fw, zs + fw, zh - fw
    if openable:
        sw = 0.075; y0, y1 = yf + 0.012, yf + 0.072
        box(p, M_FRAME, gx0, gx0 + sw, y0, y1, gz0, gz1, skip='xzZ')
        box(p, M_FRAME, gx1 - sw, gx1, y0, y1, gz0, gz1, skip='XzZ')
        box(p, M_FRAME, gx0 + sw, gx1 - sw, y0, y1, gz0, gz0 + sw, skip='xXz')
        box(p, M_FRAME, gx0 + sw, gx1 - sw, y0, y1, gz1 - sw, gz1, skip='xXZ')
        # Griff (innen, durch das Glas sichtbar)
        hz = gz0 + 0.42 * (gz1 - gz0) if (gz1 - gz0) < 2.2 else gz0 + 1.05
        hx = gx1 - sw * 0.5
        box(p, M_STEEL, hx - 0.018, hx + 0.018, y1, y1 + 0.03, hz - 0.03, hz + 0.03, skip='y')
        box(p, M_STEEL, hx - 0.012, hx + 0.012, y1 + 0.03, y1 + 0.055, hz - 0.13, hz + 0.02, skip='')
        gx0 += sw; gx1 -= sw; gz0 += sw; gz1 -= sw
    rect_y(p, M_GLASS, gx0, gx1, gz0, gz1, yf + 0.04, out=True)
    # Fensterbank-Abdeckung (Alu) mit Tropfnase
    box(p, M_FRAME, a, b, 0.024, yf + 0.004, zs, zs + 0.014, skip='zxXY')
    box(p, M_FRAME, a, b, 0.016, 0.026, zs - 0.032, zs + 0.014, skip='Z')


# ------------------------------------------------------------------ Obergeschoss-Modul
def upper(p, W, top, infill='std3', hp=HPU, gft=Z_GFT, caps=(False, False), openable=(1,), cope_caps=(False, False)):
    cb, ct = cope_z(top)
    rows = rows_for(top)
    if hp > 0:
        Lp = [(0, 0), (hp - 0.07, 0), (hp - 0.07, 0.025), (hp - 0.01, 0.025), (hp, 0.035), (hp, YB), (0, YB)]
        stack_z(p, M_STONE, Lp, gft, cb, upper_joints(top), skip=(() if caps[0] else (6,)))
        Rp = [(W - hp, 0.035), (W - hp + 0.01, 0.025), (W - hp + 0.07, 0.025), (W - hp + 0.07, 0), (W, 0), (W, YB), (W - hp, YB)]
        stack_z(p, M_STONE, Rp, gft, cb, upper_joints(top), skip=(() if caps[1] else (4,)))
    bands = [(gft, rows[0][0], True)] + [(rows[i][1], rows[i + 1][0], True) for i in range(len(rows) - 1)] + [(rows[-1][1], cb, False)]
    for zb, zt, sill in bands:
        ext_x(p, M_STONE, band_prof(zb, zt, sill), hp, W - hp, caps=(hp == 0 and caps[0], hp == 0 and caps[1]))
    if infill == 'std3':
        wo = (W - 2 * hp - 0.60) / 3
        xs = [(hp, hp + wo), (hp + wo + 0.30, hp + 2 * wo + 0.30), (hp + 2 * wo + 0.60, W - hp)]
        mulls = [(hp + wo, hp + wo + 0.30), (hp + 2 * wo + 0.30, hp + 2 * wo + 0.60)]
    elif infill == 'pair':
        c = W / 2
        xs = [(hp, c - 0.08), (c + 0.08, W - hp)]; mulls = [(c - 0.08, c + 0.08)]
    else:   # 'joint' -- schmale Glasfuge, Metallpfosten in der Mitte
        xs = [(hp, W - hp)]; mulls = []
    for (zs, zh, deep) in rows:
        for (m0, m1) in mulls:
            ext_z(p, M_STONE, [(m0, 0.115), (m0 + 0.015, 0.10), (m1 - 0.015, 0.10), (m1, 0.115), (m1, YB), (m0, YB)],
                  zs, zh, caps=(False, False))
        for j, (a, b) in enumerate(xs):
            if infill == 'joint' and b - a > 1.6:
                c = 0.5 * (a + b)
                window(p, a, c - 0.03, zs, zh, deep, openable=False)
                window(p, c + 0.03, b, zs, zh, deep, openable=False)
                box(p, M_FRAME, c - 0.03, c + 0.03, 0.30, YB, zs, zh, skip='zZ')
            else:
                window(p, a, b, zs, zh, deep, openable=(j in openable))
    ext_x(p, M_COPE, cope_prof(cb, ct), 0, W, caps=cope_caps)


# ------------------------------------------------------------------ Sockelzone (EG + 1.OG)
def door_unit(p, x0, x1, zb, zt, handles=True, yg=0.34):
    xm = 0.5 * (x0 + x1)
    box(p, M_FRAME, x0 - 0.06, x0, 0.28, 0.40, zb, zt + 0.10, skip='z')
    box(p, M_FRAME, x1, x1 + 0.06, 0.28, 0.40, zb, zt + 0.10, skip='z')
    box(p, M_FRAME, x0, x1, 0.28, 0.40, zt, zt + 0.10)
    for (l0, l1) in ((x0, xm - 0.005), (xm + 0.005, x1)):
        box(p, M_FRAME, l0, l0 + 0.09, yg - 0.035, yg + 0.035, zb, zt, skip='z')
        box(p, M_FRAME, l1 - 0.09, l1, yg - 0.035, yg + 0.035, zb, zt, skip='z')
        box(p, M_FRAME, l0 + 0.09, l1 - 0.09, yg - 0.035, yg + 0.035, zb, zb + 0.22, skip='xXz')
        box(p, M_FRAME, l0 + 0.09, l1 - 0.09, yg - 0.035, yg + 0.035, zt - 0.10, zt, skip='xXZ')
        rect_y(p, M_GLASS, l0 + 0.09, l1 - 0.09, zb + 0.22, zt - 0.10, yg)
    if handles:
        for hx in (xm - 0.16, xm + 0.16):
            rod(p, M_STEEL, (hx, yg - 0.11, zb + 0.90), (hx, yg - 0.11, zb + 2.05), 0.017, 8)
            for hz in (zb + 1.0, zb + 1.95):
                rod(p, M_STEEL, (hx, yg - 0.11, hz), (hx, yg - 0.035, hz), 0.010, 6)
            # innen
            rod(p, M_STEEL, (hx, yg + 0.11, zb + 0.90), (hx, yg + 0.11, zb + 2.05), 0.017, 8)


def glazing(p, x0, x1, z0, z1, n=2, yg=0.34, rail=0.06):
    """Pfosten-Riegel-Verglasung in einer Ebene (Rahmen 0.28..0.40)."""
    box(p, M_FRAME, x0, x1, yg - 0.05, yg + 0.05, z0, z0 + rail, skip='xX')
    box(p, M_FRAME, x0, x1, yg - 0.05, yg + 0.05, z1 - rail, z1, skip='xX')
    w = (x1 - x0) / n
    for i in range(1, n):
        xm = x0 + i * w
        box(p, M_FRAME, xm - 0.04, xm + 0.04, yg - 0.06, yg + 0.05, z0 + rail, z1 - rail, skip='zZ')
    for i in range(n):
        a = x0 + i * w + (0.04 if i > 0 else 0); b = x0 + (i + 1) * w - (0.04 if i < n - 1 else 0)
        rect_y(p, M_GLASS, a, b, z0 + rail, z1 - rail, yg)


def canopy(p, x0, x1, z=4.16, depth=1.40, yw=0.26):
    box(p, M_CGLASS, x0 + 0.06, x1 - 0.06, -depth + 0.04, yw, z, z + 0.024, skip='Y')
    box(p, M_STEEL, x0, x1, -depth - 0.045, -depth + 0.04, z - 0.075, z + 0.05)
    for xa in (x0, x1 - 0.06):
        ext_x(p, M_STEEL, [(yw, z - 0.20), (yw, z), (-depth + 0.04, z), (-depth + 0.04, z - 0.05)], xa, xa + 0.06)
    for xr in (x0 + 0.03, x1 - 0.03):
        rod(p, M_STEEL, (xr, -depth + 0.02, z + 0.05), (xr, 0.27, z + 1.40), 0.012, 6)
        box(p, M_STEEL, xr - 0.05, xr + 0.05, 0.255, 0.28, z + 1.33, z + 1.47, skip='Y')


def gf(p, W, kind='shop', canopy_on=False, hp=HPG, gft=Z_GFT, n_up=2, n_low=2):
    if hp > 0:
        Lp = [(0, 0), (hp - 0.08, 0), (hp - 0.08, 0.025), (hp - 0.01, 0.025), (hp, 0.035), (hp, YB), (0, YB)]
        stack_z(p, M_STONE, Lp, Z_BOT, gft, GF_JOINTS, skip=(6,), caps=(False, True))
        Rp = [(W - hp, 0.035), (W - hp + 0.01, 0.025), (W - hp + 0.08, 0.025), (W - hp + 0.08, 0), (W, 0), (W, YB), (W - hp, YB)]
        stack_z(p, M_STONE, Rp, Z_BOT, gft, GF_JOINTS, skip=(4,), caps=(False, True))
        ext_z(p, M_STONE, [(0, -0.014), (hp + 0.014, -0.014), (hp + 0.014, YB), (0, YB)], Z_BOT, Z_PL, caps=(False, True), skip=(3,))
        ext_z(p, M_STONE, [(W - hp - 0.014, -0.014), (W, -0.014), (W, YB), (W - hp - 0.014, YB)], Z_BOT, Z_PL, caps=(False, True), skip=(1,))
    a, b = hp, W - hp; c = 0.5 * W
    box(p, M_GRAN, a, b, -0.03, YB, Z_FL - 0.08, Z_FL, skip='xXzY')
    box(p, M_FRAME, a, a + 0.08, 0.28, 0.40, Z_FL, gft, skip='x')
    box(p, M_FRAME, b - 0.08, b, 0.28, 0.40, Z_FL, gft, skip='X')
    box(p, M_FRAME, a + 0.08, b - 0.08, 0.28, 0.40, gft - 0.10, gft, skip='xXZ')
    ia, ib = a + 0.08, b - 0.08
    # Kaempferband (Geschossdecke 1.OG) mit Schattenfuge
    box(p, M_FRAME, ia, ib, 0.25, 0.40, 3.80, 4.18, skip='xX')
    box(p, M_FRAME, ia, ib, 0.25, 0.40, 4.21, 4.60, skip='xX')
    rect_y(p, M_STEELD, ia, ib, 4.18, 4.21, 0.33)
    if kind == 'shop':
        box(p, M_FRAME, ia, ib, 0.30, 0.38, Z_FL, 0.30, skip='xXz')
        glazing(p, ia, ib, 0.30, 3.80, n=n_low)
    else:
        dw = 1.0 if (ib - ia) > 2.4 else 0.8
        d0, d1 = c - dw, c + dw
        zt = 2.55
        door_unit(p, d0, d1, Z_FL, zt, handles=(kind == 'door'))
        if d0 - 0.06 - ia > 0.12:
            box(p, M_FRAME, ia, d0 - 0.06, 0.30, 0.38, Z_FL, 0.30, skip='xXz')
            glazing(p, ia, d0 - 0.06, 0.30, 3.80, n=1)
            box(p, M_FRAME, d1 + 0.06, ib, 0.30, 0.38, Z_FL, 0.30, skip='xXz')
            glazing(p, d1 + 0.06, ib, 0.30, 3.80, n=1)
        glazing(p, d0 - 0.06, d1 + 0.06, zt + 0.10, 3.80, n=2)
    glazing(p, ia, ib, 4.60, gft - 0.10, n=n_up)
    if canopy_on:
        canopy(p, a + 0.05, b - 0.05)


def end_pier(p, x0, x1, top, skip_sides=True):
    cb, ct = cope_z(top)
    sk = (1, 3) if skip_sides else ()
    ext_z(p, M_STONE, [(x0, 0), (x1, 0), (x1, YB), (x0, YB)], Z_BOT, cb, caps=(False, False), skip=sk)
    ext_z(p, M_STONE, [(x0, -0.014), (x1, -0.014), (x1, YB), (x0, YB)], Z_BOT, Z_PL, caps=(False, True), skip=sk)
    # durchlaufende Lagerfugen in Bandhoehe
    ext_x(p, M_COPE, cope_prof(cb, ct), x0, x1, caps=(False, False))


def plain_wall(p, L, top, caps=(False, False)):
    """glatte Rundung (kleine Radien): Naturstein mit Fugenlinien in Bandhoehe."""
    cb, ct = cope_z(top)
    rows = rows_for(top)
    levels = [Z_GFT] + [r[0] for r in rows] + [r[1] for r in rows]
    P = [(0, Z_PL), (YB, Z_PL), (YB, cb), (0, cb)]
    # Fugen (V) von oben nach unten
    for z in sorted(levels, reverse=True):
        P += [(0, z + 0.012), (0.015, z), (0, z - 0.012)]
    ext_x(p, M_STONE, P, 0, L, caps=caps)
    ext_x(p, M_STONE, [(-0.014, Z_BOT), (YB, Z_BOT), (YB, Z_PL), (-0.014, Z_PL)], 0, L, caps=caps)
    ext_x(p, M_COPE, cope_prof(cb, ct), 0, L, caps=caps)


# ------------------------------------------------------------------ Plan
segs, info, LINES = kc_plan.build_plan()
FP_PTS = kc_plan.sample_path(segs, da=math.radians(2.0))
FP = orient(Polygon(FP_PTS).buffer(0), 1.0)
print('Grundflaeche %.1f m2 (LoD2 4368.4)' % FP.area)


def lod2_poly(z, idx=0):
    r = kc_plan.roof_at(z)[idx]
    return orient(Polygon(r[0], r[1]).buffer(0), 1.0)


R0 = lod2_poly(28.64)          # Technik-Einhausung
R3 = Polygon(kc_plan.roof_at(26.06)[0][0]).buffer(0)    # Staffelgeschoss (ohne Loch)
R2 = lod2_poly(22.22)          # Terrasse SW-Fluegel
R7 = lod2_poly(22.18)          # Hauptdach (mit Lichtschacht-Loch)
R1 = [orient(Polygon(r[0]).buffer(0), 1.0) for r in kc_plan.roof_at(17.79) if Polygon(r[0]).area > 300][0]   # Hof (Dachgarten 4.OG)
R4 = [orient(Polygon(r[0]).buffer(0), 1.0) for r in kc_plan.roof_at(17.79) if Polygon(r[0]).area < 300][0]   # Lichtschacht
CANOPY_R5 = Polygon(kc_plan.roof_at(5.71)[0][0])
CANOPY_R6 = Polygon(kc_plan.roof_at(5.43)[0][0])
print('R1 %.0f m2, R4 %.0f m2, R3 %.0f m2, R0 %.0f m2' % (R1.area, R4.area, R3.area, R0.area))

U = Piece()                         # Unikate (Weltkoordinaten)
MODULES = {}                        # Name -> Piece (lokal)
INST = []                           # (modul, matrix 4x4)
TEXTS = []                          # Schriftzuege fuer Blender
COL = Piece()                       # Kollision


def add_world(piece_local, fn):
    U.extend(piece_local, fn)


def layout_line(L):
    best = None
    for n in range(max(1, int(L / B) - 2), int(L / B) + 3):
        for a in np.linspace(0.30, 1.20, 181):
            k = (L - 2 * a) / (n * B)
            if not (0.955 <= k <= 1.045): continue
            cost = abs(k - 1) * 12 + abs(a - 0.45)
            if best is None or cost < best[0]: best = (cost, n, k, a)
    if best is None: raise RuntimeError('kein Raster fuer L=%.2f' % L)
    return best[1:]


def mat_line(p0, t, x, k=1.0):
    t = np.asarray(t, float); l = np.array([-t[1], t[0]])
    o = np.asarray(p0, float) + t * x
    M = np.eye(4)
    M[:2, 0] = t * k; M[:2, 1] = l; M[0, 3] = o[0]; M[1, 3] = o[1]
    return M


def module(name, fn):
    if name not in MODULES:
        p = Piece(); fn(p); MODULES[name] = p
    return name


# ------------------------------------------------------------------ Fassaden je Abschnitt
GF_PLAN = {
    # name: dict(canopy=(x0,x1) Bereich, main=[(x0,x1)], doors=[bay idx], flip=True wenn x vom Bug weg laeuft)
}
SIGNS = []        # (seg, x, kind)


def std_segment(seg):
    name = seg['name']; L = seg['L']; top = seg['top']
    n, k, a = layout_line(L)
    bw = B * k
    seg['layout'] = (n, k, a)
    fn = K.seg_map(seg)
    K.set_dxmax(None)
    p = Piece()
    end_pier(p, 0, a, top); end_pier(p, L - a, L, top)
    add_world(p, fn)
    axes = [a + i * bw for i in range(n + 1)]
    seg['axes'] = axes
    # Erdgeschoss-Nutzung je Feld
    kinds = []
    for i in range(n):
        xc = a + (i + 0.5) * bw
        s_prow = L - xc if name == 'NE' else (xc if name == 'W_nord' else None)
        kind = 'shop'; can = False
        if name == 'NE':
            can = 1.5 < s_prow < 80.5
            if 61.0 < s_prow < 68.0: kind = 'main'; can = False
            elif i % 6 == 2: kind = 'door'
        elif name == 'W_nord':
            can = 3.5 < s_prow < 40.5
            if 15.0 < s_prow < 25.4: kind = 'main'; can = False
            elif i % 4 == 1 or i in (11, 15): kind = 'door'
        elif name == 'S_ost':
            s_e = L - xc
            can = 3.5 < s_e < 26.5
            if i in (1, 4): kind = 'door'
        elif name == 'W_sued':
            if i in (2, 6, 10): kind = 'door'
        elif name == 'S_west':
            if i in (2, 6): kind = 'door'
        elif name == 'S_mitte':
            if i == 1: kind = 'door'
        kinds.append((kind, can))
    seg['kinds'] = kinds; seg['bw'] = bw
    up = module('UP%d' % round(top), lambda q: upper(q, B, top, 'std3'))
    for i in range(n):
        kind, can = kinds[i]
        gname = module('GF_%s%s' % (kind, '_c' if can else ''), lambda q, kind=kind, can=can: gf(q, B, kind, can))
        M = mat_line(seg['p0'], seg['t'], a + i * bw, k)
        INST.append((gname, M)); INST.append((up, M))
    print('  %-8s L=%6.2f  %2d Felder a %.3f m (k=%.3f) Endpfeiler %.2f' % (name, L, n, bw, k, a))
    return kinds


def curve_segment(seg, kind='shop'):
    A = seg['L']; m = max(1, int(round(A / 2.9))); W = A / m
    K.set_dxmax(0.20)
    for i in range(m):
        p = Piece()
        gf(p, W, 'shop', False, n_up=2, n_low=2)
        upper(p, W, seg['top'], 'pair', openable=(0,))
        add_world(p, K.map_arc(seg['c'], seg['r'], seg['a0'], x_off=i * W))
    K.set_dxmax(None)
    seg['bays'] = (m, W)


def round_segment(seg):
    K.set_dxmax(0.15)
    p = Piece(); plain_wall(p, seg['L'], seg['top'])
    add_world(p, K.seg_map(seg))
    K.set_dxmax(None)


def joint_segment(seg, kind='shop', caps=(False, False)):
    W = seg['L']
    p = Piece()
    gf(p, W, kind, False, hp=0.12, n_up=1 if W < 2.0 else 2, n_low=1 if W < 2.0 else 2)
    upper(p, W, seg['top'], 'joint', hp=0.12, caps=caps, cope_caps=caps)
    add_world(p, K.seg_map(seg))


def prow_segment(seg):
    """Bug zur Kroepcke-Uhr: drei gekruemmte Achsen, Kolonnade mit freistehenden Stuetzen, Glasfassade zurueckgesetzt."""
    A = seg['L']; m = 3; W = A / m; top = seg['top']
    K.set_dxmax(0.18)
    fn = K.seg_map(seg)
    p = Piece()
    YG = 1.40                       # Glasebene Kolonnade
    for i in range(m):
        q = Piece()
        upper(q, W, top, 'pair', gft=Z_PROW + 0.16, openable=(0,))
        p.extend(q, lambda V, i=i: V + np.array([i * W, 0, 0]))
    # Sturzband/Untersicht der Kolonnade
    box(p, M_STONE, 0.0, A, 0.0, YG + 0.06, Z_PROW, Z_PROW + 0.16, skip='xX')
    # Stuetzen: zwei freistehende in den Zwischenachsen (Foto 2026), Endachsen als tiefe Wandpfeiler
    cols = [W, 2 * W]
    EP = 0.80
    dep = YG + 0.06
    Lp = [(0, 0), (EP - 0.08, 0), (EP - 0.08, 0.025), (EP - 0.01, 0.025), (EP, 0.035), (EP, dep), (0, dep)]
    Rp = [(A - EP, 0.035), (A - EP + 0.01, 0.025), (A - EP + 0.08, 0.025), (A - EP + 0.08, 0), (A, 0), (A, dep), (A - EP, dep)]
    stack_z(p, M_STONE, Lp, Z_BOT, Z_PROW, GF_JOINTS + [Z_PL], skip=(6,))
    stack_z(p, M_STONE, Rp, Z_BOT, Z_PROW, GF_JOINTS + [Z_PL], skip=(4,))
    ext_z(p, M_STONE, [(0, -0.014), (EP + 0.014, -0.014), (EP + 0.014, dep), (0, dep)], Z_BOT, Z_PL, caps=(False, True), skip=(3,))
    ext_z(p, M_STONE, [(A - EP - 0.014, -0.014), (A, -0.014), (A, dep), (A - EP - 0.014, dep)], Z_BOT, Z_PL, caps=(False, True), skip=(1,))
    for xc in cols:
        hw, d = 0.40, 0.85
        prof = [(xc - hw + 0.02, 0), (xc + hw - 0.02, 0), (xc + hw, 0.02), (xc + hw, d - 0.02), (xc + hw - 0.02, d),
                (xc - hw + 0.02, d), (xc - hw, d - 0.02), (xc - hw, 0.02)]
        stack_z(p, M_STONE, prof, Z_BOT, Z_PROW, GF_JOINTS + [Z_PL], caps=(False, False))
        ext_z(p, M_STONE, [(xc - hw - 0.014, -0.014), (xc + hw + 0.014, -0.014), (xc + hw + 0.014, d + 0.014), (xc - hw - 0.014, d + 0.014)],
              Z_BOT, Z_PL, caps=(False, True))
    # Boden der Kolonnade (Granit)
    box(p, M_GRAN, EP, A - EP, -0.02, YG + 0.06, Z_FL - 0.08, Z_FL, skip='xXzY')
    # zurueckgesetzte Glasfassade (gebogen), Tuer in der Mittelachse
    xa, xb = EP, A - EP
    dc = 1.5 * W
    p2 = Piece()
    # Sockelblech + Riegel
    box(p2, M_FRAME, xa, xb, 0.30 - 0.04, 0.30 + 0.06, Z_FL, 0.20, skip='xXz')
    for (x0, x1) in ((xa, dc - 1.08), (dc + 1.08, xb)):
        glazing(p2, x0, x1, 0.20, 3.85, n=3, yg=0.34)
    door_unit(p2, dc - 1.0, dc + 1.0, Z_FL, 2.60, handles=True)
    glazing(p2, dc - 1.08, dc + 1.08, 2.70, 3.85, n=2, yg=0.34)
    box(p2, M_FRAME, xa, xb, 0.24, 0.42, 3.85, 4.15, skip='xX')
    glazing(p2, xa, xb, 4.15, Z_PROW, n=6, yg=0.34)
    box(p2, M_FRAME, xa, xa + 0.08, 0.26, 0.42, Z_FL, Z_PROW, skip='x')
    box(p2, M_FRAME, xb - 0.08, xb, 0.26, 0.42, Z_FL, Z_PROW, skip='X')
    p.extend(p2, lambda V: V + np.array([0, YG - 0.34, 0]))
    add_world(p, fn)
    K.set_dxmax(None)
    seg['cols'] = cols; seg['YG'] = YG; seg['bays'] = (m, W)


print('Fassaden:')
SEG = {}
for s in segs:
    if s.get('name'): SEG[s['name']] = s
for i, s in enumerate(segs):
    if s['kind'] == 'std': std_segment(s)
    elif s['kind'] == 'curve': curve_segment(s)
    elif s['kind'] == 'round': round_segment(s)
    elif s['kind'] == 'prow': prow_segment(s)
    elif s['kind'] == 'joint':
        prev, nxt = segs[i - 1], segs[(i + 1) % len(segs)]
        caps = (prev['top'] < s['top'] - 0.5 and False, False)
        joint_segment(s, kind=('door' if s['L'] > 2.4 and s['top'] > 24 else 'shop'))
print('  Unikate bisher %d Dreiecke' % U.tris())

# ------------------------------------------------------------------ Hauptvordaecher R5 / R6
def main_canopy(poly, ztop, seg):
    """Kragdach am Haupteingang: 0.35 m starke Platte, Untersicht hell mit Downlights."""
    P = orient(poly.buffer(0), 1.0)
    # an die Fassadenflucht anschliessen (nicht in den Stein)
    fn_seg = seg
    zb = ztop - 0.36
    slab(U, M_FRAME, P, zb, ztop, cap_bottom=False)
    # Untersicht (hell) + Downlights
    pts, tri = K.triangulate([np.array(P.exterior.coords)[:-1]])
    emit(U, M_ICEIL, np.c_[pts, np.full(len(pts), zb)], tri[:, [0, 2, 1]])
    c = np.array(P.centroid.coords[0])
    t = np.asarray(seg['t'])
    # Schriftzug auf der Stirnseite (Kante mit groesstem Abstand zur Fassade)
    ring = np.array(P.exterior.coords)[:-1]; out = np.array([t[1], -t[0]])
    best = None
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        m = 0.5 * (a + b); dd = float(np.dot(m - np.asarray(seg['p0']), out))
        if best is None or dd > best[0]: best = (dd, m, np.linalg.norm(b - a))
    _, m, Le = best
    TEXTS.append(dict(text='Peek & Cloppenburg', pos=[*(m + out * 0.004), ztop - 0.18], normal=[out[0], out[1], 0.0],
                      dir=[-out[1], out[0], 0.0], size=0.24, extrude=0.008, mat=M_SWHITE, align='CENTER', fit=min(Le * 0.75, 4.2), fit_max=True))
    for d in np.arange(-4.0, 4.01, 1.6):
        for e in (-1.6, -0.6):
            q = c + t * d + np.array([-t[1], t[0]]) * (e + 1.1)
            if P.buffer(-0.3).contains(Point(q)):
                lathe(U, M_CLIGHT, (q[0], q[1], zb - 0.012), [(0.09, 0), (0.09, 0.012)], n=10)


def slab(piece, mat, poly, z0, z1, cap_bottom=True, cap_top=True, sides=True):
    polys = [poly] if isinstance(poly, Polygon) else list(getattr(poly, 'geoms', []))
    for P in polys:
        if P.is_empty or P.area < 1e-4: continue
        P = orient(P, 1.0)
        rings = [np.array(P.exterior.coords)[:-1]] + [np.array(r.coords)[:-1] for r in P.interiors]
        pts, tri = K.triangulate(rings)
        if len(tri):
            if cap_top: emit(piece, mat, np.c_[pts, np.full(len(pts), z1)], tri)
            if cap_bottom: emit(piece, mat, np.c_[pts, np.full(len(pts), z0)], tri[:, [0, 2, 1]])
        if sides and z1 - z0 > 1e-4:
            for r in rings:
                n = len(r); V = []; F = []
                for i in range(n):
                    a, b = r[i], r[(i + 1) % n]
                    if np.allclose(a, b): continue
                    o = len(V)
                    V += [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
                    F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
                emit(piece, mat, V, F)


main_canopy(CANOPY_R5, 5.71, SEG['W_nord'])
main_canopy(CANOPY_R6, 5.43, SEG['NE'])

# ------------------------------------------------------------------ Kolonnaden-Bereich (fuer Innenraum/Kollision)
prow = [s for s in segs if s['kind'] == 'prow'][0]
_pc = Piece(); K.set_dxmax(0.2)
box(_pc, 'x', 0.80, prow['L'] - 0.80, -0.5, prow['YG'] + 0.10, 0, 1)
_v = list(_pc.arrays().values())[0][0]
COLONNADE = Polygon(K.seg_map(prow)(np.array(_v))[:, :2]).convex_hull
_ = None
K.set_dxmax(None)
# exakte Kolonnadenflaeche: Bogenband zwischen Radius r+0.5 und r-YG-0.1
def arc_band(seg, y0, y1, x0, x1, n=40):
    xs = np.linspace(x0, x1, n)
    fn = K.seg_map(seg)
    outer = fn(np.c_[xs, np.full(n, y0), np.zeros(n)])[:, :2]
    inner = fn(np.c_[xs[::-1], np.full(n, y1), np.zeros(n)])[:, :2]
    return Polygon(np.vstack([outer, inner])).buffer(0)
COLONNADE = arc_band(prow, -0.5, prow['YG'] + 0.10, 0.80, prow['L'] - 0.80)

# ------------------------------------------------------------------ Innenraeume
def interior_floor(region, depth, ffl, ceil, lights=True, edge=True):
    outer = region
    core = region.buffer(-depth, join_style=2)
    ring = outer.difference(core)
    if ring.is_empty: return
    slab(U, M_IFLOOR, ring, ffl - 0.02, ffl, cap_bottom=False, sides=False)
    slab(U, M_ICEIL, ring, ceil, ceil + 0.02, cap_top=False, sides=False)
    # Rueckwand (Kern) -- nach aussen gerichtet
    cores = [core] if isinstance(core, Polygon) else list(getattr(core, 'geoms', []))
    for c in cores:
        if c.area < 1: continue
        slab(U, M_IWALL, c, ffl, ceil, cap_top=False, cap_bottom=False)
    # Deckenleuchten im 2.4-m-Raster
    if lights:
        minx, miny, maxx, maxy = ring.bounds
        sp = 2.4
        inner_ok = ring.buffer(-0.5)
        for x in np.arange(minx + 1.2, maxx, sp):
            for y in np.arange(miny + 1.2, maxy, sp):
                if inner_ok.contains(Point(x, y)):
                    rect_z(U, M_ILIGHT, x - 0.3, x + 0.3, y - 0.3, y + 0.3, ceil - 0.006, up=False)


REG_FP = FP.buffer(-YB, join_style=2)
REG_GF = REG_FP.difference(COLONNADE.buffer(0.05))
interior_floor(REG_GF, 7.0, Z_FL, 3.78)
interior_floor(REG_GF, 5.5, 4.25, 7.10)
interior_floor(REG_FP, 5.0, 8.00, 11.78)
interior_floor(REG_FP, 5.0, 12.30, 16.08)
WALL_T = 0.40
REG_4 = REG_FP.difference(R1.buffer(WALL_T + 0.05, join_style=2)).difference(R4.buffer(WALL_T + 0.05, join_style=2))
interior_floor(REG_4, 5.0, 17.30, 20.03)
REG_5 = R3.buffer(-WALL_T, join_style=2).intersection(FP.buffer(-0.05))
interior_floor(REG_5, 4.5, 21.40, 23.93)
# Deckenrand-Schluss (Sturzbereiche zwischen Decke und naechstem Boden), aussen um REG
for (zc, zf) in ((3.78, 4.25), (7.10, 8.00), (11.78, 12.30), (16.08, 17.30), (20.03, 21.30)):
    reg = REG_GF if zc < 4 else (REG_4 if zc > 16 else REG_FP)
    polys = [reg] if isinstance(reg, Polygon) else list(reg.geoms)
    for P in polys:
        P = orient(P, 1.0)
        for r in [np.array(P.exterior.coords)[:-1]] + [np.array(q.coords)[:-1] for q in P.interiors]:
            n = len(r); V = []; F = []
            for i in range(n):
                a, b = r[i], r[(i + 1) % n]
                o = len(V); V += [(a[0], a[1], zc), (b[0], b[1], zc), (b[0], b[1], zf), (a[0], a[1], zf)]
                F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
            emit(U, M_IWALL, V, F)
print('  + Innenraeume: %d' % U.tris())


# Ladenmoebel EG (Kleiderstaender, Figurinen, Tische) entlang 2.6 m hinter der Glasebene
def shop_fittings():
    line = REG_GF.buffer(-2.2, join_style=2)
    lines = [line] if isinstance(line, Polygon) else list(line.geoms)
    k = 0
    for P in lines:
        ring = LineString(P.exterior.coords)
        L = ring.length
        for s in np.arange(1.0, L, 2.9):
            q = np.array(ring.interpolate(s).coords[0]); q2 = np.array(ring.interpolate(min(L, s + 0.3)).coords[0])
            t = q2 - q; t /= (np.linalg.norm(t) + 1e-9); nrm = np.array([-t[1], t[0]])
            typ = k % 4; k += 1
            if typ in (0, 2):        # Kleiderstange mit Ware
                for e in (-0.6, 0.6):
                    b = q + t * e
                    rod(U, M_RACK, (b[0], b[1], Z_FL), (b[0], b[1], Z_FL + 1.55), 0.015, 6, caps=False)
                rod(U, M_RACK, (*(q - t * 0.62), Z_FL + 1.52), (*(q + t * 0.62), Z_FL + 1.52), 0.012, 6)
                for j in range(6):
                    c = q + t * (-0.5 + j * 0.2)
                    mat = (M_TEX_A, M_TEX_B, M_TEX_C, M_TEX_A, M_TEX_B, M_TEX_A)[(j + k) % 6]
                    pc = Piece(); box(pc, mat, -0.06, 0.06, -0.24, 0.24, 0.55, 1.45)
                    M = np.array([[t[0], nrm[0], 0], [t[1], nrm[1], 0], [0, 0, 1]])
                    U.extend(pc, lambda V, M=M, c=c: V @ M.T + np.array([c[0], c[1], Z_FL]))
            elif typ == 1:           # zwei Figurinen
                for e in (-0.45, 0.45):
                    b = q + t * e
                    pc = Piece()
                    lathe(pc, M_STEEL, (0, 0, 0), [(0.20, 0), (0.20, 0.02), (0.015, 0.02), (0.015, 0.80)], n=12)
                    # Torso (elliptisch: Schulterbreite quer zur Blickrichtung)
                    q2 = Piece()
                    lathe(q2, M_MANN, (0, 0, 0), [(0.0, 0.78), (0.15, 0.80), (0.17, 0.92), (0.135, 1.08), (0.15, 1.22), (0.19, 1.34), (0.20, 1.40),
                                                 (0.16, 1.45), (0.055, 1.48), (0.05, 1.56), (0.0, 1.565)], n=12)
                    pc.extend(q2, lambda V: V * np.array([1.0, 0.62, 1.0]))
                    uvsphere(pc, M_MANN, (0, 0, 1.665), 0.085, 0.095, 0.11, nu=10, nv=6)
                    M = np.array([[t[0], nrm[0], 0], [t[1], nrm[1], 0], [0, 0, 1]])
                    U.extend(pc, lambda V, M=M, b=b: V @ M.T + np.array([b[0], b[1], Z_FL]))
            else:                    # Praesentationstisch
                pc = Piece(); box(pc, M_IFLOOR, -0.7, 0.7, -0.45, 0.45, 0.0, 0.78)
                box(pc, M_TEX_B, -0.55, -0.1, -0.3, 0.3, 0.78, 0.86); box(pc, M_TEX_A, 0.1, 0.55, -0.3, 0.3, 0.78, 0.84)
                M = np.array([[t[0], nrm[0], 0], [t[1], nrm[1], 0], [0, 0, 1]])
                U.extend(pc, lambda V, M=M, q=q: V @ M.T + np.array([q[0], q[1], Z_FL]))


shop_fittings()
print('  + Ladenmoebel: %d' % U.tris())


# ------------------------------------------------------------------ Ring-Fassaden (Hof, Lichtschacht, Staffelgeschoss)
def boundary_lines(solid, void, restrict=None):
    """Kanten von solid, die an void grenzen (als Liste von Polylinien, Laufrichtung CCW um solid)."""
    out = []
    S = orient(solid, 1.0).segmentize(0.4)
    rings = [np.array(S.exterior.coords)] + [np.array(r.coords) for r in S.interiors]
    vb = void.buffer(0.03)
    for r in rings:
        keep = []
        for i in range(len(r) - 1):
            a, b = r[i], r[i + 1]
            mid = Point(0.5 * (a + b))
            ok = vb.contains(mid) and (restrict is None or restrict.contains(mid))
            keep.append(ok)
        # zusammenhaengende Laeufe
        n = len(keep)
        if all(keep):
            out.append(r.copy()); continue
        if not any(keep): continue
        start = keep.index(False)
        cur = []
        for j in range(1, n + 1):
            i = (start + j) % n
            if keep[i]:
                if not cur: cur = [r[i]]
                cur.append(r[i + 1])
            else:
                if cur: out.append(np.array(cur)); cur = []
        if cur: out.append(np.array(cur))
    return out


def simplify_line(pl, tol=0.05):
    ls = LineString(pl).simplify(tol, preserve_topology=False)
    return np.array(ls.coords)


def ring_facade(solid, void, z0, top, rows, restrict=None, t=WALL_T, post_sp=1.65, railing=False, coping=True, name=''):
    lines = [simplify_line(l) for l in boundary_lines(solid, void, restrict)]
    if not lines: return
    wall_all = solid.difference(solid.buffer(-t, join_style=2))
    near = unary_union([LineString(l).buffer(t + 0.03, cap_style=2, join_style=2) for l in lines])
    wall = wall_all.intersection(near)
    cb, ct = cope_z(top)
    bands = [(z0, rows[0][0])] + [(rows[i][1], rows[i + 1][0]) for i in range(len(rows) - 1)] + [(rows[-1][1], cb)]
    for zb, zt in bands:
        slab(U, M_STONE, wall, zb, zt)
        # Fensterbank-Abdeckung als duenne Platte vor der Brustung: weggelassen (Ringfassade schlicht)
    if coping:
        cop = wall.union(unary_union([LineString(l).buffer(0.045, cap_style=2, join_style=2) for l in lines]).difference(solid.buffer(-t - 0.04, join_style=2)))
        cop = cop.intersection(near.buffer(0.05))
        slab(U, M_COPE, cop, cb, ct)
    # Pfosten + Glas je Fensterband
    posts = []
    for l in lines:
        ls = LineString(l); L = ls.length
        n = max(1, int(round(L / post_sp)))
        for j in range(n + 1):
            d = L * j / n
            a = ls.interpolate(max(0, d - 0.16)); b = ls.interpolate(min(L, d + 0.16))
            seg = LineString([a, b])
            if seg.length < 0.05: continue
            posts.append(seg.buffer(t + 0.05, cap_style=2, single_sided=False).intersection(wall))
        for v in l[1:-1]:
            posts.append(Point(v).buffer(0.32, cap_style=3).intersection(wall))
    postu = unary_union(posts)
    for (zs, zh) in rows:
        slab(U, M_STONE, postu, zs, zh, cap_bottom=False, cap_top=False)
        # Glas und Rahmen je gerader Kante, auf halber Wandtiefe
        for l in lines:
            for i in range(len(l) - 1):
                a, b = np.array(l[i]), np.array(l[i + 1]); d = b - a; Ls = np.linalg.norm(d)
                if Ls < 0.4: continue
                tt = d / Ls; inward = np.array([-tt[1], tt[0]])
                if not solid.buffer(0.01).contains(Point(0.5 * (a + b) + inward * 0.1)):
                    a, b = b, a; tt = -tt; inward = -inward
                q = Piece()
                yg = t * 0.55
                rect_y(q, M_GLASS, 0.0, Ls, zs + 0.06, zh - 0.06, yg, out=True)
                box(q, M_FRAME, 0.0, Ls, yg - 0.04, yg + 0.04, zs, zs + 0.06, skip='xXz')
                box(q, M_FRAME, 0.0, Ls, yg - 0.04, yg + 0.04, zh - 0.06, zh, skip='xXZ')
                U.extend(q, lambda V, a=a, tt=tt, inward=inward: np.c_[a[0] + V[:, 0] * tt[0] + V[:, 1] * inward[0],
                                                                      a[1] + V[:, 0] * tt[1] + V[:, 1] * inward[1], V[:, 2]])
    if railing:
        for l in lines:
            ls = LineString(l)
            off = ls.parallel_offset(t * 0.5, 'left', join_style=2)
            # Seite pruefen: Gelaender ueber der Wandmitte
            cand = [off, ls.parallel_offset(t * 0.5, 'right', join_style=2)]
            off = max(cand, key=lambda g: g.intersection(solid).length if not g.is_empty else -1)
            geoms = [off] if isinstance(off, LineString) else list(getattr(off, 'geoms', []))
            for g in geoms:
                c = np.array(g.coords)
                if len(c) < 2: continue
                Lg = g.length
                for d in np.arange(0.1, Lg, 1.5):
                    pt = np.array(g.interpolate(d).coords[0])
                    rod(U, M_STEEL, (pt[0], pt[1], ct), (pt[0], pt[1], ct + 0.92), 0.02, 6, caps=False)
                pt = np.array(g.interpolate(Lg - 0.05).coords[0])
                rod(U, M_STEEL, (pt[0], pt[1], ct), (pt[0], pt[1], ct + 0.92), 0.02, 6, caps=False)
                for i in range(len(c) - 1):
                    rod(U, M_STEEL, (*c[i], ct + 0.92), (*c[i + 1], ct + 0.92), 0.024, 6)
                    rod(U, M_STEEL, (*c[i], ct + 0.50), (*c[i + 1], ct + 0.50), 0.012, 6)


# Hof (R1) und Lichtschacht (R4): Waende ausserhalb der Loecher, 4.OG verglast
VOID_R1 = R1; VOID_R4 = R4
SOLID_N = FP.difference(R3)          # Nordteil-Dachgeschoss
ring_facade(FP.difference(R1), R1, 17.79, 22.20, [(18.00, 20.05)], restrict=R1.buffer(0.1).difference(R3.buffer(0.05)), railing=True, name='hof')
ring_facade(FP.difference(R4), R4, 17.79, 22.20, [(18.00, 20.05)], railing=True, name='schacht')
# Staffelgeschoss (R3): Kanten zum Hof ab 17.79, uebrige ab Dachniveau; buendige Teile (Mittelteil, Fuge West) ausgenommen
flush = []
for s in segs:
    if s['top'] > 24:
        fn = K.seg_map(s); n = 40
        xs = np.linspace(0, s['L'], n)
        a = fn(np.c_[xs, np.full(n, -0.2), np.zeros(n)])[:, :2]; b = fn(np.c_[xs[::-1], np.full(n, YB + 0.6), np.zeros(n)])[:, :2]
        flush.append(Polygon(np.vstack([a, b])).buffer(0.02))
FLUSH = unary_union(flush)
OUTSIDE_R3 = sbox(-200, -300, 200, 100).difference(R3)
ring_facade(R3, OUTSIDE_R3, 21.30, 26.06, [(21.60, 23.95)], restrict=sbox(-200, -300, 200, 100).difference(FLUSH).difference(R1.buffer(0.3)), name='staffel')
ring_facade(R3, OUTSIDE_R3, 17.79, 26.06, [(18.00, 20.05), (21.60, 23.95)], restrict=R1.buffer(0.3).difference(FLUSH), name='staffel_hof')
print('  + Ringfassaden: %d' % U.tris())

# ------------------------------------------------------------------ Daecher
def fp_inner(d):
    return FP.buffer(-d, join_style=2)


ROOF_Z = 21.35
roof_all = fp_inner(YB).difference(R3).difference(R1.buffer(WALL_T, join_style=2)).difference(R4.buffer(WALL_T, join_style=2)).buffer(0)
roofT = roof_all.intersection(R2.buffer(0.6, join_style=2)).buffer(0)
roofN = roof_all.difference(roofT).buffer(0)
print('Dachflaechen: Nord %.0f m2, Terrasse %.0f m2 (R7 LoD2 %.0f)' % (roofN.area, roofT.area, R7.area))
# Wartungsweg um den Hof (DOP: helle Doppellinie)
path = R1.buffer(2.3, join_style=2).difference(R1.buffer(1.1, join_style=2)).intersection(roofN.buffer(-0.6))
roof_box = Polygon([(-9.4, -53.0), (-2.6, -53.0), (-2.6, -61.0), (-9.4, -61.0)])   # Treppen-/Aufzugsaustritt (DOP)
# an NE-Flucht ausrichten
tNE = np.asarray(SEG['NE']['t']); cbx = np.array([-6.0, -57.0])
ux = -tNE; uy = np.array([-ux[1], ux[0]])
roof_box = Polygon([cbx + ux * sx + uy * sy for sx, sy in ((-2.4, -1.8), (2.4, -1.8), (2.4, 1.8), (-2.4, 1.8))])
sedum = roofN.buffer(-0.5, join_style=2).difference(path.buffer(0.15)).difference(roof_box.buffer(0.5))
gravel = roofN.difference(sedum).difference(path).difference(roof_box)
slab(U, M_MEMB, roofN, ROOF_Z - 0.06, ROOF_Z - 0.01, cap_bottom=False, sides=False)
slab(U, M_SEDUM, sedum, ROOF_Z, ROOF_Z + 0.10, cap_bottom=False)
slab(U, M_GRAVEL, gravel, ROOF_Z, ROOF_Z + 0.06, cap_bottom=False)
slab(U, M_PAVER, path, ROOF_Z, ROOF_Z + 0.12, cap_bottom=False)
# Terrasse SW (R2) -- Platten
slab(U, M_MEMB, roofT, ROOF_Z - 0.06, ROOF_Z - 0.01, cap_bottom=False, sides=False)
slab(U, M_PAVER, roofT.buffer(-0.3, join_style=2), ROOF_Z, ROOF_Z + 0.10, cap_bottom=False)
slab(U, M_GRAVEL, roofT.difference(roofT.buffer(-0.3, join_style=2)), ROOF_Z, ROOF_Z + 0.06, cap_bottom=False)
# Gelaender auf der Terrassen-Attika
def rail_on_line(pl, z, h=0.95, sp=1.5):
    ls = LineString(pl); L = ls.length
    if L < 0.5: return
    for d in list(np.arange(0.15, L - 0.1, sp)) + [L - 0.1]:
        q = np.array(ls.interpolate(d).coords[0]); rod(U, M_STEEL, (q[0], q[1], z), (q[0], q[1], z + h), 0.02, 6, caps=False)
    c = np.array(ls.coords)
    for i in range(len(c) - 1):
        rod(U, M_STEEL, (*c[i], z + h), (*c[i + 1], z + h), 0.024, 6)
        rod(U, M_STEEL, (*c[i], z + 0.45), (*c[i + 1], z + 0.45), 0.012, 6)


fpline = FP.exterior.parallel_offset(YB * 0.5, 'left', join_style=2) if False else None
terr_edge = boundary_lines(R2.intersection(FP.buffer(-0.01)), sbox(-200, -300, 200, 100).difference(FP), None)
for l in terr_edge:
    ls = LineString(l).parallel_offset(0.30, 'left', join_style=2)
    for g in ([ls] if isinstance(ls, LineString) else list(getattr(ls, 'geoms', []))):
        if g.length > 1: rail_on_line(simplify_line(np.array(g.coords), 0.03), 22.20)
# Staffelgeschoss-Dach
R3top = R3.buffer(-WALL_T, join_style=2).difference(R0.buffer(0.05)).buffer(0)
slab(U, M_MEMB, R3top, 25.19, 25.24, cap_bottom=False, sides=False)
slab(U, M_GRAVEL, R3top, 25.25, 25.31, cap_bottom=False)
# Hof-Garten (R1) und Lichtschacht-Boden (R4)
slab(U, M_LAWN, R1.buffer(-0.02), 17.40, 17.79, cap_bottom=False)
slab(U, M_GRAVEL, R4.buffer(-0.02), 17.40, 17.75, cap_bottom=False)
print('  + Daecher: %d' % U.tris())


# ------------------------------------------------------------------ Hof-Garten
def garden():
    P = R1.buffer(-0.6, join_style=2)
    # Wege: Kreuz aus 1.5-m-Plattenwegen
    c = np.array(R1.centroid.coords[0])
    tNE = np.asarray(SEG['NE']['t']); tS = np.asarray(SEG['S_ost']['t'])
    w1 = LineString([c - tNE * 30, c + tNE * 30]).buffer(0.75, cap_style=2)
    w2 = LineString([c - tS * 30, c + tS * 30]).buffer(0.75, cap_style=2)
    ways = w1.union(w2).intersection(P)
    slab(U, M_PAVER, ways, 17.74, 17.83)
    beds = P.difference(ways.buffer(0.2))
    # Hochbeete-Einfassung (Corten-artig dunkel) + Pflanzen
    for b in ([beds] if isinstance(beds, Polygon) else list(beds.geoms)):
        if b.area < 3: continue
        slab(U, M_STEELD, b.difference(b.buffer(-0.08, join_style=2)), 17.79, 18.15)
        slab(U, M_LAWN, b.buffer(-0.08, join_style=2), 17.79, 18.08, cap_bottom=False, sides=False)
        minx, miny, maxx, maxy = b.bounds
        for x in np.arange(minx + 0.5, maxx, 0.9):
            for y in np.arange(miny + 0.5, maxy, 0.9):
                q = np.array([x, y]) + rng.uniform(-0.3, 0.3, 2)
                if b.buffer(-0.35).contains(Point(q)):
                    r = rng.uniform(0.25, 0.5)
                    uvsphere(U, M_PLANT, (q[0], q[1], 18.08 + r * 0.5), r, r, r * 0.8, nu=7, nv=4)
    # Baum (DOP: kleiner Baum im Hof)
    tq = c + tS * -4.0 + tNE * 2.0
    if not R1.buffer(-1.5).contains(Point(tq)): tq = np.array(R1.buffer(-2).representative_point().coords[0])
    rod(U, M_BARK, (tq[0], tq[1], 17.79), (tq[0], tq[1], 20.6), 0.11, 8)
    for (dx, dy, dz, r) in ((0, 0, 21.4, 1.4), (0.8, 0.3, 21.0, 1.0), (-0.7, -0.4, 21.1, 1.0), (0.1, 0.8, 21.9, 0.9), (-0.2, -0.6, 22.0, 0.8)):
        uvsphere(U, M_PLANT, (tq[0] + dx, tq[1] + dy, dz), r, r, r * 0.85, nu=9, nv=6)
    # Baenke
    for e in (-3.0, 3.0):
        q = c + tNE * e + tS * 1.4
        pc = Piece(); box(pc, M_BARK, -0.9, 0.9, -0.22, 0.22, 0.40, 0.46); box(pc, M_STEELD, -0.85, -0.75, -0.18, 0.18, 0.0, 0.40); box(pc, M_STEELD, 0.75, 0.85, -0.18, 0.18, 0.0, 0.40)
        Mr = np.array([[tNE[0], -tNE[1], 0], [tNE[1], tNE[0], 0], [0, 0, 1]])
        U.extend(pc, lambda V, Mr=Mr, q=q: V @ Mr.T + np.array([q[0], q[1], 17.83]))
    # Lichtschacht: ein paar Gräser
    for i in range(10):
        q = np.array(R4.buffer(-0.5).representative_point().coords[0])
        pts = []
        minx, miny, maxx, maxy = R4.bounds
        for _ in range(200):
            qq = rng.uniform([minx, miny], [maxx, maxy])
            if R4.buffer(-0.5).contains(Point(qq)): pts.append(qq)
            if len(pts) >= 14: break
        for qq in pts:
            r = rng.uniform(0.2, 0.35)
            uvsphere(U, M_PLANT, (qq[0], qq[1], 17.75 + r * 0.6), r, r, r, nu=6, nv=4)
        break


garden()


# ------------------------------------------------------------------ Dachaufbauten
def roof_equipment():
    # Treppen-/Aufzugsaustritt
    slab(U, M_HVAC, roof_box, ROOF_Z, 24.10)
    slab(U, M_COPE, roof_box.buffer(0.06, join_style=2), 24.10, 24.22)
    cbx2 = np.array(roof_box.centroid.coords[0])
    # Tuer zur Dachflaeche (Seite Richtung Hof)
    dn = uy
    pc = Piece(); box(pc, M_STEELD, -0.55, 0.55, -1.83, -1.80, ROOF_Z + 0.12, ROOF_Z + 2.25); box(pc, M_STEEL, 0.35, 0.47, -1.88, -1.83, ROOF_Z + 1.0, ROOF_Z + 1.06)
    U.extend(pc, lambda V: np.c_[cbx2[0] + V[:, 0] * ux[0] + V[:, 1] * uy[0], cbx2[1] + V[:, 0] * ux[1] + V[:, 1] * uy[1], V[:, 2]])
    # Rauchabzugs-Lichtkuppeln (RWA) und Entlueftungshauben verteilt (deterministisch)
    inner = sedum.buffer(-1.0)
    minx, miny, maxx, maxy = inner.bounds
    pts = []
    tries = 0
    while len(pts) < 16 and tries < 4000:
        tries += 1
        qq = rng.uniform([minx, miny], [maxx, maxy])
        if inner.contains(Point(qq)) and all(np.linalg.norm(qq - p) > 7 for p in pts): pts.append(qq)
    for i, qq in enumerate(pts):
        if i % 3 == 0:     # Lichtkuppel 1.2 x 1.2
            pc = Piece(); box(pc, M_HVAC, -0.7, 0.7, -0.7, 0.7, 0.0, 0.40)
            Mr = np.array([[tNE[0], -tNE[1], 0], [tNE[1], tNE[0], 0], [0, 0, 1]])
            U.extend(pc, lambda V, Mr=Mr, qq=qq: V @ Mr.T + np.array([qq[0], qq[1], ROOF_Z + 0.10]))
            uvsphere(U, M_CGLASS, (qq[0], qq[1], ROOF_Z + 0.50), 0.62, 0.62, 0.28, nu=10, nv=4)
        else:              # Entlueftungshaube
            lathe(U, M_HVAC, (qq[0], qq[1], ROOF_Z + 0.10), [(0.09, 0), (0.09, 0.55), (0.22, 0.58), (0.20, 0.70), (0.0, 0.74)], n=10)
    # Absturzsicherungs-Anschlagpunkte
    for qq in pts[:8]:
        q2 = qq + np.array([2.0, 1.0])
        if inner.contains(Point(q2)):
            rod(U, M_STEEL, (q2[0], q2[1], ROOF_Z + 0.10), (q2[0], q2[1], ROOF_Z + 0.45), 0.03, 6)

    # --- Technik-Einhausung R0 (Lamellenwand, Stahlraster, Geraete) ---
    zb, zt = 25.25, 28.64
    P0 = orient(R0, 1.0)
    ring0 = np.array(P0.exterior.coords)
    ring0 = simplify_line(ring0, 0.08)
    # Pfosten
    ls0 = LineString(ring0); L0 = ls0.length
    for d in np.arange(0, L0, 2.7):
        q = np.array(ls0.interpolate(d).coords[0])
        pc = Piece(); box(pc, M_STEELD, -0.08, 0.08, -0.08, 0.08, zb, zt)
        U.extend(pc, lambda V, q=q: V + np.array([q[0], q[1], 0]))
    # Lamellen: geneigte Streifen entlang der geschlossenen Kontur
    out0 = P0.buffer(0.0)
    for z in np.arange(zb + 0.35, zt - 0.05, 0.24):
        band = P0.buffer(0.03, join_style=2).difference(P0.buffer(-0.03, join_style=2))
        slab(U, M_STEEL, band, z, z + 0.13)
    slab(U, M_STEEL, P0.buffer(0.06, join_style=2).difference(P0.buffer(-0.12, join_style=2)), zt - 0.18, zt)
    slab(U, M_STEELD, P0.buffer(0.06, join_style=2).difference(P0.buffer(-0.12, join_style=2)), zb, zb + 0.30)
    slab(U, M_GRAVEL, P0.buffer(-0.12, join_style=2), zb - 0.05, zb, cap_bottom=False, sides=False)
    # Stahlraster oben (DOP: Gitter)
    minx, miny, maxx, maxy = P0.bounds
    for x in np.arange(minx + 2.0, maxx, 4.0):
        g = LineString([(x, miny - 1), (x, maxy + 1)]).intersection(P0.buffer(-0.1))
        for gg in ([g] if isinstance(g, LineString) else list(getattr(g, 'geoms', []))):
            c = np.array(gg.coords)
            if len(c) < 2: continue
            d = c[1] - c[0]; Ls = np.linalg.norm(d)
            if Ls < 0.5: continue
            t = d / Ls; l = np.array([-t[1], t[0]])
            pc = Piece(); box(pc, M_STEEL, 0, Ls, -0.06, 0.06, zt - 0.30, zt - 0.02)
            U.extend(pc, lambda V, c0=c[0], t=t, l=l: np.c_[c0[0] + V[:, 0] * t[0] + V[:, 1] * l[0], c0[1] + V[:, 0] * t[1] + V[:, 1] * l[1], V[:, 2]])
    for y in np.arange(miny + 2.0, maxy, 4.0):
        g = LineString([(minx - 1, y), (maxx + 1, y)]).intersection(P0.buffer(-0.1))
        for gg in ([g] if isinstance(g, LineString) else list(getattr(g, 'geoms', []))):
            c = np.array(gg.coords)
            if len(c) < 2: continue
            pc = Piece(); box(pc, M_STEEL, min(c[:, 0]), max(c[:, 0]), y - 0.06, y + 0.06, zt - 0.30, zt - 0.02)
            U.extend(pc, lambda V: V)
    # Geraete: Rueckkuehler mit Ventilatoren, Lueftungsgeraete, Kanaele
    inner0 = P0.buffer(-1.2)
    cand = []
    minx, miny, maxx, maxy = inner0.bounds
    for x in np.arange(minx, maxx, 0.5):
        for y in np.arange(miny, maxy, 0.5):
            cand.append((x, y))
    placed = []
    def place(w, d, h, mat, fans=0, ang=0.0):
        ca, sa = math.cos(ang), math.sin(ang)
        for (x, y) in cand:
            rect = Polygon([(x + ca * sx - sa * sy, y + sa * sx + ca * sy) for sx, sy in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))])
            if inner0.contains(rect) and all(not rect.buffer(0.6).intersects(o) for o in placed):
                placed.append(rect)
                slab(U, mat, rect, zb, zb + h)
                if fans:
                    for k in range(fans):
                        fx = -w / 2 + w * (k + 0.5) / fans
                        q = (x + ca * fx, y + sa * fx)
                        lathe(U, M_FAN, (q[0], q[1], zb + h), [(0.38, 0), (0.38, 0.12), (0.34, 0.12), (0.34, 0.05), (0.0, 0.04)], n=14)
                        rod(U, M_STEELD, (q[0] - 0.36, q[1], zb + h + 0.12), (q[0] + 0.36, q[1], zb + h + 0.12), 0.01, 4)
                        rod(U, M_STEELD, (q[0], q[1] - 0.36, zb + h + 0.12), (q[0], q[1] + 0.36, zb + h + 0.12), 0.01, 4)
                return rect
        return None
    angR0 = math.atan2(*np.asarray(SEG['W_sued']['t'])[::-1])
    for _ in range(2): place(6.2, 2.4, 2.2, M_HVAC, 0, angR0)
    for _ in range(4): place(3.6, 1.2, 1.6, M_HVAC, 3, angR0)
    for _ in range(3): place(2.0, 1.4, 1.3, M_HVAC, 1, angR0)
    for _ in range(4): place(1.4, 1.0, 0.9, M_HVAC, 0, angR0)
    # Kanaele von den Lueftungsgeraeten nach unten (Schaechte)
    for r in placed[:2]:
        c = np.array(r.centroid.coords[0])
        slab(U, M_HVAC, Point(c).buffer(0.45, cap_style=3).buffer(0), zb, zb + 2.6)
    # Staffelgeschoss-Dach: Ausstieg + Hauben
    inner3 = R3top.buffer(-1.0)
    for i in range(5):
        q = np.array(inner3.representative_point().coords[0]) if i == 0 else None
        if q is None:
            minx, miny, maxx, maxy = inner3.bounds
            for _ in range(200):
                qq = rng.uniform([minx, miny], [maxx, maxy])
                if inner3.contains(Point(qq)): q = qq; break
        if q is None: continue
        if i == 0:
            slab(U, M_HVAC, Point(q).buffer(0.8, cap_style=3), 25.31, 26.2)
        else:
            lathe(U, M_HVAC, (q[0], q[1], 25.31), [(0.09, 0), (0.09, 0.5), (0.22, 0.53), (0.2, 0.65), (0.0, 0.69)], n=10)


roof_equipment()
print('  + Dachaufbauten: %d' % U.tris())


# ------------------------------------------------------------------ Werbeanlagen P&C
def blade_sign(seg, x, z0, kind):
    """Ausleger senkrecht zur Fassade an Pfeilerachse x."""
    fn = K.seg_map(seg)
    p0 = fn(np.array([[x, 0.0, 0.0]]))[0, :2]
    t = K.seg_tangent(seg, x); out = np.array([t[1], -t[0]])
    if kind == 'shield':
        w, h, th, off = 0.62, 0.70, 0.10, 0.28
        tip = p0 + out * (off + w)
        if any(Pc.buffer(0.4).intersects(LineString([p0 + out * 0.05, tip])) for Pc in (CANOPY_R5, CANOPY_R6)):
            return          # kollidiert mit Hauptvordach
        # Schild (unten gerundet) in (u = nach aussen, v = hoch)
        ang = np.linspace(math.pi, 2 * math.pi, 12)
        poly = [(off, z0 + h), (off, z0 + 0.30)] + [(off + w / 2 + (w / 2) * math.cos(a), z0 + 0.30 + 0.30 * math.sin(a)) for a in ang[1:-1]] + [(off + w, z0 + 0.30), (off + w, z0 + h)]
        Pg = Polygon(poly)
        left = Pg.intersection(sbox(off - 1, z0 - 1, off + w * 0.56, z0 + 2))
        right = Pg.intersection(sbox(off + w * 0.56, z0 - 1, off + w + 1, z0 + 2))
        for part, mat in ((left, M_SBLUE), (right, M_SRED)):
            pc = Piece()
            pts = np.array(part.exterior.coords)[:-1]
            ext_z(pc, mat, [(u, v) for u, v in pts], -th / 2, th / 2, caps=(True, True))
            # (u,v,w) -> Welt: u entlang out, v = z, w entlang t
            U.extend(pc, lambda V, p0=p0, out=out, t=t: np.c_[p0[0] + V[:, 0] * out[0] - V[:, 2] * t[0],
                                                             p0[1] + V[:, 0] * out[1] - V[:, 2] * t[1], V[:, 1]])
        # Ausleger
        rod(U, M_STEELD, (*(p0 + out * 0.0), z0 + h - 0.08), (*(p0 + out * off), z0 + h - 0.08), 0.025, 6)
        rod(U, M_STEELD, (*(p0 + out * 0.0), z0 + 0.30), (*(p0 + out * off), z0 + 0.30), 0.025, 6)
        cen = p0 + out * (off + w / 2)
        for sgn in (1, -1):
            nn = t * sgn
            TEXTS.append(dict(text='P&C', pos=[*(cen + nn * (th / 2 + 0.002)), z0 + 0.42], normal=[nn[0], nn[1], 0.0], dir=[-nn[1], nn[0], 0.0],
                              size=0.30, extrude=0.006, mat=M_SWHITE, align='CENTER'))
    else:   # 'banner' -- schwarzes Hochformat-Transparent mit Schriftzug, 2.-4. OG
        w, h, th, off = 1.30, 8.0, 0.14, 0.35
        pc = Piece(); box(pc, M_SBLACK, off, off + w, z0, z0 + h, -th / 2, th / 2)
        U.extend(pc, lambda V, p0=p0, out=out, t=t: np.c_[p0[0] + V[:, 0] * out[0] - V[:, 2] * t[0],
                                                         p0[1] + V[:, 0] * out[1] - V[:, 2] * t[1], V[:, 1]])
        for zz in (z0 + 0.6, z0 + h - 0.6):
            rod(U, M_STEELD, (*p0, zz), (*(p0 + out * off), zz), 0.03, 6)
        cen = p0 + out * (off + w / 2)
        for sgn in (1, -1):
            TEXTS.append(dict(text='Peek&Cloppenburg', pos=[*(cen + t * sgn * (th / 2 + 0.002)), z0 + h / 2], normal=[*(t * sgn), 0.0],
                              dir=[0, 0, 1], size=0.78, extrude=0.008, mat=M_SWHITE, align='CENTER',
                              fit=h - 0.8))


Wn = SEG['W_nord']; NE = SEG['NE']
ax_w = Wn['axes']; ax_ne = NE['axes']
# W-Fassade (Karmarschstrasse): Schilder an den ersten Pfeilern + zwei Banner nahe dem Bug
for i in (0, 1, 2, 3, 5, 8, 9):
    blade_sign(Wn, ax_w[i], 5.05, 'shield')
for i in (2, 3):
    blade_sign(Wn, ax_w[i], 9.4, 'banner')
# NE-Fassade (Platz): letzte Pfeiler zum Bug + zwei Banner in der Mitte
for i in (len(ax_ne) - 1, len(ax_ne) - 2, len(ax_ne) - 4, len(ax_ne) - 7, len(ax_ne) - 12, 6, 4):
    blade_sign(NE, ax_ne[i], 5.05, 'shield')
for i in (len(ax_ne) - 7, len(ax_ne) - 9):
    blade_sign(NE, ax_ne[i], 9.4, 'banner')
# Schriftzug ueber dem Bug-Eingang (Glas, P&C) -- Logo-Schild an der Mittelstuetze innen
print('  + Werbeanlagen: %d, Schriften %d' % (U.tris(), len(TEXTS)))

# ------------------------------------------------------------------ Kollision
def col_prism(poly, z0, z1):
    slab(COL, 'col', poly, z0, z1)


colpoly = FP.difference(COLONNADE.buffer(0.02))
# Stuetzen der Kolonnade
fnp = K.seg_map(prow)
for xc in prow['cols']:
    sq = fnp(np.array([[xc - 0.4, 0, 0], [xc + 0.4, 0, 0], [xc + 0.4, 0.85, 0], [xc - 0.4, 0.85, 0]]))[:, :2]
    colpoly = colpoly.union(Polygon(sq))
glass_back = arc_band(prow, prow['YG'] - 0.05, prow['YG'] + 0.2, 0.0, prow['L'])
colpoly = colpoly.union(glass_back)
col_prism(colpoly.buffer(0), -1.0, 22.20)
col_prism(R3.buffer(0), 21.0, 26.06)
col_prism(R0.buffer(0), 25.0, 28.64)

# ------------------------------------------------------------------ Kontroll-Kameras (Plan-/Blenderkoordinaten)
CAMS = {}
def facade_cam(seg, i_bay, dist, z_eye, z_tgt, side=0.0, lens=24):
    n, k, a = seg['layout']; bw = seg['bw']
    xc = a + (i_bay + 0.5) * bw
    fn = K.seg_map(seg)
    p = fn(np.array([[xc + side, -dist, z_eye]]))[0]; q = fn(np.array([[xc, 0.3, z_tgt]]))[0]
    return dict(loc=p.tolist(), tgt=q.tolist(), lens=lens)
ne_doors = [i for i, (kd, c) in enumerate(NE['kinds']) if kd == 'door' and c]
CAMS['lm_center_entrance'] = facade_cam(NE, ne_doors[-1], 5.2, 1.65, 2.3, side=-2.2, lens=22)
w_main = [i for i, (kd, c) in enumerate(Wn['kinds']) if kd == 'main']
CAMS['chk_main_w'] = facade_cam(Wn, w_main[1], 9.0, 1.7, 4.0, side=3.0, lens=22)
CAMS['chk_notch_w'] = facade_cam(SEG['W_sued'], 0, 16.0, 1.7, 10.0, side=-6.0, lens=24)
CAMS['chk_south'] = facade_cam(SEG['S_ost'], 0, 7.0, 1.7, 9.0, side=-9.0, lens=16)
CAMS['chk_sw_tip'] = facade_cam(SEG['S_west'], 1, 8.0, 1.7, 9.0, side=-9.0, lens=16)

# ------------------------------------------------------------------ Ausgabe
def pack(prefix, piece, arrs):
    for m, (V, F, UV) in piece.arrays().items():
        arrs['%s|%s|V' % (prefix, m)] = V.astype(np.float32)
        arrs['%s|%s|F' % (prefix, m)] = F.astype(np.int32)
        arrs['%s|%s|UV' % (prefix, m)] = UV.reshape(-1, 6).astype(np.float32)


arrs = {}
pack('U', U, arrs)
for name, pc in MODULES.items(): pack('M:' + name, pc, arrs)
pack('COL', COL, arrs)
np.savez_compressed(os.path.join(OUT, 'kc_geo.npz'), **arrs)
inst_tris = sum(MODULES[n].tris() for n, _ in INST)
stats = dict(unique_tris=U.tris(), module_tris={n: p.tris() for n, p in MODULES.items()}, instances=len(INST),
             instanced_tris=inst_tris, total_tris=U.tris() + inst_tris, collision_tris=COL.tris())
meta = dict(cams=CAMS, materials=MATS, instances=[dict(module=n, matrix=M.tolist()) for n, M in INST], texts=TEXTS, stats=stats,
            segments=[dict(kind=s['kind'], name=s.get('name', ''), L=s['L'], top=s['top'], layout=s.get('layout'), bays=s.get('bays')) for s in segs],
            footprint=[list(map(float, p)) for p in np.array(FP.exterior.coords)])
json.dump(meta, open(os.path.join(OUT, 'kc_geo.json'), 'w'), indent=1, default=float)
print(json.dumps(stats, indent=1))

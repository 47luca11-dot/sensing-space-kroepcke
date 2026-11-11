"""Instanzierte Prototypen fuer Ladenfronten und Schaufenster-Innenraeume (je Auslagenart mehrere Varianten).

Konvention wie protos.py: lokale Koordinaten u = rechts, v = oben, d = zur Strasse (Meter); Farben linear.
Schluessel:
  sf_fix[_t], sf_door, sf_door2, sfw_fix, sfw_door     Schaufenster-/Tuerrahmen (Ursprung Rahmen, Nennmass w0 x h0)
  dsp_<art>_bay<k>    Auslage direkt hinter dem Glas: belegt u in [-1.3, 1.3], d in [-1.6, 0], v ab Ladenboden
  dsp_<art>_back<k>   Rueckwand-Moebel: u in [-2, 2], d in [0, 0.7] (Ursprung an der Rueckwand, Front zur Strasse)
  dsp_<art>_mid<k>    Moebel in Raummitte: u in [-1.2, 1.2], d in [-0.8, 0.8]
  dsp_ceil, dsp_ceil_warm, dsp_track   Deckenleuchten (Ursprung Deckenebene, haengen nach unten), Nennflaeche 4 x 4 m
  dsp_tbl_out, dsp_chair_out, dsp_parasol, dsp_aframe   Aussenmobiliar (Ursprung Boden)
Alle Varianten sind deterministisch (Saat = Schluessel)."""
import math
import numpy as np
from protos import Proto, PROTOS, _pip
from common import hex2lin, rng_of
from gltf_io import glow_mat
glow_mat('#33CC66')

L = hex2lin
WHITE = L('#F4F2EE'); OFFW = L('#E6E2DA'); BLACK = L('#1E1E1E'); DARK = L('#2E3033'); STEEL = L('#B8BCBE'); CHROME = L('#D8DADB')
WOOD = L('#9A6A42'); WOOD_L = L('#C49A6C'); WOOD_D = L('#5A3A26'); SKIN = L('#E8D2BE'); GOLD = L('#D4AF37'); SILVER = L('#D8D8DC')
FABRICS = ['#1F2A44', '#7A1F24', '#E8E2D4', '#2E2E30', '#B8A07A', '#5A6B7A', '#C8102E', '#F2F2EE', '#3A4A3A', '#D9B8C4', '#6B4E3A',
           '#8FA3B8', '#E3C24A', '#4A3A5A', '#A8B89A', '#F0D8C0']
PRODUCTS = ['#C8102E', '#1F4FA0', '#F2C500', '#2E8B57', '#F2F2EE', '#E86A1A', '#7A3A8A', '#4AA8D8', '#1E1E1E', '#E85A8A', '#8CC63F', '#D8D0C0']
BOOKS = ['#7A1F24', '#1F2A44', '#2E5A3A', '#E8E2D4', '#C8A050', '#3A3D40', '#8E3B35', '#5A6B7A', '#D8C8A8', '#1E1E1E', '#B85A2A', '#4A6A8A']

KINDS = ('fashion', 'shoes', 'jewellery', 'optician', 'pharmacy', 'bakery', 'cafe', 'fastfood', 'bank', 'phone', 'books', 'drugstore',
         'department', 'kiosk', 'salon', 'lobby', 'office', 'home', 'florist', 'vacant', 'generic', 'sports', 'supermarket', 'toys')
NVAR = {'bay': 3, 'back': 2, 'mid': 2}


def C(h):
    return L(h) if isinstance(h, str) else np.asarray(h, float)


# ------------------------------------------------------------------ Grundformen
def blk(P, lo, hi, c, mat='trim', skip=('-v',)):
    P.box(mat, lo, hi, c=C(c), skip=skip)


def quad_d(P, u0, v0, u1, v1, d, c, mat='trim', facing=1):
    """Flaches Rechteck in Ebene d (Normale +d bzw. -d)."""
    V = [[u0, v0, d], [u1, v0, d], [u1, v1, d], [u0, v1, d]]
    P.oriented(mat, V, [[0, 1, 2], [0, 2, 3]], [0, 0, facing], C(c))


def quad_v(P, u0, d0, u1, d1, v, c, mat='trim', facing=1):
    V = [[u0, v, d0], [u1, v, d0], [u1, v, d1], [u0, v, d1]]
    P.oriented(mat, V, [[0, 1, 2], [0, 2, 3]], [0, facing, 0], C(c))


def frustum(P, mat, y0, y1, w0, d0, w1, d1, c, uc=0.0, dc=0.0, top=True, bottom=False):
    """Pyramidenstumpf (Rumpf, Kleid, Lampenschirm): unten w0 x d0, oben w1 x d1, zentriert in (uc, dc)."""
    V = []
    for (y, w, dd) in ((y0, w0, d0), (y1, w1, d1)):
        V += [[uc - w / 2, y, dc - dd / 2], [uc + w / 2, y, dc - dd / 2], [uc + w / 2, y, dc + dd / 2], [uc - w / 2, y, dc + dd / 2]]
    V = np.array(V, float)
    F = []; W = []
    ctr = np.array([uc, (y0 + y1) / 2, dc])
    for i in range(4):
        j = (i + 1) % 4
        F += [[i, j, 4 + j], [i, 4 + j, 4 + i]]
        m = (V[i] + V[j] + V[4 + j] + V[4 + i]) / 4 - ctr; m[1] = 0
        W += [m, m]
    if top:
        F += [[4, 5, 6], [4, 6, 7]]; W += [[0, 1, 0]] * 2
    if bottom:
        F += [[0, 1, 2], [0, 2, 3]]; W += [[0, -1, 0]] * 2
    P.oriented(mat, V, np.array(F), np.array(W, float), C(c))


def blob(P, mat, ctr, rad, c, seg=6, rings=4):
    """Ellipsoid (rad = (ru, rv, rd)), niedrig aufgeloest."""
    ctr = np.asarray(ctr, float); rad = np.asarray(rad, float)
    V = [ctr + [0, -rad[1], 0]]
    for k in range(1, rings):
        t = math.pi * k / rings
        for i in range(seg):
            a = 2 * math.pi * i / seg
            V.append(ctr + rad * [math.sin(t) * math.cos(a), -math.cos(t), math.sin(t) * math.sin(a)])
    V.append(ctr + [0, rad[1], 0])
    V = np.array(V); F = []
    top = len(V) - 1
    for i in range(seg):
        F.append([0, 1 + (i + 1) % seg, 1 + i])
    for k in range(rings - 2):
        b0 = 1 + k * seg; b1 = b0 + seg
        for i in range(seg):
            j = (i + 1) % seg
            F += [[b0 + i, b0 + j, b1 + j], [b0 + i, b1 + j, b1 + i]]
    b0 = 1 + (rings - 2) * seg
    for i in range(seg):
        F.append([b0 + i, b0 + (i + 1) % seg, top])
    F = np.array(F)
    W = V[F].mean(1) - ctr
    P.oriented(mat, V, F, W, C(c))


def cylv(P, mat, u, d, y0, y1, r, c, seg=8, caps=True):
    P.cyl(mat, [u, y0, d], [u, y1, d], r, seg=seg, c=C(c), caps=caps)


# ------------------------------------------------------------------ Figuren / Gegenstaende
def mannequin(P, u, d, rng, female=True, body='#F2F0EC', outfit=None, scale=1.0, turn=0.0):
    """Schaufensterpuppe (stilisiert, ~240 Dreiecke) mit Kleidung. outfit: dict top/bottom/shoes Farben."""
    s = scale
    bc = body
    o = outfit or {}
    top = o.get('top', FABRICS[int(rng.integers(len(FABRICS)))]); bot = o.get('bottom', FABRICS[int(rng.integers(len(FABRICS)))])
    dress = o.get('dress', female and rng.random() < 0.45)
    coat = o.get('coat', (not dress) and rng.random() < 0.35)
    # Standfuss
    cylv(P, 'metal', u, d, 0.0, 0.015, 0.16 * s, '#3A3D40', seg=8)
    cylv(P, 'metal', u, d, 0.015, 0.75 * s, 0.012 * s, CHROME, seg=4, caps=False)
    hw = (0.33 if female else 0.35) * s; sw = (0.37 if female else 0.44) * s
    # Beine (Hose bzw. Haut)
    legc = bot if not dress else bc
    for side in (-1, 1):
        frustum(P, 'interior', 0.06 * s, 0.86 * s, 0.09 * s, 0.1 * s, 0.13 * s, 0.15 * s, legc, uc=u + side * 0.075 * s, dc=d, top=False)
        blk(P, [u + side * 0.075 * s - 0.05 * s, 0.0, d - 0.05 * s], [u + side * 0.075 * s + 0.05 * s, 0.06 * s, d + 0.14 * s],
            '#1E1E1E' if rng.random() < 0.6 else '#6B4E3A', mat='interior')
    # Rumpf
    if dress:
        frustum(P, 'interior', 0.42 * s, 0.98 * s, 0.5 * s, 0.38 * s, hw, 0.21 * s, top, uc=u, dc=d, bottom=True, top=False)
    else:
        frustum(P, 'interior', 0.84 * s, 0.98 * s, hw, 0.21 * s, hw, 0.21 * s, bot, uc=u, dc=d, bottom=True, top=False)
    torso_c = top
    frustum(P, 'interior', 0.98 * s, 1.22 * s, hw * 0.92, 0.2 * s, (0.31 if female else 0.36) * s, 0.21 * s, torso_c, uc=u, dc=d, top=False)
    frustum(P, 'interior', 1.22 * s, 1.44 * s, (0.31 if female else 0.36) * s, 0.21 * s, sw, 0.2 * s, torso_c, uc=u, dc=d, top=False)
    frustum(P, 'interior', 1.44 * s, 1.5 * s, sw, 0.2 * s, 0.16 * s, 0.12 * s, torso_c, uc=u, dc=d, top=True)
    if coat:
        cc = FABRICS[int(rng.integers(len(FABRICS)))]
        frustum(P, 'interior', 0.6 * s, 1.47 * s, hw + 0.08 * s, 0.26 * s, sw + 0.04 * s, 0.24 * s, cc, uc=u, dc=d - 0.005, bottom=True, top=False)
    # Hals, Kopf
    cylv(P, 'interior', u, d, 1.48 * s, 1.58 * s, 0.045 * s, bc, seg=6, caps=False)
    blob(P, 'interior', [u, 1.68 * s, d + 0.01 * s], [0.085 * s, 0.115 * s, 0.1 * s], bc, seg=6, rings=4)
    # Arme (leicht abgespreizt)
    ac = top if not (dress and rng.random() < 0.5) else bc
    for side in (-1, 1):
        x0 = u + side * (sw / 2 - 0.03 * s)
        a = 0.08 * s + 0.06 * s * rng.random()
        V0 = np.array([x0, 1.45 * s, d]); V1 = np.array([x0 + side * a, 0.82 * s, d + 0.04 * s * (rng.random() - 0.3)])
        _limb(P, V0, V1, 0.05 * s, ac)


def _limb(P, p0, p1, r, c):
    P.cyl('interior', p0, p1, r, seg=5, c=C(c), caps=True)


def shoe(P, u, d, c, scale=1.0, heel=False):
    s = scale
    blk(P, [u - 0.045 * s, 0.0, d - 0.13 * s], [u + 0.045 * s, 0.025 * s, d + 0.13 * s], '#2A2A2A')
    frustum(P, 'trim', 0.025 * s, 0.09 * s, 0.09 * s, 0.24 * s, 0.07 * s, 0.12 * s, c, uc=u, dc=d - 0.05 * s)
    if heel:
        blk(P, [u - 0.02 * s, 0.0, d - 0.13 * s], [u + 0.02 * s, 0.08 * s, d - 0.09 * s], c)


def bread(P, u, v, d, c, lng=0.28):
    blob(P, 'interior', [u, v + 0.045, d], [lng / 2, 0.05, 0.07], c, seg=6, rings=3)


def glasses(P, u, v, d, c):
    """Brille (Vorderansicht): zwei Glaeser (Rahmenflaechen mit dunklem Glas) + Steg, nur Frontflaechen."""
    for s in (-1, 1):
        quad_d(P, u + s * 0.035 - 0.03, v - 0.02, u + s * 0.035 + 0.03, v + 0.02, d + 0.004, c)
        quad_d(P, u + s * 0.035 - 0.022, v - 0.013, u + s * 0.035 + 0.022, v + 0.013, d + 0.005, '#3A4048')
    quad_d(P, u - 0.008, v + 0.005, u + 0.008, v + 0.012, d + 0.004, c)


def table(P, u, d, w, dd, h, top_c, leg_c=DARK, round_=False):
    if round_:
        cylv(P, 'trim', u, d, h - 0.03, h, w / 2, top_c, seg=10)
        cylv(P, 'metal', u, d, 0.0, h - 0.03, 0.025, leg_c, seg=5, caps=False)
        cylv(P, 'metal', u, d, 0.0, 0.02, 0.2, leg_c, seg=8)
    else:
        blk(P, [u - w / 2, h - 0.035, d - dd / 2], [u + w / 2, h, d + dd / 2], top_c, skip=())
        for su in (-1, 1):
            for sd in (-1, 1):
                blk(P, [u + su * (w / 2 - 0.05) - 0.02, 0.0, d + sd * (dd / 2 - 0.05) - 0.02],
                    [u + su * (w / 2 - 0.05) + 0.02, h - 0.035, d + sd * (dd / 2 - 0.05) + 0.02], leg_c, mat='metal', skip=('-v', '+v'))


def chair(P, u, d, c, face=1, h=0.45):
    """Stuhl, Lehne auf der Seite -face (Sitzender blickt in Richtung face*d)."""
    blk(P, [u - 0.21, h - 0.04, d - 0.21], [u + 0.21, h, d + 0.21], c, skip=())
    for su in (-1, 1):
        for sd in (-1, 1):
            blk(P, [u + su * 0.18 - 0.015, 0, d + sd * 0.18 - 0.015], [u + su * 0.18 + 0.015, h - 0.04, d + sd * 0.18 + 0.015], DARK,
                mat='metal', skip=('-v', '+v'))
    db = d - face * 0.2
    blk(P, [u - 0.2, h, min(db, db - face * 0.03)], [u + 0.2, h + 0.42, max(db, db - face * 0.03)], c, skip=('-v',))


def stool(P, u, d, c, h=0.75):
    cylv(P, 'trim', u, d, h - 0.04, h, 0.17, c, seg=8)
    cylv(P, 'metal', u, d, 0.0, h - 0.04, 0.02, DARK, seg=4, caps=False)
    cylv(P, 'metal', u, d, 0.0, 0.015, 0.18, DARK, seg=8)


def poster(P, u0, u1, v0, v1, d, c, frame=DARK, rng=None):
    """Plakat/Werbetafel (Vorderseite +d) mit Bildfeld, Kopfzeile und Rahmen; Ruckseite geschlossen."""
    blk(P, [u0, v0, d - 0.03], [u1, v1, d], frame, skip=())
    quad_d(P, u0 + 0.03, v0 + 0.03, u1 - 0.03, v1 - 0.03, d + 0.001, c)
    if rng is not None:
        k = PRODUCTS[int(rng.integers(len(PRODUCTS)))]
        hh = (v1 - v0)
        quad_d(P, u0 + 0.08, v0 + hh * 0.55, u1 - 0.08, v1 - 0.08, d + 0.002, k)
        quad_d(P, u0 + 0.12, v0 + hh * 0.18, u1 - 0.3, v0 + hh * 0.26, d + 0.002, '#F4F2EE')
        quad_d(P, u0 + 0.12, v0 + hh * 0.3, u1 - 0.12, v0 + hh * 0.42, d + 0.002, '#1E1E1E')


def product_grid(P, u0, u1, v0, v1, d, rng, rows, palette=PRODUCTS, wmin=0.06, wmax=0.18, hfrac=(0.55, 0.9), boxes=False, depth=0.1):
    """Regal-Fachreihen mit Produkten (Frontflaechen bzw. flache Kaesten), Fachboeden."""
    rh = (v1 - v0) / rows
    for r in range(rows):
        y = v0 + r * rh
        blk(P, [u0, y, d - 0.32], [u1, y + 0.025, d], OFFW, skip=())
        x = u0 + 0.02
        y0 = y + 0.025
        while x < u1 - 0.04:
            w = float(rng.uniform(wmin, wmax)); w = min(w, u1 - 0.02 - x)
            h = rh * float(rng.uniform(*hfrac))
            c = palette[int(rng.integers(len(palette)))]
            if boxes:
                blk(P, [x, y0, d - depth - 0.02], [x + w - 0.01, y0 + h, d - 0.02], c, skip=('-v', '-d'))
            else:
                quad_d(P, x, y0, x + w - 0.008, y0 + h, d - 0.02, c)
            x += w


def garments(P, u0, u1, v_rail, d, rng, palette=FABRICS, length=(0.55, 0.95), side=False):
    """Kleiderstange mit haengenden Kleidungsstuecken (duenne Kaesten), Blick von vorne auf die Kanten oder Flaechen."""
    P.cyl('metal', [u0, v_rail, d], [u1, v_rail, d], 0.012, seg=5, c=CHROME, caps=False)
    x = u0 + 0.06
    while x < u1 - 0.05:
        c = palette[int(rng.integers(len(palette)))]
        ln = float(rng.uniform(*length))
        if side:                                            # Seitenansicht: Stoff quer zur Stange
            blk(P, [x - 0.012, v_rail - ln, d - 0.22], [x + 0.012, v_rail - 0.04, d + 0.22], c, mat='interior', skip=())
            x += float(rng.uniform(0.05, 0.08))
        else:
            w = float(rng.uniform(0.38, 0.48))
            frustum(P, 'interior', v_rail - ln, v_rail - 0.05, w, 0.06, w * 0.85, 0.05, c, uc=x + w / 2, dc=d, bottom=True)
            x += w + 0.03


def plant(P, u, d, h=1.1, pot='#3A3D40', rng=None):
    cylv(P, 'trim', u, d, 0.0, 0.4, 0.17, pot, seg=8)
    g = ['#3E6B3A', '#4E7A3E', '#2F5A32']
    for k in range(3):
        blob(P, 'interior', [u + (k - 1) * 0.07, 0.45 + h * (0.35 + 0.18 * k), d + (k % 2) * 0.05], [0.22, 0.28, 0.22], g[k % 3], seg=6, rings=3)


# ------------------------------------------------------------------ Rahmen: Schaufenster/Tueren
def _alu_ring(P, w, h, prof=None):
    prof = prof or [(0.0, -0.07), (0.0, 0.0), (0.05, 0.0), (0.05, -0.006), (0.056, -0.012), (0.056, -0.07)]
    P.ring_rect('trim', -w / 2, 0, w / 2, h, prof)


def _sticker(P, u, v, d):
    """Oeffnungszeiten-Aufkleber (weiss, dunkle Zeilen) auf der Glasflaeche."""
    quad_d(P, u - 0.07, v - 0.1, u + 0.07, v + 0.1, d, WHITE)
    for i in range(6):
        y = v + 0.07 - i * 0.027
        quad_d(P, u - 0.055, y - 0.006, u + (0.055 if i else 0.02), y + 0.006, d + 0.001, '#3A3D40' if i else '#C8102E')


def _mat(P, u0, u1, d0, d1):
    """Fussmatte innen hinter der Tuer."""
    blk(P, [u0, 0.0, d0], [u1, 0.018, d1], '#2A2826', skip=('-v',))


def _pushbar(P, x, y0, y1, d=0.09, c=CHROME):
    P.cyl('trim', [x, y0, d], [x, y1, d], 0.016, seg=8, c=C(c))
    for y in (y0 + 0.07, y1 - 0.07):
        P.cyl('trim', [x, y, 0.016], [x, y, d], 0.01, seg=6, c=C(c))


def shopfront_proto(key):
    if key in PROTOS:
        return PROTOS[key]
    if key in ('sf_fix', 'sf_fix_t'):
        w, h = 2.6, 3.0
        P = Proto(key, 'frame', w, h)
        _alu_ring(P, w, h)
        if key == 'sf_fix_t':
            P.bar('trim', [(-0.04, -0.07), (-0.04, 0.0), (0.04, 0.0), (0.04, -0.07)], 'u', -w / 2 + 0.05, w / 2 - 0.05, at=(h - 0.5,))
        blk(P, [-w / 2 + 0.05, 0.0, -0.05], [w / 2 - 0.05, 0.12, 0.004], STEEL, skip=('-v', '-d'))
    elif key in ('sf_door', 'sf_door2'):
        dbl = key == 'sf_door2'
        w, h = (2.2, 3.0) if dbl else (1.25, 3.0)
        P = Proto(key, 'frame', w, h)
        _alu_ring(P, w, h)
        hd = 2.35
        P.bar('trim', [(-0.045, -0.07), (-0.045, 0.0), (0.045, 0.0), (0.045, -0.07)], 'u', -w / 2 + 0.05, w / 2 - 0.05, at=(hd,))
        leaves = [(-w / 2 + 0.06, 0.0), (0.0, w / 2 - 0.06)] if dbl else [(-w / 2 + 0.06, w / 2 - 0.06)]
        for li, (a, b) in enumerate(leaves):
            P.ring_rect('trim', a, 0.005, b, hd - 0.05, [(0.0, -0.05), (0.0, 0.012), (0.01, 0.018), (0.075, 0.018), (0.085, 0.006), (0.085, -0.03)])
            blk(P, [a + 0.085, 0.005, -0.03], [b - 0.085, 0.24, 0.016], STEEL, skip=('-v',))
            hx = (b - 0.14) if (not dbl or li == 0) else (a + 0.14)
            if dbl:
                hx = b - 0.1 if li == 0 else a + 0.1
            _pushbar(P, hx, 0.8, 1.75)
            # Tuerschliesser oben
            blk(P, [a + 0.12, hd - 0.2, 0.018], [a + 0.5, hd - 0.13, 0.07], STEEL, skip=())
        _sticker(P, (leaves[0][0] + leaves[0][1]) / 2 + (0.15 if not dbl else -0.1), 1.45, -0.035)
        _mat(P, -0.6 if not dbl else -1.0, 0.6 if not dbl else 1.0, -1.25, -0.25)
    elif key in ('sfw_fix', 'sfw_door'):
        door = key == 'sfw_door'
        w, h = (1.25, 3.0) if door else (2.4, 3.0)
        P = Proto(key, 'frame', w, h)
        prof = [(0.0, -0.08), (0.0, -0.004), (0.006, 0.0), (0.07, 0.0), (0.08, -0.012), (0.08, -0.08)]
        P.ring_rect('trim', -w / 2, 0, w / 2, h, prof)
        ht = h - 0.62
        P.bar('trim', [(-0.05, -0.08), (-0.05, 0.0), (0.05, 0.0), (0.05, -0.08)], 'u', -w / 2 + 0.07, w / 2 - 0.07, at=(ht,))
        # Oberlicht mit Sprossen
        for i in range(1, 4):
            P.bar('trim', [(-0.016, -0.05), (-0.016, -0.02), (0.016, -0.02), (0.016, -0.05)], 'v', ht + 0.05, h - 0.07, at=(-w / 2 + w * i / 4,))
        if door:
            a, b = -w / 2 + 0.08, w / 2 - 0.08
            P.ring_rect('trim', a, 0.0, b, ht - 0.05, [(0.0, -0.06), (0.0, 0.01), (0.012, 0.02), (0.1, 0.02), (0.11, 0.008), (0.11, -0.03)])
            blk(P, [a + 0.11, 0.0, -0.04], [b - 0.11, 0.85, 0.0], '#FFFFFF', skip=('-v',))       # Kassette unten (Instanzfarbe)
            blk(P, [a + 0.18, 0.12, 0.0], [b - 0.18, 0.72, 0.014], '#FFFFFF', skip=('-v',))
            P.bar('trim', [(-0.015, -0.04), (-0.015, -0.02), (0.015, -0.02), (0.015, -0.04)], 'v', 0.85, ht - 0.12, at=(0.0,))
            blk(P, [b - 0.16, 1.02, 0.0], [b - 0.11, 1.2, 0.012], GOLD, skip=())
            P.cyl('trim', [b - 0.135, 1.1, 0.012], [b - 0.135, 1.1, 0.06], 0.009, seg=6, c=GOLD)
            blk(P, [b - 0.3, 1.09, 0.05], [b - 0.13, 1.11, 0.066], GOLD, skip=())
            _sticker(P, (a + b) / 2, 1.45, -0.035)
            _mat(P, -0.55, 0.55, -1.2, -0.25)
        else:
            # Sockelpaneel (Brustung der Holzfront) und Sprossenfenster
            blk(P, [-w / 2 + 0.08, 0.0, -0.05], [w / 2 - 0.08, 0.55, 0.0], '#FFFFFF', skip=('-v',))
            for s in (-1, 1):
                blk(P, [s * w / 4 - w / 4 + 0.06 if s < 0 else 0.06, 0.08, 0.0], [-0.06 if s < 0 else w / 2 - 0.14, 0.47, 0.014], '#FFFFFF', skip=())
            P.bar('trim', MUL_W, 'u', -w / 2 + 0.07, w / 2 - 0.07, at=(0.58,))
            P.bar('trim', MUL_W, 'v', 0.6, ht - 0.05, at=(0.0,))
    else:
        raise KeyError(key)
    PROTOS[key] = P.freeze()
    return PROTOS[key]


MUL_W = [(-0.04, -0.08), (-0.04, 0.0), (0.04, 0.0), (0.04, -0.08)]


# ------------------------------------------------------------------ Deckenleuchten / Aussenmobiliar
def ceiling_proto(key):
    if key in PROTOS:
        return PROTOS[key]
    P = Proto(key, 'room', 4.0, 1.0)
    if key in ('dsp_ceil', 'dsp_ceil_warm'):
        mat = 'glow' if key == 'dsp_ceil' else 'glow_warm'
        for i in range(3):
            for j in range(3):
                u = -1.33 + 1.33 * i; d = -1.33 + 1.33 * j
                blk(P, [u - 0.32, -0.035, d - 0.32], [u + 0.32, 0.0, d + 0.32], STEEL, mat='metal', skip=('+v',))
                quad_v(P, u - 0.29, d - 0.29, u + 0.29, d + 0.29, -0.036, WHITE, mat=mat, facing=-1)
    elif key == 'dsp_track':
        for d in (-1.0, 0.6):
            blk(P, [-1.9, -0.04, d - 0.02], [1.9, 0.0, d + 0.02], DARK, mat='metal', skip=('+v',))
            for i in range(7):
                u = -1.7 + i * 0.57
                cylv(P, 'metal', u, d, -0.12, -0.04, 0.008, DARK, seg=4, caps=False)
                P.cyl('metal', [u, -0.12, d], [u, -0.2, d + 0.09], 0.035, seg=6, c=DARK)
                P.cyl('glow_warm', [u, -0.2, d + 0.09], [u, -0.202, d + 0.093], 0.03, seg=6, c=WHITE, caps=True)
    PROTOS[key] = P.freeze()
    return PROTOS[key]


def outdoor_proto(key):
    if key in PROTOS:
        return PROTOS[key]
    rng = rng_of('proto', key)
    P = Proto(key, 'ground', 1.0, 1.0)
    if key == 'dsp_tbl_out':
        table(P, 0.0, 0.0, 0.7, 0.7, 0.74, '#D8D2C6', DARK, round_=True)
        for s in (-1, 1):
            chair(P, s * 0.55, 0.0, '#5A3A26', face=1, h=0.45) if False else None
        # zwei Stuehle seitlich (Lehne aussen)
        for s in (-1, 1):
            u = s * 0.58
            blk(P, [u - 0.2, 0.42, -0.2], [u + 0.2, 0.46, 0.2], '#6A4A32', skip=())
            for su in (-1, 1):
                for sd in (-1, 1):
                    blk(P, [u + su * 0.17 - 0.015, 0, sd * 0.17 - 0.015], [u + su * 0.17 + 0.015, 0.42, sd * 0.17 + 0.015], DARK,
                        mat='metal', skip=('-v', '+v'))
            ub = u + s * 0.2
            blk(P, [min(ub, ub + s * 0.03), 0.46, -0.2], [max(ub, ub + s * 0.03), 0.86, 0.2], '#6A4A32', skip=('-v',))
    elif key == 'dsp_parasol':
        cylv(P, 'metal', 0, 0, 0.0, 0.08, 0.3, '#3A3D40', seg=8)
        cylv(P, 'metal', 0, 0, 0.08, 2.35, 0.025, '#E8E4DC', seg=5, caps=False)
        n = 8; R = 1.35
        V = [[0, 2.55, 0]] + [[R * math.cos(2 * math.pi * i / n), 2.12, R * math.sin(2 * math.pi * i / n)] for i in range(n)]
        F = [[0, 1 + (i + 1) % n, 1 + i] for i in range(n)]
        P.oriented('interior', V, F, [0, 1, 0], C('#F2EDE2'))
        P.oriented('interior', V, [[f[0], f[2], f[1]] for f in F], [0, -1, 0], C('#E8E2D4'))
        for i in range(n):
            a = [R * math.cos(2 * math.pi * i / n), 2.12, R * math.sin(2 * math.pi * i / n)]
            b = [R * math.cos(2 * math.pi * (i + 1) / n), 2.12, R * math.sin(2 * math.pi * (i + 1) / n)]
            V2 = [a, b, [b[0], 2.0, b[2]], [a[0], 2.0, a[2]]]
            m = [(a[0] + b[0]) / 2, 0, (a[2] + b[2]) / 2]
            P.oriented('interior', V2, [[0, 1, 2], [0, 2, 3]], m, C('#8E2A2A'))
            P.oriented('interior', V2, [[0, 2, 1], [0, 3, 2]], [-m[0], 0, -m[2]], C('#8E2A2A'))
    elif key == 'dsp_aframe':
        for s in (-1, 1):
            V = [[-0.3, 0.0, s * 0.28], [0.3, 0.0, s * 0.28], [0.3, 1.0, s * 0.03], [-0.3, 1.0, s * 0.03]]
            P.oriented('trim', V, [[0, 1, 2], [0, 2, 3]], [0, 0.25, s], C('#3A2A20'))
            V2 = [[-0.25, 0.12, s * 0.255], [0.25, 0.12, s * 0.255], [0.25, 0.92, s * 0.055], [-0.25, 0.92, s * 0.055]]
            V2 = [[x, y, z + s * 0.004] for x, y, z in V2]
            P.oriented('trim', V2, [[0, 1, 2], [0, 2, 3]], [0, 0.25, s], C('#22262A'))
            for i in range(5):
                t0 = 0.25 + i * 0.12; ww = 0.36 if i else 0.4
                y = 0.12 + (0.92 - 0.12) * (1 - t0); z = s * (0.255 - (0.255 - 0.055) * (1 - t0)) + s * 0.006
                V3 = [[-ww / 2, y - 0.025, z], [ww / 2, y - 0.025, z], [ww / 2, y + 0.025, z], [-ww / 2, y + 0.025, z]]
                P.oriented('trim', V3, [[0, 1, 2], [0, 2, 3]], [0, 0.25, s], C('#F2F0EA' if i else '#E8C468'))
    PROTOS[key] = P.freeze()
    return PROTOS[key]


# ------------------------------------------------------------------ Auslagen je Art
def _outfit(rng, palette=FABRICS):
    return {'top': palette[int(rng.integers(len(palette)))], 'bottom': palette[int(rng.integers(len(palette)))]}


def bay_fashion(P, rng, sporty=False):
    pal = FABRICS if not sporty else ['#1E1E1E', '#F2F2EE', '#C8102E', '#1F4FA0', '#E86A1A', '#8CC63F', '#5A6B7A']
    blk(P, [-1.2, 0.0, -1.35], [1.2, 0.12, -0.25], '#F2F0EC', skip=('-v',))        # Podest
    n = 3 if rng.random() < 0.6 else 2
    body = ['#F2F0EC', '#2A2A2C', '#D8D4CC', '#F2F0EC'][int(rng.integers(4))]
    for i in range(n):
        u = -0.9 + 1.8 * (i + 0.5) / n
        mannequin(P, u, -0.75 + 0.15 * ((i % 2) - 0.5), rng, female=(i % 2 == 0) != (rng.random() < 0.3), body=body,
                  outfit=_outfit(rng, pal), scale=1.0)
        P.parts  # noqa
    if rng.random() < 0.5:
        poster(P, -1.1, -0.2, 0.9, 2.6, -1.5, pal[int(rng.integers(len(pal)))], rng=rng)
        garments(P, 0.0, 1.15, 1.85, -1.42, rng, palette=pal, side=True)
    else:
        garments(P, -1.15, 1.15, 1.85, -1.42, rng, palette=pal, length=(0.6, 1.0))
        for su in (-1.15, 1.15):                                      # Haenger-Stangen an Seilen
            P.cyl('metal', [su, 1.85, -1.42], [su, 2.6, -1.42], 0.004, seg=3, c=STEEL, caps=False)
    # gefaltete Stapel + Tasche auf Wuerfeln vorne
    for k, cu in enumerate((-1.0, 1.0)):
        h = 0.3 + 0.15 * k
        blk(P, [cu - 0.17, 0.12, -0.5], [cu + 0.17, 0.12 + h, -0.27], '#E8E4DC', skip=('-v',))
        y = 0.12 + h
        for j in range(int(rng.integers(2, 5))):
            c = pal[int(rng.integers(len(pal)))]
            blk(P, [cu - 0.14, y, -0.47], [cu + 0.14, y + 0.045, -0.3], c, mat='interior', skip=('-v',))
            y += 0.045
    bc = ['#1E1E1E', '#6B4E3A', '#B8A07A', '#7A1F24'][int(rng.integers(4))]
    blk(P, [-0.12, 0.12, -0.42], [0.12, 0.32, -0.32], bc, mat='interior', skip=('-v',))
    P.cyl('interior', [-0.07, 0.32, -0.37], [0.0, 0.42, -0.37], 0.008, seg=3, c=C(bc), caps=False)
    P.cyl('interior', [0.0, 0.42, -0.37], [0.07, 0.32, -0.37], 0.008, seg=3, c=C(bc), caps=False)


def back_fashion(P, rng):
    for side, (u0, u1) in enumerate(((-1.95, -0.1), (0.1, 1.95))):
        garments(P, u0, u1, 1.65, 0.35, rng, side=(side == 1))
        blk(P, [u0, 1.95, 0.0], [u1, 1.98, 0.38], WHITE, skip=())
        x = u0 + 0.1
        while x < u1 - 0.3:
            blk(P, [x, 1.98, 0.08], [x + 0.28, 2.06 + 0.1 * rng.random(), 0.32], FABRICS[int(rng.integers(len(FABRICS)))], mat='interior')
            x += 0.36
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.02], '#ECE8E2', skip=('-v',))


def mid_fashion(P, rng):
    table(P, -0.5, 0.0, 1.2, 0.7, 0.8, '#D8D0C0')
    for i in range(4):
        h = 0.06 + 0.1 * rng.random()
        u = -0.95 + i * 0.3
        blk(P, [u, 0.8, -0.2], [u + 0.25, 0.8 + h, 0.2], FABRICS[int(rng.integers(len(FABRICS)))], mat='interior')
    mannequin(P, 0.75, 0.0, rng, female=rng.random() < 0.5, body='#F2F0EC', outfit=_outfit(rng))


def bay_shoes(P, rng):
    for k, (h, d0, d1) in enumerate(((0.12, -0.55, -0.15), (0.38, -0.95, -0.55), (0.64, -1.35, -0.95))):
        blk(P, [-1.15, 0.0, d0], [1.15, h, d1], '#F2F0EC' if rng.random() < 0.6 else '#C8B8A0', skip=('-v',))
        n = 5 + int(rng.integers(2))
        for i in range(n):
            u = -1.0 + 2.0 * (i + 0.5) / n
            P_shoe_c = ['#1E1E1E', '#6B4E3A', '#F2F2EE', '#C8102E', '#B8A07A', '#1F2A44', '#E8C468'][int(rng.integers(7))]
            _shoe_at(P, u, h, (d0 + d1) / 2, P_shoe_c, heel=rng.random() < 0.3)
    if rng.random() < 0.5:
        poster(P, 0.2, 1.1, 0.9, 2.5, -1.5, '#E8E2D4', rng=rng)


def _shoe_at(P, u, h, d, c, heel=False):
    s = 1.0
    blk(P, [u - 0.045, h, d - 0.13], [u + 0.045, h + 0.025, d + 0.13], '#2A2A2A')
    frustum(P, 'trim', h + 0.025, h + 0.09, 0.09, 0.24, 0.07, 0.12, c, uc=u, dc=d - 0.05)
    if heel:
        blk(P, [u - 0.02, h, d - 0.13], [u + 0.02, h + 0.08, d - 0.09], c)


def back_shoes(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.4, 0.02], '#E8E2DA', skip=('-v',))
    for r in range(4):
        y = 0.35 + r * 0.48
        blk(P, [-1.95, y, 0.02], [1.95, y + 0.025, 0.36], WHITE, skip=())
        x = -1.85
        while x < 1.8:
            _shoe_at(P, x + 0.06, y + 0.025, 0.2, ['#1E1E1E', '#6B4E3A', '#F2F2EE', '#C8102E', '#B8A07A', '#1F2A44'][int(rng.integers(6))],
                     heel=False)
            x += 0.3


def mid_shoes(P, rng):
    blk(P, [-0.9, 0.0, -0.25], [0.9, 0.42, 0.25], '#6B4E3A', skip=('-v',))
    blk(P, [-0.85, 0.42, -0.2], [0.85, 0.47, 0.2], '#C8B8A0', skip=('-v',))
    for i in range(3):
        _shoe_at(P, -0.5 + 0.5 * i, 0.47, 0.0, ['#1E1E1E', '#F2F2EE', '#C8102E'][i])


def vitrine(P, u, d, w, h=0.95, rng=None, items='jewel'):
    blk(P, [u - w / 2, 0.0, d - 0.28], [u + w / 2, h - 0.3, d + 0.28], '#2A2422', skip=('-v',))
    # Glashaube (Material Glas) mit Kanten
    gx0, gx1, gd0, gd1, gy0, gy1 = u - w / 2 + 0.02, u + w / 2 - 0.02, d - 0.26, d + 0.26, h - 0.3, h
    for (a, b, cc) in (((gx0, gy0, gd1), (gx1, gy1, gd1), 'd'),):
        pass
    quad_d(P, gx0, gy0, gx1, gy1, gd1, WHITE, mat='glass')
    quad_v(P, gx0, gd0, gx1, gd1, gy1, WHITE, mat='glass')
    P.oriented('glass', [[gx0, gy0, gd0], [gx0, gy0, gd1], [gx0, gy1, gd1], [gx0, gy1, gd0]], [[0, 1, 2], [0, 2, 3]], [-1, 0, 0], WHITE)
    P.oriented('glass', [[gx1, gy0, gd0], [gx1, gy0, gd1], [gx1, gy1, gd1], [gx1, gy1, gd0]], [[0, 1, 2], [0, 2, 3]], [1, 0, 0], WHITE)
    # Lichtleiste innen oben
    quad_v(P, gx0 + 0.02, gd0 + 0.02, gx1 - 0.02, gd0 + 0.06, gy1 - 0.01, WHITE, mat='glow_warm', facing=-1)
    quad_v(P, gx0 + 0.01, gd0 + 0.01, gx1 - 0.01, gd1 - 0.01, gy0 + 0.002, '#3A1E24' if items == 'jewel' else '#E8E2DA')
    rng = rng or np.random.default_rng(1)
    if items == 'jewel':
        n = int(w / 0.12)
        for i in range(n):
            x = gx0 + 0.07 + i * (w - 0.14) / max(1, n - 1)
            dd = d + float(rng.uniform(-0.15, 0.15))
            c = GOLD if rng.random() < 0.6 else SILVER
            if rng.random() < 0.4:                           # Ring auf Kissen
                blk(P, [x - 0.025, gy0, dd - 0.02], [x + 0.025, gy0 + 0.02, dd + 0.02], '#1E1E2A')
                P.cyl('trim', [x, gy0 + 0.035, dd - 0.004], [x, gy0 + 0.035, dd + 0.004], 0.016, seg=6, c=c)
            elif rng.random() < 0.5:                         # Halskette an Buste
                frustum(P, 'interior', gy0, gy0 + 0.16, 0.08, 0.05, 0.05, 0.04, '#2A2026', uc=x, dc=dd)
                quad_d(P, x - 0.03, gy0 + 0.08, x + 0.03, gy0 + 0.14, dd + 0.021, c)
            else:                                            # Uhr
                cylv(P, 'trim', x, dd, gy0, gy0 + 0.06, 0.02, '#2A2026', seg=6)
                P.cyl('trim', [x, gy0 + 0.08, dd - 0.006], [x, gy0 + 0.08, dd + 0.006], 0.022, seg=8, c=c)


def bay_jewellery(P, rng):
    blk(P, [-1.25, 0.0, -1.5], [1.25, 1.9, -1.45], '#3A2A26', skip=('-v',))      # Rueckwand der Auslage (dunkles Holz)
    vitrine(P, -0.62, -0.6, 0.9, rng=rng)
    vitrine(P, 0.62, -0.6, 0.9, rng=rng)
    blk(P, [-1.25, 1.85, -1.5], [1.25, 1.9, -0.3], '#3A2A26', skip=())
    quad_v(P, -1.2, -1.45, 1.2, -0.35, 1.849, WHITE, mat='glow_warm', facing=-1)


def back_jewellery(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.5, 0.05], '#2E2420', skip=('-v',))
    for i in range(3):
        u = -1.33 + 1.33 * i
        blk(P, [u - 0.5, 1.0, 0.05], [u + 0.5, 1.03, 0.35], '#3A2A26', skip=())
        quad_d(P, u - 0.45, 1.05, u + 0.45, 1.7, 0.051, '#4A3A30')
        quad_v(P, u - 0.45, 0.06, u + 0.45, 0.3, 1.7, WHITE, mat='glow_warm', facing=-1)
        for k in range(5):
            x = u - 0.36 + k * 0.18
            P.cyl('trim', [x, 1.12, 0.12], [x, 1.12, 0.13], 0.02, seg=6, c=GOLD if (k + i) % 2 else SILVER)
    vitrine(P, 0.0, 0.75, 2.4, h=1.0, rng=rng)


def mid_jewellery(P, rng):
    vitrine(P, 0.0, 0.0, 1.6, h=1.0, rng=rng)
    for s in (-1, 1):
        stool(P, s * 0.45, 0.6, '#3A2A26', h=0.65)


def bay_optician(P, rng):
    for u in (-0.75, 0.75):
        cylv(P, 'metal', u, -0.7, 0.0, 0.02, 0.18, DARK, seg=8)
        blk(P, [u - 0.18, 0.02, -0.73], [u + 0.18, 1.75, -0.67], '#F2F0EC', skip=())
        for r in range(6):
            glasses(P, u, 0.45 + r * 0.22, -0.67, ['#1E1E1E', '#6B4E3A', '#C8A050', '#7A1F24', '#1F2A44'][int(rng.integers(5))])
    poster(P, -0.3, 0.3, 1.0, 1.9, -1.3, '#E8E2D4', rng=rng)


def back_optician(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.5, 0.03], '#F2F0EC', skip=('-v',))
    for r in range(6):
        y = 0.9 + r * 0.24
        blk(P, [-1.9, y - 0.05, 0.03], [1.9, y - 0.035, 0.12], '#D8D4CC', skip=())
        x = -1.82
        while x < 1.85:
            glasses(P, x, y, 0.06, ['#1E1E1E', '#6B4E3A', '#C8A050', '#7A1F24', '#1F2A44', '#8C969C'][int(rng.integers(6))])
            x += 0.17
    blk(P, [-1.9, 0.0, 0.03], [1.9, 0.8, 0.5], '#E8E2DA', skip=('-v',))


def mid_optician(P, rng):
    table(P, 0.0, 0.0, 1.2, 0.7, 0.75, '#E8E2DA')
    blk(P, [-0.15, 0.75, -0.05], [0.15, 1.15, -0.03], '#B8BCBE', mat='metal', skip=())
    for s in (-1, 1):
        stool(P, s * 0.35, 0.55, '#3A3D40', h=0.5)


def bay_pharmacy(P, rng):
    for k, (u, h) in enumerate(((-0.75, 0.3), (0.0, 0.55), (0.75, 0.3))):
        blk(P, [u - 0.32, 0.0, -0.95], [u + 0.32, h, -0.35], WHITE, skip=('-v',))
        for i in range(3):
            for j in range(2):
                blk(P, [u - 0.28 + i * 0.19, h, -0.85 + j * 0.25], [u - 0.12 + i * 0.19, h + 0.2, -0.7 + j * 0.25],
                    PRODUCTS[int(rng.integers(len(PRODUCTS)))], skip=('-v',))
    poster(P, -1.1, -0.1, 0.9, 2.3, -1.4, '#2E8B57', rng=rng)
    # gruenes Kreuz (leuchtend), abgehaengt
    for (lo, hi) in (((0.45, 1.9, -1.2), (0.65, 2.5, -1.16)), ((0.25, 2.1, -1.2), (0.85, 2.3, -1.16))):
        blk(P, lo, hi, WHITE, mat='glow_33CC66', skip=())


def back_pharmacy(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.03], WHITE, skip=('-v',))
    # Schubladenschrank unten, Regale oben
    blk(P, [-1.95, 0.0, 0.03], [1.95, 0.9, 0.55], '#F4F2EE', skip=('-v',))
    for i in range(12):
        for j in range(3):
            u = -1.9 + i * 0.32; y = 0.05 + j * 0.28
            quad_d(P, u + 0.01, y + 0.01, u + 0.3, y + 0.26, 0.551, '#ECEAE4')
            blk(P, [u + 0.12, y + 0.18, 0.551], [u + 0.19, y + 0.2, 0.57], STEEL, skip=())
    product_grid(P, -1.95, 1.95, 1.0, 2.5, 0.35, rng, rows=5, boxes=False)


def mid_pharmacy(P, rng):
    blk(P, [-1.1, 0.0, -0.35], [1.1, 1.0, 0.35], WHITE, skip=('-v',))
    blk(P, [-1.15, 1.0, -0.4], [1.15, 1.04, 0.4], '#2E8B57', skip=())
    for s in (-0.6, 0.6):
        blk(P, [s - 0.2, 1.04, 0.1], [s + 0.2, 1.3, 0.13], DARK, skip=())


def bay_bakery(P, rng):
    blk(P, [-1.2, 0.0, -1.1], [1.2, 0.7, -0.3], '#8A5A36', skip=('-v',))
    for r in range(2):
        for i in range(6):
            u = -1.0 + i * 0.4
            bread(P, u, 0.7, -0.95 + r * 0.35, ['#B8783A', '#9A5A2A', '#D8A060', '#7A4A22'][int(rng.integers(4))])
    # Etagere mit Torten
    cylv(P, 'trim', 0.75, -1.25, 0.0, 1.1, 0.03, CHROME, seg=5, caps=False)
    for y, r in ((1.1, 0.22),):
        cylv(P, 'trim', 0.75, -1.25, y, y + 0.02, r, WHITE, seg=10)
        cylv(P, 'interior', 0.75, -1.25, y + 0.02, y + 0.14, r * 0.85, ['#F0D8C0', '#6B3A22', '#E85A8A'][int(rng.integers(3))], seg=10)


def back_bakery(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.5, 0.03], '#D8C0A0', skip=('-v',))
    for r in range(4):
        y = 1.0 + r * 0.36
        V = [[-1.9, y, 0.03], [1.9, y, 0.03], [1.9, y - 0.06, 0.45], [-1.9, y - 0.06, 0.45]]
        P.oriented('trim', V, [[0, 1, 2], [0, 2, 3]], [0, 1, 0.15], WOOD)
        for i in range(12):
            bread(P, -1.75 + i * 0.32, y - 0.04, 0.25, ['#B8783A', '#9A5A2A', '#D8A060', '#7A4A22', '#C89050'][int(rng.integers(5))], lng=0.26)
    blk(P, [-1.95, 0.0, 0.03], [1.95, 0.9, 0.6], '#6B4E3A', skip=('-v',))
    blk(P, [0.9, 0.9, 0.1], [1.4, 1.35, 0.5], STEEL, mat='metal', skip=('-v',))     # Kaffeemaschine


def mid_bakery(P, rng):
    blk(P, [-1.2, 0.0, -0.4], [1.2, 0.85, 0.4], '#6B4E3A', skip=('-v',))
    # Kuehltheke mit Glas
    quad_d(P, -1.15, 0.85, 1.15, 1.3, 0.38, WHITE, mat='glass')
    quad_v(P, -1.15, -0.1, 1.15, 0.38, 1.3, WHITE, mat='glass')
    quad_v(P, -1.15, -0.35, 1.15, 0.38, 0.86, '#E8E2DA')
    for i in range(10):
        u = -1.05 + i * 0.23
        blk(P, [u, 0.86, -0.05], [u + 0.15, 0.93, 0.1], ['#F0D8C0', '#6B3A22', '#E85A8A', '#D8A060', '#F2F2EE'][int(rng.integers(5))])
    quad_v(P, -1.1, -0.05, 1.1, 0.0, 1.29, WHITE, mat='glow_warm', facing=-1)
    blk(P, [0.7, 0.85, -0.35], [1.0, 1.05, -0.1], DARK, skip=('-v',))           # Kasse


def bay_cafe(P, rng):
    if rng.random() < 0.5:
        blk(P, [-1.25, 1.02, -0.6], [1.25, 1.06, -0.25], WOOD_L, skip=())        # Fensterbar
        for s in (-0.8, 0.0, 0.8):
            stool(P, s, -0.9, ['#3A3D40', '#6B4E3A', '#1E1E1E'][int(rng.integers(3))])
        for i in range(2):
            u = float(rng.uniform(-1.0, 1.0))
            cylv(P, 'interior', u, -0.45, 1.06, 1.16, 0.04, WHITE, seg=6)
    else:
        for s in (-0.6, 0.6):
            table(P, s, -0.8, 0.6, 0.6, 0.75, ['#E8E2DA', WOOD_L, '#2E3033'][int(rng.integers(3))], round_=rng.random() < 0.5)
            chair(P, s - 0.38, -0.8, ['#6B4E3A', '#1E1E1E', '#8E3B35'][int(rng.integers(3))], face=1)
            chair(P, s + 0.38, -0.8, ['#6B4E3A', '#1E1E1E', '#8E3B35'][int(rng.integers(3))], face=1)


def back_cafe(P, rng, fast=False):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.03], '#D8C8B0' if not fast else '#E8E2DA', skip=('-v',))
    blk(P, [-1.95, 0.0, 0.5], [1.95, 1.05, 0.95], '#5A3A26' if not fast else '#C8102E', skip=('-v',))     # Theke
    blk(P, [-2.0, 1.05, 0.45], [2.0, 1.1, 1.0], '#E8E2DA' if not fast else '#3A3D40', skip=())
    blk(P, [-1.6, 1.1, 0.6], [-1.0, 1.55, 0.85], STEEL, mat='metal', skip=('-v',))        # Espressomaschine / Kasse
    for r in range(3):
        y = 1.45 + r * 0.32
        blk(P, [-1.9, y, 0.03], [1.9, y + 0.025, 0.3], WOOD_D, skip=())
        x = -1.85
        while x < 1.85:
            c = ['#F2F2EE', '#3A6B3A', '#8E3B35', '#C8A050', '#1E1E1E'][int(rng.integers(5))]
            if rng.random() < 0.5:
                blk(P, [x + 0.01, y + 0.025, 0.12], [x + 0.07, y + 0.25, 0.18], c, skip=('-v', '-d'))
            else:
                blk(P, [x, y + 0.025, 0.11], [x + 0.08, y + 0.1, 0.19], WHITE, skip=('-v', '-d'))
            x += 0.14
    # Menuetafeln
    for i in range(3 if fast else 2):
        u = -1.3 + i * (1.3 if fast else 2.6)
        if fast:
            blk(P, [u - 0.6, 1.95, 0.03], [u + 0.6, 2.5, 0.08], DARK, skip=())
            quad_d(P, u - 0.56, 1.99, u + 0.56, 2.46, 0.081, WHITE, mat='glow')
            quad_d(P, u - 0.5, 2.2, u - 0.05, 2.42, 0.082, PRODUCTS[int(rng.integers(len(PRODUCTS)))])
        else:
            blk(P, [u - 0.5, 0.0, 0.03], [u + 0.5, 0.0, 0.05], DARK, skip=()) if False else None
    if not fast:
        blk(P, [0.4, 1.3, 0.03], [1.6, 2.0, 0.06], '#22262A', skip=())
        for i in range(6):
            quad_d(P, 0.5, 1.88 - i * 0.09, 1.1 + 0.3 * rng.random(), 1.9 - i * 0.09 + 0.022, 0.061, '#F2F0EA')


def mid_cafe(P, rng):
    for s in (-0.65, 0.65):
        table(P, s, 0.0, 0.7, 0.7, 0.75, ['#E8E2DA', WOOD_L, '#2E3033'][int(rng.integers(3))], round_=rng.random() < 0.4)
        c = ['#6B4E3A', '#1E1E1E', '#8E3B35', '#3A5A6A'][int(rng.integers(4))]
        chair(P, s, -0.48, c, face=1)
        chair(P, s, 0.48, c, face=-1)


def bay_fastfood(P, rng):
    blk(P, [-0.3, 0.0, -0.9], [0.3, 0.1, -0.6], DARK, skip=('-v',))
    blk(P, [-0.12, 0.1, -0.78], [0.12, 1.75, -0.72], DARK, skip=())
    quad_d(P, -0.1, 0.95, 0.1, 1.68, -0.719, WHITE, mat='glow')
    for s in (-0.85, 0.85):
        stool(P, s, -0.8, '#C8102E')
    blk(P, [-1.25, 1.02, -0.5], [-0.4, 1.06, -0.25], '#3A3D40', skip=())
    blk(P, [0.4, 1.02, -0.5], [1.25, 1.06, -0.25], '#3A3D40', skip=())


def bay_bank(P, rng):
    blk(P, [-0.9, 0.3, -1.0], [0.4, 2.3, -0.92], DARK, skip=())
    quad_d(P, -0.86, 0.34, 0.36, 2.26, -0.919, WHITE, mat='glow')
    quad_d(P, -0.8, 1.4, 0.3, 2.2, -0.918, PRODUCTS[int(rng.integers(len(PRODUCTS)))])
    quad_d(P, -0.8, 0.6, 0.0, 0.9, -0.918, '#1E1E1E')
    plant(P, 0.85, -0.8, rng=rng)


def back_bank(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.04], '#D8D4CC', skip=('-v',))
    for i in range(3):
        u = -1.3 + i * 1.3
        blk(P, [u - 0.4, 0.0, 0.04], [u + 0.4, 1.8, 0.14], '#B8BCBE', mat='metal', skip=('-v',))
        quad_d(P, u - 0.17, 1.15, u + 0.17, 1.4, 0.141, '#1F4FA0', mat='glow')
        blk(P, [u - 0.25, 0.95, 0.14], [u + 0.25, 1.05, 0.3], DARK, skip=())
        blk(P, [u - 0.4, 1.8, 0.04], [u + 0.4, 2.0, 0.2], '#C8102E', skip=())


def mid_bank(P, rng):
    table(P, 0.0, 0.0, 1.4, 0.75, 0.75, WHITE)
    blk(P, [-0.25, 0.75, 0.1], [0.25, 1.05, 0.13], DARK, skip=())
    for s in (-0.35, 0.35):
        chair(P, s, -0.55, '#3A3D40', face=1)


def bay_phone(P, rng):
    blk(P, [-1.1, 0.0, -0.95], [1.1, 0.75, -0.3], WHITE, skip=('-v',))
    scr = ['#4AA8D8', '#E86A1A', '#8CC63F', '#7A3A8A', '#F2C500', '#E85A8A']
    for row, (dd, hh) in enumerate(((-0.82, 0.0), (-0.48, 0.0))):
        for i in range(6):
            u = -0.9 + i * 0.36 + (0.1 if row else 0.0)
            if u > 0.98:
                continue
            blk(P, [u - 0.04, 0.75, dd - 0.02], [u + 0.04, 0.83, dd + 0.02], STEEL, skip=())
            blk(P, [u - 0.042, 0.8, dd + 0.02], [u + 0.042, 0.98, dd + 0.03], '#1E1E1E', skip=())
            quad_d(P, u - 0.036, 0.81, u + 0.036, 0.97, dd + 0.031, scr[int(rng.integers(len(scr)))], mat='glow')
    for u in (-0.75, 0.75):                                          # Tablets auf Staendern
        blk(P, [u - 0.05, 0.75, -0.66], [u + 0.05, 0.9, -0.62], STEEL, skip=())
        blk(P, [u - 0.16, 0.86, -0.62], [u + 0.16, 1.1, -0.6], '#2A2A2C', skip=())
        quad_d(P, u - 0.145, 0.875, u + 0.145, 1.085, -0.599, scr[int(rng.integers(len(scr)))], mat='glow')
    for i in range(5):                                               # Uhren auf Kissen
        u = -0.4 + i * 0.2
        blk(P, [u - 0.05, 0.75, -0.38], [u + 0.05, 0.8, -0.33], '#3A3D40', skip=('-v',))
        blk(P, [u - 0.02, 0.8, -0.37], [u + 0.02, 0.85, -0.34], '#1E1E1E', skip=())
    blk(P, [-1.1, 1.05, -1.42], [1.1, 2.6, -1.36], DARK, skip=())       # Leuchtplakat
    quad_d(P, -1.06, 1.09, 1.06, 2.56, -1.359, '#1F2A44', mat='glow')
    quad_d(P, -0.9, 1.75, 0.9, 2.4, -1.358, PRODUCTS[int(rng.integers(len(PRODUCTS)))])
    quad_d(P, -0.7, 1.25, 0.7, 1.4, -1.358, '#F4F2EE')


def back_phone(P, rng):
    # Technikladen: grosse Videowand ueber die Rueckwand, darunter niedriges Zubehoerregal (keine Buecherwand-Optik)
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.03], '#F2F0EC', skip=('-v',))
    blk(P, [-1.9, 1.05, 0.03], [1.9, 2.45, 0.09], DARK, skip=())
    scr = [('#1F3F7A', '#4AA8D8'), ('#2A1F4A', '#E85A8A'), ('#16303A', '#8CC63F')][int(rng.integers(3))]
    quad_d(P, -1.86, 1.09, 1.86, 2.41, 0.091, scr[0], mat='glow')
    quad_d(P, -0.9, 1.45, 0.9, 2.15, 0.092, scr[1], mat='glow')
    quad_d(P, -0.5, 1.22, 0.5, 1.3, 0.093, '#F4F2EE', mat='glow')
    blk(P, [-1.9, 0.0, 0.03], [1.9, 0.95, 0.38], '#E8E6E2', skip=('-v',))
    product_grid(P, -1.85, 1.85, 0.95, 1.13, 0.36, rng, rows=1, boxes=True, depth=0.06, wmin=0.08, wmax=0.16)


def mid_phone(P, rng):
    table(P, 0.0, 0.0, 2.0, 0.9, 0.85, WOOD_L)
    for i in range(6):
        u = -0.8 + i * 0.32
        if i % 3 == 2:
            blk(P, [u - 0.12, 0.85, -0.1], [u + 0.12, 0.87, 0.08], '#B8BCBE', skip=())
            V = [[u - 0.12, 0.87, -0.1], [u + 0.12, 0.87, -0.1], [u + 0.12, 1.03, -0.16], [u - 0.12, 1.03, -0.16]]
            P.oriented('trim', V, [[0, 1, 2], [0, 2, 3]], [0, 0.3, 1], C('#1E1E1E'))
        else:
            blk(P, [u - 0.04, 0.85, -0.02], [u + 0.04, 0.86, 0.13], '#1E1E1E', skip=())


def book_row(P, u0, u1, y, d, rng, hmax=0.3):
    x = u0
    while x < u1 - 0.02:
        w = float(rng.uniform(0.025, 0.06)); h = hmax * float(rng.uniform(0.65, 1.0))
        quad_d(P, x, y, min(x + w - 0.003, u1), y + h, d, BOOKS[int(rng.integers(len(BOOKS)))])
        x += w


def bay_books(P, rng):
    table(P, 0.0, -0.75, 2.0, 0.8, 0.72, WOOD)
    for i in range(7):
        u = -0.85 + i * 0.28
        n = int(rng.integers(2, 6))
        for k in range(n):
            blk(P, [u - 0.1, 0.72 + k * 0.035, -0.9], [u + 0.1, 0.752 + k * 0.035, -0.62], BOOKS[int(rng.integers(len(BOOKS)))], skip=())
    for i in range(4):
        u = -0.75 + i * 0.5
        V = [[u - 0.08, 0.72 + 0.17 * 5 / 5, -0.5], [u + 0.08, 0.72, -0.5]]
        blk(P, [u - 0.08, 0.72, -0.48], [u + 0.08, 0.96, -0.46], BOOKS[int(rng.integers(len(BOOKS)))], skip=('-v',))
    poster(P, 0.3, 1.2, 1.1, 2.4, -1.5, '#E8E2D4', rng=rng)


def back_books(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.32], WOOD_D, skip=('-v', '+d'))
    quad_d(P, -1.98, 0.0, 1.98, 2.58, 0.02, '#4A3020')
    for r in range(7):
        y = 0.1 + r * 0.35
        blk(P, [-1.98, y - 0.02, 0.02], [1.98, y, 0.32], WOOD, skip=())
        book_row(P, -1.95, 1.95, y, 0.28, rng, hmax=0.28)


def mid_books(P, rng):
    table(P, 0.0, 0.0, 1.8, 1.0, 0.72, WOOD_L)
    for i in range(6):
        for j in range(2):
            u = -0.75 + i * 0.3; d = -0.25 + j * 0.5
            n = int(rng.integers(2, 7))
            blk(P, [u - 0.1, 0.72, d - 0.14], [u + 0.1, 0.72 + n * 0.03, d + 0.14], BOOKS[int(rng.integers(len(BOOKS)))], skip=('-v',))


def gondola(P, u, rng, length=2.4, h=1.6, palette=PRODUCTS):
    """Regalreihe senkrecht zum Fenster (laengs d), zweiseitig."""
    d0, d1 = -length / 2, length / 2
    blk(P, [u - 0.4, 0.0, d0], [u + 0.4, 0.12, d1], DARK, skip=('-v',))
    blk(P, [u - 0.02, 0.12, d0], [u + 0.02, h, d1], '#D8D4CC', skip=())
    for side in (-1, 1):
        for r in range(4):
            y = 0.15 + r * (h - 0.2) / 4
            blk(P, [min(u, u + side * 0.4), y, d0], [max(u, u + side * 0.4), y + 0.02, d1], '#F2F0EC', skip=())
            x = d0 + 0.03
            while x < d1 - 0.05:
                w = float(rng.uniform(0.06, 0.16)); hh = (h - 0.2) / 4 * float(rng.uniform(0.5, 0.85))
                uu = u + side * 0.36
                V = [[uu, y + 0.02, x], [uu, y + 0.02, min(x + w - 0.01, d1)], [uu, y + 0.02 + hh, min(x + w - 0.01, d1)], [uu, y + 0.02 + hh, x]]
                P.oriented('trim', V, [[0, 1, 2], [0, 2, 3]], [side, 0, 0], C(palette[int(rng.integers(len(palette)))]))
                x += w
    blk(P, [u - 0.42, h, d0 - 0.02], [u + 0.42, h + 0.25, d0 + 0.02], palette[int(rng.integers(len(palette)))], skip=())


def bay_drugstore(P, rng, accent='#C8102E'):
    for u in (-0.7, 0.7):
        for k in range(3):
            blk(P, [u - 0.4 + k * 0.1, k * 0.25, -1.0 + k * 0.1], [u + 0.4 - k * 0.1, (k + 1) * 0.25, -0.4 - k * 0.1], WHITE if k % 2 else accent,
                skip=('-v',))
        for i in range(4):
            cylv(P, 'trim', u - 0.18 + i * 0.12, -0.7, 0.75, 0.95, 0.035, PRODUCTS[int(rng.integers(len(PRODUCTS)))], seg=6)
    poster(P, -0.5, 0.5, 1.1, 2.5, -1.45, accent, rng=rng)


def back_drugstore(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.4, 0.03], WHITE, skip=('-v',))
    product_grid(P, -1.95, 1.95, 0.15, 2.2, 0.35, rng, rows=6)


def mid_drugstore(P, rng):
    gondola(P, -0.65, rng, length=1.4)
    gondola(P, 0.65, rng, length=1.4)


def bay_department(P, rng):
    # geschlossener Schaukasten: Rueckwand, Seitenwaende, Spots, Puppen auf Podesten
    col = ['#E8E2DA', '#D8D4CC', '#2E3033', '#B8A07A', '#8E3B35', '#1F2A44'][int(rng.integers(6))]
    blk(P, [-1.3, 0.0, -1.45], [1.3, 3.0, -1.4], col, skip=('-v',))
    for s in (-1, 1):
        blk(P, [s * 1.3 - (0.04 if s > 0 else 0), 0.0, -1.45], [s * 1.3 + (0.0 if s > 0 else 0.04), 3.0, -0.05], col, skip=('-v',))
    blk(P, [-1.25, 0.0, -1.4], [1.25, 0.18, -0.15], '#F2F0EC', skip=('-v',))
    for i, u in enumerate((-0.65, 0.05, 0.75)):
        mannequin(P, u, -0.85 + 0.2 * (i % 2), rng, female=i != 1, body='#F2F0EC', outfit=_outfit(rng))
    for u in (-0.8, 0.0, 0.8):
        P.cyl('metal', [u, 2.98, -0.4], [u, 2.85, -0.55], 0.04, seg=6, c=DARK)
        P.cyl('glow_warm', [u, 2.85, -0.55], [u, 2.845, -0.56], 0.035, seg=6, c=WHITE)


def bay_kiosk(P, rng):
    # Zeitschriftenregal (schraeg) + Getraenkekuehlschrank
    for r in range(4):
        y = 0.3 + r * 0.38
        V = [[-1.2, y, -0.9], [0.2, y, -0.9], [0.2, y + 0.32, -1.0], [-1.2, y + 0.32, -1.0]]
        P.oriented('trim', V, [[0, 1, 2], [0, 2, 3]], [0, 0.3, 1], C('#3A3D40'))
        x = -1.15
        while x < 0.12:
            quad_d(P, x, y + 0.02, x + 0.21, y + 0.29, -0.9 + 0.001, PRODUCTS[int(rng.integers(len(PRODUCTS)))])
            x += 0.23
    blk(P, [0.4, 0.0, -1.4], [1.2, 2.0, -0.8], '#C8102E', skip=('-v',))
    quad_d(P, 0.46, 0.15, 1.14, 1.9, -0.799, WHITE, mat='glow')
    for r in range(5):
        for i in range(6):
            cylv(P, 'trim', 0.52 + i * 0.11, -0.83, 0.2 + r * 0.34, 0.42 + r * 0.34, 0.03,
                 PRODUCTS[int(rng.integers(len(PRODUCTS)))], seg=5, caps=False)


def back_kiosk(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.4, 0.03], '#E8E2DA', skip=('-v',))
    product_grid(P, -1.95, 1.95, 1.2, 2.3, 0.3, rng, rows=4, wmin=0.04, wmax=0.08)
    blk(P, [-1.95, 0.0, 0.6], [1.95, 1.0, 1.0], WOOD_D, skip=('-v',))
    for i in range(10):
        blk(P, [-1.8 + i * 0.36, 1.0, 0.65], [-1.6 + i * 0.36, 1.12, 0.95], PRODUCTS[int(rng.integers(len(PRODUCTS)))], skip=('-v',))


def bay_salon(P, rng):
    plant(P, -0.8, -0.8, rng=rng)
    poster(P, -0.2, 1.1, 1.0, 2.4, -1.4, '#E8E2D4', rng=rng)
    blk(P, [0.0, 0.0, -0.9], [0.9, 0.45, -0.45], '#3A3D40', skip=('-v',))


def back_salon(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.03], '#ECE6DE', skip=('-v',))
    for u in (-1.3, 0.0, 1.3):
        blk(P, [u - 0.4, 0.9, 0.03], [u + 0.4, 2.0, 0.05], '#B8BCBE', mat='spandrel', skip=())
        blk(P, [u - 0.45, 0.8, 0.03], [u + 0.45, 0.85, 0.35], WHITE, skip=())
        # Friseurstuhl
        cylv(P, 'metal', u, 0.75, 0.0, 0.03, 0.25, DARK, seg=8)
        cylv(P, 'metal', u, 0.75, 0.03, 0.45, 0.04, CHROME, seg=5, caps=False)
        blk(P, [u - 0.25, 0.45, 0.55], [u + 0.25, 0.58, 0.95], '#1E1E1E', skip=())
        blk(P, [u - 0.25, 0.58, 0.5], [u + 0.25, 1.15, 0.6], '#1E1E1E', skip=('-v',))


def bay_office(P, rng):
    for u in (-0.65, 0.65):
        poster(P, u - 0.5, u + 0.5, 0.8, 2.2, -1.2, PRODUCTS[int(rng.integers(len(PRODUCTS)))], rng=rng)
        blk(P, [u - 0.03, 0.0, -1.24], [u + 0.03, 0.8, -1.2], DARK, skip=())
    plant(P, 0.0, -0.5, rng=rng)


def mid_office(P, rng):
    table(P, 0.0, 0.0, 1.6, 0.8, 0.74, WHITE)
    blk(P, [-0.3, 0.74, 0.15], [0.3, 1.1, 0.18], DARK, skip=())
    quad_d(P, -0.28, 0.77, 0.28, 1.08, 0.181, '#2A5A9A', mat='glow')
    chair(P, 0.0, 0.55, '#3A3D40', face=-1)
    for s in (-0.4, 0.4):
        chair(P, s, -0.6, '#5A6B7A', face=1)


def bay_home(P, rng):
    blk(P, [-1.2, 0.0, -1.3], [1.2, 0.08, -0.3], '#D8D0C0', skip=('-v',))
    if rng.random() < 0.5:
        # Sofa
        c = FABRICS[int(rng.integers(len(FABRICS)))]
        blk(P, [-1.0, 0.08, -1.25], [0.6, 0.5, -0.45], c, mat='interior', skip=('-v',))
        blk(P, [-1.0, 0.5, -1.25], [0.6, 0.9, -1.05], c, mat='interior', skip=('-v',))
        table(P, 0.95, -0.7, 0.45, 0.45, 0.55, WOOD_L, round_=True)
    else:
        table(P, -0.3, -0.8, 1.4, 0.8, 0.75, WOOD)
        for i in range(4):
            cylv(P, 'trim', -0.8 + i * 0.32, -0.8, 0.75, 0.95 + 0.15 * rng.random(), 0.06, PRODUCTS[int(rng.integers(len(PRODUCTS)))], seg=6)
    # Stehleuchte
    cylv(P, 'metal', 1.0, -1.2, 0.0, 1.6, 0.015, DARK, seg=4, caps=False)
    frustum(P, 'glow_warm', 1.5, 1.75, 0.38, 0.38, 0.24, 0.24, WHITE, uc=1.0, dc=-1.2, bottom=True)


def back_home(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.6, 0.03], '#E8E2DA', skip=('-v',))
    product_grid(P, -1.95, 1.95, 0.3, 2.2, 0.4, rng, rows=4, boxes=True, depth=0.2, palette=PRODUCTS + FABRICS)


def bay_florist(P, rng):
    for k, (h, d0, d1) in enumerate(((0.0, -0.6, -0.2), (0.3, -1.0, -0.6), (0.6, -1.4, -1.0))):
        if h:
            blk(P, [-1.2, 0.0, d0], [1.2, h, d1], WOOD, skip=('-v',))
        for i in range(5):
            u = -1.0 + i * 0.5
            cylv(P, 'trim', u, (d0 + d1) / 2, h, h + 0.25, 0.12, ['#7A7A78', '#C8B8A0', '#3A3D40'][int(rng.integers(3))], seg=6)
            blob(P, 'interior', [u, h + 0.45, (d0 + d1) / 2], [0.18, 0.22, 0.16], ['#3E6B3A', '#4E7A3E'][i % 2], seg=5, rings=3)
            for f in range(2):
                blob(P, 'interior', [u + (f - 0.5) * 0.09, h + 0.58 + 0.05 * f, (d0 + d1) / 2 + 0.08],
                     [0.06, 0.05, 0.06], ['#E85A8A', '#F2C500', '#C8102E', '#F2F2EE', '#7A3A8A'][int(rng.integers(5))], seg=4, rings=2)


def bay_vacant(P, rng):
    # Schaufenster mit Papier abgeklebt (leicht vergilbt), Abstand 5 cm hinter dem Glas
    quad_d(P, -1.3, 0.0, 1.3, 2.4, -0.08, '#ECE6D6')
    quad_d(P, -1.3, 2.4, 1.3, 3.2, -0.6, '#3A3836')


def bay_generic(P, rng):
    for u, h in ((-0.7, 0.45), (0.1, 0.75), (0.85, 0.3)):
        blk(P, [u - 0.28, 0.0, -1.0], [u + 0.28, h, -0.45], WHITE, skip=('-v',))
        c = PRODUCTS[int(rng.integers(len(PRODUCTS)))]
        if rng.random() < 0.5:
            blk(P, [u - 0.12, h, -0.82], [u + 0.12, h + 0.3, -0.62], c, skip=('-v',))
        else:
            cylv(P, 'trim', u, -0.72, h, h + 0.32, 0.09, c, seg=8)
    poster(P, -1.15, -0.25, 0.9, 2.4, -1.45, PRODUCTS[int(rng.integers(len(PRODUCTS)))], rng=rng)


def back_generic(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.5, 0.03], '#EAE6DE', skip=('-v',))
    product_grid(P, -1.95, 1.95, 0.4, 2.2, 0.35, rng, rows=4, boxes=True, depth=0.15)


def back_supermarket(P, rng):
    blk(P, [-2.0, 0.0, 0.0], [2.0, 2.2, 0.03], '#E8E2DA', skip=('-v',))
    blk(P, [-1.95, 0.0, 0.03], [1.95, 2.0, 0.75], '#D8D8D4', mat='metal', skip=('-v',))
    quad_d(P, -1.9, 0.1, 1.9, 1.95, 0.751, WHITE, mat='glow')
    product_grid(P, -1.88, 1.88, 0.15, 1.9, 0.7, rng, rows=5)


def bay_toys(P, rng):
    bay_generic(P, rng)
    for i in range(3):
        blob(P, 'interior', [-0.6 + i * 0.6, 0.95, -0.8], [0.14, 0.16, 0.12], ['#C89050', '#E85A8A', '#4AA8D8'][i], seg=6, rings=4)


GEN = {
    'fashion': (bay_fashion, back_fashion, mid_fashion), 'sports': (lambda P, r: bay_fashion(P, r, True), back_shoes, mid_fashion),
    'shoes': (bay_shoes, back_shoes, mid_shoes), 'jewellery': (bay_jewellery, back_jewellery, mid_jewellery),
    'optician': (bay_optician, back_optician, mid_optician), 'pharmacy': (bay_pharmacy, back_pharmacy, mid_pharmacy),
    'bakery': (bay_bakery, back_bakery, mid_bakery), 'cafe': (bay_cafe, back_cafe, mid_cafe),
    'fastfood': (bay_fastfood, lambda P, r: back_cafe(P, r, True), mid_cafe), 'bank': (bay_bank, back_bank, mid_bank),
    'phone': (bay_phone, back_phone, mid_phone), 'books': (bay_books, back_books, mid_books),
    'drugstore': (bay_drugstore, back_drugstore, mid_drugstore), 'department': (bay_department, back_fashion, mid_fashion),
    'kiosk': (bay_kiosk, back_kiosk, mid_drugstore), 'salon': (bay_salon, back_salon, mid_office),
    'lobby': (bay_office, back_generic, mid_office), 'office': (bay_office, back_generic, mid_office),
    'home': (bay_home, back_home, mid_office), 'florist': (bay_florist, back_generic, mid_office),
    'vacant': (bay_vacant, None, None), 'generic': (bay_generic, back_generic, mid_drugstore),
    'supermarket': (bay_drugstore, back_supermarket, mid_drugstore), 'toys': (bay_toys, back_generic, mid_drugstore),
}


def display_proto(key):
    """dsp_<art>_<bay|back|mid><k>"""
    if key in PROTOS:
        return PROTOS[key]
    if key in ('dsp_ceil', 'dsp_ceil_warm', 'dsp_track'):
        return ceiling_proto(key)
    if key in ('dsp_tbl_out', 'dsp_parasol', 'dsp_aframe'):
        return outdoor_proto(key)
    _, kind, part = key.split('_', 2)
    slot = part.rstrip('0123456789'); var = int(part[len(slot):] or 0)
    rng = rng_of('display', kind, slot, var)
    w0 = {'bay': 2.6, 'back': 4.0, 'mid': 2.4}[slot]
    P = Proto(key, 'room', w0, 1.0)
    g = GEN[kind][{'bay': 0, 'back': 1, 'mid': 2}[slot]]
    if g is not None:
        g(P, rng)
    if not P.parts:
        blk(P, [-0.01, 0, -0.01], [0.01, 0.01, 0.01], WHITE)
    PROTOS[key] = P.freeze()
    return PROTOS[key]


def get(key):
    if key.startswith('sf'):
        return shopfront_proto(key)
    return display_proto(key)

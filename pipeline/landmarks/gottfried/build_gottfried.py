# Gottfried's Feinkiosk im Leibnizhaus – Holzmarkt 6 (LoD2 ID_8399e9b2), Fassaden Holzmarkt (Ost) + Schloßstraße (Süd).
# Läuft IM Blender (über das Blender-MCP: exec(open(<datei>).read()); dann build() / export()), baut eine eigene Szene
# "Gottfried_Holzmarkt6" und exportiert data/landmarks_raw/gottfried_holzmarkt6.glb (Spiel-Frame: Meter, Y oben, +X Ost, -Z Nord).
# Quellen: LoD2-Grundriss (20,7 m, 6 Geschosse), Google Street View / Maps-Fotos 2025 (nur angesehen; Maße aus Proportionen
# geschätzt). Offene Punkte, für die Fotos/Maße fehlen: pipeline/landmarks/gottfried/OFFENE_PUNKTE.md
# Aufbau: Baukörper (LoD2-Grundriss) liegt 0,5 m hinter den Fassadenebenen; davor die Fassadenschicht als Wandscheiben mit
# echten Öffnungen (Laibungen), Bronze-Stehfalz-Erker, Pilaster, Gesimsbänder, Kiosk-Nische, Lichterkette, Möbel.
import bpy, bmesh, numpy as np, json, os, math
from mathutils import Matrix, Vector

ROOT = r"C:\Users\luca\Documents\Sensing Space\Claude Code\output\hannover"
OUT = os.path.join(ROOT, "data", "landmarks_raw", "gottfried_holzmarkt6.glb")
SCENE = "Gottfried_Holzmarkt6"

# ------------------------------------------------------------------ Gelände
_hm = json.load(open(os.path.join(ROOT, "web", "assets", "city", "ground", "heightfield.json")))
_hf = np.fromfile(os.path.join(ROOT, "web", "assets", "city", "ground", "heightfield.bin"), np.float32).reshape(_hm["h"], _hm["w"])
def ground_y(x, z):
    fx = (x - _hm["x0"]) / _hm["dx"]; fz = (z - _hm["z0"]) / _hm["dz"]
    i = int(np.clip(math.floor(fx), 0, _hm["w"] - 2)); j = int(np.clip(math.floor(fz), 0, _hm["h"] - 2))
    tx = min(max(fx - i, 0), 1); tz = min(max(fz - j, 0), 1)
    a, b, c, d = _hf[j, i], _hf[j, i + 1], _hf[j + 1, i], _hf[j + 1, i + 1]
    return float((a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz)

# ------------------------------------------------------------------ Rahmen
P0 = np.array([-432.39, 353.51])                       # Südostecke (Holzmarkt / Schloßstraße)
PN = np.array([-439.67, 340.99])                       # Nordende Ostfassade (Anschluss Leibnizhaus)
PS = np.array([-445.53, 360.57])                       # Westende Südfassade
uE = (PN - P0) / np.linalg.norm(PN - P0); nE = np.array([uE[1], -uE[0]])
if nE[0] < 0: nE = -nE                                  # nach außen (Holzmarkt, Osten)
uS = (PS - P0) / np.linalg.norm(PS - P0); nS = np.array([-uS[1], uS[0]])
if nS[1] < 0: nS = -nS                                  # nach außen (Schloßstraße, Süden = +z)
LE = float(np.linalg.norm(PN - P0)); LS = float(np.linalg.norm(PS - P0))
YF = round(ground_y(*(P0 + 3.0 * uE + 0.8 * nE)), 3)  # Erdgeschoss-/Gehwegniveau an der Kiosk-Nische
GF_H, FL_H, PARA = 3.45, 3.25, 1.0
YB = YF + GF_H                                         # Gesimsband über dem EG
FLOORS = [YB + k * FL_H for k in range(5)]             # Fußböden 1.–5. OG
TOP = YB + 5 * FL_H + PARA                             # Attika-Oberkante (≈ 20,7 m über Gelände)
BASE = YF - 0.6
B = -0.5                                               # Baukörper hinter der Fassadenebene

def E(u, n): p = P0 + u * uE + n * nE; return float(p[0]), float(p[1])
def S(u, n): p = P0 + u * uS + n * nS; return float(p[0]), float(p[1])

# ------------------------------------------------------------------ Geometrie-Sammler (ein Objekt je Material)
ACC = {}
def add_poly_faces(mtl, verts, faces):
    a = ACC.setdefault(mtl, {"v": [], "f": []}); o = len(a["v"])
    a["v"].extend(verts); a["f"].extend([[o + i for i in f] for f in faces])

def _ccw(pts2):
    x = [p[0] for p in pts2]; z = [p[1] for p in pts2]; n = len(pts2)
    return sum(x[i] * z[(i + 1) % n] - x[(i + 1) % n] * z[i] for i in range(n)) > 0

def prism(mtl, poly2, y0, y1):
    if _ccw(poly2): poly2 = poly2[::-1]
    n = len(poly2)
    v = [(p[0], y0, p[1]) for p in poly2] + [(p[0], y1, p[1]) for p in poly2]
    f = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))] + [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    add_poly_faces(mtl, v, f)

def fbox(mtl, F, u0, u1, n0, n1, y0, y1):
    """Quader im Fassadenrahmen F (E oder S): u entlang, n nach außen, y Höhe"""
    if u1 - u0 < 1e-5 or n1 - n0 < 1e-5 or y1 - y0 < 1e-5: return
    prism(mtl, [F(u0, n0), F(u1, n0), F(u1, n1), F(u0, n1)], y0, y1)

def fcyl(mtl, F, u, n, r, y0, y1, seg=16):
    prism(mtl, [F(u + r * math.cos(2 * math.pi * k / seg), n + r * math.sin(2 * math.pi * k / seg)) for k in range(seg)], y0, y1)

def ffrustum(mtl, F, u, n, r0, r1, y0, y1, seg=20):
    cx, cz = F(u, n); v = []
    for r, y in ((r0, y0), (r1, y1)):
        v += [(cx + r * math.cos(2 * math.pi * k / seg), y, cz + r * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
    f = [list(range(seg - 1, -1, -1)), list(range(seg, 2 * seg))] + [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)]
    add_poly_faces(mtl, v, f)

def fsphere(mtl, F, u, n, y, r, seg=8, ring=5):
    cx, cz = F(u, n)
    v = [(cx, y + r, cz)]
    for i in range(1, ring):
        th = math.pi * i / ring
        v += [(cx + r * math.sin(th) * math.cos(2 * math.pi * k / seg), y + r * math.cos(th), cz + r * math.sin(th) * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
    v.append((cx, y - r, cz)); bot = len(v) - 1
    f = [[0, 1 + (k + 1) % seg, 1 + k] for k in range(seg)]
    for i in range(ring - 2):
        a0 = 1 + i * seg; a1 = a0 + seg
        f += [[a0 + k, a0 + (k + 1) % seg, a1 + (k + 1) % seg, a1 + k] for k in range(seg)]
    last = 1 + (ring - 2) * seg
    f += [[last + k, last + (k + 1) % seg, bot] for k in range(seg)]
    add_poly_faces(mtl, v, f)

def wall_holes(mtl, F, u0, u1, n0, n1, y0, y1, holes=()):
    """Wandscheibe mit rechteckigen Öffnungen holes=[(hu0,hu1,hy0,hy1)] (Laibungen entstehen als Seitenflächen)"""
    hs = [h for h in holes if h[1] > u0 and h[0] < u1]
    us = sorted(set([u0, u1] + [min(max(h[0], u0), u1) for h in hs] + [min(max(h[1], u0), u1) for h in hs]))
    for a, b in zip(us[:-1], us[1:]):
        if b - a < 1e-4: continue
        cut = sorted([h for h in hs if h[0] <= a + 1e-6 and h[1] >= b - 1e-6], key=lambda h: h[2])
        yc = y0
        for h in cut:
            if h[2] > yc + 1e-4: fbox(mtl, F, a, b, n0, n1, yc, h[2])
            yc = max(yc, h[3])
        if y1 > yc + 1e-4: fbox(mtl, F, a, b, n0, n1, yc, y1)

# ------------------------------------------------------------------ Materialien
def srgb2lin(c): return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
def hx(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

def brick_image(name, base, dark=1.0, seed=7):
    """Klinker-Textur (Läuferverband NF 240x71 + 10 mm Fuge), 1 m x 0,972 m, 512 x 498 px, sRGB"""
    if name in bpy.data.images: bpy.data.images.remove(bpy.data.images[name])
    W, H = 512, 498; rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    b = np.array(base) * dark; mort = np.array([0.80, 0.77, 0.71]) * dark
    ch = H / 12.0; bw = W / 4.0
    for r in range(12):
        y0 = int(round(r * ch)); y1 = int(round((r + 1) * ch))
        off = (bw / 2) if r % 2 else 0
        for k in range(-1, 5):
            x0 = int(round(k * bw + off)); x1 = int(round((k + 1) * bw + off))
            xa, xb = max(x0, 0), min(x1, W)
            if xb <= xa: continue
            c = b * (1 + rng.normal(0, 0.07)) * np.array([1, 1 + rng.normal(0, 0.02), 1 + rng.normal(0, 0.04)])
            if rng.random() < 0.08: c = c * 0.85                       # vereinzelt dunklere Brände
            img[y0:y1, xa:xb] = c * (1 + rng.normal(0, 0.03, (y1 - y0, xb - xa, 1)))
            if x0 >= 0: img[y0:y1, x0:min(x0 + 5, W)] = mort         # Stoßfuge
        img[max(y1 - 5, 0):y1, :] = mort                               # Lagerfuge
    img = np.clip(img, 0, 1)
    im = bpy.data.images.new(name, W, H, alpha=False)
    im.pixels.foreach_set(np.concatenate([img[::-1], np.ones((H, W, 1), np.float32)], 2).ravel())
    im.pack(); im.colorspace_settings.name = "sRGB"
    return im

def mat(name, hexcol, rough=0.8, metal=0.0, emis=None, estr=0.0, img=None, alpha=1.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    for n in list(nt.nodes):
        if n.type == "TEX_IMAGE": nt.nodes.remove(n)
    if img is not None:
        tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = img
        nt.links.new(tx.outputs["Color"], bs.inputs["Base Color"])
    else:
        bs.inputs["Base Color"].default_value = srgb2lin(hx(hexcol)) + [1]
    bs.inputs["Roughness"].default_value = rough; bs.inputs["Metallic"].default_value = metal
    if emis:
        bs.inputs["Emission Color"].default_value = srgb2lin(hx(emis)) + [1]; bs.inputs["Emission Strength"].default_value = estr
    bs.inputs["Alpha"].default_value = alpha
    if alpha < 1:
        try: m.surface_render_method = "BLENDED"
        except Exception: pass
    m.diffuse_color = srgb2lin(hx(hexcol)) + [alpha]
    return m

M = dict(
    brick="GF_Klinker", band="GF_Klinker_Band", bronze="GF_Bronzeblech", rib="GF_Bronze_Falz", frame="GF_Fensterrahmen_Bronze",
    glass="GF_Glas", dark="GF_Raumtiefe_Dunkel", canopy="GF_Vordach_Blech", door="GF_Tuerrahmen", board="GF_Kreidetafel",
    chalk="GF_Kreide", logo="GF_Logo_Schild", logotxt="GF_Logo_Schrift", soffit="GF_Nischendecke", light="GF_Downlight",
    fairy="GF_Lichterkette", wire="GF_Kabel", steel="GF_Gelaender_Stahl", statue="GF_Figur_Bronze", plinth="GF_Sockel_Stein",
    red="GF_Stuhl_Rot", blue="GF_Stuhl_Blau", black="GF_Moebel_Schwarz", wood="GF_Holz", shade="GF_Lampenschirm",
    crate="GF_Kiste_Schwarz", fridge="GF_Kuehlschrank_Licht", roof="GF_Dach_Kies", coping="GF_Abdeckung", slab="GF_Deckenrand",
    paving="GF_Pflaster_Nische",
)
def make_materials():
    ib = brick_image("GF_Klinker_sand", [0.80, 0.665, 0.47])
    ibd = brick_image("GF_Klinker_Band", [0.80, 0.665, 0.47], dark=0.86, seed=11)
    mat(M["brick"], "#ccaa78", img=ib, rough=0.85)
    mat(M["band"], "#b0915f", img=ibd, rough=0.85)
    mat(M["bronze"], "#64503b", rough=0.55, metal=0.6)
    mat(M["rib"], "#7a634a", rough=0.5, metal=0.6)
    mat(M["frame"], "#4a3a2b", rough=0.5, metal=0.4)
    mat(M["glass"], "#2a343b", rough=0.05, alpha=0.35)
    mat(M["dark"], "#1b1d1f", rough=0.9)
    mat(M["canopy"], "#3c3f42", rough=0.5, metal=0.5)
    mat(M["door"], "#2c2e30", rough=0.5, metal=0.4)
    mat(M["board"], "#202422", rough=0.9)
    mat(M["chalk"], "#e8e6df", rough=0.9)
    mat(M["logo"], "#1d2441", rough=0.6)
    mat(M["logotxt"], "#f2f0ea", rough=0.5)
    mat(M["soffit"], "#cfc9bd", rough=0.9)
    mat(M["light"], "#fff2d6", emis="#fff0d0", estr=6.0)
    mat(M["fairy"], "#ffcf86", emis="#ffc56e", estr=8.0)
    mat(M["wire"], "#1e1e1e", rough=0.6)
    mat(M["steel"], "#3d3e40", rough=0.5, metal=0.7)
    mat(M["statue"], "#5d5444", rough=0.5, metal=0.7)
    mat(M["plinth"], "#b9b2a4", rough=0.8)
    mat(M["red"], "#c8302c", rough=0.4)
    mat(M["blue"], "#4b62c8", rough=0.4)
    mat(M["black"], "#1c1c1c", rough=0.5, metal=0.3)
    mat(M["wood"], "#a87b4c", rough=0.7)
    mat(M["shade"], "#efe3c4", emis="#ffe2a8", estr=3.0)
    mat(M["crate"], "#151515", rough=0.6)
    mat(M["fridge"], "#e9f3ff", emis="#e4efff", estr=2.5)
    mat(M["roof"], "#7b7871", rough=0.95)
    mat(M["coping"], "#5b4c3c", rough=0.5, metal=0.5)
    mat(M["slab"], "#5f4d3a", rough=0.6, metal=0.4)
    mat(M["paving"], "#8e8a83", rough=0.9)

# ------------------------------------------------------------------ Bauteile
NICHE = (1.25, 4.35, -2.2)     # Kiosk-Nische: u0, u1, Rückwand n
BAL = (11.75, None, -1.1)      # Balkonachse OG: u0, (bis LE-0.05), Vorderkante Rückwand n

def build_mass():
    west = [(-446.38, 359.33), (-446.76, 359.57), (-449.61, 361.34), (-451.63, 358.17), (-454.87, 353.09),
            (-454.94, 352.99), (-451.98, 351.11), (-451.82, 351.00), (-452.75, 349.30), (-442.04, 342.49)]
    C0 = tuple(P0 + B * nE + B * nS)                  # zurückgesetzte Ecke
    u0b, nbb = BAL[0], BAL[2] - 0.5
    east_gf = [E(LE, B), E(NICHE[1], B), E(NICHE[1], NICHE[2]), E(NICHE[0], NICHE[2]), E(NICHE[0], B)]
    east_og = [E(LE, B), E(LE - 0.05, B), E(LE - 0.05, nbb), E(u0b, nbb), E(u0b, B)]
    def ring(east):          # Ecke -> Südkante -> West/Nord (LoD2) -> Ostkante (Nord -> Süd)
        return [C0, S(LS, B), tuple(PS)] + west + [tuple(PN)] + east
    prism(M["brick"], ring(east_gf), BASE, YF + 2.95)
    prism(M["brick"], ring([E(LE, B)]), YF + 2.95, YB)            # Sturzzone: Ostkante gerade
    prism(M["brick"], ring(east_og), YB, TOP - PARA)
    prism(M["roof"], ring(east_og), TOP - PARA, TOP - PARA + 0.04)

def bands(F, u0, u1, n_face, ys, depth=0.04):
    for y in ys:
        fbox(M["band"], F, u0, u1, n_face - 0.01, n_face + depth, y - 0.08, y + 0.08)

def window(F, u0, u1, y0, y1, n_face, lights=3, recess=0.12, transom=None, depth=0.32):
    """Bronze-Fensterrahmen mit Pfosten, Glas, Fensterbank; Rahmen sitzt recess hinter n_face in der Öffnung"""
    nf = n_face - recess; t = 0.06
    fbox(M["frame"], F, u0, u1, nf - 0.06, nf, y0, y0 + t)
    fbox(M["frame"], F, u0, u1, nf - 0.06, nf, y1 - t, y1)
    fbox(M["frame"], F, u0, u0 + t, nf - 0.06, nf, y0 + t, y1 - t)
    fbox(M["frame"], F, u1 - t, u1, nf - 0.06, nf, y0 + t, y1 - t)
    w = (u1 - u0) / lights
    for k in range(1, lights):
        uc = u0 + k * w; fbox(M["frame"], F, uc - 0.035, uc + 0.035, nf - 0.06, nf, y0 + t, y1 - t)
    if transom: fbox(M["frame"], F, u0 + t, u1 - t, nf - 0.06, nf, transom - 0.035, transom + 0.035)
    fbox(M["glass"], F, u0 + t, u1 - t, nf - 0.035, nf - 0.025, y0 + t, y1 - t)
    fbox(M["dark"], F, u0, u1, nf - depth, nf - depth + 0.01, y0, y1)          # Raumtiefe (Innenraum folgt als Scan)
    if recess > 0.05:
        fbox(M["coping"], F, u0 - 0.03, u1 + 0.03, nf, n_face + 0.05, y0 - 0.04, y0)   # Fensterbank mit Überstand

def bronze_bay(F, u0, u1, n_face, win_u, lights):
    """Bronze-Stehfalzblech über die OG mit Fenster je Geschoss; Blechscheibe bis zum Baukörper (Laibungen)"""
    holes = []
    for fy in FLOORS:
        sill, head = fy + 0.85, fy + 2.35
        holes.append((win_u[0], win_u[1], sill, head))
        window(F, win_u[0], win_u[1], sill, head, n_face, lights=lights, recess=0.12)
    wall_holes(M["bronze"], F, u0, u1, B, n_face, YB, TOP - PARA, holes)
    nr = int((u1 - u0) / 0.42)
    for i in range(1, nr + 1):                                   # Stehfalze, an den Fenstern unterbrochen
        uu = u0 + i * (u1 - u0) / (nr + 1)
        inside = win_u[0] - 0.03 < uu < win_u[1] + 0.03
        segs = [(YB, TOP - PARA)] if not inside else [(YB, holes[0][2])] + [(holes[k][3], holes[k + 1][2]) for k in range(len(holes) - 1)] + [(holes[-1][3], TOP - PARA)]
        for a, b in segs:
            fbox(M["rib"], F, uu - 0.012, uu + 0.012, n_face, n_face + 0.035, a, b)
    fbox(M["brick"], F, u0, u1, B, n_face, TOP - PARA, TOP)                       # Attika

def east_facade():
    u_n0, u_n1, nb = NICHE
    # --- Eckblock (u 0..5,4): 0,35 vor der Fassadenebene, geschlossen, EG mit Kiosk-Nische
    fbox(M["brick"], E, 0.0, u_n0, B, 0.35, BASE, YB)
    fbox(M["brick"], E, u_n1, 5.4, B, 0.35, BASE, YB)
    fbox(M["brick"], E, u_n0, u_n1, B, 0.35, YF + 2.95, YB)
    fbox(M["brick"], E, 0.0, 5.4, B, 0.35, YB, TOP)
    bands(E, 0.0, 5.4, 0.35, [YB] + FLOORS[1:] + [TOP - PARA])
    fbox(M["band"], E, 0.0, 11.75, 0.30, 0.44, YB - 0.10, YB + 0.06)          # EG-Gesims (Haken der Lichterkette)
    # --- Nische
    fbox(M["soffit"], E, u_n0, u_n1, nb, 0.35, YF + 2.93, YF + 2.95)
    fbox(M["paving"], E, u_n0, u_n1, nb, 0.35, YF - 0.25, YF + 0.005)                    # Nischenboden (Gelände hat hier ein Loch)
    for uu in (u_n0 + 0.55, (u_n0 + u_n1) / 2, u_n1 - 0.55):
        fcyl(M["light"], E, uu, -0.6, 0.07, YF + 2.915, YF + 2.93, seg=12)
    du0, du1 = u_n0 + 1.35, u_n1 - 0.12                                     # Glastür + Seitenteil
    for a, b, y0, y1 in ((du0, du1, YF, YF + 0.06), (du0, du1, YF + 2.38, YF + 2.45)):
        fbox(M["door"], E, a, b, nb, nb + 0.07, y0, y1)
    for uu in (du0, du0 + 0.55, du1 - 0.06):
        fbox(M["door"], E, uu, uu + 0.06, nb, nb + 0.07, YF + 0.06, YF + 2.38)
    fbox(M["glass"], E, du0 + 0.06, du1 - 0.06, nb + 0.035, nb + 0.045, YF + 0.06, YF + 2.38)
    fbox(M["dark"], E, du0 + 0.06, du1 - 0.06, nb + 0.005, nb + 0.012, YF + 0.06, YF + 2.38)
    fbox(M["fridge"], E, du1 - 0.7, du1 - 0.12, nb + 0.012, nb + 0.018, YF + 0.1, YF + 1.9)    # Lichtschein Kühlschrank
    fbox(M["door"], E, du0 + 0.62, du0 + 0.66, nb + 0.07, nb + 0.12, YF + 0.9, YF + 1.3)        # Griffstange
    wu0, wu1 = u_n0 + 0.2, u_n0 + 1.2                                         # Fenster mit Kreidetafel
    for a, b, y0, y1 in ((wu0, wu0 + 0.06, YF + 0.35, YF + 2.45), (wu1 - 0.06, wu1, YF + 0.35, YF + 2.45),
                         (wu0, wu1, YF + 0.35, YF + 0.41), (wu0, wu1, YF + 2.39, YF + 2.45)):
        fbox(M["door"], E, a, b, nb, nb + 0.07, y0, y1)
    fbox(M["board"], E, wu0 + 0.06, wu1 - 0.06, nb + 0.005, nb + 0.012, YF + 0.41, YF + 2.39)
    fbox(M["glass"], E, wu0 + 0.06, wu1 - 0.06, nb + 0.035, nb + 0.045, YF + 0.41, YF + 2.39)
    for k in range(7):
        y = YF + 0.75 + k * 0.2; w = 0.2 + 0.3 * ((k * 37) % 5) / 5
        fbox(M["chalk"], E, wu0 + 0.15, wu0 + 0.15 + w, nb + 0.012, nb + 0.016, y, y + 0.035)
    fbox(M["canopy"], E, du0 - 0.2, u_n1, nb, nb + 0.38, YF + 2.45, YF + 2.82)              # Blechkasten über der Tür
    for uu in (du0 + 0.25, du1 - 0.3):
        fcyl(M["light"], E, uu, nb + 0.2, 0.05, YF + 2.44, YF + 2.45, seg=10)
    # --- Logo-Pfeiler unter Erker A
    fbox(M["brick"], E, 5.4, 7.6, B, 0.2, BASE, YB)
    fbox(M["logo"], E, 5.55, 7.4, 0.2, 0.24, YF + 2.2, YF + 2.95)
    logo_text(5.55, 7.4, 0.24, YF + 2.2)
    # --- Erker A (3-teilig), Pilaster, Erker B (4-teilig) mit Schaufenster im EG
    bronze_bay(E, 5.4, 7.6, 0.12, (5.78, 7.22), 3)
    for u0, u1 in ((7.6, 8.4), (11.05, 11.75)):
        fbox(M["brick"], E, u0, u1, B, 0.5, BASE, TOP)
        bands(E, u0, u1, 0.5, [YB] + FLOORS[1:] + [TOP - PARA])
    bronze_bay(E, 8.4, 11.05, 0.12, (8.72, 10.73), 4)
    shop = (8.6, 10.85, YF + 0.45, YF + 3.05)
    wall_holes(M["brick"], E, 8.4, 11.05, B, 0.15, BASE, YB, [shop])
    window(E, *shop[:2], shop[2], shop[3], 0.15, lights=4, recess=0.15, transom=YF + 2.45, depth=0.42)
    # --- Balkonachse (OG zurückgesetzt): Hauseingang im EG, französische Fenster, Stahlgitter, Deckenstirnen
    u0, u1, nf = BAL[0], LE - 0.05, BAL[2]
    door = (u0 + 0.35, u0 + 1.35, YF, YF + 2.3)
    wall_holes(M["brick"], E, u0, u1, B, 0.12, BASE, YB, [door])
    window(E, door[0], door[1], door[2], door[3], 0.12, lights=1, recess=0.2)
    holes = [(u0 + 0.35, u1 - 0.4, fy, fy + 2.3) for fy in FLOORS]
    wall_holes(M["brick"], E, u0, u1, nf - 0.5, nf, YB, TOP - PARA, holes)
    for h in holes:
        window(E, h[0], h[1], h[2], h[3], nf, lights=2, recess=0.12)
        fy = h[2]
        fbox(M["slab"], E, u0, u1, nf, 0.0, fy - 0.22, fy)
        fbox(M["steel"], E, u0 + 0.05, u1 - 0.05, -0.08, -0.02, fy + 1.0, fy + 1.05)
        fbox(M["steel"], E, u0 + 0.05, u1 - 0.05, -0.08, -0.02, fy + 0.08, fy + 0.12)
        nbar = int((u1 - u0 - 0.1) / 0.12)
        for i in range(nbar + 1):
            uu = u0 + 0.05 + i * (u1 - u0 - 0.1) / nbar
            fbox(M["steel"], E, uu - 0.008, uu + 0.008, -0.06, -0.04, fy + 0.12, fy + 1.0)
    fbox(M["slab"], E, u0, u1, nf, 0.0, TOP - PARA - 0.22, TOP - PARA)
    fbox(M["brick"], E, u0, u1, nf - 0.5, 0.0, TOP - PARA, TOP)
    # Figur auf Konsole am Anschluss Leibnizhaus
    fbox(M["plinth"], E, 13.75, 14.35, 0.12, 0.62, YF + 2.55, YF + 2.95)
    fbox(M["plinth"], E, 13.85, 14.25, 0.2, 0.55, YF + 2.95, YF + 3.15)
    statue(E, 14.05, 0.38, YF + 3.15)

def statue(F, u, n, y):
    """vereinfachte Bronzefigur (Gelehrter im Mantel, ca. 1,1 m) – Motiv/Maße offen"""
    ffrustum(M["statue"], F, u, n, 0.20, 0.14, y, y + 0.62, seg=14)
    ffrustum(M["statue"], F, u, n, 0.14, 0.11, y + 0.62, y + 0.86, seg=14)
    fsphere(M["statue"], F, u, n, y + 0.97, 0.085, seg=10, ring=6)
    fsphere(M["statue"], F, u, n - 0.02, y + 0.94, 0.095, seg=10, ring=6)
    fbox(M["statue"], F, u + 0.1, u + 0.2, n + 0.05, n + 0.13, y + 0.55, y + 0.8)

def logo_text(u0, u1, n, y0):
    """Schriftzug (extrudierter Text) auf dem Schild: Segoe Script / Arial Bold"""
    def text_mesh(body, font_path, size, uu, yy, align="CENTER"):
        cu = bpy.data.curves.new("GF_txt", "FONT"); cu.body = body
        if os.path.exists(font_path): cu.font = bpy.data.fonts.load(font_path, check_existing=True)
        cu.size = size; cu.extrude = 0.004; cu.align_x = align; cu.align_y = "CENTER"; cu.resolution_u = 3
        ob = bpy.data.objects.new("GF_txt", cu); bpy.context.scene.collection.objects.link(ob)
        px, pz = E(uu, n + 0.004)
        X = Vector((uE[0], -uE[1], 0)); Y = Vector((0, 0, 1)); Z = Vector((nE[0], -nE[1], 0))
        ob.matrix_world = Matrix(((X.x, Y.x, Z.x, px), (X.y, Y.y, Z.y, -pz), (X.z, Y.z, Z.z, yy), (0, 0, 0, 1)))
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        me.transform(ob.matrix_world)
        add_poly_faces(M["logotxt"], [(v.co.x, v.co.z, -v.co.y) for v in me.vertices], [list(p.vertices) for p in me.polygons])
        bpy.data.objects.remove(ob); bpy.data.curves.remove(cu); bpy.data.meshes.remove(me)
    um = (u0 + u1) / 2
    text_mesh("Gottfried's", r"C:\Windows\Fonts\segoescb.ttf", 0.30, um - 0.12, y0 + 0.47)
    text_mesh("FEIN", r"C:\Windows\Fonts\arialbd.ttf", 0.06, um + 0.42, y0 + 0.27, "LEFT")
    text_mesh("KIOSK", r"C:\Windows\Fonts\arialbd.ttf", 0.10, um + 0.3, y0 + 0.18)
    text_mesh("im Leibnizhaus", r"C:\Windows\Fonts\segoesc.ttf", 0.055, um + 0.3, y0 + 0.08)

def south_facade():
    # EG: geschlossene Ecke, Hauseingang (Rücksprung 0,4), Pfeiler mit Satteldach-Abdeckung + Bank, Wand mit Fenstern
    fbox(M["brick"], S, -0.35, 2.4, B, 0.2, BASE, YB)
    door = (2.55, 3.35, YF, YF + 2.35)
    wall_holes(M["brick"], S, 2.4, 3.5, B, 0.2, BASE, YB, [door])
    window(S, door[0], door[1], door[2], door[3], 0.2, lights=1, recess=0.4, depth=0.26)
    fbox(M["brick"], S, 3.5, 4.5, B, 0.42, BASE, YF + 3.05)
    fbox(M["brick"], S, 3.5, 4.5, B, 0.2, YF + 3.05, YB)
    a0, a1, a2, a3 = S(3.45, -0.02), S(4.55, -0.02), S(4.55, 0.47), S(3.45, 0.47)
    m0, m1 = S(4.0, -0.02), S(4.0, 0.47)
    yb, yt = YF + 3.05, YF + 3.38
    v = [(a0[0], yb, a0[1]), (a1[0], yb, a1[1]), (a2[0], yb, a2[1]), (a3[0], yb, a3[1]), (m0[0], yt, m0[1]), (m1[0], yt, m1[1])]
    add_poly_faces(M["band"], v, [[3, 2, 1, 0], [0, 4, 5, 3], [1, 2, 5, 4], [0, 1, 4], [3, 5, 2]])
    fbox(M["wood"], S, 2.55, 3.4, 0.28, 0.68, YF + 0.42, YF + 0.47)
    for uu in (2.62, 3.33):
        fbox(M["steel"], S, uu - 0.03, uu + 0.03, 0.32, 0.64, YF, YF + 0.42)
    gfw = [(uu, uu + 1.4, YF + 0.9, YF + 2.6) for uu in (5.6, 8.1, 10.6, 13.1)]
    wall_holes(M["brick"], S, 4.5, LS, B, 0.1, BASE, YB, gfw)
    for h in gfw: window(S, h[0], h[1], h[2], h[3], 0.1, lights=2, recess=0.14)
    fbox(M["band"], S, -0.35, LS, 0.08, 0.24, YB - 0.10, YB + 0.06)
    # OG: Bronze-Erker (vorspringend) über dem Pfeiler, Ecke geschlossen, sonst Einzelfenster
    bronze_bay(S, 0.9, 4.5, 0.45, (1.2, 4.2), 4)
    fbox(M["brick"], S, 0.9, 4.5, 0.2, 0.45, YB - 0.12, YB)
    fbox(M["brick"], S, -0.35, 0.9, B, 0.2, YB, TOP)
    bands(S, -0.35, 0.9, 0.2, FLOORS[1:] + [TOP - PARA])
    ogw = [(uu, uu + 1.25, fy + 0.85, fy + 2.35) for fy in FLOORS for uu in (5.9, 8.3, 10.7, 13.1)]
    wall_holes(M["brick"], S, 4.5, LS, B, 0.1, YB, TOP, ogw)
    for h in ogw: window(S, h[0], h[1], h[2], h[3], 0.1, lights=2, recess=0.14)
    bands(S, 4.5, LS, 0.1, FLOORS[1:] + [TOP - PARA])
    fbox(M["coping"], S, -0.35, LS, -0.3, 0.25, TOP, TOP + 0.05)
    fbox(M["coping"], E, 0.0, LE, -0.3, 0.38, TOP, TOP + 0.05)

def fairy_lights():
    """Lichterkette unter dem EG-Gesims an Haken (alle 1,15 m), Birnen alle ~0,22 m, Durchhang 16 cm"""
    hooks = np.arange(0.3, 11.3, 1.15)
    for a, b in zip(hooks[:-1], hooks[1:]):
        fbox(M["wire"], E, a - 0.01, a + 0.01, 0.40, 0.48, YB - 0.22, YB - 0.10)
        m = int((b - a) / 0.22); prev = None
        for i in range(m + 1):
            t = i / m; uu = a + (b - a) * t; yy = YB - 0.22 - 0.16 * 4 * t * (1 - t)
            if 0 < i < m: fsphere(M["fairy"], E, uu, 0.475, yy - 0.04, 0.028, seg=6, ring=4)
            if prev is not None:
                fbox(M["wire"], E, prev[0], uu, 0.47, 0.48, min(prev[1], yy) - 0.004, max(prev[1], yy) + 0.004)
            prev = (uu, yy)

def chair(u, n, y, colm, rot=0.0):
    c, s = math.cos(rot), math.sin(rot)
    R = lambda du, dn: (u + c * du - s * dn, n + s * du + c * dn)
    def qbox(mtl, pts, y0, y1): prism(mtl, [E(*R(*p)) for p in pts], y0, y1)
    for du, dn in ((-0.18, -0.18), (0.18, -0.18), (0.18, 0.18), (-0.18, 0.18)):
        fcyl(M["black"], E, *R(du, dn), 0.012, y, y + 0.45, seg=6)
    qbox(colm, [(-0.21, -0.21), (0.21, -0.21), (0.21, 0.21), (-0.21, 0.21)], y + 0.45, y + 0.48)
    qbox(colm, [(-0.21, 0.18), (0.21, 0.18), (0.21, 0.22), (-0.21, 0.22)], y + 0.55, y + 0.88)
    for du in (-0.18, 0.18):
        fcyl(M["black"], E, *R(du, 0.2), 0.012, y + 0.48, y + 0.55, seg=6)

def furniture():
    y = YF
    fcyl(M["black"], E, 2.3, 1.35, 0.3, y + 0.72, y + 0.75, seg=24); fcyl(M["black"], E, 2.3, 1.35, 0.03, y + 0.02, y + 0.72, seg=8)
    fcyl(M["black"], E, 2.3, 1.35, 0.22, y, y + 0.02, seg=16)
    chair(1.75, 1.75, y, M["red"], rot=0.6)
    chair(3.05, 1.7, y, M["blue"], rot=-0.5)
    fbox(M["wood"], E, 3.55, 4.05, 0.95, 1.45, y + 0.56, y + 0.6)
    for du, dn in ((0.03, 0.03), (0.45, 0.03), (0.45, 0.45), (0.03, 0.45)):
        fbox(M["wood"], E, 3.55 + du - 0.02, 3.55 + du + 0.02, 0.95 + dn - 0.02, 0.95 + dn + 0.02, y, y + 0.56)
    for sgn in (-1, 1):                              # Kundenstopper: zwei geneigte Tafeln (je 2 cm dick)
        for d, mtl in ((0.0, M["board"]),):
            u0, u1 = 4.7, 5.3; nb, nt = 0.95 + sgn * 0.22, 0.95 + sgn * 0.015
            c = [E(u0, nb), E(u1, nb), E(u1, nt), E(u0, nt)]
            o = [E(u0, nb + sgn * 0.02), E(u1, nb + sgn * 0.02), E(u1, nt + sgn * 0.02), E(u0, nt + sgn * 0.02)]
            v = [(c[0][0], y, c[0][1]), (c[1][0], y, c[1][1]), (c[2][0], y + 1.0, c[2][1]), (c[3][0], y + 1.0, c[3][1]),
                 (o[0][0], y, o[0][1]), (o[1][0], y, o[1][1]), (o[2][0], y + 1.0, o[2][1]), (o[3][0], y + 1.0, o[3][1])]
            add_poly_faces(mtl, v, [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
    fcyl(M["black"], E, 4.45, 0.62, 0.14, y, y + 0.03, seg=14)            # Stehlampe mit Stoffschirm
    fcyl(M["wood"], E, 4.45, 0.62, 0.025, y + 0.03, y + 1.32, seg=8)
    ffrustum(M["shade"], E, 4.45, 0.62, 0.24, 0.16, y + 1.3, y + 1.62, seg=20)
    fbox(M["crate"], E, 5.6, 6.0, 0.25, 0.55, y, y + 0.3)                 # Getränkekiste

def build():
    sc = bpy.data.scenes.get(SCENE) or bpy.data.scenes.new(SCENE)
    bpy.context.window.scene = sc
    for o in list(sc.collection.all_objects): bpy.data.objects.remove(o, do_unlink=True)
    ACC.clear()
    make_materials()
    build_mass(); east_facade(); south_facade(); fairy_lights(); furniture()
    objs = []
    for mname, a in ACC.items():
        me = bpy.data.meshes.get("Gottfried_" + mname)
        if me: bpy.data.meshes.remove(me)
        me = bpy.data.meshes.new("Gottfried_" + mname)
        me.from_pydata([(x, -z, y) for (x, y, z) in a["v"]], [], a["f"]); me.validate(); me.update()
        bm = bmesh.new(); bm.from_mesh(me)
        if mname == M["logotxt"]: bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-5)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        uvl = bm.loops.layers.uv.new("UVMap")                     # UV in Metern (Klinker 1 m x 0,972 m)
        for f in bm.faces:
            nr = f.normal
            for l in f.loops:
                co = l.vert.co
                if abs(nr.z) > 0.7: l[uvl].uv = (co.x, co.y / 0.972)
                else:
                    t = Vector((-nr.y, nr.x, 0)); t = t.normalized() if t.length > 1e-6 else Vector((1, 0, 0))
                    l[uvl].uv = (co.x * t.x + co.y * t.y, co.z / 0.972)
        bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new("Gottfried_" + mname, me)
        me.materials.append(bpy.data.materials[mname])
        sc.collection.objects.link(ob); objs.append(ob)
    return objs

def check(objs):
    """offene Kanten je Objekt (Ziel: 0 bei allen geschlossenen Bauteilen)"""
    out = {}
    for o in objs:
        bm = bmesh.new(); bm.from_mesh(o.data)
        out[o.name] = dict(faces=len(o.data.polygons), boundary=sum(1 for e in bm.edges if e.is_boundary),
                           nonmanifold=sum(1 for e in bm.edges if not e.is_manifold)); bm.free()
    return out

def export(objs):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for o in bpy.context.scene.objects: o.select_set(o in objs)
    bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_apply=True, export_yup=True,
                              export_materials="EXPORT", export_image_format="AUTO")
    return OUT

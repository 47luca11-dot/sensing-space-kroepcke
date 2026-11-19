# Hauptbahnhof Hannover – Empfangsgebäude, Front zum Ernst-August-Platz (LoD2 ID_16759a91, "Ernst-August-Platz 1").
# Läuft IM Blender (Blender-MCP: exec(open(<datei>).read()); build(); export()). Eigene Szene "Hbf_Front".
# Neorenaissance-Fassade (Hubert Stier, 1879; nach dem Krieg vereinfacht wiederaufgebaut): Klinker mit Sandstein-Gliederung,
# vorspringender Mittelbau mit drei hohen Rundbogenfenstern über drei Portalen, Attika mit Schriftzug "HAUPTBAHNHOF" und Uhr,
# Flügel mit Rundbogenfenstern in zwei Geschossen, Eckpavillons. Maße aus LoD2 (Länge ~77 m, Traufe) und Proportionen
# geschätzt – Fotos/Maße vor Ort ergänzen. Die Fassade sitzt 0,05 m vor der LoD2-Wand (LoD2 bleibt als Baukörper).
import bpy, bmesh, numpy as np, json, os, math
from mathutils import Matrix, Vector

ROOT = r"C:\Users\luca\Documents\Sensing Space\Claude Code\output\hannover"
OUT = os.path.join(ROOT, "data", "landmarks_raw", "hbf_front.glb")
SCENE = "Hbf_Front"
_hm = json.load(open(os.path.join(ROOT, "web", "assets", "city", "ground", "heightfield.json")))
_hf = np.fromfile(os.path.join(ROOT, "web", "assets", "city", "ground", "heightfield.bin"), np.float32).reshape(_hm["h"], _hm["w"])
def ground_y(x, z):
    fx = (x - _hm["x0"]) / _hm["dx"]; fz = (z - _hm["z0"]) / _hm["dz"]
    i = int(np.clip(math.floor(fx), 0, _hm["w"] - 2)); j = int(np.clip(math.floor(fz), 0, _hm["h"] - 2))
    tx = min(max(fx - i, 0), 1); tz = min(max(fz - j, 0), 1)
    a, b, c, d = _hf[j, i], _hf[j, i + 1], _hf[j + 1, i], _hf[j + 1, i + 1]
    return float((a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz)

# Fassadenlinie aus dem LoD2-Grundriss (Spielkoordinaten x, z = -Nord): Südostende A -> Nordwestende B, außen = Platz
A = np.array([204.0, -172.6]); Bp = np.array([142.4, -218.6])
U = (Bp - A) / np.linalg.norm(Bp - A); N = np.array([U[1], -U[0]])
if (np.array([112.0, -190.0]) - (A + Bp) / 2) @ N < 0: N = -N
LEN = float(np.linalg.norm(Bp - A)); N0 = 0.05
YG = round(min(ground_y(*(A + u * U + 3 * N)) for u in np.linspace(0, LEN, 9)), 3)

ACC = {}
def add(mtl, verts, faces):
    a = ACC.setdefault(mtl, {"v": [], "f": []}); o = len(a["v"])
    a["v"].extend(verts); a["f"].extend([[o + i for i in f] for f in faces])
def P(u, n, y): p = A + u * U + (N0 + n) * N; return (float(p[0]), float(y), float(p[1]))
def box(mtl, u0, u1, n0, n1, y0, y1):
    v = [P(u0, n0, y0), P(u1, n0, y0), P(u1, n1, y0), P(u0, n1, y0), P(u0, n0, y1), P(u1, n0, y1), P(u1, n1, y1), P(u0, n1, y1)]
    add(mtl, v, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
def arch_outline(u0, u1, y0, y1, seg=14):
    """Rundbogen-Umriss (u, y): Rechteck bis Kämpfer, darüber Halbkreis"""
    r = (u1 - u0) / 2; yk = y1 - r; pts = [(u0, y0), (u1, y0), (u1, yk)]
    for k in range(1, seg): a = math.pi * k / seg; pts.append((u0 + r + r * math.cos(a), yk + r * math.sin(a)))
    pts.append((u0, yk)); return pts
def plate(mtl, pts, n0, n1):
    """Prisma aus Umriss (u, y) zwischen n0 und n1"""
    m = len(pts); v = [P(u, n0, y) for u, y in pts] + [P(u, n1, y) for u, y in pts]
    f = [list(range(m - 1, -1, -1)), list(range(m, 2 * m))] + [[i, (i + 1) % m, m + (i + 1) % m, m + i] for i in range(m)]
    add(mtl, v, f)
def arch_window(u0, u1, y0, y1, n_face, lights=2, frame=0.22, lit=False):
    """Sandstein-Gewände (Ring), Glas mit Sprossen; lit = warm beleuchtete Halle dahinter"""
    o = arch_outline(u0 - frame, u1 + frame, y0 - 0.12, y1 + frame)
    i = arch_outline(u0, u1, y0, y1)
    # Gewände als Rahmen: äußerer Umriss vor, innerer zurück -> zwei Platten mit Loch sind aufwendig, daher Rahmenstücke
    plate("sandstein", o, n_face, n_face + 0.12)
    plate("glas_hell" if lit else "glas", i, n_face + 0.121, n_face + 0.13)
    r = (u1 - u0) / 2; yk = y1 - r
    for k in range(1, lights):
        uu = u0 + k * (u1 - u0) / lights; top = yk + math.sqrt(max(r * r - (uu - u0 - r) ** 2, 0)) - 0.02
        box("rahmen", uu - 0.04, uu + 0.04, n_face + 0.13, n_face + 0.17, y0, top)
    for yy in np.arange(y0 + 1.4, yk, 1.4): box("rahmen", u0, u1, n_face + 0.13, n_face + 0.17, yy - 0.035, yy + 0.035)
    box("rahmen", u0, u1, n_face + 0.13, n_face + 0.17, yk - 0.04, yk + 0.04)
    box("sandstein", u0 - frame - 0.1, u1 + frame + 0.1, n_face, n_face + 0.3, y0 - 0.3, y0 - 0.12)   # Sohlbank

def text(body, size, u, y, n, mtl, font=r"C:\Windows\Fonts\timesbd.ttf", spacing=1.25):
    cu = bpy.data.curves.new("hbf_txt", "FONT"); cu.body = body
    if os.path.exists(font): cu.font = bpy.data.fonts.load(font, check_existing=True)
    cu.size = size; cu.extrude = 0.04; cu.align_x = "CENTER"; cu.align_y = "CENTER"; cu.space_character = spacing; cu.resolution_u = 3
    ob = bpy.data.objects.new("hbf_txt", cu); bpy.context.scene.collection.objects.link(ob)
    px, py, pz = P(u, n, y)
    Z = Vector((N[0], -N[1], 0)); Y = Vector((0, 0, 1)); X = Y.cross(Z)                # von außen lesbar (rechtshändig)
    ob.matrix_world = Matrix(((X.x, Y.x, Z.x, px), (X.y, Y.y, Z.y, -pz), (X.z, Y.z, Z.z, py), (0, 0, 0, 1)))
    bpy.context.view_layer.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get())); me.transform(ob.matrix_world)
    add(mtl, [(v.co.x, v.co.z, -v.co.y) for v in me.vertices], [list(p.vertices) for p in me.polygons])
    bpy.data.objects.remove(ob); bpy.data.curves.remove(cu); bpy.data.meshes.remove(me)

def disc(mtl, u, y, n, r, seg=40, d=0.08):
    pts = [(u + r * math.cos(2 * math.pi * k / seg), y + r * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
    plate(mtl, pts, n, n + d)

# ------------------------------------------------------------------ Fassade
def fassade():
    y0 = YG
    EAVE = 16.5; C0, C1 = 26.0, 51.0                 # Mittelbau
    PAV = 8.0                                          # Eckpavillons
    # Sockel über die ganze Länge
    box("sandstein_dunkel", -0.3, LEN + 0.3, 0.0, 0.45, y0 - 0.5, y0 + 1.1)
    # --- Flügel
    for ua, ub in ((PAV, C0), (C1, LEN - PAV)):
        box("klinker", ua, ub, 0.0, 0.3, y0 + 1.1, y0 + EAVE)
        box("sandstein", ua, ub, 0.0, 0.42, y0 + 6.9, y0 + 7.3)                       # Gurtgesims
        box("sandstein", ua, ub, 0.0, 0.85, y0 + EAVE, y0 + EAVE + 0.55)              # Kranzgesims
        box("sandstein", ua, ub, 0.0, 0.5, y0 + EAVE + 0.55, y0 + EAVE + 1.6)         # Attika
        nb = int((ub - ua) / 4.3); w = (ub - ua) / nb
        for k in range(nb):
            uc = ua + (k + 0.5) * w
            box("sandstein", uc - w / 2 + 0.05, uc - w / 2 + 0.55, 0.3, 0.5, y0 + 1.1, y0 + EAVE)      # Lisenen
            arch_window(uc - 1.15, uc + 1.15, y0 + 1.6, y0 + 6.3, 0.3, lights=2)                   # EG
            arch_window(uc - 0.95, uc + 0.95, y0 + 8.4, y0 + 13.2, 0.3, lights=2)                  # OG
            box("sandstein", uc - 0.4, uc + 0.4, 0.5, 0.6, y0 + EAVE + 0.75, y0 + EAVE + 1.45)       # Balusterfeld
    # --- Eckpavillons (vorspringend, höher)
    for ua, ub in ((0.0, PAV), (LEN - PAV, LEN)):
        box("klinker", ua, ub, 0.0, 1.2, y0 + 1.1, y0 + EAVE + 3.0)
        for uu in (ua, ub - 0.7):
            box("sandstein", uu, uu + 0.7, 1.2, 1.4, y0 + 1.1, y0 + EAVE + 3.0)                     # Eckquader
        box("sandstein", ua - 0.2, ub + 0.2, 0.0, 1.8, y0 + EAVE + 3.0, y0 + EAVE + 3.6)
        box("sandstein", ua, ub, 0.0, 1.4, y0 + EAVE + 3.6, y0 + EAVE + 5.0)
        uc = (ua + ub) / 2
        arch_window(uc - 1.6, uc + 1.6, y0 + 1.8, y0 + 7.0, 1.2, lights=3)
        arch_window(uc - 1.3, uc + 1.3, y0 + 9.0, y0 + 15.5, 1.2, lights=3)
        box("sandstein", ua, ub, 1.2, 1.55, y0 + 7.6, y0 + 8.0)
    # --- Mittelbau: drei Portale, drei hohe Rundbogenfenster, Attika mit Schriftzug, Uhr im Giebel
    H = 23.0
    box("klinker", C0, C1, 0.0, 2.2, y0 + 1.1, y0 + H)
    for uu in (C0, C0 + 7.7, C1 - 7.7 - 1.0, C1 - 1.0):                                        # Pilaster
        box("sandstein", uu, uu + 1.0, 2.2, 2.5, y0 + 1.1, y0 + H)
    box("sandstein", C0 - 0.3, C1 + 0.3, 0.0, 2.9, y0 + 5.6, y0 + 6.2)                           # Gesims über den Portalen
    for k, uc in enumerate((C0 + 4.3, (C0 + C1) / 2, C1 - 4.3)):
        # Portal: Sandsteinrahmung, Glastüren, Licht aus der Halle
        plate("sandstein", arch_outline(uc - 2.4, uc + 2.4, y0, y0 + 5.4), 2.2, 2.45)
        plate("glas_hell", arch_outline(uc - 1.9, uc + 1.9, y0, y0 + 4.9), 2.46, 2.47)
        for uu in (uc - 0.95, uc, uc + 0.95): box("rahmen", uu - 0.05, uu + 0.05, 2.47, 2.52, y0, y0 + 3.0)
        box("rahmen", uc - 1.9, uc + 1.9, 2.47, 2.52, y0 + 2.95, y0 + 3.05)
        arch_window(uc - 2.6, uc + 2.6, y0 + 7.0, y0 + 17.6, 2.2, lights=4, frame=0.35, lit=True)     # Hallenfenster
    box("sandstein", C0 - 0.3, C1 + 0.3, 0.0, 3.0, y0 + H - 4.4, y0 + H - 3.8)                     # Kranzgesims Mittelbau
    box("sandstein", C0, C1, 0.0, 2.4, y0 + H - 3.8, y0 + H)                                      # Attika mit Schriftzug
    text("HAUPTBAHNHOF", 1.45, (C0 + C1) / 2, y0 + H - 1.9, 2.47, "schrift", font="", spacing=1.3)                    # Blender-Standardschrift (Times hat Buchstaben mit Innenraum verloren)
    # Giebel mit Uhr
    g = [(C0 + 6.0, y0 + H), (C1 - 6.0, y0 + H), ((C0 + C1) / 2, y0 + H + 4.0)]
    plate("sandstein", [(g[0][0], g[0][1]), (g[1][0], g[1][1]), (g[2][0], g[2][1])], 1.6, 2.4)
    disc("sandstein_dunkel", (C0 + C1) / 2, y0 + H + 1.55, 2.4, 1.35)
    disc("uhr", (C0 + C1) / 2, y0 + H + 1.55, 2.48, 1.15)
    for h_ in range(12):                                                                          # Stundenmarken
        a = h_ / 12 * 2 * math.pi; uu = (C0 + C1) / 2 + math.sin(a) * 1.0; yy = y0 + H + 1.55 + math.cos(a) * 1.0
        box("zeiger", uu - 0.04, uu + 0.04, 2.56, 2.6, yy - 0.08, yy + 0.08)
    box("zeiger", (C0 + C1) / 2 - 0.04, (C0 + C1) / 2 + 0.04, 2.6, 2.64, y0 + H + 1.55, y0 + H + 2.35)   # 12:00 bzw. 9:xx
    box("zeiger", (C0 + C1) / 2 - 0.65, (C0 + C1) / 2, 2.6, 2.64, y0 + H + 1.51, y0 + H + 1.59)
    # Vordach-Leuchten über den Portalen
    for uc in (C0 + 4.3, (C0 + C1) / 2, C1 - 4.3):
        box("licht", uc - 0.15, uc + 0.15, 2.5, 2.9, y0 + 5.5, y0 + 5.62)

MATS = {
    "klinker": ("#8e5a40", 0.85, 0.0, None), "sandstein": ("#cdb48c", 0.8, 0.0, None), "sandstein_dunkel": ("#9c8a6c", 0.85, 0.0, None),
    "glas": ("#1d262c", 0.08, 0.0, None), "glas_hell": ("#3a3222", 0.1, 0.0, ("#ffd9a0", 2.2)), "rahmen": ("#2a2c2e", 0.5, 0.5, None),
    "schrift": ("#f4efe2", 0.4, 0.0, ("#fff4dc", 1.6)), "uhr": ("#f6f1e6", 0.5, 0.0, ("#fff6e4", 1.2)), "zeiger": ("#151515", 0.5, 0.4, None),
    "licht": ("#fff2d6", 0.4, 0.0, ("#fff0d0", 6.0)),
}
def make_materials():
    for k, (hx, rough, metal, em) in MATS.items():
        name = "Hbf_" + k; m = bpy.data.materials.get(name) or bpy.data.materials.new(name); m.use_nodes = True
        b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        c = [int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5)]; c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        b.inputs["Base Color"].default_value = c + [1]; b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
        if em:
            e = [int(em[0][i:i + 2], 16) / 255 for i in (1, 3, 5)]
            b.inputs["Emission Color"].default_value = e + [1]; b.inputs["Emission Strength"].default_value = em[1]
        m.diffuse_color = c + [1]

def build():
    sc = bpy.data.scenes.get(SCENE) or bpy.data.scenes.new(SCENE); bpy.context.window.scene = sc
    for o in list(sc.collection.all_objects): bpy.data.objects.remove(o, do_unlink=True)
    ACC.clear(); make_materials(); fassade()
    objs = []
    for k, a in ACC.items():
        me = bpy.data.meshes.new("Hbf_" + k); me.from_pydata([(x, -z, y) for (x, y, z) in a["v"]], [], a["f"])
        if k != "schrift": me.validate()
        me.update()
        bm = bmesh.new(); bm.from_mesh(me)
        if k == "schrift": bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-5)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        uvl = bm.loops.layers.uv.new("UVMap")
        for f in bm.faces:
            for l in f.loops: co = l.vert.co; l[uvl].uv = (co.x * 0.5 + co.y * 0.5, co.z)
        bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new("Hbf_" + k, me); me.materials.append(bpy.data.materials["Hbf_" + k]); sc.collection.objects.link(ob); objs.append(ob)
    return objs

def export(objs):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for s in bpy.data.scenes:
        for o in s.objects:
            try: o.select_set(False, view_layer=s.view_layers[0])
            except Exception: pass
    for o in objs: o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, use_active_scene=True, export_apply=True, export_yup=True)
    return OUT

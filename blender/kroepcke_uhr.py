"""Kroepcke-Uhr (Hannover) -- Replik 1977 nach der Wettersaeule von 1885.
Masse aus Foto (Commons 2023-07-06) + bDOM10 (Gesamthoehe ~6.3 m, Oberkoerper 1.2-2.0 m).
Blender-Frame: x Ost, y Nord, z oben; Ursprung = Uhrmitte auf Strassenniveau. Export: GLB (Y-up) fuer das Spiel.
Aufruf: blender -b --factory-startup -P kroepcke_uhr.py -- <out.glb> <out.blend>"""
import bpy, bmesh, math, sys, os, random
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT_GLB = argv[0] if argv else 'kroepcke_uhr.glb'
OUT_BLEND = argv[1] if len(argv) > 1 else None
TEX = os.path.join(os.path.dirname(OUT_GLB), 'tex'); os.makedirs(TEX, exist_ok=True)
BEARING = 33.0                     # Achse Bahnhofstrasse (Grad von Nord) -> Front-Zifferblatt zeigt dorthin
random.seed(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
ROOT = bpy.data.objects.new('Kroepcke_Uhr', None); SC.collection.objects.link(ROOT)


# ---------------------------------------------------------------- Materialien
def principled(name, rgb, metal=0.0, rough=0.5, alpha=1.0, emit=None, emit_s=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (*srgb(rgb), 1)
    b.inputs['Metallic'].default_value = metal; b.inputs['Roughness'].default_value = rough
    if alpha < 1:
        b.inputs['Alpha'].default_value = alpha
        try: m.surface_render_method = 'BLENDED'
        except Exception: pass
    if emit:
        b.inputs['Emission Color'].default_value = (*srgb(emit), 1); b.inputs['Emission Strength'].default_value = emit_s
    return m


def srgb(c):
    c = [x / 255.0 if max(c) > 1 else x for x in c]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def tex_material(name, img_path, rough=0.6, metal=0.0, alpha_clip=False, emit_s=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(img_path)
    nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = rough; b.inputs['Metallic'].default_value = metal
    if alpha_clip:
        g = nt.nodes.new('ShaderNodeMath'); g.operation = 'GREATER_THAN'; g.inputs[1].default_value = 0.5
        nt.links.new(t.outputs['Alpha'], g.inputs[0]); nt.links.new(g.outputs[0], b.inputs['Alpha'])
        try: m.surface_render_method = 'DITHERED'
        except Exception: pass
    if emit_s > 0:
        nt.links.new(t.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = emit_s
    return m


def save_png(name, arr):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    a = np.ones((h, w, 4), np.float32); a[..., :arr.shape[2]] = arr
    img.pixels.foreach_set(a[::-1].ravel())                      # Blender: Zeile 0 unten
    p = os.path.join(TEX, name + '.png'); img.filepath_raw = p; img.file_format = 'PNG'; img.save()
    return p


IRON = principled('Gusseisen_Gruen', (46, 62, 55), metal=0.35, rough=0.42)
IRON_D = principled('Gusseisen_Dunkel', (30, 40, 36), metal=0.3, rough=0.5)
GOLD = principled('Blattgold', (205, 165, 80), metal=1.0, rough=0.28)
DIAL = principled('Zifferblatt', (243, 241, 234), rough=0.25)
INK = principled('Zeiger_Ziffern', (18, 18, 20), metal=0.2, rough=0.35)
GLASS = principled('Vitrinenglas', (40, 52, 56), metal=0.1, rough=0.04, alpha=0.28)
STONE = principled('Granit_Stufe', (150, 148, 142), rough=0.75)
WOOD = principled('Holz_Pflanzkuebel', (150, 112, 70), rough=0.7)
STEEL = principled('Stahl_verzinkt', (120, 124, 126), metal=0.8, rough=0.4)
SOIL = principled('Erde_Rinde', (60, 46, 34), rough=0.95)
CULM = principled('Bambus_Halm', (120, 140, 60), rough=0.55)


# ---------------------------------------------------------------- Geometrie-Helfer
def new_obj(name, bm, mat, parent=ROOT, smooth=False):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    if smooth:
        for p in me.polygons: p.use_smooth = True
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me); SC.collection.objects.link(o); o.parent = parent
    return o


def square_lathe(name, prof, mat, cap_bottom=True, cap_top=True, parent=ROOT):
    """Profil [(halbe Breite, z), ...] von unten nach oben -> quadratischer Drehkoerper (Gesims, Sockel)"""
    bm = bmesh.new(); rings = []
    for h, z in prof:
        rings.append([bm.verts.new((sx * h, sy * h, z)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(4):
            j = (i + 1) % 4
            try: bm.faces.new((a[i], a[j], b[j], b[i]))
            except ValueError: pass
    if cap_bottom: bm.faces.new(rings[0][::-1])
    if cap_top: bm.faces.new(rings[-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    return new_obj(name, bm, mat, parent)


def box(name, c, s, mat, parent=ROOT, rot=0.0):
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=s, verts=bm.verts)
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(rot, 3, 'Z'), verts=bm.verts)
    bmesh.ops.translate(bm, vec=c, verts=bm.verts)
    return new_obj(name, bm, mat, parent)


def cyl(name, c, r, h, mat, seg=24, axis='Z', r2=None, parent=ROOT, smooth=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r if r2 is None else r2, depth=h)
    if axis == 'Y': bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, 'X'), verts=bm.verts)
    if axis == 'X': bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, 'Y'), verts=bm.verts)
    bmesh.ops.translate(bm, vec=c, verts=bm.verts)
    return new_obj(name, bm, mat, parent, smooth)


def sphere(name, c, r, mat, seg=24, ring=16, parent=ROOT, scale=(1, 1, 1)):
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=ring, radius=r)
    bmesh.ops.scale(bm, vec=scale, verts=bm.verts); bmesh.ops.translate(bm, vec=c, verts=bm.verts)
    return new_obj(name, bm, mat, parent, True)


def torus(name, c, R, r, mat, normal, seg=40, rseg=10, parent=ROOT):
    bm = bmesh.new(); vs = []
    for i in range(seg):
        a = 2 * math.pi * i / seg; row = []
        for j in range(rseg):
            b = 2 * math.pi * j / rseg
            row.append(bm.verts.new(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b))))
        vs.append(row)
    for i in range(seg):
        for j in range(rseg):
            bm.faces.new((vs[i][j], vs[(i + 1) % seg][j], vs[(i + 1) % seg][(j + 1) % rseg], vs[i][(j + 1) % rseg]))
    q = Vector((0, 0, 1)).rotation_difference(Vector(normal))
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=q.to_matrix(), verts=bm.verts); bmesh.ops.translate(bm, vec=c, verts=bm.verts)
    return new_obj(name, bm, mat, parent, True)


def extrude_profile(name, pts, thick, mat, parent=ROOT):
    """2D-Profil (u=nach aussen, z) in der u-z-Ebene, Dicke entlang v; lokales Objekt, danach platzieren"""
    bm = bmesh.new()
    f0 = [bm.verts.new((u, -thick / 2, z)) for u, z in pts]
    f1 = [bm.verts.new((u, thick / 2, z)) for u, z in pts]
    bm.faces.new(f0[::-1] if True else f0); bm.faces.new(f1)
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((f0[i], f0[j], f1[j], f1[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return new_obj(name, bm, mat, parent)


def face_frame(k):
    """Seite k (0..3): Normale n, Tangente t (rechts, von vorn gesehen)"""
    a = math.radians(90 * k)
    n = Vector((math.sin(a), math.cos(a), 0)); t = Vector((math.cos(a), -math.sin(a), 0))
    return n, t


def basis(X, Y, Z, p):
    X, Y, Z = Vector(X), Vector(Y), Vector(Z)
    return Matrix(((X.x, Y.x, Z.x, p[0]), (X.y, Y.y, Z.y, p[1]), (X.z, Y.z, Z.z, p[2]), (0, 0, 0, 1)))


UP = Vector((0, 0, 1))


def place(o, n, t, origin):
    """lokal: x=Tangente, y=Normale, z=oben"""
    M = Matrix(((t.x, n.x, 0, origin[0]), (t.y, n.y, 0, origin[1]), (0, 0, 1, origin[2]), (0, 0, 0, 1)))
    o.matrix_world = M


# ---------------------------------------------------------------- 1) Granitstufe
square_lathe('Stufe_Granit', [(1.50, -0.30), (1.50, 0.13), (1.485, 0.155), (1.47, 0.16)], STONE)
# Sockelplatte (Eisen) mit Fase
square_lathe('Sockel_Platte', [(0.98, 0.16), (0.98, 0.24), (0.95, 0.28), (0.90, 0.30)], IRON)
# zurueckgesetzter Kern der Konsolenzone
square_lathe('Sockel_Kern', [(0.64, 0.30), (0.64, 0.86)], IRON_D, cap_bottom=False, cap_top=False)
# Konsolen (S-Volute) -- 3 je Seite + Ecken
CON = [(0.00, 0.30), (0.17, 0.30), (0.20, 0.33), (0.15, 0.39), (0.12, 0.46), (0.13, 0.53), (0.17, 0.60), (0.24, 0.67),
       (0.31, 0.73), (0.35, 0.79), (0.36, 0.84), (0.36, 0.86), (0.00, 0.86)]
for k in range(4):
    n, t = face_frame(k)
    for s in (-0.34, 0.0, 0.34):
        o = extrude_profile(f'Konsole_{k}_{s:+.2f}', CON, 0.09, IRON)
        o.matrix_world = basis(n, UP.cross(n), UP, n * 0.60 + t * s)
        # Volutenauge (Schnecke) beidseitig
        for side in (-1, 1):
            e = cyl(f'Volute_{k}_{s:+.2f}_{side}', (0, 0, 0), 0.045, 0.012, IRON, seg=16, axis='X')
            place(e, n, t, (n * (0.60 + 0.14) + t * (s + side * 0.05) + Vector((0, 0, 0.42))))
            r = cyl(f'Volute2_{k}_{s:+.2f}_{side}', (0, 0, 0), 0.035, 0.012, IRON, seg=16, axis='X')
            place(r, n, t, (n * (0.60 + 0.30) + t * (s + side * 0.05) + Vector((0, 0, 0.76))))
    # Eckkonsole diagonal
    d = (n + t).normalized(); td = Vector((-d.y, d.x, 0))
    o = extrude_profile(f'Konsole_Ecke_{k}', [(u * 1.25, z) for u, z in CON], 0.11, IRON)
    o.matrix_world = basis(d, UP.cross(d), UP, d * 0.84)
# Hauptplatte mit profilierter Kante
square_lathe('Sockel_Gesims', [(0.92, 0.86), (1.02, 0.87), (1.04, 0.90), (1.04, 0.95), (1.01, 0.97), (0.99, 1.00)], IRON)
# Schraege (Karnies) zum Koerper
og = []
for i in range(9):
    s = i / 8; og.append((0.99 - 0.33 * (3 * s * s - 2 * s ** 3), 1.00 + 0.30 * s))
square_lathe('Sockel_Karnies', og + [(0.66, 1.30), (0.68, 1.33), (0.68, 1.38), (0.64, 1.42)], IRON)
# Goldsterne auf dem Gesims
for k in range(4):
    n, t = face_frame(k)
    for s in np.linspace(-0.8, 0.8, 7):
        sphere(f'Stern_Sockel_{k}_{s:.2f}', tuple(n * 1.045 + t * s + Vector((0, 0, 0.925))), 0.012, GOLD, seg=8, ring=6)

# ---------------------------------------------------------------- 2) Vitrinenkoerper z 1.42 .. 3.55
Z0, Z1 = 1.42, 3.55
HB = 0.585                                     # halbe Breite Koerper
OPW, OPZ0, OPZ1, RAD = 0.80, 1.56, 3.40, 0.13  # Vitrinenoeffnung


def framed_face(k):
    """Wandscheibe mit Oeffnung (abgerundete obere Ecken), Laibung, Glas, Kunstwerk dahinter"""
    n, t = face_frame(k)
    # Kontur der Oeffnung (lokal u,z)
    hole = [(-OPW / 2, OPZ0), (OPW / 2, OPZ0)]
    for i in range(7):
        a = math.radians(0 + 90 * i / 6); hole.append((OPW / 2 - RAD + RAD * math.cos(a), OPZ1 - RAD + RAD * math.sin(a)))
    for i in range(7):
        a = math.radians(90 + 90 * i / 6); hole.append((-OPW / 2 + RAD + RAD * math.cos(a), OPZ1 - RAD + RAD * math.sin(a)))
    W = HB - 0.13                              # Wandscheibe zwischen den Eckpilastern
    L_, R_ = -OPW / 2, OPW / 2
    bm = bmesh.new()
    for (u0, z0, u1, z1) in ((-W, Z0, W, OPZ0), (-W, OPZ0, L_, Z1), (R_, OPZ0, W, Z1), (L_, OPZ1, R_, Z1)):
        bm.faces.new([bm.verts.new(p) for p in ((u0, 0, z0), (u1, 0, z0), (u1, 0, z1), (u0, 0, z1))])
    for cu, a0 in ((R_ - RAD, 0), (L_ + RAD, 90)):           # Zwickel an den runden Ecken
        corner = (R_, OPZ1) if a0 == 0 else (L_, OPZ1)
        arc = [(cu + RAD * math.cos(math.radians(a0 + 90 * i / 6)), OPZ1 - RAD + RAD * math.sin(math.radians(a0 + 90 * i / 6))) for i in range(7)]
        vc = bm.verts.new((corner[0], 0, corner[1]))
        va = [bm.verts.new((u, 0, z)) for u, z in arc]
        for i in range(6):
            bm.faces.new((vc, va[i], va[i + 1]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    for f in bm.faces:
        if f.normal.y < 0: f.normal_flip()
    DEP = 0.17                                  # Nischentiefe bis zur Kunsttafel
    vh = [bm.verts.new((u, 0, z)) for u, z in hole]; vb = [bm.verts.new((u, -DEP, z)) for u, z in hole]
    hc = Vector((0, -DEP / 2, (OPZ0 + OPZ1) / 2))
    for i in range(len(hole)):
        j = (i + 1) % len(hole)
        f = bm.faces.new((vh[i], vh[j], vb[j], vb[i])); f.normal_update()
        if f.normal.dot(hc - f.calc_center_median()) < 0: f.normal_flip()
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    o = new_obj(f'Vitrine_Wand_{k}', bm, IRON); place(o, n, t, n * HB)
    # profilierter Rahmen um die Oeffnung (Rundstab)
    pts = hole + [hole[0]]
    for i in range(len(hole)):
        a, b = Vector((pts[i][0], 0, pts[i][1])), Vector((pts[i + 1][0], 0, pts[i + 1][1]))
        L = (b - a).length
        if L < 1e-4: continue
        c = cyl(f'Rahmenstab_{k}_{i}', (0, 0, 0), 0.016, L + 0.01, GOLD if i % 2 else IRON, seg=8, axis='Z')
        d = (b - a).normalized(); rot = Vector((0, 0, 1)).rotation_difference(d).to_matrix().to_4x4()
        loc = (a + b) / 2
        c.matrix_world = Matrix(((t.x, n.x, 0, 0), (t.y, n.y, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))) @ (Matrix.Translation(loc + Vector((0, 0.012, 0))) @ rot)
        c.location += n * HB
    # Glas
    bm = bmesh.new(); vg = [bm.verts.new((u, -0.04, z)) for u, z in hole]; bm.faces.new(vg[::-1])
    g = new_obj(f'Vitrine_Glas_{k}', bm, GLASS); place(g, n, t, n * HB)
    # Kunstwerk (Kulturraum-Vitrine) dahinter
    art = ART[k]
    bm = bmesh.new(); uv = bm.loops.layers.uv.new()
    va = [bm.verts.new((u, -0.16, z)) for u, z in ((-0.40, 1.56), (0.40, 1.56), (0.40, 3.40), (-0.40, 3.40))]
    f = bm.faces.new(va[::-1])
    for l in f.loops:
        l[uv].uv = ((0.40 - l.vert.co.x) / 0.80, (l.vert.co.z - 1.56) / 1.84)
    a = new_obj(f'Vitrine_Kunst_{k}', bm, art); place(a, n, t, n * HB)


def make_art(k):
    h, w = 512, 224
    rng = np.random.default_rng(10 + k)
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    base = np.array([[0.95, 0.85, 0.35], [0.30, 0.55, 0.75], [0.85, 0.45, 0.35], [0.45, 0.70, 0.45]][k])
    img = np.ones((h, w, 3)) * base * (0.75 + 0.25 * y[..., None])
    for _ in range(14):                         # Formen
        cx, cy, r = rng.random(), rng.random(), 0.05 + 0.18 * rng.random()
        col = rng.random(3) * 0.8 + 0.1
        m = ((x - cx) ** 2 * (w / h) ** 2 * 4 + (y - cy) ** 2 * 1) < r * r * 0.6
        img[m] = img[m] * 0.25 + col * 0.75
    for _ in range(6):
        y0 = rng.random(); m = np.abs(y - y0 - 0.1 * np.sin(x * 9 + k)) < 0.006
        img[m] = 0.1
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    p = save_png(f'kroepcke_vitrine_{k}', img.astype(np.float32))
    return tex_material(f'Vitrine_Kunst_{k}', p, rough=0.6, emit_s=0.35)


ART = [make_art(k) for k in range(4)]
square_lathe('Koerper_Kern', [(0.40, Z0), (0.40, Z1)], IRON_D, cap_bottom=False, cap_top=False)
for k in range(4):
    framed_face(k)
# Eckpilaster mit Tierkreiszeichen-Medaillons (je Pilaster 3 pro Sichtseite = 12 je Ecke ... insgesamt 12 Zeichen wiederholt)
for k in range(4):
    n, t = face_frame(k); c = (n + t) * (HB - 0.065)
    box(f'Pilaster_{k}', (c.x, c.y, (Z0 + Z1) / 2), (0.15, 0.15, Z1 - Z0), IRON, rot=math.radians(-90 * k))
    for side, vec in (('a', n), ('b', t)):
        for i, z in enumerate(np.linspace(1.62, 3.30, 6)):
            p = c + vec * 0.078
            sphere(f'Medaillon_{k}{side}{i}', (p.x, p.y, z), 0.042, IRON, seg=16, ring=8, scale=(1 if vec.x else 0.35, 0.35 if vec.x else 1, 1.25) if False else ((0.35, 1, 1.25) if abs(vec.x) > abs(vec.y) else (1, 0.35, 1.25)))
            q = c + vec * 0.094
            sphere(f'Tierkreis_{k}{side}{i}', (q.x, q.y, z), 0.018, GOLD, seg=10, ring=6)
# Koerpergesims oben
square_lathe('Koerper_Gesims', [(0.60, 3.55), (0.64, 3.57), (0.66, 3.60), (0.66, 3.64), (0.64, 3.66), (0.64, 3.68)], IRON)

# ---------------------------------------------------------------- 3) Uhrengeschoss z 3.68 .. 4.56 mit 4 Zifferblaettern
ZC = 4.10; HS = 0.64
square_lathe('Uhrgeschoss', [(HS, 3.68), (HS, 4.56)], IRON, cap_bottom=False, cap_top=False)
ROMAN = ['XII', 'I', 'II', 'III', 'IIII', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI']
hands = []
for k in range(4):
    n, t = face_frame(k)
    c = n * HS + Vector((0, 0, ZC))
    torus(f'Luenette_{k}', tuple(c + n * 0.02), 0.355, 0.035, IRON, tuple(n), seg=48, rseg=10)
    torus(f'Luenette_Gold_{k}', tuple(c + n * 0.045), 0.318, 0.010, GOLD, tuple(n), seg=48, rseg=6)
    d = cyl(f'Zifferblatt_{k}', (0, 0, 0), 0.31, 0.03, DIAL, seg=64, axis='Y')
    place(d, n, t, c + n * 0.005)
    # Minutenstriche + roemische Ziffern
    for i in range(60):
        a = 2 * math.pi * i / 60; L = 0.028 if i % 5 == 0 else 0.012
        r0 = 0.285
        rr = -t * math.sin(a) + UP * math.cos(a)
        p = c + n * 0.022 + rr * (r0 - L / 2)
        m = box(f'Strich_{k}_{i}', (0, 0, 0), (0.004 if i % 5 else 0.007, L, 0.003), INK)
        m.matrix_world = basis(rr.cross(n), rr, n, p)
    for i, s in enumerate(ROMAN):
        a = 2 * math.pi * i / 12
        cu = bpy.data.curves.new(f'z{k}{i}', 'FONT'); cu.body = s; cu.size = 0.062; cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
        cu.extrude = 0.0015
        fo = bpy.data.objects.new(f'Ziffer_{k}_{i}', cu); SC.collection.objects.link(fo)
        rr = -t * math.sin(a) + UP * math.cos(a)
        p = c + n * 0.022 + rr * 0.215
        fo.matrix_world = basis(rr.cross(n), rr, n, p)          # Ziffernfuss zur Mitte, Vorderseite zum Betrachter
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(fo.evaluated_get(dg)); me.transform(fo.matrix_world)
        mo = bpy.data.objects.new(f'Ziffer_{k}_{i}_m', me); SC.collection.objects.link(mo); mo.parent = ROOT
        me.materials.append(INK); bpy.data.objects.remove(fo)
    # Zeiger: Ursprung in der Zifferblattmitte, Zeiger zeigt lokal nach +z (12 Uhr); Drehachse = Normale n
    for kind, L, W in (('hour', 0.17, 0.020), ('minute', 0.25, 0.013)):
        bm = bmesh.new()
        prof = [(-W * 0.5, -0.05), (W * 0.5, -0.05), (W * 0.7, L * 0.55), (W * 1.3, L * 0.75), (0, L), (-W * 1.3, L * 0.75), (-W * 0.7, L * 0.55)]
        f0 = [bm.verts.new((u, 0.0, z)) for u, z in prof]; f1 = [bm.verts.new((u, 0.004, z)) for u, z in prof]
        bm.faces.new(f0[::-1]); bm.faces.new(f1)
        for i in range(len(prof)):
            j = (i + 1) % len(prof); bm.faces.new((f0[i], f0[j], f1[j], f1[i]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        h = new_obj(f'Zeiger_{kind}_{k}', bm, INK)
        off = 0.030 if kind == 'hour' else 0.036
        place(h, n, t, c + n * off)
        h['clock_hand'] = kind; h['face'] = k
        hands.append(h)
    sphere(f'Zeigernabe_{k}', tuple(c + n * 0.045), 0.012, GOLD, seg=12, ring=8)
    # Zwickel-Rosetten
    for su in (-1, 1):
        for sz in (-1, 1):
            p = c + n * 0.012 + t * su * 0.48 + Vector((0, 0, sz * 0.36))
            sphere(f'Rosette_{k}_{su}_{sz}', tuple(p), 0.03, GOLD, seg=12, ring=8, scale=(1 if abs(n.x) < 0.5 else 0.4, 1 if abs(n.x) > 0.5 else 0.4, 1))
# Eck-Pilaster mit Kapitell im Uhrgeschoss
for k in range(4):
    n, t = face_frame(k); c = (n + t) * (HS - 0.04)
    box(f'Uhr_Pilaster_{k}', (c.x, c.y, 4.12), (0.13, 0.13, 0.88), IRON, rot=math.radians(-90 * k))
    box(f'Uhr_Kapitell_{k}', (c.x, c.y, 4.50), (0.17, 0.17, 0.06), GOLD, rot=math.radians(-90 * k))

# ---------------------------------------------------------------- 4) Fries mit Gitterband + Kranzgesims
square_lathe('Fries', [(0.66, 4.56), (0.66, 4.76)], IRON, cap_bottom=True, cap_top=False)
for k in range(4):
    n, t = face_frame(k)
    for i in range(16):                         # Rautengitter
        u = -0.56 + i * 0.075
        for sgn in (-1, 1):
            b = box(f'Gitter_{k}_{i}_{sgn}', (0, 0, 0), (0.008, 0.008, 0.20), GOLD)
            p = n * 0.665 + t * u + Vector((0, 0, 4.66))
            b.matrix_world = Matrix(((t.x, n.x, 0, p.x), (t.y, n.y, 0, p.y), (0, 0, 1, p.z), (0, 0, 0, 1))) @ Matrix.Rotation(sgn * 0.36, 4, 'Y')
    # Girlande (Feston) in der Mitte
    for i in range(9):
        a = math.pi * i / 8
        p = n * 0.67 + t * (-0.22 + 0.44 * i / 8) + Vector((0, 0, 4.70 - 0.06 * math.sin(a)))
        sphere(f'Feston_{k}_{i}', tuple(p), 0.022, GOLD, seg=10, ring=6)
cor = [(0.66, 4.76), (0.70, 4.77), (0.72, 4.80), (0.76, 4.83), (0.79, 4.86), (0.80, 4.89), (0.80, 4.92), (0.74, 4.93), (0.70, 4.95)]
square_lathe('Kranzgesims', cor, IRON)
for k in range(4):                              # Zahnschnitt
    n, t = face_frame(k)
    for i in range(22):
        u = -0.70 + i * (1.40 / 21)
        p = n * 0.735 + t * u
        box(f'Zahn_{k}_{i}', (p.x, p.y, 4.79), (0.035 if abs(n.y) > 0.5 else 0.03, 0.03 if abs(n.y) > 0.5 else 0.035, 0.04), IRON_D)

# ---------------------------------------------------------------- 5) gerippte Haube (Klostergewoelbe) + Kugel + Spitze
def ribbed_dome(name, half, z0, H, ribs=18, K=18, amp=0.018):
    bm = bmesh.new(); rings = []
    M = ribs * 4 * 3
    for kk in range(K + 1):
        th = (math.pi / 2) * kk / K
        s = half * math.cos(th) ** 0.85; z = z0 + H * math.sin(th)
        row = []
        for m in range(M):
            p = 4 * m / M; side = int(p); f = p - side
            u = -1 + 2 * f
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            a, b = corners[side], corners[(side + 1) % 4]
            x = a[0] + (b[0] - a[0]) * (u + 1) / 2; y = a[1] + (b[1] - a[1]) * (u + 1) / 2
            rib = amp * abs(math.cos(math.pi * ribs * f)) ** 6 * math.cos(th)
            nx, ny = (x, y)
            ln = math.hypot(nx, ny)
            row.append(bm.verts.new((x * s + nx / ln * rib, y * s + ny / ln * rib, z)))
        rings.append(row)
    for a, b in zip(rings[:-1], rings[1:]):
        for m in range(M):
            mm = (m + 1) % M
            bm.faces.new((a[m], a[mm], b[mm], b[m]))
    top = bm.verts.new((0, 0, z0 + H))
    for m in range(M):
        bm.faces.new((rings[-1][m], rings[-1][(m + 1) % M], top))
    bm.faces.new(rings[0][::-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return new_obj(name, bm, IRON, smooth=True)


square_lathe('Haube_Ring', [(0.70, 4.95), (0.70, 5.00), (0.68, 5.02)], IRON)
ribbed_dome('Haube', 0.68, 5.02, 0.50)
cyl('Hals', (0, 0, 5.56), 0.075, 0.12, IRON, seg=20)
cyl('Hals_Kragen', (0, 0, 5.62), 0.10, 0.025, GOLD, seg=20)
sphere('Kugel', (0, 0, 5.78), 0.165, IRON, seg=32, ring=20)
torus('Kugel_Band', (0, 0, 5.78), 0.166, 0.009, GOLD, (0, 0, 1), seg=40, rseg=6)
cyl('Spitze_Fuss', (0, 0, 5.97), 0.035, 0.05, GOLD, seg=12)
cyl('Spitze', (0, 0, 6.15), 0.022, 0.32, IRON, seg=12, r2=0.004)
for i in range(4):                              # kleine Lilie
    a = math.pi / 2 * i + math.radians(BEARING * 0)
    sphere(f'Lilie_{i}', (0.035 * math.cos(a), 0.035 * math.sin(a), 6.08), 0.022, GOLD, seg=10, ring=6, scale=(1, 0.4, 1.6))

# ---------------------------------------------------------------- 6) Bambus-Kuebel rund um die Uhr
def leaf_texture():
    h, w = 512, 512
    img = np.zeros((h, w, 4), np.float32)
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(70):
        cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        ang = rng.uniform(-2.6, -0.5) if rng.random() < 0.5 else rng.uniform(0.5, 2.6)
        L, Wd = rng.uniform(60, 110), rng.uniform(7, 11)
        dx, dy = xx - cx, yy - cy
        u = dx * math.cos(ang) + dy * math.sin(ang); v = -dx * math.sin(ang) + dy * math.cos(ang)
        m = (u > 0) & (u < L) & (np.abs(v) < Wd * np.sin(np.clip(u / L, 0, 1) * math.pi) ** 0.7)
        g = rng.uniform(0.75, 1.15)
        col = np.array([0.32, 0.50, 0.12]) * g + (u[m] / L)[:, None] * np.array([0.12, 0.10, 0.0])
        img[m, :3] = col; img[m, 3] = 1
    return save_png('bambus_blatt', img)


LEAF = tex_material('Bambus_Blatt', leaf_texture(), rough=0.6, alpha_clip=True)


def planter(name, c, rot):
    P = bpy.data.objects.new(name, None); SC.collection.objects.link(P); P.parent = ROOT
    P.location = c; P.rotation_euler = (0, 0, rot)
    bw, bh = 1.45, 0.88
    # Holzkasten aus Latten
    for i in range(6):
        z = 0.12 + i * 0.125
        for k in range(4):
            n, t = face_frame(k)
            b = box(f'{name}_Latte_{i}_{k}', tuple(n * (bw / 2 - 0.02) + Vector((0, 0, z))), (bw - 0.06 if abs(n.y) > 0.5 else 0.035, 0.035 if abs(n.y) > 0.5 else bw - 0.06, 0.115), WOOD, parent=P)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f'{name}_Pfosten_{sx}_{sy}', (sx * bw / 2, sy * bw / 2, (bh + 0.12) / 2), (0.06, 0.06, bh + 0.12), STEEL, parent=P)
    box(f'{name}_Erde', (0, 0, bh - 0.05), (bw - 0.1, bw - 0.1, 0.04), SOIL, parent=P)
    rng = random.Random(hash(name) & 0xffff)
    for i in range(26):
        x, y = rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55); H = rng.uniform(2.1, 3.3); lean = rng.uniform(-0.12, 0.12)
        o = cyl(f'{name}_Halm_{i}', (x, y, bh + H / 2), 0.012, H, CULM, seg=6, r2=0.006, parent=P)
        o.rotation_euler = (lean, rng.uniform(-0.12, 0.12), 0)
    for i in range(46):
        x, y = rng.uniform(-0.75, 0.75), rng.uniform(-0.75, 0.75); z = bh + rng.uniform(1.0, 3.2)
        s = rng.uniform(0.6, 1.0)
        bm = bmesh.new(); uv = bm.loops.layers.uv.new()
        v = [bm.verts.new((-s / 2, 0, -s / 2)), bm.verts.new((s / 2, 0, -s / 2)), bm.verts.new((s / 2, 0, s / 2)), bm.verts.new((-s / 2, 0, s / 2))]
        f = bm.faces.new(v)
        for l, q in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))): l[uv].uv = q
        o = new_obj(f'{name}_Blatt_{i}', bm, LEAF, parent=P)
        o.location = (x, y, z); o.rotation_euler = (rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), rng.uniform(0, math.pi))
    return P


for i, a in enumerate((45, 135, 225, 315)):
    r = math.radians(a)
    planter(f'Bambus_{i}', (3.7 * math.sin(r), 3.7 * math.cos(r), 0.0), r)

# ---------------------------------------------------------------- Ausrichtung, Fasen, Export
ROOT.rotation_euler = (0, 0, -math.radians(BEARING))
bpy.context.view_layer.update()
for o in list(ROOT.children):
    mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
bpy.context.view_layer.update()
for o in [o for o in SC.objects if o.type == 'MESH' and o.data.users == 1 and len(o.data.polygons) > 0 and o.data.materials and o.data.materials[0] in (IRON, IRON_D, STONE)]:
    if o.name.startswith(('Stufe', 'Sockel', 'Koerper', 'Uhrgeschoss', 'Fries', 'Kranz', 'Haube_Ring', 'Pilaster', 'Uhr_Pilaster')):
        m = o.modifiers.new('bevel', 'BEVEL'); m.width = 0.006; m.segments = 2; m.limit_method = 'ANGLE'; m.angle_limit = math.radians(35)
# Zeiger: Normale im Spielframe als Extra (Blender (x,y,z) -> Spiel (x,z,-y))
for h in hands:
    nw = (h.matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()
    h['face_normal_game'] = [round(nw.x, 5), round(nw.z, 5), round(-nw.y, 5)]
tris = 0
dg = bpy.context.evaluated_depsgraph_get()
for o in SC.objects:
    if o.type == 'MESH':
        me = o.evaluated_get(dg).to_mesh(); tris += sum(len(p.vertices) - 2 for p in me.polygons); o.evaluated_get(dg).to_mesh_clear()
print('KROEPCKE_UHR objects', len([o for o in SC.objects if o.type == 'MESH']), 'tris', tris)
# Draw-Calls sparen: Modifikatoren anwenden, alles ausser den Zeigern je Material zusammenfuehren
from collections import defaultdict
bpy.ops.object.select_all(action='DESELECT')
meshes = [o for o in SC.objects if o.type == 'MESH']
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.convert(target='MESH')
for o in [o for o in SC.objects if o.type == 'MESH' and o.parent is not None]:
    mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
for o in [o for o in SC.objects if o.type == 'EMPTY']: bpy.data.objects.remove(o)
grp = defaultdict(list)
for o in SC.objects:
    if o.type == 'MESH' and 'clock_hand' not in o.keys():
        grp[o.data.materials[0].name if o.data.materials else ''].append(o)
for mname, objs in grp.items():
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1: bpy.ops.object.join()
    bpy.context.view_layer.objects.active.name = 'Kroepcke_Uhr_' + mname
print('KROEPCKE_UHR nodes after join', len([o for o in SC.objects if o.type == 'MESH']))
kw = dict(filepath=OUT_GLB, export_format='GLB', export_yup=True, export_apply=True)
try:
    bpy.ops.export_scene.gltf(**kw, export_extras=True)
except TypeError:
    bpy.ops.export_scene.gltf(**kw)
if OUT_BLEND:
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)

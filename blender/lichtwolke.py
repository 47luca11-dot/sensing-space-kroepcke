"""Lichtwolke (Ulrike Brandi, 2015) ueber dem Kroepcke: gefaltete Metallbaender mit 64 LEDs, an 4 Seilen.
Blender-Frame (x Ost, y Nord, z oben), Werte aus data/lichtwolke.json (Spielframe x, y, z=-Nord)."""
import bpy, bmesh, math, json, sys, random
from mathutils import Vector, Matrix
argv = sys.argv[sys.argv.index('--') + 1:]
J = json.load(open(argv[0])); OUT = argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True); SC = bpy.context.scene
g2b = lambda p: Vector((p[0], -p[2], p[1]))
C = g2b(J['centre']); R = J['diameter'] / 2
rnd = random.Random(15)


def mat(name, rgb, metal, rough, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (*rgb, 1); b.inputs['Metallic'].default_value = metal; b.inputs['Roughness'].default_value = rough
    if emit: b.inputs['Emission Color'].default_value = (1.0, 0.93, 0.8, 1); b.inputs['Emission Strength'].default_value = emit
    return m


ALU = mat('Lichtwolke_Alu_weiss', (0.80, 0.80, 0.78), 0.6, 0.32)
LED = mat('Lichtwolke_LED', (1.0, 0.95, 0.85), 0.0, 0.3, emit=6.0)
CABLE = mat('Stahlseil', (0.25, 0.26, 0.27), 0.9, 0.35)


def obj(name, bm, m):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); me.materials.append(m)
    o = bpy.data.objects.new(name, me); SC.collection.objects.link(o); return o


bm = bmesh.new(); leds = []
for s in range(30):
    ring = rnd.uniform(0.35, 1.0) ** 0.7 * (R - 0.25)
    a0 = rnd.uniform(0, 2 * math.pi); L = rnd.uniform(1.0, 2.6); w = rnd.uniform(0.10, 0.16)
    z0 = rnd.uniform(-0.45, 0.45) * (1 - ring / R) ** 0.5 + 0.12 * math.cos(3 * a0)
    nf = max(4, int(L / 0.16)); prev = None
    tilt = rnd.uniform(-0.5, 0.5)
    for i in range(nf + 1):
        a = a0 + (L / max(ring, 0.4)) * i / nf
        p = C + Vector((ring * math.cos(a), ring * math.sin(a), z0 + (0.06 if i % 2 else -0.06)))
        tdir = Vector((-math.sin(a), math.cos(a), 0)); rdir = Vector((math.cos(a), math.sin(a), 0))
        nrm = (rdir * math.cos(tilt) + Vector((0, 0, 1)) * math.sin(tilt)).normalized()
        v0 = bm.verts.new(p - nrm * w / 2); v1 = bm.verts.new(p + nrm * w / 2)
        if prev: bm.faces.new((prev[0], v0, v1, prev[1]))
        prev = (v0, v1)
        if i % 3 == 1 and len(leds) < 64: leds.append(p - Vector((0, 0, 0.03)))
o = obj('Lichtwolke_Baender', bm, ALU)
s = o.modifiers.new('solid', 'SOLIDIFY'); s.thickness = 0.004
for i, p in enumerate(leds):
    b = bmesh.new(); bmesh.ops.create_uvsphere(b, u_segments=8, v_segments=6, radius=0.025); bmesh.ops.translate(b, vec=p, verts=b.verts)
    obj(f'LED_{i}', b, LED)
# Nabe + Seile
b = bmesh.new(); bmesh.ops.create_cone(b, cap_ends=True, segments=24, radius1=0.35, radius2=0.35, depth=0.05); bmesh.ops.translate(b, vec=C + Vector((0, 0, 0.55)), verts=b.verts)
obj('Lichtwolke_Nabe', b, CABLE)
for i, an in enumerate(J['anchors']):
    A = g2b(an); H = C + Vector((0, 0, 0.55))
    d = A - H; L = d.length
    b = bmesh.new(); bmesh.ops.create_cone(b, cap_ends=True, segments=6, radius1=0.008, radius2=0.008, depth=L)
    rot = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix()
    bmesh.ops.rotate(b, cent=(0, 0, 0), matrix=rot, verts=b.verts); bmesh.ops.translate(b, vec=(H + A) / 2, verts=b.verts)
    obj(f'Seil_{i}', b, CABLE)
    # Wandhaken: kleine Platte, an der Fassade
    b = bmesh.new(); bmesh.ops.create_cube(b, size=1); bmesh.ops.scale(b, vec=(0.2, 0.2, 0.3), verts=b.verts); bmesh.ops.translate(b, vec=A, verts=b.verts)
    obj(f'Haken_{i}', b, CABLE)
    # Abspannseile von der Nabe zum Bandring
    for k in range(3):
        a = 2 * math.pi * (k + i / 4) / 3
        Q = C + Vector((1.6 * math.cos(a), 1.6 * math.sin(a), 0.0)); d2 = Q - H
        b = bmesh.new(); bmesh.ops.create_cone(b, cap_ends=True, segments=5, radius1=0.004, radius2=0.004, depth=d2.length)
        bmesh.ops.rotate(b, cent=(0, 0, 0), matrix=Vector((0, 0, 1)).rotation_difference(d2.normalized()).to_matrix(), verts=b.verts)
        bmesh.ops.translate(b, vec=(H + Q) / 2, verts=b.verts); obj(f'Abspannung_{i}_{k}', b, CABLE)
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_yup=True, export_apply=True)
print('LICHTWOLKE', len(leds), 'LEDs')

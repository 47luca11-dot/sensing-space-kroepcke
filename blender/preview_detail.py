import bpy, glob, os, math, sys
from mathutils import Vector
D = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/rhino_export/"
OUT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/blender/previews/"
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
root = bpy.data.objects.new('Detail_Opernplatz', None); sc.collection.objects.link(root)
for fn in sorted(glob.glob(D + '*.glb')):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=fn)
    for o in set(bpy.data.objects) - before:
        if o.parent is None: o.parent = root
root.location = (84.117, -141.706, -0.16)     # Rhino-Opernplatz-Ursprung -> Kroepcke-Uhr
print('objects', len(bpy.data.objects))
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.75, 0.8, 0.88, 1); bg.inputs[1].default_value = 1.0
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.5; so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
so.rotation_euler = (math.radians(50), 0, math.radians(140))
try: sc.render.engine = 'BLENDER_EEVEE'
except TypeError as e: print(e); sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.render.resolution_x, sc.render.resolution_y = 1600, 900
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 24; cam.data.clip_end = 3000
def shot(name, loc, tgt):
    cam.location = Vector(loc); d = Vector(tgt) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = OUT + name + '.png'; bpy.ops.render.render(write_still=True)
shot('detail_aerial', (-60, 120, 160), (90, -150, 0))
shot('detail_kroepcke_eye', (0, 0, 1.7), (40, -60, 4))
shot('detail_oper', (60, -40, 1.7), (130, -170, 10))

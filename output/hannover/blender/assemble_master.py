"""Blender-Gesamtszene Hannover (headless): importiert alle Roh-GLBs (Spielframe) in Collections, Kamera/Licht,
speichert blender/hannover_master.blend und rendert Vorschauen. Erneut ausfuehren, sobald neue Teile da sind."""
import bpy, glob, os, math, sys
from mathutils import Vector
R = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/"
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
RENDER = 'render' in argv
bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
SRC = [('Detail_Opernplatz', R + 'data/detail_raw/*.glb'), ('Wahrzeichen', R + 'data/landmarks_raw/*.glb'),
       ('Gebaeude_LoD2_detailliert', R + 'data/buildings_raw/*.glb'), ('Boden_Strassen', R + 'data/ground_raw/*.glb'),
       ('TIB', R + 'data/tib_raw/tib_*.glb'), ('TIB_Umgebung', R + 'data/tib_raw/umgebung_[!n]*.glb')]
for cname, pat in SRC:
    files = sorted(glob.glob(pat))
    if not files: continue
    col = bpy.data.collections.new(cname); sc.collection.children.link(col)
    for fn in files:
        if 'collision' in os.path.basename(fn): continue
        before = set(bpy.data.objects)
        try: bpy.ops.import_scene.gltf(filepath=fn)
        except Exception as e: print('IMPORT FAIL', fn, e); continue
        for o in set(bpy.data.objects) - before:
            for c in list(o.users_collection): c.objects.unlink(o)
            col.objects.link(o)
    print('collection', cname, len(col.objects))
w = bpy.data.worlds.new('Himmel'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.72, 0.78, 0.86, 1); bg.inputs[1].default_value = 1.0
sun = bpy.data.lights.new('Sonne', 'SUN'); sun.energy = 3.2; sun.angle = math.radians(2)
so = bpy.data.objects.new('Sonne', sun); sc.collection.objects.link(so); so.rotation_euler = (math.radians(52), 0, math.radians(215))
try: sc.render.engine = 'BLENDER_EEVEE'
except TypeError: sc.render.engine = 'BLENDER_EEVEE_NEXT'
cam = bpy.data.objects.new('Kamera_Kroepcke', bpy.data.cameras.new('Kamera_Kroepcke')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.clip_end = 5000
def aim(loc, tgt, lens):
    cam.location = Vector(loc); cam.rotation_euler = (Vector(tgt) - cam.location).to_track_quat('-Z', 'Y').to_euler(); cam.data.lens = lens
aim((6, -9, 1.7), (-10, 40, 4), 24)            # Blender: y = Nord. Blick vom Kroepcke nach Norden (Bahnhofstrasse)
sc.render.resolution_x, sc.render.resolution_y = 1600, 900
bpy.ops.wm.save_as_mainfile(filepath=R + 'blender/hannover_master.blend')
if RENDER:
    os.makedirs(R + 'blender/previews', exist_ok=True)
    for name, loc, tgt, lens in (('master_kroepcke_nord', (6, -9, 1.7), (-10, 40, 4), 24),
                                 ('master_kroepcke_sued', (-4, 8, 1.7), (20, -60, 4), 24),
                                 ('master_aerial', (-220, 260, 260), (40, -60, 0), 30),
                                 ('master_tib', (-1260, 640, 90), (-1320, 735, 5), 30)):
        aim(loc, tgt, lens); sc.render.filepath = R + f'blender/previews/{name}.png'; bpy.ops.render.render(write_still=True)

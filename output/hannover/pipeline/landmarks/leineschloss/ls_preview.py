"""Leineschloss -- Vorschauen (Blender 5 headless, Eevee 1280x720, Rueckseiten-Culling AN).
Kontext: data/ground_raw + data/buildings_raw Kacheln ums Schloss (alte Fassung im Grundriss wird entfernt) + Marktkirche/Altes Rathaus.
Aufruf: blender -b --factory-startup -P ls_preview.py -- <model.glb> <replace.json> <out_prefix> <shots|all> <cams.json> [--noctx] [--samples=N]"""
import bpy, bmesh, sys, os, json, math
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, REPL, PREFIX, SHOTS, CAMF = argv[:5]
NOCTX = '--noctx' in argv
SAMPLES = int(next((a.split('=')[1] for a in argv if a.startswith('--samples=')), '24'))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
CAMS = json.load(open(CAMF))
shots = list(CAMS) if SHOTS == 'all' else SHOTS.split(',')


def B(p):   # Spielframe -> Blender
    return Vector((p[0], -p[2], p[1]))


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


model = import_glb(GLB)
print('model objects', len(model))
fp = json.load(open(REPL))['footprints']


def inside(x, y, poly):
    c = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1: c = not c
    return c


polysB = [[(p[0], -p[1]) for p in f] for f in fp]
TILES_X = range(-6, -2); TILES_Z = range(3, 6)
_ax = [p[0] for pl in polysB for p in pl]; _ay = [p[1] for pl in polysB for p in pl]
BB = (min(_ax) - 1, max(_ax) + 1, min(_ay) - 1, max(_ay) + 1)
if not NOCTX:
    for tx in TILES_X:
        for tz in TILES_Z:
            for sub, pat in (('ground_raw', 'ground_%d_%d.glb'), ('buildings_raw', 'bldg_%d_%d.glb'), ('buildings_raw', 'bldg_%d_%d_win.glb')):
                p = os.path.join(ROOT, 'data', sub, pat % (tx, tz))
                if not os.path.isfile(p): continue
                objs = import_glb(p)
                if sub != 'buildings_raw': continue
                for o in objs:
                    if o.type != 'MESH': continue
                    me = o.data; mw = o.matrix_world
                    wb = [mw @ Vector(c) for c in o.bound_box]
                    if max(v.x for v in wb) < BB[0] or min(v.x for v in wb) > BB[1] or max(v.y for v in wb) < BB[2] or min(v.y for v in wb) > BB[3]:
                        continue
                    if me.users > 1:
                        c = sum(wb, Vector()) / 8
                        if any(inside(c.x, c.y, pl) for pl in polysB): bpy.data.objects.remove(o)
                        continue
                    bm = bmesh.new(); bm.from_mesh(me)
                    kill = []
                    for f in bm.faces:
                        c = mw @ f.calc_center_median()
                        if BB[0] <= c.x <= BB[1] and BB[2] <= c.y <= BB[3] and any(inside(c.x, c.y, pl) for pl in polysB): kill.append(f)
                    if kill: bmesh.ops.delete(bm, geom=kill, context='FACES')
                    bm.to_mesh(me); bm.free()
    for extra in ('marktkirche', 'altes_rathaus'):
        p = os.path.join(ROOT, 'data', 'landmarks_raw', extra + '.glb')
        if os.path.isfile(p): import_glb(p)
for m in bpy.data.materials:
    m.use_backface_culling = True
    try: m.use_backface_culling_shadow = False
    except Exception: pass

w = bpy.data.worlds.new('W'); SC.world = w; w.use_nodes = True
bg = next(n for n in w.node_tree.nodes if n.type == 'BACKGROUND')
bg.inputs['Color'].default_value = (0.62, 0.68, 0.76, 1); bg.inputs['Strength'].default_value = 0.85
sun = bpy.data.lights.new('Sonne', 'SUN'); sun.energy = 3.2; sun.angle = math.radians(1.5)
so = bpy.data.objects.new('Sonne', sun); SC.collection.objects.link(so)
for eng in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
    try:
        SC.render.engine = eng; break
    except TypeError:
        continue
try: SC.eevee.taa_render_samples = SAMPLES
except Exception: pass
SC.render.resolution_x, SC.render.resolution_y = 1280, 720
SC.render.image_settings.file_format = 'PNG'
try: SC.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
cam = bpy.data.cameras.new('Cam'); co = bpy.data.objects.new('Cam', cam); SC.collection.objects.link(co); SC.camera = co
cam.clip_start = 0.1; cam.clip_end = 3000
for s in shots:
    c, t, lens = CAMS[s]
    co.location = B(c)
    d = B(t) - B(c)
    co.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cam.lens = lens
    az = math.atan2(d.y, d.x) + math.radians(40)
    so.rotation_euler = (math.radians(48), 0, az - math.pi / 2)
    out = PREFIX + s + '.png'
    SC.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print('RENDERED', out)

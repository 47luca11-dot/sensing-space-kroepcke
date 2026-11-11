"""Headless-Blender-Vorschau: Bodenkacheln + Moeblierung + Detailmodell (Rhino, mit Versatz) + LoD2-Kontext.
blender -b --factory-startup -P preview_blender.py -- [shot,...]"""
import bpy, sys, os, glob, math, json
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..')).replace('\\', '/') + '/'
GROUND = ROOT + 'web/assets/city/ground/'
DETAIL = ROOT + 'data/rhino_export/'
PREV = ROOT + 'blender/previews/'
CACHE = ROOT + 'data/ground_cache/'
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
only = set(argv[0].split(',')) if argv and argv[0] != 'all' else None
lowq = '--fast' in argv

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
def imp(fn, parent=None):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=fn)
    new = set(bpy.data.objects) - before
    if parent:
        for o in new:
            if o.parent is None: o.parent = parent
    return new
# Boden
for fn in sorted(glob.glob(GROUND + 'ground_*.glb')): imp(fn)
if os.path.exists(GROUND + 'furniture.glb'): imp(GROUND + 'furniture.glb')
# Detailmodell (Rhino-glTF -> Spiel: +(84.117, -0.16, 141.706); Blender: (x, -z, y))
root = bpy.data.objects.new('Detail_Opernplatz', None); sc.collection.objects.link(root)
for fn in sorted(glob.glob(DETAIL + '*.glb')):
    imp(fn, root)
root.location = (84.117, -141.706, -0.16)
if os.path.exists(CACHE + 'lod2_preview.glb'): imp(CACHE + 'lod2_preview.glb')
# Uhr/Landmarken falls vorhanden
for fn in sorted(glob.glob(ROOT + 'web/assets/city/landmarks/*.glb')):
    try: imp(fn)
    except Exception as e: print('landmark import failed', fn, e)
print('objects', len(bpy.data.objects))

# Welt: heller Himmelsverlauf + Sonne
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = nt.nodes['Background']
grad = nt.nodes.new('ShaderNodeTexGradient'); grad.gradient_type = 'LINEAR'
tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].color = (0.80, 0.84, 0.88, 1); ramp.color_ramp.elements[1].color = (0.38, 0.55, 0.80, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
bg.inputs['Strength'].default_value = 0.9
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.6; sun.angle = math.radians(1.5)
so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
so.rotation_euler = (math.radians(48), 0, math.radians(215))
try: sc.render.engine = 'BLENDER_EEVEE'
except TypeError: sc.render.engine = 'BLENDER_EEVEE_NEXT'
ee = sc.eevee
for attr, val in (('taa_render_samples', 16 if lowq else 48), ('use_shadows', True), ('use_raytracing', False)):
    try: setattr(ee, attr, val)
    except Exception: pass
try: sc.view_settings.view_transform = 'AgX'
except Exception: pass
sc.render.resolution_x, sc.render.resolution_y = (1280, 720) if lowq else (1600, 900)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.clip_start = 0.05; cam.data.clip_end = 3000

def G(x, y, z): return Vector((x, -z, y))          # Spiel -> Blender
def shot(name, loc, tgt, lens=24):
    if only and name not in only: return
    cam.location = G(*loc); d = G(*tgt) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler(); cam.data.lens = lens
    sc.render.filepath = PREV + 'ground_' + name + '.png'
    bpy.ops.render.render(write_still=True); print('rendered', name)

def top(name, cx, cz, size, res=1400):
    if only and name not in only: return
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = size
    cam.location = G(cx, 300, cz); cam.rotation_euler = (0, 0, 0)
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.filepath = PREV + 'ground_' + name + '.png'
    bpy.ops.render.render(write_still=True); print('rendered', name)
    sc.render.resolution_x, sc.render.resolution_y = rx, ry; cam.data.type = 'PERSP'
U = {}
try:
    idx = json.load(open(GROUND + 'index.json')); U = {u['id']: u for u in idx.get('ubahn_entrances', [])}
except Exception: pass
shot('aerial', (-170, 190, -260), (40, 0, 40), 24)
shot('aerial_kroepcke', (-90, 85, -150), (30, 0, -30), 26)
shot('eye_bahnhofstrasse', (0, 1.7, 0), (0, 1.7, -100), 24)
shot('eye_promenade_axis', (-12, 1.7, -8), (52, 0.5, -110), 24)
shot('seam_west', (-92, 1.7, 212), (-45, 0.0, 170), 24)
shot('seam_north', (125, 1.7, -48), (122, 0.0, 15), 26)
shot('seam_kroepcke', (-31, 1.7, -22), (10, 0.0, 8), 24)
shot('seam_aerial', (-30, 48, -70), (40, 0, 20), 28)
shot('promenade', (-24, 8, -8), (20, -4, -60), 24)
shot('promenade_low', (6, -3.0, -50), (30, -3.0, -84), 24)
shot('street_junction', (352, 1.7, 250), (366, 0.0, 320), 24)
shot('steintor_tram', (-455, 1.7, -160), (-462, 0.0, -240), 24)
shot('trees_close', (122, 1.7, -14), (127, 6.0, -50), 26)
if U:
    u = U.get('ub353115733') or list(U.values())[0]
    p = u['top']; d = u['dir']; y = u['y_top']
    shot('ubahn', (p[0] - d[0] * 4 + d[1] * 3, y + 4.5, p[1] - d[1] * 4 - d[0] * 3), (p[0] + d[0] * 6, y - 2.5, p[1] + d[1] * 6), 24)
    shot('ubahn_down', (p[0] - d[0] * 0.5, y + 1.65, p[1] - d[1] * 0.5), (p[0] + d[0] * 9, y - 3.4, p[1] + d[1] * 9), 20)
top('top_kroepcke', -10, -30, 120)
top('top_steintor', -460, -220, 90)
top('top_junction', 360, 300, 90)
top('top_overview', 0, 0, 1150, 1600)
top('top_seam_ne', 140, -10, 120)
top('top_seam_w', -70, 160, 120)

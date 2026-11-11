"""Blender (headless) preview renders of the Anzeiger landmark in context (ground tiles + LoD2 neighbours),
backface culling ON for every material (game renders FrontSide only).
blender -b --factory-startup -P preview.py -- <landmark.glb> <out_prefix> [shot,shot,...] [--samples=N]"""
import bpy, bmesh, sys, os, json, math, glob
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUTP = argv[0], argv[1]
SHOTS = argv[2].split(',') if len(argv) > 2 and not argv[2].startswith('--') else None
SAMPLES = int(next((a.split('=')[1] for a in argv if a.startswith('--samples=')), 48))

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

# ---------------- context: ground tiles
for tx in (-7, -6, -5, -4):
    for tz in (-5, -4, -3, -2):
        f = ROOT + 'data/ground_raw/ground_%d_%d.glb' % (tx, tz)
        if os.path.exists(f): bpy.ops.import_scene.gltf(filepath=f)
# ---------------- context: LoD2 neighbours (npz)
meta = json.load(open(os.path.join(HERE, 'build', 'ctx_meta.json')))
npz = np.load(os.path.join(HERE, 'build', 'ctx_parts.npz'))
for oname, od in meta['objects'].items():
    V = npz['V_' + oname].astype(np.float64); M = npz['M_' + oname]
    Vb = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1); n = len(Vb); nt = n // 3
    me = bpy.data.meshes.new(oname)
    me.vertices.add(n); me.vertices.foreach_set('co', Vb.ravel())
    me.loops.add(n); me.loops.foreach_set('vertex_index', np.arange(n, dtype=np.int32))
    me.polygons.add(nt); me.polygons.foreach_set('loop_start', np.arange(0, n, 3, dtype=np.int32)); me.polygons.foreach_set('loop_total', np.full(nt, 3, np.int32))
    me.polygons.foreach_set('material_index', M.astype(np.int32))
    for mn in od['materials']:
        m = bpy.data.materials.new(mn); m.use_nodes = True
        b = next(x for x in m.node_tree.nodes if x.type == 'BSDF_PRINCIPLED')
        c = meta['materials'][mn]['color']; b.inputs['Base Color'].default_value = (c[0], c[1], c[2], 1); b.inputs['Roughness'].default_value = 0.9
        me.materials.append(m)
    me.update(); me.shade_flat()
    ob = bpy.data.objects.new(oname, me); sc.collection.objects.link(ob)
# ---------------- context: procedural facade tiles (data/buildings_raw)
for f in meta.get('bldg_tiles', []):
    try: bpy.ops.import_scene.gltf(filepath=ROOT + 'data/buildings_raw/' + f)
    except Exception as e: print('PREVIEW tile import failed', f, e)
# ---------------- landmark
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=GLB)
lm = [o for o in set(bpy.data.objects) - before]
print('PREVIEW landmark objects', len(lm))
for m in bpy.data.materials:
    m.use_backface_culling = True
    try: m.use_backface_culling_shadow = False
    except Exception: pass

# ---------------- world / sun
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
sky = nt.nodes.new('ShaderNodeTexSky')
ok = False
for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'HOSEK_WILKIE', 'PREETHAM'):
    try: sky.sky_type = t; ok = True; break
    except Exception: pass
SUN_EL, SUN_AZ = math.radians(36), math.radians(130)     # morning sun from SE (east facade lit)
try:
    sky.sun_elevation = SUN_EL; sky.sun_rotation = SUN_AZ
except Exception: pass
nt.links.new(sky.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.35
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 4.2; sun.angle = math.radians(1.0)
so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
# blender sun direction: points along -Z of the object; azimuth measured from +Y (north) clockwise to the east
d = Vector((math.sin(SUN_AZ) * math.cos(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL)))
so.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()

# ---------------- render settings
for eng in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
    try: sc.render.engine = eng; break
    except TypeError: pass
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.render.resolution_percentage = 100
try: sc.eevee.taa_render_samples = SAMPLES
except Exception: pass
for attr, val in (('use_shadows', True), ('use_raytracing', True), ('shadow_ray_count', 2)):
    try: setattr(sc.eevee, attr, val)
    except Exception: pass
try: sc.view_settings.view_transform = 'AgX'
except Exception: pass
sc.view_settings.look = 'None' if 'None' in [i.identifier for i in bpy.types.ColorManagedViewSettings.bl_rna.properties['look'].enum_items] else sc.view_settings.look

shots = json.load(open(os.path.join(HERE, 'build', 'shots.json')))
cam = bpy.data.cameras.new('cam'); co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
cam.clip_start = 0.1; cam.clip_end = 3000
def gb(p): return Vector((p[0], -p[2], p[1]))
for name, s in shots.items():
    if SHOTS and name not in SHOTS: continue
    c = gb(s['cam']); t = gb(s['tgt'])
    co.location = c
    co.rotation_euler = (t - c).to_track_quat('-Z', 'Y').to_euler()
    cam.lens = s['lens']
    sc.render.filepath = OUTP + name + '.png'
    bpy.ops.render.render(write_still=True)
    print('RENDERED', sc.render.filepath)

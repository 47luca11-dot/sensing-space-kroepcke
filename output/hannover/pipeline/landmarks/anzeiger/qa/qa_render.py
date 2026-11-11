"""QA renders of the Anzeiger landmark (independent reviewer).
blender -b --factory-startup -P qa_render.py -- <landmark.glb> <out_prefix> <shots.json> [--holes] [--only=a,b]
--holes: landmark only, magenta emissive world, no context (back-face culled holes show magenta)."""
import bpy, sys, os, json, math
import numpy as np
from mathutils import Vector

ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
PIPE = ROOT + 'pipeline/landmarks/anzeiger/build/'
argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUTP, SHOTF = argv[0], argv[1], argv[2]
HOLES = '--holes' in argv
ONLY = next((a.split('=')[1].split(',') for a in argv if a.startswith('--only=')), None)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

if not HOLES:
    for tx in (-7, -6, -5, -4):
        for tz in (-5, -4, -3, -2):
            f = ROOT + 'data/ground_raw/ground_%d_%d.glb' % (tx, tz)
            if os.path.exists(f):
                bpy.ops.import_scene.gltf(filepath=f)
    meta = json.load(open(PIPE + 'ctx_meta.json'))
    npz = np.load(PIPE + 'ctx_parts.npz')
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
        me.update()
        ob = bpy.data.objects.new(oname, me); sc.collection.objects.link(ob)
    for f in meta.get('bldg_tiles', []):
        try:
            bpy.ops.import_scene.gltf(filepath=ROOT + 'data/buildings_raw/' + f)
        except Exception as e:
            print('QA tile import failed', f, e)

before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=GLB)
print('QA landmark objects', len(set(bpy.data.objects) - before))
for m in bpy.data.materials:
    m.use_backface_culling = True
    try: m.use_backface_culling_shadow = False
    except Exception: pass

w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
if HOLES:
    bg.inputs[0].default_value = (1.0, 0.0, 1.0, 1.0); bg.inputs[1].default_value = 1.0
    lp = nt.nodes.new('ShaderNodeLightPath'); bg2 = nt.nodes.new('ShaderNodeBackground'); bg2.inputs[0].default_value = (0.8, 0.8, 0.8, 1); bg2.inputs[1].default_value = 0.6
    mix = nt.nodes.new('ShaderNodeMixShader'); out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
    nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs[0]); nt.links.new(bg2.outputs[0], mix.inputs[1]); nt.links.new(bg.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], out.inputs[0])
    SUN_EL, SUN_AZ = math.radians(50), math.radians(200)
else:
    sky = nt.nodes.new('ShaderNodeTexSky')
    for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'HOSEK_WILKIE', 'PREETHAM'):
        try:
            sky.sky_type = t; break
        except Exception:
            pass
    SUN_EL, SUN_AZ = math.radians(40), math.radians(115)
    try:
        sky.sun_elevation = SUN_EL; sky.sun_rotation = SUN_AZ
    except Exception:
        pass
    nt.links.new(sky.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.35
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 4.0; sun.angle = math.radians(1.0)
so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
d = Vector((math.sin(SUN_AZ) * math.cos(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL)))
so.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()

for eng in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
    try:
        sc.render.engine = eng; break
    except TypeError:
        pass
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.render.resolution_percentage = 100
try: sc.eevee.taa_render_samples = 32
except Exception: pass
for attr, val in (('use_shadows', True), ('use_raytracing', True)):
    try: setattr(sc.eevee, attr, val)
    except Exception: pass
try:
    sc.view_settings.view_transform = 'Standard' if HOLES else 'AgX'
except Exception:
    pass

shots = json.load(open(SHOTF))
cam = bpy.data.cameras.new('cam'); co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
cam.clip_start = 0.05; cam.clip_end = 3000
def gb(p): return Vector((p[0], -p[2], p[1]))
for name, s in shots.items():
    if ONLY and name not in ONLY: continue
    c = gb(s['cam']); t = gb(s['tgt'])
    co.location = c
    co.rotation_euler = (t - c).to_track_quat('-Z', 'Y').to_euler()
    cam.lens = s['lens']
    sc.render.filepath = OUTP + name + '.png'
    bpy.ops.render.render(write_still=True)
    print('RENDERED', sc.render.filepath)


"""Hole finder (Blender headless): landmark only, backface culling ON, magenta world -> every see-through spot
(missing face / inverted normal) shows magenta. Cameras in game frame from a json list [[name, cam, tgt, lens], ...].
blender -b --factory-startup -P debug_view.py -- <landmark.glb> <out_prefix> <cams.json>"""
import bpy, sys, json, math
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUTP, CAMS = argv[0], argv[1], argv[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=GLB)
for m in bpy.data.materials:
    m.use_backface_culling = True
w = bpy.data.worlds.new('w'); sc.world = w
w.color = (1.0, 0.0, 1.0)
w.use_nodes = True
nt = w.node_tree
bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
bg.inputs[0].default_value = (0.7, 0.7, 0.7, 1.0); bg.inputs[1].default_value = 0.8          # lighting: neutral grey
bm = nt.nodes.new('ShaderNodeBackground'); bm.inputs[0].default_value = (1.0, 0.0, 1.0, 1.0)  # camera rays: magenta
lp = nt.nodes.new('ShaderNodeLightPath'); mx = nt.nodes.new('ShaderNodeMixShader')
out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
nt.links.new(lp.outputs['Is Camera Ray'], mx.inputs[0]); nt.links.new(bg.outputs[0], mx.inputs[1]); nt.links.new(bm.outputs[0], mx.inputs[2])
nt.links.new(mx.outputs[0], out.inputs['Surface'])
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.0
so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so); so.rotation_euler = (math.radians(50), 0, math.radians(140))
for eng in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH'):
    try: sc.render.engine = eng; break
    except TypeError: pass
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
try: sc.eevee.taa_render_samples = 8
except Exception: pass
try: sc.view_settings.view_transform = 'Standard'
except Exception: pass
cam = bpy.data.cameras.new('cam'); co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
cam.clip_start = 0.05; cam.clip_end = 2000
def gb(p): return Vector((p[0], -p[2], p[1]))
for name, c, t, lens in json.load(open(CAMS)):
    c = gb(c); t = gb(t)
    co.location = c; co.rotation_euler = (t - c).to_track_quat('-Z', 'Y').to_euler(); cam.lens = lens
    sc.render.filepath = OUTP + name + '.png'
    bpy.ops.render.render(write_still=True)
    print('RENDERED', sc.render.filepath)

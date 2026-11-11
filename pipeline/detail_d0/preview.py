"""Headless Blender preview for the D0 window upgrade.

blender -b --factory-startup -P preview.py -- <before|after> <view1,view2|all> [--fast] [--cams cams.json] [--prefix d0_]

before: walls/openings from data/detail_raw_orig, everything else from data/detail_raw
after : everything from data/detail_raw incl. d0_windows.glb
Renders 1600x900 Eevee (sun + sky, AgX) to blender/previews/<prefix><set>_<view>.png
"""
import bpy, json, os, sys, math, time
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RAW = os.path.join(ROOT, 'data', 'detail_raw')
ORIG = os.path.join(ROOT, 'data', 'detail_raw_orig')
OUT = os.path.join(ROOT, 'blender', 'previews')
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
SET = argv[0] if argv else 'after'
VIEWS = argv[1].split(',') if len(argv) > 1 else ['all']
FAST = '--fast' in argv
PREFIX = 'd0_'
CAMS = os.path.join(HERE, 'cams.json')
OUTDIR = OUT
for i, a in enumerate(argv):
    if a == '--cams':
        CAMS = argv[i + 1]
    if a == '--prefix':
        PREFIX = argv[i + 1]
    if a == '--outdir':
        OUTDIR = argv[i + 1]

FILES = ['bldg_walls', 'bldg_openings', 'bldg_roof', 'bldg_base', 'street', 'ground', 'green', 'green_trees',
         'street_furniture', 'kiosk', 'landmark_opernhaus', 'landmark_tiefgarage_pavillon', 'landmark_mahnmal', 'fix']


def g2b(p):
    return Vector((p[0], -p[2], p[1]))


def setup_scene():
    sc = bpy.context.scene
    try:
        sc.render.engine = 'BLENDER_EEVEE'
    except TypeError:
        sc.render.engine = 'BLENDER_EEVEE_NEXT'
    sc.render.resolution_x, sc.render.resolution_y = (1600, 900)
    sc.render.resolution_percentage = 50 if FAST else 100
    items = [i.identifier for i in bpy.types.ColorManagedViewSettings.bl_rna.properties['view_transform'].enum_items]
    sc.view_settings.view_transform = 'AgX' if 'AgX' in items else 'Filmic'
    try:
        sc.eevee.taa_render_samples = 16 if FAST else 32
        sc.eevee.use_shadows = True
        sc.eevee.use_raytracing = True
        sc.eevee.shadow_resolution_scale = 1.0
        sc.eevee.shadow_pool_size = '1024'
    except Exception:
        pass
    w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
    nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
    bg2 = nt.nodes.new('ShaderNodeBackground')
    tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.0; ramp.color_ramp.elements[0].color = (0.80, 0.84, 0.88, 1)
    ramp.color_ramp.elements[1].position = 0.45; ramp.color_ramp.elements[1].color = (0.50, 0.62, 0.80, 1)
    nt.links.new(tc.outputs['Generated'], sep.inputs[0]); nt.links.new(sep.outputs['Z'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bg2.inputs['Color']); bg2.inputs['Strength'].default_value = 1.0
    lp = nt.nodes.new('ShaderNodeLightPath'); mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs['Fac'])
    nt.links.new(bg.outputs['Background'], mix.inputs[1]); nt.links.new(bg2.outputs['Background'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    try:
        sky = nt.nodes.new('ShaderNodeTexSky')
        its = [i.identifier for i in sky.bl_rna.properties['sky_type'].enum_items]
        for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'SINGLE_SCATTERING', 'HOSEK_WILKIE'):
            if t in its:
                sky.sky_type = t; break
        try:
            sky.sun_elevation = math.radians(38); sky.sun_rotation = math.radians(215)
        except Exception:
            pass
        hsv = nt.nodes.new('ShaderNodeHueSaturation'); hsv.inputs['Saturation'].default_value = 0.45
        nt.links.new(sky.outputs['Color'], hsv.inputs['Color']); nt.links.new(hsv.outputs['Color'], bg.inputs['Color'])
        bg.inputs['Strength'].default_value = 0.18
    except Exception as e:
        print('sky', e)
        bg.inputs['Color'].default_value = (0.75, 0.8, 0.88, 1); bg.inputs['Strength'].default_value = 0.8
    sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.4; sun.color = (1.0, 0.95, 0.88); sun.angle = math.radians(1.0)
    so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(52), 0, math.radians(215))
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.clip_start = 0.05; cam.data.clip_end = 3000
    return sc, cam


def load(path):
    t = time.time()
    try:
        bpy.ops.import_scene.gltf(filepath=path)
    except Exception as e:
        print('import fail', path, e)
        return
    print('loaded', os.path.basename(path), '%.1fs' % (time.time() - t))


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc, cam = setup_scene()
    cams = json.load(open(CAMS, encoding='utf-8'))
    if VIEWS != ['all']:
        cams = {k: v for k, v in cams.items() if k in VIEWS}
    for f in FILES:
        src = RAW
        if SET == 'before' and f in ('bldg_walls', 'bldg_openings'):
            src = ORIG
        p = os.path.join(src, f + '.glb')
        if os.path.exists(p):
            load(p)
    if SET != 'before':
        p = os.path.join(RAW, 'd0_windows.glb')
        if os.path.exists(p):
            load(p)
    # glass as the game shows it (world.js fixMaterial): colour*0.6, opacity 0.35, glossy, no transmission
    if '--raw-glass' not in argv:
        for m in bpy.data.materials:
            if not m.use_nodes:
                continue
            b = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if b is None:
                continue
            tw = b.inputs.get('Transmission Weight') or b.inputs.get('Transmission')
            if tw is not None and tw.default_value > 0 and not tw.is_linked:
                c = b.inputs['Base Color'].default_value
                b.inputs['Base Color'].default_value = (c[0] * 0.6, c[1] * 0.6, c[2] * 0.6, 1)
                tw.default_value = 0.0
                b.inputs['Roughness'].default_value = 0.05
                b.inputs['Alpha'].default_value = 0.35
                try:
                    m.surface_render_method = 'BLENDED'
                except Exception:
                    m.blend_method = 'BLEND'
    os.makedirs(OUTDIR, exist_ok=True)
    for k, c in cams.items():
        cam.location = g2b(c['loc'])
        d = g2b(c['tgt']) - cam.location
        cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        cam.data.lens = c.get('lens', 24)
        if 'ortho' in c:
            cam.data.type = 'ORTHO'; cam.data.ortho_scale = c['ortho']
        else:
            cam.data.type = 'PERSP'
        sc.render.filepath = os.path.join(OUTDIR, PREFIX + SET + '_' + k + '.png')
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print('render', k, '%.1fs' % (time.time() - t))


main()

"""Headless Blender preview: blender -b --factory-startup -P preview_blender.py -- <tag> <signage.glb|none> view1 view2 ...
Imports the cropped scene of each view (prep_preview.py) + the signage overlay, backface culling ON for every
material (the game renders FrontSide only), glass as 35 % alpha like the game, Eevee 1280x720."""
import glob
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
TAG, SIGN, VIEWS = argv[0], argv[1], argv[2:]
TAG = "" if TAG == "-" else TAG
SCR = os.environ.get("SIGNAGE_SCRATCH", "C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/"
                     "b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/signage")
OUT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/blender/previews/"
CAMS = json.load(open(os.path.join(SCR, "prev", "cams.json")))


def g2b(p):
    return Vector((p[0], -p[2], p[1]))


def setup_materials():
    for m in bpy.data.materials:
        m.use_backface_culling = True
        if not m.use_nodes:
            continue
        bs = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bs is None:
            continue
        nm = m.name.lower()
        if "glass" in nm or "vitrine" in nm:
            try:
                bs.inputs["Transmission Weight"].default_value = 0.0
            except KeyError:
                pass
            bs.inputs["Alpha"].default_value = 0.35
            if "vitrine" not in nm:
                bs.inputs["Base Color"].default_value = (0.05, 0.06, 0.07, 1.0)
            bs.inputs["Roughness"].default_value = 0.05
            try:
                m.surface_render_method = "BLENDED"
            except Exception:
                try:
                    m.blend_method = "BLEND"
                except Exception:
                    pass


def render(view):
    c = CAMS[view]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    for fn in sorted(glob.glob(os.path.join(c["dir"], "*.glb"))):
        bpy.ops.import_scene.gltf(filepath=fn)
    if SIGN != "none" and os.path.isfile(SIGN):
        bpy.ops.import_scene.gltf(filepath=SIGN)
    setup_materials()
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.62, 0.70, 0.80, 1)
    bg.inputs[1].default_value = 0.9
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.6
    sun.angle = math.radians(3)
    so = bpy.data.objects.new("sun", sun)
    sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(48), 0, math.radians(150))
    try:
        sc.render.engine = "BLENDER_EEVEE"
    except TypeError:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    try:
        sc.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    try:
        sc.eevee.taa_render_samples = 32
    except Exception:
        pass
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = c["lens"]
    cam.data.clip_start = 0.05
    cam.data.clip_end = 600
    cam.location = g2b(c["eye"])
    d = g2b(c["target"]) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = OUT + "signage_%s%s.png" % (view, ("_" + TAG) if TAG else "")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", sc.render.filepath, flush=True)


for v in VIEWS:
    render(v)

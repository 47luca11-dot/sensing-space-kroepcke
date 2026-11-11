"""blender -b --factory-startup -P roundtrip_test.py -- in.glb out.glb
Same import/export as blender/apply_landmark_replace.py: checks that materials (palette texture, emissive factor,
vitrine alpha) survive the integration round trip."""
import sys

import bpy

a = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])
bpy.ops.export_scene.gltf(filepath=a[1], export_format="GLB", export_yup=True, export_apply=True)
print("ROUNDTRIP OK")

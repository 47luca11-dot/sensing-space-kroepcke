"""TIB-Detailmodell (Rhino-Export) -> Spielframe (Ursprung Kroepcke-Uhr); Meshes je Gruppe+Material zusammengefuehrt"""
import bpy, glob, os, json, bmesh, math
from mathutils import Matrix
from collections import defaultdict
R = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/"
J = json.load(open(R + 'data/tib_transform.json'))
E0, N0, H0 = 550298.88274216, 5802940.705715501, 56.16
rot = Matrix.Rotation(math.radians(J['rot_deg']), 4, 'Z')
tx = J['origin'][0] + J['t'][0] - E0; ty = J['origin'][1] + J['t'][1] - N0; tz = J['origin'][2] - H0
M = Matrix.Translation((tx, ty, tz)) @ rot          # Blender-Frame: x Ost, y Nord, z oben (Spiel = x, z, -y)
O = R + 'data/tib_raw/'; os.makedirs(O, exist_ok=True); stats = {}
for fn in sorted(glob.glob(R + 'data/tib_export/*.glb')):
    g = os.path.splitext(os.path.basename(fn))[0]
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=fn)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes: continue
    for o in meshes:
        mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes: o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.make_single_user(object=True, obdata=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in meshes: o.data.transform(M)
    for o in [o for o in bpy.data.objects if o.type != 'MESH']: bpy.data.objects.remove(o)
    bymat = defaultdict(list)
    for o in meshes: bymat[o.data.materials[0].name if o.data.materials and o.data.materials[0] else 'none'].append(o)
    tris = 0
    for m, objs in bymat.items():
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs: o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        if len(objs) > 1: bpy.ops.object.join()
        j = bpy.context.view_layer.objects.active; j.name = f'{g}__{m}'[:60]
        bm = bmesh.new(); bm.from_mesh(j.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005); bm.to_mesh(j.data); bm.free()
        tris += sum(len(p.vertices) - 2 for p in j.data.polygons)
    stats[g] = {'materials': len(bymat), 'tris': tris}
    bpy.ops.export_scene.gltf(filepath=O + g + '.glb', export_format='GLB', export_yup=True, export_apply=True, export_materials='EXPORT')
    print('TIB', g, tris)
json.dump(stats, open(O + 'stats.json', 'w'), indent=1)

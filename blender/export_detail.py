"""Rhino-Detailmodell (Opernplatz) -> Spielframe (Ursprung Kroepcke-Uhr), Meshes je Gruppe+Material zusammengefuehrt,
Export nach web/assets/city/detail/<gruppe>.glb (raw, danach gltf-transform/meshopt)."""
import bpy, glob, os, json, bmesh
from collections import defaultdict
D = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/rhino_export/"
O = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/detail_raw/"
os.makedirs(O, exist_ok=True)
OFF = (84.117, -141.706, -0.16)            # Blender-Frame (= Rhino XYZ) -> Spiel
stats = {}
for fn in sorted(glob.glob(D + '*.glb')):
    g = os.path.splitext(os.path.basename(fn))[0]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=fn)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    # Transformationen anwenden (inkl. Eltern), Versatz in die Vertices
    for o in meshes:
        mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes: o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.make_single_user(object=True, obdata=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in meshes:
        o.data.transform(__import__('mathutils').Matrix.Translation(OFF))
    for o in [o for o in bpy.data.objects if o.type != 'MESH']: bpy.data.objects.remove(o)
    # nach Material gruppieren und joinen
    bymat = defaultdict(list)
    for o in meshes:
        m = o.data.materials[0].name if o.data.materials and o.data.materials[0] else 'none'
        if len(o.data.materials) > 1: m = 'multi_' + m
        bymat[m].append(o)
    out = []
    for m, objs in bymat.items():
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs: o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        if len(objs) > 1: bpy.ops.object.join()
        j = bpy.context.view_layer.objects.active; j.name = f'{g}__{m}'[:60]
        # doppelte Vertices verschweissen (Rhino exportiert pro Flaeche) -> weniger Daten, keine Risse
        bm = bmesh.new(); bm.from_mesh(j.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005); bm.to_mesh(j.data); bm.free()
        out.append(j)
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in out)
    stats[g] = {'objects': len(out), 'tris': tris, 'materials': sorted(bymat)}
    bpy.ops.export_scene.gltf(filepath=O + g + '.glb', export_format='GLB', export_yup=True, export_apply=True, export_materials='EXPORT')
    print('EXPORTED', g, len(out), tris)
json.dump(stats, open(O + 'stats.json', 'w'), indent=1)

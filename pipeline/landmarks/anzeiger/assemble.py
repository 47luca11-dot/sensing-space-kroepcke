"""Blender (headless): build objects + PBR materials from build/anzeiger_parts.npz, export raw GLB (game frame via +Y up) and save .blend.
blender -b --factory-startup -P assemble.py -- <out.glb> <out.blend>"""
import bpy, bmesh, sys, os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index('--') + 1:]
OUT_GLB, OUT_BLEND = argv[0], argv[1]
TEX = os.path.join(HERE, 'tex')

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
meta = json.load(open(os.path.join(HERE, 'build', 'anzeiger_meta.json')))
npz = np.load(os.path.join(HERE, 'build', 'anzeiger_parts.npz'))


def img(name, colorspace):
    p = os.path.join(TEX, name)
    im = bpy.data.images.load(p, check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def make_material(name, d):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.use_backface_culling = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Roughness'].default_value = d.get('rough', 0.8)
    bsdf.inputs['Metallic'].default_value = d.get('metal', 0.0)
    if 'tex' in d:
        t = d['tex']
        tc = nt.nodes.new('ShaderNodeTexImage'); tc.image = img(t + '_col.png', 'sRGB'); tc.location = (-600, 300)
        nt.links.new(tc.outputs['Color'], bsdf.inputs['Base Color'])
        if os.path.exists(os.path.join(TEX, t + '_rgh.png')):
            tr = nt.nodes.new('ShaderNodeTexImage'); tr.image = img(t + '_rgh.png', 'Non-Color'); tr.location = (-600, 0)
            sep = nt.nodes.new('ShaderNodeSeparateColor'); sep.location = (-300, 0)
            nt.links.new(tr.outputs['Color'], sep.inputs['Color'])
            nt.links.new(sep.outputs[1], bsdf.inputs['Roughness'])
            nt.links.new(sep.outputs[2], bsdf.inputs['Metallic'])
        if os.path.exists(os.path.join(TEX, t + '_nrm.png')):
            tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img(t + '_nrm.png', 'Non-Color'); tn.location = (-600, -300)
            nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (-300, -300)
            nm.inputs['Strength'].default_value = 1.0
            nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
            nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    else:
        c = d.get('color', [0.8, 0.8, 0.8])
        bsdf.inputs['Base Color'].default_value = (c[0], c[1], c[2], 1.0)
    if 'emit' in d:
        e = d['emit']
        bsdf.inputs['Emission Color'].default_value = (e[0], e[1], e[2], 1.0)
        bsdf.inputs['Emission Strength'].default_value = d.get('emit_strength', 1.0)
    return m


mats = {k: make_material(k, v) for k, v in meta['materials'].items()}

col = bpy.data.collections.new('anzeiger_landmark'); sc.collection.children.link(col)
total = 0
for oname, od in meta['objects'].items():
    V = npz['V_' + oname].astype(np.float64); U = npz['U_' + oname]; M = npz['M_' + oname]
    Vb = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)            # game (x, y up, z south) -> blender (x, y north, z up)
    # object origin at the (1 cm rounded) bbox centre: small local coordinates in the GLB (float32 precision,
    # exact normals for downstream checks); the node translation puts it back in the game frame
    org = np.round((Vb.min(0) + Vb.max(0)) / 2, 2)
    Vb = Vb - org
    n = len(Vb); nt = n // 3
    me = bpy.data.meshes.new(oname)
    me.vertices.add(n); me.vertices.foreach_set('co', Vb.ravel())
    me.loops.add(n); me.loops.foreach_set('vertex_index', np.arange(n, dtype=np.int32))
    me.polygons.add(nt)
    me.polygons.foreach_set('loop_start', np.arange(0, n, 3, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(nt, 3, np.int32))
    uv = me.uv_layers.new(name='UVMap'); uv.data.foreach_set('uv', U.astype(np.float32).ravel())
    me.polygons.foreach_set('material_index', M.astype(np.int32))
    for mn in od['materials']: me.materials.append(mats[mn])
    me.update(); me.validate(clean_customdata=False)
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-4)
    bm.to_mesh(me); bm.free()
    if oname in meta['smooth']:
        me.shade_smooth()
        try: me.set_sharp_from_angle(angle=math.radians(35))
        except Exception as e: print('sharp', e)
    else:
        me.shade_flat()
    ob = bpy.data.objects.new(oname, me); col.objects.link(ob)
    ob.location = (float(org[0]), float(org[1]), float(org[2]))
    total += len(me.polygons)
    print('ASSEMBLE', oname, len(me.polygons), 'tris', len(me.vertices), 'verts')
print('ASSEMBLE total tris', total)

tmp = OUT_GLB + '.tmp.glb'
bpy.ops.export_scene.gltf(filepath=tmp, export_format='GLB', use_selection=False, export_yup=True, export_apply=False,
                          export_texcoords=True, export_normals=True, export_materials='EXPORT', export_image_format='AUTO',
                          export_cameras=False, export_lights=False, export_extras=False)
os.replace(tmp, OUT_GLB)
print('ASSEMBLE wrote', OUT_GLB, os.path.getsize(OUT_GLB))
os.makedirs(os.path.dirname(OUT_BLEND), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True)
print('ASSEMBLE saved', OUT_BLEND)

"""Kroepcke-Center -- Stufe 2: Blender-Szene aus kc_geo.npz/json aufbauen, Materialien (PBR), Instanzen,
Schriftzuege, Kollisionshuelle; Export GLB (Spielframe, Y oben) + .blend.
Aufruf: blender -b --factory-startup -P kc_blender.py -- <geo_dir> <tex_dir> <out.glb> <out.blend>
"""
import bpy, bmesh, sys, os, json, math
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index('--') + 1:]
GEO, TEX, OUT_GLB, OUT_BLEND = argv[:4]
data = np.load(os.path.join(GEO, 'kc_geo.npz'))
meta = json.load(open(os.path.join(GEO, 'kc_geo.json')))

bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
ROOT = bpy.data.objects.new('Kroepcke_Center', None); SC.collection.objects.link(ROOT)


def srgb(c):
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


IMG = {}
def image(name, noncolor=False):
    if name not in IMG:
        im = bpy.data.images.load(os.path.join(TEX, name))
        if noncolor: im.colorspace_settings.name = 'Non-Color'
        IMG[name] = im
    return IMG[name]


MAT = {}
def material(name):
    if name in MAT: return MAT[name]
    d = meta['materials'].get(name, dict(color=[0.5, 0.5, 0.5]))
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    m.use_backface_culling = True          # einseitig (glTF doubleSided=false) -- Normalen muessen stimmen
    b.inputs['Roughness'].default_value = d.get('rough', 0.5)
    b.inputs['Metallic'].default_value = d.get('metal', 0.0)
    if 'tex' in d:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = image(d['tex']); t.location = (-500, 300)
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    else:
        b.inputs['Base Color'].default_value = (*srgb(d['color']), 1)
    if 'normal' in d:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = image(d['normal'], True); t.location = (-700, -200)
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (-300, -200); nm.inputs['Strength'].default_value = 0.8
        nt.links.new(t.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    if d.get('alpha', 1.0) < 1.0:
        b.inputs['Alpha'].default_value = d['alpha']
        try: m.surface_render_method = 'BLENDED'
        except Exception: pass
        try: m.blend_method = 'BLEND'
        except Exception: pass
        m.use_backface_culling = False
        b.inputs['Specular IOR Level'].default_value = 0.5
    if 'emissive' in d:
        b.inputs['Emission Color'].default_value = (*srgb(d['emissive']), 1)
        b.inputs['Emission Strength'].default_value = d.get('emissive_strength', 1.0)
    MAT[name] = m
    return m


def groups():
    g = {}
    for k in data.files:
        pre, mat, kind = k.split('|')
        g.setdefault(pre, {}).setdefault(mat, {})[kind] = data[k]
    return g


G = groups()


def build_mesh(name, parts, merge=True, smooth_angle=30.0):
    """parts: {mat: {V,F,UV}} -> ein Mesh mit Materialslots."""
    Vs, Fs, UVs, Ms = [], [], [], []
    off = 0; mats = []
    for i, (mat, a) in enumerate(sorted(parts.items())):
        V, F, UV = a['V'], a['F'], a['UV']
        Vs.append(V); Fs.append(F + off); UVs.append(UV.reshape(-1, 2)); Ms.append(np.full(len(F), i)); off += len(V); mats.append(mat)
    V = np.concatenate(Vs); F = np.concatenate(Fs); UV = np.concatenate(UVs); MI = np.concatenate(Ms)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V)); me.vertices.foreach_set('co', V.ravel().astype(np.float32))
    me.loops.add(len(F) * 3); me.loops.foreach_set('vertex_index', F.ravel().astype(np.int32))
    me.polygons.add(len(F))
    me.polygons.foreach_set('loop_start', (np.arange(len(F)) * 3).astype(np.int32))
    me.polygons.foreach_set('loop_total', np.full(len(F), 3, np.int32))
    me.polygons.foreach_set('material_index', MI.astype(np.int32))
    uvl = me.uv_layers.new(name='UVMap'); uvl.data.foreach_set('uv', UV.ravel().astype(np.float32))
    me.update(calc_edges=True); me.validate(clean_customdata=False)
    for mat in mats:
        if mat == 'col': continue
        me.materials.append(material(mat))
    if merge:
        bm = bmesh.new(); bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0004)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_area() < 1e-9], context='FACES')
        bm.to_mesh(me); bm.free()
    if smooth_angle:
        try:
            me.shade_smooth(); me.set_sharp_from_angle(angle=math.radians(smooth_angle))
        except Exception as e:
            print('smooth', e)
    return me


# ---------------- Unikate: je Material ein Objekt (Kollision der Fassade ueber diese Netze + Huelle)
tris = 0
for mat, a in sorted(G['U'].items()):
    me = build_mesh('KC_' + mat.replace('KC ', '').replace(' ', '_'), {mat: a})
    ob = bpy.data.objects.new(me.name, me); SC.collection.objects.link(ob); ob.parent = ROOT
    tris += len(a['F'])
print('unique objects', len(G['U']), 'tris', tris)

# ---------------- Module (instanziert)
MODS = {}
for key, parts in G.items():
    if not key.startswith('M:'): continue
    name = key[2:]
    MODS[name] = build_mesh('KCM_' + name, parts)
    print('module', name, sum(len(p['F']) for p in parts.values()))
inst_root = bpy.data.objects.new('KC_Bays', None); SC.collection.objects.link(inst_root); inst_root.parent = ROOT
for i, it in enumerate(meta['instances']):
    ob = bpy.data.objects.new('KCB_%s_%03d' % (it['module'], i), MODS[it['module']])
    SC.collection.objects.link(ob); ob.parent = inst_root
    ob.matrix_world = Matrix(it['matrix'])
    tris += len(MODS[it['module']].polygons)

# ---------------- Kollisionshuelle (unsichtbar im Spiel: Name *_collision)
me = build_mesh('KC_collision', G['COL'], smooth_angle=None)
col = bpy.data.objects.new('KC_collision', me); SC.collection.objects.link(col); col.parent = ROOT
col.hide_render = True; col.display_type = 'WIRE'

# ---------------- Schriftzuege (Geometrie)
white = material('KC Sign White')
txt_objs = []
for i, t in enumerate(meta['texts']):
    cu = bpy.data.curves.new('KCT_%02d' % i, 'FONT'); cu.body = t['text']
    cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    cu.size = t['size']; cu.extrude = t['extrude']; cu.resolution_u = 3
    try: cu.font = bpy.data.fonts.load(r'C:\Windows\Fonts\arialbd.ttf', check_existing=True)
    except Exception as e: print('font', e)
    ob = bpy.data.objects.new('KCT_%02d' % i, cu); SC.collection.objects.link(ob)
    bpy.context.view_layer.update()
    # Groesse an Vorgabe (fit) anpassen
    if 'fit' in t:
        dg = bpy.context.evaluated_depsgraph_get(); ev = ob.evaluated_get(dg)
        bb = [Vector(c) for c in ev.bound_box]; w = max(v.x for v in bb) - min(v.x for v in bb)
        if w > 0 and (not t.get('fit_max') or w > t['fit']): cu.size *= t['fit'] / w
    n = Vector(t['normal']).normalized(); u = Vector(t['dir']).normalized(); v = n.cross(u).normalized()
    R = Matrix((u, v, n)).transposed().to_4x4()
    ob.matrix_world = Matrix.Translation(Vector(t['pos'])) @ R @ Matrix.Translation((0, 0, -t['extrude']))
    txt_objs.append(ob)
bpy.context.view_layer.update()
for ob in txt_objs:
    bpy.ops.object.select_all(action='DESELECT'); ob.select_set(True); bpy.context.view_layer.objects.active = ob
    bpy.ops.object.convert(target='MESH')
if txt_objs:
    bpy.ops.object.select_all(action='DESELECT')
    for ob in txt_objs: ob.select_set(True)
    bpy.context.view_layer.objects.active = txt_objs[0]
    bpy.ops.object.join()
    tob = bpy.context.view_layer.objects.active; tob.name = 'KC_Sign_Lettering'
    tob.data.materials.clear(); tob.data.materials.append(white)
    mw = tob.matrix_world.copy(); tob.parent = ROOT; tob.matrix_world = mw
    tris += sum(len(p.vertices) - 2 for p in tob.data.polygons)
print('TOTAL TRIS (gerendert, inkl. Instanzen)', tris)

# ---------------- Export
kw = dict(filepath=OUT_GLB, export_format='GLB', export_yup=True, export_apply=True, export_extras=False,
          export_image_format='AUTO')
try:
    bpy.ops.export_scene.gltf(**kw, export_materials='EXPORT')
except TypeError as e:
    print('export kw', e); bpy.ops.export_scene.gltf(filepath=OUT_GLB, export_format='GLB', export_yup=True)
print('wrote', OUT_GLB, os.path.getsize(OUT_GLB))
if OUT_BLEND:
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True)
json.dump(dict(total_tris=tris), open(os.path.join(GEO, 'kc_blender_stats.json'), 'w'))

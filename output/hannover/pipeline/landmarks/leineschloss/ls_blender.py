"""Leineschloss -- Stufe 2 (Blender 5, headless): Netze aus build/ls_geo.npz, PBR-Materialien (prozedurale Texturen),
Schweissen, Kantenglaettung nach Winkel, Instanzen als verknuepfte Duplikate, Inschriften (Textobjekte -> Netz),
Export GLB (Spielframe, Y oben) + .blend.
Aufruf: blender -b --factory-startup -P ls_blender.py -- <build_dir> <out.glb> <out.blend>"""
import bpy, bmesh, sys, os, json, math
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index('--') + 1:]
BUILD, OUT_GLB, OUT_BLEND = argv[:3]
TEX = os.path.join(BUILD, 'tex')
meta = json.load(open(os.path.join(BUILD, 'ls_geo.json')))
data = np.load(os.path.join(BUILD, 'ls_geo.npz'))
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
ROOT = bpy.data.objects.new('Leineschloss', None); SC.collection.objects.link(ROOT)

# name: (Farbtextur, Normalmap, Staerke, Rauheit, Metall, Grundfarbe ohne Textur)
MATS = {
    'LS_Sandstein': ('sandstein.jpg', 'sandstein_n.png', 0.5, 0.82, 0.0, None),
    'LS_Sandstein_Quader': ('sandstein_quader.jpg', 'sandstein_quader_n.png', 0.8, 0.85, 0.0, None),
    'LS_Rustika': ('rustika.jpg', 'rustika_n.png', 1.0, 0.86, 0.0, None),
    'LS_Putz': ('putz.jpg', 'putz_n.png', 0.4, 0.9, 0.0, None),
    'LS_Schiefer': ('schiefer.jpg', 'schiefer_n.png', 0.9, 0.6, 0.0, None),
    'LS_Zink': ('zink.jpg', 'zink_n.png', 0.3, 0.45, 0.6, None),
    'LS_Kupfer': ('kupfer.jpg', 'kupfer_n.png', 0.5, 0.55, 0.5, None),
    'LS_Marmor': ('marmor.jpg', 'marmor_n.png', 0.6, 0.35, 0.0, None),
    'LS_Granit': ('granit.jpg', 'granit_n.png', 0.5, 0.6, 0.0, None),
    'LS_Quai': ('quai.jpg', 'quai_n.png', 0.9, 0.9, 0.0, None),
    'LS_Flachdach': ('flachdach.jpg', 'flachdach_n.png', 0.5, 0.95, 0.0, None),
    'LS_Holz': ('holz.jpg', 'holz_n.png', 0.4, 0.6, 0.0, None),
    'LS_Putz_weiss': ('putz_weiss.jpg', 'putz_weiss_n.png', 0.3, 0.9, 0.0, None),
    'LS_Platten': ('platten.jpg', 'platten_n.png', 0.5, 0.75, 0.0, None),
    'LS_Rahmen': (None, None, 0, 0.45, 0.0, (0.86, 0.85, 0.81)),
    'LS_Rahmen_dunkel': (None, None, 0, 0.4, 0.7, (0.12, 0.115, 0.11)),
    'LS_Fensterglas': (None, None, 0, 0.06, 0.0, (0.035, 0.045, 0.055)),
    'LS_Eisen': (None, None, 0, 0.55, 0.6, (0.04, 0.04, 0.04)),
    'LS_Edelstahl': (None, None, 0, 0.25, 1.0, (0.62, 0.63, 0.64)),
    'LS_Bronze': (None, None, 0, 0.4, 1.0, (0.42, 0.3, 0.16)),
    'LS_Dunkel': (None, None, 0, 0.5, 0.3, (0.03, 0.03, 0.03)),
}
IMG = {}


def image(fn, noncolor=False):
    if fn not in IMG:
        im = bpy.data.images.load(os.path.join(TEX, fn))
        if noncolor: im.colorspace_settings.name = 'Non-Color'
        IMG[fn] = im
    return IMG[fn]


MAT = {}


def material(name):
    if name in MAT: return MAT[name]
    col, nor, ns, rough, metal, base = MATS[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    m.use_backface_culling = True
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if col:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = image(col); t.location = (-500, 300)
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    else:
        b.inputs['Base Color'].default_value = (*base, 1.0)
    if nor:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = image(nor, True); t.location = (-700, -200)
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (-300, -200); nm.inputs['Strength'].default_value = ns
        nt.links.new(t.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    MAT[name] = m
    return m


SUFFIX = {'main': '', 'det': '_nocol', 'fine': '_nocol_noshadow'}


def make_mesh(name, P, UV, mat, weld=0.0004, smooth_deg=34):
    nt_ = len(P) // 3
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(P)); me.vertices.foreach_set('co', P.astype(np.float64).ravel())
    me.loops.add(len(P)); me.loops.foreach_set('vertex_index', np.arange(len(P), dtype=np.int32))
    me.polygons.add(nt_)
    me.polygons.foreach_set('loop_start', np.arange(0, len(P), 3, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(nt_, 3, dtype=np.int32))
    uvl = me.uv_layers.new(name='UVMap')
    uvl.data.foreach_set('uv', UV.astype(np.float64).ravel())
    me.update(calc_edges=True); me.validate(clean_customdata=False)
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
    bm.to_mesh(me); bm.free()
    me.materials.append(material(mat))
    try:
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(smooth_deg))
    except Exception as e:
        print('smooth fallback', e)
        for p in me.polygons: p.use_smooth = False
    return me


tris_total = 0
for g in meta['groups']:
    P = data['p%d' % g['i']]; UV = data['uv%d' % g['i']]
    me = make_mesh('LS_%s_%s' % (g['grp'], g['mat']), P, UV, g['mat'], smooth_deg=34 if g['grp'] == 'main' else 40)
    ob = bpy.data.objects.new('LS_%s_%s%s' % (g['grp'], g['mat'][3:], SUFFIX[g['grp']]), me)
    SC.collection.objects.link(ob); ob.parent = ROOT
    tris_total += len(me.polygons)
# Instanzen: ein Netz je (Teil, Material), mehrere Objekte
for e in meta['instances']:
    meshes = []
    for p in e['parts']:
        P = data['ip%d_%d' % (e['j'], p['k'])]; UV = data['iuv%d_%d' % (e['j'], p['k'])]
        meshes.append((make_mesh('LSI_%s_%s' % (e['name'], p['mat']), P, UV, p['mat'], smooth_deg=50), p))
    for n, Mm in enumerate(e['M']):
        for me, p in meshes:
            ob = bpy.data.objects.new('LSI_%s_%02d_%s%s' % (e['name'], n, p['mat'][3:], SUFFIX.get(p['grp'], '_nocol') if 'Schaft' not in e['name'] and 'Basis' not in e['name'] else ''), me)
            ob.matrix_world = Matrix(Mm)
            SC.collection.objects.link(ob); ob.parent = ROOT
            tris_total += len(me.polygons)
# Inschriften
for k, t in enumerate(meta['texts']):
    cuv = bpy.data.curves.new('Inschrift', 'FONT'); cuv.body = t['text']; cuv.size = t['size']; cuv.extrude = t['extrude']
    cuv.align_x = 'CENTER'; cuv.align_y = 'CENTER'; cuv.resolution_u = 2
    tob = bpy.data.objects.new('tmp_text', cuv); SC.collection.objects.link(tob)
    x = Vector(t['x']).normalized(); z = Vector(t['z']).normalized(); y = z.cross(x)
    tob.matrix_world = Matrix.Translation(Vector(t['pos'])) @ Matrix((x, y, z)).transposed().to_4x4()
    bpy.context.view_layer.update(); dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tob.evaluated_get(dg))
    me.transform(tob.matrix_world); me.materials.clear(); me.materials.append(material(t['mat']))
    ob = bpy.data.objects.new('LS_fine_Inschrift_%d_nocol_noshadow' % k, me); SC.collection.objects.link(ob); ob.parent = ROOT
    bpy.data.objects.remove(tob)
    tris_total += sum(len(p.vertices) - 2 for p in me.polygons)
print('LS objects', len(meta['groups']), 'tris', tris_total)
tmp = OUT_GLB + '.tmp.glb'
kw = dict(filepath=tmp, export_format='GLB', export_yup=True, export_apply=True, export_extras=False,
          export_image_format='AUTO', export_materials='EXPORT', export_tangents=False)
try:
    bpy.ops.export_scene.gltf(**kw)
except TypeError as e:
    print('export kw', e); bpy.ops.export_scene.gltf(filepath=tmp, export_format='GLB', export_yup=True)
os.replace(tmp, OUT_GLB)
print('LS wrote', OUT_GLB, os.path.getsize(OUT_GLB))
if OUT_BLEND:
    os.makedirs(os.path.dirname(OUT_BLEND), exist_ok=True)
    for im in bpy.data.images:
        try: im.pack()
        except Exception as ex: print('pack', ex)
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True)
    print('LS saved', OUT_BLEND)
json.dump(dict(total_tris=tris_total), open(os.path.join(BUILD, 'blender_stats.json'), 'w'))

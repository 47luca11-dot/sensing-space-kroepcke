"""Marktkirche -- Stufe 2 (Blender 5, headless): Netze aus build/mk_geo.npz, PBR-Materialien mit prozeduralen Texturen,
Schweissen, Kantenglaettung nach Winkel, Export GLB (Spielframe, Y oben) + .blend.
Aufruf: blender -b --factory-startup -P mk_blender.py -- <build_dir> <out.glb> <out.blend>"""
import bpy, bmesh, sys, os, json, math
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:]
BUILD, OUT_GLB, OUT_BLEND = argv[:3]
TEX = os.path.join(BUILD, 'tex')
meta = json.load(open(os.path.join(BUILD, 'mk_geo.json')))
data = np.load(os.path.join(BUILD, 'mk_geo.npz'))
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
ROOT = bpy.data.objects.new('Marktkirche', None); SC.collection.objects.link(ROOT)

# name: (Farbtextur, Normalmap, Staerke, Rauheit, Metall, Grundfarbe falls ohne Textur)
MATS = {
    'Backstein': ('backstein.jpg', 'backstein_n.png', 0.8, 0.86, 0.0, None),
    'Glasurstein': ('glasurstein.jpg', 'glasurstein_n.png', 0.8, 0.42, 0.0, None),
    'Sandstein_Quader': ('sandstein_quader.jpg', 'sandstein_quader_n.png', 0.7, 0.9, 0.0, None),
    'Sandstein': ('sandstein.jpg', 'sandstein_n.png', 0.6, 0.85, 0.0, None),
    'Sandstein_dunkel': ('sandstein_dunkel.jpg', 'sandstein_dunkel_n.png', 0.7, 0.9, 0.0, None),
    'Kupfer': ('kupfer.jpg', None, 0, 0.6, 0.2, None),
    'Kupfer_Patina': ('kupfer_patina.jpg', None, 0, 0.7, 0.1, None),
    'Dachziegel': ('dachziegel.jpg', 'dachziegel_n.png', 0.9, 0.78, 0.0, None),
    'Bleiglas': ('bleiglas.jpg', 'bleiglas_n.png', 0.5, 0.12, 0.0, None),
    'Stahl': ('stahl.jpg', None, 0, 0.5, 0.55, None),
    'Lamellen': ('lamellen.jpg', None, 0, 0.85, 0.0, None),
    'Gold': (None, None, 0, 0.28, 1.0, (0.86, 0.66, 0.28)),
    'Bronze': ('bronze.jpg', None, 0, 0.62, 0.12, None),          # Patina ist mineralisch: kaum metallisch (sonst tuerkis gefaerbte Spiegelung)
    'Granit': ('granit.jpg', 'granit_n.png', 0.4, 0.62, 0.0, None),
    'Sockelstein': ('sockelstein.jpg', 'sockelstein_n.png', 0.5, 0.8, 0.0, None),
    'Zifferblatt': ('zifferblatt.png', None, 0, 0.45, 0.0, None),
    'Eisen': (None, None, 0, 0.55, 0.6, (0.05, 0.05, 0.05)),
    'Dunkel': (None, None, 0, 1.0, 0.0, (0.012, 0.011, 0.01)),
    'Fliesen': ('fliesen.jpg', 'fliesen_n.png', 0.6, 0.42, 0.0, None),
    'Tuerbronze': ('tuerbronze.jpg', 'tuerbronze_n.png', 0.5, 0.45, 0.6, None),
    'Schaukasten': ('schaukasten.png', None, 0, 0.25, 0.0, None),
}
IMG = {}


def image(fn, noncolor=False):
    if fn not in IMG:
        im = bpy.data.images.load(os.path.join(TEX, fn))
        if noncolor: im.colorspace_settings.name = 'Non-Color'
        IMG[fn] = im
    return IMG[fn]


MAT = {}


AO_MATS = ('Bronze',)


def material(name):
    if name in MAT: return MAT[name]
    col, nor, ns, rough, metal, base = MATS[name]
    m = bpy.data.materials.new('MK_' + name); m.use_nodes = True
    m.use_backface_culling = True
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if col:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = image(col); t.location = (-500, 300)
        if name in AO_MATS:     # gebackene Verdeckung (Vertexfarbe 'Col') x Textur -> glTF COLOR_0 (three.js: vertexColors)
            vc = nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'; vc.location = (-500, 0)
            mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'; mx.location = (-250, 200)
            mx.inputs['Factor'].default_value = 1.0
            nt.links.new(t.outputs['Color'], mx.inputs[6]); nt.links.new(vc.outputs['Color'], mx.inputs[7])
            nt.links.new(mx.outputs[2], b.inputs['Base Color'])
        else:
            nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    else:
        b.inputs['Base Color'].default_value = (*base, 1.0)
    if nor:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = image(nor, True); t.location = (-700, -200)
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (-300, -200); nm.inputs['Strength'].default_value = ns
        nt.links.new(t.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    MAT[name] = m
    return m


GRPNAME = {'wall': '', 'roof': '_nocol', 'det': '_nocol', 'fine': '_nocol_noshadow'}
tris_total = 0
for g in meta['groups']:
    P = data['p%d' % g['i']].astype(np.float64); UV = data['uv%d' % g['i']].astype(np.float64)
    nt_ = len(P) // 3
    me = bpy.data.meshes.new('MK_%s_%s' % (g['grp'], g['mat']))
    me.vertices.add(len(P)); me.vertices.foreach_set('co', P.ravel())
    me.loops.add(len(P)); me.loops.foreach_set('vertex_index', np.arange(len(P), dtype=np.int32))
    me.polygons.add(nt_)
    me.polygons.foreach_set('loop_start', np.arange(0, len(P), 3, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(nt_, 3, dtype=np.int32))
    uvl = me.uv_layers.new(name='UVMap')
    uvl.data.foreach_set('uv', UV.ravel())
    if ('ao%d' % g['i']) in data.files:
        ao = data['ao%d' % g['i']].astype(np.float64)
        ca = me.color_attributes.new(name='Col', type='FLOAT_COLOR', domain='CORNER')
        ca.data.foreach_set('color', np.c_[ao, ao, ao, np.ones_like(ao)].ravel())
    me.update(calc_edges=True); me.validate(clean_customdata=False)
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0004)
    deg = [f for f in bm.faces if f.calc_area() < 1e-7]                 # durch das Schweissen kollabierte Dreiecke
    if deg: bmesh.ops.delete(bm, geom=deg, context='FACES_ONLY')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose: bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(me); bm.free()
    me.materials.append(material(g['mat']))
    try:
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(32 if g['mat'] not in ('Bronze', 'Gold') else 50))
    except Exception as e:
        print('smooth fallback', e)
        for p in me.polygons: p.use_smooth = False
    ob = bpy.data.objects.new('MK_%s_%s%s' % (g['grp'], g['mat'], GRPNAME[g['grp']]), me)
    me.name = ob.name
    SC.collection.objects.link(ob); ob.parent = ROOT
    tris_total += len(me.polygons)
# Inschrift am Lutherdenkmal (Bronzebuchstaben, Blender-Standardschrift)
L = meta.get('luther')
if L:
    from mathutils import Matrix, Vector
    O = meta['origin']; U = meta['U']; V = meta['V']
    def bl(u, v, h):
        return Vector((O[0] + u * U[0] + v * V[0], -(O[1] + u * U[1] + v * V[1]), h))
    cu, cv, gl = L['center']; fu, fv = L['face']
    p0 = bl(cu, cv, 0.0); p1 = bl(cu + fu, cv + fv, 0.0); n = (p1 - p0).normalized()
    xax = (-n).cross(Vector((0, 0, 1))).normalized(); yax = Vector((0, 0, 1))
    dg = bpy.context.evaluated_depsgraph_get()
    ins = L.get('ins', [1.3, 1.08]); fr = L.get('front', 0.705)
    for body, size, hh in (('MARTIN LUTHER', 0.13, ins[0]), ('1483 - 1546', 0.11, ins[1])):
        cuv = bpy.data.curves.new('Inschrift', 'FONT'); cuv.body = body; cuv.size = size; cuv.extrude = 0.008
        cuv.align_x = 'CENTER'; cuv.align_y = 'CENTER'; cuv.resolution_u = 3
        tob = bpy.data.objects.new('tmp_text', cuv); SC.collection.objects.link(tob)
        pos = bl(cu + fu * fr, cv + fv * fr, gl + hh)
        tob.matrix_world = Matrix.Translation(pos) @ Matrix((xax, yax, n)).transposed().to_4x4()
        bpy.context.view_layer.update(); dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(tob.evaluated_get(dg))
        me.materials.clear(); me.materials.append(material('Dunkel'))   # eingehauene Inschrift (dunkel)
        bm = bmesh.new(); bm.from_mesh(me)                                  # Bereinigung im Lokalframe (float32 genau)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.ops.dissolve_degenerate(bm, dist=2e-4, edges=bm.edges[:])
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        def _sliver(f):
            l = max(e.calc_length() for e in f.edges)
            return f.calc_area() < max(1e-7, 2e-3 * l * l)
        deg = [f for f in bm.faces if _sliver(f)]                           # Schrift-Tessellierung: Nullflaechen/Splitter entfernen
        if deg: bmesh.ops.delete(bm, geom=deg, context='FACES_ONLY')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose: bmesh.ops.delete(bm, geom=loose, context='VERTS')
        bm.to_mesh(me); bm.free()
        me.transform(tob.matrix_world)
        oname = 'MK_fine_Inschrift_%s_nocol_noshadow' % body[:4]
        me.name = oname                                                     # Netzname = Knotenname (QA)
        ob = bpy.data.objects.new(oname, me); SC.collection.objects.link(ob); ob.parent = ROOT
        bpy.data.objects.remove(tob)
        tris_total += sum(len(p.vertices) - 2 for p in me.polygons)
print('MK objects', len(meta['groups']), 'tris', tris_total)
kw = dict(filepath=OUT_GLB, export_format='GLB', export_yup=True, export_apply=True, export_extras=False,
          export_image_format='AUTO', export_materials='EXPORT', export_tangents=False)
tmp = OUT_GLB + '.tmp.glb'
kw['filepath'] = tmp
try:
    bpy.ops.export_scene.gltf(**kw)
except TypeError as e:
    print('export kw', e); bpy.ops.export_scene.gltf(filepath=tmp, export_format='GLB', export_yup=True)
os.replace(tmp, OUT_GLB)
print('MK wrote', OUT_GLB, os.path.getsize(OUT_GLB))
if OUT_BLEND:
    os.makedirs(os.path.dirname(OUT_BLEND), exist_ok=True)
    for im in bpy.data.images:
        try: im.pack()
        except Exception as e: print('pack', e)
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True)
    print('MK saved', OUT_BLEND)
json.dump(dict(total_tris=tris_total), open(os.path.join(BUILD, 'blender_stats.json'), 'w'))

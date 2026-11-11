"""Headless-Blender-Vorschau: laedt die unkomprimierten Gebaeude-Kacheln (eigener GLB-Leser inkl. KHR_mesh_quantization,
EXT_mesh_gpu_instancing mit _COLOR_0 -> Geometry-Nodes-Instanzen), das Rhino-Detailmodell (Opernplatz) und ein DGM1-Gelaende,
rendert mit Eevee (Sonne + Himmel) nach blender/previews/facades_*.png.

blender -b --factory-startup -P blender_preview.py -- <raw_dir> [view1,view2,...] [--fast]"""
import bpy, bmesh, json, os, sys, struct, math, time
import numpy as np
from mathutils import Vector, Quaternion, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(HERE, 'cache')
OUT = os.path.join(ROOT, 'blender', 'previews')
RHINO = os.path.join(ROOT, 'data', 'rhino_export')
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
RAW = argv[0] if argv else os.path.join(ROOT, 'data', 'buildings_raw')
VIEWS = argv[1].split(',') if len(argv) > 1 and not argv[1].startswith('--') else None
FAST = '--fast' in argv
NO_RHINO = '--no-rhino' in argv
CAMS = argv[argv.index('--cams') + 1] if '--cams' in argv else os.path.join(CACHE, 'preview_cams.json')
if '--views' in argv:
    VIEWS = argv[argv.index('--views') + 1].split(',')
# --landmarks a,b: handgebaute Wahrzeichen (data/landmarks_raw/<name>.glb, Spielkoordinaten) als Kontext mitladen
LANDMARKS = argv[argv.index('--landmarks') + 1].split(',') if '--landmarks' in argv else []
RES = (1280, 720)

CT = {5126: np.float32, 5122: np.int16, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32, 5120: np.int8}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def g2b(P):
    """glTF (x,y,z) -> Blender (x,-z,y)"""
    P = np.asarray(P, np.float64)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def read_glb(path):
    b = open(path, 'rb').read()
    jl = struct.unpack('<I', b[12:16])[0]
    J = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl = struct.unpack('<I', b[off:off + 4])[0]
    BIN = b[off + 8:off + 8 + bl]
    return J, BIN


def accessor(J, BIN, i):
    a = J['accessors'][i]; bv = J['bufferViews'][a['bufferView']]
    dt = CT[a['componentType']]; nc = NC[a['type']]; cnt = a['count']
    st = bv.get('byteStride', np.dtype(dt).itemsize * nc)
    base = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    raw = np.frombuffer(BIN, np.uint8, count=st * (cnt - 1) + np.dtype(dt).itemsize * nc, offset=base)
    if st == np.dtype(dt).itemsize * nc:
        arr = raw.view(dt).reshape(cnt, nc)
    else:
        pad = np.zeros(st * cnt, np.uint8); pad[:len(raw)] = raw
        arr = pad.reshape(cnt, st)[:, :np.dtype(dt).itemsize * nc].copy().view(dt).reshape(cnt, nc)
    arr = arr.astype(np.float64) if a.get('normalized') is not True else arr.astype(np.float64) / {np.int16: 32767.0, np.uint16: 65535.0,
                                                                                                    np.uint8: 255.0, np.int8: 127.0}[dt]
    if a.get('normalized') and dt in (np.int16, np.int8):
        arr = np.maximum(arr, -1.0)
    return arr


IMG_CACHE = {}
MAT_CACHE = {}


def make_material(md, base_dir, instanced):
    key = (md['name'], instanced)
    if key in MAT_CACHE:
        return MAT_CACHE[key]
    m = bpy.data.materials.new(md['name'] + ('_inst' if instanced else ''))
    m.use_nodes = True
    m.use_backface_culling = True                    # wie im Spiel (FrontSide): umgedrehte Flaechen waeren unsichtbar
    nt = m.node_tree; N = nt.nodes; L = nt.links
    bsdf = next(n for n in N if n.type == 'BSDF_PRINCIPLED')
    pbr = md.get('pbrMetallicRoughness', {})
    bc = pbr.get('baseColorFactor', [1, 1, 1, 1])
    bsdf.inputs['Metallic'].default_value = pbr.get('metallicFactor', 0.0)
    bsdf.inputs['Roughness'].default_value = pbr.get('roughnessFactor', 0.8)
    vc = N.new('ShaderNodeAttribute'); vc.attribute_name = 'Col'; vc.attribute_type = 'GEOMETRY'
    mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    L.new(vc.outputs['Color'], mul.inputs['A'])
    mul.inputs['B'].default_value = (bc[0], bc[1], bc[2], 1)
    cur = mul.outputs['Result']
    tex = pbr.get('baseColorTexture')
    if tex is not None:
        J = md['_J']
        src = J['textures'][tex['index']]['source']; uri = J['images'][src]['uri']
        p = os.path.join(base_dir, uri)
        if p not in IMG_CACHE:
            IMG_CACHE[p] = bpy.data.images.load(p)
        ti = N.new('ShaderNodeTexImage'); ti.image = IMG_CACHE[p]
        m2 = N.new('ShaderNodeMix'); m2.data_type = 'RGBA'; m2.blend_type = 'MULTIPLY'; m2.inputs['Factor'].default_value = 1.0
        L.new(cur, m2.inputs['A']); L.new(ti.outputs['Color'], m2.inputs['B']); cur = m2.outputs['Result']
    if instanced:
        ic = N.new('ShaderNodeAttribute'); ic.attribute_name = 'icol'; ic.attribute_type = 'INSTANCER'
        m3 = N.new('ShaderNodeMix'); m3.data_type = 'RGBA'; m3.blend_type = 'MULTIPLY'; m3.inputs['Factor'].default_value = 1.0
        L.new(cur, m3.inputs['A']); L.new(ic.outputs['Color'], m3.inputs['B']); cur = m3.outputs['Result']
    L.new(cur, bsdf.inputs['Base Color'])
    ef = md.get('emissiveFactor')
    if ef and max(ef) > 0:
        for nm in ('Emission Color', 'Emission'):
            if nm in bsdf.inputs:
                bsdf.inputs[nm].default_value = (ef[0], ef[1], ef[2], 1); break
        if 'Emission Strength' in bsdf.inputs:
            bsdf.inputs['Emission Strength'].default_value = 1.0
    if md.get('alphaMode') == 'BLEND':
        bsdf.inputs['Alpha'].default_value = bc[3]
        if 'glass' in md['name'].lower():
            # wie im Spiel (web/js/world.js: Glas -> opacity 0.35, wenig Spiegelung): Auslagen sichtbar
            bsdf.inputs['Alpha'].default_value = 0.35
            for nm in ('Specular IOR Level', 'Specular'):
                if nm in bsdf.inputs:
                    bsdf.inputs[nm].default_value = 0.25; break
        try:
            m.surface_render_method = 'BLENDED'
        except Exception:
            m.blend_method = 'BLEND'
    MAT_CACHE[key] = m
    return m


def build_mesh(J, BIN, mi, base_dir, name, xform=None, instanced=False):
    """glTF-Mesh -> Blender-Mesh (Vertexfarbe 'Col', UV, Materialslots). xform: (T,S) Knoten-Transformation (glTF-Raum)."""
    md = J['meshes'][mi]
    Ps, Fs, Cs, Us, Ms = [], [], [], [], []
    mats = []
    nv = 0
    for pr in md['primitives']:
        P = accessor(J, BIN, pr['attributes']['POSITION'])
        if xform is not None:
            P = P * xform[1] + xform[0]
        F = accessor(J, BIN, pr['indices']).astype(np.int64).reshape(-1, 3)
        C = accessor(J, BIN, pr['attributes']['COLOR_0'])[:, :3] if 'COLOR_0' in pr['attributes'] else np.ones((len(P), 3))
        U = accessor(J, BIN, pr['attributes']['TEXCOORD_0']) if 'TEXCOORD_0' in pr['attributes'] else None
        md2 = dict(J['materials'][pr['material']]); md2['_J'] = J
        mat = make_material(md2, base_dir, instanced)
        if mat not in mats:
            mats.append(mat)
        Ps.append(g2b(P)); Fs.append(F + nv); Cs.append(C)
        Us.append(U if U is not None else np.zeros((len(P), 2)))
        Ms.append(np.full(len(F), mats.index(mat)))
        nv += len(P)
    P = np.concatenate(Ps); F = np.concatenate(Fs); C = np.concatenate(Cs); U = np.concatenate(Us); M = np.concatenate(Ms)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(P)); me.vertices.foreach_set('co', P.astype(np.float32).ravel())
    me.loops.add(len(F) * 3); me.loops.foreach_set('vertex_index', F.ravel().astype(np.int32))
    me.polygons.add(len(F)); me.polygons.foreach_set('loop_start', (np.arange(len(F)) * 3).astype(np.int32))
    me.update(calc_edges=True)
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set('material_index', M.astype(np.int32))
    ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    rgba = np.ones((len(P), 4), np.float32); rgba[:, :3] = C
    ca.data.foreach_set('color', rgba.ravel())
    uvl = me.uv_layers.new(name='UVMap')
    uvl.data.foreach_set('uv', (U[F.ravel()] * [1, -1] + [0, 1]).astype(np.float32).ravel())
    me.update()
    return me


PROTO_GN = {}


def gn_group(proto_obj):
    if proto_obj.name in PROTO_GN:
        return PROTO_GN[proto_obj.name]
    ng = bpy.data.node_groups.new('inst_' + proto_obj.name, 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N = ng.nodes; L = ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    iop = N.new('GeometryNodeInstanceOnPoints')
    oi = N.new('GeometryNodeObjectInfo'); oi.inputs['Object'].default_value = proto_obj
    try:
        oi.transform_space = 'ORIGINAL'
    except Exception:
        pass
    rot = N.new('GeometryNodeInputNamedAttribute'); rot.data_type = 'QUATERNION'; rot.inputs['Name'].default_value = 'rot'
    scl = N.new('GeometryNodeInputNamedAttribute'); scl.data_type = 'FLOAT_VECTOR'; scl.inputs['Name'].default_value = 'scl'
    L.new(gi.outputs[0], iop.inputs['Points'])
    L.new(oi.outputs['Geometry'], iop.inputs['Instance'])
    L.new(rot.outputs['Attribute'], iop.inputs['Rotation'])
    L.new(scl.outputs['Attribute'], iop.inputs['Scale'])
    L.new(iop.outputs['Instances'], go.inputs[0])
    PROTO_GN[proto_obj.name] = ng
    return ng


def load_tile(path, coll, proto_coll, proto_objs):
    J, BIN = read_glb(path)
    base = os.path.dirname(path)
    nodes = J['nodes']
    for nd in nodes:
        if 'mesh' not in nd:
            continue
        ext = nd.get('extensions', {}).get('EXT_mesh_gpu_instancing')
        if ext is None:
            T = np.array(nd.get('translation', [0, 0, 0])); S = np.array(nd.get('scale', [1, 1, 1]))
            me = build_mesh(J, BIN, nd['mesh'], base, nd['name'], (T, S))
            ob = bpy.data.objects.new(nd['name'], me); coll.objects.link(ob)
        else:
            mname = J['meshes'][nd['mesh']]['name']
            if mname not in proto_objs:
                me = build_mesh(J, BIN, nd['mesh'], base, mname, None, instanced=True)
                po = bpy.data.objects.new(mname, me); proto_coll.objects.link(po)
                proto_objs[mname] = po
            po = proto_objs[mname]
            at = ext['attributes']
            T = g2b(accessor(J, BIN, at['TRANSLATION']))
            Q = accessor(J, BIN, at['ROTATION'])                   # x,y,z,w (glTF)
            S = accessor(J, BIN, at['SCALE'])
            C = accessor(J, BIN, at['_COLOR_0']) if '_COLOR_0' in at else np.ones((len(T), 3))
            qb = np.stack([Q[:, 3], Q[:, 0], -Q[:, 2], Q[:, 1]], 1)  # w, x, -z, y (Blender)
            sb = np.stack([S[:, 0], S[:, 2], S[:, 1]], 1)
            me = bpy.data.meshes.new(nd['name'] + '_pts')
            me.vertices.add(len(T)); me.vertices.foreach_set('co', T.astype(np.float32).ravel())
            a = me.attributes.new('rot', 'QUATERNION', 'POINT'); a.data.foreach_set('value', qb.astype(np.float32).ravel())
            a = me.attributes.new('scl', 'FLOAT_VECTOR', 'POINT'); a.data.foreach_set('vector', sb.astype(np.float32).ravel())
            a = me.attributes.new('icol', 'FLOAT_COLOR', 'POINT')
            rgba = np.ones((len(T), 4), np.float32); rgba[:, :3] = C
            a.data.foreach_set('color', rgba.ravel())
            ob = bpy.data.objects.new(nd['name'], me); coll.objects.link(ob)
            mod = ob.modifiers.new('inst', 'NODES'); mod.node_group = gn_group(po)


def terrain(coll):
    p = os.path.join(CACHE, 'terrain_2m.npz')
    if not os.path.exists(p):
        return
    d = np.load(p); xs, zs, Y = d['x'], d['z'], d['y']
    X, Z = np.meshgrid(xs, zs)
    P = g2b(np.stack([X.ravel(), Y.ravel() - 0.08, Z.ravel()], 1))
    nx, nz = len(xs), len(zs)
    i = np.arange(nx - 1); j = np.arange(nz - 1)
    I, Jj = np.meshgrid(i, j)
    a = (Jj * nx + I).ravel(); b = a + 1; c = a + nx + 1; dd = a + nx
    F = np.concatenate([np.stack([a, c, b], 1), np.stack([a, dd, c], 1)])
    me = bpy.data.meshes.new('terrain')
    me.vertices.add(len(P)); me.vertices.foreach_set('co', P.astype(np.float32).ravel())
    me.loops.add(len(F) * 3); me.loops.foreach_set('vertex_index', F.ravel().astype(np.int32))
    me.polygons.add(len(F)); me.polygons.foreach_set('loop_start', (np.arange(len(F)) * 3).astype(np.int32))
    me.update(calc_edges=True)
    m = bpy.data.materials.new('ground'); m.use_nodes = True
    bs = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bs.inputs['Base Color'].default_value = (0.16, 0.16, 0.155, 1); bs.inputs['Roughness'].default_value = 0.95
    me.materials.append(m)
    ob = bpy.data.objects.new('terrain', me); coll.objects.link(ob)


def rhino(coll):
    if NO_RHINO:
        return
    root = bpy.data.objects.new('Detail_Opernplatz', None); coll.objects.link(root)
    for g in ['bldg_walls', 'bldg_openings', 'bldg_roof', 'bldg_base', 'landmark_opernhaus', 'kiosk', 'street', 'ground', 'green',
              'green_trees', 'street_furniture', 'landmark_tiefgarage_pavillon', 'landmark_mahnmal', 'fix']:
        fn = os.path.join(RHINO, g + '.glb')
        if not os.path.exists(fn):
            continue
        before = set(bpy.data.objects); mats_before = set(bpy.data.materials)
        try:
            bpy.ops.import_scene.gltf(filepath=fn)
        except Exception as e:
            print('rhino import fail', g, e); continue
        for o in set(bpy.data.objects) - before:
            if o.parent is None:
                o.parent = root
            for c in o.users_collection:
                c.objects.unlink(o)
            coll.objects.link(o)
    root.location = (84.117, -141.706, -0.16)


def landmarks(coll):
    for nm in LANDMARKS:
        fn = os.path.join(ROOT, 'data', 'landmarks_raw', nm + '.glb')
        if not os.path.exists(fn):
            print('landmark fehlt', nm); continue
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.gltf(filepath=fn)
        except Exception as e:
            print('landmark import fail', nm, e); continue
        for o in set(bpy.data.objects) - before:
            for c in o.users_collection:
                c.objects.unlink(o)
            coll.objects.link(o)
            if o.type == 'MESH':
                for m in o.data.materials:
                    if m is not None:
                        m.use_backface_culling = True
        print('landmark', nm)


def setup_scene():
    sc = bpy.context.scene
    try:
        sc.render.engine = 'BLENDER_EEVEE'
    except TypeError:
        sc.render.engine = 'BLENDER_EEVEE_NEXT'
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.view_settings.view_transform = 'AgX' if 'AgX' in [i.identifier for i in bpy.types.ColorManagedViewSettings.bl_rna.properties['view_transform'].enum_items] else 'Filmic'
    try:
        sc.eevee.taa_render_samples = 16 if FAST else 48
        sc.eevee.use_shadows = True
        sc.eevee.use_raytracing = True
    except Exception:
        pass
    w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
    nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
    bg2 = nt.nodes.new('ShaderNodeBackground')            # sichtbarer Himmel (Kamera): heller Verlauf
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
        items = [i.identifier for i in sky.bl_rna.properties['sky_type'].enum_items]
        for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'SINGLE_SCATTERING', 'HOSEK_WILKIE'):
            if t in items:
                sky.sky_type = t; break
        try:
            sky.sun_elevation = math.radians(38); sky.sun_rotation = math.radians(215)
        except Exception:
            pass
        hsv = nt.nodes.new('ShaderNodeHueSaturation'); hsv.inputs['Saturation'].default_value = 0.45
        nt.links.new(sky.outputs['Color'], hsv.inputs['Color']); nt.links.new(hsv.outputs['Color'], bg.inputs['Color'])
        bg.inputs['Strength'].default_value = 0.16
    except Exception as e:
        print('sky', e)
        bg.inputs['Color'].default_value = (0.75, 0.8, 0.88, 1); bg.inputs['Strength'].default_value = 0.8
    sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.4; sun.color = (1.0, 0.95, 0.88); sun.angle = math.radians(1.0)
    so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(52), 0, math.radians(215))
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.clip_start = 0.1; cam.data.clip_end = 3000
    return sc, cam


def shot(sc, cam, name, c):
    loc = g2b(np.array(c['loc'])); tgt = g2b(np.array(c['tgt']))
    cam.location = Vector(loc.tolist())
    d = Vector(tgt.tolist()) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = c.get('lens', 24)
    sc.render.filepath = os.path.join(OUT, name + '.png')
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print('render', name, '%.1f s' % (time.time() - t))


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc, cam = setup_scene()
    cams = json.load(open(CAMS, encoding='utf-8'))
    if VIEWS:
        cams = {k: v for k, v in cams.items() if k in VIEWS}
    idx = json.load(open(os.path.join(RAW, 'index.json'), encoding='utf-8'))
    coll = bpy.data.collections.new('buildings'); sc.collection.children.link(coll)
    pc = bpy.data.collections.new('protos'); sc.collection.children.link(pc)
    pc.hide_render = True; pc.hide_viewport = True
    env = bpy.data.collections.new('env'); sc.collection.children.link(env)
    t0 = time.time()
    # benoetigte Kacheln: Vereinigung der Radien aller Ansichten
    need = []
    for e in idx['tiles']:
        bb = e['bounds']
        if bb is None:
            continue
        cx = (bb['min'][0] + bb['max'][0]) / 2; cz = (bb['min'][2] + bb['max'][2]) / 2
        for c in cams.values():
            r = c.get('radius', 300)
            if math.hypot(cx - c['tgt'][0], cz - c['tgt'][2]) < r + 80 or math.hypot(cx - c['loc'][0], cz - c['loc'][2]) < r + 80:
                need.append(e); break
    protos = {}
    for e in need:
        load_tile(os.path.join(RAW, e['file']), coll, pc, protos)
        if e.get('windows') and os.path.exists(os.path.join(RAW, e['windows'])):
            load_tile(os.path.join(RAW, e['windows']), coll, pc, protos)
    print('Kacheln', len(need), 'geladen %.1f s' % (time.time() - t0))
    terrain(env)
    rhino(env)
    landmarks(env)
    print('Umgebung geladen %.1f s' % (time.time() - t0))
    os.makedirs(OUT, exist_ok=True)
    for k, c in cams.items():
        shot(sc, cam, k, c)


main()

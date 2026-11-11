"""Headless-Blender-Vorschau (Ausbau 2): nur Geometrie im Umkreis der Kameras laden (RAM!).
blender -b --factory-startup -P preview2.py -- <shots.json> [<tiles_dir>]
shots.json: [{"name": "...", "loc": [x,y,z], "tgt": [x,y,z], "lens": 24, "r": 160}, ...]  (Spielkoordinaten)
Detailmodell aus data/detail_raw mit Anwendung von data/detail_replace + landmarks_replace (wie apply_landmark_replace.py).
Alle Materialien mit Backface-Culling (das Spiel rendert nur FrontSide)."""
import bpy, bmesh, sys, os, glob, math, json
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..')).replace('\\', '/') + '/'
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
SHOTS = json.load(open(argv[0]))
TILES = argv[1] if len(argv) > 1 else ROOT + 'web/assets/city/ground/'
PREV = ROOT + 'blender/previews/'
RES = (1280, 720)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
cx = [s['tgt'][0] for s in SHOTS] + [s['loc'][0] for s in SHOTS]; cz = [s['tgt'][2] for s in SHOTS] + [s['loc'][2] for s in SHOTS]
R = max(s.get('r', 150) for s in SHOTS)
X0, X1, Z0, Z1 = min(cx) - R, max(cx) + R, min(cz) - R, max(cz) + R
def tile_ok(fn):
    b = os.path.basename(fn).split('.')[0].split('_')
    try: tx, tz = int(b[1]), int(b[2])
    except Exception: return False
    return not (tx * 100 + 100 < X0 or tx * 100 > X1 or tz * 100 + 100 < Z0 or tz * 100 > Z1)
def imp(fn):
    before = set(bpy.data.objects)
    try: bpy.ops.import_scene.gltf(filepath=fn)
    except Exception as e: print('import failed', fn, e); return set()
    return set(bpy.data.objects) - before
n = 0
for fn in sorted(glob.glob(TILES + 'ground_*.glb')):
    if tile_ok(fn): imp(fn); n += 1
print('ground tiles', n)
for extra in ('furniture.glb',):
    if os.path.exists(TILES + extra): imp(TILES + extra)
# Gebaeude (prozedural, unkomprimiert)
BF = [f for f in sorted(glob.glob(ROOT + 'data/buildings_raw/bldg_*.glb')) if not f.endswith('_win.glb')]
if not BF: BF = [f for f in sorted(glob.glob(ROOT + 'web/assets/city/buildings/bldg_*.glb')) if not f.endswith('_win.glb')]   # meshopt (Blender >= 4.x)
for fn in BF:
    if tile_ok(fn): imp(fn)
# Detailmodell mit Ersetzungen
REP = []
for fn in sorted(glob.glob(ROOT + 'data/landmarks_replace/*.json') + glob.glob(ROOT + 'data/detail_replace/*.json')):
    J = json.load(open(fn, encoding='utf-8'))
    files = set(J.get('files') or [])
    for fp in J['footprints']: REP.append((fp, float(J.get('ymin', -2)), float(J.get('ymax', 60)), files))
BUILDING_FILES = {'bldg_walls', 'bldg_openings', 'bldg_roof', 'bldg_base', 'd0_windows', 'fix', 'kiosk'}
def inside(poly, x, z):
    c = False; n_ = len(poly); j = n_ - 1
    for i in range(n_):
        xi, zi = poly[i]; xj, zj = poly[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi: c = not c
        j = i
    return c
near_sq = not (X1 < -70 or X0 > 240 or Z1 < -12 or Z0 > 296)
if near_sq:
    for name in ('street', 'ground', 'green', 'street_furniture', 'green_trees', 'bldg_base', 'bldg_walls', 'bldg_roof', 'fix', 'kiosk', 'landmark_opernhaus', 'landmark_mahnmal', 'landmark_tiefgarage_pavillon'):
        fn = ROOT + 'data/detail_raw/' + name + '.glb'
        if not os.path.exists(fn): continue
        new = imp(fn)
        rel = [r for r in REP if (name in r[3] if r[3] else name in BUILDING_FILES)]
        if not rel: continue
        for o in [o for o in new if o.type == 'MESH']:
            mw = o.matrix_world; bm = bmesh.new(); bm.from_mesh(o.data); kill = []
            for f in bm.faces:
                c = mw @ f.calc_center_median(); gx, gy, gz = c.x, c.z, -c.y
                for fp, y0, y1, fs in rel:
                    if y0 <= gy <= y1 and inside(fp, gx, gz): kill.append(f); break
            if kill:
                bmesh.ops.delete(bm, geom=kill, context='FACES'); bm.to_mesh(o.data)
            bm.free()
for fn in sorted(glob.glob(ROOT + 'data/landmarks_raw/*.glb')):
    if 'lichtwolke' in fn: continue
    nm_ = os.path.basename(fn)[:-4]; jf = ROOT + 'data/landmarks_replace/' + nm_ + '.json'; near = near_sq
    if os.path.exists(jf):
        try:
            fps = json.load(open(jf, encoding='utf-8'))['footprints']
            near = any(X0 <= p[0] <= X1 and Z0 <= p[1] <= Z1 for fp in fps for p in fp)
        except Exception: pass
    if near: imp(fn)
for m in bpy.data.materials:
    m.use_backface_culling = True
    try: m.use_backface_culling_shadow = False
    except Exception: pass
print('objects', len(bpy.data.objects))

w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = nt.nodes['Background']
grad = nt.nodes.new('ShaderNodeTexGradient'); tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (0, -math.pi / 2, 0)
ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].color = (0.80, 0.84, 0.88, 1); ramp.color_ramp.elements[1].color = (0.38, 0.55, 0.80, 1)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
bg.inputs['Strength'].default_value = 1.0
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.2; sun.angle = math.radians(1.5)
so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so); so.rotation_euler = (math.radians(42), 0, math.radians(205))
try: sc.render.engine = 'BLENDER_EEVEE'
except TypeError: sc.render.engine = 'BLENDER_EEVEE_NEXT'
for attr, val in (('taa_render_samples', 32), ('use_shadows', True), ('use_raytracing', False)):
    try: setattr(sc.eevee, attr, val)
    except Exception: pass
try: sc.view_settings.view_transform = 'AgX'
except Exception: pass
sc.render.resolution_x, sc.render.resolution_y = RES
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.clip_start = 0.05; cam.data.clip_end = 3000
def Gv(x, y, z): return Vector((x, -z, y))
for s in SHOTS:
    if s.get('ortho'):
        cam.data.type = 'ORTHO'; cam.data.ortho_scale = s['ortho']
        cam.location = Gv(*s['loc']); cam.rotation_euler = (0, 0, 0)
    else:
        cam.data.type = 'PERSP'
        cam.location = Gv(*s['loc']); d = Gv(*s['tgt']) - cam.location
        cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler(); cam.data.lens = s.get('lens', 24)
    sc.render.filepath = PREV + s['name'] + '.png'
    bpy.ops.render.render(write_still=True); print('rendered', s['name'], flush=True)

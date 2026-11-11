"""Marktkirche -- Vorschauen (Blender 5 headless, Eevee 1280x720, Rueckseiten-Culling AN).
Kontext: data/ground_raw + data/buildings_raw Kacheln um die Kirche (alte LoD2-Kirche im Footprint wird entfernt).
Aufruf: blender -b --factory-startup -P mk_preview.py -- <model.glb> <replace.json> <out_prefix> <shots|all> [--noctx] [--samples=N]"""
import bpy, bmesh, sys, os, json, math
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, REPL, PREFIX, SHOTS = argv[:4]
NOCTX = '--noctx' in argv
SAMPLES = int(next((a.split('=')[1] for a in argv if a.startswith('--samples=')), '24'))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene

# Kameras im Spielframe: (Kamera x,y,z), (Ziel x,y,z), Brennweite
CAMS = {
    'ref_rathaus': ((-142.0, 1.7, 362.0), (-245.0, 36.0, 300.0), 30),     # Foto 03: Ecke Karmarsch-/Schmiedestrasse (Altes Rathaus jetzt im Kontext)
    'ref_kramer': ((-372.0, 1.7, 306.5), (-262.0, 44.0, 299.0), 45),
    'ref_sued': ((-238.0, 1.7, 338.0), (-247.0, 28.0, 300.0), 22),      # Hanns-Lilje-Platz, Suedseite (frei, geprueft mit camcheck)
    'portal_w': ((-291.0, 1.7, 301.5), (-271.0, 7.0, 299.3), 26),
    'luther': ((-261.2, 1.7, 320.5), (-258.3, 3.6, 311.0), 28),
    'portal_s': ((-219.0, 1.7, 317.0), (-220.8, 5.5, 305.9), 26),
    'luftbild_so': ((-120.0, 140.0, 420.0), (-240.0, 32.0, 300.0), 32),
    'luftbild_nw': ((-385.0, 120.0, 175.0), (-240.0, 32.0, 300.0), 32),
    'helm': ((-200.0, 1.7, 320.0), (-262.0, 79.0, 299.0), 120),
    'nord': ((-215.0, 1.7, 268.0), (-245.0, 22.0, 295.0), 22),          # Marktplatz-Nordostecke
    'nord_west': ((-268.0, 1.7, 266.0), (-252.0, 20.0, 296.0), 22),
    'luther_nah': ((-259.6, 2.6, 315.2), (-258.25, 3.7, 311.06), 32),
    'portal_n': ((-256.1, 1.7, 273.0), (-254.45, 4.8, 284.1), 24),
    'chor': ((-184.0, 1.7, 291.0), (-213.0, 15.0, 293.0), 26),
    'turm_nah': ((-232.0, 74.0, 336.0), (-262.0, 68.0, 299.0), 40),
    'dach_nah': ((-222.0, 44.0, 333.0), (-236.0, 30.0, 300.0), 45),
}
# Kameras im Kirchen-Lokalframe (u = Achse W->O, v = Nord, h = Hoehe): Kontroll-/QA-Ansichten
CAMS_L = {
    'aerial_ne': ((90, 90, 140), (20, 0, 20), 28), 'base_low': ((52, 6, 1.0), (30, 25, 3), 22),
    'chor_se': ((70, -18, 1.7), (50, 0, 12), 26), 'gables_nw_high': ((-25, 25, 58), (0, 0, 58), 35),
    'gables_se_high': ((25, -25, 58), (0, 0, 58), 35), 'luther_w': ((-6, -14, 1.7), (3, -11, 3.5), 28),
    'north_wall': ((32, 20, 1.7), (22, 13, 8), 24), 'nw_eye': ((-14, 22, 1.7), (5, 5, 20), 22),
    'roof_n_high': ((25, 40, 45), (25, 5, 28), 30), 'schmiede': ((85, 0, 1.7), (50, 0, 15), 24),
    'south_sw': ((0, -32, 1.7), (25, -13, 10), 24), 'south_up': ((25, -20, 1.7), (25, -10, 18), 18),
    'spire_top': ((-20, -25, 92), (0, 0, 85), 35), 'westportal': ((-20, -2, 2.0), (-8.75, 0, 6.5), 30),
    'z_eave_soffit': ((20, 22, 24), (25, 13.5, 21), 30), 'z_lantern': ((-12, 0, 75), (0, 0, 74), 35),
    'z_portal_n': ((10.0, 23, 1.7), (10.3, 13.6, 4), 26), 'z_portal_s_ornament': ((39.0, -16.0, 2.2), (40.2, -13.3, 4.6), 24),
    'z_portal_s_steps': ((37.5, -16.5, 1.4), (39.98, -13.3, 1.2), 26), 'z_threshold_s': ((40.2, -15.6, 1.75), (39.98, -13.6, 0.9), 30),
    'z_tower_ne_quoins': ((12, 14, 38), (6.87, 8.09, 32), 30), 'z_tower_s_quoins': ((-2, -20, 25), (6.87, -8.09, 30), 30),
    'vorhalle': ((-14.5, -1.5, 1.7), (-5, 0.5, 3), 24), 'georg': ((-13.5, 2.0, 4.6), (-9.4, 3.2, 6.4), 35),
    'tod_mann': ((-13.5, -2.0, 4.6), (-9.4, -3.2, 6.4), 35), 'schaukasten': ((-11.5, 7.5, 1.6), (-7.3, 5.6, 1.6), 30),
    'uhr_sued': ((0, -26, 56.5), (0, -9, 56.8), 60), 'reliefs_nord': ((25.25, 19, 2.0), (25.25, 13.4, 3.4), 30),
    'luther_gesicht': ((0.32, -15.9, 3.4), (2.17, -12.35, 4.3), 45), 'epitaph_barock': ((-14, -5.35, 2.5), (-6.87, -5.35, 4.8), 30),
    'epitaph_luther': ((-3.5, -11.5, 2.2), (6.87, -11.0, 3.6), 32), 'laterne_nah': ((-6, -3, 72.5), (0, 0, 72.5), 40),
    'westportal_total': ((-24, 0.5, 1.7), (-8.75, 0, 5.5), 24), 'luther_kopf': ((1.2, -14.2, 5.0), (2.17, -12.35, 4.85), 50),
}
try:
    _PL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'plan_local.json')))
    _O, _U, _V = _PL['origin'], _PL['U'], _PL['V']
    _g = lambda p: (_O[0] + p[0] * _U[0] + p[1] * _V[0], p[2], _O[1] + p[0] * _U[1] + p[1] * _V[1])
    for k_, (c_, t_, l_) in CAMS_L.items():
        CAMS[k_] = (_g(c_), _g(t_), l_)
except Exception as e:
    print('local cams', e)
shots = list(CAMS) if SHOTS == 'all' else SHOTS.split(',')


def B(p):   # Spielframe -> Blender
    return Vector((p[0], -p[2], p[1]))


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


model = import_glb(GLB)
print('model objects', len(model))
fp = json.load(open(REPL))['footprints']


def inside(x, y, poly):
    c = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1: c = not c
    return c


polysB = [[(p[0], -p[1]) for p in f] for f in fp]
TILES_X = range(-4, -1); TILES_Z = range(2, 5)      # 100-m-Kacheln um die Kirche (x -400..-100, z 200..500); RAM/Zeit sparen
_ax = [p[0] for pl in polysB for p in pl]; _ay = [p[1] for pl in polysB for p in pl]
BB = (min(_ax) - 1, max(_ax) + 1, min(_ay) - 1, max(_ay) + 1)
if not NOCTX:
    for tx in TILES_X:
        for tz in TILES_Z:
            for sub, pat in (('ground_raw', 'ground_%d_%d.glb'), ('buildings_raw', 'bldg_%d_%d.glb'), ('buildings_raw', 'bldg_%d_%d_win.glb')):
                p = os.path.join(ROOT, 'data', sub, pat % (tx, tz))
                if not os.path.isfile(p): continue
                objs = import_glb(p)
                if sub != 'buildings_raw': continue
                for o in objs:
                    if o.type != 'MESH': continue
                    me = o.data; mw = o.matrix_world
                    wb = [mw @ Vector(c) for c in o.bound_box]
                    if max(v.x for v in wb) < BB[0] or min(v.x for v in wb) > BB[1] or max(v.y for v in wb) < BB[2] or min(v.y for v in wb) > BB[3]:
                        continue                                   # ganz ausserhalb des Kirchen-Grundrisses
                    if me.users > 1:                               # instanziertes Bauteil: ganzes Objekt entfernen, falls im Grundriss
                        c = sum(wb, Vector()) / 8
                        if any(inside(c.x, c.y, pl) for pl in polysB): bpy.data.objects.remove(o)
                        continue
                    bm = bmesh.new(); bm.from_mesh(me)
                    kill = []
                    for f in bm.faces:
                        c = mw @ f.calc_center_median()
                        if BB[0] <= c.x <= BB[1] and BB[2] <= c.y <= BB[3] and any(inside(c.x, c.y, pl) for pl in polysB): kill.append(f)
                    if kill: bmesh.ops.delete(bm, geom=kill, context='FACES')
                    bm.to_mesh(me); bm.free()
    for extra in ('altes_rathaus',):
        p = os.path.join(ROOT, 'data', 'landmarks_raw', extra + '.glb')
        if os.path.isfile(p): import_glb(p)
for m in bpy.data.materials:
    m.use_backface_culling = True
    try: m.use_backface_culling_shadow = False
    except Exception: pass

# Welt + Sonne
w = bpy.data.worlds.new('W'); SC.world = w; w.use_nodes = True
bg = next(n for n in w.node_tree.nodes if n.type == 'BACKGROUND')
bg.inputs['Color'].default_value = (0.6, 0.66, 0.74, 1); bg.inputs['Strength'].default_value = 0.8
sun = bpy.data.lights.new('Sonne', 'SUN'); sun.energy = 3.0; sun.angle = math.radians(1.5)
so = bpy.data.objects.new('Sonne', sun); SC.collection.objects.link(so)
so.rotation_euler = (math.radians(50), 0, math.radians(215))
for eng in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
    try:
        SC.render.engine = eng; break
    except TypeError:
        continue
try: SC.eevee.taa_render_samples = SAMPLES
except Exception: pass
try:
    SC.eevee.use_shadows = True
except Exception: pass
SC.render.resolution_x, SC.render.resolution_y = 1280, 720
SC.render.image_settings.file_format = 'PNG'
try: SC.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
cam = bpy.data.cameras.new('Cam'); co = bpy.data.objects.new('Cam', cam); SC.collection.objects.link(co); SC.camera = co
cam.clip_start = 0.1; cam.clip_end = 3000
for s in shots:
    c, t, lens = CAMS[s]
    co.location = B(c)
    d = B(t) - B(c)
    co.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cam.lens = lens
    # Sonne je Aufnahme: von hinten-seitlich der Kamera (45 Grad versetzt, 40 Grad hoch), damit die gezeigte Seite Licht hat
    az = math.atan2(d.y, d.x) + math.radians(40)
    so.rotation_euler = (math.radians(50), 0, az - math.pi / 2)   # Lichtrichtung horizontal = az
    out = PREFIX + s + '.png'
    SC.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print('RENDERED', out)

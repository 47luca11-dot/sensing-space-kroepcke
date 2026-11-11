"""Vorschau-Renderings (Eevee, 1280x720, Backface-Culling AN) des Alten Rathauses, optional im Kontext
(Boden-/Gebaeudekacheln aus data/ground_raw, data/buildings_raw, Marktkirche aus data/landmarks_raw).
Aufruf: blender -b --factory-startup -P blender_preview.py -- <rathaus.glb> <out_prefix> <shots|all> [--ctx] [--blend=<pfad>] [--ctxblend=<pfad>]
Kameras im Spielframe (x Ost, y oben, z Sued)."""
import bpy, sys, os, math, glob, json
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUTP, SHOTS = argv[0], argv[1], argv[2]
CTX = '--ctx' in argv
BLEND = next((a.split('=', 1)[1] for a in argv if a.startswith('--blend=')), None)
CTXBLEND = next((a.split('=', 1)[1] for a in argv if a.startswith('--ctxblend=')), None)
RES = next((a.split('=', 1)[1] for a in argv if a.startswith('--res=')), '1280x720')
CTXR = float(next((a.split('=', 1)[1] for a in argv if a.startswith('--ctxr=')), '170'))
RX, RY = [int(v) for v in RES.split('x')]
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/"
CX, CZ = -191.0, 356.0

if CTX and CTXBLEND and os.path.exists(CTXBLEND):
    bpy.ops.wm.open_mainfile(filepath=CTXBLEND)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if CTX:
        R = CTXR
        def tiles(folder, pat):
            out = []
            for fn in glob.glob(os.path.join(ROOT, folder, pat)):
                b = os.path.basename(fn).split('.')[0].split('_')
                try:
                    tx, tz = int(b[1]), int(b[2])
                except Exception:
                    continue
                x0, z0 = tx * 100.0, tz * 100.0
                dx = max(x0 - CX, 0, CX - (x0 + 100)); dz = max(z0 - CZ, 0, CZ - (z0 + 100))
                if math.hypot(dx, dz) < R: out.append(fn)
            return sorted(out)
        bl = tiles('data/buildings_raw', 'bldg_*.glb')
        for fn in tiles('data/ground_raw', 'ground_*.glb') + bl:
            try:
                bpy.ops.import_scene.gltf(filepath=fn)
            except Exception as e:
                print('ctx import failed', fn, e)
        for extra in ('marktkirche.glb',):
            p = os.path.join(ROOT, 'data/landmarks_raw', extra)
            if os.path.exists(p): bpy.ops.import_scene.gltf(filepath=p)
        for o in bpy.data.objects:
            if o.name.startswith('col_') or 'collision' in o.name.lower(): o.hide_render = True
        if CTXBLEND:
            bpy.ops.wm.save_as_mainfile(filepath=CTXBLEND); print('saved ctx', CTXBLEND)
sc = bpy.context.scene
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=GLB)
new = set(bpy.data.objects) - before
for o in new:
    if o.name.startswith('col_'): o.hide_render = True; o.hide_viewport = True
for m in bpy.data.materials:
    try:
        m.use_backface_culling = True
    except Exception:
        pass

if BLEND:
    # nur das Rathaus speichern (eigene Datei)
    pass

# Licht / Welt
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = nt.nodes['Background']
sky = nt.nodes.new('ShaderNodeTexSky')
for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'SINGLE_SCATTERING', 'HOSEK_WILKIE'):
    try: sky.sky_type = t; break
    except Exception: pass
SUN_EL, SUN_AZ = math.radians(38), math.radians(215)
try:
    sky.sun_elevation = SUN_EL; sky.sun_rotation = SUN_AZ; sky.altitude = 60
except Exception: pass
nt.links.new(sky.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.25
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.4; sun.angle = math.radians(0.6); sun.color = (1.0, 0.96, 0.9)
so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
d = Vector((math.sin(SUN_AZ) * math.cos(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL)))
so.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
try: sc.render.engine = 'BLENDER_EEVEE'
except TypeError: sc.render.engine = 'BLENDER_EEVEE_NEXT'
ee = sc.eevee
for k, v in (('taa_render_samples', 24), ('use_raytracing', False), ('use_shadows', True), ('use_gtao', True), ('shadow_ray_count', 1), ('shadow_step_count', 4)):
    try: setattr(ee, k, v)
    except Exception: pass
vs = sc.view_settings
try: vs.view_transform = 'AgX'; vs.look = 'AgX - Medium High Contrast'
except Exception: pass
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.clip_start = 0.1; cam.data.clip_end = 3000; cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = 36


def G(x, y, z): return Vector((x, -z, y))


SH = {}
def shot(name, loc, tgt, lens=24, ortho=None, res=None):
    SH[name] = (loc, tgt, lens, ortho, res)


# ---- Kameras (Spielframe)
# Test
shot('test_front', (6, 4, -16), (6, 5, 0), lens=30)
shot('test_oblique', (-6, 3, -10), (6, 6, 0), lens=24)
shot('test_close', (2.0, 6.5, -3.2), (2.0, 6.0, 0), lens=30)
shot('test_top', (6, 22, -8), (6, 8, 1), lens=30)
# Rathaus: Referenzfoto-Standpunkte
shot('markt_front', (-226.0, 1.9, 318.5), (-207.0, 9.5, 335.5), lens=22)                  # Foto 05 (frontal NW-Fassade)
shot('markt_w', (-245.0, 1.9, 326.0), (-210.0, 10.0, 343.0), lens=24)                     # Foto 08 (von W)
shot('ne_gable', (-178.0, 1.9, 318.0), (-188.0, 13.0, 330.5), lens=22)                    # Foto 01 (Marktgiebel NO)
shot('east', (-127.0, 1.7, 336.0), (-178.0, 11.0, 354.0), lens=24)                        # Foto 16 (Ecke Karmarsch-/Schmiedestr. von O)
shot('karmarsch', (-150.0, 1.9, 405.0), (-176.0, 10.0, 372.0), lens=24)                   # Foto 10 (SO-Fassade)
shot('south', (-180.0, 1.9, 412.0), (-190.0, 12.0, 385.0), lens=24)                       # flickr 02 (Suedecke)
shot('koebelinger', (-232.0, 1.9, 353.0), (-196.0, 9.0, 380.0), lens=24)                  # Foto 14 (Dogenpalast)
shot('west_corner', (-236.0, 1.9, 341.0), (-219.0, 9.0, 350.0), lens=24)                  # Foto 07 (Westecke)
shot('schmiede_n', (-195.5, 1.7, 289.0), (-180.0, 11.0, 338.0), lens=30)                   # Foto 09 (Schmiedestr. von N)
shot('aerial', (-120.0, 75.0, 300.0), (-191.0, 8.0, 357.0), lens=30)
shot('aerial_sw', (-270.0, 70.0, 430.0), (-191.0, 6.0, 355.0), lens=30)
shot('top', (-191.0, 120.0, 356.0), (-191.0, 0.0, 356.01), ortho=80)
shot('close_window', (-197.0, 6.5, 318.0), (-200.5, 6.5, 331.0), lens=30)
shot('close_gable', (-176.0, 14.0, 322.0), (-188.5, 17.0, 331.0), lens=30)
shot('close_arcade', (-162.0, 1.9, 384.0), (-175.0, 3.0, 375.0), lens=22)
shot('laube', (-229.0, 1.9, 357.0), (-219.5, 3.5, 349.5), lens=26)                         # Laube an der Westecke
shot('arcade_in', (-175.5, 1.7, 374.4), (-184.6, 2.2, 386.1), lens=20)                    # im Laubengang Karmarschstr.
shot('dormer', (-214.0, 13.5, 322.5), (-207.0, 13.0, 330.5), lens=30)                      # mittleres Zwerchhaus Marktseite
shot('gable_niches', (-183.6, 11.2, 328.4), (-187.0, 11.0, 331.4), lens=30)                     # Blendnischen/Masswerk NO-Giebel nah
shot('pinnacles', (-178.9, 24.0, 319.3), (-189.5, 20.5, 328.5), lens=30)                  # Giebelfialen NO-Giebel
shot('frieze_close', (-202.8, 3.2, 325.5), (-199.75, 4.85, 328.82), lens=35)              # Terrakottafries Marktseite (Foto 13)
shot('entrance_front', (-212.4, 1.7, 328.2), (-206.27, 2.4, 334.81), lens=24)             # Eingang Marktseite frontal (Foto 05)
shot('shop_viviry', (-178.6, 1.6, 386.6), (-181.6, 1.5, 381.4), lens=26)                  # Laubengang: Viviry durch das Glas
shot('shop_stichnoth', (-165.6, 1.6, 368.7), (-168.0, 1.2, 363.2), lens=26)               # Laubengang: Stichnoth-Vitrinen

# QA-Standpunkte (unabhaengige Pruefung, Ausgabe qa2_altes_rathaus_<name>.png mit --prefix)
for _n, (_l, _t, _f) in {
    'n_corner_low':     ((-199.0, 1.7, 301.0), (-194.0, 9.0, 325.0), 22),
    'f2_schmiede':      ((-150.0, 1.7, 345.0), (-190.0, 7.0, 330.0), 24),
    'e_corner':         ((-146.0, 1.7, 361.0), (-163.0, 6.0, 361.0), 24),
    'shopsigns':        ((-168.0, 1.7, 376.0), (-178.0, 3.2, 374.0), 24),
    's_corner':         ((-198.0, 1.7, 405.0), (-186.0, 7.0, 386.0), 22),
    'koeb_from_s':      ((-188.0, 1.7, 400.0), (-212.0, 6.0, 368.0), 24),
    'entrance_stairs':  ((-213.24, 1.6, 333.07), (-206.2, 1.5, 334.8), 22),
    'laube_joint':      ((-232.0, 1.7, 362.0), (-218.0, 4.0, 352.0), 24),
    'aerial_nw':        ((-265.0, 55.0, 290.0), (-192.0, 8.0, 356.0), 28),
    'aerial_se':        ((-130.0, 45.0, 420.0), (-190.0, 8.0, 356.0), 28),
    'roof_gable_back':  ((-180.0, 30.0, 365.0), (-192.0, 16.0, 330.0), 26),
    'courtyard_top':    ((-188.0, 45.0, 375.0), (-192.0, 10.0, 355.0), 30),
    'pinnacles_sw':     ((-224.8, 18.0, 360.1), (-215.6, 18.0, 352.4), 30),
    'arcade_karm_in':   ((-182.5, 1.7, 384.0), (-168.0, 2.2, 365.0), 20),
    'eave_karm_up':     ((-168.3, 1.7, 378.5), (-172.8, 15.6, 375.0), 35),
    'arcade_end_s':     ((-192.1, 1.7, 393.5), (-186.4, 3.0, 387.0), 28),
    'arcade_end_e':     ((-157.06, 1.7, 353.5), (-164.3, 3.0, 358.6), 28),
    'gable_s_windows':  ((-182.0, 17.0, 398.0), (-187.5, 17.9, 386.0), 35),
    'ridge_mf':         ((-212.0, 22.0, 330.1), (-202.5, 21.0, 340.4), 50),
    'aerial_nw2':       ((-245.0, 50.0, 300.0), (-192.0, 8.0, 356.0), 28),
    's_corner2':        ((-170.0, 1.7, 400.0), (-186.0, 7.0, 387.0), 22),
    'laube2':           ((-232.0, 1.7, 343.0), (-219.5, 3.0, 350.0), 24),
    'dogen_colonnade':  ((-205.4, 3.5, 375.5), (-200.0, 6.8, 371.0), 35),
    'gable_ring_close': ((-187.4, 20.0, 321.9), (-191.9, 19.6, 325.87), 50),
    'gable_foot_ne':    ((-183.5, 9.3, 326.2), (-188.0, 9.75, 330.1), 45),
    'gable_foot_af':    ((-167.3, 15.3, 377.5), (-172.0, 15.9, 373.8), 45),
    'shop_slit_f2a':    ((-165.15, 1.6, 354.56), (-163.44, 1.4, 358.97), 35),
}.items():
    shot(_n, _l, _t, lens=_f)

names = list(SH.keys()) if SHOTS == 'all' else SHOTS.split(',')
for name in names:
    if name not in SH: continue
    loc, tgt, lens, ortho, res = SH[name]
    sc.render.resolution_x, sc.render.resolution_y = res or (RX, RY)
    if ortho: cam.data.type = 'ORTHO'; cam.data.ortho_scale = ortho
    else: cam.data.type = 'PERSP'; cam.data.lens = lens
    cam.location = G(*loc); dd = G(*tgt) - cam.location
    cam.rotation_euler = dd.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = OUTP + name + '.png'
    bpy.ops.render.render(write_still=True)
    print('RENDERED', sc.render.filepath)

if BLEND:
    # Kontext entfernen, nur Rathaus behalten
    for o in list(bpy.data.objects):
        if o not in new and o.type == 'MESH':
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True) if hasattr(bpy.ops.outliner, 'orphans_purge') else None
    bpy.ops.wm.save_as_mainfile(filepath=BLEND, compress=True)
    print('saved blend', BLEND)

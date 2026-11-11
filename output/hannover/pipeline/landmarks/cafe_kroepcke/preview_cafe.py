"""Vorschau-Renderings des Cafe Kroepcke im Kontext des Detailmodells (altes Cafe per Ersatzpolygon entfernt).
Aufruf: blender -b --factory-startup -P preview_cafe.py -- <cafe.glb> <replace.json> <out_prefix> [shot,shot,...] [--nocontext]
Kameras im Spielframe (x Ost, y oben, z Sued)."""
import bpy, bmesh, sys, os, json, math, glob
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
CAFE, REPL, OUTP = argv[0], argv[1], argv[2]
SHOTS = argv[3].split(',') if len(argv) > 3 and not argv[3].startswith('--') else None
NOCTX = '--nocontext' in argv
FAST = '--fast' in argv
CTX = next((a.split('=', 1)[1] for a in argv if a.startswith('--ctx=')), None)   # vorbereitete Kontext-.blend
DETAIL = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/detail_raw/"

if CTX and os.path.exists(CTX):
    bpy.ops.wm.open_mainfile(filepath=CTX); NOCTX_LOAD = True
else:
    bpy.ops.wm.read_factory_settings(use_empty=True); NOCTX_LOAD = False
sc = bpy.context.scene
rep = json.load(open(REPL))
FPS = [np.array(f, float) for f in rep['footprints']]
LAYERS = set(rep.get('layers', ['bldg_walls', 'bldg_openings', 'bldg_roof', 'bldg_base']))


def inside_np(x, z, poly):
    ins = np.zeros(len(x), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        cond = (y1 > z) != (y2 > z)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = x1 + (z - y1) * (x2 - x1) / (y2 - y1)
        ins ^= cond & (x < xi)
    return ins


removed = 0
if not NOCTX and not NOCTX_LOAD:
    for fn in sorted(glob.glob(DETAIL + '*.glb')):
        layer = os.path.splitext(os.path.basename(fn))[0]
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=fn)
        if layer not in LAYERS: continue
        for o in set(bpy.data.objects) - before:
            if o.type != 'MESH': continue
            me = o.data; mw = o.matrix_world
            n = len(me.polygons)
            cen = np.zeros(n * 3); me.polygons.foreach_get('center', cen); cen = cen.reshape(-1, 3)
            M = np.array(mw); cw = cen @ M[:3, :3].T + M[:3, 3]
            gx, gy, gz = cw[:, 0], cw[:, 2], -cw[:, 1]
            sel = np.zeros(n, bool)
            for fp in FPS: sel |= inside_np(gx, gz, fp)
            sel &= (gy >= rep['ymin']) & (gy <= rep['ymax'])
            if sel.any():
                bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
                bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.nonzero(sel)[0]], context='FACES')
                bm.to_mesh(me); bm.free(); removed += int(sel.sum())
print('PREVIEW removed old faces', removed)
if not NOCTX and not NOCTX_LOAD:
    for extra in ("C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/landmarks_raw/kroepcke_uhr.glb",):
        if os.path.exists(extra): bpy.ops.import_scene.gltf(filepath=extra)
if CTX and not NOCTX_LOAD:
    bpy.ops.wm.save_as_mainfile(filepath=CTX); print('saved context', CTX)
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=CAFE)
cafe_objs = [o for o in set(bpy.data.objects) - before]
for o in cafe_objs:
    if o.name.startswith('col_'): o.hide_render = True

# ---------------------------------------------------------------- Licht, Welt, Farbmanagement
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = nt.nodes['Background']
sky = nt.nodes.new('ShaderNodeTexSky')
for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'SINGLE_SCATTERING', 'HOSEK_WILKIE'):
    try: sky.sky_type = t; break
    except Exception: pass
SUN_EL, SUN_AZ = math.radians(34), math.radians(200)          # Sonne aus SSW (Nachmittag)
try:
    sky.sun_elevation = SUN_EL; sky.sun_rotation = SUN_AZ; sky.altitude = 60; sky.air_density = 1.0; sky.dust_density = 1.5
except Exception: pass
nt.links.new(sky.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.22
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.2; sun.angle = math.radians(0.6)
sun.color = (1.0, 0.96, 0.9)
try: sun.shadow_maximum_resolution = 0.004 if FAST else 0.002
except Exception: pass
so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
# Sonnenrichtung: Azimut SUN_AZ (von Nord im Uhrzeigersinn, Blender: +Y = Nord), Hoehe SUN_EL
d = Vector((math.sin(SUN_AZ) * math.cos(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL)))
so.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
try: sc.render.engine = 'BLENDER_EEVEE'
except TypeError: sc.render.engine = 'BLENDER_EEVEE_NEXT'
ee = sc.eevee
for k, v in (('taa_render_samples', 12 if FAST else 32), ('use_raytracing', not FAST), ('use_shadows', True), ('shadow_resolution_scale', 0.5 if FAST else 1.0), ('use_gtao', True), ('shadow_ray_count', 1), ('shadow_step_count', 4 if FAST else 6)):
    try: setattr(ee, k, v)
    except Exception: pass
try:
    ee.ray_tracing_options.resolution_scale = '2'
except Exception: pass
vs = sc.view_settings
try: vs.view_transform = 'AgX'; vs.look = 'AgX - Medium High Contrast'
except Exception:
    try: vs.view_transform = 'Filmic'
    except Exception: pass
sc.render.resolution_x, sc.render.resolution_y = 1600, 900
sc.render.film_transparent = False
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.clip_start = 0.1; cam.data.clip_end = 2000; cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = 36


def G(x, y, z): return Vector((x, -z, y))


def shot(name, loc, tgt, lens=24, res=(1600, 900), dusk=False, ortho=None):
    if SHOTS and name not in SHOTS: return
    sc.render.resolution_x, sc.render.resolution_y = res
    if ortho: cam.data.type = 'ORTHO'; cam.data.ortho_scale = ortho
    else: cam.data.type = 'PERSP'
    cam.location = G(*loc); dd = G(*tgt) - cam.location
    cam.rotation_euler = dd.to_track_quat('-Z', 'Y').to_euler(); cam.data.lens = lens
    if dusk:
        sun.energy = 0.08; sun.color = (1.0, 0.75, 0.55); bg.inputs[1].default_value = 1.0; vs.exposure = 1.1
        try: sky.sun_elevation = math.radians(-1.0)
        except Exception: pass
    else:
        sun.energy = 3.2; sun.color = (1.0, 0.96, 0.9); bg.inputs[1].default_value = 0.22; vs.exposure = 0.0
        try: sky.sun_elevation = SUN_EL
        except Exception: pass
    sc.render.filepath = OUTP + name + '.png'
    bpy.ops.render.render(write_still=True)
    print('RENDERED', sc.render.filepath)


# Referenzfoto 2021-01-03 (Lockdown, Daemmerung): Kamera aus Peilungen (Uhr, Portal, Schriftzug), HFOV ~67 Grad
shot('ref', (20.4, 1.55, 41.2), (19.4, 1.55, 1.2), lens=27.0, res=(1600, 1200), dusk=True)
shot('ref_day', (20.4, 1.55, 41.2), (19.4, 1.55, 1.2), lens=27.0, res=(1600, 1200))
# Foto 'Moevenpick Februar 2016' (panoramio): Kamera aus Peilungen (Pfeiler der kleinen Boegen, rotes Portal), f ~ 1250 px / 1280
shot('foto2016', (-10.0, 1.6, 3.5), (13.48, 1.9, 22.18), lens=35.2, res=(1600, 1200))
# Panorama/Moevenpick-2016-Blickrichtung (vom Platz aus Westen)
shot('platz', (-6.0, 1.6, 26.0), (30.0, 3.5, 18.0), lens=24)
# Von der Uhr (0, 1.7, 6) zum Cafe
shot('clock', (0.0, 1.7, 6.0), (29.0, 4.0, 20.0), lens=26)
# Luftbild schraeg (von Suedwest)
shot('aerial', (-12.0, 38.0, 68.0), (34.0, 3.0, 24.0), lens=35)
shot('aerial_s', (38.0, 42.0, 92.0), (36.0, 3.0, 26.0), lens=35)
shot('top', (36.0, 80.0, 25.0), (36.0, 0.0, 25.01), ortho=64)
shot('east_top', (74.0, 22.0, 30.0), (46.0, 4.0, 26.0), lens=30)
shot('north_top', (40.0, 26.0, -2.0), (40.0, 4.0, 16.0), lens=30)
# Nahaufnahme Hauptportal (rot)
shot('entrance', (13.6, 1.6, 31.6), (20.2, 1.8, 27.8), lens=30)
# Opernplatz-Seite (Sued)
shot('south', (52.0, 1.7, 66.0), (44.0, 5.0, 38.0), lens=26)
# Nord (Gasse)
shot('north', (20.0, 1.7, -2.0), (32.0, 5.0, 10.0), lens=24)
# Dachdetails: kleine Tonnen / Kehlen / Lichtband
shot('roof_detail', (10.0, 10.5, 0.0), (24.0, 5.5, 13.0), lens=30)
shot('roof_junction', (14.0, 13.0, 30.0), (27.0, 6.5, 21.0), lens=30)
# Detail Giebel kleine Tonnen
shot('smallarches', (7.5, 2.2, 16.0), (17.5, 3.8, 14.0), lens=30)

"""Kroepcke-Center -- Vorschau-Renderings (Eevee, Sonne + Himmel, AgX) im Kontext des Detailmodells.
Aufruf: blender -b --factory-startup -P kc_preview.py -- <kc.glb> <replace.json> <outdir> <cache.blend> [shot,shot,...]
Beim ersten Lauf wird die Kontextszene (data/detail_raw/*.glb + Uhr) importiert, die alte Kroepcke-Center-
Geometrie innerhalb des Ersetzungs-Grundrisses geloescht und als cache.blend gespeichert.
"""
import bpy, bmesh, sys, os, json, math, glob
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
KC_GLB, REPL, OUTDIR, CACHE = argv[:4]
SHOTS = argv[4].split(',') if len(argv) > 4 and argv[4] else None
H = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
DETAIL = H + 'data/detail_raw/'
CONTEXT = ['bldg_walls', 'bldg_openings', 'bldg_roof', 'bldg_base', 'street', 'ground', 'green', 'green_trees',
           'street_furniture', 'kiosk', 'landmark_opernhaus', 'landmark_tiefgarage_pavillon', 'landmark_mahnmal', 'fix']


def pip(px, py, poly):
    """Punkt-in-Polygon (vektorisiert, gerade-ungerade)."""
    inside = np.zeros(len(px), bool)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        cond = ((y1 > py) != (y0 > py))
        xi = (x0 - x1) * (py - y1) / ((y0 - y1) + 1e-12) + x1
        inside ^= cond & (px < xi)
        x0, y0 = x1, y1
    return inside


def build_context():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rep = json.load(open(REPL))
    polys = [np.array([(p[0], -p[1]) for p in fp]) for fp in rep['footprints']]   # Spiel (x,z) -> Blender (x,y)
    removed = 0
    for n in CONTEXT:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=DETAIL + n + '.glb')
        new = [o for o in set(bpy.data.objects) - before]
        if n.startswith('bldg_'):
            for o in new:
                if o.type != 'MESH': continue
                me = o.data; mw = np.array(o.matrix_world)
                nv = len(me.vertices); co = np.zeros(nv * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
                co = co @ mw[:3, :3].T + mw[:3, 3]
                npoly = len(me.polygons); cen = np.zeros(npoly * 3, np.float32)
                me.polygons.foreach_get('center', cen); cen = cen.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
                kill = np.zeros(npoly, bool)
                for P in polys:
                    kill |= pip(cen[:, 0], cen[:, 1], P) & (cen[:, 2] > rep['ymin']) & (cen[:, 2] < rep['ymax'])
                if kill.any():
                    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
                    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.flatnonzero(kill)], context='FACES')
                    bm.to_mesh(me); bm.free(); removed += int(kill.sum())
    print('alte Geometrie entfernt (Dreiecke/Polygone):', removed)
    for f in (H + 'data/landmarks_raw/kroepcke_uhr.glb', H + 'data/landmarks_raw/lichtwolke.glb'):
        if os.path.exists(f): bpy.ops.import_scene.gltf(filepath=f)
    for o in bpy.data.objects: o['ctx'] = 1
    bpy.ops.wm.save_as_mainfile(filepath=CACHE)


if not os.path.exists(CACHE):
    build_context()
bpy.ops.wm.open_mainfile(filepath=CACHE)
SC = bpy.context.scene
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=KC_GLB)
for o in set(bpy.data.objects) - before:
    if 'collision' in o.name.lower(): o.hide_render = True
# Rueckseiten wie im Spiel (three.js FrontSide) ausblenden, um falsch orientierte Flaechen zu sehen
for o in set(bpy.data.objects) - before:
    if o.type == 'MESH':
        for m in o.data.materials:
            if m and m.name.startswith('KC'):
                m.use_backface_culling = not ('Glass' in m.name)
# Glas im Rendering: Alpha-Blend
for m in bpy.data.materials:
    if 'Glass' in m.name or 'Glas' in m.name:
        try: m.surface_render_method = 'BLENDED'
        except Exception: pass

# ---------------- Licht, Welt, Farbe
w = bpy.data.worlds.new('sky'); SC.world = w; w.use_nodes = True
nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == 'BACKGROUND')
sky = nt.nodes.new('ShaderNodeTexSky')
for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'HOSEK_WILKIE'):
    try: sky.sky_type = t; break
    except Exception: pass
for k, v in (('sun_disc', False), ('sun_elevation', math.radians(38)), ('sun_rotation', math.radians(200)), ('air_density', 1.0), ('aerosol_density', 1.2)):
    try: setattr(sky, k, v)
    except Exception as ex: print('sky', k, ex)
nt.links.new(sky.outputs['Color'], bg.inputs['Color']); bg.inputs['Strength'].default_value = 0.12
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.4; sun.angle = math.radians(0.6); sun.color = (1.0, 0.93, 0.84)
so = bpy.data.objects.new('sun', sun); SC.collection.objects.link(so)
# Sonne aus SSW-W (Nachmittag), damit die Nord-/Westfassaden Streiflicht bekommen
so.rotation_euler = (math.radians(52), 0, math.radians(-118))
for e in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
    try: SC.render.engine = e; break
    except TypeError: pass
ee = SC.eevee
for k, v in (('taa_render_samples', int(os.environ.get('KC_SAMPLES', '24'))), ('use_raytracing', os.environ.get('KC_RT', '0') == '1'), ('use_shadows', True),
             ('shadow_ray_count', 1), ('shadow_step_count', 6), ('fast_gi_method', 'GLOBAL_ILLUMINATION')):
    try: setattr(ee, k, v)
    except Exception as ex: print('eevee', k, ex)
try:
    SC.view_settings.view_transform = 'AgX'; SC.view_settings.look = 'AgX - Medium High Contrast'
except Exception as ex: print(ex)
SC.view_settings.exposure = -0.35
SC.render.resolution_x, SC.render.resolution_y = 1600, 900
SC.render.film_transparent = False
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); SC.collection.objects.link(cam); SC.camera = cam
cam.data.clip_start = 0.1; cam.data.clip_end = 3000


def G(x, y, z):
    """Spielkoordinaten -> Blender."""
    return Vector((x, -z, y))


def shot(name, loc, tgt, lens=24, shift_y=0.0, shift_x=0.0, roll=0.0, keep_level=False):
    if SHOTS and name not in SHOTS: return
    if not SHOTS and name.startswith('chk_'): return
    cam.location = Vector(loc); d = Vector(tgt) - cam.location
    if keep_level:
        d.z = 0
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = lens; cam.data.shift_y = shift_y; cam.data.shift_x = shift_x
    SC.render.filepath = os.path.join(OUTDIR, name + '.png')
    bpy.ops.render.render(write_still=True)
    print('render', name)


# Bodenplatte gegen den leeren Horizont
bpy.ops.mesh.primitive_plane_add(size=3000, location=(0, 0, -0.35))
gp = bpy.context.active_object; gm = bpy.data.materials.new('ground_far'); gm.use_nodes = True
next(n for n in gm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = (0.28, 0.28, 0.27, 1)
gp.data.materials.append(gm)
_geo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'kc_geo.json')
CAMS = json.load(open(_geo)).get('cams', {}) if os.path.exists(_geo) else {}
# Referenzfoto (Commons 2026, Blick von Norden auf den Bug, Karmarschstrasse rechts)
shot('lm_center_ref', (-28.3, 9.3, 1.65), (-22.6, -60.0, 1.65), lens=19, shift_y=0.19, keep_level=True)
# von der Uhr (Spiel 0,1.7,6) nach Sueden
shot('lm_center_clock', tuple(G(0, 1.7, 6)), (-6.0, -60.0, 9.0), lens=22)
# Luftbild schraeg von Nordosten
shot('lm_center_aerial', (55.0, 30.0, 85.0), (-22.0, -72.0, 8.0), lens=30)
# Nahaufnahme gerundeter Bug
shot('lm_center_corner', (-16.5, -6.5, 2.0), (-25.0, -16.5, 8.5), lens=22)
# Ladeneingang + Kontrollansichten (Kameras aus kc_geom: Fassadenkoordinaten)
for nm, c in CAMS.items():
    shot(nm, tuple(c['loc']), tuple(c['tgt']), lens=c['lens'])
# zweites Referenzfoto (Commons 2014, von Nordwesten, Karmarschstrasse rechts mit Bannern)
shot('lm_center_ref2', (-34.0, 18.0, 1.7), (-27.0, -40.0, 1.7), lens=24, shift_y=0.30, keep_level=True)
# Dach
shot('lm_center_roof', (-60.0, -150.0, 70.0), (-20.0, -75.0, 20.0), lens=28)

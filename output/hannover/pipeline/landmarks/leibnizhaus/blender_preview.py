"""Vorschau-Renderings (Eevee 1280x720, Backface-Culling AN) fuer das Landmark leibnizhaus, optional im Kontext
(Boden aus data/ground_raw, Gebaeude aus web/assets/city/buildings bzw. data/buildings_raw).
Aufruf: blender -b --factory-startup -P blender_preview.py -- <glb> <out_prefix> <shot,shot|all> [--ctx] [--ctxblend=<p>] [--blend=<p>]
Kameras im Spielframe (x Ost, y oben, z Sued)."""
import bpy, sys, os, math, glob
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUTP, SHOTS = os.path.abspath(argv[0]), os.path.abspath(argv[1]), argv[2]
CTX = '--ctx' in argv
CTXBLEND = next((os.path.abspath(a.split('=', 1)[1]) for a in argv if a.startswith('--ctxblend=')), None)
BLEND = next((os.path.abspath(a.split('=', 1)[1]) for a in argv if a.startswith('--blend=')), None)
RES = next((a.split('=', 1)[1] for a in argv if a.startswith('--res=')), '1280x720')
RX, RY = [int(v) for v in RES.split('x')]
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/"
CX, CZ, CR = -475.0, 300.0, 150.0


def tiles(folder, pat):
    out = []
    for fn in glob.glob(os.path.join(ROOT, folder, pat)):
        b = os.path.basename(fn).split('.')[0].split('_')
        if fn.endswith('_win.glb'): continue
        try: tx, tz = int(b[1]), int(b[2])
        except Exception: continue
        x0, z0 = tx * 100.0, tz * 100.0
        dx = max(x0 - CX, 0, CX - (x0 + 100)); dz = max(z0 - CZ, 0, CZ - (z0 + 100))
        if math.hypot(dx, dz) < CR: out.append(fn)
    return sorted(out)


if CTX and CTXBLEND and os.path.exists(CTXBLEND):
    bpy.ops.wm.open_mainfile(filepath=CTXBLEND)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if CTX:
        for fn in tiles('data/ground_raw', 'ground_*.glb'):
            try: bpy.ops.import_scene.gltf(filepath=fn)
            except Exception as e: print('ctx import failed', fn, e)
        raw = {os.path.basename(f): f for f in tiles('pipeline/facades/cache/full_raw', 'bldg_*.glb')}
        raw.update({os.path.basename(f): f for f in tiles('data/buildings_raw', 'bldg_*.glb')})
        for k, fn in sorted(raw.items()):
            b0 = set(bpy.data.objects)
            try: bpy.ops.import_scene.gltf(filepath=fn)
            except Exception as e: print('ctx bldg import failed', fn, e); continue
            for o in set(bpy.data.objects) - b0: o['ctx_bldg'] = 1
        # durch das Landmark ersetzte Altgebaeude ausschneiden (wie apply_landmark_replace)
        import json, bmesh
        J = json.load(open(ROOT + 'data/landmarks_replace/leibnizhaus.json', encoding='utf-8'))
        FPS = J['footprints']
        def inside(poly, x, z):
            c = False; n = len(poly); j = n - 1
            for i in range(n):
                xi, zi = poly[i]; xj, zj = poly[j]
                if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi: c = not c
                j = i
            return c
        BX = [(min(p[0] for p in f), min(p[1] for p in f), max(p[0] for p in f), max(p[1] for p in f)) for f in FPS]
        for o in [o for o in bpy.data.objects if o.type == 'MESH' and o.get('ctx_bldg')]:
            bm = bmesh.new(); bm.from_mesh(o.data); mw = o.matrix_world; kill = []
            for f in bm.faces:
                c = mw @ f.calc_center_median(); gx, gz = c.x, -c.y
                for fp, (x0, z0, x1, z1) in zip(FPS, BX):
                    if x0 <= gx <= x1 and z0 <= gz <= z1 and inside(fp, gx, gz): kill.append(f); break
            if kill:
                bmesh.ops.delete(bm, geom=kill, context='FACES'); bm.to_mesh(o.data)
            bm.free()
        for o in bpy.data.objects:
            if o.name.startswith('col_') or 'collision' in o.name.lower(): o.hide_render = True
        if CTXBLEND:
            bpy.ops.wm.save_as_mainfile(filepath=CTXBLEND); print('saved ctx', CTXBLEND)
sc = bpy.context.scene
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=GLB)
new = [o for o in bpy.data.objects if o not in before]
# Kontext-Gebaeude, die durch das Landmark ersetzt werden, ausblenden (Fassaden-Pipeline hat sie evtl. noch)
for m in bpy.data.materials:
    try: m.use_backface_culling = True
    except Exception: pass
if BLEND:
    # nur das Landmark in eine eigene .blend schreiben
    bpy.ops.wm.save_as_mainfile(filepath=BLEND, copy=True)

w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = nt.nodes['Background']
sky = nt.nodes.new('ShaderNodeTexSky')
for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'SINGLE_SCATTERING', 'HOSEK_WILKIE'):
    try: sky.sky_type = t; break
    except Exception: pass
SUN_EL, SUN_AZ = math.radians(36), math.radians(140)
try: sky.sun_elevation = SUN_EL; sky.sun_rotation = SUN_AZ; sky.altitude = 60
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


def Gv(x, y, z): return Vector((x, -z, y))


SH = {}
def shot(name, loc, tgt, lens=24, ortho=None, res=None):
    SH[name] = (loc, tgt, lens, ortho, res)


# Leibnizhaus / Holzmarkt (Referenzfoto-Standpunkte)
shot('lh_front', (-418.0, 1.6, 337.0), (-439.0, 10.5, 335.3), lens=22)            # leibnizhaus/01, 05
shot('lh_front_far', (-405.0, 1.7, 338.0), (-440.0, 11.0, 333.0), lens=24)        # leibnizhaus/07 (Abend, weit)
shot('lh_noltehaus', (-421.0, 1.6, 326.5), (-442.0, 11.0, 330.0), lens=22)        # holzmarkt/10
shot('lh_oriel', (-432.5, 1.6, 332.0), (-438.6, 1.6, 333.5), lens=24)             # leibnizhaus/02 (Utlucht EG)
shot('lh_gable', (-424.0, 15.0, 335.5), (-439.0, 19.5, 335.3), lens=30)           # Giebel nah
shot('lh_south', (-423.7, 1.6, 357.8), (-440.0, 11.0, 333.0), lens=24)            # leibnizhaus/09
shot('lh_portal', (-433.0, 1.6, 337.0), (-438.8, 0.5, 336.6), lens=24)
shot('lh_aerial', (-405.0, 45.0, 360.0), (-445.0, 12.0, 333.0), lens=28)
shot('lh_aerial_nw', (-470.0, 40.0, 300.0), (-445.0, 12.0, 335.0), lens=28)
# Brunnen
shot('fo_front', (-424.6, 1.6, 333.0), (-432.0, 3.6, 333.0), lens=24)             # oskar_winter_brunnen/01
shot('fo_ne', (-424.0, 1.6, 324.0), (-432.0, 3.5, 333.0), lens=24)                # 04 / 08
shot('fo_up', (-433.6, 1.4, 330.0), (-431.8, 7.5, 332.0), lens=20)                # leibnizhaus/04 (Blick in die Krone)
# Beginenturm / Museum
shot('tw_south', (-510.0, 1.6, 322.0), (-509.5, 11.0, 303.0), lens=22)            # beginenturm/01
shot('tw_river', (-567.0, -1.5, 319.7), (-505.0, 9.0, 292.0), lens=30)            # beginenturm/08
shot('tw_zeughaus', (-531.0, 1.6, 255.0), (-512.0, 8.0, 290.0), lens=24)          # beginenturm/03
shot('mu_corner', (-417.2, 1.7, 321.0), (-447.0, 5.5, 300.0), lens=22)            # historisches_museum/07
shot('mu_burg', (-436.0, 1.6, 236.0), (-446.0, 8.0, 285.0), lens=22)              # historisches_museum/09
shot('mu_pferde', (-470.0, 1.6, 316.0), (-505.0, 6.0, 304.0), lens=22)            # Pferdestrasse Richtung Turm
shot('mu_kiosk', (-427.0, 1.6, 316.0), (-441.0, 2.5, 304.0), lens=20)             # historisches_museum/05
shot('mu_north', (-500.0, 1.6, 228.0), (-490.0, 6.0, 250.0), lens=22)
shot('aerial_all', (-400.0, 90.0, 380.0), (-475.0, 5.0, 295.0), lens=28)
shot('aerial_w', (-590.0, 70.0, 330.0), (-480.0, 5.0, 290.0), lens=28)
shot('top', (-475.0, 150.0, 295.0), (-475.0, 0.0, 295.01), ortho=130)

names = list(SH) if SHOTS == 'all' else SHOTS.split(',')
sc.render.resolution_x = RX; sc.render.resolution_y = RY; sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
for n in names:
    if n not in SH: print('unknown shot', n); continue
    loc, tgt, lens, ortho, res = SH[n]
    cam.location = Gv(*loc)
    dirv = Gv(*tgt) - Gv(*loc)
    cam.rotation_euler = dirv.to_track_quat('-Z', 'Y').to_euler()
    if ortho:
        cam.data.type = 'ORTHO'; cam.data.ortho_scale = ortho
    else:
        cam.data.type = 'PERSP'; cam.data.lens = lens
    sc.render.filepath = OUTP + n + '.png'
    bpy.ops.render.render(write_still=True)
    print('rendered', sc.render.filepath)

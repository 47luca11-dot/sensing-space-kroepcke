"""Vorschau-Renderings (Eevee, 1280x720, Backface-Culling AN) des Ballhof-Ensembles, optional im Kontext
(Boden-/Gebaeudekacheln aus data/ground_raw, data/buildings_raw; ersetzte LoD2-Gebaeude werden aus den Kacheln entfernt).
Aufruf: blender -b --factory-startup -P bh_preview.py -- <ballhof.glb> <out_prefix> <shots|all> [--ctx] [--blend=<pfad>] [--res=WxH]
Kameras im Spielframe (x Ost, y oben, z Sued)."""
import bpy, bmesh, sys, os, math, glob, json
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUTP, SHOTS = argv[0], argv[1], argv[2]
CTX = '--ctx' in argv
BLEND = next((a.split('=', 1)[1] for a in argv if a.startswith('--blend=')), None)
RES = next((a.split('=', 1)[1] for a in argv if a.startswith('--res=')), '1280x720')
CTXR = float(next((a.split('=', 1)[1] for a in argv if a.startswith('--ctxr=')), '140'))
RX, RY = [int(v) for v in RES.split('x')]
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/"
CX, CZ = -425.0, 225.0

bpy.ops.wm.read_factory_settings(use_empty=True)
REP = json.load(open(ROOT + 'data/landmarks_replace/ballhof.json'))['footprints']


def inside(poly, x, z):
    c = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, zi = poly[i]; xj, zj = poly[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi: c = not c
        j = i
    return c


ctx_objs = []
if CTX:
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
            if math.hypot(dx, dz) < CTXR: out.append(fn)
        return sorted(out)
    for fn in tiles('data/ground_raw', 'ground_*.glb') + tiles('data/buildings_raw', 'bldg_*.glb'):
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.gltf(filepath=fn)
        except Exception as e:
            print('ctx import failed', fn, e); continue
        new = [o for o in set(bpy.data.objects) - before]
        ctx_objs += new
        if 'bldg_' in os.path.basename(fn):
            for o in new:
                if o.type != 'MESH': continue
                mw = o.matrix_world
                bm = bmesh.new(); bm.from_mesh(o.data)
                kill = []
                for f in bm.faces:
                    c = mw @ f.calc_center_median()
                    gx, gz = c.x, -c.y
                    if -500 < gx < -360 and 170 < gz < 270 and any(inside(fp, gx, gz) for fp in REP):
                        kill.append(f)
                if kill:
                    bmesh.ops.delete(bm, geom=kill, context='FACES'); bm.to_mesh(o.data)
                bm.free()
    for o in bpy.data.objects:
        if o.name.lower().startswith('col_') or 'collision' in o.name.lower(): o.hide_render = True
sc = bpy.context.scene
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=GLB)
new = set(bpy.data.objects) - before
for o in new:
    if o.name.startswith('col_'): o.hide_render = True; o.hide_viewport = True
for m in bpy.data.materials:
    try: m.use_backface_culling = True
    except Exception: pass

w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
nt = w.node_tree; bg = nt.nodes['Background']
sky = nt.nodes.new('ShaderNodeTexSky')
for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'SINGLE_SCATTERING', 'HOSEK_WILKIE'):
    try: sky.sky_type = t; break
    except Exception: pass
SUN_EL, SUN_AZ = math.radians(36), math.radians(200)
try:
    sky.sun_elevation = SUN_EL; sky.sun_rotation = SUN_AZ; sky.altitude = 60
except Exception: pass
nt.links.new(sky.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.28
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.2; sun.angle = math.radians(0.6); sun.color = (1.0, 0.96, 0.9)
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


# Referenzfoto-Standpunkte (refs.json approx_game_xz + Bildinhalt)
shot('platz_sued', (-430.0, 1.7, 203.0), (-402.0, 6.5, 232.0), lens=20)          # Foto 08 (Ballhof Nordseite + Pavillon)
shot('marktkirche_blick', (-423.1, 1.7, 210.3), (-392.0, 8.0, 222.0), lens=24)   # Foto 01/03 (Ostfluegel + Ballhof)
shot('brunnen', (-418.0, 1.7, 214.0), (-396.0, 5.5, 226.0), lens=24)            # flickr 02
shot('ballhof_tor', (-404.0, 1.7, 219.0), (-401.0, 2.8, 228.6), lens=24)          # flickr 01 (Tor + Laternen)
shot('ostfluegel', (-414.0, 1.7, 208.0), (-397.5, 4.0, 203.0), lens=24)           # Foto 13
shot('hofgefluester', (-480.0, 1.7, 240.0), (-462.0, 8.0, 217.0), lens=22)       # Foto 14 / 02
shot('burgstr_nord', (-452.0, 1.7, 266.0), (-438.0, 8.5, 246.0), lens=22)        # Foto 11 (Spittahaus)
shot('spittahaus', (-442.5, 1.7, 230.5), (-441.5, 3.5, 241.5), lens=22)          # Foto 04
shot('teestuebchen', (-433.0, 1.7, 222.0), (-443.0, 5.5, 211.0), lens=22)       # Foto 06/07
shot('burgstr_platz', (-478.0, 1.7, 230.0), (-420.0, 6.0, 222.0), lens=24)       # Foto 10
shot('pavillon', (-424.0, 1.7, 226.0), (-424.5, 2.6, 239.0), lens=24)
shot('bp3', (-418.0, 1.7, 212.0), (-430.0, 6.0, 197.0), lens=22)
shot('aerial_n', (-420.0, 70.0, 135.0), (-420.0, 5.0, 222.0), lens=30)
shot('aerial_sw', (-500.0, 55.0, 300.0), (-420.0, 5.0, 222.0), lens=30)
shot('aerial_se', (-345.0, 55.0, 285.0), (-415.0, 5.0, 222.0), lens=30)
shot('top', (-424.0, 140.0, 222.0), (-424.0, 0.0, 222.01), ortho=110)
shot('knaggen', (-406.0, 2.2, 224.0), (-402.0, 4.3, 229.5), lens=30)
shot('dach_nah', (-403.0, 15.0, 216.0), (-397.0, 11.0, 228.0), lens=30)
shot('gauben_ost', (-412.0, 9.0, 204.0), (-400.0, 8.0, 203.0), lens=30)

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
    for o in list(bpy.data.objects):
        if o in ctx_objs:
            bpy.data.objects.remove(o, do_unlink=True)
    try: bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception: pass
    os.makedirs(os.path.dirname(BLEND), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND + '.tmp.blend', compress=True)
    os.replace(BLEND + '.tmp.blend', BLEND)
    print('saved blend', BLEND)

"""Entfernt im Detailmodell (data/detail_raw) die Geometrie alter Gebaeude, die durch neu modellierte Wahrzeichen
ersetzt werden (data/landmarks_replace/*.json: footprints [[x,z],...] im Spielframe, ymin/ymax).
Schreibt data/detail_final/<datei>.glb (unkomprimiert, Spielframe). Danach meshopt -> web/assets/city/detail/.
Aufruf: blender -b --factory-startup -P apply_landmark_replace.py"""
import bpy, bmesh, glob, json, os
from mathutils import Vector

R = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/"
SRC, DST = R + 'data/detail_raw/', R + 'data/detail_final/'
os.makedirs(DST, exist_ok=True)
# nur Gebaeudeteile filtern -- Boden, Strassen, Moebel, Baeume bleiben unangetastet
BUILDING_FILES = {'bldg_walls', 'bldg_openings', 'bldg_roof', 'bldg_base', 'd0_windows', 'fix', 'kiosk'}
REP = []
for fn in sorted(glob.glob(R + 'data/landmarks_replace/*.json') + glob.glob(R + 'data/detail_replace/*.json')):
    J = json.load(open(fn, encoding='utf-8'))
    files = set(J.get('files') or [])                     # optional: welche Detail-Dateien gefiltert werden (Standard: Gebaeude)
    for fp in J['footprints']:
        REP.append((fp, float(J.get('ymin', -2)), float(J.get('ymax', 60)), os.path.basename(fn), files))
print('replace polygons', len(REP))


def inside(poly, x, z):
    c = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, zi = poly[i]; xj, zj = poly[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi: c = not c
        j = i
    return c


BOX = [(min(p[0] for p in fp), min(p[1] for p in fp), max(p[0] for p in fp), max(p[1] for p in fp)) for fp, *_ in REP]
ALLF = BUILDING_FILES | set().union(*[r[4] for r in REP]) if REP else BUILDING_FILES
stats = {}
for fn in sorted(glob.glob(SRC + '*.glb')):
    name = os.path.splitext(os.path.basename(fn))[0]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=fn)
    removed = 0
    if name in ALLF and REP:
        for o in [o for o in bpy.data.objects if o.type == 'MESH']:
            mw = o.matrix_world
            bm = bmesh.new(); bm.from_mesh(o.data)
            kill = []
            for f in bm.faces:
                c = mw @ f.calc_center_median()          # Blender: x Ost, y Nord, z oben -> Spiel x, y=z, z=-y
                gx, gy, gz = c.x, c.z, -c.y
                for (fp, y0, y1, _, fs), (bx0, bz0, bx1, bz1) in zip(REP, BOX):
                    if (name in fs if fs else name in BUILDING_FILES) and bx0 - 0.3 <= gx <= bx1 + 0.3 and bz0 - 0.3 <= gz <= bz1 + 0.3 and y0 <= gy <= y1 and inside(fp, gx, gz):
                        kill.append(f); break
            if kill:
                removed += len(kill)
                bmesh.ops.delete(bm, geom=kill, context='FACES')
                bm.to_mesh(o.data)
            bm.free()
        for o in [o for o in bpy.data.objects if o.type == 'MESH' and len(o.data.polygons) == 0]:
            bpy.data.objects.remove(o)
    stats[name] = removed
    bpy.ops.export_scene.gltf(filepath=DST + name + '.glb', export_format='GLB', export_yup=True, export_apply=True)
    print('REPLACE', name, 'removed faces', removed)
json.dump(stats, open(DST + 'replace_stats.json', 'w'), indent=1)

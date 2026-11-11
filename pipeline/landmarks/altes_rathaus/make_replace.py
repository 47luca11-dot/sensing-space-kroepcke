"""Ersatzbereich fuer das Alte Rathaus -> data/landmarks_replace/altes_rathaus.json (Spielframe [x, z], z = -Nord).
Umriss = Grundflaeche der city-LoD2 ID_dedae90c (Schmiedestrasse 1 = ganzer Rathausblock: Marktfluegel, Schmiedestrassen-
fluegel, Karmarschstrassenfluegel, Gerichtsfluegel/Standesamt + Atrium) + Puffer.
Aufruf: python make_replace.py [out.json] [buffer_m]"""
import json, sys, os
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry.polygon import orient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
OUT = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/data/landmarks_replace/altes_rathaus.json"
BUF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
IDS = ["ID_dedae90c-48f4-3b9e-9176-f04541ffbcb2"]
B = [b for b in json.load(open(ROOT + "/data/lod2_buildings.json"))['buildings'] if b['id'] in IDS]
polys = []
for b in B:
    for g in b['grounds']:
        polys.append(Polygon([(p[0], p[1]) for p in g['ext']]).buffer(0))
fp = unary_union(polys).buffer(BUF, join_style=2, mitre_limit=2.5).simplify(0.05)
geoms = list(fp.geoms) if isinstance(fp, MultiPolygon) else [fp]
fps = []
for g in geoms:
    g = orient(g, 1.0)
    fps.append([[round(x, 3), round(-y, 3)] for x, y in list(g.exterior.coords)[:-1]])
top = max(p[2] for b in B for w in b['walls'] + b['roofs'] for p in w['ext']) - 56.16
mk = os.path.join(HERE, 'build', 'model_top.json')
if os.path.exists(mk):
    top = max(top, json.load(open(mk))['top'])
doc = dict(footprints=fps, ymin=-2.0, ymax=round(top + 5.0, 1), lod2_ids=IDS,
           note=('Altes Rathaus Hannover (Karmarschstrasse 42 / Am Markte / Schmiedestrasse / Koebelinger Strasse): ersetzt durch '
                 'landmarks/altes_rathaus.glb (handmodelliert, pipeline/landmarks/altes_rathaus). Grundriss = city-LoD2 %s '
                 '(ganzer Block inkl. Gerichtsfluegel/Standesamt und Atrium) + %.2f m Puffer (Sockel, Fialen, Lauben, Treppen). '
                 'Die Fassaden-Pipeline soll diese LoD2-ID ueberspringen; das Rhino-Detailmodell deckt den Block nicht ab '
                 '(detail_mask = False). Pflaster/Boden nicht filtern.') % (IDS[0], BUF))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp'
json.dump(doc, open(tmp, 'w'), indent=1)
os.replace(tmp, OUT)
print('wrote', OUT, [len(f) for f in fps], 'area %.1f' % fp.area, 'ymax', doc['ymax'])

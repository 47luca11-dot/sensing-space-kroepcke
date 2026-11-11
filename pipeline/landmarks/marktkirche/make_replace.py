"""Ersatzbereich fuer die Marktkirche -> data/landmarks_replace/marktkirche.json (Spielframe [x, z], z = -Nord).
Umriss = Grundflaechen der city-LoD2 ID_eac4b0e4 (Marktkirche, Hanns-Lilje-Platz 11) vereinigt + Puffer.
Aufruf: python make_replace.py [out.json] [buffer_m]"""
import json, sys, os
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
OUT = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/data/landmarks_replace/marktkirche.json"
BUF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
IDS = ["ID_eac4b0e4-bb4e-3389-a0e4-73efb1359670"]
B = [b for b in json.load(open(os.path.join(HERE, 'mk_lod2.json'))) if b['id'] in IDS]
polys = []
for b in B:
    for g in b['grounds']:
        polys.append(Polygon([p[:2] for p in g['ext']]).buffer(0))
fp = unary_union(polys).buffer(BUF, join_style=2, mitre_limit=2.5).simplify(0.05)
geoms = list(fp.geoms) if isinstance(fp, MultiPolygon) else [fp]
fps = []
for g in geoms:
    g = orient(g, 1.0)
    fps.append([[round(x, 3), round(-y, 3)] for x, y in list(g.exterior.coords)[:-1]])
top = max(p[2] for b in B for w in b['walls'] + b['roofs'] for p in w['ext']) - 56.16
extra = {}
mk = os.path.join(HERE, 'build', 'model_top.json')
if os.path.exists(mk):
    top = max(top, json.load(open(mk))['top'])
doc = dict(footprints=fps, ymin=-2.0, ymax=round(top + 5.0, 1), lod2_ids=IDS,
           note=('Marktkirche St. Georgii et Jacobi (Hanns-Lilje-Platz 11): ersetzt durch landmarks/marktkirche.glb (handmodelliert, '
                 'pipeline/landmarks/marktkirche). Grundriss = city-LoD2 %s + %.2f m Puffer (Strebepfeiler, Sockel, Portale, Treppen). '
                 'Die Fassaden-Pipeline soll diese LoD2-IDs ueberspringen; im Rhino-Detailmodell liegt die Kirche nicht (detail_mask = False). '
                 'Pflaster/Boden nicht filtern. Das Lutherdenkmal (Suedseite Turm, Hanns-Lilje-Platz, ca. x -258.3 z 311.1, Sockel 1.4 x 1.2 m) '
                 'und die Portalstufen/Vorplatzplatten (bis 2.1 m vor den Seitenportalen; am Westportal liegen die 6 Stufen in der 4 m tiefen Vorhalle, Figurenkonsolen bis 1.2 m vor der Front) sowie der Schaukasten auf 2 Pfosten vor der Turm-Westseite noerdlich des Portals sind Teil des Modells '
                 'und liegen teils ausserhalb des Umrisses: dort bitte keine Moeblierung platzieren.') % (IDS[0], BUF))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp'
json.dump(doc, open(tmp, 'w'), indent=1); os.replace(tmp, OUT)
print('wrote', OUT, [len(f) for f in fps], 'area %.1f' % fp.area, 'ymax', doc['ymax'])

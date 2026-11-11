"""Ersatzbereich fuer das Ballhof-Ensemble -> data/landmarks_replace/ballhof.json (Spielframe [x, z], z = -Nord).
Umrisse = Grundflaechen der ersetzten city-LoD2-Gebaeude + Puffer (Auskragungen, Traufen, Stufen).
Aufruf: python make_replace.py [out.json] [buffer_m]"""
import json, sys, os
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry.polygon import orient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
OUT = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/data/landmarks_replace/ballhof.json"
BUF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
PRE = ['ID_c48b3643',   # Ballhofplatz 5 - Ballhof eins (Ballhaus 1649-64)
       'ID_98a9691d',   # Foyer-Pavillon "Ballhof eins / junges schauspiel" + niedrige Anbauten
       'ID_bcc9a17d',   # Ostfluegel (Ballhof, Fachwerk-OG mit Fussstreben, Gauben) + niedriger Zwischenbau
       'ID_ff0b0763',   # Hinterhaus/Buehnenanbau hinter dem Ostfluegel
       'ID_7bf3782c',   # kleiner Vorbau an der Nordspitze des Ostfluegels
       'ID_b150f6c8',   # Ballhofplatz 7 / Burgstrasse 23 - Spittahaus
       'ID_84e270ab',   # Burgstrasse 21A / Ballhofplatz 1 - [hof:gefluester]
       'ID_1a4a4aa4',   # Ballhofplatz 2 - Teestuebchen (Efeu)
       'ID_0875e4bf']   # Ballhofplatz 3
L = json.load(open(ROOT + "/data/lod2_buildings.json", encoding='utf8'))['buildings']
B = [b for b in L if any(b['id'].startswith(p) for p in PRE)]
assert len(B) == len(PRE), [b['id'] for b in B]
IDS = [b['id'] for b in B]
polys = [Polygon([(p[0], -p[1]) for p in g['ext']]).buffer(0) for b in B for g in b['grounds']]
fp = unary_union(polys).buffer(BUF, join_style=2, mitre_limit=2.5).buffer(-0.05).buffer(0.05).simplify(0.05)
geoms = list(fp.geoms) if isinstance(fp, MultiPolygon) else [fp]
fps = []
for g in geoms:
    g = orient(Polygon(g.exterior), 1.0)
    fps.append([[round(x, 3), round(z, 3)] for x, z in list(g.exterior.coords)[:-1]])
top = max(p[2] for b in B for r in b['roofs'] for p in r['ext']) - 56.16
mk = os.path.join(HERE, 'build', 'model_top.json')
if os.path.exists(mk):
    top = max(top, json.load(open(mk))['top'])
# Nachbar-Schwerpunkte duerfen nicht im Umriss liegen (facades/replaced.py ersetzt auch per Schwerpunkt)
from shapely.geometry import Point
U = unary_union([Polygon(f) for f in fps])
hit = [b['id'] for b in L if b['id'] not in IDS and U.contains(Point(b['centre'][0], -b['centre'][1]))]
assert not hit, ('Nachbarn im Umriss', hit)
doc = dict(footprints=fps, ymin=-2.0, ymax=round(top + 5.0, 1), lod2_ids=IDS,
           note=('Ballhof-Ensemble am Ballhofplatz (Ballhof eins = Ballhaus 1649-64 mit Foyer-Pavillon und Ostfluegel, '
                 'Spittahaus Ballhofplatz 7/Burgstrasse 23, Ballhofplatz 1 (Burgstrasse 21A), Ballhofplatz 2 (Teestuebchen), Ballhofplatz 3): '
                 'ersetzt durch landmarks/ballhof.glb (handmodelliert, pipeline/landmarks/ballhof). Umriss = city-LoD2-Grundflaechen + %.2f m Puffer. '
                 'Die Fassaden-Pipeline soll diese LoD2-IDs ueberspringen; das Rhino-Detailmodell deckt den Platz nicht ab. Boden nicht filtern.') % BUF)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + '.tmp'
json.dump(doc, open(tmp, 'w'), indent=1)
os.replace(tmp, OUT)
print('wrote', OUT, [len(f) for f in fps], 'area %.1f' % U.area, 'ymax', doc['ymax'], 'ids', len(IDS))

"""Ersetzungs-Grundriss fuer das alte Kroepcke-Center im Detailmodell (data/detail_raw/bldg_*.glb).
LoD2-Grundflaeche (inkl. Vordach-Anbauten R5/R6) + 0.6 m Puffer (alte Fensterbaenke/Abdeckungen ragen bis 0.3 m vor);
Nachbargebaeude liegen >= 8 m entfernt. Ausgabe im Spielframe: [x, z] mit z = -Nord.
Aufruf: python kc_replace.py <out.json>"""
import json, sys, os
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, 'kc_lod2.json')))
g = Polygon([p[:2] for p in B['grounds'][0]['ext']]).buffer(0)
fp = orient(g.buffer(0.6, join_style=2, mitre_limit=2.0).simplify(0.05), 1.0)
pts = [[round(x, 3), round(-y, 3)] for x, y in list(fp.exterior.coords)[:-1]]
out = dict(footprints=[pts], ymin=-1.0, ymax=40.0,
           note='Kroepcke-Center (Karmarschstr. 19-23 / Georgstr. / Staendehausstr.): ersetzt durch landmarks/kroepcke_center.glb. '
                'Entfernen: alle Dreiecke aus data/detail_raw/bldg_walls, bldg_openings, bldg_roof, bldg_base, deren Schwerpunkt '
                'innerhalb dieses Polygons (Spiel x,z) und zwischen ymin und ymax liegt (LoD2-Grundriss + 0.6 m; alte Geometrie '
                'reicht max. 0.3 m ueber die LoD2-Linie, Nachbarn >= 8 m entfernt). street/fix/ground nicht anfassen: die '
                'Pflasterflaechen unter dem Gebaeude (y -0.13..-0.28) liegen unter dem neuen EG-Boden (-0.10).')
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print('replace footprint', len(pts), 'pts, area %.1f' % fp.area)

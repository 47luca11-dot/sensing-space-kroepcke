"""Schreibt data/landmarks_replace/leibnizhaus.json (atomar).
Grundrisse = LoD2-Grundflaechen der ersetzten Gebaeude (Spielrahmen x,z) + Puffer fuer Gesimse/Erker/Treppen.
Aufruf: python make_replace.py [ymax]"""
import json, os, sys
import numpy as np
from shapely.geometry import Polygon
import lh_common as C

ids = C.full_ids()
ymax = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
BUF = {'leibnizhaus': 1.2, 'noltehaus': 0.8, 'museum': 0.8, 'beginenturm': 0.8}
fps = []
for k, bid in ids.items():
    p = Polygon(C.lod2_footprint(bid)).buffer(0)
    p = p.buffer(BUF[k], join_style=2, mitre_limit=2.0).simplify(0.05)
    ring = np.asarray(p.exterior.coords)[:-1]
    fps.append([[round(float(x), 3), round(float(z), 3)] for x, z in ring])
out = {
    'footprints': fps,
    'ymin': -2.0,
    'ymax': ymax,
    'lod2_ids': list(ids.values()),
    'replaces_ground_features': ['Holzmarktbrunnen (Oskar-Winter-Brunnen) aus pipeline/ground/altstadt.py holzmarkt_brunnen() - '
                                 'ist im Landmark leibnizhaus.glb enthalten und sollte im Boden-Build entfallen'],
    'note': ('Leibnizhaus (Holzmarkt 5, rekonstruierte Renaissance-Sandsteinfassade mit Utlucht und Volutengiebel), Noltehaus (Holzmarkt 4, '
             'Neorenaissance Gelbklinker + Sandstein, Ecke Pferdestrasse), Historisches Museum (Pferdestrasse 6 / Burgstrasse / Am Hohen Ufer, '
             '1966 Oesterlen, Bruchstein + Sichtbeton, inkl. Zeughauswand) und Beginenturm (Pferdestrasse 8) sowie der Oskar-Winter-Brunnen '
             'werden durch landmarks/leibnizhaus.glb ersetzt (handmodelliert, pipeline/landmarks/leibnizhaus). Grundrisse = city-LoD2 + '
             '0.8-1.2 m Puffer. Ausserhalb des Rhino-Detailquadrats (x < -400) - detail_mask hier False; Boden/Pflaster nicht filtern.'),
}
dst = os.path.join(C.DATA, 'landmarks_replace', 'leibnizhaus.json')
tmp = dst + '.tmp'
json.dump(out, open(tmp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
os.replace(tmp, dst)
print('wrote', dst, [len(f) for f in fps], ids)

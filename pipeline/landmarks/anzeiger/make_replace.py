"""Write data/landmarks_replace/anzeiger.json (atomic): footprints of the replaced LoD2 buildings
(Anzeiger-Hochhaus Goseriede 9 + Goseriedebad/Kestnergesellschaft Goseriede 11) in game frame [x,z]."""
import json, os, sys, numpy as np
from shapely.geometry import Polygon
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
S = json.load(open(os.path.join(HERE, 'site_lod2.json')))
ymax = float(sys.argv[1]) if len(sys.argv) > 1 else 55.0
fps, ids = [], []
for k in ('anz', 'bad'):
    g = np.array(S[k]['grounds'][0])[:, [0, 2]]
    p = Polygon(g).buffer(0.25, join_style=2).simplify(0.05)     # 25 cm Rand: alte Wandflaechen sicher erfasst
    fps.append([[round(x, 3), round(z, 3)] for x, z in np.array(p.exterior.coords)[:-1]])
    ids.append(S[k]['id'])
out = {'footprints': fps, 'ymin': -2.0, 'ymax': round(ymax, 2), 'lod2_ids': ids,
       'note': 'Landmark anzeiger: Anzeiger-Hochhaus (Goseriede 9, Fritz Hoeger 1928, Kuppel = Hochhaus-Lichtspiele) + '
               'Goseriedebad/Kestnergesellschaft (Goseriede 11) komplett handmodelliert; beide LoD2-Gebaeude ersetzt. '
               'Footprints = LoD2-Grundrisse + 0.25 m.'}
dst = ROOT + 'data/landmarks_replace/anzeiger.json'
tmp = dst + '.tmp'
json.dump(out, open(tmp, 'w'), indent=1)
os.replace(tmp, dst)
print('wrote', dst, [len(f) for f in fps], ids, 'ymax', ymax)

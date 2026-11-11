"""Dump city-LoD2 geometry of the Ballhof ensemble buildings (game frame x, y=up rel. 56.16, z=-N) -> build/site_lod2.json"""
import json, os, sys, numpy as np
from shapely.geometry import Polygon
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
H0 = 56.16
PRE = sys.argv[1:] or ['ID_c48b3643', 'ID_98a9691d', 'ID_bcc9a17d', 'ID_ff0b0763', 'ID_09cc9864', 'ID_7bf3782c', 'ID_b150f6c8', 'ID_84e270ab', 'ID_1a4a4aa4', 'ID_0875e4bf']
L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))
out = {}
for pre in PRE:
    b = [b for b in L['buildings'] if b['id'].startswith(pre)][0]
    cv = lambda r: [[round(p[0], 3), round(p[2] - H0, 3), round(-p[1], 3)] for p in r]
    out[pre] = dict(id=b['id'], street=b['street'], number=b['number'], height=b['height'], storeys=b.get('storeys'),
                    grounds=[cv(g['ext']) for g in b['grounds']], roofs=[cv(r['ext']) for r in b['roofs']], walls=[cv(w['ext']) for w in b['walls']])
    print('==', pre, b['street'], b['number'], 'h', b['height'], 'st', b.get('storeys'))
    for g in out[pre]['grounds']:
        a = np.array(g); print('  ground y %.2f..%.2f' % (a[:, 1].min(), a[:, 1].max()), 'n', len(g), 'A %.1f' % Polygon(a[:, [0, 2]]).area)
        print('    ', [(p[0], p[2]) for p in g])
    for r in out[pre]['roofs']:
        a = np.array(r, float)
        n = np.zeros(3)
        for i in range(len(a)):
            n += np.cross(a[i], a[(i + 1) % len(a)])
        n /= np.linalg.norm(n) + 1e-12
        slope = np.degrees(np.arccos(abs(n[1])))
        print('  roof y %.2f..%.2f slope %.1f n(%.2f,%.2f,%.2f) A %.1f pts %d' % (a[:, 1].min(), a[:, 1].max(), slope, n[0], n[1], n[2], Polygon(a[:, [0, 2]]).area if len(a) > 2 else 0, len(a)))
json.dump(out, open(os.path.join(HERE, 'build', 'site_lod2.json'), 'w'))

import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import prep as PREP, shops as SH
from common import CACHE
d = PREP.load_prep()
by = {b.id: b for b in d['bldgs']}
D = SH.db()
cams = {}
for arg in sys.argv[1:]:
    nm, suf, dist, side = arg.split(':') if arg.count(':') == 3 else (arg.split(':') + ['8', '0'])[:4]
    dist = float(dist); side = float(side)
    bid = [k for k in D['by_bldg'] if k.endswith(suf)][0]
    s = D['by_bldg'][bid][0]
    b = by[bid]
    wp = np.array(s['facade']['wall_point']); n = np.array(s['facade']['normal']); t = np.array([n[1], -n[0]])
    c = wp + n * dist + t * side
    cams['facades_test_' + nm] = {'loc': [float(c[0]), b.y_ground + 1.7, float(c[1])], 'tgt': [float(wp[0]), b.y_ground + 3.0, float(wp[1])], 'lens': 24, 'radius': 60}
json.dump(cams, open(os.path.join(CACHE, 'test_cams.json'), 'w'), indent=1)
print(cams)

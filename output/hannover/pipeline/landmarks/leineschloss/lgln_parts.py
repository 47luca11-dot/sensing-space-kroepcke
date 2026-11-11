"""Extract LGLN LoD2 building parts around a point (game frame x, y=Z-56.16, z=-N) -> json  (copied from marktkirche pipeline)"""
import sys, json, numpy as np
from lxml import etree
GML = sys.argv[1]; OUT = sys.argv[2]
E0, N0, H0 = 550298.88274216, 5802940.705715501, 56.16
CX, CZ, R = float(sys.argv[3]) if len(sys.argv) > 3 else 29.0, float(sys.argv[4]) if len(sys.argv) > 4 else 20.0, float(sys.argv[5]) if len(sys.argv) > 5 else 40.0
ns = {}
tree = etree.parse(GML); root = tree.getroot()
nsm = {k: v for k, v in root.nsmap.items() if k}
B = '{%s}' % nsm['bldg']; G = '{%s}' % nsm['gml']; GEN = '{%s}' % nsm.get('gen', '')
def poslist(el):
    a = np.array(el.text.split(), float).reshape(-1, 3)
    return [[round(x - E0, 3), round(z - H0, 3), round(-(y - N0), 3)] for x, y, z in a]
res = []
for bld in root.iter(B + 'Building'):
    parts = list(bld.iter(B + 'BuildingPart')) or [bld]
    # quick filter on any coordinate
    pl = bld.find('.//' + G + 'posList')
    if pl is None: continue
    a = np.array(pl.text.split(), float).reshape(-1, 3)
    if np.min(np.hypot(a[:, 0] - E0 - CX, -(a[:, 1] - N0) - CZ)) > R: continue
    bid = bld.get(G + 'id')
    for p in parts:
        info = {'building': bid, 'part': p.get(G + 'id')}
        for tag in ('roofType', 'function', 'measuredHeight', 'storeysAboveGround'):
            e = p.find(B + tag)
            if e is not None: info[tag] = e.text
        for s in p.iter(GEN + 'stringAttribute', GEN + 'doubleAttribute'):
            v = s.find(GEN + 'value')
            info[s.get('name')] = v.text if v is not None else None
        surf = {'roof': [], 'wall': [], 'ground': [], 'other': []}
        for bs in p.iter():
            if not isinstance(bs.tag, str): continue
            t = bs.tag.split('}')[-1]
            k = {'RoofSurface': 'roof', 'WallSurface': 'wall', 'GroundSurface': 'ground', 'ClosureSurface': 'other'}.get(t)
            if k is None: continue
            for poly in bs.iter(G + 'posList'):
                surf[k].append(poslist(poly))
        info['surf'] = surf
        res.append(info)
json.dump(res, open(OUT, 'w'))
for r in res:
    allp = [q for k in r['surf'] for poly in r['surf'][k] for q in poly]
    a = np.array(allp)
    print(r['building'][-8:], r['part'][-8:] if r['part'] else '', r.get('roofType'), r.get('function'), r.get('measuredHeight'),
          'x', a[:, 0].min().round(1), a[:, 0].max().round(1), 'z', a[:, 2].min().round(1), a[:, 2].max().round(1), 'y', a[:, 1].min().round(2), a[:, 1].max().round(2),
          {k: len(v) for k, v in r['surf'].items()})

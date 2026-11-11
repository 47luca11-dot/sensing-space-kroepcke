"""Ankerpunkte der Lichtwolke (4 Seile zu Fassaden): Strahlen ab Zentrum gegen LoD2-Grundrisse (Hoehe > Ankerhoehe)"""
import json, math, numpy as np
from shapely.geometry import Polygon, LineString, Point
B = json.load(open('data/lod2_buildings.json'))['buildings']
C = np.array([550270.75 - 550298.88274216, 5802943.85 - 5802940.705715501])   # dE, dN
H0 = 56.16; YC = 15.9; YA = 19.0
polys = []
for b in B:
    if math.hypot(b['centre'][0] - C[0], b['centre'][1] - C[1]) > 90: continue
    top = max(p[2] for r in b['roofs'] for p in r['ext']) if b['roofs'] else 0
    if top - H0 < YA + 1: continue
    for g in b['grounds']:
        try:
            pg = Polygon([p[:2] for p in g['ext']]).buffer(0)
            if pg.area > 4: polys.append(pg)
        except Exception: pass
res = []
for deg in range(0, 360, 5):
    d = np.array([math.sin(math.radians(deg)), math.cos(math.radians(deg))])
    ray = LineString([C, C + d * 80]); best = None
    for pg in polys:
        x = ray.intersection(pg.exterior)
        if x.is_empty: continue
        pts = [x] if x.geom_type == 'Point' else list(getattr(x, 'geoms', []))
        for p in pts:
            dd = Point(C).distance(p)
            if best is None or dd < best[0]: best = (dd, p.x, p.y)
    if best: res.append((deg, *best))
# 4 Richtungen moeglichst ueber Kreuz (ca. 90 Grad auseinander), kurze Distanzen bevorzugt
best = None
for off in range(0, 90, 5):
    pick = []
    for q in range(4):
        tgt = off + 90 * q
        cand = [r for r in res if min(abs(r[0] - tgt), 360 - abs(r[0] - tgt)) <= 25]
        if not cand: break
        pick.append(min(cand, key=lambda r: r[1] + 0.3 * min(abs(r[0] - tgt), 360 - abs(r[0] - tgt))))
    if len(pick) == 4:
        score = sum(p[1] for p in pick)
        if best is None or score < best[0]: best = (score, pick)
anch = [[round(p[2], 2), YA, round(-p[3], 2)] for p in best[1]]      # Spiel: x=dE, y, z=-dN
out = {'centre': [round(C[0], 2), YC, round(-C[1], 2)], 'diameter': 6.5, 'anchors': anch, 'bearings': [p[0] for p in best[1]], 'dist': [round(p[1], 1) for p in best[1]]}
json.dump(out, open('data/lichtwolke.json', 'w'), indent=1); print(out)

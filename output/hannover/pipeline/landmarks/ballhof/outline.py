"""Vereinfachte Grundrisse + Kantenklassen (offen/Brandwand) fuer die Ensemble-Gebaeude -> build/outlines.json
Kante = 'party', wenn >= 50 % ihrer Laenge innerhalb 0.4 m eines fremden LoD2-Grundrisses liegen."""
import json, os, numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
H0 = 56.16
L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))['buildings']
hj = json.load(open(ROOT + 'web/assets/city/ground/heightfield.json'))
HF = np.fromfile(ROOT + 'web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(hj['h'], hj['w'])
def H(x, z):
    fx = (np.asarray(x) - hj['x0']) / hj['dx']; fz = (np.asarray(z) - hj['z0']) / hj['dz']
    i = np.floor(fx).astype(int); j = np.floor(fz).astype(int); a = fx - i; b = fz - j
    return (1 - a) * (1 - b) * HF[j, i] + a * (1 - b) * HF[j, i + 1] + (1 - a) * b * HF[j + 1, i] + a * b * HF[j + 1, i + 1]
MINE = {'BH': 'ID_c48b3643', 'PAV': 'ID_98a9691d', 'OST': 'ID_bcc9a17d', 'HINT': 'ID_ff0b0763', 'VOR': 'ID_7bf3782c',
        'SP': 'ID_b150f6c8', 'BP1': 'ID_84e270ab', 'BP2': 'ID_1a4a4aa4', 'BP3': 'ID_0875e4bf'}
fp = {}
for b in L:
    g = unary_union([Polygon([(p[0], -p[1]) for p in gg['ext']]).buffer(0) for gg in b['grounds']])
    fp[b['id']] = g
near = [i for i, g in fp.items() if g.distance(Point(-420, 225)) < 120]
out = {}
for k, pre in MINE.items():
    bid = [i for i in fp if i.startswith(pre)][0]
    g = fp[bid]
    s = g.simplify(0.18, preserve_topology=True)
    pts = np.array(s.exterior.coords)[:-1]
    # Kollineare/kurze Kanten entfernen
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            u = b - a; v = c - b
            cr = abs(u[0] * v[1] - u[1] * v[0]) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
            if cr < 0.035 or np.linalg.norm(u) < 0.25:
                pts = np.delete(pts, i, 0); changed = True; break
    ar = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if ar < 0: pts = pts[::-1]
    others = unary_union([fp[i] for i in near if i != bid])
    edges = []
    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        ln = LineString([p0, p1]); Lg = ln.length
        ts = np.linspace(0.05, 0.95, 19)
        sp = [ln.interpolate(t, normalized=True) for t in ts]
        frac = np.mean([others.distance(q) < 0.4 for q in sp])
        nb = sorted({i2[3:11] for i2 in near if i2 != bid and fp[i2].distance(ln) < 0.4 and fp[i2].intersection(ln.buffer(0.4)).area > 0.3})
        d = p1 - p0; n = np.array([d[1], -d[0]]) / Lg
        az = (np.degrees(np.arctan2(n[0], -n[1])) + 360) % 360       # Kompass: 0 = Nord (-z), 90 = Ost
        gy = H(np.linspace(p0[0], p1[0], 9) + n[0] * 0.6, np.linspace(p0[1], p1[1], 9) + n[1] * 0.6)
        edges.append(dict(i=i, p0=p0.round(3).tolist(), p1=p1.round(3).tolist(), L=round(Lg, 2), az=round(az), party=round(float(frac), 2), nb=nb,
                          gmin=round(float(gy.min()), 2), gmax=round(float(gy.max()), 2)))
    out[k] = dict(id=bid, pts=pts.round(3).tolist(), edges=edges, area=round(Polygon(pts).area, 1), area_lod2=round(g.area, 1))
    print('==', k, bid[:11], 'n', len(pts), 'A %.1f / %.1f' % (Polygon(pts).area, g.area))
    for e in edges:
        print('   e%-2d L=%6.2f az=%3d party=%.2f g=%.2f..%.2f %s -> %s nb=%s' % (e['i'], e['L'], e['az'], e['party'], e['gmin'], e['gmax'], e['p0'], e['p1'], e['nb']))
json.dump(out, open(os.path.join(HERE, 'build', 'outlines.json'), 'w'), indent=1)

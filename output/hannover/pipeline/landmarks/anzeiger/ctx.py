"""Context for previews: neighbouring LoD2 buildings (within R of the site, excluding the replaced ones) as oriented
triangle soup -> build/ctx_parts.npz (same layout as anzeiger_parts.npz) + camera shots -> build/shots.json"""
import json, os, sys
import numpy as np
from shapely.geometry import Polygon, Point
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geo, anz, bad
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
H0 = 56.16
CX, CZ, R = -505.0, -305.0, 170.0
rep = json.load(open(ROOT + 'data/landmarks_replace/anzeiger.json'))
skip = set(rep['lod2_ids'])
# procedural facade tiles (data/buildings_raw) near the site are imported directly by preview.py;
# their LoD2 buildings are skipped here so only buildings outside those tiles remain as grey LoD2 blocks
BT = []
try:
    BI = json.load(open(ROOT + 'data/buildings_raw/index.json'))
    for t in BI['tiles']:
        tx, tz = t['tile']
        if -7 <= tx <= -4 and -5 <= tz <= -2 and os.path.exists(ROOT + 'data/buildings_raw/' + t['file']):
            BT.append(t['file'])
            if t.get('windows') and os.path.exists(ROOT + 'data/buildings_raw/' + t['windows']): BT.append(t['windows'])
            skip |= set(t.get('buildings', []))
except Exception as e:
    print('CTX no buildings_raw', e)
L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))
mb = geo.MB()
n = 0
for b in L['buildings']:
    if b['id'] in skip: continue
    g = np.array(b['grounds'][0]['ext'])
    if np.hypot(g[:, 0].mean() - CX, -g[:, 1].mean() - CZ) > R: continue
    conv = lambda ring: np.array([[p[0], p[2] - H0, -p[1]] for p in ring])
    fp = Polygon([(p[0], -p[1]) for p in g]).buffer(0)
    for w in b['walls']:
        P = conv(w['ext'])
        nn = geo.newell(P)
        if np.linalg.norm(nn) < 1e-9: continue
        nn = geo.nrm(nn); c = P.mean(0)
        h = np.array([nn[0], 0, nn[2]])
        if np.linalg.norm(h) > 1e-6:
            h = geo.nrm(h)
            if fp.buffer(0.02).contains(Point(c[0] + h[0] * 0.3, c[2] + h[2] * 0.3)): h = -h
        P[P[:, 1] < P[:, 1].min() + 1e-3, 1] -= 1.5          # extend into the ground
        mb.poly('ctx_lod2', 'ctx_wall', P, h)
    for r in b['roofs']:
        P = conv(r['ext'])
        mb.poly('ctx_lod2', 'ctx_roof', P, geo.UP)
    n += 1
# heightfield ground (preview only; real ground tiles drawn on top, this one 3 cm lower)
hj = json.load(open(ROOT + 'web/assets/city/ground/heightfield.json'))
hf = np.fromfile(ROOT + 'web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(hj['h'], hj['w'])
xs = np.arange(-560.0, -400.0, 2.0); zs = np.arange(-450.0, -180.0, 2.0)
def hval(x, z):
    i = int(round(x - hj['x0'])); j = int(round(z - hj['z0']))
    if 0 <= i < hj['w'] and 0 <= j < hj['h'] and np.isfinite(hf[j, i]): return float(hf[j, i]) - 0.03
    return -1.0
G = np.array([[hval(x, z) for x in xs] for z in zs])
T = []
for j in range(len(zs) - 1):
    for i in range(len(xs) - 1):
        a = [xs[i], G[j, i], zs[j]]; b = [xs[i + 1], G[j, i + 1], zs[j]]; c2 = [xs[i + 1], G[j + 1, i + 1], zs[j + 1]]; d = [xs[i], G[j + 1, i], zs[j + 1]]
        mb.poly('ctx_lod2', 'ctx_ground', [a, b, c2, d], geo.UP)
arr = mb.arrays()
npz = {}; meta = {'objects': {}, 'materials': {'ctx_wall': {'color': [0.36, 0.35, 0.33], 'rough': 0.9}, 'ctx_roof': {'color': [0.25, 0.24, 0.23], 'rough': 0.9}, 'ctx_ground': {'color': [0.33, 0.33, 0.32], 'rough': 0.95}}, 'smooth': []}
Vs, Us, Ms, names = [], [], [], []
for i, ((o, m), (P, UV)) in enumerate(sorted(arr.items())):
    Vs.append(P); Us.append(UV); Ms.append(np.full(len(P) // 3, i, np.int32)); names.append(m)
npz['V_ctx_lod2'] = np.concatenate(Vs); npz['U_ctx_lod2'] = np.concatenate(Us); npz['M_ctx_lod2'] = np.concatenate(Ms)
meta['objects']['ctx_lod2'] = {'materials': names, 'tris': int(sum(len(m) for m in Ms))}
meta['bldg_tiles'] = BT
os.makedirs(os.path.join(HERE, 'build'), exist_ok=True)
np.savez_compressed(os.path.join(HERE, 'build', 'ctx_parts.npz'), **npz)
json.dump(meta, open(os.path.join(HERE, 'build', 'ctx_meta.json'), 'w'))
print('CTX lod2 buildings', n, 'tris', meta['objects']['ctx_lod2']['tris'], 'facade tiles', BT)

# ------------------------------------------------------------------ camera shots (game frame) -> blender
FE, FB = anz.FE, bad.FB
def g(p): return [float(p[0]), float(p[1]), float(p[2])]
shots = {
    'front': (FE.w(15.7, 1.7, 52), FE.w(15.7, 21.5, 0), 30),              # Wikimedia 20230320 (frontal across Goseriede; moved in front of the square's canopy)
    'gose': (FB.w(10.0, 1.6, 30), FB.w(10.0, 10.5, 0), 22),               # Goseriedebad frontal (20230320)
    'square_ne': ([-428.0, 1.7, -338.0], [-497.0, 17.0, -292.0], 24),     # Goseriede square April 2020
    'dome_e': ([-330.0, 38.0, -255.0], [-499.0, 42.0, -283.0], 135),        # from the east, elevated (EAG car park 2025)
    'aerial_se': ([-410.0, 105.0, -190.0], [-505.0, 14.0, -300.0], 35),
    'aerial_nw': ([-600.0, 95.0, -400.0], [-505.0, 14.0, -300.0], 35),
    'portals': (FE.w(31.0, 1.7, 9.0), FE.w(14.0, 3.6, 0), 24),
    'crown': (FE.w(9.0, 1.7, 42.0), FE.w(14.5, 35.5, 0), 85),
    'lookup': (FE.w(9.7, 1.4, 2.6), FE.w(9.7, 16.0, -0.4), 16),
    'bay_close': (FB.w(9.5, 1.6, 8.5), FB.w(9.5, 6.0, 0), 22),
    'kestner': (bad.FK.w(11.0, 1.6, 14.0), bad.FK.w(11.0, 4.5, 0), 24),
    'west': ([-575.0, 5.0, -335.0], [-510.0, 16.0, -300.0], 28),
    'portal_eye': (FE.w(9.7, 1.7, 7.5), FE.w(9.7, 3.2, 0), 24),            # stepped portals at eye level (ref 03 / 06)
    'lightwell': ([-532.0, 28.0, -282.0], [-511.0, 8.0, -298.0], 35),      # light well Anzeiger / Goseriedebad from SW
    'north_ne': ([-490.0, 1.7, -378.0], [-505.0, 7.0, -355.0], 24),        # Goseriede walking south: north wing NE corner
    'roofs_top': ([-468.0, 75.0, -335.0], [-516.0, 10.0, -330.0], 30),     # roof windows, Kestner glazing, AC units (DOP)
}
out = {k: {'cam': g(c), 'tgt': g(t), 'lens': l} for k, (c, t, l) in shots.items()}
json.dump(out, open(os.path.join(HERE, 'build', 'shots.json'), 'w'), indent=1)
print('SHOTS', list(out))

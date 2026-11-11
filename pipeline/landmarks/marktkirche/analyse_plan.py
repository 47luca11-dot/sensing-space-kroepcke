"""Analyse LGLN-LoD2-Teile der Marktkirche (DENILD43000009SEf) im Kirchen-Lokalframe.
Lokal: u = Kirchenachse (West->Ost), v = Nord (links von u); Ursprung = Turmmitte (Grund).
Gibt Grundrisse/Hoehen + Bodenhoehen (web heightfield) aus -> build/plan_local.json"""
import json, os, math, numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
parts = [p for p in json.load(open(os.path.join(HERE, 'lgln_parts.json'))) if p['building'].endswith('00009SEf')]
others = [p for p in json.load(open(os.path.join(HERE, 'lgln_parts.json'))) if not p['building'].endswith('00009SEf')]
hfm = json.load(open(ROOT + '/web/assets/city/ground/heightfield.json'))
hf = np.fromfile(ROOT + '/web/assets/city/ground/heightfield.bin', np.float32).reshape(hfm['h'], hfm['w'])
def H(x, z):
    fx = (x - hfm['x0']) / hfm['dx']; fz = (z - hfm['z0']) / hfm['dz']
    i0, j0 = int(math.floor(fz)), int(math.floor(fx)); a, b = fz - i0, fx - j0
    return float((1-a)*(1-b)*hf[i0, j0] + (1-a)*b*hf[i0, j0+1] + a*(1-b)*hf[i0+1, j0] + a*b*hf[i0+1, j0+1])
def g2(p):  # game (x,z) of a part polygon point [x,y,z]
    return (p[0], p[2])
nave = next(p for p in parts if p['part'].endswith('41069dfe'))
gp = np.array([g2(q) for q in nave['surf']['ground'][0]])
# longest edges -> axis
e = np.diff(np.vstack([gp, gp[:1]]), axis=0); L = np.hypot(e[:, 0], e[:, 1])
print('nave ground pts', len(gp))
for i in np.argsort(-L)[:6]: print('  edge', i, 'len %.2f' % L[i], 'dir deg (game x/z) %.2f' % math.degrees(math.atan2(e[i, 1], e[i, 0])))
# axis from the long N/S wall edges (weighted)
ang = []
for i in range(len(e)):
    a = math.degrees(math.atan2(e[i, 1], e[i, 0]))
    a2 = (a + 90) % 180 - 90       # fold to [-90,90)
    if abs(a2 + 9.3) < 3: ang.append((a2, L[i]))
th = sum(a * w for a, w in ang) / sum(w for a, w in ang)
print('axis angle deg', round(th, 3))
T = math.radians(th); U = np.array([math.cos(T), math.sin(T)]); V = np.array([U[1], -U[0]])   # V = north-ish
allp = np.array([g2(q) for p in parts for g in p['surf']['ground'] for q in g])
def loc(P): P = np.asarray(P, float); return np.c_[(P - O) @ U, (P - O) @ V]
# tower: union of the 4 gable parts
tw = [p for p in parts if p.get('roofType') == '3100' and not p['part'].endswith('41069dfe')]
O = np.zeros(2)
tp = unary_union([Polygon([g2(q) for q in p['surf']['ground'][0]]).buffer(0) for p in tw])
tl = loc(np.array(tp.exterior.coords))
c = np.array([(tl[:, 0].min() + tl[:, 0].max()) / 2, (tl[:, 1].min() + tl[:, 1].max()) / 2])
O = c[0] * U + c[1] * V
print('ORIGIN (tower centre) game x/z', O.round(3), 'U', U.round(5), 'V', V.round(5))
out = dict(angle_deg=th, origin=O.tolist(), U=U.tolist(), V=V.tolist(), parts=[])
for p in parts + others:
    for g in p['surf']['ground'][:1]:
        l = loc([g2(q) for q in g])
        ys = [q[1] for k in p['surf'] for poly in p['surf'][k] for q in poly]
        rec = dict(part=p['part'][-8:], bld=p['building'][-8:], rt=p.get('roofType'), u0=l[:, 0].min(), u1=l[:, 0].max(), v0=l[:, 1].min(), v1=l[:, 1].max(),
                   y0=min(ys), y1=max(ys), eave=float(p.get('MittlereTraufHoehe', 0)) - 56.16, ridge=float(p.get('Firsthoehe', 0)) - 56.16, poly=l.round(3).tolist())
        out['parts'].append(rec)
for r in sorted(out['parts'], key=lambda r: (r['rt'], r['u0'])):
    print('%s %s rt %s u %.2f..%.2f v %.2f..%.2f y %.2f..%.2f eave %.2f ridge %.2f n %d' % (r['bld'], r['part'], r['rt'], r['u0'], r['u1'], r['v0'], r['v1'], r['y0'], r['y1'], r['eave'], r['ridge'], len(r['poly'])))
navel = loc(gp)
print('nave poly local:'); print(' '.join('(%.2f,%.2f)' % tuple(q) for q in navel))
# ground heights around footprint
fp = unary_union([Polygon([g2(q) for q in p['surf']['ground'][0]]).buffer(0) for p in parts])
ring = fp.buffer(1.0).exterior
hs = [(ring.interpolate(t).x, ring.interpolate(t).y, H(ring.interpolate(t).x, ring.interpolate(t).y)) for t in np.linspace(0, ring.length, 80, endpoint=False)]
hs = np.array(hs); lh = loc(hs[:, :2])
print('ground ring y: min %.3f max %.3f mean %.3f' % (hs[:, 2].min(), hs[:, 2].max(), hs[:, 2].mean()))
for k in range(0, 80, 5): print('  u %.1f v %.1f y %.3f' % (lh[k, 0], lh[k, 1], hs[k, 2]))
out['ground_ring'] = np.c_[lh, hs[:, 2]].round(3).tolist()
# ground grid in local frame (for the model plinth): 1 m
gu = np.arange(-14, 62, 1.0); gv = np.arange(-24, 24, 1.0)
G = np.array([[H(*(O + a * U + b * V)) for a in gu] for b in gv])
out['ground_grid'] = dict(u0=-14, v0=-24, d=1.0, nu=len(gu), nv=len(gv), y=G.round(3).tolist())
json.dump(out, open(os.path.join(HERE, 'build', 'plan_local.json'), 'w'))

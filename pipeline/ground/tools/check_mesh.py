"""Pruefung: Gelaende + Stufenwaende + Schuerzen + Bauwerke verschweisst -> offene Kanten / nicht-mannigfaltige Kanten / Normalen."""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from common import *
import geom as G
TM = np.load(CACHE + 'terrain.npz'); TP = pickle.load(open(CACHE + 'terrain_parts.pkl', 'rb')); ST = pickle.load(open(CACHE + 'structures.pkl', 'rb'))
def stats(name, parts):
    m = G.merge([dict(p, mat='x') for p in parts])[0]
    k = np.round(m['pos'] * 1000).astype(np.int64); uq, iv = np.unique(k, axis=0, return_inverse=True)
    I = iv.reshape(-1)[m['idx']]
    I = I[(I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 0] != I[:, 2])]
    E = np.sort(np.concatenate([I[:, [0, 1]], I[:, [1, 2]], I[:, [2, 0]]]), 1)
    ek = E[:, 0] * len(uq) + E[:, 1]; u, c = np.unique(ek, return_counts=True)
    bnd = u[c == 1]; P = uq / 1000.0
    bE = np.c_[bnd // len(uq), bnd % len(uq)]
    ln = np.linalg.norm(P[bE[:, 0]] - P[bE[:, 1]], axis=1)
    # Randkanten, die nicht an der Unterkante (Schuerze/Boden) liegen
    print(f'{name}: tris {len(I)}, verts {len(uq)}, open edges {len(bnd)} (total length {ln.sum():.1f} m), non-manifold edges {(c > 2).sum()}')
    return P, bE
ter = G.part('x', TM['OV'], TM['OT'], TM['ON'])
n_up = (TM['ON'][:, 1] > 0).mean()
print('terrain vertex normals pointing up:', round(float(n_up) * 100, 3), '%')
P, bE = stats('terrain only', [ter])
P, bE = stats('terrain + walls + skirts', [ter] + TP['parts'])
P, bE = stats('terrain + walls + skirts + structure collision solids', [ter] + TP['parts'] + ST['coll'])
# wo liegen die offenen Kanten der Gesamtflaeche? (Hoehe relativ zur Umgebung)
yb = np.minimum(P[bE[:, 0], 1], P[bE[:, 1], 1])
print('open edges by y: below -1.0 m (skirt bottoms / pit floors):', int((yb < -1.0).sum()), ' others:', int((yb >= -1.0).sum()))

# Detail: offene Kanten von Gelaende+Waende+Schuerzen klassifizieren
import shapely
L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
P, bE = stats('terrain + walls + skirts (again)', [ter] + TP['parts'])
bz = np.load(CACHE + 'base.npz'); BG = Grid(bz['B'].astype(np.float64), float(bz['x0']), float(bz['z0']), 1.0)
mid = (P[bE[:, 0]] + P[bE[:, 1]]) / 2
rel = mid[:, 1] - BG(mid[:, 0], mid[:, 2])
holes = shapely.unary_union([h['poly'].boundary for h in L['holes'].values()])
dh = shapely.distance(holes, shapely.points(mid[:, [0, 2]]))
outer = L['region_out'].exterior; dout = shapely.distance(outer, shapely.points(mid[:, [0, 2]]))
sqin = L['sq_in'].exterior; dsq = shapely.distance(sqin, shapely.points(mid[:, [0, 2]]))
cat = np.where(dh < 0.05, 'hole rim (structure attaches)', np.where((dout < 0.05) | (dsq < 0.05), np.where(rel < -0.4, 'skirt bottom', 'boundary top?'), 'interior'))
ln = np.linalg.norm(P[bE[:, 0]] - P[bE[:, 1]], axis=1)
for c in np.unique(cat): print(f'  {c}: {int((cat == c).sum())} edges, {ln[cat == c].sum():.1f} m')
ii = np.nonzero(cat == 'interior')[0]
print('  interior open edge samples (x,y,z, len, y-B):', [(*mid[i].round(2), round(ln[i], 3), round(rel[i], 3)) for i in ii[:12]])
# Cluster der inneren offenen Kanten (10-m-Zellen): wo liegen echte Luecken?
if len(ii):
    cell = np.floor(mid[ii][:, [0, 2]] / 10).astype(int)
    uc, inv_ = np.unique(cell, axis=0, return_inverse=True)
    tot = np.bincount(inv_.reshape(-1), weights=ln[ii]); cnt = np.bincount(inv_.reshape(-1))
    order = np.argsort(-tot)
    print('  interior open edges by 10-m cell (x,z, edges, length m, max |y-B|):')
    for o in order[:25]:
        sel_ = ii[inv_.reshape(-1) == o]
        print('   ', (uc[o] * 10).tolist(), int(cnt[o]), round(float(tot[o]), 2), round(float(np.abs(rel[sel_]).max()), 2), 'dy-span', round(float(np.ptp(np.r_[P[bE[sel_, 0], 1], P[bE[sel_, 1], 1]])), 2))

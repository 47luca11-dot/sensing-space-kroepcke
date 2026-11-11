"""Welche LoD2-Gebaeude sind schon im Rhino-Detailmodell (Opernplatz) enthalten?
Rastert die Grundrisse der Detail-Dach/Wand-Meshes (1 m) und vergleicht mit LoD2-Grundflaechen.
Ausgabe data/detail_covered.json {ids:[...], mask:{x0,z0,res,w,h}} + detail_mask.npy (Spielkoordinaten)"""
import json, numpy as np, trimesh
from matplotlib.path import Path
R = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/"
OFF = np.array([84.117, -0.16, 141.706])          # Rhino-glTF -> Spiel (Ursprung Kroepcke-Uhr)
res = 1.0; x0, z0, W, Hh = -400.0, -400.0, 800, 800
mask = np.zeros((Hh, W), bool)
for g in ['bldg_roof', 'bldg_walls', 'landmark_opernhaus', 'fix', 'landmark_tiefgarage_pavillon', 'kiosk']:
    m = trimesh.load(R + 'rhino_export/' + g + '.glb', force='scene').to_geometry()
    v = m.vertices + OFF; tri = v[m.faces]
    for t in tri[:, :, [0, 2]]:
        lo = np.floor((t.min(0) - [x0, z0]) / res).astype(int); hi = np.ceil((t.max(0) - [x0, z0]) / res).astype(int)
        lo = np.clip(lo, 0, [W - 1, Hh - 1]); hi = np.clip(hi, 0, [W, Hh])
        if (hi - lo).prod() == 0: continue
        xs, zs = np.meshgrid(np.arange(lo[0], hi[0]) * res + x0 + res / 2, np.arange(lo[1], hi[1]) * res + z0 + res / 2)
        P = np.c_[xs.ravel(), zs.ravel()]
        a, b, c = t; d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-9: continue
        l1 = ((b[1] - c[1]) * (P[:, 0] - c[0]) + (c[0] - b[0]) * (P[:, 1] - c[1])) / d
        l2 = ((c[1] - a[1]) * (P[:, 0] - c[0]) + (a[0] - c[0]) * (P[:, 1] - c[1])) / d
        ins = (l1 >= -0.05) & (l2 >= -0.05) & (1 - l1 - l2 >= -0.05)
        mask[lo[1]:hi[1], lo[0]:hi[0]] |= ins.reshape(xs.shape)
np.save(R + 'detail_mask.npy', mask)
B = json.load(open(R + 'lod2_buildings.json'))['buildings']
cov = []
for b in B:
    polys = b['grounds'] or []
    if not polys: continue
    cells = []
    for g in polys:
        P = np.array(g['ext'])[:, :2] * [1, -1]           # Spiel: x=dE, z=-dN
        lo = np.floor((P.min(0) - [x0, z0]) / res).astype(int); hi = np.ceil((P.max(0) - [x0, z0]) / res).astype(int)
        lo = np.clip(lo, 0, W - 1); hi = np.clip(hi, 1, W)
        xs, zs = np.meshgrid(np.arange(lo[0], hi[0]) * res + x0 + res / 2, np.arange(lo[1], hi[1]) * res + z0 + res / 2)
        Q = np.c_[xs.ravel(), zs.ravel()]; ins = Path(P).contains_points(Q)
        cells += [(int((q[1] - z0) // res), int((q[0] - x0) // res)) for q in Q[ins]]
    if not cells: continue
    cells = np.array(cells); f = mask[cells[:, 0], cells[:, 1]].mean()
    if f > 0.5: cov.append(b['id'])
json.dump({'ids': cov, 'mask': {'x0': x0, 'z0': z0, 'res': res, 'w': W, 'h': Hh, 'file': 'detail_mask.npy'}}, open(R + 'detail_covered.json', 'w'))
print('lod2 buildings', len(B), 'covered by detail model', len(cov), 'mask cells', int(mask.sum()))

"""Lageplan Leibnizhaus/Holzmarkt/Beginenturm: DOP10 + LoD2-Grundrisse + OSM.
Schreibt build/site_map.png und build/site_dop.npy (Ausschnitt) - nur Analyse."""
import json, os, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(H, 'data')
RES = r'C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research'
E0, N0 = 550298.88274216, 5802940.705715501

# Ausschnitt im Spielrahmen (x = dE, z = -dN)
x0, x1 = float(sys.argv[1]) if len(sys.argv) > 1 else -560, float(sys.argv[2]) if len(sys.argv) > 2 else -380
z0, z1 = float(sys.argv[3]) if len(sys.argv) > 3 else 230, float(sys.argv[4]) if len(sys.argv) > 4 else 400
out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(HERE, 'build', 'site_map.png')

im = Image.open(os.path.join(RES, 'dop10', 'dop10rgb_32_549_5802_1_ni_2025-03-06.tif'))
# Tile 549000..550000 E, 5802000..5803000 N; Zeile 0 = N 5803000
def px(x, z):
    E = E0 + x; N = N0 - z
    return (E - 549000) / 0.1, (5803000 - N) / 0.1
c0, r0 = px(x0, z0); c1, r1 = px(x1, z1)
box = (int(min(c0, c1)), int(min(r0, r1)), int(max(c0, c1)), int(max(r0, r1)))
crop = np.asarray(im.crop(box))

lod = json.load(open(os.path.join(DATA, 'lod2_buildings.json'), encoding='utf8'))
osm = json.load(open(os.path.join(DATA, 'osm_features.json'), encoding='utf8'))

fig, ax = plt.subplots(figsize=(16, 16 * (z1 - z0) / (x1 - x0)))
ax.imshow(crop, extent=(x0, x1, z1, z0))
for b in lod['buildings']:
    c = b['centre']
    if not (x0 - 30 < c[0] < x1 + 30 and z0 - 30 < -c[1] < z1 + 30): continue
    for g in b['grounds']:
        p = np.array(g['ext'])
        ax.plot(np.r_[p[:, 0], p[0, 0]], -np.r_[p[:, 1], p[0, 1]], 'y-', lw=1.2)
    zs = [q[2] for r in b['roofs'] for q in r['ext']]
    ax.text(c[0], -c[1], '%s\n%s %s\n%.1f' % (b['id'][3:11], b['street'] if b['street'] != 'None' else '', b['number'] if b['number'] != 'None' else '', max(zs) - 56.16),
            color='yellow', fontsize=6, ha='center', clip_on=True)
for w in osm['ways']:
    t = w['tags']
    p = np.array(w['pts'])
    if not len(p): continue
    if not ((x0 < p[:, 0]).any() and (p[:, 0] < x1).any() and (z0 < p[:, 1]).any() and (p[:, 1] < z1).any()): continue
    if 'building' in t or 'building:part' in t:
        ax.plot(p[:, 0], p[:, 1], 'c-', lw=0.7)
        nm = t.get('name') or t.get('addr:housenumber')
        if nm: ax.text(p[:, 0].mean(), p[:, 1].mean() + 2, nm, color='cyan', fontsize=6, ha='center', clip_on=True)
    elif 'highway' in t:
        ax.plot(p[:, 0], p[:, 1], 'w--', lw=0.5)
for n in osm['nodes']:
    x, z = n['p']
    if x0 < x < x1 and z0 < z < z1:
        t = n['tags']
        if any(k in t for k in ('amenity', 'historic', 'tourism', 'man_made', 'name')):
            ax.plot(x, z, 'r.', ms=4)
            ax.text(x, z, (t.get('name') or t.get('amenity') or t.get('historic') or '')[:30], color='red', fontsize=6, clip_on=True)
ax.set_xlim(x0, x1); ax.set_ylim(z1, z0)
ax.grid(color='white', alpha=0.3, lw=0.3)
ax.set_xticks(np.arange(np.ceil(x0 / 10) * 10, x1, 10)); ax.set_yticks(np.arange(np.ceil(z0 / 10) * 10, z1, 10))
ax.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(out, dpi=110)
print('ok', out)

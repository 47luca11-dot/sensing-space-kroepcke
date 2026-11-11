"""Site analysis for Anzeiger-Hochhaus + Goseriedebad: LoD2 (city) in game frame, DOP10 crop with overlay,
heightfield samples. Writes site_lod2.json, site_dop.png, site_dop_overlay.png, site_hf.npy into this folder."""
import json, sys, os, numpy as np, tifffile
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
RES = 'C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research/'
E0, N0, H0 = 550298.88274216, 5802940.705715501, 56.16
IDS = {'anz': 'ID_6388af31', 'bad': 'ID_5ea871fc', 'n7': 'ID_9faad61e', 'west': 'ID_2c0e0ff2', 'ffn': 'ID_147875ac'}
L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))
out = {}
for k, pre in IDS.items():
    b = [b for b in L['buildings'] if b['id'].startswith(pre)][0]
    conv = lambda ring: [[round(p[0], 3), round(p[2] - H0, 3), round(-p[1], 3)] for p in ring]
    out[k] = {'id': b['id'], 'height': b['height'], 'street': b['street'], 'number': b['number'],
              'walls': [conv(w['ext']) for w in b['walls']], 'roofs': [conv(r['ext']) for r in b['roofs']],
              'grounds': [conv(g['ext']) for g in b['grounds']]}
json.dump(out, open(os.path.join(HERE, 'site_lod2.json'), 'w'))
# ---- DOP crop
x0, x1, z0, z1 = -560.0, -470.0, -380.0, -235.0
f = RES + 'dop10/dop10rgb_32_549_5803_1_ni_2025-03-06.tif'
with tifffile.TiffFile(f) as t:
    p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
    sc = tags['ModelPixelScaleTag']; tp = tags['ModelTiepointTag']; img = p.asarray()
E_a, E_b = E0 + x0, E0 + x1; N_a, N_b = N0 - z1, N0 - z0
c0 = int((E_a - tp[3]) / sc[0]); c1 = int((E_b - tp[3]) / sc[0]); r0 = int((tp[4] - N_b) / sc[1]); r1 = int((tp[4] - N_a) / sc[1])
a = img[r0:r1, c0:c1, :3]
print('dop crop', a.shape, sc[:2], 'rows', r0, r1, 'cols', c0, c1)
UP = 3; S = UP / sc[0]
im = Image.fromarray(a).resize((a.shape[1] * UP, a.shape[0] * UP), Image.LANCZOS)
im.save(os.path.join(HERE, 'site_dop.png'))
d = ImageDraw.Draw(im)
P = lambda x, z: ((x - x0) * S, (z - z0) * S)
cols = {'anz': (255, 60, 60), 'bad': (60, 200, 255), 'n7': (255, 255, 0), 'west': (255, 128, 0), 'ffn': (200, 0, 255)}
for k, b in out.items():
    for r in b['roofs']:
        d.line([P(q[0], q[2]) for q in r] + [P(r[0][0], r[0][2])], fill=cols[k], width=1)
for gx in range(int(x0), int(x1) + 1, 5):
    d.line([P(gx, z0), P(gx, z0 + (2 if gx % 10 else 4))], fill=(255, 255, 255), width=2)
    if gx % 10 == 0: d.text(P(gx + 0.3, z0 + 4), str(gx), fill=(255, 255, 255))
for gz in range(int(z0), int(z1) + 1, 5):
    d.line([P(x0, gz), P(x0 + (2 if gz % 10 else 4), gz)], fill=(255, 255, 255), width=2)
    if gz % 10 == 0: d.text(P(x0 + 4.5, gz - 1), str(gz), fill=(255, 255, 255))
im.save(os.path.join(HERE, 'site_dop_overlay.png'))
# ---- heightfield
hj = json.load(open(ROOT + 'web/assets/city/ground/heightfield.json'))
hf = np.fromfile(ROOT + 'web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(hj['h'], hj['w'])
i0 = int(x0 - hj['x0']); i1 = int(x1 - hj['x0']); j0 = int(z0 - hj['z0']); j1 = int(z1 - hj['z0'])
sub = hf[j0:j1 + 1, i0:i1 + 1]
np.save(os.path.join(HERE, 'site_hf.npy'), sub)
print('hf sub', sub.shape, np.nanmin(sub), np.nanmax(sub), 'x0,z0', x0, z0)
np.set_printoptions(precision=2, suppress=True, linewidth=250)
print(sub[::5, ::5])

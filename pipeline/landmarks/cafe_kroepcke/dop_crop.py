"""Crop DOP10 + DOM1 around the Cafe Kroepcke and overlay LGLN LoD2 parts (game frame). Output PNGs for inspection."""
import sys, json, numpy as np, tifffile
from PIL import Image, ImageDraw
DOP, DOM, PARTS, OUT = sys.argv[1:5]
E0, N0 = 550298.88274216, 5802940.705715501
x0, x1, z0, z1 = 8.0, 62.0, 0.0, 52.0       # game window
with tifffile.TiffFile(DOP) as t:
    p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
    sc = tags['ModelPixelScaleTag']; tp = tags['ModelTiepointTag']
    print('dop', p.shape, sc[:2], tp[3:5])
    img = p.asarray()
def crop(img, sc, tp, upscale):
    E_a, E_b = E0 + x0, E0 + x1; N_a, N_b = N0 - z1, N0 - z0  # z=-dN
    c0 = int((E_a - tp[3]) / sc[0]); c1 = int((E_b - tp[3]) / sc[0])
    r0 = int((tp[4] - N_b) / sc[1]); r1 = int((tp[4] - N_a) / sc[1])
    return img[r0:r1, c0:c1]
a = crop(img, sc, tp, 1)[..., :3]
S = 1 / sc[0] * 2   # px per m after 2x upscale
im = Image.fromarray(a).resize((a.shape[1] * 2, a.shape[0] * 2), Image.LANCZOS)
d = ImageDraw.Draw(im)
parts = json.load(open(PARTS))
col = {'3900': (255, 60, 60), '1000': (60, 160, 255)}
def P(x, z): return ((x - x0) * S, (z - z0) * S)
for r in parts:
    if not r['building'].endswith('00009TB7'): continue
    for g in r['surf']['ground']:
        d.line([P(q[0], q[2]) for q in g] + [P(g[0][0], g[0][2])], fill=col.get(r['roofType'], (255, 255, 0)), width=2)
    allp = np.array([q for g in r['surf']['ground'] for q in g])
    cx, cz = allp[:, 0].mean(), allp[:, 2].mean()
    d.text(P(cx, cz), r['part'][-3:] + ' ' + r['measuredHeight'][:4], fill=(255, 255, 0))
# grid every 5 m
for gx in range(int(x0), int(x1) + 1, 5):
    d.line([P(gx, z0), P(gx, z0 + 1)], fill=(255, 255, 255)); d.text(P(gx, z0 + 1), str(gx), fill=(255, 255, 255))
for gz in range(int(z0), int(z1) + 1, 5):
    d.line([P(x0, gz), P(x0 + 1, gz)], fill=(255, 255, 255)); d.text(P(x0 + 1, gz), str(gz), fill=(255, 255, 255))
im.save(OUT + '_dop_overlay.png')
Image.fromarray(a).resize((a.shape[1] * 2, a.shape[0] * 2), Image.LANCZOS).save(OUT + '_dop.png')
# DOM1
with tifffile.TiffFile(DOM) as t:
    p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
    sc2 = tags['ModelPixelScaleTag']; tp2 = tags['ModelTiepointTag']; dom = p.asarray()
    print('dom', p.shape, sc2[:3], tp2[3:6], dom.dtype)
b = crop(dom, sc2, tp2, 1).astype(float)
np.save(OUT + '_dom.npy', b)
print('dom crop', b.shape, b.min(), b.max())
v = np.clip((b - 56) / 14, 0, 1)
Image.fromarray((v * 255).astype(np.uint8)).resize((b.shape[1] * 10, b.shape[0] * 10), Image.NEAREST).save(OUT + '_dom.png')

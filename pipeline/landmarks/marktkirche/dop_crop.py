"""Crop DOP10 (2025) + DOM1 (2016) around the Marktkirche, overlay LGLN LoD2 ground polygons (game frame) -> build/*.png"""
import sys, json, numpy as np, tifffile
from PIL import Image, ImageDraw
R = "C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research"
DOP = R + "/dop10/dop10rgb_32_550_5802_1_ni_2025-03-06.tif"; DOM = R + "/dom1/dom1_32_550_5802_1_ni_2016.tif"
E0, N0 = 550298.88274216, 5802940.705715501
x0, x1, z0, z1 = -285.0, -195.0, 265.0, 325.0
def crop(img, sc, tp):
    E_a, E_b = E0 + x0, E0 + x1; N_a, N_b = N0 - z1, N0 - z0
    c0 = int(round((E_a - tp[3]) / sc[0])); c1 = int(round((E_b - tp[3]) / sc[0]))
    r0 = int(round((tp[4] - N_b) / sc[1])); r1 = int(round((tp[4] - N_a) / sc[1]))
    return img[r0:r1, c0:c1]
with tifffile.TiffFile(DOP) as t:
    p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
    sc = tags['ModelPixelScaleTag']; tp = tags['ModelTiepointTag']; img = p.asarray()
a = crop(img, sc, tp)[..., :3]
np.save('build/dop_crop.npy', a)
UP = 2; S = UP / sc[0]
im = Image.fromarray(a).resize((a.shape[1] * UP, a.shape[0] * UP), Image.LANCZOS)
im.save('build/dop.png')
d = ImageDraw.Draw(im)
parts = json.load(open('lgln_parts.json'))
def P(x, z): return ((x - x0) * S, (z - z0) * S)
for r in parts:
    if not r['building'].endswith('00009SEf'): continue
    rt = r.get('roofType')
    c = {'1000': (255, 255, 0), '3100': (255, 60, 60), '3200': (60, 255, 60), '3500': (60, 160, 255), '4000': (255, 0, 255), '5000': (0, 255, 255)}.get(rt, (255, 255, 255))
    for g in r['surf']['ground']:
        d.line([P(q[0], q[2]) for q in g] + [P(g[0][0], g[0][2])], fill=c, width=1)
for gx in range(int(x0), int(x1) + 1, 5):
    d.line([P(gx, z0), P(gx, z0 + 1.5)], fill=(255, 255, 255), width=2); d.text(P(gx + 0.3, z0 + 1.6), str(gx), fill=(255, 255, 255))
for gz in range(int(z0), int(z1) + 1, 5):
    d.line([P(x0, gz), P(x0 + 1.5, gz)], fill=(255, 255, 255), width=2); d.text(P(x0 + 1.6, gz - 0.5), str(gz), fill=(255, 255, 255))
im.save('build/dop_overlay.png')
with tifffile.TiffFile(DOM) as t:
    p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
    sc2 = tags['ModelPixelScaleTag']; tp2 = tags['ModelTiepointTag']; dom = p.asarray()
b = crop(dom, sc2, tp2).astype(float)
np.save('build/dom_crop.npy', b)
print('dom', b.shape, b.min(), b.max())
v = np.clip((b - 56) / 40, 0, 1)
Image.fromarray((v * 255).astype(np.uint8)).resize((b.shape[1] * 10, b.shape[0] * 10), Image.NEAREST).save('build/dom.png')

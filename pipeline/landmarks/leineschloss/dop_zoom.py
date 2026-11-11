"""Hochaufgeloester DOP10-Ausschnitt (10 px/m) mit LoD2-Kanten + 5-m-Raster: dop_zoom.py x0 x1 z0 z1 out.png"""
import sys, json, numpy as np, tifffile
from PIL import Image, ImageDraw
from ls_common import *
x0, x1, z0, z1 = map(float, sys.argv[1:5]); out = sys.argv[5]
S = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0
b = load_city_building()
W = int((x1 - x0) * S); H = int((z1 - z0) * S)
canvas = np.zeros((H, W, 3), np.uint8)
for fn in [RES + "/dop10/dop10rgb_32_549_5802_1_ni_2025-03-06.tif", RES + "/dop10/dop10rgb_32_550_5802_1_ni_2025-03-06.tif"]:
    with tifffile.TiffFile(fn) as t:
        p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
        sc = tags["ModelPixelScaleTag"]; tp = tags["ModelTiepointTag"]; img = p.asarray()[..., :3]
    cs = ((E0 + x0 + (np.arange(W) + 0.5) / S) - tp[3]) / sc[0]
    rs = (tp[4] - (N0 - (z0 + (np.arange(H) + 0.5) / S))) / sc[1]
    ci = np.floor(cs).astype(int); ri = np.floor(rs).astype(int)
    okc = (ci >= 0) & (ci < img.shape[1]); okr = (ri >= 0) & (ri < img.shape[0])
    sub = img[np.clip(ri, 0, img.shape[0] - 1)][:, np.clip(ci, 0, img.shape[1] - 1)]
    m = okr[:, None] & okc[None, :]
    canvas[m] = sub[m]
im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
def P(x, z): return ((x - x0) * S, (z - z0) * S)
if "--nolod" not in sys.argv:
    for r in b["roofs"]:
        e = np.array(r["ext"])
        d.line([P(q[0], -q[1]) for q in e] + [P(e[0][0], -e[0][1])], fill=(255, 255, 0), width=1)
    for g in b["grounds"]:
        e = np.array(g["ext"])
        d.line([P(q[0], -q[1]) for q in e] + [P(e[0][0], -e[0][1])], fill=(255, 0, 0), width=1)
step = 5
for gx in range(int(np.ceil(x0 / step)) * step, int(x1) + 1, step):
    d.line([P(gx, z0), P(gx, z0 + (3 if gx % 10 == 0 else 1.5))], fill=(255, 255, 255), width=2)
    if gx % 10 == 0: d.text(P(gx + 0.3, z0 + 3.5), str(gx), fill=(255, 255, 255))
for gz in range(int(np.ceil(z0 / step)) * step, int(z1) + 1, step):
    d.line([P(x0, gz), P(x0 + (3 if gz % 10 == 0 else 1.5), gz)], fill=(255, 255, 255), width=2)
    if gz % 10 == 0: d.text(P(x0 + 3.5, gz - 1), str(gz), fill=(255, 255, 255))
im.save(out)
print(out, im.size)

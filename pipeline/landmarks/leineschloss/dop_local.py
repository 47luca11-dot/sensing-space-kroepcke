"""DOP10 im lokalen Gebaeuderahmen (u nach rechts = SO, v nach oben = NO) mit LoD2-Linien: dop_local.py u0 u1 v0 v1 S out.png [--nolod]"""
import sys, json, numpy as np, tifffile
from PIL import Image, ImageDraw
from ls_common import *
u0, u1, v0, v1, S = map(float, sys.argv[1:6]); out = sys.argv[6]
W = int((u1 - u0) * S); H = int((v1 - v0) * S)
uu = u0 + (np.arange(W) + 0.5) / S; vv = v1 - (np.arange(H) + 0.5) / S
UU, VV = np.meshgrid(uu, vv)
X, Z = local_to_game(UU, VV); E = E0 + X; N = N0 - Z
canvas = np.zeros((H, W, 3), np.uint8)
for fn in [RES + "/dop10/dop10rgb_32_549_5802_1_ni_2025-03-06.tif", RES + "/dop10/dop10rgb_32_550_5802_1_ni_2025-03-06.tif"]:
    with tifffile.TiffFile(fn) as t:
        p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
        sc = tags["ModelPixelScaleTag"]; tp = tags["ModelTiepointTag"]; img = p.asarray()[..., :3]
    ci = np.floor((E - tp[3]) / sc[0]).astype(int); ri = np.floor((tp[4] - N) / sc[1]).astype(int)
    m = (ci >= 0) & (ci < img.shape[1]) & (ri >= 0) & (ri < img.shape[0])
    canvas[m] = img[ri[m], ci[m]]
im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
def P(x, z):
    u, v = game_to_local(x, z); return ((u - u0) * S, (v1 - v) * S)
if "--nolod" not in sys.argv:
    b = load_city_building()
    for r in b["roofs"]:
        e = np.array(r["ext"]); d.line([P(q[0], -q[1]) for q in e] + [P(e[0][0], -e[0][1])], fill=(255, 255, 0), width=1)
    for g in b["grounds"]:
        e = np.array(g["ext"]); d.line([P(q[0], -q[1]) for q in e] + [P(e[0][0], -e[0][1])], fill=(255, 0, 0), width=1)
for gu in range(int(np.ceil(u0 / 5)) * 5, int(u1) + 1, 5):
    d.line([((gu - u0) * S, 0), ((gu - u0) * S, (12 if gu % 10 == 0 else 6))], fill=(255, 255, 255), width=1)
    if gu % 10 == 0: d.text(((gu - u0) * S + 2, 12), str(gu), fill=(255, 255, 255))
for gv in range(int(np.ceil(v0 / 5)) * 5, int(v1) + 1, 5):
    d.line([(0, (v1 - gv) * S), ((12 if gv % 10 == 0 else 6), (v1 - gv) * S)], fill=(255, 255, 255), width=1)
    if gv % 10 == 0: d.text((14, (v1 - gv) * S - 5), str(gv), fill=(255, 255, 255))
im.save(out); print(out, im.size)

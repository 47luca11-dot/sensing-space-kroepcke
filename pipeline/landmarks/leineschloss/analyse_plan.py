"""LoD2-Grundriss + DOP10-Ausschnitt + Hoehenraster rund ums Leineschloss -> build/*.png, build/plan.json"""
import json, numpy as np, tifffile
from PIL import Image, ImageDraw
from ls_common import *
b = load_city_building()
# --- Grundriss / Dachflaechen
gr = [np.array(g["ext"]) for g in b["grounds"]]
allp = np.concatenate([np.array(r["ext"]) for r in b["roofs"]] + [np.array(w["ext"]) for w in b["walls"]])
print("x(E) %.1f..%.1f  N %.1f..%.1f  Z %.2f..%.2f" % (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max(), allp[:, 2].min(), allp[:, 2].max()))
print("grounds", len(gr), [len(g) for g in gr], "walls", len(b["walls"]), "roofs", len(b["roofs"]))
x0, x1 = allp[:, 0].min() - 25, allp[:, 0].max() + 25
z0, z1 = -allp[:, 1].max() - 25, -allp[:, 1].min() + 25
DOP = [RES + "/dop10/dop10rgb_32_549_5802_1_ni_2025-03-06.tif", RES + "/dop10/dop10rgb_32_550_5802_1_ni_2025-03-06.tif"]
UP = 0.25  # px per 10cm -> 2.5 px/m
S = 10 * UP
W = int((x1 - x0) * S); H = int((z1 - z0) * S)
canvas = np.zeros((H, W, 3), np.uint8)
for fn in DOP:
    with tifffile.TiffFile(fn) as t:
        p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
        sc = tags["ModelPixelScaleTag"]; tp = tags["ModelTiepointTag"]; img = p.asarray()[..., :3]
    # tile pixel (c,r) -> E = tp3 + c*sc0, N = tp4 - r*sc1
    for r in range(H):
        pass
    cs = ((E0 + x0 + (np.arange(W) + 0.5) / S) - tp[3]) / sc[0]
    rs = (tp[4] - (N0 - (z0 + (np.arange(H) + 0.5) / S))) / sc[1]
    ci = np.floor(cs).astype(int); ri = np.floor(rs).astype(int)
    okc = (ci >= 0) & (ci < img.shape[1]); okr = (ri >= 0) & (ri < img.shape[0])
    sub = img[np.clip(ri, 0, img.shape[0] - 1)][:, np.clip(ci, 0, img.shape[1] - 1)]
    m = okr[:, None] & okc[None, :]
    canvas[m] = sub[m]
im = Image.fromarray(canvas)
im.save(BUILD + "/dop.png")
d = ImageDraw.Draw(im)
def P(x, z): return ((x - x0) * S, (z - z0) * S)
for r in b["roofs"]:
    e = np.array(r["ext"])
    d.line([P(q[0], -q[1]) for q in e] + [P(e[0][0], -e[0][1])], fill=(255, 255, 0), width=1)
for g in gr:
    d.line([P(q[0], -q[1]) for q in g] + [P(g[0][0], -g[0][1])], fill=(255, 0, 0), width=2)
for gx in range(int(x0) // 10 * 10, int(x1) + 1, 10):
    d.line([P(gx, z0), P(gx, z0 + 3)], fill=(255, 255, 255), width=2); d.text(P(gx + 0.3, z0 + 3.5), str(gx), fill=(255, 255, 255))
for gz in range(int(z0) // 10 * 10, int(z1) + 1, 10):
    d.line([P(x0, gz), P(x0 + 3, gz)], fill=(255, 255, 255), width=2); d.text(P(x0 + 3.5, gz - 1), str(gz), fill=(255, 255, 255))
im.save(BUILD + "/dop_overlay.png")
json.dump({"x0": x0, "x1": x1, "z0": z0, "z1": z1, "S": S}, open(BUILD + "/dop_meta.json", "w"))
# --- Hoehenraster
hf = Heightfield()
xs = np.arange(x0, x1, 1.0); zs = np.arange(z0, z1, 1.0)
X, Z = np.meshgrid(xs, zs)
Hh = hf(X, Z)
print("heightfield y range %.2f..%.2f" % (Hh.min(), Hh.max()))
v = np.clip((Hh + 8) / 8, 0, 1)
Image.fromarray((v * 255).astype(np.uint8)).resize((W, H), Image.NEAREST).save(BUILD + "/hf.png")
# Punkte entlang der Grundrisse
for g in gr:
    hs = hf(g[:, 0], -g[:, 1])
    print("ground poly n=%d  Zabs-56.16 %.2f..%.2f  hf %.2f..%.2f" % (len(g), g[:, 2].min() - H0, g[:, 2].max() - H0, hs.min(), hs.max()))

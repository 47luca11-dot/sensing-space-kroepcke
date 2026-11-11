"""Bodenkacheln (data/ground_raw) im Lokalrahmen: Materialien als Farben, Wand-/Kanten-Hoehen -> build/ground_local.png"""
import trimesh, numpy as np, json
from PIL import Image, ImageDraw
from ls_common import *
u0, v0, u1, v1, S = -125, -95, 60, 15, 6
W = int((u1 - u0) * S); H = int((v1 - v0) * S)
im = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(im)
def P(u, v): return ((u - u0) * S, (v1 - v) * S)
col = {}
rng = np.random.default_rng(3)
legend = {}
for t in ['ground_-5_3', 'ground_-5_4', 'ground_-4_3', 'ground_-4_4', 'ground_-6_3', 'ground_-6_4', 'ground_-5_5', 'ground_-4_5']:
    try: s = trimesh.load(ROOT + '/data/ground_raw/%s.glb' % t)
    except Exception: continue
    for n, g in s.geometry.items():
        mat = getattr(g.visual, 'material', None); mn = mat.name if mat is not None else n
        c = col.setdefault(mn, tuple(int(x) for x in rng.integers(60, 255, 3)))
        legend[mn] = c
        v = g.vertices; F = g.faces
        u, vv = game_to_local(v[:, 0], v[:, 2])
        nrm_ = g.face_normals
        for f, nn in zip(F, nrm_):
            if not ((u[f] > u0).any() and (u[f] < u1).any() and (vv[f] > v0).any() and (vv[f] < v1).any()): continue
            pts = [P(u[i], vv[i]) for i in f]
            if abs(nn[1]) < 0.3:  # Wand
                d.line(pts + [pts[0]], fill=(255, 255, 255), width=1)
            else:
                d.polygon(pts, fill=c)
b = load_city_building()
for gr in b["grounds"]:
    e = np.array(gr["ext"]); u, vv = game_to_local(e[:, 0], -e[:, 1])
    d.line([P(a, c) for a, c in zip(u, vv)] + [P(u[0], vv[0])], fill=(255, 0, 0), width=2)
for gu in range(-120, 61, 10):
    d.text(((gu - u0) * S + 2, 2), str(gu), fill=(255, 255, 255)); d.line([((gu - u0) * S, 0), ((gu - u0) * S, 8)], fill=(255, 255, 255))
for gv in range(-90, 15, 10):
    d.text((2, (v1 - gv) * S - 5), str(gv), fill=(255, 255, 255)); d.line([(0, (v1 - gv) * S), (8, (v1 - gv) * S)], fill=(255, 255, 255))
y = 20
for k, c in legend.items():
    d.rectangle([W - 120, y, W - 110, y + 8], fill=c); d.text((W - 105, y - 2), k, fill=(255, 255, 255)); y += 12
im.save(BUILD + "/ground_local.png"); print(im.size)

"""Ballhof site analysis: city LoD2 around Ballhofplatz (game frame), DOP10 crop with LoD2/OSM overlay,
heightfield excerpt. Writes build/site_*.png/json. Run: python site.py"""
import json, os, numpy as np, tifffile
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
RES = 'C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research/'
E0, N0, H0 = 550298.88274216, 5802940.705715501, 56.16
x0, x1, z0, z1 = -500.0, -340.0, 140.0, 320.0
os.makedirs(os.path.join(HERE, 'build'), exist_ok=True)
L = json.load(open(ROOT + 'data/lod2_buildings.json', encoding='utf8'))
sel = []
for b in L['buildings']:
    c = b['centre']
    if x0 - 20 < c[0] < x1 + 20 and z0 - 20 < -c[1] < z1 + 20:
        sel.append(b)
f = RES + 'dop10/dop10rgb_32_549_5802_1_ni_2025-03-06.tif'
with tifffile.TiffFile(f) as t:
    p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
    sc = tags['ModelPixelScaleTag']; tp = tags['ModelTiepointTag']; img = p.asarray()
E_a, E_b = E0 + x0, E0 + x1; N_a, N_b = N0 - z1, N0 - z0
c0 = int(round((E_a - tp[3]) / sc[0])); c1 = int(round((E_b - tp[3]) / sc[0])); r0 = int(round((tp[4] - N_b) / sc[1])); r1 = int(round((tp[4] - N_a) / sc[1]))
print('tile origin', tp[3], tp[4], 'scale', sc[:2], 'crop', r0, r1, c0, c1, img.shape)
a = img[max(r0, 0):r1, max(c0, 0):c1, :3]
UP = 1; S = UP / sc[0]
im = Image.fromarray(a).resize((a.shape[1] * UP, a.shape[0] * UP), Image.LANCZOS)
im.save(os.path.join(HERE, 'build', 'site_dop.png'))
d = ImageDraw.Draw(im)
P = lambda x, z: ((x - x0) * S, (z - z0) * S)
for b in sel:
    for r in b['roofs']:
        d.line([P(q[0], -q[1]) for q in r['ext']] + [P(r['ext'][0][0], -r['ext'][0][1])], fill=(255, 255, 0), width=1)
    for g in b['grounds']:
        d.line([P(q[0], -q[1]) for q in g['ext']] + [P(g['ext'][0][0], -g['ext'][0][1])], fill=(255, 0, 0), width=2)
    c = b['centre']; d.text(P(c[0] - 4, -c[1]), b['id'][3:9], fill=(0, 255, 255))
for gx in range(int(x0), int(x1) + 1, 10):
    d.line([P(gx, z0), P(gx, z0 + (3 if gx % 50 else 8))], fill=(255, 255, 255), width=2)
    if gx % 20 == 0: d.text(P(gx + 0.3, z0 + 4), str(gx), fill=(255, 255, 255))
for gz in range(int(z0), int(z1) + 1, 10):
    d.line([P(x0, gz), P(x0 + (3 if gz % 50 else 8), gz)], fill=(255, 255, 255), width=2)
    if gz % 20 == 0: d.text(P(x0 + 9, gz - 5), str(gz), fill=(255, 255, 255))
im.save(os.path.join(HERE, 'build', 'site_dop_overlay.png'))
# OSM buildings
O = json.load(open(ROOT + 'data/osm_features.json', encoding='utf8'))
for w in O['ways']:
    t = w['tags']
    if not ('building' in t or 'building:part' in t or 'amenity' in t or 'name' in t): continue
    pts = np.array(w['pts'])
    cx, cz = pts.mean(0)
    if x0 < cx < x1 and z0 < cz < z1:
        print('W', w['id'], 'c=(%.1f,%.1f)' % (cx, cz), {k: v for k, v in t.items() if k in ('building', 'building:part', 'name', 'addr:street', 'addr:housenumber', 'building:levels', 'roof:shape', 'height', 'amenity', 'highway', 'heritage', 'start_date', 'roof:levels', 'building:material', 'leisure', 'place', 'area', 'tourism')})
for n in O['nodes']:
    cx, cz = n['p']
    if x0 < cx < x1 and z0 < cz < z1 and ('name' in n['tags'] or 'amenity' in n['tags'] or 'shop' in n['tags'] or 'historic' in n['tags'] or 'tourism' in n['tags']):
        print('N', n['id'], 'p=(%.1f,%.1f)' % (cx, cz), {k: v for k, v in n['tags'].items() if k in ('name', 'amenity', 'shop', 'historic', 'addr:street', 'addr:housenumber', 'tourism', 'natural', 'highway', 'level')})

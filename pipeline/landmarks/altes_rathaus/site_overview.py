"""Overview of the Altes Rathaus site: DOP10 crop + LGLN LoD2 parts + OSM ways (game frame x east, z south).
usage: python site_overview.py <dop.tif> <lgln_parts.json> <osm_features.json> <out_prefix> [x0 x1 z0 z1 scale]"""
import sys, json, numpy as np, tifffile
from PIL import Image, ImageDraw
DOP, PARTS, OSM, OUT = sys.argv[1:5]
x0, x1, z0, z1, SC = [float(v) for v in sys.argv[5:10]] if len(sys.argv) > 9 else (-240, -150, 310, 410, 8)
E0, N0 = 550298.88274216, 5802940.705715501
with tifffile.TiffFile(DOP) as t:
    p = t.pages[0]; tags = {k.name: k.value for k in p.tags.values()}
    sc = tags['ModelPixelScaleTag']; tp = tags['ModelTiepointTag']; img = p.asarray()
c0 = int((E0 + x0 - tp[3]) / sc[0]); c1 = int((E0 + x1 - tp[3]) / sc[0])
r0 = int((tp[4] - (N0 - z0)) / sc[1]); r1 = int((tp[4] - (N0 - z1)) / sc[1])
a = img[r0:r1, c0:c1, :3]
S = SC
im = Image.fromarray(a).resize((int((x1 - x0) * S), int((z1 - z0) * S)), Image.LANCZOS)
im.save(OUT + '_dop.png')
d = ImageDraw.Draw(im)
def P(x, z): return ((x - x0) * S, (z - z0) * S)
parts = json.load(open(PARTS))
cols = [(255, 60, 60), (60, 160, 255), (255, 255, 0), (0, 255, 120), (255, 0, 255), (0, 255, 255), (255, 140, 0)]
for i, r in enumerate(parts):
    if not r['building'].endswith('00009VAT'): continue
    c = cols[i % len(cols)]
    for g in r['surf']['ground']:
        d.line([P(q[0], q[2]) for q in g] + [P(g[0][0], g[0][2])], fill=c, width=2)
    allp = np.array([q for g in r['surf']['ground'] for q in g])
    d.text(P(allp[:, 0].mean(), allp[:, 2].mean()), r['part'][-3:] + ' ' + r.get('roofType', '') + ' ' + r['measuredHeight'][:4], fill=c)
osm = json.load(open(OSM, encoding='utf8'))
for w in osm['ways']:
    t = w['tags']
    pts = w['pts']
    if not any(x0 - 20 < p[0] < x1 + 20 and z0 - 20 < p[1] < z1 + 20 for p in pts): continue
    if 'highway' in t:
        d.line([P(*p) for p in pts], fill=(255, 255, 255), width=1)
        if 'name' in t:
            m = pts[len(pts) // 2]; d.text(P(*m), t['name'], fill=(255, 255, 255))
    elif 'building' in t or 'building:part' in t:
        d.line([P(*p) for p in pts], fill=(160, 160, 160), width=1)
for gx in range(int(x0), int(x1) + 1, 10):
    d.line([P(gx, z0), P(gx, z0 + 2)], fill=(255, 255, 255)); d.text(P(gx, z0 + 2), str(gx), fill=(255, 255, 255))
for gz in range(int(z0), int(z1) + 1, 10):
    d.line([P(x0, gz), P(x0 + 2, gz)], fill=(255, 255, 255)); d.text(P(x0 + 2, gz), str(gz), fill=(255, 255, 255))
im.save(OUT + '_overlay.png')
print('ok', im.size)

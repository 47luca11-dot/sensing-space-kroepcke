"""LGLN-LoD2-Teile im Lokalrahmen auf DOP zeichnen + Liste (u,v-Grundriss, Hoehen)"""
import json, numpy as np
from PIL import Image, ImageDraw
from ls_common import *
parts = json.load(open(BUILD + "/lgln_parts.json"))
u0, v0, u1, v1, S = -125, -90, 55, 12, 5
im = Image.open(BUILD + "/local_all.png").convert("RGB")
SC = 2; im = im.resize((im.width * SC, im.height * SC)); S *= SC
d = ImageDraw.Draw(im)
def P(u, v): return ((u - u0) * S, (v1 - v) * S)
cols = {'1000': (255, 255, 0), '2100': (255, 128, 0), '3100': (255, 60, 60), '3200': (60, 255, 60), '5000': (60, 160, 255), '9999': (255, 0, 255)}
for r in parts:
    if not r['building'].endswith('00009Sif'): continue
    c = cols.get(r.get('roofType'), (255, 255, 255))
    for g in r['surf']['ground']:
        g = np.array(g); u, v = game_to_local(g[:, 0], g[:, 2])
        d.line([P(a, b) for a, b in zip(u, v)] + [P(u[0], v[0])], fill=c, width=2)
        print(r['part'][-4:], r.get('roofType'), 'top %.2f' % max(q[1] for s in r['surf']['roof'] for q in s), ' '.join('(%.1f,%.1f)' % (a, b) for a, b in zip(u, v)))
        d.text(P(u.mean(), v.mean()), r['part'][-3:] + ' ' + r.get('roofType', ''), fill=c)
    for rf in r['surf']['roof']:
        g = np.array(rf); u, v = game_to_local(g[:, 0], g[:, 2])
        d.line([P(a, b) for a, b in zip(u, v)] + [P(u[0], v[0])], fill=(200, 200, 200), width=1)
im.save(BUILD + "/parts_local.png")

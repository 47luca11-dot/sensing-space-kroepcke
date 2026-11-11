"""Extract the city LoD2 Marktkirche building (+ neighbours) from data/lod2_buildings.json -> mk_lod2.json (Blender frame: x east, y north, z abs)."""
import json, sys, numpy as np
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
d = json.load(open(ROOT + "/data/lod2_buildings.json", encoding='utf-8'))
out = []
for b in d['buildings']:
    cx, cy = b['centre']
    if abs(cx + 245) < 75 and abs(cy + 295) < 60:
        out.append(b)
json.dump(out, open('mk_lod2.json', 'w'))
for b in out:
    zs = [p[2] for w in b['walls'] + b['roofs'] for p in w['ext']]
    print(b['id'], b['centre'], b['height'], b['street'], b['number'], round(max(zs), 2))

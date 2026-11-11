"""Which detail_raw geometry lies inside the Cafe Kroepcke footprint? (game frame: x east, y up, z south)"""
import json, sys, glob, os, numpy as np, trimesh
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import vectorized
D = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/detail_raw/"
parts = [p for p in json.load(open(sys.argv[1])) if p['building'].endswith('00009TB7')]
fp = unary_union([Polygon(np.array(p['surf']['ground'][0])[:, [0, 2]]).buffer(0) for p in parts])
print('footprint area', round(fp.area, 1), 'bounds', [round(b, 2) for b in fp.bounds])
big = fp.buffer(1.0)
for fn in sorted(glob.glob(D + '*.glb')):
    sc = trimesh.load(fn)
    for name, g in sc.geometry.items():
        # apply node transform
        T = sc.graph.get(sc.graph.nodes_geometry[[sc.graph[n][1] for n in sc.graph.nodes_geometry].index(name)])[0] if False else None
        v = g.vertices; f = g.faces
        c = v[f].mean(1)
        inside = vectorized.contains(big, c[:, 0], c[:, 2])
        if inside.sum() == 0: continue
        ci = c[inside]
        print(os.path.basename(fn), name[:50], 'faces in', inside.sum(), 'of', len(f), 'y %.2f..%.2f' % (ci[:, 1].min(), ci[:, 1].max()))

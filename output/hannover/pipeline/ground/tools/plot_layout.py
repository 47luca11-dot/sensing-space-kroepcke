"""Diagnose: Klassenplan plotten.  python tools/plot_layout.py out.png xmin xmax zmin zmax [dpi]"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import *
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
COL = {'road': '#444', 'gutter': '#666', 'kerb': '#ddd', 'kerb_ch': '#fff', 'sidewalk': '#b8b2a8', 'ped': '#d8cfa8', 'setts_road': '#7a6a5a',
       'track': '#8a6a4a', 'parking': '#555a66', 'grass': '#6a6', 'bed': '#5a3a1a', 'edge': '#eee', 'treepit': '#222', 'pitframe': '#999',
       'water': '#39f', 'rim': '#ccc', 'hbs': '#c88', 'coping': '#fa0', 'cycle': '#c44', 'path': '#bba', 'gravel': '#dc9', 'railplat': '#aaa',
       'ballast': '#865', 'island': '#ccb', 'seam': '#f0f'}
def patch(g, c, ax):
    for p in getattr(g, 'geoms', [g]):
        if p.is_empty or p.geom_type != 'Polygon': continue
        verts = []; codes = []
        for r in [p.exterior] + list(p.interiors):
            a = np.array(r.coords); verts += a.tolist(); codes += [Path.MOVETO] + [Path.LINETO] * (len(a) - 2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), fc=c, ec='none', lw=0))
x0, x1, z0, z1 = [float(v) for v in sys.argv[2:6]]
dpi = int(sys.argv[6]) if len(sys.argv) > 6 else 80
fig, ax = plt.subplots(figsize=(16, 16 * (z1 - z0) / (x1 - x0)))
from shapely.geometry import box
B = box(x0, z0, x1, z1)
for k, g in L['classes'].items():
    patch(g.intersection(B), COL.get(k, '#f00'), ax)
patch(L['bld'].intersection(B), (0.2, 0.2, 0.8, 0.25), ax)
for h in L['holes'].values(): patch(h['poly'], '#000', ax)
for c, w, t in L['road_lines']:
    a = np.array(c); ax.plot(a[:, 0], a[:, 1], 'y-', lw=0.4)
for n in L['trees']: ax.plot(*n['p'], 'g.', ms=3)
ax.plot([SQ[0], SQ[2], SQ[2], SQ[0], SQ[0]], [SQ[1], SQ[1], SQ[3], SQ[3], SQ[1]], 'r-', lw=0.6)
ax.set_xlim(x0, x1); ax.set_ylim(z1, z0); ax.set_aspect('equal')
plt.savefig(sys.argv[1], dpi=dpi, bbox_inches='tight')

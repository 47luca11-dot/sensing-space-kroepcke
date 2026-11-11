"""Sample pavement height (street+ground layers of detail_raw) around the Cafe footprint + list old materials inside."""
import json, sys, numpy as np, trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import contains_xy
D = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/detail_raw/"
parts = [p for p in json.load(open(sys.argv[1])) if p['building'].endswith('00009TB7')]
fp = unary_union([Polygon(np.array(p['surf']['ground'][0])[:, [0, 2]]).buffer(0) for p in parts])
ms = []
for n in ('street', 'ground'):
    s = trimesh.load(D + n + '.glb'); ms += list(s.dump()) if hasattr(s, 'dump') else [s]
M = trimesh.util.concatenate(ms)
rmi = trimesh.ray.ray_triangle.RayMeshIntersector(M)
def h(x, z):
    o = np.array([[x, 50, z]]); d = np.array([[0, -1, 0]])
    loc, ri, ti = rmi.intersects_location(o, d)
    return loc[:, 1].max() if len(loc) else np.nan
ring = fp.buffer(1.5).exterior
out = []
for t in np.linspace(0, ring.length, 120, endpoint=False):
    p = ring.interpolate(t); out.append((p.x, p.y, h(p.x, p.y)))
out = np.array(out)
np.set_printoptions(suppress=True, linewidth=200)
print('ring heights: min %.3f max %.3f' % (np.nanmin(out[:, 2]), np.nanmax(out[:, 2])))
for r in out[::4]: print('  x %.1f z %.1f  y %.3f' % tuple(r))
inner = fp.buffer(-1).representative_point(); print('inside', h(inner.x, inner.y))
json.dump(out.tolist(), open(sys.argv[2], 'w'))
# materials of old building geometry inside footprint (0.3 m buffer)
B = fp.buffer(0.3)
for n in ('bldg_walls', 'bldg_openings', 'bldg_roof', 'bldg_base', 'fix', 'street_furniture', 'kiosk', 'landmark_tiefgarage_pavillon'):
    s = trimesh.load(D + n + '.glb')
    for g in (s.dump() if hasattr(s, 'dump') else [s]):
        c = g.triangles_center; ins = contains_xy(B, c[:, 0], c[:, 2])
        if ins.sum():
            mn = getattr(g.visual, 'material', None); mn = getattr(mn, 'name', '?')
            print(n, mn, ins.sum(), 'y %.2f..%.2f' % (c[ins, 1].min(), c[ins, 1].max()))

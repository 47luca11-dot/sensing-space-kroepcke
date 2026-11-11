"""Pruefung des exportierten GLB: Knoten, Dreiecke, Grenzen, offene Kanten je Knoten, Normalen-Stichprobe (nach aussen?)."""
import sys, numpy as np, trimesh
from collections import Counter

s = trimesh.load(sys.argv[1])
tot = 0
print('%-52s %8s %7s %s' % ('node', 'tris', 'open', 'bounds y'))
for name in s.graph.nodes_geometry:
    T, g = s.graph[name]; geo = s.geometry[g]
    m = trimesh.Trimesh(trimesh.transform_points(geo.vertices, T), geo.faces, process=False)
    m.merge_vertices(digits_vertex=4)
    e = np.sort(m.edges, axis=1); c = Counter(map(tuple, e))
    open_e = sum(1 for v in c.values() if v == 1)
    tot += len(geo.faces)
    print('%-52s %8d %7d %.2f..%.2f' % (name[:52], len(geo.faces), open_e, m.bounds[0][1], m.bounds[1][1]))
print('TOTAL tris', tot)
b = s.bounds
print('scene bounds x %.2f..%.2f y %.2f..%.2f z %.2f..%.2f' % (b[0][0], b[1][0], b[0][1], b[1][1], b[0][2], b[1][2]))
# Tonnenschale (Kupfer + Decke_innen) wasserdicht?
parts = []
for name in s.graph.nodes_geometry:
    if 'Kupfer' in name and 'Falz' not in name or 'Decke_innen' in name:
        T, g = s.graph[name]; geo = s.geometry[g]
        parts.append(trimesh.Trimesh(trimesh.transform_points(geo.vertices, T), geo.faces, process=False))
sh = trimesh.util.concatenate(parts); sh.merge_vertices(digits_vertex=3)
e = np.sort(sh.edges, axis=1); c = Counter(map(tuple, e))
print('vault shell: faces', len(sh.faces), 'edges !=2:', sum(1 for v in c.values() if v != 2), 'watertight:', sh.is_watertight, 'volume %.1f m3' % sh.volume)

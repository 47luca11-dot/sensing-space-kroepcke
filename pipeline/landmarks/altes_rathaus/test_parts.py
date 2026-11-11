"""Bauteiltest: eine kleine Fassade mit Oeffnungen, Fialen, Baendern, Dachflaeche -> build/test.glb"""
import sys, os, json
import numpy as np
from shapely.geometry import Polygon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ar_geom import Frame, MB
from ar_parts import *
from ar_export import export, tiles_for_mats

T = tiles_for_mats()
mb = MB(T)
fr = Frame.facade((0, 0), (12, 0))           # Fassade entlang +x, Normale -z (Norden)
ops = [Opening(2.0, 5.2, 1.25, 1.55, 'point', rise=0.9, ring={'w': 0.13, 'proj': 0.015}, tymp={'n': 2}, lights=2, bars_h=0.4),
       Opening(5.0, 1.8, 2.1, 1.8, 'point', rise=1.4, ring={'w': 0.30, 'proj': 0.02}, lights=4, bars_h=0.45, orders=((0, 0.10), (0.1, 0.22), (0.2, 0.36))),
       Opening(8.5, 0.0, 2.4, 2.8, 'point', rise=1.6, ring={'w': 0.35, 'proj': 0.02}, content='shop', skip_bottom=True, orders=((0, 0.25), (0.12, 0.6))),
       Opening(10.8, 2.0, 1.4, 1.4, 'seg', rise=0.25, ring={'w': 0.12, 'proj': 0.0}, lights=3, bars_h=0.35)]
cuts = [build_opening(mb, fr, op, protos=True) for op in ops]
cuts += [[(0, 4.3), (12, 4.3), (12, 4.85), (0, 4.85)], [(0, 7.8), (12, 7.8), (12, 8.2), (0, 8.2)]]
wall_face(mb, fr, [(0, -0.5), (12, -0.5), (12, 9.4), (0, 9.4)], cuts)
frieze_band(mb, fr, 0, 12, 4.3, 0.55)
lattice_band(mb, fr, 0, 12, 7.8, 8.2)
for a in (0.4, 3.6, 7.0, 11.6):
    pinnacle_shaft(mb, fr, a, 9.4, 12.5, r=0.36, crown_scale=0.36)
sweep(mb, SS, [(-0.2, 0), (12.2, 0)], [(0, -0.5), (0.08, -0.5), (0.08, 0.45), (0.0, 0.55)])
gutter(mb, [(0, -0.05), (12, -0.05)], 9.5)
downpipe(mb, fr, 6.2, 0.2, 9.4)
roof_plane(mb, Polygon([(0, -0.3), (12, -0.3), (12, 6), (0, 6)]), (0, -0.3), (1, 0), 58, 9.6)
protos = {'rosette': proto_rosette(), 'crown': proto_crown(), 'frieze': proto_frieze(np.load(os.path.join(HERE, 'build/tex/frieze_height.npy')))}
for p in protos.values(): p.tiles = T
st = {}
n = export(os.path.join(HERE, 'build/test.glb'), mb, protos, tex_dir=os.path.join(HERE, 'build/tex'), stats=st)
print('tris', n, st['groups'])

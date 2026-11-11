"""Lage des TIB-Rhino-Modells pruefen: Nachbarbauten (aus LH-LoD2) vs. CityGML-Kachel 5480_5803 (ICP 2D)"""
import numpy as np, trimesh, re
from lxml import etree
from scipy.spatial import cKDTree
E0, N0 = 548986.0, 5803674.0
m = trimesh.load('tib_export/umgebung_nachbarbauten.glb', force='scene').to_geometry()
V = m.vertices                       # glTF: x=X, y=Z, z=-Y
A = np.c_[V[:, 0], -V[:, 2]]; Az = V[:, 1]
print('nachbar local XY bbox', A.min(0).round(1), A.max(0).round(1), 'Z', Az.min().round(2), Az.max().round(2))
pts = []; zs = []
for _, el in etree.iterparse('citygml/5480_5803.gml', tag='{http://www.opengis.net/gml}posList', huge_tree=True):
    v = np.array(el.text.split(), float).reshape(-1, 3); el.clear()
    v = v[(np.abs(v[:, 0] - E0 - 40) < 300) & (np.abs(v[:, 1] - N0 - 30) < 300)]
    if len(v): pts.append(v[:, :2] - [E0, N0]); zs.append(v[:, 2])
B = np.vstack(pts); Bz = np.concatenate(zs)
print('citygml local bbox', B.min(0).round(1), B.max(0).round(1), len(B))
# ICP (2D rigid) von A auf B, nur Punkte mit Hoehe > 2 m ueber Grund vermeiden -> alle
A2 = np.unique(np.round(A, 2), axis=0); A2 = A2[(np.abs(A2[:, 0] - 40) < 250) & (np.abs(A2[:, 1] - 30) < 250)]
T = cKDTree(B); R = np.eye(2); t = np.zeros(2)
for it in range(40):
    P = A2 @ R.T + t; d, i = T.query(P); k = d < np.percentile(d, 70)
    p, q = P[k], B[i[k]]; pc, qc = p.mean(0), q.mean(0)
    U, S, Vt = np.linalg.svd((p - pc).T @ (q - qc)); r = Vt.T @ U.T
    if np.linalg.det(r) < 0: Vt[1] *= -1; r = Vt.T @ U.T
    R = r @ R; t = r @ t + qc - r @ pc
d, _ = T.query(A2 @ R.T + t)
print('ICP rot deg %.3f  t %s  median residual %.3f m  (before: %.3f)' % (np.degrees(np.arctan2(R[1, 0], R[0, 0])), t.round(3), np.median(d), np.median(T.query(A2)[0])))

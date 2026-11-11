"""Debug helper: list the triangles (object, material, plane offset, building-frame coords) near a world point
with a given normal.  python zf_debug.py x y z nx ny nz [radius]"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anz
meta = json.load(open(os.path.join(HERE, 'build', 'anzeiger_meta.json')))
npz = np.load(os.path.join(HERE, 'build', 'anzeiger_parts.npz'))
p = np.array([float(a) for a in sys.argv[1:4]]); nq = np.array([float(a) for a in sys.argv[4:7]]); nq /= np.linalg.norm(nq)
R = float(sys.argv[7]) if len(sys.argv) > 7 else 0.6
FE = anz.FE
for o, od in meta['objects'].items():
    V = npz['V_' + o].astype(float).reshape(-1, 3, 3); M = npz['M_' + o]
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]); A = np.linalg.norm(n, axis=1); n /= np.maximum(A, 1e-12)[:, None]
    c = V.mean(1)
    lo = V.min(1) - R; hi = V.max(1) + R
    sel = (lo <= p).all(1) & (hi >= p).all(1) & (n @ nq > 0.98) & (np.abs(np.einsum("ij,ij->i", n, V[:, 0]) - n @ p) < 0.02)
    for i in np.where(sel)[0]:
        d = n[i] @ V[i, 0]
        loc = [(round(float((v - FE.O) @ FE.s - anz.U_S), 3), round(float(v[1]), 3), round(float((v - FE.O) @ FE.n - anz.EW), 3)) for v in V[i]]
        print(o, od['materials'][M[i]], 'd=%.4f' % d, 'area %.4f' % (A[i] / 2), 'sye', loc)

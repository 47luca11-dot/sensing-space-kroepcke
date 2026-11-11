"""Ground contact check: upward-facing landmark faces (thresholds, landings, steps, slabs) must be clearly above the
ground mesh (>= 3 cm at every vertex, sampled from the heightfield bilinearly) or clearly buried (> 10 cm under it).
Anything in between flickers against the ground tile or is half covered by it.
Run: python ground_check.py   (reads build/anzeiger_parts.npz)"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/'
meta = json.load(open(os.path.join(HERE, 'build', 'anzeiger_meta.json')))
npz = np.load(os.path.join(HERE, 'build', 'anzeiger_parts.npz'))
hj = json.load(open(ROOT + 'web/assets/city/ground/heightfield.json'))
hf = np.fromfile(ROOT + 'web/assets/city/ground/heightfield.bin', dtype=np.float32).reshape(hj['h'], hj['w'])
def H(x, z):
    fx = (x - hj['x0']) / hj['dx']; fz = (z - hj['z0']) / hj['dz']; i = np.floor(fx).astype(int); j = np.floor(fz).astype(int); a = fx - i; b = fz - j
    return (1 - a) * (1 - b) * hf[j, i] + a * (1 - b) * hf[j, i + 1] + (1 - a) * b * hf[j + 1, i] + a * b * hf[j + 1, i + 1]
bad = []
for o, od in meta['objects'].items():
    T = npz['V_' + o].astype(np.float64).reshape(-1, 3, 3); M = npz['M_' + o]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); A = np.linalg.norm(n, axis=1) / 2; n /= np.maximum(2 * A, 1e-12)[:, None]
    up = (n[:, 1] > 0.98) & (A > 1e-4) & (T[:, :, 1].min(1) < 2.0)
    for i in np.where(up)[0]:
        P = np.vstack([T[i], T[i].mean(0)[None]])
        dz = P[:, 1] - H(P[:, 0], P[:, 2])
        if dz.min() < 0.03 and dz.max() > -0.10:
            bad.append((o, od['materials'][M[i]], np.round(T[i].mean(0), 2).tolist(), round(float(dz.min()), 3), round(float(A[i]), 3)))
print('GROUND upward faces too close to / partly under the ground: %d (%.2f m2)' % (len(bad), sum(b[4] for b in bad)))
for b in sorted(bad, key=lambda r: -r[4])[:20]: print('  ', b)
sys.exit(1 if bad else 0)

# Aufruf (aus output/hannover): uv run --no-project --python 3.11 --with numpy python pipeline/scans/steintor_eingang.py
# Danach: blender -b --factory-startup -P pipeline/pointcloud/run_headless.py   (Kacheln + Licht neu; der Scan ersetzt dort die Modellpunkte)
# Steintor-Eingangsscan (Gaussian Splats) -> Punktwolken-Zwischendatei data/pointcloud/stage/scan__steintor_eingang.npz (Spielkoordinaten)
import sys, json, numpy as np
import os
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".claude", "skills", "figur-einbauen", "scripts"))
import splat_werkzeug as SW
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "sources", "scans", "steintor_eingang.ply")
out_npz = os.path.join(REPO, "data", "pointcloud", "stage", "scan__steintor_eingang.npz"); out_json = os.path.join(REPO, "data", "pointcloud", "scan_steintor_eingang.json")
TOP = np.array([-390.8, -123.25]); DIR = np.array([-0.95, -0.31]); DIR /= np.linalg.norm(DIR); Y_TOP = -1.04
dy, drot = 0.0, 0.0
props, a = SW.lesen(src); P = SW.pos(props, a); op = SW.deckkraft(props, a); C = SW.farbe(props, a)
c0 = np.array([1.88, -0.03, -0.31]); BODEN = 0.02
# Scan-Achsen (alter Rahmen x,z): Treppe fällt Richtung +s
ax = np.array([-0.336, 0.942]); pe = np.array([-ax[1], ax[0]]); cen = np.array([0.23, -0.59])
Q = P - c0; XZ = Q[:, [0, 2]]; s = (XZ - cen) @ ax; t = (XZ - cen) @ pe; hh = -Q[:, 1] - BODEN
keep = (op > 0.5) & (s > -8.5) & (s < 6.5) & (np.abs(t + 0.3) < 4.2) & (hh > -5.5) & (hh < 3.2)
# Schwebeteilchen weg: 10-cm-Voxel mit < 4 Punkten
vk = np.floor(np.stack([s, t, hh], 1) / 0.1).astype(np.int64); key = (vk[:, 0] + 500) * 1_000_000 + (vk[:, 1] + 500) * 1000 + (vk[:, 2] + 500)
u, inv, cnt = np.unique(key[keep], return_inverse=True, return_counts=True)
kk = np.where(keep)[0]; keep[kk[cnt[inv] < 6]] = False
# Anker: Treppenbeginn (s=-2.5) in der Mitte der Treppenbreite (t=-1.0)
A = cen + (-2.5) * ax + (-1.0) * pe                              # Treppenbeginn -> oberes Podest des Spiel-Eingangs (0,6 m hinter dem OSM-Knoten)
# lokal rechtshändig (wie Spiel): x, oben = -y, z' = -z
loc = np.stack([Q[:, 0] - A[0], hh, -(Q[:, 2] - A[1])], 1)[keep]
v = np.array([ax[0], -ax[1]]); w = DIR
phi = np.arctan2(w[1], w[0]) - np.arctan2(v[1], v[0]) + np.radians(drot)
cs, sn = np.cos(phi), np.sin(phi); R2 = np.array([[cs, -sn], [sn, cs]])
xz = loc[:, [0, 2]] @ R2.T + TOP + 0.6 * DIR
G = np.stack([xz[:, 0], loc[:, 1] + Y_TOP + dy, xz[:, 1]], 1).astype(np.float32)
# Normale = kürzeste Splat-Achse
rq = a[:, SW.spalten(props, "rot_0", "rot_1", "rot_2", "rot_3")][keep].astype(np.float64); rq /= np.linalg.norm(rq, axis=1, keepdims=True)
sc = a[:, SW.spalten(props, "scale_0", "scale_1", "scale_2")][keep]; ii = np.argmin(sc, 1)
wq, xq, yq, zq = rq.T
Rm = np.stack([np.stack([1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - wq * zq), 2 * (xq * zq + wq * yq)], 1),
               np.stack([2 * (xq * yq + wq * zq), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - wq * xq)], 1),
               np.stack([2 * (xq * zq - wq * yq), 2 * (yq * zq + wq * xq), 1 - 2 * (xq * xq + yq * yq)], 1)], 1)
n = Rm[np.arange(len(ii)), :, ii]
n = np.stack([n[:, 0], -n[:, 1], -n[:, 2]], 1); nxz = n[:, [0, 2]] @ R2.T; n = np.stack([nxz[:, 0], n[:, 1], nxz[:, 1]], 1)
n[n[:, 1] < 0] *= -1                                            # zweiseitig: nach oben drehen
def oct_encode(n):
    n = n / np.maximum(np.abs(n).sum(1, keepdims=True), 1e-9); x, y, z = n[:, 0], n[:, 1], n[:, 2]; neg = z < 0
    ox = np.where(neg, (1 - np.abs(y)) * np.sign(x + 1e-12), x); oy = np.where(neg, (1 - np.abs(x)) * np.sign(y + 1e-12), y)
    return np.clip(np.round((np.stack([ox, oy], 1) * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
# Farbe = Albedo; Sonne/Schatten/AO backt bl_light.py wie bei allen Punkten. Flags: zweiseitig (1) + fein (8).
# Nicht als leuchtend (2) markieren – dann wirkte der Scan im Spiel fast weiß gegenüber der gebackenen Umgebung.
srgb = np.clip(C[keep], 0, 1)
Cc = np.zeros((len(G), 4), np.uint8); Cc[:, :3] = np.clip(np.round(srgb * 255), 0, 255); Cc[:, 3] = 1 | 8
np.savez(out_npz, P=G, C=Cc, N=oct_encode(n))
json.dump(dict(quelle="Steintor Eingang.ply (Gaussian-Splat-Scan, 875k Splats)", punkte=int(len(G)), anker_top=TOP.tolist(), richtung=DIR.tolist(),
               y_top=Y_TOP, drehung_grad=round(float(np.degrees(phi)), 2), bbox=[G.min(0).round(2).tolist(), G.max(0).round(2).tolist()]), open(out_json, "w"), indent=1)
print("Punkte", len(G), "bbox", G.min(0).round(2), G.max(0).round(2), "phi", round(float(np.degrees(phi)), 1))

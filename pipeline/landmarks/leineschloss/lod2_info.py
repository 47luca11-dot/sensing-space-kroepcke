"""LoD2-Analyse: Hauptrichtung, Dachflaechen (Neigung, Hoehe) -> build/lod2_roofs.png + Konsolenliste"""
import json, numpy as np
from PIL import Image, ImageDraw
from ls_common import *
b = load_city_building()
# Wandrichtungen
hist = np.zeros(180)
for w in b["walls"]:
    e = np.array(w["ext"])
    for i in range(len(e)):
        p, q = e[i], e[(i + 1) % len(e)]
        if abs(p[2] - q[2]) > 0.01: continue
        dx, dy = q[0] - p[0], q[1] - p[1]; L = np.hypot(dx, dy)
        if L < 0.5: continue
        a = np.degrees(np.arctan2(dy, dx)) % 90
        hist[int(a * 2) % 180] += L
k = np.argmax(hist); print("dominant wall angle mod 90 (deg, E->N ccw):", k / 2, "weight", hist[k])
top = np.argsort(hist)[::-1][:8]; print([(t / 2, round(hist[t])) for t in top])
meta = json.load(open(BUILD + "/dop_meta.json"))
im = Image.open(BUILD + "/dop.png").convert("RGB")
# hochskalieren
SC = 3
im = im.resize((im.width * SC, im.height * SC), Image.LANCZOS)
d = ImageDraw.Draw(im)
S = meta["S"] * SC
def P(x, z): return ((x - meta["x0"]) * S, (z - meta["z0"]) * S)
rows = []
for i, r in enumerate(b["roofs"]):
    e = np.array(r["ext"])
    # Normale
    n = np.zeros(3)
    for j in range(len(e)):
        p, q = e[j], e[(j + 1) % len(e)]
        n += np.cross(p - e[0], q - e[0])
    A = np.linalg.norm(n) / 2; n = n / (np.linalg.norm(n) + 1e-9)
    slope = np.degrees(np.arccos(abs(n[2])))
    az = np.degrees(np.arctan2(n[0], n[1])) % 360
    c = e.mean(0)
    rows.append((i, A, slope, az, e[:, 2].min() - H0, e[:, 2].max() - H0, c[0], -c[1]))
    col = (255, 255, 0) if slope < 5 else ((0, 255, 255) if slope < 40 else (255, 0, 255))
    d.line([P(q[0], -q[1]) for q in e] + [P(e[0][0], -e[0][1])], fill=col, width=1)
    d.text(P(c[0], -c[1]), str(i), fill=(255, 255, 255))
im.save(BUILD + "/lod2_roofs.png")
for r in sorted(rows, key=lambda r: -r[1]):
    print("roof %2d A=%7.1f slope=%5.1f az=%5.1f y=%6.2f..%6.2f c=(%.1f,%.1f)" % r)

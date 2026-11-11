"""Ersatzbereich fuer das alte Cafe Kroepcke im Detailmodell (data/detail_raw) -> data/landmarks_replace/cafe_kroepcke.json

Umriss = LGLN-LoD2-Grundriss (13 Teile vereinigt), kantenweise versetzt:
  * freie Fassaden +0.35 m (alte Fenster/Sohlbaenke/Blenden stehen bis ~0.3 m vor)
  * Brandwaende zum Nachbarn (Ostseite N6-N7-N8, Nordwand P13 N4-N5) -0.03 m, damit dessen Wandflaechen erhalten bleiben
  * Kanten zur Jim-Block-Halle (Ueberdachung 000174Fq) +0.10 m
"""
import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover/data/landmarks_replace/cafe_kroepcke.json"
NAMES = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N15', 'W5', 'S4', 'S3', 'S2', 'S1', 'S0', 'W1']
P = {'N1': (23.70, 4.53), 'N2': (29.96, 6.70), 'N3': (36.47, 8.96), 'N4': (35.25, 12.31), 'N5': (42.54, 14.85),
     'N6': (48.61, 16.97), 'N7': (47.32, 20.58), 'N8': (56.14, 36.13), 'N9': (50.20, 39.40), 'N10': (43.88, 42.88),
     'N11': (38.26, 45.99), 'N12': (33.14, 37.08), 'N13': (26.80, 40.56), 'N15': (17.25, 23.03),
     'S0': (19.23, 6.92), 'S1': (18.01, 10.33), 'S2': (16.82, 13.64), 'S3': (15.63, 16.95), 'S4': (14.45, 20.26),
     'W1': (22.34, 8.06), 'W5': (17.80, 21.31)}
OFF = {('N3', 'N4'): -0.03, ('N5', 'N6'): -0.03, ('N4', 'N5'): -0.03, ('N6', 'N7'): -0.03, ('N7', 'N8'): -0.03, ('N11', 'N12'): 0.10, ('N12', 'N13'): 0.10, ('N13', 'N15'): 0.25}


def inside(pt, poly):
    x, y = pt; ins = False
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1): ins = not ins
    return ins


def isect(p, d, q, e):
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9: return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + d[0] * t, p[1] + d[1] * t)


poly = [P[k] for k in NAMES]
n = len(poly); lines = []
for i in range(n):
    a, b = poly[i], poly[(i + 1) % n]
    L = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5; t = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
    nr = (-t[1], t[0]); mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    if inside((mid[0] + nr[0] * 0.2, mid[1] + nr[1] * 0.2), poly): nr = (-nr[0], -nr[1])
    o = OFF.get((NAMES[i], NAMES[(i + 1) % n]), 0.35)
    lines.append(((a[0] + nr[0] * o, a[1] + nr[1] * o), t))
out = []
for i in range(n):
    (p0, d0), (p1, d1) = lines[i - 1], lines[i]
    q = isect(p0, d0, p1, d1)
    out.append([round(q[0], 3), round(q[1], 3)] if q else [round(p1[0], 3), round(p1[1], 3)])
doc = {
    "footprints": [out],
    "ymin": -1.0,
    "ymax": 20.0,
    "layers": ["bldg_walls", "bldg_openings", "bldg_roof", "bldg_base"],
    "note": ("Cafe Kroepcke / Moevenpick-Pavillon (LGLN LoD2 DENILD43...9TB7, 13 Teile) ersetzt durch landmarks/cafe_kroepcke.glb. "
             "Spielframe [x, z]. Entfernen: alle Dreiecke der genannten detail_raw-Ebenen, deren Schwerpunkt im Polygon liegt und "
             "ymin <= y <= ymax. Freie Fassaden +0.35 m versetzt; Brandwaende zum Nachbarbau im Nordosten/Osten (N3-N4-N5-N6-N7-N8) -0.03 m, damit der Nachbar intakt bleibt. Pflaster (street/ground/green) NICHT filtern. Die angrenzende "
             "Jim-Block-Halle (LoD2 000174Fq, Ueberdachung) ist nicht Teil des Ersatzes."),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(doc, open(OUT, 'w'), indent=1)
print('wrote', OUT, len(out), 'pts')

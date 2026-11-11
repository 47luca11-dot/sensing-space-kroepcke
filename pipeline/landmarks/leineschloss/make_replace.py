"""Ersatz-JSON fuer das Leineschloss: Grundriss = Vereinigung der city-LoD2-Grundflaechen (ID_b1e08153...) + Puffer
+ Freitreppe/Portikus-Vorbau (aus build/extents.json, falls vorhanden). Aufruf: make_replace.py [out.json]"""
import sys, json, os, numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from ls_common import *
OUT = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/data/landmarks_replace/leineschloss.json"
b = load_city_building()
polys = []
for g in b["grounds"]:
    e = np.array(g["ext"]); polys.append(Polygon(np.c_[e[:, 0], -e[:, 1]]).buffer(0))
U = unary_union(polys)
# Freitreppe vor dem Portikus und weitere Vorbauten (lokale Rechtecke u0,u1,v0,v1)
ext = {"boxes": [[-21.0, 17.5, -1.0, 6.2]], "ymax": 25.0}
ef = os.path.join(BUILD, "extents.json")
if os.path.exists(ef):
    ext.update(json.load(open(ef)))
for (a0, a1, c0, c1) in ext["boxes"]:
    x, z = local_to_game([a0, a1, a1, a0], [c0, c0, c1, c1])
    U = U.union(Polygon(np.c_[x, z]))
U = U.buffer(0.8, join_style=2).buffer(-0.3, join_style=2).simplify(0.15)
if isinstance(U, MultiPolygon):
    U = max(U.geoms, key=lambda g: g.area)
fp = [[round(x, 3), round(z, 3)] for x, z in np.array(U.exterior.coords)[:-1]]
# Kontrolle: keine fremden LoD2-Schwerpunkte im Grundriss
d = json.load(open(ROOT + "/data/lod2_buildings.json"))
P = Polygon(fp)
other = [bb["id"] for bb in d["buildings"] if bb["id"] != LOD2_ID and P.contains(Point(bb["centre"][0], -bb["centre"][1]))]
print("foreign LoD2 centroids inside:", other)
J = {"footprints": [fp], "ymin": -8.0, "ymax": round(float(ext["ymax"]), 2), "lod2_ids": [LOD2_ID],
     "note": "Leineschloss / Niedersaechsischer Landtag (Hannah-Arendt-Platz 1): ersetzt durch landmarks/leineschloss.glb "
             "(handmodelliert, pipeline/landmarks/leineschloss). Grundriss = city-LoD2 " + LOD2_ID + " (ganzer Komplex: Portikus, "
             "Laves-Bau Leinstrasse, Plenarsaalgebaeude, Hofflügel, Leinefluegel mit Loggia und Pavillons, Nordwestfluegel) + Freitreppe + 0.5 m Puffer. "
             "ymin -8: der Leinefluegel steht direkt im Fluss (Wasser y=-7.17). Boden/Ufer nicht filtern."}
tmp = OUT + ".tmp"
json.dump(J, open(tmp, "w", encoding="utf-8"), indent=1)
os.replace(tmp, OUT)
print("wrote", OUT, "pts", len(fp), "area %.0f" % P.area)

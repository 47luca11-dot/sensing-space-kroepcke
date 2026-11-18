# Person aus einer Splat-Szene ausschneiden – mit den Werten der Scan-Viewer-Extraktion (Mitte, Oben-Richtung, Höhe).
# Die Szenen haben beliebige Einheiten (Person z. B. 0,46 oder 1,06 groß); splat_werkzeug.ausschneiden rechnet aber mit
# Metern (Bodensuche, Toleranzen). Deshalb wird die Szene zuerst so skaliert, dass die Person ~1,8 groß ist.
#
#   python aus_viewer.py <szene.ply> <viewer_extrahiert.json> <person.ply> [--groesse 1.80] [--vorschau out.png]
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splat_werkzeug as SW

szene, info_json, ziel = sys.argv[1:4]
def arg(n, std=None): return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else std
info = json.load(open(info_json, encoding="utf-8"))
props, a = SW.lesen(szene)
s0 = 1.8 / max(float(info["height"]), 1e-6)
SW.transformieren(props, a, np.eye(3), np.zeros(3), s0)
zentrum = np.array(info["center"], float) * s0
oben = np.array(info["up"], float)
b, res = SW.ausschneiden(props, a, zentrum, oben, float(arg("--groesse", 1.80)), radius=1.0)
res["vorskalierung"] = round(s0, 4)
SW.schreiben(ziel, props, b); print("INFO", json.dumps(res))
if arg("--vorschau"):
    P = SW.pos(props, b); C = SW.farbe(props, b); k = SW.deckkraft(props, b) > 0.3
    SW.png(arg("--vorschau"), SW.rundum(P[k], C[k], np.array([0, 0, 1.0]), np.array([0, 0, 0.9]), 1.0, 360, 400))

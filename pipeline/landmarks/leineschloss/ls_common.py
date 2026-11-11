"""Gemeinsame Konstanten/Hilfen fuer das Leineschloss (Spielframe: x Ost, y Hoehe-56.16, z = -Nord)."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
RES = "C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research"
E0, N0, H0 = 550298.88274216, 5802940.705715501, 56.16
LOD2_ID = "ID_b1e08153-ebb6-3af8-8652-e6862bb48eee"
BUILD = os.path.join(HERE, "build")
os.makedirs(BUILD, exist_ok=True)

def load_city_building(bid=LOD2_ID):
    d = json.load(open(ROOT + "/data/lod2_buildings.json"))
    for b in d["buildings"]:
        if b["id"] == bid:
            return b
    raise KeyError(bid)

class Heightfield:
    def __init__(self):
        m = json.load(open(ROOT + "/web/assets/city/ground/heightfield.json"))
        self.m = m
        self.a = np.fromfile(ROOT + "/web/assets/city/ground/heightfield.bin", dtype=np.float32).reshape(m["h"], m["w"])
    def __call__(self, x, z):
        m = self.m
        fx = (np.asarray(x, float) - m["x0"]) / m["dx"]; fz = (np.asarray(z, float) - m["z0"]) / m["dz"]
        i0 = np.clip(np.floor(fx).astype(int), 0, m["w"] - 2); j0 = np.clip(np.floor(fz).astype(int), 0, m["h"] - 2)
        tx = np.clip(fx - i0, 0, 1); tz = np.clip(fz - j0, 0, 1)
        a = self.a
        return (a[j0, i0] * (1 - tx) * (1 - tz) + a[j0, i0 + 1] * tx * (1 - tz) + a[j0 + 1, i0] * (1 - tx) * tz + a[j0 + 1, i0 + 1] * tx * tz)

def write_atomic(path, data_bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data_bytes)
    os.replace(tmp, path)

# ---- Lokaler Gebaeuderahmen: u = laengs Leinstrasse nach SO, v = nach NO (aus dem Portikus heraus), Ursprung = Portikus-Vorderkante Mitte
ANG = np.radians(50.5)
FV = np.array([np.cos(ANG), np.sin(ANG)])        # (E,N) Richtung v (NO)
SU = np.array([np.sin(ANG), -np.cos(ANG)])  # (E,N) Richtung u (SO)
U0, V0 = -1.80, -549.42                           # Portikus-Achse (u=0) / Gesims-Vorderkante (v=0) in (s,f) um den Spielursprung

def local_to_game(u, v):
    """lokal (u,v) -> Spiel (x,z)"""
    u = np.asarray(u, float) + U0; v = np.asarray(v, float) + V0
    E = u * SU[0] + v * FV[0]; N = u * SU[1] + v * FV[1]
    return E, -N

def game_to_local(x, z):
    E = np.asarray(x, float); N = -np.asarray(z, float)
    return E * SU[0] + N * SU[1] - U0, E * FV[0] + N * FV[1] - V0

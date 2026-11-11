"""Gemeinsame Pfade/Helfer fuer das Landmark Leibnizhaus + Beginenturm + Historisches Museum."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(H, 'data')
WEB = os.path.join(H, 'web')
BUILD = os.path.join(HERE, 'build')
TEX = os.path.join(BUILD, 'tex')
RES = r'C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research'
E0, N0, Z0 = 550298.88274216, 5802940.705715501, 56.16
os.makedirs(BUILD, exist_ok=True); os.makedirs(TEX, exist_ok=True)

_HF = None


def heightfield():
    global _HF
    if _HF is None:
        meta = json.load(open(os.path.join(WEB, 'assets', 'city', 'ground', 'heightfield.json')))
        hf = np.fromfile(os.path.join(WEB, 'assets', 'city', 'ground', 'heightfield.bin'), np.float32).reshape(meta['h'], meta['w'])
        _HF = (meta, hf)
    return _HF


def ground_y(x, z):
    """bilinear: Bodenhoehe (Spielrahmen y) an x,z (Arrays erlaubt)."""
    meta, hf = heightfield()
    x = np.asarray(x, float); z = np.asarray(z, float)
    fx = (x - meta['x0']) / meta['dx']; fz = (z - meta['z0']) / meta['dz']
    i0 = np.clip(np.floor(fx).astype(int), 0, meta['w'] - 2); j0 = np.clip(np.floor(fz).astype(int), 0, meta['h'] - 2)
    tx = np.clip(fx - i0, 0, 1); tz = np.clip(fz - j0, 0, 1)
    a = hf[j0, i0]; b = hf[j0, i0 + 1]; c = hf[j0 + 1, i0]; d = hf[j0 + 1, i0 + 1]
    return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz


_LOD = None


def lod2():
    global _LOD
    if _LOD is None:
        d = json.load(open(os.path.join(DATA, 'lod2_buildings.json'), encoding='utf8'))
        _LOD = {b['id']: b for b in d['buildings']}
    return _LOD


def lod2_footprint(bid):
    """Grundriss im Spielrahmen [[x,z],...] (erste Grundflaeche)."""
    b = lod2()[bid]
    p = np.array(b['grounds'][0]['ext'])
    return np.c_[p[:, 0], -p[:, 1]]


# LoD2-IDs, die dieses Landmark ersetzt
IDS = {
    'leibnizhaus': 'ID_c494dba7-3b4d-3ac4-9a43-5e2a3e4af8e1',
    'noltehaus': 'ID_7f103f76-2cb1-36a5-b1f5-e39a5d4db35d',
    'museum': 'ID_1e223c17-842',
    'beginenturm': 'ID_5051cc6a-c86',
}


def full_ids():
    L = lod2()
    out = {}
    for k, pre in IDS.items():
        m = [i for i in L if i.startswith(pre[:15])]
        assert len(m) == 1, (k, m)
        out[k] = m[0]
    return out

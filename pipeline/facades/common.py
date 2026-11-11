"""Gemeinsame Pfade, Koordinaten, Farben und deterministische Zufallszahlen fuer die Fassaden-Pipeline.

Spielkoordinaten (Ausgabe): Meter, Y oben, x = dE, y = Z - 56.16, z = -dN (Ursprung Kroepcke-Uhr).
"""
import hashlib, os, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))          # output/hannover
DATA = os.path.join(ROOT, 'data')
WEB_OUT = os.path.join(ROOT, 'web', 'assets', 'city', 'buildings')
PREVIEWS = os.path.join(ROOT, 'blender', 'previews')
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
RESEARCH = os.environ.get('HANNOVER_RESEARCH', os.path.abspath(os.path.join(
    os.path.expanduser('~'), 'AppData', 'Local', 'Temp', 'claude', '86a9b6371061b51e',
    'b8a76d7d-27b8-490e-b6de-334a309ea42a', 'scratchpad', 'research')))
BLENDER = os.environ.get('BLENDER_EXE', r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe')

Y_OFF = 56.16                      # Z (DHHN92) der Strassenebene an der Uhr
E0, N0 = 550298.88274216, 5802940.705715501
RHINO_OFF = np.array([84.117, -0.16, 141.706])
TILE = 100.0                       # Kachelgroesse (m)

os.makedirs(CACHE, exist_ok=True)


def to_game(p):
    """[dE, dN, Z] -> [x, y, z] (Spiel)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2] - Y_OFF, -p[..., 1]], -1)


def seed_of(*parts):
    h = hashlib.md5('|'.join(str(p) for p in parts).encode('utf-8')).hexdigest()
    return int(h[:12], 16)


def rng_of(*parts):
    return np.random.default_rng(seed_of(*parts))


def hex2lin(h):
    """'#RRGGBB' (sRGB) -> lineare RGB-Floats (glTF-Faktoren / Vertexfarben sind linear)."""
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex2srgb8(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def jitter_hex(h, rng, amt=0.04):
    """Leichte Helligkeits-/Farbvariation einer sRGB-Farbe."""
    c = hex2srgb8(h).astype(float) / 255.0
    k = 1.0 + rng.uniform(-amt, amt)
    c = np.clip(c * k + rng.uniform(-amt / 3, amt / 3, 3), 0, 1)
    return '#%02X%02X%02X' % tuple(int(round(v * 255)) for v in c)


def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, p, **kw):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, **kw)

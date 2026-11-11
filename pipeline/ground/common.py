"""Gemeinsame Konstanten/Helfer fuer die Boden-Pipeline (Kroepcke, Hannover).
Spielkoordinaten: Meter, Y oben, x = Ost (dE), y = Hoehe - 56.16 (DHHN), z = -Nord (-dN).
Ursprung = Kroepcke-Uhr (OSM node 348155428)."""
import os, json, glob
import numpy as np

E0, N0 = 550298.88274216, 5802940.705715501
H0 = 56.16
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')).replace('\\', '/') + '/'
DATA = ROOT + 'data/'
OUT = ROOT + 'web/assets/city/ground/'
CACHE = DATA + 'ground_cache/'
PREV = ROOT + 'blender/previews/'
RHINO_OFF = np.array([84.117, -0.16, 141.706])     # Rhino-glTF -> Spielkoordinaten
# Detailmodell-Quadrat (Spiel x/z), dort erzeugen wir keinen Boden
SQ = (-67.38, -9.79, 235.62, 293.21)               # xmin, zmin, xmax, zmax  (aus street.glb-Bounds)
RADIUS = 860.0                                     # Ausbau 2: Altstadt + Steintor + Leine
HF_HALF = 900                                      # Hoehenraster: -900..+900 m
TEX = 'C:/Users/luca/Downloads/Opernplatz_Kiosk+Umgebung-Detail_3D_v1_embedded_files/'
os.makedirs(CACHE, exist_ok=True)


def load_dgm():
    """DGM1-Mosaik (LGLN, CC BY 4.0) als Array in Spielkoordinaten.
    Rueckgabe: arr[iz, ix] (Hoehe absolut, NaN=keine Daten), x0, z0 (Zellmitte [0,0]), res=1.
    Zeilen laufen nach Sueden (z steigt), Spalten nach Osten."""
    import tifffile
    files = sorted(glob.glob(DATA + 'dgm1/*.tif'))
    tiles = []
    for f in files:
        with tifffile.TiffFile(f) as t:
            p = t.pages[0]; a = p.asarray().astype(np.float32)
            tp = t.geotiff_metadata['ModelTiepoint']
            tiles.append((tp[3], tp[4], a))
    emin = min(t[0] for t in tiles); nmax = max(t[1] for t in tiles)
    emax = max(t[0] + t[2].shape[1] for t in tiles); nmin = min(t[1] - t[2].shape[0] for t in tiles)
    W, H = int(emax - emin), int(nmax - nmin)
    M = np.full((H, W), np.nan, np.float32)
    for e, n, a in tiles:
        c0 = int(e - emin); r0 = int(nmax - n)
        a = a.copy(); a[a < -1000] = np.nan
        M[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
    # PixelIsArea: Zellmitte Spalte c = emin + c + 0.5, Zeile r: N = nmax - r - 0.5
    x0 = emin + 0.5 - E0
    z0 = -(nmax - 0.5 - N0)
    return M, x0, z0, 1.0


class Grid:
    """Regelmaessiges Raster in Spielkoordinaten mit bilinearer Abfrage."""
    def __init__(self, arr, x0, z0, res):
        self.a = arr; self.x0 = x0; self.z0 = z0; self.res = res

    def __call__(self, x, z):
        x = np.asarray(x, float); z = np.asarray(z, float)
        fx = (x - self.x0) / self.res; fz = (z - self.z0) / self.res
        H, W = self.a.shape
        ix = np.clip(np.floor(fx).astype(int), 0, W - 2); iz = np.clip(np.floor(fz).astype(int), 0, H - 2)
        tx = np.clip(fx - ix, 0, 1); tz = np.clip(fz - iz, 0, 1)
        a = self.a
        return (a[iz, ix] * (1 - tx) * (1 - tz) + a[iz, ix + 1] * tx * (1 - tz) +
                a[iz + 1, ix] * (1 - tx) * tz + a[iz + 1, ix + 1] * tx * tz)


def load_osm():
    return json.load(open(DATA + 'osm_features.json', encoding='utf-8'))


def in_square(x, z, pad=0.0):
    return (x > SQ[0] - pad) & (x < SQ[2] + pad) & (z > SQ[1] - pad) & (z < SQ[3] + pad)


RAW = DATA + 'ground_raw/'          # unkomprimierte Kopien fuer die Blender-Master-Szene


def publish(src_tmp, name, raw_copy=True):
    """Fertige Datei atomar an ihren Platz tauschen (Spiel laeuft waehrenddessen) und Rohkopie ablegen."""
    import shutil
    os.makedirs(OUT, exist_ok=True); os.replace(src_tmp, OUT + name)
    if raw_copy:
        os.makedirs(RAW, exist_ok=True)
        tmp = RAW + '.' + name + '.tmp'
        shutil.copyfile(OUT + name, tmp); os.replace(tmp, RAW + name)


def staging(name):
    d = OUT + '.staging/'; os.makedirs(d, exist_ok=True)
    return d + name

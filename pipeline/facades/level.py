"""Strassenniveau vor den Waenden aus dem Hoehenfeld der Boden-Pipeline (web/assets/city/ground/heightfield.bin: begehbare
Oberflaeche im Spielframe, 1 m Raster) - dasselbe Gelaende, das das Spiel zeigt. Ersatzweise DGM1 (prep.Terrain).

Verwendung: Bezugsniveau je Gebaeude (Geschossraster, b.y_ground waehrend des Fassadenbaus) und Fussniveau je Laden/Tuer, damit
Schaufenster und Tueren auf dem Gehweg stehen statt auf dem tiefsten LoD2-Bodenpunkt (Rampen, Promenade, Flussufer)."""
import os, json
import numpy as np
from common import ROOT

HF_DIR = os.path.join(ROOT, 'web', 'assets', 'city', 'ground')
_HF = None


def _load():
    global _HF
    if _HF is None:
        try:
            meta = json.load(open(os.path.join(HF_DIR, 'heightfield.json'), encoding='utf-8'))
            a = np.fromfile(os.path.join(HF_DIR, 'heightfield.bin'), np.float32).reshape(int(meta['h']), int(meta['w']))
            _HF = (float(meta['x0']), float(meta['z0']), float(meta.get('dx', 1.0)), a)
        except (OSError, ValueError, KeyError):
            _HF = False
    return _HF


_TERR = None


def sample(x, z):
    """Spiel-x,z -> y der begehbaren Oberflaeche (bilinear); nan ausserhalb."""
    x = np.atleast_1d(np.asarray(x, float)); z = np.atleast_1d(np.asarray(z, float))
    H = _load()
    if not H:
        global _TERR
        if _TERR is None:
            import prep
            _TERR = prep.Terrain()
        return _TERR.sample(x, z)
    x0, z0, dx, a = H
    fx = (x - x0) / dx; fz = (z - z0) / dx
    j = np.floor(fx).astype(int); i = np.floor(fz).astype(int)
    ok = (j >= 0) & (i >= 0) & (j < a.shape[1] - 1) & (i < a.shape[0] - 1)
    out = np.full(len(x), np.nan)
    if ok.any():
        jj, ii = j[ok], i[ok]; tx = fx[ok] - jj; tz = fz[ok] - ii
        out[ok] = (a[ii, jj] * (1 - tx) * (1 - tz) + a[ii, jj + 1] * tx * (1 - tz) + a[ii + 1, jj] * (1 - tx) * tz
                   + a[ii + 1, jj + 1] * tx * tz)
    return out


def front(w, u0, u1, ds=(0.8, 1.2, 1.6), n=None, vol=None, own=-1):
    """Niveau vor der Wand w zwischen u0 und u1 (Proben 0.8-1.6 m vor der Wand), Median; None wenn nichts brauchbar.
    Proben innerhalb fremder Gebaeudevolumen (Nachbarhaus, Durchfahrt unter Vordach) werden verworfen."""
    if u1 < u0:
        u0, u1 = u1, u0
    k = n or max(2, min(9, int((u1 - u0) / 0.8) + 2))
    m = 0.15 if u1 - u0 > 0.6 else 0.0
    us = np.linspace(u0 + m, u1 - m, k)
    P = np.concatenate([w.fr.w(np.c_[us, np.zeros(len(us)), np.full(len(us), d)]) for d in ds])
    y = sample(P[:, 0], P[:, 2])
    good = np.isfinite(y)
    if vol is not None and good.any():
        Q = P.copy(); Q[:, 1] = np.where(good, y, 0.0) + 1.0
        bo, _ = vol.blocked(Q, own)
        good &= ~bo
    if not good.any():
        return None
    return float(np.median(y[good]))

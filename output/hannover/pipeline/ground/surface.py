"""Abfrage der fertigen Bodenoberflaeche (Hoehe + Klasse) per Dreieckssuche im Gelaendemesh."""
import numpy as np
from matplotlib.tri import Triangulation
from common import *

class Surface:
    def __init__(self):
        TM = np.load(CACHE + 'terrain.npz', allow_pickle=False)
        self.V, self.T, self.lab, self.HC = TM['V'], TM['T'], TM['lab'], TM['HC']
        self.names = list(TM['names'])
        bz = np.load(CACHE + 'base.npz'); self.BG = Grid(bz['B'].astype(np.float64), float(bz['x0']), float(bz['z0']), 1.0)
        self.finder = Triangulation(self.V[:, 0], self.V[:, 1], self.T).get_trifinder()

    def __call__(self, x, z):
        x = np.atleast_1d(np.asarray(x, float)); z = np.atleast_1d(np.asarray(z, float))
        ti = self.finder(x, z); h = self.BG(x, z); ok = ti >= 0
        cls = np.full(len(x), '', dtype=object)
        if ok.any():
            t = self.T[ti[ok]]; V = self.V
            a = V[t[:, 0]]; b = V[t[:, 1]]; c = V[t[:, 2]]; p = np.c_[x[ok], z[ok]]
            v0 = b - a; v1 = c - a; v2 = p - a
            d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
            den = d00 * d11 - d01 * d01; den[den == 0] = 1
            wv = (d11 * d20 - d01 * d21) / den; ww = (d00 * d21 - d01 * d20) / den; wu = 1 - wv - ww
            hc = self.HC[ti[ok]]
            h[ok] = wu * hc[:, 0] + wv * hc[:, 1] + ww * hc[:, 2]
            cls[ok] = np.array(self.names, dtype=object)[self.lab[ti[ok]]]
        return h, cls

"""Instanzen-Sammler (EXT_mesh_gpu_instancing): je (Prototyp, Materialzuordnung) Listen von TRS + Instanzfarbe."""
import numpy as np
import protos as PR
from geom import col


def mat2quat(R):
    """Rotationsmatrix (Spalten = Bildachsen) -> Quaternion (x,y,z,w)."""
    m = R; t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w]); return q / np.linalg.norm(q)


def get_proto(key):
    if key in PR.PROTOS:
        return PR.PROTOS[key]
    if key.startswith(('sf_', 'sfw_', 'dsp_')):
        import displays
        return displays.get(key)
    if key == 'glass':
        return PR.glass_proto()
    if key.startswith('int_'):
        return PR.interior_proto(key[4:])
    if key.startswith('sill_'):
        return PR.sill_proto(key[5:])
    if key.startswith('sur_'):
        return PR.surround_proto(key[4:])
    if key.startswith('rollo_'):
        return PR.rollo_proto(int(key[6:]) / 100.0)
    if key == 'fr_bal':
        return PR.balcony_proto()
    if key.startswith('shop_'):
        d = key[5:]
        return PR.shop_proto(None if d == 'fix' else d)
    if key.startswith('door_'):
        _, kind, r = key.split('_')
        return PR.door_proto(kind, int(r) / 100.0)
    if key.startswith('lancet_'):
        _, w, h = key.split('_')
        return PR.lancet_proto(int(w) / 100.0, int(h) / 100.0)
    return PR.window_proto(key)


class InstAcc:
    def __init__(self):
        self.g = {}          # (key, matmap) -> [T, Q, S, C]

    def add(self, key, matmap, fr, u, v, d, sx, sy, sz, c):
        P = fr.w([u, v, d])
        R = np.stack([fr.t, fr.b, fr.n], 1)
        q = mat2quat(R)
        L = self.g.setdefault((key, matmap), [[], [], [], []])
        L[0].append(P); L[1].append(q); L[2].append((sx, sy, sz)); L[3].append(col(c))

    def count(self):
        return sum(len(L[0]) for L in self.g.values())

    def tri_count(self):
        n = 0
        for (key, mm), L in self.g.items():
            n += get_proto(key).ntri * len(L[0])
        return n

    def merge_from(self, other):
        for k, L in other.g.items():
            M = self.g.setdefault(k, [[], [], [], []])
            for i in range(4):
                M[i].extend(L[i])

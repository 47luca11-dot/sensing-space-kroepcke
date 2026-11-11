"""Leineschloss -- Stufe 1: gesamte Geometrie (numpy) im Gebaeude-Lokalrahmen erzeugen, nach Blender-Rahmen
(x Ost, y Nord, z oben; Ursprung Kroepcke-Uhr) umrechnen und als build/ls_geo.npz + ls_geo.json ablegen.
Aufruf: python ls_build.py [teile...]   (Standard: alle)"""
import sys, os, json, time
import numpy as np
import lslib as L
import ls_parts as P
from ls_common import *

TEX_META = os.path.join(BUILD, 'tex', 'tex_meta.json')


class Ctx:
    def __init__(self):
        self.t0 = time.time(); self.stats = {}; self.texts = []
        self.hf = Heightfield()
        self._cache = {}

    def log(self, name, n):
        self.stats[name] = self.stats.get(name, 0) + int(n)
        print('%-22s %8d tris  (%.1fs)' % (name, n, time.time() - self.t0), flush=True)

    def ground(self, u, v):
        x, z = local_to_game(np.asarray(u, float), np.asarray(v, float))
        return self.hf(x, z)

    def _c(self, key, fn):
        if key not in self._cache: self._cache[key] = fn()
        return self._cache[key]

    def tympanum(self, w, h): return self._c(('t', round(w, 2), round(h, 2)), lambda: P.tympanum_heightmap(18, w, h))
    def anthemion(self, l, h): return self._c(('a', round(l, 2), round(h, 2)), lambda: P.anthemion_heightmap(16, l, h))
    def garland(self, l, h): return self._c(('g', round(l, 2), round(h, 2)), lambda: P.garland_heightmap(20, l, h))
    def cartouche(self, w, h): return self._c(('c', round(w, 2), round(h, 2)), lambda: P.cartouche_heightmap(22, w, h))


def local_to_blender_matrix():
    M = np.eye(4)
    M[0, 0], M[0, 1] = SU[0], FV[0]
    M[1, 0], M[1, 1] = SU[1], FV[1]
    M[0, 3] = U0 * SU[0] + V0 * FV[0]
    M[1, 3] = U0 * SU[1] + V0 * FV[1]
    return M


def save(B, ctx, name='ls_geo'):
    M = local_to_blender_matrix()
    tiles = json.load(open(TEX_META)) if os.path.exists(TEX_META) else {}
    def tile(mat):
        k = mat[3:].lower() if mat.startswith('LS_') else mat.lower()
        return tiles.get(k)
    arrs = {}; groups = []
    for i, ((mat, grp), (T, UV)) in enumerate(sorted(B.arrays().items())):
        tl = tile(mat)
        if tl: UV = UV / tl
        Tw = L.xform(T, M)
        arrs['p%d' % i] = Tw.reshape(-1, 3).astype(np.float32); arrs['uv%d' % i] = UV.reshape(-1, 2).astype(np.float32)
        groups.append(dict(i=i, mat=mat, grp=grp, tris=int(len(T))))
    inst = []
    for j, (nm, e) in enumerate(sorted(B.inst.items())):
        parts = []
        for k, ((mat, grp), (T, UV)) in enumerate(sorted(e['part'].arrays().items())):
            tl = tile(mat)
            if tl: UV = UV / tl
            arrs['ip%d_%d' % (j, k)] = T.reshape(-1, 3).astype(np.float32); arrs['iuv%d_%d' % (j, k)] = UV.reshape(-1, 2).astype(np.float32)
            parts.append(dict(k=k, mat=mat, grp=grp, tris=int(len(T))))
        mats = [(M @ m).tolist() for m in e['M']]
        inst.append(dict(j=j, name=nm, parts=parts, M=mats))
    texts = []
    for t in ctx.texts:
        p = (M @ np.r_[t['pos'], 1.0])[:3]; x = M[:3, :3] @ np.asarray(t['x']); z = M[:3, :3] @ np.asarray(t['z'])
        texts.append(dict(t, pos=p.tolist(), x=x.tolist(), z=z.tolist()))
    np.savez_compressed(os.path.join(BUILD, name + '.npz'), **arrs)
    n, ni = B.tri_count()
    meta = dict(groups=groups, instances=inst, texts=texts, tris=n, inst_tris=ni, stats=ctx.stats, M_local_to_blender=M.tolist())
    json.dump(meta, open(os.path.join(BUILD, name + '.json'), 'w'), indent=1)
    print('TOTAL merged %d + instanced %d = %d tris' % (n, ni, n + ni))


def main(parts):
    B = L.Builder(); ctx = Ctx()
    import ls_b_portico
    mods = {'portico': ls_b_portico}
    for nm in ('laves', 'plenar', 'hof', 'leine', 'nordwest'):
        try:
            mods[nm] = __import__('ls_b_' + nm)
        except ImportError:
            pass
    for nm, m in mods.items():
        if parts and nm not in parts: continue
        m.build(B, ctx)
    save(B, ctx)


if __name__ == '__main__':
    main(sys.argv[1:])

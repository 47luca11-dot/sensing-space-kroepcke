"""Luecken-/Dichtheitspruefung der Gebaeudehuellen ('shell'-Ebene).

Fuer jedes Gebaeude: offene Kanten (nach Verschweissen auf 1 mm) suchen; Punkte auf diesen Kanten gelten als dicht,
wenn sie auf einer anderen Flaeche des Gebaeudes liegen (T-Stoesse, Kanten auf Wandflaechen). Unterirdische Kanten
(Wandfuss unter Gelaende) werden ignoriert. Ausgabe: Gebaeude mit sichtbaren offenen Kanten (Laenge, Ort).
Aufruf: python check_gaps.py [--limit N] [--only id,..]"""
import os, sys, argparse, time, collections
import numpy as np
import trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep as PREP
from build_all import make_ctx, build_building
from common import CACHE, save_json


def shell_tris(acc, layers=('shell', 'shell_b')):
    T = []; own = []
    for (layer, mat), parts in acc.parts.items():
        if layer not in layers:
            continue
        for V, F, C in parts:
            T.append(np.asarray(V)[np.asarray(F)]); own.append(np.full(len(F), layer == 'shell'))
    if not T:
        return np.zeros((0, 3, 3)), np.zeros(0, bool)
    return np.concatenate(T), np.concatenate(own)


def open_edges(T, b, tol=float(os.environ.get('GAP_TOL', 0.004)), vol=None, report=None):
    """Offene Kanten: Randkanten (einfach genutzt), deren Abtastpunkte nicht auf einem anderen Dreieck liegen."""
    V = T.reshape(-1, 3)
    key = np.round(V / 0.001).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    F = inv.reshape(-1, 3)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.concatenate([np.arange(len(F))] * 3)
    Es = np.sort(E, 1)
    u, first, cnt = np.unique(Es, axis=0, return_index=True, return_counts=True)
    bi = np.where(cnt == 1)[0]
    if len(bi) == 0:
        return 0.0, []
    P = np.zeros((inv.max() + 1, 3)); P[inv] = V
    A = P[u[bi, 0]]; B = P[u[bi, 1]]; own = fid[first[bi]]
    keep = np.minimum(A[:, 1], B[:, 1]) > b.y_ground + 0.05
    if report is not None:
        keep &= report[own]
    A, B, own = A[keep], B[keep], own[keep]
    if len(A) == 0:
        return 0.0, []
    TT = P[F]
    from rtree import index as rindex
    lo = TT.min(1) - 1e-3; hi = TT.max(1) + 1e-3
    idx = rindex.Index(((i, (*lo[i], *hi[i]), None) for i in range(len(TT))), properties=rindex.Property(dimension=3))
    bad = np.zeros(len(A), bool)
    for k in range(len(A)):
        for t in (0.2, 0.5, 0.8):
            p = A[k] + (B[k] - A[k]) * t
            cand = [c for c in idx.intersection((*(p - tol), *(p + tol))) if c != own[k]]
            if not cand:
                bad[k] = True; break
            cp = trimesh.triangles.closest_point(TT[cand], np.repeat(p[None], len(cand), 0))
            if np.linalg.norm(cp - p, axis=1).min() > tol:
                bad[k] = True; break
    if vol is not None and bad.any():
        # Kanten im Inneren des eigenen LoD2-Volumens (z.B. Gaubenwangen unter der Dachhaut) sind unsichtbar
        M = (A + B) / 2
        Q = M - np.array([0, 0.04, 0])
        pi, ow = vol.query(Q)
        inside = np.zeros(len(M), bool)
        if len(pi):
            gi_ok = ow >= 0
            # strenger: 4 cm unter der Dachflaeche
            inside[pi[gi_ok]] = True
        bad &= ~inside
    L = np.linalg.norm(B - A, axis=1)
    lens = L[bad]; mids = ((A + B) / 2)[bad]
    return float(lens.sum()), mids[np.argsort(-lens)][:5].round(2).tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--only', default=None)
    a = ap.parse_args()
    d = PREP.load_prep(); ctx = make_ctx(d)
    import replaced; replaced.mark(d['bldgs'], verbose=False)
    bl = [b for b in d['bldgs'] if not b.covered and not getattr(b, 'replaced', None)]
    if a.only:
        bl = [b for b in bl if any(b.id.endswith(x) for x in a.only.split(','))]
    if a.limit:
        rng = np.random.default_rng(1); bl = list(rng.choice(bl, a.limit, replace=False))
    res = []; t0 = time.time()
    for b in bl:
        acc, inst, coll, info = build_building(b, ctx)
        if info['kind'] == 'canopy':
            continue
        T, rep = shell_tris(acc)
        if len(T) == 0:
            continue
        L, where = open_edges(T, b, vol=d['vol'], report=rep)
        res.append({'id': b.id, 'kind': info['kind'], 'open_m': round(L, 2), 'where': where})
    res.sort(key=lambda r: -r['open_m'])
    tot = sum(r['open_m'] for r in res)
    print('Gebaeude geprueft:', len(res), 'offene Kantenlaenge (> %.0f mm Abstand) gesamt %.1f m' % (float(os.environ.get('GAP_TOL', 0.004)) * 1000, tot), '%.0f s' % (time.time() - t0),
          'Gebaeude mit offenen Kanten:', sum(1 for r in res if r['open_m'] > 0.05))
    for r in res[:25]:
        print(r)
    save_json(res, os.path.join(CACHE, 'gap_report.json'), indent=0)


if __name__ == '__main__':
    main()

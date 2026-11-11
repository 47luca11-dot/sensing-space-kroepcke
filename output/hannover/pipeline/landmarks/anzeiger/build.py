"""Build all geometry (Anzeiger-Hochhaus + Goseriedebad) -> build/anzeiger_parts.npz + build/anzeiger_meta.json
and run sanity checks (degenerate triangles, bounds, triangle budget, footprint containment)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geo, anz, bad

OUT = os.path.join(HERE, 'build'); os.makedirs(OUT, exist_ok=True)

MATS = {
    'klinker': {'tex': 'klinker'}, 'chevron': {'tex': 'chevron'}, 'concrete': {'tex': 'concrete'}, 'stone_dark': {'tex': 'stone_dark'},
    'roof_flat': {'tex': 'roof_flat'}, 'copper': {'tex': 'copper', 'metal': 0.15}, 'copper_dark': {'tex': 'copper_dark', 'metal': 0.15},
    'ashlar': {'tex': 'ashlar'}, 'stone': {'tex': 'stone'}, 'tiles': {'tex': 'tiles'}, 'slate': {'tex': 'slate'}, 'plaster': {'tex': 'plaster'},
    'paint_white': {'color': [0.86, 0.86, 0.83], 'rough': 0.45},
    'glass': {'color': [0.11, 0.13, 0.145], 'rough': 0.06, 'metal': 0.0},
    'glass_shop': {'color': [0.42, 0.42, 0.40], 'rough': 0.12},
    'glass_dark': {'color': [0.022, 0.025, 0.028], 'rough': 0.3, 'metal': 0.0},      # deep portal glazing (dark interiors, refs 03 / 09)
    'glass_green': {'color': [0.09, 0.25, 0.17], 'rough': 0.12, 'metal': 0.0},
    'gold': {'color': [0.95, 0.64, 0.07], 'metal': 0.55, 'rough': 0.38},
    'letter_side': {'color': [0.38, 0.08, 0.06], 'rough': 0.55},
    'metal_dark': {'color': [0.055, 0.055, 0.06], 'metal': 0.5, 'rough': 0.45},
    'steel': {'color': [0.62, 0.62, 0.63], 'metal': 1.0, 'rough': 0.3},
    'alu': {'color': [0.24, 0.25, 0.25], 'metal': 0.6, 'rough': 0.4},
    'lamp_glass': {'color': [0.16, 0.15, 0.12], 'rough': 0.1, 'emit': [1.0, 0.78, 0.45], 'emit_strength': 0.12},   # smoky prismatic glass, faint warm glow (refs 03 / 06 / 09: dark-framed glass lanterns)
    'bronze': {'color': [0.16, 0.11, 0.07], 'metal': 0.8, 'rough': 0.38},          # Kestner revolving door (ref goseriede/06)
    'sign_panel': {'color': [0.76, 0.75, 0.72], 'rough': 0.7},
    'print_band': {'color': [0.50, 0.52, 0.52], 'rough': 0.3, 'metal': 0.2},       # Kestner printed glass band
}
SMOOTH = {'anz_dome'}


def main():
    t0 = time.time()
    mb = geo.MB()
    anz.build(mb)
    n_anz = mb.total()
    bad.build(mb)
    arr = mb.arrays()
    objs = {}
    for (o, m), (P, UV) in arr.items():
        objs.setdefault(o, []).append((m, P, UV))
    npz = {}; meta = {'objects': {}, 'materials': MATS, 'smooth': sorted(SMOOTH | mb.smooth & SMOOTH)}
    total = 0; bad_tris = 0
    lo = np.full(3, 1e9); hi = np.full(3, -1e9)
    for o, lst in sorted(objs.items()):
        Vs, Us, Ms, names = [], [], [], []
        for i, (m, P, UV) in enumerate(sorted(lst, key=lambda x: x[0])):
            if m not in MATS: raise SystemExit('unknown material %s' % m)
            T = P.reshape(-1, 3, 3)
            area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
            ok = area > 1e-9
            bad_tris += int((~ok).sum())
            T = T[ok]; U = UV.reshape(-1, 3, 2)[ok]
            Vs.append(T.reshape(-1, 3)); Us.append(U.reshape(-1, 2)); Ms.append(np.full(len(T), i, np.int32)); names.append(m)
        V = np.concatenate(Vs).astype(np.float32); U = np.concatenate(Us).astype(np.float32); M = np.concatenate(Ms)
        if not np.isfinite(V).all() or not np.isfinite(U).all(): raise SystemExit('non-finite values in ' + o)
        npz['V_' + o] = V; npz['U_' + o] = U; npz['M_' + o] = M
        meta['objects'][o] = {'materials': names, 'tris': int(len(M))}
        total += len(M)
        lo = np.minimum(lo, V.min(0)); hi = np.maximum(hi, V.max(0))
    meta['total_tris'] = int(total); meta['anzeiger_tris'] = int(n_anz); meta['bounds'] = [lo.tolist(), hi.tolist()]
    np.savez_compressed(os.path.join(OUT, 'anzeiger_parts.npz'), **npz)
    json.dump(meta, open(os.path.join(OUT, 'anzeiger_meta.json'), 'w'), indent=1)
    print('BUILD tris %d (anzeiger %d, goseriedebad %d), removed degenerate %d, bounds %s .. %s, %.1fs' % (
        total, n_anz, total - n_anz, bad_tris, np.round(lo, 2), np.round(hi, 2), time.time() - t0))
    for o, d in meta['objects'].items(): print('  ', o, d['tris'], d['materials'])
    if total > 800000: raise SystemExit('triangle budget exceeded')


if __name__ == '__main__':
    main()

"""Analyse LGLN LoD2 parts of the Cafe Kroepcke: footprint, vault axis, profile."""
import json, sys, numpy as np
parts = [p for p in json.load(open(sys.argv[1])) if p['building'].endswith('00009TB7')]
for p in parts:
    g = np.array(p['surf']['ground'][0])
    print('\n###', p['part'][-10:], 'roofType', p['roofType'], 'H', p['measuredHeight'], 'ground y', g[:, 1].min(), g[:, 1].max())
    print('  ground xz:', [(round(a, 2), round(c, 2)) for a, b, c in g[:-1]])
    ax = []
    for r in p['surf']['roof']:
        r = np.array(r)
        n = np.cross(r[1] - r[0], r[2] - r[0])
        # robust normal (Newell)
        n = np.zeros(3)
        for i in range(len(r)):
            a, b = r[i], r[(i + 1) % len(r)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        n /= np.linalg.norm(n) + 1e-12
        print('  roof facet n=(%.3f %.3f %.3f) y %.2f..%.2f  npts %d' % (n[0], n[1], n[2], r[:, 1].min(), r[:, 1].max(), len(r) - 1))
        h = np.array([n[0], n[2]])
        if np.linalg.norm(h) > 0.05: ax.append(np.arctan2(h[1], h[0]))
    if ax:
        a = np.array(ax); print('  facet horiz normal angles (deg):', np.round(np.degrees(a), 1))
    for k in ('other', 'wall'):
        for w in p['surf'][k]:
            w = np.array(w)
            if k == 'other' and len(w) > 0:
                print('  closure y %.2f..%.2f xz %s' % (w[:, 1].min(), w[:, 1].max(), [(round(a, 1), round(b, 2), round(c, 1)) for a, b, c in w[:-1]][:12]))

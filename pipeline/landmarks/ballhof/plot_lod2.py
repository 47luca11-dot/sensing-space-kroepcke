"""Plot roof polygons (plan) of the ensemble buildings with heights -> build/lod2_roofs.png"""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'build', 'site_lod2.json')))
fig, axs = plt.subplots(2, 4, figsize=(28, 14)); axs = axs.ravel()
keys = ['ID_c48b3643', 'ID_bcc9a17d', 'ID_98a9691d', 'ID_b150f6c8', 'ID_84e270ab', 'ID_1a4a4aa4', 'ID_0875e4bf', 'ID_ff0b0763']
for ax, k in zip(axs, keys):
    b = S[k]
    for g in b['grounds']:
        a = np.array(g); ax.plot(np.r_[a[:, 0], a[0, 0]], np.r_[a[:, 2], a[0, 2]], 'k-', lw=2)
    for i, r in enumerate(b['roofs']):
        a = np.array(r); c = plt.cm.tab10(i % 10)
        ax.fill(a[:, 0], a[:, 2], color=c, alpha=0.3); ax.plot(np.r_[a[:, 0], a[0, 0]], np.r_[a[:, 2], a[0, 2]], '-', color=c)
        for p in a: ax.text(p[0], p[2], '%.1f' % p[1], fontsize=7, color=c)
        ax.text(a[:, 0].mean(), a[:, 2].mean(), 'R%d' % i, fontsize=10, weight='bold')
    for w in b['walls']:
        a = np.array(w)
        if a[:, 1].max() - a[:, 1].min() > 0.1:
            pass
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_title('%s %s %s' % (k, b['street'], b['number'])); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'build', 'lod2_roofs.png'), dpi=60)

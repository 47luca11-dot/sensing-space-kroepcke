"""Plot city LoD2 roofs of Anzeiger + Goseriedebad (game frame, north up) with heights above base."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'site_lod2.json')))
fig, axs = plt.subplots(1, 2, figsize=(22, 13))
for ax, keys, lim in ((axs[0], ['anz'], (-516, -483, -303, -263)), (axs[1], ['bad', 'n7'], (-550, -485, -368, -295))):
    cm = plt.get_cmap('turbo')
    for k in keys:
        for r in S[k]['roofs']:
            p = np.array(r); y = p[:, 1]
            ax.fill(p[:, 0], p[:, 2], color=cm(np.clip((y.mean() + 1) / 52, 0, 1)), ec='k', lw=0.4, alpha=0.8)
            cx, cz = p[:, 0].mean(), p[:, 2].mean()
            ax.text(cx, cz, '%.1f' % y.max() if y.max() - y.min() < 0.05 else '%.1f-%.1f' % (y.min(), y.max()), fontsize=6, ha='center')
        for g in S[k]['grounds']:
            p = np.array(g); ax.plot(p[:, 0], p[:, 2], 'r-', lw=1.2)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[3], lim[2]); ax.set_aspect('equal'); ax.grid(True, lw=0.3)
    ax.set_xticks(np.arange(lim[0], lim[1], 2 if keys == ['anz'] else 5))
    ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'lod2_heights.png'), dpi=90)
for k in ('anz', 'bad'):
    g = np.array(S[k]['grounds'][0]); print(k, 'ground ring', len(g), 'y', g[:, 1].min(), g[:, 1].max())
    print(np.round(g[:, [0, 2]], 2).tolist())

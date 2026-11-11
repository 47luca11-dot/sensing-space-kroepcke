"""Debug plot of openings / shops / assignment -> <scratch>/assign_<area>.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, DATA, jload  # noqa: E402
from assign import opening_center  # noqa: E402

SCR = os.environ.get("SIGNAGE_SCRATCH", r"C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/"
                     r"b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/signage")
AREAS = {"north": (-80, 250, -60, 80), "south": (-80, 130, 90, 300), "west": (-80, 10, -20, 280)}
O = jload(os.path.join(CACHE, "openings.json"))["openings"]
A = jload(os.path.join(CACHE, "assign.json"))["opening_shop"]
S = {s["id"]: s for s in jload(os.path.join(DATA, "shops.json"))["shops"]}
for name in sys.argv[1:] or AREAS:
    x0, x1, z0, z1 = AREAS[name]
    fig = plt.figure(figsize=(18, 18 * (z1 - z0) / (x1 - x0)))
    ax = fig.gca()
    cols = {}
    rng = np.random.default_rng(1)
    for o in O:
        c = opening_center(o)
        R = np.array(o["R"])
        g = o["glass"]["rect"]
        P0 = np.array(o["O"]) + R @ np.array([g[0], 0, 0])
        P1 = np.array(o["O"]) + R @ np.array([g[1], 0, 0])
        sid = A.get(o["id"])
        if sid not in cols:
            cols[sid] = rng.uniform(0.1, 0.9, 3) if sid else np.array([0.6, 0.6, 0.6])
        lw = 3 if o["kind"] == "window" else 5
        ax.plot([P0[0], P1[0]], [P0[2], P1[2]], "-", color=cols[sid], lw=lw)
        n = R[:, 2]
        ax.plot([c[0], c[0] + n[0] * 0.8], [c[2], c[2] + n[2] * 0.8], "-", color="k", lw=0.4)
        if o["box"]["depth"] < 0.2:
            ax.plot(c[0], c[2], "rx", ms=4)
    for sid, col in cols.items():
        if not sid:
            continue
        s = S[sid]
        f = s.get("facade") or {}
        wp = f.get("wall_point") or s["p"]
        if not (x0 < wp[0] < x1 and z0 < wp[1] < z1):
            continue
        ax.plot(wp[0], wp[1], "o", color=col, ms=5)
        ax.text(wp[0] + 0.5, wp[1], (s["name"] or "(%s)" % s.get("former_name")) + " " + (s.get("address") or "")[-6:],
                fontsize=6, color=col * 0.7)
    for s in S.values():
        if s.get("level", 0) != 0:
            continue
        f = s.get("facade") or {}
        wp = f.get("wall_point") or s["p"]
        if x0 < wp[0] < x1 and z0 < wp[1] < z1 and s["id"] not in cols:
            ax.plot(wp[0], wp[1], "k^", ms=4)
            ax.text(wp[0] + 0.5, wp[1] + 1, "!" + (s["name"] or "-"), fontsize=6, color="r")
    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(SCR, "assign_%s.png" % name), dpi=90, bbox_inches="tight")
    plt.close(fig)

"""Assign the shops of data/shops.json to the detected openings -> cache/assign.json

Each opening gets at most one shop; a shop gets the contiguous openings of its frontage.
Cost = along-facade distance from the shop's facade point (entrance node if known) + perpendicular offset
+ penalties (address / street mismatch, facade normal mismatch, closed shops).  A shop never reaches further
along the facade than half its frontage width (+ tolerance for the OSM point accuracy), so one shop cannot
swallow a whole block.  Residential entrances are only assigned when the
shop's point is right at that door (OSM shop node = entrance).  Openings no shop reaches stay 'generic'
(display interior without a name)."""
import os
import re

import numpy as np

from common import CACHE, DATA, jdump, jload, log

MAXD = 26.0


def norm_street(s):
    s = (s or "").lower().replace("ß", "ss").replace("strasse", "str").replace("straße", "str")
    return re.sub(r"[^a-z0-9äöü]", "", s)


def parse_addr(a):
    if not a:
        return None, []
    m = re.match(r"^(.*?)\s+(\d+)\s*([a-zA-Z]?)(?:\s*[-–/]\s*(\d+)\s*([a-zA-Z]?))?", a.strip())
    if not m:
        return norm_street(a), []
    st = norm_street(m.group(1))
    n0 = int(m.group(2))
    n1 = int(m.group(4)) if m.group(4) else n0
    if n1 < n0 or n1 - n0 > 12:
        n1 = n0
    return st, list(range(n0, n1 + 1))


def opening_center(o):
    O = np.array(o["O"])
    R = np.array(o["R"])
    g = o["glass"]["rect"]
    return O + R @ np.array([(g[0] + g[1]) / 2, (g[2] + g[3]) / 2, o["glass"]["zg"]])


def main():
    O = jload(os.path.join(CACHE, "openings.json"))["openings"]
    S = jload(os.path.join(DATA, "shops.json"))["shops"]
    mask = np.load(os.path.join(DATA, "detail_mask.npy"))

    def near_mask(p, r=6):
        j = int(np.floor(p[0] + 400))
        i = int(np.floor(p[1] + 400))
        if not (0 <= i < 800 and 0 <= j < 800):
            return False
        return bool(mask[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].any())

    shops = []
    for s in S:
        if s.get("level", 0) != 0 or s.get("context") not in ("street", "passage"):
            continue
        f = s.get("facade") or {}
        wp = f.get("wall_point") or s["p"]
        ent = f.get("entrance_node")
        pos = np.array(ent["p"] if ent and ent.get("dist", 99) < 4 else wp, float)
        if not near_mask(pos):
            continue
        nn = f.get("normal")
        st, nums = parse_addr(s.get("address"))
        closed = s.get("status") in ("closed", "vacant", "likely_closed")
        shops.append(dict(s=s, pos=pos, wp=np.array(wp, float), n=np.array([nn[0], 0, nn[1]]) if nn else None,
                          st=st, nums=nums, w=float(s.get("frontage_width_m") or 6.0), closed=closed))
    log("shops near the detail model", len(shops))

    C = np.array([opening_center(o) for o in O])
    N = np.array([np.array(o["R"])[:, 2] for o in O])
    Rr = np.array([np.array(o["R"])[:, 0] for o in O])
    ost = [(norm_street(o["lod2"].get("street")), o["lod2"].get("number")) for o in O]
    is_ent = np.array([o["kind"] == "entrance" for o in O])
    cost = np.full((len(O), len(shops)), np.inf)
    for j, sh in enumerate(shops):
        d = C[:, [0, 2]] - sh["pos"][None]
        dist = np.linalg.norm(d, axis=1)
        cand = np.where(dist < MAXD)[0]
        reach = max(sh["w"] / 2 + 1.5, 4.0)
        for i in cand:
            r = Rr[i]
            along = abs(d[i] @ r[[0, 2]])
            perp = abs(d[i] @ N[i][[0, 2]])
            st, num = ost[i]
            addr_match = False
            if sh["st"] and st:
                try:
                    nb = int(re.match(r"\d+", num).group(0)) if num else None
                except (AttributeError, TypeError):
                    nb = None
                addr_match = st == sh["st"] and nb is not None and nb in sh["nums"]
            facing = 1.0
            if sh["n"] is not None:
                facing = float(N[i] @ sh["n"])
            lim = reach + (2.0 if addr_match else 0.0)
            if facing < 0.3:
                # other facade plane (corner shop): only right at the corner
                lim = min(lim, 3.0)
                if facing < -0.3:
                    continue
            if is_ent[i]:
                # a house door only belongs to the shop if the shop's own point (OSM entrance) is right there
                lim = min(lim, 1.8)
            if along > lim or perp > (7.0 if addr_match else 5.0):
                continue
            c = along + 0.5 * max(0.0, perp - 1.5)
            if addr_match:
                c -= 2.0
            elif sh["st"] and st:
                c += 1.0 if st == sh["st"] else 3.0
            if facing < 0.3:
                c += 4.0
            if sh["closed"]:
                c += 3.0
            if is_ent[i]:
                c += 1.5
            cost[i, j] = c
    best = np.argmin(cost, axis=1)
    bc = cost[np.arange(len(O)), best]
    owner = np.where(np.isfinite(bc), best, -1)
    # make sure every shop with a finite cost somewhere gets at least one opening (steal from shops with > 1)
    for it in range(3):
        counts = np.bincount(owner[owner >= 0], minlength=len(shops))
        for j in np.argsort([sh["w"] for sh in shops]):
            if counts[j] > 0:
                continue
            col = cost[:, j]
            if not np.isfinite(col).any():
                continue
            for i in np.argsort(col):
                if not np.isfinite(col[i]) or col[i] > 8:
                    break
                k = owner[i]
                if k < 0 or counts[k] > 1:
                    if k >= 0:
                        counts[k] -= 1
                    owner[i] = j
                    counts[j] += 1
                    break
    # contiguity: a shop's openings on one facade must not be interrupted by another shop's opening
    # (an isolated opening between two of a neighbour's openings goes to that neighbour)
    for it in range(2):
        for i in range(len(O)):
            k = owner[i]
            if k < 0:
                continue
            same = [q for q in range(len(O)) if q != i and N[q] @ N[i] > 0.97 and abs((C[q] - C[i])[[0, 2]] @ N[i][[0, 2]]) < 0.6
                    and np.linalg.norm((C[q] - C[i])[[0, 2]]) < 4.0 and not is_ent[q]]
            if not same:
                continue
            al = [float((C[q] - C[i])[[0, 2]] @ Rr[i][[0, 2]]) for q in same]
            left = [owner[q] for q, a in zip(same, al) if -4.0 < a < 0]
            right = [owner[q] for q, a in zip(same, al) if 0 < a < 4.0]
            if left and right:
                ln = left[int(np.argmax([a for a in al if -4.0 < a < 0]))]
                rn = right[int(np.argmin([a for a in al if 0 < a < 4.0]))]
                if ln == rn and ln >= 0 and ln != k and np.isfinite(cost[i, ln]) and \
                        np.sum(owner == k) > 1:
                    owner[i] = ln
    res = {}
    for i, o in enumerate(O):
        j = owner[i]
        if j < 0:
            continue
        res[o["id"]] = shops[j]["s"]["id"]
    used = set(res.values())
    lost = [sh["s"]["name"] or sh["s"].get("former_name") for sh in shops if sh["s"]["id"] not in used]
    counts = np.bincount(owner[owner >= 0], minlength=len(shops))
    log("openings with shop", len(res), "of", len(O), "| shops placed", len(used), "of", len(shops),
        "| max openings per shop", int(counts.max()) if len(counts) else 0)
    log("shops without opening:", lost)
    jdump(dict(opening_shop=res, shops_placed=sorted(used), shops_unplaced=lost), os.path.join(CACHE, "assign.json"))


if __name__ == "__main__":
    main()

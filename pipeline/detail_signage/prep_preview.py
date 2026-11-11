"""Crop the scene around the preview cameras (keeps Blender RAM low) and write the camera list.

python prep_preview.py  -> <scratch>/prev/<view>/*.glb + <scratch>/prev/cams.json
Views (game frame [x, z]; eye height added on top of the street surface)."""
import glob
import json
import os
import sys

import numpy as np

from common import DATA, ROOT, HERE, detail_path, glbio, log

SCR = os.environ.get("SIGNAGE_SCRATCH", r"C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/"
                     r"b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/signage")
PREV = os.path.join(SCR, "prev")

# name: (eye [x, z], target [x, z], target height above street, lens mm)
VIEWS = {
    "georgstr_ost": ([47.0, 124.0], [78.0, 186.0], 2.2, 24),
    "georgstr_ost_nah": ([62.0, 166.0], [52.0, 186.5], 1.8, 20),
    "luisenstr": ([112.5, 27.0], [124.0, -16.0], 2.4, 22),
    "luisenstr_nah": ([118.0, 12.0], [127.0, 1.0], 1.8, 20),
    "rathenaustr": ([38.0, 11.0], [92.0, 27.0], 2.4, 22),
    "rathenaustr_nah": ([55.7, 13.6], [59.8, 10.8], 1.7, 20),     # street side (the old eye at 60/6.5 was 4 m inside the building)
    "rathenaustr_ost": ([140.0, 52.0], [185.0, 64.0], 2.4, 22),
    "karmarschstr": ([-37.5, 16.0], [-45.0, 62.0], 2.4, 22),
    "karmarschstr_nah": ([-44.0, 38.0], [-52.0, 36.5], 1.7, 20),
    # wall-only signs (shops without an own shopfront opening in the detail model)
    "wand_osterstr": ([-16.5, 307.0], [-26.0, 314.0], 3.0, 20),
    "wand_georgstr50": ([109.4, 273.6], [111.2, 276.9], 3.4, 18),
    "wand_packhof": ([-92.6, 6.1], [-84.1, 9.2], 3.0, 20),
    "wand_osterstr22": ([-50.2, 224.9], [-43.7, 218.7], 3.0, 20),
}
EYE = 1.65
RADIUS = 75.0   # max crop radius around the eye-target segment (see main)


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def strip_mat(m):
    m = json.loads(json.dumps(m or {"name": "default"}))
    pb = m.get("pbrMetallicRoughness", {})
    for k in list(pb):
        if k.endswith("Texture"):
            pb.pop(k)
    for k in list(m):
        if k.endswith("Texture"):
            m.pop(k)
    ext = m.get("extensions", {})
    for e in list(ext):
        for k in list(ext[e]):
            if k.endswith("Texture"):
                ext[e].pop(k)
    return m


def crop(path, out, a, b, R):
    g = glbio.read_glb(path)
    nodes = []
    for nd in g["nodes"]:
        for p in nd["prims"]:
            T = p["idx"]
            C = p["pos"][T].mean(1)
            m = seg_dist(C[:, [0, 2]], a, b) < R
            if not m.any():
                continue
            F = T[m]
            used, inv = np.unique(F.reshape(-1), return_inverse=True)
            nodes.append(dict(name=nd["name"], material=strip_mat(p["material"]), pos=p["pos"][used],
                              nrm=p["nrm"][used] if p["nrm"] is not None else None, uv=None,
                              idx=inv.reshape(-1, 3)))
    if nodes:
        glbio.write_glb(out, nodes)
    return sum(len(n["idx"]) for n in nodes)


def ground_y(x, z, files):
    best = None
    for f in files:
        g = glbio.read_glb(f)
        for nd in g["nodes"]:
            for p in nd["prims"]:
                T = p["pos"][p["idx"]]
                lo = T.min(1)
                hi = T.max(1)
                m = (lo[:, 0] <= x) & (hi[:, 0] >= x) & (lo[:, 2] <= z) & (hi[:, 2] >= z) & (T[:, :, 1].mean(1) < 6)
                for t in T[m]:
                    # barycentric in xz
                    A, B, C = t[0][[0, 2]], t[1][[0, 2]], t[2][[0, 2]]
                    v0, v1, v2 = C - A, B - A, np.array([x, z]) - A
                    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
                    den = d00 * d11 - d01 * d01
                    if abs(den) < 1e-12:
                        continue
                    u = (d11 * d20 - d01 * d21) / den
                    v = (d00 * d21 - d01 * d20) / den
                    if u >= -1e-6 and v >= -1e-6 and u + v <= 1 + 1e-6:
                        y = t[0][1] + u * (t[2][1] - t[0][1]) + v * (t[1][1] - t[0][1])
                        best = y if best is None else max(best, y)
    if best is None:
        from common import heightfield_y
        hy = heightfield_y(x, z)
        best = hy if np.isfinite(hy) else 0.0
    return best


def shop_view(name, dist=3.2, side=0.0, lens=22, th=1.9):
    """camera in front of the openings of a shop (by name), `dist` m out from the facade and `side` m along it
    (positive = to the right seen from the street), looking at the shop -> (eye, target, th, lens)."""
    from common import CACHE, jload
    O = jload(os.path.join(CACHE, "openings.json"))["openings"]
    A = jload(os.path.join(CACHE, "assign.json"))["opening_shop"]
    S = {s["id"]: s for s in jload(os.path.join(DATA, "shops.json"))["shops"]}
    ops = [o for o in O if A.get(o["id"]) and (S[A[o["id"]]]["name"] or "") == name]
    if not ops:
        raise SystemExit("no openings for " + name)
    C = []
    for o in ops:
        g = o["glass"]["rect"]
        C.append(np.array(o["O"]) + np.array(o["R"]) @ np.array([(g[0] + g[1]) / 2, 0, 0]))
    c = np.mean(C, 0)
    n = np.array(ops[0]["R"])[:, 2]
    r = np.array(ops[0]["R"])[:, 0]
    eye = c + n * dist + r * side
    return ([float(eye[0]), float(eye[2])], [float(c[0]), float(c[2])], th, lens)


def main():
    """args: view names of VIEWS, or  alias=shop:NAME@dist@side@lens@th  (alias = output name)."""
    views = sys.argv[1:] or list(VIEWS)
    for v in list(views):
        alias = None
        spec = v
        if "=" in v:
            alias, spec = v.split("=", 1)
        if spec.startswith("shop:"):
            parts = spec[5:].split("@")
            nm = parts[0]
            nums = [float(x) for x in parts[1:]]
            defaults = [3.2, 0.0, 22, 1.9]
            nums = nums + defaults[len(nums):]
            key = alias or spec.replace(":", "_").replace(" ", "_").replace("@", "_")
            VIEWS[key] = shop_view(nm, *nums)
            views[views.index(v)] = key
    os.makedirs(PREV, exist_ok=True)
    files = [detail_path(n) for n in ("bldg_walls", "bldg_openings", "bldg_base", "bldg_roof", "d0_windows", "fix",
                                      "kiosk", "street", "street_furniture", "ground", "green", "landmark_opernhaus",
                                      "landmark_tiefgarage_pavillon", "landmark_mahnmal")]
    files += [os.path.join(DATA, "landmarks_raw", n + ".glb") for n in ("cafe_kroepcke", "kroepcke_center",
                                                                          "kroepcke_uhr")]
    gfiles = [detail_path("street"), detail_path("ground"), detail_path("fix")]
    cams = {}
    for v in views:
        eye, tgt, th, lens = VIEWS[v]
        a, b = np.array(eye, float), np.array(tgt, float)
        d = os.path.join(PREV, v)
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.glb")):
            os.remove(f)
        tot = 0
        for f in files:
            if os.path.isfile(f):
                tot += crop(f, os.path.join(d, os.path.basename(f)), a, b, min(RADIUS, float(np.linalg.norm(b - a)) + 40.0))
        ye = ground_y(eye[0], eye[1], gfiles)
        yt = ground_y(tgt[0], tgt[1], gfiles)
        cams[v] = dict(eye=[eye[0], ye + EYE, eye[1]], target=[tgt[0], yt + th, tgt[1]], lens=lens, dir=d)
        log(v, "tris", tot, "eye y %.2f" % ye)
    J = {}
    p = os.path.join(PREV, "cams.json")
    if os.path.isfile(p):
        J = json.load(open(p))
    J.update(cams)
    json.dump(J, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()

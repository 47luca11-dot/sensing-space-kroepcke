"""Validation of the D0 upgrade (run after build.py):
 1. see-through: rays from street viewpoints through every opening must hit something opaque
    (glass ignored) within 2.5 m behind the wall plane
 2. gaps around the opening: rays aimed 1 cm outside the hole edge must stop at the wall surface
 3. z-fighting: coplanar same-facing overlapping faces between d0_windows and walls/openings
 4. wall boundary edges: new open edges in bldg_walls vs. original
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glbio  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "data", "detail_raw")
ORIG = os.path.join(ROOT, "data", "detail_raw_orig")


def load_mesh(paths, skip_glass=True):
    V, F, M, o = [], [], [], 0
    names = []
    for p in paths:
        g = glbio.read_glb(p)
        for n in g["nodes"]:
            mn = n["material"]["name"]
            if skip_glass and ("Glass" in mn and "Spandrel" not in mn):
                continue
            for pr in n["prims"]:
                V.append(pr["pos"]); F.append(pr["idx"] + o); o += len(pr["pos"])
                M.append(np.full(len(pr["idx"]), len(names)))
            names.append(os.path.basename(p) + ":" + mn)
    m = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=False)
    return m, np.concatenate(M), names


def main():
    W = json.load(open(os.path.join(HERE, "cache_windows.json")))
    files = [os.path.join(RAW, f + ".glb") for f in ("bldg_walls", "d0_windows", "bldg_openings", "bldg_roof", "bldg_base")]
    scene, mid, names = load_mesh(files)
    ray = scene.ray
    print("scene tris", len(scene.faces))
    rng = np.random.default_rng(1)
    # ---------------------------------------------------------------- 1 + 2
    O, D, info = [], [], []
    for w in W:
        C = np.array(w["c"]); n = np.array(w["n"]); r = np.cross([0, 1.0, 0], n); r /= np.linalg.norm(r); u = np.cross(n, r)
        hw, hh = w["w"] / 2, w["h"] / 2
        for k in range(24):
            # target point inside the opening
            tx = rng.uniform(-hw * 0.97, hw * 0.97); ty = rng.uniform(-hh * 0.97, hh * 0.97)
            T = C + r * tx + u * ty
            az = np.radians(rng.uniform(-65, 65)); el = np.radians(rng.uniform(-35, 50))
            dvec = n * np.cos(az) * np.cos(el) + r * np.sin(az) * np.cos(el) + u * np.sin(el)
            E = T + dvec * rng.uniform(3, 25)
            O.append(E); D.append(T - E); info.append((w["id"], 0, np.dot(T - E, -n)))
        for k in range(16):
            # points 1.5 cm outside the hole edge (wall / trim must be there)
            side = k % 4
            t = rng.uniform(-0.9, 0.9)
            if side == 0:
                P = C + r * (t * hw) - u * (hh + 0.015)
            elif side == 1:
                P = C + r * (hw + 0.015) + u * (t * hh)
            elif side == 2:
                P = C + r * (t * hw) + u * (hh + 0.015)
            else:
                P = C - r * (hw + 0.015) + u * (t * hh)
            E = P + n * 6.0 + r * rng.uniform(-1.5, 1.5) + u * rng.uniform(-1.5, 1.5)
            O.append(E); D.append(P - E); info.append((w["id"], 1, 0))
    O = np.array(O); D = np.array(D)
    L = np.linalg.norm(D, axis=1)
    Dn = D / L[:, None]
    locs, ri, ti = ray.intersects_location(O, Dn, multiple_hits=False)
    hitd = np.full(len(O), np.inf)
    hitd[ri] = np.linalg.norm(locs - O[ri], axis=1)
    win_n = {w["id"]: np.array(w["n"]) for w in W}
    bad_see, bad_gap = {}, {}
    for k, (wid, typ, _) in enumerate(info):
        if typ == 0:
            # depth behind the wall plane of the hit
            n = win_n[wid]
            Cw = np.array(W[wid]["c"])
            if not np.isfinite(hitd[k]):
                bad_see[wid] = bad_see.get(wid, 0) + 1
                continue
            P = O[k] + Dn[k] * hitd[k]
            depth = -np.dot(P - Cw, n)
            if depth > 2.5:
                bad_see[wid] = bad_see.get(wid, 0) + 1
        else:
            # must hit within 6 cm of the target point (wall surface, trim, sill)
            if not np.isfinite(hitd[k]) or hitd[k] > L[k] + 0.05:
                bad_gap[wid] = bad_gap.get(wid, 0) + 1
    print("see-through: %d windows with leaking rays (%d rays) of %d" % (len(bad_see), sum(bad_see.values()), len(W)))
    print("gaps around openings: %d windows (%d rays)" % (len(bad_gap), sum(bad_gap.values())))
    for wid in list(bad_see)[:15]:
        print("   see", wid, W[wid]["style"], W[wid]["mode"], np.round(W[wid]["c"], 2), bad_see[wid])
    for wid in list(bad_gap)[:15]:
        print("   gap", wid, W[wid]["style"], W[wid]["mode"], np.round(W[wid]["c"], 2), bad_gap[wid])
    # ---------------------------------------------------------------- 3 z-fighting
    d0 = glbio.read_glb(os.path.join(RAW, "d0_windows.glb"))
    allm, amid, anames = load_mesh(files, skip_glass=False)
    fn = allm.face_normals
    d0_names = [i for i, nm in enumerate(anames) if nm.startswith("d0_windows")]
    sel = np.where(np.isin(amid, d0_names))[0]
    area = allm.area_faces
    sel = sel[area[sel] > 2e-4]
    # all d0 faces (chunked)
    zf_list, pair_list = [], []
    for c0 in range(0, len(sel), 250000):
        ss = sel[c0:c0 + 250000]
        C = allm.triangles_center[ss]
        N = fn[ss]
        O = C + N * 0.004
        locs, ri, ti = allm.ray.intersects_location(O, -N, multiple_hits=True)
        dist = np.linalg.norm(locs - O[ri], axis=1)
        near = dist < 0.0055
        ri, ti, dist = ri[near], ti[near], dist[near]
        same = (np.einsum("ij,ij->i", fn[ti], N[ri]) > 0.995) & (ti != ss[ri])
        zf = same & (np.abs(dist - 0.004) < 0.0006)
        zf_list.append(ss[ri[zf]]); pair_list.append(ti[zf])
    a_all = np.concatenate(zf_list); b_all = np.concatenate(pair_list)
    zf_faces = np.unique(a_all)
    print("z-fighting candidates: %d of %d d0 faces (all checked)" % (len(zf_faces), len(sel)))
    if len(zf_faces):
        cnt = {}
        for f in zf_faces:
            cnt[anames[amid[f]]] = cnt.get(anames[amid[f]], 0) + 1
        print("   by material:", sorted(cnt.items(), key=lambda kv: -kv[1])[:10])
        other = {}
        for a, b in zip(a_all, b_all):
            k = anames[amid[a]] + " <-> " + anames[amid[b]]
            other[k] = other.get(k, 0) + 1
        print("   pairs:", sorted(other.items(), key=lambda kv: -kv[1])[:10])
        print("   e.g.", np.round(allm.triangles_center[zf_faces[:5]], 3).tolist())
    # ---------------------------------------------------------------- 4 boundary edges
    def bedges(path):
        g = glbio.read_glb(path)
        T = np.concatenate([n["prims"][0]["pos"][n["prims"][0]["idx"]] for n in g["nodes"]])
        V = T.reshape(-1, 3)
        key = np.round(V / 0.0005).astype(np.int64)
        u, inv = np.unique(key, axis=0, return_inverse=True)
        F = inv.reshape(-1, 3)
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        ue, c = np.unique(E, axis=0, return_counts=True)
        b = ue[c == 1]
        return set(map(tuple, np.concatenate([u[b[:, 0]], u[b[:, 1]]], 1)))
    b0 = bedges(os.path.join(ORIG, "bldg_walls.glb"))
    b1 = bedges(os.path.join(RAW, "bldg_walls.glb"))
    print("wall boundary edges: orig %d, new %d, added %d, removed %d" % (len(b0), len(b1), len(b1 - b0), len(b0 - b1)))


if __name__ == "__main__":
    main()

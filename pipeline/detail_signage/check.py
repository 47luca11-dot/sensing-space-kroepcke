"""Validation of data/detail_raw/signage.glb against the detail model as the game shows it (FrontSide only).

  1. intersections: every signage edge is ray-cast against the scene and every scene edge near the overlay
     against the overlay (two triangles intersect iff an edge of one crosses the other)
  2. z-fighting vs scene: overlay faces with a same-facing parallel scene face closer than 3 mm in front/behind
  3. z-fighting inside the overlay: same-facing coplanar faces (plane offset < 3 mm, measured locally, 0.05 mm
     float32 allowance) whose areas overlap
     (> 1e-5 m2) - the game camera (near 0.05 m, no log depth) separates 3 mm up to ~50 m
  4. see-through: from street eye points in front of every opening (1.2 / 3 / 7 m out, 0 / +-2.5 / +-6 m to the
     side, also openings without a box) rays through random points of the glass. First non-glass hit must be a
     front face (a back face with a front face in the same place - the closed double-sided display shells -
     counts as front): no hit = leak, overlay back face = hole, scene back face behind the glass = hole,
     a hit more than 2.2 m behind the glass = view through the building
  5. display boxes of different openings must not share volume (mitre clips included)
  6. hidden boxes: a box behind a scene panel that covers the glass (wasted triangles)
  7. normals: stored normals agree with the triangle winding
Writes cache/check.json."""
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import trimesh

from common import CACHE, DATA, RAW_OUT, detail_path, heightfield_y, jdump, jload, load_tris, log
from openings import load_scene

ZTOL = 0.003


def ray_mesh(T):
    m = trimesh.Trimesh(T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3), process=False)
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector(m)
    except Exception:
        return trimesh.ray.ray_triangle.RayMeshIntersector(m)


def edge_hits(rm, T, eps=1e-3, chunk=400000):
    """number of triangle edges of T that cross the mesh of rm (excluding touching endpoints)."""
    A = np.concatenate([T[:, 0], T[:, 1], T[:, 2]])
    Bp = np.concatenate([T[:, 1], T[:, 2], T[:, 0]])
    tid = np.concatenate([np.arange(len(T))] * 3)
    d = Bp - A
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-6
    A, d, L, tid = A[ok], d[ok] / L[ok, None], L[ok], tid[ok]
    bad = set()
    for s in range(0, len(A), chunk):
        o = A[s:s + chunk] + d[s:s + chunk] * eps
        ll = L[s:s + chunk] - 2 * eps
        idx_ray = rm.intersects_first(o, d[s:s + chunk])
        hit = idx_ray >= 0
        if not hit.any():
            continue
        hi = np.where(hit)[0]
        locs, ri, _ = rm.intersects_location(o[hi], d[s:s + chunk][hi], multiple_hits=False)
        dist = np.linalg.norm(locs - o[hi][ri], axis=1)
        close = dist < ll[hi][ri]
        for k in np.where(close)[0]:
            bad.add(int(tid[s + hi[ri[k]]]))
    return bad


def first_hits(rm, o, d):
    """index of the first hit triangle and its distance (-1 / inf if none)."""
    idx = rm.intersects_first(o, d)
    dist = np.full(len(o), np.inf)
    h = np.where(idx >= 0)[0]
    if len(h):
        locs, ri, tri = rm.intersects_location(o[h], d[h], multiple_hits=False)
        dist[h[ri]] = np.linalg.norm(locs - o[h][ri], axis=1)
        idx[h] = -1
        idx[h[ri]] = tri
    return idx, dist


def coplanar_overlaps(T, FNu, idx_a, tol_d=ZTOL, min_area=1e-5):
    """same-facing faces with plane offsets closer than tol_d whose projections overlap by more than min_area."""
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    nq = np.round(FNu[idx_a] * 40).astype(np.int64)
    dd = np.einsum("ij,ij->i", FNu[idx_a], T[idx_a, 0])
    pairs = {}
    for off in (0.0, 0.5):
        dq = np.floor(dd / (2 * tol_d) + off).astype(np.int64)
        kk = np.concatenate([nq, dq[:, None]], 1)
        _, gi = np.unique(kk, axis=0, return_inverse=True)
        gi = gi.reshape(-1)
        order = np.argsort(gi, kind="stable")
        gs = gi[order]
        br = np.r_[0, np.where(np.diff(gs))[0] + 1, len(gs)]
        for s0, s1 in zip(br[:-1], br[1:]):
            if s1 - s0 < 2:
                continue
            ids = idx_a[order[s0:s1]]
            nn = FNu[ids[0]]
            u = np.cross(nn, [0, 1, 0]) if abs(nn[1]) < 0.9 else np.cross(nn, [1, 0, 0])
            u /= np.linalg.norm(u)
            v = np.cross(nn, u)
            P = np.stack([T[ids] @ u, T[ids] @ v], -1)
            polys = [Polygon(p) for p in P]
            tree = STRtree(polys)
            a_i, b_i = tree.query(polys, predicate="intersects")
            m = a_i < b_i
            for a_, b2 in zip(a_i[m], b_i[m]):
                ia, ib = int(ids[a_]), int(ids[b2])
                if (min(ia, ib), max(ia, ib)) in pairs:
                    continue
                # separation measured locally (B's vertices against A's plane through A's own vertex): the global
                # plane offsets n.p carry float32 / normal noise of a few 0.1 mm at 300 m from the origin
                sep = float(np.abs((T[ib] - T[ia, 0]) @ FNu[ia]).mean())
                # layers designed exactly 3 mm apart pass: 0.05 mm allowance for float32 vertex storage
                if sep >= tol_d - 5e-5 or np.dot(FNu[ia], FNu[ib]) < 0.999:
                    continue
                ar = polys[a_].intersection(polys[b2]).area
                if ar > min_area:
                    pairs[(min(ia, ib), max(ia, ib))] = (ar, sep)
    return pairs


def coincident_front(ALL, FN, P, D, tids, tol=0.0015):
    """for rays whose first hit is a back face: is there a front-facing triangle in the same place (within tol)?
    Display shells are closed from both sides (inner face + dark outer reveal face in the same plane); with
    FrontSide rendering the front one is drawn, the ray caster may report either."""
    from scipy.spatial import cKDTree
    out = np.zeros(len(P), bool)
    if not len(P):
        return out
    lo, hi = P.min(0) - 0.05, P.max(0) + 0.05
    sel = np.where(np.all(ALL.max(1) >= lo, 1) & np.all(ALL.min(1) <= hi, 1))[0]
    if not len(sel):
        return out
    Tc = ALL[sel]
    cen = Tc.mean(1)
    rad = np.linalg.norm(Tc - cen[:, None], axis=2).max(1) + tol
    tree = cKDTree(P)
    hits = tree.query_ball_point(cen, rad)
    for ti, pl in zip(range(len(sel)), hits):
        if not pl:
            continue
        t = sel[ti]
        A, B_, C_ = ALL[t]
        n = FN[t]
        e0, e1 = B_ - A, C_ - A
        d00, d01, d11 = e0 @ e0, e0 @ e1, e1 @ e1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-14:
            continue
        for k in pl:
            if out[k] or tids[k] == t:
                continue
            if n @ D[k] >= -1e-4:
                continue                        # not facing the viewer
            v = P[k] - A
            if abs(v @ n) > tol:
                continue
            d20, d21 = v @ e0, v @ e1
            bv = (d11 * d20 - d01 * d21) / den
            bw = (d00 * d21 - d01 * d20) / den
            if bv >= -1e-3 and bw >= -1e-3 and bv + bw <= 1 + 1e-3:
                out[k] = True
    return out


def in_clips(L, clips):
    ok = np.ones(len(L), bool)
    for cx_, cz_, k in clips or []:
        ok &= L[:, 0] * cx_ + L[:, 2] * cz_ <= k
    return ok


def box_overlaps(OPS):
    rng = np.random.default_rng(1)
    boxes = []
    for o in OPS:
        b = o["box"]
        if b["depth"] < 0.03:
            continue
        a, bb, c, d = b["rect"]
        boxes.append((o["id"], np.array(o["O"]), np.array(o["R"]), np.array([a, c, b["z_back"]]),
                      np.array([bb, d, b["z_front"]]), b.get("clip") or []))
    hits = []
    for i in range(len(boxes)):
        idi, Oi, Ri, loi, hii, cli = boxes[i]
        ci = Oi + Ri @ ((loi + hii) / 2)
        ri = np.linalg.norm(hii - loi) / 2
        for j in range(i + 1, len(boxes)):
            idj, Oj, Rj, loj, hij, clj = boxes[j]
            cj = Oj + Rj @ ((loj + hij) / 2)
            if np.linalg.norm(ci - cj) > ri + np.linalg.norm(hij - loj) / 2:
                continue
            P = rng.uniform(loi + 0.003, hii - 0.003, (4000, 3))
            P = P[in_clips(P, cli)]
            L = (Oi + P @ Ri.T - Oj) @ Rj
            ins = np.all((L > loj + 0.003) & (L < hij - 0.003), 1) & in_clips(L, clj)
            if ins.any():
                hits.append((idi, idj, round(float(np.prod(hii - loi) * ins.mean()), 4)))
    return hits


def main():
    t0 = time.time()
    S, Sm, Snames = load_tris(RAW_OUT)
    log("overlay tris", len(S))
    lo, hi = S.reshape(-1, 3).min(0) - 2, S.reshape(-1, 3).max(0) + 2
    _, obst = load_scene(lo, hi, hi[1] + 2, t0)
    # scene triangles near the overlay (0.5 m hash of overlay triangle boxes)
    cell = 0.5

    def enc(k):
        k = k + 20000
        return (k[:, 0] * 40000 + k[:, 1]) * 40000 + k[:, 2]
    smin = np.floor(S.min(1) / cell).astype(np.int64)
    smax = np.floor(S.max(1) / cell).astype(np.int64)
    ks = [smin, smax, np.floor(S.mean(1) / cell).astype(np.int64)]
    bigs = np.where(np.any(smax - smin > 1, axis=1))[0]
    extra = []
    for t in bigs:
        a_, b_ = smin[t], smax[t]
        g = np.stack(np.meshgrid(*[np.arange(a_[i], b_[i] + 1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        extra.append(enc(g))
    base = np.unique(np.concatenate([enc(k) for k in ks] + extra))
    dec = np.stack([base // (40000 * 40000), (base // 40000) % 40000, base % 40000], 1) - 20000
    offs = np.array([[i, j, k] for i in range(-3, 2) for j in range(-3, 2) for k in range(-3, 2)])
    keyset = np.unique(np.concatenate([enc(dec + o) for o in offs]))
    omin = np.floor(obst.lo / cell).astype(np.int64)
    omax = np.floor(obst.hi / cell).astype(np.int64)
    small = np.all(omax - omin <= 2, axis=1)
    near = np.isin(enc(omin), keyset) | ~small
    O = obst.T[near]
    log("scene tris near overlay", len(O), "(%.1fs)" % (time.time() - t0))
    rm_scene = ray_mesh(O)
    bad1 = edge_hits(rm_scene, S)
    rm_sign = ray_mesh(S)
    bad2 = edge_hits(rm_sign, O)
    log("overlay tris with an edge crossing the scene:", len(bad1), "| scene tris crossing the overlay:", len(bad2),
        "(%.1fs)" % (time.time() - t0))
    # 2. z-fighting vs scene: same-facing parallel scene face within 3 mm in front of / behind each overlay face
    N = np.cross(S[:, 1] - S[:, 0], S[:, 2] - S[:, 0])
    ar = np.linalg.norm(N, axis=1)
    okn = ar > 1e-10
    Nn = N[okn] / ar[okn, None]
    C = S[okn].mean(1)
    zf_pts = []
    Onrm = np.cross(O[:, 1] - O[:, 0], O[:, 2] - O[:, 0])
    Onrm /= np.maximum(np.linalg.norm(Onrm, axis=1, keepdims=True), 1e-12)
    for sgn in (1, -1):
        o = C - sgn * Nn * 0.0005
        dd = sgn * Nn
        idx, dist = first_hits(rm_scene, o, dd)
        h = np.where((idx >= 0) & (dist < ZTOL + 0.0005))[0]
        same = np.einsum("ij,ij->i", Onrm[idx[h]], Nn[h]) > 0.995
        for k in h[same]:
            zf_pts.append([round(float(v), 2) for v in C[k]])
    log("z-fighting vs scene (same-facing parallel scene face within 3 mm):", len(zf_pts))
    # 3. z-fighting inside the overlay
    FNu = np.zeros_like(N)
    FNu[okn] = Nn
    sel = np.where(ar / 2 > 1e-7)[0]
    vit = np.array(["Vitrine" in n for n in Snames])[Sm]
    sel = sel[~vit[sel]]
    pairs = coplanar_overlaps(S, FNu, sel)
    cp = sorted(pairs.items(), key=lambda kv: -kv[1][0])
    cp_area = float(sum(v[0] for v in pairs.values()))
    log("overlay-internal coplanar same-facing overlaps (< 3 mm): %d pairs, %.4f m2" % (len(pairs), cp_area))
    for (a_, b_), (ar_, sep) in cp[:12]:
        log("   at", np.round(S[a_].mean(0), 2).tolist(), "area %.5f sep %.4f" % (ar_, sep), Snames[Sm[a_]], "|", Snames[Sm[b_]])
    # 4. see-through from the street through every opening's glass
    OPS = jload(os.path.join(CACHE, "openings.json"))["openings"]
    # whole scene (not only the triangles near the overlay): rays through a corner may travel far
    isg = np.array(["glass" in n.lower() for n in obst.matnames])[obst.mat]
    sglass = vit
    ALL = np.concatenate([obst.T[~isg], S[~sglass]])
    own = np.r_[np.zeros(int((~isg).sum()), bool), np.ones(int((~sglass).sum()), bool)]
    FN = np.cross(ALL[:, 1] - ALL[:, 0], ALL[:, 2] - ALL[:, 0])
    FN /= np.maximum(np.linalg.norm(FN, axis=1, keepdims=True), 1e-15)
    rm_all = ray_mesh(ALL)
    rng = np.random.default_rng(7)
    res = Counter()
    per_op = defaultdict(Counter)
    examples = []
    back_rays = []
    seen_glass = set()
    for o in OPS:
        gkey = o.get("base") or o["id"]
        if gkey in seen_glass:
            continue                      # sub-openings share their glass: test it once
        seen_glass.add(gkey)
        Oo, R = np.array(o["O"]), np.array(o["R"])
        g = o["glass"]["rect"]
        zg = o["glass"]["zg"]
        cxg = (g[0] + g[1]) / 2
        sy = heightfield_y(Oo[0] + R[0, 2] * 2, Oo[2] + R[2, 2] * 2)
        yb = (sy - Oo[1]) if np.isfinite(sy) else 0.0
        eyes = np.array([[cxg + dx, yb + 1.65, dz] for dz in (1.2, 3.0, 7.0) for dx in (-6.0, -2.5, 0.0, 2.5, 6.0)])
        ctr = np.array([cxg, (g[2] + g[3]) / 2, zg + 0.05])
        Eg = Oo + eyes @ R.T
        Dc = (Oo + R @ ctr) - Eg
        Lc = np.linalg.norm(Dc, axis=1)
        ih, dh = first_hits(rm_all, Eg, Dc / Lc[:, None])
        clear = ~((ih >= 0) & (dh < Lc - 0.02))
        res["eyes_blocked"] += int((~clear).sum())
        eyes = eyes[clear]
        if not len(eyes):
            continue
        nt = 6
        tg = np.stack([rng.uniform(g[0] + 0.04, g[1] - 0.04, nt), rng.uniform(g[2] + 0.04, g[3] - 0.04, nt),
                       np.full(nt, zg - 0.002)], 1)
        E = np.repeat(eyes, nt, 0)
        P = np.tile(tg, (len(eyes), 1))
        D = P - E
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        og, dg = Oo + E @ R.T, D @ R.T
        idx, dist = first_hits(rm_all, og, dg)
        for k in range(len(og)):
            t = idx[k]
            res["rays"] += 1
            if t < 0:
                cat = "nohit"
            else:
                hl = (og[k] + dg[k] * dist[k] - Oo) @ R
                back = np.dot(FN[t], dg[k]) > 1e-4
                if hl[2] > zg + 0.01:
                    cat = "front_of_glass"          # frame / mullion in front of the pane: not a view inside
                elif hl[2] < zg - 2.2:
                    cat = "deep"
                elif back:
                    cat = "overlay_back" if own[t] else "scene_back"
                    back_rays.append((gkey, cat, og[k] + dg[k] * dist[k], dg[k], t, og[k]))
                    continue
                else:
                    cat = "ok_overlay" if own[t] else "ok_scene"
            res[cat] += 1
            if cat in ("nohit", "deep"):
                per_op[gkey][cat] += 1
                if len(examples) < 60:
                    examples.append(dict(op=gkey, cat=cat, eye=np.round(og[k], 2).tolist(), dir=np.round(dg[k], 3).tolist()))
    # back-face first hits: a front face in the same plane (closed double-sided shell) is what the game draws
    if back_rays:
        Pb = np.array([b_[2] for b_ in back_rays])
        Db = np.array([b_[3] for b_ in back_rays])
        Tb = np.array([b_[4] for b_ in back_rays])
        okb = coincident_front(ALL, FN, Pb, Db, Tb)
        log("back-face first hits: %d, with a coincident front face: %d" % (len(back_rays), int(okb.sum())))
        for (gkey, cat, p_, d_, t_, o_), ok_ in zip(back_rays, okb):
            if ok_:
                res["ok_double_sided"] += 1
                continue
            res[cat] += 1
            per_op[gkey][cat] += 1
            if len(examples) < 60:
                examples.append(dict(op=gkey, cat=cat, eye=np.round(o_, 2).tolist(), dir=np.round(d_, 3).tolist(),
                                     hit=np.round(p_, 3).tolist()))
    bad_ops = sorted(per_op.items(), key=lambda kv: -sum(kv[1].values()))
    leaks = {c_: res[c_] for c_ in ("nohit", "deep", "overlay_back", "scene_back")}
    log("see-through: %d rays, %s; openings with problems: %d" % (res["rays"], leaks, len(bad_ops)))
    for k_, v_ in bad_ops[:15]:
        log("   ", k_, dict(v_))
    # 5. box overlaps
    ovl = box_overlaps(OPS)
    log("overlapping display boxes:", len(ovl), ovl[:6])
    # 6. hidden boxes: most rays from 3 m in front end on scene geometry between the glass and the box front
    hidden = []
    for o in OPS:
        b = o["box"]
        if b["depth"] < 0.08:
            continue
        Oo, R = np.array(o["O"]), np.array(o["R"])
        a, bb, c, d = b["rect"]
        zg = o["glass"]["zg"]
        P = np.stack([rng.uniform(a + 0.02, bb - 0.02, 16), rng.uniform(c + 0.02, d - 0.02, 16), np.full(16, zg - 0.001)], 1)
        eye = np.array([(a + bb) / 2, c + 1.6, 3.0])
        D = P - eye
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        idx, dist = first_hits(rm_all, np.tile(Oo + R @ eye, (16, 1)), D @ R.T)
        n_hidden = 0
        for k in range(16):
            if idx[k] >= 0 and not own[idx[k]]:
                hl = (Oo + R @ eye + (D[k] @ R.T) * dist[k] - Oo) @ R
                if b["z_front"] < hl[2] < zg:
                    n_hidden += 1
        if n_hidden >= 13:
            hidden.append(o["id"])
    log("display boxes hidden behind a scene panel:", len(hidden), hidden[:10])
    # 7. normals vs winding
    g = __import__("glbio").read_glb(RAW_OUT)
    flipped = 0
    total = 0
    for nd in g["nodes"]:
        for p in nd["prims"]:
            T = p["pos"][p["idx"]]
            Nn2 = p["nrm"][p["idx"]].mean(1)
            fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            flipped += int((np.einsum("ij,ij->i", fn, Nn2) < 0).sum())
            total += len(T)
    log("faces whose stored normal disagrees with the winding: %d of %d" % (flipped, total))
    jdump(dict(overlay_tris=int(len(S)), edge_cross_overlay=len(bad1), edge_cross_scene=len(bad2),
               zfight_scene=len(zf_pts), zfight_scene_at=zf_pts[:40],
               zfight_overlay_pairs=len(pairs), zfight_overlay_area_m2=round(cp_area, 5),
               zfight_overlay_at=[dict(c=np.round(S[a_].mean(0), 3).tolist(), area=round(v[0], 6), sep=round(v[1], 5),
                                       mats=[Snames[Sm[a_]], Snames[Sm[b_]]]) for (a_, b_), v in cp[:60]],
               see_through=dict(res), see_through_leaks=leaks, see_through_bad_openings=[[k_, dict(v_)] for k_, v_ in bad_ops[:80]],
               see_through_examples=examples, box_overlaps=ovl, hidden_boxes=hidden, normal_flips=flipped,
               bad_overlay_tris=sorted(bad1)[:200],
               bad_scene=[dict(c=[round(float(v), 3) for v in O[t].mean(0)], mat=obst.matnames[obst.mat[np.where(near)[0][t]]])
                          for t in sorted(bad2)[:200]]), os.path.join(CACHE, "check.json"))
    log("check done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

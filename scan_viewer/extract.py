"""Automatic subject extraction for 3D Gaussian Splatting PLY scans (person or objects).

Works without assumptions about units or axes:
  1. scene centre  = median of opaque splats close to the origin (3DGS scenes are centred on the capture)
  2. ground plane  = largest RANSAC plane just below the centre (prefers planes roughly perpendicular to y)
  3. candidates    = splats above the ground, near the centre, not huge background splats
  4. clustering    = DBSCAN with an eps derived from the local point density
  5. subject       = best cluster (size, closeness to the centre axis); optionally neighbouring objects
  6. refinement    = statistical outlier removal, then re-add soft low-opacity splats around the subject

CLI:  python extract.py in.ply out.ply [--group] [--radius 2.5]
The output keeps every original PLY property (SH, scale, rotation ...) and the original coordinates.
"""
import argparse, json, math, sys
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

DEFAULTS = dict(radius=2.5, group='auto', min_opacity=0.1, ground_margin=2.5, keep_soft=True)


def load(path_or_bytes):
    ply = PlyData.read(path_or_bytes)
    v = ply['vertex'].data
    names = v.dtype.names
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-v['opacity'].astype(np.float64))) if 'opacity' in names else np.ones(len(v))
    if 'scale_0' in names:
        smax = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1).astype(np.float64)).max(1)
    else:
        smax = np.zeros(len(v))
    if 'f_dc_0' in names:
        rgb = np.clip(0.5 + 0.28209479 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1), 0, 1)
    elif 'red' in names:
        rgb = np.stack([v['red'], v['green'], v['blue']], 1) / 255.0
    else:
        rgb = np.full((len(v), 3), 0.7)
    ok = np.isfinite(xyz).all(1)
    return dict(v=v, xyz=xyz, op=op, smax=smax, rgb=rgb, ok=ok, splat='rot_0' in names)


def _center(S):
    xyz, op, ok = S['xyz'], S['op'], S['ok']
    o = ok & (op > 0.3)
    r = np.linalg.norm(xyz[o], axis=1)
    rc = max(np.percentile(r, 20), 1e-3)
    c0 = np.median(xyz[o][r < rc], axis=0)
    # one refinement step around the first estimate
    r2 = np.linalg.norm(xyz[o] - c0, axis=1)
    c0 = np.median(xyz[o][r2 < rc], axis=0)
    return c0, rc


def _ground(S, c0, scale):
    xyz, op, ok = S['xyz'], S['op'], S['ok']
    sel = ok & (op > 0.3) & (np.linalg.norm(xyz - c0, axis=1) < 4 * scale)
    P = xyz[sel]
    if len(P) > 400000:
        P = P[np.random.default_rng(0).choice(len(P), 400000, replace=False)]
    cands, rest = [], P
    o3d.utility.random.seed(7)
    for _ in range(6):
        if len(rest) < 500: break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rest))
        (a, b, c, d), inl = pc.segment_plane(0.02 * scale, 3, 2000)
        n = np.array([a, b, c]); nn = np.linalg.norm(n); n, d = n / nn, d / nn
        # "oben" = die Seite mit mehr Masse (manche Scans sind kopfüber oder die Mitte liegt im Boden)
        dist = n @ c0 + d
        if dist < 0: n, d, dist = -n, -d, -dist
        if dist < 0.15 * scale:                     # Mitte liegt (fast) auf der Ebene -> Seite nach Masse wählen
            hp = P @ n + d; band = 0.04 * scale
            if (hp < -band).sum() > (hp > band).sum(): n, d, dist = -n, -d, -dist
        # das Subjekt steht darauf: Mitte nicht weit darunter, Ebene nicht zu weit entfernt
        if -0.5 * scale < dist < 3.0 * scale and len(inl) > 0.03 * len(P):
            h_in = rest[inl] @ n + d
            cands.append(dict(n=n, d=d, dist=dist, inl=len(inl), thick=float(np.percentile(np.abs(h_in), 90)),
                              score=len(inl) * (0.3 + abs(n[1]))))     # phone/COLMAP captures: up ~ +-y
        mask = np.ones(len(rest), bool); mask[inl] = False; rest = rest[mask]
    if not cands:
        raise RuntimeError('keine Bodenebene gefunden')
    top = max(cands, key=lambda c: c['score'])
    # among planes parallel to the dominant one (ground, table tops, ledges) the LOWEST with real support
    # is where the subject stands - a big ledge at hip height must not cut off the legs
    par = [c for c in cands if abs(c['n'] @ top['n']) > 0.95 and c['inl'] > 0.25 * top['inl']]
    g = max(par, key=lambda c: c['dist'])
    return g['n'], g['d'], g['thick'], g['inl']


def extract(S, radius=None, group=None, min_opacity=None, ground_margin=None, keep_soft=None, center=None, debug=None):
    p = dict(DEFAULTS)
    for k, val in dict(radius=radius, group=group, min_opacity=min_opacity, ground_margin=ground_margin, keep_soft=keep_soft).items():
        if val is not None: p[k] = val
    xyz, op, smax, ok = S['xyz'], S['op'], S['smax'], S['ok']
    c0, scale = _center(S)
    if center is not None: c0 = np.asarray(center, float)
    n, d, thick, n_in = _ground(S, c0, scale)
    h = xyz @ n + d
    c_h = n @ c0 + d
    # horizontal distance to the vertical axis through the centre
    rel = xyz - c0; rh = np.linalg.norm(rel - np.outer(rel @ n, n), axis=1)
    t = max(p['ground_margin'] * thick, 0.008 * scale)
    # Flächenmodus: liegt kaum etwas über der Ebene (Wand, Pflaster, Relief), ist die Fläche selbst das Subjekt
    near = ok & (op > 0.3) & (rh < p['radius'] * scale) & (np.abs(h) < 2 * scale)
    if near.sum() and ((h > 4 * t) & near).sum() < 0.08 * near.sum():
        cand = np.where(ok & (op > p['min_opacity']) & (rh < p['radius'] * scale) & (np.abs(h) < 0.5 * scale) & (smax < 0.15 * scale))[0]
        nnd = cKDTree(xyz[cand]).query(xyz[cand], k=2)[0][:, 1]
        eps = float(np.clip(3.0 * np.median(nnd), 0.004 * scale, 0.08 * scale))
        lab = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[cand])).cluster_dbscan(eps=eps, min_points=8))
        sizes = np.bincount(lab[lab >= 0])
        sel = cand[np.isin(lab, np.where(sizes >= 0.05 * sizes.max())[0])]
        _, ind = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[sel])).remove_statistical_outlier(20, 2.2)
        sel = sel[np.asarray(ind)]
        p['group'] = False
        ext = xyz[sel].max(0) - xyz[sel].min(0)
        info = dict(total=int(len(xyz)), selected=int(len(sel)), clusters=int(len(sizes)), taken=int((sizes >= 0.05 * sizes.max()).sum()),
                    center=c0.tolist(), up=n.tolist(), plane_d=float(d), ground_inliers=n_in, eps=eps, scale=float(scale),
                    height=float(np.linalg.norm(ext)), params=p, mode='surface',
                    bbox_min=xyz[sel].min(0).tolist(), bbox_max=xyz[sel].max(0).tolist())
        return sel, info
    hmax = max(3.2 * c_h, 2.0 * scale)
    c_h = max(c_h, 0.0)
    cand = np.where(ok & (op > p['min_opacity']) & (h > t) & (h < hmax) & (rh < p['radius'] * scale)
                    & (smax < 0.15 * scale))[0]
    if len(cand) < 50:
        raise RuntimeError('zu wenige Punkte über dem Boden')
    tree = cKDTree(xyz[cand])
    nnd = tree.query(xyz[cand], k=2)[0][:, 1]
    eps = float(np.clip(3.0 * np.median(nnd), 0.004 * scale, 0.08 * scale))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[cand]))
    lab = np.array(pc.cluster_dbscan(eps=eps, min_points=8))
    if lab.max() < 0:
        raise RuntimeError('kein Cluster gefunden')
    clusters = []
    for k in range(lab.max() + 1):
        m = lab == k
        if m.sum() < 30: continue
        ch = h[cand][m]; cr = rh[cand][m]
        clusters.append(dict(k=k, n=int(m.sum()), rmin=float(cr.min()), rmed=float(np.median(cr)),
                             hext=float(ch.max() - ch.min()), hmed=float(np.median(ch)), cent=xyz[cand][m].mean(0)))
    # subject: big, reaches the centre axis, has vertical extent
    def score(c): return c['n'] * (c['hext'] / (c['hext'] + t)) / (1 + (c['rmin'] / (0.3 * scale)) ** 2)
    main = max(clusters, key=score)
    take = [main['k']]
    # vertical continuation: clusters stacked under/over the subject inside its footprint belong to it
    # (legs separated from the torso by a gap at the jacket hem, a head separated by a scarf ...)
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u); w = np.cross(n, u)
    cp = xyz[cand]; U, Wv, H = cp @ u, cp @ w, h[cand]
    def box(mask): return np.array([U[mask].min(), U[mask].max(), Wv[mask].min(), Wv[mask].max(), H[mask].min(), H[mask].max()])
    mb = box(np.isin(lab, take)); grow = 0.05 * scale; changed = True
    while changed:
        changed = False
        for c in clusters:
            if c['k'] in take: continue
            m = lab == c['k']; cb = box(m)
            iu = max(0, min(cb[1], mb[1] + grow) - max(cb[0], mb[0] - grow)); iw = max(0, min(cb[3], mb[3] + grow) - max(cb[2], mb[2] - grow))
            area = max((cb[1] - cb[0]) * (cb[3] - cb[2]), 1e-12)
            vgap = max(cb[4] - mb[5], mb[4] - cb[5], 0)
            flat_ground = c['hext'] < 4 * t and c['hmed'] < 3 * t          # streaks of floor texture, not part of the subject
            if iu * iw / area > 0.6 and vgap < 0.15 * scale and not flat_ground:
                take.append(c['k']); mb = box(np.isin(lab, take)); changed = True
    if p['group'] == 'auto':
        # erst NACH dem Zusammenführen (Beine + Oberkörper) entscheiden: mehrere ähnlich große, aufrechte
        # Objekte nebeneinander (Säulenreihe, Stillleben) -> als Gruppe
        tm = np.isin(lab, take); mp = xyz[cand][tm]; mext = float(np.linalg.norm(mp.max(0) - mp.min(0)))
        mh = float(h[cand][tm].max() - h[cand][tm].min()); mn = int(tm.sum()); mc = mp.mean(0)
        peers = [c for c in clusters if c['k'] not in take and c['n'] > 0.3 * mn and c['hext'] > 0.7 * mh
                 and np.linalg.norm(c['cent'] - mc) < 2.5 * mext]
        p['group'] = len(peers) > 0
    sel_main = cand[np.isin(lab, take)]
    if p['group']:
        # neighbouring objects are often scanned more sparsely: re-cluster with a coarser eps and take every
        # upright object (not floor texture) inside the search radius
        up_c = cand[h[cand] > 2 * t]
        pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[up_c]))
        lab2 = np.array(pc2.cluster_dbscan(eps=2.5 * eps, min_points=5))
        extra = []
        for k in range(lab2.max() + 1):
            m = lab2 == k
            if m.sum() < 30: continue
            hh = h[up_c][m]
            if hh.max() - hh.min() > 4 * t and np.median(hh) > 3 * t and hh.max() > 8 * t: extra.append(up_c[m])   # upright, not floor texture
        if extra: sel_main = np.union1d(sel_main, np.concatenate(extra))
    sel = sel_main
    pcs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[sel]))
    _, ind = pcs.remove_statistical_outlier(20, 2.2)
    sel = sel[np.asarray(ind)]
    if p['keep_soft']:
        # soft, low-opacity splats hugging the subject make it look like the original scan
        soft = np.where(ok & (op > 0.02) & (op <= p['min_opacity']) & (h > t) & (smax < 0.15 * scale))[0]
        if len(soft):
            dd, _ = cKDTree(xyz[sel]).query(xyz[soft])
            sel = np.union1d(sel, soft[dd < eps])
    if debug is not None: debug.update(cand=cand, lab=lab, clusters=clusters, main=main, take=list(take))
    # floor-level splats survive only directly under an upright part of the subject (feet, bottle base),
    # this removes ledge edges and floor texture that touch the subject
    hsel = h[sel]; upright = sel[hsel > 6 * t]; low = sel[hsel <= 6 * t]
    if len(upright) and len(low):
        uv = lambda idx: np.c_[xyz[idx] @ u, xyz[idx] @ w]
        dd, _ = cKDTree(uv(upright)).query(uv(low))
        sel = np.union1d(upright, low[dd < 0.12 * scale])
    hs = h[sel]
    info = dict(total=int(len(xyz)), selected=int(len(sel)), clusters=len(clusters), taken=len(take),
                center=c0.tolist(), up=n.tolist(), plane_d=float(d), ground_inliers=n_in, eps=eps, scale=float(scale),
                height=float(hs.max() - max(hs.min(), 0)), params=p,
                bbox_min=xyz[sel].min(0).tolist(), bbox_max=xyz[sel].max(0).tolist())
    return sel, info


def save(S, sel, path):
    PlyData([PlyElement.describe(S['v'][sel], 'vertex')], byte_order='<').write(path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--group', action='store_true'); ap.add_argument('--radius', type=float)
    a = ap.parse_args()
    S = load(a.inp)
    sel, info = extract(S, radius=a.radius, group=a.group)
    save(S, sel, a.out)
    json.dump(info, open(a.out.rsplit('.', 1)[0] + '.json', 'w'), indent=1)
    print(json.dumps({k: info[k] for k in ['total', 'selected', 'clusters', 'taken', 'height', 'scale', 'eps', 'up']}, indent=None))

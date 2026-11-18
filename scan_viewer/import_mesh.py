"""Textur-Mesh-Scan (z. B. KIRI Engine: OBJ + MTL + JPG) freistellen und in den Scan-Viewer übernehmen.

  .venv/Scripts/python import_mesh.py <scan.obj> --name "KIRI Person" [--height 1.80]

Ablauf
  1. OBJ mit Eck-UVs laden; Bodenebene per RANSAC (Normale ~ +y, KIRI exportiert y-oben)
  2. Subjekt: Flächen knapp über dem Boden -> Zusammenhangskomponenten -> Komponente um die Aufnahmemitte
     (Boden, Wände/Plakate und Streufragmente fallen weg); Bodenfarbe (Median der Bodenflächen) bestimmt,
     wie hoch am Fuß geschnitten werden muss, damit keine Bodenreste an den Schuhen bleiben
  3. planarer Schnitt über dem Boden, Schnittränder senkrecht auf den Boden ziehen, Sohlen mit einem
     Qualitäts-Dreiecksnetz (triangle) schließen -> wasserdicht; die Original-UVs bleiben erhalten
  4. aufrecht ausrichten (Bodennormale -> +y), Blickrichtung (Fußspitzen) -> +z, auf Körpergröße skalieren
  5. Ausreißer-Spitzen glätten, 1x Loop-Unterteilung (hochauflösend), Prüfung: 0 Randkanten,
     0 nicht-mannigfaltige Kanten, konsistente Orientierung, 1 Komponente
  6. Ausgaben im Scan-Viewer-Format (Viewer-Dateien in 3DGS-Konvention: y nach unten):
       uploads/<id>_<name>.ply                 ganzer Scan als Oberflächen-Splats ("Original")
       uploads/<id>_<name>_quelle/             Original-OBJ/MTL/Textur
       extracted/<id>_<name>_extrahiert.ply    Subjekt als Oberflächen-Splats
       extracted/<id>_<name>_mesh.ply          wasserdichtes Mesh mit Vertexfarben
       extracted/<id>_<name>_mesh.glb          dasselbe Mesh mit Textur (y-oben, Meter)
       extracted/<id>_<name>_extrahiert.json   Kennzahlen
"""
import argparse, json, os, re, shutil, struct, sys, time, uuid
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import triangle as tri

ROOT = os.path.dirname(os.path.abspath(__file__))
UP, EX = os.path.join(ROOT, 'uploads'), os.path.join(ROOT, 'extracted')
FLIP = np.diag([1.0, -1.0, -1.0])            # Viewer-Dateien: 180° um x (3DGS/COLMAP: y nach unten)
SH0 = 0.28209479177387814


# ------------------------------------------------------------------ Laden
def load_obj(path):
    V, VT, FV, FT, mtl = [], [], [], [], None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                V.append(line.split()[1:4])
            elif line.startswith('vt '):
                VT.append(line.split()[1:3])
            elif line.startswith('f '):
                p = [q.split('/') for q in line.split()[1:]]
                for k in range(1, len(p) - 1):            # Polygone als Fächer
                    t = (p[0], p[k], p[k + 1])
                    FV.append([int(x[0]) for x in t]); FT.append([int(x[1]) for x in t])
            elif line.startswith('mtllib'):
                mtl = line.split(None, 1)[1].strip()
    FV = np.array(FV, np.int64); FT = np.array(FT, np.int64)
    if (FV <= 0).any() or (FT <= 0).any(): raise ValueError('relative OBJ-Indizes werden nicht unterstützt')
    return np.array(V, float), np.array(VT, float), FV - 1, FT - 1, mtl


def find_texture(obj_path, mtl, override=None):
    if override: return override
    d = os.path.dirname(obj_path)
    if mtl and os.path.isfile(os.path.join(d, mtl)):
        for line in open(os.path.join(d, mtl), encoding='utf-8', errors='replace'):
            if line.strip().lower().startswith('map_kd'):
                return os.path.join(d, line.split()[-1])
    raise FileNotFoundError('keine Textur (map_Kd) gefunden')


def sample(tex, uv):
    """bilinear, uv in [0,1], v nach oben (OBJ/glTF-Konvention umgerechnet)"""
    H, W = tex.shape[:2]
    x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1.001); y = np.clip((1 - uv[:, 1]) * H - 0.5, 0, H - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x0 + 1] * fx * (1 - fy)
            + tex[y0 + 1, x0] * (1 - fx) * fy + tex[y0 + 1, x0 + 1] * fx * fy)


# ------------------------------------------------------------------ Geometrie-Helfer
def ground_plane(V, thr, iters=4000, seed=1):
    rng = np.random.default_rng(seed)
    cand = V[V[:, 1] < np.percentile(V[:, 1], 40)]
    best = None
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); ln = np.linalg.norm(n)
        if ln < 1e-12: continue
        n /= ln
        if abs(n[1]) < 0.9: continue
        if n[1] < 0: n = -n
        d = -n @ s[0]
        c = np.count_nonzero(np.abs(cand @ n + d) < thr)
        if best is None or c > best[0]: best = (c, n, d)
    _, n, d = best
    for _ in range(3):                                        # kleinste Quadrate auf den Inliers
        inl = np.abs(cand @ n + d) < thr
        Q = cand[inl]; mu = Q.mean(0); n = np.linalg.svd(Q - mu, full_matrices=False)[2][2]
        if n[1] < 0: n = -n
        d = -n @ mu
    return n, float(d), int(inl.sum())


def face_components(F):
    m = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e.sort(1)
    fid = np.tile(np.arange(m), 3)
    key = e[:, 0] * (int(F.max()) + 1) + e[:, 1]
    o = np.argsort(key, kind='stable'); ks, fs = key[o], fid[o]
    same = ks[1:] == ks[:-1]
    G = coo_matrix((np.ones(same.sum()), (fs[:-1][same], fs[1:][same])), shape=(m, m))
    return connected_components(G, directed=False)


def compact(V, F):
    used = np.unique(F); mp = -np.ones(len(V), np.int64); mp[used] = np.arange(len(used))
    return V[used], mp[F], used


def half_edges(F):
    return np.stack([F, np.roll(F, -1, 1)], 2).reshape(-1, 2)   # (a->b) je Ecke, Reihenfolge f*3+k


def topo_check(V, F):
    he = half_edges(F); m = int(F.max()) + 1
    k_dir = he[:, 0] * m + he[:, 1]
    und = np.sort(he, 1); k_und = und[:, 0] * m + und[:, 1]
    _, cnt = np.unique(k_und, return_counts=True)
    dup_dir = len(k_dir) - len(np.unique(k_dir))
    ncomp, _ = face_components(F)
    nv = len(np.unique(F)); ne = len(cnt); nf = len(F)
    P = V[F]; cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    vol = float(np.einsum('ij,ij->i', P[:, 0], cr).sum() / 6)
    area = np.linalg.norm(cr, axis=1) / 2
    return dict(boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                misoriented_edges=int(dup_dir), components=int(ncomp), euler=int(nv - ne + nf),
                genus=int((2 - (nv - ne + nf)) // 2), volume=vol, degenerate_faces=int((area < 1e-14).sum()),
                watertight=bool((cnt == 2).all() and dup_dir == 0))


def vertex_normals(V, F):
    P = V[F]; fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])     # flächengewichtet
    N = np.zeros_like(V)
    for k in range(3): np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)


def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-12: return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


# ------------------------------------------------------------------ Schnitt, Wände, Sohlen
def slice_above(V, F, UVc, h, c):
    """Teil mit h >= c behalten; geschnittene Dreiecke aufteilen (Kantenpunkte geteilt, UVs je Ecke interpoliert)."""
    ab = h[F] >= c; na = ab.sum(1)
    keepF, keepU = [F[na == 3]], [UVc[na == 3]]
    newpos, edge_id = [], {}

    def epoint(i, j):                              # gemeinsamer Punkt auf Kante (i, j), kanonisch i < j
        k = (min(i, j), max(i, j))
        if k not in edge_id:
            a, b = k; t = (c - h[a]) / (h[b] - h[a])
            edge_id[k] = len(V) + len(newpos); newpos.append(V[a] + t * (V[b] - V[a]))
        return edge_id[k]

    def euv(fi, ka, kb):                            # UV des Kantenpunkts in Fläche fi (kanonische Reihenfolge)
        i, j = F[fi, ka], F[fi, kb]
        if i > j: i, j, ka, kb = j, i, kb, ka
        t = (c - h[i]) / (h[j] - h[i])
        return UVc[fi, ka] + t * (UVc[fi, kb] - UVc[fi, ka])

    tF, tU = [], []
    for fi in np.where((na == 1) | (na == 2))[0]:
        f = F[fi]; a = ab[fi]
        if na[fi] == 1:                                    # (A, B, C): A oben -> (A, AB, CA)
            k = int(np.where(a)[0][0]); A, B, C = k, (k + 1) % 3, (k + 2) % 3
            tF.append([f[A], epoint(f[A], f[B]), epoint(f[C], f[A])])
            tU.append([UVc[fi, A], euv(fi, A, B), euv(fi, C, A)])
        else:                                              # (X, Y, Z): X unten -> (XY, Y, Z), (XY, Z, ZX)
            k = int(np.where(~a)[0][0]); X, Y, Z = k, (k + 1) % 3, (k + 2) % 3
            xy, zx = epoint(f[X], f[Y]), epoint(f[Z], f[X])
            uxy, uzx = euv(fi, X, Y), euv(fi, Z, X)
            tF += [[xy, f[Y], f[Z]], [xy, f[Z], zx]]
            tU += [[uxy, UVc[fi, Y], UVc[fi, Z]], [uxy, UVc[fi, Z], uzx]]
    V2 = np.vstack([V, np.array(newpos).reshape(-1, 3)])
    F2 = np.vstack(keepF + [np.array(tF, np.int64).reshape(-1, 3)])
    U2 = np.concatenate(keepU + [np.array(tU).reshape(-1, 3, 2)])
    V2, F2, _ = compact(V2, F2)
    return V2, F2, U2


def boundary_half_edges(F):
    he = half_edges(F); m = int(F.max()) + 1
    k = he[:, 0] * m + he[:, 1]; kr = he[:, 1] * m + he[:, 0]
    isb = ~np.isin(k, kr)
    idx = np.where(isb)[0]
    return he[idx], idx // 3, idx % 3                     # (a, b), Fläche, Ecke von a


def walls_and_soles(V, F, UVc, rows, uv_sole, max_area):
    """Randschleifen (eben, y = c) senkrecht auf y = 0 ziehen und unten mit einem Dreiecksnetz schließen."""
    be, bf, bk = boundary_half_edges(F)
    c = float(np.median(V[np.unique(be), 1]))
    ring = np.unique(be)
    layer = {0: {int(v): int(v) for v in ring}}
    newV = []
    for r in range(1, rows + 1):
        y = c * (1 - r / rows); layer[r] = {}
        for v in ring:
            p = V[v].copy(); p[1] = y
            layer[r][int(v)] = len(V) + len(newV); newV.append(p)
    V = np.vstack([V, np.array(newV)])
    uva = UVc[bf, bk]; uvb = UVc[bf, (bk + 1) % 3]
    wF, wU = [], []
    for (a, b), ua, ub in zip(be, uva, uvb):
        for r in range(rows):                           # Viereck (b, a, a', b') unterhalb der Kante a->b
            A0, B0, A1, B1 = layer[r][a], layer[r][b], layer[r + 1][a], layer[r + 1][b]
            wF += [[B0, A0, A1], [B0, A1, B1]]; wU += [[ub, ua, ua], [ub, ua, ub]]
    # Sohlen: Polygon(e) der untersten Lage in (x, z), Segmente b'->a' (Orientierung folgt aus triangle: ccw)
    bottom = np.array([layer[rows][int(v)] for v in ring]); loc = {int(g): i for i, g in enumerate(bottom)}
    seg = np.array([[loc[layer[rows][a]], loc[layer[rows][b]]] for a, b in be])
    P2 = V[bottom][:, [0, 2]]
    T = tri.triangulate(dict(vertices=P2, segments=seg), f'pq28Ya{max_area:.10f}')
    tv = T['vertices']; tt = T['triangles']
    if len(tv) < len(P2) or not np.allclose(tv[:len(P2)], P2): raise RuntimeError('triangle hat Randpunkte verändert')
    extra = tv[len(P2):]
    gidx = np.concatenate([bottom, len(V) + np.arange(len(extra))])
    V = np.vstack([V, np.column_stack([extra[:, 0], np.zeros(len(extra)), extra[:, 1]])])
    capF = gidx[tt]
    # ccw in (x, z) ergibt Normale nach -y (unten) -> passt; zur Sicherheit an den Wandkanten prüfen
    capN = np.cross(V[capF[:, 1]] - V[capF[:, 0]], V[capF[:, 2]] - V[capF[:, 0]])[:, 1]
    if (capN > 0).mean() > 0.5: capF = capF[:, ::-1]
    F = np.vstack([F, np.array(wF, np.int64), capF])
    UVc = np.concatenate([UVc, np.array(wU), np.broadcast_to(uv_sole, (len(capF), 3, 2))])
    return V, F, UVc, c


# ------------------------------------------------------------------ Beinbrücke (verdeckter Spalt zwischen den Beinen)
TM = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])            # (x, y, z) -> Manifold (x, -z, y): Schnitte entlang y


def pick_uv(tex, U, faces, pct=50):
    """UV-Punkt in einem gleichmäßigen Texturbereich, dessen Helligkeit dem pct-Perzentil der Flächen entspricht."""
    uvl = U[faces].mean(1); lum = sample(tex, uvl).mean(1)
    target = np.percentile(lum, pct)
    Ht, Wt = tex.shape[:2]; r = max(4, Wt // 400)
    cand = faces[np.argsort(np.abs(lum - target))[:400]]
    best, bs = None, 1e9
    for f in cand:
        u = U[f].mean(0); x = int(u[0] * Wt); y = int((1 - u[1]) * Ht)
        win = tex[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        sc = win.std() + abs(win.mean() - target)
        if sc < bs: bs, best = sc, u
    return best


def to_manifold(V, F, UVc):
    import manifold3d as mf
    uvq = np.round(UVc.reshape(-1, 2) * 2 ** 22).astype(np.int64)
    uk, inv = np.unique(np.column_stack([F.ravel(), uvq]), axis=0, return_inverse=True)
    vid = uk[:, 0]; prim = np.r_[True, vid[1:] != vid[:-1]]
    rep = np.maximum.accumulate(np.where(prim, np.arange(len(vid)), 0))
    props = np.column_stack([V[vid] @ TM.T, uk[:, 1:] / 2 ** 22]).astype(np.float32)
    mesh = mf.Mesh(vert_properties=props, tri_verts=inv.ravel().reshape(-1, 3).astype(np.uint32),
                   merge_from_vert=np.where(~prim)[0].astype(np.uint32), merge_to_vert=rep[~prim].astype(np.uint32))
    return mf.Manifold(mesh)


def from_manifold(M):
    m = M.to_mesh(); P = np.asarray(m.vert_properties, float); T = np.asarray(m.tri_verts, np.int64)
    idx = np.arange(len(P)); mf_, mt = np.asarray(m.merge_from_vert, np.int64), np.asarray(m.merge_to_vert, np.int64)
    if len(mf_): idx[mf_] = mt
    UVc = P[T][:, :, 3:5]
    V, F, _ = compact(P[:, :3] @ TM, idx[T])
    return V, F, UVc


def vline_hits(polys, xs):
    """vordere/hintere Kontur (max/min z) der Schnittpolygone an senkrechten Linien x = xs"""
    zb = np.full(len(xs), np.inf); zf = np.full(len(xs), -np.inf)
    for P in polys:
        P = np.asarray(P, float); Q = np.roll(P, -1, 0)
        x0, x1, y0, y1 = P[:, 0], Q[:, 0], P[:, 1], Q[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (xs[:, None] - x0) / (x1 - x0)
        ok = (t >= 0) & (t <= 1) & (x1 != x0)
        z = -(y0 + t * (y1 - y0))
        zb = np.minimum(zb, np.where(ok, z, np.inf).min(1)); zf = np.maximum(zf, np.where(ok, z, -np.inf).max(1))
    return zb, zf


def carve_leg_bridge(V, F, UVc, H, tex, gcol, log):
    """Photogrammetrie füllt den verdeckten Raum zwischen den Beinen oft von vorn zu einer massiven Brücke.
    Erkennung über waagerechte Schnitte: unten zwei getrennte Beine, darüber ein Teil mit Kerbe (hinten/vorn)
    zwischen den Beinachsen bis zum Schritt. Dann wird ein schlanker, nach oben spitz zulaufender Keil
    (waagerecht taillenförmig wie der Spalt zwischen runden Oberschenkeln) per Boolean entfernt."""
    import manifold3d as mf
    A = to_manifold(V, F, UVc)
    levels = np.arange(0.03, 0.62, 0.004) * H
    sl = []
    for h in levels:
        cs = A.slice(float(h))
        comps = [c for c in cs.decompose() if c.area() > 2e-4]
        sl.append((h, comps, cs.to_polygons()))
    nsep = 0
    while nsep < len(sl) and len(sl[nsep][1]) >= 2: nsep += 1
    if nsep < 3 or nsep >= len(sl) - 3:
        log('Beinbrücke: keine (Beine unten nicht getrennt oder bis oben getrennt)'); return V, F, UVc, None
    gaps, centers = [], []
    for h, comps, _ in sl[:nsep]:
        b = sorted([c.bounds() for c in comps], key=lambda b: b[0] + b[2])
        gaps.append(b[-1][0] - b[0][2]); centers.append(((b[0][0] + b[0][2]) / 2, (b[-1][0] + b[-1][2]) / 2, (b[0][2] + b[-1][0]) / 2))
    # Referenz: größter Abstand der getrennten Beine (dicht unter der Brücke wachsen schon "Finnen" aufeinander zu)
    iref = int(np.argmax(gaps)); gap0 = float(gaps[iref]); h_ref = sl[iref][0]
    xL, xR, xg = np.median(np.array(centers[max(0, nsep - 4):nsep]), 0)
    xg = centers[iref][2]
    h_sep = sl[nsep - 1][0]
    xs = np.linspace(xL, xR, 60); mid = (xs > xL + 0.25 * (xR - xL)) & (xs < xR - 0.25 * (xR - xL))
    hs, xn, zm, hd = [], [], [], []
    miss = 0
    for h, comps, polys in sl[nsep:]:
        zb, zf = vline_hits(polys, xs)
        if not (np.isfinite(zb).all() and np.isfinite(zf).all()): break
        db = zb[mid].max() - max(zb[0], zb[-1]); dfr = min(zf[0], zf[-1]) - zf[mid].min()
        if max(db, dfr) < 0.012:
            miss += 1
            if miss > 2: break
            continue
        miss = 0
        hs.append(h); xn.append(xs[mid][np.argmax(zb[mid])] if db >= dfr else xs[mid][np.argmin(zf[mid])])
        zm.append((zb.min() + zf.max()) / 2); hd.append((zf.max() - zb.min()) / 2)
    # echter Schritt: Beine laufen erst ab ~0.35-0.45 H zusammen; eine Brücke beginnt viel tiefer und reicht weit hinauf
    if not hs or h_sep > 0.3 * H or hs[-1] - h_sep < 0.1 * H or gap0 < 0.01:
        log(f'Beinbrücke: keine (Beine getrennt bis {h_sep / H:.2f} H)'); return V, F, UVc, None
    h_c = hs[-1]
    a, b = np.polyfit(np.r_[hs, h_ref], np.r_[xn, xg], 1)
    zmf = np.poly1d(np.polyfit(hs, zm, 1)); hdf = np.poly1d(np.polyfit(hs, hd, 1))
    log(f'Beinbrücke erkannt: Beine getrennt bis {h_sep / H:.3f} H (Spalt {gap0 * 100:.1f} cm bei {h_ref / H:.3f} H), verbunden bis Schritt {h_c / H:.3f} H')
    # Keil aufbauen: Ringe von h_lo bis zur Spitze
    h_lo, h_top = h_sep - 0.012 * H, h_c + 0.004 * H
    lb = A.bounding_box(); z0, z1 = -lb[4] - 0.05, -lb[1] + 0.05          # Manifold-y = -z
    ring_h = np.linspace(h_lo, h_top, 48)
    zs = np.linspace(z0, z1, 15)
    rings = []
    for h in ring_h:
        f = np.clip((h_top - h) / (h_top - h_ref), 0, 1)               # Spalt läuft linear zum Schritt hin zu
        w = max(gap0 * f, 0.002)
        q = np.clip((zs - zmf(min(h, h_c))) / max(hdf(min(h, h_c)), 0.03), -1, 1)
        hw = 0.5 * w * (0.55 + 0.45 * q ** 2)
        xc = a * min(h, h_c) + b
        R = np.r_[np.column_stack([xc + hw, np.full(len(zs), h), zs]), np.column_stack([xc - hw, np.full(len(zs), h), zs])[::-1]]
        rings.append(R)
    n = len(rings[0]); K = len(rings)
    P = np.vstack(rings + [rings[0].mean(0, keepdims=True), rings[-1].mean(0, keepdims=True)])
    cb, ct = K * n, K * n + 1
    Fc = []
    for k in range(K - 1):
        for i in range(n):
            a0, a1, b0, b1 = k * n + i, k * n + (i + 1) % n, (k + 1) * n + i, (k + 1) * n + (i + 1) % n
            Fc += [[a0, a1, b1], [a0, b1, b0]]
    for i in range(n):
        Fc += [[cb, (i + 1) % n, i], [ct, (K - 1) * n + i, (K - 1) * n + (i + 1) % n]]
    Fc = np.array(Fc, np.int64)
    Pp = P[Fc]; vol = np.einsum('ij,ij->i', Pp[:, 0], np.cross(Pp[:, 1], Pp[:, 2])).sum()
    if vol < 0: Fc = Fc[:, ::-1]
    # Innenseiten der Beine: dunkler, gleichmäßiger Hosenstoff aus dem Beinbereich
    fh = V[F][:, :, 1].mean(1)
    legf = np.where((fh > h_sep) & (fh < h_c))[0]
    uv_in = pick_uv(tex, UVc, legf, 40)
    C = mf.Manifold(mf.Mesh(vert_properties=np.column_stack([P @ TM.T, np.broadcast_to(uv_in, (len(P), 2))]).astype(np.float32),
                            tri_verts=Fc.astype(np.uint32)))
    R = A - C
    parts = R.decompose(); R = max(parts, key=lambda m: m.volume())
    Vn, Fn, Un = from_manifold(R)
    # Innenseiten der Beine, auf die der Scan den Boden projiziert hat (bodenfarben) -> Hosenstoff
    cen = Vn[Fn].mean(1); h = cen[:, 1]
    wz = gap0 * np.clip((h_top - h) / (h_top - h_ref), 0, 1)
    zone = (h > h_lo) & (h < h_c) & (np.abs(cen[:, 0] - (a * np.minimum(h, h_c) + b)) < wz / 2 + 0.04)
    fc = sample(tex, Un.mean(1)); lum = fc.mean(1)
    pants = np.percentile(sample(tex, UVc[legf].mean(1)).mean(1), 90)        # hellste 10 % des Hosenstoffs
    bad = zone & ((np.linalg.norm(fc - gcol, axis=1) < 0.15) | ((lum > max(0.3, pants + 0.06)) & (np.ptp(fc, 1) < 0.12)))
    Un[bad] = uv_in
    log(f'Keil entfernt: Volumen {A.volume() - R.volume():.5f} m³, {len(parts) - 1} abgetrennte Teile verworfen, '
        f'{int(bad.sum())} bodenfarbene Innenflächen umgefärbt')
    return Vn, Fn, Un, dict(h_sep=float(h_sep), h_crotch=float(h_c), gap_m=gap0, removed_m3=float(A.volume() - R.volume()),
                            recolored_faces=int(bad.sum()))


# ------------------------------------------------------------------ Glätten, Unterteilen
def despike(V, F, k=2.5, iters=2):
    """einzelne Ausreißer-Spitzen auf den Nachbarschnitt ziehen (Abstand > k x mittlere Kantenlänge)."""
    e = np.unique(np.sort(half_edges(F), 1), axis=0)
    n_moved = 0
    for _ in range(iters):
        S = np.zeros_like(V); cnt = np.zeros(len(V)); L = np.zeros(len(V))
        ln = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
        for a, b in ((0, 1), (1, 0)):
            np.add.at(S, e[:, a], V[e[:, b]]); np.add.at(cnt, e[:, a], 1); np.add.at(L, e[:, a], ln)
        avg = S / cnt[:, None]; L /= cnt
        off = np.linalg.norm(V - avg, axis=1); bad = off > k * L
        n_moved += int(bad.sum()); V = V.copy(); V[bad] = avg[bad]
        if not bad.any(): break
    return V, n_moved


def loop_subdivide(V, F, UVc):
    """eine Loop-Unterteilung (geschlossene Fläche); UVs linear je Ecke."""
    he = half_edges(F); und = np.sort(he, 1)
    E, inv = np.unique(und, axis=0, return_inverse=True); inv = inv.ravel()
    nE = len(E); fe = inv.reshape(-1, 3)                     # Kante (k, k+1) je Fläche
    opp = F[:, [2, 0, 1]].ravel()                            # gegenüberliegende Ecke je Halbkante
    osum = np.zeros((nE, 3)); ocnt = np.zeros(nE)
    np.add.at(osum, inv, V[opp]); np.add.at(ocnt, inv, 1)
    if not (ocnt == 2).all(): raise RuntimeError('Loop-Unterteilung braucht eine geschlossene Mannigfaltigkeit')
    Ev = 0.375 * (V[E[:, 0]] + V[E[:, 1]]) + 0.125 * osum
    nS = np.zeros((len(V), 3)); deg = np.zeros(len(V))
    for a, b in ((0, 1), (1, 0)):
        np.add.at(nS, E[:, a], V[E[:, b]]); np.add.at(deg, E[:, a], 1)
    beta = (0.625 - (0.375 + 0.25 * np.cos(2 * np.pi / deg)) ** 2) / deg
    Vn = (1 - deg * beta)[:, None] * V + beta[:, None] * nS
    V2 = np.vstack([Vn, Ev]); m = len(V) + fe
    a, b, c = F[:, 0], F[:, 1], F[:, 2]; ab, bc, ca = m[:, 0], m[:, 1], m[:, 2]
    F2 = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    ua, ub, uc = UVc[:, 0], UVc[:, 1], UVc[:, 2]
    uab, ubc, uca = (ua + ub) / 2, (ub + uc) / 2, (uc + ua) / 2
    U2 = np.concatenate([np.stack([ua, uab, uca], 1), np.stack([uab, ub, ubc], 1), np.stack([uca, ubc, uc], 1), np.stack([uab, ubc, uca], 1)])
    return V2, F2, U2


# ------------------------------------------------------------------ Textur
def texture_mask(UVc, shape, dil=6):
    H, W = shape
    img = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(img)
    px = np.column_stack([UVc[..., 0].ravel() * W, (1 - UVc[..., 1].ravel()) * H]).reshape(-1, 3, 2)
    for t in px: dr.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=255)
    m = np.asarray(img) > 0
    return ndimage.binary_dilation(m, iterations=dil)


def masked_texture(tex, mask):
    """nur die vom Subjekt benutzten Texel behalten (andere Passanten, Plakate, Boden raus), Rest auffüllen."""
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    out = tex[iy, ix]
    far = ndimage.distance_transform_edt(~mask) > 24             # weit weg: neutral glätten
    out[far] = ndimage.gaussian_filter(out, (12, 12, 0))[far]
    return out


# ------------------------------------------------------------------ Splats
def quat_from_mats(R):
    """Rotationsmatrizen (n,3,3) -> Quaternionen (w, x, y, z)"""
    m = R; t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(R), 4))
    c0 = t > 0
    s = np.sqrt(np.maximum(t[c0] + 1, 1e-12)) * 2
    q[c0] = np.column_stack([0.25 * s, (m[c0, 2, 1] - m[c0, 1, 2]) / s, (m[c0, 0, 2] - m[c0, 2, 0]) / s, (m[c0, 1, 0] - m[c0, 0, 1]) / s])
    for i in np.where(~c0)[0]:
        a = m[i]
        if a[0, 0] > a[1, 1] and a[0, 0] > a[2, 2]:
            s = np.sqrt(1 + a[0, 0] - a[1, 1] - a[2, 2]) * 2; q[i] = [(a[2, 1] - a[1, 2]) / s, 0.25 * s, (a[0, 1] + a[1, 0]) / s, (a[0, 2] + a[2, 0]) / s]
        elif a[1, 1] > a[2, 2]:
            s = np.sqrt(1 + a[1, 1] - a[0, 0] - a[2, 2]) * 2; q[i] = [(a[0, 2] - a[2, 0]) / s, (a[0, 1] + a[1, 0]) / s, 0.25 * s, (a[1, 2] + a[2, 1]) / s]
        else:
            s = np.sqrt(1 + a[2, 2] - a[0, 0] - a[1, 1]) * 2; q[i] = [(a[1, 0] - a[0, 1]) / s, (a[0, 2] + a[2, 0]) / s, (a[1, 2] + a[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def surface_splats(V, F, UVc, tex, spacing, seed=0, colors=None):
    """flache Gauß-Splats gleichmäßig auf der Oberfläche (Farbe aus der Textur, Lage tangential)."""
    rng = np.random.default_rng(seed)
    P = V[F]; cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); area = np.linalg.norm(cr, axis=1) / 2
    ok = area > 0
    nper = np.floor(area / spacing ** 2 + rng.random(len(F))).astype(int); nper[~ok] = 0
    fi = np.repeat(np.arange(len(F)), nper)
    r1 = rng.random(len(fi)); r2 = rng.random(len(fi)); s1 = np.sqrt(r1)
    w = np.column_stack([1 - s1, s1 * (1 - r2), s1 * r2])
    pos = np.einsum('ij,ijk->ik', w, P[fi])
    rgb = sample(tex, np.einsum('ij,ijk->ik', w, UVc[fi])) if colors is None else colors[fi]
    n = cr[fi] / np.linalg.norm(cr[fi], axis=1, keepdims=True)
    t1 = P[fi, 1] - P[fi, 0]; t1 -= n * np.einsum('ij,ij->i', t1, n)[:, None]
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-20)
    ang = rng.random(len(fi)) * np.pi                          # zufällige Drehung in der Tangentialebene
    t2 = np.cross(n, t1); t1, t2 = t1 * np.cos(ang)[:, None] + t2 * np.sin(ang)[:, None], -t1 * np.sin(ang)[:, None] + t2 * np.cos(ang)[:, None]
    R = np.stack([t1, t2, n], 2)
    return pos, rgb, R


def write_splats(path, pos, rgb, R, spacing, opacity=0.97):
    pos = pos @ FLIP.T; R = FLIP @ R                           # Viewer-Konvention (y nach unten)
    q = quat_from_mats(R)
    n = len(pos)
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    A = np.zeros((n, len(names)), np.float32)
    A[:, 0:3] = pos; A[:, 3:6] = R[:, :, 2]
    A[:, 6:9] = (np.clip(rgb, 0, 1) - 0.5) / SH0
    A[:, 9] = np.log(opacity / (1 - opacity))
    A[:, 10] = A[:, 11] = np.log(spacing * 0.62); A[:, 12] = np.log(spacing * 0.06)
    A[:, 13:17] = q
    hdr = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n' % (n, ''.join('property float %s\n' % s for s in names))
    with open(path, 'wb') as f:
        f.write(hdr.encode()); f.write(A.tobytes())


# ------------------------------------------------------------------ Mesh-Ausgaben
def write_ply_mesh(path, V, F, C):
    V = V @ FLIP.T; N = vertex_normals(V, F)
    vt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    va = np.empty(len(V), vt)
    for i, k in enumerate('xyz'): va[k] = V[:, i]; va['n' + k] = N[:, i]
    c8 = np.clip(np.round(C * 255), 0, 255).astype(np.uint8)
    va['red'], va['green'], va['blue'] = c8[:, 0], c8[:, 1], c8[:, 2]
    ft = np.dtype([('n', 'u1'), ('i', '<u4', (3,))])
    fa = np.empty(len(F), ft); fa['n'] = 3; fa['i'] = F
    hdr = ('ply\nformat binary_little_endian 1.0\ncomment Scan-Viewer import_mesh.py\nelement vertex %d\n'
           'property double x\nproperty double y\nproperty double z\nproperty double nx\nproperty double ny\nproperty double nz\n'
           'property uchar red\nproperty uchar green\nproperty uchar blue\nelement face %d\nproperty list uchar uint vertex_indices\nend_header\n') % (len(V), len(F))
    with open(path, 'wb') as f:
        f.write(hdr.encode()); f.write(va.tobytes()); f.write(fa.tobytes())


def write_glb(path, V, F, UVc, tex_rgb, name, quality=92):
    """y-oben, Meter; Ecken mit gleicher (Vertex, UV) teilen sich einen glTF-Vertex; Normalen vom geschlossenen Mesh."""
    N = vertex_normals(V, F)
    uvq = np.round(UVc.reshape(-1, 2) * 2 ** 22).astype(np.int64)
    key = np.column_stack([F.ravel(), uvq])
    uk, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    vid = uk[:, 0]; uv = uk[:, 1:] / 2 ** 22
    pos = V[vid].astype(np.float32); nrm = N[vid].astype(np.float32)
    tc = np.column_stack([uv[:, 0], 1 - uv[:, 1]]).astype(np.float32)   # glTF: v nach unten
    idx = inv.astype(np.uint32)
    import io
    b = io.BytesIO(); Image.fromarray((np.clip(tex_rgb, 0, 1) * 255).astype(np.uint8)).save(b, 'JPEG', quality=quality, subsampling=0)
    img = b.getvalue()
    chunks, views, off = [], [], 0
    def add(data, target=None):
        nonlocal off
        pad = (-len(data)) % 4; chunks.append(data + b'\0' * pad)
        v = dict(buffer=0, byteOffset=off, byteLength=len(data))
        if target: v['target'] = target
        views.append(v); off += len(data) + pad
        return len(views) - 1
    vp = add(pos.tobytes(), 34962); vn = add(nrm.tobytes(), 34962); vt = add(tc.tobytes(), 34962); vi = add(idx.tobytes(), 34963); vimg = add(img)
    gl = dict(asset=dict(version='2.0', generator='Scan-Viewer import_mesh.py'), scene=0, scenes=[dict(nodes=[0])],
              nodes=[dict(mesh=0, name=name)],
              meshes=[dict(name=name, primitives=[dict(attributes=dict(POSITION=0, NORMAL=1, TEXCOORD_0=2), indices=3, material=0)])],
              materials=[dict(name=name, pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0.0, roughnessFactor=0.9))],
              textures=[dict(source=0, sampler=0)], samplers=[dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)],
              images=[dict(bufferView=vimg, mimeType='image/jpeg')],
              accessors=[dict(bufferView=vp, componentType=5126, count=len(pos), type='VEC3', min=pos.min(0).tolist(), max=pos.max(0).tolist()),
                         dict(bufferView=vn, componentType=5126, count=len(nrm), type='VEC3'),
                         dict(bufferView=vt, componentType=5126, count=len(tc), type='VEC2'),
                         dict(bufferView=vi, componentType=5125, count=len(idx), type='SCALAR')],
              bufferViews=views, buffers=[dict(byteLength=off)])
    js = json.dumps(gl, separators=(',', ':')).encode(); js += b' ' * ((-len(js)) % 4)
    binc = b''.join(chunks)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binc)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack('<II', len(binc), 0x004E4942)); f.write(binc)
    return len(pos)


def vertex_colors(V, F, UVc, tex):
    C = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
    s = sample(tex, UVc.reshape(-1, 2))
    np.add.at(C, F.ravel(), s); np.add.at(cnt, F.ravel(), 1)
    return C / np.maximum(cnt, 1)[:, None]


# ------------------------------------------------------------------ Hauptablauf
def run(obj, name, height=1.80, texture=None, uid=None, log=print):
    t0 = time.time()
    V, VT, FV, FT, mtl = load_obj(obj)
    tex_path = find_texture(obj, mtl, texture)
    tex = np.asarray(Image.open(tex_path).convert('RGB')).astype(np.float32) / 255
    UV0 = VT[FT]                                                  # Eck-UVs (n, 3, 2)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    log(f'geladen: {len(V)} Vertices, {len(FV)} Dreiecke, Textur {tex.shape[1]}x{tex.shape[0]}, Diagonale {diag:.3f}')

    # 1. Boden
    thr = 0.0012 * diag
    n, d, ninl = ground_plane(V, thr)
    h = V @ n + d
    hf = h[FV].mean(1)
    fcol = sample(tex, UV0.mean(1))
    gcol = np.median(fcol[np.abs(hf) < thr], 0)
    log(f'Boden: Normale {np.round(n, 4)}, Neigung {np.degrees(np.arccos(min(1, n[1]))):.2f}°, {ninl} Inlier, Bodenfarbe {np.round(gcol, 3)}')

    # 2. Subjekt
    cut0 = thr
    above = np.where(h[FV].min(1) > cut0)[0]
    ncomp, lab = face_components(FV[above])
    sizes = np.bincount(lab)
    cen = V[FV[above]].mean(1); horiz = cen - np.outer(cen @ n, n)
    cand = np.argsort(sizes)[::-1][:8]
    hmax = np.array([hf[above[lab == k]].max() for k in cand])
    dist = np.array([np.linalg.norm(horiz[lab == k].mean(0)) for k in cand])
    score = sizes[cand] / (1 + 8 * dist / diag) * (hmax > 0.25 * hmax.max())
    best = cand[int(np.argmax(score))]
    sf = above[lab == best]
    log(f'Komponenten über dem Boden: {ncomp}; Subjekt = {len(sf)} Dreiecke (Abstand zur Mitte {dist[cand == best][0]:.3f}), '
        f'verworfen: {len(FV) - len(sf)} Dreiecke (Boden, Wand/Plakate, {ncomp - 1} Fragmente)')
    SV, SF, used = compact(V, FV[sf]); SU = UV0[sf]; sh = h[used]
    H = float(sh.max())
    be, _, _ = boundary_half_edges(SF)
    hb = sh[np.unique(be)]
    log(f'Subjekthöhe {H:.4f} Einheiten; offene Ränder: {len(be)} Kanten, Höhe {hb.min():.4f}..{hb.max():.4f}')
    if hb.max() > 0.05 * H: raise RuntimeError('offene Ränder weit über dem Boden – dieser Scan braucht eine eigene Lochbehandlung')
    # Schnitthöhe: über allen Rändern und über bodenfarbenen Resten am Fuß
    fc = sample(tex, SU.mean(1)); fh = sh[SF].mean(1)
    groundlike = np.linalg.norm(fc - gcol, axis=1) < 0.12
    c = hb.max() + 0.0005 * diag
    step = 0.001 * diag
    while c < 0.03 * H:
        band = (fh >= c) & (fh < c + 2 * step)
        if band.sum() < 20 or groundlike[band].mean() < 0.03: break
        c += step
    c += 1e-6 * diag * (np.abs(sh - c).min() < 1e-6 * diag)
    log(f'Schnitt bei {c:.4f} Einheiten über dem Boden (bodenfarbene Flächen im Fußbereich: {int(groundlike[fh < 0.06 * H].sum())})')

    # 3. Ausrichten + skalieren (alles in Meter, y oben, Boden y = 0)
    s = height / H
    R1 = rot_a_to_b(n, np.array([0, 1.0, 0]))
    def to_up(P): Q = P @ R1.T; Q[:, 1] = P @ n + d; return Q        # y exakt = Höhe über der Ebene
    Q = to_up(SV)
    # Blickrichtung aus den Füßen: Hauptachse jedes Schuhs; die Ferse liegt unter dem Knöchel,
    # die Spitze ragt nach vorn -> Vorzeichen = vom Knöchel zum weiter entfernten Schuhende
    from scipy.cluster.vq import kmeans2
    foot3 = Q[(Q[:, 1] > c) & (Q[:, 1] < 0.035 * H)]
    _, fl = kmeans2(foot3[:, [0, 2]], 2, seed=3, minit='++')
    shin = Q[(Q[:, 1] > 0.07 * H) & (Q[:, 1] < 0.11 * H)]
    axes = []
    for k in range(2):
        A = foot3[fl == k][:, [0, 2]]; mu = A.mean(0); ax = np.linalg.svd(A - mu, full_matrices=False)[2][0]
        ank = shin[np.linalg.norm(shin[:, [0, 2]] - mu, axis=1) < 0.12 * H][:, [0, 2]].mean(0)
        t = (A - ank) @ ax                                          # Schuhenden relativ zum Knöchel
        if abs(np.percentile(t, 98)) < abs(np.percentile(t, 2)): ax = -ax
        axes.append(ax)
    fwd = axes[0] / np.linalg.norm(axes[0]) + axes[1] / np.linalg.norm(axes[1]); fwd /= np.linalg.norm(fwd)
    yaw = np.arctan2(fwd[0], fwd[1])                                      # Drehung, die fwd auf +z bringt
    Ry = np.array([[np.cos(-yaw), 0, np.sin(-yaw)], [0, 1, 0], [-np.sin(-yaw), 0, np.cos(-yaw)]])
    P = SV  # noqa
    Pf = V[FV[sf]]; ar = np.linalg.norm(np.cross(Pf[:, 1] - Pf[:, 0], Pf[:, 2] - Pf[:, 0]), axis=1)
    cxz = (to_up(Pf.mean(1)) @ Ry.T)[:, [0, 2]].T @ ar / ar.sum()
    def xf(P):
        Q = to_up(P) @ Ry.T; Q[:, 0] -= cxz[0]; Q[:, 2] -= cxz[1]; return Q * s
    log(f'Ausrichtung: Neigung korrigiert, Blickrichtung {np.round(fwd, 3)} -> +z, Maßstab {s:.4f} (Höhe {height:.2f} m)')
    SVm = xf(SV); cm = c * s

    # 4. Schnitt, Wände, Sohlen
    V1, F1, U1 = slice_above(SVm, SF, SU, SVm[:, 1], cm)
    # dunkle, gleichmäßige Sohlenfarbe aus dem untersten Schuhbereich
    fh1 = V1[F1][:, :, 1].mean(1); lowf = np.where(fh1 < cm + 0.02)[0]
    uvl = U1[lowf].mean(1); lum = sample(tex, uvl).mean(1)
    cands = lowf[np.argsort(lum)[:max(20, len(lowf) // 5)]]
    best_uv, best_s = None, 1e9
    Ht, Wt = tex.shape[:2]; r = max(4, Wt // 400)
    target = np.percentile(lum, 15)                                # Sohle: dunkler als der Schuh-Schnitt
    for f in cands[:300]:
        u = U1[f].mean(0); x = int(u[0] * Wt); y = int((1 - u[1]) * Ht)
        win = tex[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        sc = win.std() + abs(win.mean() - target)
        if sc < best_s: best_s, best_uv = sc, u
    edge = np.median(np.linalg.norm(V1[F1[:, 0]] - V1[F1[:, 1]], axis=1))
    V2, F2, U2, cut_y = walls_and_soles(V1, F1, U1, rows=max(2, int(round(cm / edge))), uv_sole=best_uv, max_area=0.5 * edge ** 2)
    t2 = topo_check(V2, F2)
    log(f'nach Schnitt + Sohlen: {len(F2)} Dreiecke, {t2}')
    if not t2['watertight']: raise RuntimeError('Mesh ist nach dem Schließen nicht wasserdicht')
    V2, F2, U2, bridge = carve_leg_bridge(V2, F2, U2, height, tex, gcol, log)
    if bridge:
        t2 = topo_check(V2, F2); log(f'nach Beinspalt: {len(F2)} Dreiecke, {t2}')
        if not t2['watertight']: raise RuntimeError('Mesh ist nach dem Beinspalt nicht wasserdicht')

    # 5. Spitzen glätten, hochauflösend unterteilen
    V2, nsp = despike(V2, F2)
    V3, F3, U3 = loop_subdivide(V2, F2, U2)
    V3[:, 1] -= V3[:, 1].min()                                    # Sohle exakt auf y = 0
    t3 = topo_check(V3, F3)
    log(f'Ausreißer-Spitzen geglättet: {nsp}; nach Loop-Unterteilung: {len(V3)} Vertices, {len(F3)} Dreiecke, {t3}')
    if not t3['watertight'] or t3['components'] != 1 or t3['volume'] <= 0: raise RuntimeError('Endkontrolle fehlgeschlagen')

    # 6. Ausgaben
    uid = uid or uuid.uuid4().hex[:6]
    base = f'{uid}_{name}'
    os.makedirs(UP, exist_ok=True); os.makedirs(EX, exist_ok=True)
    pre = os.path.join(EX, base)
    C3 = vertex_colors(V3, F3, U3, tex)
    write_ply_mesh(pre + '_mesh.ply', V3, F3, C3)
    mask = texture_mask(U2, tex.shape[:2])
    texm = masked_texture(tex, mask)
    nglb = write_glb(pre + '_mesh.glb', V3, F3, U3, texm, name)
    # Splats: Subjekt (aus dem fertigen Mesh) und ganzer Scan (gleiche Transformation)
    sp_subj = 0.0022
    p, rgb, R = surface_splats(V3, F3, U3, tex, sp_subj, seed=1)
    write_splats(pre + '_extrahiert.ply', p, rgb, R, sp_subj)
    sp_all = 0.0042
    Vall = xf(V)
    pa, rgba, Ra = surface_splats(Vall, FV, UV0, tex, sp_all, seed=2)
    write_splats(os.path.join(UP, base + '.ply'), pa, rgba, Ra, sp_all)
    src_dir = os.path.join(UP, base + '_quelle'); os.makedirs(src_dir, exist_ok=True)
    for fpath in [obj, tex_path] + ([os.path.join(os.path.dirname(obj), mtl)] if mtl else []):
        if os.path.isfile(fpath): shutil.copy2(fpath, src_dir)
    pf = p @ FLIP.T
    info = dict(total=int(len(pa)), selected=int(len(p)), clusters=int(ncomp), taken=1,
                center=(np.array([0, height * 0.5, 0]) @ FLIP.T).tolist(), up=[0.0, -1.0, 0.0], plane_d=0.0,
                ground_inliers=ninl, eps=sp_subj, scale=1.0, height=float(V3[:, 1].max()),
                params=dict(radius=None, group=False, min_opacity=None, ground_margin=None, keep_soft=None),
                bbox_min=pf.min(0).tolist(), bbox_max=pf.max(0).tolist(), extracted_raw=int(len(p)),
                mesh=dict(voxel=None, faces=int(len(F3)), vertices=int(len(V3)), boundary_edges=t3['boundary_edges'],
                          components=t3['components'], watertight=t3['watertight'], removed_splats=int(len(pa)),
                          kept_splats=int(len(p)), giant_or_needle=0, nonmanifold_edges=t3['nonmanifold_edges'],
                          genus=t3['genus'], glb_vertices=int(nglb), despiked=nsp),
                cleaned=True,
                source=dict(type='mesh', app='KIRI Engine', file=os.path.basename(obj), faces_total=int(len(FV)),
                            faces_subject=int(len(sf)), components_above_ground=int(ncomp), scale_m_per_unit=s,
                            cut_height_m=float(cm), ground_tilt_deg=float(np.degrees(np.arccos(min(1, n[1])))),
                            texture=os.path.basename(tex_path), glb=True, units='m', leg_bridge=bridge),
                seconds=round(time.time() - t0, 2))
    json.dump(info, open(pre + '_extrahiert.json', 'w'), indent=1)
    log(f'fertig in {info["seconds"]} s -> {base}')
    return info, base


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('obj'); ap.add_argument('--name', default=None); ap.add_argument('--height', type=float, default=1.80)
    ap.add_argument('--texture'); ap.add_argument('--id')
    ap.add_argument('--dest', help='anderer Zielordner mit uploads/ und extracted/ (Test)')
    a = ap.parse_args()
    if a.dest: UP, EX = os.path.join(a.dest, 'uploads'), os.path.join(a.dest, 'extracted')
    name = re.sub(r'[^\w\-. ()äöüÄÖÜß]', '_', a.name or os.path.splitext(os.path.basename(a.obj))[0])
    info, base = run(a.obj, name, a.height, a.texture, a.id)
    print(json.dumps(dict(base=base, faces=info['mesh']['faces'], watertight=info['mesh']['watertight'],
                          boundary=info['mesh']['boundary_edges'], comps=info['mesh']['components'], splats=info['selected'], scene_splats=info['total'])))

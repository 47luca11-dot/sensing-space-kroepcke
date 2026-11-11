"""Geometrie-Bausteine (numpy). Ein 'Part' ist dict(mat, pos(n,3), nrm(n,3), uv(n,2), idx(m,3)). Y oben."""
import math
import numpy as np

def part(mat, pos, idx, nrm=None, uv=None):
    pos = np.asarray(pos, np.float64); idx = np.asarray(idx, np.int64).reshape(-1, 3)
    if nrm is None: nrm = vertex_normals(pos, idx)
    if uv is None: uv = np.zeros((len(pos), 2))
    return {'mat': mat, 'pos': pos, 'nrm': np.asarray(nrm, np.float64), 'uv': np.asarray(uv, np.float64), 'idx': idx}

def vertex_normals(pos, idx):
    n = np.zeros_like(pos)
    if len(idx) == 0: return n + [0, 1, 0]
    fn = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    for k in range(3): np.add.at(n, idx[:, k], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True); l[l == 0] = 1
    return n / l

def merge(parts):
    """gleiche Materialien zusammenfassen"""
    by = {}
    for p in parts:
        if p is None or len(p['idx']) == 0: continue
        by.setdefault(p['mat'], []).append(p)
    out = []
    for m, ps in by.items():
        off = 0; P = []; N = []; T = []; I = []
        for p in ps:
            P.append(p['pos']); N.append(p['nrm']); T.append(p['uv']); I.append(p['idx'] + off); off += len(p['pos'])
        out.append({'mat': m, 'pos': np.concatenate(P), 'nrm': np.concatenate(N), 'uv': np.concatenate(T), 'idx': np.concatenate(I)})
    return out

def transform(p, R=None, t=(0, 0, 0), s=1.0):
    q = dict(p)
    pos = p['pos'] * s; nrm = p['nrm']
    if R is not None: pos = pos @ R.T; nrm = nrm @ R.T
    q['pos'] = pos + np.asarray(t); q['nrm'] = nrm
    return q

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    # dreht +x nach (c, 0, -s)  (rechtsdrehend um +Y)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def flat_quads(mat, quads, uvscale=1.0):
    """quads: (k,4,3) Eckpunkte gegen den Uhrzeigersinn von der Sichtseite. Flache Normalen, UV nach Kantenlaenge."""
    q = np.asarray(quads, float)
    if len(q) == 0: return None
    k = len(q)
    pos = q.reshape(-1, 3)
    n = np.cross(q[:, 1] - q[:, 0], q[:, 3] - q[:, 0])
    l = np.linalg.norm(n, axis=1, keepdims=True); l[l == 0] = 1; n = n / l
    nrm = np.repeat(n, 4, 0)
    e = np.linalg.norm(q[:, 1] - q[:, 0], axis=1); h = np.linalg.norm(q[:, 3] - q[:, 0], axis=1)
    # u entlang Welt-Laenge (fortlaufend aus Lage), v Hoehe
    u0 = (q[:, 0, 0] + q[:, 0, 2]) / uvscale
    uv = np.zeros((k, 4, 2))
    uv[:, 0] = np.c_[u0, q[:, 0, 1] / uvscale]; uv[:, 1] = np.c_[u0 + e / uvscale, q[:, 1, 1] / uvscale]
    uv[:, 2] = np.c_[u0 + e / uvscale, q[:, 2, 1] / uvscale]; uv[:, 3] = np.c_[u0, q[:, 3, 1] / uvscale]
    uv[..., 1] *= -1
    # (fast) waagerechte Flaechen: Welt-x/z als UV (sonst ist v konstant -> gestreckte Streifen auf Deckplatten/Sitzflaechen)
    hz = np.abs(n[:, 1]) > 0.7
    if hz.any():
        uv[hz] = np.stack([q[hz][..., 0], -q[hz][..., 2]], -1) / uvscale
    b = np.arange(k)[:, None] * 4
    idx = np.concatenate([b + [0, 1, 2], b + [0, 2, 3]])
    return part(mat, pos, idx, nrm, uv.reshape(-1, 2))

def box(mat, c, size, a=0.0, uvscale=1.0):
    """Quader, Mittelpunkt c (x, y_unten, z), size (sx, sy, sz), Drehung a um Y; Boden bei y=c[1]."""
    sx, sy, sz = size
    x = np.array([-sx, sx]) / 2; z = np.array([-sz, sz]) / 2
    v = np.array([[x[i], y, z[j]] for y in (0, sy) for j in (0, 1) for i in (0, 1)])  # 0..3 unten, 4..7 oben
    R = rot_y(a); v = v @ R.T + np.asarray(c, float)
    F = [(0, 1, 5, 4), (1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6), (4, 5, 7, 6), (2, 3, 1, 0)]
    quads = []
    for f in F:
        quads.append([v[f[0]], v[f[1]], v[f[2]], v[f[3]]])
    q = np.array(quads)
    # Orientierung pruefen (Normale nach aussen)
    cc = v.mean(0)
    for i in range(6):
        n = np.cross(q[i, 1] - q[i, 0], q[i, 3] - q[i, 0])
        if np.dot(n, q[i].mean(0) - cc) < 0: q[i] = q[i][::-1]
    return flat_quads(mat, q, uvscale)

def cylinder(mat, c, r0, r1, h, seg=8, cap=True, uvscale=1.0):
    """Kegelstumpf/Zylinder entlang +Y ab c (x,y,z)."""
    a = np.linspace(0, 2 * np.pi, seg + 1)
    ca, sa = np.cos(a), np.sin(a)
    bot = np.c_[r0 * ca, np.zeros(seg + 1), r0 * sa]; top = np.c_[r1 * ca, np.full(seg + 1, h), r1 * sa]
    pos = np.concatenate([bot, top]) + c
    slope = (r0 - r1) / max(h, 1e-6)
    nrm = np.c_[ca, np.full(seg + 1, slope), sa]; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.concatenate([nrm, nrm])
    circ = 2 * np.pi * max(r0, r1)
    uv = np.concatenate([np.c_[a / (2 * np.pi) * circ / uvscale, np.zeros(seg + 1)], np.c_[a / (2 * np.pi) * circ / uvscale, -np.full(seg + 1, h / uvscale)]])
    idx = []
    n = seg + 1
    for i in range(seg):
        idx += [(i, n + i + 1, i + 1), (i, n + i, n + i + 1)]
    parts = [part(mat, pos, idx, nrm, uv)]
    if cap and r1 > 0:
        tc = np.concatenate([[c + np.array([0, h, 0])], top[:-1] + c])
        ti = [(0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)]
        parts.append(part(mat, tc, ti, np.tile([0, 1, 0], (seg + 1, 1)), tc[:, [0, 2]] / uvscale))
    return merge(parts)[0]

def tube(mat, pts, r, seg=6, uvscale=1.0, caps=True):
    """Rohr entlang Polyline pts (k,3)."""
    pts = np.asarray(pts, float); k = len(pts)
    if k < 2: return None
    T = np.gradient(pts, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    up = np.array([0, 1.0, 0])
    N = np.cross(T, up)
    bad = np.linalg.norm(N, axis=1) < 1e-3
    N[bad] = np.cross(T[bad], [1.0, 0, 0])
    N /= np.linalg.norm(N, axis=1, keepdims=True); Bn = np.cross(T, N)
    a = np.linspace(0, 2 * np.pi, seg + 1)
    ring = np.cos(a)[None, :, None] * N[:, None, :] + np.sin(a)[None, :, None] * Bn[:, None, :]
    pos = (pts[:, None, :] + r * ring).reshape(-1, 3); nrm = ring.reshape(-1, 3)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    uv = np.stack(np.broadcast_arrays(L[:, None] / uvscale, a[None, :] / (2 * np.pi)), -1).reshape(-1, 2)
    n = seg + 1; idx = []
    for i in range(k - 1):
        for j in range(seg):
            a0 = i * n + j; b0 = (i + 1) * n + j
            idx += [(a0, b0 + 1, b0), (a0, a0 + 1, b0 + 1)]      # Normale nach aussen (= ring; QA 2026-11-10)
    parts = [part(mat, pos, idx, nrm, uv)]
    if caps:
        for end, sgn in ((0, -1), (k - 1, 1)):
            cpos = np.concatenate([[pts[end]], pts[end] + r * ring[end][:-1]])
            ci = [(0, 1 + i, 1 + (i + 1) % seg) if sgn > 0 else (0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)]
            parts.append(part(mat, cpos, ci, np.tile(T[end] * sgn, (seg + 1, 1)), np.zeros((seg + 1, 2))))
    return merge(parts)[0]

def extrude(mat, ring_xz, y0, y1, uvscale=1.0, top=True, bottom=False, top_mat=None):
    """Prisma aus einfachem Polygon (xz, gegen Uhrzeigersinn in x/z-Ebene egal)."""
    import mapbox_earcut as earcut
    P = np.asarray(ring_xz, float)
    y0 = np.atleast_1d(np.asarray(y0, float)); y1 = np.atleast_1d(np.asarray(y1, float))
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
        if len(y0) == len(P) + 1: y0 = y0[:-1]
        if len(y1) == len(P) + 1: y1 = y1[:-1]
    # Orientierung: Flaeche in (x, z) positiv -> von oben gesehen (y nach oben, z nach Sueden) im Uhrzeigersinn
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    if area < 0:
        P = P[::-1]
        if len(y0) > 1: y0 = y0[::-1]
        if len(y1) > 1: y1 = y1[::-1]
    n = len(P); parts = []
    y0 = np.broadcast_to(np.asarray(y0, float), (n,)); y1 = np.broadcast_to(np.asarray(y1, float), (n,))
    quads = []
    for i in range(n):
        j = (i + 1) % n
        a = [P[i, 0], y0[i], P[i, 1]]; b = [P[j, 0], y0[j], P[j, 1]]; c = [P[j, 0], y1[j], P[j, 1]]; d = [P[i, 0], y1[i], P[i, 1]]
        quads.append([a, d, c, b])
    w = flat_quads(mat, quads, uvscale)
    # Seitenorientierung pruefen
    cen = np.r_[P.mean(0)]
    nn = w['nrm'][0]; q0 = np.array(quads[0]).mean(0)
    if np.dot(nn[[0, 2]], q0[[0, 2]] - cen) < 0:
        w['idx'] = w['idx'][:, ::-1]; w['nrm'] = -w['nrm']
    parts.append(w)
    tri = earcut.triangulate_float64(P, np.array([n], np.uint32)).reshape(-1, 3)
    if top:
        pos = np.c_[P[:, 0], y1, P[:, 1]]
        t = part(top_mat or mat, pos, tri, np.tile([0, 1, 0], (n, 1)), P / uvscale)
        fn = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        if len(fn) and fn[:, 1].sum() < 0: t['idx'] = t['idx'][:, ::-1]
        parts.append(t)
    if bottom:
        pos = np.c_[P[:, 0], y0, P[:, 1]]
        t = part(mat, pos, tri, np.tile([0, -1, 0], (n, 1)), P / uvscale)
        fn = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        if len(fn) and fn[:, 1].sum() > 0: t['idx'] = t['idx'][:, ::-1]
        parts.append(t)
    return merge(parts)

def orient_up(p):
    """Dreiecke so drehen, dass die Normalen nach oben zeigen (Flaechen)"""
    pos, idx = p['pos'], p['idx']
    fn = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    flip = fn[:, 1] < 0
    idx[flip] = idx[flip][:, ::-1]
    return p

def tri_count(parts):
    return int(sum(len(p['idx']) for p in parts if p is not None))

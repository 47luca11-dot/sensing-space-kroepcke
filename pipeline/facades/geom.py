"""Geometrie-Bausteine: Mesh-Sammler je (Ebene, Material), Polygon-Triangulierung, Boxen, Profil-Sweeps mit Gehrung,
lokale Fassaden-Rahmen (u = rechts, v = oben, d = nach aussen)."""
import numpy as np
import mapbox_earcut as earcut
from common import hex2lin

UP = np.array([0.0, 1.0, 0.0])


def col(c):
    """Farbe -> lineares RGB float (3,)."""
    if isinstance(c, str):
        return hex2lin(c)
    return np.asarray(c, float)


def tri2d(outer, holes=()):
    """Polygon (mit Loechern) in 2D triangulieren. -> (V (N,2), F (M,3)) CCW orientiert."""
    rings = [np.asarray(outer, float)] + [np.asarray(h, float) for h in holes if len(h) >= 3]
    V = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    F = earcut.triangulate_float64(V, ends).reshape(-1, 3).astype(np.int64)
    if len(F):
        s = _area2(V, F)
        if (np.abs(s) <= 1e-12).any():
            F = _split_degenerate(V, F, s)
            s = _area2(V, F)
        F = F[np.abs(s) > 1e-12]
        if len(F) and len(np.unique(F)) < len(V):
            F = _insert_unused(V, F)
        s = _area2(V, F)
        F = F[np.abs(s) > 1e-12]
        s = s[np.abs(s) > 1e-12]
        flip = s < 0
        F[flip] = F[flip][:, [0, 2, 1]]
    return V, F


def _insert_unused(V, F):
    """earcut verwirft kollineare Randpunkte (filterPoints, v. a. bei Loechern). Solche Punkte wieder einfuegen: das Dreieck,
    auf dessen Kante der Punkt liegt, ueber ihn teilen (3D-Lage des Punktes bleibt so erhalten -> kein Spalt)."""
    used = np.zeros(len(V), bool); used[F.ravel()] = True
    miss = np.nonzero(~used)[0]
    F = [list(f) for f in F]
    for m in miss:
        pm = V[m]
        if any(np.allclose(V[x], pm, atol=1e-9) for x in (m - 1, m + 1) if 0 <= x < len(V) and used[x]):
            continue                                    # Doppelpunkt
        A = np.array(F)
        best = None
        for e in ((0, 1), (1, 2), (2, 0)):
            P0 = V[A[:, e[0]]]; P1 = V[A[:, e[1]]]
            d = P1 - P0; L2 = (d ** 2).sum(1)
            t = ((pm - P0) * d).sum(1) / np.maximum(L2, 1e-30)
            q = P0 + t[:, None] * d
            dist = np.linalg.norm(q - pm, axis=1)
            ok = (t > 1e-9) & (t < 1 - 1e-9) & (dist < 1e-7 * np.maximum(1.0, np.sqrt(L2)))
            for i in np.nonzero(ok)[0]:
                if best is None or dist[i] < best[0]:
                    best = (dist[i], i, e)
        if best is None:
            continue
        _, i, e = best
        f = F[i]; p, q = f[e[0]], f[e[1]]; r = [x for x in f if x != p and x != q][0]
        F[i] = [p, m, r]; F.append([m, q, r]); used[m] = True
    return np.array(F, np.int64).reshape(-1, 3)


def _area2(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _split_degenerate(V, F, s):
    """earcut erzeugt bei kollinearen Randpunkten Null-Dreiecke; einfaches Weglassen wuerde den mittleren Punkt aus dem
    Rand entfernen (in 3D liegt er oft einige mm daneben -> Spalt zum Nachbarbauteil). Stattdessen das Nachbardreieck
    an der langen Kante ueber den mittleren Punkt teilen."""
    deg = np.abs(s) <= 1e-12
    good = [list(f) for f in F[~deg]]
    pend = [list(f) for f in F[deg]]
    for _ in range(4):
        rest = []
        for f in pend:
            a, b, c = f
            p, q, m = max(((a, b, c), (b, c, a), (c, a, b)), key=lambda t: float(((V[t[0]] - V[t[1]]) ** 2).sum()))
            if float(((V[p] - V[q]) ** 2).sum()) < 1e-20 or m in (p, q):
                continue
            hit = None
            for gi in range(len(good) - 1, -1, -1):
                g = good[gi]
                if p in g and q in g:
                    hit = gi; break
            if hit is None:
                rest.append(f); continue
            g = good.pop(hit)
            r = [x for x in g if x != p and x != q][0]
            good.append([p, m, r]); good.append([m, q, r])
        pend = rest
        if not pend:
            break
    return np.array(good, np.int64).reshape(-1, 3)


class Frame:
    """Lokaler Rahmen: Ursprung O, Achsen t (u, rechts), b (v, oben), n (d, aussen)."""
    __slots__ = ('O', 't', 'b', 'n', 'M')

    def __init__(self, O, t, b, n):
        self.O = np.asarray(O, float); self.t = np.asarray(t, float); self.b = np.asarray(b, float); self.n = np.asarray(n, float)
        self.M = np.stack([self.t, self.b, self.n])

    def w(self, L):
        """lokale (u,v,d) -> Welt"""
        return self.O + np.asarray(L, float) @ self.M

    def loc(self, P):
        return (np.asarray(P, float) - self.O) @ self.M.T

    def shifted(self, du=0.0, dv=0.0, dd=0.0):
        return Frame(self.O + du * self.t + dv * self.b + dd * self.n, self.t, self.b, self.n)


def wall_frame(n, O):
    n = np.asarray(n, float).copy()
    h = np.array([n[0], 0.0, n[2]])
    if np.linalg.norm(h) < 1e-6:
        t = np.array([1.0, 0, 0])
    else:
        h /= np.linalg.norm(h)
        t = np.array([h[2], 0.0, -h[0]])               # t = up x n
    b = np.cross(n, t); b /= np.linalg.norm(b)
    return Frame(O, t, b, n)


class EdgeResolver:
    """Punkte auf dem Originalrand eines (nicht ganz ebenen) Polygons exakt auf die Original-3D-Kanten legen,
    damit benachbarte Waende/Daecher an gemeinsamen Kanten lueckenlos anschliessen."""

    def __init__(self, uv_rings, p3_rings):
        A = []; B = []; PA = []; PB = []
        for uv, P in zip(uv_rings, p3_rings):
            n = len(uv)
            for i in range(n):
                j = (i + 1) % n
                A.append(uv[i]); B.append(uv[j]); PA.append(P[i]); PB.append(P[j])
        self.A = np.array(A, float); self.B = np.array(B, float); self.PA = np.array(PA, float); self.PB = np.array(PB, float)

    def __call__(self, V2, W):
        V2 = np.asarray(V2, float)
        AB = self.B - self.A
        L2 = np.maximum((AB ** 2).sum(1), 1e-18)
        t = ((V2[:, None, :] - self.A[None]) * AB[None]).sum(2) / L2[None]
        tc = np.clip(t, 0, 1)
        C = self.A[None] + tc[..., None] * AB[None]
        d = np.linalg.norm(V2[:, None, :] - C, axis=2)
        k = np.argmin(d, 1)
        hit = d[np.arange(len(V2)), k] < 2e-5
        if hit.any():
            kk = k[hit]; tt = tc[hit, kk]
            W = W.copy()
            W[hit] = self.PA[kk] + tt[:, None] * (self.PB[kk] - self.PA[kk])
        return W


class MeshAcc:
    """Sammelt Dreiecke je (layer, material). Farben linear pro Vertex."""

    def __init__(self):
        self.parts = {}
        self.ntri = 0

    def add(self, layer, mat, V, F, c):
        V = np.asarray(V, float); F = np.asarray(F, np.int64)
        if len(F) == 0:
            return
        c = col(c)
        C = np.broadcast_to(c, (len(V), 3)) if c.ndim == 1 else c
        self.parts.setdefault((layer, mat), []).append((V, F, np.asarray(C, np.float32)))
        self.ntri += len(F)

    def merge_from(self, other):
        for k, lst in other.parts.items():
            self.parts.setdefault(k, []).extend(lst)
        self.ntri += other.ntri

    def poly(self, layer, mat, fr, outer_uv, c, holes_uv=(), d=0.0, flip=False, keep3d=None):
        """Polygon in lokalen (u,v) des Rahmens fr bei Tiefe d. keep3d: dict (u,v)->Welt fuer exakte Originalpunkte."""
        V2, F = tri2d(outer_uv, holes_uv)
        if len(F) == 0:
            return
        L = np.c_[V2, np.full(len(V2), d)]
        W = fr.w(L)
        if callable(keep3d):
            W = keep3d(V2, W)
        elif keep3d is not None:
            for i, (u, v) in enumerate(V2):
                k = keep3d.get((round(u, 4), round(v, 4)))
                if k is not None:
                    W[i] = k
        if flip:
            F = F[:, [0, 2, 1]]
        self.add(layer, mat, W, F, c)

    def quad(self, layer, mat, P, c):
        P = np.asarray(P, float)
        self.add(layer, mat, P, [[0, 1, 2], [0, 2, 3]], c)

    def box(self, layer, mat, fr, lo, hi, c, skip=()):
        """Quader in lokalen Koordinaten [lo,hi] (u,v,d). skip: Seiten weglassen aus {'-u','+u','-v','+v','-d','+d'}."""
        V, F = box_local(lo, hi, skip)
        if len(F):
            self.add(layer, mat, fr.w(V), F, c)


_BOX_FACES = {  # Ecken-Indizes (bit0=u, bit1=v, bit2=d), CCW von aussen
    '-u': (0, 4, 6, 2), '+u': (1, 3, 7, 5), '-v': (0, 1, 5, 4), '+v': (2, 6, 7, 3), '-d': (0, 2, 3, 1), '+d': (4, 5, 7, 6)}


def box_local(lo, hi, skip=()):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    V = np.array([[hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]] for i in range(8)])
    F = []
    for k, (a, b, c, d) in _BOX_FACES.items():
        if k in skip:
            continue
        F += [[a, b, c], [a, c, d]]
    return V, np.array(F, np.int64).reshape(-1, 3)


def prism_xz(acc, layer, mat, poly, y0, y1, c, top=True, bottom=False, sides=True, uvz=None):
    """Shapely-(Multi)Polygon in der x-z-Ebene als Prisma zwischen y0 und y1."""
    from shapely.geometry import Polygon, MultiPolygon
    geoms = [poly] if poly.geom_type == 'Polygon' else [g for g in getattr(poly, 'geoms', []) if g.geom_type == 'Polygon']
    for g in geoms:
        if g.is_empty or g.area < 1e-5:
            continue
        ext = np.array(g.exterior.coords)[:-1]
        ints = [np.array(r.coords)[:-1] for r in g.interiors]
        V2, F = tri2d(ext, ints)
        if len(F) == 0:
            continue
        # x-z: CCW in (x,z) hat Normale -y (weil y = z x x). Oben: Normale +y -> umdrehen
        if top:
            acc.add(layer, mat, np.c_[V2[:, 0], np.full(len(V2), y1), V2[:, 1]], F[:, [0, 2, 1]], c)
        if bottom:
            acc.add(layer, mat, np.c_[V2[:, 0], np.full(len(V2), y0), V2[:, 1]], F, c)
        if sides:
            for ring, is_ext in [(ext, True)] + [(r, False) for r in ints]:
                ring_sides(acc, layer, mat, ring, y0, y1, c, outward=is_ext)


def signed_area_xz(R):
    R = np.asarray(R, float)
    return 0.5 * np.sum(R[:, 0] * np.roll(R[:, 1], -1) - np.roll(R[:, 0], -1) * R[:, 1])


def ring_sides(acc, layer, mat, ring, y0, y1, c, outward=True):
    """Senkrechte Seitenflaechen entlang eines Rings (x,z); Normalen weg vom Polygoninneren (outward) bzw. hinein.
    Funktioniert fuer Aussen- wie Innenringe, da Orientierung ueber die Flaeche des Rings bestimmt wird.
    Fuer Innenringe (Loecher) zeigt 'outward' in das Loch hinein (= weg vom Material)."""
    R = np.asarray(ring, float)
    if len(R) < 2:
        return
    A = signed_area_xz(R)
    n = len(R)
    i = np.arange(n); j = (i + 1) % n
    P0 = R[i]; P1 = R[j]
    V = np.concatenate([np.c_[P0[:, 0], np.full(n, y0), P0[:, 1]], np.c_[P1[:, 0], np.full(n, y0), P1[:, 1]],
                        np.c_[P1[:, 0], np.full(n, y1), P1[:, 1]], np.c_[P0[:, 0], np.full(n, y1), P0[:, 1]]])
    a, b, cc, d = i, i + n, i + 2 * n, i + 3 * n
    # Kante P0->P1, Normale (dz, -dx) zeigt nach aussen fuer A>0. Quad (P0b,P1b,P1t,P0t): Normale = (P1-P0) x up
    # = (dx,0,dz) x (0,1,0) = (-dz, 0, dx) -> fuer A>0 nach innen. Daher umdrehen falls aussen gewuenscht.
    F = np.concatenate([np.c_[a, b, cc], np.c_[a, cc, d]])
    want_flip = (A > 0) == outward
    if want_flip:
        F = F[:, [0, 2, 1]]
    acc.add(layer, mat, V, F, c)


def sweep(acc, layer, mat, pts, normals, profile, c, y0=0.0, start_mitre=None, end_mitre=None,
          start_cap=True, end_cap=True, closed=False):
    """Profil entlang eines Polygonzugs (Waende) ziehen.
    pts: (K,3) Punkte auf der Wandflaeche in Bezugshoehe (y wird durch y0 + h ersetzt wenn y0 nicht None),
    normals: (K-1,3) horizontale Aussennormalen je Segment; profile: [(d,h),...] von Wand-unten ueber aussen nach Wand-oben.
    Gehrung an inneren Knoten; an den Enden optional Gehrungsvektor (start_mitre/end_mitre) statt Normaler."""
    pts = np.asarray(pts, float); N = np.asarray(normals, float); prof = np.asarray(profile, float)
    K = len(pts); J = len(prof)
    if K < 2 or J < 2:
        return
    nseg = K - 1 if not closed else K
    segN = N if not closed else N
    # Versatzvektoren an jedem Knoten
    off = np.zeros((K, 3))
    for k in range(K):
        if closed:
            n1 = segN[(k - 1) % nseg]; n2 = segN[k % nseg]
            off[k] = mitre(n1, n2)
        else:
            if k == 0:
                off[k] = start_mitre if start_mitre is not None else segN[0]
            elif k == K - 1:
                off[k] = end_mitre if end_mitre is not None else segN[-1]
            else:
                off[k] = mitre(segN[k - 1], segN[k])
    base = pts.copy()
    if y0 is not None:
        base[:, 1] = y0
    # Ringe: V[k, j] = base[k] + d_j * off[k] + h_j * up
    V = base[:, None, :] + prof[None, :, 0:1] * off[:, None, :] + prof[None, :, 1:2] * UP[None, None, :]
    Vf = V.reshape(-1, 3)
    F = []
    for s in range(nseg):
        k0 = s; k1 = (s + 1) % K
        for j in range(J - 1):
            a = k0 * J + j; b = k1 * J + j; cc = k1 * J + j + 1; d = k0 * J + j + 1
            F.append([a, b, cc]); F.append([a, cc, d])
    F = np.array(F, np.int64)
    # Orientierung: Profil laeuft unten->aussen->oben (CCW in (d,h)); Quad-Normale soll (dh*n - dd*up) entsprechen
    tri = Vf[F]
    nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    seg_idx = np.repeat(np.arange(nseg), (J - 1) * 2)
    jj = np.tile(np.repeat(np.arange(J - 1), 2), nseg)
    dd = prof[jj + 1, 0] - prof[jj, 0]; dh = prof[jj + 1, 1] - prof[jj, 1]
    exp = dh[:, None] * segN[seg_idx] - dd[:, None] * UP[None, :]
    flip = np.einsum('ij,ij->i', nn, exp) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    acc.add(layer, mat, Vf, F, c)
    if not closed:
        prof2 = prof[:, [0, 1]]
        V2, Fc = tri2d(prof2)
        if len(Fc):
            seg_dir = pts[1] - pts[0]; seg_dir[1] = 0
            if start_cap and start_mitre is None:
                Pc = base[0] + V2[:, 0:1] * off[0] + V2[:, 1:2] * UP
                _cap(acc, layer, mat, Pc, Fc, -seg_dir, c)
            seg_dir = pts[-1] - pts[-2]; seg_dir[1] = 0
            if end_cap and end_mitre is None:
                Pc = base[-1] + V2[:, 0:1] * off[-1] + V2[:, 1:2] * UP
                _cap(acc, layer, mat, Pc, Fc, seg_dir, c)


def _cap(acc, layer, mat, P, F, want, c):
    F = F.copy()
    tri = P[F]
    nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = nn @ want < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    acc.add(layer, mat, P, F, c)


def mitre(n1, n2, limit=3.0):
    n1 = np.asarray(n1, float); n2 = np.asarray(n2, float)
    dp = float(np.dot(n1, n2))
    if dp < -0.9:
        return n2
    m = (n1 + n2) / (1.0 + dp)
    L = np.linalg.norm(m)
    if L > limit:
        m = m / L * limit
    return m


def beam(acc, layer, mat, p0, p1, w, h, c, up_hint=UP, caps=True, center=True):
    """Balken (Quader) entlang Strecke p0->p1 mit Breite w (seitlich) und Hoehe h (in Richtung up_hint)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ax = p1 - p0; L = np.linalg.norm(ax)
    if L < 1e-6:
        return
    ax /= L
    s = np.cross(up_hint, ax)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(np.array([1.0, 0, 0]), ax)
    s /= np.linalg.norm(s)
    u = np.cross(ax, s)
    fr = Frame(p0, s, u, ax)                    # u=seitlich, v=hoch, d=entlang
    lo = [-w / 2, -h / 2 if center else 0, 0]; hi = [w / 2, h / 2 if center else h, L]
    acc.box(layer, mat, fr, lo, hi, c, skip=() if caps else ('-d', '+d'))


def cyl(acc, layer, mat, p0, p1, r, c, seg=8, caps=True):
    """Zylinder (Fallrohre, Lueftungsrohre)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ax = p1 - p0; L = np.linalg.norm(ax)
    if L < 1e-6:
        return
    ax /= L
    s = np.cross(UP, ax) if abs(ax[1]) < 0.99 else np.array([1.0, 0, 0])
    s /= np.linalg.norm(s); u = np.cross(ax, s)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.cos(a)[:, None] * s * r + np.sin(a)[:, None] * u * r
    V = np.concatenate([p0 + ring, p1 + ring])
    F = []
    for i in range(seg):
        j = (i + 1) % seg
        F += [[i, j, seg + j], [i, seg + j, seg + i]]
    F = np.array(F)
    tri = V[F]; nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ctr = (tri.mean(1) - p0); ctr -= np.outer(ctr @ ax, ax)
    flip = np.einsum('ij,ij->i', nn, ctr) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    acc.add(layer, mat, V, F, c)
    if caps:
        Fc = np.array([[0, i, i + 1] for i in range(1, seg - 1)])
        _cap(acc, layer, mat, p1 + ring, Fc, ax, c)

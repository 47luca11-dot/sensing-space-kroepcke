"""Geometrie-Kern fuer das Alte Rathaus (reines numpy, ausserhalb von Blender).

Spielframe: x Ost, y oben, z Sued (rechtshaendig, Meter, Ursprung Kroepcke-Uhr).
Frame: lokale Koordinaten (a, b, c) -> o + ex*a + ey*b + ez*c.  Fassadenframe: ex = Fassadentangente (vom Betrachter
aus nach rechts), ey = oben, ez = Aussennormale.  Polygone werden gegen den Uhrzeigersinn (von +c aus gesehen)
angegeben -> Normale = +c.

MB sammelt Dreiecke je Material (Position, Normale, UV in Metern); finish() schweisst identische Ecken und teilt UV durch
die Kachelgroesse des Materials.  Instanzen: MB.instance(proto_name, M4) mit Prototypen-MB.
"""
import math
import numpy as np
import mapbox_earcut as earcut

UP = np.array([0.0, 1.0, 0.0])


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def area2(p):
    """vorzeichenbehaftete Flaeche eines 2D-Rings (Spalten 0,1)"""
    p = np.asarray(p, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class Frame:
    def __init__(self, o, ex, ey, ez):
        self.o = np.asarray(o, float); self.ex = unit(ex); self.ey = unit(ey); self.ez = unit(ez)
        self.M = np.stack([self.ex, self.ey, self.ez], 1)   # columns

    @staticmethod
    def facade(p0, p1, y0=0.0):
        """p0 -> p1 (Plan x,z) von aussen gesehen links -> rechts"""
        t = unit([p1[0] - p0[0], 0.0, p1[1] - p0[1]])
        n = np.cross(t, UP)
        return Frame([p0[0], y0, p0[1]], t, UP, n)

    def P(self, pts):
        p = np.asarray(pts, float)
        return self.o + p @ self.M.T

    def D(self, d):
        return np.asarray(d, float) @ self.M.T

    def local(self, a, b, c):
        """Unterframe mit Ursprung (a,b,c) (lokal), gleiche Achsen"""
        return Frame(self.P([a, b, c]), self.ex, self.ey, self.ez)

    def rot_y(self, ang, a=0, b=0, c=0):
        """Unterframe um die lokale b-Achse gedreht (Grad)"""
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        ex = self.ex * ca - self.ez * sa
        ez = self.ex * sa + self.ez * ca
        return Frame(self.P([a, b, c]), ex, self.ey, ez)

    def to_plan(self, a, c=0.0):
        p = self.P([a, 0, c]); return (p[0], p[2])

    def from_world(self, p):
        return (np.asarray(p, float) - self.o) @ self.M


# --------------------------------------------------------------------------------------------- Mesh-Builder
class MB:
    def __init__(self, tiles=None):
        self.data = {}          # mat -> [P list, N list, UV list, C list]  (Dreiecke, je 3 Ecken; C = lineare Vertexfarbe)
        self.tiles = tiles or {}
        self.inst = {}          # proto -> list of 4x4
        self.col = None         # aktuelle Vertexfarbe (linear RGB) fuer neue Dreiecke; None = weiss

    def _acc(self, mat):
        if mat not in self.data:
            self.data[mat] = [[], [], [], []]
        return self.data[mat]

    def tris_world(self, mat, P, N, UV, C=None):
        """P (k,3,3) Weltpositionen, N (k,3,3), UV (k,3,2) in Metern, C (k,3,3) Vertexfarben (sonst self.col / weiss)"""
        d = self._acc(mat)
        P = np.asarray(P, float).reshape(-1, 3)
        d[0].append(P); d[1].append(np.asarray(N, float).reshape(-1, 3)); d[2].append(np.asarray(UV, float).reshape(-1, 2))
        if C is None:
            C = np.broadcast_to(np.asarray(self.col if self.col is not None else (1.0, 1.0, 1.0), float), P.shape)
        d[3].append(np.asarray(C, float).reshape(-1, 3))

    # ---- UV-Projektion: dominante lokale Achse
    @staticmethod
    def _uv_auto(L, n_loc):
        """L (...,3) lokale Punkte, n_loc lokale Normale -> UV (m) (u nach rechts, v nach oben)"""
        ax = np.argmax(np.abs(n_loc))
        if ax == 2:      # Front (a, b)
            return np.stack([L[..., 0] * np.sign(n_loc[2] or 1), L[..., 1]], -1)
        if ax == 0:      # Seite (c, b)
            return np.stack([-L[..., 2] * np.sign(n_loc[0] or 1), L[..., 1]], -1)
        return np.stack([L[..., 0], -L[..., 2] * np.sign(n_loc[1] or 1)], -1)   # oben/unten (a, c)

    def tri_local(self, mat, fr, L, uv=None, flip=False):
        """L (k,3,3) lokale Dreiecke (CCW = Normale). flache Schattierung."""
        L = np.asarray(L, float).reshape(-1, 3, 3)
        if flip: L = L[:, ::-1]
        n = np.cross(L[:, 1] - L[:, 0], L[:, 2] - L[:, 0])
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12
        if not ok.all():
            L = L[ok]; n = n[ok]; ln = ln[ok]
            if uv is not None: uv = np.asarray(uv)[ok]
        if not len(L): return
        n = n / ln[:, None]
        if uv is None:
            UV = np.zeros((len(L), 3, 2))
            ax = np.argmax(np.abs(n), 1)
            for k in range(3):
                m = ax == k
                if not m.any(): continue
                sgn = np.sign(n[m, k]); sgn[sgn == 0] = 1
                Lm = L[m]
                if k == 2: UV[m] = np.stack([Lm[..., 0] * sgn[:, None], Lm[..., 1]], -1)
                elif k == 0: UV[m] = np.stack([-Lm[..., 2] * sgn[:, None], Lm[..., 1]], -1)
                else: UV[m] = np.stack([Lm[..., 0], -Lm[..., 2] * sgn[:, None]], -1)
        else:
            UV = np.asarray(uv, float).reshape(-1, 3, 2)
            if flip: UV = UV[:, ::-1]
        W = fr.P(L.reshape(-1, 3)).reshape(-1, 3, 3)
        Nw = fr.D(n)
        self.tris_world(mat, W, np.repeat(Nw[:, None, :], 3, 1), UV)

    def poly(self, mat, fr, pts, holes=(), flip=False, uv=None, hint=None):
        """ebenes Polygon (lokal, CCW von der Normalen aus) mit Loechern -> earcut.
        hint: gewuenschte Normalenrichtung (lokal) -> Umlaufsinn wird danach ausgerichtet"""
        pts = np.asarray(pts, float)
        if len(pts) < 3: return
        n = np.zeros(3)
        for i in range(len(pts)):
            n += np.cross(pts[i], pts[(i + 1) % len(pts)])
        if np.linalg.norm(n) < 1e-12: return
        n = unit(n)
        if hint is not None and float(np.dot(n, hint)) < 0:
            n = -n
        if len(pts) == 3 and not holes:
            T = pts[None].copy()
            tn = np.cross(T[0, 1] - T[0, 0], T[0, 2] - T[0, 0])
            if float(np.dot(tn, n)) < 0: T = T[:, ::-1]
            self.tri_local(mat, fr, T, flip=flip); return
        # 2D-Basis in der Ebene
        a = unit(np.cross([0, 0, 1] if abs(n[2]) < 0.9 else [0, 1, 0], n))
        b = np.cross(n, a)
        rings = [pts] + [np.asarray(h, float) for h in holes]
        allp = np.concatenate(rings)
        p2 = np.stack([allp @ a, allp @ b], 1)
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        idx = earcut.triangulate_float64(p2, ends).reshape(-1, 3)
        if not len(idx): return
        T = allp[idx]
        tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        bad = (tn @ n) < 0
        T[bad] = T[bad][:, ::-1]
        self.tri_local(mat, fr, T, flip=flip)

    def quad(self, mat, fr, p0, p1, p2, p3, flip=False):
        L = np.array([[p0, p1, p2], [p0, p2, p3]], float)
        self.tri_local(mat, fr, L, flip=flip)

    def grid(self, mat, fr, G, closed_u=False, smooth=True, flip=False, uvG=None, cap=False):
        """G (nu, nv, 3) lokales Gitter; Flaechen (i,j)->(i+1,j)->(i+1,j+1)->(i,j+1).
        smooth: Eckennormalen gemittelt (gemeinsame Ecken)."""
        G = np.asarray(G, float)
        if closed_u:
            G = np.concatenate([G, G[:1]], 0)
            if uvG is not None: uvG = np.concatenate([uvG, uvG[:1] + (uvG[-1:] - uvG[-2:-1])], 0)
        nu, nv = G.shape[:2]
        a = G[:-1, :-1]; b = G[1:, :-1]; c = G[1:, 1:]; d = G[:-1, 1:]
        if flip: a, b, c, d = a, d, c, b
        fn = np.cross(c - a, d - b)
        fl = np.linalg.norm(fn, axis=-1, keepdims=True)
        fnu = fn / np.maximum(fl, 1e-12)
        if smooth:
            vn = np.zeros_like(G)
            vn[:-1, :-1] += fn; vn[1:, :-1] += fn; vn[1:, 1:] += fn; vn[:-1, 1:] += fn
            if closed_u:
                s = vn[0] + vn[-1]; vn[0] = s; vn[-1] = s
            vn = vn / np.maximum(np.linalg.norm(vn, axis=-1, keepdims=True), 1e-12)
            if flip:
                na, nb, nc, nd = vn[:-1, :-1], vn[:-1, 1:], vn[1:, 1:], vn[1:, :-1]
            else:
                na, nb, nc, nd = vn[:-1, :-1], vn[1:, :-1], vn[1:, 1:], vn[:-1, 1:]
        else:
            na = nb = nc = nd = fnu
        if uvG is None:
            # Laengen entlang u und v als UV (Meter)
            du = np.linalg.norm(np.diff(G, axis=0), axis=-1); su = np.concatenate([np.zeros((1, nv)), np.cumsum(du, 0)], 0)
            dv = np.linalg.norm(np.diff(G, axis=1), axis=-1); sv = np.concatenate([np.zeros((nu, 1)), np.cumsum(dv, 1)], 1)
            uvG = np.stack([su, sv], -1)
        ua = uvG[:-1, :-1]; ub = uvG[1:, :-1]; uc = uvG[1:, 1:]; ud = uvG[:-1, 1:]
        if flip: ub, ud = ud, ub
        valid = (fl[..., 0] > 1e-12).reshape(-1)
        A = a.reshape(-1, 3); B_ = b.reshape(-1, 3); C = c.reshape(-1, 3); D_ = d.reshape(-1, 3)
        NA, NB, NC, ND = [x.reshape(-1, 3) for x in (na, nb, nc, nd)]
        UA, UB, UC, UD = [x.reshape(-1, 2) for x in (ua, ub, uc, ud)]
        P = np.concatenate([np.stack([A, B_, C], 1), np.stack([A, C, D_], 1)])
        N = np.concatenate([np.stack([NA, NB, NC], 1), np.stack([NA, NC, ND], 1)])
        U = np.concatenate([np.stack([UA, UB, UC], 1), np.stack([UA, UC, UD], 1)])
        v2 = np.concatenate([valid, valid])
        # entartete Dreiecke entfernen
        tn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        v2 &= np.linalg.norm(tn, axis=1) > 1e-10
        P, N, U = P[v2], N[v2], U[v2]
        W = fr.P(P.reshape(-1, 3)).reshape(-1, 3, 3)
        Nw = fr.D(N.reshape(-1, 3)).reshape(-1, 3, 3)
        self.tris_world(mat, W, Nw, U)
        if cap:
            for ring, fl_ in ((G[:, 0], True), (G[:, -1], False)):
                r = ring[:-1] if closed_u else ring
                self.poly(mat, fr, r, flip=not fl_ if not flip else fl_)

    def box(self, mat, fr, a0, a1, b0, b1, c0, c1, faces='xXyYzZ'):
        """achsparalleler Quader in lokalen Koordinaten; faces: x=-a, X=+a, y=-b, Y=+b, z=-c, Z=+c"""
        P = lambda a, b, c: (a, b, c)
        if 'Z' in faces: self.quad(mat, fr, P(a0, b0, c1), P(a1, b0, c1), P(a1, b1, c1), P(a0, b1, c1))
        if 'z' in faces: self.quad(mat, fr, P(a1, b0, c0), P(a0, b0, c0), P(a0, b1, c0), P(a1, b1, c0))
        if 'X' in faces: self.quad(mat, fr, P(a1, b0, c1), P(a1, b0, c0), P(a1, b1, c0), P(a1, b1, c1))
        if 'x' in faces: self.quad(mat, fr, P(a0, b0, c0), P(a0, b0, c1), P(a0, b1, c1), P(a0, b1, c0))
        if 'Y' in faces: self.quad(mat, fr, P(a0, b1, c1), P(a1, b1, c1), P(a1, b1, c0), P(a0, b1, c0))
        if 'y' in faces: self.quad(mat, fr, P(a0, b0, c0), P(a1, b0, c0), P(a1, b0, c1), P(a0, b0, c1))

    def prism(self, mat, fr, outline, b0, b1, top=True, bottom=False, side_mat=None, top_mat=None):
        """Prisma: Grundriss outline (k,2) in (a,c) CCW von oben gesehen (also von +b), Hoehe b0..b1"""
        o = np.asarray(outline, float)
        if area2(o) > 0: o = o[::-1]          # in (a,c) muss der Umlauf negativ sein, damit Seiten nach aussen zeigen
        k = len(o)
        sm = side_mat or mat
        for i in range(k):
            p, q = o[i], o[(i + 1) % k]
            self.quad(sm, fr, (p[0], b0, p[1]), (q[0], b0, q[1]), (q[0], b1, q[1]), (p[0], b1, p[1]))
        if top: self.poly(top_mat or mat, fr, [(p[0], b1, p[1]) for p in o], hint=(0, 1, 0))
        if bottom: self.poly(mat, fr, [(p[0], b0, p[1]) for p in o], hint=(0, -1, 0))

    def extrude(self, mat, fr, outline, c0, c1, front=True, back=False, sides=True, side_mat=None, front_mat=None, holes=()):
        """Profil in (a,b) (CCW von +c gesehen), extrudiert c0..c1 (Front bei c1)"""
        o = np.asarray(outline, float); k = len(o)
        if area2(o) < 0: o = o[::-1]          # Profil CCW in (a,b)
        holes = [np.asarray(h, float)[::-1] if area2(np.asarray(h, float)) > 0 else np.asarray(h, float) for h in holes]
        if sides:
            sm = side_mat or mat
            for ring in [o] + list(holes):
                for i in range(len(ring)):
                    p, q = ring[i], ring[(i + 1) % len(ring)]
                    self.quad(sm, fr, (p[0], p[1], c0), (q[0], q[1], c0), (q[0], q[1], c1), (p[0], p[1], c1))
        if front: self.poly(front_mat or mat, fr, [(p[0], p[1], c1) for p in o], holes=[[(p[0], p[1], c1) for p in h] for h in holes], hint=(0, 0, 1))
        if back: self.poly(mat, fr, [(p[0], p[1], c0) for p in o], holes=[[(p[0], p[1], c0) for p in h] for h in holes], hint=(0, 0, -1))

    def instance(self, proto, M):
        self.inst.setdefault(proto, []).append(np.asarray(M, float))

    def merge(self, other):
        for m, d in other.data.items():
            a = self._acc(m)
            for k in range(4): a[k].extend(d[k])
        for p, L in other.inst.items():
            self.inst.setdefault(p, []).extend(L)

    def ntris(self):
        return sum(sum(len(x) for x in d[0]) // 3 for d in self.data.values())

    def finish(self, colors=False):
        """-> {mat: (P (n,3) f32, N (n,3) f32, UV (n,2) f32, I (m,3) u32)} (+ C (n,3) f32 wenn colors)"""
        out = {}
        for m, d in self.data.items():
            if not d[0]: continue
            P = np.concatenate(d[0]); N = np.concatenate(d[1]); U = np.concatenate(d[2]); C = np.concatenate(d[3])
            tw, th = self.tiles.get(m, (1.0, 1.0))
            U = np.stack([U[:, 0] / tw, -U[:, 1] / th], 1)
            N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
            key = np.concatenate([np.round(P * 2000), np.round(N * 500), np.round(U * 4000), np.round(C * 255)], 1).astype(np.int64)
            _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            I = inv.reshape(-1, 3)
            # Hygiene: Dreiecke ohne Flaeche (durch das Verschweissen zusammengefallen) und exakte Duplikate (gleiche
            # Indizes, gleicher Umlaufsinn) entfernen, danach unbenutzte Vertices kompaktieren
            Pf = P[first]
            T = Pf[I]
            ar = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
            ok = (I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 0] != I[:, 2]) & (ar > 2e-6)      # Flaeche > 1 mm2 (Splitter, die erst in float32 kollabieren)
            I = I[ok]
            if len(I):
                r0 = np.argmin(I, axis=1)
                canon = np.stack([I[np.arange(len(I)), r0], I[np.arange(len(I)), (r0 + 1) % 3], I[np.arange(len(I)), (r0 + 2) % 3]], 1)
                _, keep = np.unique(canon, axis=0, return_index=True)
                I = I[np.sort(keep)]
            if not len(I): continue
            used = np.unique(I.reshape(-1))
            remap = np.full(len(first), -1, np.int64); remap[used] = np.arange(len(used))
            sel = first[used]
            I = remap[I]
            r = (P[sel].astype(np.float32), N[sel].astype(np.float32), U[sel].astype(np.float32), I.astype(np.uint32))
            out[m] = r + (C[sel].astype(np.float32),) if colors else r
        return out


# --------------------------------------------------------------------------------------------- Bogenformen
def arch_outline(kind, w, h_spring, rise=None, n=16, y0=0.0):
    """Oeffnungsumriss (CCW von vorne), Mitte bei a=0, Unterkante y0, Kaempfer h_spring (ueber y0).
    kind: rect | seg (Segmentbogen, rise) | point (Spitzbogen, rise=Pfeilhoehe) | round (Halbkreis) | flat
    -> (k,2) Punkte (a, b); beginnt unten links"""
    hw = w / 2.0
    pts = [(-hw, y0), (hw, y0)]
    ys = y0 + h_spring
    if kind == 'rect' or kind == 'flat':
        pts += [(hw, ys), (-hw, ys)]
        return np.array(pts)
    pts.append((hw, ys))
    top = []
    if kind == 'round':
        for i in range(1, n):
            t = math.pi * i / n
            top.append((hw * math.cos(t), ys + hw * math.sin(t)))
    elif kind == 'seg':
        r = rise if rise else w * 0.18
        R = (hw * hw + r * r) / (2 * r); cy = ys + r - R
        a0 = math.atan2(ys - cy, hw)
        for i in range(1, n):
            t = a0 + (math.pi - 2 * a0) * i / n
            top.append((R * math.cos(t), cy + R * math.sin(t)))
    elif kind == 'point':
        r = rise if rise else w * 0.85
        # zwei Kreise: Mittelpunkte auf Kaempferlinie bei +-(R - hw); Spitze bei (0, ys + r)
        # R aus: (R - (R - hw))... Kreis um (hw - R, ys) durch (hw, ys) und (0, ys + r):  (R-hw)^2 + r^2 = R^2 -> R = (hw^2 + r^2)/(2 hw)
        R = (hw * hw + r * r) / (2 * hw)
        cx = hw - R
        amax = math.atan2(r, -cx)
        m = n // 2
        right = [(cx + R * math.cos(amax * i / m), ys + R * math.sin(amax * i / m)) for i in range(1, m + 1)]
        left = [(-x, y) for x, y in right[:-1]][::-1]
        top = right + left
    pts += top
    pts.append((-hw, ys))
    return np.array(pts)


def offset_outline(kind, w, h_spring, rise, inset, n=16, y0=0.0, keep_bottom=True):
    """Umriss um inset nach innen versetzt (gleiche Bogenart, gleiche Kaempferhoehe relativ)"""
    w2 = w - 2 * inset
    if kind in ('point',):
        rise2 = (rise if rise else w * 0.85) - inset * 1.05
    elif kind == 'seg':
        rise2 = max(0.02, (rise if rise else w * 0.18) - inset * 0.5)
    else:
        rise2 = rise
    yb = y0 + (0 if keep_bottom else inset)
    return arch_outline(kind, w2, h_spring + (y0 - yb), rise2, n, yb)


def arc_param(outer, inner=None, start=1):
    """Bogenlaenge (m) entlang der Mittellinie zweier Umrisse gleicher Punktzahl, beginnend bei Punkt 'start'
    (unten rechts) im Umlaufsinn -> s[i] je Punkt (fuer Keilstein-UVs: u = s)."""
    mid = np.asarray(outer, float) if inner is None else (np.asarray(outer, float) + np.asarray(inner, float)) / 2
    k = len(mid); s = np.zeros(k); acc = 0.0; prev = None
    for t in range(k):
        i = (start + t) % k
        if prev is not None: acc += float(np.linalg.norm(mid[i] - mid[prev]))
        s[i] = acc; prev = i
    s_close = acc + float(np.linalg.norm(mid[start - 1] - mid[prev])) if start > 0 else acc
    return s, s_close


def _s_pair(s, i, j, k):
    """u an den Enden des Segments i->j (Ende des Umlaufs: Schluss-Laenge statt s[0])"""
    sv, sc = s
    return sv[i], (sc if (j == 0 and k > 1) else sv[j])


def ring_between(mb, mat, fr, outer, inner, c, alt_mat=None, seg=1, skip_bottom=True, mats_fn=None, s=None):
    """Flaeche zwischen zwei Umrissen gleicher Punktzahl in Ebene c (Normale +c).
    alt_mat: abwechselnde Materialien alle 'seg' Segmente (Keilsteine).
    s: (arc_param-Ergebnis) -> UV u = Bogenlaenge, v = Abstand vom inneren Rand (radiale Keilsteine im Texturwechsel)."""
    k = len(outer)
    for i in range(k):
        j = (i + 1) % k
        if skip_bottom and i == 0: continue   # Unterkante (Brüstung) offen
        mat_i = mat
        if mats_fn is not None: mat_i = mats_fn(i)
        elif alt_mat is not None and (i // seg) % 2 == 1: mat_i = alt_mat
        A = (outer[i][0], outer[i][1], c); B = (outer[j][0], outer[j][1], c); C = (inner[j][0], inner[j][1], c); D = (inner[i][0], inner[i][1], c)
        if s is None:
            mb.quad(mat_i, fr, A, B, C, D)
        else:
            ui, uj = _s_pair(s, i, j, k)
            wi = float(np.hypot(outer[i][0] - inner[i][0], outer[i][1] - inner[i][1]))
            wj = float(np.hypot(outer[j][0] - inner[j][0], outer[j][1] - inner[j][1]))
            uv = np.array([[[ui, wi], [uj, wj], [uj, 0.0]], [[ui, wi], [uj, 0.0], [ui, 0.0]]])
            mb.tri_local(mat_i, fr, np.array([[A, B, C], [A, C, D]], float), uv=uv)


def side_band(mb, mat, fr, outline, c0, c1, skip_bottom=True, mats_fn=None, inward=True, s=None):
    """Laibung: Seitenflaechen eines Oeffnungsumrisses von c1 (vorne) bis c0 (hinten), Normalen in die Oeffnung.
    s: (arc_param-Ergebnis) -> UV u = Bogenlaenge, v = Tiefe (Steinwechsel laeuft um die Kante herum)."""
    k = len(outline)
    for i in range(k):
        if skip_bottom and i == 0: continue
        j = (i + 1) % k
        p, q = outline[i], outline[j]
        m = mats_fn(i) if mats_fn else mat
        if inward:
            Q = [(p[0], p[1], c1), (q[0], q[1], c1), (q[0], q[1], c0), (p[0], p[1], c0)]
        else:
            Q = [(q[0], q[1], c1), (p[0], p[1], c1), (p[0], p[1], c0), (q[0], q[1], c0)]
        if s is None:
            mb.quad(m, fr, *Q)
        else:
            ui, uj = _s_pair(s, i, j, k)
            # Tiefe als v (vorne = 0.29: Kopfseite der Keilsteine schliesst an die Front an)
            if inward:
                us = [(ui, 0.29), (uj, 0.29), (uj, 0.29 - (c1 - c0)), (ui, 0.29 - (c1 - c0))]
            else:
                us = [(uj, 0.29), (ui, 0.29), (ui, 0.29 - (c1 - c0)), (uj, 0.29 - (c1 - c0))]
            uv = np.array([[us[0], us[1], us[2]], [us[0], us[2], us[3]]])
            mb.tri_local(m, fr, np.array([[Q[0], Q[1], Q[2]], [Q[0], Q[2], Q[3]]], float), uv=uv)


# --------------------------------------------------------------------------------------------- Transformationen
def trs_matrix(t, yaw=0.0, s=1.0, pitch=0.0):
    """4x4: Skalierung s (Skalar oder 3), Drehung um Y (yaw, rad), dann Translation t"""
    s = np.broadcast_to(np.asarray(s, float), (3,))
    c, sn = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
    M = np.eye(4); M[:3, :3] = R @ np.diag(s); M[:3, 3] = t
    return M


def frame_matrix(fr, s=1.0):
    """4x4 aus Frame (Spalten ex, ey, ez) mit Skalierung"""
    M = np.eye(4); M[:3, :3] = fr.M * s; M[:3, 3] = fr.o
    return M

"""Dachgenerator: Dachteile (Grundrissregion + Dachebenen), Hoehenfunktion, Ebenen-Polygone, Dachaufbau mit echten
Ziegelreihen (Saegezahn-Stufen je Reihe), Traufbrett, Untersicht, Firstziegel-Instanzen, Kehlbleche, Rinnen.
Spielframe: x Ost, y oben, z Sued.  Plan = (x, z)."""
import math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box as sbox
from shapely.ops import unary_union
from bh_mesh import Frame, MB, unit, area2, ccw, offset_poly, tri_poly, UP


class Plane:
    """y = a*x + b*z + c"""

    def __init__(self, a, b, c, name=''):
        self.a, self.b, self.c, self.name = float(a), float(b), float(c), name
        self.ov_e = None; self.ov_v = None

    @staticmethod
    def p3(p0, p1, p2, name=''):
        P = np.array([p0, p1, p2], float)            # (x, y, z)
        A = np.c_[P[:, 0], P[:, 2], np.ones(3)]
        a, b, c = np.linalg.solve(A, P[:, 1])
        return Plane(a, b, c, name)

    @staticmethod
    def eave(pa, pb, y, slope, inside, name=''):
        """Traufe pa-pb (Plan), Traufhoehe y, Neigung (Grad), steigt Richtung 'inside' (Planpunkt)"""
        pa = np.asarray(pa, float); pb = np.asarray(pb, float)
        t = unit(pb - pa); n = np.array([-t[1], t[0]])
        if (np.asarray(inside, float) - pa) @ n < 0: n = -n
        k = math.tan(math.radians(slope))
        # y = y + k * ((x,z) - pa) . n
        return Plane(k * n[0], k * n[1], y - k * (pa @ n), name)

    def h(self, x, z):
        return self.a * np.asarray(x) + self.b * np.asarray(z) + self.c

    def normal(self):
        return unit([-self.a, 1.0, -self.b])

    def slope(self):
        return math.degrees(math.atan(math.hypot(self.a, self.b)))


def halfplane_le(pi, pj, L=3000.0, cx=-420.0, cz=220.0):
    """{(x,z): h_i <= h_j} als Polygon"""
    A, B, C = pi.a - pj.a, pi.b - pj.b, pi.c - pj.c
    nn = math.hypot(A, B)
    if nn < 1e-9:
        return sbox(cx - L, cz - L, cx + L, cz + L) if C <= 1e-9 else Polygon()
    n = np.array([A, B]) / nn
    p0 = -C / nn * n
    # Bezugspunkt nahe dem Gebiet
    q = np.array([cx, cz]); p0 = p0 + ((q - p0) @ np.array([-n[1], n[0]])) * np.array([-n[1], n[0]])
    d = np.array([-n[1], n[0]])
    return Polygon([p0 + d * L, p0 - d * L, p0 - d * L - n * L, p0 + d * L - n * L])


def as_polys(g):
    if g.is_empty: return []
    if isinstance(g, Polygon): return [g]
    if isinstance(g, MultiPolygon): return list(g.geoms)
    return [x for x in getattr(g, 'geoms', []) if isinstance(x, Polygon)]


class RoofSet:
    """parts: [(region Polygon, [Plane, ...])].  h(x,z) = max ueber Teile (die den Punkt enthalten) von min ueber Ebenen."""

    def __init__(self, parts, outline=None):
        self.parts = [(Polygon(r) if not isinstance(r, Polygon) else r, pl) for r, pl in parts]
        self.planes = []
        for _, pl in self.parts:
            for p in pl:
                if p not in self.planes: self.planes.append(p)
        self.outline = outline
        self._polys()

    def h(self, x, z):
        best = None
        for reg, pl in self.parts:
            if reg.buffer(1e-3).contains(Point(x, z)):
                v = min(p.h(x, z) for p in pl)
                best = v if best is None else max(best, v)
        return best

    def h_near(self, x, z):
        v = self.h(x, z)
        if v is not None: return v
        # naechster Teil
        d = [(reg.distance(Point(x, z)), k) for k, (reg, _) in enumerate(self.parts)]
        _, k = min(d)
        return min(p.h(x, z) for p in self.parts[k][1])

    def _polys(self):
        self.pp = {}
        cx, cz = (self.parts[0][0].centroid.x, self.parts[0][0].centroid.y)
        for k, (reg, pl) in enumerate(self.parts):
            for p in pl:
                R = reg
                for q in pl:
                    if q is p: continue
                    R = R.intersection(halfplane_le(p, q, cx=cx, cz=cz))
                for k2, (reg2, pl2) in enumerate(self.parts):
                    if k2 == k or p in pl2: continue
                    if not reg2.intersects(R): continue
                    D = reg2
                    for q in pl2:
                        D = D.intersection(halfplane_le(p, q, cx=cx, cz=cz).buffer(0))   # h_p <= h_q
                    # Bereich, in dem Teil 2 hoeher ist: dort h_p < min(Teil 2) -> entfernen
                    R = R.difference(D)
                if not R.is_empty:
                    self.pp[p] = unary_union([self.pp[p], R]) if p in self.pp else R
        for p in list(self.pp):
            g = self.pp[p].buffer(0)
            g = unary_union([x for x in as_polys(g) if x.area > 0.02])
            if g.is_empty: del self.pp[p]
            else: self.pp[p] = g


# ============================================================================================ Dachaufbau
class RoofBuilder:
    """Baut die Dachdeckung fuer ein RoofSet.
    opts: tile_mat, course (m), lift (m), thick, eave (Ueberstand horiz.), verge, soffit_mat, fascia_mat, party_lines
    (Liste von Plan-Segmenten, an denen das Dach an Nachbarn stoesst), flat_mat, ridge (Instanzliste)"""

    def __init__(self, rs, mb, opts, inst):
        self.rs = rs; self.mb = mb; self.o = opts; self.inst = inst
        self.eaves = []          # (p0_3d, p1_3d, outward_plan) fuer Rinnen
        self.ridges = []

    def _classify(self, p, P3, i):
        """Kante i des Ebenenpolygons (3D-Punkte P3) -> 'shared'/'eave'/'verge'/'top'/'party'"""
        a, b = P3[i], P3[(i + 1) % len(P3)]
        m = (a + b) / 2
        mp = Point(m[0], m[2])
        for q, g in self.rs.pp.items():
            if q is p: continue
            if g.boundary.distance(mp) < 0.03 and abs(q.h(m[0], m[2]) - m[1]) < 0.06:
                # Laengs-Ueberdeckung pruefen (beide Enden nahe am anderen Rand oder Kante kurz)
                if g.boundary.distance(Point(a[0], a[2])) < 0.05 or g.boundary.distance(Point(b[0], b[2])) < 0.05:
                    return 'shared', q
        for seg in self.o.get('party_lines', []):
            if LineString(seg).distance(mp) < 0.2 and LineString(seg).distance(Point(a[0], a[2])) < 0.25 and LineString(seg).distance(Point(b[0], b[2])) < 0.25:
                return 'party', None
        # Dach direkt ausserhalb hoeher (Gaube/Zwerchhaus/Dachsprung) -> Anschluss ohne Ueberstand
        d2 = np.array([b[0] - a[0], b[2] - a[2]]); d2 = d2 / (np.linalg.norm(d2) + 1e-12)
        nrm = np.array([d2[1], -d2[0]])
        poly2 = Polygon(P3[:, [0, 2]])
        if poly2.contains(Point(m[0] + nrm[0] * 0.05, m[2] + nrm[1] * 0.05)): nrm = -nrm
        qx, qz = m[0] + nrm[0] * 0.06, m[2] + nrm[1] * 0.06
        hq = self.rs.h(qx, qz)
        if hq is not None and hq > m[1] + 0.05:
            return 'abut', None
        if abs(a[1] - b[1]) < 0.06:
            ymin = min(P3[:, 1])
            if a[1] < ymin + 0.15: return 'eave', None
            return 'top', None
        return 'verge', None

    def build(self):
        o = self.o
        for p, g in self.rs.pp.items():
            for poly in as_polys(g):
                self.plane_poly(p, poly)
        # First-/Gratziegel
        done = set()
        for (p, q, A, B) in self.ridges:
            key = tuple(sorted([id(p), id(q)])) + tuple(np.round(np.r_[A + B], 1))
            if key in done: continue
            done.add(key)
            self.ridge_line(p, q, A, B)
        return self

    def plane_poly(self, p, poly):
        o = self.o
        n = p.normal()
        flat = p.slope() < 3.0
        eu = unit(np.cross(UP, n)) if not flat else np.array([1.0, 0, 0])
        ev = np.cross(n, eu)
        ext = np.array(poly.exterior.coords)[:-1]
        P3 = np.c_[ext[:, 0], p.h(ext[:, 0], ext[:, 1]), ext[:, 1]]
        o3 = P3[np.argmin(P3[:, 1])]
        F = Frame(o3, eu, ev, n)
        L2 = F.local(P3)[:, :2]
        if area2(L2) < 0:
            L2 = L2[::-1]; P3 = P3[::-1]
        holes = []
        for hh in poly.interiors:
            hc = np.array(hh.coords)[:-1]
            H3 = np.c_[hc[:, 0], p.h(hc[:, 0], hc[:, 1]), hc[:, 1]]
            holes.append(F.local(H3)[:, :2])
        nE = len(L2)
        kinds = []; offs = np.zeros(nE)
        cosv = math.cos(math.radians(p.slope()))
        for i in range(nE):
            kd, q = self._classify(p, P3, i)
            kinds.append((kd, q))
            e = L2[(i + 1) % nE] - L2[i]
            t = unit(e); m = np.array([t[1], -t[0]])          # aussen (CCW)
            hl = math.sqrt(m[0] ** 2 + (m[1] * cosv) ** 2) or 1.0
            if kd == 'eave': offs[i] = (p.ov_e if p.ov_e is not None else o.get('eave', 0.45)) / hl
            elif kd == 'verge': offs[i] = (p.ov_v if p.ov_v is not None else o.get('verge', 0.22)) / hl
            elif kd == 'abut': offs[i] = 0.03
            elif kd == 'party': offs[i] = 0.04
            elif kd == 'top': offs[i] = o.get('top', 0.06)
            else: offs[i] = 0.0
        Lo = offset_poly(L2, offs) if np.any(offs) else L2
        for i, (kd, q) in enumerate(kinds):
            if kd == 'shared' and self.rs.pp.get(q) is not None:
                a3, b3 = F.P([Lo[i, 0], Lo[i, 1], 0.0]), F.P([Lo[(i + 1) % nE, 0], Lo[(i + 1) % nE, 1], 0.0])
                mid = (a3 + b3) / 2
                cq = self.rs.pp[q].representative_point()
                cq3 = np.array([cq.x, q.h(cq.x, cq.y), cq.y])
                convex = n @ (cq3 - mid) < 0
                if convex:
                    self.ridges.append((p, q, a3, b3))
                else:
                    self.valley(p, q, a3, b3)
        edge_ls = [(LineString([Lo[i], Lo[(i + 1) % nE]]), kinds[i][0]) for i in range(nE)]
        if flat:
            self.flat_roof(p, F, Lo, holes, kinds, L2)
            return
        tile = o.get('tile_mat', 'Ziegel')
        course = o.get('course', 0.30); lift = o.get('lift', 0.028); thick = o.get('thick', 0.20)
        sm = o.get('smooth', {})
        if course is None or p.name in sm:
            self.smooth_roof(p, F, Lo, holes, kinds, sm.get(p.name, tile), thick)
            return
        v0 = Lo[:, 1].min()
        vmax = Lo[:, 1].max()
        PL = Polygon(Lo, holes)
        if not PL.is_valid: PL = PL.buffer(0)
        umin, umax = Lo[:, 0].min() - 1, Lo[:, 0].max() + 1
        nk = int(math.ceil((vmax - v0) / course - 1e-6))
        mb = self.mb
        def lift_at(v, k):
            return lift * (1.0 - (v - (v0 + k * course)) / course)
        for k in range(nk):
            va, vb = v0 + k * course, min(v0 + (k + 1) * course, vmax + 1e-6)
            S = PL.intersection(sbox(umin, va, umax, v0 + (k + 1) * course))
            for piece in as_polys(S):
                if piece.area < 1e-4: continue
                ex = np.array(piece.exterior.coords)[:-1]
                hs = [np.array(r.coords)[:-1] for r in piece.interiors]
                V, I = tri_poly(ccw(ex), [h[::-1] if area2(h) > 0 else h for h in hs])
                if not len(I): continue
                lv = lift_at(V[:, 1], k)
                P = F.P(np.c_[V, lv])
                uv = np.c_[V[:, 0], (V[:, 1] - v0)]
                mb.tris(tile, P, I, uv)
                # Stossflaechen (Reihenunterkante) und Seiten
                rings = [ccw(ex)] + [h[::-1] if area2(h) > 0 else h for h in hs]
                for ring in rings:
                    for j in range(len(ring)):
                        a2, b2 = ring[j], ring[(j + 1) % len(ring)]
                        if abs(a2[1] - va) < 1e-5 and abs(b2[1] - va) < 1e-5:
                            # Unterkante dieser Reihe: Stufe von lift 0 (Vorreihe) bis lift (diese Reihe), Blick hangabwaerts
                            lo = 0.0 if k > 0 else -thick
                            mat = tile if k > 0 else o.get('fascia_mat', 'Holz_dunkel')
                            if k == 0:
                                mb.quad(mat, F.P([a2[0], va, lo]), F.P([b2[0], va, lo]), F.P([b2[0], va, 0.0]), F.P([a2[0], va, 0.0]))
                                mb.quad(tile, F.P([a2[0], va, 0.0]), F.P([b2[0], va, 0.0]), F.P([b2[0], va, lift]), F.P([a2[0], va, lift]),
                                        np.array([[a2[0], -0.02], [b2[0], -0.02], [b2[0], 0.0], [a2[0], 0.0]]))
                            else:
                                mb.quad(tile, F.P([a2[0], va, lo]), F.P([b2[0], va, lo]), F.P([b2[0], va, lift]), F.P([a2[0], va, lift]),
                                        np.array([[a2[0], -0.02], [b2[0], -0.02], [b2[0], 0.0], [a2[0], 0.0]]))
                            continue
                        if abs(a2[1] - b2[1]) < 1e-6 and abs(a2[1] - (v0 + (k + 1) * course)) < 1e-5:
                            continue      # Oberkante der Reihe (innen)
                        # Randkante auf dem Dachumriss? -> Seitenflaeche (Ortgang/Traufe/Anschluss)
                        mp = Point((a2 + b2) / 2)
                        if PL.boundary.distance(mp) < 1e-4:
                            dk = min(edge_ls, key=lambda e_: e_[0].distance(mp))
                            if dk[1] == 'shared' and dk[0].distance(mp) < 1e-3: continue
                            la, lb = lift_at(a2[1], k), lift_at(b2[1], k)
                            mb.quad(o.get('fascia_mat', 'Holz_dunkel'), F.P([a2[0], a2[1], -thick]), F.P([b2[0], b2[1], -thick]),
                                    F.P([b2[0], b2[1], lb]), F.P([a2[0], a2[1], la]))
        # Untersicht (Sparren-/Brettschalung)
        V, I = tri_poly(ccw(Lo), [h[::-1] if area2(h) > 0 else h for h in holes])
        P = F.P(np.c_[V, np.full(len(V), -thick)])
        mb.tris(o.get('soffit_mat', 'Traufe'), P, I[:, ::-1], np.c_[V[:, 0], V[:, 1]])
        # Traufen fuer Rinnen
        for i, (kd, q) in enumerate(kinds):
            if kd == 'eave':
                a2, b2 = Lo[i], Lo[(i + 1) % nE]
                A3 = F.P([a2[0], a2[1], -thick]); B3 = F.P([b2[0], b2[1], -thick])
                out = unit(np.cross(B3 - A3, UP))
                if out @ F.D([0, -1, 0]) < 0: out = -out
                self.eaves.append((A3, B3, out, p))

    def smooth_roof(self, p, F, Lo, holes, kinds, mat, thick):
        mb = self.mb; o = self.o
        V, I = tri_poly(ccw(Lo), [h[::-1] if area2(h) > 0 else h for h in holes])
        v0 = Lo[:, 1].min()
        mb.tris(mat, F.P(np.c_[V, np.zeros(len(V))]), I, np.c_[V[:, 0], V[:, 1] - v0])
        mb.tris(o.get('soffit_mat', 'Traufe'), F.P(np.c_[V, np.full(len(V), -thick)]), I[:, ::-1], V.copy())
        n = len(Lo)
        for i, (kd, q) in enumerate(kinds):
            if kd == 'shared': continue
            a2, b2 = Lo[i], Lo[(i + 1) % n]
            mb.quad(o.get('fascia_mat', 'Holz_dunkel') if mat != 'Glasdach' else 'Stahl_dunkel',
                    F.P([a2[0], a2[1], -thick]), F.P([b2[0], b2[1], -thick]), F.P([b2[0], b2[1], 0.0]), F.P([a2[0], a2[1], 0.0]))
        if mat == 'Glasdach':
            # Sprossen
            umin, umax = Lo[:, 0].min(), Lo[:, 0].max()
            PL = Polygon(Lo)
            for u in np.arange(umin + 1.2, umax, 1.2):
                ln = PL.intersection(LineString([(u, -100), (u, 100)]))
                for g in getattr(ln, 'geoms', [ln]):
                    if g.is_empty or g.length < 0.2: continue
                    (ua, va), (ub, vb) = list(g.coords)[0], list(g.coords)[-1]
                    mb.box('Stahl_dunkel', F.sub(u, 0, 0), -0.03, 0.03, min(va, vb), max(va, vb), 0.0, 0.06, skip=('-c',))

    def flat_roof(self, p, F, Lo, holes, kinds, L2):
        """Flachdach: Kies/Bahn mit Attika-Abdeckung (Zink) an offenen Kanten"""
        mb = self.mb; o = self.o
        mat = o.get('flat_mat', 'Dachbahn')
        V, I = tri_poly(ccw(L2), holes)
        P = F.P(np.c_[V, np.full(len(V), 0.0)])
        mb.tris(mat, P, I, V.copy())
        par = o.get('parapet', 0.25)
        for i, (kd, q) in enumerate(kinds):
            if kd in ('shared', 'party', 'top'): continue
            a2, b2 = L2[i], L2[(i + 1) % len(L2)]
            t = unit(b2 - a2); m = np.array([t[1], -t[0]])
            # Attika: Aussenflaeche (Putz) bis +par, Abdeckung 0.30 breit, Innenseite
            w = 0.30
            A0, B0 = a2 + m * 0.05, b2 + m * 0.05
            A1, B1 = a2 - m * (w - 0.05), b2 - m * (w - 0.05)
            mb.quad(o.get('wall_mat', 'Putz_hell'), F.P([A0[0], A0[1], -0.4]), F.P([B0[0], B0[1], -0.4]), F.P([B0[0], B0[1], par]), F.P([A0[0], A0[1], par]))
            mb.quad('Zink', F.P([A0[0], A0[1], par]), F.P([B0[0], B0[1], par]), F.P([B1[0], B1[1], par]), F.P([A1[0], A1[1], par]))
            mb.quad(o.get('wall_mat', 'Putz_hell'), F.P([B1[0], B1[1], 0.0]), F.P([A1[0], A1[1], 0.0]), F.P([A1[0], A1[1], par]), F.P([B1[0], B1[1], par]))

    def valley(self, p, q, a3, b3):
        """Kehlblech: je Ebene ein 0.2 m breiter Streifen knapp ueber den Ziegeln"""
        if id(p) > id(q): return          # nur einmal je Paar
        lift = self.o.get('lift', 0.028)
        for pl in (p, q):
            n = pl.normal()
            e = unit(b3 - a3)
            side = unit(np.cross(n, e))
            g = self.rs.pp[pl]
            mid = (a3 + b3) / 2
            test = mid + side * 0.3
            if not g.buffer(0.01).contains(Point(test[0], test[2])): side = -side
            off = n * (lift + 0.012)
            self.mb.quad('Zink', a3 + off, b3 + off, b3 + off + side * 0.22, a3 + off + side * 0.22) if (np.cross(b3 - a3, side) @ n) > 0 else \
                self.mb.quad('Zink', b3 + off, a3 + off, a3 + off + side * 0.22, b3 + off + side * 0.22)

    def ridge_line(self, p, q, A, B):
        n = unit(p.normal() + q.normal())
        L = float(np.linalg.norm(B - A))
        if L < 0.15: return
        e = (B - A) / L
        # Firstziegel von unten nach oben (bei Graten) bzw. in einer Richtung
        if B[1] < A[1] - 0.05: A, B = B, A; e = -e
        k = max(1, int(round(L / 0.40)))
        step = L / k
        lift = self.o.get('lift', 0.028)
        ey = unit(n - (n @ e) * e)
        ez = np.cross(e, ey)
        for i in range(k):
            o3 = A + e * (i * step) + ey * (lift + 0.015)
            M = np.eye(4); M[:3, 0] = e * (step / 0.40); M[:3, 1] = ey; M[:3, 2] = ez; M[:3, 3] = o3
            self.inst.setdefault(self.o.get('ridge_proto', 'ridge'), []).append(M)

    def gutters(self, skip=lambda A, B: False, r=0.075, mat='Zink'):
        for A3, B3, out, p in self.eaves:
            if skip(A3, B3): continue
            L = np.linalg.norm(B3 - A3)
            if L < 0.3: continue
            off = out * (r + 0.01) + np.array([0, -0.02, 0])
            self.mb.tube(mat, [A3 + off, B3 + off], r, n=6, half=True)
            # Endkappen (Halbscheiben)
            for P0, d in ((A3 + off, unit(A3 - B3)), (B3 + off, unit(B3 - A3))):
                ring = []
                e1 = unit(np.cross(d, UP));
                for t in np.linspace(math.pi, 2 * math.pi, 7):
                    ring.append(P0 + r * (math.cos(t) * e1 + math.sin(t) * UP))
                for j in range(6):
                    tri = [P0, ring[j], ring[j + 1]]
                    f = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                    if f @ d < 0: tri = [P0, ring[j + 1], ring[j]]
                    self.mb.add(mat, tri, [d] * 3, [[0, 0], [0.1, 0], [0, 0.1]])
            # Rinnenhaken alle 0.9 m
            k = int(L / 0.9)
            for i in range(1, k + 1):
                c = A3 + (B3 - A3) * (i / (k + 1)) + off
                self.inst.setdefault('gutter_hook', []).append(_frame_m(c, unit(B3 - A3), UP))


def _frame_m(o, ex, ey):
    ex = unit(ex); ey = unit(ey - (ey @ ex) * ex); ez = np.cross(ex, ey)
    M = np.eye(4); M[:3, 0] = ex; M[:3, 1] = ey; M[:3, 2] = ez; M[:3, 3] = o
    return M

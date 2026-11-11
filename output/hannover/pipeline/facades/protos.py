"""Hochdetaillierte Prototypen fuer Fenster, Tueren, Schaufenster, Baenke, Faschen, Rolllaeden, Innenansichten.
Jeder Prototyp wird einmal gebaut und per EXT_mesh_gpu_instancing vielfach platziert (Instanzfarbe = Rahmen-/Stoff-/Glasfarbe).

Lokale Koordinaten: u = rechts, v = oben, d = nach aussen (Meter).
Ursprung 'frame': Mitte Unterkante der Oeffnung in der Ebene der Rahmen-Vorderkante (Wandebene - Laibungstiefe).
Ursprung 'wall' : Mitte Unterkante der Oeffnung in der Wandebene.
Nennmass (w0, h0): Instanzen werden mit (w/w0, h/h0, 1) skaliert."""
import math
import numpy as np
from geom import tri2d

PROTOS = {}          # key -> Proto


class Proto:
    def __init__(self, key, origin, w0, h0, mats=None):
        self.key = key; self.origin = origin; self.w0 = w0; self.h0 = h0
        self.parts = {}                 # mat -> [V list, F list, C list, n]

    # ------------------------------------------------------------ Grundfunktionen
    def add(self, mat, V, F, c=(1, 1, 1)):
        V = np.asarray(V, float); F = np.asarray(F, np.int64)
        if len(F) == 0:
            return
        c = np.asarray(c, float)
        C = np.broadcast_to(c, (len(V), 3)) if c.ndim == 1 else c
        L = self.parts.setdefault(mat, [[], [], [], 0])
        L[0].append(V); L[1].append(F + L[3]); L[2].append(np.asarray(C, float)); L[3] += len(V)

    def oriented(self, mat, V, F, want, c=(1, 1, 1)):
        """Dreiecke so drehen, dass ihre Normale zu want (pro Dreieck oder global) zeigt."""
        V = np.asarray(V, float); F = np.asarray(F, np.int64).copy()
        tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        want = np.broadcast_to(np.asarray(want, float), n.shape)
        flip = np.einsum('ij,ij->i', n, want) < 0
        F[flip] = F[flip][:, [0, 2, 1]]
        self.add(mat, V, F, c)

    def poly(self, mat, outer, holes=(), d=0.0, facing=1, c=(1, 1, 1), axis='d'):
        """Ebenes Polygon. axis 'd': Ebene u-v bei d; 'v': Ebene u-d bei v (outer in (u,d))."""
        V2, F = tri2d(outer, holes)
        if len(F) == 0:
            return
        if axis == 'd':
            V = np.c_[V2, np.full(len(V2), d)]; want = [0, 0, facing]
        elif axis == 'v':
            V = np.c_[V2[:, 0], np.full(len(V2), d), V2[:, 1]]; want = [0, facing, 0]
        else:  # 'u': Ebene v-d bei u
            V = np.c_[np.full(len(V2), d), V2[:, 0], V2[:, 1]]; want = [facing, 0, 0]
        self.oriented(mat, V, F, want, c)

    def box(self, mat, lo, hi, c=(1, 1, 1), skip=()):
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        V = np.array([[hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]] for i in range(8)])
        faces = {'-u': (0, 4, 6, 2), '+u': (1, 3, 7, 5), '-v': (0, 1, 5, 4), '+v': (2, 6, 7, 3), '-d': (0, 2, 3, 1), '+d': (4, 5, 7, 6)}
        F = []
        for k, (a, b, cc, d) in faces.items():
            if k not in skip:
                F += [[a, b, cc], [a, cc, d]]
        if F:
            self.add(mat, V, np.array(F), c)

    def bbox(self, mat, lo, hi, bev=0.006, c=(1, 1, 1), skip_back=True):
        """Quader mit gefasten Kanten (fangen Licht)."""
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        e = min(bev, *(0.45 * (hi - lo)))
        # 24 Ecken: je Wuerfel-Ecke 3 Punkte (eingerueckt auf je zwei Achsen)
        P = []; idx = {}
        for i in range(8):
            s = np.array([1 if i & 1 else -1, 1 if i & 2 else -1, 1 if i & 4 else -1])
            corner = np.where(s > 0, hi, lo)
            for ax in range(3):
                p = corner - s * e
                p[ax] = corner[ax]
                idx[(i, ax)] = len(P); P.append(p)
        P = np.array(P); F = []; W = []
        ctr = (lo + hi) / 2
        # Hauptflaechen
        for ax in range(3):
            for sgn in (-1, 1):
                if skip_back and ax == 2 and sgn == -1:
                    continue
                cs = [i for i in range(8) if ((i >> ax) & 1) == (1 if sgn > 0 else 0)]
                q = [idx[(i, ax)] for i in cs]
                # sortieren im Kreis
                pts = P[q]; c2 = pts.mean(0)
                o = [a for a in range(3) if a != ax]
                ang = np.arctan2(pts[:, o[1]] - c2[o[1]], pts[:, o[0]] - c2[o[0]])
                q = [q[k] for k in np.argsort(ang)]
                nrm = np.zeros(3); nrm[ax] = sgn
                F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]; W += [nrm, nrm]
        # Kantenfasen
        for i in range(8):
            for bit in range(3):
                j = i | (1 << bit)
                if j == i:
                    continue
                o = [a for a in range(3) if a != bit]
                for (a1, a2) in ((o[0], o[1]),):
                    q = [idx[(i, a1)], idx[(j, a1)], idx[(j, a2)], idx[(i, a2)]]
                    s = np.array([1 if i & 1 else -1, 1 if i & 2 else -1, 1 if i & 4 else -1]).astype(float)
                    nrm = s.copy(); nrm[bit] = 0
                    if skip_back and (nrm[2] < 0 and abs(nrm[0]) + abs(nrm[1]) == 0):
                        continue
                    F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]; W += [nrm, nrm]
        # Ecken-Dreiecke
        for i in range(8):
            s = np.array([1 if i & 1 else -1, 1 if i & 2 else -1, 1 if i & 4 else -1]).astype(float)
            F.append([idx[(i, 0)], idx[(i, 1)], idx[(i, 2)]]); W.append(s)
        self.oriented(mat, P, np.array(F), np.array(W), c)

    def cyl(self, mat, p0, p1, r, seg=6, c=(1, 1, 1), caps=True):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        ax = p1 - p0; L = np.linalg.norm(ax)
        if L < 1e-6:
            return
        ax /= L
        ref = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
        s = np.cross(ax, ref); s /= np.linalg.norm(s); u = np.cross(ax, s)
        a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        ring = np.cos(a)[:, None] * s * r + np.sin(a)[:, None] * u * r
        V = np.concatenate([p0 + ring, p1 + ring]); F = []; Wn = []
        for i in range(seg):
            j = (i + 1) % seg
            m = (ring[i] + ring[j]) / 2
            F += [[i, j, seg + j], [i, seg + j, seg + i]]; Wn += [m, m]
        if caps:
            for i in range(1, seg - 1):
                F.append([0, i, i + 1]); Wn.append(-ax)
                F.append([seg, seg + i, seg + i + 1]); Wn.append(ax)
        self.oriented(mat, V, np.array(F), np.array(Wn), c)

    def ring_rect(self, mat, x0, y0, x1, y1, prof, sides=(0, 1, 2, 3), c=(1, 1, 1), end_caps=False):
        """Profilierter Rechteck-Rahmen mit Gehrung. prof: [(s,d)] (s = Abstand nach innen vom Rand, d = Tiefe).
        Seiten 0 unten, 1 rechts, 2 oben, 3 links. Nicht gezogene Nachbarseiten -> gerade Enden (end_caps)."""
        prof = np.asarray(prof, float); J = len(prof)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)    # Richtung der Gehrung nach innen
        side_in = [np.array([0, 1.0]), np.array([-1.0, 0]), np.array([0, -1.0]), np.array([1.0, 0])]
        Vall = []; Fall = []; Wall = []; nv = 0
        # Profilnormalen in (s,d): nach aussen aus dem geschlossenen Profil
        closed = np.vstack([prof, [[prof[-1, 0], prof[:, 1].min() - 0.01], [prof[0, 0], prof[:, 1].min() - 0.01]]])
        pn = []
        for j in range(J - 1):
            a, b = prof[j], prof[j + 1]; e = b - a
            n = np.array([e[1], -e[0]]); n /= (np.linalg.norm(n) + 1e-12)
            m = (a + b) / 2
            if _pip(m + n * 1e-4, closed):
                n = -n
            pn.append(n)
        for k in sides:
            ca, cb = corners[k], corners[(k + 1) % 4]
            ia, ib = inward[k], inward[(k + 1) % 4]
            prev_on = ((k - 1) % 4) in sides; next_on = ((k + 1) % 4) in sides
            si = side_in[k]
            A = np.array([ca + (ia if prev_on else si) * s for s in prof[:, 0]])
            B = np.array([cb + (ib if next_on else si) * s for s in prof[:, 0]])
            VA = np.c_[A, prof[:, 1]]; VB = np.c_[B, prof[:, 1]]
            V = np.concatenate([VA, VB])
            for j in range(J - 1):
                F = [[j, J + j, J + j + 1], [j, J + j + 1, j + 1]]
                n3 = np.array([si[0] * pn[j][0], si[1] * pn[j][0], pn[j][1]])
                Fall += [[f + nv for f in tri] for tri in F]; Wall += [n3, n3]
            Vall.append(V); nv += len(V)
            if end_caps:
                for end, P_, nxt in ((0, VA, prev_on), (1, VB, next_on)):
                    if nxt:
                        continue
                    V2, Fc = tri2d(prof)
                    if len(Fc) == 0:
                        continue
                    base = ca if end == 0 else cb
                    Pc = np.c_[np.array([base + si * s for s in V2[:, 0]]), V2[:, 1]]
                    along = (cb - ca) / np.linalg.norm(cb - ca)
                    wnt = np.r_[-along if end == 0 else along, 0]
                    self.oriented(mat, Pc, Fc, wnt, c)
        if Vall:
            self.oriented(mat, np.concatenate(Vall), np.array(Fall), np.array(Wall), c)

    def bar(self, mat, cross, axis, a, b, at=(0.0,), c=(1, 1, 1), caps=False):
        """Stab mit Querschnitt cross [(x,d)] (offener Zug, Rueckseite fehlt) entlang axis ('v' oder 'u') von a bis b.
        at: Position auf der anderen Achse."""
        cross = np.asarray(cross, float); J = len(cross)
        closed = np.vstack([cross, [[cross[-1, 0], cross[:, 1].min() - 0.01], [cross[0, 0], cross[:, 1].min() - 0.01]]])
        V = []
        for x, d in cross:
            if axis == 'v':
                V += [[at[0] + x, a, d], [at[0] + x, b, d]]
            else:
                V += [[a, at[0] + x, d], [b, at[0] + x, d]]
        V = np.array(V); F = []; Wn = []
        for k in range(J - 1):
            i0, i1 = 2 * k, 2 * (k + 1)
            F += [[i0, i1, i1 + 1], [i0, i1 + 1, i0 + 1]]
            e = cross[k + 1] - cross[k]; n = np.array([e[1], -e[0]]); n /= (np.linalg.norm(n) + 1e-12)
            m = (cross[k] + cross[k + 1]) / 2
            if _pip(m + n * 1e-4, closed):
                n = -n
            n3 = np.array([n[0], 0, n[1]]) if axis == 'v' else np.array([0, n[0], n[1]])
            Wn += [n3, n3]
        self.oriented(mat, V, np.array(F), np.array(Wn), c)

    def extrude_u(self, mat, cross, u0, u1, c=(1, 1, 1), caps=True):
        """Geschlossener Querschnitt cross [(d,v)] entlang u extrudiert (Baenke, Gesimse) mit Stirnseiten."""
        cross = np.asarray(cross, float); J = len(cross)
        V = np.concatenate([np.c_[np.full(J, u0), cross[:, 1], cross[:, 0]], np.c_[np.full(J, u1), cross[:, 1], cross[:, 0]]])
        F = []; Wn = []
        for k in range(J):
            k2 = (k + 1) % J
            F += [[k, k2, J + k2], [k, J + k2, J + k]]
            e = cross[k2] - cross[k]; n = np.array([e[1], -e[0]])
            m = (cross[k] + cross[k2]) / 2
            if _pip(m + n / (np.linalg.norm(n) + 1e-12) * 1e-4, cross):
                n = -n
            n3 = np.array([0, n[1], n[0]]); Wn += [n3, n3]
        self.oriented(mat, V, np.array(F), np.array(Wn), c)
        if caps:
            V2, Fc = tri2d(cross)
            for uu, s in ((u0, -1), (u1, 1)):
                self.oriented(mat, np.c_[np.full(len(V2), uu), V2[:, 1], V2[:, 0]], Fc, [s, 0, 0], c)

    def freeze(self):
        out = {}
        for mat, (Vs, Fs, Cs, n) in self.parts.items():
            out[mat] = (np.concatenate(Vs).astype(np.float32), np.concatenate(Fs).astype(np.uint32), np.concatenate(Cs).astype(np.float32))
        self.mesh = out
        self.ntri = sum(len(F) for _, F, _ in out.values())
        return self


def _pip(p, poly):
    x, y = p; inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


# ======================================================================== Profile
FRAME = [(0.0, -0.07), (0.0, -0.003), (0.004, 0.0), (0.044, 0.0), (0.052, -0.008), (0.052, -0.02), (0.052, -0.07)]
SASH = [(-0.004, -0.065), (-0.004, -0.016), (0.0, -0.012), (0.046, -0.012), (0.052, -0.018), (0.052, -0.026),
        (0.064, -0.038), (0.064, -0.046), (0.064, -0.06)]
MULLION = [(-0.052, -0.07), (-0.052, -0.008), (-0.044, 0.0), (0.044, 0.0), (0.052, -0.008), (0.052, -0.07)]
SPROSSE = [(-0.016, -0.046), (-0.016, -0.03), (-0.01, -0.022), (0.01, -0.022), (0.016, -0.03), (0.016, -0.046)]
ALU_FRAME = [(0.0, -0.07), (0.0, 0.0), (0.05, 0.0), (0.05, -0.006), (0.056, -0.006), (0.056, -0.03), (0.056, -0.07)]
ALU_SASH = [(-0.004, -0.06), (-0.004, -0.008), (0.05, -0.008), (0.05, -0.024), (0.058, -0.03), (0.058, -0.046), (0.058, -0.06)]
CW_CAP = [(0.0, -0.06), (0.0, 0.0), (0.03, 0.0), (0.03, -0.012), (0.034, -0.012), (0.034, -0.03), (0.034, -0.06)]
WOOD_FRAME = [(0.0, -0.08), (0.0, -0.004), (0.006, 0.0), (0.06, 0.0), (0.07, -0.01), (0.07, -0.024), (0.07, -0.08)]


def _fields_T(w, h, ht):
    a = 0.052
    return [(-w / 2 + a, a, -a, ht - a), (a, a, w / 2 - a, ht - a), (-w / 2 + a, ht + a, w / 2 - a, h - a)]


def _handle(P, u, v, d, side=1, c=(0.86, 0.86, 0.84)):
    """Fenster-Olive auf der Raumseite des Fluegels (durch das Glas sichtbar): Rosette, Hals, Griff nach unten."""
    d = -0.066
    P.box('trim', [u - 0.014, v - 0.035, d - 0.007], [u + 0.014, v + 0.035, d], c=c, skip=('+d',))
    P.cyl('trim', [u, v, d - 0.007], [u, v, d - 0.026], 0.007, seg=4, c=c, caps=False)
    P.box('trim', [u - 0.01, v - 0.12, d - 0.034], [u + 0.01, v + 0.012, d - 0.022], c=c, skip=())


def _hinge(P, u, v, d, c=(0.7, 0.7, 0.7)):
    P.cyl('trim', [u, v - 0.035, d], [u, v + 0.035, d], 0.008, seg=4, c=c, caps=False)


def _sash(P, x0, y0, x1, y1, sprossen=None, handle=None, hinge=None, prof=SASH):
    P.ring_rect('trim', x0, y0, x1, y1, prof)
    gx0, gy0, gx1, gy1 = x0 + 0.064, y0 + 0.064, x1 - 0.064, y1 - 0.064
    if sprossen:
        nh, nv = sprossen
        for i in range(1, nh + 1):
            y = gy0 + (gy1 - gy0) * i / (nh + 1)
            P.bar('trim', SPROSSE, 'u', gx0 - 0.005, gx1 + 0.005, at=(y,))
        for i in range(1, nv + 1):
            x = gx0 + (gx1 - gx0) * i / (nv + 1)
            P.bar('trim', SPROSSE, 'v', gy0 - 0.005, gy1 + 0.005, at=(x,))
    if handle is not None:
        hu = x0 + 0.03 if handle == 'left' else x1 - 0.03
        _handle(P, hu, (y0 + y1) / 2 if (y1 - y0) < 1.6 else y0 + 1.05, -0.012)
    if hinge is not None:
        hu = x0 - 0.002 if hinge == 'left' else x1 + 0.002
        _hinge(P, hu, y0 + 0.12, -0.012); _hinge(P, hu, y1 - 0.12, -0.012)


def _board(P, w, depth=0.26, c=(0.95, 0.95, 0.93)):
    """Innenfensterbank."""
    P.bbox('trim', [-w / 2 - 0.06, -0.04, -0.07 - depth], [w / 2 + 0.06, -0.005, -0.065], bev=0.006, c=c)


# ======================================================================== Fenster
def window_proto(kind):
    if kind in PROTOS:
        return PROTOS[kind]
    if kind.startswith('alt'):
        w0, h0 = 1.2, 2.1
    elif kind.startswith('fw'):
        w0, h0 = 0.95, 1.35
    elif kind.startswith('mod') or kind.startswith('rib'):
        w0, h0 = 1.45, 1.9
    elif kind.startswith('cw'):
        w0, h0 = 1.35, 2.7
    elif kind.startswith('dorm'):
        w0, h0 = 0.9, 1.15
    elif kind == 'park':
        w0, h0 = 4.4, 1.5
    else:
        w0, h0 = 1.4, 1.5
    P = Proto(kind, 'frame', w0, h0)
    w, h = w0, h0
    if kind in ('alt_T', 'alt_T_spr'):
        ht = h * 0.72
        P.ring_rect('trim', -w / 2, 0, w / 2, h, FRAME)
        P.bar('trim', MULLION, 'u', -w / 2 + 0.05, w / 2 - 0.05, at=(ht,))
        P.bar('trim', MULLION, 'v', 0.05, ht - 0.05, at=(0.0,))
        f = _fields_T(w, h, ht)
        spr = (2, 0) if kind.endswith('spr') else None
        _sash(P, *f[0], sprossen=spr, handle='right', hinge='left')
        _sash(P, *f[1], sprossen=spr, handle='left', hinge='right')
        _sash(P, *f[2], sprossen=(0, 2) if spr else None)
        _board(P, w)
    elif kind in ('alt_K', 'alt_K_spr'):
        ht = h * 0.7
        P.ring_rect('trim', -w / 2, 0, w / 2, h, FRAME)
        P.bar('trim', MULLION, 'u', -w / 2 + 0.05, w / 2 - 0.05, at=(ht,))
        P.bar('trim', MULLION, 'v', 0.05, h - 0.05, at=(0.0,))
        a = 0.052
        spr = kind.endswith('spr')
        _sash(P, -w / 2 + a, a, -a, ht - a, sprossen=(2, 0) if spr else None, handle='right', hinge='left')
        _sash(P, a, a, w / 2 - a, ht - a, sprossen=(2, 0) if spr else None, handle='left', hinge='right')
        _sash(P, -w / 2 + a, ht + a, -a, h - a, sprossen=(0, 1) if spr else None)
        _sash(P, a, ht + a, w / 2 - a, h - a, sprossen=(0, 1) if spr else None)
        _board(P, w)
    elif kind in ('nk_2', 'nk_3', 'dorm_2'):
        P.ring_rect('trim', -w / 2, 0, w / 2, h, FRAME)
        n = 3 if kind == 'nk_3' else 2
        a = 0.052
        xs = [-w / 2 + a + (w - 2 * a) * i / n for i in range(n + 1)]
        for i in range(1, n):
            P.bar('trim', MULLION, 'v', 0.05, h - 0.05, at=(xs[i],))
        for i in range(n):
            x0 = xs[i] + (a if i > 0 else 0); x1 = xs[i + 1] - (a if i < n - 1 else 0)
            _sash(P, x0, a, x1, h - a, handle=('right' if i == 0 else 'left') if i < 2 else None, hinge='left' if i == 0 else 'right')
        _board(P, w)
    elif kind in ('mod_1', 'mod_2', 'rib_1'):
        P.ring_rect('trim', -w / 2, 0, w / 2, h, ALU_FRAME)
        a = 0.056
        if kind == 'mod_2':
            P.bar('trim', [(-0.056, -0.07), (-0.056, 0), (0.056, 0), (0.056, -0.07)], 'v', 0.05, h - 0.05, at=(0.0,))
            _sash(P, -w / 2 + a, a, -a, h - a, handle='right', hinge='left', prof=ALU_SASH)
            # fester Teil rechts: Glashalteleiste
            P.ring_rect('trim', a, a, w / 2 - a, h - a, [(-0.004, -0.06), (-0.004, -0.03), (0.012, -0.036), (0.012, -0.046)])
        else:
            ht = h * 0.78 if kind == 'rib_1' else 0
            if ht:
                P.bar('trim', [(-0.056, -0.07), (-0.056, 0), (0.056, 0), (0.056, -0.07)], 'u', -w / 2 + 0.05, w / 2 - 0.05, at=(ht,))
                _sash(P, -w / 2 + a, a, w / 2 - a, ht - a, handle='right', hinge='left', prof=ALU_SASH)
                P.ring_rect('trim', -w / 2 + a, ht + a, w / 2 - a, h - a, [(-0.004, -0.06), (-0.004, -0.03), (0.012, -0.036), (0.012, -0.046)])
            else:
                _sash(P, -w / 2 + a, a, w / 2 - a, h - a, handle='right', hinge='left', prof=ALU_SASH)
        _board(P, w, c=(0.9, 0.9, 0.9))
    elif kind == 'cw_1':
        P.ring_rect('trim', -w / 2, 0, w / 2, h, CW_CAP)
        P.bar('trim', [(-0.03, -0.06), (-0.03, -0.012), (0.03, -0.012), (0.03, -0.06)], 'u', -w / 2 + 0.03, w / 2 - 0.03, at=(0.9,))
    elif kind in ('fw_X', 'fw_X_spr', 'fw_S'):
        P.ring_rect('trim', -w / 2, 0, w / 2, h, WOOD_FRAME)
        a = 0.07; ht = h * 0.62
        P.bar('trim', MULLION, 'u', -w / 2 + 0.06, w / 2 - 0.06, at=(ht,))
        P.bar('trim', MULLION, 'v', 0.06, h - 0.06, at=(0.0,))
        spr = kind.endswith('spr')
        lo = (1, 1) if kind == 'fw_S' else ((1, 0) if spr else None)
        hi = (0, 1) if kind == 'fw_S' else None
        _sash(P, -w / 2 + a, a, -0.052, ht - 0.052, sprossen=lo, handle='right', hinge='left')
        _sash(P, 0.052, a, w / 2 - a, ht - 0.052, sprossen=lo, handle='left', hinge='right')
        _sash(P, -w / 2 + a, ht + 0.052, -0.052, h - a, sprossen=hi)
        _sash(P, 0.052, ht + 0.052, w / 2 - a, h - a, sprossen=hi)
        _board(P, w)
    elif kind == 'park':
        P.origin = 'frame'
        for y in (0.45, 0.95):
            P.cyl('metal', [-w / 2, y, 0.03], [w / 2, y, 0.03], 0.024, seg=8)
        n = max(2, int(w / 1.4))
        for i in range(n + 1):
            x = -w / 2 + 0.05 + (w - 0.1) * i / n
            P.bbox('metal', [x - 0.03, 0.0, 0.0], [x + 0.03, 1.0, 0.06], bev=0.004)
    PROTOS[kind] = P.freeze()
    return P


def glass_proto():
    if 'glass' in PROTOS:
        return PROTOS['glass']
    P = Proto('glass', 'frame', 1.0, 1.0)
    P.add('glass', [[-0.5, 0, -0.046], [0.5, 0, -0.046], [0.5, 1, -0.046], [-0.5, 1, -0.046]], [[0, 1, 2], [0, 2, 3]])
    PROTOS['glass'] = P.freeze()
    return P


# ======================================================================== Innenansichten (hinter dem Glas)
def interior_proto(kind):
    key = 'int_' + kind
    if key in PROTOS:
        return PROTOS[key]
    P = Proto(key, 'frame', 1.0, 1.0)
    w, h = 1.0, 1.0
    if kind == 'curtains':
        for side in (-1, 1):
            cw = 0.24; folds = 6
            xs = np.linspace(0, cw, 2 * folds + 1)
            V = []; F = []
            for i, x in enumerate(xs):
                d = -0.12 - (0.025 if i % 2 else 0.0)
                u = side * (w / 2 - x) - side * 0.0
                V += [[u, -0.02, d], [u, h + 0.04, d]]
            for i in range(len(xs) - 1):
                F += [[2 * i, 2 * i + 2, 2 * i + 3], [2 * i, 2 * i + 3, 2 * i + 1]]
            P.oriented('interior', V, F, [0, 0, 1])
        P.box('interior', [-w / 2 - 0.02, h - 0.03, -0.16], [w / 2 + 0.02, h + 0.02, -0.1], skip=('-d',))    # Schiene
    elif kind == 'net':
        folds = 10; xs = np.linspace(-w / 2, w / 2, 2 * folds + 1)
        V = []; F = []
        for i, x in enumerate(xs):
            d = -0.1 - (0.012 if i % 2 else 0.0)
            V += [[x, 0.02, d], [x, h, d]]
        for i in range(len(xs) - 1):
            F += [[2 * i, 2 * i + 2, 2 * i + 3], [2 * i, 2 * i + 3, 2 * i + 1]]
        P.oriented('interior', V, F, [0, 0, 1], c=(1, 1, 1))
    elif kind in ('blind', 'blind_low'):
        drop = 0.5 if kind == 'blind' else 0.85
        n = int(drop / 0.04)
        for i in range(n):
            y = h - 0.03 - i * drop / n
            P.oriented('interior', [[-w / 2, y, -0.1], [w / 2, y, -0.1], [w / 2, y - 0.03, -0.112], [-w / 2, y - 0.03, -0.112]],
                       [[0, 2, 1], [0, 3, 2]], [0, 0.4, 1])
        P.box('interior', [-w / 2, h - 0.06, -0.13], [w / 2, h, -0.09], skip=('-d',))
        P.box('interior', [-w / 2, h - drop - 0.04, -0.12], [w / 2, h - drop, -0.1], skip=('-d',))
    elif kind == 'roller':
        drop = 0.4
        P.add('interior', [[-w / 2, h - drop, -0.1], [w / 2, h - drop, -0.1], [w / 2, h, -0.1], [-w / 2, h, -0.1]], [[0, 1, 2], [0, 2, 3]])
        P.box('interior', [-w / 2, h - drop - 0.03, -0.11], [w / 2, h - drop, -0.09], skip=('-d',))
    elif kind == 'shop':
        # helle Rueckwand + Regale + Auslage
        P.add('interior', [[-w / 2, 0, -0.9], [w / 2, 0, -0.9], [w / 2, h, -0.9], [-w / 2, h, -0.9]], [[0, 1, 2], [0, 2, 3]], c=(1, 1, 1))
        for y in (0.35, 0.65):
            P.box('interior', [-w / 2, y * h, -0.9], [w / 2, y * h + 0.015, -0.6], c=(0.75, 0.72, 0.68), skip=('-d',))
        P.box('interior', [-0.3, 0, -0.6], [0.3, 0.28, -0.25], c=(0.55, 0.5, 0.48), skip=('-d',))
        P.box('interior', [-w / 2, h - 0.04, -0.9], [w / 2, h, -0.1], c=(1.0, 0.98, 0.92), skip=('-d', '+v'))
    PROTOS[key] = P.freeze()
    return P


# ======================================================================== Aussenbauteile (Ursprung Wand)
def sill_proto(kind):
    key = 'sill_' + kind
    if key in PROTOS:
        return PROTOS[key]
    P = Proto(key, 'wall', 1.3, 1.0)
    L = 1.3
    if kind == 'stone':
        cross = [(-0.36, -0.065), (-0.36, 0.022), (0.0, 0.004), (0.04, 0.0), (0.055, -0.01), (0.055, -0.05), (0.045, -0.065),
                 (0.032, -0.065), (0.032, -0.073), (0.024, -0.073), (0.024, -0.065)]
        P.extrude_u('trim', cross, -L / 2 - 0.05, L / 2 + 0.05)
    else:  # metal
        cross = [(-0.36, -0.004), (-0.36, 0.0), (0.03, -0.014), (0.042, -0.02), (0.042, -0.052), (0.036, -0.052), (0.036, -0.024),
                 (0.026, -0.018)]
        P.extrude_u('trim', cross, -L / 2 - 0.02, L / 2 + 0.02)
        for s in (-1, 1):
            u = s * (L / 2 + 0.02)
            P.box('trim', [min(u, u - s * 0.012), -0.052, -0.36], [max(u, u - s * 0.012), 0.035, 0.045], skip=('-d',))
    PROTOS[key] = P.freeze()
    return P


FASCHE = [(0.0, 0.0), (0.0, 0.028), (0.012, 0.044), (0.13, 0.044), (0.15, 0.03), (0.16, 0.0)]


def surround_proto(kind):
    """Faschen/Gewaende (Ursprung Wand, Nennmass 1.2 x 2.1)."""
    key = 'sur_' + kind
    if key in PROTOS:
        return PROTOS[key]
    w, h = 1.2, 2.1
    P = Proto(key, 'wall', w, h)
    if kind in ('fasche', 'fasche_key', 'fasche_cornice', 'fasche_pediment'):
        # U-foermige Fasche um die Oeffnung (Profil nach aussen), unten auf der Bank
        prof = [(-s, d) for s, d in FASCHE]          # s negativ = nach aussen vom Rand
        P.ring_rect('surround', -w / 2, -0.07, w / 2, h, prof, sides=(1, 2, 3), end_caps=True)
        # Bruestungsfeld unter der Bank
        P.bbox('surround', [-w / 2 + 0.02, -0.66, 0], [w / 2 - 0.02, -0.14, 0.022], bev=0.008)
        P.bbox('surround', [-w / 2 + 0.1, -0.58, 0.02], [w / 2 - 0.1, -0.22, 0.032], bev=0.006)
        if kind in ('fasche_key', 'fasche_cornice', 'fasche_pediment'):
            # Schlussstein (trapezfoermig, gefast)
            ks = [(-0.08, h - 0.04), (0.08, h - 0.04), (0.11, h + 0.2), (-0.11, h + 0.2)]
            for dd in (0.075,):
                P.poly('surround', ks, d=dd, facing=1)
            for i in range(4):
                a, b = ks[i], ks[(i + 1) % 4]
                e = np.subtract(b, a); n = np.array([e[1], -e[0]])
                ctr = np.mean(ks, 0)
                if np.dot(n, np.subtract((np.add(a, b)) / 2, ctr)) < 0: n = -n
                P.oriented('surround', [[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], 0.075], [a[0], a[1], 0.075]],
                           [[0, 1, 2], [0, 2, 3]], [n[0], n[1], 0])
        if kind in ('fasche_cornice', 'fasche_pediment'):
            y0 = h + 0.2
            cross = [(0.0, y0), (0.07, y0), (0.07, y0 + 0.035), (0.1, y0 + 0.06), (0.15, y0 + 0.08), (0.17, y0 + 0.1), (0.17, y0 + 0.16),
                     (0.0, y0 + 0.16)]
            P.extrude_u('surround', cross, -w / 2 - 0.3, w / 2 + 0.3)
            P.bbox('surround', [-w / 2 - 0.02, h + 0.04, 0], [w / 2 + 0.02, y0, 0.05], bev=0.006)       # Fries
            for s in (-1, 1):                                                                            # Konsolen
                u = s * (w / 2 + 0.2)
                P.bbox('surround', [u - 0.07, h - 0.28, 0], [u + 0.07, y0, 0.12], bev=0.01)
                P.bbox('surround', [u - 0.05, h - 0.4, 0], [u + 0.05, h - 0.28, 0.08], bev=0.008)
            if kind == 'fasche_pediment':
                yb = y0 + 0.16; ht = 0.38; xa = w / 2 + 0.3
                tri = [(-xa, yb), (xa, yb), (0, yb + ht)]
                P.poly('surround', tri, d=0.12, facing=1)
                for (p, q) in ((tri[1], tri[2]), (tri[2], tri[0])):
                    e = np.subtract(q, p); n = np.array([e[1], -e[0]])
                    if n[1] < 0: n = -n
                    P.oriented('surround', [[p[0], p[1], 0], [q[0], q[1], 0], [q[0], q[1], 0.12], [p[0], p[1], 0.12]],
                               [[0, 1, 2], [0, 2, 3]], [n[0], n[1], 0])
                    # Gesims auf den Schraegen
                    m = np.array([e[0], e[1]]) / np.linalg.norm(e)
                    off = np.array([-m[1], m[0]]) * (1 if m[0] > 0 else -1) * 0.0
                P.box('surround', [-xa - 0.02, yb, 0.0], [xa + 0.02, yb + 0.03, 0.15], skip=('-d',))
    elif kind == 'thin':
        prof = [(-s, d) for s, d in [(0.0, 0.0), (0.0, 0.018), (0.008, 0.026), (0.09, 0.026), (0.1, 0.0)]]
        P.ring_rect('surround', -w / 2, -0.07, w / 2, h, prof)
    elif kind == 'lintel':
        P.bbox('surround', [-w / 2 - 0.12, h, 0], [w / 2 + 0.12, h + 0.24, 0.03], bev=0.008)
    PROTOS[key] = P.freeze()
    return P


def rollo_proto(frac):
    """Rollladen: Fuehrungsschienen in der Laibung + teilweise heruntergelassener Panzer (Ursprung Rahmen, 1.4 x 1.5)."""
    key = 'rollo_%d' % int(frac * 100)
    if key in PROTOS:
        return PROTOS[key]
    w, h = 1.4, 1.5
    P = Proto(key, 'frame', w, h)
    rail = [(-0.025, 0.0), (-0.025, 0.05), (-0.006, 0.05), (-0.006, 0.042), (-0.018, 0.042), (-0.018, 0.0)]
    for s in (-1, 1):
        u = s * (w / 2 - 0.003)
        cr = [(s * -x, d) for x, d in rail]
        P.bar('metal', [(u + x, d) for x, d in cr], 'v', 0.0, h, at=(0.0,))
    if frac > 0:
        drop = h * frac
        n = max(2, int(drop / 0.045))
        V = []; F = []
        for i in range(n + 1):
            y = h - i * drop / n
            V += [[-w / 2 + 0.01, y, 0.03], [w / 2 - 0.01, y, 0.03]]
            if i < n:
                ym = y - drop / n * 0.85
                V += [[-w / 2 + 0.01, ym, 0.036], [w / 2 - 0.01, ym, 0.036]]
        V = np.array(V)
        k = 0; F = []
        rows = len(V) // 2
        for r in range(rows - 1):
            a, b = 2 * r, 2 * r + 2
            F += [[a, b + 1, a + 1], [a, b, b + 1]]
        P.oriented('metal', V, F, [0, 0, 1])
        y = h - drop
        P.bbox('metal', [-w / 2 + 0.01, y - 0.03, 0.022], [w / 2 - 0.01, y + 0.005, 0.045], bev=0.004)
    PROTOS[key] = P.freeze()
    return P


def balcony_proto():
    """Franzoesischer Balkon: Pfosten, Handlauf, Fussriegel, Staebe, Zierband (Ursprung Wand, Nennbreite 1.3)."""
    if 'fr_bal' in PROTOS:
        return PROTOS['fr_bal']
    w = 1.3
    P = Proto('fr_bal', 'wall', w, 1.0)
    x0, x1 = -w / 2 - 0.06, w / 2 + 0.06; d = 0.1; ht = 0.95
    for x in (x0, x1):
        P.bbox('metal', [x - 0.022, 0.0, 0.0], [x + 0.022, ht + 0.02, d + 0.025], bev=0.005)
        P.cyl('metal', [x, ht + 0.02, d], [x, ht + 0.06, d], 0.02, seg=8)
    P.extrude_u('metal', [(d - 0.03, ht - 0.01), (d + 0.03, ht - 0.01), (d + 0.03, ht + 0.02), (d + 0.015, ht + 0.035),
                          (d - 0.015, ht + 0.035), (d - 0.03, ht + 0.02)], x0, x1)
    P.box('metal', [x0, 0.07, d - 0.012], [x1, 0.1, d + 0.012], skip=())
    P.box('metal', [x0, ht - 0.22, d - 0.008], [x1, ht - 0.2, d + 0.008], skip=())
    P.box('metal', [x0, 0.24, d - 0.008], [x1, 0.26, d + 0.008], skip=())
    n = int((x1 - x0) / 0.11)
    for i in range(1, n):
        x = x0 + (x1 - x0) * i / n
        P.cyl('metal', [x, 0.1, d], [x, ht - 0.01, d], 0.009, seg=6, caps=False)
    # Zierkreise zwischen den Zierbaendern (achteckig angenaehert)
    for i in range(1, n, 2):
        x = x0 + (x1 - x0) * (i + 0.5) / n
        P.ring_rect('metal', x - 0.035, ht - 0.19, x + 0.035, ht - 0.12, [(0.0, d - 0.006), (0.0, d + 0.006), (0.008, d + 0.006), (0.008, d - 0.006)])
    PROTOS['fr_bal'] = P.freeze()
    return P


# ======================================================================== Tueren / Schaufenster (Ursprung Rahmen)
def shop_proto(door):
    key = 'shop_' + (door or 'fix')
    if key in PROTOS:
        return PROTOS[key]
    w, h = 2.6, 3.1
    P = Proto(key, 'frame', w, h)
    prof = [(0.0, -0.08), (0.0, 0.0), (0.06, 0.0), (0.06, -0.008), (0.068, -0.012), (0.068, -0.04), (0.068, -0.08)]
    P.ring_rect('trim', -w / 2, 0, w / 2, h, prof)
    ht = h - 0.55
    mcross = [(-0.04, -0.08), (-0.04, 0.0), (0.04, 0.0), (0.04, -0.08)]
    P.bar('trim', [(-0.045, -0.08), (-0.045, 0.0), (0.045, 0.0), (0.045, -0.08)], 'u', -w / 2 + 0.06, w / 2 - 0.06, at=(ht,))
    # Sockelprofil (Tritt-/Stossblech)
    P.bbox('trim', [-w / 2 + 0.06, 0.0, -0.06], [w / 2 - 0.06, 0.16, 0.004], bev=0.004)
    if door:
        dw = 1.05
        dx0 = {'left': -w / 2 + 0.068, 'right': w / 2 - 0.068 - dw, 'center': -dw / 2}[door]
        dx1 = dx0 + dw
        for x in (dx0, dx1):
            if -w / 2 + 0.1 < x < w / 2 - 0.1:
                P.bar('trim', mcross, 'v', 0.0, ht - 0.04, at=(x,))
        # Tuerfluegel: breite Rahmenprofile, unten Sockel, Glas dahinter
        lx0, lx1 = dx0 + 0.045, dx1 - 0.045
        P.ring_rect('trim', lx0, 0.005, lx1, ht - 0.05, [(0.0, -0.05), (0.0, 0.012), (0.01, 0.018), (0.085, 0.018), (0.095, 0.006), (0.095, -0.03)])
        P.bbox('trim', [lx0 + 0.09, 0.005, -0.03], [lx1 - 0.09, 0.26, 0.016], bev=0.004, c=(0.8, 0.8, 0.8))     # Trittblech
        # Stossstange mit Abstandhaltern
        hx = lx1 - 0.16 if door != 'right' else lx0 + 0.16
        P.cyl('trim', [hx, 0.75, 0.09], [hx, 1.65, 0.09], 0.016, seg=8, c=(0.85, 0.85, 0.85))
        for y in (0.82, 1.58):
            P.cyl('trim', [hx, y, 0.018], [hx, y, 0.09], 0.01, seg=6, c=(0.85, 0.85, 0.85))
        # Tuerschliesser
        P.bbox('trim', [lx0 + 0.15, ht - 0.13, 0.018], [lx0 + 0.55, ht - 0.07, 0.07], bev=0.006, c=(0.75, 0.75, 0.75))
        P.bbox('trim', [lx0 + 0.45, ht - 0.11, 0.05], [lx0 + 0.75, ht - 0.085, 0.07], bev=0.004, c=(0.75, 0.75, 0.75))
        # Schloss
        P.bbox('trim', [hx - 0.02 if door == 'right' else hx - 0.03, 1.0, 0.018], [hx + 0.03, 1.08, 0.03], bev=0.003, c=(0.8, 0.8, 0.8))
        segs = [(-w / 2 + 0.068, dx0), (dx1, w / 2 - 0.068)]
    else:
        segs = [(-w / 2 + 0.068, w / 2 - 0.068)]
    for a, b in segs:
        L = b - a
        if L < 0.6:
            continue
        nm = int(L / 1.3)
        for i in range(1, nm + 1):
            P.bar('trim', mcross, 'v', 0.0, ht - 0.04, at=(a + L * i / (nm + 1),))
    # Oberlicht-Pfosten
    for i in range(1, 3):
        P.bar('trim', [(-0.03, -0.06), (-0.03, -0.01), (0.03, -0.01), (0.03, -0.06)], 'v', ht + 0.04, h - 0.06, at=(-w / 2 + w * i / 3,))
    PROTOS[key] = P.freeze()
    return P


def door_proto(kind, r):
    """Hauseingaenge. kind 'alt2' (zweifluegelige Holztuer mit Oberlicht), 'nk' (Holz/Glas), 'glass' (Alu-Glastuer).
    r = Laibungstiefe (fuer das Klingeltableau auf der Wandflaeche)."""
    key = 'door_%s_%d' % (kind, int(round(r * 100)))
    if key in PROTOS:
        return PROTOS[key]
    if kind == 'alt2':
        w, h = 1.5, 3.0
    elif kind == 'nk':
        w, h = 1.3, 2.45
    else:
        w, h = 1.6, 2.6
    P = Proto(key, 'frame', w, h)
    c_brass = (0.85, 0.75, 0.45); c_steel = (0.82, 0.82, 0.8)
    if kind in ('alt2', 'nk'):
        fan = 0.62 if kind == 'alt2' else 0.0
        hd = h - fan
        P.ring_rect('trim', -w / 2, 0, w / 2, h, WOOD_FRAME)
        if fan:
            P.bar('trim', [(-0.07, -0.08), (-0.07, -0.01), (-0.06, 0.0), (0.06, 0.0), (0.07, -0.01), (0.07, -0.08)], 'u', -w / 2 + 0.06, w / 2 - 0.06, at=(hd,))
            for i in range(1, 4):
                P.bar('trim', SPROSSE, 'v', hd + 0.07, h - 0.07, at=(-w / 2 + w * i / 4,))
            P.add('glass', [[-w / 2 + 0.07, hd + 0.07, -0.046], [w / 2 - 0.07, hd + 0.07, -0.046], [w / 2 - 0.07, h - 0.07, -0.046],
                            [-w / 2 + 0.07, h - 0.07, -0.046]], [[0, 1, 2], [0, 2, 3]])
        leaves = 2 if kind == 'alt2' else 1
        x_in0, x_in1 = -w / 2 + 0.07, w / 2 - 0.07
        if kind == 'nk':
            # Seitenteil (Glas) rechts
            x_in1 = w / 2 - 0.07 - 0.28
            P.bar('trim', MULLION, 'v', 0.0, hd - 0.07, at=(x_in1 + 0.052,))
            P.ring_rect('trim', x_in1 + 0.104, 0.07, w / 2 - 0.07, hd - 0.07, [(-0.004, -0.06), (-0.004, -0.02), (0.02, -0.03), (0.02, -0.046)])
            P.add('glass', [[x_in1 + 0.11, 0.07, -0.046], [w / 2 - 0.07, 0.07, -0.046], [w / 2 - 0.07, hd - 0.07, -0.046], [x_in1 + 0.11, hd - 0.07, -0.046]],
                  [[0, 1, 2], [0, 2, 3]])
        lw = (x_in1 - x_in0) / leaves
        top = hd - (0.07 if fan else 0.07)
        for i in range(leaves):
            a = x_in0 + i * lw; b = a + lw
            # Blatt (geschlossen) auf -0.03
            P.add('trim', [[a, 0.0, -0.03], [b, 0.0, -0.03], [b, top, -0.03], [a, top, -0.03]], [[0, 1, 2], [0, 2, 3]])
            # Rahmenstuecke (Friese) als gefaste Leisten, Fuellungen
            P.bbox('trim', [a, 0.0, -0.03], [a + 0.11, top, 0.0], bev=0.008)
            P.bbox('trim', [b - 0.11, 0.0, -0.03], [b, top, 0.0], bev=0.008)
            P.bbox('trim', [a + 0.11, 0.0, -0.03], [b - 0.11, 0.24, 0.0], bev=0.008)
            P.bbox('trim', [a + 0.11, top - 0.13, -0.03], [b - 0.11, top, 0.0], bev=0.008)
            ym = top * 0.52
            P.bbox('trim', [a + 0.11, ym - 0.08, -0.03], [b - 0.11, ym + 0.08, 0.0], bev=0.008)
            # untere Fuellung: erhaben mit Fase
            P.bbox('trim', [a + 0.15, 0.28, -0.03], [b - 0.15, ym - 0.12, -0.006], bev=0.02)
            # obere Fuellung: Glaseinsatz mit Profilleiste
            gx0, gx1, gy0, gy1 = a + 0.14, b - 0.14, ym + 0.11, top - 0.16
            P.ring_rect('trim', gx0, gy0, gx1, gy1, [(-0.03, -0.0), (-0.03, 0.006), (-0.01, 0.012), (0.0, 0.004), (0.006, -0.02)])
            P.add('glass', [[gx0, gy0, -0.02], [gx1, gy0, -0.02], [gx1, gy1, -0.02], [gx0, gy1, -0.02]], [[0, 1, 2], [0, 2, 3]])
            if kind == 'alt2':
                # Ziergitter vor dem Glas
                for k in range(1, 4):
                    x = gx0 + (gx1 - gx0) * k / 4
                    P.cyl('metal', [x, gy0, 0.0], [x, gy1, 0.0], 0.008, seg=6, caps=False)
        if leaves == 2:
            P.bbox('trim', [x_in0 + lw - 0.025, 0.0, -0.03], [x_in0 + lw + 0.025, top, 0.025], bev=0.006)   # Schlagleiste
        # Drueckergarnitur, Briefschlitz, Trittblech
        hx = (x_in0 + lw - 0.07) if leaves == 2 else x_in1 - 0.08
        P.bbox('trim', [hx - 0.025, 0.95, 0.0], [hx + 0.025, 1.22, 0.012], bev=0.004, c=c_brass)
        P.cyl('trim', [hx, 1.12, 0.012], [hx, 1.12, 0.06], 0.009, seg=6, c=c_brass)
        P.bbox('trim', [hx - 0.12, 1.105, 0.05], [hx + 0.005, 1.135, 0.07], bev=0.005, c=c_brass)
        P.bbox('trim', [hx - 0.008, 1.0, 0.012], [hx + 0.008, 1.03, 0.018], bev=0.002, c=(0.2, 0.2, 0.2))
        lx = x_in0 + lw * 0.5
        P.bbox('trim', [lx - 0.16, 0.82, 0.0], [lx + 0.16, 0.88, 0.012], bev=0.004, c=c_brass)
        P.bbox('trim', [lx - 0.13, 0.842, 0.012], [lx + 0.13, 0.858, 0.014], bev=0.001, c=(0.1, 0.1, 0.1))
        P.bbox('trim', [x_in0, 0.0, 0.0], [x_in1, 0.16, 0.006], bev=0.003, c=c_brass if kind == 'alt2' else c_steel)
    else:  # Glas-/Alutuer
        P.ring_rect('trim', -w / 2, 0, w / 2, h, ALU_FRAME)
        ht = h - 0.45
        P.bar('trim', [(-0.05, -0.07), (-0.05, 0.0), (0.05, 0.0), (0.05, -0.07)], 'u', -w / 2 + 0.05, w / 2 - 0.05, at=(ht,))
        lx0, lx1 = -w / 2 + 0.056, w / 2 - 0.056
        P.ring_rect('trim', lx0, 0.0, lx1, ht - 0.05, [(0.0, -0.05), (0.0, 0.01), (0.01, 0.016), (0.09, 0.016), (0.1, 0.004), (0.1, -0.03)])
        P.bbox('trim', [lx0 + 0.1, 0.0, -0.03], [lx1 - 0.1, 0.22, 0.014], bev=0.004, c=c_steel)
        hx = lx1 - 0.15
        P.cyl('trim', [hx, 0.8, 0.09], [hx, 1.6, 0.09], 0.017, seg=8, c=c_steel)
        for y in (0.86, 1.54):
            P.cyl('trim', [hx, y, 0.016], [hx, y, 0.09], 0.01, seg=6, c=c_steel)
        P.bbox('trim', [lx0 + 0.12, ht - 0.14, 0.016], [lx0 + 0.5, ht - 0.08, 0.07], bev=0.006, c=c_steel)
        P.bbox('trim', [lx0 + 0.42, ht - 0.12, 0.05], [lx0 + 0.72, ht - 0.095, 0.07], bev=0.004, c=c_steel)
        P.add('glass', [[lx0 + 0.1, 0.22, -0.02], [lx1 - 0.1, 0.22, -0.02], [lx1 - 0.1, ht - 0.15, -0.02], [lx0 + 0.1, ht - 0.15, -0.02]], [[0, 1, 2], [0, 2, 3]])
        P.add('glass', [[-w / 2 + 0.05, ht + 0.05, -0.046], [w / 2 - 0.05, ht + 0.05, -0.046], [w / 2 - 0.05, h - 0.05, -0.046],
                        [-w / 2 + 0.05, h - 0.05, -0.046]], [[0, 1, 2], [0, 2, 3]])
    # Klingeltableau neben der Tuer auf der Wandflaeche (d = +r)
    bx = w / 2 + 0.18 if kind != 'nk' else -w / 2 - 0.18
    P.bbox('trim', [bx - 0.07, 1.15, r], [bx + 0.07, 1.55, r + 0.018], bev=0.004, c=c_steel)
    for i in range(5):
        y = 1.2 + i * 0.065
        P.cyl('trim', [bx - 0.03, y, r + 0.018], [bx - 0.03, y, r + 0.026], 0.008, seg=6, c=c_brass)
        P.bbox('trim', [bx - 0.01, y - 0.015, r + 0.018], [bx + 0.055, y + 0.015, r + 0.022], bev=0.002, c=(0.95, 0.95, 0.92))
    P.bbox('trim', [bx - 0.05, 1.45, r + 0.018], [bx + 0.05, 1.53, r + 0.024], bev=0.002, c=(0.3, 0.3, 0.3))
    PROTOS[key] = P.freeze()
    return P


def lancet_proto(w, h):
    """Spitzbogenfenster (Backsteingotik) inkl. eigenem Glas, Mittelpfosten, Masswerk-Stab und Bleisprossen (Ursprung Rahmen)."""
    key = 'lancet_%d_%d' % (int(w * 100), int(h * 100))
    if key in PROTOS:
        return PROTOS[key]
    P = Proto(key, 'frame', w, h)
    poly = lancet_poly(w, h)
    P.add('glass', np.c_[poly, np.full(len(poly), -0.03)], tri2d(poly)[1])
    # umlaufende Rahmenleiste als Stabwerk entlang des Polygons
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a; L = np.linalg.norm(e)
        if L < 1e-4:
            continue
        P.cyl('trim', [a[0], a[1], -0.01], [b[0], b[1], -0.01], 0.035, seg=6, caps=False)
    hs = h - w * 0.85
    P.bbox('trim', [-0.06, 0.0, -0.08], [0.06, hs + 0.2, 0.03], bev=0.015)
    k = max(2, int(hs / 0.55))
    for i in range(1, k + 1):
        y = hs * i / (k + 1)
        P.bbox('trim', [-w / 2, y - 0.012, -0.04], [w / 2, y + 0.012, -0.015], bev=0.003)
    for s in (-1, 1):
        P.cyl('trim', [s * w / 4, 0.0, -0.025], [s * w / 4, hs, -0.025], 0.01, seg=4, caps=False)
    # Vierpass-Andeutung im Bogenfeld
    cx, cy = 0.0, hs + (h - hs) * 0.45; rr = w * 0.18
    ang = np.linspace(0, 2 * np.pi, 13)
    for i in range(12):
        P.cyl('trim', [cx + rr * np.cos(ang[i]), cy + rr * np.sin(ang[i]), -0.01], [cx + rr * np.cos(ang[i + 1]), cy + rr * np.sin(ang[i + 1]), -0.01],
              0.03, seg=5, caps=False)
    PROTOS[key] = P.freeze()
    return P


def lancet_poly(w, h):
    hs = h - w * 0.85
    R = w
    right = [(w / 2 - R + R * np.cos(t), hs + R * np.sin(t)) for t in np.linspace(0, np.arcsin(min(1, (w * 0.85) / R)), 7)]
    left = [(-x, y) for x, y in reversed(right)]
    poly = [(-w / 2, 0), (w / 2, 0)] + right[:-1] + [(0, h)] + left[1:]
    return np.array(poly, float)

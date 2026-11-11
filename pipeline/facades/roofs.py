"""Daecher: Flachdach mit Attika (Aufkantung, Abdeckblech mit Tropfkante, Kies/Bahn/Gruen, Dachaufbauten),
Steildach (LoD2-Flaechen exakt, Traufueberstand mit Stirnbrett/Untersicht, Rinne, Fallrohre, Gauben, Schornsteine),
Ueberdachungen (Platte auf Stuetzen)."""
import math
import numpy as np
from shapely.geometry import Polygon, Point, box as sbox, MultiPolygon
from shapely.ops import unary_union
from shapely import contains_xy
from geom import Frame, wall_frame, sweep, mitre, prism_xz, ring_sides, beam, cyl, UP, tri2d, signed_area_xz
from common import rng_of, hex2lin, jitter_hex

TEX_MEAN = {'roof_tile': 0.55, 'roof_slate': 0.5, 'roof_metal': 0.62, 'roof_gravel': 0.6, 'roof_membrane': 0.62, 'roof_green': 0.5}


def _geoms(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    return [x for x in getattr(g, 'geoms', []) if x.geom_type == 'Polygon' and not x.is_empty]


def dop_lin(c):
    c = np.clip(np.asarray(c[:3], float) / 255.0 * 1.12, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def tint_for(mat, lin):
    return np.clip(np.asarray(lin) / TEX_MEAN.get(mat, 0.6), 0, 1)


def pitched_material(b, P):
    """Material/Farbe der Steildachflaechen aus dem Orthofoto (flaechengewichtet), sonst Stil.
    Feste Dachfarbe aus style.OVERRIDES (roof_col/roof_mat, z. B. Kreuzkirche rotes Ziegeldach) hat Vorrang."""
    if P.get('roof_fixed'):
        mat = P.get('roof_mat', 'roof_tile')
        return mat, tint_for(mat, hex2lin(P['roof_col']))
    cols = []; ws = []
    for r, c in zip(b.roofs, b.attrs.get('roof_dop', [])):
        if c is None or not (0.08 < r.n[1] <= 0.995):
            continue
        cols.append(c[:3]); ws.append(r.area)
    if cols:
        m = np.average(np.array(cols), 0, np.array(ws))
        R, G, B = m; L = m.mean()
        if G > R + 4 and B > R - 2 and G - R > 3:
            return 'roof_metal', tint_for('roof_metal', dop_lin([R * 0.9, G * 1.05, B]) * 1.15)
        if R > B + 18 and R > G + 6:
            return 'roof_tile', tint_for('roof_tile', dop_lin(m))
        if L < 85:
            mat = 'roof_slate' if P.get('roof_mat') == 'roof_slate' or P['kind'] == 'altbau' else 'roof_tile'
            return mat, tint_for(mat, dop_lin(m) * 1.1)
        if L > 120 and abs(R - B) < 14:
            return 'roof_metal', tint_for('roof_metal', dop_lin(m))
        return P.get('roof_mat', 'roof_tile'), tint_for(P.get('roof_mat', 'roof_tile'), dop_lin(m))
    mat = P.get('roof_mat', 'roof_tile')
    return mat, tint_for(mat, hex2lin(P.get('roof_col', '#9A4A32')))


def flat_material(b, P, r_idx):
    c = b.attrs.get('roof_dop', [None] * len(b.roofs))[r_idx] if r_idx is not None else None
    kind = P.get('flat_roof', 'gravel')
    if c is not None:
        R, G, B, spread = c
        L = (R + G + B) / 3
        if G > R + 6 and G > B + 4:
            return 'roof_green', tint_for('roof_green', dop_lin([R, G, B]))
        if L < 75:
            return 'roof_membrane', tint_for('roof_membrane', dop_lin([R, G, B]) * 1.05)
        if L > 150:
            return ('roof_membrane' if kind == 'membrane' else 'roof_gravel'), tint_for('roof_gravel', dop_lin([R, G, B]) * 0.95)
        mat = 'roof_gravel' if kind != 'membrane' else 'roof_membrane'
        return mat, tint_for(mat, dop_lin([R, G, B]))
    mat = {'gravel': 'roof_gravel', 'membrane': 'roof_membrane', 'green': 'roof_green'}[kind]
    return mat, tint_for(mat, hex2lin({'roof_gravel': '#9A968E', 'roof_membrane': '#5E6164', 'roof_green': '#6E7A4A'}[mat]))


# ------------------------------------------------------------------ Flachdach
def flat_groups(b):
    """Flache Dachflaechen gleicher Hoehe zusammenfassen -> [(poly_xz, y, [roof idx])]."""
    items = []
    for i, r in enumerate(b.roofs):
        if r.n[1] > 0.995:
            try:
                g = Polygon(r.P[:, [0, 2]], [h[:, [0, 2]] for h in r.holes]).buffer(0)
            except Exception:
                continue
            if not g.is_empty:
                items.append((g, float(r.P[:, 1].mean()), i))
    items.sort(key=lambda t: t[1])
    groups = []
    used = [False] * len(items)
    for a in range(len(items)):
        if used[a]:
            continue
        g, y, i = items[a]; members = [a]; used[a] = True
        changed = True
        while changed:
            changed = False
            for c in range(len(items)):
                if used[c] or abs(items[c][1] - y) > 0.06:
                    continue
                if any(items[c][0].buffer(0.02).intersects(items[m][0]) for m in members):
                    members.append(c); used[c] = True; changed = True
        poly = unary_union([items[m][0] for m in members]) if len(members) > 1 else items[members[0]][0]
        poly = _drop_slivers(poly)
        ys = [items[m][1] for m in members]; ar = [items[m][0].area for m in members]
        # Hoechste Teilhoehe: niedrigere Wandkronen (< 6 cm) liegen dann vor der Attika-Hinterlegung statt ueber der Attika
        groups.append((poly, float(max(ys)), [items[m][2] for m in members]))
    return groups


def _drop_slivers(g, amin=0.02):
    """winzige Loecher/Teile aus Vereinigungen entfernen (exakte Eckpunkte bleiben erhalten)."""
    out = []
    for p in _geoms(g):
        if p.area < amin:
            continue
        out.append(Polygon(p.exterior.coords, [r.coords for r in p.interiors if Polygon(r.coords).area >= amin]))
    if not out:
        return g
    return out[0] if len(out) == 1 else MultiPolygon(out)


def build_flat(acc, b, P, ctx, rng, wall_mat, wall_col):
    vol = ctx['vol']
    kind = P['kind']
    for poly, y1, ridx in flat_groups(b):
        area = poly.area
        if area < 0.3:
            continue
        mat, tint = flat_material(b, P, max(ridx, key=lambda i: b.roofs[i].area))
        if kind in ('shed', 'canopy') or area < 6:
            hp, t = (0.12 if kind != 'canopy' else 0.0), 0.12
        else:
            hp = min(P['parapet_h'], 0.3 + 0.05 * math.sqrt(area)); t = 0.26
        inner = poly.buffer(-t, join_style=2, mitre_limit=3.0) if hp > 0 else poly
        inner = inner if not inner.is_empty and inner.area > 0.2 else None
        if inner is None or hp <= 0:
            prism_xz(acc, 'shell', mat, poly, y1, y1, tint, top=True, bottom=False, sides=False)
            continue
        ring = poly.difference(inner)
        prism_xz(acc, 'shell', wall_mat, ring, y1, y1, wall_col, top=True, bottom=False, sides=False)
        # verdeckte Hinterlegung 2 mm hinter der Wandflaeche: schliesst mm-Abweichungen zwischen Wandkrone und Attika
        for g in _geoms(poly.buffer(-0.002, join_style=2)):
            ring_sides(acc, 'shell_b', wall_mat, np.array(g.exterior.coords)[:-1], y1 - 0.6, y1 - 0.0005, wall_col, outward=True)
            for it in g.interiors:
                ring_sides(acc, 'shell_b', wall_mat, np.array(it.coords)[:-1], y1 - 0.6, y1 - 0.0005, wall_col, outward=False)
        for g in _geoms(poly.buffer(-0.004, join_style=2)):          # Rueckseite (falls die Wand dort fehlt)
            ring_sides(acc, 'shell_b', wall_mat, np.array(g.exterior.coords)[:-1], y1 - 0.6, y1 - 0.0005, wall_col, outward=False)
            for it in g.interiors:
                ring_sides(acc, 'shell_b', wall_mat, np.array(it.coords)[:-1], y1 - 0.6, y1 - 0.0005, wall_col, outward=True)
        prism_xz(acc, 'shell', mat, inner, y1 - hp, y1 - hp, tint, top=True, bottom=False, sides=False)
        for g in _geoms(inner):
            ring_sides(acc, 'shell', wall_mat, np.array(g.exterior.coords)[:-1], y1 - hp, y1, wall_col, outward=False)
            for it in g.interiors:
                ring_sides(acc, 'shell', wall_mat, np.array(it.coords)[:-1], y1 - hp, y1, wall_col, outward=True)
        # Abdeckblech: aussen 4 cm Ueberstand (nicht zu gleich hohen/hoeheren Nachbarn hin), innen 3 cm
        blocked = []
        for j in vol.tree.query(poly.buffer(0.15)):
            if vol.owner[j] == b.idx:
                continue
            if vol.ytop[j] >= y1 - 0.3:
                blocked.append(vol.geoms[j])
        outer = poly.buffer(0.04, join_style=2, mitre_limit=3.0)
        if blocked:
            outer = outer.difference(unary_union(blocked).buffer(0.01, join_style=2))
        outer = outer.union(poly)
        cop = outer.difference(inner.buffer(-0.03, join_style=2, mitre_limit=3.0))
        cc = P.get('coping_col', '#8C969C')
        prism_xz(acc, 'detail', 'metal', cop, y1 - 0.035, y1 + 0.05, cc, top=True, bottom=True, sides=True)
        # Dachaufbauten
        if kind not in ('shed', 'canopy', 'parking'):
            equipment(acc, b, P, inner, y1 - hp, rng, wall_mat, wall_col, mat, tint)


def equipment(acc, b, P, inner, yr, rng, wall_mat, wall_col, rmat, rtint):
    area = inner.area
    if area < 25:
        return
    free = inner.buffer(-0.8, join_style=2)
    if free.is_empty:
        return
    placed = []
    nmax = int(min(16, area / 95.0 + 1))
    H = b.y_top - b.y_ground
    want = []
    if area > 280 and H > 14 and P['kind'] != 'gothic':
        want.append('lift')
    for _ in range(nmax):
        x = rng.random()
        want.append('ac' if x < 0.33 else 'vent' if x < 0.55 else 'sky' if x < 0.75 else 'hatch' if x < 0.85 else 'solar')
    x0, z0, x1, z1 = free.bounds
    for kind in want:
        sz = {'lift': (2.6, 2.3), 'ac': (1.3, 0.9), 'vent': (0.5, 0.5), 'sky': (1.8, 1.2), 'hatch': (1.1, 1.1), 'solar': (4.2, 2.2)}[kind]
        for attempt in range(25):
            cx = rng.uniform(x0, x1); cz = rng.uniform(z0, z1)
            ang = rng.choice([0.0, math.pi / 2]) + _principal_angle(inner)
            ca, sa = math.cos(ang), math.sin(ang)
            hw, hd = sz[0] / 2, sz[1] / 2
            corners = [(cx + ca * u - sa * v, cz + sa * u + ca * v) for u, v in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))]
            rect = Polygon(corners)
            if not free.contains(rect) or any(rect.buffer(0.6).intersects(p) for p in placed):
                continue
            placed.append(rect)
            t = np.array([ca, 0, sa]); n = np.array([-sa, 0, ca])
            fr = Frame([cx, yr, cz], t, UP, n)
            _equip(acc, fr, kind, hw, hd, rng, wall_mat, wall_col)
            break


def _principal_angle(poly):
    try:
        mrr = poly.minimum_rotated_rectangle
        c = np.array(mrr.exterior.coords)[:4]
        e = c[1] - c[0]; e2 = c[2] - c[1]
        if np.linalg.norm(e2) > np.linalg.norm(e):
            e = e2
        return math.atan2(e[1], e[0])
    except Exception:
        return 0.0


def _equip(acc, fr, kind, hw, hd, rng, wall_mat, wall_col):
    grey = ['#B4B8BA', '#C8CCCE', '#9AA1A6', '#D6D8D8'][int(rng.integers(4))]
    if kind == 'lift':
        acc.box('detail', wall_mat, fr, [-hw, -0.05, -hd], [hw, 2.5, hd], wall_col, skip=('-v',))
        acc.box('detail', 'metal', fr, [-hw - 0.04, 2.5, -hd - 0.04], [hw + 0.04, 2.57, hd + 0.04], '#8C969C', skip=())
        acc.box('detail', 'trim', fr, [-0.45, -0.05, hd], [0.45, 2.05, hd + 0.03], '#5B6066', skip=('-d',))
        for s in (-1, 1):
            acc.box('detail', 'metal', fr, [s * hw * 0.5 - 0.2, 2.57, -0.2], [s * hw * 0.5 + 0.2, 2.9, 0.2], grey, skip=('-v',))
    elif kind == 'ac':
        acc.box('detail', 'metal', fr, [-hw, 0.05, -hd], [hw, 1.0, hd], grey, skip=('-v',))
        for s in (-1, 1):
            acc.box('detail', 'metal', fr, [-hw - 0.05, -0.05, s * (hd - 0.08) - 0.05], [hw + 0.05, 0.08, s * (hd - 0.08) + 0.05], '#5B6066', skip=('-v',))
        for k in range(3):
            y = 0.25 + k * 0.22
            acc.box('detail', 'metal', fr, [-hw + 0.08, y, hd], [hw - 0.08, y + 0.04, hd + 0.02], '#5B6066', skip=('-d',))
        p = fr.w([0, 1.0, 0])
        cyl(acc, 'detail', 'metal', p, p + np.array([0, 0.06, 0]), min(hw, hd) * 0.75, '#3A3D40', seg=10)
    elif kind == 'vent':
        p0 = fr.w([0, -0.05, 0]); p1 = fr.w([0, float(rng.uniform(0.6, 1.2)), 0])
        cyl(acc, 'detail', 'metal', p0, p1, 0.12, grey, seg=8)
        cyl(acc, 'detail', 'metal', p1, p1 + np.array([0, 0.18, 0]), 0.2, grey, seg=8)
    elif kind == 'sky':
        acc.box('detail', 'metal', fr, [-hw, -0.05, -hd], [hw, 0.3, hd], '#C8CCCE', skip=('-v',))
        acc.box('detail', 'spandrel', fr, [-hw + 0.08, 0.3, -hd + 0.08], [hw - 0.08, 0.42, hd - 0.08], '#3A4A55', skip=('-v',))
    elif kind == 'hatch':
        acc.box('detail', 'metal', fr, [-hw, -0.05, -hd], [hw, 0.55, hd], grey, skip=('-v',))
    elif kind == 'solar':
        rows = 2; L = 2 * hw
        for i in range(rows):
            z = -hd + 0.3 + i * (2 * hd - 0.6) / max(1, rows - 1) if rows > 1 else 0
            a = fr.w([-hw, 0.25, z - 0.45]); bq = fr.w([hw, 0.25, z - 0.45]); c = fr.w([hw, 0.75, z + 0.35]); d = fr.w([-hw, 0.75, z + 0.35])
            V = np.array([a, bq, c, d]); acc.add('detail', 'spandrel', V, [[0, 2, 1], [0, 3, 2]] if np.cross(bq - a, d - a)[1] < 0 else [[0, 1, 2], [0, 2, 3]], '#1F2A44')
            acc.box('detail', 'metal', fr, [-hw, -0.05, z - 0.5], [hw, 0.25, z - 0.4], '#9AA1A6', skip=('-v',))
            acc.box('detail', 'metal', fr, [-hw, -0.05, z + 0.3], [hw, 0.75, z + 0.38], '#9AA1A6', skip=('-v',))


# ------------------------------------------------------------------ Steildach
def build_pitched(acc, inst, b, P, ctx, rng, walls, wall_mat, wall_col, bays_of):
    vol = ctx['vol']
    rmat, rtint = pitched_material(b, P)
    eaves = []
    sl = [i for i, r in enumerate(b.roofs) if 0.08 < r.n[1] <= 0.995]
    edges_all = {}
    for i in sl:
        P3 = b.roofs[i].P
        for k in range(len(P3)):
            a, c = P3[k], P3[(k + 1) % len(P3)]
            edges_all.setdefault(_ekey(a, c), []).append(i)
    for i, r in enumerate(b.roofs):
        if r.n[1] <= 0.08:
            _emit_poly3(acc, 'shell', 'roof_slate' if rmat != 'roof_metal' else rmat, r, rtint)
            continue
        if r.n[1] > 0.995:
            continue
        _emit_poly3(acc, 'shell', rmat, r, rtint)
        # verdeckte Schuerze (2 mm innen, 0.6 m tief) unter freien Dachkanten: schliesst LoD2-Abweichungen Dach/Wand
        n = len(r.P); ctr = r.P.mean(0)
        for k in range(n):
            a, c = r.P[k], r.P[(k + 1) % n]
            if len(edges_all.get(_ekey(a, c), [])) > 1:
                continue
            e = c - a; e[1] = 0
            if np.linalg.norm(e) < 0.05:
                continue
            nin = np.array([-e[2], 0, e[0]]); nin /= np.linalg.norm(nin)
            if np.dot(nin, ctr - a) < 0:
                nin = -nin
            a2 = a + nin * 0.002; c2 = c + nin * 0.002
            _quad(acc, 'shell_b', wall_mat, [a2, c2, c2 - UP * 0.6, a2 - UP * 0.6], -nin, wall_col)
            _quad(acc, 'shell_b', wall_mat, [a - nin * 0.002, c - nin * 0.002, c - nin * 0.002 - UP * 0.6, a - nin * 0.002 - UP * 0.6], nin, wall_col)
    for i in sl:
        r = b.roofs[i]
        dh = np.array([r.n[0], 0, r.n[2]]); L = np.linalg.norm(dh)
        if L < 1e-6:
            continue
        dh /= L
        g = L / r.n[1]                         # tan(Neigung)
        ymin = r.P[:, 1].min()
        ctr = r.P.mean(0)
        for k in range(len(r.P)):
            p, q = r.P[k].copy(), r.P[(k + 1) % len(r.P)].copy()
            if abs(p[1] - q[1]) > 0.03 or p[1] > ymin + 0.06:
                continue
            e = q - p; e[1] = 0; el = np.linalg.norm(e)
            if el < 0.8:
                continue
            e /= el
            ne = np.array([e[2], 0, -e[0]])
            if np.dot(ne, ctr - p) > 0:
                ne = -ne
            if np.dot(ne, dh) < 0.75:
                continue
            if len(edges_all.get(_ekey(p, q), [])) > 1:
                continue
            mid = (p + q) / 2
            probe = np.array([mid + ne * 0.6 - UP * 0.35, mid + ne * 0.25 - UP * 0.2, p + e * 0.3 + ne * 0.5 - UP * 0.3, q - e * 0.3 + ne * 0.5 - UP * 0.3])
            bo, bs = vol.blocked(probe, b.idx)
            if (bo | bs).any():
                continue
            t_e = np.cross(UP, ne)
            if np.dot(q - p, t_e) < 0:
                p, q = q, p
            eaves.append({'p': p, 'q': q, 'n': ne, 'g': g, 'roof': i, 'y': float(p[1])})
    if not eaves:
        return rmat, rtint
    # Verkettung (Walmdach-Ecken)
    for a in eaves:
        a['prev'] = a['next'] = None
    for a in eaves:
        for c in eaves:
            if a is c:
                continue
            if np.linalg.norm(a['q'] - c['p']) < 0.04 and abs(a['y'] - c['y']) < 0.05:
                a['next'] = c; c['prev'] = a
    o = P.get('overhang', 0.4)
    f = 0.18
    fascia_col = P.get('fascia_col', '#F2F0EA' if P['kind'] in ('altbau', 'fachwerk') else '#3A3D40')
    soffit_col = '#E8E4DC'
    zinc = '#9AA1A6'
    for E in eaves:
        p, q, n, g = E['p'], E['q'], E['n'], E['g']
        mp = mitre(E['prev']['n'], n) if E['prev'] is not None else n
        mq = mitre(n, E['next']['n']) if E['next'] is not None else n
        gp = (g + E['prev']['g']) / 2 if E['prev'] is not None else g
        gq = (g + E['next']['g']) / 2 if E['next'] is not None else g
        po = p + mp * o - UP * (o * gp); qo = q + mq * o - UP * (o * gq)
        ys = min(po[1], qo[1]) - f
        pw = np.array([p[0], ys, p[2]]); qw = np.array([q[0], ys, q[2]])
        pf = po - UP * f; qf = qo - UP * f
        pf[1] = qf[1] = ys
        _quad(acc, 'detail', rmat, [p, q, qo, po], UP, rtint)                    # Dachueberstand (Dachdeckung)
        _quad(acc, 'detail', 'trim', [po, qo, qf, pf], n, fascia_col)           # Stirnbrett
        _quad(acc, 'detail', 'plaster', [pw, qw, qf, pf], -UP, soffit_col)      # Untersicht
        te = (q - p) / np.linalg.norm(q - p)
        if E['prev'] is None:
            _poly(acc, 'detail', 'trim', [p, po, pf, pw], -te, fascia_col)
        if E['next'] is None:
            _poly(acc, 'detail', 'trim', [q, qo, qf, qw], te, fascia_col)
        # Dachrinne (geschlossenes Profil) mit Gehrung
        prof = [(0, 0.0), (0.035, -0.035), (0.1, -0.04), (0.135, -0.005), (0.145, 0.06), (0.125, 0.075), (0, 0.075)]
        yg = ys + 0.03
        sm = mp if E['prev'] is not None else None
        em = mq if E['next'] is not None else None
        sweep(acc, 'detail', 'metal', np.array([po, qo]), np.array([n]), prof, zinc, y0=yg, start_mitre=sm, end_mitre=em)
        # Fallrohre an offenen Enden, bei langen Traufen zusaetzlich
        Lq = np.linalg.norm(q - p)
        pos = []
        if E['prev'] is None:
            pos.append(0.35)
        if E['next'] is None:
            pos.append(Lq - 0.35)
        if Lq > 22 and not pos:
            pos.append(Lq / 2)
        for s in (pos if P['kind'] != 'gothic' and E['y'] - b.y_ground < 40 else []):
            downpipe(acc, b, vol, p + te * s, n, o, yg, zinc)
        # Gauben
        if P.get('dormers') and g > 0.55 and g < 2.2:
            dormers(acc, inst, b, P, E, te, rmat, rtint, walls, bays_of, rng, wall_mat, wall_col, vol)
    return rmat, rtint


def _ekey(a, c):
    k1 = tuple(np.round(a, 1)); k2 = tuple(np.round(c, 1))
    return (k1, k2) if k1 < k2 else (k2, k1)


def _emit_poly3(acc, layer, mat, r, c):
    n = r.n
    t = np.cross(UP, n)
    if np.linalg.norm(t) < 1e-6:
        t = np.array([1.0, 0, 0])
    t /= np.linalg.norm(t); bb = np.cross(n, t)
    fr = Frame(r.P[0], t, bb, n)
    L = fr.loc(r.P)[:, :2]
    H = [fr.loc(h)[:, :2] for h in r.holes]
    V2, F = tri2d(L, H)
    if len(F) == 0:
        return
    allP = np.concatenate([r.P] + list(r.holes)) if r.holes else r.P
    acc.add(layer, mat, allP[:len(V2)] if len(allP) == len(V2) else fr.w(np.c_[V2, np.zeros(len(V2))]), F, c)


def _quad(acc, layer, mat, P, want, c):
    P = np.asarray(P, float)
    nn = np.cross(P[1] - P[0], P[2] - P[0])
    F = [[0, 1, 2], [0, 2, 3]] if np.dot(nn, want) >= 0 else [[0, 2, 1], [0, 3, 2]]
    acc.add(layer, mat, P, F, c)


def _poly(acc, layer, mat, P, want, c):
    P = np.asarray(P, float)
    F = [[0, i, i + 1] for i in range(1, len(P) - 1)]
    F = np.array(F)
    tri = P[F]; nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = nn @ np.asarray(want, float) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    acc.add(layer, mat, P, F, c)


def downpipe(acc, b, vol, base, n, o, yg, c):
    top = base + n * (o + 0.07); top[1] = yg - 0.02
    wallp = base + n * 0.12
    # wie weit hinunter? bis zum ersten Volumen unter dem Rohr (z.B. niedrigeres Dach) oder bis zum Boden (oertliches
    # Gehwegniveau aus dem Hoehenfeld, das kann unter dem Bezugsniveau des Gebaeudes liegen)
    import level as LV
    gl = float(LV.sample([wallp[0] + n[0] * 0.3], [wallp[2] + n[2] * 0.3])[0])
    if not np.isfinite(gl) or abs(gl - b.y_ground) > 6.0:
        gl = b.y_ground
    gl = max(gl, getattr(b, 'y_base', gl - 0.3))
    ys = np.arange(yg - 1.0, gl - 0.1, -0.5)
    yend = gl - 0.05
    if len(ys):
        Q = np.c_[np.full(len(ys), wallp[0]), ys, np.full(len(ys), wallp[2])]
        bo, bs = vol.blocked(Q, -1)
        hit = np.where(bo | bs)[0]
        if len(hit):
            yend = ys[hit[0]] + 0.5
    if yend > yg - 1.0:
        return
    mid = wallp.copy(); mid[1] = yg - 0.45
    cyl(acc, 'detail', 'metal', top, top - UP * 0.12, 0.045, c, seg=8, caps=False)
    cyl(acc, 'detail', 'metal', top - UP * 0.12, mid, 0.045, c, seg=8, caps=False)
    bot = wallp.copy(); bot[1] = yend
    cyl(acc, 'detail', 'metal', mid, bot, 0.05, c, seg=8, caps=True)
    y = mid[1] - 0.6
    while y > yend + 0.4:
        p = wallp.copy(); p[1] = y
        cyl(acc, 'detail', 'metal', p - UP * 0.04, p + UP * 0.04, 0.062, '#6E7378', seg=8)
        y -= 2.0
    if yend < gl + 0.2:
        p = wallp.copy(); p[1] = yend
        cyl(acc, 'detail', 'metal', p, p + UP * 0.5, 0.07, '#5B6066', seg=8)


def dormers(acc, inst, b, P, E, te, rmat, rtint, walls, bays_of, rng, wall_mat, wall_col, b_vol=None):
    p, q, n, g = E['p'], E['q'], E['n'], E['g']
    roof = b.roofs[E['roof']]
    try:
        rp = Polygon(roof.P[:, [0, 2]]).buffer(-0.35, join_style=2)
    except Exception:
        return
    if rp.is_empty:
        return
    Lq = np.linalg.norm(q - p)
    # Achsen der Fassade darunter (gleiche Trauflinie)
    us = None
    for w in walls:
        if w.drop or not w.vertical or w.bays is None or abs(np.dot(w.n, n)) < 0.98:
            continue
        a = w.fr.loc(p); c = w.fr.loc(q)
        if abs(a[2]) < 0.1 and abs(c[2]) < 0.1:
            us = [w.fr.w([u, 0, 0]) for u, _ in w.bays]
            break
    if us:
        cand = [float(np.dot(x - p, te)) for x in us]
        cand = [s for s in cand if 1.0 < s < Lq - 1.0]
    else:
        nb = int((Lq - 2) / 3.0)
        cand = [1.0 + (Lq - 2) * (i + 0.5) / nb for i in range(nb)] if nb > 0 else []
    if not cand:
        return
    step = 1 if (P['kind'] == 'fachwerk' or len(cand) <= 3) else 2
    cand = cand[(len(cand) % 2 == 0 and step == 2):][::step] if step == 2 else cand
    gabled = P['kind'] in ('altbau', 'fachwerk') and rng.random() < 0.6
    ins = 0.75; wd = 1.35; hf = 1.6
    yr = lambda x: E['y'] + g * x
    fr0 = Frame([p[0], 0.0, p[2]], te, UP, n)
    for s in cand:
        Yt = yr(ins) + hf
        top_extra = (wd / 2 + 0.12) * 0.75 if gabled else 0.14
        xb = (Yt + top_extra + 0.25 - E['y']) / g
        if xb > 9:
            continue
        cs = [fr0.w([s + du, 0, -dd]) for du in (-wd / 2 - 0.15, wd / 2 + 0.15) for dd in (ins - 0.1, xb)]
        if not all(rp.contains(Point(c[0], c[2])) for c in cs):
            continue
        fp = Polygon([(c[0], c[2]) for c in (cs[0], cs[1], cs[3], cs[2])]).buffer(0.3)
        placed = DORMERS.setdefault(b.id, [])
        if any(fp.intersects(o) for o in placed):
            continue
        placed.append(fp)
        fr = fr0.shifted(du=s, dd=-ins)
        # Gaube muss frei liegen (keine hoehere Wand/Nachbar davor)
        probe = np.array([fr.w([du, yr(ins) + 0.9, dd]) for dd in (-0.3, -0.05, 0.25, 0.8, 1.6) for du in (-wd / 2 + 0.1, 0.0, wd / 2 - 0.1)])
        bo, bs = b_vol.blocked(probe, b.idx)
        if (bo | bs).any():
            continue
        if _wall_in_front(fr, yr(ins) + 0.9, wd, b):
            continue
        # Frontwand mit Fensteroeffnung
        ww, wh = 0.86, 1.1
        v0 = yr(ins) + 0.3
        # Oberkante von Front und Wangen exakt auf der Unterseite des Gaubendachs (sonst offene Kante im Dachpaket)
        Ych = (Yt + wd / 2 * 0.75 + (Yt - 0.12 - (Yt + wd / 2 * 0.75)) * (wd / 2) / (wd / 2 + 0.15)) if gabled else Yt
        front = [[-wd / 2, yr(ins) - 0.35], [wd / 2, yr(ins) - 0.35], [wd / 2, Ych], [-wd / 2, Ych]]
        if gabled:
            front = front[:3] + [[0, Yt + wd / 2 * 0.75]] + front[3:]
        hole = [[-ww / 2, v0], [-ww / 2, v0 + wh], [ww / 2, v0 + wh], [ww / 2, v0]]
        dm = 'plaster' if P['kind'] in ('altbau', 'fachwerk', 'nachkrieg') else 'metal'
        dc = P.get('trim_col', '#F2F0EA') if dm == 'plaster' else '#5B6066'
        if dm == 'plaster' and P.get('trim_mat') not in ('plaster',):
            dc = '#E8E4DC'
        acc.poly('shell', dm, fr, np.array(front), dc, [np.array(hole)])
        from facade import Opening, shell_opening, window_assembly, INT_MIX
        op = Opening(-ww / 2, ww / 2, v0, v0 + wh, 0.1, ri=0.3, back='#1A1918')
        _shell_layer(acc, fr, op, dm, dc)
        Pw = dict(P)
        window_assembly(op, Pw, 'dorm_2', rng, INT_MIX.get(P['kind'], []), None, 'metal')
        for (key, mm, origin, uc, vb, sx, sy, colr) in op.insts:
            d = -op.r if origin == 'frame' else 0.0
            inst.add(key, mm, fr, uc, vb, d, sx, sy, 1.0, colr)
        # Wangen
        for sgn in (-1, 1):
            u = sgn * wd / 2
            V = [fr.w([u, yr(ins) - 0.35, 0]), fr.w([u, Ych, 0]), fr.w([u, Ych, -(xb - ins)]), fr.w([u, yr(xb) - 0.35, -(xb - ins)])]
            _quad(acc, 'shell', dm, V, sgn * fr.t, dc)
        # Gaubendach
        if gabled:
            yr2 = Yt + wd / 2 * 0.75
            ov = 0.12
            for sgn in (-1, 1):
                a = fr.w([0, yr2 + 0.06, ov]); c = fr.w([sgn * (wd / 2 + 0.15), Yt - 0.06, ov])
                d = fr.w([sgn * (wd / 2 + 0.15), Yt - 0.06, -(xb - ins)]); e = fr.w([0, yr2 + 0.06, -(xb - ins)])
                _quad(acc, 'shell', rmat, [a, c, d, e], UP + sgn * fr.t * 0.7, rtint)
                _quad(acc, 'shell', rmat, [a - UP * 0.06, c - UP * 0.06, d - UP * 0.06, e - UP * 0.06], -UP, rtint)
                _quad(acc, 'shell', 'trim', [a, c, c - UP * 0.06, a - UP * 0.06], fr.n, '#E8E4DC')
                _quad(acc, 'shell', 'trim', [c, d, d - UP * 0.06, c - UP * 0.06], sgn * fr.t, '#E8E4DC')
        else:
            acc.box('shell', 'roof_metal', fr, [-wd / 2 - 0.15, Yt, -(xb - ins) - 0.2], [wd / 2 + 0.15, Yt + 0.14, 0.15],
                    np.clip(hex2lin('#8C969C') / 0.62, 0, 1), skip=())


WALL_CTX = {}
DORMERS = {}


def _wall_in_front(fr, yc, wd, b):
    """Steht innerhalb von 2.5 m vor der Gaube eine (beliebige) senkrechte LoD2-Wand auf Fensterhoehe?"""
    tree = WALL_CTX.get('tree')
    if tree is None:
        return False
    from shapely.geometry import LineString as _LS
    hit = False
    for du in (-wd / 2 + 0.1, 0.0, wd / 2 - 0.1):
        a = fr.w([du, yc, 0.05]); c = fr.w([du, yc, 2.5])
        ln = _LS([(a[0], a[2]), (c[0], c[2])])
        for j in tree.query(ln):
            bi, wi, n2, p0, area, cov = WALL_CTX['info'][j]
            P = WALL_CTX['bldgs'][bi].walls[wi].P
            if P[:, 1].min() - 0.05 <= yc <= P[:, 1].max() + 0.05 and _LS(P[:, [0, 2]]).intersects(ln):
                # genauer: Wandpolygon an der Schnittstelle hoch genug?
                hit = True; break
        if hit:
            break
    return hit


def _shell_layer(acc, fr, op, mat, c):
    from facade import shell_opening
    tmp = _LayerRemap(acc, 'shell')
    shell_opening(tmp, fr, op, mat, c)


class _LayerRemap:
    def __init__(self, acc, layer):
        self.acc = acc; self.layer = layer

    def add(self, layer, mat, V, F, c):
        self.acc.add(self.layer, mat, V, F, c)


def chimneys(acc, b, P, rng, rmat):
    n = int(P.get('chimneys', 0))
    if n <= 0:
        return
    sl = [r for r in b.roofs if 0.25 < r.n[1] <= 0.97 and r.area > 12]
    if not sl:
        return
    mat = 'brick' if rng.random() < 0.6 else 'plaster'
    col = '#FFFFFF' if mat == 'brick' else P.get('wall_col', '#D8D0C0')
    done = 0
    for attempt in range(40):
        if done >= n:
            break
        r = sl[int(rng.integers(len(sl)))]
        try:
            poly = Polygon(r.P[:, [0, 2]]).buffer(-0.7, join_style=2)
        except Exception:
            continue
        if poly.is_empty:
            continue
        x0, z0, x1, z1 = poly.bounds
        x, z = rng.uniform(x0, x1), rng.uniform(z0, z1)
        if not poly.contains(Point(x, z)):
            continue
        yr = _plane_y(r, x, z)
        ymin, ymax = r.P[:, 1].min(), r.P[:, 1].max()
        if yr < ymin + 0.6 * (ymax - ymin):
            continue
        dh = np.array([r.n[0], 0, r.n[2]]); dh /= np.linalg.norm(dh)
        t = np.cross(UP, dh)
        fr = Frame([x, 0, z], t, UP, dh)
        hw, hd = 0.36, 0.26
        ys = [_plane_y(r, *(np.array([x, z]) + np.array([t[0], t[2]]) * u + np.array([dh[0], dh[2]]) * d)) for u in (-hw, hw) for d in (-hd, hd)]
        y0 = min(ys) - 0.25; y1 = max(ymax, max(ys)) + 0.9
        acc.box('detail', mat, fr, [-hw, y0, -hd], [hw, y1, hd], col, skip=('-v',))
        acc.box('detail', 'stone', fr, [-hw - 0.06, y1, -hd - 0.06], [hw + 0.06, y1 + 0.08, hd + 0.06], '#C8C0B4')
        for u in (-0.14, 0.14):
            p = fr.w([u, y1 + 0.08, 0])
            cyl(acc, 'detail', 'metal', p, p + UP * 0.25, 0.07, '#7A3B2A', seg=8)
        done += 1


def _plane_y(r, x, z):
    p0 = r.P[0]; n = r.n
    return p0[1] - (n[0] * (x - p0[0]) + n[2] * (z - p0[2])) / n[1]


# ------------------------------------------------------------------ Ueberdachungen
def build_canopy(acc, b, P, rng, coll):
    """Platte (0.25 m) auf Stuetzen anstelle des LoD2-Blocks; Kollision = Platte + Stuetzen."""
    mat, tint = 'roof_metal', np.clip(hex2lin('#A6AEAF') / 0.62, 0, 1)
    if b.attrs.get('roof_dop'):
        c = [x for x in b.attrs['roof_dop'] if x is not None]
        if c:
            m = np.mean([x[:3] for x in c], 0)
            if m.mean() < 90:
                tint = np.clip(dop_lin(m) / 0.62, 0, 1)
    th = 0.25
    for r in b.roofs:
        if r.n[1] < 0.08:
            continue
        P3 = r.P
        top = P3; bot = P3 - UP * th
        _emit_poly3(acc, 'shell', mat, r, tint)
        V2 = _poly_local(r)
        if V2 is None:
            continue
        Vb, F = V2
        acc.add('shell', 'metal', Vb - UP * th, F[:, [0, 2, 1]], '#9AA1A6')
        n = len(P3)
        for k in range(n):
            a, c = P3[k], P3[(k + 1) % n]
            e = c - a; e[1] = 0
            if np.linalg.norm(e) < 1e-4:
                continue
            nn = np.array([e[2], 0, -e[0]])
            if np.dot(nn, a - P3.mean(0)) < 0:
                nn = -nn
            _quad(acc, 'shell', 'trim', [a, c, c - UP * th, a - UP * th], nn, '#5B6066')
            coll.append(np.array([a, c, c - UP * th, a - UP * th]))
        coll.append(top); coll.append(bot)
    # Stuetzen
    fp = b.footprint.buffer(-0.5)
    if fp.is_empty:
        return
    ang = _principal_angle(fp)
    ca, sa = math.cos(ang), math.sin(ang)
    mrr = np.array(fp.minimum_rotated_rectangle.exterior.coords)[:4]
    c0 = mrr.mean(0)
    loc = (mrr - c0) @ np.array([[ca, -sa], [sa, ca]])
    Lx = np.ptp(loc[:, 0]); Lz = np.ptp(loc[:, 1])
    nx = max(1, int(Lx / 8.0)); rows = [0.0] if Lz < 7 else [-Lz / 4, Lz / 4]
    for i in range(nx + 1):
        x = -Lx / 2 + Lx * i / nx if nx > 0 else 0
        for z in rows:
            px = c0[0] + ca * x - sa * z; pz = c0[1] + sa * x + ca * z
            if not fp.contains(Point(px, pz)):
                continue
            ytop = None
            for r in b.roofs:
                if r.n[1] < 0.08:
                    continue
                if Polygon(r.P[:, [0, 2]]).buffer(0).contains(Point(px, pz)):
                    ytop = _plane_y(r, px, pz) - th
                    break
            if ytop is None or ytop - b.y_base < 1.5:
                continue
            fr = Frame([px, 0, pz], np.array([ca, 0, sa]), UP, np.array([-sa, 0, ca]))
            acc.box('shell', 'metal', fr, [-0.15, b.y_base, -0.15], [0.15, ytop, 0.15], '#5B6066', skip=('-v', '+v'))
            q = np.array([fr.w([u, y, d]) for y in (b.y_base, ytop) for u, d in ((-0.15, -0.15), (0.15, -0.15), (0.15, 0.15), (-0.15, 0.15))])
            coll.append(q)


def _poly_local(r):
    n = r.n
    t = np.cross(UP, n)
    if np.linalg.norm(t) < 1e-6:
        t = np.array([1.0, 0, 0])
    t /= np.linalg.norm(t); bb = np.cross(n, t)
    fr = Frame(r.P[0], t, bb, n)
    V2, F = tri2d(fr.loc(r.P)[:, :2])
    if len(F) == 0:
        return None
    return fr.w(np.c_[V2, np.zeros(len(V2))]), F

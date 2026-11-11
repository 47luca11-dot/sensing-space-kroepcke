"""Marktkirche -- wiederverwendbare Bauteile (Oeffnungen, Masswerk, Lamellen, Fialen, Krabben, Figuren).
Alle Funktionen schreiben in einen mklib.Builder (Lokalframe u/v/h)."""
import math
import numpy as np
from mklib import (UP, nrm, loft, lathe, sweep, tube, box, chamfer_box, prism, extrude_profile, rect_prof, circle,
                   pointed_arc, arch_apex, opening_outline, offset_outline, arch_halfwidth_at, tri_poly, tri_poly_holes,
                   quads_to_tris, WallFrame, rect_ring)

# Materialnamen (Praefix MK_ wird im Blender-Schritt gesetzt)
BRICK, GLAZED, ASHLAR, SAND, SANDDK = 'Backstein', 'Glasurstein', 'Sandstein_Quader', 'Sandstein', 'Sandstein_dunkel'
COPPER, PATINA, TILE, GLASS, STEEL = 'Kupfer', 'Kupfer_Patina', 'Dachziegel', 'Bleiglas', 'Stahl'
WOOD, GOLD, BRONZE, GRANITE, DIAL, IRON, DARK = 'Lamellen', 'Gold', 'Bronze', 'Granit', 'Zifferblatt', 'Eisen', 'Dunkel'
FLIESEN, DOOR, BOARD = 'Fliesen', 'Tuerbronze', 'Schaukasten'


def textured_box(B, c, ax, ay, az, sx, sy, sz, front_mat, other_mat, uvrect, grp='det'):
    """Quader (Halbmasse sx, sy, sz); die +az-Seite bekommt front_mat mit UV (u0, v0, u1, v1) ueber die ganze Flaeche,
    alle anderen Seiten other_mat (UV = Mitte des Rechtecks bei gleichem Material, sonst Meter)."""
    c = np.asarray(c, float); ax, ay, az = (np.asarray(a, float) for a in (ax, ay, az))
    P = lambda i, j, k: c + i * sx * ax + j * sy * ay + k * sz * az
    u0, v0, u1, v1 = uvrect
    fr = np.array([P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)])          # CCW von vorn gesehen (ax x ay = az)
    uvf = np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    B.add(front_mat, quads_to_tris(fr[None]), grp, uv=quads_to_tris(np.c_[uvf, np.zeros(4)][None])[:, :, :2], expect=az)
    faces = [(np.array([P(-1, -1, -1), P(-1, 1, -1), P(1, 1, -1), P(1, -1, -1)]), -az),
             (np.array([P(1, -1, -1), P(1, 1, -1), P(1, 1, 1), P(1, -1, 1)]), ax),
             (np.array([P(-1, -1, -1), P(-1, -1, 1), P(-1, 1, 1), P(-1, 1, -1)]), -ax),
             (np.array([P(-1, 1, -1), P(-1, 1, 1), P(1, 1, 1), P(1, 1, -1)]), ay),
             (np.array([P(-1, -1, -1), P(1, -1, -1), P(1, -1, 1), P(-1, -1, 1)]), -ay)]
    for q, e in faces:
        T = quads_to_tris(q[None])
        uv = np.zeros((len(T), 3, 2)) + np.array([(u0 + u1) / 2, (v0 + v1) / 2]) if other_mat == front_mat else None
        B.add(other_mat, T, grp, uv=uv, expect=e)


# ------------------------------------------------------------------ Oeffnung (Fenster/Schallloch/Portal)
def opening(B, F, sp):
    """F: WallFrame. sp: dict x, sill, w, hs, R, kind('pointed'/'round'), orders=[(off, depth), ...] (aussen->innen),
    fill: 'glass'|'louvre'|'door'|'dark'|None, mat_rev (Leibungsmaterial) oder mats je Ordnung, seg.
    Rueckgabe: aeussere Kontur (2D, Wandkoordinaten) als Loch fuer die Wandflaeche."""
    x, sill, w, hs, R = sp['x'], sp['sill'], sp['w'], sp['hs'], sp.get('R', sp['w'])
    kind = sp.get('kind', 'pointed'); seg = sp.get('seg', 10)
    orders = sp.get('orders', [(0.25, 0.0), (0.25, 0.18), (0.10, 0.18), (0.10, 0.32), (0.0, 0.45)])
    mats = sp.get('mats', None); mrev = sp.get('mat_rev', BRICK); grp = sp.get('grp', 'wall')
    sd = sp.get('sill_drop', 0.0)
    fill = sp.get('fill', 'glass')
    sill_block = sp.get('sill_block', fill == 'glass')
    # Boden der Leibung weglassen, wenn eine Sohlbank / Schwelle ihn ersetzt (sonst koplanar -> Z-Fighting)
    no_floor = sp.get('no_floor', sill_block)

    def ring2(off):
        o = offset_outline(w, sill, hs, R, off, seg, kind, sill_drop=sd * off)
        return o + np.array([x, 0.0])
    rings3, outer2 = [], ring2(orders[0][0])
    rings2 = []
    for off, d in orders:
        r2 = ring2(off); rings2.append(r2)
        rings3.append(F.pts(r2, -d))

    def cumlen(r2):
        segl = np.linalg.norm(np.diff(np.vstack([r2, r2[:1]]), axis=0), axis=1)
        return np.r_[0.0, np.cumsum(segl)]
    # UV der Leibung: u = Weg quer durch die Ordnungen (Tiefe/Versatz), v = Bogenlaenge entlang der Kontur
    # (Mittel der beiden Ringe -> radiale Fugen im Bogen, waagrechte Lagen in den Gewaenden)
    s_ord = np.r_[0.0, np.cumsum([math.hypot(o1 - o0, d1 - d0) for (o0, d0), (o1, d1) in zip(orders[:-1], orders[1:])])]
    # Leibung: Loft zwischen aufeinanderfolgenden Ringen (offen), Normalen zur Oeffnungsachse / nach aussen
    for i in range(len(rings3) - 1):
        a, b = rings3[i], rings3[i + 1]
        if np.allclose(a, b): continue
        n = len(a); k = np.arange(n); k1 = (k + 1) % n
        if no_floor: k = k[1:]; k1 = k1[1:]
        Q = np.stack([a[k], a[k1], b[k1], b[k]], axis=1)
        T = quads_to_tris(Q)
        cv = 0.5 * (cumlen(rings2[i]) + cumlen(rings2[i + 1]))
        ua, ub = s_ord[i], s_ord[i + 1]
        UVq = np.stack([np.c_[np.full(len(k), ua), cv[k]], np.c_[np.full(len(k), ua), cv[k + 1]],
                        np.c_[np.full(len(k), ub), cv[k + 1]], np.c_[np.full(len(k), ub), cv[k]]], axis=1)
        UV = quads_to_tris(np.concatenate([UVq, np.zeros(UVq.shape[:2] + (1,))], axis=2))[:, :, :2]
        c = T.mean(axis=1)
        # Erwartete Richtung: zur Achse (x, clamp(y)) in der Wandebene + etwas nach aussen
        rel = c - F.o[None]
        cx = rel @ F.t; cy = c[:, 2]
        tx = x - cx; ty = np.clip(cy, sill, hs) - cy
        e = tx[:, None] * F.t[None] + ty[:, None] * UP[None]
        e = nrm(e) * 1.0 + 0.6 * F.n[None]
        m = mats[i] if mats is not None and i < len(mats) else mrev
        B.add(m, T, grp, uv=UV, expect=e)
    dk = orders[-1][1]
    inner2 = ring2(orders[-1][0])
    if fill == 'glass':
        # Glas direkt am Ende der Leibung (kein Spalt, durch den man ins Innere schauen koennte)
        B.add(GLASS, tri_poly(F.pts(inner2, -dk - 0.004)), 'det', expect=F.n)
    elif fill in ('louvre', 'dark'):
        dd = sp.get('dark_depth', 0.9)        # muss vor einer evtl. Wandrueckseite enden (Giebel: 0.5)
        B.add(DARK, tri_poly(F.pts(inner2, -dk - dd)), 'det', expect=F.n)
        # kurze dunkle Leibung bis zur Rueckwand (damit kein Spalt sichtbar ist)
        a = F.pts(inner2, -dk); b = F.pts(inner2, -dk - dd)
        n = len(a); k = np.arange(n); k1 = (k + 1) % n
        T = quads_to_tris(np.stack([a[k], a[k1], b[k1], b[k]], axis=1))
        c = T.mean(axis=1); rel = c - F.o[None]
        e = (x - rel @ F.t)[:, None] * F.t[None] + (np.clip(c[:, 2], sill, hs) - c[:, 2])[:, None] * UP[None]
        B.add(DARK, T, 'det', expect=e)
        if fill == 'louvre':
            louvres(B, F, x, sill, w + 2 * orders[-1][0], hs, R + orders[-1][0], -dk - 0.25, kind)
    elif fill == 'door':
        pass
    if sill_block:
        # Sohlbank ueber die ganze aeussere Leibungsbreite und bis hinter die Glasebene: ersetzt den Leibungsboden
        wb = w + 2 * orders[0][0] + 0.04
        prof = [(0.10, -0.16), (0.10, -0.02), (-dk - 0.012, 0.13), (-dk - 0.012, -0.16)]
        T = extrude_profile(prof, F.p(x - wb / 2, sill), F.n, UP, F.t, 0, wb)
        B.add(SAND, T, 'det', closed=True)
    return outer2


def louvres(B, F, x, sill, w, hs, R, d, kind='pointed'):
    """Schalllamellen (schraeg, 40 Grad) in einer Oeffnung, Bretter bis zum Bogen gekuerzt."""
    step = 0.27; t = 0.025; depth = 0.22
    y = sill + 0.15
    while y < (arch_apex(w, hs, R) if kind == 'pointed' else hs + w / 2) - 0.12:
        hw = arch_halfwidth_at(w, hs, R, y + 0.1) if kind == 'pointed' else (w / 2 if y < hs else math.sqrt(max((w / 2) ** 2 - (y + 0.1 - hs) ** 2, 0)))
        hw -= 0.02
        if hw > 0.06:
            c = F.p(x, y, d)
            az = nrm(UP * math.cos(math.radians(50)) + F.n * math.sin(math.radians(50)))
            ay = np.cross(az, F.t)
            B.add(WOOD, box(c, F.t, ay, az, hw, depth / 2, t / 2), 'fine', closed=True)
        y += step


# ------------------------------------------------------------------ Masswerk
def tracery(B, F, x, sill, w, hs, R, d, lights, mat=SAND, bar=0.15, depthbar=0.2, glazing=True, seg=10):
    """Stabwerk + Teilboegen + Kreis/Paesse im Bogenfeld; d = Tiefe der Masswerkebene (negativ = innen)."""
    prof = rect_prof(bar, depthbar, ch=0.03)
    nref = F.n

    def P2(xy):
        xy = np.asarray(xy, float)
        return F.pts(xy, d)

    def arcpath(cx, ww, hh, RR):
        a = pointed_arc(ww, hh, RR, seg)
        return P2(a + np.array([cx, 0.0]))

    def circ(cx, cy, r, n=24):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return P2(np.c_[cx + r * np.cos(t), cy + r * np.sin(t)])
    apex = arch_apex(w, hs, R)
    # Rahmenbogen innen (deckt Fuge zwischen Glas und Leibung)
    fr = opening_outline(w - 0.02, sill, hs, R - 0.01, seg) + np.array([x, 0.0])
    B.add(mat, sweep(P2(fr), rect_prof(0.12, depthbar, ch=0.02), nref, closed_path=True), 'det', closed=True)
    if lights >= 2:
        m = bar
        wl = (w - (lights - 1) * m) / lights
        xs = [x - w / 2 + wl / 2 + i * (wl + m) for i in range(lights)]
        # Pfosten
        for i in range(lights - 1):
            xm = x - w / 2 + (i + 1) * wl + (i + 0.5) * m
            B.add(mat, sweep(P2([(xm, sill), (xm, hs + 0.02)]), prof, nref), 'det', closed=True)
        # Teilboegen ueber jeder Bahn
        hs_l = hs
        for xc in xs:
            B.add(mat, sweep(arcpath(xc, wl + m, hs_l, (wl + m) * 0.95), prof, nref), 'det', closed=True)
            # Nasen (kleine Dreipaesse) im Teilbogen
            ap = arch_apex(wl + m, hs_l, (wl + m) * 0.95)
            r3 = min(wl * 0.22, 0.22)
            if r3 > 0.07:
                cy = ap - r3 * 2.0
                for ang in (90, 210, 330):
                    a = math.radians(ang)
                    B.add(mat, sweep(circ(xc + 0.55 * r3 * math.cos(a), cy + 0.55 * r3 * math.sin(a), r3 * 0.54, 14),
                                     rect_prof(0.06, depthbar * 0.7), nref, closed_path=True), 'fine', closed=True)
        ap_l = arch_apex(wl + m, hs_l, (wl + m) * 0.95)
        # Bogenfeld: grosser Kreis mit Paessen
        if lights == 4:
            # zwei Spitzboegen je 2 Bahnen + Kreise darin
            for s in (-1, 1):
                ww = w / 2 - m / 2 + 0.02
                cx = x + s * (w / 4 + m / 4)
                B.add(mat, sweep(arcpath(cx, ww, hs, ww * 0.95), prof, nref), 'det', closed=True)
                ap2 = arch_apex(ww, hs, ww * 0.95)
                rr = min((ap2 - ap_l) * 0.45, ww * 0.28)
                if rr > 0.1:
                    cy = ap_l + rr + 0.08
                    B.add(mat, sweep(circ(cx, cy, rr), prof * 0.8, nref, closed_path=True), 'det', closed=True)
            ap_l = arch_apex(w / 2 - m / 2 + 0.02, hs, (w / 2) * 0.95)
        # grosser Kreis oben
        top = apex - 0.12
        lo = ap_l + 0.05
        rc = (top - lo) / 2
        cy = lo + rc
        rc = min(rc, arch_halfwidth_at(w, hs, R, cy) - 0.08)
        if rc > 0.15:
            B.add(mat, sweep(circ(x, cy, rc, 28), prof, nref, closed_path=True), 'det', closed=True)
            nf = 3 if lights == 3 else 4
            rf = rc * (0.46 if nf == 3 else 0.41)
            for k in range(nf):
                a = math.radians(90 + k * 360 / nf + (45 if nf == 4 else 0))
                B.add(mat, sweep(circ(x + (rc - rf - 0.02) * math.cos(a), cy + (rc - rf - 0.02) * math.sin(a), rf, 18),
                                 rect_prof(0.08, depthbar * 0.8), nref, closed_path=True), 'fine', closed=True)
        if glazing:
            glazing_bars(B, F, x, sill, w, hs, R, d - 0.02, xs, wl, ap_l)
    else:
        if glazing:
            glazing_bars(B, F, x, sill, w, hs, R, d - 0.02, [x], w, apex)


def glazing_bars(B, F, x, sill, w, hs, R, d, xs, wl, ytop):
    """Windeisen/Sprossen (horizontal alle 0.42 m, je Bahn eine senkrechte) -- duenne Stahlstaebe vor dem Glas."""
    T = []
    for xc in xs:
        y = sill + 0.42
        while y < hs - 0.05:
            T.append(box(F.p(xc, y, d + 0.01), F.t, UP, F.n, wl / 2 - 0.01, 0.012, 0.012))
            y += 0.42
        T.append(box(F.p(xc, (sill + hs) / 2, d + 0.01), F.t, UP, F.n, 0.01, (hs - sill) / 2, 0.01))
    if T:
        for t in T: B.add(STEEL, t, 'fine', closed=True)


# ------------------------------------------------------------------ Rundstab aus Formsteinen (rot / schwarz glasiert im Wechsel)
def brick_roll(B, path, r, matA, matB, k0=0, course=0.1, seg=8, nref=UP, grp='det', seed=0):
    """Pfad nach Bogenlaenge in Steinhoehen teilen; jeder Formstein bekommt sein Material (Wechsel, um k0 versetzt)
    und eine UV-Abbildung genau auf einen Laeufer der Backsteintextur (Fuge am Steinanfang) -> saubere Ringe statt verzerrter Fugen.
    Gesamtkoerper geschlossen (Endkappen), Materialgrenzen teilen exakt dieselben Ringe."""
    P = np.asarray(path, float)
    Ls = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(2, int(round(Ls[-1] / course)))
    s = np.linspace(0, Ls[-1], n + 1)
    Q = np.stack([np.interp(s, Ls, P[:, i]) for i in range(3)], axis=1)
    d = nrm(np.diff(Q, axis=0)); Tn = nrm(np.vstack([d[:1], d]) + np.vstack([d, d[-1:]]))
    nref = nrm(nref)
    side = nrm(np.cross(Tn, nref)); up2 = np.cross(side, Tn)
    th = -math.pi / 2 + np.arange(seg) * 2 * math.pi / seg
    R = Q[:, None, :] + r * (np.cos(th)[None, :, None] * side[:, None, :] + np.sin(th)[None, :, None] * up2[:, None, :])
    j = np.arange(seg); j1 = (j + 1) % seg
    # Orientierung: Normale weg von der Achse
    q0 = np.array([R[0, 0], R[0, 1], R[1, 1]]); nq = np.cross(q0[1] - q0[0], q0[2] - q0[0])
    flip = float(nq @ (q0.mean(axis=0) - (Q[0] + Q[1]) / 2)) < 0
    rng = np.random.default_rng(seed)
    out = {matA: ([], []), matB: ([], [])}
    for i in range(n):
        m = matA if (i + k0) % 2 == 0 else matB
        c = int(rng.integers(0, 20)); ui = int(rng.integers(0, 5))
        x0 = ui * 0.4 - (c % 2) * 0.2 + 0.012; v0 = c * course - 0.005
        quads = np.stack([R[i, j], R[i, j1], R[i + 1, j1], R[i + 1, j]], axis=1)
        uq = np.stack([np.c_[x0 + 0.24 * j / seg, np.full(seg, v0)], np.c_[x0 + 0.24 * (j + 1) / seg, np.full(seg, v0)],
                       np.c_[x0 + 0.24 * (j + 1) / seg, np.full(seg, v0 + course)], np.c_[x0 + 0.24 * j / seg, np.full(seg, v0 + course)]], axis=1)
        T = quads_to_tris(quads); U = quads_to_tris(np.concatenate([uq, np.zeros((seg, 4, 1))], axis=2))[:, :, :2]
        if flip: T = T[:, ::-1]; U = U[:, ::-1]
        out[m][0].append(T); out[m][1].append(U)
    caps = np.concatenate([tri_poly(R[0], normal=-Tn[0]), tri_poly(R[-1], normal=Tn[-1])])
    cu =np.zeros((len(caps), 3, 2)) + np.array([0.13, 0.05])
    out[matA][0].append(caps); out[matA][1].append(cu)
    for m, (Ts, Us) in out.items():
        if Ts: B.add(m, np.concatenate(Ts), grp, uv=np.concatenate(Us))


# ------------------------------------------------------------------ Wand mit Oeffnungen
def wall_face(B, F, outline2, holes2, mat=BRICK, grp='wall'):
    outer = F.pts(np.asarray(outline2, float), 0.0)
    holes = [F.pts(np.asarray(h, float), 0.0) for h in holes2]
    B.add(mat, tri_poly_holes(outer, holes, F.n), grp, expect=F.n)


# ------------------------------------------------------------------ Krabbe / Kreuzblume / Fiale
def crocket(B, base, out, up, size, mat=SANDDK, grp='fine'):
    """Krabbe: eingerolltes Blatt. base Punkt, out = Richtung weg von der Kante, up = entlang der Kante (aufwaerts)."""
    out = nrm(out); up = nrm(up)
    pts, sc = [], []
    for i, t in enumerate(np.linspace(0, 1, 7)):
        a = t * math.pi * 1.25
        r = size * (0.15 + 0.85 * t)
        p = base + out * (size * 0.9 * math.sin(a) * (1 - 0.25 * t)) + up * (size * (0.6 * t + 0.35 * (1 - math.cos(a))))
        pts.append(p); sc.append(1.0 - 0.65 * t)
    prof = np.array([(size * 0.32 * math.cos(q), size * 0.2 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 7)[:-1]])
    B.add(mat, sweep(np.array(pts), prof, np.cross(out, up), scale=sc), grp, closed=True)


def finial(B, base, size, mat=SANDDK, grp='fine'):
    """Kreuzblume: Hals, 4 Krabben im Kreuz, Knauf."""
    base = np.asarray(base, float)
    B.add(mat, lathe(base, [(size * 0.18, 0), (size * 0.16, size * 0.25), (size * 0.22, size * 0.3), (size * 0.12, size * 0.38), (size * 0.12, size * 0.9)], 8), grp, closed=True)
    for k in range(4):
        a = k * math.pi / 2
        o = np.array([math.cos(a), math.sin(a), 0.0])
        crocket(B, base + UP * size * 0.32 + o * size * 0.1, o, UP, size * 0.42, mat, grp)
    B.add(mat, lathe(base + UP * size * 0.85, [(0.0, 0), (size * 0.14, size * 0.08), (size * 0.15, size * 0.2), (size * 0.07, size * 0.32), (0.0, size * 0.38)], 8), grp, closed=True)


def pinnacle(B, base, w, shaft_h, spire_h, mat=SANDDK, crockets=True, grp='det', rot=0.0):
    """Fiale: quadratischer Leib mit 4 Wimpergen, Pyramidenhelm mit Krabben, Kreuzblume. base = Fussmitte."""
    base = np.asarray(base, float)
    c, s = math.cos(rot), math.sin(rot)
    ax = np.array([c, s, 0.0]); ay = np.array([-s, c, 0.0])
    hw = w / 2
    B.add(mat, chamfer_box(base + UP * shaft_h / 2, ax, ay, UP, hw, hw, shaft_h / 2, 0.02), grp, closed=True)
    # Wimperge an den 4 Seiten
    gh = w * 0.9
    for k in range(4):
        a = rot + k * math.pi / 2
        o = np.array([math.cos(a), math.sin(a), 0.0]); t = np.array([-math.sin(a), math.cos(a), 0.0])
        p0 = base + UP * shaft_h + o * (hw + 0.01)
        tri = [(-hw - 0.02, -0.02), (hw + 0.02, -0.02), (0.0, gh)]
        B.add(mat, extrude_profile(tri, p0, t, UP, -o, -0.04, hw * 0.9), grp, closed=True)
        if crockets:
            for q in (0.35, 0.7):
                for sgn in (-1, 1):
                    pp = p0 + t * sgn * hw * (1 - q) + UP * gh * q + o * 0.02
                    crocket(B, pp, nrm(o * 0.3 + t * sgn * 0.5 + UP * 0.6), UP, w * 0.16, mat, 'fine')
    # Helm
    top = base + UP * (shaft_h + spire_h)
    r0 = hw * 1.02
    rings = [rect_ring(base[0], base[1], r0 * 0.98, r0 * 0.98, base[2] + shaft_h + 0.01, rot),
             rect_ring(base[0], base[1], 0.03, 0.03, top[2], rot)]
    B.add(mat, loft(rings), grp, closed=True)
    if crockets:
        nk = max(2, int(spire_h / (w * 0.6)))
        for i in range(1, nk + 1):
            q = i / (nk + 1)
            for k in range(4):
                a = rot + k * math.pi / 2 + math.pi / 4
                o = np.array([math.cos(a), math.sin(a), 0.0])
                rr = r0 * math.sqrt(2) * (1 - q)
                pp = base + UP * (shaft_h + spire_h * q) + o * rr
                crocket(B, pp, nrm(o + UP * 0.3), UP, w * 0.17 * (1 - q * 0.5), mat, 'fine')
    finial(B, top - UP * 0.02, w * 0.75, mat, 'fine')


def iron_finial(B, base, h, ball=0.1, grp='fine'):
    base = np.asarray(base, float)
    B.add(IRON, lathe(base, [(0.05, 0), (0.035, h * 0.15), (0.03, h)], 8), grp, closed=True)
    B.add(IRON, lathe(base + UP * (h * 0.55 - ball), [(0.0, 0), (ball * 0.7, ball * 0.3), (ball, ball), (ball * 0.7, ball * 1.7), (0.0, 2 * ball)], 10), grp, closed=True)
    B.add(IRON, lathe(base + UP * h, [(0.03, 0), (0.0, h * 0.25)], 6), grp, closed=True)


# ------------------------------------------------------------------ Figur (stilisiert, z.B. Luther, Portalfiguren)
def figure(B, foot, facing, H, mat, pose='luther', grp='det'):
    """Stehende Figur im Talar: Rumpf als Loft elliptischer Ringe mit Falten, Kopf, Arme, Haende, Buch."""
    foot = np.asarray(foot, float)
    f = nrm([facing[0], facing[1], 0.0]); r = np.cross(f, UP)   # r = rechts der Figur (vom Betrachter links)
    s = H / 1.85
    rings = []
    prof = [(0.0, 0.35, 0.31), (0.03, 0.36, 0.32), (0.3, 0.315, 0.265), (0.65, 0.275, 0.215), (0.95, 0.255, 0.185),
            (1.2, 0.26, 0.175), (1.42, 0.275, 0.165), (1.5, 0.225, 0.135), (1.55, 0.09, 0.08), (1.58, 0.06, 0.06)]
    nseg = 28
    for i, (h, a, b) in enumerate(prof):
        t = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
        fold = 1.0 + (0.1 if h < 1.0 else 0.04) * np.sin(t * 7 + i * 0.25) * (1 if h < 1.3 else 0.3) + (0.05 if h < 1.0 else 0.0) * np.sin(t * 13 + 1.0)
        x = a * s * np.cos(t) * fold; y = b * s * np.sin(t) * fold
        P = foot[None] + x[:, None] * r[None] + y[:, None] * f[None] + UP[None] * h * s
        rings.append(P)
    B.add(mat, loft(rings), grp, closed=True)
    # Kopf
    hc = foot + UP * 1.70 * s
    B.add(mat, lathe(hc - UP * 0.13 * s, [(0.0, 0), (0.07 * s, 0.02 * s), (0.10 * s, 0.09 * s), (0.105 * s, 0.15 * s), (0.09 * s, 0.22 * s), (0.05 * s, 0.26 * s), (0.0, 0.27 * s)], 12), grp, closed=True)
    B.add(mat, lathe(hc - UP * 0.17 * s, [(0.06 * s, 0), (0.055 * s, 0.07 * s)], 8), grp, closed=True)   # Hals
    # Arme
    sh_r = foot + UP * 1.42 * s + r * 0.22 * s; sh_l = foot + UP * 1.42 * s - r * 0.22 * s
    if pose == 'luther':
        # rechter Arm erhoben, Zeigefinger nach oben; linker haelt Buch vor der Brust
        el = sh_r + r * 0.18 * s + f * 0.1 * s - UP * 0.05 * s
        hd = el + UP * 0.38 * s + f * 0.08 * s + r * 0.04 * s
        B.add(mat, sweep([sh_r, el, hd], np.array([(0.095 * s * math.cos(q), 0.095 * s * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 11)[:-1]]), f, scale=[1.25, 1.05, 0.85]), grp, closed=True)
        B.add(mat, lathe(hd, [(0.0, 0), (0.045 * s, 0.02 * s), (0.04 * s, 0.09 * s), (0.0, 0.11 * s)], 8), grp, closed=True)
        B.add(mat, lathe(hd + UP * 0.1 * s, [(0.014 * s, 0), (0.0, 0.1 * s)], 6), grp, closed=True)
        el2 = sh_l - r * 0.05 * s - UP * 0.3 * s + f * 0.12 * s
        hd2 = foot + UP * 1.05 * s + f * 0.24 * s - r * 0.02 * s
        B.add(mat, sweep([sh_l, el2, hd2], np.array([(0.1 * s * math.cos(q), 0.1 * s * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 11)[:-1]]), r, scale=[1.25, 1.1, 1.0]), grp, closed=True)
        bk = hd2 + UP * 0.12 * s + f * 0.03 * s
        B.add(mat, chamfer_box(bk, r, nrm(UP + f * 0.4), np.cross(r, nrm(UP + f * 0.4)), 0.13 * s, 0.17 * s, 0.035 * s, 0.008), grp, closed=True)
    else:
        for sh, sg in ((sh_r, 1), (sh_l, -1)):
            el = sh + r * sg * 0.06 * s - UP * 0.32 * s + f * 0.08 * s
            hd = foot + UP * 1.0 * s + f * 0.2 * s + r * sg * 0.06 * s
            B.add(mat, sweep([sh, el, hd], np.array([(0.075 * s * math.cos(q), 0.075 * s * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 9)[:-1]]), r, scale=[1.2, 1.0, 0.8]), grp, closed=True)
        B.add(mat, chamfer_box(foot + UP * 1.08 * s + f * 0.24 * s, r, UP, f, 0.12 * s, 0.16 * s, 0.05 * s, 0.01), grp, closed=True)

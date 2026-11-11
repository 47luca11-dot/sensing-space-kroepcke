"""Ladenfronten des Alten Rathauses (Laubengang Karmarschstrasse, Giebelboegen, Schmiedestrasse).

Je Ladenoeffnung (Opening mit content='shop', Attribut op.shop = Ladenbeschreibung):
  * geschlossene Laibung bis zum Glas (Stahlfutter), Stahlrahmen mit Pfosten/Kaempfer, Sockelblech,
  * Glastuer mit Stossgriff (Edelstahl) und Tuerschliesser in einer Bahn,
  * durchsichtiges Glas (AR_Schaufenster_Glas, BLEND), Glasbeschriftung (Alpha-Maske),
  * dahinter ein geschlossener, beleuchteter Ladenraum (Huelle mit Vertexfarben je Ladenart, Normalen nach innen) und
    Auslagen/Moebel als instanzierte Prototypen der Fassaden-Pipeline (pipeline/facades/displays.py, gleiche Arten wie
    data/shops.json display.kind),
  * mode 'screen': Glaswand mit Tuer in einer Bogenleibung, die den Laubengang zur Seitenstrasse abschliesst (kein Raum).
Raeume werden am Ende (emit_rooms) zugeteilt: Tiefe so gross wie moeglich (bis 'depth'), ohne andere Raeume, den
Laubengang oder die Gebaeudegrenze zu schneiden.
Lokale Fassadenkoordinaten wie ar_geom: a rechts, b oben, c nach aussen."""
import os, sys, math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box as sbox
from shapely.ops import unary_union
from ar_geom import Frame, MB, side_band, frame_matrix, area2, unit

ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
FAC_DIR = os.path.join(ROOT, 'pipeline', 'facades')

GLS = 'AR_Schaufenster_Glas'; GLW = 'AR_Glaswand_Glas'; LR = 'AR_Ladenrahmen'; TG = 'AR_Tuergriff'; LD = 'AR_Laden'

# Raumfarben (sRGB hex) je Ladenart: Wand, Boden, Decke, Leuchte (wie pipeline/facades/shops.py ROOM_STYLE)
ROOM_STYLE = {
    'fashion': ('#F2F0EC', '#B8A890', '#F4F2EE', 'dsp_ceil'), 'jewellery': ('#6A5648', '#4A3A32', '#5E4E44', 'dsp_ceil_warm'),
    'cafe': ('#D8C4A8', '#6B4E3A', '#E8DCC8', 'dsp_ceil_warm'), 'bank': ('#E8E6E2', '#9A9690', '#F2F2EE', 'dsp_ceil'),
    'lobby': ('#E0D4C0', '#5A4A3A', '#F2EDE2', 'dsp_ceil_warm'), 'generic': ('#ECEAE6', '#A8A49C', '#F2F2EE', 'dsp_ceil'),
    'vacant': ('#B8B4AC', '#6E6A66', '#C8C4BC', None), 'books': ('#E8DCC8', '#7A5A3A', '#F2EDE2', 'dsp_ceil_warm'),
    'home': ('#EAE2D6', '#A88A6A', '#F2EDE2', 'dsp_ceil_warm'),
}


def hex_lin(h):
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class ShopState:
    def __init__(self):
        self.requests = []          # Raumwuensche (werden in emit_rooms zugeteilt)
        self.used = set()           # benutzte Auslagen-Prototypen
        self.rooms = []             # zugeteilte Raeume (Plan-Polygone)
        self.log = []


STATE = ShopState()


# ============================================================================ Ladenfront
def shop_front(mb, fr, op, inner, D):
    """Ladenfront in einer Oeffnung (inner = innerster Laibungsumriss, D = Laibungstiefe)."""
    spec = dict(getattr(op, 'shop', None) or {})
    mode = spec.get('mode', 'room')
    y0 = float(op.y0)
    fl = spec.get('floor')
    if fl is not None and fl > y0 + 1e-3:          # Glaswand erst ab Gangboden (darunter Schwelle)
        Pc = Polygon(inner).buffer(0).intersection(sbox(-1e4, fl, 1e4, 1e4))
        if isinstance(Pc, MultiPolygon): Pc = max(Pc.geoms, key=lambda g: g.area)
        inner = np.array(Pc.simplify(0.0005).exterior.coords)[:-1]
        if area2(inner) < 0: inner = inner[::-1]
        y0 = fl
    x0, x1 = float(inner[:, 0].min()), float(inner[:, 0].max())
    top = float(inner[:, 1].max())
    ys = op.y0 + op.hs                                      # Kaempfer (Bogenoeffnungen) bzw. Oberkante (rect)
    if mode == 'screen':
        gc = -D + spec.get('screen_in', 0.12)               # Glasebene in der Leibung
        lin0 = None
    else:
        gc = -D - 0.10                                      # Glasebene hinter dem Laibungsende
        # Stahlfutter: schliesst den Streifen zwischen Laibungsende und Glas (Normalen in die Oeffnung)
        side_band(mb, LR, fr, inner, gc, -D, skip_bottom=op.skip_bottom)
    rect = op.kind in ('rect', 'flat')
    # ---- Bahnen: rect 3 Bahnen, Bogen 3 Bahnen unter dem Kaempfer; eine Bahn ist Tuer
    W = x1 - x0
    npan = spec.get('panes', 3 if W > 2.0 else 2)
    door_i = spec.get('door')                               # Index der Tuerbahn oder None
    mw = 0.07                                               # Pfostenbreite
    edges = np.linspace(x0, x1, npan + 1)
    y_tr = ys if not rect else y0 + 2.32                    # Kaempfer-/Oberlichtriegel
    if rect and top - y_tr < 0.25: y_tr = top - 0.25
    # Schwellenblech: schliesst den Boden der Laibung bis zum Glas (Laibungen ohne Bruestung)
    if mode != 'screen' and op.skip_bottom:
        mb.poly(LR, fr, [(x0, y0 + 0.004, 0.0), (x1, y0 + 0.004, 0.0), (x1, y0 + 0.004, gc), (x0, y0 + 0.004, gc)], hint=(0, 1, 0))
        # Stirn des Schwellenblechs nur im Oeffnungsausschnitt (y0 .. y0+4 mm); darunter liegt die Wandflaeche (sonst koplanar)
        mb.poly(LR, fr, [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y0 + 0.004, 0.0), (x0, y0 + 0.004, 0.0)], hint=(0, 0, 1))
    # Glas (eine Flaeche ueber den ganzen Umriss; Rahmen liegen davor)
    gl = [(p[0], p[1], gc) for p in inner]
    mb.poly(GLW if mode == 'screen' else GLS, fr, gl, hint=(0, 0, 1))
    if mode == 'screen':        # auch von innen (Laubengang) als Glas sichtbar
        mb.poly(GLW, fr, [(p[0], p[1], gc - 0.004) for p in inner], hint=(0, 0, -1))
    # Blendrahmen umlaufend (Ring 6 cm, vor dem Glas) + innere Kante
    fw = 0.065; fd = 0.07
    P = Polygon(inner).buffer(0)
    ii = P.buffer(-fw, join_style=2)
    if not ii.is_empty:
        if isinstance(ii, MultiPolygon): ii = max(ii.geoms, key=lambda g: g.area)
        ic = np.array(ii.exterior.coords)[:-1]
        if area2(ic) < 0: ic = ic[::-1]
        mb.poly(LR, fr, [(p[0], p[1], gc + fd) for p in inner], holes=[[(p[0], p[1], gc + fd) for p in ic]], hint=(0, 0, 1))
        for i in range(len(ic)):
            p, q = ic[i], ic[(i + 1) % len(ic)]
            mb.quad(LR, fr, (p[0], p[1], gc + fd), (q[0], q[1], gc + fd), (q[0], q[1], gc), (p[0], p[1], gc))
        if mode == 'screen':     # Rueckseite des Rahmens (zum Laubengang)
            mb.poly(LR, fr, [(p[0], p[1], gc - 0.03) for p in inner], holes=[[(p[0], p[1], gc - 0.03) for p in ic]], hint=(0, 0, -1))
            for i in range(len(ic)):
                p, q = ic[i], ic[(i + 1) % len(ic)]
                mb.quad(LR, fr, (q[0], q[1], gc - 0.03), (p[0], p[1], gc - 0.03), (p[0], p[1], gc), (q[0], q[1], gc))
    cb = gc - (0.03 if mode == 'screen' else 0.0)          # Rueckseite der Profile
    fz = 'xXyYzZ' if mode == 'screen' else 'xXyYZ'
    # Kaempferriegel (ueber die ganze Breite) + Pfosten bis zum Riegel
    mb.box(LR, fr, x0 + fw, x1 - fw, y_tr - 0.05, y_tr + 0.05, cb, gc + fd, faces=fz)
    for k in range(1, npan):
        x = edges[k]
        mb.box(LR, fr, x - mw / 2, x + mw / 2, y0 + 0.02, y_tr - 0.05, cb, gc + fd, faces=fz.replace('y', '').replace('Y', ''))
    # Oberlicht im Bogen: radiale/senkrechte Sprossen
    if not rect and top - y_tr > 0.4:
        for k in range(1, npan):
            x = edges[k]
            from ar_parts import _y_at
            yt = _y_at(inner, x)
            if yt - y_tr > 0.15:
                mb.box(LR, fr, x - 0.02, x + 0.02, y_tr + 0.05, yt - 0.02, cb, gc + 0.045, faces=fz.replace('y', '').replace('Y', ''))
    # Sockelblech (Edelstahl) unter den festen Bahnen
    for k in range(npan):
        if k == door_i: continue
        a0 = edges[k] + (mw / 2 if k > 0 else fw); a1 = edges[k + 1] - (mw / 2 if k < npan - 1 else fw)
        mb.box(LR, fr, a0, a1, y0 + 0.0, y0 + 0.16, cb, gc + 0.05, faces=fz.replace('y', ''))
    # Tuer
    if door_i is not None and 0 <= door_i < npan:
        a0 = edges[door_i] + (mw / 2 if door_i > 0 else fw); a1 = edges[door_i + 1] - (mw / 2 if door_i < npan - 1 else fw)
        _door(mb, fr, a0, a1, y0, y_tr - 0.05, gc, both=(mode == 'screen'))
    # Glasbeschriftung
    dec = spec.get('decal')
    if dec:
        k = 0 if door_i != 0 else npan - 1
        a0 = edges[k] + 0.12; a1 = edges[k + 1] - 0.12
        h = (a1 - a0) / 4.0
        yd = y0 + 1.25
        uv = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
        mb.tri_local('AR_Dekor_' + dec, fr, np.array([[(a0, yd, gc + 0.006), (a1, yd, gc + 0.006), (a1, yd + h, gc + 0.006)],
                                                       [(a0, yd, gc + 0.006), (a1, yd + h, gc + 0.006), (a0, yd + h, gc + 0.006)]]), uv=uv)
    # Raumwunsch (Huelle + Auslagen werden in emit_rooms gebaut)
    if mode != 'screen':
        side = spec.get('side', 0.45)
        yc = max(top + 0.25, y0 + 3.0)
        STATE.requests.append(dict(fr=fr, xa=x0 - side, xb=x1 + side, y0=y0, yc=yc, cf=gc, want=spec.get('depth', 3.0),
                                   kind=spec.get('kind', 'generic'), bays=[(x0, x1)], prio=spec.get('prio', 5),
                                   unit=spec.get('unit'), name=spec.get('name')))
    return gc


def _door(mb, fr, a0, a1, y0, yt, gc, both=False):
    """Ganzglastuer mit Stahlrahmen (Fluegelprofil 9 cm), Sockelblech, waagrechter Stossstange und Tuerschliesser."""
    st = 0.09
    c0 = gc + 0.002; c1 = gc + 0.06
    fz = 'xXyYzZ' if both else 'xXyYZ'
    # Fluegelrahmen (vier Profile)
    mb.box(LR, fr, a0, a0 + st, y0 + 0.005, yt, c0, c1, faces=fz)
    mb.box(LR, fr, a1 - st, a1, y0 + 0.005, yt, c0, c1, faces=fz)
    mb.box(LR, fr, a0 + st, a1 - st, yt - st, yt, c0, c1, faces=fz)
    mb.box(LR, fr, a0 + st, a1 - st, y0 + 0.005, y0 + 0.24, c0, c1, faces=fz)           # Sockelprofil
    # Stossstange (Edelstahl, waagrecht) auf Abstandshaltern, aussen; bei Glaswand auch innen
    for sgn, cc in ((1, c1),) + (((-1, c0 - 0.03),) if both else ()):
        yb = y0 + 1.02
        for x in (a0 + 0.18, a1 - 0.18):
            # Abstandshalter: alle Seiten (durch das Glas ist auch die Seite am Glas sichtbar)
            # (reichen bis an die Glasscheibe bei gc: sonst schwebt die Stange vor dem Glas)
            if sgn > 0:
                mb.box(TG, fr, x - 0.012, x + 0.012, yb - 0.012, yb + 0.012, gc + 0.002, cc + 0.06, faces='xXyYzZ')
            else:
                mb.box(TG, fr, x - 0.012, x + 0.012, yb - 0.012, yb + 0.012, cc - 0.06, gc - 0.002, faces='xXyYzZ')
        if sgn > 0:
            mb.box(TG, fr, a0 + 0.12, a1 - 0.12, yb - 0.02, yb + 0.02, cc + 0.06, cc + 0.10, faces='xXyYZz')
        else:
            mb.box(TG, fr, a0 + 0.12, a1 - 0.12, yb - 0.02, yb + 0.02, cc - 0.10, cc - 0.06, faces='xXyYZz')
    # senkrechter Stossgriff am Schlossprofil
    xh = a1 - st - 0.06
    mb.box(TG, fr, xh - 0.016, xh + 0.016, y0 + 0.85, y0 + 1.75, c1 + 0.05, c1 + 0.08, faces='xXyYZz')
    for yy in (y0 + 0.92, y0 + 1.68):
        mb.box(TG, fr, xh - 0.01, xh + 0.01, yy - 0.01, yy + 0.01, gc + 0.002, c1 + 0.05, faces='xXyYz')
    # Tuerschliesser (oben, innen sichtbar durch das Glas: als Kasten auf der Innenseite)
    mb.box(LR, fr, a0 + 0.12, a0 + 0.50, yt - st - 0.07, yt - st, gc - 0.06, gc - 0.002, faces='xXyYzZ')


# ============================================================================ Raeume
def plan_rect(fr, xa, xb, c0, c1):
    pts = [fr.P([xa, 0, c0]), fr.P([xb, 0, c0]), fr.P([xb, 0, c1]), fr.P([xa, 0, c1])]
    return Polygon([(p[0], p[2]) for p in pts]).buffer(0)


def emit_rooms(mb, allowed, obstacles=(), min_depth=0.7):
    """Raumwuensche zuteilen (nach Prioritaet), Huellen und Auslagen bauen. allowed: Plan-Polygon (x, z), in dem Raeume
    liegen duerfen; obstacles: Plan-Polygone, die frei bleiben muessen (Laubengang)."""
    reqs = sorted(STATE.requests, key=lambda r: r['prio'])
    # Laeden mit gleicher Einheit (unit) in derselben Wand zu einem Raum zusammenfassen
    merged = []
    for r in reqs:
        u = r.get('unit')
        m = next((m for m in merged if u is not None and m.get('unit') == u and m['fr'] is r['fr']), None)
        if m is None:
            merged.append(dict(r))
        else:
            m['xa'] = min(m['xa'], r['xa']); m['xb'] = max(m['xb'], r['xb']); m['yc'] = max(m['yc'], r['yc'])
            m['bays'] = m['bays'] + r['bays']; m['cf'] = max(m['cf'], r['cf'])
    obst = unary_union(list(obstacles)) if obstacles else None
    for r in merged:
        fr = r['fr']
        D = r['want']
        ok = False
        while D >= min_depth - 1e-6:
            pg = plan_rect(fr, r['xa'], r['xb'], r['cf'] - 0.02, r['cf'] - D)
            bad = (not allowed.buffer(0.02).contains(pg)) or (obst is not None and pg.intersection(obst).area > 0.01) or \
                any(pg.intersection(q).area > 0.01 for q in STATE.rooms)
            if not bad:
                ok = True; break
            D -= 0.15
        if not ok:
            # Raum seitlich auf die Glasbreite verkleinern und flacher (Schaukasten), bis nichts mehr schneidet
            r['xa'] = min(b[0] for b in r['bays']) - 0.06; r['xb'] = max(b[1] for b in r['bays']) + 0.06
            D = max(min_depth, r['want'])
            while D > 0.30:
                pg = plan_rect(fr, r['xa'], r['xb'], r['cf'] - 0.02, r['cf'] - D)
                pgb = pg.buffer(0.04, join_style=2)
                if allowed.buffer(0.02).contains(pg) and not (obst is not None and pgb.intersects(obst)) and                         not any(pgb.intersects(q) for q in STATE.rooms):
                    break
                D -= 0.05
            if D <= 0.30:
                STATE.log.append('WARNUNG: kein Platz fuer Ladenraum %s (%s) -- nur dunkle Rueckwand' % (r.get('name'), r['kind']))
                D = 0.12
            pg = plan_rect(fr, r['xa'], r['xb'], r['cf'] - 0.02, r['cf'] - D)
        STATE.rooms.append(pg)
        r['D'] = D
        STATE.log.append('Laden %-10s %-9s Breite %.2f Tiefe %.2f (gewuenscht %.2f)' % (r.get('name') or '-', r['kind'], r['xb'] - r['xa'], D, r['want']))
        _room(mb, r)


def _room(mb, r):
    fr = r['fr']; xa, xb, y0, yc, cf, D = r['xa'], r['xb'], r['y0'], r['yc'], r['cf'], r['D']
    kind = r['kind']
    wc, fc, cc, light = ROOM_STYLE.get(kind, ROOM_STYLE['generic'])
    cb = cf - D
    c_front = cf + 0.0
    old = mb.col
    mb.col = hex_lin(fc)
    mb.poly(LD, fr, [(xa, y0, c_front), (xb, y0, c_front), (xb, y0, cb), (xa, y0, cb)], hint=(0, 1, 0))
    mb.col = hex_lin(cc)
    mb.poly(LD, fr, [(xa, yc, c_front), (xb, yc, c_front), (xb, yc, cb), (xa, yc, cb)], hint=(0, -1, 0))
    mb.col = hex_lin(wc)
    mb.poly(LD, fr, [(xa, y0, cb), (xb, y0, cb), (xb, yc, cb), (xa, yc, cb)], hint=(0, 0, 1))
    mb.col = hex_lin(wc) * 0.92
    mb.poly(LD, fr, [(xa, y0, c_front), (xa, y0, cb), (xa, yc, cb), (xa, yc, c_front)], hint=(1, 0, 0))
    mb.poly(LD, fr, [(xb, y0, c_front), (xb, y0, cb), (xb, yc, cb), (xb, yc, c_front)], hint=(-1, 0, 0))
    # Sockelleiste (dunkel) an Rueck- und Seitenwaenden
    mb.col = hex_lin(fc) * 0.55
    mb.box(LD, fr, xa, xb, y0, y0 + 0.08, cb, cb + 0.015, faces='YZ')
    mb.col = old
    W = xb - xa; uc = (xa + xb) / 2
    # Deckenleuchten
    if light:
        _inst(mb, light, fr, uc, yc, cf - D / 2, (W / 4.0, 1.0, D / 4.0))
    # Auslagen hinter jedem Schaufenster (Bahnen = bays), Rueckwand-Moebel, Mittelmoebel
    rng = np.random.default_rng(abs(hash((round(float(fr.o[0]), 2), round(float(xa), 2)))) % (2 ** 31))
    sz = min(1.0, (D - 0.15) / 1.6)
    if sz > 0.25:
        for (b0, b1) in r['bays']:
            ww = b1 - b0; c = (b0 + b1) / 2
            sets = [(c, ww)] if ww < 3.4 else [(c - ww / 4, ww / 2), (c + ww / 4, ww / 2)]
            for (cu, cw) in sets:
                sx = float(np.clip(cw / 2.6, 0.7, 1.25))
                _inst(mb, 'dsp_%s_bay%d' % (kind, int(rng.integers(3))), fr, cu, y0, cf - 0.03, (sx, 1.0, sz))
    if kind == 'vacant':
        return
    if D >= 2.3:
        nbk = max(1, int(round(W / 4.0)))
        for i in range(nbk):
            cu = xa + W * (i + 0.5) / nbk
            _inst(mb, 'dsp_%s_back%d' % (kind, int(rng.integers(2))), fr, cu, y0, cb + 0.005, (W / nbk / 4.0, 1.0, float(min(1.0, (D - 1.6) / 1.0))))
    if D >= 3.4:
        nm = max(1, int(W / 2.8))
        for i in range(nm):
            cu = xa + W * (i + 0.5) / nm
            _inst(mb, 'dsp_%s_mid%d' % (kind, int(rng.integers(2))), fr, cu, y0, (cf - 1.75 + cb + 1.05) / 2, (1.0, 1.0, 1.0))


def _inst(mb, key, fr, a, b, c, s):
    f = fr.local(a, b, c)
    M = np.eye(4); M[:3, :3] = f.M @ np.diag(s); M[:3, 3] = f.o
    mb.instance(key, M)
    STATE.used.add(key)


def display_protos(tiles=None):
    """benutzte Auslagen-Prototypen der Fassaden-Pipeline -> {key: MB} (Material AR_dsp_<mat>, Vertexfarben)"""
    if FAC_DIR not in sys.path: sys.path.append(FAC_DIR)
    import displays                      # pipeline/facades/displays.py
    import gltf_io
    from ar_export import dsp_material
    out = {}
    for key in sorted(STATE.used):
        P = displays.get(key)
        mb = MB(tiles or {})
        for mat, (V, F, C) in P.mesh.items():
            if not len(F): continue
            mname = dsp_material(mat, gltf_io.MATERIALS.get(mat, {}))
            T = V[F].astype(float)
            n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            ln = np.linalg.norm(n, axis=1); ok = ln > 1e-12
            T = T[ok]; n = n[ok] / ln[ok, None]
            if 'glow' in mat:      # Leuchtflaechen liegen in den Prototypen koplanar auf Blech/Leisten -> 2 mm vorziehen
                T = T + n[:, None, :] * 0.002
            Cc = C[F][ok].astype(float)
            mb.tris_world(mname, T, np.repeat(n[:, None, :], 3, 1), np.zeros((len(T), 3, 2)), C=Cc)
        out[key] = mb
    return out


# ============================================================================ Schilder
def fascia(mb, fr, a, y0, name, h=0.42, c=0.06, w=None):
    """Blendschild (1024 x 192 px Textur) mittig bei a, Unterkante y0, Front bei c, mit Stahlrahmen."""
    w = w or h * 1024 / 192
    uv = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
    y1 = y0 + h
    mb.tri_local('AR_Schild_' + name, fr, np.array([[(a - w / 2, y0, c), (a + w / 2, y0, c), (a + w / 2, y1, c)],
                                                     [(a - w / 2, y0, c), (a + w / 2, y1, c), (a - w / 2, y1, c)]]), uv=uv)
    mb.box('AR_Schild_Halter', fr, a - w / 2 - 0.025, a + w / 2 + 0.025, y0 - 0.025, y1 + 0.025, 0.0, c - 0.002, faces='xXyY')
    mb.box('AR_Schild_Halter', fr, a - w / 2 - 0.025, a + w / 2 + 0.025, y1, y1 + 0.025, c - 0.002, c + 0.008, faces='xXyYZ')
    mb.box('AR_Schild_Halter', fr, a - w / 2 - 0.025, a + w / 2 + 0.025, y0 - 0.025, y0, c - 0.002, c + 0.008, faces='xXyYZ')
    mb.box('AR_Schild_Halter', fr, a - w / 2 - 0.025, a - w / 2, y0, y1, c - 0.002, c + 0.008, faces='xXZ')
    mb.box('AR_Schild_Halter', fr, a + w / 2, a + w / 2 + 0.025, y0, y1, c - 0.002, c + 0.008, faces='xXZ')
    return w


def blade_sign(mb, fr, a, y0, name, size=0.62, out0=0.16, thick=0.045):
    """beidseitiges Ausleger-/Nasenschild an der Wand (Fassadenframe fr) bei a, Unterkante y0, senkrecht zur Wand:
    Wandkonsole + Ausleger (Stahl), Schildkoerper (Rahmen), beide Seiten mit Textur blade_<name> (lesbar von beiden Seiten)."""
    f = fr.local(a, 0, 0)
    c0, c1 = out0, out0 + size
    yb, yt = y0, y0 + size
    t = thick / 2
    m = 'AR_Ausleger_' + name
    # +a-Seite: Betrachter auf +a sieht nach -a, seine Rechte = -c -> u = (c1 - c) / size
    uvp = lambda c, b: ((c1 - c) / size, (b - yb) / size)
    uvm = lambda c, b: ((c - c0) / size, (b - yb) / size)
    A, B, C, Dd = (t, yb, c1), (t, yb, c0), (t, yt, c0), (t, yt, c1)
    mb.tri_local(m, f, np.array([[A, B, C], [A, C, Dd]]), uv=np.array([[uvp(A[2], A[1]), uvp(B[2], B[1]), uvp(C[2], C[1])],
                                                                      [uvp(A[2], A[1]), uvp(C[2], C[1]), uvp(Dd[2], Dd[1])]]))
    A, B, C, Dd = (-t, yb, c0), (-t, yb, c1), (-t, yt, c1), (-t, yt, c0)
    mb.tri_local(m, f, np.array([[A, B, C], [A, C, Dd]]), uv=np.array([[uvm(A[2], A[1]), uvm(B[2], B[1]), uvm(C[2], C[1])],
                                                                      [uvm(A[2], A[1]), uvm(C[2], C[1]), uvm(Dd[2], Dd[1])]]))
    # Schildrahmen (Kanten)
    H = 'AR_Schild_Halter'
    mb.box(H, f, -t - 0.006, t + 0.006, yb - 0.02, yb, c0 - 0.02, c1 + 0.02, faces='xXyYzZ')
    mb.box(H, f, -t - 0.006, t + 0.006, yt, yt + 0.02, c0 - 0.02, c1 + 0.02, faces='xXyYzZ')
    mb.box(H, f, -t - 0.006, t + 0.006, yb, yt, c1, c1 + 0.02, faces='xXZ')
    mb.box(H, f, -t - 0.006, t + 0.006, yb, yt, c0 - 0.02, c0, faces='xXz')
    # Ausleger + Wandplatte + Abhaenger
    mb.box(H, f, -0.015, 0.015, yt + 0.10, yt + 0.14, 0.0, c1 + 0.05, faces='xXyYZ')
    mb.box(H, f, -0.08, 0.08, yt - 0.05, yt + 0.25, 0.0, 0.02, faces='xXyYZ')
    for cc in (c0 + 0.06, c1 - 0.06):
        mb.box(H, f, -0.008, 0.008, yt + 0.02, yt + 0.10, cc - 0.008, cc + 0.008, faces='xXzZ')
    # Strebe (schraeg) von der Wandplatte zum Ausleger
    s0 = np.array([0.0, yt - 0.04, 0.02]); s1 = np.array([0.0, yt + 0.10, c0 + 0.12])
    v = s1 - s0; L = float(np.linalg.norm(v)); v = v / L
    sf = Frame(f.P(s0), f.ex, f.D(v), np.cross(f.ex, f.D(v)))
    mb.box(H, sf, -0.012, 0.012, 0.0, L, -0.012, 0.012, faces='xXzZ')

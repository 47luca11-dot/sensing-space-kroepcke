"""Schriftzuege als Geometrie: Umrisse aus freien Schriften (DejaVu, mit matplotlib ausgeliefert, Bitstream-Vera-Lizenz)
-> Polygone (mit Loechern) -> extrudierte Buchstaben in einem lokalen Fassadenrahmen (u rechts, v oben, d aussen).

Normierung: Versalhoehe ('H') = 1.0, Grundlinie bei v = 0. Ergebnisse werden je (Text, Schrift) zwischengespeichert."""
import os
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from geom import tri2d

FONT_FILES = {
    'sans': 'DejaVuSans.ttf', 'bold-sans': 'DejaVuSans-Bold.ttf', 'serif': 'DejaVuSerif.ttf', 'serif-bold': 'DejaVuSerif-Bold.ttf',
    'script': 'DejaVuSerif-BoldItalic.ttf', 'condensed': 'DejaVuSans-Bold.ttf', 'italic': 'DejaVuSans-BoldOblique.ttf',
}
_CACHE = {}
_PROPS = {}


def _prop(font):
    if font not in _PROPS:
        import matplotlib
        from matplotlib.font_manager import FontProperties
        fn = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', FONT_FILES.get(font, 'DejaVuSans.ttf'))
        _PROPS[font] = FontProperties(fname=fn)
    return _PROPS[font]


def _cap(font):
    k = ('__cap__', font)
    if k not in _CACHE:
        from matplotlib.textpath import TextPath
        v = TextPath((0, 0), 'H', size=100, prop=_prop(font)).vertices
        _CACHE[k] = float(v[:, 1].max()) / 100.0
    return _CACHE[k]


def outline(text, font='sans', tol=0.012):
    """-> (liste [(aussen (N,2), [loecher])], (x0, y0, x1, y1)) in Versalhoehen-Einheiten. tol: Vereinfachung (Einheiten)."""
    key = (text, font, tol)
    if key in _CACHE:
        return _CACHE[key]
    from matplotlib.textpath import TextPath
    if not text or not text.strip():
        _CACHE[key] = ([], (0, 0, 0, 0)); return _CACHE[key]
    tp = TextPath((0, 0), text, size=100, prop=_prop(font))
    cap = _cap(font) * 100.0
    polys = [np.asarray(p, float) / cap for p in tp.to_polygons(closed_only=True) if len(p) >= 4]
    shp = []
    for p in polys:
        try:
            g = Polygon(p).buffer(0)
        except Exception:
            continue
        if g.is_empty or g.area < 1e-5:
            continue
        shp.append(g)
    # Verschachtelung: Ringe nach Flaeche, jeder Ring ist Loch, wenn er in einer ungeraden Zahl anderer Ringe liegt
    shp.sort(key=lambda g: -g.area)
    depth = []
    for i, g in enumerate(shp):
        rp = g.representative_point()
        depth.append(sum(1 for j in range(i) if shp[j].contains(rp)))
    out = []
    for i, g in enumerate(shp):
        if depth[i] % 2:
            continue
        holes = []
        for j in range(i + 1, len(shp)):
            if depth[j] == depth[i] + 1 and g.contains(shp[j].representative_point()):
                holes.append(shp[j])
        poly = g
        for h in holes:
            poly = poly.difference(h)
        for part in ([poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))):
            if part.geom_type != 'Polygon' or part.is_empty or part.area < 1e-5:
                continue
            part = part.simplify(tol, preserve_topology=True)
            if part.is_empty or part.geom_type != 'Polygon':
                continue
            part = orient(part, 1.0)
            ext = np.asarray(part.exterior.coords)[:-1]
            ints = [np.asarray(r.coords)[:-1] for r in part.interiors if len(r.coords) >= 4]
            if len(ext) >= 3:
                out.append((ext, ints))
    if out:
        allp = np.concatenate([e for e, _ in out])
        bb = (float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max()))
    else:
        bb = (0, 0, 0, 0)
    _CACHE[key] = (out, bb)
    return _CACHE[key]


def extrude_local(polys, d0, d1, back=False):
    """Polygone (lokale u,v) zwischen Tiefe d0 und d1 extrudieren. -> V (N,3) lokal, F (M,3), W (M,3) Soll-Normalen.
    Vorderseite bei d1 (Normale +d), Seiten nach aussen; Rueckseite nur wenn back."""
    Vs = []; Fs = []; Ws = []; nv = 0
    for ext, ints in polys:
        V2, F = tri2d(ext, ints)
        if len(F) == 0:
            continue
        Vf = np.c_[V2, np.full(len(V2), d1)]
        Vs.append(Vf); Fs.append(F + nv); Ws.append(np.tile([0, 0, 1.0], (len(F), 1))); nv += len(Vf)
        if back:
            Vb = np.c_[V2, np.full(len(V2), d0)]
            Vs.append(Vb); Fs.append(F + nv); Ws.append(np.tile([0, 0, -1.0], (len(F), 1))); nv += len(Vb)
        for ring in [ext] + list(ints):
            R = np.asarray(ring, float); n = len(R)
            if n < 2:
                continue
            A = R; B = np.roll(R, -1, 0)
            e = B - A
            nrm = np.c_[e[:, 1], -e[:, 0]]                      # rechts der Laufrichtung = aussen (Material links)
            ln = np.linalg.norm(nrm, axis=1, keepdims=True); ln[ln == 0] = 1; nrm /= ln
            V = np.concatenate([np.c_[A, np.full(n, d0)], np.c_[B, np.full(n, d0)], np.c_[B, np.full(n, d1)], np.c_[A, np.full(n, d1)]])
            i = np.arange(n)
            F2 = np.concatenate([np.c_[i, i + n, i + 2 * n], np.c_[i, i + 2 * n, i + 3 * n]]) + nv
            W2 = np.concatenate([np.c_[nrm, np.zeros(n)]] * 2)
            Vs.append(V); Fs.append(F2); Ws.append(W2); nv += len(V)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3))
    return np.concatenate(Vs), np.concatenate(Fs).astype(np.int64), np.concatenate(Ws)


def _place(bb, box_w, box_h, fill_h, fill_w, max_cap, align):
    tw = bb[2] - bb[0]; th = max(bb[3], 1.0) - min(bb[1], 0.0)
    s = min(box_h * fill_h / max(th, 1e-6), box_w * fill_w / max(tw, 1e-6))
    if max_cap:
        s = min(s, max_cap)
    w = tw * s
    x0 = (box_w - w) / 2 - bb[0] * s if align != 'left' else box_w * (1 - fill_w) / 2 - bb[0] * s
    y0 = (box_h - s) / 2
    if bb[1] < -0.05:                                       # Unterlaengen: etwas anheben
        y0 = (box_h - s * (1.0 - bb[1] * 0.6)) / 2 - bb[1] * s * 0.6
    return s, w, x0, y0


def layout(text, font, box_w, box_h, fill_h=0.62, fill_w=0.9, max_cap=None, align='center', tol_m=0.006):
    """Text in ein Feld (Breite box_w, Hoehe box_h, Ursprung links unten) einpassen.
    -> (polys in Feldkoordinaten, Versalhoehe in m, Textbreite in m)."""
    polys, bb = outline(text, font)
    if not polys:
        return [], 0.0, 0.0
    s, w, x0, y0 = _place(bb, box_w, box_h, fill_h, fill_w, max_cap, align)
    if s <= 0:
        return [], 0.0, 0.0
    tol = float(np.clip(tol_m / s, 0.008, 0.06))
    polys, _ = outline(text, font, round(tol, 3))
    return [(e * s + [x0, y0], [h * s + [x0, y0] for h in hs]) for e, hs in polys], s, w


def emit_text(acc, layer, mat, fr, text, font, u0, v0, box_w, box_h, d0, d1, color, fill_h=0.62, fill_w=0.9, max_cap=None,
              back=False, align='center', tol_m=0.006):
    """Text als extrudierte Buchstaben in den MeshAcc (Rahmen fr, Feld links unten bei (u0,v0)). -> (Breite, Versalhoehe) in m."""
    from facade import _oriented
    polys, s, w = layout(text, font, box_w, box_h, fill_h, fill_w, max_cap, align, tol_m)
    if not polys:
        return 0.0, 0.0
    V, F, W = extrude_local(polys, d0, d1, back=back)
    if len(F) == 0:
        return 0.0, 0.0
    V = V + np.array([u0, v0, 0.0])
    _oriented(acc, layer, mat, fr.w(V), F, W @ fr.M, color)
    return w, s

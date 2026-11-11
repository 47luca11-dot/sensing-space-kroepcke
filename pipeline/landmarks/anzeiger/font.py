"""Tiny condensed stroke font (1920s grotesque look) -> shapely polygons -> extruded 3D letters."""
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from shapely import affinity
import geo


def arc(cx, cy, rx, ry, a0, a1, n=14):
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]


W = 0.6
G = {
    'A': [[(0, 0), (0.3, 1), (0.6, 0)], [(0.12, 0.34), (0.48, 0.34)]],
    'B': [[(0, 0), (0, 1), (0.36, 1)] + arc(0.36, 0.76, 0.22, 0.24, 90, -90) + [(0, 0.52)], [(0, 0.52), (0.38, 0.52)] + arc(0.38, 0.26, 0.22, 0.26, 90, -90) + [(0, 0)]],
    'C': [arc(0.3, 0.5, 0.3, 0.5, 40, 320, 20)],
    'D': [[(0, 0), (0, 1), (0.22, 1)] + arc(0.22, 0.5, 0.38, 0.5, 90, -90, 16) + [(0, 0)]],
    'E': [[(0.6, 1), (0, 1), (0, 0), (0.6, 0)], [(0, 0.52), (0.5, 0.52)]],
    'F': [[(0.6, 1), (0, 1), (0, 0)], [(0, 0.52), (0.5, 0.52)]],
    'G': [arc(0.3, 0.5, 0.3, 0.5, 35, 340, 20), [(0.6, 0.0), (0.6, 0.45), (0.32, 0.45)]],
    'H': [[(0, 0), (0, 1)], [(0.6, 0), (0.6, 1)], [(0, 0.52), (0.6, 0.52)]],
    'I': [[(0, 0), (0, 1)]],
    'K': [[(0, 0), (0, 1)], [(0.6, 1), (0, 0.38)], [(0.2, 0.58), (0.6, 0)]],
    'L': [[(0, 1), (0, 0), (0.55, 0)]],
    'N': [[(0, 0), (0, 1), (0.6, 0), (0.6, 1)]],
    'O': [arc(0.3, 0.5, 0.3, 0.5, 0, 360, 28)],
    'P': [[(0, 0), (0, 1), (0.36, 1)] + arc(0.36, 0.74, 0.24, 0.26, 90, -90) + [(0, 0.48)]],
    'R': [[(0, 0), (0, 1), (0.36, 1)] + arc(0.36, 0.74, 0.24, 0.26, 90, -90) + [(0, 0.48)], [(0.3, 0.48), (0.6, 0)]],
    'S': [arc(0.3, 0.75, 0.28, 0.25, 15, 270, 14) + arc(0.3, 0.25, 0.3, 0.25, 90, -165, 14)],
    'T': [[(0, 1), (0.6, 1)], [(0.3, 1), (0.3, 0)]],
    'U': [[(0, 1), (0, 0.3)] + arc(0.3, 0.3, 0.3, 0.3, 180, 360, 14) + [(0.6, 1)]],
    'Z': [[(0, 1), (0.6, 1), (0, 0), (0.6, 0)]],
    '1': [[(0.1, 0.78), (0.35, 1), (0.35, 0)]],
    '2': [arc(0.3, 0.72, 0.28, 0.28, 160, -40, 12) + [(0, 0), (0.6, 0)]],
    'M': [[(0, 0), (0, 1), (0.3, 0.42), (0.6, 1), (0.6, 0)]],
    'Q': [arc(0.3, 0.5, 0.3, 0.5, 0, 360, 28), [(0.36, 0.22), (0.62, 0.0)]],
    # accented capitals: letter body squeezed to 0.8 cap height, accent inside the cap box (text is clipped there)
    'É': [[(0.55, 0.8), (0, 0.8), (0, 0), (0.55, 0)], [(0, 0.42), (0.45, 0.42)], [(0.2, 0.88), (0.4, 1.0)]],
    'È': [[(0.55, 0.8), (0, 0.8), (0, 0), (0.55, 0)], [(0, 0.42), (0.45, 0.42)], [(0.4, 0.88), (0.2, 1.0)]],
    'Ü': [[(0, 0.8), (0, 0.25)] + arc(0.3, 0.25, 0.3, 0.25, 180, 360, 14) + [(0.6, 0.8)], [(0.15, 0.9), (0.15, 1.0)], [(0.45, 0.9), (0.45, 1.0)]],
    '-': [[(0.08, 0.48), (0.42, 0.48)]],
    '.': [[(0.05, 0.0), (0.05, 0.08)]],
}
ADV = {'I': 0.25, '1': 0.5, '-': 0.55, '.': 0.2, ' ': 0.5}


def text_shapes(s, height, stroke=0.15, spacing=0.22, cond=1.0):
    """returns list of (char, shapely polygon) in (x, y) metres, baseline y=0, starting x=0; and total width"""
    out = []; x = 0.0
    sw = stroke * height
    for ch in s:
        if ch == ' ':
            x += ADV[' '] * height * cond; continue
        strokes = G.get(ch)
        if strokes is None:
            x += 0.6 * height; continue
        lines = [LineString([(px * height * cond, py * height) for px, py in st]) for st in strokes]
        shp = unary_union([l.buffer(sw / 2, cap_style=2, join_style=2, mitre_limit=2.5) for l in lines])
        # clip to cap height (square caps overshoot top/bottom)
        from shapely.geometry import box
        shp = shp.intersection(box(-sw, 0, 2 * height, height))
        eps = min(0.004, 0.015 * height)          # morphological closing: removes mm notches left by clipped square caps
        shp = shp.buffer(eps, join_style=2, mitre_limit=3.0).buffer(-eps, join_style=2, mitre_limit=3.0).intersection(box(-sw, 0, 2 * height, height)).simplify(eps * 0.1)
        shp = affinity.translate(shp, x + sw / 2, 0)
        out.append((ch, shp))
        adv = ADV.get(ch, W) * height * cond
        x += adv + sw + spacing * height
    return out, x - spacing * height


def extrude_text(mb, obj, F, s, s0, y0, e0, height, depth, mat_face, mat_side, stroke=0.15, spacing=0.22, cond=1.0,
                 align='left', back=False, vertical=False, pitch=None):
    """3D letters in facade frame F (s along, y up, e out): front face at e0+depth.
    vertical=True stacks the characters top-down (letter centres every `pitch`), s0 = column centre, y0 = top."""
    shapes, width = text_shapes(s, height, stroke, spacing, cond)
    items = []
    if vertical:
        for i, (ch, shp) in enumerate(shapes):
            minx, miny, maxx, maxy = shp.bounds
            dx = s0 - (minx + maxx) / 2; dy = y0 - i * pitch - height
            items.append(affinity.translate(shp, dx, dy))
    else:
        off = {'left': 0, 'center': -width / 2, 'right': -width}[align]
        items = [affinity.translate(shp, s0 + off, y0) for ch, shp in shapes]
    for shp in items:
        geoms = list(shp.geoms) if shp.geom_type == 'MultiPolygon' else [shp]
        for g in geoms:
            if g.is_empty or g.area < 1e-6: continue
            ext = np.array(g.exterior.coords)[:-1]
            holes = [np.array(h.coords)[:-1] for h in g.interiors]
            P = lambda pts, e: [F.w(p[0], p[1], e) for p in pts]
            mb.poly(obj, mat_face, P(ext, e0 + depth), F.n, holes=[P(h, e0 + depth) for h in holes])
            if back:
                mb.poly(obj, mat_side, P(ext, e0), -F.n, holes=[P(h, e0) for h in holes])
            for ring, is_hole in [(ext, False)] + [(h, True) for h in holes]:
                r = np.asarray(ring)
                # ring orientation in (s,y): exterior should be CCW for outward = right-hand rule
                area = 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])
                for i in range(len(r)):
                    a, b = r[i], r[(i + 1) % len(r)]
                    d = b - a
                    if np.hypot(*d) < 1e-6: continue
                    o2 = np.array([d[1], -d[0]])          # right of edge direction
                    if (area > 0) == (not is_hole): pass
                    # outward of the solid: for CCW exterior the right side is outside; for holes the inside of the hole
                    if (area < 0 and not is_hole) or (area > 0 and is_hole): o2 = -o2
                    # make sure for holes the normal points into the hole (away from solid): handled by orientation rule
                    hint = F.s * o2[0] + geo.UP * o2[1]
                    mb.quad(obj, mat_side, F.w(a[0], a[1], e0), F.w(b[0], b[1], e0), F.w(b[0], b[1], e0 + depth), F.w(a[0], a[1], e0 + depth), hint)
    return width

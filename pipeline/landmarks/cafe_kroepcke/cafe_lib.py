"""Geometrie-Bibliothek fuer das Cafe Kroepcke (Blender 5, headless).

Konventionen
  * Planpunkte im Spielframe: (x Ost, z Sued) als 2-Tupel; Hoehen y (Spiel-Y, 0 = Strassenniveau Kroepcke-Uhr).
  * Blender-Frame: X = x, Y = -z, Z = y   (glTF-Export mit export_yup=True ergibt wieder den Spielframe).
  * Geometrie wird in "Buildern" (Vertex-/Face-Listen je Material + Gruppe) gesammelt und erst am Ende zu wenigen
    Objekten verbaut. Wiederholte Moebel/Leuchten werden als verlinkte Duplikate (Instanzen) angelegt.
"""
import bpy, bmesh, math
from mathutils import Vector, Matrix, noise

UP = Vector((0, 0, 1))


def W(p, y=0.0):
    """Planpunkt (x, z) + Hoehe y -> Blender-Vektor"""
    return Vector((p[0], -p[1], y))


def D3(d):
    """Planrichtung (dx, dz) -> Blender-Richtung"""
    return Vector((d[0], -d[1], 0.0))


def v2(a):
    return Vector((a[0], a[1]))


def lerp2(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def norm2(d):
    l = math.hypot(d[0], d[1]); return (d[0] / l, d[1] / l)


def sub2(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add2(a, b, s=1.0):
    return (a[0] + b[0] * s, a[1] + b[1] * s)


def dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def line_isect(p, d, q, e):
    """Schnitt der Geraden p + t d und q + s e (2D)"""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-12: return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + d[0] * t, p[1] + d[1] * t)


def offset_line(p, q, toward, dd):
    """Gerade p->q um dd senkrecht in Richtung des Punkts 'toward' verschieben"""
    d = norm2(sub2(q, p)); n = (-d[1], d[0])
    if dot2(n, sub2(toward, p)) < 0: n = (-n[0], -n[1])
    return add2(p, n, dd), add2(q, n, dd)


def point_in_poly(pt, poly):
    x, y = pt; ins = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x: ins = not ins
    return ins


# ------------------------------------------------------------------ Materialien
def srgb2lin(c):
    c = [x / 255.0 for x in c]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def make_mat(name, rgb, metal=0.0, rough=0.5, alpha=None, emit=None, emit_s=0.0, vcol=False, double=False, coat=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (*srgb2lin(rgb), 1)
    b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    if coat > 0:
        try: b.inputs['Coat Weight'].default_value = coat
        except Exception: pass
    if vcol:
        a = nt.nodes.new('ShaderNodeVertexColor'); a.layer_name = 'Color'
        nt.links.new(a.outputs['Color'], b.inputs['Base Color'])
    if alpha is not None:
        b.inputs['Alpha'].default_value = alpha
        try: m.surface_render_method = 'BLENDED'
        except Exception: pass
    if emit is not None:
        b.inputs['Emission Color'].default_value = (*srgb2lin(emit), 1)
        b.inputs['Emission Strength'].default_value = emit_s
    m.use_backface_culling = not double
    m.diffuse_color = (*srgb2lin(rgb), 1)
    return m


# ------------------------------------------------------------------ Builder
class Builder:
    """Sammelt Dreiecke/Polygone (Blender-Koordinaten) fuer ein Material + Exportgruppe."""

    def __init__(self, name, mat, smooth_angle=35.0, vcol=False):
        self.name = name; self.mat = mat; self.v = []; self.f = []; self.smooth_angle = smooth_angle
        self.vcol = vcol; self.col = []

    def vert(self, p, c=None):
        self.v.append(tuple(p))
        if self.vcol: self.col.append(c if c is not None else (1, 1, 1, 1))
        return len(self.v) - 1

    def face(self, idx):
        if len(set(idx)) >= 3: self.f.append(list(idx))

    def poly(self, pts, flip=False):
        ids = [self.vert(p) for p in pts]
        self.face(ids[::-1] if flip else ids)

    def add_mesh(self, verts, faces, mat=None):
        base = len(self.v)
        for p in verts: self.vert(p)
        for f in faces: self.face([i + base for i in f])

    def build(self, coll, weld=0.0):
        if not self.f: return None
        me = bpy.data.meshes.new(self.name)
        me.from_pydata(self.v, [], self.f)
        me.validate(clean_customdata=False)
        if self.vcol and self.col:
            ca = me.color_attributes.new('Color', 'FLOAT_COLOR', 'POINT')
            for i, c in enumerate(self.col[:len(me.vertices)]): ca.data[i].color = c
        if weld > 0:
            bm = bmesh.new(); bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
            bm.to_mesh(me); bm.free()
        for p in me.polygons: p.use_smooth = True
        try: me.set_sharp_from_angle(angle=math.radians(self.smooth_angle))
        except Exception: pass
        me.materials.append(self.mat)
        o = bpy.data.objects.new(self.name, me); coll.objects.link(o)
        return o


def newell(pts):
    n = Vector((0, 0, 0))
    for i in range(len(pts)):
        a = Vector(pts[i]); b = Vector(pts[(i + 1) % len(pts)])
        n.x += (a.y - b.y) * (a.z + b.z); n.y += (a.z - b.z) * (a.x + b.x); n.z += (a.x - b.x) * (a.y + b.y)
    return n


# ------------------------------------------------------------------ Grundkoerper (direkt in einen Builder)
def prism(B, poly3, h0, h1, caps=True, bottom=True):
    """Vertikales Prisma ueber Blender-XY-Polygon poly3 (Liste Vector/Tupel, z ignoriert) von h0 bis h1."""
    n = len(poly3)
    pts = [Vector((p[0], p[1], 0)) for p in poly3]
    # Umlaufsinn: CCW (von oben) erzwingen
    area = sum(pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y for i in range(n))
    if area < 0: pts = pts[::-1]
    lo = [B.vert((p.x, p.y, h0)) for p in pts]; hi = [B.vert((p.x, p.y, h1)) for p in pts]
    for i in range(n):
        j = (i + 1) % n
        B.face([lo[i], lo[j], hi[j], hi[i]])
    if caps:
        B.face(hi)
        if bottom: B.face(lo[::-1])


def obox(B, c, ax, ay, sx, sy, z0, z1, ch=0.0):
    """Orientierter Quader: Mittelpunkt c (Blender, z ignoriert), Achsen ax/ay (Blender-Horizontalvektoren), Masse sx, sy,
    Hoehe z0..z1; ch = Fase der vertikalen Kanten."""
    ax = Vector((ax[0], ax[1], 0)).normalized(); ay = Vector((ay[0], ay[1], 0)).normalized()
    c = Vector((c[0], c[1], 0)); hx, hy = sx / 2, sy / 2
    if ch > 0 and ch < min(hx, hy):
        loc = [(hx - ch, -hy), (hx, -hy + ch), (hx, hy - ch), (hx - ch, hy), (-hx + ch, hy), (-hx, hy - ch), (-hx, -hy + ch), (-hx + ch, -hy)]
    else:
        loc = [(hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
    poly = [c + ax * a + ay * b for a, b in loc]
    prism(B, poly, z0, z1)


def box3(B, p0, p1, w, h, upv=None):
    """Balken von p0 nach p1 (Blender-Vektoren) mit Querschnitt w (horizontal quer) x h (senkrecht zur Achse)."""
    p0 = Vector(p0); p1 = Vector(p1); d = (p1 - p0)
    L = d.length
    if L < 1e-6: return
    d.normalize()
    up = Vector(upv) if upv is not None else UP
    s = d.cross(up)
    if s.length < 1e-6: s = d.cross(Vector((1, 0, 0)))
    s.normalize(); t = d.cross(s).normalized()
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    a = [B.vert(p0 + s * x + t * y) for x, y in corners]
    b = [B.vert(p1 + s * x + t * y) for x, y in corners]
    for i in range(4):
        j = (i + 1) % 4
        B.face([a[i], a[j], b[j], b[i]])
    B.face(a[::-1]); B.face(b)


def tube(B, pts, r, seg=8, cap=True, closed=False):
    """Rohr entlang Polylinie pts (Blender-Vektoren) mit Radius r (paralleler Transport des Rahmens)."""
    pts = [Vector(p) for p in pts]
    if len(pts) < 2: return
    rings = []
    prev_n = None
    for i, p in enumerate(pts):
        if i == 0: t = (pts[1] - pts[0])
        elif i == len(pts) - 1: t = (pts[-1] - pts[-2])
        else: t = (pts[i + 1] - pts[i - 1])
        t.normalize()
        if prev_n is None:
            ref = UP if abs(t.dot(UP)) < 0.9 else Vector((1, 0, 0))
            n = t.cross(ref).normalized()
        else:
            n = (prev_n - t * prev_n.dot(t)).normalized()
        bvec = t.cross(n).normalized(); prev_n = n
        rings.append([B.vert(p + (n * math.cos(2 * math.pi * k / seg) + bvec * math.sin(2 * math.pi * k / seg)) * r) for k in range(seg)])
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(seg):
            l = (k + 1) % seg
            B.face([a[k], a[l], b[l], b[k]])
    if closed:
        a, b = rings[-1], rings[0]
        for k in range(seg):
            l = (k + 1) % seg; B.face([a[k], a[l], b[l], b[k]])
    elif cap:
        B.face(rings[0][::-1]); B.face(rings[-1])


def cylinder(B, p0, p1, r, seg=12, r1=None, cap=True):
    p0 = Vector(p0); p1 = Vector(p1); t = (p1 - p0).normalized()
    ref = UP if abs(t.dot(UP)) < 0.9 else Vector((1, 0, 0))
    n = t.cross(ref).normalized(); bvec = t.cross(n).normalized()
    r1 = r if r1 is None else r1
    a = [B.vert(p0 + (n * math.cos(2 * math.pi * k / seg) + bvec * math.sin(2 * math.pi * k / seg)) * r) for k in range(seg)]
    b = [B.vert(p1 + (n * math.cos(2 * math.pi * k / seg) + bvec * math.sin(2 * math.pi * k / seg)) * r1) for k in range(seg)]
    for k in range(seg):
        l = (k + 1) % seg; B.face([a[k], a[l], b[l], b[k]])
    if cap:
        B.face(a[::-1]); B.face(b)


def sphere(B, c, r, seg=16, rings=10, sz=1.0):
    c = Vector(c); top = B.vert(c + Vector((0, 0, r * sz))); bot = B.vert(c - Vector((0, 0, r * sz)))
    rs = []
    for i in range(1, rings):
        ph = math.pi * i / rings
        rs.append([B.vert(c + Vector((r * math.sin(ph) * math.cos(2 * math.pi * k / seg), r * math.sin(ph) * math.sin(2 * math.pi * k / seg), r * sz * math.cos(ph)))) for k in range(seg)])
    for k in range(seg):
        l = (k + 1) % seg
        B.face([top, rs[0][k], rs[0][l]])
        B.face([bot, rs[-1][l], rs[-1][k]])
    for a, b in zip(rs[:-1], rs[1:]):
        for k in range(seg):
            l = (k + 1) % seg; B.face([a[k], b[k], b[l], a[l]])


def sweep(B, stations, profile, closed_profile=True, caps=True):
    """stations: Liste von (origin, ax1, ax2) -- Profilpunkt (a, b) -> origin + ax1*a + ax2*b.
    Umlaufsinn wird automatisch so gewaehlt, dass die Mantelflaechen nach aussen zeigen."""
    if len(stations) < 2: return None
    area = sum(profile[i][0] * profile[(i + 1) % len(profile)][1] - profile[(i + 1) % len(profile)][0] * profile[i][1] for i in range(len(profile)))
    k = 1
    T = Vector(stations[k][0]) - Vector(stations[0][0])
    while T.length < 1e-7 and k < len(stations) - 1:
        k += 1; T = Vector(stations[k][0]) - Vector(stations[0][0])
    o0, a1, a2 = stations[0]
    cr = Vector(a1).cross(Vector(a2)).dot(T)
    if (area > 0) != (cr > 0): profile = profile[::-1]
    rings = [[B.vert(Vector(o) + Vector(x1) * a + Vector(x2) * b) for a, b in profile] for o, x1, x2 in stations]
    n = len(profile); m = n if closed_profile else n - 1
    for r0, r1 in zip(rings[:-1], rings[1:]):
        for k in range(m):
            l = (k + 1) % n
            B.face([r0[k], r0[l], r1[l], r1[k]])
    if caps and closed_profile:
        B.face(rings[0][::-1]); B.face(rings[-1])
    return rings


def orient_faces(B, start, expect, sample=None):
    """Alle Faces ab Index start umdrehen, falls die Normale einer Stichprobe (Index sample) nicht in Richtung expect zeigt."""
    if start >= len(B.f): return
    i = sample if sample is not None else start
    f = B.f[i]
    n = newell([B.v[k] for k in f])
    if n.dot(Vector(expect)) < 0:
        for k in range(start, len(B.f)): B.f[k] = B.f[k][::-1]


def polygon_cap(B, pts3, expect):
    ids = [B.vert(p) for p in pts3]
    n = newell(pts3)
    B.face(ids if n.dot(Vector(expect)) >= 0 else ids[::-1])


# ------------------------------------------------------------------ Text (Schrift als Geometrie)
def text_obj(coll, name, body, font_path, size, extrude, mat, origin, xdir, ndir, align='CENTER', shear=0.0, bevel=0.0, res=None, space=1.0):
    """Schriftzug: liegt in der Ebene (xdir, UP), Vorderseite zeigt nach ndir; origin = Grundlinie Mitte (Blender)."""
    cu = bpy.data.curves.new(name, 'FONT'); cu.body = body
    try: cu.font = bpy.data.fonts.load(font_path, check_existing=True)
    except Exception: pass
    cu.size = size; cu.extrude = extrude; cu.align_x = align; cu.align_y = 'BOTTOM_BASELINE'
    cu.shear = shear; cu.bevel_depth = bevel; cu.bevel_resolution = 1; cu.resolution_u = res if res else (2 if size < 0.3 else 3); cu.space_character = space
    o = bpy.data.objects.new(name, cu); coll.objects.link(o)
    o.data.materials.append(mat)
    Z = Vector(ndir).normalized(); X = UP.cross(Z).normalized() if xdir is None else Vector(xdir).normalized()
    Y = Z.cross(X).normalized()
    M = Matrix((X, Y, Z)).transposed().to_4x4(); M.translation = Vector(origin)
    o.matrix_world = M
    return o

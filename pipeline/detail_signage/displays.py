"""Display-window interiors fitted into the free box behind each shopfront glass.

Box (local frame): x in [a, b], y in [c, d] (c = glass bottom, d = glass top), z in [zb, zf]
(zf = just behind the glass, zb = in front of the first existing triangle).  The shell (floor, ceiling,
back wall, side walls; all facing into the box = towards the viewer) closes the box, every item is
checked against the box bounds on commit (dropped if it would poke out)."""
import math

import numpy as np

from common import hex2rgb, luminance, rgb2hex

E = 0.004          # inset of items from the shell
LAYER = 0.004      # minimum spacing of stacked flat layers (decal on panel, print on poster): no z-fighting at
#                    street distances with the game camera (near 0.05 m, 24-bit depth: ~1 mm at 30 m, ~3 mm at 50 m)
REVEAL = "#34363a"  # outer faces of the display shells (seen past a shell edge through the glass: dark reveal)


def clip_poly(P, cx_, cz_, k):
    """Sutherland-Hodgman: polygon P [(x, z)] clipped to cx*x + cz*z <= k."""
    out = []
    n = len(P)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]
        fp = cx_ * p[0] + cz_ * p[1] - k
        fq = cx_ * q[0] + cz_ * q[1] - k
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


class Ctx:
    def __init__(self, B, O, R, box, rng, shop=None, opening=None):
        self.B, self.O, self.R, self.rng = B, np.asarray(O), np.asarray(R), rng
        a, b, c, d = box["rect"]
        self.a, self.b, self.c, self.d = a, b, c, d
        self.zf, self.zb = box["z_front"], box["z_back"]
        self.W, self.H, self.D = b - a, d - c, self.zf - self.zb
        self.shop = shop or {}
        self.op = opening or {}
        sg = (self.shop.get("sign") or {})
        self.fg = sg.get("fg") or "#1a1a1a"
        self.bg = sg.get("bg") or "#f2f2f2"
        self.accent = self.bg if luminance(hex2rgb(self.bg)) < 0.85 else self.fg
        self.lo = np.array([a + E * 0.5, c, self.zb + E * 0.5])
        self.hi = np.array([b - E * 0.5, d - E * 0.5, self.zf - 0.001])
        # mitre half-planes (openings.resolve_overlaps): cx*x + cz*z <= k; items are kept E/2 inside them
        self.clip = [tuple(c_) for c_ in (box.get("clip") or [])]
        self.planes = [(c_[0], c_[1], c_[2] - E * 0.5 * math.hypot(c_[0], c_[1])) for c_ in self.clip]
        P = [(a, self.zf), (b, self.zf), (b, self.zb), (a, self.zb)]
        for c_ in self.clip:
            P = clip_poly(P, *c_)
        self.plan = P              # (x, z) polygon of the box floor plan, CCW seen from above
        self.logo = False          # set by build.py: this window shows the shop's name on its back wall
        self.reserved = []         # back-wall rects (x0, x1, y0, y1) already used (logo panel)

    def free_back(self, x0, x1, y0, y1):
        for r in self.reserved:
            if x0 < r[1] and x1 > r[0] and y0 < r[3] and y1 > r[2]:
                return False
        return True

    def xr(self, z, pad=0.0):
        """x range of the (clipped) box plan at depth z (pad inside)."""
        if not self.clip:
            return self.a + pad, self.b - pad
        xs = []
        P = self.plan
        for i in range(len(P)):
            p, q = P[i], P[(i + 1) % len(P)]
            if (p[1] - z) * (q[1] - z) <= 0 and abs(q[1] - p[1]) > 1e-12:
                xs.append(p[0] + (q[0] - p[0]) * (z - p[1]) / (q[1] - p[1]))
            elif abs(p[1] - z) < 1e-12:
                xs.append(p[0])
        if not xs:
            return self.a + pad, self.a + pad
        xa, xb = min(xs), max(xs)
        # the mitre is slanted: stay pad away from it measured perpendicular
        return xa + pad, xb - pad

    def commit(self):
        return self.B.commit(self.O, self.R, bounds=(self.lo, self.hi), planes=self.planes)

    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))


def mix(c1, c2, t):
    return rgb2hex(hex2rgb(c1) * (1 - t) + hex2rgb(c2) * t)


def shade(c, f):
    return rgb2hex(np.clip(hex2rgb(c) * f, 0, 1))


SKIN = ["#f1f0ec", "#e9e7e2", "#2b2b2d", "#d9c3a5", "#c8c4bc"]
FABRIC = ["#1f2a44", "#2e2e30", "#7a1e2c", "#c9b79c", "#ececea", "#4a6b8a", "#8a9a5b", "#b5651d", "#d8c3a5",
          "#3c5a40", "#9b2335", "#e3d6c4", "#5b5e66", "#c43c3c", "#e8b4b8", "#30475e", "#f2c14e", "#7b6d8d"]
WOOD = ["#c8a77a", "#a57a4f", "#d9bf98", "#8a5a3b", "#e2cfb0"]
FLOOR = ["#cfc8bd", "#bdb5a8", "#9c968e", "#e4e0d8", "#7d7770", "#c8a77a"]
WALL = ["#f4f2ec", "#ebe6dc", "#e6e8ea", "#f1ede4", "#dcd8d0"]


# ----------------------------------------------------------------------------- shell + light
def shell(cx, floor=None, wall=None, back=None, ceil="#f3f3f1", glow_back=True, outer=True):
    """closed display box: floor, ceiling, back wall, side walls (and the mitre wall of a clipped corner box),
    all facing into the box; the open side is the glass.  outer=True also closes the box from the outside
    (dark reveal faces): through the glass strip beside / above a box, or through the neighbouring glass of a
    corner, a viewer never looks at the back of a shell face (FrontSide rendering would show a hole)."""
    B = cx.B
    c, d, zb, zf = cx.c, cx.d, cx.zb, cx.zf
    floor = floor or cx.pick(FLOOR)
    wall = wall or cx.pick(WALL)
    back = back or wall
    P = cx.plan
    # walls, floor and ceiling run on to 2 mm behind the pane where the front sweep of openings.free_box found
    # the zone right behind the glass empty (box front = initial zg - 12 mm): no 12 mm slot between pane and
    # box through which a grazing view could pass beside the box into the open building shell
    zg = (cx.op.get("glass") or {}).get("zg")
    if zg is not None and abs(zf - (zg - 0.012)) < 2e-4 and len(P) >= 3:
        zf = zg - 0.002
        P = [(cx.a, zf), (cx.b, zf), (cx.b, zb), (cx.a, zb)]
        for c_ in cx.clip:
            P = clip_poly(P, *c_)
    if len(P) < 3:
        return floor, wall
    xs = np.array([p[0] for p in P])
    zs = np.array([p[1] for p in P])
    cen = np.array([xs.mean(), zs.mean()])
    B.poly("matte", floor, [(x, c, z) for x, z in P], (0, 1, 0))
    B.poly("glow", ceil, [(x, d, z) for x, z in P], (0, -1, 0))
    if outer:
        B.poly("matte", REVEAL, [(x, c, z) for x, z in P], (0, -1, 0))
        B.poly("matte", REVEAL, [(x, d, z) for x, z in P], (0, 1, 0))
    n = len(P)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]
        if abs(p[1] - zf) < 1e-7 and abs(q[1] - zf) < 1e-7:
            continue                                   # the glass side stays open
        if math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-6:
            continue
        inward = np.array([-(q[1] - p[1]), 0.0, (q[0] - p[0])])
        if inward[[0, 2]] @ (cen - np.array(p)) < 0:
            inward = -inward
        is_back = abs(p[1] - zb) < 1e-7 and abs(q[1] - zb) < 1e-7
        mat = ("glow" if glow_back else "matte") if is_back else "glow"
        B.quad(mat, back if is_back else wall, (p[0], c, p[1]), (q[0], c, q[1]), (q[0], d, q[1]), (p[0], d, p[1]),
               expect=inward)
        if outer:
            B.quad("matte", REVEAL, (p[0], c, p[1]), (q[0], c, q[1]), (q[0], d, q[1]), (p[0], d, p[1]), expect=-inward)
    B.commit(cx.O, cx.R)
    # skirting board along the back wall (reads as a real room corner through the glass)
    if cx.D > 0.12:
        xa, xb = cx.xr(zb + 0.009, pad=0.004 if cx.clip else 0.0)
        if xb - xa > 0.1:
            B.box("matte", shade(wall, 0.82), (xa, c, zb), (xb, c + 0.06, zb + 0.008), skip=("-z", "-y", "-x", "+x"))
            B.commit(cx.O, cx.R, planes=cx.clip)
    if cx.logo:
        back_logo(cx)
    return floor, wall


def back_logo(cx):
    """the shop's name on a brand-coloured panel high on the back wall (typical view through a shop window)."""
    import signs
    text = signs.sign_text(cx.shop)
    if not text or cx.H < 2.35 or cx.D < 0.2:
        return
    font = (cx.shop.get("sign") or {}).get("font") or "sans"
    pw = min(cx.W - 0.3, 1.8)
    ph = 0.34
    if pw < 0.6:
        return
    y1 = cx.d - 0.1
    y0 = y1 - ph
    if y0 < cx.c + 1.98:
        return
    xc = (cx.a + cx.b) / 2
    res = signs.fit_text(text, font, pw - 0.16, ph * 0.9, cap_max=0.17, cap_min=0.045)
    if not res:
        return
    lines, cap = res
    bg = cx.bg
    fg = signs.text_colour(cx.shop.get("sign") or {}, cx.fg, bg)
    z0 = cx.zb + 0.004
    cx.B.box("matte", bg, (xc - pw / 2, y0, z0), (xc + pw / 2, y1, z0 + 0.02), skip=("-z",))
    zt = z0 + 0.02
    nl = len(lines)
    gap = cap * 0.45
    tot = nl * cap + (nl - 1) * gap
    yb = y0 + (ph - tot) / 2 + (nl - 1) * (cap + gap)
    for ln in lines:
        signs.place_text(cx.B, ln, font, cap, (xc, yb, zt), (0, 0, 1), depth=0.006, mat="lit", col=fg,
                         side_col=shade(fg, 0.6))
        yb -= cap + gap
    if cx.commit():
        cx.reserved.append((xc - pw / 2 - 0.05, xc + pw / 2 + 0.05, y0 - 0.05, y1 + 0.05))


def ceiling_lights(cx, kind="spots", warm=False):
    B = cx.B
    col = "#fff4dc" if warm else "#ffffff"
    y = cx.d - 0.002
    zc = cx.zf - min(0.18, cx.D * 0.4)
    if kind == "strip" or cx.D < 0.3 or cx.W < 0.5:
        z0 = cx.zf - min(0.12, cx.D * 0.45)
        xa, xb = cx.xr(z0 + 0.025, pad=0.05)
        xa2, xb2 = cx.xr(z0 - 0.025, pad=0.05)
        xa, xb = max(xa, xa2), min(xb, xb2)
        if xb - xa < 0.08:
            return
        B.box("metal", "#cfd2d4", (xa, y - 0.03, z0 - 0.025), (xb, y, z0 + 0.025), skip=("+y",))
        # lit diffuser 4 mm below the housing (no coplanar overlap with its bottom face)
        B.quad("lit", col, (xa + 0.02, y - 0.034, z0 - 0.018), (xb - 0.02, y - 0.034, z0 - 0.018),
               (xb - 0.02, y - 0.034, z0 + 0.018), (xa + 0.02, y - 0.034, z0 + 0.018), expect=np.array([0, -1.0, 0]))
        cx.commit()
        return
    n = max(1, int(cx.W / 0.7))
    for i in range(n):
        x = cx.a + cx.W * (i + 0.5) / n
        if kind == "track":
            # spot head on a short stem, aimed down/back
            B.cyl("metal", "#2a2a2c", x, zc, 0.012, y - 0.10, y, k=6, cap_top=False)   # stem end hidden at the ceiling
            # closed head (bottom cap) + a 5 mm lens puck under it: seen from below no open can shows its inside
            B.cyl("metal", "#2a2a2c", x, zc, 0.04, y - 0.20, y - 0.10, k=8, cap_top=True, cap_bottom=True)
            B.cyl("lit", col, x, zc, 0.032, y - 0.205, y - 0.20, k=8, cap_top=False, cap_bottom=True)
        else:
            B.box("metal", "#e8e8e6", (x - 0.07, y - 0.012, zc - 0.07), (x + 0.07, y, zc + 0.07), skip=("+y",))
            B.quad("lit", col, (x - 0.05, y - 0.016, zc - 0.05), (x + 0.05, y - 0.016, zc - 0.05),
                   (x + 0.05, y - 0.016, zc + 0.05), (x - 0.05, y - 0.016, zc + 0.05), expect=np.array([0, -1.0, 0]))
        cx.commit()
    if kind == "track":
        xa, xb = cx.xr(zc, pad=0.04)
        xa2, xb2 = cx.xr(zc - 0.015, pad=0.04)
        xa, xb = max(xa, xa2), min(xb, xb2)
        xa3, xb3 = cx.xr(zc + 0.015, pad=0.04)
        xa, xb = max(xa, xa3), min(xb, xb3)
        if xb - xa > 0.1:
            B.box("metal", "#2a2a2c", (xa, y - 0.025, zc - 0.015), (xb, y, zc + 0.015), skip=("+y",))
            cx.commit()


def pendant(cx, x, z, drop=0.6, col="#fff1c9", shade_col="#2b2b2b", r=0.14):
    B = cx.B
    y = cx.d
    yb = max(cx.c + 1.9, y - drop)
    if yb >= y - 0.1:
        return
    B.box("matte", "#111111", (x - 0.003, yb + 0.12, z - 0.003), (x + 0.003, y - 0.003, z + 0.003), skip=("+y", "-y"))
    # shade with its inside (light, warm) - seen from below past the diffuser
    B.frustum("gloss", shade_col, x, z, r, 0.03, yb, yb + 0.12, k=10, cap_top=True, inner=shade("#fff1c9", 0.9))
    B.disc("lit", col, (x, yb + 0.004, z), r * 0.8, (0, -1, 0), k=10)
    cx.commit()


def poster(cx, x0, x1, y0, y1, z, cols=None, lit=False, style=None):
    """flat printed poster / screen facing +z at depth z (abstract composition in brand colours)."""
    B = cx.B
    cx.rng.random()
    if z < cx.zb + 0.05 and not cx.free_back(x0, x1, y0, y1):
        return
    bgc, f1, f2 = cols or (cx.bg, cx.fg, "#ffffff")
    m = "lit" if lit else "glow"
    w, h = x1 - x0, y1 - y0
    st = style or cx.pick(["circle", "band", "photo", "bars"])
    z1 = z + LAYER                      # prints on the poster: one layer in front of its background
    if st == "photo":
        # colour campaign photo (no placeholder icon): studio backdrop in two tones, floor band, a model in the
        # brand colour with skin-tone head and arms; caption lines at the bottom
        tone = cx.pick(["#d9cfc2", "#c9d6df", "#e6d3c8", "#cfd8c4", "#e3dccf", "#bfc9d6"])
        if luminance(hex2rgb(bgc)) < 0.85 and cx.rng.random() < 0.5:
            tone = mix(bgc, "#ffffff", 0.55)
        B.rect_z(m, tone, x0, y0 + h * 0.36, x1, y1, z)
        B.rect_z(m, shade(tone, 0.82), x0, y0, x1, y0 + h * 0.36, z)
        cloth = cx.accent if cx.rng.random() < 0.5 else cx.pick(FABRIC)
        skin = cx.pick(["#e8c4a8", "#c99a7a", "#8d5b42", "#f0d2bc"])
        fx = x0 + w * cx.u(0.38, 0.62)
        s = min(h * 0.9, w * 1.9)
        yb = y0 + h * 0.2
        zf_ = z1
        # legs, body, head (each a separate area: no overlaps on one layer)
        B.rect_z(m, shade(cloth, 0.55), fx - s * 0.07, yb, fx + s * 0.07, yb + s * 0.38, zf_)
        B.quad(m, cloth, (fx - s * 0.1, yb + s * 0.38, zf_), (fx + s * 0.1, yb + s * 0.38, zf_),
               (fx + s * 0.12, yb + s * 0.66, zf_), (fx - s * 0.12, yb + s * 0.66, zf_), expect=np.array([0, 0, 1.0]))
        B.rect_z(m, skin, fx - s * 0.03, yb + s * 0.66, fx + s * 0.03, yb + s * 0.7, zf_)
        B.disc(m, skin, (fx, yb + s * 0.77, zf_), s * 0.065, (0, 0, 1), k=12)
        for sx in (-1, 1):
            B.quad(m, skin, (fx + sx * s * 0.12, yb + s * 0.64, zf_), (fx + sx * s * 0.16, yb + s * 0.62, zf_),
                   (fx + sx * s * 0.15, yb + s * 0.4, zf_), (fx + sx * s * 0.115, yb + s * 0.41, zf_),
                   expect=np.array([0, 0, 1.0]))
        for k in range(2):
            yy = y0 + h * (0.05 + 0.06 * k)
            B.rect_z(m, "#ffffff" if luminance(hex2rgb(tone)) < 0.6 else "#2b2b2b", x0 + w * 0.12, yy,
                     x0 + w * (0.8 - 0.25 * k), yy + h * 0.025, z1)
        cx.commit()
        return
    if st == "house":
        # property photo for estate agents: sky, lawn, house with roof and windows
        B.rect_z(m, cx.pick(["#a9c8e0", "#bcd4e6", "#9fbfd8"]), x0, y0 + h * 0.4, x1, y1, z)
        B.rect_z(m, cx.pick(["#7da05a", "#8fae6a"]), x0, y0, x1, y0 + h * 0.4, z)
        wall_c = cx.pick(["#efe6d6", "#e2d6c2", "#c9b79c", "#ffffff"])
        hx0, hx1 = x0 + w * 0.2, x1 - w * 0.2
        B.rect_z(m, wall_c, hx0, y0 + h * 0.22, hx1, y0 + h * 0.62, z1)
        B.tri(m, cx.pick(["#8a3b2b", "#5d5f61", "#6b4a3a"]), (hx0 - w * 0.05, y0 + h * 0.62, z1),
              (hx1 + w * 0.05, y0 + h * 0.62, z1), ((hx0 + hx1) / 2, y0 + h * 0.85, z1), expect=np.array([0, 0, 1.0]))
        for k in range(3):
            xx = hx0 + (hx1 - hx0) * (0.12 + 0.3 * k)
            B.rect_z(m, "#3d4f61", xx, y0 + h * 0.42, xx + (hx1 - hx0) * 0.16, y0 + h * 0.54, z1 + LAYER)
        cx.commit()
        return
    B.rect_z(m, bgc, x0, y0, x1, y1, z)
    if st == "circle":
        r = min(w, h) * 0.3
        B.disc(m, f1, (x0 + w * 0.5, y0 + h * 0.58, z1), r, (0, 0, 1), k=16)
    elif st == "band":
        B.quad(m, f1, (x0, y0 + h * 0.25, z1), (x1, y0 + h * 0.45, z1), (x1, y0 + h * 0.62, z1), (x0, y0 + h * 0.42, z1),
               expect=np.array([0, 0, 1.0]))
    # caption lines: below the circle / band, same layer (no overlap with them)
    for k in range(2 if st != "bars" else 4):
        yy = y0 + h * (0.06 + 0.045 * k)
        B.rect_z(m, f2, x0 + w * 0.12, yy, x0 + w * (0.88 - 0.25 * (k % 2)), yy + h * 0.022, z1)
    cx.commit()


def plinth(cx, x0, x1, h, z0, z1, col="#f4f4f2", mat="matte"):
    cx.B.box(mat, col, (x0, cx.c, z0), (x1, cx.c + h, z1), skip=("-y",))
    cx.commit()
    return cx.c + h


# ----------------------------------------------------------------------------- figures
def mannequin(cx, x, z, base_y, height=1.78, top=None, bottom=None, dress=False, skin=None, pose=0.0):
    B = cx.B
    s = height / 1.78
    skin = skin or cx.pick(SKIN)
    top = top or cx.pick(FABRIC)
    bottom = bottom or cx.pick(FABRIC)
    y0 = base_y
    B.cyl("gloss", "#cfd1d3", x, z, 0.15 * s, y0, y0 + 0.012, k=10, rz=0.1 * s)
    B.cyl("metal", "#bfc2c4", x, z, 0.01, y0 + 0.012, y0 + 0.3 * s, k=4, cap_top=False)
    ring = B.ellipse_ring
    # legs
    for sx in (-1, 1):
        lx = x + sx * 0.075 * s
        B.loft("matte", bottom, [ring(lx, y0 + 0.02, z, 0.045 * s, 0.05 * s, 7), ring(lx, y0 + 0.45 * s, z, 0.055 * s, 0.06 * s, 7),
                                  ring(lx - sx * 0.005, y0 + 0.88 * s, z, 0.078 * s, 0.075 * s, 7)], cap_top=False)
        B.box("gloss", "#1b1b1b", (lx - 0.045 * s, y0 + 0.012, z - 0.03 * s), (lx + 0.045 * s, y0 + 0.06 * s, z + 0.11 * s),
              skip=("-y",))
    # torso
    rings = [ring(x, y0 + 0.86 * s, z, 0.165 * s, 0.10 * s, 10), ring(x, y0 + 1.0 * s, z, 0.15 * s, 0.095 * s, 10),
             ring(x, y0 + 1.1 * s, z, 0.13 * s, 0.085 * s, 10), ring(x, y0 + 1.3 * s, z, 0.165 * s, 0.10 * s, 10),
             ring(x, y0 + 1.43 * s, z, 0.19 * s, 0.085 * s, 10)]
    B.loft("matte", top, rings, cap_top=True)
    if dress:
        B.loft("matte", top, [ring(x, y0 + 0.42 * s, z, 0.27 * s, 0.17 * s, 12), ring(x, y0 + 1.0 * s, z, 0.16 * s, 0.10 * s, 12)],
               cap_top=False, cap_bottom=True)
    # neck + head
    B.cyl("gloss", skin, x, z, 0.04 * s, y0 + 1.43 * s, y0 + 1.53 * s, k=6, cap_top=False)
    B.sphere("gloss", skin, (x, y0 + 1.62 * s, z + 0.005), 0.085 * s, k=8, n=5, ry=0.11 * s, rz=0.095 * s)
    # arms (sleeves in top colour, hands skin)
    for sx in (-1, 1):
        ax0 = x + sx * 0.2 * s
        ax1 = x + sx * (0.24 + 0.04 * pose) * s
        B.loft("matte", top, [ring(ax1, y0 + 0.86 * s, z + 0.01, 0.035 * s, 0.035 * s, 6),
                               ring(ax0, y0 + 1.40 * s, z, 0.05 * s, 0.05 * s, 6)], cap_top=False, cap_bottom=True)
        B.sphere("gloss", skin, (ax1, y0 + 0.82 * s, z + 0.01), 0.035 * s, k=6, n=3)
    return cx.commit()


def bust(cx, x, z, base_y, h, col):
    """dress form on a stand (for low display windows)."""
    B = cx.B
    if h < 0.55:
        return
    s = h / 1.0
    ring = B.ellipse_ring
    B.cyl("gloss", "#2b2b2b", x, z, 0.12 * s, base_y, base_y + 0.012, k=8)
    B.cyl("metal", "#9a9ea2", x, z, 0.012, base_y + 0.012, base_y + 0.36 * s, k=4, cap_top=False)
    B.loft("matte", col, [ring(x, base_y + 0.36 * s, z, 0.15 * s, 0.09 * s, 10), ring(x, base_y + 0.52 * s, z, 0.125 * s, 0.08 * s, 10),
                          ring(x, base_y + 0.74 * s, z, 0.16 * s, 0.095 * s, 10), ring(x, base_y + 0.86 * s, z, 0.18 * s, 0.08 * s, 10)],
           cap_top=True, cap_bottom=True)
    B.cyl("gloss", "#2b2b2b", x, z, 0.035 * s, base_y + 0.86 * s, base_y + 0.95 * s, k=6)
    return cx.commit()


def shoe(cx, x, y, z, col, l=0.27, w=0.095, h=0.11, ang=0.0, heel=False):
    B = cx.B
    mk = B.mark()
    B.box("gloss", "#f0f0ee" if not heel else "#151515", (-l / 2, 0, -w / 2), (l / 2, 0.025, w / 2), skip=("-y",))
    B.box("matte", col, (-l / 2 + 0.01, 0.025, -w / 2 + 0.006), (-l / 2 + l * 0.62, h, w / 2 - 0.006), skip=("-y",))
    B.loft("matte", col, [np.array([[-l / 2 + l * 0.6, 0.025, -w / 2 + 0.006], [l / 2 - 0.005, 0.025, -w / 2 + 0.01],
                                    [l / 2 - 0.005, 0.025, w / 2 - 0.01], [-l / 2 + l * 0.6, 0.025, w / 2 - 0.006]]),
                          np.array([[-l / 2 + l * 0.6, h * 0.62, -w / 2 + 0.01], [l / 2 - 0.02, 0.05, -w / 2 + 0.016],
                                    [l / 2 - 0.02, 0.05, w / 2 - 0.016], [-l / 2 + l * 0.6, h * 0.62, w / 2 - 0.01]])],
           cap_top=True)
    c, s_ = math.cos(ang), math.sin(ang)
    B.xform_since(mk, [[c, 0, s_], [0, 1, 0], [-s_, 0, c]], [x, y, z])


def garment(cx, x, ytop, z, kind=None, col=None, scale=1.0, thick=0.022):
    """hanger + garment silhouette facing the street (front face at z + thick). ytop = hook top."""
    from shapely.geometry import Polygon
    B = cx.B
    kind = kind or cx.pick(["shirt", "shirt", "dress", "jacket", "trousers"])
    col = col or cx.pick(FABRIC)
    s_ = scale
    if kind == "dress":
        P = [(-0.07, 0), (-0.17, -0.03), (-0.15, -0.38), (-0.26, -1.0), (0.26, -1.0), (0.15, -0.38), (0.17, -0.03),
             (0.07, 0), (0, -0.07)]
    elif kind == "jacket":
        P = [(-0.08, 0), (-0.22, -0.04), (-0.3, -0.62), (-0.22, -0.64), (-0.2, -0.3), (-0.2, -0.78), (0.2, -0.78),
             (0.2, -0.3), (0.22, -0.64), (0.3, -0.62), (0.22, -0.04), (0.08, 0), (0, -0.2)]
    elif kind == "trousers":
        P = [(-0.19, 0), (0.19, 0), (0.21, -0.98), (0.04, -0.98), (0.0, -0.26), (-0.04, -0.98), (-0.21, -0.98)]
    else:
        P = [(-0.07, 0), (-0.2, -0.04), (-0.29, -0.24), (-0.21, -0.29), (-0.19, -0.2), (-0.19, -0.7), (0.19, -0.7),
             (0.19, -0.2), (0.21, -0.29), (0.29, -0.24), (0.2, -0.04), (0.07, 0), (0, -0.06)]
    hook = 0.07 * s_
    y0 = ytop - hook
    pg = Polygon([(x + px * s_, y0 + py * s_) for px, py in P])
    if not pg.is_valid:
        pg = pg.buffer(0)
    B.extrude2d("matte", col, [pg], z, z + thick, side_col=shade(col, 0.7))
    # hanger bar + hook
    hz = z + thick + 0.002
    B.box("matte", "#2b2b2b" if cx.rng.random() < 0.5 else "#c8a77a", (x - 0.16 * s_, y0 - 0.012, hz),
          (x + 0.16 * s_, y0, hz + 0.008))
    B.box("metal", "#bfc2c4", (x - 0.004, y0, hz), (x + 0.004, ytop, hz + 0.008))
    if kind in ("shirt", "jacket") and cx.rng.random() < 0.5:
        # contrasting print / placket
        B.rect_z("matte", shade(col, 1.25) if luminance(hex2rgb(col)) < 0.5 else shade(col, 0.6),
                 x - 0.08 * s_, y0 - 0.38 * s_, x + 0.08 * s_, y0 - 0.22 * s_, z + thick + LAYER)


def faceout_rail(cx, x0, x1, ybar, z, cols=None):
    """wall rail with garments hung face-out (all in a plane just in front of the back wall)."""
    if x1 - x0 < 0.5:
        return
    cx.B.box("metal", "#bfc2c4", (x0, ybar, z), (x1, ybar + 0.02, z + 0.025), skip=("-z",))
    for xb in (x0 + 0.03, x1 - 0.05):
        cx.B.box("metal", "#bfc2c4", (xb, ybar - 0.02, cx.zb + 0.004), (xb + 0.02, ybar + 0.02, z), skip=("-z",))
    cx.commit()
    n = max(1, int((x1 - x0 - 0.1) / 0.48))
    sc = min(1.0, (ybar - cx.c - 0.1) / 1.08)
    if sc < 0.5:
        return
    for i in range(n):
        x = x0 + (x1 - x0) * (i + 0.5) / n
        garment(cx, x, ybar + 0.06, z + 0.026 + 0.003 * (i % 2), col=cx.pick(cols) if cols else None, scale=sc)
        cx.commit()


# ----------------------------------------------------------------------------- display kinds
def d_fashion(cx, sport=False):
    shell(cx, floor=cx.pick(["#d9d4cb", "#bdb5a8", "#e7e3dc", "#8d8780"]),
          back=cx.pick([cx.accent, "#efede8", "#dedad2"]) if cx.rng.random() < 0.6 else None)
    ceiling_lights(cx, "track")
    D = cx.D
    pal = ["#e9e9e7", "#1d1d1f", "#e30613", "#1f4e9c", "#f2f2f2", "#ffd400"] if sport else FABRIC
    if D >= 0.28 and cx.H > 2.05 and cx.W > 1.1 and cx.rng.random() < 0.75:
        ybar = cx.c + min(1.95, cx.H - 0.3)
        if not cx.logo or ybar + 0.1 < cx.d - 0.5:
            faceout_rail(cx, cx.a + 0.08, cx.b - 0.08, ybar, cx.zb + 0.004 + 2 * LAYER, cols=pal)
    if D >= 0.28:
        ph = cx.pick([0.0, 0.0, 0.12, 0.2])
        z0, z1 = cx.zb + 0.02, cx.zf - 0.03
        base = plinth(cx, cx.a + 0.05, cx.b - 0.05, ph, z0, z1, col=cx.pick(["#f5f5f3", "#2a2a2a", "#d7d0c4"])) if ph else cx.c
        n = max(1, min(4, int((cx.W - 0.2) / 0.62)))
        hmax = cx.d - base - 0.08
        for i in range(n):
            x = cx.a + cx.W * (i + 0.5) / n + cx.u(-0.06, 0.06)
            zc = (cx.zb + cx.zf) / 2 + cx.u(-0.02, 0.02)
            h = min(1.80, hmax)
            if h < 1.25:
                bust(cx, x, zc, base, min(hmax, 1.0), cx.pick(FABRIC))
                continue
            mannequin(cx, x, zc, base, height=h, top=cx.pick(pal), bottom=cx.pick(pal), dress=(not sport) and cx.rng.random() < 0.35,
                      pose=cx.u(-1, 1))
        if sport and D > 0.3:
            for i in range(int(cx.W / 0.4)):
                shoe(cx, cx.a + 0.25 + i * 0.4, base, cx.zf - 0.12, cx.pick(["#ffffff", "#e30613", "#111111", "#1f4e9c"]))
                cx.commit()
    else:
        # shallow: garments face-out on a wall rail
        bar_y = cx.c + min(1.75, cx.H - 0.2)
        faceout_rail(cx, cx.a + 0.06, cx.b - 0.06, bar_y, cx.zb + 0.004 + 2 * LAYER, cols=pal)
    if cx.H > 2.3 and cx.W > 1.2:
        poster(cx, cx.a + 0.1, cx.a + min(cx.W * 0.45, 0.9), cx.d - 0.75, cx.d - 0.12, cx.zb + 0.004)


def d_shoes(cx):
    shell(cx, floor=cx.pick(["#cfc8bd", "#e4e0d8"]), back=cx.pick([cx.accent, "#efede8"]))
    ceiling_lights(cx, "spots")
    zr = cx.zb + 0.012
    levels = [0.35, 0.75, 1.15, 1.55]
    dep = min(0.3, cx.D - 0.04)
    for ly in levels:
        y = cx.c + ly
        if y > cx.d - 0.3:
            break
        cx.B.box("gloss", "#f6f6f4", (cx.a + 0.05, y - 0.02, zr), (cx.b - 0.05, y, zr + dep), skip=("-z",))
        cx.commit()
        n = int((cx.W - 0.2) / 0.32)
        for i in range(n):
            shoe(cx, cx.a + 0.2 + i * 0.32, y, zr + dep / 2, cx.pick(FABRIC + ["#111111", "#ffffff", "#8a5a3b"]),
                 ang=cx.u(-0.5, 0.5), heel=cx.rng.random() < 0.2)
            cx.commit()


def d_jewellery(cx):
    shell(cx, floor=cx.pick(["#2b2a2a", "#3a3634", "#e9e4da"]), wall=cx.pick(["#2c2a2b", "#e9e4da", "#d8d2c6"]),
          back=cx.pick(["#1d1c1e", "#3b3238", "#efeae0"]), ceil="#f4f2ee")
    ceiling_lights(cx, "spots", warm=True)
    n = max(1, int((cx.W - 0.1) / 0.55))
    dz = min(0.32, cx.D - 0.08)
    if dz < 0.15:
        return
    zc = cx.zf - dz / 2 - 0.04
    for i in range(n):
        x = cx.a + cx.W * (i + 0.5) / n
        ph = cx.u(0.8, 1.05)
        if cx.c + ph + 0.4 > cx.d:
            ph = cx.d - cx.c - 0.45
        if ph < 0.3:
            continue
        cx.B.box("gloss", cx.pick(["#1d1d1f", "#efeae0", "#3b3238"]), (x - 0.18, cx.c, zc - dz / 2), (x + 0.18, cx.c + ph, zc + dz / 2),
                 skip=("-y",))
        top = cx.c + ph
        cx.B.box("matte", "#2e2b33", (x - 0.16, top, zc - dz / 2 + 0.02), (x + 0.16, top + 0.02, zc + dz / 2 - 0.02), skip=("-y",))
        # bust with necklace / ring cones
        if cx.rng.random() < 0.5:
            cx.B.frustum("matte", "#26232a", x, zc, 0.07, 0.03, top + 0.02, top + 0.22, k=8)
            cx.B.cyl("metal", cx.pick(["#d4af37", "#c0c0c0", "#e6c06a"]), x, zc + 0.012, 0.045, top + 0.14, top + 0.15, k=10)
        else:
            for k in range(3):
                xx = x - 0.09 + 0.09 * k
                cx.B.frustum("matte", "#26232a", xx, zc, 0.022, 0.008, top + 0.02, top + 0.09, k=6)
                cx.B.cyl("metal", cx.pick(["#d4af37", "#c0c0c0", "#e6c06a"]), xx, zc, 0.014, top + 0.075, top + 0.082, k=8)
        # vitrine hood
        cx.B.box("vitrine", "#ffffff", (x - 0.165, top + 0.02, zc - dz / 2 + 0.015), (x + 0.165, top + 0.3, zc + dz / 2 - 0.015),
                 skip=("-y",))
        cx.commit()


def d_cafe(cx, restaurant=False, fast=False):
    warm = True
    shell(cx, floor=cx.pick(WOOD + ["#7d7770"]), wall=cx.pick(["#efe6d6", "#e2d6c2", "#d9cbb5", "#3d3a36"]),
          back=cx.pick(["#4a3b30", "#e6dcc8", cx.accent, "#2f3a33"]))
    if not fast:
        n = max(1, int(cx.W / 0.9))
        for i in range(n):
            pendant(cx, cx.a + cx.W * (i + 0.5) / n, cx.zf - min(0.25, cx.D * 0.45), drop=cx.u(0.5, 0.8),
                    shade_col=cx.pick(["#2b2b2b", "#c8a77a", "#e9e6df", "#3c5a40"]))
    else:
        ceiling_lights(cx, "spots")
    D = cx.D
    # back wall: shelf with cups/bottles + menu board
    zr = cx.zb + 0.004
    if cx.H > 2.1:
        mb_w = min(1.2, cx.W * 0.5)
        x0 = cx.a + (cx.W - mb_w) / 2
        if fast:
            for k in range(max(1, int(cx.W / 0.7))):
                xx = cx.a + 0.08 + k * 0.7
                if xx + 0.62 > cx.b - 0.05:
                    break
                poster(cx, xx, xx + 0.62, cx.d - 0.62, cx.d - 0.2, zr, cols=(cx.bg, cx.fg, "#ffffff"), lit=True, style="bars")
        else:
            cx.B.rect_z("matte", "#1f2321", x0, cx.d - 0.85, x0 + mb_w, cx.d - 0.25, zr)
            for k in range(6):
                yy = cx.d - 0.35 - k * 0.08
                cx.B.rect_z("matte", "#e8e4dc", x0 + 0.08, yy - 0.012, x0 + mb_w * cx.u(0.4, 0.85), yy, zr + LAYER)
            cx.commit()
    sh_y = cx.c + 1.45
    if sh_y < cx.d - 0.95 or not (cx.H > 2.1):
        pass
    else:
        for side in (0, 1):
            xa = cx.a + 0.05 if side == 0 else cx.b - 0.05 - min(0.5, cx.W * 0.2)
            xb = xa + min(0.5, cx.W * 0.2)
            cx.B.box("matte", cx.pick(WOOD), (xa, sh_y, zr), (xb, sh_y + 0.025, zr + 0.18), skip=("-z",))
            for k in range(int((xb - xa) / 0.09)):
                cx.B.cyl("gloss", cx.pick(["#f3f1ec", "#2f5d50", "#8a5a3b", "#e9e6df"]), xa + 0.05 + k * 0.09, zr + 0.09,
                         0.03, sh_y + 0.025, sh_y + 0.1, k=6)
            cx.commit()
    if D >= 0.75:
        # tables with chairs (1.3 m pitch: the chairs of neighbouring tables never interpenetrate)
        n = max(1, int(cx.W / 1.3))
        for i in range(n):
            x = cx.a + cx.W * (i + 0.5) / n
            zc = cx.zf - 0.45
            if zc - 0.36 < cx.zb + 0.02:
                zc = cx.zb + 0.38
            table(cx, x, zc, 0.33, cloth=restaurant)
            for sx in (-1, 1):
                chair(cx, x + sx * 0.42, zc, face=-sx)
    else:
        # window bar along the glass + stools
        yb = cx.c + 1.02
        dz = min(0.3, D * 0.5)
        if yb < cx.d - 0.5 and dz > 0.12:
            cx.B.box("matte", cx.pick(WOOD), (cx.a + 0.02, yb - 0.04, cx.zf - dz - 0.01), (cx.b - 0.02, yb, cx.zf - 0.01))
            cx.commit()
            for k in range(int(cx.W / 0.55)):
                x = cx.a + 0.3 + k * 0.55
                if x > cx.b - 0.2:
                    break
                cx.B.cyl("gloss", "#f3f1ec", x, cx.zf - dz * 0.6, 0.04, yb, yb + 0.09, k=6)
                cx.commit()
            if D > 0.45:
                for k in range(int(cx.W / 0.6)):
                    x = cx.a + 0.32 + k * 0.6
                    if x > cx.b - 0.2:
                        break
                    z = cx.zb + min(0.2, D * 0.3)
                    cx.B.cyl("metal", "#2b2b2b", x, z, 0.02, cx.c, cx.c + 0.7, k=5, cap_top=False)
                    cx.B.cyl("matte", cx.pick(["#2b2b2b", "#8a5a3b", "#c8a77a"]), x, z, min(0.17, D * 0.3), cx.c + 0.7, cx.c + 0.74, k=10)
                    cx.commit()
    if cx.rng.random() < 0.6 and cx.W > 1.0:
        plant(cx, cx.b - 0.22, cx.zb + min(0.2, D * 0.35), min(0.16, D * 0.3))


def table(cx, x, z, r, cloth=False):
    B = cx.B
    y = cx.c + 0.74
    B.cyl("metal", "#2b2b2b", x, z, 0.025, cx.c, y - 0.03, k=6, cap_top=False)
    B.cyl("metal", "#2b2b2b", x, z, 0.2, cx.c, cx.c + 0.015, k=8)
    if cloth:
        B.box("matte", "#f6f4ef", (x - r, y - 0.2, z - r), (x + r, y, z + r))
    else:
        B.box("matte", cx.pick(WOOD), (x - r, y - 0.03, z - r), (x + r, y, z + r))
    for k in range(2):
        B.cyl("vitrine", "#ffffff", x - 0.1 + 0.2 * k, z + cx.u(-0.1, 0.1), 0.03, y, y + 0.12, k=6)
    cx.commit()


def chair(cx, x, z, face=1):
    B = cx.B
    col = cx.pick(["#2b2b2b", "#8a5a3b", "#c8a77a", "#3c5a40", "#7a1e2c"])
    sy = cx.c + 0.46
    for dx in (-0.17, 0.17):
        for dz in (-0.17, 0.17):
            B.box("matte", "#222222", (x + dx - 0.015, cx.c, z + dz - 0.015), (x + dx + 0.015, sy, z + dz + 0.015), skip=("-y",))
    B.box("matte", col, (x - 0.2, sy, z - 0.2), (x + 0.2, sy + 0.04, z + 0.2))
    bx = x - face * 0.185
    B.box("matte", col, (bx - 0.015, sy + 0.04, z - 0.2), (bx + 0.015, sy + 0.45, z + 0.2))
    cx.commit()


def plant(cx, x, z, r):
    B = cx.B
    h = min(cx.u(0.35, 0.55), cx.H - 0.5 - r)
    if h < 0.2 or r < 0.06:
        return
    B.frustum("matte", cx.pick(["#e9e6df", "#2b2b2b", "#b5651d"]), x, z, r * 0.8, r, cx.c, cx.c + h, k=8)
    for k in range(5):
        ang = k * 1.256 + cx.u(0, 0.5)
        B.sphere("matte", cx.pick(["#3c6b3c", "#4f7f45", "#2e5a35"]), (x + math.cos(ang) * r * 0.5, cx.c + h + 0.15 + k * 0.06,
                                                                    z + math.sin(ang) * r * 0.4), r * 0.75, k=6, n=3)
    cx.commit()


def shelves(cx, x0, x1, levels, depth, items="boxes", board="#f4f4f2", cols=None):
    B = cx.B
    zr = cx.zb + 0.004
    depth = min(depth, cx.D - 0.03)
    if depth < 0.08:
        return
    ylim = min([r[2] for r in cx.reserved] + [cx.d])      # stay below the back-wall name panel
    for ly in levels:
        y = cx.c + ly
        if y > ylim - 0.12:
            break
        B.box("matte", board, (x0, y - 0.025, zr), (x1, y, zr + depth), skip=("-z",))
        cx.commit()
        x = x0 + 0.02
        top_room = min(0.36, ylim - y - 0.06)
        while x < x1 - 0.05:
            col = cx.pick(cols or ["#e30613", "#ffffff", "#0067b1", "#ffd400", "#2e7d32", "#f39200", "#8e44ad", "#00a3e0"])
            if items == "boxes":
                w = cx.u(0.05, 0.12)
                h = min(cx.u(0.1, 0.24), top_room)
                B.box("matte", col, (x, y, zr + 0.01), (min(x + w, x1 - 0.02), y + h, zr + min(depth - 0.01, 0.16)), skip=("-y", "-z"))
                x += w + 0.008
            elif items == "bottles":
                r = cx.u(0.022, 0.035)
                h = min(cx.u(0.12, 0.24), top_room)
                B.cyl("gloss", col, x + r, zr + depth * 0.5, r, y, y + h, k=6)
                x += 2 * r + 0.012
            elif items == "books":
                w = cx.u(0.018, 0.045)
                h = min(cx.u(0.17, 0.27), top_room)
                B.box("matte", col, (x, y, zr + 0.01), (min(x + w, x1 - 0.02), y + h, zr + min(depth - 0.01, 0.2)), skip=("-y", "-z"))
                x += w + 0.002
            elif items == "tins":
                r = 0.04
                h = min(0.12, top_room)
                B.cyl("metal", col, x + r, zr + depth * 0.5, r, y, y + h, k=8)
                x += 2 * r + 0.015
            elif items == "magazines":
                w = cx.u(0.2, 0.24)
                h = min(0.3, top_room)
                B.quad("glow", col, (x, y, zr + depth - 0.02), (x + w, y, zr + depth - 0.02), (x + w, y + h, zr + depth * 0.4),
                       (x, y + h, zr + depth * 0.4), expect=np.array([0, 0.4, 1.0]))
                x += w + 0.02
            elif items == "glasses":
                if x + 0.13 > x1 - 0.01:
                    break
                gx = x + 0.06
                gy = y + 0.04
                zz = zr + depth * 0.6
                for sx in (-1, 1):
                    cxx = gx + sx * 0.03
                    B.quad("matte", col, (cxx - 0.025, gy, zz), (cxx + 0.025, gy, zz), (cxx + 0.025, gy + 0.032, zz),
                           (cxx - 0.025, gy + 0.032, zz), expect=np.array([0, 0, 1.0]))
                B.box("matte", col, (gx - 0.055, gy + 0.025, max(zr + 0.006, zz - 0.12)), (gx - 0.05, gy + 0.03, zz), skip=())
                x += 0.15
            elif items == "bread":
                if x + 0.21 > x1 - 0.01:
                    break
                B.sphere("matte", cx.pick(["#b07a3a", "#c98f4a", "#8a5a2b", "#d9a760"]), (x + 0.1, y + 0.05, zr + depth * 0.5),
                         0.1, k=7, n=3, ry=0.05, rz=min(0.06, depth * 0.4))
                x += 0.22
            cx.commit()


def d_pharmacy(cx):
    shell(cx, floor="#e4e0d8", wall="#f6f7f5", back="#ffffff")
    ceiling_lights(cx, "spots")
    shelves(cx, cx.a + 0.05, cx.b - 0.05, [0.5, 0.9, 1.3, 1.7], 0.22, items="boxes")
    # green cross light box hanging in the window
    if cx.W > 0.9 and cx.D > 0.15:
        x = cx.a + cx.W * cx.pick([0.25, 0.75])
        y = cx.d - (0.85 if cx.reserved else 0.45)      # below the back-wall name panel
        z = cx.zf - 0.06
        s = 0.11
        cx.B.box("matte", "#111111", (x - 0.002, y + 0.15 * 1.5, z - 0.002), (x + 0.002, cx.d - 0.003, z + 0.002), skip=("+y",))
        # one cross-shaped light box (a single outline: no overlapping coplanar faces)
        from shapely.geometry import Polygon
        h2 = s / 2
        X = Polygon([(x - h2, y - 3 * h2), (x + h2, y - 3 * h2), (x + h2, y - h2), (x + 3 * h2, y - h2), (x + 3 * h2, y + h2),
                     (x + h2, y + h2), (x + h2, y + 3 * h2), (x - h2, y + 3 * h2), (x - h2, y + h2), (x - 3 * h2, y + h2),
                     (x - 3 * h2, y - h2), (x - h2, y - h2)])
        cx.B.extrude2d("lit", "#00a651", [X], z - 0.02, z + 0.02)
        cx.B.poly("lit", "#00a651", [(px, py, z - 0.02) for px, py in list(X.exterior.coords)[:-1]], (0, 0, -1))
        cx.commit()
    if cx.D > 0.3:
        x0 = cx.a + cx.W * 0.15
        for k in range(3):
            cx.B.box("matte", cx.pick(["#e30613", "#0067b1", "#ffffff", "#2e7d32"]), (x0 + k * 0.13, cx.c, cx.zf - 0.25),
                     (x0 + k * 0.13 + 0.12, cx.c + 0.16, cx.zf - 0.09), skip=("-y",))
        cx.B.box("matte", "#f39200", (x0 + 0.065, cx.c + 0.16, cx.zf - 0.24), (x0 + 0.315, cx.c + 0.32, cx.zf - 0.1), skip=("-y",))
        cx.commit()
    if cx.H > 2.2:
        poster(cx, cx.b - min(0.9, cx.W * 0.45), cx.b - 0.08, cx.c + 0.25, cx.c + 1.0, cx.zf - min(0.08, cx.D * 0.3),
               cols=("#ffffff", "#00a651", "#e30613"), style="photo")


def d_bank(cx):
    shell(cx, floor=cx.pick(["#9c968e", "#7d7770", "#cfc8bd"]), wall="#f2f2f0", back=cx.bg if luminance(hex2rgb(cx.bg)) < 0.9 else cx.fg)
    ceiling_lights(cx, "strip")
    n = max(1, min(3, int(cx.W / 0.9)))
    dz = min(0.38, cx.D - 0.03)
    if dz < 0.12:
        return
    for i in range(n):
        x = cx.a + cx.W * (i + 0.5) / n
        h = min(1.75, cx.H - 0.1)
        z0 = cx.zb + 0.004
        cx.B.box("metal", "#8e9396", (x - 0.3, cx.c, z0), (x + 0.3, cx.c + h, z0 + dz), skip=("-y", "-z"))
        # screen recess + lit screen
        sy = cx.c + 1.25
        cx.B.quad("lit", "#3d8fd1", (x - 0.17, sy, z0 + dz + LAYER), (x + 0.17, sy, z0 + dz + LAYER),
                  (x + 0.17, sy + 0.26, z0 + dz + LAYER), (x - 0.17, sy + 0.26, z0 + dz + LAYER), expect=np.array([0, 0, 1.0]))
        # keypad shelf
        cx.B.box("metal", "#3a3d40", (x - 0.22, cx.c + 0.98, z0 + dz), (x + 0.22, cx.c + 1.02, z0 + dz + min(0.12, cx.zf - z0 - dz - 0.01)))
        cx.B.rect_z("lit", "#e8f1f8", x - 0.06, cx.c + 1.08, x + 0.06, cx.c + 1.1, z0 + dz + LAYER)
        if cx.c + h + 0.32 < cx.d and cx.free_back(x - 0.3, x + 0.3, cx.c + h + 0.05, cx.c + h + 0.3):
            cx.B.box("lit", cx.bg, (x - 0.3, cx.c + h + 0.05, z0), (x + 0.3, cx.c + h + 0.3, z0 + 0.06), skip=("-z",))
        cx.commit()


def d_optician(cx):
    shell(cx, floor="#e4e0d8", wall="#f6f6f4", back=cx.pick(["#ffffff", "#ebe6dc", cx.accent]))
    ceiling_lights(cx, "spots")
    shelves(cx, cx.a + 0.08, cx.b - 0.08, [0.9, 1.15, 1.4, 1.65], 0.16, items="glasses", board="#dfe5e8",
            cols=["#111111", "#5a3b28", "#8c969c", "#7a1e2c", "#1f2a44"])
    if cx.H > 2.2 and cx.W > 1.0:
        poster(cx, cx.a + 0.1, cx.a + min(0.8, cx.W * 0.45), cx.c + 0.15, cx.c + 0.8, cx.zf - min(0.06, cx.D * 0.3), style="photo")


def d_tech(cx):
    shell(cx, floor="#e9e9e7", wall="#f4f4f4", back=cx.pick([cx.accent, "#f4f4f4"]))
    ceiling_lights(cx, "strip")
    if cx.H > 2.0:
        poster(cx, cx.a + 0.1, cx.b - 0.1, cx.c + 1.1, min(cx.d - 0.15, cx.c + 2.2), cx.zb + 0.004, lit=True)
    if cx.D > 0.3:
        y = cx.c + 0.85
        cx.B.box("gloss", "#f7f7f7", (cx.a + 0.1, cx.c, cx.zf - min(0.5, cx.D - 0.06)), (cx.b - 0.1, y, cx.zf - 0.04), skip=("-y",))
        cx.commit()
        for k in range(int((cx.W - 0.3) / 0.3)):
            x = cx.a + 0.25 + k * 0.3
            z = cx.zf - 0.2
            cx.B.box("gloss", "#111111", (x - 0.04, y, z - 0.07), (x + 0.04, y + 0.008, z + 0.07), skip=("-y",))
            cx.B.quad("lit", cx.pick(["#4fa3e0", "#8e44ad", "#f39200", "#2ecc71"]), (x - 0.035, y + 0.011, z + 0.065),
                      (x + 0.035, y + 0.011, z + 0.065), (x + 0.035, y + 0.011, z - 0.065), (x - 0.035, y + 0.011, z - 0.065),
                      expect=np.array([0, 1.0, 0]))
            cx.commit()


def d_cosmetics(cx):
    shell(cx, floor="#ece8e2", wall="#fbfaf8", back=cx.pick(["#ffffff", cx.accent]))
    ceiling_lights(cx, "spots")
    shelves(cx, cx.a + 0.08, cx.b - 0.08, [0.7, 1.05, 1.4, 1.75], 0.18, items="bottles",
            cols=["#f7c6c7", "#ffffff", "#d4af37", "#111111", "#e8b4b8", "#c0c0c0", "#9b59b6", "#f39c12"])
    if cx.H > 2.2 and cx.W > 1.2:
        poster(cx, cx.b - min(0.9, cx.W * 0.4), cx.b - 0.1, cx.c + 0.15, cx.c + 0.62, cx.zf - min(0.06, cx.D * 0.3), style="photo")


def d_books(cx):
    shell(cx, floor=cx.pick(WOOD), wall="#f1ede4", back="#efe9dd")
    ceiling_lights(cx, "spots", warm=True)
    shelves(cx, cx.a + 0.05, cx.b - 0.05, [0.35, 0.72, 1.09, 1.46, 1.83], 0.22, items="books", board=cx.pick(WOOD),
            cols=["#7a1e2c", "#1f2a44", "#e3d6c4", "#2e5a35", "#c9b79c", "#f2c14e", "#111111", "#e30613", "#4a6b8a", "#ffffff"])


def d_kiosk(cx):
    shell(cx, floor="#9c968e", wall="#f2f2f0", back="#f2f2f0")
    ceiling_lights(cx, "strip")
    w = cx.W
    if w > 1.2 and cx.D > 0.3:
        # drinks fridge (lit) on one side
        x0 = cx.b - min(0.7, w * 0.4)
        zr = cx.zb + 0.004
        dz = min(0.45, cx.D - 0.04)
        ft = min(cx.c + 1.9, cx.d - 0.05)
        x1 = cx.b - 0.03
        # side posts between the base and the top box (no coplanar faces where they would overlap)
        cx.B.box("gloss", "#dfe3e6", (x0, cx.c + 0.18, zr), (x0 + 0.04, ft - 0.12, zr + dz), skip=("-y", "+y", "-z"))
        cx.B.box("gloss", "#dfe3e6", (x1 - 0.04, cx.c + 0.18, zr), (x1, ft - 0.12, zr + dz), skip=("-y", "+y", "-z"))
        cx.B.box("gloss", "#dfe3e6", (x0, ft - 0.12, zr), (x1, ft, zr + dz), skip=("-z",))
        cx.B.box("gloss", "#dfe3e6", (x0, cx.c, zr), (x1, cx.c + 0.18, zr + dz), skip=("-y", "-z"))
        cx.B.rect_z("lit", "#eef6ff", x0 + 0.04, cx.c + 0.18, x1 - 0.04, ft - 0.12, zr + 0.002)
        cx.commit()
        for row in range(5):
            y = cx.c + 0.2 + row * 0.32
            if y + 0.22 > ft - 0.12:
                break
            cx.B.box("vitrine", "#ffffff", (x0 + 0.04, y - 0.012, zr + 0.003), (x1 - 0.04, y, zr + dz - 0.02))
            for k in range(int((x1 - x0 - 0.1) / 0.08)):
                cx.B.cyl("gloss", cx.pick(["#e30613", "#ffd400", "#2e7d32", "#0067b1", "#f39200", "#ffffff"]),
                         x0 + 0.08 + k * 0.08, zr + dz * 0.5, min(0.028, dz * 0.2), y, y + 0.2, k=6)
            cx.commit()
        cx.B.rect_z("vitrine", "#ffffff", x0 + 0.04, cx.c + 0.18, x1 - 0.04, ft - 0.12, zr + dz - 0.01)
        cx.commit()
        shelves(cx, cx.a + 0.05, x0 - 0.05, [0.6, 1.0, 1.4], 0.25, items="magazines")
    else:
        shelves(cx, cx.a + 0.05, cx.b - 0.05, [0.6, 1.0, 1.4], 0.25, items="magazines")


def d_bakery(cx, confectionery=False):
    shell(cx, floor=cx.pick(["#cfc8bd", "#e4e0d8", "#c8a77a"]), wall="#f6efe2", back=cx.pick(["#e9dcc4", cx.accent]))
    ceiling_lights(cx, "spots", warm=True)
    shelves(cx, cx.a + 0.05, cx.b - 0.05, [1.15, 1.5, 1.85], 0.25, items="boxes" if confectionery else "bread", board=cx.pick(WOOD),
            cols=["#7a1e2c", "#d4af37", "#ffffff", "#e8b4b8", "#5a3b28", "#c43c3c"] if confectionery else None)
    if cx.D > 0.42:
        # counter vitrine
        z1 = cx.zf - 0.04
        z0 = max(cx.zb + 0.05, z1 - 0.6)
        y = cx.c + 0.9
        cx.B.box("matte", "#f4f1ea", (cx.a + 0.08, cx.c, z0), (cx.b - 0.08, cx.c + 0.62, z1), skip=("-y",))
        cx.commit()
        n = int((cx.W - 0.3) / 0.12)
        for k in range(n):
            x = cx.a + 0.2 + k * 0.12
            cx.B.sphere("matte", cx.pick(["#c98f4a", "#e0b070", "#8a5a2b", "#f2d39b", "#e8b4b8", "#5a3b28"]),
                        (x, cx.c + 0.66, (z0 + z1) / 2), 0.045, k=6, n=3, ry=0.03)
        cx.B.box("vitrine", "#ffffff", (cx.a + 0.08, cx.c + 0.62, z0), (cx.b - 0.08, y, z1), skip=("-y",))
        cx.commit()


def d_houseware(cx, furniture=False):
    shell(cx, floor=cx.pick(WOOD + ["#e4e0d8"]), wall=cx.pick(WALL), back=cx.pick(WALL + [cx.accent]))
    ceiling_lights(cx, "track", warm=True)
    if furniture and cx.D > 0.5 and cx.W > 1.5 and cx.H > 1.2:
        # sofa / armchair
        x0, x1 = cx.a + 0.2, min(cx.b - 0.2, cx.a + 1.9)
        z0 = cx.zb + 0.05
        z1 = min(cx.zf - 0.05, z0 + 0.75)
        col = cx.pick(["#8c969c", "#c9b79c", "#3c5a40", "#7a1e2c", "#30475e"])
        cx.B.box("matte", "#2b2b2b", (x0 + 0.05, cx.c, z0 + 0.05), (x1 - 0.05, cx.c + 0.12, z1 - 0.05), skip=("-y",))
        cx.B.box("matte", col, (x0, cx.c + 0.12, z0), (x1, cx.c + 0.42, z1))
        cx.B.box("matte", col, (x0, cx.c + 0.42, z0), (x1, cx.c + 0.8, z0 + 0.18))
        for xa in (x0, x1 - 0.16):
            cx.B.box("matte", col, (xa, cx.c + 0.42, z0 + 0.18), (xa + 0.16, cx.c + 0.6, z1))
        cx.commit()
        plant(cx, cx.b - 0.25, cx.zb + 0.25, min(0.18, cx.D * 0.3))
        return
    shelves(cx, cx.a + 0.06, cx.b - 0.06, [0.9, 1.35, 1.8], 0.25, items=cx.pick(["tins", "bottles", "boxes"]),
            board=cx.pick(WOOD), cols=["#e9e6df", "#2b2b2b", "#c8a77a", "#3c5a40", "#b5651d", "#4a6b8a", "#d4af37", "#7a1e2c"])
    if cx.D > 0.35 and cx.H > 1.2:
        y = plinth(cx, cx.a + 0.1, cx.b - 0.1, 0.55, cx.zf - min(0.42, cx.D - 0.06), cx.zf - 0.04, col=cx.pick(WOOD))
        n = int((cx.W - 0.3) / 0.22)
        for k in range(n):
            x = cx.a + 0.25 + k * 0.22
            z = cx.zf - 0.2
            t = cx.rng.random()
            if t < 0.4:
                cx.B.frustum("gloss", cx.pick(["#e9e6df", "#3c5a40", "#b5651d", "#4a6b8a", "#111111"]), x, z, 0.05, 0.035, y, y + cx.u(0.12, 0.3), k=8)
            elif t < 0.7:
                cx.B.frustum("gloss", cx.pick(["#e9e6df", "#c8a77a", "#7a1e2c"]), x, z, 0.04, 0.08, y, y + 0.07, k=10)
            else:
                cx.B.box("matte", cx.pick(["#c9b79c", "#2b2b2b", "#e3d6c4"]), (x - 0.07, y, z - 0.06), (x + 0.07, y + 0.1, z + 0.06), skip=("-y",))
            cx.commit()


def d_hotel(cx):
    shell(cx, floor=cx.pick(["#7d7770", "#3a3634", "#c8a77a"]), wall="#e9dfcf", back=cx.pick(["#5a3b28", "#e9dfcf", "#2f3a33"]))
    n = max(1, int(cx.W / 0.9))
    for i in range(n):
        pendant(cx, cx.a + cx.W * (i + 0.5) / n, cx.zf - min(0.25, cx.D * 0.45), drop=0.55, shade_col="#d9c3a5",
                col="#ffe7b8")
    if cx.D > 0.45 and cx.H > 1.5:
        x0, x1 = cx.a + cx.W * 0.25, cx.b - cx.W * 0.25
        cx.B.box("gloss", "#5a3b28", (x0, cx.c, cx.zb + 0.02), (x1, cx.c + 1.08, cx.zb + min(0.5, cx.D - 0.1)), skip=("-y", "-z"))
        cx.B.box("gloss", "#d9cbb5", (x0 - 0.02, cx.c + 1.08, cx.zb + 0.02), (x1 + 0.02, cx.c + 1.12, cx.zb + min(0.52, cx.D - 0.08)))
        cx.commit()
    plant(cx, cx.a + 0.25, cx.zf - min(0.25, cx.D * 0.45), min(0.18, cx.D * 0.3))


def d_office(cx, travel=False, estate=False):
    shell(cx, floor=cx.pick(["#8c8f91", "#b5aea4", "#5d5f61"]), wall="#f2f2f0", back=cx.pick(["#f2f2f0", cx.accent]))
    ceiling_lights(cx, "strip")
    if cx.D > 0.5 and cx.H > 1.4:
        x0 = cx.a + cx.W * 0.2
        x1 = min(cx.b - 0.2, x0 + 1.4)
        z0 = cx.zb + 0.05
        z1 = min(cx.zf - 0.1, z0 + 0.75)
        y = cx.c + 0.74
        cx.B.box("matte", "#f4f4f2", (x0, y - 0.03, z0), (x1, y, z1))
        for xx in (x0 + 0.03, x1 - 0.06):
            cx.B.box("metal", "#8e9396", (xx, cx.c, z0 + 0.05), (xx + 0.03, y - 0.03, z1 - 0.05), skip=("-y",))
        cx.B.box("gloss", "#111111", ((x0 + x1) / 2 - 0.27, y, z0 + 0.06), ((x0 + x1) / 2 + 0.27, y + 0.36, z0 + 0.09))
        zs = z0 + 0.09 + LAYER
        cx.B.quad("lit", "#d6e6f5", ((x0 + x1) / 2 - 0.25, y + 0.03, zs), ((x0 + x1) / 2 + 0.25, y + 0.03, zs),
                  ((x0 + x1) / 2 + 0.25, y + 0.34, zs), ((x0 + x1) / 2 - 0.25, y + 0.34, zs), expect=np.array([0, 0, 1.0]))
        cx.commit()
    if travel and cx.H > 2.0:
        k = int((cx.W - 0.1) / 0.55)
        for i in range(k):
            x = cx.a + 0.1 + i * 0.55
            poster(cx, x, x + 0.5, cx.c + 0.9, cx.c + 1.6, cx.zf - min(0.05, cx.D * 0.3),
                   cols=(cx.pick(["#4fb3e8", "#2a9fd6", "#f2c14e"]), cx.pick(["#f2d39b", "#ffffff", "#e8f4fa"]), "#ffffff"), style="band")
    elif estate:
        k = int((cx.W - 0.1) / 0.38)
        for i in range(k):
            for j in range(3):
                x = cx.a + 0.1 + i * 0.38
                y = cx.c + 0.8 + j * 0.42
                if y + 0.32 > cx.d - 0.1:
                    break
                poster(cx, x, x + 0.3, y, y + 0.32, cx.zf - min(0.04, cx.D * 0.3), cols=("#ffffff", cx.pick(["#7da7c9", "#9c968e", "#c9b79c"]), cx.accent),
                       style="house")
    elif cx.rng.random() < 0.6:
        # half-height frosted film right behind the glass
        z = cx.zf - 0.003
        cx.B.rect_z("matte", "#eef0f1", cx.a + 0.002, cx.c + 0.002, cx.b - 0.002, cx.c + min(1.45, cx.H * 0.55), z)
        cx.commit()


def d_hair(cx):
    shell(cx, floor=cx.pick(["#2b2b2b", "#e4e0d8", "#c8a77a"]), wall="#f4f2ee", back=cx.pick(["#f4f2ee", "#2b2b2b", cx.accent]))
    ceiling_lights(cx, "spots")
    n = max(1, int(cx.W / 1.0))
    zr = cx.zb + 0.004
    for i in range(n):
        x = cx.a + cx.W * (i + 0.5) / n
        if cx.H < 2.0:
            break
        cx.B.rect_z("gloss", "#cfd8dc", x - 0.3, cx.c + 0.95, x + 0.3, cx.c + 1.85, zr + 0.002)
        cx.B.box("matte", "#111111", (x - 0.33, cx.c + 0.85, zr), (x + 0.33, cx.c + 0.88, zr + 0.15), skip=("-z",))
        cx.commit()
        if cx.D > 0.55:
            chair(cx, x, cx.zf - 0.3, face=1)


def d_market(cx):
    shell(cx, floor="#cfc8bd", wall="#f6f6f4", back="#f6f6f4")
    ceiling_lights(cx, "strip")
    shelves(cx, cx.a + 0.05, cx.b - 0.05, [0.3, 0.7, 1.1, 1.5, 1.9], 0.3, items="boxes")


def d_toys(cx):
    shell(cx, floor="#e4e0d8", wall="#fff8e7", back=cx.pick(["#ffd400", "#4fb3e8", "#fff8e7"]))
    ceiling_lights(cx, "spots")
    shelves(cx, cx.a + 0.05, cx.b - 0.05, [0.8, 1.3, 1.8], 0.25, items="boxes",
            cols=["#e30613", "#ffd400", "#0067b1", "#2e7d32", "#f39200", "#8e44ad"])
    for k in range(int((cx.W - 0.3) / 0.35)):
        if cx.D < 0.3:
            break
        cx.B.sphere("gloss", cx.pick(["#e30613", "#ffd400", "#0067b1", "#2e7d32"]), (cx.a + 0.3 + k * 0.35, cx.c + 0.12, cx.zf - 0.16), 0.11, k=8, n=4)
        cx.commit()


def d_vacant(cx, papered=True):
    shell(cx, floor="#8f8b85", wall="#e8e6e1", back="#dedbd5", glow_back=False)
    # layers from the glass inwards: notice print (zf-4), notice sheet (zf-7), papers (zf-10 / zf-13, alternating)
    z = cx.zf - 0.004
    if papered and cx.rng.random() < 0.7:
        x = cx.a + 0.01
        k = 0
        while x < cx.b - 0.25:
            w = min(cx.u(0.55, 0.85), cx.b - 0.01 - x)
            h = cx.u(0.6, 1.0)
            y = cx.c + cx.u(0.0, 0.5)
            cx.B.rect_z("matte", cx.pick(["#f3efe4", "#ece6d6", "#f7f5ef"]), x, y, x + w, min(cx.d - 0.02, y + h),
                        z - 2 * LAYER - LAYER * (k % 2))
            x += w + cx.u(-0.05, 0.06)
            k += 1
        cx.commit()
    if cx.W > 0.9:
        x = cx.a + cx.W * 0.5 - 0.3
        y = cx.c + 1.1
        if y + 0.85 < cx.d:
            cx.B.rect_z("matte", "#ffffff", x, y, x + 0.6, y + 0.85, z - LAYER)
            cx.B.rect_z("matte", "#d7261e", x + 0.04, y + 0.55, x + 0.56, y + 0.8, z)
            for k in range(3):
                cx.B.rect_z("matte", "#333333", x + 0.06, y + 0.1 + k * 0.12, x + 0.5 - 0.1 * (k % 2), y + 0.14 + k * 0.12, z)
            cx.commit()


def d_hallway(cx):
    """residential entrance: stair hall behind the door glass."""
    shell(cx, floor=cx.pick(["#8a8178", "#b9b0a3", "#6f6a64"]), wall=cx.pick(["#efe9dc", "#e8e4d8"]), back="#e2dccf",
          ceil="#f1efe9")
    y = cx.d - 0.002
    cx.B.sphere("lit", "#fff3d6", ((cx.a + cx.b) / 2, y - 0.09, (cx.zb + cx.zf) / 2), min(0.11, cx.D * 0.2), k=8, n=4, ry=0.08)
    cx.commit()
    if cx.D > 0.4 and cx.H > 1.2:
        # first steps of the stair at the back
        for k in range(4):
            y0 = cx.c + k * 0.17
            # stacked step blocks (each sits on the one below: no coplanar side faces of overlapping blocks)
            cx.B.box("matte", "#9c968e", (cx.a + 0.02, y0, cx.zb + 0.004), (cx.b - 0.02, y0 + 0.17, cx.zb + 0.004 + (4 - k) * 0.08),
                     skip=("-y", "-z"))
        cx.commit()
    if cx.W > 0.7 and cx.D > 0.2 and cx.H > 2.1:
        # letterboxes on the side wall
        x0 = cx.a + 0.004
        for r in range(3):
            for k in range(2):
                y0 = cx.c + 1.1 + r * 0.3
                z0 = cx.zf - 0.05 - (k + 1) * min(0.3, cx.D * 0.4)
                if z0 < cx.zb + 0.02:
                    continue
                cx.B.box("metal", "#a7abae", (x0, y0, z0), (x0 + 0.1, y0 + 0.26, z0 + min(0.28, cx.D * 0.38)), skip=("-x",))
        cx.commit()


def d_shallow(cx):
    """< 25 cm behind the glass: lit graphic back panel + light strip."""
    shell(cx, floor="#e4e0d8", wall="#f2f2f0", back="#f2f2f0")
    ceiling_lights(cx, "strip")
    if cx.W > 0.5 and cx.H > 0.8:
        poster(cx, cx.a + 0.05, cx.b - 0.05, cx.c + 0.15, cx.d - 0.15, cx.zb + LAYER, lit=False)


def d_doorway(cx):
    """behind a shop door: view into the sales floor (doormat, rear light, shelf with goods, brand graphic)."""
    shell(cx, floor=cx.pick(FLOOR), wall=cx.pick(WALL), back=cx.pick(WALL))
    ceiling_lights(cx, "spots")
    zf = cx.zf
    cx.B.box("matte", "#2c2c2c", (cx.a + 0.08, cx.c, zf - min(0.5, cx.D - 0.05)), (cx.b - 0.08, cx.c + 0.008, zf - 0.03), skip=("-y",))
    cx.commit()
    if cx.H > 2.0 and cx.W > 0.6:
        poster(cx, cx.a + 0.06, cx.b - 0.06, cx.c + 1.3, min(cx.d - 0.12, cx.c + 2.05), cx.zb + LAYER,
               cols=(cx.accent, "#ffffff", shade(cx.accent, 0.6) if luminance(hex2rgb(cx.accent)) > 0.3 else "#ffffff"),
               style=cx.pick(["band", "circle"]))
    if cx.D > 0.3 and cx.W > 0.6:
        shelves(cx, cx.a + 0.04, cx.b - 0.04, [0.45, 0.85], min(0.22, cx.D * 0.4), items="boxes",
                cols=[cx.accent, "#ffffff", "#2b2b2b", "#c9b79c", shade(cx.accent, 0.7), "#e9e6df"])


DISPLAY = {
    "fashion": d_fashion, "sports": lambda cx: d_fashion(cx, sport=True), "shoes": d_shoes, "jewellery": d_jewellery,
    "cafe": d_cafe, "restaurant": lambda cx: d_cafe(cx, restaurant=True), "fast_food": lambda cx: d_cafe(cx, fast=True),
    "bakery": d_bakery, "pharmacy": d_pharmacy, "bank": d_bank, "optician": d_optician, "phone": d_tech,
    "electronics": d_tech, "cosmetics": d_cosmetics, "drugstore": d_cosmetics, "books": d_books, "kiosk": d_kiosk,
    "supermarket": d_market, "department_store": d_fashion, "toys": d_toys, "vacant": d_vacant,
    "hallway": d_hallway, "shallow": d_shallow, "doorway": d_doorway, "hotel": d_hotel, "office": d_office,
    "travel": lambda cx: d_office(cx, travel=True), "estate": lambda cx: d_office(cx, estate=True),
    "houseware": d_houseware, "furniture": lambda cx: d_houseware(cx, furniture=True), "hair": d_hair,
    "confectionery": lambda cx: d_bakery(cx, confectionery=True),
}

CAT_KIND = {
    "hotel": "hotel", "travel_agency": "travel", "travel_agent": "travel", "estate_agent": "estate", "insurance": "office",
    "office": "office", "company": "office", "lawyer": "office", "hairdresser": "hair", "beauty": "hair",
    "houseware": "houseware", "interior_decoration": "houseware", "gift": "houseware", "furniture": "furniture",
    "bed": "furniture", "tea": "houseware", "coffee": "houseware", "chocolate": "confectionery",
    "confectionery": "confectionery", "tobacco": "jewellery", "watches": "jewellery", "pawnbroker": "jewellery",
    "car_rental": "office", "copyshop": "office", "bag": "fashion", "tailor": "fashion", "fabric": "houseware",
    "stationery": "books", "telecommunication": "phone", "mobile_phone": "phone", "hifi": "electronics",
    "medical_supply": "pharmacy", "hearing_aids": "optician", "perfumery": "cosmetics", "pub": "restaurant",
    "bar": "restaurant", "ice_cream": "cafe", "wine": "houseware", "deli": "bakery", "art": "houseware",
    "antiques": "houseware", "outdoor": "sports", "bicycle": "sports", "music": "electronics", "frame": "houseware",
    "photo": "electronics", "variety_store": "supermarket", "convenience": "kiosk", "newsagent": "kiosk",
    "bookmaker": "office", "money_lender": "office", "bank": "bank", "atm": "bank", "florist": "houseware",
}


def choose_kind(shop, opening, depth):
    if opening["kind"] == "entrance" and not shop:
        return "hallway"
    if depth < 0.25:
        return "shallow"
    if not shop:
        return None
    st = shop.get("status")
    if st in ("closed", "vacant", "likely_closed"):
        return "vacant"
    k = (shop.get("display") or {}).get("kind") or "other"
    cat = shop.get("category") or ""
    if k == "other" or k not in DISPLAY:
        k = CAT_KIND.get(cat, "houseware" if (shop.get("category_key") or "").startswith("shop=") else "office")
    if opening["kind"] in ("shopdoor", "entrance") and k not in ("cafe", "restaurant", "fast_food", "hotel", "bank"):
        return "doorway"
    return k


def generic_kind(rng, opening, depth, lod2):
    if opening["kind"] == "entrance":
        return "hallway"
    if depth < 0.25:
        return "shallow"
    if opening["kind"] == "shopdoor":
        return "doorway"
    return ["office", "houseware", "fashion", "market", "office", "cosmetics", "shoes"][int(rng.integers(7))] \
        if True else "office"


def d_pocket(cx):
    """narrow box (corner pocket between two facades' interior slabs, strip beside a column): closed lit shell,
    light strip, a plant or a small pedestal with a vase if it fits."""
    shell(cx, floor=cx.pick(["#cfc8bd", "#e4e0d8", "#bdb5a8"]), wall=cx.pick(WALL), back=cx.pick(WALL))
    ceiling_lights(cx, "strip")
    zc = (cx.zb + cx.zf) / 2
    xa, xb = cx.xr(zc, pad=0.03)
    if xb - xa > 0.3 and cx.D > 0.3:
        x = (xa + xb) / 2
        r = min(0.12, (xb - xa) / 2 - 0.04, cx.D / 2 - 0.05)
        if r > 0.06:
            if cx.rng.random() < 0.5:
                plant(cx, x, zc, r)
            else:
                h = min(0.95, cx.H - 0.6)
                cx.B.box("matte", "#f4f4f2", (x - r, cx.c, zc - r), (x + r, cx.c + h, zc + r), skip=("-y",))
                cx.B.frustum("gloss", cx.pick(["#3c5a40", "#b5651d", "#4a6b8a", "#111111"]), x, zc, r * 0.45, r * 0.3,
                             cx.c + h, cx.c + h + 0.22, k=10)
                cx.commit()


DISPLAY["pocket"] = d_pocket


def d_film(cx):
    """3-8 cm between glass and the model's dark interior panel: a closed shallow shell (side reveals, sill,
    lit head) whose back is a backlit window film with a printed graphic (no slot beside the film through which
    an oblique view would reach the open building shell)."""
    B = cx.B
    film = cx.pick(["#f2f2f0", "#ebe6dc", cx.bg if luminance(hex2rgb(cx.bg)) > 0.6 else "#efede8"])
    shell(cx, floor="#e4e0d8", wall="#f2f2f0", back=film)
    if cx.clip:
        return
    z = cx.zb
    a, b, c, d = cx.a + 0.004, cx.b - 0.004, cx.c + 0.004, cx.d - 0.004
    W, H = b - a, d - c
    if W > 0.6 and H > 1.2:
        x0 = a + W * 0.1
        x1 = b - W * 0.1
        poster(cx, x0, x1, c + H * 0.3, c + H * 0.85, z + LAYER, cols=(cx.accent, "#ffffff", shade(cx.accent, 0.6)),
               lit=True, style=cx.pick(["band", "circle", "photo", "bars"]))
    # frosted privacy band at the bottom
    B.rect_z("matte", "#dfe3e5", a, c, b, c + min(0.9, H * 0.3), z + 2 * LAYER)
    cx.commit()


DISPLAY["film"] = d_film


def build_interior(B, o, shop, rng, logo=False):
    box = o["box"]
    if 0.03 <= box["depth"] < 0.08 and o["kind"] == "window":
        cx = Ctx(B, o["O"], o["R"], box, rng, shop=shop, opening=o)
        d_film(cx)
        return "film"
    if box["depth"] < 0.08:
        return None
    kind = choose_kind(shop, o, box["depth"]) if shop else generic_kind(rng, o, box["depth"], o.get("lod2"))
    a, b, c, d = box["rect"]
    if d - c < 0.95 and kind not in ("vacant", "hallway", "shallow"):
        kind = "shallow"
    if b - a < 0.45 and kind not in ("vacant", "hallway"):
        kind = "pocket"
    if kind is None:
        kind = "pocket"
    if kind == "market":
        kind = "supermarket"
    cx = Ctx(B, o["O"], o["R"], box, rng, shop=shop, opening=o)
    cx.logo = bool(logo and shop and kind not in ("vacant", "hallway", "shallow", "doorway"))
    DISPLAY[kind](cx)
    return kind

"""Facade signs (fascia panel / channel letters / light box / blade), house-number plates, door bells.

Mounting: the facade above a shop's openings is sampled in 8 cm cells (exact triangle/prism clipping
against every building triangle).  The outermost dominant flat plane (sign band or wall) is found and the
largest flat rectangle is used; the sign sits 3 mm in front of the highest surface in that rectangle, so
it can neither intersect nor z-fight the existing geometry."""
import math

import numpy as np

from common import box_game_bounds, hex2rgb, luminance, prism_zrange, rgb2hex, to_local
from geom import text_shapes

STEP = 0.08


def fit_frame(R):
    return np.asarray(R, float)


class Mounter:
    def __init__(self, obst):
        self.obst = obst

    def tris_local(self, O, R, lo, hi):
        glo, ghi = box_game_bounds(O, R, lo, hi)
        idx = self.obst.query(glo - 0.01, ghi + 0.01)
        if not len(idx):
            return np.zeros((0, 3, 3)), idx
        return to_local(self.obst.T[idx].reshape(-1, 3), O, R).reshape(-1, 3, 3), idx

    def cells(self, O, R, x0, x1, y0, y1, zlo=-0.8, zhi=1.2, step=STEP, no_glass=True):
        """outermost surface z per cell (nan = nothing there). The material name of that outermost surface
        per cell is kept in self.last_mat (same shape) for colour decisions."""
        TL, gidx = self.tris_local(O, R, [x0, y0, zlo], [x1, y1, zhi])
        nx = max(1, int(round((x1 - x0) / step)))
        ny = max(1, int(round((y1 - y0) / step)))
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y0, y1, ny + 1)
        cz = np.full((ny, nx), np.nan)
        self.last_mat = np.full((ny, nx), "", dtype=object)
        if not len(TL):
            return xs, ys, cz
        tlo = TL.min(1)
        thi = TL.max(1)
        keep = (thi[:, 2] > zlo) & (tlo[:, 2] < zhi)
        TL, tlo, thi, gidx = TL[keep], tlo[keep], thi[keep], gidx[keep]
        for j in range(ny):
            rm = (tlo[:, 1] <= ys[j + 1]) & (thi[:, 1] >= ys[j])
            if not rm.any():
                continue
            Tr, lr, hr, gr = TL[rm], tlo[rm], thi[rm], gidx[rm]
            for i in range(nx):
                cm = (lr[:, 0] <= xs[i + 1]) & (hr[:, 0] >= xs[i])
                if not cm.any():
                    continue
                hit, zmin, zmax = prism_zrange(Tr[cm], xs[i], xs[i + 1], ys[j], ys[j + 1])
                ok = hit & (zmax > zlo) & (zmin < zhi)
                if ok.any():
                    k = int(np.argmax(np.where(ok, zmax, -np.inf)))
                    cz[j, i] = min(zmax[k], zhi)
                    self.last_mat[j, i] = self.obst.matname(gr[cm][k])
        if no_glass:
            # glazing is flat too, but nothing is ever mounted on the outside of a window pane
            isg = np.vectorize(lambda m: "glass" in (m or "").lower(), otypes=[bool])(self.last_mat)
            cz[isg] = np.nan
        return xs, ys, cz

    def surface_colour(self, xs, ys, rect):
        """dominant colour (from '#RRGGBB' in the material names) of the surface cells inside rect, or None."""
        import re
        from collections import Counter
        cx = (xs[1:] + xs[:-1]) / 2
        cy = (ys[1:] + ys[:-1]) / 2
        mi = (cx >= rect["x0"]) & (cx <= rect["x1"])
        mj = (cy >= rect["y0"]) & (cy <= rect["y1"])
        cnt = Counter()
        for nm in self.last_mat[np.ix_(mj, mi)].ravel():
            m = re.search(r"#([0-9A-Fa-f]{6})", nm or "")
            if m:
                cnt["#" + m.group(1).lower()] += 1
        return cnt.most_common(1)[0][0] if cnt else None

    def flat_rect(self, xs, ys, cz, xc=None, min_h=0.2, min_w=0.4, prefer_low=True, tol=0.016):
        """dominant outermost flat plane + best rectangle of flat cells -> dict or None."""
        v = cz[np.isfinite(cz)]
        if len(v) < 6:
            return None
        q = np.round(v / 0.01).astype(int)
        vals, cnt = np.unique(q, return_counts=True)
        # merge neighbours (+-1 cm)
        sc = np.array([cnt[np.abs(vals - k) <= 1].sum() for k in vals])
        good = sc >= max(6, 0.12 * len(v))
        if not good.any():
            return None
        zp = vals[good].max() * 0.01 if True else None
        # prefer a plane that is not tiny compared to the most frequent one
        if sc[vals == vals[good].max()][0] < 0.35 * sc.max():
            cand = vals[good & (sc >= 0.35 * sc.max())]
            zp = cand.max() * 0.01
        F = np.isfinite(cz) & (np.abs(cz - zp) <= tol)
        ny, nx = F.shape
        best = None
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        for r0 in range(ny):
            run = np.ones(nx, bool)
            for r1 in range(r0, ny):
                run &= F[r1]
                h = (r1 - r0 + 1) * dy
                if not run.any():
                    break
                # runs of True
                idx = np.where(np.diff(np.r_[0, run.astype(int), 0]) != 0)[0].reshape(-1, 2)
                for s, e in idx:
                    w = (e - s) * dx
                    if w < min_w or h < min_h:
                        continue
                    xa, xb = xs[s], xs[e]
                    score = w * min(h, 0.75)
                    if xc is not None:
                        score -= 0.4 * max(0.0, abs((xa + xb) / 2 - xc) - 0.3)
                    if prefer_low:
                        score -= 0.5 * (ys[r0] - ys[0])
                    if best is None or score > best[0]:
                        zmax = float(np.nanmax(cz[r0:r1 + 1, s:e]))
                        best = (score, dict(x0=xa, x1=xb, y0=ys[r0], y1=ys[r1 + 1], z=zmax, zp=zp))
        return best[1] if best else None

    def volume_free(self, O, R, lo, hi, ignore_below=None):
        """no triangle of the scene intersects the local box lo..hi (conservative z-interval test)."""
        TL, _ = self.tris_local(O, R, lo, hi)
        if not len(TL):
            return True
        hit, zmin, zmax = prism_zrange(TL, lo[0], hi[0], lo[1], hi[1])
        m = hit & (zmax > lo[2]) & (zmin < hi[2])
        return not m.any()


# ----------------------------------------------------------------------------- text placement
def place_text(B, text, font, cap, origin, f, up=(0, 1, 0), depth=0.04, mat="lit", col="#ffffff", side_mat="matte",
               side_col=None, squeeze=1.0, align="center"):
    """text standing on baseline `origin`, read by a viewer looking along -f; front face at +depth along f.
    origin = left/centre point of the baseline (align)."""
    polys, w, capv, desc = text_shapes(text, font, cap, squeeze=squeeze)
    if not polys:
        return 0.0
    f = np.asarray(f, float)
    up = np.asarray(up, float)
    right = np.cross(-f, up)
    mk = B.mark()
    from shapely import affinity
    dx = -w / 2 if align == "center" else (-w if align == "right" else 0.0)
    polys = [affinity.translate(p, dx, 0) for p in polys]
    # letters thinner than 6.5 mm (vinyl, printed plates, back-wall logos): no side walls - invisible, but costly
    B.extrude2d(mat, col, polys, 0.0, depth, side_mat=side_mat, side_col=side_col if side_col else shade(col, 0.6),
                sides=depth > 0.0065)
    A = np.stack([right, up, f], 1)
    B.xform_since(mk, A, np.asarray(origin, float))
    return w


def shade(c, f):
    return rgb2hex(np.clip(hex2rgb(c) * f, 0, 1))


def contrast(fg, bg):
    return abs(luminance(hex2rgb(fg)) - luminance(hex2rgb(bg)))


def rel_lum(c):
    c = hex2rgb(c)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def wcag(fg, bg):
    a, b = rel_lum(fg), rel_lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def trusted(sg):
    """sign colours documented from Street View notes or the brand's CI: kept as long as they are readable."""
    src = (sg.get("source") or "") + " " + (sg.get("colour_source") or "")
    return "streetview" in src or "brand_ci" in src or "ref_photo" in src


def text_colour(sg, fg, bg):
    """fg on bg, unless unreadable: WCAG ratio >= 3 (large text) for guessed colours, >= 2.2 for documented ones
    (e.g. TUI red #d40e14 on #1a1a1a, ratio 3.2, stays red)."""
    need = 2.2 if trusted(sg) else 3.0
    if wcag(fg, bg) >= need:
        return fg
    return "#ffffff" if wcag("#ffffff", bg) >= wcag("#1a1a1a", bg) else "#1a1a1a"


def logo_mark(B, sg, x, ybase, cap, z, col):
    """small brand mark left of the text (no copied logo, a simple shape in the brand colour):
    'smile' (red arc + dot, TUI-like), 'star' (white six-point star disc), 'disc'. Returns its width."""
    shape = (sg.get("logo") or {}).get("shape")
    if not shape:
        return 0.0
    from shapely.geometry import Point, Polygon
    r = cap * 0.62
    cxm, cym = x + r, ybase + cap * 0.5
    if shape == "smile":
        ring = Point(cxm, cym).buffer(r, 24).difference(Point(cxm, cym).buffer(r * 0.72, 24))
        lower = Polygon([(cxm - 2 * r, cym - 2 * r), (cxm + 2 * r, cym - 2 * r), (cxm + 2 * r, cym - r * 0.15),
                         (cxm - 2 * r, cym - r * 0.15)])
        arc = ring.intersection(lower)
        dot = Point(cxm, cym + r * 0.42).buffer(r * 0.2, 12)
        geoms = [g for g in getattr(arc, "geoms", [arc]) if g.geom_type == "Polygon"] + [dot]
    elif shape == "star":
        import math as _m
        pts = []
        for k in range(12):
            rr = r if k % 2 == 0 else r * 0.5
            a = _m.pi / 2 + k * _m.pi / 6
            pts.append((cxm + rr * _m.cos(a), cym + rr * _m.sin(a)))
        geoms = [Polygon(pts)]
    else:
        geoms = [Point(cxm, cym).buffer(r, 20)]
    B.extrude2d("lit", (sg.get("logo") or {}).get("col") or col, geoms, z, z + 0.006, side_col=shade(col, 0.6))
    return 2 * r + cap * 0.35


def sign_text(shop):
    sg = shop.get("sign") or {}
    t = sg.get("text") or shop.get("name") or ""
    t = t.replace("’", "'").strip()
    return t


def fit_text(text, font, wmax, hmax, cap_max=0.45, cap_min=0.09, squeeze=1.0):
    """-> (lines, cap) fitting into wmax x hmax."""
    cap = min(cap_max, hmax * 0.62)
    w = text_shapes(text, font, cap, squeeze=squeeze)[1]
    if w > wmax:
        cap *= wmax / w
    best = ([text], cap)
    words = text.split()
    if cap < 0.17 and len(words) > 1 and hmax > 0.4:
        # two lines
        bestsplit = None
        for k in range(1, len(words)):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            c2 = min(cap_max * 0.85, hmax * 0.36)
            wa = text_shapes(a, font, c2, squeeze=squeeze)[1]
            wb = text_shapes(b, font, c2, squeeze=squeeze)[1]
            ww = max(wa, wb)
            if ww > wmax:
                c2 *= wmax / ww
            if bestsplit is None or c2 > bestsplit[1]:
                bestsplit = ([a, b], c2)
        if bestsplit and bestsplit[1] > cap * 1.15:
            best = bestsplit
    if best[1] < cap_min:
        return None
    return best


def fascia(B, M, O, R, rect, shop, kind, rng, wall_col=None, cap_fix=None):
    """sign on a flat rectangle rect (local frame of O,R). kind: fascia_panel / channel_letters / light_box / logo_only.
    wall_col: colour of the facade surface behind (channel letters must stand out against it).
    cap_fix: common cap height of the signs on one shared shop front (same baseline + letter size).
    sign.backing: dark raceway band behind channel letters; sign.logo: small brand mark left of the text."""
    sg = shop.get("sign") or {}
    fg = sg.get("fg") or "#ffffff"
    bg = sg.get("bg") or "#1a1a1a"
    fg = text_colour(sg, fg, bg)
    backing = (sg.get("backing") or {}).get("col") if kind == "channel_letters" else None
    if backing:
        fg = text_colour(sg, sg.get("fg") or "#ffffff", backing)
    elif kind in ("channel_letters", "logo_only") and wall_col and contrast(fg, wall_col) < 0.3:
        # letters mounted straight on the wall / sign band: brand colour if it reads, else white or black
        opts = [c for c in (bg, "#ffffff", "#1a1a1a") if contrast(c, wall_col) >= 0.3]
        fg = opts[0] if opts else ("#ffffff" if luminance(hex2rgb(wall_col)) < 0.5 else "#1a1a1a")
    font = sg.get("font") or "sans"
    text = sign_text(shop)
    if not text:
        return False
    x0, x1, y0, y1, z0 = rect["x0"], rect["x1"], rect["y0"], rect["y1"], rect["z"] + 0.003
    W, H = x1 - x0, y1 - y0
    xc = (x0 + x1) / 2
    if kind == "light_box":
        bh = min(H - 0.04, 0.62)
        res = fit_text(text, font, min(W - 0.1, 4.5) - 0.2, bh * 0.95, cap_max=0.34)
        if not res:
            return False
        lines, cap = res
        tw = max(text_shapes(ln, font, cap)[1] for ln in lines)
        bw = min(W - 0.04, tw + 0.35)
        yb = y0 + (H - bh) / 2
        B.box("matte", "#3a3d40", (xc - bw / 2, yb, z0), (xc + bw / 2, yb + bh, z0 + 0.12), skip=("-z",),
              cols={"+z": ("lit", bg)})
        zt = z0 + 0.12
        mat_t, dep = ("matte", 0.004) if luminance(hex2rgb(fg)) < 0.35 else ("lit", 0.004)
    elif kind == "channel_letters":
        res = fit_text(text, font, W - 0.2 - (0.9 * min(H, 0.5) if sg.get("logo") else 0.0), H * 1.12, cap_max=0.5)
        if not res:
            return False
        lines, cap = res
        if cap_fix and cap_fix < cap and len(lines) == 1:
            cap = cap_fix
        zt = z0
        mat_t, dep = "lit", 0.055
        tw = max(text_shapes(ln, font, cap)[1] for ln in lines)
        if backing:
            # dark raceway band behind the letters (letters on a dark facade band)
            nl_ = len(lines)
            bh = min(H - 0.02, cap * (nl_ + 0.45 * (nl_ - 1)) + 0.24)
            ybk = y0 + (H - bh) / 2
            bw = min(W - 0.04, tw + 0.4 + (cap * 1.6 if sg.get("logo") else 0.0))
            B.box("matte", backing, (xc - bw / 2, ybk, z0), (xc + bw / 2, ybk + bh, z0 + 0.02), skip=("-z",))
            zt = z0 + 0.02
        else:
            # thin mounting rail, painted in the wall colour (a brand-coloured rail reads as a strike-through line)
            yr = y0 + H / 2 - 0.012
            B.box("matte", shade(wall_col, 0.9) if wall_col else "#8c9196", (xc - tw / 2 - 0.04, yr, z0),
                  (xc + tw / 2 + 0.04, yr + 0.024, z0 + 0.015), skip=("-z",))
    else:   # fascia_panel / logo_only
        ph = min(H - 0.03, 0.75)
        res = fit_text(text, font, W - 0.3 - (0.9 * min(ph, 0.5) if sg.get("logo") else 0.0), ph, cap_max=0.4)
        if not res:
            return False
        lines, cap = res
        if cap_fix and cap_fix < cap and len(lines) == 1:
            cap = cap_fix
        yb = y0 + (H - ph) / 2
        B.box("matte", bg, (x0 + 0.02, yb, z0), (x1 - 0.02, yb + ph, z0 + 0.03), skip=("-z",))
        # thin edge trim
        B.box("metal", "#8c9196", (x0 + 0.02, yb + ph - 0.012, z0 + 0.03), (x1 - 0.02, yb + ph, z0 + 0.034), skip=("-z",))
        zt = z0 + 0.03
        mat_t, dep = ("lit" if rng.random() < 0.5 else "gloss"), 0.012
        if kind == "logo_only":
            s = ph * 0.7
            ex = x0 + 0.15
            B.box("lit", fg, (ex, yb + (ph - s) / 2, zt), (ex + s, yb + (ph + s) / 2, zt + 0.01), skip=("-z",))
    # text lines (with a brand mark: mark + text centred together)
    nl = len(lines)
    gap = cap * 0.45
    tot = nl * cap + (nl - 1) * gap
    ybase = y0 + (H - tot) / 2 + (nl - 1) * (cap + gap)
    xt = xc
    if sg.get("logo") and nl == 1:
        tw = text_shapes(lines[0], font, cap)[1]
        lw = cap * 1.6
        x_l = xc - (tw + lw) / 2
        logo_mark(B, sg, x_l, ybase, cap, zt, fg)
        xt = x_l + lw + tw / 2
    for ln in lines:
        place_text(B, ln, font, cap, (xt, ybase, zt), (0, 0, 1), depth=dep, mat=mat_t, col=fg,
                   side_mat="matte", side_col=shade(fg, 0.55))
        ybase -= cap + gap
    return B.commit(O, R)


def blade(B, mt, O, R, x, ytop, zwall, shop, rng, side=1, size=None, pharmacy=False):
    """projecting sign perpendicular to the facade at local x, hanging below ytop; two faces with text."""
    sg = shop.get("sign") or {}
    bl = sg.get("blade") or {}
    fg = bl.get("fg") or sg.get("fg") or "#ffffff"
    bg = bl.get("bg") or sg.get("bg") or "#1a1a1a"
    rule = bl.get("rule")
    if pharmacy:
        fg, bg, rule = "#e2001a", "#ffffff", None
    fg = text_colour(sg, fg, bg)
    font = sg.get("font") or "sans"
    text = "A" if pharmacy else sign_text(shop)
    L = 0.62 if pharmacy else (size or 0.75)
    Hb = 0.62 if pharmacy else 0.45
    t = 0.05
    zg = 0.10          # gap wall -> panel (bracket)
    y1 = ytop
    y0 = ytop - Hb
    # the bracket plate needs wall right behind it (flat, at zwall) ...
    xs, ys, cz = mt.cells(O, R, x - 0.04, x + 0.04, y1 - 0.13, y1 + 0.01, step=0.04)
    if not np.isfinite(cz).all() or (np.nanmax(cz) - np.nanmin(cz)) > 0.012 or cz.max() > 0.6 or cz.min() < -0.6:
        return False
    zwall = float(np.nanmax(cz)) + 0.003
    # ... and the whole sign volume must be free
    lo = np.array([x - t / 2 - 0.02, y0 - 0.02, zwall + 0.002])
    hi = np.array([x + t / 2 + 0.02, y1 + 0.02, zwall + zg + L + 0.02])
    if not mt.volume_free(O, R, lo, hi):
        return False
    # bracket
    B.box("metal", "#3a3d40", (x - 0.012, y1 - 0.03, zwall), (x + 0.012, y1, zwall + zg + L * 0.98), skip=("-z",))
    B.box("metal", "#3a3d40", (x - 0.03, y1 - 0.12, zwall), (x + 0.03, y1 - 0.004, zwall + 0.012), skip=("-z",))
    # panel (lit edges = light box)
    B.box("matte", "#2a2c2e", (x - t / 2, y0, zwall + zg), (x + t / 2, y1 - 0.035, zwall + zg + L),
          cols={"+x": ("lit", bg), "-x": ("lit", bg)})
    hh = y1 - 0.035 - y0
    for f in (1, -1):
        fx = x + f * (t / 2)
        if pharmacy:
            cap = hh * 0.62
            place_text(B, "A", "serif", cap, (fx, y0 + (hh - cap) / 2, zwall + zg + L / 2), (f, 0, 0), depth=0.006,
                       mat="lit", col=fg, side_col=shade(fg, 0.6))
        else:
            rr = 0.06 if rule else 0.0          # room for the rule under the name
            res = fit_text(text, font, L - 0.08, hh - 0.06 - rr, cap_max=0.2, cap_min=0.045)
            if not res:
                continue
            lines, cap = res
            nl = len(lines)
            gap = cap * 0.4
            yb = y0 + rr + (hh - rr - (nl * cap + (nl - 1) * gap)) / 2 + (nl - 1) * (cap + gap)
            for ln in lines:
                place_text(B, ln, font, cap, (fx, yb, zwall + zg + L / 2), (f, 0, 0), depth=0.005, mat="lit", col=fg,
                           side_col=shade(fg, 0.6))
                yb -= cap + gap
            if rule:
                # brand-coloured rule under the name (e.g. WEITZ: green line on the black blade), 4 mm proud
                yr = yb + cap + gap - 0.03
                tw = max(text_shapes(ln, font, cap)[1] for ln in lines)
                xa_, xb_ = (fx, fx + 0.004) if f > 0 else (fx - 0.004, fx)
                B.box("lit", rule, (xa_, yr - 0.016, zwall + zg + L / 2 - tw / 2), (xb_, yr, zwall + zg + L / 2 + tw / 2),
                      skip=("-x",) if f > 0 else ("+x",))
    return B.commit(O, R)


def number_plate(B, O, R, rect, num, style="enamel"):
    x0, x1, y0, y1, z0 = rect["x0"], rect["x1"], rect["y0"], rect["y1"], rect["z"] + 0.003
    pw = min(0.30, x1 - x0 - 0.02)
    ph = min(0.21, y1 - y0 - 0.02)
    if pw < 0.16 or ph < 0.12:
        return False
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    if style == "enamel":
        B.box("gloss", "#1c3f8f", (xc - pw / 2, yc - ph / 2, z0), (xc + pw / 2, yc + ph / 2, z0 + 0.008), skip=("-z",))
        bw = 0.012
        for (a, b, c, d) in ((xc - pw / 2 + 0.01, yc + ph / 2 - 0.01 - bw, xc + pw / 2 - 0.01, yc + ph / 2 - 0.01),
                             (xc - pw / 2 + 0.01, yc - ph / 2 + 0.01, xc + pw / 2 - 0.01, yc - ph / 2 + 0.01 + bw),
                             (xc - pw / 2 + 0.01, yc - ph / 2 + 0.01, xc - pw / 2 + 0.01 + bw, yc + ph / 2 - 0.01),
                             (xc + pw / 2 - 0.01 - bw, yc - ph / 2 + 0.01, xc + pw / 2 - 0.01, yc + ph / 2 - 0.01)):
            B.rect_z("gloss", "#ffffff", a, b, c, d, z0 + 0.011)
        res = fit_text(num, "bold-sans", pw - 0.07, ph - 0.05, cap_max=0.12, cap_min=0.04)
        if res:
            lines, cap = res
            place_text(B, lines[0], "bold-sans", cap, (xc, yc - cap / 2, z0 + 0.008), (0, 0, 1), depth=0.003, mat="gloss",
                       col="#ffffff")
    else:
        res = fit_text(num, "sans", pw, ph, cap_max=0.16, cap_min=0.05)
        if not res:
            return False
        lines, cap = res
        place_text(B, lines[0], "sans", cap, (xc, yc - cap / 2, z0), (0, 0, 1), depth=0.012, mat="metal", col="#c9cdd0",
                   side_col="#9a9ea2")
    return B.commit(O, R)


def doorbell(B, O, R, rect, n_bells=6):
    x0, x1, y0, y1, z0 = rect["x0"], rect["x1"], rect["y0"], rect["y1"], rect["z"] + 0.003
    pw, ph = 0.13, min(0.42, 0.11 + 0.045 * n_bells)
    if x1 - x0 < pw + 0.02 or y1 - y0 < ph + 0.02:
        return False
    xc = (x0 + x1) / 2
    yt = y1 - 0.01
    B.box("metal", "#b9bdc0", (xc - pw / 2, yt - ph, z0), (xc + pw / 2, yt, z0 + 0.01), skip=("-z",))
    # speaker
    B.disc("matte", "#2b2b2b", (xc, yt - 0.05, z0 + 0.013), 0.03, (0, 0, 1), k=10)
    for k in range(n_bells):
        yy = yt - 0.11 - k * 0.045
        B.rect_z("matte", "#f4f4f0", xc - 0.055, yy - 0.012, xc + 0.02, yy + 0.012, z0 + 0.013)
        B.box("metal", "#d9dcde", (xc + 0.03, yy - 0.01, z0 + 0.01), (xc + 0.05, yy + 0.01, z0 + 0.016), skip=("-z",))
    return B.commit(O, R)

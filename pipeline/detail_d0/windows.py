"""Window types (local frame, opening [0,w]x[0,h], wall face z=0, z out of the wall).

make_window(spec, rng) -> Builder with all assembly geometry (frame, sashes, glass, sill, trim,
shutters, interior box, textiles). The plaster reveal itself is built by the wall stage."""
import numpy as np

import assembly as A
from assembly import Builder

FD = 0.07           # frame depth
SASH = dict(ov=0.012, sw=0.074, sd=0.068, glass_z=-0.034, sash_proud=0.004)


def sill_top(z, d_f, kind):
    """y of the sill top at depth z (z<0 inside the reveal)."""
    slope = 0.11 if kind == "stone" else 0.08
    return 0.004 - slope * z


def add_sill(B, M, s, w):
    d_f = s["d_f"]
    kind = s["sill"]
    proj = 0.04
    depth_in = d_f + 0.02
    rise = sill_top(-depth_in, d_f, kind)
    drop = sill_top(proj, d_f, kind)
    e = s.get("sill_ears", 0.03)
    if kind == "stone":
        thick = s.get("sill_thick", 0.05)
        sec = A.sill_section(depth_in, proj, thick, drop, rise)
        B.extrude(M["sill_stone"], sec, "x", -e, w + e)
    else:
        sec = A.sill_section(depth_in, proj, 0.0, drop, rise, metal=True)
        B.extrude(M["sill_metal"], sec, "x", -e, w + e)
        # end caps (Bordstuecke) rising at both ends
        for x0, x1 in ((-e - 0.004, -e + 0.002), (w + e - 0.002, w + e + 0.004)):
            B.box(M["sill_metal"], (x0, drop - 0.02, -0.0), (x1, drop + 0.03, proj + 0.004))
    return sill_top(-d_f, d_f, kind)


def add_trim(B, M, s, w, h, ys):
    """U-shaped plaster/concrete surround in front of the wall (legs down to ys)."""
    tw, tt = s["trim_w"], s["trim_t"]
    m = M["trim"]
    outer = [(-tw, ys), (0, ys), (0, h), (w, h), (w, ys), (w + tw, ys), (w + tw, h + tw), (-tw, h + tw)]
    B.poly(m, [(x, y, tt) for x, y in outer], (0, 0, 1))
    # outer faces (embedded 1 cm into the wall)
    zb = -0.01
    def side(p, q, out):
        B.quad(m, (p[0], p[1], zb), (q[0], q[1], zb), (q[0], q[1], tt), (p[0], p[1], tt), expect=np.array(out, float))
    side((-tw, ys), (-tw, h + tw), (-1, 0, 0))
    side((-tw, h + tw), (w + tw, h + tw), (0, 1, 0))
    side((w + tw, h + tw), (w + tw, ys), (1, 0, 0))
    side((-tw, ys), (0, ys), (0, -1, 0))
    side((w, ys), (w + tw, ys), (0, -1, 0))
    # inner faces z in [0, tt], following the hole subdivision (no T-junctions with the reveal)
    hole = s.get("hole_local")
    def inner(pts, out):
        for a, b in zip(pts[:-1], pts[1:]):
            B.quad(m, (a[0], a[1], a[2]), (b[0], b[1], b[2]), (b[0], b[1], tt), (a[0], a[1], tt), expect=np.array(out, float))
    if hole is not None:
        L = hole["left"][::-1]      # bottom -> top at x=0
        R = hole["right"]           # bottom -> top at x=w
        T = hole["top"][::-1]       # left -> right at y=h
        inner([(0, ys, 0.0)] + [tuple(p) for p in L], (1, 0, 0))
        inner([(w, ys, 0.0)] + [tuple(p) for p in R], (-1, 0, 0))
        inner([tuple(p) for p in T], (0, -1, 0))
    else:
        inner([(0, ys, 0), (0, h, 0)], (1, 0, 0))
        inner([(w, ys, 0), (w, h, 0)], (-1, 0, 0))
        inner([(0, h, 0), (w, h, 0)], (0, -1, 0))


def frame_and_lights(B, M, s, x0, y0, x1, y1, zf, layout, rng):
    """Frame ring [x0,x1]x[y0,y1] at front z=zf plus the sash layout.
    layout: dict(cols=[('sash'|'fixed', width_frac)], stulp=bool, transom=y or None, upper='sash'|'fixed'|'split',
                 bars=int, handle=True, mull_w=0.08)"""
    fw = s.get("fw", 0.068)
    prof = A.shop_profile(fw, FD + 0.01) if s["kind"] in ("shop", "door") else A.frame_profile(fw, FD)
    B.ring(M["joinery"], x0, y0, x1, y1, prof, zref=zf)
    ix0, iy0, ix1, iy1 = x0 + fw, y0 + fw, x1 - fw, y1 - fw
    mw = layout.get("mull_w", 0.08)
    bd = FD - 0.004
    # transom
    ty = layout.get("transom")
    lower_top = iy1
    if ty is not None:
        A.bar_h(B, M, ty, ix0, ix1, zf, mw, bd)
        lower_top = ty - mw / 2
        up_y0 = ty + mw / 2
        upper = layout.get("upper", "sash")
        if upper == "split" and layout.get("cols") and len(layout["cols"]) > 1 and not layout.get("stulp"):
            xs = _col_edges(layout["cols"], ix0, ix1, mw)
            for (a, b) in xs:
                A.fixed_light(B, M, a, up_y0, b, iy1, zf, -0.03)
        elif upper == "fixed":
            A.fixed_light(B, M, ix0, up_y0, ix1, iy1, zf, -0.03)
        else:
            tilt = s.get("tilt_top", 0.0)
            A.sash(B, M, ix0, up_y0, ix1, iy1, zf, SASH, bars=0, vbar=layout.get("upper_vbar", False), tilt=tilt)
    cols = layout["cols"]
    bars = layout.get("bars", 0)
    hs = layout.get("handle", True)
    ph = layout.get("panel_h")
    edges = _col_edges(cols, ix0, ix1, mw) if not layout.get("stulp") else [(ix0, ix1)]
    bay_y0 = [iy0] * len(edges)
    if ph:
        # opaque parapet panels (Bruestungspaneel) below a horizontal bar, per bay (not in door bays)
        yp = iy0 + ph
        zp = zf - 0.03
        bead = [(0.0, -0.012), (0.0, 0.0), (0.004, 0.0), (0.018, -0.010), (0.018, -0.012)]
        bays = edges if not layout.get("stulp") else [(ix0, ix1)]
        for k, (a, b) in enumerate(bays):
            if not layout.get("stulp") and cols[k][0] == "door":
                continue
            A.bar_h(B, M, yp, a, b, zf, mw, bd)
            B.quad(M["spandrel"], (a, iy0, zp), (b, iy0, zp), (b, yp, zp), (a, yp, zp), expect=np.array([0, 0, 1.0]))
            B.ring(M["joinery"], a, iy0, b, yp - mw / 2, bead, zref=zp + 0.012)
            bay_y0[k] = yp + mw / 2
        if layout.get("stulp"):
            iy0 = yp + mw / 2
    if layout.get("stulp") and len(cols) == 2:
        xm = (ix0 + ix1) / 2
        ov = SASH["ov"]
        A.sash(B, M, ix0, iy0, xm - ov, lower_top, zf, SASH, bars=bars, handle_side=None, tilt=s.get("tilt", 0.0))
        A.sash(B, M, xm + ov, iy0, ix1, lower_top, zf, SASH, bars=bars, handle_side="L" if hs else None)
        # Schlagleiste on the meeting stile
        zs = zf + SASH["sash_proud"]
        B.extrude(M["joinery"], [(xm - 0.024, zs), (xm + 0.024, zs), (xm + 0.024, zs + 0.010), (xm + 0.018, zs + 0.016),
                                 (xm - 0.018, zs + 0.016), (xm - 0.024, zs + 0.010)], "y", iy0 - SASH["ov"] + 0.004,
                  lower_top + SASH["ov"] - 0.004)
        return
    for k in range(len(edges) - 1):
        xm = (edges[k][1] + edges[k + 1][0]) / 2
        A.bar_v(B, M, xm, iy0, lower_top, zf, mw, bd)
    for k, ((a, b), (typ, _)) in enumerate(zip(edges, cols)):
        y0b = bay_y0[k]
        if typ in ("fixed", "panel"):
            A.fixed_light(B, M, a, y0b, b, lower_top, zf, -0.03)
        elif typ == "door":
            door_leaves(B, M, a, y0b, b, lower_top, zf, rng)
        else:
            side = "R" if k < len(cols) / 2 - 0.01 else "L"
            if len(cols) == 1:
                side = "R" if rng.random() < 0.5 else "L"
            A.sash(B, M, a, y0b, b, lower_top, zf, SASH, bars=bars, handle_side=side if hs else None,
                   tilt=s.get("tilt", 0.0) if k == 0 else 0.0)


DOOR = dict(ov=0.012, sw=0.115, sd=0.07, glass_z=-0.036, sash_proud=0.004)


def door_leaves(B, M, x0, y0, x1, y1, zf, rng):
    """glazed aluminium door: one or two leaves, kick plate, push bars outside, pull handle."""
    wd = x1 - x0
    ov = DOOR["ov"]
    if wd >= 1.6:
        xm = (x0 + x1) / 2
        leaves = [(x0, xm - ov, "R"), (xm + ov, x1, "L")]
    else:
        leaves = [(x0, x1, "R" if rng.random() < 0.5 else "L")]
    zs = zf + DOOR["sash_proud"]
    for a, b, lock in leaves:
        A.sash(B, M, a, y0, b, y1, zf, DOOR, bars=0, handle_side=None)
        # kick plate in front of the glass
        X0, X1 = a - ov + DOOR["sw"] - 0.01, b + ov - DOOR["sw"] + 0.01
        Y0 = y0 - ov + DOOR["sw"] - 0.01
        B.box(M["metal"], (X0, Y0, zs + DOOR["glass_z"] - 0.002), (X1, Y0 + 0.24, zs - 0.012))
        # vertical push bar on stand-offs at the lock side
        xb = (b + ov - DOOR["sw"] / 2) if lock == "R" else (a - ov + DOOR["sw"] / 2)
        yb0, yb1 = y0 + 0.75, min(y1 - 0.2, y0 + 1.75)
        B.extrude(M["metal"], [(xb - 0.014, zs + 0.045), (xb + 0.014, zs + 0.045), (xb + 0.014, zs + 0.072),
                               (xb - 0.014, zs + 0.072)], "y", yb0, yb1)
        for yy in (yb0 + 0.08, yb1 - 0.08):
            B.box(M["metal"], (xb - 0.009, yy - 0.009, zs), (xb + 0.009, yy + 0.009, zs + 0.046), skip=("-z",))


def _col_edges(cols, ix0, ix1, mw):
    n = len(cols)
    avail = ix1 - ix0 - (n - 1) * mw
    tot = sum(f for _, f in cols)
    out = []
    x = ix0
    for _, f in cols:
        wd = avail * f / tot
        out.append((x, x + wd))
        x += wd + mw
    return out


def make_window(s, rng):
    B = Builder()
    M = s["M"]
    w, h = s["w"], s["h"]
    kind = s["kind"]
    d_f = s["d_f"]
    zf = -d_f
    # ---- sill / threshold
    if s.get("sill"):
        yb = add_sill(B, M, s, w)
    else:
        yb = 0.0
    # ---- trim / surround
    if s.get("trim"):
        ys = -0.05 if s.get("sill") == "stone" else (-0.035 if s.get("sill") else 0.0)
        add_trim(B, M, s, w, h, ys)
    # ---- roller shutter box (top of the reveal)
    yt = h
    if s.get("shutter"):
        hb = s.get("box_h", 0.17)
        yt = h - hb
        sec = [(h, zf - 0.004), (yt, zf - 0.004), (yt, -0.042), (yt + 0.012, -0.03), (h, -0.03)]
        B.extrude(M["shutter_box"], sec, "x", -0.004, w + 0.004)
        # guide rails on the jambs
        zsh = -(0.03 + (d_f - 0.03) * 0.45)
        for xa, xb in ((0.0, 0.024), (w - 0.024, w)):
            B.box(M["metal"], (xa, yb - 0.002, zsh - 0.016), (xb, yt, zsh + 0.016))
        A.roller_shutter(B, M, 0.012, w - 0.012, yt, yb, s.get("shutter_frac", 0.0), zsh)
    # ---- frame + lights
    layout = s["layout"]
    frame_and_lights(B, M, s, 0.0, yb, w, yt, zf, layout, rng)
    # ---- railing for floor-to-ceiling narrow windows
    if s.get("railing"):
        zr = -0.05
        for yy in s["railing"]:
            B.extrude(M["metal"], [(yy - 0.012, zr - 0.012), (yy + 0.012, zr - 0.012), (yy + 0.012, zr + 0.012),
                                   (yy - 0.012, zr + 0.012)], "x", -0.01, w + 0.01, caps=False)
    # ---- door push bar (outside)
    if s.get("pushbar"):
        x, y0, y1 = s["pushbar"]
        z0 = zf + 0.012
        B.extrude(M["metal"], [(x - 0.013, z0 + 0.05), (x + 0.013, z0 + 0.05), (x + 0.013, z0 + 0.075), (x - 0.013, z0 + 0.075)],
                  "y", y0, y1)
        for yy in (y0 + 0.08, y1 - 0.08):
            B.box(M["metal"], (x - 0.008, yy - 0.008, z0), (x + 0.008, yy + 0.008, z0 + 0.052))
    # ---- interior
    zi = zf - FD
    D = s["D_int"]
    imat = "shop_interior" if kind == "shop" else "interior"
    if D > 0.02:
        A.interior_box(B, M, 0.0, 0.0, w, h, zi, D, mat=imat)
    else:
        # no room behind: dark back plate right behind the frame
        B.quad(M["interior"], (0, 0, zi), (w, 0, zi), (w, h, zi), (0, h, zi), expect=np.array([0, 0, 1.0]))
    if kind == "shop" and D > 0.5:
        # simple fit-out: counter and shelves against the back wall
        zb = zi - D
        for k in range(3):
            yy = 0.5 + 0.55 * k
            if yy + 0.03 < h - 0.2:
                B.box(M["shop_fit"], (0.1, yy, zb), (w - 0.1, yy + 0.03, zb + 0.35))
        if D > 1.0:
            B.box(M["shop_fit"], (w * 0.2, 0.0, zi - D * 0.55), (w * 0.55, 0.95, zi - D * 0.35))
    else:
        kindc = s.get("textile", "none")
        if kindc != "none" and D > 0.14:
            A.curtains(B, M, rng, 0.02, 0.02, w - 0.02, h - 0.02, zi - min(0.07, D * 0.4), kindc)
    return B

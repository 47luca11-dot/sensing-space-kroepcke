"""Signage + display-window overlay for the Rhino detail model -> data/detail_raw/signage.glb (+ web copy).

  python build.py [--reextract] [--reassign] [--no-web] [--web-only]

  1. openings.py   (cache/openings.json)  shopfront glass + free box behind it   (--reextract re-runs it)
  2. assign.py     (cache/assign.json)    shops.json -> openings
  3. interiors per opening (display kind from shops.json), door details
  4. signs per shop frontage (fascia / channel letters / light box / blade), house numbers, door bells;
     shops without an own opening get a wall-only sign at their facade point (orphan_signs);
     sign colours/types from shops.json, corrected by sign_overrides.json (Street View notes)
  5. GLB with one palette texture (6 materials), atomic write; meshopt/webp web copy via gltf-transform
     (node names carry 'nocol_noshadow': the game skips them for collision and shadow casting)

  Then: python check.py (intersections / z-fighting / see-through / normals -> cache/check.json),
        python prep_preview.py <views> + blender -b -P preview_blender.py -- - <signage.glb> <views>,
        python begehung.py (refreshes the detail_signage block in BEGEHUNGSLISTE.md)
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

from common import (CACHE, DATA, HERE, RAW_OUT, WEB_OUT, glbio, jdump, jload, load_shops, log, to_game, to_local)
import displays  # noqa: E402
import geom  # noqa: E402
import signs  # noqa: E402
from geom import PAL, Builder  # noqa: E402

PY = sys.executable
PLACED = None
TILE = 64.0


def rng_for(*key):
    h = hashlib.md5("|".join(str(k) for k in key).encode()).hexdigest()
    return np.random.default_rng(int(h[:12], 16))


# ----------------------------------------------------------------------------- GLB writer (palette texture)
MATS = {
    "matte": dict(name="Signage Matte", rough=0.85, metal=0.0),
    "gloss": dict(name="Signage Gloss", rough=0.32, metal=0.0),
    "metal": dict(name="Signage Metal", rough=0.35, metal=0.8),
    "lit": dict(name="Signage Lit", rough=0.6, metal=0.0, emis=1.0),
    "glow": dict(name="Signage Glow", rough=0.85, metal=0.0, emis=0.4),
    "vitrine": dict(name="Signage Vitrine Glass"),
}


def material(key):
    m = MATS[key]
    if key == "vitrine":
        return {"name": m["name"], "alphaMode": "BLEND",
                "pbrMetallicRoughness": {"baseColorFactor": [0.86, 0.92, 0.95, 0.22], "metallicFactor": 0.0,
                                         "roughnessFactor": 0.05}}
    out = {"name": m["name"], "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": m["metal"],
                                                      "roughnessFactor": m["rough"]}}
    if m.get("emis"):
        out["emissiveTexture"] = {"index": 0}
        out["emissiveFactor"] = [m["emis"]] * 3
    return out


def write_glb(path, B):
    from PIL import Image
    buf = glbio._Buf()
    img = Image.fromarray(PAL.image())
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    png = bio.getvalue()
    meshes, nodes, mats = [], [], []
    matidx = {}
    tris = 0
    for key in ("matte", "gloss", "metal", "lit", "glow", "vitrine"):
        if key not in B.out:
            continue
        Vs, Ns, Cs, Fs = B.out[key]
        V, N, C = np.concatenate(Vs), np.concatenate(Ns), np.concatenate(Cs)
        F, off = [], 0
        for v, f in zip(Vs, Fs):
            F.append(f + off)
            off += len(v)
        F = np.concatenate(F)
        UV = np.array([PAL.uv(int(c)) for c in range(len(PAL.cols))] or [[0.5, 0.5]], np.float32)[C]
        # 64 m tiles (same grid as the game's mesh splitting): small quantisation volumes after meshopt
        cen = V[F].mean(1)
        tk = np.floor(cen[:, [0, 2]] / TILE).astype(np.int64)
        for tkey in sorted(set(map(tuple, tk))):
            Ft = F[(tk[:, 0] == tkey[0]) & (tk[:, 1] == tkey[1])]
            used, inv0 = np.unique(Ft.reshape(-1), return_inverse=True)
            Vt, Nt, Ut, Ct = V[used], N[used], UV[used], C[used]
            Ft = inv0.reshape(-1, 3)
            # weld identical vertices (same pos/normal/colour)
            key_arr = np.c_[np.round(Vt / 1e-5), np.round(Nt * 1e4), Ct[:, None]].astype(np.int64)
            _, first, inv = np.unique(key_arr, axis=0, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            Vt, Nt, Ut = Vt[first], Nt[first], Ut[first]
            Ft = inv[Ft]
            bad = (Ft[:, 0] == Ft[:, 1]) | (Ft[:, 1] == Ft[:, 2]) | (Ft[:, 0] == Ft[:, 2])
            Ft = Ft[~bad]
            if not len(Ft):
                continue
            # winding must agree with the stored (intended) normal: the game renders FrontSide only
            V32 = Vt.astype(np.float32).astype(np.float64)
            fn = np.cross(V32[Ft[:, 1]] - V32[Ft[:, 0]], V32[Ft[:, 2]] - V32[Ft[:, 0]])
            flip = np.einsum("ij,ij->i", fn, Nt[Ft].sum(1)) < 0
            if flip.any():
                Ft[flip] = Ft[flip][:, [0, 2, 1]]
            if key not in matidx:
                matidx[key] = len(mats)
                mats.append(material(key))
            attrs = {"POSITION": buf.add(Vt.astype(np.float32), 34962, "VEC3", 5126, True),
                     "NORMAL": buf.add((Nt / np.maximum(np.linalg.norm(Nt, axis=1, keepdims=True), 1e-12)).astype(np.float32),
                                       34962, "VEC3"),
                     "TEXCOORD_0": buf.add(Ut.astype(np.float32), 34962, "VEC2")}
            ia = buf.add(Ft.astype(np.uint32).reshape(-1), 34963, "SCALAR", 5125)
            # 'nocol' / 'noshadow' (web/js/world.js): no collision mesh (signs sit on walls, interiors behind
            # glass) and no shadow casting (closed display boxes, flat signs) -> no extra BVH / shadow-map cost
            nm = "signage_%s_%d_%d_nocol_noshadow" % (key, tkey[0], tkey[1])
            meshes.append({"name": nm, "primitives": [{"attributes": attrs, "indices": ia, "material": matidx[key],
                                                      "mode": 4}]})
            nodes.append({"mesh": len(meshes) - 1, "name": nm})
            tris += len(Ft)
    # image
    pad = (-buf.len) % 4
    if pad:
        buf.parts.append(b"\0" * pad)
        buf.len += pad
    buf.views.append({"buffer": 0, "byteOffset": buf.len, "byteLength": len(png)})
    buf.parts.append(png)
    buf.len += len(png)
    j = {"asset": {"version": "2.0", "generator": "detail_signage"}, "scene": 0,
         "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": mats,
         "images": [{"bufferView": len(buf.views) - 1, "mimeType": "image/png", "name": "signage_palette"}],
         "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
         "textures": [{"sampler": 0, "source": 0}],
         "accessors": buf.accs, "bufferViews": buf.views, "buffers": [{"byteLength": 0}]}
    bb = b"".join(buf.parts)
    bb += b"\0" * ((-len(bb)) % 4)
    j["buffers"][0]["byteLength"] = len(bb)
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    import struct
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bb)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(bb), 0x004E4942))
        f.write(bb)
    os.replace(tmp, path)
    return tris


def web_copy(src, dst):
    """meshopt with 16-bit positions (per 64 m tile mesh -> ~1 mm grid); named tile nodes are not joined.
    The 256x256 palette stays a lossless PNG (a few kB): lossy WebP shifted the nearest-filtered colour cells by up
    to 12/255.  Temp files live in the scratch folder, the final rename into web/ is atomic."""
    scr = os.environ.get("SIGNAGE_SCRATCH", r"C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/"
                         r"b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/signage")
    os.makedirs(scr, exist_ok=True)
    t1 = os.path.join(scr, "signage_web_1.glb")
    t2 = os.path.join(scr, "signage_web_2.glb")
    cmds = ['npx -y @gltf-transform/cli@4 optimize "%s" "%s" --compress false --texture-compress false --simplify false '
            '--instance false --join-named false --palette false --flatten false' % (src, t1),
            'npx -y @gltf-transform/cli@4 meshopt "%s" "%s" --quantize-position 16 --quantize-normal 10 '
            '--quantize-texcoord 14' % (t1, t2)]
    for cmd in cmds:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log("gltf-transform failed:", cmd, r.stdout[-1500:], r.stderr[-1500:])
            return False
    try:
        os.replace(t2, dst)
    except OSError:
        import shutil
        tmp = dst + ".part"
        shutil.copyfile(t2, tmp)
        os.replace(tmp, dst)
    return True


# ----------------------------------------------------------------------------- per-opening helpers
def door_details(B, o, shop, rng):
    """opening-hours sticker, card stickers and a push/pull label inside the door glass (within the box).
    Layers 3 mm apart (sticker sheet zf-7 mm, its print zf-4 mm)."""
    box = o["box"]
    if box["depth"] < 0.05:
        return
    cx = displays.Ctx(B, o["O"], o["R"], box, rng, shop=shop, opening=o)
    z = cx.zf - 0.007
    # door leaf: widest pane of the door
    panes = o["glass"]["panes"]
    pane = max(panes, key=lambda p: (p[1] - p[0]) * (p[3] - p[2]))
    xa, xb = max(pane[0], cx.a) + 0.04, min(pane[1], cx.b) - 0.04
    if xb - xa < 0.3:
        return
    fy0 = o["frame"][2]
    side = rng.integers(2)
    x0 = xa + 0.02 if side == 0 else xb - 0.27
    y0 = fy0 + 1.32
    if y0 + 0.3 > cx.d - 0.02 or y0 < cx.c + 0.01:
        return
    B.rect_z("matte", "#ffffff", x0, y0, x0 + 0.25, y0 + 0.3, z)
    B.rect_z("matte", "#1a1a1a", x0 + 0.02, y0 + 0.245, x0 + 0.23, y0 + 0.275, z + displays.LAYER)
    for k in range(6):
        yy = y0 + 0.2 - k * 0.03
        B.rect_z("matte", "#555555", x0 + 0.025, yy, x0 + 0.13, yy + 0.008, z + displays.LAYER)
        B.rect_z("matte", "#555555", x0 + 0.15, yy, x0 + 0.225, yy + 0.008, z + displays.LAYER)
    # card / payment stickers
    for k in range(3):
        xx = x0 + k * 0.085
        B.rect_z("gloss", ["#1a1f71", "#eb001b", "#00a3e0", "#ffcc00", "#2e7d32"][int(rng.integers(5))], xx, y0 - 0.1,
                 xx + 0.075, y0 - 0.05, z)
    # push / pull label
    if cx.d > fy0 + 1.1:
        xl = (xa + xb) / 2 - 0.09
        B.rect_z("matte", "#f4f4f4", xl, fy0 + 1.05, xl + 0.18, fy0 + 1.1, z)
        B.rect_z("matte", "#333333", xl + 0.02, fy0 + 1.07, xl + 0.16, fy0 + 1.08, z + displays.LAYER)
    cx.commit()


def opening_frame_rect_in(o, Oref, Rref):
    O, R = np.array(o["O"]), np.array(o["R"])
    fx0, fx1, fy0, fy1 = o["frame"]
    C = to_game(np.array([[fx0, fy0, 0], [fx1, fy0, 0], [fx1, fy1, 0], [fx0, fy1, 0]], float), O, R)
    L = to_local(C, Oref, Rref)
    return L[:, 0].min(), L[:, 0].max(), L[:, 1].min(), L[:, 1].max()


def facade_groups(ops):
    """split a shop's openings into facade planes."""
    groups = []
    for o in sorted(ops, key=lambda o: -(o["glass"]["rect"][1] - o["glass"]["rect"][0])):
        R = np.array(o["R"])
        n = R[:, 2]
        O = np.array(o["O"])
        for g in groups:
            Rg, Og = g["R"], g["O"]
            if n @ Rg[:, 2] > 0.97 and abs((O - Og) @ Rg[:, 2]) < 0.6:
                g["ops"].append(o)
                break
        else:
            groups.append(dict(R=R, O=O, ops=[o]))
    return groups


def main():
    t0 = time.time()
    if "--web-only" in sys.argv:
        ok = web_copy(RAW_OUT, WEB_OUT)
        log("web copy", ok, os.path.getsize(WEB_OUT) if ok else "")
        return
    if "--reextract" in sys.argv or not os.path.isfile(os.path.join(CACHE, "openings.json")):
        subprocess.run([PY, os.path.join(HERE, "openings.py")], check=True)
    if "--reextract" in sys.argv or "--reassign" in sys.argv or not os.path.isfile(os.path.join(CACHE, "assign.json")):
        subprocess.run([PY, os.path.join(HERE, "assign.py")], check=True)
    OPS = jload(os.path.join(CACHE, "openings.json"))["openings"]
    A0 = jload(os.path.join(CACHE, "assign.json"))["opening_shop"]
    SH = {s["id"]: s for s in load_shops()}
    # shop per opening; sub-openings (corner segments '#s<k>') inherit the shop of their glass unit
    A = {o["id"]: (A0.get(o["id"]) or A0.get(o.get("base") or "")) for o in OPS}
    OPS, A, SHARED = split_shared_fronts(OPS, A, SH)
    from openings import load_scene
    allP = np.array([o["O"] for o in OPS])
    _, obst = load_scene(allP.min(0) - 8, allP.max(0) + 8, allP[:, 1].max() + 10, t0)
    mt = signs.Mounter(obst)
    global PLACED
    PLACED = Placed()
    B = Builder()
    stats = dict(kinds={}, signs={}, blades=0, numbers=0, bells=0, door_details=0, sign_fail=[], partitions=0,
                 shared_fronts={g["front"]: [SH[s]["name"] for s in g["shops"]] for g in SHARED["groups"]})
    per_opening = []
    # the shop's name on a panel on the back wall of its widest tall display window - only where the facade sign
    # does not already say it right above (no fascia / letters type), or for about every third shop
    logo_op = {}
    for o in OPS:
        sid = A.get(o["id"])
        bx = o["box"]
        if not sid or o["kind"] != "window" or bx["depth"] < 0.25 or bx.get("clip"):
            continue
        typ = ((SH[sid].get("sign") or {}).get("type") or "none").split("+")[0]
        facade_sign = typ in ("fascia_panel", "channel_letters", "light_box", "logo_only")
        if facade_sign and rng_for(sid, "backlogo").random() > 0.34:
            continue
        a_, b_, c_, d_ = bx["rect"]
        if d_ - c_ < 2.35 or b_ - a_ < 0.9:
            continue
        if sid not in logo_op or (b_ - a_) > logo_op[sid][1]:
            logo_op[sid] = (o["id"], b_ - a_)
    logo_ids = {v[0] for v in logo_op.values()}
    # ---------------- interiors
    for o in OPS:
        shop = SH.get(A.get(o["id"])) if A.get(o["id"]) else None
        rng = rng_for(o["id"])
        k = displays.build_interior(B, o, shop, rng, logo=o["id"] in logo_ids)
        stats["kinds"][str(k)] = stats["kinds"].get(str(k), 0) + 1
        if o["kind"] == "shopdoor" or (o["kind"] == "entrance" and o["box"]["depth"] > 0.05 and False):
            door_details(B, o, shop, rng_for(o["id"], "door"))
            stats["door_details"] += 1
        per_opening.append(dict(id=o["id"], shop=shop["name"] if shop else None, display=k, depth=round(o["box"]["depth"], 3)))
    # partition caps between the per-shop boxes of a shared front (closes the 4 cm gap behind the mullion)
    for cap in SHARED["caps"]:
        O_, R_ = np.array(cap["O"]), np.array(cap["R"])
        x0_, x1_, y0_, y1_, z_ = cap["rect"]
        B.rect_z("matte", "#3a3d40", x0_, y0_, x1_, y1_, z_)
        B.commit(O_, R_)
        stats["partitions"] += 1
    log("interiors done: %d tris, dropped items %d (%.1fs)" % (B.tri_count(), B.dropped, time.time() - t0))
    # ---------------- signs per shop
    by_shop = {}
    for o in OPS:
        sid = A.get(o["id"])
        if sid:
            by_shop.setdefault(sid, []).append(o)
    # shared fronts: one sign band for all shops of the front (same rect height, baseline and cap height)
    for grp in SHARED["groups"]:
        O_, R_ = np.array(grp["O"]), np.array(grp["R"])
        a_, b_, top_ = grp["x0"], grp["x1"], grp["top"]
        xs, ys, cz = mt.cells(O_, R_, a_ + 0.02, b_ - 0.02, top_ + 0.03, top_ + 1.75, no_glass=False)
        cz = PLACED.mask_cells(O_, R_, xs, ys, cz)
        rect = mt.flat_rect(xs, ys, cz, xc=(a_ + b_) / 2, min_h=0.24, min_w=min(4.0, (b_ - a_) * 0.6))
        if rect is None:
            continue
        caps = []
        for sid_, (sx0, sx1) in zip(grp["shops"], grp["segs"]):
            s_ = SH[sid_]
            font = (s_.get("sign") or {}).get("font") or "sans"
            w_ = min(rect["x1"], sx1 - 0.15) - max(rect["x0"], sx0 + 0.15) - 0.4
            caps.append(min(0.3, w_ / max(geom.text_width(signs.sign_text(s_), font, 1.0), 1e-6)))
        grp["rect"] = rect
        grp["cap"] = max(0.12, min(caps))
        grp["wall_col"] = mt.surface_colour(xs, ys, rect)
    sign_log = []
    for sid, ops in sorted(by_shop.items()):
        shop = SH[sid]
        sg = shop.get("sign") or {}
        typ = sg.get("type") or "none"
        if shop.get("status") in ("closed", "vacant", "likely_closed") or typ == "none":
            continue
        rng = rng_for(sid, "sign")
        groups = facade_groups(ops)
        groups.sort(key=lambda g: -sum(op["glass"]["rect"][1] - op["glass"]["rect"][0] for op in g["ops"]))
        placed = 0
        placed_kinds = []
        for gi, g in enumerate(groups[:2]):
            Oref, Rref = g["O"], g["R"]
            rects = [opening_frame_rect_in(op, Oref, Rref) for op in g["ops"]]
            X0 = min(r[0] for r in rects)
            X1 = max(r[1] for r in rects)
            top = max(r[3] for r in rects)
            if X1 - X0 < 1.2 and gi > 0:
                continue
            base_typ = typ.split("+")[0]
            grp = SHARED["by_shop"].get(sid) if gi == 0 else None
            cap_fix = None
            if grp is not None and grp.get("rect"):
                # shared front: the common sign band, cut to this shop's segment
                sx0, sx1 = grp["segs"][grp["shops"].index(sid)]
                R_ = grp["rect"]
                rect = dict(R_, x0=max(R_["x0"], sx0 + 0.12), x1=min(R_["x1"], sx1 - 0.12))
                if rect["x1"] - rect["x0"] < 0.8:
                    rect = None
                cap_fix = grp["cap"]
                xs = ys = None
            else:
                # above the shop's own openings a sign may also sit on upper glazing (curtain-wall buildings: panel on
                # the mullion grid); wall-only signs, blades, numbers and bells never go on glass (Mounter default)
                xs, ys, cz = mt.cells(Oref, Rref, X0 + 0.02, X1 - 0.02, top + 0.03, top + 1.75, no_glass=False)
                cz = PLACED.mask_cells(Oref, Rref, xs, ys, cz)
                rect = mt.flat_rect(xs, ys, cz, xc=(X0 + X1) / 2, min_h=0.24, min_w=min(1.2, X1 - X0))
            ok = False
            if rect is not None and base_typ in ("fascia_panel", "channel_letters", "light_box", "logo_only", "projecting_blade",
                                                 "window_lettering"):
                kind = base_typ if base_typ in ("fascia_panel", "channel_letters", "light_box", "logo_only") else "fascia_panel"
                if base_typ == "projecting_blade" and (X1 - X0) < 3.0:
                    kind = None
                if kind:
                    if grp is not None:
                        wall_col = grp.get("wall_col")
                        if kind in ("fascia_panel", "logo_only", "light_box"):
                            # compact panel: text width + margin, centred in the shop's segment
                            font = sg.get("font") or "sans"
                            want = geom.text_width(signs.sign_text(shop), font, cap_fix) + 0.7
                            if rect["x1"] - rect["x0"] > want:
                                xm = (rect["x0"] + rect["x1"]) / 2
                                rect = dict(rect, x0=xm - want / 2, x1=xm + want / 2)
                    else:
                        wall_col = mt.surface_colour(xs, ys, rect)
                        if len(g["ops"]) == 1:
                            # a single (merged) glass front: may be shared with neighbours
                            rect = narrow_to_frontage(rect, shop, Oref, Rref, kind)
                    ok = signs.fascia(B, mt, Oref, Rref, rect, shop, kind, rng, wall_col=wall_col, cap_fix=cap_fix)
                    if ok:
                        PLACED.add(Oref, Rref, [rect["x0"], rect["y0"], rect["z"]], [rect["x1"], rect["y1"], rect["z"] + 0.15])
                        stats["signs"][kind] = stats["signs"].get(kind, 0) + 1
                        placed += 1
                        placed_kinds.append(kind)
            if not ok and kind_needs_fallback(typ):
                # window lettering on the inside of the largest pane
                ok = window_lettering(B, g["ops"], shop, rng)
                if ok:
                    stats["signs"]["window_lettering"] = stats["signs"].get("window_lettering", 0) + 1
                    placed += 1
                    placed_kinds.append("window_lettering")
            # blade sign
            dk = (shop.get("display") or {}).get("kind")
            want_blade = "projecting_blade" in typ or shop.get("category") == "pharmacy" or                 (dk in ("restaurant", "cafe", "fast_food", "bakery", "optician") or shop.get("category") in ("hotel", "pub", "bar"))                 and rng.random() < 0.6
            if gi == 0 and want_blade:
                zw = float(np.nanmedian(cz)) if np.isfinite(cz).any() else 0.1
                cands = [(xb, yt) for yt in (top + 1.25, top + 0.95, top + 1.55) for xb in (X1 + 0.25, X0 - 0.25, X1 - 0.3, X0 + 0.3, X1 + 0.5, X0 - 0.5)]
                for xb, yt in cands:
                    blo = [xb - 0.06, yt - 0.66, zw - 0.3]
                    bhi = [xb + 0.06, yt + 0.02, zw + 0.9]
                    if PLACED.overlaps(Oref, Rref, blo, bhi):
                        continue
                    if signs.blade(B, mt, Oref, Rref, xb, yt, zw + 0.003, shop, rng,
                                   pharmacy=shop.get("category") == "pharmacy"):
                        stats["blades"] += 1
                        PLACED.add(Oref, Rref, blo, bhi)
                        placed += 1
                        placed_kinds.append("blade")
                        break
        sign_log.append(dict(shop=shop["name"], type=typ, placed=placed, openings=len(ops), kinds=placed_kinds))
        if not placed:
            stats["sign_fail"].append(shop["name"])
    log("signs done: %d tris (%.1fs)" % (B.tri_count(), time.time() - t0))
    # ---------------- shops of the list that found no shopfront opening: name sign on the wall only
    stats["wall_only"] = orphan_signs(B, mt, SH, set(by_shop), stats, sign_log)
    log("wall-only signs: %d (%.1fs)" % (len(stats["wall_only"]), time.time() - t0))
    # ---------------- house numbers + door bells
    numbers_done = set()
    doors = sorted([o for o in OPS if o["kind"] in ("entrance", "shopdoor")], key=lambda o: 0 if o["kind"] == "entrance" else 1)
    for o in doors:
        lod = o.get("lod2") or {}
        num = lod.get("number")
        key = (lod.get("street"), num)
        O, R = np.array(o["O"]), np.array(o["R"])
        fx0, fx1, fy0, fy1 = o["frame"]
        if num and key not in numbers_done:
            for (xa, xb) in ((fx1 + 0.06, fx1 + 0.62), (fx0 - 0.62, fx0 - 0.06)):
                xs, ys, cz = mt.cells(O, R, xa, xb, fy1 - 0.55, fy1 + 0.05, step=0.04)
                cz = PLACED.mask_cells(O, R, xs, ys, cz)
                rect = mt.flat_rect(xs, ys, cz, min_h=0.2, min_w=0.3, prefer_low=False)
                if rect and signs.number_plate(B, O, R, rect, str(num), style="enamel" if rng_for(key).random() < 0.7 else "steel"):
                    numbers_done.add(key)
                    PLACED.add(O, R, [rect["x0"], rect["y0"], rect["z"]], [rect["x1"], rect["y1"], rect["z"] + 0.03])
                    stats["numbers"] += 1
                    break
        if o["kind"] == "entrance":
            for (xa, xb) in ((fx1 + 0.05, fx1 + 0.4), (fx0 - 0.4, fx0 - 0.05)):
                xs, ys, cz = mt.cells(O, R, xa, xb, fy0 + 1.0, fy0 + 1.65, step=0.04)
                cz = PLACED.mask_cells(O, R, xs, ys, cz)
                rect = mt.flat_rect(xs, ys, cz, min_h=0.36, min_w=0.16, prefer_low=False)
                if rect and signs.doorbell(B, O, R, rect, n_bells=int(rng_for(o["id"], "b").integers(4, 9))):
                    PLACED.add(O, R, [rect["x0"], rect["y0"], rect["z"]], [rect["x1"], rect["y1"], rect["z"] + 0.03])
                    stats["bells"] += 1
                    break
    log("numbers %d bells %d" % (stats["numbers"], stats["bells"]))
    tris = write_glb(RAW_OUT, B)
    stats["triangles"] = tris
    stats["palette_colours"] = len(PAL.cols)
    stats["dropped_items"] = B.dropped
    stats["items"] = B.items
    log("wrote", RAW_OUT, tris, "tris", "palette", len(PAL.cols), "(%.1fs)" % (time.time() - t0))
    jdump(dict(stats=stats, signs=sign_log, openings=per_opening), os.path.join(CACHE, "build_stats.json"))
    if "--no-web" not in sys.argv:
        ok = web_copy(RAW_OUT, WEB_OUT)
        log("web copy", ok, os.path.getsize(WEB_OUT) if ok else "")
    log(json.dumps({k: v for k, v in stats.items() if k != "sign_fail"}))
    log("sign failures:", stats["sign_fail"])
    log("done %.1fs" % (time.time() - t0))


def split_shared_fronts(OPS, A, SH):
    """one detected glass front that spans several shops of the list (Osterstraße 29-31, d0:580: Stichnoth,
    Miss Lashes, Lia's and Wömpner behind one 19 m pane row) -> one sub-opening per shop ('<id>#p<k>'):
    the box is cut at the mullion nearest to the midpoint between neighbouring shops' facade points, a 4 cm
    partition gap (closed by the boxes' outer faces + a front cap behind the mullion) separates the displays,
    and the shops share one sign band (common baseline and cap height, see main).
    Only open street-level shops with a sign and without an own opening join a front (plus its assigned shop)."""
    have_own = {v for v in A.values() if v}
    out, A2 = [], dict(A)
    groups, caps, by_shop = [], [], {}
    for o in OPS:
        sid = A.get(o["id"])
        bx = o["box"]
        if not sid or o["kind"] != "window" or bx["depth"] < 0.25 or bx.get("clip"):
            out.append(o)
            continue
        a, b, c, d = bx["rect"]
        if b - a < 5.0:
            out.append(o)
            continue
        O, R = np.array(o["O"]), np.array(o["R"])
        n = R[:, 2]
        cands = []
        for s in SH.values():
            if s["id"] != sid and s["id"] in have_own:
                continue
            if s.get("level", 0) != 0 or s.get("context") not in ("street", "passage"):
                continue
            if s.get("status") in ("closed", "vacant", "likely_closed"):
                continue
            if s["id"] != sid and (((s.get("sign") or {}).get("type") or "none") == "none" or not signs.sign_text(s)):
                continue
            ck = s.get("category_key") or ""
            if s["id"] != sid and ck and not ck.startswith(("shop=", "amenity=cafe", "amenity=restaurant", "amenity=fast_food",
                                                             "amenity=pharmacy", "amenity=bank", "amenity=ice_cream")):
                continue
            f = s.get("facade") or {}
            wp, nn = f.get("wall_point"), f.get("normal")
            if not wp:
                continue
            if nn and float(np.dot([nn[0], 0.0, nn[1]], n)) < 0.8:
                continue
            L = (np.array([wp[0], O[1], wp[1]]) - O) @ R
            if not (a - 0.3 <= L[0] <= b + 0.3) or abs(L[2]) > 7.0:
                continue
            cands.append((float(np.clip(L[0], a, b)), s["id"]))
        if len(cands) < 2 or sid not in [c_[1] for c_ in cands]:
            out.append(o)
            continue
        cands.sort()
        # mullion gaps of the pane row (local x), boundaries snap to the nearest one within 1.2 m
        panes = sorted(o["glass"]["panes"], key=lambda p: p[0])
        gaps = [(panes[i][1] + panes[i + 1][0]) / 2 for i in range(len(panes) - 1)]
        bounds = []
        for i in range(len(cands) - 1):
            m = (cands[i][0] + cands[i + 1][0]) / 2
            if gaps:
                gsel = min(gaps, key=lambda g_: abs(g_ - m))
                if abs(gsel - m) < 1.2:
                    m = gsel
            bounds.append(m)
        edges = [a] + bounds + [b]
        segs = [(edges[i], edges[i + 1]) for i in range(len(cands))]
        if min(s1 - s0 for s0, s1 in segs) < 1.2:
            out.append(o)
            continue
        log("shared front", o["id"], "->", [(SH[c_[1]]["name"], round(s0, 2), round(s1, 2)) for c_, (s0, s1) in zip(cands, segs)])
        grp = dict(front=o["id"], O=o["O"], R=o["R"], x0=a, x1=b, top=o["frame"][3], shops=[c_[1] for c_ in cands],
                   segs=[[float(s0), float(s1)] for s0, s1 in segs])
        groups.append(grp)
        for k, ((s0, s1), (_, ssid)) in enumerate(zip(segs, cands)):
            sub = json.loads(json.dumps(o))
            sub["id"] = "%s#p%d" % (o["id"], k)
            sub["base"] = o.get("base") or o["id"]
            sub["box"]["rect"] = [s0 + (0.02 if k > 0 else 0.0), s1 - (0.02 if k < len(segs) - 1 else 0.0), c, d]
            sub["frame"] = [s0 if k > 0 else o["frame"][0], s1 if k < len(segs) - 1 else o["frame"][1], o["frame"][2],
                            o["frame"][3]]
            out.append(sub)
            A2[sub["id"]] = ssid
            by_shop[ssid] = grp
            if k < len(segs) - 1:
                caps.append(dict(O=o["O"], R=o["R"], rect=[s1 - 0.02, s1 + 0.02, c, d, bx["z_front"]]))
        A2.pop(o["id"], None)
    return out, A2, dict(groups=groups, caps=caps, by_shop=by_shop)


class Placed:
    """boxes (local frame + extents) of signs already placed: later signs avoid them."""

    def __init__(self):
        self.items = []

    def add(self, O, R, lo, hi):
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        self.items.append(to_game(C, np.asarray(O), np.asarray(R)))

    def overlaps(self, O, R, lo, hi, pad=0.02):
        for G in self.items:
            L = to_local(G, np.asarray(O), np.asarray(R))
            if (L.min(0) < np.asarray(hi) + pad).all() and (L.max(0) > np.asarray(lo) - pad).all():
                return True
        return False

    def mask_cells(self, O, R, xs, ys, cz):
        for G in self.items:
            L = to_local(G, np.asarray(O), np.asarray(R))
            lo, hi = L.min(0), L.max(0)
            if hi[2] < -1.0 or lo[2] > 1.5:
                continue
            ci = (xs[1:] > lo[0] - 0.03) & (xs[:-1] < hi[0] + 0.03)
            ri = (ys[1:] > lo[1] - 0.03) & (ys[:-1] < hi[1] + 0.03)
            cz[np.ix_(ri, ci)] = np.nan
        return cz


def narrow_to_frontage(rect, shop, O, R, kind):
    """one detected glass front can span several shops (e.g. Osterstraße 29-31): a sign wider than 1.6x the shop's
    frontage is cut to max(frontage, text + margin), centred on the shop's facade point, so neighbours keep room."""
    fw = float(shop.get("frontage_width_m") or 0)
    W = rect["x1"] - rect["x0"]
    if fw <= 0 or W <= 1.6 * fw:
        return rect
    f = shop.get("facade") or {}
    ent = f.get("entrance_node")
    p = (ent or {}).get("p") if ent and ent.get("dist", 99) < 4 else f.get("wall_point")
    if not p:
        return rect
    xl = float((np.array([p[0], 0.0, p[1]]) - np.asarray(O)) @ np.asarray(R)[:, 0])
    font = (shop.get("sign") or {}).get("font") or "sans"
    cap = min(0.36, (min(rect["y1"] - rect["y0"], 0.75) - 0.03) * 0.62)
    want = max(fw, geom.text_width(signs.sign_text(shop), font, cap) + (0.7 if kind != "channel_letters" else 0.3))
    if want >= W:
        return rect
    xc = float(np.clip(xl, rect["x0"] + want / 2, rect["x1"] - want / 2))
    return dict(rect, x0=xc - want / 2, x1=xc + want / 2)


def street_level(points_xz, files=("street", "ground", "fix")):
    """street surface y below each game-frame [x, z] point (highest detail-model ground triangle; nan if none)."""
    from common import detail_path, load_tris
    P = np.asarray(points_xz, float)
    lo, hi = P.min(0) - 3, P.max(0) + 3
    Ts = []
    for f in files:
        p = detail_path(f)
        if os.path.isfile(p):
            T, _, _ = load_tris(p, keep=lambda C: (C[:, 0] > lo[0]) & (C[:, 0] < hi[0]) & (C[:, 2] > lo[1]) &
                                (C[:, 2] < hi[1]) & (C[:, 1] < 8))
            if len(T):
                Ts.append(T)
    from common import heightfield_y
    out = np.full(len(P), np.nan)
    if not Ts:
        return np.array([heightfield_y(x, z) for x, z in P])
    T = np.concatenate(Ts)
    A, Bv, Cv = T[:, 0][:, [0, 2]], T[:, 1][:, [0, 2]], T[:, 2][:, [0, 2]]
    for k, (x, z) in enumerate(P):
        v0, v1 = Cv - A, Bv - A
        v2 = np.array([x, z])[None] - A
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        m = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
        if m.any():
            y = T[m, 0, 1] + u[m] * (T[m, 2, 1] - T[m, 0, 1]) + v[m] * (T[m, 1, 1] - T[m, 0, 1])
            out[k] = float(y.max())
        else:
            # outside the detail model's street surface: the game's walkable heightfield
            out[k] = heightfield_y(x, z)
    return out


def orphan_signs(B, mt, SH, placed_ids, stats, sign_log):
    """open shops of data/shops.json next to the detail model that got no shopfront opening (OSM point between
    two neighbours' fronts, shop behind ordinary ground-floor windows): fascia / letters on the flat wall above
    the ground floor at the shop's facade point, if the model has a wall there (not in replaced landmarks)."""
    from common import frame_from_normal
    from openings import in_landmark, load_landmark_polys
    mask = np.load(os.path.join(DATA, "detail_mask.npy"))
    polys = load_landmark_polys()
    cand = []
    for s in SH.values():
        if s["id"] in placed_ids or s.get("level", 0) != 0 or s.get("context") not in ("street", "passage"):
            continue
        if s.get("status") in ("closed", "vacant", "likely_closed"):
            continue
        sg = s.get("sign") or {}
        if (sg.get("type") or "none") == "none" or not signs.sign_text(s):
            continue
        f = s.get("facade") or {}
        wp, nn = f.get("wall_point"), f.get("normal")
        if not wp or not nn:
            continue
        j, i = int(np.floor(wp[0] + 400)), int(np.floor(wp[1] + 400))
        if not (0 <= i < 800 and 0 <= j < 800) or not mask[i, j]:
            continue
        if in_landmark(polys, np.array([wp[0], 1.0, wp[1]]), buf=1.5):
            continue
        n = np.array([nn[0], 0.0, nn[1]], float)
        n /= np.linalg.norm(n)
        cand.append((s, np.array(wp, float), n))
    if not cand:
        return []
    ys = street_level([wp + np.array([n[0], n[2]]) * 1.5 for _, wp, n in cand])
    done = []
    for (s, wp, n), yg in zip(cand, ys):
        if not np.isfinite(yg):
            continue
        R = frame_from_normal(n)
        O = np.array([wp[0], yg, wp[1]])
        W = float(np.clip(float(s.get("frontage_width_m") or 5.0), 2.5, 9.0))
        sg = s.get("sign") or {}
        typ = sg.get("type") or "fascia_panel"
        base_typ = typ.split("+")[0]
        kind = base_typ if base_typ in ("fascia_panel", "channel_letters", "light_box", "logo_only") else "fascia_panel"
        rng = rng_for(s["id"], "wall")
        ok = False
        for (ya, yb) in ((2.75, 4.3), (2.5, 3.7), (3.4, 4.9)):
            xs, ys_, cz = mt.cells(O, R, -W / 2, W / 2, ya, yb)
            # the wall must be close to the shop's facade point (no sign on a building 1 m further back / in front)
            cz[np.isfinite(cz) & ((cz < -0.6) | (cz > 0.6))] = np.nan
            cz = PLACED.mask_cells(O, R, xs, ys_, cz)
            rect = mt.flat_rect(xs, ys_, cz, xc=0.0, min_h=0.32, min_w=min(1.6, W * 0.6))
            if rect is None:
                continue
            wall_col = mt.surface_colour(xs, ys_, rect)
            # no own shopfront below: keep the sign compact (text width + margin), not a band over the whole wall
            font = sg.get("font") or "sans"
            cap = min(0.36, (min(rect["y1"] - rect["y0"], 0.75) - 0.03) * 0.62)
            tw = geom.text_width(signs.sign_text(s), font, cap)
            want = tw + (0.7 if kind in ("fascia_panel", "logo_only", "light_box") else 0.3)
            if rect["x1"] - rect["x0"] > want:
                xc = float(np.clip(0.0, rect["x0"] + want / 2, rect["x1"] - want / 2))
                rect = dict(rect, x0=xc - want / 2, x1=xc + want / 2)
            if signs.fascia(B, mt, O, R, rect, s, kind, rng, wall_col=wall_col):
                PLACED.add(O, R, [rect["x0"], rect["y0"], rect["z"]], [rect["x1"], rect["y1"], rect["z"] + 0.15])
                stats["signs"][kind] = stats["signs"].get(kind, 0) + 1
                cg = O + R @ np.array([(rect["x0"] + rect["x1"]) / 2, (rect["y0"] + rect["y1"]) / 2, rect["z"]])
                stats.setdefault("wall_only_pos", {})[s["name"]] = [round(float(v), 2) for v in cg] + \
                    [round(float(v), 3) for v in R[:, 2]]
                ok = True
                break
        sign_log.append(dict(shop=s["name"], type=typ, placed=int(ok), openings=0, wall_only=True,
                             kinds=[kind] if ok else []))
        if ok:
            done.append(s["name"])
        else:
            stats["sign_fail"].append(s["name"] + " (wall)")
    return done


def kind_needs_fallback(typ):
    return typ not in ("none",)


def window_lettering(B, ops, shop, rng):
    """white vinyl name on the inside of the widest pane (inside the box, 1.5 mm behind its front)."""
    best = max(ops, key=lambda o: (o["box"]["rect"][1] - o["box"]["rect"][0]) if o["box"]["depth"] > 0.05 else -1)
    box = best["box"]
    if box["depth"] < 0.05:
        return False
    a, b, c, d = box["rect"]
    text = signs.sign_text(shop)
    font = (shop.get("sign") or {}).get("font") or "sans"
    res = signs.fit_text(text, font, (b - a) - 0.2, 0.3, cap_max=0.14, cap_min=0.04)
    if not res:
        return False
    lines, cap = res
    y = min(d - 0.25, c + 1.75)
    col = "#ffffff" if rng.random() < 0.7 else ((shop.get("sign") or {}).get("fg") or "#ffffff")
    for ln in lines:
        signs.place_text(B, ln, font, cap, ((a + b) / 2, y, box["z_front"] - 0.0035), (0, 0, 1), depth=0.0005, mat="gloss",
                         col=col)
        y -= cap * 1.4
    lo = np.array([a, c, box["z_back"]])
    hi = np.array([b, d, box["z_front"]])
    return B.commit(best["O"], best["R"], bounds=(lo, hi), planes=box.get("clip"))


if __name__ == "__main__":
    main()

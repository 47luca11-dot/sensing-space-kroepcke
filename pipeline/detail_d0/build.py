"""D0 window upgrade - one command rebuilds everything from data/detail_raw_orig/.

  python build.py [--no-compress] [--check] [--previews]

  --check     run check.py (see-through / gap / z-fighting / boundary-edge validation)
  --previews  render blender/previews/d0_{before,after}_*.png with headless Blender (cams.json)

1. load bldg_walls / bldg_openings from data/detail_raw_orig (backup of the Rhino export)
2. fix inverted wall triangles (component parity + roof-ray vote + D0 votes)
3. extract the 'Facade Glass D0' quads, match them to coplanar wall triangles
4. cut real openings (constrained triangulation), add plaster reveals into the wall nodes
5. build detailed window assemblies, merged per material -> data/detail_raw/d0_windows.glb
6. write data/detail_raw/bldg_walls.glb, bldg_openings.glb (D0 removed)
7. meshopt-compressed web copies in web/assets/city/detail/
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glbio      # noqa: E402
import walls      # noqa: E402
import extract    # noqa: E402
import cut        # noqa: E402
import windows    # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "data", "detail_raw")
ORIG = os.path.join(ROOT, "data", "detail_raw_orig")
WEB = os.path.join(ROOT, "web", "assets", "city", "detail")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------------------- materials
def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex2lin(hx):
    hx = hx.lstrip("#")
    return [float(x) for x in srgb2lin([int(hx[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])]


def new_mat(name, hx, rough=0.8, metal=0.0, extra=None):
    m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": hex2lin(hx) + [1.0], "metallicFactor": metal,
                                                 "roughnessFactor": rough},
         "extensions": {"KHR_materials_specular": {"specularFactor": 0.5}, "KHR_materials_ior": {"ior": 1}}}
    if extra:
        m["extensions"].update(extra)
    return m


def shade(hx, f):
    hx = hx.lstrip("#")
    c = [min(255, max(0, int(round(int(hx[i:i + 2], 16) * f)))) for i in (0, 2, 4)]
    return "#%02X%02X%02X" % tuple(c)


# ----------------------------------------------------------------------------- helpers
def hrand(*keys):
    h = hashlib.md5(("|".join(str(k) for k in keys)).encode()).hexdigest()
    return np.random.default_rng(int(h[:12], 16))


def mesh_of(glbs):
    V, F, o = [], [], 0
    for g in glbs:
        for n in g["nodes"]:
            for p in n["prims"]:
                V.append(p["pos"]); F.append(p["idx"] + o); o += len(p["pos"])
    return trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=False)


def weld(T, N, U):
    """triangle soup -> indexed (pos, nrm, uv) welding identical vertices."""
    P = T.reshape(-1, 3)
    Nn = N.reshape(-1, 3)
    Uu = U.reshape(-1, 2) if U is not None else np.zeros((len(P), 2))
    key = np.concatenate([np.round(P * 1e5), np.round(Nn * 1e4), np.round(Uu * 1e5)], 1).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[first], Nn[first], (Uu[first] if U is not None else None), inv.reshape(-1, 3)


# ----------------------------------------------------------------------------- styles
JOIN_PALETTE = {
    "altbau": [("#F2F2EE", 0.78), ("#E8DDC5", 0.12), ("#4B3528", 0.10)],
    "nachkrieg": [("#F2F2EE", 0.70), ("#C9C6BE", 0.15), ("#3A3D40", 0.15)],
    "modern": [("#2F3337", 0.55), ("#8C969C", 0.25), ("#F2F2EE", 0.20)],
    "narrow": [("#2F3337", 0.50), ("#8C969C", 0.20), ("#F2F2EE", 0.30)],
}


def pick(rng, pal):
    r = rng.random()
    acc = 0
    for v, p in pal:
        acc += p
        if r <= acc:
            return v
    return pal[-1][0]


def cluster_windows(wins, eps=7.0):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    C = np.array([w.center for w in wins])
    node = np.array([w.wall_node for w in wins])
    pairs = np.array(list(cKDTree(C).query_pairs(eps)))
    if len(pairs):
        pairs = pairs[node[pairs[:, 0]] == node[pairs[:, 1]]]
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
                   shape=(len(wins), len(wins)))
    nc, lab = connected_components(A, directed=False)
    return nc, lab


def joinery_near(op, wins, lab, nc, radius=3.0):
    """area-weighted colour of existing (non-D0) window joinery near each cluster
    (upper floors only, so ground-floor doors/shopfronts do not dictate the window colour)."""
    from scipy.spatial import cKDTree
    cents, cols, areas = [], [], []
    for n in op["nodes"]:
        nm = n["material"]["name"]
        if not nm.startswith("Facade Joinery #"):
            continue
        p = n["prims"][0]
        T = p["pos"][p["idx"]]
        a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
        cents.append(T.mean(1)); areas.append(a); cols.append(np.full(len(T), nm[len("Facade Joinery "):]))
    cents = np.concatenate(cents); areas = np.concatenate(areas); cols = np.concatenate(cols)
    tree = cKDTree(cents)
    out = {}
    for k in range(nc):
        m = np.where(lab == k)[0]
        acc = {}
        for i in m:
            w = wins[i]
            if w.rel < 2.5 or w.width > 3.0:
                continue
            ids = tree.query_ball_point(w.center, radius + max(w.width, w.height) / 2)
            for j in ids:
                if cents[j][1] < w.ground + 2.5:
                    continue
                acc[cols[j]] = acc.get(cols[j], 0) + areas[j]
        if acc:
            best = max(acc.items(), key=lambda kv: kv[1])
            if best[1] > 0.4:
                out[k] = (best[0], best[1])
    return out


def cluster_style(ws_nodes, wins, idxs):
    W = np.array([wins[i].width for i in idxs])
    H = np.array([wins[i].height for i in idxs])
    rel = np.array([wins[i].rel for i in idxs])
    norm = (W < 3.0) & (rel > 1.0)
    if norm.sum() == 0:
        norm = W < 3.0
    if norm.sum() == 0:
        return "modern"
    r = np.median(H[norm] / W[norm])
    mw = np.median(W[norm])
    wall = ws_nodes[wins[idxs[0]].wall_node]["material"]["name"]
    hx = wall.split("#")[-1][:6] if "#" in wall else "D0D0D0"
    rgb = np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)])
    reddish = rgb[0] - rgb[2] > 40 and rgb.mean() < 170
    if mw <= 1.0 and r >= 1.45:
        return "narrow"
    if r >= 1.2:
        return "modern" if reddish else "altbau"
    if r >= 0.72:
        return "modern" if reddish else "nachkrieg"
    return "modern"


# ----------------------------------------------------------------------------- spec per window
def window_spec(w, cl, M_base, rng):
    s = {"w": w.width, "h": w.height}
    st = cl["style"]
    rel = w.rel
    if rel < 0.35 and w.width < 2.0 and w.height >= 1.9:
        kind = "door"
    elif rel < 1.0 and (w.width >= 2.0 or w.height >= 2.4):
        kind = "shop"
    elif w.width >= 3.0:
        kind = "ribbon"
    else:
        kind = st
    s["kind"] = kind
    M = dict(M_base)
    M["joinery"] = cl["joinery"]
    if kind in ("shop", "door", "ribbon"):
        M["joinery"] = cl["joinery_metal"]
    M["trim"] = cl["trim_mat"]
    M["shutter"] = cl["shutter_mat"]
    M["shutter_box"] = cl["shutter_box_mat"]
    s["M"] = M
    w_, h_ = w.width, w.height
    d_f = {"altbau": 0.24, "nachkrieg": 0.20, "modern": 0.18, "narrow": 0.18, "ribbon": 0.18,
           "shop": 0.18, "door": 0.18}[kind]
    s["d_f"] = d_f
    s["fw"] = 0.075 if kind == "altbau" else (0.06 if kind in ("shop", "door") else 0.068)
    s["textile"] = "none"
    s["tilt"] = 0.0
    lay = {}
    if kind == "altbau":
        s["sill"] = "stone"; s["sill_ears"] = 0.04; s["sill_thick"] = 0.055
        if cl["trim"]:
            s["trim"] = True; s["trim_w"] = 0.14; s["trim_t"] = 0.022
        inner_w = w_ - 2 * s["fw"]
        if h_ >= 1.6:
            top_h = min(0.52, max(0.34, 0.24 * h_))
            lay["transom"] = h_ - s["fw"] - top_h
            lay["upper"] = "sash"
            lay["upper_vbar"] = inner_w >= 1.0 and cl["bars"] > 0
        if inner_w >= 0.8:
            lay["cols"] = [("sash", 1), ("sash", 1)]
            lay["stulp"] = True
        else:
            lay["cols"] = [("sash", 1)]
        lay["bars"] = cl["bars"] if h_ >= 1.3 else 0
        s["textile"] = rng.choice(["side", "side", "net", "none", "none", "blind"])
    elif kind == "nachkrieg":
        s["sill"] = "stone"; s["sill_ears"] = 0.025; s["sill_thick"] = 0.045
        if cl["trim"]:
            s["trim"] = True; s["trim_w"] = 0.11; s["trim_t"] = 0.025
        s["shutter"] = cl["shutters"]
        if s["shutter"]:
            r = rng.random()
            s["shutter_frac"] = 0.0 if r < 0.45 else float(rng.choice([0.2, 0.3, 0.35, 0.5, 0.6, 0.8, 1.0]))
        inner_w = w_ - 2 * s["fw"]
        if inner_w >= 2.0:
            lay["cols"] = [("sash", 1), ("fixed", 1.2), ("sash", 1)]
        elif inner_w >= 0.95:
            lay["cols"] = [("sash", 1), ("sash", 1)]
        else:
            lay["cols"] = [("sash", 1)]
        if h_ >= 1.9:
            lay["transom"] = h_ * 0.74 - (0.17 if s["shutter"] else 0)
            lay["upper"] = "split"
        s["textile"] = rng.choice(["side", "net", "net", "none", "blind", "side"])
    elif kind in ("modern", "narrow"):
        s["sill"] = "metal"; s["sill_ears"] = 0.02
        inner_w = w_ - 2 * s["fw"]
        if kind == "narrow" or inner_w < 0.95:
            lay["cols"] = [("sash", 1)]
            if h_ >= 1.9:
                lay["panel_h"] = None
                lay["transom"] = None
                s["railing"] = [0.92, 0.5]
        elif inner_w >= 2.0:
            lay["cols"] = [("sash", 1), ("fixed", 1.3), ("sash", 1)]
        else:
            lay["cols"] = [("sash", 1), ("sash", 1)] if rng.random() < 0.6 else [("fixed", 1), ("sash", 1)]
        lay["mull_w"] = 0.075
        s["textile"] = rng.choice(["blind", "blind", "none", "net", "side"])
    elif kind == "ribbon":
        s["sill"] = "metal"; s["sill_ears"] = 0.02
        n = max(2, int(round(w_ / 1.3)))
        lay["cols"] = [("sash" if k % 2 == 0 else "fixed", 1) for k in range(n)]
        lay["mull_w"] = 0.07
        if h_ >= 2.3:
            lay["panel_h"] = 0.78
        s["textile"] = "none"
    elif kind == "shop":
        s["sill"] = None
        s["fw"] = 0.06
        n = max(1, int(round(w_ / 1.9)))
        cols = [("fixed", 1)] * n
        if w.rel < 0.4 and n >= 2 and w_ >= 2.6 and rng.random() < 0.7:
            k = n // 2 if rng.random() < 0.6 else (0 if rng.random() < 0.5 else n - 1)
            cols[k] = ("door", 1)
        lay["cols"] = cols
        lay["mull_w"] = 0.06
        lay["handle"] = False
        if h_ >= 2.9:
            lay["transom"] = 2.35
            lay["upper"] = "split" if n > 1 else "fixed"
        lay["panel_h"] = 0.28 if w.rel > -0.2 else None
    elif kind == "door":
        s["sill"] = None
        s["fw"] = 0.06
        lay["cols"] = [("door", 1)]
        lay["handle"] = False
        lay["mull_w"] = 0.06
        if h_ >= 2.35:
            lay["transom"] = h_ - 0.06 - 0.42
            lay["upper"] = "fixed"
    s["layout"] = lay
    if kind in ("altbau", "nachkrieg", "modern") and rng.random() < 0.05:
        s["tilt"] = np.radians(rng.uniform(4, 7))
    return s


# ----------------------------------------------------------------------------- main
def ray_depth(occl, wins, start_depth, frac=(0.02, 0.5, 0.98)):
    """free depth behind each window plane (min over a grid of rays), measured from the wall plane."""
    O, D = [], []
    for w, sd in zip(wins, start_depth):
        for fx in frac:
            for fy in frac:
                O.append(w.wall_origin + w.r * w.width * fx + w.u * w.height * fy - w.n * sd)
                D.append(-w.n)
    O = np.array(O); D = np.array(D)
    locs, ri, _ = occl.intersects_location(O, D, multiple_hits=False)
    dist = np.full(len(O), 9.0)
    dist[ri] = np.linalg.norm(locs - O[ri], axis=1)
    k = len(frac) ** 2
    return dist.reshape(len(wins), k).min(1) + np.asarray(start_depth)


def footprint(w, depth):
    from shapely.geometry import Polygon as P
    a = w.wall_origin
    b = a + w.r * w.width
    Q = [a, b, b - w.n * depth, a - w.n * depth]
    return P([(q[0], q[2]) for q in Q])


def resolve_conflicts(wins, specs):
    """corner windows: reveal tunnels that intersect -> flush; interior boxes that intersect -> shallower."""
    from scipy.spatial import cKDTree
    C = np.array([w.center for w in wins])
    pairs = []
    maxw = max(w.width for w in wins)
    for a, b in cKDTree(C).query_pairs(3.5 + maxw):
        wa, wb = wins[a], wins[b]
        if np.linalg.norm(C[a] - C[b]) > 3.5 + (wa.width + wb.width) / 2:
            continue
        if wa.n @ wb.n > 0.98:
            continue
        if abs(wa.center[1] - wb.center[1]) > (wa.height + wb.height) / 2:
            continue
        pairs.append((a, b))
    nflush = 0
    for a, b in pairs:
        wa, wb = wins[a], wins[b]
        if wa.mode != "cut" or wb.mode != "cut":
            continue
        ra = specs[a]["d_f"] + windows.FD + 0.03
        rb = specs[b]["d_f"] + windows.FD + 0.03
        if footprint(wa, ra).intersection(footprint(wb, rb)).area > 1e-7:
            wa.mode = wb.mode = "flush"
            wa.flush_reason = wb.flush_reason = "corner"
            nflush += 2
    nshr = 0
    for it in range(30):
        changed = False
        for a, b in pairs:
            wa, wb = wins[a], wins[b]
            if wa.mode != "cut" or wb.mode != "cut":
                continue
            da = specs[a]["d_f"] + windows.FD + specs[a]["D_int"]
            db = specs[b]["d_f"] + windows.FD + specs[b]["D_int"]
            if footprint(wa, da + 0.01).intersection(footprint(wb, db + 0.01)).area > 1e-7:
                k = a if specs[a]["D_int"] >= specs[b]["D_int"] else b
                if specs[k]["D_int"] > 0.06:
                    if specs[k]["D_int"] * 0.8 < 0.25:
                        specs[k]["tilt"] = 0.0
                    specs[k]["D_int"] = max(0.05, specs[k]["D_int"] * 0.8)
                    changed = True
                    nshr += 1
        if not changed:
            break
    return nflush, nshr


def main():
    t0 = time.time()
    compress = "--no-compress" not in sys.argv
    walls_g = glbio.read_glb(os.path.join(ORIG, "bldg_walls.glb"))
    open_g = glbio.read_glb(os.path.join(ORIG, "bldg_openings.glb"))
    roof_g = glbio.read_glb(os.path.join(RAW, "bldg_roof.glb"))
    base_g = glbio.read_glb(os.path.join(RAW, "bldg_base.glb"))
    gr = [glbio.read_glb(os.path.join(RAW, f + ".glb")) for f in ("street", "ground", "green")]
    log("loaded %.1fs" % (time.time() - t0))

    # ---------------- wall orientation
    ws = walls.WallSoup(walls_g)
    roof_ray = walls.roof_intersector(roof_g)
    wins = extract.extract_d0(open_g)
    log("D0 quads:", len(wins))
    ev = extract.d0_orientation_votes(wins, ws)
    occ_mesh = mesh_of([walls_g, roof_g, base_g, open_g])
    views = walls.viewpoints_from_ground(gr)
    flipped = walls.fix_orientation(ws, roof_ray, ev, log=log, occl_mesh=occ_mesh, views=views)

    # ---------------- windows: merge, match, ground, depth
    wins = extract.merge_touching(wins, log=log)
    extract.match_walls(wins, ws)
    for w in wins:
        w.wall_origin = w.origin + w.n * (w.wall_offset if np.isfinite(w.wall_offset) else -0.005)
    extract.match_layer2(wins, ws)
    g_ray = walls.roof_intersector(gr[0], gr[1:] + [base_g])
    gy = extract.ground_heights(wins, g_ray)
    wb = extract.facade_bottom(wins, ws)
    for i, w in enumerate(wins):
        g = gy[i]
        if not np.isfinite(g) or g > w.origin[1] + 0.3:
            g = wb[i]
        w.ground = float(g)
        w.rel = float(w.origin[1] - g)
    occl = mesh_of([walls_g, roof_g, base_g]).ray
    first = ray_depth(occl, wins, [0.03] * len(wins))
    start2 = [w.layer2["depth"] + 0.03 if (w.layer2 and w.layer2["coverage"] > 0.99) else 0.03 for w in wins]
    beyond = ray_depth(occl, wins, start2)
    n_l2 = 0
    for w, a1, a2 in zip(wins, first, beyond):
        w.cut_layer2 = bool(w.layer2 and w.layer2["coverage"] > 0.99 and a1 <= w.layer2["depth"] + 0.02)
        w.avail = float(a2 if w.cut_layer2 else a1)
        n_l2 += w.cut_layer2

    # ---------------- clusters, styles, specs
    nc, lab = cluster_windows(wins)
    jn = joinery_near(open_g, wins, lab, nc)
    mats = {}
    for g in (open_g, walls_g):
        for m in g["json"]["materials"]:
            mats.setdefault(m["name"], m)
    mats.update({
        "Facade Sill Metal": new_mat("Facade Sill Metal", "#9A9FA2", 0.45, 0.35),
        "Facade Curtain #EEE9DF": new_mat("Facade Curtain #EEE9DF", "#EEE9DF", 0.95),
        "Facade Curtain #CFC6B4": new_mat("Facade Curtain #CFC6B4", "#CFC6B4", 0.95),
        "Facade Blind #DCDDD8": new_mat("Facade Blind #DCDDD8", "#DCDDD8", 0.6),
        "Facade Interior Ceiling": new_mat("Facade Interior Ceiling", "#2E2B28", 0.95),
        "Facade Shop Interior": new_mat("Facade Shop Interior", "#8A8278", 0.9),
        "Facade Shop Fit": new_mat("Facade Shop Fit", "#C8C2B6", 0.7),
    })
    M_base = {"glass": "Facade Glass v4", "metal": "Facade Hardware Metal v2", "interior": "Facade Interior Dark",
              "interior_ceil": "Facade Interior Ceiling", "sill_stone": "Facade Sill Stone",
              "sill_metal": "Facade Sill Metal", "curtain_a": "Facade Curtain #EEE9DF",
              "curtain_b": "Facade Curtain #CFC6B4", "blind": "Facade Blind #DCDDD8",
              "shop_interior": "Facade Shop Interior", "shop_interior_ceil": "Facade Interior Ceiling",
              "shop_fit": "Facade Shop Fit", "spandrel": "Facade Spandrel Glass v3 #33434C"}
    clusters = []
    for k in range(nc):
        idxs = np.where(lab == k)[0]
        rng = hrand("cluster", k, *np.round(wins[idxs[0]].center, 1))
        st = cluster_style(walls_g["nodes"], wins, idxs)
        wall_name = walls_g["nodes"][wins[idxs[0]].wall_node]["material"]["name"]
        whex = "#" + wall_name.split("#")[-1][:6] if "#" in wall_name else "#D0D0D0"
        if k in jn and not jn[k][0].startswith("#4B3528"):
            jcol = jn[k][0][:7]
            jsrc = "nearby"
        else:
            jcol = pick(rng, JOIN_PALETTE[st])
            jsrc = "palette"
        jname = "Facade Joinery " + jcol
        if jname not in mats:
            mats[jname] = new_mat(jname, jcol, 0.6)
        jmetal = "Facade Joinery #2F3337" if rng.random() < 0.65 else "Facade Joinery #8C969C"
        trim_hex = shade(whex, 1.07) if st == "altbau" else shade(whex, 0.86)
        tname = "Facade Trim " + trim_hex
        if tname not in mats:
            mats[tname] = new_mat(tname, trim_hex, 0.85)
        sh = pick(rng, [("#C8C6BE", 0.5), ("#9EA2A4", 0.3), ("#B9AD95", 0.2)])
        shname = "Facade Shutter " + sh
        if shname not in mats:
            mats[shname] = new_mat(shname, sh, 0.55)
        box_name = jname if st == "nachkrieg" else shname
        cl = dict(style=st, joinery=jname, joinery_src=jsrc, joinery_metal=jmetal, trim_mat=tname,
                  trim=(st == "altbau" and rng.random() < 0.75) or (st == "nachkrieg" and rng.random() < 0.6),
                  bars=int(rng.random() < 0.55) if st == "altbau" else 0,
                  shutters=st == "nachkrieg" and rng.random() < 0.85,
                  shutter_mat=shname, shutter_box_mat=box_name, n=len(idxs))
        clusters.append(cl)
        log("cluster %3d n=%3d style=%-9s wall=%s joinery=%s (%s) trim=%s bars=%d shutters=%s" % (
            k, len(idxs), st, wall_name[7:], jcol, jsrc, cl["trim"], cl["bars"], cl["shutters"]))
    specs = []
    for i, w in enumerate(wins):
        rng = hrand("win", i, *np.round(w.center, 2))
        s = window_spec(w, clusters[lab[i]], M_base, rng)
        s["rng_seed"] = ("win", i, *np.round(w.center, 2))
        # cut or flush; shallower reveal if the free depth is short
        need = s["d_f"] + windows.FD + 0.15
        w.flush_reason = ""
        if not (len(w.tris) and w.coverage > 0.995):
            w.mode, w.flush_reason = "flush", "no coplanar wall"
        elif w.layer2 and not w.cut_layer2 and w.layer2["dmin"] < s["d_f"] + windows.FD + 0.05:
            # partial second wall layer right behind: shallow reveal in front of it
            dmax = w.layer2["dmin"] - windows.FD - 0.03
            if dmax >= 0.08:
                s["d_f"] = dmax
                w.mode = "cut"
                w.avail = w.layer2["dmin"] - 0.01
            else:
                w.mode, w.flush_reason = "flush", "second wall layer too close"
        elif w.avail < need:
            dmax = w.avail - windows.FD - 0.12
            if dmax >= 0.08:
                s["d_f"] = dmax
                w.mode = "cut"
            else:
                w.mode, w.flush_reason = "flush", "no depth behind wall"
        else:
            w.mode = "cut"
        s["D_int"] = float(np.clip(w.avail - s["d_f"] - windows.FD - 0.04, 0.0, 1.1)) if w.mode == "cut" else 0.0
        specs.append(s)
    for s in specs:
        # a tilted sash swings up to 9 cm into the room: needs depth and no textiles in its way
        if s.get("tilt") and s["D_int"] < 0.25:
            s["tilt"] = 0.0
        if s.get("tilt"):
            s["textile"] = "none"
    nfc, nshr = resolve_conflicts(wins, specs)
    log("layer-2 cuts %d, corner conflicts -> flush %d, interior boxes shortened %d times" % (n_l2, nfc, nshr))

    # ---------------- cut
    cut_w = [w for w in wins if w.mode == "cut"]
    proxies = []
    for w in cut_w:
        if w.cut_layer2:
            mg = w.layer2["margin"]
            for tris, cov in zip(w.layer2["layers"], w.layer2["coverages"]):
                p = extract.Win()
                p.id = -1 - w.id
                p.r, p.u, p.n, p.width, p.height = w.r, w.u, w.n, w.width + 2 * mg, w.height + 2 * mg
                p.tris = tris
                p.coverage = cov
                p.wall_origin = w.wall_origin - w.r * mg - w.u * mg - w.n * w.layer2["mean"]
                p.parent = w
                proxies.append(p)
    for attempt in range(5):
        removed, new = cut.cut(ws, cut_w + proxies, log=log)
        bad = [w for w in cut_w if not getattr(w, "cut_ok", False)] +               [p.parent for p in proxies if not getattr(p, "cut_ok", False)]
        if not bad:
            break
        for w in bad:
            w.mode, w.flush_reason = "flush", "triangulation failed"
        cut_w = [w for w in cut_w if w.mode == "cut"]
        proxies = [p for p in proxies if p.parent.mode == "cut"]
    log("cut %d, flush %d %s" % (sum(w.mode == "cut" for w in wins), sum(w.mode == "flush" for w in wins),
                                 {r: sum(w.flush_reason == r for w in wins if w.mode == "flush") for r in
                                  set(w.flush_reason for w in wins if w.mode == "flush")}))

    # ---------------- assemblies + reveals
    out = {}          # material name -> list of (V, N, F)
    rev = {}          # wall node -> list of tri soup (T, N, U)
    kinds = {}
    tri_per = []
    for i, w in enumerate(wins):
        s = specs[i]
        rng = hrand(*s["rng_seed"], "geo")
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        R = np.stack([w.r, w.u, w.n], 1)
        if w.mode == "cut":
            hole_local = {k: (v - w.wall_origin) @ R for k, v in w.hole.items()}
            s["hole_local"] = hole_local
            B = windows.make_window(s, rng)
            _reveal(rev, w, s)
        else:
            s["d_f"] = 0.0
            s["D_int"] = 0.0
            B = _flush_window(s, rng)
        n0 = B.tri_count()
        tri_per.append(n0)
        for mat, (V, N, F) in B.transformed(w.wall_origin, w.r, w.u, w.n).items():
            out.setdefault(mat, []).append((V, N, F))
    log("kinds:", kinds)
    log("tris/window: mean %.0f, max %d, total %d" % (np.mean(tri_per), np.max(tri_per), np.sum(tri_per)))

    # ---------------- write walls
    keep = ~removed
    allT = np.concatenate([ws.T[keep], new["T"]])
    allN = np.concatenate([ws.N[keep], new["N"]])
    allU = np.concatenate([ws.U[keep], new["U"]])
    allNI = np.concatenate([ws.NI[keep], new["NI"]])
    wnodes = []
    wall_tris = 0
    for ni, nd in enumerate(walls_g["nodes"]):
        m = allNI == ni
        T, N, U = allT[m], allN[m], allU[m]
        if ni in rev:
            T = np.concatenate([T] + [r[0] for r in rev[ni]])
            N = np.concatenate([N] + [r[1] for r in rev[ni]])
            U = np.concatenate([U] + [r[2] for r in rev[ni]])
        has_uv = walls_g["nodes"][ni]["prims"][0]["uv"] is not None
        P, Nn, Uu, F = weld(T, N, U if has_uv else None)
        wnodes.append(dict(name=nd["name"], material=nd["material"], pos=P, nrm=Nn, uv=Uu, idx=F))
        wall_tris += len(F)
    glbio.write_glb(os.path.join(RAW, "bldg_walls.glb"), wnodes)
    log("bldg_walls.glb: %d tris (orig %d), flipped %d" % (wall_tris, len(ws.T), int(flipped.sum())))

    # ---------------- write openings without D0
    onodes = []
    for nd in open_g["nodes"]:
        if nd["name"].endswith("Facade Glass D0"):
            continue
        p = nd["prims"][0]
        onodes.append(dict(name=nd["name"], material=nd["material"], pos=p["pos"], nrm=p["nrm"], uv=p["uv"], idx=p["idx"]))
    glbio.write_glb(os.path.join(RAW, "bldg_openings.glb"), onodes)
    log("bldg_openings.glb: %d tris" % sum(len(n["idx"]) for n in onodes))

    # ---------------- write d0_windows
    dnodes = []
    d_tris = 0
    for mat in sorted(out):
        L = out[mat]
        V = np.concatenate([a for a, _, _ in L]); N = np.concatenate([b for _, b, _ in L])
        F, o = [], 0
        for a, _, f in L:
            F.append(f + o); o += len(a)
        F = np.concatenate(F)
        T = V[F]; NN = N[F]
        P, Nn, _, Fi = weld(T, NN, None)
        dnodes.append(dict(name="d0_windows__" + mat, material=mats[mat], pos=P, nrm=Nn, uv=None, idx=Fi))
        d_tris += len(Fi)
        log("  %-40s %8d tris" % (mat, len(Fi)))
    glbio.write_glb(os.path.join(RAW, "d0_windows.glb"), dnodes)
    log("d0_windows.glb: %d tris, %d materials" % (d_tris, len(dnodes)))

    stats = dict(d0_quads=int(sum(getattr(w, "merged", 1) for w in wins)), windows=len(wins),
                 cut=sum(w.mode == "cut" for w in wins), flush=sum(w.mode == "flush" for w in wins),
                 layer2_cuts=int(sum(1 for w in wins if w.mode == "cut" and w.cut_layer2)),
                 kinds=kinds, tris_d0_windows=d_tris, tris_walls=wall_tris, tris_walls_orig=int(len(ws.T)),
                 walls_flipped=int(flipped.sum()), tris_per_window_mean=float(np.mean(tri_per)),
                 clusters=[{k: v for k, v in c.items()} for c in clusters])
    json.dump(stats, open(os.path.join(HERE, "d0_stats.json"), "w"), indent=1)
    json.dump([dict(id=i, c=[float(x) for x in w.center], n=[float(x) for x in w.n], w=float(w.width), h=float(w.height),
                    cl=int(lab[i]), style=clusters[lab[i]]["style"], mode=w.mode, rel=float(w.rel),
                    kind=specs[i]["kind"], d_f=float(specs[i]["d_f"]), D_int=float(specs[i]["D_int"]),
                    avail=float(w.avail), cut_layer2=bool(w.cut_layer2), flush_reason=w.flush_reason,
                    layer2=None if not w.layer2 else {k: v for k, v in w.layer2.items() if k != "layers"})
               for i, w in enumerate(wins)], open(os.path.join(HERE, "cache_windows.json"), "w"))
    if compress:
        compress_web()
    if "--check" in sys.argv:
        subprocess.run([sys.executable, os.path.join(HERE, "check.py")])
    if "--previews" in sys.argv:
        blender = r"C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
        for st in ("after", "before"):
            subprocess.run([blender, "-b", "--factory-startup", "-P", os.path.join(HERE, "preview.py"), "--", st, "all"])
    log("done %.1fs" % (time.time() - t0))
    open(os.path.join(HERE, "build_log.txt"), "w").write("\n".join(LOG))


def _reveal(rev, w, s):
    """plaster reveal strips from the wall face back to the frame back, into the wall node."""
    D = s["d_f"] + windows.FD
    sides = {"bottom": w.u, "right": -w.r, "top": -w.u, "left": w.r}
    T, N, U = [], [], []
    for name, inward in sides.items():
        if name == "bottom" and s.get("sill"):
            continue
        pts = w.hole[name]
        for a, b in zip(pts[:-1], pts[1:]):
            a2, b2 = a - w.n * D, b - w.n * D
            q = [a, b, b2, a2]
            nrm = np.cross(q[1] - q[0], q[2] - q[0])
            if np.linalg.norm(nrm) < 1e-12:
                continue
            if np.dot(nrm, inward) < 0:
                q = [a, a2, b2, b]
            T += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            N += [[inward] * 3, [inward] * 3]
            U += [[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]]
    if T:
        rev.setdefault(w.wall_node, []).append((np.array(T, float), np.array(N, float), np.array(U, float)))


def _flush_window(s, rng):
    """fallback: assembly on the wall surface, max 3 cm proud, fake reveal frame."""
    import assembly as A
    B = A.Builder()
    M = s["M"]
    w, h = s["w"], s["h"]
    # dark backing
    B.quad(M["interior"], (0, 0, 0.003), (w, 0, 0.003), (w, h, 0.003), (0, h, 0.003), expect=np.array([0, 0, 1.0]))
    # fake reveal: bevelled ring of wall colour
    prof = [(0.0, 0.0), (0.0, 0.03), (0.02, 0.03), (0.07, 0.008), (0.07, 0.0)]
    B.ring(M["trim"], -0.07, -0.07, w + 0.07, h + 0.07, prof, zref=0.0)
    fw = 0.05
    B.ring(M["joinery"], 0.0, 0.0, w, h, [(0.0, 0.004), (0.0, 0.022), (0.004, 0.026), (fw, 0.026), (fw, 0.004)], zref=0.0)
    B.quad(M["glass"], (fw, fw, 0.012), (w - fw, fw, 0.012), (w - fw, h - fw, 0.012), (fw, h - fw, 0.012),
           expect=np.array([0, 0, 1.0]))
    if w > 0.9:
        B.box(M["joinery"], (w / 2 - 0.035, fw, 0.004), (w / 2 + 0.035, h - fw, 0.024))
    return B


def compress_web():
    os.makedirs(WEB, exist_ok=True)
    for f in ("bldg_walls", "bldg_openings", "d0_windows"):
        src = os.path.join(RAW, f + ".glb")
        dst = os.path.join(WEB, f + ".glb")
        cmd = ["npx", "-y", "@gltf-transform/cli@4", "optimize", src, dst, "--compress", "meshopt",
               "--texture-compress", "webp", "--simplify", "false", "--instance", "false"]
        t = time.time()
        r = subprocess.run(cmd, shell=(os.name == "nt"), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log("compress failed", f, r.stderr[-2000:])
        else:
            log("web %s: %.1f MB (%.0fs)" % (f, os.path.getsize(dst) / 1e6, time.time() - t))


if __name__ == "__main__":
    main()

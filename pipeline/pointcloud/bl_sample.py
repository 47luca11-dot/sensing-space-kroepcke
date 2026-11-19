# Stadtmodell -> Punktwolke, Schritt 1 (läuft IM Blender, über das Blender-MCP: exec(open(<diese Datei>).read())).
# Importiert jede sichtbare City-GLB aus web/assets/city/manifest.json, verteilt Punkte flächenproportional
# (Dichte hängt vom Ort ab: dicht am Kröpcke, in der Altstadt/Leibnizhaus und an der Passerelle, dünner außen),
# nimmt Farbe aus Textur * Materialfaktor * Vertexfarbe (sRGB) und die Flächennormale mit und schreibt je GLB
# eine Zwischendatei data/pointcloud/stage/<name>.npz  (P float32 n×3 Spielkoordinaten, C uint8 n×4 RGB+Flags, N uint8 n×2 Okta-Normale).
# Flags (C[:,3]): 1 = zweiseitig (keine Rückseiten-Unterdrückung), 2 = leuchtend (unbeleuchtet), 4 = Glas.
# Schritt 2: bl_tile.py (Kacheln, Mischen, Quantisieren -> web/assets/city/pointcloud/).
import bpy, numpy as np, os, json, time, math, re, traceback

ROOT = r"C:\Users\luca\Documents\Sensing Space\Claude Code\output\hannover"
CITY = os.path.join(ROOT, "web", "assets", "city")
STAGE = os.path.join(ROOT, "data", "pointcloud", "stage")
os.makedirs(STAGE, exist_ok=True)
PROGRESS = os.path.join(STAGE, "_progress.json")

# ---------------------------------------------------------------- Dichte (Punkte pro m²)
CORE, BASE, FALL = 110.0, 9.0, 70.0
# Kapseln (x0,z0,x1,z1,r) in Spielkoordinaten: Kernbereiche mit voller Dichte
HOT = [
    (0, 0, 0, 0, 190),                 # Kröpcke
    (-20, -15, 175, -275, 60),         # Niki-de-Saint-Phalle-Promenade bis Hbf
    (-40, 110, -300, 300, 80),         # Karmarschstraße -> Marktkirche
    (-300, 300, -450, 340, 85),        # Altstadt -> Holzmarkt / Leibnizhaus (Gottfrieds Feinkiosk)
]
def density(x, z):
    d = np.full(x.shape, 1e9)
    for x0, z0, x1, z1, r in HOT:
        ax, az = x1 - x0, z1 - z0
        L2 = ax * ax + az * az
        t = np.zeros_like(x) if L2 == 0 else np.clip(((x - x0) * ax + (z - z0) * az) / L2, 0, 1)
        dd = np.hypot(x - (x0 + t * ax), z - (z0 + t * az)) - r
        d = np.minimum(d, dd)
    return BASE + (CORE - BASE) * np.exp(-np.maximum(d, 0) / FALL)

# Handmodelle mit vielen Kleinteilen: dichter abtasten (Faktor auf die Ortsdichte)
FILE_MULT = {"landmarks/gottfried_holzmarkt6.glb": 3.2}
CUR = {"mult": 1.0}

RX_TWO = re.compile(r"tree|baum|leaf|laub|krone|foliage|hedge|hecke|busch|bambus|bamboo|plant|green|gras|cable|kabel|seil|rope|net\b|netz|fence|zaun|gitter|grate|banner|flag|fahne|sign|schild|parasol|schirm|canopy|markise|glass|glas|scheibe|window|fenster|lichtwolke", re.I)
RX_GLASS = re.compile(r"glass|glas\b|glas_|scheibe|window glass|fenster.?glas", re.I)
RX_TREE = re.compile(r"^tree_[ab]|^tree_[ab].", re.I)        # Stadtmöbel-Bäume (Blattkarten)
RX_LEAF = re.compile(r"^tree", re.I)                              # Blatt-Materialien tree_a/tree_b/tree_top (nicht Rinde)
RX_SKIP = re.compile(r"(^col_|_collision$|^collision)", re.I)

def lin2srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)

def oct_encode(n):
    n = n / np.maximum(np.abs(n).sum(1, keepdims=True), 1e-9)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    neg = z < 0
    ox = np.where(neg, (1 - np.abs(y)) * np.sign(x + 1e-12), x)
    oy = np.where(neg, (1 - np.abs(x)) * np.sign(y + 1e-12), y)
    return np.clip(np.round((np.stack([ox, oy], 1) * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- Material auslesen
_img_cache = {}
def img_array(img):
    k = img.name
    if k in _img_cache: return _img_cache[k]
    w, h = img.size
    if w == 0 or h == 0: _img_cache[k] = None; return None
    a = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(h, w, 4)
    if img.colorspace_settings.name.lower() not in ("srgb",):   # linear gespeicherte Bilder -> sRGB
        a = a.copy(); a[..., :3] = lin2srgb(a[..., :3])
    _img_cache[k] = a
    return a

def upstream(sock, depth=0, seen=None):
    """alle Knoten oberhalb eines Eingangs"""
    out = []
    if seen is None: seen = set()
    for l in sock.links:
        n = l.from_node
        if n.name in seen: continue
        seen.add(n.name); out.append(n)
        if depth < 6:
            for s in n.inputs: out += upstream(s, depth + 1, seen)
    return out

_mat_cache = {}
def mat_info(mat):
    if mat is None: return dict(base=np.array([0.7, 0.7, 0.7]), img=None, vcol=False, alpha=1.0, emis=None, name="")
    if mat.name in _mat_cache: return _mat_cache[mat.name]
    info = dict(base=np.array(mat.diffuse_color[:3], dtype=np.float32), img=None, vcol=False, alpha=float(mat.diffuse_color[3]), emis=None, name=mat.name, img_alpha=False)
    if mat.use_nodes and mat.node_tree:
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            bc = bsdf.inputs["Base Color"]
            if bc.is_linked:
                ups = upstream(bc)
                tex = next((n for n in ups if n.type == "TEX_IMAGE" and n.image), None)
                if tex: info["img"] = tex.image
                if any(n.type in ("VERTEX_COLOR", "ATTRIBUTE") for n in ups): info["vcol"] = True
                fac = None
                for n in ups:
                    if n.type in ("MIX", "MIX_RGB"):
                        for s in n.inputs:
                            if s.type == "RGBA" and not s.is_linked:
                                fac = np.array(s.default_value[:3]); break
                    if n.type == "RGB": fac = np.array(n.outputs[0].default_value[:3])
                    if fac is not None: break
                info["base"] = fac if fac is not None else np.ones(3, dtype=np.float32)
            else:
                info["base"] = np.array(bc.default_value[:3], dtype=np.float32)
            al = bsdf.inputs["Alpha"]
            if al.is_linked:
                info["img_alpha"] = True
            else:
                info["alpha"] = float(al.default_value)
            es = bsdf.inputs.get("Emission Strength"); ec = bsdf.inputs.get("Emission Color")
            if es is not None and ec is not None and es.default_value > 0:
                e = np.array(ec.default_value[:3]) * es.default_value
                if ec.is_linked: e = np.ones(3) * es.default_value
                if e.max() > 0.05: info["emis"] = e
    if mat.blend_method in ("BLEND",) and info["alpha"] > 0.95 and not info["img_alpha"]: pass
    _mat_cache[mat.name] = info
    return info

# ---------------------------------------------------------------- Mesh -> Punkte
_mesh_cache = {}
def mesh_arrays(me):
    k = me.name
    if k in _mesh_cache: return _mesh_cache[k]
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    if nt == 0: _mesh_cache[k] = None; return None
    V = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    TV = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("vertices", TV); TV = TV.reshape(-1, 3)
    TL = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("loops", TL); TL = TL.reshape(-1, 3)
    MI = np.empty(nt, np.int32); me.loop_triangles.foreach_get("material_index", MI)
    UV = None
    if me.uv_layers.active:
        UV = np.empty(len(me.loops) * 2, np.float32); me.uv_layers.active.data.foreach_get("uv", UV); UV = UV.reshape(-1, 2)
    VC = None
    ca = me.color_attributes.active_color if hasattr(me, "color_attributes") else None
    if ca is not None and len(ca.data):
        arr = np.empty(len(ca.data) * 4, np.float32); ca.data.foreach_get("color", arr); arr = arr.reshape(-1, 4)
        VC = (ca.domain, arr)
    r = dict(V=V, TV=TV, TL=TL, MI=MI, UV=UV, VC=VC)
    _mesh_cache[k] = r
    return r

RNG = np.random.default_rng(1977)

def kahler_baum(lo, hi, dens):
    """Kahler Laubbaum (z. B. Linde/Ahorn im Winter) in die Hülle lo..hi (Spielkoordinaten, y oben).
    Rekursive Verzweigung: Stamm bis ~35 % der Höhe, 3-5 Hauptäste, je Stufe 2-3 Seitenäste (Länge x0,68, Dicke x0,6),
    leicht gekrümmt und nach außen/oben strebend, Zweigspitzen bis an die Hüllenoberfläche. Punkte auf den Astmänteln."""
    seed = int(abs(lo[0] * 73.1 + lo[2] * 19.7)) % (2 ** 31); rng = np.random.default_rng(seed)
    c = (lo + hi) / 2; H = hi[1] - lo[1]; R = max(0.6, 0.5 * max(hi[0] - lo[0], hi[2] - lo[2]))
    base = np.array([c[0], lo[1], c[2]])
    segs = []                                                         # (a, b, r)
    def ast(p, d, L, r, tiefe):
        n = max(2, int(L / 0.35)); q = p.copy(); dd = d.copy()
        for i in range(n):
            dd = dd + rng.normal(scale=0.08, size=3); dd[1] += 0.03; dd /= np.linalg.norm(dd)
            q2 = q + dd * (L / n); rr = r * (1 - 0.35 * (i + 1) / n); segs.append((q, q2, rr)); q = q2
            if tiefe < 5 and i >= n // 4 and rng.random() < 0.6:           # Seitenast unterwegs
                side = np.cross(dd, rng.normal(size=3)); side /= np.linalg.norm(side) + 1e-9
                nd = dd * 0.55 + side * 0.7 + np.array([0, 0.25, 0]); nd /= np.linalg.norm(nd)
                ast(q, nd, L * rng.uniform(0.42, 0.62), rr * 0.6, tiefe + 1)
        if tiefe >= 5:                                                   # feine Zweigspitzen (Reisig)
            for _ in range(rng.integers(3, 6)):
                nd = dd + rng.normal(scale=0.6, size=3); nd[1] += 0.3; nd /= np.linalg.norm(nd)
                segs.append((q, q + nd * rng.uniform(0.25, 0.6), 0.006))
        if tiefe < 5:                                                    # Gabelung am Ende
            for _ in range(rng.integers(2, 4)):
                side = rng.normal(size=3); side[1] = abs(side[1]) * 0.5; side /= np.linalg.norm(side)
                nd = dd * 0.6 + side * 0.65; nd /= np.linalg.norm(nd)
                # Hülle: Richtung zurück ins Ellipsoid lenken
                rel = (q - np.array([c[0], lo[1] + 0.6 * H, c[2]])) / np.array([R, 0.45 * H, R])
                if np.linalg.norm(rel) > 0.8: nd -= rel * 0.5; nd /= np.linalg.norm(nd)
                ast(q, nd, L * rng.uniform(0.6, 0.75), r * 0.6, tiefe + 1)
    stamm_h = H * rng.uniform(0.30, 0.40); r0 = float(np.clip(0.06 * H / 2.5, 0.12, 0.28))
    top = base + np.array([rng.normal(scale=0.08), stamm_h, rng.normal(scale=0.08)])
    segs.append((base, top, r0))
    nh = int(rng.integers(4, 7))
    for k in range(nh):
        a = k / nh * 2 * np.pi + rng.uniform(0, 0.8)
        d = np.array([np.cos(a) * 0.55, 0.85, np.sin(a) * 0.55]); d /= np.linalg.norm(d)
        ast(top, d, H * rng.uniform(0.28, 0.36), r0 * 0.62, 1)
    # Punkte auf den Mänteln (Dichte wie Umgebung, dünne Zweige mindestens ein paar Punkte)
    A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs]); r = np.array([s[2] for s in segs])
    Lg = np.linalg.norm(B - A, axis=1); area = 2 * np.pi * np.maximum(r, 0.01) * Lg
    # Stamm/Äste: Mantel dicht (gut lesbare Form), Zweige: Punkte entlang der Linie; Budget nur über die Zweige
    duenn = r < 0.02
    cnt = np.where(duenn, Lg / 0.045, np.maximum(area * max(dens, 60.0) * 3.0, Lg / 0.02))
    BUDGET = 50000                                                    # Punkte je Baum (Tausende Bäume in der Stadt)
    dick_n = cnt[~duenn].sum()
    if dick_n > BUDGET * 0.6: cnt[~duenn] *= BUDGET * 0.6 / dick_n; dick_n = BUDGET * 0.6
    duenn_n = cnt[duenn].sum()
    if duenn_n > BUDGET - dick_n: cnt[duenn] *= (BUDGET - dick_n) / duenn_n
    cnt = np.floor(cnt + rng.random(len(cnt))).astype(int)
    idx = np.repeat(np.arange(len(segs)), cnt); n = len(idx)
    t = rng.random(n); ax = (B - A)[idx] / np.maximum(Lg[idx, None], 1e-6)
    u = np.cross(ax, np.array([0.31, 0.95, 0.07])); u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    v = np.cross(ax, u); ang = rng.random(n) * 2 * np.pi
    nrm = u * np.cos(ang)[:, None] + v * np.sin(ang)[:, None]
    Pp = A[idx] + (B - A)[idx] * t[:, None] + nrm * r[idx, None]
    hell = 0.85 + 0.3 * rng.random(n) + 0.15 * np.clip(nrm[:, 1], 0, 1)          # Oberseiten etwas heller (Licht, Flechten)
    col = np.array([92, 84, 76]) * hell[:, None]
    jung = r[idx] < 0.03; col[jung] = col[jung] * 0.85 + np.array([12, 8, 4])     # junge Zweige rötlich-braun
    return Pp, nrm, np.clip(col, 0, 255).astype(np.uint8)

def sample_object(ob, out, stats):
    if ob.type != "MESH" or RX_SKIP.search(ob.name): return
    A = mesh_arrays(ob.data)
    if A is None: return
    M = np.array(ob.matrix_world, dtype=np.float64)
    V = A["V"] @ M[:3, :3].T + M[:3, 3]
    G = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)            # Blender (Z oben) -> Spiel (Y oben, -Z Nord)
    T = A["TV"]
    a, b, c = G[T[:, 0]], G[T[:, 1]], G[T[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    ok = area2 > 1e-10
    nrm = np.zeros_like(cr); nrm[ok] = cr[ok] / area2[ok, None]
    area = area2 * 0.5
    cen = (a + b + c) / 3
    dens = density(cen[:, 0], cen[:, 2]) * CUR["mult"]
    mats = [s.material for s in ob.material_slots] or [None]
    MI = np.clip(A["MI"], 0, len(mats) - 1)
    infos = [mat_info(m) for m in mats]
    mult = np.ones(len(T))
    # Dächer und Unterseiten sieht man vom Gehweg kaum -> dünner
    mult[(nrm[:, 1] > 0.6) & (cen[:, 1] > 5.0)] = 0.22
    mult[nrm[:, 1] < -0.6] = 0.5
    flags_t = np.zeros(len(T), np.uint8)
    obj_two = bool(RX_TWO.search(ob.name))
    for i, inf in enumerate(infos):
        sel = MI == i
        if not sel.any(): continue
        nm = inf["name"]
        glass = bool(RX_GLASS.search(nm)) or (inf["alpha"] < 0.6 and not inf["img_alpha"])
        if glass: mult[sel] *= 0.25; flags_t[sel] |= 4 | 1
        if obj_two or RX_TWO.search(nm) or inf["img_alpha"]: flags_t[sel] |= 1
        if inf["emis"] is not None: flags_t[sel] |= 2
    cnt_f = area * dens * mult
    cnt = np.floor(cnt_f + RNG.random(len(T))).astype(np.int64)
    cnt[~ok] = 0
    n = int(cnt.sum())
    stats["area"] += float(area.sum()); stats["tris"] += len(T)
    if n == 0: return
    ti = np.repeat(np.arange(len(T)), cnt)
    r1 = RNG.random(n); r2 = RNG.random(n)
    s1 = np.sqrt(r1)
    w0, w1, w2 = 1 - s1, s1 * (1 - r2), s1 * r2
    P = a[ti] * w0[:, None] + b[ti] * w1[:, None] + c[ti] * w2[:, None]
    col = np.empty((n, 3), np.float32)
    alpha_keep = np.ones(n, bool)
    mi = MI[ti]
    for i, inf in enumerate(infos):
        sel = np.nonzero(mi == i)[0]
        if len(sel) == 0: continue
        base = lin2srgb(inf["base"])
        cc = np.broadcast_to(base, (len(sel), 3)).copy()
        img = inf["img"]
        if img is not None and A["UV"] is not None:
            arr = img_array(img)
            if arr is not None:
                L = A["TL"][ti[sel]]
                uv = A["UV"][L[:, 0]] * w0[sel, None] + A["UV"][L[:, 1]] * w1[sel, None] + A["UV"][L[:, 2]] * w2[sel, None]
                h, w = arr.shape[:2]
                px = (np.floor(np.mod(uv[:, 0], 1.0) * w).astype(np.int64)) % w
                py = (np.floor(np.mod(uv[:, 1], 1.0) * h).astype(np.int64)) % h
                tx = arr[py, px]
                cc *= tx[:, :3]
                if inf["img_alpha"]: alpha_keep[sel] = tx[:, 3] > 0.45
        if inf["vcol"] and A["VC"] is not None:
            dom, vc = A["VC"]
            if dom == "CORNER":
                L = A["TL"][ti[sel]]
                v = vc[L[:, 0]] * w0[sel, None] + vc[L[:, 1]] * w1[sel, None] + vc[L[:, 2]] * w2[sel, None]
            else:
                VV = A["TV"][ti[sel]]
                v = vc[VV[:, 0]] * w0[sel, None] + vc[VV[:, 1]] * w1[sel, None] + vc[VV[:, 2]] * w2[sel, None]
            cc *= lin2srgb(v[:, :3])
        if inf["emis"] is not None:
            cc = np.maximum(cc, lin2srgb(np.clip(inf["emis"], 0, 1)))
        col[sel] = cc
    keep = alpha_keep
    nrm_pt = nrm[ti]
    # Bäume aus gekreuzten Blattkarten (tree_a/tree_b) -> realistische KAHLE Bäume (Winter, 2026-11-19): Stamm, Hauptäste,
    # Zweige rekursiv verzweigt, Punkte auf den Ast-Oberflächen in Rindenfarbe. Die Karten (Blätter + Rindenebene) entfallen;
    # als flache Punkt-Ebenen standen sie wie Mauern quer über Straße und Passage.
    if RX_TREE.search(ob.name) or RX_TREE.search(ob.data.name):
        L = P[keep]
        if len(L) > 20:
            lo, hi = np.percentile(L, 2, axis=0), np.percentile(L, 98, axis=0)
            lo[1] = float(L[:, 1].min())                                         # Stammfuß auf dem Boden
            Pb, Nb, Cb = kahler_baum(lo, hi, float(np.median(dens)) * CUR["mult"])
            out["P"].append(Pb.astype(np.float32)); out["N"].append(oct_encode(Nb))
            Cc = np.empty((len(Pb), 4), np.uint8); Cc[:, :3] = Cb; Cc[:, 3] = 1          # zweiseitig
            out["C"].append(Cc); stats["pts"] += len(Pb); stats["baeume"] = stats.get("baeume", 0) + 1
            return
    out["P"].append(P[keep].astype(np.float32))
    C = np.empty((int(keep.sum()), 4), np.uint8)
    C[:, :3] = np.clip(np.round(col[keep] * 255), 0, 255).astype(np.uint8)
    C[:, 3] = flags_t[ti][keep]
    out["C"].append(C)
    out["N"].append(oct_encode(nrm_pt[keep]))
    stats["pts"] += int(keep.sum())

WORK_SCENE = "PC_Import"          # eigene Import-Szene: andere Szenen (z. B. Gottfried_Holzmarkt6) bleiben unberührt

def work_scene():
    if bpy.context.window is None: return bpy.context.scene      # headless (blender -b): aktive Szene benutzen
    sc = bpy.data.scenes.get(WORK_SCENE) or bpy.data.scenes.new(WORK_SCENE)
    if bpy.context.window.scene != sc: bpy.context.window.scene = sc
    return sc

def clear_scene():
    sc = work_scene()
    for o in list(sc.objects):
        if len(o.users_scene) <= 1: bpy.data.objects.remove(o, do_unlink=True)
    for _ in range(2):
        for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures, bpy.data.node_groups, bpy.data.cameras, bpy.data.lights):
            for x in list(coll):
                if x.users == 0:
                    try: coll.remove(x)
                    except Exception: pass
    _img_cache.clear(); _mat_cache.clear(); _mesh_cache.clear()

def stage_name(url): return url.replace("/", "__").replace(".glb", ".npz")

def process_file(url):
    t0 = time.time()
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=os.path.join(CITY, url), loglevel=50)
    CUR["mult"] = FILE_MULT.get(url, 1.0)
    out = dict(P=[], C=[], N=[]); stats = dict(area=0.0, tris=0, pts=0)
    for ob in list(work_scene().objects): sample_object(ob, out, stats)
    if out["P"]:
        np.savez(os.path.join(STAGE, stage_name(url)), P=np.concatenate(out["P"]), C=np.concatenate(out["C"]), N=np.concatenate(out["N"]))
    else:
        np.savez(os.path.join(STAGE, stage_name(url)), P=np.zeros((0, 3), np.float32), C=np.zeros((0, 4), np.uint8), N=np.zeros((0, 2), np.uint8))
    clear_scene()
    stats["sec"] = round(time.time() - t0, 2)
    return stats

def job_list(only=None):
    man = json.load(open(os.path.join(CITY, "manifest.json"), encoding="utf-8"))
    urls = [f["url"] for f in man["files"] if f.get("kind") in ("visual", "both")]
    if only: urls = [u for u in urls if re.search(only, u)]
    return urls

# ---------------------------------------------------------------- Hintergrund-Lauf per Timer (Blender bleibt bedienbar)
STATE = {"queue": [], "done": [], "err": [], "t0": 0, "tot": dict(area=0.0, tris=0, pts=0)}
def _write_progress(running):
    json.dump({"running": running, "todo": len(STATE["queue"]), "done": len(STATE["done"]), "err": STATE["err"][-10:],
               "elapsed": round(time.time() - STATE["t0"], 1), "tot": STATE["tot"], "last": STATE["done"][-3:]},
              open(PROGRESS, "w"), indent=1)

def _tick():
    if not STATE["queue"]:
        _write_progress(False); return None
    url = STATE["queue"].pop(0)
    try:
        s = process_file(url)
        for k in ("area", "tris", "pts"): STATE["tot"][k] += s[k]
        STATE["done"].append([url, s])
    except Exception as e:
        STATE["err"].append([url, repr(e), traceback.format_exc()[-600:]])
    _write_progress(True)
    return 0.05

def start(only=None, skip_done=True):
    urls = job_list(only)
    if skip_done: urls = [u for u in urls if not os.path.exists(os.path.join(STAGE, stage_name(u)))]
    STATE.update(queue=urls, done=[], err=[], t0=time.time(), tot=dict(area=0.0, tris=0, pts=0))
    _write_progress(True)
    if bpy.app.timers.is_registered(_tick): bpy.app.timers.unregister(_tick)
    bpy.app.timers.register(_tick, first_interval=0.1)
    return len(urls)

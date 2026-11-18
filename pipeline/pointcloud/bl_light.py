# Stadtmodell -> Punktwolke, Schritt 3: Licht backen (läuft IM Blender über das MCP, nach bl_tile.py).
# 1) Alle Stadt-GLBs (ohne instanzierte Fenster) werden zu EINEM Verdecker-Mesh "PC_Verdecker" zusammengefügt (Szene "PC_Licht").
# 2) Die Punkte der Kacheln (web/assets/city/pointcloud/t_*.bin) werden stapelweise als Punkt-Mesh "PC_Stapel" geladen;
#    ein Geometry-Nodes-Modifier ("PC_Licht_Raycast") schießt je Punkt Strahlen gegen das Verdecker-Mesh:
#      - 1 Sonnenstrahl  -> harter Schlagschatten
#      - 6 kurze Strahlen (2 m) um die Normale -> Umgebungsverdeckung (Ecken, Laibungen, unter Möbeln)
#      - 4 lange Strahlen (60 m) nach oben -> Himmelssicht (Straßenschluchten, Arkaden, Passerelle dunkler)
# 3) Farbe = Albedo * (Himmel * Himmelssicht * AO + Sonne * Schatten * N·L) wird in die Kacheln zurückgeschrieben
#    (sRGB, leuchtende Punkte bleiben unbeleuchtet). index.json bekommt "baked": {...}.
# Nach jedem neuen Lauf von bl_tile.py (der die Albedo-Farben schreibt) muss dieser Schritt wiederholt werden.
import bpy, numpy as np, os, json, time, glob, math, re, traceback

ROOT = r"C:\Users\luca\Documents\Sensing Space\Claude Code\output\hannover"
CITY = os.path.join(ROOT, "web", "assets", "city")
PCD = os.path.join(CITY, "pointcloud")
PROGRESS = os.path.join(ROOT, "data", "pointcloud", "stage", "_light_progress.json")
SCENE = "PC_Licht"
SUN_G = np.array([-0.45, 0.8, 0.38]); SUN_G /= np.linalg.norm(SUN_G)        # Richtung zur Sonne (Spiel, wie main.js)
SUN_B = np.array([SUN_G[0], -SUN_G[2], SUN_G[1]])                           # Blender (Z oben)
SUN_COL = np.array([1.00, 0.90, 0.76]) * 1.55
SKY_COL = np.array([0.62, 0.72, 0.90]) * 0.80
BOUNCE = np.array([0.05, 0.045, 0.04])
BATCH = 6_000_000
RX_SKIP_FILE = re.compile(r"_win\.glb$")
RX_SKIP_OBJ = re.compile(r"glass|glas\b|glas_|scheibe|^col_|_collision$", re.I)
AO_DIRS = [(1, 0, 0.35), (-1, 0, 0.35), (0, 1, 0.35), (0, -1, 0.35), (0.7, 0.7, -0.2), (-0.7, -0.7, -0.2)]
SKY_DIRS = [(0.45, 0.45, 1), (-0.45, 0.45, 1), (0.45, -0.45, 1), (-0.45, -0.45, 1)]

def scene():
    if bpy.context.window is None: return bpy.context.scene      # headless (blender -b)
    sc = bpy.data.scenes.get(SCENE) or bpy.data.scenes.new(SCENE)
    if bpy.context.window.scene != sc: bpy.context.window.scene = sc
    return sc

# ------------------------------------------------------------------ Verdecker-Mesh
OCC = {"V": [], "T": [], "nv": 0}
def _purge_import(sc):
    for o in list(sc.objects):
        if o.name.startswith("PC_"): continue
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.node_groups):
        for x in list(coll):
            if x.users == 0 and not x.name.startswith("PC_"):
                try: coll.remove(x)
                except Exception: pass

def add_occluder_file(url):
    sc = scene()
    bpy.ops.import_scene.gltf(filepath=os.path.join(CITY, url), loglevel=50)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in list(sc.objects):
        if o.type != "MESH" or o.name.startswith("PC_") or RX_SKIP_OBJ.search(o.name): continue
        me = o.data; me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        if nt == 0: continue
        V = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
        M = np.array(o.matrix_world, np.float64)
        V = (V @ M[:3, :3].T + M[:3, 3]).astype(np.float32)
        T = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("vertices", T)
        OCC["V"].append(V); OCC["T"].append(T.reshape(-1, 3) + OCC["nv"]); OCC["nv"] += len(V)
    _purge_import(sc)

def finish_occluder():
    sc = scene()
    V = np.concatenate(OCC["V"]); T = np.concatenate(OCC["T"])
    OCC["V"].clear(); OCC["T"].clear()
    old = bpy.data.objects.get("PC_Verdecker")
    if old: bpy.data.objects.remove(old, do_unlink=True)
    me = bpy.data.meshes.new("PC_Verdecker")
    me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
    me.loops.add(T.size); me.loops.foreach_set("vertex_index", T.ravel())
    me.polygons.add(len(T)); me.polygons.foreach_set("loop_start", np.arange(0, T.size, 3, dtype=np.int32))
    me.update()
    ob = bpy.data.objects.new("PC_Verdecker", me); sc.collection.objects.link(ob)
    ob.hide_render = True
    return len(V), len(T)

# ------------------------------------------------------------------ Geometry Nodes: Raycasts
def build_nodes():
    ng = bpy.data.node_groups.get("PC_Licht_Raycast")
    if ng: bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new("PC_Licht_Raycast", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    oi = N.new("GeometryNodeObjectInfo"); oi.inputs["Object"].default_value = bpy.data.objects["PC_Verdecker"]
    oi.transform_space = "RELATIVE"
    pos = N.new("GeometryNodeInputPosition")
    na = N.new("GeometryNodeInputNamedAttribute"); na.data_type = "FLOAT_VECTOR"; na.inputs["Name"].default_value = "nrm"
    nrm = na.outputs["Attribute"]

    def vmath(op, a, b=None, scale=None):
        m = N.new("ShaderNodeVectorMath"); m.operation = op
        for i, v in ((0, a), (1, b)):
            if v is None: continue
            if isinstance(v, (tuple, list)): m.inputs[i].default_value = v
            else: L.new(v, m.inputs[i])
        if scale is not None: m.inputs["Scale"].default_value = scale
        return m.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else m.outputs["Vector"]
    def math(op, a, b=None):
        m = N.new("ShaderNodeMath"); m.operation = op
        for i, v in ((0, a), (1, b)):
            if v is None: continue
            if isinstance(v, (int, float)): m.inputs[i].default_value = v
            else: L.new(v, m.inputs[i])
        return m.outputs[0]
    def ray(src, direction, length):
        r = N.new("GeometryNodeRaycast")
        L.new(oi.outputs["Geometry"], r.inputs["Target Geometry"])
        L.new(src, r.inputs["Source Position"])
        if isinstance(direction, (tuple, list)): r.inputs["Ray Direction"].default_value = direction
        else: L.new(direction, r.inputs["Ray Direction"])
        r.inputs["Ray Length"].default_value = length
        return r.outputs["Is Hit"], r.outputs["Hit Distance"]

    src = vmath("ADD", pos.outputs["Position"], vmath("SCALE", nrm, scale=0.025))
    # Sonne
    s_src = vmath("ADD", src, tuple(SUN_B * 0.04))
    hit, _ = ray(s_src, tuple(SUN_B), 400.0)
    sunv = math("SUBTRACT", 1.0, hit)
    # AO (kurz)
    acc = None
    for d in AO_DIRS:
        dv = vmath("NORMALIZE", vmath("ADD", nrm, vmath("SCALE", (d[0], d[1], d[2]), scale=0.85)))
        h, dist = ray(src, dv, 2.0)
        vis = math("SUBTRACT", 1.0, math("MULTIPLY", h, math("SUBTRACT", 1.0, math("DIVIDE", dist, 2.0))))
        acc = vis if acc is None else math("ADD", acc, vis)
    ao = math("DIVIDE", acc, float(len(AO_DIRS)))
    # Himmelssicht (lang)
    acc = None
    for d in SKY_DIRS:
        dv = vmath("NORMALIZE", vmath("ADD", vmath("SCALE", nrm, scale=0.35), (d[0], d[1], d[2])))
        h, dist = ray(src, dv, 60.0)
        vis = math("SUBTRACT", 1.0, math("MULTIPLY", h, math("SUBTRACT", 0.92, math("MULTIPLY", math("DIVIDE", dist, 60.0), 0.6))))
        acc = vis if acc is None else math("ADD", acc, vis)
    sky = math("DIVIDE", acc, float(len(SKY_DIRS)))
    geo = gi.outputs["Geometry"]
    for name, val in (("sunv", sunv), ("ao", ao), ("sky", sky)):
        st = N.new("GeometryNodeStoreNamedAttribute"); st.data_type = "FLOAT"; st.domain = "POINT"
        st.inputs["Name"].default_value = name
        L.new(geo, st.inputs["Geometry"]); L.new(val, st.inputs["Value"])
        geo = st.outputs["Geometry"]
    L.new(geo, go.inputs["Geometry"])
    return ng

# ------------------------------------------------------------------ Kacheln lesen / schreiben
def oct_decode(e):
    e = e.astype(np.float32) / 255.0 * 2 - 1
    n = np.stack([e[:, 0], e[:, 1], 1 - np.abs(e[:, 0]) - np.abs(e[:, 1])], 1)
    neg = n[:, 2] < 0
    nx = np.where(neg, (1 - np.abs(n[:, 1])) * np.sign(n[:, 0] + 1e-12), n[:, 0])
    ny = np.where(neg, (1 - np.abs(n[:, 0])) * np.sign(n[:, 1] + 1e-12), n[:, 1])
    n[:, 0], n[:, 1] = nx, ny
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

def read_part(t, which, quant):
    n = t["na"] if which == "a" else t["n"] - t["na"]
    if n <= 0: return None
    fn = os.path.join(PCD, "t_%d_%d_%s.bin" % (t["i"], t["j"], which))
    raw = np.fromfile(fn, np.uint8)
    q = raw[:n * 6].view(np.int16).reshape(n, 3).astype(np.float32)
    col = raw[n * 6:n * 10].reshape(n, 4).copy()
    octn = raw[n * 10:n * 12].reshape(n, 2)
    P = np.stack([q[:, 0] * quant[0] + t["cx"], q[:, 1] * quant[1], q[:, 2] * quant[2] + t["cz"]], 1)
    return dict(fn=fn, n=n, raw=raw, P=P, col=col, nrm=oct_decode(octn))

def srgb2lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin2srgb(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)

def shade(part, sunv, ao, sky):
    col = part["col"]; flags = col[:, 3]
    two = (flags & 1) > 0; emis = (flags & 2) > 0
    n = part["nrm"]
    a = srgb2lin(col[:, :3].astype(np.float32) / 255.0)
    ndl = n @ SUN_G
    ndl = np.where(two, np.abs(ndl), np.maximum(ndl, 0))
    up = n[:, 1] * 0.5 + 0.5
    skyl = (sky * (0.55 + 0.45 * up) * (0.35 + 0.65 * ao))[:, None] * SKY_COL
    sunl = (sunv * ndl * (0.55 + 0.45 * ao))[:, None] * SUN_COL
    light = skyl + sunl + BOUNCE * ao[:, None]
    out = a * light
    out = out / (1.0 + 0.35 * out)                       # weiche Schulter, damit helle Flächen nicht ausbrennen
    out[emis] = a[emis] * 1.15
    col[:, :3] = np.clip(np.round(lin2srgb(out) * 255), 0, 255).astype(np.uint8)
    raw = part["raw"]; n_ = part["n"]
    raw[n_ * 6:n_ * 10] = col.ravel()
    raw.tofile(part["fn"])

# ------------------------------------------------------------------ Ablauf per Timer
ST = {"phase": "idle", "files": [], "batches": [], "done_files": 0, "done_batches": 0, "t0": 0, "err": [], "info": {}}
def progress():
    json.dump({k: v for k, v in ST.items() if k not in ("files", "batches", "idx")} | {"files_left": len(ST["files"]), "batches_left": len(ST["batches"]),
               "elapsed": round(time.time() - ST["t0"], 1)}, open(PROGRESS, "w"), indent=1)

def plan_batches():
    idx = json.load(open(os.path.join(PCD, "index.json")))
    ST["idx"] = idx
    parts = [(t, w) for t in idx["tiles"] for w in ("a", "b")]
    batches, cur, cnt = [], [], 0
    for t, w in parts:
        n = t["na"] if w == "a" else t["n"] - t["na"]
        if n <= 0: continue
        cur.append((t, w)); cnt += n
        if cnt >= BATCH: batches.append(cur); cur, cnt = [], 0
    if cur: batches.append(cur)
    return batches

def run_batch(batch):
    quant = ST["idx"]["quant"]
    parts = [p for p in (read_part(t, w, quant) for t, w in batch) if p]
    P = np.concatenate([p["P"] for p in parts]); Nn = np.concatenate([p["nrm"] for p in parts])
    PB = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1).astype(np.float32)
    NB = np.stack([Nn[:, 0], -Nn[:, 2], Nn[:, 1]], 1).astype(np.float32)
    sc = scene()
    ob = bpy.data.objects.get("PC_Stapel")
    if ob:
        me_old = ob.data; bpy.data.objects.remove(ob, do_unlink=True); bpy.data.meshes.remove(me_old)
    me = bpy.data.meshes.new("PC_Stapel")
    me.vertices.add(len(PB)); me.vertices.foreach_set("co", PB.ravel())
    at = me.attributes.new("nrm", "FLOAT_VECTOR", "POINT"); at.data.foreach_set("vector", NB.ravel())
    ob = bpy.data.objects.new("PC_Stapel", me); sc.collection.objects.link(ob)
    mod = ob.modifiers.new("Licht", "NODES"); mod.node_group = bpy.data.node_groups["PC_Licht_Raycast"]
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg); em = ev.data
    res = {}
    for k in ("sunv", "ao", "sky"):
        a = np.empty(len(PB), np.float32); em.attributes[k].data.foreach_get("value", a); res[k] = a
    o = 0
    for p in parts:
        sl = slice(o, o + p["n"]); o += p["n"]
        shade(p, res["sunv"][sl], res["ao"][sl], res["sky"][sl])
    ST["info"]["last_batch_pts"] = int(len(PB))
    ST["info"]["sun_lit_share"] = round(float(res["sunv"].mean()), 3)
    ST["info"]["ao_mean"] = round(float(res["ao"].mean()), 3)
    ST["info"]["sky_mean"] = round(float(res["sky"].mean()), 3)

def _tick():
    try:
        if ST["phase"] == "occ":
            if ST["files"]:
                add_occluder_file(ST["files"].pop(0)); ST["done_files"] += 1
            else:
                nv, nt = finish_occluder(); ST["info"]["occluder"] = [nv, nt]
                build_nodes()
                ST["batches"] = plan_batches(); ST["phase"] = "bake"
        elif ST["phase"] == "bake":
            if ST["batches"]:
                run_batch(ST["batches"].pop(0)); ST["done_batches"] += 1
            else:
                idx = ST["idx"]
                idx["baked"] = {"by": "pipeline/pointcloud/bl_light.py (Blender Geometry Nodes Raycast)", "sun": SUN_G.round(4).tolist(),
                                "rays": {"sun": 1, "ao": len(AO_DIRS), "sky": len(SKY_DIRS)}, "date": time.strftime("%Y-%m-%d %H:%M")}
                json.dump(idx, open(os.path.join(PCD, "index.json"), "w"), separators=(",", ":"))
                ST["phase"] = "done"; progress(); return None
    except Exception as e:
        ST["err"].append([ST["phase"], repr(e), traceback.format_exc()[-900:]])
        if len(ST["err"]) > 20: ST["phase"] = "failed"; progress(); return None
    progress()
    return 0.02

def start():
    man = json.load(open(os.path.join(CITY, "manifest.json"), encoding="utf-8"))
    files = [f["url"] for f in man["files"] if f.get("kind") in ("visual", "both") and not RX_SKIP_FILE.search(f["url"])]
    OCC["V"].clear(); OCC["T"].clear(); OCC["nv"] = 0
    ST.update(phase="occ", files=files, batches=[], done_files=0, done_batches=0, t0=time.time(), err=[], info={})
    scene()
    if bpy.app.timers.is_registered(_tick): bpy.app.timers.unregister(_tick)
    bpy.app.timers.register(_tick, first_interval=0.1)
    progress()
    return len(files)

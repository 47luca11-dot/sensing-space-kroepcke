# Figur mit Fingern aus einem Gaussian-Splat-Scan (ohne KI-Mesh): echte Scan-Splats + geschlossener Körper aus den Splats
# + menschliches Skelett (23 Spiel-Knochen + Rigify-Hände: Handfläche, Daumen, 4 Finger à 3 Glieder).
#
#   headless:  Blender -b --factory-startup --python figur_finger.py -- --id <id> --quelle <person.ply> --out <ordner>
#   Live-Blender (Blender-MCP): sys.argv setzen und die Datei per exec ausführen – baut in einer eigenen Szene "Figur_<id>",
#   die offene Datei des Nutzers wird nicht zurückgesetzt oder überschrieben (.blend nur als Kopie).
#
#   1) Körper: Splats -> Voxel (6 mm) -> 1,2 cm schließen -> innen füllen -> Fläche -> Remesh + Glätten (geschlossen, Finger bleiben getrennt)
#   2) Skelett an Körper + Fingerspitzen anpassen (skelett.py), 3) Füll-Mesh + Heat-Gewichte + Jacken-Regel + Hand-Gewichte je Finger
#   4) Scan-Splats auf die Körperfläche legen, Lücken/Hände mit neuen Splats füllen
#   5) Finger entspannen (nebeneinander, leicht gekrümmt) und als Ruhepose übernehmen -> im Spiel nie gespreizt
#   6) Carlos-Animationen, 7) GLB + Splats + Skin (GLB-Gelenkreihenfolge), 8) Testbilder
import bpy, bmesh, json, sys, math, struct, os
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

HIER = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.normpath(os.path.join(HIER, "..", "..", "..", ".."))
sys.path.insert(0, HIER)
import splat_werkzeug as SW
import skelett as SK

argv = sys.argv[sys.argv.index("--") + 1:]
def arg(n, std=None): return argv[argv.index(n) + 1] if n in argv else std
ID, FARBE, OUT = arg("--id"), arg("--quelle"), arg("--out")
VOX_K = float(arg("--voxel", 0.006)); SCHLIESSEN = int(arg("--schliessen", 2))
CARLOS = arg("--carlos", os.path.join(REPO, "startbildschirm", "assets", "person1.glb"))
MENSCH = json.load(open(os.path.join(HIER, "mensch_skelett.json")))["bones"]
CT = json.load(open(os.path.join(HIER, "carlos_skelett.json")))["bones"]
DREIECKE = int(arg("--dreiecke", 80000))
ARM = ["shoulder.L", "upper_arm.L", "forearm.L", "hand.L", "shoulder.R", "upper_arm.R", "forearm.R", "hand.R"]
assert ID and FARBE and OUT
os.makedirs(f"{OUT}/_test", exist_ok=True)
rep = dict(id=ID, art="splats_finger"); rng = np.random.default_rng(7)
import time as _t; _T0 = _t.time()
def log(*a): print(f"[{_t.time() - _T0:6.0f}s]", *a, flush=True)

if bpy.app.background:
    bpy.ops.wm.read_factory_settings(use_empty=True); scene = bpy.context.scene
else:                                                         # Live-Blender: eigene Szene, nichts vom Nutzer anfassen
    old = bpy.data.scenes.get("Figur_" + ID)
    if old:
        for o in list(old.objects): bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.scenes.remove(old)
    scene = bpy.data.scenes.new("Figur_" + ID); bpy.context.window.scene = scene
    # Reste früherer Läufe dieser Figur (Aktionen, Meshes, Rig) entfernen – sonst landen alte Clips mit in der GLB
    for act in [x for x in bpy.data.actions if x.name.startswith(ID + "_") or x.name.startswith("person1_")]: bpy.data.actions.remove(act)
    for coll in (bpy.data.meshes, bpy.data.armatures):
        for x in [x for x in coll if x.name.startswith(ID + "_") and x.users == 0]: coll.remove(x)

def aktiv(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[-1]

def zahlen(me):
    bm = bmesh.new(); bm.from_mesh(me)
    r = dict(punkte=len(bm.verts), dreiecke=sum(len(f.verts) - 2 for f in bm.faces), offene_kanten=sum(e.is_boundary for e in bm.edges),
             nicht_mannigfaltig=sum((not e.is_manifold) and not e.is_boundary for e in bm.edges))
    bm.free(); return r

def flicken(bm):
    for _ in range(4):
        bad = [e for e in bm.edges if not e.is_manifold and not e.is_boundary]
        if bad:
            bmesh.ops.delete(bm, geom=list({f for e in bad for f in e.link_faces}), context='FACES')
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        bnd = [e for e in bm.edges if e.is_boundary]
        if not bnd and not bad: break
        bmesh.ops.holes_fill(bm, edges=bnd, sides=0); bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

def verts(ob):
    V = np.zeros(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get("co", V); return V.reshape(-1, 3)

# ======================================================================= 1) Körper aus den Splats
log("1 Körper aus den Splats")
fprops, fa = SW.lesen(FARBE); FP = SW.pos(fprops, fa); FOP = SW.deckkraft(fprops, fa); FC = SW.farbe(fprops, fa); FG = SW.groesse(fprops, fa)
H = float(np.percentile(FP[FOP > 0.5, 2], 99.98))
VOX, K = VOX_K, SCHLIESSEN
Q = FP[(FOP > 0.25) & (FG < 0.03)]
lo = Q.min(0) - (K + 4) * VOX
ijk = np.floor((Q - lo) / VOX).astype(np.int64); dims = ijk.max(0) + K + 6
occ = np.zeros(dims, bool); occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
def weiten(g, k_):
    g = g.copy()
    for ax in range(3):
        h_ = g.copy()
        for s_ in range(1, k_ + 1): h_ |= np.roll(g, s_, ax) | np.roll(g, -s_, ax)
        g = h_
    return g
# Arme: nackte Unterarme sind im Scan oft nur dünn mit Splats belegt (bv10 links ~500 Splats auf 10 cm) -> die feine Hülle reißt am
# Unterarm ab und die Hand fällt als Schwebeteil weg. Dort zusätzlich grob schließen (doppelte Voxelgröße, 3 Voxel = 2,4 cm) –
# nur seitlich neben dem Rumpf (|x| > 22 cm), damit Rumpf/Gesicht fein bleiben.
_Qa = FP[(FOP > 0.1) & (FG < 0.04) & (FP[:, 2] > 0.09)]                 # über dem Knöchel: Boden-Reflexe nicht mit einschließen
if len(_Qa):
    _ij = np.floor((_Qa - lo) / (2 * VOX)).astype(np.int64); _ok = np.all((_ij >= 0) & (_ij < (dims + 1) // 2), 1); _ij = _ij[_ok]
    _g = np.zeros((dims + 1) // 2, bool); _g[_ij[:, 0], _ij[:, 1], _ij[:, 2]] = True
    _g = ~weiten(~weiten(_g, 3), 3)                                  # schließen (weiten + schrumpfen)
    _f = np.repeat(np.repeat(np.repeat(_g, 2, 0), 2, 1), 2, 2)[:dims[0], :dims[1], :dims[2]]
    _xs = (np.arange(dims[0]) + 0.5) * VOX + lo[0]
    if "--skelett_von" in argv:                                       # nicht in der Achsel: dort verschmolz die grobe Hülle Arm und Rumpf
        _zs = (np.arange(dims[2]) + 0.5) * VOX + lo[2]; _kk = H / 1.8
        _ax = (np.abs(_xs) > 0.13 * _kk) & (np.abs(_xs) < 0.27 * _kk); _az = (_zs > 0.55 * H) & (_zs < 0.80 * H)
        _f[np.ix_(_ax, np.ones(dims[1], bool), _az)] = False
        _f[:, :, _zs < 0.52 * H] = False                                 # nicht an den Beinen: zwischen den Beinen entstand sonst eine Brücke
    log(f"  grob geschlossen (Arme, dünne Stellen): +{int((_f & ~occ).sum())} Voxel")
    occ |= _f
dick = weiten(occ, K)
aussen = np.zeros(dims, bool)
aussen[0, :, :] = aussen[-1, :, :] = aussen[:, 0, :] = aussen[:, -1, :] = aussen[:, :, 0] = aussen[:, :, -1] = True
aussen &= ~dick
while True:
    n_ = aussen.copy()
    for ax in range(3): n_ |= np.roll(aussen, 1, ax) | np.roll(aussen, -1, ax)
    n_ &= ~dick
    if n_.sum() == aussen.sum(): break
    aussen = n_
voll = ~weiten(aussen, K); voll |= occ
quads = []
for ax in range(3):
    for d_ in (1, -1):
        rand = voll & ~np.roll(voll, -d_, ax); c_ = np.argwhere(rand)
        if not len(c_): continue
        base = c_.copy(); base[:, ax] += (1 if d_ > 0 else 0)
        u_, w_ = (ax + 1) % 3, (ax + 2) % 3
        e_u = np.zeros(3, np.int64); e_u[u_] = 1; e_w = np.zeros(3, np.int64); e_w[w_] = 1
        q_ = np.stack([base, base + e_u, base + e_u + e_w, base + e_w], 1)
        if d_ < 0: q_ = q_[:, ::-1]
        quads.append(q_)
q_ = np.concatenate(quads, 0); flat = np.ravel_multi_index(q_.reshape(-1, 3).T, dims + 1)
uq, inv = np.unique(flat, return_inverse=True)
vv = np.stack(np.unravel_index(uq, dims + 1), 1) * VOX + lo
me0 = bpy.data.meshes.new(ID + "_koerper"); me0.from_pydata(vv.tolist(), [], inv.reshape(-1, 4).tolist())
hi = bpy.data.objects.new(ID + "_koerper", me0); scene.collection.objects.link(hi); aktiv(hi)
rm = hi.modifiers.new("remesh", 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = 0.004; rm.adaptivity = 0
bpy.ops.object.modifier_apply(modifier="remesh")
sm = hi.modifiers.new("glatt", 'LAPLACIANSMOOTH'); sm.lambda_factor = 0.6; sm.iterations = 10; sm.use_volume_preserve = True
bpy.ops.object.modifier_apply(modifier="glatt")
bm = bmesh.new(); bm.from_mesh(hi.data)
bm.verts.ensure_lookup_table()
# größte zusammenhängende Insel behalten (Schwebeteile weg)
seen = set(); isl = []
for v0 in bm.verts:
    if v0.index in seen: continue
    st = [v0]; comp = []; seen.add(v0.index)
    while st:
        v1 = st.pop(); comp.append(v1)
        for e in v1.link_edges:
            o_ = e.other_vert(v1)
            if o_.index not in seen: seen.add(o_.index); st.append(o_)
    isl.append(comp)
isl.sort(key=len, reverse=True)
bmesh.ops.delete(bm, geom=[v for i_ in isl[1:] for v in i_], context='VERTS')
flicken(bm); bm.to_mesh(hi.data); bm.free()
V = verts(hi); H = float(V[:, 2].max())
rep["koerper"] = dict(zahlen(hi.data), groesse_m=round(H, 3), voxel_mm=round(VOX * 1000, 1), schwebeteile=len(isl) - 1)
def farbfeld(VOX=0.006, n_aus=16, maske=None):
    gut = (FOP > 0.3) if maske is None else ((FOP > 0.3) & maske); Q = FP[gut]; lo = Q.min(0) - (n_aus + 2) * VOX
    dims = np.ceil((Q.max(0) - lo) / VOX).astype(np.int64) + n_aus + 3
    fl = np.ravel_multi_index(np.floor((Q - lo) / VOX).astype(np.int64).T, dims)
    FS = np.zeros((int(np.prod(dims)), 3), np.float32); FW = np.zeros(int(np.prod(dims)), np.float32); w = FOP[gut]
    for ch in range(3): np.add.at(FS[:, ch], fl, FC[gut][:, ch] * w)
    np.add.at(FW, fl, w); FS = FS.reshape(*dims, 3) / np.maximum(FW.reshape(dims), 1e-9)[..., None]; B = FW.reshape(dims) > 0
    for _ in range(n_aus):
        S2 = np.zeros_like(FS); W2 = np.zeros(B.shape, np.float32)
        for ax in range(3):
            for dd in (1, -1): W2 += np.roll(B, dd, ax); S2 += np.roll(FS * B[..., None], dd, ax)
        neu = ~B & (W2 > 0); FS[neu] = S2[neu] / W2[neu][:, None]; B = B | neu
    med = np.median(FC[gut], 0)
    def feld(Pq):
        g = (Pq - lo) / VOX - 0.5; i0 = np.floor(g).astype(np.int64); f = g - i0
        acc = np.zeros((len(Pq), 3)); aw = np.zeros(len(Pq))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ii = np.clip(i0 + [dx, dy, dz], 0, dims - 1)
                    ww = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    ww = ww * B[ii[:, 0], ii[:, 1], ii[:, 2]]; acc += FS[ii[:, 0], ii[:, 1], ii[:, 2]] * ww[:, None]; aw += ww
        out = acc / np.maximum(aw, 1e-9)[:, None]; out[aw < 1e-6] = med; return np.clip(out, 0, 1)
    return feld
feld = farbfeld()
VC = np.ones((len(V), 4), np.float32); VC[:, :3] = feld(V)
hi.data.color_attributes.new("Col", 'FLOAT_COLOR', 'POINT').data.foreach_set("color_srgb", VC.ravel())
log("Körper", rep["koerper"])

# ======================================================================= 2) Skelett
log("2 Skelett")
EK = np.zeros(len(hi.data.edges) * 2, np.int64); hi.data.edges.foreach_get("vertices", EK); EK = EK.reshape(-1, 2)
SK_VON = arg("--skelett_von"); VORLAGE = None
if SK_VON:
    # Proportionen von LUCA 1 (Nutzer: LUCA 1 und 2 sind dieselbe Person, MO hat ähnliche Proportionen): Schulterbreite und
    # Armgliederung relativ zur Körperhöhe, L/R gemittelt. Lage und Richtung der Glieder kommen weiter aus diesem Scan.
    _R9 = json.load(open(os.path.join(REPO, "figuren", SK_VON, "build", SK_VON + "_skelett.json"))); _B9 = _R9["knochen"]; _H9 = _R9["hoehe"]
    _p = lambda n, e: np.array(_B9[n][e]); _ln = lambda n: float(np.linalg.norm(_p(n, "tail") - _p(n, "head")))
    _cx9 = _p("spine.003", "tail")[0]
    VORLAGE = dict(schulter_x=float(np.mean([abs(_p("upper_arm." + s_, "head")[0] - _cx9) for s_ in "LR"])) / _H9,
                   oa=float(np.mean([_ln("upper_arm." + s_) for s_ in "LR"])) / _H9,
                   ua=float(np.mean([_ln("forearm." + s_) for s_ in "LR"])) / _H9,
                   hand_tip=float(np.mean([np.linalg.norm(_p("f_middle.03." + s_, "tail") - _p("hand." + s_, "head")) for s_ in "LR"])) / _H9)
    rep["vorlage"] = {kk: round(v, 4) for kk, v in VORLAGE.items()}; log("Skelett-Proportionen von", SK_VON, rep["vorlage"])
J, ROT, info = SK.anpassen(V, H, MENSCH, CT, log, kanten=EK, vorlage=VORLAGE); info["hi_punkte"] = V
HAENDE_VON = arg("--haende_von"); HT = {}; HT_OFF = {}; HT_EL = {}
def _rot_min(a0, b0):
    v = np.cross(a0, b0); c = float(a0 @ b0); vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / max(1 + c, 1e-6)
if HAENDE_VON:
    # Hände einer Referenzfigur übernehmen (LUCA 2 = dieselbe Person wie LUCA 1; im LUCA-2-Scan sind die Hände verschmolzene
    # Klumpen): Hand-/Fingerknochen + Hand-Splats von LUCA 1, ans Handgelenk und in die Handrichtung dieses Scans gedreht
    _RH = json.load(open(os.path.join(REPO, "figuren", HAENDE_VON, "build", HAENDE_VON + "_skelett.json"))); _BH = _RH["knochen"]; _sH = H / _RH["hoehe"]
    _HT = ("hand", "palm", "f_index", "f_middle", "f_ring", "f_pinky", "thumb")
    _Dh = os.path.join(REPO, "figuren", HAENDE_VON, "build"); _p9h, _a9h = SW.lesen(os.path.join(_Dh, HAENDE_VON + "_splats.ply"))
    _G9 = SW.pos(_p9h, _a9h); _P9h = np.stack([_G9[:, 0], -_G9[:, 2], _G9[:, 1]], 1); del _a9h
    _raw9h = np.fromfile(os.path.join(_Dh, HAENDE_VON + "_skin.bin"), np.uint8).reshape(-1, 8); _j9h = json.load(open(os.path.join(_Dh, HAENDE_VON + "_skin.json")))["joints"]
    for s_ in "LR":
        # Richtung = Unterarmachse (Elle->Handgelenk) – die Hand behält LUCA 1s natürlichen Handgelenkwinkel zum Unterarm
        w9 = np.array(_BH["hand." + s_]["head"]); d9 = w9 - np.array(_BH["forearm." + s_]["head"]); d9 /= np.linalg.norm(d9)
        w0 = np.array(J[("hand." + s_, "head")]); d0 = w0 - np.array(J[("forearm." + s_, "head")]); d0 /= np.linalg.norm(d0)
        # Handgelenk aus der echten Handspitze im Scan: weitester Körperpunkt vom Ellbogen in einem Kegel (40°) um den Unterarm
        # (die Hand ist oft abgeknickt) minus die Reichweite der Referenzhand über ihr Handgelenk hinaus; Ellbogen bleibt
        _kk = H / 1.8; _el = np.array(J[("forearm." + s_, "head")]); _el9 = np.array(_BH["forearm." + s_]["head"])
        _Q = V - _el; _dq = np.linalg.norm(_Q, axis=1)
        for _it in range(3):                                            # Unterarmachse = Schwerpunkt der echten Unterarmpunkte 10-20 cm hinter dem Ellbogen
            _ms = ((_Q @ d0) > 0.64 * _dq) & (_dq > 0.10 * _kk) & (_dq < 0.20 * _kk)
            if _ms.sum() < 30: break
            _dn = V[_ms].mean(0) - _el; d0 = _dn / np.linalg.norm(_dn)
        rep.setdefault("unterarm_richtung", {})[s_] = [round(v, 3) for v in d0]
        _m = ((_Q @ d0) > 0.766 * _dq) & (_dq < 0.60 * _kk); _ua_fin = float(np.linalg.norm(w0 - _el))
        if _m.sum() > 30:
            _tip = float(np.percentile(_dq[_m], 99.7))
            _wh9 = np.zeros(len(_P9h))
            for c_ in range(4): _wh9 += np.isin(_raw9h[:, c_], [i_ for i_, n_ in enumerate(_j9h) if n_.endswith("." + s_) and n_.split(".")[0] in _HT]) * _raw9h[:, 4 + c_] / 255.0
            _reach9 = float(np.percentile(np.linalg.norm(_P9h[_wh9 > 0.5] - _el9, axis=1), 99.7)) - float(np.linalg.norm(w9 - _el9))
            _ua = _tip - _reach9 * _sH
            log(f"  Hand {s_}: Spitze {_tip:.3f} vom Ellbogen, Referenz-Reichweite {_reach9 * _sH:.3f} -> Unterarm {_ua:.3f} (bisher {np.linalg.norm(w0 - _el):.3f})")
            if VORLAGE is not None and _ua < 0.75 * VORLAGE["ua"] * H:     # Hand im Scan zur Faust gekrümmt: Reichweite ~10 cm
                _ua = max(_tip - 0.10 * _kk, 0.65 * VORLAGE["ua"] * H); HT_OFF[s_] = -0.02
                log(f"  Hand {s_}: gekrümmt (Faust) -> Unterarm {_ua:.3f}")
            if VORLAGE is None or 0.6 * VORLAGE["ua"] * H < _ua < 1.25 * VORLAGE["ua"] * H:
                rep.setdefault("handgelenk_verschoben_cm", {})[s_] = round((_ua - float(np.linalg.norm(w0 - _el))) * 100, 1)
                _ua_fin = _ua
        w0 = _el + d0 * _ua_fin; J[("forearm." + s_, "tail")] = Vector(w0); HT_EL[s_] = _el
        R_ = _rot_min(d9, d0)
        for n_ in MENSCH:
            if n_.endswith("." + s_) and n_.split(".")[0] in _HT:
                for e_ in ("head", "tail"): J[(n_, e_)] = Vector(w0 + _sH * R_ @ (np.array(_BH[n_][e_]) - w9))
                ROT[n_] = ("z", R_ @ np.array(_BH[n_]["z"]))
        _dh9 = np.array(_BH["f_middle.01." + s_]["head"]) - w9; _dh9 /= np.linalg.norm(_dh9)
        info["hand_" + s_] = dict(info["hand_" + s_], finger=False, verschmolzen=True, richtung=(R_ @ _dh9).tolist())
        HT[s_] = (w9, w0, R_, d0)
    rep["haende_von"] = HAENDE_VON; log("Hände von", HAENDE_VON, "übernommen (Maßstab", round(_sH, 3), "), Handgelenk verschoben:", rep.get("handgelenk_verschoben_cm"))
rig = SK.rig_bauen(scene, ID + "_rig", J, ROT, MENSCH, CT)
rep["skelett"] = dict(knochen=len(rig.data.bones), **{k_: v for k_, v in info.items() if k_.startswith(("schritt", "schulter", "arm_"))},
                      finger={s: info["hand_" + s]["finger"] for s in ("L", "R")})

# ======================================================================= 3) Füll-Mesh + Gewichte
log("3 Füll-Mesh + Gewichte")
game = hi.copy(); game.data = hi.data.copy(); game.name = game.data.name = ID + "_mesh"; scene.collection.objects.link(game); aktiv(game)
n_tri = sum(len(p.vertices) - 2 for p in game.data.polygons)
if n_tri > DREIECKE:
    m = game.modifiers.new("dec", 'DECIMATE'); m.ratio = DREIECKE / n_tri; m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="dec")
bm = bmesh.new(); bm.from_mesh(game.data); flicken(bm); bm.to_mesh(game.data); bm.free()
bpy.ops.object.shade_smooth()
m = game.modifiers.new("innen", 'DISPLACE'); m.strength = -0.004; m.mid_level = 0; m.direction = 'NORMAL'
bpy.ops.object.modifier_apply(modifier="innen")
game.data.color_attributes.active_color = game.data.color_attributes[0]
aktiv(game, rig); bpy.ops.object.parent_set(type='ARMATURE_AUTO')
bones = [(b.name, b.head_local.copy(), b.tail_local.copy()) for b in rig.data.bones]
def segdist(p, a, b):
    ab = b - a; t = max(0, min(1, (p - a).dot(ab) / ab.length_squared)); return (a + ab * t - p).length
fb = 0
for v in game.data.vertices:
    if sum(g.weight for g in v.groups) < 0.01:
        n = min(bones, key=lambda B: segdist(v.co, B[1], B[2]))[0]; game.vertex_groups[n].add([v.index], 1.0, 'REPLACE'); fb += 1
aktiv(game); bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
bpy.ops.object.vertex_group_smooth(group_select_mode='ALL', factor=0.5, repeat=3)
bpy.ops.object.mode_set(mode='OBJECT')
# Jacken-Regel: oberhalb der Hüftgelenke bewegt sich Kleidung mit dem Rumpf, nicht mit den Beinen
k = H / 1.8; zh = J[("thigh.L", "head")].z; gs = {g.name: g for g in game.vertex_groups}; geaendert = 0
for v in game.data.vertices:
    f = min(1.0, max(0.0, (v.co.z - (zh - 0.03 * k)) / (0.07 * k)))
    if f <= 0: continue
    weg = 0.0
    for g in list(v.groups):
        if game.vertex_groups[g.group].name.startswith("thigh."):
            weg += g.weight * f; g.weight *= (1 - f)
    if weg > 1e-4: gs["spine"].add([v.index], weg, 'ADD'); geaendert += 1
rep["hand_gewichte"] = SK.hand_gewichte(game, rig, info, k, log)
aktiv(game); bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
bpy.ops.object.vertex_group_limit_total(group_select_mode='ALL', limit=4)
bpy.ops.object.vertex_group_normalize_all(lock_active=False)
bpy.ops.object.mode_set(mode='OBJECT')
rep["fuellmesh"] = dict(zahlen(game.data), nachgetragen=fb, jacke_entkoppelt=geaendert)

# ======================================================================= 4) Splats (gespreizte Ruhepose)
log("4 Splats auf das Mesh legen")
dg = bpy.context.evaluated_depsgraph_get(); bvh = BVHTree.FromObject(hi, dg)
hand = {s: (np.array(info["hand_" + s]["handgelenk"]), np.array(info["hand_" + s]["richtung"])) for s in ("L", "R")}
def in_hand(Pq):
    m = np.zeros(len(Pq), bool)
    for w, d in hand.values(): m |= (((Pq - w) @ d) > -0.01 * k) & (np.linalg.norm(Pq - w, axis=1) < 0.26 * k)
    return m
# Scan-treu: Original-Splats bleiben exakt wie gescannt (Position, Form, Drehung, volle SH-Farbe) – nur Schwebeteile weit weg
# vom Körper fallen raus. Hände werden NICHT ersetzt (die Finger werden später über ihre Knochen entspannt).
SPl = FP.copy(); behalten = np.zeros(len(FP), bool)
for i, p in enumerate(FP):
    if FOP[i] < 0.03: continue
    loc, nrm, _, dist = bvh.find_nearest(Vector(p), 0.06)
    if loc is not None: behalten[i] = True
# Arm-Zugehörigkeit je Splat (Abstand zur Armkette, seitlich außerhalb des Rumpfprofils) – für Aufräumen und Gewichte
def _kette_abstand(Pq, pts_):
    best = np.full(len(Pq), 9.0)
    for a0, a1 in zip(pts_[:-1], pts_[1:]):
        ab = a1 - a0; t_ = np.clip(((Pq - a0) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        best = np.minimum(best, np.linalg.norm(Pq - (a0 + t_[:, None] * ab), axis=1))
    return best
def arm_maske(Pq, cx_, mit_breite=False):
    m_ = np.zeros(len(Pq), bool); tt = np.zeros(len(Pq))
    da = {}
    for s_ in "LR":
        kp = [np.array(J[(n_, e_)]) for n_, e_ in (("upper_arm." + s_, "head"), ("upper_arm." + s_, "tail"), ("forearm." + s_, "tail"), ("f_middle.03." + s_, "tail"))]
        da[s_] = (_kette_abstand(Pq, kp), kp)
    fern = (np.minimum(da["L"][0], da["R"][0]) > 0.10 * k) & (np.abs(Pq[:, 0] - cx_) < 0.26 * k)   # Rumpfprofil ohne Arme: halbe Breite je 2-cm-Höhenband
    zb = np.floor(Pq[:, 2] / 0.02).astype(int); breite = {}
    for b_ in np.unique(zb[fern]):
        xx = np.abs(Pq[fern & (zb == b_), 0] - cx_)
        if len(xx) > 20: breite[b_] = float(np.percentile(xx, 98))
    bw = np.array([breite.get(b_, 0.0) for b_ in zb])
    for s_ in "LR":
        d_, kp = da[s_]; ab = kp[1] - kp[0]; t_ = ((Pq - kp[0]) @ ab) / max(ab @ ab, 1e-9)
        sel = (d_ < 0.12 * k) & (t_ > 0.15) & ((np.abs(Pq[:, 0] - cx_) > bw + 0.004) | (t_ > 1.0)) & (d_ < np.minimum(da["L" if s_ == "R" else "R"][0], 9))
        m_ |= sel
    m_ |= in_hand(Pq) & (np.abs(Pq[:, 0] - cx_) > bw + 0.004)              # ganze Hand (Vorlage-Fingerknochen treffen die Scan-Hand nicht exakt)
    return (m_, bw) if mit_breite else m_
CX = float(J[("spine.002", "tail")].x)
if SK_VON:
    # Aufräumen (nur Löschen, nichts verändern): Boden-Reflexe (gesättigtes Orange außerhalb von Kopf/Händen, Sohlen-Halo,
    # Flecken auf der Jeans), Nadeln, vereinzelte Schwebe-Splats
    _arm_s = arm_maske(FP, CX)
    _mx = FC.max(1); _mn = FC.min(1); _sat = (_mx - _mn) / np.maximum(_mx, 1e-6); _gr = FC[:, 1] / np.maximum(FC[:, 0], 1e-6)
    _kopf = FP[:, 2] > J[("spine.004", "head")].z - 0.02 * k
    _hand = in_hand(FP)
    # heller Saum unter den Haaren am Hinterkopf (Gegenlicht-Halo): hinter dem Atlas zwischen C7 und Ohrhöhe, hell + warm
    _at, _c7 = J[("spine.005", "tail")], J[("spine.004", "head")]
    saum = ((FP[:, 1] > _at.y - 0.01 * k) & (FP[:, 2] > _c7.z - 0.02 * k) & (FP[:, 2] < _at.z + 0.06 * k) & (np.abs(FP[:, 0] - _at.x) < 0.15 * k)
            & (FC.mean(1) > 0.38) & ((FC[:, 0] - FC[:, 2]) > 0.2)) & ("--hinterkopf_saum" in argv)   # nur wo der Scan einen Gegenlicht-Saum hat (bv10)
    orange = ((FC[:, 0] >= FC[:, 1]) & (FC[:, 1] >= FC[:, 2]) & (_sat > 0.6) & (FC[:, 0] > 0.5)
              & ((FC[:, 1] - FC[:, 2]) / np.maximum(FC[:, 0] - FC[:, 2], 1e-6) > 0.15) & ~_kopf & ~_hand)
    orange |= (FP[:, 2] < 0.05 * k) & (_sat > 0.45) & (FC[:, 0] > 0.3) & (_gr < 0.6)
    orange |= (FP[:, 2] < 0.03 * k) & (FC[:, 0] - FC[:, 2] > 0.10) & (_gr < 0.55)          # dunkelrote Boden-Sprenkel unter den Sohlen
    orange |= FP[:, 2] < -0.004                                                             # unter dem Boden
    orange |= (FP[:, 2] < J[("thigh.L", "tail")].z) & (_sat > 0.45) & (FC[:, 0] > 0.35) & (_gr < 0.6) & ~_arm_s & ~_hand
    # Boden-Reflexe auf der Hose (Jeans ist blau/grau): warm-orange Splats zwischen Schuh und Hüfte, nicht an Armen/Händen
    orange |= ((FP[:, 2] > 0.17 * k) & (FP[:, 2] < J[("thigh.L", "head")].z) & (FC[:, 0] > 0.35) & (FC[:, 0] - FC[:, 2] > 0.25)
               & (_gr < 0.75) & ~_arm_s & ~_hand)
    _sc3 = np.exp(fa[:, SW.spalten(fprops, "scale_0", "scale_1", "scale_2")])
    nadel = (_sc3.max(1) / np.maximum(_sc3.min(1), 1e-6) > 12) & (_sc3.max(1) > 0.008)
    rep["scan_entfernt"] = dict(bodenreflex_orange=int((orange & behalten).sum()), nadeln=int((nadel & behalten & ~orange).sum()))
    rep["scan_entfernt"]["saum_hinterkopf"] = int((saum & behalten).sum())
    behalten &= ~orange & ~nadel & ~saum
    _abst = np.full(len(FP), 9.0)                                       # Abstand zur Körperhülle
    for _i in np.where(behalten)[0]:
        _r = bvh.find_nearest(Vector(FP[_i]), 0.06)
        if _r[0] is not None: _abst[_i] = _r[3]
    for _rd in range(2):                                               # Schwebepunkte: deckungsgewichtete Dichte unter Behaltenen
        _idx = np.where(behalten & (FOP > 0.04))[0]; _kd = KDTree(len(_idx))
        for _j, _i in enumerate(_idx): _kd.insert(FP[_i], _j)
        _kd.balance(); _weg = np.zeros(len(_idx), bool)
        for _j, _i in enumerate(_idx):
            if _abst[_i] < 0.012: continue                             # auf der Körperfläche: große, dünn gesäte T-Shirt-Splats bleiben
            _weg[_j] = sum(FOP[_idx[q[1]]] for q in _kd.find_range(FP[_i], 0.015)) < 2.5
        behalten[_idx[_weg]] = False; rep["scan_entfernt"]["vereinzelt_" + str(_rd)] = int(_weg.sum())
    log("Aufgeräumt", rep["scan_entfernt"])
if HAENDE_VON:
    def _hand_reg(Pq):
        m_ = np.zeros(len(Pq), bool)
        for s_, (_w9, _w0, _R, _d0) in HT.items():                   # radial vom Ellbogen: auch abgeknickte Scan-Hände (Faust neben der Achse)
            _q = Pq - HT_EL[s_]; _dq_ = np.linalg.norm(_q, axis=1); _L = float(np.linalg.norm(_w0 - HT_EL[s_]))
            m_ |= (_dq_ > _L + HT_OFF.get(s_, 0.0) * k) & ((_q @ _d0) > 0.34 * _dq_) & (np.linalg.norm(Pq - _w0, axis=1) < 0.24 * k)
            m_ |= (((Pq - _w0) @ _d0) > HT_OFF.get(s_, 0.0) * k) & (np.linalg.norm(Pq - _w0, axis=1) < 0.24 * k)
        return m_
    _hr = _hand_reg(FP); rep.setdefault("scan_entfernt", {})["haende_ersetzt"] = int((behalten & _hr).sum()); behalten &= ~_hr
if SK_VON:
    _am = behalten & arm_maske(FP, CX) & ~in_hand(FP) & (FOP > 0.05)
    _ca = FC[_am]; _mxa = _ca.max(1); _sa = (_mxa - _ca.min(1)) / np.maximum(_mxa, 1e-6)
    _med = np.median(_ca[(_sa < np.percentile(_sa, 60))], 0); _smed = float(np.median(_sa))
    _aus = _sa > _smed + 0.12
    if _aus.any():
        _ix = np.where(_am)[0][_aus]; _neu = 0.4 * FC[_ix] + 0.6 * _med
        fa[_ix[:, None], np.array(SW.spalten(fprops, "f_dc_0", "f_dc_1", "f_dc_2"))[None, :]] = (_neu - 0.5) / SW.SH0; FC[_ix] = _neu
        rep["arm_farbe_beruhigt"] = int(len(_ix))
    # blickabhängige Farbe (SH-Rest) auf der Armhaut dämpfen: trägt im Spiel die orangen Boden-Reflexe von unten
    _rr = np.array([fprops.index(x_) for x_ in fprops if x_.startswith("f_rest_")]); _ia = np.where(_am)[0]
    if len(_rr): fa[_ia[:, None], _rr[None, :]] *= 0.25
if SK_VON and "--hinterkopf_saum" in argv:
    # helle Gegenlicht-Flecken im Haar am Hinterkopf (Haut schimmert durch) zur Haarfarbe ziehen
    _at = J[("spine.005", "tail")]; _hk = behalten & (FP[:, 2] > _at.z - 0.02 * k) & (FP[:, 1] > _at.y - 0.02 * k) & (FOP > 0.05)
    if _hk.sum() > 100:
        _ch = FC[_hk]; _lh = _ch.mean(1); _hm = np.median(_ch[_lh <= np.percentile(_lh, 60)], 0)
        _hell = _lh > max(float(np.percentile(_lh, 60)) + 0.12, 0.38)
        _ix = np.where(_hk)[0][_hell]; _neu = 0.3 * FC[_ix] + 0.7 * _hm
        fa[_ix[:, None], np.array(SW.spalten(fprops, "f_dc_0", "f_dc_1", "f_dc_2"))[None, :]] = (_neu - 0.5) / SW.SH0; FC[_ix] = _neu
        _rr = [fprops.index(x_) for x_ in fprops if x_.startswith("f_rest_")]; fa[_ix[:, None], np.array(_rr)[None, :]] *= 0.3
        rep["hinterkopf_beruhigt"] = int(len(_ix))
if SK_VON: feld = farbfeld(maske=behalten)                             # Füllfarben nur aus behaltenen Splats (keine Boden-Reflexe)
rep["scan_splats"] = dict(gesamt=int(len(FP)), behalten=int(behalten.sum()), unveraendert=True)
# Füll-Splats nur in echten Löchern: Körperpunkte (~5 mm Raster), in deren Umkreis (2,5 cm) kein deckender Scan-Splat liegt
me = hi.data; me.calc_loop_triangles(); nt = len(me.loop_triangles)
tv = np.zeros(nt * 3, np.int64); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
A = V[tv]; area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
def abtasten(tri_maske, dichte):
    sel = np.where(tri_maske)[0]; cnt = np.floor(area[sel] * dichte + rng.random(len(sel))).astype(np.int64)
    pick = np.repeat(sel, cnt); u, w2 = rng.random(len(pick)), rng.random(len(pick)); fl = u + w2 > 1; u[fl], w2[fl] = 1 - u[fl], 1 - w2[fl]
    pts = A[pick, 0] * (1 - u - w2)[:, None] + A[pick, 1] * u[:, None] + A[pick, 2] * w2[:, None]
    nrm = np.cross(A[pick, 1] - A[pick, 0], A[pick, 2] - A[pick, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return pts, nrm
gP, gN = abtasten(np.ones(len(A), bool), 1 / 25e-6)
if HAENDE_VON: _k_ = ~_hand_reg(gP); gP, gN = gP[_k_], gN[_k_]
dicht = behalten & (FOP > 0.45)
kd2 = KDTree(int(dicht.sum()))
for i, p in enumerate(SPl[dicht]): kd2.insert(p, i)
kd2.balance()
_dl = np.array([kd2.find(p)[2] for p in gP]) if len(gP) else np.zeros(0)
frei = _dl > float(arg("--luecke", 0.025))
_kopf_neu = np.zeros(len(gP), bool)
if SK_VON:                                                             # Haar über Ohrhöhe: schon ab 1,2 cm Lücke füllen (Hinterkopf schimmert durch)
    _kopf_neu = (gP[:, 2] > J[("spine.005", "tail")].z) & (_dl > 0.012) & ~frei; frei |= _kopf_neu
GF = None
if SK_VON and len(gP):
    # Arme: nackte Haut ist im Scan nur fleckig belegt (Lücken, dunkle Löcher) -> auf der Scan-eigenen Armform (Hülle, 3 mm nach
    # innen) schon ab 9 mm Lücke füllen; Farbe = Median der umliegenden deckenden Scan-Splats desselben Arms (helle 70 %)
    _armg = arm_maske(gP, CX); _neu = _armg & (_dl > 0.009) & ~frei
    _src = np.where(behalten & (FOP > 0.45) & arm_maske(FP, CX))[0]
    GF = feld(gP)
    if len(_src) > 50:
        _kda = KDTree(len(_src))
        for _j, _i in enumerate(_src): _kda.insert(FP[_i], _j)
        _kda.balance()
        for _j in np.where(_armg & (frei | _neu))[0]:
            nb = [_src[q[1]] for q in _kda.find_n(gP[_j], 16)]
            cc = FC[nb]; lum = cc.mean(1); cc = cc[lum >= np.percentile(lum, 30)]
            GF[_j] = np.median(cc, 0)
    gN[_armg] *= 1.0; gP = gP - gN * np.where(_armg, 0.003, 0.0)[:, None]
    frei |= _neu; rep["arm_fuellung"] = int((_armg & frei).sum())
    _dsrc = np.where(dicht)[0]
    for _j in np.where(_kopf_neu)[0]:                                  # Haarfarbe aus den nächsten echten Haar-Splats (dunkle 60 %)
        _nb = [_dsrc[q[1]] for q in kd2.find_n(gP[_j], 12)]; _cc = FC[_nb]; _lm = _cc.mean(1)
        GF[_j] = np.median(_cc[_lm <= np.percentile(_lm, 60)], 0)
    GF = GF[frei]
gP, gN = gP[frei], gN[frei]
hP, hN = gP[:0], gN[:0]
rep["neue_splats"] = dict(haende=0, luecken=int(len(gP)))
def neue(pts, nrm, gr, dick, farbe=None):
    b = np.zeros((len(pts), len(fprops)))
    b[:, SW.spalten(fprops, "x", "y", "z")] = pts
    b[:, SW.spalten(fprops, "f_dc_0", "f_dc_1", "f_dc_2")] = ((feld(pts) if farbe is None else farbe) - 0.5) / SW.SH0
    b[:, fprops.index("opacity")] = math.log(0.95 / 0.05)
    b[:, SW.spalten(fprops, "scale_0", "scale_1")] = math.log(gr); b[:, fprops.index("scale_2")] = math.log(dick)
    zax = np.array([0, 0, 1.0]); ax = np.cross(np.broadcast_to(zax, nrm.shape), nrm); sn = np.linalg.norm(ax, axis=1)
    ang = np.arctan2(sn, nrm[:, 2]); ax = np.where(sn[:, None] > 1e-8, ax / (sn[:, None] + 1e-12), np.array([1.0, 0, 0]))
    b[:, SW.spalten(fprops, "rot_0", "rot_1", "rot_2", "rot_3")] = np.concatenate([np.cos(ang / 2)[:, None], ax * np.sin(ang / 2)[:, None]], 1)
    return b
sa = fa[behalten].copy()
a = np.concatenate([sa, neue(gP, gN, 0.0035, 0.001, GF)], 0)
parts = [int(len(sa)), 0, int(len(gP))]
# Gewichte per Data-Transfer vom Füll-Mesh (noch gespreizt)
joints = list(MENSCH.keys()); jpos = {n: i for i, n in enumerate(joints)}
def gewichte(PTS):
    me_ = bpy.data.meshes.new("wpts"); me_.vertices.add(len(PTS)); me_.vertices.foreach_set("co", np.asarray(PTS, np.float32).ravel())
    ob_ = bpy.data.objects.new("wpts", me_); scene.collection.objects.link(ob_)
    for n_ in joints: ob_.vertex_groups.new(name=n_)
    dt = ob_.modifiers.new("wt", 'DATA_TRANSFER'); dt.object = game; dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}; dt.vert_mapping = 'POLYINTERP_NEAREST'; dt.layers_vgroup_select_src = 'ALL'; dt.layers_vgroup_select_dst = 'NAME'
    aktiv(ob_); bpy.ops.object.modifier_apply(modifier="wt")
    gm = {g.index: jpos[g.name] for g in ob_.vertex_groups}; out = np.zeros((len(PTS), len(joints)), np.float32)
    for v in me_.vertices:
        for g in v.groups: out[v.index, gm[g.group]] = g.weight
    bpy.data.objects.remove(ob_, do_unlink=True)
    s = out.sum(1, keepdims=True); return out / np.maximum(s, 1e-6)
W = gewichte(SW.pos(fprops, a))
if SK_VON:
    # Arm und Rumpf nie über den Achselspalt mischen (die Heat-Gewichte der geschlossenen Hülle zogen Rumpf-Splats mit dem Arm
    # mit -> "Arm im Oberkörper verwachsen"): Arm-Splats ohne Rumpfgewichte, Rumpf-Splats unter der Schulter ohne Armgewichte
    _PA = SW.pos(fprops, a); _arm, _bw = arm_maske(_PA, CX, True); _getrennt = 0
    for s_ in "LR":
        _armc = [jpos[n_] for n_ in joints if n_.endswith("." + s_) and not n_.startswith(("thigh", "shin", "foot", "toe", "shoulder"))]
        _rumpfc = [jpos[n_] for n_ in joints if n_.startswith("spine")] + [jpos["shoulder." + s_]]
        h_ = np.array(J[("upper_arm." + s_, "head")])
        seite = (np.sign(_PA[:, 0] - CX) == (1 if s_ == "L" else -1))
        armm = _arm & seite
        W[np.ix_(armm, _rumpfc)] = 0.0
        rumpfm = (~_arm & seite & (_PA[:, 2] < h_[2] - 0.04 * k) & (_PA[:, 2] > J[("thigh.L", "head")].z)
                  & (np.abs(_PA[:, 0] - CX) < _bw + 0.01 * k))                 # nur im Rumpfprofil (nicht Hand/Unterarm daneben)
        W[np.ix_(rumpfm, _armc)] = 0.0; _getrennt += int(armm.sum() + rumpfm.sum())
        _leer = W.sum(1) < 1e-4
        if _leer.any():                                                 # nichts übrig: an den nächsten passenden Knochen
            for i_ in np.where(_leer & armm)[0]:
                W[i_, jpos[min(("upper_arm." + s_, "forearm." + s_, "hand." + s_), key=lambda n_: segdist(Vector(_PA[i_]), J[(n_, "head")], J[(n_, "tail")]))]] = 1.0
            W[_leer & rumpfm, jpos["spine.002"]] = 1.0
    W /= np.maximum(W.sum(1, keepdims=True), 1e-6); rep["arm_rumpf_getrennt"] = _getrennt
if HAENDE_VON:
    _D = os.path.join(REPO, "figuren", HAENDE_VON, "build"); _p9, _a9 = SW.lesen(os.path.join(_D, HAENDE_VON + "_splats.ply"))
    assert list(_p9) == list(fprops), "Splat-Spalten verschieden"
    _R_exp = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]); SW.transformieren(_p9, _a9, _R_exp.T, np.zeros(3), 1.0)   # glTF -> Blender
    _raw = np.fromfile(os.path.join(_D, HAENDE_VON + "_skin.bin"), np.uint8).reshape(-1, 8); _j9 = json.load(open(os.path.join(_D, HAENDE_VON + "_skin.json")))["joints"]
    _P9 = SW.pos(_p9, _a9); _neu_a, _neu_W = [], []
    for s_, (_w9, _w0, _R, _d0) in HT.items():
        _hb = [i_ for i_, n_ in enumerate(_j9) if n_.endswith("." + s_) and n_.split(".")[0] in _HT]
        _wh = np.zeros(len(_P9))
        for c_ in range(4): _wh += np.isin(_raw[:, c_], _hb) * _raw[:, 4 + c_] / 255.0
        _proj = (_P9 - _w9) @ (np.array(_BH["f_middle.01." + s_]["head"]) - _w9) / np.linalg.norm(np.array(_BH["f_middle.01." + s_]["head"]) - _w9)
        _lum9 = SW.farbe(_p9, _a9).mean(1)
        _sel = np.where((_wh > 0.5) & (_proj > -0.01) & ~((_proj < 0.045) & (_lum9 < 0.3)))[0]; _as = _a9[_sel].copy()   # ohne Jackenbund
        SW.transformieren(_p9, _as, _R, np.zeros(3), _sH)
        _as[:, SW.spalten(fprops, "x", "y", "z")] += _w0 - _sH * _R @ _w9
        _Wn = np.zeros((len(_sel), len(joints)), np.float32)
        for c_ in range(4):
            for r_, i_ in enumerate(_sel): _Wn[r_, jpos[_j9[_raw[i_, c_]]]] += _raw[i_, 4 + c_] / 255.0
        _elA = HT_EL[s_]; _qa = FP - _elA; _ta = _qa @ _d0; _La = float(np.linalg.norm(_w0 - _elA))
        _fa = behalten & (FOP > 0.3) & (_ta > 0.35 * _La) & (_ta < _La - 0.01) & (np.linalg.norm(_qa - np.outer(_ta, _d0), axis=1) < 0.06)
        if _fa.sum() > 30:                                             # Hautton der Referenzhand an den Unterarm dieses Scans angleichen
            _cf = FC[_fa]; _lf = _cf.mean(1); _ziel = np.median(_cf[_lf >= np.percentile(_lf, 30)], 0)
            _ch = SW.farbe(fprops, _as); _lh = _ch.mean(1); _quelle = np.median(_ch[_lh >= np.percentile(_lh, 30)], 0)
            _fak = np.clip(_ziel / np.maximum(_quelle, 1e-3), 0.8, 1.25); _fak = 1 + 0.7 * (_fak - 1)   # sanft
            _dc = SW.spalten(fprops, "f_dc_0", "f_dc_1", "f_dc_2"); _as[:, _dc] = (np.clip(_ch * _fak, 0, 1) - 0.5) / SW.SH0
            for _c in range(3):
                _rc = [fprops.index(f"f_rest_{i_}") for i_ in range(_c * 15, _c * 15 + 15) if f"f_rest_{i_}" in fprops]
                if _rc: _as[:, _rc] *= _fak[_c]
            rep.setdefault("hand_farbangleich", {})[s_] = [round(float(v), 2) for v in _fak]
        _neu_a.append(_as); _neu_W.append(_Wn / np.maximum(_Wn.sum(1, keepdims=True), 1e-6))
        # Handgelenk-Brücke (selbst modelliert): vom Ende des Scan-Unterarms bis dort, wo die Haut der Referenzhand beginnt –
        # sich verjüngender Hautzylinder in der Hautfarbe des Scan-Unterarms, Gewichte weich Unterarm -> Hand
        _el = HT_EL[s_]; _L = float(np.linalg.norm(_w0 - _el)); _t0 = _L + min(HT_OFF.get(s_, 0.0), 0.0) * k
        _t1 = _L + max(float(np.percentile(_proj[_sel], 3)), 0.0) * _sH
        _armk = behalten & (FOP > 0.3) & (np.abs(((FP - _el) @ _d0) - (_t0 - 0.03)) < 0.03) & (np.linalg.norm((FP - _el) - np.outer((FP - _el) @ _d0, _d0), axis=1) < 0.06)
        if _t1 > _t0 - 0.01 and _armk.sum() > 20:
            _ax = _el + np.outer(np.array([_t0 - 0.02, _t1 + 0.01]), _d0)
            _r0 = float(np.clip(np.median(np.linalg.norm((FP[_armk] - _el) - np.outer((FP[_armk] - _el) @ _d0, _d0), axis=1)), 0.022 * k, 0.032 * k))
            _hs = _as[:, SW.spalten(fprops, "x", "y", "z")]; _ph = (_hs - _w0) @ _d0
            _hn = _hs[np.abs(_ph - (_t1 - _L)) < 0.015]
            _r1 = float(np.clip(np.median(np.linalg.norm((_hn - _w0) - np.outer((_hn - _w0) @ _d0, _d0), axis=1)), 0.020 * k, 0.030 * k)) if len(_hn) > 20 else 0.9 * _r0
            _e1 = np.cross(_d0, [0, 0, 1.0]); _e1 /= np.linalg.norm(_e1); _e2 = np.cross(_d0, _e1)
            _n = int((_t1 - _t0 + 0.03) / 0.003) * 48; _tt = rng.uniform(_t0 - 0.02, _t1 + 0.01, _n); _ph_ = rng.uniform(0, 2 * np.pi, _n)
            _u = np.clip((_tt - (_t0 - 0.02)) / max(_t1 - _t0 + 0.03, 1e-6), 0, 1); _rr = _r0 * (1 - _u) + _r1 * _u
            _nr = np.cos(_ph_)[:, None] * _e1 + np.sin(_ph_)[:, None] * _e2
            _pts = _el + np.outer(_tt, _d0) + _rr[:, None] * _nr
            _cc = FC[_armk]; _lm = _cc.mean(1); _cc = _cc[(_lm >= np.percentile(_lm, 30)) & (_lm <= np.percentile(_lm, 95))]
            _br = neue(_pts, _nr, 0.0035, 0.0012, _cc[rng.integers(0, len(_cc), _n)])     # echte Hautfarben des Unterarms (Textur)
            _Wb = np.zeros((_n, len(joints)), np.float32); _wu = np.clip((_tt - _L) / 0.03 + 0.5, 0, 1)
            _Wb[:, jpos["forearm." + s_]] = 1 - _wu; _Wb[:, jpos["hand." + s_]] = _wu
            _neu_a.append(_br); _neu_W.append(_Wb); rep.setdefault("handgelenk_bruecke", {})[s_] = dict(splats=_n, von=round(_t0, 3), bis=round(_t1, 3))
    a = np.concatenate([a] + _neu_a, 0); W = np.concatenate([W] + _neu_W, 0); parts[1] = int(sum(len(x_) for x_ in _neu_a))
    rep["neue_splats"]["haende"] = parts[1]
log("Splats", rep["scan_splats"], rep["neue_splats"])

# ======================================================================= 5) Finger entspannen -> neue Ruhepose
log("5 Finger entspannen")
aktiv(rig); bpy.ops.object.mode_set(mode='POSE')
SK.finger_entspannen(rig, info)
mats = SK.knochen_matrizen(rig, joints)
Pn = SK.lbs(SW.pos(fprops, a), W, mats)
dom = W.argmax(1); rc = SW.spalten(fprops, "rot_0", "rot_1", "rot_2", "rot_3")
for b_i, M4 in enumerate(mats):
    if np.allclose(M4, np.eye(4), atol=1e-6): continue
    sel = dom == b_i
    if not sel.any(): continue
    q = np.array(Matrix(M4.tolist()).to_quaternion()); qq = a[sel][:, rc]; qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    a[np.where(sel)[0][:, None], np.array(rc)[None, :]] = SW.quat_mul(np.broadcast_to(q, qq.shape), qq)
a[:, SW.spalten(fprops, "x", "y", "z")] = Pn
bpy.ops.object.mode_set(mode='OBJECT')
aktiv(game); arm_mod = [mm for mm in game.modifiers if mm.type == 'ARMATURE'][0]
bpy.ops.object.modifier_apply(modifier=arm_mod.name)
aktiv(rig); bpy.ops.object.mode_set(mode='POSE'); bpy.ops.pose.armature_apply(selected=False); bpy.ops.object.mode_set(mode='OBJECT')
am = game.modifiers.new("Armature", 'ARMATURE'); am.object = rig
rep["finger_entspannt"] = True

# ======================================================================= 6) Animationen von Carlos
log("6 Animationen")
before, acts_before = set(bpy.data.objects), set(bpy.data.actions)
bpy.ops.import_scene.gltf(filepath=CARLOS)
c_objs = [o for o in bpy.data.objects if o not in before]; c_arm = [o for o in c_objs if o.type == 'ARMATURE'][0]
ratio = H / max((c_arm.matrix_world @ b.tail_local).z for b in c_arm.data.bones)
def channelbag(act):
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags: return cb
names = []; c_arm.animation_data_create(); rig.animation_data_create()
for act in [x for x in bpy.data.actions if x not in acts_before]:
    new = act.copy(); new.name = act.name.replace("person1_", ID + "_"); new.use_fake_user = True; cb = channelbag(new)
    for fc in list(cb.fcurves):
        p = fc.data_path; bone = p.split('"')[1] if '"' in p else ""
        if bone in ARM or p.endswith(".scale") or (p.endswith(".location") and bone != "spine"): cb.fcurves.remove(fc)
        elif p.endswith(".location"):
            for kp in fc.keyframe_points: kp.co.y *= ratio; kp.handle_left.y *= ratio; kp.handle_right.y *= ratio
    c_arm.animation_data.action = act
    if c_arm.animation_data.action_slot is None and act.slots: c_arm.animation_data.action_slot = act.slots[0]
    rig.animation_data.action = new
    if rig.animation_data.action_slot is None and new.slots: rig.animation_data.action_slot = new.slots[0]
    cons = []
    for n_ in ARM:
        c_ = rig.pose.bones[n_].constraints.new('COPY_ROTATION'); c_.target = c_arm; c_.subtarget = n_; c_.owner_space = c_.target_space = 'WORLD'; cons.append((n_, c_))
    rows = []
    for f in range(int(act.frame_range[0]), int(act.frame_range[1]) + 1):
        scene.frame_set(f); bpy.context.view_layer.update()
        rows.append((f, {n_: rig.convert_space(pose_bone=rig.pose.bones[n_], matrix=rig.pose.bones[n_].matrix, from_space='POSE', to_space='LOCAL').to_quaternion() for n_ in ARM}))
    for n_, c_ in cons: rig.pose.bones[n_].constraints.remove(c_)
    prev = {}
    for f, qs in rows:
        for n_, q_ in qs.items():
            if n_ in prev and prev[n_].dot(q_) < 0: q_.negate()
            prev[n_] = q_; pb = rig.pose.bones[n_]; pb.rotation_quaternion = q_; pb.keyframe_insert("rotation_quaternion", frame=f)
    names.append(new.name)
for o in c_objs: bpy.data.objects.remove(o, do_unlink=True)
for x_ in [x for x in bpy.data.actions if not x.name.startswith(ID + "_")]: bpy.data.actions.remove(x_)
rig.animation_data.action = None
for pb in rig.pose.bones: pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
rep["animationen"] = sorted(names)

# ======================================================================= 7) Export
log("7 Export")
hi.hide_render = True; hi.hide_set(True)
bpy.ops.wm.save_as_mainfile(filepath=f"{OUT}/{ID}.blend", copy=True)
for _a in [x for x in bpy.data.actions if not x.name.startswith(ID + "_")]: bpy.data.actions.remove(_a)
aktiv(game, rig); glb = f"{OUT}/{ID}.glb"
kw = dict(filepath=glb, use_selection=True, use_active_scene=True, export_format='GLB', export_animations=True, export_animation_mode='ACTIONS',
          export_skins=True, export_force_sampling=True, export_def_bones=False, export_materials='NONE')
try: bpy.ops.export_scene.gltf(**kw, export_vertex_color='ACTIVE')
except TypeError: bpy.ops.export_scene.gltf(**kw, export_colors=True)
g = open(glb, "rb").read(); n = struct.unpack('<I', g[12:16])[0]; gj = json.loads(g[20:20 + n])
glb_joints = [gj['nodes'][i]['name'] for i in gj['skins'][0]['joints']]
rep["glb"] = dict(mb=round(len(g) / 1e6, 1), knochen=len(glb_joints), clips=len(gj['animations']))
props = list(fprops)
SW.transformieren(props, a, np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), np.zeros(3), 1.0)
# volle SH-Farbe behalten (Grad 3 wie im Scan) – früher auf Grad 1 gekürzt
SW.schreiben(f"{OUT}/{ID}_splats.ply", props, a)
W = W[:, [joints.index(n_) for n_ in glb_joints]]
top = np.argsort(-W, axis=1)[:, :4]; tw = np.take_along_axis(W, top, 1); tw = tw / np.maximum(tw.sum(1, keepdims=True), 1e-6)
q8 = np.floor(tw * 255).astype(np.int32); q8[:, 0] += 255 - q8.sum(1)
np.concatenate([top.astype(np.uint8), q8.astype(np.uint8)], 1).tofile(f"{OUT}/{ID}_skin.bin")
json.dump({"count": int(len(a)), "joints": glb_joints, "parts": parts}, open(f"{OUT}/{ID}_skin.json", "w"))
rep["splats"] = dict(anzahl=int(len(a)), teile=parts, mb=round(len(a) * len(props) * 4 / 1e6, 1))

# ======================================================================= 8) Testbilder
log("8 Testbilder")
sc = scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'VERTEX'
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'
bilder = []
def foto(pfad, ort, rot, ortho, rx=300, ry=400):
    sc.render.resolution_x, sc.render.resolution_y = rx, ry; cam.location = ort; cam.rotation_euler = rot; cam.data.ortho_scale = ortho
    sc.render.filepath = pfad; bpy.ops.render.render(write_still=True); bilder.append(pfad)
for act, frames in ((f"{ID}_Walk", (0, 7, 14)), (f"{ID}_Run", (0, 5, 9)), (f"{ID}_Stumble", (12,))):
    rig.animation_data.action = bpy.data.actions[act]
    if rig.animation_data.action_slot is None and bpy.data.actions[act].slots: rig.animation_data.action_slot = bpy.data.actions[act].slots[0]
    for f in frames:
        sc.frame_set(f)
        foto(f"{OUT}/_test/{act}_{f:02d}_v.png", (0, -8, H * 0.52), (math.pi / 2, 0, 0), H * 1.25)
        foto(f"{OUT}/_test/{act}_{f:02d}_s.png", (8, 0, H * 0.52), (math.pi / 2, 0, math.pi / 2), H * 1.25)
rig.animation_data.action = None; sc.frame_set(0)
for pb in rig.pose.bones: pb.rotation_quaternion = (1, 0, 0, 0)
bpy.context.view_layer.update()
for s in ("L", "R"):                                                 # Hände in Ruhepose von vorn
    w = info["hand_" + s]["handgelenk"]; c = (w[0], -3, w[2] - 0.08)
    foto(f"{OUT}/_test/hand_{s}.png", c, (math.pi / 2, 0, 0), 0.3 * k)
Wd, Hd, cols = 300, 400, 6; rows_ = (len(bilder) + cols - 1) // cols; buf = np.ones((rows_ * Hd, cols * Wd, 4), np.float32) * 0.1
for i, pth in enumerate(bilder):
    im = bpy.data.images.load(pth); iw, ih = im.size; px = np.empty(iw * ih * 4, np.float32); im.pixels.foreach_get(px); px = px.reshape(ih, iw, 4)
    rr, cc = divmod(i, cols); y0 = (rows_ - 1 - rr) * Hd; buf[y0:y0 + ih, cc * Wd:cc * Wd + iw] = px[:Hd, :Wd]
o = bpy.data.images.new("sheet", cols * Wd, rows_ * Hd); o.pixels.foreach_set(buf.ravel()); o.filepath_raw = f"{OUT}/_test/{ID}_test.png"; o.file_format = 'PNG'; o.save()
print("REPORT", json.dumps(rep))

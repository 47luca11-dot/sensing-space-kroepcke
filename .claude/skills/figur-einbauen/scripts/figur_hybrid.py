# Hybrid-Figur: sauberes KI-Mesh (figuro-STL) als Körper + echte Scan-Splats als Oberfläche + menschliches Skelett mit Fingern.
#
#   Blender -b --factory-startup --python figur_hybrid.py -- --id <id> --stl <figuro.stl> --farbe <person.ply> --out <ordner>
#
#   1) STL auf echte Größe, per ICP (mit Maßstab) auf die ausgeschnittenen Scan-Splats ausrichten, 4 mm geschlossen neu vernetzen
#   2) Skelett: 23 Spiel-Knochen + Rigify-Hände (Handfläche, Daumen, 4 Finger à 3 Glieder) an Körper + Fingerspitzen anpassen
#   3) Füll-Mesh (80k Dreiecke, Farben aus dem Scan, 4 mm nach innen) + Heat-Gewichte + Jacken-Regel (oberhalb der Hüfte keine Bein-Gewichte)
#   4) Splats: Scan-Splats auf die Mesh-Oberfläche legen (> 3 cm daneben = verworfen), Hände + Lücken mit neuen Splats aus dem Mesh
#   5) Finger entspannen (leicht gekrümmt, nicht gespreizt) und als Ruhepose übernehmen – Mesh und Splats gleich
#   6) Carlos-Animationen (Arme per Welt-Drehung), GLB, Splats (glTF-Rahmen, SH Grad 1), Skin in GLB-Reihenfolge, Testbilder
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
ID, STL, FARBE, OUT = arg("--id"), arg("--stl"), arg("--farbe"), arg("--out")
CARLOS = arg("--carlos", os.path.join(REPO, "startbildschirm", "assets", "person1.glb"))
MENSCH = json.load(open(os.path.join(HIER, "mensch_skelett.json")))["bones"]
CT = json.load(open(os.path.join(HIER, "carlos_skelett.json")))["bones"]
DREIECKE = int(arg("--dreiecke", 80000))
ARM = ["shoulder.L", "upper_arm.L", "forearm.L", "hand.L", "shoulder.R", "upper_arm.R", "forearm.R", "hand.R"]
assert ID and STL and FARBE and OUT
os.makedirs(f"{OUT}/_test", exist_ok=True)
rep = dict(id=ID, art="hybrid"); rng = np.random.default_rng(7)
import time as _t; _T0 = _t.time()
def log(*a): print(f"[{_t.time() - _T0:6.0f}s]", *a, flush=True)

if bpy.app.background:
    bpy.ops.wm.read_factory_settings(use_empty=True); scene = bpy.context.scene
else:                                                         # Live-Blender (Blender-MCP): eigene Szene, nichts vom Nutzer anfassen
    old = bpy.data.scenes.get("Figur_" + ID)
    if old:
        for o in list(old.objects): bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.scenes.remove(old)
    scene = bpy.data.scenes.new("Figur_" + ID); bpy.context.window.scene = scene
    for act in [x for x in bpy.data.actions if x.name.startswith(ID + "_") or x.name.startswith("person1_")]: bpy.data.actions.remove(act)

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

# ======================================================================= 1) STL + Scan
log("1 STL ausrichten")
fprops, fa = SW.lesen(FARBE); FP = SW.pos(fprops, fa); FOP = SW.deckkraft(fprops, fa); FC = SW.farbe(fprops, fa)
H = float(np.percentile(FP[FOP > 0.5, 2], 99.98))
bpy.ops.wm.stl_import(filepath=STL); hi = bpy.context.selected_objects[0]; hi.name = hi.data.name = ID + "_koerper"
aktiv(hi); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
V = verts(hi); V = (V - [0, 0, V[:, 2].min()]) * (H / (V[:, 2].max() - V[:, 2].min()))
bein = V[(V[:, 2] > 0.15 * H) & (V[:, 2] < 0.4 * H)]; xl, xr = np.percentile(bein[:, 0], [1, 99])
V = V - [(xl + xr) / 2, np.median(bein[:, 1]), 0]
shin = V[(V[:, 2] > 0.14 * H) & (V[:, 2] < 0.2 * H)][:, 1].mean(); fy = V[V[:, 2] < 0.03 * H][:, 1]
if (fy.min() if abs(fy.min() - shin) > abs(fy.max() - shin) else fy.max()) > shin: V = V * [-1, -1, 1]
ziel = FP[FOP > 0.4]; ziel = ziel[rng.choice(len(ziel), min(80000, len(ziel)), replace=False)]
kd = KDTree(len(ziel))
for i, co in enumerate(ziel): kd.insert(co, i)
kd.balance()
def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    U, S_, Vt = np.linalg.svd(B0.T @ A0 / len(A)); D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: D[2, 2] = -1
    R = U @ D @ Vt; s = np.trace(np.diag(S_) @ D) / ((A0 ** 2).sum() / len(A)); return s, R, mb - s * R @ ma
probe = V[rng.choice(len(V), 20000, replace=False)]; S_, R_, t_ = 1.0, np.eye(3), np.zeros(3)
for it in range(40):
    Qp = S_ * probe @ R_.T + t_; nn = [kd.find(q) for q in Qp]
    idx = np.array([n[1] for n in nn]); d = np.array([n[2] for n in nn]); m = d < max(0.015, 2.5 * np.median(d))
    s2, R2, t2 = umeyama(Qp[m], ziel[idx[m]]); S_, R_, t_ = s2 * S_, R2 @ R_, s2 * R2 @ t_ + t2
V = S_ * V @ R_.T + t_
# Füße auf den Boden: Mesh und Scan-Splats gemeinsam verschieben (Farben bleiben deckungsgleich)
dz = float(V[:, 2].min()); V[:, 2] -= dz; FP[:, 2] -= dz
fa[:, SW.spalten(fprops, "z")] -= dz; rep["boden_korrektur_cm"] = round(dz * 100, 1)
hi.data.vertices.foreach_set("co", V.ravel().astype(np.float32)); hi.data.update()
rm = hi.modifiers.new("remesh", 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = 0.004; rm.adaptivity = 0
bpy.ops.object.modifier_apply(modifier="remesh")
bm = bmesh.new(); bm.from_mesh(hi.data); flicken(bm); bm.to_mesh(hi.data); bm.free()
V = verts(hi); H = float(V[:, 2].max()); rep["koerper"] = dict(zahlen(hi.data), groesse_m=round(H, 3), icp_massstab=round(float(S_), 3))
# Farbfeld aus dem Scan (6 mm, weit ausgebreitet)
def farbfeld(VOX=0.006, n_aus=16):
    gut = FOP > 0.3; Q = FP[gut]; lo = Q.min(0) - (n_aus + 2) * VOX
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
J, ROT, info = SK.anpassen(V, H, MENSCH, CT, log, kanten=EK); info["hi_punkte"] = V
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
m = game.modifiers.new("innen", 'DISPLACE'); m.strength = -0.015;   # tiefer innen: KI-Mesh darf die echten Splats nicht verdecken
m.mid_level = 0; m.direction = 'NORMAL'
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
dicht = behalten & (FOP > 0.45)
kd2 = KDTree(int(dicht.sum()))
for i, p in enumerate(SPl[dicht]): kd2.insert(p, i)
kd2.balance()
frei = np.array([kd2.find(p)[2] > float(arg("--luecke", 0.025)) for p in gP]) if len(gP) else np.zeros(0, bool)
gP, gN = gP[frei], gN[frei]
hP, hN = gP[:0], gN[:0]
rep["neue_splats"] = dict(haende=0, luecken=int(len(gP)))
def neue(pts, nrm, gr, dick):
    b = np.zeros((len(pts), len(fprops)))
    b[:, SW.spalten(fprops, "x", "y", "z")] = pts
    b[:, SW.spalten(fprops, "f_dc_0", "f_dc_1", "f_dc_2")] = (feld(pts) - 0.5) / SW.SH0
    b[:, fprops.index("opacity")] = math.log(0.95 / 0.05)
    b[:, SW.spalten(fprops, "scale_0", "scale_1")] = math.log(gr); b[:, fprops.index("scale_2")] = math.log(dick)
    zax = np.array([0, 0, 1.0]); ax = np.cross(np.broadcast_to(zax, nrm.shape), nrm); sn = np.linalg.norm(ax, axis=1)
    ang = np.arctan2(sn, nrm[:, 2]); ax = np.where(sn[:, None] > 1e-8, ax / (sn[:, None] + 1e-12), np.array([1.0, 0, 0]))
    b[:, SW.spalten(fprops, "rot_0", "rot_1", "rot_2", "rot_3")] = np.concatenate([np.cos(ang / 2)[:, None], ax * np.sin(ang / 2)[:, None]], 1)
    return b
sa = fa[behalten].copy()
a = np.concatenate([sa, neue(gP, gN, 0.0035, 0.001)], 0)
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

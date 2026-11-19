# Figur aus figuro-STL (sauberer Körper mit echten Fingern) + Gaussian-Splat-Scan derselben Person.
#
#   Live-Blender (Blender-MCP, per bpy.app.timers + exec):
#     sys.argv = ["blender", "--", "--id", "bv10", "--out", "<figuren/bv10/stl>"]  (STL/Scan werden aus figuren/<id>/ genommen)
#   headless:  Blender -b --factory-startup --python figur_stl.py -- --id <id> [--stl <figuro.stl>] [--quelle <person.ply>] --out <ordner>
#   danach:    gang.py -- --id <id> --out <derselbe ordner>
#
#   1) STL per ICP (mit Maßstab, ohne Arme) auf die Scan-Splats, dann GELENKWEISE nachdrehen (Kopf, Ober-/Unterarm, Ober-/Unter-
#      schenkel, Fuß drehen um ihr Gelenk auf den Scan; Hand folgt dem Unterarm) – die figuro-Pose weicht an den Armen ~5 cm ab.
#      Füße auf den Boden (Körper + Splats gemeinsam). Kein Remesh: die STL-Finger bleiben getrennt.
#   2) Skelett (skelett.anpassen) auf dem STL: Gelenke, Finger als Äste der Hand-Oberfläche
#   3) Füll-Mesh (80k) + Heat-Gewichte + Jacken-Regel + Finger-Gewichte; Splat-Gewichte per Data-Transfer (vor dem Einrücken)
#   4) Splats:
#      - Scan-Splats bleiben exakt wie gescannt (Position, Form, Drehung, volle SH); weg: > --aussen (2,2 cm) außerhalb der STL-Fläche,
#        fast durchsichtiger Schrott, halbtransparente Fransen > 1 cm außen, Splats tief im Körper; Hände (distal vom Handgelenk)
#      - Löcher: STL-Fläche ohne deckenden Scan-Splat (Deckkraft > 0,45) im Umkreis --luecke (2 cm) -> neue flache Splats auf der Fläche,
#        Farbe = lokaler Median deckender Scan-Splats DERSELBEN Körperregion (Regionen aus den Skin-Gewichten), nur DC (SH-Rest = 0)
#      - Hände: dichte Splats aus der STL-Hand (echte Finger), Hautton = Median der Unterarm-Haut aus dem Scan, leichte Variation,
#        dunkler in Falten (Höhlung) und auf der Handfläche
#   5) Finger entspannen (nur wenn getrennt erkannt) -> neue Ruhepose, 6) Carlos-Clips (Walk/Run ersetzt danach gang.py),
#   7) GLB + <id>_splats.ply + <id>_skin.bin/json (Gelenkreihenfolge = GLB), 8) Testbilder
import bpy, bmesh, json, sys, math, struct, os
import numpy as np
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

HIER = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.normpath(os.path.join(HIER, "..", "..", "..", ".."))
sys.path.insert(0, HIER)
import splat_werkzeug as SW
import skelett as SK

argv = sys.argv[sys.argv.index("--") + 1:]
def arg(n, std=None): return argv[argv.index(n) + 1] if n in argv else std
ID = arg("--id")
STL = arg("--stl", os.path.join(REPO, "figuren", ID, f"figuro_{ID}.stl"))
FARBE = arg("--quelle", arg("--farbe", os.path.join(REPO, "figuren", ID, f"{ID}_person.ply")))
OUT = arg("--out", os.path.join(REPO, "figuren", ID, "stl"))
AUSSEN = float(arg("--aussen", 0.022))       # Scan-Splats weiter außerhalb der STL-Fläche = Schwebeteile/Fransen
LUECKE = float(arg("--luecke", 0.02))        # kein deckender Scan-Splat in diesem Umkreis = Loch
DECKEND = 0.45
NUR_AUSRICHTEN = "--nur_ausrichten" in argv
CARLOS = arg("--carlos", os.path.join(REPO, "startbildschirm", "assets", "person1.glb"))
MENSCH = json.load(open(os.path.join(HIER, "mensch_skelett.json")))["bones"]
CT = json.load(open(os.path.join(HIER, "carlos_skelett.json")))["bones"]
DREIECKE = int(arg("--dreiecke", 80000))
ARM = ["shoulder.L", "upper_arm.L", "forearm.L", "hand.L", "shoulder.R", "upper_arm.R", "forearm.R", "hand.R"]
assert ID and os.path.exists(STL) and os.path.exists(FARBE)
os.makedirs(f"{OUT}/_test", exist_ok=True)
rep = dict(id=ID, art="stl_scan"); rng = np.random.default_rng(7)
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
def kanten(ob):
    E = np.zeros(len(ob.data.edges) * 2, np.int64); ob.data.edges.foreach_get("vertices", E); return E.reshape(-1, 2)
def kd_bau(P):
    kd = KDTree(len(P))
    for i, p in enumerate(P): kd.insert(p, i)
    kd.balance(); return kd
def kd_nn(kd, Q):
    idx = np.zeros(len(Q), np.int64); d = np.zeros(len(Q))
    for i, q in enumerate(Q):
        co, j, dd = kd.find(q)
        if j is None: idx[i] = 0; d[i] = 1e9
        else: idx[i] = j; d[i] = dd
    return idx, d
def segd(Pq, a, b):
    ab = b - a; t = np.clip(((Pq - a) @ ab) / max(ab @ ab, 1e-12), 0, 1); return np.linalg.norm(Pq - (a + t[:, None] * ab), axis=1)
def glaetten(D, E, n, wert=0.5):
    """Uniform-Laplace auf einem Feld je Punkt (z. B. Verschiebungen) über die Mesh-Kanten."""
    deg = np.bincount(E.ravel(), minlength=len(D)).astype(np.float64); deg[deg == 0] = 1
    for _ in range(n):
        S = np.stack([np.bincount(E[:, 0], D[E[:, 1], c], len(D)) + np.bincount(E[:, 1], D[E[:, 0], c], len(D)) for c in range(D.shape[1])], 1)
        D = (1 - wert) * D + wert * S / deg[:, None]
    return D
def signed_dist(bvh, P, maxd=0.5):
    sd = np.full(len(P), 1e9); fl = np.full(len(P), -1, np.int64)
    for i, p in enumerate(P):
        loc, nrm, fi, dist = bvh.find_nearest(Vector(p), maxd)
        if loc is None: continue
        sd[i] = dist if (Vector(p) - loc).dot(nrm) >= 0 else -dist; fl[i] = fi
    return sd, fl

# ======================================================================= 1) STL ausrichten
log("1 STL ausrichten")
fprops, fa = SW.lesen(FARBE); FP = SW.pos(fprops, fa); FOP = SW.deckkraft(fprops, fa); FC = SW.farbe(fprops, fa); FG = SW.groesse(fprops, fa)
H0 = float(np.percentile(FP[FOP > 0.5, 2], 99.98))
bpy.ops.wm.stl_import(filepath=STL); hi = bpy.context.selected_objects[0]; hi.name = hi.data.name = ID + "_koerper"
aktiv(hi); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bm = bmesh.new(); bm.from_mesh(hi.data); flicken(bm)
# kleine lose Teile weg (große Teile behalten, falls das STL aus mehreren Schalen besteht)
bm.verts.ensure_lookup_table(); seen = set(); isl = []
for v0 in bm.verts:
    if v0.index in seen: continue
    st = [v0]; comp = []; seen.add(v0.index)
    while st:
        v1 = st.pop(); comp.append(v1)
        for e in v1.link_edges:
            o_ = e.other_vert(v1)
            if o_.index not in seen: seen.add(o_.index); st.append(o_)
    isl.append(comp)
weg = [v for c in isl if len(c) < 0.005 * len(bm.verts) for v in c]
if weg: bmesh.ops.delete(bm, geom=weg, context='VERTS')
bm.to_mesh(hi.data); bm.free(); hi.data.update()
rep["stl"] = dict(inseln=len(isl), entfernt=len(weg))
V = verts(hi); V = (V - [0, 0, V[:, 2].min()]) * (H0 / (V[:, 2].max() - V[:, 2].min()))
bein = V[(V[:, 2] > 0.15 * H0) & (V[:, 2] < 0.4 * H0)]; xl, xr = np.percentile(bein[:, 0], [1, 99])
V = V - [(xl + xr) / 2, np.median(bein[:, 1]), 0]
shin = V[(V[:, 2] > 0.14 * H0) & (V[:, 2] < 0.2 * H0)][:, 1].mean(); fy = V[V[:, 2] < 0.03 * H0][:, 1]
if (fy.min() if abs(fy.min() - shin) > abs(fy.max() - shin) else fy.max()) > shin: V = V * [-1, -1, 1]
# gleichmäßig neu vernetzen (figuro ist vorn viel dichter vernetzt als hinten -> Querschnitts-Mediane im Skelett wären verschoben);
# 2 mm, damit eng anliegende Finger getrennt bleiben (4 mm hat sie verschmolzen)
hi.data.vertices.foreach_set("co", V.ravel().astype(np.float32)); hi.data.update(); aktiv(hi)
rm = hi.modifiers.new("remesh", 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = float(arg("--remesh", 0.002)); rm.adaptivity = 0
bpy.ops.object.modifier_apply(modifier="remesh")
bm = bmesh.new(); bm.from_mesh(hi.data); flicken(bm); bm.to_mesh(hi.data); bm.free(); hi.data.update()
V = verts(hi)
deck = FOP > 0.4
ZIEL = FP[deck]; kd_ziel = kd_bau(ZIEL)
def umeyama(A, B, mit_massstab=True):
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    U, S_, Vt = np.linalg.svd(B0.T @ A0 / len(A)); D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: D[2, 2] = -1
    R = U @ D @ Vt; s = np.trace(np.diag(S_) @ D) / ((A0 ** 2).sum() / len(A)) if mit_massstab else 1.0
    return s, R, mb - s * R @ ma
# globales ICP nur mit Rumpf, Beinen, Kopf (Arme stehen im STL anders als im Scan)
k0 = H0 / 1.8
ohne_arme = ~((np.abs(V[:, 0]) > 0.21 * k0) & (V[:, 2] > 0.5 * H0))
probe = V[rng.choice(np.where(ohne_arme)[0], 20000, replace=False)]; S_, R_, t_ = 1.0, np.eye(3), np.zeros(3)
for it in range(40):
    Qp = S_ * probe @ R_.T + t_; idx, d = kd_nn(kd_ziel, Qp); m = d < max(0.015, 2.5 * np.median(d))
    s2, R2, t2 = umeyama(Qp[m], ZIEL[idx[m]]); S_, R_, t_ = s2 * S_, R2 @ R_, s2 * R2 @ t_ + t2
V = S_ * V @ R_.T + t_
rep["icp"] = dict(massstab=round(float(S_), 4), median_cm=round(float(np.median(d)) * 100, 2))
log("ICP", rep["icp"])
EK = kanten(hi)

# --- gelenkweise nachstellen: Körperteile (Gruppen) drehen um ihr Gelenk auf den Scan, Kopf zusätzlich verschieben;
#     danach weich gemischt wie Linear-Blend-Skinning (Gewichte aus dem Abstand zu den Knochen der Gruppe minus Gliedradius)
GRUPPEN = {"rumpf": ["spine", "spine.001", "spine.002", "spine.003", "shoulder.L", "shoulder.R"], "hals": ["spine.004"], "kopf": ["spine.005", "spine.006"]}
RADIUS = {"rumpf": 0.11, "hals": 0.05, "kopf": 0.08}
for _s in "LR":
    GRUPPEN.update({"oa." + _s: ["upper_arm." + _s], "ua." + _s: ["forearm." + _s],
                    "hd." + _s: ["hand." + _s] + [f"{f}.0{i}.{_s}" for f in SK.FINGER for i in (1, 2, 3)],
                    "os." + _s: ["thigh." + _s], "us." + _s: ["shin." + _s], "fu." + _s: ["foot." + _s, "toe." + _s]})
    RADIUS.update({"oa." + _s: 0.045, "ua." + _s: 0.035, "hd." + _s: 0.02, "os." + _s: 0.07, "us." + _s: 0.05, "fu." + _s: 0.04})
GNAMEN = list(GRUPPEN.keys()); GI = {g: i for i, g in enumerate(GNAMEN)}
def gruppen_abstand(Vq, J):
    D = np.zeros((len(Vq), len(GNAMEN)), np.float32)
    for gi_, g in enumerate(GNAMEN):
        D[:, gi_] = np.min([segd(Vq, np.array(J[(b, "head")]), np.array(J[(b, "tail")])) for b in GRUPPEN[g]], 0) - RADIUS[g]
    return D
def gruppen_labels(Vq, J): return GNAMEN, gruppen_abstand(Vq, J).argmin(1)
def mischgewichte(D, sigma=0.012):
    E = D - D.min(1, keepdims=True); W_ = np.exp(-E / sigma) * (E < 4 * sigma); return W_ / W_.sum(1, keepdims=True)
def um_punkt(R, c, t=np.zeros(3)):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c + t - R @ c; return M
def anwenden(M, X): return X @ M[:3, :3].T + M[:3, 3]
def winkel(R): return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
def kappen(R, maxw):
    a_ = winkel(R)
    if a_ <= maxw or a_ < 1e-6: return R
    q = Matrix(R.tolist()).to_quaternion(); return np.array(Matrix.Identity(3).to_quaternion().slerp(q, maxw / a_).to_matrix())

def gelenkweise(V, runden=3):
    """Ziele je Teil = Scan-Punkte mit demselben nächsten Knochen (sonst rastet ein verschobener Arm am Rumpf ein), Zuordnung Scan -> STL
    (Scan oft nur halbe Schale). Arme/Beine nur Schwenk (Drall um die Knochenachse ist bei Zylindern unbestimmt und würde die Hand
    verdrehen); Kopf starr mit Drehung + Verschiebung (figuro-Hals ist oft kürzer), Hals bekommt die Hälfte. Hand/Fuß folgen."""
    cur = V.copy(); summe = {}; kopf_cm = 0.0
    for rd in range(runden):
        zmin = cur[:, 2].min(); Vz = cur - [0, 0, zmin]
        Jt, _, _ = SK.anpassen(Vz, float(Vz[:, 2].max()), MENSCH, CT, lambda *a: None, kanten=None)
        J = {key: np.array(v) + [0, 0, zmin] for key, v in Jt.items()}
        D = gruppen_abstand(cur, J); lab = D.argmin(1); lab_z = gruppen_abstand(ZIEL, J).argmin(1)
        T = {g: np.eye(4) for g in GNAMEN}
        piv = {"kopf": J[("spine.005", "head")], "kopf_ende": J[("spine.006", "tail")]}
        plan = [("kopf", [], "kopf_ende", 25)]
        for s in "LR":
            piv.update({"oa." + s: J[("upper_arm." + s, "head")], "ua." + s: J[("forearm." + s, "head")], "hd." + s: J[("hand." + s, "head")],
                        "os." + s: J[("thigh." + s, "head")], "us." + s: J[("shin." + s, "head")], "fu." + s: J[("foot." + s, "head")]})
            plan += [("oa." + s, ["ua." + s, "hd." + s], "ua." + s, 35), ("ua." + s, ["hd." + s], "hd." + s, 45),
                     ("os." + s, ["us." + s, "fu." + s], "us." + s, 15), ("us." + s, ["fu." + s], "fu." + s, 15)]
        piv = {g_: np.array(p, float) for g_, p in piv.items()}
        for g, kinder, ende, maxw in plan:
            if g[:3] in ("oa.", "ua."):                                  # Arm-Erkennung plausibel? (Arm steht seitlich ab)
                s_ = 1 if g.endswith("L") else -1
                if not (s_ * piv["ua." + g[-1]][0] > s_ * piv["oa." + g[-1]][0] + 0.04 and s_ * piv["hd." + g[-1]][0] > s_ * piv["ua." + g[-1]][0] + 0.03):
                    rep.setdefault("gelenk_uebersprungen", []).append(f"{rd}:{g}"); continue
            mg = lab == GI[g]; mz = lab_z == GI[g]
            if mg.sum() < 50 or mz.sum() < 50: continue
            Xg = anwenden(T[g], cur[mg]); kdx = kd_bau(Xg); Zg = ZIEL[mz]
            if len(Zg) > 5000: Zg = Zg[rng.choice(len(Zg), 5000, replace=False)]
            c = piv[g]; Tc = np.eye(4)
            for it in range(40):
                Ys = anwenden(np.linalg.inv(Tc), Zg)                     # Scan in den STL-Rahmen zurück
                j, d = kd_nn(kdx, Ys)
                m = (d < max(0.015, 2.5 * np.median(d))) & (d < 0.12)
                if m.sum() < 30: break
                if g == "kopf": _, Ri, ti = umeyama(Xg[j[m]], Ys[m], mit_massstab=False); Ti = np.eye(4); Ti[:3, :3] = Ri; Ti[:3, 3] = ti
                else:
                    A = Xg[j[m]] - c; B = Ys[m] - c
                    U, S2, Vt = np.linalg.svd(B.T @ A); Dm = np.eye(3)
                    if np.linalg.det(U @ Vt) < 0: Dm[2, 2] = -1
                    Ti = um_punkt(U @ Dm @ Vt, c)
                Tc = Tc @ Ti
                if winkel(Ti[:3, :3]) < 0.05 and np.linalg.norm(Ti[:3, 3] - (c - Ti[:3, :3] @ c)) < 1e-4: break
            R = Tc[:3, :3]
            erlaubt = max(0.0, maxw - summe.get(g, 0))
            if g == "kopf":
                t_ = anwenden(Tc, c[None])[0] - c; t_ *= min(1.0, 0.10 / max(np.linalg.norm(t_), 1e-9))
                R = kappen(R, erlaubt); M = um_punkt(R, c, t_)
                T["hals"] = um_punkt(kappen(R, winkel(R) / 2), c, t_ / 2) @ T["hals"]; kopf_cm = float(np.linalg.norm(t_)) * 100
            else:
                a_ = piv[ende] - c; a_ /= np.linalg.norm(a_)
                R = kappen(np.array(Vector(a_).rotation_difference(Vector(R @ a_)).to_matrix()), erlaubt)   # nur Schwenk
                M = um_punkt(R, c)
            for x_ in [g] + kinder: T[x_] = M @ T[x_]
            for kd_ in set(kinder + [ende]):
                if kd_ in piv and kd_ != g: piv[kd_] = anwenden(M, piv[kd_][None])[0]
            summe[g] = summe.get(g, 0) + winkel(R)
        Wm = mischgewichte(D); neu = np.zeros_like(cur)
        for g in GNAMEN:
            m = Wm[:, GI[g]] > 1e-4
            if m.any(): neu[m] += Wm[m, GI[g]][:, None] * anwenden(T[g], cur[m])
        cur = neu
    rep["gelenk_grad"] = {g_: round(v, 1) for g_, v in summe.items()}; rep["kopf_verschiebung_cm"] = round(kopf_cm, 1)
    # Beinlänge: figuro-Beine sind oft kürzer als im Scan (Schuhe des Scans stehen auf z = 0, die STL-Sohlen darüber) ->
    # unterhalb der Hüfte senkrecht strecken (weicher Übergang 10 cm), Arme/Hände ausgenommen
    zmin = cur[:, 2].min(); Vz = cur - [0, 0, zmin]
    Jt, _, _ = SK.anpassen(Vz, float(Vz[:, 2].max()), MENSCH, CT, lambda *a: None, kanten=None)
    J = {key: np.array(v) + [0, 0, zmin] for key, v in Jt.items()}
    namen, lab = gruppen_labels(cur, J)
    z_h = 0.5 * (J[("thigh.L", "head")][2] + J[("thigh.R", "head")][2])
    f = float(np.clip(z_h / max(z_h - zmin, 1e-6), 0.92, 1.10))
    arm = np.isin(lab, [GI[g_] for g_ in GNAMEN if g_[:3] in ("oa.", "ua.", "hd.")])
    w = np.clip((z_h - cur[:, 2]) / 0.10, 0, 1) * ~arm
    cur[:, 2] = cur[:, 2] - (f - 1) * (z_h - cur[:, 2]) * w
    rep["bein_streckung"] = round(f, 3)
    return cur

V = gelenkweise(V)
bvh0 = BVHTree.FromPolygons(V.tolist(), [p.vertices[:] for p in hi.data.polygons])
def statistik(tag):
    P = FP[FOP > DECKEND]; sd, _ = signed_dist(bvh0, P); x, z = P[:, 0], P[:, 2]; Hh = H0; r = {}
    for name, m in (("kopf", z > 0.88 * Hh), ("rumpf", (z > 0.5 * Hh) & (z < 0.85 * Hh) & (np.abs(x) < 0.2)), ("arme", (np.abs(x) > 0.25) & (z > 0.5 * Hh)),
                    ("beine", z < 0.45 * Hh)):
        s = sd[m]; r[name] = dict(aussen_2cm=round(float(np.mean(s > 0.02)) * 100, 1), innen_2cm=round(float(np.mean(s < -0.02)) * 100, 1), median_cm=round(float(np.median(s)) * 100, 2))
    rep["abstand_" + tag] = r; log("Abstand Scan->STL", tag, r)
statistik("nach_gelenken")
# Füße auf den Boden: Körper und Scan-Splats gemeinsam verschieben (bleiben deckungsgleich)
dz = float(V[:, 2].min()); V[:, 2] -= dz; FP[:, 2] -= dz; ZIEL[:, 2] -= dz
fa[:, SW.spalten(fprops, "z")] -= dz; rep["boden_korrektur_cm"] = round(dz * 100, 1)
hi.data.vertices.foreach_set("co", V.ravel().astype(np.float32)); hi.data.update()
H = float(V[:, 2].max()); k = H / 1.8
rep["koerper"] = dict(zahlen(hi.data), groesse_m=round(H, 3))
log("Körper", rep["koerper"], "Boden", rep["boden_korrektur_cm"], "cm")
if NUR_AUSRICHTEN:
    print("REPORT", json.dumps(rep)); raise SystemExit(0)

# ======================================================================= 2) Skelett auf dem STL
log("2 Skelett")
EK = kanten(hi)
J, ROT, info = SK.anpassen(V, H, MENSCH, CT, log, kanten=EK); info["hi_punkte"] = V
rig = SK.rig_bauen(scene, ID + "_rig", J, ROT, MENSCH, CT)
rep["skelett"] = dict(knochen=len(rig.data.bones), **{k_: v for k_, v in info.items() if k_.startswith(("schritt", "schulter", "arm_"))},
                      finger={s: info["hand_" + s]["finger"] for s in ("L", "R")}, verschmolzen={s: info["hand_" + s]["verschmolzen"] for s in ("L", "R")})

# ======================================================================= 3) Füll-Mesh + Gewichte (noch nicht eingerückt)
log("3 Füll-Mesh + Gewichte")
game = hi.copy(); game.data = hi.data.copy(); game.name = game.data.name = ID + "_mesh"; scene.collection.objects.link(game); aktiv(game)
n_tri = sum(len(p.vertices) - 2 for p in game.data.polygons)
if n_tri > DREIECKE:
    m = game.modifiers.new("dec", 'DECIMATE'); m.ratio = DREIECKE / n_tri; m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="dec")
bm = bmesh.new(); bm.from_mesh(game.data); flicken(bm); bm.to_mesh(game.data); bm.free()
bpy.ops.object.shade_smooth()
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
zh = J[("thigh.L", "head")].z; gs = {g.name: g for g in game.vertex_groups}; geaendert = 0       # Jacken-Regel
for v in game.data.vertices:
    f = min(1.0, max(0.0, (v.co.z - (zh - 0.03 * k)) / (0.07 * k)))
    if f <= 0: continue
    weg_ = 0.0
    for g in list(v.groups):
        if game.vertex_groups[g.group].name.startswith("thigh."):
            weg_ += g.weight * f; g.weight *= (1 - f)
    if weg_ > 1e-4: gs["spine"].add([v.index], weg_, 'ADD'); geaendert += 1
rep["hand_gewichte"] = SK.hand_gewichte(game, rig, info, k, log)
aktiv(game); bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
bpy.ops.object.vertex_group_limit_total(group_select_mode='ALL', limit=4)
bpy.ops.object.vertex_group_normalize_all(lock_active=False)
bpy.ops.object.mode_set(mode='OBJECT')
rep["fuellmesh"] = dict(zahlen(game.data), nachgetragen=fb, jacke_entkoppelt=geaendert)

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
    bpy.data.objects.remove(ob_, do_unlink=True); bpy.data.meshes.remove(me_)
    s = out.sum(1, keepdims=True); return out / np.maximum(s, 1e-6)

# Körperregionen aus den Skin-Gewichten (Farben werden nur innerhalb einer Region übernommen)
def region_von(n):
    if n in ("spine.005", "spine.006"): return "kopf"
    if n == "spine.004": return "hals"
    if n == "spine": return "becken"
    if n in ("spine.001", "spine.002", "spine.003", "shoulder.L", "shoulder.R"): return "rumpf"
    s = n[-1]
    for pre, r in (("upper_arm", "oberarm"), ("forearm", "unterarm"), ("hand", "hand"), ("palm", "hand"), ("thumb", "hand"), ("f_", "hand"),
                   ("thigh", "oberschenkel"), ("shin", "unterschenkel"), ("foot", "fuss"), ("toe", "fuss")):
        if n.startswith(pre): return r + "." + s
    return "rumpf"
RNAMEN = sorted({region_von(n) for n in joints}); RI = {r: i for i, r in enumerate(RNAMEN)}
ONEHOT = np.zeros((len(joints), len(RNAMEN)), np.float32)
for i, n in enumerate(joints): ONEHOT[i, RI[region_von(n)]] = 1
def regionen(Wq): return (Wq @ ONEHOT).argmax(1)

# ======================================================================= 4) Splats
log("4 Splats")
bvh = BVHTree.FromPolygons(V.tolist(), [p.vertices[:] for p in hi.data.polygons])
hand = {s: (np.array(info["hand_" + s]["handgelenk"]), np.array(info["hand_" + s]["richtung"])) for s in ("L", "R")}
def distal(Pq, s, r=0.28):
    w, d = hand[s]; return (((Pq - w) @ d) > 0) & (np.linalg.norm(Pq - w, axis=1) < r * k)
sd, fl = signed_dist(bvh, FP)
PN = np.zeros(len(hi.data.polygons) * 3); hi.data.polygons.foreach_get("normal", PN); PN = PN.reshape(-1, 3)[np.maximum(fl, 0)]
Ws = gewichte(FP); reg_s = regionen(Ws)
haende = np.zeros(len(FP), bool)
for s in "LR": haende |= distal(FP, s) & np.isin(reg_s, [RI["hand." + s], RI["unterarm." + s]])
# Bodenreflexe (oranger Flurboden) an Armen, Rumpf, Beinen: stark gesättigtes Orange (Haut ~0,4 Sättigung, Boden > 0,6)
mx_, mn_ = FC.max(1), FC.min(1); sat = (mx_ - mn_) / np.maximum(mx_, 1e-6)
orange = ((FC[:, 0] >= FC[:, 1]) & (FC[:, 1] >= FC[:, 2]) & (sat > 0.6) & (FC[:, 0] > 0.5)
          & ((FC[:, 1] - FC[:, 2]) / np.maximum(FC[:, 0] - FC[:, 2], 1e-6) > 0.15)
          & ~np.isin(reg_s, [RI["kopf"], RI["hals"], RI["fuss.L"], RI["fuss.R"]]))
if "--ohne_orange" in argv: orange[:] = False
# heller Saum unter den Haaren am Hinterkopf (Gegenlicht-Halo, z. B. bv10): hinter dem Kopfgelenk (Atlas), zwischen C7 und
# Ohrhöhe, hell und warm – die Lücke wird danach mit Haar-/Halsfarbe derselben Region gefüllt
saum = np.zeros(len(FP), bool)
if "--saum_hinterkopf" in argv:
    at_, c7_ = J[("spine.005", "tail")], J[("spine.004", "head")]
    saum = ((FP[:, 1] > at_.y - 0.01 * k) & (FP[:, 2] > c7_.z - 0.02 * k) & (FP[:, 2] < at_.z + 0.06 * k) & (np.abs(FP[:, 0] - at_.x) < 0.15 * k)
            & (FC.mean(1) > 0.38) & ((FC[:, 0] - FC[:, 2]) > 0.2))
gruende = [("schrott_durchsichtig", FOP < 0.04), ("haende_ersetzt", haende), ("aussen_" + str(AUSSEN), sd > AUSSEN),
           ("fransen_halbtransparent", (sd > 0.01) & (FOP < 0.15)), ("gross_aussen", (FG > 0.02) & (sd > 0.012)), ("tief_innen", sd < -0.08),
           ("bodenreflex_orange", orange), ("saum_hinterkopf", saum)]
weg = np.zeros(len(FP), bool); rep["scan_entfernt"] = {}
for name, m in gruende:
    rep["scan_entfernt"][name] = int((m & ~weg).sum()); weg |= m
behalten = ~weg
rep["scan_splats"] = dict(gesamt=int(len(FP)), behalten=int(behalten.sum()), entfernt=int(weg.sum()), unveraendert=True)
log("Scan-Splats", rep["scan_splats"], rep["scan_entfernt"])
QUELLE = behalten & (FOP > DECKEND)                                     # deckende Scan-Splats: Abdeckung + Farbquelle
kd_deck = kd_bau(FP[QUELLE])
kd_reg = {}
for r in RNAMEN:
    src = np.where(QUELLE & (reg_s == RI[r]))[0]
    if len(src) >= 10: kd_reg[r] = (src, kd_bau(FP[src]))
rep["farbquellen_je_region"] = {r: int(len(v[0])) for r, v in kd_reg.items()}
def lokal_median(Pq, rq):
    """Farbe = Median der nächsten deckenden Scan-Splats DERSELBEN Region (die nächste Schale von bis zu 16 Splats)."""
    out = np.zeros((len(Pq), 3)); fehlt = 0
    for r in np.unique(rq):
        sel = np.where(rq == r)[0]; name = RNAMEN[r]
        if name not in kd_reg:                                       # Region ohne Scan-Farbe: Nachbarregion entlang der Kette
            fehlt += len(sel); src, kd_ = None, None
            for alt in (name.replace("unterarm", "oberarm"), name.replace("hand", "unterarm"), name.replace("fuss", "unterschenkel"), "rumpf"):
                if alt in kd_reg: src, kd_ = kd_reg[alt]; break
        else: src, kd_ = kd_reg[name]
        for i in sel:                                                   # nächste Schale (bis 1,5 cm hinter dem nächsten Splat)
            nb = kd_.find_n(Pq[i], 16); dd = np.array([t[2] for t in nb]); jj = np.array([t[1] for t in nb])
            nah = dd < dd.min() + 0.015
            if nah.sum() >= 5: jj = jj[nah]
            out[i] = np.median(FC[src[jj]], 0)
    return out, fehlt

# Dreiecke des STL, Region je Dreieck
me = hi.data; me.calc_loop_triangles(); nt = len(me.loop_triangles)
tv = np.zeros(nt * 3, np.int64); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
A = V[tv]; area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
Wv = gewichte(V); reg_v = regionen(Wv)
reg_t = reg_v[tv[:, 0]]; cent = A.mean(1)
hand_t = np.zeros(nt, bool); hand_seite = np.full(nt, "", object)
for s in "LR":
    m = distal(cent, s) & np.isin(reg_t, [RI["hand." + s], RI["unterarm." + s]]); hand_t |= m; hand_seite[m] = s
def abtasten(sel, dichte):
    cnt = np.floor(area[sel] * dichte + rng.random(len(sel))).astype(np.int64)
    pick = np.repeat(sel, cnt); u, w2 = rng.random(len(pick)), rng.random(len(pick)); fl = u + w2 > 1; u[fl], w2[fl] = 1 - u[fl], 1 - w2[fl]
    pts = A[pick, 0] * (1 - u - w2)[:, None] + A[pick, 1] * u[:, None] + A[pick, 2] * w2[:, None]
    nrm = np.cross(A[pick, 1] - A[pick, 0], A[pick, 2] - A[pick, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return pts, nrm, pick
# --- Löcher (ohne Hände)
gP, gN, gT = abtasten(np.where(~hand_t)[0], 1 / 25e-6)
_, dcov = kd_nn(kd_deck, gP); frei = dcov > LUECKE
gP, gN, gT = gP[frei], gN[frei], gT[frei]
gC, fehlt = lokal_median(gP, reg_t[gT])
rep["neue_splats"] = dict(luecken=int(len(gP)), luecken_ohne_regionsfarbe=int(fehlt),
                          luecken_je_region={r: int((reg_t[gT] == RI[r]).sum()) for r in RNAMEN if (reg_t[gT] == RI[r]).any()})
# --- Hände: Hautton aus der Unterarm-Haut des Scans
haut = []
for s in "LR":
    e_, w_ = np.array(J[("forearm." + s, "head")]), np.array(J[("forearm." + s, "tail")]); ax = w_ - e_
    t_ = ((FP - e_) @ ax) / (ax @ ax)
    haut.append(np.where(QUELLE & (reg_s == RI["unterarm." + s]) & (t_ > 0.3) & (t_ < 0.95))[0])
haut = np.concatenate(haut); HC = FC[haut]; m0 = np.median(HC, 0)
HC = HC[np.linalg.norm(HC - m0, axis=1) < 0.15]; HAUT = np.median(HC, 0) if len(HC) > 20 else m0
rep["hautton"] = dict(rgb=[round(float(c), 3) for c in HAUT], aus_splats=int(len(HC)))
log("Hautton", rep["hautton"])
hand_vi = np.where(np.isin(reg_v, [RI["hand.L"], RI["hand.R"], RI["unterarm.L"], RI["unterarm.R"]]))[0]
kd_hv = kd_bau(V[hand_vi])
wellen = [(rng.normal(size=3) * (2 * math.pi / rng.uniform(0.025, 0.06)), rng.uniform(0, 2 * math.pi), rng.uniform(0.01, 0.025)) for _ in range(6)]
def hautfarbe(Pq, Nq, seiten):
    var = np.zeros(len(Pq))
    for kv, ph, amp in wellen: var += amp * np.sin(Pq @ kv + ph)
    var += rng.normal(0, 0.012, len(Pq))
    hoehle = np.zeros(len(Pq))
    for i, (p, n) in enumerate(zip(Pq, Nq)):                            # Falten/Zwischenräume: Nachbarfläche liegt über der Tangentialebene
        nb = kd_hv.find_range(p, 0.012 * k)
        if len(nb) < 4: continue
        Q = np.array([t[0] for t in nb]) - p; hoehle[i] = np.mean((Q @ n) > 0.0015 * k)
    hoehle = np.clip((hoehle - 0.05) / 0.35, 0, 1)
    if len(Pq) > 10:                                                   # Falten weich auslaufen lassen (Nachbarmittel 6 mm)
        kq = kd_bau(Pq); h2 = hoehle.copy()
        for i, p in enumerate(Pq):
            nb = kq.find_range(p, 0.006 * k)
            if nb: h2[i] = np.mean(hoehle[[t[1] for t in nb]])
        hoehle = h2
    palm = np.zeros(len(Pq))
    for s in "LR":
        m = seiten == s
        if m.any(): palm[m] = np.clip(Nq[m] @ np.array(info["handflaeche_" + s]), 0, 1)
    fac = (1 + var) * (1 - 0.26 * hoehle) * (1 - 0.07 * palm)
    return np.clip(HAUT[None, :] * fac[:, None], 0, 1), hoehle
hP, hN, hT = abtasten(np.where(hand_t)[0], 1 / 5e-6)
hC, hoehle = hautfarbe(hP, hN, hand_seite[hT])
rep["neue_splats"]["haende"] = int(len(hP)); rep["neue_splats"]["hand_falten_anteil"] = round(float((hoehle > 0.3).mean()), 3)
log("neue Splats", rep["neue_splats"])

def neue(pts, nrm, farbe, gr, dick, deck=0.95):
    b = np.zeros((len(pts), len(fprops)))
    b[:, SW.spalten(fprops, "x", "y", "z")] = pts
    b[:, SW.spalten(fprops, "f_dc_0", "f_dc_1", "f_dc_2")] = (np.asarray(farbe) - 0.5) / SW.SH0     # nur DC, SH-Rest = 0
    b[:, fprops.index("opacity")] = math.log(deck / (1 - deck))
    b[:, SW.spalten(fprops, "scale_0", "scale_1")] = math.log(gr); b[:, fprops.index("scale_2")] = math.log(dick)
    zax = np.array([0, 0, 1.0]); ax = np.cross(np.broadcast_to(zax, nrm.shape), nrm); sn = np.linalg.norm(ax, axis=1)
    ang = np.arctan2(sn, nrm[:, 2]); ax = np.where(sn[:, None] > 1e-8, ax / (sn[:, None] + 1e-12), np.array([1.0, 0, 0]))
    b[:, SW.spalten(fprops, "rot_0", "rot_1", "rot_2", "rot_3")] = np.concatenate([np.cos(ang / 2)[:, None], ax * np.sin(ang / 2)[:, None]], 1)
    return b
sa = fa[behalten].copy()
a = np.concatenate([sa, neue(hP, hN, hC, 0.0019, 0.0005, 0.97), neue(gP, gN, gC, 0.0035, 0.001)], 0)
parts = [int(len(sa)), int(len(hP)), int(len(gP))]
W = gewichte(SW.pos(fprops, a))

# Füll-Mesh: Farben regionsweise + Hautton an den Händen, dann nach innen rücken (unter die Scan-Splats)
GV = verts(game); GN = np.zeros(len(GV) * 3); game.data.vertices.foreach_get("normal", GN); GN = GN.reshape(-1, 3)
reg_g = regionen(gewichte(GV))
g_hand = np.zeros(len(GV), bool); g_seite = np.full(len(GV), "", object)
for s in "LR":
    m = distal(GV, s) & np.isin(reg_g, [RI["hand." + s], RI["unterarm." + s]]); g_hand |= m; g_seite[m] = s
GC, _ = lokal_median(GV, reg_g)
if g_hand.any(): GC[g_hand] = hautfarbe(GV[g_hand], GN[g_hand], g_seite[g_hand])[0]
VC = np.ones((len(GV), 4), np.float32); VC[:, :3] = GC
game.data.color_attributes.new("Col", 'FLOAT_COLOR', 'POINT').data.foreach_set("color_srgb", VC.ravel())
game.data.color_attributes.active_color = game.data.color_attributes["Col"]
sd_q = sd[QUELLE]; ein = np.full(len(GV), 0.004)
for i, p in enumerate(GV):
    if g_hand[i]: ein[i] = 0.0025; continue
    nb = kd_deck.find_n(p, 8); s_ = [sd_q[t[1]] for t in nb if t[2] < 0.03]
    if s_: ein[i] = float(np.clip(0.004 - min(np.median(s_), 0.0), 0.004, 0.02))
game.data.vertices.foreach_set("co", (GV - GN * ein[:, None]).astype(np.float32).ravel()); game.data.update()
rep["fuellmesh"]["einrueckung_mm"] = dict(median=round(float(np.median(ein)) * 1000, 1), max=round(float(ein.max()) * 1000, 1))

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
rep["finger_entspannt"] = {s: not info["hand_" + s]["verschmolzen"] for s in "LR"}

# ======================================================================= 6) Animationen von Carlos (Walk/Run/Idle ersetzt danach gang.py)
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
SW.schreiben(f"{OUT}/{ID}_splats.ply", props, a)                       # volle SH der Scan-Splats bleibt erhalten
W = W[:, [joints.index(n_) for n_ in glb_joints]]
top = np.argsort(-W, axis=1)[:, :4]; tw = np.take_along_axis(W, top, 1); tw = tw / np.maximum(tw.sum(1, keepdims=True), 1e-6)
q8 = np.floor(tw * 255).astype(np.int32); q8[:, 0] += 255 - q8.sum(1)
np.concatenate([top.astype(np.uint8), q8.astype(np.uint8)], 1).tofile(f"{OUT}/{ID}_skin.bin")
json.dump({"count": int(len(a)), "joints": glb_joints, "parts": parts}, open(f"{OUT}/{ID}_skin.json", "w"))
rep["splats"] = dict(anzahl=int(len(a)), teile_scan_hand_luecken=parts, mb=round(len(a) * len(props) * 4 / 1e6, 1))

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
for s in ("L", "R"):
    w = info["hand_" + s]["handgelenk"]; c = (w[0], -3, w[2] - 0.08)
    foto(f"{OUT}/_test/hand_{s}.png", c, (math.pi / 2, 0, 0), 0.3 * k)
Wd, Hd, cols = 300, 400, 6; rows_ = (len(bilder) + cols - 1) // cols; buf = np.ones((rows_ * Hd, cols * Wd, 4), np.float32) * 0.1
for i, pth in enumerate(bilder):
    im = bpy.data.images.load(pth); iw, ih = im.size; px = np.empty(iw * ih * 4, np.float32); im.pixels.foreach_get(px); px = px.reshape(ih, iw, 4)
    rr, cc = divmod(i, cols); y0 = (rows_ - 1 - rr) * Hd; buf[y0:y0 + ih, cc * Wd:cc * Wd + iw] = px[:Hd, :Wd]
    bpy.data.images.remove(im)
o = bpy.data.images.new("sheet", cols * Wd, rows_ * Hd); o.pixels.foreach_set(buf.ravel()); o.filepath_raw = f"{OUT}/_test/{ID}_test.png"; o.file_format = 'PNG'; o.save()
bpy.data.images.remove(o)
print("REPORT", json.dumps(rep))

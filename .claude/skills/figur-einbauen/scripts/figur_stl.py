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
AUSSEN = float(arg("--aussen", 0.012))       # Scan-Splats weiter außerhalb der STL-Fläche = Schwebeteile/Fransen
# STL-Körper (Standard seit Nutzer-Vorgabe 2026-11-19): Geometrie ganz aus dem STL, Scan nur als Farbe/Oberfläche, wo er auf dem STL liegt;
# alles daneben ist Fremdpunkt, jede nicht gedeckte STL-Stelle wird gefüllt. --scan_koerper = alte, lockere Regeln.
STRENG = "--scan_koerper" not in argv
LUECKE = float(arg("--luecke", 0.006 if STRENG else 0.02))        # kein deckender Scan-Splat in diesem Umkreis = Loch
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
            plan += [("oa." + s, ["ua." + s, "hd." + s], "ua." + s, 75), ("ua." + s, ["hd." + s], "hd." + s, 60),   # figuro hat teils T-Pose, Scan A-Pose
                     ("os." + s, ["us." + s, "fu." + s], "us." + s, 15), ("us." + s, ["fu." + s], "fu." + s, 15)]
        piv = {g_: np.array(p, float) for g_, p in piv.items()}
        for g, kinder, ende, maxw in plan:
            if g[:3] in ("oa.", "ua.", "hd.") and g[-1] in VORAB and math.degrees(VORAB[g[-1]][2]) > 20:
                continue                                                 # Arm schon per Hand-Ziel gedreht; ICP fände hier Mantel statt Arm
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

VORAB = {}
def rodrigues(p, sh, ax, ang):
    X = np.asarray(p) - sh; c_, s2_ = math.cos(ang), math.sin(ang)
    return sh + X * c_ + np.cross(ax, X) * s2_ + ax * (X @ ax) * (1 - c_)
def arme_vorab(V):
    """Grobe Arm-Haltung vorab angleichen (figuro hat teils T-Pose, der Scan A-Pose): ganzer Arm um das Schultergelenk auf die
    Richtung des Scan-Arms drehen, an der Schulter weich überblendet. Sonst bekommt der STL-Oberarm keine Scan-Punkte zugeordnet."""
    hi.data.vertices.foreach_set("co", V.ravel().astype(np.float32)); hi.data.update(); E_ = kanten(hi)
    zmin = V[:, 2].min(); Vz = V - [0, 0, zmin]
    Jt, _, _ = SK.anpassen(Vz, float(Vz[:, 2].max()), MENSCH, CT, lambda *a: None, kanten=None)
    J_ = {key: np.array(v) + [0, 0, zmin] for key, v in Jt.items()}
    _, lab = gruppen_labels(V, J_); Hs = float(ZIEL[:, 2].max())
    for s_, sg in (("L", 1), ("R", -1)):
        sh = J_[("upper_arm." + s_, "head")]
        arm = np.isin(lab, [GI["oa." + s_], GI["ua." + s_], GI["hd." + s_]])
        # Ziel = Hand: STL-Hand (Gruppe hd) gegen die äußerste Punktgruppe des Scans auf Hand-/Hüfthöhe (bei weiten Mänteln liegt
        # zwischen Schulter und Hand fast nur Mantel - Punkte "weit weg vom Körper" wären Mantel, nicht Arm)
        hand_v = lab == GI["hd." + s_]
        if hand_v.sum() < 30: continue
        d_v = V[hand_v].mean(0) - sh
        L_arm = float(np.linalg.norm(d_v))
        zone = (np.abs(np.linalg.norm(ZIEL - sh, axis=1) - L_arm) < 0.08) & (sg * ZIEL[:, 0] > 0.05) & (ZIEL[:, 2] < sh[2] - 0.15)
        if zone.sum() < 30: continue
        q_ = sg * ZIEL[zone, 0]; aus = ZIEL[zone][q_ >= np.percentile(q_, 85)]
        d_z = aus.mean(0) - sh
        q_ = Vector(d_v).rotation_difference(Vector(d_z)); ax_, ang_ = q_.to_axis_angle(); w_ = math.degrees(ang_)
        rep.setdefault("arm_vorab_grad", {})[s_] = round(w_, 1)
        if w_ < 8: continue
        # Gewicht entlang der STL-Armachse (wie ein Gelenk): medial der Schulter 0, ab 7 cm außerhalb 1; nur nahe am Arm
        ad = d_v / np.linalg.norm(d_v); tt = (V - sh) @ ad
        radial = np.linalg.norm((V - sh) - np.outer(tt, ad), axis=1)
        wt = np.clip((tt + 0.01) / 0.08, 0, 1); wt = wt * wt * (3 - 2 * wt)
        wt *= np.clip((0.16 * H0 / 1.8 - radial) / (0.04 * H0 / 1.8), 0, 1)   # Rumpf/Hals nicht mitdrehen
        wt = glaetten(wt[:, None], E_, 6, 0.5)[:, 0]
        ax_ = np.array(ax_); X = V - sh                                      # Rodrigues mit Teilwinkel je Punkt (keine Quetschung)
        VORAB[s_] = (sh.copy(), ax_.copy(), float(ang_))
        th = ang_ * wt; c_, s2_ = np.cos(th)[:, None], np.sin(th)[:, None]
        V = sh + X * c_ + np.cross(ax_, X) * s2_ + np.outer(X @ ax_, ax_) * (1 - c_)
    log("Arme vorab", rep.get("arm_vorab_grad"))
    return V
# Arm-Einbettung: Arm-/Hand-/Fingergelenke im Original-STL bestimmen (figuro-T-Pose: Arme frei, sauber erkennbar) und an die
# umgebenden STL-Punkte hängen; nach Vorab-Drehung + Gelenk-Ausrichtung mit denselben Punkten wieder ablesen. Für Figuren, deren
# Arme eng an weiter Kleidung hängen (Carlos: Mantel), findet die Arm-Verfolgung sonst Rumpf statt Arm.
V_T = V.copy()
hi.data.vertices.foreach_set("co", V_T.ravel().astype(np.float32)); hi.data.update()
_zm = V_T[:, 2].min(); _VT = V_T - [0, 0, _zm]
J_T, ROT_T, info_T = SK.anpassen(_VT, float(_VT[:, 2].max()), MENSCH, CT, lambda *a: None, kanten=kanten(hi))
J_T = {key: np.array(v) + [0, 0, _zm] for key, v in J_T.items()}
_kdT = kd_bau(V_T); ARM_KETTE = {}
for (bn, en), pv in J_T.items():
    if not bn.endswith((".L", ".R")) or bn.split(".")[0] in ("thigh", "shin", "foot", "toe", "shoulder"): continue
    if (bn, en) == ("upper_arm." + bn[-1], "head"): continue
    nb = _kdT.find_n(pv, 24); ARM_KETTE[(bn, en)] = np.array([q[1] for q in nb])
V = arme_vorab(V)
V_V = V.copy()                                                         # nach der Vorab-Drehung (für die Arm-Einbettung)
J_V = {}
for kk, pv in J_T.items():
    s_ = kk[0][-1]
    if kk[0].endswith((".L", ".R")) and s_ in VORAB and kk[0].split(".")[0] not in ("thigh", "shin", "foot", "toe", "shoulder") and kk != ("upper_arm." + s_, "head"):
        J_V[kk] = rodrigues(pv, *VORAB[s_])                              # Gelenke liegen auf der Armachse -> volle Drehung
    else: J_V[kk] = pv
V = gelenkweise(V)

def anschmiegen(V, max_aus=0.05, max_ein=0.02, n_glatt=60):
    """STL-Fläche entlang der Normalen an den Scan schmiegen: figuro schätzt Proportionen (bv11: Schultern/Rumpf schmaler als der
    echte Körper -> 12 % der Rumpf-Splats lagen > 2 cm außerhalb und wurden weggeschnitten = zu enge Schultern). Ziele nur aus
    derselben Körpergruppe (Rumpf zieht nie zum Arm -> Arme bleiben frei), Hände/Finger bleiben wie im STL. Verschiebung an jedem
    6. Punkt gemessen (Median der Normalabstände der nahen Scan-Punkte), über die Kanten gewichtet verteilt und geglättet."""
    hi.data.vertices.foreach_set("co", V.ravel().astype(np.float32)); hi.data.update()
    Nv = np.zeros(len(V) * 3); hi.data.vertices.foreach_get("normal", Nv); Nv = Nv.reshape(-1, 3)
    zmin = V[:, 2].min(); Vz = V - [0, 0, zmin]
    Jt, _, _ = SK.anpassen(Vz, float(Vz[:, 2].max()), MENSCH, CT, lambda *a: None, kanten=None)
    J = {key: np.array(v) + [0, 0, zmin] for key, v in Jt.items()}
    _, lab_v = gruppen_labels(V, J); _, lab_z = gruppen_labels(ZIEL, J)
    val = np.zeros((len(V), 2)); stich = np.arange(len(V)) % 6 == 0
    for g in GNAMEN:
        if g.startswith("hd."): continue
        mz = np.where(lab_z == GI[g])[0]
        if len(mz) < 30: continue
        kd_g = kd_bau(ZIEL[mz])
        for i in np.where(stich & (lab_v == GI[g]))[0]:
            nb = kd_g.find_range(V[i], 0.06)
            if len(nb) < 3: continue
            Q = np.array([t[0] for t in nb]) - V[i]; dn = Q @ Nv[i]; lat = np.linalg.norm(Q - np.outer(dn, Nv[i]), axis=1)
            m = lat < 0.015
            if m.sum() < 3: continue
            val[i] = (float(np.clip(np.median(dn[m]), -max_ein, max_aus)), 1.0)
    hand = np.isin(lab_v, [GI["hd.L"], GI["hd.R"]])
    val[:, 0] *= val[:, 1]
    sm = glaetten(val, EK0, n_glatt, 0.6)
    disp = np.where(sm[:, 1] > 1e-4, sm[:, 0] / np.maximum(sm[:, 1], 1e-4), 0.0)
    disp *= np.clip(sm[:, 1] / 0.05, 0, 1)                             # weit weg von jeder Messung: auslaufen lassen
    disp[hand] = 0.0
    arm = np.isin(lab_v, [GI[g_] for g_ in GNAMEN if g_.startswith(("oa.", "ua."))])
    disp[arm] = np.clip(disp[arm], -0.01, 0.025)                        # Arme: STL-Form behalten (Scan dort fleckig -> Beulen)
    disp = glaetten(disp[:, None], EK0, 30, 0.5)[:, 0]                 # weich, keine Beulen (Arm wirkte wie Perlenkette)
    disp[hand] *= 0.0
    rep["anschmiegen_cm"] = dict(median=round(float(np.median(disp)) * 100, 2), p95=round(float(np.percentile(disp, 95)) * 100, 2),
                                 p5=round(float(np.percentile(disp, 5)) * 100, 2))
    log("Anschmiegen an den Scan", rep["anschmiegen_cm"])
    return V + Nv * disp[:, None]
if "--ohne_anschmiegen" not in argv and not STRENG:
    EK0 = kanten(hi)
    V = anschmiegen(V)
bvh0 =BVHTree.FromPolygons(V.tolist(), [p.vertices[:] for p in hi.data.polygons])
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

# ======================================================================= 1b) Scan-Form (Nutzer-Entscheidung 2026-11-19 abends)
# Körperform, Skelett und Textur wie bei LUCA 1 aus dem Scan; das (ausgerichtete) STL nur dort, wo der Scan keine Punkte hat
# (Löcher, dünne Unterarme) und für die Hände (Scan-Hände sind Klumpen). figuro schätzt Proportionen (MO wurde zum Muskelprotz).
SCAN_FORM = "--scan_form" in argv
if SCAN_FORM and "--luecke" not in argv: LUECKE = 0.012                  # nur echte Löcher füllen, Scan-Textur nicht übermalen
def huelle_aus_punkten(Pts, vox=0.004, K=2):
    lo = Pts.min(0) - (K + 4) * vox; ijk = np.floor((Pts - lo) / vox).astype(np.int64); dims = ijk.max(0) + K + 6
    occ = np.zeros(dims, bool); occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    def weiten(g, k_):
        g = g.copy()
        for ax in range(3):
            h_ = g.copy()
            for s_ in range(1, k_ + 1): h_ |= np.roll(g, s_, ax) | np.roll(g, -s_, ax)
            g = h_
        return g
    dick = weiten(occ, K); aussen = np.zeros(dims, bool)
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
    return np.stack(np.unravel_index(uq, dims + 1), 1) * vox + lo, inv.reshape(-1, 4)
if SCAN_FORM:
    log("1b Scan-Form: Hülle aus dem Scan, STL nur für Löcher und Hände")
    Jst, _, _ = SK.anpassen(V, H, MENSCH, CT, lambda *a: None, kanten=None)
    _, lab_st = gruppen_labels(V, Jst)
    stl_hand = np.isin(lab_st, [GI["hd.L"], GI["hd.R"]])
    V_STL = V.copy(); kd_stl = kd_bau(V_STL[::2])
    Qi = np.where((FOP > 0.25) & (FG < 0.03))[0]; Q = FP[Qi]; QC = FC[Qi]
    _qi, dqs = kd_nn(kd_stl, Q); keepq = ~((Q[:, 2] < 0.04) & (dqs > 0.03))             # Bodenreflex-Streifen an den Füßen
    # Arme neu modelliert (Nutzer): Unterarme, Hände und der nackte Teil der Oberarme kommen aus dem STL-Arm (glatt, richtige Form);
    # der Scan liefert dort nur noch die Farbe. Der T-Shirt-Ärmel (nicht hautfarben) bleibt Scan.
    _labq = lab_st[::2][_qi]
    _haut_q = (QC.mean(1) > 0.16) & (QC[:, 0] - QC[:, 2] > 0.1) & (QC[:, 0] >= QC[:, 1])
    _arm_q = np.isin(_labq, [GI["ua.L"], GI["ua.R"], GI["hd.L"], GI["hd.R"]]) | (np.isin(_labq, [GI["oa.L"], GI["oa.R"]]) & _haut_q)
    Q = Q[keepq & ~_arm_q]; rep["scan_form_arm_punkte_ersetzt"] = int((keepq & _arm_q).sum())
    stl_arm = np.isin(lab_st, [GI["ua.L"], GI["ua.R"], GI["oa.L"], GI["oa.R"]])
    kd_h = kd_bau(V[stl_hand]); _, dh = kd_nn(kd_h, Q); Q = Q[dh > 0.035]            # Scan-Handklumpen weg
    sub = np.arange(len(V)) % 3 == 0                                                  # STL-Punkte (2 mm) ausgedünnt prüfen
    kd_q = kd_bau(Q); _, dq = kd_nn(kd_q, V[sub])
    luecke_stl = np.zeros(len(V), bool); luecke_stl[np.where(sub)[0]] = dq > 0.015
    extra = V[luecke_stl | stl_hand | stl_arm]
    rep["scan_form"] = dict(scan_punkte=int(len(Q)), stl_luecken_punkte=int(luecke_stl.sum()), stl_hand_punkte=int(stl_hand.sum()))
    vv, ff = huelle_aus_punkten(np.concatenate([Q, extra]))
    me_n = bpy.data.meshes.new(ID + "_koerper_scan"); me_n.from_pydata(vv.tolist(), [], ff.tolist())
    alt_me = hi.data; hi.data = me_n; aktiv(hi)
    rm = hi.modifiers.new("remesh", 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = 0.004; rm.adaptivity = 0
    bpy.ops.object.modifier_apply(modifier="remesh")
    sm_ = hi.modifiers.new("glatt", 'LAPLACIANSMOOTH'); sm_.lambda_factor = 0.6; sm_.iterations = 10; sm_.use_volume_preserve = True
    bpy.ops.object.modifier_apply(modifier="glatt")
    bm = bmesh.new(); bm.from_mesh(hi.data); bm.verts.ensure_lookup_table()
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
    V = verts(hi); H = float(V[:, 2].max()); k = H / 1.8
    rep["koerper"] = dict(zahlen(hi.data), groesse_m=round(H, 3)); log("Scan-Hülle", rep["koerper"], rep["scan_form"])

# ======================================================================= 2) Skelett auf dem STL
log("2 Skelett")
EK = kanten(hi)
J, ROT, info = SK.anpassen(V, H, MENSCH, CT, log, kanten=EK); info["hi_punkte"] = V
if max(rep.get("arm_vorab_grad", {}).values() or [0]) > 20 and "--ohne_einbettung" not in argv and not SCAN_FORM:
    # starre Abbildung je Arm-Segment (Oberarm/Unterarm/Hand): dieselben STL-Punkte in T-Pose und jetzt -> Kabsch
    _, lab_T = gruppen_labels(V_T, J_T)
    def kabsch(ix):
        A_, B_ = V_V[ix], V[ix]; ca, cb = A_.mean(0), B_.mean(0)
        U_, _, Vt_ = np.linalg.svd((B_ - cb).T @ (A_ - ca)); Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U_ @ Vt_))
        R_ = U_ @ Dm @ Vt_; return lambda p: R_ @ (np.asarray(p) - ca) + cb, R_
    for s_ in "LR":
        T_ = {}
        SEG = {"oa": ("upper_arm." + s_, "head", "upper_arm." + s_, "tail"), "ua": ("forearm." + s_, "head", "forearm." + s_, "tail"),
               "hd": ("hand." + s_, "head", "f_middle.03." + s_, "tail")}
        for g, (b0, e0, b1, e1) in SEG.items():
            a0, a1 = J_T[(b0, e0)], J_T[(b1, e1)]; ab = a1 - a0; tt = ((V_T - a0) @ ab) / max(ab @ ab, 1e-9)
            rad = np.linalg.norm(V_T - (a0 + np.outer(tt, ab)), axis=1)
            ix = np.where((lab_T == GI[f"{g}.{s_}"]) & (tt > 0.15) & (tt < 0.95) & (rad < 0.08))[0]   # nur echter Arm (kein Mantel an der Schulter)
            if len(ix) > 50: T_[g] = kabsch(ix)
        rep.setdefault("arm_segment_punkte", {})[s_] = {g: int(((lab_T == GI[f"{g}.{s_}"])).sum()) for g in SEG}
        if len(T_) < 3: continue
        mit = lambda p, a, b: 0.5 * (T_[a][0](p) + T_[b][0](p))
        J[("upper_arm." + s_, "tail")] = J[("forearm." + s_, "head")] = Vector(mit(J_V[("upper_arm." + s_, "tail")], "oa", "ua"))
        J[("forearm." + s_, "tail")] = J[("hand." + s_, "head")] = Vector(mit(J_V[("forearm." + s_, "tail")], "ua", "hd"))
        Rh = T_["hd"][1] @ (np.array(Matrix.Rotation(VORAB[s_][2], 3, Vector(VORAB[s_][1]))) if s_ in VORAB else np.eye(3))   # T-Pose -> Vorab -> jetzt
        for kk in list(J_T.keys()):
            bn = kk[0]
            if bn.endswith("." + s_) and bn.split(".")[0] in ("palm", "f_index", "f_middle", "f_ring", "f_pinky", "thumb") or kk == ("hand." + s_, "tail"):
                J[kk] = Vector(T_["hd"][0](J_V[kk]))
        for bn_ in [b for b in ROT_T if b.endswith("." + s_)]:
            r_ = ROT_T[bn_]
            if isinstance(r_, tuple) and len(r_) == 2 and np.ndim(r_[1]) == 1: ROT[bn_] = (r_[0], (Rh @ np.array(r_[1])).tolist())
        if ("hand_" + s_) in info_T:
            hT = info_T["hand_" + s_]
            info["hand_" + s_] = dict(hT, handgelenk=list(J[("hand." + s_, "head")]), richtung=(Rh @ np.array(hT["richtung"])).tolist())
            for nm_ in ("handflaeche_", "daumenseite_"):
                if nm_ + s_ in info_T: info[nm_ + s_] = (Rh @ np.array(info_T[nm_ + s_])).tolist()
        rep.setdefault("arm_einbettung_handgelenk", {})[s_] = [round(v, 3) for v in J[("hand." + s_, "head")]]
    log("Arm-Einbettung (starr je Segment):", rep.get("arm_einbettung_handgelenk"))
    rep["arm_einbettung"] = True
SK_VON = arg("--skelett_von")
if SK_VON:
    # Skelett einer Referenzfigur übernehmen (Nutzer: LUCA 1 und 2 sind dieselbe Person, MO hat ähnliche Proportionen):
    # Knochenlängen, Schulter-/Hüft-Abstände und Wirbelsäule von der Referenz (auf die Körperhöhe skaliert), nur die Richtung
    # jedes Glieds aus der Haltung im Scan. Hände/Finger wandern mit dem Handgelenk mit.
    R9 = json.load(open(os.path.join(REPO, "figuren", SK_VON, "build", SK_VON + "_skelett.json"))); B9 = R9["knochen"]; s9 = H / R9["hoehe"]
    p9 = lambda n, e: np.array(B9[n][e]); Jn = {kk: np.array(v) for kk, v in J.items()}; alt_J = dict(Jn)
    def richtung(n): d = Jn[(n, "tail")] - Jn[(n, "head")]; return d / max(np.linalg.norm(d), 1e-9)
    def laenge(n): return float(np.linalg.norm(p9(n, "tail") - p9(n, "head"))) * s9
    def kette(namen, start):
        cur = start
        for n in namen:
            Jn[(n, "head")] = cur; cur = cur + richtung(n) * laenge(n); Jn[(n, "tail")] = cur
        return cur
    kette(SPINE_NAMEN := ["spine", "spine.001", "spine.002", "spine.003", "spine.004", "spine.005", "spine.006"], Jn[("spine", "head")])
    for s_ in "LR":
        c7 = Jn[("spine.003", "tail")]
        Jn[("shoulder." + s_, "head")] = c7 + s9 * (p9("shoulder." + s_, "head") - p9("spine.003", "tail"))
        sh_ = c7 + s9 * (p9("upper_arm." + s_, "head") - p9("spine.003", "tail")); Jn[("shoulder." + s_, "tail")] = sh_
        w_alt = alt_J[("hand." + s_, "head")]
        w_neu = kette(["upper_arm." + s_, "forearm." + s_], sh_)
        dv = w_neu - w_alt
        for (bn, en) in list(Jn.keys()):
            if bn.endswith("." + s_) and bn.split(".")[0] in ("hand", "palm", "f_index", "f_middle", "f_ring", "f_pinky", "thumb"): Jn[(bn, en)] = alt_J[(bn, en)] + dv
        Jn[("hand." + s_, "head")] = w_neu
        hip_ = Jn[("spine", "head")] + s9 * (p9("thigh." + s_, "head") - p9("spine", "head"))
        a_alt = alt_J[("foot." + s_, "head")]
        a_neu = kette(["thigh." + s_, "shin." + s_], hip_)
        Jn[("foot." + s_, "head")] = a_neu; Jn[("foot." + s_, "tail")] = alt_J[("foot." + s_, "tail")] + (a_neu - a_alt)
        Jn[("toe." + s_, "head")] = Jn[("foot." + s_, "tail")]; Jn[("toe." + s_, "tail")] = alt_J[("toe." + s_, "tail")] + (a_neu - a_alt)
        if ("hand_" + s_) in info: info["hand_" + s_] = dict(info["hand_" + s_], handgelenk=w_neu.tolist())
        rep.setdefault("skelett_von", {})[s_] = dict(handgelenk_cm=round(float(np.linalg.norm(dv)) * 100, 1), knoechel_cm=round(float(np.linalg.norm(a_neu - a_alt)) * 100, 1))
    J = {kk: Vector(v) for kk, v in Jn.items()}
    log("Skelett von", SK_VON, "übernommen (Maßstab", round(s9, 3), "), Verschiebung:", rep["skelett_von"])
    # Körper ans übernommene Skelett anpassen ("modellier den Rest selbst"): Arm-/Beinpunkte wandern entlang ihres Glieds mit der
    # Verschiebung der nächsten Gelenke (stückweise linear), an Schulter/Hüfte weich ausgeblendet. Hülle UND Scan-Splats.
    _, labA = gruppen_labels(V, alt_J); _, labS = gruppen_labels(FP, alt_J)
    def glied_verschieben(P, lab, s_):
        out = np.zeros_like(P)
        for gruppen, kette_ in ((("oa", "ua", "hd"), [("upper_arm." + s_, "head"), ("upper_arm." + s_, "tail"), ("forearm." + s_, "tail"), ("f_middle.03." + s_, "tail")]),
                                (("os", "us", "fu"), [("thigh." + s_, "head"), ("thigh." + s_, "tail"), ("shin." + s_, "tail"), ("toe." + s_, "tail")])):
            m = np.isin(lab, [GI[f"{g}.{s_}"] for g in gruppen])
            if not m.any(): continue
            A_ = np.array([alt_J[kk] for kk in kette_]); D_ = np.array([Jn[kk] - alt_J[kk] for kk in kette_])
            Pm = P[m]; best = np.full(len(Pm), 9.0); disp = np.zeros_like(Pm)
            for i in range(len(A_) - 1):
                ab = A_[i + 1] - A_[i]; t_ = np.clip(((Pm - A_[i]) @ ab) / max(ab @ ab, 1e-9), 0, 1)
                d_ = np.linalg.norm(Pm - (A_[i] + t_[:, None] * ab), axis=1); bes = d_ < best
                disp[bes] = (D_[i] * (1 - t_[bes, None]) + D_[i + 1] * t_[bes, None]); best[bes] = d_[bes]
                if i == 0:                                                     # an Schulter/Hüfte weich zum Rumpf
                    w0 = np.clip(t_ / 0.25, 0, 1); w0 = w0 * w0 * (3 - 2 * w0); disp[bes] *= w0[bes, None]
            out[m] = disp
        return out
    dV = sum(glied_verschieben(V, labA, s_) for s_ in "LR"); dS = sum(glied_verschieben(FP, labS, s_) for s_ in "LR")
    V = V + dV; hi.data.vertices.foreach_set("co", V.ravel().astype(np.float32)); hi.data.update()
    FP = FP + dS; fa[:, SW.spalten(fprops, "x", "y", "z")] = np.stack([FP[:, 0], FP[:, 2], -FP[:, 1]], 1) if False else fa[:, SW.spalten(fprops, "x", "y", "z")] + dS
    rep["koerper_angepasst_cm"] = dict(max=round(float(np.linalg.norm(dV, axis=1).max()) * 100, 1), splats_bewegt=int((np.linalg.norm(dS, axis=1) > 0.005).sum()))
    log("Körper ans Skelett angepasst", rep["koerper_angepasst_cm"])
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
if rep.get("arm_einbettung") and "lab_T" in globals():                 # Füll-Mesh: Arm/Rumpf ebenfalls aus der freien T-Pose trennen
    _kdV = kd_bau(V); _gv = verts(game); _gl = np.array([lab_T[_kdV.find(p_)[1]] for p_ in _gv])
    _vg = {g.name: g.index for g in game.vertex_groups}; _korr = 0
    for s_ in "LR":
        _arm_n = {n_ for n_ in MENSCH if n_.endswith("." + s_) and not n_.startswith(("thigh", "shin", "foot", "toe", "shoulder"))}
        _rumpf_n = {n_ for n_ in MENSCH if n_.startswith("spine")} | {"shoulder." + s_}
        h_, t_ = np.array(J[("upper_arm." + s_, "head")]), np.array(J[("upper_arm." + s_, "tail")]); ab_ = t_ - h_
        armv = np.isin(_gl, [GI["oa." + s_], GI["ua." + s_], GI["hd." + s_]]) & ((((_gv - h_) @ ab_) / (ab_ @ ab_)) > 0.15)
        rumpfv = ~np.isin(_gl, [GI[g_] for g_ in GNAMEN if g_.startswith(("oa.", "ua.", "hd."))]) & (_gv[:, 2] < h_[2] - 0.04)
        for v in game.data.vertices:
            weg_ = _rumpf_n if armv[v.index] else (_arm_n if rumpfv[v.index] else None)
            if not weg_: continue
            rest_ = sum(g.weight for g in v.groups if game.vertex_groups[g.group].name not in weg_ and game.vertex_groups[g.group].name in MENSCH)
            if rest_ < 0.05: game.vertex_groups["upper_arm." + s_ if armv[v.index] else "spine.002"].add([v.index], 1.0, 'REPLACE')
            for g in v.groups:
                if game.vertex_groups[g.group].name in weg_ and g.weight > 0: g.weight = 0.0; _korr += 1
    rep["fuellmesh"]["arm_rumpf_tpose"] = _korr

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
STL_HAENDE = "--stl_haende" in argv          # alt: Scan-Hände durch hautfarbene STL-Hand-Splats ersetzen (wirkte wie rosa Fell)
if STL_HAENDE:
    for s in "LR": haende |= distal(FP, s) & np.isin(reg_s, [RI["hand." + s], RI["unterarm." + s]])
# Bodenreflexe (oranger Flurboden) an Armen, Rumpf, Beinen: stark gesättigtes Orange (Haut ~0,4 Sättigung, Boden > 0,6)
mx_, mn_ = FC.max(1), FC.min(1); sat = (mx_ - mn_) / np.maximum(mx_, 1e-6)
orange = ((FC[:, 0] >= FC[:, 1]) & (FC[:, 1] >= FC[:, 2]) & (sat > 0.6) & (FC[:, 0] > 0.5)
          & ((FC[:, 1] - FC[:, 2]) / np.maximum(FC[:, 0] - FC[:, 2], 1e-6) > 0.15)
          & ~np.isin(reg_s, [RI["kopf"], RI["hals"], RI["fuss.L"], RI["fuss.R"]]))
_gr = FC[:, 1] / np.maximum(FC[:, 0], 1e-6)                         # Boden ist roter als Haut/Stiefel (g/r klein)
orange |= (np.isin(reg_s, [RI["fuss.L"], RI["fuss.R"]]) & (FP[:, 2] < 0.05) & (sat > 0.45) & (FC[:, 0] > 0.3) & (_gr < 0.6))   # Sohlen-Halo
orange |= (np.isin(reg_s, [RI["unterschenkel.L"], RI["unterschenkel.R"]]) & (sat > 0.45) & (FC[:, 0] > 0.35) & (_gr < 0.6))     # Flecken auf der Jeans
if "--ohne_orange" in argv: orange[:] = False
# heller Saum unter den Haaren am Hinterkopf (Gegenlicht-Halo, z. B. bv10): hinter dem Kopfgelenk (Atlas), zwischen C7 und
# Ohrhöhe, hell und warm – die Lücke wird danach mit Haar-/Halsfarbe derselben Region gefüllt
saum = np.zeros(len(FP), bool)
if "--saum_hinterkopf" in argv:
    at_, c7_ = J[("spine.005", "tail")], J[("spine.004", "head")]
    saum = ((FP[:, 1] > at_.y - 0.01 * k) & (FP[:, 2] > c7_.z - 0.02 * k) & (FP[:, 2] < at_.z + 0.06 * k) & (np.abs(FP[:, 0] - at_.x) < 0.15 * k)
            & (FC.mean(1) > 0.38) & ((FC[:, 0] - FC[:, 2]) > 0.2))
# Haut (Unterarme + Hände): im Scan dünn, fransig, Hände als Klumpen -> Fläche kommt ganz aus dem STL, Farbe von den Scan-Splats
HAUT_NEU = "--scan_haut" not in argv
_lum = FC.mean(1)
haut_farbe = (_lum > 0.16) & (FC[:, 0] - FC[:, 2] > 0.1) & (FC[:, 0] >= FC[:, 1]) & (FC[:, 1] >= FC[:, 2] - 0.02)
# Unterarm nur dann als Haut neu bauen, wenn er im Scan überwiegend hautfarben ist (Mantel-/Jackenärmel bleiben Scan);
# Hände immer aus dem STL, Handschuhe (nicht hautfarben) behalten aber ihre Scan-Farbe
HAUT_REG, HANDSCHUH = [], set()
for r_ in ("unterarm.L", "unterarm.R", "hand.L", "hand.R"):
    if r_ not in RI: continue
    m_ = (reg_s == RI[r_]) & (FOP > 0.3); fr_ = float(haut_farbe[m_].mean()) if m_.sum() > 30 else 0.0
    rep.setdefault("hautanteil", {})[r_] = round(fr_, 2)
    if r_.startswith("hand"):
        HAUT_REG.append(RI[r_])
        if fr_ < 0.4: HANDSCHUH.add(RI[r_])
    elif fr_ > 0.5: HAUT_REG.append(RI[r_])
haut_s = np.isin(reg_s, HAUT_REG) if HAUT_NEU else np.zeros(len(FP), bool)
# nackte Oberarme unter kurzen Ärmeln (T-Shirt): hautfarbene Splats der Oberarm-Region gehören auch zur STL-Haut
OA_REG = [RI[r] for r in ("oberarm.L", "oberarm.R") if r in RI]
if HAUT_NEU: haut_s |= np.isin(reg_s, OA_REG) & haut_farbe
haut_quelle = np.where(haut_s & (FOP > 0.3) & ~orange & (np.abs(sd) < 0.025) & (FG < 0.02))[0]
if len(haut_quelle) > 50:                                              # dunkle Sprenkel (Schatten, Lücken, Armhaare) nicht als Hautfarbe
    _lq = FC[haut_quelle].mean(1); haut_quelle = haut_quelle[_lq >= np.percentile(_lq, 35)]
# Achsel-Lappen: Scan-Splats unter der Schulter, halb Arm halb Rumpf gewichtet und außerhalb der STL-Fläche -> dehnen sich bei
# jeder Armbewegung zu Stofflappen zwischen Arm und Rumpf
_armw = np.zeros(len(FP))
for s_ in "LR":
    _cols = [jpos[n_] for n_ in joints if n_.endswith("." + s_) and not n_.startswith(("thigh", "shin", "foot", "toe", "shoulder"))]
    _armw = np.maximum(_armw, Ws[:, _cols].sum(1))
ACHSEL = (_armw > 0.15) & (_armw < 0.85) & (FP[:, 2] < J[("upper_arm.L", "head")].z - 0.05 * k) & ~haut_farbe & (sd > 0.004)
# Hose unter einem Mantel liegt "tief im STL" (das STL ist die Mantelfläche) -> nahe an den Beinen unter der Hüfte nicht löschen
_db = np.full(len(FP), 9.0)
for s_ in "LR":
    for bn in ("thigh." + s_, "shin." + s_):
        a0, a1 = np.array(J[(bn, "head")]), np.array(J[(bn, "tail")]); ab = a1 - a0; t_ = np.clip(((FP - a0) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        _db = np.minimum(_db, np.linalg.norm(FP - (a0 + t_[:, None] * ab), axis=1))
HOSE_INNEN = (_db < 0.11) & (FP[:, 2] < J[("thigh.L", "head")].z + 0.02)
BODEN_STREIFEN = np.zeros(len(FP), bool)
if SCAN_FORM:
    _low = np.where(FP[:, 2] < 0.04)[0]; _, _dl = kd_nn(kd_stl, FP[_low]); BODEN_STREIFEN[_low[_dl > 0.03]] = True
# vereinzelte Splats (Schwebepunkte): weniger als 4 Nachbarn im Umkreis von 1,5 cm
# Außen-Grenze je Körperregion, mit Splat-Ausdehnung (1,5 sigma): Haar 3 cm, Stoff 2,2 cm, Beine 1,5 cm, neu gebaute Haut 0,8 cm
GRENZE = {"kopf": 0.03, "hals": 0.02, "rumpf": 0.022, "becken": 0.022, "oberarm": 0.022, "unterarm": 0.022, "hand": 0.02,
          "oberschenkel": 0.015, "unterschenkel": 0.015, "fuss": 0.015}
grenz = np.array([GRENZE.get(RNAMEN[r].split(".")[0], 0.02) for r in range(len(RNAMEN))])[reg_s]
grenz[haut_s] = 0.008
if STRENG:                                                             # nur was auf der STL-Fläche liegt (Haar darf Volumen haben)
    grenz = np.minimum(grenz, np.where(np.isin(reg_s, [RI["kopf"]]), 0.025, np.where(np.isin(reg_s, [RI["hals"]]), 0.012, 0.015 if SCAN_FORM else 0.009)))
if "--aussen" in argv: grenz = np.maximum(grenz, AUSSEN)
aussen_m = sd + 1.5 * FG > grenz
_sc3 = np.exp(fa[:, SW.spalten(fprops, "scale_0", "scale_1", "scale_2")])
nadel = (_sc3.max(1) / np.maximum(_sc3.min(1), 1e-6) > 12) & (_sc3.max(1) > 0.008)
einzeln = np.zeros(len(FP), bool)                                      # Dichte-Filter läuft unten iterativ auf den Behaltenen
gruende = [("schrott_durchsichtig", FOP < 0.04), ("haende_ersetzt", haende), ("haut_aus_stl", haut_s), ("aussen_regional", aussen_m), ("nadeln", nadel),
           ("fransen_halbtransparent", (sd > 0.005) & (FOP < 0.3)), ("gross_aussen", (FG > 0.012) & (sd > 0.006)), ("tief_innen", (sd < -0.08) | ((sd < -0.02) & STRENG & ~HOSE_INNEN)),
           ("vereinzelt", einzeln),
           ("geisterarm_flanke", np.isin(reg_s, [RI["rumpf"], RI["becken"]]) & haut_farbe & (FP[:, 2] < J[("upper_arm.L", "head")].z - 0.06 * k)),
           ("achsel_lappen", ACHSEL),
           ("boden_streifen", BODEN_STREIFEN),
           ("kopf_blobs", np.isin(reg_s, [RI["kopf"]]) & ((FG > (0.008 if STRENG else 0.015)) | ((FG > 0.009) & (FOP < 0.6)))),   # STL füllt das Haar   # Hinterkopf: große halbdurchsichtige Kleckse
           ("bodenreflex_orange", orange), ("saum_hinterkopf", saum)]
weg = np.zeros(len(FP), bool); rep["scan_entfernt"] = {}
for name, m in gruende:
    rep["scan_entfernt"][name] = int((m & ~weg).sum()); weg |= m
behalten = ~weg
for _rd in range(2):                                                   # Schwebepunkte: deckungsgewichtete Dichte nur unter Behaltenen
    _idx = np.where(behalten & (FOP > 0.04))[0]; _kd = kd_bau(FP[_idx]); _arm = np.zeros(len(_idx), bool)
    for _j, _i in enumerate(_idx):
        _arm[_j] = sum(FOP[_idx[q[1]]] for q in _kd.find_range(FP[_i], 0.015)) < 2.5
    behalten[_idx[_arm]] = False; rep["scan_entfernt"]["duenn_" + str(_rd)] = int(_arm.sum())
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
        if name == "hals" and "kopf" in kd_reg: name = "kopf"            # Hals hat zu wenig eigene Haut-Splats (Kragen mischt dunkel)
        if name not in kd_reg:                                       # Region ohne Scan-Farbe: Nachbarregion entlang der Kette
            fehlt += len(sel); src, kd_ = None, None
            for alt in (name.replace("unterarm", "oberarm"), name.replace("hand", "unterarm"), name.replace("fuss", "unterschenkel"), "rumpf"):
                if alt in kd_reg: src, kd_ = kd_reg[alt]; break
        else: src, kd_ = kd_reg[name]
        for i in sel:                                                   # nächste Schale (bis 1,5 cm hinter dem nächsten Splat)
            nb = kd_.find_n(Pq[i], 16); dd = np.array([t[2] for t in nb]); jj = np.array([t[1] for t in nb])
            nah = dd < dd.min() + 0.015
            if nah.sum() >= 5: jj = jj[nah]
            cl = haut_farbe[src[jj]]; maj = cl.mean() > 0.5                # nur aus der Mehrheitsklasse (Haut ODER Stoff)
            if (cl == maj).sum() >= 3: jj = jj[cl == maj]
            out[i] = 0.5 * np.median(FC[src[jj]], 0) + 0.5 * FC[src[jj[rng.integers(len(jj))]]]   # Textur statt Flachfarbe
    return out, fehlt

# Dreiecke des STL, Region je Dreieck
me = hi.data; me.calc_loop_triangles(); nt = len(me.loop_triangles)
tv = np.zeros(nt * 3, np.int64); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
A = V[tv]; area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
Wv = gewichte(V); reg_v = regionen(Wv)
reg_t = reg_v[tv[:, 0]]; cent = A.mean(1)
hand_t = np.zeros(nt, bool); hand_seite = np.full(nt, "", object)
for s in ("LR" if STL_HAENDE else ""):
    m = distal(cent, s) & np.isin(reg_t, [RI["hand." + s], RI["unterarm." + s]]); hand_t |= m; hand_seite[m] = s
haut_t = np.isin(reg_t, HAUT_REG) if HAUT_NEU else np.zeros(nt, bool)
if HAUT_NEU:                                                           # Oberarm-Dreiecke, deren Scan-Umgebung hautfarben ist
    _oa = np.where(np.isin(reg_s, OA_REG) & (FOP > 0.3) & (np.abs(sd) < 0.03))[0]
    if len(_oa) > 50:
        _kd = kd_bau(FP[_oa]); _hf = haut_farbe[_oa]
        for t_ in np.where(np.isin(reg_t, OA_REG))[0]:
            nb = _kd.find_n(cent[t_], 10); nb = [q for q in nb if q[2] < 0.03]
            if len(nb) >= 4 and np.mean(_hf[[q[1] for q in nb]]) > 0.5: haut_t[t_] = True
    rep["haut_oberarm_dreiecke"] = int((haut_t & np.isin(reg_t, OA_REG)).sum())
    # Ärmelsaum entlang des Oberarms: Hautanteil der Scan-Splats je 4%-Abschnitt (Schulter -> Ellbogen); unterhalb des Saums ist
    # der ganze Oberarm Haut (sonst bleiben dunkle Schatten-/Ärmelreste als Streifen auf dem Arm)
    rep["aermelsaum_t"] = {}
    for s_ in "LR":
        h_, t_ = np.array(J[("upper_arm." + s_, "head")]), np.array(J[("upper_arm." + s_, "tail")]); ab_ = t_ - h_
        m_ = (reg_s == RI["oberarm." + s_]) & (FOP > 0.3)
        if m_.sum() < 100: continue
        tt_ = ((FP[m_] - h_) @ ab_) / (ab_ @ ab_); hf_ = haut_farbe[m_]
        bins = np.arange(0.0, 1.0, 0.04); frac = np.array([hf_[(tt_ >= b) & (tt_ < b + 0.08)].mean() if ((tt_ >= b) & (tt_ < b + 0.08)).sum() > 15 else np.nan for b in bins])
        ok_ = np.where(frac > 0.5)[0]
        if not len(ok_): continue
        t_saum = None
        for i_ in ok_:                                                     # erster Abschnitt, ab dem es Haut bleibt
            rest_ = frac[i_:]; rest_ = rest_[~np.isnan(rest_)]
            if len(rest_) and (rest_ > 0.4).mean() > 0.8: t_saum = float(bins[i_]); break
        if t_saum is None or t_saum > 0.9: continue
        rep["aermelsaum_t"][s_] = round(t_saum, 2)
        tc = ((cent - h_) @ ab_) / (ab_ @ ab_)
        haut_t |= (reg_t == RI["oberarm." + s_]) & (tc > t_saum + 0.04)
        tf = ((FP - h_) @ ab_) / (ab_ @ ab_)
        haut_s_extra = (reg_s == RI["oberarm." + s_]) & (tf > t_saum + 0.04)
        haut_s |= haut_s_extra
haut_seite = np.where(np.isin(reg_t, [RI.get("unterarm.L", -1), RI.get("hand.L", -1), RI.get("oberarm.L", -1)]), "L", "R").astype(object)
def abtasten(sel, dichte):
    cnt = np.floor(area[sel] * dichte + rng.random(len(sel))).astype(np.int64)
    pick = np.repeat(sel, cnt); u, w2 = rng.random(len(pick)), rng.random(len(pick)); fl = u + w2 > 1; u[fl], w2[fl] = 1 - u[fl], 1 - w2[fl]
    pts = A[pick, 0] * (1 - u - w2)[:, None] + A[pick, 1] * u[:, None] + A[pick, 2] * w2[:, None]
    nrm = np.cross(A[pick, 1] - A[pick, 0], A[pick, 2] - A[pick, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return pts, nrm, pick
# --- Löcher (ohne Hände)
gP, gN, gT = abtasten(np.where(~hand_t & ~haut_t)[0], 1 / (10e-6 if STRENG else 25e-6))
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
    haut.append(np.where((QUELLE | (haut_s & (FOP > 0.3) & ~orange & (np.abs(sd) < 0.025))) & (reg_s == RI["unterarm." + s]) & (t_ > 0.3) & (t_ < 0.95))[0])
haut = np.concatenate(haut); HC = FC[haut]; m0 = np.median(HC, 0)
HC = HC[np.linalg.norm(HC - m0, axis=1) < 0.15]
if len(HC) > 40: HC = HC[HC.mean(1) >= np.percentile(HC.mean(1), 40)]   # beschattete/dunkle Scan-Haut nicht mitteln
HAUT = np.median(HC, 0) if len(HC) > 20 else m0
if not np.all(np.isfinite(HAUT)): HAUT = np.array([0.62, 0.45, 0.38])        # keine Armhaut im Scan (Mantel, Handschuhe)
# Gesichtshaut als Anker (Arme sind im Scan anders beleuchtet/fleckig): Hautton halb aus dem Gesicht
_ges = np.where(np.isin(reg_s, [RI["kopf"]]) & haut_farbe & (FOP > 0.5) & (FG < 0.01))[0]
if len(_ges) > 50:
    _gc = FC[_ges]; _gc = _gc[_gc.mean(1) >= np.percentile(_gc.mean(1), 30)]
    HAUT = 0.5 * HAUT + 0.5 * np.median(_gc, 0); rep["hautton_gesicht"] = [round(float(c), 3) for c in np.median(_gc, 0)]
rep["hautton"] = dict(rgb=[round(float(c), 3) for c in HAUT], aus_splats=int(len(HC)))
log("Hautton", rep["hautton"])
hand_vi = np.where(np.isin(reg_v, [RI["hand.L"], RI["hand.R"], RI["unterarm.L"], RI["unterarm.R"]]))[0]
kd_hv = kd_bau(V[hand_vi])
wellen = [(rng.normal(size=3) * (2 * math.pi / rng.uniform(0.025, 0.06)), rng.uniform(0, 2 * math.pi), rng.uniform(0.01, 0.025)) for _ in range(6)]
def hautfarbe(Pq, Nq, seiten, basis=None):
    var = np.zeros(len(Pq))
    for kv, ph, amp in wellen: var += amp * np.sin(Pq @ kv + ph)
    var += rng.normal(0, 0.012, len(Pq))
    hoehle = np.zeros(len(Pq))
    for i, (p, n) in enumerate(zip(Pq, Nq)):                            # Falten/Zwischenräume: Nachbarfläche liegt über der Tangentialebene
        nb = kd_hv.find_range(p, 0.012 * k)
        if len(nb) < 4: continue
        Q = np.array([t[0] for t in nb]) - p; hoehle[i] = np.mean((Q @ n) > 0.0015 * k)
    hoehle = np.clip((hoehle - 0.05) / 0.35, 0, 1)
    _nh = np.ones(len(Pq), bool)                                       # Faltenschatten nur an den Händen (am Arm: Streifen)
    if "hand.L" in RI:
        _kdh = kd_bau(V[np.where(np.isin(reg_v, [RI["hand.L"], RI["hand.R"]]))[0]])
        _nh = np.array([_kdh.find(p_)[2] < 0.01 for p_ in Pq])
    hoehle *= _nh
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
    # gebackene Schattierung wie im Scan-Licht: Oberseiten heller, Unterseiten und zum Körper gewandte Seiten dunkler
    up_ = 0.5 + 0.5 * Nq[:, 2]; med_ = np.clip(-np.sign(Pq[:, 0]) * Nq[:, 0], 0, 1)
    schatt = (0.80 + 0.24 * up_) * (1 - 0.14 * med_)
    fac = (1 + var) * (1 - 0.30 * hoehle) * (1 - 0.07 * palm) * schatt
    b_ = HAUT[None, :] if basis is None else basis
    return np.clip(b_ * fac[:, None], 0, 1), hoehle
hP, hN, hT = abtasten(np.where(hand_t)[0], 1 / 5e-6)
hC, hoehle = hautfarbe(hP, hN, hand_seite[hT]) if len(hP) else (np.zeros((0, 3)), np.zeros(0))
rep["neue_splats"]["haende"] = int(len(hP)); rep["neue_splats"]["hand_falten_anteil"] = round(float((hoehle > 0.3).mean()), 3)
# --- Haut aus dem STL: dichte, flache Splats; Farbe = Median der nächsten Scan-Haut-Splats (echte Hautfarbe/Flecken), geglättet,
#     Falten der Hand dunkler. Wellen-Variation aus hautfarbe() bleibt klein.
sP, sN, sT = abtasten(np.where(haut_t)[0], 1 / 7e-6)
if len(sP):
    kd_hq = kd_bau(FP[haut_quelle]) if len(haut_quelle) > 20 else None
    sB = np.tile(HAUT, (len(sP), 1)); treffer = 0
    if kd_hq is not None:
        for i_, p_ in enumerate(sP):
            nb = kd_hq.find_n(p_, 12); nb = [t for t in nb if t[2] < 0.025]
            if len(nb) >= 4: sB[i_] = np.median(FC[haut_quelle[[t[1] for t in nb]]], 0); treffer += 1
    kq_ = kd_bau(sP); sB2 = sB.copy()
    for i_, p_ in enumerate(sP):                                           # 7 mm weich mitteln (keine Splat-Sprenkel)
        nb = kq_.find_range(p_, 0.022)
        if nb: sB2[i_] = sB[[t[1] for t in nb]].mean(0)
    _hs = np.isin(reg_t[sT], list(HANDSCHUH))                          # Handschuh: nur Scan-Farbe, kein Hautton
    sB2 = np.where(_hs[:, None], sB2, 0.55 * sB2 + 0.45 * HAUT[None, :])   # mehr echte Scan-Hautfarbe (natürliche Fleckigkeit)                            # Scan-Haut ist fleckig (dünn, Schatten): ruhiger Grundton
    sC, _ = hautfarbe(sP, sN, haut_seite[sT], basis=sB2)
else:
    sC = np.zeros((0, 3)); treffer = 0
rep["neue_splats"]["haut_aus_stl"] = int(len(sP)); rep["neue_splats"]["haut_mit_scanfarbe"] = int(treffer)
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
# alte Scan-Splats, die auf der neuen STL-Haut liegen, aber nicht hautfarben sind (Ärmelfetzen, Schatten, Shirt-Reste am Arm)
pv0 = np.array([p_.vertices[0] for p_ in hi.data.polygons]); vhaut = np.zeros(len(V), bool); vhaut[tv[haut_t].ravel()] = True
auf_haut = behalten & (fl >= 0) & vhaut[pv0[np.maximum(fl, 0)]] & (np.abs(sd) < 0.03) & ~haut_farbe
behalten &= ~auf_haut; rep["scan_entfernt"]["auf_neuer_haut"] = int(auf_haut.sum())
sa = fa[behalten].copy()
# grünstichige Shirt-/Haar-Splats (Farbsaum im Scan) entsättigen
_dc = SW.spalten(fprops, "f_dc_0", "f_dc_1", "f_dc_2"); _c = np.clip(0.5 + SW.SH0 * sa[:, _dc], 0, 1)
_gruen = (_c[:, 1] > _c[:, 0] + 0.02) & (_c[:, 1] > _c[:, 2] - 0.02) & np.isin(reg_s[behalten], [RI["rumpf"], RI["kopf"], RI["hals"], RI["becken"]] + OA_REG)
_l = _c[_gruen].mean(1, keepdims=True); sa[np.where(_gruen)[0][:, None], np.array(_dc)[None, :]] = (np.repeat(_l, 3, 1) * np.array([1.0, 0.97, 1.0]) - 0.5) / SW.SH0
rep["gruenstich_entsaettigt"] = int(_gruen.sum())
# Kleidung beruhigen: helle Glanz-/Reflexsprenkel auf dunkler Kleidung (Marmor-/Wolkenmuster im Scan) zur Stofffarbe ziehen;
# sehr helle, kleine Splats (Kabel, Schmuck) bleiben
_c = np.clip(0.5 + SW.SH0 * sa[:, _dc], 0, 1); _lu = _c.mean(1); _rg = reg_s[behalten]; _fg = FG[behalten]; beruhigt = 0
for r_ in [RI["rumpf"], RI["becken"]] + OA_REG:
    m_ = (_rg == r_) & ~haut_farbe[behalten]
    if m_.sum() < 200: continue
    med = np.median(_c[m_], 0); ml = float(med.mean())
    if ml > 0.3: continue                                              # nur dunkle Stoffe
    hell = m_ & (_lu > ml + 0.035) & ~((_lu > 0.55) & (_fg < 0.004))
    k_ = np.clip((_lu[hell] - ml - 0.035) / 0.1, 0.5, 0.92)[:, None]
    neu_ = _c[hell] * (1 - k_) + med[None, :] * k_
    sa[np.where(hell)[0][:, None], np.array(_dc)[None, :]] = (neu_ - 0.5) / SW.SH0; beruhigt += int(hell.sum())
rep["kleidung_beruhigt"] = beruhigt
# Chroma-Tiefpass auf Stoff (Rumpf, Becken, Ärmel, Beine, Schuhe): Farbton zur Umgebung, Helligkeit nur +-5 % (Falten bleiben,
# Adern/Marmor/Regenbogen weg); blickabhängige SH auf 30 % (sonst kehren die Adern im Spiel zurück)
_c = np.clip(0.5 + SW.SH0 * sa[:, _dc], 0, 1); _Pb = FP[behalten]; _hf = haut_farbe[behalten]; _rest = [i for i, n_ in enumerate(fprops) if n_.startswith("f_rest")]
_STOFF = [RI[r] for r in RNAMEN if r.split(".")[0] in ("rumpf", "becken", "oberarm", "oberschenkel", "unterschenkel", "fuss")]
beruhigt2 = 0
for r_ in _STOFF:
    sel = np.where((_rg == r_) & ~_hf)[0]
    if len(sel) < 100: continue
    kd_ = kd_bau(_Pb[sel]); C_ = _c[sel]
    M_ = np.array([np.median(C_[[q[1] for q in kd_.find_n(p_, 24)]], 0) for p_ in _Pb[sel]])
    Lc, Lm = C_.mean(1, keepdims=True), M_.mean(1, keepdims=True)
    neu_ = np.clip(Lm + np.clip(Lc - Lm, -0.05, 0.05) + (C_ - Lc) * 0.3 + (M_ - Lm) * 0.7, 0, 1)
    sa[sel[:, None], np.array(_dc)[None, :]] = (neu_ - 0.5) / SW.SH0
    if _rest: sa[sel[:, None], np.array(_rest)[None, :]] *= 0.3
    beruhigt2 += len(sel)
rep["stoff_tiefpass"] = beruhigt2
# Füllfarben aus den beruhigten Stofffarben neu bestimmen (sonst übernehmen Füll-Splats Glanzflecken als große graue Flächen)
FC[np.where(behalten)[0]] = np.clip(0.5 + SW.SH0 * sa[:, _dc], 0, 1)
gC, _ = lokal_median(gP, reg_t[gT])
_gg = (gC[:, 1] > gC[:, 0] + 0.02) & (gC[:, 1] > gC[:, 2] - 0.02) & ~np.isin(reg_t[gT], [RI["oberschenkel.L"], RI["oberschenkel.R"], RI["unterschenkel.L"], RI["unterschenkel.R"]])
gC[_gg] = gC[_gg].mean(1, keepdims=True) * np.array([1.0, 0.97, 1.0])     # Füll-Splats ohne Grünstich
a = np.concatenate([sa, neue(hP, hN, hC, 0.0019, 0.0005, 0.97), neue(gP, gN, gC, 0.003 if STRENG else 0.0035, 0.0008 if STRENG else 0.001), neue(sP, sN, sC, 0.0024, 0.0006, 0.98)], 0)
parts = [int(len(sa)), int(len(hP)), int(len(gP)), int(len(sP))]
_ok = np.isfinite(a).all(1); rep["splats_ungueltig_entfernt"] = int((~_ok).sum()); a = a[_ok]
W = gewichte(SW.pos(fprops, a))
# Arm und Rumpf nie über den Luftspalt mischen (Achsel-Lappen): Kapsel-Label je Splat; Arm-Splats ab 25 % des Oberarms ohne
# Rumpf-Gewichte, Rumpf-Splats unter Schulter - 5 cm ohne Arm-Gewichte
_PA = SW.pos(fprops, a); _getrennt = 0
# Region der nächsten STL-Fläche je Splat (in der A-Pose trennt das STL Arm und Flanke sauber; Kapsel-Abstände ordneten
# T-Shirt-Splats unter der Achsel dem Oberarm zu -> dunkle Streifen auf dem Arm im Idle)
_pv0 = np.array([p_.vertices[0] for p_ in hi.data.polygons]); _regp = np.full(len(_PA), -1); _labp = np.full(len(_PA), -1)
_TPOSE = bool(rep.get("arm_einbettung")) and "lab_T" in globals()
for _i, _p in enumerate(_PA):
    _loc, _n, _fi, _d = bvh.find_nearest(Vector(_p), 0.2)
    if _fi is not None:
        _regp[_i] = reg_v[_pv0[_fi]]
        if _TPOSE: _labp[_i] = lab_T[_pv0[_fi]]
for s_ in ("LR" if _TPOSE else ""):                                  # nur bei eng am Mantel hängenden Armen (Carlos); bei T-Shirts riss
    # die harte Trennung den Ärmel vom Arm (Ärmel blieb als Lappen an der Schulter stehen)
    _armc = [jpos[n_] for n_ in joints if n_.endswith("." + s_) and not n_.startswith(("thigh", "shin", "foot", "toe", "shoulder"))]
    _rumpfc = [jpos[n_] for n_ in joints if n_.startswith("spine")] + [jpos["shoulder." + s_]]
    h_, t_ = np.array(J[("upper_arm." + s_, "head")]), np.array(J[("upper_arm." + s_, "tail")]); ab_ = t_ - h_
    tt_ = ((_PA - h_) @ ab_) / max(ab_ @ ab_, 1e-9)
    armm = np.isin(_regp, [RI[r] for r in ("oberarm." + s_, "unterarm." + s_, "hand." + s_) if r in RI]) & (tt_ > 0.2)
    if _TPOSE:                                                         # Arme eng am Mantel: Arm/Rumpf aus der freien T-Pose des STL
        armm = np.isin(_labp, [GI["oa." + s_], GI["ua." + s_], GI["hd." + s_]]) & (tt_ > 0.15)
    W[np.ix_(armm, _rumpfc)] = 0.0
    rumpfm = np.isin(_regp, [RI[r] for r in ("rumpf", "becken") if r in RI]) & (_PA[:, 2] < h_[2] - 0.04)
    if _TPOSE: rumpfm = ~np.isin(_labp, [GI[g_] for g_ in GNAMEN if g_.startswith(("oa.", "ua.", "hd."))]) & (_labp >= 0) & (_PA[:, 2] < h_[2] - 0.04)
    W[np.ix_(rumpfm, _armc)] = 0.0; _getrennt += int(armm.sum() + rumpfm.sum())
    _leer = W.sum(1) < 1e-4                                                # nichts übrig: ganz an den passenden Knochen
    W[_leer & armm, jpos["upper_arm." + s_]] = 1.0; W[_leer & rumpfm, jpos["spine.002"]] = 1.0
W /= np.maximum(W.sum(1, keepdims=True), 1e-6); rep["arm_rumpf_getrennt"] = _getrennt

# Füll-Mesh: Farben regionsweise + Hautton an den Händen, dann nach innen rücken (unter die Scan-Splats)
GV = verts(game); GN = np.zeros(len(GV) * 3); game.data.vertices.foreach_get("normal", GN); GN = GN.reshape(-1, 3)
reg_g = regionen(gewichte(GV))
g_hand = np.zeros(len(GV), bool); g_seite = np.full(len(GV), "", object)
for s in ("LR" if STL_HAENDE else ""):
    m = distal(GV, s) & np.isin(reg_g, [RI["hand." + s], RI["unterarm." + s]]); g_hand |= m; g_seite[m] = s
GC, _ = lokal_median(GV, reg_g)
if g_hand.any(): GC[g_hand] = hautfarbe(GV[g_hand], GN[g_hand], g_seite[g_hand])[0]
if HAUT_NEU: GC[np.isin(reg_g, HAUT_REG)] = HAUT                       # Füll-Mesh unter der STL-Haut: Hautton
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

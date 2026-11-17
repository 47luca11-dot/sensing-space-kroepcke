# Baut eine spielbare Figur für Startbildschirm + Spiel (gleiches Format wie Carlos/Mohammad/Daunenjacke).
#
# Aufruf (Blender headless):
#   Blender -b --factory-startup --python figur_bauen.py -- --id <id> --quelle <datei.ply> --art splats|scan
#           --modus splat|mesh --out <ordner> [--arme a|taschen] [--groesse 1.80]
#
#   --art splats : Quelle = ausgeschnittene Gaussian Splats (splat_werkzeug.py ausschneiden), Blender-Rahmen
#   --art scan   : Quelle = farbiges Scan-Mesh (KIRI *_mesh.ply, Y oben)
#   --modus splat: <id>.glb (verstecktes Füll-Mesh mit Farben) + <id>_splats.ply + <id>_skin.bin/.json
#   --modus mesh : <id>.glb (Mesh mit gebackener Farbtextur), keine Splats
#   --arme a      : Arme frei (A-Pose) -> Arm-Bewegungen von Carlos per Welt-Drehung
#   --arme taschen: Hände in den Taschen -> Arme bleiben ruhig
#
# Ergebnis zusätzlich: <id>.blend, _test/<id>_test.png (Gehen/Laufen/Stolpern), REPORT-Zeile (JSON) am Ende.
import bpy, bmesh, json, sys, math, struct, os
import numpy as np
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree

HIER = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HIER, "..", "..", "..", ".."))
sys.path.insert(0, HIER)
import splat_werkzeug as SW

argv = sys.argv[sys.argv.index("--") + 1:]
def arg(name, std=None):
    return argv[argv.index(name) + 1] if name in argv else std
ID = arg("--id"); QUELLE = arg("--quelle"); ART = arg("--art", "splats"); MODUS = arg("--modus", "splat")
OUT = arg("--out"); ARME = arg("--arme", "a")
CARLOS = arg("--carlos", os.path.join(REPO, "startbildschirm", "assets", "person1.glb"))
T = json.load(open(arg("--skelett", os.path.join(HIER, "carlos_skelett.json"))))["bones"]
T_H = 1.80
FUELL_DREIECKE = int(arg("--dreiecke", 80000))
TEX = 2048
ARM_BONES = ["shoulder.L", "upper_arm.L", "forearm.L", "hand.L", "shoulder.R", "upper_arm.R", "forearm.R", "hand.R"]
assert ID and QUELLE and OUT and ART in ("splats", "scan", "stl") and MODUS in ("splat", "mesh") and ARME in ("a", "taschen")
os.makedirs(f"{OUT}/_test", exist_ok=True)
rep = dict(id=ID, art=ART, modus=MODUS, arme=ARME)
rng = np.random.default_rng(7)
import time as _t
_T0 = _t.time()
def log(*a): print(f"[{_t.time() - _T0:6.0f}s]", *a, flush=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def aktiv(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[-1]

def mesh_zahlen(me):
    bm = bmesh.new(); bm.from_mesh(me)
    r = dict(punkte=len(bm.verts), dreiecke=sum(len(f.verts) - 2 for f in bm.faces),
             offene_kanten=sum(1 for e in bm.edges if e.is_boundary),
             nicht_mannigfaltig=sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary))
    bm.free(); return r

def inseln(bm):
    seen, out = set(), []
    for v in bm.verts:
        if v in seen: continue
        st, isl = [v], [v]; seen.add(v)
        while st:
            a = st.pop()
            for e in a.link_edges:
                b = e.other_vert(a)
                if b not in seen: seen.add(b); st.append(b); isl.append(b)
        out.append(isl)
    return sorted(out, key=len, reverse=True)

def flicken(bm, runden=4):
    """Nicht-mannigfaltige Stellen entfernen, alle Löcher schließen."""
    for _ in range(runden):
        bad = [e for e in bm.edges if not e.is_manifold and not e.is_boundary]
        if bad:
            bmesh.ops.delete(bm, geom=list({f for e in bad for f in e.link_faces}), context='FACES')
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        bnd = [e for e in bm.edges if e.is_boundary]
        if not bnd and not bad: break
        bmesh.ops.holes_fill(bm, edges=bnd, sides=0)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

def farbfeld(SP, SC, SOP, VOX=0.006, ausbreiten=16):
    """Voxel-Farbfeld aus Splats (deckende gewichtet), in leere Nachbarn ausgebreitet. Gibt feld(P)->RGB zurück."""
    gut = SOP > 0.3; Q = SP[gut]
    lo = Q.min(0) - (ausbreiten + 2) * VOX; dims = np.ceil((Q.max(0) - lo) / VOX).astype(np.int64) + ausbreiten + 3
    fl = np.ravel_multi_index(np.floor((Q - lo) / VOX).astype(np.int64).T, dims)
    FS = np.zeros((int(np.prod(dims)), 3), np.float32); FW = np.zeros(int(np.prod(dims)), np.float32); w = SOP[gut]
    for ch in range(3): np.add.at(FS[:, ch], fl, SC[gut][:, ch] * w)
    np.add.at(FW, fl, w); FS = FS.reshape(*dims, 3) / np.maximum(FW.reshape(dims), 1e-9)[..., None]; FW = FW.reshape(dims) > 0
    for _ in range(ausbreiten):
        S2 = np.zeros_like(FS); W2 = np.zeros(FW.shape)
        for ax in range(3):
            for d in (1, -1): W2 += np.roll(FW, d, ax); S2 += np.roll(FS * FW[..., None], d, ax)
        neu = ~FW & (W2 > 0); FS[neu] = S2[neu] / W2[neu][:, None]; FW = FW | neu
    def feld(Pq):
        g = (Pq - lo) / VOX - 0.5; i0 = np.floor(g).astype(np.int64); f = g - i0
        acc = np.zeros((len(Pq), 3)); accw = np.zeros(len(Pq))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ii = np.clip(i0 + [dx, dy, dz], 0, dims - 1)
                    ww = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    ww = ww * FW[ii[:, 0], ii[:, 1], ii[:, 2]]
                    acc += FS[ii[:, 0], ii[:, 1], ii[:, 2]] * ww[:, None]; accw += ww
        out = acc / np.maximum(accw, 1e-9)[:, None]
        out[accw < 1e-6] = np.median(SC[gut], 0)                    # weit weg vom Scan: mittlere Farbe
        return out, accw >= 1e-6
    return feld

def umeyama(A, B):
    """s, R, t mit B ≈ s R A + t (kleinste Quadrate)."""
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    U, S_, Vt = np.linalg.svd(B0.T @ A0 / len(A)); D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: D[2, 2] = -1
    R = U @ D @ Vt; s = np.trace(np.diag(S_) @ D) / ((A0 ** 2).sum() / len(A))
    return s, R, mb - s * R @ ma

def aufstellen(P, H):
    """Füße auf z=0, Mitte der Beine auf x=y=0, Zehen nach -Y. Gibt (P, Matrix) zurück."""
    bein = P[(P[:, 2] > 0.15 * H) & (P[:, 2] < 0.4 * H)]
    xl, xr = np.percentile(bein[:, 0], [1, 99]); t = np.array([-(xl + xr) / 2, -np.median(bein[:, 1]), -P[:, 2].min()])
    P = P + t; M = Matrix.Translation(Vector(t))
    shin = P[(P[:, 2] > 0.14 * H) & (P[:, 2] < 0.2 * H)][:, 1].mean(); fy = P[P[:, 2] < 0.03 * H][:, 1]
    vorn = fy.min() if abs(fy.min() - shin) > abs(fy.max() - shin) else fy.max()
    if vorn > shin: P = P * [-1, -1, 1]; M = Matrix.Rotation(math.pi, 4, 'Z') @ M
    return P, M

# ======================================================================= 1) Oberfläche
log("1 Oberfläche")
splat_props = splat_a = None
if ART == "splats":
    splat_props, splat_a = SW.lesen(QUELLE)
    SP = SW.pos(splat_props, splat_a); SOP = SW.deckkraft(splat_props, splat_a); SG = SW.groesse(splat_props, splat_a)
    SC = SW.farbe(splat_props, splat_a)
    H = float(np.percentile(SP[SOP > 0.5, 2], 99.98))
    # --- geschlossener Körper aus den Splats: Voxel belegen, 3 cm schließen, innen füllen ---
    VOX, K = 0.01, 3
    Q = SP[(SOP > 0.25) & (SG < 0.03)]
    lo = Q.min(0) - (K + 4) * VOX
    ijk = np.floor((Q - lo) / VOX).astype(np.int64); dims = ijk.max(0) + K + 6
    occ = np.zeros(dims, bool); occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    def weiten(g, k):
        g = g.copy()
        for ax in range(3):
            h = g.copy()
            for s in range(1, k + 1): h |= np.roll(g, s, ax) | np.roll(g, -s, ax)
            g = h
        return g
    dick = weiten(occ, K)
    aussen = np.zeros(dims, bool)
    aussen[0, :, :] = aussen[-1, :, :] = aussen[:, 0, :] = aussen[:, -1, :] = aussen[:, :, 0] = aussen[:, :, -1] = True
    aussen &= ~dick
    while True:
        n = aussen.copy()
        for ax in range(3): n |= np.roll(aussen, 1, ax) | np.roll(aussen, -1, ax)
        n &= ~dick
        if n.sum() == aussen.sum(): break
        aussen = n
    voll = ~weiten(aussen, K)                                      # innen gefüllt, dann um K zurück (Schließen)
    voll |= occ
    rep["voxel"] = dict(belegt=int(occ.sum()), voll=int(voll.sum()))
    def voxel_mesh(fest, name):
        """Geschlossene Fläche um alle festen Voxel: Würfel-Außenseiten -> Remesh -> glätten -> größte Insel."""
        quads = []
        for ax in range(3):
            for d in (1, -1):
                rand = fest & ~np.roll(fest, -d, ax)
                c = np.argwhere(rand)
                if not len(c): continue
                base = c.copy(); base[:, ax] += (1 if d > 0 else 0)
                u, w = [(ax + 1) % 3, (ax + 2) % 3]
                e_u = np.zeros(3, np.int64); e_u[u] = 1; e_w = np.zeros(3, np.int64); e_w[w] = 1
                q = np.stack([base, base + e_u, base + e_u + e_w, base + e_w], 1)
                if d < 0: q = q[:, ::-1]
                quads.append(q)
        q = np.concatenate(quads, 0)
        flat = np.ravel_multi_index(q.reshape(-1, 3).T, dims + 1)
        uq, inv = np.unique(flat, return_inverse=True)
        verts = np.stack(np.unravel_index(uq, dims + 1), 1) * VOX + lo
        me_ = bpy.data.meshes.new(name); me_.from_pydata(verts.tolist(), [], inv.reshape(-1, 4).tolist())
        ob_ = bpy.data.objects.new(name, me_); scene.collection.objects.link(ob_); aktiv(ob_)
        rm = ob_.modifiers.new("remesh", 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = VOX * 0.6; rm.adaptivity = 0
        bpy.ops.object.modifier_apply(modifier="remesh")
        sm = ob_.modifiers.new("glatt", 'LAPLACIANSMOOTH'); sm.lambda_factor = 0.7; sm.iterations = 12; sm.use_volume_preserve = True
        bpy.ops.object.modifier_apply(modifier="glatt")
        bm = bmesh.new(); bm.from_mesh(ob_.data)
        isl = inseln(bm); bmesh.ops.delete(bm, geom=[v for i in isl[1:] for v in i], context='VERTS')
        flicken(bm); bm.to_mesh(ob_.data); bm.free()
        return ob_, len(isl) - 1
    hi, weg = voxel_mesh(voll, ID + "_oberflaeche"); rep["inseln_entfernt"] = weg
    log("Hülle", len(hi.data.vertices), "Punkte")
    # Farben + Abstand zur Splat-Fläche (alles über das Voxelgitter, ohne Python-Schleifen)
    V = np.zeros(len(hi.data.vertices) * 3); hi.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    N_ = np.zeros(len(V) * 3); hi.data.vertices.foreach_get("normal", N_); N_ = N_.reshape(-1, 3)
    gut = SOP > 0.3
    stich = rng.choice(len(V), min(4000, len(V)), replace=False)
    gi_ = np.where(gut)[0]; probe = gi_[rng.choice(len(gi_), min(60000, len(gi_)), replace=False)]
    kd = KDTree(len(probe))
    for i, j in enumerate(probe): kd.insert(SP[j], i)
    kd.balance()
    abstand = float(np.median([kd.find(V[i])[2] for i in stich])); rep["huelle_abstand_cm"] = round(abstand * 100, 2)
    gidx3 = np.floor((SP[gut] - lo) / VOX).astype(np.int64)
    ok3 = np.all((gidx3 >= 0) & (gidx3 < dims), axis=1); gidx3 = gidx3[ok3]
    flat = np.ravel_multi_index(gidx3.T, dims); w_ = SOP[gut][ok3]
    FS = np.zeros((int(np.prod(dims)), 3)); FW = np.zeros(int(np.prod(dims)))
    for ch in range(3): np.add.at(FS[:, ch], flat, SC[gut][ok3, ch] * w_)
    np.add.at(FW, flat, w_)
    FS = FS.reshape(*dims, 3); FW = FW.reshape(dims); belegt_gut = FW > 0
    for _ in range(6):                                            # Farbe in leere Nachbarvoxel ausbreiten
        leer = FW == 0
        if not leer.any(): break
        S2 = np.zeros_like(FS); W2 = np.zeros_like(FW)
        for ax in range(3):
            for sh_ in (1, -1):
                W2 += np.roll(FW > 0, sh_, ax); S2 += np.roll(FS / np.maximum(FW, 1e-9)[..., None] * (FW > 0)[..., None], sh_, ax)
        neu = leer & (W2 > 0)
        FS[neu] = S2[neu] / W2[neu][:, None]; FW[neu] = 1
    def feld(Pq):                                                 # trilinear im Farbfeld
        g = (Pq - lo) / VOX - 0.5; i0 = np.floor(g).astype(np.int64); f = g - i0
        acc = np.zeros((len(Pq), 3)); accw = np.zeros(len(Pq))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ii = np.clip(i0 + [dx, dy, dz], 0, dims - 1)
                    ww = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    fw = FW[ii[:, 0], ii[:, 1], ii[:, 2]]; ww = ww * (fw > 0)
                    acc += FS[ii[:, 0], ii[:, 1], ii[:, 2]] / np.maximum(fw, 1e-9)[:, None] * ww[:, None]; accw += ww
        return acc / np.maximum(accw, 1e-9)[:, None]
    innen = V - N_ * abstand
    VC = np.ones((len(V), 4), np.float32); VC[:, :3] = np.clip(feld(innen), 0, 1)
    ca = hi.data.color_attributes.new("Col", 'FLOAT_COLOR', 'POINT'); ca.data.foreach_set("color_srgb", VC.ravel())
    # --- Lücken in den Splats: Hüll-Punkte, in deren Nähe (±1 Voxel) kein deckender Splat liegt ---
    nah = weiten(belegt_gut, 1)
    ii = np.clip(np.floor((innen - lo) / VOX).astype(np.int64), 0, dims - 1)
    luecke = ~nah[ii[:, 0], ii[:, 1], ii[:, 2]]
    LP = innen[luecke]
    zelle = np.unique(np.floor(LP / 0.008).astype(np.int64), axis=0, return_index=True)[1]
    LP, LC, LN = LP[zelle], VC[luecke][zelle, :3], N_[luecke][zelle]
    rep["luecken_splats"] = int(len(LP))
    log("Farben + Lücken fertig", len(LP), "Lücken-Splats")
    H_SPLATS = H
elif ART == "stl":
    # Farbloses Mesh (z. B. figuro, mm, Z oben). Farbe kommt aus einem ausgeschnittenen Splat-Scan derselben Person (--farbe).
    bpy.ops.wm.stl_import(filepath=QUELLE)
    hi = bpy.context.selected_objects[0]; hi.name = hi.data.name = ID + "_oberflaeche"
    aktiv(hi); bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    V = np.zeros(len(hi.data.vertices) * 3); hi.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    FARBE = arg("--farbe")
    if FARBE:
        fprops, fa = SW.lesen(FARBE); FP = SW.pos(fprops, fa); FOP = SW.deckkraft(fprops, fa); FC = SW.farbe(fprops, fa)
        ziel_h = float(np.percentile(FP[FOP > 0.5, 2], 99.98))
    else:
        ziel_h = float(arg("--groesse", 1.80))
    V = (V - [0, 0, V[:, 2].min()]) * (ziel_h / (V[:, 2].max() - V[:, 2].min()))
    V, _ = aufstellen(V, ziel_h)
    if FARBE:
        # Feinausrichtung (ICP mit Maßstab) des Meshes auf die Splats
        ziel = FP[FOP > 0.4]; ziel = ziel[rng.choice(len(ziel), min(80000, len(ziel)), replace=False)]
        kd = KDTree(len(ziel))
        for i, co in enumerate(ziel): kd.insert(co, i)
        kd.balance()
        probe = V[rng.choice(len(V), 20000, replace=False)]
        S_, R_, t_ = 1.0, np.eye(3), np.zeros(3); vorher = None
        for it in range(40):
            Qp = S_ * probe @ R_.T + t_
            nn = [kd.find(q) for q in Qp]; idx = np.array([n[1] for n in nn]); d = np.array([n[2] for n in nn])
            if vorher is None: vorher = float(np.median(d))
            m = d < max(0.015, 2.5 * np.median(d))
            s2, R2, t2 = umeyama(Qp[m], ziel[idx[m]])
            S_, R_, t_ = s2 * S_, R2 @ R_, s2 * R2 @ t_ + t2
        Qp = S_ * probe @ R_.T + t_; d = np.array([kd.find(q)[2] for q in Qp])
        V = S_ * V @ R_.T + t_
        rep["ausrichtung"] = dict(abstand_vorher_cm=round(vorher * 100, 2), abstand_nachher_cm=round(float(np.median(d)) * 100, 2),
                                  anteil_ueber_3cm=round(float((d > 0.03).mean()), 3), massstab=round(float(S_), 3))
    hi.data.vertices.foreach_set("co", V.ravel().astype(np.float32)); hi.data.update()
    # KI-/Druck-Meshes haben oft kleine Löcher und Doppelwände -> einmal geschlossen neu aufbauen (4 mm)
    aktiv(hi); rm = hi.modifiers.new("remesh", 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = float(arg("--remesh", 0.004)); rm.adaptivity = 0
    bpy.ops.object.modifier_apply(modifier="remesh")
    V = np.zeros(len(hi.data.vertices) * 3); hi.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    if FARBE:
        feld = farbfeld(FP, FC, FOP)
        rgb, getroffen = feld(V)
        rep["ausrichtung"]["ohne_scanfarbe"] = round(float(1 - getroffen.mean()), 3)
    else:
        rgb = np.full((len(V), 3), 0.6)
    VC = np.ones((len(V), 4), np.float32); VC[:, :3] = np.clip(rgb, 0, 1)
    hi.data.color_attributes.new("Col", 'FLOAT_COLOR', 'POINT').data.foreach_set("color_srgb", VC.ravel())
    bm = bmesh.new(); bm.from_mesh(hi.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0002)
    isl = inseln(bm); klein = [i for i in isl[1:] if len(i) < 0.02 * len(isl[0])]
    bmesh.ops.delete(bm, geom=[v for i in klein for v in i], context='VERTS'); rep["schwebeteile_entfernt"] = len(klein)
    flicken(bm); bm.to_mesh(hi.data); bm.free()
    log("STL ausgerichtet + eingefärbt", rep.get("ausrichtung"))
else:
    bpy.ops.wm.ply_import(filepath=QUELLE)
    hi = bpy.context.selected_objects[0]; hi.name = hi.data.name = ID + "_oberflaeche"
    hi.data.transform(Matrix.Rotation(-math.pi / 2, 4, 'X'))                 # KIRI: Höhe auf Y -> Z
    P = np.array([v.co[:] for v in hi.data.vertices])
    low = P[P[:, 2] < P[:, 2].min() + 0.35]
    off = np.array([(low[:, 0].min() + low[:, 0].max()) / 2, low[:, 1].mean(), P[:, 2].min()])
    hi.data.transform(Matrix.Translation(Vector(-off))); P = P - off
    shin = P[(P[:, 2] > 0.25) & (P[:, 2] < 0.35)][:, 1].mean(); feet = P[P[:, 2] < 0.04][:, 1]
    vorn = feet.min() if abs(feet.min() - shin) > abs(feet.max() - shin) else feet.max()
    if vorn > shin: hi.data.transform(Matrix.Rotation(math.pi, 4, 'Z'))
    groesse = arg("--groesse")
    if groesse:
        P = np.array([v.co[:] for v in hi.data.vertices]); hi.data.transform(Matrix.Scale(float(groesse) / P[:, 2].max(), 4))
    bm = bmesh.new(); bm.from_mesh(hi.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0003)
    isl = inseln(bm); klein = [i for i in isl[1:] if len(i) < 0.02 * len(isl[0])]
    bmesh.ops.delete(bm, geom=[v for i in klein for v in i], context='VERTS'); rep["schwebeteile_entfernt"] = len(klein)
    flicken(bm); bm.to_mesh(hi.data); bm.free()
P = np.array([v.co[:] for v in hi.data.vertices]); H = float(P[:, 2].max())
rep["groesse_m"] = round(H, 3); rep["oberflaeche"] = mesh_zahlen(hi.data)

# ======================================================================= 2) Skelett anpassen
log("2 Skelett")
def centroid(mask):
    s = P[mask]; return Vector(s.mean(axis=0)) if len(s) else None
r = H / T_H; x, y, z = P[:, 0], P[:, 1], P[:, 2]; J = {}
for name in ["spine", "spine.001", "spine.002", "spine.003", "spine.004", "spine.005", "spine.006"]:
    for end in ("head", "tail"):
        tz = T[name][end][2] * r; c = None
        for tol in (0.02, 0.05):
            c = centroid((np.abs(z - tz) < tol) & (np.abs(x) < 0.09))
            if c is not None: break
        J[(name, end)] = Vector((0.0, c.y if c is not None else 0.0, tz))
J[("spine.006", "tail")].z = H * 0.985
arm_info = {}
for side, s in (("L", 1), ("R", -1)):
    sm_ = s * x > 0.02
    def leg(zh):
        for tol in (0.015, 0.03, 0.06, 0.1):
            c = centroid(sm_ & (np.abs(z - zh) < tol) & (s * x < 0.25 * r))
            if c is not None: return Vector((c.x, c.y, zh))
    knee, ankle = leg(0.5 / T_H * H), leg(0.085 / T_H * H)
    hip = Vector((s * abs(T["thigh." + side]["head"][0]) * r, J[("spine", "head")].y, T["thigh." + side]["head"][2] * r))
    toe_y = y[sm_ & (z < 0.06)].min() + 0.03
    ball = Vector((ankle.x, toe_y + 0.09 * r, 0.025 * r))
    J.update({("thigh." + side, "head"): hip, ("thigh." + side, "tail"): knee, ("shin." + side, "head"): knee,
              ("shin." + side, "tail"): ankle, ("foot." + side, "head"): ankle, ("foot." + side, "tail"): ball,
              ("toe." + side, "head"): ball, ("toe." + side, "tail"): Vector((ankle.x, toe_y, 0.025 * r))})
    if ARME == "a":
        # Arm verfolgen: äußerster Querschnitt je Höhe, solange er seitlich frei neben Rumpf/Bein liegt
        sh_in = Vector(T["shoulder." + side]["head"]) * r; sh = Vector(T["shoulder." + side]["tail"]) * r
        c_sh = centroid(sm_ & (np.abs(z - sh.z) < 0.02) & (s * x > abs(sh.x) - 0.04) & (s * x < abs(sh.x) + 0.04))
        if c_sh is not None: sh.y = c_sh.y
        path = [sh]
        for zi in np.arange(sh.z - 0.06, 0.3 * H, -0.01):
            b = sm_ & (np.abs(z - zi) < 0.008)
            if b.sum() < 5: continue
            xm = (s * x[b]).max()
            if xm < 0.26 * r: continue
            c = centroid(b & (s * x > xm - 0.09 * r))
            if c is not None: path.append(c)
        seg = [(path[i + 1] - path[i]).length for i in range(len(path) - 1)]; L = sum(seg)
        def at(dist):
            acc = 0
            for i, l in enumerate(seg):
                if acc + l >= dist: return path[i].lerp(path[i + 1], (dist - acc) / max(l, 1e-6))
                acc += l
            return path[-1].copy()
        L = min(L, 0.76 * r)                                    # Schulter bis Fingerspitze ~0.76 m (Rest = Gegenstand/Artefakt)
        tip = at(L); wrist = at(L - 0.17 * r); elbow = at((L - 0.17 * r) * 0.52)
        arm_info[side] = dict(laenge=round(L, 3), handspitze_z=round(tip.z, 3))
    else:
        def armc(zh, thick=0.10):
            for tol in (0.015, 0.03, 0.05):
                b = sm_ & (np.abs(z - zh) < tol)
                if b.sum() > 10: break
            xm = (s * x[b]).max(); return centroid(b & (s * x > xm - thick))
        sh_in = Vector(T["shoulder." + side]["head"]) * r
        sh = armc(0.815 * H, 0.09); sh.z = 0.815 * H
        elbow, wrist, tip = armc(0.625 * H), armc(0.505 * H, 0.08), armc(0.445 * H, 0.07)
    J.update({("shoulder." + side, "head"): sh_in, ("shoulder." + side, "tail"): sh, ("upper_arm." + side, "head"): sh,
              ("upper_arm." + side, "tail"): elbow, ("forearm." + side, "head"): elbow, ("forearm." + side, "tail"): wrist,
              ("hand." + side, "head"): wrist, ("hand." + side, "tail"): tip})
rep["arme_info"] = arm_info
fehlt = [k for k, v in J.items() if v is None]
if fehlt: raise SystemExit(f"Gelenk nicht gefunden: {fehlt}")

arm_data = bpy.data.armatures.new(ID + "_rig"); rig = bpy.data.objects.new(ID + "_rig", arm_data)
scene.collection.objects.link(rig); rig.show_in_front = True
aktiv(rig); bpy.ops.object.mode_set(mode='EDIT')
for bn, tb in T.items():
    b = arm_data.edit_bones.new(bn); b.head = J[(bn, "head")]; b.tail = J[(bn, "tail")]
    if (b.tail - b.head).length < 0.01: b.tail = b.head + Vector((0, 0, 0.02))
    tdir = (Vector(tb["tail"]) - Vector(tb["head"])).normalized(); ndir = (b.tail - b.head).normalized()
    b.align_roll(tdir.rotation_difference(ndir) @ Vector(tb["z"]))
for bn, tb in T.items():
    if tb["parent"]: arm_data.edit_bones[bn].parent = arm_data.edit_bones[tb["parent"]]
bpy.ops.object.mode_set(mode='OBJECT')
for pb in rig.pose.bones: pb.rotation_mode = 'QUATERNION'

# ======================================================================= 3) Spiel-Mesh
log("3 Spiel-Mesh")
if ART == "splats" and MODUS == "splat":
    EROD = 2                                                   # Füll-Mesh ~1,5 cm unter den Splats
    game, _ = voxel_mesh(~weiten(~voll, EROD), ID + "_mesh")
    GV0 = np.zeros(len(game.data.vertices) * 3); game.data.vertices.foreach_get("co", GV0); GV0 = GV0.reshape(-1, 3)
    GC = np.ones((len(GV0), 4), np.float32); GC[:, :3] = np.clip(feld(GV0), 0, 1)
    game.data.color_attributes.new("Col", 'FLOAT_COLOR', 'POINT').data.foreach_set("color_srgb", GC.ravel())
    rep["fuellmesh_nach_innen_cm"] = EROD * VOX * 100
else:
    game = hi.copy(); game.data = hi.data.copy(); game.name = game.data.name = ID + "_mesh"; scene.collection.objects.link(game)
aktiv(game)
n_tri = sum(len(p_.vertices) - 2 for p_ in game.data.polygons)          # Dreiecke zählen (Remesh liefert Vierecke)
if n_tri > FUELL_DREIECKE:
    m = game.modifiers.new("dec", 'DECIMATE'); m.ratio = FUELL_DREIECKE / n_tri; m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="dec")
bm = bmesh.new(); bm.from_mesh(game.data); flicken(bm); bm.to_mesh(game.data); bm.free()
bpy.ops.object.shade_smooth()
if MODUS == "mesh":
    while game.data.color_attributes: game.data.color_attributes.remove(game.data.color_attributes[0])
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.002)
    bpy.ops.object.mode_set(mode='OBJECT')
    src = bpy.data.materials.new("quelle"); src.use_nodes = True; nt = src.node_tree; nt.nodes.clear()
    vc = nt.nodes.new("ShaderNodeVertexColor"); vc.layer_name = hi.data.color_attributes[0].name
    em = nt.nodes.new("ShaderNodeEmission"); mo = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(vc.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs["Emission"], mo.inputs["Surface"])
    hi.data.materials.clear(); hi.data.materials.append(src)
    img = bpy.data.images.new(ID + "_farbe", TEX, TEX, alpha=False)
    mat = bpy.data.materials.new(ID + "_material"); mat.use_nodes = True; nt2 = mat.node_tree
    bsdf = nt2.nodes["Principled BSDF"]; tex = nt2.nodes.new("ShaderNodeTexImage"); tex.image = img
    nt2.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"]); bsdf.inputs["Roughness"].default_value = 0.9; nt2.nodes.active = tex
    game.data.materials.clear(); game.data.materials.append(mat)
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 1
    aktiv(hi, game)
    bpy.ops.object.bake(type='EMIT', use_selected_to_active=True, cage_extrusion=0.02, max_ray_distance=0.06, margin=16)
    img.filepath_raw = f"{OUT}/{ID}_farbe.jpg"; img.file_format = 'JPEG'; scene.render.image_settings.quality = 92; img.save()
    rep["textur"] = f"{TEX}x{TEX}"
else:
    game.data.color_attributes.active_color = game.data.color_attributes[0]

# ======================================================================= 4) Gewichte
log("4 Gewichte")
aktiv(game, rig); bpy.ops.object.parent_set(type='ARMATURE_AUTO')
bones = [(b.name, b.head_local.copy(), b.tail_local.copy()) for b in rig.data.bones]
def segdist(p, a, b):
    ab = b - a; t = max(0, min(1, (p - a).dot(ab) / ab.length_squared)); return (a + ab * t - p).length
fb = 0
for v in game.data.vertices:
    if sum(g.weight for g in v.groups) < 0.01:
        n = min(bones, key=lambda B: segdist(v.co, B[1], B[2]))[0]; game.vertex_groups[n].add([v.index], 1.0, 'REPLACE'); fb += 1
aktiv(game); bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
bpy.ops.object.vertex_group_smooth(group_select_mode='ALL', factor=0.5, repeat=2)
bpy.ops.object.vertex_group_limit_total(group_select_mode='ALL', limit=4)
bpy.ops.object.vertex_group_normalize_all(lock_active=False)
bpy.ops.object.mode_set(mode='OBJECT')
rep["spielmesh"] = dict(mesh_zahlen(game.data), ohne_gewicht_nachgetragen=fb)

# ======================================================================= 5) Animationen von Carlos
log("5 Animationen")
before, acts_before = set(bpy.data.objects), set(bpy.data.actions)
bpy.ops.import_scene.gltf(filepath=CARLOS)
c_objs = [o for o in bpy.data.objects if o not in before]
c_arm = [o for o in c_objs if o.type == 'ARMATURE'][0]
ratio = H / max((c_arm.matrix_world @ b.tail_local).z for b in c_arm.data.bones)
def channelbag(act):
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags: return cb
names = []
c_arm.animation_data_create(); rig.animation_data_create()
for a in [a for a in bpy.data.actions if a not in acts_before]:
    if ARME == "taschen" and a.name.endswith("_Pockets"): continue
    new = a.copy(); new.name = a.name.replace("person1_", ID + "_"); new.use_fake_user = True
    cb = channelbag(new)
    for fc in list(cb.fcurves):
        p = fc.data_path; bone = p.split('"')[1] if '"' in p else ""
        if bone in ARM_BONES or p.endswith(".scale") or (p.endswith(".location") and bone != "spine"):
            cb.fcurves.remove(fc)
        elif p.endswith(".location"):
            for k in fc.keyframe_points: k.co.y *= ratio; k.handle_left.y *= ratio; k.handle_right.y *= ratio
    if ARME == "a":                                            # Arme: Welt-Drehung von Carlos übernehmen (andere Ruhepose)
        c_arm.animation_data.action = a
        if c_arm.animation_data.action_slot is None and a.slots: c_arm.animation_data.action_slot = a.slots[0]
        rig.animation_data.action = new
        if rig.animation_data.action_slot is None and new.slots: rig.animation_data.action_slot = new.slots[0]
        cons = []
        for n_ in ARM_BONES:
            c_ = rig.pose.bones[n_].constraints.new('COPY_ROTATION'); c_.target = c_arm; c_.subtarget = n_
            c_.owner_space = c_.target_space = 'WORLD'; cons.append((n_, c_))
        rows = []
        for f in range(int(a.frame_range[0]), int(a.frame_range[1]) + 1):
            scene.frame_set(f); bpy.context.view_layer.update()
            rows.append((f, {n_: rig.convert_space(pose_bone=rig.pose.bones[n_], matrix=rig.pose.bones[n_].matrix,
                                                   from_space='POSE', to_space='LOCAL').to_quaternion() for n_ in ARM_BONES}))
        for n_, c_ in cons: rig.pose.bones[n_].constraints.remove(c_)
        prev = {}
        for f, qs in rows:
            for n_, q_ in qs.items():
                if n_ in prev and prev[n_].dot(q_) < 0: q_.negate()
                prev[n_] = q_; pb = rig.pose.bones[n_]; pb.rotation_quaternion = q_; pb.keyframe_insert("rotation_quaternion", frame=f)
    names.append(new.name)
for o in c_objs: bpy.data.objects.remove(o, do_unlink=True)
for a in [a for a in bpy.data.actions if not a.name.startswith(ID + "_")]: bpy.data.actions.remove(a)
rig.animation_data.action = None
for pb in rig.pose.bones: pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
rep["animationen"] = sorted(names)

# ======================================================================= 6) Splats + Skin
log("6 Splats")
if MODUS == "splat":
    gidx = {g.index: g.name for g in game.vertex_groups}
    GV = np.zeros(len(game.data.vertices) * 3); game.data.vertices.foreach_get("co", GV); GV = GV.reshape(-1, 3)
    joints = list(T.keys()); jpos = {n: i for i, n in enumerate(joints)}
    GW = np.zeros((len(GV), len(joints)), np.float32)
    for v in game.data.vertices:
        for g in v.groups: GW[v.index, jpos[gidx[g.group]]] = g.weight
    kdg = KDTree(len(GV))
    for i, co in enumerate(GV): kdg.insert(co, i)
    kdg.balance()
    def gewichte(PTS):
        me_ = bpy.data.meshes.new("wpts"); me_.vertices.add(len(PTS)); me_.vertices.foreach_set("co", np.asarray(PTS, np.float32).ravel())
        ob_ = bpy.data.objects.new("wpts", me_); scene.collection.objects.link(ob_)
        for n_ in joints: ob_.vertex_groups.new(name=n_)
        dt = ob_.modifiers.new("wt", 'DATA_TRANSFER'); dt.object = game; dt.use_vert_data = True
        dt.data_types_verts = {'VGROUP_WEIGHTS'}; dt.vert_mapping = 'POLYINTERP_NEAREST'
        dt.layers_vgroup_select_src = 'ALL'; dt.layers_vgroup_select_dst = 'NAME'
        aktiv(ob_); bpy.ops.object.modifier_apply(modifier="wt")
        gmap = {g.index: jpos[g.name] for g in ob_.vertex_groups}
        out = np.zeros((len(PTS), len(joints)), np.float32)
        for v in me_.vertices:
            for g in v.groups: out[v.index, gmap[g.group]] = g.weight
        leer = out.sum(1) < 1e-6
        if leer.any():                                           # Sicherheitsnetz: nächster Mesh-Punkt
            for i in np.where(leer)[0]: out[i] = GW[kdg.find(PTS[i])[1]]
        bpy.data.objects.remove(ob_, do_unlink=True)
        return out
    if ART == "splats":
        props, a = splat_props, splat_a
        # Lücken-Splats anhängen (flache Scheiben in Flächenrichtung, Farbe aus der Umgebung)
        nl = len(LP)
        if nl:
            b = np.zeros((nl, len(props)))
            b[:, SW.spalten(props, "x", "y", "z")] = LP
            b[:, SW.spalten(props, "f_dc_0", "f_dc_1", "f_dc_2")] = (np.clip(LC, 0, 1) - 0.5) / SW.SH0
            b[:, props.index("opacity")] = math.log(0.9 / 0.1)
            b[:, SW.spalten(props, "scale_0", "scale_1")] = math.log(0.006); b[:, props.index("scale_2")] = math.log(0.0015)
            zax = np.array([0, 0, 1.0]); ax = np.cross(np.broadcast_to(zax, LN.shape), LN); sn = np.linalg.norm(ax, axis=1)
            ang = np.arctan2(sn, LN[:, 2]); ax = np.where(sn[:, None] > 1e-8, ax / (sn[:, None] + 1e-12), np.array([1.0, 0, 0]))
            b[:, SW.spalten(props, "rot_0", "rot_1", "rot_2", "rot_3")] = np.concatenate([np.cos(ang / 2)[:, None], ax * np.sin(ang / 2)[:, None]], 1)
            a = np.concatenate([a, b], 0)
        parts = [len(splat_a), nl]
        W = gewichte(SW.pos(props, a))
        # Blender-Rahmen -> glTF (x, z, -y); SH mitdrehen; nur SH-Grad 1 behalten (Spiel nutzt maxSh = 1)
        SW.transformieren(props, a, np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), np.zeros(3), 1.0)
        rest = [p for p in props if p.startswith("f_rest_")]
        if len(rest) > 9:
            k = len(rest) // 3
            keep = [props.index(f"f_rest_{ch * k + i}") for ch in range(3) for i in range(3)]
            basis = [i for i, p in enumerate(props) if not p.startswith("f_rest_")]
            cols = basis[:9] + keep + basis[9:]
            a = a[:, cols]; props = [props[i] for i in basis[:9]] + [f"f_rest_{i}" for i in range(9)] + [props[i] for i in basis[9:]]
        SW.schreiben(f"{OUT}/{ID}_splats.ply", props, a)
        N_SPL = len(a)
    else:
        # Splats von der Scan-Oberfläche abtasten (wie Mohammad)
        me = hi.data; me.calc_loop_triangles(); nt_ = len(me.loop_triangles)
        tri_v = np.zeros(nt_ * 3, np.int64); me.loop_triangles.foreach_get("vertices", tri_v); tri_v = tri_v.reshape(-1, 3)
        tri_l = np.zeros(nt_ * 3, np.int64); me.loop_triangles.foreach_get("loops", tri_l); tri_l = tri_l.reshape(-1, 3)
        V = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
        col = me.color_attributes[0]
        C = np.zeros(len(col.data) * 4, np.float32); col.data.foreach_get("color_srgb", C); C = C.reshape(-1, 4)[:, :3]
        Ct = C[tri_l] if col.domain == 'CORNER' else C[tri_v]
        A = V[tri_v]; area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
        N_SPL = int(arg("--splats", 600000))
        cnt = np.floor(area * (N_SPL / area.sum()) + rng.random(nt_)).astype(np.int64)
        pick = np.repeat(np.arange(nt_), cnt); N_SPL = len(pick)
        u, w2 = rng.random(N_SPL), rng.random(N_SPL); fl = u + w2 > 1; u[fl], w2[fl] = 1 - u[fl], 1 - w2[fl]
        bary = np.stack([1 - u - w2, u, w2], 1)
        pos_ = (A[pick] * bary[:, :, None]).sum(1); rgb = (Ct[pick] * bary[:, :, None]).sum(1)
        nrm = np.cross(A[pick, 1] - A[pick, 0], A[pick, 2] - A[pick, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        W = gewichte(pos_)
        to_gl = lambda v: np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
        pos_, nrm = to_gl(pos_), to_gl(nrm)
        rad = math.sqrt(area.sum() / N_SPL / math.pi) * 1.6
        zax = np.array([0, 0, 1.0]); ax = np.cross(np.broadcast_to(zax, nrm.shape), nrm); sn = np.linalg.norm(ax, axis=1)
        ang = np.arctan2(sn, nrm[:, 2]); ax = np.where(sn[:, None] > 1e-8, ax / (sn[:, None] + 1e-12), np.array([1.0, 0, 0]))
        props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        a = np.zeros((N_SPL, len(props)))
        a[:, 0:3] = pos_; a[:, 3:6] = nrm; a[:, 6:9] = (np.clip(rgb, 0, 1) - 0.5) / SW.SH0; a[:, 9] = math.log(0.97 / 0.03)
        a[:, 10] = a[:, 11] = math.log(rad); a[:, 12] = math.log(rad * 0.25)
        a[:, 13:17] = np.concatenate([np.cos(ang / 2)[:, None], ax * np.sin(ang / 2)[:, None]], 1)
        SW.schreiben(f"{OUT}/{ID}_splats.ply", props, a); parts = [N_SPL]
    rep["splats"] = dict(anzahl=int(N_SPL), teile=[int(p) for p in parts])   # Skin wird nach dem GLB-Export geschrieben
    # Scan-Figuren: Füll-Mesh unter die Splat-Oberfläche schieben (Splat-Figuren: schon aus geschrumpftem Körper gebaut)
    if ART in ("scan", "stl"):
        schrumpf = 0.015
        aktiv(game); m = game.modifiers.new("schrumpf", 'DISPLACE'); m.strength = -schrumpf; m.mid_level = 0; m.direction = 'NORMAL'
        game.modifiers.move(len(game.modifiers) - 1, 0); bpy.ops.object.modifier_apply(modifier="schrumpf")
        rep["fuellmesh_nach_innen_cm"] = round(schrumpf * 100, 1)

# ======================================================================= 7) GLB + Blend
log("7 Export")
hi.hide_render = True; hi.hide_set(True)
bpy.ops.wm.save_as_mainfile(filepath=f"{OUT}/{ID}.blend")
aktiv(game, rig)
glb = f"{OUT}/{ID}.glb"
kw = dict(filepath=glb, use_selection=True, export_format='GLB', export_animations=True, export_animation_mode='ACTIONS',
          export_skins=True, export_force_sampling=True)
if MODUS == "splat":
    kw.update(export_def_bones=False, export_materials='NONE')
    try: bpy.ops.export_scene.gltf(**kw, export_vertex_color='ACTIVE')
    except TypeError: bpy.ops.export_scene.gltf(**kw, export_colors=True)
else:
    bpy.ops.export_scene.gltf(**kw, export_image_format='JPEG', export_jpeg_quality=90)
g = open(glb, "rb").read(); n = struct.unpack('<I', g[12:16])[0]; gj = json.loads(g[20:20 + n])
rep["glb"] = dict(mb=round(len(g) / 1e6, 1), attribute=list(gj['meshes'][0]['primitives'][0]['attributes']),
                  knochen=len(gj['skins'][0]['joints']), clips=[x['name'] for x in gj['animations']])
if MODUS == "splat":
    # Skin-Indizes MÜSSEN der Gelenk-Reihenfolge der GLB folgen (three.js skeleton.bones), nicht der Vorlage
    glb_joints = [gj['nodes'][i]['name'] for i in gj['skins'][0]['joints']]
    W = W[:, [joints.index(n_) for n_ in glb_joints]]
    top = np.argsort(-W, axis=1)[:, :4]
    tw = np.take_along_axis(W, top, 1); tw = tw / np.maximum(tw.sum(1, keepdims=True), 1e-6)
    q = np.floor(tw * 255).astype(np.int32); q[:, 0] += 255 - q.sum(1)
    np.concatenate([top.astype(np.uint8), q.astype(np.uint8)], 1).tofile(f"{OUT}/{ID}_skin.bin")
    json.dump({"count": int(N_SPL), "joints": glb_joints, "parts": [int(p) for p in parts]}, open(f"{OUT}/{ID}_skin.json", "w"))

# ======================================================================= 8) Testbilder: Gehen / Laufen / Stolpern
log("8 Testbilder")
sc = scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'TEXTURE' if MODUS == "mesh" else 'VERTEX'
sc.render.resolution_x = 300; sc.render.resolution_y = 400
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = H * 1.25
bilder = []
for act, frames in ((f"{ID}_Walk", (0, 7, 14)), (f"{ID}_Run", (0, 5, 9)), (f"{ID}_Stumble", (12,))):
    rig.animation_data.action = bpy.data.actions[act]
    if rig.animation_data.action_slot is None and bpy.data.actions[act].slots: rig.animation_data.action_slot = bpy.data.actions[act].slots[0]
    for f in frames:
        sc.frame_set(f)
        for view in ("front", "side"):
            if view == "front": cam.location = (0, -8, H * 0.52); cam.rotation_euler = (math.pi / 2, 0, 0)
            else: cam.location = (8, 0, H * 0.52); cam.rotation_euler = (math.pi / 2, 0, math.pi / 2)
            pth = f"{OUT}/_test/{act}_{f:02d}_{view}.png"; sc.render.filepath = pth; bpy.ops.render.render(write_still=True); bilder.append(pth)
cols = 6; rows_ = (len(bilder) + cols - 1) // cols; Wd, Hd = 300, 400
buf = np.ones((rows_ * Hd, cols * Wd, 4), np.float32)
for i, pth in enumerate(bilder):
    im = bpy.data.images.load(pth); px = np.empty(Wd * Hd * 4, np.float32); im.pixels.foreach_get(px)
    rr, cc = divmod(i, cols); buf[(rows_ - 1 - rr) * Hd:(rows_ - rr) * Hd, cc * Wd:(cc + 1) * Wd] = px.reshape(Hd, Wd, 4)
o = bpy.data.images.new("sheet", cols * Wd, rows_ * Hd); o.pixels.foreach_set(buf.ravel())
o.filepath_raw = f"{OUT}/_test/{ID}_test.png"; o.file_format = 'PNG'; o.save()
rep["testbild"] = f"{OUT}/_test/{ID}_test.png"
print("REPORT", json.dumps(rep))

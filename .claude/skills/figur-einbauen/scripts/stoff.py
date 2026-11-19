# Stoff-Knochen für Mantel und Schal (läuft in Blender nach figur_stl.py/haende_proportion.py, VOR gang.py):
#   sys.argv = [..., "--", "--id", "<id>", "--out", "<ordner mit <id>_splats.ply, _skin.bin, _skin.json>"]
#
# Mantel: unterhalb der Hüfte 4 Bahnen (vorn/hinten x links/rechts), je Kette aus 2 Knochen am Becken (spine). Der Saum wird aus
#   den Splats erkannt (unter dem Schritt liegen zwischen den Beinen nur noch Mantel-Punkte). Gewichte: oben am Becken fest,
#   nach unten zunehmend auf die Bahn, die Bahnen weich nach Winkel um das Becken gemischt (keine "Hosenbeine" aus dem Mantel).
# Schal: die beiden hängenden Enden (heller als der Mantel, vor der Brust) bekommen je eine Kette aus 3 Knochen an spine.003.
# Die Bewegung (Trägheit, Fahrtwind, Beine schieben den Mantel) simuliert gang.py und backt sie in Walk/Run/Idle.
# Danach: Splat-Skin (<id>_skin.bin/json) in der neuen GLB-Gelenkreihenfolge neu geschrieben.
import bpy, sys, os, json, math, struct
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
def arg(n, std=None): return argv[argv.index(n) + 1] if n in argv else std
ID, OUT = arg("--id"), arg("--out")
HIER = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HIER)
import splat_werkzeug as SW
def log(*a): print("[stoff]", *a, flush=True)

scene = bpy.data.scenes.get("Figur_" + ID) or bpy.context.scene
if bpy.context.window: bpy.context.window.scene = scene
rig = bpy.data.objects[ID + "_rig"]; game = bpy.data.objects[ID + "_mesh"]
MW = rig.matrix_world
def bw(n, end="head"):
    b = rig.data.bones[n]; return np.array(MW @ (b.head_local if end == "head" else b.tail_local))
H = max((MW @ b.tail_local).z for b in rig.data.bones); k = H / 1.8

props, a = SW.lesen(os.path.join(OUT, ID + "_splats.ply"))
Gp = SW.pos(props, a); SP = np.stack([Gp[:, 0], -Gp[:, 2], Gp[:, 1]], 1)          # glTF -> Blender
SC = np.clip(SW.farbe(props, a), 0, 1); SO = SW.deckkraft(props, a)
GV = np.array([game.matrix_world @ v.co for v in game.data.vertices])
rep = {}

# ------------------------------------------------------------------ Mantelsaum
hip_z = 0.5 * (bw("thigh.L")[2] + bw("thigh.R")[2])
pel = bw("spine"); cx, cy = float(pel[0]), float(pel[1])
dep = SP[(np.abs(SP[:, 2] - hip_z) < 0.03) & (np.abs(SP[:, 0] - cx) < 0.2), 1]
y_vorn, y_hint = float(np.percentile(dep, 3)), float(np.percentile(dep, 97))
knie_z = 0.5 * (bw("shin.L")[2] + bw("shin.R")[2])
saum = None
for zz in np.arange(hip_z - 0.12 * k, knie_z - 0.15 * k, -0.01):
    m = (np.abs(SP[:, 2] - zz) < 0.006) & (np.abs(SP[:, 0] - cx) < 0.025) & (SO > 0.3)
    if m.sum() >= 6: saum = zz
    elif saum is not None and zz < saum - 0.04: break
MANTEL = saum is not None and saum < hip_z - 0.18 * k and "--ohne_mantel" not in argv
rep["huefte_z"] = round(hip_z, 3); rep["mantel_saum_z"] = round(float(saum), 3) if saum is not None else None
log("Hüfte", round(hip_z, 3), "Mantelsaum", rep["mantel_saum_z"], "Mantel" if MANTEL else "kein Mantel")

# ------------------------------------------------------------------ Schal-Enden
hals_z = bw("spine.004")[2]; brust = bw("spine.003", "tail")
lum = SC.mean(1); sat = (SC.max(1) - SC.min(1)) / np.maximum(SC.max(1), 1e-6)
rumpf = (SP[:, 2] > hip_z) & (SP[:, 2] < hals_z) & (np.abs(SP[:, 0] - cx) < 0.18 * k)
mantel_lum = float(np.median(lum[rumpf])) if rumpf.any() else 0.2
grau = (lum > mantel_lum + 0.08) & (sat < 0.25) & (SO > 0.3)
vor = SP[:, 1] < float(np.percentile(SP[rumpf, 1], 35)) if rumpf.any() else np.zeros(len(SP), bool)
SCHAL = []
for s, sg in (("L", 1), ("R", -1)):
    m = grau & vor & (sg * (SP[:, 0] - cx) > 0.01) & (SP[:, 2] < hals_z - 0.04 * k) & (SP[:, 2] > hip_z - 0.1 * k)
    if m.sum() < 150 or "--ohne_schal" in argv: continue
    P = SP[m]; z_top = min(hals_z - 0.05 * k, float(np.percentile(P[:, 2], 99))); z_end = float(np.percentile(P[:, 2], 1.5))
    if z_top - z_end < 0.15 * k: continue
    knoten = []
    for t in np.linspace(0, 1, 4):
        zz = z_top - t * (z_top - z_end); q = P[np.abs(P[:, 2] - zz) < 0.025]
        if len(q) < 5: q = P[np.argsort(np.abs(P[:, 2] - zz))[:20]]
        knoten.append(np.array([np.median(q[:, 0]), np.percentile(q[:, 1], 50), zz]))
    SCHAL.append(dict(seite=s, z_top=z_top, z_end=z_end, knoten=knoten, n=int(m.sum())))
rep["schal"] = [dict(seite=c["seite"], z_top=round(float(c["z_top"]), 3), z_end=round(float(c["z_end"]), 3), splats=int(c["n"])) for c in SCHAL]
log("Schal-Enden", rep["schal"])

# ------------------------------------------------------------------ Knochen anlegen
# 8 Bahnen (Berater-Gutachten): vorn/hinten je 2 neben der Mitte, dazu seitlich vorn/hinten; Winkel um das Becken, 0 = vorn (-Y), + = links (+X)
BAHNEN = (("V1L", 22.5), ("V2L", 67.5), ("H2L", 112.5), ("H1L", 157.5), ("H1R", -157.5), ("H2R", -112.5), ("V2R", -67.5), ("V1R", -22.5))
EBENEN = 3
def ring_radius(zz, phi_deg, Pq):
    phi = math.radians(phi_deg); d = Pq[:, :2] - [cx, cy]
    ang = np.degrees(np.arctan2(d[:, 0], -d[:, 1])); m = (np.abs(Pq[:, 2] - zz) < 0.015) & (np.abs((ang - phi_deg + 180) % 360 - 180) < 22)
    r = np.linalg.norm(d[m], axis=1)
    return float(np.percentile(r, 85)) - 0.01 if m.sum() > 10 else 0.15 * k
neu = []
bpy.context.view_layer.objects.active = rig
for o in bpy.context.view_layer.objects: o.select_set(o == rig)
bpy.ops.object.mode_set(mode='EDIT'); eb = rig.data.edit_bones
for n in [b.name for b in eb if b.name.startswith(("mantel.", "schal."))]: eb.remove(eb[n])
Mi = MW.inverted()
if MANTEL:
    z_top = hip_z + 0.02; z_saum = saum + 0.01; ZS = [z_top + (z_saum - z_top) * i / EBENEN for i in range(EBENEN + 1)]
    for nm, phi in BAHNEN:
        pts = []
        for zz in ZS:
            r = ring_radius(zz, phi, SP); f = math.radians(phi)
            pts.append(Vector((cx + r * math.sin(f), cy - r * math.cos(f), zz)))
        prev = eb["spine"]
        for i in range(1, EBENEN + 1):
            b = eb.new(f"mantel.{nm}.{i}"); b.head = Mi @ pts[i - 1]; b.tail = Mi @ pts[i]; b.parent = prev; b.use_connect = i > 1
            b.use_deform = True; prev = b; neu.append(b.name)
for c in SCHAL:
    prev = eb["spine.003"]
    for i in (1, 2, 3):
        b = eb.new(f"schal.{c['seite']}.{i}"); b.head = Mi @ Vector(c["knoten"][i - 1]); b.tail = Mi @ Vector(c["knoten"][i]); b.parent = prev
        b.use_connect = i > 1; b.use_deform = True; prev = b; neu.append(b.name)
bpy.ops.object.mode_set(mode='OBJECT')
log("neue Knochen", neu)
rep["knochen_neu"] = neu
if not neu:
    print("REPORT", json.dumps(rep)); raise SystemExit(0)

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

def stoff_gewichte(P, armgewicht=None):
    """Zusätzliche Gewichte je Punkt: dict bone -> array, plus 'anteil' (wie viel der alten Gewichte ersetzt wird).
    Höchstens 4 Einflüsse: Winkel linear zwischen den zwei nächsten Bahnen, Höhe linear zwischen zwei Ebenen."""
    out = {n: np.zeros(len(P)) for n in neu}; anteil = np.zeros(len(P))
    if MANTEL:
        d = P[:, :2] - [cx, cy]; ang = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
        im = (P[:, 2] < hip_z + 0.03) & (P[:, 2] > saum - 0.03)
        if armgewicht is not None: im &= armgewicht < 0.3                   # Hände/Unterarme auf Hüfthöhe gehören nicht zum Mantel
        d_bein = np.full(len(P), 9.0)                                      # Hose unter dem Mantel bleibt am Bein (nur die Mantelfläche schwingt)
        for s_ in "LR":
            for bn in ("thigh." + s_, "shin." + s_):
                a0, a1 = bw(bn), bw(bn, "tail"); ab = a1 - a0; t_ = np.clip(((P - a0) @ ab) / max(ab @ ab, 1e-9), 0, 1)
                d_bein = np.minimum(d_bein, np.linalg.norm(P - (a0 + t_[:, None] * ab), axis=1))
        im &= d_bein > 0.095 * k
        c = smooth(hip_z + 0.02, hip_z - 0.12 * k, P[:, 2]) * im
        phis = np.array([phi for _, phi in BAHNEN]); W8 = np.zeros((len(P), len(BAHNEN)))
        dif = (ang[:, None] - phis[None, :] + 180) % 360 - 180
        W8 = np.maximum(0, 1 - np.abs(dif) / 45.0); W8 /= np.maximum(W8.sum(1, keepdims=True), 1e-6)   # Hut-Funktion, Abstand 45°
        tt = np.clip((hip_z + 0.02 - P[:, 2]) / max(hip_z + 0.02 - saum, 1e-3), 0, 1) * EBENEN       # 0..3 entlang der Kette
        for j, (nm, _) in enumerate(BAHNEN):
            for i in range(1, EBENEN + 1):
                if i == 1: wi = np.clip(1 - np.maximum(tt - 0.5, 0), 0, 1)
                elif i == EBENEN: wi = np.clip(1 - np.maximum((i - 0.5) - tt, 0), 0, 1)
                else: wi = np.clip(1 - np.abs(tt - (i - 0.5)), 0, 1)
                out[f"mantel.{nm}.{i}"] += c * W8[:, j] * wi
        anteil = np.maximum(anteil, c)
    for sc_ in SCHAL:
        s, sg = sc_["seite"], (1 if sc_["seite"] == "L" else -1)
        m = grau_p(P) & (sg * (P[:, 0] - cx) > 0.0) & (P[:, 1] < y_mitte_brust) & (P[:, 2] < sc_["z_top"] + 0.02) & (P[:, 2] > sc_["z_end"] - 0.04)
        Kn = np.array(sc_["knoten"]); dmin = np.full(len(P), 9.0)                 # nah an der Schal-Mittellinie (keine Mantel-Adern/Knöpfe)
        for a0, a1 in zip(Kn[:-1], Kn[1:]):
            ab = a1 - a0; t_ = np.clip(((P - a0) @ ab) / max(ab @ ab, 1e-9), 0, 1); dmin = np.minimum(dmin, np.linalg.norm(P - (a0 + t_[:, None] * ab), axis=1))
        m &= dmin < 0.06 * k
        c = smooth(sc_["z_top"] + 0.02, sc_["z_top"] - 0.06 * k, P[:, 2]) * m
        L = sc_["z_top"] - sc_["z_end"]; tt = np.clip((sc_["z_top"] - P[:, 2]) / max(L, 1e-3), 0, 1) * 3
        for i in (1, 2, 3):
            wi = np.clip(1 - np.abs(tt - (i - 0.5)), 0, 1)
            out[f"schal.{s}.{i}"] += c * wi
        anteil = np.maximum(anteil, c)
    return out, anteil
y_mitte_brust = float(np.percentile(SP[rumpf, 1], 35)) + 0.03 if rumpf.any() else cy
_grau_lum = mantel_lum + 0.08
def grau_p(P):                                                         # für Splats: Farbe; fürs Füll-Mesh: alles in der Zone
    return np.ones(len(P), bool) if P is GV else grau_s
grau_s = grau

# ------------------------------------------------------------------ Füll-Mesh: Vertex-Gruppen
_ag = np.zeros(len(GV)); _armn = {g.index for g in game.vertex_groups if g.name.endswith((".L", ".R")) and g.name.split(".")[0] in ("upper_arm", "forearm", "hand", "palm", "thumb", "f_index", "f_middle", "f_ring", "f_pinky")}
for v in game.data.vertices: _ag[v.index] = sum(g.weight for g in v.groups if g.group in _armn)
add, anteil = stoff_gewichte(GV, _ag)
vg = {g.name: g for g in game.vertex_groups}
for n in neu:
    if n not in vg: vg[n] = game.vertex_groups.new(name=n)
for v in game.data.vertices:
    i = v.index; a_ = anteil[i]
    if a_ <= 1e-4: continue
    for g in v.groups: g.weight *= (1 - a_)
    for n in neu:
        if add[n][i] > 1e-4: vg[n].add([i], float(add[n][i]), 'REPLACE')
rep["fuellmesh_stoffpunkte"] = int((anteil > 0.05).sum())

# ------------------------------------------------------------------ GLB-Gelenkreihenfolge ermitteln (Probe-Export)
tmp = os.path.join(OUT, "_stoff_probe.glb")
for o in scene.objects: o.select_set(o in (game, rig))
bpy.context.view_layer.objects.active = rig
kw = dict(filepath=tmp, use_selection=True, use_active_scene=True, export_format='GLB', export_animations=False, export_skins=True, export_def_bones=False, export_materials='NONE')
bpy.ops.export_scene.gltf(**kw)
g_ = open(tmp, "rb").read(); n_ = struct.unpack('<I', g_[12:16])[0]; gj = json.loads(g_[20:20 + n_]); os.remove(tmp)
joints_neu = [gj['nodes'][i]['name'] for i in gj['skins'][0]['joints']]

# ------------------------------------------------------------------ Splat-Skin umschreiben
sk = json.load(open(os.path.join(OUT, ID + "_skin.json"))); J_alt = sk["joints"]
raw = np.fromfile(os.path.join(OUT, ID + "_skin.bin"), np.uint8).reshape(-1, 8); idx = raw[:, :4].astype(int); wt = raw[:, 4:].astype(np.float64) / 255.0
NJ = len(joints_neu); JI = {n: i for i, n in enumerate(joints_neu)}
Wd = np.zeros((len(idx), NJ))
for c in range(4): np.add.at(Wd, (np.arange(len(idx)), np.array([JI[J_alt[j]] for j in idx[:, c]])), wt[:, c])
_armc = [JI[n] for n in joints_neu if n.endswith((".L", ".R")) and n.split(".")[0] in ("upper_arm", "forearm", "hand", "palm", "thumb", "f_index", "f_middle", "f_ring", "f_pinky")]
add_s, anteil_s = stoff_gewichte(SP, Wd[:, _armc].sum(1))
Wd *= (1 - anteil_s)[:, None]
for n in neu: Wd[:, JI[n]] += add_s[n]
top = np.argsort(-Wd, 1)[:, :4]; tw = np.take_along_axis(Wd, top, 1); tw /= np.maximum(tw.sum(1, keepdims=True), 1e-9)
q = np.round(tw * 255).astype(np.int64); q[:, 0] += 255 - q.sum(1)
out = np.zeros((len(idx), 8), np.uint8); out[:, :4] = top; out[:, 4:] = np.clip(q, 0, 255)
out.tofile(os.path.join(OUT, ID + "_skin.bin"))
sk["joints"] = joints_neu; sk["stoff"] = neu; json.dump(sk, open(os.path.join(OUT, ID + "_skin.json"), "w"))
rep["splats_stoff"] = int((anteil_s > 0.05).sum()); rep["gelenke"] = NJ
print("REPORT", json.dumps(rep, default=float))

# Hände auf menschliche Proportion bringen (läuft in Blender, Szene "Figur_<id>", nach dem Bau und VOR gang.py):
#   Handlänge (Handgelenk -> Mittelfingerspitze) höchstens 0,105 x Körperhöhe (Norm 0,108 H, Hände wirkten im Spiel zu groß).
#   Skaliert um das Handgelenk: Hand-/Fingerknochen (Ruhepose), Füll-Mesh und Splats (Lage + Größe), weich nach Handgewicht,
#   Hand-Splats zusätzlich etwas feiner (Scan-Hände bestehen aus großen, unscharfen Splats).
#   sys.argv = [..., "--", "--id", "<id>", "--out", "<ordner mit <id>_splats.ply, _skin.bin, _skin.json>"]
import bpy, sys, os, json, math
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
def arg(n, std=None): return argv[argv.index(n) + 1] if n in argv else std
ID, OUT = arg("--id"), arg("--out")
MAXK = float(arg("--max", 0.105)); FEIN = float(arg("--fein", 0.85))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splat_werkzeug as SW
def log(*a): print("[haende]", *a, flush=True)

scene = bpy.data.scenes["Figur_" + ID]; bpy.context.window.scene = scene
rig = scene.objects[ID + "_rig"]; game = scene.objects[ID + "_mesh"]
M = rig.matrix_world; H = max((M @ b.tail_local).z for b in rig.data.bones)
GV = np.array([v.co[:] for v in game.data.vertices]); MG = np.array(game.matrix_world)
GW = GV @ MG[:3, :3].T + MG[:3, 3]

def kette(side):
    return ["hand." + side] + [b.name for b in rig.data.bones["hand." + side].children_recursive]

faktoren = {}
for s in "LR":
    w = np.array(M @ rig.data.bones["hand." + s].head_local); tip = np.array(M @ rig.data.bones["f_middle.03." + s].tail_local)
    d = (tip - w) / np.linalg.norm(tip - w)
    gi = {g.index for g in game.vertex_groups if g.name in kette(s)}
    ws = np.zeros(len(GV))
    for v in game.data.vertices:
        ws[v.index] = sum(g.weight for g in v.groups if g.group in gi)
    Q = GW - w; along = Q @ d
    m = ws > 0.5
    L = float(along[m].max()) if m.any() else float(np.linalg.norm(tip - w))
    f = min(1.0, MAXK * H / max(L, 1e-6))
    faktoren[s] = dict(f=f, w=w, ws=ws, L=L)
    log(f"Hand {s}: Länge {L:.3f} m = {L / H:.3f} H -> Faktor {f:.3f}")

# 1) Füll-Mesh (Ruhepose), weich nach Handgewicht
for s, F in faktoren.items():
    if F["f"] >= 0.999: continue
    k = 1 - (1 - F["f"]) * np.clip(F["ws"], 0, 1)
    GW = F["w"] + (GW - F["w"]) * k[:, None]
GL = (GW - MG[:3, 3]) @ np.linalg.inv(MG[:3, :3]).T
for v, p in zip(game.data.vertices, GL): v.co = Vector(p)
game.data.update()

# 2) Knochen (Ruhepose): Hand + Finger um das Handgelenk skalieren
bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True); bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')
Mi = M.inverted()
for s, F in faktoren.items():
    if F["f"] >= 0.999: continue
    wl = Mi @ Vector(F["w"])
    for n in kette(s):
        eb = rig.data.edit_bones[n]; roll = eb.roll
        eb.head = wl + (eb.head - wl) * F["f"]; eb.tail = wl + (eb.tail - wl) * F["f"]; eb.roll = roll
bpy.ops.object.mode_set(mode='OBJECT')

# 3) Splats (Datei im GLB-Rahmen: Blender (x, y, z) -> (x, z, -y))
props, a = SW.lesen(f"{OUT}/{ID}_splats.ply")
sk = json.load(open(f"{OUT}/{ID}_skin.json")); J = sk["joints"]
if sk.get("haende_fein"): FEIN = 1.0                                   # schon einmal verfeinert -> nicht doppelt
raw = np.fromfile(f"{OUT}/{ID}_skin.bin", np.uint8).reshape(-1, 8); idx, wt = raw[:, :4].astype(int), raw[:, 4:] / 255.0
P = SW.pos(props, a); Pb = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)
sc = SW.spalten(props, "scale_0", "scale_1", "scale_2")
for s, F in faktoren.items():
    jn = np.array([n in kette(s) for n in J])
    ws = (wt * jn[idx]).sum(1)
    k = 1 - (1 - F["f"]) * np.clip(ws, 0, 1)
    Pb = F["w"] + (Pb - F["w"]) * k[:, None]
    a[:, sc] += (np.log(k) + np.log(FEIN) * np.clip(ws, 0, 1))[:, None]    # kleiner und feiner
    log(f"Splats Hand {s}: {int((ws > 0.5).sum())}")
a[:, SW.spalten(props, "x", "y", "z")] = np.stack([Pb[:, 0], Pb[:, 2], -Pb[:, 1]], 1)
SW.schreiben(f"{OUT}/{ID}_splats.ply", props, a)
sk["haende_fein"] = True; json.dump(sk, open(f"{OUT}/{ID}_skin.json", "w"))
log("fertig", {s: round(F["f"], 3) for s, F in faktoren.items()})

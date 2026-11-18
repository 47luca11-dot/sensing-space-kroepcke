# Knie-/Hüftwinkel eines Gang-Clips messen und mit der Normkurve (Winter 2009, normales Gehtempo) vergleichen
import bpy, math, numpy as np
from mathutils import Vector
NORM_KNIE = [(0,4),(5,10),(10,16),(15,18),(20,15),(30,8),(40,5),(45,5),(50,8),(55,16),(60,32),(65,48),(70,59),(73,62),(78,58),(85,42),(90,24),(95,8),(100,4)]
def norm(p, tab):
    x = [a for a, _ in tab]; y = [b for _, b in tab]; return float(np.interp(p * 100, x, y))
def messen(ID, clip="Walk"):
    sc = bpy.data.scenes["Figur_" + ID]; bpy.context.window.scene = sc
    r = sc.objects[ID + "_rig"]; act = bpy.data.actions[ID + "_" + clip]
    r.animation_data.action = act
    if act.slots: r.animation_data.action_slot = act.slots[0]
    n = int(act.frame_range[1]); out = []
    for f in range(n):
        sc.frame_set(f); M = r.matrix_world
        row = {}
        for s in "LR":
            h = M @ r.pose.bones["thigh." + s].head; kn = M @ r.pose.bones["shin." + s].head; a = M @ r.pose.bones["foot." + s].head
            t = M @ r.pose.bones["toe." + s].tail
            th = (kn - h).normalized(); sh = (a - kn).normalized()
            knie = math.degrees(th.angle(sh))
            hip = math.degrees(math.atan2(-th.y, -th.z))          # + = Oberschenkel vorn (Blick -Y)
            row[s] = (round(knie, 1), round(hip, 1), round(a.z, 3), round(t.z, 3))
        out.append(row)
    return out, n

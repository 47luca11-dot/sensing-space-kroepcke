# Realistische Fortbewegung für eine fertig geriggte Figur (Szene "Figur_<id>", Rig "<id>_rig", Mesh "<id>_mesh") – läuft in Blender
# (Live über das Blender-MCP oder headless nach figur_hybrid.py / figur_finger.py).
#
# Ersetzt die von Carlos kopierten Clips <id>_Walk, <id>_Run, <id>_Idle_Breathing durch eigene Zyklen nach Gang-Biomechanik
# (Vorbild: Third-Person-Fortbewegung wie GTA V / Assassin's Creed – aufrechter, ruhiger Oberkörper, schmale Spur, lockere Arme):
#   Becken  : Höhe (tiefster Punkt beim Doppelstand bzw. Mitte Stand beim Laufen), Seitpendel zum Standbein, Rotation, Absenken der
#             Schwungseite, Vorlage
#   Rumpf   : Gegenrotation zum Becken, Kopf bleibt nach vorn stabilisiert, Atmung (Idle)
#   Beine   : IK auf Fußbahnen – Fersenaufsatz (Zehen hoch), flacher Stand, Abrollen über den Ballen, Schwung mit Bodenfreiheit
#   Arme    : IK, Pendel aus der Schulter gegengleich zu den Beinen, Ellbogen beugt beim Vorschwingen stärker; beim Laufen ~85° gebeugt
# Ablauf je Bild: FK-Knochen in Weltkoordinaten setzen -> IK-Ziele setzen -> auswerten -> lokale Drehungen ablesen -> Keyframes.
# Zyklen laufen auf der Stelle; der Standfuß gleitet gleichmäßig nach hinten (daraus misst das Spiel die Schrittlänge).
#
#   Aufruf: sys.argv = [..., "--", "--id", "<id>", "--out", "<ordner>"] und exec, oder Blender -b <id>.blend --python gang.py -- --id <id> --out <ordner>
import bpy, math, json, os, sys, struct
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(n, std=None): return argv[argv.index(n) + 1] if n in argv else std
ID = arg("--id"); OUT = arg("--out")
scene = bpy.data.scenes.get("Figur_" + ID) or bpy.context.scene
if bpy.context.window and bpy.context.window.scene != scene: bpy.context.window.scene = scene
rig = bpy.data.objects[ID + "_rig"]; game = bpy.data.objects[ID + "_mesh"]
FPS = scene.render.fps
def log(*a): print("[gang]", *a, flush=True)

SPINE = ["spine", "spine.001", "spine.002", "spine.003", "spine.004", "spine.005", "spine.006"]
FK = SPINE + ["shoulder.L", "shoulder.R"]
MW = rig.matrix_world.copy(); MWi = MW.inverted(); R_obj = MW.to_3x3().normalized()

# ------------------------------------------------------------------ Ruhepose vermessen (Welt)
rig.animation_data_create(); rig.animation_data.action = None
for pb in rig.pose.bones:
    pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0); pb.scale = (1, 1, 1)
    for c in list(pb.constraints): pb.constraints.remove(c)
bpy.context.view_layer.update()
def W(pb_name, end="head"):
    pb = rig.pose.bones[pb_name]; return MW @ (pb.head if end == "head" else pb.tail)
REST = {n: (MW @ rig.pose.bones[n].matrix).copy() for n in rig.pose.bones.keys()}
hip = {s: W("thigh." + s) for s in "LR"}; ankle = {s: W("foot." + s) for s in "LR"}; ball = {s: W("toe." + s) for s in "LR"}
sho = {s: W("upper_arm." + s) for s in "LR"}
L_up = {s: (W("upper_arm." + s, "tail") - sho[s]).length for s in "LR"}
L_fo = {s: (W("forearm." + s, "tail") - W("forearm." + s)).length for s in "LR"}
L_th = (W("thigh.L", "tail") - hip["L"]).length; L_sh = (W("shin.L", "tail") - W("shin.L")).length
LEG = L_th + L_sh
H = max((MW @ b.tail_local).z for b in rig.data.bones)
k = H / 1.8
base_ankle_z = sum(ankle[s].z for s in "LR") / 2
L_ball = sum((ball[s] - ankle[s]).length for s in "LR") / 2
pelvis0 = W("spine")
log(f"Höhe {H:.3f}  Bein {LEG:.3f}  Oberarm {L_up['L']:.3f}  Unterarm {L_fo['L']:.3f}  Fuß→Ballen {L_ball:.3f}")

# ------------------------------------------------------------------ IK-Ziele
def empty(name):
    o = bpy.data.objects.get(name)
    if o is None:
        o = bpy.data.objects.new(name, None); scene.collection.objects.link(o); o.empty_display_size = 0.05
    o.animation_data_clear(); return o
T_foot = {s: empty(f"IK_fuss_{s}") for s in "LR"}; P_knee = {s: empty(f"POL_knie_{s}") for s in "LR"}
T_hand = {s: empty(f"IK_hand_{s}") for s in "LR"}; P_elb = {s: empty(f"POL_ellbogen_{s}") for s in "LR"}
foot_rest_rot = {s: REST["foot." + s].to_3x3().normalized() for s in "LR"}

def add_ik(bone, target, pole, chain):
    c = rig.pose.bones[bone].constraints.new('IK'); c.target = target; c.pole_target = pole; c.chain_count = chain
    c.use_tail = True; c.iterations = 200; return c
IK = {}
for s in "LR":
    IK["leg" + s] = add_ik("shin." + s, T_foot[s], P_knee[s], 2)
    cr = rig.pose.bones["foot." + s].constraints.new('COPY_ROTATION'); cr.target = T_foot[s]; cr.owner_space = cr.target_space = 'WORLD'
    IK["arm" + s] = add_ik("forearm." + s, T_hand[s], P_elb[s], 2)

def set_empty(o, loc, rot3=None):
    o.location = loc
    if rot3 is not None: o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = rot3.to_quaternion()

def pole_waehlen(ik, bone, setup):
    """Pole-Winkel so wählen, dass der Knochen seinen Drall aus der Ruhepose behält (Knie nach vorn, Ellbogen nach hinten)."""
    best = None
    for ang in (-90, 0, 90, 180):
        ik.pole_angle = math.radians(ang); setup(); bpy.context.view_layer.update()
        zr = REST[bone].to_3x3() @ Vector((0, 0, 1)); zp = (MW @ rig.pose.bones[bone].matrix).to_3x3() @ Vector((0, 0, 1))
        sc_ = zr.normalized().dot(zp.normalized())
        if best is None or sc_ > best[0]: best = (sc_, ang)
    ik.pole_angle = math.radians(best[1]); return best[1]

# ------------------------------------------------------------------ Hilfen
def rot_um_gelenk(pb_name, Rw, off=Vector()):
    """Knochen um sein Gelenk drehen (Welt-Drehung Rw, 3x3) und optional verschieben – zusätzlich zur Pose der Eltern."""
    pb = rig.pose.bones[pb_name]
    M0 = MW @ pb.matrix; h = M0.translation.copy()
    Mw = Matrix.Translation(h + off) @ Rw.to_4x4() @ Matrix.Translation(-h) @ M0
    pb.matrix = MWi @ Mw; bpy.context.view_layer.update()
def E(pitch=0.0, roll=0.0, yaw=0.0):
    """Welt-Drehung: pitch um X (positiv = nach vorn neigen, Blick -Y), roll um Y, yaw um Z (Grad)."""
    return (Matrix.Rotation(math.radians(yaw), 3, 'Z') @ Matrix.Rotation(math.radians(roll), 3, 'Y') @ Matrix.Rotation(-math.radians(pitch), 3, 'X'))
VOR = Vector((0, -1, 0)); LINKS = Vector((1, 0, 0)); OBEN = Vector((0, 0, 1))

def ellbogen_dist(a, b, beuge):
    return math.sqrt(a * a + b * b + 2 * a * b * math.cos(math.radians(beuge)))

def arm_ziel(s, schwung, beuge, abd=7.0, innen=0.0, vor=0.0):
    """Handgelenk-Ziel aus Schulter: schwung (Grad, + = nach vorn), beuge (Ellbogen, Grad), abd (Grad seitlich weg)."""
    sg = 1 if s == "L" else -1
    S = W("upper_arm." + s)
    d = (math.sin(math.radians(abd)) * sg * LINKS + math.sin(math.radians(schwung)) * VOR - math.cos(math.radians(schwung)) * math.cos(math.radians(abd)) * OBEN)
    d.normalize()
    D = ellbogen_dist(L_up[s], L_fo[s], beuge)
    tgt = S + d * D - innen * sg * LINKS + vor * VOR
    pole = S + Vector((sg * 0.25, 0.35, -0.2)) * k
    return tgt, pole

def bein(s, vor_pos, hoehe, kipp, seit):
    """Fußziel: vor_pos (m nach vorn ab Hüfte-Ruhe, Knöchel bei flachem Fuß), hoehe = Bodenfreiheit, kipp (Grad, + = Zehen hoch).
    Ferse oben (kipp < 0): der Fuß dreht um den Ballen, der auf dem Boden bleibt -> Knöchel geht hoch UND etwas nach vorn."""
    sg = 1 if s == "L" else -1
    x = sg * seit; y = hip[s].y - vor_pos; z = base_ankle_z + hoehe
    if kipp < 0:
        a = math.radians(-kipp)
        z += L_ball * math.sin(a) * 0.85; y -= L_ball * (1 - math.cos(a))      # Abrollen über den Ballen
    else: z += 0.05 * k * math.sin(math.radians(kipp))                       # Fersenaufsatz (Drehung um die Ferse)
    R = Matrix.Rotation(math.radians(-kipp), 3, 'X') @ foot_rest_rot[s]          # Blick -Y: positive X-Drehung senkt die Zehen
    return Vector((x, y, z)), R

def smooth(t): t = min(1, max(0, t)); return t * t * (3 - 2 * t)
def lerp(a, b, t): return a + (b - a) * t

# ------------------------------------------------------------------ Zyklen
def gehen(p, s):
    """Gehen: Stand 60 %, linke Ferse bei p=0, rechte bei 0.5. Doppelschritt ~1,3 m (bei 1,7 m), ~130 Schritte/min bei 1,5 m/s.
    Ferse löst sich bei ~35 % des Zyklus, der Ballen bleibt bis zum Zehenabstoß (60 %) am Boden; im Schwung hebt die Kniebeugung
    den Fuß (Zehen nur 2-3 cm über dem Boden), Hüftstreckung hinten nur ~15°."""
    q = (p + (0.0 if s == "L" else 0.5)) % 1.0
    VORN, HINT, ST = 0.34 * k, -0.44 * k, 0.60
    if q < ST:
        t = q / ST; pos = lerp(VORN, HINT, t)
        if t < 0.16: kipp = lerp(21, 0, smooth(t / 0.16))                 # Fersenaufsatz, Zehen 15-20° hoch -> flach bis ~10 % des Zyklus
        elif t < 0.58: kipp = 0
        else: kipp = lerp(0, -42, smooth((t - 0.58) / 0.42) ** 1.2)      # Ferse hebt ab, Abrollen über den Ballen
        h = 0.0
    else:
        t = (q - ST) / (1 - ST)
        pos = lerp(HINT, VORN, 0.35 * t + 0.65 * smooth(t))              # gleichmäßig nach vorn, kein spätes Vorschnellen
        # Kniebeugung im Schwung (~60° bei ~73 % des Zyklus): der Fuß geht hinten hoch (Knöchel ~14 cm), dann flach nach vorn
        h = 0.10 * k * math.sin(math.pi * min(1.0, t / 0.65)) ** 1.4 + 0.015 * k * math.sin(math.pi * t)
        kipp = lerp(-42, 0, smooth(t / 0.6)) if t < 0.6 else lerp(0, 21, smooth((t - 0.6) / 0.4))
    return pos, h, kipp, 0.092 * k

def laufen(p, s):
    """Laufen: Stand 36 %, Flugphasen, Ferse schwingt hoch nach hinten, Knie treibt nach vorn."""
    # Doppelschritt ~2,9 m: bei 4,5 m/s ~170 Schritte/min (früher 1,9 m -> 280/min, Zappeln)
    q = (p + (0.0 if s == "L" else 0.5)) % 1.0
    VORN, HINT, ST = 0.26 * k, -0.50 * k, 0.27
    if q < ST:
        t = q / ST; pos = lerp(VORN, HINT, t); h = 0.0
        kipp = lerp(8, 0, smooth(t / 0.15)) if t < 0.15 else (0 if t < 0.55 else lerp(0, -45, smooth((t - 0.55) / 0.45)))
    else:
        t = (q - ST) / (1 - ST)
        pos = lerp(HINT, VORN, smooth(t) ** 0.85)
        h = (0.10 + 0.30 * math.sin(math.pi * min(1, t / 0.85)) ** 0.8) * k * (1 - smooth((t - 0.85) / 0.15) * 0.9)
        kipp = lerp(-45, -10, smooth(t / 0.5)) if t < 0.5 else lerp(-10, 8, smooth((t - 0.5) / 0.5))
    return pos, h, kipp, 0.07 * k

# Beckenhöhe aus der Beingeometrie (umgekehrtes Pendel): so hoch, wie die Standbeine es mit leicht gebeugtem Knie erlauben –
# höchster Punkt in der Standmitte, tiefster beim Doppelstand. Früher pauschal 3,5-5 cm tiefer -> dauerhaft eingeknickte Knie.
def bob_grenze(art, p):
    fn, ST, f = (gehen, 0.60, 0.998) if art == "walk" else (laufen, 0.27, 0.975)   # Stand-Anteile wie in gehen/laufen
    sw = (0.022 if art == "walk" else 0.012) * k * math.sin(2 * math.pi * p); lim = 0.0
    for s in "LR":
        if (p + (0.0 if s == "L" else 0.5)) % 1.0 >= ST: continue
        pos, h, kipp, seit = fn(p, s)
        if kipp < -8: continue                                       # Ferse hebt schon ab: Fuß kippt über den Ballen, Becken bleibt oben
        loc, _ = bein(s, pos, h, kipp, seit)
        d2 = (hip[s].x + sw - loc.x) ** 2 + (hip[s].y - loc.y) ** 2
        lim = min(lim, loc.z + math.sqrt(max(0.0, (f * LEG) ** 2 - d2)) - hip[s].z)
    return lim
BOB = {}
for _art in ("walk", "run"):
    _n = 120; _raw = np.array([bob_grenze(_art, i / _n) for i in range(_n)])
    _ker = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2); _ker /= _ker.sum()
    _sm = np.array([sum(_ker[j] * _raw[(i + j - 8) % _n] for j in range(17)) for i in range(_n)])
    # Überstand nur dort abziehen, wo das Glätten über die Beinreichweite ragt (lokal, weich verteilt) – nicht die ganze Kurve senken
    _ex = np.maximum(0.0, _sm - _raw)
    _ex = np.array([_ex[[(i + j) % _n for j in range(-6, 7)]].max() for i in range(_n)])
    _ex = np.array([sum(_ker[j] * _ex[(i + j - 8) % _n] for j in range(17)) for i in range(_n)])
    BOB[_art] = np.minimum(_sm - _ex, _raw) - 0.004 * k                    # geglättet, nie höher als die Beine reichen
    log(f"Becken {_art}: {BOB[_art].min() * 100:.1f} .. {BOB[_art].max() * 100:.1f} cm")
def bob_aus(art, p):
    t = BOB[art]; x = (p % 1.0) * len(t); i = int(x); f = x - i
    return float(t[i % len(t)] * (1 - f) + t[(i + 1) % len(t)] * f)

# ------------------------------------------------------------------ Motion-Capture-Vorlagen (echte Aufnahmen, mocap.py)
REPO_ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
MOCAP_DIR = arg("--mocap", os.path.join(REPO_, "figuren", "_mocap"))
# Mehrere echte Aufnahmen (verschiedene Personen) werden gemittelt -> keine Eigenheiten/Aufnahmefehler einer einzelnen Person
MOCAP_DATEI = {"walk": arg("--walk_bvh", "07_01.bvh").split(","),
               "run": arg("--run_bvh", "35_17.bvh,35_18.bvh,35_19.bvh,35_20.bvh").split(",")}
KONTAKT = {}
for s_ in "LR":
    a_, b_, t_ = W("foot." + s_), W("toe." + s_), W("toe." + s_, "tail")
    KONTAKT[s_] = [("foot." + s_, REST["foot." + s_].inverted() @ Vector((a_.x, a_.y + 0.06 * k, 0.0))),
                   ("toe." + s_, REST["toe." + s_].inverted() @ Vector((b_.x, b_.y, 0.0))),
                   ("toe." + s_, REST["toe." + s_].inverted() @ Vector((t_.x, t_.y, 0.0)))]
def min_kontakt():
    return min(((MW @ rig.pose.bones[b].matrix) @ loc).z for s_ in "LR" for b, loc in KONTAKT[s_])
ALLE_C = [c for pb in rig.pose.bones for c in pb.constraints]
# Finger sind bei Scan-/STL-Figuren in der Ruhepose schon entspannt (figur_finger/figur_stl) -> nur leicht nachkrümmen
FINGER_K = float(arg("--finger", 0.55))
def finger_locker(staerke=1.0):
    # Finger locker gekrümmt (lokale Z-Achse zeigt zum Handrücken -> negative Drehung um X beugt zur Handfläche).
    # Gescannte, verschmolzene Finger (haende_wie_gescannt) nur etwa halb so stark – sonst knicken die Splat-Finger.
    for s in "LR":
        f_ = staerke * (0.55 if s in rig.get("haende_wie_gescannt", "") else 1.0)
        for f, krumm in (("f_index", (14, 24, 16)), ("f_middle", (16, 27, 18)), ("f_ring", (18, 29, 19)), ("f_pinky", (20, 30, 20)), ("thumb", (0, 0, 0))):                                  # Daumen: X ist keine Beugeachse (knickte seitlich weg)
            for i, c in enumerate(krumm, 1):
                pb = rig.pose.bones.get(f"{f}.0{i}.{s}")
                if pb is not None and c: pb.rotation_quaternion = Quaternion((1, 0, 0), -math.radians(c * f_))
    bpy.context.view_layer.update()

def _ww(n, end="head"):
    pb = rig.pose.bones[n]; return MW @ (pb.head if end == "head" else pb.tail)
def handflaeche(s):
    """Normale der Handfläche (Welt) im aktuellen Bild: Handrichtung x (Kleinfinger -> Zeigefinger), Chiralität je Seite."""
    sg = 1 if s == "L" else -1
    w = _ww("hand." + s); m = _ww("f_middle.01." + s); i_ = _ww("f_index.01." + s); p_ = _ww("f_pinky.01." + s)
    d = (m - w).normalized(); q = (i_ - p_); n = d.cross(q) * sg
    return n.normalized() if n.length > 1e-6 else None
ARM_IK = []
def hand_ausrichten(s, staerke=1.0, rueck=0.3, vor=0.0):
    """Handfläche zum Oberschenkel drehen (Daumen nach vorn), wie bei locker hängenden Armen.
    60 % der Drehung übernimmt der Unterarm (Pro-/Supination um seine Achse), den Rest das Handgelenk.
    rueck/vor kippen das Ziel etwas nach hinten (Handfläche leicht nach hinten, wie beim Gehen) bzw. vorn."""
    sg = 1 if s == "L" else -1
    ziel = Vector((-sg, rueck - vor, 0.0))
    for bone, anteil in (("forearm." + s, 0.6), ("hand." + s, 1.0)):
        n = handflaeche(s)
        if n is None: return
        a = (_ww(bone, "tail") - _ww(bone)).normalized() if bone.startswith("forearm") else (_ww("f_middle.01." + s) - _ww("hand." + s)).normalized()
        n_p = (n - a * n.dot(a)); t_p = (ziel - a * ziel.dot(a))
        if n_p.length < 1e-4 or t_p.length < 1e-4: continue
        n_p.normalize(); t_p.normalize()
        ang = math.atan2(a.dot(n_p.cross(t_p)), max(-1.0, min(1.0, n_p.dot(t_p))))
        if bone.startswith("hand"): ang = max(-math.radians(45), min(math.radians(45), ang))
        rot_um_gelenk(bone, Matrix.Rotation(ang * anteil * staerke, 3, a))
def arm_ik_backen():
    """Ergebnis der Arm-IK als feste Pose übernehmen und die IK stummschalten (damit Unterarm/Hand danach gedreht werden können)."""
    bpy.context.view_layer.update()
    for s in "LR":
        Ms = {n: rig.pose.bones[n].matrix.copy() for n in ("upper_arm." + s, "forearm." + s)}
        IK["arm" + s].mute = True
        for n in ("upper_arm." + s, "forearm." + s):
            rig.pose.bones[n].matrix = Ms[n]; bpy.context.view_layer.update()
MOCAP = {}
if "--ohne_mocap" not in argv:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib, mocap as MC; importlib.reload(MC)
    for art_, dateien in MOCAP_DATEI.items():
        pfade = [os.path.join(MOCAP_DIR, d_) for d_ in dateien if os.path.exists(os.path.join(MOCAP_DIR, d_))]
        if not pfade: log("Mocap fehlt:", dateien); continue
        dauern = []
        for p_ in pfade:
            try: dauern.append(MC.zyklus(p_, rig, REST, MW, 8, lambda *a: None, art_)["dauer"])
            except Exception: pass
        n_ = max(8, int(round(sum(dauern) / max(1, len(dauern)) * FPS)))
        tab = MC.zyklus_mittel(pfade, rig, REST, MW, n_, log, art_)
        for c in ALLE_C: c.mute = True
        if art_ == "walk":
            kk = MC.knie_korrektur_berechnen(rig, MW, MWi, tab, pelvis0,
                                             lambda s_: ((MW @ rig.pose.bones[KONTAKT[s_][0][0]].matrix) @ KONTAKT[s_][0][1]).z, min_kontakt)
            log(f"Knie-Streckung um den Fersenaufsatz: max {max(kk["L"]):.0f}°")
        for c in ALLE_C: c.mute = True
        roh = []
        for i in range(n_):
            for pb in rig.pose.bones: pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
            bpy.context.view_layer.update(); MC.pose_setzen(rig, MW, MWi, tab, i, pelvis0)
            alle = min_kontakt()
            steh = [((MW @ rig.pose.bones[b].matrix) @ loc).z for s_ in "LR" if tab["kontakt"][s_][i] for b, loc in KONTAKT[s_]]
            # Boden = tiefster Punkt der Füße, die laut Aufnahme gerade aufstehen; der Schwungfuß darf nur nicht eintauchen
            roh.append(-min(steh) if steh else -alle)
            roh[-1] = max(roh[-1], -alle)
        roh = np.array(roh)
        if art_ == "walk":                                            # immer ein Fuß am Boden: je Bild aufsetzen, leicht geglättet
            dz = 0.25 * np.roll(roh, 1) + 0.5 * roh + 0.25 * np.roll(roh, -1)
        else:                                                         # Laufen hat Flugphasen: nur gemeinsam auf den Boden
            dz = np.full(n_, float(roh.max()))
        tab["dz"] = list(dz) + [float(dz[0])]
        for c in ALLE_C: c.mute = False
        MOCAP[art_] = tab
        log(f"Mocap {art_}: {n_} Bilder, Boden-Korrektur {dz.min() * 100:.1f}..{dz.max() * 100:.1f} cm")

def pose_bild(art, p):
    sg2 = 2 * math.pi * p
    for pb in rig.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    if art in MOCAP:
        tab = MOCAP[art]; i = int(round(p * tab["frames"])) % tab["frames"]
        for c in ALLE_C: c.mute = True
        MC.pose_setzen(rig, MW, MWi, tab, i, pelvis0)
        rot_um_gelenk("spine", Matrix.Identity(3), Vector((0, 0, tab["dz"][i])))
        for s_ in "LR": hand_ausrichten(s_, 0.85, rueck=0.35)
        finger_locker(FINGER_K)
        return
    for c in ALLE_C: c.mute = False
    if art == "walk":
        bob = bob_aus("walk", p)                                     # Pendel: hoch in der Standmitte, tief beim Doppelstand
        sway = 0.022 * k * math.sin(sg2); yaw = 4.0 * math.cos(sg2); roll = 3.0 * math.sin(sg2); lean = 2.5
        rumpf_yaw = -8.0 * math.cos(sg2); arm_amp, beuge0, beuge_amp = 22.0, 12.0, 14.0
        fn = gehen
    elif art == "run":
        q = (p - 0.135) % 0.5 / 0.5                                # Standmitte bei p = 0.135 (Stand 27 %)
        bob = min(-0.06 * k + 0.035 * k * (1 - math.cos(2 * math.pi * q)), bob_aus("run", p))   # tief in der Standmitte, hoch im Flug
        sway = 0.012 * k * math.sin(sg2); yaw = 8.0 * math.cos(sg2); roll = 4.0 * math.sin(sg2); lean = 9.0
        rumpf_yaw = -15.0 * math.cos(sg2); arm_amp, beuge0, beuge_amp = 38.0, 82.0, 12.0
        fn = laufen
    else:  # idle (+ Kopfneigen tiltL/tiltR): natürlicher Stand – Gewicht auf dem linken Bein (Kontrapost), rechtes Knie locker,
        # Becken zur Standseite verschoben und auf der Spielbeinseite gesenkt, Schultern gegengleich, Arme hängen locker
        breath = math.sin(sg2 * 4) if art == "idle" else math.sin(sg2 * 1.5)
        shift = 0.75 + 0.25 * math.sin(sg2) if art == "idle" else 0.75   # Gewicht bleibt links, pendelt nur leicht
        bob = -0.012 * k + 0.002 * k * breath; sway = 0.03 * k * shift; yaw = 3.0 * shift; roll = -3.0 * shift; lean = 0.5
        rumpf_yaw = -2.0 * shift
    # Becken (spine = Wurzel)
    rot_um_gelenk("spine", E(pitch=lean * 0.4, roll=roll, yaw=yaw), Vector((sway, 0, bob)))
    # Wirbelsäule: Gegenrotation, Vorlage, Atmung; Hals/Kopf stabilisieren den Blick nach vorn
    IDLE = art in ("idle", "tiltL", "tiltR")
    if IDLE:
        rot_um_gelenk("spine.001", E(roll=-roll * 0.45))                                     # Rumpf richtet sich über dem Standbein auf
        rot_um_gelenk("spine.002", E(pitch=0.6 * breath, yaw=rumpf_yaw * 0.5, roll=-roll * 0.35)); rot_um_gelenk("spine.003", E(pitch=0.5 * breath, roll=-roll * 0.2))
    else:
        rot_um_gelenk("spine.001", E(pitch=lean * 0.25, roll=-roll * 0.5)); rot_um_gelenk("spine.002", E(pitch=lean * 0.2, yaw=(rumpf_yaw - yaw) * 0.45))
        rot_um_gelenk("spine.003", E(pitch=lean * 0.15, yaw=(rumpf_yaw - yaw) * 0.55, roll=-roll * 0.3))
    # Kopf: Welt-Ausrichtung zurück auf geradeaus (Blickstabilisierung wie in Spielen)
    for n, anteil in (("spine.004", 0.35), ("spine.005", 0.35), ("spine.006", 0.3)):
        Mcur = (MW @ rig.pose.bones[n].matrix).to_3x3().normalized(); Mrest = REST[n].to_3x3().normalized()
        Rfix = (Mrest @ Mcur.inverted()).to_quaternion(); Rpart = Quaternion().slerp(Rfix, anteil)
        rot_um_gelenk(n, Rpart.to_matrix() @ (E(pitch=-0.8 * breath, roll=1.5) if IDLE and n == "spine.006" else Matrix.Identity(3)))
    if art in ("tiltL", "tiltR"):                                    # Kopf zur Seite neigen und etwas drehen, dann zurück
        env = math.sin(math.pi * p) ** 2; sgn = 1 if art == "tiltL" else -1
        rot_um_gelenk("spine.005", E(roll=sgn * 4 * env, yaw=sgn * 6 * env)); rot_um_gelenk("spine.006", E(roll=sgn * 7 * env, yaw=sgn * 10 * env, pitch=3 * env))
    # Beine
    for s in "LR":
        sg = 1 if s == "L" else -1
        if IDLE:                                                    # Standbein links unter der Hüfte, Spielbein etwas vor und außen
            loc, R = bein(s, 0.01 * k, 0, 0, 0.095 * k) if s == "L" else bein(s, 0.07 * k, 0, 0, 0.125 * k)
        else:
            pos, h, kipp, seit = fn(p, s); loc, R = bein(s, pos, h, kipp, seit)
            # reicht das Bein bei dieser Beckenhöhe nicht bis zum Knöchel, hebt sich die Ferse weiter (Abrollen über den Ballen)
            if kipp <= 0 and h < 0.002 * k:
                hz = hip[s].z + bob; d2 = (hip[s].x + sway - loc.x) ** 2 + (hip[s].y - loc.y) ** 2
                z_max = hz - math.sqrt(max(0.0, (0.998 * LEG) ** 2 - d2))
                if loc.z < z_max:
                    for _ in range(40):
                        kipp -= 1.5; loc, R = bein(s, pos, h, kipp, seit)
                        if loc.z >= z_max or kipp < -70: break
        set_empty(T_foot[s], loc, R)
        set_empty(P_knee[s], Vector((sg * 0.12 * k + sway, hip[s].y - 0.8 * k, (hip[s].z + loc.z) * 0.5)))
    bpy.context.view_layer.update()
    # Arme (nach dem Rumpf, Schulter ist dann richtig gesetzt)
    for s in "LR":
        sg = 1 if s == "L" else -1
        if IDLE:                                                    # Arme hängen locker neben dem Oberschenkel, Ellbogen leicht gebeugt
            tgt, pole = arm_ziel(s, 3.0 + 1.0 * math.sin(sg2 + (0 if s == "L" else 1.3)), 15.0, abd=9.0 if s == "L" else 10.0)   # Arme frei neben dem Rumpf
        else:
            phase = math.cos(sg2) * (-1 if s == "L" else 1)       # linkes Bein vorn (p=0) -> rechter Arm vorn
            if art == "walk":                                     # nach hinten weiter als nach vorn (+15° / -25°), Ellbogen vorn 37°
                schwung = (21.0 if phase > 0 else 26.0) * phase - 1.0         # vorn +21°, hinten -26°
                beuge = 15.0 + 28.0 * max(0.0, phase)                   # Ellbogen vorn ~43°
            else:
                schwung = arm_amp * phase + 6.0
                beuge = beuge0 + beuge_amp * max(0.0, phase)
            tgt, pole = arm_ziel(s, schwung, beuge, abd=1.5 if art == "walk" else 3.0, innen=(0.06 * k * max(0.0, phase) if art == "run" else 0.0))
        set_empty(T_hand[s], tgt); set_empty(P_elb[s], pole)
    if art in ("idle", "tiltL", "tiltR"):
        arm_ik_backen()
        for s in "LR": hand_ausrichten(s, 1.0, rueck=0.15)
    finger_locker(FINGER_K * (1.15 if art in ("idle", "tiltL", "tiltR") else 1.0))

# Pole-Winkel einmal bestimmen (mit den IK-Zyklen, Mocap kurz beiseite)
_mc = dict(MOCAP); MOCAP.clear()
for s in "LR":
    a_ = pole_waehlen(IK["leg" + s], "thigh." + s, lambda: pose_bild("walk", 0.25))
    b_ = pole_waehlen(IK["arm" + s], "upper_arm." + s, lambda: pose_bild("walk", 0.0))
    log(f"Pole {s}: Bein {a_}°, Arm {b_}°")
MOCAP.update(_mc)

# ------------------------------------------------------------------ Stoff: Mantel und Schal (Knochen aus stoff.py)
# Vorbilder: lange Wollmäntel in Gang-Referenzen/Mocap-Kleidung (Saum schwingt mit den Beinen, das vorschwingende Knie schiebt die
# Vorderbahn nach vorn, hinten schleppt der Saum etwas nach, beim Laufen flattert er nach hinten auf) und hängende Schalenden
# (federn bei jedem Schritt mit dem Becken-/Brust-Hub, schwingen seitlich, wehen beim Laufen nach hinten).
# Simulation je Clip: Verlet-Punkte an den Knochen-Enden, Federn zur angehängten Ruheform, Längen fest, Schwerkraft, Dämpfung,
# Fahrtwind (Zyklen laufen auf der Stelle -> Luft strömt mit Gehtempo von vorn), Kollision mit Ober-/Unterschenkeln bzw. Brust.
# 4 Zyklen vorlaufen, der letzte wird gebacken (nahtlos).
STOFF = sorted(b.name for b in rig.data.bones if b.name.startswith(("mantel.", "schal.")))
KETTEN = {}
for n_ in STOFF: KETTEN.setdefault(n_.rsplit(".", 1)[0], []).append(n_)
for k_ in KETTEN: KETTEN[k_].sort(key=lambda n_: int(n_.rsplit(".", 1)[1]))
TEMPO = {"walk": 1.5, "run": 4.2}
# Federn außerhalb der Doppelschritt-Frequenz (~1 Hz): Mantel ~1,3 Hz, zeta ~0,45; Schal ~0,9 Hz
STOFF_P = {"mantel": dict(k=70.0, d=7.0, wind=0.9, g=0.5, coll="beine"), "schal": dict(k=30.0, d=3.0, wind=1.6, g=0.5, coll="brust")}
WIND_RUN = 2.5
if STOFF: log("Stoff-Ketten:", {k_: len(v) for k_, v in KETTEN.items()})
def _w(n, end="head"):
    pb = rig.pose.bones[n]; return np.array(MW @ (pb.head if end == "head" else pb.tail))
def _kapsel(x, a, b, r):
    ab = b - a; t = np.clip(np.dot(x - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1); c = a + t * ab; d = x - c; l = np.linalg.norm(d)
    return x if l >= r or l < 1e-9 else c + d / l * r
# Mantel-Ring: Nachbarbahnen gleicher Ebene gekoppelt; vorn unten (Knopfleiste offen) und hinten (Gehschlitz) dürfen klaffen
_MRING = [kn for kn in KETTEN if kn.startswith("mantel.")]
_MORD = ["V1L", "V2L", "H2L", "H1L", "H1R", "H2R", "V2R", "V1R"]
_MRING = [f"mantel.{n_}" for n_ in _MORD if f"mantel.{n_}" in KETTEN]
def stoff_sim(art, frames):
    """Pass 1: Körperpose je Bild -> Zielpunkte (Stoff in Ruhehaltung angehängt) + Kollisionskörper; dann Simulation."""
    def aufnahme():
        rec = {"k": {}, "kap": []}
        for kn, bones in KETTEN.items():
            rec["k"][kn] = [_w(bones[0])] + [_w(b, "tail") for b in bones]
        for s_ in "LR":
            rec["kap"].append(("beine", _w("thigh." + s_), _w("thigh." + s_, "tail"), 0.085 * k))
            rec["kap"].append(("beine", _w("shin." + s_), _w("shin." + s_, "tail"), 0.065 * k))
        mL = 0.5 * (_w("thigh.L") + _w("thigh.L", "tail")); mR = 0.5 * (_w("thigh.R") + _w("thigh.R", "tail"))
        rec["kap"].append(("beine", mL, mR, 0.08 * k)); rec["kap"].append(("beine", _w("shin.L"), _w("shin.R"), 0.065 * k))
        rec["kap"].append(("brust", _w("spine.001"), _w("spine.003", "tail"), 0.105 * k))
        return rec
    for c in ALLE_C: c.mute = True                                     # Ruhelage (Bindpose) für die Kollisions-Abstände
    for pb in rig.pose.bones: pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
    bpy.context.view_layer.update(); RUHE = aufnahme()
    ziel = []
    for f in range(frames):
        pose_bild(art, f / frames)
        rec = {"k": {}, "kap": []}
        for kn, bones in KETTEN.items():
            rec["k"][kn] = [_w(bones[0])] + [_w(b, "tail") for b in bones]
        for s_ in "LR":
            rec["kap"].append(("beine", _w("thigh." + s_), _w("thigh." + s_, "tail"), 0.085 * k))
            rec["kap"].append(("beine", _w("shin." + s_), _w("shin." + s_, "tail"), 0.065 * k))
        # Brücken zwischen den Beinen (Oberschenkelmitte L-R, Knie L-R): kein Stoff fällt zwischen die Beine -> keine "Mantelbeine"
        mL = 0.5 * (_w("thigh.L") + _w("thigh.L", "tail")); mR = 0.5 * (_w("thigh.R") + _w("thigh.R", "tail"))
        rec["kap"].append(("beine", mL, mR, 0.08 * k)); rec["kap"].append(("beine", _w("shin.L"), _w("shin.R"), 0.065 * k))
        rec["kap"].append(("brust", _w("spine.001"), _w("spine.003", "tail"), 0.105 * k))
        ziel.append(rec)
    # Brustkapsel nie größer als der Ruheabstand des Schals (sonst drückt sie ihn dauerhaft nach vorn)
    r_brust = 0.105 * k
    for kn in KETTEN:
        if not kn.startswith("schal"): continue
        a0, a1 = ziel[0]["kap"][-1][1], ziel[0]["kap"][-1][2]
        for p_ in ziel[0]["k"][kn][1:]:
            ab = a1 - a0; t_ = np.clip(np.dot(p_ - a0, ab) / max(ab @ ab, 1e-9), 0, 1)
            r_brust = min(r_brust, float(np.linalg.norm(p_ - (a0 + t_ * ab))) - 0.005)
    # Kollisionsradius je Stoff-Punkt und Körper: höchstens der Abstand in der Ausgangslage (sonst startet der Mantel "im Bein"
    # und wird nach oben geschleudert); ab dort darf er nur nicht tiefer eindringen
    def _abst(x, a, b):
        ab = b - a; t = np.clip(np.dot(x - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1); return float(np.linalg.norm(x - (a + t * ab)))
    KR = {kn: [[min(r + 0.012, _abst(RUHE["k"][kn][i], a0, a1) - 0.004) for (nm_, a0, a1, r) in RUHE["kap"]] for i in range(len(RUHE["k"][kn]))] for kn in KETTEN}
    V = TEMPO.get(art, 0.0); n_sub = 8; dt = 1.0 / FPS / n_sub
    zust = {kn: dict(x=[p.copy() for p in ziel[0]["k"][kn]], v=[np.zeros(3) for _ in ziel[0]["k"][kn]],
                     L=[np.linalg.norm(ziel[0]["k"][kn][i + 1] - ziel[0]["k"][kn][i]) for i in range(len(KETTEN[kn]))]) for kn in KETTEN}
    ring0 = {}                                                           # Ruheabstände Nachbarbahnen je Ebene
    for j in range(len(_MRING)):
        a_, b_ = _MRING[j], _MRING[(j + 1) % len(_MRING)]
        for e in range(1, len(KETTEN[a_]) + 1):
            ring0[(a_, b_, e)] = float(np.linalg.norm(ziel[0]["k"][a_][e] - ziel[0]["k"][b_][e]))
    def offen(a_, b_, e):                                                # vorn unten (Knopfleiste) / hinten Schlitz
        par = {a_.split(".")[1], b_.split(".")[1]}
        return (par == {"V1L", "V1R"} and e >= 2) or (par == {"H1L", "H1R"} and e >= 2)
    out = [dict() for _ in range(frames)]
    for runde in range(4):
        for f in range(frames):
            z0, z1 = ziel[f], ziel[(f + 1) % frames]
            for ss in range(n_sub):
                u = (ss + 1) / n_sub
                kap = [(nm_, a0 * (1 - u) + b0 * u, a1 * (1 - u) + b1 * u, (r_brust if nm_ == "brust" else r))
                       for (nm_, a0, a1, r), (_, b0, b1, _) in zip(z0["kap"], z1["kap"])]
                alt = {}
                for kn in KETTEN:
                    P = STOFF_P["mantel" if kn.startswith("mantel") else "schal"]; st_ = zust[kn]
                    T = [z0["k"][kn][i] * (1 - u) + z1["k"][kn][i] * u for i in range(len(z0["k"][kn]))]
                    st_["T"] = T; st_["x"][0] = T[0]; alt[kn] = [x.copy() for x in st_["x"]]
                    w_ = (WIND_RUN if art == "run" and kn.startswith("mantel") else P["wind"])
                    for i in range(1, len(T)):
                        x, v = st_["x"][i], st_["v"][i]
                        rel = np.array([0.0, V, 0.0]) - v                              # Fahrtwind von vorn (Blick -Y), quadratisch
                        a_ = P["k"] * (T[i] - x) - P["d"] * v + w_ * 0.35 * rel * min(3.0, np.linalg.norm(rel)) + np.array([0, 0, -9.81 * P["g"]])
                        st_["v"][i] = v + a_ * dt; st_["x"][i] = x + st_["v"][i] * dt
                for it in range(4):
                    for kn in KETTEN:                                                   # Längen von der Wurzel aus
                        st_ = zust[kn]
                        for i in range(1, len(st_["x"])):
                            d = st_["x"][i] - st_["x"][i - 1]; l = np.linalg.norm(d)
                            if l > 1e-9: st_["x"][i] = st_["x"][i - 1] + d / l * st_["L"][i - 1]
                    for j in range(len(_MRING)):                                        # Mantel-Ring
                        a_, b_ = _MRING[j], _MRING[(j + 1) % len(_MRING)]
                        for e in range(1, len(KETTEN[a_]) + 1):
                            L0 = ring0[(a_, b_, e)]; xa, xb = zust[a_]["x"][e], zust[b_]["x"][e]; d = xb - xa; l = np.linalg.norm(d)
                            lo, hi_ = (0.85 * L0, 1.6 * L0) if offen(a_, b_, e) else (0.85 * L0, 1.03 * L0)
                            if l < 1e-9 or lo <= l <= hi_: continue
                            korr = (l - np.clip(l, lo, hi_)) * 0.5 * d / l
                            zust[a_]["x"][e] = xa + korr; zust[b_]["x"][e] = xb - korr
                    for kn in KETTEN:
                        P = STOFF_P["mantel" if kn.startswith("mantel") else "schal"]; st_ = zust[kn]
                        for j_, (nm_, a0, a1, r) in enumerate(kap):
                            if nm_ != P["coll"]: continue
                            for i in range(1, len(st_["x"])):
                                rr = min(r + 0.012, KR[kn][i][j_]) if nm_ != "brust" else r + 0.012
                                if rr > 0.01: st_["x"][i] = _kapsel(st_["x"][i], a0, a1, rr)
                for kn in KETTEN:                                                       # Geschwindigkeit aus der echten Bewegung
                    st_ = zust[kn]
                    for i in range(1, len(st_["x"])): st_["v"][i] = (st_["x"][i] - alt[kn][i]) / dt
            if runde == 3:
                for kn in KETTEN: out[f][kn] = [x.copy() for x in zust[kn]["x"]]
    return out
def stoff_anwenden(sim_f):
    for kn, bones in KETTEN.items():
        X = sim_f[kn]
        for i, b in enumerate(bones):
            h, t = _w(b), _w(b, "tail"); a_ = Vector(t - h); z_ = Vector(X[i + 1] - h)
            if a_.length < 1e-6 or z_.length < 1e-6: continue
            rot_um_gelenk(b, a_.rotation_difference(z_).to_matrix())

# ------------------------------------------------------------------ Aufnehmen -> Aktionen
def aufnehmen(art, name, frames):
    rows = []
    sim = stoff_sim(art, frames) if STOFF else None
    for f in range(frames + 1):
        pose_bild(art, (f % frames) / frames)
        if sim is not None: stoff_anwenden(sim[f % frames])
        rows.append({pb.name: (rig.convert_space(pose_bone=pb, matrix=pb.matrix, from_space='POSE', to_space='LOCAL')) for pb in rig.pose.bones})
    old = bpy.data.actions.get(name)
    if old: bpy.data.actions.remove(old)
    for c in [c for pb in rig.pose.bones for c in pb.constraints]: c.mute = True
    act = bpy.data.actions.new(name); act.use_fake_user = True
    rig.animation_data.action = act
    if rig.animation_data.action_slot is None and act.slots: rig.animation_data.action_slot = act.slots[0]
    prev = {}
    for f, row in enumerate(rows):
        for n, M in row.items():
            pb = rig.pose.bones[n]; q = M.to_quaternion()
            if n in prev and prev[n].dot(q) < 0: q.negate()
            prev[n] = q; pb.rotation_quaternion = q; pb.keyframe_insert("rotation_quaternion", frame=f)
            if n == "spine":
                pb.location = M.translation; pb.keyframe_insert("location", frame=f)
    for c in [c for pb in rig.pose.bones for c in pb.constraints]: c.mute = False
    rig.animation_data.action = None
    log(f"{name}: {frames} Bilder ({frames / FPS:.2f} s)")
    return act

acts = [aufnehmen("walk", ID + "_Walk", MOCAP["walk"]["frames"] if "walk" in MOCAP else 24),
        aufnehmen("run", ID + "_Run", MOCAP["run"]["frames"] if "run" in MOCAP else 17),
        aufnehmen("idle", ID + "_Idle_Breathing", 96),
        aufnehmen("tiltL", ID + "_HeadTilt_L", int(2.6 * FPS)), aufnehmen("tiltR", ID + "_HeadTilt_R", int(2.6 * FPS))]
_tasche = bpy.data.actions.get(ID + "_Pockets")                      # Carlos' Hände-in-die-Taschen passt nicht zu freien Scan-Armen
if _tasche: bpy.data.actions.remove(_tasche)

# IK aufräumen, Ruhepose
for pb in rig.pose.bones:
    for c in list(pb.constraints): pb.constraints.remove(c)
    pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0)
for o in list(T_foot.values()) + list(P_knee.values()) + list(T_hand.values()) + list(P_elb.values()): bpy.data.objects.remove(o, do_unlink=True)

# ------------------------------------------------------------------ Export (GLB neu, Splats/Skin bleiben – Gelenk-Reihenfolge wird geprüft)
if OUT:
    for o in scene.objects: o.select_set(o in (game, rig))
    bpy.context.view_layer.objects.active = rig
    # nur die Clips dieser Figur exportieren (Aktionen anderer Figuren haben dieselben Knochennamen und würden mitexportiert)
    for _a in [x for x in bpy.data.actions if not x.name.startswith(ID + "_")]: bpy.data.actions.remove(_a)
    glb = os.path.join(OUT, ID + ".glb")
    kw = dict(filepath=glb, use_selection=True, use_active_scene=True, export_format='GLB', export_animations=True, export_animation_mode='ACTIONS',
              export_skins=True, export_force_sampling=True, export_def_bones=False, export_materials='NONE')
    try: bpy.ops.export_scene.gltf(**kw, export_vertex_color='ACTIVE')
    except TypeError: bpy.ops.export_scene.gltf(**kw, export_colors=True)
    g = open(glb, "rb").read(); n = struct.unpack('<I', g[12:16])[0]; gj = json.loads(g[20:20 + n])
    joints = [gj['nodes'][i]['name'] for i in gj['skins'][0]['joints']]
    sk = json.load(open(os.path.join(OUT, ID + "_skin.json")))
    ok = joints == sk["joints"]
    print("REPORT", json.dumps(dict(id=ID, clips=sorted(a['name'] for a in gj['animations']), gelenke_gleich=ok, mb=round(len(g) / 1e6, 1))))
    if not ok: raise SystemExit("Gelenk-Reihenfolge geändert – Skin neu schreiben!")

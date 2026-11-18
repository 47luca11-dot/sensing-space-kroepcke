# Echte Bewegungsaufnahmen (Motion Capture, BVH) auf eine geriggte Scan-Figur übertragen – läuft in Blender, wird von gang.py benutzt.
#
# Vorlagen: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu, "free for use in research and commercial projects"),
# BVH-Umsetzung von B. Hahne (MotionBuilder-Namen, Bild 1 = T-Pose). Liegen in figuren/_mocap/ (07_01.bvh Gehen – Knie im Stand ~9°, am nächsten an der Normkurve; 35_17.bvh Joggen).
#
# Übertragung je Knochen in Weltkoordinaten:
#   D(t)  = Welt-Drehung des Quell-Gelenks zur Zeit t  ·  (Welt-Drehung in der T-Pose)^-1      (Bewegung relativ zur T-Pose)
#   C     = kürzester Bogen von der Ruhe-Richtung unseres Knochens zur T-Pose-Richtung des Quellknochens (A-Pose -> T-Pose)
#   Ziel  = D(t) · C · Ruhe-Drehung unseres Knochens
# Ein Gangzyklus (Fersenaufsatz links bis zum nächsten) wird ausgeschnitten, nahtlos geschlossen, die Vorwärtsbewegung des Beckens
# entfernt (auf der Stelle; der Standfuß gleitet nach hinten – daraus misst das Spiel das Tempo), Becken-Höhe/-Pendel mit dem
# Beinlängen-Verhältnis skaliert. Danach steht in jedem Bild der tiefste Fußpunkt (Ferse/Ballen/Zehe) auf dem Boden (geglättet).
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion

ZUORDNUNG = {
    "spine": "Hips", "spine.001": "LowerBack", "spine.002": "Spine", "spine.003": "Spine1",
    "spine.004": "Neck", "spine.005": "Neck1", "spine.006": "Head",
    "shoulder.L": "LeftShoulder", "upper_arm.L": "LeftArm", "forearm.L": "LeftForeArm", "hand.L": "LeftHand",
    "shoulder.R": "RightShoulder", "upper_arm.R": "RightArm", "forearm.R": "RightForeArm", "hand.R": "RightHand",
    "thigh.L": "LeftUpLeg", "shin.L": "LeftLeg", "foot.L": "LeftFoot", "toe.L": "LeftToeBase",
    "thigh.R": "RightUpLeg", "shin.R": "RightLeg", "foot.R": "RightFoot", "toe.R": "RightToeBase",
}
# Knochen, deren Ruhe-Richtung an die T-Pose angeglichen wird (Arme/Beine); Wirbelsäule/Kopf stehen in beiden aufrecht
ANGLEICHEN = {"shoulder", "upper_arm", "forearm", "thigh", "shin", "foot", "toe"}
REIHENFOLGE = ["spine", "spine.001", "spine.002", "spine.003", "spine.004", "spine.005", "spine.006",
               "shoulder.L", "upper_arm.L", "forearm.L", "hand.L", "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
               "thigh.L", "shin.L", "foot.L", "toe.L", "thigh.R", "shin.R", "foot.R", "toe.R"]


def _laden(pfad):
    alt = set(bpy.data.objects); szene = bpy.context.window.scene
    tmp = bpy.data.scenes.new("_mocap_tmp"); bpy.context.window.scene = tmp
    bpy.ops.import_anim.bvh(filepath=pfad, global_scale=1.0, frame_start=0, use_fps_scale=False, update_scene_fps=False,
                            update_scene_duration=False, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')
    A = [o for o in bpy.data.objects if o not in alt and o.type == 'ARMATURE'][0]
    return A, tmp, szene


def _welt(A, n):
    return (A.matrix_world @ A.pose.bones[n].matrix)


def _spiegel(q):
    """Bewegung an der Körpermitte (x = 0) spiegeln: S·R·S mit S = diag(-1, 1, 1)."""
    return Quaternion((q.w, q.x, -q.y, -q.z))


def _partner(n):
    return n[:-2] + (".R" if n.endswith(".L") else ".L") if n.endswith((".L", ".R")) else n


def zyklus(pfad, rig, REST, MW, frames, log=print, art="walk", symmetrisch=True, glatt=True):
    """Liest die BVH, schneidet einen Zyklus aus und liefert je Bild (frames Stück, p = i/frames) die Welt-Drehungen unserer Knochen
    und den Becken-Versatz. Rückgabe dict(rot={knochen: [Matrix3]}, root=[Vector], speed=m/s, dauer=s)."""
    A, tmp, szene = _laden(pfad)
    act = A.animation_data.action; f_a, f_e = int(act.frame_range[0]), int(act.frame_range[1])
    tmp.frame_set(f_a)                                            # Bild 1 = T-Pose
    T = {s: _welt(A, s).to_3x3().normalized() for s in set(ZUORDNUNG.values())}
    Tdir = {s: (A.matrix_world @ A.pose.bones[s].tail - A.matrix_world @ A.pose.bones[s].head) for s in set(ZUORDNUNG.values())}
    hipT = (A.matrix_world @ A.pose.bones["Hips"].head).copy()
    leg_src = sum((A.pose.bones[b].tail - A.pose.bones[b].head).length for b in ("LeftUpLeg", "LeftLeg"))
    th = rig.data.bones["thigh.L"]; sh = rig.data.bones["shin.L"]
    leg_tgt = th.length + sh.length
    s = leg_tgt / leg_src
    # Fußkontakte (links) über die Aufnahme: Standphase = Fuß fast in Ruhe
    fr = list(range(f_a + 1, f_e + 1))
    P = []; H = []
    for f in fr:
        tmp.frame_set(f)
        P.append(np.array(A.matrix_world @ A.pose.bones["LeftToeBase"].head)); H.append(np.array(A.matrix_world @ A.pose.bones["Hips"].head))
    P = np.array(P); H = np.array(H)
    fps_src = 120.0
    v = np.r_[0, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)] * fps_src
    v_h = np.median(np.linalg.norm(np.diff(H[:, :2], axis=0), axis=1)) * fps_src
    still = v < (0.35 if art == "walk" else 0.55) * v_h
    starts = [i for i in range(1, len(fr)) if still[i] and not still[i - 1] and still[min(i + (3 if art == "walk" else 1), len(fr) - 1)]]
    if len(starts) < 2:
        raise RuntimeError(f"Mocap {pfad}: keine zwei Fußaufsätze gefunden")
    mid = len(fr) / 2; i1 = min(range(len(starts) - 1), key=lambda j: abs((starts[j] + starts[j + 1]) / 2 - mid))
    a_i, b_i = starts[i1], starts[i1 + 1]
    fa, fb = fr[a_i], fr[b_i]
    dauer = (fb - fa) / fps_src
    H_a, H_b = H[a_i], H[b_i]
    speed = float(np.linalg.norm((H_b - H_a)[:2]) / dauer * s)
    log(f"Mocap {pfad.replace(chr(92), "/").split("/")[-1]}: Zyklus Bild {fa}-{fb} ({dauer:.2f} s), Maßstab {s:.4f}, Tempo {speed:.2f} m/s")
    # Ausrichtung: Ziel-Ruhe -> T-Pose (nur Arme/Beine)
    C = {}
    for n, src in ZUORDNUNG.items():
        if n.split(".")[0] in ANGLEICHEN:
            rd = REST[n].to_3x3().col[1].normalized(); td = Tdir[src].normalized()
            C[n] = rd.rotation_difference(td).to_matrix()
        else:
            C[n] = Matrix.Identity(3)
    for side in "LR":                                              # Hand folgt der Unterarm-Angleichung (Quell-Hand ohne Länge)
        C["hand." + side] = C["forearm." + side]
    # Abtasten
    Dq = {n: [] for n in ZUORDNUNG}; root = []; fuss_h = {"L": [], "R": []}
    for i in range(frames + 1):
        p = i / frames; ff = fa + p * (fb - fa); fi = int(math.floor(ff)); tmp.frame_set(fi, subframe=ff - fi)
        for n, src in ZUORDNUNG.items():
            Dq[n].append((_welt(A, src).to_3x3().normalized() @ T[src].inverted()).to_quaternion())
        h = A.matrix_world @ A.pose.bones["Hips"].head
        root.append(Vector(h))
        for sd, nm in (("L", "Left"), ("R", "Right")):              # tiefster Fußpunkt der Aufnahme (Bodenkontakt-Phasen)
            pb_ = A.pose.bones
            fuss_h[sd].append(min((A.matrix_world @ pb_[nm + "Foot"].head).z, (A.matrix_world @ pb_[nm + "ToeBase"].head).z,
                                  (A.matrix_world @ pb_[nm + "ToeBase"].tail).z))
    # Becken: Vorwärtsbewegung (linearer Trend) entfernen, Rest skalieren
    R0, R1 = root[0].copy(), root[-1].copy(); off = []
    for i, h in enumerate(root):
        p = i / frames; lin = R0.lerp(R1, p)
        d = h - lin
        d.z = h.z - hipT.z                                           # Höhe relativ zur T-Pose (Stand)
        off.append(d * s)
    mx = sum((o.x for o in off), 0.0) / len(off); my = sum((o.y for o in off), 0.0) / len(off)
    for o in off: o.x -= mx; o.y -= my
    # Nahtlos schließen: Fehler am Zyklusende linear über den Zyklus verteilen
    for n in ZUORDNUNG:
        q0, q1 = Dq[n][0], Dq[n][-1]
        if q0.dot(q1) < 0: q1 = -q1
        delta = q0 @ q1.inverted()                                  # dreht q1 auf q0
        for i in range(frames + 1):
            p = i / frames; corr = Quaternion().slerp(delta, p)
            Dq[n][i] = corr @ Dq[n][i]
    N = frames
    def bei(seq, x):                                                # zyklisch interpoliert (x in Bildern)
        x %= N; i0 = int(math.floor(x)); f = x - i0; a_, b_ = seq[i0 % N], seq[(i0 + 1) % N]
        if a_.dot(b_) < 0: b_ = -b_
        return a_.slerp(b_, f)
    if symmetrisch:
        # Links/rechts angleichen: jede Seite = Mittel aus sich selbst und der gespiegelten anderen Seite einen halben Zyklus
        # später (Mitte: sich selbst gespiegelt). Entfernt Eigenheiten wie ein steiferes rechtes Knie.
        neu = {}
        for n in ZUORDNUNG:
            pa = _partner(n); reihe = []
            for i in range(N):
                q = Dq[n][i]; m = _spiegel(bei(Dq[pa], i + N / 2))
                if q.dot(m) < 0: m = -m
                reihe.append(q.slerp(m, 0.5))
            neu[n] = reihe + [reihe[0].copy()]
        Dq = neu
        half = [off[int(round(i + N / 2)) % N] for i in range(N)]
        off = [Vector((0.5 * (o.x - h.x), 0.5 * (o.y + h.y), 0.5 * (o.z + h.z))) for o, h in zip(off[:N], half)] + [None]
        off[-1] = off[0].copy()
    if glatt:
        # leichte zeitliche Glättung (1-2-1) gegen Aufnahmefehler wie auf 0° festgeklemmte Knie mit anschließendem Sprung
        for n in ZUORDNUNG:
            src_ = Dq[n][:N]; reihe = []
            for i in range(N):
                a_, b_, c_ = src_[(i - 1) % N], src_[i], src_[(i + 1) % N]
                if a_.dot(b_) < 0: a_ = -a_
                if c_.dot(b_) < 0: c_ = -c_
                reihe.append(a_.slerp(c_, 0.5).slerp(b_, 0.5))
            Dq[n] = reihe + [reihe[0].copy()]
    def mittel(seq):
        ref = seq[0]; acc = Quaternion((0, 0, 0, 0))
        for q in seq:
            if q.dot(ref) < 0: q = -q
            acc = Quaternion((acc.w + q.w, acc.x + q.x, acc.y + q.y, acc.z + q.z))
        acc.normalize(); return acc
    for side in "LR":
        sg = 1 if side == "L" else -1
        for i in range(N + 1):
            # Spur: Beine je ~2° nach außen (Füße setzen nicht auf einer Linie / kreuzen nicht)
            ab = Quaternion((0, 1, 0), math.radians(-2.0 * sg))
            for b in ("thigh.", "shin."): Dq[b + side][i] = ab @ Dq[b + side][i]
        if art == "walk":
            # Arme ruhiger: Pendel um 25 % kleiner (um die Mittelhaltung), Ellbogen 30 % weniger gebeugt, Handgelenk halb so aktiv
            mu = mittel(Dq["upper_arm." + side][:N])
            for i in range(N + 1):
                u = Dq["upper_arm." + side][i]
                if u.dot(mu) < 0: u = -u
                Dq["upper_arm." + side][i] = mu.slerp(u, 0.75)
                f = Dq["forearm." + side][i]; u2 = Dq["upper_arm." + side][i]
                if f.dot(u2) < 0: f = -f
                Dq["forearm." + side][i] = u2.slerp(f, 0.7)
                h = Dq["hand." + side][i]; f2 = Dq["forearm." + side][i]
                if h.dot(f2) < 0: h = -h
                Dq["hand." + side][i] = f2.slerp(h, 0.5)
    rot = {n: [(Dq[n][i].to_matrix() @ C[n] @ REST[n].to_3x3().normalized()).to_quaternion() for i in range(N + 1)] for n in ZUORDNUNG}
    # Bodenkontakt je Seite aus der Aufnahme (Fußpunkt höchstens 2,5 cm über seinem Tiefstwert); bei Symmetrie rechts = links + N/2
    kontakt = {}
    for sd in "LR":
        hh = [(v - min(fuss_h[sd][:N])) * s for v in fuss_h[sd][:N]]
        kontakt[sd] = [x < 0.025 for x in hh]
    if symmetrisch:
        kontakt["R"] = [kontakt["L"][int(round(i + N / 2)) % N] for i in range(N)]
    # Verdrehung um die Knochenachse (Arm/Hand): die Übertragung T-Pose -> A-Pose legt nur die Richtung fest; die mittlere
    # Drehung um die eigene Achse wird auf die Ruhepose zurückgesetzt (Handfläche zum Oberschenkel, Daumen vorn – nicht nach oben)
    for side in "LR":
        for b in ("upper_arm.", "forearm.", "hand."):
            n = b + side; Rr = REST[n].to_3x3().normalized().to_quaternion(); tw = []
            for q in rot[n][:N]:
                loc = Rr.inverted() @ q                                  # Drehung relativ zur Ruhe, im Knochen-Raum
                t = Quaternion((loc.w, 0.0, loc.y, 0.0)); t.normalize()  # Anteil um die Knochenachse (y)
                a = 2 * math.atan2(t.y, t.w); a = (a + math.pi) % (2 * math.pi) - math.pi; tw.append(a)
            m = math.atan2(sum(math.sin(a) for a in tw), sum(math.cos(a) for a in tw))
            korr = Quaternion((0, 1, 0), -m)
            rot[n] = [q @ korr for q in rot[n]]
            log(f"Mocap {art}: {n} Achsdrehung {math.degrees(m):+.0f}° ausgeglichen")
    dz = off[-1].z - off[0].z
    for i, o in enumerate(off): o.z -= dz * i / frames
    off[-1] = off[0].copy()
    # Quelle aufräumen
    bpy.context.window.scene = szene
    for o in list(tmp.objects): bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.scenes.remove(tmp)
    if act.users == 0: bpy.data.actions.remove(act)
    return dict(rot={n: [q.to_matrix() for q in qs] for n, qs in rot.items()}, root=off, dauer=dauer, frames=frames, art=art, speed=speed, kontakt=kontakt)


def pose_setzen(rig, MW, MWi, tab, i, pelvis0):
    """Bild i der Tabelle auf das Rig legen (Welt-Drehungen in Hierarchie-Reihenfolge, Becken versetzt)."""
    for n in REIHENFOLGE:
        pb = rig.pose.bones[n]
        h = (MW @ pb.matrix).to_translation() if n != "spine" else pelvis0 + tab["root"][i]
        Mw = Matrix.Translation(h) @ tab["rot"][n][i].to_4x4()
        pb.matrix = MWi @ Mw
        bpy.context.view_layer.update()
    if "knie_korr" in tab:
        for side in "LR": _knie_strecken(rig, MW, MWi, side, tab["knie_korr"][side][i])


def zyklus_mittel(pfade, rig, REST, MW, frames, log=print, art="walk"):
    """Mittlerer Zyklus aus mehreren Aufnahmen (verschiedene Personen): je Knochen und Phase die Drehungen gemittelt
    (Quaternionen vorzeichen-angeglichen, normierte Summe), Becken-Versatz, Dauer und Tempo gemittelt. Glättet Eigenheiten
    und Aufnahmefehler einzelner Aufnahmen (z. B. auf 0° festgeklemmte Knie, ungleiche Seiten)."""
    tabs = []
    for p in pfade:
        try: tabs.append(zyklus(p, rig, REST, MW, frames, log, art))
        except Exception as e: log(f"Mocap {p}: übersprungen ({e})")
    if not tabs: raise RuntimeError("keine brauchbare Mocap-Aufnahme")
    rot = {}
    for n in ZUORDNUNG:
        reihe = []
        for i in range(frames + 1):
            qs = [t["rot"][n][i].to_quaternion() for t in tabs]; ref = qs[0]; acc = Quaternion((0, 0, 0, 0))
            for q in qs:
                if q.dot(ref) < 0: q = -q
                acc = Quaternion((acc.w + q.w, acc.x + q.x, acc.y + q.y, acc.z + q.z))
            acc.normalize(); reihe.append(acc.to_matrix())
        rot[n] = reihe
    root = [sum((t["root"][i] for t in tabs), Vector()) / len(tabs) for i in range(frames + 1)]
    dauer = sum(t["dauer"] for t in tabs) / len(tabs); speed = sum(t["speed"] for t in tabs) / len(tabs)
    log(f"Mocap-Mittel {art}: {len(tabs)} Aufnahmen, Zyklus {dauer:.2f} s, Tempo {speed:.2f} m/s")
    return dict(rot=rot, root=root, dauer=dauer, frames=frames, art=art, speed=speed, anzahl=len(tabs), kontakt=tabs[0]["kontakt"])


def _knie_strecken(rig, MW, MWi, side, grad):
    """Unterschenkel um das Knie Richtung gestreckt drehen (Ebene Oberschenkel/Unterschenkel), Fuß/Zehe behalten ihre Welt-Drehung."""
    if grad <= 0.05: return
    th, sh, ft, to = (rig.pose.bones[f"{b}.{side}"] for b in ("thigh", "shin", "foot", "toe"))
    Mt, Ms = MW @ th.matrix, MW @ sh.matrix
    dt = Mt.to_3x3().col[1].normalized(); ds = Ms.to_3x3().col[1].normalized()
    winkel = math.degrees(dt.angle(ds))
    g = min(grad, winkel - 3.0)
    if g <= 0.05: return
    ax = ds.cross(dt)
    if ax.length < 1e-6: return
    Rf, Rt = (MW @ ft.matrix).to_3x3().normalized(), (MW @ to.matrix).to_3x3().normalized()
    k = Ms.to_translation()
    Rw = Matrix.Rotation(math.radians(g), 3, ax.normalized())
    sh.matrix = MWi @ (Matrix.Translation(k) @ (Rw @ Ms.to_3x3().normalized()).to_4x4()); bpy.context.view_layer.update()
    for pb, R in ((ft, Rf), (to, Rt)):
        h = (MW @ pb.matrix).to_translation()
        pb.matrix = MWi @ (Matrix.Translation(h) @ R.to_4x4()); bpy.context.view_layer.update()


def knie_korrektur_berechnen(rig, MW, MWi, tab, pelvis0, ferse_z, boden_z, amp=17.0, sigma=3.2, amp_stand=7.0):
    """Streck-Korrektur je Seite: Glocke um den echten Fersenkontakt (Ferse erreicht den Boden) + kleinere Glocke in der
    Standmitte. Aufnahmen landen oft mit ~20-25° gebeugtem Knie und stehen mit ~18°; normal sind ~5° bzw. ~8-10°."""
    N = tab["frames"]; rel = {"L": [], "R": []}
    tab.pop("knie_korr", None)
    for i in range(N):
        pose_setzen(rig, MW, MWi, tab, i, pelvis0); b = boden_z()
        for s in "LR": rel[s].append(ferse_z(s) - b)
    korr = {}
    for s in "LR":
        k_ = tab.get("kontakt", {}).get(s)
        kand = [i for i in range(N) if k_ and k_[i] and not k_[(i - 1) % N]]
        if not kand:
            r = rel[s]; kand = [int(min(range(N), key=lambda i: r[i]))]
        hs = kand[0] + 0.3                                            # Kontaktschwelle 2,5 cm greift ~1-2 Bilder vor dem Aufsetzen
        reihe = []
        for i in range(N + 1):
            d = (i - hs + N / 2) % N - N / 2; d2 = (i - hs - 0.27 * N + N / 2) % N - N / 2
            reihe.append(amp * math.exp(-0.5 * (d / sigma) ** 2) + amp_stand * math.exp(-0.5 * (d2 / 2.6) ** 2))
        korr[s] = reihe
    tab["knie_korr"] = korr; tab["fersenkontakt"] = {s: None for s in "LR"}
    return korr

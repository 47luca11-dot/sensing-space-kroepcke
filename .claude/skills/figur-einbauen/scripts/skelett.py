# Menschliches Skelett (23 Spiel-Knochen + Rigify-Hände mit Fingern) an ein Körper-Mesh anpassen,
# Finger entspannen und Haut/Splats mitbewegen. Nur numpy + mathutils, läuft in Blender.
#
#   J, ROT, info = anpassen(P, H, mensch, carlos)   P = Mesh-Punkte (Z oben, Blick -Y, Füße z=0, Mitte x=y=0)
#   ROT[name] = 3x3-Drehung für die Knochenachsen (Hand/Finger), sonst kürzester Bogen wie bisher
import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion

FINGER = ["thumb", "f_index", "f_middle", "f_ring", "f_pinky"]
SPINE = ["spine", "spine.001", "spine.002", "spine.003", "spine.004", "spine.005", "spine.006"]


def _v(a): return Vector([float(t) for t in a])


def _bogen(a, b):
    """Drehmatrix (3x3 numpy), die Richtung a auf Richtung b dreht (kürzester Bogen)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    q = Vector(a).rotation_difference(Vector(b))
    return np.array(q.to_matrix())


def _rahmen(x, y):
    x = x / np.linalg.norm(x); y = y - (y @ x) * x; y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], 1)                     # Spalten = Achsen


def fingerspitzen(P, w, d, k, log=print):
    """Bis zu 5 Fingerspitzen einer Hand: Fächer um das Handgelenk w, Richtung d. Gibt (Spitzen, Quer-Achse) zurück."""
    Q = P - w; entlang = Q @ d; r = np.linalg.norm(Q, axis=1)
    m = (entlang > 0.02 * k) & (r < 0.24 * k)
    if m.sum() < 50: return [], None
    Qh = Q[m]; quer = Qh - np.outer(Qh @ d, d)
    _, _, vt = np.linalg.svd(quer - quer.mean(0), full_matrices=False)
    e1 = vt[0] - (vt[0] @ d) * d; e1 /= np.linalg.norm(e1)
    ang = np.degrees(np.arctan2(Qh @ e1, Qh @ d)); rho = np.linalg.norm(Qh, axis=1)
    kanten = np.arange(-84, 86, 2.0); fach = np.digitize(ang, kanten) - 1
    rmax = np.zeros(len(kanten)); arg = -np.ones(len(kanten), int)
    for i in range(len(kanten)):
        s = np.where(fach == i)[0]
        if len(s): j = s[np.argmax(rho[s])]; rmax[i] = rho[j]; arg[i] = j
    glatt = np.convolve(rmax, [0.25, 0.5, 0.25], mode="same")
    spitzen = []
    for i in np.argsort(-glatt):
        if glatt[i] < 0.55 * glatt.max() or arg[i] < 0: break
        lo, hi = max(0, i - 3), min(len(glatt), i + 4)
        if glatt[i] < glatt[lo:hi].max() - 1e-9: continue           # kein lokales Maximum
        if any(abs(kanten[i] - kanten[j]) < 9 for j, _ in spitzen): continue
        spitzen.append((i, w + Qh[arg[i]]))
        if len(spitzen) == 5: break
    spitzen.sort(key=lambda t: kanten[t[0]])
    log(f"  Fingerspitzen: {len(spitzen)} bei {[int(kanten[i]) for i, _ in spitzen]}°")
    return [p for _, p in spitzen], [kanten[i] for i, _ in spitzen]


def _ketten_punkte(poly, anteile):
    """Punkte entlang einer Polylinie bei kumulierten Längenanteilen (0..1)."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1); L = seg.sum(); cum = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for t in anteile:
        d = t * L; i = int(np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1)); u = (d - cum[i]) / max(seg[i], 1e-9)
        out.append(poly[i] + u * (poly[i + 1] - poly[i]))
    return out


def hand_analyse(P, kanten, w, d, s, k, log=print):
    """Fingerspitzen = Maxima der Oberflächen-Entfernung vom Handgelenk. Je Finger: Mittellinie, Knöchel (Ast-Ansatz), 4 Gelenkpunkte."""
    import heapq
    from mathutils.kdtree import KDTree
    Q = P - w; r = np.linalg.norm(Q, axis=1)
    reg = np.where(((Q @ d) > -0.01 * k) & (r < 0.26 * k))[0]
    if len(reg) < 200: return None
    loc = -np.ones(len(P), np.int64); loc[reg] = np.arange(len(reg))
    E = kanten[(loc[kanten[:, 0]] >= 0) & (loc[kanten[:, 1]] >= 0)]
    nb = [[] for _ in reg]
    for a_, b_ in E:
        la, lb = loc[a_], loc[b_]; l = float(np.linalg.norm(P[a_] - P[b_])); nb[la].append((lb, l)); nb[lb].append((la, l))
    R = P[reg]; Qr = R - w
    saat = np.where((np.abs(Qr @ d) < 0.006 * k) & (np.linalg.norm(Qr, axis=1) < 0.035 * k))[0]
    if not len(saat): return None
    dist = np.full(len(reg), np.inf); vor = -np.ones(len(reg), np.int64); hp = []
    for i in saat: dist[i] = 0.0; heapq.heappush(hp, (0.0, int(i)))
    while hp:
        dd, i = heapq.heappop(hp)
        if dd > dist[i]: continue
        for j, l in nb[i]:
            if dd + l < dist[j]: dist[j] = dd + l; vor[j] = i; heapq.heappush(hp, (dd + l, j))
    ok = np.isfinite(dist)
    kd = KDTree(len(R))
    for i, p in enumerate(R): kd.insert(p, i)
    kd.balance()
    # Verschmelzungs-Baum: Punkte von außen nach innen; jeder Finger ist ein Ast, der am Knöchel mit dem Rest verschmilzt
    par = np.arange(len(R)); spitze_von = {}; basis = {}; fertig = np.zeros(len(R), bool)
    def wurzel(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for v in np.argsort(-np.where(ok, dist, -1)):
        if not ok[v]: break
        fertig[v] = True; rs = {wurzel(u) for u, _ in nb[v] if fertig[u]}
        if not rs: spitze_von[v] = v; continue
        rs = sorted(rs, key=lambda r_: dist[spitze_von[r_]], reverse=True); haupt = rs[0]
        for r_ in rs[1:]:
            if dist[spitze_von[r_]] - dist[v] >= 0.006 * k:                 # Ast (kein Buckel); Top-5 entscheidet unten
                for t in (spitze_von[r_], spitze_von[haupt]): basis.setdefault(t, float(dist[v]))
            par[r_] = haupt
        par[v] = haupt
    kand = sorted(basis, key=lambda t: -(dist[t] - basis[t]))
    spitzen = [t for t in kand if dist[t] - basis[t] > 0.009 * k and dist[t] > 0.035 * k
               and (R[t] - w) @ d > 0.02 * k][:5]                          # nur Äste nach vorn (nicht in Ärmel/Jacke); kurze Äste = gekrümmte Finger
    if len(spitzen) < 4:
        log(f"  Hand: nur {len(spitzen)} Finger-Äste"); return None
    alle_i = np.arange(len(R)); daumen_t = min(spitzen, key=lambda t: basis[t])     # Daumen zweigt am nächsten am Handgelenk ab
    e_t = (R[daumen_t] - w) - ((R[daumen_t] - w) @ d) * d
    if np.linalg.norm(e_t) < 1e-6: return None
    e_t /= np.linalg.norm(e_t)
    andere = sorted([t for t in spitzen if t != daumen_t], key=lambda t: -((R[t] - w) @ e_t))   # vom Daumen weg
    L_hand = float(np.linalg.norm(R[andere[1]] - w))                   # Handgelenk -> Mittelfingerspitze
    anteil = {"f_index": 0.52, "f_middle": 0.53, "f_ring": 0.50, "f_pinky": 0.45}   # Knöchel-Abstand / Handlänge
    def kette(t, ziel_r):
        g0 = basis[t]; st = [t]; menge = {t}
        while st:
            i = st.pop()
            for j, _ in nb[i]:
                if j not in menge and dist[j] > g0 - 0.002 * k: menge.add(j); st.append(j)
        mi = np.array(sorted(menge)); pfad = [t]
        while vor[pfad[-1]] >= 0 and np.linalg.norm(R[pfad[-1]] - w) > ziel_r: pfad.append(int(vor[pfad[-1]]))
        cs = []
        for v in pfad[::-1]:
            if dist[v] > g0: ring = mi[np.abs(dist[mi] - dist[v]) < 0.003 * k]; rad = 0.025 * k
            else: ring = alle_i[np.abs(dist - dist[v]) < 0.003 * k]; rad = 0.015 * k
            ring = ring[np.linalg.norm(R[ring] - R[v], axis=1) < rad]
            cs.append(R[ring].mean(0) if len(ring) else R[v])
        return dict(spitze=t, mitte=np.array(cs), basis_g=g0, menge=mi)
    daumen = kette(daumen_t, 0.035 * k)                                # Daumen-Sattelgelenk ~3,5 cm vom Handgelenk
    rest = [kette(t, anteil[n] * L_hand) for n, t in zip(("f_index", "f_middle", "f_ring", "f_pinky"), andere)]
    if len(rest) == 3:                                                 # kleiner Finger verdeckt/angewachsen: neben den Ringfinger setzen
        mi_, ri_ = rest[1]["mitte"], rest[2]["mitte"]; versatz = ri_[0] - mi_[0]; b0 = ri_[0] + versatz
        neu = b0 + (ri_ - ri_[0]) * 0.82
        rest.append(dict(spitze=None, mitte=neu, basis_g=rest[2]["basis_g"], menge=np.array([], np.int64)))
        log("  Hand: kleiner Finger ergänzt (nicht frei im Mesh)")
    namen = ["f_index", "f_middle", "f_ring", "f_pinky"]; alle = dict(zip(namen, rest)); alle["thumb"] = daumen
    d_neu = alle["f_middle"]["mitte"][0] - w; d_neu /= np.linalg.norm(d_neu)
    e_t = e_t - (e_t @ d_neu) * d_neu; e_t /= np.linalg.norm(e_t)
    n_palm = s * np.cross(d_neu, e_t)
    gel, basis = {}, {}
    for f, fd in alle.items():
        poly = fd["mitte"].copy()
        spitze = R[fd["spitze"]] if fd["spitze"] is not None else poly[-1]; rueck = (poly[-2] - spitze) if fd["spitze"] is None else (poly[-1] - spitze); rl = np.linalg.norm(rueck)
        poly[-1] = spitze + (rueck / rl if rl > 1e-6 else -d_neu) * 0.005 * k            # Knochenende 5 mm unter der Fingerkuppe
        anteile = (0, 0.40, 0.73, 1) if f == "thumb" else (0, 0.49, 0.78, 1)
        gel[f] = _ketten_punkte(poly, anteile); basis[f] = poly[0]
    labels = {}
    for f, fd in alle.items():
        for i in fd["menge"]: labels.setdefault(int(reg[i]), []).append(f)
    log(f"  Hand: 5 Finger, Daumen zweigt bei {daumen['basis_g']:.3f} m ab, Knöchel bei {[round(alle[f]['basis_g'], 3) for f in namen]}")
    return dict(richtung=d_neu, gelenke=gel, basis=basis, handflaeche=n_palm, daumenseite=e_t, erkannt=True,
                labels={int(v): (fs[0] if len(fs) == 1 else fs) for v, fs in labels.items()},
                region=reg.tolist(), wdist={int(reg[i]): float(dist[i]) for i in range(len(reg)) if ok[i]})


def hand_vorlage(mensch, side, w, d, s, k):
    """Ersatz ohne Erkennung: gerade Finger nach Proportionen, Daumen seitlich."""
    quer = np.cross(d, [0, 1.0, 0]); quer /= np.linalg.norm(quer); e_t = s * quer
    n_palm = s * np.cross(d, e_t); gel, basis = {}, {}
    for i, f in enumerate(["thumb", "f_index", "f_middle", "f_ring", "f_pinky"]):
        if f == "thumb": b = w + d * 0.03 * k + e_t * 0.03 * k; dirv = (d + e_t) / np.linalg.norm(d + e_t); L = 0.11 * k
        else:
            off = (1.5 - (i - 1)) * 0.02 * k; b = w + d * 0.095 * k + e_t * off; dirv = d; L = (0.085, 0.095, 0.09, 0.07)[i - 1] * k
        gel[f] = [b + dirv * L * t for t in ((0, 0.40, 0.73, 1) if f == "thumb" else (0, 0.49, 0.78, 1))]; basis[f] = b
    return dict(richtung=d, gelenke=gel, basis=basis, handflaeche=n_palm, daumenseite=e_t, erkannt=False)


def anpassen(P, H, mensch, carlos, log=print, kanten=None):
    k = H / 1.8
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    J, ROT, info = {}, {}, {}

    def torso_y(zz, breite=0.13, nach_hinten=0.3):
        m = (np.abs(z - zz) < 0.012) & (np.abs(x) < breite * k)
        if m.sum() < 5: return 0.0
        c = float(np.median(y[m])); return c + nach_hinten * (float(y[m].max()) - c)   # Wirbelsäule eher hinten

    # ---------- Becken: Schritt = unterste Höhe, auf der zwischen den Beinen Fläche liegt
    zc = None
    for zz in np.arange(0.30 * H, 0.62 * H, 0.004):
        if ((np.abs(z - zz) < 0.003) & (np.abs(x) < 0.012)).sum() >= 3: zc = float(zz); break
    if zc is None or not (0.43 * H <= zc <= 0.50 * H): zc = 0.465 * H     # Beine berühren sich (Innenseite) -> Körperproportion
    z_hip = zc + 0.065 * k; info["schritt_z"] = round(zc, 3)

    # ---------- Schultern: Oberkante bei |x| ~ 0.17 m, Gelenk 5,5 cm darunter, 5 cm innerhalb der Außenseite
    sh = {}; info["arm_achse"] = {}
    for side, s in (("L", 1), ("R", -1)):
        # Schultergelenk aus dem Arm: Gerade durch den frei stehenden Oberarm (26-42 cm neben der Mitte), bis auf
        # Schulterbreite (17,5 cm) nach innen verlängert. Robust gegen Haare/Kapuzen über der Schulter und flache Arme.
        Aw = P[(s * x > 0.26 * k) & (s * x < 0.42 * k) & (z > 0.55 * H) & (z < 0.9 * H)]
        z_sh = None
        if len(Aw) > 50:
            c_ = Aw.mean(0); _, _, vt = np.linalg.svd(Aw - c_, full_matrices=False); d_ = vt[0]
            if d_[0] * s < 0: d_ = -d_
            if abs(d_[0]) > 0.2:
                t_ = (s * 0.175 * k - c_[0]) / d_[0]; pj = c_ + d_ * t_
                if 0.74 * H < pj[2] < 0.88 * H: z_sh = float(pj[2])
                info["arm_achse"][side] = dict(neigung_grad=round(float(np.degrees(np.arcsin(-d_[2]))), 1))
        if z_sh is None:                                               # Arme hängen eng am Körper: alte Methode (Oberkante)
            band = (s * x > 0.13 * k) & (s * x < 0.19 * k) & (z > 0.7 * H) & (z < 0.88 * H) & (np.abs(y) < 0.15 * k)
            z_sh = float(z[band].max()) - 0.055 * k if band.any() else 0.82 * H
        sl = (np.abs(z - z_sh) < 0.02 * k) & (s * x > 0.12 * k) & (s * x < 0.22 * k)
        yy = y[sl] if sl.sum() > 5 else y[np.abs(z - z_sh) < 0.03 * k]
        sh[side] = Vector((s * 0.175 * k, float(np.median(yy)), z_sh))
    z_sh = (sh["L"].z + sh["R"].z) / 2
    z_n0 = z_sh + 0.02 * k                                           # Halsansatz (C7)
    info["schulter_z"] = round(z_sh, 3)

    # ---------- Wirbelsäule + Kopf: Höhen-Anteile wie Carlos zwischen Hüfte, Halsansatz und Scheitel
    C = carlos
    c_hip, c_n0, c_top = C["spine"]["head"][2], C["spine.004"]["head"][2], 1.80
    def hz(t):
        if t <= c_n0: return z_hip + (t - c_hip) / (c_n0 - c_hip) * (z_n0 - z_hip)
        return z_n0 + (t - c_n0) / (c_top - c_n0) * (H - z_n0)
    for n in SPINE:
        for e in ("head", "tail"):
            zz = hz(C[n][e][2])
            J[(n, e)] = Vector((0.0, torso_y(min(zz, H - 0.05 * k), 0.10 if zz > z_n0 else 0.13, 0.3 if zz < z_n0 else 0.0), zz))
    J[("spine.006", "tail")].z = H * 0.985

    # ---------- Beine
    for side, s in (("L", 1), ("R", -1)):
        def bein(zz, tol=0.008):
            return P[(np.abs(z - zz) < tol) & (s * x > 0.0) & (s * x < 0.3 * k)]
        L1 = bein(zc - 0.08 * k, 0.012)
        hx = float(np.clip(abs(L1[:, 0].mean()), 0.07 * k, 0.12 * k)) if len(L1) else 0.1 * k
        hip = Vector((s * hx, torso_y(z_hip, 0.13, 0.0), z_hip))          # Hüftgelenk mittig im Becken (nicht nach hinten)
        z_ank = 0.05 * H; A = bein(0.075 * H)
        ankle = Vector((float(A[:, 0].mean()), float(A[:, 1].mean()), z_ank))
        z_knee = z_ank + 0.5 * (z_hip - z_ank); K = bein(z_knee)
        knee = Vector((float(K[:, 0].mean()), float(K[:, 1].mean()) - 0.01 * k, z_knee))   # Knie leicht vorn
        F = P[(z < 0.045 * k) & (s * x > 0) & (s * x < 0.3 * k)]
        toe_y = float(F[:, 1].min()); vorn = F[F[:, 1] < toe_y + 0.06 * k]
        fx = float(vorn[:, 0].mean())
        ball = Vector((fx, toe_y + 0.075 * k, 0.025 * k)); tip = Vector((fx, toe_y + 0.012 * k, 0.02 * k))
        J.update({("thigh." + side, "head"): hip, ("thigh." + side, "tail"): knee, ("shin." + side, "head"): knee,
                  ("shin." + side, "tail"): ankle, ("foot." + side, "head"): ankle, ("foot." + side, "tail"): ball,
                  ("toe." + side, "head"): ball, ("toe." + side, "tail"): tip})

    # ---------- Arme + Hände
    for side, s in (("L", 1), ("R", -1)):
        sj = sh[side]
        path = [sj]
        # Arm entlang seiner Richtung verfolgen (unabhängig von der Haltung: A-Pose, flache Arme, fast T-Pose):
        # von der Schulter aus jeweils den Schwerpunkt der Oberfläche in einer Schale 3-7 cm voraus (Kegel um die Laufrichtung)
        A_arm = P[s * x > sj.x * s + 0.01 * k]
        dirv = np.array([s * 1.0, 0.0, -0.75]); dirv /= np.linalg.norm(dirv)
        Aw = P[(s * x > 0.26 * k) & (s * x < 0.42 * k) & (z > 0.55 * H) & (z < 0.9 * H)]
        if len(Aw) > 50:                                                   # Startrichtung = Achse des Oberarms
            c_ = Aw.mean(0); _, _, vt = np.linalg.svd(Aw - c_, full_matrices=False); d_ = vt[0]
            if d_[0] * s < 0: d_ = -d_
            if abs(d_[0]) > 0.2: dirv = d_ / np.linalg.norm(d_)
        cur = np.array(sj)
        for _ in range(60):
            Q = A_arm - cur; dist = np.linalg.norm(Q, axis=1)
            m = None
            for r_in, r_out in ((0.03, 0.07), (0.05, 0.10), (0.07, 0.14)):          # Scan-Lücken (fleckige Ärmel) überspringen
                mm = (dist > r_in * k) & (dist < r_out * k) & ((Q @ dirv) > 0.6 * dist)
                if mm.sum() >= 8: m = mm; break
            if m is None: break
            c = A_arm[m].mean(0)
            nd = c - cur; nl = np.linalg.norm(nd)
            if nl < 1e-4: break
            dirv = 0.5 * dirv + 0.5 * nd / nl; dirv /= np.linalg.norm(dirv)
            cur = cur + dirv * 0.04 * k
            # auf die Querschnitt-Mitte ziehen (Schale um den neuen Punkt, senkrecht zur Laufrichtung)
            Q2 = A_arm - cur; ring = (np.abs(Q2 @ dirv) < 0.008 * k) & (np.linalg.norm(Q2, axis=1) < 0.09 * k)
            if ring.sum() > 8: cur = A_arm[ring].mean(0)
            path.append(Vector(cur))
        seg = [(path[i + 1] - path[i]).length for i in range(len(path) - 1)]; L = sum(seg)
        def at(dist):
            acc = 0.0
            for i, l in enumerate(seg):
                if acc + l >= dist: return path[i].lerp(path[i + 1], (dist - acc) / max(l, 1e-6))
                acc += l
            return path[-1].copy()
        L = min(L, 0.78 * k)
        # Handgelenk = schmalste Stelle zwischen 0.13 und 0.25 m vor dem Armende (hinter dem Ärmelbund)
        best = None
        for dd in np.arange(L - 0.25 * k, L - 0.12 * k, 0.005):
            c = at(dd); dirv = (at(dd + 0.02) - at(dd - 0.02)).normalized()
            Q = P - np.array(c); ent = Q @ np.array(dirv)
            ring = Q[(np.abs(ent) < 0.004)]; ring = ring[np.linalg.norm(ring, axis=1) < 0.08 * k]
            if len(ring) < 8: continue
            w_ = float(np.linalg.norm(ring - ring.mean(0), axis=1).mean())
            if best is None or w_ < best[0]: best = (w_, dd)
        d_w = best[1] if best else L - 0.19 * k
        def mitte(dd, r=0.07):                                         # Querschnitt-Mitte senkrecht zum Arm
            c = at(dd); dv = (at(dd + 0.02) - at(dd - 0.02)).normalized(); Q = P - np.array(c)
            ring = Q[(np.abs(Q @ np.array(dv)) < 0.005)]; ring = ring[np.linalg.norm(ring, axis=1) < r * k]
            return Vector(np.array(c) + ring.mean(0)) if len(ring) > 8 else c
        wrist = mitte(d_w, 0.05); elbow = mitte(d_w * 0.54, 0.08)
        d_hand = np.array((at(L) - wrist).normalized())
        info["arm_" + side] = dict(laenge=round(L, 3), handgelenk_bei=round(d_w, 3))
        J.update({("shoulder." + side, "head"): Vector((s * 0.025 * k, J[("spine.003", "tail")].y - 0.02 * k, z_sh)),
                  ("shoulder." + side, "tail"): sj, ("upper_arm." + side, "head"): sj, ("upper_arm." + side, "tail"): elbow,
                  ("forearm." + side, "head"): elbow, ("forearm." + side, "tail"): wrist})
        # --- Hand + Finger aus der Mesh-Oberfläche (Äste vom Handgelenk aus, funktioniert auch bei gekrümmten Fingern)
        w = np.array(wrist)
        hd = hand_analyse(P, kanten, w, d_hand, s, k, log) if kanten is not None else None
        if hd is None:
            log(f"  Hand {side}: Finger nicht erkannt -> Vorlage")
            hd = hand_vorlage(mensch, side, w, d_hand, s, k)
        d_hand = hd["richtung"]
        J[("hand." + side, "head")] = Vector(w); J[("hand." + side, "tail")] = Vector(w + (hd["basis"]["f_middle"] - w) * 0.8)
        for f in FINGER:
            pts = hd["gelenke"][f]
            for i in (1, 2, 3):
                n = f"{f}.0{i}.{side}"; J[(n, "head")] = Vector(pts[i - 1]); J[(n, "tail")] = Vector(pts[i]); ROT[n] = ("z", -hd["handflaeche"])
        for i, f in enumerate(("f_index", "f_middle", "f_ring", "f_pinky"), 1):
            n = f"palm.0{i}.{side}"; bas = hd["basis"][f]
            J[(n, "head")] = Vector(w + (bas - w) * 0.25); J[(n, "tail")] = Vector(bas); ROT[n] = ("z", -hd["handflaeche"])
        info["handflaeche_" + side] = hd["handflaeche"].tolist(); info["daumenseite_" + side] = hd["daumenseite"].tolist()
        info["hand_" + side] = dict(handgelenk=w.tolist(), richtung=d_hand.tolist(), finger=hd["erkannt"])
        info["handlabels_" + side] = hd.get("labels")
    return J, ROT, info


def rig_bauen(scene, name, J, ROT, mensch, carlos):
    import bpy
    arm = bpy.data.armatures.new(name); rig = bpy.data.objects.new(name, arm); scene.collection.objects.link(rig)
    rig.show_in_front = True
    bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True); bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for n, tb in mensch.items():
        b = arm.edit_bones.new(n); b.head = J[(n, "head")]; b.tail = J[(n, "tail")]
        if (b.tail - b.head).length < 0.004: b.tail = b.head + Vector((0, 0, 0.01))
        quelle = carlos.get(n, tb)
        tdir = np.array(quelle["tail"]) - np.array(quelle["head"]); ndir = np.array(b.tail - b.head)
        if n in ROT and isinstance(ROT[n], tuple):                  # Finger: Z-Achse zeigt zum Handrücken
            zv = np.array(ROT[n][1]); bd = np.array(b.tail - b.head); bd /= np.linalg.norm(bd); zn = zv - (zv @ bd) * bd
        elif n in ROT: zn = ROT[n] @ np.array(tb["z"])
        else: zn = _bogen(tdir, ndir) @ np.array(quelle["z"])
        b.align_roll(Vector(zn))
    for n, tb in mensch.items():
        if tb["parent"]: arm.edit_bones[n].parent = arm.edit_bones[tb["parent"]]
    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in rig.pose.bones: pb.rotation_mode = 'QUATERNION'
    return rig


def finger_entspannen(rig, info):
    """Finger in eine natürliche Ruhehaltung drehen: nebeneinander (kaum gespreizt), leicht zur Handfläche gekrümmt.
    Ziel-Richtungen je Glied aus Handrichtung d, Daumenseite e und Handflächen-Normale n – unabhängig davon, wie der Scan sie hält."""
    import bpy
    ziele = {"f_index": (6, (16, 26, 14)), "f_middle": (0, (18, 30, 16)), "f_ring": (-5, (20, 32, 17)), "f_pinky": (-11, (22, 34, 18))}
    def drehe(pb, R):
        h = np.array(pb.head)
        M = Matrix.Translation(Vector(h)) @ Matrix(R.tolist()).to_4x4() @ Matrix.Translation(Vector(-h))
        pb.matrix = M @ pb.matrix; bpy.context.view_layer.update()
    for side in ("L", "R"):
        if not info["hand_" + side]["finger"]: continue
        d = np.array(info["hand_" + side]["richtung"]); e = np.array(info["daumenseite_" + side]); n = np.array(info["handflaeche_" + side])
        for f, (phi, kr) in ziele.items():
            p = math.radians(phi); df = math.cos(p) * d + math.sin(p) * e; cum = 0.0
            for i, c in enumerate(kr, 1):
                cum += math.radians(c); ziel = math.cos(cum) * df + math.sin(cum) * n
                pb = rig.pose.bones[f"{f}.0{i}.{side}"]; bpy.context.view_layer.update()
                cur = np.array(pb.tail - pb.head); drehe(pb, _bogen(cur, ziel))
        # Daumen: etwas an die Hand heran, Glieder leicht gebeugt
        z1 = rig.pose.bones["thumb.01." + side]; ib = np.array(rig.pose.bones["f_index.01." + side].head)
        cur = np.array(z1.tail - z1.head); zu = ib - np.array(z1.head); zu /= np.linalg.norm(zu)
        tgt = cur / np.linalg.norm(cur) * 0.75 + zu * 0.25; drehe(z1, _bogen(cur, tgt))
        for i in (2, 3):
            pb = rig.pose.bones[f"thumb.0{i}.{side}"]; cur = np.array(pb.tail - pb.head); cur /= np.linalg.norm(cur)
            ach = np.cross(cur, n); ach /= max(np.linalg.norm(ach), 1e-9)
            drehe(pb, np.array(Matrix.Rotation(math.radians(12), 3, Vector(ach))))


def hand_gewichte(game, rig, info, k, log=print):
    """Hand-Gewichte geometrisch statt Heat: Fingerpunkte -> Glieder ihres Fingers, Handfläche -> hand, Handgelenk bleibt."""
    from mathutils.kdtree import KDTree
    GV = np.array([v.co[:] for v in game.data.vertices]); geaendert = 0
    def segd(p, a, b):
        ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1); return np.linalg.norm(a + np.outer(t, ab) - p, axis=1) if p.ndim == 2 else np.linalg.norm(a + t * ab - p)
    B = {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in rig.data.bones}
    FB = [f"{f}.0{i}" for f in ("thumb", "f_index", "f_middle", "f_ring", "f_pinky") for i in (1, 2, 3)]
    for side in ("L", "R"):
        lab = info.get("handlabels_" + side); hi_pts = info.get("hi_punkte")
        if (not lab or hi_pts is None) and ("hand_" + side) in info:
            # Ersatz ohne Finger-Erkennung (Finger im Scan verschmolzen): jeder Handpunkt -> nächste Glieder der Vorlage,
            # weich über die drei nächsten Knochen, damit das Entspannen die Hand zusammenführt statt sie zu zerreißen
            w = np.array(info["hand_" + side]["handgelenk"]); d = np.array(info["hand_" + side]["richtung"])
            Q = GV - w; sel = np.where(((Q @ d) > 0.015 * k) & (np.linalg.norm(Q, axis=1) < 0.24 * k))[0]
            kand_all = ["hand." + side] + [f"{n}.{side}" for n in FB]
            for vi in sel:
                p = GV[vi]; dd = np.array([segd(p, *B[n]) for n in kand_all]); top = np.argsort(dd)[:3]
                ww = 1.0 / (dd[top] + 0.004) ** 3; ww /= ww.sum()
                for g in list(game.data.vertices[vi].groups): g.weight = 0.0
                for t_i, x_ in zip(top, ww):
                    if x_ > 0.02: game.vertex_groups[kand_all[t_i]].add([int(vi)], float(x_), 'REPLACE')
                geaendert += 1
            continue
        if not lab or hi_pts is None: continue
        w = np.array(info["hand_" + side]["handgelenk"]); d = np.array(info["hand_" + side]["richtung"])
        idx = list(lab.keys()); kd = KDTree(len(idx))
        for i, v in enumerate(idx): kd.insert(hi_pts[v], i)
        kd.balance()
        Q = GV - w; sel = np.where(((Q @ d) > 0.012 * k) & (np.linalg.norm(Q, axis=1) < 0.26 * k))[0]
        for vi in sel:
            p = GV[vi]; co, j, dist_ = kd.find(p)
            if dist_ > 0.012 * k: kand = ["hand." + side, "thumb.01." + side] + [f"{f}.01.{side}" for f in ("f_index", "f_middle", "f_ring", "f_pinky")]   # Handfläche
            else:
                fs = lab[idx[j]]; fs = [fs] if isinstance(fs, str) else fs
                kand = ["hand." + side] + [f"{f}.0{i}.{side}" for f in fs for i in (1, 2, 3)]
            dd = np.array([segd(p, *B[n]) for n in kand]); ww = 1.0 / (dd + 0.002) ** 4; ww /= ww.sum()
            for g in list(game.data.vertices[vi].groups): g.weight = 0.0
            for n, x_ in zip(kand, ww):
                if x_ > 0.01: game.vertex_groups[n].add([int(vi)], float(x_), 'REPLACE')
            geaendert += 1
    log(f"  Hand-Gewichte geometrisch gesetzt: {geaendert} Punkte")
    return geaendert


def knochen_matrizen(rig, namen):
    """Pose-Matrizen relativ zur Ruhepose (Armature-Raum), Reihenfolge wie 'namen'."""
    return [np.array(rig.pose.bones[n].matrix @ rig.data.bones[n].matrix_local.inverted()) for n in namen]


def lbs(Pts, W, mats):
    """Linear-Blend-Skinning: Punkte (N,3), Gewichte (N,B), 4x4-Matrizen (B)."""
    out = np.zeros_like(Pts); Ph = np.concatenate([Pts, np.ones((len(Pts), 1))], 1)
    for b, M in enumerate(mats):
        w = W[:, b]; m = w > 0
        if m.any(): out[m] += w[m, None] * (Ph[m] @ M.T)[:, :3]
    s = W.sum(1); out[s < 1e-6] = Pts[s < 1e-6]
    return out

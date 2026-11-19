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
    """Finger aus der Mesh-Oberfläche: geodätische Entfernung vom Handgelenk, Merge-Tree (Persistenz) -> die 5 längsten Äste
    sind die Finger, ihr Sattel ist die Schwimmhaut. Je Finger Mittellinie aus Querschnitt-Ringen (gleiche Oberflächen-
    Entfernung), Grundgelenk 2 cm vor der Schwimmhaut, Glieder nach Fingerproportionen (Grund:Mittel:End = 45:28:27)."""
    import heapq
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
    dist = np.full(len(reg), np.inf); hp = []
    for i in saat: dist[i] = 0.0; heapq.heappush(hp, (0.0, int(i)))
    while hp:
        dd, i = heapq.heappop(hp)
        if dd > dist[i]: continue
        for j, l in nb[i]:
            if dd + l < dist[j]: dist[j] = dd + l; heapq.heappush(hp, (dd + l, j))
    ok = np.isfinite(dist)
    # Merge-Tree: Spitzen sterben, wenn ihr Ast in einen mit höherer Spitze mündet (Persistenz = Astlänge über dem Sattel)
    par = np.arange(len(R)); tip = {}; fertig = np.zeros(len(R), bool); aeste = []
    def wz(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for v in np.argsort(-np.where(ok, dist, -1)):
        if not ok[v]: break
        fertig[v] = True; rs = {wz(u) for u, _ in nb[v] if fertig[u]}
        if not rs: tip[v] = v; continue
        rs = sorted(rs, key=lambda r_: dist[tip[r_]], reverse=True); h = rs[0]
        for r_ in rs[1:]: aeste.append([int(tip[r_]), float(dist[tip[r_]] - dist[v]), float(dist[v])]); par[r_] = h
        par[v] = h
    for r_ in {wz(i) for i in range(len(R)) if ok[i]}: aeste.append([int(tip[r_]), float(dist[tip[r_]]), None])
    aeste = [a for a in aeste if a[1] > 0.012 * k and dist[a[0]] > 0.05 * k and (R[a[0]] - w) @ d > 0.02 * k]
    aeste.sort(key=lambda a: -a[1]); aeste = aeste[:5]
    if len(aeste) < 2:
        log(f"  Hand: nur {len(aeste)} Finger-Äste"); return None
    satt = [a[2] for a in aeste if a[2] is not None]
    for a in aeste:                                                   # Hauptast (längster Finger): Sattel = höchste Schwimmhaut
        if a[2] is None: a[2] = max(satt) if satt else 0.6 * dist[a[0]]
    # Daumen: zweigt am frühesten ab und ist kurz
    daumen = min(aeste, key=lambda a: a[2] + 0.5 * dist[a[0]])
    e_t = (R[daumen[0]] - w) - ((R[daumen[0]] - w) @ d) * d; e_t /= max(np.linalg.norm(e_t), 1e-9)

    def menge_von(t, g0):
        menge = {t}; st = [t]
        while st:
            i = st.pop()
            for j, _ in nb[i]:
                if j not in menge and ok[j] and dist[j] > g0 + 0.001 * k: menge.add(j); st.append(j)
        return np.array(sorted(menge))
    def mittellinie(mi, t, g0):
        cs = []
        for lev in np.arange(dist[t] - 0.004 * k, g0 - 0.001, -0.004 * k):
            ring = mi[np.abs(dist[mi] - lev) < 0.002 * k]
            if len(ring) >= 3: cs.append(R[ring].mean(0))
        return np.array(([R[t]] + cs)[::-1])                            # von der Schwimmhaut zur Spitze
    # Finger-Äste (ohne Daumen). Liegen Finger aneinander (verschmolzen), ist ein Ast mehrere Finger breit:
    # Anzahl aus der Breite des Querschnitts, dann in gleich breite Streifen quer zur Fingerrichtung teilen –
    # jeder Streifen folgt der Mesh-Form (auch gekrümmt), Spitze = entferntester Punkt im Streifen.
    rest = []
    for a in aeste:
        if a is daumen: continue
        t, _, g0 = a; mi = menge_von(t, g0)
        ax_b = R[t] - w; ax_b /= np.linalg.norm(ax_b); lat = e_t - (e_t @ ax_b) * ax_b; lat /= max(np.linalg.norm(lat), 1e-9)
        lev = g0 + 0.4 * (dist[t] - g0); ring = mi[np.abs(dist[mi] - lev) < 0.003 * k]
        u_r = (R[ring] - w) @ lat if len(ring) > 5 else np.zeros(1)
        breite = float(np.percentile(u_r, 95) - np.percentile(u_r, 5))
        rest.append(dict(t=t, g0=g0, mi=mi, lat=lat, breite=breite, n=int(np.clip(round(breite / (0.019 * k)), 1, 4))))
    if not rest:
        log("  Hand: nur Daumen erkannt"); return None
    while sum(r_["n"] for r_ in rest) > 4: max(rest, key=lambda r_: r_["n"])["n"] -= 1
    while sum(r_["n"] for r_ in rest) < 4: max(rest, key=lambda r_: r_["breite"] / (r_["n"] + 1))["n"] += 1
    finger = []; verschmolzen = any(r_["n"] > 1 for r_ in rest) or len(rest) < 4   # (Teilmenge, Spitze, Sattel)
    for r_ in rest:
        if r_["n"] == 1: finger.append((r_["mi"], r_["t"], r_["g0"])); continue
        u = (R[r_["mi"]] - w) @ r_["lat"]; lo, hi_ = np.percentile(u, 3), np.percentile(u, 97)
        grenzen = np.linspace(lo, hi_, r_["n"] + 1); grenzen[0], grenzen[-1] = -np.inf, np.inf
        for j in range(r_["n"]):
            sub = r_["mi"][(u >= grenzen[j]) & (u < grenzen[j + 1])]
            if len(sub) < 5: continue
            finger.append((sub, int(sub[np.argmax(dist[sub])]), r_["g0"]))
        log(f"  Hand: verschmolzener Ast ({r_['breite'] * 100:.1f} cm breit) in {r_['n']} Finger geteilt")
    if len(finger) < 4:
        log(f"  Hand: nur {len(finger)} Finger"); return None
    finger.sort(key=lambda f_: -((R[f_[1]] - w) @ e_t))                 # vom Daumen weg: Zeige, Mittel, Ring, Klein
    namen = ["f_index", "f_middle", "f_ring", "f_pinky"]; alle = {}
    for n, (mi, t, g0) in zip(namen, finger[:4]): alle[n] = (mi, mittellinie(mi, t, g0), [t, 0.0, g0])
    mi_d = menge_von(daumen[0], daumen[2]); alle["thumb"] = (mi_d, mittellinie(mi_d, daumen[0], daumen[2]), daumen)
    d_neu = alle["f_middle"][1][-1] - w; d_neu /= np.linalg.norm(d_neu)
    e_t = e_t - (e_t @ d_neu) * d_neu; e_t /= np.linalg.norm(e_t)
    n_palm = s * np.cross(d_neu, e_t)
    gel, basis = {}, {}
    L_hand = float(np.linalg.norm(alle["f_middle"][1][-1] - w))         # Handgelenk -> Mittelfingerspitze
    for f, (mi, cl, a) in alle.items():
        spitze = cl[-1]
        ax = cl[-1] - cl[0] if len(cl) > 2 else spitze - w
        ax = ax / max(np.linalg.norm(ax), 1e-9)
        ende = spitze - ax * 0.005 * k                                # Knochenende 5 mm unter der Fingerkuppe
        if f == "thumb":
            mcp = cl[0] - ax * 0.008 * k                              # Daumen-Grundgelenk knapp vor der Schwimmhaut
            cmc = w + (mcp - w) * 0.30 + e_t * 0.012 * k              # Sattelgelenk nahe der Handwurzel, zur Daumenseite
            ip = mcp + (ende - mcp) * 0.55
            gel[f] = [cmc, mcp, ip, ende]; basis[f] = cmc
        else:
            # Grundgelenk (Knöchel) liegt tief in der Handfläche, nicht an der Schwimmhaut: Fingerlänge Knöchel->Spitze als
            # Anteil der Handlänge (Handgelenk->Mittelfingerspitze; Anthropometrie: Zeige 0,47, Mittel 0,51, Ring 0,48, Klein 0,39),
            # mindestens 2,5 cm hinter der Schwimmhaut; Richtung = Fingerachse der unteren Fingerhälfte
            poly = np.vstack([cl[:-1], ende]) if len(cl) > 1 else np.vstack([cl[0], ende])
            sichtbar = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
            soll = {"f_index": 0.47, "f_middle": 0.51, "f_ring": 0.48, "f_pinky": 0.39}[f] * L_hand
            ub = cl[min(len(cl) - 1, max(2, len(cl) // 2))] - cl[0]
            ax0 = ub / np.linalg.norm(ub) if np.linalg.norm(ub) > 0.004 * k else ax
            mcp = cl[0] - ax0 * max(soll - sichtbar, 0.025 * k)
            poly = np.vstack([mcp, poly])
            gel[f] = _ketten_punkte(poly, (0, 0.45, 0.73, 1)); basis[f] = mcp
    labels = {}
    for f, (mi, cl, a) in alle.items():
        for i in mi: labels.setdefault(int(reg[i]), []).append(f)
    log(f"  Hand: 5 Finger, Schwimmhäute bei {[round(alle[f][2][2], 3) for f in ['thumb'] + namen]} m, "
        f"Längen {[round(float(np.linalg.norm(gel[f][3] - gel[f][0])), 3) for f in ['thumb'] + namen]}")
    return dict(richtung=d_neu, gelenke=gel, basis=basis, handflaeche=n_palm, daumenseite=e_t, erkannt=True, verschmolzen=verschmolzen,
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
    # Hüftgelenk (Hüftkopf) anatomisch: ~0,51 H (Trochanter 0,53 H, Schritt 0,47 H) – nie unter Schritt + 6 cm
    z_hip = float(np.clip(max(zc + 0.06 * k, 0.505 * H), 0.49 * H, 0.53 * H)); info["schritt_z"] = round(zc, 3)

    def mitte_y(zz, breite, tol=0.01):                                # Mitte zwischen Vorder- und Rückseite (nicht Median:
        m = (np.abs(z - zz) < tol) & (np.abs(x) < breite)               # Hände/Taschen vorne ziehen den Median nach vorn)
        if m.sum() < 8: return None
        return 0.5 * (float(np.percentile(y[m], 3)) + float(np.percentile(y[m], 97)))

    # ---------- Schultern: Oberkante bei |x| ~ 0.17 m, Gelenk 5,5 cm darunter, 5 cm innerhalb der Außenseite
    sh = {}; info["arm_achse"] = {}; tops = {}
    ymed = float(np.median(y[(z > 0.6 * H) & (z < 0.8 * H) & (np.abs(x) < 0.12 * k)]))
    for side, s in (("L", 1), ("R", -1)):
        # 1) Schulter-OBERKANTE (Akromion + Kleidung) knapp innerhalb der Außenkante: oberste Oberfläche bei |x| = 17-21 cm,
        #    Haare/Kapuze ausgeschlossen (nur Körpertiefe, unter 0,86 H). Das Gelenk (Oberarmkopf) liegt ~4 cm darunter –
        #    früher wurde die Oberarmachse bis zur Schulter verlängert: bei hängenden Armen landete das 6-8 cm zu hoch.
        tb = []
        for xx in (0.17, 0.19, 0.21):
            band = (np.abs(s * x - xx * k) < 0.012 * k) & (z > 0.7 * H) & (z < 0.86 * H) & (np.abs(y - ymed) < 0.12 * k)
            if band.sum() > 10: tb.append(float(np.percentile(z[band], 98)))
        tops[side] = float(np.clip(float(np.median(tb)) if tb else 0.82 * H, 0.78 * H, 0.845 * H))
    # symmetrisch: Haare/Locken auf einer Schulter heben deren Oberkante -> höchstens 1,5 cm über der niedrigeren
    top = min((tops["L"] + tops["R"]) / 2, min(tops.values()) + 0.015 * k)
    z_sh = float(np.clip(top - 0.047 * k, 0.77 * H, 0.81 * H))
    xs = {}
    for side, s in (("L", 1), ("R", -1)):
        # 2) seitliche Lage: Oberarmachse (frei stehender Teil 26-42 cm neben der Mitte) auf Gelenkhöhe schneiden
        xs[side] = 0.18 * k
        Aw = P[(s * x > 0.26 * k) & (s * x < 0.42 * k) & (z > 0.55 * H) & (z < 0.9 * H)]
        if len(Aw) > 50:
            c_ = Aw.mean(0); _, _, vt = np.linalg.svd(Aw - c_, full_matrices=False); d_ = vt[0]
            if abs(d_[2]) > 0.2:
                pj = c_ + d_ * (z_sh - c_[2]) / d_[2]
                xs[side] = float(np.clip(s * pj[0], 0.165 * k, 0.205 * k))
            info["arm_achse"][side] = dict(neigung_grad=round(float(np.degrees(np.arcsin(abs(d_[2])))), 1))
    x_sh = (xs["L"] + xs["R"]) / 2                                   # Schulterbreite symmetrisch
    mb = (np.abs(z - z_sh) < 0.015 * k) & (np.abs(x) < 0.15 * k)      # um die Körpermitte auf Schulterhöhe (nicht x = 0)
    cx_sh = float(np.clip(0.5 * (np.percentile(x[mb], 5) + np.percentile(x[mb], 95)), -0.03 * k, 0.03 * k)) if mb.sum() > 12 else 0.0
    # Schultergelenk (Oberarmkopf) liegt ~5 cm innerhalb der Außenseite des Deltamuskels – an der Seite gemessen, an der der Arm
    # am engsten anliegt; höchstens 0,095 H neben der Mitte (Schulterbreite zwischen den Gelenken ~0,19 H). Früher 0,10-0,113 H.
    aussen = []
    for s_ in (1, -1):
        mo = (np.abs(z - z_sh) < 0.012 * k) & (np.abs(y - ymed) < 0.14 * k) & (s_ * (x - cx_sh) > 0)
        if mo.sum() > 12: aussen.append(float(np.percentile(s_ * (x[mo] - cx_sh), 99.5)))
    if aussen: x_sh = min(x_sh, min(aussen) - 0.05 * k)
    x_sh = float(np.clip(x_sh, 0.082 * H, 0.095 * H)); info["schulter_aussen"] = [round(v, 3) for v in aussen]
    for side, s in (("L", 1), ("R", -1)):
        # 3) Tiefe: Mitte des Schulter-Querschnitts (Gelenk liegt mittig zwischen Brust- und Rückenseite)
        sl = (np.abs(z - z_sh) < 0.015 * k) & (np.abs(x - (cx_sh + s * x_sh)) < 0.025 * k) & (np.abs(y - ymed) < 0.14 * k)
        yy = y[sl] if sl.sum() > 8 else y[np.abs(z - z_sh) < 0.03 * k]
        sh[side] = Vector((cx_sh + s * x_sh, 0.5 * (float(np.percentile(yy, 5)) + float(np.percentile(yy, 95))), z_sh))
    z_n0 = float(np.clip(top + 0.025 * k, 0.82 * H, 0.86 * H))     # Halsansatz (C7) knapp über der Schulter
    info["schulter_z"] = round(z_sh, 3); info["schulter_oben"] = round(top, 3); info["schulter_x"] = round(x_sh, 3)

    # ---------- Wirbelsäule + Kopf: jede Höhe einzeln am Körper ausgerichtet
    #   seitlich: Mitte des Rumpf-/Hals-/Kopf-Querschnitts (nicht einfach x = 0)
    #   Tiefe   : natürliche Krümmung – Becken mittig, Lende 62 %, Brust 72 % der Rumpftiefe von vorn, C7 4,5 cm unter der
    #             Rückenhaut, Halsmitte leicht vor der Verbindungslinie, Kopfgelenk (Atlas) auf Ohrläppchen-Höhe hinter dem Kiefer
    def quer(zz, breite, tol=0.012):
        m = (np.abs(z - zz) < tol) & (np.abs(x) < breite)
        if m.sum() < 12: return None
        cx = 0.5 * (float(np.percentile(x[m], 5)) + float(np.percentile(x[m], 95)))
        m2 = m & (np.abs(x - cx) < 0.06 * k)
        if m2.sum() < 8: m2 = m
        return cx, float(np.percentile(y[m2], 3)), float(np.percentile(y[m2], 97))
    C = carlos
    c_hip, c_n0 = C["spine"]["head"][2], C["spine.004"]["head"][2]
    def hz(t): return z_hip + (t - c_hip) / (c_n0 - c_hip) * (z_n0 - z_hip)
    z_at = float(np.clip(H - 0.09 * H, z_n0 + 0.07 * k, H - 0.12 * k))   # Atlas ~ Ohrläppchen
    hoehen = {("spine", "head"): z_hip}
    for n in SPINE[:4]: hoehen[(n, "tail")] = hz(C[n]["tail"][2])
    hoehen[("spine.003", "tail")] = z_n0
    hoehen[("spine.004", "tail")] = 0.5 * (z_n0 + z_at); hoehen[("spine.005", "tail")] = z_at
    hoehen[("spine.006", "tail")] = H * 0.985
    tiefe = {("spine", "head"): 0.55, ("spine", "tail"): 0.62, ("spine.001", "tail"): 0.68, ("spine.002", "tail"): 0.72}
    pts = {}
    for key, zz in sorted(hoehen.items(), key=lambda t: t[1]):
        br = 0.15 * k if zz < z_n0 - 0.03 * k else 0.09 * k
        qd = quer(min(zz, H - 0.03 * k), br)
        cx, yf, yb = qd if qd is not None else (0.0, -0.1 * k, 0.1 * k)
        cx = float(np.clip(cx, -0.04 * k, 0.04 * k))
        if key in tiefe: yy = yf + tiefe[key] * (yb - yf)
        elif key == ("spine.003", "tail"): yy = yb - 0.045 * k                     # C7
        elif key == ("spine.005", "tail"): yy = yf + 0.58 * (yb - yf)              # Atlas
        else: yy = None
        pts[key] = [cx, yy, zz]
    pts[("spine.004", "tail")][1] = 0.5 * (pts[("spine.003", "tail")][1] + pts[("spine.005", "tail")][1]) - 0.006 * k
    pts[("spine.006", "tail")][1] = pts[("spine.005", "tail")][1] - 0.01 * k
    # seitliche Lage leicht glätten (Querschnitts-Rauschen), Reihenfolge von unten nach oben
    order = [("spine", "head")] + [(n, "tail") for n in SPINE]
    xs_ = np.array([pts[o][0] for o in order]); xs_[1:-1] = 0.25 * xs_[:-2] + 0.5 * xs_[1:-1] + 0.25 * xs_[2:]
    for o, xv in zip(order, xs_): pts[o][0] = float(xv)
    prev = pts[("spine", "head")]
    J[("spine", "head")] = Vector(prev)
    for n in SPINE:
        J[(n, "tail")] = Vector(pts[(n, "tail")])
        if n != "spine": J[(n, "head")] = J[(SPINE[SPINE.index(n) - 1], "tail")].copy()
    becken_x = pts[("spine", "head")][0]
    info["wirbelsaeule"] = {o[0] + "." + o[1]: [round(v, 3) for v in pts[o]] for o in order}

    # ---------- Beine
    for side, s in (("L", 1), ("R", -1)):
        def bein(zz, tol=0.008):
            return P[(np.abs(z - zz) < tol) & (s * x > 0.0) & (s * x < 0.3 * k)]
        L1 = bein(zc - 0.08 * k, 0.012)
        hx = float(np.clip(abs(L1[:, 0].mean()), 0.07 * k, 0.12 * k)) if len(L1) else 0.1 * k
        hx = float(np.clip(hx, 0.082 * k, 0.092 * k))                      # Hüftköpfe ~8,5-9 cm neben der Beckenmitte (Abstand ~0,1 H)
        hy = mitte_y(z_hip, 0.08 * k)
        hip = Vector((becken_x + s * hx, hy if hy is not None else torso_y(z_hip, 0.13, 0.0), z_hip))   # mittig im Becken
        z_ank = 0.045 * H; A = bein(0.075 * H)
        def bein_mitte(B):                                              # Querschnitt-Mitte (Ausdehnung, nicht Punktschwerpunkt)
            return (0.5 * (float(np.percentile(B[:, 0], 5)) + float(np.percentile(B[:, 0], 95))),
                    0.5 * (float(np.percentile(B[:, 1], 5)) + float(np.percentile(B[:, 1], 95))))
        ax_, ay_ = bein_mitte(A); ankle = Vector((ax_, ay_ + 0.01 * k, z_ank))   # Sprunggelenk etwas hinter der Mitte
        z_knee = z_ank + 0.5 * (z_hip - z_ank); K = bein(z_knee, 0.012)
        kx_, ky_ = bein_mitte(K); knee = Vector((kx_, ky_ - 0.005 * k, z_knee))   # Knie leicht vor der Mitte
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
        wrist = mitte(d_w, 0.05); elbow = mitte(d_w * 0.56, 0.08)        # Oberarm : Unterarm = 0,186 H : 0,146 H
        d_hand = np.array((at(L) - wrist).normalized())
        info["arm_" + side] = dict(laenge=round(L, 3), handgelenk_bei=round(d_w, 3))
        qs = quer(z_sh + 0.025 * k, 0.12 * k)                          # Schlüsselbein beginnt vorn am Brustbein (Sternoklavikulargelenk)
        cy = qs[1] + 0.03 * k if qs is not None else J[("spine.003", "tail")].y - 0.06 * k
        J.update({("shoulder." + side, "head"): Vector((J[("spine.003", "tail")].x + s * 0.02 * k, cy, z_sh + 0.025 * k)),
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
        info["hand_" + side] = dict(handgelenk=w.tolist(), richtung=d_hand.tolist(), finger=hd["erkannt"], verschmolzen=bool(hd.get("verschmolzen", True)))
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
    ziele = {"f_index": (3, (18, 28, 16)), "f_middle": (0, (20, 32, 18)), "f_ring": (-3, (22, 34, 19)), "f_pinky": (-6, (24, 36, 20))}   # eng zusammen
    def drehe(pb, R):
        h = np.array(pb.head)
        M = Matrix.Translation(Vector(h)) @ Matrix(R.tolist()).to_4x4() @ Matrix.Translation(Vector(-h))
        pb.matrix = M @ pb.matrix; bpy.context.view_layer.update()
    for side in ("L", "R"):
        if not info["hand_" + side]["finger"]: continue
        if info["hand_" + side].get("verschmolzen"):              # Finger im Scan verschmolzen (Fäustling): Knochen bleiben wie gescannt
            rig["haende_wie_gescannt"] = (rig.get("haende_wie_gescannt", "") + side); continue
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
                if dd.min() > 0.045 * k: continue                       # nicht an der Hand (Jacke/Hüfte daneben) -> Heat-Gewichte lassen
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
            if min(segd(p, *B[n]) for n in ["hand." + side] + [f"{f}.0{i}.{side}" for f in ("thumb", "f_index", "f_middle", "f_ring", "f_pinky") for i in (1, 2, 3)]) > 0.045 * k:
                continue                                                # nicht an der Hand (z. B. Jacke an der Hüfte)
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

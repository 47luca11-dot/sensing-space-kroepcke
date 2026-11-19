# Werkzeuge für Gaussian-Splat-PLYs (3DGS-Format), nur numpy – läuft in python3 und in Blender.
#   lesen/schreiben, Boden finden, Person ausschneiden, drehen (inkl. Kugelflächen-Funktionen), Vorschau-PNG.
#
# Kommandozeile:
#   python3 splat_werkzeug.py ansicht <scan.ply> <out.png> [--zentrum x,y,z] [--radius 2.5] [--oben -y]
#       -> 4 Ansichten um die Hochachse, zum Finden der Person in einer Szene
#   python3 splat_werkzeug.py ausschneiden <scan.ply> <out.ply> --zentrum x,y,z [--oben -y] [--groesse 1.80] [--vorschau out.png]
#       -> Person freistellen, Boden/Wände/Schwebeteile weg, auf echte Größe skalieren,
#          Blender-Rahmen: Z oben, Blick nach -Y, Füße auf z=0, Mitte bei x=y=0
import sys, zlib, struct, json, math, warnings
import numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # macOS-Accelerate meldet hier Fehlalarme

SH0 = 0.28209479177387814

# ---------------------------------------------------------------- PLY
def lesen(pfad):
    with open(pfad, "rb") as f:
        kopf = b""
        while not kopf.endswith(b"end_header\n"):
            z = f.readline()
            if not z: raise ValueError("kein PLY-Kopf")
            kopf += z
        zeilen = kopf.decode().split("\n")
        if "binary_little_endian" not in kopf.decode(): raise ValueError("nur binary_little_endian")
        n = int([z for z in zeilen if z.startswith("element vertex")][0].split()[-1])
        props = [z.split()[-1] for z in zeilen if z.startswith("property")]
        if any(z.split()[1] != "float" for z in zeilen if z.startswith("property")): raise ValueError("nur float-Properties")
        a = np.fromfile(f, dtype="<f4", count=n * len(props)).reshape(n, len(props))
    return props, a.astype(np.float64)

def schreiben(pfad, props, a):
    with open(pfad, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(a)
                 + "".join(f"property float {p}\n" for p in props) + "end_header\n").encode())
        f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

def spalten(props, *namen): return [props.index(n) for n in namen]

def deckkraft(props, a): return 1 / (1 + np.exp(-a[:, props.index("opacity")]))
def farbe(props, a):
    return np.clip(0.5 + SH0 * a[:, spalten(props, "f_dc_0", "f_dc_1", "f_dc_2")], 0, 1)
def groesse(props, a): return np.exp(a[:, spalten(props, "scale_0", "scale_1", "scale_2")]).max(1)
def pos(props, a): return a[:, spalten(props, "x", "y", "z")]

# ---------------------------------------------------------------- Kugelflächen-Funktionen (SH) drehen
def _sh_basis(d):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]; xx, yy, zz = x * x, y * y, z * z
    b1 = np.stack([-y, z, -x], 1)
    b2 = np.stack([x * y, y * z, 2 * zz - xx - yy, x * z, xx - yy], 1)
    b3 = np.stack([y * (3 * xx - yy), x * y * z, y * (4 * zz - xx - yy), z * (2 * zz - 3 * xx - 3 * yy),
                   x * (4 * zz - xx - yy), z * (xx - yy), x * (xx - 3 * yy)], 1)
    return [b1, b2, b3]

def sh_drehen(props, a, R):
    """f_rest so umrechnen, dass die Farbe je Blickrichtung nach der Drehung R gleich bleibt."""
    rest = [p for p in props if p.startswith("f_rest_")]
    if not rest: return
    k = len(rest) // 3; grade = {3: 1, 8: 2, 15: 3}[k]
    rng = np.random.default_rng(1); D = rng.normal(size=(400, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    A, B = _sh_basis(D), _sh_basis(D @ R)                        # B = Basis bei R^T d
    idx = [props.index(f"f_rest_{i}") for i in range(len(rest))]
    start = 0
    for l in range(grade):
        m = A[l].shape[1]
        X, *_ = np.linalg.lstsq(A[l], B[l], rcond=None)          # B = A @ X
        for ch in range(3):
            cols = idx[ch * k + start: ch * k + start + m]
            a[:, cols] = a[:, cols] @ X.T
        start += m

# ---------------------------------------------------------------- Transformation
def quat_mul(q, r):
    w1, x1, y1, z1 = q.T; w2, x2, y2, z2 = r.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)

def mat_zu_quat(R):
    w = math.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-4:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2; q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = s / 4; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return q

def transformieren(props, a, R, t, s):
    """p' = s * R @ p + t ; Rotation, Größe und SH der Splats mitziehen."""
    P = pos(props, a); a[:, spalten(props, "x", "y", "z")] = s * P @ R.T + t
    if "nx" in props: a[:, spalten(props, "nx", "ny", "nz")] = a[:, spalten(props, "nx", "ny", "nz")] @ R.T
    sc = spalten(props, "scale_0", "scale_1", "scale_2"); a[:, sc] += math.log(s)
    rc = spalten(props, "rot_0", "rot_1", "rot_2", "rot_3")
    q = a[:, rc]; q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    a[:, rc] = quat_mul(np.broadcast_to(mat_zu_quat(R), q.shape), q)
    sh_drehen(props, a, R)

# ---------------------------------------------------------------- Vorschau
def png(pfad, img):
    h, w, _ = img.shape; raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(pfad, "wb").write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + ch(b"IDAT", zlib.compress(raw, 6)) + ch(b"IEND", b""))

def bild(P, C, rechts, oben, tiefe, mitte, halb, b=420, h=None, punkt=2):
    h = h or b
    with np.errstate(all="ignore"):
        u = (P - mitte) @ rechts; v = (P - mitte) @ oben; d = (P - mitte) @ tiefe
    img = np.full((h, b, 3), 38, np.uint8)
    sk = b / (2 * halb)
    x = (u * sk + b / 2).astype(np.int64); y = (h / 2 - v * sk).astype(np.int64)
    ok = (x >= 0) & (x < b) & (y >= 0) & (y < h); o = np.argsort(-d[ok])          # weit hinten zuerst, vorne zuletzt
    xs, ys, cs = x[ok][o], y[ok][o], (C[ok][o] * 255).astype(np.uint8)
    for dx in range(punkt):
        for dy in range(punkt):
            img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, b - 1)] = cs
    return img

def rundum(P, C, oben, mitte, halb, b=360, h=None, n=4):
    """n Ansichten um die Hochachse, nebeneinander. Blick in Richtung 'tiefe' (positive Tiefe = weiter weg)."""
    oben = oben / np.linalg.norm(oben)
    hilf = np.array([1.0, 0, 0]) if abs(oben[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(oben, hilf); e1 /= np.linalg.norm(e1); e2 = np.cross(oben, e1)
    kacheln = []
    for i in range(n):
        w = 2 * math.pi * i / n; blick = math.cos(w) * e1 + math.sin(w) * e2
        rechts = np.cross(blick, oben)
        kacheln.append(bild(P, C, rechts, oben, blick, mitte, halb, b, h))
    return np.concatenate(kacheln, 1)

ACHSEN = {"x": [1, 0, 0], "-x": [-1, 0, 0], "y": [0, 1, 0], "-y": [0, -1, 0], "z": [0, 0, 1], "-z": [0, 0, -1]}

# ---------------------------------------------------------------- Ausschneiden
def _boden(P, oben, zentrum, rng):
    """Bodenebene unter der Person per RANSAC; Normale nahe 'oben'."""
    h = (P - zentrum) @ oben
    horiz = (P - zentrum) - np.outer(h, oben)
    r = np.linalg.norm(horiz, axis=1)
    saeule = (r < 0.5) & (h < -0.3) & (h > -2.6)
    if saeule.sum() < 100: raise SystemExit("kein Boden unter der Person gefunden – Zentrum/oben prüfen")
    unten = np.percentile(h[saeule], 0.5)                              # Fußsohlen ~ Boden
    kand = P[(r < 1.5) & (np.abs(h - unten) < 0.08)]
    best = (0, None)
    for _ in range(600):
        s = kand[rng.choice(len(kand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); l = np.linalg.norm(n)
        if l < 1e-9: continue
        n /= l
        if n @ oben < 0: n = -n
        if n @ oben < 0.97: continue
        inl = np.abs((kand - s[0]) @ n) < 0.012
        if inl.sum() > best[0]: best = (inl.sum(), (n, s[0]))
    n, p0 = best[1]
    inl = kand[np.abs((kand - p0) @ n) < 0.012]                        # verfeinern: Ebene durch alle Treffer
    m = inl.mean(0); _, _, vt = np.linalg.svd(inl - m, full_matrices=False); n = vt[2] if vt[2] @ oben > 0 else -vt[2]
    return n, m

def _groesste_komponente(V):
    """V: Menge belegter Voxel (int-Array Nx3). Gibt Maske der größten 26-zusammenhängenden Gruppe zurück."""
    schl = {tuple(v): i for i, v in enumerate(map(tuple, V))}
    lab = -np.ones(len(V), np.int64); nb = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    gr = []
    for s in range(len(V)):
        if lab[s] >= 0: continue
        g = len(gr); lab[s] = g; st = [s]; n = 0
        while st:
            c = st.pop(); n += 1; x, y, z = V[c]
            for i, j, k in nb:
                t = schl.get((x + i, y + j, z + k))
                if t is not None and lab[t] < 0: lab[t] = g; st.append(t)
        gr.append(n)
    return lab == int(np.argmax(gr)), gr

def ausschneiden(props, a, zentrum, oben, ziel_hoehe=1.80, radius=1.0, voxel=0.02, log=print):
    rng = np.random.default_rng(3)
    P = pos(props, a); op = deckkraft(props, a); gr = groesse(props, a)
    oben = np.asarray(oben, float); oben /= np.linalg.norm(oben); zentrum = np.asarray(zentrum, float)
    n, p0 = _boden(P[op > 0.3], oben, zentrum, rng); oben = n
    h = (P - p0) @ n
    horiz = (P - p0) - np.outer(h, n); c0 = zentrum - p0; c0 -= (c0 @ n) * n
    rh = np.linalg.norm(horiz - c0, axis=1)
    roh = (h > 0.012) & (h < 2.6) & (rh < radius)
    # Zusammenhang über Voxel: nur deckende, kleine Splats bauen die Gruppe
    bau = roh & (op > 0.35) & (gr < 0.05)
    Vi = np.floor(P[bau] / voxel).astype(np.int64)
    uv, inv = np.unique(Vi, axis=0, return_inverse=True)
    maske_v, gruppen = _groesste_komponente(uv)
    # Gruppe um 1 Voxel aufweiten, dann alle Splats darin behalten (auch halbtransparente)
    menge = {tuple(v) for v in uv[maske_v]}
    weit = {(x + i, y + j, z + k) for (x, y, z) in menge for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)}
    Vall = np.floor(P / voxel).astype(np.int64)
    in_gr = np.array([tuple(v) in weit for v in Vall[roh]])
    behalten = np.zeros(len(P), bool); idx = np.where(roh)[0][in_gr]
    behalten[idx] = True
    behalten &= (op > 0.04) & (gr < 0.08)                              # fast unsichtbare und riesige Splats weg
    fuss0 = np.percentile(h[behalten & (op > 0.5)], 0.2)                 # Sohle der Person = Nullhöhe
    p0 = p0 + fuss0 * n; h = h - fuss0
    behalten &= h > -0.01
    hoehe = np.percentile(h[behalten & (op > 0.5)], 99.98)
    s = ziel_hoehe / hoehe
    log(f"boden_normale={np.round(n, 4).tolist()} gruppen={sorted(gruppen, reverse=True)[:5]} behalten={int(behalten.sum())} "
        f"hoehe_scan={hoehe:.3f} massstab={s:.4f}")
    b = a[behalten].copy()
    # Rahmen: z = oben; x = Hauptachse der Arme (A-Pose); Blick nach -Y
    Q = (pos(props, b) - p0); hb = Q @ n
    arm = Q[(hb > 0.55 * hoehe) & (hb < 0.8 * hoehe)]
    arm = arm - np.outer(arm @ n, n); arm -= arm.mean(0)
    _, _, vt = np.linalg.svd(arm, full_matrices=False); ex = vt[0] - (vt[0] @ n) * n; ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex); R = np.stack([ex, ey, n])                    # Zeilen = neue Achsen
    Pn = Q @ R.T
    # Mitte = Mitte der Beine (Füße allein sind unzuverlässig: Bodenreste, ein Fuß dichter gescannt)
    bein = Pn[(Pn[:, 2] > 0.15 * hoehe) & (Pn[:, 2] < 0.4 * hoehe)]
    xl, xr = np.percentile(bein[:, 0], [1, 99])
    t = -np.array([(xl + xr) / 2, np.median(bein[:, 1]), 0.0])
    Pn = Pn + t
    # Blickrichtung: Zehen zeigen nach vorn (-Y). Vergleich Fußspitze vs. Schienbein.
    schien = Pn[(Pn[:, 2] > 0.14 * hoehe) & (Pn[:, 2] < 0.2 * hoehe)][:, 1].mean()
    fy = Pn[Pn[:, 2] < 0.03 * hoehe][:, 1]
    vorn = fy.min() if abs(fy.min() - schien) > abs(fy.max() - schien) else fy.max()
    if vorn > schien: R = np.diag([-1.0, -1.0, 1.0]) @ R; t = np.diag([-1.0, -1.0, 1.0]) @ t
    # p' = s * (R @ (p - p0) + t)  ->  s*R @ p + s*(t - R@p0)
    transformieren(props, b, R, s * (t - R @ p0), s)
    return b, dict(massstab=s, splats=int(len(b)), hoehe_m=ziel_hoehe)

# ---------------------------------------------------------------- Kommandozeile
def _arg(name, std=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else std

if __name__ == "__main__":
    cmd, quelle, ziel = sys.argv[1:4]
    props, a = lesen(quelle)
    oben = np.array(ACHSEN[_arg("--oben", "z")], float)
    if cmd == "ansicht":
        P = pos(props, a); op = deckkraft(props, a)
        z = np.array([float(v) for v in _arg("--zentrum").split(",")]) if _arg("--zentrum") else np.median(P[op > 0.5], 0)
        r = float(_arg("--radius", 2.5)); k = (op > 0.4) & (np.abs(P - z).max(1) < r)
        png(ziel, rundum(P[k], farbe(props, a)[k], oben, z, r)); print("punkte", int(k.sum()), "zentrum", np.round(z, 3).tolist())
    elif cmd == "ausschneiden":
        z = np.array([float(v) for v in _arg("--zentrum").split(",")])
        b, info = ausschneiden(props, a, z, oben, float(_arg("--groesse", 1.80)), float(_arg("--radius", 1.0)))
        schreiben(ziel, props, b); print("INFO", json.dumps(info))
        if _arg("--vorschau"):
            P = pos(props, b); C = farbe(props, b); k = deckkraft(props, b) > 0.3
            png(_arg("--vorschau"), rundum(P[k], C[k], np.array([0, 0, 1.0]), np.array([0, 0, 0.9]), 1.0, 360, 400))

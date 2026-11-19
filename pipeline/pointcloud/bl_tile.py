# Stadtmodell -> Punktwolke, Schritt 2 (läuft im Blender über das MCP, nach bl_sample.py).
# Liest data/pointcloud/stage/*.npz, dünnt ortsabhängig auf die Zieldichte aus, sortiert in 32-m-Kacheln (XZ),
# mischt jede Kachel zufällig (jedes Präfix = gleichmäßige Stichprobe -> Dichte im Spiel per drawRange/Rang steuerbar)
# und schreibt web/assets/city/pointcloud/:
#   index.json            Kacheln {i,j,cx,cz,n,na,y0,y1,dens}, Quantisierung
#   t_<i>_<j>_a.bin       erste na Punkte (grobe Stufe, wird weit geladen)
#   t_<i>_<j>_b.bin       restliche Punkte (feine Stufe, nur in der Nähe)
# Dateiaufbau je .bin: Int16 x,y,z (n×3) | Uint8 r,g,b,flags (n×4) | Uint8 Okta-Normale (n×2)  -> 12 Byte/Punkt.
import numpy as np, os, json, glob, time, math

ROOT = r"C:\Users\luca\Documents\Sensing Space\Claude Code\output\hannover"
STAGE = os.path.join(ROOT, "data", "pointcloud", "stage")
OUT = os.path.join(ROOT, "web", "assets", "city", "pointcloud")
TILE = 32.0
QXZ, QY = 1 / 1000.0, 1 / 250.0
LO_FRAC = 0.12
# Zieldichte (Punkte/m²) – gleiche Kernbereiche wie in bl_sample.py, aber niedriger
CORE_T, BASE_T = 110.0, 9.0       # = Abtastdichte (bl_sample): keine Ausdünnung, maximale Dichte
KEEP_FULL = ("landmarks__gottfried_holzmarkt6", "scan__")   # Handmodelle: volle Abtastdichte behalten
# Eigene Scans (data/pointcloud/stage/scan__*.npz, z. B. pipeline/scans/steintor_eingang.py) ersetzen die Modellpunkte dort,
# wo der Scan etwas zeigt: Modellpunkte in 0,5-m-Voxeln mit Scan-Punkten fallen weg (tiefer liegende, nicht gescannte Teile bleiben).
SCAN_VOX = 0.5
def _vox(P):
    q = np.floor(P / SCAN_VOX).astype(np.int64)
    return ((q[:, 0] + 8192) * 1024 + (q[:, 1] + 512)) * 16384 + (q[:, 2] + 8192)

def build(sample_ns):
    dens_src = sample_ns["density"]
    CORE_S, BASE_S, FALL = sample_ns["CORE"], sample_ns["BASE"], sample_ns["FALL"]
    rng = np.random.default_rng(2026)
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "t_*.bin")): os.remove(f)
    buckets = {}
    t0 = time.time(); total_in = total_out = 0
    scan_files = sorted(glob.glob(os.path.join(STAGE, "scan__*.npz")))
    scan_keys = scan_lo = scan_hi = None; ersetzt = 0
    if scan_files:
        Ps = np.concatenate([np.load(f)["P"] for f in scan_files]).astype(np.float64)
        scan_keys = np.unique(_vox(Ps)); scan_lo = Ps.min(0) - SCAN_VOX; scan_hi = Ps.max(0) + SCAN_VOX
    for fn in sorted(glob.glob(os.path.join(STAGE, "*.npz"))):
        d = np.load(fn)
        P, C, N = d["P"], d["C"], d["N"]
        if len(P) == 0: continue
        if scan_keys is not None and not os.path.basename(fn).startswith("scan__"):
            inb = np.all((P >= scan_lo) & (P <= scan_hi), 1)
            if inb.any():
                ii = np.where(inb)[0]; weg = ii[np.isin(_vox(P[ii].astype(np.float64)), scan_keys)]
                if len(weg):
                    m = np.ones(len(P), bool); m[weg] = False; P, C, N = P[m], C[m], N[m]; ersetzt += len(weg)
        total_in += len(P)
        x, z = P[:, 0].astype(np.float64), P[:, 2].astype(np.float64)
        src = dens_src(x, z)
        w = (src - BASE_S) / (CORE_S - BASE_S)              # 0 außen .. 1 im Kern
        tgt = BASE_T + (CORE_T - BASE_T) * w
        keep = rng.random(len(P)) < np.clip(tgt / src, 0, 1)
        if any(k in os.path.basename(fn) for k in KEEP_FULL):
            keep[:] = True; C = C.copy(); C[:, 3] |= 8      # Flag 8 = fein (im Spiel kleinere Punkte)
        P, C, N = P[keep], C[keep], N[keep]
        total_out += len(P)
        ti = np.floor(P[:, 0] / TILE).astype(np.int32); tj = np.floor(P[:, 2] / TILE).astype(np.int32)
        key = (ti.astype(np.int64) + 1000) * 4096 + (tj.astype(np.int64) + 1000)
        order = np.argsort(key, kind="stable")
        key, P, C, N = key[order], P[order], C[order], N[order]
        u, start = np.unique(key, return_index=True)
        ends = list(start[1:]) + [len(key)]
        for k, s, e in zip(u, start, ends):
            buckets.setdefault(int(k), []).append((P[s:e], C[s:e], N[s:e]))
    tiles = []
    for k, parts in buckets.items():
        i = k // 4096 - 1000; j = k % 4096 - 1000
        P = np.concatenate([p[0] for p in parts]); C = np.concatenate([p[1] for p in parts]); N = np.concatenate([p[2] for p in parts])
        n = len(P)
        if n < 8: continue
        perm = rng.permutation(n); P, C, N = P[perm], C[perm], N[perm]
        cx, cz = (i + 0.5) * TILE, (j + 0.5) * TILE
        q = np.empty((n, 3), np.int16)
        q[:, 0] = np.clip(np.round((P[:, 0] - cx) / QXZ), -32767, 32767)
        q[:, 1] = np.clip(np.round(P[:, 1] / QY), -32767, 32767)
        q[:, 2] = np.clip(np.round((P[:, 2] - cz) / QXZ), -32767, 32767)
        na = max(1, int(math.ceil(n * LO_FRAC)))
        for suf, sl in (("a", slice(0, na)), ("b", slice(na, n))):
            if sl.stop - sl.start <= 0: continue
            with open(os.path.join(OUT, "t_%d_%d_%s.bin" % (i, j, suf)), "wb") as fh:
                fh.write(np.ascontiguousarray(q[sl]).tobytes()); fh.write(np.ascontiguousarray(C[sl]).tobytes()); fh.write(np.ascontiguousarray(N[sl]).tobytes())
        dmid = float(dens_src(np.array([cx]), np.array([cz]))[0])
        w = (dmid - BASE_S) / (CORE_S - BASE_S)
        tiles.append(dict(i=int(i), j=int(j), cx=cx, cz=cz, n=int(n), na=int(na),
                          y0=round(float(P[:, 1].min()), 2), y1=round(float(P[:, 1].max()), 2),
                          dens=round(BASE_T + (CORE_T - BASE_T) * w, 1)))
    tiles.sort(key=lambda t: (t["i"], t["j"]))
    idx = dict(version=1, tile=TILE, quant=[QXZ, QY, QXZ], bytesPerPoint=12, loFrac=LO_FRAC,
               layout="int16 xyz (x,z rel. Kachelmitte, y absolut) | uint8 rgb+flags | uint8 oct normal",
               flags={"twoSided": 1, "emissive": 2, "glass": 4, "fine": 8},
               total=int(sum(t["n"] for t in tiles)), tiles=tiles,
               source="pipeline/pointcloud/bl_sample.py + bl_tile.py (Blender %s über MCP)" % sample_ns.get("BL_VERSION", ""))
    if scan_files: idx["scans"] = dict(dateien=[os.path.basename(f) for f in scan_files], modellpunkte_ersetzt=int(ersetzt))
    json.dump(idx, open(os.path.join(OUT, "index.json"), "w"), separators=(",", ":"))
    cellify()
    return dict(tiles=len(tiles), pts_in=total_in, pts_out=total_out, sec=round(time.time() - t0, 1))

CELL = 8.0   # Zellen innerhalb einer Kachel (4 x 4): das Spiel zeichnet nur Zellen im Sichtradius

def cellify(out_dir=OUT):
    """Ordnet jede Kachel in 8-m-Zellen um (Reihenfolge Zelle 0..15, innerhalb der Zelle zufällig = gleichmäßige Stichprobe).
    _a = je Zelle die ersten loFrac-Anteile (grob, für den Scan), _b = Rest. index.json: tile["cells"] = [[na_k, nb_k], ...].
    Arbeitet auf fertigen (auch schon in bl_light gebackenen) Kacheln."""
    idx = json.load(open(os.path.join(out_dir, "index.json")))
    qx, qy, qz = idx["quant"]; T = idx["tile"]; nc = int(round(T / CELL)); lo = idx.get("loFrac", LO_FRAC)
    t0 = time.time()
    for t in idx["tiles"]:
        parts = []
        for suf, n in (("a", t["na"]), ("b", t["n"] - t["na"])):
            fn = os.path.join(out_dir, "t_%d_%d_%s.bin" % (t["i"], t["j"], suf))
            if n <= 0 or not os.path.exists(fn): continue
            raw = np.fromfile(fn, np.uint8)
            parts.append((raw[:n * 6].view(np.int16).reshape(n, 3), raw[n * 6:n * 10].reshape(n, 4), raw[n * 10:n * 12].reshape(n, 2)))
        Q = np.concatenate([p[0] for p in parts]); C = np.concatenate([p[1] for p in parts]); N = np.concatenate([p[2] for p in parts])
        cu = np.clip(np.floor((Q[:, 0] * qx + T / 2) / CELL).astype(np.int32), 0, nc - 1)
        cv = np.clip(np.floor((Q[:, 2] * qz + T / 2) / CELL).astype(np.int32), 0, nc - 1)
        cid = cv * nc + cu
        order = np.argsort(cid, kind="stable")          # stabil: zufällige Reihenfolge innerhalb der Zelle bleibt
        Q, C, N, cid = Q[order], C[order], N[order], cid[order]
        counts = np.bincount(cid, minlength=nc * nc)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        a_idx, b_idx, cells = [], [], []
        for k in range(nc * nc):
            c = int(counts[k]); s = int(starts[k])
            na = int(math.ceil(c * lo)) if c else 0
            a_idx.append(np.arange(s, s + na)); b_idx.append(np.arange(s + na, s + c)); cells.append([na, c - na])
        for suf, ix in (("a", np.concatenate(a_idx)), ("b", np.concatenate(b_idx))):
            fn = os.path.join(out_dir, "t_%d_%d_%s.bin" % (t["i"], t["j"], suf))
            if len(ix) == 0:
                if os.path.exists(fn): os.remove(fn)
                continue
            with open(fn, "wb") as fh:
                fh.write(np.ascontiguousarray(Q[ix]).tobytes()); fh.write(np.ascontiguousarray(C[ix]).tobytes()); fh.write(np.ascontiguousarray(N[ix]).tobytes())
        t["na"] = int(sum(c[0] for c in cells)); t["cells"] = cells
    idx["cell"] = CELL
    json.dump(idx, open(os.path.join(out_dir, "index.json"), "w"), separators=(",", ":"))
    return round(time.time() - t0, 1)

"""Koplanare, gleich ausgerichtete Ueberdeckungen je Kachel aufloesen (Z-Fighting im Spiel, FrontSide).

Gruppen: Dreiecke gleicher Ebene (Normale auf 1/200, Abstand auf 2 mm gerundet, wie die QA-Pruefung) aus verschiedenen Bauteilen
(ein Bauteil = ein MeshAcc-Teil: Wandpolygon, Kasten, Schrift ...). Ueberdecken sich zwei Bauteile in der Ebene um mehr als
min_ov m2, gilt:
  - exakt gleiche Bauteile (gleiche Dreiecke, gleiche Orientierung; z. B. doppelt erzeugte Ortgangbretter an gemeinsamen
    Giebeln, doppelte Gesimse) -> das spaetere wird entfernt;
  - Detail gegen Huelle/Detail: das Detailbauteil (Kasten, Brett, Schrift - in sich geschlossen) wird um step (8 mm, uebersteht
    die 16-bit-Quantisierung der Web-Kacheln) entlang der Flaechennormale nach vorn geschoben;
  - Huelle gegen Huelle: nichts wird verschoben (sonst Haarrisse an den Kanten); Dreiecke des nachrangigen Teils, die vollstaendig
    vom vorrangigen Teil ueberdeckt sind, werden entfernt, teilweise ueberdeckte werden in der Ebene beschnitten (Rest neu
    trianguliert, Raender liegen auf der deckenden Flaeche bzw. auf den alten Kanten: kein Loch).
Aufruf: resolve(acc) vor write_tile; liefert Statistik."""
import collections
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _tri_n(T):
    c = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    a = np.linalg.norm(c, axis=1) / 2
    return a, c / np.maximum(2 * a, 1e-12)[:, None]


def _sig(V, F):
    """Signatur eines Bauteils (Dreiecke auf 1 mm, zyklisch normiert, Orientierung bleibt)."""
    K = np.round(np.asarray(V, float)[np.asarray(F)] / 0.001).astype(np.int64)      # (m,3,3)
    out = []
    for t in K:
        k = int(np.lexsort(t.T[::-1])[0])
        out.append(tuple(np.roll(t, -k, axis=0).reshape(-1).tolist()))
    return hash(tuple(sorted(out)))


def _resolve(acc, step, min_ov, sh):
    # verdeckte Hinterlegungen ('shell_b', 2-4 mm hinter Wandkronen/Attiken) bleiben unberuehrt: sie liegen absichtlich knapp
    # hinter sichtbaren Flaechen und schliessen dort mm-Fugen
    plist = [(k, i) for k, lst in acc.parts.items() if k[0] != 'shell_b' for i in range(len(lst))]
    if not plist:
        return collections.Counter()
    Ts = []; P = []; LI = []
    for p, (k, i) in enumerate(plist):
        V, F, C = acc.parts[k][i]
        F = np.asarray(F)
        if len(F):
            Ts.append(np.asarray(V, float)[F]); P.append(np.full(len(F), p, np.int64)); LI.append(np.arange(len(F)))
    T = np.concatenate(Ts); pid = np.concatenate(P); lid = np.concatenate(LI)
    a, n = _tri_n(T)
    sel = np.where(a > 1e-4)[0]
    T, pid, lid, a, n = T[sel], pid[sel], lid[sel], a[sel], n[sel]
    kill = collections.defaultdict(set)
    add_parts = []
    d = np.einsum('ij,ij->i', n, T[:, 0])
    # grobes Raster (Normale 1/60, Abstand 2 cm), echte Koplanaritaet wird je Paar geprueft (< 3 mm, < 0.8 Grad)
    q = np.c_[np.floor(n * 60 + sh).astype(np.int64), np.floor(d / 0.02 + sh).astype(np.int64)]
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    gp = np.unique(np.c_[inv, pid], axis=0)
    cnt = np.bincount(gp[:, 0], minlength=inv.max() + 1)
    multi = set(np.where(cnt > 1)[0].tolist())
    if not multi:
        return collections.Counter()
    order = np.argsort(inv, kind='stable'); invs = inv[order]
    starts = np.searchsorted(invs, np.arange(inv.max() + 2))
    is_det = np.array([0 if k[0].startswith('shell') else 1 for k, _ in plist])
    move = collections.defaultdict(lambda: np.zeros(3))
    drop = set(); sigs = {}
    st = collections.Counter()
    for g in multi:
        ids = order[starts[g]:starts[g + 1]]
        nrm = n[ids[0]]
        u = np.cross(nrm, [0.0, 1.0, 0.0])
        if np.linalg.norm(u) < 0.1:
            u = np.cross(nrm, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u); v = np.cross(nrm, u)
        by = collections.defaultdict(list)
        for i in ids:
            by[int(pid[i])].append(i)
        if len(by) < 2:
            continue
        # 2D-Huellrechtecke je Bauteil (schneller Vorfilter)
        box = {}
        for p, L in by.items():
            X = T[L] @ u; Y = T[L] @ v
            box[p] = (X.min(), Y.min(), X.max(), Y.max())
        ps = list(by)
        cand = set()
        for x in range(len(ps)):
            for y in range(x + 1, len(ps)):
                A, B = box[ps[x]], box[ps[y]]
                if (min(A[2], B[2]) - max(A[0], B[0])) * (min(A[3], B[3]) - max(A[1], B[1])) > min_ov and \
                        min(A[2], B[2]) > max(A[0], B[0]) and min(A[3], B[3]) > max(A[1], B[1]):
                    cand.add(ps[x]); cand.add(ps[y])
        if not cand:
            continue
        poly = {}
        pd = {}
        for p in cand:
            L = by[p]
            wts = a[L]
            pd[p] = (float(np.average(d[L], weights=wts)), np.average(n[L], 0, wts))
        for p in cand:
            L = by[p]
            pg = [Polygon(np.c_[T[i] @ u, T[i] @ v]) for i in L]
            poly[p] = unary_union([x.buffer(0) for x in pg if x.area > 1e-6])
        area = {p: poly[p].area for p in cand}
        pri = sorted(cand, key=lambda p: (is_det[p], -area[p], p))
        placed = []                                   # (pid, poly, level)
        for p in pri:
            lvl = -1; dup_of = None
            for (o, po, lo) in placed:
                if abs(pd[o][0] - pd[p][0]) > 0.003 or float(pd[o][1] @ pd[p][1]) < 0.9999:
                    continue                                   # nicht wirklich koplanar
                try:
                    ov = po.intersection(poly[p]).area
                except Exception:
                    ov = 0.0
                if ov > min_ov:
                    lvl = max(lvl, lo)
                    if dup_of is None and abs(area[o] - area[p]) < 1e-3 and abs(ov - area[p]) < 1e-3:
                        dup_of = o
            if lvl < 0:
                placed.append((p, poly[p], 0)); continue
            if not is_det[p]:
                # Huelle unter Huelle: vollstaendig ueberdeckte Dreiecke entfernen, Rest bleibt (keine Verschiebung)
                cover = unary_union([po for (o, po, lo) in placed if not is_det[o] and abs(pd[o][0] - pd[p][0]) <= 0.003
                                     and float(pd[o][1] @ pd[p][1]) >= 0.9999])
                if not cover.is_empty:
                    cb = cover.buffer(0.002)
                    part_ = []
                    for i in by[p]:
                        tp = Polygon(np.c_[T[i] @ u, T[i] @ v])
                        if tp.area <= 1e-6:
                            continue
                        if cb.contains(tp):
                            kill[p].add(int(lid[i])); st['removed_tris'] += 1
                        elif tp.intersection(cover).area > 2e-4:
                            part_.append((i, tp))
                    if part_:
                        rest = unary_union([tp for _, tp in part_]).difference(cover)
                        dpl = float(np.mean([T[i] @ nrm for i, _ in part_]).mean())
                        V, F_, C_ = acc.parts[plist[p][0]][plist[p][1]]
                        cc = np.asarray(C_, float)[np.asarray(F_)[[int(lid[i]) for i, _ in part_]].reshape(-1)].mean(0)
                        newV = []; newF = []; off = 0
                        for g in ([rest] if rest.geom_type == 'Polygon' else list(getattr(rest, 'geoms', []))):
                            if g.geom_type != 'Polygon' or g.area < 1e-4:
                                continue
                            from geom import tri2d
                            g = g.simplify(0.0005)
                            if g.is_empty or g.geom_type != 'Polygon':
                                continue
                            V2, F2 = tri2d(np.asarray(g.exterior.coords)[:-1], [np.asarray(h.coords)[:-1] for h in g.interiors])
                            if len(F2) == 0:
                                continue
                            W = V2[:, :1] * u + V2[:, 1:2] * v + dpl * nrm
                            tri = W[F2]; nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                            fl = nn @ nrm < 0
                            F2 = F2.copy(); F2[fl] = F2[fl][:, [0, 2, 1]]
                            newV.append(W); newF.append(F2 + off); off += len(W)
                        for i, _ in part_:
                            kill[p].add(int(lid[i])); st['clipped_tris'] += 1
                        if newV:
                            add_parts.append((plist[p][0], np.concatenate(newV), np.concatenate(newF),
                                              np.broadcast_to(cc.astype(np.float32), (off, 3)).copy()))
                placed.append((p, poly[p], 0)); continue
            if dup_of is not None and p not in drop and dup_of not in drop:
                ko, io = plist[dup_of]; kp, ip = plist[p]
                if ko == kp:
                    so = sigs.get(dup_of) or sigs.setdefault(dup_of, _sig(*acc.parts[ko][io][:2]))
                    sp = sigs.get(p) or sigs.setdefault(p, _sig(*acc.parts[kp][ip][:2]))
                    if so == sp:
                        drop.add(p); st['dropped_parts'] += 1
                        continue
            move[p] = move[p] + nrm * step * (lvl + 1)
            st['moved_faces'] += 1; st['ov_m2'] += float(area[p])
            placed.append((p, poly[p], lvl + 1))
    # anwenden
    newparts = collections.defaultdict(list); ntri = 0
    for k, lst in acc.parts.items():
        if k[0] == 'shell_b':
            newparts[k].extend(lst); ntri += sum(len(F) for _, F, _ in lst)
    for p, (k, i) in enumerate(plist):
        if p in drop:
            continue
        V, F, C = acc.parts[k][i]
        if p in kill:
            keep = np.ones(len(F), bool); keep[list(kill[p])] = False
            F = np.asarray(F)[keep]
            if len(F) == 0:
                continue
        if p in move:
            mv = move[p]
            if np.linalg.norm(mv) > 0.05:                  # Sicherheitsgrenze
                mv = mv / np.linalg.norm(mv) * 0.05
            V = np.asarray(V, float) + mv
        newparts[k].append((V, F, C)); ntri += len(F)
    for k, V, F, C in add_parts:
        newparts[k].append((V, F, C)); ntri += len(F)
    acc.parts = dict(newparts); acc.ntri = ntri
    st['moved_parts'] = len(move)
    return st


def resolve(acc, step=0.008, min_ov=0.003):
    """Zwei Durchgaenge mit gegeneinander versetzten Ebenen-Rastern (sonst entgehen Paare an Rastergrenzen)."""
    st = collections.Counter()
    for sh in (0.0, 0.5):
        st.update(_resolve(acc, step, min_ov, sh))
    return dict(st)

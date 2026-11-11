"""Strahlen-Pruefung der ausgelieferten (komprimierten) Kacheln, wie sie das Spiel laedt (nur FrontSide):

1. Fassaden: fuer jede freiliegende LoD2-Wand Strahlen aus 3 m Abstand senkrecht auf die Wand, in jeder Geschoss-
   Mitte und zwischen den Fensterachsen (Raster 1.25 m). Erwarteter Treffer: ein 'shell'-Dreieck, Vorderseite zum
   Betrachter, in 3 m +- 0.5 m (Wandflaeche) oder bis 1.6 m dahinter (Oeffnung mit Rueckwand); bei Fachwerk wegen der
   geschossweisen Auskragung bis 0.8 m davor.
2. Fenster: jede Glas-Instanz (aus *_win.glb) muss in einer Oeffnung der Huelle sitzen: Strahl durch die Glasmitte
   trifft die Rueckwand (Vorderseite), Strahlen 15 cm ausserhalb der vier Raender treffen Wand/Laibung.
3. Bounding-Spheres: three.js-Logik (Accessor min/max * Normalisierung * Knotenmassstab) muss die Geometrie umfassen.
4. Tuerschwellen: jede Laden-/Haustuer-Instanz (sf_door*, sfw_door, door_*) muss auf dem Gehweg davor stehen (Hoehenfeld der
   Boden-Pipeline, Median 0.8-1.6 m vor der Wand): Ladentueren +-0.15 m, Haustueren -0.15..+0.5 m (Stufen). Tueren an Gruben-
   kanten (Promenade/Treppenabgang: Gelaende 1 m und 3 m vor der Tuer weicht > 0.6 m ab) werden gesondert gezaehlt.
Aufruf: python check_rays.py [web_dir]"""
import os, sys, json, struct, time, collections, math
import numpy as np
import trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WEB_OUT, CACHE, save_json
import prep as PREP

CT = {5126: np.float32, 5122: np.int16, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
NORM = {np.int16: 32767.0, np.uint16: 65535.0, np.uint8: 255.0}


def read(path):
    b = open(path, 'rb').read(); jl = struct.unpack('<I', b[12:16])[0]
    J = json.loads(b[20:20 + jl]); BIN = b[20 + jl + 8:]
    return J, BIN


def view_bytes(J, BIN, i):
    bv = J['bufferViews'][i]
    e = bv.get('extensions', {}).get('EXT_meshopt_compression')
    if e is not None:
        import meshopt
        data = BIN[e['byteOffset']:e['byteOffset'] + e['byteLength']]
        return np.asarray(meshopt.decode_gltf_buffer(data, e['count'], e['byteStride'], e['mode'])).view(np.uint8).reshape(-1), bv
    o = bv.get('byteOffset', 0)
    return np.frombuffer(BIN, np.uint8, count=bv['byteLength'], offset=o), bv


def accessor(J, BIN, i, raw=False):
    a = J['accessors'][i]; data, bv = view_bytes(J, BIN, a['bufferView'])
    dt = CT[a['componentType']]; nc = NC[a['type']]; es = np.dtype(dt).itemsize * nc
    st = bv.get('byteStride', es)
    arr = np.zeros((a['count'], st), np.uint8)
    need = st * a['count']
    flat = np.zeros(need, np.uint8); flat[:min(need, len(data))] = data[:need]
    arr = flat.reshape(a['count'], st)[:, :es].copy().view(dt).reshape(a['count'], nc).astype(np.float64)
    if a.get('normalized') and not raw:
        arr = np.maximum(arr / NORM[dt], -1.0)
    return arr


def node_mats(J):
    """Weltmatrizen (nur T/S, keine Rotation in unseren Dateien)."""
    out = {}
    def rec(i, T, S):
        nd = J['nodes'][i]
        t = np.array(nd.get('translation', [0, 0, 0])); s = np.array(nd.get('scale', [1, 1, 1]))
        T2 = T + S * t; S2 = S * s
        out[i] = (T2, S2)
        for c in nd.get('children', []):
            rec(c, T2, S2)
    for r in J['scenes'][0]['nodes']:
        rec(r, np.zeros(3), np.ones(3))
    return out


def load_shell(path, bs_errors, inner=None):
    J, BIN = read(path)
    M = node_mats(J)
    tris = []
    for i, nd in enumerate(J['nodes']):
        if 'mesh' not in nd or '_shell' not in nd['name']:
            continue
        T, S = M[i]
        for pr in J['meshes'][nd['mesh']]['primitives']:
            P = accessor(J, BIN, pr['attributes']['POSITION']) * S + T
            I = accessor(J, BIN, pr['indices']).astype(np.int64).reshape(-1, 3)
            tris.append(P[I])
            if inner is not None:
                inner.append(np.full(len(I), J['materials'][pr['material']]['name'] in ('bldg_interior', 'bldg_shop_interior')))
            # Bounding-Sphere wie three.js GLTFLoader (min/max * Normalisierung)
            a = J['accessors'][pr['attributes']['POSITION']]
            mn = np.array(a['min'], float); mx = np.array(a['max'], float)
            if a.get('normalized'):
                mn /= NORM[CT[a['componentType']]]; mx /= NORM[CT[a['componentType']]]
            lo = mn * S + T; hi = mx * S + T
            Pw = P
            if (Pw.min(0) < np.minimum(lo, hi) - 0.05).any() or (Pw.max(0) > np.maximum(lo, hi) + 0.05).any():
                bs_errors.append((os.path.basename(path), nd['name']))
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def load_glass(path):
    J, BIN = read(path)
    out = []
    for nd in J['nodes']:
        ext = nd.get('extensions', {}).get('EXT_mesh_gpu_instancing')
        if ext is None or not nd['name'].endswith('_glass'):
            continue
        at = ext['attributes']
        T = accessor(J, BIN, at['TRANSLATION']); Q = accessor(J, BIN, at['ROTATION']); S = accessor(J, BIN, at['SCALE'])
        out.append((T, Q, S))
    if not out:
        return None
    return tuple(np.concatenate([o[k] for o in out]) for k in range(3))


def load_doors(path):
    J, BIN = read(path)
    out = []
    for nd in J['nodes']:
        ext = nd.get('extensions', {}).get('EXT_mesh_gpu_instancing')
        if ext is None:
            continue
        nm = nd['name']
        kind = 'shop' if ('_sf_door' in nm or '_sfw_door' in nm) else ('house' if '_door_' in nm else None)
        if kind is None:
            continue
        at = ext['attributes']
        T = accessor(J, BIN, at['TRANSLATION']); Q = accessor(J, BIN, at['ROTATION'])
        out.append((kind, T, Q))
    return out


def qrot(Q, v):
    """Quaternion (x,y,z,w) dreht Vektor v (N,3)."""
    q = Q[:, :3]; w = Q[:, 3:4]
    t = 2 * np.cross(q, v)
    return v + w * t + np.cross(q, t)


LAST_TRI = {}


def first_hits(mesh, O, D):
    """erster VORDERSEITIGER Treffer je Strahl (wie FrontSide im Spiel; Rueckseiten sind dort unsichtbar)
    -> (dist, front); dist=inf wenn keiner. front=False markiert: naechster Treffer insgesamt war eine Rueckseite."""
    dist = np.full(len(O), np.inf); front = np.zeros(len(O), bool)
    LAST_TRI['t'] = np.full(len(O), -1)
    if len(O) == 0:
        return dist, front
    loc, ri, ti = mesh.ray.intersects_location(O, D, multiple_hits=True)
    if len(ri) == 0:
        return dist, front
    d = np.einsum('ij,ij->i', loc - O[ri], D[ri])
    fr = np.einsum('ij,ij->i', mesh.face_normals[ti], D[ri]) < 0
    keep = fr & (d > -1e-6)
    ri, d = ri[keep], d[keep]
    order = np.lexsort((d, ri))
    ri, d = ri[order], d[order]
    firstmask = np.r_[True, ri[1:] != ri[:-1]] if len(ri) else np.zeros(0, bool)
    dist[ri[firstmask]] = d[firstmask]
    front[ri[firstmask]] = True
    tri = np.full(len(O), -1); tri[ri[firstmask]] = ti[keep][order][firstmask]
    LAST_TRI['t'] = tri
    return dist, front


def hits_through(mesh, O, D, lo, hi, rounds=4):
    """Erster Vorderseiten-Treffer; liegt er vor dem erwarteten Bereich (Verdecker wie Vordach, Nachbarbau, tieferes
    Dach), wird hinter dem Verdecker weitergesucht. Rueckgabe: Distanz ab O (inf = nichts), Anzahl Verdecker."""
    dist, front = first_hits(mesh, O, D)
    tri = LAST_TRI['t'].copy()
    tot = dist.copy(); occl = np.zeros(len(O), int)
    for _ in range(rounds):
        near = np.isfinite(tot) & (tot < lo)
        if not near.any():
            break
        idx = np.where(near)[0]
        d2, _ = first_hits(mesh, O[idx] + D[idx] * (tot[idx] + 0.02)[:, None], D[idx])
        tot[idx] = tot[idx] + 0.02 + d2
        tri[idx] = LAST_TRI['t']
        occl[idx] += 1
    LAST_TRI['hit'] = tri
    return tot, occl


def main():
    web = sys.argv[1] if len(sys.argv) > 1 else WEB_OUT
    t0 = time.time()
    idx = json.load(open(os.path.join(web, 'index.json'), encoding='utf-8'))
    d = PREP.load_prep(); vol = d['vol']
    by_id = {b.id: b for b in d['bldgs']}
    import facade as FA
    from build_all import make_ctx
    ctx = make_ctx(d)
    bs_errors = []
    kinds = {i['id']: i['kind'] for i in json.load(open(os.path.join(CACHE, 'building_info.json'), encoding='utf-8'))}
    report = {'walls_checked': 0, 'wall_rays': 0, 'wall_fail': 0, 'windows': 0, 'window_fail': 0, 'fail_examples': [], 'bs_errors': []}
    per_bldg = collections.Counter()
    # alle Huellen laden (Nachbarkacheln koennen Waende ueberdecken) -> ein Mesh
    allT = []; inner = []
    for e in idx['tiles']:
        allT.append(load_shell(os.path.join(web, e['file']), bs_errors, inner))
    T = np.concatenate(allT); INNER = np.concatenate(inner)
    mesh = trimesh.Trimesh(T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3), process=False)
    print('Huellen geladen: %d Dreiecke, %.1f s' % (len(T), time.time() - t0))
    # ---------------- 1. Fassaden
    O = []; D = []; who = []
    for e in idx['tiles']:
        for bid in e['buildings']:
            b = by_id[bid]
            if kinds.get(bid) == 'canopy':
                continue
            walls = FA.analyse_walls(b, ctx)
            for w in walls:
                if w.drop or not w.vertical or w.L < 1.0:
                    continue
                report['walls_checked'] += 1
                us = np.arange(w.umin + 0.4, w.umax - 0.3, 1.25)
                if len(us) == 0:
                    us = np.array([(w.umin + w.umax) / 2])
                vs = np.arange(max(w.vmin, b.y_ground) + 1.7, w.vmax - 0.3, 1.6)
                if len(vs) == 0:
                    continue
                UU, VV = np.meshgrid(us, vs); pts = np.c_[UU.ravel(), VV.ravel()]
                from shapely import contains_xy
                pts = pts[contains_xy(w.poly.buffer(-0.25), pts[:, 0], pts[:, 1])]
                if len(pts) == 0:
                    continue
                # nur freiliegende Stellen (kein Gebaeude innerhalb 3 m davor)
                Q = np.concatenate([w.fr.w(np.c_[pts, np.full(len(pts), dd)]) for dd in (0.3, 1.5, 2.9)])
                bo, bs = vol.blocked(Q, b.idx)
                free = ~((bo | bs).reshape(3, -1).any(0))
                pts = pts[free]
                if len(pts) == 0:
                    continue
                W3 = w.fr.w(np.c_[pts, np.zeros(len(pts))])
                O.append(W3 + w.n * 3.0); D.append(np.repeat(-w.n[None], len(pts), 0))
                who += [(bid, w.idx)] * len(pts)
    O = np.concatenate(O); D = np.concatenate(D)
    # Fachwerk: Obergeschosse kragen geschossweise vor (Auskragung bis 0.62 m vor die LoD2-Wand) -> Huelle frueher erwartet
    fw = np.array([kinds.get(x[0]) == 'fachwerk' for x in who])
    lo = np.where(fw, 2.2, 2.45)
    dist, occl = hits_through(mesh, O, D, lo, 4.65)
    front = np.isfinite(dist)
    deep = np.array([kinds.get(x[0]) == 'parking' for x in who])
    ht = LAST_TRI['hit']
    shop = (ht >= 0) & INNER[np.maximum(ht, 0)]          # Strahl durch ein Schaufenster in den (geschlossenen) Ladenraum
    report['wall_rays_into_shops'] = int((shop & front & (dist > 4.65)).sum())
    ok = front & (dist > lo) & (dist < np.where(deep | shop, 9.6, 4.65))
    report['wall_rays_occluded'] = int((occl > 0).sum())
    report['wall_rays'] = int(len(O)); report['wall_fail'] = int((~ok).sum())
    for i in np.where(~ok)[0]:
        per_bldg[who[i][0]] += 1
    bad = np.where(~ok)[0]
    for i in bad[:60]:
        report['fail_examples'].append({'type': 'wall', 'bldg': who[i][0], 'wall': int(who[i][1]), 'p': (O[i] + D[i] * 3).round(2).tolist(),
                                        'dist': None if not np.isfinite(dist[i]) else round(float(dist[i]), 2), 'front': bool(front[i])})
    print('Fassaden: %d Strahlen, %d Fehler, %.1f s' % (len(O), (~ok).sum(), time.time() - t0))
    # ---------------- 2. Fenster
    O2 = []; D2 = []; kinds = []; wid = []
    for e in idx['tiles']:
        if not e.get('windows'):
            continue
        g = load_glass(os.path.join(web, e['windows']))
        if g is None:
            continue
        Tt, Qq, Ss = g
        n = qrot(Qq, np.tile([0, 0, 1.0], (len(Tt), 1))); t = qrot(Qq, np.tile([1.0, 0, 0], (len(Tt), 1)))
        up = qrot(Qq, np.tile([0, 1.0, 0], (len(Tt), 1)))
        w_ = Ss[:, 0]; h_ = Ss[:, 1]
        C = Tt + up * (h_ / 2)[:, None]
        base = len(report['fail_examples'])
        for k, (off_t, off_v, kind) in enumerate(((0, 0, 'center'), (-1, 0, 'left'), (1, 0, 'right'), (0, 1, 'top'))):
            P = C + t * (off_t * (w_ / 2 + 0.15))[:, None] + up * (off_v * (h_ / 2 + 0.15))[:, None]
            # nur Proben, deren Weg zur Wand frei ist (kein Nachbargebaeude davor/daneben)
            Qp = np.concatenate([P + n * dd for dd in (0.3, 1.5, 2.9)])
            bo, bs = vol.blocked(Qp, -1)
            free = ~((bo | bs).reshape(3, -1).any(0))
            O2.append((P + n * 3.0)[free]); D2.append((-n)[free]); kinds += [kind] * int(free.sum())
            wid += list(np.arange(report['windows'], report['windows'] + len(Tt))[free])
        report['windows'] += len(Tt)
    O2 = np.concatenate(O2); D2 = np.concatenate(D2); kinds = np.array(kinds); wid = np.array(wid)
    dist2, occl2 = hits_through(mesh, O2, D2, 2.3, 9.0)
    front2 = np.isfinite(dist2)
    # Glasmitte: Rueckwand liegt 3 m + Laibung (0.05..0.35) + Raumtiefe (0.12..5) hinter dem Startpunkt
    okc = front2 & (dist2 > 3.0) & (dist2 < 9.0)
    oke = front2 & (dist2 > 2.3) & (dist2 < 9.0)
    ok2 = np.where(kinds == 'center', okc, oke)
    badw = np.unique(wid[~ok2])
    report['window_fail'] = int(len(badw))
    for i in np.where(~ok2)[0][:40]:
        report['fail_examples'].append({'type': 'window_' + kinds[i], 'p': (O2[i] + D2[i] * 3).round(2).tolist(),
                                        'dist': None if not np.isfinite(dist2[i]) else round(float(dist2[i]), 2), 'front': bool(front2[i])})
    # ---------------- 4. Tuerschwellen gegen das Hoehenfeld
    import level as LV
    nd = {'shop': 0, 'house': 0}; nbad = {'shop': 0, 'house': 0}; nedge = 0; offs = []
    report['door_fail_examples'] = []
    for e in idx['tiles']:
        if not e.get('windows'):
            continue
        for kind, Tt, Qq in load_doors(os.path.join(web, e['windows'])):
            n = qrot(Qq, np.tile([0, 0, 1.0], (len(Tt), 1)))
            H = np.median(np.stack([LV.sample((Tt + n * dd)[:, 0], (Tt + n * dd)[:, 2]) for dd in (1.0, 1.4, 1.8)]), 0)
            far = LV.sample((Tt + n * 3.5)[:, 0], (Tt + n * 3.5)[:, 2])
            off = Tt[:, 1] - H
            edge = np.abs(far - H) > 0.6
            lo, hi = (-0.15, 0.15) if kind == 'shop' else (-0.15, 0.5)
            bad = np.isfinite(off) & ((off < lo) | (off > hi)) & ~edge
            nd[kind] += len(Tt); nbad[kind] += int(bad.sum()); nedge += int((edge & np.isfinite(off)).sum())
            offs += list(off[np.isfinite(off)])
            for i in np.where(bad)[0][:10]:
                if len(report['door_fail_examples']) < 40:
                    report['door_fail_examples'].append({'tile': e['name'], 'kind': kind, 'p': Tt[i].round(2).tolist(),
                                                         'ground': round(float(H[i]), 2), 'offset': round(float(off[i]), 2)})
    report['doors'] = nd; report['door_fail'] = nbad; report['doors_at_pit_edges'] = nedge
    report['door_offset_median'] = round(float(np.median(offs)), 3) if offs else None
    print('Tuerschwellen: Laden %d (Fehler %d), Haus %d (Fehler %d), an Grubenkanten %d, Median-Versatz %s' % (
        nd['shop'], nbad['shop'], nd['house'], nbad['house'], nedge, report['door_offset_median']))
    report['bs_errors'] = bs_errors
    report['worst_buildings'] = per_bldg.most_common(25)
    save_json(report, os.path.join(CACHE, 'ray_report.json'), indent=1)
    print('Fenster: %d, fehlerhaft %d; Bounding-Sphere-Fehler %d; %.1f s' % (report['windows'], len(badw), len(bs_errors), time.time() - t0))
    print('schlechteste Gebaeude:', per_bldg.most_common(10))


if __name__ == '__main__':
    main()

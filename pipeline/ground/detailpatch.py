"""Detailmodell-Flicken: alte Geometrie im Fussabdruck entfernen (wie blender/apply_landmark_replace.py: Dreiecke, deren
Schwerpunkt im Polygon liegt und ymin <= y <= ymax) und die entstehende Bodenluecke mit eigener Triangulation schliessen.
Randpunkte = exakt die Ecken der verbleibenden Detail-Dreiecke (keine T-Stoesse), Hoehen/UV/Material aus den entfernten
Original-Dreiecken (baryzentrisch), in der alten Trog-Oeffnung linear aus dem Umfeld interpoliert."""
import os, re, json, math
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union
import triangle as tr
import trimesh
from common import *
import geom as G

DETAIL_RAW = DATA + 'detail_raw/'
GROUND_FILES = ('street', 'ground')

def _san(n):
    return 'd_' + re.sub(r'[^a-z0-9]+', '_', n.lower()).strip('_')

class Detail:
    def __init__(self, files=('street', 'ground', 'street_furniture')):
        self.meshes = []        # dicts: file, mat, V (n,3), F (m,3), UV (n,2)|None, C (m,3) centroids, N (m,3)
        self.mats = {}          # material name -> spec (fuer GLB)
        os.makedirs(OUT + 'tex', exist_ok=True)
        for f in files:
            sc = trimesh.load(DETAIL_RAW + f + '.glb', force='scene')
            for nn in sc.graph.nodes_geometry:
                T, g = sc.graph.get(nn); m = sc.geometry[g]
                V = trimesh.transform_points(m.vertices, T)
                F = np.asarray(m.faces)
                uv = getattr(m.visual, 'uv', None)
                mat = getattr(m.visual, 'material', None)
                mname = str(getattr(mat, 'name', 'mat'))
                C = V[F].mean(1)
                N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
                N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
                self.meshes.append({'file': f, 'mat': mname, 'V': V, 'F': F, 'UV': None if uv is None else np.asarray(uv), 'C': C, 'N': N, 'm': mat})

    def material_spec(self, mesh):
        """Material des Detailmodells fuer unsere GLBs (Textur einmalig nach tex/d_*.jpg exportieren)"""
        name = _san(mesh['mat'])
        if name in self.mats: return name
        m = mesh['m']; spec = {'rough': 0.85}
        tex = getattr(m, 'baseColorTexture', None)
        if tex is not None and mesh['UV'] is not None:
            fn = name + '.jpg'
            if not os.path.exists(OUT + 'tex/' + fn):
                im = tex.convert('RGB')
                if max(im.size) > 1024: im = im.resize((1024, 1024))
                tmp = OUT + 'tex/.' + fn + '.tmp'; im.save(tmp, format='JPEG', quality=88); os.replace(tmp, OUT + 'tex/' + fn)
            spec['tex'] = 'tex/' + fn
        bc = getattr(m, 'baseColorFactor', None)
        if bc is not None:
            bc = np.asarray(bc, float)
            if bc.max() > 1.0: bc = bc / 255.0
            if 'tex' not in spec or not np.allclose(bc[:3], 1.0): spec['color'] = tuple(float(v) for v in bc[:3]) + (1.0,)
        self.mats[name] = spec
        return name

    def select(self, poly, ymin, ymax, files):
        """Masken der zu entfernenden Dreiecke je Mesh (exakt wie apply_landmark_replace.py)"""
        shapely.prepare(poly)
        out = []
        for M in self.meshes:
            if M['file'] not in files: out.append(np.zeros(len(M['F']), bool)); continue
            C = M['C']
            m = (C[:, 1] >= ymin) & (C[:, 1] <= ymax)
            if m.any():
                ii = np.nonzero(m)[0]
                ins = shapely.contains_xy(poly, C[ii, 0], C[ii, 2])
                m[:] = False; m[ii[ins]] = True
            out.append(m)
        return out

def _tris_poly(V, F):
    return [Polygon(V[f][:, [0, 2]]) for f in F]

OLDMAT = ('8F8A82', '2F3133', 'F4F4F2', '1D5FA8', 'EDEDE8', '9C9EA0')

def build_patch_retri(det, F_poly, hole_poly, y_ref, files=GROUND_FILES, maxa=0.6, clip=None, ymin=-8.0, ymax=4.0, exclude=OLDMAT, _only_region=False):
    """F_poly: Entfernungs-Fussabdruck; hole_poly: neue Oeffnung (bleibt frei); y_ref: Strassenniveau (Filter Bodenflaechen).
    Rueckgabe: parts (Liste), rim_edges [(pu,pw,hu,hw,c)], removed counts {file: n}"""
    sel = det.select(F_poly, ymin, ymax, files)
    removed = {}
    rem_tris = []; ref = []          # Referenzflaechen (Boden) fuer Hoehe/UV/Material: (mesh_idx, face_idx)
    keep_tris = []
    zone = F_poly.buffer(4.0); shapely.prepare(zone)
    for mi, (M, s) in enumerate(zip(det.meshes, sel)):
        if s.any(): removed[M['file']] = removed.get(M['file'], 0) + int(s.sum())
        if M['file'] not in files or any(c in M['mat'] for c in exclude): continue
        ground = (M['N'][:, 1] > 0.6) & (M['C'][:, 1] > y_ref - 1.2) & (M['C'][:, 1] < y_ref + 1.0)
        near = shapely.contains_xy(zone, M['C'][:, 0], M['C'][:, 2])
        for fi in np.nonzero(s & ground)[0]: rem_tris.append((mi, fi))
        for fi in np.nonzero(~s & ground & near)[0]: keep_tris.append((mi, fi))
    def polys(lst): return [Polygon(det.meshes[mi]['V'][det.meshes[mi]['F'][fi]][:, [0, 2]]) for mi, fi in lst]
    RP = [p for p in polys(rem_tris) if p.is_valid and p.area > 1e-8]
    KP = [p for p in polys(keep_tris) if p.is_valid and p.area > 1e-8]
    REM = unary_union(RP) if RP else Polygon(); KEEP = unary_union(KP) if KP else Polygon()
    U = F_poly if _only_region else unary_union([REM, F_poly]).difference(KEEP)
    if clip is not None: U = U.intersection(clip)
    U = U.difference(hole_poly)
    U = shapely.make_valid(U)
    U = unary_union([g for g in getattr(U, 'geoms', [U]) if g.geom_type == 'Polygon' and g.area > 0.01])
    if U.is_empty: return [], [], removed
    # ---- Triangulation (keine Steinerpunkte auf dem Rand: 'Y')
    rings = []
    for p in getattr(U, 'geoms', [U]):
        rings.append(np.asarray(p.exterior.coords)[:-1])
        for h in p.interiors: rings.append(np.asarray(h.coords)[:-1])
    # Kanten dichter machen nur an der neuen Oeffnung (Rim) ist nicht noetig; Rand wie er ist
    V0 = []; S0 = []; off = 0
    for r in rings:
        n = len(r); V0.append(r); S0.append(np.c_[np.arange(n), (np.arange(n) + 1) % n] + off); off += n
    # Materialgrenzen der Referenzflaechen als innere Zwangskanten (2 cm vor dem Rand enden -> Rand bleibt ohne Zusatzpunkte):
    # sonst bekommt jedes Patch-Dreieck das Material am Schwerpunkt und die Grenze Fahrbahn/Gehweg wird zackig
    grp = {}
    for (mi, fi) in rem_tris + keep_tris:
        pg = Polygon(det.meshes[mi]['V'][det.meshes[mi]['F'][fi]][:, [0, 2]])
        if not (pg.is_valid and pg.area > 1e-8): continue
        mn = det.material_spec(det.meshes[mi])
        fam = 'asphalt' if 'asphalt' in mn else ('setts' if 'setts' in mn else ('green' if ('grass' in mn or 'green' in mn) else 'slabs'))
        grp.setdefault(fam, []).append(pg)          # nur Belagsfamilien (Fahrbahn/Pflaster/Platten/Gruen), nicht jede Plattenfarbe
    if len(grp) > 1:
        Uin = U.buffer(-0.02)
        ML = unary_union([unary_union(ps).buffer(0.002).buffer(-0.002).boundary for ps in grp.values()]).intersection(Uin)
        ML = shapely.simplify(ML, 0.03)
        for ls in getattr(ML, 'geoms', [ML]):
            if ls.is_empty or ls.geom_type not in ('LineString', 'LinearRing') or ls.length < 0.25: continue
            c = np.asarray(ls.coords)
            n = len(c); V0.append(c); S0.append(np.c_[np.arange(n - 1), np.arange(1, n)] + off); off += n
    V0 = np.concatenate(V0); S0 = np.concatenate(S0)
    key = np.round(V0 * 1e4).astype(np.int64); uk, ui, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V0 = V0[ui]; S0 = inv.reshape(-1)[S0]; S0 = S0[S0[:, 0] != S0[:, 1]]
    holes = []
    for p in getattr(U, 'geoms', [U]):
        for h in p.interiors:
            hp = Polygon(h)
            holes.append(list(hp.representative_point().coords[0]))
    # Punkte ausserhalb U (Konkavitaeten) werden durch 'p' + Loecher entfernt: zusaetzlich Loch-Punkte fuer Luecken
    tri_in = {'vertices': V0, 'segments': S0}
    if holes: tri_in['holes'] = np.array(holes)
    r = tr.triangulate(tri_in, ('pq25Ya%.3fQ' % maxa) if len(grp) <= 1 else ('pq12Ya%.3fQ' % max(maxa, 1.5)))
    V, T = r['vertices'], r['triangles']
    cen = V[T].mean(1)
    inside = shapely.contains_xy(U, cen[:, 0], cen[:, 1])
    T = T[inside]
    # ---- Hoehen / Material / UV aus Referenzflaechen
    allref = rem_tris + keep_tris
    RV = np.array([det.meshes[mi]['V'][det.meshes[mi]['F'][fi]] for mi, fi in allref])        # (k,3,3)
    def locate(pts):
        """Index in allref + baryzentrische Gewichte; -1 wenn keine Referenzflaeche"""
        A = RV[:, 0, [0, 2]]; B = RV[:, 1, [0, 2]]; C = RV[:, 2, [0, 2]]
        res = np.full(len(pts), -1); W = np.zeros((len(pts), 3))
        v0 = B - A; v1 = C - A
        d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); den = d00 * d11 - d01 * d01; den[np.abs(den) < 1e-14] = 1e-14
        for i, p in enumerate(pts):
            v2 = p - A; d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
            wv = (d11 * d20 - d01 * d21) / den; ww = (d00 * d21 - d01 * d20) / den; wu = 1 - wv - ww
            ok = (wu > -1e-6) & (wv > -1e-6) & (ww > -1e-6)
            if ok.any():
                cand = np.nonzero(ok)[0]
                j = cand[np.argmax(RV[cand, :, 1].mean(1))]          # oberste Flaeche
                res[i] = j; W[i] = (wu[j], wv[j], ww[j])
        return res, W
    vi, vw = locate(V)
    H = np.full(len(V), np.nan)
    ok = vi >= 0
    H[ok] = (RV[vi[ok], :, 1] * vw[ok]).sum(1)
    if (~ok).any():
        # in der alten Oeffnung: linear aus Referenz-Ecken (nur Boden nahe Strassenniveau)
        from scipy.interpolate import griddata
        P_ = RV.reshape(-1, 3)
        Hl = griddata(P_[:, [0, 2]], P_[:, 1], V[~ok], method='linear')
        bad = np.isnan(Hl)
        if bad.any(): Hl[bad] = griddata(P_[:, [0, 2]], P_[:, 1], V[~ok][bad], method='nearest')
        H[~ok] = Hl
    # Material je Dreieck: Referenz am Schwerpunkt (sonst naechste)
    cen = V[T].mean(1)
    ti, tw = locate(cen)
    if (ti < 0).any():
        from scipy.spatial import cKDTree
        kd = cKDTree(RV.mean(1)[:, [0, 2]]); _, nn_ = kd.query(cen[ti < 0]); ti[ti < 0] = nn_
    parts = []
    for j in np.unique(ti):
        mi, fi = allref[j]
        M = det.meshes[mi]
        mname = det.material_spec(M)
        tsel = np.nonzero(ti == j)[0]
        # gleiche Material-Gruppen zusammenfassen
        parts.append((mname, mi, tsel))
    # pro Material: UV ueber affine Abbildung (xz -> uv) aus allen Referenzflaechen dieses Meshes
    out = []
    by = {}
    for mname, mi, tsel in parts: by.setdefault((mname, mi), []).append(tsel)
    for (mname, mi), lst in by.items():
        tsel = np.concatenate(lst); M = det.meshes[mi]
        Tm = T[tsel]
        used, inv = np.unique(Tm, return_inverse=True)
        pos = np.c_[V[used, 0], H[used], V[used, 1]]
        if M['UV'] is not None:
            fis = [fi for (m2, fi) in allref if m2 == mi]
            Vr = M['V'][M['F'][fis]].reshape(-1, 3); UVr = M['UV'][M['F'][fis]].reshape(-1, 2)
            X = np.c_[Vr[:, 0], Vr[:, 2], np.ones(len(Vr))]
            A_, *_ = np.linalg.lstsq(X, UVr, rcond=None)
            uv = np.c_[pos[:, 0], pos[:, 2], np.ones(len(pos))] @ A_
            uv[:, 1] = 1 - uv[:, 1]
        else:
            uv = pos[:, [0, 2]]
        p = G.part(mname, pos, inv.reshape(-1, 3), np.tile([0, 1.0, 0], (len(pos), 1)), uv)
        p = G.orient_up(p); p['nrm'] = G.vertex_normals(p['pos'], p['idx'])
        out.append(p)
    # ---- Rim-Kanten an der neuen Oeffnung
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); Et = np.tile(np.arange(len(T)), 3)
    Es = np.sort(E, 1); u_, c_ = np.unique(Es, axis=0, return_counts=True)
    bset = set(map(tuple, u_[c_ == 1]))
    hb = hole_poly.boundary
    rim = []
    for (a, b), t in zip(E, Et):
        if tuple(sorted((a, b))) not in bset: continue
        m = (V[a] + V[b]) / 2
        if hb.distance(Point(m)) > 0.01: continue
        rim.append((V[a].tolist(), V[b].tolist(), float(H[a]), float(H[b]), V[T[t]].mean(0).tolist()))
    return out, rim, removed

def _earcut_poly(pg):
    """Polygon (mit Loechern) -> Ecken (k,2), Dreiecke (m,3); keine Zusatzpunkte"""
    import mapbox_earcut as ec
    rings = [np.asarray(pg.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in pg.interiors]
    V = np.concatenate(rings); ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    T = np.asarray(ec.triangulate_float64(V, ends)).reshape(-1, 3)
    return V, T

def build_patch(det, F_poly, hole_poly, y_ref, files=GROUND_FILES, maxa=0.6, clip=None, ymin=-8.0, ymax=4.0, exclude=OLDMAT):
    """Schnitt-Flicken: jedes entfernte Original-Bodendreieck wird an der neuen Oeffnung (hole_poly) beschnitten und behaelt
    exakt Material, UV und Ebene (baryzentrisch) -> Belagsgrenzen bleiben gerade wie im Detailmodell. Nur Flaechen, die im
    Original nicht belegt waren (alte Trog-Oeffnung), werden neu trianguliert (build_patch_retri-Verfahren, Material vom Nachbarn).
    Rueckgabe wie build_patch_retri: parts, rim_edges [(pu,pw,hu,hw,c)], removed {file: n}"""
    sel = det.select(F_poly, ymin, ymax, files)
    removed = {}
    rem = []
    for mi, (M, s) in enumerate(zip(det.meshes, sel)):
        if s.any(): removed[M['file']] = removed.get(M['file'], 0) + int(s.sum())
        if M['file'] not in files or any(c in M['mat'] for c in exclude): continue
        ground = (M['N'][:, 1] > 0.6) & (M['C'][:, 1] > y_ref - 1.2) & (M['C'][:, 1] < y_ref + 1.0)
        for fi in np.nonzero(s & ground)[0]: rem.append((mi, fi))
    region = hole_poly if clip is None else None
    by = {}                       # mi -> list of (pos (k,3), uv (k,2), tri (m,3))
    covered = []
    for mi, fi in rem:
        M = det.meshes[mi]; f = M['F'][fi]; P3 = M['V'][f]
        tp = Polygon(P3[:, [0, 2]])
        if not tp.is_valid or tp.area < 1e-8: continue
        covered.append(tp)
        pc = tp.difference(hole_poly)
        if clip is not None: pc = pc.intersection(clip)
        pc = shapely.make_valid(pc)
        polys = [g for g in getattr(pc, 'geoms', [pc]) if g.geom_type == 'Polygon' and g.area > 1e-6]
        if not polys: continue
        # baryzentrische Koordinaten im Originaldreieck
        a, b, c = P3[:, [0, 2]]
        Mt = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        if abs(np.linalg.det(Mt)) < 1e-12: continue
        Mi = np.linalg.inv(Mt)
        uvsrc = M['UV'][f] if M['UV'] is not None else None
        for pg in polys:
            if pg.equals_exact(tp, 1e-9) or (abs(pg.area - tp.area) < 1e-9 and len(pg.interiors) == 0):
                V2 = P3[:, [0, 2]]; T2 = np.array([[0, 1, 2]])
            else:
                V2, T2 = _earcut_poly(pg)
            if len(T2) == 0: continue
            w = (V2 - a) @ Mi.T; W3 = np.c_[1 - w.sum(1), w]
            pos = W3 @ P3
            pos[:, [0, 2]] = V2
            uv = (W3 @ uvsrc) if uvsrc is not None else V2.copy()
            if uvsrc is not None: uv = np.c_[uv[:, 0], 1 - uv[:, 1]]
            by.setdefault(mi, []).append((pos, uv, T2))
    out = []
    for mi, lst in by.items():
        M = det.meshes[mi]; mname = det.material_spec(M)
        off = 0; P_ = []; U_ = []; T_ = []
        for pos, uv, T2 in lst:
            P_.append(pos); U_.append(uv); T_.append(T2 + off); off += len(pos)
        p = G.part(mname, np.concatenate(P_), np.concatenate(T_), np.tile([0, 1.0, 0], (off, 1)), np.concatenate(U_))
        p = G.orient_up(p); p['nrm'] = G.vertex_normals(p['pos'], p['idx'])
        out.append(p)
    # ---- Rest: Teile des Fussabdrucks ohne Original-Boden (alte Oeffnung) -> neu triangulieren
    COV = unary_union(covered) if covered else Polygon()
    rest_parts, rest_rim = [], []
    keep_near = []
    zone = F_poly.buffer(4.0); shapely.prepare(zone)
    for mi, (M, s) in enumerate(zip(det.meshes, sel)):
        if M['file'] not in files or any(c in M['mat'] for c in exclude): continue
        ground = (M['N'][:, 1] > 0.6) & (M['C'][:, 1] > y_ref - 1.2) & (M['C'][:, 1] < y_ref + 1.0)
        near = shapely.contains_xy(zone, M['C'][:, 0], M['C'][:, 2])
        for fi in np.nonzero(~s & ground & near)[0]: keep_near.append(Polygon(M['V'][M['F'][fi]][:, [0, 2]]))
    KEEP = unary_union([k for k in keep_near if k.is_valid and k.area > 1e-8]) if keep_near else Polygon()
    R = F_poly.difference(COV).difference(KEEP).difference(hole_poly)
    if clip is not None: R = R.intersection(clip)
    R = shapely.make_valid(R)
    R = [g for g in getattr(R, 'geoms', [R]) if g.geom_type == 'Polygon' and g.area > 0.02]
    if R:
        # bestehendes Verfahren nur fuer den Rest: Fussabdruck = Rest, Original-Dreiecke liefern Hoehe/Material
        rest_parts, rest_rim, _ = build_patch_retri(det, unary_union(R).buffer(0), hole_poly, y_ref, files=files, maxa=maxa,
                                                    clip=unary_union(R).buffer(0), ymin=ymin, ymax=ymax, exclude=exclude, _only_region=True)
    out += rest_parts
    # ---- Rim-Kanten an der neuen Oeffnung (aus allen Flicken-Dreiecken)
    allp = G.merge([dict(p, mat='x') for p in out])
    rim = []
    if allp:
        m = allp[0]; Pp = m['pos']; I = m['idx']
        k = np.round(Pp[:, [0, 2]] * 1e4).astype(np.int64); uq, iv = np.unique(k, axis=0, return_inverse=True); iv = iv.reshape(-1)
        Iq = iv[I]
        Hq = np.zeros(len(uq)); Hq[iv] = Pp[:, 1]
        E = np.concatenate([Iq[:, [0, 1]], Iq[:, [1, 2]], Iq[:, [2, 0]]]); Et = np.tile(np.arange(len(Iq)), 3)
        Es = np.sort(E, 1); u_, inv_, c_ = np.unique(Es, axis=0, return_inverse=True, return_counts=True)
        hb = hole_poly.boundary
        Xq = uq / 1e4
        for e_i in np.nonzero(c_[inv_.reshape(-1)] == 1)[0]:
            a_, b_ = E[e_i]
            mpt = (Xq[a_] + Xq[b_]) / 2
            if hb.distance(Point(mpt)) > 0.01: continue
            cen = Pp[I[Et[e_i]]][:, [0, 2]].mean(0)
            rim.append((Xq[a_].tolist(), Xq[b_].tolist(), float(Hq[a_]), float(Hq[b_]), cen.tolist()))
    return out, rim, removed

def write_replace(key, polys, note, files=('street', 'ground'), ymin=-8.0, ymax=4.0):
    os.makedirs(DATA + 'detail_replace', exist_ok=True)
    fp = []
    for p in polys:
        for g in getattr(p, 'geoms', [p]):
            if g.is_empty: continue
            fp.append([[round(x, 3), round(z, 3)] for x, z in np.asarray(g.exterior.coords)[:-1]])
    J = {'footprints': fp, 'ymin': ymin, 'ymax': ymax, 'files': list(files), 'note': note}
    fn = DATA + 'detail_replace/' + key + '.json'; tmp = fn + '.tmp'
    json.dump(J, open(tmp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False); os.replace(tmp, fn)
    return fn

def footprint_poly(J_fp):
    """Polygon exakt wie im JSON (fuer die Simulation)"""
    return unary_union([Polygon(r) for r in J_fp])

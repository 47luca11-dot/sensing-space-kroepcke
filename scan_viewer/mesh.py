"""Bereinigung + geschlossenes Mesh für ein extrahiertes Splat-Subjekt (Person oder Objekt).

clean_and_mesh(S, sel, info) ->
  keep   : bool-Maske über sel (False = Floater / abgetrennte Fetzen / Riesen-Splats)
  mesh   : dict(V, F, C) in Original-Koordinaten, wasserdicht, Lücken geschlossen
  stats  : dict

Vorgehen (maßstabsunabhängig, Voxelgröße aus der Subjektgröße):
  1. ins "up = +z"-System drehen (Bodennormale aus der Extraktion)
  2. Dichtegitter aus Deckkraft, leicht geglättet, adaptiver Schwellwert
  3. Zusammenhangskomponenten: kleine, abgetrennte Fetzen fallen weg -> Splats darin werden entfernt
  4. Lücken schließen: 3D-Closing, waagerechte Scheiben schließen + füllen, 3D-Füllung
  5. Marching Cubes -> Glätten -> Remesh/Dezimieren -> Löcher schließen -> Farben von den Splats
"""
import numpy as np
import open3d as o3d
import pymeshlab
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
from skimage.morphology import convex_hull_image


def _rot_to_z(n):
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    z = np.array([0, 0, 1.0]); v = np.cross(n, z); s = np.linalg.norm(v); c = n @ z
    if s < 1e-9: return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def clean_and_mesh(S, sel, info, target_faces=300000, res=340, group=None):
    if group is None: group = bool(info.get('params', {}).get('group', False))
    xyz = S['xyz'][sel]; op = S['op'][sel]; rgb = S['rgb'][sel]; smax = S['smax'][sel]
    R = _rot_to_z(info['up'])
    P = xyz @ R.T
    # robuste Ausdehnung (Ausreißer ignorieren) -> Voxelgröße
    lo_r, hi_r = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    vox = float(max((hi_r - lo_r).max() / res, 1e-4))
    # --- 0. Riesen- und Nadel-Splats (typische Artefakte) markieren
    sc = np.exp(np.stack([S['v']['scale_0'], S['v']['scale_1'], S['v']['scale_2']], 1).astype(float))[sel] if 'scale_0' in S['v'].dtype.names else None
    bad = np.zeros(len(sel), bool)
    if sc is not None:
        bad |= smax > 12 * vox
        bad |= (sc.max(1) / np.maximum(sc.min(1), 1e-9) > 60) & (smax > 3 * vox)
    lo = P.min(0) - 6 * vox
    shape = np.ceil((P.max(0) + 6 * vox - lo) / vox).astype(int) + 1
    if np.prod(shape) > 120e6:                                   # Notbremse bei extremen Ausreißern
        vox *= (np.prod(shape) / 120e6) ** (1 / 3); lo = P.min(0) - 6 * vox
        shape = np.ceil((P.max(0) + 6 * vox - lo) / vox).astype(int) + 1
    ijk = np.clip(np.floor((P - lo) / vox).astype(int), 0, shape - 1)
    D = np.zeros(shape, np.float32)
    good = ~bad
    np.add.at(D, tuple(ijk[good].T), op[good].astype(np.float32))
    # Belegung = Voxel mit Splat-Evidenz (auch dünn gescannte Stellen), ergänzt durch das geglättete Dichtefeld
    hit = np.zeros(shape, bool); hit[tuple(ijk[good & (op > 0.12)].T)] = True
    D = ndimage.gaussian_filter(D, 1.2)
    strong = good & (op > 0.3)
    vals = D[tuple(ijk[strong].T)] if strong.any() else D[D > 0]
    occ = (D > float(np.percentile(vals, 8)) * 0.5) | ndimage.binary_dilation(hit, iterations=1)
    st = ndimage.generate_binary_structure(3, 1)
    occ = ndimage.binary_closing(np.pad(occ, 3), st, iterations=2)[3:-3, 3:-3, 3:-3]
    # --- 1. Komponenten: abgetrennte Fetzen entfernen
    def big_components(o, frac):
        lab, _ = ndimage.label(o)
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        return np.isin(lab, np.where(sizes >= max(sizes.max() * frac, 60))[0])
    frac = 0.004 if group else 0.05
    occ = big_components(occ, frac)
    occ_d = ndimage.binary_dilation(occ, st, iterations=2)
    keep = good & occ_d[tuple(ijk.T)]                         # Splats außerhalb = Floater
    # --- 2. Lücken schließen
    height = max(float(info.get('height', 0)), (P[:, 2].max() - P[:, 2].min()))
    zl = int(max(3, round(0.035 * height / vox)))            # senkrechte Lücken bis ~3.5 % der Höhe
    col = np.zeros((1, 1, 2 * zl + 1), bool); col[0, 0, :] = True
    occ = occ | ndimage.binary_closing(np.pad(occ, ((0, 0), (0, 0), (zl, zl))), col)[:, :, zl:-zl]
    occ = occ | ndimage.binary_closing(np.pad(occ, 4), st, iterations=3)[4:-4, 4:-4, 4:-4]
    st2 = ndimage.generate_binary_structure(2, 1)
    rad = int(max(3, round(0.02 * height / vox)))
    for k in range(occ.shape[2]):                              # waagerechte Schnitte: jede Teilfläche -> konvexe Hülle
        sl = occ[:, :, k]
        if not sl.any(): continue
        c = ndimage.binary_closing(np.pad(sl, rad + 1), st2, iterations=rad)[rad + 1:-rad - 1, rad + 1:-rad - 1] | sl
        lab2, _ = ndimage.label(c)
        out = np.zeros_like(c)
        for i, sl_ in enumerate(ndimage.find_objects(lab2), 1):
            comp = lab2[sl_] == i
            out[sl_] |= convex_hull_image(comp) if comp.sum() >= 6 else comp
        occ[:, :, k] = out
    occ = ndimage.binary_fill_holes(occ)
    occ = big_components(occ, frac)
    # --- 2b. logisch schließen: kleine Tunnel/Durchbrüche (Scanlücken) bis ~3.5 % der Höhe zumachen,
    #         größere echte Öffnungen (Fahrradrahmen, Bögen, Fenster) bleiben. Closing über Distanztransformation.
    def genus_of(o):
        Vg, Fg, _, _ = measure.marching_cubes(ndimage.gaussian_filter(np.pad(o, 3).astype(np.float32), 1.2), 0.5)
        mg = pymeshlab.MeshSet(); mg.add_mesh(pymeshlab.Mesh(Vg, Fg)); mg.meshing_remove_duplicate_vertices()
        return mg.get_topological_measures().get('genus') or 0
    for fr in (0.035, 0.05, 0.065, 0.08):                      # schrittweise, bis keine Tunnel mehr (große Öffnungen bleiben)
        rc = max(2.0, fr * height / vox); pad = int(rc) + 2
        o = np.pad(occ, pad)
        clo = ndimage.distance_transform_edt(ndimage.distance_transform_edt(~o) <= rc) > rc
        occ = ndimage.binary_fill_holes(clo[pad:-pad, pad:-pad, pad:-pad] | occ)
        occ = big_components(occ, frac)
        g = genus_of(occ)
        if g == 0: break
    # --- 3. Oberfläche
    F = ndimage.gaussian_filter(occ.astype(np.float32), 1.2)
    V, Fc, _, _ = measure.marching_cubes(F, 0.5, spacing=(vox, vox, vox))
    V += lo
    Fc = Fc[:, ::-1].copy()                                    # Normalen nach außen
    ms = pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(V, Fc))
    ms.meshing_remove_duplicate_vertices()
    ms.apply_coord_taubin_smoothing(stepsmoothnum=20)
    if ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservenormal=True, preservetopology=True, qualitythr=0.4)
    for _ in range(2):
        ms.meshing_repair_non_manifold_edges(); ms.meshing_repair_non_manifold_vertices()
        ms.meshing_close_holes(maxholesize=100000)
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=50)
    tm = ms.get_topological_measures()
    m = ms.current_mesh(); V = m.vertex_matrix(); Fc = m.face_matrix()
    # --- 4. Farben: deckende, behaltene Splats, gewichteter Mittelwert ohne Ausreißer
    src = keep & (op > 0.25)
    if src.sum() < 10: src = keep
    t = cKDTree(P[src]); Cs, Os = rgb[src], op[src]
    dd, ii = t.query(V, k=12)
    w = Os[ii] * np.exp(-(dd / (2.5 * vox)) ** 2) + 1e-6
    lum = Cs[ii].mean(2); med = np.median(lum, 1, keepdims=True)
    w *= np.abs(lum - med) < 0.25                               # einzelne helle/dunkle Ausreißer ignorieren
    w += 1e-9
    C = (Cs[ii] * w[..., None]).sum(1) / w.sum(1)[:, None]
    Vw = V @ R                                                  # zurück ins Original-Koordinatensystem
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vw), o3d.utility.Vector3iVector(Fc))
    stats = dict(voxel=vox, faces=int(len(Fc)), vertices=int(len(V)), boundary_edges=int(tm['boundary_edges']),
                 components=int(tm['connected_components_number']), watertight=bool(o.is_watertight()),
                 removed_splats=int((~keep).sum()), kept_splats=int(keep.sum()), giant_or_needle=int(bad.sum()))
    return keep, dict(V=Vw, F=Fc, C=np.clip(C, 0, 1)), stats


def save_mesh(mesh, path):
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh['V']), o3d.utility.Vector3iVector(mesh['F']))
    o.vertex_colors = o3d.utility.Vector3dVector(mesh['C']); o.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, o, write_ascii=False)

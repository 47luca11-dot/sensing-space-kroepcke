"""Komplette Pipeline für eine Scan-Datei: extrahieren -> bereinigen -> Mesh mit geschlossenen Lücken.

python process.py in.ply out_prefix [--group] [--radius R] [--no-mesh]
schreibt  <out_prefix>_extrahiert.ply  (bereinigte Splats, alle Original-Attribute)
          <out_prefix>_mesh.ply        (wasserdichtes Mesh mit Vertexfarben)
          <out_prefix>_extrahiert.json (Kennzahlen)
"""
import argparse, json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract as X
import mesh as M


def run(S, prefix, radius=None, group=None, do_clean=True, do_mesh=True, min_opacity=None, keep_soft=None):
    t0 = time.time()
    sel, info = X.extract(S, radius=radius, group=group, min_opacity=min_opacity, keep_soft=keep_soft)
    info['extracted_raw'] = int(len(sel))
    if do_clean or do_mesh:
        keep, mesh, ms = M.clean_and_mesh(S, sel, info, group=bool(info["params"]["group"]))
        if do_clean:
            sel = sel[keep]
        info['mesh'] = ms if do_mesh else None
        if do_mesh:
            M.save_mesh(mesh, prefix + '_mesh.ply')
    info['selected'] = int(len(sel)); info['cleaned'] = bool(do_clean)
    xs = S['xyz'][sel]; info['bbox_min'] = xs.min(0).tolist(); info['bbox_max'] = xs.max(0).tolist()
    X.save(S, sel, prefix + '_extrahiert.ply')
    info['seconds'] = round(time.time() - t0, 2)
    json.dump(info, open(prefix + '_extrahiert.json', 'w'), indent=1)
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('prefix')
    ap.add_argument('--group', action='store_true'); ap.add_argument('--radius', type=float)
    ap.add_argument('--no-mesh', action='store_true')
    a = ap.parse_args()
    info = run(X.load(a.inp), a.prefix, radius=a.radius, group=True if a.group else None, do_mesh=not a.no_mesh)
    m = info.get('mesh') or {}
    print(json.dumps(dict(splats=f"{info['selected']}/{info['total']}", raw=info['extracted_raw'], height=round(info['height'], 3),
                          s=info['seconds'], faces=m.get('faces'), watertight=m.get('watertight'), boundary=m.get('boundary_edges'),
                          comps=m.get('components'), removed=m.get('removed_splats'))))

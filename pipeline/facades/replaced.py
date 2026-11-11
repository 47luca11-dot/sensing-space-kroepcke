"""LoD2-Gebaeude, die durch handgebaute Wahrzeichen ersetzt werden (data/landmarks_replace/*.json):
Gebaeude-ID in "lod2_ids" oder Grundriss-Schwerpunkt innerhalb eines "footprints"-Polygons (Spielkoordinaten [x, z])."""
import os, glob, json
from shapely.geometry import Polygon, Point
from common import DATA


def load():
    ids = set(); polys = []; names = {}
    for fn in sorted(glob.glob(os.path.join(DATA, 'landmarks_replace', '*.json'))):
        try:
            J = json.load(open(fn, encoding='utf-8'))
        except Exception as e:                      # Datei wird evtl. gerade geschrieben
            print('landmarks_replace unlesbar:', fn, e); continue
        nm = os.path.splitext(os.path.basename(fn))[0]
        for i in J.get('lod2_ids', []) or []:
            ids.add(i); names[i] = nm
        for fp in J.get('footprints', []) or []:
            try:
                g = Polygon(fp).buffer(0)
            except Exception:
                continue
            if not g.is_empty:
                polys.append((g, nm))
    return ids, polys, names


def mark(bldgs, verbose=True):
    ids, polys, names = load()
    out = {}
    for b in bldgs:
        hit = names.get(b.id) if b.id in ids else None
        if hit is None:
            p = Point(b.cx, b.cz)
            for g, nm in polys:
                if g.contains(p):
                    hit = nm; break
        b.replaced = hit
        if hit:
            out[b.id] = hit
    if verbose:
        import collections
        print('durch Wahrzeichen ersetzt:', len(out), dict(collections.Counter(out.values())))
    return out

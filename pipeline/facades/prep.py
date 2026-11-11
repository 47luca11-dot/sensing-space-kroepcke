"""Daten vorbereiten: LoD2-Gebaeude (Spielkoordinaten), Abdeckung durch das Detailmodell, LGLN-/OSM-Attribute,
Gelaendehoehen (DGM1), Dachfarben (TrueDOP10), Volumen-Index fuer Nachbar-/Brandwand-Erkennung, Strassen-Index.
Ergebnis: cache/prep.pkl (wird von build_all.py benutzt)."""
import os, glob, json, pickle, math, collections
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box as sbox
from shapely.strtree import STRtree
from shapely import contains_xy, points as spoints, make_valid
from PIL import Image
from common import DATA, CACHE, RESEARCH, Y_OFF, E0, N0, RHINO_OFF, to_game, load_json

Image.MAX_IMAGE_PIXELS = None


def newell(P):
    a = np.asarray(P, float); b = np.roll(a, -1, 0)
    return np.array([((a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2])).sum(), ((a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0])).sum(),
                     ((a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1])).sum()])


def clean_ring(P, eps=1e-3):
    """doppelte/kollineare Punkte entfernen (3D)."""
    P = [np.asarray(p, float) for p in P]
    out = []
    for p in P:
        if not out or np.linalg.norm(p - out[-1]) > eps:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= eps:
        out.pop()
    changed = True
    while changed and len(out) >= 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            u, v = b - a, c - b
            if np.linalg.norm(np.cross(u, v)) < 1e-4 * max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9) and np.dot(u, v) > 0:
                out.pop(i); changed = True; break
    return np.array(out)


def _snap_points(P, tol):
    """Punkte innerhalb tol zu Clustern vereinigen (Union-Find) und durch den Clustermittelpunkt ersetzen."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    pairs = cKDTree(P).query_pairs(tol, output_type='ndarray')
    if len(pairs) == 0:
        return P.copy()
    par = np.arange(len(P))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(P))])
    out = P.copy()
    for r in np.unique(roots[np.nonzero(roots != np.arange(len(P)))[0]]):
        m = np.nonzero(roots == r)[0]
        c = P[m].mean(0)
        if np.linalg.norm(P[m] - c, axis=1).max() > tol and tol > 0.012:
            # Kette (z.B. fein unterteilte Rundung): nicht zu einem Punkt zusammenziehen, feiner verschweissen
            out[m] = _snap_points(P[m], tol / 2)
        else:
            out[m] = c
    return out


class Surf:
    __slots__ = ('P', 'holes', 'n', 'area', 'kind')

    def __init__(self, P, holes=(), kind='wall'):
        self.P = P; self.holes = [h for h in holes if len(h) >= 3]; self.kind = kind
        n = newell(P)
        for h in self.holes:
            n = n + newell(h)
        a = np.linalg.norm(n)
        self.area = a / 2; self.n = n / a if a > 0 else np.array([0, 1.0, 0])


class Bldg:
    def __init__(self, rec, idx):
        self.id = rec['id']; self.idx = idx
        self.height = rec.get('height'); self.storeys = rec.get('storeys')
        self.street = rec.get('street'); self.number = rec.get('number')
        self.walls = []; self.roofs = []; self.grounds = []
        # Alle Ringe sammeln und Eckpunkte innerhalb 3 cm verschweissen (LoD2-Waende/Daecher weichen teils um mm ab)
        raw = []
        for k in ('walls', 'roofs', 'grounds'):
            for poly in rec[k]:
                raw.append((k, to_game(poly['ext']), [to_game(h) for h in poly.get('holes', [])]))
        allp = np.concatenate([np.concatenate([e] + h) if h else e for _, e, h in raw])
        rep = _snap_points(allp, 0.03)
        o = 0
        snapped = []
        for k, e, hs in raw:
            e2 = rep[o:o + len(e)]; o += len(e)
            h2 = []
            for h in hs:
                h2.append(rep[o:o + len(h)]); o += len(h)
            snapped.append((k, e2, h2))
        for k, e, hs in snapped:
            dst = {'walls': self.walls, 'roofs': self.roofs, 'grounds': self.grounds}[k]
            if True:
                P = clean_ring(e)
                if len(P) < 3:
                    continue
                H = [clean_ring(h) for h in hs]
                s = Surf(P, H, k[:-1])
                if s.area > 1e-3:
                    dst.append(s)
        self.y_ground = min(float(g.P[:, 1].min()) for g in self.grounds) if self.grounds else float(min(w.P[:, 1].min() for w in self.walls))
        fp = [Polygon(g.P[:, [0, 2]], [h[:, [0, 2]] for h in g.holes]) for g in self.grounds]
        fp = [make_valid(p) for p in fp]
        u = fp[0]
        for p in fp[1:]:
            u = u.union(p)
        self.footprint = u.buffer(0)
        c = self.footprint.centroid
        self.cx, self.cz = c.x, c.y
        self.y_top = max([r.P[:, 1].max() for r in self.roofs] + [w.P[:, 1].max() for w in self.walls])
        self.covered = False
        self.attrs = {}


# ---------------------------------------------------------------- Volumen-Index
class VolumeIndex:
    """2D-Projektionen aller Dachflaechen mit Ebenengleichung -> Punkt-im-Volumen-Tests (fuer Brandwand/Hof-Erkennung)."""

    def __init__(self, bldgs):
        geoms, owner, plane, ybot, ytop = [], [], [], [], []
        for b in bldgs:
            for r in b.roofs:
                if r.n[1] < 0.08:
                    continue
                try:
                    g = Polygon(r.P[:, [0, 2]], [h[:, [0, 2]] for h in r.holes]).buffer(0)
                except Exception:
                    continue
                if g.is_empty or g.area < 0.05:
                    continue
                n = r.n; p0 = r.P[0]
                # y = p0.y - (n.x (x-p0.x) + n.z (z-p0.z)) / n.y
                geoms.append(g); owner.append(b.idx)
                plane.append((p0[0], p0[1], p0[2], n[0] / n[1], n[2] / n[1])); ybot.append(min(b.y_ground, getattr(b, 'y_base', b.y_ground)) - 3.0); ytop.append(float(r.P[:, 1].max()))
        self.geoms = geoms; self.owner = np.array(owner); self.plane = np.array(plane); self.ybot = np.array(ybot); self.ytop = np.array(ytop)
        self.tree = STRtree(geoms)

    def top_at(self, gi, x, z):
        p = self.plane[gi]
        return p[:, 1] - (p[:, 3] * (x - p[:, 0]) + p[:, 4] * (z - p[:, 2]))

    def query(self, P):
        """P: (N,3). Rueckgabe: Liste (punkt_idx, owner) fuer alle Volumen, die den Punkt enthalten."""
        P = np.asarray(P, float)
        if len(P) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        pts = spoints(P[:, 0], P[:, 2])
        pi, gi = self.tree.query(pts, predicate='intersects')
        if len(pi) == 0:
            return pi, gi
        top = self.top_at(gi, P[pi, 0], P[pi, 2])
        ok = (P[pi, 1] <= top + 0.02) & (P[pi, 1] >= self.ybot[gi])
        return pi[ok], self.owner[gi[ok]]

    def blocked(self, P, own):
        """-> (blocked_by_other[N], blocked_by_own[N])"""
        P = np.asarray(P, float)
        bo = np.zeros(len(P), bool); bs = np.zeros(len(P), bool)
        pi, ow = self.query(P)
        if len(pi):
            bs[pi[ow == own]] = True; bo[pi[ow != own]] = True
        return bo, bs


# ---------------------------------------------------------------- Gelaende (DGM1)
class Terrain:
    def __init__(self):
        self.tiles = []
        for fn in glob.glob(os.path.join(RESEARCH, 'dgm1', 'dgm1_32_*.tif')):
            im = Image.open(fn); a = np.array(im, np.float32)
            tp = im.tag_v2.get(33922); e0, n0 = tp[3], tp[4]
            self.tiles.append((e0, n0, a))

    def sample(self, x, z):
        """Spiel-x,z -> Spiel-y des Gelaendes (nan ausserhalb)."""
        x = np.atleast_1d(np.asarray(x, float)); z = np.atleast_1d(np.asarray(z, float))
        E = x + E0; N = -z + N0
        out = np.full(len(x), np.nan)
        for e0, n0, a in self.tiles:
            c = np.floor(E - e0).astype(int); r = np.floor(n0 - N).astype(int)
            m = (c >= 0) & (c < a.shape[1]) & (r >= 0) & (r < a.shape[0])
            out[m] = a[r[m], c[m]] - Y_OFF
        return out


# ---------------------------------------------------------------- Dachfarben (TrueDOP10)
class Ortho:
    def __init__(self, red=4):
        self.tiles = []; self.res = 0.1 * red
        for fn in glob.glob(os.path.join(RESEARCH, 'dop10', 'dop10rgb_32_*.tif')):
            cache = os.path.join(CACHE, os.path.basename(fn) + '.r%d.npy' % red)
            name = os.path.basename(fn).split('_')
            e0 = float(name[2]) * 1000; n1 = (float(name[3]) + 1) * 1000
            if os.path.exists(cache):
                a = np.load(cache)
            else:
                im = Image.open(fn); im.draft('RGB', (im.size[0] // red, im.size[1] // red))
                im = im.convert('RGB')
                if im.size[0] != 10000 // red:
                    im = im.resize((10000 // red, 10000 // red), Image.BOX)
                a = np.array(im, np.uint8); np.save(cache, a)
            self.tiles.append((e0, n1, a))

    def median_in(self, poly_xz, erode=0.4, max_pts=4000):
        g = poly_xz.buffer(-erode)
        if g.is_empty or g.area < 0.5:
            g = poly_xz
        x0, z0, x1, z1 = g.bounds
        step = max(self.res, math.sqrt(max(g.area, 1e-6) / max_pts))
        xs, zs = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(z0 + step / 2, z1, step))
        xs = xs.ravel(); zs = zs.ravel()
        m = contains_xy(g, xs, zs)
        xs, zs = xs[m], zs[m]
        if len(xs) < 3:
            return None
        E = xs + E0; N = -zs + N0
        cols = []
        for e0, n1, a in self.tiles:
            c = np.floor((E - e0) / self.res).astype(int); r = np.floor((n1 - N) / self.res).astype(int)
            k = (c >= 0) & (c < a.shape[1]) & (r >= 0) & (r < a.shape[0])
            if k.any():
                cols.append(a[r[k], c[k]])
        if not cols:
            return None
        C = np.concatenate(cols).astype(float)
        if len(C) < 3:
            return None
        return np.median(C, 0), np.percentile(C.mean(1), 85) - np.percentile(C.mean(1), 15)


# ---------------------------------------------------------------- Teilabdeckung durch das Detailmodell
def clip_partial(b, mask, mres=1.0, mx0=-400.0, mz0=-400.0):
    """Gebaeude, das teilweise im Detailmodell liegt, an der Trennwand des Detailmodells abschneiden.
    Rueckgabe True wenn geschnitten."""
    import trimesh
    fp = b.footprint
    x0, z0, x1, z1 = fp.bounds
    xs, zs = np.meshgrid(np.arange(np.floor(x0) + 0.5, x1, 1.0), np.arange(np.floor(z0) + 0.5, z1, 1.0))
    xs = xs.ravel(); zs = zs.ravel()
    ins = contains_xy(fp.buffer(-0.6), xs, zs); xs, zs = xs[ins], zs[ins]
    if len(xs) < 30:
        return False
    c = np.floor((xs - mx0) / mres).astype(int); r = np.floor((zs - mz0) / mres).astype(int)
    inr = (c >= 0) & (c < mask.shape[1]) & (r >= 0) & (r < mask.shape[0])
    m = np.zeros(len(xs), bool); m[inr] = mask[r[inr], c[inr]]
    if m.mean() < 0.15 or m.mean() > 0.85:
        return False
    # Rand zwischen Masken- und Nicht-Masken-Zellen innerhalb des Grundrisses
    S = set(zip(r[m], c[m])); F = set(zip(r[~m], c[~m]))
    edge = [(rr, cc) for (rr, cc) in S if any((rr + dr, cc + dc) in F for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    E = np.array([[cc * mres + mx0 + 0.5, rr * mres + mz0 + 0.5] for rr, cc in edge])
    if len(E) < 5:
        return False
    ctr = E.mean(0); U, Sv, Vt = np.linalg.svd(E - ctr); d = Vt[0]; nrm = np.array([-d[1], d[0]])
    keep_c = np.array([xs[~m].mean(), zs[~m].mean()])
    if np.dot(keep_c - ctr, nrm) < 0:
        nrm = -nrm
    # exakte Lage der Trennwand aus den Rhino-Waenden: parallele Dreiecke nahe der Linie
    wm = trimesh.load(os.path.join(DATA, 'rhino_export', 'bldg_walls.glb'), force='scene').to_geometry()
    V = wm.vertices + RHINO_OFF; T = V[wm.faces]; fc = T.mean(1); fn = wm.face_normals; fa = wm.area_faces
    n3 = np.array([nrm[0], 0, nrm[1]])
    near = (np.abs((fc[:, [0, 2]] - ctr) @ nrm) < 2.5) & (np.abs(fn @ n3) > 0.97) & \
           (np.abs((fc[:, [0, 2]] - ctr) @ d) < np.ptp((E - ctr) @ d) / 2 + 1)
    off = 0.0
    if near.sum() > 0:
        o = (fc[near][:, [0, 2]] - ctr) @ nrm
        srt = np.argsort(o); w = fa[near][srt]; cw = np.cumsum(w)
        off = float(o[srt][np.searchsorted(cw, cw[-1] / 2)])
    p0 = ctr + nrm * off                                    # Punkt auf der Schnittebene
    keep = lambda P: (P[:, [0, 2]] - p0) @ nrm              # >= 0 behalten

    def clip(P):
        out = []; s = keep(P)
        for i in range(len(P)):
            a, b_ = P[i], P[(i + 1) % len(P)]; sa, sb = s[i], s[(i + 1) % len(P)]
            if sa >= 0:
                out.append(a)
            if (sa >= 0) != (sb >= 0):
                t = sa / (sa - sb); out.append(a + t * (b_ - a))
        return np.array(out) if len(out) >= 3 else None
    cut_edges = []
    for lst in (b.walls, b.roofs, b.grounds):
        new = []
        for s_ in lst:
            P = clip(s_.P)
            if P is None:
                continue
            P = clean_ring(P)
            if len(P) < 3:
                continue
            H = [h for h in (clip(h) for h in s_.holes) if h is not None]
            ns = Surf(P, H, s_.kind)
            if ns.area > 1e-3:
                new.append(ns)
                if lst is b.roofs:
                    k = np.abs(keep(P)) < 1e-6
                    for i in range(len(P)):
                        if k[i] and k[(i + 1) % len(P)]:
                            cut_edges.append((P[i].copy(), P[(i + 1) % len(P)].copy()))
        lst[:] = new
    # Abschlusswand (Trennwand zum Detailmodell): Streifen vom Boden bis zur Dachkante
    for a, c_ in cut_edges:
        Q = np.array([a, c_, [c_[0], b.y_ground, c_[2]], [a[0], b.y_ground, a[2]]])
        s_ = Surf(Q, (), 'wall')
        if s_.area < 1e-3:
            continue
        if np.dot(s_.n[[0, 2]], nrm) > 0:          # Normale muss zum Detailmodell (weg vom behaltenen Teil) zeigen
            s_ = Surf(Q[::-1].copy(), (), 'wall')
        b.walls.append(s_)
    fp = [Polygon(g.P[:, [0, 2]]) for g in b.grounds]
    b.footprint = fp[0].buffer(0)
    for p in fp[1:]:
        b.footprint = b.footprint.union(p.buffer(0))
    c = b.footprint.centroid; b.cx, b.cz = c.x, c.y
    b.attrs['clipped'] = {'p0': p0.tolist(), 'n': nrm.tolist()}
    return True


# ---------------------------------------------------------------- Hauptfunktion
def prepare(verbose=True):
    L = load_json(os.path.join(DATA, 'lod2_buildings.json'))
    covered = set(load_json(os.path.join(DATA, 'detail_covered.json'))['ids'])
    mask = np.load(os.path.join(DATA, 'detail_mask.npy'))
    bl = [Bldg(r, i) for i, r in enumerate(L['buildings'])]
    for b in bl:
        b.covered = b.id in covered
    nclip = 0
    for b in bl:
        if not b.covered and clip_partial(b, mask):
            nclip += 1
            if verbose: print('  teilweise im Detailmodell -> geschnitten:', b.id, b.street, b.number)
    # --- LGLN-Attribute
    lg = load_json(os.path.join(CACHE, 'lgln_parts.json'))['parts']
    lgeo = []; lrec = []
    for p in lg:
        for g in p['grounds']:
            try:
                poly = Polygon([(x, -y) for x, y in g]).buffer(0)       # dE,dN -> x,z
            except Exception:
                continue
            if poly.area > 0.5:
                lgeo.append(poly); lrec.append(p)
    ltree = STRtree(lgeo)
    # --- OSM
    osm = load_json(os.path.join(DATA, 'osm_features.json'))
    ob_geo, ob_tags = [], []
    for w in osm['ways']:
        t = w['tags']
        if ('building' in t or 'building:part' in t) and len(w['pts']) >= 4:
            try:
                poly = Polygon(w['pts']).buffer(0)
            except Exception:
                continue
            if poly.area > 1:
                ob_geo.append(poly); ob_tags.append(t)
    otree = STRtree(ob_geo)
    terr = Terrain()
    ortho = Ortho()
    for b in bl:
        a = b.attrs
        # LGLN: Funktion des Gebaeudes mit groesster Ueberlappung, Dachtyp/Traufe je Teil
        best = collections.defaultdict(float); parts = []
        for j in ltree.query(b.footprint, predicate='intersects'):
            ov = b.footprint.intersection(lgeo[j]).area
            if ov > 0.25 * min(lgeo[j].area, b.footprint.area) or ov > 20:
                best[lrec[j]['function']] += ov; parts.append((ov, lrec[j]))
        a['function'] = max(best, key=best.get) if best else None
        parts.sort(key=lambda t: -t[0])
        a['lgln_parts'] = [{'ov': round(ov, 1), 'roofType': p['roofType'], 'roofName': p['roofName'], 'eaves': p['eaves'],
                            'ridge': p['ridge'], 'height': p['height'], 'pitch': p['pitch']} for ov, p in parts[:6]]
        # OSM: Geschosse/Dachform/Material der Gebaeude-Wege mit Ueberlappung
        lv = []; tags = collections.Counter(); bt = collections.Counter()
        for j in otree.query(b.footprint, predicate='intersects'):
            ov = b.footprint.intersection(ob_geo[j]).area
            if ov < 0.3 * min(ob_geo[j].area, b.footprint.area):
                continue
            t = ob_tags[j]
            try:
                if 'building:levels' in t:
                    lv.append((ov, float(t['building:levels']), float(t.get('building:min_level', 0) or 0), float(t.get('roof:levels', 0) or 0)))
            except ValueError:
                pass
            for k in ('roof:shape', 'building:material', 'roof:material', 'roof:colour', 'building:colour', 'start_date', 'name', 'shop', 'amenity',
                      'building:architecture', 'heritage', 'historic', 'addr:street', 'addr:housenumber', 'wikipedia'):
                if k in t:
                    tags[(k, t[k])] += ov
            if 'building' in t:
                bt[t['building']] += ov
        if lv:
            lv.sort(key=lambda t: -t[0])
            a['osm_levels'] = lv[0][1] + lv[0][2]; a['osm_roof_levels'] = lv[0][3]
            a['osm_levels_max'] = max(l[1] + l[2] for l in lv)
        best_tag = {}
        for (k, v), ov in tags.items():
            if k not in best_tag or ov > best_tag[k][1]:
                best_tag[k] = (v, ov)
        a['osm'] = {k: v for k, (v, ov) in best_tag.items()}
        a['osm_building'] = bt.most_common(1)[0][0] if bt else None
        # Gelaende: tiefster Punkt unter dem Umring -> Wandfuss
        ext = np.array(b.footprint.exterior.coords) if b.footprint.geom_type == 'Polygon' else \
            np.concatenate([np.array(g.exterior.coords) for g in b.footprint.geoms])
        seg = []
        for i in range(len(ext) - 1):
            L_ = np.linalg.norm(ext[i + 1] - ext[i]); k = max(1, int(L_ / 1.0))
            seg.append(ext[i] + (ext[i + 1] - ext[i]) * np.linspace(0, 1, k + 1)[:, None])
        Q = np.concatenate(seg)
        # 0.4 m nach aussen versetzt (Gehweg vor der Wand)
        outward = []
        for q in Q:
            pt = Point(q)
            outward.append(q)
        tz = terr.sample(Q[:, 0], Q[:, 1])
        tz = tz[np.isfinite(tz)]
        a['terrain_min'] = float(tz.min()) if len(tz) else b.y_ground
        a['terrain_max'] = float(tz.max()) if len(tz) else b.y_ground
        b.y_base = min(b.y_ground, a['terrain_min']) - 0.3
        # Dachfarben je Dachflaeche
        cols = []
        for r in b.roofs:
            if r.n[1] < 0.08:
                cols.append(None); continue
            try:
                g = Polygon(r.P[:, [0, 2]]).buffer(0)
                mc = ortho.median_in(g) if g.area > 2 else None
            except Exception:
                mc = None
            cols.append(None if mc is None else [float(v) for v in mc[0]] + [float(mc[1])])
        a['roof_dop'] = cols
    vol = VolumeIndex(bl)
    # Strassen (benannte Wege) fuer Ladenzonen
    streets = []
    for w in osm['ways']:
        t = w['tags']
        if t.get('highway') in ('pedestrian', 'primary', 'secondary', 'tertiary', 'residential', 'living_street', 'unclassified',
                                'service', 'footway', 'primary_link', 'secondary_link') and len(w['pts']) >= 2:
            streets.append({'name': t.get('name'), 'hw': t['highway'], 'pts': w['pts'], 'area': t.get('area') == 'yes'})
    for w in osm['ways']:
        t = w['tags']
        if 'area:highway' in t and t['area:highway'] == 'pedestrian' and len(w['pts']) >= 2:
            streets.append({'name': t.get('name'), 'hw': 'pedestrian', 'pts': w['pts'], 'area': True})
    shops = []
    for n in osm['nodes']:
        t = n['tags']
        if 'shop' in t or t.get('amenity') in ('cafe', 'restaurant', 'fast_food', 'bar', 'pub', 'pharmacy', 'bank', 'ice_cream'):
            shops.append({'p': n['p'], 'kind': t.get('shop') or t.get('amenity'), 'name': t.get('name')})
    out = {'bldgs': bl, 'mask': mask, 'streets': streets, 'shops': shops, 'nclip': nclip}
    with open(os.path.join(CACHE, 'prep.pkl'), 'wb') as f:
        pickle.dump(out, f)
    if verbose:
        print('Gebaeude', len(bl), 'davon im Detailmodell', sum(b.covered for b in bl), 'geschnitten', nclip,
              'Volumen', len(vol.geoms), 'Strassen', len(streets), 'Laeden', len(shops))
    return out


def load_prep():
    with open(os.path.join(CACHE, 'prep.pkl'), 'rb') as f:
        d = pickle.load(f)
    d['vol'] = VolumeIndex(d['bldgs'])
    return d


if __name__ == '__main__':
    import time
    import prep as _p                       # Klassen als prep.Bldg pickeln
    t = time.time(); _p.prepare(); print('%.1f s' % (time.time() - t))

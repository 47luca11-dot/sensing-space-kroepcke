"""Leine-Bruecken als zusammenhaengende Ueberbauten (QA 2026-11-10: Goethebruecke hatte zwischen den Fahrbahnplatten keinen
Ueberbau fuer die Stadtbahngleise -> 5 x 20 m Loch bis zum Wasser, Schienen auf dem Wasserspiegel).

Ein Ueberbau = OSM man_made=bridge-Polygon (falls vorhanden) mit allen Brueckenwegen darauf (highway + railway light_rail/tram,
bridge=yes; Gleise ohne layer-Tag zaehlen mit), sonst Wegpuffer (Cluster sich ueberlappender Wege).  Je Ueberbau ein lokaler
Rahmen (s laengs = mittlere Wegrichtung, t quer); Deckhoehe linear zwischen den Uferhoehen auf der Achse; Deckflaeche je
Querlinie t von Ufer zu Ufer (Enden folgen schraegen Ufern), seitlich begrenzt durch das Polygon bzw. die Wegpuffer.
Benutzt von structures.py (Schienen auf der Bruecke aussparen), transit.py (Fahrleitung/Masten), altstadt.py (Ueberbau bauen).
Ergebnis wird in data/ground_cache/bridges.pkl zwischengespeichert (neu berechnet, wenn layout.pkl neuer ist)."""
import os, pickle, math
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from common import *
from geom2 import Frame

CACHE_F = CACHE + 'bridges.pkl'
ROAD_HW = ('primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'service', 'living_street', 'trunk', 'primary_link',
           'secondary_link', 'tertiary_link')


def _layer(t):
    try: return float(str(t.get('layer', '1' if t.get('bridge') else '0')).split(';')[0])
    except ValueError: return 0.0


def compute(L, SURF, osm):
    RIV = L['classes'].get('river')
    if RIV is None or RIV.is_empty: return []
    shapely.prepare(RIV)
    RB = RIV.buffer(1.0)
    # ---- Wege
    ways = []
    by_id = {}
    for b in L.get('bridges', []):
        by_id[b['id']] = b
    for w in osm['ways']:
        t = w['tags']
        if not t.get('bridge') or t.get('bridge') == 'no' or _layer(t) < 1 or len(w['pts']) < 2: continue
        hw = t.get('highway'); rw = t.get('railway')
        if hw is None and rw not in ('tram', 'light_rail'): continue
        Ls = LineString(w['pts'])
        if not Ls.intersects(RB): continue
        if rw in ('tram', 'light_rail'):
            kind = 'rail'; wd = 3.2; half = 1.6
        elif hw in ROAD_HW:
            kind = 'road'; wd = float(by_id[w['id']]['width']) if w['id'] in by_id else 6.0; half = wd / 2 + 2.2
        else:
            kind = 'path'; wd = float(by_id[w['id']]['width']) if w['id'] in by_id else 3.0; half = wd / 2 + 0.35
        ways.append({'id': w['id'], 'line': Ls, 'kind': kind, 'w': wd, 'half': half, 'tags': t,
                     'buf': Ls.buffer(half, cap_style=2, join_style=2)})
    # ---- man_made=bridge-Polygone
    mm = []
    for w in osm['ways']:
        t = w['tags']
        if t.get('man_made') != 'bridge' or len(w['pts']) < 4: continue
        P = Polygon(w['pts']).buffer(0)
        if P.is_empty or not P.intersects(RIV): continue
        mm.append({'id': w['id'], 'poly': P, 'name': t.get('name')})
    clusters = []
    used = set()
    for m in mm:
        ws = [w for w in ways if w['line'].intersection(m['poly']).length > 2.0]
        for w in ws: used.add(w['id'])
        clusters.append({'poly': m['poly'], 'ways': ws, 'name': m['name'] or next((w['tags'].get('name') for w in ws if w['tags'].get('name')), None), 'mm': m['id']})
    rest = [w for w in ways if w['id'] not in used]
    # Union-Find ueber ueberlappende Puffer
    par = list(range(len(rest)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            if rest[i]['buf'].intersects(rest[j]['buf']): par[f(i)] = f(j)
    groups = {}
    for i in range(len(rest)): groups.setdefault(f(i), []).append(rest[i])
    for ws in groups.values():
        clusters.append({'poly': unary_union([w['buf'] for w in ws]), 'ways': ws, 'name': next((w['tags'].get('name') for w in ws if w['tags'].get('name')), None), 'mm': None})
    decks = []
    for c in clusters:
        d = _deck(c, RIV, SURF)
        if d is not None: decks.append(d)
    return decks


def _deck(c, RIV, SURF):
    P = c['poly']; ws = c['ways']
    # Achse: mittlere Wegrichtung (laengengewichtet), sonst laengere Seite des Mindestrechtecks
    mrr = np.array(P.minimum_rotated_rectangle.exterior.coords)[:4]
    sides = [mrr[1] - mrr[0], mrr[2] - mrr[1]]
    if ws:
        acc = np.zeros(2); ref = None
        for w in ws:
            cc = np.asarray(w['line'].coords); d = cc[-1] - cc[0]; ln = np.linalg.norm(d)
            if ln < 1e-6: continue
            d = d / ln
            if ref is None: ref = d
            acc += d * ln * (1 if np.dot(d, ref) >= 0 else -1)
        ax = acc / (np.linalg.norm(acc) + 1e-12)
        if c['mm'] is not None:          # auf die passende Polygonseite einrasten (Rechteck-Ueberbau)
            sd = max(sides, key=lambda v: abs(np.dot(v / np.linalg.norm(v), ax)))
            sd = sd / np.linalg.norm(sd)
            if abs(np.dot(sd, ax)) > 0.94: ax = sd * np.sign(np.dot(sd, ax))
    else:
        ax = max(sides, key=np.linalg.norm); ax = ax / np.linalg.norm(ax)
    o = np.array(P.centroid.coords[0])
    fr = Frame(o, ax)
    st = np.array([fr.st(*p) for p in np.asarray(P.exterior.coords)])
    os0, os1 = st[:, 0].min(), st[:, 0].max(); ot0, ot1 = st[:, 1].min(), st[:, 1].max()
    # Deckhoehe auf der Achse (Mitte des Polygons): Uferhoehen am ersten/letzten Achspunkt ausserhalb des Flusses
    tm = (ot0 + ot1) / 2
    if ws:                               # Achse der Wege (dort wird gegangen/gefahren), nicht Polygonmitte
        tw_ = [fr.st(*np.asarray(w['line'].interpolate(0.5, normalized=True).coords[0]))[1] for w in ws]
        tm = float(np.clip(np.mean(tw_), ot0, ot1))
    S = np.arange(os0 - 25.0, os1 + 25.0, 0.5)
    Pax = np.array([fr.xz(s, tm) for s in S])
    inr = shapely.contains_xy(RIV, Pax[:, 0], Pax[:, 1])
    if not inr.any(): return None
    i0, i1 = np.nonzero(inr)[0][[0, -1]]
    ter = SURF(Pax[:, 0], Pax[:, 1])[0]
    ka, kb = i0 - 1, i1 + 1
    while ka > 0 and inr[ka]: ka -= 1
    while kb < len(S) - 1 and inr[kb]: kb += 1
    if ka < 0 or kb >= len(S): return None
    sA0, sB0 = S[ka] - 0.0, S[kb]
    # Uferhoehe: hoechster Gelaendepunkt der ersten 3 m landseitig (Kaimauer-Fuss/Uferstreifen auf Wasserhoehe ueberspringen)
    ja = [k for k in range(ka, max(ka - 7, -1), -1) if not inr[k]]; jb = [k for k in range(kb, min(kb + 7, len(S))) if not inr[k]]
    ka = max(ja, key=lambda k: ter[k]); kb = max(jb, key=lambda k: ter[k])
    sA0, sB0 = S[ka], S[kb]
    yA0, yB0 = float(ter[ka]), float(ter[kb])
    def ydeck(s_): return yA0 + (yB0 - yA0) * np.clip((np.asarray(s_, float) - sA0) / max(sB0 - sA0, 1e-6), 0, 1)
    # Deckflaeche: je Querlinie t von Ufer zu Ufer (Ende dort, wo das Gelaende die Deckhoehe erreicht und kein Wasser ist)
    rows = []
    for t_ in np.linspace(ot0, ot1, max(3, int(math.ceil((ot1 - ot0) / 0.75)) + 1)):
        Pl = np.array([fr.xz(s, t_) for s in S])
        inl = shapely.contains_xy(RIV.buffer(0.05), Pl[:, 0], Pl[:, 1])
        tl = SURF(Pl[:, 0], Pl[:, 1])[0]
        okp = (~inl) & (tl >= ydeck(S) - 0.10)
        if not inl.any():
            continue
        j0, j1 = np.nonzero(inl)[0][[0, -1]]
        a = j0
        while a > 0 and not okp[a]: a -= 1
        b = j1
        while b < len(S) - 1 and not okp[b]: b += 1
        rows.append((t_, S[a], S[b]))
    if len(rows) < 2: return None
    ring = [fr.xz(sa, t_) for t_, sa, sb in rows] + [fr.xz(sb, t_) for t_, sa, sb in rows[::-1]]
    D = Polygon(ring).buffer(0)
    if c['mm'] is None:                  # ohne Ueberbau-Polygon: auf die Wegpuffer begrenzen
        D = D.intersection(P.buffer(0.01, join_style=2))
    if D.geom_type != 'Polygon':
        D = max(getattr(D, 'geoms', [D]), key=lambda g: g.area)
    D = Polygon(D.exterior).simplify(0.02)
    sa_ = min(r[1] for r in rows); sb_ = max(r[2] for r in rows)
    road = any(w['kind'] in ('road', 'rail') for w in ws)
    return {'poly': D, 'o': o.tolist(), 'd': ax.tolist(), 'yA0': yA0, 'yB0': yB0, 'sA0': float(sA0), 'sB0': float(sB0),
            'ot': (float(ot0), float(ot1)), 's_range': (float(sa_), float(sb_)), 'rows': rows, 'road': road, 'name': c['name'], 'mm': c['mm'],
            'ways': [{'id': w['id'], 'kind': w['kind'], 'w': w['w'], 'line': list(w['line'].coords), 'tags': w['tags']} for w in ws]}


def load(L=None, SURF=None, osm=None, force=False):
    if not force and os.path.exists(CACHE_F) and os.path.getmtime(CACHE_F) > os.path.getmtime(CACHE + 'layout.pkl') \
            and os.path.getmtime(CACHE_F) > os.path.getmtime(CACHE + 'terrain.npz') and os.path.getmtime(CACHE_F) > os.path.getmtime(__file__):
        return pickle.load(open(CACHE_F, 'rb'))
    if L is None: L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
    if SURF is None:
        from surface import Surface
        SURF = Surface()
    if osm is None: osm = load_osm()
    decks = compute(L, SURF, osm)
    tmp = CACHE_F + '.tmp'; pickle.dump(decks, open(tmp, 'wb')); os.replace(tmp, CACHE_F)
    return decks


def ydeck_fn(dk):
    fr = Frame(dk['o'], dk['d']); sA0, sB0, yA0, yB0 = dk['sA0'], dk['sB0'], dk['yA0'], dk['yB0']
    def f(x, z):
        s = (np.c_[np.atleast_1d(x), np.atleast_1d(z)] - fr.o) @ fr.d
        return yA0 + (yB0 - yA0) * np.clip((s - sA0) / max(sB0 - sA0, 1e-6), 0, 1)
    return f


class DeckHeight:
    """deck(x, z) -> Deckhoehe oder NaN (nicht auf einer Bruecke)"""
    def __init__(self, decks):
        self.decks = decks; self.fns = [ydeck_fn(d) for d in decks]
        for d in decks: shapely.prepare(d['poly'])
    def __call__(self, x, z):
        x = np.atleast_1d(np.asarray(x, float)); z = np.atleast_1d(np.asarray(z, float))
        out = np.full(len(x), np.nan)
        for d, f in zip(self.decks, self.fns):
            m = shapely.contains_xy(d['poly'], x, z)
            if m.any(): out[m] = f(x[m], z[m])
        return out
    def union(self, pad=0.0):
        return unary_union([d['poly'].buffer(pad) for d in self.decks]) if self.decks else Polygon()


if __name__ == '__main__':
    for d in load(force=True):
        print(d['name'], d['mm'], 'area %.0f' % d['poly'].area, 'y %.2f..%.2f' % (d['yA0'], d['yB0']), [(w['id'], w['kind']) for w in d['ways']])

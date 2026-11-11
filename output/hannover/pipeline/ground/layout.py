"""OSM -> ebene Flaechenaufteilung (Klassen-Polygone) fuer den Boden ausserhalb des Detail-Quadrats.
Ausgabe: data/ground_cache/layout.pkl  {classes:{name:geom}, holes:[...], lines:{...}, points:{...}, region, ...}"""
import sys, json, time, pickle, math, re
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point, box, MultiPoint
from shapely.ops import unary_union, polygonize, linemerge
from shapely import affinity
from common import *

T0 = time.time()
def log(*a): print('[layout %5.1fs]' % (time.time() - T0), *a, flush=True)

RG = RADIUS               # Radius Bodenplatte (Ausbau 2: 860 m)
STRIP_IN = 3.0             # so weit reicht unser Boden in das Detail-Quadrat hinein (unter dem Detailboden)
osm = load_osm()
EMPTY = Polygon()

def U(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return unary_union(geoms) if geoms else Polygon()

def clean(g, eps=0.0):
    if g.is_empty: return g
    g = g.buffer(0)
    if eps: g = g.buffer(-eps, join_style=2).buffer(eps, join_style=2)
    return polys_only(g)

def polys_only(g):
    if g.is_empty: return Polygon()
    if g.geom_type == 'Polygon': return g
    if g.geom_type == 'MultiPolygon': return g
    ps = [p for p in getattr(g, 'geoms', []) if p.geom_type in ('Polygon', 'MultiPolygon') and not p.is_empty]
    return unary_union(ps) if ps else Polygon()

def fnum(v, default=None):
    if v is None: return default
    m = re.match(r'\s*([0-9]+(?:[.,][0-9]+)?)', str(v))
    return float(m.group(1).replace(',', '.')) if m else default

def layer(t):
    try: return float(str(t.get('layer', '0')).split(';')[0])
    except ValueError: return 0.0

def closed(pts): return len(pts) >= 4 and pts[0] == pts[-1]

def way_poly(w):
    if not closed(w['pts']): return None
    p = Polygon(w['pts'])
    return p.buffer(0) if not p.is_valid else p

def rel_poly(r):
    outer = [LineString(m['pts']) for m in r['members'] if m['role'] in ('outer', '') and len(m['pts']) >= 2]
    inner = [LineString(m['pts']) for m in r['members'] if m['role'] == 'inner' and len(m['pts']) >= 2]
    po = U(list(polygonize(unary_union(outer)))) if outer else EMPTY
    pi = U(list(polygonize(unary_union(inner)))) if inner else EMPTY
    return polys_only(po.difference(pi))

# ------------------------------------------------------------------ Region
SQB = box(*SQ)
SQ_IN = box(SQ[0] + STRIP_IN, SQ[1] + STRIP_IN, SQ[2] - STRIP_IN, SQ[3] - STRIP_IN)
B = json.load(open(DATA + 'lod2_buildings.json'))['buildings']
bl = []
for b in B:
    for g in b['grounds'] or []:
        P = np.array(g['ext'])[:, :2] * [1, -1]
        if len(P) >= 3:
            p = Polygon(P).buffer(0)
            if p.area > 1: bl.append(p)
BLD = U(bl)
log('lod2 footprints', len(bl), 'area', round(BLD.area))
circ = Point(0, 0).buffer(RG, quad_segs=96)
far = [p for p in bl if not circ.contains(p) and p.intersects(Point(0, 0).buffer(RG + 80))]
REGION_OUT = U([circ] + [p.buffer(6, join_style=2) for p in far]).simplify(0.5)
REGION_OUT = Polygon(REGION_OUT.exterior) if REGION_OUT.geom_type == 'Polygon' else max(REGION_OUT.geoms, key=lambda g: g.area)
REGION = REGION_OUT.difference(SQ_IN)
SEAM = REGION.intersection(SQB)                  # Randstreifen im Quadrat: folgt dem Detailboden
OUTSIDE = REGION.difference(SQB)
log('region area', round(REGION.area), 'seam strip', round(SEAM.area))

def inreg(g): return g.intersects(REGION_OUT)

# ------------------------------------------------------------------ Strassen
ROAD_W = {'motorway': 16, 'trunk': 14, 'primary': 13, 'secondary': 11.5, 'tertiary': 9.0, 'unclassified': 6.5,
          'residential': 6.5, 'living_street': 5.5, 'service': 4.2, 'busway': 6.5, 'road': 6.0,
          'primary_link': 6.5, 'secondary_link': 6.0, 'tertiary_link': 5.5}
SERVICE_W = {'parking_aisle': 5.5, 'driveway': 3.2, 'alley': 3.8, 'drive-through': 3.5, 'emergency_access': 3.5}
PARK_OK = ('parallel', 'diagonal', 'perpendicular', 'lane', 'street_side', 'on_street', 'half_on_kerb', 'marked', 'yes')

def road_width(t):
    hw = t['highway']
    w = fnum(t.get('width'))
    if w and 2.5 <= w <= 40: base = w
    else:
        lanes = fnum(t.get('lanes'))
        if lanes and lanes >= 1:
            base = lanes * 3.1 + 0.4
        else:
            base = ROAD_W.get(hw, 6.0)
            if hw == 'service': base = SERVICE_W.get(t.get('service'), base)
            if t.get('oneway') == 'yes' and hw in ('secondary', 'tertiary', 'primary', 'residential', 'unclassified'):
                base *= 0.62
    # Parkstreifen
    def pk(side):
        for k in ('parking:%s' % side, 'parking:lane:%s' % side):
            v = t.get(k)
            if v in PARK_OK: return 4.5 if v in ('diagonal', 'perpendicular') else 2.0
            if v in ('no', 'separate', 'no_parking', 'no_stopping'): return 0.0
        return None
    both = pk('both')
    if both is not None and both > 0: base += 2 * both
    else:
        for s in ('left', 'right'):
            v = pk(s)
            if v: base += v
    return base

ROAD_HW = set(ROAD_W) | {'construction'}
roads = []; road_lines = []; passage = []; sett_roads = []
for w in osm['ways']:
    t = w['tags']; hw = t.get('highway')
    if hw not in ROAD_W: continue
    if t.get('area') == 'yes' or len(w['pts']) < 2: continue
    if t.get('tunnel') == 'yes' or layer(t) < 0: continue
    if t.get('bridge') and layer(t) >= 1: continue
    L = LineString(w['pts'])
    if not inreg(L.buffer(20)): continue
    wd = road_width(t)
    g = L.buffer(wd / 2, cap_style=1, join_style=1, quad_segs=6)
    roads.append(g); road_lines.append((L, wd, t))
    if t.get('tunnel') == 'building_passage' or t.get('covered') == 'yes': passage.append(g)
    if t.get('surface') in ('sett', 'cobblestone', 'unhewn_cobblestone', 'paving_stones', 'grass_paver'): sett_roads.append(g)
# Parkplatzflaechen am Strassenrand -> Fahrbahnniveau
parking = []; street_parking = []
for w in osm['ways']:
    t = w['tags']
    if t.get('amenity') not in ('parking', 'parking_space'): continue
    p = way_poly(w)
    if p is None or not inreg(p) or layer(t) != 0: continue
    if t.get('parking') in ('multi-storey', 'underground', 'rooftop'): continue
    if t.get('parking') == 'street_side' or t.get('amenity') == 'parking_space': street_parking.append(p)
    else: parking.append(p)
log('roads', len(roads), 'parking', len(parking), 'street_side', len(street_parking))

# ------------------------------------------------------------------ Gruen, Wasser, Fussgaengerflaechen
grass = []; beds = []; peds = []; water = []; rail = []; railplat = []; hbs = []; islands = []; rivers = []
GRASS_LU = ('grass', 'village_green', 'meadow', 'recreation_ground', 'cemetery', 'greenfield')
for w in osm['ways'] + [dict(r, pts=None) for r in osm['relations']]:
    t = w['tags']
    p = way_poly(w) if w.get('pts') is not None else rel_poly(w)
    if p is None or p.is_empty or not inreg(p): continue
    if layer(t) < 0 or 'building' in t: continue
    lu = t.get('landuse'); le = t.get('leisure'); na = t.get('natural')
    if t.get('water') == 'river' or t.get('waterway') == 'riverbank':
        rivers.append(p); continue
    if t.get('amenity') == 'fountain' or na == 'water' or t.get('water') or le == 'swimming_pool':
        water.append((p, t)); continue
    if lu == 'flowerbed' or na in ('scrub', 'shrubbery', 'heath'): beds.append(p); continue
    if lu in GRASS_LU or le in ('park', 'garden', 'common', 'dog_park', 'playground') or na in ('grassland', 'wood') or lu == 'forest':
        grass.append(p); continue
    if lu == 'railway': rail.append(p); continue
    if t.get('area:highway') == 'traffic_island': islands.append(p); continue
    if t.get('highway') in ('pedestrian', 'footway', 'platform') and (t.get('area') == 'yes' or w.get('pts') is None) or t.get('place') == 'square' or t.get('area:highway') in ('pedestrian', 'footway'):
        if t.get('highway') == 'platform' or t.get('public_transport') == 'platform':
            hbs.append((p, t))
        else:
            peds.append(p)
        continue
    if t.get('railway') == 'platform' or t.get('public_transport') == 'platform':
        if t.get('railway') == 'platform' and not (t.get('tram') == 'yes' or t.get('light_rail') == 'yes'): railplat.append((p, t))
        else: hbs.append((p, t))
GRASS = U(grass); BEDS = U(beds); PEDA = U(peds)
log('grass', len(grass), 'beds', len(beds), 'ped areas', len(peds), 'water', len(water), 'rail', len(rail), 'hbs', len(hbs), 'islands', len(islands))

# Stadtbahn an der Oberflaeche
lr = []
for w in osm['ways']:
    t = w['tags']
    if t.get('railway') in ('light_rail', 'tram') and t.get('tunnel') != 'yes' and layer(t) >= 0 and len(w['pts']) >= 2:
        L = LineString(w['pts'])
        if inreg(L): lr.append((L, t))
LR = U([L.buffer(1.45, cap_style=2) for L, t in lr])
# Hochbahnsteige: nur Plattformen nahe Oberflaechen-Gleisen
HBS = U([p for p, t_ in hbs if any(p.distance(L) < 6 for L, t in lr) and p.area < 900 and not (t_.get('bus') == 'yes' and t_.get('tram') != 'yes')
         and t_.get('layer', '0')[:1] != '-'])
# Bahnsteige der DB (Hbf): Niveau aus DGM, nur als Belag
RAILPLAT = U([p for p, t in railplat if layer(t) >= 0 and not any(p.distance(L) < 6 for L, t2 in lr)])
RAIL = U(rail)

# Baeume / Laternen
trees = [n for n in osm['nodes'] if n['tags'].get('natural') == 'tree' and np.hypot(*n['p']) < RG + 30]
lamps = [n for n in osm['nodes'] if n['tags'].get('highway') == 'street_lamp' and np.hypot(*n['p']) < RG + 30]
TREEP = [Point(n['p']) for n in trees]
log('trees', len(trees), 'lamps', len(lamps))

# Fuss-/Radwege (Linien)
foot_lines = []; cyc_lines = []; side_lines = []; ped_lines = []
for w in osm['ways']:
    t = w['tags']; hw = t.get('highway')
    if hw not in ('footway', 'path', 'pedestrian', 'cycleway', 'steps', 'track', 'bridleway', 'corridor'): continue
    if len(w['pts']) < 2 or layer(t) != 0 or t.get('tunnel') in ('yes', 'building_passage') or t.get('indoor') == 'yes' or t.get('area') == 'yes': continue
    if t.get('level') not in (None, '0'): continue
    L = LineString(w['pts'])
    if not inreg(L.buffer(10)): continue
    if hw == 'cycleway': cyc_lines.append((L, t))
    elif t.get('footway') == 'sidewalk' or t.get('footway') == 'crossing' or t.get('cycleway') == 'crossing':
        if t.get('footway') == 'sidewalk': side_lines.append((L, t))
    elif hw == 'pedestrian': ped_lines.append((L, t))
    else: foot_lines.append((L, t))
log('lines: foot', len(foot_lines), 'cycle', len(cyc_lines), 'sidewalk', len(side_lines), 'ped', len(ped_lines))

# ------------------------------------------------------------------ Promenade-Oeffnungen + U-Bahn-Schaechte (Loecher)
PRJ = json.load(open(CACHE + 'promenade.json'))
prom = PRJ['openings']
holes = {}
if prom:
    mu = np.array(PRJ['axis']['origin']); ax_ = np.array(PRJ['axis']['axis']); nr_ = np.array(PRJ['axis']['normal'])
    def W_(s, t): return tuple(mu + ax_ * s + nr_ * t)
    for i, o in enumerate(prom):
        s0, s1 = o['s']; t0, t1 = o['t']
        # Querkanten: Zusatzpunkte an den Grenzen der ueberdeckten Korridore zu den Nachbaroeffnungen
        def cuts(j):
            if j < 0 or j >= len(prom): return []
            q = prom[j]; a = max(t0, q['t'][0]); b = min(t1, q['t'][1])
            return [t for t in (a, b) if t0 + 0.05 < t < t1 - 0.05]
        south = sorted(cuts(i - 1)); north = sorted(cuts(i + 1))
        ring = [W_(s0, t0), W_(s1, t0)] + [W_(s1, t) for t in north] + [W_(s1, t1), W_(s0, t1)] + [W_(s0, t) for t in south[::-1]]
        holes['prom%d' % i] = {'poly': Polygon(ring), 'kind': 'promenade', 'floor': o['floor'], 'idx': i}

import ubahn_spec
SPECS = ubahn_spec.all_specs(osm, BLD)
ubahn = []
for sp in SPECS:
    if sp.get('skip'):
        log('skip subway entrance', sp['key'], sp['skip']); continue
    hp = sp['poly_open'].intersection(REGION)
    for k_, g_ in enumerate([g for g in getattr(hp, 'geoms', [hp]) if g.geom_type == 'Polygon' and g.area > 0.05]):
        holes['ub_%s_%d' % (sp['key'], k_)] = {'poly': g_, 'kind': 'ubahn', 'key': sp['key']}
    ubahn.append(sp['key'])
log('promenade openings', len(prom), 'subway entrances used', len(ubahn), 'with terrain holes', len(set(h['key'] for h in holes.values() if h['kind'] == 'ubahn')))
HOLES = U([h['poly'] for h in holes.values()])
# Einfassung: Promenade-Oeffnungen 0.35 m Granit; U-Bahn-Troege: Bruestungsring (Bruestung steht darauf)
COPING = U([U([h['poly'] for h in holes.values() if h['kind'] == 'promenade']).buffer(0.35, join_style=2)] +
           [sp['poly_parapet'].buffer(0.05, join_style=2) for sp in SPECS if not sp.get('skip')]).difference(HOLES)

# ------------------------------------------------------------------ Fahrbahn bauen
ROADS = U(roads)
# Korrekturen fuer ueberschaetzte Breiten: Gehweg-/Fusswegachsen, Baeume, Laternen, Gruenflaechen
nocross = []
RL = U([L for L, w, t in road_lines])
for L, t in side_lines + foot_lines + cyc_lines:
    if L.crosses(RL) or L.length < 3: continue
    nocross.append(L.buffer(0.9, cap_style=2))
obst = [p.buffer(1.1) for p in TREEP] + [Point(n['p']).buffer(0.5) for n in lamps]
ROADS = ROADS.difference(U(nocross + obst))
PASS = U(passage)
ROADS = U([ROADS.difference(BLD.buffer(1.2)), PASS]).difference(GRASS).difference(BEDS)
ROADS = U([ROADS, U(street_parking)])
# Abrundung an Einmuendungen (Bordradien) und Entfernen schmaler Reste
ROADS = ROADS.buffer(3.0, join_style=1, quad_segs=6).buffer(-3.0, join_style=1, quad_segs=6)
ROADS = clean(ROADS, 0.25)
# ---- Naht: Fahrbahnen des Detailmodells am Quadratrand fortsetzen (Uebergang auf 12 m zur OSM-Fahrbahn)
st = np.load(CACHE + 'strip.npz')
SX_, SZ_, SY_, SK_ = st['x'], st['z'], st['y'], st['k']
BAND = 12.0
sides = [('top', 0), ('right', 1), ('bottom', 2), ('left', 3)]
x0_, z0_, x1_, z1_ = SQ
def side_geom(i):
    # (Startpunkt, Richtung entlang, Normale nach aussen, Laenge)
    if i == 0: return np.array([x0_, z0_]), np.array([1.0, 0]), np.array([0, -1.0]), x1_ - x0_
    if i == 1: return np.array([x1_, z0_]), np.array([0, 1.0]), np.array([1.0, 0]), z1_ - z0_
    if i == 2: return np.array([x1_, z1_]), np.array([-1.0, 0]), np.array([0, 1.0]), x1_ - x0_
    return np.array([x0_, z1_]), np.array([0, -1.0]), np.array([-1.0, 0]), z1_ - z0_
tapers = []; band_all = []
for name_, i in sides:
    o, a, nout, ln = side_geom(i)
    q = np.c_[SX_ - o[0], SZ_ - o[1]]
    s_ = q @ a; inset = -(q @ nout)
    m = (~np.isnan(SY_)) & (inset > 0) & (inset < 5) & (s_ > 0) & (s_ < ln)
    nb = int(ln / 0.5) + 1
    kind = np.full(nb, -1); best = np.full(nb, 99.0)
    for ss, ii, kk in zip(s_[m], inset[m], SK_[m]):
        b = int(ss / 0.5)
        if ii < best[b]: best[b] = ii; kind[b] = kk
    road = np.isin(kind, (1, 2)).astype(int)
    # kleine Luecken schliessen
    from scipy.ndimage import binary_closing, binary_opening
    road = binary_opening(binary_closing(road, np.ones(3)), np.ones(4))
    iv = []; st0 = None
    for b in range(nb + 1):
        r = road[b] if b < nb else False
        if r and st0 is None: st0 = b
        if not r and st0 is not None: iv.append((st0 * 0.5, b * 0.5)); st0 = None
    band = Polygon([o, o + a * ln, o + a * ln + nout * BAND, o + nout * BAND])
    band_all.append(band)
    # OSM-Fahrbahn auf der Linie BAND m aussen
    lineo = LineString([o + nout * BAND - a * 5, o + a * (ln + 5) + nout * BAND])
    hit = lineo.intersection(ROADS)
    ivo = []
    for g in getattr(hit, 'geoms', [hit]):
        if g.is_empty or g.geom_type != 'LineString': continue
        cc = np.array(g.coords); ss = (cc - o) @ a; ivo.append((ss.min(), ss.max()))
    for (sa, sb) in iv:
        if sb - sa < 2.5 or sb - sa > 25: continue          # zu schmal oder laengs der Naht verlaufend
        best_o = None; bo = 0
        for (ta, tb) in ivo:
            ov = min(sb, tb) - max(sa, ta)
            if ov > bo or (best_o is None and abs((ta + tb) / 2 - (sa + sb) / 2) < 8): best_o = (ta, tb); bo = max(ov, bo)
        if best_o is None:
            # keine OSM-Fahrbahn: nur kurze Zufahrt (3 m)
            poly = Polygon([o + a * sa - nout * 0.05, o + a * sb - nout * 0.05, o + a * sb + nout * 3.0, o + a * sa + nout * 3.0])
        else:
            ta, tb = best_o
            poly = Polygon([o + a * sa - nout * 0.05, o + a * sb - nout * 0.05, o + a * tb + nout * (BAND + 0.5), o + a * ta + nout * (BAND + 0.5)]).convex_hull
        if poly.intersection(BLD).area > 0.3 * poly.area: continue
        tapers.append(poly)
    log('seam side', name_, 'detail road intervals', [(round(a_, 1), round(b_, 1)) for a_, b_ in iv], 'osm at band', [(round(a_, 1), round(b_, 1)) for a_, b_ in ivo])
BANDS = U(band_all)
ROADS = U([ROADS.difference(BANDS), U(tapers)])
ROADS = ROADS.buffer(2.0, join_style=1, quad_segs=6).buffer(-2.0, join_style=1, quad_segs=6)
ROADS = clean(ROADS, 0.25)
# Stadtbahn-Gleise ausserhalb der Fahrbahn: eigener Gleiskoerper (Fahrbahnniveau, Pflaster)
TRACK = clean(LR.difference(ROADS).difference(BLD), 0.2)
PARK = clean(U(parking).difference(BLD), 0.2)
SETTR = U(sett_roads).intersection(ROADS)
log('road area', round(ROADS.area), 'track', round(TRACK.area), 'parking', round(PARK.area))

LOW = U([ROADS, PARK, TRACK])
# ------------------------------------------------------------------ Wasserbecken
POOLS = []; RIMS = []
for p, t in water:
    if t.get('water') == 'river' or t.get('waterway'):
        POOLS.append(p); continue
    p = p.buffer(0)
    RIMS.append(p.difference(p.buffer(-0.4, join_style=2))); POOLS.append(p.buffer(-0.4, join_style=2))
for n in osm['nodes']:
    t = n['tags']
    if t.get('amenity') != 'fountain' or np.hypot(*n['p']) > RG - 5: continue
    pt = Point(n['p'])
    if any(pt.within(p.buffer(1)) for p, _ in water) or SQB.buffer(3).contains(pt) or BLD.contains(pt): continue
    r = 1.2 if t.get('fountain') == 'decorative' else 2.4
    c = pt.buffer(r, quad_segs=12)
    RIMS.append(c.difference(c.buffer(-0.4))); POOLS.append(c.buffer(-0.4))
WATER = U(POOLS); RIM = U(RIMS)
RIVER = U([p.buffer(1.5, join_style=2) for p in rivers]).buffer(-1.5, join_style=2) if rivers else EMPTY
RIVER = clean(RIVER.intersection(REGION_OUT), 0.3)
bridges = []
for w in osm['ways']:
    t = w['tags']
    if not t.get('bridge') or layer(t) < 1 or len(w['pts']) < 2: continue
    hw = t.get('highway')
    if hw is None and t.get('railway') not in ('tram', 'light_rail'): continue
    Lb = LineString(w['pts'])
    if RIVER.is_empty or not Lb.intersects(RIVER.buffer(1)): continue
    wd = road_width(t) if hw in ROAD_W else (max(fnum(t.get('width'), 3.0), 2.5) if hw else 6.0)
    bridges.append({'line': list(Lb.coords), 'width': wd, 'tags': t, 'id': w['id']})
log('river area', round(RIVER.area), 'bridges over river', len(bridges))

# ------------------------------------------------------------------ Hecken (Beet-Streifen) und Pfade im Gruen
hedge_lines = []
for w in osm['ways']:
    t = w['tags']
    if t.get('barrier') == 'hedge' and len(w['pts']) >= 2:
        L = LineString(w['pts'])
        if inreg(L): hedge_lines.append(L)
HEDGEBED = U([L.buffer(0.55, cap_style=2) for L in hedge_lines])
GREEN = U([GRASS, BEDS]).difference(LOW)
paths = []
for L, t in foot_lines + ped_lines + cyc_lines + side_lines:
    if not L.intersects(GREEN): continue
    wd = fnum(t.get('width'), 2.2 if t['highway'] != 'path' else 1.6)
    paths.append(L.buffer(min(max(wd, 1.2), 5) / 2, cap_style=1))
PATHS = U(paths).intersection(GREEN)
paved_path = U([L.buffer(1.1) for L, t in foot_lines + ped_lines if t.get('surface') in ('paving_stones', 'asphalt', 'concrete', 'sett', 'paved', None)]).intersection(PATHS)
GRAVEL = clean(PATHS.difference(paved_path), 0.1)
PATHPAVED = clean(PATHS.intersection(paved_path), 0.1)

# ------------------------------------------------------------------ Baumscheiben
def local_dir(pt):
    best = None; bd = 1e9
    for L, w, t in road_lines:
        d = L.distance(pt)
        if d < bd and d < 40: bd = d; best = L
    if best is None: return 0.0
    s = best.project(pt); a = np.array(best.interpolate(max(0, s - 1)).coords[0]); b = np.array(best.interpolate(s + 1).coords[0])
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

pits = []; frames = []
NOPIT = U([GREEN, WATER, RIM, HOLES.buffer(1), BLD, SQB.buffer(0.5), HEDGEBED])
from shapely.strtree import STRtree
for n in trees:
    pt = Point(n['p'])
    if not OUTSIDE.contains(pt) or NOPIT.contains(pt): continue
    a = local_dir(pt)
    sq = affinity.rotate(box(-0.9, -0.9, 0.9, 0.9), a, origin=(0, 0)); sq = affinity.translate(sq, *n['p'])
    if sq.intersects(NOPIT): continue
    inner = sq.buffer(-0.1, join_style=2)
    pits.append(inner); frames.append(sq.difference(inner))
TREEPIT = U(pits); PITFRAME = U(frames)
log('tree pits', len(pits))

# ------------------------------------------------------------------ Klassen per Prioritaet zusammensetzen
order = []
def add(name, geom):
    order.append((name, geom))

# Hochbahnsteige, Becken, Baumscheiben, Lochrand
add('coping', COPING)
add('river', RIVER)
add('rim', RIM)
add('water', WATER)
add('pitframe', PITFRAME)
add('treepit', TREEPIT)
# Fahrbahn + Bord: Bordstreifen (0.16 m, davon 3 cm Fase) auf der hohen Seite
HIGHSIDE_EXCL = U([LOW, WATER, RIM, HOLES, RIVER])
KERB_ALL = LOW.buffer(0.17, join_style=2, mitre_limit=1.5).difference(LOW)
KERB_CH = LOW.buffer(0.03, join_style=2, mitre_limit=1.5).difference(LOW)
add('kerb_ch', KERB_CH)
add('kerb', KERB_ALL)
GUTTER = LOW.intersection(KERB_ALL.buffer(0.32, join_style=2)).difference(TRACK).difference(PARK)
add('gutter', GUTTER)
add('setts_road', SETTR)
add('track', TRACK)
add('parking', PARK)
add('road', ROADS)
add('bed', U([HEDGEBED.difference(LOW), BEDS]))
add('gravel', GRAVEL)
add('path', PATHPAVED)
GRASS_EDGE = GREEN.buffer(0.1, join_style=2).difference(GREEN)
add('edge', GRASS_EDGE)
add('grass', GREEN)
CYC = U([L.buffer(min(fnum(t.get('width'), 1.8), 2.5) / 2, cap_style=1) for L, t in cyc_lines if L.length >= 20])
add('cycle', CYC)
add('island', U(islands))
add('railplat', RAILPLAT)
add('ballast', RAIL)
add('ped', PEDA)

classes = {}
taken = U([HOLES])
avail = OUTSIDE.difference(HOLES)
for name, g in order:
    if g is None or g.is_empty: continue
    g = polys_only(g.intersection(avail).difference(taken))
    if g.is_empty: continue
    classes[name] = polys_only(U([classes.get(name, EMPTY), g]))
    taken = U([taken, g])
    log('class', name, round(g.area, 1))
rest = polys_only(avail.difference(taken))
log('rest (paved)', round(rest.area))

# ------------------------------------------------------------------ Rest: Fussgaengerzone vs. Gehweg (Voronoi nach naechster Achse)
def sample(L, step):
    n = max(2, int(L.length / step) + 1)
    return [L.interpolate(i / (n - 1), normalized=True).coords[0] for i in range(n)]
pts = []; lab = []
kb = classes.get('kerb', EMPTY).boundary
for g in getattr(LOW.boundary, 'geoms', [LOW.boundary]):
    for c in sample(g, 2.0): pts.append(c); lab.append(0)
for L, t in side_lines + cyc_lines:
    for c in sample(L, 2.0): pts.append(c); lab.append(0)
for L, t in ped_lines + foot_lines:
    for c in sample(L, 2.0): pts.append(c); lab.append(1)
for g in getattr(PEDA.boundary, 'geoms', [PEDA.boundary]) if not PEDA.is_empty else []:
    pass
# Fussgaengerflaechen selbst: Punkte im Inneren (Raster 4 m)
if not PEDA.is_empty:
    mnx, mnz, mxx, mxz = PEDA.bounds
    gx, gz = np.meshgrid(np.arange(mnx, mxx, 4.0), np.arange(mnz, mxz, 4.0))
    ins = shapely.contains_xy(PEDA, gx.ravel(), gz.ravel())
    for x, z in zip(gx.ravel()[ins], gz.ravel()[ins]): pts.append((x, z)); lab.append(1)
pts = np.array(pts); lab = np.array(lab)
keep = np.hypot(pts[:, 0], pts[:, 1]) < RG + 60
pts = pts[keep]; lab = lab[keep]
_, ui = np.unique(np.round(pts, 2), axis=0, return_index=True)
pts = pts[ui]; lab = lab[ui]
vor = shapely.voronoi_polygons(MultiPoint(pts), extend_to=REGION_OUT.buffer(100))
cells = list(vor.geoms)
# Zelle -> Label ueber STRtree der Punkte
tree = STRtree([Point(p) for p in pts])
pedcells = []
for c in cells:
    idx = tree.query(c, predicate='contains')
    if len(idx) and lab[idx[0]] == 1: pedcells.append(c)
PEDV = U(pedcells)
PEDZ = clean(rest.intersection(PEDV), 0.8)
SIDE = polys_only(rest.difference(PEDZ))
classes['ped'] = polys_only(U([classes.get('ped', EMPTY), PEDZ]))
classes['sidewalk'] = SIDE
# ---- Altstadt: Granit-Kleinpflaster im Segmentbogen; Marktplatz/Marktkirche rotes Klinkerpflaster mit Granitraster
ALT = Polygon([(-565, 85), (-335, 62), (-262, 120), (-205, 170), (-160, 236), (-140, 300), (-150, 375), (-175, 425), (-300, 480),
               (-420, 478), (-470, 455), (-540, 425), (-575, 330)])
mk = [p for p in bl if p.distance(Point(-236, 295)) < 3 and p.area > 400]
CLINK = U([Point(-218, 324).buffer(30)] + [p.buffer(22, join_style=2) for p in mk]).intersection(Polygon([(-300, 255), (-160, 255), (-160, 372), (-300, 372)]))
HOLZ = Point(-433, 334).buffer(26)
# Ballhofplatz: rotes Klinker im Fischgraet mit Granit-Einfassung (Street View / Fotos) -> Klinker-Klasse
BALL = [Polygon(w['pts']).buffer(0) for w in osm['ways'] if w['id'] == 29179586 and len(w['pts']) >= 4]
CLINK = U([CLINK] + BALL)
for nm in ('ped', 'sidewalk', 'path', 'island', 'cycle'):
    if nm not in classes: continue
    g = classes[nm]
    inside = polys_only(g.intersection(ALT))
    if inside.is_empty: continue
    classes[nm] = polys_only(g.difference(ALT))
    ck = polys_only(inside.intersection(CLINK)) if nm in ('ped', 'sidewalk') else EMPTY
    rest_ = polys_only(inside.difference(ck))
    classes['clinker'] = polys_only(U([classes.get('clinker', EMPTY), ck]))
    classes['alt_setts'] = polys_only(U([classes.get('alt_setts', EMPTY), rest_]))
log('altstadt setts', round(classes.get('alt_setts', EMPTY).area), 'clinker', round(classes.get('clinker', EMPTY).area))
# Nahtstreifen: Belagsart des Detailmodells (Raster 0.25 m, Luecken mit naechstem Wert gefuellt) als Klassen
from scipy import ndimage as ndi
SR_ = float(st['res']); gx0_ = SQ[0] + SR_ / 2; gz0_ = SQ[1] + SR_ / 2
nxs = int(round((SQ[2] - SQ[0]) / SR_)); nzs = int(round((SQ[3] - SQ[1]) / SR_))
KG = np.full((nzs, nxs), -1, int)
ixs = np.clip(np.round((SX_ - gx0_) / SR_).astype(int), 0, nxs - 1); izs = np.clip(np.round((SZ_ - gz0_) / SR_).astype(int), 0, nzs - 1)
okk = ~np.isnan(SY_)
grp_ = np.where(np.isin(SK_, (1,)), 1, np.where(SK_ == 2, 2, np.where(SK_ == 3, 3, 0)))
KG[izs[okk], ixs[okk]] = grp_[okk]
idx_ = ndi.distance_transform_edt(KG < 0, return_distances=False, return_indices=True)
KF = KG[idx_[0], idx_[1]]
gxs = gx0_ + np.arange(nxs) * SR_; gzs = gz0_ + np.arange(nzs) * SR_
dE = np.minimum.reduce(np.meshgrid(gxs - SQ[0], gzs - SQ[1])) 
GXs, GZs = np.meshgrid(gxs, gzs)
dedge_ = np.minimum.reduce([GXs - SQ[0], SQ[2] - GXs, GZs - SQ[1], SQ[3] - GZs])
seam_cls = {}
for k_, nm_ in ((1, 'seam_road'), (2, 'seam_setts'), (3, 'seam_grass')):
    m_ = (KF == k_) & (dedge_ < STRIP_IN + 0.6)
    boxes = []
    for r_ in np.nonzero(m_.any(1))[0]:
        row = m_[r_]; d_ = np.diff(np.r_[0, row.astype(int), 0]); st_ = np.nonzero(d_ == 1)[0]; en_ = np.nonzero(d_ == -1)[0]
        for a_, b_ in zip(st_, en_):
            boxes.append(box(gxs[a_] - SR_ / 2, gzs[r_] - SR_ / 2, gxs[b_ - 1] + SR_ / 2, gzs[r_] + SR_ / 2))
    if not boxes: continue
    g_ = shapely.union_all(boxes).buffer(0.15, join_style=2).buffer(-0.3, join_style=2).buffer(0.15, join_style=2).simplify(0.08)
    g_ = polys_only(g_.intersection(SEAM).difference(HOLES))
    if not g_.is_empty: seam_cls[nm_] = g_
rest_seam = SEAM.difference(HOLES)
for nm_, g_ in seam_cls.items():
    g_ = polys_only(g_.intersection(rest_seam)); classes[nm_] = g_; rest_seam = rest_seam.difference(g_)
    log('seam class', nm_, round(g_.area, 1))
classes['seam'] = polys_only(rest_seam)
for k in list(classes):
    log('final', k, round(classes[k].area, 1), classes[k].geom_type)

out = {'classes': classes, 'holes': holes, 'specs': SPECS, 'river': RIVER, 'bridges': bridges, 'platforms': [p for p in getattr(HBS, 'geoms', [HBS]) if not p.is_empty],
       'alt_zone': ALT, 'region': REGION, 'region_out': REGION_OUT, 'sq_in': SQ_IN, 'seam': SEAM,
       'low': LOW, 'roads': ROADS, 'road_lines': [(list(L.coords), w, t) for L, w, t in road_lines],
       'lr': [(list(L.coords), t) for L, t in lr], 'hedges': [list(L.coords) for L in hedge_lines],
       'bld': BLD, 'trees': trees, 'lamps': lamps, 'ped_lines': [(list(L.coords), t) for L, t in ped_lines],
       'foot_lines': [(list(L.coords), t) for L, t in foot_lines], 'side_lines': [(list(L.coords), t) for L, t in side_lines]}
pickle.dump(out, open(CACHE + 'layout.pkl', 'wb'))
log('done')

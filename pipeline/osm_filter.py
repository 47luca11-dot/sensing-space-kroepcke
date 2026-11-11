"""OSM (Overpass JSON, out geom) -> osm_features.json in Spielkoordinaten (x=Ost, z=-Nord, m; Ursprung Kroepcke-Uhr)"""
import json, sys, collections
from pyproj import Transformer
E0, N0 = 550298.88274216, 5802940.705715501
tr = Transformer.from_crs(4326, 25832, always_xy=True)
d = json.load(open(sys.argv[1], encoding='utf-8'))
KEEP_W = ('highway', 'railway', 'area:highway', 'landuse', 'leisure', 'natural', 'amenity', 'barrier', 'man_made', 'water', 'waterway', 'public_transport', 'surface', 'footway', 'place', 'tunnel', 'bridge', 'covered', 'building', 'building:part')
KEEP_N = {'natural': {'tree'}, 'highway': {'street_lamp', 'bus_stop', 'traffic_signals', 'crossing'}, 'amenity': {'bench', 'waste_basket', 'bicycle_parking', 'clock', 'fountain', 'post_box', 'vending_machine', 'telephone', 'recycling', 'cafe', 'restaurant', 'fast_food', 'bar', 'pub', 'pharmacy', 'bank'},
          'railway': {'subway_entrance', 'stop', 'tram_stop'}, 'man_made': {'advertising', 'flagpole', 'surveillance'}, 'advertising': {'column', 'board', 'billboard'}, 'tourism': {'artwork', 'information'}, 'shop': None, 'leisure': None, 'historic': None, 'entrance': None}
def xy(lon, lat):
    e, n = tr.transform(lon, lat); return [round(e - E0, 2), round(-(n - N0), 2)]
out = {'ways': [], 'nodes': [], 'relations': []}; c = collections.Counter()
for e in d['elements']:
    t = e.get('tags', {})
    if e['type'] == 'node':
        ok = any(k in t and (v is None or t[k] in v) for k, v in KEEP_N.items())
        if ok: out['nodes'].append({'id': e['id'], 'p': xy(e['lon'], e['lat']), 'tags': t}); c['node'] += 1
    elif e['type'] == 'way' and 'geometry' in e:
        if any(k in t for k in KEEP_W):
            out['ways'].append({'id': e['id'], 'pts': [xy(g['lon'], g['lat']) for g in e['geometry'] if g], 'tags': t}); c['way'] += 1
    elif e['type'] == 'relation' and t.get('type') == 'multipolygon' and any(k in t for k in KEEP_W):
        mem = [{'role': m.get('role'), 'pts': [xy(g['lon'], g['lat']) for g in m.get('geometry', []) if g]} for m in e.get('members', []) if m.get('type') == 'way']
        out['relations'].append({'id': e['id'], 'members': mem, 'tags': t}); c['rel'] += 1
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)
print(c)

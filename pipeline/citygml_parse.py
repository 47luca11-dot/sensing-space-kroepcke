"""CityGML LoD2 (LGLN / LH Hannover, ETRS89 UTM32 + DHHN92) -> buildings.json
Koordinaten relativ zur Kroepcke-Uhr (E0,N0): [dE, dN, Z_abs]. Radius-Filter um die Uhr.
Aufruf: python citygml_parse.py <gml...> --out buildings.json --radius 520"""
import sys, json, math, argparse
from lxml import etree
E0, N0 = 550298.88274216, 5802940.705715501          # Kroepcke-Uhr (OSM node 348155428) in EPSG:25832
NS = {'gml': 'http://www.opengis.net/gml', 'bldg': 'http://www.opengis.net/citygml/building/2.0',
      'gen': 'http://www.opengis.net/citygml/generics/2.0', 'xAL': 'urn:oasis:names:tc:ciq:xsdschema:xAL:2.0'}
B = '{%s}' % NS['bldg']; G = '{%s}' % NS['gml']

def ring(el):
    pl = el.find('.//gml:posList', NS)
    if pl is not None:
        v = [float(x) for x in pl.text.split()]
        pts = [(v[i] - E0, v[i + 1] - N0, v[i + 2]) for i in range(0, len(v), 3)]
    else:
        pts = []
        for p in el.findall('.//gml:pos', NS):
            a = [float(x) for x in p.text.split()]; pts.append((a[0] - E0, a[1] - N0, a[2]))
    if len(pts) > 1 and pts[0] == pts[-1]: pts = pts[:-1]
    return [[round(c, 3) for c in p] for p in pts]

def polygons(el):
    out = []
    for poly in el.iter(G + 'Polygon'):
        ext = poly.find('gml:exterior', NS)
        if ext is None: continue
        r = ring(ext)
        if len(r) < 3: continue
        holes = [ring(i) for i in poly.findall('gml:interior', NS)]
        out.append({'ext': r, 'holes': [h for h in holes if len(h) >= 3]})
    return out

def parse(fn, radius, out):
    for _, b in etree.iterparse(fn, tag=B + 'Building', huge_tree=True):
        if b.getparent().tag.endswith('consistsOfBuildingPart'): continue
        parts = [b] + list(b.iter(B + 'BuildingPart'))
        rec = {'id': b.get(G + 'id'), 'walls': [], 'roofs': [], 'grounds': []}
        mh = b.find('bldg:measuredHeight', NS); st = b.find('bldg:storeysAboveGround', NS)
        rec['height'] = float(mh.text) if mh is not None else None
        rec['storeys'] = int(st.text) if st is not None else None
        th = b.find('.//xAL:ThoroughfareName', NS); hn = b.find('.//xAL:ThoroughfareNumber', NS)
        rec['street'] = th.text if th is not None else None; rec['number'] = hn.text if hn is not None else None
        fn_ = b.find('bldg:function', NS); rec['function'] = fn_.text if fn_ is not None else None
        for part in parts:
            for bb in part.findall('bldg:boundedBy', NS):
                for s in bb:
                    k = {'WallSurface': 'walls', 'RoofSurface': 'roofs', 'GroundSurface': 'grounds'}.get(etree.QName(s).localname)
                    if k: rec[k] += polygons(s)
        b.clear()
        pts = [p for g in rec['grounds'] for p in g['ext']] or [p for w in rec['walls'] for p in w['ext']]
        if not pts: continue
        cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
        if math.hypot(cx, cy) > radius: continue
        rec['centre'] = [round(cx, 2), round(cy, 2)]
        out[rec['id']] = rec

ap = argparse.ArgumentParser(); ap.add_argument('gml', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--radius', type=float, default=520)
a = ap.parse_args(); out = {}
for fn in a.gml:
    parse(fn, a.radius, out); print(fn, len(out))
json.dump({'origin_utm': [E0, N0], 'crs': 'EPSG:25832 + DHHN92', 'buildings': list(out.values())}, open(a.out, 'w'))
print('buildings', len(out))

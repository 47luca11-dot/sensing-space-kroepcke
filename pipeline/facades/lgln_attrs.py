"""LGLN-LoD2 (AdV-CityGML 1.0, CC BY 4.0) -> cache/lgln_parts.json
Je Gebaeude/Gebaeudeteil: Funktion (ALKIS), Dachtyp, Trauf-/Firsthoehe, Grundriss (dE,dN relativ zur Kroepcke-Uhr).
Nur Attribute fuer die Stilwahl - die Geometrie bleibt die des Stadtmodells (data/lod2_buildings.json)."""
import glob, os, sys
from lxml import etree
from common import RESEARCH, CACHE, E0, N0, save_json

NS = {'gml': 'http://www.opengis.net/gml', 'bldg': 'http://www.opengis.net/citygml/building/1.0',
      'gen': 'http://www.opengis.net/citygml/generics/1.0'}
B = '{%s}' % NS['bldg']; G = '{%s}' % NS['gml']


def ring2d(el):
    pl = el.find('.//gml:posList', NS)
    if pl is None:
        return []
    v = [float(x) for x in pl.text.split()]
    return [[round(v[i] - E0, 2), round(v[i + 1] - N0, 2)] for i in range(0, len(v), 3)]


def gen_attr(el, name):
    for a in el.findall('gen:stringAttribute', NS):
        if a.get('name') == name:
            v = a.find('gen:value', NS)
            return v.text if v is not None else None
    return None


def parse(fn, out, radius=920):
    for _, b in etree.iterparse(fn, tag=B + 'Building', huge_tree=True):
        if b.getparent().tag == B + 'consistsOfBuildingPart':
            continue
        fnc = b.find('bldg:function', NS)
        fnc = fnc.text if fnc is not None else None
        for part in [b] + list(b.iter(B + 'BuildingPart')):
            grounds = []
            for gs in part.findall('bldg:boundedBy/bldg:GroundSurface', NS):
                for poly in gs.iter(G + 'Polygon'):
                    ext = poly.find('gml:exterior', NS)
                    if ext is not None:
                        r = ring2d(ext)
                        if len(r) >= 3:
                            grounds.append(r)
            if not grounds:
                continue
            cx = sum(p[0] for p in grounds[0]) / len(grounds[0]); cy = sum(p[1] for p in grounds[0]) / len(grounds[0])
            if abs(cx) > radius or abs(cy) > radius:
                continue
            rt = part.find('bldg:roofType', NS); mh = part.find('bldg:measuredHeight', NS)
            def f(x):
                try: return float(x)
                except (TypeError, ValueError): return None
            out.append({'bid': b.get(G + 'id'), 'pid': part.get(G + 'id'), 'function': fnc,
                        'roofType': rt.text if rt is not None else None, 'roofName': gen_attr(part, 'DachtypName'),
                        'height': f(mh.text) if mh is not None else None, 'ridge': f(gen_attr(part, 'Firsthoehe')),
                        'eaves': f(gen_attr(part, 'MittlereTraufHoehe')), 'base': f(gen_attr(part, 'AbsoluteHoehe')),
                        'pitch': f(gen_attr(part, 'Dachneigung')), 'grounds': grounds})
        b.clear()


def main():
    files = sorted(glob.glob(os.path.join(RESEARCH, 'lod2', 'LoD2_32_*_ni.gml')))
    out = []
    for fn in files:
        parse(fn, out); print(os.path.basename(fn), len(out))
    save_json({'source': 'LGLN LoD2 (CC BY 4.0)', 'parts': out}, os.path.join(CACHE, 'lgln_parts.json'))
    print('parts', len(out))


if __name__ == '__main__':
    main()

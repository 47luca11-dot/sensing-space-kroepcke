"""Schreibt web/assets/city/manifest.json aus allen vorhandenen City-GLBs (Detailmodell, Wahrzeichen, Gebaeude, Boden).
Bounds aus den unkomprimierten Rohdateien (data/*_raw) bzw. aus den index.json der Generatoren."""
import json, os, glob, trimesh, numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY = os.path.join(R, 'web', 'assets', 'city')
files = []
def bounds_of(path):
    s = trimesh.load(path, force='scene')
    tris = int(sum(len(g.faces) for g in s.geometry.values()))
    return [round(float(v), 2) for v in np.r_[s.bounds[0], s.bounds[1]]], tris
VISUAL_ONLY = {'green_trees', 'lichtwolke'}
for sub, raw, group in (('detail', 'detail_raw', 'detail'), ('landmarks', 'landmarks_raw', 'landmark')):
    for fn in sorted(glob.glob(os.path.join(CITY, sub, '*.glb'))):
        name = os.path.splitext(os.path.basename(fn))[0]
        src = os.path.join(R, 'data', 'detail_final', name + '.glb') if group == 'detail' else ''
        if not os.path.isfile(src): src = os.path.join(R, 'data', raw, name + '.glb')
        b, t = bounds_of(src)
        files.append({'url': f'{sub}/{name}.glb', 'kind': 'visual' if name in VISUAL_ONLY else 'both', 'group': group, 'bounds': b, 'tris': t})
def bl(bd):
    if isinstance(bd, dict) and 'min' in bd: return [round(float(v), 2) for v in list(bd['min']) + list(bd['max'])]
    return bd
for sub in ('buildings', 'ground'):
    idx = os.path.join(CITY, sub, 'index.json')
    if not os.path.isfile(idx): continue
    J = json.load(open(idx, encoding='utf-8'))
    for it in J.get('tiles', []):
        if not os.path.isfile(os.path.join(CITY, sub, it['file'])): continue
        files.append({'url': f"{sub}/{it['file']}", 'kind': 'visual', 'group': sub, 'bounds': bl(it.get('bounds')), 'tris': it.get('triangles')})
        w = it.get('windows')
        if w and os.path.isfile(os.path.join(CITY, sub, w)):     # instanzierte Fenster/Tueren: nur im Nahbereich
            files.append({'url': f"{sub}/{w}", 'kind': 'visual', 'group': 'windows', 'bounds': bl(it.get('bounds')),
                          'tris': it.get('prototype_triangles'), 'cullDist': 230})
    for extra in ('furniture', 'extras'):
        e = J.get(extra)
        for it in (e if isinstance(e, list) else [e] if isinstance(e, dict) else []):
            if it.get('file') and os.path.isfile(os.path.join(CITY, sub, it['file'])):
                files.append({'url': f"{sub}/{it['file']}", 'kind': it.get('kind', 'visual'), 'group': sub, 'bounds': bl(it.get('bounds')), 'tris': it.get('triangles')})
    for fx in ('furniture.glb',):
        if sub == 'ground' and os.path.isfile(os.path.join(CITY, sub, fx)) and not any(f['url'] == f'{sub}/{fx}' for f in files):
            files.append({'url': f'{sub}/{fx}', 'kind': 'visual', 'group': 'furniture', 'bounds': None, 'tris': None})
    c = J.get('collision')
    if isinstance(c, dict) and os.path.isfile(os.path.join(CITY, sub, c['file'])):
        files.append({'url': f"{sub}/{c['file']}", 'kind': 'collision', 'group': sub, 'bounds': None, 'tris': c.get('triangles')})
man = {'version': 1, 'frame': 'metres, Y-up, origin = Kroepcke-Uhr at street level (EPSG:25832 E 550298.88 N 5802940.71, 56.16 m DHHN92), +X east, -Z north',
       'spawn': [0.0, 0.0, 6.0], 'spawnYaw': 0.0, 'spawnYawNote': '0 = looking north (-Z)',
       'files': files,
       'labels': [{'name': 'Kröpcke', 'p': [0, 0]}, {'name': 'Bahnhofstraße', 'p': [40, -60]}, {'name': 'Georgstraße', 'p': [-80, 10]},
                  {'name': 'Opernplatz', 'p': [110, 120]}, {'name': 'Opernhaus', 'p': [142, 114]}, {'name': 'Karmarschstraße', 'p': [-40, 120]},
                  {'name': 'Rathenaustraße', 'p': [70, 20]}, {'name': 'Café Kröpcke', 'p': [29, 20]}, {'name': 'Kröpcke-Center', 'p': [-20, 70]}],
       'clock': {'url': 'landmarks/kroepcke_uhr.glb', 'hands': 'nodes named Zeiger_hour_<k> / Zeiger_minute_<k>; extras.face_normal_game = axis; rotate by -angle about it (clockwise seen from front); rest pose = 12:00'},
       'attribution': 'Gebäude/Gelände: © GeoBasis-DE/LGLN 2026, CC BY 4.0 (verändert); 3D-Stadtmodell © Landeshauptstadt Hannover, CC BY 4.0; Kartendaten © OpenStreetMap-Mitwirkende (ODbL)'}
old = os.path.join(CITY, 'manifest.json')
if os.path.isfile(old):                                   # eigene Scans (Splats) beibehalten
    try: man['scans'] = json.load(open(old, encoding='utf-8')).get('scans', [])
    except Exception: man['scans'] = []
else: man['scans'] = []
hf = os.path.join(CITY, 'ground', 'heightfield.json')
if os.path.isfile(hf) and os.path.isfile(os.path.join(CITY, 'ground', 'heightfield.bin')): man['heightfield'] = {'bin': 'ground/heightfield.bin', 'json': 'ground/heightfield.json'}
json.dump(man, open(os.path.join(CITY, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('manifest files', len(files), 'tris', sum(f['tris'] or 0 for f in files))

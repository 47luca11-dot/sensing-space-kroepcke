"""Stadtmoeblierung als instanzierte Meshes (EXT_mesh_gpu_instancing): Leuchten (aus dem Detailmodell), Baenke,
Abfallbehaelter, Fahrradbuegel, Poller, Litfasssaeulen, Wartehallen, Haltestellenmasten, Ampeln, Briefkaesten,
Baeume (Kreuz-Billboards + Kronenkarte + Stamm). Ausgabe: furniture.glb + furniture.json"""
import os, sys, json, time, pickle, math, collections
import numpy as np
import shapely
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from pyproj import Transformer
import trimesh
from common import *
from glb import GLB, quat_y
from materials import MATS
from surface import Surface
import geom as G

T0 = time.time()
def log(*a): print('[furn %6.1fs]' % (time.time() - T0), *a, flush=True)
rng = np.random.default_rng(42)
L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
TP = pickle.load(open(CACHE + 'terrain_parts.pkl', 'rb'))
SURF = Surface()
REG = L['region']; BLD = L['bld']; SQB = shapely.box(*SQ)
HOLES = shapely.unary_union([h['poly'].buffer(0.6) for h in L['holes'].values()])
for g in (REG, BLD, HOLES): shapely.prepare(g)
FREE_CLOCK = Point(0, 0).buffer(3.0)

# ---------------------------------------------------------------- OSM-Knoten (ausfuehrliche Abfrage)
tr_ = Transformer.from_crs(4326, 25832, always_xy=True)
raw = json.load(open(DATA + 'osm/kroepcke_osm_details.json', encoding='utf-8'))
nodes = []
for e in raw['elements']:
    if e['type'] == 'node' and e.get('tags'):
        E, N = tr_.transform(e['lon'], e['lat'])
        nodes.append({'id': e['id'], 'p': np.array([E - E0, -(N - N0)]), 'tags': e['tags']})
# Ergaenzung aus osm_features (Baeume usw.), Duplikate < 0.5 m verwerfen
osm = load_osm()
have = cKDTree(np.array([n['p'] for n in nodes]))
# Ausbau 2 (Radius 860 m): Stadtmoebel im ganzen Gebiet aus osm_features ergaenzen (nur Strassenebene)
SUPP = (('natural', 'tree'), ('highway', 'street_lamp'), ('amenity', 'bench'), ('amenity', 'waste_basket'), ('amenity', 'bicycle_parking'),
        ('highway', 'bus_stop'), ('barrier', 'bollard'), ('advertising', 'column'), ('amenity', 'post_box'), ('man_made', 'flagpole'),
        ('man_made', 'street_cabinet'))
nsup = 0
for n in osm['nodes']:
    t = n['tags']
    if not any(t.get(k) == v for k, v in SUPP): continue
    if str(t.get('level', '0')).startswith('-') or str(t.get('layer', '0')).startswith('-') or t.get('indoor') == 'yes': continue
    if have.query(n['p'])[0] > 0.5: nodes.append({'id': n['id'], 'p': np.array(n['p']), 'tags': t}); nsup += 1
log('supplemented from osm_features', nsup)
# Flaechen anderer Bauwerke (Hochbahnsteige, Mendini-Haltestelle, Brunnen) freihalten
EXCL = Polygon()
try:
    _tp = pickle.load(open(CACHE + 'transit.pkl', 'rb'))
    EXCL = shapely.unary_union([Polygon(r).buffer(0.4) for r in _tp.get('occupied', [])])
except Exception as e: log('no transit.pkl', e)
try:
    _al = pickle.load(open(CACHE + 'altstadt.pkl', 'rb'))
    EXCL = shapely.unary_union([EXCL] + [Point(f_['p']).buffer(4.6 if 'Holz' in f_['name'] else 3.6) for f_ in _al['info_extra']['fountains']])
except Exception as e: log('no altstadt.pkl', e)
shapely.prepare(EXCL)
MENDINI_ZONE = shapely.unary_union([Polygon(w['pts']).buffer(18) for w in osm['ways'] if w['tags'].get('name') == 'BUSSTOPS' and len(w['pts']) >= 4]) if True else Polygon()
log('osm nodes', len(nodes))

def ok_spot(p, r=0.3, classes_ok=None):
    pt = Point(p)
    if not shapely.contains_xy(REG, p[0], p[1]): return False
    if SQB.buffer(-0.5).contains(pt) or FREE_CLOCK.contains(pt): return False
    if HOLES.contains(pt) or BLD.contains(pt) or (r > 0 and shapely.dwithin(BLD, pt, r)): return False
    if not EXCL.is_empty and shapely.dwithin(EXCL, pt, r): return False
    if classes_ok is not None:
        h, c = SURF([p[0]], [p[1]])
        if c[0] not in classes_ok: return False
    return True

RAISED = {'sidewalk', 'ped', 'kerb', 'path', 'gravel', 'grass', 'bed', 'island', 'cycle', 'edge', 'pitframe', 'treepit', 'coping', 'railplat', 'hbs'}
WALK = {'sidewalk', 'ped', 'path', 'gravel', 'island', 'cycle', 'railplat', 'hbs', 'coping'}

# Richtungsfeld (Achsen)
axl = [np.array(c) for c, w, t in L['road_lines']] + [np.array(c) for c, t in L['ped_lines']] + [np.array(c) for c, t in L['foot_lines']] + [np.array(c) for c, t in L['side_lines']]
AP = []; AA = []
for c in axl:
    for i in range(len(c) - 1):
        d = c[i + 1] - c[i]; l = np.linalg.norm(d)
        if l < 1e-6: continue
        n = max(1, int(l / 1.5))
        for t in (np.arange(n) + 0.5) / n: AP.append(c[i] + d * t); AA.append(math.atan2(d[1], d[0]))
AP = np.array(AP); AA = np.array(AA); kdA = cKDTree(AP)
def axis_dir(p):
    _, i = kdA.query(p); a = AA[i]; return np.array([math.cos(a), math.sin(a)])
# Bordkante (Grenze Fahrbahn) fuer "zur Strasse hin"
LOWB = L['low'].boundary
lb = []
for g in getattr(LOWB, 'geoms', [LOWB]):
    c = np.asarray(g.coords)
    for i in range(len(c) - 1):
        d = c[i + 1] - c[i]; l = np.linalg.norm(d)
        n = max(1, int(l / 0.7))
        for t in (np.arange(n) + 0.5) / n: lb.append(c[i] + d * t)
lb = np.array(lb); kdK = cKDTree(lb)
def to_road(p):
    d, i = kdK.query(p); v = lb[i] - p; n = np.linalg.norm(v)
    return (v / n if n > 1e-6 else axis_dir(p)), d

def yaw_to(d):
    """Drehwinkel um Y, der lokales +x auf Weltrichtung d=(dx,dz) dreht"""
    return math.atan2(-d[1], d[0])

INST = collections.defaultdict(list)      # proto -> [(x,y,z,yaw,scale(3))]
def place(proto, p, yaw, s=(1, 1, 1), y=None):
    if y is None: y = SURF([p[0]], [p[1]])[0][0]
    INST[proto].append((p[0], y, p[1], yaw, s[0], s[1], s[2]))
occupied = []
def free(p, r):
    for q, rr in occupied:
        if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < (r + rr) ** 2: return False
    return True

# ---------------------------------------------------------------- Prototypen aus dem Detailmodell
def detail_protos():
    sc = trimesh.load(DATA + 'rhino_export/street_furniture.glb', force='scene')
    parts = []
    for name, geo in sc.geometry.items():
        Tm = sc.graph.get(sc.graph.geometry_nodes[name][0])[0]
        m = geo.copy(); m.apply_transform(Tm); m.vertices += RHINO_OFF
        parts.append((m, str(getattr(geo.visual.material, 'name', ''))))
    bb = np.array([[*p.bounds[0], *p.bounds[1]] for p, _ in parts])
    big = (bb[:, 3] - bb[:, 0] > 2.5) | (bb[:, 5] - bb[:, 2] > 2.5)
    par = list(range(len(parts)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    e = 0.02
    for i in range(len(parts)):
        if big[i]: continue
        ov = (bb[:, 0] - e <= bb[i, 3]) & (bb[:, 3] + e >= bb[i, 0]) & (bb[:, 2] - e <= bb[i, 5]) & (bb[:, 5] + e >= bb[i, 2]) & ~big
        for j in np.nonzero(ov)[0]:
            a, b = f(i), f(j)
            if a != b: par[a] = b
    cl = collections.defaultdict(list)
    for i in range(len(parts)):
        if not big[i]: cl[f(i)].append(i)
    want = {'lamp_mast': (8.0, 1576), 'lamp_lantern': (4.0, 3204), 'bench': (0.5, 556)}
    found = {}
    for k, idx in cl.items():
        lo = bb[idx, :3].min(0); hi = bb[idx, 3:].max(0); tris = sum(len(parts[i][0].faces) for i in idx)
        for nm, (h, t) in want.items():
            if nm not in found and abs((hi - lo)[1] - h) < 0.15 and tris == t: found[nm] = idx
    MAP = {'Light': 'lamp_glow', 'Beech red polished': 'wood'}
    import pymeshlab
    out = {}
    for nm, idx in found.items():
        bym = collections.defaultdict(list)
        for i in idx:
            m, mn = parts[i]
            bym[MAP.get(mn, 'steel_dark')].append(m)
        allv = np.concatenate([parts[i][0].vertices for i in idx])
        base = allv[allv[:, 1] < allv[:, 1].min() + 0.4]
        org = np.array([base[:, 0].mean(), allv[:, 1].min(), base[:, 2].mean()])
        ps = []
        for mat, ms in bym.items():
            m = trimesh.util.concatenate(ms); m.merge_vertices()
            target = {'lamp_mast': 260, 'lamp_lantern': 300, 'bench': 260}[nm] * len(m.faces) / sum(len(parts[i][0].faces) for i in idx)
            if len(m.faces) > max(40, target * 1.3):
                ms_ = pymeshlab.MeshSet(); ms_.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
                ms_.meshing_decimation_quadric_edge_collapse(targetfacenum=int(max(24, target)), preservenormal=True, preservetopology=True, qualitythr=0.4)
                mm = ms_.current_mesh(); m = trimesh.Trimesh(mm.vertex_matrix(), mm.face_matrix(), process=False)
            v = m.vertices - org
            # UV: Box-Projektion
            uv = np.c_[v[:, 0] + v[:, 2], v[:, 1]] * (2.0 if mat == 'wood' else 1.0)
            ps.append(G.part(mat, v, m.faces, None, uv))
        # Ausrichtung: Mast-Leuchte -> Kopf nach +x; Bank -> Laengsachse x
        allp = np.concatenate([p['pos'] for p in ps])
        if nm == 'lamp_mast':
            top = allp[allp[:, 1] > allp[:, 1].max() - 0.8]
            d = top[:, [0, 2]].mean(0); a = math.atan2(d[1], d[0])
            R = G.rot_y(a)          # rot_y(a): x -> (cos a, 0, -sin a); wir wollen d -> +x
            R = G.rot_y(-math.atan2(-d[1], d[0]))
            ps = [G.transform(p, R) for p in ps]
        elif nm == 'bench':
            xz = allp[:, [0, 2]] - allp[:, [0, 2]].mean(0)
            U, S, Vt = np.linalg.svd(xz, full_matrices=False); d = Vt[0]
            R = G.rot_y(-math.atan2(-d[1], d[0])); ps = [G.transform(p, R) for p in ps]
            c = np.concatenate([p['pos'] for p in ps]).mean(0); ps = [G.transform(p, None, (-c[0], 0, -c[2])) for p in ps]
        out[nm] = G.merge(ps)
        log('detail proto', nm, 'tris', G.tri_count(out[nm]), 'size', np.ptp(np.concatenate([p['pos'] for p in out[nm]]), 0).round(2))
    return out

PROTO = detail_protos()

# ---------------------------------------------------------------- Prozedurale Prototypen
def proto_bin():
    return G.merge([G.cylinder('bin_grey', (0, 0, 0), 0.06, 0.06, 0.25, 8, False), G.cylinder('bin_grey', (0, 0.25, 0), 0.22, 0.23, 0.62, 10),
                    G.cylinder('steel_dark', (0, 0.87, 0), 0.25, 0.24, 0.06, 10)])
def proto_bike():
    a = np.linspace(0, math.pi, 7)
    pts = [[-0.4, 0.0, 0]] + [[-0.4 + 0.4 * (1 - math.cos(t)) * 0 + -0.32 * math.cos(t) * 0 + (-0.4 * math.cos(t)), 0.72 + 0.18 * math.sin(t), 0] for t in a] + [[0.4, 0.0, 0]]
    pts = np.array(pts); pts[1:-1, 0] = -0.4 * np.cos(a)
    return [G.tube('steel', pts, 0.024, 6)]
def proto_bollard():
    return G.merge([G.cylinder('steel_dark', (0, 0, 0), 0.085, 0.085, 0.86, 10), G.cylinder('steel', (0, 0.72, 0), 0.087, 0.087, 0.05, 10, False),
                    G.cylinder('steel_dark', (0, 0.86, 0), 0.085, 0.03, 0.06, 10)])
def proto_litfass():
    c = G.cylinder('poster', (0, 0.35, 0), 0.62, 0.62, 2.35, 16, False, uvscale=1.0)
    c['uv'][:, 0] = np.arctan2(c['pos'][:, 2], c['pos'][:, 0]) / (2 * math.pi) * 3 % 1.0
    # Naht der Abwicklung vermeiden: einfach linear ueber den Winkelindex
    n = 17; ang = np.linspace(0, 1, n)
    c['uv'][:n, 0] = ang * 3; c['uv'][n:2 * n, 0] = ang * 3; c['uv'][:n, 1] = 1.0; c['uv'][n:2 * n, 1] = 0.0
    return G.merge([G.cylinder('paint_green', (0, 0, 0), 0.68, 0.66, 0.35, 16), c, G.cylinder('paint_green', (0, 2.7, 0), 0.72, 0.72, 0.12, 16),
                    G.cylinder('paint_green', (0, 2.82, 0), 0.66, 0.05, 0.45, 16), G.cylinder('paint_green', (0, 3.27, 0), 0.05, 0.0, 0.25, 8)])
def proto_shelter():
    """Wartehalle 4 x 1.6 m, offen nach +x (Strasse)"""
    ps = []
    for x in (-0.75, 0.75):
        for z in (-1.95, 1.95):
            ps.append(G.box('steel_dark', (x, 0, z), (0.08, 2.45, 0.08)))
    ps.append(G.box('steel_dark', (0, 2.45, 0), (1.9, 0.12, 4.3)))
    glass = []
    q = lambda a, b, c, d: [a, b, c, d]
    glass.append([[-0.75, 0.1, -1.95], [-0.75, 0.1, 1.95], [-0.75, 2.4, 1.95], [-0.75, 2.4, -1.95]])
    glass.append([[-0.75, 0.1, -1.95], [0.6, 0.1, -1.95], [0.6, 2.4, -1.95], [-0.75, 2.4, -1.95]])
    glass.append([[0.6, 0.1, 1.95], [-0.75, 0.1, 1.95], [-0.75, 2.4, 1.95], [0.6, 2.4, 1.95]])
    ps.append(G.flat_quads('glass', np.array(glass)))
    ps.append(G.box('wood', (-0.5, 0.42, 0), (0.4, 0.05, 2.0)))
    ps.append(G.box('steel_dark', (-0.5, 0, -0.8), (0.06, 0.42, 0.3))); ps.append(G.box('steel_dark', (-0.5, 0, 0.8), (0.06, 0.42, 0.3)))
    return G.merge(ps)
def proto_sign(mat):
    ps = [G.cylinder('steel', (0, 0, 0), 0.04, 0.04, 2.6, 8)]
    w = 0.25; c = np.array([0, 2.45, 0])
    for sgn in (1, -1):
        pos = np.array([[0.05 * sgn, -w, -w * sgn], [0.05 * sgn, -w, w * sgn], [0.05 * sgn, w, w * sgn], [0.05 * sgn, w, -w * sgn]]) + c
        ps.append(G.part(mat, pos, [(0, 1, 2), (0, 2, 3)], np.tile([sgn, 0, 0], (4, 1)), np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)))
    ps.append(G.box('steel_dark', (0, 2.45 - w - 0.01, 0), (0.09, 2 * w + 0.02, 2 * w + 0.02)))
    return G.merge(ps)
def proto_signal():
    """Ampelmast 3.2 m, Signalgeber blickt nach +x"""
    ps = [G.cylinder('paint_green', (0, 0, 0), 0.07, 0.065, 3.3, 8) if False else G.cylinder('steel_dark', (0, 0, 0), 0.07, 0.065, 3.3, 8)]
    ps.append(G.box('steel_dark', (0.12, 2.2, 0), (0.18, 0.95, 0.32)))
    for i, m in enumerate(('signal_r', 'signal_y', 'signal_g')):
        ps.append(G.cylinder(m, (0.215, 2.98 - i * 0.3, 0), 0.09, 0.09, 0.01, 10) if False else G.box(m, (0.215, 2.86 - i * 0.3, 0), (0.02, 0.2, 0.2)))
    ps.append(G.box('steel_dark', (0.1, 1.7, 0), (0.14, 0.38, 0.22)))
    ps.append(G.box('signal_r', (0.175, 1.88 - 0.17, 0), (0.02, 0.12, 0.12)))
    ps.append(G.box('bin_grey', (-0.12, 0.95, 0), (0.12, 0.2, 0.16)))
    return G.merge(ps)
def proto_postbox():
    return G.merge([G.cylinder('steel_dark', (0, 0, 0), 0.04, 0.04, 0.85, 6), G.box('postbox', (0, 0.75, 0), (0.32, 0.62, 0.42))])
def proto_flag():
    return [G.cylinder('steel', (0, 0, 0), 0.06, 0.035, 9.0, 8)]
def proto_cabinet():
    return [G.box('bin_grey', (0, 0, 0), (0.35, 1.2, 0.9))]

def tree_top_texture():
    from PIL import Image
    src = Image.open(OUT + 'tex/tree_b.png').convert('RGBA'); a = np.array(src)
    al = a[..., 3] > 128
    ys, xs = np.nonzero(al)
    crown = ys < ys.min() + 0.6 * (ys.max() - ys.min())
    cy, cx = ys[crown], xs[crown]
    can = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    r_ = np.random.default_rng(3)
    for k in range(500):
        i = r_.integers(len(cy)); y0, x0 = int(cy[i]) - 48, int(cx[i]) - 48
        if y0 < 0 or x0 < 0 or y0 + 96 > a.shape[0] or x0 + 96 > a.shape[1]: continue
        patch = Image.fromarray(a[y0:y0 + 96, x0:x0 + 96]).rotate(r_.uniform(0, 360))
        rr = 200 * math.sqrt(r_.random()); t = r_.uniform(0, 2 * math.pi)
        can.alpha_composite(patch, (int(256 + rr * math.cos(t) - 48), int(256 + rr * math.sin(t) - 48)))
    tmp = OUT + 'tex/.tree_top.png.tmp'; can.save(tmp, format='PNG', optimize=True); os.replace(tmp, OUT + 'tex/tree_top.png')

TREE_META = json.load(open(CACHE + 'tree_meta.json'))
def card2(mat, pos, uv, centre, lift=0.25):
    """Blattkarte als zwei einseitige Flaechen (Vorder-/Rueckseite) mit gleichen, kugelfoermig nach aussen
    zeigenden Normalen: keine Normalenumkehr auf Rueckseiten (sonst schwarz in three.js bei doubleSided)."""
    pos = np.asarray(pos, float)
    n = pos - centre + np.array([0, lift, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    P = np.concatenate([pos, pos]); N = np.concatenate([n, n]); UVv = np.concatenate([uv, uv])
    idx = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
    return G.part(mat, P, idx, N, UVv)

def proto_tree(mat, crown_w, seed):
    """Baum der Einheitshoehe 1 (Instanz: sx=sz=Breite, sy=Hoehe): 4 gekreuzte Kronen-Billboards (Stamm aus der
    Textur entfernt), 12 tangentiale Blattbueschel-Karten auf dem Kronen-Ellipsoid, 3D-Stamm mit zwei Aesten.
    Alle Karten: alphaMode MASK (0.5), Vorder- und Rueckseite als eigene Dreiecke mit Kugel-Normalen."""
    r_ = np.random.default_rng(seed)
    cb = TREE_META.get(mat + '.png', {}).get('crown_base', 0.2)
    y0 = max(cb - 0.03, 0.0)
    c = np.array([0, cb + (1 - cb) * 0.48, 0])
    ps = []
    for k in range(4):
        a = k * math.pi / 4; d = np.array([math.cos(a), 0, -math.sin(a)]) * 0.5
        pos = np.array([-d + [0, y0, 0], d + [0, y0, 0], d + [0, 1, 0], -d + [0, 1, 0]])
        uv = np.array([[0, 1 - y0], [1, 1 - y0], [1, 0], [0, 0]], float)
        ps.append(card2(mat, pos, uv, c))
    # Blattbueschel auf dem Kronen-Ellipsoid (Fibonacci-Verteilung, oben dichter)
    rx = crown_w * 0.36; ry = (1 - cb) * 0.36
    n_c = 12; ga = math.pi * (3 - math.sqrt(5))
    for i in range(n_c):
        yv = 1 - (i + 0.5) / n_c * 1.35          # von oben bis leicht unter den Aequator
        yv = max(min(yv, 0.98), -0.45); rr = math.sqrt(max(0, 1 - yv * yv)); th = i * ga + r_.uniform(0, 0.4)
        dirv = np.array([rr * math.cos(th), yv, rr * math.sin(th)])
        ctr = c + dirv * np.array([rx, ry, rx])
        nrm = dirv * np.array([1 / rx, 1 / ry, 1 / rx]); nrm /= np.linalg.norm(nrm)
        t1 = np.cross(nrm, [0, 1, 0]); 
        if np.linalg.norm(t1) < 1e-3: t1 = np.array([1.0, 0, 0])
        t1 /= np.linalg.norm(t1); t2 = np.cross(nrm, t1)
        rot = r_.uniform(0, 2 * math.pi); t1, t2 = t1 * math.cos(rot) + t2 * math.sin(rot), -t1 * math.sin(rot) + t2 * math.cos(rot)
        h = crown_w * r_.uniform(0.22, 0.28)
        pos = np.array([ctr - t1 * h - t2 * h, ctr + t1 * h - t2 * h, ctr + t1 * h + t2 * h, ctr - t1 * h + t2 * h])
        uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
        ps.append(card2('tree_top', pos, uv, c, 0.15))
    ps.append(G.cylinder('bark_plane', (0, 0, 0), 0.016, 0.010, cb + 0.08, 8, False))
    for k, ang in enumerate((0.4, 2.6)):
        p0 = np.array([0, cb * 0.9, 0]); p1 = p0 + np.array([math.cos(ang) * 0.12, 0.16, -math.sin(ang) * 0.12])
        ps.append(G.tube('bark_plane', np.array([p0, p1]), 0.006, 5))
    return G.merge(ps)

tree_top_texture()
PROTOS = {'lamp_mast': PROTO.get('lamp_mast'), 'lamp_lantern': PROTO.get('lamp_lantern'), 'bench': PROTO.get('bench'), 'bin': proto_bin(),
          'bike_stand': proto_bike(), 'bollard': proto_bollard(), 'litfass': proto_litfass(), 'shelter': proto_shelter(),
          'stop_sign': proto_sign('sign_h'), 'signal': proto_signal(), 'postbox': proto_postbox(), 'flagpole': proto_flag(), 'cabinet': proto_cabinet(),
          'tree_a': proto_tree('tree_a', 0.62, 1), 'tree_b': proto_tree('tree_b', 0.85, 2)}
MATS = dict(MATS); MATS['postbox'] = {'color': (0.85, 0.62, 0.02, 1), 'rough': 0.4}

# ---------------------------------------------------------------- Platzierung
TAG = lambda n, k, v=None: (n['tags'].get(k) == v) if v is not None else (k in n['tags'])
cnt = collections.Counter()
# Baeume zuerst (Platzbedarf)
for n in nodes:
    if not TAG(n, 'natural', 'tree'): continue
    p = n['p']
    if not ok_spot(p, 0.3): continue
    h, c = SURF([p[0]], [p[1]])
    if c[0] in ('road', 'gutter', 'parking', 'setts_road', 'track', 'kerb', 'kerb_ch', 'water', 'rim', 'hbs', ''): continue
    genus = (n['tags'].get('genus') or n['tags'].get('species') or '').lower()
    proto = 'tree_a' if ('platanus' in genus or rng.random() < 0.45) else 'tree_b'
    H = float(n['tags'].get('height', 0) or 0) or rng.uniform(11, 17)
    H = min(max(H, 6), 24)
    s = H * rng.uniform(0.92, 1.08)
    place(proto, p, rng.uniform(0, 2 * math.pi), (s, H, s), h[0] - 0.02)
    occupied.append((p, 0.9)); cnt['tree'] += 1
log('trees', cnt['tree'])

lamp_pts = []
for n in nodes:
    t = n['tags']
    p = n['p']
    if t.get('highway') == 'street_lamp':
        if not ok_spot(p, 0.2): continue
        h, c = SURF([p[0]], [p[1]])
        if c[0] in ('road', 'gutter', 'parking', 'setts_road', 'track'):
            # auf die Gehwegseite verschieben
            d, dist = to_road(p); p = p - d * (dist + 0.7); h, c = SURF([p[0]], [p[1]])
        d, dist = to_road(p)
        if c[0] == 'ped' and dist > 4:
            place('lamp_lantern', p, rng.uniform(0, 2 * math.pi)); cnt['lantern'] += 1
        else:
            place('lamp_mast', p, yaw_to(d)); cnt['lamp'] += 1
        occupied.append((p, 0.5)); lamp_pts.append(p)
# zusaetzliche Mastleuchten entlang der Strassen (wo OSM keine hat)
kdL = cKDTree(np.array(lamp_pts)) if lamp_pts else None
LOWP = L['low']; shapely.prepare(LOWP)
for c, w, t in L['road_lines']:
    if t['highway'] not in ('primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'living_street', 'secondary_link', 'tertiary_link', 'busway'): continue
    Ls = LineString(c)
    side = 1
    s = 12.0
    while s < Ls.length - 6:
        a = np.array(Ls.interpolate(s - 1).coords[0]); b = np.array(Ls.interpolate(s + 1).coords[0]); m = np.array(Ls.interpolate(s).coords[0])
        dd = (b - a) / (np.linalg.norm(b - a) + 1e-9); nrm = np.array([-dd[1], dd[0]]) * side
        # Bordkante entlang der Normalen suchen
        ray = LineString([m, m + nrm * 25]).intersection(LOWP)
        q = None
        if not ray.is_empty:
            segs = [np.array(g.coords) for g in getattr(ray, 'geoms', [ray]) if g.geom_type == 'LineString']
            for sg in segs:
                ss = (sg - m) @ nrm
                if ss.min() < 0.5: q = m + nrm * (ss.max() + 0.75)
        if q is not None and (kdL is None or kdL.query(q)[0] > 18) and ok_spot(q, 0.6, WALK | {'kerb'}) and free(q, 1.2):
            hq, cq = SURF([q[0]], [q[1]])
            if cq[0] in WALK:
                place('lamp_mast', q, yaw_to(-nrm)); occupied.append((q, 0.5)); lamp_pts.append(q); cnt['lamp_synth'] += 1
                kdL = cKDTree(np.array(lamp_pts))
        s += 30.0; side = -side
log('lamps', cnt['lamp'], 'lanterns', cnt['lantern'], 'synthetic', cnt['lamp_synth'])

for n in nodes:
    t = n['tags']; p = n['p']
    if t.get('amenity') == 'bench' or (t.get('leisure') == 'picnic_table'):
        if not ok_spot(p, 0.4, RAISED) or not free(p, 0.6): continue
        place('bench', p, yaw_to(axis_dir(p))); occupied.append((p, 0.9)); cnt['bench'] += 1
    elif t.get('amenity') == 'waste_basket':
        if not ok_spot(p, 0.2, RAISED) or not free(p, 0.3): continue
        place('bin', p, rng.uniform(0, 6.28)); occupied.append((p, 0.3)); cnt['bin'] += 1
    elif t.get('amenity') == 'bicycle_parking':
        if not ok_spot(p, 0.5, RAISED): continue
        cap = t.get('capacity', '4'); cap = int(cap) if str(cap).isdigit() else 4
        k = int(min(max(math.ceil(cap / 2), 1), 10))
        d = axis_dir(p); nrm = np.array([-d[1], d[0]])
        for i in range(k):
            q = p + d * (i - (k - 1) / 2) * 1.0
            if ok_spot(q, 0.4, RAISED) and free(q, 0.35):
                place('bike_stand', q, yaw_to(nrm)); occupied.append((q, 0.35)); cnt['bike'] += 1
    elif t.get('barrier') == 'bollard':
        if not ok_spot(p, 0.1) or not free(p, 0.15): continue
        place('bollard', p, 0.0); occupied.append((p, 0.15)); cnt['bollard'] += 1
    elif t.get('advertising') == 'column':
        if not ok_spot(p, 0.8, RAISED) or not free(p, 0.8): continue
        place('litfass', p, rng.uniform(0, 6.28)); occupied.append((p, 0.9)); cnt['litfass'] += 1
    elif t.get('highway') == 'bus_stop' or (t.get('public_transport') == 'platform' and t.get('railway') is None and t.get('bus') == 'yes'):
        if MENDINI_ZONE.contains(Point(p)): continue          # eigene Haltestelle (transit.py)
        if not ok_spot(p, 0.3): continue
        h, c = SURF([p[0]], [p[1]])
        d, dist = to_road(p)
        if c[0] in ('road', 'gutter', 'parking', 'setts_road', 'track') or dist < 0.5:
            p = p - d * (max(dist, 0) + 0.8); d, dist = to_road(p)
        if not ok_spot(p, 0.3, WALK | {'kerb'}): continue
        place('stop_sign', p + d * 0.0, yaw_to(np.array([-d[1], d[0]]))); occupied.append((p, 0.3)); cnt['stop'] += 1
        if t.get('shelter') == 'yes' and not MENDINI_ZONE.contains(Point(p)):
            q = p - d * 1.6
            if ok_spot(q, 0.6, WALK) and free(q, 1.0):
                place('shelter', q, yaw_to(d)); occupied.append((q, 2.0)); cnt['shelter'] += 1
    elif t.get('amenity') == 'post_box':
        if ok_spot(p, 0.3, RAISED) and free(p, 0.3): place('postbox', p, yaw_to(axis_dir(p))); occupied.append((p, 0.3)); cnt['postbox'] += 1
    elif t.get('man_made') == 'flagpole':
        if ok_spot(p, 0.3) and free(p, 0.2): place('flagpole', p, 0.0); cnt['flag'] += 1
    elif t.get('man_made') == 'street_cabinet':
        if ok_spot(p, 0.4, RAISED) and free(p, 0.5): place('cabinet', p, yaw_to(axis_dir(p))); occupied.append((p, 0.5)); cnt['cabinet'] += 1
# Poller-Linien
for w in osm['ways']:
    if w['tags'].get('barrier') != 'bollard' or len(w['pts']) < 2: continue
    Ls = LineString(w['pts'])
    for s in np.arange(0.4, Ls.length, 1.6):
        q = np.array(Ls.interpolate(s).coords[0])
        if ok_spot(q, 0.1) and free(q, 0.15): place('bollard', q, 0.0); occupied.append((q, 0.15)); cnt['bollard'] += 1
# Ampeln an signalisierten Ueberwegen: an beiden Bordkanten
for p, d, nrm, e0, e1, kind in TP['crossings']:
    if kind != 'traffic_signals': continue
    p = np.array(p); d = np.array(d); nrm = np.array(nrm)
    for e, sgn in ((e0, -1), (e1, 1)):
        q = p + nrm * (e + sgn * 0.55) - d * 2.6
        if ok_spot(q, 0.2, WALK | {'kerb', 'grass', 'edge', 'bed'}) and free(q, 0.3):
            place('signal', q, yaw_to(-nrm * sgn)); occupied.append((q, 0.3)); cnt['signal'] += 1
log('placed', dict(cnt))

# ---------------------------------------------------------------- Schreiben
g = GLB(); meta = {}
tot = 0
for name, inst in INST.items():
    if not inst or PROTOS.get(name) is None: continue
    A = np.array(inst)
    mi = g.mesh(name, [dict(p) for p in PROTOS[name]], MATS)
    Tt = A[:, :3]; Rq = quat_y(A[:, 3]); Ss = A[:, 4:7]
    g.node(name, mi, instances=(Tt, Rq, Ss), extras={'count': len(A)})
    pt = G.tri_count(PROTOS[name])
    meta[name] = {'count': int(len(A)), 'tris_per_instance': pt}
    tot += pt * len(A)
sz = g.save(staging('furniture.glb'))
json.dump({'instances': {k: [[round(v, 3) for v in r] for r in INST[k]] for k in INST}, 'fields': 'x, y, z, yaw(rad, about +Y), sx, sy, sz',
           'prototypes': meta, 'instanced_triangles': int(tot)}, open(staging('furniture.json'), 'w'))
publish(staging('furniture.glb'), 'furniture.glb'); publish(staging('furniture.json'), 'furniture.json')
log('furniture.glb', round(sz / 1e6, 2), 'MB; instanced triangles', tot, meta)

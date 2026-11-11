"""Kacheln (100 x 100 m) + Kollisionsmesh + Hoehenraster + index.json schreiben."""
import os, sys, json, time, pickle, math, glob
import numpy as np
import shapely
from shapely.ops import unary_union
from shapely.geometry import MultiLineString
import triangle as tr
from scipy.spatial import cKDTree
from common import *
from glb import GLB
from materials import MATS
import geom as G

T0 = time.time()
def log(*a): print('[export %6.1fs]' % (time.time() - T0), *a, flush=True)
TILE = 100.0
DRY = '--dry' in sys.argv            # Probelauf: nur nach data/ground_cache/dry/ schreiben (Vorschau), nichts veroeffentlichen
if DRY:
    import shutil as _sh
    DRYD = CACHE + 'dry/'
    if os.path.isdir(DRYD): _sh.rmtree(DRYD)
    os.makedirs(DRYD); _sh.copytree(OUT + 'tex', DRYD + 'tex')
    def staging(name): return DRYD + name
os.makedirs(OUT, exist_ok=True)
old_tiles = set(os.path.basename(f) for f in glob.glob(OUT + 'ground_*.glb'))
new_files = []

TM = np.load(CACHE + 'terrain.npz', allow_pickle=False)
OV, ON, OUV, OT, tmat = TM['OV'], TM['ON'], TM['OUV'], TM['OT'], TM['tmat']
TPp = pickle.load(open(CACHE + 'terrain_parts.pkl', 'rb'))
ST = pickle.load(open(CACHE + 'structures.pkl', 'rb'))
L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
# Ausbau 2: weitere Bauwerks-Module (je parts / coll / walk)
MODS = {}
for nm in ('ubahn', 'promenade', 'transit', 'altstadt'):
    if os.path.exists(CACHE + nm + '.pkl'):
        MODS[nm] = pickle.load(open(CACHE + nm + '.pkl', 'rb'))
        log('module', nm, 'parts', len(MODS[nm]['parts']), 'tris', G.tri_count(MODS[nm]['parts']))
MATS = dict(MATS)
for M_ in MODS.values(): MATS.update(M_.get('mats', {}))
ALL_COLL = list(ST['coll']) + [p for M_ in MODS.values() for p in M_['coll']]
ALL_WALK = [p for M_ in MODS.values() for p in M_.get('walk', [])]

# ---------------------------------------------------------------- alle Dreiecke als (Material, Kachel) sammeln
def tkey(cx, cz): return np.floor(cx / TILE).astype(int), np.floor(cz / TILE).astype(int)
tiles = {}          # (tx,tz) -> mat -> list of (pos, nrm, uv, idx)
def push(tx, tz, mat, pos, nrm, uv, idx):
    tiles.setdefault((int(tx), int(tz)), {}).setdefault(mat, []).append((pos, nrm, uv, idx))

cen = OV[OT].mean(1)
kx, kz = tkey(cen[:, 0], cen[:, 2])
tile_id = kx * 1000 + kz
for tid in np.unique(tile_id):
    sel = tile_id == tid
    for m in np.unique(tmat[sel]):
        s2 = sel & (tmat == m)
        tri = OT[s2]
        used, inv = np.unique(tri, return_inverse=True)
        tx = int(np.floor(cen[s2][0, 0] / TILE)); tz = int(np.floor(cen[s2][0, 2] / TILE))
        push(tx, tz, str(m), OV[used], ON[used], OUV[used], inv.reshape(-1, 3))
log('terrain tiles', len(tiles))

def push_part(p):
    if p is None or len(p['idx']) == 0: return
    c = p['pos'][p['idx']].mean(1)
    kx, kz = tkey(c[:, 0], c[:, 2])
    for key in set(zip(kx.tolist(), kz.tolist())):
        sel = (kx == key[0]) & (kz == key[1])
        tri = p['idx'][sel]; used, inv = np.unique(tri, return_inverse=True)
        push(key[0], key[1], p['mat'], p['pos'][used], p['nrm'][used], p['uv'][used], inv.reshape(-1, 3))
for p in TPp['parts']: push_part(p)
for p in ST['parts']: push_part(p)
for M_ in MODS.values():
    for p in M_['parts']:
        if p['mat'] != 'collision': push_part(p)

index = {'origin': {'osm_node': 348155428, 'epsg25832': [E0, N0], 'height_DHHN': H0}, 'frame': 'metres, Y up, x=east, y=height-56.16, z=-north',
         'tile_size': TILE, 'tiles': [], 'detail_square': {'xmin': SQ[0], 'zmin': SQ[1], 'xmax': SQ[2], 'zmax': SQ[3], 'note': 'ground there = Rhino detail model; our tiles overlap 3 m under it (2 cm lower)'},
         'materials': {k: {kk: vv for kk, vv in v.items() if kk in ('tex', 'alpha', 'double')} for k, v in MATS.items()},
         'notes': ['Textures are external (tex/*.jpg|png) and shared by name; reuse one THREE.Material per material name across tiles.',
                   "Material 'marking' (road markings) and 'rail'/'groove' are decals 4-12 mm above the road: polygonOffset(-1,-1) recommended.",
                   'Underground parts (U-Bahn stairwells, Niki-de-Saint-Phalle-Promenade level -1) are in the tiles; their interior faces face inward.',
                   'Materials sign_*, citylight, netmap, display_tram use an emissiveTexture (illuminated signs).',
                   'Inside the detail square the old stairwells of the Rhino model are removed via data/detail_replace/*.json (apply_landmark_replace.py); the replacement paving patches are in these tiles.'],
         'attribution': ['(c) GeoBasis-DE/LGLN 2026, CC BY 4.0 (veraendert) - DGM1 terrain heights, LoD2 footprints', '(c) OpenStreetMap contributors (ODbL)',
                         'Textures partly from the Opernplatz detail model (Rhino)']}
tot_tris = 0; tot_bytes = 0
for (tx, tz), mats in sorted(tiles.items()):
    g = GLB(); prims = []
    for m, lst in mats.items():
        off = 0; P = []; N = []; U = []; I = []
        for pos, nrm, uv, idx in lst:
            P.append(pos); N.append(nrm); U.append(uv); I.append(idx + off); off += len(pos)
        prims.append({'pos': np.concatenate(P), 'nrm': np.concatenate(N), 'uv': np.concatenate(U), 'idx': np.concatenate(I), 'mat': m})
    mi = g.mesh('ground_%d_%d' % (tx, tz), prims, MATS)
    g.node('ground_%d_%d' % (tx, tz), mi)
    fn = 'ground_%d_%d.glb' % (tx, tz)
    sz = g.save(staging(fn)); new_files.append(fn)
    nt = sum(len(p['idx']) for p in prims)
    allp = np.concatenate([p['pos'] for p in prims])
    index['tiles'].append({'file': fn, 'tx': tx, 'tz': tz, 'bounds': {'min': allp.min(0).round(3).tolist(), 'max': allp.max(0).round(3).tolist()},
                           'triangles': int(nt), 'bytes': int(sz), 'materials': sorted(mats)})
    tot_tris += nt; tot_bytes += sz
log('tiles written', len(tiles), 'triangles', tot_tris, 'MB', round(tot_bytes / 1e6, 1))

# ---------------------------------------------------------------- Kollision: vereinfachte Hoehengruppen
CL = L['classes']; HOLES = L['holes']
bz = np.load(CACHE + 'base.npz'); BG = Grid(bz['B'].astype(np.float64), float(bz['x0']), float(bz['z0']), 1.0)
grp = {'low': ['road', 'gutter', 'parking', 'setts_road', 'track'], 'rim': ['rim'], 'water': ['water'], 'hbs': ['hbs'], 'seam': ['seam', 'seam_road', 'seam_setts', 'seam_grass']}
inv_g = {c: k for k, v in grp.items() for c in v}
G_ = {}
for c, g in CL.items():
    G_.setdefault(inv_g.get(c, 'raised'), []).append(g)
G_ = {k: unary_union(v) for k, v in G_.items()}
LEV = {'low': -0.14, 'raised': 0.0, 'rim': 0.45, 'water': 0.30, 'hbs': 0.72, 'seam': 0.0}
lines = []
for k, g in G_.items():
    for p in getattr(g, 'geoms', [g]):
        if p.geom_type == 'Polygon': lines += [p.exterior] + list(p.interiors)
for h in HOLES.values(): lines.append(h['poly'].exterior)
noded = shapely.segmentize(shapely.unary_union(MultiLineString([np.asarray(r.coords) for r in lines]), grid_size=0.002), 1.0)
segs = np.concatenate([np.stack([np.asarray(ls.coords)[:-1], np.asarray(ls.coords)[1:]], 1) for ls in getattr(noded, 'geoms', [noded]) if len(ls.coords) >= 2])
key = np.round(segs.reshape(-1, 2) * 2000).astype(np.int64); uk, inv = np.unique(key, axis=0, return_inverse=True)
V0 = uk / 2000.0; S0 = inv.reshape(-1, 2); S0 = S0[S0[:, 0] != S0[:, 1]]
holes = [list(h['poly'].representative_point().coords[0]) for h in HOLES.values()] + [list(L['sq_in'].representative_point().coords[0])]
r = tr.triangulate({'vertices': V0, 'segments': S0, 'holes': np.array(holes)}, 'pYa12Q')
V, T = r['vertices'], r['triangles']
c = V[T].mean(1); lab = np.full(len(T), -1); names = list(G_)
for i, k in enumerate(names):
    g = G_[k]; shapely.prepare(g); todo = lab < 0
    lab[todo] = np.where(shapely.contains_xy(g, c[todo, 0], c[todo, 1]), i, -1)
if (lab < 0).any():
    kd = cKDTree(c[lab >= 0]); _, ii = kd.query(c[lab < 0]); lab[lab < 0] = lab[lab >= 0][ii]
TMh = np.load(CACHE + 'terrain.npz')
# Nahtstreifen-Hoehen aus dem Sichtmesh uebernehmen (naechster Vertex)
seam_c = names.index('seam') if 'seam' in names else -1
kdS = cKDTree(TMh['OV'][:, [0, 2]])
HCc = np.zeros((len(T), 3))
for k in range(3):
    v = T[:, k]; h = BG(V[v, 0], V[v, 1]) + np.array([LEV[names[l]] for l in lab])
    if seam_c >= 0:
        sm = lab == seam_c
        if sm.any():
            _, ii = kdS.query(V[v[sm]]); h[sm] = TMh['OV'][ii, 1]
    HCc[:, k] = h
key = (T.ravel().astype(np.int64) * 16 + np.repeat(lab, 3))
uk2, oinv = np.unique(key, return_inverse=True)
CV = np.zeros((len(uk2), 3)); CV[oinv, 0] = V[T.ravel(), 0]; CV[oinv, 2] = V[T.ravel(), 1]; CV[oinv, 1] = HCc.ravel()
CT = oinv.reshape(-1, 3)
fn_ = np.cross(CV[CT[:, 1]] - CV[CT[:, 0]], CV[CT[:, 2]] - CV[CT[:, 0]]); fl = fn_[:, 1] < 0; CT[fl] = CT[fl][:, ::-1]
# Stufenwaende in der Kollision
E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); Et = np.tile(np.arange(len(T)), 3)
Es = np.sort(E, 1); ek = Es[:, 0].astype(np.int64) * len(V) + Es[:, 1]
o = np.argsort(ek); eks = ek[o]; same = np.nonzero(eks[1:] == eks[:-1])[0]
iA, iB = o[same], o[same + 1]
def ch(t, vtx): return (HCc[t] * (T[t] == vtx[:, None])).sum(1)
u, w = E[iA, 0], E[iA, 1]; tA, tB = Et[iA], Et[iB]
du = ch(tA, u) - ch(tB, u); dw = ch(tA, w) - ch(tB, w)
need = np.maximum(np.abs(du), np.abs(dw)) > 0.01
WQ = []
for j in np.nonzero(need)[0]:
    a, b = V[u[j]], V[w[j]]
    ya = sorted((ch(tA[j:j + 1], u[j:j + 1])[0], ch(tB[j:j + 1], u[j:j + 1])[0])); yb = sorted((ch(tA[j:j + 1], w[j:j + 1])[0], ch(tB[j:j + 1], w[j:j + 1])[0]))
    WQ.append([[a[0], ya[0], a[1]], [b[0], yb[0], b[1]], [b[0], yb[1], b[1]], [a[0], ya[1], a[1]]])
cparts = [G.part('collision', CV, CT, np.tile([0, 1, 0], (len(CV), 1)))]
if WQ: cparts.append(G.flat_quads('collision', np.array(WQ)))
cparts += [dict(p, mat='collision') for p in ALL_COLL]
cm = G.merge(cparts)[0]
# Verschweissen
kk = np.round(cm['pos'] * 1000).astype(np.int64); uq, iv = np.unique(kk, axis=0, return_inverse=True)
cpos = uq / 1000.0; cidx = iv.reshape(-1)[cm['idx']]
cidx = cidx[(cidx[:, 0] != cidx[:, 1]) & (cidx[:, 1] != cidx[:, 2]) & (cidx[:, 0] != cidx[:, 2])]
g = GLB(); mi = g.mesh('collision_ground', [{'pos': cpos, 'nrm': None, 'uv': None, 'idx': cidx, 'mat': 'collision'}], MATS)
g.node('collision_ground', mi, extras={'collision': True})
csz = g.save(staging('collision_ground.glb')); new_files.append('collision_ground.glb')
log('collision triangles', len(cidx), 'MB', round(csz / 1e6, 1))
np.savez_compressed(CACHE + 'collision.npz', pos=cpos, idx=cidx)

# ---------------------------------------------------------------- Hoehenraster 1 m (inkl. Detail-Quadrat)
import trimesh
HF0 = -float(HF_HALF); HFN = 2 * HF_HALF + 1
xs = HF0 + np.arange(HFN); zs = HF0 + np.arange(HFN)
GX, GZ = np.meshgrid(xs, zs)
hf = np.full(GX.shape, np.nan)
# Gelaende (ohne Gelaender/Hecken): Sichtmesh Boden + begehbare Bauwerksflaechen (Treppen, Boeden, Flicken, Bruecken)
ground_parts = [G.part('x', TMh['OV'], TMh['OT'])] + [p for p in ST['coll'] if p['mat'] in ('granite', 'slabs_ped', 'concrete')] + ALL_WALK
gm = G.merge([dict(p, mat='x') for p in ground_parts])[0]
mesh = trimesh.Trimesh(gm['pos'], gm['idx'], process=False)
ins = ~in_square(GX, GZ, -3.0)        # ausserhalb des (um den Nahtstreifen verkleinerten) Quadrats
O = np.c_[GX[ins], np.full(ins.sum(), 60.0), GZ[ins]]
locs, ri, ti = mesh.ray.intersects_location(O, np.tile([0, -1.0, 0], (len(O), 1)), multiple_hits=False)
tmp = np.full(len(O), np.nan); tmp[ri] = locs[:, 1]; hf[ins] = tmp
# Detail-Quadrat: Rhino-Boden (data/detail_raw, Spielframe) ohne die per detail_replace entfernten Dreiecke + eigene Flicken
import detailpatch as DP
DPD = DP.Detail(('street', 'ground', 'green'))
REPL = []
for fnj in sorted(glob.glob(DATA + 'detail_replace/*.json')) + sorted(glob.glob(DATA + 'landmarks_replace/*.json')):
    J = json.load(open(fnj, encoding='utf-8'))
    fs = set(J.get('files') or [])
    REPL.append((DP.footprint_poly(J['footprints']), float(J.get('ymin', -2)), float(J.get('ymax', 60)), fs))
KEEP = [np.ones(len(Md['F']), bool) for Md in DPD.meshes]
for poly, y0_, y1_, fs in REPL:
    if not fs: continue                          # Gebaeude-Ersetzungen betreffen den Boden nicht
    for k_, m_ in enumerate(DPD.select(poly, y0_, y1_, fs)): KEEP[k_] &= ~m_
dm = [trimesh.Trimesh(Md['V'], Md['F'][kp], process=False) for Md, kp in zip(DPD.meshes, KEEP) if kp.any()]
dm += [trimesh.Trimesh(p['pos'], p['idx'], process=False) for p in ALL_WALK]
dmesh = trimesh.util.concatenate(dm)
insq = ~ins
O = np.c_[GX[insq], np.full(insq.sum(), 60.0), GZ[insq]]
locs, ri, ti = dmesh.ray.intersects_location(O, np.tile([0, -1.0, 0], (len(O), 1)), multiple_hits=False)
tmp = np.full(len(O), np.nan); tmp[ri] = locs[:, 1]; hf[insq] = tmp
miss = np.isnan(hf)
hf[miss] = BG(GX[miss], GZ[miss])
hf = hf.astype(np.float32)
hf.tofile(staging('heightfield.bin')); new_files.append('heightfield.bin')
json.dump({'x0': HF0, 'z0': HF0, 'dx': 1.0, 'dz': 1.0, 'w': HFN, 'h': HFN, 'dtype': 'float32', 'order': 'row-major, row = z (south-increasing), col = x',
           'value': 'top walkable ground surface y (game frame, m); inside buildings = terrain under the building; in U-Bahn/Promenade pits = pit floor/stairs',
           'sources': 'own ground mesh outside the detail square; Rhino detail ground (street/ground/green.glb) inside; fallback DGM1-based base surface',
           'filled_from_base': int(miss.sum())}, open(staging('heightfield.json'), 'w'), indent=1); new_files.append('heightfield.json')
log('heightfield', hf.shape, 'fallback cells', int(miss.sum()))

index['collision'] = {'file': 'collision_ground.glb', 'triangles': int(len(cidx)), 'bytes': int(csz)}
index['heightfield'] = {'file': 'heightfield.bin', 'meta': 'heightfield.json', 'x0': HF0, 'z0': HF0, 'dx': 1.0, 'dz': 1.0, 'w': HFN, 'h': HFN, 'dtype': 'float32', 'order': 'row-major, row index = z'}
if os.path.exists(OUT + 'furniture.json'):
    fj = json.load(open(OUT + 'furniture.json'))
    index['furniture'] = {'file': 'furniture.glb', 'instances': 'furniture.json', 'extension': 'EXT_mesh_gpu_instancing', 'prototypes': fj['prototypes'], 'instanced_triangles': fj['instanced_triangles']}
index['totals'] = {'tile_triangles': int(tot_tris), 'tile_bytes': int(tot_bytes)}
index['ubahn_entrances'] = MODS.get('ubahn', {}).get('info', [])
index['detail_replace'] = [r['file'].split('/')[-1] for r in MODS.get('ubahn', {}).get('replace', [])] + [r for M_ in MODS.values() for r in M_.get('replace_files', [])]
for nm, M_ in MODS.items():
    if M_.get('info_extra'): index[nm] = M_['info_extra']
json.dump(index, open(staging('index.json'), 'w'), indent=1)
# ---------------------------------------------------------------- Austausch (atomar je Datei), dann veraltete Kacheln entfernen
import shutil
if DRY:
    log('dry run: files in', DRYD); sys.exit(0)
for fn in new_files: publish(staging(fn), fn, raw_copy=fn.endswith('.glb'))
publish(staging('index.json'), 'index.json', raw_copy=True)
for fn in sorted(old_tiles - set(new_files)):
    for d in (OUT, RAW):
        try: os.remove(d + fn)
        except FileNotFoundError: pass
# Texturen fuer die Rohkopien (gleiche relative URIs tex/...)
shutil.copytree(OUT + 'tex', RAW + 'tex', dirs_exist_ok=True)
try: os.rmdir(OUT + '.staging')
except OSError: pass
log('published', len(new_files) + 1, 'files; raw copies in', RAW)

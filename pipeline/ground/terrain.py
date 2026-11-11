"""Basisgelaende (Gehweg-Niveau) aus LGLN DGM1 + Kalibrierung/Nahtanpassung an das Rhino-Detailmodell.
Ausgabe: data/ground_cache/base.npz  (B[iz,ix] Spiel-y des Gehweg-Niveaus, x0,z0,res; Grube-Boden, Promenade-Oeffnungen)
         data/ground_cache/seam.npz  (Detailmodell-Hoehen entlang der Naht)"""
import sys, json, time
import numpy as np, scipy.ndimage as nd
import trimesh
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely import affinity
from common import *

T0 = time.time()
def log(*a): print('[terrain %5.1fs]' % (time.time() - T0), *a, flush=True)

HALF = 940                     # Rasterhalbbreite (m) -- Ausbau 2: Radius 860 m + Rand
M, mx0, mz0, res = load_dgm()
# Ausschnitt mit NaN-Rand, falls das DGM-Mosaik (4 Kacheln) nicht reicht (Osten ab x = +701 m)
n = 2 * HALF
ix0 = int(round(-HALF - mx0)); iz0 = int(round(-HALF - mz0))
A = np.full((n, n), np.nan)
r0, c0_ = max(0, iz0), max(0, ix0); r1, c1_ = min(M.shape[0], iz0 + n), min(M.shape[1], ix0 + n)
A[r0 - iz0:r1 - iz0, c0_ - ix0:c1_ - ix0] = M[r0:r1, c0_:c1_] - H0
gx0 = mx0 + ix0; gz0 = mz0 + iz0                     # Zellmitte [0,0]
XS = gx0 + np.arange(n); ZS = gz0 + np.arange(n)
miss = np.isnan(A)
log('dgm crop', A.shape, 'origin', round(gx0, 3), round(gz0, 3), 'missing cells', int(miss.sum()))
if miss.any():
    # Luecke (ausserhalb des DGM): Gelaendehoehen aus den LoD2-Grundrissen (Z der Bodenflaechen, LGLN/Stadt) + DGM-Randstreifen
    from scipy.interpolate import griddata
    Bj = json.load(open(DATA + 'lod2_buildings.json'))['buildings']
    pts = []; vals = []
    for b in Bj:
        for g in b.get('grounds') or []:
            P_ = np.array(g['ext'], float)
            if len(P_) < 3: continue
            x_ = P_[:, 0]; z_ = -P_[:, 1]
            fx = np.clip((x_ - gx0).astype(int), 0, n - 1); fz = np.clip((z_ - gz0).astype(int), 0, n - 1)
            if np.any(miss[fz, fx]):
                pts.append(np.c_[x_, z_]); vals.append(P_[:, 2] - H0)
    # DGM-Proben im 0..30-m-Streifen vor der Luecke (alle 3 m)
    near = (~miss) & nd.binary_dilation(miss, iterations=30)
    zz_, xx_ = np.nonzero(near)
    k_ = ((zz_ % 3) == 0) & ((xx_ % 3) == 0)
    pts.append(np.c_[XS[xx_[k_]], ZS[zz_[k_]]]); vals.append(A[zz_[k_], xx_[k_]])
    pts = np.concatenate(pts); vals = np.concatenate(vals)
    mz, mx = np.nonzero(miss)
    q = np.c_[XS[mx], ZS[mz]]
    lin = griddata(pts, vals, q, method='linear')
    nn_ = griddata(pts, vals, q[np.isnan(lin)], method='nearest') if np.isnan(lin).any() else None
    if nn_ is not None: lin[np.isnan(lin)] = nn_
    A[mz, mx] = lin
    # weicher Uebergang: Luecke leicht glaetten (LoD2-Bodenhoehen streuen)
    As = nd.gaussian_filter(A, 4.0)
    w_ = np.clip(nd.distance_transform_edt(miss) / 15.0, 0, 1)
    A = A * (1 - w_) + As * w_
    log('filled gap from LoD2 grounds', len(pts), 'samples; range', round(float(np.nanmin(A[miss])), 2), round(float(np.nanmax(A[miss])), 2))
if np.isnan(A).any():
    idx = nd.distance_transform_edt(np.isnan(A), return_distances=False, return_indices=True)
    A = A[tuple(idx)]

# ---------- Gruben: Promenade-Oeffnungen (explizit) und sonstige kleine Gruben (auffuellen) ----------
yy, xx = np.mgrid[-12:13, -12:13]; fp = (xx ** 2 + yy ** 2) <= 144
C = nd.grey_closing(A, footprint=fp)
D = C - A
# Bahnanlagen (landuse=railway) nicht auffuellen (Gleise liegen zwischen Bahnsteigen tiefer)
osm = load_osm()
rail_polys = []
for w in osm['ways']:
    if w['tags'].get('landuse') == 'railway' and len(w['pts']) > 3:
        p = Polygon(w['pts']).buffer(0)
        if p.area > 100: rail_polys.append(p)
from matplotlib.path import Path as MPath
GX, GZ = np.meshgrid(XS, ZS)
rail_mask = np.zeros(A.shape, bool)
for p in rail_polys:
    for g in getattr(p, 'geoms', [p]):
        rail_mask |= MPath(np.array(g.exterior.coords)).contains_points(np.c_[GX.ravel(), GZ.ravel()]).reshape(A.shape)
rail_mask = nd.binary_dilation(rail_mask, iterations=8)

# Leine (natural=water + water=river): Wasserspiegel je Polygon aus dem DGM; nicht als Grube fuellen
river_polys = []
for w in osm['ways']:
    t = w['tags']
    if (t.get('water') == 'river' or t.get('waterway') == 'riverbank') and len(w['pts']) > 3:
        p = Polygon(w['pts']).buffer(0)
        if p.area > 50: river_polys.append(p)
from shapely.ops import unary_union as _uu
RIVER = _uu(river_polys).buffer(1.5).buffer(-1.5) if river_polys else Polygon()
river_mask = np.zeros(A.shape, bool)
for p in getattr(RIVER, 'geoms', [RIVER]):
    if p.is_empty: continue
    m_ = MPath(np.array(p.exterior.coords)).contains_points(np.c_[GX.ravel(), GZ.ravel()]).reshape(A.shape)
    for hole in p.interiors: m_ &= ~MPath(np.array(hole.coords)).contains_points(np.c_[GX.ravel(), GZ.ravel()]).reshape(A.shape)
    river_mask |= m_
river_wl = []
for p in river_polys:
    m_ = MPath(np.array(p.exterior.coords)).contains_points(np.c_[GX.ravel(), GZ.ravel()]).reshape(A.shape)
    if m_.sum() > 10: river_wl.append({'poly': [list(c) for c in p.exterior.coords], 'wl': float(np.percentile(A[m_], 20))})
log('river cells', int(river_mask.sum()), 'water levels', [round(r_['wl'], 2) for r_ in river_wl])
pit = (D > 0.6) & ~rail_mask & ~nd.binary_dilation(river_mask, iterations=14)
lab, nl = nd.label(D > 2.0)
# Promenade = tiefe (>2.5 m), langgestreckte Gruben im Korridor Kroepcke -> Ernst-August-Platz
comps = []
for i in range(1, nl + 1):
    m = lab == i
    if m.sum() < 60: continue
    zz, xx_ = np.nonzero(m); px = XS[xx_]; pz = ZS[zz]
    cx, cz = px.mean(), pz.mean()
    if not (-40 < cx < 160 and -230 < cz < 0): continue
    if np.median(D[m]) < 2.5: continue
    comps.append((np.c_[px, pz], A[m], D[m]))
# gemeinsame Achse (PCA ueber alle Zellen)
allp = np.concatenate([c[0] for c in comps])
mu = allp.mean(0); U_, S_, Vt = np.linalg.svd(allp - mu, full_matrices=False)
ax_ = Vt[0] if Vt[0][1] < 0 else -Vt[0]          # Richtung nach Norden (z negativ)
nr_ = np.array([-ax_[1], ax_[0]])
prom = []
for pts, h, dd in comps:
    s = (pts - mu) @ ax_; t = (pts - mu) @ nr_
    w = dd > 3.0                                      # nur der tiefe Kern bestimmt die Kanten
    s0, s1 = np.percentile(s[w], [0.5, 99.5]); t0, t1 = np.percentile(t[w], [1, 99])
    s0 -= 0.5; s1 += 0.5; t0 -= 0.5; t1 += 0.5        # Zellmitte -> Kante
    floor = float(np.percentile(h, 15))
    rect = [mu + ax_ * ss + nr_ * tt for ss, tt in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))]
    prom.append({'rect': [list(map(float, r)) for r in rect], 's': [float(s0), float(s1)], 't': [float(t0), float(t1)], 'floor': floor,
                 'cells': int(len(pts)), 'c': [float(x) for x in pts.mean(0)]})
prom.sort(key=lambda o: o['s'][0])
prom_axis = {'origin': mu.tolist(), 'axis': ax_.tolist(), 'normal': nr_.tolist()}
log('promenade openings', len(prom), [(round(p['s'][0], 1), round(p['s'][1], 1), round(p['t'][0], 1), round(p['t'][1], 1), round(p['floor'], 2)) for p in prom])

# Auffuellen kleiner Gruben (Rampen, Treppenschaechte) und Promenade-Oeffnungen fuers Strassen-Niveau
pitd = nd.binary_dilation(pit, iterations=2)
F = np.where(pitd, C, A)
# Flussbett durch Uferhoehen ersetzen (Ufer bleiben scharf; der Fluss selbst bekommt in terrain_mesh seinen Wasserspiegel)
if river_mask.any():
    rm_ = nd.binary_dilation(river_mask, iterations=1)
    # Ufer: steile Boeschung/Kaimauer im DGM (Zellen nahe am Fluss, die deutlich unter der Umgebung liegen) ebenfalls ersetzen
    low_bank = nd.binary_dilation(river_mask, iterations=4) & (F < nd.grey_dilation(F, size=9) - 1.0)
    rm_ |= low_bank
    idx_ = nd.distance_transform_edt(rm_, return_distances=False, return_indices=True)
    F = F[tuple(idx_)]
# leichte Glaettung (Rauschen, verschmierte Bordsteinkanten)
Fs = nd.gaussian_filter(F, 1.2)

# ---------- Kalibrierung gegen das Detailmodell ----------
log('loading detail model for calibration')
# Materialarten im Detailmodell: 0 Platten/Gehweg, 1 Asphalt, 2 Pflaster(Fahrbahn), 3 Rasen, 4 sonstige
meshes = []; kinds = []
for g in ['street', 'ground', 'green']:
    sc = trimesh.load(DATA + 'rhino_export/' + g + '.glb', force='scene')
    for name, geom in sc.geometry.items():
        T = sc.graph.get(sc.graph.geometry_nodes[name][0])[0]
        m = geom.copy(); m.apply_transform(T); m.vertices += RHINO_OFF
        mn = str(getattr(getattr(m.visual, 'material', None), 'name', ''))
        if 'asphalt' in mn or '#2F3133' in mn: k = 1
        elif 'setts' in mn: k = 2 if g == 'street' else 0
        elif 'grass' in mn: k = 3
        elif 'slabs' in mn: k = 0
        else: k = 4
        meshes.append(m); kinds.append(k)
allm = trimesh.util.concatenate(meshes)
fk = np.concatenate([[kinds[i]] * len(m.faces) for i, m in enumerate(meshes)])
ray = trimesh.ray.ray_pyembree.RayMeshIntersector(allm) if trimesh.ray.has_embree else allm.ray
Bgrid = Grid(Fs, gx0, gz0, 1.0)
ROADK = (1, 2)

def cast(x, z):
    O = np.c_[x, np.full(len(x), 30.0), z]; Dd = np.tile([0, -1.0, 0], (len(x), 1))
    loc, ri, ti = ray.intersects_location(O, Dd, multiple_hits=False)
    y = np.full(len(x), np.nan); k = np.full(len(x), -1)
    y[ri] = loc[:, 1]; k[ri] = fk[ti]
    return y, k

def walk_level(y, k):
    return y + np.where(np.isin(k, ROADK), 0.14, 0.0)

# Flaechenkalibrierung (Gehweg-Niveau = Gehweg-Flaechen bzw. Fahrbahn + 0.14)
xs, zs = np.meshgrid(np.arange(SQ[0] + 2, SQ[2] - 2, 2.0), np.arange(SQ[1] + 2, SQ[3] - 2, 2.0))
xs = xs.ravel(); zs = zs.ravel()
y, k = cast(xs, zs)
ok = ~np.isnan(y) & (y > -2) & (y < 2)
c0 = float(np.median(walk_level(y, k)[ok] - Bgrid(xs[ok], zs[ok])))
log('calibration offset (sidewalk level - smoothed DGM) =', round(c0, 3), 'n', int(ok.sum()))
Fs = Fs + c0
Bgrid = Grid(Fs, gx0, gz0, 1.0)

# ---------- Randstreifen im Quadrat: Detailhoehen + Materialart auf 0.25-m-Raster ----------
STRIP = 5.0; SR = 0.25
sx = np.arange(SQ[0] + SR / 2, SQ[2], SR); sz = np.arange(SQ[1] + SR / 2, SQ[3], SR)
SX, SZ = np.meshgrid(sx, sz)
dedge = np.minimum.reduce([SX - SQ[0], SQ[2] - SX, SZ - SQ[1], SQ[3] - SZ])
sel = dedge < STRIP + 1.0
qx, qz = SX[sel], SZ[sel]
qy, qk = cast(qx, qz)
qy[(qy < -3) | (qy > 3)] = np.nan
np.savez_compressed(CACHE + 'strip.npz', x=qx, z=qz, y=qy, k=qk, d=dedge[sel], res=SR)
log('strip samples', len(qx), 'hits', int((~np.isnan(qy)).sum()))

# ---------- Naht-Korrektur des Basisniveaus (aus Streifenproben 1-5 m innen) ----------
def seam_points(step):
    x0, z0, x1, z1 = SQ
    L = [(x0, z0, x1, z0), (x1, z0, x1, z1), (x1, z1, x0, z1), (x0, z1, x0, z0)]
    P = []; S = []; s0 = 0
    for (ax, az, bx, bz) in L:
        l = np.hypot(bx - ax, bz - az); m = int(np.ceil(l / step))
        t = np.arange(m) / m
        P.append(np.c_[ax + (bx - ax) * t, az + (bz - az) * t]); S.append(s0 + t * l); s0 += l
    return np.concatenate(P), np.concatenate(S), s0

P, S, PER = seam_points(0.5)
from scipy.spatial import cKDTree
import shapely
from shapely.ops import unary_union
bpolys = []
for w in osm['ways']:
    t = w['tags']
    if 'building' in t and t.get('building') not in ('roof',) and len(w['pts']) > 3 and not str(t.get('layer', '0')).startswith('-'):
        PP = np.array(w['pts'])
        if (np.abs(PP[:, 0] - 84) < 200).all() and (np.abs(PP[:, 1] - 142) < 200).all():
            bpolys.append(Polygon(PP).buffer(0))
bun = unary_union(bpolys).buffer(1.0); shapely.prepare(bun)
inb = shapely.contains_xy(bun, qx, qz)
okq = ~np.isnan(qy) & (dedge[sel] > 0.5) & (dedge[sel] < 5.0) & np.isin(qk, (0, 1, 2)) & ~inb
lv_all = walk_level(qy, qk) - Bgrid(qx, qz)
okq &= np.abs(lv_all) < 0.7          # Rampen/Gruben im Detailmodell nicht als Niveau-Referenz
lv = lv_all[okq]
kq = cKDTree(np.c_[qx[okq], qz[okq]])
dd, ii = kq.query(P, k=24, distance_upper_bound=6.0)
dl = np.full(len(P), np.nan)
for j in range(len(P)):
    v = ii[j][np.isfinite(dd[j])]
    if len(v) >= 4: dl[j] = np.median(lv[v])
good = ~np.isnan(dl) & (np.abs(dl) < 1.5)
log('seam points', len(P), 'with detail', int(good.sum()))
from scipy.ndimage import median_filter, uniform_filter1d
d_i = np.interp(S, S[good], dl[good], period=PER)
d_s = median_filter(d_i, size=21, mode='wrap')          # ~10 m
d_s = np.clip(uniform_filter1d(d_s, size=13, mode='wrap'), -0.6, 0.6)       # ~6 m
log('seam correction range', round(float(d_s.min()), 3), round(float(d_s.max()), 3), 'median', round(float(np.median(d_s)), 3))
np.savez_compressed(CACHE + 'seam.npz', P=P, S=S, per=PER, corr=d_s)

# Korrekturfeld: Wert des naechsten Nahtpunkts, abklingend ueber 30 m nach aussen
kd = cKDTree(P)
dist, ii = kd.query(np.c_[GX.ravel(), GZ.ravel()])
w = np.clip(1 - dist / 30.0, 0, 1); w = w * w * (3 - 2 * w)
corr = (d_s[ii] * w).reshape(A.shape)
inside = in_square(GX, GZ)
corr[inside] = d_s[ii].reshape(A.shape)[inside]
B = Fs + corr

np.savez_compressed(CACHE + 'base.npz', B=B.astype(np.float32), raw=A.astype(np.float32), x0=gx0, z0=gz0, res=1.0, c0=c0)
json.dump({'openings': prom, 'axis': prom_axis}, open(CACHE + 'promenade.json', 'w'), indent=1)
json.dump({'rivers': river_wl}, open(CACHE + 'river.json', 'w'))
log('B at clock', round(float(Grid(B, gx0, gz0, 1.0)(0.0, 0.0)), 3), 'range', round(float(B.min()), 2), round(float(B.max()), 2))

"""Strahlentest fuer das Alte Rathaus: von Standpunkten rund um das Gebaeude (Augenhoehe, 1. OG, Luft) und im
Laubengang / vor den Giebelboegen werden Strahlen auf das Modell geschossen; ein erster Treffer auf eine Rueckseite
(Normale vom Betrachter weg) deutet auf ein Loch oder eine falsch orientierte Flaeche hin (das Spiel rendert nur
FrontSide). Durchsichtige Materialien (alphaMode BLEND/MASK: Schaufensterglas, Glasbeschriftung) werden durchschossen;
hinter Schaufensterglas muss der naechste Treffer eine Vorderseite des Ladenraums sein (kein Gelaende, keine Rueckseite,
nicht weiter als 12 m = Raumdiagonale).
Aufruf: python check_rays.py <glb> [n_rays_per_view]"""
import sys, os, json, struct, math
import numpy as np
import trimesh

GLB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'altes_rathaus.glb')
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 6000


def load_glb(path):
    b = open(path, 'rb').read()
    jl = struct.unpack('<I', b[12:16])[0]
    js = json.loads(b[20:20 + jl])
    bin_ = b[20 + jl + 8:]
    def acc(i):
        a = js['accessors'][i]; v = js['bufferViews'][a['bufferView']]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a['componentType']]
        nc = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
        o = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.frombuffer(bin_, dt, a['count'] * nc, o).reshape(a['count'], nc) if nc > 1 else np.frombuffer(bin_, dt, a['count'], o)
    global TRANSP
    TRANSP = set(m['name'] for m in js['materials'] if m.get('alphaMode') in ('BLEND', 'MASK'))
    V = []; F = []; M = []; n0 = 0
    for nd in js['nodes']:
        if nd['name'].startswith('col_'): continue
        mesh = js['meshes'][nd['mesh']]
        inst = nd.get('extensions', {}).get('EXT_mesh_gpu_instancing')
        mats = [np.eye(4)]
        if inst:
            T = acc(inst['attributes']['TRANSLATION']); R = acc(inst['attributes']['ROTATION']); S = acc(inst['attributes']['SCALE'])
            mats = []
            for t, q, s in zip(T, R, S):
                x, y, z, w = q
                Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                Mm = np.eye(4); Mm[:3, :3] = Rm @ np.diag(s); Mm[:3, 3] = t; mats.append(Mm)
        for pr in mesh['primitives']:
            P = acc(pr['attributes']['POSITION']).astype(np.float64); I = acc(pr['indices']).reshape(-1, 3).astype(np.int64)
            mname = js['materials'][pr['material']]['name']
            for Mm in mats:
                Pw = P @ Mm[:3, :3].T + Mm[:3, 3]
                if np.linalg.det(Mm[:3, :3]) < 0: I2 = I[:, ::-1]
                else: I2 = I
                V.append(Pw); F.append(I2 + n0); M += [mname] * len(I2); n0 += len(Pw)
    return np.concatenate(V), np.concatenate(F), np.array(M)


V, F, M = load_glb(GLB)
mesh = trimesh.Trimesh(V, F, process=False)
FN = mesh.face_normals
print('tris', len(F))
lo, hi = V.min(0), V.max(0); cen = (lo + hi) / 2
print('bbox', lo.round(2), hi.round(2))
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    rmi = RayMeshIntersector(mesh)
except Exception:
    rmi = mesh.ray
rng = np.random.default_rng(1)
views = []
R = 45.0
for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
    for y in (1.7, 8.0, 22.0, 60.0):
        r = R if y < 30 else 70.0
        views.append(np.array([cen[0] + r * math.cos(ang), y, cen[2] + r * math.sin(ang)]))
# Standpunkte im Laubengang (Karmarschstrasse) und vor den Giebelboegen (Spielframe x, z)
S_ = np.array([-184.68, 389.84]); E_ = np.array([-161.79, 360.26]); dSE = (E_ - S_) / np.linalg.norm(E_ - S_)
inF3 = np.array([dSE[1], -dSE[0]])          # nach innen (zur Rueckwand des Laubengangs)
inner_views = []
for t in np.linspace(1.0, 36.5, 20):
    for d_in in (1.5, 2.5):
        p = S_ + dSE * t + inF3 * d_in
        inner_views.append(np.array([p[0], 1.8, p[1]]))
# 1.2 m vor den Giebelboegen F4d (a 6.05 / 2.15), F2a (2.05 / 6.11), F2b (4.2 / 14.7) + schraeg davor
for (x, z) in ((-186.97, 388.98), (-189.48, 385.99), (-162.23, 357.92), (-164.90, 354.86), (-168.99, 350.15), (-175.90, 342.23),
               (-188.5, 391.0), (-160.5, 358.5)):
    inner_views.append(np.array([x, 1.7, z]))
INNER0 = len(views)
views = views + inner_views
bad = []; tot = 0
glass_n = 0; glass_bad = []
GLASS = 'AR_Schaufenster_Glas'
def cast(org, d):
    """erster undurchsichtiger Treffer (durchsichtige Materialien werden durchschossen) -> loc, ray-idx, tri, glas-loc/-flag"""
    idx = np.arange(len(org)); o = org.copy()
    out_loc = np.full((len(org), 3), np.nan); out_tri = np.full(len(org), -1); gl_loc = np.full((len(org), 3), np.nan)
    for it_ in range(5):
        if not len(idx): break
        loc, ir, it = rmi.intersects_location(o[idx], d[idx], multiple_hits=False)
        if not len(ir): break
        gi = idx[ir]
        # durchsichtig: Glas/Dekore, Innenschale (AR_Innen) und Ladenraum-Huellen (AR_Laden) von aussen -- ihre Rueckseiten werden im Spiel gecullt (nach innen gerichtete Kaesten)
        tr = np.array([(M[k] in TRANSP) or (M[k] in ('AR_Innen', 'AR_Laden') and float(np.dot(FN[k], d[gi[j]])) > 0) for j, k in enumerate(it)])
        isg = np.array([M[k] == GLASS for k in it])
        for j in np.nonzero(isg)[0]:
            if np.isnan(gl_loc[gi[j], 0]): gl_loc[gi[j]] = loc[j]
        op_ = ~tr
        out_loc[gi[op_]] = loc[op_]; out_tri[gi[op_]] = it[op_]
        nxt = gi[tr]
        o[nxt] = loc[tr] + d[nxt] * 3e-4
        idx = nxt
    hit = np.nonzero(out_tri >= 0)[0]
    return out_loc[hit], hit, out_tri[hit], gl_loc[hit]


for vi, o in enumerate(views):
    if vi >= INNER0:      # Innenstandpunkte: Ziele in einem lokalen Kasten
        tg = np.array([o[0], 0.0, o[2]]) + (rng.random((NR, 3)) - 0.5) * np.array([16.0, 0.0, 16.0]) + np.array([0, 1, 0]) * rng.random((NR, 1)) * 5.0
    else:
        tg = lo + rng.random((NR, 3)) * (hi - lo)
    d = tg - o; d /= np.linalg.norm(d, axis=1, keepdims=True)
    org = np.repeat(o[None], NR, 0)
    loc, ir, it, gl = cast(org, d)
    if not len(ir): continue
    nd = FN[it]; dd = d[ir]
    back = (nd * dd).sum(1) > 1e-4
    tot += len(ir)
    for k in np.nonzero(back)[0]:
        bad.append((o, loc[k], M[it[k]], it[k]))
    # hinter Schaufensterglas
    g = ~np.isnan(gl[:, 0])
    glass_n += int(g.sum())
    for k in np.nonzero(g)[0]:
        dist = float(np.linalg.norm(loc[k] - gl[k]))
        if back[k] or dist > 12.0 or not (M[it[k]].startswith('AR_Laden') or M[it[k]].startswith('AR_dsp_') or M[it[k]] in ('AR_Tuergriff', 'AR_Granit', 'AR_Backstein', 'AR_Putz', 'AR_Wechsel', 'AR_Dekor_white', 'AR_Schild_Halter', 'AR_Glaswand_Glas')):
            glass_bad.append((o, gl[k], loc[k], M[it[k]], dist, bool(back[k])))
print('views', len(views), '(davon innen/Boegen', len(views) - INNER0, ')')
print('rays through shop glass', glass_n, 'bad behind glass', len(glass_bad))
if glass_bad:
    from collections import Counter as _C
    print('  behind-glass problems:', _C([(b[3], b[5]) for b in glass_bad]).most_common(8))
    for b in glass_bad[:6]: print('   from', np.round(b[0], 1), 'glass', np.round(b[1], 2), 'hit', np.round(b[2], 2), b[3], 'dist %.1f back %s' % (b[4], b[5]))
print('hits', tot, 'backface first hits', len(bad), '(%.3f%%)' % (100 * len(bad) / max(tot, 1)))
from collections import Counter
print(Counter([b[2] for b in bad]).most_common(20))
# Cluster der Problemorte
if bad:
    P = np.array([b[1] for b in bad])
    key = np.round(P / 2.0).astype(int)
    cnt = Counter(map(tuple, key))
    for k, c in cnt.most_common(25):
        sel = [b for b in bad if tuple(np.round(b[1] / 2.0).astype(int)) == k]
        print('  at %s n=%d mats=%s from=%s' % (np.array(k) * 2.0, c, Counter(s[2] for s in sel).most_common(3), np.round(sel[0][0], 1)))

# Detail: CHECK_AT="x,y,z,r" -> einzelne Problemtreffer in der Naehe ausgeben
if os.environ.get('CHECK_AT') and bad:
    cx, cy, cz, cr = [float(v) for v in os.environ['CHECK_AT'].split(',')]
    sel = [b for b in bad if np.linalg.norm(b[1] - np.array([cx, cy, cz])) < cr]
    print('detail', len(sel))
    for o, p, m, ti in sel[:12]:
        d = p - o; d /= np.linalg.norm(d)
        print('   hit', np.round(p, 2), m, 'n', np.round(FN[ti], 2), 'dir', np.round(d, 2), 'tri', np.round(V[F[ti]], 2).tolist())

if os.environ.get('CHECK_MAT') and bad:
    for mname in os.environ['CHECK_MAT'].split(','):
        sel = [b for b in bad if b[2] == mname]
        key = Counter(tuple(np.round(b[1] / 3.0).astype(int)) for b in sel)
        print('MAT', mname, len(sel))
        for k, c in key.most_common(6):
            s = [b for b in sel if tuple(np.round(b[1] / 3.0).astype(int)) == k][0]
            d = s[1] - s[0]; d /= np.linalg.norm(d)
            print('   ', np.array(k) * 3.0, c, 'n', np.round(FN[s[3]], 2), 'dir', np.round(d, 2), 'tri', np.round(V[F[s[3]]], 2).tolist())

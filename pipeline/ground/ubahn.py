"""U-Bahn-Zugaenge (Ausbau 2): offener Trog mit Granitstufen (Trittkantenstreifen, Aufmerksamkeitsfeldern), Zwischenpodest,
Rolltreppen (Stufen, Balustraden, Kammplatten, Handlauf-Umlenkungen), Edelstahl-Handlaeufen (Wand + Mittelhandlauf),
Granitbruestung mit Edelstahlgelaender, Wandverkleidung, beleuchtetem U-Wuerfel mit Stationsnamen, Tunnel + geschlossener
Verteilerebenen-Stummel (Kroepcke: gelbgruenes Wandmosaik, tuerkisgruene Mosaikstuetzen mit Ovalleuchte, Lamellendecke mit
Lichtbaendern, Lichtvoute, beige Fliesen mit Leitstreifen). Freistehender Glasaufzug Kr.80A.
Im Detailmodell-Quadrat: alte Treppenschaechte/Pflaster werden per data/detail_replace/ubahn_<ref>.json entfernt und hier
lueckenlos neu gepflastert (detailpatch).  Ausgabe: data/ground_cache/ubahn.pkl"""
import sys, json, time, pickle, math
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from common import *
import geom as G
import geom2 as G2
from geom2 import Frame
import ubahn_spec as US
import detailpatch as DP
from surface import Surface
from kit import *
from kit import _strip, _ceiling_strips

T0 = time.time()
def log(*a): print('[ubahn %6.1fs]' % (time.time() - T0), *a, flush=True)

L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
TP = pickle.load(open(CACHE + 'terrain_parts.pkl', 'rb'))
HE = TP['hole_edges']
SURF = Surface()
SQB = box(*SQ)
DET = DP.Detail(('street', 'ground', 'street_furniture', 'green_trees'))
log('detail meshes', len(DET.meshes))

# Bodenflaechen des Detailmodells fuer Hoehenabfragen
OLDMAT = ('8F8A82', '2F3133', 'F4F4F2', '1D5FA8', 'EDEDE8', '9C9EA0')
_gm = [M for M in DET.meshes if M['file'] in ('street', 'ground') and not any(c in M['mat'] for c in OLDMAT)]
_GF = np.concatenate([M['V'][M['F'][M['N'][:, 1] > 0.6]] for M in _gm])         # (k,3,3)
_kd = cKDTree(_GF.mean(1)[:, [0, 2]])
def detail_h(x, z):
    ii = _kd.query_ball_point([x, z], 4.0)
    best = None
    for i in ii:
        A, B, C = _GF[i]
        v0 = B[[0, 2]] - A[[0, 2]]; v1 = C[[0, 2]] - A[[0, 2]]; v2 = np.array([x, z]) - A[[0, 2]]
        d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1; den = d00 * d11 - d01 * d01
        if abs(den) < 1e-14: continue
        wv = (d11 * (v2 @ v0) - d01 * (v2 @ v1)) / den; ww = (d00 * (v2 @ v1) - d01 * (v2 @ v0)) / den; wu = 1 - wv - ww
        if wu >= -1e-6 and wv >= -1e-6 and ww >= -1e-6:
            y = wu * A[1] + wv * B[1] + ww * C[1]
            if best is None or y > best: best = y
    return best
def ground_h(x, z):
    if SQB.buffer(-0.05).contains(Point(x, z)):
        y = detail_h(x, z)
        if y is not None: return y
    return float(SURF([x], [z])[0][0])

parts = []; coll = []; walk = []
def add(p, collide=False, walkable=False):
    if p is None: return
    if isinstance(p, (list, tuple)):
        for q in p: add(q, collide, walkable)
        return
    if len(p['idx']) == 0: return
    parts.append(p)
    if collide: coll.append(p)
    if walkable: walk.append(p)

STYLE = {
    'kroepcke': {'trough': 'granite_wall', 'wall': 'mosaic_yg', 'floor': 'tile_beige', 'ceil': 'slats', 'column': 'mosaic_teal', 'skirt': 'granite_dark', 'cove': 'lightcove'},
    'steintor': {'trough': 'granite_wall', 'wall': 'tile_white', 'floor': 'tile_beige', 'ceil': 'concrete', 'column': 'concrete', 'skirt': 'granite_dark', 'cove': 'lightline'},
    'generic': {'trough': 'granite_wall', 'wall': 'tile_white', 'floor': 'granite_dark', 'ceil': 'slats', 'column': 'concrete', 'skirt': 'granite_dark', 'cove': 'lightline'},
}
HT = 2.70            # lichte Hoehe Tunnel
HH = 3.10            # lichte Hoehe Halle

# ---------------------------------------------------------------- Bausteine
# ---------------------------------------------------------------- Treppe
# ---------------------------------------------------------------- Rolltreppe
# ---------------------------------------------------------------- Zugang komplett
INFO = []; REPLACE = []; PREV_FP = [Polygon()]
def build_entrance(sp):
    key = sp['key']; sty = STYLE[sp['style']]
    fr = Frame(sp['top'], sp['dir'])
    W = sp['W']; Lo = sp['L_open']; lanes = sp['lane_geom']; D = sp['depth']
    y_top = ground_h(*fr.xz(0.3, 0))
    y_floor = y_top - D
    cen = fr.xz(Lo / 2, 0)
    # ---- Rim-Kanten: Gelaende (ausserhalb des Detailmodells) + Detail-Flicken (innerhalb)
    rim = []
    for k, h in L['holes'].items():
        if h.get('key') != key: continue
        for e in HE.get(k, []):
            m = (np.array(e[0]) + np.array(e[1])) / 2
            if sp['in_square'] and SQB.contains(Point(m)): continue
            rim.append(e)
    if sp['in_square']:
        # Fussabdruck: Arbeitsbereich + alte Trog-Geometrie (+0.4 m)
        Fz = sp['poly_zone']
        hulls = []
        for M in DET.meshes:
            if M['file'] != 'street' or not any(c in M['mat'] for c in OLDMAT[:4]): continue
            C = M['C']; near = np.hypot(C[:, 0] - cen[0], C[:, 2] - cen[1]) < Lo / 2 + W + 8
            if not near.any(): continue
            import trimesh as _tm
            fi = np.nonzero(near)[0]
            sub = _tm.Trimesh(M['V'], M['F'][fi], process=False); sub.merge_vertices()
            labs = _tm.graph.connected_component_labels(sub.face_adjacency, node_count=len(sub.faces))
            for k in range(labs.max() + 1):
                Vk = sub.vertices[np.unique(sub.faces[labs == k])][:, [0, 2]]
                if len(Vk) < 3: continue
                hk = shapely.MultiPoint(Vk).convex_hull
                if hk.distance(Fz) < 1.0: hulls.append(hk)
        Fp = unary_union([Fz] + [h.buffer(0.4) for h in hulls])
        Fp = Polygon(Fp.convex_hull.exterior).simplify(0.05)
        # grosse Detail-Bodendreiecke, deren Schwerpunkt ausserhalb liegt, die aber in die neue Oeffnung ragen, mitnehmen
        for _it in range(3):
            add_ = []
            for M in DET.meshes:
                if M['file'] not in ('street', 'ground') or any(c in M['mat'] for c in OLDMAT): continue
                C = M['C']; nearm = (np.hypot(C[:, 0] - cen[0], C[:, 2] - cen[1]) < Lo + W + 6) & (M['N'][:, 1] > 0.3)
                for fi in np.nonzero(nearm)[0]:
                    if Fp.contains(Point(C[fi, 0], C[fi, 2])): continue
                    tp_ = Polygon(M['V'][M['F'][fi]][:, [0, 2]])
                    if tp_.is_valid and tp_.intersection(sp['poly_open'].buffer(0.02)).area > 1e-4: add_.append(tp_.buffer(0.05))
            if not add_: break
            log('  ', key, 'footprint grows by', len(add_), 'detail triangles reaching into the opening')
            Fp = Polygon(unary_union([Fp] + add_).convex_hull.exterior).simplify(0.05)
        Fp = Fp.intersection(SQB.buffer(0.5)).difference(PREV_FP[0])
        Fp = max(getattr(Fp, 'geoms', [Fp]), key=lambda g: g.area)
        Fp = Polygon(Fp.exterior)
        fn = DP.write_replace('ubahn_' + (sp['ref'] or key), [Fp], 'U-Bahn-Zugang %s %s: alter Treppenschacht/Belag entfernt, ersetzt durch ground/ubahn (pipeline/ground/ubahn.py). Pflaster im Fussabdruck lueckenlos neu (Material/UV/Hoehe aus dem Detailmodell).' % (sp['station'], sp['ref']), files=('street', 'street_furniture', 'ground'))
        J = json.load(open(fn, encoding='utf-8'))
        Fsim = DP.footprint_poly(J['footprints'])
        pp, rim_p, removed = DP.build_patch(DET, Fsim, sp['poly_open'], y_top, clip=SQB.difference(PREV_FP[0]))
        PREV_FP[0] = PREV_FP[0].union(Fsim)
        add(pp, True, True)
        rim += rim_p
        REPLACE.append({'key': key, 'file': fn, 'removed': removed})
        # Stadtmoebel/Baeume, die in den neuen Trog ragen
        for M in DET.meshes:
            if M['file'] in ('street_furniture', 'green_trees'):
                ins = shapely.contains_xy(sp['poly_open'].buffer(0.3), M['C'][:, 0], M['C'][:, 2]) & (M['C'][:, 1] > y_top - 0.5)
                if ins.sum() > 5: log('  WARN', key, M['file'], M['mat'], int(ins.sum()), 'faces stand in the new opening')
        log(key, 'patch tris', G.tri_count(pp), 'rim edges', len(rim_p), 'removed', removed)
    # ---- Rim-Kanten nach Seiten
    side_e = {'entry': [], 'far': [], 'l': [], 'r': []}
    for e in rim:
        m = (np.array(e[0]) + np.array(e[1])) / 2; s, t = fr.st(*m)
        if abs(s) < 0.05: side_e['entry'].append(e)
        elif abs(s - Lo) < 0.05: side_e['far'].append(e)
        elif abs(t + W / 2) < 0.05: side_e['l'].append(e)
        elif abs(t - W / 2) < 0.05: side_e['r'].append(e)
    if not rim: log('WARN no rim for', key)
    # Trogwaende (Granitplatten) bis unter die Stufen; Stirnwand bis Tunneldecke
    add(wall_from_edges(side_e['l'] + side_e['r'], lambda x, z: y_floor - 0.3, sty['trough'], cen, 2.4), True)
    add(wall_from_edges(side_e['far'], lambda x, z: y_floor + HT, sty['trough'], cen, 2.4), True)
    # oberes Podest: vom Rim (exakte Hoehen) bis zur ersten Kante, Aufmerksamkeitsfeld; Kanten an Spurgrenzen teilen
    bounds_t = sorted(set([g['t0'] for g in lanes] + [g['t1'] + US.DIVIDER for g in lanes[:-1]] + [W / 2]))
    def lane_at(t):
        for i_, g in enumerate(lanes):
            hi = g['t1'] + (US.DIVIDER if i_ < len(lanes) - 1 else 0.0)
            if g['t0'] - 1e-6 <= t <= hi + 1e-6: return g
        return lanes[-1]
    entry = []
    for (pu, pw, hu, hw, c) in side_e['entry']:
        su, tu = fr.st(*pu); sw, tw = fr.st(*pw)
        if tw < tu: su, tu, hu, sw, tw, hw = sw, tw, hw, su, tu, hu
        cuts = [tu] + [b_ for b_ in bounds_t if tu + 1e-4 < b_ < tw - 1e-4] + [tw]
        for a_, b_ in zip(cuts[:-1], cuts[1:]):
            fa = (a_ - tu) / (tw - tu + 1e-12); fb = (b_ - tu) / (tw - tu + 1e-12)
            entry.append((a_, b_, hu + (hw - hu) * fa, hu + (hw - hu) * fb))
    for (ta, tb, ha, hb_) in entry:
        g = lane_at((ta + tb) / 2)
        if g['type'] == 'stair':
            depth = US.TOP_LANDING
            q = G2.quad('tactile', fr.P(0, ta, ha), fr.P(0, tb, hb_), fr.P(depth - 0.05, tb, y_top), fr.P(depth - 0.05, ta, y_top),
                        uv=np.array([[ta, 0], [tb, 0], [tb, depth], [ta, depth]]) / 0.6)
            if q is not None:
                if np.cross(q['pos'][1] - q['pos'][0], q['pos'][3] - q['pos'][0])[1] < 0: q['idx'] = q['idx'][:, ::-1]
                q['nrm'] = np.tile([0, 1.0, 0], (4, 1))
                add(q, True, True)
            add(fr.hquad('nosing', depth - 0.05, depth, ta, tb, y_top, up=True), True, True)
        else:
            # Rolltreppe: Podestplatte beginnt bei s=0 auf y_top -> Stossfuge zum Rim beidseitig schliessen
            lo_ = min(ha, hb_, y_top) - 0.04; hi_ = max(ha, hb_, y_top)
            add(fr.vquad('stainless', (0, ta), (0, tb), lo_, hi_, (-1, (ta + tb) / 2)))
            add(fr.vquad('stainless', (0, ta), (0, tb), lo_, hi_, (1, (ta + tb) / 2)))
    # ---- Spuren
    for i, g in enumerate(lanes):
        t0 = g['t0']; t1 = g['t1'] + (US.DIVIDER if i < len(lanes) - 1 else 0.0)
        if g['type'] == 'stair':
            ps = build_stair(fr, g, y_top, y_floor, t0, t1, sty)
            add(ps, True, True)
            # Handlaeufe: beide Seiten, Mittelhandlauf bei breiten Treppen
            NL = stair_nose_line(g, y_top)
            rails = [t0 + 0.08, t1 - 0.08]
            if g['w'] >= 2.8: rails.append((t0 + t1) / 2)
            for tr_ in rails:
                pts = np.array([fr.P(s_, tr_, y + 0.9) for s_, y in NL])
                add(G.tube('stainless', pts, 0.022, 8))
                mid = abs(tr_ - (t0 + t1) / 2) < 0.01
                for s_, y in NL[1:-1:5]:
                    if mid: add(G2.cyl_between('stainless', fr.P(s_, tr_, y), fr.P(s_, tr_, y + 0.9), 0.022, 6))
                    else:
                        wt = t0 - 0.02 if tr_ < (t0 + t1) / 2 else t1 + 0.02
                        add(G2.cyl_between('stainless', fr.P(s_, wt, y + 0.86), fr.P(s_, tr_, y + 0.86), 0.012, 5))
                if mid:
                    # Endboegen des Mittelhandlaufs
                    for (s_, y), sg in ((NL[0], -1), (NL[-1], 1)):
                        add(G2.ring_tube('stainless', fr.P(s_, tr_, y + 0.6), 0.3, 0.022, 8, 6, axis_u=fr.vec(sg, 0), axis_v=(0, 1, 0), a0=0, a1=math.pi / 2))
            # Stufenbeleuchtung in der Wand (nur an Trogwaenden)
            for k in range(3, g['n'], 6):
                s_ = g['s_riser'][k]; y = y_top - k * g['rise']
                for tw_, sgn in ((-W / 2, 1), (W / 2, -1)):
                    if (sgn > 0 and abs(t0 + W / 2) < 0.2) or (sgn < 0 and abs(t1 - W / 2) < 0.2):
                        add(fr.box('downlight', s_ - 0.12, s_ + 0.12, tw_ - 0.001 if sgn < 0 else tw_, tw_ + 0.012 * sgn if sgn > 0 else tw_ + 0.001, y + 0.30, y + 0.36))
            # unterer Aufmerksamkeitsstreifen + Boden der Spur
            s_last = g['s_riser'][-1]
            add(fr.hquad('tactile', s_last, s_last + 0.6, t0, t1, y_floor, up=True, uvscale=0.6), True, True)
            add(fr.hquad(sty['floor'], s_last + 0.6, Lo + sp['tunnel_len'], t0, t1, y_floor, up=True, uvscale=1.2), True, True)
        else:
            vis, col = build_escalator(fr, g, y_top, y_floor, t0, t1, edge_wall_l=(i == 0), edge_wall_r=(i == len(lanes) - 1))
            add(vis, False, False); add([c for c in col if c is not None], True, False)
            walk.extend([c for c in col if c is not None])
            add(fr.hquad(sty['floor'], g['s_end'], Lo + sp['tunnel_len'], t0, t1, y_floor, up=True, uvscale=1.2), True, True)
    # ---- Bruestung (poliert) mit Abdeckplatte + Edelstahlgelaender: links, rechts, Stirnseite
    PAR = 0.30; HP = 0.72; CAP = 0.08
    def gy(s, t): return ground_h(*fr.xz(s, t))
    segs = [('l', (0.0, Lo + PAR), (-W / 2 - PAR, -W / 2)), ('r', (0.0, Lo + PAR), (W / 2, W / 2 + PAR)), ('f', (Lo, Lo + PAR), (-W / 2, W / 2))]
    for nm, (s0, s1), (t0, t1) in segs:
        n_ = max(2, int(math.ceil((s1 - s0) / 2.5)) + 1) if nm != 'f' else max(2, int(math.ceil((t1 - t0) / 2.5)) + 1)
        # in Elemente (~2.5 m) teilen, Oberkante folgt dem Gelaende
        if nm != 'f': cuts = np.linspace(s0, s1, n_)
        else: cuts = np.linspace(t0, t1, n_)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if nm != 'f': ss, tt = (a, b), (t0, t1)
            else: ss, tt = (s0, s1), (a, b)
            gmin = min(gy(ss[0], tt[0]), gy(ss[1], tt[1]), gy(ss[0], tt[1]), gy(ss[1], tt[0]))
            gmax = max(gy(ss[0], tt[0]), gy(ss[1], tt[1]), gy(ss[0], tt[1]), gy(ss[1], tt[0]))
            add(fr.box('granite_polished', ss[0], ss[1], tt[0], tt[1], gmin - 0.06, gmax + HP, uvscale=1.0), True)
            ex = 0.035
            add(fr.box('granite_polished', ss[0] - (ex if nm == 'f' else 0), ss[1] + (ex if nm == 'f' else 0), tt[0] - (ex if nm != 'f' else 0), tt[1] + (ex if nm != 'f' else 0), gmax + HP, gmax + HP + CAP, uvscale=1.0), True)
        # Gelaender auf der Abdeckung
        if nm == 'l': line = [(s0 + 0.15, (t0 + t1) / 2), (s1 - 0.15, (t0 + t1) / 2)]
        elif nm == 'r': line = [(s0 + 0.15, (t0 + t1) / 2), (s1 - 0.15, (t0 + t1) / 2)]
        else: line = [((s0 + s1) / 2, t0 + 0.1), ((s0 + s1) / 2, t1 - 0.1)]
        S_ = np.linspace(0, 1, 6)
        P3 = [fr.P(line[0][0] + (line[1][0] - line[0][0]) * u, line[0][1] + (line[1][1] - line[0][1]) * u, gy(line[0][0] + (line[1][0] - line[0][0]) * u, line[0][1] + (line[1][1] - line[0][1]) * u) + HP + CAP) for u in S_]
        add(railing_line(P3, 1.5, 0.42, 0.024, True))
    # ---- Tunnel + Stummel-Halle
    Lt = sp['tunnel_len']; Lh = sp['hall_len']; Wh = sp['hall_w']
    s_t0, s_t1 = Lo, Lo + Lt; s_h0, s_h1 = s_t1, s_t1 + Lh
    yc_t = y_floor + HT; yc_h = y_floor + HH
    # Tunnelwaende (Mosaik ueber Granitsockel), Decke mit Lichtband
    for tw_, look in ((-W / 2, 1), (W / 2, -1)):
        add(fr.vquad(sty['skirt'], (s_t0, tw_), (s_t1, tw_), y_floor - 0.3, y_floor + 0.12, (s_t0, 0)), True)
        add(fr.vquad(sty['wall'], (s_t0, tw_), (s_t1, tw_), y_floor + 0.12, yc_t, (s_t0, 0), uvscale=1.0), True)
    add(_ceiling_strips(fr, s_t0, s_t1, -W / 2, W / 2, yc_t, sty), True)
    # Halle
    add(fr.hquad(sty['floor'], s_h0, s_h1, -Wh / 2, -0.25, y_floor, up=True, uvscale=1.2), True, True)
    add(fr.hquad('guidance', s_h0, s_h1 - 1.2, -0.25, 0.25, y_floor, up=True, uvscale=0.5), True, True)
    add(fr.hquad(sty['floor'], s_h1 - 1.2, s_h1, -0.25, 0.25, y_floor, up=True, uvscale=1.2), True, True)
    add(fr.hquad(sty['floor'], s_h0, s_h1, 0.25, Wh / 2, y_floor, up=True, uvscale=1.2), True, True)
    # Tunnel-Leitstreifen bis zur Halle (im Tunnelboden schon enthalten -> nur in der Halle)
    add(_ceiling_strips(fr, s_h0, s_h1, -Wh / 2, Wh / 2, yc_h, sty), True)
    hc = (s_h0 + s_h1) / 2
    def hall_wall(a, b, look):
        add(fr.vquad(sty['skirt'], a, b, y_floor - 0.05, y_floor + 0.12, look), True)
        add(fr.vquad(sty['wall'], a, b, y_floor + 0.12, yc_h - 0.25, look, uvscale=1.0), True)
        add(fr.vquad(sty['cove'], a, b, yc_h - 0.25, yc_h, look), True)
    hall_wall((s_h0, -Wh / 2), (s_h1, -Wh / 2), (hc, 0)); hall_wall((s_h0, Wh / 2), (s_h1, Wh / 2), (hc, 0))
    # Eingangswand (s_h0) mit Tunneloeffnung
    hall_wall((s_h0, -Wh / 2), (s_h0, -W / 2), (hc, 0)); hall_wall((s_h0, W / 2), (s_h0, Wh / 2), (hc, 0))
    add(fr.vquad(sty['wall'], (s_h0, -W / 2), (s_h0, W / 2), yc_t, yc_h, (hc, 0)), True)
    # Stirnwand: geschlossenes Rollgitter (vertieft) + Stationsband + Netzplan
    gw = min(3.2, Wh - 2.0); gh = 2.6
    hall_wall((s_h1, -Wh / 2), (s_h1, -gw / 2), (hc, 0)); hall_wall((s_h1, gw / 2), (s_h1, Wh / 2), (hc, 0))
    add(fr.vquad(sty['wall'], (s_h1, -gw / 2), (s_h1, gw / 2), y_floor + gh, yc_h - 0.25, (hc, 0)), True)
    add(fr.vquad(sty['cove'], (s_h1, -gw / 2), (s_h1, gw / 2), yc_h - 0.25, yc_h, (hc, 0)), True)
    add(fr.vquad('shutter_steel', (s_h1 + 0.25, -gw / 2), (s_h1 + 0.25, gw / 2), y_floor, y_floor + gh, (hc, 0), uvscale=1.0), True)
    for tt in (-gw / 2, gw / 2):
        add(fr.vquad('stainless', (s_h1, tt), (s_h1 + 0.25, tt), y_floor, y_floor + gh, (hc, 0)), True)
    add(fr.hquad('stainless', s_h1, s_h1 + 0.25, -gw / 2, gw / 2, y_floor + gh, up=False), True)
    add(fr.hquad(sty['floor'], s_h1, s_h1 + 0.25, -gw / 2, gw / 2, y_floor, up=True), True, True)
    add(panel_box(fr, sp['sign_mat'], 'paint_dark', s_h1 - 0.05, s_h1 - 0.005, -gw / 2, gw / 2, y_floor + gh + 0.04, y_floor + gh + 0.04 + gw / 8, 's0'))
    add(panel_box(fr, 'netmap', 'stainless', s_h1 - 0.05, s_h1 - 0.005, gw / 2 + 0.35, gw / 2 + 1.55, y_floor + 0.9, y_floor + 2.1, 's0'))
    add(panel_box(fr, 'citylight', 'stainless', s_h0 + 1.5, s_h0 + 2.7, -Wh / 2 + 0.005, -Wh / 2 + 0.06, y_floor + 0.4, y_floor + 2.6, 't1'))
    add(panel_box(fr, 'sign_guide', 'paint_dark', hc - 0.02, hc + 0.02, -Wh / 2 + 0.3, Wh / 2 - 0.3, yc_h - 0.55, yc_h - 0.33, 's0'))
    # Stuetzen (Kroepcke: tuerkisgruenes Glasmosaik, ovale Leuchte)
    if Wh >= 8.0:
        for tt in (-Wh / 4 - 0.3, Wh / 4 + 0.3):
            c = fr.xz(hc + 1.0, tt)
            add(G2.oval_column(sty['column'], c[0], c[1], y_floor - 0.02, yc_h + 0.02, 0.55, 0.38, yaw=math.atan2(fr.d[1], fr.d[0]), uvscale=1.0), True)
            if sp['style'] == 'kroepcke':
                for sg in (-1, 1):
                    cc = fr.xz(hc + 1.0 + sg * 0.555, tt)
                    add(G2.oval_column('lightline', cc[0], cc[1], y_floor + 1.0, y_floor + 2.1, 0.03, 0.12, yaw=math.atan2(fr.d[1], fr.d[0]), seg=10))
    # Tunnel-Bodenleitstreifen
    # U-Mast am Trogkopf
    tsg = (W / 2 + PAR + 0.55) * (1 if sp.get('sign', 'right') == 'right' else -1)
    yb = ground_h(*fr.xz(-0.5, tsg))
    add(u_pylon(fr, -0.5, tsg, yb, sp['sign_mat']), False)
    add(fr.box('collision', -0.57, -0.43, tsg - 0.07, tsg + 0.07, yb - 0.05, yb + 2.75), True)
    INFO.append({'key': key, 'ref': sp['ref'], 'station': sp['station'], 'top': fr.xz(0, 0).tolist(), 'dir': fr.d.tolist(), 'y_top': round(y_top, 3),
                 'y_floor': round(y_floor, 3), 'width': round(W, 2), 'length_open': round(Lo, 2), 'lanes': [l['type'] + ('_' + l.get('run', '') if l['type'] == 'esc' else '') for l in sp['lanes']],
                 'hall_end': fr.xz(s_h1, 0).tolist(), 'note': sp.get('note', ''), 'detail_replaced': sp['in_square']})

# ---------------------------------------------------------------- Glasaufzug Kr.80A
def build_lift(p, door_dir):
    fr = Frame(p, door_dir)            # s Richtung Tuer (Osten)
    yb = ground_h(*p)
    a = 1.0; H = 3.2
    add(fr.box('stainless', -a, a, -a, a, yb - 0.05, yb + 0.08), True, True)
    for s_ in (-a, a):
        for t_ in (-a, a):
            add(fr.box('stainless', s_ - 0.05, s_ + 0.05, t_ - 0.05, t_ + 0.05, yb, yb + H))
    # Glaswaende (doppelseitiges Glas) N/S/W, oben ueber der Tuer
    for (A, B) in (((-a, -a), (a, -a)), ((-a, a), (a, a)), ((-a, -a), (-a, a))):
        add(fr.vquad('glass', A, B, yb + 0.08, yb + H, (0, 0)))
    add(fr.vquad('glass', (a, -a), (a, a), yb + 2.25, yb + H, (5, 0)))
    # Tueren (Edelstahl, zwei Fluegel) + Rahmen
    add(fr.box('stainless', a - 0.04, a + 0.01, -0.58, -0.005, yb + 0.08, yb + 2.2), True)
    add(fr.box('stainless', a - 0.04, a + 0.01, 0.005, 0.58, yb + 0.08, yb + 2.2), True)
    add(fr.box('paint_dark', a - 0.05, a + 0.02, -0.65, 0.65, yb + 2.2, yb + 2.27))
    for t_ in (-a + 0.05, a - 0.05):
        add(fr.box('stainless', a - 0.05, a + 0.02, t_ - 0.03 if t_ < 0 else t_ - 0.37, t_ + 0.37 if t_ < 0 else t_ + 0.03, yb + 0.08, yb + 2.25))
    # Kabine (Rueckwand, Boden, Decke-Licht)
    add(fr.box('stainless', -a + 0.12, -a + 0.16, -0.75, 0.75, yb + 0.15, yb + 2.35))
    add(fr.box('downlight', -0.6, 0.6, -0.6, 0.6, yb + 2.38, yb + 2.42))
    add(G.tube('stainless', np.array([fr.P(-a + 0.25, -0.6, yb + 1.0), fr.P(-a + 0.25, 0.6, yb + 1.0)]), 0.02, 6))
    # Dach: Glasplatte mit dunklem Rahmen + Schriftband 'Kroepcke' rundum
    o = 1.3
    add(fr.box('paint_dark', -o, o, -o, o, yb + H, yb + H + 0.05), True)
    for side in ('s0', 's1', 't0', 't1'):
        pb = panel_box(fr, 'sign_kroepcke', 'paint_dark', -o, o, -o, o, yb + H + 0.05, yb + H + 0.32, side)
        add([q for q in pb if q['mat'] == 'sign_kroepcke'])
    add(fr.hquad('paint_dark', -o, o, -o, o, yb + H + 0.32, up=True))
    add(fr.hquad('paint_dark', -o, o, -o, o, yb + H + 0.05, up=False))
    # Gitterrostmatte vor der Tuer (1 cm)
    add(fr.box('esc_plate', a + 0.05, a + 1.05, -0.7, 0.7, yb - 0.01, yb + 0.01), True, True)
    add(fr.box('collision', -a, a, -a, a, yb, yb + H), True)
    INFO.append({'key': 'lift_Kr.80A', 'p': list(p), 'y': round(yb, 3), 'note': 'Glasaufzug Kr.80A (OSM 3392490912), Tueren nach Osten (Street View)'})

# ---------------------------------------------------------------- Lauf
SPECS = [s for s in L['specs'] if not s.get('skip')]
for sp in SPECS:
    n0 = G.tri_count(parts)
    build_entrance(sp)
    log('built', sp['key'], sp['station'], 'tris', G.tri_count(parts) - n0)
# Aufzuege: Kr.80A freistehend; Kr.79B/Kr.80B liegen im Kroepcke-Center (Gebaeude) -> kein Bauwerk
build_lift((-41.7, 30.85), (1.0, 0.0))     # OSM (-41.7,31.8) um 0.95 m zwischen die Pollerreihen des Detailmodells verschoben
for M in DET.meshes:
    if M['file'] == 'street_furniture':
        near = np.hypot(M["C"][:, 0] + 41.7, M["C"][:, 2] - 30.85) < 1.1
        if near.sum() > 3: log('  WARN lift Kr.80A near street_furniture', M['mat'], int(near.sum()))
pickle.dump({'parts': parts, 'coll': coll, 'walk': walk, 'info': INFO, 'replace': REPLACE, 'mats': DET.mats}, open(CACHE + 'ubahn.pkl', 'wb'))
log('done: entrances', len(SPECS), 'tris', G.tri_count(parts), 'collision', G.tri_count(coll), 'detail materials', list(DET.mats))

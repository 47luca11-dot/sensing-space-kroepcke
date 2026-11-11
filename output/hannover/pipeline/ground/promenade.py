"""Niki-de-Saint-Phalle-Promenade (Passerelle, Ebene -1) -- Ausbau 2.
Offene Trog-Abschnitte (Oeffnungen aus dem DGM, ground_cache/promenade.json) mit Betonfertigteil-Waenden, Bruestung + Stahlrohr-
gelaender an den Laengsseiten, Glasbruestung an den Bruecken; Bruecken (Strassenquerungen) mit Stirnblende, Untersicht und
Leuchten; ueberdeckte Abschnitte mit Lamellendecke + Lichtbaendern und Rundstuetzen; Ladenfronten auf Ebene -1 mit ECHTEN Namen
(data/shops.json, Ebene -1) inkl. Schaufenster mit Auslagen, Glasvordaecher in den offenen Abschnitten; Treppen (OSM: 29 Stufen
Kroepcke-Ende, 4 Brueckentreppen mit Stahlwangen), Nordende mit Treppe + Rolltreppe; Lichtmasten mit Bannern, Baenke, Pflanzkuebel,
Wegweiser; Uebergang in die U-Bahn-Verteilerebene Kroepcke (Sueden) und Richtung Hauptbahnhof (Norden, geschlossene Glastueren).
Ausgabe: data/ground_cache/promenade.pkl"""
import json, time, pickle, math
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString, box
from common import *
import geom as G
import geom2 as G2
from geom2 import Frame
from surface import Surface
from kit import *
from kit import _strip, _ceiling_strips

T0 = time.time()
def log(*a): print('[prom %6.1fs]' % (time.time() - T0), *a, flush=True)
L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
TP = pickle.load(open(CACHE + 'terrain_parts.pkl', 'rb'))
HE = TP['hole_edges']
SURF = Surface()
bz = np.load(CACHE + 'base.npz'); BG = Grid(bz['B'].astype(np.float64), float(bz['x0']), float(bz['z0']), 1.0)
PR = json.load(open(CACHE + 'promenade.json'))
ops = sorted(PR['openings'], key=lambda o: o['s'][0])
AX = PR['axis']; mu = np.array(AX['origin']); ax = np.array(AX['axis'])
fr = Frame(mu, ax)
assert np.allclose(fr.r, AX['normal'], atol=1e-6)
ATL = json.load(open(CACHE + 'shop_atlas.json', encoding='utf-8'))['atlas']
SHOPS = json.load(open(DATA + 'shops.json', encoding='utf-8'))['shops']

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
MATS_EXTRA = {}
for k in range(4):
    MATS_EXTRA['prom_signs_%d' % k] = {'tex': 'tex/prom_signs_%d.png' % k, 'rough': 0.35, 'emissive_tex': True, 'emissive': (0.75, 0.75, 0.75)}
import shoptex
for c in shoptex.CATS:
    MATS_EXTRA['shopint_' + c] = {'tex': 'tex/shopint_%s.jpg' % c, 'rough': 0.8, 'emissive_tex': True, 'emissive': (0.38, 0.38, 0.38)}
# Schaufenster: klares, leicht getoentes Glas (nur von aussen sichtbar) + Auslagen-Farben
MATS_EXTRA['shopglass'] = {'color': (0.16, 0.21, 0.23, 0.12), 'alpha': 'BLEND', 'double': False, 'rough': 0.03, 'metal': 0.0}
for k_, c_ in (('cloth_a', (0.07, 0.09, 0.16)), ('cloth_b', (0.55, 0.06, 0.07)), ('cloth_c', (0.72, 0.64, 0.50)), ('cloth_d', (0.10, 0.10, 0.10)),
               ('cloth_e', (0.85, 0.85, 0.82)), ('cloth_f', (0.20, 0.38, 0.30)), ('mannequin', (0.86, 0.84, 0.80)), ('bread', (0.62, 0.40, 0.18))):
    MATS_EXTRA[k_] = {'color': c_ + (1.0,), 'rough': 0.8}

DECK = 0.9
sc_ = np.array([(o['s'][0] + o['s'][1]) / 2 for o in ops]); fl_ = np.array([o['floor'] for o in ops])
def yfl(s): return float(np.interp(s, sc_, fl_))
def street(s, t):
    q = fr.xz(s, t); return float(SURF([q[0]], [q[1]])[0][0])
def ceil_y(s, t):
    q = fr.xz(s, t); return float(BG(q[0], q[1])) - DECK
SF_H = 3.40           # Ladenfront-Band (Boden bis Oberkante Schild)
CAN_Y = 3.70          # Vordach-Ansatz

# ---------------------------------------------------------------- Abschnitte
S_SOUTH = 8.0; HALL_S = 7.0; S_NORTH = 12.0
secs = []
o0 = ops[0]
secs.append({'kind': 'cov', 'name': 'sued', 's0': o0['s'][0] - S_SOUTH, 's1': o0['s'][0], 't0': o0['t'][0], 't1': o0['t'][1]})
for i, o in enumerate(ops):
    secs.append({'kind': 'open', 'name': 'op%d' % i, 'i': i, 's0': o['s'][0], 's1': o['s'][1], 't0': o['t'][0], 't1': o['t'][1]})
    if i + 1 < len(ops):
        n_ = ops[i + 1]
        secs.append({'kind': 'cov', 'name': 'c%d%d' % (i, i + 1), 's0': o['s'][1], 's1': n_['s'][0], 't0': max(o['t'][0], n_['t'][0]), 't1': min(o['t'][1], n_['t'][1])})
oN = ops[-1]
NORTH_PASS_T0 = oN['t'][0] + 4.34
secs.append({'kind': 'cov', 'name': 'nord', 's0': oN['s'][1], 's1': oN['s'][1] + S_NORTH, 't0': NORTH_PASS_T0, 't1': oN['t'][1]})
for sct in secs:
    sct['bridge'] = sct['kind'] == 'cov' and (sct['s1'] - sct['s0']) < 12 and sct['name'] not in ('sued', 'nord')
    if sct['kind'] == 'cov':
        hh = [ceil_y(s_, t_) - yfl(s_) for s_ in np.linspace(sct['s0'], sct['s1'], 9) for t_ in (sct['t0'], sct['t1'])]
        sct['hcov'] = float(np.clip(min(hh) - 0.04, 2.7, SF_H))
log('sections', [(s_['name'], round(s_['s0'], 1), round(s_['s1'], 1), round(s_['t0'], 1), round(s_['t1'], 1)) for s_ in secs])

# ---------------------------------------------------------------- Treppen / Rolltreppen (lokale Rahmen)
STAIRS = []      # dicts: frame, g, y_top, y_floor, t0,t1 (lokal), side ('E'/'W'), s-Bereich global, top_s (global), glass side
def add_stair(s_top, dir_sign, t_g0, t_g1, kind='stair', name=''):
    """Treppe/Rolltreppe, oben bei s_top (Rim), absteigend in Richtung dir_sign*ax; t_g0..t_g1 globale Querlage"""
    d = ax * dir_sign; f = Frame(fr.xz(s_top, 0.0), d)
    # lokale t: right(d) = dir_sign * nr
    tl = sorted([t_g0 * dir_sign, t_g1 * dir_sign])
    tm = (t_g0 + t_g1) / 2
    y_top = street(s_top - dir_sign * 0.05, tm)
    D = y_top - yfl(s_top)
    g = stair_geom(D, top_landing=0.35) if kind == 'stair' else esc_geom(D)
    run = g['s_end'] if kind == 'stair' else g['s_end']
    st = {'f': f, 'g': g, 'y_top': y_top, 'y_floor': y_top - D, 't0': tl[0], 't1': tl[1], 'kind': kind, 'dir': dir_sign, 's_top': s_top,
          's_lo': min(s_top, s_top + dir_sign * (run + 0.6)), 's_hi': max(s_top, s_top + dir_sign * (run + 0.6)), 'tg': (min(t_g0, t_g1), max(t_g0, t_g1)), 'name': name}
    STAIRS.append(st); return st
# Kroepcke-Ende (OSM 31287201, 29 Stufen, Ostseite, abwaerts nach Norden)
add_stair(ops[0]['s'][0], +1, ops[0]['t'][1] - 2.6, ops[0]['t'][1] - 0.02, name='sued_29')
# Brueckentreppen (OSM 254129194/190/195/192): an der Ostwand hinauf zu den Bruecken
add_stair(ops[1]['s'][1], -1, ops[1]['t'][1] - 1.9, ops[1]['t'][1] - 0.02, name='bruecke1_s')
add_stair(ops[2]['s'][0], +1, ops[2]['t'][1] - 1.9, ops[2]['t'][1] - 0.02, name='bruecke1_n')
add_stair(ops[2]['s'][1], -1, ops[2]['t'][1] - 1.9, ops[2]['t'][1] - 0.02, name='bruecke2_s')
add_stair(ops[3]['s'][0], +1, ops[3]['t'][1] - 1.9, ops[3]['t'][1] - 0.02, name='bruecke2_n')
# Nordende: Treppe (West) + Rolltreppe aufwaerts daneben
add_stair(oN['s'][1], -1, oN['t'][0] + 0.02, oN['t'][0] + 2.62, name='nord_treppe')
add_stair(oN['s'][1], -1, oN['t'][0] + 2.74, oN['t'][0] + 4.19, kind='esc', name='nord_rolltreppe')
for st in STAIRS: log('stair', st['name'], st['kind'], 'D=%.2f' % (st['y_top'] - st['y_floor']), 's %.1f..%.1f' % (st['s_lo'], st['s_hi']), 't %.2f..%.2f' % st['tg'])

def stair_at(s, t):
    for st in STAIRS:
        if st['s_lo'] - 0.05 <= s <= st['s_hi'] + 0.05 and st['tg'][0] - 0.3 <= t <= st['tg'][1] + 0.3: return st
    return None

# ---------------------------------------------------------------- Rim-Waende der Oeffnungen
def sec_at(s, t=None):
    for sc in secs:
        if sc['s0'] - 1e-6 <= s <= sc['s1'] + 1e-6 and (t is None or sc['t0'] - 1e-6 <= t <= sc['t1'] + 1e-6): return sc
    return None
for i, o in enumerate(ops):
    edges0 = HE.get('prom%d' % i, [])
    cen = fr.xz((o['s'][0] + o['s'][1]) / 2, (o['t'][0] + o['t'][1]) / 2)
    # Stirnkanten an Treppen-/Korridorgrenzen teilen (Hoehen linear)
    edges = []
    for e in edges0:
        pu, pw, hu, hw, c = e
        su, tu = fr.st(*pu); sw, tw = fr.st(*pw)
        if not (abs(su - sw) < 0.05 and (abs(su - o['s'][0]) < 0.06 or abs(su - o['s'][1]) < 0.06)):
            edges.append(e); continue
        cuts = set()
        for st in STAIRS:
            if abs(st['s_top'] - su) < 0.1: cuts |= {st['tg'][0], st['tg'][1]}
        for sc in secs:
            if sc['kind'] == 'cov' and (abs(sc['s0'] - su) < 0.1 or abs(sc['s1'] - su) < 0.1): cuts |= {sc['t0'], sc['t1']}
        lo, hi = min(tu, tw), max(tu, tw)
        cs = sorted([tu, tw] + [x for x in cuts if lo + 1e-3 < x < hi - 1e-3])
        if tu > tw: cs = cs[::-1]
        for a_, b_ in zip(cs[:-1], cs[1:]):
            fa = (a_ - tu) / (tw - tu); fb = (b_ - tu) / (tw - tu)
            pa = np.array(pu) + (np.array(pw) - np.array(pu)) * fa; pb = np.array(pu) + (np.array(pw) - np.array(pu)) * fb
            edges.append((pa.tolist(), pb.tolist(), hu + (hw - hu) * fa, hu + (hw - hu) * fb, c))
    long_e = []; end_full = []; end_fas = []; end_stair = []
    for e in edges:
        m = (np.array(e[0]) + np.array(e[1])) / 2; s, t = fr.st(*m)
        if abs(s - o['s'][0]) < 0.06 or abs(s - o['s'][1]) < 0.06:
            st = stair_at(s, t)
            if st is not None and abs(s - st['s_top']) < 0.1 and st['tg'][0] - 0.01 <= t <= st['tg'][1] + 0.01: end_stair.append((e, st)); continue
            nb = [sc for sc in secs if sc['kind'] == 'cov' and (abs(sc['s1'] - s) < 0.06 or abs(sc['s0'] - s) < 0.06) and sc['t0'] - 0.01 <= t <= sc['t1'] + 0.01]
            (end_fas if nb else end_full).append(e)
        else:
            long_e.append(e)
    # Laengsseiten: obere Wandzone (Betonfertigteile) bis Vordach-Hoehe; darunter Ladenfronten (eigener Abschnitt)
    add(wall_from_edges(long_e, lambda x, z: yfl(fr.st(x, z)[0]) + CAN_Y, 'concrete_panel', cen, 5.0), True)
    add(wall_from_edges(end_full, lambda x, z: yfl(fr.st(x, z)[0]) - 0.05, 'concrete_panel', cen, 5.0), True)
    add(wall_from_edges(end_fas, lambda x, z: float(BG(x, z)) - DECK, 'concrete_panel', cen, 5.0), True)
    # Treppenkopf: Podest vom Rim bis zur ersten Stufe
    for e, st in end_stair:
        f = st['f']; pu, pw, hu, hw, c = e
        su, tu = f.st(*pu); sw, tw = f.st(*pw)
        if st['kind'] == 'stair':
            dep = 0.35
            q = G2.quad('tactile', f.P(0, tu, hu), f.P(0, tw, hw), f.P(dep - 0.05, tw, st['y_top']), f.P(dep - 0.05, tu, st['y_top']), uv=np.array([[tu, 0], [tw, 0], [tw, dep], [tu, dep]]) / 0.6)
            if q is not None:
                if np.cross(q['pos'][1] - q['pos'][0], q['pos'][3] - q['pos'][0])[1] < 0: q['idx'] = q['idx'][:, ::-1]
                q['nrm'] = np.tile([0, 1.0, 0], (4, 1)); add(q, True, True)
            add(f.hquad('nosing', dep - 0.05, dep, min(tu, tw), max(tu, tw), st['y_top']), True, True)
        else:
            lo_ = min(hu, hw, st['y_top']) - 0.04; hi_ = max(hu, hw, st['y_top'])
            add(f.vquad('stainless', (0, tu), (0, tw), lo_, hi_, (-1, (tu + tw) / 2))); add(f.vquad('stainless', (0, tu), (0, tw), lo_, hi_, (1, (tu + tw) / 2)))
    log('op', i, 'edges long', len(long_e), 'end full', len(end_full), 'fascia', len(end_fas), 'stair heads', len(end_stair))

# ---------------------------------------------------------------- Treppen bauen
for st in STAIRS:
    f, g = st['f'], st['g']; t0, t1 = st['t0'], st['t1']
    if st['kind'] == 'stair':
        add(build_stair(f, g, st['y_top'], st['y_floor'], t0, t1, None), True, True)
        NL = stair_nose_line(g, st['y_top'])
        s_last = g['s_riser'][-1]
        add(f.hquad('tactile', s_last, s_last + 0.6, t0, t1, st['y_floor']), True, True)
        # Wangen (Stahl) + Untersicht; offene Seite mit Glasbruestung, Wandseite Handlauf
        for tw_, open_side in ((t0, True), (t1, False)) if st['dir'] > 0 else ((t0, False), (t1, True)):
            pass
        wall_t = max(t0, t1, key=lambda tt: abs(f.xz(0, tt) @ fr.r - (fr.o @ fr.r)))   # Wandseite = groesstes globales t (Ostwand) bzw. West
        tg_wall = st['tg'][1] if st['name'] != 'nord_treppe' else st['tg'][0]
        t_wall = tg_wall * st['dir']; t_open = (t0 + t1) - t_wall
        sgn = 1 if t_open > t_wall else -1
        # Stahlwange (offene Seite) entlang Nasenlinie, 0.35 m hoch, und Untersicht
        S = NL[:, 0]; Y = NL[:, 1]
        A = np.array([f.P(s_, t_open + sgn * 0.02, y - 0.38) for s_, y in zip(S, Y)]); B = np.array([f.P(s_, t_open + sgn * 0.02, y + 0.02) for s_, y in zip(S, Y)])
        add(_strip('paint_grey', A, B, look=f.P(S.mean(), t_open + sgn * 3, Y.mean())), True)
        A2 = np.array([f.P(s_, t_wall, y - 0.38) for s_, y in zip(S, Y)]); B2 = np.array([f.P(s_, t_open + sgn * 0.02, y - 0.38) for s_, y in zip(S, Y)])
        add(_strip('paint_grey', A2, B2, look=f.P(S.mean(), (t0 + t1) / 2, Y.mean() - 3)))
        # Glasbruestung + Edelstahlhandlauf (offene Seite), Wandhandlauf
        Sg = S[1:-1]; Yg = Y[1:-1]
        A = np.array([f.P(s_, t_open + sgn * 0.02, y + 0.05) for s_, y in zip(Sg, Yg)]); B = np.array([f.P(s_, t_open + sgn * 0.02, y + 0.95) for s_, y in zip(Sg, Yg)])
        add(_strip('glass', A, B, look=f.P(Sg.mean(), (t0 + t1) / 2, Yg.mean())))
        add(G.tube('stainless', np.array([f.P(s_, t_open + sgn * 0.02, y + 1.0) for s_, y in zip(S, Y)]), 0.024, 8))
        for s_, y in NL[1:-1:4]: add(G2.cyl_between('stainless', f.P(s_, t_open + sgn * 0.02, y), f.P(s_, t_open + sgn * 0.02, y + 1.0), 0.018, 6))
        add(G.tube('stainless', np.array([f.P(s_, t_wall - sgn * 0.08, y + 0.9) for s_, y in zip(S, Y)]), 0.022, 8))
        for s_, y in NL[1:-1:5]: add(G2.cyl_between('stainless', f.P(s_, t_wall, y + 0.86), f.P(s_, t_wall - sgn * 0.08, y + 0.86), 0.012, 5))
        # Kollision offene Seite (Gelaender)
        A = np.array([f.P(s_, t_open + sgn * 0.02, y) for s_, y in zip(S, Y)]); B = np.array([f.P(s_, t_open + sgn * 0.02, y + 1.0) for s_, y in zip(S, Y)])
        coll.append(_strip('collision', A, B, look=f.P(S.mean(), t_open + sgn * 3, Y.mean())))
    else:
        vis, col = build_escalator(f, dict(g, t0=t0, t1=t1), st['y_top'], st['y_floor'], t0, t1, edge_wall_l=False, edge_wall_r=False)
        add(vis); add([c for c in col if c is not None], True); walk.extend([c for c in col if c is not None])
log('stairs built, tris', G.tri_count(parts))

# ---------------------------------------------------------------- Boeden, Decken, Waende je Abschnitt
def floor_strips(sc, mat, uvs=1.2, drain=False):
    s0, s1, t0, t1 = sc['s0'], sc['s1'], sc['t0'], sc['t1']
    n = max(1, int(math.ceil((s1 - s0) / 2.0))); S = np.linspace(s0, s1, n + 1)
    bands = [(t0, t1)]
    if drain:
        tm = (t0 + t1) / 2; bands = [(t0, tm - 0.08), (tm - 0.08, tm + 0.08), (tm + 0.08, t1)]
    out = []
    for bi, (a, b) in enumerate(bands):
        m = 'stainless' if (drain and bi == 1) else mat
        for sa, sb in zip(S[:-1], S[1:]):
            out.append(fr.hquad(m, sa, sb, a, b, 0, up=True, uvscale=uvs, y_s=(yfl(sa), yfl(sb))))
    return out
for sc in secs:
    add(floor_strips(sc, 'granite_dark' if sc['kind'] == 'open' else 'stone_light', 2.4 if sc['kind'] == 'open' else 2.0, drain=sc['kind'] == 'open'), True, True)
    if sc['kind'] != 'cov': continue
    s0, s1, t0, t1 = sc['s0'], sc['s1'], sc['t0'], sc['t1']
    # Decke (folgt der Strassenunterkante): Lamellen + Lichtbaender; Bruecken: Beton-Untersicht mit Einbauleuchten
    n = max(1, int(math.ceil((s1 - s0) / 2.0))); S = np.linspace(s0, s1, n + 1)
    W = t1 - t0; k = max(1, int(round(W / 2.2))); E = np.linspace(t0, t1, k + 1)
    for sa, sb in zip(S[:-1], S[1:]):
        for a, b in zip(E[:-1], E[1:]):
            m_ = (a + b) / 2
            for (ta, tb, mat) in ((a, m_ - 0.06, 'slats_dark' if not sc['bridge'] else 'concrete'), (m_ - 0.06, m_ + 0.06, 'lightline'), (m_ + 0.06, b, 'slats_dark' if not sc['bridge'] else 'concrete')):
                if sc['bridge'] and mat == 'lightline':
                    mat = 'concrete'
                P = [fr.P(sa, ta, ceil_y(sa, ta)), fr.P(sb, ta, ceil_y(sb, ta)), fr.P(sb, tb, ceil_y(sb, tb)), fr.P(sa, tb, ceil_y(sa, tb))]
                q = G2.quad(mat, *P, uv=np.array([[sa, ta], [sb, ta], [sb, tb], [sa, tb]]))
                if q is not None and q['nrm'][0][1] > 0: q['idx'] = q['idx'][:, ::-1]; q['nrm'] *= -1
                add(q)
    if sc['bridge']:
        for s_ in np.arange(s0 + 1.5, s1 - 1.0, 2.5):
            for t_ in np.linspace(t0 + 1.2, t1 - 1.2, max(2, int(W / 2.5))):
                y = ceil_y(s_, t_)
                add(fr.box('downlight', s_ - 0.2, s_ + 0.2, t_ - 0.2, t_ + 0.2, y - 0.03, y + 0.01))
    # Rundstuetzen im langen ueberdeckten Abschnitt
    if not sc['bridge'] and sc['name'] not in ('sued', 'nord') and W > 7.5:
        for s_ in np.arange(s0 + 6.0, s1 - 4.0, 9.0):
            # ovale Stuetzen mit gruenem Glasmosaik + eingelassener Ovalleuchte (wie B-Ebene Kroepcke, SV-Notiz), Granitsockel
            c = fr.xz(s_, (t0 + t1) / 2); yaw_ = math.atan2(fr.d[1], fr.d[0]); y0_ = yfl(s_); y1_ = ceil_y(s_, (t0 + t1) / 2)
            add(G2.oval_column('granite_dark', c[0], c[1], y0_ - 0.02, y0_ + 0.15, 0.62, 0.42, yaw=yaw_, seg=24, cap=True), True)
            add(G2.oval_column('mosaic_green', c[0], c[1], y0_ + 0.15, y1_ + 0.04, 0.58, 0.38, yaw=yaw_, seg=24), True)
            for sg_ in (-1, 1):
                cc = fr.xz(s_ + sg_ * 0.585, (t0 + t1) / 2)
                add(G2.oval_column('lightline', cc[0], cc[1], y0_ + 1.0, y0_ + 2.2, 0.03, 0.12, yaw=yaw_, seg=10))

# Stirnwaende der Enden: Sued -> Verteilerebene Kroepcke (offenes Portal + Halle), Nord -> Glastueren Hauptbahnhof
def hall(s0, s1, t0, t1, yf, H, sign_mat):
    """Verteilerebenen-Stummel (Kroepcke-Stil), offen zur Passerelle bei s1"""
    out = []
    sc = (s0 + s1) / 2; tc = (t0 + t1) / 2
    out.append(fr.hquad('tile_beige', s0, s1, t0, t1, yf, up=True, uvscale=1.2))
    out += _ceiling_strips(fr, s0, s1, t0, t1, yf + H, {'ceil': 'slats'})
    for a, b in (((s0, t0), (s1, t0)), ((s0, t1), (s1, t1)), ((s0, t0), (s0, t1))):
        out.append(fr.vquad('granite_dark', a, b, yf - 0.05, yf + 0.12, (sc, tc)))
        out.append(fr.vquad('mosaic_yg', a, b, yf + 0.12, yf + H - 0.25, (sc, tc)))
        out.append(fr.vquad('lightcove', a, b, yf + H - 0.25, yf + H, (sc, tc)))
    for tt in np.linspace(t0 + 2.5, t1 - 2.5, 2):
        c = fr.xz(sc, tt)
        out.append(G2.oval_column('mosaic_teal', c[0], c[1], yf - 0.02, yf + H + 0.02, 0.55, 0.38, yaw=math.atan2(fr.d[1], fr.d[0])))
        for sg in (-1, 1):
            cc = fr.xz(sc + sg * 0.555, tt)
            out.append(G2.oval_column('lightline', cc[0], cc[1], yf + 1.0, yf + 2.1, 0.03, 0.12, yaw=math.atan2(fr.d[1], fr.d[0]), seg=10))
    gw = 3.2
    out.append(fr.vquad('shutter_steel', (s0 - 0.25, tc - gw / 2), (s0 - 0.25, tc + gw / 2), yf, yf + 2.6, (sc, tc), uvscale=1.0))
    out += panel_box(fr, sign_mat, 'paint_dark', s0 + 0.005, s0 + 0.05, tc - 1.6, tc + 1.6, yf + 2.65, yf + 3.05, 's1')
    out += panel_box(fr, 'netmap', 'stainless', s0 + 0.005, s0 + 0.05, tc + 2.0, tc + 3.2, yf + 0.9, yf + 2.1, 's1')
    return out
sS = secs[0]; yS = yfl(sS['s0']); Hh = min(3.2, ceil_y(sS['s0'], (sS['t0'] + sS['t1']) / 2) - yS - 0.05)
hall_parts = hall(sS['s0'] - HALL_S, sS['s0'], sS['t0'], sS['t1'], yS, Hh, 'sign_kroepcke')
# Hallenrueckwand hat Rollgitter: Wand in drei Stuecke um das Gitter
tcS = (sS['t0'] + sS['t1']) / 2
hall_parts = [p for p in hall_parts if not (p['mat'] in ('mosaic_yg', 'granite_dark', 'lightcove') and np.allclose(np.array([fr.st(*q[[0, 2]])[0] for q in p['pos']]), sS['s0'] - HALL_S, atol=0.01))]
for (a, b) in (((sS['s0'] - HALL_S, sS['t0']), (sS['s0'] - HALL_S, tcS - 1.6)), ((sS['s0'] - HALL_S, tcS + 1.6), (sS['s0'] - HALL_S, sS['t1']))):
    hall_parts.append(fr.vquad('mosaic_yg', a, b, yS - 0.05, yS + Hh - 0.25, (sS['s0'], tcS)))
    hall_parts.append(fr.vquad('lightcove', a, b, yS + Hh - 0.25, yS + Hh, (sS['s0'], tcS)))
s_e = sS['s0'] - HALL_S
hall_parts.append(fr.vquad('mosaic_yg', (s_e, tcS - 1.6), (s_e, tcS + 1.6), yS + 2.6, yS + Hh, (sS['s0'], tcS)))
for tt in (tcS - 1.6, tcS + 1.6): hall_parts.append(fr.vquad('stainless', (s_e - 0.25, tt), (s_e, tt), yS, yS + 2.6, (sS['s0'], tcS)))
hall_parts.append(fr.hquad('stainless', s_e - 0.25, s_e, tcS - 1.6, tcS + 1.6, yS + 2.6, up=False))
hall_parts.append(fr.hquad('tile_beige', s_e - 0.25, s_e, tcS - 1.6, tcS + 1.6, yS, up=True))
add(hall_parts, True)
walk.append(hall_parts[0])
# Uebergang Halle (Decke Hh) -> Suedabschnitt (Decke ceil_y): Sturzblende
for sa, sb in ((sS['t0'], sS['t1']),):
    pass
# Nord: Glastueren + Glaswand am Ende des Durchgangs
sN = secs[-1]; s_n1 = sN['s1']
for t_ in np.linspace(sN['t0'], sN['t1'], 5)[:-1]:
    tb = t_ + (sN['t1'] - sN['t0']) / 4
    add(fr.vquad('glass', (s_n1, t_), (s_n1, tb), yfl(s_n1), ceil_y(s_n1, t_) - 0.2, (s_n1 - 2, t_)))
    add(fr.box('paint_dark', s_n1 - 0.04, s_n1 + 0.04, t_ - 0.03, t_ + 0.03, yfl(s_n1), ceil_y(s_n1, t_)))
add(fr.box('paint_dark', s_n1 - 0.04, s_n1 + 0.04, sN['t0'], sN['t1'], ceil_y(s_n1, sN['t0']) - 0.2, ceil_y(s_n1, sN['t0'])))
add(fr.vquad('interior_dark', (s_n1 + 3, sN['t0']), (s_n1 + 3, sN['t1']), yfl(s_n1), ceil_y(s_n1, sN['t0']), (s_n1, 0)))
add(fr.hquad('stone_light', s_n1, s_n1 + 3, sN['t0'], sN['t1'], yfl(s_n1), up=True))
for a, b in (((s_n1, sN['t0']), (s_n1 + 3, sN['t0'])), ((s_n1, sN['t1']), (s_n1 + 3, sN['t1']))):
    add(fr.vquad('stone_light', a, b, yfl(s_n1), ceil_y(s_n1, sN['t0']), (s_n1 + 1.5, (sN['t0'] + sN['t1']) / 2)))
add(fr.hquad('lightline', s_n1, s_n1 + 3, sN['t0'], sN['t1'], ceil_y(s_n1, sN['t0']), up=False))
coll.append(fr.box('collision', s_n1 - 0.05, s_n1 + 0.05, sN['t0'], sN['t1'], yfl(s_n1), yfl(s_n1) + 3))
add(panel_box(fr, 'sign_hbf', 'paint_dark', s_n1 - 0.6, s_n1 - 0.55, sN['t0'] + 0.3, sN['t1'] - 0.3, ceil_y(s_n1, sN['t0']) - 0.55, ceil_y(s_n1, sN['t0']) - 0.15, 's0'))

# ---------------------------------------------------------------- Ladenfronten
def shop_units():
    """Ladeneinheiten je Seite: (side, s0, s1, shop|None)"""
    cand = {'W': [], 'E': []}
    smin, smax = secs[0]['s0'], secs[-1]['s1']
    for sh in SHOPS:
        if not (sh.get('context') in ('promenade', 'ubahn_passage') or sh.get('level') == -1): continue
        w = (sh.get('facade') or {}).get('wall_point') or sh['p']
        s, t = fr.st(*w)
        if not (smin + 1 < s < smax - 1): continue
        sc = sec_at(s)
        if sc is None: continue
        tm = (sc['t0'] + sc['t1']) / 2
        if abs(t - tm) > (sc['t1'] - sc['t0']) / 2 + 3.0: continue          # andere Passage (z. B. B-Ebene)
        side = 'W' if t < tm else 'E'
        wdt = float(np.clip(sh.get('frontage_width_m') or 6.0, 3.6, 10.0))
        cand[side].append([s - wdt / 2, s + wdt / 2, sh])
    out = []
    for side in ('W', 'E'):
        C = sorted(cand[side], key=lambda c: (c[0] + c[1]) / 2)
        # Ueberlappungen aufloesen
        for a, b in zip(C[:-1], C[1:]):
            if a[1] > b[0]:
                m = ((a[0] + a[1]) / 2 + (b[0] + b[1]) / 2) / 2
                a[1] = max(a[0] + 2.5, m); b[0] = min(b[1] - 2.5, m)
        # Luecken fuellen
        s = smin; res = []
        for a in C:
            if a[0] - s >= 3.0:
                n = max(1, int((a[0] - s) / 7.0))
                for k in range(n): res.append((side, s + (a[0] - s) * k / n, s + (a[0] - s) * (k + 1) / n, None))
            res.append((side, max(a[0], s), a[1], a[2])); s = a[1]
        if smax - s >= 3.0:
            n = max(1, int((smax - s) / 7.0))
            for k in range(n): res.append((side, s + (smax - s) * k / n, s + (smax - s) * (k + 1) / n, None))
        out += res
    return out

UNITS = shop_units()
log('shop units', len(UNITS), 'named', sum(1 for u in UNITS if u[3] is not None))
rngu = np.random.default_rng(11)
def side_t(sc, side): return sc['t0'] if side == 'W' else sc['t1']

def _uv01(q, y0, y1, urep=1.0):
    """Bild-UV 0..1 auf einer senkrechten Flaeche (u entlang der Flaeche, v von oben)"""
    P = q['pos']; nrm = q['nrm'][0]; right = np.cross([0, 1.0, 0], nrm); right /= np.linalg.norm(right) + 1e-12
    u = P @ right; u = (u - u.min()) / (u.max() - u.min() + 1e-9); v = 1 - (P[:, 1] - y0) / (y1 - y0)
    q['uv'] = np.c_[u * urep, v]
    return q

CLOTH = ['cloth_a', 'cloth_b', 'cloth_c', 'cloth_d', 'cloth_e', 'cloth_f']
def mannequin(s, t, y0, m_top, m_leg):
    """Schaufensterpuppe (Sockelplatte, Beine, Rumpf, Hals, Kopf) bei lokalem (s,t) auf Hoehe y0"""
    o = []; c = fr.xz(s, t); yaw = math.atan2(fr.d[1], fr.d[0])
    o.append(G.cylinder('paint_dark', (c[0], y0, c[1]), 0.16, 0.16, 0.02, 12, cap=True))
    o.append(G.cylinder('stainless', (c[0], y0 + 0.02, c[1]), 0.012, 0.012, 0.12, 6, cap=False))
    for e in (-0.075, 0.075):
        cc = fr.xz(s + e, t)
        o.append(G.cylinder(m_leg, (cc[0], y0 + 0.12, cc[1]), 0.055, 0.07, 0.80, 8, cap=False))
    o.append(G2.oval_column(m_leg, c[0], c[1], y0 + 0.86, y0 + 1.02, 0.17, 0.11, yaw=yaw, seg=12, cap=False))
    o.append(G2.oval_column(m_top, c[0], c[1], y0 + 1.0, y0 + 1.48, 0.18, 0.11, yaw=yaw, seg=12, cap=False))
    o.append(G.cylinder(m_top, (c[0], y0 + 1.48, c[1]), 0.18, 0.07, 0.07, 12, cap=True))
    for e in (-0.205, 0.205):                                    # Arme
        cc = fr.xz(s + e, t)
        o.append(G.cylinder(m_top, (cc[0], y0 + 0.88, cc[1]), 0.035, 0.045, 0.58, 6, cap=True))
    o.append(G.cylinder('mannequin', (c[0], y0 + 1.52, c[1]), 0.035, 0.035, 0.08, 6, cap=False))
    o.append(G.cylinder('mannequin', (c[0], y0 + 1.59, c[1]), 0.085, 0.07, 0.22, 10, cap=True))
    return o

def shop_props(pa, pb, dm, yb, cat, T, sg):
    """Auslagen hinter dem Schaufenster: Podest im Fenster + kategorietypische Einbauten (Puppen, Theke, Regale, Tische).
    T(d): t-Koordinate in d m Tiefe hinter der Glasebene."""
    o = []
    lk = lambda d_: (dm, T(d_ - 3.0))                             # Blick von der Passage
    segs = [(a, b) for a, b in ((pa + 0.12, dm - 1.0), (dm + 1.0, pb - 0.12)) if b - a >= 0.8]
    food = cat in ('bakery', 'cafe', 'fast_food')
    shelf = cat in ('pharmacy', 'chemist', 'kiosk', 'generic', 'mobile_phone', 'beauty')
    yP = yb + 0.32
    if not food:
        for a, b in segs:                                          # Fensterpodest
            o.append(fr.box('stone_light' if cat != 'clothes' else 'wood', a, b, T(0.06), T(0.85), yb, yP))
    if cat in ('clothes', 'florist'):
        for a, b in segs:
            n = int(np.clip((b - a) / 1.1, 1, 3))
            for k in range(n):
                s_ = a + (b - a) * (k + 0.5) / n
                if cat == 'florist':
                    for j in range(3):
                        c = fr.xz(s_ + (j - 1) * 0.28, T(0.45))
                        o.append(G.cylinder('paint_dark', (c[0], yP, c[1]), 0.11, 0.13, 0.30, 10, cap=False))
                        o.append(G.cylinder(CLOTH[(k + j) % 3 + 1] if j != 1 else 'hedge', (c[0], yP + 0.30, c[1]), 0.16, 0.05, 0.35, 8, cap=True))
                else:
                    o += mannequin(s_, T(0.45), yP, CLOTH[int(rngu.integers(len(CLOTH)))], CLOTH[int(rngu.integers(len(CLOTH)))])
        if cat == 'clothes':                                       # Kleiderstange mit Ware an der Rueckwand
            y1 = yb + 1.75
            o.append(G2.cyl_between('stainless', fr.P(pa + 0.4, T(2.25), y1), fr.P(pb - 0.4, T(2.25), y1), 0.015, 6))
            s_ = pa + 0.45
            while s_ < pb - 0.5:
                w = float(rngu.uniform(0.05, 0.09))
                o.append(fr.box(CLOTH[int(rngu.integers(len(CLOTH)))], s_, s_ + w, T(2.0), T(2.5), y1 - float(rngu.uniform(0.6, 1.0)), y1 - 0.02))
                s_ += w + 0.012
    if food:
        # Theke (dunkel, Steinplatte) + Glasvitrine, Hochtische am Fenster
        a, b = pa + 0.35, pb - 0.35
        o.append(fr.box('paint_dark', a, b, T(1.55), T(2.15), yb, yb + 0.95))
        o.append(fr.box('stone_light', a - 0.03, b + 0.03, T(1.50), T(2.18), yb + 0.95, yb + 0.99))
        q = fr.vquad('shopglass', (a, T(1.55)), (b, T(1.55)), yb + 0.99, yb + 1.30, lk(1.55)); o.append(q)
        o.append(fr.box('stainless', a, b, T(1.53), T(2.15), yb + 1.30, yb + 1.32))
        s_ = a + 0.2
        while s_ < b - 0.4:
            o.append(fr.box('bread' if cat == 'bakery' else CLOTH[int(rngu.integers(len(CLOTH)))], s_, s_ + 0.32, T(1.62), T(1.95), yb + 0.99, yb + 1.06 + float(rngu.uniform(0.0, 0.08))))
            s_ += 0.42
        for a2, b2 in segs:
            n = int(np.clip((b2 - a2) / 1.4, 1, 2))
            for k in range(n):
                c = fr.xz(a2 + (b2 - a2) * (k + 0.5) / n, T(0.5))
                o.append(G.cylinder('paint_dark', (c[0], yb, c[1]), 0.20, 0.20, 0.02, 12, cap=True))
                o.append(G.cylinder('stainless', (c[0], yb + 0.02, c[1]), 0.03, 0.03, 1.06, 6, cap=False))
                o.append(G.cylinder('wood', (c[0], yb + 1.08, c[1]), 0.32, 0.32, 0.03, 16, cap=True))
    if shelf:
        # Wandregale an beiden Seitenwaenden + Mittelgondel, Fronten mit Warenbild
        img = 'shopint_' + cat if cat != 'mobile_phone' else 'shopint_chemist'
        for s_w, sgn in ((pa + 0.02, 1), (pb - 0.02, -1)):
            o.append(fr.box('paint_grey', min(s_w, s_w + sgn * 0.42), max(s_w, s_w + sgn * 0.42), T(0.95), T(2.55), yb, yb + 2.0))
            q = fr.vquad(img, (s_w + sgn * 0.425, T(0.98)), (s_w + sgn * 0.425, T(2.52)), yb + 0.08, yb + 1.95, (dm, T(1.7)))
            o.append(_uv01(q, yb + 0.08, yb + 1.95, 0.5))
        if pb - pa > 4.2:
            a, b = dm - 0.9, dm + 0.9
            o.append(fr.box('paint_grey', a, b, T(1.35), T(1.75), yb, yb + 1.35))
            q = fr.vquad(img, (a, T(1.345)), (b, T(1.345)), yb + 0.08, yb + 1.30, lk(1.345))
            o.append(_uv01(q, yb + 0.08, yb + 1.30, 0.6))
        if cat == 'mobile_phone':
            for a, b in segs:
                n = int(np.clip((b - a) / 0.35, 1, 6))
                for k in range(n):
                    s_ = a + (b - a) * (k + 0.5) / n
                    o.append(fr.box('black_glass', s_ - 0.04, s_ + 0.04, T(0.38), T(0.42), yP, yP + 0.16))
                    o.append(fr.box('paint_dark', s_ - 0.05, s_ + 0.05, T(0.42), T(0.50), yP, yP + 0.02))
        else:
            for a, b in segs:                                      # Warenaufbau im Fenster
                s_ = a + 0.1
                while s_ < b - 0.25:
                    w = float(rngu.uniform(0.14, 0.3)); h = float(rngu.uniform(0.15, 0.45))
                    o.append(fr.box(CLOTH[int(rngu.integers(len(CLOTH)))], s_, s_ + w, T(0.3), T(0.6), yP, yP + h))
                    s_ += w + float(rngu.uniform(0.08, 0.25))
    return [p for p in o if p is not None]
def build_unit(side, a, b, sh, sc):
    """Ladenfront im Abschnitt sc zwischen s=a..b auf Seite side. Wandebene t_w; Schaufenster 0.20 m zurueckgesetzt."""
    out = []; col = []
    tw = side_t(sc, side); sg = 1 if side == 'W' else -1          # sg: Richtung in die Passage (+t fuer West)
    inward = lambda tt: tw + sg * tt                            # tt > 0: in die Passage, < 0: in den Laden
    yf_a, yf_b = yfl(a), yfl(b)
    look = ((a + b) / 2, inward(3.0))
    top_band = sc.get('hcov', SF_H)                             # ueberdeckt: Ladenband unter der Decke
    G1 = min(2.75, top_band - 0.5); K1 = G1 + 0.10             # Glas-Oberkante, Kaempfer-Oberkante (= Innendecke)
    # Pfeiler (0.15 m je Seite) + Laibungen
    pa, pb = a + 0.15, b - 0.15
    if pb - pa < 1.0: return out, col
    for s0_, s1_ in ((a, pa), (pb, b)):
        out.append(fr.vquad('concrete_panel' if sc['kind'] == 'open' else 'stone_light', (s0_, tw), (s1_, tw), yfl(s0_) - 0.05, yfl(s0_) + top_band, look, uvscale=2.5))
    R = 0.20
    for s_, lk in ((pa, (pa + 1, inward(-0.1))), (pb, (pb - 1, inward(-0.1)))):
        out.append(fr.vquad('concrete_panel' if sc['kind'] == 'open' else 'stone_light', (s_, tw), (s_, inward(-R)), yfl(s_) - 0.05, yfl(s_) + top_band, lk))
    out.append(fr.hquad('concrete_panel', pa, pb, min(tw, inward(-R)), max(tw, inward(-R)), yfl((pa + pb) / 2) + top_band, up=False))
    tg = inward(-R)
    yb = (yf_a + yf_b) / 2
    # Sockel, Glas, Kaempfer, Schildband
    out.append(fr.vquad('paint_dark', (pa, tg), (pb, tg), yb - 0.05, yb + 0.15, look))
    vac = sh is None or sh.get('status') in ('closed', 'vacant')
    cat = 'vacant' if vac and (sh is None or rngu.random() < 0.6) else shoptex.cat_of((sh or {}).get('category'))
    if sh is None and rngu.random() < 0.5: cat = 'generic'
    out.append(fr.vquad('shopglass', (pa, tg), (pb, tg), yb + 0.15, yb + G1, look))
    out.append(fr.vquad('paint_dark', (pa, tg), (pb, tg), yb + G1, yb + K1, look))
    # Pfosten alle ~1.6 m, Tuer mittig (Rahmen + Griffe)
    nm = max(1, int((pb - pa) / 1.6))
    for k in range(1, nm):
        s_ = pa + (pb - pa) * k / nm
        out.append(fr.box('paint_dark', s_ - 0.03, s_ + 0.03, min(tg, inward(-R - 0.06)), max(tg, inward(-R - 0.06)), yb + 0.15, yb + G1))
    dm = (pa + pb) / 2
    if not vac:
        for s_ in (dm - 0.9, dm, dm + 0.9):
            out.append(fr.box('paint_dark', s_ - 0.03, s_ + 0.03, min(tg, inward(-R + 0.04)), max(tg, inward(-R + 0.04)), yb + 0.15, yb + 2.3))
        out.append(fr.box('paint_dark', dm - 0.9, dm + 0.9, min(tg, inward(-R + 0.04)), max(tg, inward(-R + 0.04)), yb + 2.27, yb + 2.33))
        for s_ in (dm - 0.15, dm + 0.15):
            out.append(G2.cyl_between('stainless', fr.P(s_, inward(-R + 0.08), yb + 0.9), fr.P(s_, inward(-R + 0.08), yb + 1.6), 0.015, 6))
    # Schild (Atlas) mittig im Schildband, Rest dunkel
    yS0, yS1 = yb + K1, yb + top_band
    key = (sh or {}).get('id') if not vac else None
    if key in ATL:
        atl, u0, v0, u1, v1 = ATL[key]
        sw = min(pb - pa - 0.3, (yS1 - yS0) * 8.0)
        s0s, s1s = dm - sw / 2, dm + sw / 2
        q = fr.vquad(atl, (s0s, tg), (s1s, tg), yS0, yS1, look)
        P = q['pos']; nrm = q['nrm'][0]; right = np.cross([0, 1.0, 0], nrm); right /= np.linalg.norm(right)
        u = P @ right; u = (u - u.min()) / (u.max() - u.min() + 1e-9); v = 1 - (P[:, 1] - yS0) / (yS1 - yS0)
        q['uv'] = np.c_[u0 + u * (u1 - u0), v0 + v * (v1 - v0)]
        out.append(q)
        for (x0_, x1_) in ((pa, s0s), (s1s, pb)):
            if x1_ - x0_ > 0.01: out.append(fr.vquad('paint_dark', (x0_, tg), (x1_, tg), yS0, yS1, look))
    else:
        out.append(fr.vquad('paint_dark', (pa, tg), (pb, tg), yS0, yS1, look))
    # Innenraum (2.6 m tief): Rueckwand mit Auslagen, Boden, Seitenwaende, Leuchtdecke
    tb_ = inward(-R - 2.6)
    lk_in = ((pa + pb) / 2, inward(-R - 1.3))
    back = fr.vquad('shopint_' + cat, (pa, tb_), (pb, tb_), yb, yb + K1, lk_in)
    P = back['pos']; nrm = back['nrm'][0]; right = np.cross([0, 1.0, 0], nrm); right /= np.linalg.norm(right)
    u = P @ right; u = (u - u.min()) / (u.max() - u.min() + 1e-9); v = 1 - (P[:, 1] - yb) / K1
    back['uv'] = np.c_[u * max(1.0, round((pb - pa) / 4.0)), v]
    out.append(back)
    out.append(fr.hquad('stone_light', pa, pb, min(tg, tb_), max(tg, tb_), yb, up=True))
    out.append(fr.hquad('lightline' if not vac else 'tile_white', pa, pb, min(tg, tb_), max(tg, tb_), yb + K1, up=False))
    for s_ in (pa, pb):
        out.append(fr.vquad('tile_white', (s_, tg), (s_, tb_), yb, yb + K1, lk_in))
    if not vac:
        out += shop_props(pa, pb, dm, yb, cat, lambda d_: inward(-R - d_), sg)
    # Kollision: Glasebene
    col.append(fr.box('collision', pa, pb, min(tg, inward(-R - 0.05)), max(tg, inward(-R - 0.05)), yb, yb + K1))
    return out, col

def blank_wall(side, a, b, sc, y_top_fn):
    tw = side_t(sc, side); sg = 1 if side == 'W' else -1
    look = ((a + b) / 2, tw + sg * 3)
    return fr.vquad('concrete_panel' if sc['kind'] == 'open' else 'stone_light', (a, tw), (b, tw), yfl(a) - 0.05, y_top_fn((a + b) / 2), look, uvscale=2.5)

n_named = 0; named_list = []
for sc in secs:
    for side in ('W', 'E'):
        # Ladenband nur dort, wo keine Treppe an der Wand steht
        tw = side_t(sc, side)
        blocks = []
        for st in STAIRS:
            if abs((st['tg'][1] if side == 'E' else st['tg'][0]) - tw) < 0.4 and st['s_hi'] > sc['s0'] and st['s_lo'] < sc['s1']:
                blocks.append((max(sc['s0'], st['s_lo'] - 0.3), min(sc['s1'], st['s_hi'] + 0.3)))
        # Einheiten dieses Abschnitts
        us = [(max(u[1], sc['s0']), min(u[2], sc['s1']), u[3]) for u in UNITS if u[0] == side and u[2] > sc['s0'] + 0.5 and u[1] < sc['s1'] - 0.5]
        s = sc['s0']
        top_fn = (lambda s_: yfl(s_) + SF_H) if sc['kind'] == 'open' else (lambda s_, tw=tw: ceil_y(s_, tw))
        for a, b, sh in us:
            # Bloecke (Treppen) ausnehmen
            segs = [(a, b)]
            for b0, b1 in blocks:
                nxt = []
                for x0, x1 in segs:
                    if b1 <= x0 or b0 >= x1: nxt.append((x0, x1)); continue
                    if b0 > x0: nxt.append((x0, b0))
                    if b1 < x1: nxt.append((b1, x1))
                segs = nxt
            if a > s + 0.01: add(blank_wall(side, s, a, sc, top_fn), True)
            for x0, x1 in segs:
                if x1 - x0 >= 2.5:
                    vis, col = build_unit(side, x0, x1, sh if (x1 - x0) > 0.6 * (b - a) else None, sc)
                    add(vis); coll.extend(col)
                    if sh is not None and (x1 - x0) > 0.6 * (b - a) and sh['id'] in ATL:
                        n_named += 1; named_list.append(sh.get('name') or sh.get('name_osm'))
                else:
                    add(blank_wall(side, x0, x1, sc, top_fn), True)
            for b0, b1 in blocks:
                if b0 < b and b1 > a: add(blank_wall(side, max(a, b0), min(b, b1), sc, top_fn), True)
            s = b
        if sc['s1'] > s + 0.01: add(blank_wall(side, s, sc['s1'], sc, top_fn), True)
        # ueberdeckt: Wandband zwischen Ladenfront und Decke
        if sc['kind'] == 'cov':
            n = max(1, int(math.ceil((sc['s1'] - sc['s0']) / 2.0))); S = np.linspace(sc['s0'], sc['s1'], n + 1)
            sg = 1 if side == 'W' else -1
            for sa, sb in zip(S[:-1], S[1:]):
                hcv = sc.get('hcov', SF_H)
                if min(ceil_y(sa, tw) - yfl(sa), ceil_y(sb, tw) - yfl(sb)) - hcv < 0.03: continue
                A0 = fr.P(sa, tw, yfl(sa) + hcv); B0 = fr.P(sb, tw, yfl(sb) + hcv); B1 = fr.P(sb, tw, ceil_y(sb, tw)); A1 = fr.P(sa, tw, ceil_y(sa, tw))
                q = G2.quad('stone_light', A0, B0, B1, A1, uvscale=2.0)
                if q is not None:
                    if np.dot(q['nrm'][0][[0, 2]], fr.r * sg) < 0: q['idx'] = q['idx'][:, ::-1]; q['nrm'] *= -1
                    add(q, True)
log('shopfronts named', n_named, 'tris', G.tri_count(parts))

# Querwaende an Abschnittswechseln (unterschiedliche Breite) unterhalb der Rim-Waende: unter Bruecken/Decken
for k in range(len(secs) - 1):
    A, B = secs[k], secs[k + 1]
    s_j = A['s1']
    for side in ('W', 'E'):
        ta, tb = side_t(A, side), side_t(B, side)
        if abs(ta - tb) < 0.05: continue
        # die schmalere Seite liegt innen; Wand vom Boden bis zur Decke (ueberdeckter Teil) bzw. Ladenband-Oberkante
        cov = A if A['kind'] == 'cov' else B
        opn = B if cov is A else A
        lo, hi = sorted((ta, tb))
        look = (s_j + (-2 if cov is A else 2) * (1 if opn is B else -1), (lo + hi) / 2)
        look = ((opn['s0'] + opn['s1']) / 2, (lo + hi) / 2)
        top = lambda tt: (yfl(s_j) + CAN_Y) if False else ceil_y(s_j, tt)
        add(fr.vquad('concrete_panel', (s_j, lo), (s_j, hi), yfl(s_j) - 0.05, yfl(s_j) + 0.5, look, ya=ceil_y(s_j, lo), yb=ceil_y(s_j, hi)), True)

# ---------------------------------------------------------------- Vordaecher (offene Abschnitte), Lichtmasten, Moebel, Schilder
for sc in secs:
    if sc['kind'] != 'open': continue
    for side in ('W', 'E'):
        tw = side_t(sc, side); sg = 1 if side == 'W' else -1
        segs = [(sc['s0'] + 0.3, sc['s1'] - 0.3)]
        for st in STAIRS:
            if abs((st['tg'][1] if side == 'E' else st['tg'][0]) - tw) < 0.4:
                nxt = []
                for x0, x1 in segs:
                    b0, b1 = st['s_lo'] - 0.5, st['s_hi'] + 0.5
                    if b1 <= x0 or b0 >= x1: nxt.append((x0, x1)); continue
                    if b0 > x0: nxt.append((x0, b0))
                    if b1 < x1: nxt.append((b1, x1))
                segs = nxt
        for x0, x1 in segs:
            if x1 - x0 < 2: continue
            S = np.linspace(x0, x1, max(2, int((x1 - x0) / 2.5)) + 1)
            A = np.array([fr.P(s_, tw, yfl(s_) + CAN_Y) for s_ in S]); B = np.array([fr.P(s_, tw + sg * 1.8, yfl(s_) + CAN_Y - 0.18) for s_ in S])
            add(_strip('glass', A, B, look=fr.P((x0 + x1) / 2, tw + sg, yfl((x0 + x1) / 2) + 10)))
            add(G.tube('stainless', B, 0.03, 6))
            for s_ in S:
                add(G2.cyl_between('stainless', fr.P(s_, tw, yfl(s_) + CAN_Y + 0.5), fr.P(s_, tw + sg * 1.8, yfl(s_) + CAN_Y - 0.16), 0.02, 6))
                add(G2.cyl_between('stainless', fr.P(s_, tw, yfl(s_) + CAN_Y - 0.02), fr.P(s_, tw + sg * 1.8, yfl(s_) + CAN_Y - 0.2), 0.025, 6))
    # Lichtmasten mit Bannern (Mittellinie, nicht auf Treppen)
    tm = (sc['t0'] + sc['t1']) / 2
    for s_ in np.arange(sc['s0'] + 6.0, sc['s1'] - 3.0, 12.0):
        if stair_at(s_, tm) is not None: continue
        c = fr.xz(s_, tm); y0 = yfl(s_); ytop = street(s_, sc['t0'] - 0.5) + 6.0
        add(G.cylinder('paint_grey', (c[0], y0, c[1]), 0.10, 0.08, ytop - y0, 10, cap=True), True)
        add(G.cylinder('paint_grey', (c[0], y0, c[1]), 0.22, 0.22, 0.35, 12, cap=True), True)
        add(G.cylinder('stainless', (c[0], ytop - 2.6, c[1]), 0.14, 0.14, 1.6, 12, cap=True))
        for sg_ in (-1, 1):
            add(fr.box('paint_grey', s_ - 0.06, s_ + 0.06, tm + sg_ * 0.05, tm + sg_ * 0.9, ytop - 0.05, ytop + 0.05))
            add(fr.box('paint_grey', s_ - 0.25, s_ + 0.25, tm + sg_ * 0.8, tm + sg_ * 1.2, ytop - 0.12, ytop + 0.06))
            add(fr.hquad('lamp_glow', s_ - 0.22, s_ + 0.22, tm + sg_ * 0.82, tm + sg_ * 1.18, ytop - 0.125, up=False))
        if 'prom_niki' in ATL:
            atl, u0, v0, u1, v1 = ATL['prom_niki']
            for sg_ in (-1, 1):
                # Banner quer zur Achse, Text senkrecht lesbar -> UV gedreht
                bb = panel_box(fr, atl, 'paint_grey', s_ - 0.01, s_ + 0.01, tm + sg_ * 0.15, tm + sg_ * 0.75, y0 + 4.2, y0 + 7.4, ('s0', 's1'), top=True, bottom=True)
                for q in bb:
                    if q['mat'] == atl:
                        # Schrift von unten nach oben lesbar, Buchstabenoberkante beim Betrachter links (je Seite, nicht gespiegelt)
                        P = q['pos']; uu = (P[:, 1] - (y0 + 4.2)) / 3.2
                        rgt = np.cross([0, 1.0, 0], q['nrm'][0]); pr = P @ rgt; vv = (pr - pr.min()) / (pr.max() - pr.min() + 1e-9)
                        q['uv'] = np.c_[u0 + uu * (u1 - u0), v0 + vv * (v1 - v0)]
                add(bb)
    # Pflanzkuebel an den Enden
    for s_ in (sc['s0'] + 2.0, sc['s1'] - 2.0):
        for t_ in (sc['t0'] + 1.6, sc['t1'] - 1.6):
            if stair_at(s_, t_) is not None: continue
            c = fr.xz(s_, t_); y0 = yfl(s_)
            add(G.cylinder('concrete', (c[0], y0, c[1]), 0.55, 0.60, 0.55, 16, cap=True), True)
            add(G.cylinder('soil', (c[0], y0 + 0.55, c[1]), 0.50, 0.50, 0.01, 16, cap=True))
            for k in range(5):
                a_ = k * 1.257; rr = 0.22
                add(G.cylinder('hedge', (c[0] + rr * math.cos(a_), y0 + 0.5, c[1] + rr * math.sin(a_)), 0.25, 0.05, 0.75, 7, cap=True))
# Baenke + Wegweiser im langen ueberdeckten Abschnitt und unter den Bruecken
for sc in secs:
    if sc['kind'] != 'cov': continue
    tm = (sc['t0'] + sc['t1']) / 2
    if not sc['bridge'] and sc['name'] not in ('sued', 'nord'):
        for s_ in np.arange(sc['s0'] + 10.5, sc['s1'] - 6.0, 9.0):
            for sg_ in (-1, 1):
                y0 = yfl(s_)
                add(fr.box('wood', s_ - 0.9, s_ + 0.9, tm + sg_ * 0.75, tm + sg_ * 1.2, y0 + 0.42, y0 + 0.47), True)
                for e_ in (-0.75, 0.75): add(fr.box('paint_grey', s_ + e_ - 0.03, s_ + e_ + 0.03, tm + sg_ * 0.78, tm + sg_ * 1.17, y0, y0 + 0.42))
    # Haengeschilder (Wegweiser) mittig
    s_ = (sc['s0'] + sc['s1']) / 2; yC = ceil_y(s_, tm)
    for key, ds in (('prom_kroepcke', -0.03), ('prom_hbf', 0.03)):
        if key not in ATL: continue
        atl, u0, v0, u1, v1 = ATL[key]
        bb = panel_box(fr, atl, 'paint_dark', s_ + ds - 0.025, s_ + ds + 0.025, tm - 1.5, tm + 1.5, yC - 0.75, yC - 0.38, 's0' if ds < 0 else 's1')
        for q in bb:
            if q['mat'] == atl: q['uv'] = np.c_[u0 + q['uv'][:, 0] * (u1 - u0), v0 + q['uv'][:, 1] * (v1 - v0)]
        add(bb)
    for t_ in (tm - 1.3, tm + 1.3):
        add(G2.cyl_between('stainless', fr.P(s_, t_, yC - 0.38), fr.P(s_, t_, yC), 0.01, 5))

# ---------------------------------------------------------------- Bruestungen + Gelaender an den Oeffnungen (Strassenniveau)
PAR_H = 0.85
for i, o in enumerate(ops):
    s0, s1 = o['s']; t0, t1 = o['t']
    sides = [('long', (s0, s1), t0, -1), ('long', (s0, s1), t1, +1)]
    for nm, (a, b), tt, sg in sides:
        # Betonfertigteile auf der Einfassung (aussen), 2.5-m-Elemente, Stahlrohrgelaender (3 Rohre)
        cuts = np.linspace(a - 0.35, b + 0.35, max(2, int(math.ceil((b - a + 0.7) / 2.5))) + 1)
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            g0 = street(x0, tt + sg * 0.15); g1 = street(x1, tt + sg * 0.15)
            add(fr.box('concrete_panel', x0 + 0.01, x1 - 0.01, min(tt, tt + sg * 0.3), max(tt, tt + sg * 0.3), min(g0, g1) - 0.06, max(g0, g1) + PAR_H, uvscale=2.5), True)
        S_ = np.linspace(a - 0.3, b + 0.3, max(2, int((b - a) / 1.25)) + 1)
        P3 = np.array([fr.P(s_, tt + sg * 0.15, street(s_, tt + sg * 0.15) + PAR_H) for s_ in S_])
        for q in P3: add(G2.cyl_between('steel', q, q + [0, 0.27, 0], 0.022, 6))
        for h_ in (0.10, 0.27): add(G.tube('steel', P3 + [0, h_, 0], 0.024, 6))
    for s_e, sg in ((s0, -1), (s1, +1)):
        # Stirnseiten (Bruecken): Glasbruestung mit Edelstahlhandlauf auf Granitsockel; Luecken an Treppenkoepfen
        cuts = [(t0 - 0.35, t1 + 0.35)]
        for st in STAIRS:
            if abs(st['s_top'] - s_e) < 0.1:
                nxt = []
                for x0, x1 in cuts:
                    b0, b1 = st['tg']
                    if b1 <= x0 or b0 >= x1: nxt.append((x0, x1)); continue
                    if b0 > x0: nxt.append((x0, b0 - 0.05))
                    if b1 < x1: nxt.append((b1 + 0.05, x1))
                cuts = nxt
        for x0, x1 in cuts:
            if x1 - x0 < 0.3: continue
            ss = s_e + sg * 0.15
            T_ = np.linspace(x0, x1, max(2, int((x1 - x0) / 1.4)) + 1)
            gy = [street(ss, t_) for t_ in T_]
            add(fr.box('granite_polished', s_e, s_e + sg * 0.3, x0, x1, min(gy) - 0.06, max(gy) + 0.12) if sg > 0 else fr.box('granite_polished', s_e - 0.3, s_e, x0, x1, min(gy) - 0.06, max(gy) + 0.12), True)
            A = np.array([fr.P(ss, t_, max(gy) + 0.12) for t_ in T_]); B = np.array([fr.P(ss, t_, max(gy) + 1.05) for t_ in T_])
            add(_strip('glass', A, B, look=fr.P(ss + sg * 3, (x0 + x1) / 2, max(gy) + 0.5)))
            add(G.tube('stainless', B + [0, 0.05, 0], 0.028, 8))
            for q in A: add(G2.cyl_between('stainless', q, q + [0, 0.95, 0], 0.02, 6))
            coll.append(_strip('collision', A, B, look=fr.P(ss + sg * 3, (x0 + x1) / 2, max(gy) + 0.5)))
    # Leuchten auf den Bruecken (Mastleuchten an der Stirnseite)
log('done parapets, tris', G.tri_count(parts))

pickle.dump({'parts': parts, 'coll': coll, 'walk': walk, 'mats': MATS_EXTRA,
             'info_extra': {'sections': [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in s_.items()} for s_ in secs],
                            'stairs': [{'name': st['name'], 'kind': st['kind'], 's': [round(st['s_lo'], 2), round(st['s_hi'], 2)], 't': [round(x, 2) for x in st['tg']]} for st in STAIRS],
                            'shops_named': named_list, 'axis': AX}}, open(CACHE + 'promenade.pkl', 'wb'))
log('done: tris', G.tri_count(parts), 'collision', G.tri_count(coll), 'named shops', n_named)

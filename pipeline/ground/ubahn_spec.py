"""U-Bahn-Zugaenge: Spezifikation (Lage, Richtung, Fahrspuren) aus OSM + Street-View-Notizen.
Gemeinsam benutzt von layout.py (Loecher im Gelaende) und ubahn.py (Bauwerke).

Lokales System je Zugang: s = Abstand entlang 'dir' (hinunter), t = quer (positiv = rechts beim Blick hinunter),
right = (-dz, dx) in (x, z).  Spuren (lanes) von links nach rechts.

Quellen der Kroepcke-Zugaenge (OSM-Wege, Stand 2026-11):
  A  Kr.73A(2)/Kr.74A(2) Rolltreppen (272783349/272783346) + Treppe (Street View: 'Treppe links + Rolltreppe rechts', Breite ~4.5 m)
  B  Kr.71A/Kr.72A Rolltreppen (265755776/31557181)
  C  Kr.69A/Kr.70A Rolltreppen + Treppe dazwischen (265755771/31557183/265755777)
  D  Kr.69B/Kr.68B Rolltreppen + Treppe (271493216/265755774/265755770)
  E  Treppe 32773750;  F  Treppe 331983409 + Rolltreppe Kr.76B (271493212)
  G  Treppe 126562335;  H  Kr.67A/Kr.68A Rolltreppen (126562333/126562338);  I  Treppe 331983711
  Aufzug Kr.80A (3392490912): freistehender Glasaufzug (Street View)."""
import math, json
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from shapely import affinity
from common import *

ESC_W = 1.45            # Rolltreppe Aussenbreite (Stufenbreite 1.0 m + Balustraden)
STAIR_RISE = 0.16; STAIR_TREAD = 0.30; LANDING = 1.5; TOP_LANDING = 0.6
ESC_FLAT = 1.3          # flacher Teil oben/unten (Kammplatte + Flachstufen)
TAN30 = math.tan(math.radians(30))
DIVIDER = 0.12

def _way(osm, wid):
    for w in osm['ways']:
        if w['id'] == wid: return np.array(w['pts'], float)
    return None

def frame_from_ways(osm, ways_up):
    """ways_up: Liste OSM-Wege, deren letzter Punkt oben liegt (incline=up von unten nach oben). -> top (Mitte), dir (hinunter)"""
    tops = []; bots = []
    for wid, rev in ways_up:
        P = _way(osm, wid)
        if P is None: continue
        a, b = (P[-1], P[0]) if not rev else (P[0], P[-1])
        tops.append(a); bots.append(b)
    tops = np.array(tops); bots = np.array(bots)
    d = (bots - tops).mean(0); d /= np.linalg.norm(d)
    return tops.mean(0), d

def lanes_width(lanes):
    return sum(l['w'] for l in lanes) + DIVIDER * (len(lanes) - 1)

def kroepcke_specs(osm):
    S = []
    def add(ref, ways, lanes, depth=4.8, sign='right', note='', shift_t=0.0, shift_s=0.0, extra=None):
        top, d = frame_from_ways(osm, ways)
        r = np.array([-d[1], d[0]])
        top = top + r * shift_t + d * shift_s
        sp = {'key': 'kroepcke_' + ref, 'ref': ref, 'station': 'Kröpcke', 'sign_mat': 'sign_kroepcke', 'style': 'kroepcke',
              'top': top.tolist(), 'dir': d.tolist(), 'lanes': lanes, 'depth': depth, 'sign': sign, 'note': note,
              'osm_ways': [w for w, _ in ways]}
        if extra: sp.update(extra)
        S.append(sp)
    E = lambda run: {'type': 'esc', 'w': ESC_W, 'run': run}
    St = lambda w: {'type': 'stair', 'w': w}
    # OSM-Wege mit incline=up laufen von unten nach oben -> letzter Punkt oben (rev=False)
    add('A', [(272783349, False), (272783346, False)], [St(1.6), E('up'), E('down')], note='Street View: Trog parallel Karmarschstrasse, Granitbruestung ~1.0 m, Edelstahlgelaender; OSM: 2 Rolltreppen; Lage nach Detailmodell-Trog', shift_t=0.52, shift_s=0.8)
    add('B', [(265755776, False), (31557181, False)], [E('up'), E('down')], note='OSM: Rolltreppen Kr.71A (auf) / Kr.72A (ab); SV: Trog ~3 m, Granitbruestung', extra={'hall': 5.0})
    add('C', [(265755771, False), (265755777, False), (31557183, False)], [E('up'), St(1.8), E('down')], note='OSM: Kr.69A auf, Treppe, Kr.70A ab', extra={'hall': 6.0})
    add('D', [(271493216, False), (265755770, False), (265755774, False)], [E('up'), St(3.0), E('down')], sign='left', note='OSM: Kr.69B auf, Treppe, Kr.68B ab; SV: U-Schild an der SO-Ecke; Breite/Lage nach Detailmodell-Trog', shift_t=0.6, shift_s=0.2)
    add('E', [(32773750, False)], [St(3.6)], depth=4.9, note='OSM Treppe 32773750 (level 0;-1); Breite/Lage nach Detailmodell-Trog', shift_t=0.28)
    add('F', [(331983409, False), (271493212, False)], [E('both'), St(2.4)], depth=4.9, sign='right', note='OSM Treppe 331983409 + Rolltreppe Kr.76B (reversibel); Lage nach Detailmodell-Trog', shift_t=1.27, shift_s=0.2)
    add('G', [(126562335, False)], [St(2.6)], depth=4.4, note='OSM Treppe 126562335; vor Ort nicht gesehen (LKW)')
    add('H', [(126562333, False), (126562338, False)], [E('up'), E('down')], depth=4.4, note='OSM Kr.67A auf / Kr.68A ab; vor Ort nicht gesehen')
    add('I', [(331983711, False)], [St(3.2)], depth=4.6, sign='left', note='SV: Trog parallel Karmarschstrasse, Breite ~3.5 m, Doppel-U-Wuerfel')
    return S

STATION_SIGN = {'Steintor': 'sign_steintor', 'Markthalle': 'sign_markthalle', 'Markthalle/Landtag': 'sign_markthalle',
                'Hauptbahnhof': 'sign_hbf', 'Aegidientorplatz': 'sign_aegi', 'Waterlooplatz': 'sign_waterloo'}

def generic_specs(osm, bld, taken):
    """alle weiteren railway=subway_entrance im Gebiet (ausser Kroepcke): Treppe 2.6 m, Richtung aus OSM-Treppe/Fussweg"""
    S = []
    ways = [w for w in osm['ways'] if w['tags'].get('highway') in ('steps', 'footway', 'corridor', 'pedestrian') and len(w['pts']) >= 2]
    for n in osm['nodes']:
        t = n['tags']
        if t.get('railway') != 'subway_entrance' or t.get('highway') == 'elevator' or t.get('indoor') == 'yes': continue
        p = np.array(n['p'], float)
        if np.hypot(*p) > RADIUS - 25: continue
        name = t.get('name') or ''
        if 'Kröpcke' in name or 'Kr' in name[:2] and 'pcke' in name: continue
        if -120 < p[0] < 170 and -40 < p[1] < 240: continue          # Kroepcke-Bereich: eigene Spezifikation
        if bld.buffer(0.3).contains(Point(p)): continue
        # Richtung: angeschlossene Treppe (bevorzugt), sonst Tunnel-/Ebene--1-Weg
        d = None
        for pref in ('steps', 'under'):
            for w in ways:
                P = np.array(w['pts']); dd = np.hypot(*(P - p).T); j = int(np.argmin(dd))
                if dd[j] > 0.6: continue
                wt = w['tags']
                if pref == 'steps' and wt.get('highway') != 'steps': continue
                if pref == 'under' and not (wt.get('tunnel') == 'yes' or str(wt.get('level', '')).startswith('-') or wt.get('incline') == 'down'): continue
                other = P[min(j + 1, len(P) - 1)] if j == 0 else P[max(j - 1, 0)]
                v = other - p
                if np.linalg.norm(v) < 1: continue
                d = v / np.linalg.norm(v); break
            if d is not None: break
        if d is None: continue
        if 'Steintor' in name or 'Steintor' in (t.get('description') or ''): st = 'Steintor'
        elif 'Markthalle' in name: st = 'Markthalle'
        elif 'Aegidientor' in name or 'Ausgang' in name and np.hypot(*(p - [310, 600])) < 150: st = 'Aegidientorplatz'
        elif 'Waterloo' in name: st = 'Waterlooplatz'
        elif np.hypot(*(p - [220, -330])) < 250: st = 'Hauptbahnhof'
        else: st = name or 'U-Bahn'
        S.append({'key': 'ub_%d' % n['id'], 'ref': t.get('ref', ''), 'station': st, 'sign_mat': STATION_SIGN.get(st, 'sign_kroepcke'),
                  'style': 'steintor' if st == 'Steintor' else 'generic', 'top': p.tolist(), 'dir': d.tolist(),
                  'lanes': [{'type': 'stair', 'w': 2.6}], 'depth': 4.6, 'sign': 'right', 'note': 'OSM node %d (%s)' % (n['id'], t.get('description') or name),
                  'osm_node': n['id']})
    return S

# ---------------------------------------------------------------- Geometrie-Ableitungen
def lane_geom(sp):
    """berechnet je Spur Profil-Kennwerte und die Oeffnungslaenge"""
    D = sp['depth']; out = []
    W = lanes_width(sp['lanes']); t = -W / 2
    end = 0.0
    for l in sp['lanes']:
        g = dict(l); g['t0'] = t; g['t1'] = t + l['w']; t = g['t1'] + DIVIDER
        if l['type'] == 'stair':
            n = int(math.ceil(D / STAIR_RISE - 1e-6)); rise = D / n
            land = n > 18
            n1 = n // 2 if land else n
            g.update(n=n, rise=rise, tread=STAIR_TREAD, n1=n1, landing=LANDING if land else 0.0)
            # Riser k (1..n) bei s_k; Tritt k von s_k bis s_{k+1}
            s = [TOP_LANDING + (k - 1) * STAIR_TREAD + (LANDING - STAIR_TREAD if (land and k > n1) else 0.0) for k in range(1, n + 1)]
            g['s_riser'] = s
            g['s_end'] = s[-1]                    # letzter Riser -> danach Boden
        else:
            L_inc = D / TAN30
            g.update(s_inc0=ESC_FLAT, s_inc1=ESC_FLAT + L_inc, s_end=ESC_FLAT + L_inc + ESC_FLAT)
        end = max(end, g['s_end'] if l['type'] == 'stair' else g['s_inc1'])
        out.append(g)
    return out, W, end + 0.4

def local(sp):
    o = np.array(sp['top'], float); d = np.array(sp['dir'], float); r = np.array([-d[1], d[0]])
    def W_(s, t): return o + d * s + r * t
    return o, d, r, W_

def derive(sp):
    """Oeffnung, Bruestungsring, Tunnel + Stummel-Halle (xz-Polygone) ins Spec schreiben"""
    lanes, W, Lo = lane_geom(sp)
    o, d, r, W_ = local(sp)
    sp['lane_geom'] = lanes; sp['W'] = W; sp['L_open'] = Lo
    rect = lambda s0, s1, t0, t1: Polygon([W_(s0, t0), W_(s1, t0), W_(s1, t1), W_(s0, t1)])
    sp['poly_open'] = rect(0, Lo, -W / 2, W / 2)
    PAR = 0.30
    sp['poly_parapet'] = rect(0, Lo + PAR, -W / 2 - PAR, W / 2 + PAR).difference(sp['poly_open'])
    sp['poly_zone'] = rect(-1.2, Lo + PAR + 0.6, -W / 2 - PAR - 0.8, W / 2 + PAR + 0.8)    # Arbeitsbereich (Pflaster neu)
    Lt = sp.get('tunnel', 3.5); Lh = sp.get('hall', 9.0); Wh = max(W + 4.0, 8.0)
    sp['tunnel_len'] = Lt; sp['hall_len'] = Lh; sp['hall_w'] = Wh
    sp['poly_tunnel'] = rect(Lo, Lo + Lt, -W / 2, W / 2)
    sp['poly_hall'] = rect(Lo + Lt, Lo + Lt + Lh, -Wh / 2, Wh / 2)
    return sp

def all_specs(osm=None, bld=None):
    osm = osm or load_osm()
    if bld is None:
        B = json.load(open(DATA + 'lod2_buildings.json'))['buildings']
        bl = []
        for b in B:
            for g in b['grounds'] or []:
                P = np.array(g['ext'])[:, :2] * [1, -1]
                if len(P) >= 3:
                    p = Polygon(P).buffer(0)
                    if p.area > 1: bl.append(p)
        bld = unary_union(bl)
    specs = [derive(s) for s in kroepcke_specs(osm)]
    try:
        PR = json.load(open(CACHE + 'promenade.json'))
        prom = unary_union([Polygon(o['rect']) for o in PR['openings']])
    except Exception:
        prom = Polygon()
    taken = unary_union([s['poly_open'].buffer(1.0) for s in specs])
    gen = []
    for s in generic_specs(osm, bld, taken):
        derive(s)
        if s['poly_open'].buffer(0.3).intersects(bld) or s['poly_parapet'].intersects(bld):
            # kuerzer versuchen (flacher Abgang, z. B. Zwischenebene)
            for dep in (3.6, 3.0):
                s['depth'] = dep; derive(s)
                if not (s['poly_open'].buffer(0.3).intersects(bld) or s['poly_parapet'].intersects(bld)): break
            else:
                s['skip'] = 'building'; gen.append(s); continue
        if s['poly_open'].buffer(1.0).intersects(taken):
            s['skip'] = 'overlap'; gen.append(s); continue
        if s['poly_open'].buffer(2.0).intersects(prom):
            s['skip'] = 'passerelle'; gen.append(s); continue
        taken = taken.union(s['poly_open'].buffer(1.0)); gen.append(s)
    for s in specs + gen:
        s['in_square'] = bool(s['poly_zone'].intersects(box(*SQ)))
    return specs + gen

if __name__ == '__main__':
    S = all_specs()
    for s in S:
        print('%-18s %-12s %s top=%s dir=%s W=%.2f L=%.2f depth=%.1f sq=%s %s' % (s['key'], s['station'], ''.join(l['type'][0] for l in s['lanes']),
              np.round(s['top'], 1).tolist(), np.round(s['dir'], 3).tolist(), s['W'], s['L_open'], s['depth'], s['in_square'], s.get('skip', '')))

"""Echte Ladenfronten aus data/shops.json (OSM + Markenrecherche) und data/streetview_notes.json:
Zuordnung zur LoD2-Wand, Aufteilung der Erdgeschossfront je Laden (Schaufenster-Achsen + Tuer), Ladenraum hinter dem Glas
(geschlossener Innenraum mit Boden/Decke/Rueckwand, 1.2-5 m tief), Auslagen je Art (instanziert, displays.py),
Werbeanlagen mit dem echten Namen als Geometrie (Kanalbuchstaben, Leuchtkasten, Blendenschild, Ausleger, Folienschrift),
Hausnummern, Aussengastronomie (OSM outdoor_seating=yes).

Keine Logos werden nachgebildet: nur der Name in einer freien Schrift (DejaVu) in den ungefaehren Markenfarben."""
import os, re, math, collections
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box as sbox
from common import DATA, load_json, rng_of, jitter_hex
from gltf_io import glow_mat
import text3d as TX

_DB = None
# Korrekturen an shops.json (Quelle: QA / Websuche), ohne die Datendatei des Recherche-Schritts zu aendern
SHOP_FIX = {
    # OSM-Tippfehler 'Cark Kuester' (Copyshop Kramerstr. 15): in Hannover existiert die Druckerei 'Carl Kuester' (seit 1886);
    # Name vor Ort pruefen (BEGEHUNGSLISTE)
    'osm-n8939589067': {'name': 'Carl Küster', 'sign': {'text': 'Carl Küster'}},
    # Hugendubel: Name als weisse Einzelbuchstaben vor dem 1. OG (style.OVERRIDES big_signs), am EG rote Tafel 'Welt der Buecher'
    'osm-n261744178': {'sign': {'text': 'Welt der Bücher', 'type': 'fascia_panel', 'bg': '#C8102E', 'fg': '#FFFFFF'}},
    # GALERIA: grosse gruene Versalien an der Metallfassade (big_signs); am EG nur der Name auf schwarzer Blende
    'osm-n261744174': {'sign': {'text': 'GALERIA', 'type': 'fascia_panel', 'bg': '#1E1E1E', 'fg': '#6CB33F'}},
}
WOOD_FRONT = ['#2A4A8A', '#4A3226', '#2E4A3A', '#6B2A26', '#22201F', '#3A4A5A']
ALU_FRONT = ['#2F3337', '#3A3D40', '#22201F', '#5B6066', '#8C969C', '#A6AEAF']


def db():
    """shops.json -> {lod2_id: [shop, ...]} (nur Strassenlaeden im EG) + OSM-Tags (Aussengastronomie)."""
    global _DB
    if _DB is not None:
        return _DB
    S = load_json(os.path.join(DATA, 'shops.json'))
    osm = load_json(os.path.join(DATA, 'osm_features.json'))
    seat = set()
    for n in osm['nodes']:
        if n['tags'].get('outdoor_seating') == 'yes':
            seat.add('node/%d' % n['id'])
    for w in osm['ways']:
        if w['tags'].get('outdoor_seating') == 'yes':
            seat.add('way/%d' % w['id'])
    by = collections.defaultdict(list)
    n = 0
    for s in S['shops']:
        if s.get('level') != 0 or s.get('context') != 'street' or not s.get('facade') or not s['facade'].get('wall_point'):
            continue
        bid = s['facade'].get('building_id') or s.get('building_lod2_id')
        if not bid:
            continue
        s = dict(s); s['outdoor'] = s.get('osm_id') in seat
        fx = SHOP_FIX.get(s['id'])
        if fx:
            s.update({k: v for k, v in fx.items() if k != 'sign'})
            if 'sign' in fx:
                s['sign'] = dict(s.get('sign') or {}, **fx['sign'])
        by[bid].append(s); n += 1
    _DB = {'by_bldg': by, 'n': n, 'used': set()}
    return _DB


# ------------------------------------------------------------------ Art der Auslage
OTHER_MAP = {
    'hairdresser': 'salon', 'beauty': 'salon', 'tattoo': 'salon', 'massage': 'salon', 'nutrition_coach': 'salon', 'wigs': 'salon',
    'hotel': 'lobby', 'hostel': 'lobby', 'cinema': 'lobby', 'fitness_centre': 'lobby', 'dance': 'lobby', 'escape_game': 'lobby',
    'bowling_alley': 'lobby', 'coworking_space': 'office',
    'travel_agency': 'office', 'travel_agent': 'office', 'insurance': 'office', 'estate_agent': 'office', 'car_rental': 'office',
    'information': 'office', 'post_office': 'office', 'bookmaker': 'office', 'ticket': 'office', 'charity': 'generic', 'laundry': 'office',
    'houseware': 'home', 'furniture': 'home', 'interior_decoration': 'home', 'bed': 'home', 'lighting': 'home', 'carpet': 'home',
    'antiques': 'home', 'gift': 'home', 'bathroom_furnishing': 'home', 'frame': 'home', 'cutlery': 'home', 'variety_store': 'generic',
    'florist': 'florist', 'garden_centre': 'florist', 'confectionery': 'bakery', 'tea': 'bakery', 'coffee': 'bakery', 'deli': 'bakery',
    'chocolate': 'bakery', 'clothes': 'fashion', 'boutique': 'fashion', 'shoes': 'shoes', 'jewelry': 'jewellery', 'sports': 'sports',
    'restaurant': 'cafe', 'cafe': 'cafe', 'fast_food': 'fastfood', 'ice_cream': 'cafe', 'pub': 'cafe', 'bar': 'cafe', 'bank': 'bank',
    'department_store': 'department', 'vacant': 'vacant', 'electronics': 'phone', 'musical_instrument': 'home', 'hunting': 'sports',
}
KIND_MAP = {'fashion': 'fashion', 'shoes': 'shoes', 'jewellery': 'jewellery', 'optician': 'optician', 'pharmacy': 'pharmacy',
            'bakery': 'bakery', 'cafe': 'cafe', 'restaurant': 'cafe', 'fast_food': 'fastfood', 'bank': 'bank', 'phone': 'phone',
            'electronics': 'phone', 'books': 'books', 'drugstore': 'drugstore', 'cosmetics': 'drugstore', 'department_store': 'department',
            'kiosk': 'kiosk', 'supermarket': 'supermarket', 'sports': 'sports', 'toys': 'toys'}


def display_kind(s):
    if s is None:
        return 'generic'
    if s.get('status') in ('closed', 'vacant', 'likely_closed') or s.get('category') == 'vacant':
        return 'vacant'
    k = s['display'].get('kind')
    if k in KIND_MAP:
        return KIND_MAP[k]
    return OTHER_MAP.get(s.get('category'), 'generic')


ROOM_STYLE = {   # Wand, Boden, Decke, Licht
    'fashion': ('#F2F0EC', '#B8A890', '#F4F2EE', 'dsp_ceil'), 'sports': ('#E8E8E6', '#6E6A66', '#F2F2EE', 'dsp_ceil'),
    'shoes': ('#EEE8E0', '#9A7A5A', '#F2F0EC', 'dsp_ceil'), 'jewellery': ('#6A5648', '#4A3A32', '#5E4E44', 'dsp_ceil_warm'),
    'optician': ('#F2F0EC', '#C8C0B4', '#F4F2EE', 'dsp_ceil'), 'pharmacy': ('#F6F6F4', '#D8D8D4', '#F6F6F4', 'dsp_ceil'),
    'bakery': ('#E0C8A8', '#8A6A4A', '#EDE3CF', 'dsp_ceil_warm'), 'cafe': ('#D8C4A8', '#6B4E3A', '#E8DCC8', 'dsp_ceil_warm'),
    'fastfood': ('#E8E2DA', '#8C8A86', '#F2F2EE', 'dsp_ceil'), 'bank': ('#E8E6E2', '#9A9690', '#F2F2EE', 'dsp_ceil'),
    'phone': ('#F4F4F2', '#D8D4CC', '#F6F6F4', 'dsp_ceil'), 'books': ('#E8DCC8', '#7A5A3A', '#F2EDE2', 'dsp_ceil_warm'),
    'drugstore': ('#F6F6F4', '#D8D8D4', '#F6F6F4', 'dsp_ceil'), 'department': ('#E8E2DA', '#C8C0B4', '#F2F0EC', 'dsp_ceil'),
    'kiosk': ('#E8E2DA', '#8C8A86', '#F2F2EE', 'dsp_ceil'), 'salon': ('#ECE6DE', '#3A3836', '#F4F2EE', 'dsp_ceil_warm'),
    'lobby': ('#E0D4C0', '#5A4A3A', '#F2EDE2', 'dsp_ceil_warm'), 'office': ('#ECEAE6', '#8C8A86', '#F2F2EE', 'dsp_ceil'),
    'home': ('#EAE2D6', '#A88A6A', '#F2EDE2', 'dsp_ceil_warm'), 'florist': ('#E2E0D4', '#7A7A72', '#F2F2EE', 'dsp_ceil_warm'),
    'vacant': ('#B8B4AC', '#6E6A66', '#C8C4BC', None), 'generic': ('#ECEAE6', '#A8A49C', '#F2F2EE', 'dsp_ceil'),
    'supermarket': ('#ECEAE6', '#B8B4AC', '#F2F2EE', 'dsp_ceil'), 'toys': ('#F2EEE6', '#A8A49C', '#F6F4EE', 'dsp_ceil'),
}
RISER = {'jewellery': 0.62, 'fashion': 0.22, 'department': 0.42, 'cafe': 0.45, 'bakery': 0.5, 'books': 0.38, 'shoes': 0.32,
         'vacant': 0.35, 'bank': 0.3, 'fastfood': 0.2, 'supermarket': 0.2, 'phone': 0.25}
GENERIC_KINDS = ['fashion', 'generic', 'shoes', 'cafe', 'home', 'office', 'drugstore', 'vacant', 'jewellery', 'books', 'optician']


# ------------------------------------------------------------------ Zuordnung zu Waenden
def assign(b, walls):
    """Laeden des Gebaeudes auf die passende Wand legen -> w.units = [{'shop', 'u', 'ue', 'W'}]."""
    for w in walls:
        w.units = []
    D = db()
    shops = D['by_bldg'].get(b.id, [])
    if not shops:
        return 0
    cand = [w for w in walls if not w.drop and w.vertical and w.L >= 2.0 and w.vmin < b.y_ground + 1.0]
    n = 0
    for s in shops:
        f = s['facade']
        wp = np.array(f['wall_point'], float); nn = np.array(f.get('normal') or [0, 0], float)
        best = None
        for w in cand:
            a = w.fr.w([w.umin, 0, 0]); c = w.fr.w([w.umax, 0, 0])
            seg = LineString([(a[0], a[2]), (c[0], c[2])])
            dist = seg.distance(Point(wp))
            al = float(np.dot(nn, w.n[[0, 2]])) if np.linalg.norm(nn) > 0.5 else 0.5
            if al < 0.3:
                continue
            # kurze Wandstuecke (gestueckelte LoD2-Front) abwerten: der Laden gehoert auf das laengere Frontsegment daneben
            sc = dist + (1 - al) * 4.0 + (3.0 if w.party_wall else 0.0) + (2.0 if w.L < 3.2 else 0.0)
            if best is None or sc < best[0]:
                best = (sc, w, dist)
        if best is None or best[2] > 4.0:
            # Ersatz: naechste freie Aussenwand desselben Gebaeudes (bis 15 m), Strassenwaende bevorzugt
            best = None
            for w in cand:
                if w.party_wall or w.L < 3.0:
                    continue
                a = w.fr.w([w.umin, 0, 0]); c = w.fr.w([w.umax, 0, 0])
                dist = LineString([(a[0], a[2]), (c[0], c[2])]).distance(Point(wp))
                sc = dist + (0.0 if w.street_dist < 26 else 6.0)
                if dist < 15.0 and (best is None or sc < best[0]):
                    best = (sc, w, dist)
            if best is None:
                continue
        w = best[1]
        u = float(np.clip(w.fr.loc([wp[0], b.y_ground, wp[1]])[0], w.umin + 1.0, w.umax - 1.0))
        ue = None
        en = f.get('entrance_node')
        if en and en.get('dist', 9) < 3.5:
            ue = float(w.fr.loc([en['p'][0], b.y_ground, en['p'][1]])[0])
        W = float(np.clip(s.get('frontage_width_m') or 6.0, 2.4, 40.0))
        w.units.append({'shop': s, 'u': u, 'ue': ue, 'W': W, 'kind': display_kind(s)})
        n += 1
    for w in walls:
        if w.units:
            _allocate(w)
    return n


def _allocate(w, margin=0.35):
    U = sorted(w.units, key=lambda x: x['u'])
    lo, hi = w.umin + margin, w.umax - margin
    for x in U:
        x['s0'] = max(lo, x['u'] - x['W'] / 2); x['s1'] = min(hi, x['u'] + x['W'] / 2)
        if x['s1'] - x['s0'] < 2.2:                         # an der Wandkante: Breite nach innen verschieben
            if x['s0'] <= lo + 1e-6:
                x['s1'] = min(hi, lo + max(2.2, x['W']))
            elif x['s1'] >= hi - 1e-6:
                x['s0'] = max(lo, hi - max(2.2, x['W']))
    for a, c in zip(U, U[1:]):
        if a['s1'] > c['s0'] - 0.05:
            m = (a['u'] + c['u']) / 2 if abs(a['u'] - c['u']) > 0.5 else (a['s0'] + c['s1']) / 2
            m = float(np.clip(m, a['s0'] + 1.0, c['s1'] - 1.0)) if c['s1'] - a['s0'] > 2.2 else m
            a['s1'] = m - 0.05; c['s0'] = m + 0.05
    w.units = [x for x in U if x['s1'] - x['s0'] >= 1.6]


# ------------------------------------------------------------------ Erdgeschoss-Layout
def layout_retail(w, b, P, G, ctx, rng, flags):
    import facade as FA
    kind = P['kind']; vol = ctx['vol']
    fy1 = FA.floor_y(b, G, 1) if G['n'] > 1 else b.y_ground + G['g']
    head = 0.95 if kind != 'curtain' else 0.55
    ylw = getattr(w, 'yl', None)
    if ylw is not None and ylw > fy1 - 2.7 and G['n'] > 1:
        # Wand an hoeher liegender Strasse (Gebaeude ueber zwei Ebenen, Hang): Ladenzone beginnt auf dem oertlichen Gehweg und
        # reicht bis zur naechsten Geschossdecke mit genug Hoehe; Obergeschossfenster darunter entfallen auf dieser Wand
        k = 1
        while k < G['n'] - 1 and FA.floor_y(b, G, k) - ylw < 3.3:
            k += 1
        fk = FA.floor_y(b, G, k)
        if fk - ylw >= 2.9 and fk < w.vcorner - 0.5:
            fy1 = fk
            w.openings = [o for o in w.openings if o.zone == 'gf' or o.v0 > fy1 - 0.3]
            w.gf_top = fy1

    def top_for(yb):
        v1 = min(fy1 - head, yb + 4.3)
        if v1 - yb < 2.3:
            v1 = min(fy1 - 0.66, yb + 2.3)            # niedriges EG: Platz fuer ein Schild (>= 0.5 m) lassen
        if v1 - yb < 2.05:
            v1 = min(fy1 - 0.3, yb + 2.1)
        return v1
    if fy1 - 0.3 - max(b.y_ground, ylw if ylw is not None else b.y_ground) < 2.1:
        return
    wood = kind in ('fachwerk',) or (kind == 'altbau' and b.street in ALT_WOOD_STREETS)
    if P.get('wood_front') is not None:
        wood = bool(P['wood_front'])
    units = list(getattr(w, 'units', []))
    # Luecken ohne bekannte Laeden: neutrale Einheiten (ohne erfundenen Namen)
    if w.retail:
        free = []
        cur = w.umin + 0.4
        for x in sorted(units, key=lambda x: x['s0']):
            if x['s0'] - cur > 3.2:
                free.append((cur, x['s0'] - 0.3))
            cur = max(cur, x['s1'] + 0.3)
        if w.umax - 0.4 - cur > 3.2:
            free.append((cur, w.umax - 0.4))
        for a, c in free:
            n = max(1, int(round((c - a) / 7.5)))
            for i in range(n):
                s0 = a + (c - a) * i / n + (0.15 if i else 0); s1 = a + (c - a) * (i + 1) / n - (0.15 if i < n - 1 else 0)
                gk = GENERIC_KINDS[int(rng_of('gen', b.id, w.idx, i, round(a, 1)).integers(len(GENERIC_KINDS)))]
                if P.get('free_kind'):
                    gk = P['free_kind']                     # z.B. leerstehendes Kaufhaus: zugeklebte/leere Schaufenster
                elif kind == 'kaufhaus':
                    gk = 'department'
                units.append({'shop': None, 'u': (s0 + s1) / 2, 'ue': None, 'W': s1 - s0, 's0': s0, 's1': s1, 'kind': gk})
    units.sort(key=lambda x: x['s0'])
    # Hauseingang in einer Luecke (zu den Obergeschossen)
    if flags.get('house_door_done') is None and G['n'] > 1:
        gaps = []
        cur = w.umin + 0.5
        for x in units:
            gaps.append((cur, x['s0'] - 0.2)); cur = x['s1'] + 0.2
        gaps.append((cur, w.umax - 0.5))
        gaps = [(a, c) for a, c in gaps if c - a >= 1.6]
        if gaps:
            a, c = max(gaps, key=lambda g: g[1] - g[0])
            uc = (a + c) / 2; dw = 1.3
            yd = _unit_base(w, b, ctx, uc - 0.8, uc + 0.8)
            dh = min(2.55, fy1 - 0.4 - yd)
            if dh > 2.05 and FA.rect_fits(w, uc - dw / 2, uc + dw / 2, yd, yd + dh, 0.2, 0.3, 0.0) and \
                    FA.open_at(w, b, vol, uc, yd + 1.5, deltas=(0.45, 1.0))[0] and not FA.is_party(w, uc):
                op = FA.Opening(uc - dw / 2, uc + dw / 2, yd, yd + dh, 0.2, ri=0.12, zone='gf', kind='door')
                dk = 'nk' if kind in ('nachkrieg', 'fachwerk', 'altbau') else 'glass'
                dp = __import__('protos').door_proto(dk, 0.2)
                if dk == 'glass':
                    op.add('glass', 'frame', uc, yd, dw, dh, '#C4CED4', unit=True)
                op.add('door_%s_20' % dk, 'frame', uc, yd, dw, dh, ['#5A3B28', '#3A3D40', '#294738', '#2F3337'][int(rng.integers(4))])
                w.openings.append(op)
                flags['house_door_done'] = True
                flags.setdefault('numbers', []).append((w, uc + dw / 2 + 0.3, yd + min(2.25, dh - 0.2), b.number))
    pier = 0.42 if not wood else 0.3
    for x in units:
        s0, s1 = x['s0'], x['s1']; W = s1 - s0
        kd = x['kind']
        yb = _unit_base(w, b, ctx, s0, s1)
        v1 = top_for(yb)
        if v1 - yb < 2.1:
            continue
        bayw = 2.7 if not wood else 2.1
        nb = max(1, int(round((W - pier) / bayw)))
        a = W / nb
        cen = [s0 + a * (i + 0.5) for i in range(nb)]
        ue = x['ue'] if x['ue'] is not None else (x['u'] if x['shop'] is not None else (s0 + s1) / 2)
        kdoor = int(np.argmin([abs(c - ue) for c in cen])) if kd != 'vacant' or rng.random() < 0.5 else -1
        fcol = (WOOD_FRONT if wood else ALU_FRONT)[int(rng_of('front', b.id, round(s0, 1)).integers(6 if not wood else 6))]
        if P.get('front_col'):
            fcol = P['front_col']
        x['bays'] = []; x['v1'] = v1; x['fy1'] = fy1; x['yb'] = yb; x['wood'] = wood; x['fcol'] = fcol
        x['yfloor'] = yb
        riser = RISER.get(kd, 0.35) if not wood else 0.08
        for i, uc in enumerate(cen):
            ww = a - pier
            if ww < 0.9:
                continue
            if FA.is_party(w, uc) or not FA.open_at(w, b, vol, uc, yb + 1.5, deltas=(0.45, 1.0))[0]:
                continue
            if i == kdoor:
                dw = 2.1 if (ww >= 2.4 and W >= 8.5 and not wood) else min(ww, 1.25)
                if ue is not None and x['ue'] is not None:
                    uc = float(np.clip(x['ue'], uc - (ww - dw) / 2, uc + (ww - dw) / 2))
                u0, u1 = uc - dw / 2, uc + dw / 2
                v0 = FA.local_level(w, b, ctx, u0 - 0.2, u1 + 0.2, lo=yb - 0.6, hi=min(yb + 0.6, v1 - 2.05))
                if not FA.rect_fits(w, u0, u1, v0, v1, 0.15, 0.3, 0.0):
                    continue
                x['yfloor'] = min(x['yfloor'], v0)
                op = FA.Opening(u0, u1, v0, v1, 0.16, ri=0.0, zone='gf', kind='shopdoor', back='#2C2824', inner='#CFC8BC')
                key = ('sfw_door' if wood else ('sf_door2' if dw > 1.8 else 'sf_door'))
                op.add(key, 'frame', uc, v0, dw, v1 - v0, fcol)
                op.add('glass', 'frame', uc, v0, dw, v1 - v0, '#D0D8DC', unit=True)
                op.room = x; op.role = 'door'
                w.openings.append(op); x['bays'].append(op)
                x['door_u'] = (u0, u1)
                # Rest der Achse neben einer schmalen Tuer: verglaste Seitenteile statt geschlossener Wand
                bu0, bu1 = cen[i] - ww / 2, cen[i] + ww / 2
                for sa, sb in ((bu0, u0 - 0.14), (u1 + 0.14, bu1)):
                    if sb - sa < 0.7:
                        continue
                    sc_ = (sa + sb) / 2
                    if FA.is_party(w, sc_) or not FA.open_at(w, b, vol, sc_, yb + 1.5, deltas=(0.45, 1.0))[0]:
                        continue
                    yl2 = FA.local_level(w, b, ctx, sa, sb, lo=yb - 0.5, hi=min(yb + 0.5, v1 - riser - 1.6))
                    vs0 = yl2 + riser
                    if not FA.rect_fits(w, sa, sb, vs0, v1, 0.15, 0.3, 0.05):
                        continue
                    x['yfloor'] = min(x['yfloor'], yl2)
                    op2 = FA.Opening(sa, sb, vs0, v1, 0.16, ri=0.0, zone='gf', kind='shopwin', back='#2C2824', inner='#CFC8BC')
                    op2.add('sfw_fix' if wood else ('sf_fix_t' if v1 - vs0 > 3.0 else 'sf_fix'), 'frame', sc_, vs0, sb - sa, v1 - vs0, fcol)
                    op2.add('glass', 'frame', sc_, vs0, sb - sa, v1 - vs0, '#D0D8DC', unit=True)
                    op2.room = x; op2.role = 'win'
                    w.openings.append(op2); x['bays'].append(op2)
            else:
                u0, u1 = uc - ww / 2, uc + ww / 2
                yl = FA.local_level(w, b, ctx, u0, u1, lo=yb - 0.5, hi=min(yb + 0.5, v1 - riser - 1.6))
                v0 = yl + riser
                if not FA.rect_fits(w, u0, u1, v0, v1, 0.15, 0.3, 0.05):
                    continue
                x['yfloor'] = min(x['yfloor'], yl)
                op = FA.Opening(u0, u1, v0, v1, 0.16, ri=0.0, zone='gf', kind='shopwin', back='#2C2824', inner='#CFC8BC')
                key = 'sfw_fix' if wood else ('sf_fix_t' if v1 - v0 > 3.0 else 'sf_fix')
                op.add(key, 'frame', uc, v0, ww, v1 - v0, fcol)
                op.add('glass', 'frame', uc, v0, ww, v1 - v0, '#D0D8DC', unit=True)
                op.room = x; op.role = 'win'
                w.openings.append(op); x['bays'].append(op)
        if x['bays']:
            x['room'] = _room(w, b, ctx, x, flags)
            flags.setdefault('units', []).append((w, x))
            if x['shop'] is not None:
                db()['used'].add(x['shop']['id'])


ALT_WOOD_STREETS = {'Kramerstraße', 'Burgstraße', 'Knochenhauerstraße', 'Holzmarkt', 'Pferdestraße', 'Ballhofplatz', 'Kreuzstraße',
                    'Roßmühle', 'Am Markte', 'Schuhstraße', 'Kreuzkirchhof'}


def _unit_base(w, b, ctx, s0, s1):
    """Fussniveau einer Ladeneinheit: Gehweg davor (Hoehenfeld der Boden-Pipeline, Median ueber die Breite), nicht das
    tiefste LoD2-Bodenniveau (Rampen/Promenade/Ufer) - sonst stehen Schaufenster und Tueren im Boden."""
    import facade as FA
    return FA.local_level(w, b, ctx, s0, s1, hi=b.y_ground + 2.5)


def _room(w, b, ctx, x, flags):
    """Tiefe des Ladenraums: soweit das eigene LoD2-Volumen reicht (max 5 m), keine Ueberschneidung mit anderen Raeumen."""
    vol = ctx['vol']; r = 0.16
    u0 = min(o.u0 for o in x['bays']) - 0.12; u1 = max(o.u1 for o in x['bays']) + 0.12
    u0 = max(u0, x['s0']); u1 = min(u1, x['s1'])
    v0 = min(x['yb'], x.get('yfloor', x['yb'])) - 0.01; v1 = x['v1'] + 0.3
    us = np.linspace(u0 + 0.15, u1 - 0.15, 4)
    D = 0.0
    for k in np.arange(0.6, 5.31, 0.35):
        Q = w.fr.w(np.c_[us, np.full(len(us), v0 + 1.4), np.full(len(us), -r - k)])
        bo, bs = vol.blocked(Q, b.idx)
        if bo.any() or not bs.all():
            break
        D = k
    D = D - 0.3
    if x['kind'] == 'department':
        D = min(D, 1.6)
    D = float(np.clip(D, 1.0, 5.0))
    rooms = flags.setdefault('rooms', [])

    def rect(Dd):
        c = [w.fr.w([u, 0, d]) for u, d in ((u0, -r), (u1, -r), (u1, -r - Dd), (u0, -r - Dd))]
        return Polygon([(p[0], p[2]) for p in c])
    pg = rect(D)
    while D > 1.0 and any(pg.intersection(q).area > 0.02 for q in rooms):
        D -= 0.35; pg = rect(D)
    D = max(D, 0.8)
    rooms.append(rect(D))
    return {'u0': u0, 'u1': u1, 'v0': v0, 'v1': v1, 'D': D, 'r': r}


# ------------------------------------------------------------------ Ladenraum (Huelle) + Auslagen (Instanzen)
def emit_rooms(acc, inst, w, b, ops_ok):
    """Fuer jede Ladeneinheit an der Wand: geschlossener Raum (Innenflaechen zeigen nach innen) + Auslagen-Instanzen."""
    from facade import _oriented
    by = collections.OrderedDict()
    for o in ops_ok:
        x = getattr(o, 'room', None)
        if x is not None and x.get('room'):
            by.setdefault(id(x), (x, []))[1].append(o)
    fr = w.fr
    for x, ops in by.values():
        R = x['room']; u0, u1, v0, v1, D, r = R['u0'], R['u1'], R['v0'], R['v1'], R['D'], R['r']
        kd = x['kind']
        wc, fc, cc, light = ROOM_STYLE.get(kd, ROOM_STYLE['generic'])
        rng = rng_of('room', b.id, round(u0, 2))
        wc = jitter_hex(wc, rng, 0.03)
        dF, dB = -r, -r - D
        L = lambda pts: fr.w(np.asarray(pts, float))
        # Boden, Decke, Rueckwand, Seiten (nach innen)
        _oriented(acc, 'shell', 'shop_interior', L([[u0, v0, dF], [u1, v0, dF], [u1, v0, dB], [u0, v0, dB]]), [[0, 1, 2], [0, 2, 3]], [fr.b] * 2, fc)
        _oriented(acc, 'shell', 'shop_interior', L([[u0, v1, dF], [u1, v1, dF], [u1, v1, dB], [u0, v1, dB]]), [[0, 1, 2], [0, 2, 3]], [-fr.b] * 2, cc)
        _oriented(acc, 'shell', 'shop_interior', L([[u0, v0, dB], [u1, v0, dB], [u1, v1, dB], [u0, v1, dB]]), [[0, 1, 2], [0, 2, 3]], [fr.n] * 2, wc)
        _oriented(acc, 'shell', 'shop_interior', L([[u0, v0, dF], [u0, v0, dB], [u0, v1, dB], [u0, v1, dF]]), [[0, 1, 2], [0, 2, 3]], [fr.t] * 2,
                  jitter_hex(wc, rng, 0.04))
        _oriented(acc, 'shell', 'shop_interior', L([[u1, v0, dF], [u1, v0, dB], [u1, v1, dB], [u1, v1, dF]]), [[0, 1, 2], [0, 2, 3]], [-fr.t] * 2,
                  jitter_hex(wc, rng, 0.04))
        # Vorderwand innen (mit den Oeffnungen als Loecher)
        holes = [np.array([[o.u0, o.v0], [o.u1, o.v0], [o.u1, o.v1], [o.u0, o.v1]]) for o in ops]
        acc.poly('shell', 'shop_interior', fr, np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]), wc, holes, d=dF, flip=True)
        # Licht
        W = u1 - u0; uc = (u0 + u1) / 2
        if light:
            inst.add(light, None, fr, uc, v1, dF - D / 2, W / 4.0, 1.0, D / 4.0, '#FFFFFF')
        if kd in ('fashion', 'sports', 'shoes', 'department') and D > 2.2:
            inst.add('dsp_track', None, fr, uc, v1, dF - 1.05, W / 4.0, 1.0, 1.0, '#FFFFFF')
        # Auslagen hinter jedem Schaufenster
        nvar = 3                                     # Varianten je Auslagenart (displays.NVAR)
        for k, o in enumerate(ops):
            if o.role != 'win':
                continue
            ww = o.u1 - o.u0; c = (o.u0 + o.u1) / 2
            sz = min(1.0, (D - 0.15) / 1.6)
            sets = [(c, ww)] if ww < 3.4 else [(c - ww / 4, ww / 2), (c + ww / 4, ww / 2)]
            for (cu, cw) in sets:
                sx = float(np.clip(cw / 2.6, 0.7, 1.25))
                var = int(rng.integers(nvar))
                inst.add('dsp_%s_bay%d' % (kd, var), None, fr, cu, v0, dF - 0.03, sx, 1.0, sz, _tint(rng))
        if kd == 'vacant':
            continue
        if D >= 2.3:
            nbk = max(1, int(round(W / 4.0)))
            for i in range(nbk):
                cu = u0 + W * (i + 0.5) / nbk
                inst.add('dsp_%s_back%d' % (kd, int(rng.integers(2))), None, fr, cu, v0, dB + 0.005, W / nbk / 4.0, 1.0,
                         float(min(1.0, (D - 1.6) / 1.0)), _tint(rng))
        if D >= 3.6:
            nm = max(1, int(W / 2.8))
            for i in range(nm):
                cu = u0 + W * (i + 0.5) / nm
                inst.add('dsp_%s_mid%d' % (kd, int(rng.integers(2))), None, fr, cu, v0, (dF - 1.75 + dB + 1.05) / 2, 1.0, 1.0, 1.0, _tint(rng))


def _tint(rng):
    v = float(rng.uniform(0.94, 1.0))
    return np.array([v, v, v])


# ------------------------------------------------------------------ Werbeanlagen
def _hexok(h, default):
    if isinstance(h, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', h):
        return h.upper()
    return default


def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def emit_signs(acc, inst, flags, b, P, ctx):
    """Schilder je Ladeneinheit (echter Name), Hausnummern, Markisen, Aussengastronomie."""
    rng = rng_of('signs', b.id)
    wall_col = P.get('gf_col_eff', P['wall_col'])
    for w, x in flags.get('units', []):
        fr = w.fr
        s = x['shop']
        v1, fy1 = x['v1'], x['fy1']
        ua = min(o.u0 for o in x['bays']) - 0.12; ub = max(o.u1 for o in x['bays']) + 0.12
        ua = max(ua, x['s0']); ub = min(ub, x['s1'])
        awn = s is not None and x['kind'] in ('cafe', 'bakery') and rng_of('awn', s['id']).random() < (0.75 if x['wood'] else 0.45)
        ys = v1 + (0.3 if awn else 0.14); ye = min(fy1 - (0.3 if w.jetty else 0.12), ys + 0.85)
        # Schildflaeche merken: Fachwerk-Knaggen/Konsolen werden hier nicht vor das Schild gesetzt
        P.setdefault('_sign_spans', []).append((w.fr, ua, ub, v1, max(ye, v1 + 0.3)))
        if awn:
            _awning(acc, fr, x, v1 + 0.05, '#7A2A26' if x['wood'] else ['#7A2E2A', '#2E4A3A', '#3A3D40', '#C8B48A', '#1F3550'][int(rng.integers(5))])
        if s is None:
            # unbekannter Laden: nur ein leeres Blendband (kein erfundener Name)
            if x['kind'] != 'vacant' and ye - ys > 0.3 and rng.random() < 0.5:
                acc.box('detail', 'trim', fr, [ua, ys, 0], [ub, ys + min(0.55, ye - ys), 0.05], ['#2F3337', '#E6E2DA', '#5B6066'][int(rng.integers(3))],
                        skip=('-d',))
            continue
        sg = s['sign']; t = sg.get('type') or 'none'
        text = (sg.get('text') if sg.get('text') is not None else ('' if t == 'logo_only' else s.get('name')) or '').strip()
        fg = _hexok(sg.get('fg'), '#FFFFFF'); bg = _hexok(sg.get('bg'), '#2F3337')
        font = sg.get('font') or 'sans'
        if (not text and t != 'logo_only') or t == 'none':
            _house_number(acc, fr, x, s, flags)
            continue
        if _lum(fg) > 0.85 and _lum(bg) > 0.85:
            fg = '#22201F'
        hh = ye - ys
        types = t.split('+')
        if hh < 0.28:
            types = ['window_lettering'] + [k for k in types if k == 'projecting_blade']
        for k in types:
            if k == 'fascia_panel':
                _fascia(acc, fr, ua, ub, ys, ys + min(hh, 0.75), bg, fg, text, font, lit=False,
                        on_letters=('channel_letters' in types))
            elif k == 'channel_letters' and 'fascia_panel' not in types:
                need_plate = abs(_lum(fg) - _lum(wall_col)) < 0.25
                _channel(acc, fr, ua, ub, ys, ys + min(hh, 0.8), fg, bg, text, font, plate=need_plate, wall_col=wall_col)
            elif k == 'light_box':
                _lightbox(acc, fr, ua, ub, ys, ys + min(hh, 0.75), bg, fg, text, font)
            elif k == 'logo_only':
                _logo(acc, fr, ua, ub, ys, ys + min(hh, 0.8), bg, fg, text, font, brand=(s.get('brand') or s.get('name') or ''))
            elif k == 'projecting_blade':
                _blade(acc, fr, w, x, b, bg, fg, text, font)
            elif k == 'window_lettering':
                _vinyl(acc, fr, x, fg if _lum(fg) < 0.9 or True else '#FFFFFF', text, font)
        # Apotheke: rotes A als Ausleger zusaetzlich (sofern nicht schon Ausleger)
        if x['kind'] == 'pharmacy' and 'projecting_blade' not in types:
            _blade(acc, fr, w, x, b, '#FFFFFF', '#E2001A', 'A', 'serif-bold', small=True)
        elif x['wood'] and 'projecting_blade' not in types and rng_of('iron', s['id']).random() < 0.55:
            # Altstadt: schmiedeeiserner Ausleger mit Holz-/Blechtafel (Name)
            _iron_blade(acc, fr, w, x, bg, fg, text, font)
        _record(flags, w, x, s)
        if x['kind'] in ('cafe', 'bakery', 'fastfood') and rng_of('aframe', s['id']).random() < 0.5:
            _aframe(inst, fr, x, ctx, b)
        if s.get('outdoor') and x['kind'] in ('cafe', 'bakery', 'fastfood'):
            _terrace(inst, fr, w, x, ctx, b)
        _house_number(acc, fr, x, s, flags)
    done = flags.setdefault('num_done', set())
    for w, u, v, num in flags.get('numbers', []):
        if num and _num_key(w.fr, num) not in done:
            done.add(_num_key(w.fr, num))
            _plaque(acc, w.fr, u, v, str(num))


def _fascia(acc, fr, u0, u1, y0, y1, bg, fg, text, font, lit=False, on_letters=False):
    acc.box('detail', 'trim', fr, [u0, y0, 0], [u1, y1, 0.06], bg, skip=('-d',))
    acc.box('detail', 'metal', fr, [u0 - 0.01, y0 - 0.025, 0], [u1 + 0.01, y0, 0.075], '#8C969C', skip=('-d',))
    acc.box('detail', 'metal', fr, [u0 - 0.01, y1, 0], [u1 + 0.01, y1 + 0.025, 0.075], '#8C969C', skip=('-d',))
    mat = glow_mat(fg) if on_letters else 'trim'
    dd = 0.05 if on_letters else 0.014
    TX.emit_text(acc, 'detail', mat, fr, text, font, u0 + 0.12, y0, u1 - u0 - 0.24, y1 - y0, 0.06, 0.06 + dd, fg, fill_h=0.6, fill_w=0.88,
                 max_cap=0.5)


def _shade(h, f):
    c = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return '#%02X%02X%02X' % tuple(int(max(0, min(255, round(v * f)))) for v in c)


def _channel(acc, fr, u0, u1, y0, y1, fg, bg, text, font, plate=False, wall_col=None):
    if plate:
        acc.box('detail', 'trim', fr, [u0, y0, 0], [u1, y1, 0.03], bg, skip=('-d',))
    d0 = 0.03 if plate else 0.0
    w, s = TX.emit_text(acc, 'detail', glow_mat(fg), fr, text, font, u0 + 0.1, y0, u1 - u0 - 0.2, y1 - y0, d0 + 0.025, d0 + 0.1, fg,
                        fill_h=0.68, fill_w=0.86, max_cap=0.6)
    if w > 0 and not plate:   # Montageschiene (Kabelkanal) hinter den Buchstaben, in Wandfarbe, im unteren Buchstabendrittel
        uc = (u0 + u1) / 2
        hy = (y1 - y0) / 2 - s / 2 + y0 + s * 0.28
        rc = _shade(wall_col, 0.9) if isinstance(wall_col, str) and wall_col.startswith('#') else '#8C8A86'
        acc.box('detail', 'metal', fr, [uc - w / 2 + 0.04, hy - 0.025, 0], [uc + w / 2 - 0.04, hy + 0.025, 0.022], rc, skip=('-d',))


def _lightbox(acc, fr, u0, u1, y0, y1, bg, fg, text, font):
    acc.box('detail', 'metal', fr, [u0, y0, 0], [u1, y1, 0.17], '#3A3D40', skip=('-d', '+d'))
    acc.box('detail', 'metal', fr, [u0, y0, 0.0], [u0 + 0.04, y1, 0.175], '#3A3D40', skip=('-d',))
    acc.box('detail', 'metal', fr, [u1 - 0.04, y0, 0.0], [u1, y1, 0.175], '#3A3D40', skip=('-d',))
    acc.box('detail', 'metal', fr, [u0 + 0.04, y0, 0.0], [u1 - 0.04, y0 + 0.04, 0.175], '#3A3D40', skip=('-d',))
    acc.box('detail', 'metal', fr, [u0 + 0.04, y1 - 0.04, 0.0], [u1 - 0.04, y1, 0.175], '#3A3D40', skip=('-d',))
    acc.poly('detail', glow_mat(bg, 0.7), fr, np.array([[u0 + 0.04, y0 + 0.04], [u1 - 0.04, y0 + 0.04], [u1 - 0.04, y1 - 0.04], [u0 + 0.04, y1 - 0.04]]),
             '#FFFFFF', d=0.165)
    TX.emit_text(acc, 'detail', 'trim', fr, text, font, u0 + 0.1, y0, u1 - u0 - 0.2, y1 - y0, 0.165, 0.178, fg, fill_h=0.58, fill_w=0.86,
                 max_cap=0.45)


def _fruit_poly(s):
    """Schlichte Apfelfrucht-Silhouette (Kreis-Vereinigung + Blatt), Hoehe s, Ursprung unten Mitte - keine Logo-Kopie."""
    from shapely.geometry import Point as _P
    from shapely.affinity import rotate, scale, translate as translate_
    body = scale(_P(0.0, 0.0).buffer(1.0, 32), 0.36 * s, 0.4 * s)            # eine glatte, konvexe Frucht
    body = translate_(body, 0.0, 0.4 * s)
    leaf = rotate(scale(_P(0.0, 0.0).buffer(0.1 * s, 16), 1.0, 0.5), 40).buffer(0)
    from shapely.affinity import translate
    leaf = translate(leaf, 0.08 * s, 0.9 * s)
    from shapely.geometry.polygon import orient as _or
    out = []
    for g in (body, leaf):
        g = g.simplify(0.01 * s)
        for p in ([g] if g.geom_type == 'Polygon' else list(g.geoms)):
            p = _or(p, 1.0)
            out.append((np.asarray(p.exterior.coords)[:-1], []))
    return out


def _logo(acc, fr, u0, u1, y0, y1, bg, fg, text, font, brand=''):
    """Markenzeichen ohne Logo-Nachbau: Leuchtkasten in Markenfarbe mit dem Markennamen (kurz) bzw. schlichte Plakette."""
    h = min(y1 - y0, 0.9)
    uc = (u0 + u1) / 2
    if not text and 'apple' in brand.lower():
        # Steinplatte + weiss hinterleuchtete, schlichte Apfel-Silhouette (Street View: kleine weisse Plakette am Stein)
        s = min(h, 0.62)
        from facade import _oriented
        acc.box('detail', 'stone', fr, [uc - s * 0.62, y0, 0], [uc + s * 0.62, y0 + s * 1.12, 0.03], '#3F4245', skip=('-d',))
        V, F, W = TX.extrude_local(_fruit_poly(s * 0.86), 0.03, 0.06)
        if len(F):
            V = V + np.array([uc, y0 + s * 0.1, 0.0])
            _oriented(acc, 'detail', glow_mat('#F5F5F7', 0.7), fr.w(V), F, W @ fr.M, '#F5F5F7')
        return
    if not text:
        s = min(h, 0.55)
        acc.box('detail', 'metal', fr, [uc - s / 2, y0, 0], [uc + s / 2, y0 + s, 0.05], '#B8BCBE', skip=('-d',))
        acc.poly('detail', glow_mat(bg if _lum(bg) > 0.5 else '#F5F5F7', 0.6), fr,
                 np.array([[uc - s / 2 + 0.04, y0 + 0.04], [uc + s / 2 - 0.04, y0 + 0.04], [uc + s / 2 - 0.04, y0 + s - 0.04],
                           [uc - s / 2 + 0.04, y0 + s - 0.04]]), '#FFFFFF', d=0.051)
        return
    _, cap, tw = TX.layout(text, font, 40.0, h, fill_h=0.55, fill_w=0.9)
    bw = float(np.clip(tw + 0.35, h, u1 - u0))
    acc.box('detail', 'metal', fr, [uc - bw / 2, y0, 0], [uc + bw / 2, y0 + h, 0.12], '#3A3D40', skip=('-d', '+d'))
    acc.poly('detail', glow_mat(bg, 0.7), fr, np.array([[uc - bw / 2, y0], [uc + bw / 2, y0], [uc + bw / 2, y0 + h], [uc - bw / 2, y0 + h]]),
             '#FFFFFF', d=0.12)
    TX.emit_text(acc, 'detail', 'trim', fr, text, font, uc - bw / 2, y0, bw, h, 0.12, 0.135, fg, fill_h=0.55, fill_w=0.86)


def _blade(acc, fr, w, x, b, bg, fg, text, font, small=False):
    from geom import Frame
    # Position: an der Kante der Einheit (zur Mitte der Wand hin), Hoehe Brustung 1. OG
    u = x['s0'] + 0.35 if (x['s0'] - w.umin) > (w.umax - x['s1']) else x['s1'] - 0.35
    y0 = x['fy1'] + 0.35
    if y0 + 1.0 > w.vcorner - 0.5:
        y0 = x['v1'] + 0.2
    tall = (len(text) > 7 and not small)
    bw, bh = (0.55, min(2.2, max(1.2, 0.17 * len(text)))) if tall else ((0.62, 0.62) if small else (0.9, 0.6))
    if y0 + bh > w.vcorner - 0.3:
        bh = max(0.5, w.vcorner - 0.3 - y0)
    th = 0.09
    up = np.array([0, 1.0, 0])
    O = fr.w([u, 0.0, 0.0])
    A = Frame(O, fr.n, up, -fr.t)          # Seite A (Normale -t), Text laeuft von der Wand weg
    B = Frame(O, -fr.n, up, fr.t)          # Seite B
    gap = 0.12
    # Halter + Arm
    acc.box('detail', 'metal', fr, [u - 0.05, y0 + bh * 0.2, 0], [u + 0.05, y0 + bh * 0.8, 0.03], '#3A3D40', skip=('-d',))
    acc.box('detail', 'metal', A, [0.0, y0 + bh * 0.45, -0.02], [gap, y0 + bh * 0.55, 0.02], '#3A3D40', skip=('-u',))
    lm = glow_mat(bg, 0.6)
    acc.box('detail', 'metal', A, [gap, y0, -th / 2], [gap + bw, y0 + bh, th / 2], '#3A3D40', skip=('-d', '+d'))
    for F, uu0 in ((A, gap), (B, -gap - bw)):
        acc.poly('detail', lm, F, np.array([[uu0 + 0.03, y0 + 0.03], [uu0 + bw - 0.03, y0 + 0.03], [uu0 + bw - 0.03, y0 + bh - 0.03],
                                             [uu0 + 0.03, y0 + bh - 0.03]]), '#FFFFFF', d=th / 2)
        acc.box('detail', 'metal', F, [uu0, y0, th / 2 - 0.02], [uu0 + bw, y0 + 0.03, th / 2 + 0.004], '#3A3D40', skip=('-d',))
        acc.box('detail', 'metal', F, [uu0, y0 + bh - 0.03, th / 2 - 0.02], [uu0 + bw, y0 + bh, th / 2 + 0.004], '#3A3D40', skip=('-d',))
        if tall:
            # senkrechte Schrift (von unten nach oben): Rahmen mit t = oben, b = zur Wand
            R = Frame(F.O + F.t * (uu0 + bw) + up * y0, up, -F.t, F.n)
            TX.emit_text(acc, 'detail', 'trim', R, text, font, 0.06, 0.0, bh - 0.12, bw, th / 2, th / 2 + 0.012, fg, fill_h=0.62, fill_w=0.92)
        else:
            TX.emit_text(acc, 'detail', 'trim', F, text, font, uu0 + 0.04, y0, bw - 0.08, bh, th / 2, th / 2 + 0.012, fg, fill_h=0.62, fill_w=0.86)


def _iron_blade(acc, fr, w, x, bg, fg, text, font):
    """Altstadt-Ausleger: schmiedeeiserner Arm mit Zierbogen und Strebe, darunter haengende Tafel (beidseitig Name).
    Bei Fachwerk mit Auskragung sitzt die Wandplatte auf der vorkragenden Geschossflaeche oberhalb der Schwelle und der Arm
    beginnt vor Staendern/Riegeln (Abstand 5 cm) - Tafel und Arm schneiden weder Schwelle, Balkenkoepfe noch Knaggen."""
    from geom import Frame, beam
    import facade as FA
    u = x['s0'] + 0.3 if (x['s0'] - w.umin) > (w.umax - x['s1']) else x['s1'] - 0.3
    ya = x['fy1'] + 0.55
    if ya + 0.3 > w.vcorner - 0.4:
        ya = x['v1'] + 0.85
    dz = FA.jetty_dz(w, ya) if w.jetty else 0.0
    off = dz + 0.045 if w.jetty else 0.0                   # Vorderkante Staender/Riegel bzw. Wandflaeche
    L = 0.95; ph = 0.5 if len(text) <= 10 else 0.62; pw = 0.78
    up = np.array([0, 1.0, 0])
    O = fr.w([u, 0.0, off])
    A = Frame(O, fr.n, up, -fr.t)                       # u' = von der Wand weg, v' = oben, d' = seitlich
    iron = '#1E1E1E'
    pl0 = ya - (0.3 if w.jetty else 0.42)
    if w.jetty and pl0 < x['fy1'] + 0.22:                # nicht in die Schwelle
        return
    acc.box('detail', 'metal', fr, [u - 0.05, pl0, off], [u + 0.05, ya + 0.06, off + 0.02], iron, skip=('-d',))   # Wandplatte
    a0 = 0.02
    beam(acc, 'detail', 'metal', A.w([a0, ya, 0.0]), A.w([L, ya, 0.0]), 0.03, 0.03, iron)                       # Arm
    beam(acc, 'detail', 'metal', A.w([a0, pl0 + 0.04, 0.0]), A.w([L * 0.62, ya - 0.015, 0.0]), 0.022, 0.022, iron)  # Strebe
    a = np.linspace(0.15, np.pi - 0.15, 7)                                                                      # Zierbogen
    pts = [A.w([0.45 + 0.17 * np.cos(t), ya + 0.02 + 0.13 * np.sin(t), 0.0]) for t in a]
    for p0, p1 in zip(pts, pts[1:]):
        beam(acc, 'detail', 'metal', p0, p1, 0.016, 0.016, iron)
    beam(acc, 'detail', 'metal', A.w([L, ya + 0.015, 0.0]), A.w([L + 0.05, ya + 0.08, 0.0]), 0.03, 0.03, '#B8902F')  # Knauf
    # Tafel an zwei Ringen
    u0p = 0.14; u1p = u0p + pw; yb = ya - 0.08 - ph
    for uu in (u0p + 0.08, u1p - 0.08):
        beam(acc, 'detail', 'metal', A.w([uu, ya - 0.015, 0.0]), A.w([uu, yb + ph, 0.0]), 0.014, 0.014, iron)
    th = 0.035
    pc = bg if _lum(bg) < 0.75 else '#3A2A20'
    acc.box('detail', 'trim', A, [u0p, yb, -th / 2], [u1p, yb + ph, th / 2], pc, skip=())
    acc.box('detail', 'metal', A, [u0p - 0.015, yb - 0.015, -th / 2 - 0.004], [u1p + 0.015, yb + 0.012, th / 2 + 0.004], '#B8902F', skip=())
    tc = fg if abs(_lum(fg) - _lum(pc)) > 0.3 else ('#E8C468' if _lum(pc) < 0.5 else '#22201F')
    B = Frame(O, -fr.n, up, fr.t)
    for F, uu0 in ((A, u0p), (B, -u1p)):
        TX.emit_text(acc, 'detail', 'trim', F, text, font, uu0 + 0.05, yb, pw - 0.1, ph, th / 2, th / 2 + 0.01, tc,
                     fill_h=0.5, fill_w=0.9)
    P_sp = getattr(w, '_iron_spans', None)
    if P_sp is not None:
        P_sp.append((u, yb, ya + 0.15))


def _record(flags, w, x, s):
    """Lage des Ladens (Weltkoordinaten) fuer building_info.json (Vorschau-Kameras, Bericht)."""
    wins = [o for o in x['bays'] if o.role == 'win'] or x['bays']
    o = max(wins, key=lambda o: o.u1 - o.u0)
    c = w.fr.w([(o.u0 + o.u1) / 2, x['yb'], 0.0])
    flags.setdefault('shop_pos', []).append({'id': s['id'], 'name': s.get('name'), 'kind': x['kind'], 'p': [round(float(c[0]), 2),
                                             round(float(c[1]), 2), round(float(c[2]), 2)], 'n': [round(float(w.n[0]), 3), round(float(w.n[2]), 3)],
                                             'sign': (s.get('sign') or {}).get('type')})


def _vinyl(acc, fr, x, fg, text, font):
    wins = [o for o in x['bays'] if o.role == 'win'] or x['bays']
    o = max(wins, key=lambda o: o.u1 - o.u0)
    d = -0.16 - 0.046 + 0.004
    col = fg if _lum(fg) > 0.2 else '#F2F2EE'
    TX.emit_text(acc, 'detail', 'trim', fr, text, font, o.u0 + 0.15, o.v1 - 0.55, o.u1 - o.u0 - 0.3, 0.42, d, d + 0.003, col, fill_h=0.6, fill_w=0.8)


def _plaque(acc, fr, u, v, num):
    num = num.strip()[:6]
    w = 0.12 + 0.07 * len(num)
    acc.box('detail', 'trim', fr, [u - w / 2, v - 0.09, 0], [u + w / 2, v + 0.09, 0.012], '#1F3F8F', skip=('-d',))
    acc.box('detail', 'trim', fr, [u - w / 2 + 0.015, v - 0.075, 0.012], [u + w / 2 - 0.015, v + 0.075, 0.014], '#F2F2EE', skip=('-d',))
    acc.box('detail', 'trim', fr, [u - w / 2 + 0.025, v - 0.065, 0.014], [u + w / 2 - 0.025, v + 0.065, 0.016], '#1F3F8F', skip=('-d',))
    TX.emit_text(acc, 'detail', 'trim', fr, num, 'bold-sans', u - w / 2 + 0.03, v - 0.065, w - 0.06, 0.13, 0.016, 0.019, '#F2F2EE',
                 fill_h=0.75, fill_w=0.85)


def _num_key(fr, num):
    """Hausnummer je Gebaeude und Fassadenrichtung (30-Grad-Sektoren) nur einmal - keine zwei '9' an derselben Front."""
    a = math.degrees(math.atan2(fr.n[2], fr.n[0])) % 360.0
    return (str(num).replace(' ', ''), int(((a + 15.0) % 360.0) // 30.0))


def _house_number(acc, fr, x, s, flags):
    m = re.search(r'\s(\d+[a-zA-Z]?(?:\s*-\s*\d+)?)\s*$', s.get('address') or '')
    if not m or 'door_u' not in x:
        return
    key = _num_key(fr, m.group(1))
    done = flags.setdefault('num_done', set())
    if key in done:
        return
    done.add(key)
    u0, u1 = x['door_u']
    u = u1 + 0.22 if u1 + 0.4 < x['s1'] + 0.3 else u0 - 0.22
    _plaque(acc, fr, u, x['yb'] + 2.3 if x['v1'] - x['yb'] > 2.6 else x['v1'] + 0.05, m.group(1).replace(' ', ''))


def _awning(acc, fr, x, ya, cc):
    from geom import beam
    u0 = min(o.u0 for o in x['bays']) - 0.05; u1 = max(o.u1 for o in x['bays']) + 0.05
    d1 = 1.3; drop = 0.5
    pts = np.array([[u0, ya, 0.0], [u1, ya, 0.0], [u1, ya - drop, d1], [u0, ya - drop, d1]])
    acc.add('detail', 'interior', fr.w(pts), [[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]], cc)
    val = np.array([[u0, ya - drop, d1], [u1, ya - drop, d1], [u1, ya - drop - 0.22, d1], [u0, ya - drop - 0.22, d1]])
    acc.add('detail', 'interior', fr.w(val), [[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]], cc)
    for uu in (u0, u1):
        tri = np.array([[uu, ya, 0.0], [uu, ya - drop, d1], [uu, ya - drop - 0.22, d1], [uu, ya - 0.05, 0.0]])
        acc.add('detail', 'interior', fr.w(tri), [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], cc)
    acc.box('detail', 'metal', fr, [u0 - 0.03, ya - 0.02, 0], [u1 + 0.03, ya + 0.14, 0.16], '#B4B8BA', skip=('-d',))
    for uu in (u0 + 0.15, u1 - 0.15):
        beam(acc, 'detail', 'metal', fr.w([uu, ya - 0.6, 0.02]), fr.w([uu, ya - drop + 0.02, d1 - 0.05]), 0.035, 0.035, '#8C969C')


def _free_front(ctx, b, fr, u, d, w=0.6):
    """Ist der Gehweg vor der Wand bei (u, d) frei von Gebaeuden?"""
    vol = ctx['vol']
    Q = fr.w(np.array([[u + du, b.y_ground + 1.0, d + dd] for du in (-w, 0, w) for dd in (-0.4, 0, 0.4)]))
    bo, bs = vol.blocked(Q, -1)
    return not (bo | bs).any()


def _ground_y(ctx, b, p):
    import level as LV
    y = LV.sample([p[0]], [p[2]])[0]
    return float(y) if np.isfinite(y) and abs(y - b.y_ground) < 2.5 else b.y_ground


def _aframe(inst, fr, x, ctx, b):
    if 'door_u' not in x:
        return
    u = x['door_u'][1] + 0.6
    if u > x['s1'] + 0.3 or not _free_front(ctx, b, fr, u, 0.9, 0.3):
        return
    p = fr.w([u, 0, 0.9])
    from geom import Frame
    inst.add('dsp_aframe', None, Frame(fr.O, fr.t, fr.b, fr.n), u, _ground_y(ctx, b, p), 0.9, 1.0, 1.0, 1.0, '#FFFFFF')


def _terrace(inst, fr, w, x, ctx, b):
    u0, u1 = x['s0'] + 0.6, x['s1'] - 0.6
    if u1 - u0 < 1.2:
        return
    n = max(1, int((u1 - u0) / 1.9))
    rng = rng_of('terrace', x['shop']['id'])
    rows = (1.5,) if rng.random() < 0.6 else (1.5, 3.0)
    for dd in rows:
        for i in range(n):
            u = u0 + (u1 - u0) * (i + 0.5) / n
            if not _free_front(ctx, b, fr, u, dd, 0.8):
                continue
            p = fr.w([u, 0, dd])
            inst.add('dsp_tbl_out', None, fr, u, _ground_y(ctx, b, p), dd, 1.0, 1.0, 1.0, '#FFFFFF')
        if rng.random() < 0.5 and dd == rows[-1]:
            for i in range(0, n, 2):
                u = u0 + (u1 - u0) * (i + 0.5) / n
                if _free_front(ctx, b, fr, u, dd, 1.3):
                    p = fr.w([u, 0, dd])
                    inst.add('dsp_parasol', None, fr, u, _ground_y(ctx, b, p), dd, 1.0, 1.0, 1.0, '#FFFFFF')


# ------------------------------------------------------------------ grosse Werbeanlagen an Obergeschossen / Glasvordach (Overrides)
def _plane_of(ws):
    """Gemeinsame senkrechte Ebene vor mehreren Wandsegmenten (z. B. gerundete Ecke): Rahmen (t, oben, n) mit Ursprung auf
    der Mitte der Segmentfolge, so weit vorn, dass kein Segment davor liegt. -> (Frame, Breite)"""
    from geom import Frame
    n2 = sum(w.n[[0, 2]] * max(w.L, 0.1) for w in ws)
    n2 = n2 / max(1e-9, np.linalg.norm(n2))
    n3 = np.array([n2[0], 0.0, n2[1]]); t3 = np.array([n2[1], 0.0, -n2[0]])
    pts = []
    for w in ws:
        pts += [w.fr.w([w.umin, 0, 0]), w.fr.w([w.umax, 0, 0])]
    pts = np.array(pts)
    st = pts @ t3; sn = pts @ n3
    s0, s1 = float(st.min()), float(st.max())
    O = t3 * (s0 + s1) / 2 + n3 * float(sn.max())
    O[1] = 0.0
    return Frame(O, t3, np.array([0, 1.0, 0]), n3), s1 - s0


def emit_big_signs(acc, walls, b, P, G):
    """style.OVERRIDES 'big_signs': grosse Einzelbuchstaben (hinterleuchtet) an Obergeschossen, z. B. H&M an der Runden Ecke
    (3. OG), Hugendubel vor dem 1. OG, GALERIA an der Metallfassade. Keine Logos - nur der Name in einer freien Schrift.
    Eintrag: {'text', 'walls': [idx..], 'y': Hoehe der Unterkante ueber Bezugsniveau, 'h': Feldhoehe, 'fg', 'font',
    optional 'w' (Feldbreite), 'du' (Versatz entlang der Wand), 'd' (Abstand vor der Wand), 'lit'}."""
    import facade as FA
    for sg in P.get('big_signs') or ():
        ws = [w for w in walls if w.idx in sg['walls'] and not w.drop and w.vertical]
        if not ws:
            continue
        fr, W = _plane_of(ws)
        y0 = (FA.floor_y(b, G, int(sg['floor'])) if 'floor' in sg else b.y_ground) + float(sg.get('y', 0.0)); h = float(sg['h'])
        bw = float(sg.get('w', W * 0.9)); uc = float(sg.get('du', 0.0))
        d0 = float(sg.get('d', 0.1))
        fg = sg.get('fg', '#FFFFFF')
        mat = glow_mat(fg, 0.75) if sg.get('lit', True) else 'trim'
        tw, cap = TX.emit_text(acc, 'detail', mat, fr, sg['text'], sg.get('font', 'bold-sans'), uc - bw / 2, y0, bw, h, d0, d0 + 0.12, fg,
                               fill_h=0.86, fill_w=0.96)
        if tw <= 0:
            continue
        # zwei Montageschienen hinter den Buchstaben (in Wandfarbe, aus der Wand heraus bis an die Buchstaben)
        wc = _shade(P['wall_col'], 0.82) if str(P.get('wall_col', '')).startswith('#') else '#8C8A86'
        yc = y0 + h / 2
        for yy in (yc - cap * 0.25, yc + cap * 0.2):
            acc.box('detail', 'metal', fr, [uc - tw / 2 + 0.05, yy - 0.03, -0.35], [uc + tw / 2 - 0.05, yy + 0.03, d0], wc, skip=('-d',))
        if sg.get('plate'):
            # Tafel unter den Buchstaben (z. B. Hugendubel 'Welt der Buecher')
            pt = sg['plate']
            pw = float(pt.get('w', 2.6)); ph = float(pt.get('h', 0.5)); py = y0 - ph - 0.12
            acc.box('detail', 'trim', fr, [uc - pw / 2, py, -0.3], [uc + pw / 2, py + ph, d0], pt.get('bg', '#C8102E'), skip=('-d',))
            TX.emit_text(acc, 'detail', 'trim', fr, pt['text'], pt.get('font', 'sans'), uc - pw / 2 + 0.08, py, pw - 0.16, ph, d0,
                         d0 + 0.015, pt.get('fg', '#FFFFFF'), fill_h=0.55, fill_w=0.9)


def emit_glass_canopy(acc, walls, b, P, G, flags):
    """style.OVERRIDES 'glass_canopy': durchlaufendes, leicht geneigtes Glasvordach mit dunklem Stahlrahmen und Zugstangen ueber
    den Schaufenstern (Hugendubel). {'walls': [idx..], 'y': Hoehe ueber Bezugsniveau (Wandanschluss), 'depth', 'rise'}."""
    from geom import beam
    cp = P.get('glass_canopy')
    if not cp:
        return
    steel = cp.get('col', '#2F3337')
    for w in walls:
        if w.idx not in cp['walls'] or w.drop or not w.vertical:
            continue
        fr = w.fr
        import facade as FA
        y = (FA.floor_y(b, G, int(cp['floor'])) if 'floor' in cp else b.y_ground) + float(cp.get('y', 0.0))
        D = float(cp.get('depth', 2.0)); rise = float(cp.get('rise', 0.25))
        spans = _span_free(w, b, y)
        for ua, ub in spans:
            if ub - ua < 2.0:
                continue
            # Glas als geschlossene, 12 mm dicke Platte (von oben und unten sichtbar)
            um = (ua + ub) / 2
            beam(acc, 'detail', 'glass', fr.w([um, y + 0.06, 0.08]), fr.w([um, y + rise + 0.06, D - 0.06]), ub - ua, 0.012, '#C4D0D6')
            # Rahmen: Wandprofil, Vorderkante, Sparren
            acc.box('detail', 'metal', fr, [ua, y - 0.04, 0.0], [ub, y + 0.1, 0.1], steel, skip=('-d',))
            beam(acc, 'detail', 'metal', fr.w([ua, y + rise + 0.02, D]), fr.w([ub, y + rise + 0.02, D]), 0.08, 0.14, steel)
            n = max(1, int((ub - ua) / 1.6))
            for k in range(n + 1):
                u = ua + 0.05 + (ub - ua - 0.1) * k / n
                beam(acc, 'detail', 'metal', fr.w([u, y + 0.02, 0.05]), fr.w([u, y + rise + 0.02, D]), 0.06, 0.1, steel)
                if k % 2 == 0:                                   # Zugstange, Anker unter der Fensterbruestung des 1. OG
                    ya = y + float(cp.get('tie', 0.62))
                    beam(acc, 'detail', 'metal', fr.w([u, ya, 0.0]), fr.w([u, y + rise + 0.08, D - 0.05]), 0.025, 0.025, steel)
            flags.setdefault('canopy_spans', []).append((w.idx, ua, ub, y))


def _span_free(w, b, y):
    """Wandabschnitte, an denen bei Hoehe y Wand vorhanden ist (fuer Vordach/Schilder)."""
    import facade as FA
    out = FA._span_at(w, y - 0.05, y + 1.4)
    return [(ua + 0.2, ub - 0.2) for ua, ub in out]

"""Kameras fuer die Vorschaubilder (Spielkoordinaten) + Gelaende-Raster (DGM1, 2 m) fuer Blender.
Schreibt cache/preview_cams.json und cache/terrain_2m.npz."""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, DATA, load_json, save_json
import prep as PREP
from style import classify


def street_axis(osm, name, xr, zr):
    pts = np.array([p for w in osm['ways'] if w['tags'].get('name') == name and 'highway' in w['tags'] for p in w['pts']
                    if xr[0] <= p[0] <= xr[1] and zr[0] <= p[1] <= zr[1]])
    c = pts.mean(0); u, s, vt = np.linalg.svd(pts - c)
    return c, vt[0]


CTX = None


def facade_cam(b, dist=13.0, eye=1.7, look_h=7.0, side=0.0):
    """Kamera vor der besten Strassenfassade (freiliegend, keine Brandwand, nahe Strasse, lang)."""
    import facade as FA
    walls = FA.analyse_walls(b, CTX)
    best = None
    for w in walls:
        if w.drop or not w.vertical or w.party_wall or w.L < 6 or w.vmin > b.y_ground + 1:
            continue
        FA.street_info(w, b, CTX)
        if w.blocked_frac > 0.3:
            continue
        sc = w.L - 0.8 * min(w.street_dist, 30)
        if best is None or sc > best[0]:
            best = (sc, w)
    if best is None:
        return None
    w = best[1]
    c = w.fr.w([(w.umin + w.umax) / 2, 0, 0]); n = w.n
    cam = c + n * dist + w.fr.t * side
    return {'loc': [float(cam[0]), b.y_ground + eye, float(cam[2])], 'tgt': [float(c[0]), b.y_ground + look_h, float(c[2])]}


def main():
    global CTX
    d = PREP.load_prep()
    from build_all import make_ctx
    CTX = make_ctx(d)
    osm = load_json(os.path.join(DATA, 'osm_features.json'))
    bl = [b for b in d['bldgs'] if not b.covered]
    cams = {}
    cams['facades_aerial'] = {'loc': [-170.0, 170.0, 190.0], 'tgt': [20.0, 0.0, -30.0], 'lens': 24, 'radius': 520}
    cams['facades_aerial_north'] = {'loc': [120.0, 200.0, -330.0], 'tgt': [-20.0, 0.0, -20.0], 'lens': 24, 'radius': 520}
    cams['facades_bahnhofstrasse_eye'] = {'loc': [0.0, 1.7, 0.0], 'tgt': [8.0, 5.0, -100.0], 'lens': 24, 'radius': 260}
    c, dvec = street_axis(osm, 'Georgstraße', (-260, -15), (-60, 80))
    if dvec[0] > 0:
        dvec = -dvec
    p0 = c - dvec * 50
    cams['facades_georgstrasse_west'] = {'loc': [float(p0[0]), 1.7, float(p0[1])], 'tgt': [float(p0[0] + dvec[0] * 100), 5.0,
                                                                                         float(p0[1] + dvec[1] * 100)], 'lens': 24, 'radius': 260}
    # Nahaufnahmen: Altbau / modern / Kaufhaus / Fachwerk / Kirche
    def pick(kind, near=(0, 0), minh=14):
        cand = [b for b in bl if classify(b) == kind and b.y_top - b.y_ground > minh and b.street]
        cand.sort(key=lambda b: math.hypot(b.cx - near[0], b.cz - near[1]))
        return cand[0] if cand else None
    for kind, nm, near, dist in (('altbau', 'facades_altbau_close', (150, -150), 14), ('modern', 'facades_modern_close', (0, 0), 16),
                                 ('kaufhaus', 'facades_kaufhaus_close', (0, 0), 20), ('fachwerk', 'facades_fachwerk_close', (-330, 150), 12),
                                 ('nachkrieg', 'facades_nachkrieg_close', (0, -60), 14), ('gothic', 'facades_marktkirche', (-200, 300), 60)):
        b = pick(kind, near, 8 if kind == 'fachwerk' else 14)
        if b is None:
            continue
        cm = facade_cam(b, dist=dist, look_h=8 if kind != 'gothic' else 25)
        if cm is None:
            continue
        cm.update({'lens': 28, 'radius': 160, 'id': b.id, 'street': b.street, 'number': b.number})
        cams[nm] = cm
    # Detailstufen-Nahaufnahme (Fenster) am Altbau
    if 'facades_altbau_close' in cams:
        b = [x for x in bl if x.id == cams['facades_altbau_close']['id']][0]
        cm = facade_cam(b, dist=4.0, eye=6.0, look_h=6.8, side=1.5)
        cm.update({'lens': 35, 'radius': 60, 'id': b.id})
        cams['facades_window_detail'] = cm
    # Kirche: Kamera naeher heran, nicht in Gebaeuden
    vol = CTX['vol']
    for k, c in list(cams.items()):
        loc = np.array(c['loc'], float); tgt = np.array(c['tgt'], float)
        for _ in range(30):
            bo, bs = vol.blocked(loc[None], -1)
            if not (bo | bs).any():
                break
            loc = loc + (tgt - loc) * 0.08
            loc[1] = c['loc'][1]
        c['loc'] = [float(x) for x in loc]
    # Dach-Schraegansicht und Ecke (Pruefung Luecken/Attika/Traufe)
    cams['facades_roofs_oblique'] = {'loc': [-95.0, 48.0, -95.0], 'tgt': [-40.0, 20.0, -40.0], 'lens': 30, 'radius': 160}
    b = pick('fachwerk', (-330, 150), 8)
    if b is not None:
        cm = facade_cam(b, dist=9.0, eye=1.7, look_h=b.y_top - b.y_ground - 4, side=-3.0)
        if cm is not None:
            cm.update({'lens': 24, 'radius': 80, 'id': b.id})
            cams['facades_eaves_close'] = cm
    save_json(cams, os.path.join(CACHE, 'preview_cams.json'), indent=1)
    # Gelaende
    T = PREP.Terrain()
    xs = np.arange(-600, 601, 2.0); zs = np.arange(-600, 601, 2.0)
    X, Z = np.meshgrid(xs, zs)
    Y = T.sample(X.ravel(), Z.ravel()).reshape(X.shape)
    Y = np.where(np.isfinite(Y), Y, np.nanmean(Y))
    np.savez(os.path.join(CACHE, 'terrain_2m.npz'), x=xs, z=zs, y=Y.astype(np.float32))
    print(json.dumps({k: (v.get('street'), v.get('number')) for k, v in cams.items()}, ensure_ascii=False))


if __name__ == '__main__' and '--cams2' not in sys.argv:
    main()


# ---------------------------------------------------------------- Vorschau 2: echte Laeden + Altstadt (facades2_*)
EXTRA_DISPLAYS = ()


def cams2():
    import shops as SH, replaced
    d = PREP.load_prep(); replaced.mark(d['bldgs'], verbose=False)
    T = PREP.Terrain()
    ok = {b.id for b in d['bldgs'] if not b.covered and not getattr(b, 'replaced', None)}
    by = {b.id: b for b in d['bldgs']}
    D = SH.db()
    info = load_json(os.path.join(CACHE, 'building_info.json')) if os.path.exists(os.path.join(CACHE, 'building_info.json')) else []
    placed = {}
    for it in info:
        for sp in it.get('shops', []):
            placed[sp['id']] = sp


    def ty(x, z, add=1.7):
        y = T.sample([x], [z])[0]
        return float(y if np.isfinite(y) else 0.0) + add

    def shop_cam(near, dist=6.0, along=7.0, look_along=-5.0, h=3.0, lens=24):
        best = None
        for bid, L in D['by_bldg'].items():
            if bid not in ok:
                continue
            for s in L:
                if s['sign'].get('type') in (None, 'none') or (placed and s['id'] not in placed):
                    continue
                wp = np.array(s['facade']['wall_point'])
                dd = np.hypot(*(wp - near))
                if best is None or dd < best[0]:
                    best = (dd, s)
        s = best[1]
        wp = np.array(s['facade']['wall_point']); n = np.array(s['facade']['normal']); t = np.array([n[1], -n[0]])
        c = wp + n * dist + t * along; g = wp + t * look_along
        return {'loc': [float(c[0]), ty(c[0], c[1]), float(c[1])], 'tgt': [float(g[0]), ty(g[0], g[1], h), float(g[1])], 'lens': lens,
                'radius': 90, 'shop': s['name']}

    def street_cam(loc, tgt, h=6.0, lens=24, radius=110):
        return {'loc': [loc[0], ty(*loc), loc[1]], 'tgt': [tgt[0], ty(*tgt, h), tgt[1]], 'lens': lens, 'radius': radius}
    cams = {}
    cams['facades2_bahnhofstrasse_shops'] = shop_cam(np.array([40.0, -75.0]), dist=5.5, along=8.0, look_along=-4.0)
    cams['facades2_georgstrasse_shops'] = shop_cam(np.array([-150.0, -40.0]), dist=6.0, along=8.0, look_along=-4.0)
    cams['facades2_kramerstrasse'] = street_cam((-352.0, 303.5), (-405.0, 314.0), h=6.5)
    fb = [b for b in d['bldgs'] if b.street == 'Burgstraße' and b.id in ok]
    if fb:
        P2 = np.array([[b.cx, b.cz] for b in fb]); c = P2.mean(0)
        cams['facades2_burgstrasse'] = street_cam((float(c[0]) + 6.0, float(c[1]) + 22.0), (float(c[0]) - 2.0, float(c[1]) - 20.0), h=6.0)
    cams['facades2_holzmarkt'] = street_cam((-424.0, 342.0), (-445.0, 314.0), h=7.0)
    cams['facades2_steintor'] = street_cam((-492.0, -168.0), (-405.0, -116.0), h=8.0, radius=140)
    cams['facades2_steintor_shops'] = shop_cam(np.array([-385.0, -135.0]), dist=6.0, along=8.0, look_along=-4.0)
    cams['facades2_holzmarkt_fachwerk'] = street_cam((-420.0, 318.0), (-446.0, 330.0), h=6.0, radius=90)
    # Nahaufnahmen von Schaufenstern (Auslagen je Sortiment) an benannten Laeden
    def named_cam(name, addr_part, dist=3.6, along=1.2, lens=28):
        # bevorzugt die tatsaechlich gebaute Schaufensterachse (building_info.json), sonst der Wandpunkt aus shops.json
        for bid, L in D['by_bldg'].items():
            if bid not in ok:
                continue
            for s in L:
                if s.get('name') == name and addr_part in (s.get('address') or '') and s['id'] in placed:
                    sp = placed[s['id']]
                    p = np.array([sp['p'][0], sp['p'][2]]); n = np.array(sp['n']); t = np.array([n[1], -n[0]])
                    c = p + n * dist + t * along
                    return {'loc': [float(c[0]), sp['p'][1] + 1.6, float(c[1])], 'tgt': [float(p[0]), sp['p'][1] + 1.5, float(p[1])],
                            'lens': lens, 'radius': 60, 'shop': name}
        for bid, L in D['by_bldg'].items():
            if bid not in ok:
                continue
            for s in L:
                if s.get('name') == name and addr_part in (s.get('address') or ''):
                    wp = np.array(s['facade']['wall_point']); n = np.array(s['facade']['normal']); t = np.array([n[1], -n[0]])
                    c = wp + n * dist + t * along
                    return {'loc': [float(c[0]), ty(c[0], c[1], 1.6), float(c[1])], 'tgt': [float(wp[0]), ty(wp[0], wp[1], 1.9), float(wp[1])],
                            'lens': lens, 'radius': 60, 'shop': name}
        return None
    for key, nm, ad in (('facades2_display_fashion', 'VAGANT', 'Georgstra'), ('facades2_display_jewellery', 'Traut Euch', 'Kramerstra'),
                        ('facades2_display_books', 'Thalia', 'Marktstra'), ('facades2_display_shoes', 'Salamander', 'Packhofstra'),
                        ('facades2_display_optician', 'eyes + more', 'Packhofstra')) + EXTRA_DISPLAYS:
        c = named_cam(nm, ad)
        if c is not None:
            cams[key] = c
    # weitere Sortimente: je Art der gebaute Laden mit Namen, der dem Kroepcke am naechsten liegt
    for kd in ('pharmacy', 'bakery', 'cafe', 'phone', 'drugstore', 'bank', 'department'):
        best = None
        for sid, sp in placed.items():
            if sp['kind'] != kd or not sp.get('name') or sp.get('sign') in (None, 'none'):
                continue
            dd = math.hypot(sp['p'][0], sp['p'][2])
            if best is None or dd < best[0]:
                best = (dd, sp)
        if best is not None:
            sp = best[1]
            p = np.array([sp['p'][0], sp['p'][2]]); n = np.array(sp['n']); t = np.array([n[1], -n[0]])
            c = p + n * 4.2 + t * 1.4
            cams['facades2_display_' + kd] = {'loc': [float(c[0]), sp['p'][1] + 1.6, float(c[1])],
                                              'tgt': [float(p[0]), sp['p'][1] + 1.7, float(p[1])], 'lens': 26, 'radius': 60, 'shop': sp['name']}
    vol = CTX['vol'] if CTX else None
    save_json(cams, os.path.join(CACHE, 'preview_cams2.json'), indent=1)
    print(json.dumps(cams, ensure_ascii=False, indent=0)[:1500])


if __name__ == '__main__' and '--cams2' in sys.argv:
    cams2()

"""Kamera-Listen fuer preview2.py:  python shots.py <gruppe> <out.json>
Gruppen: kroepcke_<ref>, kroepcke_all, passerelle, steintor, altstadt, entrance:<key>"""
import sys, json, pickle
import numpy as np
from common import *

def entrance_shots(info, prefix='ground2_ubahn_'):
    p = np.array(info['top']); d = np.array(info['dir']); r = np.array([-d[1], d[0]]); y = info['y_top']
    nm = prefix + (info['ref'] if info['station'] == 'Kröpcke' else info['key'])
    out = []
    a = p - d * 5.5 + r * 3.5
    out.append({'name': nm + '_top', 'loc': [a[0], y + 4.2, a[1]], 'tgt': [(p + d * 6)[0], y - 2.0, (p + d * 6)[1]], 'lens': 22, 'r': 120})
    b = p - d * 0.6 + r * 0.25
    out.append({'name': nm + '_down', 'loc': [b[0], y + 1.65, b[1]], 'tgt': [(p + d * 9)[0], y - 3.6, (p + d * 9)[1]], 'lens': 18, 'r': 120})
    if 'hall_end' in info:
        h = np.array(info['hall_end']); c = p + d * (info['length_open'] - 0.5)
        out.append({'name': nm + '_hall', 'loc': [c[0], info['y_floor'] + 1.65, c[1]], 'tgt': [h[0], info['y_floor'] + 1.2, h[1]], 'lens': 18, 'r': 120})
    return out

if __name__ == '__main__':
    grp = sys.argv[1]; outf = sys.argv[2]
    S = []
    UB = pickle.load(open(CACHE + 'ubahn.pkl', 'rb'))['info'] if os.path.exists(CACHE + 'ubahn.pkl') else []
    if grp.startswith('kroepcke_'):
        refs = grp.split('_', 1)[1]
        for i in UB:
            if i.get('station') == 'Kröpcke' and (refs == 'all' or i['ref'] in refs): S += entrance_shots(i)
        if 'L' in refs or refs == 'all':
            S.append({'name': 'ground2_lift_80A', 'loc': [-35.5, 1.8, 25.5], 'tgt': [-41.7, 1.6, 30.85], 'lens': 24, 'r': 100})
    elif grp.startswith('entrance:'):
        for i in UB:
            if i['key'] == grp.split(':', 1)[1]: S += entrance_shots(i)
    elif grp == 'steintor_ub':
        for i in UB:
            if i.get('station') == 'Steintor': S += entrance_shots(i)[:2]
    elif grp in ('passerelle', 'steintor', 'altstadt', 'ausbau2'):
        from surface import Surface
        SU = Surface()
        def gy(x, z): return float(SU([x], [z])[0][0])
        def shot(name, loc, tgt, lens=20, r=110, dy_loc=1.65, dy_tgt=1.2, y_loc=None, y_tgt=None):
            yl = gy(*loc) + dy_loc if y_loc is None else y_loc; yt = gy(*tgt) + dy_tgt if y_tgt is None else y_tgt
            S.append({'name': name, 'loc': [loc[0], yl, loc[1]], 'tgt': [tgt[0], yt, tgt[1]], 'lens': lens, 'r': r})
        if grp in ('passerelle', 'ausbau2'):
            PR = json.load(open(CACHE + 'promenade.json')); A = PR['axis']
            mu = np.array(A['origin']); ax = np.array(A['axis']); nr = np.array(A['normal'])
            ops = sorted(PR['openings'], key=lambda o: o['s'][0])
            sc = np.array([(o['s'][0] + o['s'][1]) / 2 for o in ops]); fl = np.array([o['floor'] for o in ops])
            W = lambda s_, t_: mu + ax * s_ + nr * t_
            yf = lambda s_: float(np.interp(s_, sc, fl))
            shot('ground2_passerelle_open', W(-56, -2.0), W(-18, 0.5), 18, 120, y_loc=yf(-56) + 1.65, y_tgt=yf(-18) + 1.8)
            shot('ground2_passerelle_bridge', W(-29, 1.2), W(-12.7, -0.5), 18, 120, y_loc=yf(-29) + 1.65, y_tgt=yf(-12.7) + 2.9)
            shot('ground2_passerelle_bridge_top', W(-12.7, -9.0), W(-12.7, 6.0), 20, 120, dy_tgt=0.6)
            shot('ground2_passerelle_bridge_air', W(-30, -0.4), W(-13, -0.4), 20, 120, y_loc=yf(-30) + 11.0, y_tgt=yf(-13) + 1.0)
            shot('ground2_passerelle_covered', W(61, -2.2), W(100, 1.0), 18, 120, y_loc=yf(61) + 1.65, y_tgt=yf(100) + 1.6)
            shot('ground2_passerelle_shops', W(75, 0.0), W(82, -4.0), 22, 100, y_loc=yf(75) + 1.65, y_tgt=yf(82) + 1.8)
            shot('ground2_passerelle_south', W(-80, 2.0), W(-108, 0.0), 18, 120, y_loc=yf(-80) + 1.65, y_tgt=yf(-108) + 1.4)
            shot('ground2_passerelle_north', W(108, 2.0), W(140, 0.0), 18, 120, y_loc=yf(108) + 1.65, y_tgt=yf(130) + 2.5)
        if grp in ('steintor', 'ausbau2'):
            shot('ground2_steintor_stop', (-516.0, -126.0), (-493.0, -117.0), 20, 110)
            shot('ground2_steintor_platform', (-503.5, -103.0), (-487.0, -137.0), 18, 110, dy_loc=0.86 + 1.65, dy_tgt=2.0)
            shot('ground2_steintor_mendini', (-383.0, -250.0), (-412.0, -252.0), 20, 110, dy_tgt=2.0)
        if grp in ('altstadt', 'ausbau2'):
            shot('ground2_holzmarkt', (-421.0, 350.5), (-432.0, 331.5), 22, 110, dy_tgt=3.5)
            shot('ground2_holzmarkt_close', (-424.0, 324.0), (-431.6, 332.2), 18, 90, dy_tgt=3.0)
            shot('ground2_marktplatz', (-250.0, 334.0), (-205.0, 314.0), 20, 130, dy_tgt=4.0)
            shot('ground2_marktplatz_high', (-250.0, 350.0), (-215.0, 320.0), 24, 130, dy_loc=22.0, dy_tgt=0.0)
            shot('ground2_ballhofplatz', (-445.0, 232.0), (-410.0, 213.0), 20, 110, dy_tgt=1.5)
            from shapely.geometry import Point as _Pt
            LA = pickle.load(open(CACHE + 'layout.pkl', 'rb')); RV = LA['classes'].get('river')
            if RV is not None and not RV.is_empty:
                # Kamera auf dem Ostufer (Am Hohen Ufer) bzw. Westufer (Leibnizufer), Blick flussabwaerts (NW) auf Wasser + Kaimauer
                from shapely.geometry import LineString as _LS
                for nm_, zc, side in (('ground2_leine_hohes_ufer', 250.0, 1), ('ground2_leine_leibnizufer', 180.0, -1), ('ground2_leine_schlossbruecke', 330.0, 1)):
                    cut = RV.intersection(_LS([(-900, zc), (-200, zc)]))
                    xs = sorted(c[0] for g in getattr(cut, 'geoms', [cut]) for c in g.coords)
                    if len(xs) < 2: continue
                    xw, xe = xs[0], xs[-1]; xm = (xw + xe) / 2
                    xl = xe + 3.0 if side > 0 else xw - 3.0
                    shot(nm_, (xl, zc), (xm - 30.0, zc - 45.0), 20, 130, y_tgt=-6.5)
                # Tunnelportal (Durchlass Friederikenplatz)
                try:
                    POR = pickle.load(open(CACHE + 'altstadt.pkl', 'rb'))['info_extra'].get('leine_portals') or []
                except Exception: POR = []
                if POR:
                    pa_ = max(POR, key=lambda q: np.hypot(*(np.array(q['b']) - q['a'])))
                    a_, b_ = np.array(pa_['a']), np.array(pa_['b']); c_ = (a_ + b_) / 2; d_ = (b_ - a_) / np.linalg.norm(b_ - a_); n_ = np.array([-d_[1], d_[0]])
                    if not RV.contains(_Pt(c_ + n_ * 1.0)): n_ = -n_
                    shot('ground2_leine_portal', c_ + n_ * 24 + d_ * 6, c_, 20, 130, y_loc=pa_['y_water'] + 5.0, y_tgt=pa_['y_water'] + 1.2)
    json.dump(S, open(outf, 'w'), indent=1)
    print(len(S), 'shots ->', outf)

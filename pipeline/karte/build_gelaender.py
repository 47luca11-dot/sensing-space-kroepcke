# Handläufe der Passerelle-Treppen fürs Spiel (Leuchtgeländer zum Runterrutschen).
#   uv run --no-project --python 3.11 --with numpy,shapely python pipeline/karte/build_gelaender.py
# Liest data/ground_cache/promenade.pkl (info_extra.stairs, Achse) -> web/assets/city/karte/gelaender.json
import pickle, json, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
d = pickle.load(open(os.path.join(ROOT, 'data/ground_cache/promenade.pkl'), 'rb'))['info_extra']
AX = d['axis']; o, a, n = AX['origin'], AX['axis'], AX['normal']
out = []
for st in d['stairs']:
    sg = 1 if st['descends'] == 'north' else -1
    s_top = st['s'][0] if sg > 0 else st['s'][1]
    run = st['steps'] * st['tread'] + st['landing']
    s_bot = s_top + sg * run
    for side in (0, 1):
        t = st['t'][side] + (0.08 if side == 0 else -0.08)
        P = lambda s: [round(o[0] + a[0] * s + n[0] * t, 3), round(o[1] + a[1] * s + n[1] * t, 3)]
        out.append({'treppe': st['name'], 'seite': side, 'oben': P(s_top - sg * 0.3), 'unten': P(s_bot + sg * 0.3),
                    'stufen': st['steps'], 'steigung': st['rise'], 'hoehe': round(st['steps'] * st['rise'], 2)})
json.dump({'quelle': 'pipeline/karte/build_gelaender.py', 'handlauf_hoehe': 0.95, 'gelaender': out},
          open(os.path.join(ROOT, 'web/assets/city/karte/gelaender.json'), 'w'), indent=1)
print(len(out), 'Handläufe')

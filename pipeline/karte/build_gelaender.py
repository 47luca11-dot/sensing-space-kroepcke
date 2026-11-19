# Handläufe der Passerelle-Treppen fürs Spiel (Leuchtgeländer zum Runterrutschen).
#   uv run --no-project --python 3.11 --with numpy,shapely python pipeline/karte/build_gelaender.py
# Liest data/ground_cache/promenade.pkl (info_extra.handlaeufe = exakt die Polylinien der Edelstahl-Handläufe, die promenade.py
# als Rohr baut: 1,0 m über der Stufen-Vorderkantenlinie, inkl. Zwischenpodest) -> web/assets/city/karte/gelaender.json
# Das Leuchtgeländer liegt damit genau auf dem echten Handlauf (früher aus Treppen-Eckdaten + Boden-Höhe geschätzt).
import pickle, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
d = pickle.load(open(os.path.join(ROOT, 'data/ground_cache/promenade.pkl'), 'rb'))['info_extra']
out = []
for h in d['handlaeufe']:
    P = h['pts']
    if P[0][1] < P[-1][1]: P = P[::-1]                                   # immer von oben nach unten
    out.append({'treppe': h['treppe'], 'seite': h['seite'], 'pts': [[round(v, 3) for v in p] for p in P],
                'hoehe': round(P[0][1] - P[-1][1], 2)})
json.dump({'quelle': 'pipeline/karte/build_gelaender.py (promenade.py handlaeufe)', 'rohr_radius': 0.024, 'gelaender': out},
          open(os.path.join(ROOT, 'web/assets/city/karte/gelaender.json'), 'w'), indent=1)
print(len(out), 'Handläufe')

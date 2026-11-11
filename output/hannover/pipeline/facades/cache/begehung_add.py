"""Erzeugt den Abschnitt 'Ergänzungen aus dem Bau – Fassaden/Läden' fuer BEGEHUNGSLISTE.md (wird angehaengt)."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prep as PREP, shops as SH, replaced
from common import ROOT, CACHE, load_json

d = PREP.load_prep(); replaced.mark(d['bldgs'], verbose=False)
by = {b.id: b for b in d['bldgs']}
info = {i['id']: i for i in load_json(os.path.join(CACHE, 'building_info.json'))}
placed = set(load_json(os.path.join(ROOT, 'web', 'assets', 'city', 'buildings', 'index.json')).get('shops_placed_ids', []))
S = load_json(os.path.join(ROOT, 'data', 'shops.json'))['shops']
miss = []
for s in S:
    if s.get('level') != 0 or s.get('context') != 'street' or not s.get('facade'):
        continue
    bid = s['facade'].get('building_id') or s.get('building_lod2_id')
    b = by.get(bid)
    if b is None or b.covered or getattr(b, 'replaced', None):
        continue
    if s['id'] not in placed:
        miss.append(s)
fw = collections.defaultdict(list)
for i in info.values():
    if i['kind'] == 'fachwerk':
        fw[i['street'] or '?'].append(i['number'] or '?')
L = []
A = L.append
A('')
A('## Ergänzungen aus dem Bau')
A('')
A('### Fassaden- und Ladengenerator (pipeline/facades, Stand 2026-11-09)')
A('')
A('Diese Stellen wurden ohne ausreichende Bildvorlage modelliert (Annahmen aus OSM, LoD2, Street-View-Textnotizen 2022/23). '
  'Bitte fotografieren (frontal + schräg, Schilder lesbar, Schaufenster, Sockel/Eingangsstufen):')
A('')
A('| ✓ | Prio | Ort | Was fehlt | x | z |')
A('|---|---|---|---|---|---|')
rows = [
    ('P1', 'Burgstraße, Fachwerkzeile (alle Häuser)', 'Fachwerkmuster je Haus (Streben, Andreaskreuze, Fächerrosetten), Balkenfarbe (rot/braun/grau), Gefachfarbe, Auskragungstiefe je Geschoss, Ladenfronten. Generator rät Muster + Farben.', -470, 215),
    ('P1', 'Kramerstraße Nr. 3–25', 'Welche Häuser echtes Fachwerk / welche Putz-Wiederaufbau; Balkenköpfe, Knaggen, Schnitzereien; Holz-Ladenfronten (blau/braun) je Laden.', -380, 308),
    ('P1', 'Holzmarkt 1–6 (ohne Leibnizhaus)', 'Fachwerkfassaden zum Platz, Gauben, Erdgeschoss-Sandstein.', -432, 330),
    ('P1', 'Ballhofplatz / Ballhofstraße / Kreuzkirchhof (Wohnhäuser)', 'Fachwerk- und Backsteinhäuser rund um Ballhof und Kreuzkirche; aktuell geschätzt.', -425, 200),
    ('P2', 'Kreuzkirche (Kreuzkirchhof)', 'Backsteingotik: Strebepfeiler, Staffelgiebel, Blendnischen, Turm mit Barockhaube – derzeit generische Kirchenfassade aus LoD2.', -410, 160),
    ('P2', 'Historisches Museum (Burgstraße/Pferdestraße)', '60er-Jahre-Bau mit Bruchstein-Bändern und verglastem EG auf Rundstützen – derzeit generische Bürofassade.', -470, 300),
    ('P2', 'Knochenhauerstraße und Schmiedestraße, beide Seiten', 'Ladenschilder, Fassadenfarben, Erdgeschosshöhen; Schilder bisher nur aus OSM/Markenfarben.', -340, 150),
    ('P2', 'Steintor: Georgstraße 1–4, Steintorstraße, Lange Laube, Münzstraße', 'Ladenschilder (Street View 2022 lückenhaft), Pylon Georgstr. 1–3, EG-Höhen.', -420, -150),
    ('P2', 'Schaufenster der Ketten (H&M, C&A, Douglas, Rossmann, dm, Thalia, Hugendubel, P&C, Apple)', 'Echte Auslagen/Dekoration, Fascia-Höhen und -Materialien; Generator nutzt typische Auslagen je Sortiment.', 0, 0),
    ('P2', 'Unverifizierte Läden in data/shops.json (%d von %d ohne Beleg ab 2025, v. a. Altstadt/Steintor/Seitenstraßen)' % (
        sum(1 for s in S if not s.get('verified')), len(S)),
     'Ob der Laden noch existiert, Schriftzug/Schildart/Farben; der Generator setzt den Namen in Markenfarben-Schätzung auf die Fassade.', 0, 0),
    ('P2', 'Gastronomie mit Außenbestuhlung (Kramerstraße, Burgstraße, Knochenhauerstraße, Holzmarkt)',
     'Anzahl/Art der Tische, Schirme, Markisen – Generator stellt Tische nur bei OSM outdoor_seating=yes auf.', -380, 300),
    ('P3', 'Hausnummern-Schilder', 'Farbe/Form der Hannoveraner Hausnummernschilder (angenommen: blau emailliert, weiße Ziffern).', 0, 0),
    ('P3', 'Straßen mit Gefälle (Laden-EG über LoD2-Grundfläche)', 'Sockelhöhen/Eingangsstufen an Läden mit Höhenversatz (z. B. Georgstraße 5, Marktstraße); Generator hebt die Ladenebene auf DGM-Höhe.', -390, -175),
]
for p, o, w, x, z in rows:
    A('| ☐ | %s | %s | %s | %s | %s |' % (p, o, w, x if (x or z) else '–', z if (x or z) else '–'))
A('')
A('Fachwerk wurde für diese Häuser generiert (Straße: Hausnummern) – bitte vor Ort bestätigen bzw. korrigieren:')
A('')
for st in sorted(fw):
    A('- %s: %s' % (st, ', '.join(sorted(set(fw[st]), key=lambda v: (len(v), v)))))
A('')
if miss:
    A('Läden aus data/shops.json, die keiner Fassadenwand zugeordnet werden konnten (Wand nicht gefunden, zu schmal oder unter Arkade) – Lage und Schild vor Ort prüfen:')
    A('')
    A('| ✓ | Name | Adresse | Art | x | z |')
    A('|---|---|---|---|---|---|')
    for s in sorted(miss, key=lambda s: (s.get('address') or '', s['name'] or '')):
        A('| ☐ | %s | %s | %s | %d | %d |' % (s['name'] or '–', s.get('address') or '–', s.get('category') or '–', round(s['p'][0]), round(s['p'][1])))
    A('')
txt = '\n'.join(L) + '\n'
p = os.path.join(ROOT, 'BEGEHUNGSLISTE.md')
cur = open(p, encoding='utf-8').read()
if '### Fassaden- und Ladengenerator (pipeline/facades' in cur:
    print('Abschnitt schon vorhanden - nicht erneut angehaengt'); sys.exit(0)
if '## Ergänzungen aus dem Bau' in cur:
    txt = txt.replace('\n## Ergänzungen aus dem Bau\n', '\n', 1)
with open(p, 'a', encoding='utf-8') as f:
    f.write(txt)
print('angehaengt: %d Zeilen, %d nicht platzierte Laeden' % (len(L), len(miss)))

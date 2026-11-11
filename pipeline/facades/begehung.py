"""Schreibt den Abschnitt 'Fassaden- und Ladengenerator' unter '## Ergänzungen aus dem Bau' in BEGEHUNGSLISTE.md.

Ist der Abschnitt schon vorhanden, wird nur dieser Abschnitt ersetzt (bis zur naechsten Ueberschrift / Markierung eines
anderen Bausteins); sonst wird er angehaengt. Schreiben atomar (temporaere Datei + Umbenennen).
Aufruf: python begehung.py   (nach build_all.py; liest cache/building_info.json und web/.../index.json)"""
import os, sys, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep as PREP, replaced
from common import ROOT, CACHE, load_json

HEAD = '### Fassaden- und Ladengenerator (pipeline/facades'


def section():
    d = PREP.load_prep(); replaced.mark(d['bldgs'], verbose=False)
    by = {b.id: b for b in d['bldgs']}
    info = load_json(os.path.join(CACHE, 'building_info.json'))
    placed = set(load_json(os.path.join(ROOT, 'web', 'assets', 'city', 'buildings', 'index.json')).get('shops_placed_ids', []))
    S = load_json(os.path.join(ROOT, 'data', 'shops.json'))['shops']
    miss = []
    for s in S:
        if s.get('level') != 0 or s.get('context') != 'street' or not s.get('facade'):
            continue
        b = by.get(s['facade'].get('building_id') or s.get('building_lod2_id'))
        if b is None or b.covered or getattr(b, 'replaced', None):
            continue
        if s['id'] not in placed:
            miss.append(s)
    fw = collections.defaultdict(list)
    iron = []
    for i in info:
        if i['kind'] == 'fachwerk':
            fw[i['street'] or '?'].append(i['number'] or '?')
    L = []
    A = L.append
    A(HEAD + ', Stand 2026-11-10)')
    A('')
    A('Diese Stellen wurden ohne ausreichende Bildvorlage modelliert (Annahmen aus OSM, LoD2, Street-View-Textnotizen 2022/23, '
      'Wikimedia-Referenzfotos). Bitte fotografieren (frontal + schräg, Schilder lesbar, Schaufenster, Sockel/Eingangsstufen):')
    A('')
    A('| ✓ | Prio | Ort | Was fehlt | x | z |')
    A('|---|---|---|---|---|---|')
    rows = [
        ('P1', 'Burgstraße, Fachwerkzeile (alle Häuser)', 'Fachwerkmuster je Haus (Fensterband oder Einzelfenster, Streben, Andreaskreuze, '
         'Fächerrosetten), Balkenfarbe (dunkelbraun / ochsenblutrot / grau), Gefachfarbe, Auskragungstiefe je Geschoss, Ladenfronten. '
         'Generator wählt Muster und Farben je Haus nach Wahrscheinlichkeit.', -470, 215),
        ('P1', 'Kramerstraße Nr. 3–25', 'Welche Häuser echtes Fachwerk / welche Putz-Wiederaufbau; Balkenköpfe, Knaggen, Schnitzereien, '
         'Inschriften auf Schwellen; Holz-Ladenfronten (blau/braun) je Laden.', -380, 308),
        ('P1', 'Holzmarkt 1–6 (ohne Leibnizhaus)', 'Fachwerkfassaden zum Platz, Gauben, Erdgeschoss-Sandstein.', -432, 330),
        ('P1', 'Ballhofplatz / Ballhofstraße / Kreuzkirchhof (Wohnhäuser)', 'Fachwerk- und Backsteinhäuser rund um Ballhof und Kreuzkirche; '
         'aktuell geschätzt (Ballhof selbst kommt als handgebautes Wahrzeichen, sobald data/landmarks_replace/ ihn enthält).', -425, 200),
        ('P1', 'Altstadt-Ausleger (schmiedeeiserne Schilder)', 'Generator hängt an gut der Hälfte der Altstadt-Läden mit Holzfront einen '
         'schmiedeeisernen Ausleger mit Namenstafel (Arm ~1 m, Tafel ~0,8 × 0,5 m, ~0,5 m über der EG-Decke). Welche Läden haben wirklich '
         'Ausleger, Form, Farbe, Höhe?', -380, 308),
        ('P2', 'Kreuzkirche (Kreuzkirchhof)', 'Jetzt grau-brauner Bruchstein-Sandstein mit Strebepfeilern und Spitzbogenfenstern aus LoD2 '
         '(kein Staffelgiebel). Maßwerk, Portale (Duve-Kapelle), Turmhelm (Kupfer) und Uhr fehlen – Fotos aller Seiten.', -422, 117),
        ('P2', 'Apple Store Bahnhofstraße 5', 'Modell: ganze Fassade dunkelgrauer Naturstein mit Steinfaschen und Gesimsbändern, EG glatter '
         'Granit, kleine weiße Apfel-Silhouette (kein Logo-Nachbau) auf dunkler Steinplatte, innen Videowand + Holztische. Steinfarbe der '
         'Obergeschosse, Gesimse und Plakette fotografieren.', 28, -65),
        ('P1', 'Kramerstraße 15, Copyshop „Carl Küster“', 'OSM schreibt „Cark Küster“ (vermutlich Tippfehler; in Hannover gibt es die '
         'Druckerei Carl Küster). Modell zeigt „Carl Küster“ auf Blende und Ausleger – Schriftzug, Schildart und ob der Laden dort ist prüfen.',
         -385, 317),
        ('P1', 'Kröpcke Nord: H&M „Runde Ecke“ (Georgstraße 31–33) und Hugendubel (Bahnhofstraße 14)', 'Modell nach Street-View-Textnotiz: '
         'H&M mit Fensterbändern und hellen Brüstungsbändern, verglaster Rundturm, rote Einzelbuchstaben „H&M“ (~1,7 m) auf der Rundung im '
         '3. OG; Hugendubel mit 2 m tiefem, geneigtem Glasvordach über dem EG, weißen Buchstaben vor dem 1. OG und roter Tafel „Welt der '
         'Bücher“. Höhe/Größe der Buchstaben, Vordachhöhe und -neigung, Treppenturm-Verglasung fotografieren.', -25, -35),
        ('P2', 'GALERIA Ernst-August-Platz 5 / Bahnhofstraße', 'Große grüne Versalien „GALERIA“ (~2,2 m) im 3. OG der Bahnhofstraßenseite '
         'nach Notiz 2022; Lage, Größe und ob „KAUFHOF“ noch dort steht, prüfen. EG teils mit Platten verschlossen (2022).', 40, -125),
        ('P2', 'Georgstraße 23 (ehem. Karstadt, leer)', 'Modell: helle Natursteinfassade, durchgehende leere/beklebte Schaufensterzone mit '
         'Eingängen, verglaster Eckturm, Relief-Emblem. Aktueller Zustand (Folien, Bauzaun, Zwischennutzung?) fotografieren.', -150, -45),
        ('P3', 'Fachwerk-Eckfasen (z. B. Holzmarkt 3, Kramerstraße 12)', 'Schmale Fasen an Hausecken bekommen jetzt einen durchgehenden '
         'Eckständer und kragen mit aus; ob dort wirklich ein Eckständer oder Putz sitzt, prüfen.', -416, 323),
        ('P2', 'Knochenhauerstraße und Schmiedestraße, beide Seiten', 'Ladenschilder, Fassadenfarben, Erdgeschosshöhen; Schilder bisher nur '
         'aus OSM/Markenfarben.', -340, 150),
        ('P2', 'Steintor: Georgstraße 1–4, Steintorstraße, Lange Laube, Münzstraße', 'Ladenschilder (Street View 2022 lückenhaft), Pylon '
         'Georgstr. 1–3, EG-Höhen.', -420, -150),
        ('P2', 'Schaufenster der Ketten (H&M, C&A, Douglas, Rossmann, dm, Thalia, Hugendubel, P&C, Apple)', 'Echte Auslagen/Dekoration, '
         'Fascia-Höhen und -Materialien; Generator nutzt typische Auslagen je Sortiment.', 0, 0),
        ('P2', 'Unverifizierte Läden in data/shops.json (%d von %d ohne Beleg ab 2025, v. a. Altstadt/Steintor/Seitenstraßen)' % (
            sum(1 for s in S if not s.get('verified')), len(S)),
         'Ob der Laden noch existiert, Schriftzug/Schildart/Farben; der Generator setzt den Namen in Markenfarben-Schätzung auf die Fassade.',
         0, 0),
        ('P2', 'Gastronomie mit Außenbestuhlung (Kramerstraße, Burgstraße, Knochenhauerstraße, Holzmarkt)', 'Anzahl/Art der Tische, Schirme, '
         'Markisen – Generator stellt Tische nur bei OSM outdoor_seating=yes auf.', -380, 300),
        ('P3', 'Hausnummern-Schilder', 'Farbe/Form der Hannoveraner Hausnummernschilder (angenommen: blau emailliert, weiße Ziffern).', 0, 0),
        ('P3', 'Straßen mit Gefälle (Laden-EG über LoD2-Grundfläche)', 'Sockelhöhen/Eingangsstufen an Läden mit Höhenversatz (z. B. '
         'Georgstraße 5, Marktstraße); Generator hebt die Ladenebene auf DGM-Höhe.', -390, -175),
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
        A('Läden aus data/shops.json, die keiner Fassadenwand zugeordnet werden konnten (Wand nicht gefunden, zu schmal oder unter Arkade) '
          '– Lage und Schild vor Ort prüfen:')
        A('')
        A('| ✓ | Name | Adresse | Art | x | z |')
        A('|---|---|---|---|---|---|')
        for s in sorted(miss, key=lambda s: (s.get('address') or '', s['name'] or '')):
            A('| ☐ | %s | %s | %s | %d | %d |' % (s['name'] or '–', s.get('address') or '–', s.get('category') or '–', round(s['p'][0]),
                                                 round(s['p'][1])))
        A('')
    return '\n'.join(L) + '\n', len(miss)


def main():
    txt, nmiss = section()
    p = os.path.join(ROOT, 'BEGEHUNGSLISTE.md')
    cur = open(p, encoding='utf-8').read()
    i = cur.find(HEAD)
    if i >= 0:
        m = re.compile(r'\n(?:#{2,3} |<!--)').search(cur, i + len(HEAD))
        j = m.start() + 1 if m else len(cur)
        new = cur[:i] + txt + cur[j:]
    else:
        add = txt if '## Ergänzungen aus dem Bau' in cur else '\n## Ergänzungen aus dem Bau\n\n' + txt
        new = cur.rstrip('\n') + '\n\n' + add
    tmp = p + '.tmp_facades'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new)
    os.replace(tmp, p)
    print('BEGEHUNGSLISTE.md: Abschnitt Fassaden aktualisiert, %d nicht platzierte Laeden' % nmiss)


if __name__ == '__main__':
    main()

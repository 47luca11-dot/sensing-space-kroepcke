# Treppen in der Niki-de-Saint-Phalle-Promenade (Passerelle): Korrektur 18.11.2026

Generator: `pipeline/ground/promenade.py`. Spielkoordinaten wie üblich (m, Y oben, Ursprung Kröpcke-Uhr, +X Ost, −Z Nord).
Promenadenrahmen im Code: s = entlang der Achse (Kröpcke → Hbf, nach NNO), t = quer (+t = Ostseite).

## Was vorher falsch war

1. **Brückentreppen standen an der falschen Stelle.** Alle Brückentreppen standen direkt an der **Ostwand** (1,9 m breit) und verdeckten dort die Ladenfronten.
   In Wirklichkeit stehen sie **frei, etwa in der Trogmitte**, mit einem Gehweg auf beiden Seiten (Fotos unten).
2. **Bei Brücke 1 fehlten zwei Läufe.** An der ersten Brücke nördlich des Kröpcke (s −66…−60, Spiel ≈ (2, −45)) gab es gar keine Treppen.
   OSM hat dort zwei Läufe: 27 Stufen (Way 409290611) und 26 Stufen (Way 436076598).
   Was bisher „Brücke 1/2“ hieß, sind in Wirklichkeit die Brücken 2 und 3.
3. **Das Nordende am Lichthof vor dem Hbf war erfunden.** Dort stand eine Treppe plus Rolltreppe an der **Westwand**, abwärts nach Süden, ohne Quelle.
   Laut OSM und Fotos von 2012 und 2016 gibt es dort **eine** Treppe in der Trogmitte: 12 Stufen, 1,5-m-Zwischenpodest, 15 Stufen, 1,6 m breit, oben am Nordrand (Ernst-August-Platz).
   Eine Rolltreppe gibt es nicht.
4. **Die 29-Stufen-Treppe am Kröpcke stand zu weit außen.** Sie stand an der Ostwand. Tatsächlich steht sie frei östlich der Trogmitte, mit Glas auf beiden Seiten und etwa 3 m Gehweg zur Ostwand.
5. **Die Stufenzahlen waren gerechnet.** Sie kamen aus der Höhe mit maximal 16,5 cm Steigung, nicht aus den OSM-`step_count`-Werten.
   Die Treppen waren geschlossene Granitstufen mit Glas auf nur einer Seite. Die Fotos zeigen offene Stahltreppen.
6. **Unter jedem Treppenkopf war eine Lücke.** Die Stirnblende der Brücke fehlte dort, man sah in die Brücke hinein.

Zum Vergleich: Die Street-View-Notiz in `data/streetview_notes.md` sagt „Treppe an der Westseite des Trogs“ (Kröpcke-Ende). Die Commons-Fotos „Hannover, inner city, 13“ und „Kirchentag 2025“ (beide nach Norden aufgenommen) zeigen die Treppe aber rechts, also auf der Ostseite. Die Notiz ist damit widerlegt.

## Wie es jetzt ist

Alle 8 Treppen sind gerade, freistehende Stahltreppen **parallel zur Promenade**, ungefähr in der Trogmitte.
- Stufen: offene Setzstufen, Stahl-Trittblech mit Steinbelag und dunkler Trittkante, Auftritt 30 cm.
- Seiten: Stahlwangen auf beiden Seiten mit Glasbrüstung und Edelstahlhandlauf.
- Unten: Betonsockel als Keil unter den untersten Stufen und ein Aufmerksamkeitsfeld.
- Oben: Kopfträger, taktiles Antrittsfeld, und die Brüstung der Brücke hat eine Lücke für den Lauf.
- Die Ladenfronten laufen jetzt an beiden Wänden durch.

| Name | Oben (Spiel x, z) | Lage im Trog | Läuft abwärts nach | Stufen | Steigung | Breite | Quelle |
|---|---|---|---|---|---|---|---|
| kroepcke_29 | (−14.0, −15.2), Südrand des Trogs | östlich der Mitte (t 3.0…5.6) | Norden | 29 | 15,9 cm | 2,6 m | OSM 31287201 |
| bruecke1_sued | (0.6, −42.8), Brücke 1 Südkante | Mitte | Süden | 27 | 15,9 cm | 2,4 m | OSM 409290611 (neu) |
| bruecke1_nord | (4.3, −47.8), Brücke 1 Nordkante | Mitte | Norden | 26 | 16,2 cm | 2,4 m | OSM 436076598 (neu) |
| bruecke2_sued | (30.3, −82.0) | Mitte | Süden | 25 | 15,5 cm | 2,4 m | OSM 254129194 |
| bruecke2_nord | (35.3, −88.7) | Mitte | Norden | 25 | 15,6 cm | 2,4 m | OSM 254129190 |
| bruecke3_sued | (60.9, −122.3) | Mitte | Süden | 24 | 15,8 cm | 2,4 m | OSM 254129195 |
| bruecke3_nord | (65.4, −128.3) | Mitte | Norden | 25 | 15,5 cm | 2,4 m | OSM 254129192 |
| hbf_lichthof | (119.3, −201.1), Nordrand Lichthof | Mitte bis leicht West (t −1.56…0.10) | Süden | 12 + Podest 1,5 m + 15 | 16,5 cm | 1,6 m lichte Breite | OSM 394588388 / 394588385 / 31443543 |

**Offene und überdeckte Abschnitte:** Die Läufe stehen immer in den offenen Abschnitten und enden oben an der Kante einer überdeckten Brücke bzw. am Trogende. Die überdeckten Abschnitte selbst haben keine Treppen.
- Die drei Brücken (Straßenquerungen) liegen bei s −66…−60, −17…−9 und 34…41.
- Der lange überdeckte Abschnitt s 58…104 (Ernst-August-Platz Süd) hat keine Treppe.
- Am Nordende von Öffnung 3 (s 58) gibt es nur die Stirnwand mit Schriftzug und den Durchgang.

**Aufzüge und Rolltreppen:** OSM hat zwischen Straße und Promenade keine. Die einzigen Rolltreppen in der Nähe führen von der Kröpcke-B-Ebene zur U-Bahn und gehören nicht zur Passerelle.

**Querlage:** Die OSM-Querlage ist hier systematisch um **≈ 4,15 m nach Osten verschoben**. Das gilt für die Treppen, die Läden und die Mauer-Linien 435784547/139432718.
- Wie bestimmt: Mittel aus 7 Treppen, Streuung ±0,3 m, Vergleich mit dem DGM-Trog.
- Was folgt daraus: Mit dieser Korrektur liegen alle Brücken- und Lichthof-Treppen in der Trogmitte, also genau dort, wo Fotos und DOP sie zeigen.
- Woher die Längslage kommt: Die Treppenköpfe liegen auf den Trogkanten aus dem DGM1.

### Quellen
- OpenStreetMap über die Overpass-API (Stand 18.11.2026): highway=steps mit `step_count`, `incline` und `level`. Die Brückentreppen wurden in OSM am 19. und 21.03.2026 geprüft (`check_date`).
- Wikimedia-Commons-Fotos (nur angesehen, nicht gespeichert):
  - „2026-01-06 Niki-de-Saint-Phalle-Promenade 0047“: freistehender Lauf zu Brücke 2 mit Gehweg auf beiden Seiten und dem Gegenlauf unter der Brücke.
  - „Hannover 20260815 Niki-de-Saint-Phalle-Promenade“
  - „Hannover, Bahnhofstrasse beneden bij de winkels … 2018-07-01“: gerader Lauf ohne Podest, Glasbrüstung, offene Stufen.
  - „Mitte, Hanover – panoramio (19)/(20)“: Untersicht mit offenen Stufen, Stahlwange und Betonsockel am Fuß.
  - „In der Niki de Saint Phalle Promenade Feb. 2016“
  - „Hannover, inner city, 13“ (2026) und „2025-04-29 Kirchentag 2025“: 29-Stufen-Treppe am Kröpcke auf der Ostseite, Lauf zu Brücke 1 in der Mitte.
  - „Hannover Hauptbahnhof – Auf dem Ernst-August-Platz … (6987970693)“ (2012) und „Hauptbahnhof bei Nacht Feb. 2016“: Lichthof-Treppe mittig, Durchgänge auf beiden Seiten.
  - „2026-01-06 … 0051/0061/1380168“: Öffnung 3 und 4 ohne weitere Treppen.
- LGLN DOP20 (`data/ground_cache/dop_0p2.npy`): In Öffnung 1 und 2 sieht man symmetrische Treppenschatten in der Trogmitte an beiden Brückenköpfen. Der Trog liegt sonst stark im Schatten.
- Google-Maps-Luftbild (nur angesehen): nicht verwertbar, weil der Trog im Schatten liegt.

## Neu erzeugte bzw. geänderte Dateien

- `pipeline/ground/promenade.py`:
  - Treppenteil neu geschrieben (`add_stair`, `stair_geom2`, `prism_sy`, `pitch_line`).
  - Unter jedem Treppenkopf ist jetzt die Stirnblende geschlossen.
  - Lichtmasten halten mindestens 1 m Abstand zu den Treppen.
  - Pflanzkübel stehen nicht mehr neben den Läufen.
  - Das `index.json` enthält jetzt Treppen-Infos: Stufen, Steigung, Breite, Kopf-Koordinaten und Quelle.
- `data/ground_cache/promenade.pkl` (Cache) und `data/ground_cache/collision.npz` (Cache des Exports).
- `web/assets/city/ground/`:
  - die Kacheln `ground_-1_-1.glb`, `ground_0_-1.glb`, `ground_0_-2.glb`, `ground_1_-2.glb` und `ground_1_-3.glb`
  - `collision_ground.glb`, `heightfield.bin` und `index.json`
  - Alle anderen Kacheln sind byteidentisch geblieben.
  - Das Höhenraster wurde nur im Promenadenband übernommen. Außerhalb gab es 5 Zellen mit 3–12 cm Raycast-Rauschen; dort bleiben die alten Werte.
- Rohkopien derselben GLBs und des `index.json` in `data/ground_raw/`, so wie es `publish()` sonst auch macht.
- `web/assets/city/manifest.json`: nur Dreieckszahl und Bounds der 5 Kacheln sowie die Kollisions-Dreieckszahl angepasst. Inzwischen hat eine andere Session das Manifest komplett neu gebaut, es ist konsistent.
- Die Boden-Kacheln waren schon vorher nicht meshopt-komprimiert. Das bleibt so, weil `export.py` Roh-GLB schreibt.

**Prüfungen:**
- Alle Treppen-Festkörper sind geschlossen (Wangen, Sockel, Trittbleche, Kopfträger), ohne offene oder nicht-mannigfaltige Kanten, und die Normalen zeigen nach außen.
- Die 18 Körper, die die Prüfung meldet, sind alte Lichtmast- und Kübelzylinder, nicht Teil dieser Änderung.
- Kollision: Für jeden Lauf gibt es ein geschlossenes Stufenprofil, das auch ins Höhenraster eingeht, dazu je Seite eine Brüstungswand und den Sockel als Festkörper.
- Die Kollision außerhalb des Promenadenbands ist dreiecksgenau unverändert.
- Blender-Vorschauen liegen in `C:\Users\luca\AppData\Local\Temp\claude\86a9b6371061b51e\8fff28a3-2832-4af9-8c96-1145fa294d87\scratchpad\prev\tp_*.png`.
- Im laufenden Spiel ist es nicht getestet.

## Nebenbefunde (nicht geändert)

- **Manifest:** `web/assets/city/manifest.json` listete bis heute nur 122 der 280 Boden-Kacheln. Gegen 02:01 hat eine andere Session (Gottfried-Landmark) `build_manifest.py` komplett neu laufen lassen. Jetzt stehen alle 280 Kacheln drin, passend zu meinem `ground/index.json` (geprüft: keine Abweichung, Kollision 756 840 Dreiecke).
- **`gottfried.json` bricht Export und Vorschau:** Eine andere Session hat `data/landmarks_replace/gottfried.json` neu angelegt, ohne `footprints`. Damit brechen `export.py` (Höhenraster) und `preview2.py` mit `KeyError: 'footprints'` ab. Ich habe den Export über einen Wrapper im Scratchpad laufen lassen, der solche Dateien überspringt. Die Datei selbst und `export.py` habe ich nicht angefasst.
- **`--dry` ändert getrackte Dateien:** `data/ground_cache/dry/` ist in Git eingecheckt, und `export.py --dry` überschreibt den Ordner. Ich habe ihn danach mit `git checkout -- data/ground_cache/dry` wiederhergestellt. Übrig geblieben ist die ungetrackte Kopie `dry/tex/d_ground_asphalt_61feabfb.jpg`.
- **Begehungsliste veraltet:** `BEGEHUNGSLISTE.md` B-12 („Treppen an den Brücken, im Modell an der Ostwand“) und die Treppen-Koordinaten in P-1 bis P-4 beruhen noch auf den unkorrigierten OSM-Lagen. Sie sollten an die Tabelle oben angepasst werden.

## Offene Stellen: Fotos und Maße vor Ort

1. **Alle 8 Treppen, Querlage.** Den Abstand von der Außenkante beider Wangen bis zur Ladenfront West und Ost messen, je am Fuß und am Kopf. Dazu ein Foto frontal in Laufrichtung von unten und eins von der Brücke herab.
   Die Laufmitte ist im Modell aus OSM minus 4,15 m bzw. als Brückenmitte gesetzt.
2. **Laufbreite und Stufen.** Lichte Breite zwischen den Wangen und Gesamtbreite messen, Wangenhöhe und -stärke, Stufen zählen, Steigung und Auftritt messen.
   Belag der Trittstufen fotografieren: Stahl, Stein oder Gitterrost? Sind die Setzstufen wirklich offen?
   Im Modell: Brückentreppen 2,4 m (Schätzung aus Fotos), Kröpcke 2,6 m, Auftritt überall 30 cm.
3. **Zwischenpodeste der Brückentreppen.** OSM nennt bei allen Brückentreppen `level=-1;-0.5;0`, die Fotos zeigen gerade Läufe ohne Podest. Prüfen: Gibt es ein Podest, und wenn ja, nach wie vielen Stufen und wie lang?
4. **Fuß und Unterbau.** Betonsockel am Fuß fotografieren und messen: Länge, Höhe, Form. Gibt es Zwischenstützen unter langen Läufen? Wie ist der Kopf an die Brückenstirn angeschlossen (Untersicht von schräg unten)?
5. **Treppenkopf auf der Brücke.** Wie ist die Brüstung (Glas, Edelstahl) an der Treppenöffnung ausgebildet, gibt es Aufmerksamkeitsfelder? Draufsicht und Seitenansicht fotografieren.
6. **Kröpcke-Ende, 29 Stufen** (oben ca. Spiel (−14, −15)). Querlage messen: Abstand zur Ostwand und zur Westwand. Breite, Glas auf beiden Seiten, Lage zum gerundeten Trogende und zum Übergang in die B-Ebene.
7. **Lichthof vor dem Hbf** (oben ca. Spiel (119, −201)).
   - Gibt es die Treppe 2026 noch? Die Commons-Fotos von 2026 zeigen nur die Südwand des Lichthofs, die Treppe ist darauf nicht zu sehen.
   - Breite (OSM 1,6 m), Podestlage, Stufenzahl 12 + 15.
   - Wie viele Durchgänge nach Norden Richtung Hbf gibt es, links und rechts der Treppe? Im Modell gibt es nur einen Durchgang auf der Ostseite, deshalb steht die Treppe dort leicht westlich der Mitte.
8. **Aufzüge.** In OSM gibt es keinen Aufzug Straße ↔ Passerelle. Vor Ort prüfen, ob es einen gibt, zum Beispiel am Kröpcke-Center („Aufzug Passerellenebene“, laut Street-View-Notiz) oder am Hbf-Ende. Wenn ja: Lage, Türseite, Kabinengröße.
9. **Rolltreppen.** Prüfen, dass es zwischen Straße und Passerelle wirklich keine gibt.
10. **Lichtmasten und Banner** neben den Treppen. Abstand zu den Läufen prüfen: Im Modell fehlt jetzt der Mast bei s 122 am Lichthof.

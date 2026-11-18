---
name: figur-einbauen
description: "Baut aus einem 3D-Scan einer Person (Gaussian-Splat-PLY oder farbiges KIRI-Mesh) eine spielbare Figur für Startbildschirm und Hannover-Spiel: ausschneiden, lückenlos machen, riggen (23 Knochen wie Carlos), Animationen übertragen, exportieren, eintragen, im Browser prüfen. Verwenden, wenn neue Scans, Fotos oder Figuren ins Spiel sollen."
---

# Figur einbauen

Ein Durchlauf macht aus einem Scan eine Figur, die im Startbildschirm steht und im Spiel läuft, rennt und stolpert.
Alle Skripte liegen in `scripts/` (Blender 5.x headless + numpy, kein pip nötig).

| Datei | Zweck |
|---|---|
| `scripts/splat_werkzeug.py` | Splat-PLY lesen/schreiben, Person in einer Szene finden (`ansicht`), freistellen + aufrichten + skalieren (`ausschneiden`) |
| `scripts/figur_bauen.py` | Blender: geschlossene Hülle, Skelett, Gewichte, Animationen von Carlos, Splats + Skin, GLB, Testbild |
| `scripts/carlos_skelett.json` | Ruhepose der 23 Knochen (Rigify-Namen) – Vorlage für jede Figur |
| `scripts/figur_hybrid.py` | **In Arbeit:** KI-Mesh (figuro-STL) als Körper + echte Scan-Splats darauf + menschliches Skelett mit Fingern |
| `scripts/skelett.py` | Skelett-Anpassung an Körper-Mesh (Becken, Schultern, Arme, Hände mit Fingererkennung), Finger entspannen, Hand-Gewichte |
| `scripts/mensch_skelett.json` | 61 Knochen: 23 Spiel-Knochen (Achsen wie Carlos) + Rigify-Hände (Handfläche, Daumen, 4 Finger à 3 Glieder) |

Blender: `/Applications/Blender.app/Contents/MacOS/Blender`. Zwischenergebnisse in den Scratchpad, nicht ins Repo.

## 0. Material sichten und zuordnen

1. **Fotos**: Kontaktbogen bauen (Blender kann JPEGs laden, Video-Bilder über den Sequencer) und Personen/Outfits unterscheiden.
2. **Splat-PLY** (Kopf hat `f_dc_0`, `opacity`, `scale_0`, `rot_0`): oft eine ganze Szene (Flur, Boden, Wände). Mit `ansicht` rendern und die Person suchen; KIRI-/COLMAP-Szenen haben meist **oben = -y**.
3. **STL** (z. B. figuro): KI-Modelle aus Fotos, ohne Farbe, Größe normiert (~1900 mm). Saubere Form (Finger, geschlossene Hülle), aber Proportionen nur geschätzt. Spielbar als Mesh-Figur, wenn es einen Splat-Scan derselben Person für die Farbe gibt (`--art stl --farbe`).
4. Zuordnung über Kleidung, Frisur, Schuhe, Kabel/Accessoires. Ergebnis dem Nutzer als Tabelle zeigen.
5. Beste Vorlage wählen: echte Splats mit Farbe > farbiges Mesh > STL. Freie Arme (A-Pose), freie Hände, Ärmel (nackte Arme werden in Splats fleckig), wenig Bodenfarbe an den Füßen.

## 1. Person ausschneiden (nur Splat-Szenen)

```bash
python3 scripts/splat_werkzeug.py ansicht <szene.ply> <out.png> --oben -y            # Übersicht, Person finden
python3 scripts/splat_werkzeug.py ansicht <szene.ply> <out.png> --oben -y --zentrum " x,y,z" --radius 1.1
python3 scripts/splat_werkzeug.py ausschneiden <szene.ply> <person.ply> --zentrum " x,y,z" --oben -y --groesse 1.80 --vorschau <out.png>
```

- `--zentrum` aus der Übersicht ablesen (Bildmitte = Zentrum, Kachelbreite = 2 × Radius). In zsh mehrere Aufrufe einzeln schreiben (`set -- $var` teilt nicht).
- Ergebnis: Z oben, Blick nach -Y, Füße auf z=0, echte Größe. Vorschau = vorne, links, hinten, rechts. **Prüfen**: Kopf oben, Gesicht in Bild 1, keine Wand-/Bodenreste.
- Boden = unterste 0,5 % der Säule unter der Person, Ebene per RANSAC. Größe ist eine Annahme (Standard 1,80 m) – im Bericht nennen.

## 2. Figur bauen

```bash
Blender -b --factory-startup --python scripts/figur_bauen.py -- --id <id> --quelle <person.ply> --art splats --modus splat --arme a --out <ordner>
Blender -b --factory-startup --python scripts/figur_bauen.py -- --id <id> --quelle "<KIRI ..._mesh.ply>" --art scan --modus mesh --arme a --out <ordner>
Blender -b --factory-startup --python scripts/figur_bauen.py -- --id <id> --quelle <figuro.stl> --art stl --farbe <person.ply> --modus mesh --arme a --out <ordner>
```

| Option | Werte |
|---|---|
| `--art` | `splats` (ausgeschnittene Splats) · `scan` (farbiges KIRI-Mesh, Y oben) · `stl` (farbloses Mesh, Z oben; wird mit Maßstab per ICP auf `--farbe <person.ply>` ausgerichtet, 4 mm geschlossen neu vernetzt und aus den Splats eingefärbt) |
| `--modus` | `splat` = Punktwolke über verstecktem Füll-Mesh (sieht am echtesten aus) · `mesh` = Mesh mit gebackener Textur (robust, wenig Daten) |
| `--arme` | `a` = Arme frei, Arm-Animation per Welt-Drehung von Carlos · `taschen` = Hände in Taschen, Arme ruhig |

Was passiert (Splat-Figuren): Splats → Voxel (1 cm) → 3 cm schließen → innen füllen → Würfel-Außenflächen → Remesh + Glätten = **geschlossene** Hülle (0 offene Kanten) → Farben aus den Splats (Voxel-Farbfeld) → Lücken in der Punktwolke mit neuen Splats füllen → Skelett an Körper anpassen → Füll-Mesh = Körper 2 Voxel geschrumpft → Heat-Gewichte → Carlos-Animationen → Splat-Gewichte per Data-Transfer vom Füll-Mesh → Splats nach glTF drehen (SH mitgedreht, nur Grad 1 behalten) → GLB → Skin in GLB-Gelenkreihenfolge → Testbild.

Dauer: ~1 min (Splat-Figur) bzw. ein paar Minuten (Mesh mit Textur-Backen) – **bei freier CPU**. Laufen Videoanruf, Archicad o. Ä., bekommt Blender nur ~10 % CPU und alles dauert 10× länger → Nutzer bitten, Programme zu schließen. Fortschritt steht sofort im Log (`[  12s] 3 Spiel-Mesh`).

**Prüfen**: `REPORT`-Zeile (offene_kanten = 0, nicht_mannigfaltig = 0, 7 Clips, 23 Knochen) und `_test/<id>_test.png` ansehen (Gehen/Laufen/Stolpern, vorne + seitlich): keine abgerissenen Teile, Arme schwingen, Füße auf dem Boden.

## 2b. Hybrid: KI-Mesh + Scan-Splats + Skelett mit Fingern (in Arbeit)

```bash
Blender -b --factory-startup --python scripts/figur_hybrid.py -- --id <id> --stl <figuro.stl> --farbe <person.ply> --out <ordner>
```

1. STL per ICP auf die Scan-Splats, 4 mm geschlossen neu vernetzt (sauberer Körper, echte Finger).
2. Skelett aus `mensch_skelett.json`: Becken/Schultern/Wirbelsäule aus Querschnitten, Hände über die Oberfläche: Entfernung vom Handgelenk über die Haut (Dijkstra) → Finger = Äste im Verschmelzungs-Baum, Daumen = Ast, der am nächsten am Handgelenk abzweigt; Knöchel nach Handproportion (Zeige 0,52 · Mittel 0,53 · Ring 0,50 · Klein 0,45 × Handlänge). Kleiner Finger, der am Ringfinger klebt, wird daneben ergänzt.
3. Gewichte: Heat + **Jacken-Regel** (oberhalb der Hüftgelenke keine Bein-Gewichte → Jacke schwingt nicht mit den Beinen) + **Hand-Gewichte geometrisch** je Finger-Ast.
4. Scan-Splats auf die Mesh-Oberfläche gelegt (> 3 cm daneben = verworfen), Hände und Lücken mit neuen Splats aus dem Mesh.
5. **Finger entspannen**: nebeneinander, leicht zur Handfläche gekrümmt; als neue Ruhepose übernommen (Mesh + Splats gleich), damit die Spiel-Animationen sie nicht spreizen.

Stand 2026-11-12: läuft durch (Trainingsjacke: 61 Knochen, 5 Finger je Hand erkannt, 0 offene Kanten). **Noch nicht im Spiel getestet** – vor dem Eintragen Testbilder und Fingerhaltung prüfen. Skelett lässt sich live über Blender MCP (Port 9877) ansehen: Körper laden, `skelett.anpassen` + `rig_bauen` ausführen, Ansicht per `region_3d` setzen, Screenshot.

## 2c. Splat-Scan mit Fingern (Standard seit 2026-11-18, im Live-Blender über das Blender-MCP)

```bash
python scripts/splat_werkzeug.py ausschneiden <szene.ply> figuren/<id>/<id>_person.ply --zentrum "x,y,z" --oben -y --groesse 1.80 --vorschau <png>
```
Dann im Live-Blender (MCP, `execute_blender_code`): `sys.argv = ["blender","--","--id",<id>,"--quelle",<person.ply>,"--out",<ordner>]` setzen,
`sys.modules.pop("skelett")`, `scripts/figur_finger.py` per `exec` in einem `bpy.app.timers`-Callback ausführen (Log in eine Datei umleiten) –
der Lauf dauert ~1,5 min und blockiert sonst den MCP-Aufruf. Headless geht auch: `Blender -b --factory-startup --python scripts/figur_finger.py -- …`.

- Körper = geschlossene Hülle aus den Splats (`--voxel 0.006` Standard; bei verschmolzenen Fingern `--voxel 0.004 --schliessen 2`).
- Skelett 61 Knochen (`mensch_skelett.json`), **Schulter aus der Oberarm-Achse** (robust gegen Haare/Kapuzen), **Arm-Verfolgung entlang des Arms**
  (A-Pose, flache Arme, Scan-Lücken bis 14 cm), Finger per Oberflächen-Ästen; ohne Erkennung Vorlage + geometrische Gewichte zur nächsten Vorlage.
- Finger werden entspannt und als Ruhepose übernommen → im Spiel nie gespreizt. Export nur aus der aktiven Szene (`use_active_scene`).
- Scan-Viewer-Extraktion erkennt „oben“ bei diesen Flur-Szenen oft falsch (Wand als Boden) – Achse selbst prüfen (KIRI/COLMAP: `-y`).
  `scripts/aus_viewer.py` nutzt die Viewer-Werte (Mitte/oben/Höhe) und skaliert die Szene vorab auf Meter.

## 2d. Bevorzugt: figuro-STL + Scan-Splats, eigene Bewegungen (Stand 2026-11-18)

Wenn es ein figuro-STL der Person gibt, immer damit bauen – geschlossener Körper mit echten Fingern, keine Löcher:
1. `figur_hybrid.py -- --id <id> --stl <figuro.stl> --farbe <person.ply> --out figuren/<id>/hybrid` (im Live-Blender per Timer wie 2c).
   ICP mit Maßstab, danach Körper + Splats gemeinsam auf den Boden; Splats auf die Fläche gelegt, **nur deckende Scan-Splats (Deckkraft > 0,45)
   zählen als Abdeckung**, überall sonst (Abstand > 4 mm, `--luecke`) neue Splats aus dem Mesh mit Scan-Farbe → lückenlos.
2. `gang.py -- --id <id> --out <ordner mit <id>_skin.json>`: ersetzt Walk/Run/Idle_Breathing durch eigene Zyklen (IK-Beine mit
   Fersenaufsatz/Abrollen, Becken-Höhe/Pendel/Rotation, Rumpf-Gegenrotation, stabilisierter Kopf, Armpendel gegengleich, Laufen mit
   Flugphase und ~85° Ellbogen), exportiert die GLB neu und prüft, dass die Gelenk-Reihenfolge zum Skin passt.
   In Blender prüfen: Aktion auf das Rig legen, `look` mit `mode=frames`, `view=right`.
Ohne STL (z. B. bv9): `figur_finger.py --voxel 0.004 --schliessen 2` + `gang.py`.

**Scan-treu**: Scan-Splats bleiben unverändert (Position, Form, volle SH); Füll-Splats nur in echten Löchern (kein deckender Splat
im Umkreis von 2,5 cm). Nie Splats aufs Mesh ziehen oder Hände ersetzen – das hat Gesicht und Details zerstört.

**Gelenke (skelett.py, geprüft mit Prüf-Agenten 2026-11-18)** – Zielwerte als Anteil der Körperhöhe H:
- Schultergelenk: 4,7·k unter der Schulter-Oberkante bei |x| 17–21 cm (≈ 0,78–0,80 H), links/rechts gleich hoch, seitlich aus der
  Oberarmachse, symmetrisch um die Körpermitte auf Schulterhöhe. Nie die Armachse bis zur Schulter verlängern (6–8 cm zu hoch).
- Hüftkopf ≈ 0,505 H (nicht unter Schritt + 6 cm), 8,2–9,2·k neben der Beckenmitte, Tiefe = Mitte des Becken-Querschnitts.
- Knie/Knöchel: Mitte des Bein-Querschnitts (Ausdehnung, nicht Punktschwerpunkt); Knöchel 0,045 H.
- Wirbelsäule je Höhe an der Querschnittsmitte; Tiefe Becken 55 %, Lende 62–68 %, Brust 72 % von vorn; C7 4,5·k unter der Rückenhaut;
  Atlas (Kopfgelenk) bei H − 0,09 H (Ohrläppchen) mittig/hinten im Kopf; Schlüsselbein beginnt vorn am Brustbein.
- Finger: geodätische Entfernung vom Handgelenk + Merge-Tree (Persistenz) → die längsten Äste sind Finger, Sattel = Schwimmhaut.
  Verschmolzene Finger: Ast nach Breite (~1,9 cm/Finger) in Streifen teilen. Knöchel tief in der Handfläche (Fingerlänge als Anteil der
  Handlänge: Zeige .47, Mittel .51, Ring .48, Klein .39). Hand-Gewichte nur für Punkte ≤ 4,5·k von Hand-/Fingerknochen (sonst hängt
  z. B. Jackensaum an der Hüfte an den Fingern → Stachel beim Armschwung).

**Stand 2026-11-18 abends – maßgeblich:**
- Alle Scan-Figuren (bv9, bv10, bv11) **nur aus dem Scan** bauen (`figur_finger.py --voxel 0.004 --schliessen 2`, Ausgabe `build/`).
  Die figuro-STL-Variante (`figur_hybrid.py`, Ausgabe `hybrid/`) verschmiert Textur an den Armen – nicht mehr verwenden.
  LEDERJACKE (bv9) ist der Qualitätsmaßstab für Textur/Detail.
- Gehen/Laufen kommen aus **echten Motion-Capture-Aufnahmen** (CMU Graphics Lab, frei auch kommerziell; `figuren/_mocap/`):
  `07_01.bvh` Gehen (Knie im Stand ~9°, Spitze ~66° bei ~67 % – am nächsten an der Normkurve), `35_17.bvh` Joggen.
  `gang.py` überträgt sie per `mocap.py` (Welt-Drehungen relativ zur T-Pose, A-Pose-Angleichung je Knochen, Zyklus nahtlos,
  Becken auf der Stelle, tiefster Fußpunkt je Bild auf den Boden). Andere Aufnahme: `--walk_bvh <datei>` / `--run_bvh <datei>`.
  `--ohne_mocap` nimmt die alten prozeduralen Zyklen (nicht empfohlen). Idle bleibt prozedural.
- Kniekurve prüfen: Hilfsskript `knie.py` (Scratchpad-Muster): Knie-/Hüftwinkel je Bild gegen die Normkurve (Winter).

**Gang prozedural (alt, nur Fallback)**: Doppelschritt ≈ 0,82·k/0,6 (≈ 1,3 m) – das Spiel spielt Walk bei 1,5 m/s; zu kurze Schritte = Trippeln/„Pferd“.
Beckenhöhe aus der Beinreichweite (Pendel ~5 cm, nicht dauerhaft eingeknickt). Ferse löst sich ab ~35 % des Zyklus, Fuß rollt über den
Ballen (Knöchel geht hoch UND nach vorn), Hüftstreckung hinten ~15°, Zehen im Schwung 2 cm über dem Boden. **Vorzeichen**: Blick −Y →
Fuß-Kipp `Rotation(-kipp, 'X')` (+kipp = Zehen hoch). Laufen: Doppelschritt ≈ 2,8 m, Stand 27 %.

## 3. Ins Spiel eintragen

1. Dateien kopieren nach `web/assets/<id>/` **und** `startbildschirm/assets/`: `<id>.glb`, bei Splat-Figuren zusätzlich `<id>_splats.ply`, `<id>_skin.bin`, `<id>_skin.json`.
2. `web/js/carlos.js` → `FIGS`: `<id>: { base: 'assets/<id>/' }` (+ `mesh: true` bzw. `handsInPockets: true`).
3. `startbildschirm/main.js` → `CHARACTERS` (Name in Großbuchstaben, Rolle, `x`-Platz, `timeOffset`, Werte, Bio; `mesh: true` bei Mesh-Figuren) und `PLAYABLE`.
4. Lange Namen: `#c-name.long` (>6 Zeichen) und `.xlong` (>9) in `startbildschirm/ui.css` greifen automatisch.

## 4. Im Browser prüfen

Dev-Server `sensing-space` (Port 5177) starten → `/startbildschirm/` → Figur auswählen → BEREIT → im Spiel laufen/rennen.
Konsole muss fehlerfrei sein (insbesondere keine Meldung `skin/splat count mismatch`). Screenshot von Startbildschirm und Spiel zeigen.

## 5. Hochladen

Große Dateien laufen über Git LFS (`.gitattributes` deckt .glb/.ply/.bin ab). iCloud-Duplikate (`… 2.glb`) nie mit committen. Branch + Pull Request für Luca.

## Bekannte Fallen

- **Gelenk-Reihenfolge**: three.js nimmt `skeleton.bones` in der Reihenfolge der GLB (`skins[0].joints`), nicht der Vorlage. Skin-Indizes deshalb erst nach dem Export schreiben (macht `figur_bauen.py`). Falsche Reihenfolge = Kopf hängt am Oberarm („Flügel“, „Geister-Gesicht“).
- **Mitte der Figur** aus den Beinen bestimmen, nicht aus den Füßen (Bodenreste/ungleich dichte Füße verschieben sonst alles um ~10 cm; dann sitzen Wirbelsäule und Kopf-Knochen neben dem Körper).
- `--art scan` + `--arme a`: Haut zwischen Arm und Rumpf (Schwimmhaut) wird noch **nicht** automatisch geschnitten – beim Stolpern sieht man Lappen unter dem Arm. Das alte `figuren_rig/daunenjacke_bauen.py` (Schritt 3b, nur lokal bei Mohammad) hat dafür eine Lösung, die noch zu übernehmen ist.
- Hände/Saum können die Bodenfarbe tragen (oranger Flurboden) – sichtbar, aber nicht störend; bei Bedarf Splats nahe Boden mit Bodenfarbe entfernen.

- **Scan-Qualität entscheidet**: Hände in Taschen lassen sich nicht nachträglich herausholen; Gegenstände in der Hand verlängern den Arm (Länge wird auf 0,76 m gedeckelt); Boden-Reflexe färben Hände/Schuhsohlen.
- Splat-Szenen haben eine Hülle aus Hintergrund-Splats bei Radius ~240 – nie über den ganzen Datensatz mitteln, immer um die Person herum arbeiten.
- numpy auf macOS meldet bei `@` Fehlalarme („divide by zero in matmul“) – sind im Werkzeug unterdrückt.
- Blender 5: Aktionen haben Slots/Channelbags (`action.layers[0].strips[0].channelbags`), Posen müssen QUATERNION sein.
- Zu Anfang sicherstellen, dass die Datei-Größe im Spiel passt: Splat-PLY mit SH Grad 1 ≈ 116 Byte je Splat.

## Rezepte (bisher gebaute Figuren)

| Figur | Quelle | Befehle |
|---|---|---|
| trainingsjacke | `Bothfeld-Vahrenheide-3.ply` (Flur-Szene, 2 Personen; ausgeschnitten: `figuren/trainingsjacke/jacke_splats_roh.ply`; Fotos: WhatsApp 14.23.32–14.23.36 (11)) | `ausschneiden … --zentrum " 0.17,-0.40,0.01" --oben -y --groesse 1.80` → `figur_bauen.py --id trainingsjacke --art splats --modus splat --arme a` |
| (frei) | `Bothfeld-Vahrenheide-2.ply`: zweite Person, schwarzes T-Shirt, kurze Locken, braune Stiefel | `--zentrum " -0.05,0.32,-0.135" --oben -y` – gut nutzbar, nackte Unterarme etwas fleckig |
| trainingsjacke hybrid (in Arbeit) | `figuren/trainingsjacke/figuro_trainingsjacke.stl` + `figuren/trainingsjacke/jacke_splats_roh.ply` | `figur_hybrid.py --id trainingsjacke --stl … --farbe …` |
| jackemesh (Test) | `figuro-19a5….stl` + Farbe aus `trainingsjacke`-Splats | `--art stl --farbe jacke_splats_roh.ply --modus mesh --arme a` – Abstand zum Scan nach Ausrichtung ~1 cm, ~20 % der Fläche > 3 cm daneben (KI-Proportionen) |
| daunenjacke | `KIRI Person Daunenjacke_mesh.ply` | `--art scan --modus mesh --arme a` (ursprünglich mit eigenem Skript gebaut) |
| bv9 „Lederjacke" | `Bothfeld-Vahrenheide (9).ply` | `ausschneiden … --zentrum "0.164,-0.2,0.02" --oben -y` → `figur_finger.py --id bv9 --voxel 0.004 --schliessen 2` (Finger verschmolzen → Vorlage + geometrische Gewichte) |
| bv10 „Locken" (jetzt figuro_bv10.stl + Scan, gang.py) | `Bothfeld-Vahrenheide (10).ply` | `--zentrum "0.349,0.0,-0.019" --oben -y`, danach 180° um z drehen (Blick war verkehrt), orange Bodenreflexe bis Knie entfernt → `figur_finger.py --id bv10 --voxel 0.004 --schliessen 2` (5+5 Finger) |
| bv11 „Boots" (jetzt figuro_bv11.stl + Scan, gang.py) | `Bothfeld-Vahrenheide (11).ply` | `--zentrum "-0.106,0.3,-0.129" --oben -y` → `figur_finger.py --id bv11` (5+5 Finger) |
| mohammad | `KIRI Person Weste_mesh.ply` | `--art scan --modus splat --arme taschen` (ursprünglich mit eigenem Skript gebaut) |

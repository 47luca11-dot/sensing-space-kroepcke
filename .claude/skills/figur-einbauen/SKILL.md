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
| mohammad | `KIRI Person Weste_mesh.ply` | `--art scan --modus splat --arme taschen` (ursprünglich mit eigenem Skript gebaut) |

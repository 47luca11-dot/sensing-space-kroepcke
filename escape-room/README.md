# Escape Room – Architekturbüro

Browserbasiertes 3D-Escape-Room-Spiel in der Ego-Perspektive mit **Vite + TypeScript**, [Three.js](https://threejs.org), [Rapier](https://rapier.rs) (Physik) und [Spark](https://sparkjs.dev) (Gaussian Splats aus Scaniverse).

Du bist in deinem eigenen Architekturbüro eingeschlossen. Ziel: den Raum durch die verschlossene Ausgangstür verlassen. Im Büro stehen Zeichentisch, Massenmodell, Dreikantmaßstab, Modellregal, ein Tresor und drei lebensgroße Maßstabsfiguren (Bauherr, Statikerin, Praktikant), die man ansprechen kann.

**Lösungsweg (Spoiler):** Der Zettel auf dem Tisch deutet an, dass man etwas erst im Dunkeln sieht. Mit dem Lichtschalter neben der Tür das Deckenlicht ausschalten: An der Ostwand über dem Regal erscheint ein nachleuchtender Code. Mit diesem Code den Tresor an der Nordwand öffnen, darin liegt der Schlüssel zur Ausgangstür. Code, Hinweistext und Position der Leuchtschrift stehen in `src/config/puzzle.ts`.

## Lokal starten – Schritt für Schritt

1. **Node.js installieren** (Version 20 oder neuer, getestet mit Node 24): <https://nodejs.org>. Prüfen im Terminal mit `node --version`.
2. **Terminal im Projektordner öffnen**, also in `escape-room/`.
3. **Abhängigkeiten installieren** (nur beim ersten Mal oder nach Änderungen an `package.json`):
   ```bash
   npm install
   ```
4. **Entwicklungsserver starten:**
   ```bash
   npm run dev
   ```
5. Die angezeigte Adresse im Browser öffnen, standardmäßig <http://localhost:5173>. Änderungen am Code erscheinen beim Speichern automatisch.
6. Auf **„Klicken zum Starten“** klicken. Die Maus wird eingefangen, **Esc** gibt sie wieder frei.

Für eine fertige, statische Version zum Hochladen auf einen Webserver: `npm run build` (Ergebnis in `dist/`), lokal ansehen mit `npm run preview`.

### Auf dem Handy testen

- **Im selben WLAN:** `npm run dev` zeigt neben „Local“ eine „Network“-Adresse (z. B. `http://10.172.139.29:5174/`). Diese im Handy-Browser öffnen. Windows fragt beim ersten Mal eventuell nach einer Firewall-Freigabe für Node.js.
- **Von unterwegs:** `npm run build:artifact` baut eine Version für ein privates claude.ai-Artifact (Binärdateien werden als Base64-Text abgelegt, weil Artifacts nur Web-Standardtypen ausliefern). Claude veröffentlicht den Ordner `dist-artifact`.
- Auf Touchgeräten erscheint automatisch die Bildschirmsteuerung: linker Daumen läuft, rechte Hälfte wischen sieht sich um, Knöpfe für Aktion, Schlag/Wurf, Zeigen/Drehen, Essen/Trinken, Springen, „⋯“ für weitere Aktionen, „❚❚“ für Pause.

## Steuerung

| Taste | Aktion |
| --- | --- |
| W A S D / Pfeiltasten | Laufen · Shift: rennen · Leertaste: springen |
| Maus | Umsehen |
| E | Anvisierten Gegenstand aufnehmen · gehaltenen ablegen · Tür/Schalter bedienen · Schlüssel benutzen |
| Linksklick | Leere Hand: Faustschlag · mit Gegenstand: werfen · mit Messer: zustechen |
| Rechte Maustaste | Leere Hand: mit ausgestrecktem Zeigefinger zeigen (Name des Ziels erscheint) · mit Gegenstand + Maus: drehen |
| F | Gehaltenes essen (Apfel, Croissant) oder trinken (Flasche, Kaffeetasse, Bier) |
| M (halten) | Geld werfen; liegende Scheine sammelst du beim Darüberlaufen wieder ein |
| Mausrad | Gegenstand weiter weg / näher halten |
| P (halten) | Urinieren, mit der Maus zielen (nach ca. 6 s leer, füllt sich langsam wieder) |
| K | Stuhlgang: kurz in die Hocke, danach liegt ein aufnehmbarer Haufen hinter dir |
| V | Erbrechen (drei Schübe, danach kurze Pause) |
| G | Spucken |
| N | Niesen |
| Esc | Pause, Mauszeiger frei |

## Hände

Die Ego-Hände stammen aus dem CC0-Modell „Posable anatomical right hand“ von Innerscene (MakeHuman), die linke Hand ist gespiegelt (Lizenz: `public/models/hands/LICENSE.md`). Hautdetails wie Poren, Rauheit und gerötete Fingerkuppen entstehen zur Laufzeit in `src/hands/skin.ts`.

- **Ruhe:** Beide Hände am unteren Bildrand, sie schwingen beim Laufen mit.
- **Anvisieren:** Die rechte Hand hebt sich und öffnet sich.
- **Halten:** Der Griff richtet sich nach der Größe: dünne Dinge (Schlüssel) zwischen Daumen und Zeigefinger, Flaschen und Tassen umgriffen, große Kisten mit beiden Händen.
- **Bedienen:** Bei Tür und Schalter tippt die rechte Hand hin.

Ein anderes, z. B. gescanntes Handmodell wird in `src/config/hands.ts` eingetragen (Datei, Knochennamen, Beugeachse). Ob das Modell eine linke oder rechte Hand ist und wo die Handfläche liegt, erkennt das Spiel selbst.

## Maßstabsfiguren und Messer

Die Figuren sind physikalisch frei (schwere Standplatte, leichter Körper): Faustschläge lassen sie wackeln, Stiche mit dem Mordmesser (Modellregal, mittleres Brett) hinterlassen Schnittspuren. Nach genug Treffern (`toughness` in der Konfiguration, Standard 6) kippen sie um. Mit E sprechen sie; einige Sätze sind Hinweise zum Rätsel. Sätze, Namen und Positionen stehen in `src/config/objects.ts` (`operable: { type: 'dummy', lines: [...] }`).

## Essen und Trinken

Essen füllt den Magen (Voraussetzung für den Stuhlgang), Trinken die Blase. Gegenstände werden über `consumable: { kind: 'food' | 'drink', bites }` essbar; Essen schrumpft mit jedem Bissen und ist danach weg, Getränke bleiben.

## Wandtextur aus einem Scan

Die Wände tragen verwitterten Sichtbeton mit Brettschalung aus dem Scan „Bothfeld-Vahrenheide“ (`public/textures/concrete-*.png`, 2,66 × 1,33 m pro Kachel, Lücken im Scan mit Flicken aus erfassten Stellen gefüllt). Eingebacken sind Laufspuren, Rostfahnen an Ankerlöchern, Risse, Lunker und Ausblühungen; der Shader in `src/room/grungeWall.ts` legt zusätzlich nicht wiederholenden Schmutz aus der Weltposition darüber (Dreck am Boden, Ruß unter der Decke, Flecken) und mischt zwei versetzte Kachelpositionen, damit kein Raster sichtbar wird. Die Texturkoordinaten kommen aus der Weltposition, das Muster läuft deshalb nahtlos über alle Wandstücke, um die Tür herum und maßstabsgetreu. Einstellungen: `ROOM.wallTexture` in `src/config/room.ts`.

Eine neue Textur aus einem anderen Scan erzeugen (Node genügt, keine weiteren Pakete):

```bash
node tools/find-planes.mjs scan.ply tex-out
node tools/render-plane.mjs scan.ply tex-out/plane-0.json tex-out/wall.png 0.003 0.08
node tools/make-grunge-wall.mjs tex-out/wall.rgba tex-out/wall.json tex-out/final
```

1. `find-planes` sucht die größten ebenen Flächen (RANSAC) und legt je Fläche eine Vorschau `plane-N.png` ab. Die passende Fläche aussuchen.
2. `render-plane` rendert die Splats dieser Fläche orthografisch als elliptische Gaussians (3 mm pro Pixel).
3. `make-grunge-wall` füllt Lücken, gleicht Licht aus, macht die Kachel nahtlos und backt Verschmutzung ein (Farbe, Normal Map, Rauheit, `preview-tiled.png`). Die schlichtere Variante ohne Verschmutzung: `make-seamless`. Sie wählt den am besten erfassten Ausschnitt, gleicht Licht und Farbflecken aus, füllt kleine Löcher, passt die Kachelhöhe an den Fugenabstand an, überblendet die Ränder und erzeugt Farb- und Normal Map plus `preview-tiled.png` zum Prüfen der Nähte. Optional als 4. Argument ein eigener Ausschnitt `x,y,breite,höhe` in Pixeln.

## Enthüllungseffekt

Splat-Objekte bauen sich beim Spielstart (und hochgeladene Scans sofort) von unten her auf: Noch nicht erreichte Bereiche sind eine lichte Punktwolke, an der Grenze läuft eine leuchtende Regenbogenfront. Der Effekt läuft komplett auf der GPU (Spark-Dyno, `src/effects/UnveilEffect.ts`).

Im Start- und Pausenbildschirm steht oben rechts das Feld „Enthüllungseffekt“: Effekt mit/ohne Regenbogen, Reichweite, Übergangsbreite, Dauer, Punktdichte, Fortschritt (zum Durchscrubben) und „Abspielen“. Standardwerte in `src/config/effects.ts`.

## Roter Raum, Geld und Bier

Hinter dem Durchgang in der Südwand des Büros liegt der rote Raum: samtrote Wände, Neonleisten, rotes Licht, in der Mitte eine runde Bühne mit Chromstange, dazu Bar, Sofas, sechs Bierflaschen und ein Stammgast. Maße und Licht: `src/config/redRoom.ts`, Einrichtung: `src/config/objects.ts`.

- **Geld werfen** (`src/game/MoneyThrower.ts`): Scheine (Spielgeld, 50 €) flattern mit eigener Papierphysik, landen auf festen Flächen, prallen von beweglichen ab. Start mit 1000 €. Solange Geld fliegt, tanzen die Maßstabsfiguren in der Nähe.
- **Bier** (5 Schlucke pro Flasche): Jeder Schluck erhöht den Pegel (`src/game/Intoxication.ts`, Anzeige in ‰ oben links). Stufen „Leicht angeheitert“, „Betrunken“, „Sturzbetrunken“: Sicht schwankt, Sichtfeld pulsiert, Bild wird unscharf, der Gang pendelt seitlich. Bei sehr hohem Pegel übergibt man sich. Der Pegel baut sich langsam ab.

## Module

Schalter in `src/config/features.ts`:

| Modul | Beschreibung |
| --- | --- |
| `redRoom` (an) | Roter Raum hinter der Südwand |
| `moneyThrowing` (an) | Geld werfen mit M |
| `alcohol` (an) | Bier macht betrunken |
| `bodilyFunctions` (an) | Körperfunktionen (`src/game/BodilyFunctions.ts`, Tasten P, K, V). Enthält ein wiederverwendbares Flüssigkeitssystem: Partikel mit Schwerkraft, Pfützen auf beliebigen Oberflächen, Impulse auf leichte Gegenstände. |

## Eigene Scans einbinden

### Schnell ausprobieren: Upload im Startbildschirm

Im Start- bzw. Pausenbildschirm unter **„Eigene Scans hinzufügen“** Dateien wählen oder hineinziehen (`.spz`, `.ply`, `.splat`, `.ksplat` oder `.glb`). Der Gegenstand erscheint direkt vor dir im Raum.

- **Typ:** beweglich (aufnehmbar, mit Schwerkraft) oder fest.
- **Kollision:** konvexe Hülle (passt sich dem Scan an), Box, Zylinder oder Kugel.
- **Maßstab:** Scaniverse exportiert in Metern, also meist `1`.
- **Y-Achse drehen:** an lassen, wenn der Scan auf dem Kopf steht, sonst ausschalten.

Uploads gelten nur für die laufende Sitzung. Für den festen Einbau den nächsten Abschnitt verwenden.

### Fest einbauen

1. **Datei ablegen** in `public/models/`, z. B. `public/models/schluessel.spz`.
   Empfehlung: **SPZ** aus Scaniverse exportieren, die Datei ist etwa 10× kleiner als PLY. Objekte vorher freistellen, z. B. mit dem `gaussian-splat-viewer` („Objekt anklicken…“ → „Auswahl als .ply“).
2. **In `src/config/objects.ts` eintragen** oder bei einem Platzhalter nur `model` austauschen:
   ```ts
   {
     id: 'key',
     label: 'Schlüssel',
     kind: 'movable',                 // 'static' | 'movable' | 'operable'
     itemId: 'exit-key',              // passt zu operable.requires der Tür
     model: { type: 'splat', url: '/models/schluessel.spz' },
     collider: { shape: 'auto', fit: 'box' },
     position: [3.3, 1.25, 1.35],     // Meter, Y = 0 ist der Boden
     rotation: [0, 70, 0],            // Grad
     scale: 1,
     mass: 0.1,                       // kg
   },
   ```
3. Speichern, der Browser lädt neu.

Die Interaktionslogik hängt nur an `kind`, `itemId` und `operable`, nie am Modell. Ein Platzhalter lässt sich deshalb jederzeit durch einen Scan ersetzen.

### Optionen im Überblick

| Feld | Bedeutung |
| --- | --- |
| `model` | `{ type: 'placeholder', shape, size, color }`, `{ type: 'splat', url, flip?, minOpacity? }` oder `{ type: 'gltf', url }` |
| `kind` | `static`: festes Raumelement · `movable`: aufnehmbar, Schwerkraft · `operable`: Tür/Schalter |
| `collider` | `box`, `sphere`, `cylinder`, `capsule`, `compound` (mehrere Teile) oder `auto` mit `fit: 'box' \| 'cylinder' \| 'sphere' \| 'hull'` |
| `flip` (Splat) | 180° um X, Standard `true`. Auf `false` setzen, wenn der Scan danach auf dem Kopf steht. |
| `minOpacity` (Splat) | Fast unsichtbare Splats (Scan-Rauschen) zählen nicht für Mittelpunkt und Kollisionskörper, Standard `0.2`. |
| `operable` | Tür/Klappe `{ type: 'door', requires?: 'exit-key', code?: '4721', openAngle: 100, exit?: true }` (Schlüssel oder Zahlenschloss; `exit` beendet den Raum) · Schalter `{ type: 'switch', target: 'ceiling' }` · Zettel `{ type: 'note', text: '…' }` |

Splat- und glTF-Modelle werden automatisch auf die Mitte ihrer Bounding Box zentriert. `position` ist also die Objektmitte. Bewegliche Gegenstände knapp über die Ablagefläche setzen, sie fallen dann sanft darauf.

**Kollisionskörper:** Komplexe Scans kollidieren nie mit dem Detailmodell, sondern mit einer einfachen Ersatzform. `auto` passt diese Form an die robusten Grenzen des Scans an (Ausreißer werden ignoriert). `hull` ist am genauesten, `box` und `cylinder` sind am stabilsten.

## Projektstruktur

```
src/
  config/        Zentrale Einstellungen: Raum (room.ts), Gegenstände (objects.ts), Typen
  room/          Raumaufbau: Boden, Wände mit Türöffnung, Decke, Licht
  controls/      Eingabe (Pointer Lock, Tastatur, Maus) und Ego-Steuerung (Rapier Character Controller)
  physics/       Rapier-Welt mit festem Zeitschritt, Kollisionsgruppen, Kollisionskörper
  objects/       Spielobjekte, Modell-Laden (Platzhalter/Splat/glTF), Platzhalter-Formen
  interaction/   Anvisieren, Hervorheben, Aufnehmen/Halten/Drehen/Werfen, Türen und Schalter
  game/          Spielablauf (Sieg, Meldungen) und Ereignisse
  ui/            HUD (Fadenkreuz, Hinweise, Overlays) und Upload-Feld
public/models/   Eigene Scans
```

**Ablauf pro Bild:** Mauseingabe → feste Physikschritte (60 Hz: Spieler, gehaltener Gegenstand, Türen, `world.step`) → Interaktion und HUD → Rendern. Modelle werden zwischen Physikschritten interpoliert und laufen daher auch auf 120/144-Hz-Monitoren flüssig.

**Neue Rätselmechanik hinzufügen:** In `src/interaction/operables.ts` eine Klasse mit `prompt()` und `interact()` schreiben, in `createOperable` registrieren und in `config/types.ts` einen `OperableConfig`-Typ ergänzen. Auf ihre Ereignisse reagiert `src/game/GameLogic.ts`.

## Debuggen

Im Entwicklungsmodus steht in der Browser-Konsole `__game` zur Verfügung, z. B.:

```js
__game.player.teleport([0, 0, 0], 90)        // Position, Blickrichtung in Grad
__game.registry.byId('key').body.translation()
__game.input.forceActive = true             // Eingaben ohne Pointer Lock
```

## Claude-Code-Skills

In `../.claude/skills/` liegen Three.js-Skills ([cloudai-x/threejs-skills](https://github.com/cloudai-x/threejs-skills), MIT) sowie `threejs-gameplay-systems` und `threejs-debug-profiler` aus [majidmanzarpour/threejs-game-skills](https://github.com/majidmanzarpour/threejs-game-skills) (MIT). Claude Code nutzt sie automatisch bei der Arbeit an diesem Projekt.

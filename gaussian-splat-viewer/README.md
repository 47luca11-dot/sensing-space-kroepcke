# Gaussian Splat Viewer

Ein schlanker Browser-Viewer für 3D Gaussian Splats auf Basis von **Vite + TypeScript**, [Three.js](https://threejs.org) und [Spark](https://sparkjs.dev) (`@sparkjsdev/spark`).

## Warum Spark?

- Spark wird aktiv entwickelt (World Labs und Community) und integriert sich direkt in Three.js (`SplatMesh` ist ein normales `Object3D`).
- Spark unterstützt `.ply`, `.spz`, `.splat`, `.ksplat` sowie SOG/`.zip`.
- Die Alternative `@mkkellogg/gaussian-splats-3d` ist laut eigener README nicht mehr in aktiver Entwicklung, empfiehlt selbst Spark und kann kein `.spz` laden.

## Voraussetzungen

- Node.js ≥ 20 (getestet mit Node 24 LTS)

## Starten

```bash
npm install
npm run dev
```

Danach die von Vite angezeigte Adresse öffnen, standardmäßig http://localhost:5173.

Für einen Produktions-Build:

```bash
npm run build
npm run preview
```

## Nutzung

| Aktion | So geht's |
| --- | --- |
| Per URL laden | URL ins Eingabefeld einfügen und „Laden“ klicken. Alternativ als Query-Parameter: `http://localhost:5173/?url=https://…/scene.ply` |
| Lokale Datei | „Datei wählen…“ klicken oder die Datei ins Browserfenster ziehen |
| Kamera | Linke Maustaste: drehen · Rechte Maustaste: verschieben · Mausrad: zoomen |
| Kamera zurücksetzen | Richtet die Kamera wieder auf den dichten Kern der Szene aus. Weit entfernte Hintergrund-Splats, wie sie z. B. Scaniverse als Hülle um den Scan exportiert, werden dabei ignoriert. |
| „Y-Achse drehen“ | Dreht die Szene um 180° um X. Viele 3DGS-Dateien (z. B. aus dem INRIA-Training) nutzen die OpenCV-Konvention mit Y nach unten. Wenn eine Szene auf dem Kopf steht, die Option umschalten. |

### Objekt freistellen (z. B. Flasche vom Tisch)

1. „Objekt anklicken…“ drücken und dann das Objekt in der Szene anklicken.
2. Der Viewer arbeitet das in drei Schritten ab:
   - **Tischebene:** RANSAC sucht um den Klickpunkt die dominante, annähernd waagerechte Ebene unter dem Objekt.
   - **Tisch entfernen:** Alles auf und unter dieser Ebene fällt weg. Raue Tischreste neben dem Objekt werden über dessen Umriss ebenfalls entfernt.
   - **Region-Growing:** Übrig bleiben nur die Splats, die mit dem Klickpunkt verbunden sind.
3. Feinjustierung:
   - **Abstand zur Tischebene:** höher stellen, wenn noch Tisch übrig ist. Niedriger stellen, wenn der Objektboden fehlt.
   - **Verbindungsabstand:** höher stellen, wenn Teile des Objekts fehlen. Niedriger stellen, wenn Nachbarobjekte mitkommen.
4. „Auswahl als .ply“ lädt das freigestellte Objekt herunter. Bei PLY-Quellen werden die Original-Zeilen 1:1 übernommen, inklusive Spherical Harmonics.

Die Freistellung setzt voraus, dass die Szene aufrecht steht (ggf. „Y-Achse drehen“ umschalten).

### Komponenten (zuschaltbare 3D-Elemente)

Im Abschnitt „Komponenten“ lassen sich klassische 3D-Objekte zur Splat-Szene hinzufügen. Jede Komponente hat eine Checkbox zum An- und Ausschalten und eigene Einstellungen.

- **Tisch:** wird automatisch unter das sichtbare bzw. freigestellte Objekt gestellt. Die Oberkante der Platte liegt genau an dessen Unterkante.
  - Form: eckig oder rund. Größe: Faktor 0,5–2×.
  - Kontaktschatten: ein weicher Schatten unter dem Objekt, denn Splats werfen selbst keinen Schatten.
  - Material für Tischplatte und Gestell jeweils getrennt einstellbar.
- **Boden:** große Fläche unter dem Tisch (bzw. direkt unter dem Objekt, wenn der Tisch aus ist), ebenfalls mit Material.

Materialeinstellungen:
- **Materialien:** Holz, Marmor, Metall (gebürstet), Beton, Lack. Die Texturen werden prozedural erzeugt, es sind keine Bilddateien nötig.
- **Anpassbar:** Farbe, Rauheit und Metallisch.

Die Maße sind auf metrische Scans ausgelegt (z. B. Scaniverse: Tisch 1,2 × 0,75 m, 74 cm hoch). Bei größeren Objekten oder anderen Einheiten wachsen sie automatisch mit.

Neue Komponenten werden in `src/components.ts` als Klasse mit dem Interface `SceneComponent` angelegt und in `main.ts` in die Liste `components` eingetragen.

### Geometrie deutlicher machen

- **Schattiert:** zeigt die Form als neutrales „Ton“-Modell. Die Normale jedes Splats ist seine flachste Achse.
- **Normalen:** färbt die Splats nach Oberflächenrichtung.
- **Splat-Größe:** verkleinert alle Gaussians. Die Struktur wird schärfer, die Oberfläche aber löchriger.
- **Mindest-Deckkraft:** blendet halbtransparente Schleier und Floater aus.

Hinweis: Transparente Objekte wie eine PET-Flasche liefern grundsätzlich unruhige Normalen. Die schattierte Ansicht wirkt dort körniger als bei matten Oberflächen.

Unterstützte Formate: `.ply`, `.splat`, `.spz` (zusätzlich `.ksplat`, `.sog`/`.zip`). `.ply` und `.spz` erkennt Spark am Dateiinhalt. `.splat` und `.ksplat` werden über die Dateiendung erkannt.

**Hinweis zu URLs:** Der Server der Datei muss CORS erlauben (`Access-Control-Allow-Origin`). Eigene Dateien kann man auch in den Ordner `public/` legen und dann als `/meine-szene.ply` laden.

## Standard-Szene

Ohne `?url=` lädt der Viewer `https://sparkjs.dev/assets/splats/butterfly.spz`, eine kleine öffentliche Beispiel-Szene aus der Spark-Dokumentation (ca. 4 MB). Die Szene wird über `DEFAULT_URL` in `src/main.ts` festgelegt.

Weitere öffentliche Testdateien, z. B. im `.splat`-Format: `https://huggingface.co/cakewalk/splat-data/resolve/main/nike.splat`

## Struktur

```
index.html      UI-Gerüst (Canvas, Panel, Drop-Overlay)
src/main.ts       Szene, Spark-Renderer, Orbit-Controls, Laden, UI-Verdrahtung
src/splatData.ts  Entpackte Splat-Daten, Darstellungsmodi, robuste Bounding-Berechnung
src/segment.ts    Klick-Picking, Tischebene (RANSAC), Region-Growing
src/plyExport.ts  PLY-Export der Auswahl
src/components.ts Zuschaltbare Komponenten (Tisch, Boden) und ihre Platzierung
src/materials.ts  Material-Presets, prozedurale Texturen, Material-Editor
src/style.css     Styles
public/scans/     Lokale Testdateien (nicht versioniert), erreichbar als /scans/<datei>.ply
```

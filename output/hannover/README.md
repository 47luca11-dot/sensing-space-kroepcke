# Sensing Space – Spaziergang ab dem Kröpcke

Begehbares 3D-Modell der Hannoveraner Innenstadt im Browser. Start an der Kröpcke-Uhr; man läuft in Third-Person als gescannter Charakter (Carlos, Gaussian Splats) durch die Stadt – gewichtig, träge, mit Gleichgewicht, Ausdauer, Stolpern und Gelände-Scan.

## Starten

Voraussetzung: [Node.js](https://nodejs.org/) ≥ 18.

```bash
cd web
node server.mjs 5177
```

Dann im Browser öffnen: <http://localhost:5177>

> Der kleine Server (`web/server.mjs`) ist nötig, weil das Spiel über `/__exists` prüft, welche Stadt-Dateien vorhanden sind. Ein reiner Static-Host (z. B. GitHub Pages) reicht ohne Anpassung nicht.

## Steuerung

| Taste | Aktion |
|---|---|
| W A S D | gehen |
| Shift | laufen |
| C | vorsichtig gehen |
| Q / E | Gleichgewicht halten |
| F | Gelände-Scan |
| Leertaste | Hopser |
| Maus / ← → | Kamera drehen (Bild ↑/↓ neigen) |

## Aufbau

- `web/` – das Spiel (three.js 0.180, Spark 2.4 für Gaussian Splats, three-mesh-bvh für Kollision). Stadt-Assets als meshopt-komprimierte GLBs unter `web/assets/city/`, gelistet in `manifest.json`.
- `pipeline/` – Python-Generatoren für Gebäude/Fassaden, Boden, Beschilderung und Wahrzeichen (Café Kröpcke, Kröpcke-Center, Marktkirche, Leineschloss, Altes Rathaus, Ballhof, Leibnizhaus, Anzeiger-Hochhaus …).
- `blender/` – Blender-/Rhino-Skripte (headless) für Exporte, Vorschauen und die Gesamtszene.
- `GOAL_PROMPT.md` – Projektziel und getroffene Entscheidungen; `BEGEHUNGSLISTE.md` – Begehungs-/QA-Liste.

- `data/` – Quellen und Roh-Exporte (LoD2, DGM, DOP, OSM, Rhino-Exporte), große Dateien über Git LFS.
- `pipeline/build_artifact_demo.mjs` – baut eine statische Demo für ein claude.ai-Artifact.

## Koordinatenframe

Meter, Y oben, Ursprung = Kröpcke-Uhr (EPSG:25832 E 550298.88 / N 5802940.71) auf Straßenniveau 56,16 m DHHN92; +X = Ost, −Z = Nord.

## Datenquellen & Lizenzen

- Geobasisdaten: © LGLN (LoD2, DGM1, TrueDOP10), Lizenz [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 3D-Stadtmodell LoD2: © Landeshauptstadt Hannover
- Kartendaten: © [OpenStreetMap](https://www.openstreetmap.org/copyright)-Mitwirkende, Lizenz ODbL
- Spark (Gaussian-Splat-Renderer): siehe `web/vendor/spark/LICENSE`

# Sensing Space – Spaziergang ab dem Kröpcke

Begehbares 3D-Modell der Hannoveraner Innenstadt im Browser. Start an der Kröpcke-Uhr; man läuft in Third-Person als gescannter Charakter (Carlos, Gaussian Splats) durch die Stadt – gewichtig, träge, mit Gleichgewicht, Ausdauer, Stolpern und Gelände-Scan.

## Einrichten

Große Dateien (`.glb`, `.ply`, `.blend`, `.3dm`, `.npy`, …) liegen in **Git LFS**. Vor dem Klonen einmal `git lfs install` ausführen:

```bash
git lfs install
git clone https://github.com/47luca11-dot/sensing-space-kroepcke.git
```

Der Klon ist ca. 7 GB groß. Nur den Code ohne LFS-Dateien: `GIT_LFS_SKIP_SMUDGE=1 git clone …`, später gezielt nachladen, z. B. `git lfs pull --include="web/**"`.

Benötigt: [Node.js](https://nodejs.org/) ≥ 18, Python 3.11 (Pipeline, Scan-Viewer), Blender 5.0, Rhino 8 (Exporte aus `.3dm`).

## Starten

```bash
node web/server.mjs 5177
```

Dann im Browser öffnen: <http://localhost:5177>. Startbildschirm mit Figurenwahl: `node startbildschirm/server.mjs 5175`. Mit Claude Code: Dev-Server aus `.claude/launch.json` (siehe `CLAUDE.md`).

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
- `data/` – Quellen und Roh-Exporte (LoD2, DGM, DOP, OSM, Rhino-Exporte).
- `sources/` – Ausgangsmodelle: eigenes Rhino-Modell Opernplatz/Kiosk (+ Texturen), TIB-Modell, LoD2-Stadtmodell Hannover (CityGML).
- `startbildschirm/` – Startbildschirm mit Figurenwahl (Carlos, Andis, Luca).
- `scan_viewer/` – Werkzeug zum Hochladen, Extrahieren, Bereinigen und Vernetzen von PLY-Scans; `uploads/` = Original-Scans, `extracted/` = Ergebnisse. Einrichten: `cd scan_viewer && uv venv --python 3.11 .venv && uv pip install -r requirements.txt`, starten: `.venv/Scripts/python server.py 5176`.
- `figuren/` – Figuren-Zwischenstände (Punktwolken, Meshes, GLBs, Blender-Datei der Startbildschirm-Figuren).
- `.claude/` – Claude-Code-Konfiguration: `launch.json`, three.js-Skills, `memory/` (Projektnotizen).
- `pipeline/build_artifact_demo.mjs` – baut eine statische Demo für ein claude.ai-Artifact.
- `GOAL_PROMPT.md` – Projektziel und getroffene Entscheidungen; `BEGEHUNGSLISTE.md` – Begehungs-/QA-Liste.

Nicht im Repo (werden neu erzeugt): `scan_viewer/.venv/`, `__pycache__/`, `artifact_demo/`.

## Koordinatenframe

Meter, Y oben, Ursprung = Kröpcke-Uhr (EPSG:25832 E 550298.88 / N 5802940.71) auf Straßenniveau 56,16 m DHHN92; +X = Ost, −Z = Nord.

## Datenquellen & Lizenzen

- Geobasisdaten: © LGLN (LoD2, DGM1, TrueDOP10), Lizenz [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 3D-Stadtmodell LoD2: © Landeshauptstadt Hannover
- Kartendaten: © [OpenStreetMap](https://www.openstreetmap.org/copyright)-Mitwirkende, Lizenz ODbL
- Spark (Gaussian-Splat-Renderer): siehe `web/vendor/spark/LICENSE`

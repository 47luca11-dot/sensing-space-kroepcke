# Sensing Space – Arbeitsordner

Kompletter Projektordner (Code, Rohdaten, Scans, Blender-/Rhino-Dateien, Claude-Konfiguration) für das Spiel **„Spaziergang ab dem Kröpcke“** und die zugehörigen Werkzeuge.

## Einrichten

Große Dateien (`.glb`, `.ply`, `.blend`, `.3dm`, `.npy`, …) liegen in **Git LFS**. Vor dem Klonen einmal:

```bash
git lfs install
git clone https://github.com/47luca11-dot/sensing-space-kroepcke.git
```

Der Klon ist mehrere GB groß. Nur den Code ohne LFS-Dateien: `GIT_LFS_SKIP_SMUDGE=1 git clone …`, später einzelne Ordner mit `git lfs pull --include="output/hannover/web/**"` nachladen.

Benötigt: [Node.js](https://nodejs.org/) ≥ 18, Python 3.11 (für `scan_viewer` und die Pipeline), Blender 5.0, Rhino 8 (für Exporte aus `.3dm`).

## Ordner

| Ordner | Inhalt | Starten |
|---|---|---|
| `output/hannover/` | **Das Spiel**: begehbares 3D-Hannover ab der Kröpcke-Uhr (`web/`), Python-Pipeline (`pipeline/`), Blender-Skripte und `.blend` (`blender/`), Rohdaten und Exporte (`data/`). Details: `output/hannover/README.md`, `GOAL_PROMPT.md` | `node output/hannover/web/server.mjs 5177` → http://localhost:5177 |
| `output/startbildschirm_demo/` | Startbildschirm mit Figurenwahl (Carlos, Andis, Luca) | `node output/startbildschirm_demo/server.mjs 5175` |
| `output/scan_viewer/` | PLY-Scans hochladen, extrahieren, bereinigen, vernetzen; `uploads/` = Original-Scans, `extracted/` = Ergebnisse | `cd output/scan_viewer && uv venv --python 3.11 .venv && uv pip install -r requirements.txt`, dann `.venv/Scripts/python server.py 5176` |
| `output/*.glb, *.ply, *.blend` | Figuren-Zwischenstände (Carlos = person1, Andis, Luca) | – |
| `sources/` | Ausgangsmodelle: eigenes Rhino-Modell Opernplatz/Kiosk, TIB-Modell, LoD2-Stadtmodell Hannover (CityGML) | – |
| `gaussian-splat-viewer/` | Viewer für Gaussian Splats (Vite + Spark) | `npm install && npm run dev` |
| `escape-room/` | Escape-Room-Spiel Architekturbüro (Vite + Three.js + Rapier) | `npm install && npm run dev` |
| `.claude/` | Claude-Code-Konfiguration: `launch.json` (Dev-Server), `skills/` (three.js-Skills), `memory/` (Projektnotizen für Claude) | – |

`node_modules/`, `.venv/` und Build-Ausgaben sind nicht im Repo und werden mit den Befehlen oben neu erzeugt.

## Mit Claude Code weiterarbeiten

Claude Code im Wurzelordner öffnen. `.claude/launch.json` enthält die Dev-Server (hannover, startbildschirm, scan-viewer …). Die Projektnotizen in `.claude/memory/` beschreiben Koordinatenframe, Datenquellen und Pipeline-Entscheidungen. Siehe auch `CLAUDE.md`.

## Daten & Lizenzen

- Geobasisdaten: © GeoBasis-DE/LGLN (LoD2, DGM1, TrueDOP10), CC BY 4.0
- 3D-Stadtmodell LoD2: © Landeshauptstadt Hannover, CC BY 4.0
- Kartendaten: © OpenStreetMap-Mitwirkende, ODbL
- Personen-Scans (Carlos, Andis, Luca) und das TIB-Modell sind nur für dieses Projekt bestimmt.

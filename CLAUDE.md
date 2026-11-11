# Hinweise für Claude Code

- Überblick über Ordner und Startbefehle: `README.md`.
- Projektnotizen (Koordinatenframe des Spiels, Datenquellen, Pipeline-Orte, getroffene Entscheidungen): `.claude/memory/` – vor Arbeiten am Hannover-Spiel oder an den Scan-Figuren lesen.
- Ziel und Regeln des Hannover-Spiels: `GOAL_PROMPT.md`.
- Regel: alle Meshes lückenlos, sauber und hochauflösend; Lücken sinnvoll schließen.
- Dev-Server über `.claude/launch.json` starten (hannover :5177, startbildschirm :5175, scan-viewer :5176).
- Große Binärdateien sind in Git LFS (`.gitattributes`); neue Dateitypen dort ergänzen, bevor sie committet werden.

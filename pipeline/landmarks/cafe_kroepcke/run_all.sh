#!/usr/bin/env bash
# Cafe Kroepcke: kompletter Lauf (Bau -> GLB/Blend -> meshopt -> Ersatz-JSON -> Vorschauen + Fotovergleiche)
# Aufruf: bash run_all.sh [--fast] [shots]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
BL="C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
PY="${PY:-C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/work/.venv/Scripts/python.exe}"
REFS="${REFS:-C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research/ref_images}"
CTX="${CTX:-C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/cafe/context.blend}"
RAW="$ROOT/data/landmarks_raw/cafe_kroepcke.glb"
WEB="$ROOT/web/assets/city/landmarks/cafe_kroepcke.glb"
BLEND="$ROOT/blender/landmarks/cafe_kroepcke.blend"
REPL="$ROOT/data/landmarks_replace/cafe_kroepcke.json"
PREV="$ROOT/blender/previews/lm_cafe_"
FAST="${1:-}"
SHOTS="${2:-ref,ref_day,foto2016,platz,clock,aerial,aerial_s,entrance,south,north,smallarches,roof_detail,roof_junction,top}"
mkdir -p "$(dirname "$RAW")" "$(dirname "$WEB")" "$(dirname "$BLEND")" "$(dirname "$REPL")"

cd "$HERE"
"$BL" -b --factory-startup -P build_cafe.py -- "$RAW" "$BLEND" > build_log.txt 2>&1
grep -E "CAFE|Traceback" build_log.txt
"$PY" make_replace.py "$REPL"
npx -y @gltf-transform/cli@4 meshopt "$RAW" "$WEB"
ls -la "$RAW" "$WEB"
"$BL" -b --factory-startup -P preview_cafe.py -- "$RAW" "$REPL" "$PREV" "$SHOTS" $FAST "--ctx=$CTX" > preview_log.txt 2>&1
grep -E "RENDERED|Traceback" preview_log.txt
[ -f "${PREV}ref.png" ] && "$PY" -I compare.py "$REFS/cafe_lockdown_2021.jpg" "${PREV}ref.png" "${PREV}vergleich_2021.png" "Foto 2021-01-03 (Wikimedia Commons)" "Modell (gleiche Kamera, Daemmerung)"
[ -f "${PREV}foto2016.png" ] && "$PY" -I compare.py "$REFS/cafe_moevenpick_2016.jpg" "${PREV}foto2016.png" "${PREV}vergleich_2016.png" "Foto Februar 2016 (Wikimedia Commons)" "Modell (rekonstruierte Kamera)"
[ -f "${PREV}clock.png" ] && "$PY" -I compare.py "$REFS/cafe_flaggen_2022.jpg" "${PREV}clock.png" "${PREV}vergleich_2022.png" "Foto 2022-03-06 (Wikimedia Commons)" "Modell (Blick von der Uhr)"
[ -f "${PREV}south.png" ] && "$PY" -I compare.py "$REFS/cafe_rueckseite_opernplatz.jpg" "${PREV}south.png" "${PREV}vergleich_opernplatz.png" "Foto Opernplatzseite (Wikimedia Commons)" "Modell (Suedgiebel)"
echo done

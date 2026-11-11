#!/usr/bin/env bash
# Marktkirche: Plan -> Texturen -> Geometrie -> Blender (GLB/.blend) -> Ersatz-JSON -> Strahlenpruefung -> webp+meshopt (Web) -> Vorschauen
# Aufruf: bash run_all.sh [shots|all|none] [--notex]
#   Vorher freien RAM pruefen (>= 5 GB), Blender laeuft headless mit --factory-startup.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
BL="C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
PY="${PY:-C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/work/.venv/Scripts/python.exe}"
RAW="$ROOT/data/landmarks_raw/marktkirche.glb"
WEB="$ROOT/web/assets/city/landmarks/marktkirche.glb"
BLEND="$ROOT/blender/landmarks/marktkirche.blend"
REPL="$ROOT/data/landmarks_replace/marktkirche.json"
PREV="$ROOT/blender/previews/lm_marktkirche_"
SHOTS="${1:-none}"
cd "$HERE"
mkdir -p build/tmp
[ -f build/plan_local.json ] || "$PY" analyse_plan.py > build/plan_log.txt
[ "$2" == "--notex" ] || "$PY" mk_tex.py > build/tex_log.txt
"$PY" mk_geom.py > build/geom_log.txt; grep -E "TOTAL" build/geom_log.txt
"$BL" -b --factory-startup -P mk_blender.py -- "$(pwd -W 2>/dev/null || pwd)/build" "$RAW" "$BLEND" > build/blender_log.txt 2>&1
grep -E "MK |Traceback|Error" build/blender_log.txt || true
"$PY" make_replace.py "$REPL"
"$PY" mk_check.py "$RAW" 400000 | grep -E "rays|by object" || true
"$PY" mk_zcheck.py "$RAW" 300000 0.012 | grep -E "rays|^  [0-9]" || true          # gestapelte Flaechen < 12 mm (Spielkamera near 0.05)
"$PY" mk_spotcheck.py "$RAW" 40000 0.012 | grep -E "TOTAL" || true                 # gezielt: Schwellen, Laterne, Gurte, Sohlbaenke ...
"$PY" mk_hygiene.py "$RAW" 0.002 | grep -E "TOTAL|coplanar" || true                 # degenerierte/Null-UV/nicht-mannigfaltige Flaechen + koplanare Flaechen (geometrisch)
# Web: Texturen -> WebP, Geometrie -> meshopt (Knotennamen bleiben erhalten: *_nocol*, *_noshadow*, Bleiglas)
# Zwischendateien in build/tmp (nicht im Web-Ordner), danach atomar umbenennen
npx -y @gltf-transform/cli@4 webp "$RAW" build/tmp/mk_webp.glb --quality 86 > build/web_log.txt 2>&1
npx -y @gltf-transform/cli@4 meshopt build/tmp/mk_webp.glb build/tmp/mk_web.glb >> build/web_log.txt 2>&1
cp build/tmp/mk_web.glb "$WEB.tmp" && mv -f "$WEB.tmp" "$WEB"
ls -la "$WEB"
if [ "$SHOTS" != "none" ]; then
  "$BL" -b --factory-startup -P mk_preview.py -- "$RAW" "$REPL" "$PREV" "$SHOTS" --samples=16 > build/preview_log.txt 2>&1
  grep -E "RENDERED|Traceback|Error" build/preview_log.txt || true
fi
echo done

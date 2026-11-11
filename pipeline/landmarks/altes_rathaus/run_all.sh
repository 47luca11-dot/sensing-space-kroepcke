#!/usr/bin/env bash
# Altes Rathaus Hannover -- komplette Kette (re-runnable):
#   Texturen -> Modell (build/altes_rathaus.glb) -> Strahlentest -> data/landmarks_raw (roh) -> web (webp + meshopt)
#   -> data/landmarks_replace/altes_rathaus.json -> Vorschauen + .blend (optional: ./run_all.sh --previews)
# Alle Dateien, die das Spiel laedt, werden atomar geschrieben (tmp + mv im selben Ordner).
set -e
cd "$(dirname "$0")"
ROOT="C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
PY="C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/work/.venv/Scripts/python.exe"
mkdir -p build/tex build/tmp
[ -f build/tex/brick_col.png ] && [ "$1" != "--tex" ] || "$PY" ar_textures.py build/tex
"$PY" build_rathaus.py build/altes_rathaus.glb | tail -5
"$PY" check_rays.py build/altes_rathaus.glb 3000 | sed -n '1,4p'
RAW="$ROOT/data/landmarks_raw/altes_rathaus.glb"
cp build/altes_rathaus.glb "$RAW.tmp" && mv -f "$RAW.tmp" "$RAW"
WEB="$ROOT/web/assets/city/landmarks/altes_rathaus.glb"
npx -y @gltf-transform/cli@4 webp build/altes_rathaus.glb build/tmp/ar_webp.glb --quality 86 > build/web_log.txt 2>&1
npx -y @gltf-transform/cli@4 meshopt build/tmp/ar_webp.glb build/tmp/ar_web.glb >> build/web_log.txt 2>&1
cp build/tmp/ar_web.glb "$WEB.tmp" && mv -f "$WEB.tmp" "$WEB"
ls -la "$RAW" "$WEB"
"$PY" make_replace.py
if [ "$1" == "--previews" ] || [ "$2" == "--previews" ]; then
  SHOTS=markt_front,markt_w,ne_gable,east,karmarsch,south,koebelinger,west_corner,schmiede_n,aerial,aerial_sw,top,close_window,close_gable,gable_niches,close_arcade,laube,arcade_in,dormer,pinnacles
  "$PY" make_previews.py $SHOTS --ctx --ctxr=170 --blend
  "$PY" make_compare.py
fi

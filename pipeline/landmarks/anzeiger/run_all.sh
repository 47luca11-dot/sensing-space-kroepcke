#!/usr/bin/env bash
# Anzeiger-Hochhaus + Goseriedebad: textures -> geometry -> Blender assemble (raw GLB + .blend) -> web GLB (webp + meshopt) -> replace json -> previews
# usage: bash run_all.sh [shots|all|none]
# hole finder (landmark only, magenta background): blender -b --factory-startup -P debug_view.py -- <raw.glb> <out_prefix> <cams.json>
# QA re-renders (context or --holes magenta): blender -b --factory-startup -P qa/qa_render.py -- <raw.glb> <prefix> qa/<shots>.json [--holes]
#   shot sets: qa/all_ctx.json (context), qa/all_holes.json + qa/shots_graze.json (grazing views along the Goseriede front) with --holes
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
BL="C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
PY="${PY:-C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/work/.venv/Scripts/python.exe}"
RAW="$ROOT/data/landmarks_raw/anzeiger.glb"
WEB="$ROOT/web/assets/city/landmarks/anzeiger.glb"
BLEND="$ROOT/blender/landmarks/anzeiger.blend"
PREV="$ROOT/blender/previews/lm_anzeiger_"
SHOTS="${1:-all}"
cd "$HERE"
waitmem() { for i in $(seq 1 30); do if powershell -c "exit ([int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt 5))"; then return 0; fi; echo "waiting for RAM"; sleep 60; done; }
"$PY" tex.py > build/tex_log.txt
"$PY" build.py | tee build/build_log.txt | head -3
"$PY" check.py > build/check_log.txt; sed -n 3p build/check_log.txt; tail -1 build/check_log.txt   # back-face-first rays + buried letters (exit 2)
"$PY" perimeter.py > build/perimeter_log.txt; head -1 build/perimeter_log.txt   # slits / gaps under walls along the outline (exit 1)
"$PY" ground_check.py > build/ground_log.txt; head -1 build/ground_log.txt    # slabs flickering against / sunk into the ground (exit 1)
"$PY" zfight.py > build/zfight_log.txt; head -1 build/zfight_log.txt          # coplanar same-facing overlaps (locally verified)
"$PY" ctx.py > build/ctx_log.txt
waitmem
"$BL" -b --factory-startup -P assemble.py -- "$RAW" "$BLEND" > build/assemble_log.txt 2>&1
grep -E "ASSEMBLE (total|wrote)|Traceback|Error" build/assemble_log.txt
"$PY" make_replace.py "$("$PY" -c "import json;print(round(json.load(open('build/anzeiger_meta.json'))['bounds'][1][1]+5,2))")"
TMPW="$HERE/build/anzeiger_webp_tmp.glb"          # intermediate stays out of the served folder
TMPM="$ROOT/web/assets/city/landmarks/anzeiger.tmp_new.glb"
# web VRAM budget (QA: 36 x 1024^2 maps ~200 MB): roughness/metal maps and the minor normal maps -> 512 px,
# colour maps of the low-detail materials -> 512 px; klinker/chevron col+nrm and stone/ashlar/tiles/copper col stay 1024
TMPR="$HERE/build/anzeiger_rs_tmp.glb"
npx -y @gltf-transform/cli@4 resize "$RAW" "$TMPR" --width 512 --height 512 --pattern "*_rgh" > build/web_log.txt 2>&1
npx -y @gltf-transform/cli@4 resize "$TMPR" "$TMPR" --width 512 --height 512 --pattern "{stone_dark,copper,copper_dark,concrete,roof_flat,slate,plaster,tiles,ashlar,stone}_nrm" >> build/web_log.txt 2>&1
npx -y @gltf-transform/cli@4 resize "$TMPR" "$TMPR" --width 512 --height 512 --pattern "{roof_flat,slate,plaster,copper_dark,concrete,stone_dark}_col" >> build/web_log.txt 2>&1
npx -y @gltf-transform/cli@4 webp "$TMPR" "$TMPW" >> build/web_log.txt 2>&1
rm -f "$TMPR"
npx -y @gltf-transform/cli@4 meshopt "$TMPW" "$TMPM" >> build/web_log.txt 2>&1
rm -f "$TMPW"; mv -f "$TMPM" "$WEB"
ls -la "$RAW" "$WEB"
if [ "$SHOTS" != "none" ]; then
  if [ "$SHOTS" = "all" ]; then S=""; else S="$SHOTS"; fi
  waitmem
  "$BL" -b --factory-startup -P preview.py -- "$RAW" "$PREV" $S > build/preview_log.txt 2>&1
  grep -E "RENDERED|Traceback|Error" build/preview_log.txt
fi
echo done

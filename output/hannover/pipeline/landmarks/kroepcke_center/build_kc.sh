#!/usr/bin/env bash
# Kroepcke-Center -- komplette Pipeline (Git-Bash). Optional: Argument "nopreview" ueberspringt die Renderings.
set -euo pipefail
H="C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
HERE="$H/pipeline/landmarks/kroepcke_center"
PY="C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/work/.venv/Scripts/python.exe"
BL="C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
BUILD="$HERE/build"; TEX="$H/data/landmarks_raw/tex"
RAW="$H/data/landmarks_raw/kroepcke_center.glb"
WEB="$H/web/assets/city/landmarks/kroepcke_center.glb"
BLEND="$H/blender/landmarks/kroepcke_center.blend"
REPL="$H/data/landmarks_replace/kroepcke_center.json"
mkdir -p "$BUILD" "$H/blender/landmarks" "$H/data/landmarks_replace" "$H/web/assets/city/landmarks"

"$PY" "$HERE/kc_tex.py" "$TEX"                       # Texturen (kc_*.png)
"$PY" "$HERE/kc_geom.py" "$BUILD"                    # Geometrie -> build/kc_geo.npz + kc_geo.json
"$PY" "$HERE/kc_replace.py" "$REPL"                  # Ersetzungs-Grundriss fuer das alte Modell
"$BL" -b --factory-startup -P "$HERE/kc_blender.py" -- "$BUILD" "$TEX" "$RAW" "$BLEND"   # GLB (unkomprimiert) + .blend

# Web: Wiederholte Achsen -> EXT_mesh_gpu_instancing, Texturen WebP, Geometrie meshopt (Knotennamen bleiben)
T1="$BUILD/kc_inst.glb"; T2="$BUILD/kc_webp.glb"
npx -y @gltf-transform/cli@4 instance "$RAW" "$T1" --min 3
npx -y @gltf-transform/cli@4 webp "$T1" "$T2" --quality 88
npx -y @gltf-transform/cli@4 meshopt "$T2" "$WEB"
ls -la "$RAW" "$WEB"

if [ "${1:-}" != "nopreview" ]; then
  "$BL" -b --factory-startup -P "$HERE/kc_preview.py" -- "$RAW" "$REPL" "$H/blender/previews" "${KC_CTX_CACHE:-$BUILD/ctx_cache.blend}"
fi

#!/usr/bin/bash
# Wartet auf das Ende der Kachel-Erzeugung (build_log3.txt) und auf >= 5 GB freien RAM, dann facades2_*-Vorschauen (Blender headless).
HERE="$(cd "$(dirname "$0")" && pwd)"
until grep -q "fertig in" "$HERE/build_log3.txt" 2>/dev/null; do sleep 10; done
RAW="$(cygpath -w "$HERE/../../../data/buildings_raw")"
CAMS="$(cygpath -w "$HERE/preview_cams2.json")"
bash "$HERE/blender_when_free.sh" "$RAW" --cams "$CAMS" "$@"

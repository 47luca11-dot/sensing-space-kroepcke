#!/usr/bin/bash
# Wartet, bis >= 5 GB RAM frei sind (Pruefung alle 60 s), dann Blender-Vorschau (headless) mit den uebergebenen Argumenten.
for i in $(seq 1 60); do
  FREE=$(powershell -NoProfile -c "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" | tr -d '\r' | tr ',' '.')
  echo "free GB: $FREE"
  if awk "BEGIN{exit !($FREE >= 5.0)}"; then break; fi
  sleep 60
done
"/c/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b --factory-startup -P "$(dirname "$0")/../blender_preview.py" -- "$@"

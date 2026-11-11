---
name: meshes-always-gapfree-highres
description: "For every 3D-scan task, always clean all meshes and fill every gap; meshes must be high-resolution and hole-free"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b8a76d7d-27b8-490e-b6de-334a309ea42a
  modified: 2026-11-05T17:06:36.769Z
---

For every task involving the 3D scans / characters (startbildschirm_demo, scan_viewer), always re-clean ALL meshes and fill all gaps, producing high-resolution, gap-free (watertight, no holes, no floaters) meshes, even when the task itself is about something else.

**Why:** the user said explicitly (2026-11-05): "füll immer bei jeder aufgabe die ich dir hier gebe alle meshes auf und bereinige sie, es sollen hochauflösende, lückenfreie meshes sein". Earlier low-poly / holey meshes and visible filler artifacts (e.g. Carlos' head) annoyed them.

**How to apply:** at the end of each scan/character task, rebuild meshes for all characters (output/*_mesh.ply → rig → GLB → splat weights) and the scan_viewer extracts, verify watertight + 0 boundary edges + visual check, and report it. Related: [[scan-pipeline]]

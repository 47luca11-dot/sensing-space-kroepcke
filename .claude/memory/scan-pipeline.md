---
name: scan-pipeline
description: "Layout of the Gaussian-splat character pipeline (start screen on :5175, scan viewer on :5176) and its non-obvious steps"
metadata:
  node_type: memory
  type: project
  originSessionId: b8a76d7d-27b8-490e-b6de-334a309ea42a
  modified: 2026-11-05T17:06:49.158Z
---

Project (since 2026-11-05) in `Documents/Sensing Space/Claude Code/output/`:
- `startbildschirm_demo/` — game start screen (three.js + vendored Spark 2.4), server `server.mjs`, launch config "startbildschirm" port 5175. Characters = original splats skinned to a Rigify rig (`assets/<id>_splats.ply`, `<id>_skin.bin`, `<id>.glb`) over a hidden "filler" mesh shrunk ~2 cm under the splats; two render passes (mesh, then splats on top).
- `scan_viewer/` — PLY upload/extract/clean/mesh tool, Python server port 5176 (launch config "scan-viewer"), pipeline `extract.py` + `mesh.py` + `process.py`, `batch.py`.
- Character build scripts live in the session scratchpad `work/` (merge2.py, extract_generic.py, mesh_vox.py, shrink.py, fixhead.py, bl_rig.py + rigs.json, splats.py) — they are not in the project; if the scratchpad is gone they must be recreated.
- Blender 5.0 is run headless (`blender.exe -b --factory-startup -P bl_rig.py`); the Blender MCP addon was not connected.
- Characters: person1 = "Carlos" (Bothfeld body + Centre head), andis, (planned) luca from "Unbenannter Scan.ply".

Related: [[meshes-always-gapfree-highres]]

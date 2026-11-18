---
name: scan-pipeline
description: "Layout of the Gaussian-splat character pipeline (start screen on :5175, scan viewer on :5176) and its non-obvious steps"
metadata:
  node_type: memory
  type: project
  originSessionId: b8a76d7d-27b8-490e-b6de-334a309ea42a
  modified: 2026-11-18T16:46:34.710Z
---

Project (since 2026-11-05); since 2026-11-11 moved into `Documents/Sensing Space/Claude Code/output/hannover/`:
- `startbildschirm/` (was output/startbildschirm_demo) — game start screen (three.js + vendored Spark 2.4), server `server.mjs`, launch config "startbildschirm" port 5175. Characters = original splats skinned to a Rigify rig (`assets/<id>_splats.ply`, `<id>_skin.bin`, `<id>.glb`) over a hidden "filler" mesh shrunk ~2 cm under the splats; two render passes (mesh, then splats on top).
- `scan_viewer/` (was output/scan_viewer) — PLY upload/extract/clean/mesh tool, Python server port 5176 (launch config "scan-viewer"), pipeline `extract.py` + `mesh.py` + `process.py`, `batch.py`.
- Character build scripts live in the session scratchpad `work/` (merge2.py, extract_generic.py, mesh_vox.py, shrink.py, fixhead.py, bl_rig.py + rigs.json, splats.py) — they are not in the project; if the scratchpad is gone they must be recreated.
- Blender 5.0 is run headless (`blender.exe -b --factory-startup -P bl_rig.py`); the Blender MCP addon was not connected.
- Character intermediates (glb/ply/blend) in `figuren/`. Characters: person1 = "Carlos" (Bothfeld body + Centre head), andis, (planned) luca from "Unbenannter Scan.ply".
- Textured mesh scans (KIRI Engine OBJ+MTL+JPG) go in via `scan_viewer/import_mesh.py` (in the project, needs `triangle` + `manifold3d`). It writes the viewer format (uploads/<id>_<name>.ply = whole scan as surface splats, `<name>_quelle/` = original OBJ, extracted/…_extrahiert.ply|_mesh.ply|_mesh.glb|.json); the viewer shows the textured GLB in the Mesh view and blocks "Extrahieren" for these. Imported 2026-11-11: `f70f76_KIRI Person Daunenjacke`, `b7daf1_KIRI Person Weste`.
- Non-obvious with KIRI: arbitrary scale → scaled to 1.80 m (cross-checked with ~30 cm shoes); photogrammetry often fills the hidden space between the legs from the front ("Beinbrücke") → import_mesh detects it by horizontal slices (legs separate low, back notch up to the crotch) and cuts a wedge by boolean. A real crotch (legs separate above 0.3 H) must not trigger it.
- `server.py` honours a `PORT` env var (preview tool moves it when 5176 is taken by another session's older server); on Windows it no longer silently shares a busy port. In a hidden Browser pane `requestAnimationFrame` stops: capture via `window.__viewer.renderer.render(...)` + `toDataURL`.

- Since 2026-11-18: rigging with finger skeleton runs IN THE LIVE BLENDER via MCP (user explicitly wants Blender MCP used): `.claude/skills/figur-einbauen/scripts/figur_finger.py` (body hull from splats, 61 bones incl. fingers, relaxed fingers as rest pose) started from a `bpy.app.timers` callback with log file; pop `skelett` from sys.modules after edits. Characters bv9/bv10/bv11 (Bothfeld-Vahrenheide 9–11) built this way and registered (carlos.js FIGS, startbildschirm CHARACTERS/PLAYABLE). In-game figures are darkened (`FIG_DARK` in web/js/carlos.js).

Related: [[meshes-always-gapfree-highres]]
- Since 2026-11-18 evening: walk/run come from real motion capture (CMU, `figuren/_mocap/`, 07_01 walk, 35_17 jog) retargeted by `gang.py` + `mocap.py`; the user rejected hand-authored procedural gaits ("Pferd"). All bv figures are scan-only (`figur_finger.py`, `build/`); figuro-STL hybrid smeared arm textures — LEDERJACKE (bv9) is the quality reference. Don't trust stills alone: the user judges motion in real time.

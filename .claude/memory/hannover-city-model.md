---
name: hannover-city-model
description: "Hannover 3D city game project (Kröpcke start, localhost 5177) — frame, data sources, pipeline locations, decisions"
metadata:
  node_type: memory
  type: project
  originSessionId: b8a76d7d-27b8-490e-b6de-334a309ea42a
  modified: 2026-11-11T18:29:57.641Z
---

Started 2026-11-08: walkable 3D model of Hannover, starting at the Kröpcke, playable in browser with Carlos (Death-Stranding-like movement), later enriched with the user's own scans/textures. Detail target = the TIB Rhino model.

- Project folder (= git repo root since 2026-11-11): `output/hannover/` (web/ = game on localhost:5177, launch config "hannover"; pipeline/ = python generators; blender/ = Blender scripts, previews, .blend; data/ = sources + raw exports; startbildschirm/, scan_viewer/, figuren/, sources/ (copies of the Downloads models + Opernplatz textures) also live here).
- Game frame: metres, Y-up, origin = Kröpcke-Uhr (OSM node 348155428, EPSG:25832 E 550298.88 N 5802940.71) at street level 56.16 m DHHN92; +X east, −Z north. Rhino Opernplatz model → game: add (84.117, −0.16, 141.706) to glTF coords.
- Reused models: `sources/Opernplatz_Kiosk+Umgebung-Detail_3D_v1.3dm` (originally in Downloads) (built earlier, exported via Rhino 8 headless script `blender/rhino_export.py` → `data/rhino_export`, `web/assets/city/detail`), TIB `sources/tib_rv_190_tragwerk_final.3dm` (origin E 548986 N 5803674 Z 54.93; ~1.5 km NW of Kröpcke).
- Data: city LoD2 `sources/Stadtmodell_Hannover_CityGML_LOD2.zip` (DHHN92, storeys), LGLN LoD2/DGM1/DOP10 (CC BY 4.0) in scratchpad research folder, OSM via Overpass (avoid `nwr;>;`).
- City assets are listed in `web/assets/city/manifest.json` (built by `pipeline/build_manifest.py`); GLBs meshopt-compressed with `npx @gltf-transform/cli@4`.
- Rhino 8 (Educational) is installed and scriptable: `Rhino.exe /nosplash /runscript="_-RunPythonScript <py>" <file.3dm>`; many breps lack render meshes, so mesh in Rhino rather than rhino3dm.
- Never modify the user's live Blender file (Opernplatz kiosk) via the Blender MCP connector; use headless Blender 5.0.

- GitHub: public repo 47luca11-dot/sensing-space-kroepcke, shared with a student colleague for collaboration. Repo root = `output/hannover` (user asked for that); contains everything project-related incl. data/, scans, .blend, `.claude/` (launch.json with repo-relative paths, skills, copy of these memory notes). Escape-room/splat-viewer are not in it. A leftover empty, locked `.git` dir sits in the workspace root; big binaries via Git LFS (~6.8 GB, near the 10 GiB free quota). User wants ALL project files pushed, not a minimal set. No `gh` CLI; repo was created via browser.
- Artifact demo (confirmed working by user 2026-11-11): https://claude.ai/artifact/RiqMNWxkf784aCyqcZVXLK, built by `pipeline/build_artifact_demo.mjs [radius]` → `artifact_demo/`. Artifacts don't serve .glb/.bin/.ply and cap 256 MB/version, 16 MB/text file, 64 MB/publish → binaries shipped as base64 `.txt` parts + fetch shim; 300 m radius ≈ 235 MB; publish in ~5 batches.

Related: [[scan-pipeline]], [[meshes-always-gapfree-highres]]

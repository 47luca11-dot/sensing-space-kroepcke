# Punktwolke komplett erzeugen – ohne Blender-Oberfläche (für frische Klone: die Kacheln sind zu groß für Git LFS).
# Aufruf aus output/hannover (Blender 5.x):
#   blender -b --factory-startup -P pipeline/pointcloud/run_headless.py
#   blender -b --factory-startup -P pipeline/pointcloud/run_headless.py -- --nur-licht      (nur Licht neu backen)
# Ablauf: bl_sample (alle Stadt-GLBs abtasten -> data/pointcloud/stage, ~8 GB, ~10 min)
#         bl_tile   (32-m-Kacheln + 8-m-Zellen -> web/assets/city/pointcloud, ~3,3 GB, ~3 min)
#         bl_light  (Sonne/Schatten/AO/Himmel per Geometry-Nodes-Raycast in die Farben backen, ~5 min)
# Im Live-Blender laufen dieselben Skripte über das Blender-MCP per Timer (siehe Kopf der Einzelskripte).
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

def load(name):
    ns = {"__name__": name}
    exec(open(os.path.join(HERE, name), encoding="utf-8").read(), ns)
    return ns

t0 = time.time()
S = load("bl_sample.py")
import bpy
S["BL_VERSION"] = bpy.app.version_string
if "--nur-licht" not in argv:
    urls = S["job_list"]()
    print(f"[Punktwolke] Abtasten: {len(urls)} Dateien", flush=True)
    for i, u in enumerate(urls, 1):
        if os.path.exists(os.path.join(S["STAGE"], S["stage_name"](u))) and "--neu" not in argv:
            continue
        st = S["process_file"](u)
        if i % 20 == 0: print(f"  {i}/{len(urls)}  {u}  {st['pts']} Punkte", flush=True)
    T = load("bl_tile.py")
    print("[Punktwolke] Kacheln …", flush=True)
    print(T["build"](S), flush=True)

L = load("bl_light.py")
man = json.load(open(os.path.join(L["CITY"], "manifest.json"), encoding="utf-8"))
files = [f["url"] for f in man["files"] if f.get("kind") in ("visual", "both") and not L["RX_SKIP_FILE"].search(f["url"])]
print(f"[Punktwolke] Licht: Verdecker aus {len(files)} Dateien", flush=True)
L["OCC"]["nv"] = 0
for u in files: L["add_occluder_file"](u)
print("  Verdecker:", L["finish_occluder"](), flush=True)
L["build_nodes"]()
L["ST"]["batches"] = L["plan_batches"]()
nb = len(L["ST"]["batches"])
for k, b in enumerate(L["ST"]["batches"], 1):
    L["run_batch"](b)
    print(f"  Licht-Stapel {k}/{nb}", flush=True)
idx = L["ST"]["idx"]
idx["baked"] = {"by": "pipeline/pointcloud/bl_light.py (Blender Geometry Nodes Raycast, headless)", "sun": L["SUN_G"].round(4).tolist(),
                "date": time.strftime("%Y-%m-%d %H:%M")}
json.dump(idx, open(os.path.join(L["PCD"], "index.json"), "w"), separators=(",", ":"))
print(f"[Punktwolke] fertig in {round((time.time() - t0) / 60, 1)} min", flush=True)

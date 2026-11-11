"""Erzeugt den kompletten Boden neu (Ausbau 2, Radius 860 m):  python run_all.py [--from <schritt>] [--dry] [--previews]
Reihenfolge: terrain (DGM1 + Naht + Promenade-Oeffnungen + Leine) -> layout (OSM-Klassen, Altstadt-Pflaster, Loecher) ->
textures + textures2 (prozedurale Texturen) -> terrain_mesh -> structures (Hecken, Mauern, Rillenschienen, Markierungen) ->
ubahn (U-Bahn-Zugaenge + detail_replace) -> shoptex (Ladenschilder Ebene -1) -> promenade (Passerelle) -> transit (Hochbahnsteige,
Mendini-Haltestelle, Oberleitung) -> altstadt (Brunnen, Leine-Gelaender) -> furniture -> export (Kacheln, Kollision, Hoehenraster;
atomarer Austausch) -> tools/check_mesh.py.
--dry: export nur nach data/ground_cache/dry/ (nichts veroeffentlichen).  --previews: Blender-Vorschauen blender/previews/ground2_*.png
(Python-Skripte bitte OHNE -I starten: die Module importieren sich gegenseitig aus diesem Ordner)."""
import subprocess, sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['terrain.py', 'layout.py', 'textures.py', 'textures2.py', 'terrain_mesh.py', 'structures.py', 'ubahn.py', 'shoptex.py',
         'promenade.py', 'transit.py', 'altstadt.py', 'furniture.py', 'export.py']
BLENDER = r'C:/Program Files/Blender Foundation/Blender 5.0/blender.exe'
t0 = time.time()
start = STEPS.index(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 0
for s in STEPS[start:]:
    print('=== ', s, flush=True)
    args = [sys.executable, os.path.join(HERE, s)] + (['--dry'] if (s == 'export.py' and '--dry' in sys.argv) else [])
    r = subprocess.run(args, cwd=HERE)
    if r.returncode: sys.exit('step failed: ' + s)
if '--dry' not in sys.argv:
    subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'check_mesh.py')], cwd=HERE)
if '--previews' in sys.argv:
    tiles = [] if '--dry' not in sys.argv else [os.path.join(HERE, '..', '..', 'data', 'ground_cache', 'dry') + '/']
    for grp in ('kroepcke_ABCDIL', 'kroepcke_EFGH', 'passerelle', 'steintor', 'steintor_ub', 'altstadt'):   # getrennt: wenig RAM je Blender-Lauf
        sj = os.path.join(HERE, '..', '..', 'data', 'ground_cache', 'shots_%s.json' % grp)
        subprocess.run([sys.executable, os.path.join(HERE, 'shots.py'), grp, sj], cwd=HERE)
        if json.load(open(sj)):
            subprocess.run([BLENDER, '-b', '--factory-startup', '-P', os.path.join(HERE, 'preview2.py'), '--', sj] + tiles, cwd=HERE)
print('done in %.0f s' % (time.time() - t0))

"""Vorschauen des Ballhof-Ensembles rendern (Blender 5 headless, Eevee 1280x720, Backface-Culling an), mit RAM-Waechter.
Aufruf: python make_previews.py <shots|all> [--ctx] [--ctxr=140] [--prefix=<pfad/lm_ballhof_>] [--blend] [--glb=<pfad>]"""
import sys, os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
BL = "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
args = sys.argv[1:]
shots = args[0] if args else 'all'
prefix = next((a.split('=', 1)[1] for a in args if a.startswith('--prefix=')), ROOT + '/blender/previews/lm_ballhof_')
glb = next((a.split('=', 1)[1] for a in args if a.startswith('--glb=')), os.path.join(HERE, 'build', 'ballhof.glb'))
extra = [a for a in args[1:] if a.startswith('--ctx') or a.startswith('--res')]
if '--blend' in args:
    extra.append('--blend=' + ROOT + '/blender/landmarks/ballhof.blend')


def free_gb():
    out = subprocess.run(['powershell', '-NoProfile', '-c', '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out) / 1024 / 1024
    except ValueError:
        return 99.0


for k in range(120):
    f = free_gb()
    if f >= 5.0: break
    print('RAM frei %.1f GB < 5 GB, warte 60 s' % f, flush=True)
    time.sleep(60)
os.makedirs(os.path.dirname(prefix), exist_ok=True)
cmd = [BL, '-b', '--factory-startup', '-P', os.path.join(HERE, 'bh_preview.py'), '--', glb, prefix, shots] + extra
print(' '.join(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
for ln in (r.stdout + r.stderr).splitlines():
    if any(k in ln for k in ('RENDERED', 'saved', 'Error', 'Traceback', 'error', 'failed')): print(ln)

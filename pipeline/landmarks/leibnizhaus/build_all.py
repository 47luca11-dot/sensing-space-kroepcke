"""Baut das Landmark 'leibnizhaus' (Leibnizhaus + Noltehaus + Oskar-Winter-Brunnen + Beginenturm + Historisches Museum),
schreibt data/landmarks_raw/leibnizhaus.glb (unkomprimiert, Spielframe) und optional die Web-Kopie (meshopt + webp, atomar).
Aufruf: python build_all.py [--parts=lh,nh,fo,tw,mu] [--web] [--out=<glb>]"""
import json, os, shutil, subprocess, sys, time
import numpy as np
from PIL import Image
import lh_common as C
import lh_geom as G
import lh_glb
from lh_materials import MATS

T0 = time.time()
def log(*a): print('[build %5.1fs]' % (time.time() - T0), *a, flush=True)

args = sys.argv[1:]
PARTS = next((a.split('=', 1)[1] for a in args if a.startswith('--parts=')), 'lh,nh,fo,tw,mu').split(',')
OUT = next((a.split('=', 1)[1] for a in args if a.startswith('--out=')), os.path.join(C.DATA, 'landmarks_raw', 'leibnizhaus.glb'))
WEB = '--web' in args

# Texturen fuer den Export: Groesse / ORM je Material reduzieren
TEX_EXPORT = os.path.join(C.BUILD, 'tex_export')
SMALL = {'bushhammer', 'green_roof', 'plaster', 'plaster_grey', 'concrete', 'concrete_dark', 'slate_clad', 'wood_dark', 'wood_door',
         'bronze', 'granite', 'majolica', 'sand_plinth', 'sand_white', 'rubble_dark'}
NO_ORM = {'plaster', 'plaster_grey', 'concrete', 'concrete_dark', 'sand_fine', 'sand_plinth', 'sand_white', 'wood_dark', 'wood_door',
          'green_roof', 'granite', 'bushhammer', 'bronze', 'ashlar'}


def export_textures(used):
    os.makedirs(TEX_EXPORT, exist_ok=True)
    for f in os.listdir(TEX_EXPORT): os.remove(os.path.join(TEX_EXPORT, f))
    for t in sorted(used):
        for suf in ('col', 'nrm', 'orm'):
            src = os.path.join(C.TEX, '%s_%s.png' % (t, suf))
            if not os.path.exists(src): continue
            if suf == 'orm' and t in NO_ORM: continue
            im = Image.open(src)
            if t in SMALL: im = im.resize((512, 512), Image.LANCZOS)
            im.save(os.path.join(TEX_EXPORT, '%s_%s.png' % (t, suf)), optimize=True)


def build_parts():
    nodes = []
    if 'lh' in PARTS:
        import build_leibnizhaus as B
        mb = B.build(); nodes.append(('leibnizhaus', mb)); log('Leibnizhaus', mb.ntris())
    if 'nh' in PARTS:
        import build_noltehaus as B
        mb = B.build(); nodes.append(('noltehaus', mb)); log('Noltehaus', mb.ntris())
    if 'fo' in PARTS:
        import build_fountain as B
        mb = B.build(); nodes.append(('oskar_winter_brunnen', mb)); log('Brunnen', mb.ntris())
    if 'tw' in PARTS:
        import build_tower as B
        mb = B.build(); nodes.append(('beginenturm', mb)); log('Beginenturm', mb.ntris())
    if 'mu' in PARTS:
        import build_museum as B
        mb = B.build(); nodes.append(('historisches_museum', mb)); log('Museum', mb.ntris())
    return nodes


def main():
    nodes = build_parts()
    used = set()
    for _, mb in nodes:
        for m in mb.parts:
            t = MATS.get(m, {}).get('tex')
            if t: used.add(t)
    export_textures(used)
    gl_nodes = [{'name': n, 'mb': mb, 'extras': {'landmark': 'leibnizhaus', 'part': n}} for n, mb in nodes]
    tot = sum(mb.ntris() for _, mb in nodes)
    ntri, _ = lh_glb.write_glb(OUT, gl_nodes, MATS, TEX_EXPORT, extras={'landmark': 'leibnizhaus', 'triangles': tot})
    log('wrote', OUT, '%.1f MB' % (os.path.getsize(OUT) / 1e6), 'tris', ntri)
    lo = np.min([mb.bounds()[0] for _, mb in nodes], axis=0); hi = np.max([mb.bounds()[1] for _, mb in nodes], axis=0)
    log('bounds', np.round(lo, 2), np.round(hi, 2))
    json.dump({'triangles': tot, 'parts': {n: mb.ntris() for n, mb in nodes},
               'materials': sorted({m for _, mb in nodes for m in mb.parts}), 'bounds': [lo.tolist(), hi.tolist()]},
              open(os.path.join(C.BUILD, 'stats.json'), 'w'), indent=1)
    if WEB:
        dst = os.path.join(C.WEB, 'assets', 'city', 'landmarks', 'leibnizhaus.glb')
        tmp = os.path.join(C.BUILD, 'web_tmp.glb')
        cmd = 'npx -y @gltf-transform/cli@4 optimize "%s" "%s" --compress meshopt --texture-compress webp --simplify false --instance false' % (OUT, tmp)
        log(cmd)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])
        if r.returncode == 0 and os.path.exists(tmp):
            tmp2 = dst + '.tmp'
            shutil.copyfile(tmp, tmp2); os.replace(tmp2, dst)
            log('web', dst, '%.1f MB' % (os.path.getsize(dst) / 1e6))
        else:
            log('WEB EXPORT FAILED')


if __name__ == '__main__':
    main()

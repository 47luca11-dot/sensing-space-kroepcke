"""Materialtabelle + Export (MB -> GLB) fuer das Alte Rathaus."""
import os, json
import numpy as np
from ar_glb import write_glb, mat_to_trs
from ar_textures import TILES

MATS = {
    'AR_Backstein': dict(color=(1, 1, 1), rough=1.0, tex='brick', nscale=1.0),
    'AR_Glasur': dict(color=(1, 1, 1), rough=1.0, tex='brick_black', nscale=1.0),
    'AR_Backstein_Nische': dict(color=(0.66, 0.60, 0.58), rough=1.0, tex='brick', nscale=1.0),    # Blendnischen-Rueckwand (Eigenschatten)
    'AR_Backstein_innen': dict(color=(0.50, 0.46, 0.44), rough=1.0, tex='brick', nscale=1.0),     # verschattete Innenflaechen (Laube) -- ersetzt fehlende AO im Spiel
    'AR_Putz_innen': dict(color=(0.78, 0.76, 0.72), rough=1.0, tex='plaster'),
    'AR_Backstein_Fiale': dict(color=(1, 1, 1), rough=1.0, tex='brick_band', nscale=1.0),
    'AR_Backstein_Giebel': dict(color=(1, 1, 1), rough=1.0, tex='brick_gable', nscale=1.0),
    'AR_Wechsel': dict(color=(1, 1, 1), rough=1.0, tex='wechsel', nscale=1.0),       # Keilsteine/Gewaende rot-schwarz im Wechsel
    'AR_Backstein19': dict(color=(0.86, 0.80, 0.78), rough=1.0, tex='brick19', nscale=1.0),
    'AR_Formstein': dict(color=(0.72, 0.62, 0.50), rough=0.8, tex=None),
    'AR_Sandstein': dict(color=(1, 1, 1), rough=1.0, tex='sandstone', nscale=0.8),
    'AR_Quader': dict(color=(0.93, 0.86, 0.75), rough=1.0, tex='ashlar', nscale=0.8),       # warmer Sandstein (Foto 14)
    'AR_Dachziegel': dict(color=(1, 1, 1), rough=1.0, tex='roof', nscale=1.0),
    'AR_First': dict(color=(1, 1, 1), rough=1.0, tex='pantile_ridge'),
    'AR_Schiefer': dict(color=(0.72, 0.72, 0.74), rough=1.0, tex='slate'),
    'AR_Fialhelm': dict(color=(0.86, 0.74, 0.70), rough=1.0, tex='roof'),      # Fialhelme rot (Terrakotta, Fotos 01/07)
    'AR_Putz': dict(color=(1, 1, 1), rough=1.0, tex='plaster'),
    'AR_Granit': dict(color=(1, 1, 1), rough=1.0, tex='granite'),
    'AR_Fries': dict(color=(1, 1, 1), rough=1.0, tex='frieze', nscale=1.0),
    'AR_Vierpass': dict(color=(1, 1, 1), rough=1.0, tex='quatrefoil'),
    'AR_Rosette': dict(color=(1, 1, 1), rough=1.0, tex='rosette'),
    'AR_Masswerk_gruen': dict(color=(1, 1, 1), rough=1.0, tex='tracery'),
    'AR_Masswerk_rot': dict(color=(1, 1, 1), rough=1.0, tex='tracery_red'),
    'AR_Masswerk_ziegel': dict(color=(1, 1, 1), rough=1.0, tex='leaf_tiles'),      # Terrakotta-Blattplatten (Laubenbruestung, Foto 06)
    'AR_Fensterrahmen': dict(color=(0.23, 0.245, 0.255), rough=0.5),
    'AR_Fensterglas': dict(color=(0.15, 0.17, 0.19), rough=0.10, metal=0.35),
    # Ladenfronten (ar_shops.py): durchsichtiges Glas, beleuchteter Ladenraum (Vertexfarben je Ladenart), Tueren, Dekore
    'AR_Schaufenster_Glas': dict(color=(0.30, 0.34, 0.37), rough=0.05, metal=0.1, alpha=0.30),
    'AR_Glaswand_Glas': dict(color=(0.30, 0.34, 0.37), rough=0.05, metal=0.1, alpha=0.28),           # Glaswand Laubengang-Enden
    'AR_Laden': dict(color=(1, 1, 1), rough=0.85, vc=True, emit=(0.20, 0.19, 0.17)),           # Raumhuelle (Wand/Boden/Decke) hell beleuchtet
    'AR_Ladenrahmen': dict(color=(0.16, 0.17, 0.18), rough=0.35, metal=0.7),                     # Stahl-/Alu-Profile der Schaufenster
    'AR_Tuergriff': dict(color=(0.80, 0.81, 0.80), rough=0.2, metal=1.0),                        # Edelstahl Stossgriff
    'AR_Schild_Halter': dict(color=(0.12, 0.12, 0.13), rough=0.4, metal=0.8),
    'AR_Zink': dict(color=(0.40, 0.42, 0.43), rough=0.55, metal=0.6),
    'AR_Eisen': dict(color=(0.14, 0.14, 0.145), rough=0.55, metal=0.6),
    'AR_Kupfer': dict(color=(0.36, 0.55, 0.45), rough=0.5, metal=0.3),
    'AR_Tuer': dict(color=(0.29, 0.20, 0.15), rough=0.6),
    'AR_Tuer_gruen': dict(color=(0.12, 0.29, 0.25), rough=0.5),
    'AR_Dunkel': dict(color=(0.07, 0.065, 0.06), rough=0.95),
    'AR_Stahl': dict(color=(0.18, 0.19, 0.2), rough=0.45, metal=0.6),
    'AR_Glasdach': dict(color=(0.30, 0.36, 0.40), rough=0.1, metal=0.4),                  # Atrium-Glasdach (opak: darunter nur das Hofdach)
    'AR_Vordach_Glas': dict(color=(0.55, 0.62, 0.64), rough=0.05, metal=0.1, alpha=0.32),   # Glasvordaecher (Foto 05/10): durchsichtig, Spiel-RX_GLASS
    'AR_Innen': dict(color=(0.045, 0.042, 0.04), rough=1.0),                               # Innenschale (nur von innen sichtbar)
    'AR_Dachbahn': dict(color=(0.24, 0.24, 0.25), rough=0.9),
    'AR_col': dict(color=(1, 0, 1), rough=1.0),
}
for s in ('sparkasse', 'stichnoth', 'viviry', 'atrium', 'white'):
    MATS['AR_Dekor_' + s] = dict(color=(1, 1, 1), rough=0.35, tex='decal_' + s, alpha_mode='MASK', cutoff=0.5)
for s in ('sparkasse', 'stichnoth', 'viviry', 'atrium', 'ratskeller'):
    MATS['AR_Schild_' + s] = dict(color=(1, 1, 1), rough=0.5, tex='sign_' + s, emit=(0.30, 0.30, 0.30), emit_tex=True)
    MATS['AR_Ausleger_' + s] = dict(color=(1, 1, 1), rough=0.5, tex='blade_' + s, emit=(0.30, 0.30, 0.30), emit_tex=True)
# Auslagen-Prototypen der Fassaden-Pipeline (pipeline/facades/displays.py): Vertexfarben (linear), Materialien wie dort
_DSP = {'trim': dict(rough=0.5), 'metal': dict(rough=0.38, metal=0.7), 'interior': dict(rough=0.9),
        'shop_interior': dict(rough=0.85, emit=(0.16, 0.155, 0.145)),
        'glass': dict(rough=0.04, alpha=0.45, color=(0.05, 0.065, 0.075)),
        'glow': dict(rough=0.6, color=(0.8, 0.8, 0.78), emit=(0.95, 0.93, 0.86)),
        'glow_warm': dict(rough=0.6, color=(0.8, 0.7, 0.55), emit=(1.0, 0.78, 0.48))}
for k, v in _DSP.items():
    MATS['AR_dsp_' + k] = dict(dict(color=(1, 1, 1), metal=0.0), lin=True, vc=True, **v)


def dsp_material(name, props):
    """weiteres Material der Auslagen (z. B. glow_RRGGBB) aus gltf_io.MATERIALS uebernehmen"""
    key = 'AR_dsp_' + name
    if key not in MATS:
        MATS[key] = dict(color=tuple(props.get('base', (1, 1, 1))), rough=props.get('rough', 0.6), metal=props.get('metal', 0.0),
                         emit=tuple(props['emissive']) if 'emissive' in props else None, lin=True, vc=True)
        if 'alpha' in props: MATS[key]['alpha'] = props['alpha']
    return key

TEXMAP = {k: v['tex'] for k, v in MATS.items() if v.get('tex')}


def tiles_for_mats():
    t = {}
    for m, tex in TEXMAP.items():
        if tex in TILES: t[m] = TILES[tex]
    return t


def export(path, mb, protos, colmb=None, tex_dir='build/tex', extras=None, stats=None):
    nodes = []
    fin = mb.finish(colors=True)
    tot = 0
    groups = {}
    for m, (P, N, U, I, C) in sorted(fin.items()):
        nodes.append({'name': m + '_nocol', 'prims': [(m, P, N, U, I, C)]})
        tot += len(I); groups[m] = int(len(I))
    for pname, L in sorted(mb.inst.items()):
        if pname not in protos or not L: continue
        pf = protos[pname].finish(colors=True)
        prims = [(m, P, N, U, I, C) for m, (P, N, U, I, C) in sorted(pf.items())]
        T = []; R = []; S = []
        for M in L:
            t, q, s = mat_to_trs(M); T.append(t); R.append(q); S.append(s)
        nodes.append({'name': pname + '_inst_nocol', 'prims': prims, 'inst': (np.array(T), np.array(R), np.array(S))})
        n = sum(len(p[4]) for p in prims) * len(L)
        tot += n; groups[pname + ' (x%d)' % len(L)] = int(n)
    if colmb is not None:
        cf = colmb.finish()
        for m, (P, N, U, I) in cf.items():
            nodes.append({'name': 'col_altes_rathaus', 'prims': [(m, P, N, U, I)]})
    size = write_glb(path, nodes, MATS, tex_dir, extras=extras)
    if stats is not None:
        stats.update({'tris': int(tot), 'groups': groups, 'bytes': int(size)})
    return tot

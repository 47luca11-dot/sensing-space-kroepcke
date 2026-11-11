"""Kamerastandpunkte fuer die Vorschauen (Lokalrahmen -> Spielframe); Augenhoehe = Boden (Hoehenraster) + 1.6 m.
Ausgabe build/cams.json: name -> [[x,y,z], [x,y,z], Brennweite]"""
import json, os
import numpy as np
from ls_common import *

# name: (Kamera (u, v, h | 'eye'), Ziel (u, v, h), Brennweite)
CAMS = {
    # Referenzfoto-Standpunkte
    'ref02_portikus_frontal': ((0.0, 18.0, 'eye'), (0.0, -3.0, 9.0), 20),
    'ref04_treppe_nah': ((-8.5, 7.5, 'eye'), (3.0, -2.0, 3.0), 18),
    'ref06_hannah_arendt_platz': ((30.0, 38.0, 'eye'), (-2.0, -3.0, 8.0), 24),
    'ref16_leinstrasse_nordecke': ((-74.0, 13.0, 'eye'), (-38.0, -10.0, 8.0), 22),
    'ref03_leinstrasse_sued': ((42.0, 7.0, 'eye'), (-20.0, -2.0, 8.0), 26),
    'ref01_leine_suedost': ((34.0, -93.0, 'eye'), (-25.0, -70.0, 6.0), 24),
    'refx18_wehrbruecke': ((62.0, -98.0, 'eye'), (8.0, -42.0, 6.0), 22),
    'refflickr04_ufer_nordwest': ((-76.0, -88.0, 'eye'), (-8.0, -73.0, 4.0), 24),
    'ref13_nordwestfluegel': ((-128.0, -80.0, 'eye'), (-60.0, -66.0, 6.0), 24),
    # Uebersicht
    'luftbild_nordost': ((45.0, 125.0, 115.0), (-32.0, -38.0, 4.0), 30),
    'luftbild_suedwest': ((-70.0, -215.0, 105.0), (-32.0, -38.0, 4.0), 30),
    # Nahansichten
    'nah_kapitell': ((4.6, 2.6, 11.4), (2.35, -1.15, 11.7), 35),
    'nah_giebel': ((0.0, 14.0, 16.0), (0.0, -0.6, 16.4), 32),
    'nah_pronaos': ((5.0, -2.2, 'eye+0.35'), (-3.0, -7.0, 5.5), 18),
    'nah_nische_inschrift': ((-15.0, -0.5, 'eye'), (-15.0, -7.0, 2.3), 26),
    'nah_loggia': ((-27.0, -90.0, 'eye'), (-27.0, -73.0, 4.5), 26),
    'nah_plenarsaal_terrasse': ((40.0, -70.0, 'eye'), (28.0, -55.0, 6.0), 22),
    'nah_laves_portikus': ((-62.0, 7.0, 'eye'), (-51.0, -5.0, 3.0), 22),
    'nah_mansarde_gauben': ((-10.0, -86.0, 22.0), (-12.0, -66.0, 17.5), 30),
}


def make():
    hf = Heightfield()
    out = {}
    for k, (c, t, lens) in CAMS.items():
        cu, cv, ch = c
        x, z = local_to_game(cu, cv)
        if isinstance(ch, str):
            g = float(hf(x, z)); ch = g + 1.6 + (float(ch.split('+')[1]) if '+' in ch else 0.0)
        tx, tz = local_to_game(t[0], t[1])
        out[k] = [[float(x), float(ch), float(z)], [float(tx), float(t[2]), float(tz)], lens]
    json.dump(out, open(os.path.join(BUILD, 'cams.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    for k, v in make().items(): print(k, v)

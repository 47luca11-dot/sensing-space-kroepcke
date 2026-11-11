"""Deterministische Stilwahl je Gebaeude (Hash der ID + Hoehe/Geschosse/Dachform/Strasse/Funktion) und Stilparameter.

Stile: altbau (Gruenderzeit), nachkrieg (1950er), modern (Buero, Bandfenster/Lochfassade), curtain (Glasfassade),
kaufhaus (geschlossene Stein-/Plattenfassade + Schaufenster), gothic (Backsteinkirche), parking (Parkhaus),
fachwerk (Altstadt-Fachwerk), canopy (Ueberdachung), shed (Kiosk/Nebengebaeude), tower (mittelalterlicher Wehrturm).

Benannte Altstadthaeuser (landmark_notes.md / streetview_notes.json) bekommen feste Werte in OVERRIDES statt Zufall je Strasse."""
import math
import numpy as np
from common import rng_of, jitter_hex

RETAIL_STREETS = {'Georgstraße', 'Bahnhofstraße', 'Karmarschstraße', 'Große Packhofstraße', 'Kleine Packhofstraße', 'Rathenaustraße',
                  'Luisenstraße', 'Kröpcke', 'Ernst-August-Platz', 'Schillerstraße', 'Theaterstraße', 'Andreaestraße', 'Osterstraße',
                  'Niki-de-Saint-Phalle-Promenade', 'Kröpckepassage', 'Nordmannpassage', 'Galerie Luise', 'Schmiedestraße',
                  'Knochenhauerstraße', 'Kramerstraße', 'Marktstraße', 'Grupenstraße', 'Heiligerstraße', 'Seilwinderstraße',
                  'Limburgstraße', 'Kanalstraße', 'Steintorstraße', 'Goseriede', 'Schuhstraße', 'Am Markte', 'Holzmarkt',
                  'Lange Laube', 'Kurt-Schumacher-Straße', 'Raschplatz', 'Ständehausstraße', 'Windmühlenstraße', 'Bahnhofsplatz'}
# Wahrscheinlichkeit Fachwerk je Strasse (Street-View-Notizen: Kramerstr. Fachwerkzeile, Burgstr. rot gestrichenes Fachwerk,
# Holzmarkt/Ballhof; OSM building:architecture=timber_frame hat Vorrang)
FACHWERK_P = {'Burgstraße': 0.85, 'Kramerstraße': 0.7, 'Holzmarkt': 0.8, 'Ballhofplatz': 0.75, 'Ballhofstraße': 0.6, 'Kreuzkirchhof': 0.55,
              'Roßmühle': 0.5, 'Pferdestraße': 0.45, 'Kreuzstraße': 0.45, 'Knochenhauerstraße': 0.12, 'Am Markte': 0.15, 'Schuhstraße': 0.15,
              'Marstall': 0.3, 'Am Hohen Ufer': 0.2}
# Balkenfarben nach landmark_notes.md / Street-View-Notizen: meist dunkelbraun bis fast schwarz (#33261d), ochsenblutrot
# (#6e2a22, Burgstrasse 23a-27, Kramerstrasse), wenige grau gestrichen (#8f969a); Gefache meist weiss, selten ocker
TIMBER = ['#33261D', '#2A201A', '#3B2A20', '#4A3226', '#2E2622']
TIMBER_RED = ['#6E2A22', '#6B2420', '#7A2A22']
TIMBER_GREY = ['#8F969A', '#7E868B', '#9AA0A3']          # grau gestrichen (Notizen: #8f969a), kein gesaettigtes Blau
INFILL = ['#F0ECE2', '#F2EFE6', '#EDE8DC', '#F0EDE6', '#EDE3CF', '#E8E0D0', '#D8B878', '#E6D2A8']
ALTSTADT_PLASTER = ['#E8D7B0', '#D9C7A7', '#E4D2C6', '#EDE3CF', '#D8B898', '#C9B79C', '#E6E0D4', '#D4C4A8', '#E8C8A8', '#CFC2A8']
ALTSTADT_STREETS = {'Burgstraße', 'Kramerstraße', 'Knochenhauerstraße', 'Ballhofplatz', 'Holzmarkt', 'Am Markte', 'Kreuzkirchhof',
                    'Kreuzstraße', 'Roßmühle', 'Pferdestraße', 'Schuhstraße', 'Marktstraße', 'Hanns-Lilje-Platz', 'Kaiserstraße'}

# Paletten (sRGB-Hex; Spielversion verwendet lineare Vertexfarben)
PLASTER_WARM = ['#E6E0D4', '#D9C7A7', '#E3D3B8', '#C9B79C', '#F2F0EA', '#D8D0C0', '#E8D7B0', '#E4D2C6', '#DCCBB4', '#EDE3CF']
PLASTER_COOL = ['#F2F0EA', '#E6E0D4', '#E0E5E8', '#D6D6D2', '#E4E6E4', '#D8D0C0', '#EFE6C9', '#D3D9D0', '#E8E4DC', '#CFD3D2', '#E9E2D2']
PANEL_COLS = ['#CFD3D2', '#BEBEC1', '#D6DADD', '#C9C6BE', '#B8B0A2', '#D9D2C4']
TRAVERTINE = ['#E3D9C6', '#D9CFBC', '#E6DED0', '#CEC6B6']
BRICK_TINTS = ['#FFFFFF', '#F2E6E0', '#E8DCD6', '#D9C9C0']          # roter Backstein (Textur bringt die Farbe)
KLINKER_TINTS = ['#9C7C70', '#8E7568', '#A8826E', '#7E6A62']        # dunkler Klinker
SANDSTONE_TINTS = ['#FFFFFF', '#F4EEE2', '#EDE5D5', '#E6DDCB']
FRAME_DARK = ['#2F3337', '#3A3D40', '#22201F', '#4B4E52']
FRAME_WHITE = ['#F2F2EE', '#ECECE6', '#F5F3EC']
FRAME_WOOD = ['#5A3B28', '#4B3528', '#294738', '#682F32']
GLASS = ['#1E2A31', '#22303A', '#1B262D', '#26323A', '#2A3338']
PLINTH = ['#786F63', '#92897C', '#655F58', '#857E74', '#6E6A66']
SHOP_SIGN = ['#2F3337', '#1F2A44', '#7A1F24', '#F2F2EE', '#3A4A3A', '#B8B0A2', '#22201F', '#5B6066']
AWNING = ['#7A2E2A', '#2E4A3A', '#3A3D40', '#C8B48A', '#1F3550', '#8A3B22']
ROOF_TILE_RED = ['#9A4A32', '#8C3F2B', '#A5563B', '#7E3A28']
ROOF_ANTH = ['#3A3D40', '#45484C', '#33363A']
ZINC = '#8C969C'; COPPER = '#6FA58E'


def _code(f):
    if not f:
        return None, None
    a, _, b = f.partition('_')
    return a, b


def roof_stats(b):
    """Flaechenanteile flach/geneigt und mittlere Neigung."""
    fa = sa = 0.0; sl = []
    for r in b.roofs:
        a2 = r.area * abs(r.n[1])
        if r.n[1] > 0.995:
            fa += a2
        elif r.n[1] > 0.08:
            sa += a2; sl.append((math.degrees(math.acos(min(1, r.n[1]))), a2))
    slope = sum(s * a for s, a in sl) / max(1e-6, sum(a for s, a in sl)) if sl else 0.0
    return fa, sa, slope


# Einzelne Gebaeude nach Street-View-/Google-Earth-Textnotizen (data/streetview_notes.json), keine Bildvorlagen uebernommen
OVERRIDES = {
    # Historisches Museum (Pferdestrasse 6): 60er-Jahre-Bau, Bruchstein-Verblendung in Baendern + weisse Betonbaender
    'ID_1e223c17-842e-3f52-a54c-082809ef52fc': {'kind': 'modern', 'P': {
        'wall_mat': 'stone', 'wall_col': '#A89A86', 'trim_mat': 'plaster', 'trim_col': '#EEECE6', 'bands': 'slab', 'cornice': 'slab',
        'ribbon': False, 'axis': 3.2, 'win_w': 1.6, 'win_h': 1.5, 'frame_col': '#3A3D40', 'sill_col': '#EEECE6', 'blinds': 0.0}},
    # Bahnhofstrasse 5 (Apple): ganze Fassade dunkelgrauer Naturstein mit Steinfaschen und Gesimsbaendern, EG glatter Granit
    # (Street-View-Notiz Aug. 2022)
    'ID_3cc4d720-7c30-3ce2-8c25-8cde108fe292': {'P': {
        'wall_mat': 'stone', 'wall_col': '#5E6266', 'trim_mat': 'stone', 'trim_col': '#6E7276', 'sill_col': '#6E7276',
        'frame_col': '#2A2C2E', 'surround': 'thin', 'bands': 'slab', 'cornice': 'slab', 'blinds': 0.0,
        'gf_override': ('panel', '#3F4245')}},
    # Georgstrasse 31-33, H&M 'Runde Ecke' (Street View Kroepcke Nord): durchlaufende Fensterbaender mit hellen Bruestungsbaendern,
    # verglaster Treppenturm (Waende 5-11), grosse rote Einzelbuchstaben auf der Rundung im 3. OG
    'ID_6efcb086-4c94-3d3c-9bc0-e0e5c18766f5': {'kind': 'modern', 'P': {
        'wall_mat': 'plaster', 'wall_col': '#E6E6E2', 'trim_mat': 'plaster', 'trim_col': '#F0F0EC', 'ribbon': True, 'axis': 1.4,
        'win_w': 1.25, 'win_h': 1.55, 'sill': 0.95, 'frame_col': '#3A3D40', 'sill_col': '#D8D8D4', 'blinds': 0.0,
        'cornice': 'slab', 'glazed_walls': [5, 6, 7, 8, 9, 10, 11],
        'big_signs': [{'text': 'H&M', 'walls': [26, 27, 28, 29], 'floor': 3, 'y': 0.45, 'h': 1.9, 'w': 3.4, 'fg': '#E50010',
                       'font': 'script', 'd': 0.12}]}},
    # Bahnhofstrasse 14, Hugendubel: geneigtes Glasvordach (dunkler Stahl, ~2 m) ueber dem EG, weisse Einzelbuchstaben vor dem
    # 1. OG; rote Tafel 'Welt der Buecher' am EG (shops.SHOP_FIX)
    'ID_e8f2a770-b0a2-3934-a092-ff331f1965c9': {'P': {
        'glass_canopy': {'walls': [5], 'floor': 1, 'y': 0.12, 'depth': 2.0, 'rise': 0.3},
        'big_signs': [{'text': 'Hugendubel', 'walls': [5], 'floor': 1, 'y': 0.95, 'h': 1.05, 'w': 8.5, 'fg': '#FFFFFF',
                       'font': 'sans', 'd': 0.14}]}},
    # Ernst-August-Platz 5, GALERIA: grosse gruene Versalien an der Metallfassade zur Bahnhofstrasse
    'ID_a8ec46b5-104f-3422-ac72-2fbe54884984': {'P': {
        'big_signs': [{'text': 'GALERIA', 'walls': [52], 'floor': 3, 'y': 0.35, 'h': 2.5, 'w': 15.0, 'fg': '#6CB33F',
                       'font': 'bold-sans', 'd': 0.15}]}},
    # Kreuzkirche (St. Crucis, Kreuzkirchhof): gotische Hallenkirche aus grau-braunem Bruchstein-Sandstein mit Quaderecken
    # (landmark_notes.md Abschnitt 17), kein Backstein, kein Staffelgiebel; Kupferhelm am Turm
    'ID_9850fd33-58c8-3cc6-bffc-a396527794b2': {'kind': 'gothic', 'P': {
        'wall_mat': 'stone', 'wall_col': '#B8AA92', 'trim_col': '#D8CDB6', 'stepgable': False, 'roof_col': '#9A4A32', 'roof_mat': 'roof_tile'}},
    # St.-Nikolai-Kapelle (Ruine am Klagesmarkt): Bruchstein
    'ID_ef6f828b-2ad8-316a-a4a1-f756d10168b1': {'kind': 'gothic', 'P': {'wall_mat': 'stone', 'wall_col': '#B0A48E', 'stepgable': False}},
    'ID_29a5539d-882e-3392-9078-337a881024cc': {'kind': 'gothic', 'P': {'wall_mat': 'stone', 'wall_col': '#E6DDCB', 'stepgable': False}},
    # Basilika St. Clemens und Neustaedter Hof- und Stadtkirche: Barock, Sandstein / heller Putz
    'ID_b9ea6769-3c12-3be8-a0b9-a5ae6d6d59e7': {'kind': 'gothic', 'P': {'wall_mat': 'stone', 'wall_col': '#E6DDCB', 'stepgable': False}},
    # ---------------- Altstadt, benannte Haeuser (landmark_notes.md Abschnitte 6, 8, 12, 13, 16, 17; streetview_notes.json)
    # Holzmarkt 4, Noltehaus: gelber Verblendziegel mit weissen Sandsteingliederungen, Staffel-/Volutengiebel zum Holzmarkt,
    # weisse Fenstergewaende mit dunklen Rahmen, EG heller Stein mit grosser dunkel gerahmter Schaufensterfront (REBMANN)
    'ID_7f103f76-2cb1-36a5-b1f5-e39a5d4db35d': {'kind': 'altbau', 'P': {
        'wall_mat': 'brick_y', 'wall_col': '#FFFFFF', 'trim_mat': 'stone', 'trim_col': '#F4F0E6', 'sill_col': '#EDE5D5',
        'frame_col': '#2A2C2E', 'alt_variant': 1, 'sprossen': True, 'head': 'cornice', 'bands': 'altbau', 'cornice': 'altbau',
        'rustic': False, 'balcony_p': 0.0, 'brickgothic': True, 'stepgable': True, 'gable_niches': False,
        'roof_col': '#8C3F2B', 'roof_mat': 'roof_tile', 'gf_override': ('stone', '#EEE8DC'),
        'wood_front': False, 'front_col': '#22201F', 'blinds': 0.0, 'dormers': True, 'chimneys': 1}},
    # Holzmarkt 5, Leibnizhaus (bis data/landmarks_replace/leibnizhaus.json existiert): heller grau-beiger Sandstein,
    # hoher Staffelgiebel, Kreuzstockfenster mit kleinen Scheiben, dunkle Rahmen, Gesimse je Geschoss
    'ID_c494dba7-3b4a-3a44-9e3d-f038d58e0852': {'kind': 'altbau', 'P': {
        'wall_mat': 'stone', 'wall_col': '#E8DFCC', 'trim_mat': 'stone', 'trim_col': '#F2ECDF', 'sill_col': '#E8DFCC',
        'frame_col': '#2E2622', 'alt_variant': 1, 'sprossen': True, 'head': 'cornice', 'bands': 'altbau', 'cornice': 'altbau',
        'rustic': False, 'balcony_p': 0.0, 'brickgothic': True, 'stepgable': True, 'gable_niches': False,
        'roof_col': '#7E3A28', 'roof_mat': 'roof_tile', 'gf_override': ('stone', '#E8DFCC'),
        'wood_front': True, 'front_col': '#3B2A20', 'blinds': 0.0, 'dormers': False, 'chimneys': 0}},
    # Beginenturm (Pferdestrasse 8): runder Wehrturm aus hellem Bruchstein, Ringgesims, wenige Schlitzfenster, flaches Kegeldach
    'ID_5051cc6a-c86e-3179-b258-2228f15d551b': {'kind': 'tower', 'P': {'wall_mat': 'rubble', 'wall_col': '#F2EEE6',
                                                                        'roof_col': '#4A4A4C', 'roof_mat': 'roof_slate'}},
    # Burgstrasse 23a (Foto burgstrasse/03 links): dunkelbraunes Fachwerk, weisse Gefache
    'ID_93528938-baef-395a-bfa8-df4f22156f68': {'kind': 'fachwerk', 'P': {'timber_col': '#33261D', 'wall_col': '#F0ECE2',
                                                                           'wall_mat': 'plaster', 'frame_col': '#F2F2EE'}},
    # Burgstrasse 25 (M&B): dunkelbraunes Fachwerk mit grossem Strassengiebel, durchlaufende Fensterbaender, weisse Fenster
    'ID_0270cc38-c629-33d2-b909-7141b285932f': {'kind': 'fachwerk', 'P': {
        'timber_col': '#33261D', 'wall_mat': 'plaster', 'wall_col': '#F0ECE2', 'frame_col': '#F2F2EE', 'axis': 1.15, 'win_w': 0.8,
        'win_gap': 0.34, 'dormers': True, 'brace': 'cross', 'gf_stone': False, 'gf_override': ('plaster', '#EDE8DC')}},
    # Burgstrasse 27/27a: rechter Teil ochsenblutrotes Fachwerk mit weissen Gefachen (Rahmenhandlung), linker Teil
    # (Wand 6, an Nr. 25) grau gestrichenes Fachwerk mit dichten Fensterreihen; Traufe zur Strasse
    'ID_9bc17b19-56c4-3270-9fda-a0f254195867': {'kind': 'fachwerk', 'P': {
        'timber_col': '#6E2A22', 'wall_mat': 'plaster', 'wall_col': '#F2EFE6', 'frame_col': '#F2F2EE', 'axis': 1.85, 'win_w': 0.92,
        'brace': 'foot', 'dormers': True, 'gf_stone': False, 'gf_override': ('plaster', '#EDE8DC'),
        'wall_P': {6: {'timber_col': '#8F969A', 'axis': 1.2, 'win_w': 0.84, 'win_gap': 0.34, 'brace': 'cross'}}}},
    # Burgstrasse 12 (aeltestes Haus, 1564): reich geschnitztes Fachwerk, ockergelbe Gefache, Schieferdach
    'ID_cd874ae8': {'kind': 'fachwerk', 'P': {'timber_col': '#3B2A20', 'wall_col': '#D8B878', 'wall_mat': 'plaster',
                                               'frame_col': '#F2F2EE', 'roof_mat': 'roof_slate', 'roof_col': '#4E5257', 'brace': 'foot'}},
    # Holzmarkt 2 und 3: hohe Fachwerk-Eckhaeuser, dunkle Balken, cremefarbene Gefache, Holzladenfronten
    'ID_4f69486f-cda3-355a-bee3-e29ac0847164': {'kind': 'fachwerk', 'P': {'timber_col': '#33261D', 'wall_col': '#EDE3CF',
                                                                           'wall_mat': 'plaster', 'frame_col': '#F2F2EE'}},
    'ID_60cbc5d9-c564-3bf4-b251-2e89d4d44626': {'kind': 'fachwerk', 'P': {'timber_col': '#2E2622', 'wall_col': '#EDE3CF',
                                                                           'wall_mat': 'plaster', 'frame_col': '#F2F2EE'}},
    # Kramerstrasse 11/12/13 (Foto kramerstrasse/03): 12 = schwarzes Fachwerk mit orangeroten Gefachen und orangerotem EG,
    # 13 = grau gestrichenes Fachwerk, 11 = dunkelbraunes Fachwerk mit cremefarbenen Gefachen; weisse Fenster
    'ID_6d256eb6': {'kind': 'fachwerk', 'P': {'timber_col': '#22201E', 'wall_mat': 'plaster', 'wall_col': '#D9582E', 'frame_col': '#F2F2EE',
                                               'gf_stone': False, 'gf_override': ('plaster', '#D9582E'), 'brace': 'foot'}},
    'ID_fd4a2d94': {'kind': 'fachwerk', 'P': {'timber_col': '#8F969A', 'wall_mat': 'plaster', 'wall_col': '#F2F0EA', 'frame_col': '#F2F2EE'}},
    'ID_f8b19745': {'kind': 'fachwerk', 'P': {'timber_col': '#3B2A20', 'wall_mat': 'plaster', 'wall_col': '#EDE3CF', 'frame_col': '#F2F2EE'}},
    # Kramerstrasse 16: geschnitzte Renaissance-Fenstergewaende rotbraun (#7a3a2e), weisse Fluegel, Inschrift-Schwelle
    'ID_ea1b603c': {'kind': 'fachwerk', 'P': {'timber_col': '#7A3A2E', 'wall_mat': 'plaster', 'wall_col': '#F0ECE2', 'frame_col': '#F2F2EE'}},
    # Kramerstrasse 24, Broyhanhaus: Fachwerk ueber steinernem Arkaden-Erdgeschoss
    'ID_ceafc391': {'kind': 'fachwerk', 'P': {'timber_col': '#33261D', 'wall_mat': 'plaster', 'wall_col': '#F0ECE2', 'frame_col': '#F2F2EE',
                                               'gf_stone': True, 'gf_override': ('stone', '#D9CFBC')}},
    # Kramerstrasse 25 (Ecke Hanns-Lilje-Platz): Neorenaissance, roter Backstein mit Sandsteinbaendern, Staffelgiebel
    'ID_91d3bc7c': {'kind': 'altbau', 'P': {'wall_mat': 'brick', 'wall_col': '#E8D8D0', 'trim_mat': 'stone', 'trim_col': '#EDE5D5',
                                             'brickgothic': True, 'stepgable': True, 'gable_niches': False, 'frame_col': '#F2F2EE',
                                             'gf_override': ('stone', '#8E5A4A'), 'rustic': False}},
    # Ballhofplatz 7, Spittahaus: Bruchstein-Erdgeschoss mit Holz-Rundbogentuer, Fachwerk darueber
    'ID_b150f6c8': {'kind': 'fachwerk', 'P': {'timber_col': '#33261D', 'wall_mat': 'plaster', 'wall_col': '#F0ECE2', 'frame_col': '#F2F2EE',
                                               'gf_stone': False, 'gf_override': ('rubble', '#E6E0D4')}},
    # Ballhofplatz 2 (Teestuebchen / Ecke Burgstrasse): EG rosa-ocker verputzt, dunkelbraunes Fachwerk, weisse Gefache
    'ID_1a4a4aa4': {'kind': 'fachwerk', 'P': {'timber_col': '#3B2A20', 'wall_mat': 'plaster', 'wall_col': '#F0ECE2', 'frame_col': '#F2F2EE',
                                               'gf_stone': False, 'gf_override': ('plaster', '#E2BFA0')}},
    # Kreuzkirchhof 5, Kreuzklappe: roter Backstein-Staffelgiebel
    'ID_c654b357': {'kind': 'altbau', 'P': {'wall_mat': 'brick', 'wall_col': '#F2E6E0', 'trim_mat': 'stone', 'trim_col': '#EDE5D5',
                                             'brickgothic': True, 'stepgable': True, 'gable_niches': True, 'frame_col': '#F2F2EE',
                                             'rustic': False}},
    # Knochenhauerstrasse 11: ockerfarbener Putz mit Stuck, Eck-Erker
    'ID_ad6d04f6': {'kind': 'altbau', 'P': {'wall_mat': 'plaster', 'wall_col': '#D9B26A', 'trim_mat': 'plaster', 'trim_col': '#F2EFE6',
                                             'frame_col': '#F2F2EE'}},
    # Georgstrasse 23 (ehem. Karstadt, leer seit 2020): heller Natursteinbehang, verglaster Eckturm, Relief-Emblem;
    # EG durchgehende Schaufensterzone (leer/beklebt), Mindesthoehe EG 4.6 m
    'ID_12b82579-2ad0-37f8-ab0d-3fd2bc2c95eb': {'kind': 'kaufhaus', 'P': {
        'wall_mat': 'stone', 'wall_col': '#E4E2DC', 'trim_mat': 'stone', 'trim_col': '#E4E2DC', 'upper': 'slits',
        'gf_override': ('panel', '#B8B6B0'), 'free_kind': 'vacant', 'front_col': '#5B6066', 'gf_min': 4.6,
        'glazed_walls': [54, 47], 'emblem_wall': 57}},
}


def _ov(bid):
    """Override zu einer Gebaeude-ID (volle ID oder eindeutiger Praefix 'ID_xxxxxxxx')."""
    if bid in OVERRIDES:
        return OVERRIDES[bid]
    return OVERRIDES.get(bid[:11])


def classify(b):
    ov = _ov(b.id)
    if ov and ov.get('kind'):
        return ov['kind']
    a = b.attrs
    fam, fcode = _code(a.get('function'))
    osm = a.get('osm', {}); obt = a.get('osm_building')
    rng = rng_of('style', b.id)
    h = rng.random()
    fa, sa, slope = roof_stats(b)
    area = b.footprint.area
    H = b.y_top - b.y_ground
    pitched = sa > 0.45 * (fa + sa) and slope > 18
    steep = pitched and slope >= 34
    dist = math.hypot(b.cx, b.cz)
    street = b.street or ''
    mat = (osm.get('building:material') or '').lower()
    name = (osm.get('name') or '').lower()
    start = osm.get('start_date') or ''
    try:
        year = int(str(start)[:4])
    except ValueError:
        year = None
    pnames = [p.get('roofName') or '' for p in a.get('lgln_parts', [])]

    if fam == '51009' or obt == 'roof':
        return 'canopy'
    if fcode in ('3041', '3043', '3045', '3042') or ('kirche' in name and 'amt' not in name):
        return 'gothic'
    if fcode in ('2461', '2462') or obt == 'parking':
        return 'parking'
    if area < 12 or H < 3.6 or (area < 60 and H < 5.5):
        return 'shed'
    if obt in ('kiosk',):
        return 'shed'
    if 'glass' in mat or (year and year >= 1995 and not pitched and h < 0.6):
        return 'curtain'
    if any(k in name for k in ('galeria', 'karstadt', 'kaufhof', 'peek', 'kaufhaus', 'c&a', 'primark', 'saturn', 'müller', 'ernst-august-galerie')):
        return 'kaufhaus'
    if fcode in ('2010', '2000', '2050', '2052') and area > 1800 and H > 14 and not steep:
        return 'kaufhaus' if h < 0.7 else 'modern'
    arch = (osm.get('building:architecture') or '').lower()
    if 'timber' in arch or 'fachwerk' in arch:
        return 'fachwerk'
    fp = FACHWERK_P.get(street, 0.0)
    if fp and pitched and area < 420 and H < 24 and rng_of('fachwerk', b.id).random() < fp:
        return 'fachwerk'
    if year and year < 1920:
        return 'altbau'
    if steep:
        return 'altbau' if h < 0.85 else 'nachkrieg'
    if pitched:
        return 'nachkrieg' if h < 0.75 else 'altbau'
    # Flachdach
    if obt in ('office', 'commercial') or fcode in ('2020', '3010', '3020', '3023', '3024', '2100'):
        if H > 30:
            return 'curtain' if h < 0.45 else 'modern'
        return 'modern' if h < 0.6 else 'nachkrieg'
    if H > 32:
        return 'modern' if h < 0.6 else 'curtain'
    if dist > 330 and h < 0.18:
        return 'altbau'
    return 'nachkrieg' if h < 0.72 else 'modern'


def params(b, kind):
    """Stilparameter je Gebaeude (alles deterministisch)."""
    r = rng_of('params', b.id)
    P = {'kind': kind}
    pick = lambda L: L[int(r.integers(len(L)))]
    P['glass'] = pick(GLASS)
    P['plinth_col'] = pick(PLINTH)
    P['sign_col'] = pick(SHOP_SIGN)
    P['awning_col'] = pick(AWNING)
    P['awnings'] = r.random() < 0.35
    P['blinds'] = 0.0
    P['balcony_p'] = 0.0
    P['dormers'] = False
    P['chimneys'] = 0
    P['rustic'] = False
    P['gf_mat'] = None
    P['flat_roof'] = pick(['gravel', 'gravel', 'membrane', 'membrane', 'green'])
    P['parapet_h'] = float(r.uniform(0.45, 0.9))
    P['coping_col'] = pick(['#8C969C', '#9AA1A6', '#6E7378', '#B4B8BA'])
    P['overhang'] = float(r.uniform(0.3, 0.5))
    if kind == 'altbau':
        m = r.random()
        if m < 0.58:
            P['wall_mat'], P['wall_col'] = 'plaster', jitter_hex(pick(PLASTER_WARM), r, 0.03)
            P['trim_mat'], P['trim_col'] = 'plaster', jitter_hex('#F2EFE6', r, 0.02) if r.random() < 0.6 else jitter_hex(P['wall_col'], r, 0.06)
        elif m < 0.8:
            P['wall_mat'], P['wall_col'] = 'stone', pick(SANDSTONE_TINTS)
            P['trim_mat'], P['trim_col'] = 'stone', '#FFFFFF'
        else:
            P['wall_mat'], P['wall_col'] = 'brick', pick(BRICK_TINTS)
            P['trim_mat'], P['trim_col'] = 'stone', pick(SANDSTONE_TINTS)
        P['storey'] = float(r.uniform(3.6, 4.1)); P['gf'] = float(r.uniform(4.0, 4.6))
        P['axis'] = float(r.uniform(2.5, 3.1)); P['win_w'] = float(r.uniform(1.12, 1.3)); P['win_h'] = float(r.uniform(1.95, 2.25))
        P['sill'] = 0.85; P['reveal'] = float(r.uniform(0.26, 0.34)); P['margin'] = 1.1
        P['frame_col'] = pick(FRAME_WHITE) if r.random() < 0.7 else pick(FRAME_WOOD)
        P['pattern'] = 'T'; P['surround'] = 'frame'; P['head'] = 'cornice' if r.random() < 0.75 else ('pediment' if r.random() < 0.6 else 'key')
        P['alt_variant'] = int(r.integers(2)); P['sprossen'] = r.random() < 0.45
        P['bands'] = 'altbau'; P['cornice'] = 'altbau'; P['rustic'] = r.random() < 0.6
        P['balcony_p'] = 0.12 if r.random() < 0.5 else 0.0
        P['dormers'] = True; P['chimneys'] = int(r.integers(1, 4))
        P['roof_col'] = pick(ROOF_TILE_RED) if r.random() < 0.55 else pick(ROOF_ANTH); P['roof_mat'] = 'roof_tile' if r.random() < 0.8 else 'roof_slate'
        P['sill_col'] = P['trim_col'] if P['trim_mat'] != 'brick' else '#D8D0C0'
        P['blinds'] = 0.06
    elif kind == 'fachwerk':
        st = b.street or ''
        q = r.random()
        red_p = {'Burgstraße': 0.3, 'Kramerstraße': 0.25}.get(st, 0.12)
        tl = TIMBER_RED if q < red_p else (TIMBER_GREY if q < red_p + 0.14 else TIMBER)
        P['timber_col'] = pick(tl)
        P['infill'] = 'brick' if r.random() < 0.07 else 'plaster'
        if P['infill'] == 'brick':
            P['wall_mat'], P['wall_col'] = 'brick', pick(BRICK_TINTS)
        else:
            P['wall_mat'], P['wall_col'] = 'plaster', jitter_hex(pick(INFILL), r, 0.02)
        P['trim_mat'], P['trim_col'] = 'trim', P['timber_col']
        P['storey'] = float(r.uniform(2.75, 3.15)); P['gf'] = float(r.uniform(3.3, 3.9))
        if r.random() < 0.65:     # Fensterband (Burgstr. 23a-27, Kramerstr.): ein Fenster je Gefach, nur Staender dazwischen
            P['axis'] = float(r.uniform(1.12, 1.4)); P['win_w'] = float(r.uniform(0.76, 0.92)); P['win_gap'] = 0.36
        else:
            P['axis'] = float(r.uniform(1.55, 1.95)); P['win_w'] = float(r.uniform(0.8, 0.98))
        P['win_h'] = float(r.uniform(1.3, 1.5))
        P['sill'] = 0.82; P['reveal'] = 0.1; P['margin'] = 0.65
        P['frame_col'] = pick(FRAME_WHITE) if r.random() < 0.85 else '#2A4A3A'
        P['pattern'] = 'cross'; P['surround'] = None; P['head'] = None; P['sprossen'] = True
        P['bands'] = 'fachwerk'; P['cornice'] = None
        P['roof_col'] = pick(ROOF_TILE_RED); P['roof_mat'] = 'roof_tile'; P['chimneys'] = 1; P['dormers'] = True
        P['sill_col'] = P['timber_col']; P['overhang'] = 0.5
        P['jetty'] = float(r.uniform(0.14, 0.22))
        P['brace'] = ['cross', 'foot', 'mann', 'cross'][int(r.integers(4))]
        P['gf_stone'] = r.random() < 0.65
        P['fascia_col'] = P['timber_col']
    elif kind == 'nachkrieg':
        m = r.random()
        if m < 0.74:
            P['wall_mat'], P['wall_col'] = 'plaster', jitter_hex(pick(PLASTER_COOL), r, 0.03)
        elif m < 0.88:
            P['wall_mat'], P['wall_col'] = 'brick', pick(KLINKER_TINTS) if r.random() < 0.5 else pick(BRICK_TINTS)
        else:
            P['wall_mat'], P['wall_col'] = 'panel', pick(TRAVERTINE)
        P['trim_mat'], P['trim_col'] = 'plaster', jitter_hex('#F0EEE8', r, 0.02)
        if P['wall_mat'] == 'plaster' and r.random() < 0.4:
            P['trim_col'] = jitter_hex(P['wall_col'], r, 0.08)
        P['storey'] = float(r.uniform(3.1, 3.35)); P['gf'] = float(r.uniform(3.6, 4.4))
        P['axis'] = float(r.uniform(2.3, 2.9)); P['win_w'] = float(r.uniform(1.25, 1.6)); P['win_h'] = float(r.uniform(1.42, 1.62))
        P['sill'] = 0.9; P['reveal'] = float(r.uniform(0.14, 0.2)); P['margin'] = 0.85
        P['frame_col'] = pick(FRAME_WHITE) if r.random() < 0.75 else pick(FRAME_DARK)
        P['pattern'] = '2' if P['win_w'] < 1.45 else '3'
        P['surround'] = 'thin' if r.random() < 0.5 else None; P['head'] = None
        P['bands'] = 'slab' if r.random() < 0.5 else None; P['cornice'] = 'slab'
        P['roof_col'] = pick(ROOF_TILE_RED) if r.random() < 0.5 else pick(ROOF_ANTH); P['roof_mat'] = 'roof_tile'
        P['dormers'] = r.random() < 0.3; P['chimneys'] = int(r.integers(0, 3))
        P['sill_col'] = pick(['#8C969C', '#B8B0A2', '#D8D0C0'])
        P['blinds'] = 0.22
        P['balcony_p'] = 0.05 if r.random() < 0.3 else 0.0
    elif kind == 'modern':
        m = r.random()
        if m < 0.45:
            P['wall_mat'], P['wall_col'] = 'panel', pick(PANEL_COLS)
        elif m < 0.75:
            P['wall_mat'], P['wall_col'] = 'plaster', jitter_hex(pick(['#F2F0EA', '#E8E8E4', '#DADCDA', '#E6E0D4']), r, 0.02)
        elif m < 0.9:
            P['wall_mat'], P['wall_col'] = 'panel', pick(['#6E6A66', '#8C8A86', '#5B6066', '#9AA1A6'])
        else:
            P['wall_mat'], P['wall_col'] = 'brick', pick(KLINKER_TINTS)
        P['trim_mat'], P['trim_col'] = P['wall_mat'], P['wall_col']
        P['storey'] = float(r.uniform(3.5, 3.9)); P['gf'] = float(r.uniform(4.2, 4.8))
        P['ribbon'] = r.random() < 0.55
        P['axis'] = float(r.uniform(1.3, 1.5)) if P['ribbon'] else float(r.uniform(2.6, 3.4))
        P['win_w'] = float(r.uniform(1.4, 1.9)); P['win_h'] = float(r.uniform(1.75, 2.15))
        P['sill'] = 0.8; P['reveal'] = float(r.uniform(0.18, 0.26)); P['margin'] = 0.6
        P['frame_col'] = pick(FRAME_DARK) if r.random() < 0.7 else pick(['#A6AEAF', '#B8BEBD', '#8C969C'])
        P['pattern'] = '2'; P['surround'] = None; P['head'] = None; P['bands'] = None; P['cornice'] = 'attic'
        P['roof_col'] = pick(ROOF_ANTH); P['roof_mat'] = 'roof_metal'
        P['sill_col'] = P['frame_col']; P['blinds'] = 0.3
    elif kind == 'curtain':
        P['wall_mat'], P['wall_col'] = 'glass', pick(['#2B3940', '#33434C', '#1F2C34', '#3B464C'])
        P['trim_mat'], P['trim_col'] = 'metal', pick(['#3A3D40', '#8C969C', '#A6AEAF', '#2F3337'])
        P['storey'] = float(r.uniform(3.6, 4.0)); P['gf'] = float(r.uniform(4.4, 5.2))
        P['axis'] = float(r.uniform(1.25, 1.5)); P['win_w'] = 0; P['win_h'] = 0
        P['sill'] = 0.0; P['reveal'] = 0.05; P['margin'] = 0.25
        P['frame_col'] = P['trim_col']; P['pattern'] = 'curtain'; P['surround'] = None; P['head'] = None
        P['bands'] = None; P['cornice'] = 'attic'; P['roof_col'] = pick(ROOF_ANTH); P['roof_mat'] = 'roof_metal'
        P['sill_col'] = P['trim_col']; P['glass'] = pick(['#2A3A44', '#30414C', '#26343C'])
    elif kind == 'kaufhaus':
        P['wall_mat'], P['wall_col'] = 'panel', pick(TRAVERTINE + ['#CFD3D2'])
        P['trim_mat'], P['trim_col'] = 'panel', P['wall_col']
        P['storey'] = float(r.uniform(4.2, 4.8)); P['gf'] = float(r.uniform(4.8, 5.6))
        P['upper'] = ['slits', 'grid', 'closed', 'bands'][int(r.integers(4))]
        P['axis'] = float(r.uniform(2.6, 3.4)); P['win_w'] = 0.9 if P['upper'] == 'grid' else 0.55
        P['win_h'] = 1.1 if P['upper'] == 'grid' else 3.2
        P['sill'] = 0.9 if P['upper'] == 'grid' else 0.5; P['reveal'] = 0.22; P['margin'] = 1.5
        P['frame_col'] = pick(FRAME_DARK); P['pattern'] = 'single'; P['surround'] = None; P['head'] = None
        P['bands'] = None; P['cornice'] = 'attic'; P['roof_col'] = pick(ROOF_ANTH); P['roof_mat'] = 'roof_metal'
        P['sill_col'] = '#8C969C'
    elif kind == 'gothic':
        P['wall_mat'], P['wall_col'] = 'brick', pick(['#E8D6CC', '#F0E0D6', '#DCC8BC'])
        P['trim_mat'], P['trim_col'] = 'stone', '#F0E8DA'
        P['storey'] = 6.0; P['gf'] = 6.0; P['axis'] = 5.5; P['win_w'] = 2.0; P['win_h'] = 8.0
        P['sill'] = 3.0; P['reveal'] = 0.45; P['margin'] = 1.6
        P['frame_col'] = '#3A3D40'; P['pattern'] = 'lancet'; P['surround'] = None; P['head'] = None
        P['bands'] = None; P['cornice'] = 'thin'; P['roof_col'] = COPPER; P['roof_mat'] = 'roof_metal'
        P['sill_col'] = '#D8D0C0'; P['glass'] = '#2A3238'
    elif kind == 'tower':
        P['wall_mat'], P['wall_col'] = 'rubble', '#F2EEE6'
        P['trim_mat'], P['trim_col'] = 'stone', '#D8CDB6'
        P['storey'] = 4.0; P['gf'] = 4.0; P['axis'] = 3.0; P['win_w'] = 0.22; P['win_h'] = 0.9
        P['sill'] = 1.0; P['reveal'] = 0.45; P['margin'] = 0.3
        P['frame_col'] = '#2A2A2A'; P['pattern'] = 'slit'; P['surround'] = None; P['head'] = None
        P['bands'] = None; P['cornice'] = 'thin'; P['rings'] = (0.42, 0.78)
        P['roof_col'] = '#4A4A4C'; P['roof_mat'] = 'roof_slate'; P['sill_col'] = '#D8CDB6'
        P['dormers'] = False; P['chimneys'] = 0; P['awnings'] = False
    elif kind == 'parking':
        P['wall_mat'], P['wall_col'] = 'plaster', pick(['#BEBEC1', '#CFD3D2', '#B8B0A2'])
        P['trim_mat'], P['trim_col'] = 'plaster', P['wall_col']
        P['storey'] = 2.9; P['gf'] = 3.2; P['axis'] = 5.0; P['win_w'] = 4.4; P['win_h'] = 1.5
        P['sill'] = 1.05; P['reveal'] = 0.3; P['margin'] = 0.6
        P['frame_col'] = '#3A3D40'; P['pattern'] = 'open'; P['surround'] = None; P['head'] = None
        P['bands'] = None; P['cornice'] = None; P['roof_col'] = pick(ROOF_ANTH); P['roof_mat'] = 'roof_metal'
        P['sill_col'] = P['wall_col']
    else:  # shed, canopy
        P['wall_mat'], P['wall_col'] = 'plaster', jitter_hex(pick(['#D6D6D2', '#E6E0D4', '#CFD3D2', '#BEBEC1']), r, 0.03)
        P['trim_mat'], P['trim_col'] = 'plaster', P['wall_col']
        P['storey'] = 3.0; P['gf'] = 3.0; P['axis'] = 2.4; P['win_w'] = 1.0; P['win_h'] = 1.2
        P['sill'] = 1.0; P['reveal'] = 0.12; P['margin'] = 0.5
        P['frame_col'] = pick(FRAME_DARK); P['pattern'] = 'single'; P['surround'] = None; P['head'] = None
        P['bands'] = None; P['cornice'] = None; P['roof_col'] = pick(ROOF_ANTH); P['roof_mat'] = 'roof_metal'
        P['sill_col'] = '#8C969C'
    if kind in ('altbau', 'nachkrieg') and (b.street in ALTSTADT_STREETS or b.street in FACHWERK_P):
        # Altstadt-Wiederaufbau: warme Putzfarben, Sandstein-Faschen, Gauben, Sprossenfenster
        P['altstadt'] = True
        if P['wall_mat'] in ('plaster', 'panel'):
            P['wall_mat'], P['wall_col'] = 'plaster', jitter_hex(pick(ALTSTADT_PLASTER), r, 0.02)
        P['dormers'] = True
        if kind == 'nachkrieg':
            P['surround'] = 'thin'; P['trim_mat'], P['trim_col'] = 'stone', pick(SANDSTONE_TINTS)
            P['frame_col'] = pick(FRAME_WHITE); P['blinds'] = 0.05
        osm = b.attrs.get('osm', {})
        if (osm.get('building:material') == 'brick' or 'heritage' in osm) and kind == 'altbau' and r.random() < 0.6:
            P['brickgothic'] = True
            P['wall_mat'], P['wall_col'] = 'brick', pick(KLINKER_TINTS + BRICK_TINTS[1:])
            P['trim_mat'], P['trim_col'] = 'stone', pick(SANDSTONE_TINTS)
    if kind in ('altbau', 'nachkrieg', 'fachwerk') and P.get('wall_mat') == 'brick':
        P['sill_col'] = '#D8D0C0' if kind != 'fachwerk' else P['sill_col']
    if kind == 'gothic':
        P['brickgothic'] = True
    ov = _ov(b.id)
    if ov:
        op = dict(ov.get('P', {}))
        P.update(op)
        if 'roof_col' in op:
            P['roof_fixed'] = True                         # Dachfarbe aus der Notiz hat Vorrang vor dem Orthofoto
        if kind == 'fachwerk' and 'timber_col' in op:
            for k in ('trim_col', 'sill_col', 'fascia_col'):
                if k not in op:
                    P[k] = op['timber_col']
        if 'wall_P' in op:
            P['wall_P'] = {int(k): v for k, v in op['wall_P'].items()}
    return P

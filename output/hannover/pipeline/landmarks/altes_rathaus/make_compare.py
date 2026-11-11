"""Vergleichsbilder Referenzfoto | Modellvorschau -> blender/previews/lm_altes_rathaus_vergleich_<shot>.png
Aufruf: python make_compare.py [prefix]   (prefix der Vorschaubilder, Standard blender/previews/lm_altes_rathaus_)"""
import sys, os
from PIL import Image, ImageDraw, ImageFont
ROOT = "C:/Users/luca/Documents/Sensing Space/Claude Code/output/hannover"
REF = "C:/Users/luca/AppData/Local/Temp/claude/86a9b6371061b51e/b8a76d7d-27b8-490e-b6de-334a309ea42a/scratchpad/research/ref_images/altes_rathaus/"
PRE = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/blender/previews/lm_altes_rathaus_'
OUT = ROOT + '/blender/previews/lm_altes_rathaus_vergleich_'
PAIRS = {'markt_front': ('05_20230321_Altes_Rathaus_Hannover.jpg', 'Marktseite (Foto 05, Wikimedia Commons)'),
         'ne_gable': ('01_Hannover_Altes_Rathaus_northeastern_facade.jpg', 'NO-Giebel Marktfluegel (Foto 01, Wikimedia Commons)'),
         'karmarsch': ('10_Hannover_das_Alte_Rathaus_Kobelinger_Strasse_und_Turm_der_Ma.jpg', 'Karmarschstrasse (Foto 10, Wikimedia Commons)'),
         'east': ('16_Altes_Rathaus_in_Hannover.jpg', 'Schmiedestrasse/Karmarschstrasse (Foto 16, Wikimedia Commons)'),
         'west_corner': ('07_L1040838Altes_Rathaus_H_Westecke_nah.jpg', 'Westecke mit Laube (Foto 07, Wikimedia Commons)'),
         'koebelinger': ('14_Hannover_Altes_Rathaus_sogen_Dogenpalast.jpg', 'Gerichtsfluegel / Koebelinger Strasse (Foto 14, Wikimedia Commons)'),
         'schmiede_n': ('09_2026_01_09_Altes_Rathaus_114646.jpg', 'Schmiedestrasse von N (Foto 09, Wikimedia Commons)')}
H = 600
try:
    F = ImageFont.truetype('arial.ttf', 15)
except Exception:
    F = ImageFont.load_default()
for shot, (rf, label) in PAIRS.items():
    pm = PRE + shot + '.png'
    if not os.path.exists(pm) or not os.path.exists(REF + rf): continue
    a = Image.open(REF + rf).convert('RGB'); b = Image.open(pm).convert('RGB')
    a = a.resize((int(a.width * H / a.height), H), Image.LANCZOS); b = b.resize((int(b.width * H / b.height), H), Image.LANCZOS)
    im = Image.new('RGB', (a.width + b.width + 10, H + 24), (255, 255, 255))
    im.paste(a, (0, 24)); im.paste(b, (a.width + 10, 24))
    d = ImageDraw.Draw(im)
    d.text((4, 4), 'Referenz: ' + label, fill=(0, 0, 0), font=F); d.text((a.width + 14, 4), 'Modell: lm_altes_rathaus_' + shot, fill=(0, 0, 0), font=F)
    tmp = OUT + shot + '.tmp.png'; im.save(tmp); os.replace(tmp, OUT + shot + '.png')
    print('vergleich', shot)

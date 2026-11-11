"""Ladenfront-Texturen fuer die Passerelle (Ebene -1): Schild-Atlanten mit echten Namen (data/shops.json: Farben = ungefaehre
Marken-CI, nicht aus Street View) und Schaufenster-/Innenraum-Texturen je Branche (prozedural).
Ausgabe: tex/prom_signs_<k>.png, tex/shopint_<kat>.jpg, data/ground_cache/shop_atlas.json"""
import os, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from common import *

TD = OUT + 'tex/'
def save(im, name, q=88):
    p = TD + name; tmp = TD + '.' + name + '.tmp'
    if name.endswith('.jpg'): im.convert('RGB').save(tmp, format='JPEG', quality=q, optimize=True)
    else: im.save(tmp, format='PNG', optimize=True)
    os.replace(tmp, p)
def font(sz, kind='bold-sans'):
    names = {'bold-sans': ['arialbd.ttf'], 'sans': ['arial.ttf'], 'serif': ['timesbd.ttf', 'georgiab.ttf'], 'script': ['segoesc.ttf', 'BRUSHSCI.TTF', 'arialbi.ttf'],
             'condensed': ['arialnb.ttf', 'arialbd.ttf'], 'light-sans': ['arial.ttf']}
    for f in names.get(kind, ['arialbd.ttf']) + ['arialbd.ttf']:
        try: return ImageFont.truetype(f, sz)
        except OSError: pass
    return ImageFont.load_default()
def hexc(h, default=(255, 255, 255)):
    try:
        h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception: return default

def sign_image(text, fg, bg, kind, w=1024, h=128):
    im = Image.new('RGB', (w, h), bg); d = ImageDraw.Draw(im)
    fs = int(h * 0.62); f = font(fs, kind)
    bb = d.textbbox((0, 0), text, font=f); tw = bb[2] - bb[0]
    while tw > w * 0.9 and fs > 12:
        fs = int(fs * 0.9); f = font(fs, kind); bb = d.textbbox((0, 0), text, font=f); tw = bb[2] - bb[0]
    d.text(((w - tw) / 2 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), text, fill=fg, font=f)
    # feiner Rahmen (Leuchtkasten)
    d.rectangle([0, 0, w - 1, h - 1], outline=tuple(int(c * 0.7) for c in bg), width=3)
    return im

def build_atlas(entries):
    """entries: [(key, text, fg, bg, font)] -> Atlanten 2048x2048 mit 2x16 Feldern (1024x128); Rueckgabe {key: (atlas, u0,v0,u1,v1)}"""
    out = {}; per = 32; k = 0
    for a in range(0, len(entries), per):
        atlas = Image.new('RGB', (2048, 2048), (30, 30, 30))
        for i, (key, text, fg, bg, fk) in enumerate(entries[a:a + per]):
            col, row = i % 2, i // 2
            atlas.paste(sign_image(text, fg, bg, fk), (col * 1024, row * 128))
            out[key] = ('prom_signs_%d' % k, col * 0.5 + 0.002, row / 16 + 0.002, col * 0.5 + 0.5 - 0.002, (row + 1) / 16 - 0.002)
        save(atlas, 'prom_signs_%d.png' % k); k += 1
    return out, k

# ---------------------------------------------------------------- Innenraum-/Schaufenster-Texturen (Rueckwand, 4 x 2.4 m)
rng = np.random.default_rng(5)
def interior(kind):
    W, H = 1024, 512
    im = Image.new('RGB', (W, H), (232, 228, 220)); d = ImageDraw.Draw(im)
    def shelves(cols, rows, item=(30, 50), bg=(240, 238, 232), board=(170, 170, 172), pal=None):
        d.rectangle([0, 0, W, H], fill=bg)
        for r in range(rows):
            y = int(H * 0.12 + r * H * 0.78 / rows)
            d.rectangle([20, y + int(H * 0.78 / rows) - 8, W - 20, y + int(H * 0.78 / rows)], fill=board)
            x = 26
            while x < W - 40:
                w = int(rng.uniform(*item)); hh = int(H * 0.78 / rows * rng.uniform(0.5, 0.85))
                c = tuple(int(v) for v in (pal[rng.integers(len(pal))] if pal else rng.integers(40, 230, 3)))
                d.rectangle([x, y + int(H * 0.78 / rows) - 8 - hh, x + w, y + int(H * 0.78 / rows) - 8], fill=c)
                x += w + int(rng.uniform(2, 8))
    if kind == 'bakery':
        d.rectangle([0, 0, W, H], fill=(214, 190, 150))
        for r in range(3):
            y = 60 + r * 110; d.rectangle([40, y + 80, W - 40, y + 88], fill=(120, 90, 60))
            for i in range(26):
                x = 50 + i * 36 + int(rng.uniform(-4, 4)); c = (int(rng.uniform(150, 200)), int(rng.uniform(100, 140)), int(rng.uniform(50, 80)))
                d.ellipse([x, y + 46, x + 32, y + 80], fill=c)
        d.rectangle([0, H - 140, W, H], fill=(70, 60, 55)); d.rectangle([0, H - 150, W, H - 140], fill=(200, 200, 205))
    elif kind == 'clothes':
        d.rectangle([0, 0, W, H], fill=(236, 234, 230))
        for rack in range(3):
            x0 = 40 + rack * 330; d.rectangle([x0, 120, x0 + 280, 126], fill=(60, 60, 60))
            x = x0 + 6
            while x < x0 + 270:
                c = tuple(int(v) for v in rng.integers(30, 230, 3)); w = int(rng.uniform(14, 26))
                d.rectangle([x, 126, x + w, 126 + int(rng.uniform(150, 260))], fill=c); x += w + 2
        for i in range(3):
            x = 120 + i * 330; d.ellipse([x, 300, x + 60, 360], fill=(220, 210, 200)); d.rectangle([x + 10, 360, x + 50, 500], fill=tuple(int(v) for v in rng.integers(30, 200, 3)))
    elif kind in ('fast_food', 'restaurant', 'cafe'):
        d.rectangle([0, 0, W, H], fill=(60, 50, 45) if kind != 'cafe' else (120, 90, 70))
        d.rectangle([0, 40, W, 150], fill=(25, 25, 25))
        for i in range(6):
            x = 30 + i * 165; d.rectangle([x, 50, x + 150, 140], fill=(230, 200, 60) if i % 2 else (220, 80, 60))
            d.rectangle([x + 10, 60, x + 140, 100], fill=(250, 240, 220))
        d.rectangle([0, 300, W, H], fill=(150, 150, 155)); d.rectangle([0, 290, W, 305], fill=(200, 200, 205))
        for i in range(5): d.rectangle([60 + i * 200, 230, 140 + i * 200, 290], fill=(240, 240, 240))
    elif kind == 'pharmacy':
        shelves(10, 5, (26, 44), (248, 248, 248), (200, 200, 200), pal=[(250, 250, 250), (230, 60, 60), (60, 140, 200), (90, 180, 90), (240, 240, 200)])
        d.rectangle([W // 2 - 50, 10, W // 2 + 50, 110], fill=(30, 160, 70)); d.rectangle([W // 2 - 15, 20, W // 2 + 15, 100], fill=(255, 255, 255)); d.rectangle([W // 2 - 40, 45, W // 2 + 40, 75], fill=(255, 255, 255))
    elif kind in ('chemist', 'cosmetics', 'perfumery', 'beauty', 'hairdresser'):
        if kind in ('hairdresser', 'beauty'):
            d.rectangle([0, 0, W, H], fill=(210, 200, 195))
            for i in range(4):
                x = 40 + i * 250; d.rectangle([x, 60, x + 180, 300], fill=(180, 200, 210)); d.rectangle([x + 4, 64, x + 176, 296], fill=(150, 175, 190))
                d.rectangle([x + 50, 330, x + 130, 480], fill=(30, 30, 30))
        else:
            shelves(12, 4, (18, 34), (250, 250, 250), (220, 220, 225), pal=[(240, 200, 210), (200, 40, 80), (250, 250, 250), (180, 140, 200), (240, 220, 160)])
    elif kind in ('kiosk', 'newsagent', 'tobacco', 'tea', 'photo', 'mobile_phone', 'key_cutter', 'tailor', 'florist', 'supermarket', 'convenience'):
        if kind == 'florist':
            d.rectangle([0, 0, W, H], fill=(90, 120, 80))
            for i in range(140):
                x, y = rng.uniform(0, W), rng.uniform(150, H); r = rng.uniform(10, 26)
                d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in ((rng.uniform(150, 255), rng.uniform(40, 200), rng.uniform(60, 200)) if rng.random() < 0.6 else (60, rng.uniform(110, 160), 50))))
        elif kind == 'mobile_phone':
            d.rectangle([0, 0, W, H], fill=(245, 245, 248))
            for i in range(4):
                x = 60 + i * 240; d.rectangle([x, 300, x + 180, 330], fill=(200, 200, 205))
                for j in range(3): d.rounded_rectangle([x + 10 + j * 58, 210, x + 50 + j * 58, 300], 8, fill=(20, 20, 25))
            d.rectangle([0, 0, W, 120], fill=(30, 60, 140))
        else:
            shelves(14, 6, (20, 36), (235, 232, 225), (120, 110, 100))
    elif kind == 'vacant':
        d.rectangle([0, 0, W, H], fill=(236, 236, 230))
        for i in range(30):
            x, y = rng.uniform(0, W), rng.uniform(0, H); d.rectangle([x, y, x + rng.uniform(40, 120), y + rng.uniform(30, 90)], fill=(248, 248, 245))
        f = font(54); d.text((W // 2 - 190, H // 2 - 30), 'ZU VERMIETEN', font=f, fill=(180, 30, 30))
    else:
        shelves(10, 4, (24, 50), (238, 236, 230), (150, 150, 150))
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    # Lichtverlauf (oben heller)
    a = np.array(im).astype(float) / 255 * np.linspace(1.12, 0.82, H)[:, None, None]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

CATS = ['bakery', 'clothes', 'fast_food', 'cafe', 'pharmacy', 'chemist', 'beauty', 'hairdresser', 'kiosk', 'florist', 'mobile_phone', 'vacant', 'generic']
CATMAP = {'restaurant': 'fast_food', 'ice_cream': 'cafe', 'cosmetics': 'chemist', 'perfumery': 'chemist', 'newsagent': 'kiosk', 'tobacco': 'kiosk',
          'tea': 'kiosk', 'photo': 'mobile_phone', 'key_cutter': 'kiosk', 'tailor': 'clothes', 'supermarket': 'kiosk', 'convenience': 'kiosk',
          'department_store': 'clothes', 'shoes': 'clothes', 'jewelry': 'beauty', 'nails': 'beauty', 'yes': 'vacant', 'fabric': 'vacant'}
def cat_of(c):
    c = c or 'generic'
    if c in CATS: return c
    return CATMAP.get(c, 'generic')

if __name__ == '__main__':
    for c in CATS: save(interior(c), 'shopint_%s.jpg' % c)
    J = json.load(open(DATA + 'shops.json', encoding='utf-8'))
    ent = []
    for sh in J['shops']:
        if sh.get('level') != -1 and sh.get('context') not in ('promenade', 'ubahn_passage'): continue
        sg = sh.get('sign') or {}
        txt = sg.get('text') or sh.get('name') or sh.get('name_osm')
        if not txt or sh.get('status') in ('closed', 'vacant'): continue
        fg = hexc(sg.get('fg') or '#ffffff'); bg = hexc(sg.get('bg') or '#1a1a1a', (26, 26, 26))
        lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        if sg.get('bg') is None or abs(lum(fg) - lum(bg)) < 90:
            bg = (240, 240, 236) if lum(fg) < 128 else (28, 28, 30)
        ent.append((sh['id'], txt, fg, bg, sg.get('font') or 'bold-sans'))
    ent.append(('generic_vacant', 'Ladenfläche zu vermieten', (40, 40, 40), (235, 235, 230), 'sans'))
    ent.append(('prom_niki', 'NIKI PROMENADE', (255, 255, 255), (200, 30, 120), 'bold-sans'))
    ent.append(('prom_kroepcke', 'U  Kröpcke  ←', (255, 255, 255), (28, 46, 110), 'bold-sans'))
    ent.append(('prom_hbf', 'Hauptbahnhof  →', (255, 255, 255), (28, 46, 110), 'bold-sans'))
    ent.append(('prom_eap', 'Ernst-August-Platz  ↑', (255, 255, 255), (28, 46, 110), 'bold-sans'))
    ent.append(('prom_bahnhofstr', 'Bahnhofstraße  ↑', (255, 255, 255), (28, 46, 110), 'bold-sans'))
    atlas, n = build_atlas(ent)
    json.dump({'atlas': atlas, 'n': n}, open(CACHE + 'shop_atlas.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('signs', len(ent), 'atlases', n)

"""Schrift-Atlas (2048 x 1024) fuer Schilder/Markisen/Inschriften am Ballhofplatz -> <tex>/signs_col.png + signs.json
Jeder Eintrag: Rechteck im Atlas (u0,v0,u1,v1 in 0..1, v von unten) und Seitenverhaeltnis.  Namen aus OSM / Fotos."""
import os, sys, json
from PIL import Image, ImageDraw, ImageFont

FONTS = 'C:/Windows/Fonts/'


def font(name, size):
    for f in (name, 'arial.ttf'):
        try:
            return ImageFont.truetype(FONTS + f, size)
        except OSError:
            continue
    return ImageFont.load_default()


W, H = 2048, 1024
# (key, w_px, h_px, bg, fg, text, fontfile, size, extra)
SIGNS = [
    ('junges_schauspiel', 1024, 96, (34, 38, 40), (238, 238, 232), 'junges schauspiel', 'arialbd.ttf', 70, None),
    ('junge_oper', 1024, 96, (34, 38, 40), (238, 238, 232), 'junge oper  |  BALLHOF EINS', 'arialbd.ttf', 66, None),
    ('ballhof_eins_v', 96, 512, (34, 38, 40), (238, 238, 232), 'BALLHOF EINS', 'arialbd.ttf', 62, 'vertical'),
    ('spitta', 2048, 64, (52, 30, 22), (186, 52, 40), 'DIE JUNGEN HABEN DIE AUFGABE NEUE WEGE ZU SUCHEN UND ZU BAHNEN UND DEN MUT SIE ZU GEHEN', 'georgiab.ttf', 44, None),
    ('craftbier', 256, 256, (24, 24, 24), (236, 236, 230), 'craft\nbier\nbar', 'georgiab.ttf', 54, 'round'),
    ('hofgefluester', 512, 128, (26, 26, 26), (240, 240, 236), '[hof:geflüster]', 'arial.ttf', 64, None),
    ('silvergold', 1024, 128, (244, 242, 236), (40, 36, 34), 'Goldschmiede  Silver & Gold  Goldschmiede', 'georgiab.ttf', 64, None),
    ('teestuebchen', 1024, 96, (238, 234, 222), (52, 46, 40), 'TEESTÜBCHEN SEIT 1970 IM FAMILIENBESITZ', 'georgia.ttf', 54, None),
    ('mariejo', 512, 128, (40, 34, 38), (236, 226, 222), 'marie jo', 'georgiai.ttf', 84, None),
    ('poster1', 256, 384, (60, 70, 120), (240, 240, 240), 'BALLHOF\nEINS\n\nSPIELZEIT\n2026/27', 'arialbd.ttf', 40, 'poster'),
    ('poster2', 256, 384, (120, 40, 50), (240, 230, 220), 'junges\nschauspiel\n\nhannover', 'arialbd.ttf', 40, 'poster'),
    ('plaque', 256, 192, (120, 92, 52), (230, 208, 150), 'BALLHOF\nerbaut 1649-1664\nals Ballspielhaus\nfür Herzog Georg Wilhelm', 'georgia.ttf', 22, 'plaque'),
]


def build(out):
    img = Image.new('RGB', (W, H), (128, 128, 128))
    d = ImageDraw.Draw(img)
    x, y, rowh = 0, 0, 0
    meta = {}
    for key, w, h, bg, fg, text, ff, size, extra in SIGNS:
        if x + w > W:
            x = 0; y += rowh; rowh = 0
        tile = Image.new('RGB', (w, h), bg)
        td = ImageDraw.Draw(tile)
        f = font(ff, size)
        if extra == 'vertical':
            t2 = Image.new('RGB', (h, w), bg); d2 = ImageDraw.Draw(t2)
            bb = d2.textbbox((0, 0), text, font=f)
            d2.text(((h - (bb[2] - bb[0])) / 2 - bb[0], (w - (bb[3] - bb[1])) / 2 - bb[1]), text, font=f, fill=fg)
            tile = t2.rotate(90, expand=True)
        elif extra in ('round', 'poster', 'plaque'):
            if extra == 'round':
                td.rectangle([0, 0, w, h], fill=(70, 66, 60))
                td.ellipse([4, 4, w - 4, h - 4], fill=bg, outline=(200, 200, 196), width=6)
            if extra == 'plaque':
                td.rectangle([6, 6, w - 7, h - 7], outline=fg, width=4)
            if extra == 'poster':
                for k in range(6):
                    td.rectangle([20 + k * 6, 200 + k * 20, w - 20 - k * 6, 210 + k * 20], fill=tuple(min(255, c + 30 * (k % 2)) for c in bg))
            bb = td.multiline_textbbox((0, 0), text, font=f, align='center')
            td.multiline_text(((w - (bb[2] - bb[0])) / 2 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), text, font=f, fill=fg, align='center')
        else:
            bb = td.textbbox((0, 0), text, font=f)
            tw = bb[2] - bb[0]
            if tw > w - 16:
                f = font(ff, int(size * (w - 16) / tw)); bb = td.textbbox((0, 0), text, font=f); tw = bb[2] - bb[0]
            td.text(((w - tw) / 2 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), text, font=f, fill=fg)
        img.paste(tile, (x, y))
        meta[key] = dict(uv=[x / W, 1 - (y + h) / H, (x + w) / W, 1 - y / H], aspect=w / h)
        x += w; rowh = max(rowh, h)
    os.makedirs(out, exist_ok=True)
    img.save(os.path.join(out, 'signs_col.png'), optimize=True)
    json.dump(meta, open(os.path.join(out, 'signs.json'), 'w'), indent=1)
    print('signs atlas', len(meta))


if __name__ == '__main__':
    build(sys.argv[1])

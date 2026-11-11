"""Zusaetzliche Texturen (Ausbau 2): U-Bahn-Zugaenge, Passerelle, Altstadt-Pflaster, Leine-Kaimauern, Haltestellen.
Alle prozedural (numpy/PIL), keine Fremdbilder. Schreibt web/assets/city/ground/tex/*.jpg|png (atomar)."""
import os, math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from common import *

TD = OUT + 'tex/'
os.makedirs(TD, exist_ok=True)

def save(im, name, q=88):
    p = TD + name; tmp = TD + '.' + name + '.tmp'
    if name.endswith('.jpg'): im.convert('RGB').save(tmp, format='JPEG', quality=q, optimize=True)
    else: im.save(tmp, format='PNG', optimize=True)
    os.replace(tmp, p)

def arr2im(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

def noise(h, w, scales=((1, 1.0),), seed=0):
    r = np.random.default_rng(seed); out = np.zeros((h, w))
    for s, a in scales:
        n = r.random((max(2, h // s), max(2, w // s)))
        # kachelbar: periodisch erweitern und bikubisch skalieren
        n = np.pad(n, ((0, 1), (0, 1)), mode='wrap')
        im = Image.fromarray((n * 255).astype(np.uint8)).resize((w + max(1, w // n.shape[1]), h + max(1, h // n.shape[0])), Image.BICUBIC)
        out += a * np.array(im)[:h, :w] / 255.0
    return out / sum(a for _, a in scales)

def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])

def speckle(h, w, base, var=0.08, seed=1, dark=0.05, light=0.04):
    r = np.random.default_rng(seed)
    n = noise(h, w, ((1, 0.5), (3, 0.3), (12, 0.2)), seed)
    sp = (r.random((h, w)) < dark) * -0.16 + (r.random((h, w)) < light) * 0.10
    v = (n - 0.5) * var * 2 + sp
    return np.clip(np.asarray(base)[None, None, :] + v[..., None], 0, 1)

FONT_B = 'arialbd.ttf'; FONT_R = 'arial.ttf'
def font(sz, bold=True):
    for f in ((FONT_B if bold else FONT_R), 'DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        try: return ImageFont.truetype(f, sz)
        except OSError: pass
    return ImageFont.load_default()

# ---------------------------------------------------------------- Granit-Stufen (1 m x 1 m Kachel), polierte Bruestung, Wandplatten
save(arr2im(speckle(512, 512, (0.66, 0.66, 0.645), 0.06, 3)), 'granite_step.jpg')
gp = speckle(512, 512, hexc('#7d7d7d'), 0.10, 4, 0.12, 0.10)
save(arr2im(gp), 'granite_polished.jpg')
# Wandverkleidung Trog: polierte graue Granitplatten 1.2 x 0.6 m (Kachel 2.4 x 2.4 m)
gw = speckle(512, 512, (0.56, 0.565, 0.57), 0.07, 5, 0.10, 0.08)
for i in range(4):
    y = i * 128; gw[y:y + 2, :] *= 0.62
    off = 0 if i % 2 == 0 else 128
    for x in range(off, 512 + off, 256): gw[y:y + 128, (x % 512):(x % 512) + 2] *= 0.62
save(arr2im(gw), 'granite_wall.jpg')
# Taktile Leitstreifen: Rippen (Aufmerksamkeitsfeld 0.6 m) und Noppen; hellgrau
tc = np.ones((256, 256, 3)) * 0.80
yy, xx = np.mgrid[0:256, 0:256]
rib = (np.cos(xx / 256 * 2 * np.pi * 8) * 0.5 + 0.5)
tc *= (0.78 + 0.22 * rib)[..., None]
tc[:, :, 2] *= 0.98
save(arr2im(tc + (noise(256, 256, ((2, 1),), 8)[..., None] - 0.5) * 0.06), 'tactile.jpg')
td = np.ones((256, 256, 3)) * 0.80
cx = (xx % 32) - 16; cy = (yy % 32) - 16; r2 = cx * cx + cy * cy
td *= np.where(r2 < 64, 1.08, np.where(r2 < 100, 0.72, 1.0))[..., None]
save(arr2im(td), 'tactile_dots.jpg')

# ---------------------------------------------------------------- Kroepcke-Verteilerebene: Glasmosaik, Decke, Fliesen
def mosaic(base_cols, tile, h=512, seed=0, grout=(0.82, 0.82, 0.78), gw=2):
    r = np.random.default_rng(seed)
    n = -(-h // tile)
    cols = np.array([hexc(c) for c in base_cols])
    pick = cols[r.integers(len(cols), size=(n, n))] * r.uniform(0.85, 1.12, (n, n, 1))
    a = np.kron(pick, np.ones((tile, tile, 1)))[:h, :h].copy()
    a[(np.arange(h) % tile) < gw, :] = grout; a[:, (np.arange(h) % tile) < gw] = grout
    # Glasglanz
    a *= (0.92 + 0.08 * noise(h, h, ((4, 1),), seed + 1))[..., None]
    return a
yy_big = np.mgrid[0:512, 0:512][0]
save(arr2im(mosaic(['#1f6b5f', '#2e7d6e', '#24705f', '#1a5e54', '#3a8a74', '#2a6f66'], 16, seed=11, grout=(0.55, 0.62, 0.58))), 'mosaic_teal.jpg')
save(arr2im(mosaic(['#d9d6a0', '#d2cf96', '#e0dca9', '#cfd09a', '#d7d29b'], 32, seed=12, grout=(0.86, 0.85, 0.80))), 'mosaic_yg.jpg')
save(arr2im(mosaic(['#3e7a3a', '#4f8a45', '#5f9a52', '#7db36a', '#467f40'], 16, seed=13, grout=(0.6, 0.66, 0.58))), 'mosaic_green.jpg')
# Lamellendecke: cremefarbene Lamellen 10 cm, dunkle Fugen (1 m Kachel)
sl = np.ones((256, 256, 3)) * hexc('#ece3c4')
for x in range(0, 256, 26): sl[:, x:x + 4] = hexc('#4a463c')
sl *= (0.95 + 0.05 * noise(256, 256, ((8, 1),), 14))[..., None]
save(arr2im(sl), 'slats.jpg')
# dunkle Linear-Lamellen (Passerelle-Decke, B-Ebene)
sd = np.ones((256, 256, 3)) * hexc('#3a3a3a')
for x in range(0, 256, 16): sd[:, x:x + 3] = hexc('#1c1c1c')
save(arr2im(sd), 'slats_dark.jpg')
# Bodenfliesen beige 30 x 60 (Kachel 1.2 x 1.2 m)
tb = np.ones((512, 512, 3)) * hexc('#d8c8a0')
r = np.random.default_rng(15)
for i in range(4):
    for j in range(2):
        tb[i * 128:(i + 1) * 128, j * 256:(j + 1) * 256] *= r.uniform(0.95, 1.04)
tb *= (0.96 + 0.04 * noise(512, 512, ((2, .5), (16, .5)), 16))[..., None]
for i in range(0, 512, 128): tb[i:i + 2, :] = hexc('#bfb08a')
for j in range(0, 512, 256): tb[:, j:j + 2] = hexc('#bfb08a')
save(arr2im(tb), 'tile_beige.jpg')
# Leitstreifen grau (Rippen in Laufrichtung)
gs = np.ones((128, 128, 3)) * 0.42
gs *= (0.8 + 0.2 * (np.cos(np.mgrid[0:128, 0:128][1] / 128 * 2 * np.pi * 6) * 0.5 + 0.5))[..., None]
save(arr2im(gs), 'guidance.jpg')
# weisse Wandfliesen (Steintor-Station)
wt = np.ones((256, 256, 3)) * 0.90
for i in range(0, 256, 32): wt[i:i + 2, :] = 0.75; wt[:, i:i + 2] = 0.75
save(arr2im(wt), 'tile_white.jpg')

# ---------------------------------------------------------------- Passerelle: Betonfertigteile, dunkle Granitplatten, Decke
cp = speckle(512, 512, hexc('#a8a8a4'), 0.05, 17, 0.03, 0.03)
cp *= (0.95 + 0.05 * noise(512, 512, ((32, 1),), 18))[..., None]
for x in (0, 256): cp[:, x:x + 3] = hexc('#5d5d5a')            # Fugen alle 2.5 m (Kachel 5 x 2.5 m -> 512 x 256?)
for y in (0, 256): cp[y:y + 3, :] = hexc('#5d5d5a')
save(arr2im(cp), 'concrete_panel.jpg')
gd = speckle(512, 512, hexc('#4a4a4c'), 0.06, 19, 0.05, 0.06)
for i in range(8):
    y = i * 64; gd[y:y + 2, :] = hexc('#2f2f30')
    off = 0 if i % 2 == 0 else 128
    for x in range(off, 512 + off, 256): gd[y:y + 64, (x % 512):(x % 512) + 2] = hexc('#2f2f30')
save(arr2im(gd), 'granite_dark.jpg')
# heller Naturstein (ueberdeckte Abschnitte)
gl = speckle(512, 512, hexc('#c9c4b8'), 0.05, 20, 0.04, 0.04)
for i in range(4):
    y = i * 128; gl[y:y + 2, :] = hexc('#a49f94')
    for x in range(0, 512, 128): gl[y:y + 128, x:x + 2] = hexc('#a49f94')
save(arr2im(gl), 'stone_light.jpg')

# ---------------------------------------------------------------- Rolltreppe: Stufenrillen, Riffelblech
es = np.ones((256, 256, 3)) * hexc('#5a5c60')
xs = np.arange(256)
es[:, (xs % 8) < 3] = hexc('#2a2b2e')
es[:, :14] = hexc('#e8c21a'); es[:, -14:] = hexc('#e8c21a')         # gelbe Seitenmarkierung
es[:10, :] = hexc('#e8c21a')                                        # Vorderkante
save(arr2im(es), 'esc_step.jpg')
ep = np.ones((256, 256, 3)) * hexc('#9a9ca0')
for i in range(0, 256, 32):
    for j in range(0, 256, 32):
        d = ImageDraw.Draw(Image.new('L', (1, 1)))
ii, jj = np.mgrid[0:256, 0:256]
lozenge = ((ii % 32 - 16) * 0.7 + (jj % 32 - 16) * 0.7) ** 2 / 30 + ((ii % 32 - 16) * 0.7 - (jj % 32 - 16) * 0.7) ** 2 / 4 < 6
ep[lozenge] *= 1.18
ep *= (0.92 + 0.08 * noise(256, 256, ((4, 1),), 21))[..., None]
save(arr2im(ep), 'esc_plate.jpg')
# Edelstahl gebuerstet (helle Streifen)
st = np.ones((256, 256, 3)) * 0.72
st *= (0.93 + 0.07 * noise(256, 256, ((1, 1),), 22)[:, :1].repeat(256, 1))[..., None]
save(arr2im(st), 'stainless.jpg')

# ---------------------------------------------------------------- Schilder: U, Stationsname, Netzplan, Wegweiser
def sign_u(size=256):
    im = Image.new('RGB', (size, size), (31, 79, 160)); d = ImageDraw.Draw(im)
    f = font(int(size * 0.80)); bb = d.textbbox((0, 0), 'U', font=f)
    d.text(((size - (bb[2] - bb[0])) / 2 - bb[0], (size - (bb[3] - bb[1])) / 2 - bb[1]), 'U', fill=(255, 255, 255), font=f)
    d.rectangle([0, 0, size - 1, size - 1], outline=(235, 235, 235), width=int(size * 0.035))
    return im
save(sign_u(), 'sign_u2.png')
def band(text, w=1024, h=128, bg=(28, 46, 110), fg=(255, 255, 255), pad=0.12, align='center', fsz=None, bold=True):
    im = Image.new('RGB', (w, h), bg); d = ImageDraw.Draw(im)
    f = font(fsz or int(h * 0.62), bold)
    bb = d.textbbox((0, 0), text, font=f); tw = bb[2] - bb[0]
    if tw > w * 0.94:
        f = font(int((fsz or int(h * 0.62)) * w * 0.94 / tw), bold); bb = d.textbbox((0, 0), text, font=f); tw = bb[2] - bb[0]
    x = (w - tw) / 2 - bb[0] if align == 'center' else w * pad - bb[0]
    d.text((x, (h - (bb[3] - bb[1])) / 2 - bb[1]), text, fill=fg, font=f)
    return im
save(band('Kröpcke'), 'sign_kroepcke.png')
save(band('Steintor'), 'sign_steintor.png')
save(band('Markthalle/Landtag'), 'sign_markthalle.png')
save(band('Hauptbahnhof'), 'sign_hbf.png')
save(band('Aegidientorplatz'), 'sign_aegi.png')
save(band('Waterlooplatz'), 'sign_waterloo.png')
# Leitsystem-Band (dunkelblau, Liniennummern farbig)
lb = Image.new('RGB', (2048, 128), (28, 46, 110)); d = ImageDraw.Draw(lb)
f = font(64); fs = font(52)
x = 30
for txt, lines in (('Passerelle · Hauptbahnhof', []), ('Bahnsteige', [('1', (230, 40, 40)), ('2', (230, 40, 40)), ('8', (230, 40, 40)), ('3', (40, 120, 220)), ('7', (40, 120, 220)), ('9', (40, 120, 220)), ('4', (250, 190, 0)), ('5', (250, 190, 0)), ('6', (250, 190, 0)), ('11', (250, 190, 0))]), ('Ausgang Kröpcke', [])):
    d.text((x, 26), txt, font=f, fill=(255, 255, 255)); x += d.textbbox((0, 0), txt, font=f)[2] + 30
    for num, col in lines:
        d.rectangle([x, 22, x + 76, 106], fill=col); bb = d.textbbox((0, 0), num, font=fs)
        d.text((x + 38 - (bb[2] - bb[0]) / 2 - bb[0], 64 - (bb[3] - bb[1]) / 2 - bb[1]), num, font=fs, fill=(255, 255, 255)); x += 90
    x += 50
save(lb, 'sign_guide.png')
# Netzplan-artige Tafel (abstrakt, keine echten Daten)
nm = Image.new('RGB', (512, 512), (246, 246, 242)); d = ImageDraw.Draw(nm)
r = np.random.default_rng(23)
for col in ((230, 40, 40), (40, 120, 220), (250, 190, 0), (60, 160, 70), (150, 80, 160), (240, 120, 30)):
    pts = [(int(r.uniform(30, 480)), int(r.uniform(30, 480)))]
    for k in range(6):
        px, py = pts[-1]
        if k % 2: px = int(np.clip(px + r.uniform(-160, 160), 20, 490))
        else: py = int(np.clip(py + r.uniform(-160, 160), 20, 490))
        pts.append((px, py))
    d.line(pts, fill=col, width=9)
    for p in pts: d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(255, 255, 255), outline=(40, 40, 40), width=2)
d.rectangle([0, 0, 511, 60], fill=(28, 46, 110)); d.text((20, 12), 'Stadtbahn Hannover', font=font(34), fill=(255, 255, 255))
save(nm, 'netmap.png')
# Werbeplakat-Vitrine (City-Light) – abstrakte Farbflaechen
cl = Image.new('RGB', (256, 512), (240, 240, 235)); d = ImageDraw.Draw(cl)
d.rectangle([0, 0, 255, 300], fill=(200, 40, 50)); d.ellipse([40, 60, 216, 236], fill=(250, 210, 60)); d.rectangle([20, 340, 236, 380], fill=(30, 30, 30)); d.rectangle([20, 400, 180, 420], fill=(120, 120, 120))
save(cl, 'citylight.png')
# Fahrgastanzeiger (schwarz, orange LED)
dp = Image.new('RGB', (512, 128), (10, 10, 10)); d = ImageDraw.Draw(dp)
fm = font(30)
for i, (ln, ziel, t) in enumerate((('10', 'Ahlem', '2 min'), ('17', 'Wallensteinstr.', '5 min'), ('10', 'Hauptbahnhof', '8 min'))):
    y = 6 + i * 40
    d.text((10, y), ln, font=fm, fill=(255, 150, 20)); d.text((70, y), ziel, font=fm, fill=(255, 150, 20)); d.text((400, y), t, font=fm, fill=(255, 150, 20))
save(dp, 'display_tram.png')
# Fahrkartenautomat-Front
ta = Image.new('RGB', (256, 512), (190, 192, 196)); d = ImageDraw.Draw(ta)
d.rectangle([20, 30, 236, 70], fill=(200, 30, 40)); d.text((34, 36), 'Fahrkarten', font=font(28), fill=(255, 255, 255))
d.rectangle([40, 100, 216, 230], fill=(25, 40, 70)); d.rectangle([48, 108, 208, 222], fill=(70, 120, 180))
for i in range(4):
    for j in range(3): d.rectangle([60 + j * 50, 260 + i * 34, 96 + j * 50, 286 + i * 34], fill=(60, 60, 64))
d.rectangle([80, 410, 176, 440], fill=(30, 30, 30)); d.rectangle([60, 460, 196, 490], fill=(80, 80, 84))
save(ta, 'ticket_machine.png')

# ---------------------------------------------------------------- Altstadt-Pflaster
def setts_arcs(h=1024, seed=30, base=(0.56, 0.545, 0.52), arc_r=1.0, stone=0.09, tile_m=4.0, redfrac=0.07):
    """Granit-Kleinpflaster im Segmentbogenverband. Kachel tile_m Meter, Boegen Radius arc_r (m)."""
    r = np.random.default_rng(seed)
    px = h / tile_m
    img = Image.new('RGB', (h, h), (112, 109, 104)); d = ImageDraw.Draw(img)
    s = stone * px
    # Boegen: Reihen von Halbkreis-Segmenten, Steine entlang konzentrischer Boegen
    R = arc_r * px
    rows = int(round(h / (R * 0.5)))
    for k in range(-2, rows + 1):
        cy = k * R * 0.5
        for cxi in range(-1, int(round(h / (R * 2))) + 1):
            cx = cxi * R * 2 + (R if k % 2 else 0)
            for ring in np.arange(s * 0.55, R * 1.02, s * 1.04):
                n = max(2, int(math.pi * ring / (s * 1.04)))
                for a in np.linspace(0.0, math.pi, n + 1)[:-1] + math.pi / (2 * (n + 1)):
                    x = cx + ring * math.cos(a); y = cy + ring * math.sin(a)
                    if not (-s < x < h + s and -s < y < h + s): continue
                    f = r.uniform(0.75, 1.2)
                    col = np.array(base) * f
                    if r.random() < redfrac: col = col * np.array([1.07, 0.96, 0.91])
                    c = tuple(int(np.clip(v, 0, 1) * 255) for v in col)
                    w2 = s * r.uniform(0.47, 0.53); h2 = s * r.uniform(0.45, 0.51)
                    ca, sa = math.cos(a), math.sin(a)
                    pts = [(x + (-w2) * -sa + (-h2) * ca, y + (-w2) * ca + (-h2) * sa), (x + (w2) * -sa + (-h2) * ca, y + (w2) * ca + (-h2) * sa),
                           (x + (w2) * -sa + (h2) * ca, y + (w2) * ca + (h2) * sa), (x + (-w2) * -sa + (h2) * ca, y + (-w2) * ca + (h2) * sa)]
                    for dx in (-h, 0, h):
                        for dy in (-h, 0, h):
                            d.polygon([(px_ + dx, py_ + dy) for px_, py_ in pts], fill=c)
    return img.filter(ImageFilter.GaussianBlur(0.7))
save(setts_arcs(), 'setts_arc.jpg')
# Klinker rot im Fischgraet, Granitband-Raster 3 m (Kachel 3 m)
def clinker(h=1024, seed=31, tile_m=3.0, band_m=0.16):
    r = np.random.default_rng(seed)
    px = h / tile_m
    img = Image.new('RGB', (h, h), (120, 70, 55)); d = ImageDraw.Draw(img)
    bl, bw = 0.20 * px, 0.10 * px            # Klinker 20 x 10 cm
    for i in range(-2, int(h / bw) + 3):
        for j in range(-2, int(h / bl) + 3):
            for o in (0, 1):
                f = r.uniform(0.82, 1.15); base = hexc('#9c4a3a') * f
                if r.random() < 0.15: base = hexc('#7a3a2e') * f
                c = tuple(int(np.clip(v, 0, 1) * 255) for v in base)
                if o == 0:
                    x0 = j * bl + (i % 2) * bw; y0 = i * bw * 2
                    pts = [(x0, y0), (x0 + bl - 2, y0), (x0 + bl - 2, y0 + bw - 2), (x0, y0 + bw - 2)]
                else:
                    x0 = j * bl + (i % 2) * bw + bl - bw; y0 = i * bw * 2 + bw
                    pts = [(x0, y0 - bw), (x0 + bw - 2, y0 - bw), (x0 + bw - 2, y0 + bl - bw - 2), (x0, y0 + bl - bw - 2)]
                # 45 Grad drehen um Kachelmitte
                q = [((p[0] - p[1]) * 0.7071 + h / 2, (p[0] + p[1]) * 0.7071 - h * 0.2) for p in pts]
                d.polygon(q, fill=c)
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    a = np.array(img).astype(float) / 255
    bpx = int(band_m * px)
    g = speckle(h, h, (0.62, 0.61, 0.58), 0.06, seed + 1)
    a[:bpx, :] = g[:bpx, :]; a[:, :bpx] = g[:, :bpx]
    return arr2im(a)
save(clinker(), 'clinker.jpg')
# Granit-Grosspflaster in Reihen (Fahrbahn Altstadt), Kachel 2 m
def setts_rows(h=512, seed=32, tile_m=2.0, base=(0.52, 0.51, 0.49)):
    r = np.random.default_rng(seed); px = h / tile_m
    img = Image.new('RGB', (h, h), (60, 58, 55)); d = ImageDraw.Draw(img)
    rh = 0.14 * px
    for i in range(int(h / rh) + 1):
        x = -r.uniform(0, 0.16 * px)
        while x < h:
            w = r.uniform(0.13, 0.20) * px; f = r.uniform(0.75, 1.2)
            c = tuple(int(np.clip(v * f, 0, 1) * 255) for v in base)
            for dx in (0, h): d.rectangle([x + dx + 1, i * rh + 1, x + dx + w - 2, i * rh + rh - 2], fill=c)
            x += w
    return img.filter(ImageFilter.GaussianBlur(0.6))
save(setts_rows(), 'setts_rows.jpg')
# Sandstein-Quader (Kaimauer), Kachel 2.4 x 2.4 m
ss = speckle(512, 512, hexc('#b8a98c'), 0.07, 33, 0.05, 0.05)
r = np.random.default_rng(34)
for i in range(8):
    y = i * 64; ss[y:y + 3, :] *= 0.7
    x = int(r.uniform(0, 128))
    while x < 512 + 128:
        ss[y:y + 64, (x % 512):(x % 512) + 3] *= 0.7
        ss[y:y + 64, (x % 512) + 3:min(512, (x % 512) + int(r.uniform(90, 170)))] *= r.uniform(0.88, 1.06)
        x += int(r.uniform(90, 170))
# Verwitterung unten dunkler (v=1 unten)
ss *= np.linspace(1.0, 0.82, 512)[:, None, None]
save(arr2im(ss), 'sandstone.jpg')
# Wasser (Fluss): dunkles Gruen-Braun mit feinen Wellen
wa = np.ones((512, 512, 3)) * hexc('#2c3a33')
wa *= (0.85 + 0.3 * noise(512, 512, ((3, .4), (12, .4), (48, .2)), 35))[..., None]
save(arr2im(wa), 'river.jpg')
# Holz-Ladenfront blau, Markise etc. nicht noetig
print('textures2 done', len(os.listdir(TD)))

"""Texturen fuer den Boden: Wiederverwendung der Detailmodell-Texturen + prozedurale Ergaenzungen.
Schreibt web/assets/city/ground/tex/*.jpg|png"""
import os, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import trimesh
from common import *

TD = OUT + 'tex/'
os.makedirs(TD, exist_ok=True)
rng = np.random.default_rng(7)

def save(im, name, q=88):
    p = TD + name; tmp = TD + '.' + name + '.tmp'
    if name.endswith('.jpg'): im.convert('RGB').save(tmp, format='JPEG', quality=q, optimize=True)
    else: im.save(tmp, format='PNG', optimize=True)
    os.replace(tmp, p)               # atomar tauschen (Spiel laeuft)
    return p

def fit(im, size):
    return im.resize((size, size), Image.LANCZOS) if im.size != (size, size) else im

def noise(h, w, scales=((1, 1.0),), seed=0):
    r = np.random.default_rng(seed); out = np.zeros((h, w))
    for s, a in scales:
        n = r.random((max(2, h // s), max(2, w // s)))
        out += a * np.array(Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255.0
    return out / sum(a for _, a in scales)

def tileable(a):
    """weiche Kachelbarkeit: Mischung mit halb verschobener Kopie an den Raendern"""
    h, w = a.shape[:2]
    b = np.roll(np.roll(a, h // 2, 0), w // 2, 1)
    y = np.abs(np.linspace(-1, 1, h))[:, None]; x = np.abs(np.linspace(-1, 1, w))[None, :]
    m = np.clip(np.maximum(x, y) * 1.6 - 0.6, 0, 1)
    if a.ndim == 3: m = m[..., None]
    return a * (1 - m) + b * m

# ---------------------------------------------------------------- Detailmodell-Texturen
src = {'asphalt.jpg': 'ground_asphalt_61feabfb.png', 'asphalt2.jpg': 'ground_asphalt_5cceef0e.png', 'grass.jpg': 'ground_grass_69e1b23b.png',
       'setts_small.jpg': 'ground_setts_e5a0df7b.png', 'pavers.jpg': 'ground_slabs_b23c554f.png', 'slabs_dark.jpg': 'ground_slabs_dc371902.png'}
for dst, s in src.items():
    im = Image.open(TEX + s).convert('RGB')
    save(fit(im, 1024 if im.size[0] > 600 else 512), dst)
# aus street.glb eingebettet: Platten (f833e41b) und Pflaster (84d7cf54) wie im Detailmodell
sc = trimesh.load(DATA + 'rhino_export/street.glb', force='scene')
got = set()
for g in sc.geometry.values():
    m = g.visual.material; n = getattr(m, 'name', '')
    for key, dst in (('slabs f833e41b', 'slabs_ped.jpg'), ('setts 84d7cf54', 'setts_road.jpg')):
        if key in n and dst not in got and getattr(m, 'baseColorTexture', None) is not None:
            save(fit(m.baseColorTexture.convert('RGB'), 1024), dst); got.add(dst)
print('extracted', got)
# Radweg: Asphalt rot eingefaerbt
a = np.array(Image.open(TD + 'asphalt.jpg')).astype(float) / 255
lum = a.mean(2, keepdims=True)
red = np.clip(lum / lum.mean() * np.array([0.46, 0.26, 0.23]), 0, 1)
save(Image.fromarray((red * 255).astype(np.uint8)), 'asphalt_red.jpg')

# ---------------------------------------------------------------- Granit-Bordstein (2 m pro Kachel, Fugen alle 1 m)
def granite(h, w, base=(0.62, 0.62, 0.60), seed=1, var=0.10):
    n = noise(h, w, ((1, 0.5), (2, 0.3), (8, 0.2)), seed)
    sp = (np.random.default_rng(seed).random((h, w)) < 0.06) * -0.18 + (np.random.default_rng(seed + 1).random((h, w)) < 0.04) * 0.12
    v = (n - 0.5) * var * 2 + sp
    img = np.clip(np.array(base)[None, None, :] + v[..., None], 0, 1)
    return img
k = granite(128, 1024, (0.60, 0.60, 0.585), 3)
for x in (0, 512):
    k[:, max(0, x - 2):x + 2] *= 0.55
k[:6, :] *= 1.08; k[-4:, :] *= 0.85            # Fase / Kante
save(Image.fromarray((np.clip(k, 0, 1) * 255).astype(np.uint8)), 'kerb.jpg')
g = tileable(granite(512, 512, (0.58, 0.58, 0.56), 5))
save(Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)), 'granite.jpg')
# Beton hell (Hochbahnsteig, Waende), mit Plattenfugen 2 m
c = tileable(granite(512, 512, (0.66, 0.655, 0.63), 9, 0.05))
c[:3, :] *= 0.7; c[:, :3] *= 0.7
save(Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)), 'concrete.jpg')
# Wandverkleidung Promenade: Natursteinplatten 1.2 x 0.6 (Kachel 2.4 x 2.4 m)
cl = tileable(granite(512, 512, (0.72, 0.70, 0.66), 11, 0.05))
for y in range(0, 512, 128): cl[y:y + 2, :] *= 0.75
for i, y in enumerate(range(0, 512, 128)):
    off = 0 if i % 2 == 0 else 128
    for x in range(off, 512 + off, 256): cl[y:y + 128, (x % 512):(x % 512) + 2] *= 0.75
save(Image.fromarray((np.clip(cl, 0, 1) * 255).astype(np.uint8)), 'cladding.jpg')

# ---------------------------------------------------------------- Mulch/Erde, Kies, Schotter, Baumrost
def speck(base, sizes, seed, contrast=0.25, n=9000, h=512):
    r = np.random.default_rng(seed)
    im = Image.new('RGB', (h, h), tuple(int(c * 255) for c in base)); d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = r.random(2) * h; s = r.uniform(*sizes)
        f = 1 + r.uniform(-contrast, contrast)
        col = tuple(int(np.clip(c * f * r.uniform(0.9, 1.1), 0, 1) * 255) for c in base)
        for dx in (-h, 0, h):
            for dy in (-h, 0, h):
                d.ellipse([x + dx - s, y + dy - s * r.uniform(0.6, 1), x + dx + s, y + dy + s], fill=col)
    return im.filter(ImageFilter.GaussianBlur(0.6))
save(speck((0.20, 0.14, 0.09), (2, 6), 21, 0.35, 7000), 'soil.jpg')
save(speck((0.66, 0.60, 0.50), (1.5, 3.5), 22, 0.18, 16000), 'gravel.jpg')
save(speck((0.42, 0.40, 0.37), (3, 7), 23, 0.30, 9000), 'ballast.jpg')
gr = Image.new('RGB', (512, 512), (40, 38, 36)); d = ImageDraw.Draw(gr)
for i in range(0, 512, 32):
    d.rectangle([i + 4, 0, i + 26, 511], fill=(22, 21, 20))       # Schlitze
for j in range(0, 512, 128): d.rectangle([0, j, 511, j + 6], fill=(58, 55, 52))
save(gr.filter(ImageFilter.GaussianBlur(0.7)), 'grate.jpg')

# ---------------------------------------------------------------- U-Bahn-Schild (blau, weisses U), Haltestellen-H
def sign(letter, bg, fg, name, size=256):
    im = Image.new('RGB', (size, size), bg); d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype('arialbd.ttf', int(size * 0.78))
    except OSError: font = ImageFont.load_default()
    bb = d.textbbox((0, 0), letter, font=font); w = bb[2] - bb[0]; h = bb[3] - bb[1]
    d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), letter, fill=fg, font=font)
    d.rectangle([0, 0, size - 1, size - 1], outline=fg, width=int(size * 0.04))
    save(im, name)
sign('U', (0, 70, 160), (255, 255, 255), 'sign_u.png')
sign('H', (250, 200, 0), (0, 110, 60), 'sign_h.png')

# ---------------------------------------------------------------- Baum-Texturen (Alpha)
meta = {}
for s, dst in (('Tree Savanna (4).png', 'tree_a.png'), ('Tree Savanna (5).png', 'tree_b.png')):
    im = Image.open(TEX + s).convert('RGBA')
    bb = im.getbbox(); im = im.crop(bb)
    sc_ = 1024 / max(im.size); im = im.resize((max(4, int(im.size[0] * sc_)), max(4, int(im.size[1] * sc_))), Image.LANCZOS)
    # auf quadratische 1024er Leinwand, unten mittig (Stamm unten)
    can = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0)); can.paste(im, ((1024 - im.size[0]) // 2, 1024 - im.size[1]), im)
    a = np.array(can); rgb = a[..., :3].astype(float); al = a[..., 3]
    # Farbsaum vermeiden: transparente Pixel mit Nachbarfarbe fuellen
    from scipy import ndimage as nd
    idx = nd.distance_transform_edt(al < 128, return_distances=False, return_indices=True)
    rgb = rgb[idx[0], idx[1]]
    # Stamm unterhalb der Krone entfernen (3D-Stamm ersetzt ihn); Kronenansatz merken
    w_ = (al > 128).sum(1); rows = np.nonzero(w_ > 0.06 * 1024)[0]
    crown_bot = int(rows.max()) if len(rows) else 1000
    al = al.copy(); al[crown_bot + 6:, :] = 0
    a = np.dstack([rgb.astype(np.uint8), al])
    save(Image.fromarray(a, 'RGBA').quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE), dst)
    meta[dst] = {'crown_base': round(1 - (crown_bot + 6) / 1024, 3)}
    print(dst, can.size, 'crown bbox', bb)
import json
json.dump(meta, open(CACHE + 'tree_meta.json', 'w'))
print('textures in', TD, sorted(os.listdir(TD)), meta)

# ---------------------------------------------------------------- weitere: Hecke, Schaufenster, Zaun (Alpha), Rolltor, Plakat
def leafy(base, seed, n=14000, h=512):
    r = np.random.default_rng(seed)
    im = Image.new('RGB', (h, h), tuple(int(c * 160) for c in base)); d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = r.random(2) * h; s = r.uniform(2, 6); f = r.uniform(0.55, 1.35)
        col = tuple(int(np.clip(c * f, 0, 1) * 255) for c in base)
        for dx in (-h, 0, h):
            for dy in (-h, 0, h):
                d.ellipse([x + dx - s, y + dy - s * 0.6, x + dx + s, y + dy + s * 0.6], fill=col)
    return im.filter(ImageFilter.GaussianBlur(0.5))
save(leafy((0.20, 0.36, 0.13), 31), 'hedge.jpg')
# Schaufenster: dunkle Verglasung mit Pfosten alle 2.4 m, Sockel, Kaempfer
sf = np.zeros((512, 512, 3)); sf[:] = (0.10, 0.13, 0.16)
yy, xx = np.mgrid[0:512, 0:512]
sf += (0.10 * (1 - yy / 512))[..., None] * np.array([0.6, 0.8, 1.0])          # Spiegelung
sf[:, :10] = (0.55, 0.56, 0.58); sf[:, 251:261] = (0.55, 0.56, 0.58)
sf[:28, :] = (0.62, 0.62, 0.60); sf[470:, :] = (0.45, 0.45, 0.44); sf[120:128, :] = (0.55, 0.56, 0.58)
for x0, col in ((40, (0.85, 0.75, 0.55)), (300, (0.75, 0.80, 0.85))):
    sf[200:420, x0:x0 + 160] = sf[200:420, x0:x0 + 160] * 0.5 + np.array(col) * 0.35   # Auslagen (hell)
save(Image.fromarray((np.clip(sf, 0, 1) * 255).astype(np.uint8)), 'shopfront.jpg')
# Zaun: vertikale Staebe mit Alpha
fz = np.zeros((256, 256, 4), np.uint8)
for x in range(0, 256, 32): fz[:, x:x + 5] = (40, 42, 44, 255)
fz[0:10, :] = (40, 42, 44, 255); fz[236:246, :] = (40, 42, 44, 255)
save(Image.fromarray(fz, 'RGBA'), 'fence.png')
# Rolltor
rt = np.zeros((256, 256, 3)); rt[:] = (0.50, 0.52, 0.54)
for y in range(0, 256, 8): rt[y:y + 2] *= 0.7
save(Image.fromarray((np.clip(rt, 0, 1) * 255).astype(np.uint8)), 'shutter.jpg')
# Plakatflaeche Litfasssaeule
r = np.random.default_rng(5); pl = Image.new('RGB', (512, 256), (230, 225, 210)); d = ImageDraw.Draw(pl)
x = 0
while x < 512:
    w = int(r.uniform(70, 130)); col = tuple(int(c) for c in r.integers(30, 230, 3))
    d.rectangle([x + 3, 8, x + w - 3, 248], fill=col)
    for k in range(3):
        y = int(r.uniform(20, 200)); d.rectangle([x + 10, y, x + w - 10, y + int(r.uniform(8, 30))], fill=tuple(int(c) for c in r.integers(0, 255, 3)))
    x += w
save(pl, 'poster.jpg')
# Holz (Bank)
wd = np.array(Image.open(TEX + 'Beech red_100_DP.jpg').convert('RGB').resize((256, 256)))
save(Image.fromarray(wd), 'wood.jpg')
print('extra textures done')

"""Marktkirche -- prozedurale, kachelbare PBR-Texturen (1024 px, numpy/PIL) -> build/tex/*.jpg|png
Farbe: JPEG (sRGB), Normalen: PNG (OpenGL, +Y = +v). Bildzeile 0 = oben = v max (Arrays werden in v-aufwaerts erzeugt und gespiegelt)."""
import os, math
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'build', 'tex'); os.makedirs(OUT, exist_ok=True)
N = 1024
rng = np.random.default_rng(1907)


def pnoise(sigma_px, seed, n=N):
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    g = np.exp(-(fx ** 2 + fy ** 2) * (2 * math.pi * sigma_px) ** 2 / 2)
    a = np.real(np.fft.ifft2(np.fft.fft2(w) * g))
    a -= a.mean(); a /= (a.std() + 1e-9)
    return a


def aniso_noise(sx, sy, seed, n=N):
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    g = np.exp(-((fx * sx) ** 2 + (fy * sy) ** 2) * (2 * math.pi) ** 2 / 2)
    a = np.real(np.fft.ifft2(np.fft.fft2(w) * g)); a -= a.mean(); a /= (a.std() + 1e-9)
    return a


def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def save_rgb(name, a, q=92):
    im = Image.fromarray((np.clip(np.flipud(a), 0, 1) * 255 + 0.5).astype(np.uint8), 'RGB')
    im.save(os.path.join(OUT, name + '.jpg'), quality=q)


def save_normal(name, h, strength):
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    n = np.dstack([-dx * strength, -dy * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    im = Image.fromarray((np.flipud(n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8), 'RGB')
    im.save(os.path.join(OUT, name + '_n.png'), optimize=True)


def smooth_edge(d, w):
    return np.clip(d / w, 0, 1) ** 0.6


yy, xx = np.mgrid[0:N, 0:N]


def brick_layout(tile_m=2.0, course=0.1, unit=0.4, s_len=0.264, joint=0.012):
    """Gotischer Verband (Laeufer/Binder im Wechsel), 20 Schichten x 5 Einheiten je 2 m."""
    m = tile_m / N
    x = (xx + 0.5) * m; y = (yy + 0.5) * m
    c = np.floor(y / course).astype(int)
    shift = (c % 2) * (unit / 2)
    xs = (x + shift) % tile_m
    ui = np.floor(xs / unit).astype(int); xu = xs - ui * unit
    stretcher = xu < s_len
    x0 = np.where(stretcher, 0.0, s_len); x1 = np.where(stretcher, s_len, unit)
    dx = np.minimum(xu - x0, x1 - xu) - joint / 2
    yc = y - c * course
    dy = np.minimum(yc, course - yc) - joint / 2
    d = np.minimum(dx, dy)
    nb = int(round(tile_m / unit))
    bid = (c % int(round(tile_m / course))) * (nb * 2) + ui * 2 + (~stretcher).astype(int)
    return d, bid, stretcher, c


def make_brick(name, palette, p_dark_head, p_dark_str, dark_pal, mortar='#9b958b', seed=0, glazed_all=False):
    d, bid, stretcher, c = brick_layout()
    r = np.random.default_rng(seed)
    nb = bid.max() + 1
    base = palette[r.integers(0, len(palette), nb)] * (0.86 + 0.24 * r.random(nb))[:, None]
    dk = dark_pal[r.integers(0, len(dark_pal), nb)] * (0.85 + 0.3 * r.random(nb))[:, None]
    roll = r.random(nb)
    is_head = np.zeros(nb, bool)
    hb = bid[~stretcher]; is_head[np.unique(hb)] = True
    usedark = np.where(is_head, roll < p_dark_head, roll < p_dark_str)
    col = np.where(usedark[:, None], dk, base)
    C = col[bid]
    n1 = pnoise(1.2, seed + 1); n2 = pnoise(6, seed + 2); n3 = pnoise(60, seed + 3)
    C = C * (1 + 0.05 * n1[..., None] + 0.06 * n2[..., None] + 0.05 * n3[..., None])
    # Brandflecken / Kanten dunkler
    edge = np.clip(1 - d / 0.01, 0, 1)
    C = C * (1 - 0.18 * edge[..., None])
    M = hexc(mortar) * (1 + 0.08 * pnoise(1.5, seed + 4)[..., None] + 0.05 * n3[..., None])
    inb = d > 0
    A = np.where(inb[..., None], C, M)
    # leichte Verschmutzung (vertikale Schlieren)
    st = aniso_noise(3, 24, seed + 5)
    A = A * (1 - 0.06 * np.clip(st, 0, 3)[..., None])
    save_rgb(name, A)
    h = np.where(inb, 0.6 + 0.4 * smooth_edge(d, 0.006), 0.0) + 0.04 * n1 + 0.03 * pnoise(3, seed + 6)
    save_normal(name, h, 6.0)


def ashlar_layout(tile_m, seed, rows):
    """Quaderverband: Reihen mit Hoehe aus rows, Blocklaengen zufaellig 0.55-1.25 m (kachelbar)."""
    r = np.random.default_rng(seed)
    m = tile_m / N
    y = (yy + 0.5) * m; x = (xx + 0.5) * m
    edges = np.cumsum([0] + rows); edges = edges / edges[-1] * tile_m
    ri = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, len(rows) - 1)
    bid = np.zeros((N, N), int); d = np.zeros((N, N))
    joint = 0.008; k = 0
    for i in range(len(rows)):
        L = []
        while sum(L) < tile_m - 0.5:
            L.append(r.uniform(0.55, 1.25))
        L[-1] += tile_m - sum(L)
        off = r.uniform(0, tile_m)
        cuts = (off + np.cumsum([0] + L)) % tile_m
        cuts = np.sort(cuts)
        msk = ri == i
        xr = x[msk]
        j = np.searchsorted(cuts, xr, side='right') - 1
        left = np.where(j >= 0, cuts[j % len(cuts)], cuts[-1] - tile_m)
        right = np.where(j + 1 < len(cuts), cuts[np.minimum(j + 1, len(cuts) - 1)], cuts[0] + tile_m)
        dxx = np.minimum(xr - left, right - xr)
        yr = y[msk]
        dyy = np.minimum(yr - edges[i], edges[i + 1] - yr)
        d[msk] = np.minimum(dxx, dyy) - joint / 2
        bid[msk] = k + (j % len(cuts))
        k += len(cuts)
    return d, bid


def make_stone(name, tile_m, base_hex, var, seed, rows=None, dark_crust=0.0, strength=5.0):
    r = np.random.default_rng(seed)
    if rows:
        d, bid = ashlar_layout(tile_m, seed, rows)
        nb = bid.max() + 1
        bc = (0.9 + 0.2 * r.random(nb))[bid]
    else:
        d = np.full((N, N), 1.0); bc = np.ones((N, N))
    base = hexc(base_hex)
    n1 = pnoise(0.8, seed + 1); n2 = pnoise(4, seed + 2); n3 = pnoise(40, seed + 3)
    st = aniso_noise(2.5, 30, seed + 4)
    A = base[None, None] * bc[..., None] * (1 + var * (0.06 * n1 + 0.08 * n2 + 0.08 * n3))[..., None]
    A = A * (1 - 0.12 * np.clip(st, 0, 3))[..., None]
    if dark_crust > 0:
        cr = np.clip((0.6 * pnoise(18, seed + 7) + 0.8 * aniso_noise(4, 40, seed + 9) - 0.4) * 0.8, 0, 1) * dark_crust
        A = A * (1 - 0.3 * cr)[..., None]
    if rows:
        A = np.where((d > 0)[..., None], A, A * 0.72)
    save_rgb(name, A)
    h = 0.05 * n1 + 0.08 * n2 + 0.05 * pnoise(1.6, seed + 8)
    if rows: h = h + np.where(d > 0, 0.5 + 0.3 * smooth_edge(d, 0.01), 0.0)
    save_normal(name, h, strength)


# ------------------------------------------------------------------------------------------- Backstein
# Farben nach Fotos 01/04/13 (QA: #8e3b2a / #7A3B2A, viele dunkle Binder), etwas entsaettigt; dunklere Fuge
red = np.array([hexc(c) for c in ('#6f3326', '#77372a', '#663024', '#7a3b2c', '#5f2c22', '#71382c', '#7e4232', '#663629')])
dark = np.array([hexc(c) for c in ('#3a241f', '#31201c', '#432a24', '#2b1e1b', '#472d26')])
make_brick('backstein', red, 0.62, 0.1, dark, mortar='#7b746a', seed=10)
gl = np.array([hexc(c) for c in ('#2e2420', '#3a2a24', '#30261f', '#43302a', '#262020')])
make_brick('glasurstein', gl, 1.0, 0.82, gl * 0.9, mortar='#857f76', seed=20)
for c in ('#7d3a2a',):
    pass
# ------------------------------------------------------------------------------------------- Naturstein
make_stone('sandstein_quader', 2.4, '#58534b', 1.2, 30, rows=[0.42, 0.38, 0.45, 0.4, 0.36, 0.39], dark_crust=1.0)   # Sockel: dunkel verwittert (Fotos 01/04)
make_stone('sandstein', 1.6, '#a2967f', 1.0, 40, None, dark_crust=0.35, strength=4.0)
make_stone('sandstein_dunkel', 2.4, '#615c53', 1.15, 50, rows=[0.46, 0.4, 0.5, 0.42, 0.38, 0.44], dark_crust=0.85)   # Fotos 04/16: Westportal + Eckquader fast schwarz verwittert
# Sockelstein Lutherdenkmal: dunkler, gruenlich verlaufener Monolith (Foto 13)
make_stone('sockelstein', 1.6, '#66655d', 1.0, 55, None, dark_crust=0.7, strength=3.0)
# Granit (Sockel, Stufen)
n1 = pnoise(0.6, 61); n2 = pnoise(2.5, 62)
g = hexc('#8c8984')[None, None] * (1 + 0.05 * n2)[..., None]
sp = rng.random((N, N))
g = np.where((sp < 0.07)[..., None], g * 0.45, g); g = np.where((sp > 0.95)[..., None], np.minimum(g * 1.35, 1), g)
save_rgb('granit', g)
save_normal('granit', 0.1 * n1 + 0.05 * n2, 2.0)
# ------------------------------------------------------------------------------------------- Kupfer
st = aniso_noise(2.0, 40, 70); n2 = pnoise(5, 71); n3 = pnoise(80, 72)
base = hexc('#454d49')
pat = hexc('#5c6a63')
t = np.clip(0.35 + 0.25 * st + 0.15 * n3, 0, 1)
A = base[None, None] * (1 - t[..., None]) + pat[None, None] * t[..., None]
A = A * (1 + 0.05 * n2[..., None])
m = 2.0 / N
yv = (yy + 0.5) * m
seamh = np.abs(((yv + 0.5) % 1.0) - 0.5) < 0.004        # Quernaehte alle 1 m
A = np.where(seamh[..., None], A * 0.8, A)
save_rgb('kupfer', A)
save_normal('kupfer', 0.03 * n2 + np.where(seamh, 0.3, 0.0) + 0.02 * st, 4.0)
st = pnoise(2.5, 75); n3 = pnoise(25, 76)
A = hexc('#77857b')[None, None] * (1 + 0.07 * st + 0.08 * n3)[..., None]          # stumpfes graugruenes Kupfer/Blei (Fotos 01, 16)
A = np.where((pnoise(6, 77) > 1.3)[..., None], A * np.array([0.75, 0.7, 0.62]), A)
save_rgb('kupfer_patina', A)
save_normal('kupfer_patina', 0.05 * st, 2.0)
# Bronze (Lutherfigur), Patina mit dunklen Laeufen
st = aniso_noise(8.0, 45, 80); n2 = pnoise(10, 81); n3 = pnoise(40, 82)
A = hexc('#5a7b72')[None, None] * (1 + 0.07 * n2 + 0.08 * n3)[..., None]          # stumpfe graugruene Patina (Foto 13), nicht tuerkis
br_ = np.clip(0.5 * pnoise(14, 83) + 0.4 * aniso_noise(6, 30, 84) - 0.35, 0, 1)[..., None]
A = A * (1 - br_) + hexc('#3f4038') * br_                                            # dunkle Bronze, wo die Patina abgewaschen ist
A = A * (1 - 0.38 * np.clip(st - 0.2, 0, 2) / 1.8)[..., None]                        # dunkle Regenlaeufe
save_rgb('bronze', A)
save_normal('bronze', 0.04 * n2, 1.5)
# ------------------------------------------------------------------------------------------- Biberschwanz
TW, TH_, NC = 1.08, 0.99, 6
mx = TW / N; my = TH_ / N
x = (xx + 0.5) * mx; y = (yy + 0.5) * my
c = np.floor(y / (TH_ / NC)).astype(int)
yc = y - c * (TH_ / NC)                         # 0 = Unterkante der Lage (traufseitig)
sh = (c % 2) * 0.09
xt = (x + sh) % TW
ti = np.floor(xt / 0.18).astype(int); xl = xt - ti * 0.18
tid = (c % NC) * 6 + ti
rr = np.random.default_rng(90)
pal = np.array([hexc(q) for q in ('#9a4a35', '#a24f37', '#8f4431', '#aa5638', '#94472f', '#86402d', '#b5532f')])
pal = pal * 0.88 * 0.9 + pal.mean(axis=1, keepdims=True) * 0.88 * 0.1              # dunkler + etwas entsaettigt (QA)
tc = pal[rr.integers(0, len(pal), NC * 6)] * (0.88 + 0.2 * rr.random(NC * 6))[:, None]
C = tc[tid]
# runder Schwanz: Unterkante als Segmentbogen; ausserhalb -> Ziegel der darunterliegenden Lage (dunkler, verschattet)
xr = (xl - 0.09) / 0.09
curve = 0.035 * (1 - np.sqrt(np.clip(1 - xr ** 2, 0, 1)))
gapx = (xl < 0.0035) | (xl > 0.18 - 0.0035)
under = yc < curve
n1 = pnoise(1.0, 91); n2 = pnoise(5, 92); n3 = pnoise(70, 93)
A = C * (1 + 0.05 * n1 + 0.05 * n2 + 0.06 * n3)[..., None]
shade = np.clip((yc - curve) / 0.05, 0, 1)
A = A * (0.72 + 0.28 * shade)[..., None]
A = np.where((under | gapx)[..., None], A * 0.45, A)
moss = np.clip(pnoise(3, 94) - 1.6, 0, 1)
A = A * (1 - 0.25 * moss)[..., None] + np.array([0.25, 0.27, 0.15]) * 0.25 * moss[..., None]
save_rgb('dachziegel', A)
h = np.where(under | gapx, 0.0, 0.3 + 0.7 * (1 - yc / (TH_ / NC))) + 0.04 * n1
save_normal('dachziegel', h, 7.0)
# ------------------------------------------------------------------------------------------- Bleiglas (opak, Rechteckscheiben)
# Foto 01: graugruene Rechteckscheiben 0.24 x 0.30 m in Bleiruten, alle 1.2 m ein kraeftiges Windeisen (Kachel 1.2 m)
m = 1.2 / N
x = (xx + 0.5) * m; y = (yy + 0.5) * m
pw, ph = 0.24, 0.30
pi_ = np.floor(x / pw).astype(int); pj = np.floor(y / ph).astype(int)
dxp = np.minimum(x - pi_ * pw, (pi_ + 1) * pw - x); dyp = np.minimum(y - pj * ph, (pj + 1) * ph - y)
lead = np.minimum(dxp, dyp) < 0.0045
bar = np.minimum(y, 1.2 - y) < 0.009                    # Windeisen an der Kachelkante
pid = (pj % 4) * 5 + (pi_ % 5)
rr = np.random.default_rng(100)
gp = np.array([hexc(q) for q in ('#4d5853', '#56615b', '#46514c', '#5c6660', '#505a52', '#4a554f', '#535a50', '#5a635f')])
pc = gp[rr.integers(0, len(gp), 20)] * (0.9 + 0.18 * rr.random(20))[:, None]
A = pc[pid] * (1 + 0.035 * pnoise(2, 101) + 0.03 * pnoise(12, 102))[..., None]
yl = (y - pj * ph) / ph
A = A * (0.92 + 0.12 * yl)[..., None]                    # leichter Himmelsreflex zur Scheibenoberkante
A = np.where(lead[..., None], hexc('#202422'), A)
A = np.where(bar[..., None], hexc('#1a1d1c'), A)
save_rgb('bleiglas', A)
save_normal('bleiglas', np.where(lead, 0.5, 0.0) + np.where(bar, 0.9, 0.0) + 0.015 * pnoise(1.5, 103), 3.0)
# ------------------------------------------------------------------------------------------- Stahl, Lamellen
A = hexc('#3d4340')[None, None] * (1 + 0.06 * pnoise(3, 110) + 0.04 * pnoise(40, 111))[..., None]
save_rgb('stahl', A)
st = aniso_noise(30, 2, 120)
A = hexc('#4f463e')[None, None] * (1 + 0.12 * st + 0.05 * pnoise(20, 121))[..., None]
save_rgb('lamellen', A)
# ------------------------------------------------------------------------------------------- Zifferblatt (fast schwarz, Gold)
S = 1024
im = Image.new('RGB', (S, S), (14, 14, 13))
dr = ImageDraw.Draw(im)
c0 = S / 2
gold = (206, 164, 78)
dr.ellipse([6, 6, S - 6, S - 6], fill=(16, 16, 15))
dr.ellipse([10, 10, S - 10, S - 10], outline=gold, width=8)
dr.ellipse([96, 96, S - 96, S - 96], outline=gold, width=4)
for k in range(60):
    a = math.radians(90 - k * 6)
    r0, r1 = (440, 488) if k % 5 else (420, 492)
    dr.line([(c0 + r0 * math.cos(a), c0 - r0 * math.sin(a)), (c0 + r1 * math.cos(a), c0 - r1 * math.sin(a))], fill=gold, width=12 if k % 5 == 0 else 5)
dr.ellipse([c0 - 26, c0 - 26, c0 + 26, c0 + 26], fill=gold)


def roman_glyph(ch, w, h):
    if ch == 'I': return [((0, 0), (0, h))], w * 0.3
    if ch == 'V': return [((-w / 2, h), (0, 0)), ((0, 0), (w / 2, h))], w
    if ch == 'X': return [((-w / 2, h), (w / 2, 0)), ((-w / 2, 0), (w / 2, h))], w


def rot_pt(px, py, lx, ly, rot):
    return (px + lx * math.cos(rot) - ly * math.sin(rot), py + lx * math.sin(rot) + ly * math.cos(rot))


ROM = ['XII', 'I', 'II', 'III', 'IIII', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI']
for k, s in enumerate(ROM):
    a = math.radians(90 - k * 30)
    rc = 335
    px, py = c0 + rc * math.cos(a), c0 - rc * math.sin(a)
    gw_, gh = 40, 96
    total = sum(roman_glyph(ch, gw_, gh)[1] for ch in s) + 12 * (len(s) - 1)
    rot = -math.radians(k * 30)
    xo = -total / 2
    for ch in s:
        segs, adv = roman_glyph(ch, gw_, gh)
        cxl = xo + (adv / 2 if ch != 'I' else 0)
        for (p0, p1) in segs:
            dr.line([rot_pt(px, py, cxl + u, v - gh / 2, rot) for (u, v) in (p0, p1)], fill=gold, width=14)
        hw_ = adv / 2 + 4 if ch != 'I' else 9
        for ly in (-gh / 2 - 3, gh / 2 + 3):          # Serifen oben/unten
            dr.line([rot_pt(px, py, cxl - hw_, ly, rot), rot_pt(px, py, cxl + hw_, ly, rot)], fill=gold, width=7)
        xo += adv + 12
im.save(os.path.join(OUT, 'zifferblatt.png'))
# ------------------------------------------------------------------------------------------- Fliesen (Vorhalle Westportal, Foto 16)
FT = 1.2; NF = 10                        # 0.12-m-Quadratfliesen, Kreuzfugen
m = FT / N
x = (xx + 0.5) * m; y = (yy + 0.5) * m
fi = np.floor(x / (FT / NF)).astype(int); fj = np.floor(y / (FT / NF)).astype(int)
fx = x - fi * (FT / NF); fy = y - fj * (FT / NF)
dj = np.minimum(np.minimum(fx, FT / NF - fx), np.minimum(fy, FT / NF - fy))
rr = np.random.default_rng(130)
fp = np.array([hexc(q) for q in ('#7a2a1e', '#83301f', '#722719', '#8b3523', '#6c2419', '#7f2d22')])
fc = fp[rr.integers(0, len(fp), NF * NF)] * (0.88 + 0.22 * rr.random(NF * NF))[:, None]
A = fc[(fj % NF) * NF + (fi % NF)]
gl_ = np.clip(1 - np.hypot(fx - 0.035, fy - 0.085) / 0.05, 0, 1) ** 2          # Glanzfleck je Fliese
A = A * (1 + 0.05 * pnoise(1.5, 131))[..., None] + 0.12 * gl_[..., None]
A = np.where((dj < 0.003)[..., None], hexc('#8a8478'), A)
save_rgb('fliesen', A)
save_normal('fliesen', np.where(dj < 0.003, 0.0, 0.6 + 0.4 * np.clip(dj / 0.006, 0, 1)) + 0.01 * pnoise(1.0, 132), 5.0)
# ------------------------------------------------------------------------------------------- Tuerbronze (dunkel, fleckig)
n1 = pnoise(2.0, 140); n2 = pnoise(20, 141); st = aniso_noise(4, 30, 142)
A = hexc('#2b2620')[None, None] * (1 + 0.12 * n1 + 0.15 * n2 + 0.1 * st)[..., None]
A = np.where((pnoise(8, 143) > 1.4)[..., None], A * 0.7 + hexc('#3c4a40') * 0.3, A)
save_rgb('tuerbronze', A)
save_normal('tuerbronze', 0.06 * n1 + 0.04 * n2, 2.0)
# ------------------------------------------------------------------------------------------- Schaukasten + Hausnummernschild (Atlas)
from PIL import ImageFont


def font(sz, bold=False):
    for fn in (('arialbd.ttf' if bold else 'arial.ttf'), 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(os.path.join('C:/Windows/Fonts', fn), sz)
        except Exception:
            continue
    return ImageFont.load_default()


im = Image.new('RGB', (1024, 1024), (240, 240, 236))
dr = ImageDraw.Draw(im)
y1 = int(1024 * 0.7)                      # v 0.3..1.0 -> Bildzeilen 0..716
dr.rectangle([0, 0, 1023, y1], fill=(28, 34, 30))
dr.rectangle([0, 0, 1023, 96], fill=(18, 22, 20))
dr.text((40, 18), 'Ev.-luth. Marktkirche', font=font(58, True), fill=(236, 236, 230))
dr.text((760, 34), 'Hannover', font=font(36), fill=(200, 200, 190))
rr = np.random.default_rng(150)
cols = [(196, 40, 40), (232, 200, 72), (60, 110, 170), (235, 235, 228), (120, 160, 90), (226, 140, 60)]
x_ = 30
for i, wd in enumerate((210, 170, 230, 160, 160)):
    top = 125 + int(rr.integers(0, 30)); bot = y1 - 30 - int(rr.integers(0, 60))
    c = cols[i % len(cols)]
    dr.rectangle([x_, top, x_ + wd, bot], fill=(244, 243, 236))
    dr.rectangle([x_ + 10, top + 12, x_ + wd - 10, top + 12 + int((bot - top) * 0.32)], fill=c)
    dr.text((x_ + 14, top + 22 + int((bot - top) * 0.32)), ['Gottesdienst', 'Konzert', 'Orgel', 'Turm', 'Termine'][i], font=font(22, True), fill=(30, 30, 30))
    for ln in range(int((bot - top) * 0.5) // 18):
        yy_ = top + 60 + int((bot - top) * 0.32) + ln * 18
        if yy_ > bot - 14: break
        dr.line([(x_ + 14, yy_), (x_ + 14 + int((wd - 30) * (0.5 + 0.5 * rr.random())), yy_)], fill=(110, 110, 110), width=5)
    x_ += wd + 22
# Hausnummernschild (u 0..0.4, v 0..0.25 -> Bildzeilen 768..1023, Spalten 0..409)
dr.rectangle([0, 768, 409, 1023], fill=(242, 242, 238))
dr.rectangle([10, 778, 399, 1013], outline=(24, 48, 110), width=14)
dr.text((205, 895), '11', font=font(170, True), fill=(24, 48, 110), anchor='mm')
im.save(os.path.join(OUT, 'schaukasten.png'))
# ------------------------------------------------------------------------------------------- Texturbudget (QA)
# kleine Flaechen / Normalen ohne Fernwirkung -> 512 px (exakt halbiert: kachelbar), fast flache Normalen entfallen
HALF = ['bronze.jpg', 'kupfer.jpg', 'kupfer_patina.jpg', 'zifferblatt.png', 'lamellen.jpg', 'stahl.jpg', 'bleiglas.jpg', 'bleiglas_n.png', 'tuerbronze.jpg', 'tuerbronze_n.png',
        'sockelstein.jpg', 'sockelstein_n.png', 'granit_n.png', 'fliesen_n.png', 'glasurstein_n.png', 'sandstein_quader_n.png',
        'sandstein_dunkel_n.png', 'sandstein_n.png']
for fn in HALF:
    pth = os.path.join(OUT, fn); im = Image.open(pth)
    im = im.resize((512, 512), Image.LANCZOS)
    if fn.endswith('.jpg'): im.save(pth, quality=92)
    else: im.save(pth, optimize=True)
for fn in ('bronze_n.png', 'kupfer_n.png', 'kupfer_patina_n.png'):
    try: os.remove(os.path.join(OUT, fn))
    except FileNotFoundError: pass
print('textures ->', OUT, sorted(os.listdir(OUT)))

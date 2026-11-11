"""Prozedurale, kachelbare PBR-Texturen (1024 px) fuer das Ballhof-Ensemble -> <out>/<name>_col/_nrm/_orm.png
ORM: R = AO, G = Rauheit, B = Metall.  Aufruf: python bh_tex.py <out_dir> [name ...]"""
import os, sys, math
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw

N = 1024
RNG = np.random.default_rng(7)


def fbm(n=N, sig=(64, 24, 8, 3, 1.2), amp=(1, 0.6, 0.35, 0.2, 0.1), seed=0, aniso=(1.0, 1.0)):
    r = np.random.default_rng(seed)
    out = np.zeros((n, n))
    for s, a in zip(sig, amp):
        w = r.standard_normal((n, n))
        g = ndi.gaussian_filter(w, (s * aniso[1], s * aniso[0]), mode='wrap')
        g /= g.std() + 1e-9
        out += a * g
    out -= out.min(); out /= out.max() + 1e-9
    return out


def normal_from_height(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    n = np.dstack([-dx, dy, np.ones_like(h)])            # glTF: +Y = oben im Bild (v nach oben)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def save(out, name, col, h=None, strength=4.0, rough=0.85, metal=0.0, ao=None, alpha=None):
    col = np.clip(col, 0, 1)
    if alpha is not None:
        img = np.dstack([col, np.clip(alpha, 0, 1)])
        Image.fromarray((img * 255).astype(np.uint8), 'RGBA').save(os.path.join(out, name + '_col.png'), optimize=True)
    else:
        Image.fromarray((col * 255).astype(np.uint8)).save(os.path.join(out, name + '_col.png'), optimize=True)
    if h is not None:
        Image.fromarray(normal_from_height(h, strength)).save(os.path.join(out, name + '_nrm.png'), optimize=True)
    R = np.broadcast_to(rough, col.shape[:2]) if np.ndim(rough) else np.full(col.shape[:2], rough)
    A = np.ones(col.shape[:2]) if ao is None else ao
    orm = np.dstack([A, R, np.full(col.shape[:2], metal)])
    Image.fromarray((np.clip(orm, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out, name + '_orm.png'), optimize=True)
    print('tex', name)


def srgb(*c):
    return np.array(c, float) / 255.0


# ----------------------------------------------------------------------------------------------- Putz (neutral, getoent)
def plaster(out):
    h = fbm(seed=1, sig=(40, 12, 4, 1.5, 0.7), amp=(0.4, 0.5, 0.5, 0.4, 0.3))
    sp = (RNG.random((N, N)) > 0.997).astype(float)
    sp = ndi.gaussian_filter(sp, 0.8, mode='wrap') * 3
    v = 0.90 + 0.08 * fbm(seed=2, sig=(120, 40, 10)) - 0.05 * sp
    col = np.dstack([v * 1.0, v * 0.99, v * 0.965])
    stain = fbm(seed=3, sig=(200, 80), amp=(1, 0.5))
    col *= (0.96 + 0.04 * stain)[..., None]
    save(out, 'plaster', col, h, strength=3.0, rough=0.92, ao=0.95 + 0.05 * h)


# ----------------------------------------------------------------------------------------------- Holz (neutral, Maserung entlang u)
def wood(out):
    g = fbm(seed=11, sig=(30, 10, 3, 1), amp=(1, 0.7, 0.5, 0.3), aniso=(14.0, 0.6))
    rings = np.sin(g * 40) * 0.5 + 0.5
    pores = fbm(seed=12, sig=(2, 0.8), amp=(1, 0.6), aniso=(10, 0.5))
    v = 0.62 + 0.25 * rings * 0.5 + 0.18 * g - 0.10 * (pores > 0.62)
    # Schwundrisse
    cr = (fbm(seed=13, sig=(6, 2), aniso=(40, 0.3)) > 0.82).astype(float)
    v -= 0.15 * ndi.gaussian_filter(cr, 0.6, mode='wrap')
    col = np.dstack([v, v * 0.97, v * 0.93])
    h = 0.6 * g + 0.25 * rings - 0.5 * cr
    save(out, 'wood', col, h, strength=5.0, rough=0.72 + 0.1 * pores)


# ----------------------------------------------------------------------------------------------- Bruchstein-/Quadermauerwerk (Ballhof, 4 m x 4 m)
def ashlar(out, name='ashlar', seed=21, tile_m=4.0, course=(0.20, 0.42), blen=(0.30, 1.0), base=(178, 166, 140), var=0.10):
    r = np.random.default_rng(seed)
    px = N / tile_m
    hs = []
    while sum(hs) < tile_m:
        hs.append(r.uniform(*course))
    hs = np.array(hs) * tile_m / sum(hs)
    col = np.zeros((N, N, 3)); h = np.zeros((N, N)); ao = np.ones((N, N))
    rough = fbm(seed=seed + 1, sig=(10, 4, 1.5, 0.7), amp=(0.6, 0.6, 0.5, 0.4))
    tex2 = fbm(seed=seed + 2, sig=(40, 15, 5), amp=(1, .6, .4))
    yy, xx = np.mgrid[0:N, 0:N]
    y0 = 0.0
    jt = 0.014 * px
    for hc in hs:
        r0, r1 = int(round(y0 * px)), int(round((y0 + hc) * px))
        x0 = r.uniform(0, tile_m)
        ls = []
        while sum(ls) < tile_m:
            ls.append(r.uniform(*blen) * (1.6 if hc > 0.33 else 1.0))
        ls = np.array(ls) * tile_m / sum(ls)
        xs = (x0 + np.r_[0, np.cumsum(ls)]) % tile_m
        for i in range(len(ls)):
            c0 = xs[i] * px; L = ls[i] * px
            rows = slice(r0, r1)
            rr = yy[rows] - r0; cc = (xx[rows] - c0) % N
            inside = cc < L
            # Abstand zur Fuge
            d = np.minimum.reduce([rr, (r1 - r0) - 1 - rr, cc, L - 1 - cc])
            d = np.where(inside, d, -1)
            tone = r.normal(0, var)
            warm = abs(r.normal(0, 0.03))
            bc = srgb(*base) * (1 + tone) + np.array([warm, warm * 0.7, warm * 0.2])
            m = d >= jt
            edge = np.clip((d - jt) / (0.05 * px), 0, 1)
            hh = np.where(m, 0.55 + 0.45 * edge ** 0.6 + 0.25 * (rough[rows] - 0.5) + r.normal(0, 0.05), 0)
            sub = col[rows]; hsub = h[rows]; asub = ao[rows]
            for k in range(3):
                sub[..., k] = np.where(m, bc[k] * (0.88 + 0.22 * tex2[rows]) * (0.93 + 0.07 * edge), sub[..., k])
            hsub[:] = np.where(m, hh, hsub)
            asub[:] = np.where(m, 0.85 + 0.15 * edge, asub)
        y0 += hc
    joint = col.sum(2) == 0
    mort = srgb(196, 190, 176) * (0.9 + 0.12 * fbm(seed=seed + 4, sig=(3, 1)))[..., None]
    col = np.where(joint[..., None], mort, col)
    ao = np.where(joint, 0.6, ao)
    # Verwitterung: dunkle Schlieren unten, Flecken
    st = fbm(seed=seed + 5, sig=(90, 30, 8), amp=(1, .6, .3))
    col *= (0.86 + 0.18 * st)[..., None]
    h = ndi.gaussian_filter(h, 0.7, mode='wrap')
    save(out, name, col, h, strength=9.0, rough=0.93, ao=ao)


# ----------------------------------------------------------------------------------------------- Werkstein glatt (Gewaende, Sockel, Stufen)
def dressed(out):
    g = fbm(seed=31, sig=(30, 8, 2, 0.8), amp=(0.5, 0.5, 0.5, 0.4))
    lines = fbm(seed=32, sig=(1.2, 0.6), aniso=(30, 0.3))
    v = 0.92 + 0.08 * g - 0.04 * (lines > 0.6)
    base = srgb(204, 192, 166)
    col = base[None, None, :] * v[..., None]
    col *= (0.94 + 0.08 * fbm(seed=33, sig=(150, 50)))[..., None]
    save(out, 'dressed', col, 0.6 * g + 0.2 * lines, strength=2.5, rough=0.85)


# ----------------------------------------------------------------------------------------------- Hohlpfannen-Dach (5 Pfannen x 4 Reihen je Kachel)
def tiles(out, name='tiles', seed=41, base=(170, 82, 56), dark=(92, 52, 40), moss=0.0, nu=5, nv=4):
    r = np.random.default_rng(seed)
    col = np.zeros((N, N, 3)); h = np.zeros((N, N)); ao = np.ones((N, N))
    yy, xx = np.mgrid[0:N, 0:N]
    cw = N / nu; ch = N / nv
    # v = 0 unten (Bildzeile N-1), Reihen von unten nach oben; jede Reihe ueberdeckt die obere
    vimg = (N - 1 - yy) / ch                       # Reihenkoordinate (0..nv)
    row = np.floor(vimg).astype(int); fv = vimg - row          # 0 = Unterkante der Reihe, 1 = Oberkante
    shift = (row % 2) * 0.0
    uimg = xx / cw + shift
    col_i = np.floor(uimg).astype(int) % nu; fu = uimg - np.floor(uimg)
    # S-Profil: Mulde links, Wulst rechts
    prof = 0.5 - 0.5 * np.cos(2 * np.pi * fu)        # 0 am Rand, 1 in der Mitte
    wave = np.where(fu < 0.62, -np.sin(np.pi * fu / 0.62) * 0.7, np.sin(np.pi * (fu - 0.62) / 0.38) * 1.0)
    lip = np.clip(1 - fv / 0.12, 0, 1)               # Wulst an der Unterkante (Fusskante)
    shade = np.clip((fv - 0.72) / 0.28, 0, 1) ** 1.5  # Schatten unter der naechsten Reihe
    tone = r.normal(0, 1, (nv, nu))
    mott = fbm(seed=seed + 1, sig=(20, 6, 2), amp=(1, .6, .4))
    b = srgb(*base); d = srgb(*dark)
    t = np.clip(0.5 + 0.22 * tone[row % nv, col_i] + 0.25 * (mott - 0.5), 0, 1)
    sooty = (r.random((nv, nu)) < 0.18)[row % nv, col_i]
    c = b[None, None] * (0.75 + 0.5 * t[..., None])
    c = np.where(sooty[..., None], d[None, None] * (0.8 + 0.4 * t[..., None]), c)
    light = 0.90 + 0.13 * (wave * 0.5 + 0.5)
    c *= light[..., None]
    c *= (1 - 0.45 * shade)[..., None]
    if moss > 0:
        mm = fbm(seed=seed + 2, sig=(12, 4, 1.5), amp=(1, .6, .5))
        mz = np.clip((mm - (1 - moss)) * 6, 0, 1) * (wave < 0)
        c = c * (1 - 0.6 * mz[..., None]) + srgb(88, 92, 60)[None, None] * 0.6 * mz[..., None]
    col = c
    h = 0.55 * wave + 0.25 * lip - 0.35 * shade + 0.08 * (mott - 0.5)
    ao = 1 - 0.5 * shade - 0.15 * (wave < -0.4)
    save(out, name, col, h, strength=10.0, rough=0.78, ao=ao)


# ----------------------------------------------------------------------------------------------- Schiefer (Gaubenverkleidung, 4 x 6 Schuppen)
def slate(out):
    r = np.random.default_rng(51)
    yy, xx = np.mgrid[0:N, 0:N]
    nv, nu = 6, 4
    ch, cw = N / nv, N / nu
    row = (N - 1 - yy) // ch; fv = ((N - 1 - yy) % ch) / ch
    u = xx / cw + 0.5 * (row % 2)
    ci = np.floor(u).astype(int) % nu; fu = u - np.floor(u)
    tone = r.normal(0, 1, (nv, nu))[row.astype(int) % nv, ci]
    v = 0.25 + 0.03 * tone + 0.04 * fbm(seed=52, sig=(4, 1.5))
    edge = (fu < 0.03) | (fu > 0.97)
    shade = np.clip((fv - 0.8) / 0.2, 0, 1)
    v = v * (1 - 0.5 * shade) * np.where(edge, 0.6, 1)
    col = np.dstack([v * 0.95, v * 1.0, v * 1.08])
    h = -0.5 * shade - 0.4 * edge + 0.1 * fbm(seed=53, sig=(3, 1))
    save(out, 'slate', col, h, strength=6.0, rough=0.6)


# ----------------------------------------------------------------------------------------------- Stehfalz-Blech (Falz bei u = 0 und 0.5)
def seam(out):
    yy, xx = np.mgrid[0:N, 0:N]
    fu = (xx / (N / 2)) % 1.0
    rib = np.exp(-((fu - 0.5) / 0.012) ** 2) + np.exp(-((fu - 0.0) / 0.012) ** 2) + np.exp(-((fu - 1.0) / 0.012) ** 2)
    g = fbm(seed=61, sig=(60, 20, 5), amp=(1, .5, .3))
    v = 0.86 + 0.08 * g + 0.06 * rib
    col = np.dstack([v, v, v])
    h = 1.0 * rib + 0.05 * g
    save(out, 'seam', col, h, strength=12.0, rough=0.45, metal=0.35)


def zinc(out):
    g = fbm(seed=65, sig=(60, 20, 5, 1.5), amp=(1, .5, .3, .2))
    v = 0.80 + 0.12 * g
    col = np.dstack([v, v * 1.01, v * 1.03])
    save(out, 'zinc', col, 0.2 * g, strength=2.0, rough=0.45, metal=0.65)


# ----------------------------------------------------------------------------------------------- Bretter (Tueren), senkrecht, 7 je Kachel
def planks(out):
    yy, xx = np.mgrid[0:N, 0:N]
    nb = 7
    fu = (xx / (N / nb)) % 1.0
    bi = (xx // (N / nb)).astype(int)
    g = fbm(seed=71, sig=(30, 10, 3, 1), amp=(1, 0.7, 0.5, 0.3), aniso=(0.6, 14.0))
    tone = np.random.default_rng(72).normal(0, 0.05, nb)[bi]
    v = 0.70 + 0.2 * g + tone
    gap = (fu < 0.025) | (fu > 0.975)
    v = np.where(gap, 0.25, v)
    col = np.dstack([v, v * 0.96, v * 0.9])
    h = 0.5 * g - 1.0 * gap
    save(out, 'planks', col, h, strength=6.0, rough=0.7)


def gravel(out):
    r = np.random.default_rng(81)
    g = fbm(seed=82, sig=(3, 1.5, 0.7), amp=(1, .8, .5))
    pebble = (fbm(seed=83, sig=(2.5, 1.2)) > 0.55).astype(float)
    v = 0.42 + 0.15 * g + 0.08 * pebble
    col = np.dstack([v, v * 0.99, v * 0.96])
    save(out, 'gravel', col, g + pebble, strength=5.0, rough=0.95)


def roofing(out):
    """Bitumenbahn mit Stoessen (Flachdaecher hinten)"""
    yy, xx = np.mgrid[0:N, 0:N]
    seamv = ((yy % (N // 2)) < 4).astype(float)
    g = fbm(seed=85, sig=(40, 10, 2), amp=(1, .5, .3))
    v = 0.24 + 0.06 * g - 0.04 * seamv
    col = np.dstack([v, v, v * 1.02])
    save(out, 'roofing', col, 0.3 * g + 0.6 * seamv, strength=4.0, rough=0.85)


# ----------------------------------------------------------------------------------------------- Efeu / wilder Wein (RGBA, Alpha-Maske)
def ivy(out):
    r = np.random.default_rng(91)
    S = N * 2
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    hgt = Image.new('L', (S, S), 0)
    d = ImageDraw.Draw(img); dh = ImageDraw.Draw(hgt)
    greens = [(52, 92, 36), (66, 110, 40), (80, 124, 46), (44, 78, 32), (96, 132, 52), (60, 100, 30), (110, 140, 60)]
    for k in range(5200):
        x, y = r.uniform(0, S), r.uniform(0, S)
        s = r.uniform(18, 46)
        ang = r.uniform(0, 2 * math.pi)
        c = greens[r.integers(len(greens))]
        sh = r.uniform(0.75, 1.15)
        c = tuple(int(min(255, v * sh)) for v in c) + (255,)
        # dreilappiges Blatt aus drei Ellipsen
        for lob, (la, ls) in enumerate(((0, 1.0), (0.9, 0.75), (-0.9, 0.75))):
            a = ang + la
            cx, cy = x + math.cos(a) * s * 0.35 * ls, y + math.sin(a) * s * 0.35 * ls
            rx, ry = s * 0.42 * ls, s * 0.3 * ls
            pts = [(cx + math.cos(a) * rx * math.cos(t) - math.sin(a) * ry * math.sin(t),
                    cy + math.sin(a) * rx * math.cos(t) + math.cos(a) * ry * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12)]
            for ox in (-S, 0, S):
                for oy in (-S, 0, S):
                    if -60 < x + ox < S + 60 and -60 < y + oy < S + 60:
                        d.polygon([(p[0] + ox, p[1] + oy) for p in pts], fill=c)
                        dh.polygon([(p[0] + ox, p[1] + oy) for p in pts], fill=int(80 + 170 * (k / 5200)))
    img = img.resize((N, N), Image.LANCZOS)
    a = np.asarray(img, float) / 255
    hh = np.asarray(hgt.resize((N, N), Image.LANCZOS), float) / 255
    col = a[..., :3]
    alpha = (a[..., 3] > 0.45).astype(float)
    save(out, 'ivy', col, hh, strength=6.0, rough=0.7, alpha=alpha)


ALL = dict(plaster=plaster, wood=wood, ashlar=ashlar, dressed=dressed, tiles=tiles, slate=slate, seam=seam, zinc=zinc,
           planks=planks, gravel=gravel, roofing=roofing, ivy=ivy,
           ashlar_light=lambda o: ashlar(o, 'ashlar_light', seed=24, tile_m=3.0, course=(0.18, 0.36), blen=(0.3, 0.8), base=(196, 186, 160), var=0.07),
           tiles_old=lambda o: tiles(o, 'tiles_old', seed=44, base=(150, 84, 62), dark=(80, 56, 46), moss=0.25),
           tiles_brown=lambda o: tiles(o, 'tiles_brown', seed=47, base=(118, 84, 70), dark=(70, 60, 56), moss=0.2))

if __name__ == '__main__':
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    names = sys.argv[2:] or list(ALL)
    for n in names:
        ALL[n](out)

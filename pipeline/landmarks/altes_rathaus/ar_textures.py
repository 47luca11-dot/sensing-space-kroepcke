"""Prozedurale PBR-Texturen (1024 px, kachelbar) fuer das Alte Rathaus -> <out>/<name>_col.png, _nrm.png, _orm.png
Kachelgroessen (Meter) stehen in TILES (vom Bauskript fuer die UV-Skalierung genutzt).
Aufruf: python ar_textures.py <out_dir>"""
import sys, os, math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

R = 1024
TILES = {
    'brick': (2.20, 2.00), 'brick_black': (2.20, 2.00), 'brick_band': (2.20, 2.00), 'brick19': (2.096, 1.54),
    'sandstone': (1.6, 1.6), 'ashlar': (3.2, 2.52), 'roof': (2.0, 2.1), 'slate': (1.2, 1.2), 'granite': (2.4, 2.4),
    'frieze': (2.4, 0.55), 'quatrefoil': (1.6, 0.40), 'rosette': (1.2, 0.30), 'tracery': (1.0, 1.0), 'plaster': (2.0, 2.0),
    'pantile_ridge': (1.0, 1.0), 'tracery_red': (1.0, 1.0), 'tracery_brick': (1.0, 1.0), 'leaf_tiles': (1.0, 1.0),
    'wechsel': (0.44, 0.44), 'brick_gable': (2.20, 2.00),
}


def rng(seed):
    return np.random.default_rng(seed)


def blur(a, r, mode='wrap'):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(a, np.float32), r, mode=mode)


def tile_noise(shape, scale, seed, octaves=4):
    """kachelbares Wertrauschen (Summe von Oktaven, periodisch)"""
    h, w = shape
    out = np.zeros(shape, np.float32); amp = 1.0; tot = 0
    g = rng(seed)
    for o in range(octaves):
        nx = max(2, int(w / scale * 2 ** o)); ny = max(2, int(h / scale * 2 ** o))
        grid = g.random((ny, nx)).astype(np.float32)
        # periodische bilineare Interpolation
        ys = np.linspace(0, ny, h, endpoint=False); xs = np.linspace(0, nx, w, endpoint=False)
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
        a = grid[y0 % ny][:, x0 % nx]; b = grid[y0 % ny][:, (x0 + 1) % nx]
        c = grid[(y0 + 1) % ny][:, x0 % nx]; d = grid[(y0 + 1) % ny][:, (x0 + 1) % nx]
        out += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
        tot += amp; amp *= 0.5
    return out / tot


def normal_from_height(h, px_m, depth_m):
    """h 0..1 Hoehe, px_m Meter je Pixel, depth_m Hoehenumfang -> Normalenkarte (OpenGL, +Y oben)"""
    hz = h * depth_m
    dx = (np.roll(hz, -1, 1) - np.roll(hz, 1, 1)) / (2 * px_m)
    dy = (np.roll(hz, 1, 0) - np.roll(hz, -1, 0)) / (2 * px_m)      # Bild-oben = +Y
    n = np.stack([-dx, -dy, np.ones_like(hz)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def save(out, name, col, h, rough, metal=None, ao=None, px_m=0.002, depth=0.01):
    col = np.clip(col, 0, 1)
    Image.fromarray((col * 255).astype(np.uint8)).save(os.path.join(out, name + '_col.png'), optimize=True)
    Image.fromarray(normal_from_height(h, px_m, depth)).save(os.path.join(out, name + '_nrm.png'), optimize=True)
    if ao is None:
        ao = np.clip(0.55 + 0.45 * (h - blur(h, 6) + 0.5) * 1.0, 0, 1)
    orm = np.stack([np.clip(ao, 0, 1), np.clip(rough, 0, 1), np.zeros_like(rough) if metal is None else metal], -1)
    Image.fromarray((orm * 255).astype(np.uint8)).save(os.path.join(out, name + '_orm.png'), optimize=True)
    print('tex', name)


def hex2(c):
    c = c.lstrip('#'); return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


# ------------------------------------------------------------------------------------------- Ziegel
def bricks(out, name, tile, course, pattern, colfn, joint=0.013, seed=1, mortar='#b8ae9c', glaze_rough=0.70, depth=0.006):
    """pattern(k) -> Liste (laenge) der Steine einer Schicht, Versatz offs(k); colfn(g, kind, k, i) -> (rgb, glazed)"""
    tw, th = tile; H = W = R
    pxm_x = tw / W; pxm_y = th / H
    col = np.zeros((H, W, 3), np.float32); hgt = np.zeros((H, W), np.float32); rough = np.full((H, W), 0.9, np.float32)
    g = rng(seed)
    ncourse = int(round(th / course))
    yy = (np.arange(H) + 0.5) * pxm_y; xx = (np.arange(W) + 0.5) * pxm_x
    mort = hex2(mortar)
    col[:] = mort * (0.92 + 0.12 * tile_noise((H, W), 60, seed + 7)[..., None])
    rough[:] = 0.95
    noise_f = tile_noise((H, W), 9, seed + 3, 3)
    noise_c = tile_noise((H, W), 120, seed + 5, 3)
    for k in range(ncourse):
        y0 = k * course; y1 = y0 + course - joint
        r0 = int(round(y0 / pxm_y)); r1 = int(round(y1 / pxm_y))
        lens, offs = pattern(k)
        x = -offs
        i = 0
        while x < tw + 0.5:
            L = lens[i % len(lens)]
            kind = 'H' if L < 0.2 else 'S'
            rgb, glazed = colfn(g, kind, k, i)
            xa, xb = x, x + L - joint
            for (a, b) in ((xa, xb), (xa - tw, xb - tw), (xa + tw, xb + tw)):
                c0 = max(0, int(round(a / pxm_x))); c1 = min(W, int(round(b / pxm_x)))
                if c1 <= c0: continue
                sub_y = slice(r0, r1); sub_x = slice(c0, c1)
                hh, ww = r1 - r0, c1 - c0
                # gerundete Kanten: Abstand zum Rand
                ey = np.minimum(np.arange(hh) + 0.5, hh - np.arange(hh) - 0.5)[:, None]
                ex = np.minimum(np.arange(ww) + 0.5, ww - np.arange(ww) - 0.5)[None, :]
                e = np.minimum(ey * pxm_y, ex * pxm_x)
                bev = np.clip(e / 0.004, 0, 1) ** 0.5
                vary = 0.9 + 0.2 * noise_f[sub_y, sub_x]
                shade = (0.975 + 0.05 * noise_c[sub_y, sub_x])[..., None]     # schwach: Kachel 2.2 m wuerde sonst sichtbar wiederholen
                col[sub_y, sub_x] = rgb[None, None, :] * shade * (vary[..., None] if not glazed else (0.95 + 0.1 * vary[..., None]))
                hgt[sub_y, sub_x] = 0.55 + 0.45 * bev * (0.85 + 0.3 * noise_f[sub_y, sub_x])
                rough[sub_y, sub_x] = glaze_rough + 0.15 * noise_f[sub_y, sub_x] if glazed else 0.82 + 0.12 * noise_f[sub_y, sub_x]
            x += L; i += 1
    hgt = blur(hgt, 0.7)
    save(out, name, col, hgt, rough, px_m=pxm_x, depth=depth)


def gothic(k):
    # gotischer Verband: Laeufer 0.293 / Binder 0.147, Schichtversatz
    return [0.293, 0.147], (k % 2) * 0.22 + (k % 4 == 3) * 0.05


def brick_red(out):
    # warmes Orange-Rot der Fotos (Marktseite ~ #a5553c, Sonne ~ #b5603a); kaum Glasurkoepfe in den Flaechen
    reds = [hex2('#a5553c'), hex2('#9c4d36'), hex2('#ad5c40'), hex2('#94482f'), hex2('#b2633f'), hex2('#a35a42'), hex2('#9a5238')]
    darks = [hex2('#76402c'), hex2('#6a3a2a')]
    glz = [hex2('#2e2a24'), hex2('#33302a')]

    def cf(g, kind, k, i):
        r = g.random()
        if kind == 'H' and r < 0.012: return glz[g.integers(len(glz))] * (0.9 + 0.2 * g.random()), True
        if r < 0.05: return darks[g.integers(len(darks))] * (0.9 + 0.2 * g.random()), False
        c = reds[g.integers(len(reds))] * (0.92 + 0.14 * g.random())
        return c, False
    bricks(out, 'brick', TILES['brick'], 0.10, gothic, cf, seed=11, mortar='#bdb3a2')


def brick_black(out):
    glz = [hex2('#2a2522'), hex2('#302b25'), hex2('#272b23'), hex2('#33281f'), hex2('#24231f')]

    def cf(g, kind, k, i):
        return glz[g.integers(len(glz))] * (0.85 + 0.3 * g.random()), True
    bricks(out, 'brick_black', TILES['brick_black'], 0.10, gothic, cf, seed=12, mortar='#9d968a')


def brick_band(out):
    """Fialen-/Lisenenschaefte (Foto 01): Schichtwechsel rot / schwarz glasiert (je zwei rote, eine glasierte Schicht)"""
    reds = [hex2('#a8553a'), hex2('#b25f40'), hex2('#9c4d34')]
    glz = [hex2('#2c2a25'), hex2('#2a3129'), hex2('#3a2e27')]

    def cf(g, kind, k, i):
        if k % 3 == 2: return glz[g.integers(len(glz))] * (0.9 + 0.2 * g.random()), True
        return reds[g.integers(len(reds))] * (0.88 + 0.22 * g.random()), False
    bricks(out, 'brick_band', TILES['brick_band'], 0.10, gothic, cf, seed=13)


def brick_gable(out):
    """Giebelflaechen (Fotos 01/07): dunkleres Rot, gotischer Verband mit vielen schwarz glasierten Koepfen (Rautenwirkung)"""
    reds = [hex2('#9a4a33'), hex2('#8f442f'), hex2('#a5553a'), hex2('#874130'), hex2('#9f5038')]
    glz = [hex2('#2e2a25'), hex2('#2b302a'), hex2('#38302a')]

    def cf(g, kind, k, i):
        r = g.random()
        if kind == 'H' and ((i + k) % 4 == 0 or r < 0.15): return glz[g.integers(len(glz))] * (0.9 + 0.2 * g.random()), True
        if r < 0.06: return hex2('#6a3a2a') * (0.9 + 0.2 * g.random()), False
        return reds[g.integers(len(reds))] * (0.9 + 0.16 * g.random()), False
    bricks(out, 'brick_gable', TILES['brick_gable'], 0.10, gothic, cf, seed=15, mortar='#b0a796')


def wechsel(out):
    """Keilsteine/Gewaende im Wechsel rot / schwarz glasiert (Fotos 01, 03): Kachel 0.44 x 0.44 m,
    u = entlang des Bogens (4 Steine a 0.11 m), v = quer zum Bogen (Steinlaenge 0.29 + 0.15)."""
    tw, th = TILES['wechsel']; W = H = R; pxu = tw / W; pxv = th / H
    g = rng(17)
    col = np.zeros((H, W, 3), np.float32); hgt = np.zeros((H, W), np.float32); rough = np.zeros((H, W), np.float32)
    mort = hex2('#b3a998')
    nf = tile_noise((H, W), 9, 171, 3); nc = tile_noise((H, W), 80, 172, 3)
    col[:] = mort * (0.9 + 0.12 * nf[..., None]); rough[:] = 0.95; hgt[:] = 0.0
    reds = [hex2('#a5553c'), hex2('#9c4d36'), hex2('#ad5c40'), hex2('#b2633f')]
    glz = [hex2('#28251f'), hex2('#252a20'), hex2('#2d2722')]
    jt = 0.011
    for i in range(4):
        u0, u1 = i * 0.11, i * 0.11 + 0.11 - jt
        c0, c1 = int(round(u0 / pxu)), int(round(u1 / pxu))
        for (v0, v1) in ((0.0, 0.29 - jt), (0.29, 0.44 - jt)):
            r0, r1 = int(round(v0 / pxv)), int(round(v1 / pxv))
            glazed = (i % 2 == 1)
            base = (glz if glazed else reds)[g.integers(3)] * (0.9 + 0.18 * g.random())
            hh, ww = r1 - r0, c1 - c0
            ey = np.minimum(np.arange(hh) + 0.5, hh - np.arange(hh) - 0.5)[:, None] * pxv
            ex = np.minimum(np.arange(ww) + 0.5, ww - np.arange(ww) - 0.5)[None, :] * pxu
            bev = np.clip(np.minimum(ey, ex) / 0.006, 0, 1) ** 0.5
            sub = (slice(r0, r1), slice(c0, c1))
            vary = 0.9 + 0.2 * nf[sub]
            if glazed:   # Glasur: glatt, leichte Wolken, Glanz
                col[sub] = base * (0.92 + 0.12 * nc[sub][..., None]) * (0.96 + 0.06 * vary[..., None])
                rough[sub] = 0.66 + 0.12 * nf[sub]
            else:
                col[sub] = base * vary[..., None]
                rough[sub] = 0.82 + 0.1 * nf[sub]
            hgt[sub] = 0.5 + 0.5 * bev * (0.9 + 0.2 * nf[sub])
    # Spalten: u-Kanten (Kopffugen) wrap-sicher, da 4 Steine genau die Kachel fuellen
    save(out, 'wechsel', col, blur(hgt, 0.7), rough, px_m=pxu, depth=0.008)


def brick_19(out):
    reds = [hex2('#8f4a37'), hex2('#86412f'), hex2('#965240'), hex2('#7e3f30')]

    def cf(g, kind, k, i):
        return reds[g.integers(len(reds))] * (0.92 + 0.14 * g.random()), False
    bricks(out, 'brick19', TILES['brick19'], 0.077, lambda k: ([0.262], (k % 2) * 0.131), cf, joint=0.011, seed=14, mortar='#a59d8f')


# ------------------------------------------------------------------------------------------- Stein
def sandstone(out):
    tw, th = TILES['sandstone']; pxm = tw / R
    n1 = tile_noise((R, R), 200, 21, 5); n2 = tile_noise((R, R), 12, 22, 3); n3 = tile_noise((R, R), 40, 23, 4)
    base = hex2('#8b8070'); dark = hex2('#5f584d'); light = hex2('#a99e8a')
    t = (n1 * 0.6 + n3 * 0.4)[..., None]
    col = base * (0.85 + 0.3 * n2[..., None]) * (1 - 0.25 * np.clip(t - 0.5, 0, 1)) + (light - base) * np.clip(0.5 - t, 0, 1)
    col = col * 0.92 + dark * 0.08 * (1 - n3[..., None])
    h = 0.5 + 0.3 * n2 + 0.2 * n3
    # Lagerfugen alle 0.4 m (fein)
    yy = (np.arange(R) + 0.5) * (th / R)
    j = np.abs(((yy + 0.2) % 0.4) - 0.2) < 0.004
    h[j] -= 0.4; col[j] *= 0.75
    save(out, 'sandstone', col, h, 0.85 + 0.1 * n2, px_m=pxm, depth=0.004)


def ashlar(out):
    tw, th = TILES['ashlar']; W = H = R; pxx = tw / W; pxy = th / H
    g = rng(31)
    n2 = tile_noise((H, W), 10, 32, 3); n3 = tile_noise((H, W), 80, 33, 4); n4 = tile_noise((H, W), 300, 34, 3)
    base = hex2('#b7ab95'); col = np.zeros((H, W, 3), np.float32); h = np.ones((H, W), np.float32) * 0.6
    course = th / 6
    yy = (np.arange(H) + 0.5) * pxy; xx = (np.arange(W) + 0.5) * pxx
    for k in range(6):
        r0 = int(k * course / pxy); r1 = int((k + 1) * course / pxy)
        x = -g.random() * 0.6
        while x < tw:
            L = 0.7 + g.random() * 0.7
            if x + L > tw - 0.3 and x + L < tw: L = tw - x
            c0 = int(max(0, x) / pxx); c1 = int(min(tw, x + L) / pxx)
            tint = base * (0.88 + 0.18 * g.random()) * np.array([1, 0.99 + 0.02 * g.random(), 0.97 + 0.05 * g.random()])
            col[r0:r1, c0:c1] = tint
            h[r1 - 2:r1, c0:c1] = 0.1; h[r0:r1, c0:c0 + 2] = 0.1
            x += L
    # Verwitterung: dunkle Streifen von oben
    streak = blur((g.random((H, W)) > 0.995).astype(np.float32), 1.5)
    streak = np.clip(np.cumsum(streak, 0) / 30, 0, 1) * 0
    col *= (0.86 + 0.2 * n2[..., None]) * (0.9 + 0.15 * n3[..., None]) * (0.92 + 0.12 * n4[..., None])
    col[h < 0.2] *= 0.72
    h = h + 0.25 * n2
    save(out, 'ashlar', col, blur(h, 0.8), 0.82 + 0.12 * n2, px_m=pxx, depth=0.006)


def granite(out):
    tw, th = TILES['granite']; pxm = tw / R
    n1 = tile_noise((R, R), 4, 41, 2); n2 = tile_noise((R, R), 60, 42, 3)
    g = rng(43)
    speck = (g.random((R, R)) > 0.82).astype(np.float32)
    col = hex2('#8d8b86') * (0.85 + 0.25 * n1[..., None]) * (0.95 + 0.1 * n2[..., None])
    col = col * (1 - 0.35 * speck[..., None]) + hex2('#c9c6bf') * 0.1 * (g.random((R, R, 1)) > 0.9)
    h = 0.6 + 0.2 * n1
    yy = (np.arange(R) + 0.5) * pxm
    jy = np.abs(((yy + 0.3) % 0.6) - 0.3) < 0.003; jx = np.abs(((yy + 0.4) % 0.8) - 0.4) < 0.003
    h[jy, :] = 0.0; h[:, jx] = 0.0; col[jy, :] *= 0.6; col[:, jx] *= 0.6
    save(out, 'granite', col, h, 0.7 + 0.15 * n1, px_m=pxm, depth=0.004)


def slate(out):
    tw, th = TILES['slate']; pxm = tw / R
    g = rng(51)
    col = np.zeros((R, R, 3), np.float32); h = np.zeros((R, R), np.float32)
    n1 = tile_noise((R, R), 8, 52, 3)
    course = 0.12; w = 0.2
    for k in range(int(th / course)):
        r0 = int(k * course / pxm); r1 = int((k + 1) * course / pxm)
        x = -(k % 2) * w / 2
        while x < tw:
            c0 = int(max(0, x) / pxm); c1 = int(min(tw, x + w) / pxm)
            t = 0.85 + 0.25 * g.random()
            col[r0:r1, c0:c1] = hex2('#4f555b') * t * np.array([1, 1, 1.02 + 0.04 * g.random()])
            ramp = np.linspace(1, 0.3, r1 - r0)[:, None]
            h[r0:r1, c0:c1] = ramp
            h[r0:r1, c0:c0 + 1] = 0.1
            x += w
    col *= (0.9 + 0.2 * n1[..., None])
    save(out, 'slate', col, h, 0.55 + 0.2 * n1, px_m=pxm, depth=0.008)


def plaster(out):
    pxm = TILES['plaster'][0] / R
    n1 = tile_noise((R, R), 6, 61, 3); n2 = tile_noise((R, R), 150, 62, 3)
    col = hex2('#d6cfc0') * (0.94 + 0.08 * n1[..., None]) * (0.95 + 0.07 * n2[..., None])
    save(out, 'plaster', col, 0.5 + 0.5 * n1, 0.9 + 0 * n1, px_m=pxm, depth=0.002)


# ------------------------------------------------------------------------------------------- Dach
def roof(out):
    tw, th = TILES['roof']; pxx = tw / R; pxy = th / R
    g = rng(71)
    course = 0.30; tile_w = 0.20
    yy = (np.arange(R) + 0.5) * pxy; xx = (np.arange(R) + 0.5) * pxx
    col = np.zeros((R, R, 3), np.float32); h = np.zeros((R, R), np.float32); rough = np.zeros((R, R), np.float32)
    cols = [hex2('#b35a3a'), hex2('#a9533a'), hex2('#bd6442'), hex2('#a04d35'), hex2('#b66040'), hex2('#9a4c36'), hex2('#c26a46')]
    n1 = tile_noise((R, R), 10, 72, 3); n2 = tile_noise((R, R), 90, 73, 4)
    nc = int(round(th / course)); ntx = int(round(tw / tile_w))
    for k in range(nc):
        # Bild oben = hoeher am Dach (v negativ nach oben); Schicht k von unten: Bildzeilen von unten
        rb = R - int(round(k * course / pxy)); rt = R - int(round((k + 1) * course / pxy))
        hh = rb - rt
        tv = np.linspace(0, 1, hh)[:, None]            # 0 oben .. 1 unten (Fuss der Pfanne)
        for i in range(ntx):
            c0 = int(round(i * tile_w / pxx)); c1 = int(round((i + 1) * tile_w / pxx))
            ww = c1 - c0
            tu = np.linspace(0, 1, ww)[None, :]
            wave = 0.5 + 0.5 * np.sin(tu * 2 * math.pi - math.pi / 2)       # S-Pfanne
            base = cols[g.integers(len(cols))] * (0.93 + 0.12 * g.random())
            sh = 0.75 + 0.25 * wave
            c = base[None, None, :] * sh[..., None] * (0.92 + 0.12 * tv[..., None])
            # Moos/Schmutz am Fuss
            if g.random() < 0.12:
                c = c * (1 - 0.10 * tv[..., None] ** 3)
            col[rt:rb, c0:c1] = c
            h[rt:rb, c0:c1] = 0.25 + 0.55 * wave + 0.2 * tv
            rough[rt:rb, c0:c1] = 0.72 + 0.1 * g.random()
            h[rt:rb, c0:c0 + 1] *= 0.6
        h[rb - 3:rb, :] *= 0.5; col[rb - 3:rb, :] *= 0.55     # Schattenkante der Ueberdeckung
    col *= (0.94 + 0.08 * n1[..., None]) * (0.98 + 0.04 * n2[..., None])
    rough = rough + 0.08 * n1
    save(out, 'roof', col, blur(h, 0.6), rough, px_m=pxx, depth=0.03)


def ridge(out):
    pxm = 1.0 / R
    n1 = tile_noise((R, R), 10, 81, 3)
    xx = (np.arange(R) + 0.5) / R
    seg = (xx * 3) % 1.0
    h = np.tile((0.6 + 0.4 * np.sin(np.clip(seg, 0, 1) * math.pi))[None, :], (R, 1))
    col = hex2('#a84f36') * (0.85 + 0.2 * n1[..., None]) * (0.8 + 0.2 * h[..., None])
    save(out, 'pantile_ridge', col, h, 0.75 + 0.1 * n1, px_m=pxm, depth=0.02)


# ------------------------------------------------------------------------------------------- Ornamentbaender
def _canvas(wpx, hpx, bg=0.0):
    im = Image.new('F', (wpx, hpx), bg); return im, ImageDraw.Draw(im)


def frieze(out):
    """Terrakotta-Fries: Rankenwerk (Wein) mit Rundmedaillons (Wappen/Bildnis), Modul 2.4 x 0.55 m"""
    tw, th = TILES['frieze']; W, H = 2048, 470
    pxm = tw / W
    im, d = _canvas(W, H, 0.15)
    g = rng(91)
    # Rahmenleisten oben/unten
    d.rectangle([0, 0, W, 26], fill=0.75); d.rectangle([0, H - 26, W, H], fill=0.75)
    d.rectangle([0, 26, W, 34], fill=0.45); d.rectangle([0, H - 34, W, H - 26], fill=0.45)
    # Trennpilaster an den Modulenden
    for x0 in (0, W - 40):
        d.rectangle([x0, 34, x0 + 40, H - 34], fill=0.7)
    cx, cy, rr = W / 2, H / 2, 175
    # Ranke: Sinuslinie mit Blaettern und Trauben
    for side in (-1, 1):
        pts = []
        for i in range(400):
            t = i / 399
            x = W / 2 + side * (rr + 15 + t * (W / 2 - rr - 70))
            y = H / 2 + 120 * math.sin(t * 2 * math.pi * 1.5 + (0 if side > 0 else math.pi))
            pts.append((x, y))
        d.line(pts, fill=0.85, width=18)
        for j in range(0, 400, 33):
            x, y = pts[j]
            for k in range(2):
                ang = g.random() * 2 * math.pi
                lx, ly = x + math.cos(ang) * 55, y + math.sin(ang) * 45
                lx = min(max(lx, 60), W - 60); ly = min(max(ly, 60), H - 60)
                # Weinblatt: drei Kreise
                for a2 in (-0.6, 0, 0.6):
                    d.ellipse([lx + math.cos(ang + a2) * 22 - 26, ly + math.sin(ang + a2) * 22 - 26, lx + math.cos(ang + a2) * 22 + 26, ly + math.sin(ang + a2) * 22 + 26], fill=0.72 + 0.1 * g.random())
            if j % 66 == 0:
                gx, gy = x + side * 30, y + (40 if y < H / 2 else -40)
                for q in range(9):
                    d.ellipse([gx + (q % 3) * 14 - 7 - 14, gy + (q // 3) * 14 - 7, gx + (q % 3) * 14 + 7 - 14, gy + (q // 3) * 14 + 7], fill=0.95)
    # Kleine Wappenmedaillons in der Ranke (Foto 13: Ring mit Schild), links Kleeblatt (Stadtzeichen), rechts Schraegbalken
    for sx_, motif in ((-1, 'klee'), (1, 'balken')):
        mx, my, mr = cx + sx_ * 560, cy + 4, 84
        d.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=0.86)
        d.ellipse([mx - mr + 15, my - mr + 15, mx + mr - 15, my + mr - 15], fill=0.30)
        sw_ = 46
        d.polygon([(mx - sw_, my - 50), (mx + sw_, my - 50), (mx + sw_, my + 8), (mx, my + 58), (mx - sw_, my + 8)], fill=0.66)
        if motif == 'klee':
            for (ox, oy) in ((0, -26), (-19, -6), (19, -6)):
                d.ellipse([mx + ox - 15, my + oy - 15, mx + ox + 15, my + oy + 15], fill=0.9)
            d.line([(mx, my), (mx + 6, my + 36)], fill=0.9, width=7)
        else:
            d.polygon([(mx - sw_, my - 20), (mx - sw_ + 26, my - 50), (mx + sw_, my + 22), (mx + sw_ - 18, my + 44)], fill=0.9)
    # Maskenkopf an der linken Ranke (Foto 13)
    hx, hy = 150, cy + 30
    d.ellipse([hx - 44, hy - 52, hx + 44, hy + 52], fill=0.8)
    for ex_ in (-17, 17):
        d.ellipse([hx + ex_ - 9, hy - 18, hx + ex_ + 9, hy - 6], fill=0.45)
    d.polygon([(hx - 6, hy - 6), (hx + 6, hy - 6), (hx, hy + 16)], fill=0.92)
    d.line([(hx - 18, hy + 28), (hx + 18, hy + 28)], fill=0.5, width=6)
    # Grosses Medaillon mit Bildnis (Foto 13: Bischof mit Mitra und Krummstab, Brustbild)
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=0.9)
    d.ellipse([cx - rr + 24, cy - rr + 24, cx + rr - 24, cy + rr - 24], fill=0.32)
    d.ellipse([cx - rr + 34, cy - rr + 34, cx + rr - 34, cy + rr - 34], fill=0.36)
    # Schultern/Gewand (Brustbild, unten vom Ring beschnitten) mit Faltenzuegen
    d.chord([cx - 118, cy + 10, cx + 118, cy + 250], 180, 360, fill=0.72)
    d.rectangle([cx - 118, cy + 128, cx + 118, cy + rr - 30], fill=0.72)
    for fx in (-70, -35, 0, 35, 70):
        d.line([(cx + fx, cy + 60), (cx + fx * 1.15, cy + rr - 34)], fill=0.6, width=5)
    d.polygon([(cx - 42, cy + 42), (cx + 42, cy + 42), (cx, cy + 105)], fill=0.85)      # Kragen/Pallium
    # Kopf, Haar, Mitra
    d.ellipse([cx - 46, cy - 58, cx + 46, cy + 52], fill=0.86)
    d.chord([cx - 58, cy - 40, cx + 58, cy + 70], 0, 180, fill=0.78)                     # Haar seitlich
    d.ellipse([cx - 40, cy - 46, cx + 40, cy + 46], fill=0.88)
    d.polygon([(cx - 44, cy - 50), (cx + 44, cy - 50), (cx + 30, cy - 120), (cx, cy - 150), (cx - 30, cy - 120)], fill=0.8)
    d.line([(cx, cy - 145), (cx, cy - 55)], fill=0.92, width=6); d.line([(cx - 40, cy - 62), (cx + 40, cy - 62)], fill=0.92, width=6)
    for ex_ in (-15, 15):
        d.ellipse([cx + ex_ - 7, cy - 14, cx + ex_ + 7, cy - 4], fill=0.6)
    d.line([(cx, cy - 6), (cx + 4, cy + 14)], fill=0.95, width=5)
    d.line([(cx - 12, cy + 26), (cx + 12, cy + 26)], fill=0.62, width=4)
    # Krummstab rechts (Schaft + Kruemme) und segnende Hand links
    d.line([(cx + 92, cy + 125), (cx + 92, cy - 70)], fill=0.9, width=10)
    d.arc([cx + 62, cy - 120, cx + 122, cy - 60], 180, 450, fill=0.9, width=10)
    d.ellipse([cx - 108, cy + 40, cx - 66, cy + 88], fill=0.84)
    h = np.asarray(im, np.float32)
    h = blur(h, 3.0) * 0.7 + blur(h, 1.0) * 0.3
    n = tile_noise((H, W), 30, 92, 3)
    # Foto 13: Relief grau-oliv (verwitterte Terrakotta), Grund rotbraun; Foto 03 aus Distanz insgesamt dunkles Band
    base = hex2('#5e5b49'); hi = hex2('#8e8869'); lo = hex2('#4a2d24')
    t = np.clip(h, 0, 1)[..., None]
    col = lo * (1 - t) + base * t
    col = col + (hi - base) * np.clip(t - 0.75, 0, 1) * 3
    col *= (0.88 + 0.2 * n[..., None])
    rough = 0.68 - 0.15 * np.clip(h - 0.6, 0, 1) + 0.1 * n
    imc = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).resize((R, 256), Image.LANCZOS)
    hr = np.asarray(Image.fromarray(h).resize((R, 256), Image.LANCZOS), np.float32)
    rr_ = np.asarray(Image.fromarray(rough.astype(np.float32)).resize((R, 256), Image.LANCZOS), np.float32)
    np.save(os.path.join(out, 'frieze_height.npy'), hr)
    col2 = np.asarray(imc, np.float32) / 255
    save(out, 'frieze', col2, hr, rr_, px_m=tw / R, depth=0.012)


def quatrefoil(out):
    tw, th = TILES['quatrefoil']; W, H = 1024, 256
    im, d = _canvas(W, H, 0.0)
    n = 4; cw = W / n
    for i in range(n):
        cx = cw * (i + 0.5); cy = H / 2; r = 52
        for dx, dy in ((r * 0.55, 0), (-r * 0.55, 0), (0, r * 0.55), (0, -r * 0.55)):
            d.ellipse([cx + dx - r * 0.62, cy + dy - r * 0.62, cx + dx + r * 0.62, cy + dy + r * 0.62], outline=1.0, width=18)
        d.rectangle([cw * i, 0, cw * i + 10, H], fill=1.0)
    d.rectangle([0, 0, W, 16], fill=1.0); d.rectangle([0, H - 16, W, H], fill=1.0)
    h = blur(np.asarray(im, np.float32), 1.5)
    nn = tile_noise((H, W), 12, 95, 3)
    col = hex2('#2a2522') * (1 - h[..., None]) + hex2('#a9492f') * h[..., None]
    col *= (0.9 + 0.15 * nn[..., None])
    save(out, 'quatrefoil', col, h, 0.8 - 0.4 * (1 - h) + 0.05 * nn, px_m=tw / W, depth=0.03)


def rosette(out):
    tw, th = TILES['rosette']; W, H = 1024, 256
    im, d = _canvas(W, H, 0.2)
    n = 4; cw = W / n
    for i in range(n):
        cx = cw * (i + 0.5); cy = H / 2
        d.rectangle([cw * i + 8, 14, cw * (i + 1) - 8, H - 14], fill=0.45)
        for k in range(6):
            a = k * math.pi / 3
            d.ellipse([cx + math.cos(a) * 40 - 30, cy + math.sin(a) * 40 - 30, cx + math.cos(a) * 40 + 30, cy + math.sin(a) * 40 + 30], fill=0.95)
        d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=0.75)
    h = blur(np.asarray(im, np.float32), 1.6)
    nn = tile_noise((H, W), 12, 96, 3)
    col = hex2('#7a3826') * (0.6 + 0.6 * h[..., None]) * (0.9 + 0.15 * nn[..., None])
    save(out, 'rosette', col, h, 0.8 + 0.05 * nn, px_m=tw / W, depth=0.02)


def tracery(out, name='tracery', bg='#2f3a2b', fg='#62805a', rough0=0.32):
    """gruen glasiertes Masswerkpaneel (UV 0..1): zwei Lanzetten mit Dreipass + Vierpass darueber
    (Varianten: tracery_red = weisses Masswerk auf rotem Grund (Zwerchhaus-/Giebeltafeln, Fotos 05/07),
     tracery_brick = Terrakotta-Masswerk vor dunklem Grund (Laubenbruestung, Foto 06))"""
    W, H = 512, 1024
    im, d = _canvas(W, H, 0.0)
    d.rectangle([0, 0, W, H], outline=1.0, width=34)
    for cx in (W * 0.3, W * 0.7):
        lw = W * 0.17
        d.rectangle([cx - lw, H * 0.45, cx + lw, H * 0.93], outline=1.0, width=20)
        d.pieslice([cx - lw, H * 0.45 - lw, cx + lw, H * 0.45 + lw], 180, 360, outline=1.0, width=20)
        for dx in (-lw * 0.5, lw * 0.5):
            d.ellipse([cx + dx - lw * 0.45, H * 0.44 - lw * 0.45, cx + dx + lw * 0.45, H * 0.44 + lw * 0.45], outline=1.0, width=10)
    cx, cy, r = W / 2, H * 0.2, W * 0.26
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1.0, width=22)
    for a in range(4):
        ang = a * math.pi / 2 + math.pi / 4
        d.ellipse([cx + math.cos(ang) * r * 0.42 - r * 0.42, cy + math.sin(ang) * r * 0.42 - r * 0.42,
                   cx + math.cos(ang) * r * 0.42 + r * 0.42, cy + math.sin(ang) * r * 0.42 + r * 0.42], outline=1.0, width=12)
    h = blur(np.asarray(im, np.float32), 2.0)
    im2 = Image.fromarray(h).resize((512, 1024))
    h = np.asarray(im2, np.float32)
    nn = tile_noise((1024, 512), 10, 97, 3)
    col = hex2(bg) * (1 - h[..., None]) + hex2(fg) * h[..., None]
    col *= (0.85 + 0.25 * nn[..., None])
    col = np.asarray(Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).resize((R, R)), np.float32) / 255
    h = np.asarray(Image.fromarray(h).resize((R, R)), np.float32)
    rough = rough0 + 0.4 * (1 - h)
    save(out, name, col, h, rough, px_m=0.4 / R, depth=0.02)


def tracery_red(out):
    tracery(out, 'tracery_red', bg='#8e3426', fg='#d9cfbf', rough0=0.45)


def tracery_brick(out):
    tracery(out, 'tracery_brick', bg='#2a1f1b', fg='#a95a3c', rough0=0.55)


def leaf_tiles(out):
    """Terrakotta-Plattenfeld der Laubenbruestung (Foto 06): 3 x 6 quadratische Platten mit Blattrelief, UV 0..1"""
    W, H = 512, 1024
    im, d = _canvas(W, H, 0.0)
    nc, nr = 3, 6
    cw, rh = W / nc, H / nr
    for r in range(nr):
        for c in range(nc):
            x0, y0 = c * cw, r * rh
            d.rectangle([x0 + 5, y0 + 5, x0 + cw - 5, y0 + rh - 5], fill=0.45)
            cx, cy = x0 + cw / 2, y0 + rh / 2
            for k in range(4):
                ang = k * math.pi / 2 + (math.pi / 4 if (r + c) % 2 else 0)
                for t, rr in ((0.18, 0.13), (0.30, 0.11), (0.40, 0.07)):
                    px, py = cx + math.cos(ang) * cw * t, cy + math.sin(ang) * rh * t * 0.9
                    d.ellipse([px - cw * rr, py - cw * rr, px + cw * rr, py + cw * rr], fill=1.0)
            d.ellipse([cx - cw * 0.08, cy - cw * 0.08, cx + cw * 0.08, cy + cw * 0.08], fill=0.8)
    h = blur(np.asarray(im, np.float32), 2.5)
    nn = tile_noise((H, W), 8, 131, 3)
    lo, mid, hi = hex2('#4a2a1c'), hex2('#a65a36'), hex2('#c47a4a')
    col = np.where(h[..., None] < 0.45, lo + (mid - lo) * (h[..., None] / 0.45), mid + (hi - mid) * ((h[..., None] - 0.45) / 0.55))
    col *= (0.88 + 0.2 * nn[..., None])
    col = np.asarray(Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).resize((R, R)), np.float32) / 255
    h = np.asarray(Image.fromarray(h).resize((R, R)), np.float32)
    save(out, 'leaf_tiles', col, h, 0.75 + 0.15 * (1 - h), px_m=0.4 / R, depth=0.03)


# ------------------------------------------------------------------------------------------- Schilder
def _font(font, size):
    try:
        return ImageFont.truetype(font, size)
    except Exception:
        return ImageFont.load_default()


def _fit_text(d, text, font, box_w, box_h, start):
    """groesste Schrift (ab start px), deren Text in box_w x box_h passt -> (font, bbox)"""
    size = int(start)
    while size > 8:
        f = _font(font, size)
        bb = d.textbbox((0, 0), text, font=f)
        if bb[2] - bb[0] <= box_w and bb[3] - bb[1] <= box_h:
            return f, bb
        size -= 2
    f = _font(font, 8)
    return f, d.textbbox((0, 0), text, font=f)


def sparkasse_logo(d, cx, cy, r, fg, bg):
    """Sparkassen-Logo (vereinfacht): kraeftiges S aus zwei Kreisbogen-Schleifen mit Punkt darueber, Farbe fg.
    cx, cy: Mitte des S-Koerpers, r: halbe S-Hoehe."""
    sw = max(2, int(r * 0.40))             # Strichstaerke
    ro = r * 0.54                          # Aussenradius je Schleife
    yt = cy - ro + sw * 0.5; yb = cy + ro - sw * 0.5
    # obere Schleife: Bogen von rechts oben gegen den Uhrzeigersinn ueber links bis zur Mitte (PIL: Winkel im Uhrzeigersinn)
    d.arc([cx - ro, yt - ro, cx + ro, yt + ro], 90, 330, fill=fg, width=sw)
    # untere Schleife: von oben-mitte rechts herum bis links unten
    d.arc([cx - ro, yb - ro, cx + ro, yb + ro], 270, 360, fill=fg, width=sw)
    d.arc([cx - ro, yb - ro, cx + ro, yb + ro], 0, 150, fill=fg, width=sw)
    # Punkt ueber dem S
    pr = r * 0.30
    py = yt - ro - pr * 1.55
    d.ellipse([cx - pr, py - pr, cx + pr, py + pr], fill=fg)


def sign(out, name, text, fg, bg, font='georgia.ttf', w=1024, h=192, logo=None, sub=None, sub_font=None):
    """Fassaden-/Ladenschild (w x h px); der Text wird automatisch eingepasst (nie abgeschnitten)."""
    im = Image.new('RGB', (w, h), bg)
    d = ImageDraw.Draw(im)
    x0 = int(h * 0.12)
    if logo == 'sparkasse':
        sparkasse_logo(d, h * 0.58, h * 0.60, h * 0.26, fg, bg)
        x0 = int(h * 1.10)
    d.rectangle([3, 3, w - 4, h - 4], outline=tuple(int(c * 0.6 + b * 0.4) for c, b in zip(fg, bg)), width=3)
    box_w = (w - x0 - h * 0.12) * 0.92
    if sub:
        f, bb = _fit_text(d, text, font, box_w, h * 0.46, h * 0.6)
        f2, bb2 = _fit_text(d, sub, sub_font or font, box_w, h * 0.16, h * 0.2)
        tw_ = bb[2] - bb[0]
        d.text((x0 + (w - x0 - tw_) / 2 - bb[0], h * 0.10 - bb[1]), text, font=f, fill=fg)
        d.text((x0 + (w - x0 - (bb2[2] - bb2[0])) / 2 - bb2[0], h * 0.70 - bb2[1]), sub, font=f2, fill=fg)
    else:
        f, bb = _fit_text(d, text, font, box_w, h * 0.58, h * 0.62)
        tw_ = bb[2] - bb[0]; th_ = bb[3] - bb[1]
        d.text((x0 + (w - x0 - tw_) / 2 - bb[0], (h - th_) / 2 - bb[1]), text, font=f, fill=fg)
    im.save(os.path.join(out, name + '_col.png'))
    print('sign', name, 'font', f.size)


def blade(out, name, lines, fg, bg, font='arialbd.ttf', logo=None, size=512):
    """Ausleger-/Nasenschild (quadratisch, beidseitig): Logo oder 1-2 Textzeilen"""
    im = Image.new('RGB', (size, size), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, size - 9, size - 9], outline=fg, width=7)
    if logo == 'sparkasse':
        sparkasse_logo(d, size * 0.5, size * 0.58, size * 0.25, fg, bg)
    else:
        n = len(lines)
        hs = [0.34, 0.12] if n == 2 else [0.30]
        ys = [0.40, 0.72] if n == 2 else [0.5]
        for (txt, fnt), hf, yc in zip(lines, hs, ys):
            f, bb = _fit_text(d, txt, fnt or font, size * 0.80, size * hf, size * hf * 1.3)
            d.text((size / 2 - (bb[2] - bb[0]) / 2 - bb[0], size * yc - (bb[3] - bb[1]) / 2 - bb[1]), txt, font=f, fill=fg)
    im.save(os.path.join(out, name + '_col.png'))
    print('blade', name)


def decal(out, name, lines, font='arial.ttf', w=1024, h=256):
    """Glasbeschriftung (weiss, RGBA, Alpha-Maske): Zeilen [(text, Hoehenanteil)]"""
    im = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(im)
    y = h * 0.10
    for txt, hf in lines:
        f, bb = _fit_text(d, txt, font, w * 0.94, h * hf, h * hf * 1.3)
        d.text((w / 2 - (bb[2] - bb[0]) / 2 - bb[0], y - bb[1]), txt, font=f, fill=(255, 255, 255, 255))
        y += h * hf + h * 0.10
    im.save(os.path.join(out, name + '_col.png'))
    print('decal', name)


def signs(out):
    sign(out, 'sign_sparkasse', 'Sparkasse', (255, 255, 255), (230, 0, 0), font='arialbd.ttf', logo='sparkasse')
    sign(out, 'sign_stichnoth', 'Stichnoth', (196, 160, 98), (24, 24, 24), font='georgia.ttf', sub='GOLDSCHMIEDE', sub_font='georgia.ttf')
    sign(out, 'sign_viviry', 'VIVIRY', (26, 26, 26), (240, 238, 232), font='arial.ttf')
    sign(out, 'sign_atrium', 'Atrium', (235, 230, 220), (40, 40, 40), font='georgia.ttf', sub='RESTAURANT', sub_font='georgia.ttf')
    sign(out, 'sign_ratskeller', 'Altes Rathaus', (235, 220, 180), (50, 34, 26), font='georgia.ttf')
    blade(out, 'blade_sparkasse', [], (255, 255, 255), (230, 0, 0), logo='sparkasse')
    blade(out, 'blade_stichnoth', [('Stichnoth', 'georgia.ttf'), ('GOLDSCHMIEDE', 'georgia.ttf')], (196, 160, 98), (24, 24, 24))
    blade(out, 'blade_viviry', [('VIVIRY', 'arial.ttf')], (26, 26, 26), (240, 238, 232))
    blade(out, 'blade_atrium', [('Atrium', 'georgia.ttf'), ('RESTAURANT', 'georgia.ttf')], (235, 230, 220), (40, 40, 40))
    blade(out, 'blade_ratskeller', [('Altes', 'georgia.ttf'), ('Rathaus', 'georgia.ttf')], (235, 220, 180), (50, 34, 26))
    decal(out, 'decal_sparkasse', [('SB-Service', 0.34), ('Geldautomaten  Kontoauszug', 0.20)], font='arialbd.ttf')
    decal(out, 'decal_stichnoth', [('Goldschmiede Stichnoth', 0.34), ('Karmarschstrasse 42', 0.20)], font='georgia.ttf')
    decal(out, 'decal_viviry', [('VIVIRY', 0.38), ('Mode', 0.18)], font='arial.ttf')
    decal(out, 'decal_atrium', [('Restaurant Atrium', 0.34), ('im Alten Rathaus', 0.20)], font='georgia.ttf')
    decal(out, 'decal_white', [('Altes Rathaus', 0.32), ('Karmarschstrasse 42', 0.20)], font='georgia.ttf')


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'tex'
    os.makedirs(out, exist_ok=True)
    which = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    jobs = {'brick': brick_red, 'brick_black': brick_black, 'brick_band': brick_band, 'brick19': brick_19,
            'sandstone': sandstone, 'ashlar': ashlar, 'granite': granite, 'slate': slate, 'plaster': plaster,
            'roof': roof, 'ridge': ridge, 'frieze': frieze, 'quatrefoil': quatrefoil, 'rosette': rosette, 'tracery': tracery,
            'tracery_red': tracery_red, 'tracery_brick': tracery_brick, 'leaf_tiles': leaf_tiles,
            'wechsel': wechsel, 'brick_gable': brick_gable}
    for k, f in jobs.items():
        if which and k not in which: continue
        f(out)
    if not which or 'signs' in which:
        signs(out)
    json.dump(TILES, open(os.path.join(out, 'tiles.json'), 'w'), indent=1)

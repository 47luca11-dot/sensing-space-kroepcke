"""Prozedurale, kachelbare PBR-Texturen (numpy, 1024 px): *_col.png (sRGB), *_nrm.png (Tangentenraum, glTF), *_orm.png.
Konvention: Bildzeile = v-Richtung (v waechst mit der Zeilennummer); UV v = Welt-oben an Waenden.
Normalen: n = (-dh/du, +dh/dv, 1) (glTF: +Y = Bild-oben = -v).
Aufruf: python lh_textures.py [name ...]  -> build/tex/"""
import math, os, sys
import numpy as np
from PIL import Image
import lh_common as C

N = 1024
OUT = C.TEX


def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def snoise(n, beta, seed, aniso=(1.0, 1.0)):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((n, n)))
    fx = np.fft.fftfreq(n) * aniso[0]; fy = np.fft.fftfreq(n)[:, None] * aniso[1]
    f = np.sqrt(fx[None, :] ** 2 + fy ** 2); f[0, 0] = 1.0
    F = F * f ** (-beta / 2.0); F[0, 0] = 0
    r = np.real(np.fft.ifft2(F))
    r -= r.min(); r /= max(r.max(), 1e-9)
    return r


def blur(a, s):
    n = a.shape[0]
    fx = np.fft.fftfreq(n); f2 = fx[None, :] ** 2 + fx[:, None] ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * (math.pi * s) ** 2 * f2)))


def save(name, col, h, tile_m, rough, metal=0.0, ao_strength=1.0, nstrength=1.0, ao=None):
    """col (n,n,3) linear-ish 0..1 (als sRGB gespeichert), h Hoehe [m], tile_m Kachelgroesse [m]."""
    n = col.shape[0]; px = tile_m / n
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * px)
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * px)
    nv = np.dstack([-gx * nstrength, gy * nstrength, np.ones_like(h)])
    nv /= np.linalg.norm(nv, axis=2, keepdims=True)
    if ao is None:
        hb = blur(h, 6.0)
        cav = np.clip((h - hb) / max(np.std(h - hb) * 3, 1e-6), -1, 1)
        ao = np.clip(1.0 - ao_strength * 0.35 * np.clip(-cav, 0, 1), 0, 1)
    r = np.broadcast_to(np.asarray(rough, float), h.shape)
    m = np.broadcast_to(np.asarray(metal, float), h.shape)
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray((np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, name + '_col.png'), optimize=True)
    Image.fromarray(((nv * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, name + '_nrm.png'), optimize=True)
    Image.fromarray((np.dstack([ao, r, m]) * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(os.path.join(OUT, name + '_orm.png'), optimize=True)
    print('tex', name)


def grid_xy(n=N):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    return x / n, y / n           # 0..1


def blocks(n, rows, seed, minw, maxw, row_h=None, warp=0.0, warp_beta=3.2, stagger=True):
    """Lagen-Mauerwerk: rows Lagen (Hoehen optional), Steinlaengen minw..maxw (Anteile 0..1 der Kachelbreite).
    Gibt (block_id, dist_to_edge [0..1 Kachelmass], row_index, local_y) zurueck; kachelbar."""
    rng = np.random.default_rng(seed)
    x, y = grid_xy(n)
    if warp > 0:
        x = x + (snoise(n, warp_beta, seed + 11) - 0.5) * warp
        y = y + (snoise(n, warp_beta, seed + 12) - 0.5) * warp
        x %= 1.0; y %= 1.0
    if row_h is None:
        hs = np.full(rows, 1.0 / rows)
    else:
        hs = np.asarray(row_h, float); hs = hs / hs.sum()
    ry = np.r_[0, np.cumsum(hs)]
    r = np.clip(np.searchsorted(ry, y, side='right') - 1, 0, len(hs) - 1)
    ly = y - ry[r]
    dy = np.minimum(ly, hs[r] - ly)
    bid = np.zeros((n, n), int); dx = np.zeros((n, n))
    base = 0
    for k in range(len(hs)):
        ws = []
        while sum(ws) < 1.0:
            ws.append(rng.uniform(minw, maxw))
        ws = np.array(ws); ws *= 1.0 / ws.sum()
        off = rng.uniform(0, 1) if stagger else 0.0
        cx = np.r_[0, np.cumsum(ws)]
        m = r == k
        xx = (x[m] + off) % 1.0
        b = np.clip(np.searchsorted(cx, xx, side='right') - 1, 0, len(ws) - 1)
        lx = xx - cx[b]
        dx[m] = np.minimum(lx, ws[b] - lx)
        bid[m] = base + b
        base += len(ws)
    return bid, np.minimum(dx, dy), r, ly, base


def palette_pick(bid, nb, cols, seed, weights=None, jitter=0.06):
    rng = np.random.default_rng(seed)
    cols = np.array([hexc(c) for c in cols])
    w = None if weights is None else np.asarray(weights, float) / np.sum(weights)
    pick = rng.choice(len(cols), size=nb, p=w)
    bc = cols[pick] * (1 + rng.uniform(-jitter, jitter, (nb, 1))) + rng.uniform(-0.008, 0.008, (nb, 3))
    return bc[bid]


# ------------------------------------------------------------------------------------------- Materialien
def ashlar(name='ashlar', base=('#d6cbb0', '#d0c4a8', '#cdbfa2', '#d8ceb6', '#c9bc9e'), joint='#b5a88d', tile=2.0, rows=6, seed=1):
    bid, d, r, ly, nb = blocks(N, rows, seed, 0.28, 0.55)
    col = palette_pick(bid, nb, base, seed + 1, jitter=0.035)
    g = snoise(N, 2.2, seed + 3); f = snoise(N, 1.0, seed + 4)
    col *= (0.93 + 0.1 * g[..., None]) * (0.97 + 0.06 * f[..., None])
    # Verwitterung: senkrechte Schlieren
    st = snoise(N, 3.0, seed + 5, aniso=(1.0, 0.15))
    col *= (0.96 + 0.06 * st[..., None])
    jw = 0.006 / tile * 1.0
    jm = d < jw
    col[jm] = col[jm] * 0.82 + hexc(joint) * 0.18
    h = np.clip(d / (0.012 / tile), 0, 1) * 0.004 + f * 0.0015 + g * 0.0006
    save(name, col, h, tile, rough=0.82)


def fine_stone(name, base='#d4c8ac', tile=1.0, seed=7, rough=0.8, var=0.08, speck=0.04):
    g = snoise(N, 2.0, seed); f = snoise(N, 0.8, seed + 1); b = snoise(N, 3.2, seed + 2)
    col = hexc(base)[None, None, :] * (1 - var / 2 + var * b[..., None]) * (1 - speck / 2 + speck * f[..., None])
    col *= (0.98 + 0.04 * g[..., None])
    h = f * 0.0008 + g * 0.0012 + b * 0.002
    save(name, col, h, tile, rough=rough)


def plain_tiles(name, cols, tile=1.0, rows=6, per_row=5, seed=21, rough=0.7):
    """Biberschwanz-Deckung: v nach oben = Richtung First; runde Unterkante; Reihen versetzt."""
    rng = np.random.default_rng(seed)
    x, y = grid_xy(N)
    rh = 1.0 / rows; tw = 1.0 / per_row
    k = np.floor(y / rh).astype(int); ly = y - k * rh
    def tile_xy(k_, x_):
        off = (k_ % 2) * tw / 2
        xx = (x_ + off) % 1.0
        i = np.floor(xx / tw).astype(int); lx = xx - i * tw
        return i, lx
    i, lx = tile_xy(k, x)
    arc = rh * 0.32 * (1 - np.sqrt(np.clip(1 - ((lx - tw / 2) / (tw / 2)) ** 2, 0, 1)))
    own = ly >= arc
    # Pixel unter dem Bogen gehoeren zur Reihe darunter (k-1), dort weit oben
    k2 = np.where(own, k, k - 1) % rows
    ly2 = np.where(own, ly, ly + rh)
    i2, lx2 = tile_xy(k2, x)
    tid = k2 * per_row + i2
    nt = rows * per_row
    col = palette_pick(tid, nt, cols, seed + 1, jitter=0.07)
    t = ly2 / (rh * 1.32)                 # 0 unten (dick) .. 1 oben (unter naechster Reihe)
    h = (1 - np.clip(t, 0, 1)) * 0.016
    gap = np.minimum(lx2, tw - lx2) < 0.0035
    h = np.where(gap, h - 0.006, h)
    sh = np.clip(1 - (ly - arc) / (rh * 0.18), 0, 1) * own          # Schatten unter dem Bogen der eigenen Reihe? (oben)
    shade = 1 - 0.45 * np.clip(1 - (ly2 - rh) / (0.32 * rh), 0, 1) * (~own)
    g = snoise(N, 2.0, seed + 5)
    col *= (0.9 + 0.15 * g[..., None]) * shade[..., None]
    col[gap] *= 0.55
    h += g * 0.001
    save(name, col, h, tile, rough=rough, ao_strength=1.6)


def brick(name, cols, joint='#d2cab9', tile=1.0, rows=12, per_row=4, seed=31, rough=0.6):
    rng = np.random.default_rng(seed)
    x, y = grid_xy(N)
    rh = 1.0 / rows; bw = 1.0 / per_row
    k = np.floor(y / rh).astype(int); ly = y - k * rh
    xx = (x + (k % 2) * bw / 2) % 1.0
    i = np.floor(xx / bw).astype(int); lx = xx - i * bw
    bid = k * per_row + i
    col = palette_pick(bid, rows * per_row, cols, seed + 1, jitter=0.06)
    jw = 0.011 / tile
    d = np.minimum(np.minimum(lx, bw - lx), np.minimum(ly, rh - ly))
    jm = d < jw / 2
    g = snoise(N, 1.2, seed + 2)
    col *= (0.94 + 0.1 * g[..., None])
    col[jm] = hexc(joint) * (0.92 + 0.12 * g[jm][:, None])
    h = np.clip((d - jw / 2) / 0.003, 0, 1) * 0.006 + g * 0.0006
    rgh = np.where(jm, 0.9, rough)
    save(name, col, h, tile, rough=rgh)


def rubble(name, cols, weights, mortar, tile=3.0, seed=41, row_min=0.06, row_max=0.11, wmin=0.07, wmax=0.2, warp=0.012,
           joint=0.010, bulge=0.025, rough=0.88):
    rng = np.random.default_rng(seed)
    hs = []
    while sum(hs) < 1.0: hs.append(rng.uniform(row_min, row_max))
    bid, d, r, ly, nb = blocks(N, len(hs), seed, wmin, wmax, row_h=hs, warp=warp)
    col = palette_pick(bid, nb, cols, seed + 1, weights=weights, jitter=0.07)
    jn = snoise(N, 2.5, seed + 3)
    jw = joint * (0.6 + 0.8 * jn)
    stone = d > jw / 2
    g = snoise(N, 1.4, seed + 4); b = snoise(N, 2.6, seed + 5)
    col *= (0.88 + 0.2 * g[..., None]) * (0.92 + 0.12 * b[..., None])
    mc = hexc(mortar) * (0.9 + 0.15 * g[..., None])
    col = np.where(stone[..., None], col, mc)
    e = np.clip((d - jw / 2) / (bulge / tile * 1.0 + 1e-6), 0, 1)
    h = np.where(stone, 0.012 + 0.018 * np.sqrt(e) + g * 0.004, 0.004 * jn)
    # dunkle Verwitterung unten an Steinen
    save(name, col, h, tile, rough=np.where(stone, rough, 0.95), ao_strength=1.8, nstrength=1.0)


def concrete(name, base='#d8d4c8', tile=2.0, seed=51, boards=True, rough=0.85):
    g = snoise(N, 2.4, seed); f = snoise(N, 0.9, seed + 1); st = snoise(N, 3.0, seed + 2, aniso=(1, 0.12))
    col = hexc(base)[None, None, :] * (0.95 + 0.06 * g[..., None]) * (0.97 + 0.05 * f[..., None]) * (0.94 + 0.08 * st[..., None])
    h = f * 0.0006 + g * 0.0008
    if boards:
        x, y = grid_xy(N)
        k = np.floor(y * 16).astype(int); ly = y * 16 - k
        h += np.where(ly < 0.03, -0.0008, 0.0)
        rng = np.random.default_rng(seed); bv = rng.uniform(0.97, 1.03, 16)
        col *= bv[k][..., None]
    save(name, col, h, tile, rough=rough)


def bushhammer(name, base='#cfc4ad', tile=1.0, seed=61):
    x, y = grid_xy(N)
    f = snoise(N, 0.5, seed); g = snoise(N, 2.0, seed + 1)
    flutes = 0.5 + 0.5 * np.cos(x * 2 * math.pi * 40)
    col = hexc(base)[None, None, :] * (0.85 + 0.2 * f[..., None]) * (0.95 + 0.08 * g[..., None]) * (0.95 + 0.05 * flutes[..., None])
    h = f * 0.002 + flutes * 0.0015
    save(name, col, h, tile, rough=0.95, ao_strength=1.4)


def slate_clad(name, tile=2.0, seed=71):
    bid, d, r, ly, nb = blocks(N, 4, seed, 0.5, 0.5, stagger=False)
    col = palette_pick(bid, nb, ['#4b4c4e', '#47484a', '#505153', '#444547'], seed + 1, jitter=0.04)
    cl = snoise(N, 1.6, seed + 2, aniso=(0.3, 1.0))
    col *= (0.85 + 0.25 * cl[..., None])
    jm = d < 0.002
    col[jm] *= 0.5
    h = cl * 0.002 + np.clip(d / 0.003, 0, 1) * 0.002
    save(name, col, h, tile, rough=0.6)


def wood(name, base='#4a3024', tile=1.0, seed=81, rough=0.6):
    g = snoise(N, 2.2, seed, aniso=(8.0, 0.25)); f = snoise(N, 1.0, seed + 1)
    col = hexc(base)[None, None, :] * (0.8 + 0.35 * g[..., None]) * (0.95 + 0.08 * f[..., None])
    h = g * 0.0006
    save(name, col, h, tile, rough=rough)


def leaded(name='leaded', tile=1.0, cols=4, rows=6, seed=91):
    """Bleiverglasung je Fensterfluegel (UV 0..1): kleine Rechteckscheiben, dunkles Glas mit Himmelsverlauf."""
    x, y = grid_xy(N)
    rng = np.random.default_rng(seed)
    cx = (x * cols) % 1.0; cy = (y * rows) % 1.0
    d = np.minimum(np.minimum(cx, 1 - cx) / cols, np.minimum(cy, 1 - cy) / rows)
    lead = d < 0.006
    pi = (np.floor(x * cols) + cols * np.floor(y * rows)).astype(int)
    tint = rng.uniform(-0.04, 0.04, cols * rows + 1)[pi]
    sky = 0.18 + 0.22 * y                                   # oben heller (Himmel)
    g = snoise(N, 2.0, seed + 1)
    col = np.dstack([sky * 0.82 + tint, sky * 0.9 + tint, sky + tint * 0.5]) * (0.9 + 0.2 * g[..., None])
    col[lead] = hexc('#3a3a38')
    h = np.where(lead, 0.002, 0.0) + g * 0.0003
    save(name, col, h, 0.6, rough=np.where(lead, 0.6, 0.08), metal=np.where(lead, 0.3, 0.0))


def majolica(name='majolica', seed=101):
    """Noltehaus-Fries: tuerkis-gruene Kacheln mit Medaillons und Rankwerk (u: 1 Kachel = 1.2 m, v = Bandhoehe)."""
    x, y = grid_xy(N)
    col = np.dstack([np.full((N, N), 0.25), np.full((N, N), 0.47), np.full((N, N), 0.42)])
    g = snoise(N, 1.8, seed)
    # Medaillon in der Mitte
    r = np.hypot((x - 0.5) * 2.6, (y - 0.5) * 1.0)
    med = r < 0.42
    ring = (r > 0.36) & (r < 0.44)
    col[med] = hexc('#d9d3bf')
    col[ring] = hexc('#b8962e')
    # Ranken links/rechts
    for cx0 in (0.17, 0.83):
        ang = np.arctan2(y - 0.5, (x - cx0) * 2.6); rr = np.hypot((x - cx0) * 2.6, y - 0.5)
        sp = np.abs(((rr * 9 - ang / math.pi) % 1.0) - 0.5) < 0.12
        sp &= rr < 0.42
        col[sp] = hexc('#e6d9a8')
    border = (y < 0.06) | (y > 0.94)
    col[border] = hexc('#c9c2ab')
    col *= (0.9 + 0.15 * g[..., None])
    h = med * 0.004 + ring * 0.006 + border * 0.003
    save(name, col, h.astype(float), 1.2, rough=np.where(border, 0.6, 0.25))


def granite(name='granite', base='#8e8b84', tile=1.0, seed=111):
    rng = np.random.default_rng(seed)
    g = snoise(N, 0.2, seed); f = snoise(N, 1.6, seed + 1)
    col = hexc(base)[None, None, :] * (0.8 + 0.35 * g[..., None]) * (0.95 + 0.08 * f[..., None])
    sp = rng.random((N, N)) < 0.04
    col[sp] *= 0.55
    h = g * 0.0008
    save(name, col, h, tile, rough=0.7)


def green_roof(name='green_roof', tile=4.0, seed=121):
    g = snoise(N, 1.0, seed); f = snoise(N, 2.4, seed + 1); b = snoise(N, 3.0, seed + 2)
    c1 = hexc('#6d6646'); c2 = hexc('#4f5a38'); c3 = hexc('#8a7a5a')
    m = np.clip((b - 0.4) * 3, 0, 1)[..., None]
    col = c1 * (1 - m) + c2 * m
    col = col * (0.8 + 0.35 * g[..., None])
    col = np.where((f > 0.75)[..., None], c3 * (0.9 + 0.2 * g[..., None]), col)
    h = g * 0.01 + f * 0.01
    save(name, col, h, tile, rough=0.95)


def bronze(name='bronze', tile=0.5, seed=131):
    g = snoise(N, 2.0, seed); f = snoise(N, 1.0, seed + 1)
    c1 = hexc('#3d4a3c'); c2 = hexc('#5f8a72')
    m = np.clip((g - 0.45) * 3, 0, 1)[..., None]
    col = (c1 * (1 - m) + c2 * m) * (0.85 + 0.25 * f[..., None])
    h = f * 0.0005
    save(name, col, h, tile, rough=0.55, metal=np.where(m[..., 0] > 0.5, 0.2, 0.6))


def plaster(name='plaster', base='#d8ceb8', tile=2.0, seed=141):
    g = snoise(N, 2.4, seed); f = snoise(N, 0.6, seed + 1); st = snoise(N, 3.0, seed + 2, aniso=(1, 0.15))
    col = hexc(base)[None, None, :] * (0.95 + 0.06 * g[..., None]) * (0.97 + 0.04 * f[..., None]) * (0.95 + 0.07 * st[..., None])
    h = f * 0.0006 + g * 0.0008
    save(name, col, h, tile, rough=0.9)


ALL = {
    'ashlar': lambda: ashlar(),
    'sand_fine': lambda: fine_stone('sand_fine', '#d2c6aa', 1.0, 7),
    'sand_white': lambda: fine_stone('sand_white', '#e4ddcc', 1.0, 9, var=0.05),
    'sand_plinth': lambda: fine_stone('sand_plinth', '#bfb092', 1.0, 11, var=0.12),
    'tile_dark': lambda: plain_tiles('tile_dark', ['#3f3a37', '#45403c', '#383331', '#4a433d', '#3c3735'], seed=21),
    'tile_red': lambda: plain_tiles('tile_red', ['#9a4c35', '#8c4430', '#a5563c', '#7f3d2c', '#94503a'], seed=23),
    'tile_brown': lambda: plain_tiles('tile_brown', ['#6e3f2f', '#633729', '#784634', '#5b3326', '#6a4031'], seed=25, rows=7, per_row=6),
    'brick_yellow': lambda: brick('brick_yellow', ['#dcb071', '#d6a869', '#e2b97c', '#cf9f60', '#e0b676', '#d3a465'], seed=31),
    'rubble_tower': lambda: rubble('rubble_tower', ['#c7bb9f', '#bdb194', '#d1c6ad', '#b4a78b', '#a99c82', '#c4b59c', '#9a8e78', '#b8a48f'],
                                   [5, 5, 4, 3, 2, 1.5, 1, 1], '#cfc6b2', tile=3.0, seed=41),
    'rubble_wall': lambda: rubble('rubble_wall', ['#cdc2a8', '#c2b79c', '#d6ccb4', '#b8ac92', '#ab9f86', '#cfc2a8', '#9f937d'],
                                  [5, 5, 4, 3, 2, 1.5, 1], '#d2cab6', tile=3.0, seed=43, row_min=0.05, row_max=0.09),
    'rubble_dark': lambda: rubble('rubble_dark', ['#6b5a4a', '#5c4e42', '#7a6654', '#4f443b', '#857261'], [3, 3, 2, 2, 1], '#8a8174',
                                  tile=2.0, seed=45, row_min=0.12, row_max=0.2, wmin=0.12, wmax=0.3),
    'splitstone': lambda: rubble('splitstone', ['#a8925f', '#9c8657', '#b49d6a', '#8f7a50', '#a38b63', '#b8a37a', '#8a7b62', '#c0a774'],
                                 [5, 4, 4, 3, 3, 2, 2, 1], '#8f8574', tile=3.0, seed=47, row_min=0.035, row_max=0.07, wmin=0.06, wmax=0.18,
                                 warp=0.006, joint=0.007, bulge=0.02),
    'concrete': lambda: concrete('concrete', '#d9d5c9', seed=51),
    'concrete_dark': lambda: concrete('concrete_dark', '#a7a49b', seed=53, boards=False),
    'bushhammer': lambda: bushhammer('bushhammer'),
    'slate_clad': lambda: slate_clad('slate_clad'),
    'wood_dark': lambda: wood('wood_dark', '#4a2f24', seed=81),
    'wood_door': lambda: wood('wood_door', '#3b241b', seed=83, rough=0.5),
    'leaded': lambda: leaded(),
    'majolica': lambda: majolica(),
    'granite': lambda: granite(),
    'green_roof': lambda: green_roof(),
    'bronze': lambda: bronze(),
    'plaster': lambda: plaster(),
    'plaster_grey': lambda: plaster('plaster_grey', '#c9c4b8', seed=143),
}

if __name__ == '__main__':
    names = sys.argv[1:] or list(ALL)
    for n_ in names: ALL[n_]()

"""Leineschloss -- prozedurale, kachelbare PBR-Texturen (numpy/PIL, 1024 px): Farbe (JPG) + Normalmap (PNG, OpenGL/glTF).
Ausgabe: build/tex/<name>.jpg, <name>_n.png  und build/tex/tex_meta.json (Kachelgroesse in Metern je Material)."""
import os, json, math
import numpy as np
from PIL import Image
from scipy import ndimage
from ls_common import BUILD

OUT = os.path.join(BUILD, 'tex'); os.makedirs(OUT, exist_ok=True)
N = 1024
RNG = np.random.default_rng(1830)


def vnoise(res, n=N, seed=None):
    """Kachelbare Wertrauschen-Oktave: Zufallsgitter res x res, kubisch periodisch hochskaliert."""
    r = np.random.default_rng(seed) if seed is not None else RNG
    g = r.random((res, res))
    z = ndimage.zoom(g, n / res, order=3, mode='grid-wrap', grid_mode=True)
    return z[:n, :n]


def fbm(base=4, oct=6, gain=0.5, n=N, seed=None):
    r = np.random.default_rng(seed)
    out = np.zeros((n, n)); amp = 1.0; tot = 0
    res = base
    for i in range(oct):
        out += amp * vnoise(min(res, n), n, r.integers(1 << 30)); tot += amp
        amp *= gain; res *= 2
    return out / tot


def normal_from_height(h, strength=4.0):
    gy, gx = np.gradient(np.pad(h, 1, mode='wrap'))
    gx = gx[1:-1, 1:-1]; gy = gy[1:-1, 1:-1]
    nx = -gx * strength; ny = gy * strength; nz = np.ones_like(h)   # OpenGL: +Y oben (Bildzeilen laufen nach unten)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / l, ny / l, nz / l], -1)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def save(name, rgb, height=None, strength=4.0):
    rgb = np.clip(rgb, 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(OUT, name + '.jpg'), quality=90)
    if height is not None:
        Image.fromarray(normal_from_height(height, strength)).save(os.path.join(OUT, name + '_n.png'))


def hexcol(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def tint(base, var, field, amt):
    return np.clip(base[None, None, :] * (1 + amt * (field[..., None] - 0.5)) + var, 0, 1)


def stone_base(col, seed, grain=0.10, patch=0.18):
    f1 = fbm(6, 6, 0.55, seed=seed); f2 = fbm(2, 3, 0.5, seed=seed + 1)
    g = RNG.random((N, N)) * 0.5 + fbm(64, 3, 0.5, seed=seed + 2) * 0.5
    rgb = col[None, None, :] * (1 + patch * (f2[..., None] - 0.5) + grain * (g[..., None] - 0.5) + 0.12 * (f1[..., None] - 0.5))
    # Lagerung (feine waagrechte Baender) im Sandstein
    band = fbm(4, 4, 0.5, seed=seed + 3)
    y = np.linspace(0, 1, N, endpoint=False)[:, None]
    rgb *= (1 + 0.012 * np.sin(2 * math.pi * (y * 9 + band * 1.5)))[..., None]
    h = 0.6 * g + 0.4 * f1
    return rgb, h


def blocks(rows, row_h_px, min_len, max_len, seed):
    """Quaderverband: liefert Block-ID-Bild und Fugenmaske (Pixel), kachelbar."""
    r = np.random.default_rng(seed)
    ids = np.zeros((N, N), int); joint = np.zeros((N, N), float)
    k = 0
    y0 = 0
    for i in range(rows):
        y1 = y0 + row_h_px
        # Stossfugen periodisch: Laengen so waehlen, dass Summe = N
        xs = [0]
        while xs[-1] < N:
            xs.append(xs[-1] + int(r.integers(min_len, max_len)))
        xs = np.array(xs, float) * N / xs[-1]
        off = r.integers(0, N)
        xs = (np.round(xs).astype(int) + off) % N
        cols = np.zeros(N, int)
        for j in range(len(xs) - 1):
            a, b = xs[j], xs[j + 1]
            if a < b: cols[a:b] = k + j
            else: cols[a:] = k + j; cols[:b] = k + j
        ids[y0:y1] = cols[None, :]
        for x in xs[:-1]:
            joint[y0:y1, (x - 2) % N] = 1; joint[y0:y1, (x - 1) % N] = 1; joint[y0:y1, x % N] = 1
        joint[y0:y0 + 3] = 1
        k += len(xs); y0 = y1
    return ids, joint


def make_all():
    meta = {}
    # 1 Sandstein glatt (Saeulen, Gesimse, Gewaende) -- heller warmer Obernkirchener Ton
    col = hexcol('d8c8a4')
    rgb, h = stone_base(col, 11)
    save('sandstein', rgb, h, 2.0); meta['sandstein'] = 2.0
    # 2 Sandstein-Quader (Portikus, Pavillons): Schichten 0.5 m, Kachel 3 m -> 6 Schichten
    ids, jt = blocks(6, N // 6, 120, 330, 21)
    rgb, h = stone_base(col, 12)
    rr = np.random.default_rng(5).random(ids.max() + 1)
    rgb *= (0.965 + 0.06 * rr[ids])[..., None]
    jt = ndimage.gaussian_filter(jt, 0.9)
    rgb = rgb * (1 - 0.22 * jt[..., None]) + 0.0
    h = h * 0.3 - jt * 1.2 + 0.25 * fbm(32, 3, 0.5, seed=13)
    save('sandstein_quader', rgb, h, 3.0); meta['sandstein_quader'] = 3.0
    # 3 Rustika (Bandrustika, Erdgeschoss): waagrechte V-Fugen alle 0.5 m, schwache Stossfugen
    ids, jt = blocks(6, N // 6, 200, 420, 31)
    rgb, h = stone_base(hexcol('d2c09b'), 14)
    rgb *= (0.94 + 0.1 * np.random.default_rng(6).random(ids.max() + 1)[ids])[..., None]
    y = (np.arange(N) % (N // 6)) / (N // 6)
    groove = np.clip(1 - np.minimum(y, 1 - y) / 0.06, 0, 1)[:, None] * np.ones((1, N))
    rgb *= (1 - 0.45 * groove ** 1.5)[..., None]
    vj = ndimage.gaussian_filter(jt * (groove < 0.01), 0.8)
    rgb *= (1 - 0.15 * vj[..., None])
    h = 0.25 * h - 1.6 * groove - 0.3 * vj
    save('rustika', rgb, h, 4.0); meta['rustika'] = 3.0
    # 4 Putz hellgelb (Leinefluegel, Nordwestfluegel)
    f = fbm(4, 7, 0.55, seed=41); g = fbm(128, 2, 0.5, seed=42)
    base = hexcol('e6d8ae')
    rgb = base[None, None, :] * (1 + 0.04 * (f[..., None] - 0.5) + 0.04 * (g[..., None] - 0.5))
    streak = fbm(16, 3, 0.5, seed=43)
    rgb *= (1 - 0.02 * np.clip(streak - 0.55, 0, 1)[..., None] * 4)
    save('putz', rgb, 0.6 * g + 0.4 * f, 1.2); meta['putz'] = 3.0
    # 5 Schiefer (Mansarddaecher): Rechteckschablonen 0.30 x 0.22 Sichtflaeche, Kachel 2 m -> 9 Reihen
    rows = 9; rh = N // rows
    r = np.random.default_rng(51)
    rgb = np.zeros((N, N, 3)); h = np.zeros((N, N))
    base = hexcol('4b525b')
    for i in range(rows + 1):
        y0 = i * rh
        w = N // 7
        off = (i % 2) * w // 2 + int(r.integers(-8, 8))
        for j in range(8):
            x0 = (j * w + off) % N
            tone = 0.86 + 0.22 * r.random()
            hue = np.array([1.0, 1.0 + 0.012 * r.standard_normal(), 1.0 + 0.02 * r.standard_normal()])
            yy = np.arange(y0, min(y0 + rh, N)); xx = (np.arange(x0, x0 + w - 3)) % N
            if len(yy) == 0: continue
            rgb[np.ix_(yy, xx)] = base * hue * tone
            prof = np.linspace(0.0, 1.0, len(yy))     # Schindel wird nach unten dicker (Ueberdeckung)
            h[np.ix_(yy, xx)] = prof[:, None] * 0.8 + 0.2 * r.random()
    g = fbm(64, 3, 0.5, seed=52)
    rgb *= (0.9 + 0.2 * g)[..., None]
    save('schiefer', rgb, h + 0.2 * g, 6.0); meta['schiefer'] = 2.0
    # 6 Zink / Stehfalzblech (vorbewittert, dunkel)
    f = fbm(8, 5, 0.5, seed=61)
    rgb = hexcol('707579')[None, None, :] * (0.92 + 0.14 * f[..., None])
    save('zink', rgb, f * 0.2, 1.0); meta['zink'] = 2.0
    # 7 Kupfer vorbewittert braun (Plenarsaal-Attika und -Dach)
    f = fbm(8, 6, 0.55, seed=71)
    rgb = hexcol('6a4a35')[None, None, :] * (0.85 + 0.3 * f[..., None])
    x = np.arange(N) % (N // 4)
    seam = (x < 4).astype(float)[None, :] * np.ones((N, 1))
    rgb *= (1 - 0.25 * seam[..., None])
    save('kupfer', rgb, f * 0.3 + seam, 3.0); meta['kupfer'] = 2.0
    # 8 Marmor-Plattenfassade (Plenarsaalgebaeude): Platten 1.50 x 0.75, Kachel 3 m
    f = fbm(3, 7, 0.6, seed=81)
    v1 = np.abs(np.sin(8 * math.pi * (np.linspace(0, 1, N, endpoint=False)[None, :] + 0.9 * fbm(4, 5, 0.55, seed=82))))
    veins = np.clip(1 - v1 / 0.06, 0, 1)
    rgb = hexcol('dcdcd6')[None, None, :] * (0.97 + 0.06 * f[..., None]) - 0.10 * veins[..., None] * np.array([1, 1, 0.95])
    pid = (np.arange(N)[:, None] // (N // 4)) * 2 + (np.arange(N)[None, :] // (N // 2))
    rgb *= (0.96 + 0.07 * np.random.default_rng(83).random(16)[pid])[..., None]
    jt = np.zeros((N, N)); jt[(np.arange(N) % (N // 4)) < 3, :] = 1; jt[:, (np.arange(N) % (N // 2)) < 3] = 1
    jt = ndimage.gaussian_filter(jt, 0.7)
    rgb *= (1 - 0.35 * jt[..., None])
    save('marmor', rgb, -jt + 0.05 * f, 3.0); meta['marmor'] = 3.0
    # 9 Granit (Treppen, Sockel) dunkelgrau, koernig
    g = RNG.random((N, N)); f = fbm(16, 4, 0.5, seed=91)
    rgb = hexcol('6b6c6c')[None, None, :] * (0.8 + 0.25 * f[..., None]) + 0.10 * (g[..., None] > 0.93) - 0.06 * (g[..., None] < 0.1)
    save('granit', rgb, 0.5 * g + 0.5 * f, 1.0); meta['granit'] = 1.2
    # 10 Kaimauer (grober Sandstein, Wasserraender, Moos unten) Kachel 4 m (Hoehe: unten = Wasser)
    ids, jt = blocks(5, N // 5, 140, 380, 101)
    rgb, h = stone_base(hexcol('b9ab8c'), 102, grain=0.18, patch=0.3)
    rgb *= (0.86 + 0.22 * np.random.default_rng(7).random(ids.max() + 1)[ids])[..., None]
    jt = ndimage.gaussian_filter(jt, 1.3)
    yy = np.linspace(0, 1, N)[:, None]
    wet = np.clip((yy - 0.55) / 0.45, 0, 1) * (0.7 + 0.3 * fbm(8, 4, 0.5, seed=103))
    rgb = rgb * (1 - 0.45 * jt[..., None]) * (1 - 0.45 * wet[..., None]) + wet[..., None] * np.array([0.02, 0.05, 0.0])
    save('quai', rgb, h * 0.5 - jt * 1.5, 4.0); meta['quai'] = 4.0
    # 11 Flachdach (Bitumen/Kies hellgrau)
    g = RNG.random((N, N)); f = fbm(8, 5, 0.5, seed=111)
    rgb = hexcol('7c7b77')[None, None, :] * (0.85 + 0.2 * f[..., None] + 0.12 * (g[..., None] - 0.5))
    save('flachdach', rgb, g * 0.5 + f * 0.5, 2.0); meta['flachdach'] = 4.0
    # 12 Holz (Tueren) dunkelbraun, senkrechte Maserung
    f = fbm(4, 5, 0.5, seed=121)
    x = np.linspace(0, 1, N, endpoint=False)[None, :]
    grain = 0.5 + 0.5 * np.sin(2 * math.pi * (x * 60 + 3 * fbm(2, 4, 0.5, seed=122)))
    rgb = hexcol('4a3326')[None, None, :] * (0.8 + 0.25 * grain[..., None] + 0.1 * f[..., None])
    save('holz', rgb, grain * 0.4, 1.0); meta['holz'] = 1.0
    # 13 Putz weiss (Portikusdecke, Laibungen innen)
    f = fbm(8, 6, 0.5, seed=131)
    rgb = hexcol('ece8de')[None, None, :] * (0.96 + 0.06 * f[..., None])
    save('putz_weiss', rgb, f * 0.3, 1.0); meta['putz_weiss'] = 3.0
    # 14 Pflaster/Granitplatten (Podest)
    ids, jt = blocks(4, N // 4, 200, 300, 141)
    g = RNG.random((N, N)); f = fbm(16, 4, 0.5, seed=142)
    rgb = hexcol('8a8984')[None, None, :] * (0.85 + 0.2 * f[..., None]) * (0.92 + 0.1 * np.random.default_rng(8).random(ids.max() + 1)[ids])[..., None]
    jt = ndimage.gaussian_filter(jt, 0.8)
    rgb *= (1 - 0.4 * jt[..., None])
    save('platten', rgb, 0.3 * g - jt, 2.0); meta['platten'] = 3.0
    json.dump(meta, open(os.path.join(OUT, 'tex_meta.json'), 'w'), indent=1)
    print('textures', len(meta))


if __name__ == '__main__':
    make_all()

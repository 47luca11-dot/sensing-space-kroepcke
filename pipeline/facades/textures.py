"""Prozedurale, kachelbare Texturen (numpy + PIL). Graustufen-/Detailtexturen werden per Vertex-/Instanzfarbe eingefaerbt.
UV-Massstab je Material: MATERIALS[...]['period'] (Meter je Texturwiederholung, u/v)."""
import os
import numpy as np
from PIL import Image, ImageFilter


def _noise(n, scales=((4, 0.5), (16, 0.3), (64, 0.2)), seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    for s, amp in scales:
        g = rng.random((s, s))
        g = np.tile(g, (3, 3))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((n * 3, n * 3), Image.BICUBIC)
        a = np.asarray(im, float)[n:2 * n, n:2 * n] / 255.0
        out += amp * (a - 0.5)
    return out


def _save(a, path, q=88):
    a = np.clip(a, 0, 1)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path, quality=q, optimize=True)


def plaster(n=512):
    a = 0.86 + 0.06 * _noise(n, ((8, 0.6), (32, 0.6), (128, 0.5), (256, 0.4)), 1)
    return a


def stone(n=1024, period=2.4):
    rng = np.random.default_rng(2)
    a = np.zeros((n, n, 3))
    base = np.array([0.86, 0.80, 0.70])
    course = 0.4; rows = int(round(period / course)); rh = n // rows
    grain = _noise(n, ((16, 0.5), (64, 0.5), (256, 0.6)), 3)
    for r in range(rows):
        x = 0
        lens = []
        while x < n:
            L = int(rng.uniform(0.6, 1.2) / period * n)
            lens.append(L); x += L
        lens[-1] -= (x - n)
        off = int(rng.integers(n))
        x = 0
        for L in lens:
            tone = 1 + rng.uniform(-0.06, 0.05)
            hue = base * tone + rng.uniform(-0.02, 0.02, 3)
            xs = (np.arange(x, x + L) + off) % n
            a[r * rh:(r + 1) * rh][:, xs] = hue
            j = (x + off) % n
            a[r * rh:(r + 1) * rh, j:j + 3] = base * 0.62
            x += L
        a[r * rh:r * rh + 3, :] = base * 0.62
    a *= (1 + 0.12 * grain)[..., None]
    return a


def brick(n=1024, col=(0.56, 0.24, 0.16), mortar=(0.72, 0.70, 0.66), seed=4):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n, 3))
    mortar = np.array(mortar)
    a[:] = mortar
    courses = 20; ch = n / courses; nb = 6; bw = n / nb
    for r in range(courses):
        y0 = int(r * ch); y1 = int((r + 1) * ch) - 6
        off = (bw / 2) * (r % 2)
        for k in range(-1, nb + 1):
            x0 = int(k * bw + off); x1 = int((k + 1) * bw + off) - 6
            c = np.array(col) * (1 + rng.uniform(-0.18, 0.12)) + rng.uniform(-0.03, 0.03, 3)
            if rng.random() < 0.08:
                c = c * 0.7
            xs = np.arange(max(x0, -n), x1) % n
            a[y0:y1][:, xs] = c
    g = _noise(n, ((32, 0.4), (256, 0.6)), 5)
    a *= (1 + 0.15 * g)[..., None]
    return a


def panel(n=1024, period=2.4):
    rng = np.random.default_rng(6)
    a = np.zeros((n, n, 3))
    pw, ph = 1.2, 0.6
    nx, ny = int(period / pw), int(period / ph)
    for i in range(nx):
        for j in range(ny):
            c = np.array([0.90, 0.87, 0.80]) * (1 + rng.uniform(-0.05, 0.04))
            a[int(j * n / ny):int((j + 1) * n / ny), int(i * n / nx):int((i + 1) * n / nx)] = c
    for i in range(nx):
        a[:, int(i * n / nx):int(i * n / nx) + 4] *= 0.6
    for j in range(ny):
        a[int(j * n / ny):int(j * n / ny) + 4, :] *= 0.6
    tr = _noise(n, ((8, 0.3), (64, 0.4), (512, 0.5)), 7)
    streak = np.abs(_noise(n, ((4, 1.0), (24, 0.6)), 8))
    a *= (1 + 0.035 * tr - 0.025 * (streak > 0.3))[..., None]
    return a


def tiles(n=1024):
    """Hohlpfannen/Biberschwanz-Anmutung: Reihen mit Schattenkante, je Ziegel Farbvariation (grau, wird eingefaerbt)."""
    rng = np.random.default_rng(9)
    a = np.zeros((n, n))
    rows = 8; cols = 8; rh = n // rows; cw = n // cols
    yy = np.arange(rh) / rh
    prof = 0.62 + 0.38 * yy ** 0.7                    # oben dunkel (Ueberdeckung), unten hell
    prof[-4:] = 0.32                                  # Schattenfuge
    for r in range(rows):
        off = (cw // 2) * (r % 2)
        for c in range(cols):
            tone = 1 + rng.uniform(-0.12, 0.1)
            xs = (np.arange(c * cw, (c + 1) * cw) + off) % n
            xx = np.linspace(0, 1, cw)
            wave = 0.88 + 0.12 * np.sin(xx * np.pi)
            a[r * rh:(r + 1) * rh][:, xs] = (prof[:, None] * wave[None, :]) * tone
            a[r * rh:(r + 1) * rh, xs[:3]] *= 0.55
    a *= 1 + 0.08 * _noise(n, ((64, 0.5), (256, 0.5)), 10)
    return a * 0.85


def slate(n=512):
    rng = np.random.default_rng(11)
    a = np.zeros((n, n)); rows = 10; rh = n // rows
    for r in range(rows):
        x = 0; off = int(rng.integers(n))
        while x < n:
            L = int(rng.uniform(0.18, 0.3) * n)
            tone = 0.7 + rng.uniform(-0.12, 0.12)
            xs = (np.arange(x, min(n, x + L)) + off) % n
            a[r * rh:(r + 1) * rh][:, xs] = tone * (0.85 + 0.15 * np.linspace(0, 1, rh))[:, None]
            a[r * rh:(r + 1) * rh, xs[:2]] *= 0.5
            x += L
        a[r * rh + rh - 3:r * rh + rh, :] *= 0.45
    a *= 1 + 0.1 * _noise(n, ((64, 0.5), (256, 0.5)), 12)
    return a


def seam(n=512):
    a = np.full((n, n), 0.62)
    k = 4
    for i in range(k):
        x = int(i * n / k)
        a[:, x:x + 5] = 0.85; a[:, x + 5:x + 9] = 0.42
    a *= 1 + 0.05 * _noise(n, ((8, 0.5), (64, 0.5)), 13)
    return a


def gravel(n=512):
    rng = np.random.default_rng(14)
    a = 0.6 + 0.1 * _noise(n, ((128, 0.6), (256, 0.8)), 15)
    pts = rng.integers(0, n, (9000, 2)); vals = rng.uniform(-0.25, 0.25, 9000)
    for (x, y), v in zip(pts, vals):
        a[y:y + 3, x:x + 4] += v
    return a


def membrane(n=512):
    a = 0.62 + 0.04 * _noise(n, ((16, 0.5), (128, 0.5)), 16)
    k = 4
    for i in range(k):
        y = int(i * n / k)
        a[y:y + 6, :] *= 0.8; a[y + 6:y + 9, :] *= 1.1
    return a


def green(n=512):
    rng = np.random.default_rng(17)
    g = _noise(n, ((32, 0.6), (128, 0.6), (256, 0.5)), 18)
    a = np.zeros((n, n, 3))
    a[..., 0] = 0.40 + 0.25 * g; a[..., 1] = 0.48 + 0.2 * g; a[..., 2] = 0.26 + 0.1 * g
    red = _noise(n, ((16, 1.0), (64, 0.5)), 19) > 0.18
    a[red] = a[red] * np.array([1.35, 0.8, 0.8])
    return a


def brick_y(n=1024):
    """Gelber/ockerfarbener Verblendziegel (Noltehaus, Neorenaissance), helle Fugen."""
    return brick(n, col=(0.86, 0.66, 0.36), mortar=(0.86, 0.84, 0.78), seed=11)


def rubble(n=1024, period=3.0):
    """Bruchsteinmauerwerk (Kalk-/Sandstein): unregelmaessige, lagig versetzte Steine, breite helle Fugen."""
    rng = np.random.default_rng(21)
    a = np.zeros((n, n, 3)); mortar = np.array([0.80, 0.77, 0.70])
    a[:] = mortar
    base = np.array([0.80, 0.75, 0.64])
    y = 0
    while y < n:
        rh = int(rng.uniform(0.12, 0.3) / period * n)
        x = int(rng.integers(n))
        x0 = x
        while x < x0 + n:
            L = int(rng.uniform(0.18, 0.55) / period * n)
            h = max(4, rh - int(rng.integers(3, 9)))
            tone = 1 + rng.uniform(-0.1, 0.06)
            hue = base * tone + rng.uniform(-0.01, 0.01) + rng.uniform(-0.008, 0.008, 3)     # kaum Buntheit (Kalkstein)
            ys = np.arange(y + int(rng.integers(0, 3)), min(n, y + h))
            xs = np.arange(x + 3, x + L - 3) % n
            if len(ys) and len(xs):
                a[np.ix_(ys, xs)] = hue
            x += L
        y += rh
    g = _noise(n, ((16, 0.5), (64, 0.5), (256, 0.6)), 23)
    a *= (1 + 0.14 * g)[..., None]
    return a


TEX = {'plaster': plaster, 'stone': stone, 'brick': brick, 'panel': panel, 'tiles': tiles, 'slate': slate, 'seam': seam,
       'gravel': gravel, 'membrane': membrane, 'green': green, 'brick_y': brick_y, 'rubble': rubble}


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for k, f in TEX.items():
        p = os.path.join(out_dir, k + '.jpg')
        _save(f(), p)
        res[k] = p
    return res


if __name__ == '__main__':
    import sys
    print(build(sys.argv[1]))

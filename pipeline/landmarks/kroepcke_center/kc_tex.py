"""Prozedurale, kachelbare Texturen fuer das Kroepcke-Center (numpy, periodisch via FFT).
Aufruf: python kc_tex.py <outdir>"""
import sys, os
import numpy as np
from PIL import Image

rng = np.random.default_rng(1977)


def pnoise(n, beta, seed=None, aniso=(1.0, 1.0)):
    """periodisches 1/f^beta-Rauschen, normiert auf [-1,1]."""
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, n))
    fy = np.fft.fftfreq(n)[:, None] * aniso[1]; fx = np.fft.fftfreq(n)[None, :] * aniso[0]
    f = np.sqrt(fx * fx + fy * fy); f[0, 0] = 1
    s = np.fft.ifft2(np.fft.fft2(w) / f ** beta).real
    s -= s.mean(); s /= np.abs(s).max() + 1e-9
    return s


def blur_periodic(a, sigma):
    n = a.shape[0]
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy))
    return np.fft.ifft2(np.fft.fft2(a) * g).real


def normal_from_height(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    n = np.dstack([-dx, dy, np.ones_like(h)])            # OpenGL (+Y oben), Bildzeilen nach unten
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def save(a, path):
    a = np.clip(a, 0, 1) if a.dtype != np.uint8 else a
    if a.dtype != np.uint8: a = (a * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)
    print('wrote', path, a.shape)


def limestone(out, n=1024):
    """Kalkstein/Travertin hell, 2 m Kachel: Wolkigkeit, feine Koernung, horizontale Poren, Lagerfugen-Andeutung nein."""
    base = np.array([0.765, 0.725, 0.645])
    cloud = pnoise(n, 1.4, 1) * 0.045 + pnoise(n, 1.0, 2) * 0.02
    grain = rng.standard_normal((n, n)) * 0.018
    # Poren: anisotrop (horizontal gestreckt) gefiltertes Rauschen, nur Spitzen
    pr = pnoise(n, 0.9, 3, aniso=(0.35, 1.6))
    pores = np.clip((pr - 0.45) / 0.25, 0, 1)
    vein = np.clip((pnoise(n, 1.2, 4, aniso=(0.25, 1.4)) - 0.6) / 0.3, 0, 1) * 0.5
    lum = 1 + cloud + grain - pores * 0.22 - vein * 0.06
    tint = pnoise(n, 1.3, 5)[..., None] * np.array([0.012, 0.004, -0.012])
    col = base[None, None, :] * lum[..., None] + tint
    save(col, os.path.join(out, 'kc_limestone.png'))
    h = -pores * 1.0 - vein * 0.2 + grain * 2
    save(normal_from_height(blur_periodic(h, 0.8), 2.2), os.path.join(out, 'kc_limestone_n.png'))


def sedum(out, n=1024):
    """Extensiv-Dachbegruenung (Sedum), 4 m Kachel, nach DOP: oliv-braun mit gruenen/roten Polstern."""
    base = np.array([0.36, 0.33, 0.22])
    m = pnoise(n, 1.1, 11)
    clumps = blur_periodic((rng.random((n, n)) < 0.012).astype(float), 3.0)
    clumps = np.clip(clumps / clumps.max() * 4.0, 0, 1)
    sel = pnoise(n, 1.2, 12)
    green = np.array([0.30, 0.38, 0.16]); red = np.array([0.42, 0.20, 0.14]); dark = np.array([0.16, 0.14, 0.10]); yel = np.array([0.55, 0.50, 0.25])
    col = base[None, None] * (1 + m[..., None] * 0.18)
    w_g = np.clip(sel, 0, 1)[..., None] * clumps[..., None]
    w_r = np.clip(-sel, 0, 1)[..., None] * clumps[..., None]
    col = col * (1 - w_g - w_r) + green * w_g + red * w_r
    spots = blur_periodic((rng.random((n, n)) < 0.03).astype(float), 1.2)
    spots = np.clip(spots / spots.max() * 3, 0, 1)[..., None]
    col = col * (1 - spots * 0.6) + dark * spots * 0.6
    fl = np.clip((pnoise(n, 0.6, 13) - 0.75) / 0.2, 0, 1)[..., None] * 0.6
    col = col * (1 - fl) + yel * fl
    col *= (1 + rng.standard_normal((n, n, 1)) * 0.05)
    save(col, os.path.join(out, 'kc_sedum.png'))


def gravel(out, n=512):
    """Rundkies 16/32 grau, 1 m Kachel (Voronoi-artig)."""
    k = 900
    pts = rng.random((k, 2)) * n
    yy, xx = np.mgrid[0:n, 0:n]
    best = np.full((n, n), 1e9); second = np.full((n, n), 1e9); idx = np.zeros((n, n), int)
    for i, (px, py) in enumerate(pts):
        dx = np.abs(xx - px); dx = np.minimum(dx, n - dx)
        dy = np.abs(yy - py); dy = np.minimum(dy, n - dy)
        d = dx * dx + dy * dy
        m2 = d < second; m1 = d < best
        second = np.where(m1, best, np.where(m2, d, second))
        idx = np.where(m1, i, idx); best = np.where(m1, d, best)
    edge = np.sqrt(second) - np.sqrt(best)
    tone = rng.random(k)[idx] * 0.35 + 0.55
    shade = np.clip(edge / 6.0, 0, 1) ** 0.5
    col = np.dstack([tone * 0.62, tone * 0.61, tone * 0.58]) * (0.35 + 0.65 * shade[..., None])
    save(col, os.path.join(out, 'kc_gravel.png'))
    save(normal_from_height(blur_periodic(shade, 1.0), 3.0), os.path.join(out, 'kc_gravel_n.png'))


def pavers(out, n=512):
    """Betonplatten 50x50 cm auf Stelzlagern, 2 m Kachel (4x4 Platten)."""
    col = np.ones((n, n, 3)) * np.array([0.60, 0.59, 0.56])
    cell = n // 4
    yy, xx = np.mgrid[0:n, 0:n]
    tint = rng.normal(0, 0.035, (4, 4))
    col *= (1 + tint[yy // cell, xx // cell])[..., None]
    col *= (1 + pnoise(n, 1.2, 21)[..., None] * 0.05 + rng.standard_normal((n, n, 1)) * 0.03)
    j = ((xx % cell) < 3) | ((yy % cell) < 3)
    col[j] *= 0.35
    save(col, os.path.join(out, 'kc_pavers.png'))


def lawn(out, n=512):
    base = np.array([0.20, 0.30, 0.10])
    col = base * (1 + pnoise(n, 1.0, 31)[..., None] * 0.25 + rng.standard_normal((n, n, 1)) * 0.10)
    save(col, os.path.join(out, 'kc_lawn.png'))


def louver(out, n=256):
    """Lamellen-Alpha nicht noetig (Geometrie) -- Lochblech fuer Gitterroste."""
    yy, xx = np.mgrid[0:n, 0:n]
    g = (((xx % 16) < 3) | ((yy % 16) < 3)).astype(float)
    col = np.dstack([g * 0.45 + 0.05] * 3)
    a = np.where(g > 0, 1.0, 0.0)
    img = np.dstack([col, a[..., None]])
    save(img, os.path.join(out, 'kc_grating.png'))


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out, exist_ok=True)
    limestone(out); sedum(out); gravel(out); pavers(out); lawn(out); louver(out)

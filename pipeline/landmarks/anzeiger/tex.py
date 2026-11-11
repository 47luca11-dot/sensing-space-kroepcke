"""Procedural PBR textures (1024 px, tiling) for the Anzeiger-Hochhaus + Goseriedebad landmark.
Writes <name>_col.png, <name>_nrm.png (OpenGL +Y), <name>_rgh.png (glTF ORM: G=roughness, B=metallic) into ./tex/.
Run: python tex.py"""
import os, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'tex'); os.makedirs(OUT, exist_ok=True)
N = 1024
rng = np.random.default_rng(1928)


def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def tile_noise(scale, octaves=4, seed=0):
    """periodic value noise in [0,1] (FFT filtered white noise -> tiles perfectly)"""
    r = np.random.default_rng(seed)
    acc = np.zeros((N, N)); amp = 1.0; tot = 0
    for o in range(octaves):
        w = r.standard_normal((N, N))
        f = np.fft.fftfreq(N)
        fx, fy = np.meshgrid(f, f)
        s = scale / (2 ** o)
        filt = np.exp(-(fx ** 2 + fy ** 2) * (s * s) * 20)
        n = np.real(np.fft.ifft2(np.fft.fft2(w) * filt))
        n = (n - n.mean()) / (n.std() + 1e-9)
        acc += amp * n; tot += amp; amp *= 0.5
    acc /= tot
    return np.clip(acc * 0.25 + 0.5, 0, 1)


def normal_from_height(h, strength):
    # periodic gradient
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    nx = -gx * strength; ny = gy * strength      # image rows go down -> +Y (OpenGL) = up in texture
    nz = np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / l, ny / l, nz / l], -1)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def save(name, col, h=None, strength=4.0, rough=None, metal=0.0):
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, name + '_col.png'))
    if h is not None:
        Image.fromarray(normal_from_height(h, strength)).save(os.path.join(OUT, name + '_nrm.png'))
    if rough is not None:
        orm = np.zeros((N, N, 3), np.uint8)
        orm[..., 0] = 255
        orm[..., 1] = (np.clip(rough, 0, 1) * 255).astype(np.uint8)
        orm[..., 2] = (np.clip(metal if np.ndim(metal) else np.full((N, N), metal), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(orm).save(os.path.join(OUT, name + '_rgh.png'))
    print('tex', name)


yy, xx = np.mgrid[0:N, 0:N].astype(float)

# Anzeiger clinker (Bockhorner-type, dark chocolate brown with a few violet / over-burnt glazed headers).
# Calibrated against Wikimedia refs 03 (overcast, median facade sRGB ~ (101,79,69)) and 08 (close-up):
# warm brown, R/G ~ 1.25, G/B ~ 1.15 - NOT red or violet.
KL_PAL = [('#5e463b', 5), ('#664c3e', 4), ('#55403a', 4), ('#4a3832', 3), ('#6f5444', 2), ('#3a2d29', 2), ('#5a4038', 2), ('#4b3b3a', 1)]
KL_JOINT = '#45423e'

# ------------------------------------------------------------------ Anzeiger clinker (2 m x 2 m)
def clinker():
    M = 2.0; ppm = N / M
    course = M / 24; unit = 0.25
    col = np.zeros((N, N, 3)); h = np.zeros((N, N)); rg = np.zeros((N, N))
    palette = KL_PAL
    cols = np.array([hexc(c) for c, w in palette]); wts = np.array([w for c, w in palette], float); wts /= wts.sum()
    joint = 0.011
    row = (yy / ppm / course).astype(int)
    fy = (yy / ppm) - row * course
    noise = tile_noise(3, 3, 5)
    for r in range(24):
        hdr = (r % 2 == 1)
        w = unit / 2 if hdr else unit
        off = (unit / 4 if hdr else 0.0) + (unit / 2 if (r // 2) % 2 else 0.0)
        sel = row == r
        xs = (xx[sel] / ppm - off) % M
        k = (xs / w).astype(int)
        fx = xs - k * w
        nb = int(np.ceil(M / w)) + 1
        ci = rng.choice(len(cols), nb, p=wts)
        jitter = rng.normal(0, 0.008, (nb, 3)) + rng.normal(0, 0.035, (nb, 1)) * np.array([1.0, 0.85, 0.8])
        glaze = rng.random(nb) < 0.14
        c = cols[ci[k % nb]] + jitter[k % nb]
        ex = np.minimum(fx, w - fx); ey = np.minimum(fy[sel], course - fy[sel])
        inside = (ex > joint / 2) & (ey > joint / 2)
        edge = np.clip(np.minimum(ex - joint / 2, ey - joint / 2) / 0.006, 0, 1)
        hh = np.where(inside, 0.55 + 0.45 * np.sqrt(edge), 0.0)
        c = np.where(inside[:, None], c, hexc(KL_JOINT) + rng.normal(0, 0.01, (sel.sum(), 1)))
        col[sel] = c; h[sel] = hh
        rg[sel] = np.where(inside, np.where(glaze[k % nb], 0.58, 0.82), 0.94)
    col *= (0.9 + 0.2 * noise)[..., None]
    h += (tile_noise(0.6, 2, 7) - 0.5) * 0.12
    save('klinker', col, h, 5.0, rg)


def chevron():
    """spandrel panel: brick courses rising toward the centre (^), u 0..1 = panel width (1.40 m), v: 2 m per tile"""
    W, H = 1.4, 2.0
    sx, sy = N / W, N / H
    X = xx / sx; Y = (N - 1 - yy) / sy
    d = np.abs(X - W / 2)
    t = Y + d * 0.8             # along-course coordinate is d, course index from t
    course = 0.083
    row = np.floor(t / course).astype(int)
    fr = t - row * course
    off = (row % 2) * 0.125
    xs = (d + off) % 0.25
    ex = np.minimum(xs, 0.25 - xs); ey = np.minimum(fr, course - fr) * 0.78
    joint = 0.0055
    inside = (ex > joint) & (ey > joint)
    rid = (row * 31 + np.floor((d + off) / 0.25).astype(int) * 7 + (X > W / 2) * 13) % 97
    palette = np.array([hexc(c) for c, w in KL_PAL])
    lut = rng.integers(0, len(palette), 97); jit = rng.normal(0, 0.01, (97, 3)) + rng.normal(0, 0.045, (97, 1))
    c = palette[lut[rid]] + jit[rid]
    col = np.where(inside[..., None], c, hexc(KL_JOINT))
    centre = np.abs(X - W / 2) < 0.012
    col[centre] = hexc(KL_JOINT)
    col *= (0.9 + 0.2 * tile_noise(3, 3, 9))[..., None]
    edge = np.clip(np.minimum(ex - joint, ey - joint) / 0.006, 0, 1)
    h = np.where(inside, 0.55 + 0.45 * np.sqrt(edge), 0)
    rg = np.where(inside, np.where(lut[rid] % 7 == 0, 0.5, 0.8), 0.94)
    save('chevron', col, h, 5.0, rg)


# ------------------------------------------------------------------ sandstone ashlar (3.2 m)
def ashlar():
    M = 3.2; ppm = N / M
    heights = [0.42, 0.36, 0.42, 0.36, 0.42, 0.36, 0.42, 0.44]
    s = sum(heights); heights = [hh * M / s for hh in heights]
    col = np.zeros((N, N, 3)); h = np.zeros((N, N))
    base = hexc('#c4ad88')
    y = (N - 1 - yy) / ppm
    edges = np.cumsum([0] + heights)
    for i in range(len(heights)):
        sel = (y >= edges[i]) & (y < edges[i + 1])
        lens = []
        tot = 0.0
        while tot < M - 0.5:
            l = rng.uniform(0.7, 1.3); lens.append(l); tot += l
        lens.append(M - tot)
        if lens[-1] < 0.35: lens[-2] += lens[-1]; lens.pop()
        off = rng.uniform(0, M)
        xs = (xx[sel] / ppm + off) % M
        be = np.cumsum([0] + lens)
        k = np.searchsorted(be, xs, side='right') - 1
        fx = xs - be[k]
        tint = rng.normal(0, 0.006, (len(lens), 3)) + rng.normal(0, 0.035, (len(lens), 1)) * np.array([1.0, 0.95, 0.85])
        c = base + tint[k]
        fy = y[sel] - edges[i]
        ex = np.minimum(fx, np.array(lens)[k] - fx); ey = np.minimum(fy, heights[i] - fy)
        joint = 0.006
        inside = (ex > joint) & (ey > joint)
        edge = np.clip(np.minimum(ex, ey) / 0.02, 0, 1)
        # weathering: slightly darker towards block bottom
        c = c * (0.95 + 0.06 * np.clip(fy / heights[i], 0, 1))[:, None]
        col[sel] = np.where(inside[:, None], c, hexc('#a39274'))
        h[sel] = np.where(inside, 0.7 + 0.3 * edge, 0.25)
    n1 = tile_noise(1.5, 4, 11); n2 = tile_noise(8, 2, 12)
    col *= (0.88 + 0.2 * n1)[..., None]
    col = col * (0.93 + 0.1 * n2)[..., None]
    h += (tile_noise(0.4, 3, 13) - 0.5) * 0.35
    rg = 0.82 + 0.1 * n1
    save('ashlar', col, h, 3.0, rg)
    # plain carved stone (cornices, ornaments) - same stone without joints
    colp = base[None, None, :] * (0.9 + 0.16 * n1)[..., None] * (0.95 + 0.06 * n2)[..., None]
    save('stone', colp, (tile_noise(0.4, 3, 14) - 0.5) * 0.4, 2.0, 0.82 + 0.1 * n1)


# ------------------------------------------------------------------ red plain tiles (Biberschwanz), 1.8 m tile
def tiles(name='tiles', base='#a9533a', var=0.06, w=0.18, c=0.15):
    M = 1.8; ppm = N / M
    y = (N - 1 - yy) / ppm; x = xx / ppm
    row = np.floor(y / c).astype(int)
    fy = y - row * c                       # 0 at lower edge of the visible course
    off = (row % 2) * w / 2
    k = np.floor((x + off) / w).astype(int)
    fx = (x + off) - k * w
    nr = int(round(M / c)); nk = int(round(M / w)) + 1
    tid = (row % nr) * 37 + (k % nk)
    lut = rng.normal(0, var, (nr * 37 + nk + 40, 1)) * np.array([1.0, 0.75, 0.6]) + rng.normal(0, 0.012, (nr * 37 + nk + 40, 3))
    dark = rng.random(nr * 37 + nk + 40) < 0.12
    b = hexc(base)
    col = b + lut[tid]
    col = np.where(dark[tid][..., None], col * 0.72, col)
    # segmental tile end (bottom edge curved)
    u = (fx / w - 0.5) * 2
    bottom = 0.018 * (u ** 2)
    gap = np.abs(fx - w / 2) > (w / 2 - 0.004)
    shadow = (fy < bottom + 0.006)
    h = 0.4 + 0.6 * np.clip((fy - bottom) / c, 0, 1)
    h = np.where(fy < bottom, 0.0, h)
    col = np.where(shadow[..., None], col * 0.55, col)
    col = np.where(gap[..., None], col * 0.6, col)
    h = np.where(gap, h * 0.6, h)
    n1 = tile_noise(2, 3, 21)
    col *= (0.86 + 0.24 * n1)[..., None]
    moss = tile_noise(6, 2, 22) > 0.72
    col = np.where(moss[..., None], col * 0.85 + hexc('#5d5a3c') * 0.15, col)
    save(name, col, h, 6.0, 0.78 + 0.12 * n1)


# ------------------------------------------------------------------ copper patina (2 m), seams every 0.5 m horizontally
def copper(name, base, streak):
    M = 2.0; ppm = N / M
    y = (N - 1 - yy) / ppm; x = xx / ppm
    n1 = tile_noise(2.5, 4, 31); n2 = tile_noise(0.5, 3, 32)
    # vertical streaks
    st = tile_noise(1, 2, 33)
    st = gaussian_filter(np.tile(st[:1, :], (N, 1)), (0, 2)) if False else st
    vs = np.repeat(tile_noise(1, 2, 34)[N // 2:N // 2 + 1, :], N, 0)
    col = hexc(base) * (0.9 + 0.18 * n1)[..., None]
    col = col * (1 - 0.18 * (vs > 0.62)[..., None]) + hexc(streak) * 0.18 * (vs > 0.62)[..., None]
    col = col * (0.95 + 0.08 * n2)[..., None]
    # cross seams (flat-lock), staggered vertical seams every 0.5 m
    hs = np.abs(((y + 0.25) % 0.5) - 0.25)
    row = np.floor(y / 0.5).astype(int)
    vsx = np.abs(((x + (row % 2) * 0.25) % 0.5) - 0.25)
    seam = np.minimum(hs, vsx)
    h = np.clip(seam / 0.01, 0, 1) * 0.6 + n2 * 0.1
    col = np.where((seam < 0.006)[..., None], col * 0.82, col)
    save(name, col, h, 3.5, 0.55 + 0.2 * n1, 0.15)


def simple(name, base, var, scale, rough, strength=1.5, seed=40):
    n1 = tile_noise(scale, 4, seed); n2 = tile_noise(scale / 6, 3, seed + 1)
    col = hexc(base)[None, None, :] * (1 - var + 2 * var * n1)[..., None] * (0.96 + 0.08 * n2)[..., None]
    save(name, col, n2 * 0.6 + n1 * 0.4, strength, rough + 0.08 * (n1 - 0.5))


def slate():
    M = 1.2; ppm = N / M
    y = (N - 1 - yy) / ppm; x = xx / ppm
    c = 0.1; w = 0.2
    row = np.floor(y / c).astype(int); fy = y - row * c
    off = (row % 2) * w / 2; k = np.floor((x + off) / w).astype(int); fx = x + off - k * w
    tid = (row * 37 + k) % 211
    lut = rng.normal(0, 0.03, (211, 1))
    col = hexc('#4d5157') + lut[tid]
    u = (fx / w - 0.5) * 2
    cut = fy < 0.03 * np.abs(u)          # pointed (Fischschuppen) ends
    col = np.where(cut[..., None], col * 0.6, col)
    h = np.where(cut, 0.0, 0.3 + 0.7 * fy / c)
    save('slate', col * (0.9 + 0.15 * tile_noise(2, 3, 51))[..., None], h, 4.0, 0.62 + 0.1 * tile_noise(1, 2, 52))


if __name__ == '__main__':
    clinker(); chevron(); ashlar()
    tiles('tiles', '#a4513a', 0.05)
    copper('copper', '#a1c8a5', '#73a088')
    copper('copper_dark', '#6c9a80', '#4b6e5f')
    slate()
    simple('roof_flat', '#5b5a57', 0.12, 1.5, 0.92, 2.5, 60)
    simple('plaster', '#d6cdbd', 0.05, 2.0, 0.88, 1.0, 61)
    simple('concrete', '#c4c0b6', 0.07, 1.0, 0.85, 1.2, 62)
    simple('stone_dark', '#5e5953', 0.08, 1.0, 0.8, 1.5, 63)

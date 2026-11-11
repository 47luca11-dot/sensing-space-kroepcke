"""Cheap geometry preview without Blender: painter's-algorithm rasteriser with backface culling (FrontSide only).
Usage: python quickview.py <out.png> <cam_x,cam_y,cam_z> <tgt_x,tgt_y,tgt_z> [fov_deg] [modules]"""
import sys, numpy as np
from PIL import Image, ImageDraw

COL = {
    'klinker': (92, 56, 46), 'chevron': (84, 52, 44), 'concrete': (196, 192, 182), 'stone_dark': (90, 85, 80), 'roof_flat': (90, 90, 88),
    'copper': (158, 196, 176), 'copper_dark': (110, 154, 134), 'paint_white': (236, 236, 230), 'glass': (40, 50, 58), 'glass_shop': (120, 130, 130),
    'gold': (216, 178, 74), 'letter_side': (90, 30, 22), 'metal_dark': (30, 30, 32), 'steel': (160, 160, 160), 'lamp_glass': (230, 225, 200),
    'sign_panel': (180, 178, 170), 'ashlar': (203, 191, 166), 'stone': (198, 186, 160), 'tiles': (168, 84, 58), 'slate': (78, 82, 88),
    'plaster': (214, 205, 189), 'brick_red': (150, 70, 52), 'glass_green': (70, 110, 100), 'alu': (150, 152, 150), 'sandstone_relief': (190, 178, 150),
}


def render(arrs, cam, tgt, fov=40, W=1280, H=720, out='q.png', bg=(200, 210, 222)):
    cam = np.array(cam, float); tgt = np.array(tgt, float)
    f = tgt - cam; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    Ts, Cs = [], []
    L = np.array([0.45, 0.8, 0.35]); L /= np.linalg.norm(L)
    for (o, m), (P, UV) in arrs.items():
        T = P.reshape(-1, 3, 3).astype(float)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ln = np.linalg.norm(n, axis=1); ok = ln > 1e-12
        T = T[ok]; n = n[ok] / ln[ok, None]
        c = T.mean(1)
        vis = np.einsum('ij,ij->i', n, cam - c) > 0          # backface culling
        T = T[vis]; n = n[vis]
        base = np.array(COL.get(m, (255, 0, 255)), float)
        sh = 0.45 + 0.55 * np.clip(n @ L, 0, 1) + 0.1 * n[:, 1]
        Cs.append(np.clip(base[None] * sh[:, None], 0, 255)); Ts.append(T)
    T = np.concatenate(Ts); C = np.concatenate(Cs)
    rel = T - cam
    x = rel @ r; y = rel @ u; z = rel @ f
    keep = (z > 0.3).all(1)
    x, y, z, C = x[keep], y[keep], z[keep], C[keep]
    fpx = (W / 2) / np.tan(np.radians(fov) / 2)
    sx = W / 2 + fpx * x / z; sy = H / 2 - fpx * y / z
    order = np.argsort(-z.mean(1))
    im = Image.new('RGB', (W, H), bg); d = ImageDraw.Draw(im)
    for i in order:
        if (sx[i] < -50).all() or (sx[i] > W + 50).all() or (sy[i] < -50).all() or (sy[i] > H + 50).all(): continue
        d.polygon([(sx[i, 0], sy[i, 0]), (sx[i, 1], sy[i, 1]), (sx[i, 2], sy[i, 2])], fill=tuple(int(v) for v in C[i]))
    im.save(out)
    return out


if __name__ == '__main__':
    import importlib
    sys.path.insert(0, __file__.rsplit('\\', 1)[0] if '\\' in __file__ else '.')
    out = sys.argv[1]
    cam = [float(v) for v in sys.argv[2].split(',')]; tgt = [float(v) for v in sys.argv[3].split(',')]
    fov = float(sys.argv[4]) if len(sys.argv) > 4 else 40
    mods = sys.argv[5].split(',') if len(sys.argv) > 5 else ['anz']
    from geo import MB
    mb = MB()
    for m in mods:
        importlib.import_module(m).build(mb)
    render(mb.arrays(), cam, tgt, fov, out=out)
    print('ok', out, mb.total())

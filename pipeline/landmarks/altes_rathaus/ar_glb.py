"""Minimaler GLB-Schreiber (glTF 2.0) mit PBR-Texturen und EXT_mesh_gpu_instancing.

nodes: [{'name', 'prims': [(mat, P, N, UV, I)], 'inst': None | (T (n,3), Rq (n,4) xyzw, S (n,3)), 'extras': {}}]
materials: {name: {'color': (r,g,b) sRGB 0..1, 'metal', 'rough', 'tex': prefix|None (prefix_col.png/_nrm.png/_orm.png),
                   'nscale', 'emit': (r,g,b)|None, 'double': bool}}
"""
import json, os, struct
import numpy as np


def _srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class _Buf:
    def __init__(self):
        self.chunks = []; self.len = 0
        self.views = []; self.accessors = []

    def add_view(self, data, target=None):
        pad = (-self.len) % 4
        if pad: self.chunks.append(b'\x00' * pad); self.len += pad
        v = {'buffer': 0, 'byteOffset': self.len, 'byteLength': len(data)}
        if target: v['target'] = target
        self.chunks.append(data); self.len += len(data)
        self.views.append(v)
        return len(self.views) - 1

    def add_acc(self, arr, ctype, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        vi = self.add_view(arr.tobytes(), target)
        a = {'bufferView': vi, 'componentType': ctype, 'count': int(arr.shape[0]), 'type': typ}
        if minmax:
            a['min'] = [float(x) for x in arr.min(0)]; a['max'] = [float(x) for x in arr.max(0)]
        self.accessors.append(a)
        return len(self.accessors) - 1


def write_glb(path, nodes, materials, tex_dir, extras=None):
    buf = _Buf()
    gl = {'asset': {'version': '2.0', 'generator': 'altes_rathaus/ar_glb.py'}, 'scenes': [{'nodes': []}], 'scene': 0,
          'nodes': [], 'meshes': [], 'materials': [], 'textures': [], 'images': [], 'samplers': [
              {'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]}
    used_ext = set()
    mat_index = {}; img_index = {}

    def image(fn):
        if fn in img_index: return img_index[fn]
        data = open(os.path.join(tex_dir, fn), 'rb').read()
        vi = buf.add_view(data)
        gl['images'].append({'bufferView': vi, 'mimeType': 'image/png' if fn.endswith('.png') else 'image/jpeg', 'name': fn})
        gl['textures'].append({'sampler': 0, 'source': len(gl['images']) - 1})
        img_index[fn] = len(gl['textures']) - 1
        return img_index[fn]

    def material(name):
        if name in mat_index: return mat_index[name]
        m = materials.get(name, {'color': (0.8, 0.8, 0.8)})
        lin = m.get('lin', False)              # Farben schon linear (Paletten der Fassaden-Pipeline)
        col = list(np.asarray(m.get('color', (1, 1, 1)), float) if lin else _srgb2lin(m.get('color', (1, 1, 1)))) + [float(m.get('alpha', 1.0))]
        pbr = {'baseColorFactor': [float(x) for x in col], 'metallicFactor': float(m.get('metal', 0.0)),
               'roughnessFactor': float(m.get('rough', 0.8))}
        md = {'name': name, 'pbrMetallicRoughness': pbr}
        t = m.get('tex')
        if t:
            pbr['baseColorTexture'] = {'index': image(t + '_col.png')}
            if os.path.exists(os.path.join(tex_dir, t + '_orm.png')):
                ti = image(t + '_orm.png')
                pbr['metallicRoughnessTexture'] = {'index': ti}
                md['occlusionTexture'] = {'index': ti, 'strength': float(m.get('ao', 0.8))}
            if os.path.exists(os.path.join(tex_dir, t + '_nrm.png')):
                md['normalTexture'] = {'index': image(t + '_nrm.png'), 'scale': float(m.get('nscale', 1.0))}
        if m.get('emit') is not None and t and m.get('emit_tex'):
            md['emissiveTexture'] = {'index': image(t + '_col.png')}
        if m.get('emit') is not None:
            md['emissiveFactor'] = [float(x) for x in (np.asarray(m['emit'], float) if lin else _srgb2lin(m['emit']))]
        am = m.get('alpha_mode') or ('BLEND' if m.get('alpha', 1.0) < 1.0 else None)
        if am:
            md['alphaMode'] = am
            if am == 'MASK': md['alphaCutoff'] = float(m.get('cutoff', 0.5))
        if m.get('double'): md['doubleSided'] = True
        gl['materials'].append(md)
        mat_index[name] = len(gl['materials']) - 1
        return mat_index[name]

    for nd in nodes:
        prims = []
        for pr in nd['prims']:
            mat, P, N, UV, I = pr[:5]
            C = pr[5] if len(pr) > 5 else None
            if not len(I): continue
            pa = buf.add_acc(P.astype(np.float32), 5126, 'VEC3', 34962, minmax=True)
            na = buf.add_acc(N.astype(np.float32), 5126, 'VEC3', 34962)
            ua = buf.add_acc(UV.astype(np.float32), 5126, 'VEC2', 34962)
            idx = I.reshape(-1).astype(np.uint32 if P.shape[0] > 65535 else np.uint16)
            ia = buf.add_acc(idx, 5125 if idx.dtype == np.uint32 else 5123, 'SCALAR', 34963)
            attrs = {'POSITION': pa, 'NORMAL': na, 'TEXCOORD_0': ua}
            if C is not None and materials.get(mat, {}).get('vc'):
                attrs['COLOR_0'] = buf.add_acc(np.clip(C, 0, 1).astype(np.float32), 5126, 'VEC3', 34962)
            prims.append({'attributes': attrs, 'indices': ia, 'material': material(mat), 'mode': 4})
        if not prims: continue
        gl['meshes'].append({'name': nd['name'], 'primitives': prims})
        node = {'name': nd['name'], 'mesh': len(gl['meshes']) - 1}
        if nd.get('extras'): node['extras'] = nd['extras']
        if nd.get('inst') is not None:
            T, R, S = nd['inst']
            node['extensions'] = {'EXT_mesh_gpu_instancing': {'attributes': {
                'TRANSLATION': buf.add_acc(np.asarray(T, np.float32), 5126, 'VEC3'),
                'ROTATION': buf.add_acc(np.asarray(R, np.float32), 5126, 'VEC4'),
                'SCALE': buf.add_acc(np.asarray(S, np.float32), 5126, 'VEC3')}}}
            used_ext.add('EXT_mesh_gpu_instancing')
        gl['nodes'].append(node)
        gl['scenes'][0]['nodes'].append(len(gl['nodes']) - 1)
    if extras: gl['scenes'][0]['extras'] = extras
    if used_ext: gl['extensionsUsed'] = sorted(used_ext)
    if not gl['textures']:
        for k in ('textures', 'images', 'samplers'): gl.pop(k)
    gl['bufferViews'] = buf.views; gl['accessors'] = buf.accessors
    bin_ = b''.join(buf.chunks); bin_ += b'\x00' * ((-len(bin_)) % 4)
    gl['buffers'] = [{'byteLength': len(bin_)}]
    js = json.dumps(gl, separators=(',', ':')).encode('utf8'); js += b' ' * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<I4s', len(js), b'JSON')); f.write(js)
        f.write(struct.pack('<I4s', len(bin_), b'BIN\x00')); f.write(bin_)
    os.replace(tmp, path)
    return total


def mat_to_trs(M):
    """4x4 (Rotation*Skalierung ohne Scherung) -> T, quat xyzw, S"""
    M = np.asarray(M, float)
    T = M[:3, 3]
    A = M[:3, :3]
    S = np.linalg.norm(A, axis=0)
    R = A / S
    if np.linalg.det(R) < 0:
        S[0] = -S[0]; R[:, 0] = -R[:, 0]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w]); q /= np.linalg.norm(q)
    return T, q, S

"""GLB-Schreiber (glTF 2.0, PBR mit Texturen, optional EXT_mesh_gpu_instancing). Atomar (tmp + os.replace).
nodes: [{'name', 'mb': MB | 'arrays': {mat:(P,N,UV,I)}, 'inst': None | list of 4x4, 'extras': {}}]
materials: {name: {'color': sRGB (r,g,b), 'metal', 'rough', 'tex': prefix|None, 'nscale', 'emit', 'alpha', 'double'}}"""
import json, os, struct
import numpy as np


def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class _Buf:
    def __init__(self):
        self.chunks = []; self.len = 0; self.views = []; self.acc = []

    def view(self, data, target=None):
        pad = (-self.len) % 4
        if pad: self.chunks.append(b'\0' * pad); self.len += pad
        v = {'buffer': 0, 'byteOffset': self.len, 'byteLength': len(data)}
        if target: v['target'] = target
        self.chunks.append(data); self.len += len(data); self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr, ctype, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = {'bufferView': self.view(arr.tobytes(), target), 'componentType': ctype, 'count': int(arr.shape[0]), 'type': typ}
        if minmax:
            a['min'] = [float(x) for x in arr.min(0)]; a['max'] = [float(x) for x in arr.max(0)]
        self.acc.append(a)
        return len(self.acc) - 1


def mat_to_trs(M):
    M = np.asarray(M, float); T = M[:3, 3]; A = M[:3, :3]
    S = np.linalg.norm(A, axis=0); R = A / S
    if np.linalg.det(R) < 0: S[0] = -S[0]; R[:, 0] = -R[:, 0]
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


def write_glb(path, nodes, materials, tex_dir, extras=None, generator='leibnizhaus/lh_glb.py'):
    buf = _Buf()
    gl = {'asset': {'version': '2.0', 'generator': generator}, 'scene': 0, 'scenes': [{'nodes': []}], 'nodes': [], 'meshes': [],
          'materials': [], 'textures': [], 'images': [], 'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]}
    used = set(); mi = {}; ti = {}

    def tex(fn):
        if fn in ti: return ti[fn]
        data = open(os.path.join(tex_dir, fn), 'rb').read()
        gl['images'].append({'bufferView': buf.view(data), 'mimeType': 'image/png' if fn.endswith('.png') else 'image/jpeg', 'name': fn})
        gl['textures'].append({'sampler': 0, 'source': len(gl['images']) - 1})
        ti[fn] = len(gl['textures']) - 1
        return ti[fn]

    def material(name):
        if name in mi: return mi[name]
        m = materials.get(name)
        if m is None:
            print('WARN material missing:', name); m = {'color': (0.8, 0.8, 0.8)}
        col = list(srgb2lin(m.get('color', (1, 1, 1)))) + [float(m.get('alpha', 1.0))]
        pbr = {'baseColorFactor': [float(x) for x in col], 'metallicFactor': float(m.get('metal', 0.0)), 'roughnessFactor': float(m.get('rough', 0.8))}
        md = {'name': name, 'pbrMetallicRoughness': pbr}
        t = m.get('tex')
        if t:
            pbr['baseColorTexture'] = {'index': tex(t + '_col.png')}
            if os.path.exists(os.path.join(tex_dir, t + '_orm.png')):
                k = tex(t + '_orm.png')
                pbr['metallicRoughnessTexture'] = {'index': k}
                md['occlusionTexture'] = {'index': k, 'strength': float(m.get('ao', 1.0))}
            if os.path.exists(os.path.join(tex_dir, t + '_nrm.png')):
                md['normalTexture'] = {'index': tex(t + '_nrm.png'), 'scale': float(m.get('nscale', 1.0))}
        if m.get('emit') is not None:
            md['emissiveFactor'] = [float(x) for x in srgb2lin(m['emit'])]
        if m.get('alpha', 1.0) < 1.0: md['alphaMode'] = 'BLEND'
        if m.get('double'): md['doubleSided'] = True
        gl['materials'].append(md); mi[name] = len(gl['materials']) - 1
        return mi[name]

    ntri = 0; ntri_inst = 0
    for nd in nodes:
        arrs = nd['arrays'] if 'arrays' in nd else nd['mb'].arrays()
        prims = []
        ninst = len(nd['inst']) if nd.get('inst') is not None else 1
        for mat in sorted(arrs):
            P, N, UV, I = arrs[mat]
            if not len(I): continue
            pa = buf.accessor(P.astype(np.float32), 5126, 'VEC3', 34962, minmax=True)
            na = buf.accessor(N.astype(np.float32), 5126, 'VEC3', 34962)
            ua = buf.accessor(UV.astype(np.float32), 5126, 'VEC2', 34962)
            idx = I.reshape(-1).astype(np.uint32 if len(P) > 65535 else np.uint16)
            ia = buf.accessor(idx, 5125 if idx.dtype == np.uint32 else 5123, 'SCALAR', 34963)
            prims.append({'attributes': {'POSITION': pa, 'NORMAL': na, 'TEXCOORD_0': ua}, 'indices': ia, 'material': material(mat), 'mode': 4})
            ntri += len(I); ntri_inst += len(I) * ninst
        if not prims: continue
        gl['meshes'].append({'name': nd['name'], 'primitives': prims})
        node = {'name': nd['name'], 'mesh': len(gl['meshes']) - 1}
        if nd.get('extras'): node['extras'] = nd['extras']
        if nd.get('inst') is not None:
            TRS = [mat_to_trs(M) for M in nd['inst']]
            node['extensions'] = {'EXT_mesh_gpu_instancing': {'attributes': {
                'TRANSLATION': buf.accessor(np.array([t[0] for t in TRS], np.float32), 5126, 'VEC3'),
                'ROTATION': buf.accessor(np.array([t[1] for t in TRS], np.float32), 5126, 'VEC4'),
                'SCALE': buf.accessor(np.array([t[2] for t in TRS], np.float32), 5126, 'VEC3')}}}
            used.add('EXT_mesh_gpu_instancing')
        gl['nodes'].append(node); gl['scenes'][0]['nodes'].append(len(gl['nodes']) - 1)
    if extras: gl['scenes'][0]['extras'] = extras
    if used: gl['extensionsUsed'] = sorted(used)
    if not gl['textures']:
        for k in ('textures', 'images', 'samplers'): gl.pop(k)
    gl['bufferViews'] = buf.views; gl['accessors'] = buf.acc
    bin_ = b''.join(buf.chunks); bin_ += b'\0' * ((-len(bin_)) % 4)
    gl['buffers'] = [{'byteLength': len(bin_)}]
    js = json.dumps(gl, separators=(',', ':')).encode('utf8'); js += b' ' * ((-len(js)) % 4)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(bin_)))
        f.write(struct.pack('<I4s', len(js), b'JSON')); f.write(js)
        f.write(struct.pack('<I4s', len(bin_), b'BIN\0')); f.write(bin_)
    os.replace(tmp, path)
    return ntri, ntri_inst

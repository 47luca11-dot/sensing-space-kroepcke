"""Minimaler glTF-2.0/GLB-Schreiber (numpy) mit Materialien, externen/eingebetteten Texturen
und EXT_mesh_gpu_instancing."""
import json, struct
import numpy as np

class GLB:
    def __init__(self):
        self.j = {'asset': {'version': '2.0', 'generator': 'hannover ground pipeline'}, 'scenes': [{'nodes': []}], 'scene': 0,
                  'nodes': [], 'meshes': [], 'accessors': [], 'bufferViews': [], 'buffers': [], 'materials': [],
                  'textures': [], 'images': [], 'samplers': []}
        self.bin = bytearray(); self.mat_idx = {}; self.img_idx = {}; self.ext = set()

    # ---------------------------------------------------------------- Puffer
    def _view(self, data, target=None):
        while len(self.bin) % 4: self.bin += b'\0'
        off = len(self.bin); self.bin += data
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target: v['target'] = target
        self.j['bufferViews'].append(v); return len(self.j['bufferViews']) - 1

    def accessor(self, arr, kind, target=None, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        ct = {np.dtype('float32'): 5126, np.dtype('uint32'): 5125, np.dtype('uint16'): 5123, np.dtype('uint8'): 5121, np.dtype('int16'): 5122}[arr.dtype]
        bv = self._view(arr.tobytes(), target)
        a = {'bufferView': bv, 'componentType': ct, 'count': int(arr.shape[0]), 'type': kind}
        if normalized: a['normalized'] = True
        if minmax:
            a['min'] = [float(x) for x in np.atleast_1d(arr.min(0))]; a['max'] = [float(x) for x in np.atleast_1d(arr.max(0))]
        self.j['accessors'].append(a); return len(self.j['accessors']) - 1

    # ---------------------------------------------------------------- Materialien
    def image(self, uri=None, data=None, mime='image/png'):
        key = uri or id(data)
        if key in self.img_idx: return self.img_idx[key]
        if uri is not None: im = {'uri': uri}
        else: im = {'bufferView': self._view(data), 'mimeType': mime}
        self.j['images'].append(im)
        if not self.j['samplers']: self.j['samplers'].append({'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497})
        self.j['textures'].append({'source': len(self.j['images']) - 1, 'sampler': 0})
        self.img_idx[key] = len(self.j['textures']) - 1
        return self.img_idx[key]

    def material(self, name, spec):
        if name in self.mat_idx: return self.mat_idx[name]
        pbr = {'baseColorFactor': list(spec.get('color', (1, 1, 1, 1))), 'metallicFactor': float(spec.get('metal', 0.0)),
               'roughnessFactor': float(spec.get('rough', 0.9))}
        if len(pbr['baseColorFactor']) == 3: pbr['baseColorFactor'].append(1.0)
        if spec.get('tex'):
            t = self.image(uri=spec['tex']) if not spec.get('embed') else self.image(data=spec['embed'], mime=spec.get('mime', 'image/png'))
            pbr['baseColorTexture'] = {'index': t}
        m = {'name': name, 'pbrMetallicRoughness': pbr}
        if spec.get('alpha') == 'MASK': m['alphaMode'] = 'MASK'; m['alphaCutoff'] = spec.get('cutoff', 0.5)
        elif spec.get('alpha') == 'BLEND': m['alphaMode'] = 'BLEND'
        if spec.get('double'): m['doubleSided'] = True
        if spec.get('emissive'): m['emissiveFactor'] = list(spec['emissive'])
        if spec.get('emissive_tex') and spec.get('tex'):
            m['emissiveTexture'] = {'index': pbr['baseColorTexture']['index']}
            m.setdefault('emissiveFactor', list(spec.get('emissive', (1.0, 1.0, 1.0))))
        if spec.get('extras'): m['extras'] = spec['extras']
        self.j['materials'].append(m); self.mat_idx[name] = len(self.j['materials']) - 1
        return self.mat_idx[name]

    # ---------------------------------------------------------------- Geometrie
    def mesh(self, name, prims, materials):
        """prims: Liste von dicts {pos (n,3), nrm (n,3)|None, uv (n,2)|None, idx (m,3), mat (name)}"""
        P = []
        for p in prims:
            if len(p['idx']) == 0: continue
            attrs = {'POSITION': self.accessor(p['pos'].astype(np.float32), 'VEC3', 34962, minmax=True)}
            if p.get('nrm') is not None: attrs['NORMAL'] = self.accessor(p['nrm'].astype(np.float32), 'VEC3', 34962)
            if p.get('uv') is not None: attrs['TEXCOORD_0'] = self.accessor(p['uv'].astype(np.float32), 'VEC2', 34962)
            n = len(p['pos'])
            idx = p['idx'].astype(np.uint16 if n < 65536 else np.uint32).ravel()
            prim = {'attributes': attrs, 'indices': self.accessor(idx, 'SCALAR', 34963), 'mode': 4,
                    'material': self.material(p['mat'], materials[p['mat']])}
            P.append(prim)
        if not P: return None
        self.j['meshes'].append({'name': name, 'primitives': P}); return len(self.j['meshes']) - 1

    def node(self, name, mesh=None, translation=None, instances=None, extras=None, parent=None):
        n = {'name': name}
        if mesh is not None: n['mesh'] = mesh
        if translation is not None: n['translation'] = [float(x) for x in translation]
        if extras: n['extras'] = extras
        if instances is not None:
            T, R, S = instances
            attrs = {'TRANSLATION': self.accessor(T.astype(np.float32), 'VEC3')}
            if R is not None: attrs['ROTATION'] = self.accessor(R.astype(np.float32), 'VEC4')
            if S is not None: attrs['SCALE'] = self.accessor(S.astype(np.float32), 'VEC3')
            n['extensions'] = {'EXT_mesh_gpu_instancing': {'attributes': attrs}}
            self.ext.add('EXT_mesh_gpu_instancing')
        self.j['nodes'].append(n); i = len(self.j['nodes']) - 1
        if parent is None: self.j['scenes'][0]['nodes'].append(i)
        else: self.j['nodes'][parent].setdefault('children', []).append(i)
        return i

    def save(self, path):
        while len(self.bin) % 4: self.bin += b'\0'
        self.j['buffers'] = [{'byteLength': len(self.bin)}]
        for k in ('textures', 'images', 'samplers', 'materials'):
            if not self.j[k]: del self.j[k]
        if self.ext: self.j['extensionsUsed'] = sorted(self.ext)
        js = json.dumps(self.j, separators=(',', ':')).encode()
        while len(js) % 4: js += b' '
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))
            f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
            f.write(struct.pack('<II', len(self.bin), 0x004E4942)); f.write(bytes(self.bin))
        return total


def quat_y(angle):
    """Quaternion (x,y,z,w) fuer Drehung um +Y"""
    a = np.asarray(angle) / 2
    return np.stack([np.zeros_like(a), np.sin(a), np.zeros_like(a), np.cos(a)], -1)

"""GLB-Schreiber: gemeinsame Materialpalette, Welt-UVs je Dreieck, Verschweissen, KHR_mesh_quantization (Positionen int16),
EXT_mesh_gpu_instancing (TRS + _COLOR_0), optional EXT_meshopt_compression (meshopt-Codec v0).
Keine Normalen: glTF-Clients berechnen flache Normalen (three.js: flatShading)."""
import json, struct, os
import numpy as np

MATERIALS = {
    'plaster': dict(tex='plaster', period=(3.0, 3.0), rough=0.92, metal=0.0),
    'stone': dict(tex='stone', period=(2.4, 2.4), rough=0.85, metal=0.0),
    'brick': dict(tex='brick', period=(1.56, 1.54), rough=0.9, metal=0.0),
    'brick_y': dict(tex='brick_y', period=(1.56, 1.54), rough=0.9, metal=0.0),        # gelber Verblendziegel (Noltehaus)
    'rubble': dict(tex='rubble', period=(3.0, 3.0), rough=0.95, metal=0.0),           # Bruchstein (Beginenturm, Zeughaus)
    'panel': dict(tex='panel', period=(2.4, 2.4), rough=0.7, metal=0.0),
    'trim': dict(rough=0.5, metal=0.0),
    'metal': dict(rough=0.38, metal=0.7),
    'spandrel': dict(rough=0.12, metal=0.35),
    'interior': dict(rough=0.9, metal=0.0),
    'shop_interior': dict(rough=0.85, metal=0.0, emissive=(0.16, 0.155, 0.145)),     # beleuchteter Ladenraum hinter dem Glas
    'glass': dict(rough=0.04, metal=0.0, alpha=0.58, base=(0.05, 0.065, 0.075)),
    'roof_tile': dict(tex='tiles', period=(2.4, 2.8), rough=0.8, metal=0.0),
    'roof_slate': dict(tex='slate', period=(2.5, 2.0), rough=0.6, metal=0.0),
    'roof_metal': dict(tex='seam', period=(2.0, 2.0), rough=0.42, metal=0.55),
    'roof_gravel': dict(tex='gravel', period=(2.0, 2.0), rough=1.0, metal=0.0),
    'roof_membrane': dict(tex='membrane', period=(4.0, 4.0), rough=0.85, metal=0.0),
    'roof_green': dict(tex='green', period=(3.0, 3.0), rough=1.0, metal=0.0),
    # selbstleuchtend (Deckenleuchten, Leuchtkaesten, Leuchtbuchstaben); Farbe ueber emissiveFactor, Vertexfarbe moduliert nur Basisfarbe
    'glow': dict(rough=0.6, metal=0.0, base=(0.8, 0.8, 0.78), emissive=(0.95, 0.93, 0.86)),
    'glow_warm': dict(rough=0.6, metal=0.0, base=(0.8, 0.7, 0.55), emissive=(1.0, 0.78, 0.48)),
}
MAT_ORDER = list(MATERIALS)


def glow_mat(hexcol, strength=0.85):
    """Selbstleuchtendes Material in (quantisierter) Markenfarbe -> Materialname (wird bei Bedarf registriert)."""
    from common import hex2lin
    h = hexcol.lstrip('#')
    q = [min(255, int(round(int(h[i:i + 2], 16) / 17.0)) * 17) for i in (0, 2, 4)]
    name = 'glow_%02X%02X%02X' % tuple(q)
    if name not in MATERIALS:
        lin = hex2lin('#%02X%02X%02X' % tuple(q))
        MATERIALS[name] = dict(rough=0.45, metal=0.0, base=tuple(float(x) * 0.45 for x in lin),
                               emissive=tuple(float(min(1.0, x * strength)) for x in lin))
    return name
UP = np.array([0.0, 1.0, 0.0])


def tri_uvs(V, F, period):
    """Planare Projektion je Dreieck (u = horizontal in der Flaeche, v = in der Flaeche nach oben). -> (3M,3), (3M,2)"""
    T = V[F]                                     # (M,3,3)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1; n /= ln
    t = np.cross(np.broadcast_to(UP, n.shape), n)
    lt = np.linalg.norm(t, axis=1, keepdims=True)
    flat = lt[:, 0] < 1e-3
    t[flat] = [1.0, 0, 0]; lt[flat] = 1.0
    t /= lt
    b = np.cross(n, t)
    P = T.reshape(-1, 3)
    tt = np.repeat(t, 3, 0); bb = np.repeat(b, 3, 0)
    uv = np.c_[(P * tt).sum(1) / period[0], -(P * bb).sum(1) / period[1]]
    return P, uv


def weld(P, C, UV=None, F=None):
    """Gleiche Ecken zusammenfassen. P (N,3) float, C (N,3) float, UV optional. F: Dreiecke (falls schon indiziert)."""
    if F is None:
        F = np.arange(len(P)).reshape(-1, 3)
    key = [np.round(P * 1e4).astype(np.int64), np.round(C * 255).astype(np.int64)]
    if UV is not None:
        key.append(np.round(UV * 2e4).astype(np.int64))
    K = np.concatenate(key, 1)
    _, first, inv = np.unique(K, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    P2 = P[first]; C2 = C[first]; UV2 = UV[first] if UV is not None else None
    F2 = inv[F]
    good = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return P2, C2, UV2, F2[good]


def build_primitive(parts, mat, local=False):
    """parts: Liste (V,F,C) -> dict pos, col, uv, idx (geschweisst)."""
    V = np.concatenate([p[0] for p in parts]).astype(np.float64)
    off = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    F = np.concatenate([p[1] + o for p, o in zip(parts, off)]).astype(np.int64)
    C = np.concatenate([np.asarray(p[2], np.float64) for p in parts])
    C = np.clip(C, 0, 1)
    m = MATERIALS[mat]
    if 'tex' in m:
        P, uv = tri_uvs(V, F, m['period'])
        Cc = np.repeat(C[F.reshape(-1)].reshape(-1, 3, 3), 1, 0).reshape(-1, 3)
        P2, C2, UV2, F2 = weld(P, Cc, uv)
    else:
        P2, C2, UV2, F2 = weld(V, C, None, F)
    return {'pos': P2, 'col': C2, 'uv': UV2, 'idx': F2.astype(np.uint32)}


class GLB:
    def __init__(self, compress=False, tex_prefix='textures/'):
        self.compress = compress
        self.J = {'asset': {'version': '2.0', 'generator': 'hannover facades pipeline'}, 'scenes': [{'nodes': []}], 'scene': 0,
                  'nodes': [], 'meshes': [], 'materials': [], 'accessors': [], 'bufferViews': [], 'buffers': []}
        self.bin = bytearray()          # unkomprimierte Daten (Rueckfall-/Rohpuffer)
        self.cbin = bytearray()         # komprimierte Daten
        self.used_ext = set(); self.req_ext = set()
        self.mat_idx = {}; self.tex_prefix = tex_prefix
        self.images = {}

    # ---------------------------------------------------------------- Materialien
    def material(self, name):
        if name in self.mat_idx:
            return self.mat_idx[name]
        m = MATERIALS[name]
        base = list(m.get('base', (1.0, 1.0, 1.0))) + [m.get('alpha', 1.0)]
        md = {'name': 'bldg_' + name, 'pbrMetallicRoughness': {'baseColorFactor': [round(x, 4) for x in base],
                                                               'metallicFactor': m['metal'], 'roughnessFactor': m['rough']}}
        if 'alpha' in m:
            md['alphaMode'] = 'BLEND'
        if 'emissive' in m:
            md['emissiveFactor'] = [round(float(x), 4) for x in m['emissive']]
        if 'tex' in m:
            md['pbrMetallicRoughness']['baseColorTexture'] = {'index': self._texture(m['tex'])}
        self.J['materials'].append(md)
        self.mat_idx[name] = len(self.J['materials']) - 1
        return self.mat_idx[name]

    def _texture(self, tex):
        if tex in self.images:
            return self.images[tex]
        J = self.J
        J.setdefault('images', []).append({'uri': self.tex_prefix + tex + '.jpg', 'name': tex})
        J.setdefault('samplers', [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}])
        J.setdefault('textures', []).append({'source': len(J['images']) - 1, 'sampler': 0, 'name': tex})
        self.images[tex] = len(J['textures']) - 1
        return self.images[tex]

    # ---------------------------------------------------------------- Puffer
    def _view(self, data, stride=None, target=None, mode=None, count=None):
        data = np.ascontiguousarray(data)
        raw = data.tobytes()
        while len(self.bin) % 16:
            self.bin += b'\0'
        bv = {'buffer': 0, 'byteOffset': len(self.bin), 'byteLength': len(raw)}
        if stride:
            bv['byteStride'] = stride
        if target:
            bv['target'] = target
        self.bin += raw
        if self.compress and mode is not None:
            import meshopt
            if mode == 'TRIANGLES':
                enc = meshopt.encode_index_buffer(data.reshape(-1).astype(np.uint32))
                es = data.dtype.itemsize
            else:
                rows = data.reshape(count, -1).view(np.uint8).reshape(count, -1)
                enc = meshopt.encode_vertex_buffer_level(rows, 2, 0)
                es = rows.shape[1]
            while len(self.cbin) % 4:
                self.cbin += b'\0'
            ext = {'buffer': 0, 'byteOffset': len(self.cbin), 'byteLength': len(enc), 'byteStride': es, 'count': count, 'mode': mode}
            self.cbin += enc
            bv['buffer'] = 1
            bv['extensions'] = {'EXT_meshopt_compression': ext}
        self.J['bufferViews'].append(bv)
        return len(self.J['bufferViews']) - 1

    def _acc(self, view, ctype, count, typ, normalized=False, mn=None, mx=None):
        a = {'bufferView': view, 'componentType': ctype, 'count': int(count), 'type': typ}
        if normalized:
            a['normalized'] = True
        if mn is not None:
            if ctype == 5126:
                a['min'] = [float(x) for x in mn]; a['max'] = [float(x) for x in mx]
            else:
                a['min'] = [int(x) for x in mn]; a['max'] = [int(x) for x in mx]
        self.J['accessors'].append(a)
        return len(self.J['accessors']) - 1

    # ---------------------------------------------------------------- Meshes
    def mesh(self, name, prims, qc=None, qs=None):
        """prims: [(matname, dict pos,col,uv,idx)]; qc/qs: Quantisierungs-Zentrum/-Halbausdehnung (int16 normalisiert)."""
        out = []
        for mat, pr in prims:
            pos, col, uv, idx = pr['pos'], pr['col'], pr['uv'], pr['idx']
            if len(idx) == 0:
                continue
            idx = np.ascontiguousarray(idx.reshape(-1).astype(np.uint32))
            try:
                import meshopt
                idx2, remap, unique = meshopt.reorder_mesh(idx, True, False)
                order = np.empty(unique, np.int64); used = remap < unique
                order[remap[used]] = np.nonzero(used)[0]
                pos, col = pos[order], col[order]
                uv = uv[order] if uv is not None else None
                idx = idx2.astype(np.uint32)
            except Exception:
                pass
            n = len(pos)
            attrs = {}
            if qc is not None:
                q = np.round((pos - qc) / qs * 32767.0).clip(-32767, 32767).astype(np.int16)
                q4 = np.zeros((n, 4), np.int16); q4[:, :3] = q
                v = self._view(q4, 8, 34962, 'ATTRIBUTES', n)
                # glTF/three.js: min/max normalisierter Accessoren in Ganzzahl-Einheiten (sonst falsche Bounding-Spheres)
                a = self._acc(v, 5122, n, 'VEC3', True, q.min(0).astype(int), q.max(0).astype(int))
                self.used_ext.add('KHR_mesh_quantization'); self.req_ext.add('KHR_mesh_quantization')
            else:
                p32 = pos.astype(np.float32)
                v = self._view(p32, 12, 34962, 'ATTRIBUTES', n)
                a = self._acc(v, 5126, n, 'VEC3', False, p32.min(0), p32.max(0))
            attrs['POSITION'] = a
            c8 = np.full((n, 4), 255, np.uint8); c8[:, :3] = np.round(np.clip(col, 0, 1) * 255)
            attrs['COLOR_0'] = self._acc(self._view(c8, 4, 34962, 'ATTRIBUTES', n), 5121, n, 'VEC4', True)
            if uv is not None:
                u32 = uv.astype(np.float32)
                attrs['TEXCOORD_0'] = self._acc(self._view(u32, 8, 34962, 'ATTRIBUTES', n), 5126, n, 'VEC2')
            if n < 65536:
                ib = idx.astype(np.uint16)
                iv = self._view(ib, None, 34963, 'TRIANGLES', len(ib))
                ia = self._acc(iv, 5123, len(ib), 'SCALAR')
            else:
                iv = self._view(idx, None, 34963, 'TRIANGLES', len(idx))
                ia = self._acc(iv, 5125, len(idx), 'SCALAR')
            out.append({'attributes': attrs, 'indices': ia, 'material': self.material(mat), 'mode': 4})
        if not out:
            return None
        self.J['meshes'].append({'name': name, 'primitives': out})
        return len(self.J['meshes']) - 1

    def node(self, name, mesh=None, translation=None, scale=None, children=None, extras=None, root=False):
        nd = {'name': name}
        if mesh is not None:
            nd['mesh'] = mesh
        if translation is not None:
            nd['translation'] = [float(x) for x in translation]
        if scale is not None:
            nd['scale'] = [float(x) for x in scale]
        if children:
            nd['children'] = children
        if extras:
            nd['extras'] = extras
        self.J['nodes'].append(nd)
        i = len(self.J['nodes']) - 1
        if root:
            self.J['scenes'][0]['nodes'].append(i)
        return i

    def instanced(self, name, mesh, T, Q, S, C):
        n = len(T)
        T = np.asarray(T, np.float32); Q = np.asarray(Q, np.float32); S = np.asarray(S, np.float32); C = np.asarray(C, np.float32)
        at = {'TRANSLATION': self._acc(self._view(T, 12, None, 'ATTRIBUTES', n), 5126, n, 'VEC3', False, T.min(0), T.max(0)),
              'ROTATION': self._acc(self._view(Q, 16, None, 'ATTRIBUTES', n), 5126, n, 'VEC4'),
              'SCALE': self._acc(self._view(S, 12, None, 'ATTRIBUTES', n), 5126, n, 'VEC3'),
              '_COLOR_0': self._acc(self._view(np.clip(C, 0, 1), 12, None, 'ATTRIBUTES', n), 5126, n, 'VEC3')}
        nd = {'name': name, 'mesh': mesh, 'extensions': {'EXT_mesh_gpu_instancing': {'attributes': at}}}
        self.used_ext.add('EXT_mesh_gpu_instancing'); self.req_ext.add('EXT_mesh_gpu_instancing')
        self.J['nodes'].append(nd)
        return len(self.J['nodes']) - 1

    # ---------------------------------------------------------------- Schreiben
    def write(self, path):
        J = self.J
        if self.compress:
            while len(self.cbin) % 4:
                self.cbin += b'\0'
            while len(self.bin) % 4:
                self.bin += b'\0'
            J['buffers'] = [{'byteLength': len(self.cbin)},
                            {'byteLength': len(self.bin), 'extensions': {'EXT_meshopt_compression': {'fallback': True}}}]
            # bufferViews ohne Kompression (falls vorhanden) zeigen auf Puffer 0 -> in cbin kopieren
            for bv in J['bufferViews']:
                if bv['buffer'] == 0:
                    raw = bytes(self.bin[bv['byteOffset']:bv['byteOffset'] + bv['byteLength']])
                    while len(self.cbin) % 16:
                        self.cbin += b'\0'
                    bv['byteOffset'] = len(self.cbin); self.cbin += raw
            J['buffers'][0]['byteLength'] = len(self.cbin)
            self.used_ext.add('EXT_meshopt_compression'); self.req_ext.add('EXT_meshopt_compression')
            data = self.cbin
        else:
            while len(self.bin) % 4:
                self.bin += b'\0'
            J['buffers'] = [{'byteLength': len(self.bin)}]
            data = self.bin
        if self.used_ext:
            J['extensionsUsed'] = sorted(self.used_ext)
        if self.req_ext:
            J['extensionsRequired'] = sorted(self.req_ext)
        for k in ('materials', 'meshes', 'accessors', 'bufferViews'):
            if not J[k]:
                del J[k]
        js = json.dumps(J, separators=(',', ':')).encode('utf-8')
        while len(js) % 4:
            js += b' '
        total = 12 + 8 + len(js) + 8 + len(data)
        with open(path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))
            f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
            f.write(struct.pack('<II', len(data), 0x004E4942)); f.write(bytes(data))
        return total

"""Minimal GLB reader/writer (numpy only) that keeps node names + materials.

read_glb(path) -> dict(json=..., nodes=[{name, material(dict), prims:[{pos,nrm,uv,idx}]}])
write_glb(path, nodes, extensions_used=None)
  nodes: list of {name, material: dict, pos (N,3) f4, nrm (N,3) f4, uv (N,2) f4 or None,
                  idx (M,3) u4, instances: optional (K, 4,4) matrices -> EXT_mesh_gpu_instancing}
"""
import json
import struct
import numpy as np

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
      5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _acc(j, b, ai):
    a = j["accessors"][ai]
    bv = j["bufferViews"][a["bufferView"]]
    dt = np.dtype(CT[a["componentType"]])
    n = NC[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    cnt = a["count"]
    if stride and stride != dt.itemsize * n:
        raw = np.frombuffer(b, dtype=np.uint8, count=stride * (cnt - 1) + dt.itemsize * n, offset=off)
        out = np.lib.stride_tricks.as_strided(raw, shape=(cnt, dt.itemsize * n), strides=(stride, 1)).copy()
        arr = out.view(dt).reshape(cnt, n)
    else:
        arr = np.frombuffer(b, dtype=dt, count=cnt * n, offset=off).reshape(cnt, n).copy()
    if a.get("normalized"):
        if dt == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif dt == np.int8:
            arr = np.maximum(arr.astype(np.float32) / 127.0, -1)
        elif dt == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif dt == np.int16:
            arr = np.maximum(arr.astype(np.float32) / 32767.0, -1)
    return arr


def read_glb(path):
    d = open(path, "rb").read()
    jl = struct.unpack("<I", d[12:16])[0]
    j = json.loads(d[20:20 + jl])
    o = 20 + jl
    b = b""
    if o < len(d):
        bl = struct.unpack("<I", d[o:o + 4])[0]
        b = d[o + 8:o + 8 + bl]
    nodes = []
    for n in j["nodes"]:
        if "mesh" not in n:
            continue
        m = j["meshes"][n["mesh"]]
        prims = []
        mat = None
        for p in m["primitives"]:
            at = p["attributes"]
            pos = _acc(j, b, at["POSITION"]).astype(np.float64)
            nrm = _acc(j, b, at["NORMAL"]).astype(np.float64) if "NORMAL" in at else None
            uv = _acc(j, b, at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else None
            idx = _acc(j, b, p["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in p else \
                np.arange(len(pos)).reshape(-1, 3)
            mat = j["materials"][p["material"]] if "material" in p else None
            prims.append(dict(pos=pos, nrm=nrm, uv=uv, idx=idx, material=mat))
        if "matrix" in n or "translation" in n or "rotation" in n or "scale" in n:
            raise RuntimeError("node transforms not supported: " + n.get("name", ""))
        nodes.append(dict(name=n.get("name", ""), prims=prims, material=mat))
    return dict(json=j, nodes=nodes)


class _Buf:
    def __init__(self):
        self.parts = []
        self.len = 0
        self.views = []
        self.accs = []

    def add(self, arr, target=None, typ="VEC3", ctype=5126, minmax=False):
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        pad = (-self.len) % 4
        if pad:
            self.parts.append(b"\0" * pad)
            self.len += pad
        bv = dict(buffer=0, byteOffset=self.len, byteLength=len(raw))
        if target:
            bv["target"] = target
        self.views.append(bv)
        self.parts.append(raw)
        self.len += len(raw)
        n = NC[typ]
        acc = dict(bufferView=len(self.views) - 1, componentType=ctype,
                   count=int(arr.size // n), type=typ)
        if minmax:
            a2 = arr.reshape(-1, n)
            acc["min"] = [float(x) for x in a2.min(0)]
            acc["max"] = [float(x) for x in a2.max(0)]
        self.accs.append(acc)
        return len(self.accs) - 1


def _mat_key(m):
    return json.dumps(m, sort_keys=True)


def write_glb(path, nodes, extensions_used=None):
    buf = _Buf()
    materials, mat_idx = [], {}
    meshes, gnodes = [], []
    ext_used = set(extensions_used or [])
    for nd in nodes:
        m = nd["material"]
        k = _mat_key(m)
        if k not in mat_idx:
            mat_idx[k] = len(materials)
            materials.append(m)
            for e in (m.get("extensions") or {}):
                ext_used.add(e)
        pos = np.asarray(nd["pos"], np.float32)
        idx = np.asarray(nd["idx"], np.uint32).reshape(-1)
        if len(idx) == 0:
            continue
        attrs = {"POSITION": buf.add(pos, 34962, "VEC3", 5126, True)}
        if nd.get("nrm") is not None:
            nr = np.asarray(nd["nrm"], np.float32)
            ln = np.linalg.norm(nr, axis=1, keepdims=True)
            ln[ln < 1e-12] = 1
            attrs["NORMAL"] = buf.add(nr / ln, 34962, "VEC3")
        if nd.get("uv") is not None:
            attrs["TEXCOORD_0"] = buf.add(np.asarray(nd["uv"], np.float32), 34962, "VEC2")
        ia = buf.add(idx, 34963, "SCALAR", 5125)
        meshes.append(dict(name=nd["name"], primitives=[dict(attributes=attrs, indices=ia,
                                                             material=mat_idx[k], mode=4)]))
        node = dict(mesh=len(meshes) - 1, name=nd["name"])
        inst = nd.get("instances")
        if inst is not None and len(inst):
            inst = np.asarray(inst, np.float64)
            T = inst[:, :3, 3].astype(np.float32)
            R = inst[:, :3, :3]
            S = np.linalg.norm(R, axis=1)  # column norms
            Rn = R / S[:, None, :]
            q = _mat2quat(Rn).astype(np.float32)
            node["extensions"] = {"EXT_mesh_gpu_instancing": {"attributes": {
                "TRANSLATION": buf.add(T, None, "VEC3"),
                "ROTATION": buf.add(q, None, "VEC4"),
                "SCALE": buf.add(S.astype(np.float32), None, "VEC3")}}}
            ext_used.add("EXT_mesh_gpu_instancing")
        gnodes.append(node)
    j = dict(asset=dict(version="2.0", generator="detail_d0"),
             scene=0, scenes=[dict(nodes=list(range(len(gnodes))))],
             nodes=gnodes, meshes=meshes, materials=materials,
             accessors=buf.accs, bufferViews=buf.views,
             buffers=[dict(byteLength=buf.len)])
    if ext_used:
        j["extensionsUsed"] = sorted(ext_used)
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    bb = b"".join(buf.parts)
    bb += b"\0" * ((-len(bb)) % 4)
    j["buffers"][0]["byteLength"] = len(bb)
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bb)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(bb), 0x004E4942))
        f.write(bb)


def _mat2quat(R):
    # R: (K,3,3) -> (K,4) xyzw
    K = R.shape[0]
    q = np.zeros((K, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(K):
        m = R[i]
        t = tr[i]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q

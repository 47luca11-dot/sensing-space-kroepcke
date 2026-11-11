"""Extract shopfront / door / sign-band placements from the Rhino detail model (read-only) into
cache/rhino_openings.json (game frame).

Rhino XYZ -> game: (x + 84.117, z - 0.16, -(y - 141.706))   (same offset as blender/export_detail.py)

Per block instance (layers Bldg::Openings::Shopfronts / Doors):
  def name, 4x4 game-frame matrix M (local -> game; local x = width, y = into the building, z = up),
  local part boxes (Frame, Glass panes, Interior panel, Sill, Hardware).
Per loose mesh on those layers: material name + game-frame triangles (for custom shopfronts).
Per sign band object: game-frame bbox (+ triangles if a render mesh is cached).
"""
import json
import os
import sys

import numpy as np
import rhino3dm

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'sources' / 'Opernplatz_Kiosk+Umgebung-Detail_3D_v1.3dm')
OUT = os.path.join(HERE, "cache", "rhino_openings.json")
OFF = np.array([84.117, -141.706, -0.16])

# Rhino(x,y,z) -> game(x, y_up, z)
R2G = np.array([[1, 0, 0, OFF[0]],
                [0, 0, 1, OFF[2]],
                [0, -1, 0, -OFF[1]],
                [0, 0, 0, 1.0]])


def xform_np(x):
    return np.array([[x.M00, x.M01, x.M02, x.M03], [x.M10, x.M11, x.M12, x.M13],
                     [x.M20, x.M21, x.M22, x.M23], [x.M30, x.M31, x.M32, x.M33]], float)


def mesh_np(m):
    V = np.array([[m.Vertices[i].X, m.Vertices[i].Y, m.Vertices[i].Z] for i in range(len(m.Vertices))], float)
    F = []
    for i in range(len(m.Faces)):
        a, b, c, d = m.Faces[i]
        F.append((a, b, c))
        if d != c:
            F.append((a, c, d))
    return V, np.array(F, int).reshape(-1, 3)


def to_game(P):
    P = np.asarray(P, float)
    return P @ R2G[:3, :3].T + R2G[:3, 3]


def comps_bboxes(V, F):
    """bounding boxes of connected components (shared vertex index or same rounded position)."""
    key = np.round(V / 1e-4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    parent = list(range(len(F)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    owner = {}
    for fi, f in enumerate(F):
        for v in inv[f]:
            if v in owner:
                ra, rb = find(fi), find(owner[v])
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[v] = fi
    groups = {}
    for fi in range(len(F)):
        groups.setdefault(find(fi), []).append(fi)
    out = []
    for g in groups.values():
        P = V[F[g].reshape(-1)]
        out.append([P.min(0).tolist(), P.max(0).tolist()])
    out.sort(key=lambda b: b[0][0])
    return out


def main():
    f = rhino3dm.File3dm.Read(SRC)
    lay = {i: f.Layers[i].FullPath for i in range(len(f.Layers))}
    lid = {v: k for k, v in lay.items()}
    idefs = {str(d.Id): d for d in f.InstanceDefinitions}

    def matname(o):
        mi = o.Attributes.MaterialIndex
        if 0 <= mi < len(f.Materials):
            return f.Materials[mi].Name
        L = f.Layers[o.Attributes.LayerIndex]
        ri = L.RenderMaterialIndex
        return f.Materials[ri].Name if 0 <= ri < len(f.Materials) else ""

    defcache = {}

    def def_parts(did):
        if did in defcache:
            return defcache[did]
        d = idefs[did]
        parts = {}
        for oid in d.GetObjectIds():
            oo = f.Objects.FindId(oid)
            if oo is None:
                continue
            g = oo.Geometry
            if str(g.ObjectType) != "ObjectType.Mesh":
                bb = g.GetBoundingBox()
                parts.setdefault(lay[oo.Attributes.LayerIndex].split("::")[-1] + "_bbox", []).append(
                    [[bb.Min.X, bb.Min.Y, bb.Min.Z], [bb.Max.X, bb.Max.Y, bb.Max.Z]])
                continue
            V, F = mesh_np(g)
            key = lay[oo.Attributes.LayerIndex].split("::")[-1]
            if key == "Glass":
                parts.setdefault("Glass", []).extend(comps_bboxes(V, F))
            else:
                parts.setdefault(key, []).append([V.min(0).tolist(), V.max(0).tolist()])
        defcache[did] = dict(name=d.Name, parts=parts)
        return defcache[did]

    blocks, loose, bands = [], [], []
    for o in f.Objects:
        li = o.Attributes.LayerIndex
        ln = lay[li]
        g = o.Geometry
        typ = str(g.ObjectType)
        if ln in ("Bldg::Openings::Shopfronts", "Bldg::Openings::Doors"):
            kind = "shop" if ln.endswith("Shopfronts") else "door"
            if typ == "ObjectType.InstanceReference":
                did = str(g.ParentIdefId)
                X = xform_np(g.Xform)
                M = R2G @ X
                dp = def_parts(did)
                blocks.append(dict(layer=kind, defname=dp["name"], M=M.tolist(), parts=dp["parts"],
                                   id=str(o.Attributes.Id)))
            elif typ == "ObjectType.Mesh":
                V, F = mesh_np(g)
                loose.append(dict(layer=kind, mat=matname(o), V=np.round(to_game(V), 4).tolist(), F=F.tolist(),
                                  id=str(o.Attributes.Id)))
        elif ln in ("Bldg::Base::SignBand",):
            bb = g.GetBoundingBox()
            C = np.array([[bb.Min.X, bb.Min.Y, bb.Min.Z], [bb.Max.X, bb.Max.Y, bb.Max.Z]])
            item = dict(mat=matname(o), bbox_rhino=C.tolist(), id=str(o.Attributes.Id), type=typ)
            tris = None
            if typ == "ObjectType.Mesh":
                V, F = mesh_np(g)
                tris = to_game(V)[F]
            elif typ == "ObjectType.Brep":
                T = []
                for fi in range(len(g.Faces)):
                    try:
                        mm = g.Faces[fi].GetMesh(rhino3dm.MeshType.Any)
                    except Exception:
                        mm = None
                    if mm is not None and len(mm.Faces):
                        V, F = mesh_np(mm)
                        T.append(to_game(V)[F])
                if T:
                    tris = np.concatenate(T)
            if tris is not None:
                item["tris"] = np.round(tris, 4).tolist()
            bands.append(item)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(blocks=blocks, loose=loose, bands=bands), open(OUT, "w"))
    print("blocks", len(blocks), "loose", len(loose), "bands", len(bands),
          "bands with tris", sum("tris" in b for b in bands))


if __name__ == "__main__":
    main()

#! python3
# Export des Opernplatz/Kroepcke-Detailmodells: Render-Meshes erzeugen, gruppenweise als GLB (Z->Y) exportieren
import Rhino, System, os, json, time, traceback
OUT = r"C:\Users\luca\Documents\Sensing Space\Claude Code\output\hannover\data\tib_export"
LOG = open(os.path.join(OUT, 'rhino_log.txt'), 'w', encoding='utf-8')
def log(*a):
    LOG.write(' '.join(str(x) for x in a) + '\n'); LOG.flush()
try:
    doc = Rhino.RhinoDoc.ActiveDoc
    log('doc', doc.Path, 'objects', doc.Objects.Count)
    mp = Rhino.Geometry.MeshingParameters.QualityRenderMesh
    mp.MinimumEdgeLength = 0.002; mp.MaximumEdgeLength = 0; mp.GridAngle = 0.35; mp.Tolerance = 0.01
    MT = Rhino.Geometry.MeshType.Render
    def ensure(o):
        try:
            if o.ObjectType in (Rhino.DocObjects.ObjectType.Brep, Rhino.DocObjects.ObjectType.Extrusion, Rhino.DocObjects.ObjectType.SubD, Rhino.DocObjects.ObjectType.Surface):
                if o.MeshCount(MT, None) == 0:
                    o.CreateMeshes(MT, mp, False)
        except Exception as e:
            log('mesh fail', o.Id, e)
    t0 = time.time(); n = 0
    for o in doc.Objects:
        ensure(o); n += 1
    for idef in doc.InstanceDefinitions:
        if idef is None or idef.IsDeleted: continue
        for o in idef.GetObjects(): ensure(o)
    log('meshes ensured', n, round(time.time() - t0, 1), 's')

    def group_of(path):
        p = path.split('::')
        if p[0] != '03_Detailmodell' or len(p) < 2: return None
        if p[1] == 'Umgebung':
            sub = p[2] if len(p) > 2 else 'misc'
            return 'umgebung_' + ''.join(c if c.isalnum() else '_' for c in sub)[:24].lower()
        return 'tib_' + p[1].lower()
    OKT = (Rhino.DocObjects.ObjectType.Brep, Rhino.DocObjects.ObjectType.Extrusion, Rhino.DocObjects.ObjectType.Mesh, Rhino.DocObjects.ObjectType.SubD, Rhino.DocObjects.ObjectType.InstanceReference, Rhino.DocObjects.ObjectType.Surface)
    groups = {}
    for o in doc.Objects:
        if o.ObjectType not in OKT: continue
        L = doc.Layers[o.Attributes.LayerIndex]
        if not L.IsVisible or not o.Visible: continue
        g = group_of(L.FullPath)
        if g: groups.setdefault(g, []).append(o.Id)
    log('groups', {k: len(v) for k, v in groups.items()})
    json.dump({k: len(v) for k, v in groups.items()}, open(os.path.join(OUT, 'groups.json'), 'w'))
    for g, ids in groups.items():
        t0 = time.time()
        doc.Objects.UnselectAll()
        for i in ids: doc.Objects.Select(i, True, True, True)
        fn = os.path.join(OUT, g + '.glb')
        ok = Rhino.RhinoApp.RunScript('_-Export "%s" _Enter _Enter _Enter' % fn, False)
        log('export', g, len(ids), ok, os.path.exists(fn) and os.path.getsize(fn), round(time.time() - t0, 1), 's')
    doc.Objects.UnselectAll()
    log('DONE')
except Exception:
    log(traceback.format_exc())
finally:
    LOG.close()
    try:
        Rhino.RhinoDoc.ActiveDoc.Modified = False
    except Exception: pass
    Rhino.RhinoApp.RunScript('_-Exit', False)

"""Fassaden-Pipeline: LoD2 (Stadtmodell Hannover) -> detaillierte, spielfertige Gebaeude-Kacheln.

Aufruf (alles neu erzeugen):
    python build_all.py                 # Attribute, Vorbereitung, Texturen, Kacheln, Index, Kollision
    python build_all.py --check --preview   # + Strahlen-/Luecken-Pruefung und Vorschaubilder mit Blender (headless)
Optionen: --reprep (Vorbereitung erzwingen), --only ID[,ID] (nur diese Gebaeude, schreibt nach cache/test/),
          --tiles tx_tz[,..], --no-compress
Ausgaben:
    web/assets/city/buildings/bldg_<tx>_<tz>.glb      Huelle + Fassadendetails (meshopt, quantisiert)
    web/assets/city/buildings/bldg_<tx>_<tz>_win.glb  instanzierte Fenster/Tueren/Schaufenster (EXT_mesh_gpu_instancing)
    web/assets/city/buildings/index.json, collision_buildings.glb, textures/*.jpg
    data/buildings_raw/...                              dieselben Dateien unkomprimiert (Blender-lesbar)
Weitere Optionen: --out DIR (Web-Ausgabe umlenken, z. B. web/assets/city/buildings/_dev/<x> fuer die Testansicht),
          --raw-out DIR (Rohdateien umlenken).
Module: shops.py (echte Laeden aus data/shops.json: Wandzuordnung, Schaufenster/Tuer, Ladenraum, Schilder mit echtem Namen),
        displays.py (instanzierte Auslagen je Sortiment, Ladenfront-/Tuer-Prototypen, Deckenleuchten, Aussengastronomie),
        text3d.py (Schriftzuege als extrudierte Geometrie, DejaVu-Schriften), replaced.py (durch Wahrzeichen ersetzte
        LoD2-Gebaeude aus data/landmarks_replace/*.json werden uebersprungen), facade.py (u. a. Fachwerk mit Auskragung,
        Backsteingotik-Giebel), style.py (Altstadt-/Fachwerk-Regeln).
Testansicht im Browser: web/assets/city/buildings/_dev/viewer.html?tiles=..&cam=x,y,z&tgt=x,y,z (nur Entwicklung).
Teilbau ohne das Spiel zu beruehren: --tiles a_b,.. IMMER mit --out web/assets/city/buildings/_dev/<x> --raw-out cache/<x>_raw
(sonst entfernt swap_in alle anderen Kacheln aus dem Spielordner).
Dichtheit (check_gaps.py): LoD2-Eckpunkte werden auf 3 cm verschweisst (prep.py), koplanare Doppelwaende nur bei echter
Ueberdeckung (> 4 cm breit) ausgeschnitten, von earcut verworfene kollineare Randpunkte wieder eingefuegt (geom.tri2d),
Flachdach-Attika auf der hoechsten Teilhoehe, Gaubenwangen/-front enden exakt unter dem Gaubendach.
Vorschau facades2_*: python preview_cams.py --cams2; blender -b --factory-startup -P blender_preview.py -- data/buildings_raw
  --cams cache/preview_cams2.json [--landmarks marktkirche,...]
  (preview_cams --cams2 liest die tatsaechlich gebauten Laeden aus cache/building_info.json -> Kameras je Auslagenart)
Wahrzeichen nachtraeglich: python build_all.py --patch-replaced [--patch-tiles tx_tz,..]  -> baut nur die Kacheln neu, in denen sich die Ersetzungen durch
  data/landmarks_replace/*.json seit dem letzten Lauf geaendert haben (index.json/Kollision/building_info werden zusammengefuehrt,
  uebrige Kacheln bleiben unberuehrt). Danach ggf. python begehung.py (Abschnitt in BEGEHUNGSLISTE.md aktualisieren).
Altstadt: Fachwerk mit Fensterband (65 %) oder Einzelfenstern, Balken dunkelbraun / ochsenblutrot / grau nach landmark_notes.md,
  Gefache meist weiss; Laeden mit Holzfront bekommen teils schmiedeeiserne Ausleger (shops._iron_blade); Kreuzkirche,
  Nikolaikapelle, St. Clemens, Neustaedter Kirche als Naturstein (style.OVERRIDES).
QA-Nachbesserung 2026-11-10: Strassenniveau je Wand aus dem Hoehenfeld (level.py, Laeden/Tueren nicht mehr im Boden),
  coplanar.py (je Kachel: koplanare Ueberdeckungen aufloesen - Details 8 mm vor, Huelle beschneiden, Duplikate entfernen;
  verdeckte Hinterlegungen 'shell_b' bleiben), style.OVERRIDES 'big_signs' (grosse Einzelbuchstaben H&M/Hugendubel/GALERIA),
  'glass_canopy' (Hugendubel), 'glazed_walls', 'tower' (Beginenturm), feste Dachfarben aus Notizen ('roof_fixed'),
  shops.SHOP_FIX (Namens-/Schildkorrekturen), Seitenfenster neben schmalen Ladentueren, Hausnummer je Front nur einmal,
  Fachwerk-Eckfasen mit Eckstaender, index.json: shops_placed_named / shops_placed_unnamed.
"""
import os, sys, json, time, argparse, math, shutil, collections
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, CACHE, WEB_OUT, ROOT, TILE, rng_of, hex2lin, save_json, load_json, jitter_hex
import prep as PREP
from geom import MeshAcc, tri2d, Frame, wall_frame
from inst import InstAcc, get_proto
from style import classify, params, roof_stats, TRAVERTINE
import facade as FA
import roofs as RF
import textures as TX
import shops as SH
import coplanar as CP
from gltf_io import GLB, build_primitive, MATERIALS

RAW_OUT = os.path.join(DATA, 'buildings_raw')


def make_ctx(d):
    streets = d['streets']
    geoms = []
    for s in streets:
        pts = s['pts']
        if s.get('area') and len(pts) >= 4:
            geoms.append(LineString(pts))
        else:
            geoms.append(LineString(pts))
    shops = d['shops']
    # Index aller senkrechten Waende (auch Detailmodell-Gebaeude) fuer koplanare Doppelwaende
    wl = []; wi = []
    for bb in d['bldgs']:
        for k, s in enumerate(bb.walls):
            if abs(s.n[1]) >= 0.05:
                continue
            n = s.n.copy(); n[1] = 0; n /= np.linalg.norm(n)
            wl.append(LineString(s.P[:, [0, 2]]).buffer(0.01)); wi.append((bb.idx, k, n, s.P[0].copy(), s.area, bb.covered))
    return {'wall_tree': STRtree(wl), 'wall_info': wi, 'bldgs': d['bldgs'], 'vol': d['vol'], 'streets': streets, 'street_geoms': geoms, 'street_tree': STRtree(geoms),
            'shops': shops, 'shop_tree': STRtree([Point(s['p']) for s in shops]), 'mask': d['mask'], 'terr': _terrain()}


_TERR = None


def _terrain():
    global _TERR
    if _TERR is None:
        _TERR = PREP.Terrain()
    return _TERR


def gf_material(P, kind, rng):
    """Erdgeschoss-Verkleidung bei Laeden (Pfeiler) bzw. Rustika bei Altbauten."""
    if kind == 'altbau':
        return ('stone', '#EEE6D6' if P['wall_mat'] != 'stone' else '#E4DACA')
    if kind == 'nachkrieg':
        return [('panel', TRAVERTINE[int(rng.integers(len(TRAVERTINE)))]), ('stone', '#9A948A'), ('panel', '#8C8A86')][int(rng.integers(3))]
    if kind in ('modern', 'kaufhaus'):
        return ('panel', ['#6E6A66', '#8C8A86', '#B8B0A2', '#4B4E52'][int(rng.integers(4))])
    if kind == 'curtain':
        return ('spandrel', '#2B3940')
    return (P['wall_mat'], P['wall_col'])


def build_building(b, ctx):
    RF.WALL_CTX.update({'tree': ctx.get('wall_tree'), 'info': ctx.get('wall_info'), 'bldgs': ctx.get('bldgs')})
    RF.DORMERS.pop(b.id, None)
    kind = classify(b)
    P = params(b, kind)
    rng = rng_of('bldg', b.id)
    acc = MeshAcc(); inst = InstAcc(); coll = []
    fa, sa, slope = roof_stats(b)
    pitched = sa > 0.45 * (fa + sa) and slope > 15
    info = {'id': b.id, 'kind': kind, 'street': b.street, 'number': b.number}
    if kind == 'canopy':
        RF.build_canopy(acc, b, P, rng, coll)
        return acc, inst, coll, info
    walls = FA.analyse_walls(b, ctx)
    globals()["_LAST_WALLS"] = walls
    fcode = (b.attrs.get('function') or '_').split('_')[-1]
    for w in walls:
        FA.street_info(w, b, ctx)
    # Bezugsniveau = Gehweg vor den Strassenwaenden (Hoehenfeld der Boden-Pipeline) statt tiefster LoD2-Bodenpunkt
    # (Rampen, Promenadenebene, Flussufer): Geschossraster, Laeden und Tueren stehen auf der Strasse; darunter Sockel.
    y0g = b.y_ground
    y_ref = FA.street_reference(b, walls, ctx)
    info['y_lod2'] = round(y0g, 2); info['y_ref'] = round(y_ref, 2)
    b.y_ground = y_ref
    try:
        return _build_rest(b, ctx, kind, P, rng, acc, inst, coll, fa, sa, slope, pitched, info, walls, fcode)
    finally:
        b.y_ground = y0g


def _build_rest(b, ctx, kind, P, rng, acc, inst, coll, fa, sa, slope, pitched, info, walls, fcode):
    SH.assign(b, walls)
    for w in walls:
        w.retail = FA.decide_retail(w, b, kind, fcode)
        FA.wall_params(w, P)
    G = FA.storey_grid(b, P, walls, pitched)
    if any(w.retail for w in walls):
        P['gf_mat_eff'], P['gf_col_eff'] = gf_material(P, kind, rng)
    elif kind == 'altbau' and P.get('rustic'):
        P['gf_mat_eff'], P['gf_col_eff'] = gf_material(P, kind, rng)
    if kind == 'fachwerk' and P.get('gf_stone'):
        P['gf_mat_eff'], P['gf_col_eff'] = 'stone', ['#E6DDCB', '#D9C9A8', '#E2D6C0', '#CDBB9A'][int(rng.integers(4))]
    if P.get('gf_override'):                                   # Erdgeschoss-Material nach Street-View-Notiz (style.OVERRIDES)
        P['gf_mat_eff'], P['gf_col_eff'] = P['gf_override']
    P['party_col'] = jitter_hex('#CFC8BC', rng, 0.04)
    flags = {}
    for w in sorted(walls, key=lambda w: (w.street_dist if not w.drop else 999, -getattr(w, 'L', 0))):
        FA.layout_wall(w, b, P, G, ctx, rng, flags)
    if P.get('emblem_wall') is not None:
        FA.emblem(acc, walls, b, P, G, clear=True)
    for w in walls:
        FA.emit_wall(acc, inst, w, b, P, G)
    if P.get('emblem_wall') is not None:
        FA.emblem(acc, walls, b, P, G)
    # Traufversatz fuer das Hauptgesims bei Steildaechern (Ueberstand + Stirnbrett)
    eaves_off = P.get('overhang', 0.4) * math.tan(math.radians(max(slope, 20))) + 0.18 + 0.02 if pitched else 0.06
    FA.emit_bands(acc, walls, b, P, G, ctx, pitched, eaves_off)
    SH.emit_signs(acc, inst, flags, b, P, ctx)
    SH.emit_glass_canopy(acc, walls, b, P, G, flags)              # Overrides: Glasvordach (Hugendubel)
    SH.emit_big_signs(acc, walls, b, P, G)                         # Overrides: grosse Buchstaben an Obergeschossen (H&M, GALERIA)
    FA.emit_fins(acc, flags, b, P, G)
    FA.emit_steps(acc, flags, b)
    if kind == 'fachwerk':
        FA.emit_fachwerk(acc, walls, b, P, G)
    if P.get('brickgothic'):
        FA.emit_brick_gable(acc, walls, b, P, G)
    wm, wc = P['wall_mat'], P['wall_col']
    if wm in ('glass', 'spandrel'):
        wm, wc = 'metal', '#5B6066'
    RF.build_flat(acc, b, P, ctx, rng, wm, wc)
    rmat, rtint = RF.build_pitched(acc, inst, b, P, ctx, rng, walls, wm, wc, None)
    if pitched:
        RF.chimneys(acc, b, P, rng, rmat)
    info.update({'storeys': G['n'], 'storey_h': round(G['s'], 2), 'gf_h': round(G['g'], 2), 'height': round(G['He'], 1),
                 'retail_walls': sum(1 for w in walls if w.retail), 'openings': sum(len(w.openings) for w in walls),
                 'pitched': bool(pitched), 'wall_mat': P['wall_mat']})
    if flags.get('shop_pos'):
        info['shops'] = flags['shop_pos']
    return acc, inst, coll, info


def lod2_collision(b):
    """LoD2-Volumen (Waende bis unter Gelaende verlaengert, Daecher, Boden) als Dreiecke."""
    tris = []
    for lst, ext in ((b.walls, True), (b.roofs, False), (b.grounds, False)):
        for s in lst:
            P = s.P.copy()
            if ext and abs(s.n[1]) < 0.05:
                low = P[:, 1] < b.y_ground + 0.05
                P[low, 1] = np.minimum(P[low, 1], b.y_base)
            n = s.n
            t = np.cross([0, 1.0, 0], n)
            if np.linalg.norm(t) < 1e-6:
                t = np.array([1.0, 0, 0])
            t /= np.linalg.norm(t); bb = np.cross(n, t)
            fr = Frame(P[0], t, bb, n)
            V2, F = tri2d(fr.loc(P)[:, :2], [fr.loc(h)[:, :2] for h in s.holes])
            if len(F) == 0:
                continue
            allP = np.concatenate([P] + list(s.holes)) if s.holes else P
            if len(allP) != len(V2):
                allP = fr.w(np.c_[V2, np.zeros(len(V2))])
            tris.append(allP[F])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def coll_tris_of(b, kind, coll):
    """Kollisionsdreiecke eines Gebaeudes: LoD2-Volumen bzw. bei Ueberdachungen Platte + Stuetzen (coll aus build_canopy)."""
    out = []
    ct = lod2_collision(b) if kind != 'canopy' else None
    if ct is not None and len(ct):
        out.append(ct)
    for q in coll:
        V2, F = tri2d(q[:, [0, 2]]) if np.ptp(q[:, 1]) < 1e-6 else (None, None)
        if V2 is not None and len(F):
            out.append(q[F])
        elif len(q) == 4:
            out.append(q[[[0, 1, 2], [0, 2, 3]]])
        elif len(q) == 8:
            for f in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
                out.append(q[[[f[0], f[1], f[2]], [f[0], f[2], f[3]]]])
    return out


def write_tile(name, acc, inst, compress, out_dir):
    """Huelle/Details (eine Datei) + Fenster-Instanzen (zweite Datei)."""
    stats = {}
    # ---------- Huelle + Details
    prims = {'shell': [], 'detail': []}
    allpos = []
    merged = collections.defaultdict(list)
    for (layer, mat), parts in acc.parts.items():
        merged[('shell' if layer.startswith('shell') else layer, mat)].extend(parts)
    for (layer, mat), parts in sorted(merged.items()):
        pr = build_primitive(parts, mat)
        prims[layer].append((mat, pr)); allpos.append(pr['pos'])
    if allpos:
        A = np.concatenate(allpos)
        lo, hi = A.min(0), A.max(0)
        qc = (lo + hi) / 2; qs = np.maximum((hi - lo) / 2, 1e-3) * 1.0005
    g = GLB(compress=compress)
    kids = []
    for layer in ('shell', 'detail'):
        if prims[layer]:
            m = g.mesh(name + '_' + layer, prims[layer], qc, qs)
            if m is not None:
                kids.append(g.node(name + '_' + layer + '_nocol', mesh=m, translation=qc, scale=qs))
            stats['tris_' + layer] = int(sum(len(p['idx']) for _, p in prims[layer]))
    g.node(name, children=kids, root=True, extras={'tile': name})
    path = os.path.join(out_dir, name + '.glb')
    stats['bytes'] = g.write(path)
    stats['bounds'] = [lo.round(2).tolist(), hi.round(2).tolist()] if allpos else None
    # ---------- Instanzen
    g = GLB(compress=compress)
    kids = []; ninst = 0; itris = 0; proto_tris = 0
    mesh_cache = {}
    for (key, mm), L in sorted(inst.g.items(), key=lambda t: (t[0][0], str(t[0][1]))):
        proto = get_proto(key)
        mmap = dict(mm) if mm else {}
        ck = (key, mm)
        if ck not in mesh_cache:
            prs = []
            for mat, (V, F, C) in proto.mesh.items():
                real = mmap.get(mat, mat)
                pr = build_primitive([(V, F, C)], real)
                prs.append((real, pr))
            mesh_cache[ck] = g.mesh('proto_' + key + ('_' + '_'.join(v for _, v in mm) if mm else ''), prs)
            proto_tris += proto.ntri
        mi = mesh_cache[ck]
        if mi is None:
            continue
        T, Q, S, C = (np.array(x) for x in L)
        kids.append(g.instanced(name + '_' + key + ('_' + '_'.join(v for _, v in mm) if mm else ''), mi, T, Q, S, C))
        ninst += len(T); itris += proto.ntri * len(T)
    g.node(name + '_win', children=kids, root=True, extras={'tile': name, 'instances': ninst})
    wpath = os.path.join(out_dir, name + '_win.glb')
    stats['win_bytes'] = g.write(wpath) if kids else 0
    if not kids and os.path.exists(wpath):
        os.remove(wpath)
    stats['instances'] = ninst; stats['instanced_tris'] = itris; stats['proto_tris'] = proto_tris
    return stats


def write_collision(path, tris, compress=False):
    g = GLB(compress=compress)
    V = tris.reshape(-1, 3).astype(np.float32)
    F = np.arange(len(V)).reshape(-1, 3)
    from gltf_io import weld
    P2, C2, _, F2 = weld(V.astype(np.float64), np.zeros((len(V), 3)), None, F)
    p32 = P2.astype(np.float32)
    v = g._view(p32, 12, 34962, 'ATTRIBUTES' if compress else None, len(p32))
    a = g._acc(v, 5126, len(p32), 'VEC3', False, p32.min(0), p32.max(0))
    idx = F2.astype(np.uint32).reshape(-1)
    iv = g._view(idx, None, 34963, 'TRIANGLES' if compress else None, len(idx))
    ia = g._acc(iv, 5125, len(idx), 'SCALAR')
    g.J['meshes'].append({'name': 'collision_buildings', 'primitives': [{'attributes': {'POSITION': a}, 'indices': ia, 'mode': 4}]})
    g.node('collision_buildings', mesh=0, root=True, extras={'collision_only': True})
    return g.write(path), len(F2)


def _replace(src, dst):
    for k in range(20):
        try:
            os.replace(src, dst); return
        except PermissionError:
            time.sleep(0.25)
    shutil.copyfile(src, dst); os.remove(src)


def swap_in(stage, final):
    """Zwischenordner -> Zielordner: erst Texturen und Kacheln (ueberschreiben), dann index.json, dann veraltete Kacheln loeschen."""
    os.makedirs(os.path.join(final, 'textures'), exist_ok=True)
    for f in os.listdir(os.path.join(stage, 'textures')):
        _replace(os.path.join(stage, 'textures', f), os.path.join(final, 'textures', f))
    new = [f for f in os.listdir(stage) if f.endswith('.glb')]
    for f in new:
        _replace(os.path.join(stage, f), os.path.join(final, f))
    _replace(os.path.join(stage, 'index.json'), os.path.join(final, 'index.json'))
    for f in os.listdir(final):
        if f.startswith('bldg_') and f.endswith('.glb') and f not in new:
            try:
                os.remove(os.path.join(final, f))
            except OSError:
                pass
    shutil.rmtree(stage, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reprep', action='store_true'); ap.add_argument('--only', default=None)
    ap.add_argument('--tiles', default=None); ap.add_argument('--no-compress', action='store_true')
    ap.add_argument('--preview', action='store_true'); ap.add_argument('--out', default=None)
    ap.add_argument('--check', action='store_true'); ap.add_argument('--raw-out', default=None)
    ap.add_argument('--patch-tiles', default=None, help='diese Kacheln (tx_tz,..) im Spielordner neu bauen (wie --patch-replaced)')
    ap.add_argument('--patch-replaced', action='store_true',
                    help='nur Kacheln neu bauen, deren Gebaeude seit dem letzten Lauf durch Wahrzeichen ersetzt (oder wieder frei) wurden')
    a = ap.parse_args()
    if a.patch_replaced or a.patch_tiles:
        return patch_replaced(a)
    t0 = time.time()
    if not os.path.exists(os.path.join(CACHE, 'lgln_parts.json')):
        import lgln_attrs; lgln_attrs.main()
    if a.reprep or not os.path.exists(os.path.join(CACHE, 'prep.pkl')):
        PREP.prepare()
    d = PREP.load_prep()
    import replaced
    repl = replaced.mark(d['bldgs'])
    ctx = make_ctx(d)
    bl = d['bldgs']
    print('Vorbereitung geladen %.1f s' % (time.time() - t0))
    test = a.only is not None
    final_web = a.out or (os.path.join(CACHE, 'test', 'web') if test else WEB_OUT)
    out_raw = a.raw_out or (os.path.join(CACHE, 'test', 'raw') if test else RAW_OUT)
    # Neue Kacheln zuerst in einen Zwischenordner schreiben und am Ende Datei fuer Datei austauschen
    # (das Spiel laeuft waehrenddessen weiter und soll keine 404 bekommen).
    out_web = os.path.join(final_web, '.staging')
    shutil.rmtree(out_web, ignore_errors=True)
    os.makedirs(out_web, exist_ok=True); os.makedirs(final_web, exist_ok=True); os.makedirs(out_raw, exist_ok=True)
    if not test:
        for f in os.listdir(out_raw):
            if f.startswith('bldg_') and f.endswith('.glb'):
                os.remove(os.path.join(out_raw, f))
    for o in (out_web, out_raw):
        TX.build(os.path.join(o, 'textures'))
    only = set(a.only.split(',')) if a.only else None
    tiles_sel = set(a.tiles.split(',')) if a.tiles else None
    mask_er = binary_erosion(d['mask'], iterations=2)
    tiles = collections.defaultdict(lambda: [MeshAcc(), InstAcc(), []])
    coll_tris = []; infos = []; intr = 0
    for b in bl:
        if b.covered or getattr(b, 'replaced', None):
            continue
        if only and not any(b.id.endswith(x) for x in only):
            continue
        tx, tz = int(math.floor(b.cx / TILE)), int(math.floor(b.cz / TILE))
        tname = 'bldg_%d_%d' % (tx, tz)
        if tiles_sel and tname[5:] not in tiles_sel:
            continue
        try:
            acc, inst, coll, info = build_building(b, ctx)
        except Exception as e:
            import traceback; traceback.print_exc()
            print('FEHLER', b.id, e); continue
        # Detailmodell-Maske: keine Geometrie hineinragen lassen (Pruefung)
        for (layer, mat), parts in acc.parts.items():
            for V, F, C in parts:
                c = np.floor(V[:, [0, 2]] + 400).astype(int)
                ok = (c >= 0).all(1) & (c < 800).all(1)
                intr += int(mask_er[c[ok, 1], c[ok, 0]].sum())
        info['tile'] = tname; info['tris'] = acc.ntri; info['instances'] = inst.count()
        infos.append(info)
        T = tiles[tname]
        T[0].merge_from(acc); T[1].merge_from(inst); T[2].append(b.id)
        coll_tris += coll_tris_of(b, info['kind'], coll)
    print('Gebaeude erzeugt: %d in %.1f s, Masken-Eindringen (Vertices): %d' % (len(infos), time.time() - t0, intr))
    index = {'version': 1, 'frame': 'game (x=dE, y=Z-56.16, z=-dN; Y up, m; origin Kroepcke clock)', 'tile_size': TILE,
             'compression': 'EXT_meshopt_compression + KHR_mesh_quantization' if not a.no_compress else 'none',
             'notes': ['Each tile: <name>.glb (shell + facade details, node *_shell_nocol / *_detail_nocol) and <name>_win.glb '
                       '(instanced window/door/shopfront assemblies, EXT_mesh_gpu_instancing with _COLOR_0 instance colours).',
                       'Load *_win.glb only for tiles within ~150-200 m of the camera; the shell already closes every opening '
                       '(reveal + dark back plane), so far tiles look right without it.',
                       'No normals are stored: three.js uses flatShading automatically.',
                       'Materials are shared by name (bldg_*); dedupe them across tiles after loading to share textures '
                       '(textures/*.jpg are referenced externally).',
                       'Collision: collision_buildings.glb (LoD2 solids; canopies as slab + posts).',
                       'Quality gates (check_rays.py): every exposed facade is hit by rays from 3 m at every storey/axis; every '
                       'window glass instance sits in a shell opening with a back plane; shell bounding spheres valid.'],
             'materials': {k: v for k, v in MATERIALS.items()}, 'tiles': []}
    tot = collections.Counter()
    for tname in sorted(tiles):
        acc, inst, ids = tiles[tname]
        cst = CP.resolve(acc)                                       # koplanare Ueberdeckungen (Z-Fighting) aufloesen
        tot['coplanar_moved_parts'] += cst.get('moved_parts', 0); tot['coplanar_dropped_parts'] += cst.get('dropped_parts', 0)
        st = write_tile(tname, acc, inst, not a.no_compress, out_web)
        sr = write_tile(tname, acc, inst, False, out_raw)
        tx, tz = map(int, tname[5:].split('_'))
        e = {'name': tname, 'tile': [tx, tz], 'file': tname + '.glb', 'windows': tname + '_win.glb' if st['instances'] else None,
             'bounds': {'min': st['bounds'][0], 'max': st['bounds'][1]} if st['bounds'] else None,
             'triangles': st.get('tris_shell', 0) + st.get('tris_detail', 0), 'triangles_shell': st.get('tris_shell', 0),
             'triangles_detail': st.get('tris_detail', 0), 'instances': st['instances'], 'instanced_triangles': st['instanced_tris'],
             'prototype_triangles': st['proto_tris'], 'bytes': st['bytes'], 'windows_bytes': st['win_bytes'],
             'raw_bytes': sr['bytes'] + sr['win_bytes'], 'buildings': ids}
        index['tiles'].append(e)
        for k in ('triangles', 'triangles_shell', 'triangles_detail', 'instances', 'instanced_triangles', 'prototype_triangles', 'bytes',
                  'windows_bytes', 'raw_bytes'):
            tot[k] += e[k]
        print('  %-14s tris %7d  inst %6d (%8d)  %6.2f MB + %5.2f MB' % (tname, e['triangles'], e['instances'], e['instanced_triangles'],
                                                                         e['bytes'] / 1e6, e['windows_bytes'] / 1e6))
    ct = np.concatenate(coll_tris) if coll_tris else np.zeros((0, 3, 3))
    cb, cn = write_collision(os.path.join(out_web, 'collision_buildings.glb'), ct)
    shutil.copy(os.path.join(out_web, 'collision_buildings.glb'), os.path.join(out_raw, 'collision_buildings.glb'))
    tot['collision_triangles'] = cn; tot['collision_bytes'] = cb
    index['collision'] = {'file': 'collision_buildings.glb', 'triangles': cn, 'bytes': cb}
    index['totals'] = dict(tot)
    index['styles'] = dict(collections.Counter(i['kind'] for i in infos))
    index['buildings'] = len(infos)
    index['replaced_by_landmarks'] = repl
    index['shops_placed'] = len(SH.db()['used'])
    index['shops_placed_ids'] = sorted(SH.db()['used'])
    _nm = {s['id']: s.get('name') for L in SH.db()['by_bldg'].values() for s in L}
    index['shops_placed_named'] = sum(1 for i in SH.db()['used'] if _nm.get(i))           # mit echtem Namen auf dem Schild
    index['shops_placed_unnamed'] = index['shops_placed'] - index['shops_placed_named']    # geschlossen/leer/ohne Namen
    save_json(index, os.path.join(out_web, 'index.json'), indent=1)
    save_json(index, os.path.join(out_raw, 'index.json'), indent=1)
    swap_in(out_web, final_web)
    save_json(infos, os.path.join(CACHE, 'building_info.json' if not (test or a.out) else 'test_info.json'), indent=0)
    print('Summe:', dict(tot))
    print('Stile:', index['styles'])
    print('fertig in %.1f s' % (time.time() - t0))
    here = os.path.dirname(os.path.abspath(__file__))
    if a.check and not test:
        import subprocess
        subprocess.run([sys.executable, os.path.join(here, 'check_rays.py'), final_web], check=False)
        subprocess.run([sys.executable, os.path.join(here, 'check_gaps.py')], check=False)
    if a.preview:
        import subprocess
        from common import BLENDER
        subprocess.run([sys.executable, os.path.join(here, 'preview_cams.py')], check=False)
        subprocess.run([BLENDER, '-b', '--factory-startup', '-P', os.path.join(here, 'blender_preview.py'), '--', out_raw], check=False)


def patch_replaced(a):
    """Nachtrag fuer spaet geschriebene data/landmarks_replace/*.json: betroffene Kacheln neu bauen, index.json zusammenfuehren,
    Kollision neu schreiben, ohne die uebrigen Kacheln anzufassen (Austausch Datei fuer Datei wie im Vollbau)."""
    t0 = time.time()
    final_web = a.out or WEB_OUT
    out_raw = a.raw_out or RAW_OUT
    old = load_json(os.path.join(final_web, 'index.json'))
    d = PREP.load_prep()
    import replaced
    repl = replaced.mark(d['bldgs'])
    prev = old.get('replaced_by_landmarks', {})
    changed = set(repl) ^ set(prev)
    bl = [b for b in d['bldgs'] if not b.covered]
    tname = lambda b: 'bldg_%d_%d' % (int(math.floor(b.cx / TILE)), int(math.floor(b.cz / TILE)))
    by = {b.id: b for b in bl}
    sel = sorted({tname(by[i]) for i in changed if i in by} | {'bldg_' + t for t in (a.patch_tiles.split(',') if a.patch_tiles else [])})
    if not sel:
        print('keine Aenderung der Wahrzeichen-Ersetzungen - nichts zu tun')
        return
    print('Kacheln neu:', sel)
    ctx = make_ctx(d)
    stage = os.path.join(final_web, '.staging')
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, 'textures'), exist_ok=True)
    TX.build(os.path.join(stage, 'textures'))
    tiles = collections.defaultdict(lambda: [MeshAcc(), InstAcc(), []])
    infos = []
    coll_tris = []
    for b in bl:
        if getattr(b, 'replaced', None):
            continue
        tn = tname(b)
        if tn in sel:
            acc, inst, coll, info = build_building(b, ctx)
            info['tile'] = tn; info['tris'] = acc.ntri; info['instances'] = inst.count()
            infos.append(info)
            T = tiles[tn]; T[0].merge_from(acc); T[1].merge_from(inst); T[2].append(b.id)
            coll_tris += coll_tris_of(b, info['kind'], coll)
        else:
            kind = classify(b)
            coll = build_building(b, ctx)[2] if kind == 'canopy' else []
            coll_tris += coll_tris_of(b, kind, coll)
    entries = {e['name']: e for e in old['tiles']}
    for tn in sel:
        entries.pop(tn, None)
        if tn not in tiles:
            continue
        acc, inst, ids = tiles[tn]
        CP.resolve(acc)
        st = write_tile(tn, acc, inst, True, stage)
        sr = write_tile(tn, acc, inst, False, out_raw)
        tx, tz = map(int, tn[5:].split('_'))
        entries[tn] = {'name': tn, 'tile': [tx, tz], 'file': tn + '.glb', 'windows': tn + '_win.glb' if st['instances'] else None,
                       'bounds': {'min': st['bounds'][0], 'max': st['bounds'][1]} if st['bounds'] else None,
                       'triangles': st.get('tris_shell', 0) + st.get('tris_detail', 0), 'triangles_shell': st.get('tris_shell', 0),
                       'triangles_detail': st.get('tris_detail', 0), 'instances': st['instances'], 'instanced_triangles': st['instanced_tris'],
                       'prototype_triangles': st['proto_tris'], 'bytes': st['bytes'], 'windows_bytes': st['win_bytes'],
                       'raw_bytes': sr['bytes'] + sr['win_bytes'], 'buildings': ids}
    idx = dict(old)
    idx['tiles'] = [entries[k] for k in sorted(entries)]
    tot = collections.Counter()
    for e in idx['tiles']:
        for k in ('triangles', 'triangles_shell', 'triangles_detail', 'instances', 'instanced_triangles', 'prototype_triangles', 'bytes',
                  'windows_bytes', 'raw_bytes'):
            tot[k] += e[k]
    ct = np.concatenate(coll_tris) if coll_tris else np.zeros((0, 3, 3))
    cb, cn = write_collision(os.path.join(stage, 'collision_buildings.glb'), ct)
    shutil.copy(os.path.join(stage, 'collision_buildings.glb'), os.path.join(out_raw, 'collision_buildings.glb'))
    tot['collision_triangles'] = cn; tot['collision_bytes'] = cb
    idx['collision'] = {'file': 'collision_buildings.glb', 'triangles': cn, 'bytes': cb}
    idx['totals'] = dict(tot)
    ip = os.path.join(CACHE, 'building_info.json')
    oldinfo = load_json(ip) if os.path.exists(ip) else []
    allinfo = [i for i in oldinfo if i.get('tile') not in sel and i['id'] not in repl] + infos
    idx['styles'] = dict(collections.Counter(i['kind'] for i in allinfo))
    idx['buildings'] = len(allinfo)
    idx['replaced_by_landmarks'] = repl
    gone = {sp['id'] for i in oldinfo if i.get('tile') in sel for sp in i.get('shops', [])}
    ids = (set(old.get('shops_placed_ids', [])) - gone) | set(SH.db()['used'])
    idx['shops_placed'] = len(ids)
    idx['shops_placed_ids'] = sorted(ids)
    save_json(idx, os.path.join(stage, 'index.json'), indent=1)
    save_json(idx, os.path.join(out_raw, 'index.json'), indent=1)
    # Austausch: neue Kacheln, Index, Kollision; leere Kacheln entfernen; uebrige Kacheln bleiben unberuehrt
    for f in os.listdir(os.path.join(stage, 'textures')):
        _replace(os.path.join(stage, 'textures', f), os.path.join(final_web, 'textures', f))
    for f in os.listdir(stage):
        if f.endswith('.glb'):
            _replace(os.path.join(stage, f), os.path.join(final_web, f))
    _replace(os.path.join(stage, 'index.json'), os.path.join(final_web, 'index.json'))
    for tn in sel:
        gone_files = [tn + '.glb', tn + '_win.glb'] if tn not in entries else ([] if entries[tn]['windows'] else [tn + '_win.glb'])
        for f in gone_files:
            for dd in (final_web, out_raw):
                try:
                    os.remove(os.path.join(dd, f))
                except OSError:
                    pass
    shutil.rmtree(stage, ignore_errors=True)
    save_json(allinfo, ip, indent=0)
    print('Nachtrag fertig: %d Kacheln, %d Gebaeude ersetzt, %.1f s' % (len(sel), len(repl), time.time() - t0))


if __name__ == '__main__':
    main()

"""Stadtbahn/Bus an der Oberflaeche (Ausbau 2).
- Hochbahnsteige (OSM-Plattformen an Oberflaechengleisen, u. a. 'Steintor' Muenzstrasse, Linien 10/17): Koerper 0.86 m ueber
  Gelaende, Kante auf 1.42 m Abstand zur Gleisachse ausgerichtet, heller Granit-Kantenstein, weisser Sicherheitsstreifen,
  Rippen-Aufmerksamkeitsstreifen, Rampen (6-8 %) mit Edelstahlgelaender, Glas-Wartehallen mit Bank + Leuchtdach,
  Fahrkartenautomat, DFI-Anzeiger, H-Mast mit Haltestellenname, Abfallbehaelter.
- Mendini-'BUSSTOPS' Steintor (Kurt-Schumacher-Strasse, OSM building=roof name=BUSSTOPS): Schachbrett-Balken (gelb/schwarz)
  auf Schachbrett-Stuetzen, Kubus-Haeuschen mit goldenem Kegelturm, kleiner Goldkegel am anderen Ende, Glas-Windschutz, Bank.
- Oberleitung entlang aller Oberflaechen-Gleise (OSM railway=light_rail/tram): Masten auf Gehweg/Insel (nie auf der Fahrbahn),
  Querverspannung Mast-Mast bzw. Ausleger, Haenger, Fahrdraht je Gleis (Zickzack +-0.2 m) auf 5.6 m.
(Rillenschienen: structures.py)   Ausgabe: data/ground_cache/transit.pkl"""
import os, json, time, pickle, math
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point, MultiLineString, box
from shapely.ops import unary_union, linemerge
import triangle as tr
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from common import *
import geom as G
import geom2 as G2
from geom2 import Frame
from surface import Surface
from kit import panel_box, railing_line, _strip

T0 = time.time()
def log(*a): print('[transit %6.1fs]' % (time.time() - T0), *a, flush=True)
L = pickle.load(open(CACHE + 'layout.pkl', 'rb'))
SURF = Surface()
osm = load_osm()
LOW = L['low']; BLD = L['bld']; REG = L['region']
HOLES = unary_union([h['poly'] for h in L['holes'].values()])
for g_ in (LOW, BLD, REG, HOLES): shapely.prepare(g_)

parts = []; coll = []; walk = []
def add(p, collide=False, walkable=False):
    if p is None: return
    if isinstance(p, (list, tuple)):
        for q in p: add(q, collide, walkable)
        return
    if len(p['idx']) == 0: return
    parts.append(p)
    if collide: coll.append(p)
    if walkable: walk.append(p)
def gy(x, z): return float(SURF([x], [z])[0][0])

# ---------------------------------------------------------------- Texturen (prozedural, atomar)
TD = OUT + 'tex/'
def save(im, name):
    tmp = TD + '.' + name + '.tmp'; im.save(tmp, format='PNG', optimize=True); os.replace(tmp, TD + name)
def font(sz, bold=True):
    for f in (('arialbd.ttf' if bold else 'arial.ttf'), 'DejaVuSans-Bold.ttf'):
        try: return ImageFont.truetype(f, sz)
        except OSError: pass
    return ImageFont.load_default()
def plate(text, w=1024, h=256, bg=(255, 255, 255), fg=(20, 20, 20), border=(0, 102, 51)):
    im = Image.new('RGB', (w, h), bg); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=border, width=14)
    f = font(int(h * 0.50)); bb = d.textbbox((0, 0), text, font=f); tw = bb[2] - bb[0]
    if tw > w * 0.88: f = font(int(h * 0.50 * w * 0.88 / tw)); bb = d.textbbox((0, 0), text, font=f); tw = bb[2] - bb[0]
    d.text(((w - tw) / 2 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), text, fill=fg, font=f)
    return im
# H-Zeichen (gruenes H auf gelbem Kreis, wie an Hannovers Haltestellen)
hs = Image.new('RGB', (256, 256), (255, 255, 255)); d = ImageDraw.Draw(hs)
d.ellipse([8, 8, 248, 248], fill=(250, 205, 0), outline=(0, 110, 55), width=10)
f = font(170); bb = d.textbbox((0, 0), 'H', font=f)
d.text((128 - (bb[2] - bb[0]) / 2 - bb[0], 128 - (bb[3] - bb[1]) / 2 - bb[1]), 'H', fill=(0, 110, 55), font=f)
save(hs, 'hst_h.png')
# Schachbrett gelb/schwarz (Mendini), 2x2 Felder je Kachel
cb = np.zeros((256, 256, 3), np.uint8); yy, xx = np.mgrid[0:256, 0:256]
chk = ((xx // 128) + (yy // 128)) % 2 == 0
cb[chk] = (230, 190, 46); cb[~chk] = (22, 22, 22)
rng = np.random.default_rng(5)
cb = np.clip(cb.astype(np.int16) + rng.integers(-6, 7, cb.shape), 0, 255).astype(np.uint8)
cb[(xx % 128 < 2) | (yy % 128 < 2)] = (120, 120, 112)
save(Image.fromarray(cb), 'checker_yb.png')
NAMEMAT = {}
def name_mat(name):
    key = 'hst_' + ''.join(c for c in name.lower().replace('ö', 'oe').replace('/', '_') if c.isalnum() or c == '_')
    if key not in NAMEMAT:
        save(plate(name), key + '.png'); NAMEMAT[key] = name
        MATS_EXTRA[key] = {'tex': 'tex/' + key + '.png', 'rough': 0.4, 'emissive_tex': True, 'emissive': (0.35, 0.35, 0.35)}
    return key
MATS_EXTRA = {'hst_h': {'tex': 'tex/hst_h.png', 'rough': 0.4}, 'checker_yb': {'tex': 'tex/checker_yb.png', 'rough': 0.3}}

# ---------------------------------------------------------------- Gleise
LRL = []
for c, t in L['lr']:
    g = LineString(c).intersection(REG.buffer(30))
    for q in getattr(g, 'geoms', [g]):
        if q.geom_type == 'LineString' and q.length > 2: LRL.append(q)
LRU = unary_union(LRL)
log('surface track lines', len(LRL), 'length', round(sum(l.length for l in LRL)))
CLR = 1.42                        # Bahnsteigkante - Gleisachse
H_PLAT = 0.86                     # Hochbahnsteig-Hoehe ueber Gelaende (Hannover)
PLATFORMS = []                    # fuer furniture/catenary: belegte Flaechen

def osm_platform_name(P):
    best = (1e9, None)
    for w in osm['ways']:
        t = w['tags']
        if t.get('public_transport') != 'platform' or not t.get('name') or len(w['pts']) < 2: continue
        d = LineString(w['pts']).distance(P) if len(w['pts']) < 3 else Polygon(w['pts']).buffer(0).distance(P)
        if d < best[0]: best = (d, t['name'])
    return best[1] if best[0] < 5 else 'Haltestelle'

# ---------------------------------------------------------------- Hochbahnsteig
def tri_pslg(rings_lines, region, maxa=0.8):
    """Triangulierung der Region mit allen Linien als Zwangskanten; -> V (n,2), T (m,3)"""
    noded = shapely.unary_union(MultiLineString([np.asarray(r.coords) for r in rings_lines]), grid_size=0.001)
    noded = shapely.segmentize(noded, 1.0)
    segs = np.concatenate([np.stack([np.asarray(ls.coords)[:-1], np.asarray(ls.coords)[1:]], 1) for ls in getattr(noded, 'geoms', [noded]) if len(ls.coords) >= 2])
    key = np.round(segs.reshape(-1, 2) * 1000).astype(np.int64); uk, inv = np.unique(key, axis=0, return_inverse=True)
    V0 = uk / 1000.0; S0 = inv.reshape(-1, 2); S0 = S0[S0[:, 0] != S0[:, 1]]
    r = tr.triangulate({'vertices': V0, 'segments': S0}, 'pq25a%.2fQ' % maxa)
    V, T = r['vertices'], r['triangles']
    c = V[T].mean(1); keep = shapely.contains_xy(region.buffer(1e-4), c[:, 0], c[:, 1])
    return V, T[keep]

# ---------------------------------------------------------------- Steintor-Ueberdachung (Muenzstrasse, 'futuristisch')
def loft(fr, mat, S, prof_fn, ybase, uvscale=1.0, caps=True):
    """Profil (t, dy) je Station s (gleiche Punktzahl, gegen den Uhrzeigersinn in (t, y) von +s gesehen) entlang s extrudieren."""
    out = []
    R = [np.array([fr.P(s_, t_, ybase(s_) + dy) for t_, dy in prof_fn(s_)]) for s_ in S]
    k = len(R[0]); Q = []
    for a, b in zip(R[:-1], R[1:]):
        for i in range(k):
            j = (i + 1) % k
            Q.append([a[i], b[i], b[j], a[j]])
    p = G.flat_quads(mat, np.array(Q), uvscale)
    # Orientierung: nach aussen (weg von der Profilmitte)
    P_, I_ = p['pos'], p['idx']; fn = np.cross(P_[I_[:, 1]] - P_[I_[:, 0]], P_[I_[:, 2]] - P_[I_[:, 0]])
    cen = []
    for a, b in zip(R[:-1], R[1:]):
        m = (a.mean(0) + b.mean(0)) / 2; cen += [m] * (2 * k)
    cen = np.array(cen)
    cen = np.concatenate([cen[0::2], cen[1::2]]) if False else cen
    tc = P_[I_].mean(1)
    # Dreiecke von flat_quads: erst alle (0,1,2), dann alle (0,2,3) -> Zentren passend wiederholen
    nq = len(Q); cq = np.repeat(np.array([(a.mean(0) + b.mean(0)) / 2 for a, b in zip(R[:-1], R[1:])]), k, 0)
    cq = np.concatenate([cq, cq])
    bad = (fn * (tc - cq)).sum(1) < 0; I_[bad] = I_[bad][:, ::-1]; p['nrm'] = G.vertex_normals(P_, I_)
    out.append(p)
    if caps:
        import mapbox_earcut as earcut
        for Rr, sg in ((R[0], -1), (R[-1], 1)):
            pr = np.array(prof_fn(S[0] if sg < 0 else S[-1]))
            tri = earcut.triangulate_float64(pr, np.array([len(pr)], np.uint32)).reshape(-1, 3)
            q = G.part(mat, Rr, tri, None, pr / uvscale)
            fn2 = np.cross(Rr[tri[:, 1]] - Rr[tri[:, 0]], Rr[tri[:, 2]] - Rr[tri[:, 0]])
            if (fn2 @ fr.vec(sg, 0)).sum() < 0: q['idx'] = q['idx'][:, ::-1]
            q['nrm'] = np.tile(fr.vec(sg, 0), (len(Rr), 1)); out.append(q)
    return out

def build_canopy(fr, sa, sb, yt, t_range, sign):
    """langes Dach (hellgrau metallisch, angefaste Kanten) auf geneigten, sich nach oben weitenden Stuetzen in Bahnsteigmitte;
    Bug ueber der Rampe (Unterkante steigt keilfoermig an); Leuchtbaender, DFI-Anzeiger, Glas-Windschutz, Baenke, Automaten."""
    trm = t_range((sa + sb) / 2); tc = (trm[0] + trm[1]) / 2 if trm else 0.0
    HB, HT, WB, WT = 3.35, 3.75, 1.65, 1.95           # Unterkante, Oberkante, halbe Breite unten/oben
    PROW = 3.2
    def prof(s_):
        lift = 0.0
        if s_ > sb: lift = (s_ - sb) / PROW * (HT - HB - 0.04)
        if s_ < sa: lift = (sa - s_) / PROW * (HT - HB - 0.04)
        return [(tc - WB, HB + lift), (tc + WB, HB + lift), (tc + WT, HT - 0.12), (tc + WT, HT), (tc - WT, HT), (tc - WT, HT - 0.12)]
    yfl = lambda s_: float(yt(np.clip(s_, sa, sb)))
    S = np.r_[np.linspace(sa - PROW, sa, 4)[:-1], np.linspace(sa, sb, max(3, int((sb - sa) / 2.0)) + 1), np.linspace(sb, sb + PROW, 4)[1:]]
    add(loft(fr, 'alu', S, prof, yfl, 2.0))
    # Untersicht-Leuchtbaender (knapp unter der Unterkante, eigene Flaechen)
    for dt in (-0.9, 0.9):
        for a_, b_ in zip(S[3:-3][:-1], S[3:-3][1:]):
            add(fr.hquad('lightline', a_, b_, tc + dt - 0.05, tc + dt + 0.05, 0, up=False, y_s=(yfl(a_) + HB - 0.01, yfl(b_) + HB - 0.01)))
    # Stuetzen alle ~9 m: unten schmal, oben breit (Trapez-Prisma)
    ncol = max(2, int(round((sb - sa) / 9.0)) + 1)
    for s_ in np.linspace(sa + 1.5, sb - 1.5, ncol):
        y0 = yfl(s_); y1 = y0 + HB + 0.02
        bot = [(-0.22, -0.16), (0.22, -0.16), (0.22, 0.16), (-0.22, 0.16)]; top = [(-0.95, -0.34), (0.95, -0.34), (0.95, 0.34), (-0.95, 0.34)]
        B = [fr.P(s_ + a_, tc + b_, y0 - 0.02) for a_, b_ in bot]; T_ = [fr.P(s_ + a_, tc + b_, y1) for a_, b_ in top]
        cc = fr.P(s_, tc, (y0 + y1) / 2)
        for i in range(4):
            j = (i + 1) % 4
            q = G2.quad('alu', B[i], B[j], T_[j], T_[i], uvscale=1.0)
            if q is None: continue
            if np.dot(q['nrm'][0], q['pos'].mean(0) - cc) < 0: q['idx'] = q['idx'][:, ::-1]; q['nrm'] = -q['nrm']
            add(q, True)
    # Glas-Windschutz (Mittelachse) zwischen den Stuetzen, Baenke beidseitig, Automaten, DFI
    cols = np.linspace(sa + 1.5, sb - 1.5, ncol)
    for k, (a_, b_) in enumerate(zip(cols[:-1], cols[1:])):
        m = (a_ + b_) / 2; y0 = yfl(m)
        if k % 2 == 0:
            add(fr.vquad('glass', (m - 2.2, tc), (m + 2.2, tc), y0 + 0.1, y0 + 2.3, (m, tc + 3)))
            add(fr.box('paint_dark', m - 2.2, m + 2.2, tc - 0.03, tc + 0.03, y0 + 2.3, y0 + 2.36))
            for e_ in (m - 2.2, m + 2.2): add(fr.box('paint_dark', e_ - 0.03, e_ + 0.03, tc - 0.03, tc + 0.03, y0, y0 + 2.3))
            coll.append(fr.box('collision', m - 2.2, m + 2.2, tc - 0.04, tc + 0.04, y0, y0 + 2.3))
            for sg in (-1, 1):
                tb = tc + sg * 0.42
                add(fr.box('wood', m - 1.6, m + 1.6, tb - 0.2, tb + 0.2, y0 + 0.44, y0 + 0.48), True)
                for e_ in (m - 1.4, m + 1.4): add(fr.box('paint_dark', e_ - 0.03, e_ + 0.03, tb - 0.17, tb + 0.17, y0, y0 + 0.44))
        else:
            add(panel_box(fr, 'ticket_machine', 'paint_grey', m - 0.36, m + 0.36, tc - 0.25, tc + 0.25, y0, y0 + 1.80, ('t0', 't1')), True)
            add(fr.box('paint_grey', m - 0.42, m + 0.42, tc - 0.3, tc + 0.3, y0 + 1.80, y0 + 1.86))
            add(panel_box(fr, 'citylight', 'paint_dark', m + 1.3, m + 1.42, tc - 0.62, tc + 0.62, y0 + 0.35, y0 + 2.15, ('s0', 's1')), True)
    for s_ in (cols[0] + 2.4, cols[-1] - 2.4):
        y0 = yfl(s_)
        add(panel_box(fr, 'display_tram', 'paint_dark', s_ - 0.07, s_ + 0.07, tc - 0.75, tc + 0.75, y0 + 2.55, y0 + 2.98, ('s0', 's1')))
        for dt in (-0.5, 0.5): add(G2.cyl_between('paint_dark', fr.P(s_, tc + dt, y0 + 2.98), fr.P(s_, tc + dt, y0 + HB), 0.02, 6))
    log('Steintor canopy s %.1f..%.1f columns %d' % (sa, sb, ncol))

def build_platform(P0, nm):
    tracks = [l for l in LRL if l.distance(P0) < 8]
    if not tracks: return
    mrr = np.array(P0.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = mrr[1] - mrr[0], mrr[2] - mrr[1]
    ax = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2; ax = ax / np.linalg.norm(ax)
    fr = Frame(np.array(P0.centroid.coords[0]), ax)
    st0 = np.array([fr.st(*p) for p in P0.exterior.coords]); s0, s1 = st0[:, 0].min(), st0[:, 0].max()
    TB = unary_union([l.buffer(CLR, cap_style=2) for l in tracks])
    near = unary_union([l.buffer(CLR + 1.6, cap_style=2) for l in tracks])
    band = Polygon([fr.xz(s0, -60), fr.xz(s1, -60), fr.xz(s1, 60), fr.xz(s0, 60)])
    P = unary_union([P0.buffer(0), P0.buffer(1.6, join_style=2).intersection(band).intersection(near)]).difference(TB)
    P = max(getattr(P, 'geoms', [P]), key=lambda g: g.area)
    P = Polygon(P.exterior).simplify(0.03)
    st = np.array([fr.st(*p) for p in P.exterior.coords]); s0, s1 = st[:, 0].min(), st[:, 0].max()
    Lp = s1 - s0
    # Rampen: an Enden mit Fussweg-/Ueberweg-Anschluss (sonst beide)
    conn = [LineString(c) for c, t in L['foot_lines'] + L['ped_lines'] + L['side_lines']]
    ends = []
    for se in (s0, s1):
        q = Point(fr.xz(se, float(np.mean(st[np.abs(st[:, 0] - se) < 1.0, 1])) if (np.abs(st[:, 0] - se) < 1.0).any() else 0.0))
        ends.append(any(c.distance(q) < 4.0 for c in conn))
    if not any(ends): ends = [True, True]
    Lr = float(np.clip(H_PLAT / 0.075, 8.0, 0.3 * Lp))
    ramp0 = s0 + Lr if ends[0] else s0; ramp1 = s1 - Lr if ends[1] else s1
    # Gelaende entlang der Achse (lineare Ausgleichsgerade)
    Ss = np.linspace(s0, s1, 25); gs = []
    for s_ in Ss:
        ln = LineString([fr.xz(s_, -40), fr.xz(s_, 40)]).intersection(P)
        q = np.array(ln.centroid.coords[0]) if not ln.is_empty else fr.xz(s_, 0)
        gs.append(gy(*q))
    cf = np.polyfit(Ss, gs, 1)
    def yt(s):
        s = np.asarray(s, float); g = np.polyval(cf, s)
        f = np.ones_like(s)
        if ends[0]: f = np.minimum(f, (s - s0) / max(ramp0 - s0, 1e-6))
        if ends[1]: f = np.minimum(f, (s1 - s) / max(s1 - ramp1, 1e-6))
        return g + 0.03 + (H_PLAT - 0.03) * np.clip(f, 0, 1)
    # Kanten am Gleis
    ring = np.array(shapely.segmentize(P.exterior, 1.0).coords)
    segs = list(zip(ring[:-1], ring[1:]))
    def is_track_edge(a, b):
        m = (a + b) / 2; d = LRU.distance(Point(m))
        return CLR - 0.06 < d < CLR + 0.12
    te = [LineString([a, b]) for a, b in segs if is_track_edge(a, b)]
    EL = linemerge(unary_union(te)) if te else None
    rings = [P.exterior]
    zones = {}
    if EL is not None and not EL.is_empty:
        Bk = P.intersection(EL.buffer(0.30, cap_style=2, join_style=2))
        Bw = P.intersection(EL.buffer(0.40, cap_style=2, join_style=2)).difference(Bk)
        Bt = P.intersection(EL.buffer(1.00, cap_style=2, join_style=2)).difference(EL.buffer(0.70, cap_style=2, join_style=2))
        for nm_, g_ in (('granite', Bk), ('marking', Bw), ('tactile', Bt)):
            g_ = g_.buffer(0)
            zones[nm_] = g_
            for p_ in getattr(g_, 'geoms', [g_]):
                if p_.geom_type == 'Polygon' and not p_.is_empty: rings += [p_.exterior] + list(p_.interiors)
    for se in (ramp0, ramp1):
        if s0 + 0.05 < se < s1 - 0.05: rings.append(LineString([fr.xz(se, -60), fr.xz(se, 60)]).intersection(P))
    rings = [r_ for r_ in rings if not r_.is_empty]
    rl = []
    for r_ in rings:
        for q in getattr(r_, 'geoms', [r_]):
            if q.geom_type in ('LineString', 'LinearRing') and len(q.coords) >= 2: rl.append(q)
    V, T = tri_pslg(rl, P, 0.8)
    c = V[T].mean(1); lab = np.full(len(T), 'stone_light', dtype=object)
    for nm_ in ('tactile', 'marking', 'granite'):
        if nm_ in zones: lab[shapely.contains_xy(zones[nm_], c[:, 0], c[:, 1])] = nm_
    sv = np.array([fr.st(x, z)[0] for x, z in V]); Y = yt(sv)
    pos = np.c_[V[:, 0], Y, V[:, 1]]
    uvs = {'stone_light': 2.0, 'granite': 1.0, 'marking': 1.0, 'tactile': 0.6}
    for m in np.unique(lab):
        Tm = T[lab == m]
        fn = np.cross(pos[Tm[:, 1]] - pos[Tm[:, 0]], pos[Tm[:, 2]] - pos[Tm[:, 0]]); Tm = Tm.copy(); Tm[fn[:, 1] < 0] = Tm[fn[:, 1] < 0][:, ::-1]
        add(G.part(m, pos, Tm, None, V / uvs[m] if m != 'tactile' else np.c_[V @ fr.d, V @ fr.r] / 0.6), True, True)
    # Seitenwaende entlang der Randkanten der Triangulierung (wasserdicht zur Oberseite)
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    ek, cnt = np.unique(E, axis=0, return_counts=True); bnd = ek[cnt == 1]
    Q_k, Q_c = [], []
    rail_segs = []
    for u, w in bnd:
        a, b = V[u], V[w]; m = (a + b) / 2
        d = b - a; ln = np.linalg.norm(d)
        if ln < 1e-6: continue
        nrm = np.array([-d[1], d[0]]) / ln
        if P.contains(Point(m + nrm * 0.05)): nrm = -nrm
        ya, yb = Y[u], Y[w]; ga, gb = gy(*a) - 0.06, gy(*b) - 0.06
        q = np.array([[a[0], ga, a[1]], [b[0], gb, b[1]], [b[0], yb, b[1]], [a[0], ya, a[1]]])
        n3 = np.cross(q[1] - q[0], q[3] - q[0])
        if np.dot(n3[[0, 2]], nrm) < 0: q = q[[1, 0, 3, 2]]
        tedge = is_track_edge(a, b)
        (Q_k if tedge else Q_c).append(q)
        sm = fr.st(*m)[0]
        in_ramp = (ends[0] and sm < ramp0 - 0.2) or (ends[1] and sm > ramp1 + 0.2)
        at_end = min(abs(sm - s0), abs(sm - s1)) < 0.6 and abs(np.dot(d / ln, fr.d)) < 0.5
        if (not tedge or in_ramp) and not at_end and max(ya - ga, yb - gb) > 0.30:
            rail_segs.append(LineString([a, b]))
    if Q_k: add(G.flat_quads('granite', np.array(Q_k), 1.0), True)
    if Q_c: add(G.flat_quads('concrete', np.array(Q_c), 2.0), True)
    # Gelaender (Edelstahl) an Rampen und gleisfernen Seiten
    rm = linemerge(unary_union(rail_segs)) if rail_segs else None
    for ch in (getattr(rm, 'geoms', [rm]) if rm is not None and not rm.is_empty else []):
        C = np.array(ch.coords); out3 = []
        for k in range(len(C)):
            a, b = C[max(k - 1, 0)], C[min(k + 1, len(C) - 1)]; d = b - a; d = d / (np.linalg.norm(d) + 1e-12)
            nrm = np.array([-d[1], d[0]])
            if not P.contains(Point(C[k] + nrm * 0.07)): nrm = -nrm
            q = C[k] + nrm * 0.07; out3.append([q[0], float(yt(fr.st(*C[k])[0])), q[1]])
        out3 = np.array(out3)
        if len(out3) < 2: continue
        add(railing_line(out3, 1.4, 1.0, 0.022, True))
        coll.append(_strip('collision', out3, out3 + [0, 1.0, 0], look=np.r_[fr.o[0], out3[:, 1].mean() + 0.5, fr.o[1]]))
    # ---- Ausstattung auf dem ebenen Teil
    island = len(tracks) >= 2 and EL is not None and len(getattr(EL, 'geoms', [EL])) >= 2
    def t_range(s_):
        ln = LineString([fr.xz(s_, -60), fr.xz(s_, 60)]).intersection(P)
        if ln.is_empty: return None
        pts = np.array([fr.st(*p)[1] for g_ in getattr(ln, 'geoms', [ln]) for p in g_.coords])
        return pts.min(), pts.max()
    def back_side(s_):
        tr_ = t_range(s_)
        if tr_ is None: return None
        ta, tb = tr_
        da = LRU.distance(Point(fr.xz(s_, ta))); db = LRU.distance(Point(fr.xz(s_, tb)))
        return (ta, tb, -1 if da > db else 1)          # -1: Rueckseite bei ta (gleisfern)
    fl0, fl1 = ramp0 + 1.5, ramp1 - 1.5
    nsh = 0 if fl1 - fl0 < 9 else (1 if fl1 - fl0 < 42 else 2)
    shel = [fl0 + (fl1 - fl0) * (k + 1) / (nsh + 1) for k in range(nsh)]
    sign = name_mat(nm)
    if nm.startswith('Steintor') and island:
        build_canopy(fr, fl0 - 0.5, fl1 + 0.5, yt, t_range, sign)
        shel = []
    for sc in shel:
        bs = back_side(sc)
        if bs is None: continue
        ta, tb, bk = bs; W = tb - ta
        if W < 2.6: continue
        y0 = float(yt(sc)); LS = 7.2
        if island and W > 3.4:
            tc_ = (ta + tb) / 2; tw0, tw1 = tc_ - 0.02, tc_ + 0.02; troof = (tc_ - 1.35, tc_ + 1.35); tbench = (tc_ - 0.55, tc_ + 0.55)
        else:
            tb_ = ta + 1.1 if bk < 0 else tb - 1.1
            tw0, tw1 = tb_ - 0.02, tb_ + 0.02; troof = (tb_ - 0.25, tb_ + 1.75) if bk < 0 else (tb_ - 1.75, tb_ + 0.25)
            tbench = (tb_ + 0.55,) if bk < 0 else (tb_ - 0.55,)
        sa, sb = sc - LS / 2, sc + LS / 2
        # Pfosten (anthrazit) + Glaswand + Glas-Stirnseiten
        for s_ in np.linspace(sa, sb, 4):
            add(fr.box('paint_dark', s_ - 0.05, s_ + 0.05, tw0 - 0.04, tw1 + 0.04, y0, y0 + 2.75), True)
        add(fr.vquad('glass', (sa, (tw0 + tw1) / 2), (sb, (tw0 + tw1) / 2), y0 + 0.10, y0 + 2.55, (sc, troof[0] - 5)))
        coll.append(fr.box('collision', sa, sb, tw0, tw1, y0, y0 + 2.6))
        for s_ in (sa, sb):
            add(fr.vquad('glass', (s_, troof[0] + 0.15), (s_, troof[1] - 0.15), y0 + 0.10, y0 + 2.45, (s_ + (1 if s_ == sa else -1), (tw0 + tw1) / 2)))
        add(fr.box('paint_dark', sa - 0.15, sb + 0.15, troof[0], troof[1], y0 + 2.75, y0 + 2.92))
        add(fr.hquad('lightline', sa + 0.3, sb - 0.3, (troof[0] + troof[1]) / 2 - 0.06, (troof[0] + troof[1]) / 2 + 0.06, y0 + 2.749, up=False))
        add(fr.box('glass', sa - 0.1, sb + 0.1, troof[0] + 0.05, troof[1] - 0.05, y0 + 2.92, y0 + 2.94))
        # Baenke (Holzlatten auf Stahlkonsolen)
        for tbn in tbench:
            for k in range(2):
                ss0, ss1 = sa + 0.5 + k * 3.3, sa + 0.5 + k * 3.3 + 2.6
                add(fr.box('wood', ss0, ss1, tbn - 0.21, tbn + 0.21, y0 + 0.44, y0 + 0.48), True)
                for e_ in (ss0 + 0.2, ss1 - 0.2): add(fr.box('paint_dark', e_ - 0.03, e_ + 0.03, tbn - 0.18, tbn + 0.18, y0, y0 + 0.44))
        # Werbevitrine an einem Ende
        add(panel_box(fr, 'citylight', 'paint_dark', sb + 0.35, sb + 0.47, (tw0 + tw1) / 2 - 0.62, (tw0 + tw1) / 2 + 0.62, y0 + 0.35, y0 + 2.15, ('s0', 's1')), True)
        add(fr.box('paint_dark', sb + 0.36, sb + 0.46, (tw0 + tw1) / 2 - 0.66, (tw0 + tw1) / 2 + 0.66, y0, y0 + 0.35))
        # Fahrgastanzeiger (DFI) unter dem Dach, beidseitig
        add(panel_box(fr, 'display_tram', 'paint_dark', sc - 0.06, sc + 0.06, troof[0] + 0.25, troof[0] + 1.55, y0 + 2.28, y0 + 2.66, ('t0', 't1')))
        add(G2.cyl_between('paint_dark', fr.P(sc, troof[0] + 0.9, y0 + 2.66), fr.P(sc, troof[0] + 0.9, y0 + 2.75), 0.03, 6))
        # Fahrkartenautomat
        tm_t = (tw0 + tw1) / 2 if island else ((tw0 + tw1) / 2 + (0.45 if bk < 0 else -0.45))
        sm_ = sa - 1.6
        add(panel_box(fr, 'ticket_machine', 'paint_grey', sm_ - 0.25, sm_ + 0.25, tm_t - 0.36, tm_t + 0.36, y0, y0 + 1.80, 's0'), True)
        add(fr.box('paint_grey', sm_ - 0.3, sm_ + 0.3, tm_t - 0.42, tm_t + 0.42, y0 + 1.80, y0 + 1.86))
    # H-Mast an jedem Rampenkopf + Abfallbehaelter
    for se, sg in ((ramp0, 1), (ramp1, -1)):
        s_ = se + sg * 1.2; bs = back_side(s_)
        if bs is None: continue
        ta, tb, bk = bs; t_ = ((ta + tb) / 2) if island else (ta + 0.45 if bk < 0 else tb - 0.45)
        y0 = float(yt(s_))
        add(G2.cyl_between('paint_grey', fr.P(s_, t_, y0), fr.P(s_, t_, y0 + 3.2), 0.045, 8), False)
        add(panel_box(fr, 'hst_h', 'paint_grey', s_ - 0.02, s_ + 0.02, t_ - 0.30, t_ + 0.30, y0 + 2.55, y0 + 3.15, ('s0', 's1')))
        add(panel_box(fr, sign, 'paint_grey', s_ - 0.02, s_ + 0.02, t_ - 0.55, t_ + 0.55, y0 + 2.18, y0 + 2.46, ('s0', 's1')))
        coll.append(fr.box('collision', s_ - 0.06, s_ + 0.06, t_ - 0.06, t_ + 0.06, y0, y0 + 2.2))
        tb_ = t_ + 0.5 if island else t_ + (0.5 if bk < 0 else -0.5)
        add(G.cylinder('bin_grey', np.array([fr.xz(s_ + sg * 0.8, tb_)[0], y0, fr.xz(s_ + sg * 0.8, tb_)[1]]), 0.22, 0.22, 0.85, 12, cap=True), True)
    PLATFORMS.append(P)
    log('platform', nm, 'len %.1f' % Lp, 'ramps', ends, 'island', island, 'shelters', nsh, 'tris so far', G.tri_count(parts))

for P0 in L['platforms']:
    if not REG.buffer(-5).intersects(P0): continue
    build_platform(P0, osm_platform_name(P0))

# ---------------------------------------------------------------- Mendini-BUSSTOPS Steintor
def build_mendini(poly, roadside_hint=None):
    mrr = np.array(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = mrr[1] - mrr[0], mrr[2] - mrr[1]
    ax = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2; Lm = max(np.linalg.norm(e1), np.linalg.norm(e2)); ax = ax / np.linalg.norm(ax)
    if ax[0] < 0: ax = -ax                                       # s nach Osten
    fr = Frame(np.array(poly.centroid.coords[0]), ax)
    # Strassenseite (+t oder -t)?
    road_r = LOW.distance(Point(fr.xz(0, 3))) < LOW.distance(Point(fr.xz(0, -3)))
    sgn = 1 if road_r else -1                                     # Richtung zur Fahrbahn
    yb = gy(*fr.xz(0, 0)); half = Lm / 2
    BW = 1.6; B0, B1 = 2.55, 3.75                                 # Balken: Breite, Unter-/Oberkante
    # Schachbrett-Balken
    add(fr.box('checker_yb', -half, half, -BW / 2, BW / 2, yb + B0, yb + B1, uvscale=0.70))
    # Stuetzen (Schachbrett) alle ~4.6 m
    ncol = max(2, int(round(Lm / 4.6)) + 1)
    for s_ in np.linspace(-half + 1.0, half - 1.0, ncol)[1:]:
        add(fr.box('checker_yb', s_ - 0.2, s_ + 0.2, -0.2, 0.2, yb - 0.02, yb + B0, uvscale=0.70), True)
    # Kubus-Haeuschen am Westende mit goldenem Kegelturm
    ks0, ks1 = -half - 0.4, -half + 2.6
    add(fr.box('checker_yb', ks0, ks1, -1.25, 1.25, yb - 0.02, yb + 3.0, uvscale=0.70), True)
    add(fr.box('paint_dark', ks0 - 0.08, ks1 + 0.08, -1.33, 1.33, yb + 3.0, yb + 3.12))
    cc = fr.xz((ks0 + ks1) / 2, 0)
    add(G.cylinder('gold', (cc[0], yb + 3.12, cc[1]), 1.15, 0.02, 2.9, 24, cap=False))
    add(G.cylinder('gold', (cc[0], yb + 6.0, cc[1]), 0.08, 0.0, 0.5, 8, cap=False))
    # Tuer/Fenster (dunkel) in der Kubus-Stirnseite zur Haltestelle
    add(fr.vquad('black_glass', (ks1 + 0.005, -0.6), (ks1 + 0.005, 0.6), yb + 0.05, yb + 2.2, (ks1 + 2, 0)))
    # kleiner Goldkegel am Ostende auf dem Balken
    ce = fr.xz(half - 0.8, 0)
    add(G.cylinder('gold', (ce[0], yb + B1, ce[1]), 0.55, 0.02, 1.5, 18, cap=False))
    # Glas-Windschutz auf der fahrbahnfernen Seite + Bank
    tg = -sgn * 0.55
    add(fr.vquad('glass', (-half + 2.8, tg), (half - 0.6, tg), yb + 0.12, yb + 2.3, (0, tg - sgn * 3)))
    add(fr.box('paint_dark', -half + 2.8, half - 0.6, tg - 0.03, tg + 0.03, yb + 2.3, yb + 2.36))
    coll.append(fr.box('collision', -half + 2.8, half - 0.6, tg - 0.04, tg + 0.04, yb, yb + 2.3))
    tbn = tg + sgn * 0.45
    add(fr.box('wood', -half + 3.2, -half + 6.4, tbn - 0.2, tbn + 0.2, yb + 0.44, yb + 0.48), True)
    for e_ in (-half + 3.4, -half + 6.2): add(fr.box('paint_dark', e_ - 0.03, e_ + 0.03, tbn - 0.17, tbn + 0.17, yb, yb + 0.44))
    # Fahrplanvitrine + H-Mast an der Bordkante
    add(panel_box(fr, 'netmap', 'paint_dark', half - 2.2, half - 1.0, tg - 0.06, tg + 0.06, yb + 0.9, yb + 2.0, ('t0', 't1')))
    ts = sgn * 1.6; sh = half - 3.5
    add(G2.cyl_between('paint_grey', fr.P(sh, ts, yb), fr.P(sh, ts, yb + 3.2), 0.045, 8))
    add(panel_box(fr, 'hst_h', 'paint_grey', sh - 0.02, sh + 0.02, ts - 0.3, ts + 0.3, yb + 2.55, yb + 3.15, ('s0', 's1')))
    add(panel_box(fr, name_mat('Steintor'), 'paint_grey', sh - 0.02, sh + 0.02, ts - 0.55, ts + 0.55, yb + 2.18, yb + 2.46, ('s0', 's1')))
    log('mendini stop at', np.round(fr.o, 1), 'len %.1f' % Lm, 'road side', sgn)
    return poly.buffer(1.5)
MENDINI = []
for w in osm['ways']:
    t = w['tags']
    if t.get('name') == 'BUSSTOPS' and t.get('building') == 'roof' and len(w['pts']) >= 4:
        P0 = Polygon(w['pts']).buffer(0)
        if REG.buffer(-5).contains(P0.centroid): MENDINI.append(build_mendini(P0))
PLATFORMS += MENDINI

# ---------------------------------------------------------------- Oberleitung
OBST = unary_union([LOW.buffer(0.35), BLD.buffer(0.6), HOLES.buffer(0.6)] + [p.buffer(0.3) for p in PLATFORMS])
shapely.prepare(OBST)
def mast_ok(q):
    return shapely.contains_xy(REG, q[0], q[1]) and not shapely.contains_xy(OBST, q[0], q[1]) and not in_square(np.array([q[0]]), np.array([q[1]]), 3.0)[0]
def tangent(Ls, s):
    a = np.array(Ls.interpolate(max(s - 1.0, 0)).coords[0]); b = np.array(Ls.interpolate(min(s + 1.0, Ls.length)).coords[0])
    d = b - a; return d / (np.linalg.norm(d) + 1e-12)
stations = []; st_kd = None
support = {i: [] for i in range(len(LRL))}         # je Gleis: (s, Punkt3d)
WIRE_H = 5.6; SPAN_H = 6.5
nm_ = 0
for i, Ls in enumerate(LRL):
    for s in np.arange(4.0, Ls.length - 2.0, 30.0):
        p = np.array(Ls.interpolate(s).coords[0])
        if stations and np.min(np.hypot(*(np.array(stations) - p).T)) < 14.0: continue
        d = tangent(Ls, s); n = np.array([-d[1], d[0]])
        cross = LineString([p - n * 18, p + n * 18])
        offs = []
        for j, Lj in enumerate(LRL):
            x = cross.intersection(Lj)
            for q in getattr(x, 'geoms', [x]):
                if q.is_empty or q.geom_type != 'Point': continue
                qq = np.array(q.coords[0]); sj = Lj.project(Point(qq))
                if abs(np.dot(tangent(Lj, sj), d)) < 0.85: continue
                offs.append((float(np.dot(qq - p, n)), j, sj, qq))
        if not offs: continue
        offs.sort(key=lambda o: o[0])
        # nur unmittelbar benachbarte Gleise (Abstand < 6 m) als Gruppe
        grp = [o for o in offs if abs(o[0]) < 0.05]
        k0 = offs.index(grp[0]) if grp else 0
        lo = hi = k0
        while lo > 0 and offs[lo][0] - offs[lo - 1][0] < 6.0: lo -= 1
        while hi < len(offs) - 1 and offs[hi + 1][0] - offs[hi][0] < 6.0: hi += 1
        G_ = offs[lo:hi + 1]
        omin, omax = G_[0][0], G_[-1][0]
        yg = max(gy(*o[3]) for o in G_)
        left = right = None
        for o in np.arange(omin - 2.0, omin - 14.0, -0.25):
            q = p + n * o
            if mast_ok(q): left = q; break
        for o in np.arange(omax + 2.0, omax + 14.0, 0.25):
            q = p + n * o
            if mast_ok(q): right = q; break
        masts = [m for m in (left, right) if m is not None]
        if not masts:
            gap = [(G_[k + 1][0] - G_[k][0], k) for k in range(len(G_) - 1)]
            gap = [g for g in gap if g[0] > 3.4]
            if not gap: continue
            k = gap[0][1]; masts = [p + n * (G_[k][0] + G_[k + 1][0]) / 2]
        for m in masts:
            ym = gy(*m)
            add(G.cylinder('paint_grey', (m[0], ym - 0.05, m[1]), 0.13, 0.09, SPAN_H + 0.9 + (yg - ym) + 0.05, 10, cap=True))
            add(G.cylinder('paint_grey', (m[0], ym - 0.05, m[1]), 0.21, 0.21, 0.45, 12, cap=True))
            coll.append(G.box('collision', (m[0], ym, m[1]), (0.3, 3.0, 0.3)))
            nm_ += 1
        yspan = yg + SPAN_H
        if len(masts) == 2:
            a3 = np.array([masts[0][0], yspan, masts[0][1]]); b3 = np.array([masts[1][0], yspan, masts[1][1]])
            mid = (a3 + b3) / 2 + [0, -0.25, 0]
            add(G.tube('steel_dark', np.array([a3, mid, b3]), 0.008, 4, caps=False))
        else:
            m = masts[0]; ofs = [np.dot(o[3] - m, n) for o in G_]
            far = max(ofs, key=abs); dirn = np.sign(far) if far != 0 else 1.0
            a3 = np.array([m[0], yspan, m[1]]); b3 = a3 + np.array([n[0], 0, n[1]]) * (far + dirn * 0.6)
            add(G.tube('paint_grey', np.array([a3, b3]), 0.045, 6, caps=True))
            add(G.tube('paint_grey', np.array([a3 + [0, -0.9, 0], b3 + [0, -0.05, 0]]), 0.025, 5, caps=True))
        for k, (o, j, sj, qq) in enumerate(G_):
            zig = 0.2 * (1 if (len(support[j]) % 2) else -1)
            wp = np.array([qq[0] + n[0] * zig, gy(*qq) + WIRE_H, qq[1] + n[1] * zig])
            support[j].append((sj, wp))
            add(G.tube('steel_dark', np.array([wp, [wp[0], yspan - 0.15, wp[2]]]), 0.006, 4, caps=False))
        stations.append(p)
# Fahrdraht je Gleis zwischen den Stuetzpunkten (Durchhang 8 cm), Enden bis zum Linienende
nwire = 0
for j, Ls in enumerate(LRL):
    S = sorted(support[j], key=lambda a: a[0])
    if len(S) < 1: continue
    pts = []
    s_first = max(0.0, S[0][0] - 15.0); s_last = min(Ls.length, S[-1][0] + 15.0)
    q = np.array(Ls.interpolate(s_first).coords[0]); pts.append([q[0], gy(*q) + WIRE_H, q[1]])
    for k, (sj, wp) in enumerate(S):
        if k > 0:
            sm = (S[k - 1][0] + sj) / 2; q = np.array(Ls.interpolate(sm).coords[0])
            pm = (np.array(S[k - 1][1]) + wp) / 2
            pts.append([pm[0], gy(*q) + WIRE_H - 0.08, pm[2]])
        pts.append(list(wp))
    q = np.array(Ls.interpolate(s_last).coords[0]); pts.append([q[0], gy(*q) + WIRE_H, q[1]])
    add(G.tube('wire', np.array(pts), 0.0065, 4, caps=False)); nwire += 1
log('catenary stations', len(stations), 'masts', nm_, 'contact wires', nwire)

coll = [c for c in coll if c is not None]
pickle.dump({'parts': parts, 'coll': coll, 'walk': walk, 'mats': MATS_EXTRA,
             'info_extra': {'platforms': [[round(c, 2) for c in p.centroid.coords[0]] for p in PLATFORMS], 'catenary_stations': len(stations),
                            'platform_height': H_PLAT, 'mendini_busstops': len(MENDINI)},
             'occupied': [list(p.exterior.coords) for p in PLATFORMS]}, open(CACHE + 'transit.pkl', 'wb'))
log('done: tris', G.tri_count(parts), 'collision', G.tri_count(coll), 'walk', G.tri_count(walk))

"""Geländerecke Treppenauge (Detailmappe Bl. 2) -> A3-Blatt 1:5 mit Bemaßung nach DIN 406.

Aufruf: python -I make_a3.py <quelle.pdf> <ausgabe.pdf>
"""
import math
import sys

import pymupdf

SRC, OUT = sys.argv[1], sys.argv[2]
FONTS = "C:/Windows/Fonts/"

MM = 72 / 25.4            # pt je mm Papier
S5 = MM / 5               # pt je mm Natur bei 1:5

# --- Projektion der Vorlage (aus Vektordaten bestimmt) ---------------------
# Senkrechte Parallelprojektion, Blick 45° gedreht: x- und y-Achse unter
# +-28,23° zur Waagerechten, z senkrecht. Ursprung = Achse Pfosten L in Höhe
# oberer Bolzenachse (= Mitte Handlauf). Koordinaten in mm, Explosionszustand.
TH = math.radians(28.2315)
AC = S5 / math.sqrt(2)                       # a*s*cos(th)
AS = S5 * math.tan(TH) / math.sqrt(2)        # a*s*sin(th)
CS = S5 * math.sqrt(1 - math.tan(TH) ** 2)   # c*s
X0, Y0 = 518.966, 275.284                    # Ursprung im Quellblatt (pt)
DX, DY = -110.0, -120.0                      # Verschiebung Quelle -> Zielblatt


def P(x, y, z):
    """3D-Punkt (mm, Explosionszustand) -> Punkt auf dem Zielblatt (pt)."""
    return pymupdf.Point(X0 + (y - x) * AC + DX, Y0 - ((x + y) * AS + z * CS) + DY)


def T(p):
    return pymupdf.Point(p.x + DX, p.y + DY)


# Draufsicht: Ursprung = Außenecke Handlauf (u entlang Deckenrand, v entlang Lauf 3)
PU0, PV0 = 330 * MM, 118 * MM


def Q(u, v):
    return pymupdf.Point(PU0 + u * S5, PV0 - v * S5)


# --- Dokument --------------------------------------------------------------
src = pymupdf.open(SRC)
drs = src[1].get_drawings()

doc = pymupdf.open()
pg = doc.new_page(width=420 * MM, height=297 * MM)
pg.insert_font(fontname="ar", fontfile=FONTS + "arial.ttf")
pg.insert_font(fontname="ari", fontfile=FONTS + "ariali.ttf")
pg.insert_font(fontname="arb", fontfile=FONTS + "arialbd.ttf")
FNT = {"ar": pymupdf.Font(fontfile=FONTS + "arial.ttf"),
       "ari": pymupdf.Font(fontfile=FONTS + "ariali.ttf"),
       "arb": pymupdf.Font(fontfile=FONTS + "arialbd.ttf")}

INK = (0.106, 0.129, 0.122)
W_THIN = 0.18 * MM        # Maß-/Hilfslinien
W_TICK = 0.35 * MM        # Schrägstriche
W_MID = 0.35 * MM
W_WIDE = 0.5 * MM
FS = 9.6                  # Maßzahlen, Schrifthöhe ~2,5 mm
OVR = 2 * MM              # Überstand Maßhilfslinie
GAP = 0.8 * MM            # Abstand Maßhilfslinie zum Körper

# --- 1) Axonometrie der Vorlage unverändert (Faktor 1,0 = 1:5) übernehmen ---
sh = pg.new_shape()
for g in drs[1:195]:      # 0 = weißer Hintergrund, ab 195 Hinweislinien/Schriftfeld
    for it in g["items"]:
        k = it[0]
        if k == "l":
            sh.draw_line(T(it[1]), T(it[2]))
        elif k == "c":
            sh.draw_bezier(T(it[1]), T(it[2]), T(it[3]), T(it[4]))
        elif k == "re":
            sh.draw_rect(it[1] + (DX, DY, DX, DY))
        elif k == "qu":
            q = it[1]
            sh.draw_quad(pymupdf.Quad(T(q.ul), T(q.ur), T(q.ll), T(q.lr)))
    lc = g.get("lineCap") or (1,)
    sh.finish(color=g.get("color") if g["type"] != "f" else None,
              fill=g.get("fill"), width=g.get("width") or 0,
              closePath=g.get("closePath", False), even_odd=g.get("even_odd", False),
              lineCap=int(max(lc)), lineJoin=int(g.get("lineJoin") or 1))
sh.commit()


# --- Hilfsfunktionen Text / Maße ------------------------------------------
def text(p, s, fs=FS, font="ar", angle=0.0, anchor="c", color=INK):
    """Text an p; angle in Grad (gegen Uhrzeigersinn), anchor l/c/r auf Grundlinie."""
    w = FNT[font].text_length(s, fontsize=fs)
    dx = {"l": 0, "c": -w / 2, "r": -w}[anchor]
    a = math.radians(angle)
    start = pymupdf.Point(p.x + dx * math.cos(a), p.y - dx * math.sin(a))
    pg.insert_text(start, s, fontsize=fs, fontname=font, color=color,
                   morph=(start, pymupdf.Matrix(angle)) if angle else None)
    return w


def seg(a, b, w=W_THIN, dashes=None, color=INK):
    pg.draw_line(a, b, color=color, width=w, dashes=dashes, lineCap=0)


def unit(a, b):
    d = b - a
    n = math.hypot(d.x, d.y)
    return pymupdf.Point(d.x / n, d.y / n)


def tick(p, d):
    """Schrägstrich 45° zur Maßlinie (DIN 406-11), Länge ~3 mm."""
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    r = pymupdf.Point(d.x * c + d.y * s, -d.x * s + d.y * c)   # 45° gegen Uhrzeigersinn
    h = 1.5 * MM
    seg(p - r * h, p + r * h, w=W_TICK)


def ext(f, p):
    """Maßhilfslinie vom Körperpunkt f bis über den Maßlinienpunkt p hinaus."""
    if abs(f - p) < 1e-6:
        return
    e = unit(f, p)
    seg(f + e * GAP, p + e * OVR)


def chain(pts, feats, labels, italic=None, side=1, outside=None):
    """Maßkette entlang pts (Punkte auf der Maßlinie), feats = Körperpunkte.

    labels[i] gehört zum Abschnitt pts[i]..pts[i+1]. side=+1 Text links/oberhalb
    der Leserichtung. outside: Indizes, deren Text neben die Kette gesetzt wird.
    """
    italic = italic or [False] * len(labels)
    outside = outside or {}
    d = unit(pts[0], pts[-1])
    if d.x < -1e-9 or (abs(d.x) < 1e-9 and d.y > 0):   # Leserichtung von unten/rechts
        d = pymupdf.Point(-d.x, -d.y)
    ang = math.degrees(math.atan2(-d.y, d.x))
    nrm = pymupdf.Point(d.y, -d.x)                         # links der Leserichtung
    over = OVR if len(pts) else 0
    seg(pts[0] - unit(pts[0], pts[-1]) * over, pts[-1] + unit(pts[0], pts[-1]) * over)
    for p, f in zip(pts, feats):
        if f is not None:
            ext(f, p)
        tick(p, d)
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        m = (pts[i] + pts[i + 1]) * 0.5
        if i in outside:
            m = m + d * outside[i]
        q = m + nrm * (side * 1.0 * MM)
        if side < 0:
            q = m - nrm * (1.0 * MM + 0.72 * FS)
        text(q, lab, font="ari" if italic[i] else "ar", angle=ang)


def bubble(c, n, target):
    r = 2.6 * MM
    d = unit(c, target)
    seg(c + d * r, target, w=W_THIN)
    pg.draw_circle(target, 0.45 * MM, color=INK, fill=INK, width=0)
    pg.draw_circle(c, r, color=INK, width=W_THIN, fill=(1, 1, 1))
    text(pymupdf.Point(c.x, c.y + 0.36 * FS), str(n), fs=FS, font="ar")


# --- 2) Maßketten in der Axonometrie (Maßzahlen = wahre Längen) ------------
# Höhenkette Geländer links (senkrechte Maßlinie bei x=450, y=-310)
HZ = [45, -45, -365, -645]
FY = [-240, -240, -202, -202]          # Vorderflächen Handlauf / Brett (exploded)
chain([P(450, -310, z) for z in HZ], [P(450, fy, z) for fy, z in zip(FY, HZ)],
      ["90", "320", "280"], italic=[False, True, False])
chain([P(450, -360, HZ[0]), P(450, -360, HZ[-1])],
      [P(450, -310, HZ[0]), P(450, -310, HZ[-1])], ["690"], italic=[True])
# Handlaufbreite 60 und Brettdicke 22 (entlang y)
chain([P(450, -240, 85), P(450, -180, 85)], [P(450, -240, 45), P(450, -180, 45)], ["60"])
chain([P(450, -202, -330), P(450, -180, -330)], [P(450, -202, -365), P(450, -180, -365)],
      ["22"], outside={0: 7 * MM})
# Bolzenachsen (über die Hutmuttern, senkrechte Linie zwischen den Pfosten)
BZ = [0, -405, -605]
chain([P(0, 115, z) for z in BZ], [P(0, 97, z) for z in BZ], ["405", "200"],
      italic=[True, True])
# Handlauf Lauf 3: waagerechtes Stück bis Knick
chain([P(-330, -120, -95), P(-330, 230, -95)], [P(-330, -120, -45), P(-330, 230, -45)],
      ["350"], italic=[True])
# Neigung am geneigten Handlauf (Beschriftung parallel zur Oberkante)
k0 = P(-330, 230, 45)
k1 = P(-330, 736, 45 - 506 * 17.8 / 26)
sa = math.degrees(math.atan2(-(k1.y - k0.y), k1.x - k0.x))
text((k0 + k1) * 0.5 + pymupdf.Point(0, -7.0 * MM), "Neigung 17,8/26 (Laufneigung)", fs=8, angle=sa)

# Positionsnummern
bubble(T(pymupdf.Point(330, 222)), 1, T(pymupdf.Point(334.6, 277.2)))
bubble(T(pymupdf.Point(318, 606)), 2, T(pymupdf.Point(349.8, 541.6)))
bubble(T(pymupdf.Point(352, 668)), 3, T(pymupdf.Point(388.3, 634.7)))
bubble(T(pymupdf.Point(436, 432)), 4, T(pymupdf.Point(470.9, 487.6)))
bubble(T(pymupdf.Point(648, 438)), 5, T(pymupdf.Point(591.1, 420.2)))
bubble(T(pymupdf.Point(555, 214)), 6, T(pymupdf.Point(551.0, 246.6)))
bubble(T(pymupdf.Point(642, 262)), 7, T(pymupdf.Point(595.1, 346.8)))

# --- 3) Draufsicht Einbauzustand ------------------------------------------
GREY = (0.82, 0.82, 0.82)


def poly(pts, w=W_MID, fill=None, closed=True, dashes=None):
    pg.draw_polyline([Q(*p) for p in pts] + ([Q(*pts[0])] if closed else []),
                     color=INK, fill=fill, width=w, dashes=dashes, closePath=closed)


def brk(a, b, n=1):
    """Bruchlinie (Zickzack) von a nach b in u/v."""
    (u1, v1), (u2, v2) = a, b
    pts = [a]
    for t, o in ((0.42, 0), (0.47, 8), (0.53, -8), (0.58, 0)):
        u = u1 + (u2 - u1) * t
        v = v1 + (v2 - v1) * t
        if u1 == u2:
            u += o
        else:
            v += o
        pts.append((u, v))
    pts.append(b)
    pg.draw_polyline([Q(*p) for p in pts], color=INK, width=W_THIN)


UL, VL = 320, 440     # Bruchkanten
# Handläufe (Oberseite)
pg.draw_polyline([Q(320, 0), Q(60, 0), Q(60, 60), Q(320, 60)], color=INK, width=W_WIDE)
pg.draw_polyline([Q(60, 0), Q(0, 0), Q(0, VL)], color=INK, width=W_WIDE)
pg.draw_polyline([Q(60, VL), Q(60, 60)], color=INK, width=W_WIDE)
brk((UL, 0), (UL, 60))
brk((0, VL), (60, VL))
seg(Q(0, 350), Q(60, 350), w=W_THIN)                       # Knick Laufneigung
# Brett darunter (verdeckt)
DASH = "[3 2] 0"
seg(Q(60, 38), Q(UL, 38), w=W_THIN, dashes=DASH)
seg(Q(38, 38), Q(38, VL), w=W_THIN, dashes=DASH)
seg(Q(38, 38), Q(60, 38), w=W_THIN, dashes=DASH)
# Pfosten
for cu, cv in ((210, 120), (120, 210)):
    pg.draw_rect(pymupdf.Rect(Q(cu - 20, cv + 20), Q(cu + 20, cv - 20)),
                 color=INK, fill=GREY, width=W_WIDE)


def bolt(axis):
    """Bolzen (Draufsicht): axis='v' -> Bolzen in v-Richtung bei u=210."""
    parts = [((60, 63), 18, None), ((63, 100), 8, None), ((140, 150), 9.5, None)]
    for (a, b), r, _ in parts:
        pts = [(210 - r, a), (210 + r, a), (210 + r, b), (210 - r, b)]
        if axis == "u":
            pts = [(v, u) for u, v in pts]
        poly(pts, w=W_THIN, fill=(1, 1, 1))
    dome = [(210 - 9.5, 150), (210 - 6, 154), (210, 155.5), (210 + 6, 154), (210 + 9.5, 150)]
    if axis == "u":
        dome = [(v, u) for u, v in dome]
    poly(dome, w=W_THIN, closed=False)
    sh_pts = [((210 - 6, 100), (210 - 6, 140)), ((210 + 6, 100), (210 + 6, 140)),
              ((210 - 6, 20), (210 - 6, 60)), ((210 + 6, 20), (210 + 6, 60))]
    for a, b in sh_pts:
        if axis == "u":
            a, b = a[::-1], b[::-1]
        seg(Q(*a), Q(*b), w=W_THIN, dashes=DASH)
    a, b = (210, -8), (210, 168)
    if axis == "u":
        a, b = a[::-1], b[::-1]
    seg(Q(*a), Q(*b), w=W_THIN, dashes="[9 2 1.5 2] 0")


bolt("v")
bolt("u")
# Pfostenachsen als Maßhilfslinien
seg(Q(210, -8), Q(210, -88), w=W_THIN, dashes="[9 2 1.5 2] 0")
seg(Q(-8, 210), Q(-88, 210), w=W_THIN, dashes="[9 2 1.5 2] 0")
seg(Q(120, 168), Q(120, -88), w=W_THIN, dashes="[9 2 1.5 2] 0")
seg(Q(168, 120), Q(-88, 120), w=W_THIN, dashes="[9 2 1.5 2] 0")

# Maßketten Draufsicht
U1 = [0, 60, 100, 140, 190, 230]
UF = [(0, 0), (60, 0), (100, 190), (140, 190), (190, 100), (230, 100)]
chain([Q(u, -35) for u in U1], [Q(*f) for f in UF], ["60", "40", "40", "50", "40"],
      italic=[False, True, False, True, False], side=-1)
chain([Q(u, -80) for u in (0, 120, 210)], [Q(0, -35), None, None], ["120", "90"],
      italic=[True, True], side=-1)
V1 = [0, 60, 100, 140, 190, 230]
VF = [(0, 0), (60, 60), (190, 100), (190, 140), (100, 190), (100, 230)]
chain([Q(-35, v) for v in V1], [Q(*f) for f in VF], ["60", "40", "40", "50", "40"],
      italic=[False, True, False, True, False])
chain([Q(-80, v) for v in (0, 120, 210, 350)], [Q(-35, 0), None, None, Q(0, 350)],
      ["120", "90", "140"], italic=[True, True, True])
chain([Q(345, v) for v in (0, 38, 60)], [Q(UL + 6, 0), Q(UL + 4, 38), Q(UL + 6, 60)],
      ["38", "22"], italic=[True, False], side=-1, outside={0: -6.5 * MM})

text(Q(70, 356), "Knick Laufneigung 17,8/26", fs=7.5, anchor="l")
text(Q(70, 410), "Lauf 3 abwärts", fs=7.5, anchor="l")
text(Q(70, 395), "Handlauf + Brett laufen durch", fs=7.5, anchor="l")
text(Q(255, 300), "Treppenauge", fs=7.5, anchor="c")
text(Q(236, 82), "Deckenrand OG1", fs=7.5, anchor="l")
text(Q(236, 68), "stößt stumpf an", fs=7.5, anchor="l")

# --- 4) Ansichtstitel ------------------------------------------------------
text(pymupdf.Point(32 * MM, 202 * MM), "Axonometrie (Explosion)  M 1:5", fs=11, font="arb", anchor="l")
text(pymupdf.Point(32 * MM, 206.5 * MM),
     "Ecke Deckenrand OG1 / Lauf 3, Blick von der Treppe – Maßzahlen = wahre Längen",
     fs=7.5, anchor="l")
text(pymupdf.Point(302 * MM, 146 * MM), "Draufsicht Einbauzustand  M 1:5", fs=11, font="arb", anchor="l")
text(pymupdf.Point(302 * MM, 150.5 * MM), "Bolzen in Höhe Handlaufmitte, Brett verdeckt",
     fs=7.5, anchor="l")

# --- 5) Rahmen, Legende, Maßstabswahl, Schriftfeld --------------------------
pg.draw_rect(pymupdf.Rect(20 * MM, 10 * MM, 410 * MM, 287 * MM), color=INK, width=W_WIDE)


def lines(x, y, rows, fs=7.5, lh=1.32):
    for r in rows:
        font = "ar"
        if isinstance(r, tuple):
            r, font = r
        text(pymupdf.Point(x, y), r, fs=fs, font=font, anchor="l")
        y += fs * lh
    return y


y = lines(25 * MM, 214 * MM, [("Positionen", "arb"),
                              "1  Handlauf Teak 60/90",
                              "2  Brüstungsbrett Teak 280/22",
                              "3  Linsenkopfschraube M12, Edelstahl",
                              "4  Rosette Ø36 + Distanzhülse Ø16, Edelstahl",
                              "5  Pfosten Stahl □ 40/40, schwarz, paarweise",
                              "6  Hutmutter M12, Edelstahl",
                              "7  Eckstoß stumpf: Brett + Handlauf Lauf 3 laufen durch"])
lines(118 * MM, 214 * MM, [("Hinweise", "arb"),
                           "Maße in mm. Maßzahlen gerade = in der Vorlage beschriftet,",
                           ("kursiv = aus der PDF-Vektorgeometrie gemessen/abgeleitet.", "ari"),
                           "Axonometrie: senkr. Parallelprojektion, 45° gedreht; Achsen x/y",
                           "unter 28,2°, Verkürzung x/y 0,803, z 0,844 (1:5 in der Bildebene).",
                           "Explosion: Holzteile 120 mm von den Pfosten abgerückt.",
                           "Pfosten oben/unten gekürzt (dargestellt 1000). Kanten scharf,",
                           "keine Radien. Hülse/Rosette in der Vorlage angenommen."])

y = lines(302 * MM, 159 * MM, [("Maßstabswahl", "arb"),
                               "Detail in der Bildebene (gemessen): 1245 × 875 mm",
                               "Zeichenfläche A3: 390 × 277 mm (Rand 20/10)",
                               "– Schriftfeld/Draufsicht: frei ≈ 270 × 185 mm",
                               "1:2    →  623 × 438 mm   passt nicht",
                               "1:2,5  →  498 × 350 mm   passt nicht",
                               "1:5    →  249 × 175 mm   passt (+ Maßketten",
                               "              ≈ 262 × 182 mm)",
                               "1:10   →  125 ×  88 mm   passt, aber Bauteile",
                               "              22 mm → 2,2 mm, zu klein",
                               ("Gewählt: 1:5 (größter passender Normmaßstab)", "arb")])

# Maßstabsleiste 0..200 mm bei 1:5
x0, y0 = 302 * MM, 229 * MM
for i in range(4):
    pg.draw_rect(pymupdf.Rect(x0 + i * 50 * S5, y0, x0 + (i + 1) * 50 * S5, y0 + 1.2 * MM),
                 color=INK, fill=INK if i % 2 == 0 else (1, 1, 1), width=W_THIN)
    text(pymupdf.Point(x0 + i * 50 * S5, y0 - 1 * MM), str(i * 50), fs=6.5)
text(pymupdf.Point(x0 + 200 * S5, y0 - 1 * MM), "200 mm", fs=6.5, anchor="l")

# Schriftfeld 180 × 50 mm unten rechts
L, R, TOP, BOT = 230 * MM, 410 * MM, 237 * MM, 287 * MM
pg.draw_rect(pymupdf.Rect(L, TOP, R, BOT), color=INK, width=W_WIDE)
for yy in (257, 272):
    seg(pymupdf.Point(L, yy * MM), pymupdf.Point(R, yy * MM), w=W_MID)
for xx in (290, 350):
    seg(pymupdf.Point(xx * MM, 257 * MM), pymupdf.Point(xx * MM, BOT), w=W_MID)
text(pymupdf.Point(233 * MM, 246 * MM), "Geländerecke Treppenauge – Bemaßung", fs=14, font="arb", anchor="l")
text(pymupdf.Point(233 * MM, 252.5 * MM),
     "Ecke Deckenrand OG1 / Lauf 3 · Grundlage: Detailmappe Treppe Kern 335, Bl. 2",
     fs=7.5, anchor="l")
cells = [(233, 261, "Projekt", "Bachelorthesis „Der Anfang“"),
         (293, 261, "Ort", "TIB, Treppenkern 335"),
         (353, 261, "Vorlage", "Jan Andis Böhm"),
         (233, 276, "Maßstab", "1:5"),
         (293, 276, "Format", "A3 (420 × 297)"),
         (353, 276, "Datum", "25.10.2026")]
for cx, cy, k, v in cells:
    text(pymupdf.Point(cx * MM, cy * MM), k, fs=6, anchor="l")
    text(pymupdf.Point(cx * MM, (cy + 6.5) * MM), v, fs=10 if k == "Maßstab" else 8.5,
         font="arb" if k == "Maßstab" else "ar", anchor="l")

doc.set_metadata({"title": "Geländerecke Treppenauge – Bemaßung 1:5 (A3)",
                  "subject": "Treppe Kern 335, Detailmappe Bl. 2, bemaßt nach DIN 406"})
doc.subset_fonts()
doc.save(OUT, garbage=3, deflate=True)
print("ok", OUT)

# Kontrolle: Ausdehnung der übernommenen Axonometrie
bb = pymupdf.Rect()
for g in drs[1:195]:
    bb |= g["rect"]
print("Axo-Bbox Papier %.1f x %.1f mm -> Natur (Bildebene) %.0f x %.0f mm"
      % (bb.width / MM, bb.height / MM, bb.width / S5, bb.height / S5))

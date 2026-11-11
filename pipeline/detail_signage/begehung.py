"""Append (or refresh) the detail_signage part of section 'Ergänzungen aus dem Bau' in BEGEHUNGSLISTE.md:
places where the overlay had no visual information (sign looks guessed, unknown shops behind glass, shops
that found no shopfront in the detail model).  Only the block between the markers below is ever rewritten.

  python begehung.py [--dry]
"""
import os
import re
import sys

import numpy as np

from common import CACHE, DATA, ROOT, jload, load_shops, log

MD = os.path.join(ROOT, "BEGEHUNGSLISTE.md")
HEAD = "## Ergänzungen aus dem Bau"
START = "<!-- detail_signage:start -->"
END = "<!-- detail_signage:end -->"


def centre(o):
    O, R = np.array(o["O"]), np.array(o["R"])
    g = o["glass"]["rect"]
    return O + R @ np.array([(g[0] + g[1]) / 2, (g[2] + g[3]) / 2, 0.0])


def main():
    OPS = jload(os.path.join(CACHE, "openings.json"))["openings"]
    AS = jload(os.path.join(CACHE, "assign.json"))
    A = AS["opening_shop"]
    SH = {s["id"]: s for s in load_shops()}
    ST = jload(os.path.join(CACHE, "build_stats.json"))
    by_shop = {}
    for o in OPS:
        if o["id"] in A:
            by_shop.setdefault(A[o["id"]], []).append(o)
    lines = [START, "", "### Detailmodell Opernplatz: Ladenschilder und Schaufenster (Overlay `detail/signage.glb`)", "",
             "Gebaut aus `data/shops.json` + OSM; das Aussehen der Schilder ist bei den meisten Läden geschätzt "
             "(Markenfarben aus Wissen oder Platzhalter). Vor Ort bitte je Laden: Schild frontal (Schrift, Farben, "
             "Schildtyp: Blende / Einzelbuchstaben / Leuchtkasten / Ausleger), ein Schaufenster schräg (Tiefe, "
             "Rückwand, Deko), Tür mit Aufklebern, Hausnummernschild.", ""]
    # 1. placed shops whose sign data is a guess
    rows = []
    for sid, ops in by_shop.items():
        s = SH[sid]
        sg = s.get("sign") or {}
        if sg.get("source") in ("streetview_note_2022",) and sg.get("confidence") not in ("low",):
            continue
        c = np.mean([centre(o) for o in ops], 0)
        rows.append((s.get("address") or "", s.get("name") or s.get("former_name") or "(ohne Namen)",
                     sg.get("type") or "-", sg.get("source") or "-", c[0], c[2], len(ops)))
    rows.sort()
    lines += ["#### Schild und Schaufenster geschätzt (%d Läden im Detailmodell)" % len(rows), "",
              "| ✓ | Adresse | Name | Schildtyp (angenommen) | Datenquelle | Fenster | x | z |",
              "|---|---|---|---|---|---:|---:|---:|"]
    for a, n, t, src, x, z, k in rows:
        lines.append("| ☐ | %s | %s | %s | %s | %d | %.0f | %.0f |" % (a, n, t, src, k, x, z))
    # 2. openings without a known shop
    gen = {}
    for o in OPS:
        if o["id"] in A or o["kind"] == "entrance":
            continue
        lod = o.get("lod2") or {}
        key = "%s %s" % (lod.get("street") or "?", lod.get("number") or "")
        gen.setdefault(key.strip(), []).append(centre(o))
    lines += ["", "#### Schaufenster ohne bekannten Laden (Platzhalter-Auslage, kein Name)", "",
              "Welcher Laden / welche Nutzung steckt hinter diesen Fenstern (oder leer / Büro)?", "",
              "| ✓ | Adresse (LoD2) | Fenster | x | z |", "|---|---|---:|---:|---:|"]
    for k in sorted(gen):
        c = np.mean(gen[k], 0)
        lines.append("| ☐ | %s | %d | %.0f | %.0f |" % (k, len(gen[k]), c[0], c[2]))
    # 3. shops near the model that found no shopfront
    lost = []
    placed = set(AS["shops_placed"])
    mask = np.load(os.path.join(DATA, "detail_mask.npy"))
    from openings import in_landmark, load_landmark_polys
    polys = load_landmark_polys()
    wall_only = set(ST["stats"].get("wall_only") or [])
    shared = {n for names in (ST["stats"].get("shared_fronts") or {}).values() for n in names}
    for s in SH.values():
        if s["id"] in placed or s.get("level", 0) != 0 or s.get("context") not in ("street", "passage"):
            continue
        f = s.get("facade") or {}
        wp = f.get("wall_point") or s["p"]
        j, i = int(np.floor(wp[0] + 400)), int(np.floor(wp[1] + 400))
        if not (0 <= i < 800 and 0 <= j < 800) or not mask[max(0, i - 6):i + 7, max(0, j - 6):j + 7].any():
            continue
        lm = in_landmark(polys, np.array([wp[0], 1.0, wp[1]]), buf=1.5)
        nm = s.get("name") or s.get("former_name") or "(ohne Namen)"
        if lm:
            how = "in Landmarke %s (dort modelliert)" % lm.replace(".json", "")
        elif s.get("name") in shared:
            how = "teilt sich eine durchgehende Glasfront: eigener Auslagenabschnitt + Schild im gemeinsamen Schildband"
        elif s.get("name") in wall_only:
            how = "nur Schild an der Wand, Fenster gehören Nachbarn/keine Ladenfront"
        elif not (0 <= i < 800 and 0 <= j < 800) or not mask[i, j]:
            how = "knapp außerhalb des Detailmodells"
        else:
            how = "kein Schild (keine freie Wandfläche / geschlossen / ohne Namen)"
        lost.append((s.get("address") or "", nm, s.get("status"), how, wp[0], wp[1]))
    lost.sort()
    lines += ["", "#### Läden aus der Liste ohne passende Ladenfront im Detailmodell", "",
              "Liegen in umgebauten Landmarken (Kröpcke-Center, Café Kröpcke), teilen sich eine Front mit "
              "Nachbarn oder die OSM-Position passt nicht. Vor Ort: Lage der Tür, Breite der Front und Schild prüfen.", "",
              "| ✓ | Adresse | Name | Status | im Overlay | x | z |", "|---|---|---|---|---|---:|---:|"]
    for a, n, st, how, x, z in lost:
        lines.append("| ☐ | %s | %s | %s | %s | %.0f | %.0f |" % (a, n, st, how, x, z))
    # 4. general gaps
    shallow = [o for o in OPS if o["box"]["depth"] < 0.08]
    lines += ["", "#### Allgemein", "",
              "- ☐ Schaufenstertiefe: die Rhino-Ladenfronten haben eine Innenplatte ~0,56 m hinter dem Glas, die "
              "Auslagen sind deshalb flache Vitrinen. Echte Tiefe und Rückwand je Straße an 2-3 Beispielen schräg "
              "fotografieren (Georgstraße 36-50, Karmarschstraße, Luisenstraße, Rathenaustraße).",
              "- ☐ %d Glasflächen haben weniger als 8 cm bis zur dunklen Innenplatte (nur Folie/Plakat möglich), "
              "z. B. Osterstraße 31, Windmühlenstraße 1D: was ist dort wirklich hinter dem Glas?" % len(shallow),
              "- ☐ Hausnummernschilder (%d gesetzt, Stil blau emailliert / Edelstahl geraten) und Klingeltableaus "
              "(%d) an den Hauseingängen: Stil je Haus fotografieren." % (ST["stats"]["numbers"], ST["stats"]["bells"]),
              "- ☐ Ausleger-Schilder (%d gesetzt, v. a. Apotheken/Gastronomie): Position und Form prüfen."
              % ST["stats"]["blades"],
              "- ☐ Schilder nach den Street-View-Notizen (Stand Aug. 2022, `pipeline/detail_signage/sign_overrides.json` "
              "und shops.json-Quelle streetview_note_2022): %s - aktuellen Zustand (Farben, Schildtyp) bestätigen."
              % ", ".join(sorted({s["name"] for s in SH.values() if s["id"] in by_shop and
                                  "streetview_note_2022" in ((s.get("sign") or {}).get("source") or "")})),
              "- ☐ Eckschaufenster mit Glas-an-Glas-Ecke (enercity expo Café Ständehausstraße 6 bei x 67 z 50, "
              "x 78 z 51, Bockhorst Rathenaustraße 22 bei x 83 z 26, Ecke x 90 z 44, x 95 z 37): hinter der "
              "Ecke steht im Modell ein Pfosten/die Innenplatte der Nachbarfassade, die Auslage ist dort als zwei "
              "auf Gehrung geschnittene Eckboxen geraten - Eckauslage und Pfosten von beiden Straßen aus fotografieren.",
              "- ☐ Osterstraße 29-31: eine durchgehende Glasfront im Modell, laut Liste mehrere Läden (Miss Lashes, "
              "Stichnoth, Lia's, Wömpner, Petit Bateau) - Aufteilung der Front und Schilder vor Ort klären.",
              "", END]
    block = "\n".join(lines) + "\n"
    if "--dry" in sys.argv:
        print(block)
        return
    txt = open(MD, encoding="utf-8").read()
    if START in txt and END in txt:
        pre = txt[:txt.index(START)]
        post = txt[txt.index(END) + len(END):].lstrip("\n")
        txt = pre + block + ("\n" + post if post else "")
    else:
        if HEAD not in txt:
            txt = txt.rstrip("\n") + "\n\n" + HEAD + "\n\n"
        else:
            txt = txt.rstrip("\n") + "\n\n"
        txt += block
    tmp = MD + ".detail_signage.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(txt)
    os.replace(tmp, MD)
    log("BEGEHUNGSLISTE: %d guessed signs, %d unknown addresses, %d unplaced shops" % (len(rows), len(gen), len(lost)))


if __name__ == "__main__":
    main()

// Bahnhaltestellen fürs Spiel: Name, Kartenpunkt, Zielpunkt (Ankunft) und die gebauten Eingänge (aus der Bodenpipeline).
//   node pipeline/karte/build_stations.mjs   -> web/assets/city/karte/stationen.json
// Spielrahmen: Meter, Ursprung Kröpcke-Uhr (EPSG:25832 E 550298.88 N 5802940.71), +X Ost, -Z Nord.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

// WGS84 -> UTM Zone 32N (Krüger-Reihe, mm-genau genug)
function utm(lat, lon) {
  const a = 6378137, f = 1 / 298.257223563, k0 = 0.9996, lon0 = 9 * Math.PI / 180;
  const n = f / (2 - f), A = a / (1 + n) * (1 + n * n / 4 + n ** 4 / 64);
  const al = [n / 2 - 2 * n * n / 3 + 5 * n ** 3 / 16, 13 * n * n / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240];
  const p = lat * Math.PI / 180, l = lon * Math.PI / 180 - lon0;
  const t = Math.sinh(Math.atanh(Math.sin(p)) - 2 * Math.sqrt(n) / (1 + n) * Math.atanh(2 * Math.sqrt(n) / (1 + n) * Math.sin(p)));
  const xi = Math.atan(t / Math.cos(l)), eta = Math.atanh(Math.sin(l) / Math.sqrt(1 + t * t));
  let E = eta, N = xi;
  for (let j = 1; j <= 3; j++) { E += al[j - 1] * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta); N += al[j - 1] * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta); }
  return [500000 + k0 * A * E, k0 * A * N];
}
const game = (lat, lon) => { const [E, N] = utm(lat, lon); return [+(E - 550298.88).toFixed(2), +(-(N - 5802940.71)).toFixed(2)]; };

// gebaute Eingänge (pipeline/ground/ubahn.py INFO; top = Oberkante Mitte, dir = Richtung hinunter, y_top = Höhe oben)
const E = (key, station, top, dir, y_top, ref = '') => ({ key, station, ref, top, dir, y_top });
const eingaenge = [
  E('kroepcke_A', 'Kröpcke', [-46.63, 51.65], [0.27, -0.96], 0.14, 'A'), E('kroepcke_B', 'Kröpcke', [-52.49, -12.88], [0.95, 0.30], 0.24, 'B'),
  E('kroepcke_C', 'Kröpcke', [-48.11, -28.12], [0.96, 0.29], -0.04, 'C'), E('kroepcke_D', 'Kröpcke', [18.19, 44.97], [-0.48, -0.88], 0.03, 'D'),
  E('kroepcke_E', 'Kröpcke', [101.68, 211.79], [-0.51, -0.86], -0.45, 'E'), E('kroepcke_F', 'Kröpcke', [93.38, 218.35], [-0.49, -0.87], -0.52, 'F'),
  E('kroepcke_G', 'Kröpcke', [-64.52, 162.38], [0.26, -0.97], -0.42, 'G'), E('kroepcke_H', 'Kröpcke', [-78.22, 158.76], [0.25, -0.97], -0.52, 'H'),
  E('kroepcke_I', 'Kröpcke', [-67.64, 117.84], [-0.25, 0.97], -0.22, 'I'),
  E('ub_299064939', 'Waterlooplatz', [-531.6, 633.53], [0.89, -0.45], -3.99), E('ub_324517470', 'Aegidientorplatz', [347.02, 570.03], [-0.23, 0.97], -1.86),
  E('ub_1274066800', 'Steintor', [-502.7, -141.57], [0.43, -0.90], -0.94), E('ub_1274066802', 'Steintor', [-390.8, -123.25], [-0.95, -0.31], -1.04),
  E('ub_1396666085', 'Steintor', [-357.06, -135.33], [-0.99, -0.10], -1.08), E('ub_1396666113', 'Steintor', [-377.41, -150.75], [-0.95, -0.31], -1.06),
  E('ub_2747495414', 'Hauptbahnhof', [306.9, -477.11], [-0.44, 0.90], -2.6), E('ub_4048984722', 'Hauptbahnhof', [182.79, -278.21], [0.61, -0.79], 2.28),
  E('ub_4048984735', 'Hauptbahnhof', [202.65, -305.05], [-0.60, 0.80], 2.22),
  E('ub_4373301942', 'Markthalle/Landtag', [-229.26, 466.54], [0.70, -0.71], -0.70), E('ub_4373301943', 'Markthalle/Landtag', [-237.85, 457.6], [0.72, -0.69], -0.72),
];
// Passerelle (Niki-de-Saint-Phalle-Promenade): Achse aus data/ground_cache/promenade.json, s nach Nordost zum Hauptbahnhof
const PR = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/ground_cache/promenade.json'), 'utf8')).axis;
const sMax = Math.max(...JSON.parse(fs.readFileSync(path.join(ROOT, 'data/ground_cache/promenade.json'), 'utf8')).openings.map((o) => o.s[1])) + 12;
const passerelle = { origin: PR.origin, axis: PR.axis, normal: PR.normal, s_ende: +sMax.toFixed(1), breite: 9 };

// Haltestellen (Kartenpunkt = OSM-Haltestelle; Ankunft = oben an einem Eingang, mit dem Rücken zur Treppe)
const osm = { 'Kröpcke': [52.3738655, 9.7383552], 'Hauptbahnhof': [52.3769504, 9.7416533], 'Steintor': [52.3758039, 9.7326886],
  'Markthalle/Landtag': [52.3708584, 9.7362366], 'Aegidientorplatz': [52.3689, 9.7432], 'Waterlooplatz': [52.3680475, 9.7296964] };
const linien = { 'Kröpcke': 'U1 U2 U3 U4 U5 U6 U7 U8 U9 U10 U11 U17', 'Hauptbahnhof': 'U1 U2 U8 U10 U17 · S-Bahn · Fernverkehr',
  'Steintor': 'U4 U5 U6 U10 U11 U17', 'Markthalle/Landtag': 'U3 U7 U9', 'Aegidientorplatz': 'U1 U2 U4 U5 U6 U8 U11', 'Waterlooplatz': 'U3 U7 U9' };
const stationen = Object.entries(osm).map(([name, [la, lo]]) => {
  const ein = eingaenge.filter((e) => e.station === name);
  let ankunft;
  if (name === 'Kröpcke') ankunft = { x: 2.0, z: 6.0, yaw: 180 };                     // an der Kröpcke-Uhr
  else if (name === 'Hauptbahnhof') { ankunft = { x: 162.4, z: -181.2, yaw: 143 };   // Ernst-August-Platz, Blick auf die Bahnhofsfront
  } else {
    const e = ein[0]; const x = e.top[0] - e.dir[0] * 3.5, z = e.top[1] - e.dir[1] * 3.5;
    ankunft = { x: +x.toFixed(2), z: +z.toFixed(2), yaw: +(Math.atan2(-e.dir[0], -e.dir[1]) * 180 / Math.PI).toFixed(1) };   // Spieler blickt entlang (sin yaw, cos yaw)
  }
  return { name, punkt: game(la, lo), linien: linien[name], ankunft, eingaenge: ein.map((e) => e.key) };
});
const out = { quelle: 'pipeline/karte/build_stations.mjs (OSM + pipeline/ground/ubahn.py)', stationen, eingaenge, passerelle };
fs.mkdirSync(path.join(ROOT, 'web/assets/city/karte'), { recursive: true });
fs.writeFileSync(path.join(ROOT, 'web/assets/city/karte/stationen.json'), JSON.stringify(out, null, 1));
for (const s of stationen) console.log(s.name.padEnd(20), s.punkt, JSON.stringify(s.ankunft), s.eingaenge.length);
console.log('Passerelle', JSON.stringify(passerelle));

// Eigene Scans (Gaussian Splats) in der Stadt platzieren -- Schnittstelle für die spätere Ergänzung mit selbst
// gescannten Objekten/Texturen. Einträge in assets/city/manifest.json:
//   "scans": [{ "url": "scans/bank.ply", "position": [x, y, z], "rotationDeg": [rx, ry, rz], "scale": 1,
//               "maxSh": 1, "flipY": true, "name": "Bank am Kröpcke" }]
// Spielframe: Meter, Y oben, Ursprung Kröpcke-Uhr, +X Ost, -Z Nord. Viele Scan-Apps (Scaniverse, Polycam, ...)
// exportieren PLY mit Y nach unten -> "flipY": true dreht 180° um X (Standard).
// Gerendert im Splat-Durchgang (Layer 1, tiefengeprüft gegen die Stadt) wie Carlos.
import * as THREE from 'three';
import { SplatMesh } from '@sparkjsdev/spark';

export async function loadScans(scene, base = 'assets/city/', status = () => {}) {
  let man;
  try { const r = await fetch(base + 'manifest.json', { cache: 'no-store' }); if (!r.ok) return []; man = await r.json(); } catch { return []; }
  const list = man.scans || [];
  const out = [];
  for (const [i, s] of list.entries()) {
    try {
      status(`Lade Scan ${i + 1}/${list.length} …`);
      const m = new SplatMesh({ url: base + s.url });
      await m.initialized;
      m.maxSh = s.maxSh ?? 1;
      const r = (s.rotationDeg || [0, 0, 0]).map(THREE.MathUtils.degToRad);
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(r[0], r[1], r[2], 'YXZ'));
      if (s.flipY !== false) q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI));
      m.quaternion.copy(q);
      m.position.fromArray(s.position || [0, 0, 0]);
      m.scale.setScalar(s.scale ?? 1);
      m.name = s.name || s.url;
      m.layers.set(1);
      scene.add(m); out.push(m);
    } catch (e) { console.warn('Scan konnte nicht geladen werden:', s.url, e); }
  }
  return out;
}

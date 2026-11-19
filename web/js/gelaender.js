// Treppengeländer in der Passerelle als Interaktion: unsichtbar, bis man sie anvisiert (hell leuchtend) oder mit F scannt
// (die Geländer-Punkte glühen blau, solange der Scan im Gedächtnis ist). Anvisieren + klicken -> die Figur setzt sich auf den Handlauf und rutscht
// auf dem Hintern hinunter (Vorbilder: Geländer-Rutschen in "Mirror's Edge"/"Dying Light"-Parkour, Sitz-Rutschen wie im
// Referenzvideo "Guy Slides Down A Handrail On His Butt"): seitlich aufsitzen, Beine zur Treppe, Hände links/rechts am Rohr,
// Schwerkraft beschleunigt (Reibung bremst), Funken-Glühen am Kontaktpunkt, unten abspringen und abfedern.
// Geometrie: gelaender.json enthält exakt die Polylinien der echten Edelstahl-Handläufe aus promenade.py (inkl. Zwischenpodest),
// das Leuchten sitzt also direkt auf dem Rohr.
import * as THREE from 'three';

const G = 9.81, REIBUNG = 0.22, MAX_V = 7.5, AUFSITZ_R = 2.6, ZIEL_R = 7;

export class Gelaender {
  constructor({ world, scene, camera, hud, base = 'assets/city/' }) {
    this.world = world; this.scene = scene; this.camera = camera; this.hud = hud; this.base = base + 'karte/';
    this.rails = []; this.aktiv = null; this.ziel = null; this.t = 0;
    this.group = new THREE.Group(); this.group.name = 'Leuchtgelaender'; scene.add(this.group);
  }
  async load() {
    const j = await (await fetch(this.base + 'gelaender.json')).json();
    const R = (j.rohr_radius || 0.024) + 0.006;                    // Punkte auf der Rohroberfläche
    this.matPkt = () => new THREE.PointsMaterial({ color: new THREE.Color(0.25, 0.9, 2.6), size: 1.6, sizeAttenuation: false, transparent: true, vertexColors: true,
      opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
    for (const g of j.gelaender) {
      if (!g.pts || g.pts.length < 2) continue;
      const pts = g.pts.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      if (pts[0].y - pts[pts.length - 1].y < 1.0) continue;       // keine echte Treppe
      const cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
      const L = cum[cum.length - 1];
      const rail = { ...g, pts, cum, len: L, a: pts[0], b: pts[pts.length - 1], scanT: -1e9, hover: 0 };
      // Punkte gleichmäßig über die Länge, je Punkt eigene Helligkeit (einzelne leuchtende Punkte)
      const n = Math.round(L * 70), P = new Float32Array(n * 3), v = new THREE.Vector3(), t3 = new THREE.Vector3();
      for (let i = 0; i < n; i++) {
        this.punktBei(rail, Math.random() * L, v, t3);
        const s1 = new THREE.Vector3(0, 1, 0).cross(t3).normalize(), s2 = t3.clone().cross(s1).normalize(), ang = Math.random() * 6.283;
        v.addScaledVector(s1, Math.cos(ang) * R).addScaledVector(s2, Math.sin(ang) * R); P.set([v.x, v.y, v.z], i * 3);
      }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(P, 3));
      const Cc = new Float32Array(n * 3); for (let i = 0; i < n; i++) { const k = 0.35 + Math.random() ** 2 * 1.1; Cc.set([k, k, k], i * 3); }
      geo.setAttribute('color', new THREE.BufferAttribute(Cc, 3));
      const mesh = new THREE.Points(geo, this.matPkt()); mesh.layers.set(0); mesh.visible = false; mesh.renderOrder = 5;
      this.group.add(mesh); rail.mesh = mesh;
      this.rails.push(rail);
    }
    // Funken/Glühpunkt am Kontakt
    const sg = new THREE.SphereGeometry(0.09, 10, 8);
    this.funke = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: new THREE.Color(3, 2.4, 1.4), toneMapped: false }));
    this.funke.visible = false; this.scene.add(this.funke);
    return this.rails.length;
  }
  // Position (und Tangente, Richtung bergab) auf dem Handlauf bei Bogenlänge u
  punktBei(r, u, out, tan) {
    u = THREE.MathUtils.clamp(u, 0, r.len);
    let i = 1; while (i < r.cum.length - 1 && r.cum[i] < u) i++;
    const a = r.pts[i - 1], b = r.pts[i], l = Math.max(1e-6, r.cum[i] - r.cum[i - 1]);
    out.copy(a).lerp(b, (u - r.cum[i - 1]) / l);
    if (tan) tan.copy(b).sub(a).normalize();
    return out;
  }
  // nächster Punkt auf einem Handlauf zu p (Welt), Rückgabe {rail, u (0..len), dist, q}
  naechster(p) {
    let best = null; const q = new THREE.Vector3(), ab = new THREE.Vector3();
    for (const r of this.rails) for (let i = 1; i < r.pts.length; i++) {
      const a = r.pts[i - 1], b = r.pts[i]; ab.copy(b).sub(a); const l2 = ab.lengthSq();
      const t = l2 > 0 ? THREE.MathUtils.clamp(p.clone().sub(a).dot(ab) / l2, 0, 1) : 0;
      q.copy(a).addScaledVector(ab, t); const d = q.distanceTo(p);
      if (!best || d < best.dist) best = { rail: r, u: r.cum[i - 1] + t * Math.sqrt(l2), dist: d, q: q.clone() };
    }
    return best;
  }
  // Ziel unter dem Fadenkreuz: Strahl aus der Kamera, Handlauf in Reichweite der Figur
  zielen(player) {
    if (!this.rails.length) return null;
    const ray = new THREE.Ray(this.camera.position.clone(), new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion));
    const hip = player.pos.clone().setY(player.pos.y + 0.9);
    let best = null; const ps = new THREE.Vector3(), pr = new THREE.Vector3();
    for (const r of this.rails) for (let i = 1; i < r.pts.length; i++) {
      const a = r.pts[i - 1], b = r.pts[i];
      const d2 = ray.distanceSqToSegment(a, b, ps, pr);
      if (d2 > 0.6 * 0.6 || ps.distanceTo(this.camera.position) > ZIEL_R + 4) continue;
      if (pr.distanceTo(hip) > AUFSITZ_R) continue;
      if (!best || d2 < best.d2) best = { rail: r, d2, u: r.cum[i - 1] + pr.distanceTo(a), q: pr.clone() };
    }
    return best;
  }
  hervorheben(z) { this._hl = z ? z.rail : null; }
  // F-Scan: Geländer in Reichweite glühen blau, solange der Scan im Gedächtnis ist
  gescannt(origin, T, reichweite = 150) {
    for (const r of this.rails) {
      const q = this.punktBei(r, r.len / 2, new THREE.Vector3());
      if (q.distanceTo(origin) > reichweite) continue;
      const d = q.clone().sub(origin), dist = d.length(); d.normalize();
      const frei = this.world.raycast(origin, d, dist) >= dist - 1.2;       // nur was der Scanner sieht
      if (frei) r.scanT = T + dist / 42;                                     // erreicht, wenn die Scanwelle ankommt
    }
  }
  starten(z) {
    if (!z || this.aktiv) return false;
    const r = z.rail, u = THREE.MathUtils.clamp(z.u, 0.1, r.len - 1.0);
    this.aktiv = { rail: r, u, v: 0.6, t: 0 };
    this.hervorheben(null);
    return true;
  }
  // Rückgabe: { pos (Füße der Figur), yaw, sitz 0..1, ende } oder null
  update(dt, T) {
    const puls = 0.8 + 0.2 * Math.sin(T * 6);
    for (const r of this.rails) {
      const ziel = (this._hl === r || (this.aktiv && this.aktiv.rail === r)) ? 1 : 0;
      r.hover += (ziel - r.hover) * Math.min(1, dt * 12);
      const age = T - r.scanT, scan = age < 0 ? 0 : age < 0.4 ? age / 0.4 : age < 6 ? 1 : Math.max(0, 1 - (age - 6) / 4);
      const m = r.mesh.material, o = Math.max(r.hover * puls, scan * 0.75);
      m.opacity = o; r.mesh.visible = o > 0.01;
      m.color.setRGB(0.25 + 1.8 * r.hover, 0.9 + 1.5 * r.hover, 2.6 + 0.6 * r.hover); m.size = 1.6 + 0.8 * r.hover;
    }
    const A = this.aktiv; if (!A) { this.funke.visible = false; return null; }
    const r = A.rail, q = new THREE.Vector3(), tan = new THREE.Vector3();
    this.punktBei(r, A.u, q, tan);
    const sinA = Math.max(0, -tan.y);                                // Neigung des aktuellen Abschnitts (Podest = flach)
    A.t += dt;
    const auf = Math.min(1, A.t / 0.28);                             // aufsitzen
    if (auf >= 1) { A.v += (G * sinA - REIBUNG * G * Math.sqrt(1 - sinA * sinA)) * dt; A.v = Math.min(MAX_V, Math.max(0.6, A.v)); A.u += A.v * dt; }
    this.punktBei(r, A.u, q, tan);
    const yaw = Math.atan2(tan.x, tan.z);
    // Figur sitzt auf dem Rohr: Gesäß auf dem Rohr, Füße hängen ~0,6 m darunter (Wurzel = Füße)
    const pos = q.clone(); pos.y = q.y - 0.6;
    const wack = Math.sin(A.t * 13) * 0.02 * Math.min(1, A.v / 3);
    this.funke.visible = auf >= 1 && A.v > 1.2; this.funke.position.copy(q).setY(q.y + 0.03);
    this.funke.scale.setScalar(0.6 + 0.6 * Math.random() * Math.min(1, A.v / 4));
    if (A.u >= r.len) {                                              // unten: abspringen
      this.aktiv = null; this.funke.visible = false;
      const flach = tan.clone().setY(0).normalize();
      const land = r.b.clone().addScaledVector(flach, 0.9); land.y = r.b.y - 1.0;
      return { pos: land, yaw, sitz: 0, ende: true, v: A.v };
    }
    return { pos, yaw, sitz: auf, ende: false, wack, v: A.v };
  }
}

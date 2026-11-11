// Über-die-Schulter-Kamera: träge Verfolgung, Maus-Orbit, Zoom, Kollision mit der Stadt (mehrere Strahlen).
import * as THREE from 'three';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));

export class CameraRig {
  constructor(camera, world) {
    this.camera = camera; this.world = world;
    this.yaw = 0; this.pitch = 0.2;          // pitch > 0: Kamera oberhalb, blickt leicht nach unten
    this.dist = 3.1; this.cur = 3.1; this.side = 1; this.sideS = 1;
    this.sens = 0.0021;
    this.follow = new THREE.Vector3(); this.pivot = new THREE.Vector3();
    this.shake = 0; this.t = 0; this.fov = 58;
    this._dir = new THREE.Vector3(); this._o = new THREE.Vector3(); this._r = new THREE.Vector3(); this._u = new THREE.Vector3();
    this._head = new THREE.Vector3(); this._g = {};
  }
  snap(p, playerYaw) {
    this.follow.set(p.x, p.y + 1.62, p.z);
    this.yaw = playerYaw + Math.PI; this.cur = this.dist;
  }
  update(dt, look, player) {
    const cam = this.camera, w = this.world;
    this.t += dt;
    this.yaw -= look.dx * this.sens;
    this.pitch = clamp(this.pitch + look.dy * this.sens, -0.55, 1.15);
    if (look.wheel) this.dist = clamp(this.dist * (1 + look.wheel * 0.0012), 1.3, 7.5);
    this.sideS = damp(this.sideS, this.side, 6, dt);

    // Ziel: Kopfhöhe über dem (geglätteten) Spieler, etwas Vorausschau in Laufrichtung
    const p = player.pos, vy = player.visualY;
    const ahead = 0.12;
    const tx = p.x + player.vel.x * ahead, tz = p.z + player.vel.z * ahead, ty = vy + 1.62;
    this.follow.x = damp(this.follow.x, tx, 7.5, dt);
    this.follow.z = damp(this.follow.z, tz, 7.5, dt);
    this.follow.y = damp(this.follow.y, ty, 5, dt);

    // Schulterversatz (rechts, leicht über der Schulter), gegen Wände gekürzt
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    const right = this._r.set(cy, 0, -sy);
    const head = this._head.copy(this.follow);
    const off = 0.52 * this.sideS;
    const dirS = this._o.copy(right).multiplyScalar(Math.sign(off) || 1);
    const sideFree = w.raycast(head, dirS, Math.abs(off) + 0.25);
    const offLen = Math.min(Math.abs(off), Math.max(0, sideFree - 0.25));
    this.pivot.copy(head).addScaledVector(dirS, offLen); this.pivot.y += 0.12;

    // Kamerarichtung und Kollision (5 Strahlen ~ Kugel mit 0.2 m Radius)
    const cp = Math.cos(this.pitch), dir = this._dir.set(sy * cp, Math.sin(this.pitch), cy * cp);
    const up = this._u.crossVectors(dir, right).normalize();
    let allowed = this.dist;
    const offs = [[0, 0], [0.2, 0], [-0.2, 0], [0, 0.16], [0, -0.16]];
    for (const [a, b] of offs) {
      this._o.copy(this.pivot).addScaledVector(right, a).addScaledVector(up, b);
      const d = w.raycast(this._o, dir, this.dist + 0.3);
      if (d < Infinity) allowed = Math.min(allowed, d - 0.28);
    }
    allowed = Math.max(0.35, allowed);
    this.cur = allowed < this.cur ? allowed : damp(this.cur, allowed, 2.5, dt);
    cam.position.copy(this.pivot).addScaledVector(dir, this.cur);
    // nicht unter das Gelände
    const g = w.groundAt(cam.position.x, cam.position.z, cam.position.y + 0.05, 6, this._g);   // nur Flächen unter der Kamera
    if (g && cam.position.y < g.y + 0.3) cam.position.y = g.y + 0.3;

    // leichte Kamera-Erschütterung (Stolpern/Landung) und Gehbewegung
    this.shake *= Math.exp(-4 * dt);
    const s = this.shake * 0.05, bob = Math.min(1, player.actualSpeed / 4.5) * 0.012;
    cam.position.x += Math.sin(this.t * 37) * s; cam.position.y += Math.sin(this.t * 43) * s + Math.sin(this.t * 9.5) * bob;
    this._o.copy(cam.position).sub(dir);
    cam.lookAt(this._o);

    // Sichtfeld weitet sich beim Laufen leicht
    const fov = this.fov + clamp((player.actualSpeed - 1.5) / 3, 0, 1) * 6;
    if (Math.abs(cam.fov - fov) > 0.01) { cam.fov = damp(cam.fov, fov, 3, dt); cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld();
  }
}

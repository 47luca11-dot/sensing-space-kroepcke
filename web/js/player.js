// Fortbewegung im Stil von Death Stranding: träge Beschleunigung, weiches Drehen, Bodenhaftung per Raycast,
// Stufen bis ~27 cm, Rutschen an Steilhängen, Schwerkraft, Ausdauer und ein Gleichgewichtssystem.
import * as THREE from 'three';

export const P = {
  walk: 1.5, run: 4.5, careful: 0.8,
  accelWalk: 2.0, accelRun: 1.7, decel: 3.0, decelHard: 5.5,     // m/s²
  turnStand: 4.2, turnWalk: 3.0, turnRun: 1.7,                     // rad/s
  gravity: 13.5, stepUp: 0.27, snapDown: 0.42, radius: 0.3, height: 1.85,
  slideStart: Math.cos(THREE.MathUtils.degToRad(35)), slideAccel: 8.5,
  hopV: 3.1,
  staminaRun: 0.10, staminaRegenWalk: 0.10, staminaRegenStand: 0.2, runMin: 0.22,
  stumbleDur: 1.42, recoverDur: 0.6,
};
// Admin-Modus: keine Ausdauer, kein Stolpern, sehr schnell (Taste M schaltet um)
export const PA = {
  walk: 3.2, run: 14.0, careful: 1.6,
  accelWalk: 7.0, accelRun: 11.0, decel: 9.0, decelHard: 14.0,
  turnStand: 7.0, turnWalk: 6.0, turnRun: 4.2,
};

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));

function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

export class Player {
  constructor(world) {
    this.world = world;
    this.pos = new THREE.Vector3();
    this.prev = new THREE.Vector3();
    this.yaw = 0;                 // Blickrichtung des Körpers: vorwärts = (sin yaw, 0, cos yaw)
    this.speed = 0;               // Vorwärtsgeschwindigkeit (gewollt)
    this.actualSpeed = 0;         // gemessene horizontale Geschwindigkeit (geglättet)
    this.vel = new THREE.Vector3();          // tatsächliche Geschwindigkeit (für Kamera)
    this.slide = new THREE.Vector3();        // Rutsch-/Flugimpuls
    this.vy = 0; this.grounded = true; this.airTime = 0;
    this.normal = new THREE.Vector3(0, 1, 0);
    this.visualY = 0;
    this.yawRate = 0; this.longAccel = 0;
    this.grade = 0; this.cross = 0; this.rough = 0;
    this.stamina = 1; this.exhausted = false;
    this.bal = { x: 0, y: 0, vx: 0, vy: 0, nx: 0, ny: 0 };     // x: + = kippt nach rechts, y: + = nach vorn
    this.stumbleT = -1; this.recoverT = 0; this.stumbles = 0;
    this.mode = 'walk'; this.running = false; this.careful = false; this.stabilize = false;
    this.idleTime = 0;
    this.admin = false;           // Admin-Modus (siehe PA)
    this.events = [];             // 'stumble', 'land', 'step' … (vom Hauptprogramm abgeholt)
    this.rand = rng(77);
    this._g = {}; this._s = [{}, {}, {}, {}, {}];
  }

  spawn(x, y, z, yaw) {
    const g = this.world.groundAt(x, z, y + 3, 400, this._g) || this.world.groundAt(x, z, y + 300, 1000, this._g);
    this.pos.set(x, g ? g.y : y, z); this.prev.copy(this.pos);
    this.visualY = this.pos.y; this.yaw = yaw; this.speed = 0; this.vy = 0; this.slide.set(0, 0, 0);
    this.grounded = true; this.bal.x = this.bal.y = this.bal.vx = this.bal.vy = 0; this.stumbleT = -1; this.recoverT = 0;
  }

  get stumbling() { return this.stumbleT >= 0; }

  triggerStumble(reason = 'balance') {
    if (this.stumbleT >= 0 || this.admin) return;
    this.stumbleT = 0; this.stumbles++;
    this.speed *= 0.35;
    this.stamina = Math.max(0, this.stamina - 0.12);
    this.events.push({ type: 'stumble', reason });
  }

  // Bodenhöhe an einer Stelle relativ zur aktuellen Höhe (für Steigung/Unebenheit)
  sample(x, z, fallback) {
    const g = this.world.groundAt(x, z, this.pos.y + 1.2, 3, this._g);
    return g ? g.y : fallback;
  }

  update(dt, inp, camYaw) {
    const w = this.world;
    this.prev.copy(this.pos);
    const stumbling = this.stumbleT >= 0;
    if (stumbling) { this.stumbleT += dt; if (this.stumbleT >= P.stumbleDur) { this.stumbleT = -1; this.recoverT = P.recoverDur; } }
    if (this.recoverT > 0) this.recoverT -= dt;

    // ---------- gewünschte Richtung relativ zur Kamera ----------
    const ix = (inp.right ? 1 : 0) - (inp.left ? 1 : 0), iz = (inp.forward ? 1 : 0) - (inp.back ? 1 : 0);
    let mag = Math.hypot(ix, iz);
    let targetYaw = this.yaw;
    if (mag > 0) {
      const fx = -Math.sin(camYaw), fz = -Math.cos(camYaw), rx = Math.cos(camYaw), rz = -Math.sin(camYaw);
      const dx = fx * iz + rx * ix, dz = fz * iz + rz * ix;
      targetYaw = Math.atan2(dx, dz); mag = Math.min(1, mag);
    }
    const control = !stumbling && this.grounded;
    if (!control) mag = 0;

    // ---------- Modus ----------
    this.stabilize = inp.balL && inp.balR;
    this.careful = inp.careful && !stumbling;
    const A = this.admin ? PA : P;
    if (this.admin) { this.stamina = 1; this.exhausted = false; }
    if (this.stamina <= 0.001) this.exhausted = true;
    if (this.exhausted && this.stamina > P.runMin) this.exhausted = false;
    this.running = inp.run && !this.careful && !this.exhausted && mag > 0 && control;
    let maxSpeed = this.careful ? A.careful : this.running ? A.run : A.walk;
    if (this.stabilize) maxSpeed = Math.min(maxSpeed, P.walk * 0.7);
    if (this.recoverT > 0) maxSpeed = Math.min(maxSpeed, 0.6);
    this.mode = stumbling ? 'stumble' : !this.grounded ? 'air' : this.careful ? 'careful' : this.running ? 'run' : 'walk';

    // ---------- Drehen (begrenzt, langsamer bei Tempo) ----------
    const prevYaw = this.yaw;
    const delta = wrap(targetYaw - this.yaw);
    if (mag > 0) {
      const s = clamp(this.speed / A.run, 0, 1);
      let rate = this.speed < 0.4 ? A.turnStand : THREE.MathUtils.lerp(A.turnWalk, A.turnRun, s);
      if (this.careful) rate *= 0.75;
      const step = Math.sign(delta) * Math.min(Math.abs(delta), rate * dt, Math.abs(delta) * 10 * dt + 0.002);
      this.yaw = wrap(this.yaw + step);
    }
    this.yawRate = damp(this.yawRate, wrap(this.yaw - prevYaw) / Math.max(dt, 1e-4), 12, dt);

    // ---------- Steigung voraus / Querneigung / Unebenheit (5 Bodenproben) ----------
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    const px = this.pos.x, pz = this.pos.z, h0 = this.grounded ? this.pos.y : this.sample(px, pz, this.pos.y);
    const hF = this.sample(px + sy * 0.35, pz + cy * 0.35, h0);
    const hB = this.sample(px - sy * 0.35, pz - cy * 0.35, h0);
    const hL = this.sample(px + cy * 0.17, pz - sy * 0.17, h0);        // linker Fuß (+X lokal)
    const hR = this.sample(px - cy * 0.17, pz + sy * 0.17, h0);
    const hC = this.sample(px, pz, h0);
    const grade = clamp((hF - hB) / 0.7, -1.2, 1.2);
    this.grade = damp(this.grade, grade, 8, dt);
    this.cross = damp(this.cross, clamp((hL - hR) / 0.34, -1, 1), 8, dt);
    const rough = Math.abs(hF + hB - 2 * hC) * 0.8 + Math.abs(hL + hR - 2 * hC) * 1.4;
    this.rough = damp(this.rough, Math.min(rough, 0.5), 6, dt);

    // ---------- Tempo (träge) ----------
    const align = mag > 0 ? Math.max(0, Math.cos(delta)) : 0;
    const uphill = Math.max(0, this.grade), downhill = Math.max(0, -this.grade);
    const slopeFactor = 1 / (1 + uphill * 1.7) * (1 + Math.min(0.12, downhill * 0.35));
    const target = maxSpeed * mag * Math.pow(align, 1.5) * slopeFactor;
    const prevSpeed = this.speed;
    if (target > this.speed) this.speed = Math.min(target, this.speed + (this.running ? A.accelRun : A.accelWalk) * (1 - uphill * 0.5) * dt);
    else this.speed = Math.max(target, this.speed - (stumbling || Math.abs(delta) > 2.2 ? A.decelHard : A.decel) * dt);
    this.longAccel = damp(this.longAccel, (this.speed - prevSpeed) / Math.max(dt, 1e-4), 10, dt);

    // ---------- Bewegung + Kollision ----------
    const g = this._g;
    const vx = sy * this.speed + this.slide.x, vz = cy * this.speed + this.slide.z;
    const n = Math.max(1, Math.ceil(Math.hypot(vx, vz) * dt / 0.12));
    const sdt = dt / n;
    let stepJolt = 0;
    for (let i = 0; i < n; i++) {
      const ox = this.pos.x, oz = this.pos.z, oy = this.pos.y;
      this.pos.x += vx * sdt; this.pos.z += vz * sdt;
      w.collideCapsule(this.pos, P.radius, P.stepUp + P.radius + 0.02, P.height - P.radius);
      const b = w.bounds; if (b) { this.pos.x = clamp(this.pos.x, b[0], b[2]); this.pos.z = clamp(this.pos.z, b[1], b[3]); }
      const hit = w.groundAt(this.pos.x, this.pos.z, oy + P.stepUp + 0.35, 400, g);
      if (this.grounded) {
        if (hit && hit.y <= oy + P.stepUp + 0.01 && hit.y >= oy - P.snapDown) {
          stepJolt += Math.abs(hit.y - oy) > 0.06 ? hit.y - oy : 0;
          this.pos.y = hit.y; this.normal.set(hit.nx, hit.ny, hit.nz);
        } else if (hit && hit.y > oy + P.stepUp) {
          this.pos.x = ox; this.pos.z = oz;                  // zu hoch (Kante/Wand ohne Kapselkontakt)
          this.speed *= 0.6;
        } else {
          this.grounded = false; this.vy = 0; this.airTime = 0;
          this.slide.x = vx; this.slide.z = vz; this.speed = 0;   // Schwung mitnehmen
        }
      }
    }

    // ---------- Luft ----------
    if (!this.grounded) {
      this.airTime += dt;
      this.vy -= P.gravity * dt;
      this.pos.y += this.vy * dt;
      this.slide.multiplyScalar(Math.exp(-0.3 * dt));
      const hit = w.groundAt(this.pos.x, this.pos.z, this.pos.y + P.stepUp + 0.4, 400, g);
      if (hit && this.pos.y <= hit.y && this.vy <= 0) {
        const impact = -this.vy;
        this.pos.y = hit.y; this.grounded = true; this.vy = 0; this.normal.set(hit.nx, hit.ny, hit.nz);
        this.speed = Math.hypot(this.slide.x, this.slide.z) * 0.5;
        this.slide.set(0, 0, 0);
        this.events.push({ type: 'land', impact });
        if (impact > 7.2) this.triggerStumble('fall');
        else if (impact > 3.2) { this.bal.vy += impact * 0.12; this.bal.vx += (this.rand() - 0.5) * impact * 0.16; }
      }
      if (this.pos.y < (w.minY ?? -100)) { this.events.push({ type: 'respawn' }); return; }
    } else {
      // Rutschen an Steilhängen
      const ny = this.normal.y;
      if (ny < P.slideStart) {
        const hx = this.normal.x, hz = this.normal.z, hl = Math.hypot(hx, hz) || 1;
        const k = (P.slideStart - ny) / P.slideStart;
        this.slide.x += (hx / hl) * P.slideAccel * (0.4 + k * 2) * dt; this.slide.z += (hz / hl) * P.slideAccel * (0.4 + k * 2) * dt;
        const along = (sy * hx + cy * hz) / hl;              // < 0: läuft bergauf
        if (along < 0) this.speed *= Math.exp(-4 * dt);
        this.sliding = true;
      } else {
        this.slide.multiplyScalar(Math.exp(-6 * dt));
        this.sliding = this.slide.lengthSq() > 0.25;
      }
      if (inp.jump && !stumbling && this.stamina > 0.08 && this.recoverT <= 0) {
        this.grounded = false; this.vy = P.hopV; this.airTime = 0;
        this.slide.x += sy * this.speed; this.slide.z += cy * this.speed; this.speed = 0;
        this.stamina -= 0.06;
        this.events.push({ type: 'hop' });
      }
    }

    // gemessene Geschwindigkeit (gegen Wände laufen -> Tempo sinkt)
    const mdx = this.pos.x - this.prev.x, mdz = this.pos.z - this.prev.z;
    const moved = Math.hypot(mdx, mdz) / Math.max(dt, 1e-4);
    this.vel.set(mdx / Math.max(dt, 1e-4), (this.pos.y - this.prev.y) / Math.max(dt, 1e-4), mdz / Math.max(dt, 1e-4));
    if (this.grounded && moved < this.speed - 0.3) this.speed = Math.max(moved + 0.3, this.speed - 6 * dt);
    this.actualSpeed = damp(this.actualSpeed, this.grounded ? Math.min(moved, (this.admin ? PA : P).run * 1.3) : this.actualSpeed, 14, dt);
    // sichtbare Höhe: Stufen hoch weich, runter etwas schneller
    this.visualY = this.grounded ? damp(this.visualY, this.pos.y, this.pos.y > this.visualY ? 16 : 22, dt) : this.pos.y;
    if (Math.abs(this.visualY - this.pos.y) > 0.6) this.visualY = this.pos.y;

    // ---------- Ausdauer (im Admin-Modus abgeschaltet) ----------
    if (this.admin) this.stamina = 1;
    else if (this.running && this.actualSpeed > 2.2) this.stamina -= P.staminaRun * (1 + uphill * 2) * dt;
    else if (!stumbling) this.stamina += (this.actualSpeed < 0.2 ? P.staminaRegenStand : this.careful ? 0.14 : P.staminaRegenWalk) * dt * (this.stabilize ? 0.5 : 1);
    this.stamina = clamp(this.stamina, 0, 1);
    this.idleTime = this.actualSpeed < 0.08 && this.grounded && !stumbling ? this.idleTime + dt : 0;

    this.updateBalance(dt, inp, stepJolt, stumbling);
  }

  // ---------- Gleichgewicht: zwei gedämpfte Federn (seitlich, vorn/hinten) mit Störungen ----------
  updateBalance(dt, inp, stepJolt, stumbling) {
    const b = this.bal, v = this.actualSpeed;
    if (this.admin) {                               // Admin-Modus: immer im Gleichgewicht
      b.x = damp(b.x, 0, 6, dt); b.y = damp(b.y, 0, 6, dt); b.vx = b.vy = 0; return;
    }
    if (stumbling || this.recoverT > 0) {          // während des Stolperns zurück zur Mitte
      b.x = damp(b.x, 0, 3, dt); b.y = damp(b.y, 0, 3, dt); b.vx = b.vy = 0; return;
    }
    if (!this.grounded) { b.vx *= Math.exp(-1 * dt); b.vy *= Math.exp(-1 * dt); b.x += b.vx * dt; b.y += b.vy * dt; return; }
    const load = 0.35 + v / P.walk * 0.65;          // schneller = empfindlicher
    // Rauschen (geglättet): Unebenheit + Tempo
    const a = Math.exp(-6 * dt);
    b.nx = b.nx * a + (this.rand() * 2 - 1) * (1 - a) * 3.2;
    b.ny = b.ny * a + (this.rand() * 2 - 1) * (1 - a) * 3.2;
    const roughAmp = this.rough * 34 * load * (this.careful ? 0.45 : 1) + (v > 3.2 ? 0.5 : 0);
    let fx = 0.4 * v * this.yawRate                // Fliehkraft in Kurven (kippt nach außen)
      + this.cross * 1.6 * load                     // Querneigung: Richtung Tal
      + b.nx * roughAmp;
    let fy = -this.longAccel * 0.16                 // bremsen -> nach vorn, anfahren -> nach hinten
      + Math.max(0, -this.grade) * v * 1.25         // bergab
      + b.ny * roughAmp * 0.7;
    if (this.sliding) { fx += b.nx * 6; fy += 2.2; }
    // Stufen geben einen kleinen Stoß
    if (stepJolt) { b.vy += -stepJolt * v * 1.6; b.vx += (this.rand() - 0.5) * Math.abs(stepJolt) * v * 2.4; }

    let K = this.careful ? 10 : clamp(10 - 1.55 * v, 3.2, 10);
    let C = 2 * 0.78 * Math.sqrt(K);
    // Korrektur: Q hält nach links (wirkt gegen Kippen nach links), E nach rechts; beide = stabilisieren
    const corr = 7.5;
    if (this.stabilize) { K *= 2.6; C *= 2.0; fx *= 0.45; fy *= 0.45; }
    else {
      if (inp.balL) fx += b.x < 0 ? corr * Math.min(1, -b.x * 3 + 0.35) : corr * 0.12;
      if (inp.balR) fx -= b.x > 0 ? corr * Math.min(1, b.x * 3 + 0.35) : corr * 0.12;
      if (inp.balL || inp.balR) fy -= b.y * 2.5;
    }
    // jenseits ~0.6 kippt es von selbst weiter (instabil)
    const U = 9;
    const ux = Math.abs(b.x) > 0.6 ? U * (Math.abs(b.x) - 0.6) * Math.sign(b.x) : 0;
    const uy = Math.abs(b.y) > 0.6 ? U * (Math.abs(b.y) - 0.6) * Math.sign(b.y) : 0;
    b.vx += (fx - K * b.x - C * b.vx + ux) * dt;
    b.vy += (fy - K * b.y - C * b.vy + uy) * dt;
    b.x += b.vx * dt; b.y += b.vy * dt;
    if (Math.abs(b.x) > 1 || Math.abs(b.y) > 1) this.triggerStumble('balance');
  }

  // für Animation/HUD
  get balanceMag() { return Math.max(Math.abs(this.bal.x), Math.abs(this.bal.y)); }
}

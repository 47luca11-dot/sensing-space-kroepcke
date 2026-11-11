import * as THREE from 'three';
import type { Input } from '../controls/Input';
import type { PlayerController } from '../controls/PlayerController';
import type { ObjectRegistry } from '../objects/ObjectFactory';
import { RAPIER, type PhysicsWorld } from '../physics/PhysicsWorld';
import type { Hud } from '../ui/Hud';

const KEY = 'KeyM';
const NOTE_VALUE = 50;
const START_MONEY = 1000;
const THROW_INTERVAL = 0.11;
const MAX_BILLS = 400;
const GRAVITY = -9.81;
/** Luftwiderstand: Scheine fallen langsam (Endgeschwindigkeit ca. 1,5 m/s). */
const DRAG = 3.2;
/** Erst nach dieser Liegezeit wird ein Schein beim Darüberlaufen eingesammelt. */
const PICKUP_DELAY = 1.0;
const PICKUP_RADIUS = 0.55;

interface Bill {
  p: THREE.Vector3;
  v: THREE.Vector3;
  q: THREE.Quaternion;
  spin: THREE.Vector3;
  phase: number;
  landed: boolean;
  landedAt: number;
}

/**
 * „Geld werfen“: M gedrückt halten wirft Scheine aus der rechten Hand.
 * Eigene kleine Physik statt Rapier, weil Papier flattert: starker Luftwiderstand,
 * seitliches Pendeln, Taumeln. Gelandet wird per Strahltest auf jeder festen Fläche;
 * an beweglichen Dingen prallen Scheine ab. Liegende Scheine sammelt man beim Darüberlaufen wieder ein.
 */
export class MoneyThrower {
  private readonly bills: Bill[] = [];
  private readonly mesh: THREE.InstancedMesh;
  private readonly ray = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: -1, z: 0 });
  private readonly dummy = new THREE.Object3D();
  private wallet = START_MONEY;
  private cooldown = 0;
  private time = 0;
  /** Wird bei jedem Wurf aufgerufen (Handbewegung). */
  onThrow: () => void = () => {};

  constructor(
    scene: THREE.Scene,
    private readonly physics: PhysicsWorld,
    private readonly player: PlayerController,
    private readonly camera: THREE.Camera,
    private readonly input: Input,
    private readonly registry: ObjectRegistry,
    private readonly hud: Hud,
  ) {
    const material = new THREE.MeshStandardMaterial({ map: billTexture(), side: THREE.DoubleSide, roughness: 0.85 });
    this.mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.15, 0.072), material, MAX_BILLS);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
    this.mesh.receiveShadow = true;
    scene.add(this.mesh);
  }

  update(dt: number) {
    this.time += dt;
    this.cooldown -= dt;
    if (this.input.active && this.input.isDown(KEY) && this.cooldown <= 0) {
      this.cooldown = THROW_INTERVAL;
      if (this.wallet >= NOTE_VALUE) this.throwBill();
      else if (this.input.wasPressed(KEY)) this.hud.toast('Kein Geld mehr. Heb welches auf.');
    }
    this.simulate(dt);
    this.collect();
    this.render();
  }

  private throwBill() {
    this.wallet -= NOTE_VALUE;
    this.hud.setWallet(`${this.wallet} €`);
    if (this.bills.length >= MAX_BILLS) this.bills.shift();
    this.camera.updateMatrixWorld();
    // Aus der rechten Hand, schräg nach vorn oben, leicht gestreut
    const origin = new THREE.Vector3(0.16, -0.12, -0.32).applyMatrix4(this.camera.matrixWorld);
    const forward = this.camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(forward, THREE.Object3D.DEFAULT_UP).normalize();
    const v = forward
      .multiplyScalar(2.2 + Math.random() * 1.2)
      .add(new THREE.Vector3(0, 1.4 + Math.random() * 0.8, 0))
      .addScaledVector(right, (Math.random() - 0.5) * 1.4);
    this.bills.push({
      p: origin,
      v,
      q: new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.random() * 6, Math.random() * 6, Math.random() * 6)),
      spin: new THREE.Vector3((Math.random() - 0.5) * 14, (Math.random() - 0.5) * 10, (Math.random() - 0.5) * 14),
      phase: Math.random() * Math.PI * 2,
      landed: false,
      landedAt: 0,
    });
    this.onThrow();
  }

  private simulate(dt: number) {
    const world = this.physics.world;
    const playerHandle = this.player.collider.handle;
    const step = new THREE.Vector3();
    const rot = new THREE.Quaternion();
    for (const b of this.bills) {
      if (b.landed) continue;
      // Schwerkraft, Luftwiderstand, Flattern quer zur Fallrichtung
      b.v.y += GRAVITY * dt;
      b.v.multiplyScalar(Math.max(0, 1 - DRAG * dt));
      const flutter = Math.sin(this.time * 7 + b.phase) * 1.8;
      b.v.x += Math.cos(b.phase) * flutter * dt;
      b.v.z += Math.sin(b.phase) * flutter * dt;
      // Taumeln, langsam abklingend
      const angle = b.spin.length() * dt;
      if (angle > 1e-5) b.q.multiply(rot.setFromAxisAngle(b.spin.clone().normalize(), angle));
      b.spin.multiplyScalar(Math.max(0, 1 - 0.6 * dt));

      step.copy(b.v).multiplyScalar(dt);
      const len = step.length();
      if (len < 1e-6) continue;
      this.ray.origin = { x: b.p.x, y: b.p.y, z: b.p.z };
      this.ray.dir = { x: step.x / len, y: step.y / len, z: step.z / len };
      const hit = world.castRayAndGetNormal(this.ray, len, true, undefined, undefined, undefined, undefined, (c) => c.handle !== playerHandle);
      if (!hit) {
        b.p.add(step);
        continue;
      }
      const normal = new THREE.Vector3(hit.normal.x, hit.normal.y, hit.normal.z);
      const obj = this.registry.fromCollider(hit.collider);
      const point = b.p.clone().addScaledVector(step, hit.timeOfImpact / len);
      if ((obj && !obj.body.isFixed()) || normal.y < 0.35) {
        // Bewegliche Dinge und steile Flächen: abprallen und weiterflattern
        b.p.copy(point).addScaledVector(normal, 0.01);
        b.v.reflect(normal).multiplyScalar(0.25);
        continue;
      }
      // Liegen bleiben, flach auf der Fläche, zufällig gedreht
      b.landed = true;
      b.landedAt = this.time;
      b.p.copy(point).addScaledVector(normal, 0.002 + Math.random() * 0.002);
      b.q
        .setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal)
        .multiply(rot.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.random() * Math.PI * 2));
    }
  }

  /** Liegende Scheine unter den Füßen einsammeln. */
  private collect() {
    const t = this.player.body.translation();
    const feet = t.y - 0.9;
    let gained = 0;
    for (let i = this.bills.length - 1; i >= 0; i--) {
      const b = this.bills[i];
      if (!b.landed || this.time - b.landedAt < PICKUP_DELAY) continue;
      if (Math.abs(b.p.y - feet) > 0.35) continue;
      if (Math.hypot(b.p.x - t.x, b.p.z - t.z) > PICKUP_RADIUS) continue;
      this.bills.splice(i, 1);
      gained += NOTE_VALUE;
    }
    if (gained) {
      this.wallet += gained;
      this.hud.setWallet(`+${gained} € · ${this.wallet} €`);
    }
  }

  private render() {
    let n = 0;
    for (const b of this.bills) {
      this.dummy.position.copy(b.p);
      this.dummy.quaternion.copy(b.q);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(n++, this.dummy.matrix);
    }
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

/** Erkennbar Spielgeld: eigener Entwurf, keine echte Währung. */
function billTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 124;
  const g = canvas.getContext('2d')!;
  const grad = g.createLinearGradient(0, 0, 256, 124);
  grad.addColorStop(0, '#9fc9a2');
  grad.addColorStop(1, '#5f9c7a');
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 124);
  g.strokeStyle = '#2f5c45';
  g.lineWidth = 4;
  g.strokeRect(6, 6, 244, 112);
  g.lineWidth = 1;
  for (let i = 0; i < 18; i++) {
    g.beginPath();
    g.arc(70, 62, 8 + i * 3, 0, Math.PI * 2);
    g.stroke();
  }
  g.fillStyle = '#1f3d2e';
  g.font = 'bold 54px Georgia, serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('50', 180, 58);
  g.font = 'bold 14px system-ui, sans-serif';
  g.fillText('SPIELGELD', 180, 98);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

import * as THREE from 'three';
import type { Input } from '../controls/Input';
import type { PlayerController } from '../controls/PlayerController';
import type { GameObject } from '../objects/GameObject';
import { createObject, type ObjectRegistry } from '../objects/ObjectFactory';
import { RAPIER, type PhysicsWorld } from '../physics/PhysicsWorld';
import type { Hud } from '../ui/Hud';

type Fluid = 'pee' | 'puke' | 'spit' | 'sneeze';

const KEYS = { pee: 'KeyP', poop: 'KeyK', puke: 'KeyV', spit: 'KeyG', sneeze: 'KeyN' };

const FLUIDS: Record<Fluid, { colors: string[]; puddle: string; opacity: number; size: [number, number] }> = {
  pee: { colors: ['#f0d83c', '#f6e66a', '#dcc22c'], puddle: '#d9c23c', opacity: 0.55, size: [0.006, 0.01] },
  puke: { colors: ['#a9b23e', '#cdb45c', '#8e982f', '#d8c47e', '#7d6a35'], puddle: '#a6a046', opacity: 0.9, size: [0.012, 0.03] },
  spit: { colors: ['#f4f2ea', '#e6e3d6', '#ffffff'], puddle: '#ebe8dc', opacity: 0.6, size: [0.005, 0.011] },
  sneeze: { colors: ['#eef3f0', '#dfe8e4', '#ffffff'], puddle: '#e2e8e4', opacity: 0.35, size: [0.0015, 0.0035] },
};

const MAX_PARTICLES = 1500;
const MAX_PUDDLES = 80;
const MAX_POOPS = 20;
const GRAVITY = -9.81;

/** Blase in Sekunden Strahl, füllt sich langsam wieder. */
const BLADDER_SECONDS = 6;
const BLADDER_REFILL = 1 / 12;
const PUKE_DURATION = 1.8;
const PUKE_COOLDOWN = 5;
const POOP_DURATION = 1.4;
const POOP_COOLDOWN = 5;
/** So viel Mageninhalt braucht ein Stuhlgang. */
const POOP_STOMACH = 0.3;
const SNEEZE_WINDUP = 0.7;

interface Particle {
  kind: Fluid;
  p: THREE.Vector3;
  v: THREE.Vector3;
  size: number;
  color: THREE.Color;
  life: number;
}

interface Puddle {
  kind: Fluid;
  mesh: THREE.Mesh;
  radius: number;
  normal: THREE.Vector3;
  parent: THREE.Object3D;
}

/**
 * Körperfunktionen: [P] gedrückt halten = pinkeln, [V] = kotzen, [K] = kacken.
 * Flüssigkeiten sind Partikel mit Schwerkraft, die per Rapier-Strahl auf Oberflächen treffen,
 * dort Pfützen bilden (auch auf beweglichen Gegenständen) und leichte Dinge anschubsen.
 */
export class BodilyFunctions {
  private readonly particles: Particle[] = [];
  private readonly puddles: Puddle[] = [];
  private readonly poops: GameObject[] = [];
  private readonly instanced: THREE.InstancedMesh;
  private readonly puddleMaterials: Record<Fluid, THREE.MeshStandardMaterial>;
  private readonly puddleGeometry = new THREE.CircleGeometry(1, 24);
  /** Weicher, leicht unregelmäßiger Rand für Pfützen. */
  private readonly softEdge = softEdgeTexture();
  private readonly ray = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: -1, z: 0 });

  private bladder = 1;
  /** Mageninhalt 0–1: Essen füllt ihn, jeder Stuhlgang braucht einen Teil davon. */
  private stomach = 0.6;
  private sneezeTime = -1;
  private bladderWarned = false;
  private pukeTime = -1;
  private pukeCooldown = 0;
  private poopTime = -1;
  private poopCooldown = 0;
  private poopSpawned = false;
  private crouch = 0;
  private time = 0;
  private poopCount = 0;

  private readonly dummy = new THREE.Object3D();
  private readonly tmp = new THREE.Vector3();
  private readonly forward = new THREE.Vector3();

  constructor(
    private readonly scene: THREE.Scene,
    private readonly physics: PhysicsWorld,
    private readonly player: PlayerController,
    private readonly camera: THREE.PerspectiveCamera,
    private readonly input: Input,
    private readonly registry: ObjectRegistry,
    private readonly hud: Hud,
  ) {
    const material = new THREE.MeshStandardMaterial({ roughness: 0.15, transparent: true, opacity: 0.85 });
    this.instanced = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 8, 6), material, MAX_PARTICLES);
    this.instanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.instanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3);
    this.instanced.frustumCulled = false;
    this.instanced.count = 0;
    scene.add(this.instanced);

    this.puddleMaterials = {
      pee: this.makePuddleMaterial('pee'),
      puke: this.makePuddleMaterial('puke'),
      spit: this.makePuddleMaterial('spit'),
      sneeze: this.makePuddleMaterial('sneeze'),
    };
  }

  /** Pro Bild nach `player.updateCamera`: Eingaben, Partikel, Pfützen, Kamera-Effekte. */
  update(dt: number) {
    this.time += dt;
    const active = this.input.active;

    // Pinkeln
    const peeing = active && this.input.isDown(KEYS.pee) && this.bladder > 0 && this.poopTime < 0;
    if (peeing) {
      this.bladder = Math.max(0, this.bladder - dt / BLADDER_SECONDS);
      this.emitPee(dt);
      this.bladderWarned = false;
    } else {
      this.bladder = Math.min(1, this.bladder + dt * BLADDER_REFILL);
    }
    if (active && this.input.isDown(KEYS.pee) && this.bladder === 0 && !this.bladderWarned) {
      this.hud.toast('Blase leer.');
      this.bladderWarned = true;
    }

    // Kotzen
    this.pukeCooldown -= dt;
    if (active && this.input.wasPressed(KEYS.puke)) {
      if (this.pukeCooldown > 0 || this.pukeTime >= 0) this.hud.toast('Noch nicht wieder möglich.');
      else {
        this.pukeTime = 0;
        this.hud.toast('Übelkeit …');
      }
    }
    if (this.pukeTime >= 0) {
      this.pukeTime += dt;
      // Drei Schübe
      const surge = Math.max(0, Math.sin((this.pukeTime / PUKE_DURATION) * Math.PI * 3));
      if (surge > 0.2) this.emitPuke(dt, surge);
      if (this.pukeTime > PUKE_DURATION) {
        this.pukeTime = -1;
        this.pukeCooldown = PUKE_COOLDOWN;
      }
    }

    // Kacken
    this.poopCooldown -= dt;
    if (active && this.input.wasPressed(KEYS.poop)) {
      if (this.poopCooldown > 0 || this.poopTime >= 0) this.hud.toast('Noch nicht wieder möglich.');
      else if (this.stomach < POOP_STOMACH) this.hud.toast('Der Magen ist leer. Erst etwas essen.');
      else {
        this.stomach -= POOP_STOMACH;
        this.poopTime = 0;
        this.poopSpawned = false;
      }
    }

    // Spucken
    if (active && this.input.wasPressed(KEYS.spit)) this.emitSpit();

    // Niesen: kurzes Ansetzen, dann ein feiner Sprühstoß
    if (active && this.input.wasPressed(KEYS.sneeze) && this.sneezeTime < 0) {
      this.sneezeTime = 0;
      this.hud.toast('Ha… ha…');
    }
    if (this.sneezeTime >= 0) {
      const before = this.sneezeTime;
      this.sneezeTime += dt;
      if (before < SNEEZE_WINDUP && this.sneezeTime >= SNEEZE_WINDUP) {
        this.emitSneeze();
        this.hud.toast('Hatschi!');
      }
      if (this.sneezeTime > SNEEZE_WINDUP + 0.5) this.sneezeTime = -1;
    }
    if (this.poopTime >= 0) {
      this.poopTime += dt;
      if (!this.poopSpawned && this.poopTime > POOP_DURATION * 0.65) {
        this.poopSpawned = true;
        void this.spawnPoop();
      }
      if (this.poopTime > POOP_DURATION) {
        this.poopTime = -1;
        this.poopCooldown = POOP_COOLDOWN;
      }
    }

    this.simulate(dt);
    this.applyCameraEffects(dt);
  }

  get isPeeing() {
    return this.input.isDown(KEYS.pee) && this.bladder > 0;
  }

  /** Erbrechen von außen auslösen (z. B. zu viel Alkohol). */
  startPuke() {
    if (this.pukeTime >= 0) return;
    this.pukeTime = 0;
  }

  /** Ein Bissen: füllt den Magen. */
  eat(amount = 0.2) {
    this.stomach = Math.min(1, this.stomach + amount);
  }

  /** Ein Schluck: füllt die Blase. */
  drink(amount = 0.35) {
    this.bladder = Math.min(1, this.bladder + amount);
    this.bladderWarned = false;
  }

  private emitSpit() {
    const eye = this.player.eyePosition(this.tmp);
    const look = this.camera.getWorldDirection(this.forward);
    const origin = eye.clone().add(new THREE.Vector3(0, -0.1, 0)).addScaledVector(look, 0.12);
    for (let i = 0; i < 18; i++) {
      const v = look.clone().multiplyScalar(4.2 + Math.random() * 0.6).add(new THREE.Vector3(0, 0.6, 0)).add(jitter(0.15));
      this.spawn('spit', origin.clone().add(jitter(0.004)), v);
    }
  }

  private emitSneeze() {
    const eye = this.player.eyePosition(this.tmp);
    const look = this.camera.getWorldDirection(this.forward);
    const origin = eye.clone().add(new THREE.Vector3(0, -0.06, 0)).addScaledVector(look, 0.1);
    for (let i = 0; i < 260; i++) {
      const v = look.clone().multiplyScalar(3 + Math.random() * 3).add(jitter(1.3));
      this.spawn('sneeze', origin.clone().add(jitter(0.01)), v);
    }
  }

  // ── Emission ─────────────────────────────────────────────────────────

  private emitPee(dt: number) {
    const eye = this.player.eyePosition(this.tmp);
    const look = this.camera.getWorldDirection(this.forward);
    const flat = new THREE.Vector3(look.x, 0, look.z).normalize();
    const origin = eye.clone().add(new THREE.Vector3(0, -0.72, 0)).addScaledVector(flat, 0.18);
    // Zielen mit der Maus: Blick nach oben = weiter
    const pitchBoost = THREE.MathUtils.clamp(look.y + 0.4, 0, 1.2);
    const count = Math.ceil(220 * dt);
    for (let i = 0; i < count; i++) {
      const v = flat.clone().multiplyScalar(1.9 + pitchBoost * 1.2).add(new THREE.Vector3(0, 0.6 + pitchBoost * 1.6, 0));
      v.add(jitter(0.07));
      // Leichtes Schwanken des Strahls
      v.x += Math.sin(this.time * 7) * 0.05;
      this.spawn('pee', origin.clone().add(jitter(0.004)), v);
    }
  }

  private emitPuke(dt: number, surge: number) {
    const eye = this.player.eyePosition(this.tmp);
    const look = this.camera.getWorldDirection(this.forward);
    const origin = eye.clone().add(new THREE.Vector3(0, -0.12, 0)).addScaledVector(look, 0.15);
    const count = Math.ceil(320 * dt * surge);
    for (let i = 0; i < count; i++) {
      const v = look.clone().multiplyScalar(2.2 + Math.random() * 1.6 * surge).add(new THREE.Vector3(0, 0.3, 0));
      v.add(jitter(0.55));
      this.spawn('puke', origin.clone().add(jitter(0.02)), v);
    }
  }

  private spawn(kind: Fluid, p: THREE.Vector3, v: THREE.Vector3) {
    if (this.particles.length >= MAX_PARTICLES) this.particles.shift();
    const fluid = FLUIDS[kind];
    this.particles.push({
      kind,
      p,
      v,
      size: THREE.MathUtils.lerp(fluid.size[0], fluid.size[1], Math.random()),
      color: new THREE.Color(fluid.colors[Math.floor(Math.random() * fluid.colors.length)]),
      life: 4,
    });
  }

  // ── Simulation ───────────────────────────────────────────────────────

  private simulate(dt: number) {
    const world = this.physics.world;
    const playerHandle = this.player.collider.handle;
    const step = new THREE.Vector3();
    let n = 0;

    for (let i = this.particles.length - 1; i >= 0; i--) {
      const pt = this.particles[i];
      pt.life -= dt;
      pt.v.y += GRAVITY * dt;
      step.copy(pt.v).multiplyScalar(dt);
      const len = step.length();

      let hit: RAPIER.RayColliderIntersection | null = null;
      if (len > 1e-6) {
        this.ray.origin = { x: pt.p.x, y: pt.p.y, z: pt.p.z };
        this.ray.dir = { x: step.x / len, y: step.y / len, z: step.z / len };
        hit = world.castRayAndGetNormal(this.ray, len, true, undefined, undefined, undefined, undefined, (c) => c.handle !== playerHandle);
      }

      if (hit) {
        const point = pt.p.clone().addScaledVector(step, hit.timeOfImpact / len);
        const normal = new THREE.Vector3(hit.normal.x, hit.normal.y, hit.normal.z);
        const obj = this.registry.fromCollider(hit.collider);
        if (obj?.isMovable) {
          // Leichte Gegenstände werden vom Strahl angeschubst
          const imp = pt.v.clone().multiplyScalar(pt.kind === 'puke' ? 0.004 : 0.0015);
          obj.body.applyImpulseAtPoint(imp, point, true);
        }
        this.addToPuddle(pt.kind, point, normal, obj, pt.size);
        this.particles.splice(i, 1);
        continue;
      }
      if (pt.life <= 0) {
        this.particles.splice(i, 1);
        continue;
      }
      pt.p.add(step);
    }

    // Instanzen schreiben
    for (const pt of this.particles) {
      if (n >= MAX_PARTICLES) break;
      this.dummy.position.copy(pt.p);
      // In Flugrichtung gestreckt, wirkt wie ein Strahl
      const speed = pt.v.length();
      this.dummy.scale.set(pt.size, pt.size, pt.size * (1 + speed * 0.9));
      this.dummy.lookAt(this.tmp.copy(pt.p).add(pt.v));
      this.dummy.updateMatrix();
      this.instanced.setMatrixAt(n, this.dummy.matrix);
      this.instanced.setColorAt(n, pt.color);
      n++;
    }
    this.instanced.count = n;
    this.instanced.instanceMatrix.needsUpdate = true;
    if (this.instanced.instanceColor) this.instanced.instanceColor.needsUpdate = true;
  }

  private addToPuddle(kind: Fluid, point: THREE.Vector3, normal: THREE.Vector3, obj: GameObject | null, size: number) {
    // Auf beweglichen Gegenständen klebt die Pfütze am Gegenstand
    const parent = obj?.isSimulated ? obj.root : this.scene;
    const local = parent.worldToLocal(point.clone());
    // Fläche pro Tropfen: volle Blase (6 s) ergibt eine Pfütze von gut 50 cm Durchmesser
    const areaFactor = { pee: 0.9, puke: 1.6, spit: 1.4, sneeze: 1.2 }[kind];
    const area = Math.PI * (size * areaFactor) ** 2;

    let best: Puddle | null = null;
    let bestDist = Infinity;
    for (const p of this.puddles) {
      if (p.kind !== kind || p.parent !== parent || p.normal.dot(normal) < 0.9) continue;
      const d = p.mesh.position.distanceTo(local);
      if (d < p.radius * 0.8 + 0.02 && d < bestDist) {
        best = p;
        bestDist = d;
      }
    }

    if (best) {
      const maxRadius = kind === 'puke' ? 0.45 : 0.35;
      best.radius = Math.min(maxRadius, Math.sqrt(best.radius ** 2 + area / Math.PI));
      best.mesh.scale.set(best.radius * best.mesh.userData.sx, best.radius, 1);
      return;
    }

    if (this.puddles.length >= MAX_PUDDLES) {
      const old = this.puddles.shift()!;
      old.parent.remove(old.mesh);
    }
    const mesh = new THREE.Mesh(this.puddleGeometry, this.puddleMaterials[kind]);
    mesh.userData.sx = 0.75 + Math.random() * 0.5;
    const radius = Math.sqrt(area / Math.PI) * 1.5;
    mesh.scale.set(radius * mesh.userData.sx, radius, 1);
    // Ausrichtung an der Oberflächennormale (im lokalen Raum des Elternobjekts)
    const worldQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    const parentQ = parent.getWorldQuaternion(new THREE.Quaternion()).invert();
    mesh.quaternion.copy(parentQ.multiply(worldQ)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.random() * Math.PI));
    const localNormal = normal.clone().applyQuaternion(parent.getWorldQuaternion(new THREE.Quaternion()).invert());
    mesh.position.copy(local).addScaledVector(localNormal, 0.002);
    mesh.renderOrder = 1;
    mesh.receiveShadow = true;
    parent.add(mesh);
    this.puddles.push({ kind, mesh, radius, normal: normal.clone(), parent });
  }

  // ── Kacken ───────────────────────────────────────────────────────────

  private async spawnPoop() {
    const t = this.player.body.translation();
    const look = this.camera.getWorldDirection(this.forward);
    const back = new THREE.Vector3(-look.x, 0, -look.z).normalize().multiplyScalar(0.25);
    const obj = await createObject(
      {
        id: `poop-${++this.poopCount}`,
        label: 'Kothaufen',
        kind: 'movable',
        model: { type: 'placeholder', shape: 'poop', color: '#5b3a1f' },
        position: [t.x + back.x, 0.15, t.z + back.z],
        rotation: [0, Math.random() * 360, 0],
        scale: 0.85 + Math.random() * 0.4,
        mass: 0.4,
        bounciness: 0,
      },
      this.physics,
    );
    this.registry.add(obj);
    this.scene.add(obj.root);
    this.poops.push(obj);
    if (this.poops.length > MAX_POOPS) {
      const old = this.poops.shift()!;
      if (this.registry.all.includes(old)) {
        this.registry.remove(old);
        this.scene.remove(old.root);
        this.physics.world.removeRigidBody(old.body);
      }
    }
    
  }

  // ── Kamera ───────────────────────────────────────────────────────────

  private applyCameraEffects(dt: number) {
    // Hocke beim Kacken
    const squat = this.poopTime >= 0 ? 0.55 : 0;
    this.crouch = THREE.MathUtils.damp(this.crouch, squat, 8, dt);
    this.camera.position.y -= this.crouch;
    if (this.poopTime >= 0) this.camera.rotateZ(Math.sin(this.time * 30) * 0.004);

    // Würgen: Kopf nach vorn, Wackeln
    if (this.pukeTime >= 0) {
      const k = Math.sin((this.pukeTime / PUKE_DURATION) * Math.PI);
      this.camera.rotateX(-0.12 * k + Math.sin(this.time * 40) * 0.01 * k);
      this.camera.rotateZ(Math.sin(this.time * 23) * 0.015 * k);
    }

    // Niesen: Kopf hebt sich beim Ansetzen und ruckt dann nach vorn
    if (this.sneezeTime >= 0) {
      const t = this.sneezeTime;
      const lift = t < SNEEZE_WINDUP ? (t / SNEEZE_WINDUP) * 0.12 : 0;
      const jerk = t >= SNEEZE_WINDUP ? Math.sin(Math.min(1, (t - SNEEZE_WINDUP) / 0.4) * Math.PI) * 0.22 : 0;
      this.camera.rotateX(lift - jerk);
    }
  }

  private makePuddleMaterial(kind: Fluid) {
    return new THREE.MeshStandardMaterial({
      color: FLUIDS[kind].puddle,
      alphaMap: this.softEdge,
      roughness: 0.06,
      metalness: 0,
      transparent: true,
      opacity: FLUIDS[kind].opacity,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
      side: THREE.DoubleSide,
    });
  }
}

function softEdgeTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const g = canvas.getContext('2d')!;
  g.fillStyle = '#000';
  g.fillRect(0, 0, 128, 128);
  // Mehrere versetzte Flecken ergeben einen unregelmäßigen Umriss
  for (let i = 0; i < 7; i++) {
    const a = (i / 7) * Math.PI * 2;
    const cx = 64 + Math.cos(a) * 14;
    const cy = 64 + Math.sin(a) * 14;
    const grad = g.createRadialGradient(cx, cy, 0, cx, cy, 46);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.7, 'rgba(255,255,255,0.9)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
  }
  return new THREE.CanvasTexture(canvas);
}

function jitter(amount: number) {
  return new THREE.Vector3((Math.random() - 0.5) * 2 * amount, (Math.random() - 0.5) * 2 * amount, (Math.random() - 0.5) * 2 * amount);
}

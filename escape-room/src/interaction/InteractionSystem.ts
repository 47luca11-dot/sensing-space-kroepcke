import * as THREE from 'three';
import type { PlayerController } from '../controls/PlayerController';
import type { Input } from '../controls/Input';
import type { EventBus } from '../game/events';
import type { GameObject } from '../objects/GameObject';
import type { ObjectRegistry } from '../objects/ObjectFactory';
import { Groups, RAPIER, type PhysicsWorld } from '../physics/PhysicsWorld';
import type { Hud } from '../ui/Hud';

const REACH = 2.6;
const HOLD_MIN = 0.5;
const HOLD_MAX = 2.0;
const HOLD_DROP = 0.08;
const THROW_SPEED = 7;
const ROTATE_SENSITIVITY = 0.006;
/** Folgt der Gegenstand zu weit nicht (z. B. hinter einer Wand eingeklemmt), wird er losgelassen. */
const BREAK_DISTANCE = 1.2;
/** So lange darf ein Gegenstand brauchen, bis er in der Hand ankommt (z. B. vom Boden hochgeholt). */
const ARRIVE_TIMEOUT = 1.5;

const PUNCH_REACH = 1.0;
const PUNCH_COOLDOWN = 0.35;
const STAB_DURATION = 0.32;
const STAB_HIT_TIME = 0.13;
const EAT_DURATION = 0.9;
const EAT_BITE_TIME = 0.5;
/** Weiter als so zeigt der Finger nicht „auf“ etwas, sondern nur in die Richtung. */
const POINT_RANGE = 15;

const IDENTITY = { x: 0, y: 0, z: 0, w: 1 };

type HeldAction = { kind: 'stab' | 'eat'; t: number; done: boolean };

/**
 * Anvisieren, Hervorheben, Aufnehmen, Halten, Drehen, Werfen und Bedienen,
 * dazu Schlagen (Linksklick, leere Hand), Zeigen (rechte Maustaste, leere Hand),
 * Zustechen (Linksklick mit Messer) und Essen/Trinken (F).
 * Gehaltene Gegenstände bleiben dynamische Körper: Sie werden per Geschwindigkeit zum Haltepunkt
 * gezogen und stoßen daher weiterhin an Wände und andere Gegenstände.
 */
export class InteractionSystem {
  private focused: GameObject | null = null;
  private held: GameObject | null = null;
  private holdDistance = 1;
  private holdTime = 0;
  private arrived = false;
  /** Drehung des gehaltenen Gegenstands relativ zur Kamera. */
  private readonly holdRotation = new THREE.Quaternion();
  private action: HeldAction | null = null;
  private punchCooldown = 0;
  private pointing = false;
  private readonly pointTarget = new THREE.Vector3();
  /** Verbleibende Bissen je Essen */
  private readonly bitesLeft = new Map<GameObject, number>();

  private readonly ray = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -1 });
  private readonly aimAssist = new RAPIER.Ball(0.05);
  /** Getroffene Stelle des anvisierten Objekts (Weltkoordinaten). */
  private readonly focusPoint = new THREE.Vector3();
  private readonly tmpDir = new THREE.Vector3();
  private readonly tmpEye = new THREE.Vector3();
  private readonly tmpTarget = new THREE.Vector3();
  private readonly tmpQuat = new THREE.Quaternion();

  constructor(
    private readonly physics: PhysicsWorld,
    private readonly registry: ObjectRegistry,
    private readonly player: PlayerController,
    private readonly camera: THREE.Camera,
    private readonly input: Input,
    private readonly hud: Hud,
    private readonly events: EventBus,
    private readonly scene: THREE.Scene,
  ) {}

  get heldObject() {
    return this.held;
  }

  get focusedObject() {
    return this.focused;
  }

  /** Ziel des ausgestreckten Zeigefingers (Weltkoordinaten) oder null. */
  get pointAt(): THREE.Vector3 | null {
    return this.pointing ? this.pointTarget : null;
  }

  /** Rechte Maustaste gedrückt, während man etwas hält: Maus dreht den Gegenstand statt der Kamera. */
  get isRotating() {
    return !!this.held && this.input.isMouseDown(2);
  }

  rotateHeld(dx: number, dy: number) {
    const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), dx * ROTATE_SENSITIVITY);
    const pitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), dy * ROTATE_SENSITIVITY);
    // Vormultiplizieren = Drehung um die Achsen der Kamera, nicht des Gegenstands.
    this.holdRotation.premultiply(pitch).premultiply(yaw);
  }

  /** Einmal pro Bild: Ziel bestimmen, Hinweise zeigen, Tasten auswerten. */
  update(dt: number) {
    this.punchCooldown -= dt;
    if (!this.input.active) {
      this.setFocus(null);
      this.pointing = false;
      this.hud.setPrompt(null);
      this.hud.setHint(null);
      return;
    }

    this.setFocus(this.pick());
    const focused = this.focused;
    const behavior = focused?.behavior ?? null;
    const held = this.held;

    if (held) {
      const wheel = this.input.consumeWheel();
      // Große Gegenstände nicht so nah heranholen, dass sie das Bild füllen
      const min = Math.max(HOLD_MIN, held.radius + 0.35);
      if (wheel) this.holdDistance = THREE.MathUtils.clamp(this.holdDistance - wheel * 0.0015, min, HOLD_MAX);
    }

    // Zeigen: rechte Maustaste mit leerer Hand
    this.pointing = !held && this.input.isMouseDown(2);
    let pointedLabel: string | null = null;
    if (this.pointing) {
      const hit = this.castAny(POINT_RANGE);
      if (hit) {
        this.pointTarget.copy(hit.point);
        pointedLabel = hit.object?.label ?? null;
      } else {
        this.pointTarget.copy(this.camera.position).addScaledVector(this.camera.getWorldDirection(this.tmpDir), POINT_RANGE);
      }
    }

    // Hinweise
    if (this.pointing) this.hud.setPrompt(pointedLabel ? `Zeigen · ${pointedLabel}` : 'Zeigen');
    else if (behavior) this.hud.setPrompt(behavior.prompt(held));
    else if (focused?.isMovable && !held) this.hud.setPrompt(`[E] Aufnehmen · ${focused.label}`);
    else this.hud.setPrompt(null);
    this.hud.setHint(held ? `${held.label} — ${this.holdHint(held)}` : null);

    // Aktionen
    if (this.input.wasPressed('KeyE')) {
      if (behavior) {
        this.events.emit('interacted', { object: focused!, point: this.focusPoint.clone() });
        if (behavior.interact(held) === 'consume-held' && held) this.consume(held);
      } else if (held) {
        this.release(0);
      } else if (focused?.isMovable) {
        this.pickUp(focused);
      }
    } else if (this.input.wasClicked(0)) {
      if (held?.config.weapon) {
        if (!this.action) this.action = { kind: 'stab', t: 0, done: false };
      } else if (held) {
        this.release(THROW_SPEED);
      } else if (this.punchCooldown <= 0) {
        this.punch();
      }
    } else if (this.input.wasPressed('KeyF') && held?.config.consumable && !this.action) {
      this.action = { kind: 'eat', t: 0, done: false };
    }
  }

  /** In jedem Physikschritt: gehaltenen Gegenstand zum Haltepunkt vor der Kamera ziehen. */
  fixedUpdate(dt: number) {
    for (const obj of this.registry.all) obj.behavior?.fixedUpdate?.(dt);

    const held = this.held;
    if (!held) {
      this.action = null;
      return;
    }
    const body = held.body;
    const look = this.player.lookQuaternion;
    const eye = this.player.eyePosition(this.tmpEye);
    // Unter dem Fadenkreuz halten, große Dinge tiefer, damit man darüber hinwegsehen kann
    let drop = HOLD_DROP + held.radius * 0.6;
    let distance = this.holdDistance;

    // Zustechen und Essen verschieben den Haltepunkt
    const action = this.action;
    if (action) {
      action.t += dt;
      if (action.kind === 'stab') {
        distance += Math.sin(Math.PI * Math.min(1, action.t / STAB_DURATION)) * 0.55;
        if (!action.done && action.t >= STAB_HIT_TIME) {
          action.done = true;
          this.strike('stab', distance + 0.35);
        }
        if (action.t >= STAB_DURATION) this.action = null;
      } else {
        const k = Math.sin(Math.PI * Math.min(1, action.t / EAT_DURATION));
        distance = THREE.MathUtils.lerp(distance, 0.2, k);
        drop = THREE.MathUtils.lerp(drop, 0.13, k);
        if (!action.done && action.t >= EAT_BITE_TIME) {
          action.done = true;
          this.bite(held);
          if (!this.held) return; // aufgegessen
        }
        if (action.t >= EAT_DURATION) this.action = null;
      }
    }

    const target = this.tmpTarget.set(0, -drop, -distance).applyQuaternion(look).add(eye);
    const p = body.translation();
    const delta = target.sub(this.tmpDir.set(p.x, p.y, p.z));
    // Erst prüfen, wenn der Gegenstand einmal in der Hand angekommen ist; sonst fiele alles,
    // was man aus über einem Meter Entfernung aufnimmt, sofort wieder herunter.
    this.holdTime += dt;
    if (delta.length() < 0.25) this.arrived = true;
    if ((this.arrived && delta.length() > BREAK_DISTANCE) || (!this.arrived && this.holdTime > ARRIVE_TIMEOUT)) {
      this.release(0);
      return;
    }
    // Weiche Feder: einen Teil des Abstands pro Schritt schließen, Geschwindigkeit begrenzen.
    const velocity = delta.multiplyScalar(action?.kind === 'stab' ? 25 : 12);
    if (velocity.length() > 12) velocity.setLength(12);
    body.setLinvel(velocity, true);

    // Drehung: Fehler-Quaternion → Winkelgeschwindigkeit
    const r = body.rotation();
    const current = this.tmpQuat.set(r.x, r.y, r.z, r.w);
    const wanted = new THREE.Quaternion().multiplyQuaternions(look, this.holdRotation);
    const error = wanted.multiply(current.invert()).normalize();
    if (error.w < 0) error.set(-error.x, -error.y, -error.z, -error.w);
    // Rundungsfehler können w knapp über 1 heben; ohne Klemmen entsteht NaN und Rapier legt den Körper still.
    const w = Math.min(1, error.w);
    const angle = 2 * Math.acos(w);
    const s = Math.sqrt(Math.max(0, 1 - w * w));
    if (s < 1e-4) {
      body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    } else {
      const gain = Math.min(angle * 10, 15) / s;
      body.setAngvel({ x: error.x * gain, y: error.y * gain, z: error.z * gain }, true);
    }
  }

  private holdHint(held: GameObject) {
    const parts = ['[E] Ablegen', held.config.weapon ? '[Linksklick] Zustechen' : '[Linksklick] Werfen'];
    const c = held.config.consumable;
    if (c) parts.push(c.kind === 'drink' ? '[F] Trinken' : '[F] Essen');
    parts.push('[Rechte Maustaste + Maus] Drehen', '[Mausrad] Näher / weiter');
    return parts.join(' · ');
  }

  /** Faustschlag mit leerer Hand. */
  private punch() {
    this.punchCooldown = PUNCH_COOLDOWN;
    const hit = this.strike('punch', PUNCH_REACH);
    this.events.emit('punched', { hit });
  }

  /** Trifft das Erste in Blickrichtung: Figuren reagieren selbst, bewegliche Dinge bekommen einen Stoß. */
  private strike(kind: 'punch' | 'stab', reach: number) {
    const hit = this.castAny(reach);
    if (!hit) return false;
    const dir = this.camera.getWorldDirection(new THREE.Vector3());
    if (hit.object?.behavior?.onHit) {
      hit.object.behavior.onHit(kind, hit.point, hit.normal, dir);
    } else if (hit.object && !hit.object.body.isFixed() && hit.object.body.isDynamic()) {
      const body = hit.object.body;
      const impulse = Math.min(body.mass() * (kind === 'stab' ? 2 : 3.5), 25);
      body.applyImpulseAtPoint({ x: dir.x * impulse, y: dir.y * impulse, z: dir.z * impulse }, hit.point, true);
    }
    return true;
  }

  /** Ein Bissen oder Schluck. Essen schrumpft und ist nach den Bissen weg, Getränke bleiben. */
  private bite(obj: GameObject) {
    const c = obj.config.consumable!;
    if (c.kind === 'drink') {
      // Getränke mit begrenzten Schlucken (Bier) werden leer, sonst nie
      let finished = false;
      if (c.sips) {
        const left = (this.bitesLeft.get(obj) ?? c.sips) - 1;
        this.bitesLeft.set(obj, left);
        finished = left <= 0;
      }
      this.events.emit('consumed', { object: obj, kind: 'drink', finished, alcohol: c.alcohol ?? 0 });
      if (finished) {
        obj.config.consumable = undefined;
        obj.config.label = `Leere ${obj.config.label}flasche`;
      }
      return;
    }
    const left = (this.bitesLeft.get(obj) ?? c.bites ?? 3) - 1;
    this.bitesLeft.set(obj, left);
    obj.model.object.scale.multiplyScalar(0.78);
    const finished = left <= 0;
    this.events.emit('consumed', { object: obj, kind: 'food', finished, alcohol: 0 });
    if (finished) this.consume(obj);
  }

  /** Erster Treffer entlang der Blickrichtung, ohne Spieler und gehaltenen Gegenstand. */
  private castAny(range: number) {
    const origin = this.camera.position;
    const dir = this.camera.getWorldDirection(this.tmpDir);
    this.ray.origin = { x: origin.x, y: origin.y, z: origin.z };
    this.ray.dir = { x: dir.x, y: dir.y, z: dir.z };
    const playerHandle = this.player.collider.handle;
    const held = this.held;
    const hit = this.physics.world.castRayAndGetNormal(this.ray, range, true, undefined, undefined, undefined, undefined, (c) =>
      c.handle !== playerHandle && (!held || this.registry.fromCollider(c) !== held),
    );
    if (!hit) return null;
    return {
      point: origin.clone().addScaledVector(dir, hit.timeOfImpact),
      normal: new THREE.Vector3(hit.normal.x, hit.normal.y, hit.normal.z),
      object: this.registry.fromCollider(hit.collider),
    };
  }

  private pick(): GameObject | null {
    const origin = this.camera.position;
    const dir = this.camera.getWorldDirection(this.tmpDir);
    this.ray.origin = { x: origin.x, y: origin.y, z: origin.z };
    this.ray.dir = { x: dir.x, y: dir.y, z: dir.z };
    const playerHandle = this.player.collider.handle;
    const held = this.held;
    const notSelf = (c: RAPIER.Collider) => c.handle !== playerHandle && (!held || this.registry.fromCollider(c) !== held);
    const interactive = (obj: GameObject | null) => !!obj && (obj.isMovable || !!obj.behavior);

    const world = this.physics.world;
    const hit = world.castRay(this.ray, REACH, true, undefined, undefined, undefined, undefined, notSelf);
    const obj = hit ? this.registry.fromCollider(hit.collider) : null;
    if (interactive(obj)) {
      this.focusPoint.copy(dir).multiplyScalar(hit!.timeOfImpact).add(origin);
      return obj;
    }

    // Zielhilfe für kleine Dinge (Schlüssel): dünne Kugel entlang der Blickrichtung,
    // die nur interaktive Objekte trifft und nicht weiter reicht als der direkte Treffer.
    const cast = world.castShape(this.ray.origin, IDENTITY, this.ray.dir, this.aimAssist, 0, REACH, true, undefined, undefined, undefined, undefined, (c) =>
      notSelf(c) && interactive(this.registry.fromCollider(c)),
    );
    if (cast && (!hit || cast.time_of_impact <= hit.timeOfImpact + 0.1)) {
      this.focusPoint.copy(dir).multiplyScalar(cast.time_of_impact).add(origin);
      return this.registry.fromCollider(cast.collider);
    }
    return null;
  }

  private setFocus(obj: GameObject | null) {
    if (obj === this.focused) return;
    this.focused?.setHighlight(false);
    this.focused = obj;
    obj?.setHighlight(true);
  }

  private pickUp(obj: GameObject) {
    this.held = obj;
    this.holdTime = 0;
    this.arrived = false;
    this.action = null;
    this.setFocus(null);
    const body = obj.body;
    body.setGravityScale(0, true);
    obj.setCollisionGroups(Groups.held);

    // Abstand: so, dass der Gegenstand nicht in die Kamera ragt
    this.holdDistance = THREE.MathUtils.clamp(0.45 + obj.radius * 1.5, 0.7, HOLD_MAX);
    if (obj.config.weapon) {
      // Klinge (lokal +X) zeigt nach vorn, leicht nach unten geneigt
      this.holdDistance = 0.55;
      this.holdRotation.setFromEuler(new THREE.Euler(0, Math.PI / 2, -0.15, 'YXZ'));
    } else {
      const r = body.rotation();
      this.holdRotation
        .copy(this.player.lookQuaternion)
        .invert()
        .multiply(new THREE.Quaternion(r.x, r.y, r.z, r.w));
    }
    this.events.emit('item-picked-up', obj);
  }

  /** Loslassen; speed > 0 wirft in Blickrichtung. */
  private release(speed: number) {
    const obj = this.held;
    if (!obj) return;
    this.held = null;
    this.action = null;
    const body = obj.body;
    body.setGravityScale(1, true);
    obj.setCollisionGroups(Groups.prop);
    const v = body.linvel();
    if (speed > 0) {
      const dir = this.camera.getWorldDirection(this.tmpDir).multiplyScalar(speed);
      body.setLinvel({ x: dir.x, y: dir.y, z: dir.z }, true);
    } else {
      body.setLinvel({ x: v.x * 0.3, y: v.y * 0.3, z: v.z * 0.3 }, true);
      body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    }
  }

  /** Gegenstand wurde benutzt oder entfernt und verschwindet (z. B. Schlüssel steckt im Schloss). */
  consume(obj: GameObject) {
    if (this.held === obj) {
      this.held = null;
      this.action = null;
    }
    if (this.focused === obj) this.focused = null;
    this.registry.remove(obj);
    this.scene.remove(obj.root);
    this.physics.world.removeRigidBody(obj.body);
  }
}

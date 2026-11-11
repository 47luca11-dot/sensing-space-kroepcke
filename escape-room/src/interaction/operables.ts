import * as THREE from 'three';
import type { EventBus } from '../game/events';
import type { GameObject } from '../objects/GameObject';
import type { Room } from '../room/Room';

/** Verhalten eines bedienbaren Objekts (Tür, Schalter …). */
export interface Operable {
  /** Hinweistext beim Anvisieren. `held` = Gegenstand in der Hand oder null. */
  prompt(held: GameObject | null): string;
  /** Taste E. Gibt 'consume-held' zurück, wenn der gehaltene Gegenstand verbraucht wurde (z. B. Schlüssel). */
  interact(held: GameObject | null): 'consume-held' | void;
  fixedUpdate?(dt: number): void;
  /** Treffer durch Faust oder Messer. `point` und `dir` in Weltkoordinaten. */
  onHit?(kind: 'punch' | 'stab', point: THREE.Vector3, normal: THREE.Vector3, dir: THREE.Vector3): void;
  /** Tanzen für die angegebene Dauer (z. B. wenn Geld fliegt). */
  dance?(seconds: number): void;
}

/** Bildschirmdialoge, die bedienbare Objekte öffnen dürfen. */
export interface OperableUi {
  /** Zahlenfeld; `check` gibt true zurück, wenn der Code stimmt (Dialog schließt dann). */
  openKeypad(title: string, digits: number, check: (code: string) => boolean): void;
  showNote(text: string): void;
}

export interface OperableContext {
  events: EventBus;
  room: Room;
  ui: OperableUi;
}

export function createOperable(obj: GameObject, ctx: OperableContext): Operable | null {
  const cfg = obj.config.operable;
  if (!cfg) return null;
  switch (cfg.type) {
    case 'door':
      return new DoorBehavior(obj, ctx, cfg.requires, cfg.code, cfg.openAngle ?? 100);
    case 'switch':
      return new SwitchBehavior(obj, ctx, cfg.target);
    case 'note':
      return new NoteBehavior(ctx, cfg.text);
    case 'dummy':
      return new DummyBehavior(obj, ctx, cfg.lines, cfg.toughness ?? 6);
  }
}

const DOOR_SPEED = THREE.MathUtils.degToRad(110); // Grad pro Sekunde

class DoorBehavior implements Operable {
  private locked: boolean;
  private open = false;
  private wasOpened = false;
  private angle = 0;
  private readonly baseRotation: THREE.Quaternion;
  private readonly openAngle: number;

  constructor(
    private readonly obj: GameObject,
    private readonly ctx: OperableContext,
    private readonly requires: string | undefined,
    private readonly code: string | undefined,
    openAngleDegrees: number,
  ) {
    this.locked = !!requires || !!code;
    this.openAngle = THREE.MathUtils.degToRad(openAngleDegrees);
    const r = obj.body.rotation();
    this.baseRotation = new THREE.Quaternion(r.x, r.y, r.z, r.w);
  }

  prompt(held: GameObject | null) {
    const name = this.obj.label;
    if (this.locked && this.code) return `[E] Zahlenschloss · ${name}`;
    if (this.locked) {
      return this.isKey(held) ? `[E] ${held!.label} benutzen` : 'Verschlossen – hier fehlt ein Schlüssel';
    }
    return this.open ? `[E] ${name} schließen` : `[E] ${name} öffnen`;
  }

  interact(held: GameObject | null) {
    if (this.locked && this.code) {
      const code = this.code;
      this.ctx.ui.openKeypad(this.obj.label, code.length, (entered) => {
        if (entered !== code) return false;
        this.locked = false;
        this.open = true;
        this.ctx.events.emit('door-unlocked', this.obj);
        this.ctx.events.emit('message', 'Das Schloss gibt nach.');
        return true;
      });
      return;
    }
    if (this.locked) {
      if (!this.isKey(held)) {
        this.ctx.events.emit('message', `${this.obj.label}: verschlossen.`);
        return;
      }
      this.locked = false;
      this.open = true;
      this.ctx.events.emit('door-unlocked', this.obj);
      this.ctx.events.emit('message', 'Klick – der Schlüssel passt!');
      return 'consume-held' as const;
    }
    this.open = !this.open;
  }

  fixedUpdate(dt: number) {
    const target = this.open ? this.openAngle : 0;
    if (this.angle === target) return;
    const step = DOOR_SPEED * dt;
    this.angle = Math.abs(target - this.angle) <= step ? target : this.angle + Math.sign(target - this.angle) * step;
    const q = this.baseRotation.clone().multiply(new THREE.Quaternion().setFromAxisAngle(THREE.Object3D.DEFAULT_UP, this.angle));
    this.obj.body.setNextKinematicRotation(q);
    if (this.angle === this.openAngle && !this.wasOpened) {
      this.wasOpened = true;
      this.ctx.events.emit('door-opened', this.obj);
    }
  }

  private isKey(held: GameObject | null) {
    return !!held && !!this.requires && held.config.itemId === this.requires;
  }
}

class DummyBehavior implements Operable {
  private line = 0;
  private damage = 0;
  private down = false;
  private danceTime = 0;
  private beat = Math.random() * 4;
  private readonly cuts: THREE.Object3D;
  private static readonly cutGeometry = new THREE.PlaneGeometry(0.06, 0.009);
  private static readonly cutMaterial = new THREE.MeshStandardMaterial({
    color: '#4a1414',
    roughness: 0.4,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    side: THREE.DoubleSide,
  });

  constructor(
    private readonly obj: GameObject,
    private readonly ctx: OperableContext,
    private readonly lines: string[],
    private readonly toughness: number,
  ) {
    this.cuts = obj.root.getObjectByName('cuts') ?? obj.root;
  }

  prompt() {
    return this.down ? `${this.obj.label} (umgekippt)` : `[E] Ansprechen · ${this.obj.label}`;
  }

  interact() {
    if (this.down) {
      this.ctx.events.emit('message', `${this.obj.label}: …`);
      return;
    }
    const text = this.lines[this.line % this.lines.length];
    this.line++;
    this.ctx.events.emit('message', `${this.obj.label}: „${text}“`);
  }

  onHit(kind: 'punch' | 'stab', point: THREE.Vector3, normal: THREE.Vector3, dir: THREE.Vector3) {
    const body = this.obj.body;
    const strength = kind === 'stab' ? 14 : 22;
    body.applyImpulseAtPoint({ x: dir.x * strength, y: dir.y * strength * 0.3, z: dir.z * strength }, point, true);
    if (kind === 'stab') this.addCut(point, normal);
    this.damage += kind === 'stab' ? 1 : 0.4;
    if (!this.down && this.damage >= this.toughness) {
      this.down = true;
      // Kopf nach hinten stoßen: kippt um und bleibt liegen
      const t = body.translation();
      body.applyImpulseAtPoint({ x: dir.x * 60, y: 0, z: dir.z * 60 }, { x: t.x, y: t.y + 1.5, z: t.z }, true);
      body.setAngularDamping(0.4);
      this.ctx.events.emit('message', `${this.obj.label} kippt um.`);
    }
  }

  dance(seconds: number) {
    if (!this.down) this.danceTime = Math.max(this.danceTime, seconds);
  }

  /**
   * Tanz als Regelung statt fester Animation: Die Figur wird zu einer im Takt
   * schwingenden Neigung gedreht, dreht sich dabei und hüpft leicht. So bleibt sie
   * physikalisch (Schläge wirken weiter) und kippt nicht um.
   */
  fixedUpdate(dt: number) {
    if (this.danceTime <= 0 || this.down) return;
    this.danceTime -= dt;
    this.beat += dt;
    const body = this.obj.body;
    const r = body.rotation();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(new THREE.Quaternion(r.x, r.y, r.z, r.w));
    const phase = this.beat * Math.PI * 2 * 1.1; // ca. 130 BPM halbe Takte
    const sway = Math.sin(phase) * 0.18;
    const nod = Math.sin(phase * 0.5) * 0.08;
    const desired = new THREE.Vector3(Math.sin(sway), Math.cos(sway), Math.sin(nod)).normalize();
    const correct = new THREE.Vector3().crossVectors(up, desired).multiplyScalar(10);
    body.setAngvel({ x: correct.x, y: Math.sin(this.beat * 1.3) * 1.8, z: correct.z }, true);
    // Kleiner Hüpfer auf jedem Schlag, nur vom Boden aus
    if (Math.sin(phase) > 0.97 && body.translation().y < 0.05 && body.linvel().y <= 0.05) {
      body.applyImpulse({ x: 0, y: body.mass() * 1.1, z: 0 }, true);
    }
  }

  private addCut(point: THREE.Vector3, normal: THREE.Vector3) {
    const cut = new THREE.Mesh(DummyBehavior.cutGeometry, DummyBehavior.cutMaterial);
    const local = this.cuts.worldToLocal(point.clone().addScaledVector(normal, 0.002));
    cut.position.copy(local);
    const worldQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    const parentQ = this.cuts.getWorldQuaternion(new THREE.Quaternion()).invert();
    cut.quaternion.copy(parentQ.multiply(worldQ)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), (Math.random() - 0.5) * 1.2));
    this.cuts.add(cut);
  }
}

class NoteBehavior implements Operable {
  constructor(
    private readonly ctx: OperableContext,
    private readonly text: string,
  ) {}

  prompt() {
    return '[E] Lesen';
  }

  interact() {
    this.ctx.ui.showNote(this.text);
  }
}

class SwitchBehavior implements Operable {
  private readonly lever: THREE.Object3D | undefined;

  constructor(
    obj: GameObject,
    private readonly ctx: OperableContext,
    private readonly target: string,
  ) {
    this.lever = obj.root.getObjectByName('lever');
    this.updateLever();
  }

  prompt() {
    return this.ctx.room.isLightOn(this.target) ? '[E] Licht ausschalten' : '[E] Licht einschalten';
  }

  interact() {
    const on = !this.ctx.room.isLightOn(this.target);
    this.ctx.room.setLight(this.target, on);
    this.updateLever();
    this.ctx.events.emit('light-toggled', { id: this.target, on });
  }

  private updateLever() {
    if (this.lever) this.lever.rotation.x = this.ctx.room.isLightOn(this.target) ? -0.35 : 0.35;
  }
}

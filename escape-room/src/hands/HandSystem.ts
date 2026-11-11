import * as THREE from 'three';
import { HANDS } from '../config/hands';
import type { GameObject } from '../objects/GameObject';
import { HandRig, type HandPose } from './HandRig';

const POSES = {
  relaxed: { thumb: 0.15, index: 0.15, middle: 0.2, ring: 0.25, pinky: 0.3 },
  open: { thumb: 0, index: 0, middle: 0.02, ring: 0.05, pinky: 0.08 },
  point: { thumb: 0.55, index: 0, middle: 0.85, ring: 0.9, pinky: 0.9 },
  // Ausgemessen: Daumen- und Zeigefingerkuppe liegen ca. 1 cm auseinander
  pinch: { thumb: 0.6, index: 0.45, middle: 0.75, ring: 0.85, pinky: 0.9 },
  flat: { thumb: 0.05, index: 0.08, middle: 0.1, ring: 0.1, pinky: 0.12 },
  fist: { thumb: 0.8, index: 1, middle: 1, ring: 1, pinky: 1 },
} satisfies Record<string, HandPose>;

const PUNCH_DURATION = 0.3;
const PUNCH_PEAK = 0.08;

const grip = (curl: number): HandPose => ({
  thumb: Math.min(0.75, curl * 0.9),
  index: curl,
  middle: curl * 1.05,
  ring: curl * 1.08,
  pinky: curl * 1.12,
});

/** Haltung eines gehaltenen Gegenstands, abgeleitet aus seiner Größe in Blickrichtung. */
type GripKind = 'pinch' | 'grip' | 'two-hands';
const PINCH_MAX_THICKNESS = 0.016; // halbe Dicke, darunter: zwischen Daumen und Zeigefinger
const TWO_HANDS_MIN_HALF_WIDTH = 0.17;

const REACH_DURATION = 0.5;
const REACH_MAX_DISTANCE = 0.55;

interface Placement {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  pose: HandPose;
  visible: boolean;
}

/** Drehung im Kameraraum: Rollen um die Fingerachse, Kippen, Schwenken nach innen. */
const orient = (roll: number, pitch: number, yaw: number) =>
  new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'));

/** Spiegelt eine Platzierung der rechten Hand auf die linke Seite. */
function smooth(t: number) {
  t = Math.min(1, Math.max(0, t));
  return t * t * (3 - 2 * t);
}

function mirror(p: Placement): Placement {
  const q = p.quaternion;
  return {
    position: p.position.clone().setX(-p.position.x),
    quaternion: new THREE.Quaternion(q.x, -q.y, -q.z, q.w),
    pose: p.pose,
    visible: p.visible,
  };
}

/**
 * Ego-Hände: hängen an der Kamera, ruhen am unteren Bildrand, greifen gehaltene Gegenstände
 * je nach Größe (Pinzettengriff, Faustgriff, beidhändig) und tippen Schalter und Türen an.
 */
export class HandSystem {
  private readonly anchors: Record<'right' | 'left', THREE.Group>;
  private reach: { point: THREE.Vector3; t: number } | null = null;
  private time = 0;
  /** Laufzeit des aktuellen Faustschlags, −1 = keiner. */
  private punchTime = -1;
  private gesture: 'punch' | 'flick' = 'punch';

  /** Kontaktpunkte im Hand-Koordinatensystem, einmal beim Laden ausgemessen. */
  private readonly palmContact: THREE.Vector3;
  private readonly knuckleContact: THREE.Vector3;
  private readonly pinchPoint: THREE.Vector3;
  private readonly pointTip: THREE.Vector3;

  private readonly tmp = new THREE.Vector3();
  private readonly box = new THREE.Box3();

  private constructor(
    private readonly camera: THREE.Camera,
    private readonly right: HandRig,
    private readonly left: HandRig,
  ) {
    this.anchors = { right: new THREE.Group(), left: new THREE.Group() };
    this.anchors.right.add(right.object);
    this.anchors.left.add(left.object);
    camera.add(this.anchors.right, this.anchors.left);
    // Schwaches Fülllicht von schräg oben, damit die Hände nicht als Silhouette erscheinen.
    const fill = new THREE.PointLight('#ffe9dc', 0.5, 1.2, 2);
    fill.position.set(0.25, 0.35, 0.1);
    camera.add(fill);

    right.snapPose(POSES.pinch);
    this.pinchPoint = right.tipLocal('thumb').add(right.tipLocal('index')).multiplyScalar(0.5);
    right.snapPose(POSES.point);
    this.pointTip = right.tipLocal('index');
    right.snapPose(POSES.flat);
    // Handflächenmitte: zwischen Handgelenk und Fingeransatz, etwas unter der Knochenlinie
    const knuckle = right.tipLocal('middle').multiplyScalar(0.42);
    this.palmContact = knuckle.add(new THREE.Vector3(0, -0.018, 0));
    // Fingeransatz auf der Handflächenseite: Hier liegt ein umgriffener Gegenstand an.
    this.knuckleContact = right.boneLocal(HANDS.fingers.middle[0]).add(new THREE.Vector3(0, -0.016, 0));
    right.snapPose(POSES.relaxed);
    left.snapPose(POSES.relaxed);

    const idle = this.idlePlacement(0);
    for (const side of ['right', 'left'] as const) {
      const p = side === 'right' ? idle : mirror(idle);
      this.anchors[side].position.copy(p.position);
      this.anchors[side].quaternion.copy(p.quaternion);
    }
  }

  static async create(camera: THREE.Camera) {
    const { right, left } = await HandRig.load();
    return new HandSystem(camera, right, left);
  }

  /** Kurzes Hinlangen zum Punkt (Weltkoordinaten), z. B. beim Drücken eines Schalters. */
  reachTo(point: THREE.Vector3) {
    this.reach = { point: point.clone(), t: 0 };
  }

  /** Faustschlag der rechten Hand nach vorn. */
  punch() {
    this.punchTime = 0;
    this.gesture = 'punch';
  }

  /** Lockeres Schnippen aus dem Handgelenk (Geld werfen). */
  flick() {
    if (this.punchTime >= 0 && this.punchTime < PUNCH_PEAK) return;
    this.punchTime = 0;
    this.gesture = 'flick';
  }

  /**
   * @param held gehaltener Gegenstand oder null
   * @param focused anvisierter Gegenstand oder null
   * @param speed Laufgeschwindigkeit in m/s (für das Mitschwingen)
   * @param pointAt Ziel des ausgestreckten Zeigefingers (Weltkoordinaten) oder null
   */
  update(dt: number, held: GameObject | null, focused: GameObject | null, speed: number, pointAt: THREE.Vector3 | null) {
    this.time += dt;
    this.camera.updateMatrixWorld();

    let right: Placement;
    let left: Placement;
    const idle = this.idlePlacement(speed);
    if (this.punchTime >= 0) this.punchTime += dt;
    if (this.punchTime > PUNCH_DURATION) this.punchTime = -1;

    if (held) {
      [right, left] = this.holdPlacements(held, idle);
    } else if (this.punchTime >= 0) {
      right = this.punchPlacement(idle, this.punchTime);
      left = mirror(idle);
    } else if (pointAt) {
      right = this.reachPlacement(pointAt, idle, 1);
      left = mirror(idle);
    } else if (this.reach) {
      this.reach.t += dt;
      const k = Math.sin(Math.PI * Math.min(1, this.reach.t / REACH_DURATION));
      right = this.reachPlacement(this.reach.point, idle, k);
      left = mirror(idle);
      if (this.reach.t >= REACH_DURATION) this.reach = null;
    } else if (focused?.isMovable) {
      // Vorfreude: rechte Hand hebt sich leicht und öffnet sich
      right = { ...idle, position: idle.position.clone().add(new THREE.Vector3(-0.02, 0.035, -0.03)), pose: POSES.open };
      left = mirror(idle);
    } else {
      right = idle;
      left = mirror(idle);
    }

    // Beim Halten und Schlagen schnell nachführen, sonst weich
    const lambda = held || this.punchTime >= 0 ? 30 : 12;
    this.apply('right', this.right, right, dt, lambda);
    this.apply('left', this.left, left, dt, lambda);
  }

  private apply(side: 'right' | 'left', rig: HandRig, p: Placement, dt: number, lambda: number) {
    const anchor = this.anchors[side];
    const k = 1 - Math.exp(-lambda * dt);
    anchor.position.lerp(p.position, k);
    anchor.quaternion.slerp(p.quaternion, k);
    anchor.visible = p.visible;
    rig.setPose(p.pose);
    rig.update(dt);
  }

  private idlePlacement(speed: number): Placement {
    const sway = Math.min(speed / 3, 1);
    const bobY = Math.sin(this.time * 9) * 0.006 * sway;
    const bobX = Math.cos(this.time * 4.5) * 0.004 * sway;
    return {
      position: new THREE.Vector3(0.2 + bobX, -0.27 + bobY, -0.36),
      quaternion: orient(-0.55, 0.3, 0.3),
      pose: POSES.relaxed,
      visible: HANDS.showIdle,
    };
  }

  private reachPlacement(worldPoint: THREE.Vector3, idle: Placement, k: number): Placement {
    const target = this.camera.worldToLocal(this.tmp.copy(worldPoint));
    // Weit entfernte Ziele: Hand nur in diese Richtung ausstrecken
    if (target.length() > REACH_MAX_DISTANCE) target.setLength(REACH_MAX_DISTANCE);
    const quaternion = orient(-0.5, 0.05, 0.12);
    const wrist = target.sub(this.pointTip.clone().applyQuaternion(quaternion));
    return {
      position: idle.position.clone().lerp(wrist, k),
      quaternion: idle.quaternion.clone().slerp(quaternion, k),
      pose: k > 0.15 ? POSES.point : POSES.relaxed,
      visible: true,
    };
  }

  /** Kurz vorschnellende Faust, Knöchel voran. */
  private punchPlacement(idle: Placement, t: number): Placement {
    const k = t < PUNCH_PEAK ? smooth(t / PUNCH_PEAK) : 1 - smooth((t - PUNCH_PEAK) / (PUNCH_DURATION - PUNCH_PEAK));
    if (this.gesture === 'flick') {
      // Hand schnellt nach vorn oben, Handfläche nach oben, Finger offen
      return {
        position: idle.position.clone().lerp(new THREE.Vector3(0.12, -0.08, -0.48), k),
        quaternion: idle.quaternion.clone().slerp(orient(-2.4, 0.45, 0.2), k),
        pose: k > 0.5 ? POSES.open : POSES.relaxed,
        visible: true,
      };
    }
    return {
      position: idle.position.clone().lerp(new THREE.Vector3(0.06, -0.13, -0.58), k),
      quaternion: idle.quaternion.clone().slerp(orient(-0.35, 0.12, 0.08), k),
      pose: POSES.fist,
      visible: true,
    };
  }

  /** Messer im Faustgriff: Hand über dem Griff, Finger schließen sich darum, Klinge zeigt nach vorn. */
  private knifePlacement(toCamera: THREE.Matrix4, idle: Placement): [Placement, Placement] {
    const handle = new THREE.Vector3(-0.05, 0, 0).applyMatrix4(toCamera);
    const quaternion = orient(-0.25, 0.05, -Math.PI / 2 + 0.25);
    const position = handle.add(new THREE.Vector3(0, 0.014, 0)).sub(this.knuckleContact.clone().applyQuaternion(quaternion));
    return [{ position, quaternion, pose: grip(0.82), visible: true }, mirror(idle)];
  }

  /** Wählt den Griff nach den Maßen des Gegenstands im Kameraraum (oder nach `grip` in der Konfiguration). */
  private holdPlacements(held: GameObject, idle: Placement): [Placement, Placement] {
    // Grenzen des Gegenstands in Kamerakoordinaten (dreht sich mit, wenn man ihn dreht)
    const toCamera = new THREE.Matrix4().copy(this.camera.matrixWorldInverse).multiply(held.root.matrixWorld);
    if (held.config.weapon === 'knife') return this.knifePlacement(toCamera, idle);
    this.box.copy(held.model.bounds).applyMatrix4(toCamera);
    const center = this.box.getCenter(new THREE.Vector3());
    const half = this.box.getSize(new THREE.Vector3()).multiplyScalar(0.5);
    const thinnest = Math.min(half.x, half.y, half.z);

    let kind: GripKind = 'grip';
    if (thinnest < PINCH_MAX_THICKNESS) kind = 'pinch';
    else if (half.x > TWO_HANDS_MIN_HALF_WIDTH) kind = 'two-hands';
    if (held.config.grip) kind = held.config.grip;

    if (kind === 'pinch') {
      const quaternion = orient(-0.9, 0.15, 0.35);
      const contact = center.clone().add(new THREE.Vector3(half.x * 0.7, 0, 0));
      const position = contact.sub(this.pinchPoint.clone().applyQuaternion(quaternion));
      return [{ position, quaternion, pose: POSES.pinch, visible: true }, mirror(idle)];
    }

    if (kind === 'two-hands') {
      const quaternion = orient(-Math.PI / 2, 0, 0.1);
      const contact = center.clone().add(new THREE.Vector3(half.x + 0.008, 0, 0));
      const position = contact.sub(this.palmContact.clone().applyQuaternion(quaternion));
      const right: Placement = { position, quaternion, pose: POSES.flat, visible: true };
      // Linke Hand an der gegenüberliegenden Seite: Spiegelung um die Objektmitte
      const left = mirror(right);
      left.position.x = 2 * center.x - right.position.x;
      return [right, left];
    }

    // Umgreifen: Fingeransatz an der rechten Kante, die Finger legen sich um die Rückseite.
    // Krümmung passend zum Radius: Fingerglieder (~3 cm) folgen einem Kreis mit Radius + Fingerdicke.
    const wrap = Math.min(half.x, half.z);
    const curl = THREE.MathUtils.clamp(0.0225 / (wrap + 0.01), 0.15, 0.85);
    // Finger leicht nach oben und innen, damit das Handgelenk nach unten aus dem Bild zeigt
    const quaternion = orient(-1.25, 0.3, 0.25);
    const contact = center.clone().add(new THREE.Vector3(half.x + 0.004, -half.y * 0.2, Math.min(half.z, 0.04) * 0.3));
    const position = contact.sub(this.knuckleContact.clone().applyQuaternion(quaternion));
    return [{ position, quaternion, pose: grip(curl), visible: true }, mirror(idle)];
  }
}

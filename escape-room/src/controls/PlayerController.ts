import * as THREE from 'three';
import { Groups, RAPIER, type PhysicsWorld } from '../physics/PhysicsWorld';
import type { Vec3 } from '../config/types';
import type { Input } from './Input';

const RADIUS = 0.3;
const HALF_HEIGHT = 0.6; // Kapsel gesamt: 2 × (0,6 + 0,3) = 1,8 m
const EYE_HEIGHT = 1.65; // über dem Boden
const WALK_SPEED = 2.6;
const RUN_SPEED = 4.5;
const JUMP_SPEED = 4.2;
const GRAVITY = -16;
const MOUSE_SENSITIVITY = 0.0022;

/**
 * Ego-Perspektive mit Rapiers kinematischem Character Controller: rutscht an Wänden entlang,
 * steigt kleine Stufen hoch, bleibt am Boden und schiebt leichte Gegenstände beiseite.
 */
export class PlayerController {
  readonly body: RAPIER.RigidBody;
  readonly collider: RAPIER.Collider;
  private readonly controller: RAPIER.KinematicCharacterController;

  private yaw: number;
  /** Alkoholpegel (0 = nüchtern), lässt die Laufrichtung pendeln. */
  wobble = 0;
  private drunkTime = 0;
  private pitch = 0;
  private readonly velocity = new THREE.Vector3();
  private grounded = false;
  private readonly prevPosition = new THREE.Vector3();
  private readonly currPosition = new THREE.Vector3();
  private readonly look = new THREE.Quaternion();
  private readonly euler = new THREE.Euler(0, 0, 0, 'YXZ');

  constructor(
    physics: PhysicsWorld,
    private readonly camera: THREE.PerspectiveCamera,
    private readonly input: Input,
    spawn: Vec3,
    yawDegrees: number,
  ) {
    const world = physics.world;
    const center = spawn[1] + HALF_HEIGHT + RADIUS;
    this.body = world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(spawn[0], center, spawn[2]));
    this.collider = world.createCollider(
      RAPIER.ColliderDesc.capsule(HALF_HEIGHT, RADIUS).setCollisionGroups(Groups.player),
      this.body,
    );

    this.controller = world.createCharacterController(0.02);
    this.controller.enableAutostep(0.25, 0.15, false);
    this.controller.enableSnapToGround(0.3);
    this.controller.setMaxSlopeClimbAngle(THREE.MathUtils.degToRad(45));
    this.controller.setApplyImpulsesToDynamicBodies(true);
    this.controller.setCharacterMass(25);

    this.yaw = THREE.MathUtils.degToRad(yawDegrees);
    this.currPosition.set(spawn[0], center, spawn[2]);
    this.prevPosition.copy(this.currPosition);
    this.updateLook();
  }

  /** Mausbewegung in Blickrichtung umsetzen. */
  rotateView(dx: number, dy: number) {
    this.yaw -= dx * MOUSE_SENSITIVITY;
    this.pitch = THREE.MathUtils.clamp(this.pitch - dy * MOUSE_SENSITIVITY, -1.55, 1.55);
    this.updateLook();
  }

  fixedUpdate(dt: number) {
    const input = this.input;
    let forward = 0;
    let strafe = 0;
    if (input.active) {
      forward = Number(input.isDown('KeyW') || input.isDown('ArrowUp')) - Number(input.isDown('KeyS') || input.isDown('ArrowDown'));
      strafe = Number(input.isDown('KeyD') || input.isDown('ArrowRight')) - Number(input.isDown('KeyA') || input.isDown('ArrowLeft'));
      // Virtueller Stick auf Touchgeräten
      if (!forward && !strafe) {
        forward = input.moveAxis.y;
        strafe = input.moveAxis.x;
      }
    }
    const speed = input.isDown('ShiftLeft') || input.isDown('ShiftRight') ? RUN_SPEED : WALK_SPEED;

    // Wunschrichtung in der Ebene, abhängig von der Blickrichtung (yaw)
    const wish = new THREE.Vector3(strafe, 0, -forward);
    if (wish.lengthSq() > 1) wish.normalize();
    wish.applyAxisAngle(THREE.Object3D.DEFAULT_UP, this.yaw).multiplyScalar(speed);
    // Betrunken: Laufrichtung pendelt seitlich
    this.drunkTime += dt;
    if (this.wobble > 0 && wish.lengthSq() > 0) {
      wish.applyAxisAngle(THREE.Object3D.DEFAULT_UP, Math.sin(this.drunkTime * 0.9) * 0.55 * Math.min(this.wobble, 1.4));
    }

    const accel = this.grounded ? 14 : 3;
    this.velocity.x = THREE.MathUtils.damp(this.velocity.x, wish.x, accel, dt);
    this.velocity.z = THREE.MathUtils.damp(this.velocity.z, wish.z, accel, dt);

    if (this.grounded) {
      this.velocity.y = input.active && input.isDown('Space') ? JUMP_SPEED : -1;
    } else {
      this.velocity.y += GRAVITY * dt;
    }

    const desired = this.velocity.clone().multiplyScalar(dt);
    this.controller.computeColliderMovement(this.collider, desired, RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, Groups.player);
    const move = this.controller.computedMovement();
    this.grounded = this.controller.computedGrounded();
    // An die Decke gestoßen: Aufwärtsgeschwindigkeit abbauen
    if (this.velocity.y > 0 && move.y < desired.y * 0.5) this.velocity.y = 0;

    const t = this.body.translation();
    this.body.setNextKinematicTranslation({ x: t.x + move.x, y: t.y + move.y, z: t.z + move.z });
  }

  afterStep() {
    this.prevPosition.copy(this.currPosition);
    const t = this.body.translation();
    this.currPosition.set(t.x, t.y, t.z);
  }

  /** Kamera geglättet zwischen den Physikschritten positionieren. */
  updateCamera(alpha: number) {
    this.camera.position.lerpVectors(this.prevPosition, this.currPosition, alpha);
    this.camera.position.y += EYE_HEIGHT - (HALF_HEIGHT + RADIUS);
    this.camera.quaternion.copy(this.look);
  }

  /** Augenposition im aktuellen Physikzustand (für gehaltene Gegenstände). */
  eyePosition(out: THREE.Vector3) {
    const t = this.body.translation();
    return out.set(t.x, t.y + EYE_HEIGHT - (HALF_HEIGHT + RADIUS), t.z);
  }

  /** Horizontale Laufgeschwindigkeit in m/s. */
  get speed() {
    return Math.hypot(this.velocity.x, this.velocity.z);
  }

  /** Punkt auf dem Boden `distance` Meter vor dem Spieler (nur Blickrichtung in der Ebene). */
  groundPointAhead(distance: number, sideways = 0) {
    const t = this.body.translation();
    const offset = new THREE.Vector3(sideways, 0, -distance).applyAxisAngle(THREE.Object3D.DEFAULT_UP, this.yaw);
    return new THREE.Vector3(t.x + offset.x, 0, t.z + offset.z);
  }

  get lookQuaternion(): Readonly<THREE.Quaternion> {
    return this.look;
  }

  /** Für Tests und spätere Spawnpunkte in weiteren Räumen. */
  teleport(position: Vec3, yawDegrees?: number) {
    const center = position[1] + HALF_HEIGHT + RADIUS;
    this.body.setTranslation({ x: position[0], y: center, z: position[2] }, true);
    this.currPosition.set(position[0], center, position[2]);
    this.prevPosition.copy(this.currPosition);
    this.velocity.set(0, 0, 0);
    if (yawDegrees !== undefined) {
      this.yaw = THREE.MathUtils.degToRad(yawDegrees);
      this.pitch = 0;
      this.updateLook();
    }
  }

  setPitch(degrees: number) {
    this.pitch = THREE.MathUtils.degToRad(degrees);
    this.updateLook();
  }

  private updateLook() {
    this.euler.set(this.pitch, this.yaw, 0);
    this.look.setFromEuler(this.euler);
  }
}

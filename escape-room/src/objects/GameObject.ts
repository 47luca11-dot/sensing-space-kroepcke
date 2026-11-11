import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import type { ObjectConfig } from '../config/types';
import type { Operable } from '../interaction/operables';
import type { LoadedModel } from './models';

/**
 * Ein Gegenstand im Raum: sichtbares Modell + Rapier-Körper. Das Modell folgt immer dem Körper,
 * die Physik ist die einzige Quelle für Position und Drehung.
 */
export class GameObject {
  readonly root = new THREE.Group();
  behavior: Operable | null = null;

  private readonly prevPosition = new THREE.Vector3();
  private readonly currPosition = new THREE.Vector3();
  private readonly prevQuaternion = new THREE.Quaternion();
  private readonly currQuaternion = new THREE.Quaternion();

  constructor(
    readonly config: ObjectConfig,
    readonly model: LoadedModel,
    readonly body: RAPIER.RigidBody,
    readonly colliders: RAPIER.Collider[],
  ) {
    this.root.name = config.id;
    this.root.add(model.object);
    this.captureState();
    this.captureState();
    this.syncVisual(1);
  }

  get label() {
    return this.config.label;
  }

  get isMovable() {
    return this.config.kind === 'movable';
  }

  /** Bewegt sich der Körper überhaupt (dynamisch oder kinematisch)? */
  get isSimulated() {
    return !this.body.isFixed();
  }

  /** Nach jedem Physikschritt: aktuellen Zustand merken, vorherigen für die Interpolation behalten. */
  captureState() {
    this.prevPosition.copy(this.currPosition);
    this.prevQuaternion.copy(this.currQuaternion);
    const t = this.body.translation();
    const r = this.body.rotation();
    this.currPosition.set(t.x, t.y, t.z);
    this.currQuaternion.set(r.x, r.y, r.z, r.w);
  }

  /** Glättet zwischen zwei Physikschritten, damit es auch bei 120/144 Hz flüssig aussieht. */
  syncVisual(alpha: number) {
    this.root.position.lerpVectors(this.prevPosition, this.currPosition, alpha);
    this.root.quaternion.slerpQuaternions(this.prevQuaternion, this.currQuaternion, alpha);
  }

  setCollisionGroups(groups: number) {
    for (const c of this.colliders) c.setCollisionGroups(groups);
  }

  setHighlight(on: boolean) {
    this.model.setHighlight(on);
  }

  /** Grobe Größe, z. B. für den Halteabstand vor der Kamera. */
  get radius() {
    return this.model.bounds.getSize(new THREE.Vector3()).length() / 2;
  }
}

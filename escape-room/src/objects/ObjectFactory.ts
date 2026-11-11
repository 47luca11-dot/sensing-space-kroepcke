import * as THREE from 'three';
import type { ObjectConfig } from '../config/types';
import { buildColliders } from '../physics/colliders';
import { Groups, RAPIER, type PhysicsWorld } from '../physics/PhysicsWorld';
import { GameObject } from './GameObject';
import { loadModel, type LoadedModel } from './models';

/** Alle Spielobjekte, auffindbar über den Collider, den ein Strahl trifft. */
export class ObjectRegistry {
  readonly all: GameObject[] = [];
  private readonly byCollider = new Map<number, GameObject>();

  add(obj: GameObject) {
    this.all.push(obj);
    for (const c of obj.colliders) this.byCollider.set(c.handle, obj);
  }

  remove(obj: GameObject) {
    this.all.splice(this.all.indexOf(obj), 1);
    for (const c of obj.colliders) this.byCollider.delete(c.handle);
  }

  fromCollider(collider: RAPIER.Collider) {
    return this.byCollider.get(collider.handle) ?? null;
  }

  byId(id: string) {
    return this.all.find((o) => o.config.id === id) ?? null;
  }
}

/**
 * Erzeugt aus einem Eintrag in config/objects.ts Modell, Körper und Kollisionskörper.
 * Schlägt das Laden eines Modells fehl, erscheint eine magentafarbene Ersatzbox, damit das Spiel weiterläuft
 * (mit `fallback: false` wird der Fehler stattdessen weitergereicht, z. B. für Uploads).
 */
export async function createObject(config: ObjectConfig, physics: PhysicsWorld, fallback = true): Promise<GameObject> {
  const scale = config.scale ?? 1;
  let model: LoadedModel;
  try {
    model = await loadModel(config.model, scale);
  } catch (error) {
    if (!fallback) throw error;
    console.error(`Modell für „${config.id}“ konnte nicht geladen werden:`, error);
    model = await loadModel({ type: 'placeholder', shape: 'box', size: [0.2, 0.2, 0.2], color: '#ff00ff' }, scale);
  }

  const colliderConfig = config.collider ?? model.defaultCollider ?? { shape: 'auto' };
  const descs = buildColliders(colliderConfig, model, scale);

  const [rx, ry, rz] = (config.rotation ?? [0, 0, 0]).map(THREE.MathUtils.degToRad);
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz));

  // Maßstabsfiguren sind bedienbar (E = ansprechen), aber physikalisch frei: sie wackeln und kippen.
  const isDummy = config.operable?.type === 'dummy';
  const dynamic = config.kind === 'movable' || isDummy;
  let bodyDesc: RAPIER.RigidBodyDesc;
  if (isDummy) {
    bodyDesc = RAPIER.RigidBodyDesc.dynamic().setLinearDamping(0.4).setAngularDamping(1.2);
  } else if (config.kind === 'movable') {
    bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setLinearDamping(0.1)
      .setAngularDamping(0.3)
      // Kleine, schnelle Gegenstände würden sonst durch dünne Bretter tunneln.
      .setCcdEnabled(true);
  } else if (config.operable?.type === 'door') {
    bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased();
  } else {
    bodyDesc = RAPIER.RigidBodyDesc.fixed();
  }
  bodyDesc.setTranslation(...config.position).setRotation({ x: q.x, y: q.y, z: q.z, w: q.w });
  const body = physics.world.createRigidBody(bodyDesc);

  const colliders = descs.map((desc) => {
    desc
      .setFriction(0.7)
      .setRestitution(config.bounciness ?? 0.1)
      .setCollisionGroups(dynamic ? Groups.prop : Groups.world);
    if (dynamic) desc.setMass((config.mass ?? 1) / descs.length);
    return physics.world.createCollider(desc, body);
  });
  // Figuren: Gewicht in der Standplatte, damit sie nicht schon vom Anstupsen umfallen
  if (isDummy && colliders.length > 1) {
    const total = config.mass ?? 40;
    colliders[0].setMass(total * 0.65);
    colliders.slice(1).forEach((c) => c.setMass((total * 0.35) / (colliders.length - 1)));
  }

  return new GameObject(config, model, body, colliders);
}

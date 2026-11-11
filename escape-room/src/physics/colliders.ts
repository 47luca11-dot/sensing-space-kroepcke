import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import type { ColliderConfig, Vec3 } from '../config/types';
import type { LoadedModel } from '../objects/models';

const ZERO: Vec3 = [0, 0, 0];
const scaled = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];

/**
 * Baut die Rapier-Kollisionskörper für ein Objekt. Komplexe Scans bekommen bewusst nur einfache Formen
 * (Box, Zylinder, Kugel oder konvexe Hülle), nie das Detailmodell selbst.
 */
export function buildColliders(config: ColliderConfig, model: LoadedModel, scale: number): RAPIER.ColliderDesc[] {
  if (config.shape === 'compound') return config.parts.flatMap((part) => buildColliders(part, model, scale));
  if (config.shape === 'auto') return autoColliders(config.fit ?? 'box', model);

  const offset = scaled(config.offset ?? ZERO, scale);
  let desc: RAPIER.ColliderDesc;
  switch (config.shape) {
    case 'box':
      desc = RAPIER.ColliderDesc.cuboid(...scaled(config.halfExtents, scale));
      break;
    case 'sphere':
      desc = RAPIER.ColliderDesc.ball(config.radius * scale);
      break;
    case 'cylinder':
      desc = RAPIER.ColliderDesc.cylinder(config.halfHeight * scale, config.radius * scale);
      break;
    case 'capsule':
      desc = RAPIER.ColliderDesc.capsule(config.halfHeight * scale, config.radius * scale);
      break;
  }
  return [desc.setTranslation(...offset)];
}

/** Passt eine Grundform an die Modellgrenzen an. Die Grenzen sind bereits skaliert. */
function autoColliders(fit: 'box' | 'cylinder' | 'sphere' | 'hull', model: LoadedModel): RAPIER.ColliderDesc[] {
  const c = model.bounds.getCenter(new THREE.Vector3());
  // Sehr dünne Kanten tunneln leicht durch Tischplatten, daher Mindestmaß.
  const half = model.bounds
    .getSize(new THREE.Vector3())
    .multiplyScalar(0.5)
    .max(new THREE.Vector3(0.01, 0.01, 0.01));
  const box = () => RAPIER.ColliderDesc.cuboid(half.x, half.y, half.z).setTranslation(c.x, c.y, c.z);
  switch (fit) {
    case 'box':
      return [box()];
    case 'cylinder':
      return [RAPIER.ColliderDesc.cylinder(half.y, Math.max(half.x, half.z)).setTranslation(c.x, c.y, c.z)];
    case 'sphere':
      return [RAPIER.ColliderDesc.ball(Math.max(half.x, half.y, half.z)).setTranslation(c.x, c.y, c.z)];
    case 'hull':
      return [(model.hullPoints && RAPIER.ColliderDesc.convexHull(model.hullPoints)) || box()];
  }
}

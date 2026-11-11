import { ROOM } from './room';
import type { Vec3 } from './types';

/** Roter Raum südlich des Büros. Die Nordseite ist die Südwand des Büros mit dem Durchgang. */
const minZ = ROOM.depth / 2 + ROOM.wallThickness;
const depth = 5;

export const RED_ROOM = {
  minX: -2.5,
  maxX: 3.5,
  minZ,
  maxZ: minZ + depth,
  height: ROOM.height,
  /** Mitte des Raums, dort stehen Bühne und Stange. */
  center: [0.5, 0, minZ + depth / 2] as Vec3,
  wallColor: '#6e0d16',
  floorColor: '#0c0b0d',
  lights: [
    { position: [0.5, 2.5, minZ + depth / 2] as Vec3, color: '#ff2a3d', intensity: 7, distance: 7 },
    { position: [-1.6, 2.4, minZ + 1.2] as Vec3, color: '#ff1844', intensity: 4, distance: 5 },
    { position: [2.6, 2.4, minZ + depth - 1] as Vec3, color: '#ff3b2a', intensity: 4, distance: 5 },
  ],
  /** Spot von oben auf die Stange */
  spot: { color: '#ffd6e0', intensity: 9 },
};

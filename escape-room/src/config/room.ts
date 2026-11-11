import type { Vec3 } from './types';

export interface LightConfig {
  id: string;
  position: Vec3;
  color: string;
  intensity: number;
  distance: number;
  castShadow?: boolean;
  /** Sichtbare Lampe an der Decke, leuchtet nur, solange das Licht an ist. */
  fixture?: boolean;
}

/** Testraum. Ursprung = Raummitte auf dem Boden, −Z ist die Nordwand mit der Ausgangstür. */
export const ROOM = {
  width: 7,
  depth: 6,
  height: 2.8,
  wallThickness: 0.2,
  /** Türöffnung in der Nordwand, `x` = Mitte der Öffnung. */
  doorway: { x: 1.4, width: 1.0, height: 2.1 },
  /** Offener Durchgang in der Südwand zum roten Raum. */
  southDoorway: { x: 0.6, width: 1.1, height: 2.2 },
  spawn: { position: [0, 0, 2.2] as Vec3, yawDegrees: 0 },
  /**
   * Wandtextur, kachelbar, aus dem Scan „Bothfeld-Vahrenheide“ erzeugt (tools/make-grunge-wall.mjs).
   * `tileMeters` = reale Größe einer Kachel, damit die Schalungsbretter maßstabsgetreu bleiben.
   */
  wallTexture: {
    color: '/textures/concrete-color.png',
    normal: '/textures/concrete-normal.png',
    roughness: '/textures/concrete-roughness.png',
    tileMeters: [2.661, 1.326] as [number, number],
  },
  /** Grundhelligkeit, wenn alle Lampen aus sind. */
  ambient: 0.35,
  lights: [
    { id: 'ceiling', position: [0, 2.6, 0], color: '#ffe8c8', intensity: 14, distance: 12, castShadow: true, fixture: true },
    { id: 'exit', position: [1.4, 2.4, -2.7], color: '#7dffa0', intensity: 1.2, distance: 3 },
  ] satisfies LightConfig[],
};

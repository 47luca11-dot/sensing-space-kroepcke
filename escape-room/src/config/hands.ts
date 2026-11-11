import type { Vec3 } from './types';

export type FingerName = 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';

/**
 * Handmodell für die Ego-Perspektive. Ein anderes (z. B. gescanntes) Modell lässt sich einsetzen,
 * indem man `url` und die Knochennamen anpasst; die Greiflogik bleibt gleich.
 */
export const HANDS = {
  /** Rechte Hand, rigged. Die linke wird daraus gespiegelt. */
  url: '/models/hands/hand-right.glb',
  wristBone: 'hand_r',
  /** Fingerknochen von der Wurzel zur Spitze. */
  fingers: {
    thumb: ['thumb_01_r', 'thumb_02_r', 'thumb_03_r'],
    index: ['index_01_r', 'index_02_r', 'index_03_r'],
    middle: ['middle_01_r', 'middle_02_r', 'middle_03_r'],
    ring: ['ring_01_r', 'ring_02_r', 'ring_03_r'],
    pinky: ['pinky_01_r', 'pinky_02_r', 'pinky_03_r'],
  } satisfies Record<FingerName, string[]>,
  /** Lokale Achse, um die sich ein Fingerglied zur Handfläche beugt, und Drehsinn (+1/−1). */
  flexAxis: [0, 0, 1] as Vec3,
  flexSign: -1,
  /** Das Daumengrundgelenk dreht anders: um diese Achse schwenkt der Daumen vor die Finger (Opposition). */
  thumbOppositionAxis: [1, 0, 0] as Vec3,
  /** Maximale Beugung je Glied (Grad) bei Krümmung 1; beim Daumen ist der erste Wert die Opposition. */
  maxFlex: { finger: [75, 95, 70] as Vec3, thumb: [45, 30, 45] as Vec3 },
  skinColor: '#d9a184',
  /** Hände ohne Gegenstand am unteren Bildrand zeigen. */
  showIdle: true,
};

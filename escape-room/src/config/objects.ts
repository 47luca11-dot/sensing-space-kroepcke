import { FEATURES } from './features';
import { PUZZLE } from './puzzle';
import { RED_ROOM } from './redRoom';
import { ROOM } from './room';
import type { ObjectConfig } from './types';

/**
 * Alle Gegenstände des Testraums (Architekturbüro) an einer Stelle.
 *
 * Ein Platzhalter wird zum Scan, indem man nur `model` tauscht, z. B.
 *   model: { type: 'splat', url: '/models/schluessel.spz' }
 * Interaktionstyp, Position, Drehung (Grad), Maßstab, Masse und Kollisionskörper bleiben unabhängig davon.
 */

const tableTop = 0.76;
const northWallZ = -ROOM.depth / 2 - ROOM.wallThickness / 2;
const doorWidth = ROOM.doorway.width - 0.04;

// Tresor an der Nordwand, Öffnung nach Süden (+Z)
const safe = { x: -2.3, z: -ROOM.depth / 2 + 0.26, w: 0.6, h: 0.6, d: 0.5, wall: 0.04 };
// Regal an der Ostwand: Bretter auf 0,40 / 0,80 / 1,20 m
const shelfX = ROOM.width / 2 - 0.2;

const FIGURE_COLOR = '#ecebe6';
const RED_ROOM_OBJECTS = redRoomObjects();

export const OBJECTS: ObjectConfig[] = [
  // ── Feste Raumelemente ─────────────────────────────────────────────
  {
    id: 'table',
    label: 'Zeichentisch',
    kind: 'static',
    model: { type: 'placeholder', shape: 'table', size: [1.6, tableTop, 0.9], color: '#c7a77a' },
    position: [-0.4, 0, -0.4],
  },
  {
    id: 'shelf',
    label: 'Modellregal',
    kind: 'static',
    model: { type: 'placeholder', shape: 'shelf', size: [1.1, 1.6, 0.4], color: '#4c4c4e' },
    position: [shelfX, 0, 1.2],
    rotation: [0, -90, 0],
  },
  {
    id: 'safe',
    label: 'Tresor',
    kind: 'static',
    model: { type: 'placeholder', shape: 'safe', size: [safe.w, safe.h, safe.d], color: '#3d4148' },
    position: [safe.x, 0, safe.z],
  },

  // ── Bewegliche Gegenstände ─────────────────────────────────────────
  {
    id: 'bottle',
    label: 'Lila Flasche',
    kind: 'movable',
    // Echter Scaniverse-Scan (Gaussian Splat). Kollision: automatisch angepasster Zylinder.
    model: { type: 'splat', url: '/models/flasche-lila.ply' },
    collider: { shape: 'auto', fit: 'cylinder' },
    // Knapp über der Tischplatte (Flasche ist ca. 22 cm hoch), damit sie beim Start nicht umkippt.
    position: [-0.7, tableTop + 0.115, -0.5],
    mass: 0.6,
    consumable: { kind: 'drink' },
  },
  {
    id: 'massing-model',
    label: 'Massenmodell',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'massing', color: '#f4f2ec' },
    position: [-0.05, tableTop + 0.08, -0.35],
    rotation: [0, 20, 0],
    mass: 1.2,
  },
  {
    id: 'ruler',
    label: 'Dreikantmaßstab',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'ruler', color: '#e8e4d8' },
    position: [0.15, tableTop + 0.03, 0.0],
    rotation: [0, 35, 0],
    mass: 0.1,
  },
  {
    id: 'styrofoam-ball',
    label: 'Styroporkugel',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'sphere', size: [0.22, 0.22, 0.22], color: '#f6f5f0' },
    position: [-1.5, 0.3, 1.0],
    mass: 0.05,
    bounciness: 0.5,
  },
  {
    id: 'book',
    label: 'Fachbuch',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'book', size: [0.21, 0.05, 0.3], color: '#1f1f22' },
    position: [0.25, tableTop + 0.05, -0.65],
    rotation: [0, -15, 0],
    mass: 1.2,
  },
  {
    id: 'mug',
    label: 'Kaffeetasse',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'mug', size: [0.085, 0.1, 0.085], color: '#e8e1d4' },
    position: [-1.0, tableTop + 0.08, -0.1],
    mass: 0.3,
    consumable: { kind: 'drink' },
  },
  {
    id: 'apple',
    label: 'Apfel',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'apple', color: '#b8322a' },
    position: [-0.95, tableTop + 0.05, -0.65],
    mass: 0.15,
    consumable: { kind: 'food', bites: 4 },
  },
  {
    id: 'croissant',
    label: 'Croissant',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'croissant', color: '#c98a3d' },
    position: [-1.05, tableTop + 0.04, -0.3],
    rotation: [0, 40, 0],
    mass: 0.06,
    consumable: { kind: 'food', bites: 3 },
  },
  {
    id: 'knife',
    label: 'Mordmesser',
    kind: 'movable',
    weapon: 'knife',
    model: { type: 'placeholder', shape: 'knife', color: '#2b2522' },
    // Mittleres Regalbrett
    position: [shelfX, 0.84, 0.95],
    rotation: [0, 70, 0],
    mass: 0.2,
  },
  {
    id: 'crate-large',
    label: 'Materialkiste',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'box', size: [0.55, 0.45, 0.55], color: '#9c6b3f' },
    position: [-2.8, 0.25, 1.9],
    rotation: [0, 10, 0],
    mass: 9,
  },
  {
    id: 'key',
    label: 'Schlüssel',
    kind: 'movable',
    itemId: 'exit-key',
    model: { type: 'placeholder', shape: 'key', color: '#d9b44a' },
    // Liegt im Tresor
    position: [safe.x, safe.wall + 0.015, safe.z],
    rotation: [0, 20, 0],
    mass: 0.1,
  },

  // ── Maßstabsfiguren (sprechen, wackeln, kippen um) ─────────────────
  {
    id: 'figure-client',
    label: 'Bauherr',
    kind: 'operable',
    model: { type: 'placeholder', shape: 'dummy', color: FIGURE_COLOR },
    position: [-2.6, 0.02, 0.2],
    rotation: [0, 90, 0],
    mass: 45,
    operable: {
      type: 'dummy',
      lines: [
        'Ich hätte das gern bis Montag.',
        'Geht das nicht auch etwas günstiger?',
        'Wo genau ist hier eigentlich der Ausgang?',
        'Machen Sie doch einfach mal das Licht aus und schauen genauer hin.',
      ],
    },
  },
  {
    id: 'figure-engineer',
    label: 'Statikerin',
    kind: 'operable',
    model: { type: 'placeholder', shape: 'dummy', color: FIGURE_COLOR },
    position: [2.3, 0.02, 2.4],
    rotation: [0, -135, 0],
    mass: 45,
    operable: {
      type: 'dummy',
      lines: [
        'Das hält. Wahrscheinlich.',
        'Ohne Code kein Tresor, ohne Tresor kein Schlüssel.',
        'Beton ist ehrlich. Er zeigt jede Fuge.',
        'Der Tresor steht genau in der Achse. Sehr vernünftig.',
      ],
    },
  },
  {
    id: 'figure-intern',
    label: 'Praktikant',
    kind: 'operable',
    model: { type: 'placeholder', shape: 'dummy', color: FIGURE_COLOR },
    position: [-1.3, 0.02, 1.7],
    rotation: [0, 160, 0],
    mass: 40,
    operable: {
      type: 'dummy',
      toughness: 4,
      lines: [
        'Ich stehe hier nur für den Maßstab.',
        'Die Zahlen an der Wand sieht man nur im Dunkeln, habe ich gehört.',
        'Soll ich noch ein Modell bauen?',
        'Bitte nicht mit dem Messer …',
      ],
    },
  },

  // ── Bedienbare Objekte ─────────────────────────────────────────────
  {
    id: 'exit-door',
    label: 'Ausgangstür',
    kind: 'operable',
    model: { type: 'placeholder', shape: 'door', size: [doorWidth, ROOM.doorway.height - 0.02, 0.05], color: '#5b6b78' },
    // Ursprung der Tür = Scharnier an der linken Kante der Öffnung
    position: [ROOM.doorway.x - doorWidth / 2, 0, northWallZ],
    operable: { type: 'door', requires: 'exit-key', openAngle: 100, exit: true },
  },
  {
    id: 'light-switch',
    label: 'Lichtschalter',
    kind: 'operable',
    model: { type: 'placeholder', shape: 'switch', color: '#f2efe8' },
    position: [ROOM.doorway.x + 0.85, 1.25, -ROOM.depth / 2],
    operable: { type: 'switch', target: 'ceiling' },
  },
  {
    id: 'safe-door',
    label: 'Tresor',
    kind: 'operable',
    model: { type: 'placeholder', shape: 'door', size: [safe.w - 0.01, safe.h - 0.01, 0.04], color: '#4a4f57' },
    // Scharnier links vorn am Korpus, öffnet nach vorn (negativer Winkel)
    position: [safe.x - safe.w / 2 + 0.005, 0.005, safe.z + safe.d / 2 + 0.021],
    operable: { type: 'door', code: PUZZLE.safeCode, openAngle: -105 },
  },
  {
    id: 'note',
    label: 'Skizze',
    kind: 'operable',
    model: { type: 'placeholder', shape: 'note', size: [0.3, 0.002, 0.21], color: '#f3ecd9' },
    position: [-0.25, tableTop, -0.05],
    rotation: [0, 12, 0],
    operable: { type: 'note', text: PUZZLE.noteText },
  },
  ...(FEATURES.redRoom ? RED_ROOM_OBJECTS : []),
];

// ── Roter Raum ─────────────────────────────────────────────────────────
function redRoomObjects(): ObjectConfig[] {
  const [cx, , cz] = RED_ROOM.center;
  const barX = RED_ROOM.maxX - 0.4;
  const barTop = 1.05;
  const beers: ObjectConfig[] = [0, 1, 2, 3, 4, 5].map((i) => ({
    id: `beer-${i + 1}`,
    label: 'Bier',
    kind: 'movable',
    model: { type: 'placeholder', shape: 'beer', color: '#5a2e0e' },
    position: [barX - 0.08 + (i % 2) * 0.1, barTop + 0.075, cz - 0.6 + i * 0.22],
    mass: 0.5,
    consumable: { kind: 'drink', sips: 5, alcohol: 0.07 },
  }));
  return [
    {
      id: 'stage',
      label: 'Bühne',
      kind: 'static',
      model: { type: 'placeholder', shape: 'stage', size: [1.8, 0.15, 1.8], color: '#1b1214' },
      position: [cx, 0, cz],
    },
    {
      id: 'pole',
      label: 'Stange',
      kind: 'static',
      model: { type: 'placeholder', shape: 'pole', size: [0.05, RED_ROOM.height, 0.05], color: '#e6e8ec' },
      position: [cx, 0, cz],
    },
    {
      id: 'bar',
      label: 'Bar',
      kind: 'static',
      model: { type: 'placeholder', shape: 'bar', size: [0.6, barTop, 2.4], color: '#2a1518' },
      position: [barX, 0, cz],
    },
    {
      id: 'sofa-1',
      label: 'Sofa',
      kind: 'static',
      model: { type: 'placeholder', shape: 'sofa', size: [1.8, 0.85, 0.85], color: '#8e1020' },
      position: [RED_ROOM.minX + 0.45, 0, cz - 1.1],
      rotation: [0, 90, 0],
    },
    {
      id: 'sofa-2',
      label: 'Sofa',
      kind: 'static',
      model: { type: 'placeholder', shape: 'sofa', size: [1.8, 0.85, 0.85], color: '#8e1020' },
      position: [cx - 0.3, 0, RED_ROOM.maxZ - 0.45],
      rotation: [0, 180, 0],
    },
    ...beers,
    {
      id: 'figure-guest',
      label: 'Stammgast',
      kind: 'operable',
      model: { type: 'placeholder', shape: 'dummy', color: '#e9e6df' },
      position: [barX - 0.75, 0.02, cz + 1.0],
      rotation: [0, -100, 0],
      mass: 45,
      operable: {
        type: 'dummy',
        lines: [
          'Das Bier hier ist gut. Das fünfte noch besser.',
          'Wirf ruhig, das Geld sammelt sich schon wieder ein.',
          'Rot ist die einzig ehrliche Wandfarbe.',
          'Wer hat eigentlich die Stange statisch nachgewiesen?',
        ],
      },
    },
  ];
}

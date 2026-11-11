import type { Vec3 } from './types';

/** Rätsel im Testraum: Code im Dunkeln finden → Tresor öffnen → Schlüssel → Ausgangstür. */
export const PUZZLE = {
  safeCode: '4721',
  /** Nachleuchtende Ziffern, nur bei ausgeschaltetem Deckenlicht sichtbar. */
  glowCode: {
    position: [3.49, 2.05, 1.2] as Vec3,
    /** Drehung um Y in Grad; −90 = Schrift zeigt von der Ostwand in den Raum. */
    rotationY: -90,
    width: 0.9,
    color: '#b8ff9e',
    /** Lampe, deren Ausschalten den Code sichtbar macht. */
    light: 'ceiling',
  },
  noteText: 'Gute Architektur zeigt sich auch im Dunkeln.',
};

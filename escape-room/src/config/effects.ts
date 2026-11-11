/** Enthüllungseffekt für Gaussian-Splat-Objekte (siehe src/effects/UnveilEffect.ts). */
export const UNVEIL = {
  enabled: true,
  /** Leuchtende Regenbogenfront an der Grenze zwischen verdeckt und sichtbar. */
  rainbow: true,
  /** Reichweite in Metern ab dem Startpunkt; 0 = automatisch bis zum entferntesten Splat. */
  range: 0,
  /** Breite des weichen Übergangs (und der Regenbogenfront) in Metern. */
  blendRange: 0.03,
  /** Dauer in Sekunden. */
  duration: 3,
  /** Größe der Vorschau-Punkte im noch verdeckten Bereich in Metern. */
  pointSize: 0.0025,
  /** Anteil der Splats, die vorab als Punktwolke sichtbar sind (0–1). */
  pointDensity: 0.35,
  /** Startpunkt im Objekt: unten mittig oder Mitte. */
  origin: 'bottom' as 'bottom' | 'center',
};

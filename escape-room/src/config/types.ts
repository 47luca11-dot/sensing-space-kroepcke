export type Vec3 = [number, number, number];

/** Wie ein Objekt im Spiel behandelt wird. */
export type ObjectKind =
  /** Festes Raumelement: kollidiert, bewegt sich nie (Tisch, Regal). */
  | 'static'
  /** Beweglicher Gegenstand: Schwerkraft, Kollisionen, lässt sich aufnehmen. */
  | 'movable'
  /** Bedienbares Objekt: reagiert auf E (Tür, Schalter). Verhalten steht in `operable`. */
  | 'operable';

export type PlaceholderShape =
  | 'box'
  | 'sphere'
  | 'cylinder'
  | 'book'
  | 'mug'
  | 'key'
  | 'table'
  | 'shelf'
  | 'door'
  | 'switch'
  | 'poop'
  | 'safe'
  | 'note'
  | 'apple'
  | 'croissant'
  | 'knife'
  | 'dummy'
  | 'massing'
  | 'ruler'
  | 'pole'
  | 'stage'
  | 'bar'
  | 'sofa'
  | 'beer';

/** Woher das sichtbare Modell kommt. Austauschen, ohne die Interaktionslogik anzufassen. */
export type ModelSource =
  /** Einfaches Ersatzobjekt aus Three.js-Grundformen. `size` in Metern (Bedeutung je Form). */
  | { type: 'placeholder'; shape: PlaceholderShape; size?: Vec3; color?: string }
  /**
   * Gaussian Splat (.spz, .ply, .splat, .ksplat) über Spark.
   * flip: dreht um 180° um X (OpenCV-Konvention, Y nach unten). Standard: true, wie bei Scaniverse.
   * minOpacity: Splats darunter zählen nicht für Mittelpunkt und automatischen Kollisionskörper.
   */
  | { type: 'splat'; url: string; flip?: boolean; minOpacity?: number; file?: File }
  /** Klassisches Mesh (.glb/.gltf). */
  | { type: 'gltf'; url: string; file?: File };
// `file` (bei Splat und glTF) setzt das Upload-Feld im Startbildschirm; `url` ist dann nur der Dateiname.

/**
 * Vereinfachter Kollisionskörper. Maße in Objekteinheiten, sie werden mit `scale` multipliziert.
 * Splat- und glTF-Modelle sind auf die Mitte ihrer Bounding Box zentriert, Offsets beziehen sich darauf.
 */
export type ColliderConfig =
  | { shape: 'box'; halfExtents: Vec3; offset?: Vec3 }
  | { shape: 'sphere'; radius: number; offset?: Vec3 }
  | { shape: 'cylinder'; radius: number; halfHeight: number; offset?: Vec3 }
  | { shape: 'capsule'; radius: number; halfHeight: number; offset?: Vec3 }
  | { shape: 'compound'; parts: ColliderConfig[] }
  /** Aus den Modellgrenzen berechnet. 'hull' = konvexe Hülle um die Punkte des Scans. */
  | { shape: 'auto'; fit?: 'box' | 'cylinder' | 'sphere' | 'hull' };

export type OperableConfig =
  /**
   * Drehtür oder Klappe, Ursprung des Modells = Scharnier.
   * `requires` = itemId des passenden Schlüssels, `code` = Zahlenschloss, `exit` = Öffnen beendet den Raum.
   * `openAngle` in Grad, negativ = andere Richtung.
   */
  | { type: 'door'; requires?: string; code?: string; openAngle?: number; exit?: boolean }
  /** Kippschalter, schaltet die Lampe mit der id `target` (siehe config/room.ts). */
  | { type: 'switch'; target: string }
  /** Lesbarer Zettel. */
  | { type: 'note'; text: string }
  /** Lebensgroße Maßstabsfigur: spricht (E), reagiert auf Schläge und Stiche, kippt um. */
  | { type: 'dummy'; lines: string[]; toughness?: number };

export interface ObjectConfig {
  id: string;
  /** Name im Hinweistext („[E] Aufnehmen · …“). */
  label: string;
  kind: ObjectKind;
  model: ModelSource;
  /** Weltposition in Metern. Y = 0 ist der Boden. */
  position: Vec3;
  /** Drehung in Grad (X, Y, Z). */
  rotation?: Vec3;
  /** Gleichmäßiger Maßstab für Modell und Kollisionskörper. */
  scale?: number;
  /** Fehlt er, wird der Standard des Platzhalters bzw. { shape: 'auto' } benutzt. */
  collider?: ColliderConfig;
  /** Masse in kg (nur 'movable'). */
  mass?: number;
  /** Elastizität 0–1 (nur 'movable'). */
  bounciness?: number;
  /** Kennung für Rätsel, z. B. 'exit-key'. Türen prüfen sie über `operable.requires`. */
  itemId?: string;
  operable?: OperableConfig;
  /** Essbar oder trinkbar (Taste F, solange man es hält). `bites` = Bissen bis aufgegessen; Getränke werden nie leer. */
  consumable?: {
    kind: 'food' | 'drink';
    bites?: number;
    /** Getränk mit begrenzten Schlucken (danach leer). Ohne Angabe wird es nie leer. */
    sips?: number;
    /** Alkohol pro Schluck (0–1), lässt die Sicht schwanken. */
    alcohol?: number;
  };
  /** Waffe: Linksklick sticht statt zu werfen. */
  weapon?: 'knife';
  /** Griff erzwingen statt ihn aus der Größe abzuleiten. */
  grip?: 'pinch' | 'grip' | 'two-hands';
}

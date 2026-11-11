import RAPIER from '@dimforge/rapier3d-compat';

/** Kollisionsgruppen (Bitmasken). Wer mit wem kollidiert, steht in `Groups` darunter. */
export const Layer = {
  WORLD: 0x1,
  PLAYER: 0x2,
  PROP: 0x4,
  HELD: 0x8,
} as const;

/** Rapier-InteractionGroups: obere 16 Bit = Mitgliedschaft, untere 16 Bit = kollidiert mit. */
const groups = (membership: number, filter: number) => (membership << 16) | filter;

export const Groups = {
  world: groups(Layer.WORLD, Layer.PLAYER | Layer.PROP | Layer.HELD),
  player: groups(Layer.PLAYER, Layer.WORLD | Layer.PROP),
  prop: groups(Layer.PROP, Layer.WORLD | Layer.PLAYER | Layer.PROP | Layer.HELD),
  /** Gehaltener Gegenstand: stößt an Wände und andere Gegenstände, aber nicht an den Spieler. */
  held: groups(Layer.HELD, Layer.WORLD | Layer.PROP),
};

export const FIXED_DT = 1 / 60;

/**
 * Rapier-Welt mit festem Zeitschritt. `step` ruft `beforeStep` vor jedem Teilschritt auf
 * (Spieler, gehaltener Gegenstand, Türen) und `afterStep` danach (Transformationen übernehmen).
 */
export class PhysicsWorld {
  readonly world: RAPIER.World;
  private accumulator = 0;

  private constructor() {
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    this.world.timestep = FIXED_DT;
  }

  static async create(): Promise<PhysicsWorld> {
    await RAPIER.init();
    return new PhysicsWorld();
  }

  /** Gibt den Interpolationsfaktor (0–1) zwischen letztem und aktuellem Physikzustand zurück. */
  step(deltaSeconds: number, beforeStep: (dt: number) => void, afterStep: () => void): number {
    // Begrenzen, damit ein Tab-Wechsel nicht hunderte Nachhol-Schritte auslöst.
    this.accumulator += Math.min(deltaSeconds, 0.1);
    while (this.accumulator >= FIXED_DT) {
      beforeStep(FIXED_DT);
      this.world.step();
      afterStep();
      this.accumulator -= FIXED_DT;
    }
    return this.accumulator / FIXED_DT;
  }
}

export { RAPIER };

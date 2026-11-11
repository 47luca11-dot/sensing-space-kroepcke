import * as THREE from 'three';
import type { PlayerController } from '../controls/PlayerController';
import type { Hud } from '../ui/Hud';

/** Abbau pro Sekunde (ein Bier ≈ 0,35 ist nach gut einer halben Minute weg). */
const DECAY = 0.012;
const STAGES = [
  { from: 0.25, text: 'Leicht angeheitert.' },
  { from: 0.6, text: 'Betrunken.' },
  { from: 1.0, text: 'Sturzbetrunken.' },
];
/** Ab hier wird automatisch erbrochen (wenn Körperfunktionen an sind). */
const PUKE_LEVEL = 1.3;

/**
 * Betrunken werden: Pegel steigt mit jedem Schluck Alkohol und sinkt langsam.
 * Wirkung: schwankende Sicht, pulsierendes Sichtfeld, unscharfes Bild, schiefer Gang,
 * ab sehr hohem Pegel übergibt man sich.
 */
export class Intoxication {
  private level = 0;
  private time = 0;
  private stage = 0;
  private baseFov: number;
  /** Wird aufgerufen, wenn der Pegel zum Erbrechen reicht. */
  onOverdose: () => void = () => {};

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    private readonly player: PlayerController,
    private readonly hud: Hud,
    private readonly canvas: HTMLCanvasElement,
  ) {
    this.baseFov = camera.fov;
  }

  add(amount: number) {
    this.level = Math.min(1.6, this.level + amount);
  }

  get promille() {
    return this.level * 2;
  }

  /** Nach `player.updateCamera`, vor dem Rendern. */
  update(dt: number) {
    this.time += dt;
    this.level = Math.max(0, this.level - DECAY * dt);
    const s = this.level;
    const t = this.time;

    // Stufen melden
    const stage = STAGES.filter((st) => s >= st.from).length;
    if (stage > this.stage) this.hud.toast(STAGES[stage - 1].text);
    this.stage = stage;
    this.hud.setStatus(s > 0.03 ? `${this.promille.toFixed(1).replace('.', ',')} ‰` : null);

    if (s >= PUKE_LEVEL) {
      this.level -= 0.35;
      this.onOverdose();
    }

    // Gang wird schief
    this.player.wobble = s;

    if (s <= 0.001) {
      if (this.camera.fov !== this.baseFov) {
        this.camera.fov = this.baseFov;
        this.camera.updateProjectionMatrix();
      }
      if (this.canvas.style.filter) this.canvas.style.filter = '';
      return;
    }

    // Schwankende Sicht: langsame Überlagerung mehrerer Schwingungen
    this.camera.rotateZ((Math.sin(t * 0.9) * 0.05 + Math.sin(t * 2.3) * 0.012) * s);
    this.camera.rotateX(Math.sin(t * 0.7 + 1) * 0.03 * s);
    this.camera.rotateY(Math.sin(t * 0.45) * 0.04 * s);
    this.camera.translateX(Math.sin(t * 0.8) * 0.04 * s);

    // Sichtfeld pulsiert
    const fov = this.baseFov + Math.sin(t * 0.6) * 5 * s;
    if (Math.abs(fov - this.camera.fov) > 0.05) {
      this.camera.fov = fov;
      this.camera.updateProjectionMatrix();
    }

    // Unscharf und etwas übersättigt ab „betrunken“
    const blur = Math.max(0, s - 0.45) * 2.4 * (0.8 + 0.2 * Math.sin(t * 1.3));
    this.canvas.style.filter = blur > 0.05 ? `blur(${blur.toFixed(2)}px) saturate(${(1 + s * 0.3).toFixed(2)})` : '';
  }
}

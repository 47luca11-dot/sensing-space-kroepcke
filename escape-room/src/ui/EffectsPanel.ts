import GUI from 'lil-gui';
import type { UnveilEffects } from '../effects/UnveilEffect';

/**
 * Einstellungsfeld für den Enthüllungseffekt (oben rechts, nur im Start-/Pausenbildschirm sichtbar).
 * Werte gelten sofort; „Abspielen“ startet den Effekt auf allen Splat-Objekten neu.
 */
export class EffectsPanel {
  private readonly gui: GUI;

  constructor(effects: UnveilEffects) {
    const s = effects.settings;
    const state = {
      effect: s.rainbow ? 'Mit Regenbogen' : 'Ohne Regenbogen',
      progress: 1,
      play: () => {
        state.progress = 0;
        effects.playAll();
        this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
      },
    };
    this.gui = new GUI({ title: 'Enthüllungseffekt' });
    this.gui.domElement.style.zIndex = '20';
    const apply = () => effects.applySettings();
    this.gui
      .add(state, 'effect', ['Mit Regenbogen', 'Ohne Regenbogen'])
      .name('Effekt')
      .onChange((v: string) => {
        s.rainbow = v === 'Mit Regenbogen';
        apply();
      });
    this.gui.add(s, 'range', 0, 5, 0.01).name('Reichweite (m, 0 = auto)').onChange(apply);
    this.gui.add(s, 'blendRange', 0.005, 0.3, 0.005).name('Übergang (m)').onChange(apply);
    this.gui.add(s, 'duration', 0.5, 10, 0.1).name('Dauer (s)');
    this.gui.add(s, 'pointDensity', 0, 1, 0.01).name('Punktdichte').onChange(apply);
    this.gui
      .add(state, 'progress', 0, 1, 0.01)
      .name('Fortschritt')
      .onChange((v: number) => effects.setProgress(v));
    this.gui.add(state, 'play').name('Abspielen');
  }

  setVisible(visible: boolean) {
    this.gui.show(visible);
  }
}

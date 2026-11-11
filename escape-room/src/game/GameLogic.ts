import type { Input } from '../controls/Input';
import type { Hud } from '../ui/Hud';
import type { EventBus } from './events';

/**
 * Spielablauf des Testraums: Schlüssel finden, Tür aufschließen, raus.
 * Weitere Räume und Rätsel hängen sich hier an die Ereignisse.
 */
export class GameLogic {
  private startTime: number | null = null;
  private finished = false;

  constructor(events: EventBus, private readonly hud: Hud, private readonly input: Input) {
    events.on('message', (text) => hud.toast(text));
    events.on('item-picked-up', (obj) => {
      if (obj.config.itemId === 'exit-key') hud.toast('Schlüssel gefunden.');
    });
    events.on('light-toggled', ({ on }) => hud.toast(on ? 'Licht an' : 'Licht aus'));
    events.on('door-opened', (door) => {
      if (door.config.operable?.type === 'door' && door.config.operable.exit) this.win();
    });
  }

  /** Erster Klick ins Spiel startet die Uhr. */
  start() {
    this.startTime ??= performance.now();
  }

  private win() {
    if (this.finished) return;
    this.finished = true;
    const seconds = (performance.now() - (this.startTime ?? performance.now())) / 1000;
    this.input.exitLock();
    this.hud.showSuccess(seconds);
  }
}

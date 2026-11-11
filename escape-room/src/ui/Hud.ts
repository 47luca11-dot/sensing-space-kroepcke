const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** Fadenkreuz, Hinweise, Meldungen und Overlays (Start, Pause, Laden, Erfolg). */
export class Hud {
  private readonly prompt = $('prompt');
  private readonly hint = $('hint');
  private readonly toastEl = $('toast');
  private readonly crosshair = $('crosshair');
  private readonly startOverlay = $('start-overlay');
  private readonly startTitle = $('start-title');
  private readonly startButton = $<HTMLButtonElement>('start-button');
  private readonly loading = $('loading');
  private readonly success = $('success-overlay');
  private readonly successTime = $('success-time');
  private readonly keypad = $('keypad');
  private readonly keypadTitle = $('keypad-title');
  private readonly keypadDisplay = $('keypad-display');
  private readonly note = $('note');
  private readonly noteText = $('note-text');
  private toastTimer = 0;
  private walletTimer = 0;
  private keypadState: { digits: number; entered: string; check: (code: string) => boolean } | null = null;
  /** Touch-Gerät: Texte für Tippen statt Klicken. */
  touchMode = false;
  /** Wird aufgerufen, wenn ein Dialog auf- oder zugeht (Spielsteuerung pausieren). */
  onDialogChange: (open: boolean) => void = () => {};

  constructor(onStart: () => void, onContinue: () => void) {
    this.startButton.addEventListener('click', onStart);
    $('continue-button').addEventListener('click', onContinue);
    $('restart-button').addEventListener('click', () => location.reload());
    window.addEventListener('keydown', (e) => {
      if (!e.repeat) this.handleDialogKey(e.code);
    });
    // Zifferntasten auf dem Bildschirm (Touch, Maus)
    const pad = $('keypad-buttons');
    for (const [label, code] of [
      ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => [String(n), `Digit${n}`]),
      ['⌫', 'Backspace'],
      ['0', 'Digit0'],
      ['OK', 'Enter'],
    ]) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.addEventListener('click', () => this.handleDialogKey(code));
      pad.append(b);
    }
    for (const b of document.querySelectorAll('[data-close]')) b.addEventListener('click', () => this.closeDialogs());
  }

  /** Geldanzeige oben links, blendet nach ein paar Sekunden aus. */
  setWallet(text: string) {
    const el = $('wallet');
    el.textContent = text;
    el.hidden = false;
    clearTimeout(this.walletTimer);
    this.walletTimer = window.setTimeout(() => (el.hidden = true), 2500);
  }

  /** Dauerhafte Statuszeile oben links (z. B. Promille), null = ausblenden. */
  setStatus(text: string | null) {
    const el = $('status');
    if (el.textContent !== (text ?? '')) el.textContent = text ?? '';
    el.hidden = !text;
  }

  get dialogOpen() {
    return !this.keypad.hidden || !this.note.hidden;
  }

  openKeypad(title: string, digits: number, check: (code: string) => boolean) {
    this.keypadState = { digits, entered: '', check };
    this.keypadTitle.textContent = title;
    this.keypad.classList.remove('error', 'success');
    this.renderKeypad();
    this.keypad.hidden = false;
    this.onDialogChange(true);
  }

  showNote(text: string) {
    this.noteText.textContent = text;
    this.note.hidden = false;
    this.onDialogChange(true);
  }

  closeDialogs() {
    if (!this.dialogOpen) return;
    this.keypad.hidden = true;
    this.note.hidden = true;
    this.keypadState = null;
    this.onDialogChange(false);
  }

  /** Taste im offenen Dialog (Tastatur oder Bildschirmknopf). */
  private handleDialogKey(code: string) {
    if (!this.dialogOpen) return;
    if (code === 'KeyE' || code === 'Escape') {
      this.closeDialogs();
      return;
    }
    const state = this.keypadState;
    if (!state || this.keypad.classList.contains('success')) return;
    const digit = /^(Digit|Numpad)(\d)$/.exec(code)?.[2];
    if (digit && state.entered.length < state.digits) {
      state.entered += digit;
      this.keypad.classList.remove('error');
    } else if (code === 'Backspace') {
      state.entered = state.entered.slice(0, -1);
    } else if ((code === 'Enter' || code === 'NumpadEnter') && state.entered.length === state.digits) {
      if (state.check(state.entered)) {
        this.keypad.classList.add('success');
        window.setTimeout(() => this.closeDialogs(), 600);
      } else {
        this.keypad.classList.remove('error');
        void this.keypad.offsetWidth; // Animation neu starten
        this.keypad.classList.add('error');
        state.entered = '';
      }
    }
    this.renderKeypad();
  }

  private renderKeypad() {
    const state = this.keypadState;
    if (!state) return;
    this.keypadDisplay.replaceChildren(
      ...Array.from({ length: state.digits }, (_, i) => {
        const cell = document.createElement('span');
        cell.textContent = state.entered[i] ?? '';
        return cell;
      }),
    );
  }

  setLoading(text: string | null) {
    this.loading.hidden = text === null;
    if (text !== null) this.loading.textContent = text;
  }

  /** Start- bzw. Pausenbildschirm, solange der Mauszeiger frei ist. */
  showStart(visible: boolean, paused = false) {
    this.startOverlay.hidden = !visible;
    this.crosshair.hidden = visible;
    this.startTitle.textContent = paused ? 'Pause' : 'Escape Room – Architekturbüro';
    this.startButton.textContent = paused ? 'Weiterspielen' : this.touchMode ? 'Tippen zum Starten' : 'Klicken zum Starten';
  }

  setReady(ready: boolean) {
    this.startButton.disabled = !ready;
  }

  setPrompt(text: string | null) {
    this.setText(this.prompt, text);
  }

  setHint(text: string | null) {
    this.setText(this.hint, text);
  }

  /** Kurze Meldung für ein paar Sekunden. */
  toast(text: string) {
    this.toastEl.textContent = text;
    this.toastEl.classList.add('visible');
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => this.toastEl.classList.remove('visible'), 2600);
  }

  showSuccess(seconds: number) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    this.successTime.textContent = `Deine Zeit: ${m}:${s} min`;
    this.success.hidden = false;
    this.startOverlay.hidden = true;
  }

  hideSuccess() {
    this.success.hidden = true;
  }

  get successVisible() {
    return !this.success.hidden;
  }

  private setText(el: HTMLElement, text: string | null) {
    if (el.textContent !== (text ?? '')) el.textContent = text ?? '';
    el.hidden = !text;
  }
}

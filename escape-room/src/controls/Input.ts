/**
 * Tastatur und Maus. Eingaben zählen nur, solange der Mauszeiger gesperrt ist (Klick ins Spiel);
 * Escape gibt ihn frei (Browser-Standard).
 */
export class Input {
  /** Zum Testen ohne Pointer Lock (siehe window.__game im Dev-Modus). */
  forceActive = false;
  private locked = false;
  private readonly keys = new Set<string>();
  private readonly pressed = new Set<string>();
  private readonly buttons = new Set<number>();
  private readonly clicked = new Set<number>();
  private mouseDX = 0;
  private mouseDY = 0;
  private wheel = 0;
  private readonly lockListeners: ((locked: boolean) => void)[] = [];

  constructor(private readonly element: HTMLElement) {
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === element;
      if (!this.locked) this.releaseAll();
      for (const fn of this.lockListeners) fn(this.locked);
    });
    window.addEventListener('keydown', (e) => {
      if (!this.active) return;
      if (!e.repeat) this.pressed.add(e.code);
      this.keys.add(e.code);
      if (e.code === 'Space' || e.code === 'Tab') e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.releaseAll());
    document.addEventListener('mousemove', (e) => {
      if (!this.active) return;
      this.mouseDX += e.movementX;
      this.mouseDY += e.movementY;
    });
    document.addEventListener('mousedown', (e) => {
      if (!this.active) return;
      this.buttons.add(e.button);
      this.clicked.add(e.button);
    });
    document.addEventListener('mouseup', (e) => this.buttons.delete(e.button));
    document.addEventListener('wheel', (e) => {
      if (this.active) this.wheel += e.deltaY;
    });
    document.addEventListener('contextmenu', (e) => {
      if (this.active) e.preventDefault();
    });
  }

  /** Solange ein Dialog offen ist (Zahlenschloss, Zettel), ruht die Spielsteuerung. */
  suspended = false;

  /** Touch-Steuerung läuft (auf Mobilgeräten gibt es keinen Pointer Lock). */
  touchActive = false;
  /** Analoge Laufrichtung vom virtuellen Stick: x = seitlich, y = vorwärts, je −1 … 1. */
  readonly moveAxis = { x: 0, y: 0 };

  get active() {
    return (this.locked || this.forceActive || this.touchActive) && !this.suspended;
  }

  /** Taste von einem Bildschirmknopf. */
  virtualKey(code: string, down: boolean) {
    if (down) {
      if (!this.keys.has(code)) this.pressed.add(code);
      this.keys.add(code);
    } else {
      this.keys.delete(code);
    }
  }

  /** Maustaste von einem Bildschirmknopf (0 = links, 2 = rechts). */
  virtualMouse(button: number, down: boolean) {
    if (down) {
      this.buttons.add(button);
      this.clicked.add(button);
    } else {
      this.buttons.delete(button);
    }
  }

  /** Umsehen per Wischen. */
  addLook(dx: number, dy: number) {
    this.mouseDX += dx;
    this.mouseDY += dy;
  }

  requestLock(onFail?: () => void) {
    // Manche Browser liefern ein Promise, das bei zu schnellem Wiederholen abgelehnt wird.
    try {
      Promise.resolve(this.element.requestPointerLock()).catch(() => onFail?.());
    } catch {
      onFail?.();
    }
  }

  get isLocked() {
    return this.locked;
  }

  exitLock() {
    if (this.locked) document.exitPointerLock();
  }

  onLockChange(fn: (locked: boolean) => void) {
    this.lockListeners.push(fn);
  }

  isDown(code: string) {
    return this.keys.has(code);
  }

  /** Nur im Frame des Drückens true. */
  wasPressed(code: string) {
    return this.pressed.has(code);
  }

  isMouseDown(button: number) {
    return this.buttons.has(button);
  }

  wasClicked(button: number) {
    return this.clicked.has(button);
  }

  consumeMouse() {
    const delta = { dx: this.mouseDX, dy: this.mouseDY };
    this.mouseDX = this.mouseDY = 0;
    return delta;
  }

  consumeWheel() {
    const w = this.wheel;
    this.wheel = 0;
    return w;
  }

  /** Am Ende jedes Frames: einmalige Ereignisse zurücksetzen. */
  endFrame() {
    this.pressed.clear();
    this.clicked.clear();
  }

  private releaseAll() {
    this.keys.clear();
    this.buttons.clear();
    this.mouseDX = this.mouseDY = this.wheel = 0;
  }
}

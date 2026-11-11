import type { Input } from './Input';

/** Läuft das Spiel auf einem Gerät mit Touch als Hauptbedienung (Handy, Tablet)? */
export const isTouchDevice = () => matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 1;

const LOOK_SENSITIVITY = 1.6;

/**
 * Bildschirmsteuerung für Mobilgeräte:
 *  - linker Daumen: virtueller Stick zum Laufen (ab ~85 % Auslenkung wird gerannt)
 *  - rechte Bildschirmhälfte wischen: umsehen
 *  - Knöpfe rechts: Aktion (E), Schlag/Werfen (Linksklick), Zeigen/Drehen (rechte Maustaste),
 *    Essen/Trinken (F), Springen; „⋯“ öffnet weitere Tasten
 */
export class TouchControls {
  private readonly root = document.createElement('div');
  private stickId: number | null = null;
  private readonly stickOrigin = { x: 0, y: 0 };
  private readonly lookTouches = new Map<number, { x: number; y: number }>();
  private readonly knob: HTMLDivElement;
  private readonly base: HTMLDivElement;

  constructor(
    private readonly input: Input,
    onPause: () => void,
    extraKeys: { label: string; code: string; hold?: boolean }[],
  ) {
    this.root.id = 'touch-controls';
    this.root.hidden = true;
    document.body.append(this.root);

    // Linke Hälfte: Stick, rechte Hälfte: Umsehen
    const lookZone = div('touch-look');
    const stickZone = div('touch-stick-zone');
    this.base = div('touch-stick-base');
    this.knob = div('touch-stick-knob');
    this.base.append(this.knob);
    stickZone.append(this.base);
    this.root.append(lookZone, stickZone);

    stickZone.addEventListener('touchstart', (e) => this.stickStart(e), { passive: false });
    stickZone.addEventListener('touchmove', (e) => this.stickMove(e), { passive: false });
    stickZone.addEventListener('touchend', (e) => this.stickEnd(e));
    stickZone.addEventListener('touchcancel', (e) => this.stickEnd(e));

    lookZone.addEventListener('touchstart', (e) => {
      e.preventDefault();
      for (const t of e.changedTouches) this.lookTouches.set(t.identifier, { x: t.clientX, y: t.clientY });
    }, { passive: false });
    lookZone.addEventListener('touchmove', (e) => {
      e.preventDefault();
      for (const t of e.changedTouches) {
        const last = this.lookTouches.get(t.identifier);
        if (!last) continue;
        this.input.addLook((t.clientX - last.x) * LOOK_SENSITIVITY, (t.clientY - last.y) * LOOK_SENSITIVITY);
        last.x = t.clientX;
        last.y = t.clientY;
      }
    }, { passive: false });
    const endLook = (e: TouchEvent) => {
      for (const t of e.changedTouches) this.lookTouches.delete(t.identifier);
    };
    lookZone.addEventListener('touchend', endLook);
    lookZone.addEventListener('touchcancel', endLook);

    // Knöpfe
    const buttons = div('touch-buttons');
    buttons.append(
      this.button('Aktion', 'big', (down) => this.input.virtualKey('KeyE', down)),
      this.button('Schlag / Wurf', '', (down) => this.input.virtualMouse(0, down)),
      // Gedrückt halten: zeigen, mit Gegenstand + Wischen: drehen
      this.button('Zeigen / Drehen', '', (down) => this.input.virtualMouse(2, down)),
      this.button('Essen / Trinken', '', (down) => this.input.virtualKey('KeyF', down)),
      this.button('Springen', '', (down) => this.input.virtualKey('Space', down)),
    );
    const more = div('touch-more');
    more.hidden = true;
    for (const k of extraKeys) more.append(this.button(k.label, 'small', (down) => this.input.virtualKey(k.code, down)));
    const top = div('touch-top');
    top.append(
      this.button('⋯', 'round', (down) => {
        if (down) more.hidden = !more.hidden;
      }),
      this.button('❚❚', 'round', (down) => {
        if (down) onPause();
      }),
    );
    this.root.append(buttons, more, top);
  }

  setVisible(visible: boolean) {
    this.root.hidden = !visible;
    if (!visible) {
      this.input.moveAxis.x = this.input.moveAxis.y = 0;
      this.stickId = null;
      this.lookTouches.clear();
      this.base.classList.remove('active');
    }
  }

  private button(label: string, variant: string, onChange: (down: boolean) => void) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = `touch-btn ${variant}`;
    b.textContent = label;
    const down = (e: Event) => {
      e.preventDefault();
      b.classList.add('pressed');
      onChange(true);
    };
    const up = (e: Event) => {
      e.preventDefault();
      if (!b.classList.contains('pressed')) return;
      b.classList.remove('pressed');
      onChange(false);
    };
    b.addEventListener('touchstart', down, { passive: false });
    b.addEventListener('touchend', up);
    b.addEventListener('touchcancel', up);
    return b;
  }

  private stickStart(e: TouchEvent) {
    e.preventDefault();
    if (this.stickId !== null) return;
    const t = e.changedTouches[0];
    this.stickId = t.identifier;
    this.stickOrigin.x = t.clientX;
    this.stickOrigin.y = t.clientY;
    // Stick erscheint dort, wo der Daumen aufsetzt
    this.base.style.left = `${t.clientX}px`;
    this.base.style.top = `${t.clientY}px`;
    this.base.classList.add('active');
    this.knob.style.transform = 'translate(-50%, -50%)';
  }

  private stickMove(e: TouchEvent) {
    e.preventDefault();
    for (const t of e.changedTouches) {
      if (t.identifier !== this.stickId) continue;
      const radius = 55;
      let dx = t.clientX - this.stickOrigin.x;
      let dy = t.clientY - this.stickOrigin.y;
      const len = Math.hypot(dx, dy);
      if (len > radius) {
        dx = (dx / len) * radius;
        dy = (dy / len) * radius;
      }
      this.knob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
      this.input.moveAxis.x = dx / radius;
      this.input.moveAxis.y = -dy / radius;
      // Weit ausgelenkt = rennen
      this.input.virtualKey('ShiftLeft', Math.min(1, len / radius) > 0.85);
    }
  }

  private stickEnd(e: TouchEvent) {
    for (const t of e.changedTouches) {
      if (t.identifier !== this.stickId) continue;
      this.stickId = null;
      this.input.moveAxis.x = this.input.moveAxis.y = 0;
      this.input.virtualKey('ShiftLeft', false);
      this.base.classList.remove('active');
    }
  }
}

function div(className: string) {
  const d = document.createElement('div');
  d.className = className;
  return d;
}

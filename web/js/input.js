// Tastatur/Maus (Pointer Lock oder Ersatzmodus) + skriptbare Eingabe für Tests (window.__game.setInput)
// Kamera drehen geht IMMER: Maus (gefangen), Maus ziehen / an den Bildrand fahren (Ersatzmodus),
// Pfeiltasten ← → (drehen) und Bild↑/Bild↓ (neigen), auch per Skript (lookX/lookY).
export class Input {
  constructor(canvas) {
    this.canvas = canvas;
    this.keys = new Set(); this.edges = new Set();
    this.mouseL = false; this.mouseR = false;
    this.dx = 0; this.dy = 0; this.wheel = 0;
    this.script = {};
    this.locked = false;
    this.free = false;                 // Ersatzmodus ohne Pointer Lock
    this.careful = false;              // C schaltet um
    this.px = 0.5; this.py = 0.5; this.inside = false; this.lastX = null; this.lastY = null;
    this.keyTurn = 2.4;                // rad/s für Pfeiltasten
    addEventListener('keydown', (e) => {
      if (!this.active && !this.script.__enabled) return;
      if (['Space', 'Tab', 'KeyF', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight', 'F3', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'PageUp', 'PageDown'].includes(e.code)) e.preventDefault();
      if (!e.repeat) this.edges.add(e.code);
      this.keys.add(e.code);
      if (e.code === 'KeyC' && !e.repeat) this.careful = !this.careful;
    });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => { this.keys.clear(); this.mouseL = this.mouseR = false; });
    addEventListener('mousemove', (e) => {
      this.px = e.clientX / Math.max(1, innerWidth); this.py = e.clientY / Math.max(1, innerHeight); this.inside = true;
      let mx = e.movementX, my = e.movementY;
      // manche eingebetteten Browser liefern ohne Pointer Lock keine movementX -> selbst aus clientX ableiten
      if (!this.locked && this.lastX !== null && (mx === undefined || (mx === 0 && my === 0))) { mx = e.clientX - this.lastX; my = e.clientY - this.lastY; }
      this.lastX = e.clientX; this.lastY = e.clientY;
      if (!this.active) return;
      if (this.locked) { this.dx += mx || 0; this.dy += my || 0; }
      else { this.dx += (mx || 0) * 1.4; this.dy += (my || 0) * 1.0; }   // Ersatzmodus: Maus bewegt die Kamera direkt
    });
    document.addEventListener('mouseleave', () => { this.inside = false; });
    addEventListener('mouseout', (e) => { if (!e.relatedTarget) this.inside = false; });
    addEventListener('mousedown', (e) => { if (!this.active) return; if (e.button === 0) this.mouseL = true; if (e.button === 2) this.mouseR = true; });
    addEventListener('mouseup', (e) => { if (e.button === 0) this.mouseL = false; if (e.button === 2) this.mouseR = false; });
    addEventListener('contextmenu', (e) => e.preventDefault());
    addEventListener('wheel', (e) => { if (this.active) this.wheel += e.deltaY; }, { passive: true });
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === canvas;
      if (!this.locked && !this.free) { this.keys.clear(); this.mouseL = this.mouseR = false; }
    });
  }
  get active() { return this.locked || this.free; }
  lock() {
    const plain = () => { try { const p = this.canvas.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch { /* Ersatzmodus übernimmt */ } };
    try { const p = this.canvas.requestPointerLock({ unadjustedMovement: true }); if (p && p.catch) p.catch(plain); } catch { plain(); }
  }
  k(...codes) { return codes.some((c) => this.keys.has(c)); }
  // aktueller Steuerzustand; edge = einmalige Tastendrücke seit dem letzten Aufruf
  state() {
    const s = this.script;
    const st = {
      forward: this.k('KeyW', 'ArrowUp') || !!s.forward,
      back: this.k('KeyS', 'ArrowDown') || !!s.back,
      left: this.k('KeyA') || !!s.left,
      right: this.k('KeyD') || !!s.right,
      run: this.k('ShiftLeft', 'ShiftRight') || !!s.run,
      careful: this.careful || !!s.careful,
      // Maustasten nur bei gefangener Maus fürs Gleichgewicht (im Ersatzmodus dient Ziehen dem Drehen)
      balL: this.k('KeyQ') || (this.locked && this.mouseL) || !!s.balL,
      balR: this.k('KeyE') || (this.locked && this.mouseR) || !!s.balR,
      jump: this.edges.has('Space') || !!s.jump,
      scan: this.edges.has('KeyF') || !!s.scan,
      swap: this.edges.has('KeyV'),
      help: this.edges.has('KeyH'),
      admin: this.edges.has('KeyM') || !!s.admin,
      debug: this.edges.has('F3') || this.edges.has('Backquote'),
    };
    s.jump = false; s.scan = false; s.admin = false;     // Skript-Impulse nur einmal
    this.edges.clear();
    return st;
  }
  // dt: Sekunden seit dem letzten Aufruf. Liefert Kamera-Drehung in "Pixeln" (CameraRig.sens rad/px).
  consumeLook(dt = 1 / 60, sens = 0.0021) {
    let dx = this.dx, dy = this.dy;
    if (this.active || this.script.__enabled) {
      const s = this.script, px = this.keyTurn * dt / sens;
      // Pfeiltasten / Skript: drehen und neigen
      const turn = (this.k('ArrowRight', 'Numpad6', 'KeyL') ? 1 : 0) - (this.k('ArrowLeft', 'Numpad4', 'KeyJ') ? 1 : 0) + (s.lookX || 0);
      const tilt = (this.k('PageDown', 'Numpad2', 'KeyK') ? 1 : 0) - (this.k('PageUp', 'Numpad8', 'KeyI') ? 1 : 0) + (s.lookY || 0);
      dx += turn * px; dy += tilt * px * 0.5;
      // Ersatzmodus ohne gefangene Maus: am linken/rechten Bildrand dreht die Kamera weiter (sonst endet die Drehung am Fensterrand)
      if (this.free && !this.locked && this.inside) {
        const edge = 0.1;
        if (this.px < edge) dx -= ((edge - this.px) / edge) * px * 1.1;
        else if (this.px > 1 - edge) dx += ((this.px - (1 - edge)) / edge) * px * 1.1;
      }
    }
    const o = { dx, dy, wheel: this.wheel }; this.dx = this.dy = this.wheel = 0; return o;
  }
}

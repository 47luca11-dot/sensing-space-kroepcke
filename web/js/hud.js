// Minimales HUD: Kompass, Ortsname, Gleichgewicht (Bogen), Ausdauer, Hinweise, Meldungen.
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// Bogen unter der Bildmitte: Mittelpunkt (120, 0), Radius 66, ±ARC um die Senkrechte
const CX = 120, CY = 4, RAD = 66, ARC = 52 * Math.PI / 180;
const pt = (a, r = RAD) => [CX + Math.sin(a) * r, CY + Math.cos(a) * r];
const arcPath = (a0, a1, r = RAD) => { const [x0, y0] = pt(a0, r), [x1, y1] = pt(a1, r); return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 0 0 ${x1.toFixed(1)} ${y1.toFixed(1)}`; };

export class Hud {
  constructor() {
    this.el = $('hud');
    this.marker = $('bal-marker'); this.kQ = $('kQ'); this.kE = $('kE');
    this.stam = $('stamina'); this.stamBar = this.stam.querySelector('i');
    this.adminEl = $('admin');
    this.meters = $('meters'); this.mode = $('mode');
    this.place = $('place'); this.placeName = this.place.querySelector('span');
    this.streamEl = $('stream');
    this.hint = $('hint'); this.toastEl = $('toast'); this.flash = $('scanflash'); this.fps = $('fps');
    // Bogen: von links (a = -ARC) nach rechts (a = +ARC) über den tiefsten Punkt
    $('bal-track').setAttribute('d', arcPath(-ARC, ARC));
    $('bal-warnL').setAttribute('d', arcPath(-ARC, -ARC * 0.72));
    $('bal-warnR').setAttribute('d', arcPath(ARC * 0.72, ARC));
    const [cx0, cy0] = pt(0, RAD - 8), [cx1, cy1] = pt(0, RAD + 8);
    const c = $('bal-center'); c.setAttribute('x1', cx0); c.setAttribute('y1', cy0); c.setAttribute('x2', cx1); c.setAttribute('y2', cy1);
    // Kompass
    this.strip = $('compass-strip'); this.ticks = [];
    const names = { 0: 'N', 45: 'NO', 90: 'O', 135: 'SO', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };
    for (let d = 0; d < 360; d += 15) {
      const e = document.createElement(names[d] !== undefined ? 'b' : 'i');
      if (names[d] !== undefined) { e.textContent = names[d]; if (d === 0) e.className = 'n'; }
      this.strip.appendChild(e); this.ticks.push({ e, d });
    }
    this.hintT = 16; this.toastT = 0; this.flashT = 0; this.vis = 1; this.lastPlace = null; this.debug = false;
    this.cache = {};
  }
  set(k, v, fn) { if (this.cache[k] !== v) { this.cache[k] = v; fn(v); } }
  show(on) { this.el.classList.toggle('off', !on); }
  toast(text, sec = 1.8) { this.toastEl.textContent = text; this.toastEl.classList.add('on'); this.toastT = sec; }
  scanFlash() { this.flashT = 1; }
  toggleHint() { this.hintT = this.hint.classList.contains('off') ? 1e9 : 0; }

  update(dt, s) {
    // --- Kompass: Blickrichtung der Kamera (0° = Norden = -Z) ---
    const bearing = ((-s.camYaw * 180 / Math.PI) % 360 + 360) % 360;
    const W = this.strip.clientWidth || 440, pxPerDeg = W / 150;
    for (const t of this.ticks) {
      let d = t.d - bearing; d = ((d + 540) % 360) - 180;
      const x = W / 2 + d * pxPerDeg;
      const vis = Math.abs(d) < 80;
      t.e.style.display = vis ? '' : 'none';
      if (vis) t.e.style.left = x.toFixed(1) + 'px';
    }
    // --- Ort ---
    const name = s.place ? s.place.name : null;
    if (name !== this.lastPlace) {
      this.lastPlace = name;
      if (name) { this.placeName.textContent = name; this.place.classList.add('on'); }
      else this.place.classList.remove('on');
    }
    // --- Gleichgewicht ---
    const bx = clamp(s.bal.x, -1.1, 1.1), by = clamp(s.bal.y, -1.1, 1.1);
    const a = bx * ARC, r = RAD + by * 9;                 // rechts kippen -> Marker nach rechts; vorn -> nach außen
    const [mx, my] = pt(a, r);
    this.marker.setAttribute('transform', `translate(${mx.toFixed(1)} ${my.toFixed(1)})`);
    const m = Math.max(Math.abs(bx), Math.abs(by));
    this.set('mcls', m > 0.72 ? 'bad' : m > 0.42 ? 'warn' : '', (v) => this.marker.setAttribute('class', v));
    this.set('kQ', s.inp.balL ? 'held' : bx < -0.42 ? 'need' : '', (v) => (this.kQ.className = v));
    this.set('kE', s.inp.balR ? 'held' : bx > 0.42 ? 'need' : '', (v) => (this.kE.className = v));
    // --- Ausdauer ---
    this.set('admin', s.admin ? 'on' : '', (v) => { this.stam.style.display = v ? 'none' : ''; if (this.adminEl) this.adminEl.classList.toggle('on', !!v); });
    if (this.adminEl && s.admin) this.set('aspd', `ADMIN-MODUS · ${Math.round((s.speed || 0) * 3.6)} KM/H`, (v) => (this.adminEl.textContent = v));
    this.stamBar.style.transform = `scaleX(${s.stamina.toFixed(3)})`;
    this.set('scls', s.exhausted ? 'empty' : s.stamina < 0.3 ? 'low' : '', (v) => (this.stam.className = v));
    const modeText = { run: 'LAUFEN', careful: 'VORSICHTIG', stumble: 'GESTOLPERT', air: '', walk: '' }[s.mode] || '';
    this.set('mode', s.stabilize ? 'STABILISIEREN' : modeText, (v) => (this.mode.textContent = v));
    // HUD-Elemente blenden sich zurück, wenn alles ruhig ist
    const busy = s.admin || m > 0.2 || s.stamina < 0.98 || s.mode === 'run' || s.inp.balL || s.inp.balR || s.mode === 'careful';
    this.vis += ((busy ? 1 : 0.28) - this.vis) * Math.min(1, dt * (busy ? 6 : 0.8));
    this.meters.style.opacity = this.vis.toFixed(2);
    // --- Hinweise / Meldungen ---
    if (this.hintT > 0) { this.hintT -= dt; if (this.hintT <= 0) this.hint.classList.add('off'); else this.hint.classList.remove('off'); }
    if (this.toastT > 0) { this.toastT -= dt; if (this.toastT <= 0) this.toastEl.classList.remove('on'); }
    if (this.flashT > 0) { this.flashT = Math.max(0, this.flashT - dt * 1.5); this.flash.style.opacity = this.flashT.toFixed(2); }
    if (this.debug) this.fps.textContent = s.debugText || '';
    const st = s.stream;
    this.set('stream', st && st.active ? `STADT LÄDT … ${st.loaded}/${st.total}` : '', (v) => (this.streamEl.textContent = v));
  }
  toggleDebug() { this.debug = !this.debug; this.fps.classList.toggle('on', this.debug); }
}

// Karte + Minimap aus dem Gedächtnis: Sichtbar ist nur, was man mit F gescannt (oder aus nächster Nähe gesehen) hat.
// - Kartenbild: Draufsicht der ganzen Punktwolke (pipeline/karte/build_map.mjs -> assets/city/karte/karte.png)
// - Gedächtnis: Maske (1 m/px). Jeder Scan wirft 2x360 Strahlen gegen die Kollisionsgeometrie und füllt das sichtbare
//   Polygon (bis 150 m) – wie die Scanwelle. Wird im Browser gespeichert und beim nächsten Spiel wieder geladen.
// - Minimap oben rechts (dreht mit der Kamera), große Karte mit M (ziehen = verschieben, Mausrad = Zoom).
// - Reisen: Bahn-Eingang hinunter oder ans Ende der Passerelle -> "Wohin willst du?" -> Haltestelle anklicken.
import * as THREE from 'three';

const MEM_KEY = 'hannover.karte.gedaechtnis.v1';
const MASK_MPP = 1.0;            // Gedächtnis-Auflösung (m/px)
const SCAN_R = 150, NAH_R = 9;   // Scan-Reichweite, Sicht aus nächster Nähe

const CSS = `
#minimap{position:absolute;right:22px;top:20px;width:196px;height:196px;border-radius:50%;overflow:hidden;
  box-shadow:0 0 0 1px rgba(170,215,255,.35),0 0 24px rgba(60,140,255,.18);background:#020304;pointer-events:none}
#minimap canvas{width:100%;height:100%;display:block}
#minimap b{position:absolute;left:50%;top:6px;transform:translateX(-50%);font:600 10px/1 var(--mono,monospace);color:#9fd0ff;letter-spacing:.12em}
#minimap .hint{position:absolute;left:0;right:0;bottom:9px;text-align:center;font:500 9px/1 var(--mono,monospace);color:rgba(190,220,255,.55);letter-spacing:.14em}
#karte{position:fixed;inset:0;z-index:40;background:#020304;display:none;cursor:grab;user-select:none}
#karte.on{display:block} #karte.drag{cursor:grabbing}
#karte canvas{position:absolute;inset:0;width:100%;height:100%}
#karte .kopf{position:absolute;left:0;right:0;top:0;padding:22px 30px;display:flex;align-items:baseline;gap:22px;
  background:linear-gradient(#020304ee,#02030400);font-family:var(--mono,monospace);color:#cfe6ff;pointer-events:none}
#karte .kopf h2{margin:0;font:700 22px/1 var(--display,inherit);letter-spacing:.28em}
#karte .kopf span{font-size:11px;letter-spacing:.16em;opacity:.7}
#karte .liste{position:absolute;right:26px;top:84px;width:290px;display:none;flex-direction:column;gap:8px}
#karte.reise .liste{display:flex}
#karte .liste button{all:unset;cursor:pointer;padding:11px 14px;border:1px solid rgba(160,210,255,.25);background:rgba(8,16,26,.82);
  color:#d9ecff;font:600 13px/1.25 var(--mono,monospace);letter-spacing:.08em}
#karte .liste button small{display:block;font-weight:400;font-size:10px;opacity:.6;margin-top:3px;letter-spacing:.04em}
#karte .liste button:hover,#karte .liste button.hl{border-color:#7fd0ff;background:rgba(20,60,100,.85);box-shadow:0 0 18px rgba(80,170,255,.35)}
#karte .liste button.hier{opacity:.45;cursor:default}
#karte .fuss{position:absolute;left:30px;bottom:22px;font:500 11px/1.6 var(--mono,monospace);color:rgba(190,220,255,.6);letter-spacing:.12em;pointer-events:none}
#reiseblende{position:fixed;inset:0;background:#000;opacity:0;pointer-events:none;transition:opacity .55s;z-index:39;
  display:flex;align-items:center;justify-content:center;font:700 26px/1 var(--display,inherit);letter-spacing:.4em;color:#bfe0ff}
#reiseblende.on{opacity:1}
`;

export class Karte {
  constructor({ world, hud, base = 'assets/city/' }) {
    this.world = world; this.hud = hud; this.base = base + 'karte/';
    this.ready = false; this.offen = false; this.modus = 'karte';
    this.view = { cx: 0, cz: 0, mpp: 1.2 };          // große Karte: Mitte + Meter pro Bildschirmpixel
    this.zone = null; this.sperre = 0; this.onTravel = null; this.onClose = null;
    this._mini = 0;
  }

  async load() {
    const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
    const [meta, stat] = await Promise.all([fetch(this.base + 'karte.json').then((r) => r.json()), fetch(this.base + 'stationen.json').then((r) => r.json())]);
    this.meta = meta; this.stat = stat;
    this.img = new Image(); this.img.decoding = 'async';
    await new Promise((res, rej) => { this.img.onload = res; this.img.onerror = rej; this.img.src = this.base + 'karte.png'; });
    // Gedächtnis-Maske
    const mw = Math.ceil((meta.x1 - meta.x0) / MASK_MPP), mh = Math.ceil((meta.z1 - meta.z0) / MASK_MPP);
    this.mask = document.createElement('canvas'); this.mask.width = mw; this.mask.height = mh;
    this.mctx = this.mask.getContext('2d');
    await this.ladeGedaechtnis();
    // Minimap
    this.miniEl = document.createElement('div'); this.miniEl.id = 'minimap';
    this.mini = document.createElement('canvas'); this.mini.width = this.mini.height = 392;
    this.miniEl.append(this.mini); const n = document.createElement('b'); n.textContent = 'N'; this.miniN = n; this.miniEl.append(n);
    const h = document.createElement('div'); h.className = 'hint'; h.textContent = 'M · KARTE'; this.miniEl.append(h);
    (document.getElementById('hud') || document.body).append(this.miniEl);
    this.tmp = document.createElement('canvas'); this.tmp.width = this.tmp.height = 392;
    // große Karte
    this.el = document.createElement('div'); this.el.id = 'karte';
    this.el.innerHTML = '<canvas></canvas><div class="kopf"><h2>KARTE</h2><span></span></div><div class="liste"></div><div class="fuss"></div>';
    document.body.append(this.el);
    this.cv = this.el.querySelector('canvas'); this.ctx = this.cv.getContext('2d');
    this.layer = document.createElement('canvas'); this.lctx = this.layer.getContext('2d');
    this.blende = document.createElement('div'); this.blende.id = 'reiseblende'; document.body.append(this.blende);
    this.bindEvents();
    this.ready = true;
  }

  // ---------- Gedächtnis ----------
  w2m(x, z) { return [(x - this.meta.x0) / MASK_MPP, (z - this.meta.z0) / MASK_MPP]; }
  async ladeGedaechtnis() {
    let url = null; try { url = localStorage.getItem(MEM_KEY); } catch { /* ohne Speicher */ }
    if (!url) return;
    const im = new Image(); await new Promise((r) => { im.onload = r; im.onerror = r; im.src = url; });
    if (im.width) this.mctx.drawImage(im, 0, 0, this.mask.width, this.mask.height);
  }
  speichern() {
    clearTimeout(this._saveT);
    this._saveT = setTimeout(() => {
      try {
        const s = document.createElement('canvas'); s.width = this.mask.width >> 1; s.height = this.mask.height >> 1;
        s.getContext('2d').drawImage(this.mask, 0, 0, s.width, s.height);
        localStorage.setItem(MEM_KEY, s.toDataURL('image/png'));
      } catch { /* Speicher voll/gesperrt: Gedächtnis gilt dann nur für diese Sitzung */ }
    }, 1500);
  }
  vergessen() { this.mctx.clearRect(0, 0, this.mask.width, this.mask.height); try { localStorage.removeItem(MEM_KEY); } catch { /* egal */ } }
  _poly(pts, weich) {
    const c = this.mctx; c.save(); c.fillStyle = '#fff';
    if (weich) c.filter = `blur(${weich}px)`;
    c.beginPath(); pts.forEach(([u, v], i) => (i ? c.lineTo(u, v) : c.moveTo(u, v))); c.closePath(); c.fill(); c.restore();
  }
  // Scan: sichtbares Gebiet vom Scan-Ursprung (zwei Höhen: über Hindernisse hinweg blicken wie die Scanwelle)
  scan(o) {
    if (!this.ready) return;
    const N = 360, pts = [], d = new THREE.Vector3(), p = new THREE.Vector3();
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2; d.set(Math.sin(a), 0, Math.cos(a));
      let r = 0;
      for (const hy of [0.4, 2.2]) {
        p.set(o.x, o.y - 1.55 + hy, o.z);
        const hit = this.world.raycast(p, d, SCAN_R);
        r = Math.max(r, Math.min(SCAN_R, hit === Infinity ? SCAN_R : hit + 1.5));
      }
      pts.push(this.w2m(o.x + d.x * r, o.z + d.z * r));
    }
    this._poly(pts, 1.5);
    this.speichern();
  }
  // was man aus nächster Nähe sieht, bleibt auch im Gedächtnis (kleiner Radius)
  nah(pos) {
    const [u, v] = this.w2m(pos.x, pos.z), r = NAH_R / MASK_MPP, c = this.mctx;
    const g = c.createRadialGradient(u, v, r * 0.4, u, v, r); g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    c.fillStyle = g; c.fillRect(u - r, v - r, 2 * r, 2 * r);
  }

  // ---------- Zeichnen ----------
  // Ausschnitt der Karte (Weltmitte cx/cz, mpp Meter je Zielpixel, Drehung rot) in ctx zeichnen, nur Gedächtnis sichtbar
  _ausschnitt(ctx, W, H, cx, cz, mpp, rot = 0) {
    const L = this.layer; if (L.width !== W || L.height !== H) { L.width = W; L.height = H; }
    const l = this.lctx, m = this.meta;
    for (const [src, smpp] of [[this.img, m.mpp], [this.mask, MASK_MPP]]) {
      const c = src === this.img ? ctx : l;
      c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, W, H);
      c.translate(W / 2, H / 2); c.rotate(rot); c.scale(smpp / mpp, smpp / mpp);
      c.translate(-(cx - m.x0) / smpp, -(cz - m.z0) / smpp);
      c.imageSmoothingEnabled = src !== this.img || mpp > m.mpp * 1.2;
      c.drawImage(src, 0, 0); c.restore();
    }
    ctx.save(); ctx.globalCompositeOperation = 'destination-in'; ctx.drawImage(L, 0, 0); ctx.restore();
  }
  _bearing(yaw) { return Math.atan2(Math.sin(yaw), -Math.cos(yaw)); }   // Spieler-yaw -> Kompasswinkel (0 = Nord)

  minimap(player, camYaw) {
    const c = this.mini.getContext('2d'), S = this.mini.width, camB = -camYaw;   // Kamera-Kompasswinkel
    c.fillStyle = '#020304'; c.fillRect(0, 0, S, S);
    this._ausschnitt(c, S, S, player.pos.x, player.pos.z, 70 / (S / 2), -camB);
    // Haltestellen in Reichweite
    c.save(); c.translate(S / 2, S / 2); c.rotate(-camB);
    for (const st of this.stat.stationen) {
      const dx = (st.punkt[0] - player.pos.x) / (70 / (S / 2)), dz = (st.punkt[1] - player.pos.z) / (70 / (S / 2));
      if (dx * dx + dz * dz > (S / 2 - 14) ** 2) continue;
      this._uSymbol(c, dx, dz, 9, false);
    }
    c.restore();
    // Spieler-Pfeil (zeigt seine Laufrichtung relativ zur Kamera)
    c.save(); c.translate(S / 2, S / 2); c.rotate(this._bearing(player.yaw) - camB);
    c.fillStyle = '#9fe0ff'; c.shadowColor = '#4fb8ff'; c.shadowBlur = 14;
    c.beginPath(); c.moveTo(0, -15); c.lineTo(10, 11); c.lineTo(0, 5); c.lineTo(-10, 11); c.closePath(); c.fill(); c.restore();
    // Rand + Norden
    c.strokeStyle = 'rgba(150,205,255,.35)'; c.lineWidth = 2; c.beginPath(); c.arc(S / 2, S / 2, S / 2 - 1, 0, Math.PI * 2); c.stroke();
    const nb = -camB, R = 196 / 2 - 10;
    this.miniN.style.left = (98 + Math.sin(nb) * R) + 'px'; this.miniN.style.top = (98 - Math.cos(nb) * R - 5) + 'px'; this.miniN.style.transform = 'translateX(-50%)';
  }
  _uSymbol(c, x, y, r, hell) {
    c.save(); c.translate(x, y);
    c.fillStyle = hell ? '#1c6fd6' : 'rgba(28,111,214,.85)'; c.strokeStyle = '#e8f4ff'; c.lineWidth = Math.max(1.2, r * 0.16);
    c.beginPath(); c.rect(-r, -r, 2 * r, 2 * r); c.fill(); c.stroke();
    c.fillStyle = '#fff'; c.font = `700 ${Math.round(r * 1.4)}px sans-serif`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('U', 0, r * 0.08);
    c.restore();
  }

  zeichneGross() {
    const dpr = Math.min(devicePixelRatio || 1, 2), W = Math.round(innerWidth * dpr), H = Math.round(innerHeight * dpr);
    if (this.cv.width !== W || this.cv.height !== H) { this.cv.width = W; this.cv.height = H; }
    const c = this.ctx, v = this.view, mpp = v.mpp / dpr;
    c.fillStyle = '#020304'; c.fillRect(0, 0, W, H);
    this._ausschnitt(c, W, H, v.cx, v.cz, mpp, 0);
    // feines Raster (100 m)
    c.save(); c.strokeStyle = 'rgba(120,180,255,.06)'; c.lineWidth = 1;
    const x0 = v.cx - W / 2 * mpp, z0 = v.cz - H / 2 * mpp;
    for (let x = Math.ceil(x0 / 100) * 100; x < x0 + W * mpp; x += 100) { const u = (x - x0) / mpp; c.beginPath(); c.moveTo(u, 0); c.lineTo(u, H); c.stroke(); }
    for (let z = Math.ceil(z0 / 100) * 100; z < z0 + H * mpp; z += 100) { const y = (z - z0) / mpp; c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
    c.restore();
    const S = (x, z) => [(x - x0) / mpp, (z - z0) / mpp];
    // Passerelle-Ende (Reiseziel-Auswahl)
    this._ziele = [];
    for (const st of this.stat.stationen) {
      const [u, y] = S(st.punkt[0], st.punkt[1]); const hier = st.name === this.hier;
      const hl = this._hover === st.name;
      this._uSymbol(c, u, y, (hl ? 15 : 12) * dpr, hl);
      c.save(); c.font = `600 ${12 * dpr}px monospace`; c.fillStyle = hier ? 'rgba(220,235,255,.5)' : '#e4f2ff'; c.shadowColor = '#000'; c.shadowBlur = 6;
      c.fillText(st.name.toUpperCase() + (hier ? '  · HIER' : ''), u + 18 * dpr, y + 4 * dpr); c.restore();
      this._ziele.push({ st, u: u / dpr, y: y / dpr });
    }
    // Spieler
    if (this.player) {
      const [u, y] = S(this.player.pos.x, this.player.pos.z);
      c.save(); c.translate(u, y); c.rotate(this._bearing(this.player.yaw)); c.fillStyle = '#9fe0ff'; c.shadowColor = '#4fb8ff'; c.shadowBlur = 18 * dpr;
      const k = dpr * 1.1; c.beginPath(); c.moveTo(0, -14 * k); c.lineTo(10 * k, 10 * k); c.lineTo(0, 5 * k); c.lineTo(-10 * k, 10 * k); c.closePath(); c.fill(); c.restore();
    }
    // Maßstab
    c.save(); c.fillStyle = 'rgba(200,225,255,.7)'; c.font = `500 ${11 * dpr}px monospace`;
    const L = 100 / mpp; c.fillRect(W - 40 * dpr - L, H - 30 * dpr, L, 2 * dpr); c.fillText('100 m', W - 40 * dpr - L, H - 38 * dpr); c.restore();
  }

  // ---------- Öffnen / Reisen ----------
  oeffnen(modus = 'karte', hier = null) {
    if (!this.ready) return;
    this.offen = true; this.modus = modus; this.hier = hier;
    if (this.player) { this.view.cx = this.player.pos.x; this.view.cz = this.player.pos.z; }
    this.view.mpp = modus === 'reise' ? 1.15 : 0.9;
    if (modus === 'reise') {                                   // Ausschnitt so, dass alle Haltestellen sichtbar sind
      const xs = this.stat.stationen.map((s) => s.punkt[0]), zs = this.stat.stationen.map((s) => s.punkt[1]);
      this.view.cx = (Math.min(...xs) + Math.max(...xs)) / 2 + 60; this.view.cz = (Math.min(...zs) + Math.max(...zs)) / 2;
      this.view.mpp = Math.max((Math.max(...xs) - Math.min(...xs) + 420) / innerWidth, (Math.max(...zs) - Math.min(...zs) + 260) / innerHeight);
    }
    this.el.classList.add('on'); this.el.classList.toggle('reise', modus === 'reise');
    this.el.querySelector('h2').textContent = modus === 'reise' ? 'WOHIN WILLST DU?' : 'KARTE';
    this.el.querySelector('.kopf span').textContent = modus === 'reise'
      ? (hier ? 'DU BIST: ' + hier.toUpperCase() + ' · HALTESTELLE WÄHLEN' : 'HALTESTELLE WÄHLEN')
      : 'NUR WAS DU GESCANNT HAST · ' + Math.round(this.anteil() * 100) + ' % ERKUNDET';
    this.el.querySelector('.fuss').textContent = modus === 'reise' ? 'KLICKEN = HINFAHREN · ESC = ZURÜCK' : 'ZIEHEN = VERSCHIEBEN · MAUSRAD = ZOOM · M / ESC = SCHLIESSEN';
    const li = this.el.querySelector('.liste'); li.innerHTML = '';
    for (const st of this.stat.stationen) {
      const b = document.createElement('button'); b.innerHTML = `${st.name.toUpperCase()}<small>${st.linien}</small>`;
      if (st.name === hier) b.classList.add('hier');
      else b.addEventListener('click', () => this.reisen(st));
      b.addEventListener('mouseenter', () => { this._hover = st.name; this.zeichneGross(); });
      b.addEventListener('mouseleave', () => { this._hover = null; this.zeichneGross(); });
      li.append(b);
    }
    this.zeichneGross();
  }
  schliessen(grund = 'zu') {
    if (!this.offen) return;
    this.offen = false; this.el.classList.remove('on');
    if (this.onClose) this.onClose(grund, this.modus);
  }
  reisen(st) {
    this.blende.textContent = st.name.toUpperCase(); this.blende.classList.add('on');
    this.schliessen('reise');
    setTimeout(() => { if (this.onTravel) this.onTravel(st); setTimeout(() => this.blende.classList.remove('on'), 900); }, 650);
  }
  anteil() {
    try {
      const s = document.createElement('canvas'); s.width = 182; s.height = 202; const c = s.getContext('2d'); c.drawImage(this.mask, 0, 0, 182, 202);
      const d = c.getImageData(0, 0, 182, 202).data; let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] > 100; return n / (182 * 202) / 0.62;   // Stadtkreis ≈ 62 % des Rechtecks
    } catch { return 0; }
  }
  bindEvents() {
    let drag = null;
    this.el.addEventListener('pointerdown', (e) => { if (e.target.closest('button')) return; drag = { x: e.clientX, y: e.clientY, cx: this.view.cx, cz: this.view.cz, moved: false }; this.el.classList.add('drag'); });
    addEventListener('pointermove', (e) => {
      if (!this.offen) return;
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
        this.view.cx = drag.cx - dx * this.view.mpp; this.view.cz = drag.cz - dy * this.view.mpp; this.zeichneGross(); return;
      }
      const h = this._treffer(e.clientX, e.clientY); const nm = h ? h.st.name : null;
      if (nm !== this._hover) { this._hover = nm; this.el.querySelectorAll('.liste button').forEach((b, i) => b.classList.toggle('hl', this.stat.stationen[i].name === nm)); this.zeichneGross(); }
      this.el.style.cursor = h && this.modus === 'reise' ? 'pointer' : '';
    });
    addEventListener('pointerup', (e) => {
      if (!drag) return; const d = drag; drag = null; this.el.classList.remove('drag');
      if (!d.moved && this.modus === 'reise') { const h = this._treffer(e.clientX, e.clientY); if (h && h.st.name !== this.hier) this.reisen(h.st); }
    });
    this.el.addEventListener('wheel', (e) => {
      e.preventDefault(); const k = Math.exp(e.deltaY * 0.0015), v = this.view;
      const mx = (e.clientX - innerWidth / 2) * v.mpp, mz = (e.clientY - innerHeight / 2) * v.mpp;
      const n = Math.min(3.5, Math.max(0.12, v.mpp * k)); v.cx += mx * (1 - n / v.mpp); v.cz += mz * (1 - n / v.mpp); v.mpp = n; this.zeichneGross();
    }, { passive: false });
    addEventListener('resize', () => { if (this.offen) this.zeichneGross(); });
  }
  _treffer(x, y) { return (this._ziele || []).find((z) => Math.hypot(z.u - x, z.y - y) < 18) || null; }

  // ---------- Auslöser: Bahn-Eingänge und Passerelle-Ende ----------
  pruefeZonen(pos) {
    if (!this.ready || this.offen) return null;
    for (const e of this.stat.eingaenge) {
      const dx = pos.x - e.top[0], dz = pos.z - e.top[1];
      const s = dx * e.dir[0] + dz * e.dir[1], t = dx * -e.dir[1] + dz * e.dir[0];
      if (s > 2.0 && s < 9 && Math.abs(t) < 3.2 && pos.y < e.y_top - 0.7) return { art: 'eingang', e, station: e.station };
    }
    const P = this.stat.passerelle, dx = pos.x - P.origin[0], dz = pos.z - P.origin[1];
    const s = dx * P.axis[0] + dz * P.axis[1], t = dx * P.normal[0] + dz * P.normal[1];
    if (s > P.s_ende - 7 && s < P.s_ende + 6 && Math.abs(t) < P.breite && pos.y < -2.5) return { art: 'passerelle', station: 'Hauptbahnhof' };
    return null;
  }

  update(dt, player, camYaw) {
    if (!this.ready) return null;
    this.player = player;
    if ((this._nahT = (this._nahT || 0) - dt) <= 0) { this._nahT = 0.25; this.nah(player.pos); this._nahN = (this._nahN || 0) + 1; if (this._nahN % 40 === 0) this.speichern(); }
    if ((this._mini -= dt) <= 0) { this._mini = 1 / 20; this.minimap(player, camYaw); }
    this.sperre = Math.max(0, this.sperre - dt);
    const z = this.pruefeZonen(player.pos);
    if (z && !this.zone && this.sperre <= 0) { this.zone = z; return z; }
    if (!z) this.zone = null;
    return null;
  }
}

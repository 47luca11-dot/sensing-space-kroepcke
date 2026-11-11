import * as THREE from 'three';

export type MaterialPresetId = 'holz' | 'marmor' | 'metall' | 'beton' | 'lack';

interface MaterialPreset {
  label: string;
  color: string;
  roughness: number;
  metalness: number;
  /** Erzeugt eine Graustufen-Textur; der Farbton kommt aus `color`. */
  texture: ((size: number) => HTMLCanvasElement) | null;
}

export const PRESETS: Record<MaterialPresetId, MaterialPreset> = {
  holz: { label: 'Holz', color: '#a8744a', roughness: 0.6, metalness: 0, texture: woodTexture },
  marmor: { label: 'Marmor', color: '#f1eee8', roughness: 0.15, metalness: 0, texture: marbleTexture },
  metall: { label: 'Metall (gebürstet)', color: '#c4c8ce', roughness: 0.35, metalness: 1, texture: brushedTexture },
  beton: { label: 'Beton', color: '#9a968f', roughness: 0.9, metalness: 0, texture: concreteTexture },
  lack: { label: 'Lack (einfarbig)', color: '#e9e3d8', roughness: 0.3, metalness: 0, texture: null },
};

// --- Prozedurale Texturen (keine externen Dateien nötig) ------------------------------

function makeNoise(seed: number) {
  const perm = new Uint8Array(512);
  let s = seed;
  for (let i = 0; i < 256; i++) perm[i] = i;
  for (let i = 255; i > 0; i--) {
    s = (s * 16807) % 2147483647;
    const j = s % (i + 1);
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  for (let i = 0; i < 256; i++) perm[i + 256] = perm[i];
  const wrap = (i: number, p: number) => ((i % p) + p) % p;
  const hash = (x: number, y: number) => perm[perm[x & 255] + (y & 255)] / 255;
  const fade = (t: number) => t * t * (3 - 2 * t);
  // Periodisches Value-Noise: wiederholt sich nach px × py Zellen, damit die Textur nahtlos kachelt.
  const noise = (x: number, y: number, px: number, py: number) => {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = fade(x - xi), yf = fade(y - yi);
    const x0 = wrap(xi, px), x1 = wrap(xi + 1, px), y0 = wrap(yi, py), y1 = wrap(yi + 1, py);
    const a = hash(x0, y0), b = hash(x1, y0), c = hash(x0, y1), d = hash(x1, y1);
    return a + (b - a) * xf + (c - a) * yf + (a - b - c + d) * xf * yf;
  };
  /** Fraktales Noise über u, v ∈ [0, 1) mit ganzzahligen Grundfrequenzen fx, fy (kachelbar). */
  return (u: number, v: number, fx: number, fy: number, octaves = 4) => {
    let sum = 0, amp = 0.5, freq = 1;
    for (let o = 0; o < octaves; o++) {
      sum += amp * noise(u * fx * freq, v * fy * freq, fx * freq, fy * freq);
      amp *= 0.5;
      freq *= 2;
    }
    return sum;
  };
}

function paint(size: number, fn: (u: number, v: number) => number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const g = Math.round(255 * Math.min(1, Math.max(0, fn(x / size, y / size))));
      const k = (y * size + x) * 4;
      img.data[k] = img.data[k + 1] = img.data[k + 2] = g;
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Sinus-Terme nutzen ganzzahlige Perioden über [0, 1), damit auch sie nahtlos kacheln.
function woodTexture(size: number) {
  const fbm = makeNoise(7);
  return paint(size, (u, v) => {
    // Lange, leicht wellige Jahresringe entlang u, dazu Längsstreifen und feine Fasern.
    const warp = fbm(u, v, 1, 6, 5) * 5;
    const rings = 0.5 + 0.5 * Math.sin((v * 14 + warp) * 2 * Math.PI);
    const streaks = fbm(u, v, 3, 48, 4);
    const fiber = fbm(u, v, 64, 256, 2);
    return 0.72 + 0.14 * Math.pow(rings, 6) - 0.1 * Math.pow(1 - rings, 8) + 0.22 * (streaks - 0.5) + 0.08 * (fiber - 0.5);
  });
}

function marbleTexture(size: number) {
  const fbm = makeNoise(13);
  return paint(size, (u, v) => {
    const turb = fbm(u, v, 3, 3, 6);
    const vein = Math.abs(Math.sin((u * 2 + v + turb * 3) * Math.PI));
    const cloud = fbm(u, v, 6, 6, 5);
    return 0.95 - 0.2 * Math.pow(1 - vein, 14) - 0.1 * Math.pow(1 - vein, 3) - 0.08 * cloud;
  });
}

function brushedTexture(size: number) {
  const fbm = makeNoise(21);
  return paint(size, (u, v) => 0.82 + 0.12 * fbm(u, v, 2, 256, 3));
}

function concreteTexture(size: number) {
  const fbm = makeNoise(5);
  return paint(size, (u, v) => {
    const pores = fbm(u, v, 64, 64, 2) > 0.72 ? -0.15 : 0;
    return 0.72 + 0.25 * fbm(u, v, 8, 8, 5) + pores;
  });
}

const textureCache = new Map<MaterialPresetId, THREE.CanvasTexture | null>();
function presetTexture(id: MaterialPresetId): THREE.CanvasTexture | null {
  if (!textureCache.has(id)) {
    const make = PRESETS[id].texture;
    const tex = make ? new THREE.CanvasTexture(make(512)) : null;
    if (tex) {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      tex.anisotropy = 8;
    }
    textureCache.set(id, tex);
  }
  return textureCache.get(id)!;
}

/** Ein bearbeitbares PBR-Material samt kleiner Bedienoberfläche. */
export class MaterialEditor {
  readonly material = new THREE.MeshStandardMaterial();
  private preset: MaterialPresetId;
  /** Texturwiederholungen pro Szeneneinheit, damit die Maserung unabhängig von der Größe bleibt. */
  private repeatPerUnit = 1;
  private size = new THREE.Vector2(1, 1);

  constructor(initial: MaterialPresetId) {
    this.preset = initial;
    this.applyPreset(initial);
  }

  /** Passt die Texturwiederholung an die Größe der Fläche an (in Szeneneinheiten). */
  setSurfaceSize(width: number, depth: number, repeatPerUnit: number) {
    this.size.set(width, depth);
    this.repeatPerUnit = repeatPerUnit;
    this.updateRepeat();
  }

  private updateRepeat() {
    const map = this.material.map;
    if (map) map.repeat.set(this.size.x * this.repeatPerUnit, this.size.y * this.repeatPerUnit);
  }

  applyPreset(id: MaterialPresetId) {
    this.preset = id;
    const p = PRESETS[id];
    const tex = presetTexture(id);
    // Jede Fläche bekommt ihre eigene Texturkopie, damit "repeat" unabhängig bleibt.
    this.material.map = tex ? tex.clone() : null;
    if (this.material.map) this.material.map.needsUpdate = true;
    this.material.color.set(p.color);
    this.material.roughness = p.roughness;
    this.material.metalness = p.metalness;
    this.material.needsUpdate = true;
    this.updateRepeat();
  }

  /** Baut Auswahl, Farbe, Rauheit und Metallisch in `container`. */
  mount(container: HTMLElement) {
    container.innerHTML = '';
    const select = document.createElement('select');
    for (const [id, p] of Object.entries(PRESETS)) select.add(new Option(p.label, id));
    select.value = this.preset;

    const color = document.createElement('input');
    color.type = 'color';
    const rough = slider(0, 1, 0.01);
    const metal = slider(0, 1, 0.01);

    const sync = () => {
      color.value = `#${this.material.color.getHexString()}`;
      rough.input.value = String(this.material.roughness);
      metal.input.value = String(this.material.metalness);
      rough.out.textContent = this.material.roughness.toFixed(2);
      metal.out.textContent = this.material.metalness.toFixed(2);
    };

    select.addEventListener('change', () => {
      this.applyPreset(select.value as MaterialPresetId);
      sync();
    });
    color.addEventListener('input', () => this.material.color.set(color.value));
    rough.input.addEventListener('input', () => {
      this.material.roughness = Number(rough.input.value);
      sync();
    });
    metal.input.addEventListener('input', () => {
      this.material.metalness = Number(metal.input.value);
      sync();
    });

    const row = document.createElement('div');
    row.className = 'row';
    row.append(labelled('Material', select), labelled('Farbe', color));
    container.append(row, rough.wrap('Rauheit'), metal.wrap('Metallisch'));
    sync();
  }
}

// --- kleine DOM-Helfer ------------------------------------------------------------------

export function slider(min: number, max: number, step: number) {
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  const out = document.createElement('output');
  return {
    input,
    out,
    wrap(text: string) {
      const label = document.createElement('label');
      label.className = 'slider';
      label.append(`${text} `, out, input);
      return label;
    },
  };
}

export function labelled(text: string, el: HTMLElement) {
  const label = document.createElement('label');
  label.className = 'inline';
  label.append(`${text} `, el);
  return label;
}

import * as THREE from 'three';
import { MaterialEditor, labelled, slider } from './materials';

/** Wo das (freigestellte) Splat-Objekt steht, in Weltkoordinaten (Y = oben). */
export interface Placement {
  /** Mittelpunkt der Standfläche (x, z) bzw. Unterkante (y). */
  base: THREE.Vector3;
  /** Horizontaler Radius des Objekts. */
  footprint: number;
  /** Höhe des Objekts. */
  height: number;
}

/** Ein zuschaltbares 3D-Element neben den Splats. */
export interface SceneComponent {
  id: string;
  label: string;
  enabled: boolean;
  object: THREE.Object3D;
  /** Baut die Einstellungen der Komponente in `container`. */
  mount(container: HTMLElement, onChange: () => void): void;
  /** Richtet die Komponente am Objekt aus. */
  place(p: Placement, components: SceneComponent[]): void;
}

/** Maßstab, damit Möbel bei metrischen Scans (Flasche ~0,2 m) realistisch groß sind,
 *  bei Szenen in anderen Einheiten aber mitwachsen. */
function sceneScale(p: Placement) {
  return Math.max(1, p.height / 0.4, p.footprint / 0.15);
}

// --- Tisch ---------------------------------------------------------------------------

type TableShape = 'eckig' | 'rund';

export class TableComponent implements SceneComponent {
  id = 'tisch';
  label = 'Tisch';
  enabled = false;
  object = new THREE.Group();
  /** Oberkante der Tischplatte, damit der Boden darunter liegen kann. */
  floorY = 0;

  private top = new MaterialEditor('holz');
  private legs = new MaterialEditor('metall');
  private shape: TableShape = 'eckig';
  private sizeFactor = 1;
  private showShadow = true;
  private shadow: THREE.Mesh;
  private last: Placement | null = null;

  constructor() {
    this.legs.material.color.set('#2b2b2e');
    this.legs.material.roughness = 0.45;
    this.shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: contactShadowTexture(), transparent: true, depthWrite: false }),
    );
    this.shadow.renderOrder = 1;
  }

  mount(container: HTMLElement, onChange: () => void) {
    const shape = document.createElement('select');
    shape.add(new Option('Eckig', 'eckig'));
    shape.add(new Option('Rund', 'rund'));
    shape.value = this.shape;
    shape.addEventListener('change', () => {
      this.shape = shape.value as TableShape;
      onChange();
    });

    const size = slider(0.5, 2, 0.05);
    size.input.value = String(this.sizeFactor);
    size.out.textContent = `${this.sizeFactor.toFixed(2)}×`;
    size.input.addEventListener('input', () => {
      this.sizeFactor = Number(size.input.value);
      size.out.textContent = `${this.sizeFactor.toFixed(2)}×`;
      onChange();
    });

    const shadow = document.createElement('input');
    shadow.type = 'checkbox';
    shadow.checked = this.showShadow;
    shadow.addEventListener('change', () => {
      this.showShadow = shadow.checked;
      onChange();
    });

    const row = document.createElement('div');
    row.className = 'row';
    row.append(labelled('Form', shape), labelled('Kontaktschatten', shadow));

    const topBox = document.createElement('div');
    const legBox = document.createElement('div');
    this.top.mount(topBox);
    this.legs.mount(legBox);
    container.append(row, size.wrap('Größe'), heading('Tischplatte'), topBox, heading('Beine / Gestell'), legBox);
  }

  place(p: Placement) {
    this.last = p;
    for (const child of [...this.object.children]) {
      if (child !== this.shadow) (child as THREE.Mesh).geometry.dispose();
    }
    this.object.clear();
    const s = sceneScale(p) * this.sizeFactor;
    const height = 0.74 * sceneScale(p);
    const thick = 0.035 * s;
    const topY = p.base.y;
    this.floorY = topY - height;

    if (this.shape === 'eckig') {
      const w = 1.2 * s, d = 0.75 * s;
      this.top.setSurfaceSize(w, d, 1 / sceneScale(p));
      const top = new THREE.Mesh(new THREE.BoxGeometry(w, thick, d), this.top.material);
      top.position.set(0, -thick / 2, 0);
      this.object.add(top);
      const legW = 0.045 * s, legH = height - thick;
      for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(legW, legH, legW), this.legs.material);
        leg.position.set(sx * (w / 2 - legW * 1.5), -thick - legH / 2, sz * (d / 2 - legW * 1.5));
        this.object.add(leg);
      }
    } else {
      const r = 0.45 * s;
      this.top.setSurfaceSize(2 * r, 2 * r, 1 / sceneScale(p));
      const top = new THREE.Mesh(new THREE.CylinderGeometry(r, r, thick, 96), this.top.material);
      top.position.y = -thick / 2;
      const columnH = height - thick - 0.02 * s;
      const column = new THREE.Mesh(new THREE.CylinderGeometry(0.04 * s, 0.05 * s, columnH, 32), this.legs.material);
      column.position.y = -thick - columnH / 2;
      const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.25 * s, 0.27 * s, 0.02 * s, 64), this.legs.material);
      foot.position.y = -height + 0.01 * s;
      this.object.add(top, column, foot);
    }

    if (this.showShadow) {
      const size = p.footprint * 3.2;
      this.shadow.scale.set(size, 1, size);
      this.shadow.position.y = 0.0005 * sceneScale(p);
      this.object.add(this.shadow);
    }
    this.object.position.set(p.base.x, topY, p.base.z);
  }

  /** Neu aufbauen mit den letzten Platzierungsdaten (nach UI-Änderungen). */
  rebuild() {
    if (this.last) this.place(this.last);
  }
}

// --- Boden ---------------------------------------------------------------------------

export class FloorComponent implements SceneComponent {
  id = 'boden';
  label = 'Boden';
  enabled = false;
  object: THREE.Mesh;
  private editor = new MaterialEditor('beton');

  constructor() {
    this.object = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), this.editor.material);
  }

  mount(container: HTMLElement) {
    const box = document.createElement('div');
    this.editor.mount(box);
    container.append(box);
  }

  place(p: Placement, components: SceneComponent[]) {
    const table = components.find((c): c is TableComponent => c instanceof TableComponent);
    const y = table?.enabled ? table.floorY : p.base.y;
    const size = 12 * sceneScale(p);
    this.object.scale.set(size, 1, size);
    this.editor.setSurfaceSize(size, size, 0.5 / sceneScale(p));
    this.object.position.set(p.base.x, y, p.base.z);
  }
}

// --- Helfer --------------------------------------------------------------------------

function heading(text: string) {
  const h = document.createElement('div');
  h.className = 'subheading';
  h.textContent = text;
  return h;
}

/** Weicher, radialer Schatten unter dem Objekt – Splats werfen selbst keine Schatten. */
function contactShadowTexture() {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(0,0,0,0.6)');
  g.addColorStop(0.35, 'rgba(0,0,0,0.35)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

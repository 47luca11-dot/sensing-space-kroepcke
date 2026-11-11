import type { ObjectConfig, ObjectKind } from '../config/types';
import type { GameObject } from '../objects/GameObject';

const SPLAT_EXTENSIONS = ['spz', 'ply', 'splat', 'ksplat'];
const MESH_EXTENSIONS = ['glb'];

type Fit = 'hull' | 'box' | 'cylinder' | 'sphere';

export interface UploadCallbacks {
  /** Lädt das Objekt und stellt es vor den Spieler. Wirft bei einer unlesbaren Datei. */
  spawn(config: ObjectConfig): Promise<GameObject>;
  remove(obj: GameObject): void;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/**
 * Upload-Feld im Start- und Pausenbildschirm: eigene Scans auswählen oder hineinziehen,
 * sie erscheinen sofort als interaktive Gegenstände im Raum. Gilt nur für die laufende Sitzung.
 */
export class AssetUpload {
  private readonly input = $<HTMLInputElement>('upload-input');
  private readonly dropzone = $('upload-dropzone');
  private readonly list = $('upload-list');
  private readonly kind = $<HTMLSelectElement>('upload-kind');
  private readonly fit = $<HTMLSelectElement>('upload-fit');
  private readonly scale = $<HTMLInputElement>('upload-scale');
  private readonly flip = $<HTMLInputElement>('upload-flip');
  private counter = 0;
  private ready = false;
  private readonly queue: File[] = [];

  constructor(private readonly callbacks: UploadCallbacks) {
    this.input.addEventListener('change', () => {
      this.addFiles([...(this.input.files ?? [])]);
      this.input.value = '';
    });

    // Ohne preventDefault öffnet der Browser eine fallengelassene Datei als neue Seite.
    for (const type of ['dragover', 'drop'] as const) window.addEventListener(type, (e) => e.preventDefault());
    this.dropzone.addEventListener('dragenter', () => this.dropzone.classList.add('dragging'));
    this.dropzone.addEventListener('dragleave', () => this.dropzone.classList.remove('dragging'));
    this.dropzone.addEventListener('drop', (e) => {
      this.dropzone.classList.remove('dragging');
      this.addFiles([...(e.dataTransfer?.files ?? [])]);
    });
  }

  /** Erst wenn Physik und Raum stehen, werden Uploads gespawnt; vorher gewählte Dateien warten. */
  setReady() {
    this.ready = true;
    for (const file of this.queue.splice(0)) void this.load(file);
  }

  private addFiles(files: File[]) {
    for (const file of files) {
      if (this.ready) void this.load(file);
      else this.queue.push(file);
    }
  }

  private async load(file: File) {
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    const isSplat = SPLAT_EXTENSIONS.includes(ext);
    const row = this.addRow(file.name);
    if (!isSplat && !MESH_EXTENSIONS.includes(ext)) {
      row.setStatus('Format nicht unterstützt', true);
      return;
    }

    const name = file.name.replace(/\.[^.]+$/, '');
    const config: ObjectConfig = {
      id: `upload-${++this.counter}`,
      label: name,
      kind: this.kind.value as ObjectKind,
      model: isSplat
        ? { type: 'splat', url: file.name, file, flip: this.flip.checked }
        : { type: 'gltf', url: file.name, file },
      collider: { shape: 'auto', fit: this.fit.value as Fit },
      position: [0, 0, 0],
      scale: Math.max(0.01, Number(this.scale.value) || 1),
      mass: 1,
    };

    row.setStatus(`lädt … (${(file.size / 1e6).toFixed(1)} MB)`);
    try {
      const obj = await this.callbacks.spawn(config);
      row.setStatus('im Raum');
      row.onRemove(() => this.callbacks.remove(obj));
    } catch (error) {
      console.error(error);
      row.setStatus('konnte nicht geladen werden', true);
    }
  }

  private addRow(fileName: string) {
    const li = document.createElement('li');
    const name = document.createElement('span');
    name.className = 'upload-name';
    name.textContent = fileName;
    const status = document.createElement('span');
    status.className = 'upload-status';
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'link';
    remove.textContent = 'Entfernen';
    remove.hidden = true;
    li.append(name, status, remove);
    this.list.append(li);
    return {
      setStatus(text: string, error = false) {
        status.textContent = text;
        status.classList.toggle('error', error);
      },
      onRemove(fn: () => void) {
        remove.hidden = false;
        remove.addEventListener('click', () => {
          fn();
          li.remove();
        });
      },
    };
  }
}

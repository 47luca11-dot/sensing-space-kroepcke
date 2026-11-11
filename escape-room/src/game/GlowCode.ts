import * as THREE from 'three';
import { PUZZLE } from '../config/puzzle';
import type { Room } from '../room/Room';

/**
 * Nachleuchtende Ziffern an der Wand: Bei Licht unsichtbar, im Dunkeln blenden sie langsam ein,
 * wie Leuchtfarbe, die sich erst an die Dunkelheit „gewöhnen“ muss.
 */
export class GlowCode {
  private readonly material: THREE.MeshBasicMaterial;
  private readonly mesh: THREE.Mesh;
  private visibility = 0;

  constructor(
    scene: THREE.Scene,
    private readonly room: Room,
  ) {
    const { position, rotationY, width, color } = PUZZLE.glowCode;
    const texture = codeTexture(PUZZLE.safeCode, color);
    this.material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    });
    const height = width * (texture.image.height / texture.image.width);
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), this.material);
    this.mesh.position.set(...position);
    this.mesh.rotation.y = THREE.MathUtils.degToRad(rotationY);
    this.mesh.visible = false;
    scene.add(this.mesh);
  }

  update(dt: number) {
    const dark = !this.room.isLightOn(PUZZLE.glowCode.light);
    // Einblenden langsam (ca. 2 s), Ausblenden schnell
    this.visibility = THREE.MathUtils.damp(this.visibility, dark ? 1 : 0, dark ? 1.4 : 8, dt);
    this.material.opacity = this.visibility * (0.85 + Math.sin(performance.now() / 900) * 0.05);
    this.mesh.visible = this.visibility > 0.01;
  }
}

/** Handgemalt wirkende Ziffern mit weichem Leuchten. */
function codeTexture(code: string, color: string) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 200;
  const g = canvas.getContext('2d')!;
  g.font = 'bold 150px "Segoe Print", "Bradley Hand", cursive';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  const step = 512 / code.length;
  for (const blur of [26, 10, 0]) {
    g.shadowColor = color;
    g.shadowBlur = blur;
    g.fillStyle = blur ? 'rgba(184, 255, 158, 0.55)' : color;
    [...code].forEach((digit, i) => {
      g.save();
      g.translate(step * (i + 0.5), 100 + (i % 2 ? 6 : -4));
      g.rotate((i % 2 ? 1 : -1) * 0.06);
      g.fillText(digit, 0, 0);
      g.restore();
    });
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

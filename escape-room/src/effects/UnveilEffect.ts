import * as THREE from 'three';
import { dyno, type SplatMesh } from '@sparkjsdev/spark';
import { UNVEIL } from '../config/effects';

interface Instance {
  mesh: SplatMesh;
  progress: ReturnType<typeof dyno.dynoFloat>;
  range: ReturnType<typeof dyno.dynoFloat>;
  /** Reichweite (automatisch) in Rohkoordinaten des Splats */
  autoRange: number;
  /** Meter → Rohkoordinaten (SplatMesh kann skaliert sein) */
  metersToRaw: number;
  t: number;
  playing: boolean;
}

/**
 * „Unveil“-Effekt für Gaussian Splats, komplett auf der GPU (Spark-Dyno):
 * Vom Startpunkt aus breitet sich eine Kugel aus. Außerhalb ist nur eine lichte Punktwolke zu sehen,
 * an der Grenze läuft eine leuchtende Regenbogenfront, innerhalb erscheint der Scan vollständig.
 */
export class UnveilEffects {
  readonly settings = { ...UNVEIL };
  private readonly instances: Instance[] = [];
  /** Gemeinsame Uniforms für alle Splat-Objekte */
  private readonly time = dyno.dynoFloat(0);
  private readonly rainbow = dyno.dynoFloat(UNVEIL.rainbow ? 1 : 0);
  private readonly blend = dyno.dynoFloat(UNVEIL.blendRange);
  private readonly pointSize = dyno.dynoFloat(UNVEIL.pointSize);
  private readonly density = dyno.dynoFloat(UNVEIL.pointDensity);
  private manualProgress: number | null = null;

  /** Effekt an ein Splat-Objekt hängen. `bounds` = Grenzen im Elternraum des Meshes (wie LoadedModel.bounds). */
  attach(mesh: SplatMesh, bounds: THREE.Box3, playNow: boolean) {
    if (!this.settings.enabled) return;
    mesh.updateMatrix();
    const toRaw = mesh.matrix.clone().invert();
    const originParent = bounds.getCenter(new THREE.Vector3());
    if (this.settings.origin === 'bottom') originParent.y = bounds.min.y;
    const origin = originParent.clone().applyMatrix4(toRaw);
    // Entferntester Punkt der Box → automatische Reichweite
    const corners = [bounds.min, bounds.max].flatMap((a) => [bounds.min, bounds.max].flatMap((b) => [bounds.min, bounds.max].map((c) => new THREE.Vector3(a.x, b.y, c.z))));
    const metersToRaw = 1 / mesh.scale.x;
    const autoRange = Math.max(...corners.map((c) => c.distanceTo(originParent))) * metersToRaw;

    const instance: Instance = {
      mesh,
      progress: dyno.dynoFloat(0),
      range: dyno.dynoFloat(autoRange),
      autoRange,
      metersToRaw,
      t: 0,
      playing: playNow,
    };
    const originU = dyno.dynoVec3(origin);
    const metersToRawU = dyno.dynoFloat(metersToRaw);

    const shader = new dyno.Dyno({
      inTypes: {
        gsplat: dyno.Gsplat,
        progress: 'float',
        range: 'float',
        origin: 'vec3',
        toRaw: 'float',
        time: 'float',
        rainbow: 'float',
        blend: 'float',
        pointSize: 'float',
        density: 'float',
      },
      outTypes: { gsplat: dyno.Gsplat },
      globals: () => [
        dyno.unindent(`
          vec3 unveilHue(float h) {
            vec3 p = abs(fract(vec3(h) + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
            return clamp(p - 1.0, 0.0, 1.0);
          }
        `),
      ],
      statements: ({ inputs, outputs }) =>
        dyno.unindentLines(`
          ${outputs.gsplat} = ${inputs.gsplat};
          float dist = length(${inputs.gsplat}.center - ${inputs.origin});
          float blend = max(${inputs.blend} * ${inputs.toRaw}, 1e-4);
          float front = ${inputs.progress} * (${inputs.range} + blend);
          float ahead = dist - front;
          // 0 = noch verdeckt, 1 = enthüllt
          float shown = 1.0 - smoothstep(-blend, 0.0, ahead);
          if (shown < 1.0) {
            // Verdeckter Bereich: nur ein Teil der Splats, stark verkleinert → lichte Punktwolke
            float pick = step(1.0 - ${inputs.density}, fract(float(${inputs.gsplat}.index) * 0.61803398875));
            vec3 point = min(${inputs.gsplat}.scales, vec3(${inputs.pointSize} * ${inputs.toRaw}));
            ${outputs.gsplat}.scales = mix(point, ${inputs.gsplat}.scales, shown);
            ${outputs.gsplat}.rgba.a = ${inputs.gsplat}.rgba.a * mix(pick * 0.9, 1.0, shown);
          }
          if (${inputs.rainbow} > 0.5 && ${inputs.progress} < 1.0) {
            float glow = 1.0 - smoothstep(0.0, blend * 1.5, abs(ahead + blend * 0.5));
            vec3 rainbow = unveilHue(fract(dist / (blend * 6.0) - ${inputs.time} * 0.6));
            ${outputs.gsplat}.rgba.rgb = mix(${outputs.gsplat}.rgba.rgb, rainbow * 1.4, glow * 0.85);
            ${outputs.gsplat}.rgba.a = max(${outputs.gsplat}.rgba.a, glow * ${inputs.gsplat}.rgba.a);
          }
        `),
    });

    mesh.objectModifier = dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => ({
      gsplat: shader.apply({
        gsplat,
        progress: instance.progress,
        range: instance.range,
        origin: originU,
        toRaw: metersToRawU,
        time: this.time,
        rainbow: this.rainbow,
        blend: this.blend,
        pointSize: this.pointSize,
        density: this.density,
      }).gsplat,
    }));
    mesh.updateGenerator();
    this.instances.push(instance);
  }

  /** Alle Objekte von vorn enthüllen. */
  playAll() {
    this.manualProgress = null;
    for (const inst of this.instances) {
      inst.t = 0;
      inst.playing = true;
    }
  }

  /** Fortschritt von Hand setzen (Panel); null = wieder abspielen lassen. */
  setProgress(value: number | null) {
    this.manualProgress = value;
  }

  get currentProgress() {
    const inst = this.instances.find((i) => i.playing) ?? this.instances[0];
    return inst ? inst.progress.value : 1;
  }

  /** Geänderte Einstellungen in die Uniforms übernehmen. */
  applySettings() {
    this.rainbow.value = this.settings.rainbow ? 1 : 0;
    this.blend.value = this.settings.blendRange;
    this.pointSize.value = this.settings.pointSize;
    this.density.value = this.settings.pointDensity;
    for (const inst of this.instances) {
      inst.range.value = this.settings.range > 0 ? this.settings.range * inst.metersToRaw : inst.autoRange;
      inst.mesh.updateVersion();
    }
  }

  update(dt: number) {
    this.time.value += dt;
    for (const inst of this.instances) {
      const before = inst.progress.value;
      if (this.manualProgress !== null) {
        inst.progress.value = this.manualProgress;
      } else if (inst.playing) {
        inst.t += dt;
        inst.progress.value = Math.min(1, inst.t / Math.max(0.1, this.settings.duration));
        if (inst.progress.value >= 1) inst.playing = false;
      }
      // Neu berechnen, wenn sich der Fortschritt ändert oder die Regenbogenfront gerade läuft
      const animating = inst.progress.value > 0 && inst.progress.value < 1;
      if (inst.progress.value !== before || animating) inst.mesh.updateVersion();
    }
  }
}

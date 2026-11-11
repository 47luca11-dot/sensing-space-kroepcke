import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import { HANDS, type FingerName } from '../config/hands';
import { assetUrl, fetchBytes } from '../objects/assets';
import { applySkin } from './skin';

export const FINGERS: FingerName[] = ['thumb', 'index', 'middle', 'ring', 'pinky'];

/** Krümmung je Finger, 0 = gestreckt, 1 = ganz gebeugt. */
export type HandPose = Record<FingerName, number>;

/**
 * Eine Hand in einem einheitlichen Koordinatensystem, egal wie das Modell exportiert wurde:
 * Handgelenk im Ursprung, Finger zeigen nach −Z, Handrücken nach +Y, Daumen (rechte Hand) nach −X.
 * Die linke Hand ist gespiegelt, ihr Daumen zeigt nach +X. Ob das Modell selbst eine linke oder
 * rechte Hand ist, wird beim Laden erkannt.
 */
export class HandRig {
  readonly object = new THREE.Group();
  private readonly fingers: Record<FingerName, { bone: THREE.Bone; rest: THREE.Quaternion }[]>;
  private readonly current: HandPose = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
  private readonly target: HandPose = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
  private readonly axis = new THREE.Vector3(...HANDS.flexAxis);
  private readonly oppositionAxis = new THREE.Vector3(...HANDS.thumbOppositionAxis);
  private readonly tmp = new THREE.Quaternion();

  private constructor(model: THREE.Object3D, readonly side: 'left' | 'right', modelIsLeft: boolean) {
    this.object.name = `hand-${side}`;
    this.object.add(model);
    // Gespiegelt wird, wenn Modell und gewünschte Seite nicht zusammenpassen.
    if ((side === 'left') !== modelIsLeft) this.object.scale.x = -1;

    const byName = (name: string) => {
      const bone = model.getObjectByName(name);
      if (!(bone as THREE.Bone)?.isBone) throw new Error(`Handknochen „${name}“ fehlt im Modell`);
      return bone as THREE.Bone;
    };
    this.fingers = Object.fromEntries(
      FINGERS.map((f) => [f, HANDS.fingers[f].map((n) => ({ bone: byName(n), rest: byName(n).quaternion.clone() }))]),
    ) as HandRig['fingers'];
  }

  /** Lädt das Modell einmal und erzeugt beide Hände. */
  static async load(): Promise<{ right: HandRig; left: HandRig }> {
    const gltf = await new GLTFLoader().parseAsync(await fetchBytes(assetUrl(HANDS.url)), '');
    const model = gltf.scene;
    const { isLeft } = normalizeFrame(model);
    model.traverse((child) => {
      const mesh = child as THREE.SkinnedMesh;
      if (!mesh.isSkinnedMesh) return;
      mesh.frustumCulled = false; // Die Bounding Box des Rigs passt nach dem Posieren nicht mehr.
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      applySkin(mesh);
    });
    return { right: new HandRig(model, 'right', isLeft), left: new HandRig(cloneSkinned(model), 'left', isLeft) };
  }

  /** Zielhaltung, wird in `update` weich angefahren. */
  setPose(pose: HandPose) {
    Object.assign(this.target, pose);
  }

  /** Haltung sofort übernehmen (z. B. zum Ausmessen von Fingerspitzen). */
  snapPose(pose: HandPose) {
    Object.assign(this.target, pose);
    Object.assign(this.current, pose);
    this.applyBones();
  }

  update(dt: number) {
    for (const f of FINGERS) this.current[f] = THREE.MathUtils.damp(this.current[f], this.target[f], 14, dt);
    this.applyBones();
  }

  /** Position eines Knochens im Hand-Koordinatensystem (aktuelle Haltung). */
  boneLocal(name: string, out = new THREE.Vector3()) {
    this.object.updateMatrixWorld(true);
    const bone = this.object.getObjectByName(name)!;
    return out.copy(this.object.worldToLocal(bone.getWorldPosition(new THREE.Vector3())));
  }

  /** Position der Fingerspitze im Hand-Koordinatensystem (aktuelle Haltung). */
  tipLocal(finger: FingerName, out = new THREE.Vector3()) {
    const bones = this.fingers[finger];
    const last = bones[bones.length - 1].bone;
    const prev = bones[bones.length - 2].bone;
    this.object.updateMatrixWorld(true);
    const a = prev.getWorldPosition(new THREE.Vector3());
    const b = last.getWorldPosition(new THREE.Vector3());
    // Spitze ≈ letztes Gelenk + Länge des vorletzten Glieds × 0,8 in Gliedrichtung
    const tip = b.clone().add(b.clone().sub(a).multiplyScalar(0.8));
    return out.copy(this.object.worldToLocal(tip));
  }

  private applyBones() {
    for (const f of FINGERS) {
      const limits = f === 'thumb' ? HANDS.maxFlex.thumb : HANDS.maxFlex.finger;
      this.fingers[f].forEach(({ bone, rest }, i) => {
        const amount = THREE.MathUtils.degToRad(limits[i]) * this.current[f];
        if (f === 'thumb' && i === 0) this.tmp.setFromAxisAngle(this.oppositionAxis, amount);
        else this.tmp.setFromAxisAngle(this.axis, amount * HANDS.flexSign);
        bone.quaternion.copy(rest).multiply(this.tmp);
      });
    }
  }
}

/**
 * Dreht und verschiebt das Modell so, dass es im einheitlichen Hand-Koordinatensystem liegt.
 * Abgeleitet aus den Knochenpositionen: Handgelenk → Mittelfinger = Fingerrichtung,
 * Kleinfinger → Zeigefinger = Querrichtung zum Daumen.
 */
function normalizeFrame(model: THREE.Object3D) {
  model.updateMatrixWorld(true);
  const pos = (name: string) => model.getObjectByName(name)!.getWorldPosition(new THREE.Vector3());
  const wrist = pos(HANDS.wristBone);
  const forward = pos(HANDS.fingers.middle[0]).sub(wrist).normalize();
  const towardThumb = pos(HANDS.fingers.index[0]).sub(pos(HANDS.fingers.pinky[0])).normalize();

  const z = forward.clone().negate();
  const x = towardThumb.clone().negate();
  x.sub(z.clone().multiplyScalar(x.dot(z))).normalize();
  const y = new THREE.Vector3().crossVectors(z, x);

  const toCanonical = new THREE.Matrix4().makeBasis(x, y, z).invert();
  const q = new THREE.Quaternion().setFromRotationMatrix(toCanonical);
  model.quaternion.premultiply(q);
  model.position.applyQuaternion(q).sub(wrist.clone().applyQuaternion(q));
  model.updateMatrixWorld(true);

  // Finger sind in der Ruhepose leicht zur Handfläche gekrümmt. Zeigt die Krümmung nach +Y,
  // liegt die Handfläche oben: 180° um die Fingerachse drehen, damit der Handrücken oben ist.
  const [root, , last] = HANDS.fingers.middle.map(pos);
  if (last.y > root.y) {
    const flip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI);
    model.quaternion.premultiply(flip);
    model.position.applyQuaternion(flip);
    model.updateMatrixWorld(true);
  }
  // Daumen auf +X bedeutet: Das Modell ist (z. B. durch einen gespiegelten Export) eine linke Hand.
  return { isLeft: pos(HANDS.fingers.thumb[0]).x > 0 };
}

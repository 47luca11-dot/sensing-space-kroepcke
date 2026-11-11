import * as THREE from 'three';
import { FEATURES } from '../config/features';
import { ROOM } from '../config/room';
import type { Vec3 } from '../config/types';
import { Groups, RAPIER, type PhysicsWorld } from '../physics/PhysicsWorld';
import { grungeWallMaterial } from './grungeWall';

interface RoomLight {
  light: THREE.PointLight;
  intensity: number;
  fixture?: THREE.MeshStandardMaterial;
  on: boolean;
}

/** Raumhülle (Boden, Wände, Decke), Licht und feste Kollisionskörper. */
export class Room {
  private readonly lights = new Map<string, RoomLight>();
  private readonly ambient: THREE.HemisphereLight;
  private readonly wallMat = grungeWallMaterial();

  constructor(scene: THREE.Scene, private readonly physics: PhysicsWorld) {
    const { width: w, depth: d, height: h, wallThickness: t, doorway } = ROOM;

    // Draußen hinter der Tür: heller Himmel als Belohnung
    scene.background = new THREE.Color('#b9d3e6');

    const floorMat = new THREE.MeshStandardMaterial({ map: tileTexture(), roughness: 0.85 });
    floorMat.map!.repeat.set(w, d);
    const trimMat = new THREE.MeshStandardMaterial({ color: '#5f564c', roughness: 0.8 });
    const ceilingMat = new THREE.MeshStandardMaterial({ color: '#efece6', roughness: 1 });

    this.addBox(scene, [w, 0.1, d], [0, -0.05, 0], floorMat);
    this.addBox(scene, [w + 2 * t, t, d + 2 * t], [0, h + t / 2, 0], ceilingMat);
    // Großer unsichtbarer Boden, damit man auch vor der Tür stehen kann
    this.addCollider([40, 0.1, 40], [0, -0.15, 0]);
    // Rasen vor der Tür
    this.addBox(scene, [6, 0.1, 4], [doorway.x, -0.06, -d / 2 - t - 2], new THREE.MeshStandardMaterial({ color: '#6f8f4e', roughness: 1 }));

    // Ost- und Westwand
    this.addWall(scene, [t, h, d + 2 * t], [w / 2 + t / 2, h / 2, 0]);
    this.addWall(scene, [t, h, d + 2 * t], [-w / 2 - t / 2, h / 2, 0]);
    // Südwand, mit Durchgang zum roten Raum
    const zS = d / 2 + t / 2;
    if (FEATURES.redRoom) {
      const sd = ROOM.southDoorway;
      const sLeft = sd.x - sd.width / 2 + w / 2;
      const sRight = w / 2 - (sd.x + sd.width / 2);
      this.addWall(scene, [sLeft, h, t], [-w / 2 + sLeft / 2, h / 2, zS]);
      this.addWall(scene, [sRight, h, t], [w / 2 - sRight / 2, h / 2, zS]);
      this.addWall(scene, [sd.width, h - sd.height, t], [sd.x, (h + sd.height) / 2, zS]);
    } else {
      this.addWall(scene, [w, h, t], [0, h / 2, zS]);
    }

    // Nordwand mit Türöffnung: links, rechts, Sturz
    const zN = -d / 2 - t / 2;
    const left = doorway.x - doorway.width / 2 + w / 2;
    const right = w / 2 - (doorway.x + doorway.width / 2);
    this.addWall(scene, [left, h, t], [-w / 2 + left / 2, h / 2, zN]);
    this.addWall(scene, [right, h, t], [w / 2 - right / 2, h / 2, zN]);
    this.addWall(scene, [doorway.width, h - doorway.height, t], [doorway.x, (h + doorway.height) / 2, zN]);
    // Türrahmen
    const frame = 0.06;
    this.addVisual(scene, [frame, doorway.height, t + 0.04], [doorway.x - doorway.width / 2 - frame / 2, doorway.height / 2, zN], trimMat);
    this.addVisual(scene, [frame, doorway.height, t + 0.04], [doorway.x + doorway.width / 2 + frame / 2, doorway.height / 2, zN], trimMat);
    this.addVisual(scene, [doorway.width + 2 * frame, frame, t + 0.04], [doorway.x, doorway.height + frame / 2, zN], trimMat);

    // Sockelleisten
    for (const [s, c] of [
      ...(FEATURES.redRoom ? [] : [[[w, 0.08, 0.02], [0, 0.04, d / 2 - 0.01]]]),
      [[0.02, 0.08, d], [w / 2 - 0.01, 0.04, 0]],
      [[0.02, 0.08, d], [-w / 2 + 0.01, 0.04, 0]],
    ] as [Vec3, Vec3][]) {
      this.addVisual(scene, s, c, trimMat);
    }

    // Notausgangsschild über der Tür
    const signTexture = exitSignTexture();
    const sign = new THREE.Mesh(
      new THREE.BoxGeometry(0.42, 0.16, 0.03),
      new THREE.MeshStandardMaterial({ map: signTexture, emissive: '#ffffff', emissiveMap: signTexture, emissiveIntensity: 0.9 }),
    );
    sign.position.set(doorway.x, doorway.height + 0.25, -d / 2 + 0.02);
    scene.add(sign);

    // Licht
    this.ambient = new THREE.HemisphereLight('#fff6e8', '#4a4038', ROOM.ambient);
    scene.add(this.ambient);
    for (const cfg of ROOM.lights) {
      const light = new THREE.PointLight(cfg.color, cfg.intensity, cfg.distance, 1.6);
      light.position.set(...(cfg.position as Vec3));
      if ('castShadow' in cfg && cfg.castShadow) {
        light.castShadow = true;
        light.shadow.mapSize.set(1024, 1024);
        light.shadow.bias = -0.002;
        light.shadow.radius = 4;
      }
      scene.add(light);
      let fixture: THREE.MeshStandardMaterial | undefined;
      if ('fixture' in cfg && cfg.fixture) {
        fixture = new THREE.MeshStandardMaterial({ color: '#fffaf0', emissive: cfg.color, emissiveIntensity: 1.5 });
        const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.3, 0.08, 32), fixture);
        lamp.position.set(cfg.position[0], h - 0.04, cfg.position[2]);
        scene.add(lamp);
      }
      this.lights.set(cfg.id, { light, intensity: cfg.intensity, fixture, on: true });
    }
  }

  isLightOn(id: string) {
    return this.lights.get(id)?.on ?? false;
  }

  /** Schaltet über die Intensität statt `visible`, damit Three.js keine Shader neu kompilieren muss. */
  setLight(id: string, on: boolean) {
    const entry = this.lights.get(id);
    if (!entry) return;
    entry.on = on;
    entry.light.intensity = on ? entry.intensity : 0;
    if (entry.fixture) entry.fixture.emissiveIntensity = on ? 1.5 : 0;
  }

  /** Wand mit Textur-Koordinaten aus der Weltposition: Das Muster läuft über alle Wandstücke durch. */
  private addWall(scene: THREE.Scene, size: Vec3, center: Vec3) {
    const geometry = worldUvBox(size, center, ROOM.wallTexture.tileMeters);
    const mesh = new THREE.Mesh(geometry, this.wallMat);
    mesh.position.set(...center);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    scene.add(mesh);
    this.addCollider(size, center);
  }

  private addBox(scene: THREE.Scene, size: Vec3, center: Vec3, material: THREE.Material) {
    this.addVisual(scene, size, center, material);
    this.addCollider(size, center);
  }

  private addVisual(scene: THREE.Scene, size: Vec3, center: Vec3, material: THREE.Material) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
    mesh.position.set(...center);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    scene.add(mesh);
  }

  private addCollider(size: Vec3, center: Vec3) {
    const body = this.physics.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(...center));
    this.physics.world.createCollider(
      RAPIER.ColliderDesc.cuboid(size[0] / 2, size[1] / 2, size[2] / 2).setFriction(0.8).setCollisionGroups(Groups.world),
      body,
    );
  }
}

/**
 * Box, deren UVs aus der Weltposition kommen (Meter / Kachelgröße).
 * Seitenflächen nutzen die waagerechte Wandrichtung, Ober- und Unterseite x/z.
 */
function worldUvBox(size: Vec3, center: Vec3, tile: [number, number]) {
  const geometry = new THREE.BoxGeometry(...size);
  const pos = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  const uv = geometry.getAttribute('uv');
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) + center[0];
    const y = pos.getY(i) + center[1];
    const z = pos.getZ(i) + center[2];
    const nx = Math.abs(normal.getX(i));
    const ny = Math.abs(normal.getY(i));
    if (ny > 0.5) uv.setXY(i, x / tile[0], z / tile[1]);
    else if (nx > 0.5) uv.setXY(i, z / tile[0], y / tile[1]);
    else uv.setXY(i, x / tile[0], y / tile[1]);
  }
  uv.needsUpdate = true;
  return geometry;
}

/** Fliesenboden, eine Fliese pro Meter. */
function tileTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256;
  const g = canvas.getContext('2d')!;
  g.fillStyle = '#8c8174';
  g.fillRect(0, 0, 256, 256);
  g.fillStyle = '#a69a8b';
  g.fillRect(4, 4, 248, 248);
  for (let i = 0; i < 400; i++) {
    g.fillStyle = `rgba(${Math.random() < 0.5 ? '255,255,255' : '0,0,0'},0.04)`;
    g.fillRect(Math.random() * 256, Math.random() * 256, 6, 6);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 8;
  return texture;
}

function exitSignTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 96;
  const g = canvas.getContext('2d')!;
  g.fillStyle = '#138a3e';
  g.fillRect(0, 0, 256, 96);
  g.fillStyle = '#ffffff';
  g.font = 'bold 44px system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('AUSGANG', 128, 50);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

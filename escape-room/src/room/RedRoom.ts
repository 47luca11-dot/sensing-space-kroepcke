import * as THREE from 'three';
import { RED_ROOM } from '../config/redRoom';
import { ROOM } from '../config/room';
import type { Vec3 } from '../config/types';
import { Groups, RAPIER, type PhysicsWorld } from '../physics/PhysicsWorld';

/**
 * Roter Raum hinter der Südwand des Büros: samtrote Wände, schwarz glänzender Boden,
 * rotes Licht, Neonleisten unter der Decke und ein Spot auf die Stange in der Mitte.
 * Einrichtung (Bühne, Stange, Bar, Sofas, Bier) steht in config/objects.ts.
 */
export function buildRedRoom(scene: THREE.Scene, physics: PhysicsWorld) {
  const { minX, maxX, minZ, maxZ, height: h, center } = RED_ROOM;
  const t = ROOM.wallThickness;
  const w = maxX - minX;
  const d = maxZ - minZ;
  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;

  const wall = new THREE.MeshStandardMaterial({ color: RED_ROOM.wallColor, roughness: 0.95, map: velvetTexture() });
  wall.map!.repeat.set(3, 1.5);
  const floor = new THREE.MeshStandardMaterial({ color: RED_ROOM.floorColor, roughness: 0.18, metalness: 0.2 });
  const ceiling = new THREE.MeshStandardMaterial({ color: '#120a0c', roughness: 1 });

  const box = (size: Vec3, pos: Vec3, material: THREE.Material, collide = true) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
    mesh.position.set(...pos);
    mesh.receiveShadow = true;
    scene.add(mesh);
    if (collide) {
      const body = physics.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(...pos));
      physics.world.createCollider(RAPIER.ColliderDesc.cuboid(size[0] / 2, size[1] / 2, size[2] / 2).setFriction(0.8).setCollisionGroups(Groups.world), body);
    }
  };

  box([w, 0.1, d], [cx, -0.05, cz], floor);
  box([w + 2 * t, t, d + t], [cx, h + t / 2, cz + t / 2], ceiling);
  box([t, h, d + t], [minX - t / 2, h / 2, cz + t / 2], wall);
  box([t, h, d + t], [maxX + t / 2, h / 2, cz + t / 2], wall);
  box([w, h, t], [cx, h / 2, maxZ + t / 2], wall);
  // Rote Wandseite an der Bürowand (nur Optik, die Bürowand kollidiert schon)
  const sd = ROOM.southDoorway;
  const inner = minZ + 0.005;
  const leftW = sd.x - sd.width / 2 - minX;
  const rightW = maxX - (sd.x + sd.width / 2);
  box([leftW, h, 0.01], [minX + leftW / 2, h / 2, inner], wall, false);
  box([rightW, h, 0.01], [maxX - rightW / 2, h / 2, inner], wall, false);
  box([sd.width, h - sd.height, 0.01], [sd.x, (h + sd.height) / 2, inner], wall, false);

  // Licht
  for (const l of RED_ROOM.lights) {
    const light = new THREE.PointLight(l.color, l.intensity, l.distance, 1.6);
    light.position.set(...l.position);
    scene.add(light);
  }
  const spot = new THREE.SpotLight(RED_ROOM.spot.color, RED_ROOM.spot.intensity, 4, 0.38, 0.6, 1.4);
  spot.position.set(center[0], h - 0.05, center[2]);
  spot.target.position.set(center[0], 0, center[2]);
  scene.add(spot, spot.target);

  // Neonleisten entlang der Decke
  const neon = new THREE.MeshStandardMaterial({ color: '#ff2050', emissive: '#ff1a4a', emissiveIntensity: 2.2 });
  const n = 0.025;
  box([w - 0.1, n, n], [cx, h - 0.06, maxZ - 0.04], neon, false);
  box([n, n, d - 0.1], [minX + 0.04, h - 0.06, cz], neon, false);
  box([n, n, d - 0.1], [maxX - 0.04, h - 0.06, cz], neon, false);

  // Leuchtschrift über dem Durchgang (Raumseite)
  const sign = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 0.22),
    new THREE.MeshBasicMaterial({ map: neonSignTexture('ROTER RAUM'), transparent: true, toneMapped: false }),
  );
  sign.position.set(sd.x, sd.height + 0.32, inner + 0.01);
  scene.add(sign);
}

function velvetTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256;
  const g = canvas.getContext('2d')!;
  g.fillStyle = '#ffffff';
  g.fillRect(0, 0, 256, 256);
  // Senkrechte Falten wie bei einem Samtvorhang
  for (let x = 0; x < 256; x++) {
    const v = 0.78 + 0.22 * Math.sin((x / 256) * Math.PI * 12) ** 2 + (Math.random() - 0.5) * 0.04;
    g.fillStyle = `rgb(${Math.round(v * 255)},${Math.round(v * 255)},${Math.round(v * 255)})`;
    g.fillRect(x, 0, 1, 256);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function neonSignTexture(text: string) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const g = canvas.getContext('2d')!;
  g.font = 'bold 76px system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  for (const [blur, color] of [[28, '#ff2a55'], [10, '#ff5a7a'], [0, '#ffe3ea']] as const) {
    g.shadowColor = '#ff2a55';
    g.shadowBlur = blur;
    g.fillStyle = color;
    g.fillText(text, 256, 66);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

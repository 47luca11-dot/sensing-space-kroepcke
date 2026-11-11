import * as THREE from 'three';
import type { ColliderConfig, PlaceholderShape, Vec3 } from '../config/types';

export interface Placeholder {
  object: THREE.Object3D;
  /** Passender vereinfachter Kollisionskörper, falls die Konfiguration keinen vorgibt. */
  collider: ColliderConfig;
}

type Builder = (size: Vec3 | undefined, color: string) => Placeholder;

const mat = (color: string, roughness = 0.7, metalness = 0) =>
  new THREE.MeshStandardMaterial({ color, roughness, metalness });

function mesh(geometry: THREE.BufferGeometry, material: THREE.Material, position: Vec3 = [0, 0, 0]) {
  const m = new THREE.Mesh(geometry, material);
  m.position.set(...position);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

const boxPart = (size: Vec3, center: Vec3, material: THREE.Material) =>
  mesh(new THREE.BoxGeometry(...size), material, center);

const boxCollider = (size: Vec3, center: Vec3): ColliderConfig => ({
  shape: 'box',
  halfExtents: [size[0] / 2, size[1] / 2, size[2] / 2],
  offset: center,
});

/** Ursprung = Mittelpunkt. */
const box: Builder = (size = [0.3, 0.3, 0.3], color) => ({
  object: boxPart(size, [0, 0, 0], mat(color, 0.85)),
  collider: boxCollider(size, [0, 0, 0]),
});

/** size[0] = Durchmesser. */
const sphere: Builder = (size = [0.2, 0.2, 0.2], color) => ({
  object: mesh(new THREE.SphereGeometry(size[0] / 2, 32, 16), mat(color, 0.45)),
  collider: { shape: 'sphere', radius: size[0] / 2 },
});

/** size = [Durchmesser, Höhe, –]. */
const cylinder: Builder = (size = [0.1, 0.25, 0.1], color) => ({
  object: mesh(new THREE.CylinderGeometry(size[0] / 2, size[0] / 2, size[1], 32), mat(color, 0.5)),
  collider: { shape: 'cylinder', radius: size[0] / 2, halfHeight: size[1] / 2 },
});

/** size = [Breite, Dicke, Höhe des liegenden Buchs]. */
const book: Builder = (size = [0.17, 0.04, 0.24], color) => {
  const [w, t, d] = size;
  const group = new THREE.Group();
  group.add(boxPart([w, t, d], [0, 0, 0], mat(color, 0.8)));
  group.add(boxPart([w * 0.94, t * 0.8, d * 0.96], [0.004, 0, 0], mat('#f1ead8', 0.95)));
  return { object: group, collider: boxCollider(size, [0, 0, 0]) };
};

/** size = [Durchmesser, Höhe, –]. */
const mug: Builder = (size = [0.085, 0.1, 0.085], color) => {
  const [d, h] = size;
  const material = mat(color, 0.35);
  const group = new THREE.Group();
  group.add(mesh(new THREE.CylinderGeometry(d / 2, d / 2 * 0.92, h, 32), material));
  const handle = mesh(new THREE.TorusGeometry(h * 0.28, d * 0.08, 8, 24, Math.PI), material, [d / 2, 0, 0]);
  handle.rotation.z = -Math.PI / 2;
  group.add(handle);
  return { object: group, collider: { shape: 'cylinder', radius: d / 2, halfHeight: h / 2 } };
};

/** Flach liegender Bartschlüssel, ca. 15 cm lang. */
const key: Builder = (_size, color) => {
  const material = mat(color, 0.3, 0.9);
  const group = new THREE.Group();
  const ring = mesh(new THREE.TorusGeometry(0.022, 0.007, 12, 32), material, [-0.05, 0, 0]);
  ring.rotation.x = Math.PI / 2;
  const shaft = mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.1, 12), material, [0.022, 0, 0]);
  shaft.rotation.z = Math.PI / 2;
  group.add(ring, shaft);
  group.add(boxPart([0.012, 0.008, 0.026], [0.06, 0, 0.012], material));
  group.add(boxPart([0.01, 0.008, 0.018], [0.042, 0, 0.009], material));
  // Etwas dicker als die Optik, damit der kleine Schlüssel nicht durch Bretter fällt.
  return { object: group, collider: boxCollider([0.15, 0.024, 0.06], [0, 0, 0]) };
};

/** Ursprung = Boden-Mitte. size = [Breite, Höhe, Tiefe]. */
const table: Builder = (size = [1.6, 0.76, 0.9], color) => {
  const [w, h, d] = size;
  const material = mat(color, 0.6);
  const top = 0.05;
  const leg = 0.07;
  const group = new THREE.Group();
  const parts: ColliderConfig[] = [];
  group.add(boxPart([w, top, d], [0, h - top / 2, 0], material));
  parts.push(boxCollider([w, top, d], [0, h - top / 2, 0]));
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const c: Vec3 = [sx * (w / 2 - 0.08), (h - top) / 2, sz * (d / 2 - 0.08)];
      group.add(boxPart([leg, h - top, leg], c, material));
      parts.push(boxCollider([leg, h - top, leg], c));
    }
  }
  return { object: group, collider: { shape: 'compound', parts } };
};

/** Offenes Regal, Ursprung = Boden-Mitte, Öffnung zeigt nach +Z. size = [Breite, Höhe, Tiefe]. */
const shelf: Builder = (size = [1.1, 1.6, 0.4], color) => {
  const [w, h, d] = size;
  const material = mat(color, 0.75);
  const t = 0.03;
  const group = new THREE.Group();
  const parts: ColliderConfig[] = [];
  const add = (s: Vec3, c: Vec3) => {
    group.add(boxPart(s, c, material));
    parts.push(boxCollider(s, c));
  };
  add([t, h, d], [-w / 2 + t / 2, h / 2, 0]);
  add([t, h, d], [w / 2 - t / 2, h / 2, 0]);
  add([w, h, 0.015], [0, h / 2, -d / 2 + 0.0075]);
  for (const y of [0.06, h * 0.25, h * 0.5, h * 0.75, h - t / 2]) add([w - 2 * t, t, d], [0, y, 0]);
  return { object: group, collider: { shape: 'compound', parts } };
};

/** Drehtür, Ursprung = unteres Scharnier, Türblatt erstreckt sich nach +X. size = [Breite, Höhe, Dicke]. */
const door: Builder = (size = [0.96, 2.08, 0.05], color) => {
  const [w, h, t] = size;
  const group = new THREE.Group();
  group.add(boxPart([w, h, t], [w / 2, h / 2, 0], mat(color, 0.55, 0.2)));
  const metal = mat('#c9c4b8', 0.25, 0.9);
  // Beschläge auf Griffhöhe, bei kleinen Türen (Tresor) mittig
  const handleY = Math.min(1.0, h * 0.5);
  const scale = Math.min(1, w / 0.9);
  for (const side of [-1, 1]) {
    group.add(boxPart([0.12 * scale, 0.02, 0.02], [w - 0.14 * scale, handleY, side * (t / 2 + 0.03)], metal));
    group.add(boxPart([0.02, 0.02, 0.03], [w - 0.09 * scale, handleY, side * (t / 2 + 0.015)], metal));
  }
  // Schlüsselloch-Beschlag auf der Raumseite (+Z)
  group.add(boxPart([0.04 * scale, 0.09 * scale, 0.006], [w - 0.09 * scale, handleY - 0.1 * scale, t / 2 + 0.003], metal));
  return { object: group, collider: boxCollider(size, [w / 2, h / 2, 0]) };
};

/** Lichtschalter an der Wand, Ursprung = Wandoberfläche, ragt nach +Z. Kind „lever“ wird animiert. */
const lightSwitch: Builder = (_size, color) => {
  const group = new THREE.Group();
  group.add(boxPart([0.08, 0.12, 0.012], [0, 0, 0.006], mat(color, 0.4)));
  const lever = boxPart([0.03, 0.05, 0.016], [0, 0, 0.016], mat('#d8d3c8', 0.4));
  lever.name = 'lever';
  group.add(lever);
  return { object: group, collider: boxCollider([0.1, 0.14, 0.04], [0, 0, 0.02]) };
};

/** Geringelter Haufen, Ursprung = Mitte. Etwa 14 cm breit. */
const poop: Builder = (_size, color) => {
  const material = new THREE.MeshPhysicalMaterial({ color, roughness: 0.45, clearcoat: 0.6, clearcoatRoughness: 0.35 });
  const group = new THREE.Group();
  const rings: [number, number, number][] = [
    [0.05, 0.022, -0.035],
    [0.038, 0.019, -0.005],
    [0.025, 0.016, 0.022],
  ];
  for (const [r, tube, y] of rings) {
    const ring = mesh(new THREE.TorusGeometry(r, tube, 10, 24), material, [0, y, 0]);
    ring.rotation.x = Math.PI / 2;
    group.add(ring);
  }
  group.add(mesh(new THREE.ConeGeometry(0.018, 0.035, 12), material, [0.004, 0.05, 0]));
  return { object: group, collider: { shape: 'cylinder', radius: 0.07, halfHeight: 0.05, offset: [0, 0, 0] } };
};

/** Tresorkorpus ohne Tür, Ursprung = Boden-Mitte, Öffnung nach +Z. size = [Breite, Höhe, Tiefe]. */
const safe: Builder = (size = [0.6, 0.6, 0.5], color) => {
  const [w, h, d] = size;
  const t = 0.04;
  const material = mat(color, 0.4, 0.6);
  const inner = mat('#2a2a2c', 0.9);
  const group = new THREE.Group();
  const parts: ColliderConfig[] = [];
  const add = (s: Vec3, c: Vec3, m = material) => {
    group.add(boxPart(s, c, m));
    parts.push(boxCollider(s, c));
  };
  add([t, h, d], [-w / 2 + t / 2, h / 2, 0]);
  add([t, h, d], [w / 2 - t / 2, h / 2, 0]);
  add([w, t, d], [0, t / 2, 0], inner);
  add([w, t, d], [0, h - t / 2, 0]);
  add([w, h, t], [0, h / 2, -d / 2 + t / 2]);
  return { object: group, collider: { shape: 'compound', parts } };
};

/** Zettel, flach liegend. size = [Breite, –, Länge]. */
const note: Builder = (size = [0.15, 0.002, 0.21], color) => {
  const [w, , l] = size;
  const canvas = document.createElement('canvas');
  // Grundriss-Skizze: Seitenverhältnis wie das Blatt
  canvas.width = 512;
  canvas.height = Math.round((512 * l) / w);
  const g = canvas.getContext('2d')!;
  const cw = canvas.width, ch = canvas.height;
  g.fillStyle = color;
  g.fillRect(0, 0, cw, ch);
  g.strokeStyle = 'rgba(30, 36, 60, 0.85)';
  g.lineCap = 'round';
  // Außenwände
  g.lineWidth = 7;
  g.strokeRect(cw * 0.12, ch * 0.15, cw * 0.62, ch * 0.66);
  // Innenwände
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(cw * 0.42, ch * 0.15);
  g.lineTo(cw * 0.42, ch * 0.5);
  g.moveTo(cw * 0.12, ch * 0.5);
  g.lineTo(cw * 0.3, ch * 0.5);
  g.stroke();
  // Türaufschlag
  g.lineWidth = 2;
  g.beginPath();
  g.arc(cw * 0.42, ch * 0.62, cw * 0.08, -Math.PI / 2, 0);
  g.stroke();
  // Treppe
  for (let i = 0; i < 8; i++) {
    g.beginPath();
    g.moveTo(cw * 0.55, ch * (0.2 + i * 0.035));
    g.lineTo(cw * 0.7, ch * (0.2 + i * 0.035));
    g.stroke();
  }
  // Maßketten und Notizen
  g.strokeStyle = 'rgba(160, 40, 40, 0.7)';
  g.beginPath();
  g.moveTo(cw * 0.12, ch * 0.9);
  g.lineTo(cw * 0.74, ch * 0.9);
  g.stroke();
  g.fillStyle = 'rgba(30, 36, 60, 0.85)';
  g.font = `${Math.round(ch * 0.07)}px "Segoe Print", "Bradley Hand", cursive`;
  g.fillText('EG  M 1:100', cw * 0.78, ch * 0.2);
  g.fillText('Licht?', cw * 0.8, ch * 0.55);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const paper = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.95 });
  const plane = mesh(new THREE.BoxGeometry(w, 0.002, l), paper, [0, 0.001, 0]);
  plane.castShadow = false;
  return { object: plane, collider: boxCollider([w, 0.01, l], [0, 0.005, 0]) };
};

/** Apfel, Ursprung = Mitte. size[0] = Durchmesser. */
const apple: Builder = (size = [0.08, 0.08, 0.08], color) => {
  const d = size[0];
  const group = new THREE.Group();
  const body = mesh(new THREE.SphereGeometry(d / 2, 24, 16), new THREE.MeshPhysicalMaterial({ color, roughness: 0.35, clearcoat: 0.5 }));
  body.scale.y = 0.9;
  group.add(body);
  group.add(mesh(new THREE.CylinderGeometry(0.002, 0.003, 0.02, 6), mat('#5a3d22'), [0, d * 0.48, 0]));
  return { object: group, collider: { shape: 'sphere', radius: d / 2 } };
};

/** Croissant als gebogener Wulst. */
const croissant: Builder = (_size, color) => {
  const material = mat(color, 0.6);
  const group = new THREE.Group();
  const body = mesh(new THREE.TorusGeometry(0.05, 0.022, 12, 24, Math.PI * 1.1), material);
  body.rotation.x = Math.PI / 2;
  body.rotation.z = -Math.PI * 0.05;
  body.scale.z = 0.8;
  group.add(body);
  return { object: group, collider: { shape: 'box', halfExtents: [0.07, 0.02, 0.05], offset: [0, 0, 0.015] } };
};

/** Messer, Klinge entlang +X, Ursprung = Übergang Griff/Klinge. */
const knife: Builder = (_size, color) => {
  const group = new THREE.Group();
  const steel = mat('#d8dbe0', 0.18, 1);
  group.add(boxPart([0.11, 0.022, 0.016], [-0.055, 0, 0], mat(color, 0.5)));
  group.add(boxPart([0.012, 0.03, 0.02], [0.006, 0, 0], steel));
  const blade = mesh(new THREE.BoxGeometry(0.15, 0.028, 0.003), steel, [0.085, -0.002, 0]);
  group.add(blade);
  const tip = mesh(new THREE.ConeGeometry(0.014, 0.04, 4), steel, [0.18, -0.002, 0]);
  tip.rotation.z = -Math.PI / 2;
  tip.scale.z = 0.12;
  group.add(tip);
  return { object: group, collider: { shape: 'box', halfExtents: [0.105, 0.016, 0.012], offset: [0.04, 0, 0] } };
};

/**
 * Lebensgroße Maßstabsfigur wie in Architekturmodellen: gesichtslos, mattweiß, auf runder Standplatte.
 * Ursprung = Boden-Mitte, Blick nach +Z. Kind „cuts“ sammelt die Schnittspuren.
 */
const dummy: Builder = (_size, color) => {
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.85 });
  const group = new THREE.Group();
  const add = (g: THREE.BufferGeometry, p: Vec3, r: Vec3 = [0, 0, 0]) => {
    const m = mesh(g, material, p);
    m.rotation.set(...r);
    group.add(m);
    return m;
  };
  add(new THREE.CylinderGeometry(0.22, 0.24, 0.03, 32), [0, 0.015, 0]);
  // Beine
  for (const s of [-1, 1]) {
    add(new THREE.CapsuleGeometry(0.06, 0.38, 6, 12), [s * 0.09, 0.27, 0]);
    add(new THREE.CapsuleGeometry(0.07, 0.36, 6, 12), [s * 0.095, 0.72, 0]);
  }
  add(new THREE.CapsuleGeometry(0.16, 0.12, 8, 16), [0, 0.98, 0]).scale.set(1, 1, 0.65); // Becken
  add(new THREE.CapsuleGeometry(0.17, 0.32, 8, 16), [0, 1.28, 0]).scale.set(1, 1, 0.62); // Rumpf
  // Arme hängend
  for (const s of [-1, 1]) {
    add(new THREE.CapsuleGeometry(0.045, 0.26, 6, 12), [s * 0.23, 1.3, 0], [0, 0, s * 0.08]);
    add(new THREE.CapsuleGeometry(0.04, 0.26, 6, 12), [s * 0.255, 0.97, 0.02], [0.1, 0, s * 0.04]);
  }
  add(new THREE.CylinderGeometry(0.045, 0.05, 0.08, 12), [0, 1.53, 0]); // Hals
  add(new THREE.SphereGeometry(0.1, 24, 16), [0, 1.66, 0]).scale.set(0.9, 1.15, 1); // Kopf
  const cuts = new THREE.Group();
  cuts.name = 'cuts';
  group.add(cuts);
  return {
    object: group,
    collider: {
      shape: 'compound',
      parts: [
        { shape: 'cylinder', radius: 0.22, halfHeight: 0.015, offset: [0, 0.015, 0] },
        { shape: 'capsule', radius: 0.17, halfHeight: 0.6, offset: [0, 0.98, 0] },
      ],
    },
  };
};

/** Weißes Massenmodell (Baukörperstudie) auf einer Grundplatte, Ursprung = Mitte. */
const massing: Builder = (_size, color) => {
  const material = mat(color, 0.9);
  const group = new THREE.Group();
  group.add(boxPart([0.32, 0.012, 0.24], [0, -0.05, 0], mat('#c9b48f', 0.8)));
  const blocks: [Vec3, Vec3][] = [
    [[0.1, 0.09, 0.08], [-0.08, -0.0, -0.04]],
    [[0.06, 0.06, 0.12], [0.05, -0.014, 0.03]],
    [[0.05, 0.03, 0.05], [0.1, -0.03, -0.07]],
    [[0.08, 0.025, 0.06], [-0.04, -0.032, 0.07]],
  ];
  for (const [s, c] of blocks) group.add(boxPart(s, c, material));
  return { object: group, collider: boxCollider([0.32, 0.11, 0.24], [0, -0.0, 0]) };
};

/** Dreikantmaßstab, entlang X, Ursprung = Mitte. */
const ruler: Builder = (_size, color) => {
  const body = mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.3, 3), mat(color, 0.4));
  body.rotation.z = Math.PI / 2;
  const group = new THREE.Group();
  group.add(body);
  return { object: group, collider: boxCollider([0.3, 0.024, 0.026], [0, 0, 0]) };
};

/** Verchromte Stange vom Boden bis zur Decke, Ursprung = Boden-Mitte. size[1] = Höhe. */
const pole: Builder = (size = [0.05, 2.8, 0.05], color) => {
  const [dia, h] = size;
  const chrome = new THREE.MeshStandardMaterial({ color, roughness: 0.08, metalness: 1 });
  const group = new THREE.Group();
  group.add(mesh(new THREE.CylinderGeometry(dia / 2, dia / 2, h, 24), chrome, [0, h / 2, 0]));
  group.add(mesh(new THREE.CylinderGeometry(dia * 1.6, dia * 1.6, 0.02, 24), chrome, [0, h - 0.01, 0]));
  return { object: group, collider: { shape: 'cylinder', radius: dia / 2 + 0.01, halfHeight: h / 2, offset: [0, h / 2, 0] } };
};

/** Runde Bühne, Ursprung = Boden-Mitte. size = [Durchmesser, Höhe, –]. */
const stage: Builder = (size = [1.8, 0.15, 1.8], color) => {
  const [dia, h] = size;
  const group = new THREE.Group();
  group.add(mesh(new THREE.CylinderGeometry(dia / 2, dia / 2, h, 48), new THREE.MeshStandardMaterial({ color, roughness: 0.2, metalness: 0.3 }), [0, h / 2, 0]));
  // Leuchtrand
  const ring = mesh(
    new THREE.TorusGeometry(dia / 2, 0.012, 8, 64),
    new THREE.MeshStandardMaterial({ color: '#ff3060', emissive: '#ff2a55', emissiveIntensity: 2 }),
    [0, h, 0],
  );
  ring.rotation.x = Math.PI / 2;
  ring.castShadow = false;
  group.add(ring);
  return { object: group, collider: { shape: 'cylinder', radius: dia / 2, halfHeight: h / 2, offset: [0, h / 2, 0] } };
};

/** Bartresen, Ursprung = Boden-Mitte, Gästeseite nach −X. size = [Tiefe, Höhe, Länge]. */
const bar: Builder = (size = [0.6, 1.05, 2.2], color) => {
  const [dx, h, len] = size;
  const group = new THREE.Group();
  group.add(boxPart([dx, h - 0.05, len], [0, (h - 0.05) / 2, 0], mat(color, 0.6)));
  group.add(boxPart([dx + 0.1, 0.05, len + 0.06], [-0.05, h - 0.025, 0], new THREE.MeshStandardMaterial({ color: '#1a1416', roughness: 0.15, metalness: 0.4 })));
  return { object: group, collider: boxCollider([dx + 0.1, h, len + 0.06], [-0.05, h / 2, 0]) };
};

/** Sofa mit Rückenlehne nach −Z, Ursprung = Boden-Mitte. size = [Breite, Höhe Lehne, Tiefe]. */
const sofa: Builder = (size = [1.8, 0.85, 0.85], color) => {
  const [w, h, d] = size;
  const velvet = mat(color, 0.95);
  const group = new THREE.Group();
  const parts: ColliderConfig[] = [];
  const add = (s: Vec3, c: Vec3) => {
    group.add(boxPart(s, c, velvet));
    parts.push(boxCollider(s, c));
  };
  add([w, 0.42, d], [0, 0.21, 0]);
  add([w, h - 0.42, 0.2], [0, 0.42 + (h - 0.42) / 2, -d / 2 + 0.1]);
  add([0.18, 0.62, d], [-w / 2 + 0.09, 0.31, 0]);
  add([0.18, 0.62, d], [w / 2 - 0.09, 0.31, 0]);
  return { object: group, collider: { shape: 'compound', parts } };
};

/** Bierflasche (braun, Etikett), Ursprung = Mitte. */
const beer: Builder = (_size, color) => {
  const glass = new THREE.MeshPhysicalMaterial({ color, roughness: 0.15, metalness: 0, clearcoat: 1, transparent: true, opacity: 0.92 });
  const group = new THREE.Group();
  group.add(mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.15, 24), glass, [0, -0.045, 0]));
  group.add(mesh(new THREE.CylinderGeometry(0.014, 0.032, 0.05, 24), glass, [0, 0.055, 0]));
  group.add(mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.05, 16), glass, [0, 0.105, 0]));
  group.add(mesh(new THREE.CylinderGeometry(0.0145, 0.0145, 0.01, 16), mat('#c9a43a', 0.3, 0.9), [0, 0.135, 0]));
  // Etikett
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 96;
  const g = canvas.getContext('2d')!;
  g.fillStyle = '#efe4c6';
  g.fillRect(0, 0, 256, 96);
  g.fillStyle = '#7a1420';
  g.font = 'bold 44px Georgia, serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('BIER', 128, 52);
  const label = new THREE.CanvasTexture(canvas);
  label.colorSpace = THREE.SRGBColorSpace;
  group.add(mesh(new THREE.CylinderGeometry(0.0325, 0.0325, 0.06, 24, 1, true), new THREE.MeshStandardMaterial({ map: label, roughness: 0.7 }), [0, -0.04, 0]));
  return { object: group, collider: { shape: 'cylinder', radius: 0.033, halfHeight: 0.07, offset: [0, 0, 0] } };
};

const builders: Record<PlaceholderShape, Builder> = {
  box,
  sphere,
  cylinder,
  book,
  mug,
  key,
  table,
  shelf,
  door,
  switch: lightSwitch,
  poop,
  safe,
  note,
  apple,
  croissant,
  knife,
  dummy,
  massing,
  ruler,
  pole,
  stage,
  bar,
  sofa,
  beer,
};

export function buildPlaceholder(shape: PlaceholderShape, size: Vec3 | undefined, color = '#9aa0a6'): Placeholder {
  return builders[shape](size, color);
}

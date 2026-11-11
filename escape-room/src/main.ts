import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { OBJECTS } from './config/objects';
import { ROOM } from './config/room';
import { Input } from './controls/Input';
import { PlayerController } from './controls/PlayerController';
import { FEATURES } from './config/features';
import { BodilyFunctions } from './game/BodilyFunctions';
import { EventBus } from './game/events';
import { GameLogic } from './game/GameLogic';
import { GlowCode } from './game/GlowCode';
import { Intoxication } from './game/Intoxication';
import { MoneyThrower } from './game/MoneyThrower';
import { HandSystem } from './hands/HandSystem';
import { InteractionSystem } from './interaction/InteractionSystem';
import { createOperable } from './interaction/operables';
import { createObject, ObjectRegistry } from './objects/ObjectFactory';
import { PhysicsWorld } from './physics/PhysicsWorld';
import { Room } from './room/Room';
import { buildRedRoom } from './room/RedRoom';
import { isTouchDevice, TouchControls } from './controls/TouchControls';
import { UnveilEffects } from './effects/UnveilEffect';
import { AssetUpload } from './ui/AssetUpload';
import { EffectsPanel } from './ui/EffectsPanel';
import { Hud } from './ui/Hud';
import './style.css';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

// ── Rendering ──────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, 1, 0.03, 100);
// Spark sortiert und zeichnet alle Gaussian Splats der Szene.
scene.add(new SparkRenderer({ renderer }));

const touch = isTouchDevice();
if (touch) document.body.classList.add('touch');

function resize() {
  // Auf Handys etwas niedriger auflösen, das spart viel Grafikleistung
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, touch ? 1.5 : 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ── Spielsysteme ───────────────────────────────────────────────────────
const input = new Input(canvas);
const unveil = new UnveilEffects();
const effectsPanel = new EffectsPanel(unveil);
let unveilStarted = false;
const events = new EventBus();
// Spiel betreten: am Rechner per Pointer Lock, auf Touchgeräten mit Bildschirmsteuerung
function enterGame() {
  if (!touch) {
    input.requestLock();
    return;
  }
  input.touchActive = true;
  hud.showStart(false);
  effectsPanel.setVisible(false);
  touchControls?.setVisible(true);
}
function pauseTouch() {
  input.touchActive = false;
  hud.closeDialogs();
  touchControls?.setVisible(false);
  effectsPanel.setVisible(true);
  hud.showStart(true, true);
}

const hud = new Hud(
  () => {
    logic.start();
    // Beim ersten Start enthüllen sich die Scans
    if (!unveilStarted) {
      unveilStarted = true;
      unveil.playAll();
    }
    enterGame();
  },
  () => {
    hud.hideSuccess();
    enterGame();
  },
);
hud.touchMode = touch;
hud.showStart(true);
const touchControls = touch
  ? new TouchControls(input, pauseTouch, [
      { label: 'Geld werfen', code: 'KeyM' },
      { label: 'Urinieren', code: 'KeyP' },
      { label: 'Stuhlgang', code: 'KeyK' },
      { label: 'Erbrechen', code: 'KeyV' },
      { label: 'Spucken', code: 'KeyG' },
      { label: 'Niesen', code: 'KeyN' },
    ])
  : null;
hud.setReady(false);
hud.setLoading('Physik wird geladen …');

const physics = await PhysicsWorld.create();
const room = new Room(scene, physics);
if (FEATURES.redRoom) buildRedRoom(scene, physics);
const registry = new ObjectRegistry();
const player = new PlayerController(physics, camera, input, ROOM.spawn.position, ROOM.spawn.yawDegrees);
const interaction = new InteractionSystem(physics, registry, player, camera, input, hud, events, scene);
const logic = new GameLogic(events, hud, input);

// Ego-Hände hängen an der Kamera, daher muss die Kamera Teil der Szene sein.
scene.add(camera);
hud.setLoading('Hände werden geladen …');
const hands = await HandSystem.create(camera).catch((error) => {
  console.error('Handmodell konnte nicht geladen werden:', error);
  return null;
});
events.on('interacted', ({ point }) => hands?.reachTo(point));
const body = FEATURES.bodilyFunctions ? new BodilyFunctions(scene, physics, player, camera, input, registry, hud) : null;
events.on('punched', () => hands?.punch());
// Essen füllt den Magen, Trinken die Blase
// Alkohol: Sicht schwankt, Gang wird schief, zu viel → Erbrechen
const intoxication = FEATURES.alcohol ? new Intoxication(camera, player, hud, canvas) : null;
if (intoxication) intoxication.onOverdose = () => body?.startPuke();
events.on('consumed', ({ kind, finished, alcohol }) => {
  if (kind === 'drink') {
    body?.drink();
    if (alcohol > 0) intoxication?.add(alcohol);
    hud.toast(finished ? 'Flasche leer.' : alcohol > 0 ? 'Prost!' : 'Ein Schluck.');
  } else {
    body?.eat();
    hud.toast(finished ? 'Aufgegessen.' : 'Ein Bissen.');
  }
});

// Dialoge (Zahlenschloss, Zettel) pausieren die Spielsteuerung
// Am Rechner wird dabei die Maus freigegeben, damit man die Knöpfe anklicken kann;
// auf dem Handy verschwindet die Touch-Steuerung, damit sie die Knöpfe nicht verdeckt.
let unlockedForDialog = false;
hud.onDialogChange = (open) => {
  input.suspended = open;
  if (touch) {
    touchControls?.setVisible(!open && input.touchActive);
    return;
  }
  if (open && input.isLocked) {
    unlockedForDialog = true;
    input.exitLock();
  } else if (!open && unlockedForDialog) {
    unlockedForDialog = false;
    // Klappt das erneute Fangen nicht (z. B. ohne Klick), erscheint der Pausenbildschirm
    input.requestLock(() => hud.showStart(true, true));
  }
};
const glowCode = new GlowCode(scene, room);

// Geld werfen: Scheine fliegen, die Hand schnippt, Figuren in der Nähe tanzen
const money = FEATURES.moneyThrowing ? new MoneyThrower(scene, physics, player, camera, input, registry, hud) : null;
if (money) {
  money.onThrow = () => {
    if (!interaction.heldObject) hands?.flick();
    const p = player.body.translation();
    for (const obj of registry.all) {
      if (!obj.behavior?.dance) continue;
      const t = obj.body.translation();
      if (Math.hypot(t.x - p.x, t.z - p.z) < 9) obj.behavior.dance(3);
    }
  };
}

input.onLockChange((locked) => {
  // Absichtlich für einen Dialog freigegeben: nicht pausieren
  if (!locked && unlockedForDialog) return;
  effectsPanel.setVisible(!locked);
  if (!locked) hud.closeDialogs();
  if (locked) hud.showStart(false);
  else if (!hud.successVisible) hud.showStart(true, true);
});

// ── Gegenstände laden (parallel, Splats brauchen etwas) ────────────────
hud.setLoading(`Gegenstände werden geladen … (0/${OBJECTS.length})`);
let loaded = 0;
const objects = await Promise.all(
  OBJECTS.map(async (config) => {
    const obj = await createObject(config, physics);
    hud.setLoading(`Gegenstände werden geladen … (${++loaded}/${OBJECTS.length})`);
    return obj;
  }),
);
for (const obj of objects) {
  obj.behavior = createOperable(obj, { events, room, ui: hud });
  if (obj.model.object instanceof SplatMesh) unveil.attach(obj.model.object, obj.model.bounds, false);
  registry.add(obj);
  scene.add(obj.root);
}
hud.setLoading(null);
hud.setReady(true);

// ── Eigene Scans aus dem Upload-Feld ───────────────────────────────────
let uploadSlot = 0;
const upload = new AssetUpload({
  async spawn(config) {
    const obj = await createObject(config, physics, false);
    // Vor den Spieler auf den Boden stellen; mehrere Uploads nebeneinander.
    const sideways = [0, 0.45, -0.45, 0.9, -0.9][uploadSlot++ % 5];
    const spot = player.groundPointAhead(0.8 + obj.radius, sideways);
    const margin = 0.3 + obj.radius;
    spot.x = THREE.MathUtils.clamp(spot.x, -ROOM.width / 2 + margin, ROOM.width / 2 - margin);
    spot.z = THREE.MathUtils.clamp(spot.z, -ROOM.depth / 2 + margin, ROOM.depth / 2 - margin);
    obj.body.setTranslation({ x: spot.x, y: -obj.model.bounds.min.y + 0.02, z: spot.z }, true);
    obj.captureState();
    obj.captureState();
    obj.syncVisual(1);
    registry.add(obj);
    scene.add(obj.root);
    if (obj.model.object instanceof SplatMesh) unveil.attach(obj.model.object, obj.model.bounds, true);
    return obj;
  },
  remove: (obj) => interaction.consume(obj),
});
upload.setReady();

// ── Spielschleife ──────────────────────────────────────────────────────
// Reihenfolge: Eingabe → feste Physikschritte → Interaktion/HUD → Rendern.
let last = performance.now();
renderer.setAnimationLoop((now) => {
  const dt = (now - last) / 1000;
  last = now;

  const { dx, dy } = input.consumeMouse();
  if (interaction.isRotating) interaction.rotateHeld(dx, dy);
  else player.rotateView(dx, dy);

  const alpha = physics.step(
    dt,
    (fixedDt) => {
      player.fixedUpdate(fixedDt);
      interaction.fixedUpdate(fixedDt);
    },
    () => {
      player.afterStep();
      for (const obj of registry.all) if (obj.isSimulated) obj.captureState();
    },
  );

  player.updateCamera(alpha);
  for (const obj of registry.all) if (obj.isSimulated) obj.syncVisual(alpha);
  body?.update(Math.min(dt, 0.1));
  glowCode.update(Math.min(dt, 0.1));
  money?.update(Math.min(dt, 0.1));
  intoxication?.update(Math.min(dt, 0.1));
  unveil.update(Math.min(dt, 0.1));
  interaction.update(Math.min(dt, 0.1));
  hands?.update(Math.min(dt, 0.1), interaction.heldObject, interaction.focusedObject, player.speed, interaction.pointAt);

  renderer.render(scene, camera);
  input.endFrame();
});

// Zum Debuggen in der Browser-Konsole: __game.player.teleport([0, 0, 0], 90) usw.
if (import.meta.env.DEV) {
  Object.assign(window, { __game: { scene, camera, physics, player, interaction, registry, room, input, events, hands, body, unveil, money, intoxication } });
}

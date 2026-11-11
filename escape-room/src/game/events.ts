import type { Vector3 } from 'three';
import type { GameObject } from '../objects/GameObject';

/** Ereignisse, über die Interaktionen der Spiellogik Bescheid geben. */
export interface GameEvents {
  /** Kurze Meldung am unteren Bildschirmrand. */
  message: string;
  'door-unlocked': GameObject;
  'door-opened': GameObject;
  'light-toggled': { id: string; on: boolean };
  'item-picked-up': GameObject;
  /** Ein bedienbares Objekt wurde mit E benutzt; `point` = getroffene Stelle in Weltkoordinaten. */
  interacted: { object: GameObject; point: Vector3 };
  /** Faustschlag ausgeführt (`hit` = etwas getroffen). */
  punched: { hit: boolean };
  /** Ein Bissen gegessen oder ein Schluck getrunken. */
  consumed: { object: GameObject; kind: 'food' | 'drink'; finished: boolean; alcohol: number };
}

type Listener<T> = (payload: T) => void;

export class EventBus {
  private readonly listeners = new Map<keyof GameEvents, Listener<never>[]>();

  on<K extends keyof GameEvents>(type: K, fn: Listener<GameEvents[K]>) {
    const list = this.listeners.get(type) ?? [];
    list.push(fn as Listener<never>);
    this.listeners.set(type, list);
  }

  emit<K extends keyof GameEvents>(type: K, payload: GameEvents[K]) {
    for (const fn of this.listeners.get(type) ?? []) (fn as Listener<GameEvents[K]>)(payload);
  }
}

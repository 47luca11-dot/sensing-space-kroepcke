/**
 * Pfade zu Dateien im public-Ordner. Für die Veröffentlichung als claude.ai-Artifact
 * (VITE_ASSET_ENCODING=b64) liegen Binärdateien (Scans, Handmodell) als Base64-Text daneben,
 * weil dort nur Web-Standardtypen ausgeliefert werden; siehe tools/encode-assets.mjs.
 */
const BINARY = /\.(ply|spz|splat|ksplat|glb)$/i;
const ENCODED = import.meta.env.VITE_ASSET_ENCODING === 'b64';

/** Pfade mit „/“ am Anfang zeigen auf den public-Ordner, auch wenn das Spiel in einem Unterordner läuft. */
export function assetUrl(path: string) {
  const url = path.startsWith('/') ? import.meta.env.BASE_URL + path.slice(1) : path;
  return ENCODED && path.startsWith('/') && BINARY.test(path) ? `${url}.b64.txt` : url;
}

/** Lädt eine Datei als Bytes, egal ob direkt oder als Base64-Text abgelegt. */
export async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (!url.endsWith('.b64.txt')) return res.arrayBuffer();
  const text = (await res.text()).trim();
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

export const isEncoded = (url: string) => url.endsWith('.b64.txt');

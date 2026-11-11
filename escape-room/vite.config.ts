import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    // Auch im WLAN erreichbar, damit man auf dem Handy testen kann (Adresse zeigt Vite beim Start an)
    host: true,
  },
  preview: {
    host: true,
    // Fertige Version über einen Cloudflare-Tunnel von unterwegs öffnen (zufällige *.trycloudflare.com-Adresse)
    allowedHosts: ['.trycloudflare.com'],
  },
});

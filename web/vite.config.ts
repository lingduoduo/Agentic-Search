import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    cssCodeSplit: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          icons: ["lucide-react"],
        },
      },
    },
  },
  server: {
    // Proxy every backend prefix the app calls (api.ts). The admin panels use
    // unprefixed routers (/admin, /analytics, /chat); without these the panels
    // silently fail in dev because requests hit the Vite server, not FastAPI.
    proxy: {
      "/api": "http://127.0.0.1:7860",
      "/health": "http://127.0.0.1:7860",
      "/admin": "http://127.0.0.1:7860",
      "/analytics": "http://127.0.0.1:7860",
      "/chat": "http://127.0.0.1:7860",
    },
  },
});

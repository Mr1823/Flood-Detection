import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Localhost only - no base path, because the dashboard is not deployed anywhere.
// /api is proxied to the local prediction service (python src/serve_api.py), so the
// browser sees one origin and no CORS preflight in the common case.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    open: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});

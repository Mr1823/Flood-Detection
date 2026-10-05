import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Localhost only - no base path, because the dashboard is not deployed anywhere.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, open: true },
});

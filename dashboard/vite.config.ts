/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

// The built SPA is committed into the Python package and served by the
// FastAPI dashboard app (gated by the optional [dashboard] extra), so the
// build emits straight into the package's static dir. `base: "./"` keeps
// asset URLs relative regardless of the mount path.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    outDir: "../src/synara/features/dashboard/static",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    // `bun run dev` proxies the API to a locally running
    // `SYNARA_DASHBOARD=true synara-mcp` (default bind 127.0.0.1:8765).
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});

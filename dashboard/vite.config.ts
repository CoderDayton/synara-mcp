/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

// The built SPA is committed into the Python package and served by the
// FastAPI dashboard app (gated by the optional [dashboard] extra), so the
// build emits straight into the package's static dir. `base: "./"` keeps
// asset URLs relative regardless of the mount path.
// React Compiler (formerly React Forget) auto-memoizes component
// output. With this in place hand-written `useMemo`/`useCallback`/
// `React.memo` become redundant — values keep stable identity when
// their inputs are stable, without the boilerplate.
//
// Wiring follows @vitejs/plugin-react v6's recommended pattern: the
// main `react()` plugin runs the SWC-based JSX/HMR transform, then a
// `@rolldown/plugin-babel` pass applies the compiler preset to the
// same file set. `target: "19"` emits against React 19's built-in
// compiler runtime (no `react-compiler-runtime` polyfill needed).

export default defineConfig({
  base: "./",
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset({ target: "19" })] }),
    tailwindcss(),
  ],
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

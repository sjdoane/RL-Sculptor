import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Backend default is port 8000 (see backend/config.py). Override via
// the RS_BACKEND_URL env var if the user runs uvicorn elsewhere.
const BACKEND_URL = process.env.RS_BACKEND_URL ?? "http://localhost:8000";
const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: BACKEND_URL,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
      "/ws": {
        target: BACKEND_WS_URL,
        ws: true,
        changeOrigin: true,
        // No rewrite: the backend registers WS routes at
        // `/ws/projects/.../events` + `/ws/.../frames`, so the
        // `/ws/` prefix must reach uvicorn unchanged. Stripping it
        // caused every WS connect to 403 (no route match).
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Carve Monaco + its language workers into their own chunks so
        // the main bundle stays slim (Prompt 7 precondition R2: main
        // chunk under ~200 KB gzipped). We only load Monaco when the
        // Rewards tab is opened, via React.lazy() in the page itself.
        manualChunks(id) {
          if (id.includes("node_modules/monaco-editor")) return "monaco";
          if (id.includes("node_modules/@monaco-editor")) return "monaco";
          // Recharts pulls d3-scale / d3-shape / d3-path which together
          // are ~90 KB gzipped. Split them into a `charts` chunk so
          // the main bundle stays small — the Runs tab's lazy import
          // boundary still controls when this loads.
          if (id.includes("node_modules/recharts")) return "charts";
          if (id.includes("node_modules/d3-")) return "charts";
          if (id.includes("node_modules/react-window")) return "charts";
          return undefined;
        },
      },
    },
    chunkSizeWarningLimit: 2000,
  },
});

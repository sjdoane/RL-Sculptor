import { lazy, Suspense } from "react";
import { Loader2 } from "lucide-react";

import type { OnChange, OnMount } from "@monaco-editor/react";

/** Lazy-loaded wrapper around Monaco. Per Prompt 7 precondition R2,
 *  Monaco must ship in a separate async chunk so the main bundle stays
 *  under ~200 KB gzipped. Vite's code splitter emits the `monaco`
 *  chunk because of the dynamic import here and the `manualChunks`
 *  rule in vite.config.ts.
 *
 *  Only import this inside the Rewards tab — don't bring it into the
 *  main bundle via a top-level import. */

const MonacoEditor = lazy(() =>
  import("@monaco-editor/react").then((m) => ({ default: m.Editor })),
);

const MonacoDiffEditor = lazy(() =>
  import("@monaco-editor/react").then((m) => ({ default: m.DiffEditor })),
);

export interface MonacoLazyProps {
  value: string;
  onChange?: OnChange;
  onMount?: OnMount;
  readOnly?: boolean;
  height?: string | number;
  language?: string;
}

export function MonacoLazy(props: MonacoLazyProps) {
  const { value, onChange, onMount, readOnly, height = 420, language = "python" } = props;
  return (
    <Suspense fallback={<MonacoFallback height={height} />}>
      <MonacoEditor
        value={value}
        onChange={onChange}
        onMount={onMount}
        language={language}
        theme="vs-light"
        height={height}
        options={{
          readOnly,
          fontSize: 13,
          fontFamily:
            "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace",
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          smoothScrolling: true,
          wordWrap: "off",
          renderWhitespace: "selection",
          lineNumbers: "on",
          automaticLayout: true,
          tabSize: 4,
        }}
      />
    </Suspense>
  );
}

export interface MonacoDiffLazyProps {
  original: string;
  modified: string;
  height?: string | number;
  language?: string;
  /** Default true — the reward / physics diff views are meant to be
   * read-only: changes are produced by Claude, not typed in here. */
  readOnly?: boolean;
}

/** Read-only side-by-side (or inline) diff. Used by the Rewards tab's
 * "Diff vs parent" toggle and the Physics tab's after-commit diff
 * preview. Shares the same lazy-loaded Monaco chunk as `MonacoLazy`. */
export function MonacoDiffLazy(props: MonacoDiffLazyProps) {
  const {
    original,
    modified,
    height = 420,
    language = "python",
    readOnly = true,
  } = props;
  return (
    <Suspense fallback={<MonacoFallback height={height} />}>
      <MonacoDiffEditor
        original={original}
        modified={modified}
        language={language}
        theme="vs-light"
        height={height}
        options={{
          readOnly,
          fontSize: 13,
          fontFamily:
            "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace",
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          smoothScrolling: true,
          renderSideBySide: true,
          automaticLayout: true,
        }}
      />
    </Suspense>
  );
}

function MonacoFallback({ height }: { height: string | number }) {
  const h = typeof height === "number" ? `${height}px` : height;
  return (
    <div
      className="flex items-center justify-center rounded-md border bg-muted/20 text-sm text-muted-foreground"
      style={{ height: h }}
    >
      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
      Loading editor…
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { PaperDetailModal } from "@/components/PaperDetailModal";
import { kgGraphHtmlUrl } from "@/lib/api";

/** Fullscreen-ish modal with the pyvis graph embedded in an iframe.
 *  Hits `?regenerate=true` each time it opens so provenance highlights
 *  reflect the latest state.
 *
 *  M7 Phase 7e: the graph HTML's click-forwarder posts
 *  `{type: "kg_node_click", kind, arxiv_id}` messages to the parent
 *  window. When a Paper node is clicked, stack a PaperDetailModal on
 *  top so the user gets inline detail without leaving the graph. */
export function GraphModal({
  slug,
  open,
  onOpenChange,
}: {
  slug: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  // `iframeSrc` stays stable across re-renders within the same mount so
  // the iframe doesn't reload on every parent update — but bumps each
  // time the modal is reopened to force a fresh pyvis render.
  const iframeSrc = useMemo(
    () => (open ? kgGraphHtmlUrl(slug, { regenerate: true }) : ""),
    [open, slug],
  );

  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setSelectedArxivId(null);
      return;
    }
    const onMsg = (ev: MessageEvent) => {
      const data = ev.data as
        | { type?: string; kind?: string; arxiv_id?: string | null }
        | undefined;
      if (!data || data.type !== "kg_node_click") return;
      if (data.kind === "Paper" && typeof data.arxiv_id === "string") {
        setSelectedArxivId(data.arxiv_id);
      }
      // Non-Paper nodes (Technique / FailureMode / …) are no-ops for
      // now — the side-pane pattern would need per-kind detail views;
      // flagged as a Phase-7 follow-up.
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, [open]);

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        {/* `flex flex-col` overrides DialogContent's default `grid` so
            the iframe's `flex-1` actually claims the remaining height.
            Pre-fix (2026-04-23): grid + auto rows let the iframe
            collapse to its ~300px intrinsic height, leaving the top
            half of the modal blank. */}
        <DialogContent className="flex h-[92vh] w-[95vw] max-w-[95vw] flex-col gap-2 p-4 sm:rounded-lg">
          <DialogHeader className="space-y-0">
            <DialogTitle className="text-sm">Knowledge graph</DialogTitle>
            <DialogDescription className="text-[11px]">
              Interactive pyvis render · drag nodes, hover for details,
              click a paper to see its full record
            </DialogDescription>
          </DialogHeader>
          <div className="min-h-0 flex-1 overflow-hidden rounded border">
            {iframeSrc && (
              <iframe
                key={iframeSrc}
                src={iframeSrc}
                title="Knowledge graph"
                className="h-full w-full border-0"
                sandbox="allow-scripts allow-same-origin"
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
      <PaperDetailModal
        slug={slug}
        arxivId={selectedArxivId}
        open={Boolean(selectedArxivId)}
        onOpenChange={(o) => {
          if (!o) setSelectedArxivId(null);
        }}
      />
    </>
  );
}

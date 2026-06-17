import { useEffect, useMemo, useState } from "react";

import { Modal } from "@/components/rs/primitives";
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
      {open && (
        <Modal
          full
          flush
          icon="network"
          title="Knowledge graph"
          subtitle="Interactive pyvis render · drag nodes, hover for details, click a paper to see its full record"
          onClose={() => onOpenChange(false)}
        >
          {/* flex:1 wrapper so the iframe claims the body's full height
              (rs-modal.full flexes column-wise). */}
          <div style={{ flex: 1, minHeight: 0, overflow: "hidden", borderTop: "1px solid var(--hairline)", background: "var(--canvas)" }}>
            {iframeSrc && (
              <iframe
                key={iframeSrc}
                src={iframeSrc}
                title="Knowledge graph"
                style={{ width: "100%", height: "100%", border: 0, display: "block" }}
                sandbox="allow-scripts allow-same-origin"
              />
            )}
          </div>
        </Modal>
      )}
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

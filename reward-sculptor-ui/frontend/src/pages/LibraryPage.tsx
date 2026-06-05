import { Link } from "react-router-dom";

import { LibraryBrowser } from "@/components/RobotConfig";
import { Icon } from "@/components/rs/icon";
import { useLibraryRobots, useSystemGpu } from "@/hooks/useLibrary";

/** Standalone library page — the primary entry point for creating a
 *  new project. Reuses LibraryBrowser from RobotConfig so the in-
 *  project robot picker and the top-level library share filter state
 *  + card rendering exactly. */
export default function LibraryPage() {
  const robots = useLibraryRobots();
  const gpu = useSystemGpu();

  const total = robots.data?.total ?? 0;
  const mjlabReady =
    robots.data?.robots.filter((r) => r.training_support === "mjlab_ready").length ?? 0;
  const cudaOk = !!gpu.data?.cuda_available;

  return (
    <div className="rs-scroll">
      <div className="rs-pad" style={{ display: "flex", flexDirection: "column", gap: 18, maxWidth: 1320, margin: "0 auto" }}>
        <header style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
            <Link to="/" aria-label="Back to dashboard" className="rs-btn rs-btn-ghost rs-btn-sm" style={{ marginTop: 2 }}>
              <Icon name="arrow-left" size={14} />Dashboard
            </Link>
            <div>
              <h1 className="rs-h2" style={{ display: "flex", alignItems: "center", gap: 9, margin: 0 }}>
                <Icon name="library" size={20} />Robot library
              </h1>
              <p className="rs-sub" style={{ margin: "4px 0 0", maxWidth: 620, lineHeight: 1.5, fontSize: 13 }}>
                Pick a robot to spin up a new project. Every mjlab-ready entry scaffolds with a batched reward template and auto-seeds the knowledge graph with canonical references.
              </p>
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4, textAlign: "right" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="rs-badge emerald"><Icon name="sparkles" size={11} />{mjlabReady} ready to train</span>
              <span className="rs-badge slate">{total} total</span>
            </div>
            <span style={{ fontSize: 11, color: cudaOk ? "var(--rs-muted)" : "var(--st-amber-fg)" }}>
              {cudaOk
                ? `GPU: ${gpu.data?.devices[0]?.name ?? "detected"}`
                : "No GPU detected — mjlab adapter disabled"}
            </span>
          </div>
        </header>

        <LibraryBrowser />
      </div>
    </div>
  );
}

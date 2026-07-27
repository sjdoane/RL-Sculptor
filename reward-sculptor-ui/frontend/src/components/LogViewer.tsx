import { useEffect, useMemo, useRef, useState } from "react";
import { FixedSizeList } from "react-window";

import { Icon } from "@/components/rs/icon";
import { Toggle } from "@/components/rs/primitives";
import type { RunEvent } from "@/lib/types";

const LINE_HEIGHT = 18;

type FilterMode = "all" | "logs" | "iters" | "edits" | "run";

/** Virtualized log tail (react-window). Renders only visible lines so
 *  10k+ lines stay responsive. Autoscroll follows the tail when enabled
 *  and the user hasn't scrolled up. Reskinned to the rs dark log; height
 *  measured so it fills the rs-mid-col (audit C2). */
export function LogViewer({ events }: { events: RunEvent[] }) {
  const [filter, setFilter] = useState<FilterMode>("all");
  const [autoscroll, setAutoscroll] = useState(true);
  const listRef = useRef<FixedSizeList>(null);
  const scrollTop = useRef(0);
  const userScrolled = useRef(false);

  // Measure the scroll region so react-window fills the flex pane
  // instead of a fixed 420px box.
  const bodyRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(420);
  useEffect(() => {
    const el = bodyRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const h = entries[0]?.contentRect.height;
      if (h && h > 0) setHeight(h);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const filtered = useMemo(() => events.filter((e) => matchesFilter(e, filter)), [events, filter]);

  useEffect(() => {
    if (autoscroll && !userScrolled.current && listRef.current && filtered.length > 0) {
      listRef.current.scrollToItem(filtered.length - 1, "end");
    }
  }, [filtered, autoscroll]);

  return (
    <div
      className="rs-stack"
      style={{ minHeight: 0, flex: 1, border: "1px solid var(--hairline)", borderRadius: "var(--radius-lg)", overflow: "hidden" }}
    >
      <div className="rs-log-bar">
        <Icon name="terminal" size={15} color="var(--rs-muted)" />
        <span style={{ fontSize: 13, fontWeight: 600 }}>Event log</span>
        <span className="rs-grow" />
        <div className="rs-select">
          <select value={filter} onChange={(e) => setFilter(e.target.value as FilterMode)} aria-label="Log filter">
            <option value="all">All events</option>
            <option value="logs">Log lines</option>
            <option value="iters">Iter events</option>
            <option value="edits">Edits + citations</option>
            <option value="run">Run lifecycle</option>
          </select>
        </div>
        <span className="rs-flex rs-gap-6 rs-sub" style={{ fontSize: 12 }}>
          Autoscroll
          <Toggle on={autoscroll} onChange={(v) => { setAutoscroll(v); userScrolled.current = false; }} label="Autoscroll" />
        </span>
        <span className="rs-num" style={{ fontSize: 11.5, color: "var(--rs-muted)" }}>{filtered.length}</span>
      </div>
      <div ref={bodyRef} className="rs-log" style={{ flex: 1, minHeight: 0, padding: 0 }}>
        <FixedSizeList
          ref={listRef}
          height={height}
          width="100%"
          itemCount={filtered.length}
          itemSize={LINE_HEIGHT}
          onScroll={({ scrollOffset, scrollUpdateWasRequested }) => {
            scrollTop.current = scrollOffset;
            if (scrollUpdateWasRequested) return;
            const contentHeight = filtered.length * LINE_HEIGHT;
            const distanceFromBottom = contentHeight - height - scrollOffset;
            userScrolled.current = distanceFromBottom > LINE_HEIGHT;
          }}
          overscanCount={20}
        >
          {({ index, style }) => <LogRow style={style} event={filtered[index]} />}
        </FixedSizeList>
      </div>
    </div>
  );
}

function matchesFilter(ev: RunEvent, mode: FilterMode): boolean {
  switch (mode) {
    case "all": return true;
    case "logs": return ev.type === "log_line";
    case "iters":
      return ev.type === "iter_started" || ev.type === "iter_completed" || ev.type === "rollout_done" || ev.type === "diagnosed";
    case "edits": return ev.type === "edit_applied" || ev.type === "citation_added";
    case "run":
      return ev.type === "run_started" || ev.type === "run_completed" || ev.type === "run_errored" || ev.type === "run_stopped" || ev.type === "early_stop" || ev.type === "connected" || ev.type.startsWith("remote_");
    default: return true;
  }
}

function LogRow({ style, event }: { style: React.CSSProperties; event: RunEvent }) {
  const label = prettyLabel(event);
  return (
    <div className="rs-log-line" style={{ ...style, alignItems: "center", gap: 10, overflow: "hidden", whiteSpace: "nowrap", padding: "0 16px", lineHeight: "18px" }}>
      <span className="rs-log-ts">{timeOf(event.ts)}</span>
      <EventBadge ev={event} />
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", color: "#d7d3c4" }} title={label}>{label}</span>
    </div>
  );
}

function timeOf(ts: string | undefined): string {
  if (!ts) return "--:--:--";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts.slice(11, 19);
  return d.toLocaleTimeString([], { hour12: false });
}

function prettyLabel(ev: RunEvent): string {
  if (ev.type === "log_line") return String(ev.text ?? "");
  const iter = ev.iter !== undefined ? ` iter=${ev.iter}` : "";
  switch (ev.type) {
    case "connected": return `connected (replay_count=${(ev as { replay_count?: number }).replay_count ?? "?"})`;
    case "iter_started": return `iter_started${iter} reward_version_before=${ev.reward_version_before ?? "?"}`;
    case "rollout_done": return `rollout_done${iter} size=${ev.size_bytes ?? "?"}`;
    case "diagnosed": return `diagnosed${iter} failure_modes=${JSON.stringify(ev.failure_modes ?? [])}`;
    case "edit_applied": return `edit_applied v${ev.reward_version_before}→v${ev.reward_version_after} paper_refs=${JSON.stringify(ev.paper_refs ?? [])}`;
    case "citation_added": return `citation_added v${ev.reward_version} arxiv=${ev.arxiv_id}`;
    case "iter_completed": return `iter_completed${iter} metric=${fmtNum(ev.primary_metric)} Δ=${fmtNum(ev.metric_delta)} failure_modes=${JSON.stringify(ev.failure_modes ?? [])}`;
    case "metric_history": return `metric_history len=${Array.isArray(ev.history) ? (ev.history as unknown[]).length : "?"}`;
    // §Ship 34: objective fitness-in-the-loop.
    case "iter_fitness": return `iter_fitness${iter} fitness=${fmtNum(ev.fitness)} prog=${fmtNum(ev.progress)} best=${fmtNum(ev.best_so_far)} Δ=${fmtNum(ev.delta_vs_previous)}`;
    case "best_reward_selected": return `best_reward_selected${iter} fitness=${fmtNum(ev.fitness)} reward=${String(ev.reward ?? "?")}`;
    // §Ship 20: best-of-K reward edit. `selected` is a 0-based winner index.
    case "edit_candidates_ranked": {
      const n = (ev as { n?: number }).n;
      const valid = (ev as { valid?: number }).valid;
      const sel = (ev as { selected?: number }).selected;
      return `best-of-K reward: evaluated ${n ?? "?"}, kept candidate ${typeof sel === "number" ? sel + 1 : "?"} (${valid ?? "?"} valid)`;
    }
    case "fitness_metric_warning": return `⚠ fitness_metric_warning ${String(ev.message ?? "")}`;
    // §Ship 20 (de-siloing): stage/mission persistence lifecycle.
    case "stage_final_selection": {
      const src = (ev as { source?: string }).source;
      const pass = (ev as { criterion_pass?: boolean }).criterion_pass;
      return `Kept iter ${ev.iter ?? "?"}${src ? ` (${src})` : ""} as the stage policy${typeof pass === "boolean" ? ` — criterion ${pass ? "passed" : "unmet"}` : ""}`;
    }
    case "mission_stage_archived":
    case "mission_archived": {
      const b = (ev as { total_bytes?: number }).total_bytes;
      const mb = typeof b === "number" ? `${(b / 1_048_576).toFixed(1)} MB` : "?";
      return `Saved to archive (${mb})`;
    }
    case "mission_archive_failed": return `Archive failed: ${String(ev.error ?? "")}`;
    case "mission_saved": return "Mission saved";
    case "early_stop": return `early_stop at iter=${ev.at_iter ?? "?"} reason=${ev.reason ?? ""}`;
    case "run_started": return `run_started iterations=${ev.iterations} goal=${ev.behavior_goal ?? ""}`;
    case "run_completed": return `run_completed rc=${ev.return_code} iters=${ev.iterations_run}`;
    case "run_errored": return `run_errored rc=${ev.return_code} ${ev.error ?? ""}`;
    case "run_stopped": return `run_stopped (${ev.source ?? "user"})`;
    // §Ship 23d: remote-dispatch lifecycle (host carried on every event).
    case "remote_dispatch_started": return `remote_dispatch_started${iter} host=${ev.host ?? "?"} phase=${ev.phase ?? "?"}`;
    case "remote_upload_completed": return `remote_upload_completed${iter} host=${ev.host ?? "?"} ${fmtNum(ev.seconds)}s`;
    case "remote_job_launched": return `remote_job_launched${iter} host=${ev.host ?? "?"} pgid=${ev.pgid ?? "?"}`;
    case "remote_job_reattached": return `remote_job_reattached${iter} host=${ev.host ?? "?"} (resumed a running job)`;
    case "remote_stale_job_killed": return `remote_stale_job_killed${iter} host=${ev.host ?? "?"}`;
    case "remote_connection_lost": return `remote_connection_lost${iter} host=${ev.host ?? "?"} attempt=${ev.attempt ?? "?"} retry_in=${fmtNum(ev.retry_in_s)}s`;
    case "remote_version_skew": return `remote_version_skew host=${ev.host ?? "?"} local torch=${ev.local_torch ?? "?"} remote torch=${ev.remote_torch ?? "?"}`;
    case "remote_job_finished": return `remote_job_finished${iter} host=${ev.host ?? "?"} rc=${ev.exit_code ?? "?"} ${fmtNum(ev.seconds)}s`;
    case "remote_artifacts_synced": return `remote_artifacts_synced${iter} host=${ev.host ?? "?"} ${fmtNum(ev.seconds)}s`;
    case "remote_dispatch_failed": return `remote_dispatch_failed${iter} host=${ev.host ?? "?"} reason=${ev.reason ?? "?"} ${String(ev.detail ?? "").slice(0, 160)}`;
    case "remote_config_ignored": return `remote_config_ignored reason=${ev.reason ?? "?"} ${ev.detail ?? ""}`;
    default: return `${ev.type} ${JSON.stringify({ ...ev, type: undefined, seq: undefined, ts: undefined })}`;
  }
}

function fmtNum(v: unknown): string {
  if (typeof v === "number" && Number.isFinite(v)) return v.toFixed(3);
  return "—";
}

const BADGE_STYLES: Record<string, string> = {
  log_line: "#2c2a20|#aaa698",
  connected: "#152836|#86b7e6",
  run_started: "#152836|#86b7e6",
  run_completed: "#16302a|#5fd0a0",
  run_errored: "#3a1620|#f08aa6",
  run_stopped: "#3a2c12|#f0b35a",
  early_stop: "#3a2c12|#f0b35a",
  iter_started: "#1e1b3a|#a99cf0",
  iter_completed: "#1e1b3a|#a99cf0",
  rollout_done: "#241a3a|#c0a8dd",
  diagnosed: "#241a3a|#c0a8dd",
  edit_applied: "#13302c|#5fd0c0",
  citation_added: "#13302c|#5fd0c0",
  metric_history: "#2c2a20|#aaa698",
  // §Ship 34: objective fitness — violet for the per-iter signal, green
  // for the best-by-fitness pick (matches run_completed's success green).
  iter_fitness: "#1e1b3a|#a99cf0",
  best_reward_selected: "#16302a|#5fd0a0",
  // §Ship 20: best-of-K reward selection — teal like the other edit events.
  edit_candidates_ranked: "#13302c|#5fd0c0",
  fitness_metric_warning: "#3a2c12|#f0b35a",
  // §Ship 20 (de-siloing): stage/mission persistence — green for the
  // kept-policy selection + archive/save, rose for a failed archive.
  stage_final_selection: "#16302a|#5fd0a0",
  mission_stage_archived: "#16302a|#5fd0a0",
  mission_archived: "#16302a|#5fd0a0",
  mission_saved: "#16302a|#5fd0a0",
  mission_archive_failed: "#3a1620|#f08aa6",
  // §Ship 23d: remote-dispatch lifecycle — teal for progress, amber for
  // degraded-but-recovering, rose for failures.
  remote_dispatch_started: "#13302c|#5fd0c0",
  remote_upload_completed: "#13302c|#5fd0c0",
  remote_job_launched: "#13302c|#5fd0c0",
  remote_job_finished: "#13302c|#5fd0c0",
  remote_artifacts_synced: "#16302a|#5fd0a0",
  remote_job_reattached: "#3a2c12|#f0b35a",
  remote_stale_job_killed: "#3a2c12|#f0b35a",
  remote_connection_lost: "#3a2c12|#f0b35a",
  remote_version_skew: "#3a2c12|#f0b35a",
  remote_config_ignored: "#3a2c12|#f0b35a",
  remote_dispatch_failed: "#3a1620|#f08aa6",
};

function EventBadge({ ev }: { ev: RunEvent }) {
  const [bg, fg] = (BADGE_STYLES[ev.type] ?? "#2c2a20|#aaa698").split("|");
  return (
    <span
      className="mono"
      style={{
        flexShrink: 0, display: "inline-flex", alignItems: "center", borderRadius: 4,
        padding: "0 4px", fontSize: 9, fontWeight: 600, textTransform: "uppercase",
        letterSpacing: 0.4, background: bg, color: fg,
      }}
    >
      {ev.type}
    </span>
  );
}

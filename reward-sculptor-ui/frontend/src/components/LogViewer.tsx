import { useEffect, useMemo, useRef, useState } from "react";
import { FixedSizeList } from "react-window";
import { ArrowDownToDot, Filter } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type { RunEvent } from "@/lib/types";

const LINE_HEIGHT = 18;

type FilterMode =
  | "all"
  | "logs"
  | "iters"
  | "edits"
  | "run";

/** Virtualized log tail. Renders only visible lines so 10k+ lines stay
 *  responsive. Autoscroll follows the tail when enabled and the user
 *  hasn't scrolled up. */
export function LogViewer({ events }: { events: RunEvent[] }) {
  const [filter, setFilter] = useState<FilterMode>("all");
  const [autoscroll, setAutoscroll] = useState(true);
  const listRef = useRef<FixedSizeList>(null);
  const scrollTop = useRef(0);
  const userScrolled = useRef(false);

  const filtered = useMemo(
    () => events.filter((e) => matchesFilter(e, filter)),
    [events, filter],
  );

  // Scroll-to-bottom whenever the filtered list grows AND either the
  // user hasn't scrolled away or autoscroll was just toggled on. The
  // key insight from the Test 1 round 3 (2026-04-22) bug: pre-fix
  // `onItemsRendered` set userScrolled based on visibleStopIndex,
  // which flipped to true during programmatic scrolls (react-window
  // emits multiple render passes as the scroll animates) → next event
  // saw userScrolled=true → autoscroll never refired. Using `onScroll`
  // with `scrollUpdateWasRequested` lets us distinguish USER scrolls
  // from PROGRAMMATIC ones.
  useEffect(() => {
    if (autoscroll && !userScrolled.current && listRef.current && filtered.length > 0) {
      listRef.current.scrollToItem(filtered.length - 1, "end");
    }
  }, [filtered, autoscroll]);

  return (
    <div className="flex min-h-0 flex-col rounded-md border bg-slate-950 text-slate-100">
      <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-2 py-1.5 text-[11px]">
        <div className="flex items-center gap-1.5">
          <Filter className="h-3 w-3 text-slate-400" />
          <Select
            value={filter}
            onChange={(e) => setFilter(e.target.value as FilterMode)}
            className="h-6 w-28 border-slate-700 bg-slate-900 text-[11px] text-slate-100"
          >
            <option value="all">all events</option>
            <option value="logs">log lines</option>
            <option value="iters">iter events</option>
            <option value="edits">edits + citations</option>
            <option value="run">run lifecycle</option>
          </Select>
          <span className="text-slate-500">· {filtered.length} entries</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            "h-6 gap-1 px-1.5 text-[11px]",
            autoscroll
              ? "text-slate-100 hover:bg-slate-800"
              : "text-slate-500 hover:bg-slate-800",
          )}
          onClick={() => {
            setAutoscroll((v) => !v);
            userScrolled.current = false;
          }}
        >
          <ArrowDownToDot className="h-3 w-3" />
          {autoscroll ? "autoscroll on" : "autoscroll off"}
        </Button>
      </div>
      <div className="relative min-h-0 flex-1">
        <FixedSizeList
          ref={listRef}
          height={420}
          width="100%"
          itemCount={filtered.length}
          itemSize={LINE_HEIGHT}
          onScroll={({ scrollOffset, scrollUpdateWasRequested }) => {
            scrollTop.current = scrollOffset;
            // Programmatic scroll from our own scrollToItem call —
            // don't mark the user as having scrolled away. react-window
            // sets scrollUpdateWasRequested=true when WE called
            // scrollTo / scrollToItem.
            if (scrollUpdateWasRequested) return;
            // User wheel / drag / keyboard scroll — detach autoscroll
            // if the scroll offset isn't within one item-height of the
            // bottom. Using offset math (not visibleStopIndex) avoids
            // the race where onItemsRendered fires mid-animation.
            const contentHeight = filtered.length * LINE_HEIGHT;
            const distanceFromBottom = contentHeight - 420 - scrollOffset;
            userScrolled.current = distanceFromBottom > LINE_HEIGHT;
          }}
          overscanCount={20}
        >
          {({ index, style }) => {
            const ev = filtered[index];
            return <LogRow style={style} event={ev} />;
          }}
        </FixedSizeList>
      </div>
    </div>
  );
}

function matchesFilter(ev: RunEvent, mode: FilterMode): boolean {
  switch (mode) {
    case "all":
      return true;
    case "logs":
      return ev.type === "log_line";
    case "iters":
      return ev.type === "iter_started" || ev.type === "iter_completed" ||
        ev.type === "rollout_done" || ev.type === "diagnosed";
    case "edits":
      return ev.type === "edit_applied" || ev.type === "citation_added";
    case "run":
      return ev.type === "run_started" || ev.type === "run_completed" ||
        ev.type === "run_errored" || ev.type === "run_stopped" ||
        ev.type === "early_stop" || ev.type === "connected";
    default:
      return true;
  }
}

function LogRow({
  style,
  event,
}: {
  style: React.CSSProperties;
  event: RunEvent;
}) {
  const label = prettyLabel(event);
  return (
    <div
      style={style}
      className="flex items-center gap-2 overflow-hidden whitespace-nowrap px-3 font-mono text-[11px] leading-[18px]"
    >
      <span className="shrink-0 text-slate-500">
        {timeOf(event.ts)}
      </span>
      <EventBadge ev={event} />
      <span className="truncate text-slate-200" title={label}>
        {label}
      </span>
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
    case "connected":
      return `connected (replay_count=${(ev as { replay_count?: number }).replay_count ?? "?"})`;
    case "iter_started":
      return `iter_started${iter} reward_version_before=${ev.reward_version_before ?? "?"}`;
    case "rollout_done":
      return `rollout_done${iter} size=${ev.size_bytes ?? "?"}`;
    case "diagnosed":
      return `diagnosed${iter} failure_modes=${JSON.stringify(ev.failure_modes ?? [])}`;
    case "edit_applied":
      return `edit_applied v${ev.reward_version_before}→v${ev.reward_version_after} paper_refs=${JSON.stringify(ev.paper_refs ?? [])}`;
    case "citation_added":
      return `citation_added v${ev.reward_version} arxiv=${ev.arxiv_id}`;
    case "iter_completed":
      return `iter_completed${iter} metric=${fmtNum(ev.primary_metric)} Δ=${fmtNum(ev.metric_delta)} failure_modes=${JSON.stringify(ev.failure_modes ?? [])}`;
    case "metric_history":
      return `metric_history len=${Array.isArray(ev.history) ? (ev.history as unknown[]).length : "?"}`;
    case "early_stop":
      return `early_stop at iter=${ev.at_iter ?? "?"} reason=${ev.reason ?? ""}`;
    case "run_started":
      return `run_started iterations=${ev.iterations} goal=${ev.behavior_goal ?? ""}`;
    case "run_completed":
      return `run_completed rc=${ev.return_code} iters=${ev.iterations_run}`;
    case "run_errored":
      return `run_errored rc=${ev.return_code} ${ev.error ?? ""}`;
    case "run_stopped":
      return `run_stopped (${ev.source ?? "user"})`;
    default:
      return `${ev.type} ${JSON.stringify({ ...ev, type: undefined, seq: undefined, ts: undefined })}`;
  }
}

function fmtNum(v: unknown): string {
  if (typeof v === "number" && Number.isFinite(v)) return v.toFixed(3);
  return "—";
}

const BADGE_STYLES: Record<string, string> = {
  log_line:       "bg-slate-800 text-slate-400",
  connected:      "bg-sky-900/60 text-sky-300",
  run_started:    "bg-sky-900/60 text-sky-300",
  run_completed:  "bg-emerald-900/60 text-emerald-300",
  run_errored:    "bg-rose-900/60 text-rose-300",
  run_stopped:    "bg-amber-900/60 text-amber-300",
  early_stop:     "bg-amber-900/60 text-amber-300",
  iter_started:   "bg-indigo-900/60 text-indigo-300",
  iter_completed: "bg-indigo-900/60 text-indigo-300",
  rollout_done:   "bg-violet-900/60 text-violet-300",
  diagnosed:      "bg-purple-900/60 text-purple-300",
  edit_applied:   "bg-teal-900/60 text-teal-300",
  citation_added: "bg-teal-900/60 text-teal-300",
  metric_history: "bg-slate-800 text-slate-400",
};

function EventBadge({ ev }: { ev: RunEvent }) {
  const cls = BADGE_STYLES[ev.type] ?? "bg-slate-800 text-slate-400";
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center rounded-sm px-1 py-0 text-[9px] font-semibold uppercase tracking-wide",
        cls,
      )}
    >
      {ev.type}
    </span>
  );
}

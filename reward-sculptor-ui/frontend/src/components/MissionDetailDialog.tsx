import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  CircleDot,
  Hourglass,
  Loader2,
  Play,
  Radio,
  StopCircle,
  Trash2,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  useDeleteMission,
  useMission,
  useRunMission,
} from "@/hooks/useMissions";
import { useMissionEvents } from "@/hooks/useMissionEvents";
import { ApiError } from "@/lib/api";
import { cn, formatRelative } from "@/lib/utils";
import type {
  MissionEvent,
  MissionLifecycleStatus,
  MissionSummary,
  StageSchema,
  StageStatus,
} from "@/lib/types";

const MAX_INDENT_DEPTH = 4;

export function MissionDetailDialog({
  slug,
  missionSlug,
  summary,
  open,
  onOpenChange,
}: {
  slug: string;
  missionSlug: string | null;
  summary: MissionSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const detail = useMission(slug, missionSlug ?? undefined, {
    enabled: open,
  });
  const run = useRunMission(slug);
  const del = useDeleteMission(slug);

  const mission = detail.data;
  const liveSummary = mission ?? summary;
  const activeJobId = liveSummary?.active_job_id ?? null;

  // Open the WS only when an active job is known to exist on the
  // current detail OR the row's summary. See useMissionEvents.ts for
  // the Ship 18a finding-A rationale. The `activeJobId` value comes
  // from `liveSummary` which prefers the detail (post-runMission's
  // optimistic setQueryData write) and falls back to the row's
  // summary, so the WS opens within a single render after Run is
  // clicked — no waiting for the next list refetch.
  const wsEnabled = open && missionSlug != null && activeJobId != null;

  // §Ship-19c: distinguish "mission.json missing because decompose
  // is still running" from "mission.json missing because the user
  // navigated to a bogus slug". The summary's active_job_kind tells
  // us — when a decompose job is active, the GET 404 is expected.
  const isDecomposing =
    liveSummary?.active_job_kind === "mission_decompose" &&
    activeJobId != null;
  const events = useMissionEvents(
    slug,
    missionSlug ?? undefined,
    wsEnabled,
  );

  const stageDepths = useMemo(
    () => (mission?.stages ? computeStageDepths(mission.stages) : new Map()),
    [mission?.stages],
  );

  // §Ship-19c: derive per-stage iter history from the WS structured-
  // event stream. iter_started / iter_completed / rollout_done events
  // fire from inside each stage's sculpt_run subprocess; they don't
  // carry stage_name themselves, so we attribute them to whichever
  // stage was last `stage_started`-emitted (orchestrator emits one
  // before invoking sculpt_run per stage). Replaces the deferred
  // "fill the Runs tab with stage runs" feature for Ship 19 (which
  // would need backend mods to register child Job entries — Ship
  // 19d). The dialog gives the same "is iter N running, what's
  // its metric" visibility without leaving the Missions tab.
  const stageIters = useMemo(
    () => deriveStageIters(events.structuredEvents),
    [events.structuredEvents],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[90vh] flex-col gap-4 sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span className="truncate">
              {liveSummary?.goal ?? "Mission"}
            </span>
            {liveSummary && (
              <MissionLifecycleBadge lifecycle={liveSummary.lifecycle} />
            )}
          </DialogTitle>
          <DialogDescription className="font-mono text-[11px]">
            {liveSummary
              ? `${liveSummary.mission_slug} · ${liveSummary.current_stage_idx}/${liveSummary.n_stages} stages · created ${formatRelative(liveSummary.created_at)}`
              : "Loading…"}
          </DialogDescription>
        </DialogHeader>

        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pr-1 scrollbar-thin">
          {/* §Ship-19c: while a mission is being decomposed, mission.json
              doesn't exist yet so the GET returns 404. Don't surface
              that as a destructive error — render a "Decomposing"
              placeholder + the live WS event stream below so the user
              sees Claude's stdout in real time. */}
          {detail.isLoading && !mission && !isDecomposing && (
            <p className="text-sm text-muted-foreground">Loading mission…</p>
          )}
          {isDecomposing && !mission && (
            <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
              <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-amber-700" />
              <div>
                <div className="font-medium text-amber-700 dark:text-amber-300">
                  Decomposing — Claude is building the curriculum
                </div>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  Typically 30-90 s. Stages will appear here when the
                  decompose job completes; the live event stream below
                  shows progress in real time.
                </p>
              </div>
            </div>
          )}
          {detail.error && !isDecomposing && (
            <p className="rounded border border-destructive/40 bg-destructive/5 p-2 font-mono text-[11px] text-destructive">
              {(detail.error as Error).message}
            </p>
          )}

          {mission?.lifecycle === "errored" && (
            <div className="flex items-start gap-2 rounded-md border border-rose-500/40 bg-rose-500/5 p-3 text-xs">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
              <div>
                <div className="font-medium text-rose-700 dark:text-rose-300">
                  mission.json is unreadable
                </div>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  The on-disk mission file failed to parse. The mission
                  can still be deleted to recover the slug.
                </p>
              </div>
            </div>
          )}

          {mission?.decomposition_rationale && (
            <section>
              <h3 className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                Decomposition rationale
              </h3>
              <p className="whitespace-pre-wrap rounded border bg-muted/30 p-3 text-xs">
                {mission.decomposition_rationale}
              </p>
            </section>
          )}

          {mission && mission.stages.length > 0 && (
            <section>
              <h3 className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                Stages ({mission.current_stage_idx}/{mission.n_stages})
              </h3>
              <div className="flex flex-col gap-1.5">
                {mission.stages.map((s, idx) => (
                  <StageCard
                    key={s.name}
                    stage={s}
                    depth={stageDepths.get(s.name) ?? 0}
                    isCurrent={idx === mission.current_stage_idx}
                    iters={stageIters.get(s.name) ?? []}
                  />
                ))}
              </div>
            </section>
          )}

          {(wsEnabled ||
            events.logLines.length > 0 ||
            events.structuredEvents.length > 0) && (
            <section className="flex min-h-0 flex-col gap-2">
              <div className="flex items-center justify-between">
                <h3 className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                  Live events
                </h3>
                <WsStatusChip events={events} />
              </div>

              {events.disconnected && !events.terminal && (
                <div className="rounded border border-amber-500/40 bg-amber-500/5 p-2 text-[11px] text-amber-800 dark:text-amber-200">
                  WebSocket disconnected — refresh the page to retry.
                  (Auto-reconnect lands in Ship 18c.)
                </div>
              )}
              {events.noActiveJob && (
                <div className="rounded border bg-muted/30 p-2 text-[11px] text-muted-foreground">
                  No active job — events from prior runs are not replayed.
                </div>
              )}

              {events.structuredEvents.length > 0 && (
                <StructuredEventList events={events.structuredEvents} />
              )}
              {(events.logLines.length > 0 || wsEnabled) && (
                <LogScroller events={events.logLines} />
              )}
            </section>
          )}
        </div>

        <DialogFooter className="border-t pt-3">
          {mission && mission.lifecycle !== "errored" && (
            <Button
              variant="outline"
              onClick={() => {
                if (!missionSlug) return;
                run.mutate(missionSlug, {
                  onSuccess: () => toast.success("Mission run queued"),
                  onError: (err) => {
                    const detailMsg =
                      err instanceof ApiError
                        ? err.problem.detail ?? err.problem.title
                        : err.message;
                    toast.error("Could not run mission", {
                      description: detailMsg,
                    });
                  },
                });
              }}
              disabled={
                run.isPending ||
                mission.lifecycle !== "ready" ||
                activeJobId != null
              }
              title={
                mission.lifecycle !== "ready"
                  ? `Lifecycle is ${mission.lifecycle}; run is only allowed when ready.`
                  : activeJobId
                    ? "An active job is already running for this mission."
                    : undefined
              }
            >
              {run.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Launching…
                </>
              ) : (
                <>
                  <Play />
                  Run mission
                </>
              )}
            </Button>
          )}
          {liveSummary && (
            <Button
              variant="outline"
              onClick={() => {
                if (!missionSlug) return;
                const ok = window.confirm(
                  `Delete mission ${missionSlug}? On-disk artifacts (.missions/${missionSlug}/) will be removed; the project itself stays. This is destructive.`,
                );
                if (!ok) return;
                del.mutate(missionSlug, {
                  onSuccess: (r) => {
                    onOpenChange(false);
                    toast.success("Mission deleted", {
                      description: `freed ${(r.freed_bytes / 1_048_576).toFixed(1)} MiB`,
                    });
                  },
                  onError: (err) => {
                    const detailMsg =
                      err instanceof ApiError
                        ? err.problem.detail ?? err.problem.title
                        : err.message;
                    toast.error("Could not delete mission", {
                      description: detailMsg,
                    });
                  },
                });
              }}
              disabled={del.isPending || activeJobId != null}
              title={
                activeJobId
                  ? "An active job is running. Wait for it to finish before deleting."
                  : undefined
              }
            >
              {del.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 />
              )}
              Delete
            </Button>
          )}
          <Button onClick={() => onOpenChange(false)}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── lifecycle / status badges ────────────────────────────────────────

const LIFECYCLE_STYLES: Record<
  MissionLifecycleStatus,
  {
    label: string;
    cls: string;
    icon: React.ComponentType<{ className?: string }>;
  }
> = {
  ready: {
    label: "ready",
    cls: "bg-emerald-50 text-emerald-700 border-emerald-200",
    icon: Hourglass,
  },
  running: {
    label: "running",
    cls: "bg-amber-50 text-amber-700 border-amber-200",
    icon: Radio,
  },
  completed: {
    label: "completed",
    cls: "bg-sky-50 text-sky-700 border-sky-200",
    icon: CheckCircle2,
  },
  halted: {
    label: "halted",
    cls: "bg-slate-100 text-slate-600 border-slate-200",
    icon: StopCircle,
  },
  errored: {
    label: "errored",
    cls: "bg-rose-50 text-rose-700 border-rose-200",
    icon: AlertTriangle,
  },
};

export function MissionLifecycleBadge({
  lifecycle,
}: {
  lifecycle: MissionLifecycleStatus;
}) {
  const m = LIFECYCLE_STYLES[lifecycle];
  const Icon = m.icon;
  return (
    <span
      role="status"
      aria-label={`mission lifecycle: ${m.label}`}
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide",
        m.cls,
        lifecycle === "running" && "motion-safe:animate-pulse",
      )}
    >
      <Icon className="h-2.5 w-2.5" />
      {m.label}
    </span>
  );
}

const STAGE_STATUS_STYLES: Record<
  StageStatus,
  { cls: string; icon: React.ComponentType<{ className?: string }> }
> = {
  pending: {
    cls: "bg-muted text-muted-foreground border-border",
    icon: CircleDot,
  },
  training: {
    cls: "bg-amber-50 text-amber-700 border-amber-200",
    icon: Radio,
  },
  succeeded: {
    cls: "bg-emerald-50 text-emerald-700 border-emerald-200",
    icon: CheckCircle2,
  },
  failed: {
    cls: "bg-rose-50 text-rose-700 border-rose-200",
    icon: XCircle,
  },
  skipped: {
    cls: "bg-slate-100 text-slate-600 border-slate-200",
    icon: StopCircle,
  },
};

function StageStatusBadge({ status }: { status: StageStatus }) {
  const m = STAGE_STATUS_STYLES[status];
  const Icon = m.icon;
  return (
    <span
      role="status"
      aria-label={`stage status: ${status}`}
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide",
        m.cls,
        status === "training" && "motion-safe:animate-pulse",
      )}
    >
      <Icon className="h-2.5 w-2.5" />
      {status}
    </span>
  );
}

// ── stage tree depth (DFS, cycle-safe) ───────────────────────────────

function computeStageDepths(stages: StageSchema[]): Map<string, number> {
  const byName = new Map(stages.map((s) => [s.name, s] as const));
  const depths = new Map<string, number>();

  const compute = (name: string, visited: Set<string>): number => {
    const cached = depths.get(name);
    if (cached !== undefined) return cached;
    if (visited.has(name)) return 0;
    const stage = byName.get(name);
    if (!stage) return 0;
    if (!stage.parent_stage) {
      depths.set(name, 0);
      return 0;
    }
    const parent = byName.get(stage.parent_stage);
    if (!parent) {
      depths.set(name, 0);
      return 0;
    }
    visited.add(name);
    const parentDepth = compute(stage.parent_stage, visited);
    visited.delete(name);
    const d = Math.min(parentDepth + 1, MAX_INDENT_DEPTH);
    depths.set(name, d);
    return d;
  };

  for (const s of stages) compute(s.name, new Set());
  return depths;
}

// §Ship-19c: per-iter row derived from the WS event stream.
type IterRow = {
  iter_index: number;
  primary_metric: number | null;
  rollout_done: boolean;
  completed: boolean;
};

function deriveStageIters(
  events: MissionEvent[],
): Map<string, IterRow[]> {
  const out = new Map<string, IterRow[]>();
  let currentStage: string | null = null;

  const ensure = (name: string): IterRow[] => {
    let arr = out.get(name);
    if (!arr) {
      arr = [];
      out.set(name, arr);
    }
    return arr;
  };

  for (const ev of events) {
    const t = ev.type;
    if (t === "stage_started") {
      currentStage = (ev.stage_name as string | null) ?? null;
      continue;
    }
    if (!currentStage) continue;
    if (t === "iter_started") {
      const iter = (ev as { iter?: number }).iter ?? 0;
      const arr = ensure(currentStage);
      // Don't double-add if a replay re-fires iter_started.
      if (arr.find((r) => r.iter_index === iter)) continue;
      arr.push({
        iter_index: iter,
        primary_metric: null,
        rollout_done: false,
        completed: false,
      });
    } else if (t === "iter_completed") {
      const iter = (ev as { iter?: number }).iter ?? 0;
      const m = (ev as { primary_metric?: number | null }).primary_metric;
      const arr = ensure(currentStage);
      const row = arr.find((r) => r.iter_index === iter);
      if (row) {
        row.primary_metric = typeof m === "number" ? m : null;
        row.completed = true;
      } else {
        arr.push({
          iter_index: iter,
          primary_metric: typeof m === "number" ? m : null,
          rollout_done: false,
          completed: true,
        });
      }
    } else if (t === "rollout_done") {
      const iter = (ev as { iter?: number }).iter ?? 0;
      const arr = ensure(currentStage);
      const row = arr.find((r) => r.iter_index === iter);
      if (row) row.rollout_done = true;
    }
  }
  // Sort each stage's iters by index so display order is stable.
  for (const arr of out.values()) {
    arr.sort((a, b) => a.iter_index - b.iter_index);
  }
  return out;
}

function StageCard({
  stage,
  depth,
  isCurrent,
  iters,
}: {
  stage: StageSchema;
  depth: number;
  isCurrent: boolean;
  iters: IterRow[];
}) {
  const orphan = stage.parent_stage !== null && depth === 0;
  return (
    <div
      className={cn(
        "rounded-md border p-2 text-xs",
        isCurrent ? "border-foreground/40 bg-accent/40" : "bg-muted/20",
      )}
      style={{ marginLeft: depth * 16 }}
    >
      <div className="flex flex-wrap items-center gap-2">
        <StageStatusBadge status={stage.status} />
        <span className="font-mono text-[11px] font-semibold">
          {stage.name}
        </span>
        {stage.parent_stage && (
          <span className="text-[10px] text-muted-foreground">
            parent: <code>{stage.parent_stage}</code>
          </span>
        )}
        {orphan && (
          <Badge
            variant="outline"
            className="border-amber-500/40 bg-amber-500/10 text-[9px] text-amber-700"
          >
            orphan parent_ref
          </Badge>
        )}
        {stage.redecomposition_attempts > 0 && (
          <Badge
            variant="outline"
            className="border-purple-500/40 bg-purple-500/10 text-[9px] text-purple-700"
          >
            redecomp ×{stage.redecomposition_attempts}
          </Badge>
        )}
      </div>
      <p className="mt-1 text-[11px]">{stage.goal_text}</p>
      <p className="mt-1 break-all rounded bg-background/60 px-1.5 py-1 font-mono text-[10.5px] text-muted-foreground">
        {stage.success_criterion}
      </p>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        <span>
          iters {stage.iterations_used}/{stage.max_iterations}
        </span>
        {stage.best_metric != null && (
          <span>best metric {stage.best_metric.toFixed(3)}</span>
        )}
        {stage.kg_seed_papers.length > 0 && (
          <span>kg refs {stage.kg_seed_papers.length}</span>
        )}
      </div>
      {iters.length > 0 && <IterRibbon iters={iters} />}
    </div>
  );
}

function IterRibbon({ iters }: { iters: IterRow[] }) {
  return (
    <div
      role="list"
      aria-label="Per-iteration progress"
      className="mt-1.5 flex flex-wrap items-center gap-1"
    >
      {iters.map((r) => (
        <IterChip key={r.iter_index} row={r} />
      ))}
    </div>
  );
}

function IterChip({ row }: { row: IterRow }) {
  const label = `iter ${row.iter_index}`;
  let cls: string;
  let metricStr: string;
  if (row.completed) {
    cls = "bg-emerald-50 text-emerald-700 border-emerald-200";
    metricStr =
      row.primary_metric != null
        ? row.primary_metric.toFixed(3)
        : "—";
  } else if (row.rollout_done) {
    cls = "bg-violet-50 text-violet-700 border-violet-200";
    metricStr = "rollout";
  } else {
    cls = "bg-amber-50 text-amber-700 border-amber-200 motion-safe:animate-pulse";
    metricStr = "training…";
  }
  return (
    <span
      role="listitem"
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1 py-0.5 text-[9.5px] font-mono",
        cls,
      )}
      title={`${label} · ${metricStr}`}
    >
      <span className="font-semibold uppercase tracking-wide">{label}</span>
      <span>{metricStr}</span>
    </span>
  );
}

// ── live events panels ───────────────────────────────────────────────

function WsStatusChip({
  events,
}: {
  events: ReturnType<typeof useMissionEvents>;
}) {
  let label: string;
  let cls: string;
  let Icon: React.ComponentType<{ className?: string }> = CircleDot;
  if (events.terminal) {
    label = "ended";
    cls = "bg-sky-50 text-sky-700 border-sky-200";
    Icon = CheckCircle2;
  } else if (events.disconnected) {
    label = "disconnected";
    cls = "bg-rose-50 text-rose-700 border-rose-200";
    Icon = XCircle;
  } else if (events.connected) {
    label = "live";
    cls = "bg-emerald-50 text-emerald-700 border-emerald-200";
    Icon = Radio;
  } else if (events.noActiveJob) {
    label = "no job";
    cls = "bg-muted text-muted-foreground border-border";
    Icon = StopCircle;
  } else {
    label = "connecting";
    cls = "bg-muted text-muted-foreground border-border";
    Icon = Loader2;
  }
  return (
    <span
      role="status"
      aria-label={`websocket: ${label}`}
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide",
        cls,
      )}
    >
      <Icon
        className={cn(
          "h-2.5 w-2.5",
          label === "connecting" && "motion-safe:animate-spin",
        )}
      />
      {label}
    </span>
  );
}

const EVENT_BADGE_STYLES: Record<string, string> = {
  mission_started: "bg-sky-100 text-sky-800 border-sky-300",
  mission_completed: "bg-emerald-100 text-emerald-800 border-emerald-300",
  mission_halted: "bg-rose-100 text-rose-800 border-rose-300",
  mission_halted_terminal: "bg-rose-100 text-rose-800 border-rose-300",
  mission_decompose_completed:
    "bg-emerald-100 text-emerald-800 border-emerald-300",
  mission_decompose_errored: "bg-rose-100 text-rose-800 border-rose-300",
  mission_execute_completed:
    "bg-emerald-100 text-emerald-800 border-emerald-300",
  mission_execute_errored: "bg-rose-100 text-rose-800 border-rose-300",
  stage_started: "bg-indigo-100 text-indigo-800 border-indigo-300",
  stage_succeeded: "bg-emerald-100 text-emerald-800 border-emerald-300",
  stage_failed: "bg-rose-100 text-rose-800 border-rose-300",
  stage_skipped: "bg-slate-100 text-slate-700 border-slate-300",
  stage_completed_training: "bg-violet-100 text-violet-800 border-violet-300",
  stage_criterion_evaluated:
    "bg-violet-100 text-violet-800 border-violet-300",
  stage_warm_start_resolved:
    "bg-teal-100 text-teal-800 border-teal-300",
  warm_start_skipped: "bg-amber-100 text-amber-800 border-amber-300",
  stage_scaffolded: "bg-sky-100 text-sky-800 border-sky-300",
  stage_v1_materialized: "bg-sky-100 text-sky-800 border-sky-300",
  stage_redecomposition_started:
    "bg-purple-100 text-purple-800 border-purple-300",
  stage_redecomposed: "bg-purple-100 text-purple-800 border-purple-300",
  redecomposition_skipped: "bg-amber-100 text-amber-800 border-amber-300",
  stage_redecomposition_failed:
    "bg-rose-100 text-rose-800 border-rose-300",
  feedback_read_degraded: "bg-amber-100 text-amber-800 border-amber-300",
};

function StructuredEventList({ events }: { events: MissionEvent[] }) {
  // Render newest-first; cap visual rows so the dialog doesn't grow.
  const slice = useMemo(() => events.slice(-200).reverse(), [events]);
  return (
    <div
      role="log"
      aria-label="Mission events"
      aria-live="polite"
      className="flex flex-col gap-1 rounded-md border bg-muted/10 p-2"
    >
      {slice.map((ev, i) => (
        <StructuredEventRow key={i} event={ev} />
      ))}
    </div>
  );
}

function StructuredEventRow({ event }: { event: MissionEvent }) {
  const cls =
    EVENT_BADGE_STYLES[event.type] ??
    "bg-slate-100 text-slate-700 border-slate-300";
  const detail = describeEvent(event);
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
      <span
        className={cn(
          "inline-flex shrink-0 rounded-sm border px-1 py-0 text-[9px] font-semibold uppercase tracking-wide",
          cls,
        )}
      >
        {event.type}
      </span>
      {event.stage_name && (
        <code className="text-[10.5px] text-muted-foreground">
          {event.stage_name}
        </code>
      )}
      <span className="break-all text-muted-foreground">{detail}</span>
    </div>
  );
}

function describeEvent(ev: MissionEvent): string {
  const type = ev.type;
  if (type === "stage_criterion_evaluated") {
    const passed = (ev as { passed?: boolean }).passed;
    const metric = (ev as { metric?: number | null }).metric;
    return `passed=${passed} metric=${fmtMetric(metric)}`;
  }
  if (type === "stage_completed_training") {
    const it = (ev as { iterations_run?: number }).iterations_run;
    return `iterations_run=${it ?? "?"}`;
  }
  if (type === "stage_failed") {
    const reason = (ev as { reason?: string }).reason ?? "";
    return reason;
  }
  if (type === "stage_redecomposed") {
    const subs = (ev as { sub_stage_names?: string[] }).sub_stage_names;
    return `sub_stages=${Array.isArray(subs) ? subs.length : "?"}`;
  }
  if (type === "redecomposition_skipped") {
    return (ev as { reason?: string }).reason ?? "";
  }
  if (type === "stage_redecomposition_failed") {
    return (ev as { reason?: string }).reason ?? "";
  }
  if (type === "warm_start_skipped") {
    return (ev as { reason?: string }).reason ?? "";
  }
  if (type === "feedback_read_degraded") {
    const missing = (ev as { missing_signals?: string[] }).missing_signals;
    return `missing=${JSON.stringify(missing ?? [])}`;
  }
  if (type === "mission_started") {
    const sc = (ev as { stage_count?: number }).stage_count;
    return `stage_count=${sc ?? "?"}`;
  }
  if (type === "mission_decompose_errored" || type === "mission_execute_errored") {
    return (ev as { error?: string }).error ?? "";
  }
  return "";
}

function fmtMetric(v: unknown): string {
  if (typeof v === "number" && Number.isFinite(v)) return v.toFixed(3);
  return "—";
}

function LogScroller({ events }: { events: MissionEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const userScrolledUp = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (userScrolledUp.current) return;
    el.scrollTop = el.scrollHeight;
  }, [events]);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    const distanceFromBottom =
      el.scrollHeight - el.clientHeight - el.scrollTop;
    userScrolledUp.current = distanceFromBottom > 20;
  };

  return (
    <div className="rounded-md border bg-slate-950 text-slate-100">
      <div className="flex items-center justify-between border-b border-slate-800 px-2 py-1 text-[10px] text-slate-400">
        <span>log_line · last {events.length}/200</span>
        {userScrolledUp.current && (
          <button
            type="button"
            className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200 hover:bg-slate-700"
            onClick={() => {
              userScrolledUp.current = false;
              const el = ref.current;
              if (el) el.scrollTop = el.scrollHeight;
            }}
          >
            scroll to bottom
          </button>
        )}
      </div>
      <div
        ref={ref}
        onScroll={onScroll}
        className="max-h-64 overflow-y-auto scrollbar-thin px-2 py-1 font-mono text-[11px] leading-snug"
      >
        {events.length === 0 ? (
          <p className="text-slate-500">waiting for stdout…</p>
        ) : (
          events.map((ev, i) => {
            const text =
              typeof (ev as { text?: unknown }).text === "string"
                ? (ev as { text: string }).text
                : JSON.stringify(ev);
            return (
              <div
                key={i}
                className="whitespace-pre-wrap break-all text-slate-200"
              >
                {text}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}


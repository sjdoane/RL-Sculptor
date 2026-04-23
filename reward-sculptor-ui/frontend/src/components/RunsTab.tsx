import { useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  Loader2,
  Radio,
  RefreshCw,
  StopCircle,
  Wand2,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useSystemGpu } from "@/hooks/useLibrary";
import { LogViewer } from "@/components/LogViewer";
import { MetricChart } from "@/components/MetricChart";
import { NewRunDialog } from "@/components/NewRunDialog";
import { useRunEvents } from "@/hooks/useRunEvents";
import { useRegenerateRewardTemplate } from "@/hooks/useRewards";
import { useKillRun, useRun, useRuns } from "@/hooks/useRuns";
import { ApiError } from "@/lib/api";
import { cn, formatRelative } from "@/lib/utils";
import type {
  ErrorClassification,
  IterEventSummary,
  JobStatus,
  ProjectDetail,
  RunDetail,
  RunEvent,
  RunSummary,
} from "@/lib/types";

// ── public entry ──────────────────────────────────────────────────────
export default function RunsTab({
  slug,
  project,
}: {
  slug: string;
  project: ProjectDetail;
}) {
  const list = useRuns(slug);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const runs = list.data ?? [];
  const selected = selectedRunId ?? runs[0]?.run_id ?? null;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 py-3">
          <div>
            <CardTitle className="text-sm">Runs</CardTitle>
            <CardDescription className="text-[11px]">
              Each run is a <code>sculpt run</code> subprocess — its
              JobManager kind is <code>sculpt_run</code>.
            </CardDescription>
          </div>
          <NewRunDialog
            slug={slug}
            project={project}
            onLaunched={(id) => setSelectedRunId(id)}
          />
        </CardHeader>
      </Card>

      {list.isLoading && (
        <p className="text-sm text-muted-foreground">Loading runs…</p>
      )}

      {!list.isLoading && runs.length === 0 && (
        <Card className="p-8 text-center">
          <Wand2 className="mx-auto h-8 w-8 text-muted-foreground" />
          <p className="mt-2 text-sm">No runs yet.</p>
          <p className="text-xs text-muted-foreground">
            Launch one above — keep it to a few iterations for the first pass.
          </p>
        </Card>
      )}

      {runs.length > 0 && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[240px_minmax(0,1fr)]">
          <RunSidebar
            runs={runs}
            selected={selected}
            onSelect={setSelectedRunId}
          />
          {selected ? (
            <RunDetailPane slug={slug} runId={selected} />
          ) : (
            <Card className="flex items-center justify-center p-8 text-xs text-muted-foreground">
              Select a run.
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

// ── sidebar ───────────────────────────────────────────────────────────
function RunSidebar({
  runs,
  selected,
  onSelect,
}: {
  runs: RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <Card className="max-h-[640px] overflow-y-auto scrollbar-thin">
      <CardHeader className="py-3">
        <CardTitle className="text-sm">History</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-1 p-2 pt-0">
        {runs.map((r) => (
          <button
            key={r.run_id}
            type="button"
            onClick={() => onSelect(r.run_id)}
            className={cn(
              "flex flex-col gap-1 rounded-md border px-2 py-2 text-left text-xs transition-colors",
              selected === r.run_id
                ? "border-foreground/30 bg-accent"
                : "border-transparent hover:bg-accent/60",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-mono text-[11px]">
                {r.run_id.replace(/^job_/, "")}
              </span>
              <RunStatusBadge status={r.status} />
            </div>
            <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
              <Sparkline history={r.primary_metric_history} />
              <span>
                {r.iterations_completed}/{r.iterations_requested}
              </span>
            </div>
            <span className="truncate text-[10px] text-muted-foreground">
              {r.started_at ? formatRelative(r.started_at) : "—"}
              {r.started_at && r.ended_at && ` · ${durationStr(r.started_at, r.ended_at)}`}
            </span>
          </button>
        ))}
      </CardContent>
    </Card>
  );
}

function durationStr(start: string, end: string): string {
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function RunStatusBadge({ status }: { status: JobStatus }) {
  const map: Record<JobStatus, { label: string; cls: string; icon?: React.ComponentType<{ className?: string }> }> = {
    queued:     { label: "queued",    cls: "bg-muted text-muted-foreground border-border" },
    running:    { label: "running",   cls: "bg-amber-50 text-amber-700 border-amber-200", icon: Radio },
    completed:  { label: "completed", cls: "bg-emerald-50 text-emerald-700 border-emerald-200", icon: CheckCircle2 },
    errored:    { label: "errored",   cls: "bg-rose-50 text-rose-700 border-rose-200", icon: XCircle },
    stopped:    { label: "stopped",   cls: "bg-slate-100 text-slate-600 border-slate-200", icon: StopCircle },
  };
  const m = map[status] ?? map.queued;
  const Icon = m.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 rounded-sm border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide",
        m.cls,
        status === "running" && "animate-pulse",
      )}
    >
      {Icon && <Icon className="h-2.5 w-2.5" />}
      {m.label}
    </span>
  );
}

function Sparkline({ history }: { history: Array<number | null> }) {
  const nums = history.filter((v): v is number => typeof v === "number");
  if (nums.length < 2) return <span className="h-3 w-10" />;
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const range = max - min || 1;
  const w = 40;
  const h = 12;
  const step = w / (nums.length - 1);
  const points = nums
    .map((v, i) => `${(i * step).toFixed(1)},${(h - ((v - min) / range) * h).toFixed(1)}`)
    .join(" ");
  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      className="shrink-0 text-foreground"
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth={1}
      />
    </svg>
  );
}

// ── detail pane ───────────────────────────────────────────────────────
function RunDetailPane({ slug, runId }: { slug: string; runId: string }) {
  const run = useRun(slug, runId);
  const events = useRunEvents(slug, runId);
  const kill = useKillRun(slug);
  const [selectedIter, setSelectedIter] = useState<number | null>(null);

  const iters = run.data?.iterations ?? [];
  const history = run.data?.primary_metric_history ?? [];
  const isActive = run.data?.status === "running" || run.data?.status === "queued";

  // Derive the freshest iteration list from (a) server snapshot and
  // (b) streaming events — events win for the live tail. We splice any
  // in-flight iter_started events into the server's list.
  const mergedIters = useMergedIterations(iters, events.events);

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-[220px_minmax(0,1fr)] lg:grid-cols-[220px_minmax(0,1fr)_240px]">
      <IterationTimeline
        iters={mergedIters}
        selected={selectedIter}
        onSelect={setSelectedIter}
      />

      <div className="flex min-h-0 flex-col gap-3">
        <RunHeader
          run={run.data}
          isActive={isActive}
          wsConnected={events.connected}
          onKill={() => {
            if (!run.data) return;
            const ok = window.confirm(
              `Stop run ${run.data.run_id}? The subprocess will be terminated.`,
            );
            if (!ok) return;
            kill.mutate(run.data.run_id, {
              onSuccess: () => toast.success("Kill signal sent"),
              onError: (err) => {
                const detail = err instanceof ApiError
                  ? err.problem.detail ?? err.problem.title
                  : err.message;
                toast.error("Could not kill run", { description: detail });
              },
            });
          }}
        />
        <LogViewer events={events.events} />
      </div>

      <div className="flex flex-col gap-3 md:col-span-full lg:col-span-1">
        <MetricChart history={history} />
        {isActive && <RunGpuCard />}
        {selectedIter !== null && (
          <IterationDetailCard
            iter={mergedIters.find((it) => it.iter_index === selectedIter) ?? null}
          />
        )}
        {run.data?.error && (
          <RunErrorCard
            slug={slug}
            error={run.data.error}
            classification={run.data.error_classification ?? null}
          />
        )}
      </div>
    </div>
  );
}

// ── Run-viewer GPU card (M7 Phase 7d) ────────────────────────────────
//
// Polls /system/gpu every 2 s while the run is active. Hidden when
// the host has no CUDA (CPU-only gym_sb3 runs) — no need to surface
// "no GPU" noise when the run isn't touching one.
function RunGpuCard() {
  const gpu = useSystemGpu({ refetchIntervalMs: 2000 });
  if (!gpu.data) return null;
  if (!gpu.data.cuda_available || gpu.data.devices.length === 0) return null;
  const dev = gpu.data.devices[0];
  const totalGb = dev.total_memory_bytes / (1024 ** 3);
  const usedBytes =
    typeof dev.used_memory_bytes === "number"
      ? dev.used_memory_bytes
      : dev.total_memory_bytes - dev.free_memory_bytes;
  const usedGb = usedBytes / (1024 ** 3);
  const memPct = totalGb > 0 ? (usedGb / totalGb) * 100 : 0;
  const util =
    typeof dev.utilization_percent === "number" ? dev.utilization_percent : null;
  const temp =
    typeof dev.temperature_c === "number" ? dev.temperature_c : null;

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="flex items-center gap-2 text-xs">
          <Cpu className="h-3.5 w-3.5" />
          GPU · {dev.name}
        </CardTitle>
        <CardDescription className="text-[10px]">
          live · polled every 2 s
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 pt-0 text-xs">
        <div>
          <div className="mb-0.5 flex items-baseline justify-between text-[10px] text-muted-foreground">
            <span>VRAM</span>
            <span className="font-mono">
              {usedGb.toFixed(2)} / {totalGb.toFixed(2)} GiB
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-sm bg-muted">
            <div
              className={cn(
                "h-full transition-all",
                memPct < 70
                  ? "bg-emerald-500"
                  : memPct < 90
                  ? "bg-amber-500"
                  : "bg-rose-500",
              )}
              style={{ width: `${Math.min(100, memPct)}%` }}
            />
          </div>
        </div>
        {util != null && (
          <div>
            <div className="mb-0.5 flex items-baseline justify-between text-[10px] text-muted-foreground">
              <span>utilization</span>
              <span className="font-mono">{util.toFixed(0)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-sm bg-muted">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${Math.min(100, util)}%` }}
              />
            </div>
          </div>
        )}
        {temp != null && (
          <div className="flex items-baseline justify-between text-[10px]">
            <span className="text-muted-foreground">temperature</span>
            <span
              className={cn(
                "font-mono",
                temp < 75
                  ? "text-muted-foreground"
                  : temp < 85
                  ? "text-amber-700"
                  : "text-rose-700",
              )}
            >
              {temp.toFixed(0)} °C
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RunHeader({
  run,
  isActive,
  wsConnected,
  onKill,
}: {
  run: RunDetail | undefined;
  isActive: boolean;
  wsConnected: boolean;
  onKill: () => void;
}) {
  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-3 px-4 py-3 text-xs">
        <span className="font-mono font-semibold">
          {run ? run.run_id.replace(/^job_/, "") : "…"}
        </span>
        {run && <RunStatusBadge status={run.status} />}
        {run && (
          <span className="truncate text-muted-foreground">
            <span className="text-foreground">{run.behavior_goal}</span>
          </span>
        )}
        <span className="ml-auto flex items-center gap-3 text-[10px] uppercase tracking-wide text-muted-foreground">
          <span className="flex items-center gap-1">
            <span
              className={cn(
                "inline-block h-2 w-2 rounded-full",
                wsConnected ? "bg-emerald-500" : "bg-rose-500",
              )}
            />
            ws {wsConnected ? "open" : "closed"}
          </span>
          {run && <span>iters {run.iterations_completed}/{run.iterations_requested}</span>}
        </span>
        {isActive && (
          <Button variant="destructive" size="sm" onClick={onKill}>
            <StopCircle />
            Kill
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// ── iteration timeline ────────────────────────────────────────────────
function IterationTimeline({
  iters,
  selected,
  onSelect,
}: {
  iters: IterEventSummary[];
  selected: number | null;
  onSelect: (n: number) => void;
}) {
  return (
    <Card className="max-h-[640px] overflow-y-auto scrollbar-thin">
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Timeline</CardTitle>
        <CardDescription className="text-[11px]">
          {iters.length} iter{iters.length === 1 ? "" : "s"}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-1 p-2 pt-0">
        {iters.length === 0 && (
          <p className="px-2 py-1 text-[11px] text-muted-foreground">
            no iterations yet
          </p>
        )}
        {iters.map((it) => (
          <button
            key={it.iter_index}
            type="button"
            onClick={() => onSelect(it.iter_index)}
            className={cn(
              "flex flex-col gap-1 rounded-md border px-2 py-2 text-left text-xs transition-colors",
              selected === it.iter_index
                ? "border-foreground/30 bg-accent"
                : "border-transparent hover:bg-accent/60",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono font-semibold">iter {it.iter_index}</span>
              {it.status === "running" ? (
                <Loader2 className="h-3 w-3 animate-spin text-amber-600" />
              ) : it.status === "completed" ? (
                <CheckCircle2 className="h-3 w-3 text-emerald-600" />
              ) : it.status === "errored" ? (
                <XCircle className="h-3 w-3 text-rose-600" />
              ) : null}
            </div>
            {it.status === "running" && typeof it.rl_total === "number" && it.rl_total > 0 && (
              <IterProgressBar
                rlIter={it.rl_iter ?? 0}
                rlTotal={it.rl_total}
                pct={it.pct ?? 0}
                etaS={it.eta_s ?? null}
              />
            )}
            {(it.reward_version_before !== null || it.reward_version_after !== null) && (
              <span className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
                v{it.reward_version_before ?? "?"}
                <ArrowRight className="h-2.5 w-2.5" />
                v{it.reward_version_after ?? "?"}
              </span>
            )}
            {it.primary_metric !== null && (
              <span className="font-mono text-[11px]">
                {it.primary_metric.toFixed(3)}{" "}
                {it.metric_delta !== null && (
                  <span
                    className={cn(
                      it.metric_delta > 0
                        ? "text-emerald-700"
                        : it.metric_delta < 0
                        ? "text-rose-700"
                        : "text-muted-foreground",
                    )}
                  >
                    ({it.metric_delta >= 0 ? "+" : ""}
                    {it.metric_delta.toFixed(3)})
                  </span>
                )}
              </span>
            )}
            {it.failure_modes.length > 0 && (
              <span className="truncate text-[10px] text-muted-foreground">
                {it.failure_modes.join(", ")}
              </span>
            )}
            {it.realism_audit &&
              typeof it.realism_audit.verdict === "string" &&
              it.realism_audit.verdict !== "ok" &&
              it.realism_audit.verdict !== "unknown" && (
                <span
                  className={cn(
                    "inline-flex w-fit items-center rounded px-1.5 py-0.5 text-[10px] font-semibold",
                    it.realism_audit.verdict === "severe"
                      ? "bg-rose-100 text-rose-800"
                      : "bg-amber-100 text-amber-800",
                  )}
                  title={`torque saturation: ${Math.round(
                    (it.realism_audit.torque_saturation_frac ?? 0) * 100,
                  )}% overall, worst joint ${Math.round(
                    (it.realism_audit.any_joint_saturation_max ?? 0) * 100,
                  )}%`}
                >
                  physics: {it.realism_audit.verdict}
                </span>
              )}
            {it.physics_edit_suggestion && it.physics_edit_suggestion.prompt && (
              (() => {
                // §Ship-8c hotfix: chip reflects sculpt-side auto-apply
                // progress. Disabled while sculpt is applying (prevents
                // click-race); shows a checkmark once applied.
                const state = it.physics_edit_suggestion.auto_apply_state;
                const disabled = state === "in_progress";
                const label =
                  state === "applied"
                    ? "physics auto-applied"
                    : state === "in_progress"
                    ? "applying physics fix…"
                    : state === "rejected" || state === "errored"
                    ? "physics fix failed — click to retry"
                    : "apply physics fix";
                const cls =
                  state === "applied"
                    ? "bg-emerald-100 text-emerald-800"
                    : state === "in_progress"
                    ? "bg-slate-100 text-slate-600 cursor-wait"
                    : state === "rejected" || state === "errored"
                    ? "bg-amber-100 text-amber-800 hover:bg-amber-200 cursor-pointer"
                    : "bg-indigo-100 text-indigo-800 hover:bg-indigo-200 cursor-pointer";
                return (
                  <span
                    onClick={(e) => {
                      e.stopPropagation();
                      if (disabled || state === "applied") return;
                      try {
                        sessionStorage.setItem(
                          "pendingPhysicsPrompt",
                          it.physics_edit_suggestion!.prompt,
                        );
                      } catch {
                        void navigator.clipboard?.writeText(
                          it.physics_edit_suggestion!.prompt,
                        );
                      }
                      const match = window.location.pathname.match(
                        /\/projects\/([^/]+)/,
                      );
                      if (match) {
                        window.location.assign(`/projects/${match[1]}/physics`);
                      }
                    }}
                    className={cn(
                      "inline-flex w-fit items-center rounded px-1.5 py-0.5 text-[10px] font-semibold",
                      cls,
                    )}
                    title={
                      it.physics_edit_suggestion.auto_apply_reason ??
                      it.physics_edit_suggestion.auto_apply_summary ??
                      "Open Physics tab with this auto-generated prompt pre-filled"
                    }
                  >
                    {label}
                  </span>
                );
              })()
            )}
          </button>
        ))}
      </CardContent>
    </Card>
  );
}

function RunErrorCard({
  slug,
  error,
  classification,
}: {
  slug: string;
  error: string;
  classification: ErrorClassification | null;
}) {
  const regen = useRegenerateRewardTemplate(slug);
  const isContractMismatch =
    classification?.kind === "reward_contract_mismatch"
    || classification?.action?.kind === "regenerate_reward_template";

  const onRegenerate = () => {
    regen.mutate(undefined, {
      onSuccess: () => {
        toast.success("Reward template regenerated", {
          description: "rewards/v0.py rewritten for this project's adapter.",
        });
      },
      onError: (err) => {
        const msg =
          err instanceof ApiError
            ? err.problem.detail ?? err.problem.title
            : (err as Error).message;
        toast.error("Could not regenerate template", { description: msg });
      },
    });
  };

  return (
    <Card className="border-rose-300/50 bg-rose-50">
      <CardHeader className="py-2">
        <CardTitle className="text-xs text-rose-800">
          {classification?.title ?? "Error"}
        </CardTitle>
        {classification?.detail && (
          <CardDescription className="text-[11px] text-rose-900/80">
            {classification.detail}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="space-y-2 pt-0 text-[11px] text-rose-900">
        <pre className="whitespace-pre-wrap break-words font-mono text-[10px] leading-snug opacity-90">
          {error}
        </pre>
        {classification?.suggestions && classification.suggestions.length > 0 && (
          <ul className="list-disc space-y-0.5 pl-4 text-[11px]">
            {classification.suggestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        )}
        {isContractMismatch && (
          <Button
            size="sm"
            variant="outline"
            onClick={onRegenerate}
            disabled={regen.isPending}
            className="mt-1 border-rose-300 text-rose-900 hover:bg-rose-100"
          >
            {regen.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            {regen.isPending ? "Regenerating…" : "Regenerate reward template"}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}


function IterationDetailCard({ iter }: { iter: IterEventSummary | null }) {
  if (!iter) return null;
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Iter {iter.iter_index}</CardTitle>
        <CardDescription className="text-[11px]">
          {iter.status}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-1 text-xs">
        {iter.failure_modes.length > 0 && (
          <div>
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              failure modes
            </span>
            <div className="mt-0.5 flex flex-wrap gap-1">
              {iter.failure_modes.map((f) => (
                <span
                  key={f}
                  className="rounded-sm border bg-muted/40 px-1 py-0.5 font-mono text-[10px]"
                >
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}
        {iter.edit_count !== null && (
          <div className="flex justify-between">
            <span className="text-muted-foreground">edits</span>
            <span>{iter.edit_count}</span>
          </div>
        )}
        {iter.paper_refs.length > 0 && (
          <div>
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              paper refs
            </span>
            <ul className="mt-0.5 space-y-0.5">
              {iter.paper_refs.map((r) => (
                <li key={r} className="font-mono text-[11px]">
                  <a
                    href={`https://arxiv.org/abs/${r}`}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="hover:underline"
                  >
                    {r}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Status-rank for monotonic iter transitions. An iter can only move
// forward: queued → running → completed/errored. A later-arriving
// REST snapshot that happens to see status="running" (e.g. because
// the fs watcher on the backend hasn't seen the completion event yet)
// must NOT overwrite an already-completed slot. Issue E (Test 1).
const _ITER_STATUS_RANK: Record<string, number> = {
  queued: 0,
  running: 1,
  completed: 2,
  errored: 2,
};

function _mergeIterSlot(
  prev: IterEventSummary | undefined,
  next: IterEventSummary,
): IterEventSummary {
  if (!prev) return next;
  const prevRank = _ITER_STATUS_RANK[prev.status] ?? 0;
  const nextRank = _ITER_STATUS_RANK[next.status] ?? 0;
  // Keep the further-along status; fill in any null fields from the
  // losing side so we don't lose data (e.g. REST has completed_at but
  // the WS-derived slot doesn't).
  const winner = nextRank >= prevRank ? next : prev;
  const loser = nextRank >= prevRank ? prev : next;
  return {
    ...loser,
    ...winner,
    // Null-preservation merge for every optional field — winner wins
    // on non-null, loser fills in on null. Avoids a completed REST
    // snapshot overwriting paper_refs from a WS `edit_applied` event.
    started_at: winner.started_at ?? loser.started_at,
    completed_at: winner.completed_at ?? loser.completed_at,
    reward_version_before: winner.reward_version_before ?? loser.reward_version_before,
    reward_version_after: winner.reward_version_after ?? loser.reward_version_after,
    primary_metric: winner.primary_metric ?? loser.primary_metric,
    metric_delta: winner.metric_delta ?? loser.metric_delta,
    failure_modes: winner.failure_modes.length ? winner.failure_modes : loser.failure_modes,
    edit_count: winner.edit_count ?? loser.edit_count,
    paper_refs: winner.paper_refs.length ? winner.paper_refs : loser.paper_refs,
    rollout_ready: winner.rollout_ready || loser.rollout_ready,
    diagnosed: winner.diagnosed || loser.diagnosed,
    realism_audit: winner.realism_audit ?? loser.realism_audit,
    physics_edit_suggestion:
      winner.physics_edit_suggestion ?? loser.physics_edit_suggestion,
  };
}

// ── utility: merge REST iterations with in-flight WS events ──────────
// Uses a useRef-backed Map that PERSISTS across renders so iters never
// vanish when REST transiently returns fewer rows than previously seen
// (Issue E from Test 1 2026-04-22: Sam saw "iter 1 disappeared after
// iter 0 finished, reappeared when iter 2 started"). Any iter ever
// known to exist is kept; later updates merge via _mergeIterSlot which
// enforces status-rank monotonicity.
function useMergedIterations(
  rest: IterEventSummary[],
  events: RunEvent[],
): IterEventSummary[] {
  const stickyMap = useRef<Map<number, IterEventSummary>>(new Map());
  return useMemo(() => {
    const map = stickyMap.current;
    // First pass: REST snapshots. These are authoritative on the
    // fields the fs watcher sees (completed_at, primary_metric from
    // metrics.json, etc.) and get monotonic-merged into the sticky
    // map so a transiently-empty REST doesn't evict known rows.
    for (const r of rest) {
      map.set(r.iter_index, _mergeIterSlot(map.get(r.iter_index), r));
    }
    // Second pass: WS events. Produce a per-iter slot from the events
    // for that iter, then monotonic-merge into the sticky map.
    const eventSlots = new Map<number, IterEventSummary>();
    for (const ev of events) {
      const iter = (ev as { iter?: number }).iter;
      if (typeof iter !== "number") continue;
      const slot: IterEventSummary = eventSlots.get(iter) ?? map.get(iter) ?? {
        iter_index: iter,
        status: "running",
        started_at: null,
        completed_at: null,
        reward_version_before: null,
        reward_version_after: null,
        primary_metric: null,
        metric_delta: null,
        failure_modes: [],
        edit_count: null,
        paper_refs: [],
        rollout_ready: false,
        diagnosed: false,
        realism_audit: null,
        physics_edit_suggestion: null,
      };
      if (ev.type === "iter_started") {
        slot.started_at = slot.started_at ?? ev.ts;
        if (ev.reward_version_before !== undefined) {
          slot.reward_version_before = Number(ev.reward_version_before);
        }
      }
      if (ev.type === "rollout_done") slot.rollout_ready = true;
      if (ev.type === "physics_edit_suggested") {
        const prompt = (ev as { prompt?: unknown }).prompt;
        if (typeof prompt === "string" && prompt.trim().length > 0) {
          slot.physics_edit_suggestion = {
            prompt,
            verdict: typeof ev.verdict === "string" ? ev.verdict : null,
            top_joints_saturation: Array.isArray(ev.top_joints_saturation)
              ? (ev.top_joints_saturation as Array<{ name: string; value: number }>)
              : [],
          };
        }
      }
      if (ev.type === "realism_audited") {
        const audit = (ev as { audit?: Record<string, unknown> }).audit;
        if (audit && typeof audit === "object") {
          slot.realism_audit = audit as IterEventSummary["realism_audit"];
        } else if (slot.realism_audit == null) {
          slot.realism_audit = {
            verdict: typeof ev.verdict === "string" ? ev.verdict : "unknown",
            torque_saturation_frac:
              typeof ev.torque_saturation_frac === "number"
                ? ev.torque_saturation_frac
                : null,
            any_joint_saturation_max:
              typeof ev.any_joint_saturation_max === "number"
                ? ev.any_joint_saturation_max
                : null,
            joint_vel_p99_max:
              typeof ev.joint_vel_p99_max === "number" ? ev.joint_vel_p99_max : null,
            joint_limit_violation_frac:
              typeof ev.joint_limit_violation_frac === "number"
                ? ev.joint_limit_violation_frac
                : null,
            top_joints_saturation: Array.isArray(ev.top_joints_saturation)
              ? (ev.top_joints_saturation as Array<{ name: string; value: number }>)
              : [],
          };
        }
      }
      if (ev.type === "diagnosed") {
        slot.diagnosed = true;
        slot.failure_modes = Array.isArray(ev.failure_modes)
          ? (ev.failure_modes as string[])
          : slot.failure_modes;
      }
      if (ev.type === "edit_applied") {
        if (ev.reward_version_after !== undefined) {
          slot.reward_version_after = Number(ev.reward_version_after);
        }
        if (ev.reward_version_before !== undefined && slot.reward_version_before === null) {
          slot.reward_version_before = Number(ev.reward_version_before);
        }
        if (Array.isArray(ev.paper_refs)) {
          slot.paper_refs = ev.paper_refs as string[];
        }
      }
      if (ev.type === "iter_completed") {
        slot.status = "completed";
        slot.completed_at = ev.ts;
        if (typeof ev.primary_metric === "number") slot.primary_metric = ev.primary_metric;
        if (typeof ev.metric_delta === "number") slot.metric_delta = ev.metric_delta;
        if (typeof ev.edit_count === "number") slot.edit_count = ev.edit_count;
        if (Array.isArray(ev.paper_refs)) slot.paper_refs = ev.paper_refs as string[];
        if (Array.isArray(ev.failure_modes)) slot.failure_modes = ev.failure_modes as string[];
      }
      if (ev.type === "iter_progress") {
        if (typeof ev.rl_iter === "number") slot.rl_iter = ev.rl_iter;
        if (typeof ev.rl_total === "number") slot.rl_total = ev.rl_total;
        if (typeof ev.pct === "number") slot.pct = ev.pct;
        if (typeof ev.elapsed_s === "number") slot.elapsed_s = ev.elapsed_s;
        if (typeof ev.eta_s === "number" || ev.eta_s === null) {
          slot.eta_s = ev.eta_s as number | null;
        }
      }
      eventSlots.set(iter, slot);
    }
    for (const [iter, slot] of eventSlots) {
      map.set(iter, _mergeIterSlot(map.get(iter), slot));
    }
    return Array.from(map.values()).sort((a, b) => a.iter_index - b.iter_index);
  }, [rest, events]);
}


function IterProgressBar({
  rlIter,
  rlTotal,
  pct,
  etaS,
}: {
  rlIter: number;
  rlTotal: number;
  pct: number;
  etaS: number | null;
}) {
  const clamped = Math.max(0, Math.min(100, pct));
  const etaLabel =
    etaS == null
      ? ""
      : etaS < 60
      ? ` · ETA ${Math.round(etaS)}s`
      : etaS < 3600
      ? ` · ETA ${Math.round(etaS / 60)}m`
      : ` · ETA ${(etaS / 3600).toFixed(1)}h`;
  return (
    <div className="flex flex-col gap-0.5">
      <div className="h-1 overflow-hidden rounded-sm bg-amber-100 dark:bg-amber-950/40">
        <div
          className="h-full bg-amber-500 transition-[width] duration-500 ease-out"
          style={{ width: `${clamped}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-muted-foreground">
        {rlIter}/{rlTotal} ({clamped.toFixed(1)}%){etaLabel}
      </span>
    </div>
  );
}

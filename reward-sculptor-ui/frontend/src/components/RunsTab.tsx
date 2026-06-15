import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Badge, Btn, Delta, EmptyState, MetricChart, Sparkline } from "@/components/rs/primitives";
import { LogViewer } from "@/components/LogViewer";
import { NewRunDialog } from "@/components/NewRunDialog";
import { NewMissionDialog } from "@/components/NewMissionDialog";
import { MissionDetailDialog } from "@/components/MissionDetailDialog";
import { useSystemGpu } from "@/hooks/useLibrary";
import { useRunEvents } from "@/hooks/useRunEvents";
import { useMissions } from "@/hooks/useMissions";
import { useRegenerateRewardTemplate, useRewards } from "@/hooks/useRewards";
import { useControlRun, useKillRun, useRun, useRuns } from "@/hooks/useRuns";
import { ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/utils";
import type {
  ErrorClassification,
  IterEventSummary,
  MissionSummary,
  ProjectDetail,
  RunDetail,
  RunEvent,
  RunSummary,
} from "@/lib/types";

// ── public entry ──────────────────────────────────────────────────────
export default function RunsTab({ slug, project }: { slug: string; project: ProjectDetail }) {
  const missions = useMissions(slug);
  // §Ship 21d: keep /runs polling through stage boundaries while a mission
  // is active (preserved verbatim).
  const missionActive = useMemo(
    () => (missions.data ?? []).some((m) => m.active_job_id != null),
    [missions.data],
  );
  const list = useRuns(slug, { keepPolling: missionActive });
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [missionDialogSlug, setMissionDialogSlug] = useState<string | null>(null);

  const runs = list.data ?? [];
  const { sculptRuns, missionGroups } = useMemo(
    () => partitionRuns(runs, missions.data ?? []),
    [runs, missions.data],
  );
  const allOrderedRunIds = useMemo(
    () => [
      ...sculptRuns.map((r) => r.run_id),
      ...missionGroups.flatMap((g) => g.stages.map((r) => r.run_id)),
    ],
    [sculptRuns, missionGroups],
  );
  const selected = selectedRunId ?? allOrderedRunIds[0] ?? null;
  const missionDialogSummary =
    missionDialogSlug != null
      ? (missions.data ?? []).find((m) => m.mission_slug === missionDialogSlug) ?? null
      : null;

  const empty = !list.isLoading && sculptRuns.length === 0 && missionGroups.length === 0;

  if (list.isLoading) {
    return <div className="rs-scroll"><div className="rs-pad"><p className="rs-sub">Loading runs…</p></div></div>;
  }
  if (empty) {
    return (
      <div className="rs-scroll">
        <div className="rs-pad">
          <div className="rs-flex-between rs-wrap rs-gap-12" style={{ marginBottom: 16 }}>
            <h2 className="rs-h2">Runs</h2>
            <div className="rs-flex rs-gap-8">
              <NewMissionDialog slug={slug} onCreated={(s) => setMissionDialogSlug(s)} />
              <NewRunDialog slug={slug} project={project} onLaunched={(id) => setSelectedRunId(id)} />
            </div>
          </div>
          <div className="rs-card">
            <EmptyState
              icon="activity"
              title="No runs yet"
              sub="Launch a single training run with New run, or decompose a complex goal into a curriculum with New mission."
            />
          </div>
        </div>
        <MissionDetailDialog slug={slug} missionSlug={missionDialogSlug} summary={missionDialogSummary} open={missionDialogSlug != null} onOpenChange={(o) => { if (!o) setMissionDialogSlug(null); }} />
      </div>
    );
  }

  return (
    <div style={{ flex: 1, minHeight: 0, overflow: "hidden", display: "flex" }}>
      <div className="rs-runs-layout">
        <RunSidebar
          slug={slug}
          project={project}
          sculptRuns={sculptRuns}
          missionGroups={missionGroups}
          selected={selected}
          onSelectRun={setSelectedRunId}
          onOpenMissionDialog={setMissionDialogSlug}
          onLaunchedRun={(id) => setSelectedRunId(id)}
        />
        {selected ? (
          <RunDetailPane slug={slug} runId={selected} runs={runs} />
        ) : (
          <div className="rs-flex" style={{ justifyContent: "center", alignItems: "center", color: "var(--rs-muted)", fontSize: 13 }}>
            Select a run.
          </div>
        )}
      </div>
      <MissionDetailDialog slug={slug} missionSlug={missionDialogSlug} summary={missionDialogSummary} open={missionDialogSlug != null} onOpenChange={(o) => { if (!o) setMissionDialogSlug(null); }} />
    </div>
  );
}

// ── logic (preserved verbatim) ───────────────────────────────────────
interface MissionGroup {
  missionSlug: string;
  mission: MissionSummary | null;
  stages: RunSummary[];
}

function partitionRuns(runs: RunSummary[], missions: MissionSummary[]): { sculptRuns: RunSummary[]; missionGroups: MissionGroup[] } {
  const sculptRuns: RunSummary[] = [];
  const stagesByMission = new Map<string, RunSummary[]>();
  for (const r of runs) {
    if (r.kind === "mission_stage_run" && r.mission_slug) {
      const arr = stagesByMission.get(r.mission_slug);
      if (arr) arr.push(r);
      else stagesByMission.set(r.mission_slug, [r]);
    } else {
      sculptRuns.push(r);
    }
  }
  const missionsBySlug = new Map<string, MissionSummary>(missions.map((m) => [m.mission_slug, m]));
  const seen = new Set<string>();
  const missionGroups: MissionGroup[] = [];
  for (const m of missions) {
    const stages = stagesByMission.get(m.mission_slug);
    if (!stages || stages.length === 0) continue;
    seen.add(m.mission_slug);
    stages.sort((a, b) => (a.stage_index ?? 0) - (b.stage_index ?? 0) || a.run_id.localeCompare(b.run_id));
    missionGroups.push({ missionSlug: m.mission_slug, mission: m, stages });
  }
  for (const [missionSlug, stages] of stagesByMission) {
    if (seen.has(missionSlug)) continue;
    stages.sort((a, b) => (a.stage_index ?? 0) - (b.stage_index ?? 0) || a.run_id.localeCompare(b.run_id));
    missionGroups.push({ missionSlug, mission: missionsBySlug.get(missionSlug) ?? null, stages });
  }
  return { sculptRuns, missionGroups };
}

function missionRunStateLabel(m: MissionSummary): string {
  const { current_stage_idx: i, n_stages: n, lifecycle } = m;
  if (n === 0) return "Planning…";
  if (lifecycle === "running") return `Stage ${Math.max(1, i + 1)} of ${n}`;
  if (lifecycle === "ready") return `${n} stages planned`;
  if (lifecycle === "completed") return `${n} of ${n} stages complete`;
  return `${i} of ${n} stages complete`;
}

function durationStr(start: string, end: string): string {
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// ── sidebar ───────────────────────────────────────────────────────────
function RunSidebar({
  slug, project, sculptRuns, missionGroups, selected, onSelectRun, onOpenMissionDialog, onLaunchedRun,
}: {
  slug: string;
  project: ProjectDetail;
  sculptRuns: RunSummary[];
  missionGroups: MissionGroup[];
  selected: string | null;
  onSelectRun: (id: string) => void;
  onOpenMissionDialog: (missionSlug: string) => void;
  onLaunchedRun: (id: string) => void;
}) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const toggle = (s: string) => setCollapsed((st) => ({ ...st, [s]: !st[s] }));

  return (
    <div className="rs-runs-side">
      <div className="rs-side-head">
        <span className="rs-h3" style={{ fontSize: 15 }}>Runs</span>
        <div className="rs-flex rs-gap-6">
          <NewMissionDialog slug={slug} onCreated={(s) => onOpenMissionDialog(s)} />
          <NewRunDialog slug={slug} project={project} onLaunched={onLaunchedRun} />
        </div>
      </div>

      {missionGroups.length > 0 && <div className="rs-side-group">Missions</div>}
      {missionGroups.map((g) => {
        const isCollapsed = collapsed[g.missionSlug] ?? false;
        const selectedInGroup = g.stages.some((s) => s.run_id === selected);
        const open = !isCollapsed || selectedInGroup;
        return (
          <div key={g.missionSlug} className="rs-mission">
            <div className="rs-mhead" role="button" tabIndex={0}
              onClick={() => toggle(g.missionSlug)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(g.missionSlug); } }}
              aria-expanded={open}
            >
              <Icon name={open ? "chevron-down" : "chevron-right"} size={15} color="var(--rs-muted)" />
              <Icon name="sparkles" size={15} color="var(--rs-primary)" />
              <span style={{ flex: 1, minWidth: 0, overflow: "hidden" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  {g.mission && <Badge status={g.mission.lifecycle} label="" />}
                  <span style={{ fontWeight: 500, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {g.mission ? missionRunStateLabel(g.mission) : g.missionSlug}
                  </span>
                </span>
                {g.mission?.goal && (
                  <span className="rmeta" style={{ display: "block", marginTop: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={g.mission.goal}>
                    {g.mission.goal}
                  </span>
                )}
              </span>
              <button
                className="rs-iconbtn"
                style={{ width: 26, height: 26 }}
                aria-label="Open mission curriculum and run controls"
                title="Plan: decomposition rationale, stages, Run/Delete"
                onClick={(e) => { e.stopPropagation(); onOpenMissionDialog(g.missionSlug); }}
              >
                <Icon name="list" size={14} />
              </button>
            </div>
            {open && g.stages.map((r) => (
              <RunRow key={r.run_id} run={r} selected={selected === r.run_id} onSelect={() => onSelectRun(r.run_id)} stageContext />
            ))}
          </div>
        );
      })}

      {sculptRuns.length > 0 && <div className="rs-side-group">Single runs</div>}
      {sculptRuns.map((r) => (
        <RunRow key={r.run_id} run={r} selected={selected === r.run_id} onSelect={() => onSelectRun(r.run_id)} />
      ))}
    </div>
  );
}

function RunRow({
  run: r, selected, onSelect, stageContext = false,
}: { run: RunSummary; selected: boolean; onSelect: () => void; stageContext?: boolean }) {
  const titleText = stageContext ? r.stage_name ?? r.run_id.replace(/^job_/, "") : r.run_id.replace(/^job_/, "");
  const itersDenom = r.iterations_requested || "?";
  return (
    <button className={"rs-runrow" + (selected ? " on" : "") + (stageContext ? " rs-stage" : "")} onClick={onSelect}>
      <Badge status={r.status} label="" />
      <span style={{ minWidth: 0, flex: 1 }}>
        <span className="rid" style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {stageContext && typeof r.stage_index === "number" && <span style={{ color: "var(--rs-muted)" }}>{r.stage_index + 1}. </span>}
          {titleText}
        </span>
        <span className="rmeta" style={{ display: "block" }}>
          {r.iterations_completed}/{itersDenom}
          {r.started_at ? ` · ${formatRelative(r.started_at)}` : ""}
          {r.started_at && r.ended_at ? ` · ${durationStr(r.started_at, r.ended_at)}` : ""}
        </span>
      </span>
      <Sparkline
        // §Ship 35: foreground objective fitness when the run has it.
        data={(r.fitness_history && r.fitness_history.some((v) => typeof v === "number"))
          ? r.fitness_history
          : r.primary_metric_history}
        w={46}
        h={20}
        color={r.status === "errored" ? "var(--st-rose)" : r.status === "running" ? "var(--st-amber)" : "var(--st-emerald)"}
      />
    </button>
  );
}

// ── detail pane ───────────────────────────────────────────────────────
function RunDetailPane({ slug, runId, runs }: { slug: string; runId: string; runs: RunSummary[] }) {
  const run = useRun(slug, runId);
  const events = useRunEvents(slug, runId);
  const kill = useKillRun(slug);
  const control = useControlRun(slug);
  const [selectedIter, setSelectedIter] = useState<number | null>(null);

  const iters = run.data?.iterations ?? [];
  const history = run.data?.primary_metric_history ?? [];
  const isActive = run.data?.status === "running" || run.data?.status === "queued";

  // §Ship 39 (H1): derive interactive state (paused-awaiting-feedback + the
  // current mode) from the event stream, seeded by the run's persisted mode
  // so a reconnect restores the Auto/Manual switch.
  const { awaiting, awaitingIter, mode } = useMemo(() => {
    let aw = false;
    let awIter: number | null = null;
    let m: "manual" | "auto" = run.data?.mode === "manual" ? "manual" : "auto";
    for (const ev of events.events) {
      const e = ev as { type?: string; mode?: string; iter?: number };
      if (e.type === "run_control_updated" && (e.mode === "manual" || e.mode === "auto")) m = e.mode;
      else if (e.type === "awaiting_feedback") { aw = true; awIter = typeof e.iter === "number" ? e.iter : null; }
      else if (e.type === "feedback_resumed" || e.type === "feedback_timeout" || e.type === "iter_started" || e.type === "run_completed" || e.type === "run_errored" || e.type === "run_stopped" || e.type === "early_stop") aw = false;
    }
    return { awaiting: aw, awaitingIter: awIter, mode: m };
  }, [events.events, run.data?.mode]);

  // §Ship 43: launch-time objective-metric generation as run-phase 0. Fold the
  // metric_generation_* events into a single phase view (progress while running;
  // accepted / rejected-with-reasons outcome — never silent).
  const genPhase = useMemo(() => {
    let started = false;
    let last: { stage?: string; message?: string; attempt?: number; max?: number } | null = null;
    let outcome: "accepted" | "rejected" | "failed" | null = null;
    let genId: string | null = null;
    let reasons: string[] = [];
    let concerns: string[] = [];
    let errorMsg: string | null = null;
    // §Ship 45: paused awaiting a one-click retry/continue decision.
    let awaiting = false;
    // §Ship 44: launch-time calibration outcome.
    let calib: {
      status: "running" | "done" | "skipped";
      builtin?: string; calibrated?: boolean; spearman?: number | null;
    } | null = null;
    for (const ev of events.events) {
      const e = ev as {
        type?: string; stage?: string; message?: string; attempt?: number;
        max?: number; gen_id?: string; reasons?: string[]; concerns?: string[];
        error?: string; builtin?: string; calibrated?: boolean; spearman?: number | null;
      };
      if (e.type === "metric_generation_started") { started = true; outcome = null; last = null; calib = null; awaiting = false; }
      else if (e.type === "metric_generation_progress") {
        last = { stage: e.stage, message: e.message, attempt: e.attempt, max: e.max };
      }
      else if (e.type === "metric_generated") { outcome = "accepted"; genId = e.gen_id ?? null; awaiting = false; }
      else if (e.type === "metric_generation_rejected") {
        outcome = "rejected"; genId = e.gen_id ?? null;
        reasons = e.reasons ?? []; concerns = e.concerns ?? [];
      }
      else if (e.type === "metric_generation_awaiting_decision") { awaiting = true; }
      else if (e.type === "metric_generation_failed") { outcome = "failed"; errorMsg = e.error ?? null; awaiting = false; }
      else if (e.type === "metric_calibration_started") { calib = { status: "running", builtin: e.builtin }; }
      else if (e.type === "metric_calibration_done") {
        calib = { status: "done", builtin: e.builtin, calibrated: e.calibrated, spearman: e.spearman };
      }
      else if (e.type === "metric_calibration_skipped") { calib = { status: "skipped" }; }
    }
    return started ? { last, outcome, genId, reasons, concerns, errorMsg, calib, awaiting } : null;
  }, [events.events]);

  const summary = useMemo(() => runs.find((r) => r.run_id === runId) ?? null, [runs, runId]);
  const isStageRun = (summary?.kind === "mission_stage_run") || (run.data?.kind === "mission_stage_run");
  const missionSlug = summary?.mission_slug ?? run.data?.mission_slug ?? null;
  const stageName = summary?.stage_name ?? run.data?.stage_name ?? null;
  const stageRewardsScope = isStageRun && missionSlug && stageName ? `${missionSlug}/${stageName}` : null;

  const mergedIters = useMergedIterations(iters, events.events);
  const isPending = run.data?.status === "queued" && mergedIters.length === 0;

  // §Ship 35: fitness is the PRIMARY tracked metric when an objective
  // fitness exists (observe or steer). Prefer the per-run history; fall
  // back to deriving from the merged iterations (covers live runs before
  // the REST summary refreshes). Degrade to the reward metric otherwise.
  const fitnessHistory: Array<number | null> =
    (run.data?.fitness_history && run.data.fitness_history.length
      ? run.data.fitness_history
      : mergedIters.map((it) => (typeof it.fitness === "number" ? it.fitness : null)));
  const hasFitness = fitnessHistory.some((v) => typeof v === "number");

  return (
    <div className="rs-runs-detail">
      {isPending ? (
        <div className="rs-iter-col">
          <div className="rs-eyebrow" style={{ marginBottom: 12 }}>Iterations</div>
          <EmptyState icon="clock" title="Not started" sub="This run is queued. Iterations appear once training begins." />
        </div>
      ) : (
        <IterationTimeline iters={mergedIters} selected={selectedIter} onSelect={setSelectedIter} />
      )}

      <div className="rs-mid-col">
        {isStageRun && summary && <StageContextCard run={summary} />}
        <RunHeader
          run={run.data}
          isActive={isActive}
          wsConnected={events.connected}
          mode={mode}
          togglePending={control.isPending}
          onToggleMode={() => {
            if (!run.data) return;
            const next = mode === "manual" ? "auto" : "manual";
            control.mutate({ runId: run.data.run_id, mode: next }, {
              onSuccess: () => toast.success(next === "manual" ? "Manual: pausing each iteration for your feedback" : "Auto: running straight through"),
              onError: (err) => toast.error("Could not change mode", { description: err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message }),
            });
          }}
          onKill={() => {
            if (!run.data) return;
            const ok = window.confirm(`Stop run ${run.data.run_id}? The subprocess will be terminated.`);
            if (!ok) return;
            kill.mutate(run.data.run_id, {
              onSuccess: () => toast.success("Kill signal sent"),
              onError: (err) => {
                const detail = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
                toast.error("Could not kill run", { description: detail });
              },
            });
          }}
        />
        {isActive && awaiting && run.data && (
          <FeedbackPanel
            iterIndex={awaitingIter}
            pending={control.isPending}
            onSubmit={(text, goAuto) => {
              if (!run.data) return;
              control.mutate(
                { runId: run.data.run_id, resume: true, feedback: text || null, ...(goAuto ? { mode: "auto" as const } : {}) },
                { onError: (err) => toast.error("Could not continue", { description: err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message }) },
              );
            }}
          />
        )}
        {genPhase && (
          <div style={{ padding: "0 16px" }}>
            <MetricGenPhase
              phase={genPhase}
              busy={control.isPending}
              onRetry={() => control.mutate({ runId, gen_retry: true }, {
                onError: (err) => toast.error("Could not retry generation", { description: err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message }),
              })}
              onContinueBlind={() => control.mutate({ runId, gen_continue: true }, {
                onError: (err) => toast.error("Could not continue", { description: err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message }),
              })}
            />
          </div>
        )}
        {run.data?.error && (
          <div style={{ padding: "0 16px" }}>
            <RunErrorCard slug={slug} error={run.data.error} classification={run.data.error_classification ?? null} />
          </div>
        )}
        <LogViewer events={events.events} />
      </div>

      <div className="rs-extra-col">
        <div className="rs-card">
          <div className="rs-card-head"><div className="rs-card-title" style={{ fontSize: 13 }}><Icon name="trending-up" size={15} />{hasFitness ? "Objective fitness" : "Mean reward"}</div>{isActive && <span className="rs-dot live" />}</div>
          <div style={{ padding: "8px 4px 4px" }}>
            <MetricChart
              data={hasFitness ? fitnessHistory : history}
              live={isActive}
              label={hasFitness ? "Objective fitness per iteration" : "Mean reward per iteration"}
              decimals={hasFitness ? 2 : 1}
            />
          </div>
          {/* §Ship 35: reward metric demoted to a secondary sparkline when
              fitness is the primary signal. */}
          {hasFitness && history.some((v) => typeof v === "number") && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "2px 10px 8px", fontSize: 11, color: "var(--rs-muted)" }}>
              <span>reward</span>
              <Sparkline data={history} w={120} h={16} color="var(--rs-muted)" />
            </div>
          )}
        </div>
        {stageRewardsScope && <StageRewardsCard slug={slug} stage={stageRewardsScope} />}
        {isActive && <RunGpuCard />}
        {selectedIter !== null && <IterationDetailCard iter={mergedIters.find((it) => it.iter_index === selectedIter) ?? null} />}
      </div>
    </div>
  );
}

function StageContextCard({ run }: { run: RunSummary }) {
  return (
    <div className="rs-card rs-card-pad" style={{ marginBottom: 0 }}>
      <div className="rs-flex rs-wrap rs-gap-8" style={{ fontSize: 12 }}>
        <Icon name="sparkles" size={14} color="var(--rs-primary)" />
        <span style={{ fontWeight: 500 }}>
          {typeof run.stage_index === "number" ? `Stage ${run.stage_index + 1}: ` : "Stage: "}
          <code className="mono">{run.stage_name ?? "(unnamed)"}</code>
        </span>
        <span style={{ color: "var(--rs-muted)" }}>mission <code className="mono">{run.mission_slug}</code></span>
        {run.behavior_goal && <span style={{ color: "var(--rs-muted)" }}>{run.behavior_goal}</span>}
      </div>
    </div>
  );
}

function StageRewardsCard({ slug, stage }: { slug: string; stage: string }) {
  const versions = useRewards(slug, stage);
  return (
    <div className="rs-card">
      <div className="rs-card-head"><div className="rs-card-title" style={{ fontSize: 13 }}><Icon name="file-code" size={15} />Stage rewards</div></div>
      <div className="rs-verlist">
        {versions.isLoading && <p className="rs-sub" style={{ padding: "10px 14px", fontSize: 11 }}>Loading…</p>}
        {versions.error && <p style={{ padding: "10px 14px", fontSize: 11, color: "var(--st-rose)" }}>{(versions.error as Error).message}</p>}
        {versions.data && versions.data.length === 0 && <p className="rs-sub" style={{ padding: "10px 14px", fontSize: 11 }}>No reward versions yet.</p>}
        {versions.data?.map((v, i) => (
          <div key={v.version} className="rs-verrow" style={{ cursor: "default" }}>
            <span className="vn">v{v.version}</span>
            <span className="rs-grow" />
            {typeof v.primary_metric === "number" && <span className="rs-num" style={{ fontSize: 13 }}>{v.primary_metric.toFixed(1)}</span>}
            {i > 0 && typeof v.metric_delta === "number" && <Delta value={v.metric_delta} />}
          </div>
        ))}
      </div>
    </div>
  );
}

function RunGpuCard() {
  const gpu = useSystemGpu({ refetchIntervalMs: 2000 });
  if (!gpu.data || !gpu.data.cuda_available || gpu.data.devices.length === 0) return null;
  const dev = gpu.data.devices[0];
  const totalGb = dev.total_memory_bytes / (1024 ** 3);
  const usedBytes = typeof dev.used_memory_bytes === "number" ? dev.used_memory_bytes : dev.total_memory_bytes - dev.free_memory_bytes;
  const usedGb = usedBytes / (1024 ** 3);
  const memPct = totalGb > 0 ? (usedGb / totalGb) * 100 : 0;
  const util = typeof dev.utilization_percent === "number" ? dev.utilization_percent : null;
  const temp = typeof dev.temperature_c === "number" ? dev.temperature_c : null;
  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-card-title" style={{ fontSize: 13, marginBottom: 12 }}><Icon name="cpu" size={15} />GPU</div>
      <div className="rs-vgap-8" style={{ fontSize: 12.5 }}>
        <div className="rs-flex-between"><span className="rs-sub" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{dev.name.replace(/^NVIDIA\s+/, "")}</span>{temp != null && <span className="rs-num">{Math.round(temp)}°C</span>}</div>
        <div className="rs-vram"><i style={{ width: `${Math.min(100, memPct)}%`, background: memPct < 70 ? "var(--st-emerald)" : memPct < 90 ? "var(--st-amber)" : "var(--st-rose)" }} /></div>
        <div className="rs-flex-between"><span className="rs-sub">VRAM</span><span className="rs-num">{usedGb.toFixed(1)} / {totalGb.toFixed(1)} GB</span></div>
        {util != null && (
          <>
            <div className="rs-vram"><i style={{ width: `${Math.min(100, util)}%`, background: "var(--st-blue)" }} /></div>
            <div className="rs-flex-between"><span className="rs-sub">utilization</span><span className="rs-num">{util.toFixed(0)}%</span></div>
          </>
        )}
      </div>
    </div>
  );
}

function RunHeader({ run, isActive, wsConnected, mode, onToggleMode, togglePending, onKill }: { run: RunDetail | undefined; isActive: boolean; wsConnected: boolean; mode: "manual" | "auto"; onToggleMode: () => void; togglePending: boolean; onKill: () => void }) {
  return (
    <div className="rs-run-header">
      <Icon name="activity" size={17} color="var(--rs-muted)" />
      <span className="mono" style={{ fontSize: 14, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>
        {run ? run.run_id.replace(/^job_/, "") : "…"}
      </span>
      {run && <Badge status={run.status} />}
      {run && <span className="rs-sub" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>{run.behavior_goal}</span>}
      <span className="rs-grow" />
      <span className="rs-flex rs-gap-8 rs-eyebrow" style={{ flexShrink: 0 }}>
        <span className="rs-flex rs-gap-6"><span className="rs-dot" style={{ background: wsConnected ? "var(--st-emerald)" : "var(--st-rose)" }} />ws {wsConnected ? "open" : "closed"}</span>
        {run && <span>iters {run.iterations_completed}/{run.iterations_requested}</span>}
      </span>
      {/* §Ship 39 (H1): the big Auto/Manual switch — flip at ANY point. */}
      {isActive && (
        <Btn
          kind={mode === "manual" ? "primary" : "quiet"}
          size="sm"
          icon={mode === "manual" ? "clock" : "activity"}
          disabled={togglePending}
          onClick={onToggleMode}
          title={mode === "manual"
            ? "Manual: pausing for your feedback each iteration. Click to run on auto."
            : "Auto: running straight through. Click to pause for feedback each iteration."}
        >
          {mode === "manual" ? "Manual" : "Auto"}
        </Btn>
      )}
      {isActive && <Btn kind="danger" size="sm" icon="square" onClick={onKill}>Stop</Btn>}
    </div>
  );
}

// §Ship 39 (H1): the inline feedback prompt shown when the loop is paused
// awaiting the human's observation (manual mode).
function FeedbackPanel({ iterIndex, pending, onSubmit }: { iterIndex: number | null; pending: boolean; onSubmit: (text: string, goAuto: boolean) => void }) {
  const [text, setText] = useState("");
  return (
    <div className="rs-card" style={{ margin: "0 16px 12px", borderColor: "var(--st-amber)", borderWidth: 1, borderStyle: "solid" }}>
      <div className="rs-flex rs-gap-6" style={{ alignItems: "center", marginBottom: 6 }}>
        <Icon name="clock" size={15} color="var(--st-amber)" />
        <span className="rs-eyebrow">Awaiting your feedback{typeof iterIndex === "number" ? ` · after iteration ${iterIndex}` : ""}</span>
      </div>
      <div className="rs-sub" style={{ marginBottom: 8 }}>
        Watch the latest rollout, then tell the diagnoser what you see (optional) — it steers the next iteration.
      </div>
      <textarea
        className="rs-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        placeholder="e.g. it's standing still and flailing its arms — not actually kicking"
        style={{ width: "100%", resize: "vertical" }}
      />
      <div className="rs-flex rs-gap-8" style={{ marginTop: 8 }}>
        <Btn kind="primary" size="sm" icon={pending ? "loader" : "play"} disabled={pending} onClick={() => onSubmit(text, false)}>Continue</Btn>
        <Btn kind="quiet" size="sm" disabled={pending} onClick={() => onSubmit(text, true)}>Continue + go Auto</Btn>
      </div>
    </div>
  );
}

// §Ship 21c reward-transition logic — preserved verbatim, restyled.
function RewardVersionTransition({
  versionBefore, versionAfter, editCount, failureModes, status,
}: {
  versionBefore: number | null;
  versionAfter: number | null;
  editCount: number | null;
  failureModes: string[];
  status: "running" | "completed" | "errored" | "stopped";
}) {
  if (status === "running" || status === ("queued" as string)) {
    return <span className="ver">v{versionBefore ?? "?"} → v?</span>;
  }
  if (versionAfter !== null) {
    return <span className="ver">v{versionBefore ?? "?"} → v{versionAfter}</span>;
  }
  const noFailures = failureModes.length === 1 && failureModes[0] === "none";
  const hadEdits = editCount !== null && editCount > 0;
  if (noFailures) {
    return <span className="ver" style={{ color: "var(--st-emerald)" }} title="Diagnoser found no failure modes; no reward edit needed.">v{versionBefore ?? "?"} · held</span>;
  }
  if (hadEdits) {
    return <span className="ver" style={{ color: "var(--st-amber)" }} title={`${editCount} edit(s) proposed but filtered at pre-flight.`}>v{versionBefore ?? "?"} · {editCount} edit{editCount === 1 ? "" : "s"} filtered</span>;
  }
  return <span className="ver" title="No new reward version this iter.">v{versionBefore ?? "?"} · no edit</span>;
}

// ── iteration timeline ────────────────────────────────────────────────
function IterationTimeline({ iters, selected, onSelect }: { iters: IterEventSummary[]; selected: number | null; onSelect: (n: number) => void }) {
  return (
    <div className="rs-iter-col">
      <div className="rs-eyebrow" style={{ marginBottom: 12 }}>Iterations</div>
      {iters.length === 0 && <p className="rs-sub" style={{ fontSize: 11 }}>no iterations yet</p>}
      {iters.map((it) => (
        <button
          key={it.iter_index}
          className={"rs-itercard" + (selected === it.iter_index ? " on" : "")}
          style={{ width: "100%", textAlign: "left", cursor: "pointer" }}
          onClick={() => onSelect(it.iter_index)}
        >
          <div className="rs-itercard-top">
            <span className="it"><Badge status={it.status} label="" />iter {it.iter_index}</span>
            {/* §Ship 35: objective fitness is the PRIMARY metric (prominent,
                violet); the reward metric is demoted to a small muted value.
                Falls back to reward-as-primary on blind runs (no fitness). */}
            {typeof it.fitness === "number" ? (
              <span style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                <span className="rs-num" title="objective fitness (spec_score, 0-1) — primary metric" style={{ fontSize: 15, fontWeight: 600, color: "#b9aef5" }}>
                  fit {it.fitness.toFixed(2)}{typeof it.best_fitness === "number" && it.best_fitness > it.fitness + 1e-9 ? ` (best ${it.best_fitness.toFixed(2)})` : ""}
                </span>
                {it.primary_metric !== null && (
                  <span className="rs-num" title="reward metric (secondary)" style={{ fontSize: 11, color: "var(--rs-muted)" }}>r {it.primary_metric.toFixed(1)}</span>
                )}
              </span>
            ) : (
              it.primary_metric !== null && <span className="rs-num" style={{ fontSize: 13 }}>{it.primary_metric.toFixed(1)}</span>
            )}
          </div>
          {it.status === "running" && typeof it.rl_total === "number" && it.rl_total > 0 && (
            <IterProgressBar rlIter={it.rl_iter ?? 0} rlTotal={it.rl_total} pct={it.pct ?? 0} etaS={it.eta_s ?? null} />
          )}
          {(it.reward_version_before !== null || it.reward_version_after !== null) && (
            <div className="rs-flex-between" style={{ marginTop: 5 }}>
              <RewardVersionTransition versionBefore={it.reward_version_before} versionAfter={it.reward_version_after} editCount={it.edit_count} failureModes={it.failure_modes} status={it.status} />
              {it.metric_delta !== null && <Delta value={it.metric_delta} />}
            </div>
          )}
          {it.failure_modes.length > 0 && !(it.failure_modes.length === 1 && it.failure_modes[0] === "none") && (
            <span className="ver" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>{it.failure_modes.join(", ")}</span>
          )}
          {it.realism_audit && typeof it.realism_audit.verdict === "string" && it.realism_audit.verdict !== "ok" && it.realism_audit.verdict !== "unknown" && (
            <span className="rs-tag" style={{ marginTop: 4, fontSize: 10, background: it.realism_audit.verdict === "severe" ? "var(--st-rose-bg)" : "var(--st-amber-bg)", color: it.realism_audit.verdict === "severe" ? "var(--st-rose-fg)" : "var(--st-amber-fg)" }}>
              physics: {it.realism_audit.verdict}
            </span>
          )}
          {it.physics_edit_suggestion && it.physics_edit_suggestion.prompt && (() => {
            const state = it.physics_edit_suggestion.auto_apply_state;
            const disabled = state === "in_progress" || state === "applied";
            const label = state === "applied" ? "physics auto-applied" : state === "in_progress" ? "applying physics fix…" : state === "rejected" || state === "errored" ? "physics fix failed — retry" : "apply physics fix";
            return (
              <span
                onClick={(e) => {
                  e.stopPropagation();
                  if (disabled) return;
                  try { sessionStorage.setItem("pendingPhysicsPrompt", it.physics_edit_suggestion!.prompt); }
                  catch { void navigator.clipboard?.writeText(it.physics_edit_suggestion!.prompt); }
                  const match = window.location.pathname.match(/\/projects\/([^/]+)/);
                  if (match) window.location.assign(`/projects/${match[1]}`);
                }}
                className="rs-tag"
                style={{ marginTop: 4, fontSize: 10, cursor: disabled ? "default" : "pointer", background: state === "applied" ? "var(--st-emerald-bg)" : "var(--st-blue-bg)", color: state === "applied" ? "var(--st-emerald-fg)" : "var(--st-blue-fg)" }}
                title={it.physics_edit_suggestion.auto_apply_reason ?? "Open Physics tab with this prompt pre-filled"}
              >
                {label}
              </span>
            );
          })()}
        </button>
      ))}
    </div>
  );
}

function RunErrorCard({ slug, error, classification }: { slug: string; error: string; classification: ErrorClassification | null }) {
  const regen = useRegenerateRewardTemplate(slug);
  const isContractMismatch = classification?.kind === "reward_contract_mismatch" || classification?.action?.kind === "regenerate_reward_template";
  const onRegenerate = () => {
    regen.mutate(undefined, {
      onSuccess: () => toast.success("Reward template regenerated", { description: "rewards/v0.py rewritten for this adapter." }),
      onError: (err) => {
        const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
        toast.error("Could not regenerate template", { description: msg });
      },
    });
  };
  return (
    <div className="rs-banner err" style={{ flexDirection: "column", alignItems: "stretch", gap: 8 }}>
      <div className="rs-flex rs-gap-8">
        <Icon name="alert-triangle" size={17} />
        <span className="rs-grow"><b>{classification?.title ?? "Run errored"}.</b> {classification?.detail ?? ""}</span>
      </div>
      <pre className="mono" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 10, margin: 0, opacity: 0.9 }}>{error}</pre>
      {classification?.suggestions && classification.suggestions.length > 0 && (
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>{classification.suggestions.map((s, i) => <li key={i}>{s}</li>)}</ul>
      )}
      {isContractMismatch && (
        <div><Btn kind="ghost" size="sm" icon="refresh-cw" onClick={onRegenerate} disabled={regen.isPending}>{regen.isPending ? "Regenerating…" : "Regenerate reward template"}</Btn></div>
      )}
    </div>
  );
}

function IterationDetailCard({ iter }: { iter: IterEventSummary | null }) {
  if (!iter) return null;
  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-card-title" style={{ fontSize: 13, marginBottom: 10 }}><Icon name="circle-dot" size={15} />Iter {iter.iter_index} · {iter.status}</div>
      <div className="rs-vgap-8">
        {iter.failure_modes.length > 0 && (
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 4 }}>failure modes</div>
            <div className="rs-flex rs-wrap rs-gap-6">{iter.failure_modes.map((f) => <span key={f} className="rs-tag mono" style={{ fontSize: 10 }}>{f}</span>)}</div>
          </div>
        )}
        {iter.edit_count !== null && <div className="rs-flex-between" style={{ fontSize: 12 }}><span className="rs-sub">edits</span><span className="rs-num">{iter.edit_count}</span></div>}
        {iter.paper_refs.length > 0 && (
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 4 }}>paper refs</div>
            <div className="rs-vgap-8">
              {iter.paper_refs.map((r) => (
                <a key={r} href={`https://arxiv.org/abs/${r}`} target="_blank" rel="noreferrer noopener" className="mono" style={{ fontSize: 11, color: "var(--ink)", display: "block" }}>{r}</a>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── iter-merge logic (preserved verbatim) ────────────────────────────
const _ITER_STATUS_RANK: Record<string, number> = { queued: 0, running: 1, completed: 2, errored: 2 };

function _mergeIterSlot(prev: IterEventSummary | undefined, next: IterEventSummary): IterEventSummary {
  if (!prev) return next;
  const prevRank = _ITER_STATUS_RANK[prev.status] ?? 0;
  const nextRank = _ITER_STATUS_RANK[next.status] ?? 0;
  const winner = nextRank >= prevRank ? next : prev;
  const loser = nextRank >= prevRank ? prev : next;
  return {
    ...loser,
    ...winner,
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
    physics_edit_suggestion: winner.physics_edit_suggestion ?? loser.physics_edit_suggestion,
    // §Ship 34: keep fitness even if the winning (completed) slot didn't
    // carry it (iter_fitness fires before iter_completed).
    fitness: winner.fitness ?? loser.fitness,
    best_fitness: winner.best_fitness ?? loser.best_fitness,
  };
}

// §Ship 43: the launch-time objective-metric generation phase (run-phase 0).
// Streams the Ship-40 stages into the Runs timeline; surfaces acceptance and —
// crucially — REJECTIONS with the exact reasons (never silent). Ship 45 adds a
// one-click retry.
function MetricGenPhase({ phase, busy, onRetry, onContinueBlind }: {
  phase: {
    last: { stage?: string; message?: string; attempt?: number; max?: number } | null;
    outcome: "accepted" | "rejected" | "failed" | null;
    genId: string | null;
    reasons: string[];
    concerns: string[];
    errorMsg: string | null;
    calib: {
      status: "running" | "done" | "skipped";
      builtin?: string; calibrated?: boolean; spearman?: number | null;
    } | null;
    awaiting: boolean;
  };
  busy: boolean;
  onRetry: () => void;
  onContinueBlind: () => void;
}) {
  const { last, outcome, genId, reasons, concerns, errorMsg, calib, awaiting } = phase;
  const running = outcome === null;
  const bad = outcome === "rejected" || outcome === "failed";
  const tone = outcome === "accepted" ? "var(--st-green-fg, var(--ink))"
    : bad ? "var(--st-amber-fg)" : "var(--ink)";
  return (
    <div className="rs-card" style={{ marginBottom: 10 }}>
      <div className="rs-card-head">
        <div className="rs-card-title" style={{ fontSize: 13 }}>
          <Icon name={running ? "loader" : outcome === "accepted" ? "check-circle" : "alert-triangle"} size={15} />
          Objective metric generation
        </div>
        {running && <span className="rs-dot live" />}
      </div>
      <div style={{ padding: "8px 12px 12px", fontSize: 12.5, color: "var(--ink)" }}>
        {running && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Icon name="loader" size={13} />
            <span>
              {last?.message ?? "Starting generation…"}
              {typeof last?.attempt === "number" && typeof last?.max === "number"
                ? ` (attempt ${last.attempt}/${last.max})` : ""}
            </span>
          </div>
        )}
        {outcome === "accepted" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ color: tone }}>
              Generated <code className="mono">{genId}</code>.
            </div>
            {calib?.status === "running" && (
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Icon name="loader" size={12} />
                <span>Calibrating vs <code className="mono">{calib.builtin}</code>…</span>
              </div>
            )}
            {calib?.status === "done" && calib.calibrated && (
              <div style={{ color: "var(--st-green-fg, var(--ink))" }}>
                Calibrated vs <code className="mono">{calib.builtin}</code>
                {typeof calib.spearman === "number" ? ` (Spearman ${calib.spearman})` : ""}
                {" "}— <strong>steering</strong> the run.
              </div>
            )}
            {calib?.status === "done" && !calib.calibrated && (
              <div style={{ color: "var(--st-amber-fg)" }}>
                Did not pass calibration vs <code className="mono">{calib.builtin}</code>
                {typeof calib.spearman === "number" ? ` (Spearman ${calib.spearman} < 0.7)` : ""}
                {" "}— runs <strong>observe-only</strong>.
              </div>
            )}
            {calib?.status === "skipped" && (
              <div style={{ color: "var(--rs-muted)" }}>
                No matching built-in ground truth — runs observe-only.
              </div>
            )}
          </div>
        )}
        {outcome === "failed" && (
          <div style={{ color: tone }}>Generation failed: {errorMsg ?? "unknown error"}. Running blind.</div>
        )}
        {outcome === "rejected" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ color: tone, fontWeight: 600 }}>
              No metric was accepted — the run continues blind. Why:
            </div>
            {reasons.length > 0 && (
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {reasons.map((r, i) => <li key={`r${i}`} className="mono" style={{ fontSize: 11 }}>{r}</li>)}
              </ul>
            )}
            {concerns.length > 0 && (
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {concerns.map((c, i) => <li key={`c${i}`} style={{ fontSize: 11 }}>reviewer: {c}</li>)}
              </ul>
            )}
            {awaiting && (
              <div className="rs-flex rs-gap-6" style={{ marginTop: 4 }}>
                <Btn kind="primary" size="sm" icon={busy ? "loader" : "sparkles"} disabled={busy} onClick={onRetry}>
                  Retry generation
                </Btn>
                <Btn kind="quiet" size="sm" disabled={busy} onClick={onContinueBlind}>
                  Continue blind
                </Btn>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


function useMergedIterations(rest: IterEventSummary[], events: RunEvent[]): IterEventSummary[] {
  const stickyMap = useRef<Map<number, IterEventSummary>>(new Map());
  return useMemo(() => {
    const map = stickyMap.current;
    for (const r of rest) {
      map.set(r.iter_index, _mergeIterSlot(map.get(r.iter_index), r));
    }
    const eventSlots = new Map<number, IterEventSummary>();
    for (const ev of events) {
      const iter = (ev as { iter?: number }).iter;
      if (typeof iter !== "number") continue;
      const slot: IterEventSummary = eventSlots.get(iter) ?? map.get(iter) ?? {
        iter_index: iter, status: "running", started_at: null, completed_at: null,
        reward_version_before: null, reward_version_after: null, primary_metric: null,
        metric_delta: null, failure_modes: [], edit_count: null, paper_refs: [],
        rollout_ready: false, diagnosed: false, realism_audit: null, physics_edit_suggestion: null,
      };
      if (ev.type === "iter_started") {
        slot.started_at = slot.started_at ?? ev.ts;
        if (ev.reward_version_before !== undefined) slot.reward_version_before = Number(ev.reward_version_before);
      }
      if (ev.type === "rollout_done") slot.rollout_ready = true;
      if (ev.type === "physics_edit_suggested") {
        const prompt = (ev as { prompt?: unknown }).prompt;
        if (typeof prompt === "string" && prompt.trim().length > 0) {
          slot.physics_edit_suggestion = {
            prompt,
            verdict: typeof ev.verdict === "string" ? ev.verdict : null,
            top_joints_saturation: Array.isArray(ev.top_joints_saturation) ? (ev.top_joints_saturation as Array<{ name: string; value: number }>) : [],
          };
        }
      }
      if (ev.type === "realism_audited") {
        const audit = (ev as { audit?: Record<string, unknown> }).audit;
        if (audit && typeof audit === "object") {
          slot.realism_audit = audit as unknown as IterEventSummary["realism_audit"];
        } else if (slot.realism_audit == null) {
          slot.realism_audit = {
            verdict: typeof ev.verdict === "string" ? ev.verdict : "unknown",
            torque_saturation_frac: typeof ev.torque_saturation_frac === "number" ? ev.torque_saturation_frac : null,
            any_joint_saturation_max: typeof ev.any_joint_saturation_max === "number" ? ev.any_joint_saturation_max : null,
            joint_vel_p99_max: typeof ev.joint_vel_p99_max === "number" ? ev.joint_vel_p99_max : null,
            joint_limit_violation_frac: typeof ev.joint_limit_violation_frac === "number" ? ev.joint_limit_violation_frac : null,
            top_joints_saturation: Array.isArray(ev.top_joints_saturation) ? (ev.top_joints_saturation as Array<{ name: string; value: number }>) : [],
          };
        }
      }
      if (ev.type === "diagnosed") {
        slot.diagnosed = true;
        slot.failure_modes = Array.isArray(ev.failure_modes) ? (ev.failure_modes as string[]) : slot.failure_modes;
      }
      if (ev.type === "edit_applied") {
        if (ev.reward_version_after !== undefined) slot.reward_version_after = Number(ev.reward_version_after);
        if (ev.reward_version_before !== undefined && slot.reward_version_before === null) slot.reward_version_before = Number(ev.reward_version_before);
        if (Array.isArray(ev.paper_refs)) slot.paper_refs = ev.paper_refs as string[];
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
        if (typeof ev.eta_s === "number" || ev.eta_s === null) slot.eta_s = ev.eta_s as number | null;
      }
      // §Ship 34: objective fitness-in-the-loop, shown beside the metric.
      if (ev.type === "iter_fitness") {
        if (typeof ev.fitness === "number") slot.fitness = ev.fitness;
        if (typeof ev.best_so_far === "number") slot.best_fitness = ev.best_so_far;
      }
      if (ev.type === "best_reward_selected") {
        if (typeof ev.fitness === "number") slot.best_fitness = ev.fitness;
      }
      eventSlots.set(iter, slot);
    }
    for (const [iter, slot] of eventSlots) {
      map.set(iter, _mergeIterSlot(map.get(iter), slot));
    }
    return Array.from(map.values()).sort((a, b) => a.iter_index - b.iter_index);
  }, [rest, events]);
}

function IterProgressBar({ rlIter, rlTotal, pct, etaS }: { rlIter: number; rlTotal: number; pct: number; etaS: number | null }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const etaLabel = etaS == null ? "" : etaS < 60 ? ` · ETA ${Math.round(etaS)}s` : etaS < 3600 ? ` · ETA ${Math.round(etaS / 60)}m` : ` · ETA ${(etaS / 3600).toFixed(1)}h`;
  return (
    <div style={{ marginTop: 6 }}>
      <div className="rs-iterbar"><i style={{ width: `${clamped}%` }} /></div>
      <span className="mono" style={{ fontSize: 10, color: "var(--rs-muted)" }}>{rlIter}/{rlTotal} ({clamped.toFixed(1)}%){etaLabel}</span>
    </div>
  );
}

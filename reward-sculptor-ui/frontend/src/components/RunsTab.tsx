import { useEffect, useMemo, useRef, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Badge, Btn, Delta, EmptyState, MetricChart, Sparkline, STATUS_META } from "@/components/rs/primitives";
import { LogViewer } from "@/components/LogViewer";
import { NewRunDialog } from "@/components/NewRunDialog";
import { NewMissionDialog } from "@/components/NewMissionDialog";
import { MissionDetailDialog } from "@/components/MissionDetailDialog";
import { useSystemGpu } from "@/hooks/useLibrary";
import { useRunEvents } from "@/hooks/useRunEvents";
import {
  useBackfillFitness, useMission, useMissions, useStageIterations, useStageSelection,
} from "@/hooks/useMissions";
import { useReferenceIndex } from "@/hooks/useReferences";
import { useRegenerateRewardTemplate, useRewards } from "@/hooks/useRewards";
import { usePolicies } from "@/hooks/usePolicies";
import { useControlRun, useKillRun, useProjectIterations, useRun, useRuns } from "@/hooks/useRuns";
import {
  ApiError, getMission, getStageIterDetail, getStageObjectiveMetric,
  policyExportUrl, projectIterRolloutUrl, stageCheckpointUrl, stageExportUrl,
  stageRolloutUrl,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import {
  formatContradictionTooltip, formatIterMetrics, naturalnessChipText, selectionLabel, selectionSentence,
} from "@/lib/selection";
import { failureReasonText, stageLabel, supersededText } from "@/lib/stageDisplay";
import { formatRelative, sanitizeConsoleText } from "@/lib/utils";
import type {
  ErrorClassification,
  IterEventSummary,
  MissionDetail,
  MissionSummary,
  ProjectDetail,
  RunDetail,
  RunEvent,
  RunSummary,
  SelectedStage,
  StageIterDetail,
  StageIterPaperRef,
  StageIteration,
  StageMetricReference,
  StageObjectiveMetric,
  StageSelectionCandidate,
  StageSelectionReport,
} from "@/lib/types";

// ── benign-error display mapping ────────────────────────────────────────
// Some RunSummary.error strings mark benign, designed-for outcomes rather
// than real crashes — a stage that trained fine but missed its success
// criterion (mission auto-replans it), or a stage run that was collateral
// from the user cancelling the whole mission. Both should read as
// something other than a red "Errored" badge. STATUS_META/Badge stay
// generic (shared by many surfaces); this maps at the call site instead.
const CRITERION_NOT_MET_ERROR = "criterion_not_met";
const MISSION_TERMINATED_ERROR = "parent mission_execute terminated mid-stage";

interface RunDisplayStatus {
  label: string;
  cls: "slate" | "blue" | "amber" | "emerald" | "rose";
  icon: string;
}

/** Maps a run's raw status/error to a display descriptor for badges,
 *  falling back to the shared STATUS_META lookup for everything else. */
function runDisplayStatus(run: Pick<RunSummary, "status" | "error">): RunDisplayStatus {
  if (run.error === CRITERION_NOT_MET_ERROR) {
    return { label: "Criterion not met", cls: "amber", icon: "alert-circle" };
  }
  if (run.error === MISSION_TERMINATED_ERROR) {
    return { label: "Stopped (mission cancelled)", cls: "slate", icon: "square" };
  }
  const m = STATUS_META[run.status] ?? STATUS_META.draft;
  return { label: m.label, cls: m.cls, icon: m.icon };
}

/** Badge variant that honors the benign-error remap above; otherwise
 *  identical to <Badge status={run.status} />. */
function RunStatusBadge({
  run, big, label,
}: { run: Pick<RunSummary, "status" | "error">; big?: boolean; label?: string }) {
  const d = runDisplayStatus(run);
  return (
    <span className={"rs-badge " + d.cls + (big ? " big" : "")}>
      <Icon name={d.icon} size={12} />
      {label !== undefined ? label : d.label}
    </span>
  );
}

// ── public entry ──────────────────────────────────────────────────────
export default function RunsTab({
  slug, project, selectedStage, setSelectedStage,
}: {
  slug: string;
  project: ProjectDetail;
  selectedStage?: SelectedStage | null;
  setSelectedStage?: (value: SelectedStage | null) => void;
}) {
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
  // §Increment 4: display_label lookup (mission_slug -> stage_name ->
  // {displayLabel, topLevelCount}) for every stage-numbering surface below.
  // Called unconditionally, before the isLoading/empty early returns, so
  // rules-of-hooks holds even though those branches don't render RunSidebar.
  const stageLabels = useMissionStageLabels(slug, missions.data ?? []);
  const allOrderedRunIds = useMemo(
    () => [
      ...sculptRuns.map((r) => r.run_id),
      ...missionGroups.flatMap((g) => g.stages.map((r) => r.run_id)),
    ],
    [sculptRuns, missionGroups],
  );
  // §Ship de-silo (Training tab): selecting a mission STAGE row now sets
  // both the live run_id (best-effort, for the active stage) AND the
  // shared cross-tab `selectedStage` (mission_slug/stage_name) — the
  // latter is disk-truth and survives the run_id going stale when the
  // stage is superseded or a later stage errors.
  const selectStageRow = (run: RunSummary) => {
    setSelectedRunId(run.run_id);
    if (run.mission_slug && run.stage_name && setSelectedStage) {
      setSelectedStage({ missionSlug: run.mission_slug, stageName: run.stage_name });
    }
  };
  // Deep-linked `?stage=` (e.g. from the mission dialog's "view rewards
  // for this stage" pattern, or a bookmark) with no in-tab click yet:
  // resolve it to that stage's run_id so the detail pane opens on it
  // instead of defaulting to the first run in the list.
  const stageDeepLinkRunId = useMemo(() => {
    if (!selectedStage) return null;
    for (const g of missionGroups) {
      if (g.missionSlug !== selectedStage.missionSlug) continue;
      const row = g.stages.find((r) => r.stage_name === selectedStage.stageName);
      if (row) return row.run_id;
    }
    return null;
  }, [selectedStage, missionGroups]);
  const selected = selectedRunId ?? stageDeepLinkRunId ?? allOrderedRunIds[0] ?? null;
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
            <h2 className="rs-h2">Training</h2>
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
          stageLabels={stageLabels}
          selected={selected}
          selectedStage={selectedStage ?? null}
          onSelectRun={setSelectedRunId}
          onSelectStageRow={selectStageRow}
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
    // Zero-run missions are still listed (stages: []) — a decomposed
    // mission must be reachable (curriculum dialog, reference picker,
    // per-stage metrics) BEFORE its first training run, or it stays
    // invisible until trained (house rule: every feature UI-reachable).
    const stages = stagesByMission.get(m.mission_slug) ?? [];
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

// ── stage display-label plumbing (Increment 4) ───────────────────────
// display_label ("1", "1.1", …) only lives on StageSchema (mission detail),
// not on RunSummary rows — so every mission-scoped numbering surface in
// this tab needs a mission_slug -> stage_name -> StageSchema lookup.
// Mirrors RewardsTab's RewardsScopeSelector useQueries-per-mission pattern
// (same query keys, so the cache is shared rather than duplicated).
interface StageLabelInfo {
  displayLabel: string;
  topLevelCount: number;
}

function useMissionStageLabels(
  slug: string,
  missions: MissionSummary[],
): Map<string, Map<string, StageLabelInfo>> {
  const details = useQueries({
    queries: missions.map((m) => ({
      queryKey: qk.mission(slug, m.mission_slug),
      queryFn: () => getMission(slug, m.mission_slug),
      staleTime: 30_000,
    })),
  });
  return useMemo(() => {
    const out = new Map<string, Map<string, StageLabelInfo>>();
    missions.forEach((m, i) => {
      const d = details[i]?.data as MissionDetail | undefined;
      if (!d) return;
      const topLevelCount = d.stages.filter((s) => !(s.display_label ?? "").includes(".")).length;
      const byName = new Map<string, StageLabelInfo>();
      d.stages.forEach((s, idx) => {
        byName.set(s.name, { displayLabel: stageLabel(s, idx + 1), topLevelCount });
      });
      out.set(m.mission_slug, byName);
    });
    return out;
  }, [missions, details]);
}

// §Ship de-silo fix: `n` now prefers the REAL stage-run count seen on disk
// (`knownStages`, from `missionGroups[].stages.length`) over the mission's
// possibly-stale `n_stages` — for a terminal mission with a later stage
// errored, `n_stages`/`current_stage_idx` can undercount or freeze on the
// stage that was live when it errored, showing "Stage 1 of 3" forever.
// `viewedStageIndex1based`, when given (the row the user is actually
// looking at, from disk-truth `stage_index`), takes priority for the
// "Stage N" part so switching stages updates the header immediately.
//
// §Increment 4: "Stage N of M" now prefers display_label-derived values —
// `viewedStageLabel` (e.g. "1.2") for N, and `topLevelCount` (labels with
// no dot — replan children don't inflate M) for the denominator. Both fall
// back to the pre-existing index/knownStages arithmetic while the mission
// detail query (which carries display_label) is still loading.
function missionRunStateLabel(
  m: MissionSummary,
  knownStages: number,
  viewedStageIndex1based?: number | null,
  viewedStageLabel?: string | null,
  topLevelCount?: number | null,
): string {
  const { current_stage_idx: i, n_stages, lifecycle } = m;
  const n = topLevelCount ?? Math.max(n_stages, knownStages);
  if (n === 0) return "Planning…";
  if (lifecycle === "running") {
    const shown = viewedStageLabel ?? viewedStageIndex1based ?? Math.max(1, i + 1);
    return `Stage ${shown} of ${n}`;
  }
  if (lifecycle === "ready") return `${n} stages planned`;
  if (lifecycle === "completed") return `${n} of ${n} stages complete`;
  // Terminal (errored/halted) mission: show the stage being VIEWED, not
  // the stale current_stage_idx from whichever stage was live when a
  // later one errored.
  if (viewedStageLabel != null) return `Stage ${viewedStageLabel} of ${n}`;
  if (viewedStageIndex1based != null) return `Stage ${viewedStageIndex1based} of ${n}`;
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
  slug, project, sculptRuns, missionGroups, stageLabels, selected, selectedStage, onSelectRun, onSelectStageRow, onOpenMissionDialog, onLaunchedRun,
}: {
  slug: string;
  project: ProjectDetail;
  sculptRuns: RunSummary[];
  missionGroups: MissionGroup[];
  stageLabels: Map<string, Map<string, StageLabelInfo>>;
  selected: string | null;
  selectedStage: SelectedStage | null;
  onSelectRun: (id: string) => void;
  onSelectStageRow: (run: RunSummary) => void;
  onOpenMissionDialog: (missionSlug: string) => void;
  onLaunchedRun: (id: string) => void;
}) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const toggle = (s: string) => setCollapsed((st) => ({ ...st, [s]: !st[s] }));

  return (
    <div className="rs-runs-side">
      <div className="rs-side-head">
        <span className="rs-h3" style={{ fontSize: 15 }}>Runs</span>
        {/* rs-wrap: in the narrow sidebar the pair drops to its own row
            instead of forcing a horizontal scrollbar. */}
        <div className="rs-flex rs-gap-6 rs-wrap">
          <NewMissionDialog slug={slug} onCreated={(s) => onOpenMissionDialog(s)} />
          <NewRunDialog slug={slug} project={project} onLaunched={onLaunchedRun} />
        </div>
      </div>

      {missionGroups.length > 0 && <div className="rs-side-group">Missions</div>}
      {missionGroups.map((g) => {
        const isCollapsed = collapsed[g.missionSlug] ?? false;
        // A row counts as "selected" either by live run_id (normal case)
        // OR by the shared disk-truth selectedStage (keeps the group open
        // and the row highlighted even after its run_id goes stale).
        const isRowSelected = (s: RunSummary) =>
          s.run_id === selected ||
          (selectedStage != null && selectedStage.missionSlug === g.missionSlug && selectedStage.stageName === s.stage_name);
        const selectedInGroup = g.stages.some(isRowSelected);
        const open = !isCollapsed || selectedInGroup;
        // Which stage the user is actually looking at, for the header's
        // "Stage N of M" — prefers the row matching selectedStage/selected
        // over the mission's live (possibly stale) current_stage_idx.
        const viewedRow = g.stages.find(isRowSelected);
        const viewedStageIndex1based =
          viewedRow && typeof viewedRow.stage_index === "number" ? viewedRow.stage_index + 1 : null;
        const groupLabels = stageLabels.get(g.missionSlug) ?? null;
        const viewedStageLabel = viewedRow?.stage_name ? groupLabels?.get(viewedRow.stage_name)?.displayLabel ?? null : null;
        const topLevelCount = groupLabels ? [...groupLabels.values()].reduce((max, v) => Math.max(max, v.topLevelCount), 0) || null : null;
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
                    {g.mission ? missionRunStateLabel(g.mission, g.stages.length, viewedStageIndex1based, viewedStageLabel, topLevelCount) : g.missionSlug}
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
            {/* §Ship de-silo: a visible divider under the mission head so
                the full stage list reads as one clearly-grouped block. */}
            {open && g.stages.length > 0 && <div className="rs-stage-divider" aria-hidden="true" />}
            {open && g.stages.map((r) => (
              <RunRow
                key={r.run_id}
                run={r}
                displayLabel={r.stage_name ? groupLabels?.get(r.stage_name)?.displayLabel ?? null : null}
                selected={isRowSelected(r)}
                onSelect={() => onSelectStageRow(r)}
                stageContext
              />
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
  run: r, displayLabel = null, selected, onSelect, stageContext = false,
}: { run: RunSummary; displayLabel?: string | null; selected: boolean; onSelect: () => void; stageContext?: boolean }) {
  // §increment 3: the synthetic project-level disk row gets a human
  // name — "disk:project" is an API id, not a label.
  const titleText = stageContext
    ? r.stage_name ?? r.run_id.replace(/^job_/, "")
    : r.run_id === "disk:project"
      ? "project runs (recovered)"
      : r.run_id.replace(/^job_/, "");
  const itersDenom = r.iterations_requested || "?";
  // §Increment 4: prefer the server's hierarchical display_label ("1.2")
  // over stage_index+1 — falls back while the mission-detail label map is
  // still loading (or for disk-reconstructed rows with no stage_index).
  const numberLabel = displayLabel ?? (typeof r.stage_index === "number" ? String(r.stage_index + 1) : null);
  return (
    <button className={"rs-runrow" + (selected ? " on" : "") + (stageContext ? " rs-stage" : "")} onClick={onSelect}>
      <RunStatusBadge run={r} label="" />
      <span style={{ minWidth: 0, flex: 1 }}>
        <span className="rid" style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {stageContext && numberLabel && <span style={{ color: "var(--rs-muted)" }}>{numberLabel}. </span>}
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
  const summaryEarly = useMemo(() => runs.find((r) => r.run_id === runId) ?? null, [runs, runId]);
  const isStageRunEarly = summaryEarly?.kind === "mission_stage_run";
  const missionSlugEarly = summaryEarly?.mission_slug ?? null;
  const stageNameEarly = summaryEarly?.stage_name ?? null;

  // §Ship de-silo: a mission stage row is only "live" (safe to source from
  // the in-memory /runs+WS path) while it's the run the mission is
  // ACTIVELY training right now. Any other stage — completed, superseded
  // by a later stage starting, or left behind because a LATER stage
  // errored — must read disk-truth instead, or its data disappears the
  // moment the live run_id moves on. `useRun`/`iterRolloutUrl` are scoped
  // to a single run_id and go stale/empty exactly in those cases; the
  // mission detail tells us definitively whether this run is the one
  // currently training.
  const missionDetail = useMission(slug, missionSlugEarly ?? undefined, {
    enabled: isStageRunEarly && !!missionSlugEarly,
  });
  const isLiveStage =
    !isStageRunEarly ||
    (missionDetail.data?.active_job_id != null &&
      missionDetail.data.stages[missionDetail.data.current_stage_idx]?.name === stageNameEarly) ||
    // Before the mission detail loads, assume live if the run itself is
    // still running/queued — avoids a one-frame flash to disk-truth for
    // the common case of a freshly-launched stage.
    (missionDetail.isLoading && (summaryEarly?.status === "running" || summaryEarly?.status === "queued"));
  const useDiskTruth = isStageRunEarly && !isLiveStage;

  // §increment 3: the synthetic project-level row (run_id "disk:project",
  // synthesized by list_runs after a backend restart) has no JobManager
  // entry — LiveRunDetailPane's useRun/useRunEvents would 404 on it.
  // Route it to the project-scoped disk-truth pane instead.
  if (summaryEarly?.run_id === "disk:project") {
    return <ProjectDiskDetailPane slug={slug} summary={summaryEarly} />;
  }

  return useDiskTruth && missionSlugEarly && stageNameEarly ? (
    <StageDetailPane
      slug={slug}
      missionSlug={missionSlugEarly}
      stageName={stageNameEarly}
      summary={summaryEarly}
      mission={missionDetail.data ?? null}
    />
  ) : (
    <LiveRunDetailPane slug={slug} runId={runId} runs={runs} mission={isStageRunEarly ? missionDetail.data ?? null : null} />
  );
}

// Disk-truth stage view: sources iterations + rollout from
// `useStageIterations`/`stageRolloutUrl` (the same disk-truth path proven
// by MissionDetailDialog's StagePanel) instead of the live `useRun` +
// `iterRolloutUrl`, so a completed/superseded/left-behind stage keeps
// showing its data no matter what the live job is doing now.
function StageDetailPane({
  slug, missionSlug, stageName, summary, mission,
}: {
  slug: string;
  missionSlug: string;
  stageName: string;
  summary: RunSummary | null;
  mission: MissionDetail | null;
}) {
  const stage = mission?.stages.find((s) => s.name === stageName) ?? null;
  const iters = useStageIterations(slug, missionSlug, stageName);
  const rows = iters.data ?? [];

  const [picked, setPicked] = useState<number | null>(null);
  // A picked iteration is meaningful only within the stage it was picked
  // in — switching stages must fall back to the new stage's own kept/
  // default iteration, not carry iter N across.
  useEffect(() => {
    setPicked(null);
  }, [missionSlug, stageName]);
  const defaultIter = useMemo(() => {
    if (rows.length === 0) return null;
    const kept =
      stage?.selected_iter_index != null
        ? rows.find((r) => r.iter_index === stage.selected_iter_index)
        : undefined;
    if (kept?.has_rollout) return kept.iter_index;
    const newestWithRollout = [...rows].reverse().find((r) => r.has_rollout);
    if (newestWithRollout) return newestWithRollout.iter_index;
    return rows[rows.length - 1].iter_index;
  }, [rows, stage?.selected_iter_index]);
  const activeIter = picked ?? defaultIter;
  const activeRow = rows.find((r) => r.iter_index === activeIter) ?? null;

  // §narrate-completed-stage: the objective metric that guided this whole
  // stage (once per stage, not per iteration).
  const objectiveMetric = useQuery<StageObjectiveMetric>({
    queryKey: ["stageObjectiveMetric", slug, missionSlug, stageName],
    queryFn: () => getStageObjectiveMetric(slug, missionSlug, stageName),
    enabled: !!slug && !!missionSlug && !!stageName,
    staleTime: 30_000,
  });

  // §selection-report UI: the keep-decision report ("why this iteration
  // was kept") — synthesized from mission.json for stages that predate
  // live selection.json writing.
  const selection = useStageSelection(slug, missionSlug, stageName);

  // §selection-report UI: recover on-disk fitness from run logs, for
  // stages that finished before objective fitness was recorded live.
  // Gated on the MISSION (not just this stage) not being live — the
  // backend 409s regardless of which stage is asked about while any job
  // for this mission is running.
  const backfillFitness = useBackfillFitness(slug);
  const [backfillNote, setBackfillNote] = useState<string | null>(null);
  const missionRunning = mission?.active_job_id != null;
  const handleBackfillFitness = () => {
    setBackfillNote(null);
    backfillFitness.mutate(missionSlug, {
      onSuccess: (res) => {
        const inStage = res.stages[stageName] ?? 0;
        setBackfillNote(
          `recovered fitness for ${res.written} iteration${res.written === 1 ? "" : "s"} across the mission` +
          (inStage > 0 ? ` (${inStage} in this stage)` : ""),
        );
      },
      onError: (err) => {
        setBackfillNote(
          err instanceof ApiError && err.status === 409
            ? "a run is live — try after it finishes"
            : err.message,
        );
      },
    });
  };

  // §narrate-completed-stage: per-iteration reasoning (reward description,
  // diagnosis evidence, cited papers, components) for the selected iter.
  const iterDetail = useQuery<StageIterDetail>({
    queryKey: ["stageIterDetail", slug, missionSlug, stageName, activeIter],
    queryFn: () => getStageIterDetail(slug, missionSlug, stageName, activeIter as number),
    enabled: !!slug && !!missionSlug && !!stageName && activeIter != null,
    staleTime: 30_000,
  });

  // §UX polish (fitness backfill): the disk-truth list endpoint
  // (`useStageIterations` above) reads `fitness` from only
  // `rollout/behavior.json`'s bare `fitness` key, which is empty for
  // every stage on this deployment (active or not) — so the iteration
  // list showed an objective-fitness badge ONLY for the live-training
  // stage, which sources it from in-memory run events instead. The
  // per-iteration `/detail` endpoint extracts the same value more
  // thoroughly (reward_spec.json / a dedicated fitness json / a regex
  // fallback off the diagnoser evidence prose) and already succeeds
  // where the list endpoint returns null — see `_extract_objective_fitness`
  // in backend/routes/missions.py. Backfill from there per row so
  // previous stages' iterations show their fitness too. Shares the exact
  // query key `iterDetail` above uses for `activeIter`, so selecting that
  // iter never double-fetches. A real backend fix would teach
  // `list_stage_iterations` to call `_extract_objective_fitness` instead
  // of the narrow `behavior.get("fitness")` read — out of scope here
  // (backend is off-limits while a mission is training).
  const missingFitnessRows = useMemo(() => rows.filter((r) => r.fitness == null), [rows]);
  const fitnessBackfill = useQueries({
    queries: missingFitnessRows.map((r) => ({
      queryKey: ["stageIterDetail", slug, missionSlug, stageName, r.iter_index],
      queryFn: () => getStageIterDetail(slug, missionSlug, stageName, r.iter_index),
      enabled: !!slug && !!missionSlug && !!stageName,
      staleTime: 30_000,
    })),
  });
  const rowsWithFitness = useMemo(() => {
    if (missingFitnessRows.length === 0) return rows;
    const byIter = new Map<number, number | undefined>();
    missingFitnessRows.forEach((r, i) => {
      const detail = fitnessBackfill[i]?.data as StageIterDetail | undefined;
      byIter.set(r.iter_index, detail?.objective_fitness ?? undefined);
    });
    return rows.map((r) => {
      const backfilled = byIter.get(r.iter_index);
      return backfilled != null ? { ...r, fitness: backfilled } : r;
    });
  }, [rows, missingFitnessRows, fitnessBackfill]);

  // §selection-report UI (E): the row-level fitness/steer/progress/
  // fitness_source fields for the currently-selected iter, so the
  // reasoning card can show the real values instead of "not tracked".
  const activeRowForDetail = useMemo(
    () => rowsWithFitness.find((r) => r.iter_index === activeIter) ?? activeRow,
    [rowsWithFitness, activeIter, activeRow],
  );

  return (
    <div className="rs-runs-detail">
      <div className="rs-iter-col">
        <div className="rs-eyebrow" style={{ marginBottom: 12 }}>Iterations</div>
        {iters.isLoading && <p className="rs-sub" style={{ fontSize: 11 }}>Loading…</p>}
        {iters.error && <p style={{ fontSize: 11, color: "var(--st-rose)" }}>{(iters.error as Error).message}</p>}
        {!iters.isLoading && rows.length === 0 && <p className="rs-sub" style={{ fontSize: 11 }}>No iterations on disk yet for this stage.</p>}
        {rowsWithFitness.map((r) => (
          <StageIterCard
            key={r.iter_index}
            row={r}
            selected={activeIter === r.iter_index}
            kept={stage?.selected_iter_index === r.iter_index}
            selectionSource={stage?.selection_source}
            onSelect={() => setPicked(r.iter_index)}
          />
        ))}
      </div>

      {/* §UX polish: this column stacks the objective-metric card, rollout
          player, and per-iteration reasoning panel with no internal
          self-scrolling child (unlike LiveRunDetailPane's mid-col, whose
          LogViewer scrolls itself) — override the shared `.rs-mid-col`
          `overflow: hidden` so long diagnoser evidence/reward-component
          text stays reachable instead of clipping silently. */}
      <div className="rs-mid-col" style={{ overflowY: "auto" }}>
        {summary && <StageContextCard run={summary} displayLabel={stage?.display_label ?? null} />}
        {stage?.status === "superseded" && (
          <div className="rs-banner" style={{ margin: "0 16px 8px", background: "var(--canvas-soft)" }}>
            <Icon name="git-branch" size={15} />
            <span className="rs-grow" style={{ fontSize: 12 }}>{supersededText(stage)}</span>
          </div>
        )}
        {stage?.status === "failed" && stage.failure_reason && (
          <p className="rs-sub" style={{ margin: "0 16px 8px", fontSize: 11.5 }}>
            {failureReasonText(stage.failure_reason, stage.iterations_used)}
          </p>
        )}
        <div className="rs-run-header">
          <Icon name="activity" size={17} color="var(--rs-muted)" />
          <span className="mono" style={{ fontSize: 14, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>
            {stageName}
          </span>
          {summary && <RunStatusBadge run={summary} />}
          <span className="rs-grow" />
          <span className="rs-eyebrow" style={{ flexShrink: 0 }}>disk-truth · not the live job</span>
        </div>

        {/* §narrate-completed-stage: the objective metric that steered
            this stage — the thing the screen recording wants to point at. */}
        <div style={{ padding: "0 16px 12px" }}>
          <StageObjectiveMetricCard query={objectiveMetric} />
        </div>

        {/* §selection-report UI: the keep-decision report — which iter
            was kept and why, the per-candidate ranking, and (when this
            stage still has un-recovered fitness on disk) a button to
            pull it from run logs. */}
        <div style={{ padding: "0 16px 12px" }}>
          <StageSelectionCard
            query={selection}
            missingFitnessCount={missingFitnessRows.length}
            missionRunning={missionRunning}
            onBackfill={handleBackfillFitness}
            backfillPending={backfillFitness.isPending}
            backfillNote={backfillNote}
          />
        </div>

        {/* Rollout player for the picked iter — same disk-truth source as
            MissionDetailDialog's StagePanel, so it survives the run_id
            going stale. */}
        {activeIter != null && activeRow?.has_rollout ? (
          <div style={{ margin: "0 16px 12px", border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
            <div className="rs-log-bar" style={{ gap: 8 }}>
              <Icon name="video" size={13} color="var(--rs-muted)" />
              <span style={{ fontSize: 11.5, fontWeight: 600 }}>iter {activeIter} rollout</span>
              <span className="rs-grow" />
              <a
                href={stageRolloutUrl(slug, missionSlug, stageName, activeIter)}
                download={`${stageName}_iter_${activeIter}.mp4`}
                className="rs-btn rs-btn-quiet rs-btn-xs"
              >
                <Icon name="download" size={13} />MP4
              </a>
            </div>
            <video
              key={activeIter}
              src={stageRolloutUrl(slug, missionSlug, stageName, activeIter)}
              style={{ width: "100%", aspectRatio: "16/9", background: "#16150f", display: "block" }}
              controls
              playsInline
              preload="metadata"
            >
              <track kind="captions" />
            </video>
          </div>
        ) : activeIter != null ? (
          <p className="rs-sub" style={{ margin: "0 16px 12px", fontSize: 11 }}>iter {activeIter} has no rollout video.</p>
        ) : null}

        {/* §UX honesty pass: export reachable whenever the selected
            iteration has a checkpoint — independent of has_rollout. */}
        {activeIter != null && activeRow?.has_checkpoint && (
          <div className="rs-flex rs-gap-8" style={{ margin: "0 16px 12px" }}>
            <a
              href={stageExportUrl(slug, missionSlug, stageName, activeIter)}
              download
              className="rs-btn rs-btn-quiet rs-btn-xs"
              title="Download the deployment bundle: checkpoint + ONNX + TorchScript + reward/env spec + DEPLOY.md (builds server-side, may take a moment)"
            >
              <Icon name="package" size={13} />Export bundle
            </a>
            <a
              href={stageCheckpointUrl(slug, missionSlug, stageName, activeIter)}
              download
              className="rs-btn rs-btn-quiet rs-btn-xs"
              title="Download the raw checkpoint file only"
            >
              <Icon name="download" size={13} />raw .pt
            </a>
          </div>
        )}

        {/* §narrate-completed-stage: per-iteration score/reasoning/papers
            for whichever iter is selected on the left. */}
        {activeIter != null && (
          <div style={{ padding: "0 16px 12px" }}>
            <StageIterDetailCard iterIndex={activeIter} query={iterDetail} row={activeRowForDetail ?? null} />
          </div>
        )}

        <div style={{ padding: "0 16px" }}>
          <p className="rs-sub" style={{ fontSize: 12, lineHeight: 1.6 }}>
            This stage isn't the one currently training, so its data is read straight off
            disk — it stays available even after a later stage supersedes or errors.
          </p>
        </div>
      </div>

      <div className="rs-extra-col">
        {missionSlug && stageName && <StageRewardsCard slug={slug} stage={`${missionSlug}/${stageName}`} />}
      </div>
    </div>
  );
}

// §narrate-completed-stage: status → chip tone/label/icon for the stage
// objective-metric card.
const OBJECTIVE_METRIC_STATUS_META: Record<
  StageObjectiveMetric["status"],
  { label: string; cls: "slate" | "amber" | "emerald"; icon: string }
> = {
  accepted:  { label: "Objective metric ✓", cls: "emerald", icon: "check-circle" },
  rejected:  { label: "Rejected — blind fallback", cls: "amber", icon: "alert-triangle" },
  inherited: { label: "Inherited", cls: "slate", icon: "git-branch" },
  none:      { label: "No objective metric", cls: "slate", icon: "minus" },
};

/** The objective metric that guided this stage — generated once at stage
 *  launch, then used to score every iteration. This is the "here's the
 *  metric that steered training" panel a screen recording narrates. */
function StageObjectiveMetricCard({ query }: { query: ReturnType<typeof useQuery<StageObjectiveMetric>> }) {
  const { data, isLoading, error } = query;
  // §R1 remainder (plan §9): resolve clip_id -> text/tier for the
  // "Certified against reference" row below. Only fetched when the
  // metric actually carries references — the index call is cheap (one
  // GET /references) but no need to fire it for the common no-reference
  // case.
  const refIndex = useReferenceIndex({ enabled: (data?.references.length ?? 0) > 0 });
  if (isLoading) {
    return <div className="rs-card rs-card-pad"><p className="rs-sub" style={{ fontSize: 11 }}>Loading objective metric…</p></div>;
  }
  if (error) {
    return <div className="rs-card rs-card-pad"><p className="rs-sub" style={{ fontSize: 11, color: "var(--st-rose)" }}>{(error as Error).message}</p></div>;
  }
  if (!data) return null;
  const meta = OBJECTIVE_METRIC_STATUS_META[data.status] ?? OBJECTIVE_METRIC_STATUS_META.none;
  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-flex rs-gap-8" style={{ alignItems: "flex-start" }}>
        <div className="rs-card-title" style={{ fontSize: 13, flex: 1, minWidth: 0 }}>
          <Icon name="target" size={15} />Objective metric
        </div>
        <span className={"rs-badge " + meta.cls}><Icon name={meta.icon} size={12} />{meta.label}</span>
      </div>
      {data.behavior_goal && (
        <p style={{ margin: "8px 0 0", fontSize: 12.5, lineHeight: 1.5 }}>
          <span className="rs-sub">Generated from: </span>{data.behavior_goal}
        </p>
      )}
      <div className="rs-flex rs-wrap rs-gap-8" style={{ marginTop: 8 }}>
        {data.calibrated != null && (
          <span className="rs-tag" style={{ fontSize: 10.5 }}>
            <Icon name={data.calibrated ? "check" : "minus"} size={11} />
            {data.calibrated ? "calibrated" : "not calibrated"}
          </span>
        )}
        {typeof data.n_candidates === "number" && (
          <span className="rs-tag" style={{ fontSize: 10.5 }}>{data.n_candidates} candidate{data.n_candidates === 1 ? "" : "s"} sampled</span>
        )}
      </div>
      {data.review_summary && (
        <p className="rs-sub" style={{ margin: "8px 0 0", fontSize: 12, lineHeight: 1.5 }}>{data.review_summary}</p>
      )}
      {data.references.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div className="rs-sub" style={{ fontSize: 10.5, marginBottom: 6 }}>Certified against reference</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {data.references.map((reference) => (
              <ReferenceCertificationRow
                key={reference.clip_id}
                reference={reference}
                indexRow={refIndex.data?.find((r) => r.clip_id === reference.clip_id) ?? null}
              />
            ))}
          </div>
        </div>
      )}
      {data.metric_source && (
        <details style={{ marginTop: 10 }}>
          <summary style={{ cursor: "pointer", fontSize: 11.5, color: "var(--rs-muted)", userSelect: "none" }}>
            View metric source (metric.py)
          </summary>
          <pre
            className="mono"
            style={{
              margin: "8px 0 0", padding: 10, fontSize: 10.5, lineHeight: 1.5,
              background: "var(--canvas-soft)", border: "1px solid var(--hairline)",
              borderRadius: "var(--radius-sm)", overflow: "auto", maxHeight: 320,
              whiteSpace: "pre",
            }}
          >
            {data.metric_source}
          </pre>
        </details>
      )}
    </div>
  );
}

function fmtScore(v: number | null | undefined, digits = 3): string {
  return typeof v === "number" ? v.toFixed(digits) : "—";
}

/** §selection-report UI: status meta for the keep-decision headline —
 *  emerald when a criterion-backed pick was kept, amber when it's a
 *  fitness-only fallback (criterion unmet), slate when nothing was kept
 *  at all. */
function selectionHeadlineMeta(
  data: StageSelectionReport,
): { icon: string; cls: "slate" | "amber" | "emerald" } {
  if (data.selected_iter_index == null) return { icon: "minus", cls: "slate" };
  if (data.selection_source === "fitness_fallback") return { icon: "alert-triangle", cls: "amber" };
  return { icon: "check-circle", cls: "emerald" };
}

/** §selection-report UI: "why this iteration was kept" — the keep-best
 *  decision (headline + criterion text + per-candidate ranking table),
 *  plus (when this stage still has un-recovered on-disk fitness) a
 *  quiet button to pull it from run logs. This is the panel that turns
 *  "iter 3 got kept" into "iter 3 got kept BECAUSE...". */
function StageSelectionCard({
  query, missingFitnessCount, missionRunning, onBackfill, backfillPending, backfillNote,
}: {
  query: ReturnType<typeof useStageSelection>;
  missingFitnessCount: number;
  missionRunning: boolean;
  onBackfill: () => void;
  backfillPending: boolean;
  backfillNote: string | null;
}) {
  const { data, isLoading, error } = query;
  if (isLoading) {
    return <div className="rs-card rs-card-pad"><p className="rs-sub" style={{ fontSize: 11 }}>Loading keep-decision…</p></div>;
  }
  if (error) {
    return <div className="rs-card rs-card-pad"><p className="rs-sub" style={{ fontSize: 11, color: "var(--st-rose)" }}>{(error as Error).message}</p></div>;
  }
  if (!data) return null;

  const meta = selectionHeadlineMeta(data);
  const candidates = data.candidates ?? [];
  const noteText = data.start_state_mismatch || data.criterion_error || data.failure_detail;

  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-flex rs-gap-8" style={{ alignItems: "flex-start" }}>
        <div className="rs-card-title" style={{ fontSize: 13, flex: 1, minWidth: 0 }}>
          <Icon name="flag" size={15} />Why this iteration was kept
        </div>
        <span className={"rs-badge " + meta.cls}>
          <Icon name={meta.icon} size={12} />
          {data.selected_iter_index != null ? `Kept iter ${data.selected_iter_index}` : "No iteration kept"}
        </span>
      </div>
      <p style={{ margin: "8px 0 0", fontSize: 12.5, lineHeight: 1.5 }} title={selectionSentence(data.selection_source)}>
        {data.selected_iter_index != null ? (
          <>
            <span className="rs-sub">{selectionLabel(data.selection_source)} — </span>
            {selectionSentence(data.selection_source)}
          </>
        ) : (
          "No iteration was kept for this stage."
        )}
      </p>

      {data.criterion && (
        <div style={{ marginTop: 8 }}>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>Success criterion</div>
          <p
            className="mono"
            style={{
              margin: 0, wordBreak: "break-all", borderRadius: "var(--radius-sm)",
              background: "var(--canvas-soft)", border: "1px solid var(--hairline)",
              padding: "5px 8px", fontSize: 10.5, color: "var(--rs-muted)",
            }}
          >
            {data.criterion}
          </p>
        </div>
      )}

      {noteText && (
        <div
          style={{
            marginTop: 8, padding: "6px 8px", borderRadius: "var(--radius-sm)",
            background: "var(--st-rose-bg)", color: "var(--st-rose-fg)", fontSize: 11.5, lineHeight: 1.5,
          }}
        >
          {data.start_state_mismatch && <p style={{ margin: 0 }}>Start-state mismatch: {data.start_state_mismatch}</p>}
          {data.criterion_error && (
            <p style={{ margin: data.start_state_mismatch ? "4px 0 0" : 0 }}>Criterion error: {data.criterion_error}</p>
          )}
          {data.failure_detail && (
            <p style={{ margin: (data.start_state_mismatch || data.criterion_error) ? "4px 0 0" : 0 }}>
              {data.failure_reason ? `${data.failure_reason}: ` : ""}{data.failure_detail}
            </p>
          )}
        </div>
      )}

      {candidates.length > 0 && (
        <details style={{ marginTop: 10 }} open={candidates.length <= 6}>
          <summary style={{ cursor: "pointer", fontSize: 11.5, color: "var(--rs-muted)", userSelect: "none" }}>
            Candidates ({candidates.length})
          </summary>
          <div style={{ overflowX: "auto", marginTop: 8 }}>
            <table className="mono" style={{ width: "100%", borderCollapse: "collapse", fontSize: 10.5 }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--rs-muted)" }}>
                  <th style={{ padding: "3px 6px" }}>iter</th>
                  <th style={{ padding: "3px 6px" }}>criterion</th>
                  <th style={{ padding: "3px 6px", textAlign: "right" }}>fit</th>
                  <th style={{ padding: "3px 6px", textAlign: "right" }}>steer</th>
                  <th style={{ padding: "3px 6px", textAlign: "right" }}>prog</th>
                  <th style={{ padding: "3px 6px", textAlign: "right" }}>r</th>
                  <th style={{ padding: "3px 6px" }} />
                </tr>
              </thead>
              <tbody>
                {candidates.map((c: StageSelectionCandidate) => {
                  const gated =
                    c.fitness != null && c.steer_fitness != null &&
                    Math.abs(c.fitness - c.steer_fitness) > 1e-6;
                  return (
                    <tr
                      key={c.iter_index}
                      style={{ background: c.selected ? "var(--st-emerald-bg)" : undefined }}
                    >
                      <td style={{ padding: "3px 6px", fontWeight: c.selected ? 600 : 400 }}>{c.iter_index}</td>
                      <td
                        style={{ padding: "3px 6px" }}
                        title={
                          c.criterion_pass == null
                            ? "not evaluated / unknown"
                            : (c.criterion_error ?? undefined)
                        }
                      >
                        {c.criterion_pass == null ? "—" : c.criterion_pass ? "✓" : "✗"}
                        {c.gate_mismatched && (
                          <span title="start-state gate mismatch" style={{ marginLeft: 4, color: "var(--st-amber-fg)" }}>⚠</span>
                        )}
                      </td>
                      <td style={{ padding: "3px 6px", textAlign: "right" }} title={gated ? "realism-gated" : undefined}>{fmtScore(c.fitness)}</td>
                      <td
                        style={{ padding: "3px 6px", textAlign: "right", color: gated ? "var(--st-amber-fg)" : undefined }}
                        title={gated ? "realism-gated" : undefined}
                      >
                        {fmtScore(c.steer_fitness)}
                      </td>
                      <td style={{ padding: "3px 6px", textAlign: "right" }}>{fmtScore(c.progress)}</td>
                      <td style={{ padding: "3px 6px", textAlign: "right" }}>{fmtScore(c.primary_metric, 2)}</td>
                      <td style={{ padding: "3px 6px" }}>
                        {c.selected && <span className="rs-badge emerald" style={{ fontSize: 8 }}><Icon name="check" size={9} />kept</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {data.synthesized && (
        <p className="rs-sub" style={{ margin: "8px 0 0", fontSize: 10.5, fontStyle: "italic" }}>
          reconstructed from disk — this stage ran before selection recording existed; criterion column unknown
        </p>
      )}

      {missingFitnessCount > 0 && (
        <div style={{ marginTop: 10 }}>
          {missionRunning ? (
            <p className="rs-sub" style={{ margin: 0, fontSize: 10.5 }}>a run is live — try after it finishes</p>
          ) : (
            <button
              type="button"
              className="rs-btn rs-btn-quiet rs-btn-xs"
              disabled={backfillPending}
              onClick={onBackfill}
            >
              <Icon name={backfillPending ? "loader" : "refresh-cw"} size={13} className={backfillPending ? "rs-spin" : undefined} />
              {backfillPending ? "Recovering…" : "Recover fitness from run logs"}
            </button>
          )}
          {backfillNote && <p className="rs-sub" style={{ margin: "6px 0 0", fontSize: 10.5 }}>{backfillNote}</p>}
        </div>
      )}
    </div>
  );
}

// §R1 remainder (plan §9): the three §5 per-reference gates, short labels
// for the pass/fail row. Order matches _validate_references's write order
// (nondegeneracy, monotonicity, negatives).
const REFERENCE_GATE_LABELS: Record<string, string> = {
  reference_nondegeneracy: "nondegenerate",
  reference_monotonicity: "monotonic",
  reference_negatives: "rejects negatives",
};

/** One certified-reference row: clip text/id + tier badge (resolved from
 *  the reference index when available, else bare clip_id) + a tiny
 *  check/cross per gate. This is the "certified against reference" line
 *  from plan §9 — it reuses the SAME gate keys `_validate_references`
 *  writes, so it never drifts from what actually gated the metric. */
function ReferenceCertificationRow({
  reference, indexRow,
}: {
  reference: StageMetricReference;
  indexRow: { text: string; tier: string } | null | undefined;
}) {
  const gateEntries = Object.entries(reference.gates);
  return (
    <div
      style={{
        display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8,
        padding: "6px 8px", fontSize: 11.5,
        border: "1px solid var(--hairline)", borderRadius: "var(--radius-sm)",
        background: "var(--canvas-soft)",
      }}
    >
      <Icon name="video" size={12} color="var(--rs-muted)" />
      <span style={{ fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 220 }}>
        {indexRow?.text ?? reference.clip_id}
      </span>
      <span className="rs-badge slate" style={{ fontSize: 9 }}>
        {indexRow?.tier ? `tier ${indexRow.tier}` : reference.clip_id}
      </span>
      {gateEntries.length > 0 && (
        <span className="rs-flex rs-gap-8" style={{ marginLeft: "auto" }}>
          {gateEntries.map(([gate, passed]) => (
            <span
              key={gate}
              title={`${REFERENCE_GATE_LABELS[gate] ?? gate}: ${passed ? "passed" : "failed"}`}
              style={{ display: "inline-flex", alignItems: "center", gap: 3, fontSize: 10, color: passed ? "var(--st-emerald-fg)" : "var(--st-rose)" }}
            >
              <Icon name={passed ? "check" : "x"} size={11} />
              {REFERENCE_GATE_LABELS[gate] ?? gate}
            </span>
          ))}
        </span>
      )}
    </div>
  );
}

/** Merges reward_references + literature_context into one paper-citation
 *  list, deduping by arxiv_id/citation so the same paper isn't shown twice
 *  when both fields happen to carry it. */
function mergePaperRefs(a: StageIterPaperRef[], b: StageIterPaperRef[]): StageIterPaperRef[] {
  const seen = new Set<string>();
  const out: StageIterPaperRef[] = [];
  for (const ref of [...a, ...b]) {
    const key = (ref.arxiv_id ?? "") + "|" + (ref.citation ?? ref.description ?? "");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(ref);
  }
  return out;
}

/** Per-iteration reasoning: objective fitness / mean return, the reward's
 *  own description, the diagnoser's evidence prose, failure-mode chips,
 *  cited papers, and reward-component values. This is the "narrate the
 *  loop's thinking" panel — every field is best-effort/nullable since
 *  older iterations won't have all of them. */
function StageIterDetailCard({
  iterIndex, query, row,
}: {
  iterIndex: number;
  query: ReturnType<typeof useQuery<StageIterDetail>>;
  /** §selection-report UI (E): the matching disk-truth iteration row —
   *  carries fitness/steer_fitness/progress/fitness_source, which the
   *  /detail endpoint's `objective_fitness` doesn't always have. The
   *  /detail value still wins when present (it's the more thoroughly
   *  extracted one); this only fills gaps. */
  row?: StageIteration | null;
}) {
  const { data, isLoading, error } = query;
  if (isLoading) {
    return <div className="rs-card rs-card-pad"><p className="rs-sub" style={{ fontSize: 11 }}>Loading iteration detail…</p></div>;
  }
  if (error) {
    return <div className="rs-card rs-card-pad"><p className="rs-sub" style={{ fontSize: 11, color: "var(--st-rose)" }}>{(error as Error).message}</p></div>;
  }
  if (!data) return null;

  const failureModes = data.failure_modes.filter((f) => f && f !== "none");
  const papers = mergePaperRefs(data.reward_references ?? [], data.literature_context ?? []);
  // §selection-report UI (E): /detail's own objective_fitness takes
  // precedence (more thorough extraction); fall back to the row's
  // disk-truth fitness (possibly log-backfilled) when /detail has none.
  const objectiveFitness = data.objective_fitness ?? row?.fitness ?? null;
  const steerFitness = row?.steer_fitness ?? null;
  const steerDiffers =
    steerFitness != null && objectiveFitness != null && Math.abs(steerFitness - objectiveFitness) > 1e-6;
  const denseProgress = row?.progress ?? null;
  const recoveredFromLogs = row?.fitness_source === "log_backfill";
  const hasScores = objectiveFitness != null || data.primary_metric != null || data.reward_version;
  const componentEntries = data.components ? Object.entries(data.components) : [];
  const maxComponentAbs = componentEntries.length
    ? Math.max(...componentEntries.map(([, v]) => Math.abs(v)), 1e-9)
    : 1;

  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-card-title" style={{ fontSize: 13, marginBottom: 10 }}>
        <Icon name="circle-dot" size={15} />Iteration {iterIndex} reasoning
      </div>

      {hasScores && (
        <div className="rs-flex rs-wrap rs-gap-16" style={{ marginBottom: 10 }}>
          {objectiveFitness != null ? (
            <span title="objective fitness — the loop's steering score">
              <span className="rs-sub" style={{ fontSize: 11 }}>objective fitness</span><br />
              <span className="rs-num" style={{ fontSize: 17, fontWeight: 600, color: "#b9aef5" }}>
                {objectiveFitness.toFixed(3)}
              </span>
              {steerDiffers && (
                <span className="rs-sub" style={{ fontSize: 10.5, marginLeft: 5 }} title="realism-gated — differs from the plain objective fitness above">
                  steer {steerFitness!.toFixed(3)}
                </span>
              )}
              {denseProgress != null && (
                <span className="rs-sub" style={{ display: "block", fontSize: 10.5 }} title="dense per-iter progress signal — not the same scale as fitness">
                  dense progress {denseProgress.toFixed(3)}
                </span>
              )}
              {recoveredFromLogs && (
                <span className="rs-sub" style={{ display: "block", fontSize: 10, fontStyle: "italic" }}>
                  (recovered from run logs)
                </span>
              )}
            </span>
          ) : (
            <span>
              <span className="rs-sub" style={{ fontSize: 11 }}>objective fitness</span><br />
              <span className="rs-sub" style={{ fontSize: 12 }}>not tracked this iter — see notes below</span>
            </span>
          )}
          {data.primary_metric != null && (
            <span title="mean return over the rollout">
              <span className="rs-sub" style={{ fontSize: 11 }}>mean return</span><br />
              <span className="rs-num" style={{ fontSize: 15 }}>{data.primary_metric.toFixed(2)}</span>
            </span>
          )}
          {data.reward_version && (
            <span>
              <span className="rs-sub" style={{ fontSize: 11 }}>reward</span><br />
              <span className="rs-num" style={{ fontSize: 15 }}>{data.reward_version}</span>
            </span>
          )}
          {data.confidence != null && (
            <span title="diagnoser's confidence in this analysis">
              <span className="rs-sub" style={{ fontSize: 11 }}>confidence</span><br />
              <span className="rs-num" style={{ fontSize: 15 }}>{(data.confidence * 100).toFixed(0)}%</span>
            </span>
          )}
        </div>
      )}

      {data.reward_description && (
        <div style={{ marginBottom: 10 }}>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>Reward reasoning</div>
          <p style={{ margin: 0, fontSize: 12.5, lineHeight: 1.55 }}>{sanitizeConsoleText(data.reward_description)}</p>
        </div>
      )}

      {data.evidence && (
        <div style={{ marginBottom: 10 }}>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>Diagnosis</div>
          <p style={{ margin: 0, fontSize: 12.5, lineHeight: 1.55, whiteSpace: "pre-wrap" }}>{sanitizeConsoleText(data.evidence)}</p>
        </div>
      )}

      {failureModes.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>Failure modes</div>
          <div className="rs-flex rs-wrap rs-gap-6">
            {failureModes.map((f) => (
              <span key={f} className="rs-tag mono" style={{ fontSize: 10 }}>{f}</span>
            ))}
          </div>
        </div>
      )}

      {papers.length > 0 && (
        <div style={{ marginBottom: componentEntries.length > 0 ? 10 : 0 }}>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>Cited papers</div>
          <div className="rs-vgap-8">
            {papers.map((p, i) => (
              <div key={i} style={{ fontSize: 12, lineHeight: 1.45 }}>
                {p.arxiv_id && (
                  <a
                    href={`https://arxiv.org/abs/${p.arxiv_id}`}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="mono"
                    style={{ color: "var(--rs-primary)", marginRight: 6 }}
                  >
                    {p.arxiv_id}
                  </a>
                )}
                <span>{p.citation ?? p.description ?? "(untitled reference)"}</span>
                {p.grounded === false && (
                  <span
                    className="rs-badge amber"
                    style={{ fontSize: 9.5, marginLeft: 6 }}
                    title="cited by the model but not retrieved from the knowledge graph this iteration"
                  >
                    model-recalled
                  </span>
                )}
                {p.citation && p.description && p.description !== p.citation && (
                  <span className="rs-sub" style={{ display: "block", fontSize: 11 }}>{p.description}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {componentEntries.length > 0 && (
        <div>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>Reward components (mean over rollout)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {componentEntries.map(([name, value]) => (
              <div key={name} className="rs-flex rs-gap-8" style={{ alignItems: "center", fontSize: 11.5 }}>
                <span className="mono rs-sub" style={{ width: 140, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={name}>{name}</span>
                <span style={{ flex: 1, height: 6, background: "var(--canvas-soft)", borderRadius: 3, overflow: "hidden" }}>
                  <span
                    style={{
                      display: "block", height: "100%",
                      width: `${Math.min(100, (Math.abs(value) / maxComponentAbs) * 100)}%`,
                      background: value < 0 ? "var(--st-rose)" : "var(--st-emerald)",
                      marginLeft: value < 0 ? "auto" : 0,
                    }}
                  />
                </span>
                <span className="rs-num" style={{ width: 56, textAlign: "right", flexShrink: 0 }}>{value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {!data.reward_description && !data.evidence && papers.length === 0 && componentEntries.length === 0 && !hasScores && (
        <p className="rs-sub" style={{ fontSize: 11.5 }}>No reasoning recorded for this iteration.</p>
      )}
    </div>
  );
}

// §increment 3: disk-truth PROJECT-level view — the detail pane for the
// synthetic "disk:project" row. Mirrors StageDetailPane, but sources
// iterations + rollout from the project-scoped endpoints
// (useProjectIterations / projectIterRolloutUrl) instead of a
// mission-stage's, since plain sculpt runs write straight into
// <project_dir>/runs/iter_*/.
function ProjectDiskDetailPane({
  slug, summary,
}: { slug: string; summary: RunSummary | null }) {
  const iters = useProjectIterations(slug);
  const rows = iters.data ?? [];

  const [picked, setPicked] = useState<number | null>(null);
  const defaultIter = useMemo(() => {
    if (rows.length === 0) return null;
    const newestWithRollout = [...rows].reverse().find((r) => r.has_rollout);
    if (newestWithRollout) return newestWithRollout.iter_index;
    return rows[rows.length - 1].iter_index;
  }, [rows]);
  const activeIter = picked ?? defaultIter;
  const activeRow = rows.find((r) => r.iter_index === activeIter) ?? null;

  return (
    <div className="rs-runs-detail">
      <div className="rs-iter-col">
        <div className="rs-eyebrow" style={{ marginBottom: 12 }}>Iterations</div>
        {iters.isLoading && <p className="rs-sub" style={{ fontSize: 11 }}>Loading…</p>}
        {iters.error && <p style={{ fontSize: 11, color: "var(--st-rose)" }}>{(iters.error as Error).message}</p>}
        {!iters.isLoading && rows.length === 0 && <p className="rs-sub" style={{ fontSize: 11 }}>No iterations on disk yet for this project.</p>}
        {rows.map((r) => (
          <StageIterCard
            key={r.iter_index}
            row={r}
            selected={activeIter === r.iter_index}
            kept={false}
            onSelect={() => setPicked(r.iter_index)}
          />
        ))}
      </div>

      {/* Same clipped-overflow defect as StageDetailPane's mid-col — no
          self-scrolling child here either, so it needs its own scroll. */}
      <div className="rs-mid-col" style={{ overflowY: "auto" }}>
        <div className="rs-run-header">
          <Icon name="activity" size={17} color="var(--rs-muted)" />
          <span className="mono" style={{ fontSize: 14, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>
            project runs
          </span>
          {summary && <RunStatusBadge run={summary} />}
          <span className="rs-grow" />
          <span className="rs-eyebrow" style={{ flexShrink: 0 }}>disk-truth · recovered after restart</span>
        </div>

        {activeIter != null && activeRow?.has_rollout ? (
          <div style={{ margin: "0 16px 12px", border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
            <div className="rs-log-bar" style={{ gap: 8 }}>
              <Icon name="video" size={13} color="var(--rs-muted)" />
              <span style={{ fontSize: 11.5, fontWeight: 600 }}>iter {activeIter} rollout</span>
              <span className="rs-grow" />
              <a
                href={projectIterRolloutUrl(slug, activeIter)}
                download={`${slug}_iter_${activeIter}.mp4`}
                className="rs-btn rs-btn-quiet rs-btn-xs"
              >
                <Icon name="download" size={13} />MP4
              </a>
            </div>
            <video
              key={activeIter}
              src={projectIterRolloutUrl(slug, activeIter)}
              style={{ width: "100%", aspectRatio: "16/9", background: "#16150f", display: "block" }}
              controls
              playsInline
              preload="metadata"
            >
              <track kind="captions" />
            </video>
          </div>
        ) : activeIter != null ? (
          <p className="rs-sub" style={{ margin: "0 16px 12px", fontSize: 11 }}>iter {activeIter} has no rollout video.</p>
        ) : null}

        <div style={{ padding: "0 16px" }}>
          <p className="rs-sub" style={{ fontSize: 12, lineHeight: 1.6 }}>
            The backend restarted since these runs ended, so their live logs are
            gone — this view reads the trained iterations straight off disk.
            Iterations accumulate here across every single run of the project.
          </p>
        </div>
      </div>

      <div className="rs-extra-col">
        {activeIter != null && activeRow?.has_checkpoint && (
          <div className="rs-card rs-card-pad">
            <div className="rs-card-title" style={{ fontSize: 13, marginBottom: 8 }}>
              <Icon name="package" size={15} />Deploy this policy
            </div>
            <p className="rs-sub" style={{ margin: "0 0 10px", fontSize: 12 }}>
              Download iter {activeIter}'s checkpoint bundled with its
              reward, env spec, and an ONNX/TorchScript policy network.
            </p>
            <a
              href={policyExportUrl(slug, activeIter)}
              download
              className="rs-btn rs-btn-primary rs-btn-sm"
            >
              <Icon name="download" size={14} />Export policy bundle
            </a>
          </div>
        )}
        <StageRewardsCard slug={slug} stage={null} />
      </div>
    </div>
  );
}

/** §D24 (F4): render the top few `fitness_components` entries as a plain
 *  "name: value" tooltip string for the contradiction badge — no charts,
 *  just enough to localize which channel zeroed the fitness at a glance. */
function StageIterCard({
  row, selected, kept, selectionSource, onSelect,
}: {
  row: StageIteration;
  selected: boolean;
  kept: boolean;
  /** §UX honesty pass: `stage.selection_source` — carries WHY this
   *  iteration was kept, so the badge isn't a bare unexplained "kept". */
  selectionSource?: string | null;
  onSelect: () => void;
}) {
  const m = formatIterMetrics(row);
  return (
    <button
      className={"rs-itercard" + (selected ? " on" : "")}
      style={{ width: "100%", textAlign: "left", cursor: "pointer" }}
      onClick={onSelect}
    >
      <div className="rs-itercard-top">
        <span className="it">iter {row.iter_index}</span>
        <span style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
          {m.fitnessText && (
            <span className="rs-num" style={{ fontSize: 15, fontWeight: 600, color: "#b9aef5" }}>{m.fitnessText}</span>
          )}
          {m.rewardText && (
            <span className="rs-num" style={{ fontSize: m.fitnessText ? 11 : 13, color: m.fitnessText ? "var(--rs-muted)" : undefined }} title={m.rewardTitle}>{m.rewardText}</span>
          )}
        </span>
      </div>
      <span className="rs-flex rs-gap-6" style={{ marginTop: 4 }}>
        {row.has_rollout && <Icon name="video" size={11} color="var(--rs-muted)" />}
        {row.reward_version && <span className="rs-sub" style={{ fontSize: 10.5 }}>reward {row.reward_version}</span>}
        {kept && (
          <span className="rs-badge emerald" style={{ fontSize: 8.5 }} title={selectionSentence(selectionSource)}>
            <Icon name="check" size={9} />kept · {selectionLabel(selectionSource)}
          </span>
        )}
        {row.fitness_contradiction && (
          <span
            className="rs-badge rose"
            style={{ fontSize: 8.5 }}
            title={formatContradictionTooltip(row.fitness_components)}
          >
            <Icon name="alert-triangle" size={9} />criterion✓ fitness 0
          </span>
        )}
        {(() => {
          const nat = naturalnessChipText(row.naturalness_flag, row.naturalness_hard_reject);
          if (!nat) return null;
          return (
            <span
              className={"rs-badge " + (row.naturalness_hard_reject ? "rose" : "amber")}
              style={{ fontSize: 8.5 }}
              title={nat.title}
            >
              <Icon name="alert-triangle" size={9} />{nat.label}
            </span>
          );
        })()}
      </span>
    </button>
  );
}

// The original live-job detail pane (unchanged behavior) — used for
// single sculpt runs and for whichever mission stage is the one
// ACTIVELY training right now, so "watch it train live" keeps working.
function LiveRunDetailPane({
  slug, runId, runs, mission,
}: { slug: string; runId: string; runs: RunSummary[]; mission?: MissionDetail | null }) {
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
    // §Ship 44: launch-time calibration outcome. §Ship 51: + task-derived.
    let calib: {
      status: "running" | "done" | "skipped";
      builtin?: string; calibrated?: boolean; spearman?: number | null;
      method?: string; agreement_fraction?: number | null; reason?: string | null;
      trust?: number | null;
    } | null = null;
    for (const ev of events.events) {
      const e = ev as {
        type?: string; stage?: string; message?: string; attempt?: number;
        max?: number; gen_id?: string; reasons?: string[]; concerns?: string[];
        error?: string; builtin?: string; calibrated?: boolean; spearman?: number | null;
        method?: string; agreement_fraction?: number | null; reason?: string | null;
        trust?: number | null;
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
      else if (e.type === "metric_calibration_started") { calib = { status: "running", builtin: e.builtin, method: e.method }; }
      else if (e.type === "metric_calibration_done") {
        calib = { status: "done", builtin: e.builtin, calibrated: e.calibrated, spearman: e.spearman,
                  method: e.method, agreement_fraction: e.agreement_fraction, reason: e.reason,
                  trust: e.trust };
      }
      else if (e.type === "metric_calibration_skipped") { calib = { status: "skipped", reason: e.reason }; }
    }
    return started ? { last, outcome, genId, reasons, concerns, errorMsg, calib, awaiting } : null;
  }, [events.events]);

  const summary = useMemo(() => runs.find((r) => r.run_id === runId) ?? null, [runs, runId]);
  const isStageRun = (summary?.kind === "mission_stage_run") || (run.data?.kind === "mission_stage_run");
  const missionSlug = summary?.mission_slug ?? run.data?.mission_slug ?? null;
  const stageName = summary?.stage_name ?? run.data?.stage_name ?? null;
  const stageRewardsScope = isStageRun && missionSlug && stageName ? `${missionSlug}/${stageName}` : null;
  const liveStageLabel = mission?.stages.find((s) => s.name === stageName)?.display_label ?? null;

  const mergedIters = useMergedIterations(iters, events.events);

  // Exportable trained checkpoints (disk-backed). Stage runs export from
  // their own runs tree; plain sculpt runs share the project tree.
  const policies = usePolicies(slug, isStageRun ? { runId } : undefined);
  const exportRunId = isStageRun ? runId : undefined;
  const exportableIters = useMemo(
    () => new Set((policies.data ?? []).map((p) => p.iter_index)),
    [policies.data],
  );
  // Latest exportable iter OF THIS RUN — iter dirs accumulate across runs
  // in the project tree, so intersect with the run's own iterations.
  const latestExportable = useMemo(() => {
    const runIters = new Set(mergedIters.map((it) => it.iter_index));
    let best: number | null = null;
    for (const p of policies.data ?? []) {
      if (runIters.has(p.iter_index) && (best === null || p.iter_index > best)) {
        best = p.iter_index;
      }
    }
    return best;
  }, [policies.data, mergedIters]);
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
        {isStageRun && summary && <StageContextCard run={summary} displayLabel={liveStageLabel} />}
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
            <RunErrorCard
              slug={slug}
              error={run.data.error}
              classification={run.data.error_classification ?? null}
              iterationsCompleted={run.data.iterations_completed}
            />
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
        {!isActive && latestExportable !== null && (
          <div className="rs-card rs-card-pad">
            <div className="rs-card-title" style={{ fontSize: 13, marginBottom: 8 }}>
              <Icon name="package" size={15} />Deploy this policy
            </div>
            <p className="rs-sub" style={{ margin: "0 0 10px", fontSize: 12 }}>
              Download iter {latestExportable}'s checkpoint bundled with its
              reward, env spec, and an ONNX/TorchScript policy network.
            </p>
            <a
              href={policyExportUrl(slug, latestExportable, exportRunId)}
              download
              className="rs-btn rs-btn-primary rs-btn-sm"
            >
              <Icon name="download" size={14} />Export policy bundle
            </a>
          </div>
        )}
        {isActive && <RunGpuCard />}
        {selectedIter !== null && (
          <IterationDetailCard
            iter={mergedIters.find((it) => it.iter_index === selectedIter) ?? null}
            exportHref={
              selectedIter !== null && exportableIters.has(selectedIter)
                ? policyExportUrl(slug, selectedIter, exportRunId)
                : null
            }
          />
        )}
      </div>
    </div>
  );
}

function StageContextCard({ run, displayLabel = null }: { run: RunSummary; displayLabel?: string | null }) {
  // §Increment 4: prefer the server's display_label; fall back to
  // stage_index+1 while the mission-detail lookup is loading (or absent).
  const numberLabel = displayLabel ?? (typeof run.stage_index === "number" ? String(run.stage_index + 1) : null);
  return (
    <div className="rs-card rs-card-pad" style={{ marginBottom: 0 }}>
      <div className="rs-flex rs-wrap rs-gap-8" style={{ fontSize: 12 }}>
        <Icon name="sparkles" size={14} color="var(--rs-primary)" />
        <span style={{ fontWeight: 500 }}>
          {numberLabel ? `Stage ${numberLabel}: ` : "Stage: "}
          <code className="mono">{run.stage_name ?? "(unnamed)"}</code>
        </span>
        <span style={{ color: "var(--rs-muted)" }}>mission <code className="mono">{run.mission_slug}</code></span>
        {run.behavior_goal && <span style={{ color: "var(--rs-muted)" }}>{run.behavior_goal}</span>}
      </div>
    </div>
  );
}

// `stage` null ⇒ the project-global reward versions (used by the
// disk-truth project pane); a `<missionSlug>/<stageName>` string keeps
// the original per-stage behavior.
function StageRewardsCard({ slug, stage }: { slug: string; stage: string | null }) {
  const versions = useRewards(slug, stage);
  return (
    <div className="rs-card">
      <div className="rs-card-head"><div className="rs-card-title" style={{ fontSize: 13 }}><Icon name="file-code" size={15} />{stage ? "Stage rewards" : "Rewards"}</div></div>
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
      {run && <RunStatusBadge run={run} />}
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
                {/* §Convergence loop 1: dense sub-success progress — the
                    ranking signal below the completion gate. Shown only
                    when the metric emits progress_score. */}
                {typeof it.progress === "number" && (
                  <span className="rs-num" title="dense progress (metric progress_score, 0-1) — ranks iterations below the success gate" style={{ fontSize: 11, color: "#8ec8a6" }}>prog {it.progress.toFixed(3)}</span>
                )}
                {it.primary_metric !== null && (
                  <span className="rs-num" title="reward metric (secondary)" style={{ fontSize: 11, color: "var(--rs-muted)" }}>r {it.primary_metric.toFixed(1)}</span>
                )}
              </span>
            ) : (
              it.primary_metric !== null && (
                <span className="rs-num" title="reward metric — no objective fitness tracked this iter" style={{ fontSize: 13 }}>r {it.primary_metric.toFixed(1)}</span>
              )
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
            <span className="ver" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }} title={it.failure_modes.join(", ")}>{it.failure_modes.join(", ")}</span>
          )}
          {it.realism_audit && typeof it.realism_audit.verdict === "string" && it.realism_audit.verdict !== "ok" && it.realism_audit.verdict !== "unknown" && (
            <span className="rs-tag" style={{ marginTop: 4, fontSize: 10, background: it.realism_audit.verdict === "severe" ? "var(--st-rose-bg)" : "var(--st-amber-bg)", color: it.realism_audit.verdict === "severe" ? "var(--st-rose-fg)" : "var(--st-amber-fg)" }}>
              physics: {it.realism_audit.verdict}
            </span>
          )}
          {it.env_spec_update && (it.env_spec_update.applied.length > 0 || it.env_spec_update.rejected.length > 0) && (
            <span
              className="rs-tag mono"
              style={{ marginTop: 4, fontSize: 10, background: "var(--st-blue-bg)", color: "var(--st-blue-fg)", display: "inline-block" }}
              title={[
                ...it.env_spec_update.applied.map((a) => `applied: ${a}`),
                ...it.env_spec_update.rejected.map((r) => `rejected ${r.parameter}: ${r.reason}`),
              ].join("; ") + " | training-only env curriculum; takes effect next iteration"}
            >
              env {it.env_spec_update.new_version ? `→ ${it.env_spec_update.new_version}` : "edits rejected"}
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
                  if (match) window.location.assign(`/projects/${match[1]}?tab=physics`);
                }}
                className="rs-tag"
                style={{ marginTop: 4, fontSize: 10, cursor: disabled ? "default" : "pointer", background: state === "applied" ? "var(--st-emerald-bg)" : "var(--st-blue-bg)", color: state === "applied" ? "var(--st-emerald-fg)" : "var(--st-blue-fg)" }}
                title={it.physics_edit_suggestion.auto_apply_reason ?? "Open Physics tab with this prompt pre-filled"}
              >
                {label}
              </span>
            );
          })()}
          {it.env_extension_suggestion && it.env_extension_suggestion.terms.length > 0 && (
            <span
              className="rs-tag"
              style={{ marginTop: 4, fontSize: 10, background: "var(--st-amber-bg)", color: "var(--st-amber-fg)" }}
              title={
                `The diagnoser proposed ${it.env_extension_suggestion.terms.join(", ")} ` +
                `but the adapter doesn't expose the fields they need, so it was deferred. ` +
                `Extend the adapter's expected_info_keys to unblock this skill.` +
                ((it.env_extension_suggestion.rationales || []).length
                  ? "\n\n" + (it.env_extension_suggestion.rationales || []).join("\n\n")
                  : "")
              }
            >
              needs adapter channels: {it.env_extension_suggestion.terms.join(", ")}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

function RunErrorCard({
  slug, error, classification, iterationsCompleted,
}: { slug: string; error: string; classification: ErrorClassification | null; iterationsCompleted?: number }) {
  const regen = useRegenerateRewardTemplate(slug);
  const isContractMismatch = classification?.kind === "reward_contract_mismatch" || classification?.action?.kind === "regenerate_reward_template";

  // §Ship: these two error strings mark benign, designed-for outcomes, not
  // crashes — render them as friendly non-alarming cards instead of the
  // raw-string red error banner.
  if (error === CRITERION_NOT_MET_ERROR) {
    return (
      <div className="rs-banner warn" style={{ flexDirection: "column", alignItems: "stretch", gap: 8 }}>
        <div className="rs-flex rs-gap-8">
          <Icon name="alert-circle" size={17} />
          <span className="rs-grow"><b>Success criterion not met.</b></span>
        </div>
        <p style={{ margin: 0, fontSize: 12.5 }}>
          Training completed {iterationsCompleted ?? "N"} iteration{iterationsCompleted === 1 ? "" : "s"} — the
          stage's success criterion wasn't met, so the mission re-planned the stage.
        </p>
      </div>
    );
  }
  if (error === MISSION_TERMINATED_ERROR) {
    return (
      <div className="rs-banner" style={{ flexDirection: "column", alignItems: "stretch", gap: 8, background: "var(--canvas-soft)" }}>
        <div className="rs-flex rs-gap-8">
          <Icon name="square" size={17} />
          <span className="rs-grow"><b>Stopped — mission was cancelled.</b></span>
        </div>
        <p style={{ margin: 0, fontSize: 12.5, color: "var(--rs-muted)" }}>
          This stage run was in progress when the mission was stopped; it did not fail on its own.
        </p>
      </div>
    );
  }

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

function IterationDetailCard({ iter, exportHref }: { iter: IterEventSummary | null; exportHref: string | null }) {
  if (!iter) return null;
  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-card-title" style={{ fontSize: 13, marginBottom: 10 }}><Icon name="circle-dot" size={15} />Iter {iter.iter_index} · {iter.status}</div>
      <div className="rs-vgap-8">
        {exportHref && (
          <div>
            <a
              href={exportHref}
              download
              className="rs-btn rs-btn-ghost rs-btn-sm"
              title="Download this iteration's checkpoint bundled with its reward, env spec, and ONNX/TorchScript policy"
            >
              <Icon name="download" size={14} />Export policy bundle
            </a>
          </div>
        )}
        {iter.failure_modes.length > 0 && (
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 4 }}>failure modes</div>
            <div className="rs-flex rs-wrap rs-gap-6">{iter.failure_modes.map((f) => <span key={f} className="rs-tag mono" style={{ fontSize: 10 }}>{f}</span>)}</div>
          </div>
        )}
        {iter.edit_count !== null && <div className="rs-flex-between" style={{ fontSize: 12 }}><span className="rs-sub">edits</span><span className="rs-num">{iter.edit_count}</span></div>}
        {iter.env_spec_update && (
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 4 }}>
              env spec{iter.env_spec_update.new_version ? ` → ${iter.env_spec_update.new_version}` : ""}
            </div>
            <div className="rs-flex rs-wrap rs-gap-6">
              {iter.env_spec_update.applied.map((a) => (
                <span key={a} className="rs-tag mono" style={{ fontSize: 10 }} title="applied to the training-only env curriculum (next iteration)">{a}</span>
              ))}
              {iter.env_spec_update.rejected.map((r, i) => (
                <span key={`rej-${i}`} className="rs-tag mono" style={{ fontSize: 10, opacity: 0.55, textDecoration: "line-through" }} title={r.reason}>{r.parameter}</span>
              ))}
            </div>
          </div>
        )}
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
    progress: winner.progress ?? loser.progress,
    env_spec_update: winner.env_spec_update ?? loser.env_spec_update,
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
      method?: string; agreement_fraction?: number | null; reason?: string | null;
      trust?: number | null;
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
                <span>{calib.method === "task_derived"
                  ? "Calibrating vs 3 independently-generated competence ladders…"
                  : <>Calibrating vs <code className="mono">{calib.builtin}</code>…</>}</span>
              </div>
            )}
            {calib?.status === "done" && calib.calibrated && (
              <div style={{ color: "var(--st-green-fg, var(--ink))" }}>
                {calib.method === "task_derived" ? (
                  <>Competence ladders agree
                    {typeof calib.spearman === "number" ? ` (rho_min ${calib.spearman})` : ""}
                    {" "}— <strong>steering</strong> the run.</>
                ) : (
                  <>Calibrated vs <code className="mono">{calib.builtin}</code>
                    {typeof calib.spearman === "number" ? ` (Spearman ${calib.spearman})` : ""}
                    {" "}— <strong>steering</strong> the run.</>
                )}
                {typeof calib.trust === "number" ? (
                  <span style={{ color: "var(--rs-muted)" }}> · trust {calib.trust}</span>
                ) : ""}
              </div>
            )}
            {calib?.status === "done" && !calib.calibrated && (
              <div style={{ color: "var(--st-amber-fg)" }}>
                {calib.method === "task_derived" ? (
                  <>{calib.reason || "Task-derived ladders disagree"} — runs <strong>observe-only</strong>.</>
                ) : (
                  <>Did not pass calibration vs <code className="mono">{calib.builtin}</code>
                    {typeof calib.spearman === "number" ? ` (Spearman ${calib.spearman} < 0.7)` : ""}
                    {" "}— runs <strong>observe-only</strong>.</>
                )}
              </div>
            )}
            {calib?.status === "skipped" && (
              <div style={{ color: "var(--rs-muted)" }}>
                {calib.reason || "No matching built-in ground truth"} — runs observe-only.
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
        if (typeof ev.progress === "number") slot.progress = ev.progress;
      }
      if (ev.type === "best_reward_selected") {
        if (typeof ev.fitness === "number") slot.best_fitness = ev.fitness;
      }
      // §env generalization: the diagnoser's env-curriculum change.
      if (ev.type === "env_spec_updated") {
        slot.env_spec_update = {
          new_version: typeof ev.new_version === "string" ? ev.new_version : null,
          applied: Array.isArray(ev.applied) ? (ev.applied as string[]) : [],
          rejected: Array.isArray(ev.rejected) ? (ev.rejected as Array<{ parameter: string; reason: string }>) : [],
        };
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

import { useEffect, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";

import { ActuatorLimitsCard } from "@/components/ActuatorLimitsCard";
import { Icon } from "@/components/rs/icon";
import { Badge, Btn, EmptyState } from "@/components/rs/primitives";
import { useMission, useMissions, useStageIterations } from "@/hooks/useMissions";
import { usePolicies } from "@/hooks/usePolicies";
import {
  ApiError,
  getReportsSources,
  policyExportUrl,
  stageCheckpointUrl,
  stageExportUrl,
  stageRolloutUrl,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import { stageLabel } from "@/lib/stageDisplay";
import type { SelectedStage, StageIteration, StageSchema } from "@/lib/types";

/** A report source: the project's standalone runs, or one mission. */
type ReportSource =
  | { kind: "project" }
  | { kind: "mission"; missionSlug: string };

const PROJECT_SOURCE_VALUE = "__project__";

function sourceToValue(s: ReportSource): string {
  return s.kind === "project" ? PROJECT_SOURCE_VALUE : s.missionSlug;
}

/** URL bases differ per source: project reports live under /reports,
 *  a mission's under /missions/{ms}/report. */
function reportBase(slug: string, source: ReportSource): string {
  return source.kind === "project"
    ? `/api/projects/${slug}/reports`
    : `/api/projects/${slug}/missions/${source.missionSlug}/report`;
}

async function fetchReportMd(slug: string, source: ReportSource): Promise<string> {
  const r = await fetch(`${reportBase(slug, source)}/final_report.md`);
  if (r.status === 404) return ""; // "not built yet" signal
  if (!r.ok) {
    let body: Record<string, unknown> = {};
    try {
      body = await r.json();
    } catch {
      /* ignore */
    }
    throw new ApiError({
      type: String(body.type ?? "about:blank"),
      title: String(body.title ?? r.statusText),
      status: r.status,
      detail: String(body.detail ?? ""),
    });
  }
  return await r.text();
}

// §Ship 25b (H2): decomposition-quality telemetry per mission.
interface MissionQualityRecord {
  mission_slug: string;
  goal: string;
  n_stages_at_start: number;
  n_stages_final: number;
  stages_executed: number;
  stages_succeeded: number;
  stage_success_rate: number | null;
  redecompositions: number;
  iterations_total: number;
  completed: boolean;
  halted_reason: string | null;
  recorded_at: string;
}

async function fetchMissionQuality(slug: string): Promise<MissionQualityRecord[]> {
  const r = await fetch(`/api/projects/${slug}/reports/mission-quality`);
  if (!r.ok) return [];
  const body = (await r.json()) as { missions?: MissionQualityRecord[] };
  return body.missions ?? [];
}

async function buildReport(slug: string, source: ReportSource): Promise<void> {
  // The build endpoint is always /reports/build; a mission build passes
  // {mission_slug} in the body (empty body = project-runs build).
  const init: RequestInit = { method: "POST" };
  if (source.kind === "mission") {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify({ mission_slug: source.missionSlug });
  }
  const r = await fetch(`/api/projects/${slug}/reports/build`, init);
  if (!r.ok) {
    let body: Record<string, unknown> = {};
    try {
      body = await r.json();
    } catch {
      /* ignore */
    }
    throw new ApiError({
      type: String(body.type ?? "about:blank"),
      title: String(body.title ?? r.statusText),
      status: r.status,
      detail: String(body.detail ?? ""),
    });
  }
}

export function ReportsTab({
  slug, selectedStage, setSelectedStage,
}: {
  slug: string;
  // §Ship de-silo: the shared cross-tab stage selection (owned by
  // ProjectDetail, URL-synced as `?stage=missionSlug/stageName`). When
  // present it takes precedence over this tab's own sticky source pick —
  // see `sourceValue` effect below, mirrored from RewardsTab's scope
  // precedence (commits b5c1d4f / 01b7df6).
  selectedStage?: SelectedStage | null;
  setSelectedStage?: (value: SelectedStage | null) => void;
}) {
  const qc = useQueryClient();

  // §Ship 20: report source picker — the project's standalone runs, or
  // any mission. Fed by GET /reports/sources.
  const sources = useQuery({
    queryKey: [...qk.project(slug), "report", "sources"],
    queryFn: () => getReportsSources(slug),
    staleTime: 10_000,
  });
  const missions = useMissions(slug);

  const selectedStageMission = selectedStage?.missionSlug ?? null;

  // The "best default" mission when nothing has been explicitly picked:
  // a running mission wins (that's what the user is watching), else the
  // most recently created one. Only used before any user/prop pick lands.
  const defaultMissionSlug = useMemo(() => {
    const list = missions.data ?? [];
    if (list.length === 0) return null;
    const running = list.find((m) => m.lifecycle === "running");
    if (running) return running.mission_slug;
    return list.reduce((best, m) =>
      new Date(m.created_at).getTime() > new Date(best.created_at).getTime() ? m : best,
    ).mission_slug;
  }, [missions.data]);

  // Sticky manual/derived pick. `null` = "not decided yet" (falls through
  // to the precedence below); PROJECT_SOURCE_VALUE = explicit project
  // pick; anything else = a mission slug.
  const [sourceValue, setSourceValueRaw] = useState<string | null>(null);
  // Tracks the last selectedStageMission we reacted to, so a transition
  // to a NEW non-null mission always wins over a stale local pick — same
  // "external click should win" rule RewardsTab uses for scopeOverride,
  // but a transition to null must NOT clear an explicit user pick.
  const [lastPropMission, setLastPropMission] = useState<string | null>(null);
  useEffect(() => {
    if (selectedStageMission !== lastPropMission) {
      setLastPropMission(selectedStageMission);
      if (selectedStageMission !== null && selectedStageMission !== sourceValue) {
        setSourceValueRaw(selectedStageMission);
      }
    }
  }, [selectedStageMission, lastPropMission, sourceValue]);

  const setSourceValue = (next: string) => {
    // A user picking "Project runs" from the dropdown while a shared
    // stage is selected should stick — clear the shared prop too, or the
    // effect above would immediately snap back to the mission.
    if (setSelectedStage && selectedStage) setSelectedStage(null);
    setSourceValueRaw(next);
  };

  // Effective precedence: explicit shared selectedStage prop > sticky
  // local pick (dropdown, incl. a prior prop-driven pick) > best-default
  // mission > "Project runs" only when the project has no missions.
  const effectiveSourceValue: string = useMemo(() => {
    if (selectedStageMission !== null) return selectedStageMission;
    if (sourceValue !== null) return sourceValue;
    if (defaultMissionSlug !== null) return defaultMissionSlug;
    return PROJECT_SOURCE_VALUE;
  }, [selectedStageMission, sourceValue, defaultMissionSlug]);

  const source: ReportSource = useMemo(
    () =>
      effectiveSourceValue === PROJECT_SOURCE_VALUE
        ? { kind: "project" }
        : { kind: "mission", missionSlug: effectiveSourceValue },
    [effectiveSourceValue],
  );

  // §Increment 3: which stage within the picked mission. Sticky local
  // pick (cleared whenever the mission source changes so a stale stage
  // name from a different mission can't leak); defaults to the shared
  // selectedStage's stage when it belongs to this mission, else the
  // mission's current stage.
  const sourceMissionSlug = source.kind === "mission" ? source.missionSlug : null;
  const missionDetail = useMission(slug, sourceMissionSlug ?? undefined);
  const [stageOverride, setStageOverride] = useState<string | null>(null);
  useEffect(() => {
    setStageOverride(null);
  }, [sourceMissionSlug]);

  const stages = missionDetail.data?.stages ?? [];
  const effectiveStageName: string | null = useMemo(() => {
    if (sourceMissionSlug === null) return null;
    if (stageOverride && stages.some((s) => s.name === stageOverride)) return stageOverride;
    if (selectedStage && selectedStage.missionSlug === sourceMissionSlug
        && stages.some((s) => s.name === selectedStage.stageName)) {
      return selectedStage.stageName;
    }
    // current_stage_idx indexes mission.json's own stages[] — synthetic
    // on_disk_only entries are UNION insertions, so filter them out before
    // indexing or the pointer lands on a recovered superseded parent.
    const runnable = stages.filter((s) => !s.on_disk_only);
    const current = missionDetail.data
      ? runnable[missionDetail.data.current_stage_idx]
      : undefined;
    if (current) return current.name;
    return stages.length > 0 ? stages[stages.length - 1].name : null;
  }, [sourceMissionSlug, stageOverride, selectedStage, stages, missionDetail.data]);

  const onPickStage = (stageName: string) => {
    setStageOverride(stageName);
    if (setSelectedStage && sourceMissionSlug !== null) {
      setSelectedStage({ missionSlug: sourceMissionSlug, stageName });
    }
  };

  // §Problem 4b: the hero video is a CLEAN SINGLE-ROBOT playback of the
  // mission's FINAL stage — the last entry in mission.stages — not the
  // report's tiled final.mp4 and not necessarily the currently-selected
  // stage above. Prefer the last non-superseded stage (a superseded
  // parent's own rollout isn't the mission's final policy); fall back to
  // the literal last entry if every stage is superseded (edge case).
  const finalStage: StageSchema | null = useMemo(() => {
    if (stages.length === 0) return null;
    const notSuperseded = stages.filter((s) => s.status !== "superseded");
    return notSuperseded.length > 0
      ? notSuperseded[notSuperseded.length - 1]
      : stages[stages.length - 1];
  }, [stages]);
  const finalStageIters = useStageIterations(
    slug, sourceMissionSlug ?? undefined, finalStage?.name,
  );
  // Prefer the stage's KEPT (best) iter; fall back to the newest iter on
  // disk that actually has a rollout (kept iter may predate a rollout,
  // or selection may not have run yet).
  const heroIterIndex: number | null = useMemo(() => {
    const kept = finalStage?.selected_iter_index ?? null;
    const rows = finalStageIters.data ?? [];
    if (kept != null && rows.some((r) => r.iter_index === kept && r.has_rollout)) return kept;
    const withRollout = rows.filter((r) => r.has_rollout);
    if (withRollout.length === 0) return null;
    return withRollout.reduce((best, r) => (r.iter_index > best.iter_index ? r : best)).iter_index;
  }, [finalStage, finalStageIters.data]);
  const heroVideoUrl: string | null =
    sourceMissionSlug && finalStage && heroIterIndex != null
      ? stageRolloutUrl(slug, sourceMissionSlug, finalStage.name, heroIterIndex)
      : null;

  const md = useQuery<string>({
    queryKey: [...qk.project(slug), "report", "md", sourceToValue(source)],
    queryFn: () => fetchReportMd(slug, source),
    staleTime: 10_000,
  });
  const build = useMutation<void, Error, void>({
    mutationFn: () => buildReport(slug, source),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...qk.project(slug), "report"] });
      toast.success("Report built", { description: "final_report.md + final.mp4 regenerated" });
    },
    onError: (err) => {
      const detail = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
      toast.error("Report build failed", { description: detail });
    },
  });

  const quality = useQuery<MissionQualityRecord[]>({
    queryKey: [...qk.project(slug), "report", "mission-quality"],
    queryFn: () => fetchMissionQuality(slug),
    staleTime: 10_000,
  });

  const hasReport = (md.data ?? "").trim().length > 0;
  const base = reportBase(slug, source);
  const mp4Url = `${base}/final.mp4`;
  const mdUrl = `${base}/final_report.md`;
  const missionList = sources.data?.missions ?? [];

  return (
    <div className="rs-scroll">
      <div className="rs-pad">
        <div className="rs-flex-between rs-wrap rs-gap-12" style={{ marginBottom: 22 }}>
          <div>
            <div className="rs-eyebrow">policies · report · actuator limits</div>
            <h2 className="rs-h2" style={{ marginTop: 6 }}>Results</h2>
          </div>
          <div className="rs-flex rs-gap-8">
            {missionList.length > 0 && (
              <div className="rs-select" style={{ display: "flex", alignItems: "center" }}>
                <select
                  value={effectiveSourceValue}
                  onChange={(e) => setSourceValue(e.target.value)}
                  aria-label="Report source"
                  title="Choose which run or mission to report on"
                >
                  <option value={PROJECT_SOURCE_VALUE}>
                    Project runs ({sources.data?.project_runs.n_iters ?? 0} iters)
                  </option>
                  {missionList.map((m) => (
                    <option key={m.mission_slug} value={m.mission_slug}>
                      Mission: {m.mission_slug}
                      {m.has_report ? " ✓" : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}
            {hasReport && (
              <Btn
                kind="ghost"
                icon="copy"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(md.data ?? "");
                    toast.success("Markdown copied");
                  } catch {
                    toast.error("Clipboard unavailable");
                  }
                }}
              >
                Copy markdown
              </Btn>
            )}
            {hasReport && (
              <a href={mdUrl} download="final_report.md" className="rs-btn rs-btn-ghost">
                <Icon name="download" size={15} />
                Download
              </a>
            )}
            <Btn kind={hasReport ? "ghost" : "primary"} icon="refresh-cw" disabled={build.isPending} onClick={() => build.mutate()}>
              {build.isPending ? "Building…" : hasReport ? "Rebuild" : "Build report"}
            </Btn>
          </div>
        </div>

        {source.kind === "project" ? (
          <PoliciesCard slug={slug} />
        ) : (
          <>
            <StagePicker
              stages={stages}
              isLoading={missionDetail.isLoading}
              selected={effectiveStageName}
              onSelect={onPickStage}
            />
            <StageCheckpointsCard
              slug={slug}
              missionSlug={source.missionSlug}
              stageName={effectiveStageName}
              stage={stages.find((s) => s.name === effectiveStageName) ?? null}
            />
          </>
        )}

        {source.kind === "mission" ? (
          <ActuatorLimitsCard
            slug={slug}
            missionSlug={source.missionSlug}
            stageName={effectiveStageName}
          />
        ) : (
          <ActuatorLimitsCard slug={slug} />
        )}

        {source.kind === "mission" && (
          <HeroVideoCard
            stage={finalStage}
            stageIndex1Based={stages.length}
            iterIndex={heroIterIndex}
            videoUrl={heroVideoUrl}
            isLoading={missionDetail.isLoading || finalStageIters.isLoading}
          />
        )}

        {(quality.data?.length ?? 0) > 0 && (
          <div className="rs-card" style={{ marginBottom: 22 }}>
            <div className="rs-card-head">
              <div className="rs-card-title">
                <Icon name="layers" size={16} />Mission quality
              </div>
              <span className="rs-sub" style={{ fontSize: 12 }}>
                decomposition telemetry
              </span>
            </div>
            <div className="rs-card-pad rs-vgap-8">
              {quality.data!.map((m) => (
                <div
                  key={m.mission_slug}
                  className="rs-flex rs-gap-12 rs-wrap"
                  style={{
                    border: "1px solid var(--hairline)",
                    borderRadius: "var(--radius-md)",
                    padding: "10px 12px",
                    background: "var(--canvas-soft)",
                    alignItems: "baseline",
                    fontSize: 12.5,
                  }}
                >
                  <span style={{ fontWeight: 500, minWidth: 140 }} title={m.goal}>
                    {m.mission_slug}
                  </span>
                  <span className="rs-num">
                    stages {m.stages_succeeded}/{m.stages_executed}
                    {m.stage_success_rate != null &&
                      ` (${Math.round(m.stage_success_rate * 100)}%)`}
                  </span>
                  <span className="rs-num">
                    {m.n_stages_at_start === m.n_stages_final
                      ? `${m.n_stages_final} planned`
                      : `${m.n_stages_at_start}→${m.n_stages_final} planned`}
                  </span>
                  <span className="rs-num">
                    {m.redecompositions} redecompose{m.redecompositions === 1 ? "" : "s"}
                  </span>
                  <span className="rs-num">{m.iterations_total} iters</span>
                  <span
                    style={{
                      marginLeft: "auto",
                      color: m.completed ? "var(--st-emerald)" : "var(--st-amber)",
                      fontWeight: 500,
                    }}
                  >
                    {m.completed ? "completed" : (m.halted_reason ?? "halted")}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {md.isLoading ? (
          <p className="rs-sub">Loading report…</p>
        ) : md.error ? (
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">{(md.error as Error).message}</span>
          </div>
        ) : !hasReport ? (
          <div className="rs-card">
            <EmptyState
              icon="file-text"
              title="No report built yet"
              sub={
                source.kind === "mission"
                  ? `Build the report for mission ${source.missionSlug} to render its per-stage report + stitched timelapse.`
                  : "Complete a sculpt run, then click Build report to render final_report.md + the timelapse."
              }
            />
          </div>
        ) : (
          <div className="rs-report rs-card rs-card-pad" style={{ padding: "32px 36px" }}>
            <div className="rs-flex rs-gap-8" style={{ marginBottom: 20, alignItems: "center" }}>
              <span className="rs-sub" style={{ fontSize: 11.5 }}>
                Side-by-side timelapse of every stage, stitched into one clip:
              </span>
              <a href={mp4Url} download="final.mp4" className="rs-btn rs-btn-quiet rs-btn-sm">
                <Icon name="video" size={13} />final.mp4
                <Icon name="download" size={13} />
              </a>
            </div>
            <div className="rs-md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{md.data ?? ""}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Hero video (final policy, clean single view) ───────────────────────
// §Problem 4b: the report's final.mp4 is a side-by-side timelapse the
// backend TILES per stage — useful as an archive artifact but a bad
// "here's the robot" clip. The hero is instead a single clean rollout
// of the mission's FINAL stage at its kept (or newest-with-rollout)
// iteration, sourced straight from the stage rollout endpoint.
function HeroVideoCard({
  stage, stageIndex1Based, iterIndex, videoUrl, isLoading,
}: {
  stage: StageSchema | null;
  stageIndex1Based: number;
  iterIndex: number | null;
  videoUrl: string | null;
  isLoading: boolean;
}) {
  const label = stage ? stageLabel(stage, stageIndex1Based) : null;
  const displayName = stage?.display_label ?? stage?.name ?? null;
  return (
    <div className="rs-card" style={{ marginBottom: 22 }}>
      <div className="rs-card-head">
        <div className="rs-card-title">
          <Icon name="video" size={16} />
          Final policy{displayName ? ` — ${displayName}` : label ? ` — stage ${label}` : ""}
          {iterIndex != null ? ` · iter ${iterIndex}` : ""}
        </div>
        <span className="rs-sub" style={{ fontSize: 12 }}>
          clean single-robot rollout · last stage's kept checkpoint
        </span>
      </div>
      <div className="rs-card-pad">
        {isLoading ? (
          <p className="rs-sub" style={{ margin: 0 }}>Loading…</p>
        ) : !videoUrl ? (
          <EmptyState
            icon="video"
            title="No rollout yet"
            sub="The mission's final stage hasn't produced a rollout video yet."
          />
        ) : (
          <div className="rs-viewer">
            <video
              key={videoUrl}
              src={videoUrl}
              className="rs-viewer-stage"
              style={{ width: "100%", aspectRatio: "16/9", background: "#16150f" }}
              controls
              playsInline
              preload="metadata"
            >
              <track kind="captions" />
            </video>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Stage picker (mission sources) ─────────────────────────────────────
// Lists every stage the disk-union endpoint returns — including
// superseded ones, since their checkpoints/rollouts still matter for a
// report. Selecting a stage propagates through setSelectedStage so the
// other tabs follow (mirrors RewardsTab's scope-selector onChange).
function StagePicker({
  stages, isLoading, selected, onSelect,
}: {
  stages: StageSchema[];
  isLoading: boolean;
  selected: string | null;
  onSelect: (stageName: string) => void;
}) {
  if (isLoading) {
    return (
      <div className="rs-card rs-card-pad" style={{ marginBottom: 22 }}>
        <p className="rs-sub" style={{ margin: 0 }}>Loading stages…</p>
      </div>
    );
  }
  if (stages.length === 0) return null;
  return (
    <div className="rs-card" style={{ marginBottom: 22 }}>
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="list" size={16} />Stage</div>
        <span className="rs-sub" style={{ fontSize: 12 }}>{stages.length} on disk</span>
      </div>
      <div className="rs-card-pad rs-flex rs-wrap rs-gap-8">
        {stages.map((s) => {
          const isSel = s.name === selected;
          return (
            <button
              key={s.name}
              onClick={() => onSelect(s.name)}
              className="rs-flex rs-gap-8"
              style={{
                border: "1px solid " + (isSel ? "var(--rs-primary)" : "var(--hairline)"),
                background: isSel ? "rgba(245,78,0,0.06)" : "var(--canvas-soft)",
                borderRadius: "var(--radius-md)",
                padding: "7px 11px",
                cursor: "pointer",
                fontSize: 12.5,
                alignItems: "center",
              }}
              title={s.goal_text || s.name}
            >
              <span className="mono" style={{ fontWeight: 600 }}>{s.display_label ?? s.name}</span>
              <span className="rs-sub" style={{ fontSize: 11.5 }}>{s.name}</span>
              <Badge status={s.status} label={s.status === "superseded" ? "Superseded" : undefined} />
              {s.on_disk_only && (
                <span className="rs-tag" style={{ fontSize: 9.5 }} title="Reconstructed from disk, not mission.json">
                  disk-only
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ── Per-stage checkpoints (mission sources) ──────────────────────────
// Replaces the unscoped project-level PoliciesCard for mission sources:
// the mission's per-stage checkpoints live under the stage's own runs
// tree, served by the disk-truth iterations endpoint — not the
// project's runs tree that usePolicies(slug) sees.
function StageCheckpointsCard({
  slug, missionSlug, stageName, stage,
}: {
  slug: string;
  missionSlug: string;
  stageName: string | null;
  stage: StageSchema | null;
}) {
  const iters = useStageIterations(slug, missionSlug, stageName ?? undefined);
  const rows: StageIteration[] = iters.data ?? [];
  const selectedIter = stage?.selected_iter_index ?? null;

  return (
    <div className="rs-card" style={{ marginBottom: 22 }}>
      <div className="rs-card-head">
        <div className="rs-card-title">
          <Icon name="package" size={16} />Stage checkpoints
        </div>
        <span className="rs-sub" style={{ fontSize: 12 }}>
          {stageName ? <code className="mono">{stageName}</code> : "no stage"} · disk-truth iterations
        </span>
      </div>
      {!stageName ? (
        <EmptyState icon="package" title="No stage selected" sub="Pick a stage above to see its trained checkpoints." />
      ) : iters.isLoading ? (
        <p className="rs-sub" style={{ padding: "12px 16px" }}>Loading…</p>
      ) : rows.length === 0 ? (
        <EmptyState
          icon="package"
          title="No iterations on disk yet"
          sub="This stage hasn't produced a training iteration yet."
        />
      ) : (
        <div className="rs-card-pad rs-vgap-8">
          {stage?.selection_source && selectedIter != null && (
            <p className="rs-sub" style={{ margin: "0 0 4px", fontSize: 11.5 }}>
              kept iter <b className="mono">{selectedIter}</b> as the stage's policy — {stage.selection_source}
            </p>
          )}
          {rows.map((it) => {
            const isSelected = selectedIter != null && it.iter_index === selectedIter;
            return (
              <div
                key={it.iter_index}
                className="rs-flex rs-gap-12 rs-wrap"
                style={{
                  border: "1px solid " + (isSelected ? "var(--rs-primary)" : "var(--hairline)"),
                  borderRadius: "var(--radius-md)",
                  padding: "10px 12px",
                  background: isSelected ? "rgba(245,78,0,0.05)" : "var(--canvas-soft)",
                  alignItems: "center",
                  fontSize: 12.5,
                }}
              >
                <span style={{ fontWeight: 500, minWidth: 60 }}>iter {it.iter_index}</span>
                {it.reward_version && (
                  <span className="rs-tag mono" style={{ fontSize: 10 }}>
                    reward {it.reward_version}
                  </span>
                )}
                {it.fitness != null ? (
                  <span className="rs-num" title="objective fitness (0-1)">fit {it.fitness.toFixed(2)}</span>
                ) : it.primary_metric != null ? (
                  <span className="rs-num" title="mean return">{it.primary_metric.toFixed(1)}</span>
                ) : null}
                {isSelected && (
                  <span className="rs-tag" style={{ fontSize: 10, color: "var(--st-emerald)" }} title={stage?.selection_source ?? undefined}>
                    kept
                  </span>
                )}
                <span style={{ marginLeft: "auto" }} className="rs-flex rs-gap-8" title={
                  it.has_checkpoint ? "Builds ONNX + TorchScript + reward/env spec + DEPLOY.md…" : undefined
                }>
                  {it.has_rollout && (
                    <a
                      href={stageRolloutUrl(slug, missionSlug, stageName, it.iter_index)}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="rs-btn rs-btn-ghost rs-btn-sm"
                      title="Open the rollout video"
                    >
                      <Icon name="video" size={14} />Rollout
                    </a>
                  )}
                  {it.has_checkpoint && (
                    <a
                      href={stageExportUrl(slug, missionSlug, stageName, it.iter_index)}
                      download
                      className={"rs-btn rs-btn-sm " + (isSelected ? "rs-btn-primary" : "rs-btn-ghost")}
                      title="Download the deployment bundle: checkpoint + ONNX + TorchScript + reward/env spec + DEPLOY.md (builds server-side, may take a moment)"
                    >
                      <Icon name="package" size={14} />Export bundle (.zip)
                    </a>
                  )}
                  {it.has_checkpoint && (
                    <a
                      href={stageCheckpointUrl(slug, missionSlug, stageName, it.iter_index)}
                      download
                      className="rs-btn rs-btn-quiet rs-btn-sm"
                      title="Download the raw checkpoint file only"
                    >
                      <Icon name="download" size={14} />raw .pt
                    </a>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Trained policies (deployment-bundle export) ───────────────────────
function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

function PoliciesCard({ slug }: { slug: string }) {
  const policies = usePolicies(slug);
  const rows = policies.data ?? [];
  // Rank on ONE scale: fitness (0-1) when any row has it, else the reward
  // metric — mixing the two in a single max lands "best" on the wrong row.
  const anyFitness = rows.some((r) => r.fitness != null);
  const rankOf = (r: (typeof rows)[number]) =>
    anyFitness ? r.fitness : r.primary_metric;
  const best = rows.reduce<number | null>((acc, r) => {
    const v = rankOf(r);
    if (v == null) return acc;
    return acc == null || v > acc ? v : acc;
  }, null);
  return (
    <div className="rs-card" style={{ marginBottom: 22 }}>
      <div className="rs-card-head">
        <div className="rs-card-title">
          <Icon name="package" size={16} />Trained policies
        </div>
        <span className="rs-sub" style={{ fontSize: 12 }}>
          self-contained bundles for sim-to-real deployment
        </span>
      </div>
      {policies.isLoading ? (
        <p className="rs-sub" style={{ padding: "12px 16px" }}>Loading…</p>
      ) : rows.length === 0 ? (
        <EmptyState
          icon="package"
          title="No trained checkpoints yet"
          sub="Each completed training iteration leaves an exportable checkpoint here — bundle it with its reward, env spec, and an ONNX/TorchScript policy in one click."
        />
      ) : (
        <div className="rs-card-pad rs-vgap-8">
          {rows.map((p) => {
            const metric = rankOf(p);
            const isBest = best != null && metric != null && metric >= best;
            return (
              <div
                key={p.iter_index}
                className="rs-flex rs-gap-12 rs-wrap"
                style={{
                  border: "1px solid var(--hairline)",
                  borderRadius: "var(--radius-md)",
                  padding: "10px 12px",
                  background: "var(--canvas-soft)",
                  alignItems: "center",
                  fontSize: 12.5,
                }}
              >
                <span style={{ fontWeight: 500, minWidth: 60 }}>iter {p.iter_index}</span>
                {p.reward_version && (
                  <span className="rs-tag mono" style={{ fontSize: 10 }}>
                    reward {p.reward_version}
                  </span>
                )}
                {p.fitness != null ? (
                  <span className="rs-num" title="objective fitness (0-1)">
                    fit {p.fitness.toFixed(2)}
                  </span>
                ) : p.primary_metric != null ? (
                  <span className="rs-num" title="mean return">
                    {p.primary_metric.toFixed(1)}
                  </span>
                ) : null}
                {isBest && rows.length > 1 && (
                  <span className="rs-tag" style={{ fontSize: 10, color: "var(--st-emerald)" }}>best</span>
                )}
                <span className="rs-sub rs-num" style={{ fontSize: 11 }}>
                  {fmtBytes(p.checkpoint_bytes)}
                </span>
                <span style={{ marginLeft: "auto" }}>
                  <a
                    href={policyExportUrl(slug, p.iter_index)}
                    download
                    className="rs-btn rs-btn-ghost rs-btn-sm"
                    title="Download a zip with the checkpoint, ONNX/TorchScript policy, reward + env spec snapshots, and a DEPLOY.md recipe"
                  >
                    <Icon name="download" size={14} />Export
                  </a>
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default ReportsTab;

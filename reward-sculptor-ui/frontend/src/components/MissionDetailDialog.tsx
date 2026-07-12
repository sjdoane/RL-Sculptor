import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Badge, Banner, Btn, Modal } from "@/components/rs/primitives";
import {
  useDeleteMission,
  useMission,
  useRegenerateStageMetric,
  useStageEnvSpec,
  useStageIterations,
} from "@/hooks/useMissions";
import { useDetachStageReference } from "@/hooks/useReferences";
import { useJob } from "@/hooks/useJob";
import { RunMissionDialog } from "@/components/RunMissionDialog";
import { ReferencePickerDialog } from "@/components/ReferencePickerDialog";
import { useMissionEvents } from "@/hooks/useMissionEvents";
import { ApiError, stageRolloutUrl } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import { useQueryClient } from "@tanstack/react-query";
import { failureReasonText, stageLabel, supersededText } from "@/lib/stageDisplay";
import { formatRelative } from "@/lib/utils";
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
  const navigate = useNavigate();
  // §Ship-19d: clicking "Run mission" now opens RunMissionDialog (a
  // separate Dialog component) so the user can configure iterations
  // / Goal A / Goal B per launch. The actual mutation is owned by
  // RunMissionDialog. We still keep `del` here for the Delete button.
  const del = useDeleteMission(slug);

  // §Ship 20 (de-siloing): which stage's disk-truth iterations to show.
  // null = the collapsed stage list; selecting a stage expands its panel.
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  // Reset the selection whenever the dialog closes or the mission changes,
  // so re-opening starts from the stage list.
  useEffect(() => {
    if (!open) setSelectedStage(null);
  }, [open]);
  useEffect(() => {
    setSelectedStage(null);
  }, [missionSlug]);

  // Deep-link the Rewards tab scoped to a stage, then close this dialog.
  const viewStageRewards = (stageName: string) => {
    if (!missionSlug) return;
    onOpenChange(false);
    navigate(`/projects/${slug}?tab=rewards&stage=${encodeURIComponent(`${missionSlug}/${stageName}`)}`);
  };

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

  // §Ship 20 Goal #2: derive each stage's *effective* max iterations
  // from the WS event stream. When the user passed `iterations_
  // override` to RunMissionDialog, the stage runs against that cap,
  // not the authored `stage.max_iterations` — but the persisted
  // mission.json doesn't reflect the override (by design: the
  // override is per-launch, not curriculum-level). stage_started +
  // stage_completed_training carry `effective_max_iterations`; we
  // walk events to map stage_name → effective cap. StageCard reads
  // this map and shows a tooltip when the cap differs from the
  // authored value.
  const stageEffectiveMaxIters = useMemo(
    () => deriveStageEffectiveMaxIters(events.structuredEvents),
    [events.structuredEvents],
  );

  if (!open) return null;

  const showEvents =
    wsEnabled ||
    events.logLines.length > 0 ||
    events.structuredEvents.length > 0;

  return (
    <Modal
      wide
      icon="target"
      title={liveSummary?.goal ?? "Mission"}
      subtitle={
        <span style={{ display: "inline-flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span className="mono">
            {liveSummary
              ? `${liveSummary.mission_slug} · ${liveSummary.current_stage_idx}/${liveSummary.n_stages} stages · created ${formatRelative(liveSummary.created_at)}`
              : "Loading…"}
          </span>
          {liveSummary && <MissionLifecycleBadge lifecycle={liveSummary.lifecycle} />}
        </span>
      }
      onClose={() => onOpenChange(false)}
      footer={
        <>
          <div style={{ display: "flex", gap: 8 }}>
            {liveSummary && (
              <Btn
                kind="danger"
                icon={del.isPending ? "loader" : "trash"}
                disabled={del.isPending || activeJobId != null}
                title={activeJobId ? "Already training. Wait for the run to finish before deleting." : undefined}
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
                      toast.error("Could not delete mission", { description: detailMsg });
                    },
                  });
                }}
              >
                Delete
              </Btn>
            )}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {mission && mission.lifecycle !== "errored" && missionSlug && (
              <RunMissionDialog
                slug={slug}
                missionSlug={missionSlug}
                mission={mission}
                disabled={mission.lifecycle !== "ready" || activeJobId != null}
                disabledTitle={
                  mission.lifecycle !== "ready"
                    ? `Run is only allowed once planning finishes (currently ${mission.lifecycle}).`
                    : activeJobId
                      ? "Already training. Wait for the run to finish."
                      : undefined
                }
              />
            )}
            <Btn kind="primary" onClick={() => onOpenChange(false)}>Close</Btn>
          </div>
        </>
      }
    >
      {/* §Ship-19c: while a mission is being decomposed, mission.json
          doesn't exist yet so the GET returns 404. Don't surface
          that as a destructive error — render a "Decomposing"
          placeholder + the live WS event stream below so the user
          sees Claude's stdout in real time. */}
      {detail.isLoading && !mission && !isDecomposing && (
        <p className="rs-sub" style={{ fontSize: 13 }}>Loading mission…</p>
      )}
      {isDecomposing && !mission && (
        <Banner kind="warn" icon="loader">
          <span style={{ fontWeight: 600, display: "block" }}>Planning — Claude is breaking your goal into stages</span>
          <span style={{ fontSize: 12 }}>Typically 30-90 s. Stages will appear here when planning finishes. Live progress streams below.</span>
        </Banner>
      )}
      {/* §Ship 21a defense-in-depth: only the genuinely-failed-404 case. */}
      {detail.error && !isDecomposing && !liveSummary && (
        <Banner kind="err" icon="alert-triangle">
          <span className="mono" style={{ fontSize: 12 }}>{(detail.error as Error).message}</span>
        </Banner>
      )}

      {mission?.lifecycle === "errored" && (
        <Banner kind="err" icon="alert-triangle">
          <span style={{ fontWeight: 600, display: "block" }}>mission.json is unreadable</span>
          <span style={{ fontSize: 12 }}>The on-disk mission file failed to parse. The mission can still be deleted to recover the slug.</span>
        </Banner>
      )}

      {mission?.decomposition_rationale && (
        <section>
          <PHead>Decomposition rationale</PHead>
          <p style={{ whiteSpace: "pre-wrap", border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", background: "var(--surface-strong)", padding: 12, fontSize: 12.5, lineHeight: 1.55, margin: 0 }}>
            {mission.decomposition_rationale}
          </p>
        </section>
      )}

      {mission && mission.stages.length > 0 && missionSlug && (
        <section>
          <PHead>Stages ({mission.current_stage_idx}/{mission.n_stages})</PHead>
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {mission.stages.map((s, idx) => {
              const isCurrent = idx === mission.current_stage_idx;
              const selected = selectedStage === s.name;
              return (
                <div key={s.name}>
                  <StageCard
                    slug={slug}
                    missionSlug={missionSlug}
                    stage={s}
                    fallbackNumber={idx + 1}
                    depth={stageDepths.get(s.name) ?? 0}
                    isCurrent={isCurrent}
                    iters={stageIters.get(s.name) ?? []}
                    effectiveMaxIters={stageEffectiveMaxIters.get(s.name) ?? null}
                    selected={selected}
                    onToggle={() => setSelectedStage(selected ? null : s.name)}
                  />
                  {selected && (
                    <StagePanel
                      slug={slug}
                      missionSlug={missionSlug}
                      stage={s}
                      depth={stageDepths.get(s.name) ?? 0}
                      isActive={isCurrent && activeJobId != null}
                      onViewRewards={() => viewStageRewards(s.name)}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {showEvents && (
        <section style={{ display: "flex", flexDirection: "column", gap: 9 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <PHead noMargin>Live events</PHead>
            <WsStatusChip events={events} />
          </div>

          {events.disconnected && !events.terminal && (
            <Banner kind="warn" icon="alert-circle">Live stream interrupted. Refresh to reconnect.</Banner>
          )}
          {events.noActiveJob && (
            <Banner kind="info" icon="info">Nothing running. Launch a run to watch live events.</Banner>
          )}

          {events.structuredEvents.length > 0 && (
            <StructuredEventList events={events.structuredEvents} />
          )}
          {(events.logLines.length > 0 || wsEnabled) && (
            <LogScroller events={events.logLines} />
          )}
        </section>
      )}
    </Modal>
  );
}

// Small uppercase section heading (replaces the shadcn h3 muted caption).
function PHead({ children, noMargin }: { children: React.ReactNode; noMargin?: boolean }) {
  return (
    <h3 style={{ margin: noMargin ? 0 : "0 0 7px", fontSize: 10.5, fontWeight: 600, letterSpacing: 0.6, textTransform: "uppercase", color: "var(--rs-muted)" }}>
      {children}
    </h3>
  );
}

// ── lifecycle / status badges (rs Badge — STATUS_META covers both) ────

export function MissionLifecycleBadge({
  lifecycle,
}: {
  lifecycle: MissionLifecycleStatus;
}) {
  return <Badge status={lifecycle} />;
}

function StageStatusBadge({ status }: { status: StageStatus }) {
  return <Badge status={status} />;
}

// §mission-persistence increment 2: per-stage objective-metric
// visibility — the whole point of this increment is that a rejected
// stage metric (silently falling back to blind/mission-level) is no
// longer invisible in the UI.
function StageMetricChip({
  status,
  steeringMetric,
}: {
  status: StageSchema["metric_status"];
  steeringMetric: StageSchema["steering_metric"];
}) {
  if (status === "accepted") {
    return (
      <span className="rs-badge emerald" style={{ fontSize: 9.5 }} title={steeringMetric ?? undefined}>
        <Icon name="check" size={10} />metric ✓
      </span>
    );
  }
  if (status === "inherited") {
    return (
      <span className="rs-badge slate" style={{ fontSize: 9.5 }} title="Falls back to the mission-level fitness metric">
        metric: inherited
      </span>
    );
  }
  if (status === "rejected") {
    return (
      <span className="rs-badge amber" style={{ fontSize: 9.5 }} title="Generated metric failed the trust gate — this stage falls back to the mission-level metric (or runs blind)">
        <Icon name="alert-triangle" size={10} />metric rejected — blind fallback
      </span>
    );
  }
  return (
    <span className="rs-badge slate" style={{ fontSize: 9.5 }}>
      no metric
    </span>
  );
}

// §R1 (reference library): row after StageMetricChip showing the
// attached reference clip (text + tier), or a ghost "Pick reference"
// button that opens ReferencePickerDialog. Kept modest — this is the v1
// approval surface, not a full library page (spec decision 12).
function ReferenceRow({
  slug,
  missionSlug,
  stage,
}: {
  slug: string;
  missionSlug: string;
  stage: StageSchema;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const detach = useDetachStageReference(slug);

  const doDetach = () => {
    detach.mutate(
      { missionSlug, stageName: stage.name },
      {
        onSuccess: () => toast.success("Reference detached", { description: stage.name }),
        onError: (err) => {
          if (err instanceof ApiError && err.status === 409) {
            toast.error("Mission is busy", {
              description: "Wait for the running job to finish before detaching a reference.",
            });
            return;
          }
          const detail = err instanceof ApiError ? (err.problem.detail ?? err.problem.title) : err.message;
          toast.error("Could not detach reference", { description: String(detail) });
        },
      },
    );
  };

  return (
    <div style={{ marginTop: 6, display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
      {stage.start_pose && (
        <span
          className="rs-badge slate"
          style={{ fontSize: 9.5 }}
          title="Physical configuration the robot is in at this stage's episode start"
        >
          <Icon name="user" size={10} />
          start: {stage.start_pose}
        </span>
      )}
      {stage.reference_clip_id ? (
        <>
          <span
            className="rs-badge slate"
            style={{ fontSize: 9.5 }}
            title={
              stage.reference_match_confidence != null
                ? `auto-matched ${stage.reference_match_confidence.toFixed(2)}`
                : undefined
            }
          >
            <Icon name="video" size={10} />
            {stage.reference_clip_id}
            {stage.reference_tier && <span style={{ opacity: 0.7 }}>&nbsp;· tier {stage.reference_tier}</span>}
          </span>
          <Btn
            kind="ghost" size="xs" icon="pencil"
            onClick={(e) => { e.stopPropagation(); setPickerOpen(true); }}
          >
            Change
          </Btn>
          <Btn
            kind="ghost" size="xs" icon={detach.isPending ? "loader" : "x"}
            disabled={detach.isPending}
            onClick={(e) => { e.stopPropagation(); doDetach(); }}
          >
            {detach.isPending ? "Detaching…" : "Detach"}
          </Btn>
        </>
      ) : (
        <>
          <span className="rs-sub" style={{ fontSize: 10.5 }}>
            no reference matched —
          </span>
          <Btn
            kind="ghost" size="xs" icon="video"
            onClick={(e) => { e.stopPropagation(); setPickerOpen(true); }}
          >
            pick one
          </Btn>
        </>
      )}
      {pickerOpen && (
        <ReferencePickerDialog
          slug={slug}
          missionSlug={missionSlug}
          stageName={stage.name}
          currentClipId={stage.reference_clip_id}
          initialQuery={stage.goal_text}
          onClose={() => setPickerOpen(false)}
        />
      )}
    </div>
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

// §Ship 20 Goal #2: walk events for stage_started + stage_completed_
// training entries, return Map<stage_name, effective_max_iterations>.
// We update on the latest seen value per stage so a re-run within
// the same WS session shows the new cap. "Last value wins" is fine
// because (a) within one mission_run the cap is constant per stage,
// (b) re-runs of the same mission emit a new stage_started before
// any new iter events, so the new cap is in place before anything
// else displays.
//
// Known limitation: if the structuredEvents 5000-cap (see
// useMissionEvents.ts) evicts the original stage_started before the
// run completes, the UI falls back to `stage.max_iterations` and the
// override tooltip won't fire. Acceptable for v1 — typical missions
// emit far fewer than 5000 events.
function deriveStageEffectiveMaxIters(
  events: MissionEvent[],
): Map<string, number> {
  const out = new Map<string, number>();
  for (const ev of events) {
    if (
      ev.type === "stage_started" ||
      ev.type === "stage_completed_training"
    ) {
      const name = (ev as { stage_name?: string }).stage_name;
      const eff = (ev as { effective_max_iterations?: number })
        .effective_max_iterations;
      if (typeof name === "string" && typeof eff === "number") {
        out.set(name, eff);
      }
    }
  }
  return out;
}

function StageCard({
  slug,
  missionSlug,
  stage,
  fallbackNumber,
  depth,
  isCurrent,
  iters,
  effectiveMaxIters,
  selected,
  onToggle,
}: {
  /** §mission-persistence increment 2: needed for the per-stage
   *  "Regenerate metric" mutation. */
  slug: string;
  missionSlug: string;
  stage: StageSchema;
  /** 1-based array position, used only while `stage.display_label` is
   *  absent (older missions predating the field). */
  fallbackNumber: number;
  depth: number;
  isCurrent: boolean;
  iters: IterRow[];
  /** Cap derived from the WS event stream (per-mount; volatile). Used
   *  ONLY as a fallback when the persisted `stage.effective_max_
   *  iterations` is null (e.g., a stage that hasn't run yet but the
   *  user is mid-launch). The persisted field is the source of truth
   *  once a run has started. */
  effectiveMaxIters: number | null;
  /** §Ship 20 (de-siloing): whether this stage's disk panel is open. */
  selected: boolean;
  onToggle: () => void;
}) {
  const qc = useQueryClient();
  const regen = useRegenerateStageMetric(slug);
  const [regenJobId, setRegenJobId] = useState<string | null>(null);
  useJob(regenJobId ?? undefined, {
    enabled: regenJobId != null,
    onTerminal: (job) => {
      setRegenJobId(null);
      qc.invalidateQueries({ queryKey: qk.mission(slug, missionSlug) });
      if (job.status === "completed") {
        toast.success("Metric regeneration finished", { description: `stage ${stage.name}` });
      } else if (job.status === "errored") {
        toast.error("Metric regeneration failed", { description: `stage ${stage.name}` });
      }
    },
  });
  const regenerateMetric = () => {
    regen.mutate(
      { missionSlug, stageName: stage.name },
      {
        onSuccess: (job) => {
          setRegenJobId(job.job_id);
          toast.success("Metric regeneration started", { description: `stage ${stage.name}` });
        },
        onError: (err) => {
          const detail = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
          toast.error("Could not regenerate metric", { description: detail });
        },
      },
    );
  };
  const regenBusy = regen.isPending || regenJobId != null;
  const orphan = stage.parent_stage !== null && depth === 0;
  // §Ship 20a fallback chain:
  //   1. stage.effective_max_iterations (PERSISTED — source of truth
  //      once orchestrator wrote it; survives page refresh + WS
  //      event-cap eviction).
  //   2. effectiveMaxIters from the live event stream (transient;
  //      only relevant for the brief window before the first
  //      stage_completed/failed event triggers a save).
  //   3. stage.max_iterations (Claude's authored value; the
  //      pre-Ship-20 default and final fallback).
  const persistedEff = stage.effective_max_iterations;
  const liveEff = effectiveMaxIters;
  const effective = persistedEff ?? liveEff ?? null;
  const displayMax = effective ?? stage.max_iterations;
  const overrideActive =
    effective !== null && effective !== stage.max_iterations;
  const itersTooltip = overrideActive
    ? `Claude allocated ${stage.max_iterations} rounds; this run capped at ${effective}.`
    : undefined;
  return (
    <div
      style={{
        marginLeft: depth * 16,
        border: "1px solid " + (isCurrent || selected ? "var(--rs-primary)" : "var(--hairline)"),
        background: isCurrent ? "rgba(245,78,0,0.04)" : "var(--surface-strong)",
        borderRadius: selected ? "var(--radius-md) var(--radius-md) 0 0" : "var(--radius-md)",
        padding: "10px 12px",
        fontSize: 12,
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={selected}
        title={selected ? "Hide this stage's iterations" : "Show this stage's iterations, rollouts and RSI"}
        style={{
          display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, width: "100%",
          background: "none", border: 0, padding: 0, margin: 0, cursor: "pointer",
          textAlign: "left", font: "inherit", color: "inherit",
        }}
      >
        <Icon name={selected ? "chevron-down" : "chevron-right"} size={14} color="var(--rs-muted)" />
        <span
          className="mono rs-sub"
          style={{ fontSize: 10.5, fontWeight: 600 }}
          title={stage.on_disk_only ? "recovered from disk (not in mission.json)" : undefined}
        >
          {stageLabel(stage, fallbackNumber)}
        </span>
        <StageStatusBadge status={stage.status} />
        <span className="mono" style={{ fontSize: 11.5, fontWeight: 600 }}>{stage.name}</span>
        {stage.parent_stage && (
          <span className="rs-sub" style={{ fontSize: 10.5 }}>
            parent: <code className="mono">{stage.parent_stage}</code>
          </span>
        )}
        {orphan && <Badge status="held" label="Missing parent" />}
        {stage.redecomposition_attempts > 0 && (
          <span className="rs-badge slate" style={{ fontSize: 9.5 }}>replanned ×{stage.redecomposition_attempts}</span>
        )}
        {stage.on_disk_only && (
          <span className="rs-badge slate" style={{ fontSize: 9.5 }} title="Reconstructed from an on-disk stages/ directory (not in mission.json)">
            <Icon name="history" size={10} />recovered from disk
          </span>
        )}
        <StageMetricChip status={stage.metric_status} steeringMetric={stage.steering_metric} />
      </button>
      <ReferenceRow slug={slug} missionSlug={missionSlug} stage={stage} />
      {stage.status === "superseded" && (
        <p className="rs-sub" style={{ margin: "6px 0 0", fontSize: 11 }}>{supersededText(stage)}</p>
      )}
      {stage.status === "failed" && stage.failure_reason && (
        <p className="rs-sub" style={{ margin: "6px 0 0", fontSize: 11 }}>
          {failureReasonText(stage.failure_reason, stage.iterations_used)}
        </p>
      )}
      {/* §mission-persistence increment 2: only offer regeneration for
          stages that will actually run again (pending) and currently
          lack a usable metric (rejected/none). A stage mid-training or
          already succeeded/failed is exempt — the backend's own
          regenerate route also 409s on a training stage. */}
      {stage.status === "pending" &&
        (stage.metric_status === "rejected" || stage.metric_status === "none" || stage.metric_status == null) && (
          <div style={{ marginTop: 6 }}>
            <Btn
              kind="ghost" size="xs" icon={regenBusy ? "loader" : "sparkles"}
              disabled={regenBusy}
              onClick={(e) => { e.stopPropagation(); regenerateMetric(); }}
            >
              {regenBusy ? "Regenerating…" : "Regenerate metric"}
            </Btn>
          </div>
        )}
      <p style={{ margin: "6px 0 0", fontSize: 11.5, lineHeight: 1.5 }}>{stage.goal_text}</p>
      {stage.success_criterion && (
        <p className="mono" style={{ margin: "6px 0 0", wordBreak: "break-all", borderRadius: "var(--radius-sm)", background: "var(--canvas-soft)", border: "1px solid var(--hairline)", padding: "5px 8px", fontSize: 10.5, color: "var(--rs-muted)" }}>
          {stage.success_criterion}
        </p>
      )}
      <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", columnGap: 12, rowGap: 2, fontSize: 10.5, color: "var(--rs-muted)" }}>
        <span
          title={itersTooltip}
          aria-label={
            overrideActive
              ? `rounds ${stage.iterations_used} of ${displayMax}; ` +
                `per-launch override (Claude allocated ${stage.max_iterations})`
              : undefined
          }
          style={overrideActive ? { textDecoration: "underline dotted" } : undefined}
        >
          rounds {stage.iterations_used}/{displayMax}
          {overrideActive && (
            <span aria-hidden="true" style={{ marginLeft: 2, color: "var(--st-amber)" }}>*</span>
          )}
        </span>
        {stage.best_metric != null && <span>best metric {stage.best_metric.toFixed(3)}</span>}
        {stage.kg_seed_papers.length > 0 && <span>kg refs {stage.kg_seed_papers.length}</span>}
      </div>
      {iters.length > 0 && <IterRibbon iters={iters} />}
    </div>
  );
}

// ── Stage panel (de-siloed disk view) ────────────────────────────────
// Labels for the KEPT iteration by how selection chose it.
const SELECTION_LABEL: Record<string, string> = {
  "criterion+fitness": "best passing iter",
  criterion_newest: "newest passing iter",
  fitness_fallback: "best fitness (criterion unmet)",
  last: "last iter",
};

function selectionLabel(source: string | null | undefined): string {
  if (!source) return "kept";
  return `kept — ${SELECTION_LABEL[source] ?? source}`;
}

/** Expanded, disk-truth view of one stage: its iterations (independent
 *  of live scope), a rollout player for the picked iter, an RSI chip
 *  when reference-state-init was applied, and a deep-link to the stage's
 *  reward versions. This is what de-silos completed stages. */
function StagePanel({
  slug, missionSlug, stage, depth, isActive, onViewRewards,
}: {
  slug: string;
  missionSlug: string;
  stage: StageSchema;
  depth: number;
  isActive: boolean;
  onViewRewards: () => void;
}) {
  const iters = useStageIterations(slug, missionSlug, stage.name, {
    refetchIntervalMs: isActive ? 4000 : null,
  });
  const envSpec = useStageEnvSpec(slug, missionSlug, stage.name);
  const rows = iters.data ?? [];

  // Which iteration's rollout to show. Prefer the kept iter, else the
  // newest with a rollout, else the newest.
  const [picked, setPicked] = useState<number | null>(null);
  const defaultIter = useMemo(() => {
    if (rows.length === 0) return null;
    const kept =
      stage.selected_iter_index != null
        ? rows.find((r) => r.iter_index === stage.selected_iter_index)
        : undefined;
    if (kept?.has_rollout) return kept.iter_index;
    const newestWithRollout = [...rows].reverse().find((r) => r.has_rollout);
    if (newestWithRollout) return newestWithRollout.iter_index;
    return rows[rows.length - 1].iter_index;
  }, [rows, stage.selected_iter_index]);
  const activeIter = picked ?? defaultIter;
  const activeRow = rows.find((r) => r.iter_index === activeIter) ?? null;

  const source = envSpec.data?.current?.meta?.source ?? null;
  const rsiApplied = typeof source === "string" && source.startsWith("reference:");

  return (
    <div
      style={{
        marginLeft: depth * 16,
        border: "1px solid var(--rs-primary)",
        borderTop: "none",
        borderRadius: "0 0 var(--radius-md) var(--radius-md)",
        background: "var(--surface-card)",
        padding: "10px 12px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div className="rs-flex rs-wrap rs-gap-8" style={{ alignItems: "center" }}>
        {rsiApplied && (
          <span
            className="rs-badge emerald"
            style={{ fontSize: 9.5 }}
            title={`Reference-state-init curriculum applied (${source})`}
          >
            <Icon name="sparkles" size={11} />RSI
          </span>
        )}
        <Btn kind="ghost" size="xs" icon="file-code" onClick={onViewRewards}>
          View rewards for this stage
        </Btn>
      </div>

      {iters.isLoading && <p className="rs-sub" style={{ fontSize: 11, margin: 0 }}>Loading iterations…</p>}
      {iters.error && (
        <p style={{ fontSize: 11, margin: 0, color: "var(--st-rose)" }}>
          {(iters.error as Error).message}
        </p>
      )}
      {!iters.isLoading && !iters.error && rows.length === 0 && (
        <p className="rs-sub" style={{ fontSize: 11, margin: 0 }}>
          No iterations on disk yet for this stage.
        </p>
      )}

      {rows.length > 0 && (
        <>
          {/* Rollout player for the picked iter. */}
          {activeIter != null && activeRow?.has_rollout ? (
            <div style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
              <div className="rs-log-bar" style={{ gap: 8 }}>
                <Icon name="video" size={13} color="var(--rs-muted)" />
                <span style={{ fontSize: 11.5, fontWeight: 600 }}>iter {activeIter} rollout</span>
                <span className="rs-grow" />
                <a
                  href={stageRolloutUrl(slug, missionSlug, stage.name, activeIter)}
                  download={`${stage.name}_iter_${activeIter}.mp4`}
                  className="rs-btn rs-btn-quiet rs-btn-xs"
                >
                  <Icon name="download" size={13} />MP4
                </a>
              </div>
              <video
                key={activeIter}
                src={stageRolloutUrl(slug, missionSlug, stage.name, activeIter)}
                style={{ width: "100%", aspectRatio: "16/9", background: "#16150f", display: "block" }}
                controls
                playsInline
                preload="metadata"
              >
                <track kind="captions" />
              </video>
            </div>
          ) : activeIter != null ? (
            <p className="rs-sub" style={{ fontSize: 11, margin: 0 }}>
              iter {activeIter} has no rollout video.
            </p>
          ) : null}

          {/* Iteration chips — click to switch the rollout; the kept one
              carries a labelled badge. */}
          <div role="list" aria-label="Stage iterations" className="rs-flex rs-wrap rs-gap-6">
            {rows.map((r) => {
              const isKept =
                stage.selected_iter_index != null &&
                r.iter_index === stage.selected_iter_index;
              const isPicked = r.iter_index === activeIter;
              return (
                <button
                  key={r.iter_index}
                  role="listitem"
                  type="button"
                  onClick={() => setPicked(r.iter_index)}
                  title={
                    `iter ${r.iter_index}` +
                    (r.reward_version ? ` · reward ${r.reward_version}` : "") +
                    (r.has_rollout ? "" : " · no rollout") +
                    (r.has_checkpoint ? " · checkpoint" : "")
                  }
                  className="mono"
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    borderRadius: 5, padding: "3px 7px", fontSize: 10,
                    cursor: "pointer",
                    border: "1px solid " + (isPicked ? "var(--rs-primary)" : "var(--hairline)"),
                    background: isPicked ? "rgba(245,78,0,0.08)" : "var(--canvas-soft)",
                    color: "var(--ink)",
                  }}
                >
                  <span style={{ fontWeight: 600 }}>iter {r.iter_index}</span>
                  {r.fitness != null ? (
                    <span style={{ color: "var(--rs-muted)" }} title="objective fitness (0-1)">fit {r.fitness.toFixed(2)}</span>
                  ) : r.primary_metric != null ? (
                    <span style={{ color: "var(--rs-muted)" }} title="primary metric">{r.primary_metric.toFixed(1)}</span>
                  ) : (
                    <span style={{ color: "var(--rs-muted)" }}>—</span>
                  )}
                  {r.has_rollout && <Icon name="video" size={10} color="var(--rs-muted)" />}
                  {isKept && (
                    <span
                      className="rs-badge emerald"
                      style={{ fontSize: 8.5, padding: "0 4px" }}
                      title={selectionLabel(stage.selection_source)}
                    >
                      <Icon name="check" size={9} />{selectionLabel(stage.selection_source)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function IterRibbon({ iters }: { iters: IterRow[] }) {
  return (
    <div
      role="list"
      aria-label="Per-iteration progress"
      style={{ marginTop: 7, display: "flex", flexWrap: "wrap", alignItems: "center", gap: 4 }}
    >
      {iters.map((r) => (
        <IterChip key={r.iter_index} row={r} />
      ))}
    </div>
  );
}

// Iter pill — three states (training / rollout / completed). Colors via
// rs status tokens (no violet token, so rollout uses the accent tint).
function IterChip({ row }: { row: IterRow }) {
  const label = `iter ${row.iter_index}`;
  let bg: string, fg: string, metricStr: string, pulse = false;
  if (row.completed) {
    bg = "var(--st-emerald-bg)"; fg = "var(--st-emerald-fg)";
    metricStr = row.primary_metric != null ? row.primary_metric.toFixed(3) : "—";
  } else if (row.rollout_done) {
    bg = "rgba(245,78,0,0.10)"; fg = "var(--rs-primary)";
    metricStr = "rollout";
  } else {
    bg = "var(--st-amber-bg)"; fg = "var(--st-amber-fg)"; pulse = true;
    metricStr = "training…";
  }
  return (
    <span
      role="listitem"
      className={"mono" + (pulse ? " rs-pulse-soft" : "")}
      style={{ display: "inline-flex", alignItems: "center", gap: 4, borderRadius: 4, padding: "1px 5px", fontSize: 9.5, background: bg, color: fg }}
      title={`${label} · ${metricStr}`}
    >
      <span style={{ fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.3 }}>{label}</span>
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
  let label: string, cls: string, icon: string, spin = false;
  if (events.terminal) {
    label = "ended"; cls = "blue"; icon = "check-circle";
  } else if (events.disconnected) {
    label = "disconnected"; cls = "rose"; icon = "x";
  } else if (events.connected) {
    label = "live"; cls = "emerald"; icon = "activity";
  } else if (events.noActiveJob) {
    label = "no job"; cls = "slate"; icon = "square";
  } else {
    label = "connecting"; cls = "slate"; icon = "loader"; spin = true;
  }
  return (
    <span role="status" aria-label={`websocket: ${label}`} className={"rs-badge " + cls}>
      <Icon name={icon} size={11} className={spin ? "rs-spin" : undefined} />
      {label}
    </span>
  );
}

// Map an event type to one of the rs status colors (decorative).
function eventCat(type: string): "rose" | "amber" | "emerald" | "blue" | "slate" {
  if (/errored|failed|halted/.test(type)) return "rose";
  // §Ship 20: a rejected stage-metric is a soft fall-back to the mission
  // metric, not an error — amber like skipped/degraded.
  if (/skipped|degraded|rejected/.test(type)) return "amber";
  // §Ship 20 (de-siloing): archive/save/final-selection are positive
  // milestones — emerald alongside completed/accepted/rsi.
  if (/completed|succeeded|materialized|scaffolded|warm_start_resolved|rsi_applied|accepted|archived|saved|final_selection/.test(type)) return "emerald";
  if (/started|redecompos|criterion|training/.test(type)) return "blue";
  return "slate";
}

// Dark-pill bg|fg per category (matches the LogViewer event-tag palette).
const ECAT_DARK: Record<string, string> = {
  rose: "#3a1620|#f08aa6",
  amber: "#3a2c12|#f0b35a",
  emerald: "#16302a|#5fd0a0",
  blue: "#152836|#86b7e6",
  slate: "#2c2a20|#aaa698",
};

function StructuredEventList({ events }: { events: MissionEvent[] }) {
  // Render newest-first; cap visual rows so the dialog doesn't grow.
  const slice = useMemo(() => events.slice(-200).reverse(), [events]);
  return (
    <div
      role="log"
      aria-label="Mission events"
      aria-live="polite"
      className="rs-log"
      style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 240, overflowY: "auto", borderRadius: "var(--radius-md)", padding: "8px 10px" }}
    >
      {slice.map((ev, i) => (
        <StructuredEventRow key={i} event={ev} />
      ))}
    </div>
  );
}

function StructuredEventRow({ event }: { event: MissionEvent }) {
  const [bg, fg] = (ECAT_DARK[eventCat(event.type)] ?? ECAT_DARK.slate).split("|");
  const detail = describeEvent(event);
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6, fontSize: 11 }}>
      <span
        className="mono"
        style={{ flexShrink: 0, borderRadius: 4, padding: "0 4px", fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.4, background: bg, color: fg }}
      >
        {event.type}
      </span>
      {event.stage_name && <code className="mono" style={{ fontSize: 10.5, color: "#aaa698" }}>{event.stage_name}</code>}
      <span style={{ wordBreak: "break-all", color: "#d7d3c4" }}>{detail}</span>
    </div>
  );
}

function fmtMB(bytes: unknown): string {
  if (typeof bytes === "number" && Number.isFinite(bytes)) {
    return `${(bytes / 1_048_576).toFixed(1)} MB`;
  }
  return "?";
}

function describeEvent(ev: MissionEvent): string {
  const type = ev.type;
  // §Ship 20 (de-siloing): stage/mission persistence lifecycle.
  if (type === "stage_final_selection") {
    const iter = (ev as { iter?: number }).iter;
    const src = (ev as { source?: string }).source;
    const pass = (ev as { criterion_pass?: boolean }).criterion_pass;
    const metric = (ev as { metric?: number | null }).metric;
    return (
      `Kept iter ${iter ?? "?"}` +
      (src ? ` (${src})` : "") +
      ` as the stage policy` +
      (typeof pass === "boolean" ? ` — criterion ${pass ? "passed" : "unmet"}` : "") +
      (metric != null ? `, metric ${fmtMetric(metric)}` : "")
    );
  }
  if (type === "mission_stage_archived" || type === "mission_archived") {
    return `Saved to archive (${fmtMB((ev as { total_bytes?: number }).total_bytes)})`;
  }
  if (type === "mission_archive_failed") {
    return `Archive failed: ${(ev as { error?: string }).error ?? ""}`;
  }
  if (type === "mission_saved") {
    return "Mission saved";
  }
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
  // §Ship 20: DeepMimic reference-state-init curriculum for airborne
  // stages. stage_name rides the row chip; describe the clip + spec here.
  if (type === "stage_reference_rsi_applied") {
    const clip = (ev as { clip?: string }).clip;
    return `reference curriculum applied${clip ? ` (clip: ${clip})` : ""}`;
  }
  if (type === "stage_reference_rsi_failed") {
    const err = (ev as { error?: string }).error ?? "";
    return `reference curriculum failed${err ? `: ${err}` : ""}`;
  }
  // §Ship 20: per-stage trust-gated steering metrics. The bookends carry
  // no stage; the per-stage gen events carry `stage` (not stage_name) so
  // name it inline.
  if (type === "mission_stage_metrics_started") {
    const n = (ev as { n_stages?: number }).n_stages;
    return `generating a trust-gated metric per stage${
      typeof n === "number" ? ` (${n} stages)` : ""
    }…`;
  }
  if (type === "mission_stage_metrics_completed") {
    const gen = (ev as { generated?: unknown[] }).generated;
    const rej = (ev as { rejected?: unknown[] }).rejected;
    const nGen = Array.isArray(gen) ? gen.length : 0;
    const nRej = Array.isArray(rej) ? rej.length : 0;
    return `per-stage metrics: ${nGen} generated, ${nRej} fell back to mission metric`;
  }
  if (type === "stage_metric_gen_started") {
    const st = (ev as { stage?: string }).stage;
    return `stage ${st ?? "?"}: generating metric…`;
  }
  if (type === "stage_metric_gen_accepted") {
    const st = (ev as { stage?: string }).stage;
    return `stage ${st ?? "?"}: metric generated ✓`;
  }
  if (type === "stage_metric_gen_rejected") {
    const st = (ev as { stage?: string }).stage;
    const reason = (ev as { reason?: string }).reason ?? "";
    return `stage ${st ?? "?"}: metric rejected — falls back to mission metric${
      reason ? ` (${reason})` : ""
    }`;
  }
  if (type === "stage_metric_gen_failed") {
    const st = (ev as { stage?: string }).stage;
    const reason = (ev as { reason?: string }).reason ?? "";
    return `stage ${st ?? "?"}: metric generation failed${
      reason ? `: ${reason}` : ""
    }`;
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
    <div style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
      <div className="rs-log-bar">
        <Icon name="terminal" size={13} color="var(--rs-muted)" />
        <span style={{ fontSize: 11.5, fontWeight: 600 }}>log_line</span>
        <span className="rs-grow" />
        <span style={{ fontSize: 11, color: "var(--rs-muted)" }}>last {events.length}/200</span>
        {userScrolledUp.current && (
          <button
            type="button"
            className="rs-btn rs-btn-quiet rs-btn-xs"
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
        className="rs-log mono"
        style={{ maxHeight: 256, overflowY: "auto", padding: "8px 12px", fontSize: 11, lineHeight: 1.5 }}
      >
        {events.length === 0 ? (
          <p style={{ color: "#807d72", margin: 0 }}>waiting for stdout…</p>
        ) : (
          events.map((ev, i) => {
            const raw = (ev as { text?: unknown }).text;
            const text = typeof raw === "string" ? raw : JSON.stringify(ev);
            return (
              <div key={i} style={{ whiteSpace: "pre-wrap", wordBreak: "break-all", color: "#d7d3c4" }}>
                {text}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

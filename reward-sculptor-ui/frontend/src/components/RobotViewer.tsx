import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { useLiveClips } from "@/hooks/useLiveClips";
import { useProjectPreview } from "@/hooks/useProjectPreview";
import { useMission, useMissions, useStageIterations } from "@/hooks/useMissions";
import { useRunEvents } from "@/hooks/useRunEvents";
import { useRuns } from "@/hooks/useRuns";
import {
  clipUrl, getStageIterations, iterRolloutUrl, previewUrl, projectIterRolloutUrl,
  stageExportUrl, stageRolloutUrl,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import { failureReasonText, stageLabel, supersededText } from "@/lib/stageDisplay";
import type {
  CameraAngle, MissionSummary, RunSummary, SelectedStage, StageIteration, StageSchema,
} from "@/lib/types";
import { CAMERA_ANGLES } from "@/lib/types";

const ANGLE_LABEL: Record<CameraAngle, string> = {
  iso: "Iso",
  front: "Front",
  side: "Side",
  top: "Top",
};

const STAGE_STYLE: React.CSSProperties = {
  position: "relative",
  aspectRatio: "16 / 9",
  background: "#16150f",
  overflow: "hidden",
  display: "block",
};

type Mode = "static" | "live" | "replay";

/** Three-mode robot viewer. Sticky-mode rules (Prompt 9 R2):
 *    - Static → Live auto-transition when a run starts AND the user
 *      hasn't explicitly picked a mode.
 *    - Static → Live on initial page load if a run is already active.
 *    - Replay persists until the user explicitly leaves.
 *    - Live does NOT auto-switch to Static when a run completes — it
 *      stays on the last clip with a "Run completed" overlay.
 *
 * §Ship de-silo: when `selectedStage` names a mission stage, the Live
 * layer sources that stage's rollout from disk-truth (useStageIterations
 * + stageRolloutUrl, keyed by mission/stage — not run_id) so a stage
 * stays reachable after a LATER stage supersedes or errors it. A stage
 * picker (chips) lets the user jump between a mission's stages; picking
 * one calls `setSelectedStage`, which is URL-synced (`?stage=`) by
 * ProjectDetail and shared with RunsTab/RewardsTab.
 */
export function RobotViewer({
  slug, selectedStage, setSelectedStage,
}: {
  slug: string;
  selectedStage?: SelectedStage | null;
  setSelectedStage?: (value: SelectedStage | null) => void;
}) {
  // §Ship 21d: keep /runs polling through mission stage boundaries so
  // the live-video run selection doesn't freeze on a stale stage when
  // one completes and the next starts.
  const missions = useMissions(slug);
  const missionActive = useMemo(
    () => (missions.data ?? []).some((m) => m.active_job_id != null),
    [missions.data],
  );
  const runs = useRuns(slug, { keepPolling: missionActive });
  const activeRun = useMemo(() => pickActiveRun(runs.data ?? []), [runs.data]);
  const mostRecentRun = useMemo(() => pickMostRecent(runs.data ?? []), [runs.data]);
  const liveRun = activeRun ?? mostRecentRun;
  const liveRunId = liveRun?.run_id ?? null;
  const replayRun = activeRun ?? mostRecentRun;

  // The mission that owns the currently live/most-recent run (if any),
  // used to enumerate stages for the picker and to decide the terminal
  // (no-active-job) default stage below.
  const liveMissionSlug = liveRun?.kind === "mission_stage_run" ? liveRun.mission_slug ?? null : null;
  const fallbackMissionSlug = useMemo(() => pickDefaultMissionSlug(missions.data ?? []), [missions.data]);
  const missionSlugForPicker = selectedStage?.missionSlug ?? liveMissionSlug ?? fallbackMissionSlug;
  const missionDetail = useMission(slug, missionSlugForPicker ?? undefined, {
    enabled: !!missionSlugForPicker,
  });

  const [mode, setMode] = useState<Mode>("static");
  const [userPicked, setUserPicked] = useState(false);

  useEffect(() => {
    if (userPicked) return;
    if (activeRun && mode === "static") {
      setMode("live");
    }
  }, [activeRun, mode, userPicked]);

  const pickMode = useCallback((m: Mode) => {
    setMode(m);
    setUserPicked(true);
  }, []);

  const [replayIter, setReplayIter] = useState<number | null>(null);

  useEffect(() => {
    setReplayIter(null);
  }, [replayRun?.run_id]);

  // Foolproof default (req. 3): while a job is actively training and the
  // user hasn't made an explicit stage pick, keep following the live run
  // unchanged — don't fight the live auto-follow with a stage default.
  // Once the mission is terminal (no active job), default to the most
  // recent stage that actually has a rollout on disk, so an errored
  // final stage never blanks the viewer.
  useEffect(() => {
    if (selectedStage || !setSelectedStage) return;
    if (missionActive) return;
    if (!missionDetail.data || missionDetail.data.active_job_id != null) return;
    const fallback = pickDefaultStage(missionDetail.data.stages);
    if (fallback) setSelectedStage({ missionSlug: missionDetail.data.mission_slug, stageName: fallback });
  }, [selectedStage, setSelectedStage, missionActive, missionDetail.data]);

  const showStagePicker = !!missionSlugForPicker && (missionDetail.data?.stages.length ?? 0) > 1;
  // §UX honesty pass (Replay stage nav): whether ANY iteration has a
  // rollout on disk, per stage — used to disable (not hide) picker chips
  // for stages that haven't produced a rollout yet, instead of omitting
  // them (a stage with zero runs is still real curriculum the user should
  // be able to see and pick, just not play). Shares the useStageIterations
  // query cache/key, so this doesn't duplicate the fetch for whichever
  // stage is already the active selection.
  const rolloutAvailability = useStageRolloutAvailability(
    slug, missionSlugForPicker, missionDetail.data?.stages ?? [],
  );

  // §Ship de-silo: an explicit stage selection always wins EXCEPT while
  // that exact stage is the one actively training right now — in which
  // case we keep the live run_id-scoped path so "watch it train" doesn't
  // regress. This mirrors RunsTab's RunDetailPane `isLiveStage` check.
  // Hoisted out of LiveLayer (was computed inside it) so the choice
  // between <StageLayer/> and <LiveLayer/> happens as sibling top-level
  // renders in RobotViewer, and LiveLayer never conditionally skips its
  // own hooks (rules-of-hooks fix).
  const selectionIsLiveStage =
    missionActive &&
    liveRun?.kind === "mission_stage_run" &&
    liveRun.mission_slug === selectedStage?.missionSlug &&
    liveRun.stage_name === selectedStage?.stageName;
  const liveModeSelectsStage = !!selectedStage && !selectionIsLiveStage;
  // §UX honesty pass (Replay stage nav): Replay becomes stage-navigable
  // under the same "does this project have mission stages" condition Live
  // uses (`showStagePicker`) — plain (non-mission) projects keep the
  // original ReplayLayer, scoped to the most recent/active run.
  const replayModeSelectsStage =
    mode === "replay" && showStagePicker && !!selectedStage &&
    selectedStage.missionSlug === missionSlugForPicker;

  return (
    <div className="rs-viewer">
      <div className="rs-viewer-bar">
        <div className="rs-card-title">
          <Icon name="video" size={16} />
          Robot viewer
          {activeRun && <span className="rs-dot live" style={{ marginLeft: 4 }} />}
        </div>
        <ModeSwitcher mode={mode} onPick={pickMode} hasReplay={!!replayRun} hasLive={!!liveRunId} />
      </div>
      {showStagePicker && setSelectedStage && (mode === "live" || mode === "replay") && (
        <>
          <div className="rs-stage-divider" />
          <StagePicker
            missionSlug={missionSlugForPicker!}
            stages={missionDetail.data?.stages ?? []}
            selectedStageName={selectedStage?.missionSlug === missionSlugForPicker ? selectedStage.stageName : null}
            liveStageName={missionActive ? liveRun?.stage_name ?? null : null}
            rolloutAvailability={rolloutAvailability}
            onPick={(stageName) => setSelectedStage({ missionSlug: missionSlugForPicker!, stageName })}
          />
        </>
      )}
      <div className="rs-viewer-stage" style={STAGE_STYLE}>
        {mode === "static" && <StaticLayer slug={slug} />}
        {mode === "live" && liveModeSelectsStage && (
          <StageLayer
            slug={slug}
            missionSlug={selectedStage!.missionSlug}
            stageName={selectedStage!.stageName}
          />
        )}
        {mode === "live" && !liveModeSelectsStage && liveRun?.kind === "mission_stage_run" && (
          <LiveStageRollout slug={slug} runId={liveRunId} run={liveRun} />
        )}
        {mode === "live" && !liveModeSelectsStage && liveRun?.kind !== "mission_stage_run" && (
          <LiveLayer
            slug={slug}
            runId={liveRunId}
            run={liveRun}
          />
        )}
        {mode === "replay" && replayModeSelectsStage && (
          <StageReplayLayer
            slug={slug}
            missionSlug={selectedStage!.missionSlug}
            stageName={selectedStage!.stageName}
          />
        )}
        {mode === "replay" && !replayModeSelectsStage && (
          <ReplayLayer slug={slug} run={replayRun} iter={replayIter} onPickIter={setReplayIter} />
        )}
      </div>
    </div>
  );
}

/** §UX honesty pass (Replay stage nav): bulk has-any-rollout per stage,
 *  for the picker's disable/tooltip logic. Query key matches
 *  `useStageIterations`'s (`qk.stageIters`), so React Query dedupes this
 *  against whichever stage is already fetched for the active selection —
 *  no extra network traffic for that one. */
function useStageRolloutAvailability(
  slug: string,
  missionSlug: string | null,
  stages: StageSchema[],
): Map<string, boolean> {
  const results = useQueries({
    queries: stages.map((s) => ({
      queryKey: missionSlug ? qk.stageIters(slug, missionSlug, s.name) : ["stageIters", "_none", s.name],
      queryFn: () => getStageIterations(slug, missionSlug as string, s.name),
      enabled: !!missionSlug,
      staleTime: 30_000,
    })),
  });
  return useMemo(() => {
    const out = new Map<string, boolean>();
    stages.forEach((s, i) => {
      const rows = results[i]?.data as StageIteration[] | undefined;
      out.set(s.name, !!rows && rows.some((r) => r.has_rollout));
    });
    return out;
  }, [stages, results]);
}

/** Terminal-default helper: the newest mission with any job history,
 *  used only as a last resort when there's no live/most-recent run to
 *  anchor the picker to (e.g. static preview never having run yet). */
function pickDefaultMissionSlug(missions: MissionSummary[]): string | null {
  if (missions.length === 0) return null;
  const sorted = [...missions].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
  );
  return sorted[0].mission_slug;
}

/** req. 3: most recent stage that actually has a rollout on disk — never
 *  an errored/pending stage with nothing to show. "Most recent" = highest
 *  index among stages with `status !== "pending"`, since StageSchema
 *  doesn't carry has_rollout itself (that's per-iteration); we fall back
 *  through stages newest-first and let StageLayer's own disk-truth check
 *  render the "no rollout" note if this stage turns out empty too. */
function pickDefaultStage(stages: StageSchema[]): string | null {
  const attempted = stages.filter((s) => s.status !== "pending");
  if (attempted.length === 0) return null;
  const succeeded = [...attempted].reverse().find((s) => s.status === "succeeded");
  if (succeeded) return succeeded.name;
  return attempted[attempted.length - 1].name;
}

function pickMostRecent(runs: RunSummary[]): RunSummary | null {
  if (runs.length === 0) return null;
  const sorted = [...runs].sort((a, b) => {
    return runSortTime(b) - runSortTime(a);
  });
  return sorted[0];
}

function pickActiveRun(runs: RunSummary[]): RunSummary | null {
  return pickMostRecent(runs.filter((r) => r.status === "running" || r.status === "queued"));
}

function runSortTime(run: RunSummary): number {
  return run.started_at ? new Date(run.started_at).getTime() : 0;
}

// ── Mode switcher (rs-seg) ───────────────────────────────────────────
function ModeSwitcher({
  mode, onPick, hasLive, hasReplay,
}: { mode: Mode; onPick: (m: Mode) => void; hasLive: boolean; hasReplay: boolean }) {
  const items: Array<{ key: Mode; label: string; icon: string; disabled?: boolean; hint?: string }> = [
    { key: "static", label: "Static", icon: "camera" },
    { key: "live", label: "Live", icon: "activity", disabled: !hasLive, hint: hasLive ? undefined : "no runs yet" },
    { key: "replay", label: "Replay", icon: "history", disabled: !hasReplay, hint: hasReplay ? undefined : "no runs yet" },
  ];
  return (
    <div className="rs-seg" role="tablist">
      {items.map((it) => (
        <button
          key={it.key}
          role="tab"
          aria-selected={mode === it.key}
          disabled={it.disabled}
          title={it.hint}
          className={mode === it.key ? "on" : ""}
          style={it.disabled ? { opacity: 0.4, cursor: "not-allowed" } : undefined}
          onClick={() => onPick(it.key)}
        >
          <Icon name={it.icon} size={14} />
          {it.label}
        </button>
      ))}
    </div>
  );
}

// ── Static layer ─────────────────────────────────────────────────────
function StaticLayer({ slug }: { slug: string }) {
  const [angle, setAngle] = useState<CameraAngle>("iso");
  const [reRendering, setReRendering] = useState(false);
  const { data: url, isLoading, error, invalidate, refetch } = useProjectPreview(slug, { angle });

  const onReRender = async () => {
    setReRendering(true);
    try {
      const res = await fetch(previewUrl(slug, { angle, regenerate: true }), { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await res.blob();
      invalidate();
      await refetch();
      toast.success("Preview re-rendered");
    } catch (e) {
      toast.error("Re-render failed", { description: (e as Error).message });
    } finally {
      setReRendering(false);
    }
  };

  return (
    <>
      <div className="rs-overlay"><Icon name="camera" size={13} />static · {ANGLE_LABEL[angle]}</div>
      <div className="rs-overlay" style={{ left: "auto", right: 12, gap: 8, padding: "5px 8px" }}>
        <div className="rs-select">
          <select
            value={angle}
            onChange={(e) => setAngle(e.target.value as CameraAngle)}
            aria-label="Camera angle"
            style={{ height: 26, background: "rgba(255,255,255,0.06)", color: "#f3f1e8", border: "1px solid rgba(255,255,255,0.18)" }}
          >
            {CAMERA_ANGLES.map((a) => (
              <option key={a} value={a} style={{ color: "#16150f" }}>{ANGLE_LABEL[a]}</option>
            ))}
          </select>
        </div>
        <button
          className="rs-btn rs-btn-xs"
          style={{ background: "rgba(255,255,255,0.06)", color: "#f3f1e8", borderColor: "rgba(255,255,255,0.18)" }}
          aria-label="Re-render static preview from the selected camera angle"
          onClick={onReRender}
          disabled={reRendering || isLoading}
        >
          <Icon name="refresh-cw" size={12} className={reRendering ? "rs-spin" : undefined} />
          Re-render
        </button>
      </div>
      {(isLoading || reRendering) && <Shimmer />}
      {url && (
        <img src={url} alt={`${ANGLE_LABEL[angle]} static preview`} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }} />
      )}
      {error && !isLoading && <EmptyOverlay error={(error as Error).message} />}
    </>
  );
}

// ── Live layer ───────────────────────────────────────────────────────
// §Ship de-silo: the caller (RobotViewer) decides whether an explicit
// stage selection should render <StageLayer/> instead of this component
// (see `liveModeSelectsStage`), and also whether a mission_stage_run
// should render <LiveStageRollout/> instead — LiveLayer itself is only
// ever mounted for the plain run_id-scoped live path, so it never needs
// to conditionally bail before its own hooks run (rules-of-hooks fix for
// #85ddd70).
function LiveLayer({
  slug, runId, run,
}: {
  slug: string;
  runId: string | null;
  run: RunSummary | null;
}) {
  const { clips, skipped, terminal } = useLiveClips(slug, runId ?? undefined);
  const latest = clips[clips.length - 1] ?? null;
  const [hasEverPlayed, setHasEverPlayed] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (latest && videoRef.current) {
      setHasEverPlayed(true);
      videoRef.current.load();
      videoRef.current.play().catch(() => {});
    }
  }, [latest?.url]);

  const runCompleted = run && (run.status === "completed" || run.status === "errored" || run.status === "stopped");
  const errorNoClips = runCompleted && clips.length === 0 && run?.status === "errored";

  return (
    <>
      <div className="rs-overlay">
        <span className={"rs-dot" + (run?.status === "running" ? " live" : "")} style={run?.status === "running" ? undefined : { background: "var(--st-slate)" }} />
        live{latest !== null ? ` · iter ${latest.iter}` : ""}
        {latest && <span style={{ opacity: 0.8 }}>· {fmtMetricAt(run, latest.iter)}</span>}
      </div>
      {errorNoClips && <EmptyOverlay title="Run errored before any clips rendered" error={run?.error ?? undefined} />}
      {!errorNoClips && latest && (
        <video
          ref={videoRef}
          key={latest.url}
          src={latest.url}
          aria-label={`Live robot rollout, iteration ${latest.iter}`}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
          muted playsInline autoPlay loop
        />
      )}
      {!errorNoClips && !latest && !hasEverPlayed && <Shimmer label="waiting for first clip…" />}
      {skipped.length > 0 && (
        <div className="rs-overlay" style={{ top: "auto", bottom: 12, background: "rgba(138,86,0,0.85)" }}>
          <Icon name="alert-circle" size={12} />
          {skipped.length} clip{skipped.length === 1 ? "" : "s"} skipped
        </div>
      )}
      {runCompleted && latest && !errorNoClips && (
        <div className="rs-overlay" style={{ top: "auto", bottom: 12, left: "auto", right: 12, background: "rgba(15,102,71,0.85)" }}>
          Run {run?.status}
        </div>
      )}
      {terminal && <span className="sr-only">Terminal event received.</span>}
    </>
  );
}

function fmtMetricAt(run: RunSummary | null, iter: number): string {
  if (!run) return "";
  const v = run.primary_metric_history[iter];
  if (typeof v !== "number") return "";
  return `metric=${v.toFixed(3)}`;
}

// ── Live (mission_stage_run) — Ship 21b ──────────────────────────────
function LiveStageRollout({ slug, runId, run }: { slug: string; runId: string | null; run: RunSummary | null }) {
  const events = useRunEvents(slug, runId ?? undefined);
  const videoRef = useRef<HTMLVideoElement>(null);

  const latestIter = useMemo(() => {
    let best: number | null = null;
    let activeIter: number | null = null;
    for (const ev of events.events) {
      const i = (ev as { iter?: unknown }).iter;
      if (ev.type === "iter_started" && typeof i === "number") {
        activeIter = i;
      }
      if (ev.type === "iter_rolled_out" || ev.type === "rollout_done" || ev.type === "iter_completed") {
        // The mjlab adapter's `rollout_done` event describes the capture
        // (steps/frames/timing) but historically omits `iter`. Correlate it
        // with the latest `iter_started` event so Live can show the finished
        // clip immediately instead of waiting for the whole iteration (and
        // its potentially long LLM diagnosis) to commit.
        const rolloutIter = typeof i === "number" ? i : activeIter;
        if (rolloutIter !== null && (best === null || rolloutIter > best)) {
          best = rolloutIter;
        }
      }
    }
    if (best === null && run && run.iterations_completed > 0) best = run.iterations_completed - 1;
    return best;
  }, [events.events, run]);

  useEffect(() => {
    if (latestIter === null || !videoRef.current) return;
    videoRef.current.load();
    videoRef.current.play().catch(() => {});
  }, [latestIter]);

  const src = useMemo(() => {
    if (!run || latestIter === null) return null;
    return iterRolloutUrl(slug, run.run_id, latestIter);
  }, [slug, run, latestIter]);

  const runCompleted = run && (run.status === "completed" || run.status === "errored" || run.status === "stopped");
  const errorNoFrames = runCompleted && latestIter === null && run?.status === "errored";

  return (
    <>
      <div className="rs-overlay">
        <span className={"rs-dot" + (run?.status === "running" ? " live" : "")} style={run?.status === "running" ? undefined : { background: "var(--st-slate)" }} />
        live{run?.stage_name ? ` · ${run.stage_name}` : ""}{latestIter !== null ? ` · iter ${latestIter}` : ""}
        {latestIter !== null && <span style={{ opacity: 0.8 }}>· {fmtMetricAt(run, latestIter)}</span>}
      </div>
      {errorNoFrames && <EmptyOverlay title="Stage errored before any rollout rendered" error={run?.error ?? undefined} />}
      {!errorNoFrames && src && (
        <video
          ref={videoRef}
          key={src}
          src={src}
          aria-label={run?.stage_name ? `Live stage rollout, ${run.stage_name}, iteration ${latestIter ?? 0}` : `Live stage rollout, iteration ${latestIter ?? 0}`}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
          muted playsInline autoPlay loop
          onError={() => {}}
        />
      )}
      {!errorNoFrames && !src && <Shimmer label="waiting for first rollout…" />}
      {runCompleted && src && !errorNoFrames && (
        <div className="rs-overlay" style={{ top: "auto", bottom: 12, left: "auto", right: 12, background: "rgba(15,102,71,0.85)" }}>
          Stage {run?.status}
        </div>
      )}
      {events.terminal && <span className="sr-only">Terminal event received.</span>}
    </>
  );
}

// ── Stage layer (disk-truth, de-siloed) — §Ship de-silo ──────────────
// Sources a selected stage's rollout from useStageIterations +
// stageRolloutUrl — keyed by (missionSlug, stageName), not run_id — so a
// completed/errored/superseded stage keeps showing its video no matter
// what the live job is doing now. Mirrors RunsTab's StageDetailPane and
// MissionDetailDialog's StagePanel default-iter selection.
function StageLayer({
  slug, missionSlug, stageName,
}: { slug: string; missionSlug: string; stageName: string }) {
  const iters = useStageIterations(slug, missionSlug, stageName);
  const rows = iters.data ?? [];
  const videoRef = useRef<HTMLVideoElement>(null);

  // Prefer the newest iteration that actually has a rollout — the
  // "kept" iter (selected_iter_index) isn't available here without a
  // second mission fetch, and newest-with-rollout is what StagePanel
  // falls back to anyway when nothing is explicitly kept yet.
  const defaultIter = useMemo(() => {
    if (rows.length === 0) return null;
    const newestWithRollout = [...rows].reverse().find((r) => r.has_rollout);
    if (newestWithRollout) return newestWithRollout.iter_index;
    return rows[rows.length - 1].iter_index;
  }, [rows]);

  const activeIter = defaultIter;
  const activeRow = rows.find((r) => r.iter_index === activeIter) ?? null;
  const src = activeIter != null && activeRow?.has_rollout
    ? stageRolloutUrl(slug, missionSlug, stageName, activeIter)
    : null;

  useEffect(() => {
    if (!src || !videoRef.current) return;
    videoRef.current.load();
    videoRef.current.play().catch(() => {});
  }, [src]);

  const noRolloutAtAll = !iters.isLoading && !iters.error && rows.length > 0 && rows.every((r) => !r.has_rollout);
  const noIterationsAtAll = !iters.isLoading && !iters.error && rows.length === 0;

  return (
    <>
      <div className="rs-overlay">
        <Icon name="video" size={13} />
        {stageName}{activeIter !== null ? ` · iter ${activeIter}` : ""}
        <span style={{ opacity: 0.7 }}>· disk-truth</span>
      </div>
      {iters.isLoading && <Shimmer label="loading stage rollout…" />}
      {iters.error && (
        <EmptyOverlay title="Couldn't load this stage" error={(iters.error as Error).message} />
      )}
      {(noIterationsAtAll || noRolloutAtAll) && (
        <EmptyOverlay title="Stage errored, no rollout" error="No rollout was rendered for this stage before it stopped." />
      )}
      {src && (
        <video
          ref={videoRef}
          key={src}
          src={src}
          aria-label={`${stageName} rollout, iteration ${activeIter ?? 0}`}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
          muted playsInline autoPlay loop
        />
      )}
      {/* §UX honesty pass: export reachable wherever an iteration with a
          checkpoint is shown, independent of whether it has a rollout. */}
      {activeRow?.has_checkpoint && activeIter != null && (
        <a
          href={stageExportUrl(slug, missionSlug, stageName, activeIter)}
          download
          className="rs-overlay"
          style={{ left: "auto", right: 12, top: "auto", bottom: 12, textDecoration: "none", cursor: "pointer" }}
          title="Download the deployment bundle: checkpoint + ONNX + TorchScript + reward/env spec + DEPLOY.md"
        >
          <Icon name="package" size={13} />Export bundle
        </a>
      )}
    </>
  );
}

// ── Stage-scoped replay layer — §UX honesty pass (Replay stage nav) ────
// Reuses StageLayer's disk-truth data source (useStageIterations +
// stageRolloutUrl) but gives it Replay's UX: pick ANY iteration with a
// rollout on disk via a scrubber row, not just the newest.
function StageReplayLayer({
  slug, missionSlug, stageName,
}: { slug: string; missionSlug: string; stageName: string }) {
  const iters = useStageIterations(slug, missionSlug, stageName);
  const rows = iters.data ?? [];
  const videoRef = useRef<HTMLVideoElement>(null);
  const [picked, setPicked] = useState<number | null>(null);

  useEffect(() => {
    setPicked(null);
  }, [missionSlug, stageName]);

  const rowsWithRollout = useMemo(() => rows.filter((r) => r.has_rollout), [rows]);
  const defaultIter = rowsWithRollout.length > 0
    ? rowsWithRollout[rowsWithRollout.length - 1].iter_index
    : null;
  const activeIter = picked ?? defaultIter;
  const activeRow = rows.find((r) => r.iter_index === activeIter) ?? null;
  const src = activeIter != null && activeRow?.has_rollout
    ? stageRolloutUrl(slug, missionSlug, stageName, activeIter)
    : null;

  useEffect(() => {
    if (!src || !videoRef.current) return;
    videoRef.current.load();
  }, [src]);

  const noRolloutAtAll = !iters.isLoading && !iters.error && rowsWithRollout.length === 0;

  return (
    <>
      <div className="rs-overlay">
        <Icon name="history" size={13} />
        replay · {stageName}{activeIter !== null ? ` · iter ${activeIter}` : ""}
      </div>
      {(src || activeRow?.has_checkpoint) && (
        <div className="rs-overlay" style={{ left: "auto", right: 12, gap: 10 }}>
          {src && (
            <a
              href={src}
              download={`${stageName}_iter_${activeIter}.mp4`}
              style={{ display: "inline-flex", alignItems: "center", gap: 4, textDecoration: "none", cursor: "pointer", color: "inherit" }}
              onClick={(e) => e.stopPropagation()}
            >
              <Icon name="download" size={13} />Download
            </a>
          )}
          {activeRow?.has_checkpoint && activeIter != null && (
            <a
              href={stageExportUrl(slug, missionSlug, stageName, activeIter)}
              download
              style={{ display: "inline-flex", alignItems: "center", gap: 4, textDecoration: "none", cursor: "pointer", color: "inherit" }}
              title="Download the deployment bundle: checkpoint + ONNX + TorchScript + reward/env spec + DEPLOY.md"
            >
              <Icon name="package" size={13} />Export
            </a>
          )}
        </div>
      )}
      {iters.isLoading && <Shimmer label="loading stage rollouts…" />}
      {iters.error && (
        <EmptyOverlay title="Couldn't load this stage" error={(iters.error as Error).message} />
      )}
      {noRolloutAtAll && (
        <EmptyOverlay title="No rollouts recorded" error="This stage hasn't produced a rollout video yet." />
      )}
      {src && activeIter !== null && (
        <video
          ref={videoRef}
          key={src}
          src={src}
          aria-label={`${stageName} rollout, iteration ${activeIter}`}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
          controls
          playsInline
        />
      )}
      {rowsWithRollout.length > 0 && (
        <div className="rs-scrub" style={{ position: "absolute", insetInline: 0, bottom: 0, background: "rgba(20,19,13,0.82)", padding: "8px 10px" }}>
          {rowsWithRollout.map((r) => (
            <button
              key={r.iter_index}
              type="button"
              onClick={() => setPicked(r.iter_index)}
              className="rs-btn rs-btn-xs"
              style={
                r.iter_index === activeIter
                  ? { background: "var(--rs-primary)", color: "#fff", borderColor: "var(--rs-primary)" }
                  : { background: "rgba(255,255,255,0.06)", color: "#cfcdc4", borderColor: "rgba(255,255,255,0.16)" }
              }
            >
              iter {r.iter_index}
            </button>
          ))}
        </div>
      )}
    </>
  );
}

// ── Stage picker chips — §Ship de-silo ───────────────────────────────
// Lists a mission's stages so the user can jump straight to any of them
// (including an errored one, which still shows in the picker but renders
// the "no rollout" note rather than breaking).
function StagePicker({
  missionSlug, stages, selectedStageName, liveStageName, rolloutAvailability, onPick,
}: {
  missionSlug: string;
  stages: StageSchema[];
  selectedStageName: string | null;
  liveStageName: string | null;
  /** §UX honesty pass: has-any-rollout per stage name, from
   *  `useStageRolloutAvailability`. Missing entries (not yet fetched) are
   *  treated as no-rollout — the chip disables until the query resolves. */
  rolloutAvailability: Map<string, boolean>;
  onPick: (stageName: string) => void;
}) {
  void missionSlug;
  // §UX honesty pass: list EVERY stage in mission order — including
  // untrained "pending" stages and not-yet-run sub-stages, which used to
  // be silently dropped from this picker. They're disabled (not hidden)
  // below instead, via `rolloutAvailability`.
  if (stages.length === 0) return null;

  return (
    <div className="rs-flex rs-wrap rs-gap-6" style={{ padding: "8px 14px" }} role="list" aria-label="Mission stages">
      {stages.map((s, idx) => {
        const isSelected = selectedStageName === s.name || (selectedStageName === null && liveStageName === s.name);
        const isErrored = s.status === "failed";
        const isSuperseded = s.status === "superseded";
        const isLive = liveStageName === s.name;
        const hasRollout = rolloutAvailability.get(s.name) ?? false;
        const isDisabled = !hasRollout && !isLive;
        const label = stageLabel(s, idx + 1);
        const tooltip = isErrored
          ? `${label}. ${s.name} — ${failureReasonText(s.failure_reason, s.iterations_used) ?? "errored"}`
          : isSuperseded
            ? `${label}. ${s.name} — ${supersededText(s)}`
            : isDisabled
              ? `${label}. ${s.name} — no rollouts recorded`
              : `${label}. ${s.name}`;
        return (
          <button
            key={s.name}
            type="button"
            role="listitem"
            disabled={isDisabled}
            onClick={() => onPick(s.name)}
            title={tooltip}
            className="mono"
            style={{
              display: "inline-flex", alignItems: "center", gap: 5,
              borderRadius: 5, padding: "3px 8px", fontSize: 11,
              cursor: isDisabled ? "not-allowed" : "pointer",
              opacity: isDisabled ? 0.45 : 1,
              border: "1px solid " + (isSelected ? "var(--rs-primary)" : "var(--hairline)"),
              background: isSelected ? "rgba(245,78,0,0.08)" : "var(--canvas-soft)",
              color: "var(--ink)",
            }}
          >
            {isLive && <span className="rs-dot live" />}
            <span style={{ color: "var(--rs-muted)" }}>{label}.</span>
            <span style={{ fontWeight: 600 }}>{s.name}</span>
            {isErrored && <Icon name="alert-circle" size={11} color="var(--st-rose-fg)" />}
            {isSuperseded && <Icon name="git-branch" size={11} color="var(--rs-muted)" />}
          </button>
        );
      })}
    </div>
  );
}

// ── Replay layer ─────────────────────────────────────────────────────
function ReplayLayer({
  slug, run, iter, onPickIter,
}: { slug: string; run: RunSummary | null; iter: number | null; onPickIter: (n: number) => void }) {
  // Disk-reconstructed rows (run_id "disk:<mission>/<stage>", synthesized
  // after a backend restart) have no JobManager entry: run-scoped clip/WS
  // routes can't resolve them. Their rollouts live at the mission-stage
  // disk-truth endpoint instead.
  const isDiskRow = !!run && run.run_id.startsWith("disk:");
  const liveState = useLiveClips(slug, isDiskRow ? undefined : run?.run_id);
  const availableIters = useMemo(() => {
    const set = new Set<number>();
    liveState.clips.forEach((c) => set.add(c.iter));
    if (run) for (let i = 0; i < run.iterations_completed; i++) set.add(i);
    return Array.from(set).sort((a, b) => a - b);
  }, [liveState.clips, run]);

  useEffect(() => {
    if (iter === null && availableIters.length > 0) onPickIter(availableIters[availableIters.length - 1]);
  }, [iter, availableIters, onPickIter]);

  const src = useMemo(() => {
    if (!run || iter === null) return null;
    if (isDiskRow) {
      if (run.mission_slug && run.stage_name) {
        return stageRolloutUrl(slug, run.mission_slug, run.stage_name, iter);
      }
      // §increment 3: the project-level disk row ("disk:project") has
      // no mission/stage — its rollouts come from the project-scoped
      // disk-truth endpoint.
      return projectIterRolloutUrl(slug, iter);
    }
    return iterRolloutUrl(slug, run.run_id, iter);
  }, [slug, run, iter, isDiskRow]);

  const clipSrc = useMemo(() => {
    if (!run || iter === null || isDiskRow) return null;
    return clipUrl(slug, run.run_id, iter);
  }, [slug, run, iter, isDiskRow]);

  if (!run) return <EmptyOverlay title="No runs yet" />;

  return (
    <>
      <div className="rs-overlay"><Icon name="history" size={13} />replay{iter !== null ? ` · iter ${iter}` : ""}</div>
      {src && (
        <a
          href={src}
          download={`${isDiskRow ? (run.stage_name ?? "project") : run.run_id}_iter_${iter}.mp4`}
          className="rs-overlay"
          style={{ left: "auto", right: 12, textDecoration: "none", cursor: "pointer" }}
          onClick={(e) => e.stopPropagation()}
        >
          <Icon name="download" size={13} />Download
        </a>
      )}
      {src && iter !== null && (
        <video
          key={src}
          src={src}
          aria-label={`Replay of robot rollout, iteration ${iter}`}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }}
          controls playsInline
          onError={(e) => {
            const video = e.currentTarget;
            if (clipSrc && video.src !== window.location.origin + clipSrc) video.src = clipSrc;
          }}
        />
      )}
      {availableIters.length > 0 && (
        <div className="rs-scrub" style={{ position: "absolute", insetInline: 0, bottom: 0, background: "rgba(20,19,13,0.82)", padding: "8px 10px" }}>
          {availableIters.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => onPickIter(n)}
              className="rs-btn rs-btn-xs"
              style={
                n === iter
                  ? { background: "var(--rs-primary)", color: "#fff", borderColor: "var(--rs-primary)" }
                  : { background: "rgba(255,255,255,0.06)", color: "#cfcdc4", borderColor: "rgba(255,255,255,0.16)" }
              }
            >
              iter {n}
            </button>
          ))}
        </div>
      )}
    </>
  );
}

// ── Shared empty + shimmer ────────────────────────────────────────────
function Shimmer({ label }: { label?: string } = {}) {
  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#16150f" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#8d8979" }}>
        <Icon name="loader" size={16} className="rs-spin" />
        {label ?? "Loading…"}
      </div>
    </div>
  );
}

function EmptyOverlay({ title, error }: { title?: string; error?: string }) {
  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, padding: 24, textAlign: "center" }}>
      <Icon name="camera" size={28} color="#6a6657" />
      <div style={{ fontSize: 14, fontWeight: 500, color: "#d7d3c4" }}>{title ?? "No preview available"}</div>
      {error && <p style={{ maxWidth: 420, fontFamily: "var(--font-mono)", fontSize: 12, color: "#8d8979" }}>{error}</p>}
    </div>
  );
}

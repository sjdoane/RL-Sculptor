import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  Download,
  ImageOff,
  Loader2,
  RefreshCcw,
  Video,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { useLiveClips } from "@/hooks/useLiveClips";
import { useProjectPreview } from "@/hooks/useProjectPreview";
import { useMissions } from "@/hooks/useMissions";
import { useRunEvents } from "@/hooks/useRunEvents";
import { useRuns } from "@/hooks/useRuns";
import { clipUrl, iterRolloutUrl, previewUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { CameraAngle, RunSummary } from "@/lib/types";
import { CAMERA_ANGLES } from "@/lib/types";

const ANGLE_LABEL: Record<CameraAngle, string> = {
  iso: "Iso",
  front: "Front",
  side: "Side",
  top: "Top",
};

type Mode = "static" | "live" | "replay";

/** Three-mode robot viewer. Sticky-mode rules (Prompt 9 R2):
 *    - Static → Live auto-transition when a run starts AND the user
 *      hasn't explicitly picked a mode.
 *    - Static → Live on initial page load if a run is already active.
 *    - Replay persists until the user explicitly leaves.
 *    - Live does NOT auto-switch to Static when a run completes — it
 *      stays on the last clip with a "Run completed" overlay.
 */
export function RobotViewer({ slug }: { slug: string }) {
  // §Ship 21d: keep /runs polling through mission stage boundaries so
  // the live-video run selection doesn't freeze on a stale stage when
  // one completes and the next starts.
  const missions = useMissions(slug);
  const missionActive = useMemo(
    () => (missions.data ?? []).some((m) => m.active_job_id != null),
    [missions.data],
  );
  const runs = useRuns(slug, { keepPolling: missionActive });
  const activeRun = useMemo(
    () => (runs.data ?? []).find((r) => r.status === "running" || r.status === "queued") ?? null,
    [runs.data],
  );
  // The "most-recent run" is where replay clips come from even after
  // the run ends.
  const mostRecentRun = useMemo(
    () => pickMostRecent(runs.data ?? []),
    [runs.data],
  );
  const trackedRunId = mostRecentRun?.run_id ?? null;

  const [mode, setMode] = useState<Mode>("static");
  const [userPicked, setUserPicked] = useState(false);

  // Auto-transition Static → Live when a run becomes active and user
  // hasn't overridden.
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

  // Replay state — chosen iter.
  const [replayIter, setReplayIter] = useState<number | null>(null);

  return (
    <div className="flex flex-col overflow-hidden rounded-lg border bg-card">
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2 text-xs">
        <div className="flex items-center gap-2">
          <Video className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-semibold">Robot viewer</span>
          {activeRun && (
            <span className="rounded-sm border border-amber-300/60 bg-amber-50 px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-900">
              run active
            </span>
          )}
        </div>
        <ModeSwitcher
          mode={mode}
          onPick={pickMode}
          hasReplay={!!trackedRunId}
          hasLive={!!trackedRunId}
        />
      </div>
      <div className="relative aspect-video w-full bg-slate-950">
        {mode === "static" && <StaticLayer slug={slug} />}
        {mode === "live" && (
          <LiveLayer
            slug={slug}
            runId={trackedRunId}
            run={mostRecentRun}
          />
        )}
        {mode === "replay" && (
          <ReplayLayer
            slug={slug}
            run={mostRecentRun}
            iter={replayIter}
            onPickIter={setReplayIter}
          />
        )}
      </div>
    </div>
  );
}

function pickMostRecent(runs: RunSummary[]): RunSummary | null {
  if (runs.length === 0) return null;
  // runs are sorted newest-first by the backend (`list`), but be
  // defensive in case the backend ordering changes.
  const sorted = [...runs].sort((a, b) => {
    const ta = a.started_at ? new Date(a.started_at).getTime() : 0;
    const tb = b.started_at ? new Date(b.started_at).getTime() : 0;
    return tb - ta;
  });
  return sorted[0];
}

// ── Mode switcher ────────────────────────────────────────────────────
function ModeSwitcher({
  mode,
  onPick,
  hasLive,
  hasReplay,
}: {
  mode: Mode;
  onPick: (m: Mode) => void;
  hasLive: boolean;
  hasReplay: boolean;
}) {
  const items: Array<{ key: Mode; label: string; disabled?: boolean; hint?: string }> = [
    { key: "static", label: "Static" },
    { key: "live", label: "Live", disabled: !hasLive, hint: hasLive ? undefined : "no runs yet" },
    { key: "replay", label: "Replay", disabled: !hasReplay, hint: hasReplay ? undefined : "no runs yet" },
  ];
  return (
    <div className="inline-flex rounded-md border bg-background p-0.5">
      {items.map((it) => (
        <button
          key={it.key}
          type="button"
          disabled={it.disabled}
          onClick={() => onPick(it.key)}
          title={it.hint}
          className={cn(
            "rounded-sm px-2 py-0.5 text-[11px] font-medium transition-colors",
            mode === it.key
              ? "bg-foreground text-background"
              : "text-muted-foreground hover:text-foreground",
            it.disabled && "cursor-not-allowed opacity-40 hover:text-muted-foreground",
          )}
        >
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
  const { data: url, isLoading, error, invalidate } = useProjectPreview(slug, { angle });

  const onReRender = async () => {
    setReRendering(true);
    try {
      const res = await fetch(previewUrl(slug, { angle, regenerate: true }));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      invalidate();
    } catch (e) {
      toast.error("Re-render failed", { description: (e as Error).message });
    } finally {
      setReRendering(false);
    }
  };

  return (
    <>
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between gap-2 bg-gradient-to-b from-black/50 to-transparent px-3 py-2 text-xs text-slate-100">
        <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider">
          <Camera className="h-3 w-3" />
          static
        </span>
        <div className="flex items-center gap-1.5">
          <Select
            value={angle}
            onChange={(e) => setAngle(e.target.value as CameraAngle)}
            className="h-6 w-20 border-slate-700 bg-slate-900/80 text-[11px] text-slate-100"
            aria-label="Camera angle"
          >
            {CAMERA_ANGLES.map((a) => (
              <option key={a} value={a}>{ANGLE_LABEL[a]}</option>
            ))}
          </Select>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 gap-1 px-1.5 text-[11px] text-slate-100 hover:bg-slate-800"
            onClick={onReRender}
            disabled={reRendering || isLoading}
          >
            <RefreshCcw className={cn("h-3 w-3", reRendering && "animate-spin")} />
            Re-render
          </Button>
        </div>
      </div>
      {(isLoading || reRendering) && <Shimmer />}
      {url && (
        <img
          src={url}
          alt={`${ANGLE_LABEL[angle]} static preview`}
          className="absolute inset-0 h-full w-full object-contain"
        />
      )}
      {error && !isLoading && <EmptyState error={(error as Error).message} />}
    </>
  );
}

// ── Live layer ───────────────────────────────────────────────────────
function LiveLayer({
  slug,
  runId,
  run,
}: {
  slug: string;
  runId: string | null;
  run: RunSummary | null;
}) {
  // §Ship 21b: rollout_streamer is wired only for top-level sculpt_run
  // jobs (run_manager.py). Mission stage runs (kind="mission_stage_run")
  // produce per-iter rollout.mp4 files inside their stage_dir but no
  // 2s live clips. Branch to LiveStageRollout which polls the per-iter
  // rollout endpoint as iter_completed events fire on the run WS.
  if (run?.kind === "mission_stage_run") {
    return <LiveStageRollout slug={slug} runId={runId} run={run} />;
  }

  const { clips, skipped, terminal } = useLiveClips(slug, runId ?? undefined);
  const latest = clips[clips.length - 1] ?? null;
  const [hasEverPlayed, setHasEverPlayed] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (latest && videoRef.current) {
      setHasEverPlayed(true);
      videoRef.current.load();
      videoRef.current.play().catch(() => {
        // Browsers may block autoplay with sound; we're muted so this
        // should succeed. If it doesn't, the user clicks play.
      });
    }
  }, [latest?.url]);

  const runCompleted = run && (run.status === "completed" || run.status === "errored" || run.status === "stopped");
  const errorNoClips = runCompleted && clips.length === 0 && run?.status === "errored";

  return (
    <>
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between gap-2 bg-gradient-to-b from-black/60 to-transparent px-3 py-2 text-xs text-slate-100">
        <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider">
          <span className={cn(
            "inline-block h-1.5 w-1.5 rounded-full",
            run?.status === "running" ? "animate-pulse bg-rose-500" : "bg-slate-500",
          )} />
          live{latest !== null ? ` · iter ${latest.iter}` : ""}
        </span>
        {latest && (
          <span className="font-mono text-[10px] text-slate-300">
            {fmtMetricAt(run, latest.iter)}
          </span>
        )}
      </div>
      {errorNoClips && (
        <EmptyState
          title="Run errored before any clips rendered"
          error={run?.error ?? undefined}
        />
      )}
      {!errorNoClips && latest && (
        <video
          ref={videoRef}
          key={latest.url}
          src={latest.url}
          className="absolute inset-0 h-full w-full object-contain"
          muted
          playsInline
          autoPlay
          loop
        />
      )}
      {!errorNoClips && !latest && !hasEverPlayed && <Shimmer label="waiting for first clip…" />}
      {skipped.length > 0 && (
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-1 rounded-md bg-amber-900/80 px-2 py-1 text-[10px] text-amber-100">
          <AlertCircle className="h-3 w-3" />
          {skipped.length} clip{skipped.length === 1 ? "" : "s"} skipped
          (render backpressure)
        </div>
      )}
      {runCompleted && latest && !errorNoClips && (
        <div className="absolute bottom-3 right-3 z-10 rounded-md bg-emerald-900/80 px-2 py-1 text-[10px] uppercase tracking-wider text-emerald-100">
          Run {run?.status}
        </div>
      )}
      {terminal && (
        <span className="sr-only">Terminal event received.</span>
      )}
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
//
// rollout_streamer (run_manager.py) is wired only for top-level
// sculpt_run jobs and writes 2s live clips to
// `<project>/uploads/live_clips/<run_id>/iter_N.mp4`. Mission stage
// runs DON'T get those clips — but their inner sculpt_run subprocess
// DOES write a full per-iter rollout to
// `<project>/.missions/<m>/stages/<s>/runs/iter_N/rollout/rollout.mp4`.
// Ship 21's `_resolve_run_root` fix made the `iter_rollout` endpoint
// resolve those paths for mission_stage_run kind.
//
// This layer uses the run WS to track which iter most recently
// completed (via `iter_completed` / `iter_rolled_out` / `rollout_done`
// events tee'd by mission_jobs._stream_stdout) and shows that iter's
// full rollout.mp4. As iters complete, the video auto-advances. No
// clip backpressure to worry about — full rollouts already exist on
// disk by the time iter_rolled_out fires.
function LiveStageRollout({
  slug,
  runId,
  run,
}: {
  slug: string;
  runId: string | null;
  run: RunSummary | null;
}) {
  const events = useRunEvents(slug, runId ?? undefined);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Most-recent iter that has a rollout.mp4 on disk. Prefer
  // `iter_rolled_out` (rollout artifact written) > `rollout_done`
  // (some adapters emit this name) > `iter_completed` (artifacts may
  // still be flushing — accepted as fallback).
  const latestIter = useMemo(() => {
    let best: number | null = null;
    for (const ev of events.events) {
      if (
        ev.type === "iter_rolled_out"
        || ev.type === "rollout_done"
        || ev.type === "iter_completed"
      ) {
        const i = (ev as { iter?: unknown }).iter;
        if (typeof i === "number" && (best === null || i > best)) {
          best = i;
        }
      }
    }
    // Fallback: the run's iterations_completed - 1 (server snapshot
    // is more reliable than the WS tail when the WS just opened).
    if (best === null && run && run.iterations_completed > 0) {
      best = run.iterations_completed - 1;
    }
    return best;
  }, [events.events, run]);

  // Auto-play on each iter advance.
  useEffect(() => {
    if (latestIter === null || !videoRef.current) return;
    videoRef.current.load();
    videoRef.current.play().catch(() => {
      // Autoplay blocked despite muted — user can click play.
    });
  }, [latestIter]);

  const src = useMemo(() => {
    if (!run || latestIter === null) return null;
    return iterRolloutUrl(slug, run.run_id, latestIter);
  }, [slug, run, latestIter]);

  const runCompleted =
    run && (run.status === "completed" || run.status === "errored" || run.status === "stopped");
  const errorNoFrames =
    runCompleted && latestIter === null && run?.status === "errored";

  return (
    <>
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between gap-2 bg-gradient-to-b from-black/60 to-transparent px-3 py-2 text-xs text-slate-100">
        <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider">
          <span className={cn(
            "inline-block h-1.5 w-1.5 rounded-full",
            run?.status === "running" ? "animate-pulse bg-rose-500" : "bg-slate-500",
          )} />
          live
          {run?.stage_name && (
            <span className="ml-1 normal-case text-slate-300">
              · {run.stage_name}
            </span>
          )}
          {latestIter !== null ? ` · iter ${latestIter}` : ""}
        </span>
        {latestIter !== null && (
          <span className="font-mono text-[10px] text-slate-300">
            {fmtMetricAt(run, latestIter)}
          </span>
        )}
      </div>
      {errorNoFrames && (
        <EmptyState
          title="Stage errored before any rollout rendered"
          error={run?.error ?? undefined}
        />
      )}
      {!errorNoFrames && src && (
        <video
          ref={videoRef}
          key={src}
          src={src}
          className="absolute inset-0 h-full w-full object-contain"
          muted
          playsInline
          autoPlay
          loop
          onError={() => {
            // Iter rollout 404: still flushing. Silent — the next
            // iter_completed event will trigger another fetch.
          }}
        />
      )}
      {!errorNoFrames && !src && (
        <Shimmer label="waiting for first rollout…" />
      )}
      {runCompleted && src && !errorNoFrames && (
        <div className="absolute bottom-3 right-3 z-10 rounded-md bg-emerald-900/80 px-2 py-1 text-[10px] uppercase tracking-wider text-emerald-100">
          Stage {run?.status}
        </div>
      )}
      {events.terminal && (
        <span className="sr-only">Terminal event received.</span>
      )}
    </>
  );
}

// ── Replay layer ─────────────────────────────────────────────────────
function ReplayLayer({
  slug,
  run,
  iter,
  onPickIter,
}: {
  slug: string;
  run: RunSummary | null;
  iter: number | null;
  onPickIter: (n: number) => void;
}) {
  const liveState = useLiveClips(slug, run?.run_id);
  // Prefer iterations from the run detail (they know which ones have
  // rollouts on disk), fall back to the clips we've seen.
  const availableIters = useMemo(() => {
    const set = new Set<number>();
    liveState.clips.forEach((c) => set.add(c.iter));
    if (run) {
      for (let i = 0; i < run.iterations_completed; i++) set.add(i);
    }
    return Array.from(set).sort((a, b) => a - b);
  }, [liveState.clips, run]);

  // Auto-select the latest iter on first entry to replay mode.
  useEffect(() => {
    if (iter === null && availableIters.length > 0) {
      onPickIter(availableIters[availableIters.length - 1]);
    }
  }, [iter, availableIters, onPickIter]);

  const src = useMemo(() => {
    if (!run || iter === null) return null;
    // Prefer the per-iter full rollout.mp4 (richer than a 2s clip).
    return iterRolloutUrl(slug, run.run_id, iter);
  }, [slug, run, iter]);

  const clipSrc = useMemo(() => {
    if (!run || iter === null) return null;
    return clipUrl(slug, run.run_id, iter);
  }, [slug, run, iter]);

  if (!run) {
    return <EmptyState title="No runs yet" />;
  }

  return (
    <>
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between gap-2 bg-gradient-to-b from-black/60 to-transparent px-3 py-2 text-xs text-slate-100">
        <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider">
          <Video className="h-3 w-3" />
          replay{iter !== null ? ` · iter ${iter}` : ""}
        </span>
        {src && (
          <a
            href={src}
            download={`${run.run_id}_iter_${iter}.mp4`}
            className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900/80 px-2 py-0.5 text-[11px] hover:bg-slate-800"
            onClick={(e) => {
              // Fallback: if the full rollout 404s (not yet written),
              // try the 2s clip instead.
              e.stopPropagation();
            }}
          >
            <Download className="h-3 w-3" />
            Download
          </a>
        )}
      </div>
      {src && iter !== null && (
        <video
          key={src}
          src={src}
          className="absolute inset-0 h-full w-full object-contain"
          controls
          playsInline
          onError={(e) => {
            // Fall back to the 2s clip if the iter rollout isn't on
            // disk yet (edge case: clicked replay immediately after
            // iter_completed but before rollout.mp4 fully flushed).
            const video = e.currentTarget;
            if (clipSrc && video.src !== window.location.origin + clipSrc) {
              video.src = clipSrc;
            }
          }}
        />
      )}
      {availableIters.length > 0 && (
        <div className="absolute inset-x-0 bottom-0 z-10 flex items-center gap-1 overflow-x-auto border-t border-slate-800 bg-slate-900/80 px-2 py-1">
          {availableIters.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => onPickIter(n)}
              className={cn(
                "shrink-0 rounded-sm border px-2 py-0.5 font-mono text-[10px] transition-colors",
                n === iter
                  ? "border-slate-400 bg-slate-800 text-slate-100"
                  : "border-slate-700 bg-slate-950 text-slate-400 hover:bg-slate-800",
              )}
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
    <div className="absolute inset-0 z-0 flex items-center justify-center bg-slate-900">
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" />
        {label ?? "Loading…"}
      </div>
      <div
        className="absolute inset-0 animate-pulse bg-[linear-gradient(90deg,rgba(255,255,255,0)_0%,rgba(255,255,255,0.04)_50%,rgba(255,255,255,0)_100%)]"
        style={{ animationDuration: "2s" }}
      />
    </div>
  );
}

function EmptyState({ title, error }: { title?: string; error?: string }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center">
      <ImageOff className="h-8 w-8 text-slate-500" />
      <div className="text-sm font-medium text-slate-200">
        {title ?? "No preview available"}
      </div>
      {error && (
        <p className="max-w-md font-mono text-xs text-slate-400">{error}</p>
      )}
    </div>
  );
}

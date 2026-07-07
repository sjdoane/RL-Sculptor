import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { MissionAdvanced, type MissionRenderSize } from "@/components/NewMissionDialog";
import { Btn, Modal } from "@/components/rs/primitives";
import { useRunMission, type RunMissionVariables } from "@/hooks/useMissions";
import { useProjectMetrics } from "@/hooks/useMetrics";
import { ApiError, type RunMissionRequestBody } from "@/lib/api";
import type { MissionDetail } from "@/lib/types";

/** Per-launch configuration for a mission run.
 *
 * Three regions (rendered via the shared MissionAdvanced):
 *   1. Iteration overrides (max-iters per stage, steps_per_iter, seed).
 *   2. Adaptive early-finish: stop a stage when its goal is met.
 *   3. Adaptive extension: keep training a stage that's still improving.
 *
 * Defaults preserve the standard behavior: no overrides, neither
 * adaptive option enabled, full per-stage budget from the mission's
 * curriculum. The user must opt in explicitly so a "normal" mission
 * run is unchanged. */
export function RunMissionDialog({
  slug,
  missionSlug,
  mission,
  disabled,
  disabledTitle,
  trigger,
}: {
  slug: string;
  missionSlug: string;
  mission: MissionDetail | null | undefined;
  disabled?: boolean;
  disabledTitle?: string;
  /** Optional custom trigger element. When omitted a default Play-icon
   *  button is used. Clicking the trigger opens the dialog. */
  trigger?: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);

  // Iteration overrides (empty string = "use config.toml default").
  const [iterations, setIterations] = useState<number | "">("");
  const [stepsPerIter, setStepsPerIter] = useState<number | "">("");
  const [seed, setSeed] = useState<number | "">("");

  // Goal A.
  const [earlyStopOnCriterion, setEarlyStopOnCriterion] = useState(false);
  const [stabilityWindow, setStabilityWindow] = useState<number | "">(1);

  // Goal B.
  const [extendOnImprovement, setExtendOnImprovement] = useState(false);
  const [maxExtensions, setMaxExtensions] = useState<number | "">(1);
  const [extensionFactor, setExtensionFactor] = useState<number | "">(0.5);
  const [extensionThreshold, setExtensionThreshold] =
    useState<number | "">(0.05);
  // §Ship 34/35: objective fitness-in-the-loop (uniform across stages).
  // string holds built-in names AND generated "gen:<id>" refs.
  const [fitnessMetric, setFitnessMetric] = useState<string | null>(null);
  const [fitnessMode, setFitnessMode] = useState<"observe" | "steer">("steer");
  // §MISSION_RUN_PARITY: per-launch knobs mirrored from NewRunDialog.
  const [editCandidates, setEditCandidates] = useState<number | "">("");
  const [rolloutEpisodes, setRolloutEpisodes] = useState<number | "">("");
  const [maxEpisodeSteps, setMaxEpisodeSteps] = useState<number | "">("");
  const [playbackSpeed, setPlaybackSpeed] = useState<number | "">("");
  const [renderSize, setRenderSize] = useState<MissionRenderSize>("default");
  const [fitnessPatience, setFitnessPatience] = useState<number | "">("");
  const [numEnvs, setNumEnvs] = useState<number | "">("");
  const [device, setDevice] = useState<string>("");
  const projectMetrics = useProjectMetrics(slug, open);

  const run = useRunMission(slug);

  // §MISSION_RUN_PARITY: bundle the parity knobs for the shared form.
  const knobs = {
    editCandidates, setEditCandidates,
    rolloutEpisodes, setRolloutEpisodes,
    maxEpisodeSteps, setMaxEpisodeSteps,
    playbackSpeed, setPlaybackSpeed,
    renderSize, setRenderSize,
    fitnessPatience, setFitnessPatience,
    numEnvs, setNumEnvs,
    device, setDevice,
  };

  // Pre-fill iteration override with the mission's max-stage iters
  // when first opening, so the default reflects what the user already
  // signed up for via Claude's decomposition. Bumping it down clamps;
  // bumping it up applies to every stage uniformly.
  const suggestedIters = useMemo(() => {
    if (!mission?.stages?.length) return null;
    return Math.max(...mission.stages.map((s) => s.max_iterations || 0));
  }, [mission]);

  // §Ship 21a: pre-fill from `mission.run_defaults` if the user set
  // them at creation time via the NewMissionDialog Advanced tab. The
  // run_defaults take precedence over `suggestedIters` (which is just
  // Claude's authored cap). User can still tweak before launching.
  // Tracked-once so re-renders don't clobber user edits mid-session.
  const [appliedDefaults, setAppliedDefaults] = useState(false);
  useEffect(() => {
    if (!open) {
      setAppliedDefaults(false);
      return;
    }
    if (appliedDefaults) return;
    const rd = mission?.run_defaults;
    if (rd) {
      if (typeof rd.iterations_override === "number") {
        setIterations(rd.iterations_override);
      } else if (suggestedIters) {
        setIterations(suggestedIters);
      }
      if (typeof rd.steps_per_iter === "number") {
        setStepsPerIter(rd.steps_per_iter);
      }
      if (typeof rd.seed === "number") {
        setSeed(rd.seed);
      }
      if (rd.early_stop_on_criterion) {
        setEarlyStopOnCriterion(true);
        if (typeof rd.criterion_stability_window === "number") {
          setStabilityWindow(rd.criterion_stability_window);
        }
      }
      if (rd.extend_on_improvement) {
        setExtendOnImprovement(true);
        if (typeof rd.max_extensions_per_stage === "number") {
          setMaxExtensions(rd.max_extensions_per_stage);
        }
        if (typeof rd.extension_factor === "number") {
          setExtensionFactor(rd.extension_factor);
        }
        if (typeof rd.extension_improvement_threshold === "number") {
          setExtensionThreshold(rd.extension_improvement_threshold);
        }
      }
      if (rd.fitness_metric) {
        setFitnessMetric(rd.fitness_metric);
        if (rd.fitness_mode === "observe" || rd.fitness_mode === "steer") {
          setFitnessMode(rd.fitness_mode);
        }
      }
      // §MISSION_RUN_PARITY: pre-fill the per-launch knobs too.
      if (typeof rd.edit_candidates === "number") setEditCandidates(rd.edit_candidates);
      if (typeof rd.rollout_episodes === "number") setRolloutEpisodes(rd.rollout_episodes);
      if (typeof rd.max_episode_steps === "number") setMaxEpisodeSteps(rd.max_episode_steps);
      if (typeof rd.playback_speed === "number") setPlaybackSpeed(rd.playback_speed);
      if (typeof rd.fitness_patience === "number") setFitnessPatience(rd.fitness_patience);
      if (typeof rd.num_envs_override === "number") setNumEnvs(rd.num_envs_override);
      if (typeof rd.device_override === "string" && rd.device_override) {
        setDevice(rd.device_override);
      }
      if (typeof rd.render_width === "number" && typeof rd.render_height === "number") {
        const combo = `${rd.render_width}x${rd.render_height}`;
        if (combo === "1920x1080" || combo === "960x540" || combo === "320x240") {
          setRenderSize(combo);
        }
      }
      setAppliedDefaults(true);
      return;
    }
    // No persisted defaults — fall back to the Ship 19d behavior
    // (Claude's authored max from suggestedIters).
    if (iterations === "" && suggestedIters) {
      setIterations(suggestedIters);
      setAppliedDefaults(true);
    }
  }, [open, mission, suggestedIters, appliedDefaults, iterations]);

  const submit = () => {
    const body: RunMissionRequestBody = {};
    if (typeof iterations === "number" && iterations !== suggestedIters) {
      body.iterations_override = iterations;
    }
    if (typeof stepsPerIter === "number") {
      body.steps_per_iter = stepsPerIter;
    }
    if (typeof seed === "number") {
      body.seed = seed;
    }
    if (earlyStopOnCriterion) {
      body.early_stop_on_criterion = true;
      if (typeof stabilityWindow === "number") {
        body.criterion_stability_window = stabilityWindow;
      }
    }
    if (extendOnImprovement) {
      body.extend_on_improvement = true;
      if (typeof maxExtensions === "number") {
        body.max_extensions_per_stage = maxExtensions;
      }
      if (typeof extensionFactor === "number") {
        body.extension_factor = extensionFactor;
      }
      if (typeof extensionThreshold === "number") {
        body.extension_improvement_threshold = extensionThreshold;
      }
    }
    if (fitnessMetric) {
      body.fitness_metric = fitnessMetric;
      body.fitness_mode = fitnessMode;
      // §MISSION_RUN_PARITY: patience only meaningful with a metric set.
      if (typeof fitnessPatience === "number") body.fitness_patience = fitnessPatience;
    }
    // §MISSION_RUN_PARITY: per-launch knobs (blank = inherited config).
    if (typeof editCandidates === "number") body.edit_candidates = editCandidates;
    if (typeof rolloutEpisodes === "number") body.rollout_episodes = rolloutEpisodes;
    if (typeof maxEpisodeSteps === "number") body.max_episode_steps = maxEpisodeSteps;
    if (typeof playbackSpeed === "number") body.playback_speed = playbackSpeed;
    if (renderSize !== "default") {
      body.render_width = Number(renderSize.split("x")[0]);
      body.render_height = Number(renderSize.split("x")[1]);
    }
    if (typeof numEnvs === "number") body.num_envs_override = numEnvs;
    if (device.trim()) body.device_override = device.trim();
    const variables: RunMissionVariables = { missionSlug, body };
    run.mutate(variables, {
      onSuccess: () => {
        setOpen(false);
        toast.success("Mission run queued", {
          description: Object.keys(body).length
            ? "Custom config applied; watch the live event stream."
            : "Defaults used; watch the live event stream.",
        });
      },
      onError: (err) => {
        const detail =
          err instanceof ApiError
            ? err.problem.detail ?? err.problem.title
            : err.message;
        toast.error("Could not run mission", { description: detail });
      },
    });
  };

  const eta = useMemo(() => {
    // Ballpark estimate: per-stage budget × 60 s/iter (mjlab Cartpole)
    // up to ~25 min for G1. Useful sanity-check before submission.
    if (!mission?.stages?.length) return null;
    const itersPerStage =
      typeof iterations === "number" ? iterations : suggestedIters ?? 3;
    return itersPerStage * mission.stages.length;
  }, [iterations, mission, suggestedIters]);

  return (
    <>
      {trigger ? (
        <span style={{ display: "contents" }} onClick={() => { if (!disabled) setOpen(true); }}>{trigger}</span>
      ) : (
        <Btn kind="ghost" icon="play" disabled={disabled} title={disabledTitle} onClick={() => setOpen(true)}>
          Run mission
        </Btn>
      )}
      {open && (
        <Modal
          title="Configure mission run"
          subtitle="Applies to every stage. Defaults match the curriculum Claude planned; the two adaptive options are independent opt-ins."
          icon="play"
          onClose={() => { if (!run.isPending) setOpen(false); }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => setOpen(false)} disabled={run.isPending}>Cancel</Btn>
              <Btn kind="primary" icon={run.isPending ? "loader" : "play"} onClick={submit} disabled={run.isPending}>
                {run.isPending ? "Launching…" : "Launch"}
              </Btn>
            </>
          }
        >
          <MissionAdvanced
            disabled={run.isPending}
            iterations={iterations} setIterations={setIterations}
            stepsPerIter={stepsPerIter} setStepsPerIter={setStepsPerIter}
            seed={seed} setSeed={setSeed}
            earlyStopOnCriterion={earlyStopOnCriterion} setEarlyStopOnCriterion={setEarlyStopOnCriterion}
            stabilityWindow={stabilityWindow} setStabilityWindow={setStabilityWindow}
            extendOnImprovement={extendOnImprovement} setExtendOnImprovement={setExtendOnImprovement}
            maxExtensions={maxExtensions} setMaxExtensions={setMaxExtensions}
            extensionFactor={extensionFactor} setExtensionFactor={setExtensionFactor}
            extensionThreshold={extensionThreshold} setExtensionThreshold={setExtensionThreshold}
            fitnessMetric={fitnessMetric} setFitnessMetric={setFitnessMetric}
            fitnessMode={fitnessMode} setFitnessMode={setFitnessMode}
            metrics={projectMetrics.data ?? []}
            showIterationsHint={suggestedIters?.toString() ?? "3"}
            knobs={knobs}
          />
          {eta !== null && (
            <p className="rs-hintline" style={{ marginTop: 4 }}>
              Rough estimate — ~{eta} rounds across {mission?.stages?.length ?? 0} stage(s); multiply by
              your per-round wall-clock (Cartpole ≈ 30 s, G1 ≈ 25 min).
            </p>
          )}
        </Modal>
      )}
    </>
  );
}

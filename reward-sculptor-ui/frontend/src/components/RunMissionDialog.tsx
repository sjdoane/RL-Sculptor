import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { MissionAdvanced, type MissionRenderSize } from "@/components/NewMissionDialog";
import { Banner, Btn, Modal } from "@/components/rs/primitives";
import {
  useRegenerateStageMetric,
  useRunMission,
  type RunMissionVariables,
} from "@/hooks/useMissions";
import { useJob } from "@/hooks/useJob";
import { useProjectMetrics } from "@/hooks/useMetrics";
import { ApiError, type RunMissionRequestBody } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import { useQueryClient } from "@tanstack/react-query";
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
  const qc = useQueryClient();

  // §mission-persistence increment 2: when POST .../run 409s with
  // type "/problems/stage-metrics-missing", the backend names the
  // offending stages in `missing_stages` (see run_mission in
  // backend/routes/missions.py). Render an inline banner instead of
  // just a toast so the user can act (regenerate or proceed blind)
  // without leaving the dialog. Cleared on open/close and on the next
  // successful or differently-typed submit.
  const [missingStages, setMissingStages] = useState<string[] | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const regen = useRegenerateStageMetric(slug);
  const [regenStage, setRegenStage] = useState<string | null>(null);
  const [regenJobId, setRegenJobId] = useState<string | null>(null);
  useJob(regenJobId ?? undefined, {
    enabled: regenJobId != null,
    onTerminal: (job) => {
      setRegenJobId(null);
      setRegenStage(null);
      qc.invalidateQueries({ queryKey: qk.mission(slug, missionSlug) });
      if (job.status === "completed") {
        toast.success("Metric regeneration finished");
      } else if (job.status === "errored") {
        toast.error("Metric regeneration failed");
      }
    },
  });
  const regenerateStage = (stageName: string) => {
    setRegenStage(stageName);
    regen.mutate(
      { missionSlug, stageName },
      {
        onSuccess: (job) => {
          setRegenJobId(job.job_id);
          toast.success("Metric regeneration started", { description: `stage ${stageName}` });
        },
        onError: (err) => {
          setRegenStage(null);
          const detail = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
          toast.error("Could not regenerate metric", { description: detail });
        },
      },
    );
  };

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

  // Clear any stale 409 banner + in-flight regen tracking whenever the
  // dialog opens or closes, so a re-open starts clean.
  useEffect(() => {
    setMissingStages(null);
    setLaunchError(null);
    setRegenStage(null);
    setRegenJobId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

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

  const buildBody = (): RunMissionRequestBody => {
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
    return body;
  };

  // §mission-persistence increment 2: `proceedBlind` re-submits the
  // IDENTICAL run request with `proceed_blind: true` set, bypassing
  // the stage_metric_required 409 guard for this one launch. Normal
  // submits never set it.
  const submit = (proceedBlind?: boolean) => {
    const ranges: Array<[string, number | "", number, number]> = [
      ["Rounds per stage", iterations, 1, 200],
      ["Steps per round", stepsPerIter, 100, 200_000],
      ["Seed", seed, 0, 2_000_000_000],
      ["Edit candidates", editCandidates, 1, 5],
      ["Rollout episodes", rolloutEpisodes, 1, 32],
      ["Episode steps", maxEpisodeSteps, 50, 5_000],
      ["Playback speed", playbackSpeed, 0.1, 10],
      ["num envs", numEnvs, 1, 8_192],
      ["Fitness patience", fitnessPatience, 1, 50],
    ];
    if (earlyStopOnCriterion) ranges.push(["Stability window", stabilityWindow, 1, 10]);
    if (extendOnImprovement) {
      ranges.push(
        ["Max extensions", maxExtensions, 0, 3],
        ["Extension factor", extensionFactor, 0.1, 1.5],
        ["Improvement threshold", extensionThreshold, 0, 1],
      );
    }
    const invalid = ranges.find(([, value, min, max]) =>
      typeof value === "number" && (!Number.isFinite(value) || value < min || value > max));
    if (invalid) {
      const [label, , min, max] = invalid;
      const message = `${label} must be between ${min.toLocaleString()} and ${max.toLocaleString()}.`;
      setLaunchError(message);
      toast.error("Check mission configuration", { description: message });
      return;
    }
    const integerLabels = new Set([
      "Rounds per stage", "Steps per round", "Seed", "Edit candidates",
      "Rollout episodes", "Episode steps", "num envs", "Fitness patience",
      "Stability window", "Max extensions",
    ]);
    const nonInteger = ranges.find(([label, value]) =>
      integerLabels.has(label) && typeof value === "number" && !Number.isInteger(value));
    if (nonInteger) {
      const message = `${nonInteger[0]} must be a whole number.`;
      setLaunchError(message);
      toast.error("Check mission configuration", { description: message });
      return;
    }
    setLaunchError(null);
    const body = buildBody();
    if (proceedBlind) body.proceed_blind = true;
    const variables: RunMissionVariables = { missionSlug, body };
    run.mutate(variables, {
      onSuccess: () => {
        setOpen(false);
        setMissingStages(null);
        setLaunchError(null);
        toast.success("Mission run queued", {
          description: Object.keys(body).length
            ? "Custom config applied; watch the live event stream."
            : "Defaults used; watch the live event stream.",
        });
      },
      onError: (err) => {
        if (
          err instanceof ApiError &&
          err.status === 409 &&
          err.type === "/problems/stage-metrics-missing"
        ) {
          const stages = err.problem.missing_stages;
          setMissingStages(Array.isArray(stages) ? stages.filter((s): s is string => typeof s === "string") : []);
          return;
        }
        setMissingStages(null);
        const detail =
          err instanceof ApiError
            ? err.problem.detail ?? err.problem.title
            : err.message;
        toast.error("Could not run mission", { description: detail });
        setLaunchError(detail);
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

  // §mission-persistence increment 2: RunMissionDialog has no
  // genStageMetrics toggle (the mission is already created) — derive
  // the Advanced-tab consistency hint from whether any stage actually
  // has a per-stage metric ref (accepted/rejected/inherited all count;
  // only "none"/null means the stage never got one).
  const stageMetricsMode = useMemo<"on" | "off">(() => {
    const stages = mission?.stages ?? [];
    const anyStageMetric = stages.some(
      (s) => !!s.steering_metric || (s.metric_status && s.metric_status !== "none"),
    );
    return anyStageMetric ? "on" : "off";
  }, [mission]);

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
              <Btn kind="primary" icon={run.isPending ? "loader" : "play"} onClick={() => submit()} disabled={run.isPending}>
                {run.isPending ? "Launching…" : "Launch"}
              </Btn>
            </>
          }
        >
          {launchError && (
            <Banner kind="err" icon="alert-triangle">
              <span style={{ fontWeight: 600, display: "block" }}>Check mission configuration</span>
              <span style={{ fontSize: 12, display: "block" }}>{launchError}</span>
            </Banner>
          )}
          {missingStages && missingStages.length > 0 && (
            <Banner kind="warn" icon="alert-triangle">
              <span style={{ fontWeight: 600, display: "block" }}>
                Stage metrics missing — this mission requires them
              </span>
              <span style={{ fontSize: 12, display: "block", marginBottom: 8 }}>
                The following runnable stage(s) have no accepted objective metric:{" "}
                <code className="mono">{missingStages.join(", ")}</code>. Regenerate their
                metrics, or proceed without per-stage steering for this launch.
              </span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
                {missingStages.map((name) => {
                  const busy = regenStage === name && (regen.isPending || regenJobId != null);
                  return (
                    <Btn
                      key={name}
                      kind="ghost" size="xs" icon={busy ? "loader" : "sparkles"}
                      disabled={busy}
                      onClick={() => regenerateStage(name)}
                    >
                      {busy ? `Regenerating ${name}…` : `Regenerate ${name}`}
                    </Btn>
                  );
                })}
                <Btn
                  kind="quiet" size="xs" icon="play"
                  disabled={run.isPending}
                  onClick={() => submit(true)}
                >
                  Proceed blind anyway
                </Btn>
              </div>
            </Banner>
          )}
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
            stageMetricsMode={stageMetricsMode}
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

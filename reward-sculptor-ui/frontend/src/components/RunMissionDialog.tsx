import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { MissionAdvanced } from "@/components/NewMissionDialog";
import { Btn, Modal } from "@/components/rs/primitives";
import { useRunMission, type RunMissionVariables } from "@/hooks/useMissions";
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

  const run = useRunMission(slug);

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
            showIterationsHint={suggestedIters?.toString() ?? "3"}
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

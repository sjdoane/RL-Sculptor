import { useEffect, useMemo, useState } from "react";
import { Loader2, Play } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useRunMission, type RunMissionVariables } from "@/hooks/useMissions";
import { ApiError, type RunMissionRequestBody } from "@/lib/api";
import type { MissionDetail } from "@/lib/types";

/** Per-launch configuration for a mission run.
 *
 * Three regions:
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
  /** Optional custom trigger element (e.g., the existing "Run mission"
   *  button styled to match the dialog footer). When omitted a
   *  default Play-icon button is used. */
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
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (disabled && o) return;
        setOpen(o);
      }}
    >
      <DialogTrigger asChild>
        {trigger ?? (
          <Button
            variant="outline"
            disabled={disabled}
            title={disabledTitle}
          >
            <Play className="h-3.5 w-3.5" />
            Run mission
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Configure mission run</DialogTitle>
          <DialogDescription className="text-[11px]">
            Settings apply to <strong>every stage</strong> of this
            mission run. Defaults match the curriculum Claude planned;
            tweak to speed up testing or run an overnight job. The
            two adaptive options below are independent opt-ins.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          {/* ── Iteration overrides ─────────────────────────────────── */}
          <section className="rounded-md border bg-muted/20 p-3">
            <h3 className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Iteration overrides
            </h3>
            <div className="grid grid-cols-3 gap-2">
              <div className="grid gap-1">
                <Label htmlFor="iters" className="text-[11px]">
                  Rounds per stage
                </Label>
                <Input
                  id="iters"
                  type="number"
                  min={1}
                  max={200}
                  value={iterations}
                  onChange={(e) => {
                    const v = e.target.value;
                    setIterations(v === "" ? "" : Number(v));
                  }}
                  disabled={run.isPending}
                  placeholder={suggestedIters?.toString() ?? "3"}
                />
                <p className="text-[10px] text-muted-foreground">
                  Train→evaluate→edit cycles. Smaller = faster test.
                </p>
              </div>
              <div className="grid gap-1">
                <Label htmlFor="steps" className="text-[11px]">
                  Steps per round
                </Label>
                <Input
                  id="steps"
                  type="number"
                  min={100}
                  max={200000}
                  value={stepsPerIter}
                  onChange={(e) => {
                    const v = e.target.value;
                    setStepsPerIter(v === "" ? "" : Number(v));
                  }}
                  disabled={run.isPending}
                  placeholder="project default"
                />
                <p className="text-[10px] text-muted-foreground">
                  Training steps inside one round. Leave blank to
                  use the project's default.
                </p>
              </div>
              <div className="grid gap-1">
                <Label htmlFor="rseed" className="text-[11px]">
                  Seed
                </Label>
                <Input
                  id="rseed"
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(e) => {
                    const v = e.target.value;
                    setSeed(v === "" ? "" : Number(v));
                  }}
                  disabled={run.isPending}
                  placeholder="42"
                />
                <p className="text-[10px] text-muted-foreground">
                  Per-round base seed.
                </p>
              </div>
            </div>
            {eta !== null && (
              <p className="mt-2 text-[10px] text-muted-foreground">
                Rough estimate — ~{eta} rounds across{" "}
                {mission?.stages?.length ?? 0} stage(s); multiply by
                your per-round wall-clock (Cartpole ≈30 s, G1 ≈25 min).
              </p>
            )}
          </section>

          {/* ── Adaptive early-finish ───────────────────────────────── */}
          <section className="rounded-md border bg-muted/20 p-3">
            <label className="flex cursor-pointer items-start gap-2 text-[11px]">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={earlyStopOnCriterion}
                onChange={(e) => setEarlyStopOnCriterion(e.target.checked)}
                disabled={run.isPending}
              />
              <span>
                <span className="font-semibold">
                  Stop when the goal is met
                </span>
                <p className="mt-0.5 text-[10.5px] font-normal text-muted-foreground">
                  Exit a stage as soon as its{" "}
                  <code>success_criterion</code> holds, instead of
                  running every round in the budget. Default off.
                </p>
              </span>
            </label>
            {earlyStopOnCriterion && (
              <div className="ml-6 mt-2 grid gap-1">
                <Label htmlFor="stab" className="text-[11px]">
                  Stability window
                </Label>
                <Input
                  id="stab"
                  type="number"
                  min={1}
                  max={10}
                  value={stabilityWindow}
                  onChange={(e) => {
                    const v = e.target.value;
                    setStabilityWindow(v === "" ? "" : Number(v));
                  }}
                  disabled={run.isPending}
                  className="w-24"
                />
                <p className="text-[10px] text-muted-foreground">
                  Consecutive rounds the criterion must hold before
                  exiting. <code>1</code> = exit on first pass; bump
                  to <code>2-3</code> for noisy metrics.
                </p>
              </div>
            )}
          </section>

          {/* ── Adaptive extension ──────────────────────────────────── */}
          <section className="rounded-md border bg-muted/20 p-3">
            <label className="flex cursor-pointer items-start gap-2 text-[11px]">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={extendOnImprovement}
                onChange={(e) => setExtendOnImprovement(e.target.checked)}
                disabled={run.isPending}
              />
              <span>
                <span className="font-semibold">
                  Keep training while still improving
                </span>
                <p className="mt-0.5 text-[10.5px] font-normal text-muted-foreground">
                  If a stage runs out of rounds but the metric is still
                  trending up, grant extra rounds. Default off. Will
                  not extend if the metric has plateaued.
                </p>
              </span>
            </label>
            {extendOnImprovement && (
              <div className="ml-6 mt-2 grid grid-cols-3 gap-2">
                <div className="grid gap-1">
                  <Label htmlFor="ext-max" className="text-[11px]">
                    Max extensions
                  </Label>
                  <Input
                    id="ext-max"
                    type="number"
                    min={0}
                    max={3}
                    value={maxExtensions}
                    onChange={(e) => {
                      const v = e.target.value;
                      setMaxExtensions(v === "" ? "" : Number(v));
                    }}
                    disabled={run.isPending}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Hard cap (≤ 3).
                  </p>
                </div>
                <div className="grid gap-1">
                  <Label htmlFor="ext-factor" className="text-[11px]">
                    Factor
                  </Label>
                  <Input
                    id="ext-factor"
                    type="number"
                    step={0.1}
                    min={0.1}
                    max={1.5}
                    value={extensionFactor}
                    onChange={(e) => {
                      const v = e.target.value;
                      setExtensionFactor(v === "" ? "" : Number(v));
                    }}
                    disabled={run.isPending}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    × max-iters per extension.
                  </p>
                </div>
                <div className="grid gap-1">
                  <Label htmlFor="ext-thresh" className="text-[11px]">
                    Threshold
                  </Label>
                  <Input
                    id="ext-thresh"
                    type="number"
                    step={0.01}
                    min={0}
                    max={1}
                    value={extensionThreshold}
                    onChange={(e) => {
                      const v = e.target.value;
                      setExtensionThreshold(v === "" ? "" : Number(v));
                    }}
                    disabled={run.isPending}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Recent vs prior best.
                  </p>
                </div>
              </div>
            )}
          </section>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setOpen(false)}
            disabled={run.isPending}
          >
            Cancel
          </Button>
          <Button onClick={submit} disabled={run.isPending}>
            {run.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Launching…
              </>
            ) : (
              <>
                <Play />
                Launch
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

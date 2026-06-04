import { useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
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
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useCreateMission } from "@/hooks/useMissions";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { MissionRunDefaults } from "@/lib/types";

const SLUG_PATTERN = /^[a-z][a-z0-9_-]{0,63}$/;
const GOAL_MIN = 8;
const GOAL_MAX = 2000;

export function NewMissionDialog({
  slug,
  onCreated,
}: {
  slug: string;
  /** Called with the new mission's slug on successful decompose-job
   *  submission. Lets the parent (RunsTab in Ship 21+) auto-open the
   *  detail dialog so the user sees the live decompose stream rather
   *  than staring at a row that pulses for 30-90s with no other
   *  feedback. */
  onCreated?: (missionSlug: string) => void;
}) {
  const [open, setOpen] = useState(false);

  // ── Basic-tab state ──────────────────────────────────────────────
  const [goal, setGoal] = useState("");
  const [missionSlug, setMissionSlug] = useState("");
  const [noKg, setNoKg] = useState(false);

  // ── Advanced-tab state (mirrors RunMissionDialog v2 fields) ───────
  // Empty-string sentinels for numeric inputs so users can clear a
  // field without it snapping to 0. Submission converts empty → omit.
  const [iterations, setIterations] = useState<number | "">("");
  const [stepsPerIter, setStepsPerIter] = useState<number | "">("");
  const [seed, setSeed] = useState<number | "">("");

  // Adaptive early-finish (formerly "Goal A").
  const [earlyStopOnCriterion, setEarlyStopOnCriterion] = useState(false);
  const [stabilityWindow, setStabilityWindow] = useState<number | "">(1);

  // Adaptive extension (formerly "Goal B").
  const [extendOnImprovement, setExtendOnImprovement] = useState(false);
  const [maxExtensions, setMaxExtensions] = useState<number | "">(1);
  const [extensionFactor, setExtensionFactor] = useState<number | "">(0.5);
  const [extensionThreshold, setExtensionThreshold] =
    useState<number | "">(0.05);

  const create = useCreateMission(slug);

  const reset = () => {
    setGoal("");
    setMissionSlug("");
    setNoKg(false);
    setIterations("");
    setStepsPerIter("");
    setSeed("");
    setEarlyStopOnCriterion(false);
    setStabilityWindow(1);
    setExtendOnImprovement(false);
    setMaxExtensions(1);
    setExtensionFactor(0.5);
    setExtensionThreshold(0.05);
  };

  // §Ship 21a: collect Advanced-tab inputs into a MissionRunDefaults
  // payload. Returns null when no field has been touched away from
  // its default — keeps the wire payload empty for users who only
  // touched the Basic tab.
  const buildRunDefaults = (): MissionRunDefaults | null => {
    const out: MissionRunDefaults = {};
    let touched = false;
    if (typeof iterations === "number") {
      out.iterations_override = iterations;
      touched = true;
    }
    if (typeof stepsPerIter === "number") {
      out.steps_per_iter = stepsPerIter;
      touched = true;
    }
    if (typeof seed === "number") {
      out.seed = seed;
      touched = true;
    }
    if (earlyStopOnCriterion) {
      out.early_stop_on_criterion = true;
      if (typeof stabilityWindow === "number") {
        out.criterion_stability_window = stabilityWindow;
      }
      touched = true;
    }
    if (extendOnImprovement) {
      out.extend_on_improvement = true;
      if (typeof maxExtensions === "number") {
        out.max_extensions_per_stage = maxExtensions;
      }
      if (typeof extensionFactor === "number") {
        out.extension_factor = extensionFactor;
      }
      if (typeof extensionThreshold === "number") {
        out.extension_improvement_threshold = extensionThreshold;
      }
      touched = true;
    }
    return touched ? out : null;
  };

  const submit = () => {
    const trimmedGoal = goal.trim();
    if (trimmedGoal.length < GOAL_MIN) {
      toast.error("Goal too short", {
        description: `At least ${GOAL_MIN} characters.`,
      });
      return;
    }
    if (trimmedGoal.length > GOAL_MAX) {
      toast.error("Goal too long", {
        description: `At most ${GOAL_MAX} characters.`,
      });
      return;
    }
    const trimmedSlug = missionSlug.trim();
    if (trimmedSlug && !SLUG_PATTERN.test(trimmedSlug)) {
      toast.error("Invalid mission slug", {
        description:
          "Lowercase letter first, then letters/digits/underscores/hyphens; ≤64 chars.",
      });
      return;
    }
    const runDefaults = buildRunDefaults();
    create.mutate(
      {
        goal: trimmedGoal,
        mission_slug: trimmedSlug || undefined,
        no_kg: noKg || undefined,
        run_defaults: runDefaults ?? undefined,
      },
      {
        onSuccess: (job) => {
          setOpen(false);
          reset();
          // Backend's POST /missions returns 202 with `job.params`
          // populated (mission_slug + goal + no_kg + run_defaults).
          // The frontend `JobSummary` type omits `params` (only
          // JobDetail has it) so we type-cast at this single call
          // site rather than widen the shared type. The HTTP payload
          // always carries it — see backend/routes/missions.py.
          const ms = (
            job as unknown as { params?: { mission_slug?: string } }
          ).params?.mission_slug;
          if (ms && onCreated) onCreated(ms);
          toast.success("Decompose job queued", {
            description: runDefaults
              ? "Advanced settings will pre-fill Run mission once decomposition completes."
              : `job_id: ${job.job_id}`,
          });
        },
        onError: (err) => {
          const detail =
            err instanceof ApiError
              ? err.problem.detail ?? err.problem.title
              : err.message;
          toast.error("Could not create mission", { description: detail });
        },
      },
    );
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (!o) reset();
      }}
    >
      <DialogTrigger asChild>
        <Button size="sm">
          <Sparkles />
          New mission
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>New mission</DialogTitle>
          <DialogDescription>
            Claude decomposes the goal into a curriculum of stages. The
            decompose job runs immediately; once it finishes, the
            mission becomes <code>ready</code> and can be run.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          {/* ── Basic tab ──────────────────────────────────────────── */}
          <TabsContent value="basic" className="mt-3 space-y-3">
            <div className="grid gap-1.5">
              <Label htmlFor="mission-goal">Goal</Label>
              <Textarea
                id="mission-goal"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="Stand on one leg and kick a ball with the other."
                rows={4}
                maxLength={GOAL_MAX}
                disabled={create.isPending}
                aria-invalid={
                  goal.trim().length > 0 && goal.trim().length < GOAL_MIN
                }
                aria-describedby="mission-goal-hint"
              />
              <p
                id="mission-goal-hint"
                className={cn(
                  "text-[10.5px]",
                  goal.trim().length > 0 && goal.trim().length < GOAL_MIN
                    ? "text-destructive"
                    : "text-muted-foreground",
                )}
              >
                {goal.trim().length} / {GOAL_MAX} chars · min {GOAL_MIN}
              </p>
            </div>

            <div className="grid gap-1.5">
              <Label htmlFor="mission-slug">
                Mission slug{" "}
                <span className="text-[10.5px] font-normal text-muted-foreground">
                  (optional override)
                </span>
              </Label>
              <Input
                id="mission-slug"
                value={missionSlug}
                onChange={(e) => setMissionSlug(e.target.value)}
                placeholder="auto — derived from goal"
                maxLength={64}
                disabled={create.isPending}
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
              <p className="text-[10.5px] text-muted-foreground">
                Lowercase letter first, then [a-z0-9_-]; ≤64 chars. Leave
                blank to auto-derive.
              </p>
            </div>

            <label className="flex cursor-pointer items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={noKg}
                onChange={(e) => setNoKg(e.target.checked)}
                disabled={create.isPending}
              />
              <span>
                <span className="font-medium">--no-kg</span>
                <span className="ml-1 text-muted-foreground">
                  ablation: decomposer ignores the literature graph
                </span>
              </span>
            </label>
          </TabsContent>

          {/* ── Advanced tab ───────────────────────────────────────── */}
          <TabsContent value="advanced" className="mt-3 space-y-3">
            <p className="text-[10.5px] text-muted-foreground">
              Run-time defaults persisted on the mission. When you
              later click <strong>Run mission</strong>, these pre-fill
              the launch dialog — you can still tweak per-launch.
              Leave blank to use Claude's per-stage budget at run time.
            </p>

            {/* Iteration overrides */}
            <section className="rounded-md border bg-muted/20 p-3">
              <h3 className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                Iteration overrides
              </h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="grid gap-1">
                  <Label htmlFor="adv-iters" className="text-[11px]">
                    Rounds per stage
                  </Label>
                  <Input
                    id="adv-iters"
                    type="number"
                    min={1}
                    max={200}
                    value={iterations}
                    onChange={(e) => {
                      const v = e.target.value;
                      setIterations(v === "" ? "" : Number(v));
                    }}
                    disabled={create.isPending}
                    placeholder="claude default"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Train→evaluate→edit cycles per stage.
                  </p>
                </div>
                <div className="grid gap-1">
                  <Label htmlFor="adv-steps" className="text-[11px]">
                    Steps per round
                  </Label>
                  <Input
                    id="adv-steps"
                    type="number"
                    min={100}
                    max={200000}
                    value={stepsPerIter}
                    onChange={(e) => {
                      const v = e.target.value;
                      setStepsPerIter(v === "" ? "" : Number(v));
                    }}
                    disabled={create.isPending}
                    placeholder="project default"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Training steps inside one round.
                  </p>
                </div>
                <div className="grid gap-1">
                  <Label htmlFor="adv-seed" className="text-[11px]">
                    Seed
                  </Label>
                  <Input
                    id="adv-seed"
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(e) => {
                      const v = e.target.value;
                      setSeed(v === "" ? "" : Number(v));
                    }}
                    disabled={create.isPending}
                    placeholder="42"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Per-round base seed.
                  </p>
                </div>
              </div>
            </section>

            {/* Adaptive early-finish */}
            <section className="rounded-md border bg-muted/20 p-3">
              <label className="flex cursor-pointer items-start gap-2 text-[11px]">
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={earlyStopOnCriterion}
                  onChange={(e) => setEarlyStopOnCriterion(e.target.checked)}
                  disabled={create.isPending}
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
                  <Label htmlFor="adv-stab" className="text-[11px]">
                    Stability window
                  </Label>
                  <Input
                    id="adv-stab"
                    type="number"
                    min={1}
                    max={10}
                    value={stabilityWindow}
                    onChange={(e) => {
                      const v = e.target.value;
                      setStabilityWindow(v === "" ? "" : Number(v));
                    }}
                    disabled={create.isPending}
                    className="w-24"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Consecutive rounds the criterion must hold before
                    exiting. <code>1</code> = exit on first pass.
                  </p>
                </div>
              )}
            </section>

            {/* Adaptive extension */}
            <section className="rounded-md border bg-muted/20 p-3">
              <label className="flex cursor-pointer items-start gap-2 text-[11px]">
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={extendOnImprovement}
                  onChange={(e) => setExtendOnImprovement(e.target.checked)}
                  disabled={create.isPending}
                />
                <span>
                  <span className="font-semibold">
                    Keep training while still improving
                  </span>
                  <p className="mt-0.5 text-[10.5px] font-normal text-muted-foreground">
                    If a stage runs out of rounds but the metric is
                    still trending up, grant extra rounds. Default off.
                    Will not extend if the metric has plateaued.
                  </p>
                </span>
              </label>
              {extendOnImprovement && (
                <div className="ml-6 mt-2 grid grid-cols-3 gap-2">
                  <div className="grid gap-1">
                    <Label htmlFor="adv-ext-max" className="text-[11px]">
                      Max extensions
                    </Label>
                    <Input
                      id="adv-ext-max"
                      type="number"
                      min={0}
                      max={3}
                      value={maxExtensions}
                      onChange={(e) => {
                        const v = e.target.value;
                        setMaxExtensions(v === "" ? "" : Number(v));
                      }}
                      disabled={create.isPending}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      Hard cap (≤ 3).
                    </p>
                  </div>
                  <div className="grid gap-1">
                    <Label htmlFor="adv-ext-factor" className="text-[11px]">
                      Factor
                    </Label>
                    <Input
                      id="adv-ext-factor"
                      type="number"
                      step={0.1}
                      min={0.1}
                      max={1.5}
                      value={extensionFactor}
                      onChange={(e) => {
                        const v = e.target.value;
                        setExtensionFactor(v === "" ? "" : Number(v));
                      }}
                      disabled={create.isPending}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      × max-rounds per extension.
                    </p>
                  </div>
                  <div className="grid gap-1">
                    <Label htmlFor="adv-ext-thresh" className="text-[11px]">
                      Threshold
                    </Label>
                    <Input
                      id="adv-ext-thresh"
                      type="number"
                      step={0.01}
                      min={0}
                      max={1}
                      value={extensionThreshold}
                      onChange={(e) => {
                        const v = e.target.value;
                        setExtensionThreshold(v === "" ? "" : Number(v));
                      }}
                      disabled={create.isPending}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      Recent vs prior best.
                    </p>
                  </div>
                </div>
              )}
            </section>
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setOpen(false)}
            disabled={create.isPending}
          >
            Cancel
          </Button>
          <Button onClick={submit} disabled={create.isPending}>
            {create.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Queueing…
              </>
            ) : (
              <>
                <Sparkles />
                Decompose
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

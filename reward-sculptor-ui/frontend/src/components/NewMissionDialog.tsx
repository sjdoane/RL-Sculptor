import { useState } from "react";
import { toast } from "sonner";

import { Btn, Field, Modal, ToggleRow } from "@/components/rs/primitives";
import { useCreateMission } from "@/hooks/useMissions";
import { ApiError } from "@/lib/api";
import type { MissionRunDefaults } from "@/lib/types";

const SLUG_PATTERN = /^[a-z][a-z0-9_-]{0,63}$/;
const GOAL_MIN = 8;
const GOAL_MAX = 2000;

export function NewMissionDialog({
  slug,
  onCreated,
}: {
  slug: string;
  onCreated?: (missionSlug: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"basic" | "advanced">("basic");

  const [goal, setGoal] = useState("");
  const [missionSlug, setMissionSlug] = useState("");
  const [noKg, setNoKg] = useState(false);

  const [iterations, setIterations] = useState<number | "">("");
  const [stepsPerIter, setStepsPerIter] = useState<number | "">("");
  const [seed, setSeed] = useState<number | "">("");
  const [earlyStopOnCriterion, setEarlyStopOnCriterion] = useState(false);
  const [stabilityWindow, setStabilityWindow] = useState<number | "">(1);
  const [extendOnImprovement, setExtendOnImprovement] = useState(false);
  const [maxExtensions, setMaxExtensions] = useState<number | "">(1);
  const [extensionFactor, setExtensionFactor] = useState<number | "">(0.5);
  const [extensionThreshold, setExtensionThreshold] = useState<number | "">(0.05);

  const create = useCreateMission(slug);

  const reset = () => {
    setGoal(""); setMissionSlug(""); setNoKg(false); setTab("basic");
    setIterations(""); setStepsPerIter(""); setSeed("");
    setEarlyStopOnCriterion(false); setStabilityWindow(1);
    setExtendOnImprovement(false); setMaxExtensions(1);
    setExtensionFactor(0.5); setExtensionThreshold(0.05);
  };

  const buildRunDefaults = (): MissionRunDefaults | null => {
    const out: MissionRunDefaults = {};
    let touched = false;
    if (typeof iterations === "number") { out.iterations_override = iterations; touched = true; }
    if (typeof stepsPerIter === "number") { out.steps_per_iter = stepsPerIter; touched = true; }
    if (typeof seed === "number") { out.seed = seed; touched = true; }
    if (earlyStopOnCriterion) {
      out.early_stop_on_criterion = true;
      if (typeof stabilityWindow === "number") out.criterion_stability_window = stabilityWindow;
      touched = true;
    }
    if (extendOnImprovement) {
      out.extend_on_improvement = true;
      if (typeof maxExtensions === "number") out.max_extensions_per_stage = maxExtensions;
      if (typeof extensionFactor === "number") out.extension_factor = extensionFactor;
      if (typeof extensionThreshold === "number") out.extension_improvement_threshold = extensionThreshold;
      touched = true;
    }
    return touched ? out : null;
  };

  const submit = () => {
    const trimmedGoal = goal.trim();
    if (trimmedGoal.length < GOAL_MIN) { toast.error("Goal too short", { description: `At least ${GOAL_MIN} characters.` }); return; }
    if (trimmedGoal.length > GOAL_MAX) { toast.error("Goal too long", { description: `At most ${GOAL_MAX} characters.` }); return; }
    const trimmedSlug = missionSlug.trim();
    if (trimmedSlug && !SLUG_PATTERN.test(trimmedSlug)) {
      toast.error("Invalid mission slug", { description: "Lowercase letter first, then letters/digits/underscores/hyphens; ≤64 chars." });
      return;
    }
    const runDefaults = buildRunDefaults();
    create.mutate(
      { goal: trimmedGoal, mission_slug: trimmedSlug || undefined, no_kg: noKg || undefined, run_defaults: runDefaults ?? undefined },
      {
        onSuccess: (job) => {
          setOpen(false);
          reset();
          const ms = (job as unknown as { params?: { mission_slug?: string } }).params?.mission_slug;
          if (ms && onCreated) onCreated(ms);
          toast.success("Decompose job queued", {
            description: runDefaults ? "Advanced settings will pre-fill Run mission once decomposition completes." : `job_id: ${job.job_id}`,
          });
        },
        onError: (err) => {
          const detail = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
          toast.error("Could not create mission", { description: detail });
        },
      },
    );
  };

  const goalShort = goal.trim().length > 0 && goal.trim().length < GOAL_MIN;

  return (
    <>
      <Btn kind="ghost" size="sm" icon="sparkles" onClick={() => setOpen(true)}>New mission</Btn>
      {open && (
        <Modal
          title="New mission"
          subtitle="Decompose a goal into an auto-curriculum"
          icon="sparkles"
          onClose={() => { if (!create.isPending) { setOpen(false); reset(); } }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => { setOpen(false); reset(); }} disabled={create.isPending}>Cancel</Btn>
              <Btn kind="primary" icon={create.isPending ? "loader" : "sparkles"} onClick={submit} disabled={create.isPending}>
                {create.isPending ? "Queueing…" : "Plan mission"}
              </Btn>
            </>
          }
        >
          <div className="rs-mtabs">
            <button className={tab === "basic" ? "on" : ""} onClick={() => setTab("basic")}>Basic</button>
            <button className={tab === "advanced" ? "on" : ""} onClick={() => setTab("advanced")}>Advanced</button>
          </div>

          {tab === "basic" ? (
            <>
              <Field label="Goal" hint={`${goal.trim().length} / ${GOAL_MAX}`} htmlFor="mission-goal">
                <textarea
                  id="mission-goal"
                  className="rs-textarea"
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder="e.g. Make the G1 squat to ~90° then launch into a stable vertical jump and land upright."
                  style={{ minHeight: 96 }}
                  maxLength={GOAL_MAX}
                  disabled={create.isPending}
                  aria-invalid={goalShort}
                  aria-describedby={goalShort ? "mission-goal-err" : undefined}
                  autoFocus
                />
                {goalShort && <span id="mission-goal-err" className="hint" style={{ color: "var(--st-rose)" }}>min {GOAL_MIN} chars</span>}
              </Field>
              <Field label="Mission slug" hint="optional" htmlFor="mission-slug">
                <input
                  id="mission-slug"
                  className="rs-input mono"
                  value={missionSlug}
                  onChange={(e) => setMissionSlug(e.target.value)}
                  placeholder="auto — derived from goal"
                  maxLength={64}
                  disabled={create.isPending}
                  spellCheck={false}
                  autoCapitalize="off"
                  autoCorrect="off"
                />
              </Field>
              <ToggleRow
                on={noKg} onChange={setNoKg} label="No knowledge graph"
                title={<><code className="mono">--no-kg</code> · skip knowledge-graph seeding</>}
                desc="plan from the goal alone, without paper grounding"
              />
            </>
          ) : (
            <MissionAdvanced
              disabled={create.isPending}
              iterations={iterations} setIterations={setIterations}
              stepsPerIter={stepsPerIter} setStepsPerIter={setStepsPerIter}
              seed={seed} setSeed={setSeed}
              earlyStopOnCriterion={earlyStopOnCriterion} setEarlyStopOnCriterion={setEarlyStopOnCriterion}
              stabilityWindow={stabilityWindow} setStabilityWindow={setStabilityWindow}
              extendOnImprovement={extendOnImprovement} setExtendOnImprovement={setExtendOnImprovement}
              maxExtensions={maxExtensions} setMaxExtensions={setMaxExtensions}
              extensionFactor={extensionFactor} setExtensionFactor={setExtensionFactor}
              extensionThreshold={extensionThreshold} setExtensionThreshold={setExtensionThreshold}
            />
          )}
        </Modal>
      )}
    </>
  );
}

type NumOr = number | "";

// Shared Advanced form — also used by RunMissionDialog (same MissionRunDefaults
// fields). Exported so the two stay in lockstep.
export function MissionAdvanced({
  disabled,
  iterations, setIterations, stepsPerIter, setStepsPerIter, seed, setSeed,
  earlyStopOnCriterion, setEarlyStopOnCriterion, stabilityWindow, setStabilityWindow,
  extendOnImprovement, setExtendOnImprovement, maxExtensions, setMaxExtensions,
  extensionFactor, setExtensionFactor, extensionThreshold, setExtensionThreshold,
  showIterationsHint,
}: {
  disabled: boolean;
  iterations: NumOr; setIterations: (v: NumOr) => void;
  stepsPerIter: NumOr; setStepsPerIter: (v: NumOr) => void;
  seed: NumOr; setSeed: (v: NumOr) => void;
  earlyStopOnCriterion: boolean; setEarlyStopOnCriterion: (v: boolean) => void;
  stabilityWindow: NumOr; setStabilityWindow: (v: NumOr) => void;
  extendOnImprovement: boolean; setExtendOnImprovement: (v: boolean) => void;
  maxExtensions: NumOr; setMaxExtensions: (v: NumOr) => void;
  extensionFactor: NumOr; setExtensionFactor: (v: NumOr) => void;
  extensionThreshold: NumOr; setExtensionThreshold: (v: NumOr) => void;
  showIterationsHint?: string;
}) {
  const numInput = (v: NumOr, set: (x: NumOr) => void, props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input
      {...props}
      type="number"
      className="rs-input mono"
      value={v}
      onChange={(e) => set(e.target.value === "" ? "" : Number(e.target.value))}
      disabled={disabled}
    />
  );
  return (
    <>
      <div className="rs-row3">
        <Field label="Rounds per stage" htmlFor="adv-iters">{numInput(iterations, setIterations, { id: "adv-iters", min: 1, max: 200, placeholder: showIterationsHint ?? "claude default" })}</Field>
        <Field label="Steps per round" htmlFor="adv-steps">{numInput(stepsPerIter, setStepsPerIter, { id: "adv-steps", min: 100, max: 200000, placeholder: "project default" })}</Field>
        <Field label="Seed" htmlFor="adv-seed">{numInput(seed, setSeed, { id: "adv-seed", min: 0, placeholder: "42" })}</Field>
      </div>

      <ToggleRow
        on={earlyStopOnCriterion} onChange={setEarlyStopOnCriterion} label="Stop when goal met"
        title="Stop when the goal is met"
        desc="end a stage once its success_criterion holds"
      />
      {earlyStopOnCriterion && (
        <Field label="Stability window" hint="consecutive rounds" htmlFor="adv-stab">
          {numInput(stabilityWindow, setStabilityWindow, { id: "adv-stab", min: 1, max: 10 })}
        </Field>
      )}

      <ToggleRow
        on={extendOnImprovement} onChange={setExtendOnImprovement} label="Keep improving"
        title="Keep training while still improving"
        desc="extend a stage past its budget if the metric climbs"
      />
      {extendOnImprovement && (
        <div className="rs-row3">
          <Field label="Max extensions" htmlFor="adv-ext-max">{numInput(maxExtensions, setMaxExtensions, { id: "adv-ext-max", min: 0, max: 3 })}</Field>
          <Field label="Factor" htmlFor="adv-ext-factor">{numInput(extensionFactor, setExtensionFactor, { id: "adv-ext-factor", step: 0.1, min: 0.1, max: 1.5 })}</Field>
          <Field label="Threshold" htmlFor="adv-ext-thresh">{numInput(extensionThreshold, setExtensionThreshold, { id: "adv-ext-thresh", step: 0.01, min: 0, max: 1 })}</Field>
        </div>
      )}
    </>
  );
}

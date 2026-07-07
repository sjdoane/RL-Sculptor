import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Btn, Field, Modal, ToggleRow } from "@/components/rs/primitives";
import { useCreateMission } from "@/hooks/useMissions";
import { useProjectMetrics } from "@/hooks/useMetrics";
import { ApiError } from "@/lib/api";
import type { MetricSummary, MissionRunDefaults } from "@/lib/types";
import { SPEC_METRIC_NAMES } from "@/lib/types";

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
  // §MISSION_RUN_PARITY: per-stage trust-gated metric generation. Default
  // ON; best-of-N candidates sampled per stage (shown when the toggle is on).
  const [genStageMetrics, setGenStageMetrics] = useState(true);
  const [stageMetricCandidates, setStageMetricCandidates] = useState(1);

  const [iterations, setIterations] = useState<number | "">("");
  const [stepsPerIter, setStepsPerIter] = useState<number | "">("");
  const [seed, setSeed] = useState<number | "">("");
  const [earlyStopOnCriterion, setEarlyStopOnCriterion] = useState(false);
  const [stabilityWindow, setStabilityWindow] = useState<number | "">(1);
  const [extendOnImprovement, setExtendOnImprovement] = useState(false);
  const [maxExtensions, setMaxExtensions] = useState<number | "">(1);
  const [extensionFactor, setExtensionFactor] = useState<number | "">(0.5);
  const [extensionThreshold, setExtensionThreshold] = useState<number | "">(0.05);
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

  const create = useCreateMission(slug);

  const reset = () => {
    setGoal(""); setMissionSlug(""); setNoKg(false); setTab("basic");
    setGenStageMetrics(true); setStageMetricCandidates(1);
    setIterations(""); setStepsPerIter(""); setSeed("");
    setEarlyStopOnCriterion(false); setStabilityWindow(1);
    setExtendOnImprovement(false); setMaxExtensions(1);
    setExtensionFactor(0.5); setExtensionThreshold(0.05);
    setFitnessMetric(null); setFitnessMode("steer");
    setEditCandidates(""); setRolloutEpisodes(""); setMaxEpisodeSteps("");
    setPlaybackSpeed(""); setRenderSize("default"); setFitnessPatience("");
    setNumEnvs(""); setDevice("");
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
    if (fitnessMetric) {
      out.fitness_metric = fitnessMetric;
      out.fitness_mode = fitnessMode;
      // §MISSION_RUN_PARITY: patience only meaningful with a metric set.
      if (typeof fitnessPatience === "number") { out.fitness_patience = fitnessPatience; touched = true; }
      touched = true;
    }
    // §MISSION_RUN_PARITY: per-launch knobs (blank = inherited config).
    if (typeof editCandidates === "number") { out.edit_candidates = editCandidates; touched = true; }
    if (typeof rolloutEpisodes === "number") { out.rollout_episodes = rolloutEpisodes; touched = true; }
    if (typeof maxEpisodeSteps === "number") { out.max_episode_steps = maxEpisodeSteps; touched = true; }
    if (typeof playbackSpeed === "number") { out.playback_speed = playbackSpeed; touched = true; }
    if (renderSize !== "default") {
      out.render_width = Number(renderSize.split("x")[0]);
      out.render_height = Number(renderSize.split("x")[1]);
      touched = true;
    }
    if (typeof numEnvs === "number") { out.num_envs_override = numEnvs; touched = true; }
    if (device.trim()) { out.device_override = device.trim(); touched = true; }
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
      {
        goal: trimmedGoal,
        mission_slug: trimmedSlug || undefined,
        no_kg: noKg || undefined,
        // §MISSION_RUN_PARITY: per-stage metric generation. Send the toggle
        // always (so an explicit OFF sticks) + best-of-N only when on.
        gen_stage_metrics: genStageMetrics,
        stage_metric_candidates: genStageMetrics ? stageMetricCandidates : undefined,
        run_defaults: runDefaults ?? undefined,
      },
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
              {/* §MISSION_RUN_PARITY: the headline gap — per-stage metric
                  generation. Each stage gets a fresh objective metric from
                  its own goal, validated by the adversarial trust pipeline
                  before it can steer that stage; rejected → falls back to
                  the mission-level metric. */}
              <ToggleRow
                on={genStageMetrics} onChange={setGenStageMetrics}
                label="Auto-generate a trust-gated metric per stage"
                title="Per-stage objective metric"
                desc="Each stage gets a fresh metric generated from its own goal and validated by the adversarial pipeline before it can steer that stage. Rejected → the stage falls back to the mission-level metric."
              />
              {genStageMetrics && (
                <Field
                  label="Metric candidates per stage"
                  hint="best-of-N: sample N per stage, keep the most-discriminating valid one (each ~1-2 min)"
                  htmlFor="mission-stage-cand"
                >
                  <div className="rs-select">
                    <select
                      id="mission-stage-cand"
                      value={stageMetricCandidates}
                      onChange={(e) => setStageMetricCandidates(Number(e.target.value))}
                      disabled={create.isPending}
                      aria-label="Metric candidates per stage"
                    >
                      <option value={1}>1 candidate (fast)</option>
                      <option value={2}>best-of-2</option>
                      <option value={3}>best-of-3</option>
                      <option value={4}>best-of-4</option>
                    </select>
                  </div>
                </Field>
              )}
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
              fitnessMetric={fitnessMetric} setFitnessMetric={setFitnessMetric}
              fitnessMode={fitnessMode} setFitnessMode={setFitnessMode}
              metrics={projectMetrics.data ?? []}
              knobs={knobs}
            />
          )}
        </Modal>
      )}
    </>
  );
}

type NumOr = number | "";
export type MissionRenderSize = "default" | "1920x1080" | "960x540" | "320x240";

// §MISSION_RUN_PARITY: the per-launch knobs mirrored from NewRunDialog.
// Grouped in one optional bag so both callers can spread it in without
// each having to declare 9 more prop pairs when they don't use them
// (NewMissionDialog + RunMissionDialog both DO, but keeping them optional
// means an omitting caller degrades gracefully — the section hides).
export interface MissionParityKnobs {
  editCandidates: NumOr; setEditCandidates: (v: NumOr) => void;
  rolloutEpisodes: NumOr; setRolloutEpisodes: (v: NumOr) => void;
  maxEpisodeSteps: NumOr; setMaxEpisodeSteps: (v: NumOr) => void;
  playbackSpeed: NumOr; setPlaybackSpeed: (v: NumOr) => void;
  renderSize: MissionRenderSize; setRenderSize: (v: MissionRenderSize) => void;
  fitnessPatience: NumOr; setFitnessPatience: (v: NumOr) => void;
  numEnvs: NumOr; setNumEnvs: (v: NumOr) => void;
  device: string; setDevice: (v: string) => void;
}

// Shared Advanced form — also used by RunMissionDialog (same MissionRunDefaults
// fields). Exported so the two stay in lockstep.
export function MissionAdvanced({
  disabled,
  iterations, setIterations, stepsPerIter, setStepsPerIter, seed, setSeed,
  earlyStopOnCriterion, setEarlyStopOnCriterion, stabilityWindow, setStabilityWindow,
  extendOnImprovement, setExtendOnImprovement, maxExtensions, setMaxExtensions,
  extensionFactor, setExtensionFactor, extensionThreshold, setExtensionThreshold,
  fitnessMetric, setFitnessMetric, fitnessMode, setFitnessMode,
  metrics = [],
  showIterationsHint,
  knobs,
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
  fitnessMetric: string | null; setFitnessMetric: (v: string | null) => void;
  fitnessMode: "observe" | "steer"; setFitnessMode: (v: "observe" | "steer") => void;
  metrics?: MetricSummary[];
  showIterationsHint?: string;
  // §MISSION_RUN_PARITY: when supplied, renders the "Rollout video + RL
  // knobs" section + num_envs/device + fitness-patience row so a mission
  // run reaches NewRunDialog parity. Omit to hide (back-compat).
  knobs?: MissionParityKnobs;
}) {
  // §Ship 35: generated metrics selectable here; an uncalibrated one is
  // observe-locked (force observe), mirroring NewRunDialog.
  const genMetrics = metrics.filter((m) => m.accepted);
  const selectedGen = fitnessMetric?.startsWith("gen:")
    ? genMetrics.find((m) => `gen:${m.id}` === fitnessMetric) ?? null
    : null;
  const steerLocked = selectedGen !== null && !selectedGen.calibrated;
  useEffect(() => {
    if (steerLocked && fitnessMode !== "observe") setFitnessMode("observe");
  }, [steerLocked, fitnessMode, setFitnessMode]);
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

      {/* §Ship 34/35: objective fitness — built-ins + generated (uniform across stages). */}
      <Field label="Objective fitness metric" hint="applied to every stage; built-ins steer, generated observe until calibrated" htmlFor="adv-fitness">
        <div className="rs-select">
          <select
            id="adv-fitness"
            value={fitnessMetric ?? "none"}
            onChange={(e) => { const v = e.target.value; setFitnessMetric(v === "none" ? null : v); }}
            disabled={disabled}
            aria-label="Objective fitness metric"
          >
            <option value="none">none (blind loop)</option>
            <optgroup label="built-in (ground truth)">
              {SPEC_METRIC_NAMES.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </optgroup>
            {genMetrics.length > 0 && (
              <optgroup label="auto-generated for this project">
                {genMetrics.map((m) => (
                  <option key={m.id} value={`gen:${m.id}`}>
                    {m.id}{m.calibrated ? " ✓ calibrated" : " (observe-only)"}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
        </div>
      </Field>

      {/* §Ship 35: observe vs steer — only when a metric is chosen. */}
      {fitnessMetric !== null && (
        <Field
          label="Fitness mode"
          hint={steerLocked
            ? "generated metric is observe-only until calibrated"
            : "observe = compute & chart only, no influence"}
          htmlFor="adv-fitness-mode"
        >
          <div className="rs-select">
            <select
              id="adv-fitness-mode"
              value={steerLocked ? "observe" : fitnessMode}
              onChange={(e) => setFitnessMode(e.target.value === "observe" ? "observe" : "steer")}
              disabled={disabled || steerLocked}
              aria-label="Fitness mode"
            >
              <option value="steer">steer (drives selection)</option>
              <option value="observe">observe (display only)</option>
            </select>
          </div>
        </Field>
      )}

      {/* §MISSION_RUN_PARITY: fitness-plateau patience — the LIVE early
          stop on a steered stage (mirrors NewRunDialog). Only with a
          metric chosen + knobs supplied. Blank → sculpt default (2). */}
      {knobs && fitnessMetric !== null && (
        <Field
          label="Fitness patience"
          hint="rounds with no new best fitness before a stage stops (sculpt default 2; 4+ gives a hard skill room to escape a local optimum)"
          htmlFor="adv-fitness-patience"
        >
          {numInput(knobs.fitnessPatience, knobs.setFitnessPatience, {
            id: "adv-fitness-patience", min: 1, max: 50, step: 1, placeholder: "2",
          })}
        </Field>
      )}

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

      {/* §MISSION_RUN_PARITY: rollout-video + RL knobs + num_envs/device,
          applied uniformly to every stage. Mirrors NewRunDialog's
          Advanced tab so a mission run is UI-reachable end-to-end. Blank
          = the stage's inherited config default. */}
      {knobs && (
        <>
          <div style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", padding: 13 }}>
            <p style={{ margin: "0 0 11px", fontSize: 10.5, fontWeight: 600, letterSpacing: 0.6, textTransform: "uppercase", color: "var(--rs-muted)" }}>
              Rollout video + RL knobs · every stage
            </p>
            <div className="rs-row2">
              <Field label="Edit candidates" htmlFor="adv-editcand">
                {numInput(knobs.editCandidates, knobs.setEditCandidates, { id: "adv-editcand", min: 1, max: 5, placeholder: "1" })}
                <p className="rs-hintline">Best-of-K reward rewrites per diagnosis; only the winner trains.</p>
              </Field>
              <Field label="Episode steps" htmlFor="adv-epsteps">
                {numInput(knobs.maxEpisodeSteps, knobs.setMaxEpisodeSteps, { id: "adv-epsteps", min: 50, max: 5000, placeholder: "500" })}
                <p className="rs-hintline">Env steps per rollout episode. Longer = longer video.</p>
              </Field>
              <Field label="Playback speed" htmlFor="adv-playback">
                {numInput(knobs.playbackSpeed, knobs.setPlaybackSpeed, { id: "adv-playback", step: 0.1, min: 0.1, max: 10, placeholder: "1.0 (real-time)" })}
                <p className="rs-hintline">1.0 = real-time · 0.5 = slow-mo · 2.0 = 2×.</p>
              </Field>
              <Field label="Rollout episodes" htmlFor="adv-rolleps">
                {numInput(knobs.rolloutEpisodes, knobs.setRolloutEpisodes, { id: "adv-rolleps", min: 1, max: 32, placeholder: "6" })}
                <p className="rs-hintline">Episodes captured per round for behavior metrics.</p>
              </Field>
              <Field label="Video resolution" htmlFor="adv-resolution">
                <div className="rs-select">
                  <select
                    id="adv-resolution"
                    value={knobs.renderSize}
                    onChange={(e) => knobs.setRenderSize(e.target.value as MissionRenderSize)}
                    disabled={disabled}
                    aria-label="Rollout video resolution"
                  >
                    <option value="default">1280×720 (default)</option>
                    <option value="1920x1080">1920×1080</option>
                    <option value="960x540">960×540</option>
                    <option value="320x240">320×240 (legacy)</option>
                  </select>
                </div>
                <p className="rs-hintline">Render cost is resolution-independent — high-res is free.</p>
              </Field>
            </div>
          </div>

          <div className="rs-row2">
            <Field label="num_envs (override)" htmlFor="adv-numenvs">
              {numInput(knobs.numEnvs, knobs.setNumEnvs, { id: "adv-numenvs", min: 1, max: 8192, placeholder: "inherited" })}
              <p className="rs-hintline">mjlab only. Drop if a stage OOMs; halve + snap to power-of-two.</p>
            </Field>
            <Field label="device (override)" htmlFor="adv-device">
              <input
                id="adv-device"
                className="rs-input mono"
                value={knobs.device}
                onChange={(e) => knobs.setDevice(e.target.value)}
                disabled={disabled}
                placeholder="inherited (e.g. cuda:0)"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
              <p className="rs-hintline"><code className="mono">cpu</code> forces CPU (very slow for mjlab).</p>
            </Field>
          </div>
        </>
      )}
    </>
  );
}

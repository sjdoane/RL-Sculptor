import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { ReferencePickerDialog } from "@/components/ReferencePickerDialog";
import { Btn, Field, Modal, ToggleRow } from "@/components/rs/primitives";
import { courseBreakdownText } from "@/components/WorldTab";
import { useBehaviorDraft, useSaveBehaviorDraft } from "@/hooks/useBehaviorDraft";
import { referenceRobotForProject } from "@/lib/referenceRobot";
import { useSystemInfo } from "@/hooks/useDashboard";
import { useSystemGpu } from "@/hooks/useLibrary";
import { useCalibrateMetric, useGenerateMetric, useMetricGenProgress, useProjectMetrics } from "@/hooks/useMetrics";
import { useLaunchRun } from "@/hooks/useRuns";
import { useWorldSelection, useWorldValidation } from "@/hooks/useWorlds";
import { ApiError, getProjectSettings } from "@/lib/api";
import type { ProjectDetail } from "@/lib/types";
import { SPEC_METRIC_NAMES } from "@/lib/types";


// ── Per-adapter wall-clock model (S8 / §7.7) ─────────────────────────
//
// Used only for an honest pre-launch expectation, never as a gate.  mjlab
// training is modeled from the controls that dominate runtime: PPO iterations
// and parallel environment count, plus fixed rollout/LLM overhead.  Go1 is
// calibrated from the 2026-07-20 authored-world showcase on an RTX 5070
// Laptop: 750 iters × 1024 envs took ~34 min/cycle on the flat baseline, plus
// ~3 min for the final fresh best-policy evaluation. Authored legged-robot
// worlds can be materially slower because scene objects and perceptive sensors
// add work to every vectorized step. The 2026-07-22 G1 slalom (four objects,
// five regions, height scan) ran at roughly 4.5× the flat-world calibration.
const LEGACY_SECONDS_PER_CYCLE: Record<string, number> = {
  gym_sb3: 180,
};

const MAX_BEHAVIOR_GOAL_LENGTH = 500;


const MJLAB_TIMING: Record<string, {
  fixedSeconds: number;
  secondsPerTrainingIter: number;
  referenceEnvs: number;
  finalEvalSeconds: number;
}> = {
  mjlab_cartpole: {
    fixedSeconds: 20, secondsPerTrainingIter: 0.8,
    referenceEnvs: 256, finalEvalSeconds: 20,
  },
  mjlab_go1: {
    fixedSeconds: 450, secondsPerTrainingIter: 2.15,
    referenceEnvs: 1024, finalEvalSeconds: 180,
  },
  mjlab_g1: {
    fixedSeconds: 480, secondsPerTrainingIter: 2.6,
    referenceEnvs: 1024, finalEvalSeconds: 210,
  },
  mjlab_other: {
    fixedSeconds: 300, secondsPerTrainingIter: 1.5,
    referenceEnvs: 1024, finalEvalSeconds: 150,
  },
};

function humanizeSeconds(sec: number): string {
  if (sec < 90) return `${Math.round(sec)} s`;
  const min = sec / 60;
  if (min < 90) return `${min.toFixed(min < 5 ? 1 : 0)} min`;
  const hr = min / 60;
  return `${hr.toFixed(hr < 10 ? 1 : 0)} h`;
}

function formatEta(seconds: number): { label: string; finishAt: string } {
  const finish = new Date(Date.now() + seconds * 1000);
  const sameDay = finish.toDateString() === new Date().toDateString();
  const finishStr = sameDay
    ? finish.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : finish.toLocaleString([], {
        weekday: "short",
        hour: "2-digit",
        minute: "2-digit",
      });
  return { label: humanizeSeconds(seconds), finishAt: finishStr };
}

function ReadinessCell({
  icon, label, state, detail,
}: {
  icon: string;
  label: string;
  state: "ready" | "checking" | "attention" | "blocked";
  detail: string;
}) {
  const color = state === "ready"
    ? "var(--st-emerald)"
    : state === "blocked"
      ? "var(--st-rose)"
      : state === "attention"
        ? "var(--st-amber)"
        : "var(--rs-muted)";
  return (
    <div style={{ padding: "10px 11px", background: "var(--surface-strong)", minWidth: 0 }}>
      <div style={{ display: "flex", gap: 6, alignItems: "center", color, fontSize: 11.5, fontWeight: 650 }}>
        <Icon name={state === "ready" ? "check-circle" : state === "checking" ? "loader" : icon} size={13} className={state === "checking" ? "rs-spin" : undefined} />
        <span>{label}</span>
      </div>
      <div style={{ marginTop: 3, fontSize: 10.5, lineHeight: 1.35, color: "var(--rs-muted)", overflow: "hidden", textOverflow: "ellipsis" }}>
        {detail}
      </div>
    </div>
  );
}


// ── Per-adapter defaults (M7 Phase 4) ─────────────────────────────────
//
// The "iterations" field always means OUTER sculpt iterations
// (train → diagnose → edit → commit cycles). "training_iterations" is
// the inner loop budget per cycle — meaning depends on adapter:
//   - gym_sb3: env steps (overrides config.toml's `steps_per_iter`).
//   - mjlab:   rsl_rl policy-update iterations (becomes `max_iterations`).
//
// Defaults are sized so a full sculpt run fits in the time/compute
// envelope Sam documented in the M7 plan:
//   - gym_sb3:       20 outer × 50k steps    = ~3-5 min/cycle CPU
//   - mjlab cartpole: 15 outer × 500 iters   = ~30 s/cycle GPU
//   - mjlab Go1:     12 outer × 1000 iters   = ~90 s/cycle GPU
//   - mjlab G1:       8 outer × 1500 iters   = ~3 min/cycle GPU
// `String(undefined)` is the string "undefined", which is truthy, so the
// `|| "cuda:0"` fallbacks below never fired for an mjlab project whose
// adapter_config has no explicit device. The dialog then prefilled the literal
// "undefined" and the launch 422'd on an unparseable device.
function adapterDevice(project: ProjectDetail): string {
  const d = project.adapter_config?.device;
  return typeof d === "string" && d.trim() ? d.trim() : "";
}

function pickAdapterDefaults(project: ProjectDetail): {
  kind: "gym_sb3" | "mjlab_cartpole" | "mjlab_go1" | "mjlab_g1" | "mjlab_other";
  iterations: number;
  training_iterations: number;
  num_envs: number;
  device: string;
  training_label: string;
} {
  const cls = project.adapter_class || "";
  if (cls.includes("gym_sb3")) {
    return {
      kind: "gym_sb3",
      iterations: 20,
      training_iterations: 50_000,
      num_envs: 4,
      device: "cpu",
      training_label: "training steps / cycle",
    };
  }
  const taskId =
    typeof project.adapter_config?.task_id === "string"
      ? (project.adapter_config.task_id as string)
      : "";
  if (cls.includes("mjlab") && /Cartpole/i.test(taskId)) {
    return {
      kind: "mjlab_cartpole",
      iterations: 15,
      training_iterations: 500,
      num_envs: Number(project.adapter_config?.num_envs) || 1024,
      device: adapterDevice(project) || "cuda:0",
      training_label: "rsl_rl iters / cycle",
    };
  }
  if (cls.includes("mjlab") && /(Go1|Go2)/i.test(taskId)) {
    return {
      kind: "mjlab_go1",
      iterations: 12,
      training_iterations: 1000,
      num_envs: Number(project.adapter_config?.num_envs) || 2048,
      device: adapterDevice(project) || "cuda:0",
      training_label: "rsl_rl iters / cycle",
    };
  }
  if (cls.includes("mjlab") && /G1/i.test(taskId)) {
    return {
      kind: "mjlab_g1",
      iterations: 8,
      training_iterations: 1500,
      num_envs: Number(project.adapter_config?.num_envs) || 2048,
      device: adapterDevice(project) || "cuda:0",
      training_label: "rsl_rl iters / cycle",
    };
  }
  // Any other mjlab task (T1, ANYmal, custom).
  return {
    kind: "mjlab_other",
    iterations: 10,
    training_iterations: 1000,
    num_envs: Number(project.adapter_config?.num_envs) || 1024,
    device: adapterDevice(project) || "cuda:0",
    training_label: "rsl_rl iters / cycle",
  };
}

type RunProfile = "custom" | "pipeline" | "rehearsal" | "overnight";

function profileBudget(
  profile: Exclude<RunProfile, "custom">,
  defaults: ReturnType<typeof pickAdapterDefaults>,
) {
  const gym = defaults.kind === "gym_sb3";
  const cartpole = defaults.kind === "mjlab_cartpole";
  if (profile === "pipeline") {
    return {
      iterations: 1,
      trainingIters: gym ? 1_000 : 100,
      numEnvs: gym ? defaults.num_envs : Math.min(defaults.num_envs, 256),
      dryRun: true,
      interactive: false,
      maxEpisodeSteps: 180,
      rolloutEpisodes: 1,
      renderSize: "960x540" as const,
    };
  }
  if (profile === "rehearsal") {
    return {
      iterations: 2,
      trainingIters: gym ? 15_000 : cartpole ? 250 : 350,
      numEnvs: gym ? defaults.num_envs : Math.min(defaults.num_envs, 512),
      dryRun: false,
      interactive: true,
      maxEpisodeSteps: 300,
      rolloutEpisodes: 2,
      renderSize: "960x540" as const,
    };
  }
  return {
    iterations: gym ? 6 : 4,
    trainingIters: gym ? 40_000 : cartpole ? 500 : 750,
    numEnvs: gym ? defaults.num_envs : Math.min(defaults.num_envs, 1024),
    dryRun: false,
    interactive: false,
    maxEpisodeSteps: 500,
    rolloutEpisodes: 2,
    renderSize: "960x540" as const,
  };
}


export function NewRunDialog({
  slug,
  project,
  onLaunched,
  onOpenWorld,
  triggerLabel = "New run",
  triggerKind = "primary",
}: {
  slug: string;
  project: ProjectDetail;
  onLaunched: (runId: string) => void;
  onOpenWorld?: () => void;
  triggerLabel?: string;
  triggerKind?: "primary" | "quiet" | "ghost";
}) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"basic" | "advanced">("basic");
  const defaults = useMemo(() => pickAdapterDefaults(project), [project]);
  const isMjlab = defaults.kind.startsWith("mjlab");

  const [behavior, setBehavior] = useState("");
  const [profile, setProfile] = useState<RunProfile>("custom");
  type RenderSize = "default" | "1920x1080" | "960x540" | "320x240";
  const [iterations, setIterations] = useState(defaults.iterations);
  const [trainingIters, setTrainingIters] = useState<number | "">(
    defaults.training_iterations,
  );
  const [numEnvs, setNumEnvs] = useState<number | "">(defaults.num_envs);
  const [device, setDevice] = useState<string>(defaults.device);
  const [noKg, setNoKg] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  // §Ship 39 (H1): pause-for-feedback by default (the requested default);
  // flippable to Auto at any time from the run header once it's running.
  const [interactive, setInteractive] = useState(true);
  // Recovery-only: reject mutable diagnosis drafts and restore the reward/env
  // refs from selection_current.json before the subprocess starts.
  const [resumeExactTuple, setResumeExactTuple] = useState(false);
  // Policy-only recovery: explicitly select a completed iteration checkpoint
  // without changing reward/environment inputs or objective-metric authority.
  const [warmStartIteration, setWarmStartIteration] = useState<number | "">("");
  const [allowDefaultWorld, setAllowDefaultWorld] = useState(false);
  const [allowRobotMismatch, setAllowRobotMismatch] = useState(false);
  const [validatingLaunch, setValidatingLaunch] = useState(false);
  // §Ship-7: rollout-video + RL knobs. Empty string = "leave blank →
  // use runner/config default".
  const [maxEpisodeSteps, setMaxEpisodeSteps] = useState<number | "">("");
  const [playbackSpeed, setPlaybackSpeed] = useState<number | "">("");
  const [rolloutEpisodes, setRolloutEpisodes] = useState<number | "">("");
  const [seed, setSeed] = useState<number | "">("");
  const [renderEnvIndex, setRenderEnvIndex] = useState<number | "">(0);
  // Rollout video resolution. "default" = runner default (1280×720);
  // render cost is resolution-independent so this is mostly a debug knob.
  const [renderSize, setRenderSize] = useState<RenderSize>("default");
  const [autoAdjustPhysics, setAutoAdjustPhysics] = useState<boolean | null>(null);
  // §Ship 34/35: objective fitness. null = blind loop. A built-in spec
  // name OR a generated-metric ref ("gen:<id>").
  const [fitnessMetric, setFitnessMetric] = useState<string | null>(null);
  const [fitnessMode, setFitnessMode] = useState<"observe" | "steer">("steer");
  // §Ship 48: patience for the fitness-plateau early stop (the LIVE early
  // stop on a steered run; the sculpt-lib default is 2, which truncated the
  // g1-kick-v3 run at iter 4). Default 4 here so a hard exploratory skill
  // gets room to escape a local optimum; "" → sculpt default.
  const [fitnessPatience, setFitnessPatience] = useState<number | "">(4);
  const [referenceClipId, setReferenceClipId] = useState<string | null>(null);
  const [referencePickerOpen, setReferencePickerOpen] = useState(false);
  const referenceRobot = useMemo(
    () => referenceRobotForProject(project),
    [project],
  );
  const launch = useLaunchRun(slug);
  // §env-authoring: the authored world this run will train under (404 →
  // undefined for legacy projects; the atomic selection file drives the
  // adapter, so this is disclosure, not configuration).
  const worldSel = useWorldSelection(open ? slug : undefined);
  const worldValidation = useWorldValidation(
    open ? slug : undefined,
    !!worldSel.data,
  );
  // What this project is building, shared with compose and per-mode reward.
  const draft = useBehaviorDraft(open ? slug : undefined);
  // The persistent `[iteration]` defaults this dialog's Advanced tab
  // overrides. Disclosed there so the user can see which value wins.
  const projectSettings = useQuery({
    queryKey: ["project-settings", slug],
    queryFn: () => getProjectSettings(slug),
    enabled: open && tab === "advanced",
    staleTime: 30_000,
  });
  const saveDraft = useSaveBehaviorDraft(slug);
  // A tracking-first run binds its generated reward to the authored world
  // atomically, so `sculpt` refuses to start without a promoted selection.
  // Block at the button rather than letting the run die in the subprocess.
  const referenceNeedsWorld =
    !!referenceClipId && worldSel.isFetched && !worldSel.data?.selection;
  const systemInfo = useSystemInfo();
  const gpuInfo = useSystemGpu();
  // §Ship 35: per-project generated metrics + the generate action.
  const projectMetrics = useProjectMetrics(slug, open);
  const genMetric = useGenerateMetric(slug);
  const genProgress = useMetricGenProgress(slug, genMetric.isPending);
  const calibrate = useCalibrateMetric(slug);
  const [calibrateAgainst, setCalibrateAgainst] = useState<string>("go1_trot");
  // §best-of-N: how many candidate metrics to sample + keep the most-discriminating
  // valid one (1 = single-shot-with-retry). Applies to BOTH the Generate button and
  // the generate-at-launch path. Each candidate is a ~1-2 min LLM call.
  const [metricCandidates, setMetricCandidates] = useState<number>(1);
  const genMetrics = (projectMetrics.data ?? []).filter((m) => m.accepted);
  const selectedGen = fitnessMetric?.startsWith("gen:")
    ? genMetrics.find((m) => `gen:${m.id}` === fitnessMetric) ?? null
    : null;
  // An accepted-but-uncalibrated generated metric may ONLY observe.
  const steerLocked = selectedGen !== null && !selectedGen.calibrated;
  // §Ship 42: deferred generation — generate the metric AT LAUNCH (run-phase 0)
  // instead of picking an existing one. Observe-only by definition (uncalibrated
  // until launch-time calibration, Ship 44).
  const isLaunchGen = fitnessMetric === "generate-at-launch";

  // S8 / §7.7 — ETA estimate + resume-warning banner. Pure view-layer
  // logic; no new API call. Uses the documented per-cycle budget for
  // the active adapter kind. When dry-run is checked, override to 50 s
  // total (the comment on the --dry-run checkbox promises "~50 s total").
  const etaSeconds = useMemo(() => {
    if (dryRun) return 50;
    const timing = MJLAB_TIMING[defaults.kind];
    if (isMjlab && timing) {
      const innerIters = typeof trainingIters === "number"
        ? trainingIters
        : defaults.training_iterations;
      const envs = typeof numEnvs === "number" ? numEnvs : defaults.num_envs;
      // More parallel envs increase simulator work, but GPU batching keeps the
      // scaling sublinear. Bound the ratio so a malformed transient UI value
      // cannot produce a zero/negative estimate.
      const envRatio = Math.max(0.125, envs / timing.referenceEnvs);
      const envScale = Math.pow(envRatio, 0.35);
      // Keep the flat-task calibration for projects without an authored world,
      // but price in the measured cost of authored scenes for legged robots.
      // This is deliberately conservative: an honest overestimate is safer than
      // telling a user a four-cycle overnight run will finish after one cycle.
      const authoredWorldScale = worldSel.data
        && defaults.kind !== "mjlab_cartpole"
        ? 4
        : 1;
      const perCycle = timing.fixedSeconds
        + timing.secondsPerTrainingIter * innerIters * envScale * authoredWorldScale;
      const finalBestEval = fitnessMetric ? timing.finalEvalSeconds : 0;
      return perCycle * iterations + finalBestEval;
    }

    const perCycle = LEGACY_SECONDS_PER_CYCLE[defaults.kind] ?? 300;
    const innerScale = typeof trainingIters === "number"
      ? trainingIters / defaults.training_iterations
      : 1;
    return perCycle * iterations * Math.max(0.1, innerScale);
  }, [
    dryRun, defaults.kind, defaults.num_envs,
    defaults.training_iterations, fitnessMetric, isMjlab, iterations,
    numEnvs, trainingIters, worldSel.data,
  ]);
  const eta = useMemo(() => formatEta(etaSeconds), [etaSeconds]);
  const isLongRun = etaSeconds >= 30 * 60; // 30 min
  const hasPriorIters = project.n_iterations_completed > 0;

  // Pre-fill behavior from the project's behavior draft (falling back to the
  // sidecar description); reset advanced fields when the adapter changes
  // (defaults drift with the project).
  //
  // The draft is why the goal and the motion prior survive between launches.
  // Before it, both were cleared on every open — so the documented flow was
  // literally "re-paste the goal and re-attach the motion prior; the dialog
  // remembers neither", including immediately after composing the clip in a
  // dialog two clicks away.
  useEffect(() => {
    if (open) {
      setBehavior((current) =>
        current || draft.data?.behavior_goal || project.description || "");
      setIterations(defaults.iterations);
      setTrainingIters(defaults.training_iterations);
      setNumEnvs(defaults.num_envs);
      setDevice(defaults.device);
      setProfile("custom");
      setResumeExactTuple(false);
      setWarmStartIteration("");
      setAllowDefaultWorld(false);
      setAllowRobotMismatch(false);
      // Only carry a draft clip that belongs to this project's embodiment;
      // a clip from another robot's namespace would fail at launch.
      setReferenceClipId(
        draft.data?.reference_clip_id
        && (!draft.data.reference_robot
            || draft.data.reference_robot === referenceRobot)
          ? draft.data.reference_clip_id
          : null,
      );
      setReferencePickerOpen(false);
      setRenderEnvIndex(0);
    }
  }, [open, project.description, defaults, draft.data, referenceRobot]);

  const applyProfile = (next: Exclude<RunProfile, "custom">) => {
    const budget = profileBudget(next, defaults);
    setProfile(next);
    setIterations(budget.iterations);
    setTrainingIters(budget.trainingIters);
    setNumEnvs(budget.numEnvs);
    setDryRun(budget.dryRun);
    setInteractive(budget.interactive);
    setMaxEpisodeSteps(budget.maxEpisodeSteps);
    setRolloutEpisodes(budget.rolloutEpisodes);
    setRenderSize(budget.renderSize);
    setSeed(42);
  };

  // Single entry point for the Generate button AND the inline retry below,
  // so the error affordance re-runs exactly what failed. The api call carries
  // a client-side timeout scaled by n_candidates, so isPending ALWAYS
  // resolves — no more frozen "Generating…" spinner on a dropped connection.
  const runGenerate = () => {
    if (behavior.trim().length > MAX_BEHAVIOR_GOAL_LENGTH) {
      toast.error("Behavior goal too long", {
        description: `Use at most ${MAX_BEHAVIOR_GOAL_LENGTH} characters.`,
      });
      return;
    }
    genMetric.mutate(
      { behavior_goal: behavior.trim(), n_candidates: metricCandidates },
      {
        onSuccess: (m) => {
          if (m.accepted) {
            setFitnessMetric(`gen:${m.id}`);
            setFitnessMode("observe");
            const pick = (m.n_candidates && m.n_candidates > 1
              && m.selected_candidate != null)
              ? ` — picked candidate ${m.selected_candidate + 1}/${m.n_candidates}`
              : "";
            toast.success(`Generated ${m.id} (observe-only until calibrated)${pick}`, {
              description: m.validator_basis
                ? `Validated from ${m.validator_basis.replaceAll("+", " + ")}.`
                : m.review?.summary ?? "Validated + reviewed.",
            });
          } else {
            const why = (m.reasons && m.reasons[0])
              || (m.review?.concerns && m.review.concerns[0])
              || "validation/review rejected it";
            toast.error("Generated metric rejected", { description: why });
          }
        },
        onError: (err) => {
          const d = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
          toast.error("Could not generate metric", { description: d });
        },
      },
    );
  };

  const submit = async () => {
    if (behavior.trim().length < 4) {
      toast.error("Behavior goal too short", { description: "At least 4 chars." });
      return;
    }
    if (behavior.trim().length > MAX_BEHAVIOR_GOAL_LENGTH) {
      toast.error("Behavior goal too long", {
        description: `Use at most ${MAX_BEHAVIOR_GOAL_LENGTH} characters.`,
      });
      return;
    }
    if (!dryRun && systemInfo.data?.anthropic_api_key_set === false) {
      toast.error("Live-run API key is not configured", {
        description: "Open Settings → Anthropic API, save a key, then launch again.",
      });
      return;
    }
    if (!worldSel.data && !allowDefaultWorld) {
      toast.error("Choose the training world", {
        description: "Author a world, or explicitly confirm the project default scene.",
      });
      return;
    }
    if (
      worldSel.data?.shared_summary.robot_matches_project === false
      && !allowRobotMismatch
    ) {
      toast.error("Authored-world robot differs from the project", {
        description: "Re-author for the project robot, or explicitly confirm the mismatch.",
      });
      return;
    }
    if (worldSel.data) {
      setValidatingLaunch(true);
      const checked = await worldValidation.refetch();
      setValidatingLaunch(false);
      if (checked.error || !checked.data?.ok) {
        toast.error("Authored world failed launch verification", {
          description: checked.data?.errors.join("; ")
            || checked.error?.message
            || "The authoritative tuple could not be verified.",
        });
        return;
      }
    }
    const body = {
      behavior_goal: behavior.trim(),
      iterations,
      no_kg: noKg,
      dry_run: dryRun,
      training_iterations:
        typeof trainingIters === "number" ? trainingIters : null,
      num_envs_override:
        isMjlab && typeof numEnvs === "number" ? numEnvs : null,
      device_override: isMjlab ? device : null,
      // §Ship-7: only forward when set; null = runner / config default.
      max_episode_steps:
        typeof maxEpisodeSteps === "number" ? maxEpisodeSteps : null,
      playback_speed:
        typeof playbackSpeed === "number" ? playbackSpeed : null,
      rollout_episodes:
        typeof rolloutEpisodes === "number" ? rolloutEpisodes : null,
      render_width:
        renderSize === "default" ? null : Number(renderSize.split("x")[0]),
      render_height:
        renderSize === "default" ? null : Number(renderSize.split("x")[1]),
      render_env_index:
        typeof renderEnvIndex === "number" ? renderEnvIndex : null,
      seed: typeof seed === "number" ? seed : null,
      auto_adjust_physics: autoAdjustPhysics,
      // §Ship 34: null = blind loop; a spec name turns on fitness-guided
      // selection + plateau early-stop for this run.
      fitness_metric: fitnessMetric,
      // §Ship 35: observe vs steer (ignored when no metric). An
      // uncalibrated generated metric is forced to observe; §Ship 42: a
      // launch-time-generated metric is uncalibrated by definition.
      fitness_mode: steerLocked || isLaunchGen ? "observe" : fitnessMode,
      // §best-of-N: candidates for a generate-at-launch metric (1 = single-shot).
      // Only meaningful on the launch-gen path; harmless otherwise.
      metric_n_candidates: isLaunchGen ? metricCandidates : 1,
      // §Ship 48: fitness-plateau patience (only with a metric; the live
      // early stop). "" → sculpt default (2).
      fitness_patience:
        fitnessMetric !== null && typeof fitnessPatience === "number"
          ? fitnessPatience
          : null,
      // §Ship 39 (H1): manual = pause for human feedback each iteration.
      start_mode: interactive ? ("manual" as const) : ("auto" as const),
      // Explicitly opt in: ordinary iterative resumes must continue to consume
      // their newly generated drafts, while recovery can reject them in favor
      // of the last promoted atomic tuple.
      resume_exact_tuple: resumeExactTuple,
      warm_start_iteration:
        typeof warmStartIteration === "number" ? warmStartIteration : null,
      reference_clip_id: referenceClipId,
      reference_robot: referenceClipId ? referenceRobot : null,
    };
    launch.mutate(body, {
      onSuccess: (r) => {
        // Remember what was actually launched, so the next open starts from
        // it instead of from a blank box.
        saveDraft.mutate({
          behavior_goal: behavior.trim() || null,
          reference_clip_id: referenceClipId,
          reference_robot: referenceClipId ? referenceRobot : null,
        });
        setOpen(false);
        onLaunched(r.run_id);
        toast.success("Sculpt run launched", {
          description: `run_id: ${r.run_id}`,
        });
      },
      onError: (err) => {
        const detail = err instanceof ApiError
          ? err.problem.detail ?? err.problem.title
          : err.message;
        toast.error("Could not launch run", { description: detail });
      },
    });
  };

  const numField = (
    v: number | "",
    set: (x: number | "") => void,
    props: React.InputHTMLAttributes<HTMLInputElement>,
  ) => (
    <input
      {...props}
      type="number"
      className="rs-input mono"
      value={v}
      onChange={(e) => {
        set(e.target.value === "" ? "" : Number(e.target.value));
        setProfile("custom");
      }}
      disabled={launch.isPending}
    />
  );

  const launchBusy = launch.isPending || validatingLaunch;
  const apiReady = dryRun || systemInfo.data?.anthropic_api_key_set === true;
  const gpuReady = !isMjlab || (
    gpuInfo.data?.cuda_available === true
    && gpuInfo.data?.mjlab_available === true
    && gpuInfo.data?.rsl_rl_available === true
  );
  const worldReady = !!worldSel.data && worldValidation.data?.ok === true;

  return (
    <>
      {/* Reset any stale generate error from a previous open — the mutation
          outlives the Modal (it's conditionally rendered below). */}
      <Btn kind={triggerKind} size="sm" icon="play" onClick={() => { genMetric.reset(); setOpen(true); }}>{triggerLabel}</Btn>
      {open && (
        <Modal
          title="Launch a sculpt run"
          subtitle={
            isMjlab ? (
              <>GPU adapter (<code className="mono">{defaults.kind}</code>) · defaults sized for RTX 5070 Laptop (8 GiB VRAM)</>
            ) : (
              <>CPU adapter (<code className="mono">gym_sb3</code>) · advanced has steps_per_iter + KG toggles</>
            )
          }
          icon="play"
          onClose={() => { if (!launchBusy) setOpen(false); }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => setOpen(false)} disabled={launchBusy}>Cancel</Btn>
              <Btn
                kind="primary"
                icon={launchBusy ? "loader" : "play"}
                onClick={() => void submit()}
                disabled={launchBusy || referenceNeedsWorld}
                title={referenceNeedsWorld
                  ? "A motion prior requires a promoted authored world"
                  : undefined}
              >
                {validatingLaunch ? "Verifying world…" : launch.isPending ? "Launching…" : "Launch"}
              </Btn>
            </>
          }
        >
          <div style={{ display: "grid", gap: 8 }}>
            <div className="rs-eyebrow">Run plan</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8 }}>
              {([
                ["pipeline", "Pipeline check", "~50 s · stubbed LLM · no GPU commitment"],
                ["rehearsal", "Live rehearsal", "2 real cycles · pauses for feedback"],
                ["overnight", "Overnight showcase", "4 real cycles · auto · call-ready evidence"],
              ] as const).map(([value, label, description]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => applyProfile(value)}
                  disabled={launchBusy}
                  style={{
                    border: profile === value
                      ? "1px solid var(--rs-primary)"
                      : "1px solid var(--hairline)",
                    borderRadius: "var(--radius-md)",
                    background: profile === value
                      ? "color-mix(in srgb, var(--rs-primary) 7%, var(--surface-strong))"
                      : "var(--surface-strong)",
                    padding: "10px 11px", textAlign: "left", cursor: "pointer",
                    color: "var(--ink)", font: "inherit",
                  }}
                >
                  <span style={{ display: "block", fontSize: 12.5, fontWeight: 650 }}>{label}</span>
                  <span style={{ display: "block", marginTop: 3, fontSize: 10.5, lineHeight: 1.4, color: "var(--rs-muted)" }}>{description}</span>
                </button>
              ))}
            </div>
            {profile === "custom" && (
              <div className="rs-hintline">Project defaults loaded. Pick a plan or tune Advanced directly.</div>
            )}
          </div>

          <div
            style={{
              display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
              gap: 1, overflow: "hidden", borderRadius: "var(--radius-md)",
              border: "1px solid var(--hairline)", background: "var(--hairline)",
            }}
          >
            <ReadinessCell
              icon="key" label="Live authoring"
              state={apiReady ? "ready" : systemInfo.isLoading ? "checking" : "blocked"}
              detail={dryRun ? "not needed for pipeline check" : apiReady ? "API key configured" : "save a key in Settings"}
            />
            <ReadinessCell
              icon="cpu" label="Training runtime"
              state={gpuReady ? "ready" : gpuInfo.isLoading ? "checking" : "blocked"}
              detail={!isMjlab ? "CPU adapter ready" : gpuReady ? `${device} · mjlab + rsl_rl ready` : "CUDA/mjlab preflight failed"}
            />
            <ReadinessCell
              icon="globe" label="Authored world"
              state={worldReady ? "ready" : worldValidation.isLoading ? "checking" : worldSel.data ? "blocked" : "attention"}
              detail={worldReady
                ? `v${worldSel.data!.selection.selection_version} · tuple verified`
                : worldSel.data ? "integrity verification required" : "project default unless authored"}
            />
          </div>
          {/* ETA estimate + resume-warning (S8 / §7.7) */}
          <div
            style={{
              display: "flex", flexDirection: "column", gap: 6,
              borderRadius: "var(--radius-md)", padding: "10px 13px", fontSize: 12,
              border: isLongRun ? "1px solid color-mix(in srgb, var(--st-amber) 40%, transparent)" : "1px solid var(--hairline)",
              background: isLongRun ? "var(--st-amber-bg)" : "var(--surface-strong)",
              color: isLongRun ? "var(--st-amber-fg)" : "var(--rs-muted)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 7, fontWeight: 600 }}>
              <Icon name="clock" size={14} />
              <span>
                Estimated wall-clock: <strong>{eta.label}</strong>
                {!dryRun && <> · finishes ~<strong>{eta.finishAt}</strong></>}
              </span>
            </div>
            {isLongRun && !dryRun && (
              <div style={{ fontSize: 11, lineHeight: 1.45 }}>
                Heads up — long run ahead. Laptop sleep, power loss, or a backend restart will leave
                completed iters on disk, and resume is on by default so rerunning picks up where it
                stopped. Consider --dry-run first to shake the pipeline.
              </div>
            )}
            {hasPriorIters && (
              <div style={{ display: "flex", alignItems: "flex-start", gap: 7, fontSize: 11, lineHeight: 1.45 }}>
                <span style={{ marginTop: 1, flexShrink: 0 }}><Icon name="skip-forward" size={12} /></span>
                <span>
                  Resume is enabled — this project has <strong>{project.n_iterations_completed}</strong> iter(s)
                  on disk; existing <code className="mono">runs/iter_&lt;N&gt;/</code> artifacts (checkpoint,
                  rollout, trajectory) are reused rather than retrained.
                </span>
              </div>
            )}
          </div>

          {/* §env-authoring: which world this run trains under. Only shown
              when the project has a promoted authored selection — legacy
              projects train the config.toml task on the default scene. */}
          {worldSel.data && (
            <div
              style={{
                display: "flex", flexDirection: "column", gap: 4,
                borderRadius: "var(--radius-md)", padding: "10px 13px", fontSize: 12,
                border: "1px solid var(--hairline)", background: "var(--surface-strong)",
                color: "var(--rs-muted)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 7, fontWeight: 600 }}>
                <Icon name="globe" size={14} />
                <span>
                  Training world: authored selection{" "}
                  <strong>v{worldSel.data.selection.selection_version}</strong>
                  {" · "}{worldSel.data.shared_summary.robot ?? "—"}
                  {" · "}{worldSel.data.shared_summary.terrain_kind ?? "plane"}
                  {" · "}
                  {courseBreakdownText(
                    worldSel.data.shared_summary.course_breakdown,
                    worldSel.data.shared_summary.course_elements,
                  )}
                </span>
              </div>
              {worldSel.data.shared_summary.robot_matches_project === false && (
                <div style={{ fontSize: 11, lineHeight: 1.45, color: "var(--st-amber-fg)" }}>
                  <div>
                    This world&apos;s robot differs from the project robot
                    ({worldSel.data.shared_summary.project_capability_id}) — the run
                    trains the authored world&apos;s robot.
                  </div>
                  <label style={{ display: "flex", gap: 7, alignItems: "center", marginTop: 6, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={allowRobotMismatch}
                      onChange={(event) => setAllowRobotMismatch(event.target.checked)}
                    />
                    I intend to train the authored world&apos;s robot.
                  </label>
                </div>
              )}
            </div>
          )}
          {!worldSel.isLoading && !worldSel.data && (
            <div className="rs-banner warn">
              <Icon name="globe" size={17} />
              <div className="rs-grow" style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 650 }}>No authored world is selected.</div>
                <div style={{ marginTop: 2, fontSize: 11, lineHeight: 1.45 }}>
                  Author terrain, objects, and task semantics in the World tab,
                  or explicitly continue with the project&apos;s built-in scene.
                </div>
                <label style={{ display: "flex", gap: 7, alignItems: "center", marginTop: 7, cursor: "pointer", fontSize: 11.5 }}>
                  <input
                    type="checkbox"
                    checked={allowDefaultWorld}
                    onChange={(event) => setAllowDefaultWorld(event.target.checked)}
                  />
                  Use the project default scene for this run.
                </label>
              </div>
              {onOpenWorld && (
                <Btn
                  kind="ghost" size="sm" icon="globe"
                  onClick={() => { setOpen(false); onOpenWorld(); }}
                >
                  Author world
                </Btn>
              )}
            </div>
          )}

          <div className="rs-mtabs">
            <button className={tab === "basic" ? "on" : ""} onClick={() => setTab("basic")}>Basic</button>
            <button className={tab === "advanced" ? "on" : ""} onClick={() => setTab("advanced")}>Advanced</button>
          </div>

          {tab === "basic" ? (
            <>
              <Field
                label="Behavior goal"
                htmlFor="run-goal"
                hint={`${behavior.length}/${MAX_BEHAVIOR_GOAL_LENGTH}`}
              >
                <textarea
                  id="run-goal"
                  className="rs-textarea"
                  value={behavior}
                  onChange={(e) => setBehavior(
                    e.target.value.slice(0, MAX_BEHAVIOR_GOAL_LENGTH),
                  )}
                  placeholder="Run forward as fast as possible without falling."
                  style={{ minHeight: 84 }}
                  maxLength={MAX_BEHAVIOR_GOAL_LENGTH}
                  disabled={launch.isPending}
                  autoFocus
                />
              </Field>
              <div
                style={{
                  display: "flex", alignItems: "center", gap: 11,
                  border: "1px solid var(--hairline)",
                  borderRadius: "var(--radius-md)",
                  padding: "11px 12px",
                  background: referenceClipId
                    ? "color-mix(in srgb, var(--rs-primary) 5%, var(--surface-strong))"
                    : "var(--surface-strong)",
                }}
              >
                <div
                  style={{
                    width: 30, height: 30, flex: "0 0 30px",
                    borderRadius: "var(--radius-sm)",
                    display: "grid", placeItems: "center",
                    background: "var(--canvas-soft)",
                    color: referenceClipId ? "var(--rs-primary)" : "var(--rs-muted)",
                  }}
                >
                  <Icon name="video" size={15} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 650 }}>
                    {referenceClipId ? "Motion prior attached" : "Pre-existing motion (optional)"}
                  </div>
                  <div
                    style={{
                      marginTop: 3, fontSize: 10.8, lineHeight: 1.45,
                      color: "var(--rs-muted)",
                    }}
                  >
                    {referenceClipId
                      ? <><code className="mono">{referenceRobot}/{referenceClipId}</code> becomes the immutable tracking base; this goal can only add a bounded task residual.</>
                      : "Choose a retargeted clip to preserve its gait or pose sequence while the authored world and route RSI teach the novel task."}
                  </div>
                  {/* A tracking-first run binds its generated reward to the
                      authored world atomically and refuses to start without a
                      promoted selection. That refusal used to happen inside
                      the subprocess, so the run "launched" and then died. */}
                  {referenceClipId && worldSel.isFetched && !worldSel.data?.selection && (
                    <div
                      style={{
                        marginTop: 6, fontSize: 10.8, lineHeight: 1.45,
                        color: "var(--st-amber)",
                      }}
                    >
                      This project has no promoted world, and a motion prior
                      requires one — the tracking reward is bound to the world
                      atomically. Author and promote a world first, or clear
                      the motion to launch without it.
                    </div>
                  )}
                </div>
                {referenceClipId && (
                  <Btn
                    kind="ghost" size="xs" icon="x"
                    onClick={() => setReferenceClipId(null)}
                    disabled={launchBusy}
                  >
                    Clear
                  </Btn>
                )}
                <Btn
                  kind={referenceClipId ? "quiet" : "ghost"}
                  size="sm"
                  icon="search"
                  onClick={() => setReferencePickerOpen(true)}
                  disabled={launchBusy}
                >
                  {referenceClipId ? "Change" : "Choose motion"}
                </Btn>
              </div>
              <ToggleRow
                on={dryRun} onChange={(value) => { setDryRun(value); setProfile("custom"); }} label="Dry run"
                title={<><code className="mono">--dry-run</code> · smoke-test the pipeline</>}
                desc="training steps capped at 1000, LLM calls stubbed — ~50 s total"
              />
              <ToggleRow
                on={interactive} onChange={(value) => { setInteractive(value); setProfile("custom"); }} label="Pause for my feedback each iteration"
                title={<>Manual mode · review the rollout, then steer the next iteration</>}
                desc="Default on. Flip to Auto at any point from the run header while it's running."
              />
            </>
          ) : (
            <>
              {/* The same knobs appear under three names in three places, and
                  nothing said which one wins. Settings (gear icon) edits the
                  persistent `[iteration]` block in config.toml; this tab
                  overrides it for one run only; project creation seeds it.
                  Show the stored value beside every field that has one so
                  "am I changing the default or just this run" is answerable
                  without leaving the dialog. */}
              <div
                style={{
                  border: "1px solid var(--hairline)",
                  borderRadius: "var(--radius-md)",
                  padding: "10px 12px",
                  background: "var(--surface-strong)",
                  fontSize: 11, lineHeight: 1.5,
                }}
              >
                <div className="rs-flex rs-gap-8" style={{ alignItems: "center" }}>
                  <Icon name="info" size={13} color="var(--rs-muted)" />
                  <strong style={{ fontSize: 11.5 }}>
                    These override the project defaults, for this run only
                  </strong>
                </div>
                <div className="rs-sub" style={{ marginTop: 4 }}>
                  Persistent defaults live in <code className="mono">config.toml</code>
                  {" "}under <code className="mono">[iteration]</code> and are edited
                  from the gear icon in the project header. A field left blank
                  here falls back to that value; a field with a number in it
                  wins for this launch and is not saved back.
                  {projectSettings.data && (
                    <>
                      {" "}Stored now:{" "}
                      <code className="mono">
                        steps_per_iter={String(
                          projectSettings.data.iteration?.steps_per_iter ?? "unset")}
                      </code>,{" "}
                      <code className="mono">
                        max_episode_steps={String(
                          projectSettings.data.iteration?.max_episode_steps ?? "unset")}
                      </code>,{" "}
                      <code className="mono">
                        rollout_episodes={String(
                          projectSettings.data.iteration?.rollout_episodes ?? "unset")}
                      </code>.
                    </>
                  )}
                </div>
                <div className="rs-sub" style={{ marginTop: 6 }}>
                  <strong>Note:</strong> the field this dialog calls
                  {" "}<code className="mono">training_iterations</code> is the same
                  flag Settings calls <code className="mono">steps_per_iter</code>.
                  Its unit depends on the adapter — rsl_rl policy-update
                  iterations on mjlab, environment steps on gym_sb3 — which is
                  why the two screens show different sensible ranges for it.
                </div>
              </div>
              {hasPriorIters && worldSel.data && (
                <ToggleRow
                  on={resumeExactTuple}
                  onChange={(value) => { setResumeExactTuple(value); setProfile("custom"); }}
                  label="Resume exact promoted tuple"
                  title={<>Recovery mode · restore selection v{worldSel.data.selection.selection_version} before training</>}
                  desc="Reject unpromoted reward/environment drafts and resume from the exact hash-verified atomic tuple."
                />
              )}
              {hasPriorIters && (
                <div
                  style={{
                    border: "1px solid var(--hairline)",
                    borderRadius: "var(--radius-md)",
                    padding: 13,
                    background: "var(--surface-strong)",
                  }}
                >
                  <Field
                    label="Warm-start checkpoint (optional)"
                    htmlFor="run-warm-start-iteration"
                  >
                    {numField(
                      warmStartIteration,
                      (value) => {
                        setWarmStartIteration(value);
                        setProfile("custom");
                      },
                      {
                        id: "run-warm-start-iteration",
                        min: 0,
                        max: 999999,
                        placeholder: "Iteration, e.g. 22",
                      },
                    )}
                    <p className="rs-hintline">
                      Explicit policy recovery only: loads actor + critic from{" "}
                      <code className="mono">runs/iter_N/checkpoint</code>. The
                      selected reward, world tuple, and objective-metric mode do
                      not change.
                    </p>
                  </Field>
                </div>
              )}
              <div className="rs-row2">
                <Field label="Sculpt iters (outer)" htmlFor="run-iters">
                  {numField(iterations, (v) => setIterations(typeof v === "number" ? v : 1), { id: "run-iters", min: 1, max: 100 })}
                  <p className="rs-hintline">
                    Full train → diagnose → edit cycles. Runs no longer auto-kill on reward dips.
                  </p>
                </Field>
                <Field label={defaults.training_label} htmlFor="run-trainiters">
                  {numField(trainingIters, setTrainingIters, { id: "run-trainiters", min: 100, max: 200000 })}
                  <p className="rs-hintline">Inner-loop budget per cycle. Blank → config.toml.</p>
                </Field>
              </div>

              {isMjlab && (
                <div className="rs-row2">
                  <Field label="num_envs (override)" htmlFor="run-numenvs">
                    {numField(numEnvs, setNumEnvs, { id: "run-numenvs", min: 1, max: 8192 })}
                    <p className="rs-hintline">Drop if OOM. Halve + snap to power-of-two is safe.</p>
                  </Field>
                  <Field label="device" htmlFor="run-device">
                    <input
                      id="run-device"
                      className="rs-input mono"
                      value={device}
                      onChange={(e) => { setDevice(e.target.value); setProfile("custom"); }}
                      disabled={launch.isPending}
                      placeholder="cuda:0"
                    />
                    <p className="rs-hintline"><code className="mono">cpu</code> forces CPU (very slow for mjlab).</p>
                  </Field>
                </div>
              )}

              {/* §Ship-7: rollout-video + RL knobs — UI-reachable so the user
                   never needs to sed into config.toml. Blank = runner default. */}
              <div style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", padding: 13 }}>
                <p style={{ margin: "0 0 11px", fontSize: 10.5, fontWeight: 600, letterSpacing: 0.6, textTransform: "uppercase", color: "var(--rs-muted)" }}>
                  Rollout video + RL knobs
                </p>
                <div className="rs-row2">
                  <Field label="Episode steps" htmlFor="run-epsteps">
                    {numField(maxEpisodeSteps, setMaxEpisodeSteps, { id: "run-epsteps", min: 50, max: 5000, placeholder: "500" })}
                    <p className="rs-hintline">Env steps per rollout episode. Longer = longer video.</p>
                  </Field>
                  <Field label="Playback speed" htmlFor="run-playback">
                    {numField(playbackSpeed, setPlaybackSpeed, { id: "run-playback", step: 0.1, min: 0.1, max: 10, placeholder: "1.0 (real-time)" })}
                    <p className="rs-hintline">1.0 = real-time · 0.5 = slow-mo · 2.0 = 2×.</p>
                  </Field>
                  <Field label="Rollout episodes" htmlFor="run-rolleps">
                    {numField(rolloutEpisodes, setRolloutEpisodes, { id: "run-rolleps", min: 1, max: 32, placeholder: "6" })}
                    <p className="rs-hintline">Episodes captured per iter for behavior metrics.</p>
                  </Field>
                  <Field label="Seed" htmlFor="run-seed">
                    {numField(seed, setSeed, { id: "run-seed", min: 0, placeholder: "42" })}
                    <p className="rs-hintline">Base RNG seed; iter N uses seed + N.</p>
                  </Field>
                  <Field label="Video resolution" htmlFor="run-resolution">
                    <div className="rs-select">
                      <select
                        id="run-resolution"
                        value={renderSize}
                        onChange={(e) => { setRenderSize(e.target.value as RenderSize); setProfile("custom"); }}
                        disabled={launch.isPending}
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
                  <Field label="Evidence environment" htmlFor="run-render-env">
                    {numField(renderEnvIndex, setRenderEnvIndex, {
                      id: "run-render-env",
                      min: 0,
                      max: 63,
                      placeholder: "0",
                    })}
                    <p className="rs-hintline">
                      Precommit the parallel lane shown in video. Full-batch metrics stay unchanged.
                    </p>
                  </Field>
                </div>
              </div>

              {/* auto-physics selects, expand-kg / no-kg toggles */}
              <div style={{ display: "flex", flexDirection: "column", gap: 11, border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", padding: 13 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 12.5 }}>
                  <span style={{ fontWeight: 500, color: "var(--ink)" }}>Auto-physics on severe</span>
                  <div className="rs-select">
                    <select
                      value={autoAdjustPhysics === null ? "default" : autoAdjustPhysics ? "on" : "off"}
                      onChange={(e) => {
                        const v = e.target.value;
                        setAutoAdjustPhysics(v === "default" ? null : v === "on");
                        setProfile("custom");
                      }}
                      disabled={launch.isPending}
                      aria-label="Auto-physics on severe"
                    >
                      <option value="default">project default</option>
                      <option value="on">on</option>
                      <option value="off">off</option>
                    </select>
                  </div>
                  <span style={{ color: "var(--rs-muted)" }}>emits a physics-edit chip on severe realism audits.</span>
                </div>

                {/* §Ship 34/35: objective fitness metric — built-ins + auto-generated. */}
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 12.5 }}>
                  <span style={{ fontWeight: 500, color: "var(--ink)" }}>Objective fitness metric</span>
                  <div className="rs-select">
                    <select
                      value={fitnessMetric ?? "none"}
                      onChange={(e) => { const v = e.target.value; setFitnessMetric(v === "none" ? null : v); }}
                      disabled={launch.isPending || genMetric.isPending}
                      aria-label="Objective fitness metric"
                    >
                      <option value="none">none (blind loop)</option>
                      <option value="generate-at-launch">✨ Generate a metric from this goal (at launch)</option>
                      <optgroup label="built-in (ground truth)">
                        {SPEC_METRIC_NAMES.map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </optgroup>
                      {genMetrics.length > 0 && (
                        <optgroup label="auto-generated for this project">
                          {genMetrics.map((m) => (
                            <option key={m.id} value={`gen:${m.id}`}>
                              {m.id}{m.calibrated ? " ✓ calibrated" : " (observe-only)"} — {m.behavior_goal?.slice(0, 40)}
                            </option>
                          ))}
                        </optgroup>
                      )}
                    </select>
                  </div>
                  {/* §best-of-N: sample N candidate metrics and keep the most-
                      discriminating valid one (offline selection). Applies to BOTH
                      the Generate button and the generate-at-launch path. */}
                  <div
                    className="rs-select"
                    title="Sample N candidate metrics and keep the most-discriminating valid one. Each candidate is a ~1-2 min LLM call."
                  >
                    <select
                      value={metricCandidates}
                      onChange={(e) => setMetricCandidates(Number(e.target.value))}
                      disabled={launch.isPending || genMetric.isPending}
                      aria-label="Best-of-N candidates"
                    >
                      <option value={1}>1 candidate (fast)</option>
                      <option value={2}>best-of-2</option>
                      <option value={3}>best-of-3</option>
                      <option value={4}>best-of-4</option>
                    </select>
                  </div>
                  {!isLaunchGen && <Btn
                    kind="quiet"
                    icon={genMetric.isPending ? "loader" : "sparkles"}
                    disabled={genMetric.isPending || behavior.trim().length < 4}
                    title={metricCandidates > 1
                      ? `Sample ${metricCandidates} candidates and keep the most-discriminating one`
                      : "Auto-generate an objective metric from the behavior goal above"}
                    onClick={runGenerate}
                  >
                    {genMetric.isPending ? "Generating… (~1-2 min)" : "Generate from goal"}
                  </Btn>}
                  <span style={{ color: "var(--rs-muted)" }}>
                    {isLaunchGen
                      ? "generated at launch from the prompt — no stored trajectory required; runs observe-only until calibrated."
                      : "prompt-native validation works without a stored trajectory; generated metrics observe until calibration."}
                  </span>
                </div>

                {/* §Ship 40: live generation progress (generate → validate →
                    regenerate-on-failure → review), polled from the backend.
                    Polling is keyed to isPending, so it stops the moment the
                    mutation settles — success, error, or timeout. */}
                {genMetric.isPending && (
                  <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, paddingLeft: 14, color: "var(--ink)" }}>
                    <Icon name="loader" size={13} />
                    <span>{genProgress.data?.message ?? "Starting generation…"}</span>
                  </div>
                )}

                {/* A failed/timed-out generate must never strand the dialog:
                    surface the error HERE (not just a transient toast) with a
                    one-click retry. */}
                {genMetric.isError && !genMetric.isPending && (
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", fontSize: 12.5, paddingLeft: 14, color: "var(--st-rose-fg)" }}>
                    <Icon name="alert-triangle" size={13} />
                    <span style={{ flex: 1, minWidth: 180 }}>
                      {genMetric.error?.name === "TimeoutError"
                        ? "Metric generation timed out — the backend didn't respond. It may still be running; retry or pick a metric manually."
                        : `Metric generation failed: ${genMetric.error?.message ?? "unknown error"}`}
                    </span>
                    <Btn
                      kind="quiet"
                      size="xs"
                      icon="refresh-cw"
                      disabled={behavior.trim().length < 4}
                      onClick={runGenerate}
                    >
                      Retry
                    </Btn>
                  </div>
                )}

                {/* §Ship 35: observe vs steer — only when a metric is chosen. */}
                {fitnessMetric !== null && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 12.5, paddingLeft: 14 }}>
                    <span style={{ fontWeight: 500, color: "var(--ink)" }}>Fitness mode</span>
                    <div className="rs-select">
                      <select
                        value={steerLocked || isLaunchGen ? "observe" : fitnessMode}
                        onChange={(e) => setFitnessMode(e.target.value === "observe" ? "observe" : "steer")}
                        disabled={launch.isPending || steerLocked || isLaunchGen}
                        aria-label="Fitness mode"
                      >
                        <option value="steer">steer (drives selection)</option>
                        <option value="observe">observe (display only)</option>
                      </select>
                    </div>
                    <span style={{ color: "var(--rs-muted)" }}>
                      {isLaunchGen
                        ? "the metric is generated at launch and runs observe-only; it earns steer-rights if launch-time calibration passes."
                        : steerLocked
                        ? "this generated metric is observe-only until it passes calibration (Spearman vs a built-in ground truth)."
                        : "observe = compute & chart it without influencing the run (for a blind-vs-guided A/B)."}
                    </span>
                  </div>
                )}

                {/* §Ship 48: fitness-plateau patience — the LIVE early stop on a
                    steered run (distinct from the legacy early-stop no-op). */}
                {fitnessMetric !== null && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 12.5, paddingLeft: 14 }}>
                    <span style={{ fontWeight: 500, color: "var(--ink)" }}>Fitness patience</span>
                    <div style={{ width: 72 }}>
                      {numField(fitnessPatience, setFitnessPatience, {
                        "aria-label": "Fitness patience",
                        min: 1, max: 50, step: 1, placeholder: "4",
                      } as React.InputHTMLAttributes<HTMLInputElement>)}
                    </div>
                    <span style={{ color: "var(--rs-muted)" }}>
                      iters with no new best fitness before the run stops (sculpt default 2;
                      4+ gives a hard exploratory skill room to escape a local optimum).
                    </span>
                  </div>
                )}

                {/* §Ship 35: earn steer-rights — calibrate an uncalibrated
                    generated metric against a hand-authored ground truth. */}
                {steerLocked && selectedGen && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 12.5, paddingLeft: 14 }}>
                    <span style={{ fontWeight: 500, color: "var(--ink)" }}>Earn steer-rights</span>
                    <span style={{ color: "var(--rs-muted)" }}>calibrate vs</span>
                    <div className="rs-select">
                      <select
                        value={calibrateAgainst}
                        onChange={(e) => setCalibrateAgainst(e.target.value)}
                        disabled={calibrate.isPending}
                        aria-label="Calibrate against built-in metric"
                      >
                        {SPEC_METRIC_NAMES.map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                    </div>
                    <Btn
                      kind="quiet"
                      icon={calibrate.isPending ? "loader" : "git-compare"}
                      disabled={calibrate.isPending}
                      onClick={() => {
                        calibrate.mutate(
                          { metricId: selectedGen.id, against: calibrateAgainst },
                          {
                            onSuccess: (m) => {
                              if (m.calibrated) {
                                setFitnessMode("steer");
                                toast.success(`${m.id} calibrated — steer unlocked`, {
                                  description: `Spearman ${m.calibration?.spearman ?? "?"} vs ${calibrateAgainst}`,
                                });
                              } else {
                                toast.error("Did not pass calibration", {
                                  description: `Spearman ${m.calibration?.spearman ?? "?"} < threshold; stays observe-only.`,
                                });
                              }
                            },
                            onError: (err) => {
                              const d = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
                              toast.error("Calibration failed", { description: d });
                            },
                          },
                        );
                      }}
                    >
                      {calibrate.isPending ? "Calibrating…" : "Calibrate"}
                    </Btn>
                  </div>
                )}

                {/* "Expand knowledge graph" used to live here, labelled
                    `--expand-kg`. That flag does not exist in the CLI: the
                    backend accepts the field and stores it on the job, and
                    nothing ever reads it. A toggle that cannot change the run
                    is worse than no toggle, so it is gone rather than
                    disabled. Research a topic explicitly from the Knowledge
                    tab instead. */}
                <ToggleRow
                  on={noKg} onChange={setNoKg} label="No knowledge graph"
                  title={<><code className="mono">--no-kg</code> · ablation</>}
                  desc="diagnoser sees empty literature context"
                />
              </div>
            </>
          )}
        </Modal>
      )}
      {referencePickerOpen && (
        <ReferencePickerDialog
          slug={slug}
          currentClipId={referenceClipId}
          initialQuery={behavior}
          robot={referenceRobot}
          onPick={({ clipId }) => {
            setReferenceClipId(clipId);
            // Persist immediately, not only on launch: composing a motion and
            // then closing the dialog to author its per-mode rewards is the
            // documented order, and the clip has to survive that.
            saveDraft.mutate({ reference_clip_id: clipId,
                               reference_robot: referenceRobot });
          }}
          onClose={() => setReferencePickerOpen(false)}
        />
      )}
    </>
  );
}

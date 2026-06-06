import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, Field, Modal, ToggleRow } from "@/components/rs/primitives";
import { useLaunchRun } from "@/hooks/useRuns";
import { ApiError } from "@/lib/api";
import type { ProjectDetail } from "@/lib/types";


// ── Per-adapter expected seconds-per-cycle (S8 / §7.7) ───────────────
//
// Rough wall-clock budget per outer sculpt iter, derived from the
// envelope at the top of pickAdapterDefaults + real-world numbers in
// CONTEXT.md. Used ONLY to warn the user before a long run — never
// used to gate anything. Honest enough to give a useful number, loose
// enough that the user doesn't anchor on it.
//
//   gym_sb3:        ~180 s/cycle (3 min)     — train + rollout + LLM
//   mjlab_cartpole: ~60 s/cycle              — fast toy task
//   mjlab_go1:      ~1320 s/cycle (22 min)   — Sam's observed Go1 at 1500 iters
//   mjlab_g1:       ~1500 s/cycle (25 min)   — heavier humanoid
//   mjlab_other:    ~600 s/cycle (10 min)    — conservative
const SECONDS_PER_CYCLE: Record<string, number> = {
  gym_sb3: 180,
  mjlab_cartpole: 60,
  mjlab_go1: 1320,
  mjlab_g1: 1500,
  mjlab_other: 600,
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
      device: String(project.adapter_config?.device) || "cuda:0",
      training_label: "rsl_rl iters / cycle",
    };
  }
  if (cls.includes("mjlab") && /(Go1|Go2)/i.test(taskId)) {
    return {
      kind: "mjlab_go1",
      iterations: 12,
      training_iterations: 1000,
      num_envs: Number(project.adapter_config?.num_envs) || 2048,
      device: String(project.adapter_config?.device) || "cuda:0",
      training_label: "rsl_rl iters / cycle",
    };
  }
  if (cls.includes("mjlab") && /G1/i.test(taskId)) {
    return {
      kind: "mjlab_g1",
      iterations: 8,
      training_iterations: 1500,
      num_envs: Number(project.adapter_config?.num_envs) || 2048,
      device: String(project.adapter_config?.device) || "cuda:0",
      training_label: "rsl_rl iters / cycle",
    };
  }
  // Any other mjlab task (T1, ANYmal, custom).
  return {
    kind: "mjlab_other",
    iterations: 10,
    training_iterations: 1000,
    num_envs: Number(project.adapter_config?.num_envs) || 1024,
    device: String(project.adapter_config?.device) || "cuda:0",
    training_label: "rsl_rl iters / cycle",
  };
}


export function NewRunDialog({
  slug,
  project,
  onLaunched,
}: {
  slug: string;
  project: ProjectDetail;
  onLaunched: (runId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"basic" | "advanced">("basic");
  const defaults = useMemo(() => pickAdapterDefaults(project), [project]);
  const isMjlab = defaults.kind.startsWith("mjlab");

  const [behavior, setBehavior] = useState("");
  const [iterations, setIterations] = useState(defaults.iterations);
  const [trainingIters, setTrainingIters] = useState<number | "">(
    defaults.training_iterations,
  );
  const [numEnvs, setNumEnvs] = useState<number | "">(defaults.num_envs);
  const [device, setDevice] = useState<string>(defaults.device);
  const [noKg, setNoKg] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [expandKg, setExpandKg] = useState(false);
  // §Ship-7: rollout-video + RL knobs. Empty string = "leave blank →
  // use runner/config default".
  const [maxEpisodeSteps, setMaxEpisodeSteps] = useState<number | "">("");
  const [playbackSpeed, setPlaybackSpeed] = useState<number | "">("");
  const [rolloutEpisodes, setRolloutEpisodes] = useState<number | "">("");
  const [seed, setSeed] = useState<number | "">("");
  const [autoAdjustPhysics, setAutoAdjustPhysics] = useState<boolean | null>(null);
  const launch = useLaunchRun(slug);

  // S8 / §7.7 — ETA estimate + resume-warning banner. Pure view-layer
  // logic; no new API call. Uses the documented per-cycle budget for
  // the active adapter kind. When dry-run is checked, override to 50 s
  // total (the comment on the --dry-run checkbox promises "~50 s total").
  const etaSeconds = useMemo(() => {
    if (dryRun) return 50;
    const perCycle = SECONDS_PER_CYCLE[defaults.kind] ?? 300;
    return perCycle * iterations;
  }, [dryRun, defaults.kind, iterations]);
  const eta = useMemo(() => formatEta(etaSeconds), [etaSeconds]);
  const isLongRun = etaSeconds >= 30 * 60; // 30 min
  const hasPriorIters = project.n_iterations_completed > 0;

  // Pre-fill behavior from sidecar description; reset advanced fields
  // when the adapter changes (defaults drift with the project).
  useEffect(() => {
    if (open) {
      if (!behavior && project.description) setBehavior(project.description);
      setIterations(defaults.iterations);
      setTrainingIters(defaults.training_iterations);
      setNumEnvs(defaults.num_envs);
      setDevice(defaults.device);
    }
  }, [open, project.description, defaults, behavior]);

  const submit = () => {
    if (behavior.trim().length < 4) {
      toast.error("Behavior goal too short", { description: "At least 4 chars." });
      return;
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
      expand_kg: expandKg,
      // §Ship-7: only forward when set; null = runner / config default.
      max_episode_steps:
        typeof maxEpisodeSteps === "number" ? maxEpisodeSteps : null,
      playback_speed:
        typeof playbackSpeed === "number" ? playbackSpeed : null,
      rollout_episodes:
        typeof rolloutEpisodes === "number" ? rolloutEpisodes : null,
      seed: typeof seed === "number" ? seed : null,
      auto_adjust_physics: autoAdjustPhysics,
    };
    launch.mutate(body, {
      onSuccess: (r) => {
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
      onChange={(e) => set(e.target.value === "" ? "" : Number(e.target.value))}
      disabled={launch.isPending}
    />
  );

  return (
    <>
      <Btn kind="primary" size="sm" icon="play" onClick={() => setOpen(true)}>New run</Btn>
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
          onClose={() => { if (!launch.isPending) setOpen(false); }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => setOpen(false)} disabled={launch.isPending}>Cancel</Btn>
              <Btn kind="primary" icon={launch.isPending ? "loader" : "play"} onClick={submit} disabled={launch.isPending}>
                {launch.isPending ? "Launching…" : "Launch"}
              </Btn>
            </>
          }
        >
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

          <div className="rs-mtabs">
            <button className={tab === "basic" ? "on" : ""} onClick={() => setTab("basic")}>Basic</button>
            <button className={tab === "advanced" ? "on" : ""} onClick={() => setTab("advanced")}>Advanced</button>
          </div>

          {tab === "basic" ? (
            <>
              <Field label="Behavior goal" htmlFor="run-goal">
                <textarea
                  id="run-goal"
                  className="rs-textarea"
                  value={behavior}
                  onChange={(e) => setBehavior(e.target.value)}
                  placeholder="Run forward as fast as possible without falling."
                  style={{ minHeight: 84 }}
                  maxLength={500}
                  disabled={launch.isPending}
                  autoFocus
                />
              </Field>
              <ToggleRow
                on={dryRun} onChange={setDryRun} label="Dry run"
                title={<><code className="mono">--dry-run</code> · smoke-test the pipeline</>}
                desc="training steps capped at 1000, LLM calls stubbed — ~50 s total"
              />
            </>
          ) : (
            <>
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
                      onChange={(e) => setDevice(e.target.value)}
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
                </div>
              </div>

              {/* auto-physics selects, expand-kg / no-kg toggles */}
              <div style={{ display: "flex", flexDirection: "column", gap: 11, border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", padding: 13 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 12.5 }}>
                  <span style={{ fontWeight: 500, color: "var(--ink)" }}>Auto-physics on severe</span>
                  <div className="rs-select">
                    <select
                      value={autoAdjustPhysics === null ? "default" : autoAdjustPhysics ? "on" : "off"}
                      onChange={(e) => { const v = e.target.value; setAutoAdjustPhysics(v === "default" ? null : v === "on"); }}
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

                <ToggleRow
                  on={expandKg} onChange={setExpandKg} label="Expand knowledge graph"
                  title={<><code className="mono">--expand-kg</code> · auto-research thin topics</>}
                  desc="Claude researches thin topics before Stage-2 diagnose (Phase 2)"
                />
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
    </>
  );
}

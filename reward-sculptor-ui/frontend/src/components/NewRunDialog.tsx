import { useEffect, useMemo, useState } from "react";
import { Clock, Loader2, Play, SkipForward } from "lucide-react";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
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
  // §Ship-9a: per-run early-stop controls (null = project default).
  const [earlyStopEnabled, setEarlyStopEnabled] = useState<boolean | null>(null);
  const [earlyStopPatience, setEarlyStopPatience] = useState<number | "">("");
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
      early_stop_enabled: earlyStopEnabled,
      early_stop_patience:
        typeof earlyStopPatience === "number" ? earlyStopPatience : null,
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

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm">
          <Play />
          New run
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Launch a sculpt run</DialogTitle>
          <DialogDescription>
            {isMjlab ? (
              <>
                GPU adapter (<code>{defaults.kind}</code>). Advanced tab
                has GPU-flavored overrides. Defaults sized for RTX 5070
                Laptop (8 GiB VRAM).
              </>
            ) : (
              <>
                CPU adapter (<code>gym_sb3</code>). Advanced tab has
                `steps_per_iter` override + KG toggles.
              </>
            )}
          </DialogDescription>
        </DialogHeader>

        <div
          className={`flex flex-col gap-1.5 rounded-md border px-3 py-2 text-xs ${
            isLongRun
              ? "border-amber-500/40 bg-amber-500/5 text-amber-800 dark:text-amber-200"
              : "border-border bg-muted/30 text-muted-foreground"
          }`}
        >
          <div className="flex items-center gap-1.5 font-medium">
            <Clock className="h-3.5 w-3.5" />
            <span>
              Estimated wall-clock: <strong>{eta.label}</strong>
              {!dryRun && (
                <> · finishes ~<strong>{eta.finishAt}</strong></>
              )}
            </span>
          </div>
          {isLongRun && !dryRun && (
            <div className="text-[10.5px]">
              Heads up — long run ahead. Laptop sleep, power loss, or a
              backend restart will leave completed iters on disk, and
              resume is on by default so rerunning picks up where it
              stopped. Consider --dry-run first to shake the pipeline.
            </div>
          )}
          {hasPriorIters && (
            <div className="flex items-start gap-1.5 text-[10.5px]">
              <SkipForward className="mt-0.5 h-3 w-3 shrink-0" />
              <span>
                Resume is enabled — this project has{" "}
                <strong>{project.n_iterations_completed}</strong> iter(s)
                on disk; existing `runs/iter_&lt;N&gt;/` artifacts
                (checkpoint, rollout, trajectory) will be reused rather
                than retrained. `rm -rf runs/iter_&lt;N&gt;/` to force
                a fresh train of a specific iter.
              </span>
            </div>
          )}
        </div>

        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>
          <TabsContent value="basic" className="mt-3 space-y-3">
            <div className="grid gap-1.5">
              <Label htmlFor="goal">Behavior goal</Label>
              <Textarea
                id="goal"
                value={behavior}
                onChange={(e) => setBehavior(e.target.value)}
                placeholder="Run forward as fast as possible without falling."
                rows={3}
                maxLength={500}
                disabled={launch.isPending}
              />
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
                disabled={launch.isPending}
              />
              <span>
                <span className="font-medium">--dry-run</span>
                <span className="ml-1 text-muted-foreground">
                  training steps capped at 1000, LLM calls stubbed — ~50 s total
                </span>
              </span>
            </label>
          </TabsContent>

          <TabsContent value="advanced" className="mt-3 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-1.5">
                <Label htmlFor="iters" className="text-xs">
                  Sculpt iters (outer)
                </Label>
                <Input
                  id="iters"
                  type="number"
                  min={1}
                  max={100}
                  value={iterations}
                  onChange={(e) => setIterations(Number(e.target.value) || 1)}
                  disabled={launch.isPending}
                />
                <p className="text-[10px] text-muted-foreground">
                  Full train → diagnose → edit cycles.
                  {earlyStopEnabled === false
                    ? " Early-stop disabled — runs full budget."
                    : ` Early-stop after ${
                        typeof earlyStopPatience === "number"
                          ? earlyStopPatience
                          : 3
                      } no-improvement iter${
                        (typeof earlyStopPatience === "number"
                          ? earlyStopPatience
                          : 3) === 1
                          ? ""
                          : "s"
                      } (project default unless overridden below).`}
                </p>
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="trainiters" className="text-xs">
                  {defaults.training_label}
                </Label>
                <Input
                  id="trainiters"
                  type="number"
                  min={100}
                  max={200000}
                  value={trainingIters}
                  onChange={(e) => {
                    const v = e.target.value;
                    setTrainingIters(v === "" ? "" : Number(v));
                  }}
                  disabled={launch.isPending}
                />
                <p className="text-[10px] text-muted-foreground">
                  Inner-loop budget per cycle. Leave blank to use config.toml.
                </p>
              </div>
            </div>

            {isMjlab && (
              <div className="grid grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <Label htmlFor="num_envs" className="text-xs">
                    num_envs (override)
                  </Label>
                  <Input
                    id="num_envs"
                    type="number"
                    min={1}
                    max={8192}
                    value={numEnvs}
                    onChange={(e) => {
                      const v = e.target.value;
                      setNumEnvs(v === "" ? "" : Number(v));
                    }}
                    disabled={launch.isPending}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Drop if OOM. Halve + snap to power-of-two is safe.
                  </p>
                </div>
                <div className="grid gap-1.5">
                  <Label htmlFor="device" className="text-xs">
                    device
                  </Label>
                  <Input
                    id="device"
                    value={device}
                    onChange={(e) => setDevice(e.target.value)}
                    disabled={launch.isPending}
                    placeholder="cuda:0"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    `cpu` forces CPU (very slow for mjlab).
                  </p>
                </div>
              </div>
            )}

            {/* §Ship-7: rollout-video + RL knobs. Per Sam's feedback
                 2026-04-23 — these must be UI-reachable so the user
                 never needs to `sed -i` into config.toml. Blank =
                 runner default. */}
            <div className="rounded-md border bg-muted/10 p-3">
              <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                Rollout video + RL knobs
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <Label htmlFor="ep_steps" className="text-xs">
                    Episode steps
                  </Label>
                  <Input
                    id="ep_steps"
                    type="number"
                    min={50}
                    max={5000}
                    value={maxEpisodeSteps}
                    onChange={(e) => {
                      const v = e.target.value;
                      setMaxEpisodeSteps(v === "" ? "" : Number(v));
                    }}
                    disabled={launch.isPending}
                    placeholder="500"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Env steps per rollout episode. Longer = longer video.
                  </p>
                </div>
                <div className="grid gap-1.5">
                  <Label htmlFor="playback" className="text-xs">
                    Playback speed
                  </Label>
                  <Input
                    id="playback"
                    type="number"
                    step={0.1}
                    min={0.1}
                    max={10}
                    value={playbackSpeed}
                    onChange={(e) => {
                      const v = e.target.value;
                      setPlaybackSpeed(v === "" ? "" : Number(v));
                    }}
                    disabled={launch.isPending}
                    placeholder="1.0 (real-time)"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    1.0 = real-time; 0.5 = slow-mo; 2.0 = 2x speed.
                  </p>
                </div>
                <div className="grid gap-1.5">
                  <Label htmlFor="roll_eps" className="text-xs">
                    Rollout episodes
                  </Label>
                  <Input
                    id="roll_eps"
                    type="number"
                    min={1}
                    max={32}
                    value={rolloutEpisodes}
                    onChange={(e) => {
                      const v = e.target.value;
                      setRolloutEpisodes(v === "" ? "" : Number(v));
                    }}
                    disabled={launch.isPending}
                    placeholder="6"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Episodes captured per iter for behavior metrics.
                  </p>
                </div>
                <div className="grid gap-1.5">
                  <Label htmlFor="seed" className="text-xs">
                    Seed
                  </Label>
                  <Input
                    id="seed"
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(e) => {
                      const v = e.target.value;
                      setSeed(v === "" ? "" : Number(v));
                    }}
                    disabled={launch.isPending}
                    placeholder="42"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Base RNG seed; iter N uses seed + N.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-2 rounded-md border bg-muted/20 p-3">
              {/* §Ship-7: auto_adjust_physics UI toggle. null = use project
                   config.toml default (true for new projects). */}
              <div className="mb-1 flex items-center gap-2 text-xs">
                <span className="font-medium">Auto-physics on severe</span>
                <select
                  value={
                    autoAdjustPhysics === null
                      ? "default"
                      : autoAdjustPhysics
                      ? "on"
                      : "off"
                  }
                  onChange={(e) => {
                    const v = e.target.value;
                    setAutoAdjustPhysics(
                      v === "default" ? null : v === "on",
                    );
                  }}
                  disabled={launch.isPending}
                  className="rounded border bg-background px-2 py-0.5 text-xs"
                >
                  <option value="default">project default</option>
                  <option value="on">on</option>
                  <option value="off">off</option>
                </select>
                <span className="text-muted-foreground">
                  — emits physics-edit suggestion chip on severe realism audits.
                </span>
              </div>
              {/* §Ship-9a: early-stop per-run controls. */}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-medium">Early-stop</span>
                <select
                  value={
                    earlyStopEnabled === null
                      ? "default"
                      : earlyStopEnabled
                      ? "on"
                      : "off"
                  }
                  onChange={(e) => {
                    const v = e.target.value;
                    setEarlyStopEnabled(v === "default" ? null : v === "on");
                  }}
                  disabled={launch.isPending}
                  className="rounded border bg-background px-2 py-0.5 text-xs"
                >
                  <option value="default">project default</option>
                  <option value="on">on</option>
                  <option value="off">off (run full budget)</option>
                </select>
                <Label
                  htmlFor="es_patience"
                  className="ml-2 text-[11px] font-normal text-muted-foreground"
                >
                  patience
                </Label>
                <Input
                  id="es_patience"
                  type="number"
                  min={1}
                  max={100}
                  value={earlyStopPatience}
                  onChange={(e) => {
                    const v = e.target.value;
                    setEarlyStopPatience(v === "" ? "" : Number(v));
                  }}
                  disabled={launch.isPending || earlyStopEnabled === false}
                  placeholder="3"
                  className="h-7 w-16 text-xs"
                />
                <span className="text-muted-foreground">
                  iters of no primary_metric improvement before truncation.
                </span>
              </div>
              <label className="flex cursor-pointer items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={expandKg}
                  onChange={(e) => setExpandKg(e.target.checked)}
                  disabled={launch.isPending}
                />
                <span>
                  <span className="font-medium">--expand-kg</span>
                  <span className="ml-1 text-muted-foreground">
                    Claude auto-researches thin topics before Stage-2
                    diagnose (Phase 2)
                  </span>
                </span>
              </label>
              <label className="flex cursor-pointer items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={noKg}
                  onChange={(e) => setNoKg(e.target.checked)}
                  disabled={launch.isPending}
                />
                <span>
                  <span className="font-medium">--no-kg</span>
                  <span className="ml-1 text-muted-foreground">
                    ablation: diagnoser sees empty literature context
                  </span>
                </span>
              </label>
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={launch.isPending}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={launch.isPending}>
            {launch.isPending ? (
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

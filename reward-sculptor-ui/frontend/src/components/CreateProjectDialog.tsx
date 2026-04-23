import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowRight,
  BookOpen,
  Cpu,
  ExternalLink,
  Hourglass,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useLibraryAdapters, useSystemGpu } from "@/hooks/useLibrary";
import { ApiError, createProject } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { AdapterInfo, LibraryRobot, ProjectDetail } from "@/lib/types";

/** Static formula from MJLAB_PIVOT_DESIGN §9 / preflight fallback:
 *  `1.5 GiB + 0.5 MB × num_envs`. Matches the backend's fallback so
 *  the UI estimate and the backend preflight round-trip agree when no
 *  measured per-env coefficient is available yet. */
function estimateVramGb(numEnvs: number): number {
  return 1.5 + (0.5 / 1024) * numEnvs;
}

type OomError = {
  title: string;
  detail: string;
  suggestedNumEnvs: number | null;
  freeVramGb: number | null;
  estimatedRequiredGb: number | null;
  deviceName: string | null;
};

function parseOomError(err: unknown): OomError | null {
  if (!(err instanceof ApiError)) return null;
  if (err.problem.type !== "/problems/insufficient-vram") return null;
  const body = err.problem as unknown as Record<string, unknown>;
  return {
    title: err.problem.title,
    detail: err.problem.detail ?? "",
    suggestedNumEnvs:
      typeof body.suggested_num_envs === "number"
        ? (body.suggested_num_envs as number)
        : null,
    freeVramGb:
      typeof body.free_vram_gb === "number" ? (body.free_vram_gb as number) : null,
    estimatedRequiredGb:
      typeof body.estimated_required_gb === "number"
        ? (body.estimated_required_gb as number)
        : null,
    deviceName:
      typeof body.device_name === "string" ? (body.device_name as string) : null,
  };
}

export interface CreateProjectDialogProps {
  robot: LibraryRobot | null;
  onClose: () => void;
}

/** Pick the default adapter for a library robot based on training_support.
 *  mjlab_ready → "mjlab"; gymnasium_compatible or preview_only → "gym_sb3"
 *  (preview-only robots scaffold with gym_sb3 so the uploaded URDF
 *  renders; the UI still disables training). */
function defaultAdapterFor(robot: LibraryRobot | null): string {
  if (!robot) return "gym_sb3";
  return robot.training_support === "mjlab_ready" ? "mjlab" : "gym_sb3";
}

export function CreateProjectDialog({ robot, onClose }: CreateProjectDialogProps) {
  const navigate = useNavigate();
  const gpu = useSystemGpu();
  const adapters = useLibraryAdapters();

  // Form state — initialised (and re-initialised) from the incoming robot.
  const defaultTask = robot?.preconfigured_tasks?.[0];
  const [name, setName] = useState(robot?.display_name ?? "");
  const [taskId, setTaskId] = useState<string>(defaultTask?.task_id ?? "");
  const [numEnvs, setNumEnvs] = useState(
    defaultTask?.recommended_num_envs ?? 1024,
  );
  const [deviceIdx, setDeviceIdx] = useState(0);
  const [oom, setOom] = useState<OomError | null>(null);
  const [adapterName, setAdapterName] = useState<string>(defaultAdapterFor(robot));

  useEffect(() => {
    setName(robot?.display_name ?? "");
    setTaskId(defaultTask?.task_id ?? "");
    setNumEnvs(defaultTask?.recommended_num_envs ?? 1024);
    setDeviceIdx(0);
    setOom(null);
    setAdapterName(defaultAdapterFor(robot));
  }, [robot, defaultTask]);

  // Task switcher: when the user picks a different task_id from the
  // robot's preconfigured list, snap num_envs to that task's
  // recommended value (unless the user has manually overridden — we
  // only auto-snap if it currently matches the previously-selected
  // task's recommendation). For Cartpole that's Balance vs Swingup;
  // for Go1 that's Velocity-Flat vs Velocity-Rough. Issue 7 from
  // Test 1 round 3 (2026-04-22): pre-fix the form had no task
  // selector so `preconfigured_tasks[0]` was silently picked.
  const selectedTask = robot?.preconfigured_tasks?.find(
    (t) => t.task_id === taskId,
  );

  const adapterList: AdapterInfo[] = adapters.data ?? [];
  const selectedAdapter =
    adapterList.find((a) => a.name === adapterName) ?? null;
  const isComingSoon = selectedAdapter?.status === "coming_soon";

  const create = useMutation<
    ProjectDetail,
    unknown,
    { name: string; adapter: string; num_envs?: number; gpu_device?: string; task_id?: string }
  >({
    mutationFn: async (payload) => {
      return createProject({
        name: payload.name,
        adapter: payload.adapter,
        // library_slug only applies to ready adapters that correspond to
        // the library's training_support. When a user overrides to a
        // coming-soon adapter, the library association is still
        // recorded so KG seeds + metadata surface, but training is
        // gated by adapter_unavailable.
        library_slug: robot!.slug,
        ...(payload.task_id && { task_id: payload.task_id }),
        ...(payload.num_envs != null && { num_envs: payload.num_envs }),
        ...(payload.gpu_device && { gpu_device: payload.gpu_device }),
      });
    },
    onSuccess: (proj) => {
      toast.success(`Created ${proj.display_name}`, {
        description: `Library: ${robot?.display_name}`,
      });
      setOom(null);
      onClose();
      navigate(`/projects/${proj.slug}`);
    },
    onError: (err) => {
      const oomDetails = parseOomError(err);
      if (oomDetails) {
        setOom(oomDetails);
        return;
      }
      const msg =
        err instanceof ApiError
          ? err.problem.detail ?? err.problem.title
          : (err as Error).message;
      toast.error("Could not create project", { description: msg });
    },
  });

  const selectedDevice = gpu.data?.devices?.[deviceIdx];
  const freeGb = selectedDevice
    ? selectedDevice.free_memory_bytes / 1024 ** 3
    : null;
  const estGb = estimateVramGb(numEnvs);
  const overBudget =
    freeGb != null ? estGb > 0.85 * freeGb : false;
  const tightBudget =
    freeGb != null ? estGb > 0.7 * freeGb && !overBudget : false;

  if (!robot) return null;

  // isMjlab now reflects the SELECTED adapter, not just the robot's
  // training_support. A user can override to any adapter, including
  // coming-soon ones.
  const isMjlab = adapterName === "mjlab";
  const isGym = adapterName === "gym_sb3";
  const isPreview =
    robot.training_support === "preview_only" && adapterName === "gym_sb3";

  return (
    <Dialog open onOpenChange={(open) => !open && !create.isPending && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Create project
            <span className="text-sm font-normal text-muted-foreground">
              / {robot.display_name}
            </span>
          </DialogTitle>
          <DialogDescription>
            {isMjlab &&
              "mjlab-ready — pick a device and num_envs to match your GPU."}
            {isGym && "Gymnasium-compatible — CPU-friendly, training launches in-process."}
            {isPreview &&
              "Preview only — we'll scaffold the project so you can render the robot, but training isn't wired up for this robot yet."}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <Label htmlFor="cpd-name">Project name</Label>
            <Input
              id="cpd-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={robot.display_name}
            />
          </div>

          <div className="flex flex-col gap-1">
            <Label htmlFor="cpd-adapter">RL adapter</Label>
            {adapters.isLoading && !adapters.data ? (
              <div className="rounded-md border bg-muted/30 p-2 text-xs text-muted-foreground">
                Loading adapters…
              </div>
            ) : (
              <select
                id="cpd-adapter"
                value={adapterName}
                onChange={(e) => setAdapterName(e.target.value)}
                className={cn(
                  "flex h-9 w-full items-center rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                )}
              >
                {adapterList.map((a) => (
                  <option key={a.name} value={a.name}>
                    {a.status === "coming_soon" ? "⏳ " : ""}
                    {a.display_name}
                    {a.status === "coming_soon" && " (coming soon)"}
                  </option>
                ))}
              </select>
            )}
          </div>

          {isComingSoon && selectedAdapter && (
            <ComingSoonConfirmCard adapter={selectedAdapter} />
          )}

          {/* Task selector — visible when the library robot exposes
              more than one preconfigured task (e.g. Cartpole has
              Balance + Swingup; Go1 has Velocity-Flat + Velocity-Rough).
              Pre-fix the form silently picked preconfigured_tasks[0],
              so Sam couldn't pick Swingup for his Cartpole project. */}
          {robot.preconfigured_tasks.length > 1 && (
            <div className="flex flex-col gap-1">
              <Label htmlFor="cpd-task">Task</Label>
              <select
                id="cpd-task"
                value={taskId}
                onChange={(e) => {
                  const newTaskId = e.target.value;
                  setTaskId(newTaskId);
                  // Snap num_envs to the new task's recommendation if
                  // the user hasn't manually overridden away from the
                  // previous task's recommendation.
                  const newTask = robot.preconfigured_tasks.find(
                    (t) => t.task_id === newTaskId,
                  );
                  const prevTask = robot.preconfigured_tasks.find(
                    (t) => t.task_id === taskId,
                  );
                  if (
                    newTask &&
                    (!prevTask || numEnvs === prevTask.recommended_num_envs)
                  ) {
                    setNumEnvs(newTask.recommended_num_envs);
                  }
                }}
                className={cn(
                  "flex h-9 w-full items-center rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                )}
              >
                {robot.preconfigured_tasks.map((t) => (
                  <option key={t.task_id} value={t.task_id}>
                    {t.display_name} — {t.task_id}
                  </option>
                ))}
              </select>
              {selectedTask && (
                <p className="text-[10px] text-muted-foreground">
                  recommended: {selectedTask.recommended_num_envs} envs
                </p>
              )}
            </div>
          )}

          {isMjlab && (
            <>
              <div className="flex flex-col gap-1">
                <Label htmlFor="cpd-device">CUDA device</Label>
                {gpu.data?.cuda_available && gpu.data.devices.length > 0 ? (
                  <select
                    id="cpd-device"
                    value={deviceIdx}
                    onChange={(e) => setDeviceIdx(Number(e.target.value))}
                    className={cn(
                      "flex h-9 w-full items-center rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    )}
                  >
                    {gpu.data.devices.map((d) => (
                      <option key={d.index} value={d.index}>
                        cuda:{d.index} — {d.name} (
                        {(d.free_memory_bytes / 1024 ** 3).toFixed(1)} GiB free)
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="rounded-md border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
                    No CUDA device detected. Pick a Gymnasium-compatible robot.
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex items-baseline justify-between">
                  <Label htmlFor="cpd-num-envs">Parallel envs</Label>
                  <span className="font-mono text-xs text-muted-foreground">
                    {numEnvs} envs
                  </span>
                </div>
                <input
                  id="cpd-num-envs"
                  type="range"
                  min={128}
                  max={4096}
                  step={128}
                  value={numEnvs}
                  onChange={(e) => setNumEnvs(Number(e.target.value))}
                  className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-foreground"
                />
                <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                  <span>128</span>
                  <span>
                    recommended: {defaultTask?.recommended_num_envs ?? "—"}
                  </span>
                  <span>4096</span>
                </div>
              </div>

              <div
                className={cn(
                  "flex items-start gap-2 rounded-md border p-2 text-xs",
                  overBudget
                    ? "border-rose-500/40 bg-rose-500/5"
                    : tightBudget
                      ? "border-amber-500/40 bg-amber-500/5"
                      : "border-border bg-muted/30",
                )}
              >
                <Cpu className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <div className="flex-1">
                  <div className="font-medium">
                    Estimated VRAM: {estGb.toFixed(1)} GiB
                    {freeGb != null && ` of ${freeGb.toFixed(1)} GiB free`}
                  </div>
                  <p className="text-muted-foreground">
                    {overBudget
                      ? "Above 85% of free VRAM — backend will refuse unless you lower num_envs."
                      : tightBudget
                        ? "Above 70% — likely fits but close to the headroom ceiling."
                        : "Comfortable headroom for the default policy size."}
                  </p>
                </div>
              </div>
            </>
          )}

          {oom && (
            <OomRetryBanner
              oom={oom}
              onRetry={() => {
                if (oom.suggestedNumEnvs == null) return;
                setNumEnvs(oom.suggestedNumEnvs);
                setOom(null);
                create.mutate({
                  name,
                  adapter: adapterName,
                  num_envs: oom.suggestedNumEnvs,
                  gpu_device: `cuda:${deviceIdx}`,
                  ...(taskId && { task_id: taskId }),
                });
              }}
            />
          )}
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={onClose}
            disabled={create.isPending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            disabled={
              create.isPending ||
              !name.trim() ||
              (isMjlab && (!gpu.data?.cuda_available || overBudget))
            }
            onClick={() =>
              create.mutate({
                name: name.trim() || robot.display_name,
                adapter: adapterName,
                ...(taskId && { task_id: taskId }),
                ...(isMjlab && {
                  num_envs: numEnvs,
                  gpu_device: `cuda:${deviceIdx}`,
                }),
              })
            }
          >
            {create.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Creating…
              </>
            ) : isComingSoon ? (
              <>
                Create anyway <ArrowRight className="h-3.5 w-3.5" />
              </>
            ) : (
              <>
                Create project <ArrowRight className="h-3.5 w-3.5" />
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ComingSoonConfirmCard({ adapter }: { adapter: AdapterInfo }) {
  // Adoption guides are relative paths in the sculptor repo; we render
  // them as GitHub links. If the path is already absolute (https://...)
  // we link it directly.
  const guideHref = adapter.adoption_guide_url.startsWith("http")
    ? adapter.adoption_guide_url
    : `https://github.com/sjdoane/RL-Sculptor/blob/main/RewardSculptor/${adapter.adoption_guide_url}`;
  return (
    <div className="flex flex-col gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
      <div className="flex items-start gap-2">
        <Hourglass className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div className="flex-1">
          <div className="font-medium text-amber-700 dark:text-amber-300">
            {adapter.display_name} — scaffolded, not yet implemented
          </div>
          <p className="mt-0.5 text-muted-foreground">
            Project will be created but training will be disabled until
            this adapter is implemented. Continue?
          </p>
          <div className="mt-1 flex items-center gap-2 text-muted-foreground">
            <BookOpen className="h-3 w-3" />
            <a
              href={guideHref}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 hover:underline"
            >
              Adoption guide <ExternalLink className="h-3 w-3" />
            </a>
            {adapter.estimated_effort && (
              <>
                <span>·</span>
                <span>Estimated effort: {adapter.estimated_effort}</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function OomRetryBanner({
  oom,
  onRetry,
}: {
  oom: OomError;
  onRetry: () => void;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-md border border-rose-500/40 bg-rose-500/5 p-3 text-xs">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
        <div className="flex-1">
          <div className="font-medium text-rose-700 dark:text-rose-300">
            {oom.deviceName ?? "GPU"} has only{" "}
            {oom.freeVramGb?.toFixed(1) ?? "?"} GiB free
          </div>
          <p className="mt-0.5 text-muted-foreground">
            {oom.detail}{" "}
            {oom.estimatedRequiredGb != null &&
              `Needed: ~${oom.estimatedRequiredGb.toFixed(1)} GiB.`}
          </p>
        </div>
      </div>
      {oom.suggestedNumEnvs != null && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onRetry}
          className="self-start"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Retry with num_envs={oom.suggestedNumEnvs}
        </Button>
      )}
    </div>
  );
}

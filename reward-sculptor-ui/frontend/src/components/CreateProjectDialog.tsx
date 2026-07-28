import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, Field, Modal } from "@/components/rs/primitives";
import { useLibraryAdapters, useSystemGpu } from "@/hooks/useLibrary";
import { ApiError, createProject } from "@/lib/api";
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
  const [description, setDescription] = useState("");
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
        ...(description.trim() && { description: description.trim() }),
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
    <Modal
      icon="folder"
      title="Create project"
      subtitle={
        <span style={{ display: "inline-flex", gap: 6, flexWrap: "wrap" }}>
          <span className="mono">/ {robot.display_name}</span>
          <span>·{" "}
            {isMjlab && "mjlab-ready — pick a device and num_envs to match your GPU."}
            {isGym && "Gymnasium-compatible — CPU-friendly, training launches in-process."}
            {isPreview && "Preview only — scaffolds so you can render the robot; training isn't wired up yet."}
          </span>
        </span>
      }
      onClose={() => { if (!create.isPending) onClose(); }}
      footer={
        <>
          <Btn kind="quiet" onClick={onClose} disabled={create.isPending}>Cancel</Btn>
          <Btn
            kind="primary"
            iconRight={create.isPending ? undefined : "arrow-right"}
            icon={create.isPending ? "loader" : undefined}
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
            {create.isPending ? "Creating…" : isComingSoon ? "Create anyway" : "Create project"}
          </Btn>
        </>
      }
    >
      <Field label="Project name" htmlFor="cpd-name">
        <input
          id="cpd-name"
          className="rs-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={robot.display_name}
          autoFocus
        />
      </Field>

      {/* The project header renders this, and with nothing able to set it,
          every project read "No description." permanently. */}
      <Field
        label="Description"
        hint="optional"
        htmlFor="cpd-description"
      >
        <textarea
          id="cpd-description"
          className="rs-input"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What should this robot learn to do? Shown on the project header."
          rows={2}
          style={{ resize: "vertical", minHeight: 46 }}
        />
      </Field>

      <Field label="RL adapter" htmlFor="cpd-adapter">
        {adapters.isLoading && !adapters.data ? (
          <div style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--hairline)", background: "var(--surface-strong)", padding: 8, fontSize: 12, color: "var(--rs-muted)" }}>
            Loading adapters…
          </div>
        ) : (
          <div className="rs-select" style={{ display: "flex" }}>
            <select id="cpd-adapter" style={{ width: "100%" }} value={adapterName} onChange={(e) => setAdapterName(e.target.value)}>
              {adapterList.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.status === "coming_soon" ? "⏳ " : ""}
                  {a.display_name}
                  {a.status === "coming_soon" && " (coming soon)"}
                </option>
              ))}
            </select>
          </div>
        )}
      </Field>

      {isComingSoon && selectedAdapter && <ComingSoonConfirmCard adapter={selectedAdapter} />}

      {/* Task selector — visible when the library robot exposes more than
          one preconfigured task (Cartpole: Balance + Swingup; Go1:
          Velocity-Flat + Velocity-Rough). */}
      {robot.preconfigured_tasks.length > 1 && (
        <Field label="Task" hint={selectedTask ? `recommended: ${selectedTask.recommended_num_envs} envs` : undefined} htmlFor="cpd-task">
          <div className="rs-select" style={{ display: "flex" }}>
            <select
              id="cpd-task"
              style={{ width: "100%" }}
              value={taskId}
              onChange={(e) => {
                const newTaskId = e.target.value;
                setTaskId(newTaskId);
                const newTask = robot.preconfigured_tasks.find((t) => t.task_id === newTaskId);
                const prevTask = robot.preconfigured_tasks.find((t) => t.task_id === taskId);
                if (newTask && (!prevTask || numEnvs === prevTask.recommended_num_envs)) {
                  setNumEnvs(newTask.recommended_num_envs);
                }
              }}
            >
              {robot.preconfigured_tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.display_name} — {t.task_id}
                </option>
              ))}
            </select>
          </div>
        </Field>
      )}

      {isMjlab && (
        <>
          <Field label="CUDA device" htmlFor="cpd-device">
            {gpu.data?.cuda_available && gpu.data.devices.length > 0 ? (
              <div className="rs-select" style={{ display: "flex" }}>
                <select id="cpd-device" style={{ width: "100%" }} value={deviceIdx} onChange={(e) => setDeviceIdx(Number(e.target.value))}>
                  {gpu.data.devices.map((d) => (
                    <option key={d.index} value={d.index}>
                      cuda:{d.index} — {d.name} ({(d.free_memory_bytes / 1024 ** 3).toFixed(1)} GiB free)
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--st-rose-bg)", background: "var(--st-rose-bg)", color: "var(--st-rose-fg)", padding: 8, fontSize: 12 }}>
                No CUDA device detected. Pick a Gymnasium-compatible robot.
              </div>
            )}
          </Field>

          <Field label={<>Parallel envs <span className="mono" style={{ color: "var(--rs-muted)", fontWeight: 400 }}>{numEnvs} envs</span></>} htmlFor="cpd-num-envs">
            <input
              id="cpd-num-envs"
              type="range"
              min={128}
              max={4096}
              step={128}
              value={numEnvs}
              onChange={(e) => setNumEnvs(Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--rs-primary)" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "var(--rs-muted)" }}>
              <span>128</span>
              <span>recommended: {defaultTask?.recommended_num_envs ?? "—"}</span>
              <span>4096</span>
            </div>
          </Field>

          <div
            style={{
              display: "flex", alignItems: "flex-start", gap: 8, borderRadius: "var(--radius-md)", padding: 10, fontSize: 12,
              border: "1px solid " + (overBudget ? "color-mix(in srgb, var(--st-rose) 40%, transparent)" : tightBudget ? "color-mix(in srgb, var(--st-amber) 40%, transparent)" : "var(--hairline)"),
              background: overBudget ? "var(--st-rose-bg)" : tightBudget ? "var(--st-amber-bg)" : "var(--surface-strong)",
              color: overBudget ? "var(--st-rose-fg)" : tightBudget ? "var(--st-amber-fg)" : "var(--rs-muted)",
            }}
          >
            <span style={{ marginTop: 1, flexShrink: 0 }}><Icon name="cpu" size={14} /></span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600 }}>
                Estimated VRAM: {estGb.toFixed(1)} GiB{freeGb != null && ` of ${freeGb.toFixed(1)} GiB free`}
              </div>
              <p style={{ margin: "2px 0 0" }}>
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
    </Modal>
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
    <div style={{ borderRadius: "var(--radius-md)", border: "1px solid color-mix(in srgb, var(--st-amber) 40%, transparent)", background: "var(--st-amber-bg)", color: "var(--st-amber-fg)", padding: 12, fontSize: 12 }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
        <span style={{ marginTop: 1, flexShrink: 0 }}><Icon name="clock" size={16} /></span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 600 }}>{adapter.display_name} — scaffolded, not yet implemented</div>
          <p style={{ margin: "2px 0 0", color: "var(--rs-muted)" }}>
            Project will be created but training will be disabled until this adapter is implemented. Continue?
          </p>
          <div style={{ marginTop: 4, display: "flex", alignItems: "center", gap: 8, color: "var(--rs-muted)" }}>
            <Icon name="book" size={12} />
            <a href={guideHref} target="_blank" rel="noopener noreferrer" style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "inherit" }}>
              Adoption guide <Icon name="external" size={12} />
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
    <div style={{ display: "flex", flexDirection: "column", gap: 8, borderRadius: "var(--radius-md)", border: "1px solid color-mix(in srgb, var(--st-rose) 40%, transparent)", background: "var(--st-rose-bg)", color: "var(--st-rose-fg)", padding: 12, fontSize: 12 }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
        <span style={{ marginTop: 1, flexShrink: 0 }}><Icon name="alert-triangle" size={16} /></span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 600 }}>
            {oom.deviceName ?? "GPU"} has only {oom.freeVramGb?.toFixed(1) ?? "?"} GiB free
          </div>
          <p style={{ margin: "2px 0 0", color: "var(--rs-muted)" }}>
            {oom.detail}{" "}
            {oom.estimatedRequiredGb != null && `Needed: ~${oom.estimatedRequiredGb.toFixed(1)} GiB.`}
          </p>
        </div>
      </div>
      {oom.suggestedNumEnvs != null && (
        <Btn kind="ghost" size="sm" icon="refresh-cw" onClick={onRetry} style={{ alignSelf: "flex-start" }}>
          Retry with num_envs={oom.suggestedNumEnvs}
        </Btn>
      )}
    </div>
  );
}

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Save, Settings as SettingsIcon, Trash2 } from "lucide-react";
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
import { useDeleteProject } from "@/hooks/useProjects";
import {
  ApiError,
  getProjectSettings,
  patchProjectSettings,
  type IterationSettings,
} from "@/lib/api";
import type { ProjectDetail } from "@/lib/types";

/** Project settings dialog (M7 Phase 7c). Opens from a gear icon in
 * the ProjectDetail header. First pass surfaces:
 *   - project summary (read-only key/value pairs).
 *   - danger zone: delete project (type-to-confirm).
 *
 * Follow-ups: editable environment_tag + KG auto-research toggle +
 * iteration defaults (needs a PATCH /projects/{slug} endpoint). */
export function ProjectSettingsDialog({
  project,
}: {
  project: ProjectDetail;
}) {
  const [open, setOpen] = useState(false);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" title="Project settings">
          <SettingsIcon className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Project settings</DialogTitle>
          <DialogDescription>
            Inspect adapter + library config. Manage destructive
            actions from the danger zone.
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[70vh] space-y-4 overflow-y-auto py-2">
          <SummarySection project={project} />
          <IterationSettingsSection project={project} open={open} />
          <DangerZone project={project} onDeleted={() => setOpen(false)} />
        </div>
      </DialogContent>
    </Dialog>
  );
}


function IterationSettingsSection({
  project,
  open,
}: {
  project: ProjectDetail;
  open: boolean;
}) {
  // §Ship-8: editable `[iteration]` block from config.toml. Each field
  // is optional — blanking a field PATCHes no-op (we only send fields
  // the user actually changed). Per Sam's rule: no terminal edits.
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["project-settings", project.slug],
    queryFn: () => getProjectSettings(project.slug),
    enabled: open,
  });
  const mut = useMutation({
    mutationFn: (iteration: IterationSettings) =>
      patchProjectSettings(project.slug, { iteration }),
    onSuccess: (data) => {
      qc.setQueryData(["project-settings", project.slug], data);
      toast.success("Settings saved");
    },
    onError: (err) => {
      const msg = err instanceof ApiError
        ? err.problem.detail ?? err.problem.title
        : (err as Error).message;
      toast.error("Could not save settings", { description: msg });
    },
  });

  // Local form state; seeded from the loaded iteration block.
  const [form, setForm] = useState<IterationSettings>({});
  useEffect(() => {
    if (q.data) {
      setForm(q.data.iteration ?? {});
    }
  }, [q.data]);

  const loaded = q.data?.iteration ?? {};
  const dirty = JSON.stringify(form) !== JSON.stringify(loaded);

  const fields: Array<
    | { key: keyof IterationSettings; label: string; type: "number"; step?: number; min?: number; max?: number; hint?: string }
    | { key: keyof IterationSettings; label: string; type: "bool"; hint?: string }
    | { key: keyof IterationSettings; label: string; type: "text"; hint?: string }
  > = [
    { key: "steps_per_iter", label: "steps_per_iter (training)", type: "number", min: 100, max: 500_000, hint: "rsl_rl max_iterations for mjlab" },
    { key: "primary_metric", label: "primary_metric", type: "text", hint: "e.g. mean_return, max_episode_length" },
    { key: "rollout_episodes", label: "rollout_episodes", type: "number", min: 1, max: 32, hint: "episodes captured per iter" },
    { key: "auto_adjust_physics", label: "auto_adjust_physics", type: "bool", hint: "§7.4 — suggest MJCF edit on severe realism verdict" },
    { key: "max_episode_steps", label: "max_episode_steps (rollout)", type: "number", min: 50, max: 5000, hint: "env steps per rollout episode" },
    { key: "playback_speed", label: "playback_speed", type: "number", step: 0.1, min: 0.1, max: 10, hint: "1.0 = real-time; 0.5 = slow-mo" },
    { key: "render_every", label: "render_every", type: "number", min: 1, max: 100, hint: "capture every Nth step (advanced)" },
    { key: "rollout_fps", label: "rollout_fps (override)", type: "number", step: 1, min: 1, max: 240, hint: "force playback fps (blank = auto)" },
    { key: "seed", label: "seed", type: "number", min: 0, hint: "base RNG seed; iter N uses seed + N" },
    { key: "early_stop_enabled", label: "early_stop_enabled", type: "bool", hint: "§Ship-9a: flip off for overnight runs where metric dips may mask real progress" },
    { key: "early_stop_patience", label: "early_stop_patience", type: "number", min: 1, max: 100, hint: "consecutive no-improvement iters before truncation (default 3)" },
  ];

  const update = (key: keyof IterationSettings, raw: string | boolean) => {
    setForm((prev) => {
      const next = { ...prev };
      if (typeof raw === "boolean") {
        (next[key] as unknown) = raw;
        return next;
      }
      if (raw === "") {
        (next[key] as unknown) = null;
        return next;
      }
      // Numeric vs text — pick based on field def.
      const def = fields.find((f) => f.key === key);
      if (def && def.type === "number") {
        const n = Number(raw);
        (next[key] as unknown) = Number.isFinite(n) ? n : null;
      } else {
        (next[key] as unknown) = raw;
      }
      return next;
    });
  };

  return (
    <section className="space-y-2 rounded-md border bg-muted/10 p-3 text-xs">
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
          Iteration settings (config.toml)
        </div>
        {q.isLoading && <Loader2 className="h-3 w-3 animate-spin" />}
      </div>
      <p className="text-[11px] text-muted-foreground">
        Persistent project defaults — used whenever a run omits the override.
      </p>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {fields.map((f) => (
          <div key={String(f.key)} className="grid gap-1">
            <Label htmlFor={`set-${String(f.key)}`} className="text-[11px]">
              {f.label}
            </Label>
            {f.type === "bool" ? (
              <select
                id={`set-${String(f.key)}`}
                value={
                  form[f.key] == null
                    ? "unset"
                    : (form[f.key] as boolean)
                    ? "true"
                    : "false"
                }
                onChange={(e) => {
                  const v = e.target.value;
                  update(f.key, v === "unset" ? ("" as unknown as boolean) : v === "true");
                  if (v === "unset") {
                    setForm((prev) => ({ ...prev, [f.key]: null }));
                  }
                }}
                disabled={mut.isPending || q.isLoading}
                className="rounded border bg-background px-2 py-1 text-xs"
              >
                <option value="unset">(unset / default)</option>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <Input
                id={`set-${String(f.key)}`}
                type={f.type === "number" ? "number" : "text"}
                step={"step" in f ? f.step : undefined}
                min={"min" in f ? f.min : undefined}
                max={"max" in f ? f.max : undefined}
                value={
                  form[f.key] == null
                    ? ""
                    : String(form[f.key] as number | string)
                }
                onChange={(e) => update(f.key, e.target.value)}
                placeholder="(unset)"
                disabled={mut.isPending || q.isLoading}
                className="h-7 text-xs"
              />
            )}
            {f.hint && (
              <p className="text-[10px] text-muted-foreground">{f.hint}</p>
            )}
          </div>
        ))}
      </div>
      <div className="flex items-center justify-end gap-2 pt-1">
        <Button
          size="sm"
          variant="outline"
          onClick={() => setForm(loaded)}
          disabled={!dirty || mut.isPending}
        >
          Revert
        </Button>
        <Button
          size="sm"
          onClick={() => mut.mutate(form)}
          disabled={!dirty || mut.isPending}
        >
          {mut.isPending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Save className="h-3.5 w-3.5" />
          )}
          Save
        </Button>
      </div>
    </section>
  );
}

function SummarySection({ project }: { project: ProjectDetail }) {
  const adapterShort = project.adapter_class.split(".").slice(-1)[0];
  const taskId =
    typeof project.adapter_config?.task_id === "string"
      ? (project.adapter_config.task_id as string)
      : typeof project.adapter_config?.env_id === "string"
      ? (project.adapter_config.env_id as string)
      : null;
  const numEnvs =
    typeof project.adapter_config?.num_envs === "number"
      ? project.adapter_config.num_envs
      : null;
  const device =
    typeof project.adapter_config?.device === "string"
      ? (project.adapter_config.device as string)
      : null;

  const rows: Array<[string, string]> = [
    ["slug", project.slug],
    ["adapter", adapterShort],
    ...(taskId ? [["task", taskId] as [string, string]] : []),
    ...(numEnvs != null ? [["num_envs", String(numEnvs)]] : []),
    ...(device ? [["device", device]] as [string, string][] : []),
    ...(project.library_slug
      ? [["library", project.library_slug] as [string, string]]
      : []),
    ["created", project.created_at],
  ];

  return (
    <section className="space-y-1.5 rounded-md border bg-muted/20 p-3 text-xs">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        Summary
      </div>
      <dl className="grid grid-cols-[110px_1fr] gap-x-3 gap-y-0.5 font-mono">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="truncate">{v}</dd>
          </div>
        ))}
      </dl>
      {project.adapter_unavailable && (
        <p className="text-[11px] text-amber-700">
          Adapter is coming-soon — training disabled.
        </p>
      )}
    </section>
  );
}

function DangerZone({
  project,
  onDeleted,
}: {
  project: ProjectDetail;
  onDeleted: () => void;
}) {
  const [confirm, setConfirm] = useState("");
  const del = useDeleteProject();
  const nav = useNavigate();
  const matches = confirm.trim() === project.slug;

  const onDelete = () => {
    if (!matches) return;
    del.mutate(project.slug, {
      onSuccess: () => {
        toast.success("Project deleted", {
          description: `Removed ${project.slug} + all runs.`,
        });
        onDeleted();
        nav("/projects");
      },
      onError: (err) => {
        const msg =
          err instanceof ApiError
            ? err.problem.detail ?? err.problem.title
            : (err as Error).message;
        toast.error("Could not delete project", { description: msg });
      },
    });
  };

  return (
    <section className="space-y-2 rounded-md border border-rose-300/50 bg-rose-50/40 p-3 text-xs">
      <div className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-rose-700">
        <Trash2 className="h-3 w-3" />
        Danger zone
      </div>
      <p className="text-muted-foreground">
        Deleting removes the project directory + all runs. The sculptor
        library source is untouched. Type the slug to confirm.
      </p>
      <div className="space-y-1.5">
        <Label htmlFor="confirm-slug" className="text-[11px]">
          Type <code className="font-mono">{project.slug}</code> to confirm
        </Label>
        <Input
          id="confirm-slug"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          placeholder={project.slug}
          className="font-mono text-xs"
          disabled={del.isPending}
        />
      </div>
      <DialogFooter>
        <Button
          variant="destructive"
          size="sm"
          onClick={onDelete}
          disabled={!matches || del.isPending}
        >
          {del.isPending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Trash2 className="h-3.5 w-3.5" />
          )}
          Delete project
        </Button>
      </DialogFooter>
    </section>
  );
}

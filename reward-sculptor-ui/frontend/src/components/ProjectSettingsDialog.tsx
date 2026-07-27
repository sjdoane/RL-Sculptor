import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, Field, IconBtn, Modal } from "@/components/rs/primitives";
import { useDeleteProject } from "@/hooks/useProjects";
import {
  ApiError,
  getProjectSettings,
  patchProjectSettings,
  type IterationSettings,
} from "@/lib/api";
import type { ProjectDetail } from "@/lib/types";

/** Project settings dialog (M7 Phase 7c). Opens from a gear icon in
 * the ProjectDetail header. Surfaces:
 *   - project summary (read-only key/value pairs).
 *   - iteration settings (editable config.toml [iteration] block).
 *   - danger zone: delete project (type-to-confirm). */
export function ProjectSettingsDialog({
  project,
}: {
  project: ProjectDetail;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <IconBtn icon="settings" label="Project settings" onClick={() => setOpen(true)} />
      {open && (
        <Modal
          icon="settings"
          title="Project settings"
          subtitle="Inspect adapter + library config. Manage destructive actions from the danger zone."
          onClose={() => setOpen(false)}
          footer={<Btn kind="primary" onClick={() => setOpen(false)}>Done</Btn>}
        >
          <SummarySection project={project} />
          <IterationSettingsSection project={project} open={open} />
          <DangerZone project={project} onDeleted={() => setOpen(false)} />
        </Modal>
      )}
    </>
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
    { key: "eval_seeds", label: "eval_seeds", type: "number", min: 1, max: 10, hint: "rollouts per iter; keep-best selects on the MEDIAN (1 = single-roll legacy)" },
    { key: "progress_epsilon", label: "progress_epsilon", type: "number", step: 0.00001, min: 0, max: 0.1, hint: "noise band for progress tie-breaks (default 1e-5; 0 = strict)" },
    { key: "fresh_eval_seeds", label: "fresh_eval_seeds", type: "number", min: 0, max: 10, hint: "end-of-run re-rolls of the kept best on held-out seeds (0 = off)" },
    { key: "hack_income_screen", label: "hack_income_screen", type: "bool", hint: "reject edits that make a caught exploit MORE profitable" },
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
    <section style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", background: "var(--surface-strong)", padding: 13 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <span className="rs-caption" style={{ margin: 0 }}>Iteration settings (config.toml)</span>
        {q.isLoading && <Icon name="loader" size={12} className="rs-spin" />}
      </div>
      <p className="rs-hintline" style={{ margin: "0 0 10px" }}>Persistent project defaults — used whenever a run omits the override.</p>
      <div className="rs-row2">
        {fields.map((f) => (
          <Field key={String(f.key)} label={f.label} htmlFor={`set-${String(f.key)}`}>
            {f.type === "bool" ? (
              <div className="rs-select" style={{ display: "flex" }}>
                <select
                  id={`set-${String(f.key)}`}
                  style={{ width: "100%" }}
                  value={form[f.key] == null ? "unset" : (form[f.key] as boolean) ? "true" : "false"}
                  onChange={(e) => {
                    const v = e.target.value;
                    update(f.key, v === "unset" ? ("" as unknown as boolean) : v === "true");
                    if (v === "unset") {
                      setForm((prev) => ({ ...prev, [f.key]: null }));
                    }
                  }}
                  disabled={mut.isPending || q.isLoading}
                >
                  <option value="unset">(unset / default)</option>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </div>
            ) : (
              <input
                id={`set-${String(f.key)}`}
                className="rs-input mono"
                type={f.type === "number" ? "number" : "text"}
                step={"step" in f ? f.step : undefined}
                min={"min" in f ? f.min : undefined}
                max={"max" in f ? f.max : undefined}
                value={form[f.key] == null ? "" : String(form[f.key] as number | string)}
                onChange={(e) => update(f.key, e.target.value)}
                placeholder="(unset)"
                disabled={mut.isPending || q.isLoading}
              />
            )}
            {f.hint && <p className="rs-hintline">{f.hint}</p>}
          </Field>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8, paddingTop: 12 }}>
        <Btn kind="quiet" size="sm" onClick={() => setForm(loaded)} disabled={!dirty || mut.isPending}>Revert</Btn>
        <Btn kind="primary" size="sm" icon={mut.isPending ? "loader" : "check"} onClick={() => mut.mutate(form)} disabled={!dirty || mut.isPending}>Save</Btn>
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
    ...(taskId ? ([["task", taskId]] as [string, string][]) : []),
    ...(numEnvs != null ? ([["num_envs", String(numEnvs)]] as [string, string][]) : []),
    ...(device ? ([["device", device]] as [string, string][]) : []),
    ...(project.library_slug ? ([["library", project.library_slug]] as [string, string][]) : []),
    ["created", project.created_at],
  ];

  return (
    <section style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", background: "var(--surface-strong)", padding: 13 }}>
      <span className="rs-caption">Summary</span>
      <dl className="mono" style={{ display: "grid", gridTemplateColumns: "110px 1fr", columnGap: 12, rowGap: 3, margin: 0, fontSize: 12 }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <dt style={{ color: "var(--rs-muted)" }}>{k}</dt>
            <dd style={{ margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v}</dd>
          </div>
        ))}
      </dl>
      {project.adapter_unavailable && (
        <p style={{ marginTop: 8, fontSize: 11.5, color: "var(--st-amber-fg)" }}>Adapter is coming-soon — training disabled.</p>
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
        toast.success("Moved to Trash", {
          description: `${project.slug} is recoverable from Settings → Trash.`,
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
    <section style={{ border: "1px solid color-mix(in srgb, var(--st-rose) 35%, transparent)", borderRadius: "var(--radius-md)", background: "var(--st-rose-bg)", padding: 13 }}>
      <span className="rs-caption" style={{ color: "var(--st-rose-fg)", display: "inline-flex", alignItems: "center", gap: 6 }}>
        <Icon name="trash" size={12} /> Danger zone
      </span>
      <p style={{ margin: "0 0 10px", fontSize: 12, color: "var(--rs-muted)" }}>
        Moves the project directory + all runs to Trash — recoverable from Settings → Trash. The sculptor library source is untouched. Type the slug to confirm.
      </p>
      <Field label={<>Type <code className="mono">{project.slug}</code> to confirm</>} htmlFor="confirm-slug">
        <input
          id="confirm-slug"
          className="rs-input mono"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          placeholder={project.slug}
          disabled={del.isPending}
        />
      </Field>
      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
        <Btn kind="danger" size="sm" icon={del.isPending ? "loader" : "trash"} onClick={onDelete} disabled={!matches || del.isPending}>
          Move to Trash
        </Btn>
      </div>
    </section>
  );
}

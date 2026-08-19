import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, Field, IconBtn, Modal } from "@/components/rs/primitives";
import { useDeleteProject } from "@/hooks/useProjects";
import {
  ApiError,
  editProjectEnvSpecTrain,
  getProjectEnvSpec,
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
          <EnvSpecTrainSection project={project} open={open} />
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
  // Raw text for the comma-separated list fields. Parsing on every keystroke
  // and re-joining for display would swallow the separator the instant you
  // typed it, so what the user is typing is held verbatim here and parsed
  // into an array only on the way into `form`.
  const [listDraft, setListDraft] = useState<Record<string, string>>({});
  useEffect(() => {
    if (q.data) {
      const iteration = q.data.iteration ?? {};
      setForm(iteration);
      setListDraft(
        Object.fromEntries(
          Object.entries(iteration)
            .filter(([, v]) => Array.isArray(v))
            .map(([k, v]) => [k, (v as string[]).join(", ")]),
        ),
      );
    }
  }, [q.data]);

  const loaded = q.data?.iteration ?? {};
  const dirty = JSON.stringify(form) !== JSON.stringify(loaded);

  const fields: Array<
    | { key: keyof IterationSettings; label: string; type: "number"; step?: number; min?: number; max?: number; hint?: string }
    | { key: keyof IterationSettings; label: string; type: "bool"; hint?: string }
    | { key: keyof IterationSettings; label: string; type: "text"; hint?: string }
    | { key: keyof IterationSettings; label: string; type: "list"; hint?: string }
  > = [
    { key: "steps_per_iter", label: "steps_per_iter (training)", type: "number", min: 100, max: 500_000, hint: "rsl_rl max_iterations for mjlab" },
    { key: "primary_metric", label: "primary_metric", type: "text", hint: "e.g. mean_return, max_episode_length" },
    { key: "behavior_metrics", label: "behavior_metrics", type: "list", hint: "comma-separated; which behavior metrics each iteration computes (blank = adapter default)" },
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

  const updateList = (key: keyof IterationSettings, raw: string) => {
    setListDraft((d) => ({ ...d, [String(key)]: raw }));
    const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
    setForm((prev) => ({ ...prev, [key]: parts.length > 0 ? parts : null }));
  };

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
      <p className="rs-hintline" style={{ margin: "0 0 10px" }}>
        Persistent project defaults — used whenever a run omits the override.
        Note that New run → <strong>Advanced</strong> pre-fills its fields with
        adapter defaults rather than leaving them blank, so in practice a
        launch overrides most of these unless you clear the field. Same knobs,
        different names: <code className="mono">steps_per_iter</code> here is
        <code className="mono"> training_iterations</code> there.
      </p>
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
            ) : f.type === "list" ? (
              <input
                id={`set-${String(f.key)}`}
                className="rs-input mono"
                type="text"
                value={listDraft[String(f.key)] ?? ""}
                onChange={(e) => updateList(f.key, e.target.value)}
                placeholder="(unset)"
                disabled={mut.isPending || q.isLoading}
              />
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

/** Short, plain-language notes for the knobs whose effect is not obvious
 *  from the name. Everything else is rendered generically — the editable
 *  set comes from the backend so this list can lag without hiding a key. */
const ENV_TRAIN_HINTS: Record<string, string> = {
  friction_range:
    "Train-only foot/ground friction randomization; evaluation remains at the " +
    "nominal setting. Keep mild ranges near the model's nominal friction.",
  entropy_coef_scale:
    "Multiplies PPO's entropy bonus. Above 1 the policy's action-noise std " +
    "climbs all run, and the action-rate penalty grows with the square of it " +
    "— at 3.0 that penalty overtook the task reward and mean return fell from " +
    "358 to 38. Raise it only for short explosive skills.",
  min_base_height_termination_m:
    "Ends the episode when the base drops below this world height. Pairs with " +
    "RSI resets; too high and it kills legitimate crouches.",
  fell_over_termination:
    "Train-only. Turn off for a stage that RESETS in a fallen pose (get-up), " +
    "or every episode ends at step 0.",
  com_offset_m: "Per-link centre-of-mass jitter for sim2real. Keep small (0.02–0.05).",
};

/** The environment the loop trains under. Read-only until now: these decide
 *  whether a run can succeed, and the only way to change one was to wait for
 *  the diagnoser to try it between iterations. */
export function EnvSpecTrainSection({
  project, open,
}: { project: ProjectDetail; open: boolean }) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["project-env-spec", project.slug],
    queryFn: () => getProjectEnvSpec(project.slug),
    enabled: open,
  });
  const [form, setForm] = useState<Record<string, string>>({});
  const [adding, setAdding] = useState(false);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const loaded = (q.data?.current?.train ?? {}) as Record<string, unknown>;

  useEffect(() => {
    if (q.data) {
      setForm(Object.fromEntries(Object.entries(
        (q.data.current?.train ?? {}) as Record<string, unknown>,
      ).map(([k, v]) => [k, JSON.stringify(v)])));
    }
  }, [q.data]);

  const mut = useMutation({
    mutationFn: (edits: Array<{ parameter: string; new_value: unknown; rationale: string }>) =>
      editProjectEnvSpecTrain(project.slug, edits),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["project-env-spec", project.slug] });
      toast.success(`Saved as ${res.new_version}`, {
        description: res.applied.join(", "),
      });
      // Partial success is a normal outcome here, so say what did not land.
      for (const [param, reason] of res.rejected) {
        toast.error(`${param} not applied`, { description: reason });
      }
    },
    onError: (err) => {
      const msg = err instanceof ApiError
        ? err.problem.detail ?? err.problem.title
        : (err as Error).message;
      toast.error("Could not save environment settings", { description: msg });
    },
  });

  // The backend owns the editable-key catalog. Existing values stay inline;
  // unset values require an explicit add flow below.
  const editableKeys = q.data?.editable_train_keys ?? [];
  const keys = editableKeys.filter((k) => k in loaded);
  const unsetKeys = editableKeys.filter((k) => !(k in loaded));
  const edits = keys.flatMap((k) => {
    const raw = form[k] ?? "";
    if (raw === JSON.stringify(loaded[k])) return [];
    try {
      return [{ parameter: k, new_value: JSON.parse(raw),
                rationale: "edited from project settings" }];
    } catch {
      return [];  // mid-typing / not valid JSON yet — not an edit
    }
  });
  const malformed = keys.filter((k) => {
    const raw = form[k] ?? "";
    if (raw === JSON.stringify(loaded[k])) return false;
    try { JSON.parse(raw); return false; } catch { return true; }
  });
  let parsedNewValue: { valid: true; value: unknown } | { valid: false } | null = null;
  if (newValue.trim() !== "") {
    try {
      parsedNewValue = { valid: true, value: JSON.parse(newValue) };
    } catch {
      parsedNewValue = { valid: false };
    }
  }

  const cancelAdd = () => {
    setAdding(false);
    setNewKey("");
    setNewValue("");
  };

  const saveNewSetting = () => {
    if (!newKey || !parsedNewValue?.valid) return;
    mut.mutate(
      [{
        parameter: newKey,
        new_value: parsedNewValue.value,
        rationale: "added from project settings",
      }],
      { onSuccess: cancelAdd },
    );
  };

  if (q.data && !q.data.active) return null;

  return (
    <section style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", background: "var(--surface-strong)", padding: 13 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <span className="rs-caption" style={{ margin: 0 }}>
          Environment the loop trains under
          {q.data?.current?.meta && typeof (q.data.current.meta as { version?: string }).version === "string" && (
            <> · <code className="mono">{(q.data.current.meta as { version: string }).version}</code></>
          )}
        </span>
        {(q.isLoading || mut.isPending) && <Icon name="loader" size={12} className="rs-spin" />}
      </div>
      <p className="rs-hintline" style={{ margin: "0 0 10px" }}>
        Train-only knobs — resets, terminations and PPO exploration. Saving
        writes a new spec version and repoints <code className="mono">current</code>;
        the next run picks it up. Values are JSON, so a range is{" "}
        <code className="mono">[0.0, 1.5]</code> and a switch is{" "}
        <code className="mono">false</code>.
      </p>
      <div className="rs-row2">
        {keys.map((k) => (
          <Field key={k} label={k} htmlFor={`env-${k}`}>
            <input
              id={`env-${k}`}
              className="rs-input mono"
              value={form[k] ?? ""}
              onChange={(e) => setForm((p) => ({ ...p, [k]: e.target.value }))}
              disabled={mut.isPending || q.isLoading}
            />
            {ENV_TRAIN_HINTS[k] && <p className="rs-hintline">{ENV_TRAIN_HINTS[k]}</p>}
          </Field>
        ))}
      </div>
      {adding ? (
        <div
          style={{
            border: "1px solid var(--hairline)",
            borderRadius: "var(--radius-sm)",
            marginTop: 12,
            padding: 12,
          }}
        >
          <p className="rs-caption" style={{ margin: "0 0 8px" }}>Add train setting</p>
          <div className="rs-row2">
            <Field label="Train setting" htmlFor="env-new-key">
              <select
                id="env-new-key"
                className="rs-input mono"
                value={newKey}
                onChange={(e) => {
                  setNewKey(e.target.value);
                  setNewValue("");
                }}
                disabled={mut.isPending || q.isLoading}
              >
                <option value="" disabled>Select an unset setting</option>
                {unsetKeys.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
              {newKey && ENV_TRAIN_HINTS[newKey] && (
                <p className="rs-hintline">{ENV_TRAIN_HINTS[newKey]}</p>
              )}
            </Field>
            <Field label="JSON value" htmlFor="env-new-value">
              <input
                id="env-new-value"
                className="rs-input mono"
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                placeholder="Enter a value, for example [0.8, 1.2]"
                aria-invalid={parsedNewValue?.valid === false}
                disabled={!newKey || mut.isPending || q.isLoading}
              />
              <p className="rs-hintline">
                This is validated by the same bounds and whole-spec checks as automated edits.
              </p>
            </Field>
          </div>
          {parsedNewValue?.valid === false && (
            <p
              className="rs-hintline"
              role="status"
              aria-live="polite"
              style={{ color: "var(--st-amber)", marginTop: 8 }}
            >
              Enter a valid JSON value before saving.
            </p>
          )}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8, paddingTop: 10 }}>
            <Btn kind="quiet" size="sm" onClick={cancelAdd} disabled={mut.isPending}>
              Cancel
            </Btn>
            <Btn
              kind="primary"
              size="sm"
              icon={mut.isPending ? "loader" : "check"}
              onClick={saveNewSetting}
              disabled={!newKey || !parsedNewValue?.valid || mut.isPending}
            >
              Save setting
            </Btn>
          </div>
        </div>
      ) : unsetKeys.length > 0 ? (
        <div style={{ paddingTop: 10 }}>
          <Btn
            kind="quiet"
            size="sm"
            icon="plus"
            onClick={() => setAdding(true)}
            disabled={mut.isPending || q.isLoading}
          >
            Add train setting
          </Btn>
        </div>
      ) : null}
      {malformed.length > 0 && (
        <p className="rs-hintline" role="status" aria-live="polite" style={{ color: "var(--st-amber)", marginTop: 8 }}>
          Not valid JSON yet: {malformed.join(", ")}
        </p>
      )}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8, paddingTop: 12 }}>
        <Btn kind="quiet" size="sm" disabled={edits.length === 0 || mut.isPending}
             onClick={() => setForm(Object.fromEntries(
               Object.entries(loaded).map(([k, v]) => [k, JSON.stringify(v)])))}>
          Revert
        </Btn>
        <Btn kind="primary" size="sm" icon={mut.isPending ? "loader" : "check"}
             disabled={edits.length === 0 || mut.isPending}
             onClick={() => mut.mutate(edits)}>
          Save
        </Btn>
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

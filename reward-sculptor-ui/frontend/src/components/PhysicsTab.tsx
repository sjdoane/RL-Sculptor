import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn } from "@/components/rs/primitives";
import { MonacoDiffLazy, MonacoLazy } from "@/components/MonacoLazy";
import { useJob } from "@/hooks/useJob";
import {
  usePhysics,
  usePhysicsPromptEdit,
  usePhysicsRematerialize,
} from "@/hooks/usePhysics";
import {
  ApiError,
  applyMotorLimits,
  extractDatasheetPdf,
  type MotorLimitsRequest,
  type MotorSpec,
} from "@/lib/api";
import type { MjcfSummary, ProjectDetail } from "@/lib/types";

const _MOTOR_LIMITS_TEMPLATE = `Update the MJCF actuator forceranges and joint damping/armature/frictionloss to match my motor datasheet. Apply the standard workflow:
  - forcerange = ±peak_torque (stall torque from datasheet)
  - armature = rotor_inertia × gear_ratio²
  - damping = viscous_friction (from no-load current × motor torque constant / no-load speed)
  - frictionloss = Coulomb breakaway torque (stall breakaway, reflected through gear ratio)

Per-joint specs (fill in from your datasheet; delete rows that don't apply):
  joint_name_1:
    peak_torque: ___ Nm
    peak_speed: ___ rad/s
    gear_ratio: ___
    rotor_inertia: ___ kg·m^2
    no_load_current: ___ A (optional; for damping estimate)
    torque_constant: ___ Nm/A (optional; for damping estimate)

  joint_name_2:
    (same structure)

If the robot has series-elastic actuators, also set spring stiffness + damping on the relevant <tendon> or coupled joint using the SEA characterization method from arXiv:1902.05346 (max torque transmissibility).

Cite 2312.17507 for why forcerange enforcement matters (prevents the policy exploiting infinite-torque assumptions), 1901.08652 for the actuator-inertia-damping workflow, and 2410.08650 for the extended-friction model if frictionloss isn't already set.`;

export function PhysicsTab({ slug, project: _project }: { slug: string; project: ProjectDetail }) {
  const phys = usePhysics(slug);
  const [prompt, setPrompt] = useState("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [lastRejection, setLastRejection] = useState<RejectionState | null>(null);
  const promptRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    try {
      const pending = sessionStorage.getItem("pendingPhysicsPrompt");
      if (pending && pending.trim().length > 0) {
        setPrompt(pending);
        sessionStorage.removeItem("pendingPhysicsPrompt");
        setTimeout(() => promptRef.current?.focus(), 0);
      }
    } catch {
      /* sessionStorage unavailable */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const mutate = usePhysicsPromptEdit(slug);
  const rematerialize = usePhysicsRematerialize(slug);
  const job = useJob(activeJobId ?? undefined, { intervalMs: 1500 });
  const qc = useQueryClient();

  useEffect(() => {
    if (!job.data || !activeJobId) return;
    if (job.data.status === "completed") {
      const result = (job.data.result as Record<string, any>) ?? {};
      const committed = Boolean(result.committed);
      const summary = (result.summary as string) ?? "";
      if (committed) {
        toast.success("Physics edit committed", { description: summary.slice(0, 120) });
        setLastRejection(null);
        setPrompt("");
      } else {
        setLastRejection({
          reason: (result.rejected_reason as string) || summary || "unknown",
          rejectedAt: (result.rejected_at as RejectionState["rejectedAt"]) ?? null,
          oldXml: (result.old_xml as string) ?? "",
          newXml: (result.new_xml as string) ?? "",
          claudeOutputRaw: (result.claude_output_raw as string | null) ?? null,
          summary,
        });
        toast.error("Edit rejected — see card below the prompt");
      }
      qc.invalidateQueries({ queryKey: ["physics", slug] });
      setActiveJobId(null);
    } else if (job.data.status === "errored") {
      toast.error("Physics edit failed", { description: job.data.error ?? "see /jobs for details" });
      setActiveJobId(null);
    }
  }, [job.data, activeJobId, slug, qc]);

  const inFlight =
    Boolean(activeJobId) && job.data?.status !== "completed" && job.data?.status !== "errored";

  const handleRematerialize = () => {
    rematerialize.mutate(undefined, {
      onSuccess: () => {
        toast.success("MJCF re-materialized", { description: "Assets copied from the library source." });
        qc.invalidateQueries({ queryKey: ["physics", slug] });
      },
      onError: (err) => {
        const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
        toast.error("Re-materialize failed", { description: msg });
      },
    });
  };

  const submit = () => {
    if (prompt.trim().length < 3) {
      toast.error("Prompt too short", { description: "Describe the physics change (≥ 3 chars)." });
      return;
    }
    mutate.mutate(
      { prompt: prompt.trim() },
      {
        onSuccess: (j) => {
          setActiveJobId(j.job_id);
          toast.success("Claude is editing the MJCF…", { description: "~30-90 s for a small change" });
        },
        onError: (err) => {
          const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
          toast.error("Could not start physics edit", { description: msg });
        },
      },
    );
  };

  if (phys.isLoading) {
    return <div className="rs-scroll"><div className="rs-pad"><p className="rs-sub">Loading MJCF…</p></div></div>;
  }
  if (phys.error) {
    const err = phys.error as Error;
    const detail = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
    return (
      <div className="rs-scroll">
        <div className="rs-pad">
          <div className="rs-banner warn">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow"><b>Physics editor unavailable.</b> {detail}</span>
          </div>
        </div>
      </div>
    );
  }
  if (!phys.data) return null;
  const summary = phys.data.summary;

  return (
    <div className="rs-scroll">
      <div className="rs-pad rs-vgap-24">
        <div>
          <div className="rs-eyebrow">MJCF model</div>
          <h2 className="rs-h2" style={{ marginTop: 6 }}>Physics</h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(300px,360px)", gap: 18, alignItems: "start" }}>
          <section className="rs-vgap-16" style={{ minWidth: 0 }}>
            {/* Prompt hero */}
            <div className="rs-prompt">
              <div className="rs-flex rs-gap-8" style={{ marginBottom: 8 }}>
                <Icon name="sparkles" size={15} color="var(--rs-primary)" />
                <span className="rs-eyebrow">Edit physics with Claude</span>
              </div>
              <textarea
                ref={promptRef}
                value={prompt}
                onChange={(e) => { setPrompt(e.target.value); if (lastRejection) setLastRejection(null); }}
                placeholder={"e.g. Increase hip joint damping by 50% to reduce oscillation · Drop timestep to 0.001 · Add a parallel-elastic spring across the knee"}
                rows={4}
                maxLength={2000}
                disabled={inFlight}
                aria-label="Physics change prompt"
              />
              <div className="rs-prompt-foot">
                <span className="rs-sub" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
                  <Icon name="info" size={13} />
                  source <code className="mono">{phys.data.mjcf_source_kind}</code> · edits regenerate + re-validate the MJCF
                </span>
                <div className="rs-flex rs-gap-8">
                  <Btn kind="ghost" size="sm" disabled={inFlight} onClick={() => setPrompt(_MOTOR_LIMITS_TEMPLATE)} title="Fill the prompt with a motor-limit template.">
                    Motor template
                  </Btn>
                  <Btn kind="primary" size="sm" icon={inFlight ? "loader" : "sparkles"} onClick={submit} disabled={inFlight}>
                    {inFlight ? "Claude is editing…" : "Prompt Claude"}
                  </Btn>
                </div>
              </div>
            </div>

            {lastRejection && (
              <RejectionCard state={lastRejection} onRetry={() => { setLastRejection(null); promptRef.current?.focus(); }} />
            )}

            {summary.parse_error && (
              <div className="rs-card rs-card-pad rs-vgap-8">
                <div className="rs-banner warn">
                  <Icon name="alert-triangle" size={17} />
                  <span className="rs-grow">
                    <b>MJCF won't parse.</b> <span className="mono" style={{ fontSize: 11 }}>{summary.parse_error}</span> — usually missing mesh/texture assets. Re-materialize re-fetches the library source.
                  </span>
                </div>
                <div className="rs-flex" style={{ justifyContent: "flex-end" }}>
                  <Btn kind="ghost" size="sm" icon={rematerialize.isPending ? "loader" : "refresh-cw"} onClick={handleRematerialize} disabled={rematerialize.isPending}>
                    {rematerialize.isPending ? "Re-materializing…" : "Re-materialize MJCF"}
                  </Btn>
                </div>
              </div>
            )}

            <MotorLimitsCard slug={slug} summary={summary} />

            {/* Current MJCF (Monaco) */}
            <div className="rs-code">
              <div className="rs-code-bar">
                <span className="rs-code-file">
                  <Icon name="file-code" size={14} color="var(--rs-primary)" />
                  {phys.data.mjcf_path ? phys.data.mjcf_path.split("/").slice(-2).join("/") : "model.xml"}
                </span>
                <span className="rs-sub" style={{ fontSize: 11 }}>{phys.data.xml_source.split("\n").length} lines · read-only</span>
              </div>
              <MonacoLazy value={phys.data.xml_source} readOnly height={420} language="xml" />
            </div>
          </section>

          <aside className="rs-vgap-16">
            <SummaryPanel summary={summary} />
          </aside>
        </div>
      </div>
    </div>
  );
}

function SummaryPanel({ summary }: { summary: MjcfSummary }) {
  return (
    <>
      <div className="rs-card">
        <div className="rs-card-head"><div className="rs-card-title"><Icon name="gauge" size={16} />MJCF summary</div></div>
        <div className="rs-card-pad">
          <div className="rs-sysgrid">
            <div className="rs-sysitem"><span className="lab">joints</span><span className="val" style={{ fontSize: 20 }}>{summary.joints.length}</span></div>
            <div className="rs-sysitem"><span className="lab">actuators</span><span className="val" style={{ fontSize: 20 }}>{summary.actuators.length}</span></div>
            <div className="rs-sysitem"><span className="lab">geoms</span><span className="val" style={{ fontSize: 20 }}>{(summary.geoms ?? []).length}</span></div>
          </div>
          <hr className="rs-divider" style={{ margin: "14px 0" }} />
          <div className="rs-kv">
            <div className="k">timestep</div><div className="v">{fmt(summary.timestep)}</div>
            <div className="k">gravity</div><div className="v">{fmtVec(summary.gravity)}</div>
            <div className="k">integrator</div><div className="v">{summary.integrator ?? "—"}</div>
            <div className="k">solver</div><div className="v">{summary.solver ?? "—"}</div>
          </div>
        </div>
      </div>

      <div className="rs-card">
        <div className="rs-card-head"><div className="rs-card-title"><Icon name="cpu" size={16} />Joints</div><span className="rs-num" style={{ color: "var(--rs-muted)", fontSize: 12 }}>{summary.joints.length}</span></div>
        <div className="rs-card-pad" style={{ maxHeight: 220, overflowY: "auto", paddingTop: 8 }}>
          {summary.joints.length === 0 ? (
            <p className="rs-sub" style={{ fontStyle: "italic" }}>(none)</p>
          ) : (
            <div className="rs-kv">
              {summary.joints.slice(0, 40).map((j) => (
                <div key={j.name} style={{ display: "contents" }}>
                  <div className="k" style={{ textTransform: "none" }}>{j.name}</div>
                  <div className="v">d={fmt(j.damping)} a={fmt(j.armature)}</div>
                </div>
              ))}
            </div>
          )}
          {summary.joints.length > 40 && <p className="rs-sub" style={{ fontSize: 11, marginTop: 6 }}>+ {summary.joints.length - 40} more</p>}
        </div>
      </div>

      <div className="rs-card">
        <div className="rs-card-head"><div className="rs-card-title"><Icon name="layers" size={16} />Actuators</div><span className="rs-num" style={{ color: "var(--rs-muted)", fontSize: 12 }}>{summary.actuators.length}</span></div>
        <div className="rs-card-pad" style={{ maxHeight: 220, overflowY: "auto", paddingTop: 8 }}>
          {summary.actuators.length === 0 ? (
            <p className="rs-sub" style={{ fontStyle: "italic" }}>(none)</p>
          ) : (
            <div className="rs-kv">
              {summary.actuators.slice(0, 40).map((a) => (
                <div key={a.name} style={{ display: "contents" }}>
                  <div className="k" style={{ textTransform: "none" }}>{a.name}</div>
                  <div className="v">gear={fmt(a.gear)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

function fmt(n: number | null | undefined): string {
  if (n == null) return "—";
  return typeof n === "number" ? n.toString() : String(n);
}
function fmtVec(vs: number[] | null | undefined): string {
  if (!vs) return "—";
  return `[${vs.map((v) => v.toFixed(3)).join(", ")}]`;
}

type RejectionState = {
  reason: string;
  rejectedAt: "parse" | "claude_rejected" | "mujoco_validate" | null;
  oldXml: string;
  newXml: string;
  claudeOutputRaw: string | null;
  summary: string;
};

const REJECTED_AT_LABEL: Record<NonNullable<RejectionState["rejectedAt"]>, string> = {
  parse: "response format",
  claude_rejected: "claude refused",
  mujoco_validate: "mujoco rejected",
};

function RejectionCard({ state, onRetry }: { state: RejectionState; onRetry: () => void }) {
  const tag = state.rejectedAt ? REJECTED_AT_LABEL[state.rejectedAt] : "unknown";
  return (
    <div className="rs-why">
      <div className="rs-why-head" style={{ cursor: "default" }}>
        <Icon name="alert-triangle" size={16} />
        Edit rejected
        <span className="rs-failchip" style={{ marginLeft: 8 }}>{tag}</span>
      </div>
      <div className="rs-card-pad rs-vgap-8" style={{ background: "var(--surface-card)" }}>
        <p className="mono" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 11, color: "var(--body)", margin: 0 }}>{state.reason}</p>
        <details>
          <summary style={{ cursor: "pointer", fontSize: 13, fontWeight: 500 }}>View what Claude wrote</summary>
          <div style={{ marginTop: 8, border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
            {state.rejectedAt === "parse" ? (
              <pre className="mono" style={{ maxHeight: 260, overflow: "auto", whiteSpace: "pre-wrap", padding: 12, fontSize: 11, margin: 0 }}>
                {state.claudeOutputRaw ?? "(no raw output captured)"}
              </pre>
            ) : (
              <MonacoDiffLazy original={state.oldXml} modified={state.newXml} language="xml" height={320} />
            )}
          </div>
        </details>
        <div className="rs-flex" style={{ justifyContent: "flex-end" }}>
          <Btn kind="ghost" size="sm" onClick={onRetry}>Retry with a different prompt</Btn>
        </div>
      </div>
    </div>
  );
}

// ── §Ship-9b: motor-specs form (logic preserved verbatim) ────────────
function MotorLimitsCard({ slug, summary }: { slug: string; summary: MjcfSummary }) {
  const [open, setOpen] = useState(false);
  const [specs, setSpecs] = useState<Record<string, MotorSpec>>({});
  const actuators = useMemo(() => (summary.actuators ?? []).map((a) => a.name), [summary.actuators]);
  const rowNames = useMemo(() => {
    const set = new Set<string>(actuators);
    for (const k of Object.keys(specs)) set.add(k);
    return Array.from(set);
  }, [actuators, specs]);

  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const extract = useMutation({
    mutationFn: (pdf: File) => extractDatasheetPdf(slug, pdf, actuators),
    onSuccess: (data) => {
      const incoming = data.motors ?? {};
      const nMotors = Object.keys(incoming).length;
      if (nMotors === 0) {
        toast.info("Datasheet uploaded but no motor specs were extracted", { description: "Check that the PDF has text (not a scanned image)." });
        return;
      }
      setSpecs((prev) => {
        const merged: Record<string, MotorSpec> = { ...prev };
        for (const [name, spec] of Object.entries(incoming)) merged[name] = spec;
        return merged;
      });
      setOpen(true);
      toast.success(`Extracted ${nMotors} motor spec${nMotors === 1 ? "" : "s"}`, { description: "Review + edit before hitting Apply." });
    },
    onError: (err) => {
      const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
      toast.error("Datasheet extraction failed", { description: msg });
    },
  });

  const apply = useMutation({
    mutationFn: (body: MotorLimitsRequest) => applyMotorLimits(slug, body),
    onSuccess: (job) => {
      toast.success("Motor-limits edit started", { description: `job ${job.job_id.slice(0, 8)}… — watch for a commit toast` });
      setTimeout(() => { qc.invalidateQueries({ queryKey: ["physics", slug] }); }, 2000);
    },
    onError: (err) => {
      const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
      toast.error("Motor-limits edit failed to start", { description: msg });
    },
  });

  const update = (name: string, key: keyof MotorSpec, raw: string) => {
    setSpecs((prev) => {
      const row: MotorSpec = { ...(prev[name] ?? {}) };
      if (raw === "") {
        delete (row as Record<string, unknown>)[key];
      } else if (key === "notes") {
        row.notes = raw;
      } else {
        const n = Number(raw);
        if (!Number.isFinite(n) || n <= 0) delete (row as Record<string, unknown>)[key];
        else (row as Record<string, unknown>)[key] = n;
      }
      return { ...prev, [name]: row };
    });
  };

  const nonEmptyCount = Object.values(specs).filter((s) =>
    s && (s.peak_torque_nm != null || s.peak_speed_rads != null || s.gear_ratio != null || s.rotor_inertia_kgm2 != null || s.peak_current_a != null || (s.notes && s.notes.length > 0)),
  ).length;

  const filledNotInMjcf: string[] = [];
  for (const [name, s] of Object.entries(specs)) {
    if (!s) continue;
    const hasValue = s.peak_torque_nm != null || s.peak_speed_rads != null || s.gear_ratio != null || s.rotor_inertia_kgm2 != null || s.peak_current_a != null || (s.notes && s.notes.length > 0);
    if (hasValue && !actuators.includes(name)) filledNotInMjcf.push(name);
  }

  const submit = () => {
    const filtered: Record<string, MotorSpec> = {};
    for (const [name, s] of Object.entries(specs)) {
      if (!s) continue;
      const clean: MotorSpec = {};
      if (s.peak_torque_nm != null) clean.peak_torque_nm = s.peak_torque_nm;
      if (s.peak_speed_rads != null) clean.peak_speed_rads = s.peak_speed_rads;
      if (s.gear_ratio != null) clean.gear_ratio = s.gear_ratio;
      if (s.rotor_inertia_kgm2 != null) clean.rotor_inertia_kgm2 = s.rotor_inertia_kgm2;
      if (s.peak_current_a != null) clean.peak_current_a = s.peak_current_a;
      if (s.notes) clean.notes = s.notes;
      if (Object.keys(clean).length > 0) filtered[name] = clean;
    }
    if (Object.keys(filtered).length === 0) {
      toast.error("Fill in at least one motor spec first.");
      return;
    }
    apply.mutate({ motors: filtered });
  };

  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="zap" size={16} />Motor specifications</div>
        <div className="rs-flex rs-gap-8">
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (!f || extract.isPending || apply.isPending) return;
              if (f.size > 10 * 1024 * 1024) { toast.error("Datasheet too large", { description: `${(f.size / 1024 / 1024).toFixed(1)}MB — trim to ≤10 MB.` }); return; }
              if (!f.type.includes("pdf") && !f.name.toLowerCase().endsWith(".pdf")) { toast.error("Not a PDF", { description: "Upload a real PDF." }); return; }
              extract.mutate(f);
            }}
          />
          <Btn kind="ghost" size="sm" icon={extract.isPending ? "loader" : "upload"} disabled={extract.isPending || apply.isPending} onClick={() => fileInputRef.current?.click()} title="Upload a motor datasheet PDF; Claude extracts specs to pre-fill the table.">
            Upload datasheet
          </Btn>
          <Btn kind="ghost" size="sm" onClick={() => setOpen((v) => !v)}>
            {open ? "Collapse" : `Configure (${actuators.length})`}
          </Btn>
        </div>
      </div>
      {open && (
        <div className="rs-card-pad">
          {rowNames.length === 0 ? (
            <p className="rs-sub" style={{ fontStyle: "italic", fontSize: 12 }}>(no actuators in the MJCF — upload a datasheet to seed rows)</p>
          ) : (
            <div style={{ maxHeight: 360, overflow: "auto" }}>
              <table className="rs-table">
                <thead>
                  <tr>
                    <th>joint</th><th>peak τ (Nm)</th><th>peak ω (rad/s)</th><th>gear</th><th>rotor I</th><th>peak i (A)</th>
                  </tr>
                </thead>
                <tbody>
                  {rowNames.map((name) => {
                    const s = specs[name] ?? {};
                    const extra = !actuators.includes(name);
                    return (
                      <tr key={name}>
                        <td className="name">
                          {name}
                          {extra && <span className="rs-tag" style={{ marginLeft: 6, background: "var(--st-amber-bg)", color: "var(--st-amber-fg)", padding: "1px 6px", fontSize: 10 }} title="Not in the MJCF — Claude will reject 'unknown joints'.">not in MJCF</span>}
                        </td>
                        {(["peak_torque_nm", "peak_speed_rads", "gear_ratio", "rotor_inertia_kgm2", "peak_current_a"] as const).map((k) => (
                          <td key={k}>
                            <input
                              type="number" step="any" min="0.0001"
                              value={s[k] == null ? "" : String(s[k] as number)}
                              onChange={(e) => update(name, k, e.target.value)}
                              className="rs-input mono"
                              style={{ height: 28, padding: "2px 6px", textAlign: "right", width: 90 }}
                              disabled={apply.isPending}
                              placeholder="—"
                              aria-label={`${name} ${k.replace(/_/g, " ")}`}
                            />
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <div className="rs-flex-between rs-gap-12" style={{ marginTop: 12 }}>
            <p className="rs-sub" style={{ fontSize: 12, margin: 0 }}>
              {filledNotInMjcf.length > 0 ? (
                <span style={{ color: "var(--st-amber-fg)" }}>Fix or delete {filledNotInMjcf.length} row{filledNotInMjcf.length === 1 ? "" : "s"} not in the MJCF: {filledNotInMjcf.slice(0, 3).join(", ")}{filledNotInMjcf.length > 3 ? "…" : ""}</span>
              ) : nonEmptyCount === 0 ? "Fill any row then Apply — empty rows are skipped." : `${nonEmptyCount} joint${nonEmptyCount === 1 ? "" : "s"} ready to apply.`}
            </p>
            <Btn kind="primary" size="sm" icon={apply.isPending ? "loader" : "sparkles"} onClick={submit} disabled={apply.isPending || nonEmptyCount === 0 || filledNotInMjcf.length > 0}>
              Apply motor limits
            </Btn>
          </div>
        </div>
      )}
    </div>
  );
}

export default PhysicsTab;

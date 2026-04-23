import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Atom,
  Cog,
  FileUp,
  Gauge,
  Layers,
  Loader2,
  RefreshCw,
  Wand2,
  Zap,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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

// Pre-filled template for the "Motor limits" button. Sam enters torque
// / speed / gear ratio numbers from his motor datasheet; Claude maps
// them to MJCF `<actuator forcerange>`, `<joint armature>`, `<joint
// damping>`, `<joint frictionloss>`. Grounded in the actuator-modeling
// papers seeded by the 2026-04-22 cartwheel-ingest pass (ANYmal
// actuator-net 1901.08652; Extended Friction 2410.08650;
// Actuator-Constrained RL 2312.17507; SEA Max Torque 1902.05346).
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


/** Physics tab (M7 Phase 5, prompt-first). Left column is a Claude
 * prompt textarea + live diff view; right column is a read-only
 * summary of the current MJCF's key physics parameters. Form-driven
 * editing is deferred to a follow-up pass. */
export function PhysicsTab({
  slug,
  project,
}: {
  slug: string;
  project: ProjectDetail;
}) {
  const phys = usePhysics(slug);
  const [prompt, setPrompt] = useState("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [lastRejection, setLastRejection] = useState<RejectionState | null>(
    null,
  );
  const promptRef = useRef<HTMLTextAreaElement | null>(null);

  // §7.4: pick up an auto-physics suggestion parked in sessionStorage
  // by RunsTab's "apply physics fix" chip. Pull-once-and-clear so a
  // subsequent navigation doesn't re-prefill the stale prompt. Only
  // runs on first mount; user edits to `prompt` afterwards are sticky.
  useEffect(() => {
    try {
      const pending = sessionStorage.getItem("pendingPhysicsPrompt");
      if (pending && pending.trim().length > 0) {
        setPrompt(pending);
        sessionStorage.removeItem("pendingPhysicsPrompt");
        // Give the textarea focus so Sam can review + tweak before
        // hitting "Prompt Claude".
        setTimeout(() => promptRef.current?.focus(), 0);
      }
    } catch {
      // sessionStorage unavailable — nothing to pre-fill.
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const mutate = usePhysicsPromptEdit(slug);
  const rematerialize = usePhysicsRematerialize(slug);
  const job = useJob(activeJobId, { refetchIntervalMs: 1500 });
  const qc = useQueryClient();

  useEffect(() => {
    if (!job.data || !activeJobId) return;
    if (job.data.status === "completed") {
      const result = (job.data.result as Record<string, any>) ?? {};
      const committed = Boolean(result.committed);
      const summary = (result.summary as string) ?? "";
      if (committed) {
        toast.success("Physics edit committed", {
          description: summary.slice(0, 120),
        });
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
        // Keep the prompt text so the user can edit + retry.
      }
      qc.invalidateQueries({ queryKey: ["physics", slug] });
      setActiveJobId(null);
    } else if (job.data.status === "errored") {
      toast.error("Physics edit failed", {
        description: job.data.error ?? "see /jobs for details",
      });
      setActiveJobId(null);
    }
  }, [job.data, activeJobId, slug, qc]);

  const inFlight =
    Boolean(activeJobId) &&
    job.data?.status !== "completed" &&
    job.data?.status !== "errored";

  const handleRematerialize = () => {
    rematerialize.mutate(undefined, {
      onSuccess: () => {
        toast.success("MJCF re-materialized", {
          description: "Assets copied from the library source.",
        });
        qc.invalidateQueries({ queryKey: ["physics", slug] });
      },
      onError: (err) => {
        const msg =
          err instanceof ApiError
            ? err.problem.detail ?? err.problem.title
            : (err as Error).message;
        toast.error("Re-materialize failed", { description: msg });
      },
    });
  };

  const submit = () => {
    if (prompt.trim().length < 3) {
      toast.error("Prompt too short", {
        description: "Describe the physics change (≥ 3 chars).",
      });
      return;
    }
    mutate.mutate(
      { prompt: prompt.trim() },
      {
        onSuccess: (j) => {
          setActiveJobId(j.job_id);
          toast.success("Claude is editing the MJCF…", {
            description: "~30-90 s for a small change",
          });
        },
        onError: (err) => {
          const msg =
            err instanceof ApiError
              ? err.problem.detail ?? err.problem.title
              : (err as Error).message;
          toast.error("Could not start physics edit", { description: msg });
        },
      },
    );
  };

  if (phys.isLoading) {
    return <p className="text-sm text-muted-foreground">Loading MJCF…</p>;
  }
  if (phys.error) {
    const err = phys.error as Error;
    const detail =
      err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
    return (
      <Card className="border-amber-300/60 bg-amber-50 dark:bg-amber-900/20">
        <CardContent className="flex items-start gap-2 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-600" />
          <div>
            <div className="font-medium">Physics editor unavailable</div>
            <p className="text-xs text-muted-foreground">{detail}</p>
          </div>
        </CardContent>
      </Card>
    );
  }
  if (!phys.data) return null;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <section className="flex min-w-0 flex-col gap-4">
        <Card className="border-primary/20 bg-primary/5">
          <CardContent className="flex flex-col gap-3 p-4">
            <div className="flex items-start gap-3">
              <span className="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Atom className="h-4 w-4" />
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold">
                  Edit physics with Claude
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Describe the change in plain English. Claude rewrites
                  the MJCF, validates it via{" "}
                  <code>mujoco.MjModel.from_xml_string</code>, and commits
                  the new model to this project's git repo. Form-based
                  field editing (sliders etc.) comes in a follow-up.
                </p>
              </div>
            </div>
            <Textarea
              ref={promptRef}
              value={prompt}
              onChange={(e) => {
                setPrompt(e.target.value);
                if (lastRejection) setLastRejection(null);
              }}
              placeholder={
                "e.g.\n" +
                "  • Increase hip joint damping by 50% to reduce oscillation\n" +
                "  • Drop timestep to 0.001 for tighter integration\n" +
                "  • Add a parallel-elastic spring across the knee joint"
              }
              rows={4}
              maxLength={2000}
              disabled={inFlight}
              className="text-xs"
            />
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                <span>
                  Source: <code>{phys.data.mjcf_source_kind}</code>
                  {phys.data.mjcf_source_kind === "library" && !phys.data.materialized && (
                    <>
                      {" "}· first edit will copy the MJCF into{" "}
                      <code>uploads/robot/base.xml</code>
                    </>
                  )}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={inFlight}
                  onClick={() => setPrompt(_MOTOR_LIMITS_TEMPLATE)}
                  title="Fill the prompt with a motor-limit template. Edit the numbers to match your motor datasheet, then Prompt Claude."
                  className="h-6 px-2 text-[10px]"
                >
                  Motor limits template
                </Button>
              </div>
              <Button size="sm" onClick={submit} disabled={inFlight}>
                {inFlight ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Wand2 className="h-3.5 w-3.5" />
                )}
                {inFlight ? "Claude is editing…" : "Prompt Claude"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {lastRejection && (
          <RejectionCard
            state={lastRejection}
            onRetry={() => {
              setLastRejection(null);
              promptRef.current?.focus();
            }}
          />
        )}

        {phys.data.summary.parse_error && (
          <Card className="border-amber-400/60 bg-amber-50 dark:border-amber-500/40 dark:bg-amber-900/20">
            <CardContent className="flex flex-col gap-3 p-4 text-sm">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                <div className="min-w-0 flex-1">
                  <div className="font-medium">MJCF won't parse</div>
                  <p className="mt-1 break-words font-mono text-[11px] text-muted-foreground">
                    {phys.data.summary.parse_error}
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    This usually means referenced mesh or texture files
                    are missing from{" "}
                    <code>uploads/robot/</code>. Click{" "}
                    <strong>Re-materialize</strong> to wipe the local copy
                    and re-fetch the library source + assets.
                  </p>
                </div>
              </div>
              <div className="flex items-center justify-end">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleRematerialize}
                  disabled={rematerialize.isPending}
                >
                  {rematerialize.isPending ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5" />
                  )}
                  {rematerialize.isPending
                    ? "Re-materializing…"
                    : "Re-materialize MJCF"}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        <MotorLimitsCard slug={slug} summary={phys.data.summary} />

        <Card className="overflow-hidden">
          <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 border-b py-3">
            <div>
              <CardTitle className="text-sm">
                Current MJCF{" "}
                {phys.data.mjcf_path && (
                  <code className="font-mono text-[10px] text-muted-foreground">
                    ({phys.data.mjcf_path.split("/").slice(-2).join("/")})
                  </code>
                )}
              </CardTitle>
              <CardDescription className="text-[11px]">
                {phys.data.xml_source.split("\n").length} lines · read-only
                — use the prompt above to edit.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <MonacoLazy
              value={phys.data.xml_source}
              readOnly
              height={420}
              language="xml"
            />
          </CardContent>
        </Card>
      </section>

      <aside className="order-1 lg:order-2">
        <SummaryPanel summary={phys.data.summary} />
      </aside>
    </div>
  );
}

function SummaryPanel({ summary }: { summary: MjcfSummary }) {
  return (
    <div className="flex flex-col gap-3">
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Gauge className="h-3.5 w-3.5" />
            Simulation options
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-1 pt-0 font-mono text-[11px]">
          <KV label="timestep" v={fmt(summary.timestep)} />
          <KV label="gravity" v={fmtVec(summary.gravity)} />
          <KV label="integrator" v={summary.integrator ?? "—"} />
          <KV label="solver" v={summary.solver ?? "—"} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Cog className="h-3.5 w-3.5" />
            Joints ({summary.joints.length})
          </CardTitle>
          <CardDescription className="text-[11px]">
            damping + armature drive SEA compliance. Free joint (ball/
            slide/hinge) rows listed; excludes root.
          </CardDescription>
        </CardHeader>
        <CardContent className="max-h-56 overflow-y-auto pt-0 font-mono text-[11px]">
          {summary.joints.length === 0 ? (
            <p className="italic text-muted-foreground">(none)</p>
          ) : (
            <ul className="space-y-0.5">
              {summary.joints.slice(0, 30).map((j) => (
                <li
                  key={j.name}
                  className="flex items-center justify-between gap-2 border-b border-border/40 py-0.5"
                >
                  <span className="truncate">{j.name}</span>
                  <span className="text-muted-foreground">
                    d={fmt(j.damping)} a={fmt(j.armature)}
                  </span>
                </li>
              ))}
              {summary.joints.length > 30 && (
                <li className="pt-1 text-[10px] italic text-muted-foreground">
                  + {summary.joints.length - 30} more
                </li>
              )}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Layers className="h-3.5 w-3.5" />
            Actuators ({summary.actuators.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="max-h-56 overflow-y-auto pt-0 font-mono text-[11px]">
          {summary.actuators.length === 0 ? (
            <p className="italic text-muted-foreground">(none)</p>
          ) : (
            <ul className="space-y-0.5">
              {summary.actuators.slice(0, 30).map((a) => (
                <li
                  key={a.name}
                  className="flex items-center justify-between gap-2 border-b border-border/40 py-0.5"
                >
                  <span className="truncate">{a.name}</span>
                  <span className="text-muted-foreground">
                    gear={fmt(a.gear)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function KV({ label, v }: { label: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-1 border-b border-border/40 py-0.5">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <span>{v}</span>
    </div>
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

const REJECTED_AT_LABEL: Record<
  NonNullable<RejectionState["rejectedAt"]>,
  string
> = {
  parse: "response format",
  claude_rejected: "claude refused",
  mujoco_validate: "mujoco rejected",
};

function RejectionCard({
  state,
  onRetry,
}: {
  state: RejectionState;
  onRetry: () => void;
}) {
  const tag = state.rejectedAt
    ? REJECTED_AT_LABEL[state.rejectedAt]
    : "unknown";
  return (
    <Card className="border-red-300 bg-red-50 dark:border-red-500/40 dark:bg-red-900/20">
      <CardContent className="flex flex-col gap-3 p-4 text-sm">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">Edit rejected</span>
              <span className="rounded bg-red-200/70 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-red-900 dark:bg-red-500/30 dark:text-red-100">
                {tag}
              </span>
            </div>
            <p className="mt-2 whitespace-pre-wrap break-words font-mono text-[11px] text-muted-foreground">
              {state.reason}
            </p>
          </div>
        </div>
        <details className="rounded border bg-background text-xs">
          <summary className="cursor-pointer select-none px-3 py-2 font-medium">
            View what Claude wrote
          </summary>
          <div className="border-t">
            {state.rejectedAt === "parse" ? (
              <pre className="max-h-[260px] overflow-auto whitespace-pre-wrap p-3 font-mono text-[11px]">
                {state.claudeOutputRaw ?? "(no raw output captured)"}
              </pre>
            ) : (
              <MonacoDiffLazy
                original={state.oldXml}
                modified={state.newXml}
                language="xml"
                height={320}
              />
            )}
          </div>
        </details>
        <div className="flex items-center justify-end">
          <Button size="sm" variant="outline" onClick={onRetry}>
            Retry with a different prompt
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}


// ── §Ship-9b: motor-specs form ──────────────────────────────────────────
/**
 * Structured motor-limits entry point: one row per actuator in the
 * current MJCF, four numeric inputs (+ optional notes). On submit the
 * backend synthesizes a physics-editor NL prompt from the structured
 * data and delegates to the same Claude-backed edit path.
 */
function MotorLimitsCard({
  slug,
  summary,
}: {
  slug: string;
  summary: MjcfSummary;
}) {
  const [open, setOpen] = useState(false);
  const [specs, setSpecs] = useState<Record<string, MotorSpec>>({});
  const actuators = useMemo(
    () => (summary.actuators ?? []).map((a) => a.name),
    [summary.actuators],
  );
  // §Ship-9c: after a datasheet upload, Claude may return joint names
  // that aren't in `summary.actuators` (the datasheet might use
  // different labeling). Union the two so the user sees every row
  // that needs review.
  const rowNames = useMemo(() => {
    const set = new Set<string>(actuators);
    for (const k of Object.keys(specs)) set.add(k);
    return Array.from(set);
  }, [actuators, specs]);

  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // §Ship-9c: datasheet PDF upload. Server returns structured specs
  // (Claude-extracted); we merge into local state so user can review
  // + edit before hitting Apply.
  const extract = useMutation({
    mutationFn: (pdf: File) => extractDatasheetPdf(slug, pdf),
    onSuccess: (data) => {
      const incoming = data.motors ?? {};
      const nMotors = Object.keys(incoming).length;
      if (nMotors === 0) {
        toast.info(
          "Datasheet uploaded but no motor specs were extracted",
          { description: "Check that the PDF has text (not scanned image)." },
        );
        return;
      }
      setSpecs((prev) => {
        // Merge: extracted specs win over any previously-entered row
        // with the same joint name, but untouched joints stay.
        const merged: Record<string, MotorSpec> = { ...prev };
        for (const [name, spec] of Object.entries(incoming)) {
          merged[name] = spec;
        }
        return merged;
      });
      setOpen(true);
      toast.success(`Extracted ${nMotors} motor spec${nMotors === 1 ? "" : "s"}`, {
        description: "Review + edit before hitting Apply.",
      });
    },
    onError: (err) => {
      const msg = err instanceof ApiError
        ? err.problem.detail ?? err.problem.title
        : (err as Error).message;
      toast.error("Datasheet extraction failed", { description: msg });
    },
  });

  const apply = useMutation({
    mutationFn: (body: MotorLimitsRequest) => applyMotorLimits(slug, body),
    onSuccess: (job) => {
      toast.success("Motor-limits edit started", {
        description: `job ${job.job_id.slice(0, 8)}… — watch for a commit toast`,
      });
      // §Ship-9b hotfix (critique medium-5): invalidate physics query
      // so the Current MJCF panel refreshes once the edit commits.
      // Small delay lets the backend finish the write before refetch.
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ["physics", slug] });
      }, 2000);
    },
    onError: (err) => {
      const msg = err instanceof ApiError
        ? err.problem.detail ?? err.problem.title
        : (err as Error).message;
      toast.error("Motor-limits edit failed to start", { description: msg });
    },
  });

  const update = (name: string, key: keyof MotorSpec, raw: string) => {
    setSpecs((prev) => {
      const row: MotorSpec = { ...(prev[name] ?? {}) };
      if (raw === "") {
        // Clear → remove the key entirely rather than store null so we
        // don't send `null` to the backend on Apply (backend model
        // would pass it through, but the wire stays clean).
        delete (row as Record<string, unknown>)[key];
      } else if (key === "notes") {
        row.notes = raw;
      } else {
        const n = Number(raw);
        // §Ship-9b hotfix (critique medium-4): backend requires gt=0,
        // so drop non-positive values on the client and flash a toast
        // the FIRST time so the user understands.
        if (!Number.isFinite(n) || n <= 0) {
          delete (row as Record<string, unknown>)[key];
        } else {
          (row as Record<string, unknown>)[key] = n;
        }
      }
      return { ...prev, [name]: row };
    });
  };

  const nonEmptyCount = Object.values(specs).filter((s) =>
    s && (
      s.peak_torque_nm != null || s.peak_speed_rads != null ||
      s.gear_ratio != null || s.rotor_inertia_kgm2 != null ||
      s.peak_current_a != null || (s.notes && s.notes.length > 0)
    )
  ).length;

  // §Ship-9c hotfix (critique critical-3): any FILLED row with a joint
  // name that isn't in `summary.actuators` will make Claude reject on
  // "unknown joints" — wasting a ~60s API round-trip. Surface it early
  // by disabling Apply until the user fixes / deletes those rows.
  const filledNotInMjcf: string[] = [];
  for (const [name, s] of Object.entries(specs)) {
    if (!s) continue;
    const hasValue =
      s.peak_torque_nm != null || s.peak_speed_rads != null ||
      s.gear_ratio != null || s.rotor_inertia_kgm2 != null ||
      s.peak_current_a != null || (s.notes && s.notes.length > 0);
    if (hasValue && !actuators.includes(name)) {
      filledNotInMjcf.push(name);
    }
  }

  const submit = () => {
    // Filter out rows where everything is null/empty.
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
    <Card>
      <CardHeader className="py-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Zap className="h-3.5 w-3.5" />
              Motor specs (structured form)
            </CardTitle>
            <CardDescription className="text-[11px]">
              §Ship-9b — enter per-joint peak torque / speed / gear / rotor
              inertia from your datasheet. Only rows you fill in are applied.
              Claude translates to `forcerange` / `armature` / `damping`
              with KG-cited papers.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {/* §Ship-9c: datasheet PDF upload. Extracts structured
                 motor specs via Claude, then pre-fills the table below. */}
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf,.pdf"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                e.target.value = "";  // allow re-uploading the same file
                // §Ship-9c hotfix (critique medium-9): reject double-click
                // while an extract is already in flight.
                if (!f || extract.isPending || apply.isPending) return;
                // Quick client-side size/type guard — the real cap is
                // enforced server-side (10 MB), but warn earlier.
                if (f.size > 10 * 1024 * 1024) {
                  toast.error("Datasheet too large", {
                    description: `${(f.size / 1024 / 1024).toFixed(1)}MB — trim to ≤10 MB first.`,
                  });
                  return;
                }
                if (!f.type.includes("pdf") && !f.name.toLowerCase().endsWith(".pdf")) {
                  toast.error("Not a PDF", {
                    description: "Upload a real PDF (not a PNG/JPG of one).",
                  });
                  return;
                }
                extract.mutate(f);
              }}
            />
            <Button
              size="sm"
              variant="outline"
              disabled={extract.isPending || apply.isPending}
              onClick={() => fileInputRef.current?.click()}
              title="Upload a motor datasheet PDF; Claude extracts specs to pre-fill the table."
            >
              {extract.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <FileUp className="h-3.5 w-3.5" />
              )}
              Upload datasheet
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setOpen((v) => !v)}
            >
              {open ? "Collapse" : `Configure (${actuators.length} joints)`}
            </Button>
          </div>
        </div>
      </CardHeader>
      {open && (
        <CardContent className="pt-0">
          {rowNames.length === 0 ? (
            <p className="text-[11px] italic text-muted-foreground">
              (no actuators found in the current MJCF — upload a datasheet
              to seed rows)
            </p>
          ) : (
            <div className="max-h-[360px] overflow-y-auto rounded-md border">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 bg-muted/60 font-semibold">
                  <tr>
                    <th className="px-2 py-1.5 text-left">joint</th>
                    <th className="px-2 py-1.5 text-right">peak τ (Nm)</th>
                    <th className="px-2 py-1.5 text-right">peak ω (rad/s)</th>
                    <th className="px-2 py-1.5 text-right">gear</th>
                    <th className="px-2 py-1.5 text-right">rotor I (kg·m²)</th>
                    <th className="px-2 py-1.5 text-right">peak i (A)</th>
                  </tr>
                </thead>
                <tbody>
                  {rowNames.map((name) => {
                    const s = specs[name] ?? {};
                    const extra = !actuators.includes(name);
                    return (
                      <tr key={name} className="border-t">
                        <td className="px-2 py-1 font-mono">
                          {name}
                          {extra && (
                            <span
                              className="ml-1 rounded bg-amber-100 px-1 py-0 text-[9px] text-amber-800"
                              title="This joint name came from the datasheet but isn't in the MJCF. Confirm the label matches before applying, or Claude will reject on 'unknown joints'."
                            >
                              not in MJCF
                            </span>
                          )}
                        </td>
                        {(
                          [
                            "peak_torque_nm",
                            "peak_speed_rads",
                            "gear_ratio",
                            "rotor_inertia_kgm2",
                            "peak_current_a",
                          ] as const
                        ).map((k) => (
                          <td key={k} className="px-1 py-1">
                            <Input
                              type="number"
                              step="any"
                              min="0.0001"
                              value={
                                s[k] == null ? "" : String(s[k] as number)
                              }
                              onChange={(e) =>
                                update(name, k, e.target.value)
                              }
                              className="h-6 w-full text-right font-mono text-[11px]"
                              disabled={apply.isPending}
                              placeholder="—"
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
          <div className="mt-2 flex items-center justify-between gap-2">
            <p className="text-[10px] text-muted-foreground">
              {filledNotInMjcf.length > 0 ? (
                <span className="text-amber-700">
                  Fix or delete {filledNotInMjcf.length} row
                  {filledNotInMjcf.length === 1 ? "" : "s"} whose joint name
                  isn't in the MJCF: {filledNotInMjcf.slice(0, 3).join(", ")}
                  {filledNotInMjcf.length > 3 ? "…" : ""}
                </span>
              ) : nonEmptyCount === 0 ? (
                "Fill any row then Apply — empty rows are skipped."
              ) : (
                `${nonEmptyCount} joint${nonEmptyCount === 1 ? "" : "s"} ready to apply.`
              )}
            </p>
            <Button
              size="sm"
              onClick={submit}
              disabled={
                apply.isPending ||
                nonEmptyCount === 0 ||
                filledNotInMjcf.length > 0
              }
            >
              {apply.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Wand2 className="h-3.5 w-3.5" />
              )}
              Apply motor limits
            </Button>
          </div>
        </CardContent>
      )}
    </Card>
  );
}

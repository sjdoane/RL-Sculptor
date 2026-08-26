import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import "@/components/StartingPointPickerDialog.css";

import { Icon } from "@/components/rs/icon";
import { Btn, Modal } from "@/components/rs/primitives";
import {
  ApiError,
  listPolicyRecoverySnapshots,
  listStartingSkills,
  uploadStartingSkill,
} from "@/lib/api";
import { isDeployablePolicy } from "@/lib/policyEvidence";
import type {
  PolicyEvidenceValue,
  PolicyRecoverySnapshot,
  PolicySummary,
  StartingPointKind,
  StartingPointSelection,
  StartingSkillInitializationMode,
  StartingSkillReceipt,
  StartingSkillsResponse,
} from "@/lib/types";

const ACCEPTED_BUNDLE_EXTENSIONS = [".rskill"] as const;
type ProjectPolicySource = "completed" | "interrupted";

const INITIALIZATION_COPY: Record<
  StartingSkillInitializationMode,
  { label: string; description: string; advanced?: boolean }
> = {
  actor_only: {
    label: "Actor only",
    description: "Start from the learned behavior and fit a fresh critic for this task.",
  },
  actor_critic: {
    label: "Actor + critic",
    description: "Transfer both networks. Faster when objectives match; easier to bias when they do not.",
    advanced: true,
  },
  reference_only: {
    label: "Motion only",
    description: "Use the bundled motion as a reference without loading policy weights.",
  },
  full_resume: {
    label: "Full training resume",
    description: "Restore optimizer and training state. Available only for an exact, trusted match.",
    advanced: true,
  },
};

const DEFAULT_SELECTION: StartingPointSelection = {
  kind: "scratch",
  warm_start_iteration: null,
  warm_start_snapshot: null,
  warm_start_snapshot_display: null,
  starting_skill_id: null,
  initialization_mode: null,
  reference_clip_id: null,
  reference_robot: null,
  import_manifest_digest: null,
  compatibility_contract_provenance_status: null,
  acknowledge_legacy_reconstructed_initialization: false,
  policy_contract_migration: null,
};

export interface StartingPointPickerDialogProps {
  slug: string;
  projectRobot?: string | null;
  projectTaskId?: string | null;
  /** Disk-backed project checkpoints, newest first in most callers. The
   * picker also permits an explicit iteration when this list has not loaded. */
  checkpoints?: PolicySummary[];
  checkpointsLoading?: boolean;
  value: StartingPointSelection;
  onChange: (value: StartingPointSelection) => void;
  onClose: () => void;
}

function problemMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return String(error.problem.detail ?? error.problem.title);
  }
  if (error instanceof Error) return error.message;
  return "The server could not validate this bundle.";
}

function shortDigest(value: string | null | undefined): string {
  if (!value) return "Not reported";
  return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
}

function hasImmutableManifest(receipt: StartingSkillReceipt): boolean {
  return /^[a-f0-9]{64}$/.test(receipt.skill.manifest_digest ?? "");
}

function formatBytes(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "Not reported";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(2)} GiB`;
}

function humanize(value: string | null | undefined): string {
  if (!value) return "Not reported";
  return value.replaceAll("_", " ");
}

function evidenceValue(value: PolicyEvidenceValue | null): string {
  if (!value) return "Not recorded";
  if (value.kind === "fraction") {
    return `${value.value.toFixed(4)} (${(value.value * 100).toFixed(1)}%)`;
  }
  if (value.kind === "count" || value.kind === "frames") {
    return Number.isInteger(value.value)
      ? String(value.value)
      : value.value.toFixed(2);
  }
  return value.value.toFixed(4);
}

function DigestValue({ value }: { value: string | null | undefined }) {
  return (
    <span className="mono" title={value ?? undefined}>
      {shortDigest(value)}
    </span>
  );
}

function contractSummary(contract: Record<string, unknown> | null | undefined): string {
  if (!contract) return "Not reported";
  const observations = contract.observations as Record<string, unknown> | undefined;
  const actions = contract.actions as Record<string, unknown> | undefined;
  const policy = contract.policy as Record<string, unknown> | undefined;
  const actor = policy?.actor as Record<string, unknown> | undefined;
  const observationShape = Array.isArray(observations?.shape)
    ? observations.shape.join("x")
    : null;
  const actionShape = Array.isArray(actions?.shape)
    ? actions.shape.join("x")
    : null;
  const hidden = Array.isArray(actor?.hidden_dims)
    ? actor.hidden_dims.join(" -> ")
    : null;
  const parts = [
    observationShape ? `obs ${observationShape}` : null,
    actionShape ? `actions ${actionShape}` : null,
    hidden ? `actor ${hidden}` : null,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(" / ") : "Recorded; expand the manifest for full structure";
}

function CheckpointReceipt({ checkpoint }: { checkpoint: PolicySummary }) {
  const metricLabel = checkpoint.metric_id
    ? `${checkpoint.metric_id}${checkpoint.metric_version ? ` / ${checkpoint.metric_version}` : ""}`
    : "Metric identity not recorded";
  return (
    <div
      role="region"
      aria-label={`Iteration ${checkpoint.iter_index} evidence`}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        padding: 12,
        border: "1px solid var(--hairline)",
        borderRadius: "var(--radius-md)",
        background: "var(--canvas-soft)",
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 7 }}>
        <strong style={{ fontSize: 12.5 }}>Iteration {checkpoint.iter_index}</strong>
        {checkpoint.selected && <span className="rs-badge emerald">selected</span>}
        <span className={`rs-badge ${checkpoint.criterion_status === "passed" ? "emerald" : checkpoint.criterion_status === "failed" ? "rose" : "slate"}`}>
          criterion {humanize(checkpoint.criterion_status)}
        </span>
        <span className={`rs-badge ${checkpoint.rollout_available ? "blue" : "slate"}`}>
          {checkpoint.rollout_available ? "rollout available" : "no rollout artifact"}
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: 8 }}>
        <SummaryCell
          icon="target"
          label="Objective metric"
          title={metricLabel}
          description={checkpoint.metric_id
            ? checkpoint.metric_sha256
              ? `${humanize(checkpoint.metric_source)} / ${shortDigest(checkpoint.metric_sha256)}`
              : `${humanize(checkpoint.metric_source)}; metric digest not recorded, so cross-run score comparison is unverified.`
            : "A scalar may exist, but scores without identity and digest are not comparable."}
          tone={checkpoint.metric_id && checkpoint.metric_sha256 ? "good" : "warning"}
        />
        <SummaryCell
          icon="check-circle"
          label="Evaluation coverage"
          title={humanize(checkpoint.evidence_status)}
          description="Metric-specific evidence retained by the evaluator"
          tone={checkpoint.evidence_status === "complete" ? "good" : "neutral"}
        />
        {checkpoint.route_evidence && (
          <SummaryCell
            icon="activity"
            label="Route evidence"
            title={evidenceValue(checkpoint.route_evidence)}
            description={checkpoint.route_evidence.key}
            tone="good"
          />
        )}
        {checkpoint.contact_evidence && (
          <SummaryCell
            icon="shield-check"
            label="Contact evidence"
            title={evidenceValue(checkpoint.contact_evidence)}
            description={checkpoint.contact_evidence.key}
            tone="good"
          />
        )}
        {checkpoint.hold_evidence && (
          <SummaryCell
            icon="pause"
            label="Hold evidence"
            title={evidenceValue(checkpoint.hold_evidence)}
            description={checkpoint.hold_evidence.key}
            tone="good"
          />
        )}
      </div>
      <div className="rs-sub" style={{ fontSize: 10.5, lineHeight: 1.45 }}>
        Checkpoint SHA-256: <DigestValue value={checkpoint.checkpoint_sha256} />.
        {" "}
        {checkpoint.selected
          ? `Selection authority: ${checkpoint.selection_source ?? "not recorded"}. `
          : "This checkpoint is not marked selected by the project receipt. "}
        Evidence coverage: {checkpoint.evidence_status ?? "unavailable"}.
        {checkpoint.fitness != null ? ` Recorded score: ${checkpoint.fitness.toFixed(4)}.` : " No objective score recorded."}
      </div>
    </div>
  );
}

function RecoverySnapshotReceipt({
  snapshot,
  acknowledged,
  legacyAcknowledged,
  onAcknowledgedChange,
  onLegacyAcknowledgedChange,
}: {
  snapshot: PolicyRecoverySnapshot;
  acknowledged: boolean;
  legacyAcknowledged: boolean;
  onAcknowledgedChange: (checked: boolean) => void;
  onLegacyAcknowledgedChange: (checked: boolean) => void;
}) {
  const legacy = snapshot.provenance_status === "legacy_reconstructed";
  return (
    <div
      role="region"
      aria-label={`Cycle ${snapshot.iteration} PPO snapshot ${snapshot.ppo_step} recovery disclosure`}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        padding: 12,
        border: "1px solid var(--st-amber)",
        borderRadius: "var(--radius-md)",
        background: "var(--st-amber-bg)",
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 7 }}>
        <strong style={{ fontSize: 12.5 }}>
          Cycle {snapshot.iteration} · PPO snapshot {snapshot.ppo_step}
        </strong>
        <span className="rs-badge amber">recovery input</span>
        <span className="rs-badge slate">unevaluated</span>
        {legacy && <span className="rs-badge amber">legacy receipt</span>}
      </div>
      <div className="rs-sub" style={{ fontSize: 11.5, lineHeight: 1.5 }}>
        The run last reported PPO iteration {snapshot.last_observed_ppo_iteration} and
        ended with status <strong>{humanize(snapshot.source_job_status)}</strong>.
        This recovery save has no rollout, objective score, criterion result, or success claim.
      </div>
      <dl
        style={{
          display: "grid",
          gridTemplateColumns: "max-content minmax(0, 1fr)",
          gap: "5px 12px",
          margin: 0,
          fontSize: 11,
          lineHeight: 1.45,
        }}
      >
        <dt className="rs-sub">What loads</dt>
        <dd style={{ margin: 0 }}>Actor + critic weights</dd>
        <dt className="rs-sub">What resets</dt>
        <dd style={{ margin: 0 }}>Optimizer, counters, and exploration state</dd>
        <dt className="rs-sub">Checkpoint</dt>
        <dd style={{ margin: 0 }}>
          <DigestValue value={snapshot.checkpoint_sha256} /> · {formatBytes(snapshot.checkpoint_bytes)}
        </dd>
        <dt className="rs-sub">Attestation receipt</dt>
        <dd style={{ margin: 0 }}><DigestValue value={snapshot.receipt_digest} /></dd>
        <dt className="rs-sub">Source job</dt>
        <dd className="mono" style={{ margin: 0, overflowWrap: "anywhere" }}>
          {snapshot.source_job_id}
        </dd>
      </dl>
      {snapshot.blocker && (
        <div role="alert" style={{ fontSize: 11.5, lineHeight: 1.45, color: "var(--st-rose-fg)" }}>
          <strong>Blocked:</strong> {snapshot.blocker}
        </div>
      )}
      <label
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: 9,
          minHeight: 44,
          cursor: snapshot.selectable ? "pointer" : "not-allowed",
          opacity: snapshot.selectable ? 1 : 0.55,
          fontSize: 11.5,
          lineHeight: 1.45,
        }}
      >
        <input
          type="checkbox"
          checked={acknowledged}
          disabled={!snapshot.selectable}
          onChange={(event) => onAcknowledgedChange(event.target.checked)}
          style={{ marginTop: 3 }}
        />
        <span>
          I understand this snapshot is unevaluated and may be worse than the
          run&apos;s starting policy.
        </span>
      </label>
      {legacy && (
        <label
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 9,
            minHeight: 44,
            cursor: snapshot.selectable ? "pointer" : "not-allowed",
            opacity: snapshot.selectable ? 1 : 0.55,
            fontSize: 11.5,
            lineHeight: 1.45,
          }}
        >
          <input
            type="checkbox"
            checked={legacyAcknowledged}
            disabled={!snapshot.selectable}
            onChange={(event) => onLegacyAcknowledgedChange(event.target.checked)}
            style={{ marginTop: 3 }}
          />
          <span>
            I understand this snapshot receipt was reconstructed after the
            interruption from retained job and log evidence.
          </span>
        </label>
      )}
    </div>
  );
}

function isAdmittedTrust(
  receipt: StartingSkillReceipt,
  mode: StartingSkillInitializationMode | null = null,
): boolean {
  if (receipt.trust.status === "sanitized"
      || receipt.trust.status === "verified_local") {
    return true;
  }
  // ``validated`` is intentionally narrower than sanitized policy trust. It
  // may admit a data-only trajectory, but can never authorize actor/critic
  // loading. Keep the rule local and structural so a future backend bug that
  // attaches policy roles to a validated receipt still fails closed here.
  const referenceOnly = receipt.trust.status === "validated"
    && receipt.components.policy_roles.length === 0
    && receipt.components.reference != null
    && receipt.compatibility.allowed_initialization_modes.length === 1
    && receipt.compatibility.allowed_initialization_modes[0] === "reference_only";
  return referenceOnly && (mode == null || mode === "reference_only");
}

function preferredInitialization(
  receipt: StartingSkillReceipt,
): StartingSkillInitializationMode | null {
  const modes = receipt.compatibility.allowed_initialization_modes;
  if (modes.includes("actor_only")) return "actor_only";
  if (modes.includes("reference_only")) return "reference_only";
  if (modes.includes("actor_critic")) return "actor_critic";
  if (modes.includes("full_resume")) return "full_resume";
  return null;
}

function loadedRoles(
  mode: StartingSkillInitializationMode | null,
  receipt: StartingSkillReceipt,
): string[] {
  if (mode === "actor_only") return ["actor"];
  if (mode === "actor_critic") return ["actor", "critic"];
  if (mode === "reference_only") return [];
  if (mode === "full_resume") return receipt.components.policy_roles;
  return [];
}

function moveWithinRadioGroup(event: ReactKeyboardEvent<HTMLButtonElement>) {
  const direction = event.key === "ArrowRight" || event.key === "ArrowDown"
    ? 1
    : event.key === "ArrowLeft" || event.key === "ArrowUp"
      ? -1
      : 0;
  const jumpToEdge = event.key === "Home" ? "first" : event.key === "End" ? "last" : null;
  if (direction === 0 && jumpToEdge === null) return;

  const group = event.currentTarget.closest<HTMLElement>('[role="radiogroup"]');
  if (!group) return;
  const radios = Array.from(
    group.querySelectorAll<HTMLButtonElement>(':scope > [role="radio"]'),
  ).filter((radio) => !radio.disabled);
  const currentIndex = radios.indexOf(event.currentTarget);
  if (currentIndex < 0 || radios.length === 0) return;

  event.preventDefault();
  const nextIndex = jumpToEdge === "first"
    ? 0
    : jumpToEdge === "last"
      ? radios.length - 1
      : (currentIndex + direction + radios.length) % radios.length;
  const next = radios[nextIndex];
  next.click();
  // Selection can reveal an auto-focused detail control. Arrow navigation
  // should still keep the user's focus in the radio group they are moving.
  next.focus();
}

function KindChoice({
  kind,
  selected,
  icon,
  title,
  description,
  onSelect,
}: {
  kind: StartingPointKind;
  selected: boolean;
  icon: string;
  title: string;
  description: string;
  onSelect: (kind: StartingPointKind) => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      tabIndex={selected ? 0 : -1}
      onClick={() => onSelect(kind)}
      onKeyDown={moveWithinRadioGroup}
      style={{
        display: "flex",
        minHeight: 118,
        flexDirection: "column",
        alignItems: "flex-start",
        gap: 7,
        padding: 14,
        textAlign: "left",
        borderRadius: "var(--radius-lg)",
        border: `1px solid ${selected ? "var(--rs-primary)" : "var(--hairline-strong)"}`,
        background: selected ? "rgba(245,78,0,0.05)" : "var(--surface-card)",
        color: "var(--ink)",
        font: "inherit",
        cursor: "pointer",
      }}
    >
      <span
        style={{
          display: "inline-flex",
          width: 30,
          height: 30,
          alignItems: "center",
          justifyContent: "center",
          borderRadius: "var(--radius-md)",
          background: selected ? "rgba(245,78,0,0.12)" : "var(--surface-strong)",
          color: selected ? "var(--rs-primary)" : "var(--rs-muted)",
        }}
      >
        <Icon name={icon} size={16} />
      </span>
      <span style={{ fontWeight: 650, fontSize: 13 }}>{title}</span>
      <span className="rs-sub" style={{ fontSize: 11.5, lineHeight: 1.4 }}>
        {description}
      </span>
    </button>
  );
}

function SummaryCell({
  icon,
  label,
  title,
  description,
  tone = "neutral",
}: {
  icon: string;
  label: string;
  title: string;
  description: string;
  tone?: "neutral" | "good" | "warning";
}) {
  const toneColor = tone === "good"
    ? "var(--st-emerald-fg)"
    : tone === "warning"
      ? "var(--st-amber-fg)"
      : "var(--rs-muted)";
  return (
    <div
      style={{
        minWidth: 0,
        padding: 12,
        border: "1px solid var(--hairline)",
        borderRadius: "var(--radius-md)",
        background: "var(--surface-strong)",
      }}
    >
      <div
        className="rs-sub"
        style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10.5, textTransform: "uppercase", letterSpacing: ".05em" }}
      >
        <Icon name={icon} size={13} color={toneColor} />
        {label}
      </div>
      <div style={{ marginTop: 7, fontSize: 12.5, fontWeight: 650, overflowWrap: "anywhere" }}>
        {title}
      </div>
      <div className="rs-sub" style={{ marginTop: 3, fontSize: 11.5, lineHeight: 1.4 }}>
        {description}
      </div>
    </div>
  );
}

function ReceiptSummary({
  receipt,
  mode,
  referenceSelected,
}: {
  receipt: StartingSkillReceipt;
  mode: StartingSkillInitializationMode | null;
  referenceSelected: boolean;
}) {
  const admittedTrust = isAdmittedTrust(receipt, mode);
  const candidate = receipt.selectable && admittedTrust
    && hasImmutableManifest(receipt);
  const roles = loadedRoles(mode, receipt);
  const reference = receipt.components.reference;
  const world = receipt.components.world;
  const controller = receipt.components.controller;
  const sourceFormat = receipt.trust.source_format ?? receipt.skill.source_format;
  const checkpointFormat = receipt.trust.checkpoint_format ?? receipt.skill.checkpoint_format;
  const authorizationDetail = receipt.authorization?.detail
    ?? "This legacy receipt predates launch-authorization evidence. Re-import the original .rskill to produce a current receipt before training.";
  const modeGates = mode ? receipt.authorization?.mode_gates?.[mode] ?? [] : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 7,
          padding: "10px 12px",
          borderRadius: "var(--radius-md)",
          background: candidate ? "var(--st-amber-bg)" : "var(--st-rose-bg)",
          color: candidate ? "var(--st-amber-fg)" : "var(--st-rose-fg)",
          fontSize: 12,
        }}
      >
        <Icon name={candidate ? "check-circle" : "alert-circle"} size={15} />
        <strong>{candidate
          ? "Validated starting-point candidate"
          : receipt.selectable
            ? "Structural compatibility passed; trust admission is incomplete"
            : "Starting point is blocked for this project"}</strong>
        <span style={{ opacity: 0.85 }}>· {humanize(receipt.compatibility.status)}</span>
        <span className={`rs-badge ${admittedTrust ? "emerald" : "amber"}`} style={{ marginLeft: "auto" }}>
          {admittedTrust ? "trusted" : "trust unknown"} · {humanize(receipt.trust.status)}
        </span>
      </div>

      <div
        style={{
          padding: "9px 11px",
          border: "1px solid var(--hairline-strong)",
          borderRadius: "var(--radius-md)",
          background: "var(--st-amber-bg)",
          color: "var(--st-amber-fg)",
          fontSize: 11.5,
          lineHeight: 1.45,
        }}
      >
        <strong>This import receipt does not authorize training.</strong>{" "}
        {authorizationDetail}
        {modeGates.length ? (
          <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {modeGates.map((gate) => (
              <li key={gate}>{gate}</li>
            ))}
          </ul>
        ) : null}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8 }}>
        <SummaryCell
          icon="cpu"
          label="Starting policy"
          title={roles.length ? roles.join(" + ") : "No weights selected"}
          description={roles.length
            ? `Only ${roles.join(" and ")} will load with this choice.`
            : "The policy checkpoint will not be loaded."}
          tone={roles.length ? "good" : "neutral"}
        />
        <SummaryCell
          icon="activity"
          label="Starting motion"
          title={reference
            ? referenceSelected
              ? reference.clip_id
              : `${reference.clip_id} · bundled, not selected`
            : "Not bundled"}
          description={reference
            ? referenceSelected
              ? `Registered candidate under ${reference.robot}; attaching selects it independently from the policy, but does not grant training authorization.`
              : `Registered candidate under ${reference.robot}; the bundle contains it, but this run will not attach it.`
            : "You can still choose a motion separately."}
          tone={reference && referenceSelected ? "good" : "neutral"}
        />
        <SummaryCell
          icon="globe"
          label="Training environment"
          title={world.included ? "Source digest only" : "Not bundled"}
          description={world.included
            ? "A digest of the declared source-world manifest was recorded. Its bytes were discarded and cannot be activated; this run keeps the project's current world."
            : "This run keeps the project's current world."}
          tone={world.included ? "warning" : "neutral"}
        />
        <SummaryCell
          icon="file-code"
          label="Controller source"
          title={controller ? humanize(controller.kind) : "Not bundled"}
          description={controller
            ? "Only the controller declaration digest was retained. Uploaded JSON or code bytes were discarded and never executed."
            : "No controller source was declared."}
          tone={controller ? "warning" : "neutral"}
        />
      </div>

      {mode === "actor_critic" && (
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 8,
            padding: "9px 11px",
            borderRadius: "var(--radius-md)",
            background: "var(--st-amber-bg)",
            color: "var(--st-amber-fg)",
            fontSize: 11.5,
            lineHeight: 1.45,
          }}
        >
          <Icon name="alert-triangle" size={14} />
          <span><strong>Advanced transfer.</strong> A critic trained on a different reward can slow or destabilize early updates.</span>
        </div>
      )}

      {receipt.warnings.length > 0 && (
        <div
          role="status"
          style={{
            padding: "9px 11px",
            borderRadius: "var(--radius-md)",
            border: "1px solid color-mix(in srgb, var(--st-amber) 35%, transparent)",
            background: "var(--st-amber-bg)",
            color: "var(--st-amber-fg)",
            fontSize: 11.5,
          }}
        >
          <strong>Review before training</strong>
          <ul style={{ margin: "5px 0 0", paddingLeft: 18 }}>
            {receipt.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        </div>
      )}

      <details style={{ fontSize: 11.5 }}>
        <summary style={{ cursor: "pointer", color: "var(--body)", fontWeight: 600 }}>
          Validation and provenance details
        </summary>
        <dl
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(120px, auto) 1fr",
            gap: "6px 12px",
            margin: "10px 0 0",
            padding: 12,
            border: "1px solid var(--hairline)",
            borderRadius: "var(--radius-md)",
            background: "var(--canvas-soft)",
          }}
        >
          <dt className="rs-sub">Robot / task</dt>
          <dd style={{ margin: 0 }}>{receipt.skill.robot_slug || "Not reported"} · {receipt.skill.task_id || "Not reported"}</dd>
          <dt className="rs-sub">Adapter</dt>
          <dd className="mono" style={{ margin: 0, overflowWrap: "anywhere" }}>{receipt.skill.adapter_class || "Not reported"}</dd>
          <dt className="rs-sub">Source format</dt>
          <dd style={{ margin: 0 }}>{sourceFormat ? humanize(sourceFormat) : "Not reported by this bundle"}</dd>
          <dt className="rs-sub">Retained checkpoint</dt>
          <dd style={{ margin: 0 }}>{checkpointFormat ? humanize(checkpointFormat) : "No retained checkpoint"}</dd>
          <dt className="rs-sub">Trust action</dt>
          <dd style={{ margin: 0 }}>
            {receipt.trust.detail
              ?? (receipt.trust.status === "sanitized"
                ? "Server-sanitized; the raw uploaded checkpoint is not loaded at training time."
                : receipt.trust.status === "verified_local"
                  ? "Generated and verified by this RewardSculptor installation."
                  : `Unknown (${receipt.trust.status || "not reported"}); treat as untrusted.`)}
          </dd>
          <dt className="rs-sub">Bundle roles</dt>
          <dd style={{ margin: 0 }}>{receipt.components.policy_roles.length ? receipt.components.policy_roles.join(", ") : "None"}</dd>
          <dt className="rs-sub">Excluded</dt>
          <dd style={{ margin: 0 }}>{receipt.components.excluded.length ? receipt.components.excluded.join(", ") : "None reported"}</dd>
          <dt className="rs-sub">Controller metadata</dt>
          <dd style={{ margin: 0 }}>
            {receipt.components.controller
              ? `${receipt.components.controller.kind} · declaration digest only; uploaded implementation discarded and never executed`
              : "Not bundled"}
          </dd>
          <dt className="rs-sub">Interface contract</dt>
          <dd style={{ margin: 0 }}>{contractSummary(receipt.skill.compatibility_contract)}</dd>
          <dt className="rs-sub">Contract digest</dt>
          <dd style={{ margin: 0 }}><DigestValue value={receipt.skill.compatibility_contract_digest} /></dd>
          <dt className="rs-sub">Contract origin</dt>
          <dd style={{ margin: 0 }}>
            {humanize(receipt.skill.compatibility_contract_provenance_status)}
            {" · "}<DigestValue value={receipt.skill.compatibility_contract_provenance_digest} />
          </dd>
          <dt className="rs-sub">Tensor contract</dt>
          <dd
            aria-label={`Tensor contract ${receipt.skill.tensor_contract_verified ? "verified" : "not verified"}`}
            style={{ margin: 0 }}
          >
            {receipt.skill.tensor_contract_verified ? "Verified" : "Not verified"}
            {" · "}<DigestValue value={receipt.skill.tensor_signature_sha256} />
          </dd>
          <dt className="rs-sub">Import identity</dt>
          <dd style={{ margin: 0 }}><DigestValue value={receipt.skill.identity_digest} /></dd>
          <dt className="rs-sub">Uploaded source weights</dt>
          <dd style={{ margin: 0 }}><DigestValue value={receipt.skill.source_weights_sha256} /></dd>
          <dt className="rs-sub">Reference bytes / provenance fields</dt>
          <dd style={{ margin: 0 }}>
            <DigestValue value={receipt.skill.reference_sha256} />
            {" · "}<DigestValue value={receipt.skill.reference_provenance_sha256} />
          </dd>
          {reference?.admission && (
            <>
              <dt className="rs-sub">Reference admission</dt>
              <dd style={{ margin: 0 }}>
                {humanize(reference.admission.status)} · training {reference.admission.training_authorized ? "authorized" : "not authorized"}. {reference.admission.next_gate}
              </dd>
            </>
          )}
          <dt className="rs-sub">Discarded world / controller declarations</dt>
          <dd style={{ margin: 0 }}>
            <DigestValue value={receipt.skill.world_bundle_sha256} />
            {" · "}<DigestValue value={receipt.skill.controller_sha256} />
          </dd>
          <dt className="rs-sub">Manifest digest</dt>
          <dd style={{ margin: 0 }}><DigestValue value={receipt.skill.manifest_digest} /></dd>
          <dt className="rs-sub">Checkpoint SHA-256</dt>
          <dd style={{ margin: 0 }}><DigestValue value={receipt.trust.checkpoint_sha256} /></dd>
          <dt className="rs-sub">Checkpoint size</dt>
          <dd style={{ margin: 0 }}>{formatBytes(receipt.skill.checkpoint_size_bytes)}</dd>
          {receipt.compatibility.reasons.length > 0 && (
            <>
              <dt className="rs-sub">Compatibility notes</dt>
              <dd style={{ margin: 0 }}>{receipt.compatibility.reasons.join(" ")}</dd>
            </>
          )}
        </dl>
      </details>
    </div>
  );
}

function InitializationChoices({
  receipt,
  value,
  onChange,
}: {
  receipt: StartingSkillReceipt;
  value: StartingSkillInitializationMode | null;
  onChange: (mode: StartingSkillInitializationMode) => void;
}) {
  const allowed = receipt.compatibility.allowed_initialization_modes;
  const visibleModes: StartingSkillInitializationMode[] = [
    "actor_only",
    "actor_critic",
    "full_resume",
    ...(allowed.includes("reference_only") ? ["reference_only" as const] : []),
  ];
  return (
    <fieldset style={{ margin: 0, padding: 0, border: 0 }}>
      <legend style={{ fontSize: 12, fontWeight: 650, marginBottom: 7 }}>
        What should training load?
      </legend>
      <div role="radiogroup" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: 7 }}>
        {visibleModes.map((mode) => {
          const copy = INITIALIZATION_COPY[mode];
          const enabled = allowed.includes(mode);
          const modeReasons = receipt.compatibility.mode_reasons?.[mode] ?? [];
          const unavailableReason = modeReasons.length > 0
            ? modeReasons.join(" ")
            : mode === "full_resume"
              ? "Unavailable unless every training-state contract matches exactly."
              : "Not available in this bundle.";
          return (
            <button
              key={mode}
              type="button"
              role="radio"
              aria-checked={value === mode}
              disabled={!enabled}
              title={!enabled ? unavailableReason : undefined}
              onClick={() => onChange(mode)}
              style={{
                padding: 10,
                textAlign: "left",
                borderRadius: "var(--radius-md)",
                border: `1px solid ${value === mode ? "var(--rs-primary)" : "var(--hairline)"}`,
                background: value === mode ? "rgba(245,78,0,0.05)" : "var(--surface-card)",
                color: "var(--ink)",
                font: "inherit",
                cursor: enabled ? "pointer" : "not-allowed",
                opacity: enabled ? 1 : 0.48,
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11.5, fontWeight: 650 }}>
                {value === mode && <Icon name="check" size={13} color="var(--rs-primary)" />}
                {copy.label}
                {copy.advanced && <span className="rs-badge amber" style={{ fontSize: 8, height: 17, padding: "0 6px" }}>advanced</span>}
              </span>
              <span className="rs-sub" style={{ display: "block", marginTop: 4, fontSize: 10.5, lineHeight: 1.35 }}>
                {enabled
                  ? copy.description
                  : unavailableReason}
              </span>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

export function StartingPointPickerDialog({
  slug,
  projectRobot,
  projectTaskId,
  checkpoints = [],
  checkpointsLoading = false,
  value,
  onChange,
  onClose,
}: StartingPointPickerDialogProps) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryKey = useMemo(() => ["starting-skills", slug] as const, [slug]);
  const [draft, setDraft] = useState<StartingPointSelection>({ ...DEFAULT_SELECTION, ...value });
  const [projectPolicySource, setProjectPolicySource] = useState<ProjectPolicySource>(
    value.warm_start_snapshot ? "interrupted" : "completed",
  );
  const [dragOver, setDragOver] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  useEffect(() => {
    setDraft({ ...DEFAULT_SELECTION, ...value });
    setProjectPolicySource(value.warm_start_snapshot ? "interrupted" : "completed");
  }, [value]);

  const skillsQuery = useQuery({
    queryKey,
    queryFn: () => listStartingSkills(slug),
  });
  const recoverySnapshotsQuery = useQuery({
    queryKey: ["policy-recovery-snapshots", slug],
    queryFn: () => listPolicyRecoverySnapshots(slug),
    enabled: draft.kind === "project_checkpoint"
      && projectPolicySource === "interrupted",
    staleTime: 10_000,
    retry: false,
  });
  const recoverySnapshots = recoverySnapshotsQuery.data ?? [];
  const receipts = skillsQuery.data?.skills ?? [];
  const evaluatedCheckpoints = useMemo(
    () => checkpoints.filter(isDeployablePolicy),
    [checkpoints],
  );
  const selectedReceipt = receipts.find(
    (receipt) => receipt.skill.skill_id === draft.starting_skill_id,
  ) ?? null;
  const selectedCheckpoint = evaluatedCheckpoints.find(
    (checkpoint) => checkpoint.iter_index === draft.warm_start_iteration,
  ) ?? null;
  const selectedRecoverySnapshot = recoverySnapshots.find(
    (snapshot) => snapshot.snapshot_id === draft.warm_start_snapshot?.snapshot_id,
  ) ?? null;

  const upload = useMutation({
    mutationFn: (file: File) => uploadStartingSkill(slug, file),
    onSuccess: (receipt) => {
      queryClient.setQueryData<StartingSkillsResponse>(queryKey, (current) => ({
        skills: [
          receipt,
          ...(current?.skills ?? []).filter(
            (candidate) => candidate.skill.skill_id !== receipt.skill.skill_id,
          ),
        ],
      }));
      setFileError(null);
      const initializationMode = preferredInitialization(receipt);
      const attachRequiredReference = initializationMode === "reference_only";
      setDraft({
        kind: "shared_skill",
        warm_start_iteration: null,
        starting_skill_id: receipt.skill.skill_id,
        initialization_mode: initializationMode,
        reference_clip_id: attachRequiredReference
          ? receipt.components.reference?.clip_id ?? null
          : null,
        reference_robot: attachRequiredReference
          ? receipt.components.reference?.robot ?? null
          : null,
        import_manifest_digest: receipt.skill.manifest_digest,
        compatibility_contract_provenance_status:
          receipt.skill.compatibility_contract_provenance_status ?? null,
        acknowledge_legacy_reconstructed_initialization: false,
        policy_contract_migration:
          receipt.compatibility.policy_contract_migration ?? null,
      });
      if (receipt.selectable && isAdmittedTrust(receipt)
          && hasImmutableManifest(receipt)) {
        toast.success("Starting skill validated", {
          description: "Review exactly what will load, then use this starting point.",
        });
      } else {
        toast.warning("Bundle imported as a blocked candidate", {
          description: "Open its receipt to see the compatibility or trust blocker. It cannot be selected for a run.",
        });
      }
    },
    onError: (error) => setFileError(problemMessage(error)),
  });

  const submitBundle = (file: File) => {
    const lowerName = file.name.toLowerCase();
    if (!ACCEPTED_BUNDLE_EXTENSIONS.some((extension) => lowerName.endsWith(extension))) {
      setFileError("Choose a data-only RewardSculptor .rskill. Deployment .zip bundles and raw checkpoints are not portable imports.");
      return;
    }
    if (file.size === 0) {
      setFileError("This file is empty.");
      return;
    }
    setFileError(null);
    upload.mutate(file);
  };

  const chooseKind = (kind: StartingPointKind) => {
    if (kind === "scratch") {
      setProjectPolicySource("completed");
      setDraft({ ...DEFAULT_SELECTION });
      return;
    }
    if (kind === "project_checkpoint") {
      if (draft.kind === "project_checkpoint") {
        setProjectPolicySource(
          draft.warm_start_snapshot ? "interrupted" : "completed",
        );
        return;
      }
      setProjectPolicySource("completed");
      const marked = evaluatedCheckpoints.filter((checkpoint) => checkpoint.selected);
      const canonicalSelection = marked.length === 1 ? marked[0] : null;
      setDraft({
        ...DEFAULT_SELECTION,
        kind,
        warm_start_iteration: canonicalSelection?.iter_index ?? null,
      });
      return;
    }
    setProjectPolicySource("completed");
    const remembered = receipts.find(
      (receipt) => receipt.skill.skill_id === draft.starting_skill_id,
    );
    const firstCompatible = receipts.find(
      (receipt) => receipt.selectable && isAdmittedTrust(receipt)
        && hasImmutableManifest(receipt),
    );
    const choice = remembered ?? firstCompatible;
    const initializationMode = remembered
      ? draft.initialization_mode
      : choice ? preferredInitialization(choice) : null;
    const attachRequiredReference = initializationMode === "reference_only";
    setDraft({
      ...DEFAULT_SELECTION,
      kind,
      starting_skill_id: choice?.skill.skill_id ?? null,
      initialization_mode: initializationMode,
      reference_clip_id: attachRequiredReference
        ? choice?.components.reference?.clip_id ?? null
        : null,
      reference_robot: attachRequiredReference
        ? choice?.components.reference?.robot ?? null
        : null,
      import_manifest_digest: choice?.skill.manifest_digest ?? null,
      compatibility_contract_provenance_status:
        choice?.skill.compatibility_contract_provenance_status ?? null,
      acknowledge_legacy_reconstructed_initialization: false,
      policy_contract_migration:
        choice?.compatibility.policy_contract_migration ?? null,
    });
  };

  const chooseProjectPolicySource = (source: ProjectPolicySource) => {
    setProjectPolicySource(source);
    if (source === "completed") {
      const marked = evaluatedCheckpoints.filter((checkpoint) => checkpoint.selected);
      const canonicalSelection = marked.length === 1 ? marked[0] : null;
      setDraft((current) => ({
        ...DEFAULT_SELECTION,
        kind: "project_checkpoint",
        warm_start_iteration: current.warm_start_snapshot == null
          ? current.warm_start_iteration
          : canonicalSelection?.iter_index ?? null,
      }));
      return;
    }
    setDraft({
      ...DEFAULT_SELECTION,
      kind: "project_checkpoint",
    });
  };

  const chooseRecoverySnapshot = (snapshot: PolicyRecoverySnapshot) => {
    setDraft({
      ...DEFAULT_SELECTION,
      kind: "project_checkpoint",
      warm_start_snapshot: {
        snapshot_id: snapshot.snapshot_id,
        checkpoint_sha256: snapshot.checkpoint_sha256,
        receipt_digest: snapshot.receipt_digest,
        acknowledge_interrupted_snapshot: false,
        acknowledge_legacy_reconstructed_snapshot: false,
      },
      warm_start_snapshot_display: {
        iteration: snapshot.iteration,
        ppo_step: snapshot.ppo_step,
        last_observed_ppo_iteration: snapshot.last_observed_ppo_iteration,
        checkpoint_bytes: snapshot.checkpoint_bytes,
        provenance_status: snapshot.provenance_status,
      },
    });
  };

  const chooseReceipt = (receipt: StartingSkillReceipt) => {
    const initializationMode = preferredInitialization(receipt);
    const attachRequiredReference = initializationMode === "reference_only";
    setDraft({
      kind: "shared_skill",
      warm_start_iteration: null,
      starting_skill_id: receipt.skill.skill_id,
      initialization_mode: initializationMode,
      reference_clip_id: attachRequiredReference
        ? receipt.components.reference?.clip_id ?? null
        : null,
      reference_robot: attachRequiredReference
        ? receipt.components.reference?.robot ?? null
        : null,
      import_manifest_digest: receipt.skill.manifest_digest,
      compatibility_contract_provenance_status:
        receipt.skill.compatibility_contract_provenance_status ?? null,
      acknowledge_legacy_reconstructed_initialization: false,
      policy_contract_migration:
        receipt.compatibility.policy_contract_migration ?? null,
    });
  };

  const checkpointValid = selectedCheckpoint !== null
    && draft.warm_start_iteration === selectedCheckpoint.iter_index
    && /^[a-f0-9]{64}$/.test(selectedCheckpoint.checkpoint_sha256);
  const snapshotRef = draft.warm_start_snapshot ?? null;
  const interruptedSnapshotValid = selectedRecoverySnapshot?.selectable === true
    && snapshotRef?.snapshot_id === selectedRecoverySnapshot.snapshot_id
    && snapshotRef.checkpoint_sha256 === selectedRecoverySnapshot.checkpoint_sha256
    && snapshotRef.receipt_digest === selectedRecoverySnapshot.receipt_digest
    && /^[a-f0-9]{64}$/.test(snapshotRef.checkpoint_sha256)
    && /^[a-f0-9]{64}$/.test(snapshotRef.receipt_digest)
    && snapshotRef.acknowledge_interrupted_snapshot
    && (
      selectedRecoverySnapshot.provenance_status !== "legacy_reconstructed"
      || snapshotRef.acknowledge_legacy_reconstructed_snapshot === true
    );
  const requiredReferenceValid = draft.initialization_mode !== "reference_only"
    || (
      draft.reference_clip_id === selectedReceipt?.components.reference?.clip_id
      && draft.reference_robot === selectedReceipt?.components.reference?.robot
    );
  const legacyReconstructedPolicy = (
    selectedReceipt?.skill.compatibility_contract_provenance_status
      === "legacy_reconstructed"
    && draft.initialization_mode !== "reference_only"
  );
  const sharedSkillValid = selectedReceipt?.selectable === true
    && isAdmittedTrust(selectedReceipt, draft.initialization_mode)
    && hasImmutableManifest(selectedReceipt)
    && draft.initialization_mode != null
    && selectedReceipt.compatibility.allowed_initialization_modes.includes(draft.initialization_mode)
    && requiredReferenceValid
    && (
      !legacyReconstructedPolicy
      || draft.acknowledge_legacy_reconstructed_initialization
    );
  const canApply = draft.kind === "scratch"
    || (
      draft.kind === "project_checkpoint"
      && (
        projectPolicySource === "completed"
          ? checkpointValid
          : interruptedSnapshotValid
      )
    )
    || (draft.kind === "shared_skill" && sharedSkillValid);

  return (
    <Modal
      wide
      icon="package"
      title="Choose a starting point"
      subtitle="Reuse a working behavior without hiding what the next run will load."
      onClose={() => { if (!upload.isPending) onClose(); }}
      footer={
        <>
          <Btn kind="quiet" onClick={onClose} disabled={upload.isPending}>Cancel</Btn>
          <Btn
            kind="primary"
            iconRight="arrow-right"
            disabled={!canApply || upload.isPending}
            onClick={() => {
              onChange(draft.kind === "shared_skill"
                ? draft
                : { ...draft, initialization_mode: null });
              onClose();
            }}
          >
            Use this starting point
          </Btn>
        </>
      }
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div role="radiogroup" aria-label="Starting point type" className="rs-starting-point-kinds">
          <KindChoice
            kind="scratch"
            selected={draft.kind === "scratch"}
            icon="sparkles"
            title="From scratch"
            description="Use the project robot, reward, and world with newly initialized networks."
            onSelect={chooseKind}
          />
          <KindChoice
            kind="project_checkpoint"
            selected={draft.kind === "project_checkpoint"}
            icon="history"
            title="Project policy"
            description="Choose an evaluated policy or an explicitly acknowledged recovery checkpoint."
            onSelect={chooseKind}
          />
          <KindChoice
            kind="shared_skill"
            selected={draft.kind === "shared_skill"}
            icon="package"
            title="Imported skill"
            description="Validate a data-only .rskill containing portable weights, motion, and declarative provenance. Imports register candidates; they do not certify a reference for live training."
            onSelect={chooseKind}
          />
        </div>

        {draft.kind === "scratch" && (
          <div className="rs-card" style={{ padding: 16 }}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
              <Icon name="sparkles" size={17} color="var(--rs-primary)" />
              <div>
                <div style={{ fontSize: 13, fontWeight: 650 }}>A clean experiment</div>
                <div className="rs-sub" style={{ marginTop: 3, fontSize: 12, lineHeight: 1.5 }}>
                  No policy weights are imported: actor and critic start newly
                  initialized. The Starting motion and Training environment in
                  New Run remain separate choices and are not cleared here.
                </div>
              </div>
            </div>
          </div>
        )}

        {draft.kind === "project_checkpoint" && (
          <div className="rs-card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
            <fieldset style={{ margin: 0, padding: 0, border: 0 }}>
              <legend style={{ fontSize: 12, fontWeight: 650, marginBottom: 7 }}>
                Choose project policy evidence
              </legend>
              <div role="radiogroup" aria-label="Project policy source" className="rs-project-policy-sources">
                <button
                  type="button"
                  role="radio"
                  aria-checked={projectPolicySource === "completed"}
                  tabIndex={projectPolicySource === "completed" ? 0 : -1}
                  onClick={() => chooseProjectPolicySource("completed")}
                  onKeyDown={moveWithinRadioGroup}
                  className={projectPolicySource === "completed" ? "selected" : undefined}
                >
                  <span>Evaluated policies</span>
                  <small>Checkpoint with a server-validated completion receipt</small>
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={projectPolicySource === "interrupted"}
                  tabIndex={projectPolicySource === "interrupted" ? 0 : -1}
                  onClick={() => chooseProjectPolicySource("interrupted")}
                  onKeyDown={moveWithinRadioGroup}
                  className={projectPolicySource === "interrupted" ? "selected" : undefined}
                >
                  <span>Interrupted or unevaluated</span>
                  <small>Server-attested recovery input; not deployment evidence</small>
                </button>
              </div>
            </fieldset>

            {projectPolicySource === "completed" ? (
              <>
                <div>
                  <label htmlFor="starting-point-checkpoint" style={{ display: "block", fontSize: 12, fontWeight: 650, marginBottom: 6 }}>
                    Evaluated iteration
                  </label>
                  {checkpointsLoading ? (
                    <div role="status" className="rs-sub" style={{ fontSize: 11.5 }}>
                      Loading evaluated policies…
                    </div>
                  ) : evaluatedCheckpoints.length > 0 ? (
                    <select
                      id="starting-point-checkpoint"
                      className="rs-input"
                      value={draft.warm_start_iteration ?? ""}
                      onChange={(event) => setDraft((current) => ({
                        ...current,
                        warm_start_iteration: event.target.value === "" ? null : Number(event.target.value),
                      }))}
                    >
                      <option value="">Choose an iteration…</option>
                      {[...evaluatedCheckpoints]
                        .sort((a, b) => b.iter_index - a.iter_index)
                        .map((checkpoint) => (
                          <option key={checkpoint.iter_index} value={checkpoint.iter_index}>
                            Iteration {checkpoint.iter_index}
                            {checkpoint.selected ? " · selected" : ""}
                            {checkpoint.criterion_status && checkpoint.criterion_status !== "not_recorded"
                              ? ` · criterion ${checkpoint.criterion_status}`
                              : ""}
                            {checkpoint.rollout_available ? " · rollout" : ""}
                            {checkpoint.fitness != null ? ` · score ${checkpoint.fitness.toFixed(3)}` : ""}
                          </option>
                        ))}
                    </select>
                  ) : (
                    <div role="status" className="rs-sub" style={{ fontSize: 11.5, lineHeight: 1.45 }}>
                      No evaluated policy is available. Preserved checkpoints appear under
                      Interrupted or unevaluated only after server attestation.
                    </div>
                  )}
                </div>
                {selectedCheckpoint ? (
                  <CheckpointReceipt checkpoint={selectedCheckpoint} />
                ) : evaluatedCheckpoints.length > 0 ? (
                  <div
                    role="status"
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 8,
                      padding: "10px 12px",
                      borderRadius: "var(--radius-md)",
                      background: "var(--st-amber-bg)",
                      color: "var(--st-amber-fg)",
                      fontSize: 11.5,
                      lineHeight: 1.45,
                    }}
                  >
                    <Icon name="alert-triangle" size={14} />
                    <span>
                      Choose a checkpoint explicitly. RewardSculptor does not assume the newest
                      checkpoint is the best; only a coherent selection receipt may preselect one.
                    </span>
                  </div>
                ) : null}
                <div style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "10px 12px", borderRadius: "var(--radius-md)", background: "var(--st-blue-bg)", color: "var(--st-blue-fg)", fontSize: 11.5, lineHeight: 1.45 }}>
                  <Icon name="info" size={14} />
                  <span>
                    This is a policy transfer, not a full resume: actor and critic load, while optimizer state, iteration counters, and exploration state start fresh.
                  </span>
                </div>
              </>
            ) : (
              <>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "10px 12px", borderRadius: "var(--radius-md)", background: "var(--st-amber-bg)", color: "var(--st-amber-fg)", fontSize: 11.5, lineHeight: 1.45 }}>
                  <Icon name="alert-triangle" size={14} />
                  <span>
                    Warm-start from an interrupted or unevaluated checkpoint only when preserving
                    partial training is worth the uncertainty. It is not an evaluated policy, full
                    resume, deployment artifact, or success claim.
                  </span>
                </div>
                {recoverySnapshotsQuery.isLoading ? (
                  <div role="status" className="rs-sub" style={{ fontSize: 11.5 }}>
                    Verifying recovery receipts…
                  </div>
                ) : recoverySnapshotsQuery.isError ? (
                  <div role="alert" style={{ color: "var(--st-rose-fg)", fontSize: 11.5, lineHeight: 1.45 }}>
                    Recovery snapshots could not be verified: {problemMessage(recoverySnapshotsQuery.error)}{" "}
                    No manual path or iteration fallback is allowed.
                  </div>
                ) : recoverySnapshots.length === 0 ? (
                  <div role="status" className="rs-sub" style={{ fontSize: 11.5, lineHeight: 1.45 }}>
                    No server-attested recovery checkpoints are available for this project.
                  </div>
                ) : (
                  <div role="radiogroup" aria-label="Interrupted or unevaluated checkpoints" className="rs-recovery-snapshot-list">
                    {[...recoverySnapshots]
                      .sort((a, b) => b.iteration - a.iteration || b.ppo_step - a.ppo_step)
                      .map((snapshot, index) => {
                        const selected = snapshot.snapshot_id === snapshotRef?.snapshot_id;
                        const descriptionId = `recovery-${snapshot.snapshot_id}-description`;
                        return (
                          <button
                            key={snapshot.snapshot_id}
                            type="button"
                            role="radio"
                            aria-checked={selected}
                            tabIndex={selected || (!snapshotRef && index === 0) ? 0 : -1}
                            aria-label={`Cycle ${snapshot.iteration}, PPO snapshot ${snapshot.ppo_step}, unevaluated recovery input, ${snapshot.selectable ? "selectable" : "blocked"}`}
                            aria-describedby={descriptionId}
                            onClick={() => chooseRecoverySnapshot(snapshot)}
                            onKeyDown={moveWithinRadioGroup}
                            className={selected ? "selected" : undefined}
                          >
                            <span className="rs-recovery-snapshot-title">
                              <strong>Cycle {snapshot.iteration} · PPO snapshot {snapshot.ppo_step}</strong>
                              <span className="rs-badge amber">recovery input</span>
                              <span className="rs-badge slate">unevaluated</span>
                              {!snapshot.selectable && <span className="rs-badge rose">blocked</span>}
                            </span>
                            <small id={descriptionId}>
                              Actor + critic transfer · optimizer/counters reset · SHA {shortDigest(snapshot.checkpoint_sha256)} · {formatBytes(snapshot.checkpoint_bytes)}
                            </small>
                          </button>
                        );
                      })}
                  </div>
                )}
                {selectedRecoverySnapshot ? (
                  <RecoverySnapshotReceipt
                    snapshot={selectedRecoverySnapshot}
                    acknowledged={snapshotRef?.acknowledge_interrupted_snapshot === true}
                    legacyAcknowledged={snapshotRef?.acknowledge_legacy_reconstructed_snapshot === true}
                    onAcknowledgedChange={(checked) => setDraft((current) => ({
                      ...current,
                      warm_start_snapshot: current.warm_start_snapshot
                        ? { ...current.warm_start_snapshot, acknowledge_interrupted_snapshot: checked }
                        : null,
                    }))}
                    onLegacyAcknowledgedChange={(checked) => setDraft((current) => ({
                      ...current,
                      warm_start_snapshot: current.warm_start_snapshot
                        ? { ...current.warm_start_snapshot, acknowledge_legacy_reconstructed_snapshot: checked }
                        : null,
                    }))}
                  />
                ) : recoverySnapshots.length > 0 && !recoverySnapshotsQuery.isLoading ? (
                  <div role="status" className="rs-sub" style={{ fontSize: 11.5 }}>
                    Choose one recovery checkpoint explicitly. None is selected by recency.
                  </div>
                ) : null}
              </>
            )}
          </div>
        )}

        {draft.kind === "shared_skill" && (
          <div className="rs-starting-point-workbench">
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div
                aria-busy={upload.isPending}
                onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragOver(false);
                  const file = event.dataTransfer.files?.[0];
                  if (file && !upload.isPending) submitBundle(file);
                }}
                className={`rs-drop${dragOver ? " over" : ""}`}
                style={{ padding: "20px 14px", opacity: upload.isPending ? 0.62 : 1 }}
              >
                <Icon name={upload.isPending ? "loader" : "upload"} size={23} className={upload.isPending ? "rs-spin" : undefined} color="var(--rs-muted)" />
                <div style={{ fontSize: 12.5, fontWeight: 650 }}>
                  {upload.isPending ? "Quarantining and validating…" : "Drop a data-only .rskill"}
                </div>
                <div className="rs-sub" style={{ fontSize: 10.5, lineHeight: 1.4 }}>
                  Deployment ZIPs, raw checkpoints, Python, TorchScript, native binaries, and unknown members are rejected before validation.
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".rskill"
                  style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)", clipPath: "inset(50%)", whiteSpace: "nowrap" }}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) submitBundle(file);
                    event.currentTarget.value = "";
                  }}
                />
                <Btn kind="ghost" size="xs" icon="upload" disabled={upload.isPending} onClick={(event) => {
                  event.stopPropagation();
                  fileInputRef.current?.click();
                }}>
                  Choose bundle
                </Btn>
              </div>

              <details
                style={{
                  padding: "9px 11px",
                  border: "1px solid var(--hairline)",
                  borderRadius: "var(--radius-md)",
                  fontSize: 11,
                  lineHeight: 1.45,
                }}
              >
                <summary style={{ cursor: "pointer", fontWeight: 650 }}>
                  What goes in a portable skill?
                </summary>
                <div className="rs-sub" style={{ marginTop: 7 }}>
                  A ZIP-compatible <code className="mono">.rskill</code> has a
                  content-addressed <code className="mono">manifest.json</code>{" "}
                  plus compatible <code className="mono">policy/weights.safetensors</code>,
                  a bounded <code className="mono">motion/clip.npz</code> with
                  provenance, or both. A bounded controller JSON and source-world
                  manifest can be validated and hashed, but their uploaded bytes are
                  discarded and never become executable inputs. Importing a motion
                  stores its exact clip and provenance as a candidate. Before live
                  training, run a separate <code className="mono">sculpt refs track</code>{" "}
                  Tier-D exact-schedule tracking evidence job; New Run only
                  re-verifies the resulting evidence and never creates that
                  certificate. Export
                  a policy bundle with <code className="mono">sculpt export --portable --robot {projectRobot ?? "<project-robot>"} --config &lt;project&gt;/config.toml</code>,
                  or a motion-only bundle with <code className="mono">sculpt refs export-skill --robot {projectRobot ?? "<project-robot>"} --clip &lt;clip&gt; --out motion.rskill</code>.
                  The normal <code className="mono">sculpt export</code> ZIP is a
                  separate operator deployment artifact and cannot be uploaded here.
                </div>
              </details>

              {fileError && (
                <div role="alert" style={{ display: "flex", alignItems: "flex-start", gap: 7, padding: "9px 10px", borderRadius: "var(--radius-md)", background: "var(--st-rose-bg)", color: "var(--st-rose-fg)", fontSize: 11.5 }}>
                  <Icon name="alert-circle" size={14} />
                  <span>{fileError}</span>
                </div>
              )}

              <div>
                <div className="rs-starting-point-scope">
                  <span style={{ fontSize: 11.5, fontWeight: 650 }}>Imported candidates</span>
                  {projectRobot && <span className="rs-badge slate">{projectRobot}{projectTaskId ? ` · ${projectTaskId}` : ""}</span>}
                </div>
                <div role="radiogroup" aria-label="Validated starting skills" style={{ display: "flex", maxHeight: 245, overflowY: "auto", flexDirection: "column", gap: 6 }}>
                  {skillsQuery.isLoading && (
                    <div className="rs-sub" style={{ display: "flex", alignItems: "center", gap: 7, padding: 10, fontSize: 11.5 }}>
                      <Icon name="loader" size={14} className="rs-spin" /> Loading imported candidates…
                    </div>
                  )}
                  {skillsQuery.isError && (
                    <div role="alert" style={{ padding: 10, borderRadius: "var(--radius-md)", background: "var(--st-rose-bg)", color: "var(--st-rose-fg)", fontSize: 11.5 }}>
                      Could not load the skill library. {problemMessage(skillsQuery.error)}
                    </div>
                  )}
                  {!skillsQuery.isLoading && !skillsQuery.isError && receipts.length === 0 && (
                    <div className="rs-sub" style={{ padding: 12, border: "1px dashed var(--hairline-strong)", borderRadius: "var(--radius-md)", textAlign: "center", fontSize: 11.5 }}>
                      No portable starting-point candidates have been imported for this project yet.
                    </div>
                  )}
                  {receipts.map((receipt) => {
                    const selected = draft.starting_skill_id === receipt.skill.skill_id;
                    return (
                      <button
                        key={receipt.skill.skill_id}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        onClick={() => chooseReceipt(receipt)}
                        style={{
                          padding: "10px 11px",
                          textAlign: "left",
                          borderRadius: "var(--radius-md)",
                          border: `1px solid ${selected ? "var(--rs-primary)" : "var(--hairline)"}`,
                          background: selected ? "rgba(245,78,0,0.05)" : "var(--surface-card)",
                          color: "var(--ink)",
                          font: "inherit",
                          cursor: "pointer",
                        }}
                      >
                        <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                          <span style={{ minWidth: 0, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 12, fontWeight: 650 }}>
                            {receipt.skill.alias || receipt.skill.skill_id}
                          </span>
                          <span className={`rs-badge ${receipt.selectable && isAdmittedTrust(receipt) && hasImmutableManifest(receipt) ? "amber" : "rose"}`}>
                            {receipt.selectable && isAdmittedTrust(receipt) && hasImmutableManifest(receipt)
                              ? receipt.skill.compatibility_contract_provenance_status === "legacy_reconstructed"
                                ? "legacy reconstructed"
                                : "candidate"
                              : "blocked"}
                          </span>
                        </span>
                        <span className="rs-sub" style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4, fontSize: 10.5 }}>
                          <span>{receipt.skill.robot_slug || "robot unknown"}</span>
                          <span>·</span>
                          <span>{receipt.skill.task_id || "task unknown"}</span>
                          <span>·</span>
                          <span className="mono">{shortDigest(receipt.skill.manifest_digest)}</span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="rs-card" style={{ padding: 14, minHeight: 260 }}>
              {selectedReceipt ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 13.5, fontWeight: 700 }}>{selectedReceipt.skill.alias || selectedReceipt.skill.skill_id}</div>
                    <div className="rs-sub" style={{ marginTop: 3, fontSize: 11 }}>
                      Immutable manifest <span className="mono">{shortDigest(selectedReceipt.skill.manifest_digest)}</span>
                    </div>
                  </div>
                  {selectedReceipt.selectable && (
                    <>
                      <InitializationChoices
                        receipt={selectedReceipt}
                        value={draft.initialization_mode}
                        onChange={(mode) => setDraft((current) => ({
                          ...current,
                          initialization_mode: mode,
                          acknowledge_legacy_reconstructed_initialization: false,
                          reference_clip_id: mode === "reference_only"
                            ? selectedReceipt.components.reference?.clip_id ?? null
                            : current.reference_clip_id,
                          reference_robot: mode === "reference_only"
                            ? selectedReceipt.components.reference?.robot ?? null
                            : current.reference_robot,
                        }))}
                      />
                      {legacyReconstructedPolicy && (
                        <div
                          role="note"
                          style={{
                            display: "grid",
                            gap: 9,
                            padding: "11px 12px",
                            border: "1px solid var(--st-amber-fg)",
                            borderRadius: "var(--radius-md)",
                            background: "var(--st-amber-bg)",
                            color: "var(--ink)",
                          }}
                        >
                          <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                            <Icon name="alert-triangle" size={15} />
                            <div style={{ display: "grid", gap: 3 }}>
                              <strong style={{ fontSize: 11.5 }}>Historical contract reconstruction</strong>
                              <span style={{ fontSize: 11, lineHeight: 1.45 }}>
                                Reconstructed from immutable historical evidence. Suitable only for actor/critic initialization—not exact resume or optimizer restoration.
                              </span>
                            </div>
                          </div>
                          <label
                            htmlFor="legacy-reconstructed-initialization-ack"
                            style={{ display: "flex", alignItems: "flex-start", gap: 8, fontSize: 11, lineHeight: 1.4, cursor: "pointer" }}
                          >
                            <input
                              id="legacy-reconstructed-initialization-ack"
                              type="checkbox"
                              checked={draft.acknowledge_legacy_reconstructed_initialization}
                              onChange={(event) => setDraft((current) => ({
                                ...current,
                                acknowledge_legacy_reconstructed_initialization: event.target.checked,
                              }))}
                            />
                            <span>I understand this compatibility contract was reconstructed after training from retained evidence.</span>
                          </label>
                        </div>
                      )}
                      {selectedReceipt.components.reference && (
                        <label
                          style={{
                            display: "flex",
                            alignItems: "flex-start",
                            gap: 8,
                            padding: "9px 11px",
                            border: "1px solid var(--hairline)",
                            borderRadius: "var(--radius-md)",
                            fontSize: 11.5,
                            lineHeight: 1.4,
                          }}
                        >
                          <input
                            type="checkbox"
                            checked={
                              draft.reference_clip_id === selectedReceipt.components.reference.clip_id
                              && draft.reference_robot === selectedReceipt.components.reference.robot
                            }
                            disabled={draft.initialization_mode === "reference_only"}
                            onChange={(event) => setDraft((current) => ({
                              ...current,
                              reference_clip_id: event.target.checked
                                ? selectedReceipt.components.reference?.clip_id ?? null
                                : null,
                              reference_robot: event.target.checked
                                ? selectedReceipt.components.reference?.robot ?? null
                                : null,
                            }))}
                          />
                          <span>
                            <strong>Attach bundled motion</strong><br />
                            This is independent from policy transfer and can be cleared or replaced in Starting motion.
                            {draft.initialization_mode === "reference_only" && " Motion-only initialization requires it."}
                          </span>
                        </label>
                      )}
                    </>
                  )}
                  <ReceiptSummary
                    receipt={selectedReceipt}
                    mode={draft.initialization_mode}
                    referenceSelected={
                      draft.reference_clip_id
                        === selectedReceipt.components.reference?.clip_id
                      && draft.reference_robot
                        === selectedReceipt.components.reference?.robot
                    }
                  />
                </div>
              ) : (
                <div style={{ display: "flex", minHeight: 230, alignItems: "center", justifyContent: "center", textAlign: "center" }}>
                  <div style={{ maxWidth: 270 }}>
                    <Icon name="shield-check" size={26} color="var(--rs-muted)" />
                    <div style={{ marginTop: 8, fontSize: 12.5, fontWeight: 650 }}>Select a validation receipt</div>
                    <div className="rs-sub" style={{ marginTop: 4, fontSize: 11.5, lineHeight: 1.45 }}>
                      You will see retained weight roles and hashes, the registered motion candidate, discarded source-world declarations, trust actions, and exclusions before anything is selected.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}

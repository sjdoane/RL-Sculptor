import { useState } from "react";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { ModeTimeline } from "@/components/ModeTimeline";
import { Btn, EmptyState, Modal } from "@/components/rs/primitives";
import { useComposeReference, useReferenceSearch } from "@/hooks/useReferences";
import { ApiError } from "@/lib/api";
import type { ComposeResult } from "@/lib/types";

/** A phase of the motion being built, as edited in this dialog. */
type Draft = {
  key: number;
  clipId: string | null;
  clipText: string;
  label: string;
  startS: string;
  endS: string;
};

let nextKey = 1;
const emptyDraft = (): Draft => ({
  key: nextKey++, clipId: null, clipText: "", label: "",
  startS: "", endS: "",
});

/** Slugify a free-text motion name into the library's clip-id charset
 *  (lowercase alnum + `_-`, must start alnum). The backend rejects anything
 *  else, so mirror it here rather than surfacing a 400. */
function toClipId(name: string, robot: string): string {
  const base = name.toLowerCase().replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+/, "").replace(/-+$/, "").slice(0, 80);
  return base ? `${base}--${robot}` : "";
}

type ParsedNumber = { value: number | null; error: string | null };

export const MAX_COMPOSE_FPS = 240;
export const MAX_COMPOSE_BLEND_S = 10;
export const MAX_COMPOSE_SEGMENTS = 16;
const DECIMAL_NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

/** Parse a researcher-entered numeric field without turning malformed input
 * into an omitted/default value. `Number("1,5")` and `Number("abc")` are both
 * invalid, not aliases for a blank field. */
export function parseNumericInput(
  raw: string,
  label: string,
  opts: {
    optional?: boolean;
    minimum?: number;
    exclusiveMinimum?: boolean;
    maximum?: number;
  } = {},
): ParsedNumber {
  const text = raw.trim();
  if (!text) {
    return opts.optional === false
      ? { value: null, error: `${label} is required.` }
      : { value: null, error: null };
  }
  const value = Number(text);
  if (!DECIMAL_NUMBER.test(text) || !Number.isFinite(value)) {
    return {
      value: null,
      error: `${label} must be a number using a decimal point (for example 0.2).`,
    };
  }
  if (opts.minimum != null) {
    const invalid = opts.exclusiveMinimum
      ? value <= opts.minimum
      : value < opts.minimum;
    if (invalid) {
      return {
        value: null,
        error: opts.exclusiveMinimum
          ? `${label} must be greater than ${opts.minimum}.`
          : `${label} must be at least ${opts.minimum}.`,
      };
    }
  }
  if (opts.maximum != null && value > opts.maximum) {
    return {
      value: null,
      error: `${label} must be at most ${opts.maximum}.`,
    };
  }
  return { value, error: null };
}

function parseTrim(draft: Draft): {
  start: number | null;
  end: number | null;
  error: string | null;
} {
  const start = parseNumericInput(draft.startS, "Trim start", { minimum: 0 });
  const end = parseNumericInput(draft.endS, "Trim end", { minimum: 0 });
  const orderError = start.value != null && end.value != null
    && end.value <= start.value
    ? "Trim end must be greater than trim start."
    : null;
  return {
    start: start.value,
    end: end.value,
    error: start.error ?? end.error ?? orderError,
  };
}

/** Recover cadence from a sampled trajectory's exact `(N - 1) / fps`
 * duration contract. This must match the runtime reference clock. */
export function sampledTrajectoryFps(nFrames: number, durationS: number): number {
  if (!Number.isFinite(nFrames) || nFrames < 1
    || !Number.isFinite(durationS) || durationS <= 0) return 0;
  return Math.max(1, nFrames - 1) / durationS;
}

/** Exact library identity accepted by deterministic retrieval. The search
 * endpoint is already robot-filtered, while the copyable UI receipt includes
 * that namespace; accepting both forms makes paste-back lossless. */
export function isExactReferenceQuery(
  query: string, robot: string, clipId: string,
): boolean {
  const normalized = query.trim().toLowerCase();
  const id = clipId.toLowerCase();
  return normalized === id || normalized === `${robot.toLowerCase()}/${id}`;
}

/** One phase row: search the library, pick a clip, optionally trim it. */
function SegmentRow({
  draft, index, robot, canRemove, trimError, onChange, onRemove, onMove,
}: {
  draft: Draft;
  index: number;
  robot: string;
  canRemove: boolean;
  trimError: string | null;
  onChange: (patch: Partial<Draft>) => void;
  onRemove: () => void;
  onMove: (delta: number) => void;
}) {
  const [query, setQuery] = useState("");
  const trimmed = query.trim();
  const search = useReferenceSearch(trimmed, { robot, enabled: trimmed.length > 1 });
  const trimErrorId = `compose-phase-${draft.key}-trim-error`;

  return (
    <div
      style={{
        border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)",
        padding: 10, background: "var(--surface-strong)",
      }}
    >
      <div className="rs-flex rs-gap-8" style={{ marginBottom: 8, alignItems: "center" }}>
        <span className="rs-badge slate" style={{ fontSize: 9 }}>phase {index + 1}</span>
        <input
          value={draft.label}
          onChange={(e) => onChange({ label: e.target.value })}
          placeholder="name this phase, e.g. approach"
          aria-label={`Phase ${index + 1} label`}
          style={{
            flex: 1, minWidth: 0, fontSize: 12, height: 26, padding: "0 8px",
            border: "1px solid var(--hairline)", borderRadius: "var(--radius-sm)",
            background: "var(--canvas-soft)", color: "var(--ink)", outline: "none",
          }}
        />
        <Btn kind="quiet" icon="chevron-up" aria-label={`Move phase ${index + 1} earlier`}
          onClick={() => onMove(-1)} disabled={index === 0} />
        <Btn kind="quiet" icon="chevron-down" aria-label={`Move phase ${index + 1} later`}
          onClick={() => onMove(1)} />
        <Btn kind="quiet" icon="trash" aria-label={`Remove phase ${index + 1}`}
          onClick={onRemove} disabled={!canRemove} />
      </div>

      {draft.clipId ? (
        <div
          className="rs-flex rs-gap-8"
          style={{
            alignItems: "center", padding: "6px 8px",
            border: "1px solid var(--rs-primary)", borderRadius: "var(--radius-sm)",
            background: "rgba(245,78,0,0.05)",
          }}
        >
          <Icon name="video" size={13} color="var(--rs-primary)" />
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {draft.clipText || draft.clipId}
            </div>
            <div
              className="mono"
              style={{ fontSize: 10, color: "var(--rs-muted)", overflowWrap: "anywhere" }}
            >
              {robot}/{draft.clipId}
            </div>
          </div>
          <Btn kind="quiet" onClick={() => onChange({ clipId: null, clipText: "" })}>
            Change
          </Btn>
        </div>
      ) : (
        <>
          <div
            className="rs-flex rs-gap-8"
            style={{
              background: "var(--canvas-soft)", border: "1px solid var(--hairline-strong)",
              borderRadius: "var(--radius-sm)", padding: "0 9px", height: 28,
            }}
          >
            <Icon name="search" size={13} color="var(--rs-muted)" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Describe a motion or paste ${robot}/clip_id…`}
              aria-label={`Search a clip for phase ${index + 1}`}
              style={{
                border: 0, background: "none", outline: "none", fontSize: 12,
                width: "100%", color: "var(--ink)",
              }}
            />
          </div>
          {search.isLoading && <p className="rs-sub" style={{ marginTop: 6 }}>Searching…</p>}
          {trimmed.length > 1 && !search.isLoading && search.isError && (
            <p role="alert" className="rs-sub" style={{ marginTop: 6, fontSize: 11 }}>
              Reference search is unavailable. Retry; this is not evidence that the clip is absent.
            </p>
          )}
          {trimmed.length > 1 && !search.isLoading && !search.isError
            && (search.data ?? []).length === 0 && (
            <p className="rs-sub" style={{ marginTop: 6, fontSize: 11 }}>No matches.</p>
          )}
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6, maxHeight: 168, overflowY: "auto" }}>
            {!search.isError && (search.data ?? []).slice(0, 6).map((m) => {
              const exactId = isExactReferenceQuery(trimmed, robot, m.clip_id);
              return (
                <button
                  key={m.clip_id}
                  type="button"
                  onClick={() => onChange({ clipId: m.clip_id, clipText: m.text })}
                  style={{
                    textAlign: "left", padding: "5px 8px", cursor: "pointer",
                    border: "1px solid var(--hairline)", borderRadius: "var(--radius-sm)",
                    background: "var(--canvas-soft)", font: "inherit", color: "inherit",
                  }}
                >
                  <div style={{ fontSize: 11.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {m.text}
                  </div>
                  <div
                    className="mono"
                    title={`${robot}/${m.clip_id}`}
                    style={{
                      marginTop: 2, fontSize: 9.5, color: "var(--rs-muted)",
                      overflowWrap: "anywhere",
                    }}
                  >
                    {robot}/{m.clip_id}
                  </div>
                  <div
                    className="rs-sub"
                    style={{
                      marginTop: 2, fontSize: 9.5, display: "flex",
                      flexWrap: "wrap", alignItems: "center", gap: 5,
                    }}
                  >
                    <span>declared tier {m.tier}</span>
                    <span>· {m.duration_s.toFixed(1)}s</span>
                    {exactId ? (
                      <span className="rs-badge slate" style={{ fontSize: 8.5 }}>
                        exact ID match
                      </span>
                    ) : (
                      <span>· score {m.score.toFixed(1)}</span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}

      <div className="rs-flex rs-gap-8" style={{ marginTop: 8, alignItems: "center" }}>
        <span className="rs-sub" style={{ fontSize: 10.5 }}>trim (s)</span>
        <input
          value={draft.startS}
          onChange={(e) => onChange({ startS: e.target.value })}
          placeholder="start"
          inputMode="decimal"
          aria-label={`Phase ${index + 1} start seconds`}
          aria-invalid={trimError ? "true" : undefined}
          aria-describedby={trimError ? trimErrorId : undefined}
          style={{
            width: 64, fontSize: 11.5, height: 24, padding: "0 6px",
            border: "1px solid var(--hairline)", borderRadius: "var(--radius-sm)",
            background: "var(--canvas-soft)", color: "var(--ink)", outline: "none",
          }}
        />
        <span className="rs-sub" style={{ fontSize: 10.5 }}>→</span>
        <input
          value={draft.endS}
          onChange={(e) => onChange({ endS: e.target.value })}
          placeholder="end"
          inputMode="decimal"
          aria-label={`Phase ${index + 1} end seconds`}
          aria-invalid={trimError ? "true" : undefined}
          aria-describedby={trimError ? trimErrorId : undefined}
          style={{
            width: 64, fontSize: 11.5, height: 24, padding: "0 6px",
            border: "1px solid var(--hairline)", borderRadius: "var(--radius-sm)",
            background: "var(--canvas-soft)", color: "var(--ink)", outline: "none",
          }}
        />
        <span className="rs-sub" style={{ fontSize: 10, marginLeft: "auto" }}>
          blank = whole clip
        </span>
      </div>
      {trimError && (
        <div id={trimErrorId} role="alert" style={{ marginTop: 5, fontSize: 10, color: "var(--st-red)" }}>
          {trimError}
        </div>
      )}
    </div>
  );
}

/** The seam measurements, shown rather than reduced to a badge — this is
 *  what tells you whether the composite is worth a tracking run. */
function ResultPanel(
  { result, segmentLabels }: { result: ComposeResult; segmentLabels: string[] },
) {
  const comp = result.qc.composition;
  const worst = comp.seams.reduce(
    (acc, s) => Math.max(acc, s.max_joint_jump_rad ?? 0), 0);
  return (
    <div
      style={{
        border: "1px solid var(--hairline-strong)", borderRadius: "var(--radius-md)",
        padding: 12, background: "var(--canvas-soft)",
      }}
    >
      <div className="rs-flex rs-gap-8" style={{ alignItems: "center", marginBottom: 8 }}>
        <Icon name="check" size={15} color="var(--rs-primary)" />
        <strong style={{ fontSize: 13 }}>{result.clip_id}</strong>
        <span className="rs-badge slate" style={{ fontSize: 9 }}>
          declared tier {result.tier}
        </span>
        <span className="rs-badge amber" style={{ fontSize: 9 }}>
          tracking evidence unverified
        </span>
      </div>
      <div
        style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(118px, 1fr))",
          gap: 8, marginBottom: 10,
        }}
      >
        {[
          ["duration", `${result.qc.duration_s.toFixed(2)} s`],
          ["sources", String(result.qc.n_sources)],
          ["seams", String(comp.seams.length)],
          ["worst seam", `${worst.toFixed(3)} rad`],
          ["peak joint vel", comp.peak_joint_vel_rad_s != null
            ? `${comp.peak_joint_vel_rad_s.toFixed(1)} rad/s` : "—"],
        ].map(([k, v]) => (
          <div key={k}>
            <div className="rs-sub" style={{ fontSize: 10 }}>{k}</div>
            <div style={{ fontSize: 12.5, fontWeight: 500 }}>{v}</div>
          </div>
        ))}
      </div>
      <div className="rs-sub" style={{ fontSize: 10.5, marginBottom: 10 }}>
        composed from {result.parent_clip_ids.join(" → ")}
      </div>

      {/* The composite's segments become an OGMP-inspired phase scaffold —
          same derivation the library performs, shown before the clip is
          attached and trained on. */}
      <div style={{ marginBottom: 10 }}>
        <ModeTimeline
          segments={result.parent_clip_ids.map((id, i) => ({
            index: i, label: segmentLabels[i], source_id: id,
          }))}
          seamFrames={comp.seams.map((s) => s.frame)}
          nFrames={result.qc.n_frames}
          fps={sampledTrajectoryFps(result.qc.n_frames, result.qc.duration_s)}
        />
      </div>
      <div className="rs-banner" style={{ fontSize: 11 }}>
        <Icon name="alert-triangle" size={15} />
        <span className="rs-grow">{result.next_step}</span>
      </div>
    </div>
  );
}

/**
 * Build a reference for a motion that exists in NO single library clip.
 *
 * The library holds thousands of solved trajectories, but a novel goal's
 * phases are usually spread across several of them — the approach in one
 * clip, the launch in another, the landing in a third. This dialog stitches
 * those spans into one candidate reference so the tracking-first reward has
 * something real to track, instead of the goal falling through to a
 * synthesized sketch or running blind.
 */
export function ComposeMotionDialog({
  robot,
  onClose,
  onComposed,
}: {
  robot: string;
  onClose: () => void;
  /** Fired after a successful compose so the caller can select the new
   *  clip — it is normally the one about to be attached. */
  onComposed?: (clipId: string) => void;
}) {
  const [name, setName] = useState("");
  const [drafts, setDrafts] = useState<Draft[]>(() => [emptyDraft(), emptyDraft()]);
  const [result, setResult] = useState<ComposeResult | null>(null);
  // Labels as they were AT SUBMIT — editing a phase afterwards must not
  // relabel a composite that is already registered.
  const [composedLabels, setComposedLabels] = useState<string[]>([]);
  // Seam handling. The backend has always taken these; the dialog never sent
  // them, so every composite was built at blend_s=0.2 / strict / source fps.
  // That mattered most on a refusal: `strict` reports the seam gap in radians
  // and the crossfade length is the knob that closes it, so the one control
  // that could act on the error message was the one not on screen.
  const [blendS, setBlendS] = useState("0.2");
  const [targetFps, setTargetFps] = useState("");
  //: Set only by "Compose anyway" after the gate has already refused once, so
  //  turning the gate off is a reply to a specific measurement rather than a
  //  checkbox someone leaves unticked.
  const [seamRefusal, setSeamRefusal] = useState<{
    detail: string;
    requestFingerprint: string;
  } | null>(null);
  const compose = useComposeReference();

  const clipId = toClipId(name, robot);
  const parsedTrims = drafts.map(parseTrim);
  const parsedBlend = parseNumericInput(blendS, "Crossfade", {
    optional: false,
    minimum: 0,
    maximum: MAX_COMPOSE_BLEND_S,
  });
  const parsedFps = parseNumericInput(targetFps, "Target fps", {
    minimum: 1,
    maximum: MAX_COMPOSE_FPS,
  });
  const numericError = parsedTrims.some((trim) => trim.error)
    || parsedBlend.error !== null
    || parsedFps.error !== null;
  const ready = drafts.length >= 2
    && drafts.every((d) => d.clipId)
    && clipId.length > 0
    && !numericError;

  const buildRequest = (strict: boolean) => {
    const fps = parsedFps.value;
    return {
      clip_id: clipId,
      robot,
      text: name.trim(),
      labels: ["novel"],
      segments: drafts.map((d, i) => ({
        clip_id: d.clipId as string,
        label: d.label.trim() || `phase_${i + 1}`,
        t_start_s: parsedTrims[i].start,
        t_end_s: parsedTrims[i].end,
      })),
      blend_s: parsedBlend.value as number,
      ...(fps === null ? {} : { target_fps: fps }),
      strict,
    };
  };
  const requestFingerprint = ready
    ? JSON.stringify({ ...buildRequest(true), strict: undefined })
    : null;
  const activeSeamRefusal = seamRefusal
    && seamRefusal.requestFingerprint === requestFingerprint
    ? seamRefusal
    : null;
  const atSegmentLimit = drafts.length >= MAX_COMPOSE_SEGMENTS;

  const patch = (key: number, p: Partial<Draft>) =>
    setDrafts((ds) => ds.map((d) => (d.key === key ? { ...d, ...p } : d)));

  const move = (index: number, delta: number) =>
    setDrafts((ds) => {
      const to = index + delta;
      if (to < 0 || to >= ds.length) return ds;
      const next = [...ds];
      [next[index], next[to]] = [next[to], next[index]];
      return next;
    });

  const submit = (opts: { strict?: boolean } = {}) => {
    if (!ready) return;
    const strict = opts.strict !== false;
    if (!strict && !activeSeamRefusal) return;
    const request = buildRequest(strict);
    const submittedFingerprint = JSON.stringify({
      ...request,
      strict: undefined,
    });
    compose.mutate(
      request,
      {
        onSuccess: (res) => {
          setResult(res);
          setSeamRefusal(null);
          setComposedLabels(
            drafts.map((d, i) => d.label.trim() || `phase_${i + 1}`));
          toast.success("Motion composed", { description: res.clip_id });
          onComposed?.(res.clip_id);
        },
        onError: (err) => {
          const detail = err instanceof ApiError
            ? (err.problem.detail ?? err.problem.title)
            : (err as Error).message;
          // ComposeError detail carries the actual measurement (e.g. the seam
          // gap in radians), so surface it verbatim rather than a generic
          // "failed" — it is what tells the user which phase to re-pick.
          toast.error("Could not compose these phases", { description: String(detail) });
          // Only a refusal BY the gate offers to bypass the gate. Any other
          // failure (a missing clip, a span past the end) is not something
          // `strict=false` would fix, and offering it there would just teach
          // people to turn the gate off on every error.
          setSeamRefusal(
            strict && err instanceof ApiError
              && err.type === "/problems/reference-compose-gate"
              ? {
                  detail: String(detail),
                  requestFingerprint: submittedFingerprint,
                }
              : null,
          );
        },
      },
    );
  };

  return (
    <Modal
      icon="video"
      title="Compose a novel motion"
      subtitle={`Stitch spans of solved clips · ${robot}`}
      onClose={onClose}
      footer={
        <>
          <Btn kind="quiet" onClick={onClose} disabled={compose.isPending}>
            {result ? "Done" : "Cancel"}
          </Btn>
          {activeSeamRefusal && !result && (
            <Btn
              kind="danger"
              icon="alert-triangle"
              onClick={() => submit({ strict: false })}
              disabled={!ready || compose.isPending}
              title="Register the composite even though a seam exceeds the gate"
            >
              Compose anyway
            </Btn>
          )}
          <Btn
            kind="primary"
            icon={compose.isPending ? "loader" : "plus"}
            onClick={() => submit()}
            disabled={!ready || compose.isPending}
          >
            {compose.isPending ? "Composing…" : "Compose"}
          </Btn>
        </>
      }
    >
      <p className="rs-sub" style={{ fontSize: 11.5, marginBottom: 12 }}>
        For a goal no single clip covers. Each phase contributes real solved
        frames; seams are SE(2)-aligned and cross-faded, and the result is a
        kinematic candidate that still needs Tier-D exact-schedule tracking
        evidence.
      </p>

      <label className="rs-sub" style={{ fontSize: 10.5, display: "block", marginBottom: 4 }}>
        Motion name
      </label>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="e.g. running jump kick"
        aria-label="Motion name"
        autoFocus
        style={{
          width: "100%", fontSize: 13, height: 32, padding: "0 10px",
          border: "1px solid var(--hairline-strong)", borderRadius: "var(--radius-md)",
          background: "var(--canvas-soft)", color: "var(--ink)", outline: "none",
        }}
      />
      {clipId && (
        <div className="rs-sub" style={{ fontSize: 10, marginTop: 4 }}>
          clip id: <code>{clipId}</code>
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 8, margin: "14px 0" }}>
        {drafts.map((d, i) => (
          <SegmentRow
            key={d.key}
            draft={d}
            index={i}
            robot={robot}
            canRemove={drafts.length > 2}
            trimError={parsedTrims[i].error}
            onChange={(p) => patch(d.key, p)}
            onRemove={() => setDrafts((ds) => ds.filter((x) => x.key !== d.key))}
            onMove={(delta) => move(i, delta)}
          />
        ))}
      </div>

      <Btn
        kind="quiet"
        icon="plus"
        onClick={() => setDrafts((ds) => (
          ds.length >= MAX_COMPOSE_SEGMENTS ? ds : [...ds, emptyDraft()]
        ))}
        disabled={atSegmentLimit}
        aria-describedby="compose-phase-count"
      >
        Add phase
      </Btn>
      <span
        id="compose-phase-count"
        className="rs-sub"
        style={{ marginLeft: 8, fontSize: 10.5 }}
      >
        {drafts.length} of {MAX_COMPOSE_SEGMENTS} phases
        {atSegmentLimit ? " · maximum reached" : ""}
      </span>

      {!result && (
        <div
          style={{
            marginTop: 14, paddingTop: 12,
            borderTop: "1px solid var(--hairline)",
          }}
        >
          <div className="rs-sub" style={{ fontSize: 10.5, marginBottom: 6 }}>
            Seam handling
          </div>
          <div className="rs-flex rs-gap-10" style={{ alignItems: "flex-end" }}>
            <label style={{ flex: 1, minWidth: 0 }}>
              <span className="rs-sub" style={{ fontSize: 10, display: "block" }}>
                Crossfade (s)
              </span>
              <input
                value={blendS}
                onChange={(e) => setBlendS(e.target.value)}
                inputMode="decimal"
                aria-label="Crossfade seconds"
                aria-invalid={parsedBlend.error ? "true" : undefined}
                aria-describedby={parsedBlend.error ? "compose-seam-numeric-error" : undefined}
                style={{
                  width: "100%", fontSize: 12.5, height: 28, padding: "0 8px",
                  marginTop: 3,
                  border: "1px solid var(--hairline-strong)",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--canvas-soft)", color: "var(--ink)",
                  outline: "none",
                }}
              />
            </label>
            <label style={{ flex: 1, minWidth: 0 }}>
              <span className="rs-sub" style={{ fontSize: 10, display: "block" }}>
                Resample to fps (optional)
              </span>
              <input
                value={targetFps}
                onChange={(e) => setTargetFps(e.target.value)}
                inputMode="decimal"
                placeholder="source fps"
                aria-label="Target fps"
                aria-invalid={parsedFps.error ? "true" : undefined}
                aria-describedby={parsedFps.error ? "compose-seam-numeric-error" : undefined}
                style={{
                  width: "100%", fontSize: 12.5, height: 28, padding: "0 8px",
                  marginTop: 3,
                  border: "1px solid var(--hairline-strong)",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--canvas-soft)", color: "var(--ink)",
                  outline: "none",
                }}
              />
            </label>
          </div>
          {(parsedBlend.error || parsedFps.error) && (
            <div id="compose-seam-numeric-error" role="alert" style={{ fontSize: 10, marginTop: 5, color: "var(--st-red)" }}>
              {parsedBlend.error ?? parsedFps.error}
            </div>
          )}
          <div className="rs-sub" style={{ fontSize: 10, marginTop: 5, lineHeight: 1.45 }}>
            A longer crossfade is what closes a seam the gate refuses; phases
            of different fps are resampled to the first one unless you set a
            target.
          </div>
          {activeSeamRefusal && (
            <div
              role="alert"
              style={{
                marginTop: 8, fontSize: 10.8, lineHeight: 1.45,
                color: "var(--st-amber)",
              }}
            >
              <strong>Kinematic gate measurement:</strong>{" "}
              {activeSeamRefusal.detail}{" "}
              Widen
              the crossfade or move a phase boundary and compose again, or use{" "}
              <b>Compose anyway</b> to register it as measured. The result is
              declared tier K and lacks verified Tier-D exact-schedule tracking
              evidence either way. A flagged seam is therefore a warning that
              the exact schedule may not be trackable; it is not itself a
              general dynamics verdict.
            </div>
          )}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 14 }}>
          <ResultPanel result={result} segmentLabels={composedLabels} />
        </div>
      )}
      {!result && drafts.every((d) => !d.clipId) && (
        <div style={{ marginTop: 14 }}>
          <EmptyState
            icon="video"
            title="Pick a clip for each phase"
            sub="Search the library above — at least two phases are needed to compose."
          />
        </div>
      )}
    </Modal>
  );
}

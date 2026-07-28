import { Icon } from "@/components/rs/icon";

/**
 * The OGMP mode structure of a composed reference, drawn as a timeline.
 *
 * A composite's phases ARE its modes: `sculptor.modes.modes_from_composition`
 * derives one mode per composed segment, bounded by the seams, with a
 * transition at each seam. This renders exactly that derivation so the
 * hybrid automaton is visible at the point of use — before you attach the
 * clip and train on it — rather than only inside a library artifact.
 *
 * Derivation is intentionally mirrored from the Python rather than fetched:
 * segment order, seam frames, fps and frame count are all already on the
 * clip's provenance, so a separate endpoint would add a round trip and a
 * second place for the boundary rule to drift.
 */
export type ModeSegment = { index: number; label?: string | null; source_id?: string | null };

export type DerivedMode = {
  name: string;
  startFrame: number;
  endFrame: number;
  startS: number;
  endS: number;
  sourceId: string | null;
};

/** Segment i spans [boundary[i], boundary[i+1]) — the same tiling
 *  `modes_from_composition` produces: no frame unowned or double-owned. */
export function deriveModes(
  segments: ModeSegment[],
  seamFrames: number[],
  nFrames: number,
  fps: number,
): DerivedMode[] {
  if (!segments.length || !nFrames || !fps) return [];
  const bounds = [0, ...seamFrames, nFrames];
  if (bounds.length !== segments.length + 1) return [];
  const used = new Set<string>();
  return segments.map((seg, i) => {
    const raw = (seg.label ?? `mode_${i + 1}`).trim().toLowerCase().replace(/\s+/g, "_");
    let name = raw || `mode_${i + 1}`;
    let k = 2;
    while (used.has(name)) name = `${raw}_${k++}`;
    used.add(name);
    return {
      name,
      startFrame: bounds[i],
      endFrame: bounds[i + 1],
      startS: bounds[i] / fps,
      endS: bounds[i + 1] / fps,
      sourceId: seg.source_id ?? null,
    };
  });
}

const BAND = ["var(--rs-primary)", "var(--st-blue, #3b82f6)", "var(--st-emerald, #10b981)",
  "var(--st-amber, #f59e0b)", "var(--st-rose, #f43f5e)"];

/** The same modes, read off a graph the backend already derived.
 *
 *  `GET /references/{clip}/modes` returns the authoritative automaton, so a
 *  caller holding one should not re-derive it from segments — that is a
 *  second place for the boundary rule to drift. */
export function modesFromGraph(graph: {
  fps: number;
  modes: { name: string; frame_range: [number, number]; start_s: number;
           end_s: number; source_clip_id: string | null }[];
}): DerivedMode[] {
  return graph.modes.map((m) => ({
    name: m.name,
    startFrame: m.frame_range[0],
    endFrame: m.frame_range[1],
    startS: m.start_s,
    endS: m.end_s,
    sourceId: m.source_clip_id,
  }));
}

export function ModeTimeline({
  segments, seamFrames, nFrames, fps, compact, modes: given, transitions,
}: {
  segments?: ModeSegment[];
  seamFrames?: number[];
  nFrames?: number;
  fps?: number;
  compact?: boolean;
  /** Pre-derived modes. Takes precedence over the segment inputs. */
  modes?: DerivedMode[];
  /** Rendered as the guard label between bands when supplied. Previously
   *  fetched alongside the mode graph and then discarded. */
  transitions?: { from_mode: string; to_mode: string; guard_kind: string;
                  at_phase: number | null }[];
}) {
  const modes = given ?? deriveModes(
    segments ?? [], seamFrames ?? [], nFrames ?? 0, fps ?? 0);
  if (!modes.length) return null;
  const total = (nFrames || modes[modes.length - 1].endFrame) || 1;

  return (
    <div>
      <div className="rs-flex rs-gap-8" style={{ alignItems: "center", marginBottom: 6 }}>
        <Icon name="git-branch" size={13} color="var(--rs-muted)" />
        <span className="rs-sub" style={{ fontSize: 10.5 }}>
          {modes.length} modes · {modes.length - 1} guarded transition
          {modes.length - 1 === 1 ? "" : "s"}
          {transitions?.length ? (
            <>
              {" · "}
              {Array.from(new Set(transitions.map((t) => t.guard_kind))).join(", ")}
              {" guards"}
            </>
          ) : null}
        </span>
      </div>

      {/* Proportional band: width is the mode's share of the clip. */}
      <div
        style={{
          display: "flex", width: "100%", height: compact ? 8 : 12,
          borderRadius: "var(--radius-sm)", overflow: "hidden",
          border: "1px solid var(--hairline)",
        }}
        role="img"
        aria-label={`Mode timeline: ${modes.map((m) => m.name).join(", ")}`}
      >
        {modes.map((m, i) => (
          <div
            key={m.name}
            title={`${m.name} · ${m.startS.toFixed(2)}–${m.endS.toFixed(2)}s`}
            style={{
              width: `${((m.endFrame - m.startFrame) / total) * 100}%`,
              background: BAND[i % BAND.length],
              opacity: 0.85,
              borderRight: i < modes.length - 1 ? "2px solid var(--surface-card)" : undefined,
            }}
          />
        ))}
      </div>

      {!compact && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 8 }}>
          {modes.map((m, i) => (
            <div key={m.name} className="rs-flex rs-gap-8" style={{ alignItems: "center", fontSize: 11 }}>
              <span
                style={{
                  width: 8, height: 8, borderRadius: 2, flex: "0 0 auto",
                  background: BAND[i % BAND.length],
                }}
              />
              <code style={{ fontSize: 10.5 }}>{m.name}</code>
              <span className="rs-sub" style={{ fontSize: 10 }}>
                {m.startS.toFixed(2)}–{m.endS.toFixed(2)}s
              </span>
              {m.sourceId && (
                <span
                  className="rs-sub"
                  style={{
                    fontSize: 9.5, marginLeft: "auto", maxWidth: "45%",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}
                  title={m.sourceId}
                >
                  {m.sourceId}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** World tab (env-authoring item 5): the promoted world tuple + its
 *  immutable selection lineage, with the authoring dialog as entry point.
 *  The selection endpoint 404s before the first authoring — that renders
 *  as the empty state, with the dialog as the action. */
import { useState } from "react";
import { toast } from "sonner";

import AuthorWorldDialog from "@/components/AuthorWorldDialog";
import { Icon } from "@/components/rs/icon";
import { Badge, Btn, EmptyState } from "@/components/rs/primitives";
import WorldEntityInspector from "@/components/WorldEntityInspector";
import WorldViewer3D from "@/components/WorldViewer3D";
import {
  useEditWorldVariations,
  useWorldCurriculum,
  useWorldLineage,
  useWorldScene,
  useWorldSelection,
  useWorldValidation,
} from "@/hooks/useWorlds";
import type { WorldEventProgram } from "@/lib/types";
import { WORLD_ROBOT_MISMATCH_DETAIL } from "@/lib/worldLaunch";

/** "5 elements — 3 platform · 2 gap"; gaps carry no geometry, so call
 *  that out rather than letting the total look inconsistent with the
 *  rendered scene. */
export function courseBreakdownText(
  breakdown: Record<string, number> | undefined,
  total: number,
): string {
  const parts = Object.entries(breakdown ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([kind, count]) => `${count} ${kind}`);
  if (parts.length === 0) return total > 0 ? `${total} elements` : "none";
  const gaps = breakdown?.["gap"] ?? 0;
  const suffix = gaps > 0 ? " (gaps are spacing — no geometry)" : "";
  return `${total} elements — ${parts.join(" · ")}${suffix}`;
}

function eventPhaseCopy(
  phaseId: string,
  program: WorldEventProgram,
): { gate: string; detail: string } {
  if (phaseId === "route") {
    return {
      gate: "Raw goal completion",
      detail: "Complete the authored waypoint goal in order.",
    };
  }
  if (phaseId === "jump") {
    return {
      gate: `Both feet · ${program.minimum_air_time_s.toFixed(2)} s · ${program.minimum_height_delta_m.toFixed(2)} m`,
      detail: "Leave support, clear the height gate, then land on both feet.",
    };
  }
  return {
    gate: `Landing, then ${program.terminal_hold_duration_s.toFixed(1)} s`,
    detail: "Remain in the terminal phase until the hold completes.",
  };
}

/** The compact rail is the visible counterpart of the selected immutable
 * TaskSpec. It describes only the admitted linear automaton. */
export function TaskProgramCard({
  program,
}: {
  program: WorldEventProgram;
}) {
  const resetMix = program.ordered_phase_ids
    .map((phase) =>
      `${Math.round((program.train_only_phase_sampling[phase] ?? 0) * 100)}% ${phase.toUpperCase()}`)
    .join(" · ");
  return (
    <section className="rs-card" aria-labelledby="task-program-title">
      <div className="rs-card-head">
        <div>
          <div id="task-program-title" className="rs-card-title">
            Task program <Badge status="promoted" label="Immutable" />
          </div>
          <div className="rs-hintline" style={{ marginTop: 3 }}>
            One ordered attempt: finish the route, execute one jump, then hold.
          </div>
        </div>
        <code className="mono" style={{ fontSize: 11, color: "var(--rs-muted)" }}>
          {program.id}
        </code>
      </div>
      <div className="rs-card-pad" style={{ display: "grid", gap: 12 }}>
        <div
          role="region"
          aria-label="Ordered task phases; scroll horizontally on a narrow screen"
          tabIndex={0}
          style={{ overflowX: "auto", paddingBottom: 3 }}
        >
          <ol
            aria-label="ROUTE then JUMP then HOLD"
            style={{
              display: "flex", alignItems: "stretch", minWidth: 590,
              listStyle: "none", margin: 0, padding: 0,
            }}
          >
            {program.ordered_phase_ids.map((phase, index) => {
              const copy = eventPhaseCopy(phase, program);
              return (
                <li
                  key={phase}
                  style={{
                    display: "flex", alignItems: "center", flex: 1,
                    minWidth: 0,
                  }}
                >
                  <div
                    style={{
                      alignSelf: "stretch", flex: 1, minWidth: 0,
                      border: "1px solid var(--hairline)",
                      borderTop: "2px solid var(--rs-primary)",
                      borderRadius: "var(--radius-md)",
                      background: "var(--surface-strong)", padding: "10px 11px",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
                      <span className="mono" aria-hidden style={{ fontSize: 10, color: "var(--rs-muted)" }}>
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <strong style={{ fontSize: 12.5, letterSpacing: ".05em" }}>
                        {phase.toUpperCase()}
                      </strong>
                    </div>
                    <div style={{ marginTop: 7, fontSize: 11.5, fontWeight: 650 }}>
                      {copy.gate}
                    </div>
                    <div className="rs-hintline" style={{ marginTop: 3, lineHeight: 1.4 }}>
                      {copy.detail}
                    </div>
                  </div>
                  {index < program.ordered_phase_ids.length - 1 && (
                    <span
                      aria-hidden="true"
                      style={{
                        flex: "0 0 30px", textAlign: "center",
                        color: "var(--rs-primary)", fontSize: 18,
                      }}
                    >
                      →
                    </span>
                  )}
                </li>
              );
            })}
          </ol>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 8,
          }}
        >
          <div className="rs-hintline">
            <strong style={{ color: "var(--ink)" }}>Episode</strong><br />
            {program.episode_length_s} s maximum
          </div>
          <div className="rs-hintline">
            <strong style={{ color: "var(--ink)" }}>Training resets only</strong><br />
            {resetMix}
          </div>
          <div className="rs-hintline">
            <strong style={{ color: "var(--ink)" }}>Evaluation</strong><br />
            Always starts at {program.evaluation_start_phase.toUpperCase()}
          </div>
          <div className="rs-hintline">
            <strong style={{ color: "var(--ink)" }}>Policy input</strong><br />
            <code className="mono">{program.observation_extension.term}</code>
            {" · "}{program.observation_extension.width}-wide one-hot
          </div>
        </div>

        <details>
          <summary style={{ cursor: "pointer", fontSize: 11.5, fontWeight: 650 }}>
            Exact program JSON and provenance
          </summary>
          <pre
            className="mono"
            style={{
              margin: "8px 0 0", maxHeight: 300, overflow: "auto",
              border: "1px solid var(--hairline)",
              borderRadius: "var(--radius-md)",
              background: "var(--canvas-soft)", padding: 11,
              fontSize: 10.5, lineHeight: 1.45, whiteSpace: "pre-wrap",
              overflowWrap: "anywhere",
            }}
          >
            {JSON.stringify(program, null, 2)}
          </pre>
        </details>
      </div>
    </section>
  );
}

export default function WorldTab({
  slug, launchAction,
}: {
  slug: string;
  launchAction?: React.ReactNode;
}) {
  const selection = useWorldSelection(slug);
  const validation = useWorldValidation(slug, !!selection.data);
  const lineage = useWorldLineage(slug);
  const curriculum = useWorldCurriculum(slug);
  const scene = useWorldScene(slug, !!selection.data);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const editVariations = useEditWorldVariations(slug);
  const runIntegrityCheck = async () => {
    const result = await validation.refetch();
    if (result.data?.ok) {
      toast.success(
        `Integrity verified: selection v${result.data.selection_version}, ` +
        "every artifact hash matches",
      );
    } else {
      toast.error(
        `Integrity FAILED: ${result.data?.errors.join("; ") || result.error?.message || "unknown error"}`,
      );
    }
  };
  const onEditVariation = (
    variationId: string, distribution: Record<string, unknown>,
  ) => {
    editVariations.mutate(
      { edits: [{ variation_id: variationId, distribution }] },
      {
        onSuccess: (result) => {
          if (result.applied.length > 0 && result.selection) {
            toast.success(
              `Variation ${variationId} updated — promoted selection `
              + `v${result.selection.selection_version} (evaluation `
              + "unchanged)");
          } else {
            toast.error(`Edit rejected: ${result.rejected
              .map((r) => r.reason).join("; ") || "no change applied"}`);
          }
        },
        onError: (err) =>
          toast.error(err instanceof Error ? err.message : String(err)),
      },
    );
  };

  if (selection.isLoading) return null;

  if (!selection.data) {
    return (
      <EmptyState
        icon="globe"
        title="No authored world yet"
        sub={"Describe the terrain, obstacles, objects, and goal in one "
          + "prompt — the system drafts a parametric world, asks about "
          + "load-bearing choices (each with a system-decides default), "
          + "then gates and promotes it atomically."}
        action={<AuthorWorldDialog slug={slug} />}
      />
    );
  }

  const s = selection.data;
  const entries = lineage.data ?? [];
  const robotMismatch = s.shared_summary.robot_matches_project === false;
  return (
    <div style={{ display: "grid", gap: 14 }}>
      {robotMismatch && (
        <div className="rs-banner warn">
          <Icon name="alert-triangle" size={17} />
          <span className="rs-grow">
            <b>Training blocked: robot differs from the project.</b> This world
            was authored
            for <code className="mono">{s.shared_summary.robot}</code>, but the
            project&apos;s configured robot is{" "}
            <code className="mono">{s.shared_summary.project_capability_id}</code>.
            {" "}{WORLD_ROBOT_MISMATCH_DETAIL} Leave the robot field blank
            while re-authoring to use the project robot.
          </span>
        </div>
      )}
      {validation.data && !validation.data.ok && (
        <div className="rs-banner err">
          <Icon name="alert-triangle" size={17} />
          <span className="rs-grow">
            <b>Training blocked: the authored tuple failed integrity verification.</b>{" "}
            {validation.data.errors.join("; ")}
          </span>
        </div>
      )}
      {s.event_program && <TaskProgramCard program={s.event_program} />}
      <div className="rs-card">
        <div className="rs-card-head">
          <div className="rs-card-title">
            Authoritative world tuple
            {validation.data?.ok && (
              <Badge status="completed" label="Verified for launch" />
            )}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <Btn icon="shield-check" size="sm"
                 onClick={() => void runIntegrityCheck()}
                 disabled={validation.isFetching}>
              {validation.isFetching ? "Verifying…" : "Verify integrity"}
            </Btn>
            <AuthorWorldDialog slug={slug} />
            {validation.data?.ok && launchAction}
          </div>
        </div>
        <div className="rs-card-pad">
          <div className="rs-kv">
            <span>Selection</span>
            <span className="mono">
              v{s.selection.selection_version} · {s.selection.tuple_hash.slice(0, 12)}
            </span>
            <span>Lineage</span>
            <span className="mono">{s.selection.evaluation_lineage}</span>
            <span>Robot</span>
            <span>
              {s.shared_summary.robot ?? "—"}
              {robotMismatch && (
                <span style={{ color: "var(--st-amber-fg)", marginLeft: 8 }}>
                  ≠ project robot ({s.shared_summary.project_capability_id})
                </span>
              )}
            </span>
            <span>Terrain</span>
            <span>{s.shared_summary.terrain_kind ?? "plane"}</span>
            <span>Course</span>
            <span>
              {courseBreakdownText(s.shared_summary.course_breakdown,
                s.shared_summary.course_elements)}
            </span>
            <span>Objects / zones</span>
            <span>
              {s.shared_summary.objects.length} object(s) ·{" "}
              {s.shared_summary.zones.length} zone(s)
            </span>
            <span>Goal</span>
            <span className="mono">{String(s.goal["type"] ?? "—")}</span>
            <span>Prompt</span>
            <span>{s.world_meta.prompt ?? "—"}</span>
            <span>Clarifications</span>
            <span>
              {Object.entries(s.clarifications?.answer_sources ?? {})
                .map(([source, count]) => `${count} ${source}`)
                .join(" · ") || "—"}
            </span>
          </div>
          {s.train_variations.length > 0 && (
            <>
              <div className="rs-card-title" style={{ marginTop: 12 }}>
                Train variations (diagnoser-editable)
              </div>
              {s.train_variations.map((v) => (
                <div key={v.id} className="rs-hintline mono">
                  {v.id} → {v.target} ({v.class}):{" "}
                  {JSON.stringify(v.distribution)}
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      {(curriculum.data?.iterations?.length ?? 0) > 0 && (
        <div className="rs-card">
          <div className="rs-card-head">
            <div className="rs-card-title">
              Terrain curriculum — {curriculum.data!.run}
            </div>
          </div>
          <div className="rs-card-pad">
            <div className="rs-hintline">
              Mean difficulty level per iteration (rising = the policy is
              earning promotion to harder terrain rows).
            </div>
            {curriculum.data!.iterations.map((entry) => (
              <div key={entry.iter}
                   style={{ display: "flex", gap: 10, alignItems: "baseline",
                            padding: "2px 0" }}>
                <span className="mono" style={{ width: 56 }}>
                  iter {entry.iter}
                </span>
                <span className="mono">
                  mean {entry.mean_level ?? "?"} / max {entry.max_level ?? "?"}
                </span>
                <span className="rs-hintline">
                  {Object.entries(entry.histogram ?? {})
                    .map(([level, count]) => `L${level}:${count}`)
                    .join(" ")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="rs-card">
        <div className="rs-card-head">
          <div className="rs-card-title">Evaluation scene</div>
        </div>
        <div className="rs-card-pad">
          {scene.data ? (
            <div style={{ display: "grid",
                          gridTemplateColumns: "minmax(0, 2fr) minmax(240px, 1fr)",
                          gap: 14, alignItems: "start" }}>
              <WorldViewer3D
                scene={scene.data}
                selectedEntity={selectedEntity}
                onSelectEntity={setSelectedEntity}
              />
              <WorldEntityInspector
                scene={scene.data}
                selected={selectedEntity}
                onSelect={setSelectedEntity}
                onEditVariation={onEditVariation}
                editBusy={editVariations.isPending}
              />
            </div>
          ) : scene.isLoading ? (
            <div className="rs-flex rs-gap-8 rs-sub">
              <Icon name="loader" size={14} className="rs-spin" />
              Loading compiled scene…
            </div>
          ) : (
            <img
              src={`/api/projects/${slug}/worlds/preview?v=${s.selection.selection_version}`}
              alt="Materialized evaluation scene"
              style={{ width: "100%", borderRadius: 8, display: "block" }}
            />
          )}
          <div className="rs-hintline" style={{ marginTop: 6 }}>
            Built from the materialized evaluation model — exactly the
            scene fitness is scored on. Drag to orbit, scroll to zoom,
            right-drag to pan, click an element to inspect it.
          </div>
        </div>
      </div>

      <div className="rs-card">
        <div className="rs-card-head">
          <div className="rs-card-title">Selection lineage</div>
        </div>
        <div className="rs-card-pad">
          {entries.length === 0 && (
            <div className="rs-hintline">No promoted selections yet.</div>
          )}
          {[...entries].reverse().map((entry) => {
            const isCurrent = entry.tuple_hash === s.selection.tuple_hash;
            return (
            <div key={entry.selection_version}
                 style={{ display: "flex", gap: 10, alignItems: "baseline",
                          padding: "4px 0" }}>
              <Badge
                status={isCurrent ? "promoted" : "superseded"}
                label={isCurrent
                  ? `v${entry.selection_version} · promoted`
                  : `v${entry.selection_version}`}
              />
              <span className="mono" style={{ fontSize: 12 }}>
                {entry.tuple_hash.slice(0, 12)}
              </span>
              <span className="rs-hintline" style={{ flex: 1 }}>
                world {String(entry.refs["world"]?.version ?? "?")} · task{" "}
                {String(entry.refs["task"]?.version ?? "?")} · reward{" "}
                {String(entry.refs["reward"]?.version ?? "?")} · eval{" "}
                <span className="mono">
                  {entry.eval_model_hash
                    ? entry.eval_model_hash.replace("sha256:", "").slice(0, 12)
                    : "?"}
                </span>{" "}
                · {entry.evaluation_lineage}
              </span>
              <span className="rs-hintline">
                {new Date(entry.created_at * 1000).toLocaleString()}
              </span>
            </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

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
            <b>Robot differs from the project.</b> This world was authored
            for <code className="mono">{s.shared_summary.robot}</code>, but the
            project&apos;s configured robot is{" "}
            <code className="mono">{s.shared_summary.project_capability_id}</code>.
            Training under this selection uses the authored world&apos;s robot.
            Re-author (leave the robot field blank to use the project robot)
            if that is not intended.
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

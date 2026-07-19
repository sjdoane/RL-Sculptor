/** World tab (env-authoring item 5): the promoted world tuple + its
 *  immutable selection lineage, with the authoring dialog as entry point.
 *  The selection endpoint 404s before the first authoring — that renders
 *  as the empty state, with the dialog as the action. */
import AuthorWorldDialog from "@/components/AuthorWorldDialog";
import { Badge, EmptyState } from "@/components/rs/primitives";
import { useWorldLineage, useWorldSelection } from "@/hooks/useWorlds";

export default function WorldTab({ slug }: { slug: string }) {
  const selection = useWorldSelection(slug);
  const lineage = useWorldLineage(slug);

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
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div className="rs-card">
        <div className="rs-card-head">
          <div className="rs-card-title">Authoritative world tuple</div>
          <AuthorWorldDialog slug={slug} />
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
            <span>{s.shared_summary.robot ?? "—"}</span>
            <span>Terrain</span>
            <span>{s.shared_summary.terrain_kind ?? "plane"}</span>
            <span>Course / objects / zones</span>
            <span>
              {s.shared_summary.course_elements} elements ·{" "}
              {s.shared_summary.objects.length} object(s) ·{" "}
              {s.shared_summary.zones.length} zone(s)
            </span>
            <span>Goal</span>
            <span className="mono">{String(s.goal["type"] ?? "—")}</span>
            <span>Prompt</span>
            <span>{s.world_meta.prompt ?? "—"}</span>
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

      <div className="rs-card">
        <div className="rs-card-head">
          <div className="rs-card-title">Evaluation scene</div>
        </div>
        <div className="rs-card-pad">
          <img
            src={`/api/projects/${slug}/worlds/preview?v=${s.selection.selection_version}`}
            alt="Materialized evaluation scene"
            style={{ width: "100%", borderRadius: 8, display: "block" }}
          />
          <div className="rs-hintline" style={{ marginTop: 6 }}>
            Rendered from the materialized evaluation model — exactly the
            scene fitness is scored on.
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
          {[...entries].reverse().map((entry) => (
            <div key={entry.selection_version}
                 style={{ display: "flex", gap: 10, alignItems: "baseline",
                          padding: "4px 0" }}>
              <Badge
                status={entry.tuple_hash === s.selection.tuple_hash
                  ? "running" : "draft"}
                label={`v${entry.selection_version}`}
              />
              <span className="mono" style={{ fontSize: 12 }}>
                {entry.tuple_hash.slice(0, 12)}
              </span>
              <span className="rs-hintline" style={{ flex: 1 }}>
                world v{String(entry.refs["world"]?.version ?? "?")} · task v
                {String(entry.refs["task"]?.version ?? "?")} · reward{" "}
                {String(entry.refs["reward"]?.version ?? "?")} ·{" "}
                {entry.evaluation_lineage}
              </span>
              <span className="rs-hintline">
                {new Date(entry.created_at * 1000).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

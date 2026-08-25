import { useMemo, useState } from "react";

import { Icon } from "@/components/rs/icon";
import { Banner, Modal } from "@/components/rs/primitives";
import { usePaper } from "@/hooks/useKG";
import type {
  KGEntitySummary,
  ResearchCapabilitySummary,
} from "@/lib/types";

export function PaperDetailModal({
  slug,
  arxivId,
  open,
  onOpenChange,
}: {
  slug: string;
  arxivId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { data, isLoading, error } = usePaper(
    slug,
    open ? arxivId ?? undefined : undefined,
  );
  if (!open) return null;
  return (
    <Modal
      wide
      icon="file-text"
      title={data?.title ?? (arxivId ? `arxiv:${arxivId}` : "Paper")}
      subtitle={
        data ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8, flexWrap: "wrap", width: "100%" }}>
            <span className="mono">{data.arxiv_id}</span>
            {data.year && <span>· {data.year}</span>}
            {data.authors.length > 0 && <span>· {data.authors.join(", ")}</span>}
            <a
              href={data.source_url || `https://arxiv.org/abs/${data.arxiv_id}`}
              target="_blank"
              rel="noreferrer noopener"
              style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 4, color: "var(--ink)" }}
            >
              Open pinned source <Icon name="external" size={12} />
            </a>
          </span>
        ) : undefined
      }
      onClose={() => onOpenChange(false)}
    >
      {isLoading && (
        <div className="rs-flex rs-gap-6" style={{ alignItems: "center", padding: "20px 0", color: "var(--rs-muted)", fontSize: 13 }}>
          <Icon name="loader" size={16} className="rs-spin" /> Loading paper…
        </div>
      )}
      {error && <Banner kind="err" icon="alert-triangle">{(error as Error).message}</Banner>}
      {data && (
        <>
          {data.tags.length > 0 && (
            <div className="rs-flex rs-wrap rs-gap-6" aria-label="Paper tags">
              {data.tags.map((tag) => <span key={tag} className="rs-tag mono">{tag}</span>)}
            </div>
          )}
          {data.rationale && (
            <section>
              <h3 className="rs-caption">Why this paper is here</h3>
              <p style={{ fontSize: 13, lineHeight: 1.55, margin: 0 }}>{data.rationale}</p>
            </section>
          )}
          <CapabilityGroup items={data.capabilities} />
          {data.abstract && (
            <section>
              <h3 className="rs-caption">Abstract</h3>
              <p style={{ fontSize: 13.5, lineHeight: 1.6, margin: 0 }}>{data.abstract}</p>
            </section>
          )}
          <EntityGroup title="Techniques" items={data.entities.techniques} />
          <EntityGroup title="Failure modes" items={data.entities.failure_modes} />
          <EntityGroup title="Reward components" items={data.entities.reward_components} />
          <EntityGroup title="Environments" items={data.entities.environments} />
          {!data.extracted && data.capabilities.length === 0 && (
            <p style={{ borderRadius: "var(--radius-md)", border: "1px dashed var(--hairline-strong)", padding: "8px 12px", fontSize: 12, color: "var(--rs-muted)", margin: 0 }}>
              Entities have not been extracted yet — run "Add papers" with
              <code className="mono" style={{ margin: "0 4px" }}>auto_extract=true</code> and a valid ANTHROPIC_API_KEY.
            </p>
          )}
        </>
      )}
    </Modal>
  );
}

interface ParameterRow {
  path: string;
  value: string;
}

export function flattenCapabilityParameters(
  value: Record<string, unknown>,
  prefix = "",
): ParameterRow[] {
  const rows: ParameterRow[] = [];
  for (const key of Object.keys(value).sort()) {
    const path = prefix ? `${prefix}.${key}` : key;
    const item = value[key];
    if (item !== null && typeof item === "object" && !Array.isArray(item)) {
      const nested = flattenCapabilityParameters(item as Record<string, unknown>, path);
      rows.push(...(nested.length > 0 ? nested : [{ path, value: "{}" }]));
      continue;
    }
    rows.push({
      path,
      value: Array.isArray(item) ? JSON.stringify(item) : String(item),
    });
  }
  return rows;
}

export function CapabilityGroup({
  items,
}: {
  items: ResearchCapabilitySummary[];
}) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return items;
    return items.filter((item) => (
      [
        item.name,
        item.description,
        item.scope,
        item.implementation_status,
        item.source_version,
        item.paper_role,
        JSON.stringify(item.parameters),
      ].join(" ").toLocaleLowerCase().includes(needle)
    ));
  }, [items, query]);

  if (items.length === 0) return null;
  return (
    <section aria-label="Reviewed paper capabilities">
      <div
        style={{
          border: "1px solid var(--hairline-strong)",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          background: "var(--surface-strong)",
        }}
      >
        <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--hairline)" }}>
          <div className="rs-flex-between rs-gap-12 rs-wrap">
            <div>
              <h3 className="rs-caption" style={{ margin: 0 }}>Reviewed paper mechanisms · {items.length}</h3>
              <p style={{ margin: "4px 0 0", color: "var(--rs-muted)", fontSize: 11.5, lineHeight: 1.45 }}>
                Parameters describe the cited source. Each status says whether RewardSculptor executes that mechanism today.
              </p>
            </div>
            <div style={{ position: "relative", width: "min(100%, 270px)" }}>
              <span style={{ position: "absolute", left: 10, top: 9, lineHeight: 0, pointerEvents: "none" }}>
                <Icon name="search" size={14} color="var(--rs-muted)" />
              </span>
              <input
                className="rs-input mono"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Find 50 Hz, FSQ, planner…"
                aria-label="Search reviewed paper parameters"
                style={{ paddingLeft: 30, height: 32, fontSize: 11.5 }}
              />
            </div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {filtered.map((capability) => (
            <CapabilityCard
              key={capability.id}
              capability={capability}
              expandParameters={query.trim().length > 0}
            />
          ))}
          {filtered.length === 0 && (
            <div style={{ padding: "16px 14px", fontSize: 12.5, color: "var(--rs-muted)" }}>
              No reviewed mechanism or parameter matches “{query.trim()}”.
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function CapabilityCard({
  capability,
  expandParameters,
}: {
  capability: ResearchCapabilitySummary;
  expandParameters: boolean;
}) {
  const rows = flattenCapabilityParameters(capability.parameters);
  const statusClass = capability.implementation_status === "implemented"
    ? "emerald"
    : capability.implementation_status === "metadata_only" ? "amber" : "rose";
  const statusLabel = capability.implementation_status === "metadata_only"
    ? "Metadata only"
    : capability.implementation_status === "unsupported"
      ? "Unsupported in RewardSculptor"
      : "Implemented";
  return (
    <article style={{ padding: "12px 14px", borderBottom: "1px solid var(--hairline)" }}>
      <div className="rs-flex-between rs-gap-8 rs-wrap">
        <strong style={{ fontSize: 13 }}>{capability.name}</strong>
        <span className={`rs-badge ${statusClass}`} title={capability.status_definition}>
          {statusLabel}
        </span>
      </div>
      {capability.description && (
        <p style={{ fontSize: 12, lineHeight: 1.5, color: "var(--rs-muted)", margin: "5px 0 0" }}>
          {capability.description}
        </p>
      )}
      {rows.length > 0 && (
        <details open={expandParameters || undefined} style={{ marginTop: 8 }}>
          <summary className="mono" style={{ cursor: "pointer", fontSize: 11.5, color: "var(--ink)" }}>
            {rows.length} exact source field{rows.length === 1 ? "" : "s"}
          </summary>
          <dl style={{ display: "grid", gridTemplateColumns: "minmax(150px, 0.85fr) minmax(0, 1.15fr)", margin: "8px 0 0", borderTop: "1px solid var(--hairline)" }}>
            {rows.map((row) => (
              <div key={row.path} style={{ display: "contents" }}>
                <dt className="mono" style={{ padding: "6px 8px 6px 0", borderBottom: "1px solid var(--hairline)", fontSize: 10.5, color: "var(--rs-muted)", overflowWrap: "anywhere" }}>
                  {row.path}
                </dt>
                <dd className="mono" style={{ margin: 0, padding: "6px 0 6px 8px", borderBottom: "1px solid var(--hairline)", fontSize: 10.5, overflowWrap: "anywhere" }}>
                  {row.value}
                </dd>
              </div>
            ))}
          </dl>
        </details>
      )}
      {(capability.source_version || capability.source_locator) && (
        <div className="rs-flex rs-gap-6 rs-wrap" style={{ marginTop: 8, fontSize: 10.5, color: "var(--rs-muted)" }}>
          {capability.source_version && <span className="mono">{capability.source_version}</span>}
          {capability.source_locator && (
            <a href={capability.source_locator} target="_blank" rel="noreferrer noopener" style={{ color: "var(--ink)" }}>
              Source location <Icon name="external" size={10} />
            </a>
          )}
        </div>
      )}
    </article>
  );
}

function EntityGroup({
  title,
  items,
}: {
  title: string;
  items: KGEntitySummary[];
}) {
  if (items.length === 0) return null;
  return (
    <section>
      <h3 className="rs-caption">{title} · {items.length}</h3>
      <ul style={{ display: "flex", flexDirection: "column", gap: 6, margin: 0, padding: 0, listStyle: "none" }}>
        {items.map((entity) => (
          <li key={entity.id} style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--hairline)", background: "var(--surface-strong)", padding: "7px 10px", fontSize: 12.5 }}>
            <div style={{ fontWeight: 500 }}>{entity.name}</div>
            {entity.description && <div style={{ fontSize: 11.5, color: "var(--rs-muted)", marginTop: 2 }}>{entity.description}</div>}
          </li>
        ))}
      </ul>
    </section>
  );
}

import { Icon } from "@/components/rs/icon";
import { Banner, Modal } from "@/components/rs/primitives";
import { usePaper } from "@/hooks/useKG";
import type { KGEntitySummary } from "@/lib/types";

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
              href={`https://arxiv.org/abs/${data.arxiv_id}`}
              target="_blank"
              rel="noreferrer noopener"
              style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 4, color: "var(--ink)" }}
            >
              arxiv.org <Icon name="external" size={12} />
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
          {!data.extracted && (
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
        {items.map((e) => (
          <li key={e.id} style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--hairline)", background: "var(--surface-strong)", padding: "7px 10px", fontSize: 12.5 }}>
            <div style={{ fontWeight: 500 }}>{e.name}</div>
            {e.description && <div style={{ fontSize: 11.5, color: "var(--rs-muted)", marginTop: 2 }}>{e.description}</div>}
          </li>
        ))}
      </ul>
    </section>
  );
}

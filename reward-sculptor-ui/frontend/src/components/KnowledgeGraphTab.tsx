import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";

import { ActiveJobsIndicator } from "@/components/ActiveJobsIndicator";
import { AddSeedsDialog, PendingSeedJobWatcher } from "@/components/AddSeedsDialog";
import { ResearchTopicDialog } from "@/components/ResearchTopicDialog";
import { GraphModal } from "@/components/GraphModal";
import { PaperDetailModal } from "@/components/PaperDetailModal";
import { Icon } from "@/components/rs/icon";
import { Badge, Btn, EmptyState } from "@/components/rs/primitives";
import { useAddSeeds, usePapers, usePendingSeeds, useTechniques } from "@/hooks/useKG";
import { ApiError, healStubTitles, indexFullText, ingestGlobalKgSeeds } from "@/lib/api";
import type { JobSummary } from "@/lib/types";

export function KnowledgeGraphTab({ slug }: { slug: string }) {
  const [search, setSearch] = useState("");
  const [selectedArxiv, setSelectedArxiv] = useState<string | null>(null);
  const [graphOpen, setGraphOpen] = useState(false);
  const [lastJobId, setLastJobId] = useState<string | null>(null);

  const filter = useMemo(() => ({ search: search.trim() || undefined }), [search]);
  const papers = usePapers(slug, filter);
  const pending = usePendingSeeds(slug);
  const techniques = useTechniques(slug);

  const onJobSubmitted = (j: JobSummary) => setLastJobId(j.job_id);

  const bulkSeed = useMutation({
    mutationFn: () => ingestGlobalKgSeeds(slug, true),
    onSuccess: (j) => {
      setLastJobId(j.job_id);
      toast.success("Bulk KG seeding started", { description: `Canonical paper library — job ${j.job_id.slice(0, 8)}…` });
    },
    onError: (err) => {
      const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
      toast.error("Could not start bulk ingest", { description: msg });
    },
  });

  const healStubs = useMutation({
    mutationFn: () => healStubTitles(slug),
    onSuccess: (resp) => {
      const { healed, still_stubbed, errored, total } = resp.summary;
      if (total === 0) toast.success("KG is clean", { description: "No stub-titled papers found." });
      else toast.success(`Healed ${healed}/${total} stubs`, { description: `${still_stubbed} still stubbed; ${errored} errored` });
      papers.refetch();
    },
    onError: (err) => {
      const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
      toast.error("Heal-stubs failed", { description: msg });
    },
  });

  const indexBodies = useMutation({
    mutationFn: () => indexFullText(slug),
    onSuccess: ({ indexed, missing, skipped }) => {
      if (skipped > 0) {
        toast.warning("Full-text search unavailable", {
          description: "This SQLite build lacks FTS5 — search stays abstract-only.",
        });
      } else {
        toast.success(`Indexed ${indexed} paper ${indexed === 1 ? "body" : "bodies"}`, {
          description: missing > 0
            ? `${missing} had no stored body (ingested without a PDF).`
            : "Paper search now reaches body text, not just abstracts.",
        });
      }
    },
    onError: (err) => {
      const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
      toast.error("Full-text indexing failed", { description: msg });
    },
  });

  const paperList = papers.data ?? [];
  const pendingList = pending.data?.pending ?? [];
  const techList = techniques.data ?? [];

  // Ingest the seeds already sitting in kg_seeds.yml (queued by a mission
  // or a prior partial run) but not yet in the graph — same job pattern
  // AddSeedsDialog uses (POST /kg/seeds → job handle → onJobSubmitted
  // hands it to PendingSeedJobWatcher for progress/completion toast), so
  // there's no second ingest code path to keep in sync.
  const ingestPending = useAddSeeds(slug);
  const ingestPendingSeeds = () => {
    ingestPending.mutate(
      { seeds: pendingList.map((s) => ({ arxiv_id: s.arxiv_id })), auto_extract: true },
      {
        onSuccess: (j) => {
          onJobSubmitted(j);
          toast.message("Ingest queued", {
            description: `${pendingList.length} pending seed(s) — you can close this and come back.`,
          });
        },
        onError: (err) => {
          if (err instanceof ApiError && err.status === 409) {
            toast.error("KG job already running", {
              description: "Wait for the current ingest/extract to finish before retrying.",
            });
            return;
          }
          toast.error("Could not queue pending seeds", { description: err.message });
        },
      },
    );
  };

  return (
    <div className="rs-scroll">
      <div className="rs-pad rs-vgap-16">
        <div className="rs-flex-between rs-wrap rs-gap-12">
          <div>
            <div className="rs-eyebrow">Shared knowledge graph</div>
            <h2 className="rs-h2" style={{ marginTop: 6 }}>Knowledge Graph</h2>
          </div>
          <div className="rs-flex rs-gap-8 rs-wrap">
            <ActiveJobsIndicator slug={slug} />
            <Btn kind="ghost" size="sm" icon="network" onClick={() => setGraphOpen(true)}>View graph</Btn>
            <Btn
              kind="ghost"
              size="sm"
              icon={bulkSeed.isPending ? "loader" : "sparkles"}
              disabled={bulkSeed.isPending}
              onClick={() => bulkSeed.mutate()}
              title="Ingest ~50 curated papers (research log + library refs + foundational RL)"
            >
              Bulk-seed
            </Btn>
            <Btn
              kind="ghost"
              size="sm"
              icon={healStubs.isPending ? "loader" : "refresh-cw"}
              disabled={healStubs.isPending}
              onClick={() => healStubs.mutate()}
              title="Re-ingest papers whose title is still 'arxiv:XXXX' (rate-limit stubs)"
            >
              Heal stubs
            </Btn>
            <Btn
              kind="ghost"
              size="sm"
              icon={indexBodies.isPending ? "loader" : "search"}
              disabled={indexBodies.isPending}
              onClick={() => indexBodies.mutate()}
              title="Index paper bodies so search reaches full text, not just abstracts (local, no network)"
            >
              Index bodies
            </Btn>
            <AddSeedsDialog slug={slug} onJobSubmitted={onJobSubmitted} />
            <ResearchTopicDialog slug={slug} onJobSubmitted={onJobSubmitted} />
          </div>
        </div>

        {paperList.length === 0 && pendingList.length === 0 && !papers.isLoading && (
          <div className="rs-banner info">
            <Icon name="network" size={17} />
            <span className="rs-grow">
              Knowledge graph is empty. mjlab-ready projects auto-seed their robot's canonical papers;
              add more by arxiv ID with "Add papers", or describe a topic with "Research".
            </span>
          </div>
        )}
        {pendingList.length > 0 && (
          <div className="rs-banner warn">
            <Icon name="clock" size={17} />
            <span className="rs-grow">
              {pendingList.length} seed paper{pendingList.length === 1 ? "" : "s"} from kg_seeds.yml {pendingList.length === 1 ? "is" : "are"} not in the knowledge graph yet.
            </span>
            <Btn
              kind="ghost"
              size="sm"
              icon={ingestPending.isPending ? "loader" : "download"}
              disabled={ingestPending.isPending}
              onClick={ingestPendingSeeds}
              title="Ingest the pending kg_seeds.yml papers now"
            >
              {ingestPending.isPending ? "Queueing…" : "Ingest now"}
            </Btn>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.3fr) minmax(300px,1fr)", gap: 20, alignItems: "start" }}>
          <div className="rs-vgap-16">
            {/* Papers — title is the primary line; authors/year/arxiv-id fold
                into one muted secondary line, and extracted-status reads as a
                quiet badge instead of a bare icon. */}
            <div className="rs-card">
              <div className="rs-card-head">
                <div className="rs-card-title">
                  <Icon name="library" size={16} />Papers
                  {paperList.length > 0 && (
                    <span className="rs-num" style={{ color: "var(--rs-muted)", fontSize: 12, fontWeight: 400 }}>{paperList.length}</span>
                  )}
                </div>
                {/* search styled to rs-input tokens (hairline-strong border,
                    radius-md, 32px control height) */}
                <div className="rs-flex rs-gap-8" style={{ background: "var(--canvas-soft)", border: "1px solid var(--hairline-strong)", borderRadius: "var(--radius-md)", padding: "0 11px", height: 32, width: 230, flexShrink: 0 }}>
                  <Icon name="search" size={14} color="var(--rs-muted)" />
                  <input
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search title, author, arxiv…"
                    aria-label="Search papers"
                    style={{ border: 0, background: "none", outline: "none", fontSize: 13, width: "100%", color: "var(--ink)" }}
                  />
                </div>
              </div>
              {papers.isLoading ? (
                <div style={{ padding: 16 }}><p className="rs-sub">Loading…</p></div>
              ) : paperList.length === 0 ? (
                <div style={{ padding: 8 }}>
                  <EmptyState icon="search" title="No papers" sub={search ? `Nothing matches "${search}".` : "The KG has no papers yet."} />
                </div>
              ) : (
                paperList.map((p) => (
                  <button
                    key={p.id}
                    className="rs-paper"
                    onClick={() => setSelectedArxiv(p.arxiv_id)}
                    style={{ width: "100%", textAlign: "left", alignItems: "flex-start", background: selectedArxiv === p.arxiv_id ? "var(--surface-strong)" : "none", border: 0, borderBottom: "1px solid var(--hairline-soft)", cursor: "pointer" }}
                  >
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <p className="rs-paper-t">{p.title}</p>
                      <span className="rs-paper-a">
                        {p.authors.slice(0, 3).join(", ")}{p.authors.length > 3 ? "…" : ""}
                        {p.year ? ` · ${p.year}` : ""}
                        {" · "}
                        <span className="mono" style={{ color: "var(--st-blue)" }}>arXiv:{p.arxiv_id}</span>
                      </span>
                    </div>
                    <span
                      className="rs-tag"
                      style={{ flexShrink: 0, fontSize: 10.5, ...(p.extracted
                        ? { background: "var(--st-emerald-bg)", color: "var(--st-emerald-fg)" }
                        : { color: "var(--rs-muted)" }) }}
                      title={p.extracted
                        ? "Techniques + failure modes extracted into the graph"
                        : "Ingested — extraction hasn't run yet"}
                    >
                      {p.extracted ? "extracted" : "pending"}
                    </span>
                  </button>
                ))
              )}
            </div>

            {/* Techniques — the shared KG can hold hundreds; show the most-used
                in a bounded, scrollable band rather than rendering them all. */}
            {techList.length > 0 && (() => {
              const TECH_CAP = 60;
              const sorted = [...techList].sort((a, b) => b.n_addresses - a.n_addresses);
              const shown = sorted.slice(0, TECH_CAP);
              return (
                <div className="rs-card">
                  <div className="rs-card-head">
                    <div className="rs-card-title"><Icon name="layers" size={16} />Techniques</div>
                    <span className="rs-num" style={{ color: "var(--rs-muted)", fontSize: 12 }}>
                      {shown.length < techList.length ? `top ${shown.length} of ${techList.length}` : techList.length}
                    </span>
                  </div>
                  <div style={{ padding: "16px 18px", display: "flex", gap: 9, flexWrap: "wrap", maxHeight: 240, overflowY: "auto" }}>
                    {shown.map((t) => (
                      <span key={t.id} className="rs-tag" title={t.description}>
                        {t.name}
                        {t.n_addresses > 0 && <b className="mono" style={{ opacity: 0.7 }}>×{t.n_addresses}</b>}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Pending seeds */}
            {pendingList.length > 0 && (
              <div className="rs-card">
                <div className="rs-card-head"><div className="rs-card-title"><Icon name="package" size={16} />Pending seeds</div><span className="rs-num" style={{ color: "var(--rs-muted)", fontSize: 12 }}>{pendingList.length} queued</span></div>
                <div className="rs-card-pad">
                  {pendingList.map((s) => (
                    <div className="rs-seed" key={s.arxiv_id}>
                      <Icon name="sparkles" size={15} color="var(--rs-primary)" />
                      <span style={{ flex: 1, fontSize: 13, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.title || s.arxiv_id}</span>
                      <span className="rs-paper-id" style={{ width: "auto" }}>{s.arxiv_id}</span>
                      <Badge status="queued" label="Pending" />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Graph */}
          <div className="rs-card">
            <div className="rs-card-head"><div className="rs-card-title"><Icon name="network" size={16} />Knowledge graph</div></div>
            <div style={{ padding: 14 }}>
              <button
                className="rs-graphviz"
                onClick={() => setGraphOpen(true)}
                style={{ width: "100%", border: 0, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
                aria-label="Open interactive knowledge graph"
              >
                <EmptyState
                  icon="network"
                  title="Interactive graph"
                  sub="Explore papers, techniques, failures, and the exact policy/reference/world/run artifact lineage. Hover edges for structured receipts."
                />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* modals */}
      <PaperDetailModal slug={slug} arxivId={selectedArxiv} open={!!selectedArxiv} onOpenChange={(o) => !o && setSelectedArxiv(null)} />
      <GraphModal slug={slug} open={graphOpen} onOpenChange={setGraphOpen} />
      {lastJobId && <PendingSeedJobWatcher slug={slug} jobId={lastJobId} onDone={() => setLastJobId(null)} />}
    </div>
  );
}

export default KnowledgeGraphTab;

import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, Circle, Loader2, Network, Search, Sparkles, Wand2 } from "lucide-react";
import { toast } from "sonner";

import { ActiveJobsIndicator } from "@/components/ActiveJobsIndicator";
import {
  AddSeedsDialog,
  PendingSeedJobWatcher,
} from "@/components/AddSeedsDialog";
import { ResearchTopicDialog } from "@/components/ResearchTopicDialog";
import { ApiError, healStubTitles, ingestGlobalKgSeeds } from "@/lib/api";
import { GraphModal } from "@/components/GraphModal";
import { PaperDetailModal } from "@/components/PaperDetailModal";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { usePapers, usePendingSeeds } from "@/hooks/useKG";
import { cn } from "@/lib/utils";
import type { JobSummary, PaperSummary } from "@/lib/types";

type ExtractedFilter = "all" | "yes" | "no";

export function KnowledgeGraphTab({ slug }: { slug: string }) {
  const [extracted, setExtracted] = useState<ExtractedFilter>("all");
  const [search, setSearch] = useState("");
  const [selectedArxiv, setSelectedArxiv] = useState<string | null>(null);
  const [graphOpen, setGraphOpen] = useState(false);
  const [lastJobId, setLastJobId] = useState<string | null>(null);

  const filter = useMemo(
    () => ({
      extracted:
        extracted === "all" ? undefined : extracted === "yes" ? true : false,
      search: search.trim() || undefined,
    }),
    [extracted, search],
  );
  const papers = usePapers(slug, filter);
  const pending = usePendingSeeds(slug);

  const onJobSubmitted = (j: JobSummary) => setLastJobId(j.job_id);

  const bulkSeed = useMutation({
    mutationFn: () => ingestGlobalKgSeeds(slug, true),
    onSuccess: (j) => {
      setLastJobId(j.job_id);
      toast.success("Bulk KG seeding started", {
        description: `50 canonical papers — job ${j.job_id.slice(0, 8)}…`,
      });
    },
    onError: (err) => {
      const msg =
        err instanceof ApiError
          ? err.problem.detail ?? err.problem.title
          : (err as Error).message;
      toast.error("Could not start bulk ingest", { description: msg });
    },
  });

  // §Ship-7: re-ingest Paper nodes whose title is still
  // `arxiv:XXXX.XXXXX` (stubs left by arxiv rate-limit during bulk
  // ingest). Synchronous — typically ~2 min for ~10 stubs.
  const healStubs = useMutation({
    mutationFn: () => healStubTitles(slug),
    onSuccess: (resp) => {
      const { healed, still_stubbed, errored, total } = resp.summary;
      if (total === 0) {
        toast.success("KG is clean", {
          description: "No stub-titled papers found.",
        });
      } else {
        toast.success(`Healed ${healed}/${total} stubs`, {
          description:
            `${still_stubbed} still stubbed${
              still_stubbed ? " (retry after ~2 min)" : ""
            }; ${errored} errored`,
        });
      }
      papers.refetch();
    },
    onError: (err) => {
      const msg =
        err instanceof ApiError
          ? err.problem.detail ?? err.problem.title
          : (err as Error).message;
      toast.error("Heal-stubs failed", { description: msg });
    },
  });

  const ingestedCount = papers.data?.length ?? 0;
  const pendingCount = pending.data?.pending.length ?? 0;

  return (
    <div className="flex flex-col gap-4">
      {ingestedCount === 0 && pendingCount === 0 && (
        <div className="flex items-start gap-2 rounded-md border border-dashed bg-muted/20 p-3 text-xs">
          <Network className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <div>
            <div className="font-medium">Knowledge graph is empty</div>
            <p className="text-muted-foreground">
              Projects created from <strong>mjlab-ready</strong> library
              robots auto-seed with the robot's canonical papers. Legacy
              Gymnasium projects (Hopper / Ant / …) start empty — add
              papers by arxiv ID via "Add seeds" below.
            </p>
          </div>
        </div>
      )}
      {ingestedCount === 0 && pendingCount > 0 && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
          <Circle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
          <div>
            <div className="font-medium text-amber-700 dark:text-amber-300">
              {pendingCount} seed{pendingCount === 1 ? "" : "s"} pending —
              no papers ingested yet
            </div>
            <p className="text-muted-foreground">
              Either the ingest job is still running (check the Dashboard's
              active jobs card) or it failed. Use "Add seeds" with
              "auto extract" checked to re-trigger, or watch the active
              jobs list for an error.
            </p>
          </div>
        </div>
      )}
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 py-3">
          <div>
            <CardTitle className="text-sm">Knowledge Graph</CardTitle>
            <CardDescription className="text-[11px]">
              Papers, techniques, failure modes, reward components —
              citations for every sculptor edit land here.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <ActiveJobsIndicator slug={slug} />
            <Button
              variant="outline"
              size="sm"
              onClick={() => setGraphOpen(true)}
            >
              <Network />
              View graph
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={bulkSeed.isPending}
              onClick={() => bulkSeed.mutate()}
              title="Ingest ~50 curated papers: AME456 research log + mjlab library refs + foundational RL"
            >
              {bulkSeed.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              Bulk-seed library (50 papers)
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={healStubs.isPending}
              onClick={() => healStubs.mutate()}
              title="Re-ingest papers whose title is still 'arxiv:XXXX.XXXXX' (left behind when arxiv API rate-limited during bulk ingest). §7.7 — takes ~2 min for ~10 stubs."
            >
              {healStubs.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wand2 className="h-4 w-4" />
              )}
              Heal stub titles
            </Button>
            <AddSeedsDialog slug={slug} onJobSubmitted={onJobSubmitted} />
            <ResearchTopicDialog slug={slug} onJobSubmitted={onJobSubmitted} />
          </div>
        </CardHeader>
      </Card>

      {pending.data && pending.data.pending.length > 0 && (
        <PendingSeedsCard count={pending.data.pending.length} />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <PapersList
          papers={papers.data ?? []}
          loading={papers.isLoading}
          search={search}
          onSearch={setSearch}
          extracted={extracted}
          onExtracted={setExtracted}
          selected={selectedArxiv}
          onSelect={setSelectedArxiv}
        />
        <PaperPreviewPane
          slug={slug}
          selectedArxiv={selectedArxiv}
          onOpen={() => selectedArxiv && setSelectedArxiv(selectedArxiv)}
        />
      </div>

      {/* modals */}
      <PaperDetailModal
        slug={slug}
        arxivId={selectedArxiv}
        open={!!selectedArxiv}
        onOpenChange={(o) => !o && setSelectedArxiv(null)}
      />
      <GraphModal
        slug={slug}
        open={graphOpen}
        onOpenChange={setGraphOpen}
      />

      {lastJobId && (
        <PendingSeedJobWatcher
          slug={slug}
          jobId={lastJobId}
          onDone={() => setLastJobId(null)}
        />
      )}
    </div>
  );
}

// ── Pending seeds ────────────────────────────────────────────────────
function PendingSeedsCard({ count }: { count: number }) {
  return (
    <div className="rounded-md border border-dashed bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
      <span className="font-medium text-foreground">{count}</span> seed
      {count === 1 ? "" : "s"} in <code>kg_seeds.yml</code> not yet
      ingested. Use "Add papers" to kick off the ingest job.
    </div>
  );
}

// ── Papers list ──────────────────────────────────────────────────────
function PapersList({
  papers,
  loading,
  search,
  onSearch,
  extracted,
  onExtracted,
  selected,
  onSelect,
}: {
  papers: PaperSummary[];
  loading: boolean;
  search: string;
  onSearch: (s: string) => void;
  extracted: ExtractedFilter;
  onExtracted: (e: ExtractedFilter) => void;
  selected: string | null;
  onSelect: (arxivId: string) => void;
}) {
  return (
    <Card className="flex min-h-[420px] flex-col">
      <CardHeader className="space-y-2 border-b py-3">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => onSearch(e.target.value)}
              placeholder="Search title, author, or arxiv_id…"
              className="h-8 pl-7 text-xs"
            />
          </div>
          <Select
            value={extracted}
            onChange={(e) => onExtracted(e.target.value as ExtractedFilter)}
            className="h-8 w-32 text-xs"
            aria-label="Extracted filter"
          >
            <option value="all">all</option>
            <option value="yes">extracted</option>
            <option value="no">not extracted</option>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-y-auto scrollbar-thin p-0">
        {loading && (
          <p className="px-3 py-4 text-xs text-muted-foreground">Loading…</p>
        )}
        {!loading && papers.length === 0 && (
          <p className="px-3 py-4 text-xs text-muted-foreground">
            No papers in the KG yet.
          </p>
        )}
        <ul className="divide-y">
          {papers.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                onClick={() => onSelect(p.arxiv_id)}
                className={cn(
                  "flex w-full flex-col gap-1 px-3 py-2 text-left text-xs transition-colors",
                  p.arxiv_id === selected
                    ? "bg-accent"
                    : "hover:bg-accent/60",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-medium">{p.title}</span>
                  {p.extracted ? (
                    <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                  ) : (
                    <Circle className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  )}
                </div>
                <div className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                  <span>{p.arxiv_id}</span>
                  {p.year && <span>· {p.year}</span>}
                  {p.authors.length > 0 && (
                    <span className="truncate">
                      · {p.authors.slice(0, 2).join(", ")}
                      {p.authors.length > 2 ? "…" : ""}
                    </span>
                  )}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

// ── Preview pane (side-by-side hint; real detail is in the modal) ────
function PaperPreviewPane({
  selectedArxiv,
}: {
  slug: string;
  selectedArxiv: string | null;
  onOpen: () => void;
}) {
  if (!selectedArxiv) {
    return (
      <Card className="flex min-h-[420px] items-center justify-center text-xs text-muted-foreground">
        Select a paper to see its entities.
      </Card>
    );
  }
  return (
    <Card className="flex min-h-[420px] flex-col items-center justify-center gap-2 text-xs text-muted-foreground">
      <p>Opening paper <code className="font-mono">{selectedArxiv}</code>…</p>
    </Card>
  );
}

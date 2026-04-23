import { ExternalLink, Loader2 } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-3xl overflow-y-auto scrollbar-thin">
        <DialogHeader>
          <DialogTitle>
            {data?.title ?? (arxivId ? `arxiv:${arxivId}` : "Paper")}
          </DialogTitle>
          {data && (
            <DialogDescription className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="font-mono">{data.arxiv_id}</span>
              {data.year && <span>· {data.year}</span>}
              {data.authors.length > 0 && (
                <span>· {data.authors.join(", ")}</span>
              )}
              <a
                href={`https://arxiv.org/abs/${data.arxiv_id}`}
                target="_blank"
                rel="noreferrer noopener"
                className="ml-auto inline-flex items-center gap-1 text-foreground underline-offset-2 hover:underline"
              >
                arxiv.org <ExternalLink className="h-3 w-3" />
              </a>
            </DialogDescription>
          )}
        </DialogHeader>

        {isLoading && (
          <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading paper…
          </div>
        )}
        {error && (
          <p className="text-sm text-rose-700">
            {(error as Error).message}
          </p>
        )}
        {data && (
          <div className="space-y-4">
            {data.abstract && (
              <section>
                <h3 className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                  Abstract
                </h3>
                <p className="text-sm leading-relaxed text-foreground">
                  {data.abstract}
                </p>
              </section>
            )}
            <EntityGroup title="Techniques" items={data.entities.techniques} />
            <EntityGroup title="Failure modes" items={data.entities.failure_modes} />
            <EntityGroup
              title="Reward components"
              items={data.entities.reward_components}
            />
            <EntityGroup title="Environments" items={data.entities.environments} />
            {!data.extracted && (
              <p className="rounded border border-dashed px-3 py-2 text-xs text-muted-foreground">
                Entities have not been extracted yet — run "Add papers" with
                <code className="mx-1">auto_extract=true</code> and a valid
                ANTHROPIC_API_KEY.
              </p>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
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
      <h3 className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {title} · {items.length}
      </h3>
      <ul className="space-y-1">
        {items.map((e) => (
          <li
            key={e.id}
            className="rounded border bg-muted/20 px-2 py-1.5 text-xs"
          >
            <div className="font-medium">{e.name}</div>
            {e.description && (
              <div className="text-[11px] text-muted-foreground">
                {e.description}
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

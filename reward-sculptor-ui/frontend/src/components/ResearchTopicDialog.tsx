import { useState } from "react";
import { Loader2, Search as SearchIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useResearchTopic } from "@/hooks/useKG";
import { ApiError } from "@/lib/api";
import type { JobSummary } from "@/lib/types";

/** "Research a topic" button + modal. Click → describe the topic in
 * plain English → Claude picks 5-10 arxiv papers, the backend
 * dedupes + ingests + extracts them into the shared KG. M7 Phase 2.
 *
 * Mirrors AddSeedsDialog's pattern (R3: fire job + close dialog +
 * hand job_id up to the KG tab's ActiveJobsIndicator). */
export function ResearchTopicDialog({
  slug,
  onJobSubmitted,
}: {
  slug: string;
  onJobSubmitted: (job: JobSummary) => void;
}) {
  const [open, setOpen] = useState(false);
  const [topic, setTopic] = useState("");
  const [maxPapers, setMaxPapers] = useState(10);
  const research = useResearchTopic(slug);

  const submit = () => {
    const trimmed = topic.trim();
    if (trimmed.length < 3) {
      toast.error("Topic too short", {
        description: "Describe what you want Claude to research (≥ 3 chars).",
      });
      return;
    }
    research.mutate(
      { topic: trimmed, max_papers: maxPapers, auto_extract: true },
      {
        onSuccess: (job) => {
          setOpen(false);
          setTopic("");
          onJobSubmitted(job);
          toast.success("Researching topic…", {
            description: `Claude → arxiv → extract (≤ ${maxPapers} papers, ~2-5 min)`,
          });
        },
        onError: (err) => {
          const msg =
            err instanceof ApiError
              ? err.problem.detail ?? err.problem.title
              : (err as Error).message;
          toast.error("Could not start research", { description: msg });
        },
      },
    );
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !research.isPending && setOpen(o)}
    >
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          title="Ask Claude to find arxiv papers on a topic and add them to the KG"
        >
          <SearchIcon className="h-4 w-4" />
          Research a topic
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Research a topic</DialogTitle>
          <DialogDescription>
            Claude Opus 4.7 picks arxiv IDs directly relevant to your
            topic, dedupes against the shared KG, then ingests + extracts
            entities for the new ones. Writes to the user-wide KG so
            other projects benefit too.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1.5">
            <Label htmlFor="research-topic" className="text-xs">
              Topic
            </Label>
            <Textarea
              id="research-topic"
              placeholder={
                "Describe what you want Claude to find. Examples:\n" +
                "  • SEA physics parameters for quadruped robots\n" +
                "  • Sparse-reward exploration in continuous control\n" +
                "  • Contact-rich manipulation with tactile sensing"
              }
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              rows={5}
              maxLength={500}
              disabled={research.isPending}
            />
            <p className="text-[10px] text-muted-foreground">
              {topic.length}/500 characters
            </p>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="research-max" className="text-xs">
              Max papers: <span className="font-mono">{maxPapers}</span>
            </Label>
            <input
              id="research-max"
              type="range"
              min={1}
              max={20}
              step={1}
              value={maxPapers}
              onChange={(e) => setMaxPapers(Number(e.target.value))}
              disabled={research.isPending}
              className="w-full accent-primary"
            />
            <p className="text-[10px] text-muted-foreground">
              Soft cap — Claude may return fewer if coverage is thin.
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setOpen(false)}
            disabled={research.isPending}
          >
            Cancel
          </Button>
          <Button onClick={submit} disabled={research.isPending}>
            {research.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <SearchIcon className="h-4 w-4" />
            )}
            {research.isPending ? "Starting…" : "Research"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

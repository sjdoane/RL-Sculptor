import { useState } from "react";
import { toast } from "sonner";

import { Btn, Field, Modal } from "@/components/rs/primitives";
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
    <>
      <Btn
        kind="ghost"
        size="sm"
        icon="search"
        onClick={() => setOpen(true)}
        title="Ask Claude to find arxiv papers on a topic and add them to the KG"
      >
        Research a topic
      </Btn>
      {open && (
        <Modal
          icon="search"
          title="Research a topic"
          subtitle="Claude picks arxiv IDs relevant to your topic, dedupes against the shared KG, then ingests + extracts the new ones. Writes to the user-wide KG so other projects benefit too."
          onClose={() => { if (!research.isPending) setOpen(false); }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => setOpen(false)} disabled={research.isPending}>Cancel</Btn>
              <Btn kind="primary" icon={research.isPending ? "loader" : "search"} onClick={submit} disabled={research.isPending}>
                {research.isPending ? "Starting…" : "Research"}
              </Btn>
            </>
          }
        >
          <Field label="Topic" hint={`${topic.length}/500`} htmlFor="research-topic">
            <textarea
              id="research-topic"
              className="rs-textarea"
              placeholder={
                "Describe what you want Claude to find. Examples:\n" +
                "  • SEA physics parameters for quadruped robots\n" +
                "  • Sparse-reward exploration in continuous control\n" +
                "  • Contact-rich manipulation with tactile sensing"
              }
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              style={{ minHeight: 116 }}
              maxLength={500}
              disabled={research.isPending}
              autoFocus
            />
          </Field>

          <Field label={<>Max papers: <span className="mono" style={{ color: "var(--ink)" }}>{maxPapers}</span></>} htmlFor="research-max">
            <input
              id="research-max"
              type="range"
              min={1}
              max={20}
              step={1}
              value={maxPapers}
              onChange={(e) => setMaxPapers(Number(e.target.value))}
              disabled={research.isPending}
              style={{ width: "100%", accentColor: "var(--rs-primary)" }}
            />
            <p className="rs-hintline">Soft cap — Claude may return fewer if coverage is thin.</p>
          </Field>
        </Modal>
      )}
    </>
  );
}

import { useState } from "react";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, Field, Modal } from "@/components/rs/primitives";
import { useAddSeeds } from "@/hooks/useKG";
import { useJob } from "@/hooks/useJob";
import { ApiError } from "@/lib/api";
import type { JobSummary } from "@/lib/types";

const ARXIV_RE = /^\d{4}\.\d{4,5}(v\d+)?$/;

/** Click "Add papers" → dialog → on submit we POST seeds, close
 *  immediately (precondition R3), and hand the job_id up to the KG tab
 *  which surfaces it via ActiveJobsIndicator + useJob polling for the
 *  completion toast. */
export function AddSeedsDialog({
  slug,
  onJobSubmitted,
}: {
  slug: string;
  onJobSubmitted: (job: JobSummary) => void;
}) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [localErrors, setLocalErrors] = useState<string[]>([]);
  const add = useAddSeeds(slug);

  const submit = () => {
    const lines = text
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      setLocalErrors(["Paste at least one arxiv_id."]);
      return;
    }
    const bad = lines.filter((l) => !ARXIV_RE.test(l));
    if (bad.length > 0) {
      setLocalErrors([
        `Not a recognized arxiv_id: ${bad.slice(0, 3).join(", ")}${
          bad.length > 3 ? "…" : ""
        }`,
        "Expected the modern form like 2403.12345 or 1707.06347v2.",
      ]);
      return;
    }
    setLocalErrors([]);
    add.mutate(
      {
        seeds: lines.map((id) => ({ arxiv_id: id })),
        auto_extract: true,
      },
      {
        onSuccess: (job) => {
          // R3: close the dialog as soon as the job is queued.
          setOpen(false);
          setText("");
          onJobSubmitted(job);
          toast.message("Ingest queued", {
            description: `${lines.length} paper(s) — you can close this and come back.`,
          });
        },
        onError: (err) => {
          if (err instanceof ApiError && err.status === 409) {
            toast.error("KG job already running", {
              description:
                "Wait for the current ingest/extract to finish before adding more papers.",
            });
            setOpen(false);
            return;
          }
          setLocalErrors([err.message]);
        },
      },
    );
  };

  return (
    <>
      <Btn kind="primary" size="sm" icon="plus" onClick={() => setOpen(true)}>Add papers</Btn>
      {open && (
        <Modal
          icon="book"
          title="Add papers to the project KG"
          subtitle="Paste one arxiv_id per line. We'll ingest each paper (metadata + PDF), then run entity extraction via Claude."
          onClose={() => { if (!add.isPending) setOpen(false); }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => setOpen(false)} disabled={add.isPending}>Cancel</Btn>
              <Btn kind="primary" icon={add.isPending ? "loader" : "plus"} onClick={submit} disabled={add.isPending}>
                {add.isPending ? "Queueing…" : "Queue ingest"}
              </Btn>
            </>
          }
        >
          <Field label="arxiv_ids" htmlFor="arxiv-ids">
            <textarea
              id="arxiv-ids"
              className="rs-textarea mono"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={"1707.06347\n2010.04159"}
              style={{ minHeight: 120 }}
              disabled={add.isPending}
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
            />
          </Field>
          {localErrors.length > 0 && (
            <ul style={{ display: "flex", flexDirection: "column", gap: 3, margin: 0, padding: "8px 10px", listStyle: "none", borderRadius: "var(--radius-md)", border: "1px solid var(--st-rose-bg)", background: "var(--st-rose-bg)", color: "var(--st-rose-fg)" }}>
              {localErrors.map((e, i) => (
                <li key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6, fontSize: 12 }}>
                  <span style={{ marginTop: 1, flexShrink: 0 }}><Icon name="alert-circle" size={12} /></span>
                  <span className="mono" style={{ wordBreak: "break-word" }}>{e}</span>
                </li>
              ))}
            </ul>
          )}
        </Modal>
      )}
    </>
  );
}

/** Invisible companion that polls the last-submitted job every 3 s and
 *  toasts on completion. Must be mounted somewhere in the tree so the
 *  polling continues if the dialog closes. */
export function PendingSeedJobWatcher({
  slug,
  jobId,
  onDone,
}: {
  slug: string;
  jobId: string;
  onDone: () => void;
}) {
  useJob(jobId, {
    onTerminal: (job) => {
      onDone();
      if (job.status === "completed") {
        const ingest = job.result?.ingest as
          | { ingested?: number; already_present?: number }
          | { ingested?: string[]; errors?: Record<string, string> }
          | null
          | undefined;
        const extract = job.result?.extract as
          | { succeeded?: number; skipped?: boolean }
          | null
          | undefined;
        const noExtract = extract && (extract as { skipped?: boolean }).skipped;
        // `ingested` can be a number (bulk-seed) or a string[] of
        // newly-added arxiv_ids (research job). Normalise.
        const ingestedVal = ingest?.ingested as number | string[] | undefined;
        const n = Array.isArray(ingestedVal)
          ? ingestedVal.length
          : (ingestedVal ?? 0);
        const m = (extract?.succeeded ?? 0);
        // kg_research jobs that find no new papers (because Claude's
        // top matches are already in the KG, or Claude returned 0
        // matches) used to show the same bare "Added 0 paper(s)"
        // toast as a failing seed. The research runner now puts
        // `papers_returned_by_claude` + `papers_deduped_against_kg`
        // on `job.result` so the toast can explain WHICH stage ate
        // the papers. Surfaces Test 1 Issue D from 2026-04-22.
        const researchClaudeReturned = job.result?.papers_returned_by_claude as
          | number
          | undefined;
        const researchDeduped = job.result?.papers_deduped_against_kg as
          | number
          | undefined;
        const topic = job.result?.topic as string | undefined;
        const coverageNote = job.result?.coverage_note as string | undefined;
        const isResearchJob = job.kind === "kg_research";

        if (isResearchJob && n === 0 && researchClaudeReturned !== undefined) {
          if (researchClaudeReturned === 0) {
            toast.message(
              `Research: Claude found 0 papers for ${topic ?? "that topic"}`,
              {
                description: coverageNote
                  ? `Coverage note: ${coverageNote.slice(0, 200)}${coverageNote.length > 200 ? "…" : ""}`
                  : "Claude had no strong matches. Try a more specific query.",
                id: `seed-job-${slug}`,
              },
            );
          } else if ((researchDeduped ?? 0) > 0 && researchDeduped === researchClaudeReturned) {
            toast.success(
              `Research: Claude returned ${researchClaudeReturned} paper(s), all already in KG`,
              {
                description: coverageNote
                  ? coverageNote.slice(0, 200)
                  : "0 new papers to ingest — your KG already covers this topic.",
                id: `seed-job-${slug}`,
              },
            );
          } else {
            toast.message(
              `Research: Claude returned ${researchClaudeReturned}, ${researchDeduped ?? 0} already in KG, 0 ingested`,
              { id: `seed-job-${slug}` },
            );
          }
        } else {
          toast.success(
            noExtract
              ? `Added ${n} paper(s) — extraction skipped (no API key)`
              : `Added ${n} paper(s), extracted ${m}`,
            { id: `seed-job-${slug}` },
          );
        }
      } else if (job.status === "errored") {
        toast.error("Ingest job failed", {
          description: job.error ?? "unknown error",
          id: `seed-job-${slug}`,
        });
      } else {
        toast.message("Ingest job stopped", {
          description: "You stopped the job before it finished.",
          id: `seed-job-${slug}`,
        });
      }
    },
  });
  return null;
}

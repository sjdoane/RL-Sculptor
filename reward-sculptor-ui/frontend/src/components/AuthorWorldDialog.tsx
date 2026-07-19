/** Environment-authoring dialog (env-authoring item 5).
 *
 *  Two phases in one modal:
 *   1. prompt — describe the world/task in natural language, author a
 *      draft (KG-grounded, offline-deterministic on the backend);
 *   2. clarify — the draft's paginated clarification questions, each with
 *      its explicit choices AND the disclosed "System decides" default
 *      (the clarifier contract: every load-bearing ambiguity can be
 *      handed back). Unanswered = system default. Apply runs the full
 *      admission gate chain and atomically promotes the tuple.
 */
import { useState } from "react";
import { toast } from "sonner";

import { Btn, Field, Modal } from "@/components/rs/primitives";
import { useApplyWorldAuthor, useAuthorWorld } from "@/hooks/useWorlds";
import { ApiError } from "@/lib/api";
import type { WorldAuthorResponse } from "@/lib/types";

function errText(err: unknown): string {
  if (err instanceof ApiError) {
    return err.problem.detail ?? err.problem.title;
  }
  return err instanceof Error ? err.message : String(err);
}

export default function AuthorWorldDialog({
  slug, onApplied,
}: { slug: string; onApplied?: () => void }) {
  const [open, setOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [robot, setRobot] = useState("");
  const [draft, setDraft] = useState<WorldAuthorResponse | null>(null);
  const [page, setPage] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const author = useAuthorWorld(slug);
  const apply = useApplyWorldAuthor(slug);
  const busy = author.isPending || apply.isPending;

  const reset = () => {
    setDraft(null);
    setPage(0);
    setAnswers({});
  };

  const close = () => {
    if (busy) return;
    setOpen(false);
    reset();
  };

  const runAuthor = () => {
    author.mutate(
      { prompt, robot_capability_id: robot.trim() || null },
      {
        onSuccess: (d) => {
          setDraft(d);
          setPage(0);
          setAnswers({});
        },
        onError: (err) => toast.error(errText(err)),
      },
    );
  };

  const runApply = () => {
    if (!draft) return;
    apply.mutate(
      {
        session_id: draft.session_id,
        answers: Object.entries(answers)
          .filter(([, choice]) => choice !== "system_default")
          .map(([question_id, choice_id]) => ({ question_id, choice_id })),
      },
      {
        onSuccess: (result) => {
          toast.success(
            `World promoted (selection v${result.selection.selection_version}, ` +
            `${result.admission.ok ? "all gates passed" : "admission issue"})`,
          );
          setOpen(false);
          reset();
          onApplied?.();
        },
        onError: (err) => toast.error(errText(err)),
      },
    );
  };

  const pages = draft?.clarification_plan.pages ?? [];
  const lastPage = page >= pages.length - 1;

  return (
    <>
      <Btn kind="primary" icon="globe" onClick={() => setOpen(true)}>
        Author world
      </Btn>
      {open && (
        <Modal
          title="Author environment"
          subtitle={draft
            ? `Robot ${draft.capability_id} — ${pages.length} clarification page(s); unanswered questions use the disclosed system default`
            : "Describe the world and task; the system drafts a parametric, versioned environment"}
          icon="globe"
          onClose={close}
          footer={draft ? (
            <>
              <Btn onClick={close} disabled={busy}>Cancel</Btn>
              {page > 0 && (
                <Btn onClick={() => setPage(page - 1)} disabled={busy}>
                  Back
                </Btn>
              )}
              {!lastPage && (
                <Btn kind="primary" onClick={() => setPage(page + 1)}
                     disabled={busy}>
                  Next
                </Btn>
              )}
              {lastPage && (
                <Btn kind="primary" icon={apply.isPending ? "loader" : "check"}
                     onClick={runApply} disabled={busy}>
                  {apply.isPending ? "Admitting…" : "Apply & promote"}
                </Btn>
              )}
            </>
          ) : (
            <>
              <Btn onClick={close} disabled={busy}>Cancel</Btn>
              <Btn kind="primary"
                   icon={author.isPending ? "loader" : "sparkles"}
                   onClick={runAuthor}
                   disabled={busy || !prompt.trim()}>
                {author.isPending ? "Authoring…" : "Draft world"}
              </Btn>
            </>
          )}
        >
          {!draft ? (
            <>
              <Field label="Environment prompt"
                     hint="e.g. “stay stable and walk on uneven rough terrain”">
                <textarea
                  className="rs-textarea"
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the terrain, obstacles, objects, and goal…"
                />
              </Field>
              <Field label="Robot capability"
                     hint="optional — auto-selected from the prompt when blank">
                <input
                  className="rs-input"
                  value={robot}
                  onChange={(e) => setRobot(e.target.value)}
                  placeholder="unitree_g1:base"
                />
              </Field>
            </>
          ) : (
            <div>
              <div className="rs-hintline">
                Page {page + 1} / {pages.length} · draft{" "}
                <span className="mono">{draft.draft_hash.slice(0, 12)}</span>
                {draft.kg_grounding.length > 0 &&
                  ` — grounded on ${draft.kg_grounding.length} knowledge-graph node(s)`}
              </div>
              {(pages[page]?.questions ?? []).map((q) => {
                const selected = answers[q.question_id] ?? "system_default";
                return (
                  <Field key={q.question_id} label={q.prompt}
                         hint={<span className="mono">{q.parameter_path}</span>}>
                    <div role="radiogroup" aria-label={q.prompt}>
                      {[...q.choices.map((c) => ({
                        id: c.choice_id, label: c.label,
                      })), {
                        id: "system_default",
                        label: q.system_default.label,
                      }].map((choice) => (
                        <label key={choice.id}
                               style={{ display: "flex", gap: 8,
                                        alignItems: "baseline",
                                        padding: "3px 0", cursor: "pointer" }}>
                          <input
                            type="radio"
                            name={q.question_id}
                            checked={selected === choice.id}
                            onChange={() => setAnswers({
                              ...answers, [q.question_id]: choice.id,
                            })}
                          />
                          <span style={{ fontSize: 13 }}>{choice.label}</span>
                        </label>
                      ))}
                    </div>
                  </Field>
                );
              })}
            </div>
          )}
        </Modal>
      )}
    </>
  );
}

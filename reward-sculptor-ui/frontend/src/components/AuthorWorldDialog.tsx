/** Environment-authoring dialog (env-authoring item 5) — the world
 *  builder.
 *
 *  Two phases in one modal:
 *   1. prompt — describe the world/task in natural language, author a
 *      draft (KG-grounded, offline-deterministic on the backend);
 *   2. build — the draft's paginated clarification questions beside a
 *      live 3D preview of the COMPILED scene. "Preview scene" runs the
 *      full admission gate chain server-side WITHOUT promoting (a dry
 *      run under worlds/<session>/); adjust answers → preview again →
 *      "Apply & promote" commits the atomic tuple through the exact
 *      same gates. Every question still shows its explicit choices AND
 *      the disclosed "System decides" default; unanswered = default.
 */
import { useState } from "react";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, Field, Modal } from "@/components/rs/primitives";
import WorldEntityInspector from "@/components/WorldEntityInspector";
import WorldViewer3D from "@/components/WorldViewer3D";
import {
  useApplyWorldAuthor,
  useAuthorWorld,
  usePreviewWorldDraft,
} from "@/hooks/useWorlds";
import { ApiError } from "@/lib/api";
import type { WorldAuthorResponse, WorldDraftPreview } from "@/lib/types";

function errText(err: unknown): string {
  if (err instanceof ApiError) {
    return err.problem.detail ?? err.problem.title;
  }
  return err instanceof Error ? err.message : String(err);
}

/** Admission-gate outcome strip for a draft dry-run: one chip per gate,
 *  violations spelled out — the same chain "Apply & promote" will run. */
function GateStrip({ preview }: { preview: WorldDraftPreview }) {
  const gates = preview.admission.gates;
  const failed = gates.filter((g) => !g.ok);
  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {gates.map((gate) => (
          <span key={gate.gate} className="rs-tag"
                title={gate.ok ? `${gate.gate}: passed`
                  : gate.violations.map((v) => v.message).join("; ")}
                style={{
                  fontSize: 11,
                  color: gate.ok ? undefined : "var(--st-rose-fg)",
                  borderColor: gate.ok ? undefined : "var(--st-rose-fg)",
                }}>
          {gate.ok ? "✓" : "✕"} {gate.gate}
          </span>
        ))}
      </div>
      {failed.map((gate) => (
        <div key={gate.gate} className="rs-hintline"
             style={{ color: "var(--st-rose-fg)", marginTop: 4 }}>
          {gate.gate}: {gate.violations.map((v) => v.message).join("; ")
            || "failed"}
        </div>
      ))}
    </div>
  );
}

export default function AuthorWorldDialog({
  slug, onApplied,
}: { slug: string; onApplied?: () => void }) {
  const [open, setOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [robot, setRobot] = useState("");
  const [kgGrounding, setKgGrounding] = useState(true);
  const [draft, setDraft] = useState<WorldAuthorResponse | null>(null);
  const [page, setPage] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [preview, setPreview] = useState<WorldDraftPreview | null>(null);
  // Answers changed since the shown preview → the scene is out of date.
  const [previewStale, setPreviewStale] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);

  const author = useAuthorWorld(slug);
  const apply = useApplyWorldAuthor(slug);
  const previewMutation = usePreviewWorldDraft(slug);
  const busy = author.isPending || apply.isPending;

  const reset = () => {
    setDraft(null);
    setPage(0);
    setAnswers({});
    setPreview(null);
    setPreviewStale(false);
    setSelectedEntity(null);
  };

  const answerPayload = (base: Record<string, string>) =>
    Object.entries(base)
      .filter(([, choice]) => choice !== "system_default")
      .map(([question_id, choice_id]) => ({ question_id, choice_id }));

  const runPreview = (draftNow: WorldAuthorResponse,
                      answersNow: Record<string, string>) => {
    previewMutation.mutate(
      { session_id: draftNow.session_id, answers: answerPayload(answersNow) },
      {
        onSuccess: (result) => {
          setPreview(result);
          setPreviewStale(false);
          if (!result.ok) {
            const failed = result.admission.gates
              .filter((g) => !g.ok).map((g) => g.gate);
            toast.error(`Draft failed gates: ${failed.join(", ")}`, {
              description: "Adjust answers and preview again — nothing "
                + "was promoted.",
            });
          }
        },
        onError: (err) => toast.error(errText(err)),
      },
    );
  };

  const close = () => {
    if (busy) return;
    setOpen(false);
    reset();
  };

  const runAuthor = () => {
    author.mutate(
      { prompt, robot_capability_id: robot.trim() || null, kg_grounding: kgGrounding },
      {
        onSuccess: (d) => {
          setDraft(d);
          setPage(0);
          setAnswers({});
          setPreview(null);
          setSelectedEntity(null);
          // Immediately compile the all-defaults draft so the builder
          // opens with a live scene instead of an empty pane.
          runPreview(d, {});
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
        answers: answerPayload(answers),
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
          title={draft ? "World builder" : "Author environment"}
          subtitle={draft
            ? `Robot ${draft.capability_id} — adjust answers, preview the compiled scene, promote when satisfied`
            : "Describe the world and task; the system drafts a parametric, versioned environment"}
          icon="globe"
          full={!!draft}
          onClose={close}
          footer={draft ? (
            <>
              <Btn onClick={close} disabled={busy}>Cancel</Btn>
              <Btn
                icon={previewMutation.isPending ? "loader" : "eye"}
                onClick={() => runPreview(draft, answers)}
                disabled={busy || previewMutation.isPending}
              >
                {previewMutation.isPending
                  ? "Compiling…"
                  : previewStale ? "Preview scene (stale)" : "Preview scene"}
              </Btn>
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
                     hint={"optional — blank uses this project's configured "
                       + "robot (or prompt-based auto-select if the project "
                       + "robot has no capability descriptor)"}>
                <input
                  className="rs-input"
                  value={robot}
                  onChange={(e) => setRobot(e.target.value)}
                  placeholder="project robot"
                />
              </Field>
              {/* The CLI has had `--kg-grounding/--no-kg-grounding` all
                  along; the dialog sent neither, so authoring here was
                  always grounded. Retrieval is best-effort and never blocks,
                  so the reason to turn it off is a noisy or off-topic graph,
                  not speed. */}
              <label
                className="rs-flex rs-gap-8"
                style={{ alignItems: "flex-start", fontSize: 12 }}
              >
                <input
                  type="checkbox"
                  checked={kgGrounding}
                  onChange={(e) => setKgGrounding(e.target.checked)}
                  style={{ marginTop: 2 }}
                />
                <span>
                  Ground in the knowledge graph
                  <span className="rs-sub" style={{ display: "block", fontSize: 11 }}>
                    Cites this project's ingested papers when choosing terrain
                    and task structure. Best-effort — it never blocks authoring.
                  </span>
                </span>
              </label>
            </>
          ) : (
            <div style={{ display: "grid", minHeight: 0, height: "100%",
                          gridTemplateColumns:
                            "minmax(0, 1.5fr) minmax(320px, 1fr)",
                          gap: 16 }}>
              {/* ── left: live compiled-scene preview ─────────────── */}
              <div style={{ display: "flex", flexDirection: "column",
                            gap: 8, minWidth: 0 }}>
                {preview?.scene ? (
                  <>
                    <div style={{ opacity: previewStale ? 0.55 : 1,
                                  transition: "opacity 120ms" }}>
                      <WorldViewer3D
                        scene={preview.scene}
                        selectedEntity={selectedEntity}
                        onSelectEntity={setSelectedEntity}
                        height={340}
                      />
                    </div>
                    {previewStale && (
                      <div className="rs-hintline">
                        Answers changed — press “Preview scene” to recompile.
                      </div>
                    )}
                    <GateStrip preview={preview} />
                    <WorldEntityInspector
                      scene={preview.scene}
                      selected={selectedEntity}
                      onSelect={setSelectedEntity}
                    />
                  </>
                ) : previewMutation.isPending ? (
                  <div className="rs-flex rs-gap-8 rs-sub"
                       style={{ padding: 20 }}>
                    <Icon name="loader" size={15} className="rs-spin" />
                    Compiling draft scene through the admission gates…
                  </div>
                ) : preview && !preview.ok ? (
                  <GateStrip preview={preview} />
                ) : (
                  <p className="rs-hintline" style={{ padding: 8 }}>
                    Press “Preview scene” to compile the draft and see it
                    here.
                  </p>
                )}
              </div>

              {/* ── right: clarification pages ────────────────────── */}
              <div style={{ minWidth: 0, overflowY: "auto" }}>
                {draft.robot_matches_project === false && (
                  <div className="rs-banner warn" style={{ marginBottom: 10 }}>
                    <span className="rs-grow" style={{ fontSize: 12.5 }}>
                      <b>Robot mismatch:</b> this draft uses{" "}
                      <code className="mono">{draft.capability_id}</code>, but
                      the project&apos;s robot is{" "}
                      <code className="mono">{draft.project_capability_id}</code>.
                      Training under this world uses the draft&apos;s robot.
                    </span>
                  </div>
                )}
                <div className="rs-hintline">
                  {pages.length > 0
                    ? `Page ${page + 1} / ${pages.length}`
                    : "No clarification needed"}{" "}
                  · draft{" "}
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
                                          padding: "3px 0",
                                          cursor: "pointer" }}>
                            <input
                              type="radio"
                              name={q.question_id}
                              checked={selected === choice.id}
                              onChange={() => {
                                setAnswers({
                                  ...answers, [q.question_id]: choice.id,
                                });
                                setPreviewStale(true);
                              }}
                            />
                            <span style={{ fontSize: 13 }}>
                              {choice.label}
                            </span>
                          </label>
                        ))}
                      </div>
                    </Field>
                  );
                })}
              </div>
            </div>
          )}
        </Modal>
      )}
    </>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Icon } from "@/components/rs/icon";
import { Btn } from "@/components/rs/primitives";
import { ModeTimeline, modesFromGraph } from "@/components/ModeTimeline";
import { useSaveBehaviorDraft } from "@/hooks/useBehaviorDraft";
import {
  ApiError,
  authorModeReward,
  browseReferences,
  getJob,
  getModeEvidence,
  getReferenceModes,
  listModeRewards,
  promoteModeReward,
  recordModeEvidenceReceipt,
  scaffoldModeReward,
  searchReferences,
  type ModeRewardFile,
  type ModeRewardResult,
  type ModeEvidenceStatus,
  type PromotedModeReward,
  type ReferenceModeGraph,
} from "@/lib/api";

type ReferencePickerHit = { clip_id: string; text?: string };

const SHA256_RE = /^[a-f0-9]{64}$/;

/** Match the backend's whitespace-only behavior-goal canonicalization. */
export function normalizeModeAuthoringGoal(value?: string | null): string {
  return String(value ?? "").trim().split(/\s+/).filter(Boolean).join(" ");
}

export type ModeRewardReadiness = {
  ready: boolean;
  blocker: string | null;
};

/** One authority for the banner and both mutating actions. Missing evidence is
 * stale evidence: legacy/malformed responses must never become authorable by
 * virtue of JavaScript truthiness. */
export function modeRewardReadiness(
  reward: ModeRewardResult | null,
  requestedGoal: string,
  expected: { robot: string; clipId: string },
): ModeRewardReadiness {
  if (!reward) return { ready: false, blocker: "Scaffold a mode reward first." };
  if (
    reward.clip_id !== expected.clipId
    || reward.reference_robot !== expected.robot
  ) {
    return {
      ready: false,
      blocker:
        `The scaffold identity does not match selected reference ${expected.robot}/${expected.clipId}.`,
    };
  }
  if (reward.context_blocker) {
    return { ready: false, blocker: reward.context_blocker };
  }
  if (
    reward.context_current !== true
    || !SHA256_RE.test(reward.execution_context_digest ?? "")
  ) {
    return {
      ready: false,
      blocker: "The scaffold does not prove the current execution context.",
    };
  }
  if (
    reward.authoring_intent_valid !== true
    || !SHA256_RE.test(reward.authoring_intent_sha256 ?? "")
    || typeof reward.authoring_goal !== "string"
  ) {
    return {
      ready: false,
      blocker: "The scaffold does not contain a verifiable authoring intent.",
    };
  }
  if (
    normalizeModeAuthoringGoal(requestedGoal)
    !== normalizeModeAuthoringGoal(reward.authoring_goal)
  ) {
    return {
      ready: false,
      blocker:
        "The behavior goal changed after this scaffold was created. Re-scaffold to pin the new research intent.",
    };
  }
  return { ready: true, blocker: null };
}

/** Select only the exact robot-scoped reference. A verified current file wins
 * over a newer stale derivative; ties prefer valid/matching intent, then mtime. */
export function selectModeRewardFile(
  files: ModeRewardFile[],
  clipId: string,
  robot: string,
  requestedGoal: string,
): ModeRewardFile | undefined {
  const requested = normalizeModeAuthoringGoal(requestedGoal);
  return files
    .filter((file) => file.clip_id === clipId && file.reference_robot === robot)
    .sort((left, right) => {
      const rank = (file: ModeRewardFile) => [
        file.context_current === true ? 1 : 0,
        file.context_blocker == null ? 1 : 0,
        file.authoring_intent_valid === true ? 1 : 0,
        normalizeModeAuthoringGoal(file.authoring_goal) === requested ? 1 : 0,
        Number.isFinite(file.mtime) ? file.mtime : 0,
      ];
      const a = rank(left);
      const b = rank(right);
      for (let index = 0; index < a.length; index += 1) {
        if (a[index] !== b[index]) return b[index] - a[index];
      }
      return 0;
    })[0];
}

function rewardResultFromFile(file: ModeRewardFile): ModeRewardResult {
  return {
    path: file.path,
    filename: file.filename,
    clip_id: file.clip_id,
    reference_robot: file.reference_robot,
    execution_context_digest: file.execution_context_digest,
    authoring_goal: file.authoring_goal,
    authoring_intent_sha256: file.authoring_intent_sha256,
    authoring_intent_valid: file.authoring_intent_valid,
    context_blocker: file.context_blocker,
    context_current: file.context_current,
    tracking: file.tracking_enabled,
    modes: file.modes,
    unauthored: file.unauthored,
    duration_qa: file.duration_qa,
  };
}

/**
 * Per-mode reward authoring for a composed reference.
 *
 * A composite's segments become OGMP-inspired reward phases, and this is where each one gets
 * its own reward terms. The division of labour is the point: the phase clock,
 * the windows and the dispatch that pays a mode only inside its own window
 * are GENERATED from the automaton — both Tier-D failures in this repo were
 * clock bugs, not reward bugs — and a model is asked only for one mode's
 * function body at a time.
 *
 * So the flow here is deliberately two steps rather than one button. The
 * scaffold is trainable the moment it is written (it carries the tracking
 * backbone); authoring adds task terms on top of it, one window at a time,
 * and each one is re-probed against the project's reward contract before it
 * is kept.
 */
export function ModeRewardPanel({
  slug,
  clipId: initialClipId,
  robot,
  goal = "",
}: {
  slug: string;
  /** The attached reference, when the reward already names one. Absent is
   *  the normal case — per-mode authoring is what you do BEFORE there is a
   *  tracking reward — so the panel can also find a composite itself. */
  clipId?: string;
  robot: string;
  goal?: string;
}) {
  const [clipId, setClipId] = useState<string>(initialClipId ?? "");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<ReferencePickerHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [graph, setGraph] = useState<ReferenceModeGraph | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [reward, setReward] = useState<ModeRewardResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [modeGoals, setModeGoals] = useState<Record<string, string>>({});
  // What the reward chain actually holds, not "did I press Promote in this
  // browser tab". The version number alone was session state: reloading the
  // page after promoting showed "Not in the reward chain yet" over a reward
  // that was already training, and offered to promote it a second time.
  const [promoted, setPromoted] = useState<PromotedModeReward | null>(null);
  // sha256 of the mode-reward file on disk. Compared against the promoted
  // version's `source_sha256` to answer "is what trains still what I
  // authored?" — a filename cannot, because authoring chains to a new name
  // every call while a re-scaffold rewrites the same one.
  const [digest, setDigest] = useState<string>("");
  const [resumed, setResumed] = useState(false);
  const [confirmRescaffold, setConfirmRescaffold] = useState(false);
  // Include the reference-tracking backbone. On by default, matching the
  // endpoint — see the checkbox below for what turning it off costs.
  const [tracking, setTracking] = useState(true);
  const [confirmPartial, setConfirmPartial] = useState(false);
  const [evidence, setEvidence] = useState<ModeEvidenceStatus | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [evidenceBusy, setEvidenceBusy] = useState(false);
  const qc = useQueryClient();
  const saveDraft = useSaveBehaviorDraft(slug);
  const selectionKey = JSON.stringify([
    robot,
    clipId,
    normalizeModeAuthoringGoal(goal),
  ]);
  const selectionGuard = useRef({ key: selectionKey, generation: 0 });
  // Update synchronously with render. An older promise can otherwise resolve
  // between a selection-changing render and that render's effect cleanup.
  if (selectionGuard.current.key !== selectionKey) {
    selectionGuard.current = {
      key: selectionKey,
      generation: selectionGuard.current.generation + 1,
    };
  }

  // `useState` reads its argument once, at mount — and at mount the behavior
  // draft that names the clip is still in flight, so the panel opened on the
  // "pick a composed reference" search even for a project whose scaffold was
  // already on disk. Adopt the clip when it arrives, but only while nothing
  // has been chosen here: a late-resolving query must not overwrite a pick.
  useEffect(() => {
    if (initialClipId && !clipId) setClipId(initialClipId);
  }, [initialClipId, clipId]);

  /** Re-read what is on disk: which version is promoted, and the digest of
   *  this clip's mode-reward file.
   *
   *  Called after every action that changes either. Scaffolding and authoring
   *  used to call `setPromoted(null)` instead, which re-enabled the promote
   *  button by pretending nothing was promoted — true enough until the page
   *  reloaded, at which point the real promoted version came back from disk
   *  and the button locked again, permanently, over a stale reward.
   *
   *  `adopt` also pulls the file itself into view, for the first-visit case
   *  where the panel has no `reward` yet.
   */
  const refresh = useCallback(
    async ({
      adopt = false,
      expectedGeneration = selectionGuard.current.generation,
      isLive,
    }: {
      adopt?: boolean;
      expectedGeneration?: number;
      isLive?: () => boolean;
    } = {}) => {
      const { mode_rewards: files, promoted: p } = await listModeRewards(slug);
      if (
        selectionGuard.current.generation !== expectedGeneration
        || (isLive && !isLive())
      ) {
        return false;
      }
      setPromoted(p);
      const mine = selectModeRewardFile(files, clipId, robot, goal);
      setDigest(mine?.digest ?? "");
      if (adopt && mine) {
        setReward(rewardResultFromFile(mine));
        setTracking(mine.tracking_enabled);
        setResumed(true);
      }
      return true;
    },
    [clipId, goal, robot, slug],
  );

  useEffect(() => {
    if (!clipId) {
      setGraph(null);
      setGraphError(null);
      return;
    }
    let live = true;
    setGraph(null);
    setGraphError(null);
    setReward(null);
    setPromoted(null);
    setResumed(false);
    setConfirmRescaffold(false);
    setConfirmPartial(false);
    const generation = selectionGuard.current.generation;
    getReferenceModes(clipId, robot)
      .then((g) => live && setGraph(g))
      .catch((e) =>
        live &&
        setGraphError(
          e instanceof ApiError ? e.message : "could not read this clip's modes",
        ),
      );
    // Pick up any scaffold already on disk for this clip. Authoring progress
    // used to live only in this component's state, so a reload showed a panel
    // whose only button was "Scaffold reward" — which overwrote the very
    // bodies the reload had hidden.
    refresh({
      adopt: true,
      expectedGeneration: generation,
      isLive: () => live,
    }).catch(() => {
      /* no scaffold yet is the normal first-visit case */
    });
    return () => {
      live = false;
    };
  }, [clipId, refresh, robot, slug]);

  useEffect(() => {
    if (!clipId) return;
    let live = true;
    setEvidence(null);
    setEvidenceError(null);
    getModeEvidence(slug, clipId, robot)
      .then((value) => live && setEvidence(value))
      .catch((e) => {
        if (!live) return;
        setEvidenceError(
          e instanceof ApiError ? e.message : "Could not attest mode evidence",
        );
      });
    return () => {
      live = false;
    };
  }, [
    clipId,
    promoted?.context_current,
    promoted?.execution_context_digest,
    promoted?.selection_current,
    promoted?.source_sha256,
    robot,
    slug,
  ]);

  async function runSearch() {
    setSearching(true);
    setSearchError(null);
    try {
      const trimmed = query.trim();
      // The exact-aware retrieval endpoint accepts both a semantic
      // description and a pasted `robot/clip_id`. The faceted browse endpoint
      // is retained only for the useful empty-query "recent composites"
      // state; its substring filter cannot round-trip a scoped identity.
      if (trimmed) {
        setHits(await searchReferences(trimmed, {
          robot, k: 8, useLlm: false,
        }));
      } else {
        const r = await browseReferences({ robot, composed: true, limit: 8 });
        setHits(r.rows);
      }
    } catch (e) {
      setHits(null);
      setSearchError(
        e instanceof ApiError
          ? e.message
          : "Reference search is unavailable. Try again when the library service is reachable.",
      );
    } finally {
      setSearching(false);
    }
  }

  // Surfaced whenever there is no query yet, so the panel opens on the
  // composites instead of an empty box the user has to guess at.
  useEffect(() => {
    if (clipId || hits !== null) return;
    let live = true;
    browseReferences({ robot, composed: true, limit: 8 })
      .then((r) => {
        if (!live) return;
        setHits(r.rows);
        setSearchError(null);
      })
      .catch(() => {
        if (!live) return;
        setSearchError(
          "Reference search is unavailable. Try again when the library service is reachable.",
        );
      });
    return () => {
      live = false;
    };
  }, [clipId, hits, robot]);

  const picker = (
    <div style={{ marginTop: 10 }}>
      <div className="rs-flex rs-gap-8">
        <input
          className="rs-input rs-grow"
          style={{ fontSize: 12 }}
          aria-label="Search composed references by description or exact robot/clip ID"
          placeholder={`Describe a motion or paste ${robot}/clip_id`}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runSearch()}
        />
        <Btn kind="ghost" size="sm" icon="search" onClick={runSearch}
             disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </Btn>
      </div>
      {searchError && (
        <div
          className="rs-banner err"
          role="alert"
          style={{ fontSize: 11, marginTop: 8 }}
        >
          <Icon name="alert-triangle" size={14} />
          <span className="rs-grow">{searchError}</span>
        </div>
      )}
      {hits && hits.length === 0 && (
        <div className="rs-sub" style={{ fontSize: 11, marginTop: 8 }}>
          No clips matched. Compose one from the reference picker first — a
          per-mode reward needs a motion with more than one phase.
        </div>
      )}
      {hits && hits.length > 0 && (
        <div style={{ display: "grid", gap: 4, marginTop: 8 }}>
          {hits.map((h) => (
            <button
              key={h.clip_id}
              className="rs-btn rs-btn-quiet rs-btn-sm"
              aria-label={`Select exact reference ${robot}/${h.clip_id}`}
              style={{ justifyContent: "flex-start", fontSize: 12,
                       border: "1px solid var(--hairline)" }}
              onClick={() => {
                setClipId(h.clip_id);
                saveDraft.mutate({ reference_clip_id: h.clip_id,
                                   reference_robot: robot });
              }}
            >
              <span className="mono">{robot}/{h.clip_id}</span>
              {h.text && (
                <span className="rs-sub" style={{ marginLeft: 8 }}>
                  {h.text}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  if (!clipId || graphError) {
    return (
      <div className="rs-card" style={{ padding: 14 }}>
        <div className="rs-flex rs-gap-8" style={{ marginBottom: 4 }}>
          <Icon name="layers" size={16} />
          <strong style={{ fontSize: 13 }}>Per-mode reward</strong>
        </div>
        <div className="rs-sub" style={{ fontSize: 11 }}>
          {graphError ? (
            <>
              No mode automaton for <code>{robot}/{clipId}</code> — {graphError}. A
              per-mode reward needs a composed reference; a single clip is one
              mode with nothing to transition to.
            </>
          ) : (
            <>
              A composed reference can be split into time-windowed reward
              phases. Pick one and each phase gets its own terms, paid only
              inside its own window. This is OGMP-inspired; it is not yet a
              closed-loop oracle or a mode-conditioned policy.
            </>
          )}
        </div>
        {picker}
      </div>
    );
  }
  if (!graph) {
    return (
      <div className="rs-card" style={{ padding: 14, fontSize: 12.5 }}>
        <span className="rs-sub">Reading the automaton…</span>
      </div>
    );
  }

  const authored = new Set(
    (reward?.modes ?? []).filter((m) => m.authored).map((m) => m.name),
  );
  const doneCount = authored.size;
  const readiness = modeRewardReadiness(reward, goal, { robot, clipId });
  // Is the version `current.py` points at the file in front of me, byte for
  // byte? An empty digest on either side means "cannot tell", which resolves
  // to false — the safe direction, since the cost of a needless second
  // promotion is a file copy and the cost of the other error is training a
  // reward the user believes they replaced.
  const promotedIsCurrent =
    promoted !== null
    && promoted.clip_id === clipId
    && promoted.reference_robot === robot
    && promoted.context_current === true
    && promoted.selection_current === true
    && promoted.context_blocker == null
    && promoted.promotion_blocker == null
    && promoted.authoring_intent_valid === true
    && SHA256_RE.test(promoted.authoring_intent_sha256 ?? "")
    && normalizeModeAuthoringGoal(promoted.authoring_goal)
      === normalizeModeAuthoringGoal(goal)
    && digest !== ""
    && promoted.source_sha256 === digest;
  const durationQa = reward?.duration_qa;
  const durationWarning = durationQa?.warnings[0];
  async function onScaffold(overwrite = false) {
    setBusy("scaffold");
    setError(null);
    try {
      // `overwrite` defaults to FALSE. It used to be hardcoded true, which
      // defeated the backend's deliberate 409 and silently discarded every
      // authored mode body on a second click.
      const r = await scaffoldModeReward(slug, clipId, {
        robot,
        goal,
        tracking,
        overwrite,
      });
      setReward(r);
      setResumed(false);
      setConfirmRescaffold(false);
      // The scaffold response proves what was written, while the list
      // endpoint is the sole authority for whether those exact bytes are
      // current in the project's selected execution context.
      await refresh({ adopt: true });
      setResumed(false);
      // Record what is being built so the run dialog and the flow card can
      // find it without the user re-searching the library for the clip.
      saveDraft.mutate({
        reference_clip_id: clipId,
        reference_robot: robot,
        mode_reward_filename: r.filename,
      });
      setLog((l) => [...l, `scaffolded ${r.filename} — ${r.modes.length} modes`]);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        // Reload what is actually there rather than reporting a conflict the
        // user can do nothing about.
        const files = await listModeRewards(slug)
          .then((r) => r.mode_rewards)
          .catch(() => [] as ModeRewardFile[]);
        const mine = selectModeRewardFile(files, clipId, robot, goal);
        if (mine) {
          setReward(rewardResultFromFile(mine));
          setTracking(mine.tracking_enabled);
          setResumed(true);
          setLog((l) => [...l, `found existing ${mine.filename} — resumed`]);
          return;
        }
      }
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onAuthor(mode: string) {
    if (!reward || !readiness.ready) return;
    setBusy(mode);
    setError(null);
    try {
      const job = await authorModeReward(slug, clipId, {
        mode,
        robot,
        filename: reward.filename,
        goal,
        mode_goal: modeGoals[mode] ?? "",
      });
      setLog((l) => [...l, `authoring ${mode}… (job ${job.job_id.slice(0, 8)})`]);
      // Poll rather than stream: one authoring call is a single Claude
      // request with a repair retry, so there is no intermediate state worth
      // rendering — only whether it was accepted.
      for (;;) {
        await new Promise((r) => setTimeout(r, 2000));
        const d = await getJob(job.job_id);
        if (d.status === "completed") {
          const out = (d.result ?? {}) as {
            filename?: string;
            modes?: { name: string; authored: boolean }[];
            pending?: string[];
          };
          setReward((prev) =>
            prev
              ? {
                  ...prev,
                  filename: out.filename ?? prev.filename,
                  modes: prev.modes.map((m) => ({
                    ...m,
                    authored:
                      out.modes?.find((x) => x.name === m.name)?.authored ??
                      m.authored,
                  })),
                  unauthored: out.pending ?? prev.unauthored,
                }
              : prev,
          );
          await refresh({ adopt: true });
          setResumed(false);
          setLog((l) => [...l, `${mode}: authored → ${out.filename}`]);
          break;
        }
        if (d.status === "errored" || d.status === "stopped") {
          // Say what the gate said. An authoring rejection is informative —
          // "reads info keys this env does not publish", "still reads as an
          // unauthored stub" — and hiding it behind "failed" wastes it.
          setError(d.error ?? `${mode}: authoring ${d.status}`);
          setLog((l) => [...l, `${mode}: rejected`]);
          break;
        }
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onPromote(allowUnauthored: boolean) {
    if (!reward || !readiness.ready) return;
    setBusy("promote");
    setError(null);
    try {
      const r = await promoteModeReward(slug, clipId, {
        filename: reward.filename,
        allow_unauthored: allowUnauthored,
      });
      // Re-read rather than construct: the promoted record now carries the
      // digest of the file it came from, and a locally-built one would be
      // guessing at it.
      await refresh();
      setConfirmPartial(false);
      // The Rewards tab reads the version chain through react-query; without
      // this the new v<n>.py did not appear until a manual reload, which read
      // as "promote did nothing".
      qc.invalidateQueries({ queryKey: ["rewards", slug] });
      qc.invalidateQueries({ queryKey: ["project", slug] });
      qc.invalidateQueries({ queryKey: ["modeRewards", slug] });
      setLog((l) => [
        ...l,
        `promoted ${r.source_filename} → v${r.version}.py (current.py now points at it)`,
      ]);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onRecordEvidence() {
    setEvidenceBusy(true);
    setEvidenceError(null);
    try {
      setEvidence(await recordModeEvidenceReceipt(slug, clipId, robot));
    } catch (e) {
      setEvidenceError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setEvidenceBusy(false);
    }
  }

  return (
    <div className="rs-card" style={{ padding: 14 }}>
      <div className="rs-flex rs-gap-8" style={{ marginBottom: 4 }}>
        <Icon name="layers" size={16} />
        <strong style={{ fontSize: 13 }}>Per-mode reward</strong>
        <span
          className="mono rs-sub"
          style={{ fontSize: 11 }}
          aria-label={`Selected exact reference ${robot}/${clipId}`}
        >
          {robot}/{clipId}
        </span>
        <span className="rs-grow" />
        {reward && (
          <span className="rs-sub" style={{ fontSize: 11 }}>
            {doneCount}/{reward.modes.length} authored · {reward.filename}
          </span>
        )}
      </div>
      <div className="rs-sub" style={{ fontSize: 11, marginBottom: 10 }}>
        {graph.modes.length} reward phases at {graph.fps.toFixed(0)} fps. The
        fixed-clip windows and episode-time dispatch are generated; a model
        writes only one phase's terms per call. Guards are inspectable metadata
        and do not currently drive runtime handover.
      </div>

      <details
        style={{
          marginBottom: 12,
          padding: "9px 11px",
          border: "1px solid var(--hairline)",
          borderRadius: "var(--radius-md)",
          background: "var(--canvas-soft)",
          fontSize: 11,
        }}
      >
        <summary style={{ cursor: "pointer", fontWeight: 650 }}>
          Implemented capability · fixed phase-window scaffold
        </summary>
        <div className="rs-sub" style={{ marginTop: 7, lineHeight: 1.5 }}>
          {graph.capability.summary}
        </div>
        <div style={{ display: "grid", gap: 4, marginTop: 8 }}>
          <span><strong>Active:</strong> fixed composed reference, per-phase rewards, immutable episode-time dispatch.</span>
          <span><strong>Not active:</strong> runtime guard handover, mode-conditioned policy, ρ-bounded exploration, receding-horizon oracle.</span>
        </div>
      </details>

      <div
        style={{
          marginBottom: 12,
          padding: "10px 11px",
          border: "1px solid var(--hairline)",
          borderLeft: "3px solid var(--st-amber, #d97706)",
          borderRadius: "var(--radius-md)",
          background: "var(--canvas-soft)",
        }}
      >
        <div className="rs-flex rs-gap-8" style={{ alignItems: "flex-start" }}>
          <Icon name="shield-check" size={15} color="var(--st-amber, #d97706)" />
          <div className="rs-grow">
            <div style={{ fontSize: 12, fontWeight: 650 }}>
              Independent mode evidence · observe only
            </div>
            <div className="rs-sub" style={{ fontSize: 10.8, lineHeight: 1.5, marginTop: 3 }}>
              {evidence ? (
                <>
                  No generated, validated, and calibrated objective metric set
                  is registered for this exact reward context. This does not
                  count as fitness or selection evidence.
                  {evidence.recorded && (
                    <> Receipt <code>{evidence.receipt_sha256.slice(0, 12)}</code>.</>
                  )}
                </>
              ) : evidenceError ? (
                evidenceError
              ) : (
                "Checking the active reward, reference, graph, manifest, and selection…"
              )}
            </div>
          </div>
          <Btn
            kind="quiet"
            size="sm"
            icon="file-text"
            disabled={!promotedIsCurrent || evidenceBusy}
            title={
              promotedIsCurrent
                ? "Write an immutable receipt for this exact execution context"
                : "Promote this exact mode reward before recording evidence"
            }
            onClick={onRecordEvidence}
          >
            {evidenceBusy ? "Recording…" : "Record readiness receipt"}
          </Btn>
        </div>
      </div>

      {/* The same timeline the compose dialog showed. It was built once,
          rendered once, and then thrown away — this is the screen where
          "which slice am I authoring" actually matters. */}
      <div style={{ marginBottom: 12 }}>
        <div className="rs-sub" style={{ fontSize: 10.5, marginBottom: 4 }}>
          The clip's own timeline, in clip seconds.
        </div>
        <ModeTimeline
          modes={modesFromGraph(graph)}
          fps={graph.fps}
          transitions={graph.transitions}
        />
        {reward && (
          <div className="rs-sub" style={{ fontSize: 10.5, marginTop: 6 }}>
            These windows preserve the certified clip cadence exactly. If the
            episode outlasts the clip, the executor holds the terminal mode and
            final reference frame; it never stretches a certified motion.
          </div>
        )}
        {durationQa && durationWarning && (
          <div
            className="rs-banner"
            role="note"
            style={{
              fontSize: 11,
              lineHeight: 1.5,
              marginTop: 8,
              borderLeft: "3px solid var(--st-amber, #d97706)",
            }}
          >
            <Icon name="info" size={14} color="var(--st-amber, #d97706)" />
            <span className="rs-grow">
              <b>Duration advisory.</b> <code>{durationWarning.mode}</code> owns{" "}
              {durationWarning.duration_s.toFixed(2)} of{" "}
              {durationQa.episode_duration_s.toFixed(2)} seconds ({
                (durationWarning.episode_share * 100).toFixed(1)
              }%). Reward is summed raw per step, not normalized by duration.
              Gate stationary bonuses on task progress and compare per-mode
              reward mass after rollout. This is advisory and does not block
              authoring or launch.
            </span>
          </div>
        )}
      </div>

      {resumed && (
        <div className="rs-banner" style={{ fontSize: 11.5, marginBottom: 10 }}>
          <Icon name="history" size={14} />
          <span className="rs-grow">
            Resumed <code>{reward?.filename}</code> from disk — {doneCount} of{" "}
            {reward?.modes.length} modes already authored.
          </span>
        </div>
      )}

      {reward && !readiness.ready && (
        <div
          className="rs-banner err"
          role="alert"
          style={{ fontSize: 11.5, marginBottom: 10, alignItems: "flex-start" }}
        >
          <Icon name="alert-triangle" size={15} />
          <span className="rs-grow" style={{ lineHeight: 1.5 }}>
            <b>Mode reward is not ready.</b> {readiness.blocker} Authoring and
            promotion are blocked so the warning and the actions use the same
            exact authority.
          </span>
          <Btn
            kind="quiet"
            size="sm"
            icon="refresh-cw"
            disabled={busy !== null}
            onClick={() => setConfirmRescaffold(true)}
          >
            Re-scaffold
          </Btn>
        </div>
      )}

      {!reward && (
        <div style={{ display: "grid", gap: 8 }}>
          <Btn kind="primary" size="sm" icon="layers"
               disabled={busy !== null} onClick={() => onScaffold(false)}>
            {busy === "scaffold" ? "Scaffolding…" : "Scaffold reward"}
          </Btn>
          {/* The backbone was hardcoded on, so pure-task OGMP — modes that
              pay for what the robot achieves rather than for matching the
              clip pose-by-pose — could not be scaffolded from the UI at
              all. It is off the happy path, hence a plain checkbox with the
              consequence stated rather than a prominent control. */}
          <label
            className="rs-flex rs-gap-6"
            style={{ fontSize: 11.5, color: "var(--rs-muted)", alignItems: "flex-start" }}
          >
            <input
              type="checkbox"
              checked={!tracking}
              disabled={busy !== null}
              onChange={(e) => setTracking(!e.target.checked)}
              style={{ marginTop: 2 }}
            />
            <span>
              Task terms only — omit the reference-tracking backbone.
              {!tracking && (
                <b style={{ color: "var(--st-amber, #d97706)", display: "block" }}>
                  Every mode starts as a stub paying zero, so the reward is not
                  trainable until you have authored all {graph.modes.length}.
                </b>
              )}
            </span>
          </label>
        </div>
      )}

      {reward && (
        <div style={{ display: "grid", gap: 8 }}>
          {reward.modes.map((m) => {
            const isDone = authored.has(m.name);
            return (
              <div key={m.name} className="rs-flex rs-gap-10">
                <Icon
                  name={isDone ? "check-circle" : "circle"}
                  size={15}
                  color={isDone ? "var(--st-emerald, #10b981)" : "var(--rs-muted)"}
                />
                <div style={{ minWidth: 108 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 500 }}>{m.name}</div>
                  <div className="rs-sub" style={{ fontSize: 10.5 }}>
                    {m.start_s.toFixed(2)}s – {m.end_s.toFixed(2)}s
                  </div>
                </div>
                <input
                  className="rs-input rs-grow"
                  style={{ fontSize: 12 }}
                  placeholder={`what "${m.name}" has to do`}
                  value={modeGoals[m.name] ?? ""}
                  disabled={isDone || busy !== null}
                  onChange={(e) =>
                    setModeGoals((g) => ({ ...g, [m.name]: e.target.value }))
                  }
                />
                <Btn
                  kind={isDone ? "quiet" : "primary"}
                  size="sm"
                  icon={isDone ? "check" : "sparkles"}
                  disabled={isDone || busy !== null || !readiness.ready}
                  title={!readiness.ready ? readiness.blocker ?? undefined : undefined}
                  onClick={() => onAuthor(m.name)}
                >
                  {busy === m.name ? "Authoring…" : isDone ? "Authored" : "Author"}
                </Btn>
              </div>
            );
          })}
        </div>
      )}

      {reward && (
        <div
          className="rs-flex rs-gap-8"
          style={{ marginTop: 12, paddingTop: 11,
                   borderTop: "1px solid var(--hairline)" }}
        >
          <div className="rs-grow rs-sub" style={{ fontSize: 11 }}>
            {!readiness.ready ? (
              <>
                This file cannot be authored or selected for training until
                its context and pinned research intent are current.
              </>
            ) : promotedIsCurrent ? (
              <>
                Promoted as <b>v{promoted!.version}.py</b> — <code>current.py</code>{" "}
                points at it, so this is what a run trains.
              </>
            ) : promoted !== null ? (
              <>
                A run still trains <b>v{promoted.version}.py</b>
                {promoted.source_filename
                  ? <>, promoted from <code>{promoted.source_filename}</code></>
                  : null}
                {" "}— not the file above. Promote to replace it.
              </>
            ) : confirmPartial ? (
              <>
                <b>{reward.unauthored.join(", ") || "Some modes"}</b>{" "}
                {reward.unauthored.length === 1 ? "is" : "are"} still a stub —
                that slice of every episode pays exactly zero. Promote anyway?
              </>
            ) : doneCount < reward.modes.length ? (
              <>
                {reward.modes.length - doneCount} mode(s) still pay nothing.
                {reward.tracking
                  ? " Promoting now trains the tracking backbone alone in those modes."
                  : " Promoting now leaves those modes at zero reward; no tracking backbone is active."}
              </>
            ) : (
              <>
                Not in the reward chain yet. Until you promote it, a run trains
                whatever <code>current.py</code> points at.
              </>
            )}
          </div>
          {confirmPartial && (
            <Btn kind="quiet" size="sm"
                 onClick={() => setConfirmPartial(false)}>
              Cancel
            </Btn>
          )}
          <Btn
            kind={
              confirmPartial
                ? "danger"
                : doneCount === reward.modes.length
                  ? "primary"
                  : "ghost"
            }
            size="sm"
            icon="check-circle"
            // Gated on "the promoted version IS this exact file", not on
            // "something is promoted". The latter meant the first promotion
            // disabled the control forever: re-scaffold with a corrected clock,
            // re-author all four modes, and there was no click anywhere that
            // could make the new reward the one a run trains.
            disabled={busy !== null || promotedIsCurrent || !readiness.ready}
            title={!readiness.ready ? readiness.blocker ?? undefined : undefined}
            onClick={() => {
              // `allow_unauthored` used to be passed automatically whenever a
              // mode was unauthored, which turned the backend's explicit
              // opt-in into a silent default — the refusal the user is
              // supposed to see never happened.
              if (doneCount < reward.modes.length && !confirmPartial) {
                setConfirmPartial(true);
                return;
              }
              onPromote(doneCount < reward.modes.length);
            }}
          >
            {busy === "promote"
              ? "Promoting…"
              : !readiness.ready
                ? "Re-scaffold required"
                : promotedIsCurrent
                ? `Training v${promoted!.version}`
                : confirmPartial
                  ? "Promote incomplete"
                  : promoted !== null
                    ? `Use for training (replaces v${promoted.version})`
                    : "Use for training"}
          </Btn>
        </div>
      )}

      {/* Promotion is not a one-way door. This was gated on `promoted === null`,
          so the moment you promoted — which you must do to train — the only
          control that can regenerate the automaton disappeared for good. That
          left no way at all to pick up a corrected scaffold: when the mode
          windows were found to be on the clip's clock instead of the episode's,
          a project that had already trained could not be re-scaffolded from the
          UI, at any point, by any sequence of clicks. */}
      {reward && (readiness.ready || confirmRescaffold) && (
        <div className="rs-flex rs-gap-8" style={{ marginTop: 8 }}>
          <span className="rs-grow" />
          {confirmRescaffold ? (
            <>
              <span className="rs-sub" style={{ fontSize: 10.5 }}>
                Re-scaffolding discards all {doneCount} authored{" "}
                {doneCount === 1 ? "body" : "bodies"}.
                {promoted !== null && (
                  <> <code>v{promoted.version}.py</code> stays the active
                  reward until you promote again.</>
                )}
              </span>
              <Btn kind="quiet" size="sm"
                   onClick={() => setConfirmRescaffold(false)}>
                Cancel
              </Btn>
              <Btn kind="danger" size="sm" disabled={busy !== null}
                   onClick={() => onScaffold(true)}>
                Discard and re-scaffold
              </Btn>
            </>
          ) : (
            <Btn kind="quiet" size="sm" icon="refresh-cw" disabled={busy !== null}
                 onClick={() => setConfirmRescaffold(true)}>
              Re-scaffold
            </Btn>
          )}
        </div>
      )}

      {error && (
        <div className="rs-banner err" style={{ fontSize: 11.5, marginTop: 10 }}>
          <Icon name="alert-triangle" size={15} />
          <span className="rs-grow">{error}</span>
        </div>
      )}

      {log.length > 0 && (
        <div
          className="rs-sub"
          style={{ fontSize: 10.5, marginTop: 10, lineHeight: 1.6 }}
        >
          {log.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}

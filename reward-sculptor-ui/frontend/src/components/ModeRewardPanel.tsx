import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Icon } from "@/components/rs/icon";
import { Btn } from "@/components/rs/primitives";
import { ModeTimeline } from "@/components/ModeTimeline";
import {
  ApiError,
  authorModeReward,
  browseReferences,
  getJob,
  getReferenceModes,
  listModeRewards,
  promoteModeReward,
  scaffoldModeReward,
  type ModeRewardResult,
  type ReferenceModeGraph,
} from "@/lib/api";
import type { RefIndexRow } from "@/lib/types";

/**
 * Per-mode reward authoring for a composed reference.
 *
 * A composite's phases ARE its OGMP modes, and this is where each one gets
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
  robot = "g1",
  goal = "",
}: {
  slug: string;
  /** The attached reference, when the reward already names one. Absent is
   *  the normal case — per-mode authoring is what you do BEFORE there is a
   *  tracking reward — so the panel can also find a composite itself. */
  clipId?: string;
  robot?: string;
  goal?: string;
}) {
  const [clipId, setClipId] = useState<string>(initialClipId ?? "");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<RefIndexRow[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [graph, setGraph] = useState<ReferenceModeGraph | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [reward, setReward] = useState<ModeRewardResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [modeGoals, setModeGoals] = useState<Record<string, string>>({});
  const [promoted, setPromoted] = useState<number | null>(null);
  const [resumed, setResumed] = useState(false);
  const [confirmRescaffold, setConfirmRescaffold] = useState(false);
  const [confirmPartial, setConfirmPartial] = useState(false);
  const qc = useQueryClient();

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
    listModeRewards(slug)
      .then((files) => {
        if (!live) return;
        const mine = files.find((f) => f.clip_id === clipId);
        if (!mine) return;
        setReward({
          path: mine.path,
          filename: mine.filename,
          clip_id: mine.clip_id,
          tracking: true,
          modes: mine.modes,
          unauthored: mine.unauthored,
        });
        setResumed(true);
      })
      .catch(() => {/* no scaffold yet is the normal first-visit case */});
    return () => {
      live = false;
    };
  }, [clipId, robot, slug]);

  async function runSearch() {
    setSearching(true);
    try {
      // Browse, not semantic search: a composite you just made is found by
      // its own name, and `listReferences` capped that at the ten
      // alphabetically-first clips of ~6000.
      const r = await browseReferences({ robot, q: query, limit: 8 });
      setHits(r.rows as RefIndexRow[]);
    } catch {
      setHits([]);
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
      .then((r) => live && setHits(r.rows as RefIndexRow[]))
      .catch(() => {});
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
          placeholder="find a composed reference, e.g. 'running jump kick'"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runSearch()}
        />
        <Btn kind="ghost" size="sm" icon="search" onClick={runSearch}
             disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </Btn>
      </div>
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
              style={{ justifyContent: "flex-start", fontSize: 12,
                       border: "1px solid var(--hairline)" }}
              onClick={() => setClipId(h.clip_id)}
            >
              <span className="mono">{h.clip_id}</span>
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
              No mode automaton for <code>{clipId}</code> — {graphError}. A
              per-mode reward needs a composed reference; a single clip is one
              mode with nothing to transition to.
            </>
          ) : (
            <>
              A composed reference's phases ARE its OGMP modes. Pick one and
              each phase gets its own reward terms, paid only inside its own
              window.
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
        tracking: true,
        overwrite,
      });
      setReward(r);
      setPromoted(null);
      setResumed(false);
      setConfirmRescaffold(false);
      setLog((l) => [...l, `scaffolded ${r.filename} — ${r.modes.length} modes`]);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        // Reload what is actually there rather than reporting a conflict the
        // user can do nothing about.
        const files = await listModeRewards(slug).catch(() => []);
        const mine = files.find((f) => f.clip_id === clipId);
        if (mine) {
          setReward({
            path: mine.path, filename: mine.filename, clip_id: mine.clip_id,
            tracking: true, modes: mine.modes, unauthored: mine.unauthored,
          });
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
    if (!reward) return;
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
          setPromoted(null);
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
    if (!reward) return;
    setBusy("promote");
    setError(null);
    try {
      const r = await promoteModeReward(slug, clipId, {
        filename: reward.filename,
        allow_unauthored: allowUnauthored,
      });
      setPromoted(r.version);
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

  return (
    <div className="rs-card" style={{ padding: 14 }}>
      <div className="rs-flex rs-gap-8" style={{ marginBottom: 4 }}>
        <Icon name="layers" size={16} />
        <strong style={{ fontSize: 13 }}>Per-mode reward</strong>
        <span className="mono rs-sub" style={{ fontSize: 11 }}>{clipId}</span>
        <span className="rs-grow" />
        {reward && (
          <span className="rs-sub" style={{ fontSize: 11 }}>
            {doneCount}/{reward.modes.length} authored · {reward.filename}
          </span>
        )}
      </div>
      <div className="rs-sub" style={{ fontSize: 11, marginBottom: 10 }}>
        {graph.modes.length} modes at {graph.fps.toFixed(0)} fps. The windows
        and the dispatch are generated from the automaton; a model writes only
        one mode's terms per call.
      </div>

      {/* The same timeline the compose dialog showed. It was built once,
          rendered once, and then thrown away — this is the screen where
          "which slice am I authoring" actually matters. */}
      <div style={{ marginBottom: 12 }}>
        <ModeTimeline graph={graph} />
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

      {!reward && (
        <Btn kind="primary" size="sm" icon="layers"
             disabled={busy !== null} onClick={() => onScaffold(false)}>
          {busy === "scaffold" ? "Scaffolding…" : "Scaffold reward"}
        </Btn>
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
                  disabled={isDone || busy !== null}
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
            {promoted !== null ? (
              <>
                Promoted to <b>v{promoted}.py</b> — a run will train this
                reward.
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
                Promoting now trains the tracking backbone alone.
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
            disabled={busy !== null || promoted !== null}
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
              : promoted !== null
                ? `Training v${promoted}`
                : confirmPartial
                  ? "Promote incomplete"
                  : "Use for training"}
          </Btn>
        </div>
      )}

      {reward && promoted === null && (
        <div className="rs-flex rs-gap-8" style={{ marginTop: 8 }}>
          <span className="rs-grow" />
          {confirmRescaffold ? (
            <>
              <span className="rs-sub" style={{ fontSize: 10.5 }}>
                Re-scaffolding discards all {doneCount} authored{" "}
                {doneCount === 1 ? "body" : "bodies"}.
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

import { useEffect, useState } from "react";
import { Icon } from "@/components/rs/icon";
import {
  ApiError,
  authorModeReward,
  getJob,
  getReferenceModes,
  scaffoldModeReward,
  type ModeRewardResult,
  type ReferenceModeGraph,
} from "@/lib/api";

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
  clipId,
  robot = "g1",
  goal = "",
}: {
  slug: string;
  clipId: string;
  robot?: string;
  goal?: string;
}) {
  const [graph, setGraph] = useState<ReferenceModeGraph | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [reward, setReward] = useState<ModeRewardResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [modeGoals, setModeGoals] = useState<Record<string, string>>({});

  useEffect(() => {
    let live = true;
    setGraph(null);
    setGraphError(null);
    getReferenceModes(clipId, robot)
      .then((g) => live && setGraph(g))
      .catch((e) =>
        live &&
        setGraphError(
          e instanceof ApiError ? e.message : "could not read this clip's modes",
        ),
      );
    return () => {
      live = false;
    };
  }, [clipId, robot]);

  if (graphError) {
    return (
      <div className="rs-card" style={{ padding: 14 }}>
        <div className="rs-banner info" style={{ fontSize: 12 }}>
          <Icon name="info" size={15} />
          <span className="rs-grow">
            No mode automaton for <code>{clipId}</code> — {graphError}. Per-mode
            rewards need a composed reference; a single clip is one mode with
            nothing to transition to.
          </span>
        </div>
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

  async function onScaffold() {
    setBusy("scaffold");
    setError(null);
    try {
      const r = await scaffoldModeReward(slug, clipId, {
        robot,
        goal,
        tracking: true,
        overwrite: true,
      });
      setReward(r);
      setLog((l) => [...l, `scaffolded ${r.filename} — ${r.modes.length} modes`]);
    } catch (e) {
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

  return (
    <div className="rs-card" style={{ padding: 14 }}>
      <div className="rs-row" style={{ alignItems: "center", marginBottom: 4 }}>
        <Icon name="layers" size={16} />
        <strong style={{ fontSize: 13 }}>Per-mode reward</strong>
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

      {!reward && (
        <button
          className="rs-btn"
          disabled={busy !== null}
          onClick={onScaffold}
        >
          {busy === "scaffold" ? "Scaffolding…" : "Scaffold reward"}
        </button>
      )}

      {reward && (
        <div style={{ display: "grid", gap: 8 }}>
          {reward.modes.map((m) => {
            const isDone = authored.has(m.name);
            return (
              <div
                key={m.name}
                className="rs-row"
                style={{ alignItems: "center", gap: 8 }}
              >
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
                <button
                  className="rs-btn"
                  disabled={isDone || busy !== null}
                  onClick={() => onAuthor(m.name)}
                >
                  {busy === m.name ? "Authoring…" : isDone ? "Authored" : "Author"}
                </button>
              </div>
            );
          })}
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

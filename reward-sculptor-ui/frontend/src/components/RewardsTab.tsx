import { useEffect, useMemo, useState } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { AuthorBadge, Btn, Delta, Modal } from "@/components/rs/primitives";
import { MonacoDiffLazy, MonacoLazy } from "@/components/MonacoLazy";
import { useJob, useJobEvents } from "@/hooks/useJob";
import { useMissions } from "@/hooks/useMissions";
import { useRuns } from "@/hooks/useRuns";
import {
  useRegenerateRewardTemplate,
  useReward,
  useRewardDiagnosis,
  useRewardPromptEdit,
  useRewards,
  useSaveReward,
} from "@/hooks/useRewards";
import { ApiError, getMission } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  MissionDetail,
  MissionSummary,
  ProjectDetail,
  RewardVersionDetail,
  RewardVersionSummary,
  RunSummary,
  SelectedStage,
} from "@/lib/types";

const SCULPT_LOCK_NOTE =
  "Sculpt run in progress — manual edits locked until the run completes or is stopped.";

// What-is-this-page explainer for first-time users. Dismissal persists per
// browser (localStorage), so it shows once and stays gone.
const REWARDS_EXPLAINER_KEY = "rs.rewardsExplainer.dismissed";

function RewardsExplainer() {
  const [dismissed, setDismissed] = useState(() => {
    try { return localStorage.getItem(REWARDS_EXPLAINER_KEY) === "1"; }
    catch { return false; /* private mode */ }
  });
  if (dismissed) return null;
  const dismiss = () => {
    setDismissed(true);
    try { localStorage.setItem(REWARDS_EXPLAINER_KEY, "1"); }
    catch { /* private mode, ignore */ }
  };
  return (
    <div className="rs-banner info" style={{ alignItems: "flex-start" }}>
      <Icon name="info" size={17} />
      <span className="rs-grow" style={{ lineHeight: 1.55 }}>
        <b>How reward evolution works.</b> Each training iteration, the sculptor trains a policy
        with the current reward, watches the rollout, diagnoses the failure, and writes the next
        reward version. Versions that improve the tracked metric are kept; regressions are
        reverted. The list below is that history — every version can be read, diffed against its
        parent, and traced to the diagnosis that produced it. You can also fork any version by
        hand, or ask Claude for a rewrite in the prompt box.
      </span>
      <button
        className="rs-modal-x"
        style={{ flexShrink: 0 }}
        onClick={dismiss}
        aria-label="Dismiss explainer"
        title="Dismiss — won't show again"
      >
        <Icon name="x" size={15} />
      </button>
    </div>
  );
}

export function RewardsTab({
  slug, project, selectedStage, setSelectedStage,
}: {
  slug: string;
  project: ProjectDetail;
  // §Ship de-silo: the shared cross-tab stage selection (owned by
  // ProjectDetail, URL-synced as `?stage=missionSlug/stageName`). When
  // present it takes precedence over this tab's own live/sticky-stage
  // auto-detection — see `effectiveScope` below.
  selectedStage?: SelectedStage | null;
  setSelectedStage?: (value: SelectedStage | null) => void;
}) {
  // §Ship 21b/21d scope logic — preserved verbatim from the prior version.
  // When a mission_stage_run is active, the project's global rewards/v0.py
  // doesn't change; Claude's edits land in the STAGE's rewards dir. Auto-
  // scope here so versions accumulate live; keep useRuns polling across
  // stage boundaries; make the stage scope STICKY so it bridges gaps +
  // survives mission end for review.
  const missions = useMissions(slug);
  const missionActive = useMemo(
    () => (missions.data ?? []).some((m) => m.active_job_id != null),
    [missions.data],
  );
  const runs = useRuns(slug, { keepPolling: missionActive });
  const activeStageRun = useMemo(() => {
    return (runs.data ?? []).find(
      (r) =>
        r.kind === "mission_stage_run"
        && (r.status === "running" || r.status === "queued")
        && r.mission_slug && r.stage_name,
    ) ?? null;
  }, [runs.data]);
  const liveStageScope = activeStageRun
    ? `${activeStageRun.mission_slug}/${activeStageRun.stage_name}`
    : null;
  const [stickyStageScope, setStickyStageScope] = useState<string | null>(null);
  useEffect(() => {
    if (liveStageScope && liveStageScope !== stickyStageScope) {
      setStickyStageScope(liveStageScope);
    }
  }, [liveStageScope, stickyStageScope]);
  const stageScope = liveStageScope ?? stickyStageScope;

  // §Ship de-silo: `?stage=` is now owned entirely by ProjectDetail, which
  // parses it into the `selectedStage` prop and passes a setter back down.
  // (Previously this tab also parsed/wrote `?stage=` itself as a one-shot
  // seed for `scopeOverride` — that's removed: two components racing to
  // read+rewrite the same URL param caused exactly the "double-parse"
  // fight this increment closes. The prop now carries the same
  // information, so nothing is lost.)
  const selectedStageScope = selectedStage
    ? `${selectedStage.missionSlug}/${selectedStage.stageName}`
    : null;

  // Manual dropdown pick, used only when there's no shared selection to
  // route through (see RewardsScopeSelector's onChange below). Cleared
  // whenever the shared prop changes to a new value, so a stage clicked
  // in Training/Overview always wins over a stale local pick.
  const [scopeOverride, setScopeOverride] = useState<string | null>(null);
  const [lastPropScope, setLastPropScope] = useState<string | null>(null);
  useEffect(() => {
    if (selectedStageScope !== lastPropScope) {
      setLastPropScope(selectedStageScope);
      // Only clear the local override when the shared prop transitions to a
      // NEW non-null stage — that's the "external stage click should win"
      // case. A transition to null (e.g. our own onChange below calling
      // setSelectedStage(null) right before setScopeOverride(next)) must
      // NOT clear it, or a manual "project" pick self-inflicts a clear one
      // tick later and snaps back to the stage scope.
      if (selectedStageScope !== null && selectedStageScope !== scopeOverride) {
        setScopeOverride(null);
      }
    }
  }, [selectedStageScope, lastPropScope, scopeOverride]);

  // Precedence: explicit shared selectedStage prop > local manual override
  // (dropdown pick, only reachable when no setSelectedStage setter was
  // given — see RewardsScopeSelector's onChange) > live stage > sticky
  // stage. The prop sits above the manual override because a stage click
  // in Training/Overview should win over a stale local pin; but when this
  // tab's own dropdown IS the routing path (no setter provided), the
  // override still needs to beat the live/sticky auto-scope, matching the
  // pre-existing manual-pick-wins-over-auto-follow behavior.
  const effectiveScope: string | null = useMemo(() => {
    if (selectedStageScope !== null) return selectedStageScope;
    if (scopeOverride === "project") return null;
    if (scopeOverride) return scopeOverride;
    return stageScope;
  }, [selectedStageScope, scopeOverride, stageScope]);
  const isStageScope = effectiveScope !== null;
  const pollMs = missionActive || liveStageScope != null ? 3000 : null;

  const list = useRewards(slug, effectiveScope, pollMs);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [draftSource, setDraftSource] = useState<string | null>(null);
  const [draftParentVersion, setDraftParentVersion] = useState<number | null>(null);

  useEffect(() => {
    if (selectedVersion === null && list.data && list.data.length > 0) {
      setSelectedVersion(list.data[0].version);
    }
  }, [list.data, selectedVersion]);

  useEffect(() => {
    setSelectedVersion(null);
    setDraftSource(null);
    setDraftParentVersion(null);
  }, [effectiveScope]);

  useEffect(() => {
    if (!isStageScope || !list.data || list.data.length === 0) return;
    const newest = list.data[0].version;
    if (selectedVersion !== null && newest > selectedVersion && draftSource === null) {
      setSelectedVersion(newest);
    }
  }, [list.data, isStageScope, selectedVersion, draftSource]);

  const detail = useReward(slug, selectedVersion ?? undefined, effectiveScope, pollMs);

  useEffect(() => {
    setDraftSource(null);
    setDraftParentVersion(null);
  }, [selectedVersion]);

  const isRunning = project.status === "running";
  const isDrafting = draftSource !== null;
  const editsLockedByStageScope = isStageScope;
  const latestVersion = list.data && list.data.length > 0
    ? Math.max(...list.data.map((v) => v.version))
    : null;

  return (
    <div className="rs-scroll">
      <div className="rs-pad rs-vgap-16">
        <RewardsExplainer />
        <PromptEditHero
          slug={slug}
          disabled={Boolean(project.adapter_unavailable) || project.ready_to_train === false}
          adapterUnavailable={Boolean(project.adapter_unavailable)}
          latestVersion={latestVersion}
        />

        <RewardsScopeSelector
          slug={slug}
          missions={missions.data ?? []}
          activeStageRun={activeStageRun}
          stageScope={stageScope}
          scopeOverride={scopeOverride}
          effectiveScope={effectiveScope}
          onChange={(next) => {
            // "Auto-follow" (null) and "project" pins never correspond to a
            // SelectedStage — always local. A stage pick routes through the
            // shared setter when one exists, so picking a stage here also
            // updates Training/Overview and the `?stage=` URL; otherwise it
            // falls back to a local-only override.
            if (next && next !== "project" && setSelectedStage) {
              const idx = next.indexOf("/");
              if (idx > 0) {
                setSelectedStage({ missionSlug: next.slice(0, idx), stageName: next.slice(idx + 1) });
                return;
              }
            }
            if (setSelectedStage) setSelectedStage(null);
            setScopeOverride(next);
          }}
        />

        <div className="rs-twocol">
          {/* Versions */}
          <div className="rs-card">
            <div className="rs-card-head">
              <div className="rs-card-title"><Icon name="history" size={15} />Versions</div>
              <span className="rs-num" style={{ color: "var(--rs-muted)", fontSize: 12 }}>
                {list.data?.length ?? 0}
                {isStageScope && pollMs != null && <span style={{ color: "var(--st-amber-fg)" }}> · live</span>}
              </span>
            </div>
            <div className="rs-verlist">
              {list.isLoading && <p className="rs-sub" style={{ padding: "12px 14px" }}>Loading…</p>}
              {list.data?.length === 0 && !list.isLoading && (
                <p className="rs-sub" style={{ padding: "12px 14px", fontSize: 12 }}>
                  {isStageScope ? "No reward versions yet — stage hasn't materialized v1." : "No reward versions yet."}
                </p>
              )}
              {list.data?.map((v) => (
                <VersionRow key={v.version} v={v} selected={v.version === selectedVersion} onSelect={() => setSelectedVersion(v.version)} />
              ))}
            </div>
          </div>

          {/* Detail */}
          <div className="rs-vgap-16">
            {isRunning && <LockBanner note={SCULPT_LOCK_NOTE} />}
            {editsLockedByStageScope && !isRunning && (
              <LockBanner note="Viewing a mission stage's reward versions — read-only. Switch to Project scope to edit." />
            )}

            {detail.isLoading && <p className="rs-sub">Loading reward…</p>}

            {detail.data && !isDrafting && (
              <ReadOnlyPane
                slug={slug}
                detail={detail.data}
                canEdit={!isRunning && !editsLockedByStageScope}
                stageScope={effectiveScope}
                onNewHumanEdit={() => {
                  setDraftSource(detail.data!.source);
                  setDraftParentVersion(detail.data!.version);
                }}
              />
            )}

            {detail.data && isDrafting && draftParentVersion !== null && (
              <EditorPane
                parentDetail={detail.data}
                draftParentVersion={draftParentVersion}
                initialSource={draftSource!}
                locked={isRunning}
                onDiscard={() => { setDraftSource(null); setDraftParentVersion(null); }}
                onSaved={(newVersion) => {
                  setDraftSource(null);
                  setDraftParentVersion(null);
                  setSelectedVersion(newVersion);
                  toast.success(`Saved v${newVersion}`);
                }}
                onConcurrencyConflict={(current) => {
                  setDraftSource(null);
                  setDraftParentVersion(null);
                  setSelectedVersion(current);
                  toast.error("Version out of date", {
                    description: `The project's latest is now v${current}. Your unsaved edits were discarded — re-open the version, click "New human edit", and try again.`,
                  });
                }}
                slug={slug}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Scope selector (dropdown: Project + every stage of every mission) ─
// §Ship 20 (de-siloing): the old two-button Project/stage switch only
// reached the LIVE stage. This dropdown enumerates every stage of every
// mission (from each mission's detail) so any completed stage's reward
// versions are reachable, while "Auto-follow live stage" stays the
// default when a run is active.
const AUTO_VALUE = "__auto__";
const PROJECT_VALUE = "project";

function RewardsScopeSelector({
  slug, missions, activeStageRun, stageScope, scopeOverride, effectiveScope, onChange,
}: {
  slug: string;
  missions: MissionSummary[];
  activeStageRun: RunSummary | null;
  stageScope: string | null;
  scopeOverride: string | null;
  effectiveScope: string | null;
  onChange: (next: string | null) => void;
}) {
  // Fetch each mission's detail to enumerate its stages. Cheap + cached
  // under the same keys the mission dialog uses.
  const details = useQueries({
    queries: missions.map((m) => ({
      queryKey: qk.mission(slug, m.mission_slug),
      queryFn: () => getMission(slug, m.mission_slug),
      staleTime: 30_000,
    })),
  });
  const missionStages = useMemo(() => {
    return missions.map((m, i) => {
      const d = details[i]?.data as MissionDetail | undefined;
      return { mission: m, stages: d?.stages ?? [] };
    });
  }, [missions, details]);

  const hasAnyStage = missionStages.some((ms) => ms.stages.length > 0);
  // Nothing to scope to (no missions/stages, no live stage) → hide.
  if (!hasAnyStage && !stageScope && !scopeOverride) return null;

  // The <select> value: explicit local override, else whatever scope is
  // actually in effect (e.g. driven by the shared selectedStage prop or
  // the live/sticky stage), else "auto" (nothing pinned yet).
  const value = scopeOverride ?? effectiveScope ?? AUTO_VALUE;
  const isStage = effectiveScope !== null;

  return (
    <div className="rs-flex-between rs-wrap rs-gap-12">
      <div className="rs-flex rs-gap-8" style={{ alignItems: "center", minWidth: 0 }}>
        <span className="rs-eyebrow" style={{ whiteSpace: "nowrap" }}>Reward scope</span>
        <div className="rs-select" style={{ display: "flex", alignItems: "center" }}>
          <select
            aria-label="Reward scope"
            value={value}
            onChange={(e) => {
              const v = e.target.value;
              onChange(v === AUTO_VALUE ? null : v);
            }}
          >
            {/* Auto-follow only makes sense while a stage is live. */}
            {stageScope && (
              <option value={AUTO_VALUE}>
                Auto-follow live stage{activeStageRun?.stage_name ? ` (${activeStageRun.stage_name})` : ""}
              </option>
            )}
            <option value={PROJECT_VALUE}>Project (global rewards)</option>
            {missionStages.map(({ mission, stages }) =>
              stages.length > 0 ? (
                <optgroup key={mission.mission_slug} label={mission.mission_slug}>
                  {stages.map((s) => (
                    <option
                      key={`${mission.mission_slug}/${s.name}`}
                      value={`${mission.mission_slug}/${s.name}`}
                    >
                      {s.name}
                    </option>
                  ))}
                </optgroup>
              ) : null,
            )}
          </select>
        </div>
      </div>
      <div className="rs-flex rs-gap-12">
        {scopeOverride && stageScope && (
          <button
            className="rs-sub"
            style={{ background: "none", border: 0, fontSize: 12, textDecoration: "underline", textUnderlineOffset: 2, color: "var(--rs-muted)" }}
            onClick={() => onChange(null)}
            title="Stop pinning; follow the active mission stage automatically"
          >
            Auto-follow live stage
          </button>
        )}
        {isStage && activeStageRun && effectiveScope === stageScope && (
          <span className="rs-flex rs-gap-6 rs-sub" style={{ fontSize: 12.5 }}>
            <span className="rs-dot live" />live · polling every 3s
          </span>
        )}
      </div>
    </div>
  );
}

// ── Prompt-edit hero (rs-prompt) ─────────────────────────────────────
function PromptEditHero({
  slug, disabled, adapterUnavailable, latestVersion,
}: { slug: string; disabled: boolean; adapterUnavailable: boolean; latestVersion: number | null }) {
  const [prompt, setPrompt] = useState("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const mutate = useRewardPromptEdit(slug);
  const qc = useQueryClient();
  const job = useJob(activeJobId ?? undefined, { intervalMs: 1500 });
  const events = useJobEvents(activeJobId ?? undefined);

  useEffect(() => {
    if (!job.data || !activeJobId) return;
    if (job.data.status === "completed") {
      const newVersion = (job.data.result?.new_version as number | undefined) ?? null;
      const kgMatches = (job.data.result?.kg_matches as { technique: string; relevance_score: number; arxiv_ids: string[] }[] | undefined) ?? [];
      const minSim = (job.data.result?.kg_min_similarity as number | undefined) ?? 0.35;
      qc.invalidateQueries({ queryKey: qk.rewards(slug) });
      const kgBlurb = kgMatches.length > 0
        ? `KG: ${kgMatches.length} match${kgMatches.length === 1 ? "" : "es"} above sim=${minSim.toFixed(2)} — top: ${kgMatches.slice(0, 2).map((m) => m.technique).join(", ")}${kgMatches.length > 2 ? "…" : ""}`
        : `KG: no matches above sim=${minSim.toFixed(2)} — Claude grounded in physics only`;
      toast.success(newVersion != null ? `Claude wrote v${newVersion}.py` : "Claude wrote a new reward version", { description: kgBlurb });
      setActiveJobId(null);
      setPrompt("");
    } else if (job.data.status === "errored") {
      toast.error("Reward rewrite failed", { description: job.data.error ?? "see /jobs for details" });
      setActiveJobId(null);
    }
  }, [job.data, activeJobId, slug, qc]);

  const inFlight = Boolean(activeJobId) && job.data?.status !== "completed" && job.data?.status !== "errored";

  const onSubmit = () => {
    if (disabled || latestVersion == null) return;
    const trimmed = prompt.trim();
    if (trimmed.length < 3) {
      toast.error("Prompt too short", { description: "Give Claude at least a sentence to work with." });
      return;
    }
    mutate.mutate(
      { prompt: trimmed, expected_parent_version: latestVersion },
      {
        onSuccess: (j) => {
          setActiveJobId(j.job_id);
          toast.success("Claude is rewriting…", { description: `Forking from v${latestVersion}.py — ~30-60 s.` });
        },
        onError: (err) => {
          const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
          toast.error("Could not start prompt edit", { description: msg });
        },
      },
    );
  };

  return (
    <div className="rs-prompt">
      <div className="rs-flex rs-gap-8" style={{ marginBottom: 8 }}>
        <Icon name="sparkles" size={15} color="var(--rs-primary)" />
        <span className="rs-eyebrow">Iterate this reward with Claude</span>
      </div>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder={latestVersion == null ? "No reward file yet — scaffold the project first." : "e.g. 'reward the robot for jumping 30 cm with low torque and minimal lateral drift'"}
        rows={3}
        maxLength={2000}
        disabled={disabled || inFlight || latestVersion == null}
        aria-label="Reward rewrite prompt"
      />
      <div className="rs-prompt-foot">
        <span className="rs-sub" style={{ fontSize: 12.5, display: "flex", alignItems: "center", gap: 6 }}>
          {latestVersion != null ? (
            <>
              <Icon name="git-branch" size={13} />
              branches from <b className="mono" style={{ color: "var(--ink)" }}>v{latestVersion}</b> → v{latestVersion + 1}
            </>
          ) : (
            "no reward yet"
          )}
        </span>
        <Btn kind="primary" size="sm" icon={inFlight ? "loader" : "sparkles"} iconRight={inFlight ? undefined : "arrow-right"} onClick={onSubmit} disabled={disabled || inFlight || latestVersion == null}>
          {inFlight ? "Claude is writing…" : latestVersion != null ? `Generate v${latestVersion + 1}` : "Generate"}
        </Btn>
      </div>
      {(inFlight || events.events.length > 0) && (
        <div style={{ marginTop: 10 }}>
          <PromptActivityPanel events={events.events} connected={events.connected} inFlight={inFlight} />
        </div>
      )}
      {disabled && (
        <p className="rs-sub" style={{ color: "var(--st-amber-fg)", fontSize: 12, marginTop: 8 }}>
          {adapterUnavailable ? "Training is disabled — this project uses a coming-soon adapter." : "Training is disabled for this project."}
        </p>
      )}
    </div>
  );
}

function PromptActivityPanel({
  events, connected, inFlight,
}: { events: { text?: string; ts?: string; type: string }[]; connected: boolean; inFlight: boolean }) {
  const logs = events.filter((e) => e.type === "log_line");
  const tail = logs.slice(-20);
  return (
    <div style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", background: "var(--canvas-soft)", padding: 10, fontSize: 11 }}>
      <div className="rs-flex rs-gap-6" style={{ marginBottom: 6, color: "var(--rs-muted)", fontWeight: 500 }}>
        {inFlight ? <Icon name="loader" size={12} className="rs-spin" /> : <span className="rs-dot" style={{ background: connected ? "var(--st-emerald)" : "var(--rs-muted-soft)" }} />}
        Activity <span style={{ opacity: 0.7 }}>({logs.length})</span>
        {!connected && inFlight && <span style={{ marginLeft: "auto", opacity: 0.7 }}>reconnecting…</span>}
      </div>
      {tail.length === 0 ? (
        <div className="rs-sub" style={{ fontStyle: "italic" }}>{inFlight ? "waiting for first event…" : "no events"}</div>
      ) : (
        <ul className="mono" style={{ maxHeight: 160, overflow: "auto", fontSize: 10.5, lineHeight: 1.4, margin: 0, padding: 0, listStyle: "none" }}>
          {tail.map((ev, i) => (
            <li key={`${ev.ts ?? i}-${i}`} style={{ display: "flex", gap: 8, color: "var(--rs-muted)" }}>
              {ev.ts && <span style={{ flexShrink: 0, opacity: 0.6 }}>{new Date(ev.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>}
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ev.text ?? ""}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ── Version row (rs-verrow) ──────────────────────────────────────────
function VersionRow({ v, selected, onSelect }: { v: RewardVersionSummary; selected: boolean; onSelect: () => void }) {
  return (
    <button className={"rs-verrow" + (selected ? " on" : "")} onClick={onSelect}>
      <span className="vn">v{v.version}</span>
      <AuthorBadge author={v.author} />
      <span className="vmetric" title="Primary metric this version's policy achieved">
        {v.primary_metric != null ? v.primary_metric.toFixed(1) : "—"}
        <br />
        <Delta value={v.metric_delta} title="Metric change vs the previous version" />
      </span>
    </button>
  );
}

function LockBanner({ note }: { note: string }) {
  return (
    <div className="rs-banner warn">
      <span className="rs-dot live" style={{ background: "var(--st-amber)" }} />
      <span className="rs-grow">{note}</span>
    </div>
  );
}

// ── Read-only pane ────────────────────────────────────────────────────
function ReadOnlyPane({
  slug, detail, canEdit, stageScope, onNewHumanEdit,
}: {
  slug: string;
  detail: RewardVersionDetail;
  canEdit: boolean;
  stageScope: string | null;
  onNewHumanEdit: () => void;
}) {
  const [diffMode, setDiffMode] = useState(false);
  const parentVersion = detail.version > 0 ? detail.version - 1 : null;
  const parent = useReward(diffMode && parentVersion !== null ? slug : undefined, parentVersion ?? undefined, stageScope);

  return (
    <>
      <div className="rs-flex-between rs-wrap rs-gap-12">
        <div className="rs-flex rs-gap-10">
          <span className="rs-h3" style={{ fontSize: 20 }}>Reward <span className="mono" style={{ color: "var(--rs-primary)" }}>v{detail.version}</span></span>
          <AuthorBadge author={detail.author} />
        </div>
        <div className="rs-flex rs-gap-8">
          {parentVersion !== null && (
            <button className={"rs-iconbtn" + (diffMode ? " on" : "")} style={{ width: "auto", padding: "0 11px", gap: 6, fontSize: 13 }} onClick={() => setDiffMode((v) => !v)} title={diffMode ? "Show full source" : `Diff v${detail.version} vs v${parentVersion}`}>
              <Icon name="git-compare" size={14} />{diffMode ? "Hide diff" : "Diff vs parent"}
            </button>
          )}
          {detail.version === 0 && <RegenerateTemplateButton slug={slug} disabled={!canEdit} />}
          <Btn kind="ghost" size="sm" icon="pencil" onClick={onNewHumanEdit} disabled={!canEdit} title={canEdit ? "Clone this version into an editable draft" : "Disabled while a sculpt run is active / in stage scope"}>
            New human edit
          </Btn>
        </div>
      </div>

      <div className="rs-code">
        <div className="rs-code-bar">
          <span className="rs-code-file">
            <Icon name="file-code" size={14} color="var(--rs-primary)" />
            {detail.file_name}
            {diffMode && parentVersion !== null && <span style={{ color: "var(--rs-muted)" }}> vs v{parentVersion}.py</span>}
          </span>
          <span className="rs-sub" style={{ fontSize: 11 }}>parent_hash {detail.parent_hash.slice(0, 12) || "—"}</span>
        </div>
        {diffMode && parentVersion !== null ? (
          parent.isLoading ? (
            <div style={{ height: 420, display: "flex", alignItems: "center", justifyContent: "center" }} className="rs-sub">
              <Icon name="loader" size={14} className="rs-spin" /> Loading v{parentVersion}.py…
            </div>
          ) : parent.data ? (
            <MonacoDiffLazy original={parent.data.source} modified={detail.source} height={420} />
          ) : (
            <div style={{ height: 420, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--st-rose)" }}>Could not load parent version.</div>
          )
        ) : (
          <MonacoLazy value={detail.source} readOnly height={420} />
        )}
      </div>

      {detail.author === "sculptor" && detail.version > 0 && (
        <WhyThisEditPanel slug={slug} version={detail.version} stageScope={stageScope} />
      )}

      <SpecPanel detail={detail} />
    </>
  );
}

// ── "Why this edit?" (rs-why) ────────────────────────────────────────
function WhyThisEditPanel({ slug, version, stageScope }: { slug: string; version: number; stageScope?: string | null }) {
  const [open, setOpen] = useState(true);
  const diag = useRewardDiagnosis(slug, version, { enabled: open, stage: stageScope ?? null });
  return (
    <div className="rs-why">
      <button className="rs-why-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <Icon name="alert-circle" size={16} />Why this edit?
        <span className="rs-grow" />
        <Icon name={open ? "chevron-up" : "chevron-down"} size={16} />
      </button>
      {open && (
        <div className="rs-card-pad rs-vgap-8" style={{ background: "var(--surface-card)" }}>
          {diag.isLoading && <p className="rs-sub">Loading diagnosis…</p>}
          {diag.error && <p className="rs-sub">No diagnosis recorded — the version may be human-authored or never reached the diagnose stage.</p>}
          {diag.data && (
            <>
              {diag.data.failure_modes && diag.data.failure_modes.length > 0 && (
                <div>
                  <div className="rs-eyebrow" style={{ marginBottom: 8 }}>Failure modes observed in parent</div>
                  <div className="rs-failchips">
                    {diag.data.failure_modes.map((fm) => (
                      <span key={fm} className="rs-failchip"><Icon name="x" size={11} />{fm}</span>
                    ))}
                  </div>
                </div>
              )}
              {diag.data.evidence && (
                <div>
                  <div className="rs-eyebrow" style={{ marginBottom: 6 }}>Evidence</div>
                  <p className="rs-sub" style={{ lineHeight: 1.6, margin: 0, whiteSpace: "pre-wrap" }}>{diag.data.evidence}</p>
                </div>
              )}
              {diag.data.proposed_edits && diag.data.proposed_edits.length > 0 && (
                <div>
                  <div className="rs-eyebrow" style={{ marginBottom: 6 }}>Proposed edits</div>
                  <div className="rs-vgap-8">
                    {diag.data.proposed_edits.map((e, i) => (
                      <div key={i} style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", padding: 10, background: "var(--canvas-soft)" }}>
                        <div className="rs-flex rs-gap-6 mono" style={{ fontSize: 11 }}>
                          <span className="rs-tag" style={{ background: "var(--timeline-edit)", color: "#2c2257", padding: "1px 7px", fontSize: 10 }}>{e.operation}</span>
                          <span style={{ color: "var(--ink)" }}>{e.target_term}</span>
                          {e.suggested_value && (<><span style={{ color: "var(--rs-muted)" }}>→</span><code style={{ color: "var(--rs-muted)" }}>{e.suggested_value}</code></>)}
                        </div>
                        <p className="rs-sub" style={{ margin: "6px 0 0", lineHeight: 1.5, fontSize: 12.5 }}>{e.rationale}</p>
                        {e.paper_refs && e.paper_refs.length > 0 && (
                          <div className="rs-flex rs-wrap rs-gap-6" style={{ marginTop: 6 }}>
                            {e.paper_refs.map((aid) => (
                              <a key={aid} href={`https://arxiv.org/abs/${aid}`} target="_blank" rel="noreferrer noopener" className="rs-tag mono" style={{ fontSize: 10, color: "var(--ink)" }}>
                                {aid}<Icon name="external" size={11} />
                              </a>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {typeof diag.data.confidence === "number" && (
                <div className="rs-sub" style={{ fontSize: 11 }}>confidence: {diag.data.confidence.toFixed(2)}</div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Regenerate-template button (rs Modal) ────────────────────────────
function RegenerateTemplateButton({ slug, disabled }: { slug: string; disabled: boolean }) {
  const [open, setOpen] = useState(false);
  const regen = useRegenerateRewardTemplate(slug);
  const onConfirm = () => {
    regen.mutate(undefined, {
      onSuccess: () => { setOpen(false); toast.success("Reward template regenerated", { description: "rewards/v0.py rewritten from the adapter scaffold." }); },
      onError: (err) => toast.error("Could not regenerate template", { description: err.message }),
    });
  };
  return (
    <>
      <Btn kind="ghost" size="sm" icon="refresh-cw" onClick={() => setOpen(true)} disabled={disabled} title={disabled ? "Disabled while a sculpt run is active" : "Rewrite v0.py from the adapter scaffold"}>
        Regenerate
      </Btn>
      {open && (
        <Modal
          title="Regenerate reward template?"
          icon="refresh-cw"
          onClose={() => { if (!regen.isPending) setOpen(false); }}
          footer={
            <>
              <Btn kind="quiet" onClick={() => setOpen(false)} disabled={regen.isPending}>Cancel</Btn>
              <Btn kind="primary" icon="refresh-cw" onClick={onConfirm} disabled={regen.isPending}>{regen.isPending ? "Regenerating…" : "Regenerate"}</Btn>
            </>
          }
        >
          <p className="rs-sub" style={{ margin: 0 }}>
            This overwrites <code className="mono">rewards/v0.py</code> with the scaffold template for this project's adapter
            (<code className="mono">compute_reward</code> for gym_sb3, <code className="mono">compute_reward_batched</code> for mjlab).
          </p>
          <p className="rs-sub" style={{ margin: 0, color: "var(--st-amber-fg)" }}>Any manual edits to v0.py will be lost. Git history preserves the previous file.</p>
        </Modal>
      )}
    </>
  );
}

// ── Editor pane (draft) ──────────────────────────────────────────────
function EditorPane({
  parentDetail, draftParentVersion, initialSource, locked, onDiscard, onSaved, onConcurrencyConflict, slug,
}: {
  parentDetail: RewardVersionDetail;
  draftParentVersion: number;
  initialSource: string;
  locked: boolean;
  onDiscard: () => void;
  onSaved: (newVersion: number) => void;
  onConcurrencyConflict: (current: number) => void;
  slug: string;
}) {
  const [source, setSource] = useState(initialSource);
  const [note, setNote] = useState("");
  const save = useSaveReward(slug, draftParentVersion);

  const onSave = () => {
    if (locked) return;
    save.mutate(
      { source, expected_parent_version: draftParentVersion, note: note.trim() },
      {
        onSuccess: (d) => onSaved(d.version),
        onError: (err) => {
          if (err instanceof ApiError) {
            if (err.status === 409 && err.type === "/problems/concurrency-conflict") {
              const current = (err.problem.current as number | undefined) ?? draftParentVersion;
              onConcurrencyConflict(current);
              return;
            }
            if (err.status === 422 && err.type === "/problems/reward-validation") {
              const vs = (err.problem.violations as string[] | undefined) ?? [err.problem.detail ?? err.message];
              toast.error("Reward validation failed", { description: vs.join(" · ") });
              return;
            }
          }
          toast.error("Could not save reward", { description: err.message });
        },
      },
    );
  };

  const violations = useMemo(() => {
    if (save.error instanceof ApiError && save.error.type === "/problems/reward-validation") {
      return (save.error.problem.violations as string[] | undefined) ?? [];
    }
    return [];
  }, [save.error]);

  return (
    <>
      <div className="rs-code">
        <div className="rs-code-bar">
          <span className="rs-code-file">
            <Icon name="file-code" size={14} color="var(--rs-primary)" />
            Editing from v{draftParentVersion}.py → saves as v{draftParentVersion + 1}.py
          </span>
          <div className="rs-flex rs-gap-8">
            <Btn kind="quiet" size="xs" onClick={onDiscard}>Discard</Btn>
            <Btn kind="primary" size="xs" icon="check" onClick={onSave} disabled={locked || save.isPending}>{save.isPending ? "Saving…" : "Save"}</Btn>
          </div>
        </div>
        <MonacoLazy value={source} onChange={(v) => setSource(v ?? "")} readOnly={locked} height={480} />
        <div style={{ borderTop: "1px solid var(--hairline)", padding: 12 }}>
          <label className="rs-eyebrow" htmlFor="reward-note" style={{ display: "block", marginBottom: 6 }}>Note</label>
          <input
            id="reward-note"
            type="text"
            className="rs-input"
            placeholder="Short description (author must be human)"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            maxLength={500}
            disabled={locked}
          />
          {violations.length > 0 && (
            <div className="rs-banner err" style={{ marginTop: 12, alignItems: "flex-start" }}>
              <Icon name="alert-triangle" size={17} />
              <span className="rs-grow">
                <b>Validation blocked the save</b>
                <ul className="mono" style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 11 }}>
                  {violations.map((v, i) => <li key={i} style={{ wordBreak: "break-word" }}>{v}</li>)}
                </ul>
              </span>
            </div>
          )}
        </div>
      </div>
      <SpecPanel detail={parentDetail} label="Parent REWARD_SPEC" />
    </>
  );
}

// ── REWARD_SPEC panel (rs) ───────────────────────────────────────────
function SpecPanel({ detail, label = "REWARD_SPEC" }: { detail: RewardVersionDetail; label?: string }) {
  const spec = detail.spec;
  const hparams = Object.entries(spec.hyperparameters);
  const grounding = Object.entries(spec.grounding ?? {});
  const arxivRe = /^(?:\d{4}\.\d{4,5}|[a-z-]+\/\d{7})/i;
  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="file-text" size={15} />{label}</div>
        <span className="mono" style={{ fontSize: 11, color: "var(--rs-muted)" }}>{detail.parent_hash.slice(0, 8) || "—"}</span>
      </div>
      <div className="rs-card-pad">
        {spec.description && <p className="rs-sub" style={{ margin: "0 0 12px" }}>{spec.description}</p>}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, fontSize: 12 }}>
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 6 }}>Hyperparameters</div>
            {hparams.length === 0 ? <p className="rs-sub" style={{ fontStyle: "italic" }}>(none)</p> : (
              <ul className="mono" style={{ listStyle: "none", margin: 0, padding: 0, fontSize: 11 }}>
                {hparams.map(([k, v]) => (<li key={k} style={{ display: "flex", justifyContent: "space-between", gap: 8 }}><span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{k}</span><span style={{ color: "var(--rs-muted)" }}>{v}</span></li>))}
              </ul>
            )}
          </div>
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 6 }}>Grounding</div>
            {grounding.length === 0 ? <p className="rs-sub" style={{ fontStyle: "italic" }}>(pre-grounding-mandate)</p> : (
              <ul className="mono" style={{ listStyle: "none", margin: 0, padding: 0, fontSize: 11 }}>
                {grounding.map(([k, v]) => {
                  const m = v.match(arxivRe);
                  const aid = m ? m[0] : null;
                  return (
                    <li key={k} style={{ marginBottom: 4 }}>
                      <div style={{ fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis" }}>{k}</div>
                      {aid ? (
                        <a href={`https://arxiv.org/abs/${aid}`} target="_blank" rel="noreferrer noopener" style={{ fontSize: 10, color: "var(--ink)", display: "flex", alignItems: "center", gap: 4 }} title={v}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v}</span><Icon name="external" size={11} />
                        </a>
                      ) : <span style={{ fontSize: 10, color: "var(--rs-muted)" }} title={v}>{v}</span>}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 6 }}>References</div>
            {spec.references.length === 0 ? <p className="rs-sub" style={{ fontStyle: "italic" }}>(novel)</p> : (
              <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                {spec.references.map((r) => (
                  <li key={r.arxiv_id} style={{ marginBottom: 4 }}>
                    <a href={`https://arxiv.org/abs/${r.arxiv_id}`} target="_blank" rel="noreferrer noopener" className="mono" style={{ fontSize: 11, color: "var(--ink)", display: "inline-flex", alignItems: "center", gap: 4 }}>
                      {r.arxiv_id}<Icon name="external" size={11} />
                    </a>
                    {r.citation && <div className="rs-sub" style={{ fontSize: 10, overflow: "hidden", textOverflow: "ellipsis" }}>{r.citation}</div>}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 6 }}>Probe</div>
            {detail.components_probe.ok ? (
              <>
                <div className="mono" style={{ fontSize: 11, marginBottom: 4 }}>total = <span style={{ color: "var(--rs-muted)" }}>{detail.components_probe.total?.toFixed(4) ?? "—"}</span></div>
                <ul className="mono" style={{ listStyle: "none", margin: 0, padding: 0, fontSize: 11 }}>
                  {Object.entries(detail.components_probe.components ?? {}).map(([k, v]) => (<li key={k} style={{ display: "flex", justifyContent: "space-between", gap: 8 }}><span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{k}</span><span style={{ color: "var(--rs-muted)" }}>{v}</span></li>))}
                </ul>
              </>
            ) : <p style={{ color: "var(--st-rose)", fontSize: 11 }}>{detail.components_probe.error ?? "probe failed"}</p>}
          </div>
        </div>
      </div>
    </div>
  );
}

export default RewardsTab;

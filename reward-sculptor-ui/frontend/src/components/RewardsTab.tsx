import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FilePlus2,
  GitCompare,
  Loader2,
  RefreshCw,
  Save,
  Sparkles,
  User2,
  Wand2,
} from "lucide-react";
import { useRuns } from "@/hooks/useRuns";
import { useMissions } from "@/hooks/useMissions";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { MonacoDiffLazy, MonacoLazy } from "@/components/MonacoLazy";
import { useJob, useJobEvents } from "@/hooks/useJob";
import {
  useRegenerateRewardTemplate,
  useReward,
  useRewardDiagnosis,
  useRewardPromptEdit,
  useRewards,
  useSaveReward,
} from "@/hooks/useRewards";
import { ApiError } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import { cn, formatRelative } from "@/lib/utils";
import type {
  ProjectDetail,
  RewardVersionDetail,
  RewardVersionSummary,
  RunSummary,
} from "@/lib/types";

const SCULPT_LOCK_NOTE =
  "Sculpt run in progress — manual edits locked until the run completes or is stopped.";

// §Ship 21b: when an active mission stage is training, default the
// Rewards tab to that stage's scoped rewards/ dir so the user sees
// Claude's iterative reward edits land in real time. Surface a small
// pill toggle so the user can flip to the project-global rewards.
function RewardsScopeSelector({
  activeStageRun,
  stageScope,
  scopeOverride,
  effectiveScope,
  onChange,
}: {
  activeStageRun: RunSummary | null;
  stageScope: string | null;
  scopeOverride: string | null;
  effectiveScope: string | null;
  onChange: (next: string | null) => void;
}) {
  const isProject = effectiveScope === null;
  const isStage = effectiveScope !== null;
  const stageLabel = activeStageRun
    ? `Stage: ${activeStageRun.stage_name ?? "unknown"}${
        typeof activeStageRun.stage_index === "number"
          ? ` (${activeStageRun.stage_index + 1})`
          : ""
      }`
    : stageScope
      ? `Stage: ${stageScope.split("/")[1] ?? stageScope}`
      : "Stage";
  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-3 p-3 text-[11px]">
        <span className="font-medium">Scope:</span>
        <div className="inline-flex rounded-md border bg-background p-0.5">
          <button
            type="button"
            onClick={() => onChange("project")}
            className={cn(
              "rounded-sm px-2 py-0.5 text-[11px] font-medium transition-colors",
              isProject
                ? "bg-foreground text-background"
                : "text-muted-foreground hover:text-foreground",
            )}
            title="Project-global rewards/v0.py, v1.py, ..."
          >
            Project
          </button>
          <button
            type="button"
            onClick={() => onChange(stageScope ?? scopeOverride)}
            disabled={!stageScope && !scopeOverride}
            className={cn(
              "rounded-sm px-2 py-0.5 text-[11px] font-medium transition-colors",
              isStage
                ? "bg-foreground text-background"
                : "text-muted-foreground hover:text-foreground",
              !stageScope && !scopeOverride && "cursor-not-allowed opacity-40",
            )}
            title={
              stageScope
                ? `Stage rewards under .missions/${stageScope}/rewards/`
                : "No active stage — kick off a mission run to enable"
            }
          >
            {stageLabel}
          </button>
        </div>
        {scopeOverride && (
          <button
            type="button"
            onClick={() => onChange(null)}
            className="text-[10.5px] text-muted-foreground underline-offset-2 hover:underline"
            title="Stop pinning; follow the active mission stage automatically"
          >
            Unpin (auto-follow active stage)
          </button>
        )}
        {isStage && activeStageRun && (
          <span className="ml-auto inline-flex items-center gap-1 rounded-sm border border-amber-300/60 bg-amber-50 px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-wide text-amber-900">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
            Live
          </span>
        )}
      </CardContent>
    </Card>
  );
}

export function RewardsTab({
  slug,
  project,
}: {
  slug: string;
  project: ProjectDetail;
}) {
  // §Ship 21b/21d: when a mission_stage_run is active, the project's
  // global rewards/v0.py doesn't change — Claude's edits land in the
  // STAGE's rewards dir at <project>/.missions/<m>/stages/<s>/rewards/.
  // Auto-scope the Rewards tab to the active stage so the user sees
  // versions accumulate live as the mission trains.
  //
  // §Ship 21d reliability rewrite. The old version derived the scope
  // purely from "is a stage RUNNING at this exact instant" via
  // useRuns. That was racy:
  //   1. useRuns stopped polling at stage boundaries (no running run
  //      for a beat between stages) → activeStageRun froze stale.
  //   2. The moment it froze as "completed", scope flipped to project
  //      and reward-polling stopped → "rewards only sometimes update."
  //   3. After the mission ended there was no way to review a stage's
  //      rewards (selector hidden, scope = project).
  // Fixes: (a) keep useRuns polling while a mission is active so the
  // stage list stays fresh; (b) make the stage scope STICKY so it
  // bridges the between-stage gaps and survives mission end for
  // review; (c) poll rewards while the mission is active OR a stage
  // is live, not only at the exact instant a run shows "running".
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
  // The scope of whatever stage is live RIGHT NOW (null between
  // stages / after the mission).
  const liveStageScope = activeStageRun
    ? `${activeStageRun.mission_slug}/${activeStageRun.stage_name}`
    : null;
  // Sticky: remember the last live stage scope so the view doesn't
  // flicker to project rewards during between-stage gaps, and so the
  // user can still review the final stage's rewards after the mission
  // ends. Updates whenever a new live stage appears (incl. a new
  // mission's first stage); never clears on its own.
  const [stickyStageScope, setStickyStageScope] = useState<string | null>(null);
  useEffect(() => {
    if (liveStageScope && liveStageScope !== stickyStageScope) {
      setStickyStageScope(liveStageScope);
    }
  }, [liveStageScope, stickyStageScope]);
  const stageScope = liveStageScope ?? stickyStageScope;

  // User can override the default scope; null = follow `stageScope`
  // automatically (default), "project" = pinned to project rewards,
  // a "<m>/<s>" string = pinned to that stage.
  const [scopeOverride, setScopeOverride] = useState<string | null>(null);
  const effectiveScope: string | null = useMemo(() => {
    if (scopeOverride === "project") return null;
    if (scopeOverride && scopeOverride !== "project") return scopeOverride;
    return stageScope;
  }, [scopeOverride, stageScope]);
  const isStageScope = effectiveScope !== null;
  // Poll rewards every 3s while the mission is active OR a stage is
  // live — covers the between-stage gaps so new vN files surface the
  // moment the scope updates to the next stage. Stops once the
  // mission is fully done (the last fetched version list stays
  // visible for review).
  const pollMs = missionActive || liveStageScope != null ? 3000 : null;

  const list = useRewards(slug, effectiveScope, pollMs);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [draftSource, setDraftSource] = useState<string | null>(null);
  const [draftParentVersion, setDraftParentVersion] = useState<number | null>(null);

  // Default selection: newest version.
  useEffect(() => {
    if (selectedVersion === null && list.data && list.data.length > 0) {
      setSelectedVersion(list.data[0].version);
    }
  }, [list.data, selectedVersion]);

  // §Ship 21b: when scope changes (project ↔ stage), reset the
  // selected version so the next render picks the newest in the new
  // scope rather than holding a stale cross-scope index.
  useEffect(() => {
    setSelectedVersion(null);
    setDraftSource(null);
    setDraftParentVersion(null);
  }, [effectiveScope]);

  // §Ship 21b: also auto-advance to the latest version when a new one
  // appears during a live stage run (so the user sees Claude's
  // newest edit without clicking).
  useEffect(() => {
    if (!isStageScope || !list.data || list.data.length === 0) return;
    const newest = list.data[0].version;
    if (
      selectedVersion !== null
      && newest > selectedVersion
      && draftSource === null
    ) {
      setSelectedVersion(newest);
    }
  }, [list.data, isStageScope, selectedVersion, draftSource]);

  const detail = useReward(
    slug,
    selectedVersion ?? undefined,
    effectiveScope,
    pollMs,
  );

  // When switching versions, drop any unsaved draft.
  useEffect(() => {
    setDraftSource(null);
    setDraftParentVersion(null);
  }, [selectedVersion]);

  const isRunning = project.status === "running";
  const isDrafting = draftSource !== null;
  // §Ship 21b: editing endpoints (PUT /rewards, POST /rewards/prompt)
  // are project-scoped only. When viewing a stage's rewards, edits
  // would target the wrong dir AND clobber Claude's in-flight work.
  // Lock editing while in stage scope; user can still browse Monaco.
  const editsLockedByStageScope = isStageScope;

  const latestVersion = list.data && list.data.length > 0
    ? Math.max(...list.data.map((v) => v.version))
    : null;

  return (
    <div className="flex flex-col gap-4">
      <PromptEditHero
        slug={slug}
        disabled={Boolean(project.adapter_unavailable) || project.ready_to_train === false}
        adapterUnavailable={Boolean(project.adapter_unavailable)}
        latestVersion={latestVersion}
      />
      {/* §Ship 21b/21d: scope selector. Visible whenever there's a
          sticky stage scope (during a mission AND after it ends, for
          review) or the user has pinned a scope. Hidden only in the
          common no-missions-ever case to keep the UI clean. */}
      {(stageScope || scopeOverride) && (
        <RewardsScopeSelector
          activeStageRun={activeStageRun}
          stageScope={stageScope}
          scopeOverride={scopeOverride}
          effectiveScope={effectiveScope}
          onChange={setScopeOverride}
        />
      )}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="order-2 lg:order-1">
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Versions</CardTitle>
            <CardDescription className="text-[11px]">
              {list.data?.length ?? 0} version
              {list.data?.length === 1 ? "" : "s"}
              {isStageScope && pollMs != null && (
                <span className="ml-1 text-amber-700">
                  · live (3s poll)
                </span>
              )}
              {isStageScope && pollMs == null && (
                <span className="ml-1 text-muted-foreground">
                  · stage (finished)
                </span>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 p-2 pt-0">
            {list.isLoading && <p className="text-xs">Loading…</p>}
            {list.data?.length === 0 && !list.isLoading && (
              <p className="text-[10.5px] text-muted-foreground">
                {isStageScope
                  ? "No reward versions yet — stage hasn't materialized v1."
                  : "No reward versions yet."}
              </p>
            )}
            {list.data?.map((v) => (
              <VersionRow
                key={v.version}
                v={v}
                selected={v.version === selectedVersion}
                onSelect={() => setSelectedVersion(v.version)}
              />
            ))}
          </CardContent>
        </Card>
      </aside>

      <section className="order-1 lg:order-2 flex min-w-0 flex-col gap-4">
        {isRunning && (
          <LockBanner note={SCULPT_LOCK_NOTE} />
        )}
        {editsLockedByStageScope && !isRunning && (
          <LockBanner note="Viewing a mission stage's reward versions — read-only. Switch to Project scope to edit." />
        )}

        {detail.isLoading && (
          <p className="text-sm text-muted-foreground">Loading reward…</p>
        )}

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
            onDiscard={() => {
              setDraftSource(null);
              setDraftParentVersion(null);
            }}
            onSaved={(newVersion) => {
              setDraftSource(null);
              setDraftParentVersion(null);
              setSelectedVersion(newVersion);
              toast.success(`Saved v${newVersion}`);
            }}
            onConcurrencyConflict={(current) => {
              // Per R1: discard the draft, don't merge. Toast the
              // current latest so the user can re-clone it.
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
      </section>
      </div>
    </div>
  );
}

// ── Prompt-edit hero — Claude rewrite without running training ───────
function PromptEditHero({
  slug,
  disabled,
  adapterUnavailable,
  latestVersion,
}: {
  slug: string;
  disabled: boolean;
  adapterUnavailable: boolean;
  latestVersion: number | null;
}) {
  const [prompt, setPrompt] = useState("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const mutate = useRewardPromptEdit(slug);
  const qc = useQueryClient();
  const job = useJob(activeJobId, { refetchIntervalMs: 1500 });
  // Live log stream while the prompt-edit job runs. Bug #6.1 / T2:
  // pre-S3 the UI was silent during the 30–60 s Claude call; Sam
  // saw "no progress" and retried → 409. Activity panel surfaces
  // every `log_line` emitted by reward_jobs.py + sculptor.edit.
  const events = useJobEvents(activeJobId ?? undefined);

  // Watch the job; when it completes, refresh + toast.
  useEffect(() => {
    if (!job.data || !activeJobId) return;
    if (job.data.status === "completed") {
      const newVersion = (job.data.result?.new_version as number | undefined) ?? null;
      // Surface KG consultation in the completion toast so Sam can
      // tell whether Claude actually had KG context to draw on.
      // Test 1 round 4 (2026-04-22): Sam's "references says (novel)
      // even though 5 matches exist" confusion stemmed from having
      // no UI surface for what Claude considered vs. what it cited.
      const kgMatches = (job.data.result?.kg_matches as
        | { technique: string; relevance_score: number; arxiv_ids: string[] }[]
        | undefined) ?? [];
      const minSim = (job.data.result?.kg_min_similarity as number | undefined) ?? 0.35;
      qc.invalidateQueries({ queryKey: qk.rewards(slug) });
      const kgBlurb =
        kgMatches.length > 0
          ? `KG: ${kgMatches.length} match${kgMatches.length === 1 ? "" : "es"} above sim=${minSim.toFixed(2)} — top: ${kgMatches
              .slice(0, 2)
              .map((m) => m.technique)
              .join(", ")}${kgMatches.length > 2 ? "…" : ""}`
          : `KG: no matches above sim=${minSim.toFixed(2)} — Claude grounded in physics only`;
      toast.success(
        newVersion != null
          ? `Claude wrote v${newVersion}.py`
          : "Claude wrote a new reward version",
        { description: kgBlurb },
      );
      setActiveJobId(null);
      setPrompt("");
    } else if (job.data.status === "errored") {
      toast.error("Reward rewrite failed", {
        description: job.data.error ?? "see /jobs for details",
      });
      setActiveJobId(null);
    }
  }, [job.data, activeJobId, slug, qc]);

  const inFlight = Boolean(activeJobId) && job.data?.status !== "completed"
    && job.data?.status !== "errored";

  const onSubmit = () => {
    if (disabled || latestVersion == null) return;
    const trimmed = prompt.trim();
    if (trimmed.length < 3) {
      toast.error("Prompt too short", {
        description: "Give Claude at least a sentence to work with.",
      });
      return;
    }
    mutate.mutate(
      { prompt: trimmed, expected_parent_version: latestVersion },
      {
        onSuccess: (j) => {
          setActiveJobId(j.job_id);
          toast.success("Claude is rewriting…", {
            description: `Forking from v${latestVersion}.py — ~30-60 s.`,
          });
        },
        onError: (err) => {
          const msg =
            err instanceof ApiError
              ? err.problem.detail ?? err.problem.title
              : (err as Error).message;
          toast.error("Could not start prompt edit", { description: msg });
        },
      },
    );
  };

  return (
    <Card className="border-primary/20 bg-primary/5">
      <CardContent className="flex flex-col gap-3 p-4">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Sparkles className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold">
              Iterate this reward with Claude
            </div>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Describe the behavior in plain English and Claude rewrites
              the reward directly — no training loop required. For the
              full sculpt loop (train → diagnose → edit → commit), use
              the <strong>New run</strong> button at the top right.
            </p>
          </div>
        </div>

        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={
            latestVersion == null
              ? "No reward file yet — scaffold the project first."
              : "e.g. 'reward the robot for jumping 30 cm with low torque and minimal lateral drift'"
          }
          rows={3}
          maxLength={2000}
          disabled={disabled || inFlight || latestVersion == null}
          className="text-xs"
        />

        <div className="flex items-center justify-between gap-2">
          <div className="text-[10px] text-muted-foreground">
            {latestVersion != null && (
              <>
                Forks from <code className="font-mono">v{latestVersion}.py</code>
                {" "}→{" "}
                <code className="font-mono">v{latestVersion + 1}.py</code>
              </>
            )}
          </div>
          <Button
            size="sm"
            onClick={onSubmit}
            disabled={disabled || inFlight || latestVersion == null}
          >
            {inFlight ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Wand2 className="h-3.5 w-3.5" />
            )}
            {inFlight ? "Claude is writing…" : "Prompt Claude"}
          </Button>
        </div>

        {(inFlight || events.events.length > 0) && (
          <PromptActivityPanel
            events={events.events}
            connected={events.connected}
            inFlight={inFlight}
          />
        )}

        {disabled && (
          <p className="text-xs text-amber-700 dark:text-amber-300">
            {adapterUnavailable
              ? "Training is disabled because this project uses a coming-soon adapter. See the adoption guide to enable it."
              : "Training is disabled for this project."}
          </p>
        )}
      </CardContent>
    </Card>
  );
}


// ── Activity panel (bug #6.1 / T2) ───────────────────────────────────
// Renders the tail of log_line events streamed over /ws/jobs/{id}/events
// while a reward-prompt-edit job is in flight. Closed-state shows the
// last 20; open-state (details) shows up to 100.
function PromptActivityPanel({
  events,
  connected,
  inFlight,
}: {
  events: { text?: string; ts?: string; type: string }[];
  connected: boolean;
  inFlight: boolean;
}) {
  const logs = events.filter((e) => e.type === "log_line");
  const tail = logs.slice(-20);
  return (
    <div className="rounded-md border border-border/60 bg-background/60 p-2 text-[11px]">
      <div className="mb-1 flex items-center justify-between">
        <div className="flex items-center gap-1.5 font-medium text-muted-foreground">
          {inFlight ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <span
              className={cn(
                "inline-block h-1.5 w-1.5 rounded-full",
                connected ? "bg-emerald-500" : "bg-muted-foreground/40",
              )}
            />
          )}
          <span>Activity</span>
          <span className="text-muted-foreground/70">
            ({logs.length} event{logs.length === 1 ? "" : "s"})
          </span>
        </div>
        {!connected && inFlight && (
          <span className="text-muted-foreground/70">reconnecting…</span>
        )}
      </div>
      {tail.length === 0 ? (
        <div className="italic text-muted-foreground">
          {inFlight ? "waiting for first event…" : "no events"}
        </div>
      ) : (
        <ul className="max-h-40 space-y-0.5 overflow-auto font-mono text-[10.5px] leading-tight">
          {tail.map((ev, i) => (
            <li
              key={`${ev.ts ?? i}-${i}`}
              className="flex items-baseline gap-2 text-muted-foreground"
            >
              {ev.ts && (
                <span className="shrink-0 text-muted-foreground/60">
                  {new Date(ev.ts).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
              )}
              <span className="truncate">{ev.text ?? ""}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ── Version list row ──────────────────────────────────────────────────
function VersionRow({
  v,
  selected,
  onSelect,
}: {
  v: RewardVersionSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "flex w-full flex-col gap-1 rounded-md border px-2 py-2 text-left text-xs transition-colors",
        selected
          ? "border-foreground/30 bg-accent"
          : "border-transparent hover:bg-accent/60",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono font-semibold">v{v.version}</span>
        <AuthorBadge author={v.author} />
      </div>
      {v.parent_version !== null && (
        <span className="font-mono text-[10px] text-muted-foreground">
          ← v{v.parent_version}
        </span>
      )}
      {v.primary_metric !== null && (
        <MetricRow metric={v.primary_metric} delta={v.metric_delta} />
      )}
      <span className="truncate text-[10px] text-muted-foreground">
        {formatRelative(v.created_at)}
      </span>
    </button>
  );
}

function AuthorBadge({ author }: { author: RewardVersionSummary["author"] }) {
  const config = {
    human: {
      icon: User2,
      cls: "bg-blue-50 text-blue-700 border-blue-200",
    },
    sculptor: {
      icon: Sparkles,
      cls: "bg-violet-50 text-violet-700 border-violet-200",
    },
    unknown: {
      icon: User2,
      cls: "bg-muted text-muted-foreground border-border",
    },
  }[author];
  const Icon = config.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1 py-0.5 text-[9px] font-medium uppercase tracking-wide",
        config.cls,
      )}
    >
      <Icon className="h-2.5 w-2.5" />
      {author}
    </span>
  );
}

function MetricRow({ metric, delta }: { metric: number; delta: number | null }) {
  const deltaClass =
    delta == null
      ? "text-muted-foreground"
      : delta > 0
      ? "text-emerald-700"
      : delta < 0
      ? "text-rose-700"
      : "text-muted-foreground";
  return (
    <span className="flex items-baseline gap-1.5 font-mono text-[11px]">
      <span className="text-foreground">{metric.toFixed(3)}</span>
      {delta !== null && (
        <span className={deltaClass}>
          {delta >= 0 ? "+" : ""}
          {delta.toFixed(3)}
        </span>
      )}
    </span>
  );
}

// ── Lock banner ───────────────────────────────────────────────────────
function LockBanner({ note }: { note: string }) {
  return (
    <div className="flex items-center gap-2 rounded-md border border-amber-300/60 bg-amber-50 px-3 py-2 text-sm text-amber-900">
      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-500" />
      <span>{note}</span>
    </div>
  );
}

// ── Read-only pane ────────────────────────────────────────────────────
function ReadOnlyPane({
  slug,
  detail,
  canEdit,
  stageScope,
  onNewHumanEdit,
}: {
  slug: string;
  detail: RewardVersionDetail;
  canEdit: boolean;
  /** §Ship 21d: "<missionSlug>/<stageName>" when viewing a mission
   *  stage's rewards, null for project rewards. Threaded into the
   *  "Why this edit?" panel so its diagnosis fetch hits the stage's
   *  runs dir. */
  stageScope: string | null;
  onNewHumanEdit: () => void;
}) {
  const [diffMode, setDiffMode] = useState(false);
  const parentVersion = detail.version > 0 ? detail.version - 1 : null;
  // §Ship 21d: the diff-vs-parent view must fetch the parent from the
  // SAME scope, else a stage version diffs against the project's
  // v(N-1) (wrong file / 404).
  const parent = useReward(
    diffMode && parentVersion !== null ? slug : undefined,
    parentVersion ?? undefined,
    stageScope,
  );

  return (
    <>
      <ContractPreamble />
      <Card className="overflow-hidden">
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 border-b py-3">
          <div>
            <CardTitle className="text-sm">
              <code className="font-mono">{detail.file_name}</code>
              {diffMode && parentVersion !== null && (
                <span className="ml-2 text-[10px] font-normal text-muted-foreground">
                  vs <code className="font-mono">v{parentVersion}.py</code>
                </span>
              )}
            </CardTitle>
            <CardDescription className="text-[11px]">
              author: {detail.author} · parent_hash: {detail.parent_hash.slice(0, 12) || "—"}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {parentVersion !== null && (
              <Button
                size="sm"
                variant={diffMode ? "default" : "outline"}
                onClick={() => setDiffMode((v) => !v)}
                title={
                  diffMode
                    ? "Switch back to the full reward source"
                    : `Diff v${detail.version}.py vs v${parentVersion}.py`
                }
              >
                <GitCompare />
                {diffMode ? "Hide diff" : "Diff vs parent"}
              </Button>
            )}
            {detail.version === 0 && (
              <RegenerateTemplateButton slug={slug} disabled={!canEdit} />
            )}
            <Button
              size="sm"
              onClick={onNewHumanEdit}
              disabled={!canEdit}
              title={
                canEdit
                  ? "Clone this version into an editable draft"
                  : "Disabled while a sculpt run is active"
              }
            >
              <FilePlus2 />
              New human edit
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {diffMode && parentVersion !== null ? (
            parent.isLoading ? (
              <div className="flex h-[420px] items-center justify-center text-xs text-muted-foreground">
                <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                Loading v{parentVersion}.py for diff…
              </div>
            ) : parent.data ? (
              <MonacoDiffLazy
                original={parent.data.source}
                modified={detail.source}
                height={420}
              />
            ) : (
              <div className="flex h-[420px] items-center justify-center text-xs text-rose-700">
                Could not load parent version.
              </div>
            )
          ) : (
            <MonacoLazy value={detail.source} readOnly height={420} />
          )}
        </CardContent>
      </Card>
      <SpecPanel detail={detail} />
      {detail.author === "sculptor" && detail.version > 0 && (
        <WhyThisEditPanel
          slug={slug}
          version={detail.version}
          stageScope={stageScope}
        />
      )}
    </>
  );
}

// ── "Why this edit?" collapsible ──────────────────────────────────────
function WhyThisEditPanel({
  slug,
  version,
  stageScope,
}: {
  slug: string;
  version: number;
  /** §Ship 21d: stage scope so the diagnosis fetch resolves to the
   *  stage's runs dir for mission stage rewards. */
  stageScope?: string | null;
}) {
  const [open, setOpen] = useState(false);
  const diag = useRewardDiagnosis(slug, version, {
    enabled: open,
    stage: stageScope ?? null,
  });

  return (
    <Card>
      <CardHeader className="py-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center gap-2 text-left text-sm font-medium"
        >
          {open ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
          Why this edit?
          <span className="text-[10px] font-normal text-muted-foreground">
            (diagnosis.json from iter_{version - 1}/)
          </span>
        </button>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3 pt-0 text-xs">
          {diag.isLoading && (
            <p className="text-muted-foreground">Loading diagnosis…</p>
          )}
          {diag.error && (
            <p className="text-muted-foreground">
              No diagnosis recorded — the version may be human-authored
              or the run never reached the diagnose stage.
            </p>
          )}
          {diag.data && (
            <>
              {diag.data.failure_modes && diag.data.failure_modes.length > 0 && (
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                    Failure modes
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {diag.data.failure_modes.map((fm) => (
                      <span
                        key={fm}
                        className="inline-flex items-center rounded-sm border border-rose-200 bg-rose-50 px-1.5 py-0.5 font-mono text-[10px] text-rose-800"
                      >
                        {fm}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {diag.data.evidence && (
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                    Evidence
                  </div>
                  <p className="whitespace-pre-wrap leading-snug">
                    {diag.data.evidence}
                  </p>
                </div>
              )}
              {diag.data.proposed_edits && diag.data.proposed_edits.length > 0 && (
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                    Proposed edits
                  </div>
                  <ul className="space-y-1">
                    {diag.data.proposed_edits.map((e, i) => (
                      <li
                        key={i}
                        className="rounded-md border border-border/60 bg-muted/20 p-2"
                      >
                        <div className="flex items-baseline gap-1.5 font-mono text-[11px]">
                          <span className="rounded-sm bg-violet-100 px-1 text-[9px] uppercase tracking-wide text-violet-800">
                            {e.operation}
                          </span>
                          <span className="text-foreground">{e.target_term}</span>
                          {e.suggested_value && (
                            <>
                              <span className="text-muted-foreground">→</span>
                              <code className="text-muted-foreground">
                                {e.suggested_value}
                              </code>
                            </>
                          )}
                        </div>
                        <p className="mt-1 leading-snug">{e.rationale}</p>
                        {e.paper_refs && e.paper_refs.length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {e.paper_refs.map((aid) => (
                              <a
                                key={aid}
                                href={`https://arxiv.org/abs/${aid}`}
                                target="_blank"
                                rel="noreferrer noopener"
                                className="inline-flex items-center gap-0.5 rounded-sm border border-border bg-background px-1 font-mono text-[10px] text-foreground hover:underline"
                              >
                                {aid}
                                <ExternalLink className="h-2.5 w-2.5" />
                              </a>
                            ))}
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {typeof diag.data.confidence === "number" && (
                <div className="text-[10px] text-muted-foreground">
                  confidence: {diag.data.confidence.toFixed(2)}
                </div>
              )}
            </>
          )}
        </CardContent>
      )}
    </Card>
  );
}

// ── Regenerate-template button + confirmation dialog ──────────────────
function RegenerateTemplateButton({
  slug,
  disabled,
}: {
  slug: string;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const regen = useRegenerateRewardTemplate(slug);

  const onConfirm = () => {
    regen.mutate(undefined, {
      onSuccess: () => {
        setOpen(false);
        toast.success("Reward template regenerated", {
          description:
            "rewards/v0.py rewritten using the adapter's scaffold template.",
        });
      },
      onError: (err) => {
        toast.error("Could not regenerate template", {
          description: err.message,
        });
      },
    });
  };

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
        disabled={disabled}
        title={
          disabled
            ? "Disabled while a sculpt run is active"
            : "Rewrite v0.py using the scaffold template for this project's adapter"
        }
      >
        <RefreshCw />
        Regenerate template
      </Button>
      <Dialog
        open={open}
        onOpenChange={(o) => !regen.isPending && setOpen(o)}
      >
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Regenerate reward template?</DialogTitle>
            <DialogDescription asChild>
              <div className="space-y-2 text-sm">
                <p>
                  This overwrites{" "}
                  <code className="font-mono">rewards/v0.py</code> with the
                  scaffold template matching this project's adapter —{" "}
                  <code>compute_reward</code> for gym_sb3, or{" "}
                  <code>compute_reward_batched</code> for mjlab.
                </p>
                <p className="text-amber-700 dark:text-amber-400">
                  Any manual edits to v0.py will be lost. Git history
                  preserves the previous file.
                </p>
                <p>
                  Use this to fix a project that fails at training time with{" "}
                  <code className="font-mono">
                    missing compute_reward_batched
                  </code>
                  .
                </p>
              </div>
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setOpen(false)}
              disabled={regen.isPending}
            >
              Cancel
            </Button>
            <Button onClick={onConfirm} disabled={regen.isPending}>
              <RefreshCw />
              {regen.isPending ? "Regenerating…" : "Regenerate"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

// ── Editor pane (draft) ───────────────────────────────────────────────
function EditorPane({
  parentDetail,
  draftParentVersion,
  initialSource,
  locked,
  onDiscard,
  onSaved,
  onConcurrencyConflict,
  slug,
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
      {
        source,
        expected_parent_version: draftParentVersion,
        note: note.trim(),
      },
      {
        onSuccess: (detail) => onSaved(detail.version),
        onError: (err) => {
          if (err instanceof ApiError) {
            if (err.status === 409 && err.type === "/problems/concurrency-conflict") {
              const current = (err.problem.current as number | undefined) ?? draftParentVersion;
              onConcurrencyConflict(current);
              return;
            }
            if (err.status === 422 && err.type === "/problems/reward-validation") {
              const vs = (err.problem.violations as string[] | undefined) ?? [
                err.problem.detail ?? err.message,
              ];
              toast.error("Reward validation failed", {
                description: vs.join(" · "),
              });
              return;
            }
          }
          toast.error("Could not save reward", { description: err.message });
        },
      },
    );
  };

  const violations = useMemo(() => {
    if (
      save.error instanceof ApiError &&
      save.error.type === "/problems/reward-validation"
    ) {
      return (save.error.problem.violations as string[] | undefined) ?? [];
    }
    return [];
  }, [save.error]);

  return (
    <>
      <ContractPreamble draft />
      <Card className="overflow-hidden">
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 border-b py-3">
          <div>
            <CardTitle className="text-sm">
              Editing from <code className="font-mono">v{draftParentVersion}.py</code>
            </CardTitle>
            <CardDescription className="text-[11px]">
              will save as v{draftParentVersion + 1}.py · author must be{" "}
              <code>"human"</code>
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={onDiscard}>
              Discard
            </Button>
            <Button size="sm" onClick={onSave} disabled={locked || save.isPending}>
              <Save />
              {save.isPending ? "Saving…" : "Save"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <MonacoLazy
            value={source}
            onChange={(v) => setSource(v ?? "")}
            readOnly={locked}
            height={480}
          />
        </CardContent>
        <div className="border-t p-3">
          <label className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Note
          </label>
          <input
            type="text"
            className="h-8 w-full rounded-md border border-input bg-transparent px-2 text-xs"
            placeholder="Short description (goes into CHANGELOG eventually)"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            maxLength={500}
            disabled={locked}
          />
          {violations.length > 0 && (
            <div className="mt-3 space-y-1 rounded-md border border-destructive/30 bg-destructive/5 p-3 text-xs">
              <div className="font-semibold text-destructive">
                Validation blocked the save
              </div>
              <ul className="ml-4 list-disc text-muted-foreground">
                {violations.map((v, i) => (
                  <li key={i} className="font-mono break-words">{v}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </Card>
      <SpecPanel detail={parentDetail} label="Parent REWARD_SPEC" />
    </>
  );
}

// ── Contract preamble ─────────────────────────────────────────────────
function ContractPreamble({ draft }: { draft?: boolean } = {}) {
  return (
    <Card className="border-dashed">
      <CardContent className="space-y-2 p-3 text-xs">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted-foreground">
          Contract
          {draft && (
            <span className="rounded-sm border border-amber-300/60 bg-amber-50 px-1 py-0.5 text-[9px] font-medium uppercase text-amber-900">
              locked preamble
            </span>
          )}
        </div>
        <pre className="overflow-x-auto rounded border bg-muted/30 p-2 font-mono text-[11px] leading-relaxed">
{`def compute_reward(state, action, next_state, info):
    """MUST have this signature — do not rename args."""
    ...
    return reward, components   # components: dict[str, float]`}
        </pre>
        <p className="text-muted-foreground">
          Server-side rules: exact signature above, <code>REWARD_SPEC</code> dict
          literal with <code>version/parent_hash/author/description/
          hyperparameters/references</code>, <code>author == "human"</code>,
          parent_hash matches sha256 of the parent file, every reference
          arxiv_id is in this project's KG.
        </p>
      </CardContent>
    </Card>
  );
}

// ── REWARD_SPEC panel ─────────────────────────────────────────────────
function SpecPanel({
  detail,
  label = "REWARD_SPEC",
}: {
  detail: RewardVersionDetail;
  label?: string;
}) {
  const spec = detail.spec;
  const hparams = Object.entries(spec.hyperparameters);
  // Grounding dict maps hparam_name → citation-or-justification per
  // the 2026-04-22 05:30 KG-grounding mandate. Pre-mandate rewards
  // have `grounding = {}`; newer ones have one entry per hparam.
  const grounding = Object.entries(spec.grounding ?? {});
  const arxivRe = /^(?:\d{4}\.\d{4,5}|[a-z-]+\/\d{7})/i;
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm">{label}</CardTitle>
        {spec.description && (
          <CardDescription>{spec.description}</CardDescription>
        )}
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-4 text-xs md:grid-cols-4">
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            Hyperparameters
          </div>
          {hparams.length === 0 ? (
            <p className="italic text-muted-foreground">(none)</p>
          ) : (
            <ul className="space-y-0.5 font-mono text-[11px]">
              {hparams.map(([k, v]) => (
                <li key={k} className="flex justify-between gap-2">
                  <span className="truncate">{k}</span>
                  <span className="text-muted-foreground">{v}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            Grounding
          </div>
          {grounding.length === 0 ? (
            <p className="italic text-muted-foreground">
              (pre-grounding-mandate)
            </p>
          ) : (
            <ul className="space-y-1 font-mono text-[11px]">
              {grounding.map(([k, v]) => {
                const match = v.match(arxivRe);
                const arxivId = match ? match[0] : null;
                return (
                  <li key={k} className="flex flex-col gap-0.5">
                    <span className="truncate font-medium">{k}</span>
                    {arxivId ? (
                      <a
                        href={`https://arxiv.org/abs/${arxivId}`}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="inline-flex items-center gap-1 truncate text-[10px] text-foreground underline-offset-2 hover:underline"
                        title={v}
                      >
                        <span className="truncate">{v}</span>
                        <ExternalLink className="h-3 w-3 shrink-0" />
                      </a>
                    ) : (
                      <span
                        className="truncate text-[10px] text-muted-foreground"
                        title={v}
                      >
                        {v}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            References
          </div>
          {spec.references.length === 0 ? (
            <p className="italic text-muted-foreground">(novel)</p>
          ) : (
            <ul className="space-y-1">
              {spec.references.map((r) => (
                <li key={r.arxiv_id} className="flex items-baseline gap-1.5">
                  <a
                    href={`https://arxiv.org/abs/${r.arxiv_id}`}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1 font-mono text-[11px] text-foreground underline-offset-2 hover:underline"
                  >
                    {r.arxiv_id}
                    <ExternalLink className="h-3 w-3" />
                  </a>
                  {r.citation && (
                    <span className="truncate text-[10px] text-muted-foreground">
                      {r.citation}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            Probe
          </div>
          {detail.components_probe.ok ? (
            <>
              <div className="mb-1 font-mono text-[11px]">
                total ={" "}
                <span className="text-muted-foreground">
                  {detail.components_probe.total?.toFixed(4) ?? "—"}
                </span>
              </div>
              <ul className="space-y-0.5 font-mono text-[11px]">
                {Object.entries(detail.components_probe.components ?? {}).map(
                  ([k, v]) => (
                    <li key={k} className="flex justify-between gap-2">
                      <span className="truncate">{k}</span>
                      <span className="text-muted-foreground">{v}</span>
                    </li>
                  ),
                )}
              </ul>
            </>
          ) : (
            <p className="text-rose-700">
              {detail.components_probe.error ?? "probe failed"}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

import { useState } from "react";
import { Loader2, Play, Sparkles, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MissionDetailDialog, MissionLifecycleBadge } from "@/components/MissionDetailDialog";
import { NewMissionDialog } from "@/components/NewMissionDialog";
import {
  useDeleteMission,
  useMissions,
} from "@/hooks/useMissions";
import { ApiError } from "@/lib/api";
import { cn, formatRelative } from "@/lib/utils";
import type { MissionJobKind, MissionSummary } from "@/lib/types";

// §Ship 20 Goal #4: replace internal `MISSION_EXECUTE` / `MISSION_
// DECOMPOSE` enum chips with English labels. The CLI/backend keep the
// internal kind names; this is purely the user-facing string.
function missionJobKindLabel(kind: MissionJobKind | null): string {
  if (kind === "mission_decompose") return "Planning";
  if (kind === "mission_execute") return "Training";
  return "active";
}

// §Ship 20 Goal #4: lifecycle-aware progress label so `0/3 stages`
// (which reads as a test failure) becomes phrasing that fits the
// state. Kept in sync with RunsTab.tsx's `missionRunStateLabel` so
// the same mission reads the same way across tabs.
function stageProgressLabel(m: MissionSummary): string {
  const { current_stage_idx: i, n_stages: n, lifecycle } = m;
  if (n === 0) return "Planning…";
  if (lifecycle === "ready" && i === 0) return `${n} stages planned`;
  if (lifecycle === "completed") return `${n} of ${n} stages complete`;
  if (lifecycle === "running") return `Stage ${i + 1} of ${n}`;
  return `${i} of ${n} stages complete`;
}

export function MissionsTab({ slug }: { slug: string }) {
  const list = useMissions(slug);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);

  const missions = list.data ?? [];
  const selected =
    selectedSlug != null
      ? (missions.find((m) => m.mission_slug === selectedSlug) ?? null)
      : null;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 py-3">
          <div>
            <CardTitle className="text-sm">Missions</CardTitle>
            <CardDescription className="text-[11px]">
              Define a goal. Claude breaks it into stages and trains
              them in order, warm-starting each from the previous.
            </CardDescription>
          </div>
          <NewMissionDialog
            slug={slug}
            onCreated={(missionSlug) => {
              // §Ship-19c loading visual: jump straight into the
              // detail dialog so the user watches the live decompose
              // event stream instead of staring at a row badge that
              // pulses for 30-90s with no other feedback.
              setSelectedSlug(missionSlug);
            }}
          />
        </CardHeader>
      </Card>

      {list.isLoading && (
        <p className="text-sm text-muted-foreground">Loading missions…</p>
      )}
      {list.error && (
        <p className="rounded border border-destructive/40 bg-destructive/5 p-2 font-mono text-[11px] text-destructive">
          {(list.error as Error).message}
        </p>
      )}
      {!list.isLoading && missions.length === 0 && (
        <Card className="p-8 text-center">
          <Sparkles className="mx-auto h-8 w-8 text-muted-foreground" />
          <p className="mt-2 text-sm">No missions yet.</p>
          <p className="text-xs text-muted-foreground">
            Click <strong>New mission</strong> above to decompose a goal.
            The decompose job typically runs ~30–90 s.
          </p>
        </Card>
      )}

      {missions.length > 0 && (
        <Card className="overflow-hidden p-0">
          <ul className="divide-y">
            {missions.map((m) => (
              <MissionRow
                key={m.mission_slug}
                slug={slug}
                mission={m}
                onOpen={() => setSelectedSlug(m.mission_slug)}
              />
            ))}
          </ul>
        </Card>
      )}

      <MissionDetailDialog
        slug={slug}
        missionSlug={selected?.mission_slug ?? null}
        summary={selected}
        open={!!selected}
        onOpenChange={(open) => {
          if (!open) setSelectedSlug(null);
        }}
      />
    </div>
  );
}

function MissionRow({
  slug,
  mission,
  onOpen,
}: {
  slug: string;
  mission: MissionSummary;
  onOpen: () => void;
}) {
  // §Ship-19d: per-row Run button now opens the detail dialog (where
  // the new RunMissionDialog can be configured), instead of firing a
  // mutation with default config. Sam's feedback: defaults run 5000-
  // step iters silently, blowing wall-clock when testing.
  const del = useDeleteMission(slug);
  const isErrored = mission.lifecycle === "errored";
  const hasActiveJob = mission.active_job_id != null;
  const canRun = !isErrored && mission.lifecycle === "ready" && !hasActiveJob;
  const canDelete = !hasActiveJob;

  return (
    <li
      className={cn(
        "flex flex-col gap-2 px-4 py-3 text-sm transition-colors hover:bg-accent/40 sm:flex-row sm:items-center",
      )}
    >
      <button
        type="button"
        onClick={onOpen}
        className="flex min-w-0 flex-1 flex-col gap-1 text-left"
      >
        <div className="flex flex-wrap items-center gap-2">
          <MissionLifecycleBadge lifecycle={mission.lifecycle} />
          {hasActiveJob && (
            <span className="rounded bg-amber-50 px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-amber-700 ring-1 ring-amber-200">
              {missionJobKindLabel(mission.active_job_kind)}
            </span>
          )}
        </div>
        <div
          className="line-clamp-2 text-xs"
          title={mission.goal}
        >
          {mission.goal}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10.5px] text-muted-foreground">
          <span>{stageProgressLabel(mission)}</span>
          <span>{formatRelative(mission.created_at)}</span>
          <code
            className="truncate font-mono text-[10px]"
            title={`mission slug: ${mission.mission_slug}`}
          >
            {mission.mission_slug}
          </code>
        </div>
      </button>

      <div className="flex shrink-0 items-center gap-2 self-start sm:self-center">
        {!isErrored && (
          <Button
            variant="outline"
            size="sm"
            disabled={!canRun}
            title={
              isErrored
                ? "Mission is errored; only Delete is available."
                : mission.lifecycle !== "ready"
                  ? `Run is only allowed once decomposition finishes (currently ${mission.lifecycle}).`
                  : hasActiveJob
                    ? "Already training. Wait for the run to finish."
                    : "Open mission to configure and launch a run."
            }
            onClick={onOpen}
          >
            <Play className="h-3.5 w-3.5" />
            Run
          </Button>
        )}
        <Button
          variant="outline"
          size="sm"
          disabled={!canDelete || del.isPending}
          title={
            hasActiveJob
              ? "Already training. Wait for the run to finish."
              : undefined
          }
          onClick={() => {
            const ok = window.confirm(
              `Delete mission ${mission.mission_slug}? On-disk artifacts will be removed.`,
            );
            if (!ok) return;
            del.mutate(mission.mission_slug, {
              onSuccess: (r) => {
                toast.success("Mission deleted", {
                  description: `freed ${(r.freed_bytes / 1_048_576).toFixed(1)} MiB`,
                });
              },
              onError: (err) => {
                const detail =
                  err instanceof ApiError
                    ? err.problem.detail ?? err.problem.title
                    : err.message;
                toast.error("Could not delete mission", {
                  description: detail,
                });
              },
            });
          }}
        >
          {del.isPending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Trash2 className="h-3.5 w-3.5" />
          )}
          Delete
        </Button>
      </div>
    </li>
  );
}

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
  useRunMission,
} from "@/hooks/useMissions";
import { ApiError } from "@/lib/api";
import { cn, formatRelative } from "@/lib/utils";
import type { MissionSummary } from "@/lib/types";

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
              Claude decomposes a goal into a curriculum of stages, then
              chains warm-started <code>sculpt run</code> jobs across
              them. Mission state persists in{" "}
              <code>.missions/&lt;slug&gt;/mission.json</code>.
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
  const run = useRunMission(slug);
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
          <code className="truncate font-mono text-[11px] text-muted-foreground">
            {mission.mission_slug}
          </code>
          {hasActiveJob && (
            <span className="rounded bg-amber-50 px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-amber-700 ring-1 ring-amber-200">
              {mission.active_job_kind ?? "active"}
            </span>
          )}
        </div>
        <div
          className="line-clamp-2 text-xs"
          title={isErrored ? mission.goal : mission.goal}
        >
          {mission.goal}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10.5px] text-muted-foreground">
          <span>
            {mission.current_stage_idx}/{mission.n_stages} stages
          </span>
          <span>{formatRelative(mission.created_at)}</span>
          <span className="font-mono">{mission.decomposition_model}</span>
        </div>
      </button>

      <div className="flex shrink-0 items-center gap-2 self-start sm:self-center">
        {!isErrored && (
          <Button
            variant="outline"
            size="sm"
            disabled={!canRun || run.isPending}
            title={
              isErrored
                ? "Mission is errored; only Delete is available."
                : mission.lifecycle !== "ready"
                  ? `Lifecycle is ${mission.lifecycle}; run is only allowed when ready.`
                  : hasActiveJob
                    ? "An active job is already running."
                    : undefined
            }
            onClick={() =>
              run.mutate(mission.mission_slug, {
                onSuccess: () => {
                  toast.success("Mission run queued");
                  onOpen();
                },
                onError: (err) => {
                  const detail =
                    err instanceof ApiError
                      ? err.problem.detail ?? err.problem.title
                      : err.message;
                  toast.error("Could not run mission", {
                    description: detail,
                  });
                },
              })
            }
          >
            {run.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            Run
          </Button>
        )}
        <Button
          variant="outline"
          size="sm"
          disabled={!canDelete || del.isPending}
          title={
            hasActiveJob
              ? "An active job is running. Wait for it to finish."
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

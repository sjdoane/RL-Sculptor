import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { controlRun, getRun, killRun, launchRun, listRuns } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  RunControlState,
  RunDetail,
  RunParamsPayload,
  RunSummary,
} from "@/lib/types";

const RUN_POLL_MS = 3000;

export function useRuns(
  slug: string | undefined,
  /** §Ship 21d: when `keepPolling` is true, keep refetching `/runs`
   *  even when no run currently shows `running`/`queued`. Callers
   *  pass this while a MISSION is active so the list stays fresh
   *  through stage boundaries — between a stage completing and the
   *  next stage's child job registering, there's no "running" run,
   *  which previously made the interval return false and freeze.
   *  Once React Query clears the interval it does NOT auto-resume,
   *  so `activeStageRun` would go permanently stale for the rest of
   *  the mission (rewards/live-video stop updating). */
  opts?: { keepPolling?: boolean },
) {
  const keepPolling = opts?.keepPolling ?? false;
  return useQuery<RunSummary[]>({
    queryKey: slug ? qk.runs(slug) : ["runs", "_none"],
    queryFn: () => listRuns(slug!),
    enabled: !!slug,
    refetchInterval: (q) => {
      const data = q.state.data as RunSummary[] | undefined;
      if (!data) return RUN_POLL_MS;
      const hasActive = data.some(
        (r) => r.status === "running" || r.status === "queued",
      );
      if (hasActive || keepPolling) return RUN_POLL_MS;
      return false;
    },
  });
}

export function useRun(
  slug: string | undefined,
  runId: string | undefined,
) {
  return useQuery<RunDetail>({
    queryKey:
      slug && runId ? qk.run(slug, runId) : ["run", "_none"],
    queryFn: () => getRun(slug!, runId!),
    enabled: !!slug && !!runId,
  });
}

export function useLaunchRun(slug: string) {
  const qc = useQueryClient();
  return useMutation<RunSummary, Error, RunParamsPayload>({
    mutationFn: (payload) => launchRun(slug, payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.runs(slug) });
      qc.invalidateQueries({ queryKey: qk.project(slug) });
    },
  });
}

export function useKillRun(slug: string) {
  const qc = useQueryClient();
  return useMutation<RunSummary, Error, string>({
    mutationFn: (runId) => killRun(slug, runId),
    onSuccess: (_r, runId) => {
      qc.invalidateQueries({ queryKey: qk.runs(slug) });
      qc.invalidateQueries({ queryKey: qk.run(slug, runId) });
    },
  });
}

/** §Ship 39 (H1): interactive control for a live run — flip Auto/Manual,
 *  resume a pause (optionally with feedback), or stop cleanly. */
export interface RunControlVars {
  runId: string;
  mode?: "manual" | "auto";
  resume?: boolean;
  feedback?: string | null;
  stop?: boolean;
}

export function useControlRun(slug: string) {
  const qc = useQueryClient();
  return useMutation<RunControlState, Error, RunControlVars>({
    mutationFn: ({ runId, ...body }) => controlRun(slug, runId, body),
    onSuccess: (_r, { runId }) => {
      qc.invalidateQueries({ queryKey: qk.run(slug, runId) });
    },
  });
}

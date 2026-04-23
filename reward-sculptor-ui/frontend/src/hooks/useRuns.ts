import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getRun, killRun, launchRun, listRuns } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  RunDetail,
  RunParamsPayload,
  RunSummary,
} from "@/lib/types";

const RUN_POLL_MS = 3000;

export function useRuns(slug: string | undefined) {
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
      return hasActive ? RUN_POLL_MS : false;
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

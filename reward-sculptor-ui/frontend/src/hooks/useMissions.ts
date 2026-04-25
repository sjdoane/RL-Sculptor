import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  createMission,
  deleteMission,
  getMission,
  listMissions,
  runMission,
  type RunMissionRequestBody,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  CreateMissionRequest,
  DeleteMissionResponse,
  JobSummary,
  MissionDetail,
  MissionJobKind,
  MissionLifecycleStatus,
  MissionSummary,
} from "@/lib/types";

const MISSION_LIST_POLL_MS = 5000;

/** Poll the list while ANY mission has an active job (decompose or
 *  execute). Mirrors useRuns. Note: a decomposing mission has
 *  `lifecycle="ready"` until the job finishes — so we gate on
 *  `active_job_id`, not `lifecycle`. (Plan-audit CRITICAL #1.) */
export function useMissions(slug: string | undefined) {
  return useQuery<MissionSummary[]>({
    queryKey: slug ? qk.missions(slug) : ["missions", "_none"],
    queryFn: () => listMissions(slug!),
    enabled: !!slug,
    refetchInterval: (q) => {
      const data = q.state.data as MissionSummary[] | undefined;
      if (!data) return MISSION_LIST_POLL_MS;
      const hasActive = data.some((m) => m.active_job_id != null);
      return hasActive ? MISSION_LIST_POLL_MS : false;
    },
  });
}

export function useMission(
  slug: string | undefined,
  missionSlug: string | undefined,
  opts?: { enabled?: boolean },
) {
  const enabled = !!slug && !!missionSlug && (opts?.enabled ?? true);
  return useQuery<MissionDetail>({
    queryKey:
      slug && missionSlug
        ? qk.mission(slug, missionSlug)
        : ["mission", "_none"],
    queryFn: () => getMission(slug!, missionSlug!),
    enabled,
  });
}

export function useCreateMission(slug: string) {
  const qc = useQueryClient();
  return useMutation<JobSummary, Error, CreateMissionRequest>({
    mutationFn: (body) => createMission(slug, body),
    onSuccess: (job, body) => {
      // §Ship-19c: optimistic-insert the new mission row with the
      // decompose job_id already attached, so the auto-opened detail
      // dialog's WS hook sees `active_job_id` immediately on the next
      // render and subscribes to the live decompose stream — no
      // 5 s polling lag staring at a "no job" placeholder. The list
      // refetch follows up and reconciles any divergence.
      const missionSlug = (
        job as unknown as { params?: { mission_slug?: string } }
      ).params?.mission_slug;
      if (missionSlug) {
        qc.setQueryData<MissionSummary[] | undefined>(
          qk.missions(slug),
          (old) => {
            const placeholder: MissionSummary = {
              mission_slug: missionSlug,
              project_slug: slug,
              goal: body.goal,
              n_stages: 0,
              current_stage_idx: 0,
              decomposition_model: "",
              created_at: new Date().toISOString(),
              lifecycle: "running" as MissionLifecycleStatus,
              active_job_id: job.job_id,
              active_job_kind: "mission_decompose" as MissionJobKind,
            };
            if (!old) return [placeholder];
            // Don't double-insert if a refetch already landed it.
            if (old.some((m) => m.mission_slug === missionSlug)) return old;
            return [placeholder, ...old];
          },
        );
      }
      qc.invalidateQueries({ queryKey: qk.missions(slug) });
    },
  });
}

/** Optimistically set `active_job_id` + `active_job_kind` on the
 *  mission detail so the WS subscription opens immediately on the
 *  next render — without waiting for the next list/detail refetch.
 *  This addresses Ship 18a audit-deferred finding A: open the WS only
 *  when an active job is known to exist. The optimistic write makes
 *  that signal available the moment the mutation resolves.
 *
 *  Order matters: setQueryData runs synchronously and writes to the
 *  cache BEFORE invalidateQueries schedules a refetch. While the
 *  refetch is in flight, the cache returns the optimistic value, so
 *  the WS hook's `enabled` flag flips true immediately and stays
 *  true through the refetch (the backend has already persisted
 *  active_job_id by the time runMission's POST resolves). */
/** §Ship-19d: the `variables` parameter is now an object so the
 *  mutation can carry an optional body (iteration overrides + Goal
 *  A/B flags). `string` would also work via overload but a single
 *  shape keeps useMutation's TypeScript happy. */
export interface RunMissionVariables {
  missionSlug: string;
  body?: RunMissionRequestBody;
}

export function useRunMission(slug: string) {
  const qc = useQueryClient();
  return useMutation<JobSummary, Error, RunMissionVariables>({
    mutationFn: ({ missionSlug, body }) =>
      runMission(slug, missionSlug, body),
    onSuccess: (job, { missionSlug }) => {
      qc.setQueryData<MissionDetail | undefined>(
        qk.mission(slug, missionSlug),
        (old) =>
          old
            ? {
                ...old,
                active_job_id: job.job_id,
                active_job_kind: "mission_execute" as MissionJobKind,
                lifecycle: "running" as MissionLifecycleStatus,
              }
            : old,
      );
      qc.setQueryData<MissionSummary[] | undefined>(
        qk.missions(slug),
        (old) =>
          old
            ? old.map((m) =>
                m.mission_slug === missionSlug
                  ? {
                      ...m,
                      active_job_id: job.job_id,
                      active_job_kind:
                        "mission_execute" as MissionJobKind,
                      lifecycle: "running" as MissionLifecycleStatus,
                    }
                  : m,
              )
            : old,
      );
      qc.invalidateQueries({ queryKey: qk.missions(slug) });
      qc.invalidateQueries({ queryKey: qk.mission(slug, missionSlug) });
    },
  });
}

export function useDeleteMission(slug: string) {
  const qc = useQueryClient();
  return useMutation<DeleteMissionResponse, Error, string>({
    mutationFn: (missionSlug) => deleteMission(slug, missionSlug),
    onSuccess: (_r, missionSlug) => {
      qc.invalidateQueries({ queryKey: qk.missions(slug) });
      qc.removeQueries({ queryKey: qk.mission(slug, missionSlug) });
    },
  });
}

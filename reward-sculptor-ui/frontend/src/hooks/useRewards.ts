import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getReward,
  getRewardDiagnosis,
  listRewards,
  promptReward,
  putReward,
  regenerateRewardTemplate,
} from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type {
  JobSummary,
  ManualEditPayload,
  RewardDiagnosisPayload,
  RewardVersionDetail,
  RewardVersionSummary,
} from "@/lib/types";

export function useRewards(
  slug: string | undefined,
  /** §Ship 21: optional `<missionSlug>/<stageName>` to fetch that
   *  stage's reward versions instead of the project-global ones.
   *  When set, the cache key is namespaced so stage views don't
   *  clobber the project rewards cache. */
  stage?: string | null,
  /** §Ship 21b: refetch every N ms while a stage run is active so
   *  new vN.py files surface in the UI as Claude iterates them.
   *  Caller passes 3000 while the active mission_stage_run for this
   *  scope is in `running` state; null/undefined to disable. */
  refetchIntervalMs?: number | null,
) {
  const stageKey = stage ?? null;
  return useQuery<RewardVersionSummary[]>({
    queryKey: slug
      ? stageKey
        ? ["rewards", slug, "stage", stageKey]
        : qk.rewards(slug)
      : ["rewards", "_none"],
    queryFn: () => listRewards(slug!, stageKey ?? undefined),
    enabled: !!slug,
    refetchInterval:
      typeof refetchIntervalMs === "number" && refetchIntervalMs > 0
        ? refetchIntervalMs
        : false,
  });
}

export function useReward(
  slug: string | undefined,
  version: number | undefined,
  stage?: string | null,
  /** §Ship 21b: same polling rationale as useRewards — when an active
   *  stage run is producing v0/v1/v2 the *content* of the latest
   *  version may also change (e.g., REWARD_SPEC docstring updates as
   *  Claude iterates). Caller passes 3000 to keep the right-pane
   *  Monaco view fresh; null/undefined to disable. */
  refetchIntervalMs?: number | null,
) {
  const stageKey = stage ?? null;
  return useQuery<RewardVersionDetail>({
    queryKey:
      slug !== undefined && version !== undefined
        ? stageKey
          ? ["reward", slug, version, "stage", stageKey]
          : qk.reward(slug, version)
        : ["reward", "_none"],
    queryFn: () => getReward(slug!, version!, stageKey ?? undefined),
    enabled: slug !== undefined && version !== undefined,
    refetchInterval:
      typeof refetchIntervalMs === "number" && refetchIntervalMs > 0
        ? refetchIntervalMs
        : false,
  });
}

export function useSaveReward(slug: string, version: number) {
  const qc = useQueryClient();
  return useMutation<RewardVersionDetail, Error, ManualEditPayload>({
    mutationFn: (payload) => putReward(slug, version, payload),
    onSuccess: (detail) => {
      qc.invalidateQueries({ queryKey: qk.rewards(slug) });
      qc.setQueryData(qk.reward(slug, detail.version), detail);
      qc.invalidateQueries({ queryKey: qk.project(slug) });
    },
  });
}

/** Rewrite rewards/v0.py using the scaffold template for the project's
 * current adapter. Destructive — the UI should confirm before calling. */
export function useRegenerateRewardTemplate(slug: string) {
  const qc = useQueryClient();
  return useMutation<RewardVersionDetail, Error, void>({
    mutationFn: () => regenerateRewardTemplate(slug),
    onSuccess: (detail) => {
      qc.invalidateQueries({ queryKey: qk.rewards(slug) });
      qc.setQueryData(qk.reward(slug, detail.version), detail);
      qc.invalidateQueries({ queryKey: qk.project(slug) });
    },
  });
}

export interface RewardPromptPayload {
  prompt: string;
  expected_parent_version: number;
}

/** GET /projects/{slug}/rewards/{version}/diagnosis — the
 * diagnosis.json that triggered this version's edit. Disabled for
 * v0 (no diagnosis) + enabled only when the UI surfaces the
 * "Why this edit?" panel. */
export function useRewardDiagnosis(
  slug: string | undefined,
  version: number | undefined,
  options: { enabled?: boolean; stage?: string | null } = {},
) {
  const { enabled = true, stage = null } = options;
  return useQuery<RewardDiagnosisPayload>({
    queryKey:
      slug !== undefined && version !== undefined
        ? stage
          ? ["reward", "diagnosis", slug, version, "stage", stage]
          : ["reward", "diagnosis", slug, version]
        : ["reward", "diagnosis", "_none"],
    queryFn: () => getRewardDiagnosis(slug!, version!, stage ?? undefined),
    enabled: enabled && slug !== undefined && version !== undefined && version > 0,
    staleTime: 60_000,
    retry: false,
  });
}

/** POST /projects/{slug}/rewards/prompt — Claude rewrites v<n+1>.py
 * from the user's natural-language prompt. Returns a JobSummary (202);
 * the caller should poll the job and re-fetch rewards on completion. */
export function useRewardPromptEdit(slug: string) {
  const qc = useQueryClient();
  return useMutation<JobSummary, Error, RewardPromptPayload>({
    mutationFn: (payload) => promptReward(slug, payload),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: qk.rewards(slug) });
    },
  });
}

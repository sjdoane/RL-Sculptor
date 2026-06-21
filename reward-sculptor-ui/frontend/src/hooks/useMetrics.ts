import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  calibrateProjectMetric,
  generateProjectMetric,
  getMetricGenProgress,
  listProjectMetrics,
} from "@/lib/api";
import type { MetricGenProgress, MetricSummary } from "@/lib/types";

/** §Ship 35: the project's auto-generated objective metrics. */
export function useProjectMetrics(slug: string, enabled = true) {
  return useQuery<MetricSummary[]>({
    queryKey: ["metrics", slug],
    queryFn: () => listProjectMetrics(slug),
    enabled,
  });
}

/** §Ship 40: poll live generation progress while a generate is running.
 *  `enabled` should track the generate mutation's pending state. */
export function useMetricGenProgress(slug: string, enabled: boolean) {
  return useQuery<MetricGenProgress>({
    queryKey: ["metric-gen-progress", slug],
    queryFn: () => getMetricGenProgress(slug),
    enabled,
    refetchInterval: enabled ? 1200 : false,
    gcTime: 0,
  });
}

/** Generate (LLM; ~1-2 min). Invalidates the project metrics on success. */
export function useGenerateMetric(slug: string) {
  const qc = useQueryClient();
  return useMutation<
    MetricSummary, Error,
    { behavior_goal: string; review?: boolean; n_candidates?: number;
      calibrate_against?: string | null }
  >({
    mutationFn: (body) => generateProjectMetric(slug, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["metrics", slug] }),
  });
}

/** Calibrate a generated metric against a built-in ground truth. */
export function useCalibrateMetric(slug: string) {
  const qc = useQueryClient();
  return useMutation<MetricSummary, Error, { metricId: string; against: string }>({
    mutationFn: ({ metricId, against }) =>
      calibrateProjectMetric(slug, metricId, against),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["metrics", slug] }),
  });
}

import { useQuery } from "@tanstack/react-query";

import { listPolicies } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { PolicySummary } from "@/lib/types";

/** Disk-backed exportable checkpoints for a project (or a mission
 *  stage's own runs tree when `runId` points at a stage run). */
export function usePolicies(
  slug: string | undefined,
  opts?: { runId?: string },
) {
  return useQuery<PolicySummary[]>({
    queryKey: slug
      ? qk.policies(slug, opts?.runId)
      : ["policies", "_none"],
    queryFn: () => listPolicies(slug!, opts?.runId),
    enabled: !!slug,
  });
}

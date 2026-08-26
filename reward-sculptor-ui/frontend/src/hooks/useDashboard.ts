import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { getDashboard, getSystemInfo } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { DashboardSummary, SystemInfo } from "@/lib/types";

export function useDashboard() {
  return useQuery<DashboardSummary>({
    queryKey: qk.dashboard(),
    queryFn: getDashboard,
    // Aggressive refresh while active jobs might exist; gets more
    // idle once the dashboard data is steady.
    refetchInterval: (q) => {
      const d = q.state.data as DashboardSummary | undefined;
      if (!d) return 3000;
      return d.active_jobs.length > 0 ? 3000 : 15000;
    },
    // Serve stale for smooth UX while refetching in the background.
    staleTime: 1000,
  });
}

/** §Ship 37: the slugs that have a run in flight right now.
 *
 *  `ProjectSummary.status` cannot answer this. The backend's `_compute_status`
 *  never returns "running" (its docstring says the state is "owned by the job
 *  manager"), and in practice it returns "draft" for *every* project — its
 *  first branch matches a raw CHANGE_ME substring, and the scaffold ships
 *  `environment_tag = "CHANGE_ME"` that nothing fills in. Measured on this
 *  install: 38/38 projects report "draft", 13 of them with completed
 *  iterations, and 38/39 config.toml files still contain the placeholder.
 *
 *  The dashboard's active-jobs feed is the honest source, and it is already
 *  polled here every 3s while anything is live. */
export function useRunningSlugs(): Set<string> {
  const dash = useDashboard();
  return useMemo(
    () =>
      new Set(
        (dash.data?.active_jobs ?? [])
          .filter((j) => j.status === "running" || j.status === "queued")
          .map((j) => j.project_slug)
          .filter((s): s is string => !!s),
      ),
    [dash.data],
  );
}

export function useSystemInfo() {
  return useQuery<SystemInfo>({
    queryKey: qk.systemInfo(),
    queryFn: getSystemInfo,
    staleTime: 30_000,
  });
}

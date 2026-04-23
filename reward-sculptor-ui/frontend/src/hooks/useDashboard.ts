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

export function useSystemInfo() {
  return useQuery<SystemInfo>({
    queryKey: qk.systemInfo(),
    queryFn: getSystemInfo,
    staleTime: 30_000,
  });
}

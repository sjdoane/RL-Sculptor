import { useQuery } from "@tanstack/react-query";

import {
  getLibraryCategories,
  getLibraryRobot,
  getSystemGpu,
  getSystemKgStats,
  listLibraryAdapters,
  listLibraryRobots,
} from "@/lib/api";
import type {
  AdapterInfo,
  LibraryListResponse,
  LibraryRobot,
  SystemGpuResponse,
  SystemKgStatsResponse,
} from "@/lib/types";

export function useLibraryCategories() {
  return useQuery<string[]>({
    queryKey: ["library", "categories"],
    queryFn: getLibraryCategories,
    staleTime: Infinity, // static enum
  });
}

export function useLibraryRobots(params?: {
  category?: string;
  training_support?: string;
  search?: string;
}) {
  return useQuery<LibraryListResponse>({
    queryKey: ["library", "robots", params ?? {}],
    queryFn: () => listLibraryRobots(params),
    staleTime: 60_000,
  });
}

export function useLibraryRobot(slug: string | undefined) {
  return useQuery<LibraryRobot>({
    queryKey: ["library", "robot", slug],
    queryFn: () => getLibraryRobot(slug!),
    enabled: !!slug,
    staleTime: 60_000,
  });
}

export function useLibraryAdapters() {
  return useQuery<AdapterInfo[]>({
    queryKey: ["library", "adapters"],
    queryFn: listLibraryAdapters,
    staleTime: Infinity, // static across a backend lifetime
  });
}

export function useSystemGpu(opts?: { refetchIntervalMs?: number | false }) {
  return useQuery<SystemGpuResponse>({
    queryKey: ["system", "gpu"],
    queryFn: getSystemGpu,
    refetchInterval: opts?.refetchIntervalMs ?? false,
    staleTime: 2_000,
  });
}

/** User-wide shared KG stats. Refreshes on focus; invalidate via the
 * query key after a bulk-seed / research / extract job completes. */
export function useSharedKgStats(opts?: { refetchIntervalMs?: number | false }) {
  return useQuery<SystemKgStatsResponse>({
    queryKey: ["system", "kg", "stats"],
    queryFn: getSystemKgStats,
    refetchInterval: opts?.refetchIntervalMs ?? false,
    staleTime: 30_000,
  });
}

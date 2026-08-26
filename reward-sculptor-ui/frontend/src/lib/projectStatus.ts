import type { ProjectStatus } from "@/lib/types";


/** Project `completed` means at least one run finished, not research success. */
export function projectBadgeStatus(
  status: ProjectStatus,
  hasActiveRun = false,
): string {
  if (hasActiveRun) return "running";
  return status === "completed" ? "project-history" : status;
}

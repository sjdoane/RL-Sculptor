import type { ProjectDetail } from "@/lib/types";

/**
 * Which reference-library embodiment namespace this project's clips live in.
 *
 * Shared because getting it wrong is silent and expensive. The project API
 * exposes the catalog's explicit reference namespace; the browser must not
 * independently tokenize an asset slug or task id and arrive at a different
 * robot than run/mission admission.
 */
export function referenceRobotForProject(project: ProjectDetail): string {
  const namespace = project.reference_robot;
  if (typeof namespace === "string" && namespace.trim()) {
    return namespace.trim();
  }
  // The backend owns the only legacy allowlist. Unknown embodiments fail
  // closed with an empty picker instead of reimplementing slug/task heuristics
  // in the browser.
  return "unassigned";
}

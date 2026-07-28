import type { ProjectDetail } from "@/lib/types";

/**
 * Which reference-library embodiment namespace this project's clips live in.
 *
 * Shared because getting it wrong is silent and expensive: several call sites
 * used to hardcode `"g1"`, so on a Go1 project the picker offered G1 motion,
 * the attach succeeded (the backend scans every robot directory), the clip
 * rendered fine in the card — and the run failed at training time with
 * `reference_tracking_seed_failed`, unrecoverable without hand-editing JSON,
 * because the mismatch is not representable in the stored state.
 */
export function referenceRobotForProject(project: ProjectDetail): string {
  const hints = [
    project.adapter_config?.reference_robot,
    project.library_slug,
    project.adapter_config?.robot,
    project.adapter_config?.task_id,
    project.env_id,
  ];
  for (const raw of hints) {
    if (typeof raw !== "string" || !raw.trim()) continue;
    const tokens = raw.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    for (const token of [...tokens].reverse()) {
      // Asset-family suffixes such as "29dof" carry morphology detail but
      // are not library namespaces; prefer the preceding embodiment token.
      if (/^\d+$/.test(token) || /^\d+dof$/.test(token)) continue;
      if (["mjlab", "velocity", "flat", "unitree", "booster"].includes(token)) {
        continue;
      }
      return token;
    }
  }
  // Unknown embodiments fail closed with an empty picker instead of silently
  // borrowing another robot's motion namespace.
  return "unassigned";
}

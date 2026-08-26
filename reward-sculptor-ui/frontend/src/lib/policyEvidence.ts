import type { PolicySummary } from "@/lib/types";

/** The backend's shared completion authority is the sole deployment gate.
 * Evidence coverage is descriptive: generic tasks need not expose the
 * route/contact/hold component names used by the current showcase metric. */
export function isDeployablePolicy(policy: PolicySummary): boolean {
  return policy.deployable;
}

import type { RefDetail, TierDCertificationScope } from "@/lib/types";

const SHA256_RE = /^[a-f0-9]{64}$/;

export const TIER_D_SCOPE_EXPECTATION: TierDCertificationScope = {
  schema: "reward-sculptor-tier-d-scope-v1",
  claim: "exact-schedule joint-position and root-height tracking",
  gated_evidence: [
    "mean_joint_position_error",
    "root_z_rmse",
    "duration_coverage",
    "non_vacuous_reference_motion",
    "beats_static_pose_baseline",
  ],
  measured_only: [
    "maximum_joint_position_error",
    "orientation_error",
    "motion_ratio",
  ],
  not_certified: [
    "root_xy_tracking",
    "contact_safety",
    "collision_avoidance",
    "general_dynamics_feasibility",
  ],
};

function exactStringList(actual: unknown, expected: string[]): boolean {
  return Array.isArray(actual)
    && actual.every((value) => typeof value === "string")
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

export function hasExactTierDScope(
  scope: TierDCertificationScope | null | undefined,
): scope is TierDCertificationScope {
  return scope?.schema === TIER_D_SCOPE_EXPECTATION.schema
    && scope.claim === TIER_D_SCOPE_EXPECTATION.claim
    && exactStringList(
      scope.gated_evidence,
      TIER_D_SCOPE_EXPECTATION.gated_evidence,
    )
    && exactStringList(
      scope.measured_only,
      TIER_D_SCOPE_EXPECTATION.measured_only,
    )
    && exactStringList(
      scope.not_certified,
      TIER_D_SCOPE_EXPECTATION.not_certified,
    );
}

/**
 * One client-side authority for the Tier-D receipt shown by readiness cards
 * and consumed by the primary launch action.  The backend remains the source
 * of truth; this guard prevents a stale or partial response from being
 * presented as launch-ready.
 */
export function hasExactTierDReceipt(
  detail: RefDetail | null | undefined,
): boolean {
  const admission = detail?.dynamics_admission;
  const artifact = detail?.artifact_identity;
  if (!admission || !artifact) return false;
  const scope = admission.certification_scope;
  return admission.admitted === true
    && admission.tier === "D"
    && admission.artifact_hash_verified === true
    && artifact.verified === true
    && SHA256_RE.test(admission.certificate_digest ?? "")
    && SHA256_RE.test(admission.clip_sha256 ?? "")
    && SHA256_RE.test(admission.rollout_sha256 ?? "")
    && SHA256_RE.test(admission.execution_contract_sha256 ?? "")
    && SHA256_RE.test(admission.execution_boundary_sha256 ?? "")
    && SHA256_RE.test(admission.reference_clock_sha256 ?? "")
    && hasExactTierDScope(scope)
    && artifact.clip_sha256 === admission.clip_sha256
    && artifact.provenance_clip_sha256 === admission.clip_sha256;
}

/** Technical receipt lines for the certified motion boundary.

The active reward's frozen tracking-backbone digest is intentionally not
invented from Tier-D motion certification. It is a distinct worker-observed
launch/runtime fact and stays labeled as such here.
*/
export function referenceExecutionReceiptLines(
  detail: RefDetail | null | undefined,
): string[] {
  const admission = detail?.dynamics_admission;
  return [
    `execution contract sha256: ${admission?.execution_contract_sha256 ?? "not admitted"}`,
    `execution boundary sha256: ${admission?.execution_boundary_sha256 ?? "not admitted"}`,
    `reference clock sha256: ${admission?.reference_clock_sha256 ?? "not admitted"}`,
    "active reward tracking backbone sha256: verified only at launch/runtime; not part of Tier-D motion certification",
  ];
}

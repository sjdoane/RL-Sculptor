import type { RefDetail } from "@/lib/types";

const SHA256_RE = /^[a-f0-9]{64}$/;

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
  return admission.admitted === true
    && admission.tier === "D"
    && admission.artifact_hash_verified === true
    && artifact.verified === true
    && SHA256_RE.test(admission.certificate_digest ?? "")
    && SHA256_RE.test(admission.clip_sha256 ?? "")
    && SHA256_RE.test(admission.rollout_sha256 ?? "")
    && artifact.clip_sha256 === admission.clip_sha256
    && artifact.provenance_clip_sha256 === admission.clip_sha256;
}

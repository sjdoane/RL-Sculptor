import { describe, expect, test } from "vitest";

import {
  hasExactTierDReceipt,
  referenceExecutionReceiptLines,
} from "@/lib/referenceAdmission";
import type { RefDetail } from "@/lib/types";

function detail(): RefDetail {
  return {
    index_row: {
      clip_id: "hop", robot: "g1", text: "hop", labels: ["hop"],
      tier: "D", license: "cc0", n_frames: 120, fps: 60,
      duration_s: 1.983333, root_z_range: [0.7, 1.0], has_preview: true,
    },
    provenance: {},
    artifact_identity: {
      verified: true,
      clip_sha256: "b".repeat(64),
      provenance_clip_sha256: "b".repeat(64),
      source_content_sha256: "a".repeat(64),
      reason: null,
    },
    dynamics_admission: {
      admitted: true,
      tier: "D",
      certificate_digest: "c".repeat(64),
      clip_sha256: "b".repeat(64),
      source_content_sha256: "a".repeat(64),
      artifact_hash_verified: true,
      rollout_sha256: "d".repeat(64),
      execution_contract_sha256: "e".repeat(64),
      execution_boundary_sha256: "f".repeat(64),
      reference_clock_sha256: "0".repeat(64),
      certification_scope: {
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
      },
      reason: null,
    },
  };
}

describe("exact Tier-D receipt authority", () => {
  test("admits a complete, mutually consistent receipt", () => {
    expect(hasExactTierDReceipt(detail())).toBe(true);
  });

  test.each([
    "certificate_digest",
    "clip_sha256",
    "rollout_sha256",
    "execution_contract_sha256",
    "execution_boundary_sha256",
    "reference_clock_sha256",
  ] as const)("rejects a missing or malformed %s", (field) => {
    const value = detail();
    value.dynamics_admission[field] = "short";
    expect(hasExactTierDReceipt(value)).toBe(false);
  });

  test("rejects an artifact/certificate hash mismatch", () => {
    const value = detail();
    value.artifact_identity!.clip_sha256 = "e".repeat(64);
    expect(hasExactTierDReceipt(value)).toBe(false);
  });

  test("rejects an admitted label without verified artifact bytes", () => {
    const value = detail();
    value.dynamics_admission.artifact_hash_verified = false;
    expect(hasExactTierDReceipt(value)).toBe(false);
  });

  test("rejects missing or broadened certification scope", () => {
    const missing = detail();
    missing.dynamics_admission.certification_scope = null;
    expect(hasExactTierDReceipt(missing)).toBe(false);

    const broadened = detail();
    broadened.dynamics_admission.certification_scope!.not_certified = [];
    expect(hasExactTierDReceipt(broadened)).toBe(false);
  });

  test.each([
    ["gated_evidence", "root_z_rmse"],
    ["measured_only", "orientation_error"],
    ["not_certified", "root_xy_tracking"],
  ] as const)("rejects a scope missing %s entry %s", (field, entry) => {
    const value = detail();
    const scope = value.dynamics_admission.certification_scope!;
    scope[field] = scope[field].filter((candidate) => candidate !== entry);
    expect(hasExactTierDReceipt(value)).toBe(false);
  });

  test("rejects reordered scope evidence rather than inferring authority", () => {
    const value = detail();
    value.dynamics_admission.certification_scope!.measured_only.reverse();
    expect(hasExactTierDReceipt(value)).toBe(false);
  });

  test("keeps certified execution identities separate from runtime backbone proof", () => {
    const lines = referenceExecutionReceiptLines(detail());

    expect(lines).toContain(
      `execution contract sha256: ${"e".repeat(64)}`,
    );
    expect(lines).toContain(
      `execution boundary sha256: ${"f".repeat(64)}`,
    );
    expect(lines).toContain(
      `reference clock sha256: ${"0".repeat(64)}`,
    );
    expect(lines.at(-1)).toMatch(
      /tracking backbone sha256: verified only at launch\/runtime; not part of Tier-D motion certification/i,
    );
  });
});

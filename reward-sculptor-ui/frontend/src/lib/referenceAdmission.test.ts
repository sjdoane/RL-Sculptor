import { describe, expect, test } from "vitest";

import { hasExactTierDReceipt } from "@/lib/referenceAdmission";
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
});

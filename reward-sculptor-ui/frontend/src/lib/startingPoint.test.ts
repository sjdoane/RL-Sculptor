import { expect, test } from "vitest";

import {
  interruptedSnapshotReadinessIssue,
  resolveBundleMotionUpdate,
  startingPointRunFields,
  type MotionSelection,
} from "@/lib/startingPoint";
import type { StartingPointSelection } from "@/lib/types";

const selection = (
  referenceRobot: string | null,
  referenceClipId: string | null,
): StartingPointSelection => ({
  kind: "shared_skill",
  warm_start_iteration: null,
  starting_skill_id: "skill-2",
  initialization_mode: "actor_only",
  reference_clip_id: referenceClipId,
  reference_robot: referenceRobot,
  import_manifest_digest: "b".repeat(64),
  compatibility_contract_provenance_status: "origin_persisted",
  acknowledge_legacy_reconstructed_initialization: false,
});

const priorBundle: MotionSelection = {
  clipId: "prior-motion",
  robot: "g1",
  source: "bundle",
};

test("attaches only the exact robot-scoped bundle reference", () => {
  expect(resolveBundleMotionUpdate(
    selection("g1", "new-motion"), "g1", priorBundle,
  )).toEqual({
    kind: "attach",
    motion: { clipId: "new-motion", robot: "g1", source: "bundle" },
  });
});

test("clears stale bundle motion when a new bundle reference mismatches", () => {
  expect(resolveBundleMotionUpdate(
    selection("go1", "new-motion"), "g1", priorBundle,
  )).toEqual({ kind: "clear" });
  expect(resolveBundleMotionUpdate(
    selection(null, null), "g1", priorBundle,
  )).toEqual({ kind: "clear" });
});

test("preserves a separately selected library motion", () => {
  expect(resolveBundleMotionUpdate(
    selection("go1", "new-motion"),
    "g1",
    { ...priorBundle, source: "library" },
  )).toEqual({ kind: "preserve" });
});

test("serializes interrupted recovery with opaque pins and no UI display or path fields", () => {
  const interrupted: StartingPointSelection = {
    ...selection(null, null),
    kind: "project_checkpoint",
    warm_start_iteration: null,
    warm_start_snapshot: {
      snapshot_id: "snap_7fd3a41b",
      checkpoint_sha256: "a".repeat(64),
      receipt_digest: "b".repeat(64),
      acknowledge_interrupted_snapshot: true,
      acknowledge_legacy_reconstructed_snapshot: true,
    },
    warm_start_snapshot_display: {
      iteration: 2,
      ppo_step: 50,
      last_observed_ppo_iteration: 58,
      checkpoint_bytes: 6_202_705,
      provenance_status: "legacy_reconstructed",
    },
  };

  const fields = startingPointRunFields(interrupted);
  expect(fields).toEqual({
    warm_start_iteration: null,
    warm_start_snapshot: {
      snapshot_id: "snap_7fd3a41b",
      checkpoint_sha256: "a".repeat(64),
      receipt_digest: "b".repeat(64),
      acknowledge_interrupted_snapshot: true,
      acknowledge_legacy_reconstructed_snapshot: true,
    },
    starting_skill_id: null,
    expected_starting_skill_manifest_digest: null,
    initialization_mode: null,
    acknowledge_legacy_reconstructed_initialization: false,
  });
  expect(fields.warm_start_snapshot).not.toHaveProperty("path");
  expect(fields.warm_start_snapshot).not.toHaveProperty("iteration");
  expect(fields.warm_start_snapshot).not.toHaveProperty("ppo_step");
});

const interruptedSelection = (
  provenanceStatus: "origin_persisted" | "legacy_reconstructed" =
    "origin_persisted",
): StartingPointSelection => ({
  ...selection(null, null),
  kind: "project_checkpoint",
  warm_start_iteration: null,
  warm_start_snapshot: {
    snapshot_id: `snap_${"7".repeat(24)}`,
    checkpoint_sha256: "a".repeat(64),
    receipt_digest: "b".repeat(64),
    acknowledge_interrupted_snapshot: true,
    acknowledge_legacy_reconstructed_snapshot:
      provenanceStatus === "legacy_reconstructed",
  },
  warm_start_snapshot_display: {
    iteration: 2,
    ppo_step: 50,
    last_observed_ppo_iteration: 58,
    checkpoint_bytes: 6_202_705,
    provenance_status: provenanceStatus,
  },
});

test("fails closed when an interrupted selection has lost its display receipt", () => {
  const interrupted = interruptedSelection();
  interrupted.warm_start_snapshot_display = null;
  expect(interruptedSnapshotReadinessIssue(interrupted)).toMatch(
    /provenance disclosure can be verified/i,
  );
});

test("requires the acknowledgement that matches snapshot provenance", () => {
  const legacy = interruptedSelection("legacy_reconstructed");
  legacy.warm_start_snapshot!.acknowledge_legacy_reconstructed_snapshot = false;
  expect(interruptedSnapshotReadinessIssue(legacy)).toMatch(
    /legacy-reconstructed receipt/i,
  );

  const persisted = interruptedSelection("origin_persisted");
  persisted.warm_start_snapshot!.acknowledge_legacy_reconstructed_snapshot = true;
  expect(interruptedSnapshotReadinessIssue(persisted)).toMatch(
    /acknowledgement contradicts the current receipt/i,
  );
});

test("admits only a complete, acknowledged snapshot disclosure", () => {
  expect(interruptedSnapshotReadinessIssue(interruptedSelection())).toBeNull();
  expect(
    interruptedSnapshotReadinessIssue(
      interruptedSelection("legacy_reconstructed"),
    ),
  ).toBeNull();
});

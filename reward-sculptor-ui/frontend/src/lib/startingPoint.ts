import type {
  RunParamsPayload,
  StartingPointSelection,
} from "@/lib/types";

export interface MotionSelection {
  clipId: string;
  robot: string;
  source: "draft" | "library" | "bundle";
}

export type BundleMotionUpdate =
  | { kind: "attach"; motion: MotionSelection }
  | { kind: "clear" }
  | { kind: "preserve" };

type StartingPointRunFields = Pick<
  RunParamsPayload,
  | "warm_start_iteration"
  | "warm_start_snapshot"
  | "starting_skill_id"
  | "expected_starting_skill_manifest_digest"
  | "initialization_mode"
  | "acknowledge_legacy_reconstructed_initialization"
>;

/** Convert UI/provenance context to the narrow run API contract. Snapshot
 * display fields are intentionally dropped; the request carries only an
 * opaque id, immutable digests, and explicit acknowledgements. */
export function startingPointRunFields(
  startingPoint: StartingPointSelection,
): StartingPointRunFields {
  const snapshot = startingPoint.kind === "project_checkpoint"
    ? startingPoint.warm_start_snapshot ?? null
    : null;
  return {
    warm_start_iteration:
      startingPoint.kind === "project_checkpoint" && snapshot == null
        ? startingPoint.warm_start_iteration
        : null,
    warm_start_snapshot: snapshot
      ? {
          snapshot_id: snapshot.snapshot_id,
          checkpoint_sha256: snapshot.checkpoint_sha256,
          receipt_digest: snapshot.receipt_digest,
          acknowledge_interrupted_snapshot:
            snapshot.acknowledge_interrupted_snapshot,
          acknowledge_legacy_reconstructed_snapshot:
            snapshot.acknowledge_legacy_reconstructed_snapshot ?? false,
        }
      : null,
    starting_skill_id:
      startingPoint.kind === "shared_skill"
        ? startingPoint.starting_skill_id
        : null,
    expected_starting_skill_manifest_digest:
      startingPoint.kind === "shared_skill"
        ? startingPoint.import_manifest_digest
        : null,
    initialization_mode:
      startingPoint.kind === "shared_skill"
        ? startingPoint.initialization_mode
        : null,
    acknowledge_legacy_reconstructed_initialization:
      startingPoint.kind === "shared_skill"
        ? startingPoint.acknowledge_legacy_reconstructed_initialization
        : false,
  };
}

/** Return the first launch-blocking disclosure problem for an interrupted
 * snapshot. The display receipt is required as well as the opaque server pins:
 * without it the UI cannot know which provenance acknowledgement the backend
 * will require, so a rehydrated or stale selection must fail closed. */
export function interruptedSnapshotReadinessIssue(
  startingPoint: StartingPointSelection,
): string | null {
  if (
    startingPoint.kind !== "project_checkpoint"
    || startingPoint.warm_start_snapshot == null
  ) {
    return null;
  }
  const snapshot = startingPoint.warm_start_snapshot;
  if (
    !/^snap_[a-f0-9]{24}$/.test(snapshot.snapshot_id)
    || !/^[a-f0-9]{64}$/.test(snapshot.checkpoint_sha256)
    || !/^[a-f0-9]{64}$/.test(snapshot.receipt_digest)
  ) {
    return "Re-select the interrupted snapshot so its immutable receipt and checkpoint can be pinned.";
  }
  const display = startingPoint.warm_start_snapshot_display;
  if (display == null) {
    return "Re-select the interrupted snapshot so its provenance disclosure can be verified.";
  }
  if (!snapshot.acknowledge_interrupted_snapshot) {
    return "Acknowledge that the interrupted snapshot is unevaluated before launch.";
  }
  const legacyAcknowledged =
    snapshot.acknowledge_legacy_reconstructed_snapshot === true;
  if (
    display.provenance_status === "legacy_reconstructed"
    && !legacyAcknowledged
  ) {
    return "Acknowledge the interrupted snapshot's legacy-reconstructed receipt before launch.";
  }
  if (
    display.provenance_status === "origin_persisted"
    && legacyAcknowledged
  ) {
    return "Refresh the interrupted snapshot because its historical-reconstruction acknowledgement contradicts the current receipt.";
  }
  return null;
}

/** Resolve a picker change without letting a prior bundle's motion leak into
 * a newly selected bundle. Robot and clip id form one immutable identity;
 * neither field is compared or retained independently. */
export function resolveBundleMotionUpdate(
  next: StartingPointSelection,
  projectRobot: string,
  current: MotionSelection | null,
): BundleMotionUpdate {
  if (
    next.reference_clip_id
    && next.reference_robot
    && next.reference_robot === projectRobot
  ) {
    return {
      kind: "attach",
      motion: {
        clipId: next.reference_clip_id,
        robot: next.reference_robot,
        source: "bundle",
      },
    };
  }
  return current?.source === "bundle" ? { kind: "clear" } : { kind: "preserve" };
}

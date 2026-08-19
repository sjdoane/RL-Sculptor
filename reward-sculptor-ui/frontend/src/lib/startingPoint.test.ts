import { expect, test } from "vitest";

import {
  resolveBundleMotionUpdate,
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

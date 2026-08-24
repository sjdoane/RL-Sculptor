import { describe, expect, it } from "vitest";

import {
  deriveModeRewardReadiness,
  resolveModeAuthoringGoal,
} from "@/lib/behaviorFlow";
import type { ModeRewardFile, PromotedModeReward } from "@/lib/api";

const file = (overrides: Partial<ModeRewardFile> = {}): ModeRewardFile => ({
  filename: "mode_reward_v3.py",
  path: "/project/rewards/mode_reward_v3.py",
  clip_id: "parkour",
  reference_robot: "g1",
  execution_context_digest: "a".repeat(64),
  context_current: true,
  tracking_enabled: true,
  mtime: 3,
  digest: "b".repeat(64),
  modes: [
    { name: "run", start_s: 0, end_s: 1, authored: true },
    { name: "vault", start_s: 1, end_s: 2, authored: true },
  ],
  unauthored: [],
  ...overrides,
});

const promoted = (
  overrides: Partial<PromotedModeReward> = {},
): PromotedModeReward => ({
  version: 8,
  filename: "v8.py",
  clip_id: "parkour",
  reference_robot: "g1",
  execution_context_digest: "a".repeat(64),
  context_current: true,
  selection_current: true,
  promotion_blocker: null,
  tracking_enabled: true,
  source_sha256: "b".repeat(64),
  source_filename: "mode_reward_v3.py",
  modes: file().modes,
  unauthored: [],
  ...overrides,
});

const readiness = (
  files: ModeRewardFile[],
  active: PromotedModeReward | null = promoted(),
) => deriveModeRewardReadiness({
  files,
  promoted: active,
  clipId: "parkour",
  robot: "g1",
});

describe("deriveModeRewardReadiness", () => {
  it("accepts only exact current source and selection authority", () => {
    const result = readiness([file()]);
    expect(result.authoredCurrent).toBe(true);
    expect(result.promotedExact).toBe(true);
    expect(result.modeBlocker).toBeNull();
    expect(result.promotionBlocker).toBeNull();
  });

  it("does not show authored completion after the world context changes", () => {
    const result = readiness([file({ context_current: false })]);
    expect(result.authoredCount).toBe(2);
    expect(result.authoredCurrent).toBe(false);
    expect(result.modeBlocker).toMatch(/context changed/i);
  });

  it("rejects another robot even when clip ids and source hashes collide", () => {
    const result = readiness([file({ reference_robot: "go1" })]);
    expect(result.modeFile).toBeNull();
    expect(result.authoredCurrent).toBe(false);
    expect(result.promotedExact).toBe(false);
  });

  it("rejects stale selection and source-byte drift independently", () => {
    expect(readiness(
      [file()], promoted({ selection_current: false }),
    ).promotionBlocker).toMatch(/selection/i);
    expect(readiness(
      [file()], promoted({ source_sha256: "c".repeat(64) }),
    ).promotionBlocker).toMatch(/source bytes/i);
  });
});

describe("resolveModeAuthoringGoal", () => {
  it("keeps the explicit behavior draft ahead of generated descriptions", () => {
    expect(resolveModeAuthoringGoal({
      draftGoal: "  traverse four low rails with controlled hops  ",
      projectDescription: "Generic locomotion project",
      rewardDescription: "Track the reference pose",
    })).toBe("traverse four low rails with controlled hops");
  });

  it("falls back from project intent to reward description", () => {
    expect(resolveModeAuthoringGoal({
      draftGoal: " ",
      projectDescription: "Cross the obstacle course",
      rewardDescription: "Track motion",
    })).toBe("Cross the obstacle course");
    expect(resolveModeAuthoringGoal({
      draftGoal: null,
      projectDescription: "",
      rewardDescription: "Track motion",
    })).toBe("Track motion");
  });
});

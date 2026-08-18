import { describe, expect, it } from "vitest";

import {
  WORLD_ROBOT_MISMATCH_DETAIL,
  worldRobotMismatchBlocker,
} from "@/lib/worldLaunch";


describe("worldRobotMismatchBlocker", () => {
  it("fails closed when the authored world targets another robot", () => {
    expect(worldRobotMismatchBlocker({
      shared_summary: { robot_matches_project: false },
    })).toBe(WORLD_ROBOT_MISMATCH_DETAIL);
  });

  it("does not invent a mismatch when the backend reports a match or no mapping", () => {
    expect(worldRobotMismatchBlocker({
      shared_summary: { robot_matches_project: true },
    })).toBeNull();
    expect(worldRobotMismatchBlocker({
      shared_summary: { robot_matches_project: null },
    })).toBeNull();
    expect(worldRobotMismatchBlocker(null)).toBeNull();
  });
});

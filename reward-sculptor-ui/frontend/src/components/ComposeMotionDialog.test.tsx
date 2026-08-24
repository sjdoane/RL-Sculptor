import { describe, expect, it } from "vitest";

import { sampledTrajectoryFps } from "./ComposeMotionDialog";

describe("sampledTrajectoryFps", () => {
  it("uses N - 1 sample intervals like the runtime reference clock", () => {
    expect(sampledTrajectoryFps(229, 3.8)).toBeCloseTo(60, 10);
  });

  it("keeps one-frame references well-defined and rejects invalid duration", () => {
    expect(sampledTrajectoryFps(1, 1 / 60)).toBeCloseTo(60, 10);
    expect(sampledTrajectoryFps(10, 0)).toBe(0);
  });
});

import { describe, expect, it } from "vitest";

import { referenceRobotForProject } from "@/lib/referenceRobot";
import type { ProjectDetail } from "@/lib/types";


function project(overrides: Partial<ProjectDetail>): ProjectDetail {
  return {
    slug: "robot-map",
    display_name: "Robot map",
    description: "",
    status: "ready",
    created_at: "2026-08-17T00:00:00Z",
    env_id: "Mjlab-Velocity-Flat-Unitree-G1",
    n_iterations_completed: 0,
    project_dir: "/tmp/robot-map",
    adapter_class: "sculptor.adapters.mjlab.MjlabAdapter",
    adapter_config: { task_id: "Mjlab-Velocity-Flat-Unitree-G1" },
    ...overrides,
  };
}


describe("referenceRobotForProject", () => {
  it("uses the backend-provided namespace instead of the catalog slug", () => {
    expect(referenceRobotForProject(project({
      library_slug: "unitree_g1",
      reference_robot: "g1",
    }))).toBe("g1");
  });

  it("does not infer an embodiment from a catalog slug or task id", () => {
    expect(referenceRobotForProject(project({
      library_slug: "unitree_g1",
      reference_robot: null,
    }))).toBe("unassigned");
  });
});

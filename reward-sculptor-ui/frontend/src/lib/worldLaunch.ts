export const WORLD_ROBOT_MISMATCH_TITLE =
  "Training environment targets another robot";

export const WORLD_ROBOT_MISMATCH_DETAIL =
  "Re-author this training environment for the project robot before launching.";

type WorldRobotSelection = {
  shared_summary: {
    robot_matches_project: boolean | null;
  };
};

/** One fail-closed authority shared by launch readiness and submit-time UX. */
export function worldRobotMismatchBlocker(
  selection: WorldRobotSelection | null | undefined,
): string | null {
  return selection?.shared_summary.robot_matches_project === false
    ? WORLD_ROBOT_MISMATCH_DETAIL
    : null;
}

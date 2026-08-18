import type { StartingPointSelection } from "@/lib/types";

export interface MotionSelection {
  clipId: string;
  robot: string;
  source: "draft" | "library" | "bundle";
}

export type BundleMotionUpdate =
  | { kind: "attach"; motion: MotionSelection }
  | { kind: "clear" }
  | { kind: "preserve" };

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

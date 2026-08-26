import type {
  ModeRewardFile,
  PromotedModeReward,
} from "@/lib/api";

export type ModeRewardReadiness = {
  modeFile: ModeRewardFile | null;
  authoredCount: number;
  modeCount: number;
  authoredCurrent: boolean;
  promotedExact: boolean;
  modeBlocker: string | null;
  promotionBlocker: string | null;
};

/** Pick the researcher-authored behavior intent for per-mode authoring.
 *
 * A generated reward description is useful fallback context, but it must not
 * overwrite the behavior draft the researcher explicitly reviewed. Project
 * description sits between them because it is durable human-facing intent,
 * while reward description describes one implementation of that intent.
 */
export function resolveModeAuthoringGoal({
  draftGoal,
  projectDescription,
  rewardDescription,
}: {
  draftGoal?: string | null;
  projectDescription?: string | null;
  rewardDescription?: string | null;
}): string {
  return [draftGoal, projectDescription, rewardDescription]
    .map((value) => value?.trim() ?? "")
    .find(Boolean) ?? "";
}

/** One authority calculation shared by overview readiness and run admission.
 *
 * A file being fully authored is not enough: it was grounded against one
 * immutable execution context. Likewise a promoted version is usable only if
 * it came from those exact source bytes and the current pinned selection names
 * it. Keeping these checks in a pure function makes the green-check contract
 * independently testable instead of duplicating looser UI heuristics.
 */
export function deriveModeRewardReadiness({
  files,
  promoted,
  clipId,
  robot,
}: {
  files: ModeRewardFile[];
  promoted: PromotedModeReward | null;
  clipId: string;
  robot: string;
}): ModeRewardReadiness {
  const candidates = files.filter(
    (file) => file.clip_id === clipId && file.reference_robot === robot,
  );
  // Prefer a file grounded in the active context. The backend returns newest
  // first, so this retains that order within the same authority class.
  const modeFile = candidates.find((file) => file.context_current)
    ?? candidates[0]
    ?? null;
  const authoredCount = modeFile?.modes.filter((mode) => mode.authored).length
    ?? 0;
  const modeCount = modeFile?.modes.length ?? 0;
  const allAuthored = modeCount > 0 && authoredCount === modeCount;
  const authoredCurrent = allAuthored && modeFile?.context_current === true;

  let modeBlocker: string | null = null;
  if (modeFile && !modeFile.context_current) {
    modeBlocker = "World or execution context changed; regenerate the scaffold.";
  } else if (modeFile && !allAuthored) {
    modeBlocker = `${modeCount - authoredCount} mode${modeCount - authoredCount === 1 ? "" : "s"} still unauthored.`;
  }

  const sameIdentity = !!(
    promoted
    && modeFile
    && promoted.clip_id === clipId
    && promoted.reference_robot === robot
    && promoted.source_filename === modeFile.filename
    && promoted.source_sha256 === modeFile.digest
  );
  const promotedExact = !!(
    sameIdentity
    && promoted?.context_current
    && promoted.selection_current
    && promoted.unauthored.length === 0
  );

  let promotionBlocker: string | null = null;
  if (promoted && !promoted.context_current) {
    promotionBlocker = "Promoted reward was authored for a stale execution context.";
  } else if (promoted && !promoted.selection_current) {
    promotionBlocker = "Current artifact selection does not pin the promoted reward.";
  } else if (promoted && modeFile && !sameIdentity) {
    promotionBlocker = "Promoted reward does not match the exact authored source bytes.";
  } else if (promoted?.promotion_blocker) {
    promotionBlocker = promoted.promotion_blocker;
  }

  return {
    modeFile,
    authoredCount,
    modeCount,
    authoredCurrent,
    promotedExact,
    modeBlocker,
    promotionBlocker,
  };
}

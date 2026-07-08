// §Ship 20 (de-siloing) increment 4: shared display helpers for the new
// stage metadata (display_label, failure_reason, superseded status) so
// every stage-listing surface (RunsTab, MissionDetailDialog, RobotViewer,
// RewardsTab) renders the same numbering and copy instead of drifting.
import type { StageSchema } from "@/lib/types";

/** Server-computed hierarchical label ("1", "1.1", …) when available;
 *  falls back to the 1-based array position for stages predating the
 *  field (or while a partial payload is still loading). */
export function stageLabel(
  stage: Pick<StageSchema, "display_label"> | null | undefined,
  fallback1Based: number,
): string {
  return stage?.display_label || String(fallback1Based);
}

const FAILURE_REASON_TEXT: Record<string, (iterationsUsed: number) => string> = {
  criterion_not_met: (n) => `Trained ${n} iteration${n === 1 ? "" : "s"} — success criterion not met`,
  training_errored: () => "Training crashed",
  extension_errored: () => "Training extension crashed",
  v1_materialization_errored: () => "Reward materialization failed (before training)",
  no_checkpoint: () => "No usable checkpoint produced",
  criterion_errored: () => "Success criterion raised an error",
  scaffold_errored: () => "Stage scaffolding failed",
  adapter_mismatch: () => "Adapter mismatch on resume",
};

/** Human-readable failure explanation for a stage's `failure_reason`.
 *  Unknown/unmapped reasons fall back to the raw string so nothing is
 *  ever silently dropped. Null/undefined reason yields null (caller
 *  decides whether to render anything). */
export function failureReasonText(
  reason: string | null | undefined,
  iterationsUsed: number,
): string | null {
  if (!reason) return null;
  const fn = FAILURE_REASON_TEXT[reason];
  return fn ? fn(iterationsUsed) : reason;
}

/** Copy for a stage that was superseded by redecomposition children —
 *  status is terminal but its artifacts (checkpoints, rewards, rollouts)
 *  are retained on disk and still reachable. */
export function supersededText(_stage?: Pick<StageSchema, "name"> | null): string {
  return "Re-planned into sub-stages — artifacts retained";
}

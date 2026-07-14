// §UX honesty pass: shared per-iteration labeling helpers so every
// disk-truth iteration surface (RunsTab, MissionDetailDialog, ReportsTab)
// renders the same "why was this kept" copy and the same honest metric
// labels instead of drifting between call sites. Moved out of
// MissionDetailDialog (the original owner of SELECTION_LABEL) so RunsTab
// and ReportsTab can share it instead of re-deriving their own strings.
import type { StageIteration } from "@/lib/types";

/** Short human descriptor per `selection_source` value — no "kept" prefix,
 *  callers own how that word is composed into the surrounding UI. */
export const SELECTION_LABEL: Record<string, string> = {
  "criterion+fitness": "best passing iter",
  criterion_newest: "newest passing iter",
  fitness_fallback: "best fitness (criterion unmet)",
  last: "last iter",
};

/** Longer sentence per `selection_source`, for tooltips — paired with the
 *  raw source key so the tooltip never hides which enum value fired. */
const SELECTION_SENTENCE: Record<string, string> = {
  "criterion+fitness": "passed the success criterion and ranked best by objective fitness",
  criterion_newest: "passed the success criterion; kept the newest passing iteration",
  fitness_fallback: "the success criterion was not met — kept the iteration with the best objective fitness",
  last: "no criterion or fitness ranking applied — kept the last iteration trained",
};

/** Short reason text for a `selection_source` value, e.g. "best passing
 *  iter". Falls back to the raw source string for anything unmapped, and
 *  to "kept" when no source was recorded at all — never renders blank. */
export function selectionLabel(source: string | null | undefined): string {
  if (!source) return "kept";
  return SELECTION_LABEL[source] ?? source;
}

/** Full-sentence explanation for tooltips, e.g. "criterion+fitness: passed
 *  the success criterion and ranked best by objective fitness". */
export function selectionSentence(source: string | null | undefined): string {
  if (!source) return "kept — selection reason not recorded on disk";
  const detail = SELECTION_SENTENCE[source];
  return detail ? `${source}: ${detail}` : source;
}

/** Honest per-iteration metric labels. `fitness` (objective, 0-1) is the
 *  primary steering score when the loop tracked it; `reward` (mean return)
 *  is always labelled so nothing ever renders as an unlabeled bare number.
 *  `rewardTitle` explains the scale, and flags when fitness is missing —
 *  reward alone can look like an objective score if unlabeled. */
export function formatIterMetrics(
  row: Pick<StageIteration, "fitness" | "primary_metric">,
): { fitnessText: string | null; rewardText: string | null; rewardTitle: string } {
  const fitnessText = typeof row.fitness === "number" ? `fit ${row.fitness.toFixed(2)}` : null;
  const rewardText = typeof row.primary_metric === "number" ? `r ${row.primary_metric.toFixed(1)}` : null;
  const rewardTitle =
    row.fitness == null
      ? "reward (mean return) — objective fitness was not recorded on disk for this iteration"
      : "reward (mean return) — reward-version scale, not comparable across stages";
  return { fitnessText, rewardText, rewardTitle };
}

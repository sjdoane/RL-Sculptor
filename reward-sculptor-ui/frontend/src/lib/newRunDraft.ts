import type { MotionSelection } from "@/lib/startingPoint";
import type { StartingPointSelection } from "@/lib/types";

export type NewRunProfile = "custom" | "pipeline" | "rehearsal" | "overnight";
export type NewRunRenderSize = "default" | "1920x1080" | "960x540" | "320x240";

/** Serializable state that defines one New Run plan.
 *
 * Transient picker/modal state is intentionally absent. Everything that can
 * change the launch request or its researcher-facing review receipt is
 * present, including the independently selected policy, reference, world
 * acknowledgement, objective, and advanced overrides.
 */
export interface NewRunPlanDraft {
  tab: "basic" | "advanced";
  behavior: string;
  profile: NewRunProfile;
  iterations: number;
  trainingIters: number | "";
  numEnvs: number | "";
  device: string;
  noKg: boolean;
  dryRun: boolean;
  interactive: boolean;
  resumeExactTuple: boolean;
  startingPoint: StartingPointSelection;
  allowDefaultWorld: boolean;
  maxEpisodeSteps: number | "";
  playbackSpeed: number | "";
  rolloutEpisodes: number | "";
  seed: number | "";
  renderEnvIndex: number | "";
  renderSize: NewRunRenderSize;
  autoAdjustPhysics: boolean | null;
  fitnessMetric: string | null;
  allowBlindFitness: boolean;
  fitnessMode: "observe" | "steer";
  fitnessPatience: number | "";
  motion: MotionSelection | null;
  metricCandidates: number;
  calibrateAgainst: string;
}

export interface NewRunDraftContext {
  slug: string;
  projectDir: string;
  adapterClass: string;
}

interface StoredNewRunPlanDraft {
  schema: 1;
  context: NewRunDraftContext;
  draft: NewRunPlanDraft;
}

export type NewRunDraftRestore =
  | { status: "none" }
  | { status: "restored"; draft: NewRunPlanDraft }
  | { status: "rejected"; reason: string };

const STORAGE_PREFIX = "reward-sculptor:new-run-plan:";

function storageKey(slug: string): string {
  return `${STORAGE_PREFIX}${encodeURIComponent(slug)}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNumberOrBlank(value: unknown): value is number | "" {
  return value === "" || (typeof value === "number" && Number.isFinite(value));
}

function isStartingPoint(value: unknown): value is StartingPointSelection {
  if (!isRecord(value)) return false;
  return value.kind === "scratch"
    || value.kind === "project_checkpoint"
    || value.kind === "shared_skill";
}

function isMotion(value: unknown): value is MotionSelection | null {
  if (value === null) return true;
  return isRecord(value)
    && typeof value.clipId === "string"
    && typeof value.robot === "string"
    && (value.source === "draft" || value.source === "library" || value.source === "bundle");
}

function isDraft(value: unknown): value is NewRunPlanDraft {
  if (!isRecord(value)) return false;
  const numberOrBlank = [
    value.trainingIters,
    value.numEnvs,
    value.maxEpisodeSteps,
    value.playbackSpeed,
    value.rolloutEpisodes,
    value.seed,
    value.renderEnvIndex,
    value.fitnessPatience,
  ];
  return (value.tab === "basic" || value.tab === "advanced")
    && typeof value.behavior === "string"
    && ["custom", "pipeline", "rehearsal", "overnight"].includes(String(value.profile))
    && typeof value.iterations === "number"
    && Number.isFinite(value.iterations)
    && numberOrBlank.every(isNumberOrBlank)
    && typeof value.device === "string"
    && typeof value.noKg === "boolean"
    && typeof value.dryRun === "boolean"
    && typeof value.interactive === "boolean"
    && typeof value.resumeExactTuple === "boolean"
    && isStartingPoint(value.startingPoint)
    && typeof value.allowDefaultWorld === "boolean"
    && ["default", "1920x1080", "960x540", "320x240"].includes(String(value.renderSize))
    && (value.autoAdjustPhysics === null || typeof value.autoAdjustPhysics === "boolean")
    && (value.fitnessMetric === null || typeof value.fitnessMetric === "string")
    && typeof value.allowBlindFitness === "boolean"
    && (value.fitnessMode === "observe" || value.fitnessMode === "steer")
    && isMotion(value.motion)
    && typeof value.metricCandidates === "number"
    && Number.isFinite(value.metricCandidates)
    && typeof value.calibrateAgainst === "string";
}

function sessionStore(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null;
  }
}

/** Save a one-shot detour draft. The project and adapter identity are bound to
 * the payload so another project or a reconfigured adapter cannot consume it.
 */
export function saveNewRunPlanDraft(
  context: NewRunDraftContext,
  draft: NewRunPlanDraft,
): void {
  const storage = sessionStore();
  if (!storage) return;
  const envelope: StoredNewRunPlanDraft = { schema: 1, context, draft };
  storage.setItem(storageKey(context.slug), JSON.stringify(envelope));
}

/** Consume the saved detour draft. Invalid, stale, or cross-context bytes are
 * rejected explicitly instead of partially applying fields with fallbacks.
 */
export function takeNewRunPlanDraft(
  context: NewRunDraftContext,
): NewRunDraftRestore {
  const storage = sessionStore();
  if (!storage) return { status: "none" };
  const key = storageKey(context.slug);
  const raw = storage.getItem(key);
  if (raw === null) return { status: "none" };
  storage.removeItem(key);
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { status: "rejected", reason: "The saved run plan was not valid JSON." };
  }
  if (!isRecord(parsed) || parsed.schema !== 1 || !isRecord(parsed.context)) {
    return { status: "rejected", reason: "The saved run plan used an unsupported schema." };
  }
  if (
    parsed.context.slug !== context.slug
    || parsed.context.projectDir !== context.projectDir
    || parsed.context.adapterClass !== context.adapterClass
  ) {
    return {
      status: "rejected",
      reason: "The saved run plan belongs to a different project or adapter configuration.",
    };
  }
  if (!isDraft(parsed.draft)) {
    return { status: "rejected", reason: "The saved run plan was incomplete or malformed." };
  }
  return { status: "restored", draft: parsed.draft };
}

export function clearNewRunPlanDraft(slug: string): void {
  sessionStore()?.removeItem(storageKey(slug));
}

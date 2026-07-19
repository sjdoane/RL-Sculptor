/** Canonical React Query keys. Centralized so invalidations hit the
 *  right entries — particularly important for the preview endpoint,
 *  which is called from multiple surfaces (list card, detail pane,
 *  library picker). Any call site using a hand-rolled key is a bug. */

import type { CameraAngle } from "./types";

export const qk = {
  health: () => ["health"] as const,
  projects: () => ["projects"] as const,
  project: (slug: string) => ["project", slug] as const,
  robot: (slug: string) => ["robot", slug] as const,
  preview: (
    slug: string,
    params?: { angle?: CameraAngle; regenerate?: boolean },
  ) =>
    [
      "preview",
      slug,
      {
        angle: params?.angle ?? "iso",
        regenerate: !!params?.regenerate,
      },
    ] as const,

  rewards: (slug: string) => ["rewards", slug] as const,
  reward: (slug: string, version: number) =>
    ["reward", slug, version] as const,

  kgPapers: (slug: string, filter?: { extracted?: boolean; search?: string }) =>
    ["kg", "papers", slug, filter ?? {}] as const,
  kgPaper: (slug: string, arxivId: string) =>
    ["kg", "paper", slug, arxivId] as const,
  kgTechniques: (slug: string) => ["kg", "techniques", slug] as const,
  kgPendingSeeds: (slug: string) => ["kg", "pendingSeeds", slug] as const,

  job: (jobId: string) => ["job", jobId] as const,
  projectJobs: (slug: string) => ["jobs", "project", slug] as const,

  runs: (slug: string) => ["runs", slug] as const,
  run: (slug: string, runId: string) => ["run", slug, runId] as const,
  policies: (slug: string, runId?: string) =>
    ["policies", slug, runId ?? "_project"] as const,

  missions: (slug: string) => ["missions", slug] as const,
  mission: (slug: string, missionSlug: string) =>
    ["mission", slug, missionSlug] as const,
  // §Ship 20 (de-siloing): disk-truth stage iterations + env spec.
  stageIters: (slug: string, missionSlug: string, stageName: string) =>
    ["stageIters", slug, missionSlug, stageName] as const,
  // §increment 3: disk-truth PROJECT-level iterations (plain runs).
  projectIters: (slug: string) => ["projectIters", slug] as const,
  stageEnvSpec: (slug: string, missionSlug: string, stageName: string) =>
    ["stageEnvSpec", slug, missionSlug, stageName] as const,
  // §selection-report UI: the stage's keep-best decision report.
  stageSelection: (slug: string, missionSlug: string, stageName: string) =>
    ["stageSelection", slug, missionSlug, stageName] as const,

  dashboard: () => ["dashboard"] as const,
  systemInfo: () => ["systemInfo"] as const,

  // Recoverable-delete bin (Settings → Trash).
  trash: () => ["trash"] as const,

  // Saved-missions library (durable disk archive).
  saved: () => ["saved"] as const,
  savedEntry: (entryId: string) => ["saved", entryId] as const,

  // Reference library (R1): search results + slim index browse.
  referenceSearch: (query: string, robot: string) =>
    ["references", "search", robot, query] as const,
  referenceIndex: (robot: string) => ["references", "index", robot] as const,

  // Environment authoring (env-authoring item 5): the promoted world
  // tuple + its immutable selection history.
  worldSelection: (slug: string) => ["worldSelection", slug] as const,
  worldLineage: (slug: string) => ["worldLineage", slug] as const,
  worldCurriculum: (slug: string) => ["worldCurriculum", slug] as const,
};

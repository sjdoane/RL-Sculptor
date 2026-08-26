import { beforeEach, describe, expect, test } from "vitest";

import {
  saveNewRunPlanDraft,
  takeNewRunPlanDraft,
  type NewRunDraftContext,
  type NewRunPlanDraft,
} from "@/lib/newRunDraft";

const context: NewRunDraftContext = {
  slug: "g1-reference-evolution",
  projectDir: "/projects/g1-reference-evolution",
  adapterClass: "sculptor.adapters.mjlab.MjlabAdapter",
};

const completeDraft: NewRunPlanDraft = {
  tab: "advanced",
  behavior: "Land a reference cartwheel on a narrow platform, then hold still.",
  profile: "custom",
  iterations: 5,
  trainingIters: 900,
  numEnvs: 768,
  device: "cuda:0",
  noKg: true,
  dryRun: false,
  interactive: false,
  resumeExactTuple: true,
  startingPoint: {
    kind: "shared_skill",
    warm_start_iteration: null,
    warm_start_snapshot: null,
    warm_start_snapshot_display: null,
    starting_skill_id: "g1-certified-cartwheel",
    initialization_mode: "actor_critic",
    reference_clip_id: "cartwheel-v3",
    reference_robot: "g1",
    import_manifest_digest: "a".repeat(64),
    compatibility_contract_provenance_status: "origin_persisted",
    acknowledge_legacy_reconstructed_initialization: false,
    policy_contract_migration: null,
  },
  allowDefaultWorld: false,
  maxEpisodeSteps: 1200,
  playbackSpeed: 0.85,
  rolloutEpisodes: 4,
  seed: 73,
  renderEnvIndex: 10,
  renderSize: "1920x1080",
  autoAdjustPhysics: false,
  fitnessMetric: "gen:platform-cartwheel-hold",
  allowBlindFitness: false,
  fitnessMode: "observe",
  fitnessPatience: 6,
  motion: { clipId: "cartwheel-v3", robot: "g1", source: "bundle" },
  metricCandidates: 4,
  calibrateAgainst: "g1_upright_hold",
};

beforeEach(() => sessionStorage.clear());

describe("New Run environment-detour drafts", () => {
  test("round-trips the complete policy, reference, objective, and advanced plan once", () => {
    saveNewRunPlanDraft(context, completeDraft);

    expect(takeNewRunPlanDraft(context)).toEqual({
      status: "restored",
      draft: completeDraft,
    });
    expect(takeNewRunPlanDraft(context)).toEqual({ status: "none" });
  });

  test("fails closed instead of applying a draft under another adapter", () => {
    saveNewRunPlanDraft(context, completeDraft);

    const result = takeNewRunPlanDraft({
      ...context,
      adapterClass: "sculptor.adapters.gym.GymAdapter",
    });

    expect(result.status).toBe("rejected");
    expect(result).toMatchObject({
      reason: expect.stringMatching(/different project or adapter/i),
    });
  });
});

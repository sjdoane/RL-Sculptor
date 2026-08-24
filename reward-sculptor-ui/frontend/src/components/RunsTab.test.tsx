import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import {
  AuthoredWorldExecutionCard,
  EvaluationFailureNotice,
  IterationTimeline,
  PolicyAvailabilityCard,
  ReferenceAdmissionCard,
  RunStatusBadge,
  StartingPolicyCard,
  deriveReferenceAdmissionState,
  deriveStartingPolicyState,
} from "@/components/RunsTab";
import type { IterEventSummary, RunEvent } from "@/lib/types";

test("explains an evaluation failure without presenting the checkpoint as deployable", () => {
  render(
    <EvaluationFailureNotice
      classification={{
        kind: "post_training_rollout_failed",
        title: "Training checkpoint preserved; evaluation failed",
        detail: "The evaluator exited after checkpoint persistence.",
        suggestions: [],
        problem_type: "/problems/post-training-rollout-failed",
        action: null,
        evidence: {
          failure_stage: "evaluation",
          iteration: 2,
          rl_iter: 750,
          rl_total: 750,
          checkpoint_preserved: true,
        },
      }}
      error="evaluation subprocess exited 1"
    />,
  );

  const alert = screen.getByRole("alert");
  expect(alert).toHaveTextContent("Training checkpoint preserved; evaluation failed");
  expect(alert).toHaveTextContent("checkpoint was preserved");
  expect(alert).toHaveTextContent("not deployment evidence");
  expect(alert).toHaveTextContent("Interrupted or unevaluated");
});

test("labels the primary run badge as evaluation failed after checkpoint preservation", () => {
  render(
    <RunStatusBadge run={{
      status: "errored",
      error: "sculpt exited with code 1",
      error_classification: {
        kind: "post_training_rollout_failed",
        title: "Training checkpoint preserved; evaluation failed",
        detail: "Rollout failed after PPO completed.",
        suggestions: [],
        problem_type: "/problems/post-training-rollout-failed",
        action: null,
      },
    }} />,
  );

  expect(screen.getByText("Evaluation failed")).toBeInTheDocument();
  expect(screen.queryByText("Errored")).not.toBeInTheDocument();
});

test("offers recovery guidance instead of deployment for an unevaluated checkpoint", () => {
  render(
    <PolicyAvailabilityCard iteration={2} evaluated={false} exportHref={null} />,
  );

  expect(screen.getByRole("status", {
    name: /Iteration 2 preserved unevaluated checkpoint/i,
  })).toHaveTextContent("recovery input, not a deployable policy");
  expect(screen.queryByText("Deploy this policy")).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Export policy bundle/i }))
    .not.toBeInTheDocument();
});

test("keeps deployment available for an evaluated policy", () => {
  render(
    <PolicyAvailabilityCard
      iteration={1}
      evaluated
      exportHref="/api/projects/g1/policies/1/export"
    />,
  );

  expect(screen.getByText("Deploy this policy")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /Export policy bundle/i }))
    .toHaveAttribute("href", "/api/projects/g1/policies/1/export");
});

test("renders the terminal evaluation failure instead of live PPO progress", () => {
  const iteration: IterEventSummary = {
    iter_index: 2,
    status: "running",
    started_at: "2026-08-19T00:00:00Z",
    completed_at: null,
    reward_version_before: 7,
    reward_version_after: null,
    primary_metric: null,
    metric_delta: null,
    failure_modes: [],
    edit_count: null,
    paper_refs: [],
    rollout_ready: false,
    diagnosed: false,
    realism_audit: null,
    physics_edit_suggestion: null,
    rl_iter: 749,
    rl_total: 750,
    pct: 99.9,
    eta_s: 1,
  };

  render(
    <IterationTimeline
      iters={[iteration]}
      selected={2}
      onSelect={vi.fn()}
      evaluationFailedIteration={2}
    />,
  );

  expect(screen.getByRole("button", { name: /iter 2.*evaluation failed/i }))
    .toBeInTheDocument();
  expect(screen.queryByText(/749\/750/)).not.toBeInTheDocument();
});

test("states the narrow Tier-D tracking scope without implying general feasibility", () => {
  render(
    <ReferenceAdmissionCard
      outcome="admitted"
      tier="D"
      status="tierd_verified"
      robot="g1"
      clipId="cartwheel"
      clipSha256={"1".repeat(64)}
      rolloutSha256={"2".repeat(64)}
      certificateSha256={"3".repeat(64)}
      executionContractSha256={"4".repeat(64)}
      executionBoundarySha256={"5".repeat(64)}
      trainingAuthorized
      certificationScope={{
        schema: "reward-sculptor-tier-d-scope-v1",
        claim: "exact-schedule joint-position and root-height tracking",
        gated_evidence: ["mean_joint_position_error"],
        measured_only: ["orientation_error"],
        not_certified: [
          "root_xy_tracking",
          "contact_safety",
          "collision_avoidance",
          "general_dynamics_feasibility",
        ],
      }}
      reason={null}
      observedSchedule={null}
      observedScheduleMatchesAdmission={null}
      completionProofSha256={null}
    />,
  );

  expect(screen.getByText("Reference launch admission")).toBeInTheDocument();
  const resolvedAdmission = screen.getByText(/Resolved admission:/).parentElement;
  expect(resolvedAdmission).toHaveTextContent(
    new RegExp(`certificate ${"3".repeat(64)}`),
  );
  expect(resolvedAdmission).toHaveTextContent(
    new RegExp(`execution contract ${"4".repeat(64)}`),
  );
  expect(screen.getByText(/Observed runtime:/).parentElement).toHaveTextContent(
    /Launch admission alone does not prove/i,
  );
  expect(screen.getByText(/Claim: exact-schedule joint-position and root-height tracking/))
    .toBeInTheDocument();
  expect(screen.getByText(/Gated evidence: mean joint position error/))
    .toBeInTheDocument();
  expect(screen.getByText(/Measured, not gated: orientation error/))
    .toBeInTheDocument();
  expect(screen.getByText(
    /Not certified: root xy tracking, contact safety, collision avoidance, general dynamics feasibility/,
  )).toBeInTheDocument();
});

function event(type: string, payload: Record<string, unknown>): RunEvent {
  return {
    type,
    seq: 1,
    ts: "2026-08-24T00:00:00Z",
    ...payload,
  };
}

test("keeps a reference-only import out of policy weight verification", () => {
  const state = deriveStartingPolicyState([
    event("starting_skill_resolved", {
      starting_skill_id: "g1-cartwheel-motion",
      initialization_mode: "reference_only",
      manifest_digest: "b".repeat(64),
      checkpoint_sha256: null,
      trust_status: "validated",
    }),
  ]);

  expect(state?.requested?.roles).toEqual([]);
  expect(state?.resolved?.roles).toEqual([]);
  expect(state?.referenceOnly).toBe(true);
  render(<StartingPolicyCard {...state!} />);
  expect(screen.getByText("Reference-only starting point")).toBeInTheDocument();
  expect(screen.getByText(/No actor or critic weights were requested/i)).toBeInTheDocument();
  expect(screen.queryByText("Starting policy verification")).not.toBeInTheDocument();
  expect(screen.queryByText("Initialized from")).not.toBeInTheDocument();
});

test("separates Tier-D launch admission from worker runtime schedule proof", () => {
  const admission = deriveReferenceAdmissionState([
    event("reference_feasibility_admitted", {
      tier: "D",
      status: "tierd_verified",
      reference_robot: "g1",
      reference_clip_id: "cartwheel",
      clip_sha256: "1".repeat(64),
      rollout_sha256: "2".repeat(64),
      certificate_sha256: "3".repeat(64),
      execution_contract_sha256: "4".repeat(64),
      execution_boundary_sha256: "5".repeat(64),
      training_authorized: true,
    }),
    event("reference_runtime_schedule_admitted", {
      source: "sculpt_run_boundary",
      reference_robot: "g1",
      reference_clip_id: "cartwheel",
      reference_target_sha256: "1".repeat(64),
      phase_mode: "linear",
      phase_duration_s: 2,
      n_phase_targets: 120,
      tracking_backbone_sha256: "6".repeat(64),
    }),
  ]);

  expect(admission?.observedScheduleMatchesAdmission).toBe(true);
  render(<ReferenceAdmissionCard {...admission!} />);
  const observedRuntime = screen.getByText(/Observed runtime:/).parentElement;
  expect(observedRuntime).toHaveTextContent(
    /worker admitted the exact schedule/i,
  );
  expect(observedRuntime).toHaveTextContent(/120 targets/i);
});

const worldPin = {
  selection_version: 4,
  selection_path: "/project/env/selection_v4.json",
  selection_sha256: "7".repeat(64),
  tuple_hash: "8".repeat(64),
};

test("earns Executes in only from an exact worker-authored-world pin", () => {
  render(
    <AuthoredWorldExecutionCard
      receipt={{ requested: worldPin, observed: { ...worldPin } }}
    />,
  );

  expect(screen.getByText("Executes in")).toBeInTheDocument();
  expect(screen.getByText(/Observed:/).parentElement).toHaveTextContent(
    /worker pinned the exact requested tuple and selection bytes/i,
  );
});

test("does not upgrade a requested or mismatched world to execution proof", () => {
  const { rerender } = render(
    <AuthoredWorldExecutionCard
      receipt={{ requested: worldPin, observed: null }}
    />,
  );
  expect(screen.queryByText("Executes in")).not.toBeInTheDocument();
  expect(screen.getByText(/Observed:/).parentElement).toHaveTextContent(
    /requested world does not prove worker execution/i,
  );

  rerender(
    <AuthoredWorldExecutionCard
      receipt={{
        requested: worldPin,
        observed: { ...worldPin, tuple_hash: "9".repeat(64) },
      }}
    />,
  );
  expect(screen.queryByText("Executes in")).not.toBeInTheDocument();
  expect(screen.getByText(/Observed:/).parentElement).toHaveTextContent(
    /differs from the requested world/i,
  );
});

test("does not call a resolved or raw-loaded policy initialized", () => {
  const digest = "a".repeat(64);
  const state = deriveStartingPolicyState([
    event("starting_skill_resolved", {
      starting_skill_id: "g1-cartwheel",
      initialization_mode: "actor_only",
      manifest_digest: "b".repeat(64),
      checkpoint_sha256: digest,
      trust_status: "sanitized",
    }),
    event("warm_start_loaded", {
      source: "/library/checkpoint.pt",
      source_sha256: digest,
      loaded_checkpoint: "/library/checkpoint.pt",
      loaded_checkpoint_sha256: digest,
      load_cfg_keys: ["actor"],
    }),
  ]);

  expect(state).not.toBeNull();
  expect(state?.verified).toBe(false);
  expect(state?.observed).toBeNull();
  render(<StartingPolicyCard {...state!} />);
  expect(screen.queryByText("Initialized from")).not.toBeInTheDocument();
  expect(screen.getByText("Starting policy verification")).toBeInTheDocument();
  expect(screen.getByText(/Requested:/)).toBeInTheDocument();
  expect(screen.getByText(/Resolved:/)).toBeInTheDocument();
  expect(screen.getByText(/does not prove that the weights loaded/i)).toBeInTheDocument();
});

test.each([
  ["actor_only", ["actor"]],
  ["actor_critic", ["actor", "critic"]],
] as const)(
  "earns Initialized from only from the nested verified %s receipt",
  (mode, roles) => {
    const digest = "c".repeat(64);
    const state = deriveStartingPolicyState([
      event("starting_policy_initialization_verified", {
        source: "stdout",
        receipt: {
          schema: 1,
          requested: {
            kind: "starting_skill",
            id: "g1-cartwheel",
            initialization_mode: mode,
            roles: [...roles],
            manifest_digest: "d".repeat(64),
            trust_status: "sanitized",
          },
          resolved: {
            checkpoint: "/library/source.pt",
            checkpoint_sha256: digest,
            initialization_mode: mode,
            roles: [...roles],
          },
          observed: {
            source: "/library/source.pt",
            source_sha256: digest,
            loaded_checkpoint: "/run/loaded.pt",
            loaded_checkpoint_sha256: digest,
            adapted: false,
            initialization_mode: mode,
            roles: [...roles],
            load_cfg_keys: [...roles],
          },
        },
      }),
    ]);

    expect(state?.verified).toBe(true);
    render(<StartingPolicyCard {...state!} />);
    expect(screen.getByText("Initialized from")).toBeInTheDocument();
    const observed = screen.getByText(/exact backend-verified/);
    expect(observed).toHaveTextContent(roles.join(" + "));
    expect(observed).toHaveTextContent("loaded.pt");
  },
);

test("accepts an exact adapted combined-migration receipt with derived bytes", () => {
  const sourceDigest = "5".repeat(64);
  const loadedDigest = "6".repeat(64);
  const migration = {
    type: "zero_initialized_observation_extensions",
    from_schema: 2,
    to_schema: 4,
    extension_width: 4,
    extensions: [
      {
        type: "zero_initialized_event_phase_observation",
        from_schema: 2,
        to_schema: 3,
        extension_width: 3,
        optimizer_resume: false,
      },
      {
        type: "zero_initialized_reference_clock_observation",
        from_schema: 3,
        to_schema: 4,
        extension_width: 1,
        reference_clock_sha256: "7".repeat(64),
        optimizer_resume: false,
      },
    ],
    optimizer_resume: false,
  };
  const reorderedObservedMigration = {
    optimizer_resume: false,
    extensions: migration.extensions,
    extension_width: migration.extension_width,
    to_schema: migration.to_schema,
    from_schema: migration.from_schema,
    type: migration.type,
  };
  const state = deriveStartingPolicyState([
    event("starting_policy_initialization_verified", {
      receipt: {
        schema: 1,
        requested: {
          kind: "starting_skill",
          id: "g1-route-cartwheel",
          initialization_mode: "actor_critic",
          roles: ["actor", "critic"],
        },
        resolved: {
          checkpoint: "/library/source.pt",
          checkpoint_sha256: sourceDigest,
          initialization_mode: "actor_critic",
          roles: ["actor", "critic"],
          target_policy_contract_sha256: "8".repeat(64),
          policy_contract_migration: migration,
        },
        observed: {
          source: "/library/source.pt",
          source_sha256: sourceDigest,
          loaded_checkpoint: "/run/adapted.pt",
          loaded_checkpoint_sha256: loadedDigest,
          adapted: true,
          initialization_mode: "actor_critic",
          roles: ["actor", "critic"],
          load_cfg_keys: ["actor", "critic"],
          effective_policy_contract_sha256: "8".repeat(64),
          policy_contract_migration: reorderedObservedMigration,
        },
      },
    }),
  ]);

  expect(state?.verified).toBe(true);
  render(<StartingPolicyCard {...state!} />);
  expect(screen.getByText("Initialized from")).toBeInTheDocument();
  expect(screen.getByText(/exact backend-verified/)).toHaveTextContent(
    /exact admitted interface migration applied/i,
  );
});

test("rejects a nested receipt whose observed roles differ", () => {
  const digest = "e".repeat(64);
  const state = deriveStartingPolicyState([
    event("starting_policy_initialization_verified", {
      receipt: {
        schema: 1,
        requested: {
          initialization_mode: "actor_critic",
          roles: ["actor", "critic"],
        },
        resolved: {
          checkpoint_sha256: digest,
          initialization_mode: "actor_critic",
          roles: ["actor", "critic"],
        },
        observed: {
          source_sha256: digest,
          loaded_checkpoint_sha256: digest,
          initialization_mode: "actor_critic",
          roles: ["actor"],
          load_cfg_keys: ["actor"],
        },
      },
    }),
  ]);

  expect(state?.verified).toBe(false);
  expect(state?.invalidVerificationReceipt).toBe(true);
  render(<StartingPolicyCard {...state!} />);
  expect(screen.queryByText("Initialized from")).not.toBeInTheDocument();
  expect(screen.getByText(/malformed initialization receipt was rejected/i)).toBeInTheDocument();
});

import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import {
  canonicalSelectedIteration,
  PolicyEvidenceReceipt,
  reportActionsAreCurrent,
} from "@/components/ReportsTab";
import type { PolicySummary, StageIteration } from "@/lib/types";

function policy(overrides: Partial<PolicySummary> = {}): PolicySummary {
  return {
    iter_index: 12,
    checkpoint: "checkpoint.pt",
    checkpoint_bytes: 4096,
    checkpoint_sha256: "a".repeat(64),
    deployable: true,
    artifact_purpose: "reproducibility",
    completion_authority: "attested",
    deployment_status: "qualified",
    deployment_blockers: [],
    physical_scene_status: "aligned",
    lineage_status: "verified",
    origin_receipt_sha256: "c".repeat(64),
    reference_clock_sha256: null,
    primary_metric: 100,
    fitness: 0.91,
    reward_version: "v8",
    metric_id: "weave-stop",
    metric_version: "v2",
    metric_source: "generated",
    metric_sha256: "b".repeat(64),
    criterion_status: "passed",
    evidence_status: "complete",
    route_evidence: {
      key: "actual_route_complete_frac", value: 1, kind: "fraction",
      comparison: "gte", threshold: 1, passed: true,
      semantics_source: "reward-sculptor-objective-evidence-semantics-v1",
    },
    contact_evidence: {
      key: "forbidden_contact_free_frac", value: 1, kind: "fraction",
      comparison: "gte", threshold: 1, passed: true,
      semantics_source: "reward-sculptor-objective-evidence-semantics-v1",
    },
    hold_evidence: {
      key: "strict_hold_count", value: 7, kind: "count",
      comparison: "gte", threshold: 1, passed: true,
      semantics_source: "reward-sculptor-objective-evidence-semantics-v1",
    },
    objective_proof_status: "passed",
    objective_proof_blockers: [],
    lane_evidence_status: "verified",
    requested_evidence_env_index: 10,
    resolved_evidence_env_index: 10,
    resolved_episode_percentile: 0.8125,
    evidence_lane_selection: "precommitted",
    rollout_available: true,
    selected: true,
    selection_source: "objective_criterion",
    ...overrides,
  };
}

test("shows the passed physical-channel and resolved-lane receipt", () => {
  render(<PolicyEvidenceReceipt policy={policy()} defaultOpen />);

  const receipt = screen.getByLabelText("Iteration 12 objective proof receipt");
  expect(receipt).toHaveTextContent(/Objective proof receipt · passed/i);
  expect(receipt).toHaveTextContent("actual_route_complete_frac");
  expect(receipt).toHaveTextContent("forbidden_contact_free_frac");
  expect(receipt).toHaveTextContent("strict_hold_count");
  expect(receipt).toHaveTextContent(/requested 10 → resolved 10/i);
  expect(receipt).toHaveTextContent(/precommitted · percentile 81.3%/i);
  expect(receipt).toHaveTextContent(/deployment qualification passed/i);
});

test("fails closed when lane identity fields are absent", () => {
  render(
    <PolicyEvidenceReceipt
      policy={policy({
        lane_evidence_status: "unavailable",
        requested_evidence_env_index: null,
        resolved_evidence_env_index: null,
        resolved_episode_percentile: null,
        evidence_lane_selection: null,
        objective_proof_status: "incomplete",
        objective_proof_blockers: [
          "worker-authored evidence lane receipt is incomplete",
        ],
      })}
      defaultOpen
    />,
  );

  const receipt = screen.getByLabelText("Iteration 12 objective proof receipt");
  expect(receipt).toHaveTextContent(/Objective proof receipt · incomplete/i);
  expect(receipt).toHaveTextContent(/requested missing → resolved missing/i);
  expect(receipt).toHaveTextContent(/worker-authored evidence lane receipt is incomplete/i);
  expect(receipt).toHaveTextContent(/never inferred/i);
});

test("shows present-but-failing evidence as failed, never complete", () => {
  render(
    <PolicyEvidenceReceipt
      policy={policy({
        route_evidence: {
          key: "actual_route_complete_frac", value: 0, kind: "fraction",
          comparison: "gte", threshold: 1, passed: false,
          semantics_source: "reward-sculptor-objective-evidence-semantics-v1",
        },
        objective_proof_status: "failed",
        objective_proof_blockers: [
          "route evidence failed its declared comparison",
        ],
      })}
      defaultOpen
    />,
  );

  const receipt = screen.getByLabelText("Iteration 12 objective proof receipt");
  expect(receipt).toHaveTextContent(/Objective proof receipt · failed/i);
  expect(receipt).toHaveTextContent(/fail · ≥ 100.0%/i);
  expect(receipt).toHaveTextContent(/route evidence failed its declared comparison/i);
  expect(receipt).not.toHaveTextContent(/receipt · complete/i);
});

test("uses only the canonical selection receipt for the project hero", () => {
  const iteration = (iterIndex: number, fitness: number): StageIteration => ({
    iter_index: iterIndex,
    primary_metric: fitness,
    fitness,
    has_rollout: true,
    has_checkpoint: true,
    reward_version: "v1",
    fitness_contradiction: false,
    fitness_components: null,
    steer_fitness: fitness,
    progress: fitness,
    naturalness_flag: null,
    naturalness_hard_reject: false,
    fitness_source: "live",
    fresh_rollout_count: 1,
  });
  const kept = policy({ iter_index: 4, selected: true, fitness: 0.8 });
  const higherFailed = policy({
    iter_index: 5,
    selected: false,
    fitness: 0.99,
    criterion_status: "failed",
    objective_proof_status: "failed",
  });

  expect(canonicalSelectedIteration(
    [iteration(4, 0.8), iteration(5, 0.99)],
    [kept, higherFailed],
  )?.iter_index).toBe(4);
  expect(canonicalSelectedIteration(
    [iteration(4, 0.8), iteration(5, 0.99)],
    [{ ...kept, selected: false }, higherFailed],
  )).toBeNull();
});

test("withholds report actions unless the receipt is current", () => {
  expect(reportActionsAreCurrent({
    state: "current",
    reason: null,
    claim_status: "descriptive_only",
    selected_iter_index: null,
  })).toBe(true);
  expect(reportActionsAreCurrent({
    state: "stale",
    reason: "evidence changed",
    claim_status: "verified",
    selected_iter_index: 4,
  })).toBe(false);
  expect(reportActionsAreCurrent(undefined)).toBe(false);
});

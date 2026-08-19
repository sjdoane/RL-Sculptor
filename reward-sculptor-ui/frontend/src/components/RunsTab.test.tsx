import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import {
  EvaluationFailureNotice,
  IterationTimeline,
  PolicyAvailabilityCard,
  RunStatusBadge,
} from "@/components/RunsTab";
import type { IterEventSummary } from "@/lib/types";

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

import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import { PolicyInterfaceMigrationNotice } from "@/components/NewRunDialog";
import type {
  StartingPointSelection,
  WorldEventProgram,
} from "@/lib/types";

const EVENT_PROGRAM: WorldEventProgram = {
  id: "route_jump_hold",
  ordered_phase_ids: ["route", "jump", "hold"],
  transition_spec: [
    { from: "route", to: "jump", when: { event: "goal_complete" } },
    { from: "jump", to: "hold", when: { event: "bilateral_support_cycle" } },
    { from: "hold", to: "terminal", when: { event: "minimum_hold_elapsed" } },
  ],
  minimum_air_time_s: 0.06,
  minimum_height_delta_m: 0.18,
  support_selectors: [
    ["robot:left_foot", "world:terrain"],
    ["robot:right_foot", "world:terrain"],
  ],
  terminal_hold_duration_s: 2,
  episode_length_s: 24,
  train_only_phase_sampling: { route: 0.5, jump: 0.4, hold: 0.1 },
  evaluation_start_phase: "route",
  observation_extension: {
    term: "authored_event_phase",
    encoding: "one_hot",
    width: 3,
  },
  provenance: {
    selection_version: 7,
    selection_tuple_hash: "abc123",
    task_artifact: {
      kind: "task",
      version: "v7",
      path: "env/task_v7.json",
      sha256: "f".repeat(64),
    },
  },
};

const checkpoint: StartingPointSelection = {
  kind: "project_checkpoint",
  warm_start_iteration: 12,
  starting_skill_id: null,
  initialization_mode: null,
  reference_clip_id: null,
  reference_robot: null,
  import_manifest_digest: null,
  compatibility_contract_provenance_status: null,
  acknowledge_legacy_reconstructed_initialization: false,
  policy_contract_migration: null,
};

test("does not claim direct-checkpoint migration passed before launch", () => {
  render(
    <PolicyInterfaceMigrationNotice
      startingPoint={checkpoint}
      eventProgram={EVENT_PROGRAM}
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface admission");
  expect(notice).toHaveTextContent(
    "Policy interface admission · verified at launch",
  );
  expect(notice).toHaveTextContent(/3-wide one-hot phase observations/i);
  expect(notice).toHaveTextContent(/target actor and critic interfaces/i);
  expect(notice).toHaveTextContent(/optimizer state is not resumed/i);
  expect(notice).toHaveTextContent(/If this checkpoint is legacy schema 2/i);
  expect(notice).toHaveTextContent(/schema-2 → schema-3 zero-initialized extension/i);
  expect(notice).toHaveTextContent(/Otherwise it must match schema 3 exactly/i);
  expect(notice).toHaveTextContent(/no import preflight receipt/i);
  expect(notice).toHaveTextContent(/verification occurs only when launch begins/i);
});

test("labels an interrupted project snapshot as unevaluated and reverified", () => {
  render(
    <PolicyInterfaceMigrationNotice
      startingPoint={{
        ...checkpoint,
        warm_start_iteration: null,
        warm_start_snapshot: {
          snapshot_id: "snap_7fd3a41b",
          checkpoint_sha256: "a".repeat(64),
          receipt_digest: "b".repeat(64),
          acknowledge_interrupted_snapshot: true,
        },
      }}
      eventProgram={EVENT_PROGRAM}
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface admission");
  expect(notice).toHaveTextContent(/Interrupted snapshot admission · reverified at launch/i);
  expect(notice).toHaveTextContent(/remains unevaluated/i);
  expect(notice).toHaveTextContent(/opaque receipt id/i);
  expect(notice).toHaveTextContent(/warm_start_loaded/i);
  expect(notice).toHaveTextContent(/optimizer state and counters reset/i);
});

test("shows the verified migration type from an imported skill receipt", () => {
  const imported: StartingPointSelection = {
    kind: "shared_skill",
    warm_start_iteration: null,
    starting_skill_id: "g1-route-prior",
    initialization_mode: "actor_only",
    reference_clip_id: null,
    reference_robot: null,
    import_manifest_digest: "a".repeat(64),
    compatibility_contract_provenance_status: "origin_persisted",
    acknowledge_legacy_reconstructed_initialization: false,
    policy_contract_migration: {
      type: "zero_initialized_event_phase_observation",
      from_schema: 2,
      to_schema: 3,
      observation_term: "authored_event_phase",
      extension_width: 3,
      ordered_phase_ids: ["route", "jump", "hold"],
      optimizer_resume: false,
    },
  };
  render(
    <PolicyInterfaceMigrationNotice
      startingPoint={imported}
      eventProgram={EVENT_PROGRAM}
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface migration");
  expect(notice).toHaveTextContent(
    "Policy interface migration · reverified at launch",
  );
  expect(notice).toHaveTextContent(/Verified import migration type/i);
  expect(notice).toHaveTextContent("zero_initialized_event_phase_observation");
  expect(notice).toHaveTextContent(
    /inherited actor receives the zero-initialized extension and the critic starts fresh/i,
  );
  expect(notice).toHaveTextContent(/launch still revalidates/i);
});

test.each(["reference_only", "full_resume"] as const)(
  "omits non-transfer policy migration copy for %s imports",
  (initializationMode) => {
    const nonTransfer: StartingPointSelection = {
      ...checkpoint,
      kind: "shared_skill",
      warm_start_iteration: null,
      starting_skill_id: "non-transfer",
      initialization_mode: initializationMode,
      import_manifest_digest: "b".repeat(64),
    };
    const { container } = render(
      <PolicyInterfaceMigrationNotice
        startingPoint={nonTransfer}
        eventProgram={EVENT_PROGRAM}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  },
);

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";

import {
  ExactLaunchReceipt,
  NewRunDialog,
  PolicyInterfaceMigrationNotice,
  persistClearedReference,
  startingSkillManifestIssue,
} from "@/components/NewRunDialog";
import {
  getBehaviorDraft,
  getReference,
  getSystemGpu,
  getSystemInfo,
  getWorldSelection,
  getWorldValidate,
  listModeRewards,
  listPolicies,
  listProjectMetrics,
  listStartingSkills,
} from "@/lib/api";
import type {
  PolicyContractMigration,
  PolicySummary,
  ProjectDetail,
  StartingPointSelection,
  WorldSelection,
  WorldEventProgram,
} from "@/lib/types";

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    getBehaviorDraft: vi.fn(),
    getReference: vi.fn(),
    getSystemGpu: vi.fn(),
    getSystemInfo: vi.fn(),
    getWorldSelection: vi.fn(),
    getWorldValidate: vi.fn(),
    listModeRewards: vi.fn(),
    listPolicies: vi.fn(),
    listProjectMetrics: vi.fn(),
    listStartingSkills: vi.fn(),
  };
});

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

const project: ProjectDetail = {
  slug: "g1-reference-evolution",
  display_name: "G1 reference evolution",
  description: "Evolve a certified cartwheel into a precise obstacle transition.",
  status: "ready",
  created_at: "2026-08-24T00:00:00Z",
  env_id: null,
  n_iterations_completed: 0,
  project_dir: "/projects/g1-reference-evolution",
  adapter_class: "sculptor.adapters.mjlab.MjlabAdapter",
  adapter_config: {
    task_id: "Mjlab-G1-Reference-Evolution",
    num_envs: 1024,
    device: "cuda:0",
  },
  ready_to_train: true,
  library_slug: "unitree-g1",
  reference_robot: "g1",
};

const world: WorldSelection = {
  selection: {
    selection_version: 4,
    tuple_hash: "a".repeat(64),
    evaluation_lineage: "eval-g1-reference-evolution",
    refs: {
      environment: {
        kind: "environment",
        version: "v4",
        path: "env/environment_v4.py",
        sha256: "b".repeat(64),
      },
      task: {
        kind: "task",
        version: "v4",
        path: "env/task_v4.json",
        sha256: "c".repeat(64),
      },
    },
  },
  world_meta: { version: "v4" },
  task_meta: {},
  shared_summary: {
    terrain_kind: "plane",
    objects: ["landing-box"],
    zones: ["finish"],
    course_elements: 1,
    course_breakdown: { box: 1 },
    robot: "g1",
    project_capability_id: "g1",
    robot_matches_project: true,
  },
  goal: {},
  train_variations: [],
  clarifications: { answer_sources: {}, answers: [] },
};

const evaluatedPolicy: PolicySummary = {
  iter_index: 7,
  checkpoint: "checkpoint.pt",
  checkpoint_bytes: 4096,
  checkpoint_sha256: "d".repeat(64),
  deployable: true,
  artifact_purpose: "reproducibility",
  completion_authority: "attested",
  deployment_status: "qualified",
  deployment_blockers: [],
  physical_scene_status: "aligned",
  lineage_status: "verified",
  origin_receipt_sha256: "f".repeat(64),
  reference_clock_sha256: null,
  primary_metric: 0.8,
  fitness: 0.9,
  reward_version: "v6",
  metric_id: "landing_precision",
  metric_version: "v1",
  metric_source: "generated",
  metric_sha256: "e".repeat(64),
  criterion_status: "passed",
  evidence_status: "complete",
  route_evidence: null,
  contact_evidence: null,
  hold_evidence: null,
  objective_proof_status: "passed",
  objective_proof_blockers: [],
  lane_evidence_status: "unavailable",
  requested_evidence_env_index: null,
  resolved_evidence_env_index: null,
  resolved_episode_percentile: null,
  evidence_lane_selection: null,
  rollout_available: true,
  selected: true,
  selection_source: "objective_criterion",
};

beforeEach(() => {
  sessionStorage.clear();
  vi.mocked(getBehaviorDraft).mockResolvedValue({
    behavior_goal: "Evolve a certified cartwheel into a precise obstacle transition.",
  });
  vi.mocked(getSystemGpu).mockResolvedValue({
    cuda_available: true,
    mjlab_available: true,
    rsl_rl_available: true,
  } as never);
  vi.mocked(getSystemInfo).mockResolvedValue({
    anthropic_api_key_set: true,
  } as never);
  vi.mocked(getWorldSelection).mockResolvedValue(world);
  vi.mocked(getWorldValidate).mockResolvedValue({
    ok: true,
    errors: [],
    selection_version: 4,
    tuple_hash: "a".repeat(64),
  });
  vi.mocked(listModeRewards).mockResolvedValue({ promoted: null } as never);
  vi.mocked(listPolicies).mockResolvedValue([evaluatedPolicy]);
  vi.mocked(listProjectMetrics).mockResolvedValue([]);
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
});

function renderNewRun(onOpenWorld = vi.fn()) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <NewRunDialog
        slug={project.slug}
        project={project}
        onLaunched={vi.fn()}
        onOpenWorld={onOpenWorld}
      />
    </QueryClientProvider>,
  );
  return onOpenWorld;
}

test("does not claim direct-checkpoint migration passed before launch", () => {
  render(
    <PolicyInterfaceMigrationNotice
      startingPoint={checkpoint}
      eventProgram={EVENT_PROGRAM}
      hasReference={false}
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface admission");
  expect(notice).toHaveTextContent(
    "Policy interface admission · verified at launch",
  );
  expect(notice).toHaveTextContent(/3-wide one-hot event phase/i);
  expect(notice).not.toHaveTextContent(/every added actor and critic input column starts at zero/i);
  expect(notice).toHaveTextContent(/If the backend admits an exact supported extension/i);
  expect(notice).toHaveTextContent(/only then are its added columns zero-initialized/i);
  expect(notice).toHaveTextContent(/optimizer state and training counters are reset/i);
  expect(notice).toHaveTextContent(/exact target contract/i);
  expect(notice).toHaveTextContent(/event-phase, reference-clock, or combined migration/i);
  expect(notice).toHaveTextContent(/no import preflight receipt/i);
  expect(notice).toHaveTextContent(/verification happens when launch begins/i);
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
      hasReference={false}
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface admission");
  expect(notice).toHaveTextContent(/Interrupted snapshot admission · reverified at launch/i);
  expect(notice).toHaveTextContent(/remains unevaluated/i);
  expect(notice).toHaveTextContent(/opaque receipt id/i);
  expect(notice).toHaveTextContent(/backend's verified initialization receipt/i);
  expect(notice).toHaveTextContent(/optimizer state and training counters are reset/i);
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
      hasReference={false}
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface migration");
  expect(notice).toHaveTextContent(
    "Policy interface migration · reverified at launch",
  );
  expect(notice).toHaveTextContent(/Verified import migration type/i);
  expect(notice).toHaveTextContent("zero_initialized_event_phase_observation");
  expect(notice).toHaveTextContent(
    /inherited actor; the target critic starts fresh/i,
  );
  expect(notice).toHaveTextContent(/launch still revalidates/i);
});

test("does not claim zero initialization for an exact imported interface", () => {
  const exactImport: StartingPointSelection = {
    ...checkpoint,
    kind: "shared_skill",
    warm_start_iteration: null,
    starting_skill_id: "g1-schema4-exact",
    initialization_mode: "actor_critic",
    import_manifest_digest: "e".repeat(64),
    policy_contract_migration: null,
  };
  render(
    <PolicyInterfaceMigrationNotice
      startingPoint={exactImport}
      eventProgram={EVENT_PROGRAM}
      hasReference
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface admission");
  expect(notice).toHaveTextContent(/does not declare a migration type/i);
  expect(notice).not.toHaveTextContent(
    /verified import receipt declares these as added policy inputs/i,
  );
  expect(notice).not.toHaveTextContent(
    /every added actor and critic input column starts at zero/i,
  );
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
        hasReference={false}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  },
);

test("discloses a standalone schema-4 reference clock without an event program", () => {
  render(
    <PolicyInterfaceMigrationNotice
      startingPoint={{
        ...checkpoint,
        kind: "shared_skill",
        warm_start_iteration: null,
        starting_skill_id: "g1-cartwheel",
        initialization_mode: "actor_only",
        import_manifest_digest: "a".repeat(64),
        policy_contract_migration: {
          type: "zero_initialized_reference_clock_observation",
          from_schema: 3,
          to_schema: 4,
          observation_term: "reference_phase",
          extension_width: 1,
          reference_clock_sha256: "b".repeat(64),
          optimizer_resume: false,
        },
      }}
      eventProgram={undefined}
      hasReference
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface migration");
  expect(notice).toHaveTextContent(/one normalized reference playback phase/i);
  expect(notice).toHaveTextContent("reference_phase");
  expect(notice).toHaveTextContent(/schema 3 → 4/i);
  expect(notice).toHaveTextContent(/mean 0 and variance 1/i);
});

test("discloses the exact combined event and reference-clock migration", () => {
  const eventMigration = {
    type: "zero_initialized_event_phase_observation",
    from_schema: 2,
    to_schema: 3,
    observation_term: "authored_event_phase",
    extension_width: 3,
    ordered_phase_ids: ["route", "jump", "hold"],
    optimizer_resume: false as const,
  } satisfies PolicyContractMigration;
  const clockMigration = {
    type: "zero_initialized_reference_clock_observation",
    from_schema: 3,
    to_schema: 4,
    observation_term: "reference_phase",
    extension_width: 1,
    reference_clock_sha256: "c".repeat(64),
    optimizer_resume: false as const,
  } satisfies PolicyContractMigration;
  render(
    <PolicyInterfaceMigrationNotice
      startingPoint={{
        ...checkpoint,
        kind: "shared_skill",
        warm_start_iteration: null,
        starting_skill_id: "g1-route-cartwheel",
        initialization_mode: "actor_critic",
        import_manifest_digest: "d".repeat(64),
        policy_contract_migration: {
          type: "zero_initialized_observation_extensions",
          from_schema: 2,
          to_schema: 4,
          extension_width: 4,
          extensions: [eventMigration, clockMigration],
          optimizer_resume: false,
        },
      }}
      eventProgram={EVENT_PROGRAM}
      hasReference
    />,
  );

  const notice = screen.getByLabelText("Warm-start policy interface migration");
  expect(notice).toHaveTextContent(/3-wide one-hot event phase/i);
  expect(notice).toHaveTextContent(/one normalized reference playback phase/i);
  expect(notice).toHaveTextContent("zero_initialized_observation_extensions");
  expect(notice).toHaveTextContent("zero_initialized_event_phase_observation");
  expect(notice).toHaveTextContent("zero_initialized_reference_clock_observation");
  expect(notice).toHaveTextContent(/schema 2 → 4/i);
});

test("keeps exact identities inside one expandable launch receipt", () => {
  render(
    <ExactLaunchReceipt
      sections={[
        { title: "Policy", summary: "Imported actor.", details: [`manifest sha256: ${"a".repeat(64)}`] },
        {
          title: "Reference",
          summary: "Tier-D admitted.",
          details: [
            `execution contract sha256: ${"c".repeat(64)}`,
            `execution boundary sha256: ${"d".repeat(64)}`,
            `reference clock sha256: ${"e".repeat(64)}`,
            "active reward tracking backbone sha256: verified only at launch/runtime; not part of Tier-D motion certification",
            "not certified: root_xy_tracking",
          ],
        },
        { title: "Training environment", summary: "Tuple verified.", details: [`tuple sha256: ${"b".repeat(64)}`] },
        { title: "Objective fitness", summary: "Observe only.", details: ["effective authority: observe"] },
      ]}
    />,
  );

  const receipt = screen.getByLabelText("Exact launch receipt");
  expect(receipt.tagName).toBe("DETAILS");
  expect(receipt).not.toHaveAttribute("open");
  expect(screen.getByText("Exact launch receipt")).toBeInTheDocument();
  expect(screen.getByLabelText("Policy receipt")).toHaveTextContent("manifest sha256");
  expect(screen.getByLabelText("Reference receipt")).toHaveTextContent("root_xy_tracking");
  expect(screen.getByLabelText("Reference receipt")).toHaveTextContent(
    `execution contract sha256: ${"c".repeat(64)}`,
  );
  expect(screen.getByLabelText("Reference receipt")).toHaveTextContent(
    /tracking backbone sha256: verified only at launch\/runtime/i,
  );
});

test("persists clearing a reference as an explicit null draft", () => {
  const writes: Array<{ reference_clip_id: null; reference_robot: null }> = [];
  persistClearedReference((draft) => writes.push(draft));
  expect(writes).toEqual([{
    reference_clip_id: null,
    reference_robot: null,
  }]);
});

test("gives a repair path when a legacy imported skill has no manifest", () => {
  expect(startingSkillManifestIssue({
    ...checkpoint,
    kind: "shared_skill",
    warm_start_iteration: null,
    starting_skill_id: "legacy-policy",
    initialization_mode: "actor_only",
    import_manifest_digest: null,
  })).toMatch(/re-import the original \.rskill bundle.*cannot repair it/i);
});

test("ties the scratch policy, authored world receipt, and launch authority together", async () => {
  const user = userEvent.setup();
  const onOpenWorld = renderNewRun();

  await user.click(screen.getByRole("button", { name: "New run" }));
  const goal = await screen.findByPlaceholderText(
    "Run forward as fast as possible without falling.",
  );
  await waitFor(() => expect(screen.getByRole("button", { name: "Run pipeline check" })).toBeEnabled());
  expect(screen.getByLabelText("Objective fitness receipt")).toHaveTextContent(
    /Pipeline check is unscored and non-certifying; live training still requires an objective or an acknowledged blind ablation/i,
  );

  expect(screen.getByLabelText("Policy receipt")).toHaveTextContent(
    /Fresh actor and critic; no inherited policy bytes/i,
  );
  expect(screen.getByLabelText("Training environment receipt")).toHaveTextContent(
    new RegExp(`tuple sha256: ${"a".repeat(64)}`),
  );

  await user.clear(goal);
  await user.type(goal, "Preserve this draft while I inspect a different world.");
  await user.selectOptions(
    screen.getByLabelText("Primary objective fitness metric"),
    "g1_jump",
  );
  await user.click(screen.getByRole("button", { name: "Advanced" }));
  const evidenceEnvironment = screen.getByLabelText("Evidence environment");
  await user.clear(evidenceEnvironment);
  await user.type(evidenceEnvironment, "10");
  const trainingIterations = screen.getByLabelText(/rsl_rl iters/i);
  await user.clear(trainingIterations);
  await user.type(trainingIterations, "777");
  await user.click(screen.getByRole("button", { name: /Change environment/i }));
  expect(onOpenWorld).toHaveBeenCalledOnce();

  await user.click(screen.getByRole("button", { name: "New run" }));
  expect(await screen.findByText(/Run plan restored/i)).toBeInTheDocument();
  expect(screen.getByLabelText("Evidence environment")).toHaveValue(10);
  expect(screen.getByLabelText(/rsl_rl iters/i)).toHaveValue(777);
  await user.click(screen.getByRole("button", { name: "Basic" }));
  expect(await screen.findByPlaceholderText(
    "Run forward as fast as possible without falling.",
  )).toHaveValue(
    "Preserve this draft while I inspect a different world.",
  );
  expect(screen.getByLabelText("Primary objective fitness metric")).toHaveValue(
    "g1_jump",
  );
});

test("pins an evaluated project checkpoint digest before enabling launch", async () => {
  const user = userEvent.setup();
  renderNewRun();

  await user.click(screen.getByRole("button", { name: "New run" }));
  await user.click(await screen.findByRole("button", {
    name: "Choose policy or skill starting point",
  }));
  await user.click(screen.getByRole("radio", { name: /Project policy/i }));
  await waitFor(() => expect(screen.getByLabelText(/Evaluated iteration/i)).toHaveValue("7"));
  expect(screen.getByLabelText(/Iteration 7 evidence/i)).toHaveTextContent(
    /Checkpoint SHA-256: dddddddddd…dddddd/i,
  );
  await user.click(screen.getByRole("button", { name: /Use this starting point/i }));

  await waitFor(() => expect(screen.getByRole("button", { name: "Run pipeline check" })).toBeEnabled());
  expect(screen.getByLabelText("Policy receipt")).toHaveTextContent(
    new RegExp(`sha256 ${"d".repeat(64)}`),
  );
  expect(screen.queryByText(/sha256 is resolved server-side at launch/i))
    .not.toBeInTheDocument();
});

test("blocks launch when the active reward belongs to another reference", async () => {
  vi.mocked(getBehaviorDraft).mockResolvedValue({
    behavior_goal: "Evolve the certified hop into a low-rail traversal.",
    reference_clip_id: "selected-hop",
    reference_robot: "g1",
  });
  vi.mocked(listModeRewards).mockResolvedValue({
    mode_rewards: [],
    promoted: {
      version: 9,
      filename: "v9.py",
      clip_id: "other-hop",
      reference_robot: "g1",
      execution_context_digest: "1".repeat(64),
      context_current: true,
      selection_current: true,
      promotion_blocker: null,
      tracking_enabled: true,
      source_sha256: "2".repeat(64),
      source_filename: "mode_reward_v9.py",
      modes: [{ name: "flight", start_s: 0, end_s: 1, authored: true }],
      unauthored: [],
    },
  });
  vi.mocked(getReference).mockResolvedValue({
    index_row: {
      clip_id: "selected-hop", robot: "g1", text: "Selected hop",
      labels: ["hop"], tier: "D", license: "research", n_frames: 60,
      fps: 30, duration_s: 2, root_z_range: [0.6, 1.1], has_preview: true,
    },
    provenance: {},
    artifact_identity: {
      verified: true,
      clip_sha256: "3".repeat(64),
      provenance_clip_sha256: "3".repeat(64),
      source_content_sha256: "4".repeat(64),
      reason: null,
    },
    dynamics_admission: {
      admitted: true,
      tier: "D",
      certificate_digest: "5".repeat(64),
      clip_sha256: "3".repeat(64),
      artifact_hash_verified: true,
      rollout_sha256: "6".repeat(64),
      execution_contract_sha256: "7".repeat(64),
      execution_boundary_sha256: "8".repeat(64),
      reference_clock_sha256: "9".repeat(64),
      certification_scope: {
        schema: "reward-sculptor-tier-d-scope-v1",
        claim: "exact-schedule joint-position and root-height tracking",
        gated_evidence: [
          "mean_joint_position_error",
          "root_z_rmse",
          "duration_coverage",
          "non_vacuous_reference_motion",
          "beats_static_pose_baseline",
        ],
        measured_only: [
          "maximum_joint_position_error",
          "orientation_error",
          "motion_ratio",
        ],
        not_certified: [
          "root_xy_tracking",
          "contact_safety",
          "collision_avoidance",
          "general_dynamics_feasibility",
        ],
      },
      reason: null,
    },
  } as never);
  const user = userEvent.setup();
  renderNewRun();

  await user.click(screen.getByRole("button", { name: "New run" }));

  expect(await screen.findByText(/Launch is blocked because the active reward and selected motion disagree/i))
    .toHaveTextContent(/g1\/selected-hop/i);
  const launch = screen.getByRole("button", { name: "Inspect candidate" });
  expect(launch).toBeDisabled();
  expect(launch).toHaveAttribute(
    "title",
    "The active reward is bound to a different reference motion. Promote a reward for this exact robot and clip before launch.",
  );
  expect(screen.queryByText(/replaces it with a flat tracking reward/i))
    .not.toBeInTheDocument();
});

test("keeps reference inspection blocked when the selected motion query fails", async () => {
  vi.mocked(getBehaviorDraft).mockResolvedValue({
    behavior_goal: "Inspect this candidate without starting training.",
    reference_clip_id: "missing-candidate",
    reference_robot: "g1",
  });
  vi.mocked(getReference).mockRejectedValue(new Error("reference unavailable"));
  const user = userEvent.setup();
  renderNewRun();

  await user.click(screen.getByRole("button", { name: "New run" }));

  const launch = await screen.findByRole("button", { name: "Inspect candidate" });
  await waitFor(() => expect(launch).toHaveAttribute(
    "title",
    "The selected motion could not be loaded and its evidence cannot be verified.",
  ));
  expect(launch).toBeDisabled();
  expect(screen.getByText("tracking evidence unavailable")).toBeInTheDocument();
  expect(screen.getByText(/inspection and training remain blocked/i))
    .toBeInTheDocument();
});

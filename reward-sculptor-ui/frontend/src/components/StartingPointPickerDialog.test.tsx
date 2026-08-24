import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";

import { StartingPointPickerDialog } from "@/components/StartingPointPickerDialog";
import {
  listPolicyRecoverySnapshots,
  listStartingSkills,
  uploadStartingSkill,
} from "@/lib/api";
import type {
  PolicyRecoverySnapshot,
  PolicySummary,
  StartingPointSelection,
  StartingSkillReceipt,
} from "@/lib/types";

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    listPolicyRecoverySnapshots: vi.fn(),
    listStartingSkills: vi.fn(),
    uploadStartingSkill: vi.fn(),
  };
});

const receipt: StartingSkillReceipt = {
  skill: {
    skill_id: "g1-parkour",
    alias: "G1 parkour prior",
    created_at: "2026-08-17T12:00:00Z",
    adapter_class: "sculptor.adapters.mjlab.MjlabAdapter",
    task_id: "Mjlab-G1-Parkour",
    robot_slug: "g1",
    source: "imported",
    checkpoint_sha256: "a".repeat(64),
    checkpoint_size_bytes: 4096,
    checkpoint_format: "sanitized_pt",
    source_format: "safetensors",
    manifest_digest: "b".repeat(64),
    identity_digest: "c".repeat(64),
    source_weights_sha256: "d".repeat(64),
    reference_clip_id: "parkour-reference",
    reference_robot: "g1",
    reference_sha256: "e".repeat(64),
    reference_provenance_sha256: "f".repeat(64),
    world_bundle_sha256: "1".repeat(64),
    controller_sha256: "2".repeat(64),
    compatibility_contract: {
      observations: { shape: [48] },
      actions: { shape: [29] },
      policy: { actor: { hidden_dims: [512, 256] } },
    },
    compatibility_contract_digest: "3".repeat(64),
    compatibility_contract_provenance: {
      schema: 1,
      status: "origin_persisted",
      capabilities: {
        initialization_modes: ["actor_only", "actor_critic"],
        optimizer_resume: false,
        exact_resume: false,
      },
      evidence: {
        origin_policy_contract: {
          path: "provenance/origin_policy_contract.json",
          sha256: "5".repeat(64),
          bytes: 4096,
        },
      },
    },
    compatibility_contract_provenance_digest: "6".repeat(64),
    compatibility_contract_provenance_status: "origin_persisted",
    tensor_contract_verified: true,
    tensor_signature_sha256: "4".repeat(64),
    initialization_modes: ["actor_only", "reference_only"],
    policy_roles: ["actor"],
    trust_status: "sanitized",
  },
  compatible: true,
  selectable: true,
  training_authorized: false,
  authorization: {
    status: "candidate",
    receipt_scope: "structural_selectability_only",
    training_authorized: false,
    mode_gates: {
      actor_only: [
        "revalidate the current project policy contract at launch",
        "observe warm_start_loaded with the exact digest and actor role",
      ],
      reference_only: [
        "complete a separate Tier-D exact-schedule tracking evidence job for the exact clip and target execution boundary before live launch",
        "re-attest the exact clip, immutable provenance, rollout, certificate, execution contract, and boundary at launch",
      ],
    },
    detail: "Import/list admission makes this starting point selectable, not trainable. A reference requires a separate Tier-D exact-schedule tracking evidence job before live launch; launch only re-verifies that existing evidence.",
    policy_present: true,
  },
  compatibility: {
    status: "partially_compatible",
    allowed_initialization_modes: ["actor_only", "reference_only"],
    reasons: [],
    mode_reasons: {
      actor_critic: ["Critic observation contract differs from this project."],
      full_resume: ["Optimizer state is not portable."],
    },
  },
  trust: {
    status: "sanitized",
    detail: "Safetensors were checked and reserialized by the server.",
    source_format: "safetensors",
    checkpoint_format: "sanitized_pt",
    manifest_digest: "b".repeat(64),
    checkpoint_sha256: "a".repeat(64),
    compatibility_contract_digest: "3".repeat(64),
    tensor_contract_verified: true,
    tensor_signature_sha256: "4".repeat(64),
  },
  components: {
    policy_roles: ["actor"],
    reference: {
      clip_id: "parkour-reference",
      robot: "g1",
      admission: {
        status: "registered_candidate",
        structural_checks: ["bounded arrays"],
        training_authorized: false,
        next_gate: "Run a separate target-specific Tier-D exact-schedule tracking evidence job before live launch; launch only re-verifies the resulting exact evidence.",
      },
    },
    world: {
      included: true,
      status: "digest_recorded_bytes_discarded",
      bytes_retained: false,
      activatable: false,
    },
    controller: {
      kind: "reference_tracker",
      status: "digest_recorded_bytes_discarded",
      bytes_retained: false,
      activatable: false,
    },
    excluded: [
      "raw checkpoints, pickle, Python, TorchScript, native binaries, and unknown members are rejected before admission",
    ],
  },
  warnings: ["Bundled world digest was recorded but its bytes were not retained."],
};

const referenceOnlyReceipt: StartingSkillReceipt = {
  ...receipt,
  skill: {
    ...receipt.skill,
    skill_id: "g1-motion-only",
    alias: "G1 motion only",
    checkpoint_sha256: null,
    checkpoint_size_bytes: null,
    checkpoint_format: "none",
    source_format: "npz",
    initialization_modes: ["reference_only"],
    policy_roles: [],
    trust_status: "validated",
  },
  compatibility: {
    status: "reference_only",
    allowed_initialization_modes: ["reference_only"],
    reasons: [],
  },
  trust: {
    ...receipt.trust,
    status: "validated",
    detail: "Bounded trajectory arrays and provenance were validated.",
    source_format: "npz",
    checkpoint_format: "none",
    checkpoint_sha256: null,
  },
  components: {
    ...receipt.components,
    policy_roles: [],
    world: {
      included: false,
      status: "absent",
      bytes_retained: false,
      activatable: false,
    },
    controller: null,
  },
  warnings: [],
};

const scratch: StartingPointSelection = {
  kind: "scratch",
  warm_start_iteration: null,
  starting_skill_id: null,
  initialization_mode: null,
  reference_clip_id: null,
  reference_robot: null,
  import_manifest_digest: null,
  compatibility_contract_provenance_status: null,
  acknowledge_legacy_reconstructed_initialization: false,
};

function renderPicker(
  onChange = vi.fn(),
  checkpoints: PolicySummary[] = [],
  projectRobot = "g1",
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <StartingPointPickerDialog
        slug="g1-parkour"
        projectRobot={projectRobot}
        projectTaskId="Mjlab-G1-Parkour"
        checkpoints={checkpoints}
        value={scratch}
        onChange={onChange}
        onClose={vi.fn()}
      />
    </QueryClientProvider>,
  );
  return onChange;
}

test("describes scratch as a policy-only reset while preserving independent inputs", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  renderPicker();

  expect(screen.getByText(/No policy weights are imported/i)).toBeInTheDocument();
  expect(screen.getByText(/Starting motion and Training environment/i))
    .toHaveTextContent(/remain separate choices and are not cleared here/i);
  expect(screen.queryByText(/No imported weights, motion, controller, or world/i))
    .not.toBeInTheDocument();
});

test("supports conventional arrow-key navigation across custom radio cards", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  vi.mocked(listPolicyRecoverySnapshots).mockResolvedValue([]);
  const user = userEvent.setup();
  renderPicker();

  const scratch = screen.getByRole("radio", { name: /From scratch/i });
  const project = screen.getByRole("radio", { name: /Project policy/i });
  scratch.focus();
  await user.keyboard("{ArrowRight}");
  expect(project).toHaveFocus();
  expect(project).toBeChecked();

  const completed = screen.getByRole("radio", { name: /Evaluated policies/i });
  completed.focus();
  await user.keyboard("{ArrowDown}");
  const interrupted = screen.getByRole("radio", { name: /Interrupted or unevaluated/i });
  expect(interrupted).toHaveFocus();
  expect(interrupted).toBeChecked();

  await user.keyboard("{Home}");
  expect(completed).toHaveFocus();
  expect(completed).toBeChecked();
});

test("makes policy, motion, world, trust, and initialization semantics explicit", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [receipt] });
  const user = userEvent.setup();
  const onChange = renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", { name: /G1 parkour prior/i }));

  expect(screen.getByText("Starting policy")).toBeInTheDocument();
  expect(screen.getByText("Starting motion")).toBeInTheDocument();
  expect(screen.getByText("Training environment")).toBeInTheDocument();
  expect(screen.getByText("Controller source")).toBeInTheDocument();
  expect(screen.getByText(/partially compatible/i)).toBeInTheDocument();
  expect(screen.getByText(/does not authorize training/i)).toBeInTheDocument();
  expect(screen.getByText(/warm_start_loaded with the exact digest/i)).toBeInTheDocument();
  expect(screen.getByText("Source digest only")).toBeInTheDocument();
  expect(screen.getByText(/content-attested|trusted/i)).toBeInTheDocument();
  expect(screen.getAllByText(/registered candidate/i)).toHaveLength(2);
  expect(screen.getByText(
    /Run a separate target-specific Tier-D exact-schedule tracking evidence job/i,
  )).toBeInTheDocument();
  expect(screen.getAllByText(/launch only re-verifies/i).length).toBeGreaterThan(0);
  expect(screen.getByText("obs 48 / actions 29 / actor 512 -> 256")).toBeInTheDocument();
  expect(screen.getByLabelText("Tensor contract verified")).toBeInTheDocument();
  expect(screen.getByRole("radio", { name: /Actor \+ critic/i })).toBeDisabled();
  expect(screen.getByRole("radio", { name: /Actor \+ critic/i })).toHaveTextContent(
    /Critic observation contract differs/i,
  );
  expect(screen.getByRole("radio", { name: /Full training resume/i })).toBeDisabled();

  await user.click(screen.getByRole("button", { name: /Use this starting point/i }));
  expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
    kind: "shared_skill",
    starting_skill_id: "g1-parkour",
    initialization_mode: "actor_only",
    reference_clip_id: null,
    reference_robot: null,
    import_manifest_digest: "b".repeat(64),
  }));
});

test("fails legacy receipts closed without crashing the picker", async () => {
  const legacyReceipt: StartingSkillReceipt = {
    ...receipt,
    skill: {
      ...receipt.skill,
      skill_id: "legacy-g1-prior",
      alias: "Legacy G1 prior",
    },
    selectable: false,
    authorization: undefined,
  };
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [legacyReceipt] });
  const user = userEvent.setup();
  renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", { name: /Legacy G1 prior/i }));

  expect(screen.getByText(/legacy receipt predates launch-authorization evidence/i))
    .toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Use this starting point/i }))
    .toBeDisabled();
});

test("requires explicit acknowledgement for a reconstructed policy contract", async () => {
  const reconstructedReceipt: StartingSkillReceipt = {
    ...receipt,
    skill: {
      ...receipt.skill,
      skill_id: "iter38-reconstructed",
      alias: "Iter 38 reconstructed policy",
      compatibility_contract_provenance: {
        ...receipt.skill.compatibility_contract_provenance!,
        status: "legacy_reconstructed",
      },
      compatibility_contract_provenance_digest: "7".repeat(64),
      compatibility_contract_provenance_status: "legacy_reconstructed",
    },
  };
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [reconstructedReceipt] });
  const user = userEvent.setup();
  const onChange = renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", {
    name: /Iter 38 reconstructed policy/i,
  }));

  expect(screen.getByText("Historical contract reconstruction")).toBeInTheDocument();
  expect(screen.getByText(/not exact resume or optimizer restoration/i))
    .toBeInTheDocument();
  const apply = screen.getByRole("button", { name: /Use this starting point/i });
  expect(apply).toBeDisabled();

  await user.click(screen.getByRole("checkbox", {
    name: /reconstructed after training from retained evidence/i,
  }));
  expect(apply).toBeEnabled();
  await user.click(apply);

  expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
    starting_skill_id: "iter38-reconstructed",
    initialization_mode: "actor_only",
    compatibility_contract_provenance_status: "legacy_reconstructed",
    acknowledge_legacy_reconstructed_initialization: true,
  }));
});

test.each(["controller.pt", "deployment.zip"])(
  "rejects non-portable %s artifacts before upload",
  async (filename) => {
    vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
    vi.mocked(uploadStartingSkill).mockClear();
    // Exercise the component's own fail-closed validation. Browsers normally
    // filter this extension via `accept`, but drag/drop and crafted events can
    // still deliver it.
    const user = userEvent.setup({ applyAccept: false });
    renderPicker();

    await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();
    await user.upload(input!, new File(["untrusted"], filename));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Deployment \.zip bundles and raw checkpoints are not portable imports/i,
    );
    await waitFor(() => expect(uploadStartingSkill).not.toHaveBeenCalled());
  },
);

test("keeps bundled motion opt-in for policy transfer", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [receipt] });
  const user = userEvent.setup();
  const onChange = renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", { name: /G1 parkour prior/i }));
  expect(screen.getByRole("checkbox", { name: /Attach bundled motion/i })).not.toBeChecked();
  expect(screen.getByText(/bundled, not selected/i)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /Use this starting point/i }));

  expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
    starting_skill_id: "g1-parkour",
    initialization_mode: "actor_only",
    reference_clip_id: null,
    reference_robot: null,
  }));
});

test("attaches bundled motion only after an explicit opt-in", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [receipt] });
  const user = userEvent.setup();
  const onChange = renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", { name: /G1 parkour prior/i }));
  await user.click(screen.getByRole("checkbox", { name: /Attach bundled motion/i }));
  await user.click(screen.getByRole("button", { name: /Use this starting point/i }));

  expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
    reference_clip_id: "parkour-reference",
    reference_robot: "g1",
  }));
});

const checkpoint = (
  iterIndex: number,
  overrides: Partial<PolicySummary> = {},
): PolicySummary => ({
  iter_index: iterIndex,
  checkpoint: "checkpoint.pt",
  checkpoint_bytes: 4096,
  checkpoint_sha256: "f".repeat(64),
  deployable: false,
  primary_metric: null,
  fitness: null,
  reward_version: "v1",
  metric_id: null,
  metric_version: null,
  metric_source: null,
  metric_sha256: null,
  criterion_status: "not_recorded",
  evidence_status: "unavailable",
  route_evidence: null,
  contact_evidence: null,
  hold_evidence: null,
  objective_proof_status: "incomplete",
  objective_proof_blockers: ["objective evidence unavailable"],
  lane_evidence_status: "unavailable",
  requested_evidence_env_index: null,
  resolved_evidence_env_index: null,
  resolved_episode_percentile: null,
  evidence_lane_selection: null,
  rollout_available: false,
  selected: false,
  selection_source: null,
  ...overrides,
});

test("keeps unevaluated checkpoints out of the evaluated policy picker", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  const user = userEvent.setup();
  renderPicker(vi.fn(), [checkpoint(4), checkpoint(5)]);

  await user.click(screen.getByRole("radio", { name: /Project policy/i }));

  expect(screen.getByText(/No evaluated policy is available/i)).toBeInTheDocument();
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
  expect(screen.getByText(/only after server attestation/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Use this starting point/i })).toBeDisabled();
});

test("prefers an older evidenced selection over a newer failed checkpoint", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  const user = userEvent.setup();
  const onChange = renderPicker(vi.fn(), [
    checkpoint(4, {
      deployable: true,
      fitness: 0.94,
      metric_id: "weave-stop",
      metric_version: "v3",
      metric_source: "generated",
      metric_sha256: "c".repeat(64),
      criterion_status: "passed",
      evidence_status: "complete",
      route_evidence: {
        key: "order_ok_frac", value: 1, kind: "fraction",
        comparison: "gte", threshold: 1, passed: true,
        semantics_source: "builtin:order_ok_frac",
      },
      contact_evidence: {
        key: "contact_frac", value: 0, kind: "fraction",
        comparison: "lte", threshold: 0, passed: true,
        semantics_source: "builtin:contact_frac",
      },
      hold_evidence: {
        key: "ch_hold", value: 1, kind: "score",
        comparison: "gte", threshold: 1, passed: true,
        semantics_source: "builtin:ch_hold",
      },
      rollout_available: true,
      selected: true,
      selection_source: "objective_criterion",
    }),
    checkpoint(5, {
      fitness: 0.98,
      criterion_status: "failed",
    }),
  ]);

  await user.click(screen.getByRole("radio", { name: /Project policy/i }));

  expect(screen.getByLabelText(/Evaluated iteration/i)).toHaveValue("4");
  expect(screen.queryByRole("option", { name: /Iteration 5/i })).not.toBeInTheDocument();
  expect(screen.getByLabelText(/Iteration 4 evidence/i)).toHaveTextContent("weave-stop / v3");
  expect(screen.getByLabelText(/Iteration 4 evidence/i)).toHaveTextContent("order_ok_frac");
  expect(screen.getByLabelText(/Iteration 4 evidence/i)).toHaveTextContent("objective_criterion");
  await user.click(screen.getByRole("button", { name: /Use this starting point/i }));
  expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
    kind: "project_checkpoint",
    warm_start_iteration: 4,
  }));
});

test("keeps a server-completed generic policy selectable without showcase evidence keys", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  const user = userEvent.setup();
  renderPicker(vi.fn(), [checkpoint(3, {
    deployable: true,
    rollout_available: true,
    evidence_status: "unavailable",
  })]);

  await user.click(screen.getByRole("radio", { name: /Project policy/i }));

  await user.selectOptions(screen.getByLabelText(/Evaluated iteration/i), "3");
  expect(screen.getByLabelText(/Evaluated iteration/i)).toHaveValue("3");
  expect(screen.getByRole("button", { name: /Use this starting point/i })).toBeEnabled();
});

const recoverySnapshot = (
  overrides: Partial<PolicyRecoverySnapshot> = {},
): PolicyRecoverySnapshot => ({
  snapshot_id: "snap_7fd3a41b",
  iteration: 2,
  ppo_step: 50,
  source_job_id: "job_d41e199695d2d7d8",
  source_job_status: "errored",
  last_observed_ppo_iteration: 58,
  checkpoint_bytes: 6_202_705,
  checkpoint_sha256: "a".repeat(64),
  receipt_digest: "b".repeat(64),
  provenance_status: "origin_persisted",
  selectable: true,
  blocker: null,
  ...overrides,
});

test("separates interrupted PPO snapshots and requires an explicit unevaluated acknowledgement", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  vi.mocked(listPolicyRecoverySnapshots).mockResolvedValue([recoverySnapshot()]);
  const user = userEvent.setup();
  const onChange = renderPicker();

  await user.click(screen.getByRole("radio", { name: /Project policy/i }));
  expect(screen.getByRole("radio", { name: /Evaluated policies/i })).toBeChecked();
  await user.click(screen.getByRole("radio", { name: /Interrupted or unevaluated/i }));

  const row = await screen.findByRole("radio", {
    name: /Cycle 2, PPO snapshot 50, unevaluated recovery input/i,
  });
  expect(row).toHaveAccessibleDescription(/actor \+ critic transfer/i);
  expect(row).toHaveAccessibleDescription(/optimizer\/counters reset/i);
  expect(row).toHaveTextContent(/actor \+ critic transfer/i);
  expect(row).toHaveTextContent(/optimizer\/counters reset/i);
  expect(screen.getByRole("button", { name: /Use this starting point/i })).toBeDisabled();

  await user.click(row);
  expect(screen.getByLabelText(/Cycle 2 PPO snapshot 50 recovery disclosure/i))
    .toHaveTextContent(/no rollout, objective score, criterion result, or success claim/i);
  const acknowledgement = screen.getByRole("checkbox", {
    name: /snapshot is unevaluated and may be worse/i,
  });
  await user.click(acknowledgement);
  const apply = screen.getByRole("button", { name: /Use this starting point/i });
  expect(apply).toBeEnabled();
  await user.click(apply);

  const selection = onChange.mock.calls[0][0] as StartingPointSelection;
  expect(selection).toMatchObject({
    kind: "project_checkpoint",
    warm_start_iteration: null,
    warm_start_snapshot: {
      snapshot_id: "snap_7fd3a41b",
      checkpoint_sha256: "a".repeat(64),
      receipt_digest: "b".repeat(64),
      acknowledge_interrupted_snapshot: true,
    },
  });
  expect(selection.warm_start_snapshot).not.toHaveProperty("checkpoint");
  expect(selection.warm_start_snapshot).not.toHaveProperty("path");
});

test("fails closed when interrupted snapshot discovery cannot be verified", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  vi.mocked(listPolicyRecoverySnapshots).mockRejectedValue(
    new Error("attestation store unavailable"),
  );
  const user = userEvent.setup();
  renderPicker();

  await user.click(screen.getByRole("radio", { name: /Project policy/i }));
  await user.click(screen.getByRole("radio", { name: /Interrupted or unevaluated/i }));

  expect(await screen.findByRole("alert")).toHaveTextContent(
    /No manual path or iteration fallback is allowed/i,
  );
  expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Use this starting point/i })).toBeDisabled();
});

test("requires a separate acknowledgement for a legacy-reconstructed snapshot receipt", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  vi.mocked(listPolicyRecoverySnapshots).mockResolvedValue([
    recoverySnapshot({ provenance_status: "legacy_reconstructed" }),
  ]);
  const user = userEvent.setup();
  renderPicker();

  await user.click(screen.getByRole("radio", { name: /Project policy/i }));
  await user.click(screen.getByRole("radio", { name: /Interrupted or unevaluated/i }));
  await user.click(await screen.findByRole("radio", {
    name: /Cycle 2, PPO snapshot 50/i,
  }));
  await user.click(screen.getByRole("checkbox", {
    name: /snapshot is unevaluated and may be worse/i,
  }));
  const apply = screen.getByRole("button", { name: /Use this starting point/i });
  expect(apply).toBeDisabled();
  await user.click(screen.getByRole("checkbox", {
    name: /receipt was reconstructed after the interruption/i,
  }));
  expect(apply).toBeEnabled();
});

test("admits validated reference-only data without granting policy loading", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [referenceOnlyReceipt] });
  const user = userEvent.setup();
  const onChange = renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", { name: /G1 motion only/i }));

  expect(screen.getByText(
    /Imports register candidates; they do not certify a reference for live training/i,
  )).toBeInTheDocument();
  expect(screen.getAllByText(
    /separate Tier-D exact-schedule tracking evidence job/i,
  ).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/launch only re-verifies/i).length).toBeGreaterThan(0);
  expect(screen.getByRole("radio", { name: /^Motion only/i })).toBeEnabled();
  expect(screen.getByRole("radio", { name: /^Motion only/i })).toBeChecked();
  expect(screen.getByRole("radio", { name: /Actor only/i })).toBeDisabled();
  expect(screen.getByRole("radio", { name: /Actor \+ critic/i })).toBeDisabled();

  await user.click(screen.getByRole("button", { name: /Use this starting point/i }));
  expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
    starting_skill_id: "g1-motion-only",
    initialization_mode: "reference_only",
    reference_clip_id: "parkour-reference",
    reference_robot: "g1",
  }));
});

test("validated trust never authorizes actor loading", async () => {
  const invalidValidatedPolicy: StartingSkillReceipt = {
    ...receipt,
    skill: { ...receipt.skill, trust_status: "validated" },
    trust: { ...receipt.trust, status: "validated" },
  };
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [invalidValidatedPolicy] });
  const user = userEvent.setup();
  renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", { name: /G1 parkour prior/i }));

  expect(screen.getByText("blocked")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Use this starting point/i })).toBeDisabled();
});

test("a selectable receipt must pin the stored immutable manifest", async () => {
  const staleReceipt: StartingSkillReceipt = {
    ...receipt,
    skill: { ...receipt.skill, manifest_digest: null },
    // A stale trust display value must not substitute for the stored record
    // digest that the route and worker re-resolve.
    trust: { ...receipt.trust, manifest_digest: "b".repeat(64) },
  };
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [staleReceipt] });
  const user = userEvent.setup();
  renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  await user.click(await screen.findByRole("radio", { name: /G1 parkour prior/i }));

  expect(screen.getByText("blocked")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Use this starting point/i })).toBeDisabled();
});

test("keeps blocked candidates inspectable without claiming the row is disabled", async () => {
  const blockedReceipt: StartingSkillReceipt = {
    ...receipt,
    compatible: false,
    selectable: false,
    authorization: {
      ...receipt.authorization!,
      status: "blocked",
      detail: "No initialization mode currently passes structural admission.",
    },
    compatibility: {
      status: "incompatible",
      allowed_initialization_modes: [],
      reasons: [
        "project_robot_unresolved: select a project robot before using this skill",
      ],
      reason_codes: ["project_robot_unresolved"],
    },
  };
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [blockedReceipt] });
  const user = userEvent.setup();
  renderPicker();

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));
  const row = await screen.findByRole("radio", { name: /G1 parkour prior/i });
  expect(row).not.toHaveAttribute("aria-disabled");
  await user.click(row);

  expect(screen.getByText(/project_robot_unresolved/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Use this starting point/i })).toBeDisabled();
});

test("uses the canonical project robot in portable export examples", async () => {
  vi.mocked(listStartingSkills).mockResolvedValue({ skills: [] });
  const user = userEvent.setup();
  renderPicker(vi.fn(), [], "go1");

  await user.click(screen.getByRole("radio", { name: /Imported skill/i }));

  expect(screen.getByText(/sculpt export --portable --robot go1/)).toBeInTheDocument();
  expect(screen.getByText(/sculpt refs export-skill --robot go1/)).toBeInTheDocument();
});

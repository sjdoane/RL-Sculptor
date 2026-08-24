import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import type { ProjectDetail, RefDetail } from "@/lib/types";

const mocks = vi.hoisted(() => ({
  getReference: vi.fn(),
  listModeRewards: vi.fn(),
  useBehaviorDraft: vi.fn(),
  useSaveBehaviorDraft: vi.fn(),
  usePolicies: vi.fn(),
  useRewards: vi.fn(),
  useHasActiveRun: vi.fn(),
  useWorldSelection: vi.fn(),
}));

vi.mock("@/components/ReferencePickerDialog", () => ({
  ReferencePickerDialog: () => null,
}));
vi.mock("@/hooks/useBehaviorDraft", () => ({
  useBehaviorDraft: mocks.useBehaviorDraft,
  useSaveBehaviorDraft: mocks.useSaveBehaviorDraft,
}));
vi.mock("@/hooks/usePolicies", () => ({ usePolicies: mocks.usePolicies }));
vi.mock("@/hooks/useRewards", () => ({ useRewards: mocks.useRewards }));
vi.mock("@/hooks/useRuns", () => ({
  useHasActiveRun: mocks.useHasActiveRun,
}));
vi.mock("@/hooks/useWorlds", () => ({
  useWorldSelection: mocks.useWorldSelection,
}));
vi.mock("@/lib/api", () => ({
  getReference: mocks.getReference,
  listModeRewards: mocks.listModeRewards,
}));

import {
  BehaviorFlow,
  tierDCertificationCommands,
} from "@/components/BehaviorFlow";

const PROJECT = {
  slug: "g1-research",
  display_name: "G1 research",
  description: "Adapt a cartwheel into a difficult recovery task.",
  status: "ready",
  created_at: "2026-08-23T00:00:00Z",
  env_id: "Mjlab-G1-v0",
  n_iterations_completed: 0,
  project_dir: "/research/projects/g1-research",
  adapter_class: "sculptor.adapters.mjlab:MjlabAdapter",
  adapter_config: {},
  library_slug: "g1",
  reference_robot: "g1",
} satisfies ProjectDetail;

function referenceDetail(admitted: boolean): RefDetail {
  return {
    index_row: {
      clip_id: "cartwheel-composite",
      robot: "g1",
      text: "Cartwheel",
      labels: ["cartwheel"],
      tier: admitted ? "D" : "K",
      license: "research",
      n_frames: 180,
      fps: 30,
      duration_s: 6,
      root_z_range: [0.45, 1.2],
      has_preview: true,
    },
    provenance: {},
    artifact_identity: {
      verified: true,
      clip_sha256: "b".repeat(64),
      provenance_clip_sha256: "b".repeat(64),
      source_content_sha256: "d".repeat(64),
      reason: null,
    },
    dynamics_admission: {
      admitted,
      tier: admitted ? "D" : "K",
      certificate_digest: admitted ? "a".repeat(64) : null,
      clip_sha256: admitted ? "b".repeat(64) : null,
      artifact_hash_verified: true,
      rollout_sha256: admitted ? "c".repeat(64) : null,
      execution_contract_sha256: admitted ? "e".repeat(64) : null,
      execution_boundary_sha256: admitted ? "f".repeat(64) : null,
      reference_clock_sha256: admitted ? "0".repeat(64) : null,
      certification_scope: admitted ? {
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
      } : null,
      reason: admitted ? null : "no Tier-D certificate is present",
    },
  };
}

function renderFlow() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <BehaviorFlow
        slug={PROJECT.slug}
        project={PROJECT}
        robotConfigured
        onGoTo={vi.fn()}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  mocks.listModeRewards.mockResolvedValue({
    mode_rewards: [],
    promoted: null,
  });
  mocks.useWorldSelection.mockReturnValue({
    data: { selection: { selection_version: 1 } },
    isLoading: false,
  });
  mocks.useRewards.mockReturnValue({ data: [], isLoading: false });
  mocks.usePolicies.mockReturnValue({ data: [], isLoading: false });
  mocks.useHasActiveRun.mockReturnValue(false);
  mocks.useSaveBehaviorDraft.mockReturnValue({ mutate: vi.fn() });
  mocks.useBehaviorDraft.mockReturnValue({
    data: {
      behavior_goal: "Adapt a cartwheel into a difficult recovery task.",
      reference_clip_id: "cartwheel-composite",
      reference_robot: "g1",
    },
  });
});

describe("BehaviorFlow reference-motion readiness", () => {
  test("keeps a Tier-K draft visibly blocked behind external certification", async () => {
    mocks.getReference.mockResolvedValue(referenceDetail(false));
    const user = userEvent.setup();

    renderFlow();

    expect(screen.getByText("Certify motion")).toBeInTheDocument();
    expect(await screen.findByText(/Tier K candidate · live training blocked/))
      .toHaveTextContent("no Tier-D certificate is present");
    expect(screen.getByText(/It becomes an immutable tracking candidate/))
      .toHaveTextContent(
        /the next step checks Tier-D exact-schedule tracking evidence/i,
      );
    expect(screen.getByText(/This UI verifies evidence but does not create certificates/i))
      .toBeInTheDocument();
    expect(screen.getByText(/Choose an existing trusted donor project/i))
      .toBeInTheDocument();
    expect(screen.queryByLabelText("External certification command"))
      .not.toBeInTheDocument();

    await user.type(
      screen.getByLabelText("Trusted Tier-D donor project"),
      "/research/projects/g1-trusted-locomotion",
    );
    expect(screen.getByLabelText("External certification command"))
      .toHaveTextContent(
        "sculpt refs export-tierd-interface --donor-project "
        + "'/research/projects/g1-trusted-locomotion'",
      );
    expect(screen.getByLabelText("External certification command"))
      .toHaveTextContent("sculpt refs track --clip-id 'cartwheel-composite' --robot 'g1'");
    expect(screen.getByLabelText("External certification command"))
      .toHaveTextContent("--dry-run");
    expect(screen.getByLabelText("External certification command"))
      .not.toHaveTextContent(`--donor-project '${PROJECT.project_dir}'`);
    expect(screen.getByRole("button", { name: "Refresh status" }))
      .toBeEnabled();
    expect(mocks.getReference).toHaveBeenCalledWith("g1", "cartwheel-composite");
  });

  test("never treats the fresh target project as its own Tier-D donor", () => {
    const result = tierDCertificationCommands({
      clipId: "cartwheel-composite",
      robot: "g1",
      donorProject: PROJECT.project_dir,
      targetProject: PROJECT.project_dir,
    });

    expect(result.commands).toEqual([]);
    expect(result.error).toMatch(/new target project is not a donor/i);
  });

  test("shows exact Tier-D evidence without claiming the UI issued it", async () => {
    mocks.getReference.mockResolvedValue(referenceDetail(true));

    renderFlow();

    expect(await screen.findByText(/Tier-D exact-schedule tracking evidence verified/)).toHaveTextContent(
      "certificate aaaaaaaaaaaa · rollout cccccccccccc",
    );
    expect(screen.getByText(/Tier-D evidence was earned by an external exact-schedule tracking job/i))
      .toHaveTextContent(
        /does not certify root-XY tracking, contact safety, collision avoidance, general dynamics feasibility/i,
      );
    expect(screen.queryByLabelText("External certification command"))
      .not.toBeInTheDocument();
    const receipt = screen.getByText(
      "Exact Tier-D tracking receipt",
    ).closest("details");
    expect(receipt).toHaveTextContent(`robot=g1`);
    expect(receipt).toHaveTextContent(`clip_id=cartwheel-composite`);
    expect(receipt).toHaveTextContent(
      `artifact_clip_sha256=${"b".repeat(64)}`,
    );
    expect(receipt).toHaveTextContent(`raw_source_sha256=${"d".repeat(64)}`);
    expect(receipt).toHaveTextContent(`certificate_digest=${"a".repeat(64)}`);
    expect(receipt).toHaveTextContent(`rollout_sha256=${"c".repeat(64)}`);
    expect(screen.getByRole("button", { name: "Re-check evidence" }))
      .toBeEnabled();
  });

  test("fails closed when an admitted response lacks an exact digest receipt", async () => {
    const incomplete = referenceDetail(true);
    incomplete.dynamics_admission.rollout_sha256 = null;
    mocks.getReference.mockResolvedValue(incomplete);

    renderFlow();

    expect(await screen.findByText(/Tier D evidence is incomplete/))
      .toHaveTextContent(/live training blocked/i);
    expect(screen.getByLabelText("Trusted Tier-D donor project"))
      .toBeInTheDocument();
    expect(screen.queryByLabelText("External certification command"))
      .not.toBeInTheDocument();
    await userEvent.type(
      screen.getByLabelText("Trusted Tier-D donor project"),
      "/research/projects/g1-trusted-locomotion",
    );
    expect(screen.getByLabelText("External certification command"))
      .toBeInTheDocument();
    expect(screen.queryByText("Exact Tier-D receipt")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh status" })).toBeEnabled();
  });

  test("fails closed when artifact bytes do not match the admitted clip", async () => {
    const mismatched = referenceDetail(true);
    mismatched.artifact_identity!.clip_sha256 = "e".repeat(64);
    mocks.getReference.mockResolvedValue(mismatched);

    renderFlow();

    expect(await screen.findByText(/Tier D evidence is incomplete/))
      .toHaveTextContent(/live training blocked/i);
    await userEvent.type(
      screen.getByLabelText("Trusted Tier-D donor project"),
      "/research/projects/g1-trusted-locomotion",
    );
    expect(screen.getByLabelText("External certification command"))
      .toBeInTheDocument();
    expect(screen.queryByText("Exact Tier-D receipt")).not.toBeInTheDocument();
  });

  test("describes the implemented fixed-window runtime and omits a gate without a motion", async () => {
    mocks.useBehaviorDraft.mockReturnValue({ data: {} });

    renderFlow();

    expect(await screen.findByText(/Runtime dispatch follows immutable episode-time windows/))
      .toHaveTextContent(/transition guards are inspectable metadata only/i);
    expect(screen.queryByText("Certify motion")).not.toBeInTheDocument();
    expect(screen.queryByText(/advances only when its compiled transition predicate/i))
      .not.toBeInTheDocument();
    expect(mocks.getReference).not.toHaveBeenCalled();
  });
});

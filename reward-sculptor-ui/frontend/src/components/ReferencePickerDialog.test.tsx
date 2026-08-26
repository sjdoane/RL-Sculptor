import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import type { RefDetail, RefIndexRow } from "@/lib/types";

const mocks = vi.hoisted(() => ({
  getReference: vi.fn(),
  useAttachStageReference: vi.fn(),
  useReferenceIndex: vi.fn(),
  useReferenceSearch: vi.fn(),
}));

vi.mock("@/components/ComposeMotionDialog", () => ({
  ComposeMotionDialog: () => null,
}));
vi.mock("@/hooks/useReferences", () => ({
  useAttachStageReference: mocks.useAttachStageReference,
  useReferenceIndex: mocks.useReferenceIndex,
  useReferenceSearch: mocks.useReferenceSearch,
}));
vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return { ...original, getReference: mocks.getReference };
});

import { ReferencePickerDialog } from "@/components/ReferencePickerDialog";

const row: RefIndexRow = {
  clip_id: "four-rail-hop",
  robot: "g1",
  text: "Four-rail traveling hop",
  labels: ["hop"],
  tier: "D",
  license: "cc-by-4.0",
  n_frames: 229,
  fps: 60,
  duration_s: 3.8,
  root_z_range: [0, 0.16],
  has_preview: true,
};

function detail({
  admitted = true,
  complete = true,
  reason = null,
}: {
  admitted?: boolean;
  complete?: boolean;
  reason?: string | null;
} = {}): RefDetail {
  return {
    index_row: row,
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
      tier: "D",
      certificate_digest: complete ? "a".repeat(64) : null,
      clip_sha256: "b".repeat(64),
      artifact_hash_verified: true,
      rollout_sha256: complete ? "c".repeat(64) : null,
      execution_contract_sha256: complete ? "e".repeat(64) : null,
      execution_boundary_sha256: complete ? "f".repeat(64) : null,
      reference_clock_sha256: complete ? "0".repeat(64) : null,
      certification_scope: complete ? {
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
      reason,
    },
  };
}

function renderPicker(onPick = vi.fn()) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <ReferencePickerDialog
        slug="g1-reference-hop"
        currentClipId={row.clip_id}
        robot="g1"
        onPick={onPick}
        onClose={vi.fn()}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  mocks.useAttachStageReference.mockReturnValue({
    isPending: false,
    mutate: vi.fn(),
  });
  mocks.useReferenceSearch.mockReturnValue({
    data: [], isLoading: false, isError: false, isFetching: false,
  });
  mocks.useReferenceIndex.mockReturnValue({
    data: {
      rows: [row],
      total: 1,
      facets: { total: 1, composed: 0, labels: { hop: 1 }, tiers: { D: 1 } },
    },
    isLoading: false,
    isError: false,
    isFetching: false,
  });
});

describe("ReferencePickerDialog evidence authority", () => {
  test("fetches exact robot-scoped detail and explains verified Tier-D scope", async () => {
    mocks.getReference.mockResolvedValue(detail());

    renderPicker();

    expect(screen.getByText("declared tier D")).toBeInTheDocument();
    expect(await screen.findByText(
      "Tier-D exact-schedule tracking evidence verified",
    )).toBeInTheDocument();
    expect(screen.getByText(/passed exact-schedule joint-position and root-height tracking/))
      .toHaveTextContent(/does not certify root-XY tracking, contact safety, collision avoidance, general dynamics feasibility/);
    expect(mocks.getReference).toHaveBeenCalledWith("g1", "four-rail-hop");
    expect(screen.getByRole("button", {
      name: "Four-rail traveling hop — g1/four-rail-hop",
    })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use motion" })).toBeEnabled();
  });

  test("does not turn a declared Tier D index row into verified authority", async () => {
    mocks.getReference.mockResolvedValue(detail({
      complete: false,
      reason: "the certificate digest is missing",
    }));

    renderPicker();

    expect(await screen.findByText(
      "Declared tier D is not verified launch authority",
    )).toBeInTheDocument();
    expect(screen.getByText(/the certificate digest is missing/))
      .toHaveTextContent(/live training will block/i);
    expect(screen.queryByText(
      "Tier-D exact-schedule tracking evidence verified",
    )).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use motion" })).toBeEnabled();
  });

  test("keeps an unavailable detail selectable while stating the launch gate", async () => {
    mocks.getReference.mockRejectedValue(new Error("offline"));

    renderPicker();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Evidence unavailable for g1\/four-rail-hop/i,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      /still select and preview this candidate, but live training will block/i,
    );
    expect(screen.getByRole("button", { name: "Use motion" })).toBeEnabled();
  });

  test("lets a researcher choose a Tier-K candidate without calling it launch-ready", async () => {
    const onPick = vi.fn();
    const tierKRow = { ...row, tier: "K" };
    mocks.useReferenceIndex.mockReturnValue({
      data: {
        rows: [tierKRow],
        total: 1,
        facets: { total: 1, composed: 0, labels: { hop: 1 }, tiers: { K: 1 } },
      },
      isLoading: false,
      isError: false,
      isFetching: false,
    });
    const tierKDetail = detail({
      admitted: false,
      complete: false,
      reason: "no Tier-D evidence is present",
    });
    tierKDetail.index_row = tierKRow;
    tierKDetail.dynamics_admission.tier = "K";
    mocks.getReference.mockResolvedValue(tierKDetail);

    renderPicker(onPick);

    expect(await screen.findByText(
      "Declared tier K is not verified launch authority",
    )).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Use motion" }));
    expect(onPick).toHaveBeenCalledWith({
      clipId: "four-rail-hop",
      robot: "g1",
    });
    expect(screen.queryByText(
      "Tier-D exact-schedule tracking evidence verified",
    )).not.toBeInTheDocument();
  });
});

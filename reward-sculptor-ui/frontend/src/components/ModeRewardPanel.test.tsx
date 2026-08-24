import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModeRewardFile, PromotedModeReward } from "@/lib/api";

const mocks = vi.hoisted(() => ({
  authorModeReward: vi.fn(),
  browseReferences: vi.fn(),
  getJob: vi.fn(),
  getModeEvidence: vi.fn(),
  getReferenceModes: vi.fn(),
  listModeRewards: vi.fn(),
  promoteModeReward: vi.fn(),
  recordModeEvidenceReceipt: vi.fn(),
  scaffoldModeReward: vi.fn(),
  searchReferences: vi.fn(),
  saveDraft: vi.fn(),
}));

vi.mock("@/components/ModeTimeline", () => ({
  ModeTimeline: () => <div aria-label="Mode timeline" />,
  modesFromGraph: () => [],
}));

vi.mock("@/hooks/useBehaviorDraft", () => ({
  useSaveBehaviorDraft: () => ({ mutate: mocks.saveDraft }),
}));

vi.mock("@/lib/api", () => {
  class ApiError extends Error {
    status: number;
    constructor(message: string, status = 500) {
      super(message);
      this.status = status;
    }
  }
  return {
    ApiError,
    authorModeReward: mocks.authorModeReward,
    browseReferences: mocks.browseReferences,
    getJob: mocks.getJob,
    getModeEvidence: mocks.getModeEvidence,
    getReferenceModes: mocks.getReferenceModes,
    listModeRewards: mocks.listModeRewards,
    promoteModeReward: mocks.promoteModeReward,
    recordModeEvidenceReceipt: mocks.recordModeEvidenceReceipt,
    scaffoldModeReward: mocks.scaffoldModeReward,
    searchReferences: mocks.searchReferences,
  };
});

import {
  ModeRewardPanel,
  modeRewardReadiness,
  selectModeRewardFile,
} from "./ModeRewardPanel";

const GOAL = "Clear the rail and land upright";
const CLIP_ID = "four-rail-hop";
const ROBOT = "g1";

function modeFile(overrides: Partial<ModeRewardFile> = {}): ModeRewardFile {
  return {
    filename: "mode_reward_v2.py",
    path: "/project/rewards/mode_reward_v2.py",
    clip_id: CLIP_ID,
    reference_robot: ROBOT,
    execution_context_digest: "a".repeat(64),
    authoring_goal: GOAL,
    authoring_intent_sha256: "b".repeat(64),
    authoring_intent_valid: true,
    context_blocker: null,
    context_current: true,
    tracking_enabled: true,
    mtime: 10,
    digest: "c".repeat(64),
    modes: [{ name: "jump", start_s: 0, end_s: 1, authored: false }],
    unauthored: ["jump"],
    ...overrides,
  };
}

function promotedReward(
  overrides: Partial<PromotedModeReward> = {},
): PromotedModeReward {
  return {
    version: 2,
    filename: "v2.py",
    clip_id: CLIP_ID,
    reference_robot: ROBOT,
    execution_context_digest: "a".repeat(64),
    authoring_goal: GOAL,
    authoring_intent_sha256: "b".repeat(64),
    authoring_intent_valid: true,
    context_blocker: null,
    context_current: true,
    selection_current: true,
    promotion_blocker: null,
    tracking_enabled: true,
    source_sha256: "c".repeat(64),
    source_filename: "mode_reward_v2.py",
    modes: [{ name: "jump", start_s: 0, end_s: 1, authored: false }],
    unauthored: ["jump"],
    ...overrides,
  };
}

function renderPanel(props: { clipId?: string; goal?: string } = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const element = (next: { clipId?: string; goal?: string }) => (
    <QueryClientProvider client={client}>
      <ModeRewardPanel
        slug="hard-hop"
        clipId={next.clipId === undefined ? CLIP_ID : next.clipId}
        robot={ROBOT}
        goal={next.goal ?? GOAL}
      />
    </QueryClientProvider>
  );
  const view = render(element(props));
  return {
    ...view,
    rerenderPanel: (next: { clipId?: string; goal?: string }) => {
      view.rerender(element(next));
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

beforeEach(() => {
  mocks.getReferenceModes.mockResolvedValue({
    clip_id: CLIP_ID,
    fps: 30,
    capability: {
      kind: "phase_window_reference_scaffold",
      paper_alignment: "ogmp_inspired",
      dispatch_authority: "episode_time_window",
      reference_generator: "fixed_composed_clip",
      runtime_transition_guards: false,
      policy_mode_conditioning: false,
      rho_bounded_exploration: false,
      closed_loop_receding_horizon_oracle: false,
      preference_conditioning: false,
      implementation_status: {},
      summary: "Fixed immutable phase windows.",
    },
    modes: [{
      name: "jump",
      frame_range: [0, 30],
      start_s: 0,
      end_s: 1,
      source_clip_id: CLIP_ID,
      reference_clip_id: CLIP_ID,
      reward_terms: [],
      success_predicate: null,
    }],
    transitions: [],
  });
  mocks.listModeRewards.mockResolvedValue({
    mode_rewards: [modeFile()],
    promoted: null,
  });
  mocks.getModeEvidence.mockResolvedValue({ recorded: false, receipt_sha256: "" });
  mocks.browseReferences.mockResolvedValue({ rows: [] });
  mocks.searchReferences.mockResolvedValue([]);
});

describe("mode reward authority", () => {
  it("prefers an exact current robot/clip file over a newer stale derivative", () => {
    const stale = modeFile({
      filename: "newer-stale.py",
      context_current: false,
      context_blocker: "world selection changed",
      mtime: 100,
    });
    const current = modeFile({ filename: "older-current.py", mtime: 1 });
    const wrongRobot = modeFile({
      filename: "wrong-robot.py",
      reference_robot: "h1",
      mtime: 1000,
    });

    expect(selectModeRewardFile(
      [stale, wrongRobot, current], CLIP_ID, ROBOT, GOAL,
    )?.filename).toBe("older-current.py");
  });

  it("fails closed for missing legacy currentness and intent fields", () => {
    const legacy = modeFile();
    delete (legacy as Partial<ModeRewardFile>).context_current;
    delete legacy.authoring_goal;
    delete legacy.authoring_intent_sha256;
    delete legacy.authoring_intent_valid;

    expect(modeRewardReadiness({
      ...legacy,
      tracking: legacy.tracking_enabled,
    }, GOAL, { robot: ROBOT, clipId: CLIP_ID })).toEqual({
      ready: false,
      blocker: "The scaffold does not prove the current execution context.",
    });

    const malformedIntent = modeFile({ authoring_intent_valid: undefined });
    delete malformedIntent.authoring_intent_sha256;
    expect(modeRewardReadiness({
      ...malformedIntent,
      tracking: malformedIntent.tracking_enabled,
    }, GOAL, { robot: ROBOT, clipId: CLIP_ID })).toEqual({
      ready: false,
      blocker: "The scaffold does not contain a verifiable authoring intent.",
    });
  });

  it.each([
    ["wrong clip", { clip_id: "different-hop" }],
    ["wrong robot", { reference_robot: "h1" }],
  ])("fails closed for a %s identity", (_label, override) => {
    const file = modeFile(override);
    expect(modeRewardReadiness({
      ...file,
      tracking: file.tracking_enabled,
    }, GOAL, { robot: ROBOT, clipId: CLIP_ID })).toEqual({
      ready: false,
      blocker:
        `The scaffold identity does not match selected reference ${ROBOT}/${CLIP_ID}.`,
    });
  });

  it("blocks Author and Promote when the execution context is stale", async () => {
    mocks.listModeRewards.mockResolvedValue({
      mode_rewards: [modeFile({
        context_current: false,
        context_blocker: "The selected world tuple changed.",
      })],
      promoted: null,
    });

    renderPanel();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /selected world tuple changed/i,
    );
    expect(screen.getByRole("button", { name: "Author" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Re-scaffold required" }))
      .toBeDisabled();
    expect(screen.getByRole("button", { name: "Re-scaffold" })).toBeEnabled();
  });

  it("blocks the old scaffold when the normalized behavior goal changed", async () => {
    renderPanel({ goal: "Clear the rail, then recover on one foot" });

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /behavior goal changed after this scaffold was created/i,
    );
    expect(screen.getByRole("button", { name: "Author" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Re-scaffold" })).toBeEnabled();
  });

  it("does not call a promoted reward current without exact selection evidence", async () => {
    mocks.listModeRewards.mockResolvedValue({
      mode_rewards: [modeFile()],
      promoted: promotedReward({ selection_current: false }),
    });

    renderPanel();

    expect(await screen.findByRole("button", {
      name: "Use for training (replaces v2)",
    })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Training v2" }))
      .not.toBeInTheDocument();
    expect(screen.getByLabelText(`Selected exact reference ${ROBOT}/${CLIP_ID}`))
      .toHaveTextContent(`${ROBOT}/${CLIP_ID}`);
  });

  it("ignores a stale refresh that resolves after the authoring goal changes", async () => {
    const old = deferred<{
      mode_rewards: ModeRewardFile[];
      promoted: PromotedModeReward | null;
    }>();
    const newGoal = "Clear the rail, then recover on one foot";
    mocks.listModeRewards
      .mockReturnValueOnce(old.promise)
      .mockResolvedValueOnce({
        mode_rewards: [modeFile({ authoring_goal: newGoal })],
        promoted: null,
      });

    const view = renderPanel({ goal: GOAL });
    view.rerenderPanel({ goal: newGoal });

    expect(await screen.findByRole("button", { name: "Author" })).toBeEnabled();

    await act(async () => {
      old.resolve({ mode_rewards: [modeFile()], promoted: null });
      await old.promise;
    });

    expect(screen.getByRole("button", { name: "Author" })).toBeEnabled();
    expect(screen.queryByText(/behavior goal changed after this scaffold/i))
      .not.toBeInTheDocument();
  });
});

describe("reference search feedback", () => {
  it("distinguishes an unavailable search from a valid empty result", async () => {
    mocks.browseReferences.mockRejectedValue(new Error("offline"));

    renderPanel({ clipId: "" });

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /reference search is unavailable/i,
    );
    expect(screen.queryByText(/No clips matched/i)).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", {
      name: "Search composed references by description or exact robot/clip ID",
    })).toBeInTheDocument();
  });
});

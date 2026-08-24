import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { RefMatch } from "@/lib/types";

const mocks = vi.hoisted(() => ({
  useComposeReference: vi.fn(),
  useReferenceSearch: vi.fn(),
}));

vi.mock("@/hooks/useReferences", () => ({
  useComposeReference: mocks.useComposeReference,
  useReferenceSearch: mocks.useReferenceSearch,
}));

import {
  ComposeMotionDialog,
  MAX_COMPOSE_SEGMENTS,
  isExactReferenceQuery,
  parseNumericInput,
  sampledTrajectoryFps,
} from "./ComposeMotionDialog";

const identicalDescriptionHits: RefMatch[] = [
  {
    clip_id: "50009_one_leg_jump_poses_60_jpos--origin-relative",
    text: "50009 one leg jump poses 60 jpos",
    score: 33.6401,
    match_confidence: null,
    reason: null,
    tier: "K",
    license: "CC BY-NC-ND 4.0",
    n_frames: 229,
    fps: 60,
    duration_s: 3.8,
  },
  {
    clip_id: "50009_one_leg_jump_poses_60_jpos",
    text: "50009 one leg jump poses 60 jpos",
    score: 33.6401,
    match_confidence: null,
    reason: null,
    tier: "K",
    license: "CC BY-NC-ND 4.0",
    n_frames: 229,
    fps: 60,
    duration_s: 3.8,
  },
];

beforeEach(() => {
  mocks.useComposeReference.mockReturnValue({
    isPending: false,
    mutate: vi.fn(),
  });
  mocks.useReferenceSearch.mockImplementation((query: string) => ({
    data: query ? identicalDescriptionHits : [],
    isLoading: false,
    isError: false,
  }));
});

async function makeReady(
  user: ReturnType<typeof userEvent.setup>,
  name = "hard jump",
) {
  await user.type(
    screen.getByRole("textbox", { name: "Search a clip for phase 1" }),
    "jump",
  );
  await user.click(screen.getAllByRole("button", {
    name: /g1\/50009_one_leg_jump_poses_60_jpos/,
  })[0]);
  await user.type(
    screen.getByRole("textbox", { name: "Search a clip for phase 2" }),
    "land",
  );
  await user.click(screen.getAllByRole("button", {
    name: /g1\/50009_one_leg_jump_poses_60_jpos/,
  })[0]);
  await user.type(screen.getByRole("textbox", { name: "Motion name" }), name);
}

const composeError = (type: string, detail: string) => new ApiError({
  type,
  title: "cannot compose these segments",
  status: 400,
  detail,
});

describe("sampledTrajectoryFps", () => {
  it("uses N - 1 sample intervals like the runtime reference clock", () => {
    expect(sampledTrajectoryFps(229, 3.8)).toBeCloseTo(60, 10);
  });

  it("keeps one-frame references well-defined and rejects invalid duration", () => {
    expect(sampledTrajectoryFps(1, 1 / 60)).toBeCloseTo(60, 10);
    expect(sampledTrajectoryFps(10, 0)).toBe(0);
  });
});

describe("numeric composition inputs", () => {
  it("distinguishes optional blanks from malformed values", () => {
    expect(parseNumericInput("", "Trim start", { minimum: 0 }))
      .toEqual({ value: null, error: null });
    expect(parseNumericInput("abc", "Trim start", { minimum: 0 }).error)
      .toMatch(/must be a number/i);
    expect(parseNumericInput("1,5", "Trim start", { minimum: 0 }).error)
      .toMatch(/must be a number/i);
    expect(parseNumericInput("0x10", "Trim start", { minimum: 0 }).error)
      .toMatch(/must be a number/i);
    expect(parseNumericInput("Infinity", "Trim start", { minimum: 0 }).error)
      .toMatch(/must be a number/i);
    expect(parseNumericInput("-0.1", "Trim start", { minimum: 0 }).error)
      .toMatch(/at least 0/i);
    expect(parseNumericInput("0", "Target fps", {
      minimum: 0, exclusiveMinimum: true,
    }).error).toMatch(/greater than 0/i);
    expect(parseNumericInput("241", "Target fps", { maximum: 240 }).error)
      .toMatch(/at most 240/i);
    expect(parseNumericInput("10.01", "Crossfade", { maximum: 10 }).error)
      .toMatch(/at most 10/i);
  });

  it("shows malformed seam input and never submits it as a default", async () => {
    const mutate = vi.fn();
    mocks.useComposeReference.mockReturnValue({ isPending: false, mutate });
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);

    await makeReady(user);

    const blend = screen.getByRole("textbox", { name: "Crossfade seconds" });
    await user.clear(blend);
    await user.type(blend, "1,5");

    expect(screen.getByRole("alert")).toHaveTextContent(/Crossfade must be a number/i);
    expect(blend).toHaveAttribute("aria-invalid", "true");
    expect(blend).toHaveAccessibleDescription(/Crossfade must be a number/i);
    const compose = screen.getByRole("button", { name: "Compose" });
    expect(compose).toBeDisabled();
    await user.click(compose);
    expect(mutate).not.toHaveBeenCalled();
  });

  it.each([
    ["Crossfade seconds", "10.01", /at most 10/i],
    ["Target fps", "241", /at most 240/i],
  ])("blocks an over-limit %s before submit", async (label, value, message) => {
    const mutate = vi.fn();
    mocks.useComposeReference.mockReturnValue({ isPending: false, mutate });
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);
    await makeReady(user);

    const field = screen.getByRole("textbox", { name: label });
    await user.clear(field);
    await user.type(field, value);

    expect(screen.getByRole("alert")).toHaveTextContent(message);
    expect(field).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: "Compose" })).toBeDisabled();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("rejects an inverted trim before the request is built", async () => {
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);

    await user.type(
      screen.getByRole("textbox", { name: "Phase 1 start seconds" }), "2",
    );
    await user.type(
      screen.getByRole("textbox", { name: "Phase 1 end seconds" }), "1",
    );

    expect(screen.getByRole("alert")).toHaveTextContent(
      /Trim end must be greater than trim start/i,
    );
  });
});

describe("phase count boundary", () => {
  it("disables Add phase at the shared segment cap", async () => {
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);
    const add = screen.getByRole("button", { name: "Add phase" });

    for (let index = 2; index < MAX_COMPOSE_SEGMENTS; index += 1) {
      await user.click(add);
    }

    expect(screen.getAllByRole("textbox", { name: /Phase \d+ label/ }))
      .toHaveLength(MAX_COMPOSE_SEGMENTS);
    expect(add).toBeDisabled();
    expect(add).toHaveAccessibleDescription(
      `${MAX_COMPOSE_SEGMENTS} of ${MAX_COMPOSE_SEGMENTS} phases · maximum reached`,
    );
  });
});

describe("kinematic gate bypass authority", () => {
  it("offers Compose anyway only for the structured gate type and retries the exact request", async () => {
    const gate = composeError(
      "/problems/reference-compose-gate",
      "seam discontinuity 0.710 rad exceeds 0.350 rad",
    );
    const mutate = vi.fn()
      .mockImplementationOnce((_request, options) => options.onError(gate))
      .mockImplementationOnce(() => undefined);
    mocks.useComposeReference.mockReturnValue({ isPending: false, mutate });
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);
    await makeReady(user);

    await user.click(screen.getByRole("button", { name: "Compose" }));

    const measurement = await screen.findByRole("alert");
    expect(measurement).toHaveTextContent(/0\.710 rad exceeds 0\.350 rad/i);
    const anyway = screen.getByRole("button", { name: "Compose anyway" });
    await user.click(anyway);

    expect(mutate).toHaveBeenCalledTimes(2);
    const first = mutate.mock.calls[0][0];
    const second = mutate.mock.calls[1][0];
    expect(first.strict).toBe(true);
    expect(second.strict).toBe(false);
    expect({ ...second, strict: true }).toEqual(first);
  });

  it("does not offer a bypass for a non-gate error that happens to say jump", async () => {
    const mutate = vi.fn((_request, options) => options.onError(composeError(
      "/problems/reference-compose",
      "jump source clip is missing",
    )));
    mocks.useComposeReference.mockReturnValue({ isPending: false, mutate });
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);
    await makeReady(user);

    await user.click(screen.getByRole("button", { name: "Compose" }));

    expect(screen.queryByRole("button", { name: "Compose anyway" }))
      .not.toBeInTheDocument();
  });

  it.each([
    ["motion name", async (user: ReturnType<typeof userEvent.setup>) => {
      const field = screen.getByRole("textbox", { name: "Motion name" });
      await user.clear(field);
      await user.type(field, "harder jump");
    }],
    ["trim", async (user: ReturnType<typeof userEvent.setup>) => {
      await user.type(
        screen.getByRole("textbox", { name: "Phase 1 start seconds" }),
        "0.1",
      );
    }],
    ["crossfade", async (user: ReturnType<typeof userEvent.setup>) => {
      const field = screen.getByRole("textbox", { name: "Crossfade seconds" });
      await user.clear(field);
      await user.type(field, "0.3");
    }],
    ["target fps", async (user: ReturnType<typeof userEvent.setup>) => {
      await user.type(screen.getByRole("textbox", { name: "Target fps" }), "120");
    }],
    ["phase label", async (user: ReturnType<typeof userEvent.setup>) => {
      await user.type(screen.getByRole("textbox", { name: "Phase 1 label" }), "launch");
    }],
  ])("invalidates a refused request after changing its %s", async (_field, change) => {
    const mutate = vi.fn((_request, options) => options.onError(composeError(
      "/problems/reference-compose-gate",
      "seam discontinuity 0.710 rad exceeds 0.350 rad",
    )));
    mocks.useComposeReference.mockReturnValue({ isPending: false, mutate });
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);
    await makeReady(user);
    await user.click(screen.getByRole("button", { name: "Compose" }));
    expect(await screen.findByRole("button", { name: "Compose anyway" }))
      .toBeInTheDocument();

    await change(user);

    expect(screen.queryByRole("button", { name: "Compose anyway" }))
      .not.toBeInTheDocument();
  });
});

describe("reference search feedback", () => {
  it("distinguishes an unavailable search from an empty result", async () => {
    mocks.useReferenceSearch.mockImplementation((query: string) => ({
      data: query ? identicalDescriptionHits : [],
      isLoading: false,
      isError: Boolean(query),
    }));
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);

    await user.type(
      screen.getByRole("textbox", { name: "Search a clip for phase 1" }),
      "jump",
    );

    expect(screen.getByRole("alert")).toHaveTextContent(/search is unavailable/i);
    expect(screen.queryByText("No matches.")).not.toBeInTheDocument();
    expect(screen.queryByText("50009 one leg jump poses 60 jpos"))
      .not.toBeInTheDocument();
  });

  it("labels a successful empty search as no matches", async () => {
    mocks.useReferenceSearch.mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
    });
    const user = userEvent.setup();
    render(<ComposeMotionDialog robot="g1" onClose={vi.fn()} />);

    await user.type(
      screen.getByRole("textbox", { name: "Search a clip for phase 1" }),
      "jump",
    );

    expect(screen.getByText("No matches.")).toBeInTheDocument();
    expect(screen.queryByText(/search is unavailable/i)).not.toBeInTheDocument();
  });
});

describe("exact reference identity", () => {
  it("accepts bare and matching robot-scoped IDs only", () => {
    const id = "50009_one_leg_jump_poses_60_jpos--origin-relative";
    expect(isExactReferenceQuery(id, "g1", id)).toBe(true);
    expect(isExactReferenceQuery(` G1/${id} `, "g1", id)).toBe(true);
    expect(isExactReferenceQuery(`t1/${id}`, "g1", id)).toBe(false);
  });

  it("disambiguates identical descriptions and preserves the selected scope", async () => {
    const user = userEvent.setup();
    render(
      <ComposeMotionDialog robot="g1" onClose={vi.fn()} />,
    );
    const id = "50009_one_leg_jump_poses_60_jpos--origin-relative";

    await user.type(
      screen.getByRole("textbox", { name: "Search a clip for phase 1" }),
      `g1/${id}`,
    );

    expect(screen.getAllByText("50009 one leg jump poses 60 jpos"))
      .toHaveLength(2);
    const exactButton = screen.getByRole("button", {
      name: new RegExp(`g1/${id}`),
    });
    expect(within(exactButton).getByText("exact ID match")).toBeInTheDocument();
    expect(screen.getByText(
      "g1/50009_one_leg_jump_poses_60_jpos",
    )).toBeInTheDocument();

    await user.click(exactButton);

    expect(screen.getByText(`g1/${id}`)).toBeInTheDocument();
    expect(screen.queryByText("exact ID match")).not.toBeInTheDocument();
  });
});

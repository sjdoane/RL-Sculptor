import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import { TaskProgramCard } from "@/components/WorldTab";
import type { WorldEventProgram } from "@/lib/types";

export const EVENT_PROGRAM: WorldEventProgram = {
  id: "route_jump_hold",
  ordered_phase_ids: ["route", "jump", "hold"],
  transition_spec: [
    {
      from: "route",
      to: "jump",
      when: { event: "goal_complete" },
    },
    {
      from: "jump",
      to: "hold",
      when: {
        event: "bilateral_support_cycle",
        support_contacts: [
          ["robot:left_foot", "world:terrain"],
          ["robot:right_foot", "world:terrain"],
        ],
        min_air_time_s: 0.06,
        min_height_delta_m: 0.18,
      },
    },
    {
      from: "hold",
      to: "terminal",
      when: { event: "minimum_hold_elapsed", minimum_hold_s: 2 },
    },
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

test("renders the immutable ROUTE to JUMP to HOLD program and evaluation boundary", async () => {
  const user = userEvent.setup();
  render(<TaskProgramCard program={EVENT_PROGRAM} />);

  const rail = screen.getByRole("list", { name: "ROUTE then JUMP then HOLD" });
  const phases = within(rail).getAllByRole("listitem");
  expect(phases.map((phase) => phase.textContent)).toEqual([
    expect.stringMatching(/ROUTE.*Raw goal completion/s),
    expect.stringMatching(/JUMP.*Both feet · 0\.06 s · 0\.18 m/s),
    expect.stringMatching(/HOLD.*Landing, then 2\.0 s/s),
  ]);
  expect(screen.getByText("24 s maximum")).toBeInTheDocument();
  expect(screen.getByText("50% ROUTE · 40% JUMP · 10% HOLD")).toBeInTheDocument();
  expect(screen.getByText("Always starts at ROUTE")).toBeInTheDocument();
  expect(screen.getAllByText(/authored_event_phase/)[0]).toBeInTheDocument();
  expect(screen.getByRole("region", { name: /scroll horizontally/i }))
    .toHaveAttribute("tabindex", "0");

  await user.click(screen.getByText("Exact program JSON and provenance"));
  expect(screen.getByText((_, element) =>
    element?.tagName === "PRE"
    && Boolean(element.textContent?.includes('"selection_tuple_hash": "abc123"'))
    && Boolean(element.textContent?.includes('"sha256"'))))
    .toBeInTheDocument();
});

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";

import { EnvSpecTrainSection } from "@/components/ProjectSettingsDialog";
import type { ProjectDetail } from "@/lib/types";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("adds an unset train setting only after valid JSON is provided", async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "PUT") {
      return new Response(JSON.stringify({
        applied: ["friction_range"],
        rejected: [],
        new_version: "v2",
        current: {
          meta: { version: "v2" },
          train: { entropy_coef_scale: 1, friction_range: [0.8, 1.2] },
        },
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response(JSON.stringify({
      active: true,
      current: {
        meta: { version: "v1" },
        train: { entropy_coef_scale: 1 },
      },
      versions: ["v1"],
      editable_train_keys: ["entropy_coef_scale", "friction_range"],
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const user = userEvent.setup();

  render(
    <QueryClientProvider client={client}>
      <EnvSpecTrainSection
        project={{ slug: "fresh-g1-project" } as ProjectDetail}
        open
      />
    </QueryClientProvider>,
  );

  const addButton = await screen.findByRole("button", { name: "Add train setting" });
  expect(screen.queryByLabelText("Train setting")).not.toBeInTheDocument();
  await user.click(addButton);

  await user.selectOptions(screen.getByLabelText("Train setting"), "friction_range");
  expect(screen.getByText(/Train-only foot\/ground friction randomization/))
    .toBeInTheDocument();

  const valueInput = screen.getByLabelText("JSON value");
  const saveButton = screen.getByRole("button", { name: "Save setting" });
  await user.type(valueInput, "[[0.8,");
  expect(saveButton).toBeDisabled();
  expect(screen.getByText("Enter a valid JSON value before saving."))
    .toBeInTheDocument();

  await user.clear(valueInput);
  await user.type(valueInput, "[[0.8, 1.2]");
  expect(saveButton).toBeEnabled();
  await user.click(saveButton);

  await waitFor(() => {
    const putCall = fetchMock.mock.calls.find(([, init]) => init?.method === "PUT");
    expect(putCall).toBeDefined();
    expect(JSON.parse(String(putCall?.[1]?.body))).toEqual({
      edits: [{
        parameter: "friction_range",
        new_value: [0.8, 1.2],
        rationale: "added from project settings",
      }],
    });
  });
});

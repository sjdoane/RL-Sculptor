import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import { ProjectTabList } from "@/pages/ProjectDetail";


function TabsHarness() {
  const [tab, setTab] = useState<"overview" | "world" | "rewards" | "physics" | "knowledge" | "training" | "results">("overview");
  return <ProjectTabList tab={tab} onSelect={setTab} />;
}


test("project tabs use roving focus, arrow keys, and explicit panel ownership", async () => {
  const user = userEvent.setup();
  render(<TabsHarness />);

  const tablist = screen.getByRole("tablist", { name: "Project workspace" });
  const overview = screen.getByRole("tab", { name: "Overview" });
  const world = screen.getByRole("tab", { name: "World" });
  expect(tablist).toHaveAttribute("aria-orientation", "horizontal");
  expect(overview).toHaveAttribute("tabindex", "0");
  expect(world).toHaveAttribute("tabindex", "-1");
  expect(overview).toHaveAttribute("aria-controls", "project-panel-overview");

  overview.focus();
  await user.keyboard("{ArrowRight}");
  expect(world).toHaveFocus();
  expect(world).toHaveAttribute("aria-selected", "true");
  expect(world).toHaveAttribute("tabindex", "0");
  expect(overview).toHaveAttribute("tabindex", "-1");

  await user.keyboard("{End}");
  expect(screen.getByRole("tab", { name: "Results" })).toHaveFocus();
  await user.keyboard("{Home}");
  expect(overview).toHaveFocus();
});

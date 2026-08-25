import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import {
  CapabilityGroup,
  flattenCapabilityParameters,
} from "@/components/PaperDetailModal";
import type { ResearchCapabilitySummary } from "@/lib/types";

function sonicCapability(): ResearchCapabilitySummary {
  return {
    id: "capability:sonic-public-controller-contract",
    name: "SONIC public controller contract",
    description: "Camera-free local motion tracking for the 29-DoF G1.",
    scope: "paper_mechanism",
    parameters: {
      controlled_dof: 29,
      control_rate_hz: 50,
      observations: {
        proprioceptive_history_steps: 10,
      },
    },
    implementation_status: "unsupported",
    status_definition: "The runtime does not execute this mechanism.",
    code_evidence: [],
    provenance: "paper_claim",
    paper_role: "paper_mechanism",
    source_version: "arXiv:2511.07820v4",
    source_locator: "https://arxiv.org/html/2511.07820v4#S3.SS2",
  };
}

test("flattens nested reviewed parameters into stable searchable paths", () => {
  expect(flattenCapabilityParameters(sonicCapability().parameters)).toEqual([
    { path: "control_rate_hz", value: "50" },
    { path: "controlled_dof", value: "29" },
    { path: "observations.proprioceptive_history_steps", value: "10" },
  ]);
});

test("labels paper claims as unsupported and searches exact parameters", async () => {
  const user = userEvent.setup();
  const { container } = render(<CapabilityGroup items={[sonicCapability()]} />);

  expect(screen.getByText("Unsupported in RewardSculptor")).toBeInTheDocument();
  expect(screen.getByText(/Parameters describe the cited source/i)).toBeInTheDocument();

  const search = screen.getByRole("textbox", {
    name: "Search reviewed paper parameters",
  });
  await user.type(search, "50");
  expect(screen.getByText("SONIC public controller contract")).toBeInTheDocument();
  expect(container.querySelector("details")?.open).toBe(true);

  await user.clear(search);
  await user.type(search, "camera input");
  expect(screen.getByText(/No reviewed mechanism or parameter matches/i)).toBeInTheDocument();
});

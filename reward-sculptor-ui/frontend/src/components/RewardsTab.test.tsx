import { render, screen } from "@testing-library/react";
import { beforeEach, expect, test } from "vitest";

import { RewardsExplainer } from "@/components/RewardsTab";


beforeEach(() => localStorage.clear());


test("describes steer, observe-only, and blind reward-selection authority", () => {
  render(<RewardsExplainer />);

  const explainer = screen.getByText("How reward evolution works.").parentElement;
  expect(explainer).toHaveTextContent(/steered objective may keep or revert/i);
  expect(explainer).toHaveTextContent(
    /observe-only objective records evidence without choosing the reward/i,
  );
  expect(explainer).toHaveTextContent(/blind ablation.*no objective-success claim/i);
  expect(explainer).not.toHaveTextContent(
    /Versions that improve the tracked metric are kept/i,
  );
});

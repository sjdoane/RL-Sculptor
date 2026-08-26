import { expect, test } from "vitest";

import { projectBadgeStatus } from "@/lib/projectStatus";


test("does not present a project with finished runs as scientific success", () => {
  expect(projectBadgeStatus("completed")).toBe("project-history");
  expect(projectBadgeStatus("completed", true)).toBe("running");
  expect(projectBadgeStatus("errored")).toBe("errored");
});

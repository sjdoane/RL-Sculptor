import { StrictMode, useEffect } from "react";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation, useNavigate, type NavigateFunction } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import Layout from "@/components/Layout";

const status = vi.hoisted(() => ({ mounted: 0, setTheme: vi.fn() }));
vi.mock("@/hooks/useLibrary", () => ({
  useSystemGpu: () => {
    useEffect(() => { status.mounted++; return () => { status.mounted--; }; }, []);
    return { data: undefined };
  },
}));
vi.mock("@/hooks/useDashboard", () => ({ useSystemInfo: () => ({ data: undefined }) }));
vi.mock("@/hooks/useTheme", () => ({ useTheme: () => ({ theme: "light", setTheme: status.setTheme }) }));

let narrow = true;
let listeners: Set<() => void>;
beforeEach(() => {
  listeners = new Set();
  narrow = true;
  status.setTheme.mockClear();
  vi.stubGlobal("matchMedia", vi.fn((query: string) => ({
    get matches() { return query === "(max-width: 767px)" && narrow; },
    addEventListener: (_: string, fn: () => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
  })));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); expect(status.mounted).toBe(0); });

function setup(strict = false) {
  let navigate: NavigateFunction;
  function Page() {
    navigate = useNavigate();
    const location = useLocation();
    return <><button>Page action</button><span data-testid="route">{location.pathname}</span></>;
  }
  const app = <MemoryRouter><Routes><Route element={<Layout />}><Route path="*" element={<Page />} /></Route></Routes></MemoryRouter>;
  const view = render(strict ? <StrictMode>{app}</StrictMode> : app);
  return { ...view, navigate: (path: string) => navigate(path), user: userEvent.setup() };
}
function resize(value: boolean) {
  act(() => { narrow = value; listeners.forEach((fn) => fn()); });
}

test("compact dialog has all five destinations, one status subtree and current-link dismissal", async () => {
  const { user } = setup();
  const menu = screen.getByRole("button", { name: "Menu" });
  expect(status.mounted).toBe(0);
  expect(screen.queryByRole("navigation")).not.toBeInTheDocument();
  await user.click(menu);
  const dialog = screen.getByRole("dialog", { name: "Navigation" });
  expect(menu).toHaveAttribute("aria-expanded", "true");
  expect(within(dialog).getAllByRole("link").map((a) => a.textContent)).toEqual([
    "Dashboard", "Projects", "Saved Missions", "Robot Library", "Settings",
  ]);
  expect(status.mounted).toBe(1);
  expect(within(dialog).getByText("No CUDA")).toBeInTheDocument();
  await user.click(within(dialog).getByRole("button", { name: "Toggle color theme" }));
  expect(status.setTheme).toHaveBeenCalledWith("dark");
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  const current = within(dialog).getByRole("link", { name: "Dashboard" });
  expect(current).toHaveAttribute("aria-current", "page");
  await user.click(current);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(menu).toHaveFocus();
  expect(status.mounted).toBe(0);
});

test("each navigation link closes the dialog, including Saved Missions", async () => {
  const { user } = setup();
  for (const [name, path] of [["Projects", "/projects"], ["Saved Missions", "/saved"], ["Robot Library", "/library"], ["Settings", "/settings"], ["Dashboard", "/"]]) {
    await user.click(screen.getByRole("button", { name: "Menu" }));
    await user.click(screen.getByRole("link", { name }));
    expect(screen.getByTestId("route")).toHaveTextContent(path);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  }
});

test("route changes outside the navigation also dismiss an open dialog", async () => {
  const { user, navigate } = setup();
  await user.click(screen.getByRole("button", { name: "Menu" }));
  act(() => navigate("/settings"));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

test("close, Escape and armed scrim dismiss; inside clicks do not", async () => {
  const { user } = setup();
  const menu = screen.getByRole("button", { name: "Menu" });
  for (const mode of ["close", "escape", "scrim"]) {
    await user.click(menu);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toContainElement(document.activeElement as HTMLElement);
    await user.click(within(dialog).getByText("System"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    if (mode === "close") await user.click(within(dialog).getByRole("button", { name: "Close" }));
    if (mode === "escape") await user.keyboard("{Escape}");
    if (mode === "scrim") {
      const scrim = dialog.parentElement!;
      // A release on the scrim without a matching pointer-down must not close.
      fireEvent.click(scrim);
      expect(screen.getByRole("dialog")).toBeInTheDocument();
      fireEvent.pointerDown(scrim); fireEvent.click(scrim);
    }
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(menu).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Page action" })).toHaveFocus();
  }
});

test("dialog traps forward and reverse Tab, then releases after dismissal", async () => {
  // jsdom has no layout. Model visibility for the existing Modal focus query;
  // actual responsive sizing and keyboard trapping are checked in a browser.
  vi.spyOn(HTMLElement.prototype, "offsetParent", "get").mockImplementation(function (this: HTMLElement) { return this.parentElement; });
  const { user } = setup();
  await user.click(screen.getByRole("button", { name: "Menu" }));
  const close = screen.getByRole("button", { name: "Close" });
  const theme = screen.getByRole("button", { name: "Toggle color theme" });
  expect(close).toHaveFocus();
  await user.tab({ shift: true });
  expect(theme).toHaveFocus();
  await user.tab();
  expect(close).toHaveFocus();
});

test("767 to 768 unmounts modal/status trap, focuses visible current desktop link and reopens closed", async () => {
  const { user } = setup();
  await user.click(screen.getByRole("button", { name: "Menu" }));
  resize(false);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Menu" })).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Dashboard" })).toHaveFocus();
  expect(screen.getAllByRole("navigation")).toHaveLength(1);
  expect(status.mounted).toBe(1);
  await user.tab();
  expect(screen.getByRole("link", { name: "Projects" })).toHaveFocus();
  resize(true);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Menu" })).toHaveAttribute("aria-expanded", "false");
  expect(status.mounted).toBe(0);
});

test("StrictMode breakpoint listeners are cleaned up on unmount", () => {
  const { unmount } = setup(true);
  expect(listeners.size).toBe(1);
  unmount();
  expect(listeners.size).toBe(0);
});

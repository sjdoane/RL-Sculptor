import { useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "rs.ui.theme";

/** Minimal theme toggler. `system` tracks the user-agent preference
 *  via the `prefers-color-scheme` media query; explicit choices win
 *  and persist to localStorage.
 *
 *  Applies the theme by toggling `.dark` on `<html>` — Tailwind's
 *  `darkMode: "class"` in tailwind.config.js reads that class.
 */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(readStored);

  useEffect(() => {
    applyTheme(theme);
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (theme !== "system" || !mq) return;
    const onChange = () => applyTheme("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);

  const setTheme = (next: Theme) => {
    setThemeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* private mode — ignore */
    }
  };

  return { theme, setTheme };
}

function readStored(): Theme {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s === "light" || s === "dark" || s === "system") return s;
  } catch {
    /* ignore */
  }
  return "system";
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  const effective =
    theme === "system"
      ? window.matchMedia?.("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : theme;
  if (effective === "dark") root.classList.add("dark");
  else root.classList.remove("dark");
}

/** Call once at app boot to apply the stored theme before React
 *  mounts — avoids a flash of the wrong theme. */
export function bootstrapTheme(): void {
  applyTheme(readStored());
}

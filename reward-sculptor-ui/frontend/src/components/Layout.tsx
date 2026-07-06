import { NavLink, Outlet } from "react-router-dom";

import { Icon } from "@/components/rs/icon";
import { useSystemGpu } from "@/hooks/useLibrary";
import { useSystemInfo } from "@/hooks/useDashboard";
import { useTheme } from "@/hooks/useTheme";

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

const NAV_ITEMS: readonly NavItem[] = [
  { to: "/", label: "Dashboard", icon: "layout-grid" },
  { to: "/projects", label: "Projects", icon: "folder" },
  // The library is the entry point for creating projects — every "New
  // project" button lands here, so it must be a first-class destination.
  { to: "/library", label: "Robot Library", icon: "library" },
  { to: "/settings", label: "Settings", icon: "settings" },
];

function SystemMini() {
  const gpu = useSystemGpu({ refetchIntervalMs: 5000 });
  const info = useSystemInfo();
  const dev = gpu.data?.devices?.[0];
  const total = dev?.total_memory_bytes ?? 0;
  const used =
    dev?.used_memory_bytes ??
    (dev && dev.free_memory_bytes != null ? total - dev.free_memory_bytes : 0);
  const usedGb = used / 1e9;
  const totalGb = total / 1e9;
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  const gpuName = (dev?.name ?? "No GPU").replace(/^NVIDIA\s+/, "");
  const cudaOk = gpu.data?.cuda_available ?? false;
  const apiKey = info.data?.anthropic_api_key_set ?? false;

  return (
    <div className="rs-mini-sys">
      <div className="rs-mini-row">
        <span style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
          <Icon name="cpu" size={13} />
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {gpuName}
          </span>
        </span>
      </div>
      <div className="rs-vram">
        <i style={{ width: pct + "%" }} />
      </div>
      <div className="rs-mini-row">
        <span className="rs-num">
          {totalGb > 0 ? `${usedGb.toFixed(1)} / ${totalGb.toFixed(1)} GB` : "— GB"}
        </span>
        {dev?.temperature_c != null && (
          <span className="rs-num" style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Icon name="thermometer" size={12} />
            {dev.temperature_c}°C
          </span>
        )}
      </div>
      <div className="rs-mini-row" style={{ marginTop: 2 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            className="rs-dot"
            style={{ background: cudaOk ? "var(--st-emerald)" : "var(--st-slate)" }}
          />
          {cudaOk ? `CUDA ${gpu.data?.cuda_version ?? ""}`.trim() : "No CUDA"}
        </span>
        <span
          style={{ display: "flex", alignItems: "center", gap: 5 }}
          title={apiKey ? "Anthropic API key detected" : "No Anthropic API key"}
        >
          <Icon name="key" size={12} color={apiKey ? "var(--st-emerald)" : "var(--st-rose)"} />
          key
        </span>
      </div>
    </div>
  );
}

export default function Layout() {
  // Mounting useTheme here keeps the theme applied app-wide (and tracks
  // system-preference changes) rather than only while Settings is open.
  const { theme, setTheme } = useTheme();
  const effective =
    theme === "system"
      ? window.matchMedia?.("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : theme;

  return (
    <div className="rs-app">
      <nav className="rs-rail" aria-label="Primary">
        <div className="rs-rail-mark">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="8" fill="var(--ink)" />
            <path
              d="M11 9 L19 16 L11 23"
              fill="none"
              stroke="var(--rs-primary)"
              strokeWidth="2.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <line x1="20" y1="23" x2="24" y2="23" stroke="var(--canvas)" strokeWidth="2.6" strokeLinecap="round" />
          </svg>
          <span className="rs-rail-word">
            rl·<b>sculptor</b>
          </span>
        </div>

        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) => "rs-navitem" + (isActive ? " on" : "")}
          >
            <Icon name={item.icon} size={17} />
            {item.label}
          </NavLink>
        ))}

        <div className="rs-rail-sec">System</div>
        <SystemMini />

        <div className="rs-rail-foot">
          <button
            className="rs-theme-btn"
            onClick={() => setTheme(effective === "dark" ? "light" : "dark")}
            aria-label="Toggle color theme"
          >
            <Icon name={effective === "dark" ? "sun" : "moon"} size={15} />
            {effective === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </nav>

      <main className="rs-main">
        <Outlet />
      </main>
    </div>
  );
}

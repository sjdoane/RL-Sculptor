import { NavLink, Outlet } from "react-router-dom";
import { FolderKanban, LayoutDashboard, Settings } from "lucide-react";

import { cn } from "@/lib/utils";

interface NavItem {
  to: string;
  label: string;
  icon: typeof FolderKanban;
}

const NAV_ITEMS: readonly NavItem[] = [
  { to: "/",         label: "Dashboard", icon: LayoutDashboard },
  { to: "/projects", label: "Projects",  icon: FolderKanban },
  { to: "/settings", label: "Settings",  icon: Settings },
];

export default function Layout() {
  return (
    <div className="flex h-full w-full">
      <aside className="flex h-full w-56 shrink-0 flex-col border-r bg-secondary/30">
        <div className="flex h-12 items-center border-b px-4">
          <div className="flex flex-col">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              reward sculptor
            </span>
            <span className="text-xs font-medium text-foreground">
              control panel
            </span>
          </div>
        </div>
        <nav className="flex flex-col gap-0.5 p-2">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2 rounded px-2 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
                  isActive
                    ? "bg-accent text-accent-foreground font-medium"
                    : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                )
              }
              end={item.to === "/"}
            >
              <item.icon className="h-4 w-4" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto border-t px-4 py-3 text-[10px] text-muted-foreground">
          v0.1 · localhost
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto scrollbar-thin">
        <Outlet />
      </main>
    </div>
  );
}

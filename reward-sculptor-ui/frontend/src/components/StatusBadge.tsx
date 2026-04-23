import { cn } from "@/lib/utils";
import type { ProjectStatus } from "@/lib/types";

const STATUS_STYLES: Record<ProjectStatus, { label: string; cls: string }> = {
  draft: {
    label: "draft",
    cls: "bg-muted text-muted-foreground border-border",
  },
  configured: {
    label: "configured",
    cls: "bg-muted text-muted-foreground border-border",
  },
  ready: {
    label: "ready",
    cls: "bg-blue-50 text-blue-700 border-blue-200",
  },
  running: {
    label: "running",
    cls: "bg-amber-50 text-amber-700 border-amber-200 animate-pulse",
  },
  completed: {
    label: "completed",
    cls: "bg-emerald-50 text-emerald-700 border-emerald-200",
  },
  errored: {
    label: "errored",
    cls: "bg-red-50 text-red-700 border-red-200",
  },
};

export function StatusBadge({ status }: { status: ProjectStatus }) {
  const s = STATUS_STYLES[status] ?? STATUS_STYLES.draft;
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        s.cls,
      )}
    >
      {s.label}
    </span>
  );
}

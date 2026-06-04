import { cn } from "@/lib/utils";
import type { ProjectStatus } from "@/lib/types";

// §Ship 21e (review fix): darkened text to clear WCAG AA 4.5:1 on the
// -50 tints (amber-700/emerald-700/blue-700 measured ~3-4:1 at this
// micro size; -800 clears it). `animate-pulse` guarded by motion-safe.
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
    cls: "bg-blue-50 text-blue-800 border-blue-200",
  },
  running: {
    label: "running",
    cls: "bg-amber-50 text-amber-800 border-amber-200 motion-safe:animate-pulse",
  },
  completed: {
    label: "completed",
    cls: "bg-emerald-50 text-emerald-800 border-emerald-200",
  },
  errored: {
    label: "errored",
    cls: "bg-red-50 text-red-800 border-red-200",
  },
};

export function StatusBadge({ status }: { status: ProjectStatus }) {
  const s = STATUS_STYLES[status] ?? STATUS_STYLES.draft;
  return (
    <span
      role="status"
      aria-label={`project status: ${s.label}`}
      className={cn(
        "inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        s.cls,
      )}
    >
      {s.label}
    </span>
  );
}

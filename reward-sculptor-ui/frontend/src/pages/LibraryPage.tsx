import { Link } from "react-router-dom";
import { ArrowLeft, Library, Sparkles } from "lucide-react";

import { LibraryBrowser } from "@/components/RobotConfig";
import { Button } from "@/components/ui/button";
import { useLibraryRobots, useSystemGpu } from "@/hooks/useLibrary";
import { cn } from "@/lib/utils";

/** Standalone library page — the primary entry point for creating a
 *  new project. Reuses LibraryBrowser from RobotConfig so the in-
 *  project robot picker and the top-level library share filter state
 *  + card rendering exactly. */
export default function LibraryPage() {
  const robots = useLibraryRobots();
  const gpu = useSystemGpu();

  const total = robots.data?.total ?? 0;
  const mjlabReady =
    robots.data?.robots.filter((r) => r.training_support === "mjlab_ready").length ?? 0;
  const cudaOk = !!gpu.data?.cuda_available;

  return (
    <div className="mx-auto flex h-full max-w-7xl flex-col gap-4 px-6 pb-16 pt-6">
      <header className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Button asChild variant="ghost" size="sm">
            <Link to="/" aria-label="Back to dashboard">
              <ArrowLeft className="h-4 w-4" />
              Dashboard
            </Link>
          </Button>
          <div>
            <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
              <Library className="h-5 w-5" />
              Robot library
            </h1>
            <p className="text-xs text-muted-foreground">
              Pick a robot to spin up a new project. Every mjlab-ready entry
              scaffolds with a batched reward template and auto-seeds the
              knowledge graph with canonical references.
            </p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-0.5 text-right text-xs">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5">
              <Sparkles className="h-3 w-3 text-emerald-500" />
              {mjlabReady} ready to train
            </span>
            <span className="rounded-full border px-2 py-0.5">
              {total} total
            </span>
          </div>
          <span
            className={cn(
              "text-[11px]",
              cudaOk ? "text-muted-foreground" : "text-amber-600",
            )}
          >
            {cudaOk
              ? `GPU: ${gpu.data?.devices[0]?.name ?? "detected"}`
              : "No GPU detected — mjlab adapter disabled"}
          </span>
        </div>
      </header>

      <LibraryBrowser />
    </div>
  );
}

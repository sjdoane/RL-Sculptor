import { useQuery } from "@tanstack/react-query";
import { Loader2, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { useStopJob } from "@/hooks/useJob";
import { ApiError, listJobs } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import { JOB_TERMINAL_STATES, type JobSummary } from "@/lib/types";

/** Small spinner + count for in-flight KG jobs on the current project.
 *  Shown in the KG tab header per Prompt 7 precondition R3: the user
 *  must be free to navigate away from the "Add papers" dialog while
 *  the job runs. Polls /jobs?project_slug= every 3s.
 *
 *  M7 Phase 6: adds a stop button that calls `POST /jobs/{id}/stop`
 *  so a stuck extract / ingest / research job can be killed from the
 *  UI (no more `curl -X POST /jobs/…/stop` workaround). */
export function ActiveJobsIndicator({ slug }: { slug: string }) {
  const { data } = useQuery<JobSummary[]>({
    queryKey: qk.projectJobs(slug),
    queryFn: () => listJobs({ projectSlug: slug }),
    refetchInterval: 3000,
    staleTime: 1500,
  });
  const stop = useStopJob(slug);

  const active = (data ?? []).filter(
    (j) =>
      !JOB_TERMINAL_STATES.includes(j.status) &&
      (j.kind === "kg_ingest" ||
        j.kind === "kg_extract" ||
        j.kind === "kg_ingest_extract" ||
        j.kind === "kg_research"),
  );
  if (active.length === 0) return null;
  const head = active[0];

  const onStop = () => {
    stop.mutate(head.job_id, {
      onSuccess: () => {
        toast.success("Job stopped", {
          description: `${head.kind} (${head.job_id.slice(0, 8)}…)`,
        });
      },
      onError: (err) => {
        const msg =
          err instanceof ApiError
            ? err.problem.detail ?? err.problem.title
            : err.message;
        toast.error("Could not stop job", { description: msg });
      },
    });
  };

  return (
    <div className="flex items-center gap-2 rounded-md border border-amber-300/50 bg-amber-50 px-2 py-1 text-[11px] text-amber-900">
      <Loader2 className="h-3 w-3 animate-spin" />
      <span className="truncate">
        {active.length > 1 ? `${active.length} jobs · ` : ""}
        {head.message ?? head.kind}
      </span>
      <Button
        variant="ghost"
        size="sm"
        onClick={onStop}
        disabled={stop.isPending}
        className="ml-1 h-5 w-5 p-0 text-amber-900 hover:bg-amber-100 hover:text-amber-900"
        title={`Stop this ${head.kind} job`}
      >
        <X className="h-3 w-3" />
      </Button>
    </div>
  );
}

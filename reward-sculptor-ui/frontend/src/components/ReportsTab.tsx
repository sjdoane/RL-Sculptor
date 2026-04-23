import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Download, FileText, Loader2, Play, RefreshCw } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ApiError } from "@/lib/api";
import { qk } from "@/lib/queryKeys";

async function fetchReportMd(slug: string): Promise<string> {
  const r = await fetch(`/api/projects/${slug}/reports/final_report.md`);
  if (r.status === 404) return ""; // "not built yet" signal
  if (!r.ok) {
    let body: Record<string, unknown> = {};
    try {
      body = await r.json();
    } catch {
      /* ignore */
    }
    throw new ApiError({
      type: String(body.type ?? "about:blank"),
      title: String(body.title ?? r.statusText),
      status: r.status,
      detail: String(body.detail ?? ""),
    });
  }
  return await r.text();
}

async function buildReport(slug: string): Promise<void> {
  const r = await fetch(`/api/projects/${slug}/reports/build`, {
    method: "POST",
  });
  if (!r.ok) {
    let body: Record<string, unknown> = {};
    try {
      body = await r.json();
    } catch {
      /* ignore */
    }
    throw new ApiError({
      type: String(body.type ?? "about:blank"),
      title: String(body.title ?? r.statusText),
      status: r.status,
      detail: String(body.detail ?? ""),
    });
  }
}

export function ReportsTab({ slug }: { slug: string }) {
  const qc = useQueryClient();
  const md = useQuery<string>({
    queryKey: [...qk.project(slug), "report", "md"],
    queryFn: () => fetchReportMd(slug),
    staleTime: 10_000,
  });
  const build = useMutation<void, Error, void>({
    mutationFn: () => buildReport(slug),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...qk.project(slug), "report"] });
      toast.success("Report built", {
        description: "final_report.md + final.mp4 regenerated",
      });
    },
    onError: (err) => {
      const detail =
        err instanceof ApiError
          ? err.problem.detail ?? err.problem.title
          : err.message;
      toast.error("Report build failed", { description: detail });
    },
  });

  const hasReport = (md.data ?? "").trim().length > 0;
  const mp4Url = `/api/projects/${slug}/reports/final.mp4`;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 py-3">
          <div>
            <CardTitle className="text-sm">Reports</CardTitle>
            <CardDescription className="text-[11px]">
              Final report + timelapse over a completed sculpt run
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant={hasReport ? "outline" : "default"}
              onClick={() => build.mutate()}
              disabled={build.isPending}
            >
              {build.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Building…
                </>
              ) : (
                <>
                  <RefreshCw /> {hasReport ? "Rebuild" : "Build report"}
                </>
              )}
            </Button>
            {hasReport && (
              <a
                href={`/api/projects/${slug}/reports/final_report.md`}
                download="final_report.md"
                className="inline-flex items-center gap-1 rounded-md border border-input bg-background px-2.5 py-1 text-xs hover:bg-accent"
              >
                <Download className="h-3.5 w-3.5" /> Markdown
              </a>
            )}
          </div>
        </CardHeader>
      </Card>

      {md.isLoading && (
        <p className="text-sm text-muted-foreground">Loading report…</p>
      )}

      {md.error && !md.isLoading && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="p-4 text-sm text-destructive">
            {(md.error as Error).message}
          </CardContent>
        </Card>
      )}

      {!md.isLoading && !md.error && !hasReport && (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 p-8 text-center">
            <FileText className="h-8 w-8 text-muted-foreground" />
            <p className="text-sm font-medium">No report built yet</p>
            <p className="text-xs text-muted-foreground">
              Complete a sculpt run, then click "Build report".
            </p>
          </CardContent>
        </Card>
      )}

      {hasReport && (
        <>
          <TimelapseCard slug={slug} mp4Url={mp4Url} />
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-sm">
                <FileText className="mr-1 inline h-3.5 w-3.5 align-text-bottom" />
                final_report.md
              </CardTitle>
            </CardHeader>
            <CardContent className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {md.data ?? ""}
              </ReactMarkdown>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function TimelapseCard({ mp4Url }: { slug: string; mp4Url: string }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0 py-3">
        <CardTitle className="text-sm">
          <Play className="mr-1 inline h-3.5 w-3.5 align-text-bottom" />
          Timelapse
        </CardTitle>
        <a
          href={mp4Url}
          download="final.mp4"
          className="inline-flex items-center gap-1 rounded-md border border-input bg-background px-2.5 py-1 text-xs hover:bg-accent"
        >
          <Download className="h-3.5 w-3.5" /> MP4
        </a>
      </CardHeader>
      <CardContent className="p-0">
        <video
          src={mp4Url}
          className="aspect-video w-full bg-slate-950"
          controls
          playsInline
          preload="metadata"
        >
          <track kind="captions" />
        </video>
      </CardContent>
    </Card>
  );
}

export default ReportsTab;

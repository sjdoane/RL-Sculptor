import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, EmptyState } from "@/components/rs/primitives";
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
  const r = await fetch(`/api/projects/${slug}/reports/build`, { method: "POST" });
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
      toast.success("Report built", { description: "final_report.md + final.mp4 regenerated" });
    },
    onError: (err) => {
      const detail = err instanceof ApiError ? err.problem.detail ?? err.problem.title : err.message;
      toast.error("Report build failed", { description: detail });
    },
  });

  const hasReport = (md.data ?? "").trim().length > 0;
  const mp4Url = `/api/projects/${slug}/reports/final.mp4`;
  const mdUrl = `/api/projects/${slug}/reports/final_report.md`;

  return (
    <div className="rs-scroll">
      <div className="rs-pad">
        <div className="rs-flex-between rs-wrap rs-gap-12" style={{ marginBottom: 22 }}>
          <div>
            <div className="rs-eyebrow">final_report.md</div>
            <h2 className="rs-h2" style={{ marginTop: 6 }}>Reports</h2>
          </div>
          <div className="rs-flex rs-gap-8">
            {hasReport && (
              <Btn
                kind="ghost"
                icon="copy"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(md.data ?? "");
                    toast.success("Markdown copied");
                  } catch {
                    toast.error("Clipboard unavailable");
                  }
                }}
              >
                Copy markdown
              </Btn>
            )}
            {hasReport && (
              <a href={mdUrl} download="final_report.md" className="rs-btn rs-btn-ghost">
                <Icon name="download" size={15} />
                Download
              </a>
            )}
            <Btn kind={hasReport ? "ghost" : "primary"} icon="refresh-cw" disabled={build.isPending} onClick={() => build.mutate()}>
              {build.isPending ? "Building…" : hasReport ? "Rebuild" : "Build report"}
            </Btn>
          </div>
        </div>

        {md.isLoading ? (
          <p className="rs-sub">Loading report…</p>
        ) : md.error ? (
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">{(md.error as Error).message}</span>
          </div>
        ) : !hasReport ? (
          <div className="rs-card">
            <EmptyState
              icon="file-text"
              title="No report built yet"
              sub="Complete a sculpt run, then click Build report to render final_report.md + the timelapse."
            />
          </div>
        ) : (
          <div className="rs-report rs-card rs-card-pad" style={{ padding: "32px 36px" }}>
            <div className="rs-viewer" style={{ marginBottom: 24 }}>
              <div className="rs-viewer-bar">
                <div className="rs-card-title"><Icon name="video" size={16} />final.mp4</div>
                <a href={mp4Url} download="final.mp4" className="rs-btn rs-btn-quiet rs-btn-sm">
                  <Icon name="download" size={14} />MP4
                </a>
              </div>
              <video src={mp4Url} className="rs-viewer-stage" style={{ width: "100%", aspectRatio: "16/9", background: "#16150f" }} controls playsInline preload="metadata">
                <track kind="captions" />
              </video>
            </div>
            <div className="rs-md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{md.data ?? ""}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ReportsTab;

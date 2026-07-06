import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";

import { ActuatorLimitsCard } from "@/components/ActuatorLimitsCard";
import { Icon } from "@/components/rs/icon";
import { Btn, EmptyState } from "@/components/rs/primitives";
import { usePolicies } from "@/hooks/usePolicies";
import { ApiError, policyExportUrl } from "@/lib/api";
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

// §Ship 25b (H2): decomposition-quality telemetry per mission.
interface MissionQualityRecord {
  mission_slug: string;
  goal: string;
  n_stages_at_start: number;
  n_stages_final: number;
  stages_executed: number;
  stages_succeeded: number;
  stage_success_rate: number | null;
  redecompositions: number;
  iterations_total: number;
  completed: boolean;
  halted_reason: string | null;
  recorded_at: string;
}

async function fetchMissionQuality(slug: string): Promise<MissionQualityRecord[]> {
  const r = await fetch(`/api/projects/${slug}/reports/mission-quality`);
  if (!r.ok) return [];
  const body = (await r.json()) as { missions?: MissionQualityRecord[] };
  return body.missions ?? [];
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

  const quality = useQuery<MissionQualityRecord[]>({
    queryKey: [...qk.project(slug), "report", "mission-quality"],
    queryFn: () => fetchMissionQuality(slug),
    staleTime: 10_000,
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

        <PoliciesCard slug={slug} />

        <ActuatorLimitsCard slug={slug} />

        {(quality.data?.length ?? 0) > 0 && (
          <div className="rs-card" style={{ marginBottom: 22 }}>
            <div className="rs-card-head">
              <div className="rs-card-title">
                <Icon name="layers" size={16} />Mission quality
              </div>
              <span className="rs-sub" style={{ fontSize: 12 }}>
                decomposition telemetry
              </span>
            </div>
            <div className="rs-card-pad rs-vgap-8">
              {quality.data!.map((m) => (
                <div
                  key={m.mission_slug}
                  className="rs-flex rs-gap-12 rs-wrap"
                  style={{
                    border: "1px solid var(--hairline)",
                    borderRadius: "var(--radius-md)",
                    padding: "10px 12px",
                    background: "var(--canvas-soft)",
                    alignItems: "baseline",
                    fontSize: 12.5,
                  }}
                >
                  <span style={{ fontWeight: 500, minWidth: 140 }} title={m.goal}>
                    {m.mission_slug}
                  </span>
                  <span className="rs-num">
                    stages {m.stages_succeeded}/{m.stages_executed}
                    {m.stage_success_rate != null &&
                      ` (${Math.round(m.stage_success_rate * 100)}%)`}
                  </span>
                  <span className="rs-num">
                    {m.n_stages_at_start === m.n_stages_final
                      ? `${m.n_stages_final} planned`
                      : `${m.n_stages_at_start}→${m.n_stages_final} planned`}
                  </span>
                  <span className="rs-num">
                    {m.redecompositions} redecompose{m.redecompositions === 1 ? "" : "s"}
                  </span>
                  <span className="rs-num">{m.iterations_total} iters</span>
                  <span
                    style={{
                      marginLeft: "auto",
                      color: m.completed ? "var(--st-emerald)" : "var(--st-amber)",
                      fontWeight: 500,
                    }}
                  >
                    {m.completed ? "completed" : (m.halted_reason ?? "halted")}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

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

// ── Trained policies (deployment-bundle export) ───────────────────────
function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

function PoliciesCard({ slug }: { slug: string }) {
  const policies = usePolicies(slug);
  const rows = policies.data ?? [];
  const best = rows.reduce<number | null>((acc, r) => {
    const v = r.fitness ?? r.primary_metric;
    if (v == null) return acc;
    return acc == null || v > acc ? v : acc;
  }, null);
  return (
    <div className="rs-card" style={{ marginBottom: 22 }}>
      <div className="rs-card-head">
        <div className="rs-card-title">
          <Icon name="package" size={16} />Trained policies
        </div>
        <span className="rs-sub" style={{ fontSize: 12 }}>
          self-contained bundles for sim-to-real deployment
        </span>
      </div>
      {policies.isLoading ? (
        <p className="rs-sub" style={{ padding: "12px 16px" }}>Loading…</p>
      ) : rows.length === 0 ? (
        <EmptyState
          icon="package"
          title="No trained checkpoints yet"
          sub="Each completed training iteration leaves an exportable checkpoint here — bundle it with its reward, env spec, and an ONNX/TorchScript policy in one click."
        />
      ) : (
        <div className="rs-card-pad rs-vgap-8">
          {rows.map((p) => {
            const metric = p.fitness ?? p.primary_metric;
            const isBest = best != null && metric != null && metric >= best;
            return (
              <div
                key={p.iter_index}
                className="rs-flex rs-gap-12 rs-wrap"
                style={{
                  border: "1px solid var(--hairline)",
                  borderRadius: "var(--radius-md)",
                  padding: "10px 12px",
                  background: "var(--canvas-soft)",
                  alignItems: "center",
                  fontSize: 12.5,
                }}
              >
                <span style={{ fontWeight: 500, minWidth: 60 }}>iter {p.iter_index}</span>
                {p.reward_version && (
                  <span className="rs-tag mono" style={{ fontSize: 10 }}>
                    reward {p.reward_version}
                  </span>
                )}
                {p.fitness != null ? (
                  <span className="rs-num" title="objective fitness (0-1)">
                    fit {p.fitness.toFixed(2)}
                  </span>
                ) : p.primary_metric != null ? (
                  <span className="rs-num" title="mean return">
                    {p.primary_metric.toFixed(1)}
                  </span>
                ) : null}
                {isBest && rows.length > 1 && (
                  <span className="rs-tag" style={{ fontSize: 10, color: "var(--st-emerald)" }}>best</span>
                )}
                <span className="rs-sub rs-num" style={{ fontSize: 11 }}>
                  {fmtBytes(p.checkpoint_bytes)}
                </span>
                <span style={{ marginLeft: "auto" }}>
                  <a
                    href={policyExportUrl(slug, p.iter_index)}
                    download
                    className="rs-btn rs-btn-ghost rs-btn-sm"
                    title="Download a zip with the checkpoint, ONNX/TorchScript policy, reward + env spec snapshots, and a DEPLOY.md recipe"
                  >
                    <Icon name="download" size={14} />Export
                  </a>
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default ReportsTab;

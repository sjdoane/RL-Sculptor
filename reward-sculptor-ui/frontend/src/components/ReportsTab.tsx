import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";

import { ActuatorLimitsCard } from "@/components/ActuatorLimitsCard";
import { Icon } from "@/components/rs/icon";
import { Btn, EmptyState } from "@/components/rs/primitives";
import { usePolicies } from "@/hooks/usePolicies";
import { ApiError, getReportsSources, policyExportUrl } from "@/lib/api";
import { qk } from "@/lib/queryKeys";

/** A report source: the project's standalone runs, or one mission. */
type ReportSource =
  | { kind: "project" }
  | { kind: "mission"; missionSlug: string };

const PROJECT_SOURCE_VALUE = "__project__";

function sourceToValue(s: ReportSource): string {
  return s.kind === "project" ? PROJECT_SOURCE_VALUE : s.missionSlug;
}

/** URL bases differ per source: project reports live under /reports,
 *  a mission's under /missions/{ms}/report. */
function reportBase(slug: string, source: ReportSource): string {
  return source.kind === "project"
    ? `/api/projects/${slug}/reports`
    : `/api/projects/${slug}/missions/${source.missionSlug}/report`;
}

async function fetchReportMd(slug: string, source: ReportSource): Promise<string> {
  const r = await fetch(`${reportBase(slug, source)}/final_report.md`);
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

async function buildReport(slug: string, source: ReportSource): Promise<void> {
  // The build endpoint is always /reports/build; a mission build passes
  // {mission_slug} in the body (empty body = project-runs build).
  const init: RequestInit = { method: "POST" };
  if (source.kind === "mission") {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify({ mission_slug: source.missionSlug });
  }
  const r = await fetch(`/api/projects/${slug}/reports/build`, init);
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

  // §Ship 20: report source picker — the project's standalone runs, or
  // any mission. Fed by GET /reports/sources.
  const sources = useQuery({
    queryKey: [...qk.project(slug), "report", "sources"],
    queryFn: () => getReportsSources(slug),
    staleTime: 10_000,
  });
  const [sourceValue, setSourceValue] = useState<string>(PROJECT_SOURCE_VALUE);
  // Auto-select the single mission when project runs is empty and exactly
  // one mission exists — kills the "empty Results tab" confusion when all
  // the work lives under a mission. Runs once, before the user picks.
  const autoSelected = useRef(false);
  useEffect(() => {
    if (autoSelected.current || !sources.data) return;
    const { project_runs, missions } = sources.data;
    if (project_runs.n_iters === 0 && missions.length === 1) {
      setSourceValue(missions[0].mission_slug);
    }
    autoSelected.current = true;
  }, [sources.data]);

  const source: ReportSource = useMemo(
    () =>
      sourceValue === PROJECT_SOURCE_VALUE
        ? { kind: "project" }
        : { kind: "mission", missionSlug: sourceValue },
    [sourceValue],
  );

  const md = useQuery<string>({
    queryKey: [...qk.project(slug), "report", "md", sourceToValue(source)],
    queryFn: () => fetchReportMd(slug, source),
    staleTime: 10_000,
  });
  const build = useMutation<void, Error, void>({
    mutationFn: () => buildReport(slug, source),
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
  const base = reportBase(slug, source);
  const mp4Url = `${base}/final.mp4`;
  const mdUrl = `${base}/final_report.md`;
  const missionList = sources.data?.missions ?? [];

  return (
    <div className="rs-scroll">
      <div className="rs-pad">
        <div className="rs-flex-between rs-wrap rs-gap-12" style={{ marginBottom: 22 }}>
          <div>
            <div className="rs-eyebrow">policies · report · actuator limits</div>
            <h2 className="rs-h2" style={{ marginTop: 6 }}>Results</h2>
          </div>
          <div className="rs-flex rs-gap-8">
            {missionList.length > 0 && (
              <div className="rs-select" style={{ display: "flex", alignItems: "center" }}>
                <select
                  value={sourceValue}
                  onChange={(e) => setSourceValue(e.target.value)}
                  aria-label="Report source"
                  title="Choose which run or mission to report on"
                >
                  <option value={PROJECT_SOURCE_VALUE}>
                    Project runs ({sources.data?.project_runs.n_iters ?? 0} iters)
                  </option>
                  {missionList.map((m) => (
                    <option key={m.mission_slug} value={m.mission_slug}>
                      Mission: {m.mission_slug}
                      {m.has_report ? " ✓" : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}
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
              sub={
                source.kind === "mission"
                  ? `Build the report for mission ${source.missionSlug} to render its per-stage report + stitched timelapse.`
                  : "Complete a sculpt run, then click Build report to render final_report.md + the timelapse."
              }
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
  // Rank on ONE scale: fitness (0-1) when any row has it, else the reward
  // metric — mixing the two in a single max lands "best" on the wrong row.
  const anyFitness = rows.some((r) => r.fitness != null);
  const rankOf = (r: (typeof rows)[number]) =>
    anyFitness ? r.fitness : r.primary_metric;
  const best = rows.reduce<number | null>((acc, r) => {
    const v = rankOf(r);
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
            const metric = rankOf(p);
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

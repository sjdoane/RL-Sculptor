import { useNavigate } from "react-router-dom";

import { Icon } from "@/components/rs/icon";
import { Badge, Btn, EmptyState, Sparkline } from "@/components/rs/primitives";
import { useDashboard, useSystemInfo } from "@/hooks/useDashboard";
import { useSystemGpu } from "@/hooks/useLibrary";
import { useProjects } from "@/hooks/useProjects";
import { formatRelative } from "@/lib/utils";
import type { JobSummary, ProjectSummary } from "@/lib/types";

function humanizeSlug(s: string | null | undefined): string {
  if (!s) return "—";
  return s
    .split(/[_-]/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function adapterShort(cls: string | null | undefined): string {
  if (!cls) return "—";
  if (cls.includes("mjlab")) return "mjlab";
  if (cls.toLowerCase().includes("gym")) return "gym_sb3";
  return cls.split(".").pop() ?? cls;
}

const JOB_KIND_LABEL: Record<string, string> = {
  sculpt_run: "sculpt run",
  mission_execute: "mission",
  mission_decompose: "planning",
  mission_stage_run: "stage",
  kg_ingest: "KG ingest",
  kg_extract: "KG extract",
  kg_ingest_extract: "KG ingest",
  kg_viz_render: "KG graph",
};

function ProjectCard({ p, onOpen }: { p: ProjectSummary; onOpen: () => void }) {
  return (
    <button className="rs-pcard" onClick={onOpen} aria-label={"Open project " + p.display_name}>
      <div className="rs-pcard-top">
        <div style={{ minWidth: 0 }}>
          <div className="rs-pcard-name" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {p.display_name}
          </div>
          <div className="rs-pcard-meta" style={{ marginTop: 6 }}>
            <Icon name="package" size={13} />
            {adapterShort(p.adapter_class)}
            <span style={{ color: "var(--hairline-strong)" }}>·</span>
            {humanizeSlug(p.library_slug) || (p.env_id ?? "—")}
          </div>
        </div>
        <Badge status={p.status} />
      </div>
      <div className="rs-pcard-foot">
        <div>
          <div className="rs-eyebrow" style={{ marginBottom: 4 }}>Best reward</div>
          <div
            className="rs-num"
            style={{ fontSize: 22, color: p.primary_metric == null ? "var(--rs-muted-soft)" : "var(--ink)" }}
          >
            {p.primary_metric == null ? "—" : p.primary_metric.toFixed(1)}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <Sparkline
            data={p.primary_metric_history ?? []}
            w={92}
            h={30}
            color={p.status === "errored" ? "var(--st-rose)" : "var(--st-emerald)"}
          />
          <div className="rs-pcard-meta" style={{ justifyContent: "flex-end", marginTop: 4, fontSize: 11.5 }}>
            <Icon name="clock" size={11} />
            {formatRelative(p.created_at)}
          </div>
        </div>
      </div>
    </button>
  );
}

function ActiveJobRow({ job, onOpen }: { job: JobSummary; onOpen: () => void }) {
  const pct = job.progress != null ? Math.round(job.progress * 100) : null;
  return (
    <div className="rs-job">
      <span className="rs-dot live" />
      <div style={{ minWidth: 150, flex: 1, overflow: "hidden" }}>
        <div className="rs-job-name" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {job.project_slug ?? "global"}
        </div>
        <div className="rs-pcard-meta" style={{ fontSize: 11.5, marginTop: 2 }}>
          {JOB_KIND_LABEL[job.kind] ?? job.kind}
          {job.message ? ` · ${job.message}` : ""}
        </div>
      </div>
      {pct != null && (
        <div className="rs-job-bar">
          <i style={{ width: pct + "%" }} />
        </div>
      )}
      {job.project_slug && (
        <Btn kind="ghost" size="sm" icon="arrow-right" onClick={onOpen}>
          Open
        </Btn>
      )}
    </div>
  );
}

function SystemCard() {
  const gpu = useSystemGpu({ refetchIntervalMs: 5000 });
  const info = useSystemInfo();
  const dev = gpu.data?.devices?.[0];
  const total = dev?.total_memory_bytes ?? 0;
  const used =
    dev?.used_memory_bytes ??
    (dev && dev.free_memory_bytes != null ? total - dev.free_memory_bytes : 0);
  const pct = total > 0 ? (used / total) * 100 : 0;
  return (
    <div className="rs-card rs-card-pad">
      <div className="rs-sysgrid">
        <div className="rs-sysitem">
          <span className="lab">GPU</span>
          <span className="val">{dev?.name ?? "No GPU detected"}</span>
          <div className="rs-vram" style={{ marginTop: 4 }}>
            <i style={{ width: pct + "%" }} />
          </div>
          <span className="rs-num" style={{ fontSize: 12, color: "var(--rs-muted)" }}>
            {total > 0 ? `${(used / 1e9).toFixed(1)} / ${(total / 1e9).toFixed(1)} GB VRAM` : "— GB"}
          </span>
        </div>
        <div className="rs-sysitem">
          <span className="lab">Temperature</span>
          <span className="val">{dev?.temperature_c != null ? `${Math.round(dev.temperature_c)}°C` : "—"}</span>
          <span className="rs-sub" style={{ fontSize: 12 }}>
            {dev?.utilization_percent != null ? `${Math.round(dev.utilization_percent)}% util` : "nominal"}
          </span>
        </div>
        <div className="rs-sysitem">
          <span className="lab">CUDA</span>
          <span className="val" style={{ color: gpu.data?.cuda_available ? "var(--st-emerald)" : "var(--rs-muted)" }}>
            {gpu.data?.cuda_available ? `detected · ${gpu.data.cuda_version ?? ""}`.trim() : "not detected"}
          </span>
          {gpu.data?.torch_version && (
            <span className="rs-sub" style={{ fontSize: 12 }}>torch {gpu.data.torch_version}</span>
          )}
        </div>
        <div className="rs-sysitem">
          <span className="lab">Anthropic API key</span>
          <span
            className="val"
            style={{
              color: info.data?.anthropic_api_key_set ? "var(--st-emerald)" : "var(--st-rose)",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <Icon name={info.data?.anthropic_api_key_set ? "check-circle" : "alert-circle"} size={15} />
            {info.data?.anthropic_api_key_set ? "set" : "not set"}
          </span>
          {info.data?.anthropic_api_key_masked && (
            <span className="rs-sub mono" style={{ fontSize: 12 }}>{info.data.anthropic_api_key_masked}</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const nav = useNavigate();
  const dash = useDashboard();
  const projects = useProjects();
  const active = dash.data?.active_jobs ?? [];
  const list = projects.data ?? [];

  return (
    <div className="rs-scroll">
      <div className="rs-pad rs-vgap-32">
        <div className="rs-flex-between rs-wrap rs-gap-12">
          <div>
            <div className="rs-eyebrow">Localhost · control panel</div>
            <h1 className="rs-h2" style={{ marginTop: 6 }}>Dashboard</h1>
          </div>
          <Btn kind="primary" icon="plus" onClick={() => nav("/library")}>
            New project
          </Btn>
        </div>

        {/* Active jobs */}
        <section>
          <div className="rs-flex rs-gap-8" style={{ marginBottom: 12 }}>
            <span className={"rs-dot" + (active.length ? " live" : "")} style={active.length ? undefined : { background: "var(--st-slate)" }} />
            <h2 className="rs-h3">Active jobs</h2>
            <span className="rs-num" style={{ color: "var(--rs-muted)", fontSize: 13 }}>{active.length} running</span>
          </div>
          <div className="rs-card rs-jobs">
            {active.length === 0 ? (
              <div style={{ padding: 18 }}>
                <EmptyState icon="activity" title="Nothing running" sub="Launched runs and missions appear here while they train." />
              </div>
            ) : (
              active.map((j) => (
                <ActiveJobRow key={j.job_id} job={j} onOpen={() => j.project_slug && nav(`/projects/${j.project_slug}`)} />
              ))
            )}
          </div>
        </section>

        {/* Projects grid */}
        <section>
          <h2 className="rs-h3" style={{ marginBottom: 12 }}>Projects</h2>
          {projects.isLoading ? (
            <div className="rs-sub">Loading projects…</div>
          ) : projects.error ? (
            <div className="rs-banner err">
              <Icon name="alert-triangle" size={17} />
              <span className="rs-grow">Failed to load projects: {(projects.error as Error).message}</span>
            </div>
          ) : (
            <div className="rs-grid-cards">
              {list.map((p) => (
                <ProjectCard key={p.slug} p={p} onOpen={() => nav(`/projects/${p.slug}`)} />
              ))}
              <button className="rs-newcard" onClick={() => nav("/library")}>
                <Icon name="plus" size={22} />
                <span style={{ fontSize: 14, fontWeight: 500 }}>New project</span>
              </button>
            </div>
          )}
        </section>

        {/* System */}
        <section>
          <h2 className="rs-h3" style={{ marginBottom: 12 }}>System</h2>
          <SystemCard />
        </section>
      </div>
    </div>
  );
}

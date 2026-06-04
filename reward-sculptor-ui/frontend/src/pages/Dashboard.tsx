import { Link } from "react-router-dom";
import {
  ArrowRight,
  Compass,
  ExternalLink,
  Library,
  Play,
  Radio,
  Sparkles,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useDashboard } from "@/hooks/useDashboard";
import { useSystemGpu } from "@/hooks/useLibrary";
import { cn, formatRelative } from "@/lib/utils";
import type {
  DashboardKGAddition,
  DashboardRecentRun,
  JobStatus,
  JobSummary,
} from "@/lib/types";

export default function Dashboard() {
  const { data, isLoading, error } = useDashboard();

  if (isLoading) {
    return (
      <Shell>
        <p className="text-sm text-muted-foreground">Loading dashboard…</p>
      </Shell>
    );
  }
  if (error) {
    return (
      <Shell>
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="p-4 text-sm text-destructive">
            Dashboard failed to load: {(error as Error).message}
          </CardContent>
        </Card>
      </Shell>
    );
  }
  if (!data || !data.has_any_projects) {
    return (
      <Shell>
        <EmptyState />
      </Shell>
    );
  }

  return (
    <Shell>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 flex flex-col gap-4">
          <ActiveJobsCard jobs={data.active_jobs} />
          <RecentRunsCard runs={data.recent_runs} />
        </div>
        <div className="flex flex-col gap-4">
          <QuickLaunchCard hasProjects={data.has_any_projects} />
          <GpuWidget activeJobs={data.active_jobs} />
          <KGAdditionsCard additions={data.recent_kg_additions} />
        </div>
      </div>
    </Shell>
  );
}

// ── GPU widget (M4) ──────────────────────────────────────────────────
function GpuWidget({ activeJobs }: { activeJobs: JobSummary[] }) {
  // Refresh faster (3 s) when at least one job is running; else 10 s.
  const hasRunning = activeJobs.some((j) => j.status === "running");
  const { data } = useSystemGpu({ refetchIntervalMs: hasRunning ? 3000 : 10000 });
  if (!data || !data.cuda_available) return null;
  const devices = data.devices ?? [];
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Radio className={cn("h-3.5 w-3.5", hasRunning && "motion-safe:animate-pulse text-emerald-500")} />
          GPUs
        </CardTitle>
        <CardDescription className="text-[11px]">
          {hasRunning
            ? `${activeJobs.filter((j) => j.status === "running").length} run(s) active`
            : "Idle"}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 pt-0">
        {devices.map((d) => {
          const memPct =
            d.total_memory_bytes > 0
              ? ((d.used_memory_bytes ?? d.total_memory_bytes - d.free_memory_bytes) /
                  d.total_memory_bytes) *
                100
              : 0;
          const utilPct = d.utilization_percent ?? 0;
          return (
            <div key={d.index} className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between">
                <span className="truncate text-xs font-medium">{d.name}</span>
                {d.temperature_c != null && (
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {Math.round(d.temperature_c)}°C
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="w-10 text-muted-foreground">mem</span>
                <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className={cn(
                      "h-1 transition-[width]",
                      memPct > 85 ? "bg-rose-500" : memPct > 70 ? "bg-amber-500" : "bg-emerald-500",
                    )}
                    style={{ width: `${memPct}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono">
                  {memPct.toFixed(0)}%
                </span>
              </div>
              {d.utilization_percent != null && (
                <div className="flex items-center gap-1.5 text-[10px]">
                  <span className="w-10 text-muted-foreground">util</span>
                  <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-1 bg-sky-500 transition-[width]"
                      style={{ width: `${utilPct}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono">
                    {utilPct.toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

// ── shell ────────────────────────────────────────────────────────────
function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-4 px-6 pb-16 pt-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Dashboard</h1>
          <p className="text-xs text-muted-foreground">
            At-a-glance view across all projects
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button asChild variant="outline" size="sm">
            <Link to="/projects">
              All projects <ArrowRight />
            </Link>
          </Button>
          <Button asChild size="sm">
            <Link to="/library">
              <Library className="h-4 w-4" />
              New from library
            </Link>
          </Button>
        </div>
      </header>
      {children}
    </div>
  );
}

// ── starter step card ───────────────────────────────────────────────
function StarterStep({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <Card>
      <CardContent className="flex flex-col gap-2 p-4">
        <div className="flex items-center gap-2 text-xs font-medium text-foreground">
          <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary">
            {icon}
          </span>
          {title}
        </div>
        <p className="text-xs text-muted-foreground">{body}</p>
      </CardContent>
    </Card>
  );
}

// ── empty state ──────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="flex flex-col gap-4">
      <Card className="overflow-hidden border-0 bg-gradient-to-br from-primary/10 via-background to-background">
        <CardContent className="flex flex-col items-start gap-5 px-8 py-12">
          <div className="flex items-center gap-2 rounded-full border bg-background/80 px-3 py-1 text-[11px] font-medium backdrop-blur">
            <Sparkles className="h-3 w-3 text-amber-500" />
            Welcome to Reward Sculptor
          </div>
          <div className="max-w-2xl space-y-2">
            <h2 className="text-2xl font-semibold tracking-tight">
              Sculpt rewards on 63 robots, grounded in RL literature.
            </h2>
            <p className="text-sm text-muted-foreground">
              Pick a robot from the library, describe the behavior you want,
              and Claude iterates on the reward function — training,
              diagnosing failures, querying a knowledge graph of RL
              papers, and rewriting with citations. Live clips and
              metrics stream back here.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button asChild size="lg">
              <Link to="/library">
                <Library className="h-4 w-4" />
                Browse the library
              </Link>
            </Button>
            <a
              href="https://github.com/sjdoane/RL-Sculptor"
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex items-center gap-1 text-sm text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
            >
              What is Reward Sculptor? <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <StarterStep
          icon={<Library className="h-4 w-4" />}
          title="1. Pick a robot"
          body="6 mjlab-ready platforms (G1, Go1, ANYmal-C, T1, Yam, Cartpole) train on GPU in minutes. 5 legacy Gymnasium envs for CPU. Or upload a custom URDF."
        />
        <StarterStep
          icon={<Compass className="h-4 w-4" />}
          title="2. Describe the behavior"
          body="Natural language: e.g. 'walk forward without falling'. Sculptor captures this as the run's goal and surfaces it to Claude on every diagnose + edit."
        />
        <StarterStep
          icon={<Play className="h-4 w-4" />}
          title="3. Launch a sculpt run"
          body="Dry-run (no API calls) finishes in ~60s on CPU. Live runs iterate train → rollout → diagnose → rewrite → commit — reward history + paper citations saved per-iteration."
        />
      </div>
    </div>
  );
}

// ── active jobs ──────────────────────────────────────────────────────
function ActiveJobsCard({ jobs }: { jobs: JobSummary[] }) {
  const sculptRuns = jobs.filter((j) => j.kind === "sculpt_run");
  const otherJobs = jobs.filter((j) => j.kind !== "sculpt_run");
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Radio
            className={cn(
              "h-3.5 w-3.5",
              jobs.length > 0 ? "text-rose-600 motion-safe:animate-pulse" : "text-muted-foreground",
            )}
          />
          Active jobs
          <span className="text-[10px] font-normal text-muted-foreground">
            {jobs.length} running
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 pt-0">
        {jobs.length === 0 ? (
          <p className="py-2 text-xs text-muted-foreground">Nothing running.</p>
        ) : (
          <>
            {sculptRuns.map((j) => (
              <Link
                key={j.job_id}
                to={`/projects/${j.project_slug}`}
                className="flex items-center justify-between gap-2 rounded-md border bg-muted/20 px-3 py-2 text-xs transition-colors hover:bg-accent"
              >
                <div className="flex flex-col gap-0.5 overflow-hidden">
                  <span className="truncate font-medium">
                    {j.project_slug} · sculpt run
                  </span>
                  <span className="truncate text-[10px] text-muted-foreground">
                    {j.message ?? "running"}
                  </span>
                </div>
                <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" />
              </Link>
            ))}
            {otherJobs.map((j) => (
              <div
                key={j.job_id}
                className="flex items-center gap-2 rounded-md border bg-muted/20 px-3 py-2 text-xs"
              >
                <span className="font-mono text-[10px] text-muted-foreground">
                  {j.kind}
                </span>
                <span className="truncate">{j.project_slug ?? "global"}</span>
                <span className="ml-auto text-[10px] text-muted-foreground">
                  {j.message ?? j.status}
                </span>
              </div>
            ))}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// ── recent runs ──────────────────────────────────────────────────────
function RecentRunsCard({ runs }: { runs: DashboardRecentRun[] }) {
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Recent runs</CardTitle>
        <CardDescription className="text-[11px]">
          Last 10 sculpt runs across every project
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        {runs.length === 0 ? (
          <p className="py-2 text-xs text-muted-foreground">
            No runs yet.
          </p>
        ) : (
          <ul className="divide-y">
            {runs.map((r) => (
              <li key={r.run_id}>
                <Link
                  to={`/projects/${r.project_slug}`}
                  className="flex items-center justify-between gap-3 px-0 py-2 text-xs transition-colors hover:bg-accent/40"
                >
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-medium">
                        {r.project_display_name}
                      </span>
                      <JobStatusPill status={r.status} />
                    </div>
                    <span className="truncate text-[10px] text-muted-foreground">
                      {r.behavior_goal || "(no goal)"} ·{" "}
                      {r.iterations_completed}/{r.iterations_requested} iters ·{" "}
                      {r.started_at ? formatRelative(r.started_at) : "—"}
                    </span>
                  </div>
                  <Sparkline history={r.primary_metric_history} />
                </Link>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function JobStatusPill({ status }: { status: JobStatus }) {
  const map: Record<JobStatus, string> = {
    queued:    "bg-muted text-muted-foreground border-border",
    running:   "bg-amber-50 text-amber-800 border-amber-200 motion-safe:animate-pulse",
    completed: "bg-emerald-50 text-emerald-700 border-emerald-200",
    errored:   "bg-rose-50 text-rose-700 border-rose-200",
    stopped:   "bg-slate-100 text-slate-600 border-slate-200",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide",
        map[status] ?? map.queued,
      )}
    >
      {status}
    </span>
  );
}

function Sparkline({ history }: { history: Array<number | null> }) {
  const nums = history.filter((v): v is number => typeof v === "number");
  if (nums.length < 2) return <span className="h-3 w-16 shrink-0" />;
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const range = max - min || 1;
  const w = 64;
  const h = 16;
  const step = w / (nums.length - 1);
  const points = nums
    .map((v, i) => `${(i * step).toFixed(1)},${(h - ((v - min) / range) * h).toFixed(1)}`)
    .join(" ");
  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      className="shrink-0 text-foreground"
      aria-label={`Metric trajectory: ${nums.map((n) => n.toFixed(2)).join(", ")}`}
    >
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth={1} />
    </svg>
  );
}

// ── quick launch ─────────────────────────────────────────────────────
function QuickLaunchCard({ hasProjects }: { hasProjects: boolean }) {
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Quick launch</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 pt-0">
        <Button asChild size="sm">
          <Link to="/library">
            <Play />
            New from library
          </Link>
        </Button>
        {hasProjects && (
          <Button asChild variant="outline" size="sm">
            <Link to="/projects">
              Browse existing projects
            </Link>
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// ── kg additions ─────────────────────────────────────────────────────
function KGAdditionsCard({ additions }: { additions: DashboardKGAddition[] }) {
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Recent KG additions</CardTitle>
        <CardDescription className="text-[11px]">
          Papers recently ingested into any project
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        {additions.length === 0 ? (
          <p className="py-2 text-xs text-muted-foreground">None yet.</p>
        ) : (
          <ul className="space-y-1">
            {additions.map((p) => (
              <li key={`${p.project_slug}-${p.arxiv_id}`}>
                <Link
                  to={`/projects/${p.project_slug}`}
                  className="flex flex-col gap-0.5 rounded-md px-2 py-1.5 text-xs hover:bg-accent/40"
                >
                  <span className="truncate font-medium">{p.title}</span>
                  <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                    <span className="font-mono">{p.arxiv_id}</span>
                    {p.extracted ? (
                      <span className="rounded-sm bg-emerald-100 px-1 py-0 text-[9px] font-semibold uppercase tracking-wide text-emerald-800">
                        extracted
                      </span>
                    ) : (
                      <span className="rounded-sm bg-muted px-1 py-0 text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">
                        pending
                      </span>
                    )}
                    <span>· {p.project_display_name}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

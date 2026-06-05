import { lazy, Suspense, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ExternalLink, Loader2, Wrench } from "lucide-react";

import { Badge as UIBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Icon } from "@/components/rs/icon";
import { Badge, Btn, EmptyState, FactChip } from "@/components/rs/primitives";
import { KnowledgeGraphTab } from "@/components/KnowledgeGraphTab";
import { PhysicsTab } from "@/components/PhysicsTab";
import { ProjectSettingsDialog } from "@/components/ProjectSettingsDialog";
import { NewRunDialog } from "@/components/NewRunDialog";
import { RewardsTab } from "@/components/RewardsTab";
import { RobotConfig } from "@/components/RobotConfig";
import { RobotViewer } from "@/components/RobotViewer";
import { useLibraryRobot } from "@/hooks/useLibrary";
import { usePhysics } from "@/hooks/usePhysics";
import { useProject } from "@/hooks/useProjects";
import { useRobot } from "@/hooks/useRobot";
import { formatRelative } from "@/lib/utils";
import type { ProjectDetail as ProjectDetailShape, RobotStateResponse } from "@/lib/types";

const RunsTabLazy = lazy(() => import("@/components/RunsTab"));
const ReportsTabLazy = lazy(() => import("@/components/ReportsTab"));

const TABS = [
  { value: "overview", label: "Overview", icon: "gauge" },
  { value: "rewards", label: "Rewards", icon: "file-code" },
  { value: "physics", label: "Physics", icon: "cpu" },
  { value: "kg", label: "Knowledge Graph", icon: "network" },
  { value: "runs", label: "Runs", icon: "activity" },
  { value: "reports", label: "Reports", icon: "file-text" },
] as const;

function humanizeSlug(s: string | null | undefined): string {
  if (!s) return "";
  return s.split(/[_-]/).filter(Boolean).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}
function adapterShort(cls: string | null | undefined): string {
  if (!cls) return "—";
  if (cls.includes("mjlab")) return "mjlab";
  if (cls.toLowerCase().includes("gym")) return "gym_sb3";
  return cls.split(".").pop() ?? cls;
}

// Transitional wrapper for tabs not yet reskinned (Overview/Physics/
// Rewards). Mirrors the old ProjectDetail scroll container so the legacy
// shadcn content keeps scrolling inside the new rs- shell.
function LegacyTab({ children }: { children: React.ReactNode }) {
  return (
    <div className="rs-scroll">
      <div style={{ padding: "24px 32px 60px", maxWidth: 1320 }}>{children}</div>
    </div>
  );
}

function FactsBand({ project }: { project: ProjectDetailShape }) {
  const cfg = project.adapter_config || {};
  const taskId = typeof cfg.task_id === "string" ? cfg.task_id : null;
  const numEnvs = typeof cfg.num_envs === "number" ? cfg.num_envs : null;
  const device = typeof cfg.device === "string" ? cfg.device : null;
  const robot = humanizeSlug(project.library_slug);
  return (
    <div className="rs-facts">
      <FactChip k="adapter" icon="package">{adapterShort(project.adapter_class)}</FactChip>
      {robot && <FactChip k="robot" icon="bot">{robot}</FactChip>}
      {(taskId || project.env_id) && <FactChip k="task" icon="target">{taskId ?? project.env_id}</FactChip>}
      {device && <FactChip k="device" icon="cpu">{device}</FactChip>}
      {numEnvs != null && <FactChip k="num_envs" icon="layers">{numEnvs.toLocaleString()}</FactChip>}
    </div>
  );
}

function WarningBanners({ project }: { project: ProjectDetailShape }) {
  const adapterShortName = (project.adapter_class || "").split(".").pop() || "—";
  return (
    <div style={{ padding: "12px 32px 0" }} className="rs-vgap-8">
      {project.adapter_unavailable && (
        <div className="rs-banner warn">
          <Icon name="clock" size={17} />
          <span className="rs-grow">
            <b>Adapter not yet implemented — training disabled.</b> Created with a coming-soon
            adapter (<code className="mono">{adapterShortName}</code>). Reward editing + preview
            still work.
          </span>
        </div>
      )}
      {project.migration_warning && (
        <div className="rs-banner warn">
          <Icon name="alert-triangle" size={17} />
          <span className="rs-grow">{project.migration_warning}</span>
          <Btn kind="ghost" size="sm" icon="library" onClick={() => { window.location.href = "/library"; }}>
            Fork to a current adapter
          </Btn>
        </div>
      )}
    </div>
  );
}

export default function ProjectDetail() {
  const { slug } = useParams<{ slug: string }>();
  const nav = useNavigate();
  const project = useProject(slug);
  const robot = useRobot(slug);
  const [tab, setTab] = useState<string>("overview");
  const p = project.data;
  const canRun = !!p && !p.adapter_unavailable && p.ready_to_train !== false;

  return (
    <>
      <div className="rs-phead">
        <button className="rs-back" onClick={() => nav("/projects")} aria-label="Back to projects">
          <Icon name="arrow-left" size={17} />
        </button>
        <div className="rs-phead-title">
          <Icon name="folder" size={18} color="var(--rs-muted)" />
          <span className="rs-phead-name" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {p?.display_name ?? slug}
          </span>
          {p && <Badge status={p.status} big />}
        </div>
        <div className="rs-phead-spacer" />
        {p && <ProjectSettingsDialog project={p} />}
        {p && canRun && <NewRunDialog slug={slug!} project={p} onLaunched={() => setTab("runs")} />}
      </div>

      {project.isLoading ? (
        <LegacyTab><p className="rs-sub">Loading…</p></LegacyTab>
      ) : project.error ? (
        <LegacyTab>
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">Could not load project: {(project.error as Error).message}</span>
          </div>
        </LegacyTab>
      ) : !p ? (
        <LegacyTab><p className="rs-sub">No project.</p></LegacyTab>
      ) : (
        <>
          <div className="rs-tabs" role="tablist">
            {TABS.map((t) => (
              <button
                key={t.value}
                role="tab"
                aria-selected={tab === t.value}
                className={"rs-tab" + (tab === t.value ? " on" : "")}
                onClick={() => setTab(t.value)}
              >
                <Icon name={t.icon} size={15} />
                {t.label}
              </button>
            ))}
          </div>

          {tab !== "runs" && <FactsBand project={p} />}
          {tab !== "runs" && (p.adapter_unavailable || p.migration_warning) && <WarningBanners project={p} />}

          {tab === "overview" && (
            <LegacyTab><OverviewTab slug={slug!} project={p} robot={robot.data} /></LegacyTab>
          )}
          {tab === "rewards" && (
            <LegacyTab><RewardsTab slug={slug!} project={p} /></LegacyTab>
          )}
          {tab === "physics" && (
            <LegacyTab><PhysicsTab slug={slug!} project={p} /></LegacyTab>
          )}
          {tab === "kg" && <KnowledgeGraphTab slug={slug!} />}
          {tab === "runs" && (
            <Suspense fallback={<TabFallback />}>
              <RunsTabLazy slug={slug!} project={p} />
            </Suspense>
          )}
          {tab === "reports" && (
            <Suspense fallback={<TabFallback />}>
              <ReportsTabLazy slug={slug!} />
            </Suspense>
          )}
        </>
      )}
    </>
  );
}

function TabFallback() {
  return (
    <div className="rs-scroll">
      <div className="rs-empty">
        <Loader2 className="rs-spin" />
        <span className="rs-sub">Loading…</span>
      </div>
    </div>
  );
}

// ── Overview (transitional — reskinned in a later ship) ───────────────
function OverviewTab({
  slug, project, robot,
}: { slug: string; project: ProjectDetailShape; robot: RobotStateResponse | undefined }) {
  const configured = isRobotConfigured(robot, project);
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <div className="lg:col-span-2 flex flex-col gap-4">
        {configured ? <RobotViewer slug={slug} /> : <RobotConfig slug={slug} />}
        <Card>
          <CardHeader><CardTitle className="text-sm">Description</CardTitle></CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">{project.description || "No description."}</p>
          </CardContent>
        </Card>
      </div>
      <div className="flex flex-col gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Metadata</CardTitle>
            <CardDescription>Read from <code>config.toml</code> + sidecar.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <KV k="slug" v={project.slug} mono />
            <KV k="adapter" v={project.adapter_class} mono />
            <KV k="env_id" v={project.env_id && project.env_id !== "CHANGE_ME" ? project.env_id : "—"} mono />
            <KV k="iterations" v={String(project.n_iterations_completed)} />
            <KV k="created" v={formatRelative(project.created_at)} />
            <KV k="directory" v={project.project_dir} mono truncate />
          </CardContent>
        </Card>
        {configured && robot && (
          <Card>
            <CardHeader><CardTitle className="text-sm">Robot</CardTitle></CardHeader>
            <CardContent className="space-y-2 text-xs">
              <KV k="kind" v={robot.kind} mono />
              {robot.library_name && <KV k="library" v={robot.library_name} mono />}
              {robot.env_id && <KV k="env_id" v={robot.env_id} mono />}
              {robot.model_file && <KV k="model" v={robot.model_file} mono truncate />}
              {robot.original_filename && <KV k="uploaded" v={robot.original_filename} mono truncate />}
              {robot.mesh_paths.length > 0 && <KV k="meshes" v={`${robot.mesh_paths.length}`} />}
            </CardContent>
          </Card>
        )}
        {configured && (robot?.library_name || project.library_slug) && (
          <RobotLibraryCard slug={slug} librarySlug={(robot?.library_name ?? project.library_slug)!} />
        )}
      </div>
    </div>
  );
}

function isRobotConfigured(robot: RobotStateResponse | undefined, project: ProjectDetailShape): boolean {
  if (!robot) return false;
  if (robot.kind === "none") return false;
  if (robot.kind === "library") return !!(robot.library_name || project.library_slug);
  return !!robot.model_file;
}

function KV({ k, v, mono, truncate }: { k: string; v: string; mono?: boolean; truncate?: boolean }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="shrink-0 text-[10px] uppercase tracking-wider text-muted-foreground">{k}</span>
      <span
        className={`min-w-0 flex-1 ${mono ? "font-mono text-[11px]" : ""} ${truncate ? "truncate" : "break-words"}`}
        title={truncate ? v : undefined}
      >
        {v}
      </span>
    </div>
  );
}

function RobotLibraryCard({ slug, librarySlug }: { slug: string; librarySlug: string }) {
  const lib = useLibraryRobot(librarySlug);
  const phys = usePhysics(slug);
  const entry = lib.data;
  if (lib.isLoading) {
    return (
      <Card>
        <CardHeader><CardTitle className="text-sm">Library entry</CardTitle></CardHeader>
        <CardContent className="text-xs text-muted-foreground"><Loader2 className="inline h-3 w-3 animate-spin" /> Loading…</CardContent>
      </Card>
    );
  }
  if (!entry) return null;
  const summary = phys.data?.summary;
  const njnt = summary?.joints?.length ?? null;
  const nu = summary?.actuators?.length ?? null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          {entry.display_name}
          <UIBadge variant="outline" className="text-[10px]">{entry.category}</UIBadge>
        </CardTitle>
        <CardDescription className="text-xs">{entry.description || "Library-sourced robot."}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <KV k="slug" v={entry.slug} mono />
        <KV k="source" v={entry.source} mono />
        {entry.menagerie_package && <KV k="menagerie" v={entry.menagerie_package} mono truncate />}
        <KV k="training" v={entry.training_support} mono />
        {njnt !== null && <KV k="joints" v={String(njnt)} mono />}
        {nu !== null && <KV k="actuators" v={String(nu)} mono />}
        {summary?.parse_error && (
          <p className="rounded border border-amber-500/40 bg-amber-500/5 px-1.5 py-1 text-[10px] text-amber-700 dark:text-amber-300">
            MJCF parse error — check the Physics tab: {summary.parse_error}
          </p>
        )}
        {entry.references.length > 0 && (
          <div>
            <div className="mb-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">References</div>
            <ul className="space-y-0.5">
              {entry.references.slice(0, 6).map((r, i) => (
                <li key={i} className="flex items-baseline gap-1.5 text-[10.5px]">
                  <a href={r.url} target="_blank" rel="noreferrer noopener"
                    className="inline-flex items-center gap-1 truncate text-foreground underline-offset-2 hover:underline" title={r.citation}>
                    <span className="truncate">{r.citation}</span>
                    <ExternalLink className="h-3 w-3 shrink-0" />
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="pt-1">
          <Button asChild size="sm" variant="outline" className="w-full">
            <a href="#physics" onClick={(e) => {
              e.preventDefault();
              const t = document.querySelector<HTMLButtonElement>('[role="tab"]:nth-of-type(3)');
              t?.click();
            }}>
              <Wrench className="mr-1 h-3 w-3" />
              Edit robot in Physics tab
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

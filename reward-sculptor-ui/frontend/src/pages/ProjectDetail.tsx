import { lazy, Suspense, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  Box,
  BookOpen,
  Cpu,
  ExternalLink,
  Hourglass,
  Library,
  Loader2,
  Wrench,
  Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { KnowledgeGraphTab } from "@/components/KnowledgeGraphTab";
import { MissionsTab } from "@/components/MissionsTab";
import { PhysicsTab } from "@/components/PhysicsTab";
import { ProjectSettingsDialog } from "@/components/ProjectSettingsDialog";
import { NewRunDialog } from "@/components/NewRunDialog";
import { RewardsTab } from "@/components/RewardsTab";
import { RobotConfig } from "@/components/RobotConfig";
import { RobotViewer } from "@/components/RobotViewer";
import { StatusBadge } from "@/components/StatusBadge";
import { useLibraryRobot } from "@/hooks/useLibrary";
import { usePhysics } from "@/hooks/usePhysics";
import { useProject } from "@/hooks/useProjects";
import { useRobot } from "@/hooks/useRobot";

// Lazy-load the Runs tab — it pulls in recharts + react-window (~90 KB
// gzipped combined) which would otherwise blow the main-bundle budget.
// The Vite `manualChunks` rule splits those into a `charts` chunk.
const RunsTabLazy = lazy(() => import("@/components/RunsTab"));
// Reports tab pulls in react-markdown + remark-gfm. Lazy-load so those
// stay out of the main bundle.
const ReportsTabLazy = lazy(() => import("@/components/ReportsTab"));
import { formatRelative } from "@/lib/utils";
import type { ProjectDetail as ProjectDetailShape, RobotStateResponse } from "@/lib/types";

function ProjectFactsBand({ project }: { project: ProjectDetailShape }) {
  const adapterShort = (project.adapter_class || "").split(".").pop() || "—";
  const cfg = project.adapter_config || {};
  const taskId = typeof cfg.task_id === "string" ? cfg.task_id : null;
  const numEnvs = typeof cfg.num_envs === "number" ? cfg.num_envs : null;
  const envId = typeof cfg.env_id === "string" ? cfg.env_id : null;
  const device = typeof cfg.device === "string" ? cfg.device : null;

  const adapterBadge =
    adapterShort === "MjlabAdapter"
      ? { tone: "emerald" as const, label: "mjlab" }
      : adapterShort === "GymSB3Adapter"
        ? { tone: "sky" as const, label: "gym_sb3" }
        : adapterShort === "IsaacLabAdapter"
          ? { tone: "amber" as const, label: "isaac (stub)" }
          : adapterShort === "MjxAdapter"
            ? { tone: "amber" as const, label: "mjx (stub)" }
            : adapterShort === "RllibAdapter"
              ? { tone: "amber" as const, label: "rllib (stub)" }
              : { tone: "slate" as const, label: adapterShort };

  return (
    <div className="flex flex-col gap-3">
      {project.adapter_unavailable && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
          <Hourglass className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div>
            <div className="font-medium text-amber-700 dark:text-amber-300">
              Adapter not yet implemented — training disabled
            </div>
            <p className="mt-0.5 text-xs text-muted-foreground">
              This project was created with a coming-soon adapter
              (`{adapterShort}`). Reward editing + preview still work; the
              Train button is gated until the adapter lands. See the
              adoption guide in <code>RewardSculptor/docs/adapters/</code>.
            </p>
          </div>
        </div>
      )}

      {project.migration_warning && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div className="flex-1">
            <div className="font-medium text-amber-700 dark:text-amber-300">
              Deprecated adapter
            </div>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {project.migration_warning}
            </p>
            <Button asChild variant="outline" size="sm" className="mt-2">
              <Link to="/library">
                <Library className="h-3.5 w-3.5" />
                Fork to a current adapter
              </Link>
            </Button>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-lg border bg-muted/30 px-4 py-3 text-xs">
        <Fact icon={<Box className="h-3.5 w-3.5" />} label="adapter">
          <Badge
            variant="outline"
            className={
              adapterBadge.tone === "emerald"
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                : adapterBadge.tone === "sky"
                  ? "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300"
                  : adapterBadge.tone === "amber"
                    ? "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                    : ""
            }
          >
            {adapterBadge.label}
          </Badge>
        </Fact>
        {taskId && (
          <Fact icon={<Zap className="h-3.5 w-3.5" />} label="task">
            <code className="font-mono text-[11px]">{taskId}</code>
          </Fact>
        )}
        {envId && (
          <Fact icon={<Zap className="h-3.5 w-3.5" />} label="env">
            <code className="font-mono text-[11px]">{envId}</code>
          </Fact>
        )}
        {numEnvs != null && (
          <Fact icon={<Cpu className="h-3.5 w-3.5" />} label="num_envs">
            <code className="font-mono text-[11px]">{numEnvs}</code>
          </Fact>
        )}
        {device && (
          <Fact icon={<Cpu className="h-3.5 w-3.5" />} label="device">
            <code className="font-mono text-[11px]">{device}</code>
          </Fact>
        )}
        {project.library_slug && (
          <Fact icon={<Library className="h-3.5 w-3.5" />} label="library">
            <Link
              to="/library"
              className="font-mono text-[11px] underline-offset-2 hover:underline"
            >
              {project.library_slug}
            </Link>
          </Fact>
        )}
        {project.ready_to_train === false && (
          <Badge
            variant="outline"
            className="ml-auto border-amber-500/40 bg-amber-500/10 text-[10px] text-amber-700 dark:text-amber-300"
          >
            Training disabled
          </Badge>
        )}
      </div>
    </div>
  );
}

function Fact({
  icon,
  label,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-muted-foreground">{icon}</span>
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      {children}
    </div>
  );
}

const TABS = [
  { value: "overview",  label: "Overview" },
  { value: "rewards",   label: "Rewards" },
  { value: "physics",   label: "Physics" },
  { value: "kg",        label: "Knowledge Graph" },
  { value: "runs",      label: "Runs" },
  { value: "missions",  label: "Missions" },
  { value: "reports",   label: "Reports" },
] as const;

export default function ProjectDetail() {
  const { slug } = useParams<{ slug: string }>();
  const project = useProject(slug);
  const robot = useRobot(slug);
  const [activeTab, setActiveTab] = useState<string>("overview");

  return (
    <div className="flex h-full flex-col">
      <header className="sticky top-0 z-10 flex h-12 items-center gap-3 border-b bg-background/80 px-6 backdrop-blur">
        <Button asChild variant="ghost" size="sm">
          <Link to="/projects" aria-label="Back to projects">
            <ArrowLeft />
            Projects
          </Link>
        </Button>
        <span className="text-muted-foreground">/</span>
        <h1 className="truncate text-sm font-semibold">
          {project.data?.display_name ?? slug}
        </h1>
        {project.data && <StatusBadge status={project.data.status} />}
        <div className="ml-auto flex items-center gap-2">
          {project.data && <ProjectSettingsDialog project={project.data} />}
          {project.data &&
            !project.data.adapter_unavailable &&
            project.data.ready_to_train !== false && (
              <NewRunDialog
                slug={slug!}
                project={project.data}
                onLaunched={() => setActiveTab("runs")}
              />
            )}
        </div>
      </header>

      <div className="flex-1 overflow-y-auto scrollbar-thin px-6 pb-16 pt-6">
        {project.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : project.error ? (
          <ErrorBlock error={project.error as Error} />
        ) : !project.data ? (
          <p className="text-sm text-muted-foreground">No project.</p>
        ) : (
          <div className="mx-auto flex max-w-6xl flex-col gap-4">
            <ProjectFactsBand project={project.data} />
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className=""
          >
            <TabsList>
              {TABS.map((t) => (
                <TabsTrigger key={t.value} value={t.value}>
                  {t.label}
                </TabsTrigger>
              ))}
            </TabsList>

            <TabsContent value="overview">
              <OverviewTab
                slug={slug!}
                project={project.data}
                robot={robot.data}
              />
            </TabsContent>

            <TabsContent value="rewards">
              <RewardsTab slug={slug!} project={project.data} />
            </TabsContent>

            <TabsContent value="physics">
              {activeTab === "physics" && (
                <PhysicsTab slug={slug!} project={project.data} />
              )}
            </TabsContent>

            <TabsContent value="kg">
              <KnowledgeGraphTab slug={slug!} />
            </TabsContent>

            <TabsContent value="runs">
              {activeTab === "runs" && (
                <Suspense fallback={<RunsTabFallback />}>
                  <RunsTabLazy slug={slug!} project={project.data} />
                </Suspense>
              )}
            </TabsContent>

            <TabsContent value="missions">
              {activeTab === "missions" && <MissionsTab slug={slug!} />}
            </TabsContent>

            <TabsContent value="reports">
              {activeTab === "reports" && (
                <Suspense fallback={<RunsTabFallback />}>
                  <ReportsTabLazy slug={slug!} />
                </Suspense>
              )}
            </TabsContent>
          </Tabs>
          </div>
        )}
      </div>
    </div>
  );
}

function RunsTabFallback() {
  return (
    <div className="flex h-48 items-center justify-center rounded-md border bg-muted/20 text-xs text-muted-foreground">
      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
      Loading runs tab (charts + virtualized log)…
    </div>
  );
}

// ── Overview ──────────────────────────────────────────────────────────
function OverviewTab({
  slug,
  project,
  robot,
}: {
  slug: string;
  project: ProjectDetailShape;
  robot: RobotStateResponse | undefined;
}) {
  const configured = isRobotConfigured(robot, project);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <div className="lg:col-span-2 flex flex-col gap-4">
        {configured ? (
          <RobotViewer slug={slug} />
        ) : (
          <RobotConfig slug={slug} />
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Description</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              {project.description || "No description."}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-col gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Metadata</CardTitle>
            <CardDescription>
              Read from <code>config.toml</code> + sidecar.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <KV k="slug" v={project.slug} mono />
            <KV k="adapter" v={project.adapter_class} mono />
            <KV
              k="env_id"
              v={
                project.env_id && project.env_id !== "CHANGE_ME"
                  ? project.env_id
                  : "—"
              }
              mono
            />
            <KV
              k="iterations"
              v={String(project.n_iterations_completed)}
            />
            <KV k="created" v={formatRelative(project.created_at)} />
            <KV k="directory" v={project.project_dir} mono truncate />
          </CardContent>
        </Card>

        {configured && robot && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Robot</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <KV k="kind" v={robot.kind} mono />
              {robot.library_name && (
                <KV k="library" v={robot.library_name} mono />
              )}
              {robot.env_id && <KV k="env_id" v={robot.env_id} mono />}
              {robot.model_file && (
                <KV k="model" v={robot.model_file} mono truncate />
              )}
              {robot.original_filename && (
                <KV k="uploaded" v={robot.original_filename} mono truncate />
              )}
              {robot.mesh_paths.length > 0 && (
                <KV k="meshes" v={`${robot.mesh_paths.length}`} />
              )}
            </CardContent>
          </Card>
        )}

        {configured && (robot?.library_name || project.library_slug) && (
          <RobotLibraryCard
            slug={slug}
            librarySlug={(robot?.library_name ?? project.library_slug)!}
          />
        )}
      </div>
    </div>
  );
}

function isRobotConfigured(
  robot: RobotStateResponse | undefined,
  project: ProjectDetailShape,
): boolean {
  if (!robot) return false;
  if (robot.kind === "none") return false;
  if (robot.kind === "library") {
    // A library-kind robot is configured when the library_name (slug)
    // is set. Previously we additionally required env_id !== CHANGE_ME
    // — that gate was gym-specific and rejected mjlab_builtin /
    // menagerie robots (Cartpole, Go1, G1, ANYmal) whose task is
    // steered via adapter_config.task_id instead of env_id. Bug #6.2.
    return !!(robot.library_name || project.library_slug);
  }
  // mjcf / urdf
  return !!robot.model_file;
}

// ── KV row ────────────────────────────────────────────────────────────
function KV({
  k,
  v,
  mono,
  truncate,
}: {
  k: string;
  v: string;
  mono?: boolean;
  truncate?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="shrink-0 text-[10px] uppercase tracking-wider text-muted-foreground">
        {k}
      </span>
      <span
        className={`min-w-0 flex-1 ${mono ? "font-mono text-[11px]" : ""} ${
          truncate ? "truncate" : "break-words"
        }`}
        title={truncate ? v : undefined}
      >
        {v}
      </span>
    </div>
  );
}

// ── Library-robot dashboard card (S5 / bug #6.2) ──────────────────────
// Surfaces the selected library robot's metadata directly in the
// Overview sidebar — category, source, references, thumbnail, MJCF
// summary. Pre-S5 Sam's Cartpole project rendered the generic library
// GRID here (copy of /projects/new) because isRobotConfigured required
// env_id !== CHANGE_ME, which is only set for gymnasium_builtin robots.
function RobotLibraryCard({
  slug,
  librarySlug,
}: {
  slug: string;
  librarySlug: string;
}) {
  const lib = useLibraryRobot(librarySlug);
  const phys = usePhysics(slug);
  const entry = lib.data;
  if (lib.isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Library entry</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          <Loader2 className="inline h-3 w-3 animate-spin" /> Loading…
        </CardContent>
      </Card>
    );
  }
  if (!entry) {
    return null; // API miss; sibling cards still render.
  }
  const summary = phys.data?.summary;
  const njnt = summary?.joints?.length ?? null;
  const nu = summary?.actuators?.length ?? null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          {entry.display_name}
          <Badge variant="outline" className="text-[10px]">
            {entry.category}
          </Badge>
        </CardTitle>
        <CardDescription className="text-xs">
          {entry.description || "Library-sourced robot."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <KV k="slug" v={entry.slug} mono />
        <KV k="source" v={entry.source} mono />
        {entry.menagerie_package && (
          <KV k="menagerie" v={entry.menagerie_package} mono truncate />
        )}
        <KV k="training" v={entry.training_support} mono />
        {njnt !== null && <KV k="joints" v={String(njnt)} mono />}
        {nu !== null && <KV k="actuators" v={String(nu)} mono />}
        {summary?.parse_error && (
          <p className="rounded border border-amber-500/40 bg-amber-500/5 px-1.5 py-1 text-[10px] text-amber-700 dark:text-amber-300">
            MJCF parse error — check the Physics tab: {summary.parse_error}
          </p>
        )}
        {entry.preconfigured_tasks.length > 0 && (
          <div>
            <div className="mb-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
              Tasks
            </div>
            <ul className="space-y-0.5">
              {entry.preconfigured_tasks.map((t) => (
                <li
                  key={t.task_id}
                  className="flex items-baseline gap-2 font-mono text-[10.5px]"
                >
                  <span className="truncate" title={t.task_id}>
                    {t.task_id}
                  </span>
                  <span className="shrink-0 text-muted-foreground">
                    {t.recommended_num_envs} envs
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {entry.references.length > 0 && (
          <div>
            <div className="mb-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
              References
            </div>
            <ul className="space-y-0.5">
              {entry.references.slice(0, 6).map((r, i) => (
                <li key={i} className="flex items-baseline gap-1.5 text-[10.5px]">
                  <a
                    href={r.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1 truncate text-foreground underline-offset-2 hover:underline"
                    title={r.citation}
                  >
                    <span className="truncate">{r.citation}</span>
                    <ExternalLink className="h-3 w-3 shrink-0" />
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="pt-1">
          <Button
            asChild
            size="sm"
            variant="outline"
            className="w-full"
          >
            <a href="#physics" onClick={(e) => {
              e.preventDefault();
              // Flip tabs via URL hash so deep-links keep working
              // alongside the trigger click. The Tabs component reads
              // its `value` from the URL hash in ProjectDetail.
              const tab = document.querySelector<HTMLButtonElement>(
                '[role="tab"][value="physics"]'
              );
              tab?.click();
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


function ErrorBlock({ error }: { error: Error }) {
  return (
    <div className="mx-auto max-w-md rounded-lg border border-destructive/40 bg-destructive/5 p-6">
      <h2 className="text-sm font-semibold text-destructive">
        Could not load project
      </h2>
      <p className="mt-1 font-mono text-xs text-muted-foreground">
        {error.message}
      </p>
    </div>
  );
}

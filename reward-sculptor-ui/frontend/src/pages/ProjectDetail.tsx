import { lazy, Suspense, useCallback } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { Icon } from "@/components/rs/icon";
import { Badge, Btn, FactChip } from "@/components/rs/primitives";
import { BehaviorFlow } from "@/components/BehaviorFlow";
import { KnowledgeGraphTab } from "@/components/KnowledgeGraphTab";
import { PhysicsTab } from "@/components/PhysicsTab";
import { ProjectSettingsDialog } from "@/components/ProjectSettingsDialog";
import { NewRunDialog } from "@/components/NewRunDialog";
import { RewardsTab } from "@/components/RewardsTab";
import { RobotConfig } from "@/components/RobotConfig";
import { RobotViewer } from "@/components/RobotViewer";
import WorldTab, { courseBreakdownText } from "@/components/WorldTab";
import { useLibraryRobot } from "@/hooks/useLibrary";
import { useWorldSelection } from "@/hooks/useWorlds";
import { usePhysics } from "@/hooks/usePhysics";
import { useProject } from "@/hooks/useProjects";
import { useRobot } from "@/hooks/useRobot";
import { useHasActiveRun } from "@/hooks/useRuns";
import { formatRelative } from "@/lib/utils";
import type {
  ProjectDetail as ProjectDetailShape,
  ProjectStatus,
  RobotStateResponse,
  SelectedStage,
} from "@/lib/types";

const RunsTabLazy = lazy(() => import("@/components/RunsTab"));
const ReportsTabLazy = lazy(() => import("@/components/ReportsTab"));

// Ordered as the workflow reads: set up → design the reward → train →
// take the results. Values appear in the URL (?tab=) so keep them stable.
const TABS = [
  { value: "overview", label: "Overview", icon: "gauge" },
  { value: "world", label: "World", icon: "globe" },
  { value: "rewards", label: "Rewards", icon: "file-code" },
  { value: "physics", label: "Physics", icon: "cpu" },
  { value: "knowledge", label: "Knowledge", icon: "network" },
  { value: "training", label: "Training", icon: "activity" },
  { value: "results", label: "Results", icon: "file-text" },
] as const;

type TabValue = (typeof TABS)[number]["value"];

// Old bookmark / in-app values from before the IA rename.
const LEGACY_TABS: Record<string, TabValue> = {
  kg: "knowledge",
  runs: "training",
  reports: "results",
};

function normalizeTab(raw: string | null): TabValue {
  if (raw && raw in LEGACY_TABS) return LEGACY_TABS[raw];
  if (raw && TABS.some((t) => t.value === raw)) return raw as TabValue;
  return "overview";
}

// A stage selection shared across tabs (currently consumed by Training;
// Overview/Rewards receive the props so their wiring lands in a later
// increment). URL-synced via `?stage=missionSlug/stageName` so the
// selection survives refresh, mirroring the `?tab=` pattern above.
// (SelectedStage itself is defined in lib/types.ts to avoid a circular
// import back into this page module from RobotViewer/RunsTab.)
function parseStageParam(raw: string | null): SelectedStage | null {
  if (!raw) return null;
  const idx = raw.indexOf("/");
  if (idx <= 0 || idx === raw.length - 1) return null;
  return { missionSlug: raw.slice(0, idx), stageName: raw.slice(idx + 1) };
}

function stageParam(s: SelectedStage): string {
  return `${s.missionSlug}/${s.stageName}`;
}

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

// Generic padded scroll container for a tab's page content.
function ScrollPad({ children }: { children: React.ReactNode }) {
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
            adapter (<code className="mono">{adapterShortName}</code>). Reward editing + preview still work.
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
  // Tab lives in the URL (?tab=training) so refresh keeps your place and
  // any view is deep-linkable.
  const [searchParams, setSearchParams] = useSearchParams();
  const tab = normalizeTab(searchParams.get("tab"));
  const setTab = (value: TabValue) => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (value === "overview") next.delete("tab");
        else next.set("tab", value);
        return next;
      },
      // replace, not push: Back should leave the page, not unwind every
      // tab click. Deep-linking only needs the URL set.
      { replace: true },
    );
  };

  // Shared stage selection (Training tab consumes this increment;
  // Overview/Rewards get the plumbing now, wired up later).
  const selectedStage = parseStageParam(searchParams.get("stage"));
  const setSelectedStage = useCallback((value: SelectedStage | null) => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (!value) next.delete("stage");
        else next.set("stage", stageParam(value));
        return next;
      },
      { replace: true },
    );
  }, [setSearchParams]);
  const p = project.data;
  const canRun = !!p && !p.adapter_unavailable && p.ready_to_train !== false;
  // §Ship 37: the backend's derived `project.status` never reports "running"
  // (see `hasActiveRun`), so this page showed "Draft" through an entire run.
  // Overlay the live signal onto the badge; the other derived states —
  // configured / ready / completed / errored — still come from the project.
  const liveRun = useHasActiveRun(slug);
  const shownStatus = (s: ProjectStatus): ProjectStatus => (liveRun ? "running" : s);

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
          {p && <Badge status={shownStatus(p.status)} big />}
        </div>
        <div className="rs-phead-spacer" />
        {p && <ProjectSettingsDialog project={p} />}
        {p && canRun && (
          <NewRunDialog
            slug={slug!}
            project={p}
            onLaunched={() => setTab("training")}
            onOpenWorld={() => setTab("world")}
          />
        )}
      </div>

      {project.isLoading ? (
        <ScrollPad><p className="rs-sub">Loading…</p></ScrollPad>
      ) : project.error ? (
        <ScrollPad>
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">Could not load project: {(project.error as Error).message}</span>
          </div>
        </ScrollPad>
      ) : !p ? (
        <ScrollPad><p className="rs-sub">No project.</p></ScrollPad>
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

          {tab !== "training" && <FactsBand project={p} />}
          {tab !== "training" && (p.adapter_unavailable || p.migration_warning) && <WarningBanners project={p} />}

          {tab === "overview" && (
            <OverviewTab
              slug={slug!}
              project={p}
              robot={robot.data}
              onGoTo={setTab}
              selectedStage={selectedStage}
              setSelectedStage={setSelectedStage}
            />
          )}
          {tab === "rewards" && (
            <ScrollPad>
              <RewardsTab
                slug={slug!}
                project={p}
                selectedStage={selectedStage}
                setSelectedStage={setSelectedStage}
              />
            </ScrollPad>
          )}
          {tab === "world" && (
            <ScrollPad>
              <WorldTab
                slug={slug!}
                launchAction={canRun ? (
                  <NewRunDialog
                    slug={slug!}
                    project={p}
                    onLaunched={() => setTab("training")}
                    triggerLabel="Train this world"
                  />
                ) : undefined}
              />
            </ScrollPad>
          )}
          {tab === "physics" && <ScrollPad><PhysicsTab slug={slug!} project={p} /></ScrollPad>}
          {tab === "knowledge" && <KnowledgeGraphTab slug={slug!} />}
          {tab === "training" && (
            <Suspense fallback={<TabFallback />}>
              <RunsTabLazy
                slug={slug!}
                project={p}
                onOpenWorld={() => setTab("world")}
                selectedStage={selectedStage}
                setSelectedStage={setSelectedStage}
              />
            </Suspense>
          )}
          {tab === "results" && (
            <Suspense fallback={<TabFallback />}>
              <ReportsTabLazy
                slug={slug!}
                selectedStage={selectedStage}
                setSelectedStage={setSelectedStage}
              />
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
        <Icon name="loader" size={20} className="rs-spin" />
        <span className="rs-sub">Loading…</span>
      </div>
    </div>
  );
}

// ── Overview ──────────────────────────────────────────────────────────
function OverviewTab({
  slug, project, robot, onGoTo, selectedStage, setSelectedStage,
}: {
  slug: string;
  project: ProjectDetailShape;
  robot: RobotStateResponse | undefined;
  onGoTo: (tab: TabValue) => void;
  selectedStage: SelectedStage | null;
  setSelectedStage: (value: SelectedStage | null) => void;
}) {
  const configured = isRobotConfigured(robot, project);
  // §Ship 37: same overlay as the page header — see `hasActiveRun`.
  const liveRun = useHasActiveRun(slug);
  const cfg = project.adapter_config || {};
  const taskId = typeof cfg.task_id === "string" ? cfg.task_id : null;
  const numEnvs = typeof cfg.num_envs === "number" ? cfg.num_envs : null;
  const device = typeof cfg.device === "string" ? cfg.device : null;
  return (
    <div className="rs-scroll">
      <div className="rs-pad" style={{ display: "grid", gridTemplateColumns: "minmax(0,1.6fr) minmax(300px,1fr)", gap: 22, alignItems: "start" }}>
        {/* Anchored so the Behavior flow's first step can bring it into
            view. Both cards live in the other column of this same screen, so
            that step's button used to switch to the tab it was already on. */}
        <div className="rs-vgap-16" id="robot-config">
          {configured ? (
            <RobotViewer slug={slug} selectedStage={selectedStage} setSelectedStage={setSelectedStage} />
          ) : (
            <RobotConfig slug={slug} />
          )}
          <div className="rs-card rs-card-pad">
            <div className="rs-card-title" style={{ marginBottom: 10 }}><Icon name="flag" size={16} />What this project is</div>
            <p className="rs-sub" style={{ lineHeight: 1.6, margin: 0 }}>{project.description || "No description."}</p>
          </div>
        </div>

        <div className="rs-vgap-16">
          <BehaviorFlow
            slug={slug}
            project={project}
            robotConfigured={configured}
            onGoTo={onGoTo}
          />
          <div className="rs-card">
            <div className="rs-card-head"><div className="rs-card-title"><Icon name="info" size={16} />Project facts</div></div>
            <div className="rs-kv">
              <div className="k">status</div><div className="v"><Badge status={liveRun ? "running" : project.status} /></div>
              <div className="k">adapter</div><div className="v">{adapterShort(project.adapter_class)}</div>
              {humanizeSlug(project.library_slug) && (<><div className="k">robot</div><div className="v">{humanizeSlug(project.library_slug)}</div></>)}
              {(taskId || project.env_id) && (<><div className="k">task_id</div><div className="v">{taskId ?? project.env_id}</div></>)}
              {device && (<><div className="k">device</div><div className="v">{device}</div></>)}
              {numEnvs != null && (<><div className="k">num_envs</div><div className="v">{numEnvs.toLocaleString()}</div></>)}
              <div className="k">iterations</div><div className="v">{project.n_iterations_completed}</div>
              <div className="k">created</div><div className="v">{formatRelative(project.created_at)}</div>
              <div className="k">directory</div><div className="v" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={project.project_dir}>{project.project_dir}</div>
            </div>
          </div>

          <AuthoredWorldCard slug={slug} onGoTo={onGoTo} />

          {configured && (robot?.library_name || project.library_slug) && (
            <RobotLibraryCard slug={slug} librarySlug={(robot?.library_name ?? project.library_slug)!} />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Authored world (env-authoring): the world training runs under ─────
function AuthoredWorldCard({ slug, onGoTo }: { slug: string; onGoTo: (tab: TabValue) => void }) {
  const selection = useWorldSelection(slug);
  const s = selection.data;
  if (!s) return null; // no authored world yet — the World tab is the entry point
  const mismatch = s.shared_summary.robot_matches_project === false;
  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title"><Icon name="globe" size={16} />Authored world</div>
        <Btn kind="ghost" size="sm" icon="arrow-right" onClick={() => onGoTo("world")}>World tab</Btn>
      </div>
      <div className="rs-kv">
        <div className="k">selection</div>
        <div className="v mono">v{s.selection.selection_version} · {s.selection.tuple_hash.slice(0, 12)}</div>
        <div className="k">robot</div>
        <div className="v">
          {s.shared_summary.robot ?? "—"}
          {mismatch && (
            <span style={{ color: "var(--st-amber-fg)", marginLeft: 6 }}>
              ≠ project robot
            </span>
          )}
        </div>
        <div className="k">terrain</div>
        <div className="v">{s.shared_summary.terrain_kind ?? "plane"}</div>
        <div className="k">course</div>
        <div className="v">{courseBreakdownText(s.shared_summary.course_breakdown, s.shared_summary.course_elements)}</div>
        {s.world_meta.prompt && (
          <><div className="k">prompt</div><div className="v" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={s.world_meta.prompt}>{s.world_meta.prompt}</div></>
        )}
      </div>
      <div className="rs-card-pad" style={{ paddingTop: 8 }}>
        <p className="rs-hintline" style={{ margin: 0 }}>
          Sculpt runs on this project train and evaluate inside this world
          (atomic selection <span className="mono">{s.selection.evaluation_lineage}</span>).
        </p>
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

function RobotLibraryCard({ slug, librarySlug }: { slug: string; librarySlug: string }) {
  const lib = useLibraryRobot(librarySlug);
  const phys = usePhysics(slug);
  const entry = lib.data;
  if (lib.isLoading) {
    return (
      <div className="rs-card rs-card-pad">
        <div className="rs-flex rs-gap-8 rs-sub"><Icon name="loader" size={14} className="rs-spin" />Loading library entry…</div>
      </div>
    );
  }
  if (!entry) return null;
  const summary = phys.data?.summary;
  const njnt = summary?.joints?.length ?? null;
  const nu = summary?.actuators?.length ?? null;
  return (
    <div className="rs-card">
      <div className="rs-card-head">
        <div className="rs-card-title">{entry.display_name}<span className="rs-tag" style={{ marginLeft: 6 }}>{entry.category}</span></div>
      </div>
      <div className="rs-card-pad rs-vgap-8">
        <p className="rs-sub" style={{ margin: 0, fontSize: 13 }}>{entry.description || "Library-sourced robot."}</p>
        <div className="rs-kv">
          <div className="k">slug</div><div className="v">{entry.slug}</div>
          <div className="k">source</div><div className="v">{entry.source}</div>
          {entry.menagerie_package && (<><div className="k">menagerie</div><div className="v" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={entry.menagerie_package}>{entry.menagerie_package}</div></>)}
          <div className="k">training</div><div className="v">{entry.training_support}</div>
          {njnt !== null && (<><div className="k">joints</div><div className="v">{njnt}</div></>)}
          {nu !== null && (<><div className="k">actuators</div><div className="v">{nu}</div></>)}
        </div>
        {summary?.parse_error && (
          <div className="rs-banner warn"><Icon name="alert-triangle" size={15} /><span className="rs-grow">MJCF parse error — see Physics tab: {summary.parse_error}</span></div>
        )}
        {entry.references.length > 0 && (
          <div>
            <div className="rs-eyebrow" style={{ marginBottom: 6 }}>References</div>
            <div className="rs-vgap-8">
              {entry.references.slice(0, 6).map((r, i) => (
                <a key={i} href={r.url} target="_blank" rel="noreferrer noopener"
                  className="rs-flex rs-gap-6" style={{ fontSize: 12.5, color: "var(--ink)", overflow: "hidden" }} title={r.citation}>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.citation}</span>
                  <Icon name="external" size={12} />
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

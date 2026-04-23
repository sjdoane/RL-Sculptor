/** Typed fetch wrappers for every endpoint we consume in v1.
 *  All calls go through `/api/*`, which Vite's dev server proxies to
 *  the backend on localhost:8000. See vite.config.ts. */

import type {
  AdapterInfo,
  AddSeedsPayload,
  CameraAngle,
  CreateProjectPayload,
  HealthResponse,
  JobDetail,
  JobSummary,
  LibraryListResponse,
  LibraryRobot,
  LibraryRobotName,
  ManualEditPayload,
  PaperDetail,
  PaperSummary,
  PendingSeedsResponse,
  ProblemDetail,
  ProjectDetail,
  ProjectSummary,
  RewardVersionDetail,
  RewardVersionSummary,
  PhysicsLoadResponse,
  RewardDiagnosisPayload,
  RobotStateResponse,
  SystemGpuResponse,
  SystemKgStatsResponse,
  TechniqueSummary,
} from "./types";

export class ApiError extends Error {
  status: number;
  type: string;
  problem: ProblemDetail;
  constructor(problem: ProblemDetail) {
    super(problem.detail || problem.title);
    this.status = problem.status;
    this.type = problem.type;
    this.problem = problem;
    this.name = "ApiError";
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.ok) {
    if (res.status === 204) return undefined as unknown as T;
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) return (await res.json()) as T;
    return (await res.text()) as unknown as T;
  }
  let problem: ProblemDetail;
  try {
    problem = (await res.json()) as ProblemDetail;
  } catch {
    problem = {
      type: "about:blank",
      title: res.statusText || "Request failed",
      status: res.status,
      detail: null,
    };
  }
  throw new ApiError(problem);
}

// ── Meta ──────────────────────────────────────────────────────────────
export async function getHealth(): Promise<HealthResponse> {
  return handle<HealthResponse>(await fetch("/api/health"));
}

// ── Projects ──────────────────────────────────────────────────────────
export async function listProjects(): Promise<ProjectSummary[]> {
  return handle<ProjectSummary[]>(await fetch("/api/projects"));
}

export async function getProject(slug: string): Promise<ProjectDetail> {
  return handle<ProjectDetail>(await fetch(`/api/projects/${slug}`));
}

export async function createProject(
  payload: CreateProjectPayload,
): Promise<ProjectDetail> {
  return handle<ProjectDetail>(
    await fetch("/api/projects", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function deleteProject(slug: string): Promise<void> {
  await handle<void>(
    await fetch(`/api/projects/${slug}`, { method: "DELETE" }),
  );
}

// ── Robot ─────────────────────────────────────────────────────────────
export async function getRobot(slug: string): Promise<RobotStateResponse> {
  return handle<RobotStateResponse>(
    await fetch(`/api/projects/${slug}/robot`),
  );
}

export async function pickLibraryRobot(
  slug: string,
  robotName: LibraryRobotName,
): Promise<RobotStateResponse> {
  return handle<RobotStateResponse>(
    await fetch(`/api/projects/${slug}/robot/library`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ robot_name: robotName }),
    }),
  );
}

// ── Robot upload ─────────────────────────────────────────────────────
export async function uploadRobotModel(
  slug: string,
  modelFile: File,
  meshesZip?: File,
): Promise<RobotStateResponse> {
  const fd = new FormData();
  fd.append("model_file", modelFile);
  if (meshesZip) fd.append("meshes_zip", meshesZip);
  return handle<RobotStateResponse>(
    await fetch(`/api/projects/${slug}/robot/urdf`, {
      method: "POST",
      body: fd,
    }),
  );
}

// ── Library (M3) ─────────────────────────────────────────────────────
export async function getLibraryCategories(): Promise<string[]> {
  return handle<string[]>(await fetch("/api/library/categories"));
}

export async function listLibraryRobots(params?: {
  category?: string;
  training_support?: string;
  search?: string;
}): Promise<LibraryListResponse> {
  const qs = new URLSearchParams();
  if (params?.category) qs.set("category", params.category);
  if (params?.training_support) qs.set("training_support", params.training_support);
  if (params?.search) qs.set("search", params.search);
  const url = qs.toString()
    ? `/api/library/robots?${qs.toString()}`
    : "/api/library/robots";
  return handle<LibraryListResponse>(await fetch(url));
}

export async function getLibraryRobot(slug: string): Promise<LibraryRobot> {
  return handle<LibraryRobot>(await fetch(`/api/library/robots/${slug}`));
}

export async function ingestGlobalKgSeeds(
  projectSlug: string,
  autoExtract: boolean = true,
): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(
      `/api/projects/${projectSlug}/kg/ingest-global?auto_extract=${autoExtract}`,
      { method: "POST" },
    ),
  );
}

/** §Ship-7: POST /projects/{slug}/kg/heal-stubs — re-ingests every Paper
 * node whose title is still `arxiv:XXXX.XXXXX`. Synchronous; ~2 min for
 * ~10 stubs. */
export interface HealStubsResponse {
  results: Record<string, string>;
  summary: {
    healed: number;
    still_stubbed: number;
    errored: number;
    total: number;
  };
}

export async function healStubTitles(
  projectSlug: string,
): Promise<HealStubsResponse> {
  return handle<HealStubsResponse>(
    await fetch(`/api/projects/${projectSlug}/kg/heal-stubs`, {
      method: "POST",
    }),
  );
}

/** §Ship-8: per-project settings (config.toml [iteration] block).
 *  §Ship-9a: adds early_stop_enabled + early_stop_patience. */
export interface IterationSettings {
  steps_per_iter?: number | null;
  primary_metric?: string | null;
  behavior_metrics?: string[] | null;
  rollout_episodes?: number | null;
  auto_adjust_physics?: boolean | null;
  max_episode_steps?: number | null;
  playback_speed?: number | null;
  render_every?: number | null;
  rollout_fps?: number | null;
  seed?: number | null;
  early_stop_enabled?: boolean | null;
  early_stop_patience?: number | null;
}

export interface ProjectSettings {
  iteration: IterationSettings;
}

export async function getProjectSettings(
  projectSlug: string,
): Promise<ProjectSettings> {
  return handle<ProjectSettings>(
    await fetch(`/api/projects/${projectSlug}/settings`),
  );
}

export async function patchProjectSettings(
  projectSlug: string,
  body: ProjectSettings,
): Promise<ProjectSettings> {
  return handle<ProjectSettings>(
    await fetch(`/api/projects/${projectSlug}/settings`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** §Ship-9b: structured motor-spec payload for POST /physics/motor-limits. */
export interface MotorSpec {
  peak_torque_nm?: number | null;
  peak_speed_rads?: number | null;
  gear_ratio?: number | null;
  rotor_inertia_kgm2?: number | null;
  peak_current_a?: number | null;
  notes?: string | null;
}

export interface MotorLimitsRequest {
  motors: Record<string, MotorSpec>;
}

export async function applyMotorLimits(
  projectSlug: string,
  body: MotorLimitsRequest,
): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(`/api/projects/${projectSlug}/physics/motor-limits`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** §Ship-9c: upload a motor datasheet PDF and get structured specs back.
 * Response shape matches `MotorLimitsRequest` so the UI can pre-fill
 * the motor-limits form; user reviews/edits, then `applyMotorLimits`. */
export async function extractDatasheetPdf(
  projectSlug: string,
  pdf: File,
): Promise<MotorLimitsRequest> {
  const formData = new FormData();
  formData.append("pdf", pdf);
  return handle<MotorLimitsRequest>(
    await fetch(
      `/api/projects/${projectSlug}/physics/datasheet-pdf`,
      { method: "POST", body: formData },
    ),
  );
}

/** POST /projects/{slug}/kg/research — prompt-time KG research (M7 P2).
 * Claude picks arxiv IDs relevant to `topic`, dedupes, ingests + extracts. */
export async function researchKgTopic(
  projectSlug: string,
  body: { topic: string; max_papers?: number; auto_extract?: boolean },
): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(`/api/projects/${projectSlug}/kg/research`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        topic: body.topic,
        max_papers: body.max_papers ?? 10,
        auto_extract: body.auto_extract ?? true,
      }),
    }),
  );
}

/** POST /jobs/{jobId}/stop — cooperative cancel. Used by the
 * ActiveJobsIndicator's stop button to kill an in-flight KG / run /
 * prompt job without curl. */
export async function stopJob(jobId: string): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(`/api/jobs/${encodeURIComponent(jobId)}/stop`, {
      method: "POST",
    }),
  );
}

export async function listLibraryAdapters(): Promise<AdapterInfo[]> {
  return handle<AdapterInfo[]>(await fetch("/api/library/adapters"));
}

export function libraryThumbnailUrl(slug: string): string {
  return `/api/library/robots/${slug}/thumbnail`;
}

// ── System (M2 + M4) ─────────────────────────────────────────────────
export async function getSystemGpu(): Promise<SystemGpuResponse> {
  return handle<SystemGpuResponse>(await fetch("/api/system/gpu"));
}

/** GET /system/kg/stats — aggregate counts over the shared KG (M7 P1). */
export async function getSystemKgStats(): Promise<SystemKgStatsResponse> {
  return handle<SystemKgStatsResponse>(await fetch("/api/system/kg/stats"));
}

// ── Physics (M7 Phase 5) ────────────────────────────────────────────
export async function getPhysics(
  slug: string,
): Promise<PhysicsLoadResponse> {
  return handle<PhysicsLoadResponse>(
    await fetch(`/api/projects/${slug}/physics`),
  );
}

export async function promptPhysicsEdit(
  slug: string,
  prompt: string,
): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(`/api/projects/${slug}/physics/prompt`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt }),
    }),
  );
}

export async function rematerializePhysics(
  slug: string,
): Promise<PhysicsLoadResponse> {
  return handle<PhysicsLoadResponse>(
    await fetch(`/api/projects/${slug}/physics/rematerialize`, {
      method: "POST",
    }),
  );
}

// ── Preview ───────────────────────────────────────────────────────────
export interface PreviewQueryParams {
  angle?: CameraAngle;
  regenerate?: boolean;
}

/** Build the preview URL. Passing `regenerate=true` forces the server
 *  to re-render. The returned URL is stable for identical inputs so
 *  React Query / the browser can cache it. */
export function previewUrl(slug: string, opts?: PreviewQueryParams): string {
  const u = new URL(`/api/projects/${slug}/preview`, window.location.origin);
  if (opts?.angle) u.searchParams.set("angle", opts.angle);
  if (opts?.regenerate) u.searchParams.set("regenerate", "true");
  return u.pathname + (u.search ? u.search : "");
}

/** Fetch the preview as a blob + produce an object URL. Throws ApiError
 *  on 409 (robot not configured) / 500 (render failed) so React Query
 *  can surface the exact problem to the user. */
export async function fetchPreviewBlob(
  slug: string,
  opts?: PreviewQueryParams,
): Promise<string> {
  const res = await fetch(previewUrl(slug, opts));
  if (!res.ok) {
    let problem: ProblemDetail;
    try {
      problem = (await res.json()) as ProblemDetail;
    } catch {
      problem = {
        type: "about:blank",
        title: res.statusText,
        status: res.status,
        detail: null,
      };
    }
    throw new ApiError(problem);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ── Rewards ───────────────────────────────────────────────────────────
export async function listRewards(slug: string): Promise<RewardVersionSummary[]> {
  return handle<RewardVersionSummary[]>(
    await fetch(`/api/projects/${slug}/rewards`),
  );
}

export async function getReward(
  slug: string,
  version: number,
): Promise<RewardVersionDetail> {
  return handle<RewardVersionDetail>(
    await fetch(`/api/projects/${slug}/rewards/${version}`),
  );
}

export async function putReward(
  slug: string,
  version: number,
  body: ManualEditPayload,
): Promise<RewardVersionDetail> {
  return handle<RewardVersionDetail>(
    await fetch(`/api/projects/${slug}/rewards/${version}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** GET /projects/{slug}/rewards/{version}/diagnosis — diagnosis.json
 * for the iter that produced this version. Only populated for
 * sculptor-authored versions (v1+ from a real sculpt run). Raw
 * payload returned as `unknown`; the UI types it locally. */
export async function getRewardDiagnosis(
  slug: string,
  version: number,
): Promise<RewardDiagnosisPayload> {
  return handle<RewardDiagnosisPayload>(
    await fetch(`/api/projects/${slug}/rewards/${version}/diagnosis`),
  );
}

/** Rewrite rewards/v0.py using the scaffold template for the project's
 * current adapter. Destructive — overwrites any edits to v0.py. Used as
 * the remediation action when a run fails with `reward_contract_mismatch`
 * (mjlab project whose v0 only exports scalar `compute_reward`). */
export async function regenerateRewardTemplate(
  slug: string,
): Promise<RewardVersionDetail> {
  return handle<RewardVersionDetail>(
    await fetch(`/api/projects/${slug}/rewards/regenerate-template`, {
      method: "POST",
    }),
  );
}

/** POST /projects/{slug}/rewards/prompt — Claude rewrites v<n+1>.py
 * from a natural-language prompt. Returns a JobSummary (202); poll via
 * GET /jobs/{id} for completion. */
export async function promptReward(
  slug: string,
  body: { prompt: string; expected_parent_version: number },
): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(`/api/projects/${slug}/rewards/prompt`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

// ── Knowledge graph ───────────────────────────────────────────────────
export async function listPapers(
  slug: string,
  opts?: { extracted?: boolean; search?: string },
): Promise<PaperSummary[]> {
  const u = new URL(`/api/projects/${slug}/kg/papers`, window.location.origin);
  if (opts?.extracted !== undefined) u.searchParams.set("extracted", String(opts.extracted));
  if (opts?.search) u.searchParams.set("search", opts.search);
  return handle<PaperSummary[]>(await fetch(u.pathname + (u.search || "")));
}

export async function getPaper(
  slug: string,
  arxivId: string,
): Promise<PaperDetail> {
  return handle<PaperDetail>(
    await fetch(`/api/projects/${slug}/kg/papers/${arxivId}`),
  );
}

export async function listTechniques(slug: string): Promise<TechniqueSummary[]> {
  return handle<TechniqueSummary[]>(
    await fetch(`/api/projects/${slug}/kg/techniques`),
  );
}

export async function getPendingSeeds(
  slug: string,
): Promise<PendingSeedsResponse> {
  return handle<PendingSeedsResponse>(
    await fetch(`/api/projects/${slug}/kg/pending-seeds`),
  );
}

export async function addSeeds(
  slug: string,
  payload: AddSeedsPayload,
): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(`/api/projects/${slug}/kg/seeds`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

/** URL for the pre-rendered pyvis graph. Hit with `regenerate=true`
 *  once on each "View graph" modal open so the UI always sees fresh
 *  provenance highlighting. */
export function kgGraphHtmlUrl(
  slug: string,
  opts?: { regenerate?: boolean },
): string {
  return `/api/projects/${slug}/kg/graph.html${
    opts?.regenerate ? "?regenerate=true" : ""
  }`;
}

// ── Jobs ──────────────────────────────────────────────────────────────
export async function getJob(jobId: string): Promise<JobDetail> {
  return handle<JobDetail>(await fetch(`/api/jobs/${jobId}`));
}

export async function listJobs(opts?: {
  projectSlug?: string;
}): Promise<JobSummary[]> {
  const u = new URL("/api/jobs", window.location.origin);
  if (opts?.projectSlug) u.searchParams.set("project_slug", opts.projectSlug);
  return handle<JobSummary[]>(await fetch(u.pathname + (u.search || "")));
}

// ── Runs ──────────────────────────────────────────────────────────────
import type {
  RunDetail,
  RunParamsPayload,
  RunSummary,
} from "./types";

export async function listRuns(slug: string): Promise<RunSummary[]> {
  return handle<RunSummary[]>(
    await fetch(`/api/projects/${slug}/runs`),
  );
}

export async function getRun(
  slug: string, runId: string,
): Promise<RunDetail> {
  return handle<RunDetail>(
    await fetch(`/api/projects/${slug}/runs/${runId}`),
  );
}

export async function launchRun(
  slug: string, payload: RunParamsPayload,
): Promise<RunSummary> {
  return handle<RunSummary>(
    await fetch(`/api/projects/${slug}/runs`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function killRun(
  slug: string, runId: string,
): Promise<RunSummary> {
  return handle<RunSummary>(
    await fetch(`/api/projects/${slug}/runs/${runId}`, {
      method: "DELETE",
    }),
  );
}

/** WebSocket URL for a run's event stream. Vite's dev proxy routes
 *  `/ws/*` to the backend (see vite.config.ts). In prod the browser
 *  hits the same origin so no rewrite is needed. */
export function runEventsWsUrl(slug: string, runId: string): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/projects/${slug}/runs/${runId}/events`;
}

/** WebSocket URL for a single job's event stream (reward/physics
 *  prompt edit, KG extract, etc.). Replays buffered log_line events
 *  on connect + tails live emits; closes with `terminal` when the job
 *  reaches a terminal state. */
export function jobEventsWsUrl(jobId: string): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/jobs/${jobId}/events`;
}

/** Filtered WS that only carries new_clip / clip_skipped events for
 *  the live-rollout viewer. */
export function runFramesWsUrl(slug: string, runId: string): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/projects/${slug}/runs/${runId}/frames`;
}

export function iterRolloutUrl(
  slug: string, runId: string, iterIndex: number,
): string {
  return `/api/projects/${slug}/runs/${runId}/iterations/${iterIndex}/rollout`;
}

export function clipUrl(
  slug: string, runId: string, iterIndex: number,
): string {
  return `/api/projects/${slug}/runs/${runId}/clips/iter_${iterIndex}.mp4`;
}

// ── Dashboard + System ───────────────────────────────────────────────
import type { DashboardSummary, SystemInfo } from "./types";

export async function getDashboard(): Promise<DashboardSummary> {
  return handle<DashboardSummary>(await fetch("/api/dashboard"));
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return handle<SystemInfo>(await fetch("/api/system/info"));
}

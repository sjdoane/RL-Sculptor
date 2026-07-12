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
  RemoteDoctorResponse,
  RemoteSettings,
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
    // FastAPI 422s put an ARRAY of validation errors in `detail` —
    // stringify it so toasts don't render "[object Object]".
    super(
      typeof problem.detail === "string" && problem.detail
        ? problem.detail
        : problem.detail != null && typeof problem.detail === "object"
          ? JSON.stringify(problem.detail)
          : problem.title,
    );
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
  // §Selection statistics: multi-seed eval, progress-tie noise band,
  // fresh-seed re-eval of the kept best, hack-income screen toggle.
  eval_seeds?: number | null;
  progress_epsilon?: number | null;
  fresh_eval_seeds?: number | null;
  hack_income_screen?: boolean | null;
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
 * the motor-limits form; user reviews/edits, then `applyMotorLimits`.
 *
 * §Ship-19c: optional `actuatorNames` — when supplied, the backend
 * tells Claude to MAP datasheet entries to these MJCF names instead
 * of fabricating new ones, so the upload populates existing rows
 * in the motor-specs table rather than adding orphan rows. */
export async function extractDatasheetPdf(
  projectSlug: string,
  pdf: File,
  actuatorNames?: string[],
): Promise<MotorLimitsRequest> {
  const formData = new FormData();
  formData.append("pdf", pdf);
  if (actuatorNames && actuatorNames.length > 0) {
    formData.append("actuator_names", actuatorNames.join(","));
  }
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

// ── Remote GPU dispatch (§Ship 23d) ─────────────────────────────────
export async function getRemoteSettings(): Promise<RemoteSettings> {
  return handle<RemoteSettings>(await fetch("/api/system/remote"));
}

export async function putRemoteSettings(body: RemoteSettings): Promise<RemoteSettings> {
  return handle<RemoteSettings>(
    await fetch("/api/system/remote", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** Runs ssh connectivity checks against the SAVED settings — PUT first.
 * Can take ~30 s when the host is unreachable. */
export async function runRemoteDoctor(): Promise<RemoteDoctorResponse> {
  return handle<RemoteDoctorResponse>(
    await fetch("/api/system/remote/doctor", { method: "POST" }),
  );
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
  const res = await fetch(previewUrl(slug, opts), { cache: "no-store" });
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
export async function listRewards(
  slug: string,
  /** §Ship 21: optional `<missionSlug>/<stageName>` to scope rewards
   *  to that mission stage's rewards/ dir. Omit for project rewards. */
  stage?: string,
): Promise<RewardVersionSummary[]> {
  const qs = stage ? `?stage=${encodeURIComponent(stage)}` : "";
  return handle<RewardVersionSummary[]>(
    await fetch(`/api/projects/${slug}/rewards${qs}`),
  );
}

export async function getReward(
  slug: string,
  version: number,
  /** §Ship 21: optional `<missionSlug>/<stageName>` to scope. */
  stage?: string,
): Promise<RewardVersionDetail> {
  const qs = stage ? `?stage=${encodeURIComponent(stage)}` : "";
  return handle<RewardVersionDetail>(
    await fetch(`/api/projects/${slug}/rewards/${version}${qs}`),
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
  /** §Ship 21d: optional `<missionSlug>/<stageName>` to fetch the
   *  diagnosis from that stage's runs dir instead of the project
   *  runs dir. Required for the "Why this edit?" panel to work on
   *  mission stage reward versions. */
  stage?: string,
): Promise<RewardDiagnosisPayload> {
  const qs = stage ? `?stage=${encodeURIComponent(stage)}` : "";
  return handle<RewardDiagnosisPayload>(
    await fetch(`/api/projects/${slug}/rewards/${version}/diagnosis${qs}`),
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
  RunControlState,
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

// §Ship 39 (H1): interactive control for a live run — flip Auto/Manual,
// resume a pause (optionally with human feedback), or stop cleanly.
export async function controlRun(
  slug: string,
  runId: string,
  body: {
    mode?: "manual" | "auto";
    resume?: boolean;
    feedback?: string | null;
    stop?: boolean;
    gen_retry?: boolean;
    gen_continue?: boolean;
  },
): Promise<RunControlState> {
  return handle<RunControlState>(
    await fetch(`/api/projects/${slug}/runs/${runId}/control`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
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

// ── Policies (trained checkpoints + deployment-bundle export) ─────────
import type { PolicySummary } from "./types";

/** Disk-backed list of exportable trained iterations. Pass `runId` only
 *  for mission stage runs (their iters live in the stage's own tree). */
export async function listPolicies(
  slug: string, runId?: string,
): Promise<PolicySummary[]> {
  const q = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return handle<PolicySummary[]>(
    await fetch(`/api/projects/${slug}/policies${q}`),
  );
}

/** Download URL for a policy deployment bundle (zip: checkpoint + ONNX/
 *  TorchScript + reward/env-spec/config snapshots + DEPLOY.md). */
export function policyExportUrl(
  slug: string, iterIndex: number, runId?: string,
): string {
  const q = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return `/api/projects/${slug}/policies/${iterIndex}/export${q}`;
}

// ── Missions (Ship 18a) ──────────────────────────────────────────────
import type {
  CreateMissionRequest,
  DeleteMissionResponse,
  MissionDetail,
  MissionSummary,
} from "./types";

export async function listMissions(slug: string): Promise<MissionSummary[]> {
  return handle<MissionSummary[]>(
    await fetch(`/api/projects/${slug}/missions`),
  );
}

export async function getMission(
  slug: string, missionSlug: string,
): Promise<MissionDetail> {
  return handle<MissionDetail>(
    await fetch(`/api/projects/${slug}/missions/${missionSlug}`),
  );
}

export async function createMission(
  slug: string, body: CreateMissionRequest,
): Promise<JobSummary> {
  return handle<JobSummary>(
    await fetch(`/api/projects/${slug}/missions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** §Ship-19d optional body. The backend's RunMissionRequest is
 *  open-shape — every field is optional. An empty body preserves the
 *  Ship 16 default (per-stage budget from mission.json, no Goal A/B). */
export interface RunMissionRequestBody {
  iterations_override?: number | null;
  steps_per_iter?: number | null;
  seed?: number | null;
  early_stop_on_criterion?: boolean;
  criterion_stability_window?: number;
  extend_on_improvement?: boolean;
  max_extensions_per_stage?: number;
  extension_factor?: number;
  extension_improvement_threshold?: number;
  // §Ship 34/35: per-stage objective fitness-in-the-loop (built-in name
  // or "gen:<id>") + observe/steer mode.
  fitness_metric?: string | null;
  fitness_mode?: "observe" | "steer";
  // §MISSION_RUN_PARITY: per-launch knobs mirrored from NewRunDialog,
  // applied uniformly to every stage. blank/undefined = inherited config.
  edit_candidates?: number | null;
  rollout_episodes?: number | null;
  max_episode_steps?: number | null;
  playback_speed?: number | null;
  render_width?: number | null;
  render_height?: number | null;
  fitness_patience?: number | null;
  num_envs_override?: number | null;
  device_override?: string | null;
  // §mission-persistence increment 2: one-shot bypass of the
  // `stage_metric_required` 409 guard for this launch only. Does not
  // mutate the persisted run_defaults.
  proceed_blind?: boolean;
}

export async function runMission(
  slug: string, missionSlug: string,
  body?: RunMissionRequestBody,
): Promise<JobSummary> {
  const init: RequestInit = { method: "POST" };
  if (body && Object.keys(body).length > 0) {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify(body);
  }
  return handle<JobSummary>(
    await fetch(`/api/projects/${slug}/missions/${missionSlug}/run`, init),
  );
}

export async function deleteMission(
  slug: string, missionSlug: string,
): Promise<DeleteMissionResponse> {
  return handle<DeleteMissionResponse>(
    await fetch(`/api/projects/${slug}/missions/${missionSlug}`, {
      method: "DELETE",
    }),
  );
}

/** WebSocket URL for a mission's active-job event stream (decompose
 *  or execute). Mirrors `runEventsWsUrl`. */
export function missionEventsWsUrl(
  slug: string, missionSlug: string,
): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/projects/${slug}/missions/${missionSlug}/events`;
}

// ── Stage de-siloing (disk-truth iterations + env spec) ───────────────
import type { StageEnvSpec, StageIteration } from "./types";

/** GET .../stages/{stage}/iterations — every iteration on disk for a
 *  stage, regardless of which stage the live UI is scoped to. */
export async function getStageIterations(
  slug: string, missionSlug: string, stageName: string,
): Promise<StageIteration[]> {
  return handle<StageIteration[]>(
    await fetch(
      `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/iterations`,
    ),
  );
}

/** Rollout mp4 URL for a specific stage iteration. */
export function stageRolloutUrl(
  slug: string, missionSlug: string, stageName: string, iterIndex: number,
): string {
  return `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/iterations/${iterIndex}/rollout`;
}

/** Checkpoint download URL for a specific stage iteration. */
export function stageCheckpointUrl(
  slug: string, missionSlug: string, stageName: string, iterIndex: number,
): string {
  return `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/iterations/${iterIndex}/checkpoint`;
}

/** Deployment-bundle (.zip) download URL for a stage iteration — the
 *  checkpoint + ONNX + TorchScript + reward/env spec + DEPLOY.md. */
export function stageExportUrl(
  slug: string, missionSlug: string, stageName: string, iterIndex: number,
): string {
  return `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/iterations/${iterIndex}/export`;
}

/** GET .../stages/{stage}/iterations/{i}/detail — reasoning behind a
 *  finished iteration (diagnosis, cited papers, reward summary). */
export async function getStageIterDetail(
  slug: string, missionSlug: string, stageName: string, iterIndex: number,
): Promise<import("./types").StageIterDetail> {
  return handle(
    await fetch(
      `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/iterations/${iterIndex}/detail`,
    ),
  );
}

/** GET .../stages/{stage}/metric — the stage's objective-metric record. */
export async function getStageObjectiveMetric(
  slug: string, missionSlug: string, stageName: string,
): Promise<import("./types").StageObjectiveMetric> {
  return handle(
    await fetch(
      `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/metric`,
    ),
  );
}

/** GET /projects/{slug}/iterations — disk-truth iteration list for the
 *  PROJECT-level runs tree (plain sculpt runs). No JobManager entry
 *  required, so it keeps working after a backend restart — the data
 *  source for the synthetic "disk:project" run row. Same row shape as
 *  the per-stage endpoint. */
export async function getProjectIterations(
  slug: string,
): Promise<StageIteration[]> {
  return handle<StageIteration[]>(
    await fetch(`/api/projects/${slug}/iterations`),
  );
}

/** Rollout mp4 URL for a project-level iteration (disk-truth; no
 *  JobManager entry required, unlike iterRolloutUrl). */
export function projectIterRolloutUrl(slug: string, iterIndex: number): string {
  return `/api/projects/${slug}/iterations/${iterIndex}/rollout`;
}

/** GET .../stages/{stage}/env-spec — the applied env curriculum for a
 *  stage. `current.meta.source` starting "reference:" ⇒ RSI applied. */
export async function getStageEnvSpec(
  slug: string, missionSlug: string, stageName: string,
): Promise<StageEnvSpec> {
  return handle<StageEnvSpec>(
    await fetch(
      `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/env-spec`,
    ),
  );
}

/** POST .../stages/{stage}/metric/regenerate — user-triggered
 *  regeneration of one stage's steering metric. 202 JobDetail; 409 if
 *  the mission has any other active mission-scoped job (decompose,
 *  execute, a per-stage run, or another regenerate) in flight. */
export async function regenerateStageMetric(
  slug: string, missionSlug: string, stageName: string,
  opts?: { nCandidates?: number },
): Promise<JobDetail> {
  // Default 2 candidates: a user clicking Regenerate is paying attention
  // and wants it to LAND — live evidence (D28): single-candidate regens
  // failed repeatedly on author-quality rolls (near-constant metric, then
  // a forbidden-name loop) where a second candidate would have covered.
  const body = { n_candidates: opts?.nCandidates ?? 2 };
  return handle<JobDetail>(
    await fetch(
      `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/metric/regenerate`,
      {
        method: "POST",
        ...(body
          ? { headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
          : {}),
      },
    ),
  );
}


// ── Dashboard + System ───────────────────────────────────────────────
import type { DashboardSummary, SystemInfo } from "./types";

export async function getDashboard(): Promise<DashboardSummary> {
  return handle<DashboardSummary>(await fetch("/api/dashboard"));
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return handle<SystemInfo>(await fetch("/api/system/info"));
}

// ── §Ship 35: auto-generated objective metrics ────────────────────────
import type { MetricGenProgress, MetricSummary } from "./types";

export async function listProjectMetrics(slug: string): Promise<MetricSummary[]> {
  return handle<MetricSummary[]>(await fetch(`/api/projects/${slug}/metrics`));
}

// §Ship 40: poll live generation progress while a generate is in flight.
export async function getMetricGenProgress(slug: string): Promise<MetricGenProgress> {
  return handle<MetricGenProgress>(
    await fetch(`/api/projects/${slug}/metrics/generate/progress`));
}

export async function generateProjectMetric(
  slug: string,
  body: { behavior_goal: string; review?: boolean; n_candidates?: number;
          calibrate_against?: string | null },
): Promise<MetricSummary> {
  // Each candidate is a ~1-2 min LLM call plus validate/review/regenerate
  // retries, so budget generously — but DO time out: a dropped connection or
  // wedged backend used to leave this promise pending forever, freezing the
  // "Generating…" spinner in NewRunDialog with no way out.
  const timeoutMs = 5 * 60_000 * Math.max(1, body.n_candidates ?? 1);
  return handle<MetricSummary>(
    await fetch(`/api/projects/${slug}/metrics/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    }),
  );
}

export async function calibrateProjectMetric(
  slug: string, metricId: string, against: string,
): Promise<MetricSummary> {
  return handle<MetricSummary>(
    await fetch(`/api/projects/${slug}/metrics/${metricId}/calibrate?against=${encodeURIComponent(against)}`,
      { method: "POST" }),
  );
}

// ── Reports (project-runs vs per-mission) ─────────────────────────────
import type { ReportsSources } from "./types";

export async function getReportsSources(slug: string): Promise<ReportsSources> {
  return handle<ReportsSources>(
    await fetch(`/api/projects/${slug}/reports/sources`),
  );
}

// ── Saved missions (durable disk archive) ─────────────────────────────
import type { SavedEntryDetail, SavedEntrySummary } from "./types";

export async function listSaved(): Promise<SavedEntrySummary[]> {
  return handle<SavedEntrySummary[]>(await fetch("/api/saved"));
}

export async function getSaved(entryId: string): Promise<SavedEntryDetail> {
  return handle<SavedEntryDetail>(
    await fetch(`/api/saved/${encodeURIComponent(entryId)}`),
  );
}

/** URL for an archived file (mp4 / md) under a saved entry. `relpath`
 *  is a "/"-joined path from the manifest (e.g. a stage video); each
 *  segment is encoded but the slashes are preserved. */
export function savedFileUrl(entryId: string, relpath: string): string {
  const encoded = relpath.split("/").map(encodeURIComponent).join("/");
  return `/api/saved/${encodeURIComponent(entryId)}/file/${encoded}`;
}

/** DELETE /saved/{id} — moves the entry to Trash (kind="saved"),
 *  recoverable from Settings → Trash. Returns 204. */
export async function deleteSaved(entryId: string): Promise<void> {
  await handle<void>(
    await fetch(`/api/saved/${encodeURIComponent(entryId)}`, {
      method: "DELETE",
    }),
  );
}

/** POST /projects/{slug}/missions/{ms}/save — schedule a mission_save
 *  job (202 JobDetail). `pinned` maps stage name → iters to force-keep. */
export async function saveMission(
  slug: string, missionSlug: string,
  body?: { pinned?: Record<string, number[]> },
): Promise<JobDetail> {
  const init: RequestInit = { method: "POST" };
  if (body && body.pinned && Object.keys(body.pinned).length > 0) {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify({ pinned: body.pinned });
  }
  return handle<JobDetail>(
    await fetch(`/api/projects/${slug}/missions/${missionSlug}/save`, init),
  );
}

// ── Trash (recoverable deletes) ───────────────────────────────────────
import type { TrashEntry } from "./types";

export async function listTrash(): Promise<TrashEntry[]> {
  return handle<TrashEntry[]>(await fetch("/api/trash"));
}

export async function restoreTrash(entryId: string): Promise<void> {
  await handle<void>(
    await fetch(`/api/trash/${encodeURIComponent(entryId)}/restore`, {
      method: "POST",
    }),
  );
}

/** Permanent delete. The backend requires `confirm` to equal the
 *  entry_id as a fat-finger guard, so the UI echoes it. */
export async function purgeTrash(entryId: string): Promise<void> {
  await handle<void>(
    await fetch(
      `/api/trash/${encodeURIComponent(entryId)}?confirm=${encodeURIComponent(entryId)}`,
      { method: "DELETE" },
    ),
  );
}

// ── Reference library (R1) ─────────────────────────────────────────────
import type { RefDetail, RefIndexRow, RefMatch } from "./types";

/** GET /references?robot=&q=&k=&llm= — search hits, ranked. `useLlm`
 *  defaults to false: the UI's as-you-type path stays deterministic
 *  (zero API cost, no rerank latency); pass true for an explicit
 *  "rerank with Claude" affordance if one is ever added. */
export async function searchReferences(
  query: string,
  opts?: { robot?: string; k?: number; useLlm?: boolean },
): Promise<RefMatch[]> {
  const u = new URL("/api/references", window.location.origin);
  u.searchParams.set("q", query);
  u.searchParams.set("robot", opts?.robot ?? "g1");
  u.searchParams.set("k", String(opts?.k ?? 10));
  u.searchParams.set("llm", opts?.useLlm ? "1" : "0");
  return handle<RefMatch[]>(await fetch(u.pathname + u.search));
}

/** GET /references (no q) — the slim index listing. Used for the
 *  picker's empty-query state ("browse everything"). */
export async function listReferences(opts?: { robot?: string }): Promise<RefIndexRow[]> {
  const u = new URL("/api/references", window.location.origin);
  u.searchParams.set("robot", opts?.robot ?? "g1");
  return handle<RefIndexRow[]>(await fetch(u.pathname + u.search));
}

export async function getReference(clipId: string): Promise<RefDetail> {
  return handle<RefDetail>(
    await fetch(`/api/references/${encodeURIComponent(clipId)}`),
  );
}

/** Preview keyframe-strip PNG URL. No existence check here — the
 *  backend 404s cleanly when a clip has no preview.png; callers should
 *  hide the <img> on error (see ReferencePickerDialog). */
export function getReferencePreviewUrl(clipId: string): string {
  return `/api/references/${encodeURIComponent(clipId)}/preview`;
}

/** POST .../stages/{stage}/reference body {clip_id} — attach a
 *  reference clip to a stage. 409 if mission-scoped jobs are live (same
 *  guard family as regenerateStageMetric). Returns the updated mission. */
export async function attachStageReference(
  slug: string, missionSlug: string, stageName: string, clipId: string,
): Promise<MissionDetail> {
  return handle<MissionDetail>(
    await fetch(
      `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/reference`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ clip_id: clipId }),
      },
    ),
  );
}

/** DELETE .../stages/{stage}/reference — detach. Same 409 guard. */
export async function detachStageReference(
  slug: string, missionSlug: string, stageName: string,
): Promise<MissionDetail> {
  return handle<MissionDetail>(
    await fetch(
      `/api/projects/${slug}/missions/${missionSlug}/stages/${encodeURIComponent(stageName)}/reference`,
      { method: "DELETE" },
    ),
  );
}

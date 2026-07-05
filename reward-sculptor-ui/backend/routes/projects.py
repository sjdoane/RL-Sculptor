"""Project-lifecycle endpoints.

Contract references (API contract §3.1):
  POST   /projects          → create_project
  GET    /projects          → list_projects
  GET    /projects/{slug}   → get_project
  DELETE /projects/{slug}   → delete_project
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse

from backend.models.project import (
    CreateProjectRequest,
    EnvSpecInfo,
    IterationSettings,
    ProblemDetail,
    ProjectDetail,
    ProjectSettings,
    ProjectSummary,
)
from backend.services import sculptor_bridge
from backend.services.job_manager import JobManager
from backend.services.project_store import BusyError, ProjectStore
from backend.services.robot_library import RobotEntry, get_library


router = APIRouter(prefix="/projects", tags=["projects"])


# ── DI ─────────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


# ── helpers ────────────────────────────────────────────────────────────
def _problem(
    status_code: int,
    title: str,
    *,
    detail: str | None = None,
    type_: str = "about:blank",
    **extra: Any,
) -> JSONResponse:
    body = ProblemDetail(
        type=type_, title=title, status=status_code, detail=detail
    ).model_dump()
    body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


# ── endpoints ──────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=ProjectDetail,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"model": ProblemDetail, "description": "slug collision"},
        503: {"model": ProblemDetail, "description": "sculptor unavailable"},
    },
)
def create_project(
    body: CreateProjectRequest,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    if not sculptor_bridge.sculptor_ok():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "sculptor library not importable",
            detail=sculptor_bridge.sculptor_error(),
            type_="/problems/sculptor-unavailable",
        )

    # M3: resolve library_slug -> adapter + task derivation BEFORE the
    # mjlab preflight so the preflight sees the correct body fields.
    library_entry: Optional[RobotEntry] = None
    if body.library_slug:
        library_entry = _resolve_library_entry(body)
        if isinstance(library_entry, JSONResponse):
            return library_entry  # structured 404

    # M5: reject adapters not in the registry. Coming-soon adapters
    # (isaac, mjx, rllib) are allowed — they scaffold with the gym_sb3
    # placeholder + adapter_unavailable=true.
    from sculptor.adapters import ADAPTER_REGISTRY  # type: ignore

    adapter_coming_soon = False
    adapter_info = ADAPTER_REGISTRY.get(body.adapter)
    if adapter_info is not None and adapter_info.status == "coming_soon":
        adapter_coming_soon = True

    # mjlab preflight (MJLAB_PIVOT_DESIGN §9). Runs BEFORE scaffold —
    # fail fast on missing GPU / missing mjlab / VRAM headroom.
    if body.adapter == "mjlab":
        mjlab_err = _validate_mjlab_preflight(body)
        if mjlab_err is not None:
            return mjlab_err

    try:
        # sculpt_init knows `gym_sb3` and `mjlab` natively — it writes the
        # correct `rewards/v0.py` template per adapter contract (scalar vs
        # batched `compute_reward_batched`). Coming-soon adapters still
        # scaffold as gym_sb3 since they have no real template yet;
        # [adapter] is rewritten afterward.
        scaffold_adapter = "gym_sb3" if adapter_coming_soon else body.adapter
        detail = store.create(
            display_name=body.display_name,
            adapter=scaffold_adapter,
            description=body.description,
            iteration_budget=body.iteration_budget,
            behavior_goal=body.behavior_goal,
        )
    except FileExistsError as e:
        return _problem(
            status.HTTP_409_CONFLICT,
            "project already exists",
            detail=str(e),
            type_="/problems/slug-collision",
        )
    except ValueError as e:
        return _problem(
            status.HTTP_409_CONFLICT,
            "could not allocate slug",
            detail=str(e),
            type_="/problems/slug-collision",
        )
    except RuntimeError as e:
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "sculptor scaffold failed",
            detail=str(e),
            type_="/problems/sculptor-unavailable",
        )

    # mjlab post-scaffold: rewrite [adapter] section + [iteration].
    if body.adapter == "mjlab":
        adapter_cfg = {
            "task_id": body.task_id,
            "num_envs": body.num_envs or 2048,
            "device": body.gpu_device or "cuda:0",
        }
        store.set_adapter_section(
            detail.slug,
            "sculptor.adapters.mjlab.MjlabAdapter",
            adapter_cfg,
        )
        # `steps_per_iter` is passed straight through to rsl_rl's
        # `max_iterations`. The scaffold's 50_000 default is fine for
        # gym_sb3 (env steps) but catastrophic for mjlab — ~1 s per
        # rsl_rl iter on an 8 GiB laptop GPU means 50k learning iters
        # ≈ 14 h for a single sculpt iter. mjlab's own default is 1500
        # (~25 min / sculpt iter), which is what we stamp here.
        _override_iteration_key(
            detail.project_dir, "steps_per_iter", 1500
        )

    # M5: coming-soon adapter post-scaffold — rewrite [adapter] +
    # mark metadata.adapter_unavailable so the UI disables the Train
    # button with a tooltip pointing at the adoption guide.
    if adapter_coming_soon and adapter_info is not None:
        store.set_adapter_section(
            detail.slug,
            adapter_info.class_path,
            {},  # coming-soon adapters take no config yet
        )
        try:
            meta = store._read_metadata(detail.slug) or {}  # noqa: SLF001
            rs = dict(meta.get("robot_source") or {})
            rs["adapter_unavailable"] = True
            rs["adapter_name"] = adapter_info.name
            rs["adoption_guide_url"] = adapter_info.adoption_guide_url
            meta["robot_source"] = rs
            store._write_metadata(detail.slug, meta)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass

    # M3: library-driven post-scaffold — write robot_source + auto-seed KG.
    if library_entry is not None:
        seeded_n = _finalize_library_scaffold(store, detail.slug, library_entry)
        # Fire auto-ingest if ANTHROPIC_API_KEY is set + there's at
        # least one arxiv seed. Non-fatal on any failure (seeds stay
        # on disk for manual ingest from the KG tab).
        if seeded_n > 0:
            _maybe_fire_kg_ingest(request, store, detail.slug)
        # Re-read so ready_to_train + library_slug surface in the response.
        detail = store.get(detail.slug) or detail
    elif body.adapter == "mjlab" or adapter_coming_soon:
        # Re-read so adapter_class/adapter_config reflect the rewrite.
        detail = store.get(detail.slug) or detail

    return detail


def _resolve_library_entry(
    body: CreateProjectRequest,
) -> "RobotEntry | JSONResponse":
    """Look up the library entry and override body fields to match.

    For mjlab_ready: body.adapter="mjlab", body.task_id from the first
    preconfigured_task, body.num_envs from the recommendation.
    For gymnasium_compatible: body.adapter="gym_sb3", env_id derived
    from the first preconfigured_task's task_id (which is the gym
    env_id for this category — e.g. "Hopper-v4").
    For preview_only: body.adapter="gym_sb3" for preview-only scaffold.
    """
    library = get_library()
    entry = library.get_robot(body.library_slug or "")
    if entry is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "library robot not found",
            detail=f"no library entry for slug {body.library_slug!r}",
            type_="/problems/library-slug-not-found",
        )

    if entry.training_support == "mjlab_ready":
        if not entry.preconfigured_tasks:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "mjlab-ready robot has no tasks",
                detail=(
                    f"library entry {entry.slug!r} is marked mjlab_ready "
                    f"but has no preconfigured_tasks (D-guard may have "
                    f"demoted all entries)."
                ),
                type_="/problems/library-entry-invalid",
            )
        first_task = entry.preconfigured_tasks[0]
        body.adapter = "mjlab"
        # Only override if the caller did not explicitly specify these.
        if not body.task_id:
            body.task_id = first_task.task_id
        if not body.num_envs:
            body.num_envs = first_task.recommended_num_envs
    elif entry.training_support == "gymnasium_compatible":
        body.adapter = "gym_sb3"
    else:  # preview_only
        body.adapter = "gym_sb3"
    return entry


def _finalize_library_scaffold(
    store: ProjectStore,
    slug: str,
    entry: RobotEntry,
) -> int:
    """Write robot_source sidecar + seed kg_seeds.yml. Returns the number
    of arxiv paper seeds appended to kg_seeds.yml (0 if the library
    entry had no paper references)."""
    # robot_source → metadata.json
    robot_source = {
        "kind": "library",
        "library_slug": entry.slug,
        "training_support": entry.training_support,
        "category": entry.category,
        "menagerie_package": entry.menagerie_package,
        # Repo references stashed here for the future "Related repos"
        # KG panel (MJLAB_PIVOT_DESIGN §6.3); not ingested into sqlite.
        "related_repos": [
            {"url": r.url, "citation": r.citation}
            for r in entry.references if r.kind == "repo"
        ],
    }
    store.write_robot_source(slug, robot_source)

    # Seed kg_seeds.yml from paper references (arxiv only — repos are
    # not ingestible by sculptor's ingest pipeline).
    paper_seeds: list[dict] = []
    for ref in entry.references:
        if ref.kind != "paper":
            continue
        arxiv_id = _extract_arxiv_id(ref.url)
        if arxiv_id:
            paper_seeds.append({
                "arxiv_id": arxiv_id,
                "citation": ref.citation,
            })
    if paper_seeds:
        try:
            store.append_seeds(slug, paper_seeds)
        except Exception as e:  # noqa: BLE001
            # Non-fatal — project creation should still succeed even if
            # seed-write fails.
            import logging
            logging.getLogger("reward-sculptor-ui").warning(
                "append_seeds failed for %s: %s", slug, e,
            )
            return 0
    return len(paper_seeds)


def _maybe_fire_kg_ingest(
    request: Request,
    store: ProjectStore,
    slug: str,
) -> None:
    """If ANTHROPIC_API_KEY is set and the app has a JobManager, fire an
    ingest+extract job for the newly-seeded project. Best-effort — any
    failure (missing dep, JobManager not bound, etc.) is swallowed and
    the seeds remain on disk for the user to trigger manually.
    """
    import os

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return
    try:
        jobs = getattr(request.app.state, "job_manager", None)
        if jobs is None:
            return
        from backend.services.kg_jobs import run_ingest_extract_job
        project_dir = store._project_dir(slug)  # noqa: SLF001
        jobs.submit(
            kind="kg_ingest_extract",
            project_slug=slug,
            fn=run_ingest_extract_job(
                project_dir=project_dir,
                auto_extract=True,
            ),
            params={"auto_from_library": True},
        )
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger("reward-sculptor-ui").warning(
            "auto KG ingest failed to enqueue for %s: %s", slug, e,
        )


_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def _extract_arxiv_id(url: str) -> Optional[str]:
    m = _ARXIV_ID_RE.search(url or "")
    return m.group("id") if m else None


def _override_iteration_key(project_dir: str, key: str, value: Any) -> None:
    """Rewrite a single `key = N` line under the `[iteration]` table
    in the project's config.toml. Leaves the rest of the file intact.
    No-op if the key isn't present (template has it unless the caller
    hand-edited the file)."""
    from pathlib import Path

    path = Path(project_dir) / "config.toml"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    # Match the existing line under the current `[iteration]` block —
    # simple single-line rewrite so we don't touch adjacent keys.
    pattern = re.compile(
        rf"^{re.escape(key)}\s*=\s*\S+.*$", re.MULTILINE
    )
    # Callable repl so `value` can contain `\`/`"` without re.sub
    # eating the escapes (same bugfix pattern as `_upsert_iteration_key`).
    _repl = f"{key} = {value}"
    new_text, n = pattern.subn(lambda _m: _repl, text, count=1)
    if n:
        path.write_text(new_text, encoding="utf-8")


def _toml_value(v: Any) -> str:
    """Format `v` as a TOML literal. Raises ValueError for values
    that would silently corrupt the file (NaN / Inf / multi-line
    string / None / unsupported type). §Ship-8 hotfix — caller layer
    turns the exception into a 422.
    """
    import math

    if v is None:
        raise ValueError("None is not representable in TOML — use unset.")
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError(f"non-finite float {v!r} not allowed in TOML")
        return str(v)
    if isinstance(v, str):
        if "\n" in v or "\r" in v:
            raise ValueError(
                "multi-line strings not supported — TOML basic strings "
                "cannot contain raw newlines."
            )
        esc = v.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{esc}"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise ValueError(f"unsupported TOML value type: {type(v).__name__}")


def _find_iteration_section_span(text: str) -> tuple[int, int, bool]:
    """Return `(start_body, end_body, exists)` for the `[iteration]`
    table. §Ship-8 hotfix (critique #1): scope every edit to this
    range so a `seed = N` in `[adapter]` can't be clobbered.
    """
    header = re.search(r"^\[iteration\]\s*\n", text, flags=re.MULTILINE)
    if header is None:
        return -1, -1, False
    start = header.end()
    nxt = re.search(r"^\[[^\]]+\]\s*$", text[start:], flags=re.MULTILINE)
    end = len(text) if nxt is None else start + nxt.start()
    return start, end, True


def _upsert_iteration_key(project_dir: str, key: str, value: Any) -> None:
    """Write `key = <toml value>` under the `[iteration]` table of
    config.toml. Replaces in place if the key exists, otherwise
    appends inside the section. Section-scoped so cross-table
    collisions can't happen.
    """
    from pathlib import Path

    path = Path(project_dir) / "config.toml"
    if not path.is_file():
        return

    new_line = f"{key} = {_toml_value(value)}"
    text = path.read_text(encoding="utf-8")
    start, end, exists = _find_iteration_section_span(text)

    if not exists:
        # Create the section at EOF with a blank-line separator.
        trimmed = text.rstrip()
        suffix = "" if trimmed == "" else "\n\n"
        new_text = trimmed + f"{suffix}[iteration]\n{new_line}\n"
        path.write_text(new_text, encoding="utf-8")
        return

    section = text[start:end]
    key_pattern = re.compile(
        rf"^{re.escape(key)}\s*=\s*.*$", re.MULTILINE
    )
    # Callable repl so `new_line` escapes aren't reinterpreted.
    new_section, n = key_pattern.subn(lambda _m: new_line, section, count=1)

    if n == 0:
        # Append inside the section. Preserve the trailing blank-line
        # separator (if any) that would otherwise end up before the
        # next table header; otherwise just ensure we end with `\n`.
        if section.endswith("\n\n"):
            new_section = section[:-1] + new_line + "\n\n"
        elif section.endswith("\n") or section == "":
            new_section = section + new_line + "\n"
        else:
            new_section = section + "\n" + new_line + "\n"

    new_text = text[:start] + new_section + text[end:]
    path.write_text(new_text, encoding="utf-8")


class _ConfigTomlCorrupt(RuntimeError):
    """Raised when the project's config.toml fails to parse.

    §Ship-8 hotfix (critique medium-8): swallowing the TOML decode
    error and returning `{}` let the UI display a blank form over a
    corrupt file — the next PATCH would then silently append junk.
    Surface the error so the route layer returns 500 + a clear reason.
    """


def _read_iteration_dict(project_dir: str) -> dict[str, Any]:
    """Read the `[iteration]` table as a plain dict. Returns `{}` for
    a missing file (fresh project). Raises `_ConfigTomlCorrupt` when
    the file exists but doesn't parse."""
    from pathlib import Path
    import tomllib

    path = Path(project_dir) / "config.toml"
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise _ConfigTomlCorrupt(
            f"config.toml at {path} is malformed: {e}"
        ) from e
    section = cfg.get("iteration") or {}
    return section if isinstance(section, dict) else {}


def _validate_mjlab_preflight(body: CreateProjectRequest) -> Optional[JSONResponse]:
    """Preflight for mjlab projects. Returns None when valid, or a
    ProblemDetail 412 JSONResponse when any gate fails.

    Per MJLAB_PIVOT_DESIGN §9 + verification #3 of M2.
    """
    # (a) task_id provided
    if not body.task_id:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "mjlab task_id required",
            detail=(
                "Projects with adapter='mjlab' must include `task_id` "
                "(e.g. 'Mjlab-Velocity-Flat-Unitree-Go1')."
            ),
            type_="/problems/mjlab-task-required",
        )

    # (b) mjlab importable (find_spec — no actual import).
    if not sculptor_bridge.mjlab_available():
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "mjlab not installed",
            detail=(
                "The sculptor environment does not have mjlab installed. "
                "Run `uv add 'mjlab[cu128]'` inside ~/projects/RewardSculptor "
                "and restart the backend."
            ),
            type_="/problems/mjlab-missing",
        )

    # (c) GPU + CUDA version.
    info = sculptor_bridge.gpu_info()
    if not info.get("cuda_available"):
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "mjlab requires an NVIDIA GPU",
            detail=(
                "mjlab training requires CUDA. "
                f"Detected torch_version={info.get('torch_version')}, "
                f"cuda_available={info.get('cuda_available')}. Pick a "
                "Gymnasium-compatible adapter or install CUDA drivers."
            ),
            type_="/problems/gpu-required",
        )
    if not sculptor_bridge.cuda_version_ok():
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "CUDA version too old for mjlab",
            detail=(
                f"mjlab requires CUDA 12.4 or newer for CUDA-graph "
                f"conditional execution. Detected: {info.get('cuda_version')!r}."
            ),
            type_="/problems/cuda-version",
        )

    # (d) Live-VRAM preflight (M4 §6). Uses pynvml-backed free-memory
    # snapshot + per-env coefficient cache when available.
    from backend.services.preflight import check_mjlab_preflight
    num_envs = body.num_envs or 2048
    result = check_mjlab_preflight(
        task_id=body.task_id or "",
        num_envs=num_envs,
        device=body.gpu_device or "cuda:0",
        project_dir=None,  # project doesn't exist yet at create-time
        min_gpu_memory_gb=0.0,
    )
    if not result.ok:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "device preflight failed",
            detail=result.reason or "",
            type_=result.problem_type or "/problems/device-unavailable",
            suggested_num_envs=result.suggested_num_envs,
            free_vram_gb=round(result.free_vram_gb, 2),
            total_vram_gb=round(result.total_vram_gb, 2),
            estimated_required_gb=round(result.estimated_required_gb, 2),
            device_index=result.device_index,
            device_name=result.device_name,
        )

    return None


@router.get("", response_model=list[ProjectSummary])
def list_projects(
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> list[ProjectSummary]:
    summaries = store.list()
    # §Ship 22b (Finding A): attach the latest sculpt_run's metric +
    # history per project so the Projects/Dashboard cards can show a best
    # reward + sparkline. JobManager is the source of truth for run
    # metrics (same data the /dashboard endpoint surfaces). `jobs.list`
    # is newest-first, so the first job seen per slug is the latest run.
    latest_by_slug: dict[str, list[Optional[float]]] = {}
    for job in jobs.list(kind="sculpt_run"):
        slug = job.project_slug or ""
        if not slug or slug in latest_by_slug:
            continue
        hist_raw = (job.params or {}).get("primary_metric_history") or []
        history: list[Optional[float]] = []
        if isinstance(hist_raw, list):
            for v in hist_raw:
                history.append(float(v) if isinstance(v, (int, float)) else None)
        latest_by_slug[slug] = history
    for s in summaries:
        history = latest_by_slug.get(s.slug)
        if history:
            s.primary_metric_history = history
            vals = [v for v in history if v is not None]
            s.primary_metric = max(vals) if vals else None
    return summaries


@router.get(
    "/{slug}",
    response_model=ProjectDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_project(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    return detail


@router.delete(
    "/{slug}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail}},
)
def delete_project(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    try:
        ok = store.delete(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT,
            "project busy",
            detail=str(e),
            type_="/problems/state-conflict",
        )
    if not ok:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── §env generalization: read-only env-spec surface ────────────────────
@router.get(
    "/{slug}/env-spec",
    response_model=EnvSpecInfo,
    responses={404: {"model": ProblemDetail}},
)
def get_project_env_spec(
    slug: str, store: ProjectStore = Depends(get_store),
) -> Any:
    """The project's environment-adaptation spec: the active
    env/current.json (goal-conditioned, diagnoser-iterated) plus the
    version list. `active=false` for projects on task defaults. Read-only
    — the spec is generated and iterated by the sculpt loop; per-iter
    changes ride the `env_spec_updated` run events."""
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    env_dir = Path(detail.project_dir) / "env"
    current: Optional[dict] = None
    cur_path = env_dir / "current.json"
    if cur_path.is_file():
        try:
            loaded = json.loads(cur_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, ValueError):
            current = None
    versions: list[str] = []
    if env_dir.is_dir():
        for p in env_dir.glob("v*.json"):
            if p.stem[1:].isdigit():
                versions.append(p.stem)
        versions.sort(key=lambda s: int(s[1:]))
    return EnvSpecInfo(
        active=current is not None, current=current, versions=versions)


# ── §Ship-8: per-project settings (editable config.toml [iteration]) ───
@router.get(
    "/{slug}/settings",
    response_model=ProjectSettings,
    responses={404: {"model": ProblemDetail}},
)
def get_project_settings(
    slug: str, store: ProjectStore = Depends(get_store),
) -> Any:
    """Return the project's `[iteration]` config.toml block.

    Known keys (per `IterationSettings` schema) are surfaced; any
    hand-added custom keys are silently dropped from the typed view
    — edit them via the terminal if you need them.
    """
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )
    try:
        data = _read_iteration_dict(detail.project_dir)
    except _ConfigTomlCorrupt as e:
        # §Ship-8 hotfix — surface corruption as 500 instead of an
        # empty form that would mask the real problem on save.
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "config.toml is malformed",
            detail=str(e),
            type_="/problems/state-conflict",
        )
    known = set(IterationSettings.model_fields.keys())
    clean = {k: v for k, v in data.items() if k in known}
    return ProjectSettings(iteration=IterationSettings(**clean))


@router.patch(
    "/{slug}/settings",
    response_model=ProjectSettings,
    responses={
        200: {"description": "Settings persisted; returns the merged view."},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def patch_project_settings(
    slug: str,
    body: ProjectSettings,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Upsert fields under `[iteration]` in config.toml. Only the keys
    the caller sends (non-None) are touched; existing keys not in the
    body are left untouched so the UI can PATCH a single field."""
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )
    # `model_dump(exclude_unset=True)` ensures we only touch fields
    # the client actually sent — distinct from "sent null" which we
    # treat the same as "unset" for now (the TOML key stays put).
    iteration = body.iteration.model_dump(exclude_unset=True)

    # §Ship-8 hotfix (critique medium-4): empty PATCH body is a no-op
    # that returns the current state — UI dirty-then-revert flows
    # shouldn't 422.
    if not iteration:
        try:
            fresh = _read_iteration_dict(detail.project_dir)
        except _ConfigTomlCorrupt as e:
            return _problem(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "config.toml is malformed", detail=str(e),
                type_="/problems/state-conflict",
            )
        known = set(IterationSettings.model_fields.keys())
        return ProjectSettings(
            iteration=IterationSettings(
                **{k: v for k, v in fresh.items() if k in known}
            )
        )

    try:
        lock = store.acquire_lock(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT, "project busy",
            detail=str(e), type_="/problems/state-conflict",
        )
    try:
        for key, value in iteration.items():
            if value is None:
                continue
            try:
                _upsert_iteration_key(detail.project_dir, key, value)
            except ValueError as e:
                # §Ship-8 hotfix — `_toml_value` rejects NaN/Inf/None/
                # multi-line strings. Surface as 422 with the offending
                # field so the UI can highlight it.
                return _problem(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "invalid setting value",
                    detail=f"{key}: {e}",
                    type_="/problems/validation-error",
                )
    finally:
        lock.release()

    try:
        fresh = _read_iteration_dict(detail.project_dir)
    except _ConfigTomlCorrupt as e:
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "config.toml became malformed after write", detail=str(e),
            type_="/problems/state-conflict",
        )
    known = set(IterationSettings.model_fields.keys())
    return ProjectSettings(
        iteration=IterationSettings(
            **{k: v for k, v in fresh.items() if k in known}
        )
    )

"""Reward-version endpoints.

  GET  /projects/{slug}/rewards
  GET  /projects/{slug}/rewards/{version}
  PUT  /projects/{slug}/rewards/{version}   ← manual edit, creates v+1

Per R1 (Prompt 7): if a sculpt run is active for the project the PUT is
refused with 409. The UI disables Monaco in that case, so this is a
belt-and-braces check for race conditions between status probe and PUT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from backend.models.kg import JobSummary
from backend.models.project import ProblemDetail
from backend.models.reward import (
    ManualEditRequest,
    RewardVersionDetail,
    RewardVersionSummary,
)
from backend.services import kg_store, reward_store, sculptor_bridge
from backend.services.job_manager import JobManager
from backend.services.project_store import BusyError, ProjectStore
from backend.services.reward_store import (
    ConcurrencyError,
    RewardValidationError,
)


router = APIRouter(tags=["rewards"])


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


def _rewards_dir(store: ProjectStore, slug: str) -> Path | None:
    detail = store.get(slug)
    if detail is None:
        return None
    return Path(detail.project_dir) / "rewards"


# ── list ──────────────────────────────────────────────────────────────
@router.get(
    "/projects/{slug}/rewards",
    response_model=list[RewardVersionSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_rewards(
    slug: str, store: ProjectStore = Depends(get_store)
) -> Any:
    rewards_dir = _rewards_dir(store, slug)
    if rewards_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    history = store.read_metric_history(slug)
    out: list[RewardVersionSummary] = []
    for n, path in reward_store.list_versions(rewards_dir):
        out.append(reward_store.summarize_version(rewards_dir, n, path, history))
    # Newest first.
    out.sort(key=lambda s: s.version, reverse=True)
    return out


# ── detail ────────────────────────────────────────────────────────────
@router.get(
    "/projects/{slug}/rewards/{version}",
    response_model=RewardVersionDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_reward(
    slug: str,
    version: int,
    store: ProjectStore = Depends(get_store),
) -> Any:
    rewards_dir = _rewards_dir(store, slug)
    if rewards_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    history = store.read_metric_history(slug)
    detail = reward_store.load_detail(rewards_dir, version, history)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "reward version not found",
            detail=f"no v{version}.py in project {slug!r}",
            type_="/problems/not-found",
        )
    return detail


# ── PUT manual edit ───────────────────────────────────────────────────
@router.put(
    "/projects/{slug}/rewards/{version}",
    response_model=RewardVersionDetail,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        422: {"model": ProblemDetail},
    },
)
def put_reward(
    slug: str,
    version: int,
    body: ManualEditRequest,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    rewards_dir = Path(detail.project_dir) / "rewards"

    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT,
            "sculpt run in progress",
            detail=(
                "Manual edits are locked while a sculpt run is active. "
                "Stop the run or wait for it to finish, then try again."
            ),
            type_="/problems/state-conflict",
        )

    # Parent-version path param is advisory — the real concurrency guard
    # is `expected_parent_version` in the body.
    if body.expected_parent_version != version:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "parent version mismatch",
            detail=(
                f"URL has version={version} but body has "
                f"expected_parent_version={body.expected_parent_version}; "
                "they must agree."
            ),
            type_="/problems/validation-error",
        )

    kg_ids = _kg_arxiv_ids(store, slug)

    try:
        lock = store.acquire_lock(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT,
            "project busy",
            detail=str(e),
            type_="/problems/state-conflict",
        )
    try:
        try:
            result = reward_store.apply_manual_edit(
                rewards_dir=rewards_dir,
                body=body,
                kg_arxiv_ids=kg_ids,
            )
        except FileNotFoundError as e:
            return _problem(
                status.HTTP_404_NOT_FOUND,
                "no reward versions yet",
                detail=str(e),
                type_="/problems/not-found",
            )
        except ConcurrencyError as e:
            return _problem(
                status.HTTP_409_CONFLICT,
                "parent version is stale",
                detail=str(e),
                type_="/problems/concurrency-conflict",
                expected=e.expected,
                current=e.current,
            )
        except RewardValidationError as e:
            return _problem(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "reward failed validation",
                detail="; ".join(e.violations),
                type_="/problems/reward-validation",
                violations=e.violations,
                suggestions=e.suggestions,
            )
    finally:
        lock.release()

    history = store.read_metric_history(slug)
    return reward_store.load_detail(rewards_dir, result.new_version, history)


# ── POST regenerate template ─────────────────────────────────────────
@router.post(
    "/projects/{slug}/rewards/regenerate-template",
    response_model=RewardVersionDetail,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def regenerate_template(
    slug: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    """Rewrite `rewards/v0.py` with the scaffold template matching the
    project's current adapter.

    Destructive by design — overwrites any manual edits to v0.py. The UI
    confirms before calling. Refused while a sculpt run is active, same
    lock rule as manual PUT edits. If `v1.py` or later exist, `current.py`
    keeps pointing at the latest version (does not regress).
    """
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    if not sculptor_bridge.sculptor_ok():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "sculptor library not importable",
            detail=sculptor_bridge.sculptor_error(),
            type_="/problems/sculptor-unavailable",
        )
    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT,
            "sculpt run in progress",
            detail=(
                "Template regeneration is locked while a sculpt run is "
                "active. Stop the run or wait for it to finish, then "
                "try again."
            ),
            type_="/problems/state-conflict",
        )

    try:
        lock = store.acquire_lock(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT,
            "project busy",
            detail=str(e),
            type_="/problems/state-conflict",
        )
    try:
        try:
            from sculptor.sculpt import (
                regenerate_reward_template as _regen,
            )
        except ImportError as e:
            return _problem(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "sculptor library regenerate helper missing",
                detail=f"{type(e).__name__}: {e}",
                type_="/problems/sculptor-unavailable",
            )
        try:
            _regen(Path(detail.project_dir))
        except FileNotFoundError as e:
            return _problem(
                status.HTTP_404_NOT_FOUND,
                "project missing config.toml or rewards/",
                detail=str(e),
                type_="/problems/not-found",
            )
    finally:
        lock.release()

    rewards_dir = Path(detail.project_dir) / "rewards"
    history = store.read_metric_history(slug)
    v0_detail = reward_store.load_detail(rewards_dir, 0, history)
    if v0_detail is None:
        # Should never happen — regenerate writes v0.py.
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "v0.py missing after regenerate",
            detail="expected rewards/v0.py to exist after regenerate",
            type_="/problems/internal-error",
        )
    return v0_detail


# ── POST prompt-driven reward edit (Phase 3) ─────────────────────────
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated


class PromptEditRequest(BaseModel):
    """POST /projects/{slug}/rewards/prompt — Claude rewrites v<n+1>.py
    from a natural-language prompt, without running a full sculpt loop.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: Annotated[
        str,
        Field(
            min_length=3, max_length=2000,
            description=(
                "Natural-language instruction, e.g. 'reward the robot "
                "for jumping 30 cm with low torque and minimal lateral "
                "drift'. Fed verbatim into the edit_rewriter prompt's "
                "rationale field."
            ),
        ),
    ]
    expected_parent_version: int = Field(ge=0)


@router.post(
    "/projects/{slug}/rewards/prompt",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        422: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def prompt_reward_edit(
    slug: str,
    body: PromptEditRequest,
    request: Request,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    """Prompt Claude for a reward rewrite without running the full
    sculpt train→diagnose→edit loop. Fires a background job; poll
    `GET /jobs/{job_id}` for completion. On success, the new v<n+1>.py
    appears in `GET /projects/{slug}/rewards`."""
    import os

    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if not sculptor_bridge.sculptor_ok():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "sculptor library not importable",
            detail=sculptor_bridge.sculptor_error(),
            type_="/problems/sculptor-unavailable",
        )

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ANTHROPIC_API_KEY required for prompt-driven reward edits",
            detail=(
                "Set ANTHROPIC_API_KEY in the shell launching the "
                "backend OR in ../RewardSculptor/.env, then restart."
            ),
            type_="/problems/anthropic-key-missing",
        )

    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT,
            "sculpt run in progress",
            detail=(
                "Reward edits are locked while a sculpt run is active. "
                "Stop the run or wait for it to finish, then try again."
            ),
            type_="/problems/state-conflict",
        )

    # Also refuse if another reward_prompt_edit is already mid-flight
    # for this project — two concurrent rewrites would race on v<n+1>.
    for existing in jobs.list(project_slug=slug):
        if existing.kind == "reward_prompt_edit" and existing.status in (
            "queued", "running",
        ):
            return _problem(
                status.HTTP_409_CONFLICT,
                "prompt edit already active",
                detail=(
                    "Another prompt-driven reward edit is in progress. "
                    "Wait for it to finish before firing another."
                ),
                type_="/problems/job-busy",
                active_job_id=existing.job_id,
            )

    from backend.services.reward_jobs import run_reward_prompt_edit_job

    job = jobs.submit(
        kind="reward_prompt_edit",  # type: ignore[arg-type]
        project_slug=slug,
        fn=run_reward_prompt_edit_job(
            project_dir=Path(detail.project_dir),
            user_prompt=body.prompt,
            expected_parent_version=body.expected_parent_version,
        ),
        params={
            "prompt_preview": body.prompt[:200],
            "expected_parent_version": body.expected_parent_version,
        },
    )
    return job.to_summary()


# ── GET diagnosis for a version (Phase 7a) ────────────────────────────
@router.get(
    "/projects/{slug}/rewards/{version}/diagnosis",
    responses={
        404: {"model": ProblemDetail},
    },
)
def get_reward_diagnosis(
    slug: str,
    version: int,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Return the diagnosis.json that triggered the edit producing
    `v<version>.py`. Mapping: v<n>.py is written by `iter_<n-1>`, so
    the diagnosis lives at `<project>/runs/iter_<n-1>/diagnosis.json`.

    Returns the raw payload as a dict. 404 when the project, version,
    or diagnosis file is missing — common cases: v0 (no triggering
    diagnosis), human-authored versions (no sculpt diagnose step),
    or a run that never reached the diagnose stage.
    """
    import json

    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    if version <= 0:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "no diagnosis for v0",
            detail=(
                "v0.py is the scaffolded starter — there was no "
                "diagnose step that produced it."
            ),
            type_="/problems/not-found",
        )
    diag_path = (
        Path(detail.project_dir) / "runs" / f"iter_{version - 1}" / "diagnosis.json"
    )
    if not diag_path.is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "no diagnosis recorded for this version",
            detail=(
                f"expected {diag_path.name} under iter_{version - 1}/ — "
                "the version may be human-authored or the sculpt run "
                "did not reach the diagnose stage."
            ),
            type_="/problems/not-found",
        )
    try:
        payload = json.loads(diag_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "diagnosis.json malformed",
            detail=f"{type(e).__name__}: {e}",
            type_="/problems/internal-error",
        )
    return payload


def _kg_arxiv_ids(store: ProjectStore, slug: str) -> set[str]:
    """Every arxiv_id currently in the project KG. Used as the
    allowlist for reward references."""
    detail = store.get(slug)
    if detail is None:
        return set()
    try:
        papers = kg_store.list_papers(Path(detail.project_dir))
    except Exception:  # noqa: BLE001
        return set()
    return {p.arxiv_id for p in papers}

"""Physics-editor endpoints (M7 Phase 5, prompt-first).

  GET  /projects/{slug}/physics              → current MJCF + summary.
  POST /projects/{slug}/physics/prompt       → Claude rewrite job (202).
  PUT  /projects/{slug}/physics/fields       → 501 (form-edit deferred).

Each prompt-edit commits the new MJCF as a `physics: ...` commit in
the project's git repo. Invalid XML (failed `MjModel.from_xml_string`)
is rejected without touching the file on disk.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.models.kg import JobSummary
from backend.models.project import ProblemDetail
from backend.services.job_manager import Job, JobManager
from backend.services.project_store import ProjectStore

router = APIRouter(tags=["physics"])


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


def _project_dir(store: ProjectStore, slug: str) -> Optional[Path]:
    detail = store.get(slug)
    if detail is None:
        return None
    return Path(detail.project_dir)


# ── models ─────────────────────────────────────────────────────────────
class PhysicsLoadResponse(BaseModel):
    """GET /projects/{slug}/physics response shape."""

    model_config = ConfigDict(extra="allow")

    xml_source: str
    mjcf_source_kind: str  # "library" | "upload" | "none"
    mjcf_path: Optional[str] = None
    materialized: bool = False
    summary: dict


class PhysicsPromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Annotated[
        str,
        Field(
            min_length=3, max_length=2000,
            description=(
                "Natural-language request (e.g. 'increase hip joint "
                "damping by 50%', 'add a parallel-elastic spring on "
                "the knee', 'drop timestep to 0.001'). Fed verbatim "
                "to the physics_editor Claude prompt."
            ),
        ),
    ]


class MotorSpec(BaseModel):
    """§Ship-9b: one motor's mechanical spec. All fields optional —
    only the fields the user supplies land in the synthesized prompt."""

    model_config = ConfigDict(extra="forbid")

    peak_torque_nm: Optional[Annotated[float, Field(gt=0, le=10_000)]] = None
    peak_speed_rads: Optional[Annotated[float, Field(gt=0, le=1_000)]] = None
    gear_ratio: Optional[Annotated[float, Field(gt=0, le=1_000)]] = None
    rotor_inertia_kgm2: Optional[
        Annotated[float, Field(gt=0, le=1.0)]
    ] = None
    peak_current_a: Optional[Annotated[float, Field(gt=0, le=1_000)]] = None
    notes: Optional[Annotated[str, Field(max_length=200)]] = None


# §Ship-9b hotfix (critique critical-2): joint-name validator. Without
# this a key like `hip">\n\n--- ADDITIONAL INSTRUCTIONS ---\n` would
# land verbatim in Claude's context → prompt injection. Restrict to
# the character class MuJoCo itself accepts in name attributes.
_JOINT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")


def _validate_joint_name(name: str) -> str:
    if not _JOINT_NAME_RE.match(name):
        raise ValueError(
            f"joint name {name!r} must match ^[A-Za-z_][A-Za-z0-9_.-]{{0,63}}$"
        )
    return name


class MotorLimitsRequest(BaseModel):
    """§Ship-9b: structured motor specs per joint/actuator. Route
    translates these into a KG-cited NL prompt and delegates to the
    existing Claude-backed physics-editor pipeline."""

    model_config = ConfigDict(extra="forbid")

    motors: Annotated[
        dict[str, MotorSpec],
        Field(
            min_length=1, max_length=64,
            description=(
                "Map from joint name → motor spec. Only joints with at "
                "least one non-null field will be cited in the synthesized "
                "prompt; empty rows are skipped. Joint names must match "
                "^[A-Za-z_][A-Za-z0-9_.-]{0,63}$ (MuJoCo-compatible, "
                "prompt-injection-safe)."
            ),
        ),
    ]

    @field_validator("motors")
    @classmethod
    def _motors_have_safe_keys(cls, v: dict[str, MotorSpec]) -> dict[str, MotorSpec]:
        for name in v.keys():
            _validate_joint_name(str(name))
        return v


# ── GET /projects/{slug}/physics ─────────────────────────────────────
@router.get(
    "/projects/{slug}/physics",
    response_model=PhysicsLoadResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_physics(
    slug: str,
    store: ProjectStore = Depends(get_store),
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    from backend.services.physics import load_mjcf, mjcf_unavailable_reason

    result = load_mjcf(project_dir)
    if result is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "no MJCF configured for this project",
            detail=mjcf_unavailable_reason(project_dir),
            type_="/problems/mjcf-unavailable",
        )

    return PhysicsLoadResponse(
        xml_source=result.xml_source,
        mjcf_source_kind=result.mjcf_source_kind,
        mjcf_path=str(result.mjcf_path),
        materialized=result.materialized,
        summary=result.summary.to_dict(),
    )


# ── POST /projects/{slug}/physics/prompt ─────────────────────────────
@router.post(
    "/projects/{slug}/physics/prompt",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def prompt_physics_edit(
    slug: str,
    body: PhysicsPromptRequest,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Claude rewrites the project's MJCF per the user's NL request.
    Validates via `MjModel.from_xml_string` round-trip; commits as
    `physics: <summary>` on success."""
    import os

    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ANTHROPIC_API_KEY required for physics edits",
            detail=(
                "Set ANTHROPIC_API_KEY in the shell launching the "
                "backend OR in ../RewardSculptor/.env, then restart."
            ),
            type_="/problems/anthropic-key-missing",
        )

    # Mutex with concurrent physics edits on this project + any
    # sculpt run that would re-read the model mid-edit.
    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT,
            "sculpt run in progress",
            detail=(
                "Physics edits are locked while a sculpt run is "
                "active — MJCF swaps mid-training destabilize the env. "
                "Stop the run or wait for it to finish."
            ),
            type_="/problems/state-conflict",
        )
    for existing in jobs.list(project_slug=slug):
        if existing.kind == "physics_prompt_edit" and existing.status in (
            "queued", "running",
        ):
            return _problem(
                status.HTTP_409_CONFLICT,
                "physics edit already active",
                detail=(
                    "Another physics prompt edit is in progress. Wait "
                    "for it to finish before firing another."
                ),
                type_="/problems/job-busy",
                active_job_id=existing.job_id,
            )

    from backend.services.reward_jobs import run_physics_prompt_edit_job

    job = jobs.submit(
        kind="physics_prompt_edit",  # type: ignore[arg-type]
        project_slug=slug,
        fn=run_physics_prompt_edit_job(
            project_dir=project_dir,
            user_prompt=body.prompt,
        ),
        params={"prompt_preview": body.prompt[:200]},
    )
    return job.to_summary()


def _synthesize_motor_limits_prompt(motors: dict[str, MotorSpec]) -> str:
    """§Ship-9b: turn a structured {joint: MotorSpec} map into a
    natural-language physics-editor prompt. Skips joints with no
    non-null fields; cites canonical KG papers so the editor stage's
    reference validator accepts the edit.
    """
    # Keep only joints with at least one concrete value.
    nonempty: dict[str, MotorSpec] = {
        name: spec for name, spec in motors.items()
        if any(
            v is not None for v in (
                spec.peak_torque_nm, spec.peak_speed_rads,
                spec.gear_ratio, spec.rotor_inertia_kgm2,
                spec.peak_current_a,
            )
        )
    }
    if not nonempty:
        raise ValueError("no motor specs provided — every row was empty")

    # Per-motor one-liners describing the fields that ARE set. Anything
    # unset is omitted rather than filled with a default (user didn't
    # specify, so Claude shouldn't fabricate).
    motor_lines: list[str] = []
    for name, spec in sorted(nonempty.items()):
        parts: list[str] = []
        if spec.peak_torque_nm is not None:
            parts.append(f"peak_torque={spec.peak_torque_nm:.3g}Nm")
        if spec.peak_speed_rads is not None:
            parts.append(f"peak_speed={spec.peak_speed_rads:.3g}rad/s")
        if spec.gear_ratio is not None:
            parts.append(f"gear_ratio={spec.gear_ratio:.3g}")
        if spec.rotor_inertia_kgm2 is not None:
            parts.append(f"rotor_inertia={spec.rotor_inertia_kgm2:.3g}kg*m^2")
        if spec.peak_current_a is not None:
            parts.append(f"peak_current={spec.peak_current_a:.3g}A")
        if spec.notes:
            # Strip newlines — TOML-free format but safer for the prompt.
            clean_notes = spec.notes.replace("\n", " ").replace("\r", " ")
            parts.append(f"notes='{clean_notes[:80]}'")
        motor_lines.append(f"  - {name}: {', '.join(parts)}")

    # §Ship-9b hotfix (critique medium-6): cap prompt size. 64 motors
    # × ~120 chars/line + boilerplate tops out ~9 KB. Cap at 8000 chars
    # so Claude always gets a prompt it can digest and we never slip
    # past `apply_mjcf_edit`'s 3-2000 char check via this path (we
    # don't go through that check on the synthesized path, but future
    # refactors may and an explicit cap documents the intent).
    _PROMPT_MAX_CHARS = 8000

    def _finish(body: str) -> str:
        if len(body) > _PROMPT_MAX_CHARS:
            raise ValueError(
                f"synthesized prompt is {len(body)} chars > "
                f"{_PROMPT_MAX_CHARS} ceiling; reduce motor count or "
                f"shorten notes."
            )
        return body

    return _finish(
        "Apply these measured motor specs to the MJCF. Each spec comes "
        "from the user's datasheet or lab characterization.\n\n"
        "Per-joint translation rules (apply ONLY the fields the user "
        "supplied — don't invent values for missing fields):\n"
        "  * peak_torque → set the actuator's forcerange to "
        "[-peak_torque, +peak_torque] (symmetric limits) AND set "
        "actuator_forcelimited='true'.\n"
        "  * rotor_inertia + gear_ratio → set the joint's armature to "
        "rotor_inertia * gear_ratio^2 (reflected inertia on the joint "
        "side, per ANYmal actuator-net 1901.08652).\n"
        "  * peak_speed → add or tighten a joint velocity soft-limit. "
        "For position servos, cap ctrlrange; for general actuators, "
        "add a velocity-proportional damping term to the joint damping.\n"
        "  * peak_current — optional heat / thermal hint; surface in "
        "REWARD_SPEC.description if relevant but don't change MJCF "
        "numerically unless other fields pin a value.\n"
        "  * Joint damping in the absence of measured viscous friction: "
        "use 0.05 * peak_torque / peak_speed as a first-order estimate "
        "(Extended Friction 2410.08650).\n\n"
        "Motor specs:\n"
        + "\n".join(motor_lines) + "\n\n"
        "Cite these papers in REWARD_SPEC.references:\n"
        "  - arXiv:2312.17507 (Actuator-Constrained RL — torque + speed bounds)\n"
        "  - arXiv:1901.08652 (ANYmal actuator-net — gear-reflected inertia)\n"
        "  - arXiv:2410.08650 (Extended Friction — viscous damping model)\n\n"
        "If any joint name in the spec doesn't exist in the current MJCF, "
        "emit `<?rs-summary REJECTED: unknown joints: ... ?>` so the user "
        "can correct the names before retrying. Do NOT silently ignore."
    )


# ── POST /projects/{slug}/physics/motor-limits ────────────────────────


@router.post(
    "/projects/{slug}/physics/motor-limits",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ProblemDetail},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def apply_motor_limits(
    slug: str,
    body: MotorLimitsRequest,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """§Ship-9b: apply structured motor specs to the MJCF.

    Synthesizes an NL prompt from the `{joint: spec}` map + delegates
    to the same Claude-backed physics-edit job the /physics/prompt
    route uses. Response: same `JobSummary` shape — UI polls the job
    for progress + result.
    """
    import os

    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ANTHROPIC_API_KEY required for physics edits",
            detail=(
                "Set ANTHROPIC_API_KEY in the shell launching the "
                "backend OR in ../RewardSculptor/.env, then restart."
            ),
            type_="/problems/anthropic-key-missing",
        )
    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT, "sculpt run in progress",
            detail=(
                "Physics edits are locked while a sculpt run is "
                "active — stop the run or wait for it to finish."
            ),
            type_="/problems/state-conflict",
        )
    for existing in jobs.list(project_slug=slug):
        if existing.kind == "physics_prompt_edit" and existing.status in (
            "queued", "running",
        ):
            return _problem(
                status.HTTP_409_CONFLICT, "physics edit already active",
                detail="Wait for the current physics edit to finish.",
                type_="/problems/job-busy",
                active_job_id=existing.job_id,
            )

    try:
        prompt = _synthesize_motor_limits_prompt(body.motors)
    except ValueError as e:
        return _problem(
            status.HTTP_400_BAD_REQUEST, "empty motor-limits request",
            detail=str(e), type_="/problems/validation-error",
        )

    from backend.services.reward_jobs import run_physics_prompt_edit_job

    job = jobs.submit(
        kind="physics_prompt_edit",  # type: ignore[arg-type]
        project_slug=slug,
        fn=run_physics_prompt_edit_job(
            project_dir=project_dir,
            user_prompt=prompt,
        ),
        params={
            "prompt_preview": prompt[:200],
            # Stash the structured input so the UI can re-show it on
            # the job-detail page.
            "motor_limits": body.model_dump(exclude_none=True),
        },
    )
    return job.to_summary()


# ── POST /projects/{slug}/physics/datasheet-pdf ───────────────────────
_DATASHEET_SYSTEM_PROMPT = (
    "You are a motor-datasheet parser. Extract structured motor specs "
    "from the provided datasheet text. Return STRICT JSON matching this "
    "schema (and nothing else — no prose, no markdown fence):\n"
    "  {\n"
    '    "motors": {\n'
    '      "<joint_name>": {\n'
    '        "peak_torque_nm": <float>,\n'
    '        "peak_speed_rads": <float>,\n'
    '        "gear_ratio": <float>,\n'
    '        "rotor_inertia_kgm2": <float>,\n'
    '        "peak_current_a": <float>,\n'
    '        "notes": "<optional string>"\n'
    "      }\n"
    "    }\n"
    "  }\n\n"
    "Rules:\n"
    "  - Joint names must match ^[A-Za-z_][A-Za-z0-9_.-]{0,63}$. "
    "If the datasheet doesn't name a joint explicitly, fabricate a "
    "reasonable snake_case label (e.g. 'motor_1', 'actuator_a').\n"
    "  - Every numeric field is optional per joint — omit the key "
    "when the datasheet doesn't state that spec.\n"
    "  - Convert units: datasheet may use RPM (→ rad/s), oz-in (→ Nm), "
    "g-cm² (→ kg·m²). Write ONLY the SI-unit fields listed above; "
    "drop non-convertible fields.\n"
    "  - `notes` (one-line, ≤200 chars) may cite the datasheet page/table "
    "so the user can audit the extraction.\n"
    "  - If the datasheet is unreadable, return `{\"motors\": {}}`."
)


# §Ship-9c: no `response_model` because the empty-motors success case
# returns `{"motors": {}}`, which violates the `motors` min_length=1
# on `MotorLimitsRequest`. The shape is still documented in the
# `extract_datasheet_pdf` docstring + the UI's `extractDatasheetPdf`
# typing matches `MotorLimitsRequest`.
@router.post(
    "/projects/{slug}/physics/datasheet-pdf",
    responses={
        200: {"description": "`{motors: {...}}` — same shape as /motor-limits."},
        400: {"model": ProblemDetail},
        404: {"model": ProblemDetail},
        413: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
async def extract_datasheet_pdf(
    slug: str,
    pdf: UploadFile = File(..., description="Motor datasheet PDF"),
    store: ProjectStore = Depends(get_store),
) -> Any:
    """§Ship-9c: extract motor specs from a datasheet PDF.

    Accepts a PDF upload, extracts text via `pypdf`, asks Claude to
    parse structured specs matching `MotorLimitsRequest`. Returns the
    parsed dict for the user to review before applying — the apply
    step is a separate `POST /physics/motor-limits` call, so the user
    can edit any field Claude got wrong.
    """
    import os

    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ANTHROPIC_API_KEY required for datasheet extraction",
            detail="Set ANTHROPIC_API_KEY and restart.",
            type_="/problems/anthropic-key-missing",
        )

    # Bound the upload size. 10 MB covers any reasonable motor
    # datasheet (~100-page multi-language dumps rarely exceed this).
    _MAX_PDF_BYTES = 10 * 1024 * 1024
    data = await pdf.read(_MAX_PDF_BYTES + 1)
    if len(data) > _MAX_PDF_BYTES:
        return _problem(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "datasheet PDF too large",
            detail=f"Received >10MB. Trim the file or split it.",
            type_="/problems/validation-error",
        )
    # §Ship-9c hotfix (critique medium-7): strip any BOM / leading
    # whitespace some valid PDFs carry. Compare on the first 1 KB so
    # a partial download where `%PDF-` is late in the header still
    # rejects rather than hanging pypdf.
    _data_head = data[:1024].lstrip(b"\r\n\t \xef\xbb\xbf")
    if not _data_head.startswith(b"%PDF-"):
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "not a PDF",
            detail=(
                "File magic doesn't match `%PDF-`. Upload a real PDF — "
                "scanned images via PNG / JPG aren't supported."
            ),
            type_="/problems/validation-error",
        )

    # Extract text via pypdf (already a transitive backend dep via
    # sculptor's KG ingest pipeline).
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages[:20]]
        text = "\n\n".join(pages).strip()
    except Exception as e:  # noqa: BLE001
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "PDF parse failed",
            detail=f"pypdf could not read the file: {type(e).__name__}: {e}",
            type_="/problems/validation-error",
        )
    if len(text) < 20:
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "datasheet has no extractable text",
            detail=(
                "PDF is likely a scanned image without OCR. Run OCR "
                "(macOS Preview / Adobe Acrobat / ocrmypdf) first."
            ),
            type_="/problems/validation-error",
        )
    # Cap input to Claude at ~40 KB.
    _MAX_TEXT_CHARS = 40_000
    text_for_claude = text[:_MAX_TEXT_CHARS]

    # Claude-side extraction.
    # §Ship-9c hotfix (critique critical-1+2): bound the call with
    # asyncio.wait_for AND run the sync SDK call inside a thread so
    # the event loop doesn't park for up to the per-request timeout.
    try:
        import anthropic
        client = anthropic.Anthropic(max_retries=2, timeout=60.0)

        def _call_claude() -> str:
            resp = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=4_000,
                system=_DATASHEET_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text_for_claude}],
            )
            return "".join(
                b.text for b in resp.content
                if getattr(b, "type", None) == "text"
            ).strip()

        raw = await asyncio.wait_for(
            asyncio.to_thread(_call_claude), timeout=130.0,
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Claude extraction timed out",
            detail=(
                "Datasheet extraction did not finish within 130 s. "
                "Split the PDF or retry later."
            ),
            type_="/problems/anthropic-error",
        )
    except Exception as e:  # noqa: BLE001
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Claude extraction failed",
            detail=f"{type(e).__name__}: {e}",
            type_="/problems/anthropic-error",
        )

    # Parse the JSON response.
    import json as _json

    # Strip an accidental markdown code fence just in case.
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = _json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "Claude response was not JSON",
            detail=f"Raw response start: {raw[:200]!r}. Parse error: {e}",
            type_="/problems/validation-error",
            claude_output_raw=raw[:2000],
        )
    # §Ship-9c: empty-motors is the documented "nothing extractable"
    # shape (system prompt asks Claude to emit `{"motors": {}}` when
    # the datasheet is unreadable). Short-circuit before pydantic rejects
    # the empty dict for `min_length=1`.
    if isinstance(parsed, dict) and parsed.get("motors") == {}:
        return {"motors": {}}

    # Validate via the same model that /motor-limits accepts — this
    # catches joint-name injection + out-of-range values + shape drift.
    try:
        req = MotorLimitsRequest.model_validate(parsed)
    except Exception as e:  # noqa: BLE001
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "extracted specs fail validation",
            detail=f"{type(e).__name__}: {e}",
            type_="/problems/validation-error",
            claude_output_raw=raw[:2000],
        )
    return req


# ── POST /projects/{slug}/physics/rematerialize ───────────────────────
@router.post(
    "/projects/{slug}/physics/rematerialize",
    response_model=PhysicsLoadResponse,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def rematerialize_physics(
    slug: str,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Wipe `<project>/uploads/robot/` and re-copy from the library
    source, meshes + textures + includes included. Heals broken
    states where the original materialize dropped assets (pre-H1 bug).
    Rejected while a physics prompt edit or sculpt run is active —
    don't race a writer."""
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT,
            "sculpt run in progress",
            detail=(
                "Re-materializing the MJCF swaps model files underneath "
                "an active training run. Stop the run or wait for it to "
                "finish."
            ),
            type_="/problems/state-conflict",
        )
    for existing in jobs.list(project_slug=slug):
        if existing.kind == "physics_prompt_edit" and existing.status in (
            "queued", "running",
        ):
            return _problem(
                status.HTTP_409_CONFLICT,
                "physics edit already active",
                detail=(
                    "Wait for the active prompt edit to finish before "
                    "re-materializing."
                ),
                type_="/problems/job-busy",
                active_job_id=existing.job_id,
            )

    from backend.services.physics import load_mjcf, rematerialize_mjcf

    local = rematerialize_mjcf(project_dir)
    if local is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "no MJCF source to re-materialize from",
            detail=(
                "Re-materialize needs a library-registered MJCF. "
                "Upload-kind projects with a local-only MJCF can't be "
                "re-materialized — re-upload the robot instead."
            ),
            type_="/problems/mjcf-unavailable",
        )
    result = load_mjcf(project_dir)
    if result is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "re-materialize succeeded but load failed",
            detail="unexpected — check backend logs",
            type_="/problems/mjcf-unavailable",
        )
    return PhysicsLoadResponse(
        xml_source=result.xml_source,
        mjcf_source_kind=result.mjcf_source_kind,
        mjcf_path=str(result.mjcf_path),
        materialized=result.materialized,
        summary=result.summary.to_dict(),
    )


# ── PUT /projects/{slug}/physics/fields (placeholder) ─────────────────
@router.put(
    "/projects/{slug}/physics/fields",
    responses={
        404: {"model": ProblemDetail},
        501: {"model": ProblemDetail},
    },
)
def put_physics_fields(
    slug: str,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Form-driven direct edits (per-joint damping sliders etc.) are
    deferred to a follow-up pass. Endpoint stubbed so the UI surface
    can be built against a stable contract."""
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            type_="/problems/not-found",
        )
    return _problem(
        status.HTTP_501_NOT_IMPLEMENTED,
        "form-driven physics edits not yet implemented",
        detail=(
            "M7 Phase 5 ships prompt-driven editing only. Form-based "
            "editing (per-joint sliders etc.) is a follow-up. Use "
            "POST /projects/{slug}/physics/prompt for now."
        ),
        type_="/problems/not-implemented",
    )

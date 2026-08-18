"""Project-scoped starting-skill bundle admission and listing."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from backend.models.project import ProblemDetail
from backend.services.artifact_lineage import record_admitted_starting_skill
from backend.services.project_store import ProjectStore
from sculptor.skill_bundle import (
    ImportTarget,
    MAX_ARCHIVE_BYTES,
    SkillBundleError,
    StartingSkillBundleImporter,
    receipt_for,
)
from sculptor.skill_library import SkillLibrary


router = APIRouter(tags=["starting-skills"])


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def _problem(
    status_code: int,
    title: str,
    *,
    detail: str,
    type_: str,
    **extra: Any,
) -> JSONResponse:
    body = ProblemDetail(
        type=type_, title=title, status=status_code, detail=detail,
    ).model_dump()
    body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


def _target_for(detail: Any) -> ImportTarget:
    cfg = detail.adapter_config or {}
    task_id = cfg.get("task_id") or cfg.get("env_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise SkillBundleError(
            "this project does not declare an adapter task_id, so policy "
            "compatibility cannot be established",
            code="project_contract_missing",
        )
    contract = None
    contract_error = None
    try:
        from sculptor.policy_contract import build_project_policy_contract

        contract = build_project_policy_contract(Path(detail.project_dir))
    except Exception as exc:
        # A reference-only seed has no policy tensors to compare and remains a
        # valid starting point even when this project's policy contract cannot
        # currently be constructed.  Preserve the identity target and attach a
        # precise per-policy denial instead of making the whole picker fail.
        contract_error = (
            "project_contract_missing: could not establish the project's "
            "full policy interface contract: "
            f"{type(exc).__name__}: {exc}"
        )
    project_robot = (
        str(detail.reference_robot).strip()
        if isinstance(detail.reference_robot, str)
        and detail.reference_robot.strip()
        else None
    )
    robot_contract_error = None
    if project_robot is None:
        robot_contract_error = (
            "project_robot_unresolved: select a project robot with an exact "
            "reference namespace before using an imported starting point"
        )
    return ImportTarget(
        adapter_class=str(detail.adapter_class),
        task_id=task_id,
        robot_slug=project_robot,
        compatibility_contract=contract,
        policy_contract_error=contract_error,
        robot_contract_error=robot_contract_error,
    )


@router.get(
    "/projects/{slug}/starting-skills",
    responses={404: {"model": ProblemDetail}, 412: {"model": ProblemDetail}},
)
def list_starting_skills(
    slug: str,
    store: ProjectStore = Depends(get_store),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    try:
        target = _target_for(detail)
    except SkillBundleError as exc:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "project training contract is incomplete",
            detail=str(exc),
            type_="/problems/starting-skill-project-contract",
        )

    rows = [receipt_for(record, target) for record in SkillLibrary()]
    rows.sort(
        key=lambda row: (
            not bool(row["compatible"]),
            str(row["skill"].get("created_at") or ""),
        )
    )
    return {"skills": rows}


@router.post(
    "/projects/{slug}/starting-skills",
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ProblemDetail},
        404: {"model": ProblemDetail},
        412: {"model": ProblemDetail},
        413: {"model": ProblemDetail},
    },
)
async def import_starting_skill(
    slug: str,
    bundle: UploadFile = File(...),
    store: ProjectStore = Depends(get_store),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    try:
        target = _target_for(detail)
    except SkillBundleError as exc:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "project training contract is incomplete",
            detail=str(exc),
            type_="/problems/starting-skill-project-contract",
        )

    upload_name = Path(bundle.filename or "").name
    if not upload_name.lower().endswith(".rskill"):
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "starting-skill bundle was rejected",
            detail=(
                "upload a data-only .rskill artifact; deployment .zip "
                "bundles and raw checkpoints are not portable imports"
            ),
            type_="/problems/starting-skill-invalid_bundle",
            code="invalid_bundle",
        )

    quarantine_root = Path(detail.project_dir).parent / ".import-quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    quarantine_dir = Path(tempfile.mkdtemp(prefix="starting-skill-", dir=quarantine_root))
    upload_path = quarantine_dir / "bundle.rskill"
    total = 0
    try:
        with upload_path.open("xb") as output:
            while True:
                chunk = await bundle.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    return _problem(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        "starting-skill bundle is too large",
                        detail=f"uploads are limited to {MAX_ARCHIVE_BYTES} bytes",
                        type_="/problems/starting-skill-too-large",
                    )
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total == 0:
            return _problem(
                status.HTTP_400_BAD_REQUEST,
                "starting-skill bundle is empty",
                detail="upload a non-empty data-only .rskill artifact",
                type_="/problems/starting-skill-invalid",
            )
        importer = StartingSkillBundleImporter()
        project_dir = Path(detail.project_dir)

        def _publish_lineage(imported) -> None:
            record_admitted_starting_skill(
                project_dir,
                record=imported.record,
                receipt=imported.receipt,
                target=target,
            )

        imported = await asyncio.to_thread(
            importer.import_archive,
            upload_path,
            target=target,
            admission_callback=_publish_lineage,
        )
        return imported.receipt
    except SkillBundleError as exc:
        error_status = {
            "project_contract_missing": status.HTTP_412_PRECONDITION_FAILED,
            "lineage_publication_failed": (
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
        }.get(exc.code, status.HTTP_400_BAD_REQUEST)
        return _problem(
            error_status,
            "starting-skill bundle was rejected",
            detail=str(exc),
            type_=f"/problems/starting-skill-{exc.code}",
            code=exc.code,
        )
    finally:
        await bundle.close()
        shutil.rmtree(quarantine_dir, ignore_errors=True)

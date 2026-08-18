"""Reference-motion preparation for ordinary ``sculpt run`` launches.

Mission stages already had a tracking-first path.  This module exposes the
same invariant to the normal project run without coupling it to mission/task
names:

* resolve one exact ``(reference_robot, reference_clip_id)`` pair;
* deterministically build the immutable phase-indexed motion prior;
* let the behavior prompt author only the bounded task residual;
* verify that editing did not replace the reference target.

World curricula (including route-aware RSI) remain separate.  That lets a
single run combine "start at different points on this physical course" with
"retain the gait/style/pose sequence in this clip" instead of using a motion
clip as a substitute for physical scene interaction.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sculptor.adapters.base import _import_reward_module
from sculptor.edit import apply_prompt_edit
from sculptor.reference import load_clip
from sculptor.refs import library
from sculptor.refs.track import generate_tracking_residual_reward_source


class ReferenceRunError(RuntimeError):
    """The selected reference cannot safely seed an ordinary run."""


@dataclass(frozen=True)
class ReferenceRewardBuild:
    source: str
    clip_id: str
    robot: str
    clip_sha256: str
    target_sha256: str
    phase_mode: str
    phase_duration_s: float
    task_residual_authored: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact_reference_motion(
    *, clip_id: str, robot: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load and verify an exact reference-library pair.

    No fallback robot, fuzzy clip match, or task-name inference is allowed at
    this boundary.  Cross-embodiment motions must first be retargeted and
    registered into the target robot's namespace, preserving the retarget
    provenance the library already records.
    """
    clip_path = library.clip_dir(robot, clip_id) / library.CLIP_FILENAME
    provenance_path = (
        library.clip_dir(robot, clip_id) / library.PROVENANCE_FILENAME
    )
    if not clip_path.is_file() or not provenance_path.is_file():
        raise ReferenceRunError(
            f"reference motion {robot}/{clip_id} is incomplete or missing"
        )
    provenance = library.read_provenance(robot, clip_id)
    if provenance.get("clip_id") != clip_id or provenance.get("robot") != robot:
        raise ReferenceRunError(
            "reference provenance does not match the selected "
            f"pair {robot}/{clip_id}"
        )
    errors = library.validate_provenance(provenance)
    if errors:
        raise ReferenceRunError(
            "reference provenance is invalid: " + "; ".join(errors)
        )
    return load_clip(clip_path), provenance, _sha256_file(clip_path)


def build_reference_guided_reward(
    *,
    clip_id: str,
    robot: str,
    behavior_goal: str,
    reward_version: str,
    reward_contract: Any,
    kg_store: Any = None,
    dry_run: bool = False,
) -> ReferenceRewardBuild:
    """Build a tracking-first reward fully before touching project state.

    Live runs pass the deterministic base through the normal guarded reward
    editor so the behavior prompt becomes the bounded residual task.  Dry runs
    retain the zero-residual base: they validate clip loading and the runtime
    reward shape without spending an LLM call.
    """
    clip, _provenance, clip_sha256 = load_exact_reference_motion(
        clip_id=clip_id, robot=robot,
    )
    base_source = generate_tracking_residual_reward_source(
        clip=clip,
        clip_id=clip_id,
        robot=robot,
        version=reward_version,
    )

    with tempfile.TemporaryDirectory(prefix="rs_reference_run_") as raw_tmp:
        staging = Path(raw_tmp)
        base_path = staging / "v0.py"
        base_path.write_text(base_source, encoding="utf-8")
        base_module = _import_reward_module(base_path)
        source = base_source
        authored = False

        if not dry_run:
            edited_path = apply_prompt_edit(
                current_reward_path=base_path,
                user_prompt=behavior_goal,
                new_iter_id=reward_version,
                reward_contract=reward_contract,
                kg_store=kg_store,
            )
            source = edited_path.read_text(encoding="utf-8")
            authored = True

        final_path = staging / "_verified.py"
        final_path.write_text(source, encoding="utf-8")
        final_module = _import_reward_module(final_path)

        base_target = str(
            getattr(base_module, "REFERENCE_TARGET_SHA256", "") or ""
        )
        final_target = str(
            getattr(final_module, "REFERENCE_TARGET_SHA256", "") or ""
        )
        composition = (
            getattr(final_module, "REWARD_SPEC", {}).get("composition") or {}
        )
        if (
            not base_target
            or final_target != base_target
            or composition.get("type") != "reference_tracking_residual"
            or composition.get("reference_clip_id") != clip_id
            or composition.get("reference_target_sha256") != base_target
        ):
            raise ReferenceRunError(
                "reward editing changed or removed the immutable "
                "reference-motion target"
            )

        return ReferenceRewardBuild(
            source=source,
            clip_id=clip_id,
            robot=robot,
            clip_sha256=clip_sha256,
            target_sha256=base_target,
            phase_mode=str(composition.get("phase_mode") or "hold"),
            phase_duration_s=float(
                composition.get("phase_duration_s") or 0.0
            ),
            task_residual_authored=authored,
        )


def reference_input_hash(
    *, clip_id: str, robot: str, clip_sha256: str, behavior_goal: str,
) -> str:
    """Stable identity for idempotent UI resume of the same motion+goal."""
    payload = {
        "clip_id": clip_id,
        "robot": robot,
        "clip_sha256": clip_sha256,
        "behavior_goal": behavior_goal.strip(),
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

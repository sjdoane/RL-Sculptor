"""Fail-closed authority for the reference embedded in the active reward.

The motion picker is not the only way a reference reaches training.  A
promoted reward can contain immutable reference arrays and ``current.py`` can
continue selecting that reward after the picker is cleared.  Launch admission
therefore derives reference authority from the exact reward that will execute,
never from UI state alone.

This module is intentionally data-only: it parses ``REWARD_SPEC`` with the AST
reader, hashes files, and never imports reward code.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from sculptor.mode_rewards import reward_spec_from_source


class ActiveReferenceAuthorityError(ValueError):
    """The active reward mentions a reference but cannot attest it exactly."""


_CURRENT_TARGET_RE = re.compile(r"/\s*(['\"])(v(\d+)\.py)\1")
_VERSION_RE = re.compile(r"v(\d+)\.py")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_reward_path(rewards_dir: Path) -> tuple[Path, Optional[Path]] | None:
    """Mirror runtime reward selection without importing executable code.

    ``sculpt_run`` materializes ``current.py`` from the highest numbered
    immutable reward only when the selector is absent.  If the selector exists
    but is malformed, runtime behavior is no longer provable and launch must
    stop instead of guessing.
    """
    rewards_dir = Path(rewards_dir).resolve()
    current = rewards_dir / "current.py"
    if current.exists():
        if not current.is_file() or current.is_symlink():
            raise ActiveReferenceAuthorityError(
                "rewards/current.py must be a regular selector file"
            )
        try:
            selector_source = current.read_text(encoding="utf-8")
        except OSError as exc:
            raise ActiveReferenceAuthorityError(
                f"cannot read rewards/current.py: {exc}"
            ) from exc
        matches = list(_CURRENT_TARGET_RE.finditer(selector_source))
        targets = {match.group(2) for match in matches}
        if len(targets) != 1:
            raise ActiveReferenceAuthorityError(
                "rewards/current.py does not select one exact immutable "
                "v<n>.py reward"
            )
        target = (rewards_dir / next(iter(targets))).resolve()
        if target.parent != rewards_dir or not target.is_file():
            raise ActiveReferenceAuthorityError(
                "rewards/current.py selects a missing or out-of-tree reward"
            )
        return target, current.resolve()

    versions: list[tuple[int, Path]] = []
    if rewards_dir.is_dir():
        for candidate in rewards_dir.iterdir():
            match = _VERSION_RE.fullmatch(candidate.name)
            if match is not None and candidate.is_file():
                versions.append((int(match.group(1)), candidate.resolve()))
    if not versions:
        return None
    return max(versions, key=lambda item: item[0])[1], None


def _nonempty_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _one_exact_claim(label: str, claims: list[str]) -> Optional[str]:
    unique = set(claims)
    if len(unique) > 1:
        raise ActiveReferenceAuthorityError(
            f"active reward has conflicting {label} claims: {sorted(unique)}"
        )
    return next(iter(unique)) if unique else None


@dataclass(frozen=True)
class ActiveReferenceAuthority:
    """Immutable receipt for a reference-bearing active reward."""

    schema: int
    kind: str
    reference_clip_id: str
    reference_robot: str
    reward_path: str
    reward_sha256: str
    selector_path: Optional[str]
    selector_sha256: Optional[str]
    reference_clip_sha256: Optional[str]
    reference_target_sha256: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_active_reference_authority(
    rewards_dir: Path,
) -> Optional[ActiveReferenceAuthority]:
    """Return the reference embedded in the reward runtime will execute.

    A non-reference reward returns ``None``.  Once any reference claim is
    present, the robot/clip pair and relevant content hashes become mandatory;
    legacy generated rewards without a robot identity are deliberately blocked
    and must be regenerated or promoted with a complete binding.
    """
    resolved = _active_reward_path(rewards_dir)
    if resolved is None:
        return None
    reward_path, selector_path = resolved
    try:
        source = reward_path.read_text(encoding="utf-8")
        spec = reward_spec_from_source(source)
    except (OSError, ValueError, TypeError) as exc:
        raise ActiveReferenceAuthorityError(
            f"cannot parse active reward REWARD_SPEC: {exc}"
        ) from exc
    if not isinstance(spec, Mapping):
        raise ActiveReferenceAuthorityError(
            "active reward REWARD_SPEC must be an object"
        )

    composition_raw = spec.get("composition")
    composition = (
        composition_raw if isinstance(composition_raw, Mapping) else {}
    )
    binding_raw = spec.get("mode_binding")
    binding = binding_raw if isinstance(binding_raw, Mapping) else {}

    clip_claims = [
        value
        for value in (
            _nonempty_string(spec.get("reference_clip_id")),
            _nonempty_string(composition.get("reference_clip_id")),
            _nonempty_string(binding.get("clip_id")),
        )
        if value is not None
    ]
    reference_clip_id = _one_exact_claim("reference clip", clip_claims)
    is_tracking = (
        composition.get("type") == "reference_tracking_residual"
        or spec.get("tracking_enabled") is True
        or bool(binding)
    )
    if reference_clip_id is None and not is_tracking:
        return None
    if reference_clip_id is None:
        raise ActiveReferenceAuthorityError(
            "active reference reward does not identify its exact clip"
        )

    robot_claims = [
        value
        for value in (
            _nonempty_string(spec.get("reference_robot")),
            _nonempty_string(composition.get("reference_robot")),
            _nonempty_string(binding.get("robot")),
        )
        if value is not None
    ]
    reference_robot = _one_exact_claim("reference robot", robot_claims)
    if reference_robot is None:
        raise ActiveReferenceAuthorityError(
            "active reference reward has no immutable reference_robot; "
            "regenerate or re-promote it before training"
        )

    clip_sha = _nonempty_string(binding.get("clip_sha256"))
    target_sha = _nonempty_string(
        composition.get("reference_target_sha256")
    )
    for label, digest in (
        ("reference clip", clip_sha),
        ("reference target", target_sha),
    ):
        if digest is not None and _SHA256_RE.fullmatch(digest) is None:
            raise ActiveReferenceAuthorityError(
                f"active reward {label} SHA-256 is malformed"
            )
    kind = "mode_reference" if binding else "tracking_reference"
    if kind == "mode_reference" and clip_sha is None:
        raise ActiveReferenceAuthorityError(
            "active mode reward binding has no exact reference clip SHA-256"
        )
    if kind == "tracking_reference" and target_sha is None:
        raise ActiveReferenceAuthorityError(
            "active tracking reward has no immutable reference target SHA-256"
        )

    return ActiveReferenceAuthority(
        schema=1,
        kind=kind,
        reference_clip_id=reference_clip_id,
        reference_robot=reference_robot,
        reward_path=str(reward_path),
        reward_sha256=_sha256_file(reward_path),
        selector_path=(str(selector_path) if selector_path else None),
        selector_sha256=(
            _sha256_file(selector_path) if selector_path else None
        ),
        reference_clip_sha256=clip_sha,
        reference_target_sha256=target_sha,
    )


def require_active_reference_receipt(
    rewards_dir: Path,
    expected: Mapping[str, Any],
) -> ActiveReferenceAuthority:
    """Re-attest an admission receipt at the last responsible moment."""
    actual = resolve_active_reference_authority(rewards_dir)
    if actual is None or actual.to_dict() != dict(expected):
        raise ActiveReferenceAuthorityError(
            "active reference reward changed after launch admission"
        )
    return actual

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sculptor.reference_authority import (
    ActiveReferenceAuthorityError,
    require_active_reference_receipt,
    resolve_active_reference_authority,
)


def _write_current(rewards: Path, target: str) -> None:
    (rewards / "current.py").write_text(
        "from pathlib import Path\n"
        "_HERE = Path(__file__).resolve().parent\n"
        f"_LATEST = _HERE / {target!r}\n",
        encoding="utf-8",
    )


def _tracking_source(*, robot: str | None = "g1") -> str:
    robot_line = "" if robot is None else f"'reference_robot': {robot!r},"
    composition_robot = (
        "" if robot is None else f"'reference_robot': {robot!r},"
    )
    return f"""REWARD_SPEC = {{
        'reference_tracking': True,
        {robot_line}
        'composition': {{
            'type': 'reference_tracking_residual',
            'reference_clip_id': 'parkour',
            {composition_robot}
            'reference_target_sha256': {'a' * 64!r},
        }},
    }}
"""


def test_active_tracking_reward_is_authority_even_without_picker_state(
    tmp_path: Path,
) -> None:
    rewards = tmp_path / "rewards"
    rewards.mkdir()
    reward = rewards / "v4.py"
    reward.write_text(_tracking_source(), encoding="utf-8")
    _write_current(rewards, "v4.py")

    authority = resolve_active_reference_authority(rewards)

    assert authority is not None
    assert authority.reference_clip_id == "parkour"
    assert authority.reference_robot == "g1"
    assert authority.kind == "tracking_reference"
    assert authority.reward_sha256 == hashlib.sha256(
        reward.read_bytes()
    ).hexdigest()
    assert authority.selector_sha256 is not None


def test_reference_reward_without_exact_robot_is_blocked(tmp_path: Path) -> None:
    rewards = tmp_path / "rewards"
    rewards.mkdir()
    (rewards / "v0.py").write_text(
        _tracking_source(robot=None), encoding="utf-8",
    )
    _write_current(rewards, "v0.py")

    with pytest.raises(ActiveReferenceAuthorityError, match="reference_robot"):
        resolve_active_reference_authority(rewards)


def test_mode_binding_conflict_is_blocked(tmp_path: Path) -> None:
    rewards = tmp_path / "rewards"
    rewards.mkdir()
    (rewards / "v2.py").write_text(
        f"REWARD_SPEC = {{"
        "'reference_clip_id': 'jump',"
        "'reference_robot': 'g1',"
        "'mode_binding': {"
        f"'clip_id': 'flip', 'robot': 'g1', "
        f"'clip_sha256': {'b' * 64!r}"
        "}}\n",
        encoding="utf-8",
    )
    _write_current(rewards, "v2.py")

    with pytest.raises(ActiveReferenceAuthorityError, match="conflicting"):
        resolve_active_reference_authority(rewards)


def test_worker_receipt_detects_reward_or_selector_drift(tmp_path: Path) -> None:
    rewards = tmp_path / "rewards"
    rewards.mkdir()
    (rewards / "v1.py").write_text(_tracking_source(), encoding="utf-8")
    _write_current(rewards, "v1.py")
    admitted = resolve_active_reference_authority(rewards)
    assert admitted is not None

    (rewards / "v1.py").write_text(
        _tracking_source().replace("'parkour'", "'parkour-v2'"),
        encoding="utf-8",
    )

    with pytest.raises(
        ActiveReferenceAuthorityError, match="changed after launch admission",
    ):
        require_active_reference_receipt(rewards, admitted.to_dict())


def test_missing_current_mirrors_runtime_latest_version(tmp_path: Path) -> None:
    rewards = tmp_path / "rewards"
    rewards.mkdir()
    (rewards / "v2.py").write_text("REWARD_SPEC = {}\n", encoding="utf-8")
    (rewards / "v9.py").write_text(_tracking_source(), encoding="utf-8")

    authority = resolve_active_reference_authority(rewards)

    assert authority is not None
    assert Path(authority.reward_path).name == "v9.py"
    assert authority.selector_path is None

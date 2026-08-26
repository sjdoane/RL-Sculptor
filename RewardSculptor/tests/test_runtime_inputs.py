from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from sculptor.runtime_inputs import (
    ENVIRONMENT_ARTIFACT_SCHEMA,
    REWARD_MODULE_ARTIFACT_SCHEMA,
    REWARD_SELECTOR_SCHEMA,
    capture_environment_artifacts,
    capture_reward_module_artifact,
    environment_artifacts_for_phase,
    validate_environment_artifacts,
)
from sculptor.world.artifacts import canonical_json_bytes, sha256_bytes


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_reward_selector_receipt_binds_loader_and_selected_bytes(
    tmp_path: Path,
) -> None:
    from sculptor.edit import _write_current_reexport

    rewards = tmp_path / "rewards"
    rewards.mkdir()
    selected = rewards / "v7.py"
    selected.write_text(
        "REWARD_SPEC = {}\n"
        "def compute_reward(state, action, next_state, info): return 0.0, {}\n",
        encoding="utf-8",
    )
    _write_current_reexport(rewards, selected)

    receipt = capture_reward_module_artifact(rewards / "current.py")

    assert receipt["schema"] == REWARD_MODULE_ARTIFACT_SCHEMA
    assert receipt["selection_kind"] == "selector"
    assert receipt["loader"]["path"] == str((rewards / "current.py").resolve())
    assert receipt["selected"]["path"] == str(selected.resolve())
    assert receipt["selected"]["sha256"] == _sha(selected.read_bytes())
    assert receipt["loader"]["sha256"] == _sha(
        (rewards / "current.py").read_bytes()
    )
    assert receipt["loader"]["sha256"] != receipt["selected"]["sha256"]
    selector = ast.literal_eval(
        next(
            node.value
            for node in ast.parse(
                (rewards / "current.py").read_text(encoding="utf-8")
            ).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "SCULPTOR_REWARD_SELECTOR"
                for target in node.targets
            )
        )
    )
    assert selector == {
        "schema": REWARD_SELECTOR_SCHEMA,
        "filename": "v7.py",
        "sha256": receipt["selected"]["sha256"],
    }


def test_reward_selector_receipt_rejects_selected_bytes_changed_after_write(
    tmp_path: Path,
) -> None:
    from sculptor.edit import _write_current_reexport

    rewards = tmp_path / "rewards"
    rewards.mkdir()
    selected = rewards / "v0.py"
    selected.write_text(
        "REWARD_SPEC = {}\n"
        "def compute_reward(state, action, next_state, info): return 0.0, {}\n",
        encoding="utf-8",
    )
    _write_current_reexport(rewards, selected)
    selected.write_text("REWARD_SPEC = {'tampered': True}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable digest"):
        capture_reward_module_artifact(rewards / "current.py")


def test_direct_reward_receipt_uses_same_loader_and_selected_identity(
    tmp_path: Path,
) -> None:
    reward = tmp_path / "tier_d_reward.py"
    reward.write_text(
        "REWARD_SPEC = {}\n"
        "def compute_reward(state, action, next_state, info): return 0.0, {}\n",
        encoding="utf-8",
    )

    receipt = capture_reward_module_artifact(reward)

    assert receipt["selection_kind"] == "direct"
    assert receipt["loader"] == receipt["selected"]
    assert receipt["selected"]["sha256"] == _sha(reward.read_bytes())


def _write_selection(
    project: Path,
    *,
    component_path: str = "env/task_v1.json",
    component_sha256: str,
) -> Path:
    selection = project / "env" / "selection_current.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    refs = {
        "task": {
            "kind": "task",
            "version": "v1",
            "path": component_path,
            "sha256": component_sha256,
        }
    }
    selection.write_text(
        json.dumps(
            {
                "tuple_hash": sha256_bytes(canonical_json_bytes(refs)),
                "refs": refs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return selection


def test_environment_receipt_rehashes_world_selection_components(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    component = project / "env" / "task_v1.json"
    component.parent.mkdir(parents=True)
    component_bytes = b'{"task":"weave"}\n'
    component.write_bytes(component_bytes)
    selection = _write_selection(
        project, component_sha256=_sha(component_bytes),
    )
    env_spec = project / "env" / "current.json"
    env_spec.write_text('{"env_spec_version":1}\n', encoding="utf-8")

    receipt = capture_environment_artifacts(
        env_spec_path=env_spec,
        world_selection_path=selection,
    )

    assert receipt["schema"] == ENVIRONMENT_ARTIFACT_SCHEMA
    assert receipt["world_selection"]["refs"]["task"]["sha256"] == _sha(
        component_bytes
    )
    assert validate_environment_artifacts(receipt) == []
    train = environment_artifacts_for_phase(receipt, "train")
    assert set(train) == {"schema", "env_spec", "world_selection"}


def test_environment_receipt_rejects_component_mutated_after_selection(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    component = project / "env" / "task_v1.json"
    component.parent.mkdir(parents=True)
    original = b'{"task":"weave"}\n'
    component.write_bytes(original)
    selection = _write_selection(project, component_sha256=_sha(original))
    component.write_bytes(b'{"task":"mutated"}\n')

    with pytest.raises(ValueError, match="sha256 mismatch"):
        capture_environment_artifacts(world_selection_path=selection)


def test_world_selection_receipt_parses_the_same_byte_snapshot_it_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    component = project / "env" / "task_v1.json"
    component.parent.mkdir(parents=True)
    component_bytes = b'{"task":"weave"}\n'
    component.write_bytes(component_bytes)
    selection = _write_selection(
        project, component_sha256=_sha(component_bytes),
    )
    snapshot_a = selection.read_bytes()
    payload_b = json.loads(snapshot_a.decode("utf-8"))
    payload_b["refs"]["task"]["version"] = "v2"
    payload_b["tuple_hash"] = sha256_bytes(
        canonical_json_bytes(payload_b["refs"])
    )
    selection.write_text(json.dumps(payload_b, sort_keys=True), encoding="utf-8")
    original_read_bytes = Path.read_bytes
    selection_reads = 0

    def read_snapshot(path: Path) -> bytes:
        nonlocal selection_reads
        if path.resolve() == selection.resolve():
            selection_reads += 1
            if selection_reads == 1:
                return snapshot_a
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_snapshot)
    receipt = capture_environment_artifacts(
        world_selection_path=selection,
    )["world_selection"]

    assert selection_reads == 1
    assert receipt["sha256"] == _sha(snapshot_a)
    assert receipt["tuple_hash"] == json.loads(
        snapshot_a.decode("utf-8")
    )["tuple_hash"]


def test_environment_receipt_rejects_forged_world_tuple_hash(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    component = project / "env" / "task_v1.json"
    component.parent.mkdir(parents=True)
    component_bytes = b'{"task":"weave"}\n'
    component.write_bytes(component_bytes)
    selection = _write_selection(
        project, component_sha256=_sha(component_bytes),
    )
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["tuple_hash"] = "a" * 64
    selection.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="tuple_hash mismatch"):
        capture_environment_artifacts(world_selection_path=selection)


def test_environment_receipt_validator_rejects_forged_world_tuple_hash(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    component = project / "env" / "task_v1.json"
    component.parent.mkdir(parents=True)
    component_bytes = b'{"task":"weave"}\n'
    component.write_bytes(component_bytes)
    selection = _write_selection(
        project, component_sha256=_sha(component_bytes),
    )
    receipt = capture_environment_artifacts(
        world_selection_path=selection,
    )
    receipt["world_selection"]["tuple_hash"] = "0" * 64

    assert "world_selection.tuple_hash does not match refs" in (
        validate_environment_artifacts(receipt)
    )


@pytest.mark.parametrize("declared_path", ["../outside.json", "C:/outside.json"])
def test_environment_receipt_rejects_escaping_or_absolute_component_path(
    tmp_path: Path,
    declared_path: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside\n")
    selection = _write_selection(
        project,
        component_path=declared_path,
        component_sha256=_sha(outside.read_bytes()),
    )

    with pytest.raises(ValueError, match="project-relative|escapes"):
        capture_environment_artifacts(world_selection_path=selection)

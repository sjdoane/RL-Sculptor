from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.world.gates import run_admission_gates
from sculptor.world.project import (
    WorldProjectService,
    WorldPromotionError,
    load_selected_world,
)
from tests.test_world_foundation import _task, _world


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    rewards = project / "rewards"
    rewards.mkdir(parents=True)
    (rewards / "v0.py").write_text("REWARD_SPEC = {}\n", encoding="utf-8")
    (rewards / "current.py").write_text(
        "from pathlib import Path\n"
        "_HERE = Path(__file__).resolve().parent\n"
        "_LATEST = _HERE / 'v0.py'\n",
        encoding="utf-8",
    )
    return project


def _admissible_world_task() -> tuple[dict, dict]:
    world, task = _world(), _task()
    world["shared"]["obstacles"]["course"] = []
    world["train"]["variations"] = []
    world["shared"]["objects"]["ball"]["nominal"].update({
        "radius_m": 0.08,
        "pose": {"position_m": [0.35, 0.0, 0.8]},
    })
    return world, task


def test_promote_creates_one_verified_atomic_tuple(tmp_path: Path) -> None:
    project = _project(tmp_path)
    world, task = _admissible_world_task()
    asset_dir = project / "env" / "direct_eval_assets"
    report, compiled = run_admission_gates(
        world, task, materialize_dir=asset_dir, settle_steps=20)
    assert report.ok and compiled is not None
    catalog = compiled.channel_catalog
    resolved = compiled.resolved_eval.to_dict()

    promoted = WorldProjectService(project).promote(
        world=world, task=task, resolved_eval=resolved,
        channel_catalog=catalog,
        clarifications={"questions": [], "answers": []},
        evaluation_lineage="eval-test",
    )

    pinned = project / "env" / (
        f"selection_v{promoted.selection.selection_version}.json")
    _, selected, bundle = load_selected_world(pinned)
    assert selected.tuple_hash == promoted.selection.tuple_hash
    assert bundle["world"] == world
    assert bundle["task"] == task
    assert bundle["channel_catalog"]["catalog_hash"] == catalog.catalog_hash
    assert bundle["env_spec"]["shared"] == {}
    assert Path(bundle["reward_path"]).name == "v0.py"


def test_failed_admission_cannot_change_selection(tmp_path: Path) -> None:
    project = _project(tmp_path)
    world, task = _admissible_world_task()
    report, compiled = run_admission_gates(
        world, task,
        materialize_dir=project / "env" / "rejected_eval_assets",
        settle_steps=20,
    )
    assert report.ok and compiled is not None
    rejected = compiled.resolved_eval.with_admission({
        "ok": False, "status": "rejected", "violations": [],
    })

    with pytest.raises(WorldPromotionError, match="not passed admission"):
        WorldProjectService(project).promote(
            world=world, task=task,
            resolved_eval=rejected.to_dict(),
            channel_catalog=compiled.channel_catalog, clarifications={},
            evaluation_lineage="eval-rejected",
        )
    assert not (project / "env" / "selection_current.json").exists()


def test_sculpt_iteration_rebinds_reward_without_changing_world(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from sculptor.sculpt import (
        _pin_authored_selection,
        _promote_iteration_selection,
    )

    project = _project(tmp_path)
    world, task = _admissible_world_task()
    initial_admission = WorldProjectService(project).admit_and_promote(
        world=world, task=task,
        clarifications={},
        evaluation_lineage="eval-fixed",
    )
    initial = initial_admission.promoted
    adapter = SimpleNamespace(
        world_selection_path=str(project / "env" / "selection_current.json"))
    assert _pin_authored_selection(adapter, project) == (
        initial.selection.tuple_hash)

    reward_v1 = project / "rewards" / "v1.py"
    reward_v1.write_text("REWARD_SPEC = {'version': 'v1'}\n", encoding="utf-8")
    tuple_hash, path = _promote_iteration_selection(
        adapter, project, reward_path=reward_v1,
        env_spec_version=initial.selection.refs["env_spec"].version)

    assert tuple_hash and path and path.is_file()
    _, selection, bundle = load_selected_world(path)
    assert Path(bundle["reward_path"]).name == "v1.py"
    assert bundle["world"] == world
    assert selection.evaluation_lineage == "eval-fixed"


def test_admission_materializes_then_atomically_promotes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    world, task = _admissible_world_task()

    admitted = WorldProjectService(project).admit_and_promote(
        world=world, task=task,
        clarifications={"answers": [], "source": "test"},
        evaluation_lineage="eval-admitted",
    )

    assert admitted.admission["ok"] is True
    assert (admitted.asset_dir / "evaluation_scene.mjb").is_file()
    assert (project / "env" / "selection_current.json").is_file()
    _, selection, bundle = load_selected_world(
        project / "env" /
        f"selection_v{admitted.promoted.selection.selection_version}.json")
    assert selection.evaluation_lineage == "eval-admitted"
    assert bundle["resolved_eval"]["admission"]["ok"] is True

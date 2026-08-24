"""§env-authoring §10.2: diagnoser world-edit surface + tuple revert.

Covers: the # WORLD_VARIATIONS render block; diagnose() packing gated on
an authored selection with registered-ID membership; hallucinated edits
dropped without a surface; and `_promote_iteration_selection`'s
``base_selection`` — the world-half keep/revert primitive that restores
the five immutable refs from a prior pinned selection as one tuple.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sculptor.diagnose import (
    _GroundedModel,
    _PreliminaryModel,
    _ProposedWorldEditModel,
    _render_world_variations_block,
    diagnose,
)

_TESTS_DIR = str(Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


def test_render_world_variations_block_contents() -> None:
    block = _render_world_variations_block([
        {"id": "ball_mass",
         "target": "/shared/objects/ball/nominal/mass_kg",
         "class": "model_field",
         "distribution": {"kind": "uniform", "low": 0.15, "high": 0.25}},
    ])
    assert "# WORLD_VARIATIONS" in block
    assert "ball_mass" in block and "model_field" in block
    assert "byte-verified" in block  # the frozen-eval instruction
    assert _render_world_variations_block(None) == ""
    assert _render_world_variations_block([]) == ""


def _diagnose_fixture_dirs(tmp_path: Path) -> tuple[Path, Path]:
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    (iter_dir / "metrics.json").write_text(json.dumps({"metrics": {}}))
    (iter_dir / "behavior.json").write_text(json.dumps({"mean_return": 0.0}))
    (iter_dir / "reward_spec.json").write_text(json.dumps({"version": "v0"}))
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.gym_sb3.GymSB3Adapter"\n'
        'config = { env_id = "Hopper-v4", n_envs = 1 }\n')
    return iter_dir, cfg_path


def _stub_models(world_edits):
    prelim = _PreliminaryModel(
        failure_modes=["none"], evidence="e", confidence=0.5)
    grounded = _GroundedModel(
        proposed_world_edits=world_edits, confidence=0.5)
    return prelim, grounded


def test_diagnose_world_block_and_packing_with_authored_selection(
        tmp_path, monkeypatch) -> None:
    """With an authored selection pinned on the adapter, the grounded
    prompt carries # WORLD_VARIATIONS and only registered-ID edits pack."""
    from test_diagnose import _StubClient
    from test_world_project import _admitted_with_variation

    project, _service, admitted = _admitted_with_variation(tmp_path)
    sel = (project / "env" /
           f"selection_v{admitted.promoted.selection.selection_version}.json")
    iter_dir, cfg_path = _diagnose_fixture_dirs(tmp_path)

    import sculptor.diagnose as D
    real_load = D.load_adapter

    def load_with_world(p):
        a = real_load(p)
        a.world_selection_path = str(sel)
        return a

    monkeypatch.setattr(D, "load_adapter", load_with_world)

    prelim, grounded = _stub_models([
        _ProposedWorldEditModel(
            variation_id="ball_mass",
            new_distribution={"kind": "uniform", "low": 0.1, "high": 0.3},
            rationale="widen mass randomization"),
        _ProposedWorldEditModel(
            variation_id="not_registered",
            new_distribution={"kind": "uniform", "low": 0.0, "high": 1.0},
            rationale="hallucinated"),
    ])
    client = _StubClient(prelim, grounded)
    d = diagnose(iter_dir=iter_dir, behavior_goal="move the ball",
                 config=cfg_path, client=client, skip_kg=True)

    assert [e.variation_id for e in d.proposed_world_edits] == ["ball_mass"]
    assert d.proposed_world_edits[0].new_distribution["low"] == 0.1
    grounded_prompt = (
        client.messages.captured_prompts[1]["messages"][0]["content"])
    assert "# WORLD_VARIATIONS" in grounded_prompt
    assert "ball_mass" in grounded_prompt
    # the persisted diagnosis carries the world edits for audit
    saved = json.loads((iter_dir / "diagnosis.json").read_text())
    assert saved["proposed_world_edits"][0]["variation_id"] == "ball_mass"


def test_diagnose_world_edits_dropped_without_surface(
        tmp_path, monkeypatch) -> None:
    """No authored selection => no # WORLD_VARIATIONS block, and any world
    edits the model hallucinates are dropped at packing."""
    from test_diagnose import _StubClient

    iter_dir, cfg_path = _diagnose_fixture_dirs(tmp_path)
    prelim, grounded = _stub_models([
        _ProposedWorldEditModel(
            variation_id="ball_mass",
            new_distribution={"kind": "uniform", "low": 0.1, "high": 0.3},
            rationale="hallucinated"),
    ])
    client = _StubClient(prelim, grounded)
    d = diagnose(iter_dir=iter_dir, behavior_goal="move the ball",
                 config=cfg_path, client=client, skip_kg=True)
    assert d.proposed_world_edits == []
    grounded_prompt = (
        client.messages.captured_prompts[1]["messages"][0]["content"])
    assert "# WORLD_VARIATIONS" not in grounded_prompt


def test_promote_iteration_selection_base_selection_reverts_world(
        tmp_path) -> None:
    """After a world edit advances the tuple, base_selection restores the
    complete world half from the prior pinned selection — never one file
    in isolation — while reward/env still come from the arguments."""
    from sculptor.sculpt import _promote_iteration_selection
    from sculptor.world.project import (
        WorldVariationEdit,
        apply_world_variation_edits,
    )
    from test_world_project import _admitted_with_variation

    project, service, admitted = _admitted_with_variation(tmp_path)
    v1_version = admitted.promoted.selection.selection_version
    v1_pin = project / "env" / f"selection_v{v1_version}.json"

    edit = apply_world_variation_edits(project, [WorldVariationEdit(
        variation_id="ball_mass",
        new_distribution={"kind": "uniform", "low": 0.1, "high": 0.3},
        rationale="widen")], service=service)
    v2_pin = (project / "env" /
              f"selection_v{edit['selection']['selection_version']}.json")
    adapter = SimpleNamespace(world_selection_path=str(v2_pin))
    reward = project / "rewards" / "v0.py"

    # forward promotion carries the edited world…
    _, forward_pin = _promote_iteration_selection(
        adapter, project, reward_path=reward, env_spec_version=None)
    forward = json.loads(forward_pin.read_text())
    assert forward["refs"]["world"]["version"] == edit["world_version"]

    # …and base_selection restores the pre-edit world half as one tuple.
    _, reverted_pin = _promote_iteration_selection(
        adapter, project, reward_path=reward, env_spec_version=None,
        base_selection=v1_pin)
    reverted = json.loads(reverted_pin.read_text())
    assert (reverted["refs"]["world"]["version"]
            == admitted.promoted.world_ref.version)
    assert (reverted["refs"]["resolved_eval"]["sha256"]
            == json.loads(v1_pin.read_text())["refs"]["resolved_eval"]["sha256"])
    assert reverted["evaluation_lineage"] == "eval-variation-test"
    # the adapter now points at the reverted pin for the next train step
    assert adapter.world_selection_path == str(reverted_pin.resolve())


def test_core_uses_requested_immutable_selection_after_current_changes(
    tmp_path,
) -> None:
    from sculptor.sculpt import (
        _pin_authored_selection,
        _verify_requested_authored_selection,
    )
    from sculptor.world.project import (
        WorldVariationEdit,
        apply_world_variation_edits,
    )
    from test_world_project import _admitted_with_variation

    project, service, admitted = _admitted_with_variation(tmp_path)
    version = admitted.promoted.selection.selection_version
    requested_path = project / "env" / f"selection_v{version}.json"
    expected_sha = hashlib.sha256(requested_path.read_bytes()).hexdigest()
    expected_tuple = admitted.promoted.selection.tuple_hash

    # Promote a different current selection before core initialization.
    apply_world_variation_edits(
        project,
        [WorldVariationEdit(
            variation_id="ball_mass",
            new_distribution={"kind": "uniform", "low": 0.1, "high": 0.3},
            rationale="new authoring after launch admission",
        )],
        service=service,
    )

    receipt = _verify_requested_authored_selection(
        project,
        selection_path=requested_path,
        expected_selection_sha256=expected_sha,
        expected_tuple_hash=expected_tuple,
    )
    assert receipt is not None
    adapter = SimpleNamespace(
        world_selection_path=str(project / "env" / "selection_current.json")
    )
    adapter.world_selection_path = receipt["selection_path"]

    assert _pin_authored_selection(adapter, project) == expected_tuple
    assert Path(adapter.world_selection_path).name == f"selection_v{version}.json"


def test_core_rejects_changed_immutable_selection_bytes(tmp_path) -> None:
    from sculptor.sculpt import _verify_requested_authored_selection
    from test_world_project import _admitted_with_variation

    project, _service, admitted = _admitted_with_variation(tmp_path)
    version = admitted.promoted.selection.selection_version
    requested_path = project / "env" / f"selection_v{version}.json"
    expected_sha = hashlib.sha256(requested_path.read_bytes()).hexdigest()
    requested_path.write_bytes(requested_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="bytes changed"):
        _verify_requested_authored_selection(
            project,
            selection_path=requested_path,
            expected_selection_sha256=expected_sha,
            expected_tuple_hash=admitted.promoted.selection.tuple_hash,
        )


def test_write_world_curriculum_stats_histogram_and_fail_soft(tmp_path) -> None:
    """§10: the runner exports the terrain-level histogram after training;
    envs without a terrain curriculum write nothing and never raise."""
    import torch

    from sculptor.adapters._mjlab_runner import _write_world_curriculum_stats

    terrain = SimpleNamespace(
        terrain_levels=torch.tensor([0, 0, 1, 2, 2, 2]), max_terrain_level=6)
    env = SimpleNamespace(unwrapped=SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain)))
    _write_world_curriculum_stats(env, tmp_path)
    stats = json.loads((tmp_path / "world_curriculum_stats.json").read_text())
    assert stats["num_envs"] == 6
    assert stats["max_level"] == 6
    assert stats["histogram"] == {"0": 2, "1": 1, "2": 3}
    assert abs(stats["mean_level"] - 7 / 6) < 1e-3

    # plane / legacy env: no terrain_levels -> no file, no exception
    bare = SimpleNamespace(unwrapped=SimpleNamespace(
        scene=SimpleNamespace(terrain=None)))
    out2 = tmp_path / "bare"
    out2.mkdir()
    _write_world_curriculum_stats(bare, out2)
    assert not (out2 / "world_curriculum_stats.json").exists()


def test_world_block_renders_curriculum_stats(tmp_path) -> None:
    from sculptor.diagnose import _render_world_variations_block

    variations = [{"id": "ball_mass", "target": "/t", "class": "model_field",
                   "distribution": {"kind": "uniform", "low": 0.1,
                                     "high": 0.2}}]
    stats = {"num_envs": 4, "mean_level": 1.5, "max_level": 6,
             "histogram": {"1": 2, "2": 2}}
    block = _render_world_variations_block(variations, stats)
    assert "terrain curriculum after this iteration" in block
    assert '"mean_level": 1.5' in block
    # stats alone (no registered variations) must not create a surface
    assert _render_world_variations_block([], stats) == ""
    # variations without stats render the block without the stats section
    no_stats = _render_world_variations_block(variations, {})
    assert "terrain curriculum" not in no_stats

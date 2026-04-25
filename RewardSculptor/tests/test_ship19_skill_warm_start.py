"""tests/test_ship19_skill_warm_start.py — Ship 19 integration.

Cross-mission skill library wired into the mission orchestrator and
the decomposer. Stubs `sculpt_run` + `apply_prompt_edit` + load_adapter
so no GPU / Anthropic calls fire.

Covers:
  * `_run_one_stage` resolves `init_skill_id` to a checkpoint via
    the handle and threads it as `init_policy_path` (skill wins over
    parent_ckpt — audit fix C1).
  * Skill is preferred ONLY when explicitly set; otherwise the
    Ship 16 parent_ckpt wins.
  * Unknown skill_id → stage cold-starts + skill_warm_start_skipped
    with reason="skill_not_found".
  * Successful stage triggers `stage_skill_published`.
  * Publish is suppressed when handle is None, on re-decomposition
    sub-stages, on stage failure.
  * `decompose_task` injects the skill-library block AND validates
    `init_skill_id` membership against the rendered slice (audit fix
    BIGGEST HOLE: caught at decompose time, not at runtime).
  * Pydantic normalizes empty-string `init_skill_id` → None (audit H4).
  * stage_warm_start_chosen event shape — for Ship 19's UI surface
    (Ship 19b will render this).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from sculptor.mission import Mission, Stage, save_mission
from sculptor.skill_library import (
    SkillLibrary,
    SkillLibraryHandle,
)


# ── Fixtures + helpers (mirror test_mission_run.py) ──────────────────

class _FakeContract:
    expected_info_keys = []
    expected_components = None
    supports_batched = False
    training_device = "any"
    min_gpu_memory_gb = None
    state_schema = None


class _FakeAdapter:
    """Stub with **kwargs — `adapter_supports_warm_start` returns True
    so the publish gate passes."""
    def reward_contract(self):
        return _FakeContract()

    def train(self, **_kw):
        pass


def _stub_load_adapter(_config_path):
    return _FakeAdapter()


@pytest.fixture
def stub_adapter(monkeypatch):
    monkeypatch.setattr(
        "sculptor.adapters.base.load_adapter", _stub_load_adapter,
    )


def _fabricate_rollout_artifacts(
    iter_dir: Path,
    *,
    behavior: dict[str, Any] | None = None,
) -> None:
    rollout = iter_dir / "rollout"
    rollout.mkdir(parents=True, exist_ok=True)
    beh = behavior if behavior is not None else {
        "n_episodes": 4, "mean_return": 0.9,
        "mean_episode_length": 400.0, "max_episode_length": 500,
    }
    (rollout / "behavior.json").write_text(json.dumps(beh))


def _fake_sculpt_run_factory(*, metric: float):
    """Same shape as test_mission_run.py's helper, but exposes the
    last-passed `init_policy_path` for assertions."""
    captured: dict[str, Any] = {"init_policy_path": None}

    def fake(*, config_path, behavior_goal, iterations=3,
             steps_per_iter=None, seed=None, init_policy_path=None, **_kw):
        captured["init_policy_path"] = init_policy_path
        project = Path(config_path).parent
        iter_dir = project / "runs" / f"iter_{iterations}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "checkpoint.pt").write_bytes(b"fake-ckpt")
        _fabricate_rollout_artifacts(
            iter_dir,
            behavior={"n_episodes": 1, "mean_return": metric,
                      "mean_episode_length": 400.0, "max_episode_length": 500},
        )
        from sculptor.sculpt import IterOutcome, SculptRunResult
        outcome = IterOutcome(
            iter_index=iterations, iter_dir=iter_dir,
            reward_path_before=project / "rewards" / "v1.py",
            reward_path_after=project / "rewards" / f"v{iterations}.py",
            primary_metric=metric, behavior={"mean_return": metric},
            failure_modes=[], edit_count=0,
        )
        return SculptRunResult(
            iterations_run=iterations,
            completed_iters=[outcome],
            primary_metric_history=[metric],
        )
    return fake, captured


def _stub_apply_prompt_edit(*_a, **kw):
    current = Path(kw["current_reward_path"])
    new_iter = kw["new_iter_id"]
    new_path = current.parent / f"{new_iter}.py"
    new_path.write_text(
        "REWARD_SPEC = {'version':'v1','hyperparameters':{},'references':[]}\n"
        "def compute_reward(s,a,n,i): return 0.0, {}\n"
    )
    return new_path


def _make_mission(tmp_path: Path, *, init_skill_id: str | None = None) -> Mission:
    """1-stage mission with the given init_skill_id."""
    stages = [Stage(
        name="stage_0",
        goal_text="do step 0",
        success_criterion="metric > 0.5",
        max_iterations=2,
        parent_stage=None,
        reward_seed_prompt="seed for stage 0",
        init_skill_id=init_skill_id,
    )]
    m = Mission(
        goal="ship19 test",
        stages=stages,
        decomposition_model="claude-opus-4-7",
        decomposition_rationale="test",
    )
    mission_dir = tmp_path / "mission"
    save_mission(m, mission_dir)
    m.mission_dir = str(mission_dir.resolve())
    for stage in stages:
        stage_dir = mission_dir / "stages" / stage.name
        (stage_dir / "rewards").mkdir(parents=True, exist_ok=True)
        (stage_dir / "runs").mkdir(exist_ok=True)
        (stage_dir / "config.toml").write_text(
            '[target]\nname = "x"\n'
            '[adapter]\nclass = "stubbed"\nconfig = {}\n'
            '[iteration]\n'
        )
        (stage_dir / "rewards" / "__init__.py").write_text("")
        (stage_dir / "rewards" / "v0.py").write_text(
            "REWARD_SPEC={'version':'v0','hyperparameters':{},'references':[]}\n"
            "def compute_reward(s,a,n,i): return 0.0, {}\n"
        )
    return m


def _publish_dummy_skill(
    library: SkillLibrary, *,
    adapter_class: str = "sculptor.adapters.mjlab.MjlabAdapter",
    task_id: str = "T",
    checkpoint_path: Path,
) -> str:
    rec = library.publish_from_stage(
        stage=Stage(
            name="seed_stage", goal_text="x", success_criterion="metric > 0",
            max_iterations=2, parent_stage=None, reward_seed_prompt="seed",
            status="succeeded", iterations_used=1,
        ),
        mission=Mission(
            goal="seed mission", stages=[],
            decomposition_model="x", decomposition_rationale="x",
        ),
        adapter_class=adapter_class, task_id=task_id, robot_slug=None,
        checkpoint_path=checkpoint_path,
        final_metric=0.5, source_iter_index=0,
    )
    return rec.skill_id


# ── 1. Skill warm-start resolution wiring ───────────────────────────

def test_run_one_stage_uses_skill_when_init_skill_id_set(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Stage with init_skill_id and no parent → init_policy_path
    threaded through to sculpt_run is the SKILL's checkpoint."""
    from sculptor import sculpt as sculpt_mod

    # Pre-populate the library.
    library = SkillLibrary(root=tmp_path / "lib")
    src_ckpt = tmp_path / "src" / "checkpoint.pt"
    src_ckpt.parent.mkdir(parents=True, exist_ok=True)
    src_ckpt.write_bytes(b"SKILL_BYTES")
    skill_id = _publish_dummy_skill(library, checkpoint_path=src_ckpt)

    handle = SkillLibraryHandle(
        library=library,
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T", publish=True,
    )
    m = _make_mission(tmp_path, init_skill_id=skill_id)

    fake, captured = _fake_sculpt_run_factory(metric=0.9)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append, skill_library_handle=handle,
    )

    # The init_policy_path passed to sculpt_run must point at the
    # library's copy of the skill checkpoint, not the src_ckpt.
    chosen = captured["init_policy_path"]
    assert chosen is not None
    assert Path(chosen).is_file()
    assert library.root in Path(chosen).parents
    # And the corresponding event names the source as the library.
    chosen_events = [e for e in events if e.get("type") == "stage_warm_start_chosen"]
    assert chosen_events
    assert chosen_events[0]["source"] == "skill_library"
    assert chosen_events[0]["source_id"] == skill_id


def test_run_one_stage_falls_back_to_parent_when_init_skill_id_unset(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """No init_skill_id → Ship 16 parent_ckpt wins (when present)."""
    from sculptor import sculpt as sculpt_mod

    # 2-stage mission, both with metric > 0.5 so first succeeds.
    stages = [
        Stage(
            name="parent", goal_text="do parent",
            success_criterion="metric > 0.5", max_iterations=2,
            parent_stage=None, reward_seed_prompt="parent seed",
        ),
        Stage(
            name="child", goal_text="do child",
            success_criterion="metric > 0.5", max_iterations=2,
            parent_stage="parent", reward_seed_prompt="child seed",
            init_skill_id=None,
        ),
    ]
    m = Mission(
        goal="g", stages=stages,
        decomposition_model="claude-opus-4-7",
        decomposition_rationale="t",
    )
    mission_dir = tmp_path / "mission"
    save_mission(m, mission_dir)
    m.mission_dir = str(mission_dir.resolve())
    for s in stages:
        sd = mission_dir / "stages" / s.name
        (sd / "rewards").mkdir(parents=True, exist_ok=True)
        (sd / "runs").mkdir(exist_ok=True)
        (sd / "config.toml").write_text(
            '[target]\nname = "x"\n[adapter]\nclass = "stubbed"\nconfig = {}\n'
            '[iteration]\n',
        )
        (sd / "rewards" / "__init__.py").write_text("")
        (sd / "rewards" / "v0.py").write_text(
            "REWARD_SPEC={'version':'v0','hyperparameters':{},'references':[]}\n"
            "def compute_reward(s,a,n,i): return 0.0, {}\n"
        )

    fake, captured = _fake_sculpt_run_factory(metric=0.9)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    handle = SkillLibraryHandle(
        library=SkillLibrary(root=tmp_path / "lib"),
        adapter_class="A", task_id="T", publish=False,
    )
    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append, skill_library_handle=handle,
    )

    chosen_events = [e for e in events if e.get("type") == "stage_warm_start_chosen"]
    # The first stage cold-starts; the second uses parent_stage.
    assert chosen_events[0]["source"] == "none"
    assert chosen_events[1]["source"] == "parent_stage"
    assert chosen_events[1]["source_id"] == "parent"


def test_run_one_stage_skill_overrides_parent_when_both_set(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Audit fix C1: explicit skill beats implicit parent. Emits
    `warm_start_skipped(reason="skill_overrides_parent")`."""
    from sculptor import sculpt as sculpt_mod

    library = SkillLibrary(root=tmp_path / "lib")
    src = tmp_path / "src" / "checkpoint.pt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"SKILL")
    skill_id = _publish_dummy_skill(library, checkpoint_path=src)

    stages = [
        Stage(
            name="parent", goal_text="do parent",
            success_criterion="metric > 0.5", max_iterations=2,
            parent_stage=None, reward_seed_prompt="parent",
        ),
        Stage(
            name="child", goal_text="do child",
            success_criterion="metric > 0.5", max_iterations=2,
            parent_stage="parent", reward_seed_prompt="child",
            init_skill_id=skill_id,
        ),
    ]
    m = Mission(goal="g", stages=stages,
                decomposition_model="x", decomposition_rationale="x")
    mission_dir = tmp_path / "mission"
    save_mission(m, mission_dir)
    m.mission_dir = str(mission_dir.resolve())
    for s in stages:
        sd = mission_dir / "stages" / s.name
        (sd / "rewards").mkdir(parents=True, exist_ok=True)
        (sd / "runs").mkdir(exist_ok=True)
        (sd / "config.toml").write_text(
            '[target]\nname = "x"\n[adapter]\nclass = "stubbed"\nconfig = {}\n'
            '[iteration]\n',
        )
        (sd / "rewards" / "__init__.py").write_text("")
        (sd / "rewards" / "v0.py").write_text(
            "REWARD_SPEC={'version':'v0','hyperparameters':{},'references':[]}\n"
            "def compute_reward(s,a,n,i): return 0.0, {}\n"
        )

    fake, captured = _fake_sculpt_run_factory(metric=0.9)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    handle = SkillLibraryHandle(
        library=library,
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T", publish=False,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append, skill_library_handle=handle,
    )
    # The CHILD stage's warm_start_chosen says "skill_library"
    chosen = [e for e in events if e.get("type") == "stage_warm_start_chosen"
              and e.get("stage_name") == "child"]
    assert chosen and chosen[0]["source"] == "skill_library"
    # And we emitted the override-warning event.
    overrides = [
        e for e in events
        if e.get("type") == "warm_start_skipped"
        and e.get("reason") == "skill_overrides_parent"
    ]
    assert len(overrides) == 1


def test_run_one_stage_skips_skill_on_unknown_id(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Unknown skill_id → cold-start (or parent), with skipped event."""
    from sculptor import sculpt as sculpt_mod

    handle = SkillLibraryHandle(
        library=SkillLibrary(root=tmp_path / "lib"),
        adapter_class="A", task_id="T", publish=False,
    )
    m = _make_mission(tmp_path, init_skill_id="000000000000")
    fake, captured = _fake_sculpt_run_factory(metric=0.9)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append, skill_library_handle=handle,
    )
    skipped = [
        e for e in events
        if e.get("type") == "skill_warm_start_skipped"
        and e.get("reason") == "skill_not_found"
    ]
    assert len(skipped) == 1
    chosen = [e for e in events if e.get("type") == "stage_warm_start_chosen"]
    assert chosen[0]["source"] == "none"


# ── 2. Publish wiring ────────────────────────────────────────────────

def test_run_one_stage_publishes_to_library_on_success(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    from sculptor import sculpt as sculpt_mod

    library = SkillLibrary(root=tmp_path / "lib")
    handle = SkillLibraryHandle(
        library=library,
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T", publish=True,
    )
    m = _make_mission(tmp_path)
    fake, _ = _fake_sculpt_run_factory(metric=0.9)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append, skill_library_handle=handle,
    )
    pubs = [e for e in events if e.get("type") == "stage_skill_published"]
    assert len(pubs) == 1
    assert pubs[0]["task_id"] == "T"
    # Library reflects it.
    listed = library.list_compatible(
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T",
    )
    assert len(listed) == 1
    assert listed[0].source_stage_name == "stage_0"


def test_run_one_stage_skips_publish_when_handle_is_none(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    from sculptor import sculpt as sculpt_mod
    m = _make_mission(tmp_path)
    fake, _ = _fake_sculpt_run_factory(metric=0.9)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append, skill_library_handle=None,
    )
    pubs = [e for e in events if e.get("type") == "stage_skill_published"]
    skipped = [e for e in events if e.get("type") == "stage_skill_publish_skipped"]
    assert pubs == []
    assert skipped == []  # no event when handle is None — silent path


def test_run_one_stage_skips_publish_for_failed_stage(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    from sculptor import sculpt as sculpt_mod
    library = SkillLibrary(root=tmp_path / "lib")
    handle = SkillLibraryHandle(
        library=library,
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T", publish=True,
    )
    m = _make_mission(tmp_path)
    # Pre-exhaust budget so failure → halt (no Ship 17 redecomp).
    m.stages[0].redecomposition_attempts = 1
    fake, _ = _fake_sculpt_run_factory(metric=0.3)  # < 0.5
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append, skill_library_handle=handle,
    )
    pubs = [e for e in events if e.get("type") == "stage_skill_published"]
    assert pubs == []
    # Library is empty.
    assert library.list_compatible(
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T",
    ) == []


# ── 3. decompose_task integration ────────────────────────────────────

def test_decompose_task_renders_skill_library_block_when_handle_provided(
    tmp_path: Path,
):
    """The user_content passed to Claude must include a SKILL_LIBRARY
    section listing the compatible record's skill_id."""
    from sculptor import decompose as dc

    library = SkillLibrary(root=tmp_path / "lib")
    src = tmp_path / "src" / "checkpoint.pt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"X")
    skill_id = _publish_dummy_skill(library, checkpoint_path=src)
    handle = SkillLibraryHandle(
        library=library,
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T", publish=False,
    )

    block, ids = dc._render_skill_library_context(handle)
    assert "SKILL_LIBRARY" in block
    assert skill_id in block
    assert ids == [skill_id]


def test_decompose_task_validates_init_skill_id_membership(
    tmp_path: Path,
):
    """Audit fix BIGGEST HOLE / decompose layer: stage citing an
    unknown skill_id is rejected at validation time."""
    from sculptor import decompose as dc
    from sculptor.mission import MissionValidationError

    stages = [
        Stage(
            name="stage_a", goal_text="x", success_criterion="metric > 0",
            max_iterations=2, parent_stage=None, reward_seed_prompt="x",
            init_skill_id="fakeid01234",  # not in any rendered slice
        ),
    ]
    with pytest.raises(MissionValidationError, match="unknown skills"):
        dc._validate_skill_ids(stages, available_skill_ids=set())


def test_decompose_task_accepts_validated_init_skill_id():
    """A skill_id that IS in the rendered slice passes through."""
    from sculptor import decompose as dc
    stages = [
        Stage(
            name="stage_a", goal_text="x", success_criterion="metric > 0",
            max_iterations=2, parent_stage=None, reward_seed_prompt="x",
            init_skill_id="abcdef012345",
        ),
    ]
    # No raise.
    dc._validate_skill_ids(stages, available_skill_ids={"abcdef012345"})


def test_pydantic_normalizes_empty_init_skill_id_to_none():
    """Audit fix H4: '' / whitespace become None at parse time so a
    Claude response of `init_skill_id: ""` doesn't trip the validator."""
    from sculptor.decompose import _StageModel
    model = _StageModel(
        name="stage_a", goal_text="x", success_criterion="metric > 0",
        max_iterations=2, parent_stage=None, reward_seed_prompt="abcdef",
        init_skill_id="",
    )
    assert model.init_skill_id is None
    model2 = _StageModel(
        name="stage_a", goal_text="x", success_criterion="metric > 0",
        max_iterations=2, parent_stage=None, reward_seed_prompt="abcdef",
        init_skill_id="   ",
    )
    assert model2.init_skill_id is None


# ── 4. Backward compatibility ────────────────────────────────────────

def test_old_mission_json_loads_with_init_skill_id_none(tmp_path: Path):
    """Backward-compat: a mission.json from before Ship 19 (no
    init_skill_id key) loads cleanly with the field defaulting to
    None via the filter-unknown-keys path."""
    from sculptor.mission import Mission

    legacy_payload = {
        "schema_version": 1,
        "goal": "legacy",
        "decomposition_model": "claude-opus-4-7",
        "decomposition_rationale": "legacy",
        "stages": [{
            "name": "s0", "goal_text": "x", "success_criterion": "metric > 0",
            "max_iterations": 2, "parent_stage": None,
            "reward_seed_prompt": "abcdef",
            "kg_seed_papers": [],
            # No init_skill_id key — must default to None.
        }],
    }
    m = Mission.from_dict(legacy_payload)
    assert m.stages[0].init_skill_id is None


def test_redecompose_substages_have_init_skill_id_none():
    """Re-decomp sub-stages cold-start (or chain via parent). The
    Stage construction in `redecompose_stage` does NOT propagate
    init_skill_id from the model_stage so this is a contract via
    source inspection — guard against future refactors."""
    import inspect
    from sculptor import decompose as dc
    src = inspect.getsource(dc.redecompose_stage)
    # The Stage(...) construction in the loop body shouldn't pass
    # init_skill_id (so it defaults to None on the new sub-stage).
    # If a future PR adds it, this test fails — at which point the
    # author should explicitly think about whether sub-stages should
    # carry the skill reference.
    construction_block_starts = src.find("sub_stages.append(Stage(")
    assert construction_block_starts != -1
    construction_block_ends = src.find("))", construction_block_starts)
    assert construction_block_ends != -1
    construction_block = src[construction_block_starts:construction_block_ends]
    assert "init_skill_id" not in construction_block, (
        "redecompose_stage must NOT propagate init_skill_id to sub-"
        "stages. If you change this, update this guard test AND the "
        "Ship 19 prompt rules."
    )

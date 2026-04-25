"""tests/test_skill_library.py — Ship 19: cross-mission skill library.

Unit tests for the file-system-backed registry. Integration with
mission_run / decompose_task lives in test_ship19_skill_warm_start.py.

Covers:
  * derive_skill_id determinism + non-collision on whitespace
  * full-file SHA-256 (not first-N-bytes — audit fix C4)
  * publish writes metadata atomically (tmp+rename — audit fix H2)
  * publish skips re-decomposition sub-stages (audit fix H1)
  * publish skips when adapter lacks warm-start support (audit fix
    BIGGEST HOLE mitigation)
  * publish uses BEST-iter checkpoint, not last-iter (audit fix C2)
  * list_compatible filters by (adapter_class, task_id), orders by
    final_metric desc, top_k caps results
  * list_compatible skips records with corrupt metadata.json
  * env-var override for library root + explicit `root=` constructor
  * concurrent publish via filelock (no metadata clobber)
  * SkillLibraryHandle.maybe_load skip-reason taxonomy
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from sculptor.mission import Mission, Stage
from sculptor.skill_library import (
    ENV_LIBRARY_ROOT,
    METADATA_FILENAME,
    SCHEMA_VERSION,
    SkillLibrary,
    SkillLibraryHandle,
    SkillLibraryError,
    SkillRecord,
    adapter_supports_warm_start,
    default_library_root,
    derive_skill_id,
)


# ── Helpers ──────────────────────────────────────────────────────────
def _make_stage(
    name: str = "stand",
    *,
    redecomposition_attempts: int = 0,
    iterations_used: int = 3,
    init_skill_id: str | None = None,
) -> Stage:
    return Stage(
        name=name,
        goal_text=f"do {name}",
        success_criterion="metric > 0.5",
        max_iterations=5,
        parent_stage=None,
        reward_seed_prompt=f"seed for {name}",
        kg_seed_papers=[],
        init_skill_id=init_skill_id,
        status="succeeded",
        final_policy_path="/dev/null",
        iterations_used=iterations_used,
        redecomposition_attempts=redecomposition_attempts,
    )


def _make_mission(goal: str = "test goal", stages: list[Stage] | None = None) -> Mission:
    return Mission(
        goal=goal,
        stages=stages or [_make_stage()],
        decomposition_model="claude-opus-4-7",
        decomposition_rationale="test",
    )


def _write_ckpt(path: Path, content: bytes = b"FAKE_CKPT_BYTES_v1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class _FakeIterOutcome:
    def __init__(self, iter_dir: Path) -> None:
        self.iter_dir = iter_dir


class _FakeSculptResult:
    def __init__(
        self,
        completed_iters: list[_FakeIterOutcome],
        primary_metric_history: list[float],
    ) -> None:
        self.completed_iters = completed_iters
        self.primary_metric_history = primary_metric_history


class _AdapterWithWarmStart:
    def train(self, *, init_policy_path=None, **_kw) -> None:
        pass


class _AdapterWithKwargs:
    def train(self, **_kw) -> None:
        pass


class _AdapterNoWarmStart:
    def train(self, x: int, y: int) -> None:
        pass


# ── derive_skill_id ──────────────────────────────────────────────────

def test_skill_id_derivation_deterministic():
    a = derive_skill_id("X.Adapter", "Task-A", "abc123")
    b = derive_skill_id("X.Adapter", "Task-A", "abc123")
    assert a == b
    assert len(a) == 12


def test_skill_id_changes_on_any_input():
    a = derive_skill_id("X.Adapter", "Task-A", "abc")
    b = derive_skill_id("X.Adapter", "Task-A", "def")
    c = derive_skill_id("Y.Adapter", "Task-A", "abc")
    d = derive_skill_id("X.Adapter", "Task-B", "abc")
    assert len({a, b, c, d}) == 4


def test_skill_id_strips_whitespace():
    """Audit fix C3: textual normalization. Same content, different
    whitespace must collide to the same id (otherwise re-publishes
    of the same policy fragment the library)."""
    a = derive_skill_id(" X.Adapter ", "Task-A\n", " abc ")
    b = derive_skill_id("X.Adapter", "Task-A", "abc")
    assert a == b


def test_skill_id_independent_of_reward_seed_prompt():
    """Two stages w/ identical (adapter, task, ckpt-bytes) but
    different reward_seed_prompt collapse to the same skill_id —
    identity is policy bytes + task, NOT description (audit C3)."""
    # Confirms reward_seed_prompt is NOT in derive_skill_id's signature.
    import inspect
    sig = inspect.signature(derive_skill_id)
    assert "reward_seed_prompt" not in sig.parameters
    assert "success_criterion" not in sig.parameters


# ── adapter_supports_warm_start introspection ────────────────────────

def test_adapter_supports_warm_start_explicit_kwarg():
    assert adapter_supports_warm_start(_AdapterWithWarmStart()) is True


def test_adapter_supports_warm_start_var_kwargs():
    """Audit fix BIGGEST HOLE: adapters with **kwargs catch-all
    ARE considered supporting (mirrors Ship 15 _train_or_resume)."""
    assert adapter_supports_warm_start(_AdapterWithKwargs()) is True


def test_adapter_does_not_support_warm_start():
    assert adapter_supports_warm_start(_AdapterNoWarmStart()) is False


def test_adapter_supports_warm_start_handles_missing_train_method():
    class _NoTrain:
        pass
    assert adapter_supports_warm_start(_NoTrain()) is False


# ── default_library_root ─────────────────────────────────────────────

def test_default_library_root_respects_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "explicit_root"
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(target))
    assert default_library_root() == target


def test_default_library_root_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_LIBRARY_ROOT, raising=False)
    root = default_library_root()
    # Fallback path; just check it's under home.
    assert "sculptor" in str(root)


# ── SkillLibrary.publish_from_stage ──────────────────────────────────

def test_publish_writes_metadata_and_checkpoint(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    ckpt = tmp_path / "stage" / "iter_3" / "checkpoint.pt"
    _write_ckpt(ckpt)
    stage = _make_stage()
    mission = _make_mission()

    rec = lib.publish_from_stage(
        stage=stage,
        mission=mission,
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        robot_slug="g1_humanoid",
        checkpoint_path=ckpt,
        final_metric=0.93,
        source_iter_index=3,
    )

    record_dir = (tmp_path / "lib") / rec.skill_id
    assert (record_dir / METADATA_FILENAME).is_file()
    assert (record_dir / "checkpoint.pt").is_file()

    persisted = json.loads((record_dir / METADATA_FILENAME).read_text())
    assert persisted["adapter_class"] == "sculptor.adapters.mjlab.MjlabAdapter"
    assert persisted["task_id"] == "Mjlab-Velocity-Flat-Unitree-Go1"
    assert persisted["final_metric"] == pytest.approx(0.93)
    assert persisted["source_iter_index"] == 3
    assert persisted["checkpoint_filename"] == "checkpoint.pt"
    assert persisted["checkpoint_size_bytes"] > 0
    assert persisted["schema_version"] == SCHEMA_VERSION


def test_publish_atomic_via_tmp_rename(tmp_path: Path):
    """Audit fix H2: metadata.json must be tmp+rename so concurrent
    readers never see a half-written file."""
    lib = SkillLibrary(root=tmp_path / "lib")
    ckpt = tmp_path / "stage" / "iter_3" / "checkpoint.pt"
    _write_ckpt(ckpt)
    rec = lib.publish_from_stage(
        stage=_make_stage(),
        mission=_make_mission(),
        adapter_class="A",
        task_id="T",
        robot_slug=None,
        checkpoint_path=ckpt,
        final_metric=0.5,
        source_iter_index=0,
    )
    record_dir = lib.root / rec.skill_id
    # No `.tmp` files left over after a successful publish.
    leftovers = list(record_dir.glob("*.tmp"))
    assert leftovers == [], f"unexpected tmp files: {leftovers}"


def test_publish_full_file_sha256_in_metadata(tmp_path: Path):
    """Audit fix C4: full-file SHA-256, NOT first-N bytes."""
    lib = SkillLibrary(root=tmp_path / "lib")
    body = b"x" * (32 * 1024)  # 32 KB so a "first 8 KB" hash would miss the tail.
    ckpt = tmp_path / "stage" / "iter_3" / "checkpoint.pt"
    _write_ckpt(ckpt, body)
    rec = lib.publish_from_stage(
        stage=_make_stage(), mission=_make_mission(),
        adapter_class="A", task_id="T", robot_slug=None,
        checkpoint_path=ckpt,
        final_metric=0.5, source_iter_index=0,
    )
    expected = hashlib.sha256(body).hexdigest()
    assert rec.checkpoint_sha256 == expected


def test_publish_raises_on_missing_checkpoint(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    with pytest.raises(SkillLibraryError, match="checkpoint not found"):
        lib.publish_from_stage(
            stage=_make_stage(), mission=_make_mission(),
            adapter_class="A", task_id="T", robot_slug=None,
            checkpoint_path=tmp_path / "nonexistent.pt",
            final_metric=0.5, source_iter_index=0,
        )


# ── SkillLibrary.list_compatible ─────────────────────────────────────

def test_list_compatible_filters_by_adapter_and_task(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    for i, (adapter, task) in enumerate([
        ("A", "T1"), ("A", "T1"), ("A", "T2"), ("B", "T1"),
    ]):
        ckpt = tmp_path / f"stage_{i}" / "checkpoint.pt"
        _write_ckpt(ckpt, content=f"v{i}".encode())
        lib.publish_from_stage(
            stage=_make_stage(name=f"s{i}"),
            mission=_make_mission(),
            adapter_class=adapter, task_id=task, robot_slug=None,
            checkpoint_path=ckpt,
            final_metric=0.1 * i, source_iter_index=i,
        )
    matches = lib.list_compatible(adapter_class="A", task_id="T1", top_k=10)
    assert len(matches) == 2
    assert all(r.adapter_class == "A" and r.task_id == "T1" for r in matches)


def test_list_compatible_orders_by_final_metric_desc(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    metrics = [0.3, 0.9, 0.6]
    for i, m in enumerate(metrics):
        ckpt = tmp_path / f"stage_{i}" / "checkpoint.pt"
        _write_ckpt(ckpt, content=f"v{i}".encode())
        lib.publish_from_stage(
            stage=_make_stage(name=f"s{i}"),
            mission=_make_mission(),
            adapter_class="A", task_id="T", robot_slug=None,
            checkpoint_path=ckpt,
            final_metric=m, source_iter_index=i,
        )
    matches = lib.list_compatible(adapter_class="A", task_id="T", top_k=10)
    assert [r.final_metric for r in matches] == [0.9, 0.6, 0.3]


def test_list_compatible_top_k_caps(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    for i in range(7):
        ckpt = tmp_path / f"stage_{i}" / "checkpoint.pt"
        _write_ckpt(ckpt, content=f"v{i}".encode())
        lib.publish_from_stage(
            stage=_make_stage(name=f"s{i}"),
            mission=_make_mission(),
            adapter_class="A", task_id="T", robot_slug=None,
            checkpoint_path=ckpt,
            final_metric=float(i), source_iter_index=i,
        )
    assert len(lib.list_compatible(adapter_class="A", task_id="T", top_k=3)) == 3


def test_list_compatible_skips_corrupt_metadata(tmp_path: Path):
    """Audit fix H2: a single corrupt metadata.json must NOT raise
    when listing — it should be silently skipped so the rest of
    the library remains queryable."""
    lib = SkillLibrary(root=tmp_path / "lib")
    ckpt_good = tmp_path / "good" / "checkpoint.pt"
    _write_ckpt(ckpt_good)
    rec = lib.publish_from_stage(
        stage=_make_stage(), mission=_make_mission(),
        adapter_class="A", task_id="T", robot_slug=None,
        checkpoint_path=ckpt_good,
        final_metric=0.5, source_iter_index=0,
    )
    # Drop a corrupt sibling.
    bad_dir = lib.root / "bad_corrupt_dir"
    bad_dir.mkdir()
    (bad_dir / METADATA_FILENAME).write_text("{not valid json")
    matches = lib.list_compatible(adapter_class="A", task_id="T")
    assert len(matches) == 1
    assert matches[0].skill_id == rec.skill_id


def test_load_unknown_skill_returns_none(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    assert lib.load("000000000000") is None


# ── Concurrent publish via filelock ──────────────────────────────────

def test_concurrent_publish_serialized_via_filelock(tmp_path: Path):
    """Audit Risk 3: per-(adapter, task) lock prevents two concurrent
    publishes from interleaving metadata writes. The atomic-rename
    on success makes the OUTCOME deterministic; the lock prevents
    two writers from racing on the SAME skill_id directory."""
    lib = SkillLibrary(root=tmp_path / "lib")

    # Two stages with DIFFERENT checkpoint contents → different
    # skill_ids, but same (adapter, task) so they share a lock dir.
    ckpts = []
    for i in range(2):
        c = tmp_path / f"stage_{i}" / "checkpoint.pt"
        _write_ckpt(c, content=f"variant-{i}".encode())
        ckpts.append(c)

    results: list[SkillRecord] = []
    errors: list[Exception] = []

    def _worker(i: int) -> None:
        try:
            r = lib.publish_from_stage(
                stage=_make_stage(name=f"s{i}"),
                mission=_make_mission(),
                adapter_class="A", task_id="T", robot_slug=None,
                checkpoint_path=ckpts[i],
                final_metric=0.5 + 0.1 * i, source_iter_index=i,
            )
            results.append(r)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 2
    # Both records ended up in the library, atomic + serialized.
    listed = lib.list_compatible(adapter_class="A", task_id="T")
    assert len(listed) == 2


# ── SkillLibraryHandle.maybe_load_for_stage ──────────────────────────

def _record_emit():
    events: list[dict[str, Any]] = []

    def _emit(payload: dict[str, Any]) -> None:
        events.append(payload)

    return events, _emit


def test_handle_maybe_load_returns_none_when_init_skill_id_unset(tmp_path: Path):
    handle = SkillLibraryHandle(
        library=SkillLibrary(root=tmp_path / "lib"),
        adapter_class="A", task_id="T",
    )
    events, emit = _record_emit()
    assert handle.maybe_load_for_stage(_make_stage(init_skill_id=None), emit) is None
    # No event emitted for the silent "not requested" path.
    assert events == []


def test_handle_maybe_load_emits_skipped_on_unknown_id(tmp_path: Path):
    handle = SkillLibraryHandle(
        library=SkillLibrary(root=tmp_path / "lib"),
        adapter_class="A", task_id="T",
    )
    events, emit = _record_emit()
    out = handle.maybe_load_for_stage(
        _make_stage(init_skill_id="000000000000"), emit,
    )
    assert out is None
    assert any(
        e.get("type") == "skill_warm_start_skipped"
        and e.get("reason") == "skill_not_found"
        for e in events
    )


def test_handle_maybe_load_emits_skipped_on_adapter_class_mismatch(tmp_path: Path):
    """Record was published under adapter A but the handle is for B —
    even if the id resolves, we refuse to load (would TypeError or
    silently corrupt)."""
    lib = SkillLibrary(root=tmp_path / "lib")
    ckpt = tmp_path / "stage" / "checkpoint.pt"
    _write_ckpt(ckpt)
    rec = lib.publish_from_stage(
        stage=_make_stage(), mission=_make_mission(),
        adapter_class="A", task_id="T", robot_slug=None,
        checkpoint_path=ckpt,
        final_metric=0.5, source_iter_index=0,
    )
    handle = SkillLibraryHandle(library=lib, adapter_class="B", task_id="T")
    events, emit = _record_emit()
    out = handle.maybe_load_for_stage(
        _make_stage(init_skill_id=rec.skill_id), emit,
    )
    assert out is None
    assert any(
        e.get("type") == "skill_warm_start_skipped"
        and e.get("reason") == "adapter_class_mismatch"
        for e in events
    )


def test_handle_maybe_load_emits_skipped_on_task_id_mismatch(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    ckpt = tmp_path / "stage" / "checkpoint.pt"
    _write_ckpt(ckpt)
    rec = lib.publish_from_stage(
        stage=_make_stage(), mission=_make_mission(),
        adapter_class="A", task_id="T1", robot_slug=None,
        checkpoint_path=ckpt,
        final_metric=0.5, source_iter_index=0,
    )
    handle = SkillLibraryHandle(library=lib, adapter_class="A", task_id="T2")
    events, emit = _record_emit()
    out = handle.maybe_load_for_stage(
        _make_stage(init_skill_id=rec.skill_id), emit,
    )
    assert out is None
    assert any(
        e.get("type") == "skill_warm_start_skipped"
        and e.get("reason") == "task_id_mismatch"
        for e in events
    )


def test_handle_maybe_load_returns_path_on_match(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    ckpt = tmp_path / "stage" / "checkpoint.pt"
    _write_ckpt(ckpt)
    rec = lib.publish_from_stage(
        stage=_make_stage(), mission=_make_mission(),
        adapter_class="A", task_id="T", robot_slug=None,
        checkpoint_path=ckpt,
        final_metric=0.5, source_iter_index=0,
    )
    handle = SkillLibraryHandle(library=lib, adapter_class="A", task_id="T")
    events, emit = _record_emit()
    out = handle.maybe_load_for_stage(
        _make_stage(init_skill_id=rec.skill_id), emit,
    )
    assert out is not None
    assert out.is_file()
    assert "skill_warm_start_skipped" not in [e.get("type") for e in events]


# ── SkillLibraryHandle.maybe_publish gates ───────────────────────────

def _make_publish_inputs(tmp_path: Path):
    lib = SkillLibrary(root=tmp_path / "lib")
    handle = SkillLibraryHandle(
        library=lib,
        adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
        task_id="T",
        robot_slug=None,
        publish=True,
    )
    iter_dir = tmp_path / "stage" / "runs" / "iter_3"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "checkpoint.pt").write_bytes(b"ok")
    sculpt_result = _FakeSculptResult(
        completed_iters=[_FakeIterOutcome(iter_dir)],
        primary_metric_history=[0.7],
    )
    return lib, handle, sculpt_result


def test_publish_skipped_when_handle_publish_disabled(tmp_path: Path):
    lib, handle, sculpt_result = _make_publish_inputs(tmp_path)
    handle.publish = False
    events, emit = _record_emit()
    rec = handle.maybe_publish(
        stage=_make_stage(),
        mission=_make_mission(),
        adapter=_AdapterWithWarmStart(),
        sculpt_result=sculpt_result,
        emit=emit,
    )
    assert rec is None
    assert any(
        e.get("type") == "stage_skill_publish_skipped"
        and e.get("reason") == "handle_publish_disabled"
        for e in events
    )


def test_publish_skipped_for_redecomposition_artifact(tmp_path: Path):
    """Audit fix H1: don't publish stages born of re-decomposition."""
    _, handle, sculpt_result = _make_publish_inputs(tmp_path)
    events, emit = _record_emit()
    rec = handle.maybe_publish(
        stage=_make_stage(redecomposition_attempts=1),
        mission=_make_mission(),
        adapter=_AdapterWithWarmStart(),
        sculpt_result=sculpt_result,
        emit=emit,
    )
    assert rec is None
    assert any(
        e.get("type") == "stage_skill_publish_skipped"
        and e.get("reason") == "redecomposition_artifact"
        for e in events
    )


def test_publish_skipped_when_adapter_lacks_warm_start(tmp_path: Path):
    """Audit fix BIGGEST HOLE mitigation: don't publish skills no
    future mission can load."""
    _, handle, sculpt_result = _make_publish_inputs(tmp_path)
    events, emit = _record_emit()
    rec = handle.maybe_publish(
        stage=_make_stage(),
        mission=_make_mission(),
        adapter=_AdapterNoWarmStart(),
        sculpt_result=sculpt_result,
        emit=emit,
    )
    assert rec is None
    assert any(
        e.get("type") == "stage_skill_publish_skipped"
        and e.get("reason") == "adapter_does_not_support_warm_start"
        for e in events
    )


def test_publish_uses_best_iter_checkpoint_not_last(tmp_path: Path):
    """Audit fix C2: when iter_3 metric was 0.9 but iter_5 metric was
    0.4 (regressed), publish the iter_3 checkpoint."""
    lib = SkillLibrary(root=tmp_path / "lib")
    handle = SkillLibraryHandle(
        library=lib, adapter_class="A", task_id="T",
    )
    # Build a sculpt_result with 5 iters; best is iter index 2 (metric 0.9).
    history = [0.3, 0.5, 0.9, 0.6, 0.4]
    completed = []
    for i, m in enumerate(history):
        d = tmp_path / f"runs/iter_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoint.pt").write_bytes(f"ckpt-iter-{i}".encode())
        completed.append(_FakeIterOutcome(d))
    sculpt_result = _FakeSculptResult(completed, history)

    events, emit = _record_emit()
    rec = handle.maybe_publish(
        stage=_make_stage(),
        mission=_make_mission(),
        adapter=_AdapterWithWarmStart(),
        sculpt_result=sculpt_result,
        emit=emit,
    )
    assert rec is not None
    assert rec.source_iter_index == 2
    assert rec.final_metric == pytest.approx(0.9)
    # Confirm the COPIED checkpoint matches iter_2's bytes, NOT iter_4.
    copied = (lib.root / rec.skill_id / rec.checkpoint_filename).read_bytes()
    assert copied == b"ckpt-iter-2"


def test_publish_skipped_when_history_empty(tmp_path: Path):
    _, handle, _ = _make_publish_inputs(tmp_path)
    sculpt_result = _FakeSculptResult(completed_iters=[], primary_metric_history=[])
    events, emit = _record_emit()
    rec = handle.maybe_publish(
        stage=_make_stage(),
        mission=_make_mission(),
        adapter=_AdapterWithWarmStart(),
        sculpt_result=sculpt_result,
        emit=emit,
    )
    assert rec is None
    assert any(
        e.get("type") == "stage_skill_publish_skipped"
        and e.get("reason") == "no_metric_history"
        for e in events
    )


def test_atomic_helpers_fsync_parent_directory_in_source():
    """Code-audit BIGGEST BUG regression guard. Both `_atomic_write_text`
    and `_atomic_copy` must fsync the parent directory after the
    rename so the new dirent is durable across power-loss / kernel
    crash. Without this, recovery can show the temp file gone but
    the permanent file missing because the directory inode wasn't
    persisted. Source-inspection guard against future re-introduction
    of the bug — there's no portable way to crash-test this in unit
    tests."""
    import inspect
    from sculptor import skill_library as sl
    write_src = inspect.getsource(sl._atomic_write_text)
    copy_src = inspect.getsource(sl._atomic_copy)
    assert "_fsync_dir(" in write_src, (
        "_atomic_write_text must call _fsync_dir on the parent "
        "directory after os.replace to make the rename durable."
    )
    assert "_fsync_dir(" in copy_src, (
        "_atomic_copy must call _fsync_dir on the parent directory "
        "after os.replace to make the rename durable."
    )


def test_publish_emits_stage_skill_published_on_success(tmp_path: Path):
    _, handle, sculpt_result = _make_publish_inputs(tmp_path)
    events, emit = _record_emit()
    rec = handle.maybe_publish(
        stage=_make_stage(),
        mission=_make_mission(),
        adapter=_AdapterWithWarmStart(),
        sculpt_result=sculpt_result,
        emit=emit,
    )
    assert rec is not None
    pubs = [e for e in events if e.get("type") == "stage_skill_published"]
    assert len(pubs) == 1
    assert pubs[0]["skill_id"] == rec.skill_id
    assert pubs[0]["final_metric"] == pytest.approx(0.7)

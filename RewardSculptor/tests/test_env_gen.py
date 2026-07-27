"""§RL_SCULPTOR_AUDIT (env generalization 2/4): goal-conditioned env
spec generation + project-side versioning. Offline — the LLM is a stub
returning fixed pydantic models (test_diagnose.py convention);
validation runs the REAL validate_env_spec gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sculptor import env_gen, env_spec as es
from sculptor.env_gen import _EnvSpecModel, generate_env_spec
from sculptor.sculpt import _maybe_seed_env_spec


# ── Stubs (test_diagnose.py convention: messages.parse) ────────────────────
class _StubResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, *outputs):
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._outputs:
            raise AssertionError("stub ran out of responses")
        out = self._outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return _StubResponse(out)


class _StubClient:
    def __init__(self, *outputs):
        self.messages = _StubMessages(*outputs)


class _ExplodingClient:
    class _M:
        def parse(self, **kwargs):
            raise ConnectionError("no API key / network down")

    messages = _M()


def _good_model() -> _EnvSpecModel:
    return _EnvSpecModel.model_validate({
        "reasoning": "in-place explosive skill: zero commands, raise "
                     "orientation cut, RSI + sunk pairing, entropy up.",
        "shared": {
            "zero_velocity_commands": True,
            "orientation_termination_deg": 120.0,
            "episode_length_s": 10.0,
            "push_events": {"enabled": False},
        },
        "train": {
            "reset_height_offset_m": [0.0, 0.4],
            "reset_vertical_velocity_mps": [-0.5, 2.0],
            "min_base_height_termination_m": 0.3,
            "entropy_coef_scale": 2.0,
        },
    })


def _out_of_bounds_model() -> _EnvSpecModel:
    m = _good_model()
    m.train.entropy_coef_scale = 99.0     # fails the bounds gate
    return m


# ── generate_env_spec ──────────────────────────────────────────────────────
def test_generation_happy_path_one_call() -> None:
    client = _StubClient(_good_model())
    spec = generate_env_spec(
        behavior_goal="a standing tuck jump", task_id="Mjlab-X-G1",
        client=client)
    assert es.validate_env_spec(spec) == []
    assert spec["meta"]["source"] == "generated"
    assert spec["meta"]["behavior_goal"] == "a standing tuck jump"
    assert spec["shared"]["zero_velocity_commands"] is True
    assert spec["train"]["min_base_height_termination_m"] == 0.3
    assert len(client.messages.calls) == 1
    # Bounds ride in the user prompt, rendered from the validator tables.
    user = client.messages.calls[0]["messages"][0]["content"]
    assert "HARD BOUNDS" in user
    assert "entropy_coef_scale" in user


def test_generation_omitted_fields_dropped() -> None:
    m = _EnvSpecModel.model_validate({
        "reasoning": "defaults are right; only shorten episodes.",
        "shared": {"episode_length_s": 8.0},
        "train": {},
    })
    spec = generate_env_spec(
        behavior_goal="walk forward", task_id="T", client=_StubClient(m))
    assert spec["shared"] == {"episode_length_s": 8.0}
    assert spec["train"] == {}


def test_generation_retry_carries_all_violations_then_succeeds() -> None:
    client = _StubClient(_out_of_bounds_model(), _good_model())
    spec = generate_env_spec(
        behavior_goal="jump", task_id="T", client=client)
    assert es.validate_env_spec(spec) == []
    assert len(client.messages.calls) == 2
    retry_user = client.messages.calls[1]["messages"][0]["content"]
    assert "failed validation" in retry_user
    assert "entropy_coef_scale" in retry_user


def test_generation_fails_after_two_bad_attempts() -> None:
    client = _StubClient(_out_of_bounds_model(), _out_of_bounds_model())
    with pytest.raises(ValueError, match="twice"):
        generate_env_spec(behavior_goal="jump", task_id="T", client=client)
    assert len(client.messages.calls) == 2


def test_generation_parse_error_counts_as_attempt() -> None:
    client = _StubClient(RuntimeError("parse boom"), _good_model())
    spec = generate_env_spec(behavior_goal="jump", task_id="T", client=client)
    assert es.validate_env_spec(spec) == []
    assert len(client.messages.calls) == 2


def test_generator_models_cover_the_schema_exactly() -> None:
    """Drift guard: every schema key must be emittable by the generator
    models and vice versa — a field added to env_spec.py without a model
    field would be silently un-generatable (increment-2 verifier)."""
    from sculptor.env_gen import _SharedModel, _TrainModel
    from sculptor.env_spec import _SHARED_KEYS, _TRAIN_KEYS

    assert set(_SharedModel.model_fields) == _SHARED_KEYS
    assert set(_TrainModel.model_fields) == _TRAIN_KEYS


# ── Versioning helpers ─────────────────────────────────────────────────────
def test_write_version_stamps_and_repoints_current(tmp_path) -> None:
    env_dir = tmp_path / "env"
    p0 = es.write_env_spec_version(env_dir, es.jump_preset_spec())
    assert p0.name == "v0.json"
    cur = es.read_current_env_spec(env_dir)
    assert cur["meta"]["version"] == "v0"
    # Next write bumps monotonically + repoints.
    spec2 = es.jump_preset_spec()
    spec2["shared"]["episode_length_s"] = 8.0
    p1 = es.write_env_spec_version(env_dir, spec2)
    assert p1.name == "v1.json"
    assert es.read_current_env_spec(env_dir)["shared"]["episode_length_s"] == 8.0
    # Repoint back (revert path).
    es.repoint_env_current(env_dir, "v0")
    cur = es.read_current_env_spec(env_dir)
    assert cur["meta"]["version"] == "v0"
    assert cur["shared"]["episode_length_s"] == 10.0


def test_write_version_refuses_invalid(tmp_path) -> None:
    bad = es.jump_preset_spec()
    bad["train"]["entropy_coef_scale"] = 99.0
    with pytest.raises(ValueError, match="refusing to persist"):
        es.write_env_spec_version(tmp_path / "env", bad)
    assert not (tmp_path / "env" / "current.json").exists()


def test_read_current_none_when_absent(tmp_path) -> None:
    assert es.read_current_env_spec(tmp_path / "env") is None


def test_repoint_missing_version_raises(tmp_path) -> None:
    env_dir = tmp_path / "env"
    es.write_env_spec_version(env_dir, es.jump_preset_spec())
    with pytest.raises(ValueError):
        es.repoint_env_current(env_dir, "v7")


# ── _maybe_seed_env_spec (loop wiring) ─────────────────────────────────────
class _SpecAdapter:
    """Duck-typed adapter with env-spec support."""

    task_id = "Mjlab-Velocity-Flat-Unitree-G1"
    env_profile = ""
    env_spec_path = ""


class _NoSpecAdapter:
    """gym_sb3-style adapter — no env_spec_path attribute."""

    task_id = "Hopper-v4"


def test_seed_env_spec_happy_path(tmp_path, monkeypatch) -> None:
    adapter = _SpecAdapter()
    monkeypatch.setattr(
        "sculptor.env_gen.generate_env_spec",
        lambda **kw: {
            "env_spec_version": 1,
            "meta": {"source": "generated"},
            "shared": {"episode_length_s": 9.0},
            "train": {"entropy_coef_scale": 1.5},
        })
    path = _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="tuck jump", adapter=adapter)
    assert path is not None and path.name == "v0.json"
    assert (tmp_path / "env" / "current.json").is_file()
    assert adapter.env_spec_path == str(
        (tmp_path / "env" / "current.json").resolve())
    cur = es.read_current_env_spec(tmp_path / "env")
    assert cur["shared"]["episode_length_s"] == 9.0


def test_seed_env_spec_skips_when_spec_exists(tmp_path, monkeypatch) -> None:
    es.write_env_spec_version(tmp_path / "env", es.jump_preset_spec())
    adapter = _SpecAdapter()
    monkeypatch.setattr(
        "sculptor.env_gen.generate_env_spec",
        lambda **kw: (_ for _ in ()).throw(AssertionError("LLM consulted")))
    assert _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="g", adapter=adapter) is None
    assert adapter.env_spec_path == ""   # load_adapter convention handles it


def test_seed_env_spec_respects_explicit_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sculptor.env_gen.generate_env_spec",
        lambda **kw: (_ for _ in ()).throw(AssertionError("LLM consulted")))
    a1 = _SpecAdapter()
    a1.env_profile = "jump"
    assert _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="g", adapter=a1) is None
    a2 = _SpecAdapter()
    a2.env_spec_path = "/somewhere/spec.json"
    assert _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="g", adapter=a2) is None
    a3 = _NoSpecAdapter()
    assert _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="g", adapter=a3) is None
    assert not (tmp_path / "env").exists()


def test_seed_env_spec_failure_keeps_defaults(tmp_path, monkeypatch) -> None:
    adapter = _SpecAdapter()
    monkeypatch.setattr(
        "sculptor.env_gen.generate_env_spec",
        lambda **kw: (_ for _ in ()).throw(ConnectionError("network down")))
    assert _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="g", adapter=adapter) is None
    assert adapter.env_spec_path == ""
    assert not (tmp_path / "env" / "current.json").exists()


def test_seed_env_spec_real_client_path_uses_stub(tmp_path) -> None:
    """End-to-end through generate_env_spec with the parse-level stub —
    no monkeypatching of env_gen internals."""
    adapter = _SpecAdapter()
    path = _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="a standing tuck jump",
        adapter=adapter, client=_StubClient(_good_model()))
    assert path is not None
    cur = es.read_current_env_spec(tmp_path / "env")
    assert cur["shared"]["zero_velocity_commands"] is True
    assert cur["meta"]["version"] == "v0"


def test_seed_env_spec_exploding_client_is_safe(tmp_path) -> None:
    adapter = _SpecAdapter()
    assert _maybe_seed_env_spec(
        project=tmp_path, behavior_goal="g", adapter=adapter,
        client=_ExplodingClient()) is None

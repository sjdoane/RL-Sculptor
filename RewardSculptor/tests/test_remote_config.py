"""Unit tests for RemoteConfig (§Ship 23) + the load_adapter `[remote]`
table plumb.

Pure-python — no mjlab, no GPU, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sculptor.adapters.base import (
    ComponentProbe,
    RewardContract,
    RolloutResult,
    SculptorAdapter,
    TrainResult,
)
from sculptor.adapters._remote import RemoteConfig


# ── from_sources: nothing configured ─────────────────────────────────


def test_from_sources_returns_none_when_unconfigured() -> None:
    assert RemoteConfig.from_sources(None, {}) is None
    assert RemoteConfig.from_sources({}, {}) is None
    # Unrelated env vars don't count as configuration.
    assert RemoteConfig.from_sources(None, {"PATH": "/usr/bin"}) is None


# ── from_sources: TOML table ─────────────────────────────────────────


def test_from_sources_parses_toml_table() -> None:
    cfg = RemoteConfig.from_sources(
        {
            "enabled": True,
            "host": "1.2.3.4",
            "port": 41234,
            "user": "root",
            "key_path": "~/.ssh/id_runpod",
            "remote_workdir": "/workspace/sculptor",
            "remote_python": "/workspace/venv/bin/python",
            "device": "cuda:0",
            "rollout_remote": True,
        },
        {},
    )
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.host == "1.2.3.4"
    assert cfg.port == 41234
    assert cfg.user == "root"
    assert cfg.key_path == "~/.ssh/id_runpod"
    assert cfg.remote_workdir == "/workspace/sculptor"
    assert cfg.remote_python == "/workspace/venv/bin/python"
    assert cfg.device == "cuda:0"
    assert cfg.rollout_remote is True
    assert cfg.target == "root@1.2.3.4"


def test_target_without_user_is_bare_host() -> None:
    cfg = RemoteConfig.from_sources({"enabled": True, "host": "pod.example"}, {})
    assert cfg is not None
    assert cfg.target == "pod.example"


# ── from_sources: env overrides win ──────────────────────────────────


def test_env_overrides_win_over_toml() -> None:
    cfg = RemoteConfig.from_sources(
        {"enabled": False, "host": "toml-host", "port": 22, "user": "toml-user"},
        {
            "SCULPTOR_REMOTE_ENABLED": "1",
            "SCULPTOR_REMOTE_HOST": "env-host",
            "SCULPTOR_REMOTE_PORT": "2222",
        },
    )
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.host == "env-host"
    assert cfg.port == 2222
    # Non-overridden TOML fields survive.
    assert cfg.user == "toml-user"


def test_env_only_configuration_works() -> None:
    """The UI backend injects SCULPTOR_REMOTE_* with no TOML at all."""
    cfg = RemoteConfig.from_sources(
        None,
        {
            "SCULPTOR_REMOTE_ENABLED": "true",
            "SCULPTOR_REMOTE_HOST": "9.9.9.9",
            "SCULPTOR_REMOTE_USER": "root",
            "SCULPTOR_REMOTE_KEY_PATH": "/keys/k",
            "SCULPTOR_REMOTE_ROLLOUT": "yes",
        },
    )
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.host == "9.9.9.9"
    assert cfg.rollout_remote is True


def test_empty_env_values_are_ignored() -> None:
    """Empty-string env vars (unset-but-exported) must not mask TOML."""
    cfg = RemoteConfig.from_sources(
        {"enabled": True, "host": "toml-host"},
        {"SCULPTOR_REMOTE_HOST": ""},
    )
    assert cfg is not None
    assert cfg.host == "toml-host"
    assert cfg.enabled is True


# ── enabled forcing + robustness ─────────────────────────────────────


def test_enabled_forced_false_without_host() -> None:
    cfg = RemoteConfig.from_sources({"enabled": True}, {})
    assert cfg is not None
    assert cfg.enabled is False

    cfg2 = RemoteConfig.from_sources(
        None, {"SCULPTOR_REMOTE_ENABLED": "1"},
    )
    assert cfg2 is not None
    assert cfg2.enabled is False


def test_falsy_enabled_strings() -> None:
    for raw in ("0", "false", "no", "off", "", "garbage"):
        cfg = RemoteConfig.from_sources(
            {"enabled": raw, "host": "h"}, {},
        )
        assert cfg is not None
        assert cfg.enabled is False, f"enabled={raw!r} should be falsy"


def test_bad_port_falls_back_to_22() -> None:
    cfg = RemoteConfig.from_sources(
        {"enabled": True, "host": "h", "port": "not-a-port"}, {},
    )
    assert cfg is not None
    assert cfg.port == 22


def test_bad_poll_interval_falls_back() -> None:
    cfg = RemoteConfig.from_sources(
        {"enabled": True, "host": "h", "poll_interval_s": "nan-ish-junk"}, {},
    )
    assert cfg is not None
    # str->float of junk falls back to the default.
    assert cfg.poll_interval_s == 5.0 or cfg.poll_interval_s != cfg.poll_interval_s


# ── observable misconfiguration (never a silent local fallback) ─────


def test_enabled_without_host_emits_observable_warning(capsys) -> None:
    from sculptor.adapters import _remote

    _remote._config_warned.clear()
    cfg = RemoteConfig.from_sources({"enabled": True}, {})
    assert cfg is not None and cfg.enabled is False
    out = capsys.readouterr().out
    assert "remote_config_ignored" in out
    assert "enabled_without_host" in out
    # Once per process — config is re-resolved every train/rollout call.
    RemoteConfig.from_sources({"enabled": True}, {})
    assert "remote_config_ignored" not in capsys.readouterr().out


def test_unrecognized_keys_emit_observable_warning(capsys) -> None:
    from sculptor.adapters import _remote

    _remote._config_warned.clear()
    assert RemoteConfig.from_sources({"hostname": "typo.example"}, {}) is None
    out = capsys.readouterr().out
    assert "remote_config_ignored" in out
    assert "no_recognized_keys" in out


# ── load_adapter `[remote]` plumb ────────────────────────────────────


class _DummyRemoteAdapter(SculptorAdapter):
    """Minimal concrete adapter that records its `remote` kwarg."""

    def __init__(self, remote: dict | None = None, marker: str = "") -> None:
        self.remote = remote
        self.marker = marker

    def train(self, reward_module_path: Path, output_dir: Path, steps: int,
              seed: int, **kw: Any) -> TrainResult:  # pragma: no cover - stub
        raise NotImplementedError

    def rollout(self, checkpoint_path: Path, output_dir: Path,
                n_episodes: int, **kw: Any) -> RolloutResult:  # pragma: no cover
        raise NotImplementedError

    def compute_behavior_metrics(self, rollout: RolloutResult) -> dict[str, Any]:  # pragma: no cover
        return {}

    def reward_contract(self) -> RewardContract:  # pragma: no cover - stub
        return RewardContract(observation_space_spec=None, action_space_spec=None)

    def probe_component(self, reward_module_path: Path) -> ComponentProbe:  # pragma: no cover
        return ComponentProbe(ok=True, components={}, total=0.0, error=None)


class _DummyNoRemoteAdapter(_DummyRemoteAdapter):
    """Adapter WITHOUT a `remote` kwarg — the plumb must skip it."""

    def __init__(self, marker: str = "") -> None:
        super().__init__(remote=None, marker=marker)


def test_load_adapter_plumbs_top_level_remote_table(tmp_path: Path) -> None:
    from sculptor.adapters.base import load_adapter

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[adapter]\n'
        'class = "tests.test_remote_config._DummyRemoteAdapter"\n'
        'config = { marker = "m1" }\n'
        '\n'
        '[remote]\n'
        'enabled = true\n'
        'host = "1.2.3.4"\n'
        'user = "root"\n'
    )
    adapter = load_adapter(cfg)
    assert isinstance(adapter, _DummyRemoteAdapter)
    assert adapter.marker == "m1"
    assert adapter.remote == {"enabled": True, "host": "1.2.3.4", "user": "root"}


def test_load_adapter_without_remote_table_passes_none(tmp_path: Path) -> None:
    from sculptor.adapters.base import load_adapter

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[adapter]\n'
        'class = "tests.test_remote_config._DummyRemoteAdapter"\n'
        'config = { marker = "m2" }\n'
    )
    adapter = load_adapter(cfg)
    assert adapter.remote is None


def test_load_adapter_skips_remote_for_adapters_without_kwarg(tmp_path: Path) -> None:
    """An adapter whose __init__ has no `remote` param (gym_sb3 et al.)
    must instantiate cleanly even when `[remote]` is present."""
    from sculptor.adapters.base import load_adapter

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[adapter]\n'
        'class = "tests.test_remote_config._DummyNoRemoteAdapter"\n'
        'config = { marker = "m3" }\n'
        '\n'
        '[remote]\n'
        'enabled = true\n'
        'host = "1.2.3.4"\n'
    )
    adapter = load_adapter(cfg)
    assert adapter.marker == "m3"
    assert adapter.remote is None


def test_load_adapter_explicit_config_remote_wins(tmp_path: Path) -> None:
    """`remote` already in [adapter].config is NOT clobbered by the
    top-level table (explicit beats plumbed)."""
    from sculptor.adapters.base import load_adapter

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[adapter]\n'
        'class = "tests.test_remote_config._DummyRemoteAdapter"\n'
        '[adapter.config]\n'
        'marker = "m4"\n'
        '[adapter.config.remote]\n'
        'host = "explicit-host"\n'
        '\n'
        '[remote]\n'
        'host = "table-host"\n'
    )
    adapter = load_adapter(cfg)
    assert adapter.remote == {"host": "explicit-host"}

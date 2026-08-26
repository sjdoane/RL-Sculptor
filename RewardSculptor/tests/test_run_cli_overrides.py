"""CLI contract for launch-scoped hardware overrides."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from sculptor.cli import app


def test_run_cli_forwards_num_envs_and_device(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[adapter]\nclass = 'unused.Adapter'\n", encoding="utf-8")
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            iterations_run=1,
            early_stopped=False,
            early_stop_reason=None,
            final_reward_path=None,
        )

    monkeypatch.setattr("sculptor.sculpt.sculpt_run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "run", "traverse rough terrain", "--config", str(config),
            "--iterations", "1", "--num-envs", "512",
            "--device", "cuda:0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["num_envs"] == 512
    assert captured["device"] == "cuda:0"


def test_run_cli_rejects_invalid_device(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[adapter]\nclass = 'unused.Adapter'\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["run", "goal", "--config", str(config), "--device", "gpu:7"],
    )
    assert result.exit_code != 0
    assert "--device must be" in result.output


def test_run_cli_forwards_exact_authored_world_pin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[adapter]\nclass = 'unused.Adapter'\n", encoding="utf-8")
    selection = tmp_path / "selection_v7.json"
    selection.write_text("{}\n", encoding="utf-8")
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            iterations_run=1,
            early_stopped=False,
            early_stop_reason=None,
            final_reward_path=None,
        )

    monkeypatch.setattr("sculptor.sculpt.sculpt_run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "run", "keep exact world", "--config", str(config),
            "--world-selection", str(selection),
            "--expected-world-selection-sha256", "a" * 64,
            "--expected-world-tuple-hash", "b" * 64,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["world_selection_path"] == selection
    assert captured["expected_world_selection_sha256"] == "a" * 64
    assert captured["expected_world_tuple_hash"] == "b" * 64

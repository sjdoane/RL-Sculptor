from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sculptor.cli import app
from tests.test_world_project import _project


def test_world_cli_authors_shows_and_validates_generic_arm(tmp_path: Path) -> None:
    project = _project(tmp_path)
    runner = CliRunner()

    authored = runner.invoke(app, [
        "world", "author",
        "Use a gripper to move a ball into a goal region",
        "--project", str(project),
        "--robot", "yam:parallel_gripper",
        "--yes", "--json",
    ])
    assert authored.exit_code == 0, authored.output
    result = json.loads(authored.stdout)
    assert result["ok"] is True
    assert result["capability_id"] == "yam:parallel_gripper"
    assert result["admission"]["ok"] is True
    assert (project / "env" / "selection_current.json").is_file()

    shown = runner.invoke(app, [
        "world", "show", "--project", str(project), "--json",
    ])
    assert shown.exit_code == 0, shown.output
    bundle = json.loads(shown.stdout)
    assert bundle["world"]["shared"]["robot"]["capability_id"] == (
        "yam:parallel_gripper")
    assert bundle["task"]["shared"]["goal"]["type"] == "object_to_region"

    validated = runner.invoke(app, [
        "world", "validate", "--project", str(project), "--json",
    ])
    assert validated.exit_code == 0, validated.output
    validation = json.loads(validated.stdout)
    assert validation["ok"] is True
    assert validation["model_hash_match"] is True


def test_world_cli_interactive_defaults_and_headless_json_are_clean(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    interactive_project = _project(tmp_path / "interactive")
    interactive = runner.invoke(app, [
        "world", "author",
        "Walk to the finish region",
        "--project", str(interactive_project),
        "--robot", "unitree_g1:base",
        "--interactive", "--json",
    ], input="\n" * 64)
    assert interactive.exit_code == 0, interactive.output
    assert json.loads(interactive.stdout)["ok"] is True

    headless_project = _project(tmp_path / "headless")
    headless = runner.invoke(app, [
        "world", "author",
        "Push the ball into the goal region",
        "--project", str(headless_project),
        "--json",
    ])
    assert headless.exit_code == 0, headless.output
    payload = json.loads(headless.stdout)
    assert payload["ok"] is True
    assert payload["capability_id"] in {
        "unitree_g1:base", "unitree_go1:base",
    }

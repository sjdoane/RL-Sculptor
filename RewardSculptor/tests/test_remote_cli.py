"""`sculpt remote doctor` CLI tests (§Ship 23b). RemoteExecutor.doctor
is mocked — connectivity itself is covered by test_remote_executor.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from sculptor.cli import app

runner = CliRunner()

OK_REPORT = {
    "ok": True,
    "host": "1.2.3.4",
    "port": 22,
    "checks": [
        {"name": "ssh reachable", "ok": True, "detail": "root@1.2.3.4:22 reachable"},
        {"name": "nvidia driver/GPU", "ok": True, "detail": "RTX 5090, 580.65"},
    ],
}
FAIL_REPORT = {
    "ok": False,
    "host": "1.2.3.4",
    "port": 22,
    "checks": [
        {"name": "ssh reachable", "ok": False, "detail": "connection refused"},
    ],
}


def _cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p


def test_doctor_without_any_remote_exits_2(tmp_path: Path, monkeypatch) -> None:
    for k in list(__import__("os").environ):
        if k.startswith("SCULPTOR_REMOTE_"):
            monkeypatch.delenv(k)
    cfg = _cfg(tmp_path, '[adapter]\nclass = "x.Y"\n')
    result = runner.invoke(app, ["remote", "doctor", "--config", str(cfg)])
    assert result.exit_code == 2
    assert "no remote configured" in result.output


def test_doctor_missing_config_file_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["remote", "doctor", "--config", str(tmp_path / "nope.toml")],
    )
    assert result.exit_code == 2
    assert "config not found" in result.output


def test_doctor_malformed_config_exits_2(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "[remote\nthis is not toml")
    result = runner.invoke(app, ["remote", "doctor", "--config", str(cfg)])
    assert result.exit_code == 2
    assert "not parseable" in result.output


def test_doctor_green_exits_0_and_prints_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, '[remote]\nenabled = true\nhost = "1.2.3.4"\nuser = "root"\n')
    with patch(
        "sculptor.adapters._remote.RemoteExecutor.doctor", return_value=OK_REPORT,
    ):
        result = runner.invoke(app, ["remote", "doctor", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "ssh reachable" in result.output
    assert "all checks passed" in result.output


def test_doctor_failure_exits_1(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, '[remote]\nenabled = true\nhost = "1.2.3.4"\n')
    with patch(
        "sculptor.adapters._remote.RemoteExecutor.doctor", return_value=FAIL_REPORT,
    ):
        result = runner.invoke(app, ["remote", "doctor", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "some checks FAILED" in result.output


def test_doctor_json_output_is_machine_readable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, '[remote]\nenabled = true\nhost = "1.2.3.4"\n')
    with patch(
        "sculptor.adapters._remote.RemoteExecutor.doctor", return_value=OK_REPORT,
    ):
        result = runner.invoke(
            app, ["remote", "doctor", "--config", str(cfg), "--json"],
        )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["checks"][0]["name"] == "ssh reachable"


def test_doctor_works_before_enabled_is_flipped(tmp_path: Path) -> None:
    """Doctor must run against `enabled = false` + host — the whole
    point is testing the connection BEFORE turning dispatch on."""
    cfg = _cfg(tmp_path, '[remote]\nenabled = false\nhost = "1.2.3.4"\n')
    with patch(
        "sculptor.adapters._remote.RemoteExecutor.doctor", return_value=OK_REPORT,
    ):
        result = runner.invoke(app, ["remote", "doctor", "--config", str(cfg)])
    assert result.exit_code == 0, result.output

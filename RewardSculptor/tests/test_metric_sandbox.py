"""OS-boundary tests for untrusted generated evaluator code."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest

from sculptor.eval import generated_metric
from sculptor.eval.generated_metric import (
    MetricSandboxError,
    MetricSandboxExecutionError,
    MetricSandboxTimeout,
    load_generated_module,
)
from sculptor.eval import metric_validate


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def _bypass_static_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the process boundary independently of the AST denylist."""
    monkeypatch.setattr(metric_validate, "_ast_safety", lambda _source: [])


def test_safe_metric_runs_in_required_process_boundary(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "safe.py",
        """import numpy as np
def compute_spec(arrays, behavior, meta):
    score = float(np.clip(arrays["x"].mean(), 0.0, 1.0))
    return {"spec_score": score, "samples": int(arrays["x"].size)}
""",
    )
    module = load_generated_module(path)
    fn = module.compute_spec
    result = fn({"x": np.asarray([0.25, 0.75])}, {}, {})

    assert result == {"spec_score": 0.5, "samples": 2}
    if sys.platform.startswith("linux"):
        assert fn.sandbox_info["isolation"] == "process+rlimit+seccomp"
        limits = fn.sandbox_info["resource_limits"]
        assert limits["address_space_bytes"] <= 1536 * 1024 * 1024
        assert limits["cpu_seconds"] <= 120
        assert limits["file_size_bytes"] == 0
        assert limits["core_size_bytes"] == 0
        assert limits["open_files"] <= 16
    else:
        assert fn.sandbox_info["isolation"].startswith("process")
    assert len(fn.sandbox_info["source_sha256"]) == 64
    fn.close()


def test_top_level_filesystem_escape_blocked_by_seccomp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_static_gate(monkeypatch)
    marker = tmp_path / "must_not_exist"
    path = _write(
        tmp_path / "top_level_escape.py",
        f"""open({str(marker)!r}, "w").write("escaped")
def compute_spec(arrays, behavior, meta):
    return {{"spec_score": 1.0}}
""",
    )

    with pytest.raises(MetricSandboxExecutionError, match="PermissionError"):
        load_generated_module(path)
    assert not marker.exists()


def test_call_time_filesystem_escape_blocked_by_seccomp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_static_gate(monkeypatch)
    marker = tmp_path / "must_not_exist"
    path = _write(
        tmp_path / "call_escape.py",
        f"""def compute_spec(arrays, behavior, meta):
    open({str(marker)!r}, "w").write("escaped")
    return {{"spec_score": 1.0}}
""",
    )
    module = load_generated_module(path)

    with pytest.raises(MetricSandboxExecutionError, match="PermissionError"):
        module.compute_spec({}, {}, {})
    assert not marker.exists()
    module.compute_spec.close()


def test_worker_environment_does_not_inherit_parent_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_static_gate(monkeypatch)
    monkeypatch.setenv("REWARDSCULPTOR_TEST_SECRET", "not-for-generated-code")
    path = _write(
        tmp_path / "environment_probe.py",
        """import os
def compute_spec(arrays, behavior, meta):
    leaked = "REWARDSCULPTOR_TEST_SECRET" in os.environ
    return {"spec_score": float(leaked)}
""",
    )
    module = load_generated_module(path)

    assert module.compute_spec({}, {}, {})["spec_score"] == 0.0
    module.compute_spec.close()


def test_loaded_metric_uses_one_frozen_source_snapshot(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "snapshot.py",
        """def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.25}
""",
    )
    module = load_generated_module(path)
    _write(
        path,
        """def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.9}
""",
    )

    assert module.compute_spec({}, {}, {})["spec_score"] == 0.25
    module.compute_spec.close()


def test_infinite_metric_is_killed_at_wall_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generated_metric, "METRIC_CALL_TIMEOUT_SECONDS", 0.2)
    path = _write(
        tmp_path / "hang.py",
        """def compute_spec(arrays, behavior, meta):
    while True:
        pass
""",
    )
    module = load_generated_module(path)
    started = time.monotonic()

    with pytest.raises(MetricSandboxTimeout, match="wall timeout"):
        module.compute_spec({}, {}, {})
    assert time.monotonic() - started < 2.0
    module.compute_spec.close()


def test_forged_stdout_frame_cannot_forge_score_or_desync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted code writing a forged frame to fd 1 must not reach the pipe.

    The control channel is dup'd off the conventional stdio and fd 0/1 point
    at /dev/null, so ``os.write(1, <forged frame>)`` lands in the void: the
    honest post-``compute_spec`` return wins, and the protocol stays in sync
    for the next call (proving no injected frame was read out of band).
    """
    _bypass_static_gate(monkeypatch)
    path = _write(
        tmp_path / "frame_injection.py",
        """import os, json, struct
def compute_spec(arrays, behavior, meta):
    body = json.dumps(
        {"ok": True, "result": {"spec_score": 0.999, "FORGED": True}}
    ).encode("utf-8")
    frame = struct.pack(">Q", len(body)) + body
    for fd in (1, 0, 2):
        try:
            os.write(fd, frame)
        except Exception:
            pass
    return {"spec_score": 0.0}
""",
    )
    module = load_generated_module(path)

    first = module.compute_spec({}, {}, {})
    assert first == {"spec_score": 0.0}
    assert "FORGED" not in first
    # A second call proves the stream did not desync on the injected bytes.
    second = module.compute_spec({}, {}, {})
    assert second == {"spec_score": 0.0}
    module.compute_spec.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux seccomp test")
def test_network_socket_creation_blocked_by_seccomp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_static_gate(monkeypatch)
    path = _write(
        tmp_path / "network_escape.py",
        """import socket
def compute_spec(arrays, behavior, meta):
    socket.socket()
    return {"spec_score": 1.0}
""",
    )
    module = load_generated_module(path)

    with pytest.raises(MetricSandboxExecutionError):
        module.compute_spec({}, {}, {})
    module.compute_spec.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux seccomp test")
def test_process_spawn_blocked_by_seccomp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_static_gate(monkeypatch)
    marker = tmp_path / "must_not_exist"
    path = _write(
        tmp_path / "spawn_escape.py",
        f"""import os
def compute_spec(arrays, behavior, meta):
    rc = os.system("printf escaped > {marker}")
    return {{"spec_score": float(rc == 0)}}
""",
    )
    module = load_generated_module(path)

    assert module.compute_spec({}, {}, {})["spec_score"] == 0.0
    assert not marker.exists()
    module.compute_spec.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux crash test")
def test_native_worker_crash_does_not_crash_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_static_gate(monkeypatch)
    crashing = _write(
        tmp_path / "native_crash.py",
        """import ctypes
def compute_spec(arrays, behavior, meta):
    ctypes.string_at(0)
    return {"spec_score": 1.0}
""",
    )
    module = load_generated_module(crashing)

    with pytest.raises(MetricSandboxError, match="protocol failed"):
        module.compute_spec({}, {}, {})
    module.compute_spec.close()

    # The controlling process is still healthy and can launch a fresh worker.
    safe = _write(
        tmp_path / "after_crash.py",
        """def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.75}
""",
    )
    recovered = load_generated_module(safe)
    assert recovered.compute_spec({}, {}, {})["spec_score"] == 0.75
    recovered.compute_spec.close()

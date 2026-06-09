"""RemoteExecutor lifecycle tests (§Ship 23) against a scripted fake
transport. No network, no GPU, no mjlab.

The FakeCommandRunner emulates the remote host as a directory tree under
tmp_path ("remote /" = `<tmp>/remote_fs/`), interpreting the exact
ssh/rsync command shapes the executor builds. The heaviest coverage is
on remote process lifecycle — the part that burns rented GPUs when it
goes wrong: reattach-vs-kill on cmd-hash, kill-on-exception, the
exitcode file as source of truth, and the ordered atomic download
(completion key LAST).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from sculptor.adapters._remote import (
    _EVENT_TAG,
    _NODIR,
    _POLL_SEP,
    RemoteConfig,
    RemoteDispatchError,
    RemoteExecutor,
    RunnerJob,
)


# Matches a possibly-shlex-quoted path token in a composed ssh command
# ("'/dir with space/x'" or "/plain/x"); unquote with _unq.
_PATH = r"('.*?'|\S+)"


def _unq(token: str) -> str:
    import shlex
    return shlex.split(token)[0]


def _local_version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:  # noqa: BLE001
        return "0.0-unknown"


class FakeRemote:
    """State of the emulated remote host."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.home = "/root"
        self.alive_pgids: set[int] = set()
        self._next_pgid = 4242
        self.launch_count = 0
        self.kill_count = 0
        self.ssh_down = False
        self.fail_polls = 0          # next N polls fail at the ssh layer
        # Per-poll script: list of (exitcode_or_None, [new stdout lines]).
        # When exitcode is not None the fake also writes stderr_content
        # and finish_artifacts into the job's output dir (the real
        # runner's behavior: artifacts exist by the time exitcode does).
        self.poll_plan: list[tuple[int | None, list[str]]] = []
        self.stderr_content = ""
        self.finish_artifacts: dict[str, bytes] = {}
        # Default to the local versions so no skew event fires unless a
        # test asks for one.
        self.versions = {"torch": _local_version("torch"),
                         "mjlab": _local_version("mjlab")}

    def path(self, posix: str) -> Path:
        return self.root / posix.lstrip("/")

    def seed_job(self, jd_posix: str, cmd_json: str, alive: bool) -> int:
        """Pre-create a job dir as if a previous dispatch left it."""
        jd = self.path(jd_posix)
        jd.mkdir(parents=True, exist_ok=True)
        (jd / "cmd.json").write_text(cmd_json)
        pgid = self._next_pgid
        self._next_pgid += 1
        (jd / "pgid").write_text(str(pgid))
        if alive:
            self.alive_pgids.add(pgid)
        return pgid


class FakeCommandRunner:
    """Interprets the executor's ssh/rsync argv against the FakeRemote."""

    def __init__(self, remote: FakeRemote) -> None:
        self.remote = remote
        self.calls: list[list[str]] = []

    # -- helpers ------------------------------------------------------
    def _done(self, argv, rc=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(argv, rc, stdout, stderr)

    def _strip_target(self, spec: str) -> str:
        # "root@fake-host:'/abs/path'" -> "/abs/path" — the executor
        # shell-quotes the remote side of every rsync spec (H1 fix),
        # so un-quote the way the remote shell would.
        import shlex
        raw = spec.split(":", 1)[1]
        return shlex.split(raw)[0] if raw.strip() else raw

    def run(self, argv, *, timeout=None, input_text=None):
        self.calls.append(list(argv))
        if argv[0] == "rsync":
            return self._rsync(argv)
        assert argv[0] == "ssh", f"unexpected transport: {argv[0]}"
        if self.remote.ssh_down:
            return self._done(argv, rc=255, stderr="ssh: connect refused")
        return self._ssh(argv, argv[-1], input_text)

    # -- rsync emulation ----------------------------------------------
    def _rsync(self, argv):
        if self.remote.ssh_down:
            return self._done(argv, rc=255, stderr="rsync: connection refused")
        src, dst = argv[-2], argv[-1]
        if ":" in dst:  # upload
            rpath = self._strip_target(dst)
            if rpath.rstrip("/").endswith("/code/sculptor"):
                # Code sync — record only; tests don't need 50 files copied.
                self.remote.path(rpath.rstrip("/")).mkdir(parents=True, exist_ok=True)
                return self._done(argv)
            local = Path(src)
            target = self.remote.path(rpath)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, target)
            return self._done(argv)
        # download: remote dir -> local staging
        rpath = self._strip_target(src).rstrip("/")
        src_dir = self.remote.path(rpath)
        dst_dir = Path(dst.rstrip("/"))
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.is_dir():
            return self._done(argv, rc=23, stderr="rsync: no such dir")
        excludes = set()
        for i, a in enumerate(argv):
            if a == "--exclude":
                excludes.add(argv[i + 1])
        for entry in src_dir.iterdir():
            if entry.name in excludes:
                continue
            t = dst_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, t, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, t)
        return self._done(argv)

    # -- ssh emulation -------------------------------------------------
    def _ssh(self, argv, cmd: str, input_text):
        r = self.remote
        if cmd == "true":
            return self._done(argv)
        if cmd == "echo $HOME":
            return self._done(argv, stdout=r.home + "\n")
        if "importlib.metadata" in cmd:
            return self._done(argv, stdout=json.dumps(r.versions) + "\n")
        if "torch.cuda.is_available" in cmd:
            return self._done(argv, stdout='{"cuda": true, "torch": "x"}\n')
        if "command -v rsync" in cmd:
            return self._done(argv, stdout="/usr/bin/rsync\n")
        if "--version" in cmd:
            return self._done(argv, stdout="Python 3.13.5\n")
        if "nvidia-smi" in cmd:
            return self._done(
                argv, stdout="NVIDIA GeForce RTX 5090, 32607 MiB, 580.65\n",
            )
        if cmd.startswith("df -h") or " df -h" in cmd:
            return self._done(argv, stdout="/dev/vda1 100G 20G 80G 20% /\n")

        m = re.search(rf"cat > {_PATH}/run\.sh", cmd)
        if m:
            jd = r.path(_unq(m.group(1)))
            jd.mkdir(parents=True, exist_ok=True)
            (jd / "run.sh").write_text(input_text or "")
            return self._done(argv)
        m = re.search(rf"cat > {_PATH}/cmd\.json", cmd)
        if m:
            (r.path(_unq(m.group(1))) / "cmd.json").write_text(input_text or "")
            return self._done(argv)

        if "setsid nohup bash run.sh" in cmd:
            m = re.search(rf"cd {_PATH} &&", cmd)
            assert m, cmd
            jd = r.path(_unq(m.group(1)))
            for f in ("exitcode", "stdout.log", "stderr.log", "pgid"):
                (jd / f).unlink(missing_ok=True)
            pgid = r._next_pgid
            r._next_pgid += 1
            (jd / "pgid").write_text(str(pgid))
            r.alive_pgids.add(pgid)
            r.launch_count += 1
            return self._done(argv, stdout=f"{pgid}\n")

        if "NOJOB" in cmd:  # stale-job check
            m = re.search(rf"cd {_PATH} 2>/dev/null", cmd)
            assert m, cmd
            jd = r.path(_unq(m.group(1)))
            if not jd.is_dir():
                return self._done(argv, stdout="NOJOB\n")
            if (jd / "exitcode").is_file():
                return self._done(argv, stdout="DONE\n")
            pgid_file = jd / "pgid"
            if not pgid_file.is_file():
                return self._done(argv, stdout="NOJOB\n")
            pgid = int(pgid_file.read_text().strip())
            if pgid in r.alive_pgids:
                import hashlib
                sha = hashlib.sha256((jd / "cmd.json").read_bytes()).hexdigest()
                return self._done(argv, stdout=f"ALIVE\n{sha}\n")
            return self._done(argv, stdout="DEAD\n")

        if "kill -TERM" in cmd:
            m = re.search(rf"cat {_PATH}/pgid", cmd)
            assert m, cmd
            r.kill_count += 1
            pgid_file = r.path(_unq(m.group(1))) / "pgid"
            if pgid_file.is_file():
                r.alive_pgids.discard(int(pgid_file.read_text().strip()))
            return self._done(argv)

        if _POLL_SEP in cmd:
            if r.fail_polls > 0:
                r.fail_polls -= 1
                return self._done(argv, rc=255, stderr="ssh: timed out")
            m = re.search(rf"cat {_PATH}/exitcode", cmd)
            mo = re.search(r"tail -c \+(\d+)", cmd)
            assert m and mo, cmd
            jd = r.path(_unq(m.group(1)))
            if not jd.is_dir():  # `else echo NODIR` branch
                return self._done(argv, stdout=_NODIR + "\n")
            # Advance the scripted job by one step.
            if r.poll_plan:
                exitcode, new_lines = r.poll_plan.pop(0)
                log = jd / "stdout.log"
                with log.open("a") as f:
                    for ln in new_lines:
                        f.write(ln + "\n")
                if exitcode is not None:
                    (jd / "stderr.log").write_text(r.stderr_content)
                    out_dir = jd.parent
                    for name, data in r.finish_artifacts.items():
                        (out_dir / name).write_bytes(data)
                    (jd / "exitcode").write_text(f"{exitcode}\n")
                    pgid_file = jd / "pgid"
                    if pgid_file.is_file():
                        r.alive_pgids.discard(int(pgid_file.read_text().strip()))
            head = ""
            if (jd / "exitcode").is_file():
                head = (jd / "exitcode").read_text()
            offset = int(mo.group(1))  # 1-based byte offset (tail -c +N)
            tail_text = ""
            if (jd / "stdout.log").is_file():
                data = (jd / "stdout.log").read_bytes()
                tail_text = data[offset - 1:].decode("utf-8")
            return self._done(argv, stdout=head + _POLL_SEP + "\n" + tail_text)

        m = re.search(rf"tail -c (\d+) {_PATH}/stderr\.log", cmd)
        if m:
            f = r.path(_unq(m.group(2))) / "stderr.log"
            return self._done(argv, stdout=f.read_text() if f.is_file() else "")

        if cmd.startswith("mkdir -p"):
            import shlex as _shlex
            for tok in _shlex.split(cmd)[2:]:
                r.path(tok).mkdir(parents=True, exist_ok=True)
            return self._done(argv)

        raise AssertionError(f"FakeCommandRunner: unhandled ssh command: {cmd!r}")


# ── fixtures / helpers ───────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_version_cache():
    RemoteExecutor._version_probe_cache.clear()
    yield
    RemoteExecutor._version_probe_cache.clear()


def make_executor(tmp_path: Path, **overrides) -> tuple[RemoteExecutor, FakeRemote]:
    fake = FakeRemote(tmp_path / "remote_fs")
    table = {
        "enabled": True,
        "host": "fake-host",
        "user": "root",
        "poll_interval_s": 0.0,
        **overrides,
    }
    cfg = RemoteConfig.from_sources(table, {})
    assert cfg is not None and cfg.enabled
    ex = RemoteExecutor(cfg, runner=FakeCommandRunner(fake), sleep_fn=lambda s: None)
    return ex, fake


def make_job(tmp_path: Path, **kw) -> RunnerJob:
    out = tmp_path / "proj" / "missions" / "m1" / "iter_3"
    out.mkdir(parents=True, exist_ok=True)
    reward = tmp_path / "proj" / "rewards" / "current.py"
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text("# reward module\n")
    defaults = dict(
        subcommand="train",
        options={"--task-id": "Mjlab-Test", "--num-envs": "8"},
        input_paths={"--reward-module-path": reward},
        output_dir=out,
        required_artifacts=("metrics.json", "checkpoint.pt"),
        remote_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    defaults.update(kw)
    return RunnerJob(**defaults)


def events_of(capsys) -> list[dict]:
    out = capsys.readouterr().out
    evs = []
    for line in out.splitlines():
        if line.startswith(_EVENT_TAG + " "):
            evs.append(json.loads(line[len(_EVENT_TAG) + 1:]))
    return evs


HAPPY_ARTIFACTS = {"metrics.json": b'{"ok": true}', "checkpoint.pt": b"CKPT" * 64}


# ── path mapping ─────────────────────────────────────────────────────


def test_mirror_maps_absolute_paths(tmp_path: Path) -> None:
    ex, _ = make_executor(tmp_path)
    local = tmp_path / "a" / "b" / "iter_0"
    mirrored = ex._mirror(local)
    assert mirrored == str(
        PurePosixPath("/root/.sculptor_remote/mirror")
        / str(local.resolve()).replace("\\", "/").lstrip("/")
    )
    # Stable: same input → same mirror (warm-start re-upload no-op relies on it).
    assert ex._mirror(local) == mirrored


def test_iter_hint_parsed_from_output_dir(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    assert job.iter_hint == 3


def test_remote_python_tilde_expanded(tmp_path: Path) -> None:
    """`remote_python = "~/venv/bin/python"` is shlex-quoted into ssh
    commands and run.sh — quoting suppresses tilde expansion, so the
    executor must expand it against the remote $HOME itself."""
    ex, _ = make_executor(tmp_path, remote_python="~/venv/bin/python")
    job = make_job(tmp_path)
    argv = ex._remote_argv(job)
    assert argv[0] == "/root/venv/bin/python"


# ── happy path ───────────────────────────────────────────────────────


def test_execute_happy_path(tmp_path: Path, capsys) -> None:
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [
        (None, ["Learning iteration 1/2", '[SCULPT-EVENT] {"type": "iter_progress"}']),
        (0, ["Learning iteration 2/2"]),
    ]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)

    proc = ex.execute(job)

    assert proc.returncode == 0
    # Artifacts synced into the SAME local output_dir.
    assert (job.output_dir / "checkpoint.pt").read_bytes() == HAPPY_ARTIFACTS["checkpoint.pt"]
    assert (job.output_dir / "metrics.json").read_bytes() == HAPPY_ARTIFACTS["metrics.json"]
    # Staging dir cleaned up.
    assert not (job.output_dir / ".remote_incoming").exists()
    # Input mirrored to its absolute-path mirror location.
    reward_local = job.input_paths["--reward-module-path"]
    assert fake.path(ex._mirror(reward_local)).read_text() == "# reward module\n"
    # Remote stdout re-emitted locally (collected + printed).
    assert "Learning iteration 1/2" in proc.stdout
    assert "Learning iteration 2/2" in proc.stdout

    types = [e["type"] for e in events_of(capsys)]
    for expected in (
        "remote_dispatch_started", "remote_upload_completed",
        "remote_job_launched", "remote_job_finished", "remote_artifacts_synced",
    ):
        assert expected in types, f"missing {expected} in {types}"
    assert "remote_dispatch_failed" not in types


def test_launch_script_shape(tmp_path: Path) -> None:
    """run.sh must self-record its pgid, pin PYTHONPATH at the synced
    code, export the job env, and write `exitcode` atomically."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)
    ex.execute(job)

    jd = fake.path(ex._job_dir(job))
    run_sh = (jd / "run.sh").read_text()
    assert "echo $$ >" in run_sh                       # pgid self-record
    assert "/root/.sculptor_remote/code" in run_sh     # PYTHONPATH at synced code
    assert "PYTHONUNBUFFERED=1" in run_sh
    assert "WANDB_MODE=${WANDB_MODE:-disabled}" in run_sh
    assert "export CUDA_VISIBLE_DEVICES=0" in run_sh   # job remote_env
    assert "exitcode.tmp" in run_sh and "mv" in run_sh  # atomic exitcode

    # cmd.json is the verbatim argv (the reattach hash source) with
    # mirrored input + output paths.
    argv = json.loads((jd / "cmd.json").read_text())
    assert argv[:3] == ["python3", "-m", "sculptor.adapters._mjlab_runner"]
    assert "train" in argv
    assert ex._mirror(job.input_paths["--reward-module-path"]) in argv
    assert ex._mirror(job.output_dir) in argv


# ── reattach vs kill ─────────────────────────────────────────────────


def test_reattach_on_matching_cmd_hash(tmp_path: Path, capsys) -> None:
    """A still-running remote job with an identical cmd.json is the
    same logical dispatch (local crash + restart) → reattach, never
    double-launch."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    argv = ex._remote_argv(job)
    fake.seed_job(ex._job_dir(job), json.dumps(argv), alive=True)
    fake.poll_plan = [(0, ["resumed line"])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)

    proc = ex.execute(job)

    assert proc.returncode == 0
    assert fake.launch_count == 0, "reattach must not launch a second job"
    types = [e["type"] for e in events_of(capsys)]
    assert "remote_job_reattached" in types
    assert "remote_stale_job_killed" not in types


def test_stale_job_killed_on_hash_mismatch(tmp_path: Path, capsys) -> None:
    """A still-running job with a DIFFERENT command is stale — kill it
    before launching ours (no orphaned GPU burn, no double-train)."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.seed_job(ex._job_dir(job), json.dumps(["python3", "-m", "other", "cmd"]),
                  alive=True)
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)

    proc = ex.execute(job)

    assert proc.returncode == 0
    assert fake.kill_count >= 1
    assert fake.launch_count == 1
    types = [e["type"] for e in events_of(capsys)]
    assert "remote_stale_job_killed" in types
    assert "remote_job_reattached" not in types


def test_finished_job_dir_does_not_block_relaunch(tmp_path: Path) -> None:
    """An exitcode file from a PREVIOUS run means no live job — fresh
    launch, no kill."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    jd = fake.path(ex._job_dir(job))
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "exitcode").write_text("0\n")
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)

    proc = ex.execute(job)
    assert proc.returncode == 0
    assert fake.launch_count == 1
    assert fake.kill_count == 0


# ── streaming / polling ──────────────────────────────────────────────


def test_poll_offset_no_duplicate_lines(tmp_path: Path) -> None:
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [
        (None, ["L1", "L2"]),
        (None, ["L3"]),
        (None, []),          # idle poll — nothing new
        (0, ["L4"]),
    ]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)
    proc = ex.execute(job)
    assert proc.stdout.splitlines() == ["L1", "L2", "L3", "L4"]


def test_connection_lost_retries_and_recovers(tmp_path: Path, capsys) -> None:
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(0, ["done"])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)
    # First poll attempt fails at the ssh layer; the job survives by
    # construction and the next poll finds it.
    original_run = ex.runner.run
    state = {"launched": False, "failed": False}

    def flaky_run(argv, *, timeout=None, input_text=None):
        if (
            state["launched"] and not state["failed"]
            and argv[0] == "ssh" and _POLL_SEP in argv[-1]
        ):
            state["failed"] = True
            return subprocess.CompletedProcess(argv, 255, "", "ssh: broken pipe")
        result = original_run(argv, timeout=timeout, input_text=input_text)
        if argv[0] == "ssh" and "setsid nohup" in argv[-1]:
            state["launched"] = True
        return result

    ex.runner = type("R", (), {"run": staticmethod(flaky_run)})()
    proc = ex.execute(job)
    assert proc.returncode == 0
    types = [e["type"] for e in events_of(capsys)]
    assert "remote_connection_lost" in types
    assert "remote_artifacts_synced" in types


def test_gives_up_after_max_poll_failures_without_killing(tmp_path: Path) -> None:
    """Unreachable host: raise ssh_unreachable, and do NOT try to kill —
    the job is reattachable once the host comes back."""
    ex, fake = make_executor(tmp_path, reattach_max_failures=2)
    job = make_job(tmp_path)
    fake.poll_plan = []          # never finishes
    # Launch succeeds, then the host vanishes.
    original_run = ex.runner.run

    def run_then_die(argv, *, timeout=None, input_text=None):
        if argv[0] == "ssh" and _POLL_SEP in argv[-1]:
            return subprocess.CompletedProcess(argv, 255, "", "ssh: timed out")
        return original_run(argv, timeout=timeout, input_text=input_text)

    ex.runner = type("R", (), {"run": staticmethod(run_then_die)})()
    with pytest.raises(RemoteDispatchError) as ei:
        ex.execute(job)
    assert ei.value.reason == "ssh_unreachable"
    assert fake.kill_count == 0


# ── atomic download ──────────────────────────────────────────────────


def test_missing_required_artifact_raises_and_leaves_no_partial(
    tmp_path: Path, capsys,
) -> None:
    """Zero-byte metrics.json → artifacts_missing; the local output_dir
    must NOT contain checkpoint.pt (a partial state that resume logic
    could mistake for a finished train)."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(0, [])]
    fake.stderr_content = "Traceback: something exploded before metrics\n"
    fake.finish_artifacts = {
        "metrics.json": b"",                      # required but empty
        "checkpoint.pt": b"CKPT",                 # present and non-empty
    }
    with pytest.raises(RemoteDispatchError) as ei:
        ex.execute(job)
    assert ei.value.reason == "artifacts_missing"
    assert "metrics.json" in ei.value.detail
    assert "exploded" in ei.value.detail          # remote stderr tail included
    assert not (job.output_dir / "checkpoint.pt").exists()
    assert not (job.output_dir / "metrics.json").exists()
    assert not (job.output_dir / ".remote_incoming").exists()
    types = [e["type"] for e in events_of(capsys)]
    assert "remote_dispatch_failed" in types


def test_completion_key_promoted_last(tmp_path: Path, monkeypatch) -> None:
    """Promotion order: aux files first, then required artifacts in
    declared order — checkpoint.pt (the resume key) strictly last."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = {
        **HAPPY_ARTIFACTS,
        "reward_trajectory.json": b"[]",          # aux, non-required
    }
    order: list[str] = []
    real_replace = Path.replace

    def recording_replace(self: Path, target):
        order.append(Path(target).name)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    ex.execute(job)
    promoted = [n for n in order
                if n in ("reward_trajectory.json", "metrics.json", "checkpoint.pt")]
    assert promoted[-1] == "checkpoint.pt"
    assert promoted.index("metrics.json") < promoted.index("checkpoint.pt")
    assert promoted.index("reward_trajectory.json") < promoted.index("metrics.json")


# ── failure classification ───────────────────────────────────────────


def test_oom_classified_from_stderr(tmp_path: Path, capsys) -> None:
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(1, ["dying"])]
    fake.stderr_content = "torch.cuda.OutOfMemoryError: CUDA out of memory.\n"
    proc = ex.execute(job)                        # nonzero exit does NOT raise
    assert proc.returncode == 1
    assert "out of memory" in proc.stderr.lower()
    evs = events_of(capsys)
    failed = [e for e in evs if e["type"] == "remote_dispatch_failed"]
    assert failed and failed[0]["reason"] == "remote_oom"
    # No download attempted on failure.
    assert not (job.output_dir / "checkpoint.pt").exists()


def test_runner_failure_classified_generic(tmp_path: Path, capsys) -> None:
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(3, [])]
    fake.stderr_content = "ValueError: bad reward module\n"
    proc = ex.execute(job)
    assert proc.returncode == 3
    evs = events_of(capsys)
    failed = [e for e in evs if e["type"] == "remote_dispatch_failed"]
    assert failed and failed[0]["reason"] == "runner_failed"
    assert "bad reward module" in failed[0]["detail"]


# ── kill-on-exception (mirrors _run_with_cleanup) ────────────────────


def test_local_exception_kills_remote_job(tmp_path: Path) -> None:
    """A KeyboardInterrupt mid-wait (UI cancel / Ctrl-C) must SIGTERM
    the remote process group — never leave a rented GPU burning."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(None, ["still running"])]  # never finishes

    def interrupting_sleep(s):
        raise KeyboardInterrupt

    ex._sleep = interrupting_sleep
    with pytest.raises(KeyboardInterrupt):
        ex.execute(job)
    assert fake.kill_count >= 1
    assert not fake.alive_pgids, "remote pgid should be dead after kill"


def test_preflight_unreachable_raises_before_anything(tmp_path: Path) -> None:
    ex, fake = make_executor(tmp_path)
    fake.ssh_down = True
    job = make_job(tmp_path)
    with pytest.raises(RemoteDispatchError) as ei:
        ex.execute(job)
    assert ei.value.reason == "ssh_unreachable"
    assert fake.launch_count == 0


# ── version skew ─────────────────────────────────────────────────────


def test_version_skew_emits_event_but_proceeds(tmp_path: Path, capsys) -> None:
    ex, fake = make_executor(tmp_path)
    fake.versions = {"torch": "0.0.1-skewed", "mjlab": "0.0.1-skewed"}
    job = make_job(tmp_path)
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)
    proc = ex.execute(job)
    assert proc.returncode == 0                   # warn, don't block
    types = [e["type"] for e in events_of(capsys)]
    assert "remote_version_skew" in types


# ── doctor ───────────────────────────────────────────────────────────


def test_doctor_all_green(tmp_path: Path) -> None:
    ex, _ = make_executor(tmp_path)
    report = ex.doctor()
    assert report["host"] == "fake-host"
    names = [c["name"] for c in report["checks"]]
    assert "ssh reachable" in names
    assert "nvidia driver/GPU" in names
    assert "torch.cuda available" in names
    # Everything except (possibly) version-match is green; with the fake
    # mirroring local versions, all must pass.
    assert report["ok"] is True, report["checks"]


def test_doctor_ssh_down_short_circuits(tmp_path: Path) -> None:
    ex, fake = make_executor(tmp_path)
    fake.ssh_down = True
    report = ex.doctor()
    assert report["ok"] is False
    names = [c["name"] for c in report["checks"]]
    # Remote-side checks are skipped when ssh is down.
    assert "nvidia driver/GPU" not in names
    assert any(c["name"] == "ssh reachable" and not c["ok"] for c in report["checks"])


# ── review-fix regressions (Ship 23a audit) ──────────────────────────


def test_stale_check_transport_failure_raises_not_nojob(tmp_path: Path) -> None:
    """C1: an ssh failure during the stale-job check must NOT be read
    as 'no job' — launching over a possibly-live job double-dispatches
    and clobbers its pgid file (unkillable orphan on a rented GPU)."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    original_run = ex.runner.run

    def run(argv, *, timeout=None, input_text=None):
        if argv[0] == "ssh" and "NOJOB" in argv[-1]:
            return subprocess.CompletedProcess(argv, 255, "", "ssh: connection reset")
        return original_run(argv, timeout=timeout, input_text=input_text)

    ex.runner = type("R", (), {"run": staticmethod(run)})()
    with pytest.raises(RemoteDispatchError) as ei:
        ex.execute(job)
    assert ei.value.reason == "ssh_unreachable"
    assert fake.launch_count == 0
    assert fake.kill_count == 0          # unreachable → reattachable, no kill


def test_paths_with_spaces_quoted_in_rsync_specs(tmp_path: Path) -> None:
    """H1: the remote side of an rsync spec is shell-evaluated — paths
    with spaces must arrive quoted, end to end."""
    ex, fake = make_executor(tmp_path)
    out = tmp_path / "dir with space" / "iter_1"
    out.mkdir(parents=True)
    reward = tmp_path / "dir with space" / "current.py"
    reward.write_text("# r")
    job = RunnerJob(
        subcommand="train",
        options={"--task-id": "T"},
        input_paths={"--reward-module-path": reward},
        output_dir=out,
        required_artifacts=("metrics.json", "checkpoint.pt"),
    )
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)

    proc = ex.execute(job)
    assert proc.returncode == 0
    assert (out / "checkpoint.pt").exists()

    remote_specs = [
        a for call in ex.runner.calls if call[0] == "rsync"
        for a in call[-2:] if a.startswith("root@fake-host:")
    ]
    spaced = [s for s in remote_specs if " " in s]
    assert spaced, "expected remote rsync specs containing the spaced path"
    for spec in spaced:
        assert spec.split(":", 1)[1].startswith("'"), f"unquoted remote path: {spec}"


def test_reattach_skips_upload(tmp_path: Path, capsys) -> None:
    """M1: reattaching to a live matching job must not re-upload (the
    job has its inputs; a sync blip here must never kill it)."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    argv = ex._remote_argv(job)
    fake.seed_job(ex._job_dir(job), json.dumps(argv), alive=True)
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)
    ex.execute(job)
    types = [e["type"] for e in events_of(capsys)]
    assert "remote_job_reattached" in types
    assert "remote_upload_completed" not in types
    assert not any(c[0] == "rsync" and c[-1].startswith("root@")
                   for c in ex.runner.calls), "reattach must not rsync uploads"


def test_partial_trailing_line_buffered_until_complete(tmp_path: Path) -> None:
    """M2: a partially-flushed line (no newline yet) must be buffered,
    never emitted truncated — a split [SCULPT-EVENT] JSON line would
    corrupt the UI event stream."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(None, []), (None, []), (0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)
    jd_local = fake.path(ex._job_dir(job))
    original_run = ex.runner.run
    state = {"polls": 0}

    def run(argv, *, timeout=None, input_text=None):
        if argv[0] == "ssh" and _POLL_SEP in argv[-1] and jd_local.is_dir():
            state["polls"] += 1
            if state["polls"] == 1:
                (jd_local / "stdout.log").write_bytes(
                    b'[SCULPT-EVENT] {"type": "iter_pro'
                )
            elif state["polls"] == 2:
                with (jd_local / "stdout.log").open("ab") as f:
                    f.write(b'gress"}\n')
        return original_run(argv, timeout=timeout, input_text=input_text)

    ex.runner = type("R", (), {"run": staticmethod(run)})()
    proc = ex.execute(job)
    assert proc.returncode == 0
    assert proc.stdout.count('[SCULPT-EVENT] {"type": "iter_progress"}') == 1


def test_download_rsync_failure_raises_sync_failed(tmp_path: Path) -> None:
    """T5: a failed artifact download must raise sync_failed and leave
    neither promoted artifacts nor a stale staging dir."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = [(0, [])]
    fake.finish_artifacts = dict(HAPPY_ARTIFACTS)
    original_run = ex.runner.run

    def run(argv, *, timeout=None, input_text=None):
        if argv[0] == "rsync" and argv[-2].startswith("root@fake-host:"):
            return subprocess.CompletedProcess(
                argv, 23, "", "rsync: connection unexpectedly closed",
            )
        return original_run(argv, timeout=timeout, input_text=input_text)

    ex.runner = type("R", (), {"run": staticmethod(run)})()
    with pytest.raises(RemoteDispatchError) as ei:
        ex.execute(job)
    assert ei.value.reason == "sync_failed"
    assert not (job.output_dir / "checkpoint.pt").exists()
    assert not (job.output_dir / ".remote_incoming").exists()


def test_vanished_job_dir_fails_fast_as_job_lost(tmp_path: Path, capsys) -> None:
    """M4: a wiped job dir (spot reclaim) is job_lost — fail fast with
    a distinct reason instead of burning the reattach budget on
    misleading connection_lost retries."""
    ex, fake = make_executor(tmp_path)
    job = make_job(tmp_path)
    fake.poll_plan = []
    original_run = ex.runner.run

    def run(argv, *, timeout=None, input_text=None):
        if argv[0] == "ssh" and _POLL_SEP in argv[-1]:
            shutil.rmtree(fake.path(ex._job_dir(job)), ignore_errors=True)
        return original_run(argv, timeout=timeout, input_text=input_text)

    ex.runner = type("R", (), {"run": staticmethod(run)})()
    with pytest.raises(RemoteDispatchError) as ei:
        ex.execute(job)
    assert ei.value.reason == "job_lost"
    types = [e["type"] for e in events_of(capsys)]
    assert "remote_connection_lost" not in types
    assert "remote_dispatch_failed" in types


def test_doctor_emits_no_events_even_on_skew(tmp_path: Path, capsys) -> None:
    """H2: `sculpt remote doctor --json` pipes stdout into json.loads
    in the UI backend — doctor must never print [SCULPT-EVENT] lines,
    even when the version probe finds skew (the report row carries it)."""
    ex, fake = make_executor(tmp_path)
    fake.versions = {"torch": "0.0.1-skewed", "mjlab": "0.0.1-skewed"}
    report = ex.doctor()
    assert report["ok"] is False  # skew row fails
    skew_rows = [c for c in report["checks"] if c["name"] == "torch/mjlab version match"]
    assert skew_rows and not skew_rows[0]["ok"]
    assert _EVENT_TAG not in capsys.readouterr().out


def test_doctor_never_raises(tmp_path: Path) -> None:
    class ExplodingRunner:
        def run(self, argv, *, timeout=None, input_text=None):
            raise OSError("transport meltdown")

    cfg = RemoteConfig.from_sources({"enabled": True, "host": "h"}, {})
    assert cfg is not None
    ex = RemoteExecutor(cfg, runner=ExplodingRunner(), sleep_fn=lambda s: None)
    report = ex.doctor()
    assert report["ok"] is False

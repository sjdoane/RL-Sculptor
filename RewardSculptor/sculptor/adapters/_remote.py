"""sculptor/adapters/_remote.py — SSH+rsync remote dispatch for the
GPU-heavy runner steps (Ship 23).

Design (NEXT_LEVEL_BRIEF §8 + the Ship-23 plan): the mjlab train /
rollout step is already a self-contained subprocess whose inputs are
small files (reward .py, optional warm-start checkpoint) and whose
outputs are files in `output_dir`. Remote dispatch therefore:

  1. mirrors local absolute paths under `<workdir>/mirror/...` on the
     pod (so warm-start checkpoints *born* remotely re-upload as
     zero-byte rsync no-ops),
  2. rsyncs the local `sculptor/` package source to the pod and runs
     the runner with PYTHONPATH pointing at it (sculptor version skew
     is structurally impossible),
  3. launches the runner detached (`setsid nohup`, file-backed
     stdout/stderr/exitcode) so it survives SSH disconnects,
  4. polls one combined ssh call per interval: exitcode + new stdout
     lines, re-emitting them on the local stdout so `[SCULPT-EVENT]`
     progress reaches the UI exactly like the local tee does,
  5. downloads artifacts into a staging dir and promotes them with the
     completion-key artifact (checkpoint.pt) moved LAST, so resume
     logic can never mistake a partial sync for a finished train.

Lifecycle safety (the part that bites — see plan §0):
  * any local exception (UI cancel, Ctrl-C) SIGTERMs the remote
    process group over SSH, mirroring `_run_with_cleanup`;
  * a stale remote job is REATTACHED when its `cmd.json` hash matches
    the new dispatch (idempotent recovery after a local crash) and
    killed when it doesn't (no double-dispatch);
  * the on-remote `exitcode` file — never the SSH channel — is the
    single source of truth for job completion.

Everything here is observable: each phase emits a `[SCULPT-EVENT]`
line; failures raise `RemoteDispatchError(reason=...)` which callers
surface through the existing recoverable iteration-failure paths.
No daemons, no job queue — the smallest thing that works reliably.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Protocol

_EVENT_TAG = "[SCULPT-EVENT]"
_POLL_SEP = "__RS_REMOTE_SEP__"
_NODIR = "__RS_REMOTE_NODIR__"
_RUNNER_MODULE = "sculptor.adapters._mjlab_runner"

#: Once-per-process guard for config-misconfiguration warnings (a
#: config is re-resolved on every train/rollout call — warn once).
_config_warned: set[str] = set()

#: Env-var override names (these win over the `[remote]` TOML table —
#: the UI backend's injection path).
_ENV_PREFIX = "SCULPTOR_REMOTE_"
_ENV_KEYS = {
    "enabled": _ENV_PREFIX + "ENABLED",
    "host": _ENV_PREFIX + "HOST",
    "port": _ENV_PREFIX + "PORT",
    "user": _ENV_PREFIX + "USER",
    "key_path": _ENV_PREFIX + "KEY_PATH",
    "remote_workdir": _ENV_PREFIX + "WORKDIR",
    "remote_python": _ENV_PREFIX + "PYTHON",
    "device": _ENV_PREFIX + "DEVICE",
    "rollout_remote": _ENV_PREFIX + "ROLLOUT",
}
_TRUTHY = {"1", "true", "yes", "on"}


def _emit(payload: dict[str, Any]) -> None:
    """Print a `[SCULPT-EVENT]` JSON line on stdout — same wire format
    as sculpt.py's `_emit_event` / the runner's progress lines, so the
    UI's stdout-event passthrough picks these up with no backend change.
    Never raises."""
    try:
        line = _EVENT_TAG + " " + json.dumps(payload, default=str)
    except Exception:  # noqa: BLE001
        return
    print(line, flush=True)


def _warn_config_ignored(reason: str, detail: str) -> None:
    """Observable (once-per-process) warning when a `[remote]` config
    is present but resolves to disabled — never a silent local fallback."""
    if reason in _config_warned:
        return
    _config_warned.add(reason)
    _emit({"type": "remote_config_ignored", "reason": reason, "detail": detail})


class RemoteDispatchError(RuntimeError):
    """A dispatch-level failure (NOT a nonzero runner exit — that flows
    back as a normal CompletedProcess). `reason` is machine-readable:
    ssh_unreachable | launch_failed | sync_failed | artifacts_missing
    | job_lost (remote job dir vanished — e.g. spot-instance reclaim).
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"remote dispatch failed ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class RemoteConfig:
    """Resolved remote-execution settings. Built via `from_sources`
    (TOML `[remote]` table + `SCULPTOR_REMOTE_*` env overrides, env
    wins). `enabled` is forced False when no host is configured."""

    enabled: bool = False
    host: str = ""
    port: int = 22
    user: str = ""
    key_path: Optional[str] = None
    remote_workdir: str = "~/.sculptor_remote"
    remote_python: str = "python3"
    device: Optional[str] = None
    rollout_remote: bool = False
    connect_timeout_s: int = 15
    poll_interval_s: float = 5.0
    reattach_max_failures: int = 60  # ~5 min of unreachability at 5s polls

    @classmethod
    def from_sources(
        cls,
        toml_table: Optional[Mapping[str, Any]],
        environ: Mapping[str, str],
    ) -> Optional["RemoteConfig"]:
        """Merge the `[remote]` TOML table with env overrides. Returns
        None when neither source configures anything (the common local
        case — callers fall through to the local subprocess path)."""
        merged: dict[str, Any] = {}
        if toml_table:
            for f in cls.__dataclass_fields__:
                if f in toml_table:
                    merged[f] = toml_table[f]
        env_seen = False
        for field_name, env_key in _ENV_KEYS.items():
            raw = environ.get(env_key)
            if raw is None or raw == "":
                continue
            env_seen = True
            merged[field_name] = raw
        if not merged and not env_seen:
            # A non-empty [remote] table whose keys are ALL unrecognized
            # (typos like `hostname = ...`) must not silently fall back
            # to local training — warn once, observably.
            if toml_table:
                _warn_config_ignored(
                    "no_recognized_keys",
                    f"[remote] table keys {sorted(toml_table)} are not "
                    f"recognized; remote dispatch stays disabled",
                )
            return None

        def _bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in _TRUTHY

        def _int(key: str, default: int) -> int:
            try:
                return int(merged.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        try:
            poll = float(merged.get("poll_interval_s", 5.0))
        except (TypeError, ValueError):
            poll = 5.0

        host = str(merged.get("host", "") or "").strip()
        enabled_requested = _bool(merged.get("enabled", False))
        if enabled_requested and not host:
            _warn_config_ignored(
                "enabled_without_host",
                "remote.enabled is true but no host is configured; "
                "training will run LOCALLY",
            )
        enabled = enabled_requested and bool(host)
        return cls(
            enabled=enabled,
            host=host,
            port=_int("port", 22),
            user=str(merged.get("user", "") or "").strip(),
            key_path=str(merged["key_path"]) if merged.get("key_path") else None,
            remote_workdir=str(merged.get("remote_workdir", "~/.sculptor_remote")),
            remote_python=str(merged.get("remote_python", "python3")),
            device=str(merged["device"]) if merged.get("device") else None,
            rollout_remote=_bool(merged.get("rollout_remote", False)),
            connect_timeout_s=_int("connect_timeout_s", 15),
            poll_interval_s=poll,
            reattach_max_failures=_int("reattach_max_failures", 60),
        )

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


class CommandRunner(Protocol):
    """Injectable transport — tests substitute a fake that scripts the
    'remote' filesystem under a tmp_path. The default shells out."""

    def run(
        self,
        argv: list[str],
        *,
        timeout: Optional[float] = None,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess: ...


class SubprocessCommandRunner:
    """Default transport: runs `ssh` / `rsync` as local subprocesses."""

    def run(
        self,
        argv: list[str],
        *,
        timeout: Optional[float] = None,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
        )


@dataclass
class RunnerJob:
    """One dispatchable runner invocation (train or rollout).

    `options` are plain flag→value pairs passed through verbatim;
    `input_paths` are flag→local-Path pairs that get uploaded to their
    mirror locations and rewritten to remote paths in the argv;
    `required_artifacts` is the ORDERED promotion list — the last entry
    is the completion key (moved into place last; for train that's
    `checkpoint.pt`, which is also sculpt.py's resume key)."""

    subcommand: str                                # "train" | "rollout"
    options: dict[str, str]                        # e.g. {"--task-id": ...}
    input_paths: dict[str, Path]                   # flag -> local file
    output_dir: Path                               # local dir, mirrored
    required_artifacts: tuple[str, ...]            # ordered; last == completion key
    remote_env: dict[str, str] = field(default_factory=dict)
    output_flag: str = "--output-dir"
    iter_hint: Optional[int] = None                # parsed "iter_<i>" if any
    #: Directories mirrored wholesale (no argv rewrite). Needed when an
    #: input file loads siblings at import time — sculpt's
    #: rewards/current.py is a shim that exec's its neighbouring
    #: v<N>.py, so the whole rewards/ dir must exist on the pod.
    aux_dirs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.iter_hint is None:
            name = Path(self.output_dir).name
            if name.startswith("iter_"):
                try:
                    self.iter_hint = int(name.split("_", 1)[1])
                except (ValueError, IndexError):
                    self.iter_hint = None


class RemoteExecutor:
    """Executes a RunnerJob on the configured SSH host. One instance
    per adapter; per-process caches (remote $HOME, version probe) keyed
    on the instance."""

    # Class-level so repeated adapter instantiations in one sculpt run
    # don't re-probe versions per iteration.
    _version_probe_cache: dict[str, dict[str, str]] = {}

    def __init__(
        self,
        cfg: RemoteConfig,
        runner: Optional[CommandRunner] = None,
        sleep_fn=time.sleep,
    ) -> None:
        if not cfg.enabled:
            raise ValueError("RemoteExecutor requires an enabled RemoteConfig")
        self.cfg = cfg
        self.runner: CommandRunner = runner or SubprocessCommandRunner()
        self._sleep = sleep_fn
        self._remote_home: Optional[str] = None

    # ── argv builders ────────────────────────────────────────────────
    def _ssh_argv(self, remote_cmd: str) -> list[str]:
        argv = [
            "ssh",
            "-o", f"ConnectTimeout={self.cfg.connect_timeout_s}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(self.cfg.port),
        ]
        if self.cfg.key_path:
            argv += ["-i", str(Path(self.cfg.key_path).expanduser())]
        argv += [self.cfg.target, remote_cmd]
        return argv

    def _rsync_ssh_opt(self) -> str:
        parts = [
            "ssh",
            f"-o ConnectTimeout={self.cfg.connect_timeout_s}",
            "-o BatchMode=yes",
            "-o StrictHostKeyChecking=accept-new",
            f"-p {self.cfg.port}",
        ]
        if self.cfg.key_path:
            parts.append(f"-i {shlex.quote(str(Path(self.cfg.key_path).expanduser()))}")
        return " ".join(parts)

    def _rsync_argv(self, src: str, dst: str, *extra: str) -> list[str]:
        # -rltz, NOT -az: archive mode implies -o/-g/-p (chown/chmod),
        # which network filesystems like RunPod's /workspace mfs reject
        # even for root ("chown ... Operation not permitted", exit 23).
        # We only need recursive + symlinks + mtimes (warm-start no-op
        # detection) + compression.
        return ["rsync", "-rltz", "-e", self._rsync_ssh_opt(), *extra, src, dst]

    def _remote_spec(self, posix_path: str) -> str:
        """`host:path` rsync spec with the path shell-quoted — the
        REMOTE side of an rsync transfer is word-split and glob-expanded
        by the remote shell, so unquoted spaces in mirrored absolute
        paths would silently break every sync."""
        return f"{self.cfg.target}:{shlex.quote(posix_path)}"

    # ── path mapping ─────────────────────────────────────────────────
    def _home(self) -> str:
        if self._remote_home is None:
            proc = self.runner.run(self._ssh_argv("echo $HOME"), timeout=30)
            if proc.returncode != 0 or not (proc.stdout or "").strip():
                raise RemoteDispatchError(
                    "ssh_unreachable",
                    f"cannot resolve remote $HOME on {self.cfg.target}: "
                    f"{(proc.stderr or '')[-400:]}",
                )
            self._remote_home = proc.stdout.strip().splitlines()[-1]
        return self._remote_home

    def _workdir(self) -> str:
        wd = self.cfg.remote_workdir
        if wd == "~" or wd.startswith("~/"):
            wd = self._home() + wd[1:]
        return str(PurePosixPath(wd))

    def _remote_python(self) -> str:
        """remote_python with a leading `~` expanded against the remote
        $HOME — the path gets shlex-quoted into ssh commands and into
        run.sh, where quoting would otherwise suppress tilde expansion."""
        p = self.cfg.remote_python
        if p == "~" or p.startswith("~/"):
            p = self._home() + p[1:]
        return p

    def _mirror(self, local: Path) -> str:
        """`<workdir>/mirror/<absolute local path>` — stable across
        iterations and mission stages, so a checkpoint born remotely is
        a zero-byte re-upload when it later appears as a warm-start."""
        rel = str(Path(local).resolve()).replace("\\", "/").lstrip("/")
        return str(PurePosixPath(self._workdir()) / "mirror" / rel)

    def _job_dir(self, job: RunnerJob) -> str:
        return str(PurePosixPath(self._mirror(job.output_dir)) / ".remote_job")

    # ── events ───────────────────────────────────────────────────────
    def _event(self, type_: str, job: Optional[RunnerJob] = None, **kw: Any) -> None:
        payload: dict[str, Any] = {"type": type_, "host": self.cfg.host}
        if job is not None:
            payload["phase"] = job.subcommand
            payload["output_dir"] = str(job.output_dir)
            if job.iter_hint is not None:
                payload["iter"] = job.iter_hint
        payload.update(kw)
        _emit(payload)

    # ── phases ───────────────────────────────────────────────────────
    def _preflight(self) -> None:
        proc = self.runner.run(self._ssh_argv("true"), timeout=self.cfg.connect_timeout_s + 15)
        if proc.returncode != 0:
            raise RemoteDispatchError(
                "ssh_unreachable",
                f"ssh preflight to {self.cfg.target}:{self.cfg.port} failed: "
                f"{(proc.stderr or '')[-400:]}",
            )

    def _check_version_skew(
        self, job: Optional[RunnerJob] = None, emit: bool = True,
    ) -> None:
        """Probe remote torch/mjlab versions; on mismatch emit a
        `remote_version_skew` event (suppressed with emit=False — the
        doctor path computes skew itself and must keep `--json` stdout
        pure for the UI backend)."""
        cache_key = f"{self.cfg.target}:{self.cfg.port}"
        if cache_key in self._version_probe_cache:
            return
        probe = (
            "import json\n"
            "import importlib.metadata as m\n"
            "out = {}\n"
            "for pkg in ('torch', 'mjlab'):\n"
            "    try:\n"
            "        out[pkg] = m.version(pkg)\n"
            "    except Exception as e:\n"
            "        out[pkg] = f'ERROR: {type(e).__name__}'\n"
            "print(json.dumps(out))\n"
        )
        cmd = f"{shlex.quote(self._remote_python())} -c {shlex.quote(probe)}"
        proc = self.runner.run(self._ssh_argv(cmd), timeout=60)
        remote: dict[str, str] = {}
        try:
            remote = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        if not remote:
            # Don't cache a failed probe — a transient ssh blip would
            # otherwise disable skew detection for the whole process.
            return
        self._version_probe_cache[cache_key] = remote
        local: dict[str, str] = {}
        from importlib.metadata import version as _v
        for pkg in ("torch", "mjlab"):
            try:
                local[pkg] = _v(pkg)
            except Exception:  # noqa: BLE001
                local[pkg] = "unknown"
        skewed = {
            p: (local.get(p), remote.get(p))
            for p in ("torch", "mjlab")
            if remote.get(p) and local.get(p) and remote[p] != local[p]
        }
        if skewed and emit:
            self._event(
                "remote_version_skew", job,
                local_torch=local.get("torch"), remote_torch=remote.get("torch"),
                local_mjlab=local.get("mjlab"), remote_mjlab=remote.get("mjlab"),
            )

    @staticmethod
    def _cmd_hash(argv: list[str]) -> str:
        return hashlib.sha256(json.dumps(argv).encode("utf-8")).hexdigest()

    def _stale_job_action(self, job: RunnerJob, cmd_hash: str) -> str:
        """Returns 'reattach' | 'killed' | 'none'. A running remote job
        whose cmd.json hash matches this dispatch is the SAME logical
        job (local crash + restart) → reattach instead of double-
        dispatching. Anything else still alive gets killed."""
        jd = self._job_dir(job)
        check = (
            f"cd {shlex.quote(jd)} 2>/dev/null || {{ echo NOJOB; exit 0; }}; "
            f"if [ -f exitcode ]; then echo DONE; exit 0; fi; "
            f"PG=$(cat pgid 2>/dev/null); "
            f"if [ -z \"$PG\" ]; then echo NOJOB; exit 0; fi; "
            f"if kill -0 -- -\"$PG\" 2>/dev/null; then "
            f"echo ALIVE; sha256sum cmd.json 2>/dev/null | cut -d' ' -f1; "
            f"else echo DEAD; rm -f pgid; fi"
        )
        proc = self.runner.run(self._ssh_argv(check), timeout=60)
        if proc.returncode != 0:
            # The check script always exits 0 by construction — a
            # nonzero rc means SSH itself failed. Treating that as
            # NOJOB would double-dispatch onto a possibly-live job
            # (and `rm -f pgid` at launch would orphan it unkillably).
            raise RemoteDispatchError(
                "ssh_unreachable",
                f"stale-job check failed: {(proc.stderr or '')[-400:]}",
            )
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        state = lines[0] if lines else "NOJOB"
        if state != "ALIVE":
            return "none"
        running_hash = lines[1] if len(lines) > 1 else ""
        # cmd.json hash on remote is sha of the file CONTENT — we store
        # the argv-json verbatim, so content sha == our _cmd_hash of it.
        if running_hash and running_hash == cmd_hash:
            self._event("remote_job_reattached", job)
            return "reattach"
        self._kill_remote(job)
        self._event("remote_stale_job_killed", job)
        return "killed"

    def _upload(self, job: RunnerJob) -> None:
        t0 = time.monotonic()
        # 1. sculptor package source → <workdir>/code/sculptor.
        pkg_dir = Path(__file__).resolve().parent.parent  # .../sculptor
        code_dst = str(PurePosixPath(self._workdir()) / "code")
        mk = self.runner.run(
            self._ssh_argv(f"mkdir -p {shlex.quote(code_dst)}"), timeout=60,
        )
        if mk.returncode != 0:
            raise RemoteDispatchError(
                "sync_failed", f"mkdir {code_dst}: {(mk.stderr or '')[-400:]}",
            )
        proc = self.runner.run(
            self._rsync_argv(
                str(pkg_dir) + "/",
                self._remote_spec(f"{code_dst}/sculptor/"),
                "--delete",
                "--exclude", "__pycache__",
            ),
            timeout=600,
        )
        if proc.returncode != 0:
            raise RemoteDispatchError(
                "sync_failed",
                f"rsync sculptor package failed: {(proc.stderr or '')[-400:]}",
            )
        # 2. Input files → mirror paths; 3. output mirror dir.
        dirs = {str(PurePosixPath(self._mirror(job.output_dir)))}
        for p in job.input_paths.values():
            dirs.add(str(PurePosixPath(self._mirror(p)).parent))
        for d in job.aux_dirs:
            dirs.add(str(PurePosixPath(self._mirror(d))))
        mk = self.runner.run(
            self._ssh_argv("mkdir -p " + " ".join(shlex.quote(d) for d in sorted(dirs))),
            timeout=60,
        )
        if mk.returncode != 0:
            raise RemoteDispatchError(
                "sync_failed", f"mkdir mirror dirs: {(mk.stderr or '')[-400:]}",
            )
        for flag, local in job.input_paths.items():
            local = Path(local)
            if not local.is_file():
                raise RemoteDispatchError(
                    "sync_failed", f"input for {flag} not found locally: {local}",
                )
            proc = self.runner.run(
                self._rsync_argv(str(local), self._remote_spec(self._mirror(local))),
                timeout=1800,
            )
            if proc.returncode != 0:
                raise RemoteDispatchError(
                    "sync_failed",
                    f"rsync {local} failed: {(proc.stderr or '')[-400:]}",
                )
        for d in job.aux_dirs:
            d = Path(d)
            if not d.is_dir():
                raise RemoteDispatchError(
                    "sync_failed", f"aux dir not found locally: {d}",
                )
            proc = self.runner.run(
                self._rsync_argv(
                    str(d) + "/",
                    self._remote_spec(self._mirror(d) + "/"),
                    "--exclude", "__pycache__",
                ),
                timeout=1800,
            )
            if proc.returncode != 0:
                raise RemoteDispatchError(
                    "sync_failed",
                    f"rsync aux dir {d} failed: {(proc.stderr or '')[-400:]}",
                )
        self._event(
            "remote_upload_completed", job,
            seconds=round(time.monotonic() - t0, 2),
        )

    def _remote_argv(self, job: RunnerJob) -> list[str]:
        argv = [self._remote_python(), "-m", _RUNNER_MODULE, job.subcommand]
        for flag, value in job.options.items():
            argv += [flag, str(value)]
        for flag, local in job.input_paths.items():
            argv += [flag, self._mirror(Path(local))]
        argv += [job.output_flag, self._mirror(job.output_dir)]
        return argv

    def _launch(self, job: RunnerJob, argv: list[str], cmd_hash: str) -> None:
        jd = self._job_dir(job)
        env_lines = "".join(
            f"export {k}={shlex.quote(v)}\n" for k, v in sorted(job.remote_env.items())
        )
        run_sh = (
            "#!/bin/bash\n"
            f"echo $$ > {shlex.quote(jd)}/pgid\n"
            f"export PYTHONPATH={shlex.quote(str(PurePosixPath(self._workdir()) / 'code'))}\n"
            "export PYTHONUNBUFFERED=1\n"
            "export WANDB_MODE=${WANDB_MODE:-disabled}\n"
            f"{env_lines}"
            f"cd {shlex.quote(self._mirror(job.output_dir))}\n"
            f"{shlex.join(argv)} > {shlex.quote(jd)}/stdout.log 2> {shlex.quote(jd)}/stderr.log\n"
            f"rc=$?\n"
            f"echo $rc > {shlex.quote(jd)}/exitcode.tmp "
            f"&& mv {shlex.quote(jd)}/exitcode.tmp {shlex.quote(jd)}/exitcode\n"
        )
        # Write run.sh + cmd.json (stdin), then launch detached. run.sh
        # writes its own pgid: under `setsid` it IS the session leader,
        # so $$ == pgid and the job survives any SSH disconnect.
        proc = self.runner.run(
            self._ssh_argv(
                f"mkdir -p {shlex.quote(jd)} && cat > {shlex.quote(jd)}/run.sh "
                f"&& chmod +x {shlex.quote(jd)}/run.sh"
            ),
            timeout=60,
            input_text=run_sh,
        )
        if proc.returncode != 0:
            raise RemoteDispatchError(
                "launch_failed", f"writing run.sh: {(proc.stderr or '')[-400:]}",
            )
        proc = self.runner.run(
            self._ssh_argv(f"cat > {shlex.quote(jd)}/cmd.json"),
            timeout=60,
            input_text=json.dumps(argv),
        )
        if proc.returncode != 0:
            raise RemoteDispatchError(
                "launch_failed", f"writing cmd.json: {(proc.stderr or '')[-400:]}",
            )
        # `rm` runs SYNCHRONOUSLY before launch (a bare trailing `&`
        # would background the whole && list, racing the pgid read
        # against the rm), and the pgid read polls up to 10s instead of
        # a fixed sleep — a loaded pod can take >0.5s to start bash.
        launch = (
            f"cd {shlex.quote(jd)} && rm -f exitcode stdout.log stderr.log pgid && "
            f"( setsid nohup bash run.sh </dev/null >/dev/null 2>&1 & ); "
            f"for _ in $(seq 1 40); do [ -s pgid ] && break; sleep 0.25; done; "
            f"cat {shlex.quote(jd)}/pgid 2>/dev/null"
        )
        proc = self.runner.run(self._ssh_argv(launch), timeout=60)
        pgid = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else ""
        if proc.returncode != 0 or not pgid.isdigit():
            raise RemoteDispatchError(
                "launch_failed",
                f"could not launch / read pgid: rc={proc.returncode} "
                f"stdout={(proc.stdout or '')[-200:]} stderr={(proc.stderr or '')[-400:]}",
            )
        self._event("remote_job_launched", job, pgid=int(pgid), cmd_hash=cmd_hash[:12])

    def _kill_remote(self, job: RunnerJob) -> None:
        """Best-effort SIGTERM→SIGKILL of the remote process group.
        Mirrors `_run_with_cleanup`'s killpg so a UI cancel can never
        leave a rented GPU burning.

        The KILL escalation is detached (`setsid ... &`) on the pod:
        the local process is often inside its own 5 s kill-grace window
        (backend SIGTERM→SIGKILL) when this runs, and an in-channel
        `sleep 2` could be cut off with the ssh client — TERM returns
        immediately, the escalation survives channel close."""
        jd = self._job_dir(job)
        cmd = (
            f"PG=$(cat {shlex.quote(jd)}/pgid 2>/dev/null); "
            f"if [ -n \"$PG\" ]; then "
            f"kill -TERM -- -\"$PG\" 2>/dev/null; "
            f"setsid bash -c \"sleep 2; kill -KILL -- -$PG 2>/dev/null\" "
            f"</dev/null >/dev/null 2>&1 & "
            f"fi; true"
        )
        try:
            self.runner.run(self._ssh_argv(cmd), timeout=60)
        except Exception:  # noqa: BLE001 — best-effort by design
            pass

    def _poll_once(self, job: RunnerJob, byte_offset: int) -> tuple[Optional[int], str]:
        """One combined round trip: (exitcode or None, raw new stdout
        text since `byte_offset`, 1-based as in `tail -c +N`).

        Byte offsets, not line offsets: `tail -n` counts only `\\n`
        while Python's splitlines() also splits on `\\r` etc., so
        carriage-return progress bars (tqdm, warp compile) would
        desynchronize a line-based offset and silently drop output.
        Raises job_lost when the job dir itself has vanished (e.g.
        spot-instance reclaim wiped ephemeral storage) — retrying
        polls against a missing dir would just burn the backoff budget.
        """
        jd = self._job_dir(job)
        cmd = (
            f"if [ -d {shlex.quote(jd)} ]; then "
            f"cat {shlex.quote(jd)}/exitcode 2>/dev/null; "
            f"echo {_POLL_SEP}; "
            f"tail -c +{byte_offset} {shlex.quote(jd)}/stdout.log 2>/dev/null; "
            f"else echo {_NODIR}; fi"
        )
        proc = self.runner.run(self._ssh_argv(cmd), timeout=120)
        if proc.returncode != 0:
            raise RemoteDispatchError(
                "ssh_unreachable", f"poll failed: {(proc.stderr or '')[-300:]}",
            )
        out = proc.stdout or ""
        if out.strip().startswith(_NODIR):
            raise RemoteDispatchError(
                "job_lost",
                f"remote job dir vanished ({jd}) — pod restarted or "
                f"ephemeral storage was reclaimed; the stage will retrain",
            )
        head, sep, tail = out.partition(_POLL_SEP + "\n")
        if not sep:
            head, sep, tail = out.partition(_POLL_SEP)
        exitcode: Optional[int] = None
        head = head.strip()
        if head:
            try:
                exitcode = int(head.splitlines()[-1])
            except ValueError:
                exitcode = None
        return exitcode, tail

    def _wait(self, job: RunnerJob) -> tuple[int, str]:
        """Poll until the remote `exitcode` file exists. New stdout
        is re-emitted on local stdout (live UI progress) and collected
        for the CompletedProcess; only COMPLETE lines are emitted (a
        partial trailing line is buffered until its newline arrives, so
        a `[SCULPT-EVENT]` JSON line can never reach the UI truncated).
        SSH failures back off and retry — the job itself survives them
        by construction."""
        byte_offset = 1
        pending = ""
        collected: list[str] = []
        failures = 0
        while True:
            try:
                exitcode, chunk = self._poll_once(job, byte_offset)
                failures = 0
            except RemoteDispatchError as e:
                if e.reason != "ssh_unreachable":
                    raise
                failures += 1
                if failures > self.cfg.reattach_max_failures:
                    raise RemoteDispatchError(
                        "ssh_unreachable",
                        f"gave up after {failures} consecutive poll failures: {e.detail}",
                    ) from e
                self._event(
                    "remote_connection_lost", job,
                    attempt=failures,
                    retry_in_s=min(30.0, self.cfg.poll_interval_s * 2),
                )
                self._sleep(min(30.0, self.cfg.poll_interval_s * 2))
                continue
            if chunk:
                byte_offset += len(chunk.encode("utf-8"))
                pending += chunk
                complete, nl, rest = pending.rpartition("\n")
                if nl:
                    for ln in complete.split("\n"):
                        print(ln, flush=True)
                    collected.append(complete + "\n")
                    pending = rest
            if exitcode is not None:
                if pending:  # final line without trailing newline
                    print(pending, flush=True)
                    collected.append(pending)
                return exitcode, "".join(collected)
            self._sleep(self.cfg.poll_interval_s)

    def _stderr_tail(self, job: RunnerJob, n_bytes: int = 4000) -> str:
        jd = self._job_dir(job)
        try:
            proc = self.runner.run(
                self._ssh_argv(f"tail -c {int(n_bytes)} {shlex.quote(jd)}/stderr.log 2>/dev/null"),
                timeout=60,
            )
            return proc.stdout or ""
        except Exception:  # noqa: BLE001
            return ""

    def _download(self, job: RunnerJob) -> None:
        """Stage artifacts under `.remote_incoming/`, then promote with
        the completion-key artifact LAST. A failure mid-promotion can
        therefore never leave a state that looks complete."""
        t0 = time.monotonic()
        local_out = Path(job.output_dir)
        staging = local_out / ".remote_incoming"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        proc = self.runner.run(
            self._rsync_argv(
                self._remote_spec(self._mirror(job.output_dir) + "/"),
                str(staging) + "/",
                "--exclude", ".remote_job",
            ),
            timeout=3600,
        )
        if proc.returncode != 0:
            shutil.rmtree(staging, ignore_errors=True)
            raise RemoteDispatchError(
                "sync_failed",
                f"artifact download failed: {(proc.stderr or '')[-400:]}",
            )
        missing = [
            a for a in job.required_artifacts
            if not (staging / a).is_file() or (staging / a).stat().st_size == 0
        ]
        if missing:
            stderr_tail = self._stderr_tail(job)
            shutil.rmtree(staging, ignore_errors=True)
            raise RemoteDispatchError(
                "artifacts_missing",
                f"remote run produced no {missing}; remote stderr tail:\n{stderr_tail[-1500:]}",
            )
        required = set(job.required_artifacts)
        # 1. everything that is NOT a required artifact (logs/, aux json …)
        for entry in sorted(staging.iterdir()):
            if entry.name in required:
                continue
            target = local_out / entry.name
            if entry.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                entry.rename(target)
            else:
                entry.replace(target)
        # 2. required artifacts in declared order — completion key last.
        for name in job.required_artifacts:
            (staging / name).replace(local_out / name)
        shutil.rmtree(staging, ignore_errors=True)
        self._event(
            "remote_artifacts_synced", job,
            seconds=round(time.monotonic() - t0, 2),
        )

    # ── public API ───────────────────────────────────────────────────
    def execute(self, job: RunnerJob) -> subprocess.CompletedProcess:
        """Run the job remotely; return a CompletedProcess shaped like
        `_run_with_cleanup`'s so callers' post-subprocess validation
        runs unchanged. Raises RemoteDispatchError for dispatch-level
        failures; a nonzero RUNNER exit flows back via returncode."""
        self._event("remote_dispatch_started", job)
        self._preflight()
        self._check_version_skew(job)
        t_start = time.monotonic()
        # The UI cancels runs with a bare SIGTERM, which terminates
        # CPython WITHOUT unwinding — the except-BaseException kill
        # below would never run and the rented-GPU job would orphan
        # (verified live on a RunPod 5090). Convert SIGTERM to
        # SystemExit for the duration of the dispatch; restore after.
        # Main-thread only — elsewhere signal.signal raises ValueError
        # and we fall back to reattach-or-stale-kill on next dispatch.
        import signal as _signal

        prev_term = None
        installed = False
        try:
            def _raise_exit(signum: int, frame: Any) -> None:  # noqa: ARG001
                raise SystemExit(143)

            prev_term = _signal.signal(_signal.SIGTERM, _raise_exit)
            installed = True
        except (ValueError, OSError):
            pass
        try:
            # Stale-job check BEFORE upload: on reattach (local crash +
            # restart onto a still-running matching job) we skip the
            # upload entirely — the job already has its inputs, and a
            # transient rsync failure here must not trigger the
            # kill-on-exception path against work worth preserving.
            argv = self._remote_argv(job)
            cmd_hash = self._cmd_hash(argv)
            action = self._stale_job_action(job, cmd_hash)
            if action != "reattach":
                self._upload(job)
                self._launch(job, argv, cmd_hash)
            exitcode, stdout_text = self._wait(job)
            self._event(
                "remote_job_finished", job,
                exit_code=exitcode,
                seconds=round(time.monotonic() - t_start, 2),
            )
            stderr_text = ""
            if exitcode != 0:
                stderr_text = self._stderr_tail(job)
                low = stderr_text.lower()
                if "out of memory" in low or "outofmemoryerror" in low:
                    self._event(
                        "remote_dispatch_failed", job,
                        reason="remote_oom", detail=stderr_text[-600:],
                    )
                else:
                    self._event(
                        "remote_dispatch_failed", job,
                        reason="runner_failed", detail=stderr_text[-600:],
                    )
            else:
                self._download(job)
            return subprocess.CompletedProcess(
                argv, exitcode, stdout=stdout_text, stderr=stderr_text,
            )
        except BaseException as e:
            # Mirror _run_with_cleanup: ANY local exception (cancel,
            # Ctrl-C, our own dispatch errors) must not leave a rented
            # GPU burning. Reattachable jobs are the exception we keep:
            # an ssh_unreachable timeout means we CAN'T kill it anyway.
            if not (isinstance(e, RemoteDispatchError) and e.reason == "ssh_unreachable"):
                self._kill_remote(job)
            if isinstance(e, RemoteDispatchError):
                self._event(
                    "remote_dispatch_failed", job,
                    reason=e.reason, detail=e.detail[-600:],
                )
            raise
        finally:
            if installed:
                try:
                    _signal.signal(_signal.SIGTERM, prev_term)
                except (ValueError, OSError):
                    pass

    def doctor(self) -> dict[str, Any]:
        """Connectivity / environment checks for `sculpt remote doctor`
        and the UI's Test-connection button. Never raises — each check
        is `{name, ok, detail}`."""
        checks: list[dict[str, Any]] = []

        def _check(name: str, fn) -> None:
            try:
                ok, detail = fn()
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"{type(e).__name__}: {e}"
            checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

        def _ssh_ok():
            proc = self.runner.run(self._ssh_argv("true"), timeout=self.cfg.connect_timeout_s + 15)
            return proc.returncode == 0, (
                f"{self.cfg.target}:{self.cfg.port} reachable" if proc.returncode == 0
                else (proc.stderr or "ssh failed")[-300:]
            )

        def _rsync_local():
            ok = shutil.which("rsync") is not None and shutil.which("ssh") is not None
            return ok, "ssh + rsync on PATH" if ok else "ssh and/or rsync missing locally (WSL/Linux required)"

        def _rsync_remote():
            proc = self.runner.run(self._ssh_argv("command -v rsync"), timeout=30)
            out = (proc.stdout or "").strip()
            return bool(out), out or "rsync not found on remote"

        def _python():
            cmd = f"{shlex.quote(self._remote_python())} --version"
            proc = self.runner.run(self._ssh_argv(cmd), timeout=30)
            out = ((proc.stdout or "") + (proc.stderr or "")).strip()
            return proc.returncode == 0, out or f"{self.cfg.remote_python} not runnable"

        def _gpu():
            proc = self.runner.run(
                self._ssh_argv(
                    "nvidia-smi --query-gpu=name,memory.total,driver_version "
                    "--format=csv,noheader"
                ),
                timeout=30,
            )
            out = (proc.stdout or "").strip()
            if proc.returncode != 0 or not out:
                return False, (proc.stderr or "nvidia-smi failed")[-300:]
            try:
                driver = out.split(",")[-1].strip()
                major = int(driver.split(".")[0])
                if major < 570:
                    return False, f"{out} — driver < R570 (Blackwell needs >= 570)"
            except (ValueError, IndexError):
                pass
            return True, out

        def _torch_cuda():
            probe = (
                "import json, torch; "
                "print(json.dumps({'cuda': torch.cuda.is_available(), "
                "'torch': torch.__version__}))"
            )
            cmd = f"{shlex.quote(self._remote_python())} -c {shlex.quote(probe)}"
            proc = self.runner.run(self._ssh_argv(cmd), timeout=120)
            out = (proc.stdout or "").strip()
            try:
                payload = json.loads(out.splitlines()[-1])
            except (ValueError, IndexError):
                return False, (proc.stderr or out or "torch probe failed")[-300:]
            return bool(payload.get("cuda")), f"torch {payload.get('torch')} cuda={payload.get('cuda')}"

        def _versions():
            self._version_probe_cache.pop(f"{self.cfg.target}:{self.cfg.port}", None)
            self._check_version_skew(emit=False)  # keep doctor --json stdout pure
            remote = self._version_probe_cache.get(f"{self.cfg.target}:{self.cfg.port}") or {}
            if not remote:
                return False, "could not read remote torch/mjlab versions"
            from importlib.metadata import version as _v
            local = {}
            for pkg in ("torch", "mjlab"):
                try:
                    local[pkg] = _v(pkg)
                except Exception:  # noqa: BLE001
                    local[pkg] = "unknown"
            skew = {p for p in ("torch", "mjlab") if remote.get(p) != local.get(p)}
            detail = f"local={local} remote={remote}"
            return not skew, detail

        def _disk():
            proc = self.runner.run(
                self._ssh_argv(f"df -h {shlex.quote(self._workdir())} 2>/dev/null | tail -1 || df -h ~ | tail -1"),
                timeout=30,
            )
            out = (proc.stdout or "").strip()
            return bool(out), out or "df failed"

        _check("local ssh/rsync binaries", _rsync_local)
        _check("ssh reachable", _ssh_ok)
        # Remaining checks only meaningful if ssh works.
        ssh_ok = checks[-1]["ok"]
        if ssh_ok:
            _check("remote rsync", _rsync_remote)
            _check("remote python", _python)
            _check("nvidia driver/GPU", _gpu)
            _check("torch.cuda available", _torch_cuda)
            _check("torch/mjlab version match", _versions)
            _check("free disk at workdir", _disk)
        return {
            "ok": all(c["ok"] for c in checks),
            "host": self.cfg.host,
            "port": self.cfg.port,
            "checks": checks,
        }

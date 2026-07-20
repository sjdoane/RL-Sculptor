"""Private worker for executing generated metrics outside the parent process.

This file is launched by absolute path instead of ``python -m`` so importing it
does not execute ``sculptor.eval.__init__``.  It speaks a small length-framed
protocol over stdin/stdout, installs resource limits, and on Linux installs a
libseccomp filter before *any* untrusted module code executes.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import errno
import json
import os
import socket  # noqa: F401 - preload to test seccomp independently of imports
import struct
import sys
import types
from typing import Any, BinaryIO

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX portability fallback
    resource = None  # type: ignore[assignment]

import numpy as np
import numpy.fft  # noqa: F401 - preload before filesystem syscalls are denied
import numpy.linalg  # noqa: F401 - preload before filesystem syscalls are denied
import numpy.ma  # noqa: F401 - np.median lazily reaches masked-array helpers
import numpy.polynomial  # noqa: F401 - legitimate root numpy surface
import numpy.random  # noqa: F401 - allowed generated-metric submodule


_MAX_FRAME_BYTES = 512 * 1024 * 1024
_MAX_RESPONSE_BYTES = 1024 * 1024
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000
_PR_SET_NO_NEW_PRIVS = 38

_RESOURCE_LIMITS = {
    "address_space_bytes": 1536 * 1024 * 1024,
    "cpu_seconds": 120,
    "file_size_bytes": 0,
    "core_size_bytes": 0,
    "open_files": 16,
}


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("metric sandbox protocol closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream: BinaryIO) -> bytes:
    size = struct.unpack(">Q", _read_exact(stream, 8))[0]
    if size > _MAX_FRAME_BYTES:
        raise ValueError(f"metric sandbox request exceeds {_MAX_FRAME_BYTES} bytes")
    return _read_exact(stream, size)


def _write_frame(stream: BinaryIO, payload: bytes) -> None:
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("metric sandbox response exceeds protocol limit")
    stream.write(struct.pack(">Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 4096:
            raise ValueError("generated metric returned an oversized array")
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _response(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _set_resource_limits() -> dict[str, int | None]:
    if resource is None:
        return {name: None for name in _RESOURCE_LIMITS}
    limits = (
        ("address_space_bytes", resource.RLIMIT_AS),
        ("cpu_seconds", resource.RLIMIT_CPU),
        ("file_size_bytes", resource.RLIMIT_FSIZE),
        ("core_size_bytes", resource.RLIMIT_CORE),
        ("open_files", resource.RLIMIT_NOFILE),
    )
    applied: dict[str, int | None] = {}
    for name, kind in limits:
        requested = _RESOURCE_LIMITS[name]
        try:
            _, hard = resource.getrlimit(kind)
            value = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
            resource.setrlimit(kind, (value, value))
            applied[name] = int(resource.getrlimit(kind)[0])
        except (OSError, ValueError):
            applied[name] = None
    return applied


def _install_seccomp() -> tuple[bool, str | None]:
    """Deny filesystem, network, process-spawn, and kernel escape syscalls."""
    if not sys.platform.startswith("linux"):
        return False, "seccomp is Linux-only"
    library_name = ctypes.util.find_library("seccomp")
    if not library_name:
        return False, "libseccomp is unavailable"
    try:
        lib = ctypes.CDLL(library_name, use_errno=True)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [
            ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_ulong, ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            return False, f"PR_SET_NO_NEW_PRIVS failed: errno {ctypes.get_errno()}"
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
        lib.seccomp_rule_add.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
        ]
        lib.seccomp_rule_add.restype = ctypes.c_int
        lib.seccomp_load.argtypes = [ctypes.c_void_p]
        lib.seccomp_load.restype = ctypes.c_int
        context = lib.seccomp_init(_SCMP_ACT_ALLOW)
        if not context:
            return False, "seccomp_init failed"
        blocked = {
            # Filesystem read/write and namespace escape.
            "open", "openat", "openat2", "creat", "unlink", "unlinkat",
            "rename", "renameat", "renameat2", "mkdir", "mkdirat", "rmdir",
            "link", "linkat", "symlink", "symlinkat", "truncate", "ftruncate",
            "chmod", "fchmod", "fchmodat", "chown", "fchown", "fchownat",
            "lchown", "mknod", "mknodat", "mount", "umount2", "pivot_root",
            "chroot", "name_to_handle_at", "open_by_handle_at",
            "access", "faccessat", "faccessat2", "stat", "lstat",
            "newfstatat", "statx", "readlink", "readlinkat", "getdents",
            "getdents64", "chdir", "fchdir", "utime", "utimes",
            "futimesat", "utimensat", "setxattr", "lsetxattr", "fsetxattr",
            "removexattr", "lremovexattr", "fremovexattr", "swapon",
            "swapoff", "quotactl",
            # Network creation and use.
            "socket", "socketpair", "connect", "bind", "listen", "accept",
            "accept4", "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg",
            "recvmmsg", "shutdown",
            # New processes, code images, tracing, and cross-process mutation.
            "execve", "execveat", "fork", "vfork", "clone", "clone3",
            "unshare", "setns", "ptrace", "process_vm_readv",
            "process_vm_writev", "kill", "tkill", "tgkill",
            "pidfd_open", "pidfd_getfd", "pidfd_send_signal", "prlimit64",
            # Kernel-program and async-IO surfaces that can bypass open rules.
            "bpf", "perf_event_open", "userfaultfd", "io_uring_setup",
            "io_uring_register", "io_uring_enter", "keyctl", "add_key",
            "request_key", "reboot", "kexec_load", "kexec_file_load",
            "init_module", "finit_module", "delete_module", "memfd_create",
            # Cross-process System-V IPC is unrelated to numerical scoring.
            "shmget", "shmat", "shmdt", "shmctl", "msgget", "msgsnd",
            "msgrcv", "msgctl", "semget", "semop", "semtimedop", "semctl",
        }
        action = _SCMP_ACT_ERRNO | errno.EPERM
        for name in sorted(blocked):
            number = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                continue
            rc = lib.seccomp_rule_add(context, action, number, 0)
            if rc != 0:
                lib.seccomp_release(context)
                return False, f"seccomp rule failed for {name}: {rc}"
        rc = lib.seccomp_load(context)
        lib.seccomp_release(context)
        if rc != 0:
            return False, f"seccomp_load failed: {rc}"
        return True, None
    except Exception as exc:  # noqa: BLE001 - reported to parent, never hidden
        return False, f"{type(exc).__name__}: {exc}"


def _decode_request(payload: bytes) -> tuple[dict[str, np.ndarray], dict, dict]:
    if len(payload) < 8:
        raise ValueError("metric sandbox request header is truncated")
    header_size = struct.unpack(">Q", payload[:8])[0]
    if header_size > len(payload) - 8:
        raise ValueError("metric sandbox request header length is invalid")
    header = json.loads(payload[8:8 + header_size].decode("utf-8"))
    raw = memoryview(payload)[8 + header_size:]
    arrays: dict[str, np.ndarray] = {}
    for item in header.get("arrays", []):
        name = item["name"]
        dtype = np.dtype(item["dtype"])
        if dtype.hasobject:
            raise ValueError(f"object dtype is forbidden for array {name!r}")
        offset = int(item["offset"])
        nbytes = int(item["nbytes"])
        shape = tuple(int(value) for value in item["shape"])
        if offset < 0 or nbytes < 0 or offset + nbytes > len(raw):
            raise ValueError(f"array payload bounds are invalid for {name!r}")
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != nbytes:
            raise ValueError(f"array byte count is invalid for {name!r}")
        arrays[str(name)] = np.frombuffer(
            raw[offset:offset + nbytes], dtype=dtype,
        ).reshape(shape)
    behavior = header.get("behavior")
    meta = header.get("meta")
    if not isinstance(behavior, dict) or not isinstance(meta, dict):
        raise ValueError("metric sandbox behavior/meta must be objects")
    return arrays, behavior, meta


def main() -> int:
    # Untrusted metric code shares this process and can write to any file
    # descriptor it can name. The control channel therefore MUST NOT stay on
    # the conventional stdio fds: with the protocol still on fd 1, a metric
    # doing ``os.write(1, <forged frame>)`` injected its own response frame and
    # desynced every following call (adversarial finding, 2026-07-19).
    # Duplicate the inherited pipes onto private descriptors, then repoint fd
    # 0/1 at /dev/null so ``print``, ``os.write(1, …)``, and any inherited-stdio
    # write land in the void. /proc fd enumeration is seccomp-denied, so the
    # private descriptor is not reachable by convention. This is the process
    # boundary standing on its own — not the AST gate — for score integrity.
    control_in_fd = os.dup(0)
    control_out_fd = os.dup(1)
    # Buffered streams so a >PIPE_BUF frame is fully written on flush rather
    # than partially by a single raw write() syscall.
    input_stream = os.fdopen(control_in_fd, "rb")
    output_stream = os.fdopen(control_out_fd, "wb")
    devnull = open(os.devnull, "w", encoding="utf-8")  # noqa: PTH123
    os.dup2(devnull.fileno(), 0)
    os.dup2(devnull.fileno(), 1)
    try:
        load = json.loads(_read_frame(input_stream).decode("utf-8"))
        source = load.get("source") if isinstance(load, dict) else None
        if not isinstance(source, str):
            raise ValueError("metric sandbox load request has no source")
        code = compile(source, "<generated_metric>", "exec")

        # Warm common lazy numpy paths before filesystem access is denied.
        probe = np.asarray([0.0, 1.0], dtype=np.float64)
        _ = (probe.mean(), np.linalg.norm(probe), np.fft.rfft(probe))
        resource_limits = _set_resource_limits()
        rlimits = all(
            value is not None and value <= _RESOURCE_LIMITS[name]
            for name, value in resource_limits.items()
        )
        if sys.platform.startswith("linux") and not rlimits:
            _write_frame(output_stream, _response({
                "ok": False,
                "error": "required resource limits could not be established",
                "resource_limits": resource_limits,
            }))
            return 3
        seccomp, seccomp_error = _install_seccomp()
        if sys.platform.startswith("linux") and not seccomp:
            _write_frame(output_stream, _response({
                "ok": False,
                "error": f"required seccomp isolation unavailable: {seccomp_error}",
            }))
            return 3
        isolation = "process"
        if rlimits:
            isolation += "+rlimit"
        if seccomp:
            isolation += "+seccomp"

        module = types.ModuleType("_sandboxed_generated_metric")
        module.__dict__["__builtins__"] = __builtins__
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            exec(code, module.__dict__)
        fn = module.__dict__.get("compute_spec")
        if not callable(fn):
            raise ValueError("generated metric lacks callable compute_spec")
        _write_frame(output_stream, _response({
            "ok": True,
            "isolation": isolation,
            "seccomp_error": seccomp_error,
            "resource_limits": resource_limits,
        }))

        while True:
            try:
                payload = _read_frame(input_stream)
            except EOFError:
                return 0
            try:
                arrays, behavior, meta = _decode_request(payload)
                with (
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    result = fn(arrays, behavior, meta)
                encoded = _response({"ok": True, "result": result})
            except BaseException as exc:  # worker boundary catches SystemExit too
                encoded = _response({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
            _write_frame(output_stream, encoded)
    except BaseException as exc:
        try:
            _write_frame(output_stream, _response({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }))
        except BaseException:
            pass
        return 2
    finally:
        devnull.close()


if __name__ == "__main__":
    raise SystemExit(main())

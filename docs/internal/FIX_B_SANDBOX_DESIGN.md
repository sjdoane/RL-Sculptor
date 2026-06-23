# Fix B — restricted subprocess sandbox for generated-metric exec (DESIGN)

Status: **designed, not yet implemented.** Produced by a 4-architecture design workflow
(2026-06-23, rounds 27–29 autonomous-hardening arc) whose agents EMPIRICALLY verified every
claim below on this exact box (WSL2 Ubuntu 24.04, kernel 6.6.87.2-microsoft-standard-WSL2,
py3.13, numpy 2.4.4, rootless user, `sudo` needs a password).

## Why (the forcing function)

The generated objective metric is UNTRUSTED LLM-authored python. The static AST gate
`metric_validate._ast_safety` is the only containment, and rounds 27–29 EMPIRICALLY proved it
cannot be made complete: **two CRITICAL escape CLASSES were reproduced in consecutive rounds**,
each slipping the prior round's denylist patch —

- round-28: `(i for i in ()).gi_frame.f_builtins["<chr-assembled __import__>"]("os")` (frame reflection)
- round-29: `np.random.seed.func_globals` / `func_code` + `code.replace` + `FunctionType` (numpy-cython Py2 aliases)

A leaf/prefix denylist over the reflection graph is unbounded. The durable fix is to run the
module exec + `compute_spec` in a restricted process so containment does NOT depend on
enumerating reflection paths.

## Threat model

The metric source is **system-generated** — the platform's own Claude call authors it from the
user's goal — NOT adversarial third-party input. So the CURRENT realistic risk is LOW (the
comprehensive denylist is an adequate live gate today). Fix B is the precondition for accepting
**adversarial / untrusted `behavior_goal` sources** later, and a robustness win now
(out-of-process exec means a crashing/hanging/memory-bombing metric can't take down the parent).

## Hard constraints (code-verified)

1. **Empty/curated `__builtins__` is NOT viable.** numpy ndarray methods (`.mean`/`.max`/`.std`)
   call `__import__` internally, so an emptied `__builtins__` breaks legitimate metrics.
   `mission_runtime._evaluate_success_criterion` (line 314-324) deliberately KEEPS real
   `__builtins__` for this reason and relies on a strict no-attribute-access AST validator — an
   approach a numpy-COMPUTATION metric cannot use. So the sandbox must keep real builtins INSIDE
   and lock the process down OUTSIDE.
2. **Score identity is mandatory.** 937 sculptor tests assert specific `spec_score` floats across
   every exec path. The sandbox must be byte-identical. **A forked worker guarantees this** (same
   interpreter, same numpy build, same `_ast_safety+exec+compute_spec` code path as today); JSON
   float round-trip is byte-exact via `repr`.
3. **Never-raise contract.** Every path degrades any failure to `{"spec_score": 0.0, "error": …}`.
   Timeout / crash / nonzero-exit / malformed-output → honest 0.0.
4. **Top-level exploits run at module-exec time** → `REQUIRED_JOINT_ROLES` must be read STATICALLY
   (AST, no exec). It is always a literal `["str", …]`, so `ast.literal_eval` on the RHS works.

## Recommended architecture (synthesis of all 4)

**One lean PERSISTENT forked worker per grant** (Arch 4 process model) that installs **all four**
containment layers (Arch 1 + Arch 2 + Arch 3-via-unshare), behind a flag, gated by a parity harness.

- **Process model — fork-pool (Arch 4).** A `MetricSandbox` object forks ONE worker per grant
  (or per training run). fork() happens AFTER numpy + `sculptor.eval.generated_metric` +
  `metric_validate` are imported in the parent (they always are at every call site) → the child
  needs ZERO re-import; lockdown is sub-ms. Request loop over a `socketpair`: parent sends
  `{op, path, arrays_pickle, behavior, meta}`, child runs `_ast_safety + exec + compute_spec`
  EXACTLY as today and returns a JSON-safe dict. Per-call overhead ≈ 0.1–0.5 ms (pickle ~112 KiB
  synthetic / ~15 KiB real npz + socket round-trip). **Per-grant added latency ≈ 2–12 ms**
  (invisible vs the multi-second LLM ladder-author calls). A crashed/timed-out/escaped worker is
  SIGKILLed + respawned; the in-flight request degrades to 0.0. (Do NOT use a fresh `python -m
  worker` per call — that's ~0.25–0.4 s × 25 ≈ 10 s/grant. Do NOT add shared memory — at ~112 KiB
  pickle-over-pipe is fine.)
- **Crash / hang / memory / parent isolation (Arch 1).** Separate process + parent-side
  `select()` wall-clock deadline (SIGKILL on expiry) + `RLIMIT_CPU` backstop + `RLIMIT_AS` memory
  cap (child OOMs alone). **Empirically: this is all Arch 1 contains.**
- **File write/delete isolation — FILESYSTEM, not syscall.** `chdir` to a throwaway tmpfs cwd +
  `RLIMIT_FSIZE=0`. *(Empirically: `RLIMIT_FSIZE=0` blocks write CONTENT — `open('w')` succeeds,
  `write` raises OSError 27 — but does NOT block `unlink`/`rename`/truncate. seccomp ALSO cannot
  reliably block writes: `openat` arg-filtering of `O_ACCMODE`/`O_CREAT` via masked-EQ is fragile
  and `openat2` passes flags via a struct pointer seccomp can't dereference. So write/delete
  containment must come from there being NOTHING valuable writable in the namespace — a tmpfs cwd,
  and ideally a mount-namespace that ro-binds only the venv + the one metric.py + the rollout dir.)*
- **Network / exec / reflection-to-shell — seccomp-bpf (Arch 2).** A ~60-line ctypes shim against
  `libseccomp.so.2` (PRESENT on this box; no pip dep; `PR_SET_NO_NEW_PRIVS` returns 0 rootless;
  `actions_avail` has the full set). **Default-ALLOW + targeted-deny (ERRNO, not KILL)** on
  `socket`/`connect`/`execve`/`execveat`/`fork+exec`/`unlink`/`unlinkat`/`rename`/`renameat`/
  `ptrace`/`mount`. ERRNO (not SIGSYS-kill) so that if numpy ever needs a mis-denied syscall it
  gets a clean `-EPERM` → OSError → honest 0.0, not a crash. **Empirically verified: numpy
  compute (mean/std/max/clip/gradient/select/abs on 3-D arrays) is byte-identical post-filter;
  `socket()` → EPERM; `os.system('touch …')` → no file, execve denied.** This DECISIVELY closes
  the round-28/29 reflection-to-shell class even if `_ast_safety` is fully bypassed.
- **Stronger option — unprivileged namespaces (Arch 3).** `os.unshare(CLONE_NEWUSER|NEWNET|NEWNS)`
  (all confirmed working rootless here: `max_user_namespaces=62137`, `unshare --net` blocks the
  network, tmpfs mount + `pivot_root` succeed) additionally contains file READS (mount-ns
  ro-binds). bubblewrap is the polished form but is NOT installed and `sudo` needs a password →
  adopting bwrap means vendoring the `.deb` (supply-chain surface). **Prefer the pure-Python
  `unshare` worker (zero new dep, ~90% of bwrap's containment, same ~0.16 s spawn); make bwrap an
  OPTIONAL hardened launcher.**
- **Static role reader (ship FIRST, independently).** `read_required_roles_static(source)` parses
  the top-level `REQUIRED_JOINT_ROLES = [...]` assignment with `ast.literal_eval` on the RHS — no
  exec. Removes the last untrusted top-level exec from the parent metadata path even before the
  sandbox lands. **This is the safe, score-irrelevant down-payment — implement it now.**

## Rollout plan (gates the high blast radius)

1. Land `read_required_roles_static` (low risk, score-irrelevant, suite-validated). ✅ done in the
   round-29-followup increment.
2. Land the fork-pool `MetricSandbox` as the single parent chokepoint behind `RS_METRIC_SANDBOX`
   (**default OFF**); route `compute_generated_metric` + the calibration `gen_fn` calls + the
   validation battery through it. Batch the validation battery into ONE worker call.
3. Add a **byte-parity harness** over the 937-test corpus asserting identical `spec_score` through
   both the in-process and sandbox paths. Flip `RS_METRIC_SANDBOX` ON only when parity is green.
4. Add the ctypes seccomp filter + the `unshare` lockdown as the worker's first acts.
5. **Fail-CLOSED** on any sandbox-launch failure (honest 0.0) — NEVER silently fall back to
   in-process exec (that would re-open the hole the sandbox closes).
6. For the test suite: a session/module-scoped worker fixture (or keep the in-process path as the
   default for pure-math unit asserts and run a dedicated sandbox-parity subset) to keep CI fast.

## What each architecture does / does NOT contain (empirical)

| layer | crash/hang/mem | parent isolation | FS write/delete | network | exec/shell | FS read |
|---|---|---|---|---|---|---|
| Arch 1 (rlimit subprocess) | ✅ | ✅ | write-size only (no unlink) | ❌ | ❌ | ❌ |
| + seccomp (Arch 2) | ✅ | ✅ | ❌ (use fs-isolation) | ✅ | ✅ | ❌ |
| + tmpfs cwd / unshare-mount (Arch 3) | ✅ | ✅ | ✅ | ✅ | ✅ (inherits sandbox) | ✅ |
| fork-pool (Arch 4) = process model for all of the above; guarantees score identity via fork |

The recommended build is **Arch 4 fork-pool carrying Arch 1 rlimits + Arch 3 tmpfs/unshare + Arch 2
seccomp** — the union closes all reproduced + foreseeable escape classes with ~2–12 ms/grant
overhead and byte-identical scores.

# Generated metric process sandbox

RewardSculptor treats every LLM-authored objective metric as untrusted code.
Static AST validation remains useful for rejecting code outside the numerical
metric contract, but it is not the security boundary. All generated-module
top-level code and every `compute_spec` call execute in a dedicated worker
process created by `load_generated_module`.

## Enforcement model

The loader reads the metric once and hashes that immutable source snapshot.
`REQUIRED_JOINT_ROLES` is parsed statically from the same snapshot, preventing
the validated metadata and executed code from diverging if the file changes.
The worker is started eagerly so syntax errors, top-level failures, or a missing
`compute_spec` remain load-time failures.

Each worker has:

- a fresh private working directory and a small allowlisted environment that
  excludes parent credentials and API keys;
- a 1.5 GiB address-space cap, 120-second cumulative CPU cap, zero-byte regular
  file/core-dump caps, and at most 16 open file descriptors;
- a three-second parent-enforced wall timeout for each metric call;
- bounded length-framed IPC using JSON metadata and raw contiguous numeric
  arrays, never pickle; and
- on Linux, a fail-closed libseccomp filter installed before untrusted module
  execution. It denies filesystem access/mutation, networking, new processes
  and executable images, tracing or cross-process mutation, System V IPC,
  kernel-programming interfaces, and namespace escape. `NO_NEW_PRIVS` is set
  explicitly before the filter loads.

The worker preloads the vetted NumPy computation surface before filesystem
syscalls are sealed. This is necessary because otherwise legitimate operations
such as `np.median` can trigger lazy NumPy imports after containment begins.
Generated code retains real builtins inside the worker for the same reason;
those builtins never execute in the campaign process.

## Platform behavior

Linux is the research/campaign deployment target. It fails closed if either
the required resource limits or libseccomp cannot be established. Install the
distribution's libseccomp runtime (for example, `libseccomp2` on Debian/Ubuntu)
in every campaign image.

Other platforms retain process, IPC, environment, and wall-time isolation and
use resource limits when their Python runtime exposes them, but do not claim
the Linux syscall-containment guarantee. Research evidence intended for a
campaign should therefore be collected on Linux.

The callable proxy exposes read-only diagnostic provenance at
`compute_spec.sandbox_info`, including the isolation level, source SHA-256,
resource limits, and call timeout. Runtime score semantics and the generated
metric result schema are unchanged.

## Scope and non-goals

This boundary protects the parent from generated **objective metric** code. It
does not decide whether a metric is scientifically valid: AST contract checks,
axiom tests, adversarial archetypes, calibration, and the metric firewall still
provide that evidence. It also does not yet isolate the separately validated
mission success-criterion expression path; that remains a distinct hardening
slice rather than an implied guarantee of this implementation.

The sandbox is a containment and availability boundary, not a proof against
all microarchitectural side channels. A metric can still fail, exhaust its own
budget, or terminate its own worker; those events become observable score-path
errors while the campaign process stays alive.

## Verification

Run the focused boundary and compatibility suites with:

```bash
.venv/bin/python -m pytest tests/test_metric_sandbox.py -q
.venv/bin/python -m pytest \
  tests/test_generated_metric.py \
  tests/test_metric_axioms.py \
  tests/test_task_derived_calibration.py \
  tests/test_reference_anchored_validation.py \
  tests/test_novel_metric_robustness.py \
  tests/test_joint_resolver.py \
  tests/test_channel_catalog_metric.py \
  tests/test_metric_sandbox.py -q
```

The boundary suite deliberately bypasses the static validator to prove the OS
layer independently: it tests top-level and call-time file writes, socket
creation, process spawning, inherited-secret probing, frozen-source identity,
infinite-loop termination, and a native worker crash.

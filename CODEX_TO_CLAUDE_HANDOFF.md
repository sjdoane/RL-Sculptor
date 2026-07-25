> **⚠️ SUPERSEDED — the current handoff is [`HANDOFF.md`](HANDOFF.md) at the repo root.**
> On "read handoff", read `HANDOFF.md` and begin its Task 1. This file below is
> older, task-specific history kept for reference only.

---

# Latest handoff — mission success-criterion process isolation (2026-07-19)

Codex completed and verified the remaining evaluator-isolation half of P0.11
while Claude continued the A4 evidence track. This slice is self-contained in
the commit that includes this handoff. It starts from `e2d4a4d`, touches no
A4-owned adapter/spec-audit/calibration/evidence files, and launches no GPU
work. The pre-existing transient `.fleaven*`, `.ingest*`, `.metric*`, and
`.pytest*` artifacts remain untracked and were not staged.

Decision and implementation:

- `mission_runtime._evaluate_success_criterion` previously AST-screened an
  LLM-authored expression, then compiled/evaluated it in the campaign process
  with real builtins. The AST gate was strong but explicitly non-proving, so
  this was the remaining process-isolation gap documented by the roadmap.
- Criteria now reuse the already adversarially hardened generated-metric worker
  rather than creating a second syscall/resource policy that could drift. The
  parent still parses and validates the exact immutable expression; compilation,
  eval, and boolean coercion happen only after the worker has established its
  private cwd/environment, rlimits, `NO_NEW_PRIVS`, and fail-closed Linux
  seccomp filter.
- A trusted adapter transports `metric`, `behavior`, `components`,
  `trajectory`, and `info` through the bounded JSON/raw-array protocol. It
  preserves `info is trajectory` when aliased and keeps the mappings distinct
  when a caller supplies them separately. Helper functions are reconstructed
  from a fixed builtin allowlist; caller-supplied callables never cross IPC.
- Every criterion decision gets a fresh worker. This is intentional: a bypassed
  expression can mutate or crash its own interpreter, but cannot poison the
  next stage's authority decision. Hangs retain the parent-enforced 3-second
  wall limit, and native crashes become `CriterionEvalError` without taking
  down the campaign.
- Runtime diagnostics remain compatible: remote `KeyError` still becomes the
  recoverable `CriterionMissingKeyError`; NumPy ambiguous-array results keep
  the `.all()`/`.any()` guidance; unknown-name error ordering is unchanged.
  The AST validator now also enforces its previously documented but missing
  rule that bare calls must target the fixed numerical helper allowlist.

Files changed/added:

- `RewardSculptor/sculptor/mission_runtime.py`
- `RewardSculptor/tests/test_criterion_sandbox.py` (new)
- `RewardSculptor/docs/generated_metric_sandbox.md` (now documents both
  evaluator paths and their explicit non-goals)
- `docs/internal/REWARDSCULPTOR_RESEARCH_GRADE_ROADMAP.md` (metric + criterion
  isolation item marked complete)

Verification evidence:

- Criterion boundary suite: **9 passed**. It independently bypasses the AST
  validator and proves filesystem/socket attempts are denied, parent secrets
  are absent, infinite expressions time out, a native `ctypes` crash is local,
  and interpreter/builtins poisoning does not cross decisions. It also pins
  normal numerical semantics and distinct `info`/`trajectory` mappings.
- Focused mission/decomposition/criterion battery: **260 passed in 37.14s**.
- Repository-wide requested command:
  `MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q
  --ignore=tests/test_refs_preview.py` -> **2,134 passed, 1 optional-JAX skip,
  152 warnings in 229.69s (3:49)**.
- `uvx ruff check`, targeted `compileall`, and `git diff --check` passed.

Honest boundary: objective metrics and mission success criteria are now both
OS-isolated. This does **not** claim isolation for generated reward
implementations, tools, or algorithm code; those remain separate A8 capability
boundaries. Static semantic gates and evaluator calibration also remain
necessary—the process sandbox prevents parent compromise/DoS, not reward
gaming or a scientifically invalid criterion.

---

# Latest handoff — generated objective-metric process isolation (2026-07-19)

Codex completed a separate CPU-only P0.11 slice while Claude continued A4 GPU
evidence collection. It was committed as `70a87d8` on
`ship-20-ux-revamp`. No A4-owned adapter,
spec-audit/metric-axiom/calibration, evidence-doc, benchmark-doc, or GPU file was
modified; no GPU command was launched. The already-committed coordinator was
also left untouched.

Decision and implementation:

- Generated objective metrics were the highest-impact file-disjoint open
  security gap: the AST denylist was explicitly documented as non-proving, but
  `load_generated_module` still executed accepted LLM code in the campaign
  process. It now returns a module-like callable proxy and executes both module
  top-level code and every `compute_spec` call only in a persistent worker.
  Existing validation and calibration paths inherit the boundary without edits
  to Claude-owned `metric_calibration.py`.
- The loader reads one immutable source snapshot, hashes it, screens it, sends
  that exact snapshot to the worker, and parses `REQUIRED_JOINT_ROLES` from the
  same bytes. Editing the file after load cannot split validated metadata from
  deployed code.
- IPC is bounded, length-framed JSON plus raw contiguous numeric arrays. Pickle
  is never used. Results stay schema-compatible; diagnostic provenance is
  available at `compute_spec.sandbox_info` (isolation level, source SHA-256,
  limits, timeout).
- Workers get a private temp cwd and an allowlisted, credential-free
  environment. Each call has a 3-second parent wall timeout. Worker limits are
  fail-closed on Linux: 1.5 GiB address space, 120 seconds cumulative CPU,
  zero-byte regular-file/core caps, and <=16 FDs.
- Linux also fails closed unless libseccomp loads. `NO_NEW_PRIVS` is explicit;
  the filter is installed before any untrusted exec and denies filesystem
  access/mutation, network, fork/clone/exec, ptrace/cross-process/PID-fd access,
  namespace escape, kernel-programming/async-IO surfaces, and System V IPC.
  Other platforms retain the process/IPC/environment/wall-time boundary but do
  not claim Linux syscall containment.
- Trusted NumPy `linalg`/`random`/`fft`/`polynomial`/masked-array paths preload
  before open syscalls are sealed. This was required for score equivalence:
  `np.median` otherwise lazily tried to import `numpy.ma`. Common remote
  numerical exceptions such as `IndexError` remain their native exception type
  because task-derived calibration uses those diagnostics; OS denials stay
  explicit `MetricSandboxExecutionError`s.

Files changed/added:

- `RewardSculptor/sculptor/eval/generated_metric.py`
- `RewardSculptor/sculptor/eval/_metric_sandbox_worker.py` (new)
- `RewardSculptor/tests/test_metric_sandbox.py` (new)
- `RewardSculptor/docs/generated_metric_sandbox.md` (new)
- `docs/internal/REWARDSCULPTOR_RESEARCH_GRADE_ROADMAP.md` (marks only the
  generated-metric half of the isolation item complete)

Verification evidence:

- Adversarial boundary suite: **9 passed**. It bypasses `_ast_safety` on purpose
  and independently proves top-level/call-time file writes, sockets, and child
  process creation are blocked; parent secrets are absent; source snapshots are
  frozen; hangs are killed; a native `ctypes` segfault kills only the worker and
  a fresh worker scores normally afterward; configured rlimits and Linux
  seccomp provenance are asserted.
- Generated-metric/calibration compatibility battery: **335 passed** in 84.54s
  (generated metrics, metric axioms, task-derived calibration, reference
  anchoring, robustness, joint roles, channel catalogs, sandbox).
- Repository-wide requested command passed:
  `MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q
  --ignore=tests/test_refs_preview.py` -> **2,124 passed, 1 optional-JAX skip,
  152 warnings in 278.71s (4:38)**.
- `uvx ruff check` on all new/changed Python files, targeted `compileall`, and
  `git diff --check` all pass.

Honest residual: this slice isolates generated **objective metrics**, not the
separate mission success-criterion expression evaluator. Criterion isolation
remains an explicit P0.11/A8 follow-up; do not represent all generated code as
isolated yet. Suggested next action is adversarial review of the worker/filter,
then commit this slice separately from A4. Do not stage/delete the pre-existing
`.fleaven*`, `.ingest*`, `.metric*`, or `.pytest*` files.

## Verified + hardened 2026-07-19 (Claude)

Adversarial review (the suggested next action) CONFIRMED every containment
claim: eight live escape payloads through the real `load_generated_module`
path — env-secret exfil, `os.kill(getppid)`, raw-`ctypes` syscalls
bypassing glibc, `fork`+`memfd_create`, object-dtype return, oversized
return, at-import socket — were all blocked at the OS layer (seccomp
EPERM / JSON-encode refusal / no parent DoS), with the env allowlist
carrying no PATH/HOME/LD_PRELOAD and `np.median` scoring identically in
and out of the sandbox. One genuine defect was found and fixed before
commit:

- **IPC frame injection via shared fd 1 (integrity).** The worker only
  redirected the Python-level `sys.stdout`; raw fd 1 stayed wired to the
  control pipe, so a metric doing `os.write(1, <forged frame>)` injected
  its own response — the verifier made a call return `spec_score: 0.999`
  and desynced every following call. This is score forgery, not a sandbox
  escape (the return is JSON-parsed, no pickle/RCE), but scores drive
  selection/best-of-N/calibration, so it matters. The static AST gate
  blocks the write primitives today, but this slice's whole premise is
  that AST gates get bypassed, so the process boundary must stand alone.
  Fix (`_metric_sandbox_worker.py`, worker-owned): the control channel is
  dup'd onto private descriptors and fd 0/1 are repointed at `/dev/null`
  before any untrusted code runs, so `print`/`os.write(1, …)`/inherited
  stdio land in the void; /proc fd enumeration is already seccomp-denied.
  Regression test `test_forged_stdout_frame_cannot_forge_score_or_desync`
  (mutation-verified: it FAILS on the pre-fix worker — the forged
  `{'FORGED': True}` reaches the parent — and PASSES on the fixed one).

Sandbox suite now 10 passed; broad suite green (2,124 passed / 1
optional-JAX skip pre-fix; the fix touches only the worker + one test).
Accepted residual (documented by the verifier, not fixed): `error_type`
lets a metric spoof its OWN failure's reported exception class — cosmetic,
grants no capability. The mission success-criterion evaluator remains
un-isolated as Codex documented.

---

# Latest handoff — charter-aware global-matrix coordinator (2026-07-19)

Codex completed the CPU-only, file-disjoint coordinator task. The work is
intentionally **uncommitted** on `ship-20-ux-revamp`. Claude's A4/GPU-owned
files (`sculptor/adapters/*`, the four named `eval/spec_*`/`metric_*` files,
and `docs/audits/*` / `docs/benchmarks/*`) were not modified, and no training
or evidence-collection GPU job was launched.

What was implemented:

- New `sculptor/eval/sharding.py` creates one full benchmark × condition × seed
  design and charters it once at the campaign root. It seals a deterministic
  global partition, then emits transportable `shards/shard-NNN` directories.
  Every manifest embeds the complete partition and only assigns jobs from it;
  no worker creates or verifies a subset design.
- Every shard carries a byte-identical charter replica plus frozen external
  input replicas. A worker reconstructs the full config and frozen benchmark
  definitions, verifies current source/dependency identity against the full
  charter, verifies physical inputs, then seals `shard_run_identity.json`
  before running its first job. The charter replica is verification material,
  not a second charter.
- The charter now records explicit per-file `pyproject.toml` / `uv.lock`
  hashes and a dependency-identity hash in addition to the aggregate source
  tree hash. Manifests, worker identities, and merge provenance must match all
  of these fields exactly.
- KG isolation is preserved across pods: `campaign_inputs/kg_base.db` at the
  global root is the chartered source; each shard receives a verified
  byte-identical replica; existing harness logic then creates a private
  writable `inputs/kg.db` per assigned job. Altered shard or root KG bytes are
  rejected.
- Worker crash-resume uses the existing `_run_job` artifacts but adds checks
  that cached result identity equals the assigned atomic tuple. The sealed
  manifest and worker runtime identity must also match exactly on resume.
- Merge verifies the root runtime, coordinator, shard manifest, charter
  replica, runtime/dependency identity, external inputs, result lineage,
  result path/tuple, and frozen assignment. It rejects a tuple claimed by two
  shards before assignment handling. It copies verified job trees into the
  global campaign output and seals cumulative per-result provenance in
  `campaign_merge.json`; repeated merge is idempotent.
- Missing shards/jobs are allowed only as explicit partial evidence. JSON and
  HTML reports use `status: incomplete_coverage`, include exact expected /
  completed / missing counts, enumerate missing tuples, and warn that partial
  aggregates are not a complete campaign result.
- Added `sculpt eval shard prepare`, `sculpt eval shard run`, and
  `sculpt eval shard merge`. `docs/campaign_plan.md` now gives the supported
  three-pod recipe and retains an explicit ban on ad hoc merging of separately
  chartered subset campaigns.
- `CampaignConfig.validate` now rejects empty or duplicate benchmark and
  condition axes, preventing a malformed Cartesian product at its source.

Files changed/added:

- `RewardSculptor/sculptor/eval/sharding.py` (new)
- `RewardSculptor/tests/test_eval_sharding.py` (new)
- `RewardSculptor/sculptor/eval/charter.py`
- `RewardSculptor/sculptor/eval/harness.py`
- `RewardSculptor/sculptor/eval/__init__.py`
- `RewardSculptor/sculptor/cli.py`
- `RewardSculptor/docs/campaign_plan.md`

Verification evidence:

- Focused coordinator suite: **7 passed**. It covers foreign-charter rejection,
  duplicate tuple rejection across shards, honest missing-shard coverage,
  altered KG-base rejection, deliberately re-sealed runtime-hash drift,
  exact shard resume, and a real two-process CPU stub matrix merged end to end.
- Existing eval harness suite: **20 passed**.
- New files: `uvx ruff check` passed; `compileall` passed.
- Requested broad suite passed exactly:
  `MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q
  --ignore=tests/test_refs_preview.py` → **2112 passed, 1 optional-JAX skip,
  152 warnings in 302.91s (5:02)**.
- `git diff --check` passed before the broad run. Run it once more before any
  commit after inspecting the final diff.

Do not stage/delete the pre-existing `.fleaven*`, `.ingest*`, `.metric*`, or
`.pytest*` files. Do not promote YAM or change any `A0_rejected` /
`compile_only` status. Suggested next action: review the uncommitted diff,
rerun `tests/test_eval_sharding.py` if modifying coordinator logic, then commit
this slice separately from the concurrent A4 evidence work.

## Verified + hardened 2026-07-19 (Claude)

Adversarial verification CONFIRMED all coordinator claims (independent
broad suite 2,112 passed / 1 optional-JAX skip; focused 7 + 20 passed;
file-disjointness from the concurrent A4 work holds exactly). Mutation
battery: foreign-charter-at-merge, cross-shard duplicate, and
runtime-reference no-op mutations were all caught by existing tests; two
findings were hardened before commit:

- **Merge now trusts only attested results.** The verifier fabricated
  `final_spec_score: 999.0` in a shard `result.json` after the worker
  sealed its `shard_report.json`; merge accepted and reported it, because
  the sealed report was never consulted. `_scan_shard_results` now
  requires a sealed, charter/runtime-verified report bound to the exact
  manifest and run identity, and every merged result must equal its
  attested job record (`spec_series`/`cached` runtime annotations
  excluded). Results without a report refuse to merge ("resume the shard
  to completion before merging"). Honest scope note: with no secrets, a
  tamperer who rewrites BOTH files and reseals still passes — this is
  tamper-evidence depth, not access control; post-first-merge drift was
  already pinned by provenance `result_sha256`.
- **`_verify_charter_reference` is now pinned.** No-op'ing it survived
  all 7 sharding tests (it is redundant with byte-level charter checks on
  every path) — a direct unit test now kills that mutation. The duplicate
  -job test forges the attestation (reseal) so the cross-shard duplicate
  path stays independently tested.

Documented residuals (accepted): `missing_shards` lists only absent shard
dirs, not present-but-empty shards (job-level coverage is exact);
`_differences` emits a noisy-but-correct diff preview on runtime-identity
mismatches; partial-merge reports carry aggregates over the partial set —
quoting `aggregates` without `status`/`coverage` is a reader error the
JSON/HTML markers already guard.

After hardening: sharding+harness focused 30 passed; broad suite rerun
green (2,115 passed / 1 optional-JAX skip expected — 3 new tests).

---

# Previous handoff — generic temporal manipulation evaluator (2026-07-19)

Read this section first, then inspect the current diff with **WSL Git** from
`/home/samjd/projects` (Windows Git misreports WSL executable bits). The branch
is `ship-20-ux-revamp`. Do not delete or stage the pre-existing untracked
`.fleaven*`, `.ingest*`, `.metric*`, or `.pytest*` files.

Codex continued from commits `c4610ac` and `e628827` and implemented the next
highest-impact slice. The work is currently uncommitted:

- Manipulation telemetry is schema 2. Registered vectorized rollouts now write
  `rollout_valid` and `rollout_terminal`; the done sample is explicitly invalid
  because mjlab has already auto-reset its scene state. The sidecar declares
  the world-space target contract and task success threshold.
- Grasp discovery is capability-driven for future robots: left/right finger
  roles work as before; a generic multi-body `gripper` is split into independent
  contact-evidence groups. `grasp__<object>` means at least two groups contact
  simultaneously. A one-contact suction-like tool fails closed until it has a
  real retention-evidence adapter. There are no YAM/G1/task-name branches.
- New built-in `object_lift_hold` dynamically loads object/contact keys from the
  sidecar and requires: valid first command segment, initial separation/no
  initial grasp, 8 cm clearance, goal error <= declared threshold capped at
  5 cm, 0.5 s stable hold, >=80% two-group contact with <=0.1 s dropout,
  plausible finite kinematics, and <=5 cm non-target disturbance. Its
  `spec_score` is binary completion rate; continuous values are diagnostics.
  Missing/legacy/tampered telemetry earns an observable zero.
- Spec-audit evidence hashes now include `manipulation_telemetry.json`.
  `tests/test_object_lift_hold_spec.py` covers the competent target-aware
  multi-object case, all ten A4 attack classes, wrong-target flicker,
  distractor disturbance, forged grasp evidence, schema downgrade, spawn
  settling, and calibration-authority refusal. These are implementation tests,
  **not A4 evidence**.
- The metric-axiom layer has a real manipulation family rather than a vacuous
  exemption: world translation/yaw/object renaming must be invariant, while
  wrong-target selection, distractor motion, proxy-only contact, truncation,
  and unstable holds must not improve the score.
- `docs/audits/object_lift_hold_a4_evidence_plan.md` is the frozen real-evidence
  plan. The cross-embodiment frontier is truthfully versioned to 1.2.0 but YAM
  remains `compile_only`, `campaign_ready=false`, `A0_rejected`, and has no
  metric/certificate reference until separate real-rollout A4 audits and frozen
  splits pass for single- and multi-cube tasks.
- CLI/decomposer help and the UI `SpecMetricName` registry include the new
  metric. Legacy synthetic generated-metric calibration explicitly refuses this
  dynamic manipulation metric; task-derived frozen rollouts are required.

Verification already completed:

- Targeted integration suite: 194 passed. Manipulation axioms/audit subset:
  46 passed. Core new files pass Ruff; `compileall` and `git diff --check` pass.
- Broad Python suite: 2,104 passed, 1 optional-JAX skip in 3m27s, with only
  `tests/test_refs_preview.py` excluded because its real OpenGL tests can abort
  this headless WSL process.
- Native WSL frontend TypeScript check: passed (`pnpm typecheck`).
- Real CUDA/EGL YAM rollouts reused `/tmp/manip_smoke/checkpoint.pt` without
  retraining. A 60-step, 64-env schema-2 artifact had 13 manipulation channels,
  correct target index 0, 64/64 structurally and physically valid envs after
  real-data calibration, and honest score 0 for an untrained/no-grasp policy.
- A 1,000-step reset-boundary rollout proved every env records 999 valid samples
  followed by exactly one invalid terminal sample; the evaluator stayed valid
  and scored 0. Artifacts: `/tmp/manip_smoke_schema2_egl` and
  `/tmp/manip_smoke_schema2_reset`.
- The first renderer attempt failed on unavailable
  X11 `:0`; `MUJOCO_GL=egl` is the working headless path.

Resume by running the focused and broad suites after inspecting the diff, then
fix any failures. Do **not** promote YAM. The next research step is to train or
obtain a genuinely competent lift policy, freeze disjoint seeds/world/layout
splits, collect real artifacts for every A4 class following the evidence plan,
run separate single-/multi-cube audits, and promote only a passing certificate.

## Verified + hardened 2026-07-19 (Claude)

The slice above was verified and committed. Broad suite: 2,104 passed /
1 optional-JAX skip before hardening; frontend `pnpm typecheck` clean. An
adversarial subagent CONFIRMED every claim (including the runner's
valid/terminal mask contract, end-to-end) and ran a mutation battery.
Three mutations were caught by the existing tests; one compound mutation
was not: **removing BOTH the grasp-fraction and grasp-gap gates survived
all 101 focused tests and the axiom checker** — the two gates are
mutually redundant on dense evidence, and no test asserted that a
"telekinesis" transport with uniformly absent (self-consistent) contact
evidence scores zero. Hardening applied before commit:

- `tests/test_object_lift_hold_spec.py`: new
  `test_telekinesis_lift_without_grasp_evidence_scores_zero`.
- `eval/metric_axioms.py`: new `no_grasp_evidence` negative in the
  manipulation family (kills the compound mutation).
- `eval/spec_metrics.py`: two honesty fixes — the plausibility-gate
  comment falsely claimed teleports need forged huge velocities (a
  sub-0.12 m/step crawl with forged zero velocities passes the physics
  gates; provenance hashing is the actual defense, now stated), and the
  docstring now says non-finite telemetry / forged grasp disagreement
  fail the WHOLE artifact closed (the implementation raises globally,
  stricter than the per-env wording claimed).

Documented residuals (accepted, not fixed): sub-floor slow-teleport and
median-gate laundering are only reachable via forged artifacts, which
the spec-audit evidence hashes and simulator-path requirement exclude;
descriptor `gripper` body lists are not deduplicated (a malicious
descriptor could alias one body into two "independent" groups — the
capability registry is trusted input today).

A4 evidence collection has started per
`docs/audits/object_lift_hold_a4_evidence_plan.md`: a full 5000-iteration
`Mjlab-Lift-Cube-Yam` training run (1024 envs, seed 42, upstream RL
config) is running detached with artifacts under
`~/rs_evidence/object_lift_hold_v1/`. Evaluation seeds/splits will be
frozen in a `freeze.json` citing this commit BEFORE any outcome rollouts
are collected; training seed 42 is reserved for the policy and excluded
from evaluation splits. Mid-training checkpoints (every 100 iters) are
candidates for the `falling`/partial-competence attack classes.

---

# RewardSculptor implementation handoff

Read this file, then inspect the current diff before changing anything. The
active repo is `/home/samjd/projects` on branch `ship-20-ux-revamp`; the Python
package is `/home/samjd/projects/RewardSculptor`.

## Goal and decision

Sam asked us to begin the highest-impact items in
`docs/internal/REWARDSCULPTOR_RESEARCH_GRADE_ROADMAP.md`, rigorously and for
all current/future robot embodiments—not only G1 or gripper robots. I chose the
scientific-integrity foundation first because new World features cannot
support publishable claims while campaigns can mix experimental identities or
evaluators lack a human anchor.

## Implemented in this slice

1. **Frozen campaign charter**
   (`RewardSculptor/sculptor/eval/charter.py`, integrated in `eval/harness.py`)
   freezes config, exact benchmarks/conditions, analysis and failure policy,
   runtime source/dependency hash, and external-input hashes before GPU/LLM
   work. Resumes require an exact match. Existing results without a charter
   are rejected rather than retroactively “pre-registered.” Every result and
   report carries the charter design hash. `sculpt eval charter OUT` verifies
   charter and result lineage. Also fixed the recorded final-selection rule
   for fitness-guided sculpt/mission runs (`best_across_iterations`).

2. **KG isolation for paired campaigns** (`eval/harness.py`). The old harness
   opened the mutable shared KG, so later seeds/conditions could inherit cases
   from earlier jobs. The runner now makes one transactionally consistent
   `campaign_inputs/kg_base.db`, charters its hash, and gives each KG-enabled
   `(benchmark, condition, seed)` job a private writable copy. Only a crash
   resume of that exact job reuses its KG.

3. **Robot-agnostic metric gauntlet and blind human-anchor tooling**
   (`eval/gauntlet.py`). `sculpt eval gauntlet build` validates a private
   labeled video manifest, balances behavior-class comparisons, creates two
   exactly counterbalanced A/B forms, adds optional hidden reliability
   repeats, strips media metadata/audio with ffmpeg, anonymizes assets, and
   writes hashed public packets plus a separate tamper-evident key.
   `sculpt eval gauntlet analyze` validates JSONL responses and reports human
   accuracy/consensus, Krippendorff alpha, repeat consistency, order bias,
   evaluator-human agreement, evaluator accuracy, false competence by exploit
   class, and task/robot/embodiment/motion breakdowns with pair-level records.
   No humanoid-, leg-, hand-, or gripper-specific assumptions are in this
   code. Protocol: `RewardSculptor/docs/metric_gauntlet.md`.

4. Updated `README.md` and `docs/campaign_plan.md`. The old “different
   benchmark lists writing to one output” sharding recipe is now explicitly
   unsafe because it describes different frozen designs. Do not weaken the
   charter to restore it; implement a charter-aware global-matrix coordinator
   later.

5. **Capability-described external benchmark suites**
   (`eval/benchmarks.py`, `docs/benchmarks/`). `--benchmark-manifest` adds
   strict, non-overriding tasks with robot/embodiment/task families, required
   capabilities, evaluation tier, readiness, limitations, and evaluator
   authority. The real registered YAM lift and multi-cube arm/gripper tasks are
   included as honest `compile_only` frontiers: the generic rollout explicitly
   lacks object/end-effector/grasp telemetry, so the harness refuses to launch
   them as benchmarks. A campaign-ready external task must cite a verified
   passing A4 audit certificate; merely naming a metric is rejected. Manifest
   and certificate hashes enter the campaign charter. Built-ins remain usable
   but are visibly labeled `legacy_provisional` in events, JSON, HTML, and CLI.

6. **Adversarial success-spec certificates** (`eval/spec_audit.py`, protocol
   in `docs/spec_audit.md`). `sculpt eval spec-audit` evaluates frozen artifacts
   against precommitted score bounds for competent, still, fallen, oscillatory,
   explosive, early-terminated, threshold-flicker, reset-artifact,
   time-truncated, and proxy-only cases. Authority coverage is explicit from
   A1 through A4; missing coverage or one failed case yields A0. Evidence
   inputs, manifest, per-case outcome, and certificate are hashed, and old
   output cannot be silently overwritten.

## Verification

- Focused final: 47/47 across manifests, A4 audit enforcement, campaign,
  gauntlet, spec audit, and Eureka paths.
- Broad final: 2,078 passed, 1 optional-JAX skip in 3m31s with
  `tests/test_refs_preview.py` excluded.
- Preview subset: 10 non-GL tests passed. One existing preview ingest test
  failed because X11 display `:0` is unavailable; the two real MuJoCo renderer
  tests can abort the Python process in this headless WSL environment. This is
  environmental and outside the touched paths.
- Real ffmpeg smoke passed with valid generated MP4s, two counterbalanced
  packets, stripped/anonymized assets, and verified hashes.
- `compileall`, CLI help, and `git diff --check` passed.

## Files and worktree caution

Do not delete or stage the pre-existing untracked `.fleaven*`, `.ingest*`,
`.metric*`, or `.pytest*` artifacts in `RewardSculptor/`. The roadmap document
and this handoff are also untracked new work. No commit was made.

## Continued 2026-07-19 (Claude): generic manipulation telemetry

The first paragraph of the "best next implementation target" below is now
implemented (`sculptor/adapters/manipulation_telemetry.py`, wired into
the rollout path of `_mjlab_runner.py`):

- **Discovery is purely semantic.** The robot is the scene entity whose
  cfg declares actuators; objects are the passive entities; the
  end-effector site/body, left/right finger groups, and grasp
  eligibility come from the installed capability descriptors matched by
  role-name sets against a CPU compile of the robot asset. Zero robot or
  task name strings; a future arm inherits everything by registering a
  descriptor. Ambiguity (no robot, two actuated entities) records
  nothing rather than guessing.
- **Channels** (feature-detected per step, dropped when absent, authored
  ChannelCatalog vocabulary): `ee_pos_w`, `ee_quat_w`,
  `object__<name>__{pos_w,quat_w,lin_vel_w,ang_vel_w}` for EVERY
  embodiment (non-prehensile tasks included),
  `contact__{left,right,gripper}__<object>` from contact sensors
  injected pre-construction (primary = capability finger-group subtree,
  secondary = the object entity), `grasp__<object>` = bilateral finger
  contact (explicitly documented as a mechanical proxy, not
  force-closure), `target__pos_w` + per-step `target_object_index`
  (duck-typed from the command term: `target_selection`/`entity_names`
  for multi-object commands, static `entity_name` otherwise, remapped to
  the discovered object order). A sidecar
  `manipulation_telemetry.json` records provenance, channel shapes, and
  every derivation so the future lift spec and A4 audit consume declared
  semantics rather than guessed arrays.
- Authored runs are excluded (their catalog recorder is authoritative;
  no double-writing); the whole path is fail-soft and event-logged
  (`manipulation_telemetry_discovered`).
- The frontier manifest is updated to the new truth (suite_version
  1.1.0): the telemetry limitation is closed, the missing-A4-spec and
  frozen-split limitations remain, tier stays `compile_only` and
  authority stays `A0_rejected` — no false readiness.
- Tests: `tests/test_manipulation_telemetry.py` (7) — role-name
  capability matching (complete/incomplete), REAL registered YAM lift +
  multi-cube cfg discovery and idempotent sensor injection (CPU spec
  compile only), classification refusals, and recorder semantics on a
  fake env (bilateral grasp, target remapping, manifest content).
- **Observed end-to-end on GPU**, not only mechanism-verified: a
  3-iteration train + real rollout of `Mjlab-Lift-Cube-Yam` produced all
  11 channels in `trajectory.npz` (T=100, N=64) plus the sidecar
  manifest — capability matched `yam:parallel_gripper`, cube z within
  its spawn range, EE pose sane, and contacts/grasp honestly all-false
  for an untrained policy. Adversarially verified by a subagent
  (mutation test confirmed the target-remap test kills an identity-remap
  bug; all claims CONFIRMED).

Still open from the list below: the temporal lift-clear-and-hold spec,
genuinely held-out A4 evidence across the ten attack classes, the YAM
manifest promotion to campaign-ready, the real gauntlet manifest from
archived rollouts, and the charter-aware global-matrix coordinator.

## Best next implementation target

Continue the concrete YAM path rather than adding more placeholder task names:
extend the mjlab rollout artifact contract generically with end-effector pose,
object poses/velocities, target identity, gripper contacts, and grasp state;
do not key this on the string `Yam`. Use scene/capability discovery so future
arms and grippers inherit it. Then implement a temporal lift-clear-and-hold
spec, assemble genuinely held-out A4 evidence across the ten attack classes,
and only then version the YAM manifest from `compile_only` to campaign-ready.
In parallel, assemble the first real gauntlet manifest from archived
honest/gamed rollouts, keeping human-test clips disjoint from metric generation
and calibration. A charter-aware global-matrix coordinator is also needed
before restoring multi-pod sharding into one campaign output.

## Release-candidate UI + launch hardening 2026-07-20 (Codex)

Sam is now the sole operator and requires the lab-call workflow to be UI-only
after the one-time `./run.sh` startup. This slice makes the World-to-training
path honest and usable without terminal configuration:

- Fixed the previously cosmetic launch controls: `num_envs` and `device`
  overrides now flow from the React dialog through FastAPI, the subprocess CLI,
  `sculpt_run`, the effective `run_context.json`, and the adapter instance.
  Application is capability/attribute based, with no robot or task-name keying;
  project `config.toml` remains immutable.
- Added a fail-closed server gate before job submission. If a project has a
  promoted authored selection, the backend verifies the atomic tuple again and
  returns a 412 `/problems/world-integrity` before LLM/GPU work on drift. Legacy
  projects with no authored selection retain their existing built-in scene.
- Added localhost UI key configuration in Settings. It writes atomically under
  the user project-data root with mode 0600, activates immediately, restores on
  later launches when no environment key overrides it, and only returns a
  masked suffix. `run.sh` now uses pinned pnpm 9.12 through Corepack and gives
  correct guidance when only a UI-saved key may exist.
- Added a launch-readiness rail (LLM key, CUDA/mjlab/rsl_rl, authored tuple),
  three explicit plans (pipeline, rehearsal, overnight), scaled ETA, launch-time
  tuple revalidation, no-world and robot-mismatch confirmations, and truthful
  custom-plan state after manual edits. World now precedes Rewards in the
  project workflow and offers `Train this world` only with a valid selection.
- Added the full UI-only operator procedure and recovery path at
  `reward-sculptor-ui/docs/LAB_CALL_DEMO_RUNBOOK.md`; linked it from the UI
  README. The reliable showcase intentionally uses the built-in `go1_trot`
  ground-truth metric in steer mode instead of adding launch-time generated-
  metric variance.

Prepared local demo state (user data, not committed): project
`lab-call-authored-rough-terrain`, Unitree Go1 / mjlab /
`Mjlab-Velocity-Rough-Unitree-Go1`. World selection v2 was authored and
promoted entirely in the UI from:
“Traverse a parkour course of ascending boxes with gaps, moving forward
steadily without falling.” It materializes three platforms plus two gaps,
preserves v1 in lineage, and passes schema, capability, budget, build,
initial-penetration, settle, placement, and reachability gates. Selection v2
hash prefix is `af0d135c5b49`.

Verification on the final pre-commit tree:

- UI focused: 49 passed (`test_system.py`, `test_runs.py`).
- Core focused: 41 passed (`test_run_cli_overrides.py`, `test_sculpt.py`).
- Full UI backend: 554 passed in 3m04s.
- Full core command required by Sam: 2,137 passed / 1 optional-JAX skip in
  4m23s with `MUJOCO_GL=egl` and `test_refs_preview.py` excluded.
- Frontend `tsc -b && vite build`: passed (2,761 modules).
- Ruff on both new Python files, compileall on the touched Python trees, and
  `git diff --check`: passed. Whole legacy-file Ruff still reports pre-existing
  unused-import/E402 debt in large files, so the scoped new-file result is the
  claimed lint evidence.

The real 4-cycle GPU showcase run is intentionally launched only after this
release-candidate commit, so Vite/uvicorn reloads cannot interrupt it. Append a
second handoff section with its run id, artifact paths, timings, fitness values,
and rollout inspection before declaring the overnight rehearsal complete.

## Live-launch blocker found and fixed 2026-07-20 (Codex)

The first UI launch (`job_8bae6f6985e66471`) did its job as a rehearsal and
failed before completing iteration 0. It verified the new override plumbing
(`steps_per_iter=750`, `num_envs=1024`, `device='cuda:0'`) and pinned tuple
`af0d135c...`, then exposed a real authored-world integration bug: overlaying
the course's plane onto the registered rough task left mjlab's base
`terrain_levels` curriculum enabled. That curriculum asserts that a live
terrain generator exists during the first reset.

Fixed in `sculptor/world/compiler.py`: after either train or frozen-evaluation
scene overlay, reconcile curricula against the actual terrain capability. If
there is no live generator, remove only terms whose declared term/function
semantics are `terrain_levels`/`terrain_levels_*`; preserve unrelated terms
such as `command_vel`. The decision never inspects a robot or task name. The
runner emits the adjustment, and `ResolvedWorldBundle.runtime_adjustments`
records it for callers/tests.

Regression evidence:

- The new test builds and promotes a plane selection, applies it to the real
  registered `Mjlab-Velocity-Rough-Unitree-Go1` cfg, proves only
  `terrain_levels` is removed, and preserves `command_vel`.
- 150 focused world/env-spec/mjlab tests passed.
- Post-fix broad suite: 2,138 passed / 1 optional-JAX skip in 4m09s.
- A real CPU construction + reset with two environments and the actual local
  selection v2 succeeded (`RESET_OK af0d135c... ['command_vel']`). This reaches
  the exact reset site that asserted in the failed launch.
- Compileall and diff check passed. Scoped Ruff passed with `F401` ignored only
  because the two inspected legacy files already carry three pre-existing
  unused `mujoco` imports; the new code introduced no additional Ruff finding.

Also added ignore rules for the explicitly protected local `.fleaven*`,
`.ingest*`, `.metric*`, and `.pytest*` transcripts. The files remain untouched
and unstaged, but they no longer make `run_context.json` falsely label a clean
source commit dirty.

For the call, use the clean successor project
`lab-call-authored-parkour` (“Lab Call — Authored Parkour”), not the rehearsal
project above. It was created entirely through the UI after the fix, has no
failed-run history, and promoted the matching parkour prompt as World selection
v1 with tuple-hash prefix `34caeae995be`; all eight admission gates passed.
The lab-call runbook now points to this final project.

## Full showcase recovery hardening 2026-07-20 (Codex)

The clean UI-only overnight run for `lab-call-authored-parkour` launched as
`job_78e64f36f2525215` from source `b38a6b8...`. Iteration 0 completed the
entire train/rollout/audit/diagnose/edit pipeline: 750 PPO iterations, a
7,029,739-byte checkpoint, 500-frame MP4 plus trajectory/reward trace, realism
audit `ok`, raw return `37.4471`, firewall fitness `0.00095`, and three grounded
reward edits. The low firewall fitness honestly identified reset-like travel
despite a high simulator return, which is useful lab-call evidence of the
metric firewall doing real work.

Iteration 1 then exposed a long-horizon dependency failure near its final PPO
update: rsl_rl's registered Go1 config uses an unconstrained trainable
`GaussianDistribution` standard deviation (`std_type="scalar"`); one action's
value crossed below zero and `torch.normal` raised `normal expects all elements
of std >= 0.0`. The World build, GPU memory, and physics were healthy. The run
was left intact as failure evidence and was not resumed across a changed source
hash.

Recovery is generic and policy-capability based. `_mjlab_runner.py` now detects
only distributions exposing the legacy direct `std_param`, clamps an invalid
resumed value before sampling, and installs a PyTorch optimizer post-step hook
that enforces a `1e-4` minimum before the next minibatch. Log-std and
non-Gaussian distributions remain untouched. The hook is removed
deterministically after training. The UI backend also classifies this exact
signature as `policy_distribution_instability` with an actionable explanation
instead of an empty unknown-error card.

Focused evidence before the replacement launch: 36 mjlab-adapter tests and 12
error-classifier tests passed; scoped Ruff, compileall, and `git diff --check`
passed. The replacement must be launched as a new UI run after committing this
fix so its immutable run context records the new clean source hash. Preserve
the original errored run and its `iter_0` artifacts for provenance.

## Completed UI-only lab showcase 2026-07-21 (Codex)

The replacement run completed successfully from the UI with no terminal-side
launch configuration:

- Project: `lab-call-authored-parkour`; job:
  `job_434b10c7d3fd8eb2`; clean source:
  `8a2de1d45c8e40eb55544d9397054b3f9f0a498e`.
- Start/end: `2026-07-21T00:12:52Z` → `2026-07-21T02:34:26Z`
  (2 h 21 m 34 s).
- Exact controls: 4 outer cycles, 750 rsl_rl iterations/cycle, 1024 envs,
  `cuda:0`, 500 episode steps, two rollout episodes, seed 42, 960×540 video,
  `go1_trot` in steer mode with patience 4, KG enabled.
- Fitness: iter 2 `0.00159`, iter 3 `0.20701`, iter 4 `0.26848` (selected
  best), iter 5 `0.23281`. Iter indices continue the project's preserved
  provenance; the job still ran exactly four new cycles.
- Best selection pinned iter 4 reward `v4.py`, env spec v1, tuple
  `2ec5d679...`, and then ran a fresh held-out seed 90001. Fresh fitness was
  `0.25805` versus selected fitness `0.26848`.
- Visual inspection confirmed coherent, upright, sustained locomotion in the
  selected rollout and its fresh replay. It did not clearly show platform
  clearing, so the defensible claim is reproducible authored-world optimization
  and locomotion progress, not solved parkour.
- Best video:
  `~/.local/share/reward-sculptor/projects/lab-call-authored-parkour/runs/iter_4/rollout/rollout.mp4`.
  Fresh replay:
  `~/.local/share/reward-sculptor/projects/lab-call-authored-parkour/runs/iter_4/rollout_fresh_0/rollout.mp4`.

Post-run hardening fixes three honest demo gaps. `_mjlab_runner` now moves
CUDA-backed actuator/joint limit tensors to CPU before NumPy snapshotting, so
realism metadata no longer emits a nonfatal conversion warning. The Run Detail
API rehydrates all recorded `RunParams` rather than falsely returning null for
the advanced UI controls that were actually used. The New Run ETA is calibrated
from the completed Go1 run and models PPO iterations, environment count, fixed
cycle overhead, and final fresh evaluation; the old UI showed 44 minutes for a
job that took 2 h 21 m.

The Results tab is now call-ready without filesystem or terminal work. Policy
fitness comes from each iteration's authoritative `fitness.json` (with the old
`behavior.json` field retained only as a legacy fallback), so iter 4 is visibly
marked best. A dedicated evidence card shows its selection rollout and the
separate fresh held-out replay side by side, labels the steering fitness, and
reports the fresh replay count. The backend serves fresh rollouts through a
bounded project/iteration/index route rather than exposing local paths. The UI
`Build report` action was exercised on the real project and successfully
rendered the reward/reference/changelog report.

Final verification on this uncommitted slice:

- Full core suite: 2,142 passed / 1 optional-JAX collection skip in 4 m 58 s
  with the required `MUJOCO_GL=egl` command and `test_refs_preview.py` excluded.
- Full UI backend suite: 558 passed in 3 m 10 s.
- Focused post-run suites: core export + mjlab adapter 70 passed; UI runs,
  policies, and project-disk routes 68 passed.
- Frontend production build passed (2,761 transformed modules).
- Scoped Ruff, compileall, and `git diff --check` passed.
- Browser verification covered the World integrity/provenance view, the
  measured 2.3 h launch ETA, both real video endpoints, and report generation.

The complete UI-only rehearsal and exact call script live in
`reward-sculptor-ui/docs/LAB_CALL_DEMO_RUNBOOK.md`. Keep the scientific claim
narrow: the run demonstrates an immutable authored-world tuple, end-to-end
reward/environment optimization, metric-firewall selection, and fresh-seed
locomotion reproduction. It does not yet demonstrate solved platform clearing.

## Critical parkour correction 2026-07-21 (Codex)

Sam visually rejected the showcase, correctly: Go1 never mounted a platform
and ran away from the visible course. The old run is **not valid parkour
evidence**. Direct inspection of `iter_4/rollout/trajectory.npz` found the
root cause, not a subjective policy-quality issue:

- mjlab placed the 64 rollout robots at tiled `scene.env_origins`, while the
  compiler emitted only one static course at global `(0, 0)`;
- both NumPy and Torch waypoint runtimes compared world-space robot positions
  to unshifted local waypoint coordinates;
- initial first-waypoint distances were therefore 1.1–10.5 m depending on the
  environment tile, and every `goal__complete_course__waypoint_index` remained
  exactly zero for all 500 recorded steps;
- the waypoint target also used the platform box-center Z in full 3D, making a
  robot root on top of a platform fail a nominal 0.25 m reach tolerance;
- the sculpted `command_vel` channel requested obsolete command term
  `base_velocity`, while installed mjlab calls the actual observed velocity
  command `twist`, so the custom velocity term silently trained against zeros;
- the base velocity task randomized spawn yaw over ±π and issued lateral,
  backward, and turning commands despite the authored linear course being +X.

Immediate source fixes (intentionally committed without tests at Sam's
request):

1. The course/zone spec editor discovers mjlab's authoritative
   `env_origin_<n>` sites and emits one high-contrast, high-friction copy of
   authored geometry per unique environment origin. This fixes collision and
   rollout rendering together, including non-Go1 robots because the logic is
   scene-semantic only.
2. Region and waypoint producers translate world-space entity positions into
   each environment's local frame. Course reach/progress uses XY distance;
   physical platform geometry, rather than an unreachable root-height target,
   enforces climbing.
3. Velocity command capture discovers a term by its `lin_vel_x/lin_vel_y/
   ang_vel_z` range contract, supporting current `twist`, legacy
   `base_velocity`, and future semantic equivalents without task-name keying.
4. Waypoint-course application structurally aligns reset pose/yaw and forward
   velocity commands with course +X, disables lateral/turning/standing command
   sampling, and removes only the velocity-command curriculum that would later
   re-widen those ranges.
5. The Results evidence card now warns explicitly that selection fitness is not
   authored-goal completion and requires task-channel plus visual validation.

Because the compiler/runtime source changed, the old promoted tuple must fail
exact-match verification. Re-author/promote the same World prompt in the UI to
produce a new admitted tuple, then launch a new run. Do not reuse the old
checkpoint or show its Result card as successful evidence. No tests were run
for this emergency slice per Sam's explicit instruction; the next agent should
review and test it before claiming the replacement run works.

## Tracking-first reward checkpoint 2026-07-21 (Codex)

Implemented HANDOFF.md Task 1. Referenced stages now seed a deterministic
16-phase joint/velocity/root/orientation tracking base before v1 generation.
Only `_residual_task_numpy` and `_residual_task_batched` are editable; the
targets, hash, kernels, weights, final composition wrappers, fall gate, and cap
are mechanically frozen in both LLM and manual UI edit paths. No-reference
stages are unchanged. The Rewards tab shows an explicit tracking/residual band,
clip id, and target hash. Also fixed the outstanding World-runtime
`env_origins` regressions with a zero-origin fallback for single-world scenes.

Evidence: core 2,178 passed / 1 optional-JAX skip; UI backend 562 passed;
focused suites 51 + 25 passed; frontend typecheck/build passed; scoped Ruff,
compileall, and diff check passed. Next: run HANDOFF.md §6 live browser
verification, fix every UI defect, and commit that evidence before Task 2.

## Prompt-native compound objective validation 2026-07-22 (Codex)

Fixed the live four-box mission's no-reference validation failure without
introducing robot/task-name special cases. The generated metric already emitted
an embodiment-neutral `ABSTRACT_OBJECTIVE`, and the promoted World already
provided competent task-channel fixtures, but the validator evaluated them as
separate positives: `catalog_competent` inherited a physically still rollout.
Consequently any honest metric requiring both traversal and waypoint success
scored zero even though each half of the validator existed.

`metric_validate.py` now composes the prompt-derived task-space traversal with
the competent authored-channel state for `catalog_competent`. Negative catalog
cases remain still plus their failing task state, preserving the firewall.
Phase programs accept up to 12 ordered/repeated phases, and dwell/land/recover
receive duration-weighted windows; this makes real pause/settle gates possible
within a fixed 120-frame synthetic rollout (the previous equal eight-way split
was shorter than smoothing plus the requested hold duration).

No tests were run for this emergency slice, per Sam's explicit request. Editing
the reloader-watched core file restarted the UI backend and interrupted the old
in-memory mission job `job_dabcf4eb43259bda`; regenerate the affected stage
metric in the UI against this committed validator before using it as evidence.

## Authoritative prompt-native objective compiler 2026-07-22 (Codex)

Closed the remaining no-trajectory mismatch exposed by the recent Unitree Go1
box-course generations. Metric authoring and validation now receive one shared,
system-compiled `ABSTRACT_OBJECTIVE`: the generator is told to copy its ordered
phases exactly, while the validator treats the prompt compiler as authoritative
and uses a generated module's declaration only when the deterministic compiler
has no vocabulary for the goal. Untrusted metric code can therefore no longer
weaken its own validator by collapsing a multi-box course into a generic jump.

The compiler now recognizes singular/plural box, platform, step, and level
language; box-to-box / each-box sequences; pauses; and terminal “as far as
possible” jumps. These compound goals resolve as novel traversal rather than a
stationary jump family, preventing calibration and nondegeneracy from selecting
the wrong built-in archetype. The existing embodiment-neutral retargeter then
synthesizes the competent physical/task-channel probe directly from the prompt;
no stored trajectory and no robot/task-name keying are required. The New Run UI
now says this explicitly beside metric generation.

No tests were run for this emergency slice, exactly as Sam requested. `git diff
--check` was clean. Regenerate the Go1 objective metric from the UI; prior
rejected `gen_001`/`gen_002` artifacts retain their old validation record and
must not be represented as having passed the new compiler.

## Live UI recovery + bounded training evidence 2026-07-22 (Codex)

Verified the recovery/UI changes against a real in-app-browser launch of
`tracking-first-ui-verification / four-box-parkour-demo`, entirely through the
product UI. The launch dialog now rejects an invalid 2-step override inline
without losing the application, then accepted a bounded 1-round, 100-step,
64-environment GPU run. The live Training view streamed launch, reference,
training, rollout, objective-fitness, realism-audit, and reward-edit events;
its GPU and fitness panels stayed responsive. The Overview robot viewer was
checked in Live, Static (including all camera choices and re-render), and
Replay modes. The live view now correctly correlates a `rollout_done` event
that omits `iter` with its preceding `iter_started` event (commit `0e47450`).

The bounded run produced iter-0 MP4/checkpoint/export links, objective fitness
0.0, and an honest “criterion unmet” selection. It is plumbing evidence, not
parkour-success evidence: the robot terminated after eight rollout steps, and
the mission replanner opened an upright-balance sub-stage. That sub-stage's
first model edit had a syntax error and entered the existing retry path. The
mission was then stopped through the UI's Stop button and native confirmation.
The persisted UI returned to `Ready`; the active sub-stage showed `Stopped
(mission cancelled)`, `ws closed`, and `run_stopped (user)`. A fresh browser
console contained no errors (only the two pre-existing React Router v7 future
flag warnings).

This run was created before the authoritative prompt-native metric compiler
commit `a2f8ead`, so its disk artifact correctly still shows the older rejected
blind fallback. Do not use it to claim the compiler fix or successful course
completion. The next meaningful UI run must regenerate/promote the objective
metric, use a newly admitted post-parkour-fix World tuple, and allow enough
training budget to judge locomotion and platform traversal rather than only
the UI/runtime path.

Verification already completed for the two UI commits: frontend typecheck and
production build passed after both `433aed0` and `0e47450`; `git diff --check`
was clean. No additional automated tests were launched during the live run.

## Prompt-only abstract validator expansion 2026-07-22 (Codex)

Objective-metric validation now uses the independently compiled, embodiment-
neutral `ABSTRACT_OBJECTIVE` trace as the competent positive for every novel
goal and every compound goal when no stored reference trajectory exists, not
only for traversal/parkour. The same trace is composed with the authored-world
competent channel fixture, so physical motion and task-state completion must
agree. Recognized single-skill goals retain their hardened fixed archetypes;
real references remain stronger evidence when present. `meta.json` now persists
the exact abstract program plus `stored_trajectory_required: false`, making the
prompt-only validator basis visible and auditable from UI artifacts. No tests
were run, per Sam's explicit emergency instruction; only `git diff --check` was
performed.

## Authored-world policy observability fix 2026-07-22 (Codex)

The first post-shape-fix G1 slalom training run was stopped after a read-only
audit proved it could not learn the authored route: TaskSpec observation
bindings were persisted but never installed into MJLab actor/critic inputs;
explicit waypoint-zone routes skipped reset/command alignment whenever the
obstacle `course` list was empty; and region-relative reward channels mixed
local zone coordinates with replicated world-space robot positions. The core
compiler/runtime now installs body-frame region, object, semantic end-effector,
and authored height-scan observations for every compatible embodiment in both
train and evaluation, recognizes explicit named waypoint routes for alignment,
and localizes NumPy/Torch region channels by `env_origins`.

Evidence: focused suites 58 passed; GPU runtime construction on the real G1
selection produced actor shape `(2, 168)` with five authored region vectors and
the 54-ray authored scan, command ranges `(0.45,1.0)/(0,0)/(0,0)`, no command
curriculum, and finite `(2,3)` finish-relative samples. The invalid run
`job_08a12eb8b01dea07` was stopped before relaunch.

## Compound prompt validator chronology fix 2026-07-22 (Codex)

The remaining Go1 no-trajectory failure was a chronology bug in the shared
abstract-objective compiler, not a need for a robot-specific demonstration.
For “jump from each box to the next, pause on each, jump as far as possible,
land, then run,” the compiler previously dropped the explicit landing and the
terminal run. The generated metric therefore required post-landing evidence
that its independent competent validator never performed, and every validation
archetype scored exactly zero.

The compiler now treats `jump_off` and an explicitly requested `land` as
separate phases, recognizes run/sprint/dash/race language, and defers that
planar phase until after a staged climb/jump/landing course. The exact prompt
now compiles to `climb, dwell, climb, dwell, jump_off, land, move_forward`,
which is also what both recent Go1 metric candidates independently declared.
The existing embodiment-neutral retargeter supplies the kinematics, contact
schedule, and authored-world competent channels; a stored trajectory remains
optional. The active G1 slalom job was cooperatively stopped before editing the
reload-watched core so its partial artifacts were not killed mid-write. No
tests or test-like validation commands were run, per Sam's explicit request.

## Goal-conditioned authored route commands 2026-07-22 (Codex)

The first corrected-observation G1 slalom rollout proved the policy could
locomote (one rendered environment reached roughly x=7.8 m), but it traveled
almost straight along one side: no environment advanced beyond the first
ordered waypoint. The root cause was the world compiler's fixed +X velocity
command. It directly contradicted any authored route with lateral turns, while
the sculpted reward paid for speed magnitude rather than target-directed
velocity.

Authored `waypoint_sequence` tasks now replace any compatible base velocity
command with a robot/task-name-independent `WaypointVelocityCommand`. It keeps
private per-environment route state, continuously rotates the current authored
target direction into the robot body frame, commands both planar velocity and
yaw toward that target, slows near each gate, advances only inside the frozen
goal tolerance, and emits exactly zero after final completion so terminal hold
and stillness are learnable. Tasks without a velocity-command surface remain
untouched. The existing command observation and base velocity-tracking reward
now provide dense, directed supervision without leaking metric-only completion
channels into reward code.

Evidence: focused compiler contract 13 passed; focused world/runtime group 41
passed; a real CUDA G1 smoke constructed the 168-D authored observation set,
reported `WaypointVelocityCommand` as the live command term, completed one PPO
iteration, and wrote a checkpoint. The first smoke reached construction but
stopped at an unconfigured W&B login; rerunning with the same disabled-W&B
setting used by the UI completed cleanly. The old slalom job was already
cooperatively stopped, so no active GPU work was interrupted by this slice.

## Live G1 route learning + visible task zones 2026-07-22 (Codex)

The UI-launched showcase job `job_00197adcc90c9911` is actively training
selection v8 / reward v3 / metric `gen_003` on the GPU; do not edit the
reload-watched core while its `_mjlab_runner train` worker is alive. The
goal-conditioned command produced real ordered progress: checkpoint 300
reached waypoint 1 in 15/64 environments; checkpoint 350 reached waypoint 1
in 49/64 and waypoint 2 in 15/64 while surviving the full 1,000-step rollout.
A different-seed checkpoint-400 robustness sample reached 33/64 and 3/64.
No sample has reached waypoint 3 yet, so this is learning evidence, not a
success claim. Only 4/64 checkpoint-350 environments touched any forbidden
box; the validator honestly returned completion 0 and progress 0.030.

The World tab's exact-scene viewer had a separate demo-facing bug: MuJoCo zone
sites are thin cylinders, but every non-sphere site was rendered as a box,
turning `[radius, 0.01, 0]` into invisible zero-Z geometry. `WorldViewer3D`
now honors sphere, ellipsoid, cylinder, capsule, and box site geometry using
MuJoCo's Z-axis and half-size conventions. The live compiled slalom scene now
shows four alternating green waypoint disks plus the larger finish disk;
selecting a disk highlights it and exposes its exact authored parameters.
Frontend typecheck, production build, `git diff --check`, and live UI visual /
interaction checks passed before commit.

The two pre-existing React Router v7 future-flag warnings were also removed by
opting `BrowserRouter` into `v7_startTransition` and `v7_relativeSplatPath`.
A fresh live Training tab then reported an empty warning/error console; the
frontend typecheck and production build passed for this follow-up as well.

`reward-sculptor-ui/docs/LAB_CALL_DEMO_RUNBOOK.md` was replaced with the
current G1 showcase workflow: exact world and behavior prompts, UI-only metric
generation, exact 4×750 / 1024-env / 1,000-step / 1080p launch settings,
physical acceptance criteria, call narrative, honest incomplete-run fallback,
and preserved warnings about the invalid historical Go1 evidence. Its current
status paragraph intentionally says the live run is incomplete until the
official selected rollout proves the full course and terminal hold.

## Frozen abstract objective companion 2026-07-22 (Codex)

Prompt-only metric generation now treats the abstract task program as one
first-class contract shared by metric authoring and its independent validator.
When the deterministic compiler recognizes the prompt, the exact ordered phase
list is passed through every best-of-N, retry, and review-repair validation call;
a generated metric whose `ABSTRACT_OBJECTIVE` omits, merges, renames, or reorders
those phases is rejected. This closes the remaining seam where author and
validator could silently certify different interpretations of the same prompt.

For genuinely novel prompts outside the deterministic parser, generation no
longer freezes an empty "authoritative" phase list. The metric author must emit
a non-empty, inert `ABSTRACT_OBJECTIVE` companion beside `compute_spec`; the
validator literal-parses that data (never executes it), retargets it onto the
universal task-space probe and any exact authored-world channels, and rejects a
missing/empty companion. The exact phase program actually used is persisted in
`meta.json`. This is embodiment-neutral and requires no stored trajectory; real
references, when present, remain additive stronger evidence rather than a
prerequisite. No tests were run for this urgent slice, per Sam's explicit
instruction; only `git diff --check` was performed before commit.

## Full-strength authored command supervision 2026-07-22 (Codex)

The interrupted G1 slalom trace exposed a generic weighting bug: the compiled
`WaypointVelocityCommand` was producing the correct body-frame lateral/yaw
targets and the policy tracked them accurately, but `_cmd_train` multiplied
MJLab's `track_linear_velocity` and `track_angular_velocity` terms by the same
0.3 realism-floor scale used for posture/smoothness priors. The generated reward
only observed speed magnitude and waypoint distance, so the only dense signal
that distinguished left/right route turns was unnecessarily three times weak.

When—and only when—the immutable World manifest declares a
`waypoint_sequence` and the compiler confirms it actually installed a
goal-conditioned command on a compatible base environment, those two nominal
command-tracking rewards now retain full weight. All other default terms remain
at the 0.3 realism floor, and Worlds with no installed command surface remain
unchanged. Detection uses schema/runtime-adjustment semantics only; there is no
robot name or simulator task-id keying. Focused adapter/compiler/runtime tests:
58 passed. Scoped Ruff, compileall, and `git diff --check` passed.

## Interrupted-train policy recovery 2026-07-22 (Codex)

The UI-launched showcase run ended during outer iteration 3 when the backend's
reload watcher reacted to an urgent core edit. It had valid rsl_rl checkpoints
through `iter_3/logs/model_600.pt`, but no promoted `checkpoint.pt`; the old
resume path therefore discarded that current-iteration policy and warm-started
again from the previous outer iteration.

`_train_or_resume` now scans an incomplete iteration's existing
`logs/model_<iteration>.pt` files newest-first, verifies each with `torch.load`,
and uses the newest parseable policy as the warm start. A torn newest file falls
back to the next valid model. The recovery emits `partial_train_recovered` with
the exact source and any superseded prior-iteration warm start. Adapters without
the existing `init_policy_path` contract remain on their prior path. Focused
resume/warm-start tests: 57 passed; compileall and `git diff --check` passed.
Whole-file Ruff still reports unrelated pre-existing unused imports/locals in
the 7k-line `sculpt.py` and its historical test module; no finding points at the
new recovery code.

## Prompt-native validator composition + restart-safe recovery 2026-07-22 (Codex)

The first broad verification after the frozen abstract-objective slice exposed
23 regressions: legacy/generated candidates that omitted the new inert
declaration were rejected even when the deterministic prompt compiler already
owned an exact phase program. Generation now composes that frozen data literal
into every best-of-N, normal retry, and review-repair candidate as a system
operation. It inserts after module docstrings and `__future__` imports, leaves
explicit author declarations untouched so drift is still rejected, and leaves
truly novel empty compiler programs to the jointly-authored companion path.
The contract is never inferred from a stored trajectory.

The abstract probe also no longer mistakes directional adjectives such as
"kick forward" or "bend forward" for an extra locomotion phase. Bend/bow
programs retarget torso tilt onto symmetric named hip flexion and recover that
pose, making prompt-native compound validators non-vacuous without raw joint
index assumptions in production logic.

A live stop/resume revealed a second edge case in interrupted training: rsl_rl
restarts its numeric model counter, so a newly written `model_50.pt` can coexist
with an older `model_600.pt`. Partial-policy recovery now orders candidates by
filesystem write time (numeric counter only breaks ties), verifies newest to
oldest, and therefore preserves the actual newest learning. The live UI trace
showing full-strength authored command supervision and partial recovery is saved
as `reward-sculptor-ui/.ui-verification/2026-07-22-lokesh-final/29-live-resume-command-supervision-light.png`.

Verification: focused metric + warm-start suites 144 passed; broad CPU suite
2,213 passed / 1 expected optional-JAX skip in 4m22s; compileall and
`git diff --check` passed. Scoped Ruff still reports only historical E702/F401/
F841 findings in the long pre-existing validator/test modules, with no finding
on the newly added composition or recovery lines.

## Official iter-3 slalom audit + reset-safe terminal phase 2026-07-22 (Codex)

UI job `80d549c83b41a134` completed cleanly from the recovered iter-3 policy.
The frozen official card says `fit 0.00 / progress 0.005`, but trajectory and
video inspection isolated two different facts that must not be conflated:

- Route competence is real: 59/64 environments reached ordered waypoint index
  5, 58/64 asserted authored success, 51/64 completed without any forbidden
  box contact, and no environment fell. The full 20-second 1080p video is
  valid and visibly shows traversal; its earlier `moov atom not found` clip
  warning was a race while ffmpeg was still writing.
- Terminal stillness remains incomplete: after excluding the automatic reset
  sample, terminal speed averaged 0.145 m/s and only one environment satisfied
  the exact continuous two-second `<0.12 m/s` gate. This is not yet a solved
  claim.

MJLab auto-resets inside `step`. The runner recorded that next-episode state as
the final trajectory row: every root jumped roughly 7.6 m back to its spawn
while goal channels remained complete. `gen_003` therefore measured 3.92 m/s
terminal speed and 0.019 m net travel. Removing only that reset row raises the
same frozen metric's dense progress from 0.005 to 0.810 and exposes the one
true full-gate environment, without changing the underlying policy.

Subsequent rollouts now persist a per-environment first-episode validity mask
and replace post-reset samples with absorbing last-valid state instead of
stitching a second attempt. The authored channel recorder's private route state
is reset per done environment. The finish command also uses a longer 2 m
braking approach only for the final target, while intermediate gates retain
their crossing-speed floor. Finally, `rollout_done` is emitted only after the
MP4 and all artifacts are fully closed, eliminating the UI clip race. None of
this rewrites the frozen iter-3 evidence.

Verification: focused compiler/runner/runtime 60 passed; focused channel,
generated-metric, fitness, and realism 137 passed; broad CPU suite 2,215
passed / 1 expected optional-JAX skip in 4m35s; scoped Ruff, compileall, and
`git diff --check` passed before commit.

## Gap-safe UI resume warm-start 2026-07-22 (Codex)

The first post-fix UI launch (`a640baaa954fc1f7`) correctly pinned clean code
`354fee1`, reward v5, env v1, and immutable selection v11, but exposed a
separate generic resume gap before useful GPU work began. Prompt-authored
rewards had advanced v3 to v5 without an `iter_4`; `sculpt_run` used the reward
number as `start_iter` and passed no warm start, so iter 5 began from random
weights instead of the competent iter-3 policy. The job was cooperatively
stopped in the UI at RL iteration 0 and produced no policy checkpoint.

UI Resume now resolves the newest valid policy from actual preceding
`runs/iter_<N>` directories whenever the new iteration has no exact-tuple
checkpoint or partial model and the user did not supply an explicit init
policy. It searches across missing indices, prefers a promoted checkpoint
within an iteration, falls back to the newest parseable partial model, and
skips corrupt newer artifacts. Explicit user choice remains highest priority;
same-iteration crash recovery still supersedes the preceding-policy warm start
inside `_train_or_resume`. The logic is adapter/robot/task-name independent and
the adapter retains final checkpoint compatibility validation.

Verification: focused warm-start suite 20 passed; broad CPU suite 2,217 passed
/ 1 expected optional-JAX skip in 4m54s; scoped Ruff E9/F63/F7/F82, compileall,
and `git diff --check` passed before commit.

Committed as `919d20c`. The replacement UI job is
`55bbca2ef13a4c4a`: clean `919d20c`, reward v5 + env v1, iter selection v12 /
tuple `785b6c62f942d41b250618c4bda3bb5a2d53023fb12b51bf8299c0efd11eedef`,
one 750-PPO cycle, 1,024 envs, seed 42, two 1,000-step 1080p rollouts,
`gen_003` observe-only, Auto. Its live log proves
`resume_warm_start_resolved` from `runs/iter_3/checkpoint.pt` (sha8
`05de8e0f`) and `warm_start_loaded` for actor + critic before PPO iteration 0.
The train worker is active: do not edit reload-watched core or run GPU audits
until it finishes.

## Official iter-5 audit + finite terminal arrival 2026-07-22 (Codex)

UI job `55bbca2ef13a4c4a` finished and preserved
`runs/iter_5/checkpoint.pt`. The official first-episode-safe trajectory and the
complete 20-second 1920×1080 MP4 prove that the automatic diagnosis is wrong:
the rendered robot visibly traverses past the four-box course, and an
independent region-crossing audit finds all four intermediate regions in exact
order for 64/64 environments. The batch remained upright with zero falls;
62/64 entered the finish and 57/64 had no forbidden contact.

The run is still not accepted as solved. Only 39/64 reached authored waypoint
index 5 before timeout, 16/64 asserted the authored success channel, terminal
horizontal speed averaged 0.13373 m/s, the final-window still fraction averaged
0.561, and only environment 33 satisfied the full conjunctive validator. The
rendered environment reached index 5 with zero contact and correct order but
ended at 0.157 m/s with only 43% of its last two seconds below the speed gate.
The full video and late keyframes show the actual failure: upright route
completion followed by foot/torso shifting at the finish, not off-course
loitering.

The 2 m terminal brake introduced after iter 3 used a linear speed scale with
no floor. It therefore approached the 0.35 m completion tolerance
asymptotically: median index-5 time regressed from step 799 in iter 3 to step
966 in iter 5, leaving too little dwell. The embodiment-neutral
`WaypointVelocityCommand` now retains a 0.35 minimum speed scale until the
finish tolerance is crossed, scales yaw authority down with the terminal
approach instead of circling the target, then latches the base command's
standing semantic and exact zero velocity. Intermediate route behavior is
unchanged.

Diagnosis is hardened without weakening the metric firewall. Every authored
rollout now writes a batch-wide `reward_visible_rollout_evidence` summary into
`behavior.json`, derived only from catalogued `shared_shaping` progress and
entity-motion channels over each environment's first episode. All
`metric_only` success/contact/objective channels are structurally excluded.
The preliminary prompt must treat this 64-environment evidence as more
representative than four frames from one percentile-labelled episode. On the
archived iter-5 data, the safe summary reports goal waypoint-distance median
2.16975→0.0 and finish-relative magnitude median 8.03509→0.79244, which directly
prevents the false “never approached the route/finish” claim without exposing
held-out completion truth.

The UI's filesystem watcher also now waits for post-encode `behavior.json`
before emitting its one-shot `rollout_done` and starting clip generation. The
runner's explicit event was already correctly ordered, but the synthetic file
event fired as soon as a growing MP4 exceeded 2 KiB and exhausted the bounded
ffmpeg retry before the moov atom landed.

Verification: focused core 65 passed; focused UI clip suite 7 passed; broad core
2,218 passed / 1 expected optional-JAX skip in 4m44s; compileall,
`git diff --check`, and scoped Ruff (ignoring only pre-existing F401 findings)
passed. A pre-existing test-double drift exposed by the full UI suite was also
corrected: metric generator fakes now accept the already-production
`channel_catalog` argument; the affected metrics/run group is 49 passed.
The complete UI backend suite then passed 563/563 in 3m08s.
The false auto-generated reward v6 and env v3 remain preserved as provenance,
but `selection_current.json` still correctly pins reward v5 + env v1 (selection
v12). Do not train v6/env v3. The next action is a UI Resume from iter 5 after
this code commit, using the pinned v5/env-v1 tuple and one focused recovery
cycle; inspect the same full acceptance conjunction afterward.

## UI exact-promoted-tuple recovery 2026-07-22 (Codex)

The prior instruction to resume v5/env-v1 exposed a UI integrity gap: ordinary
Resume intentionally trains the newly diagnosed mutable `rewards/current.py`
and `env/current.json`, which now point to the preserved-but-rejected v6/env-v3
drafts. Selecting v5 in the Rewards viewer does not and should not silently
change training inputs. The user therefore had no honest UI-only way to reject
those drafts and continue from atomic selection v12.

New Run → Advanced now has an explicit **Resume exact promoted tuple** recovery
switch (off by default so normal iterative resumes still consume their new
drafts). When enabled, the backend locks the artifact store, reads
`selection_current.json`, verifies the tuple hash and every referenced artifact
SHA-256, confines reward/env refs to their project-local version stores,
validates/compiles both sources, restores `rewards/current.py` and
`env/current.json`, and emits `promoted_tuple_restored` with selection, tuple,
artifact versions, and hashes before spawning `sculptor.cli`. Any mismatch
emits `promoted_tuple_restore_failed` and prevents the subprocess/GPU from
starting. This path is artifact-kind driven and contains no robot/task-name
keying. Reward-pointer rewrites now also use tmp+replace so readers cannot see
a truncated module after a crash.

Verification: three new recovery tests pass, including exact restore, hash-drift
rejection with both mutable pointers unchanged, and restore-before-subprocess
ordering/provenance. TypeScript project checking passes; compileall and scoped
Ruff pass (ignoring only the repository's pre-existing E402/F401 findings).
Browser verification against the live app shows the new switch with selection
v12 and exact hash-verified recovery language. The next launch must enable this
switch, use one 750-PPO recovery cycle, Auto, 1,024 envs, seed 42, two 1,000-step
1080p episodes, and `gen_003` observe-only. Confirm the restore event names
reward v5/env v1 before PPO begins, then audit the full conjunctive acceptance
gate.

Committed as `83413d9`. The corrected continuation is now running from the UI
as `job_556e643b0b1ad22b` with exactly those settings. Its visible event log
recorded `promoted_tuple_restored` before any training output: selection v12,
tuple `785b6c62f942d41b250618c4bda3bb5a2d53023fb12b51bf8299c0efd11eedef`,
reward v5 SHA `67680111…`, and env v1 SHA `35b6122c…`. The actual worker command
loads `runs/iter_5/checkpoint.pt`, trains against `rewards/current.py` and
`env/current.json` after restoration, and pins `env/selection_v13.json`; that
iteration artifact has the same tuple hash and exact reward-v5/env-v1 refs.
PPO is active. Do not edit reload-watched core or run intermediate GPU audits.
After it finishes, inspect the official first-episode-safe trajectory, metric
and fitness artifacts, keyframes, and full MP4 against the entire acceptance
conjunction before making any further change.

## Official iter-6 audit + terminal whole-body supervision 2026-07-22 (Codex)

UI job `556e643b0b1ad22b` completed from exact promoted selection v12 and
preserved `runs/iter_6/checkpoint.pt`. The exact-restore event, immutable
tuple hashes, iter-5 warm start, full authored command weights, and valid
20-second 1920×1080 MP4 are all proven in the official artifacts.

The policy now solves route traversal but still does not satisfy the literal
terminal requirement. Independent first-episode inspection found ordered
region crossings in 64/64 environments, waypoint index 5 in 63/64, finish
entry in 63/64, uprightness in 64/64, and zero forbidden contact in 55/64.
The rendered environment reached index 5, entered the finish, stayed upright,
and had zero box contact with terminal horizontal speed 0.08203 m/s. The full
video visibly shows the weave and an upright finish.

This is not accepted as solved: batch terminal speed averaged 0.11179 m/s but
only 41/64 ended below 0.12 m/s, and no environment remained continuously
below 0.12 m/s for all 100 frames of the required two seconds. The longest
continuous quiet run was 75 frames. `gen_003` reports three full-gate
environments because its frozen hold proxy asks for more than 90% quiet
samples rather than true continuity; that score remains useful observe-only
evidence but cannot override the stronger physical acceptance audit.

The rollout also emitted
`WorldChannelRuntime object has no attribute reset` after simulator
auto-reset. The generic runtime now implements selective per-environment reset
for hold, predicate, and waypoint temporal state, preventing later episodes
from inheriting a completed route. A second generic addition installs dense
whole-body stillness supervision only when a compiled authored command both
advertises terminal standing and has a positive dwell contract. It combines
horizontal base velocity, full angular velocity, and joint RMS velocity and is
identically zero before route completion. Discovery uses compiled command
capabilities and scene articulation surfaces, never an embodiment or simulator
task name. Existing full-weight linear and angular command tracking is
unchanged.

Verification: focused runtime/adapter suite 51 passed in 13 seconds; scoped
Ruff, compileall, `git diff --check`, and the focused suite all passed. The
next UI Resume should consume the iter-6 diagnosis drafts reward v7/env v4
(settling weight 1.6, finish double support, entropy scale 0.75), leave exact
promoted-tuple recovery off, warm-start iter 6, and run one 750-PPO recovery
cycle. Require the terminal-stillness provenance line before PPO starts, then
repeat the full continuous-hold audit.

That recovery is now active as UI job `5f7e50d020ead92c`, iter 7, from clean
commit `2b84fab`. The UI launched one 750-PPO cycle with 1,024 environments,
seed 42, two 1,000-step 1920×1080 episodes, Auto, and `gen_003` observe-only.
Normal Resume correctly pinned reward v7 + env v4 as selection v14 / tuple
`de07325bab038d29fa6705148f795d201d8159c42d93b8ddd92c4ec41f2226db`.
The live log proves warm-start loading of actor + critic from iter 6 (sha8
`ee4ab29e`), full-weight authored linear/angular command supervision,
goal-conditioned terminal braking, entropy coefficient 0.0075, and terminal
whole-body stillness supervision at weight 1 before PPO iteration 0. Do not
edit reload-watched core or run an intermediate GPU audit while iter 7 is
alive. The heartbeat monitor has been updated to this job and the literal
100-frame acceptance rule.

## Official iter-7 audit + continuity-validator hardening 2026-07-22 (Codex)

UI job `5f7e50d020ead92c` completed and preserved
`runs/iter_7/checkpoint.pt`. The official first-episode mask contains 999
valid samples for every environment (the timeout/reset row is correctly
absorbed). All 64 environments independently crossed the four authored disks
in order, reached waypoint index 5, entered the actual 0.9 m finish disk, and
remained upright; 55/64 had zero forbidden contact and 62/64 asserted the
authored two-second success channel. Mean terminal horizontal speed improved
from 0.11179 to 0.09803 m/s, with 52/64 terminal means below 0.12 m/s.

The literal continuous hold improved from 0/64 to 4/64. Environments
18, 31, 39, and 57 each produced an uninterrupted post-completion,
inside-finish, upright run of at least 100 speed samples below 0.12 m/s; the
longest was 156 frames. Those same four satisfy the full physical conjunction,
including zero contact. The rendered environment 0 completed the course in
order at step 742, asserted authored success at 842, entered the finish with
zero contact, remained upright, ended at exactly zero instantaneous speed, and
averaged 0.06675 m/s over the terminal window. It still reached only 63
consecutive quiet frames, so the official video cannot yet be claimed as the
literal two-second demonstration. Visual inspection of the full valid
20-second 1920×1080 MP4 and 5 Hz terminal sheet agrees: a correct weave and
calm upright finish remain punctuated by small foot/arm adjustments.

`gen_003` reports 6/64 completion because its immutable gate accepts more
than 90% quiet samples. Future prompt-native metrics are now hardened:
duration-qualified or explicitly continuous dwell goals receive an adversarial
competent fixture with nine sparse 0.30 m/s interruptions across the terminal
two seconds. It retains over 90% quiet samples and all authored completion
state but has no uninterrupted hold. The metric must floor that fixture at
0.05 or below. The authoring prompt also explicitly requires consecutive-run
logic using duration / step_dt and forbids mean, percentile, or fraction
proxies for continuity. This uses universal root motion plus compiled task
state and contains no robot/task-name keying.

Do not train automatic drafts reward v8/env v5 next. Their diagnosis claims
0.24 m/s residual motion despite the official frozen trajectory measuring
0.09803 m/s, and both reward edits were partition-gate flagged. Preserve them
as provenance. The next safe continuation is **Resume exact promoted tuple**
from selection v14 (reward v7/env v4), warm-starting iter 7 for one more
750-PPO cycle with the same UI settings. This keeps the proven route and
stillness gradient while rejecting stale/partition-flagged drafts.

Verification for the continuity hardening: channel-catalog metric suite 10/10;
generated metric, reference-generation, and spec-metric suites 124/124;
compileall, `git diff --check`, and scoped Ruff passed. Ruff ignored only the
file's pre-existing E402/F401/E702 debt; no new finding remains.

## Iter-8 exact-tuple continuation launched 2026-07-22 (Codex)

The safe continuation is now running from the UI as job
`3b5f34bedc5af06d`. New Run used the exact behavior goal above, Auto, one
750-PPO cycle, 1,024 environments, seed 42, two 1,000-step 1920x1080
episodes, `gen_003` observe-only, and **Resume exact promoted tuple**.

The pre-training provenance is fully verified. The UI emitted
`promoted_tuple_restored`, and iter 8 pinned `selection_v15.json` with the
same promoted tuple
`de07325bab038d29fa6705148f795d201d8159c42d93b8ddd92c4ec41f2226db`.
Its immutable refs are reward v7 SHA
`b6c65d349b9f23f5b36de68ec25eb5d48879ac5b84f1aa86a30949e5a4290df9`
and env v4 SHA
`db049dafa3fb1fa0bc5ce590c485ec62c469b3bfa4aab757255b36adcadcbb39`.
The captured code is clean commit `c28e36a604e00540aed341c9ae6699a23a4705c1`.

The worker resolved and loaded `runs/iter_7/checkpoint.pt` (sha8
`6cb79ed3`) for both actor and critic. The live log also proves
goal-conditioned waypoint traversal with terminal braking, full authored
`track_linear_velocity` and `track_angular_velocity` weights of 2.0,
entropy coefficient 0.0075, and `sculptor_terminal_stillness` weight 1.0.
PPO is active. Do not edit reload-watched core or run an intermediate GPU
audit while this worker is alive.

After iter 8 finishes, preserve/promote its checkpoint and inspect the official
first-episode-safe trajectory, objective/fitness artifacts, keyframes, and
full MP4. Acceptance still requires the entire physical conjunction, including
zero forbidden contact and a literal uninterrupted 100-frame post-completion,
inside-finish, upright run below 0.12 m/s. `gen_003`'s 90%-quiet proxy is not
sufficient by itself.

## Official iter-8 audit + continuity-aware supervision 2026-07-22 (Codex)

UI job `3b5f34bedc5af06d` completed from the exact promoted reward-v7/env-v4
tuple and preserved `runs/iter_8/checkpoint.pt`. The official rollout contains
999 valid first-episode rows for every one of 64 environments and a valid
20-second 1920x1080 MP4. Frozen `gen_003` reports fitness 0.13858, progress
0.83574, completion gate 18/64, contact in 5/64, zero falls, success seen in
62/64, and terminal speed 0.10458 m/s.

The stricter independent audit found 62/64 complete ordered crossings of all
four actual 0.45 m waypoint disks plus the actual 0.9 m finish disk, 62/64
waypoint-index-5 and authored-success observations, 59/64 zero-contact
trajectories, and 64/64 terminal uprightness. Terminal mean horizontal speed
was 0.10560 m/s; 52/64 terminal means were below 0.12 m/s. Nine environments
(`4, 12, 13, 24, 34, 37, 48, 49, 50`) satisfied the literal uninterrupted
100-frame post-completion, inside-finish, upright hold below 0.12 m/s, and all
nine also had zero contact and the full ordered route. This improves the
literal conjunction from 4/64 to 9/64.

The rendered environment 0 completed all five disks in order, reached index 5
at step 759, asserted authored success at 859, stayed upright, and had zero
forbidden contact. It ended inside the finish at distance 0.2914 m with
terminal mean speed 0.09638 m/s, but its longest uninterrupted quiet run was
only 41 frames. Visual inspection of the official full and terminal contact
sheets agrees: the weave and upright finish are clear, but small corrective
steps remain. The showcase therefore remains honestly incomplete.

The remaining control defect is generic: frame-wise dense stillness gives
nearly the same aggregate credit to a continuous dwell and to quiet samples
separated by corrective steps. Terminal supervision is now a stateful reward
term with a private per-environment quiet streak. It activates only when a
compiled compatible waypoint command exposes terminal standing and a positive
authored dwell duration. Consecutive progress grows toward that exact duration;
an interruption loses the accumulated progress and resets the streak.
Selective reward-manager resets clear only the affected environments. Dense
horizontal, angular, and joint-velocity shaping remains, and no embodiment,
task id, or prompt name is used.

Verification: focused Mjlab adapter suite 47/47 passed; scoped Ruff,
compileall, and `git diff --check` passed. The automatic iter-8 diagnosis
created env v6 and filtered both reward edits. Preserve it as provenance and
do not train it: the next safe run must again use **Resume exact promoted
tuple** from selection v15 (reward v7/env v4), warm-start iter 8, and run one
750-PPO recovery cycle. Confirm the new continuity-aware supervision line
before PPO begins.

## Completed-iteration Resume allocator hardening 2026-07-22 (Codex)

The first post-fix UI Resume, job `6d377b31a78cfbca`, restored the exact
selection-v15 tuple correctly but exposed a generic counter bug before any GPU
worker launched: reward numbering still pointed at 8, so the runner selected
completed `iter_8`, skipped its checkpoint and rollout, and repeated only the
diagnosis/edit phases. That no-op run is preserved as failure provenance. It
created unused automatic drafts reward v9 and env v7; do not train them. The
next exact-tuple restore must continue from immutable reward v7/env v4.

The runner now writes an atomic `iteration_complete.json` only after an
iteration has completed training, rollout, objective/realism evaluation,
diagnosis, and edits. Resume begins at the latest reward number and advances
only across a contiguous sequence of valid, matching completion markers.
Missing, corrupt, wrong-schema, wrong-state, and wrong-index markers retain
the existing same-iteration crash-resume path. This closes the no-edit/no-op
case without weakening partial-run recovery or assuming contiguous reward and
run indices. The UI log emits `iteration_completion_marked` at completion and
`resume_completed_iterations_advanced` when the marker, rather than a new
reward file, advances the counter.

Focused orchestrator + warm-start verification is 65/65 passing, including
contiguous advance, invalid-marker rejection, gap preservation, and an
end-to-end dry-run that checks the durable marker. Scoped Ruff (with only the
files' recorded pre-existing debt ignored), compileall, and `git diff --check`
also pass. Relaunch from the UI only after this slice is committed; expected
next iteration is 9 and its live command must warm-start from
`runs/iter_8/checkpoint.pt`.

The first real iter-9 launch, UI job `d81e0e8471509b7a`, proved the allocator
fix (`iter 9` appeared immediately) but failed before PPO during reward-manager
construction. Mjlab instantiates class-backed reward terms with keyword
arguments `cfg=` and `env=`; the new continuity term had named the first
argument `_cfg`, so Python rejected the manager's exact contract. The
constructor now accepts the canonical `cfg` name and explicitly discards the
unused value. Its regression test also constructs the term with the real
keyword call. No GPU training or checkpoint was lost. Focused Mjlab adapter
verification is 47/47 passing; Ruff, compileall, and `git diff --check` pass.
Relaunch the same exact-tuple iter-9 configuration and require the continuity
installation line plus actor/critic warm-start before PPO iteration 0.

The corrected UI relaunch is active as job `dde47f043fe792ec` from clean commit
`e1b5d50`. It restored selection v17, then pinned selection v18 with the same
tuple `de07325bab038d29fa6705148f795d201d8159c42d93b8ddd92c4ec41f2226db`.
The runner correctly allocated iter 9, resolved `runs/iter_8/checkpoint.pt`,
and loaded both actor and critic (source sha8 `8a7e6a83`). Live configuration
proves goal-conditioned waypoint traversal with terminal braking, full
`track_linear_velocity` and `track_angular_velocity` weights of 2.0, entropy
coefficient 0.0075, and continuity-aware terminal stillness at weight 1 with
`hold_s=2` and continuity scale 2. PPO iteration 0 is active. Do not edit
reload-watched core or run an intermediate GPU audit while this worker lives.
After completion, apply the same strict official-artifact/video acceptance
audit used for iter 8.

## Official iter-9 audit + timestep-invariant continuity 2026-07-23 (Codex)

UI job `dde47f043fe792ec` completed, atomically marked iter 9 complete, and
preserved `runs/iter_9/checkpoint.pt`. The frozen metric reports fitness
0.10508, progress 0.84581, ordered course 64/64, authored success 63/64,
contact in 8/64, zero falls, terminal speed 0.09795 m/s, and its permissive
completion gate in 14/64.

The independent first-episode-safe audit used 999 valid samples in every
environment and the actual authored geometry: four 0.45 m horizontal waypoint
disks and the 0.9 m finish disk. It found 63/64 actual ordered course+finish
traversals, 63/64 waypoint-index-5 and authored-success observations, 56/64
zero-contact trajectories, and 64/64 without a sustained fall. Fourteen
environments produced a literal uninterrupted 100-frame post-completion,
inside-finish, upright hold below 0.12 m/s; ten of those also had zero
forbidden contact and therefore satisfy the full physical conjunction. The
longest hold was 172 frames. Terminal mean speed was 0.09883 m/s, with 52/64
terminal means below 0.12 m/s.

Rendered environment 0 again completed the entire route with zero contact and
no fall: waypoint entries were steps 121/268/431/569, finish entry 674,
waypoint index 5 at 774, and authored success at 874. Its longest literal hold
improved from 41 to 85 frames and terminal mean speed improved to 0.07587 m/s,
but final instantaneous speed was 0.18832 m/s. Full and terminal video sheets
agree that the robot is upright and inside the finish while continuing small
corrective steps. The showcase remains honestly incomplete.

The audit exposed a generic units bug in the continuity term. Mjlab's reward
manager integrates term values by multiplying them by `step_dt`; the
state-potential difference was returned as a dimensionless per-step delta, so
both the gain for building a streak and the penalty for breaking it were
attenuated by 0.02 at 50 Hz. The term now returns the potential difference as
`delta / step_dt`. Its integrated effect is therefore invariant to simulator
timestep, and a corrective step loses the accumulated potential at the
intended strength. Dense stillness and phase gating are unchanged; there is no
robot, task, channel-name, or prompt-name keying.

Focused Mjlab verification is 47/47 passing, including a regression requiring
a partial-streak interruption to remain strongly negative after downstream dt
integration. Ruff, compileall, and `git diff --check` pass. Preserve automatic
reward v10/env v8 as diagnosis provenance and do not train them: both reward
edits were partition-gate flagged. The next safe run is one exact-tuple
750-PPO continuation from selection v18 (reward v7/env v4), warm-starting the
iter-9 checkpoint after this fix is committed.

## Iter-10 timestep-correct recovery launched 2026-07-23 (Codex)

The safe continuation is active from the UI as job `100d2d25b054acf2`.
New Run used the exact behavior goal above, Auto, one 750-PPO cycle, 1,024
environments, seed 42, two 1,000-step 1920x1080 episodes, `gen_003`
observe-only, and **Resume exact promoted tuple**.

Launch provenance is verified. The UI emitted `promoted_tuple_restored` for
selection v18, then iter 10 pinned `selection_v19.json` with the unchanged
tuple
`de07325bab038d29fa6705148f795d201d8159c42d93b8ddd92c4ec41f2226db`,
reward v7 SHA
`b6c65d349b9f23f5b36de68ec25eb5d48879ac5b84f1aa86a30949e5a4290df9`,
and env v4 SHA
`db049dafa3fb1fa0bc5ce590c485ec62c469b3bfa4aab757255b36adcadcbb39`.
The captured code is clean commit `f30f14a639a09184446338352f0534a9d7502f00`.

The worker resolved `runs/iter_9/checkpoint.pt` (sha8 `75129866`) and loaded
both actor and critic. Its live log proves goal-conditioned waypoint traversal
with terminal braking, full authored `track_linear_velocity` and
`track_angular_velocity` weights of 2.0, entropy coefficient 0.0075, and the
timestep-correct continuity-aware whole-body stillness term at weight 1,
`hold_s=2`, and continuity scale 2. PPO iteration 0 is active.

Do not run an intermediate GPU audit or edit reload-watched core while this
worker is alive. After it completes and preserves/promotes the checkpoint,
inspect the official first-episode-safe trajectory, objective/fitness
artifacts, keyframes, and full MP4. Acceptance remains the full physical
conjunction: ordered actual waypoint disks, actual finish entry, index 5, zero
forbidden contact, no sustained fall, uprightness, terminal horizontal speed
below 0.12 m/s, and a literal uninterrupted 100-frame post-completion hold
inside the finish. The frozen metric's 90%-quiet proxy is not sufficient.

## Official iter-10 audit + balanced terminal supervision 2026-07-23 (Codex)

UI job `100d2d25b054acf2` completed, preserved
`runs/iter_10/checkpoint.pt`, and atomically marked iter 10 complete. Frozen
`gen_003` reports fitness 0.16316, progress 0.83372, ordered-course evidence
in 63/64, success in 62/64, contact in 6/64, falls in 2/64, terminal speed
0.09115 m/s, and its permissive completion gate in 19/64. The realism audit
is clean: no joint-limit or torque saturation, no reset launch, and no
naturalness rejection.

The independent first-episode-safe audit used the actual authored horizontal
geometry and every valid sample. It found 62/64 actual ordered traversals of
all four 0.45 m waypoint disks plus the 0.9 m finish disk, 62/64
waypoint-index-5 and success observations, 58/64 zero-contact trajectories,
and 62/64 full-length/no-sustained-fall trajectories. Twenty-five environments
held continuously for at least 100 frames after index 5 while inside the
finish, upright, and below 0.12 m/s. Twenty-three (`6, 7, 10, 12, 15, 16, 17,
21, 22, 30, 31, 32, 33, 34, 35, 40, 41, 45, 50, 51, 52, 57, 62`) also had
the complete ordered route, zero forbidden contact, success, and no fall, so
they satisfy the full physical conjunction. This more than doubles iter 9's
10/64 full-conjunction count. The longest hold was 147 frames; mean and median
longest holds were 88.52 and 91.5 frames. Terminal mean speed averaged
0.10063 m/s; 53/64 terminal means were below 0.12 m/s.

Rendered environment 0 again completed every actual disk in order, with
entries at steps 131/287/450/592/689, reached waypoint index 5 at 778 and
authored success at 878, and had zero forbidden contact and no fall. Its
terminal mean speed was 0.07826 m/s, but its longest literal hold was only
90 frames and final instantaneous speed was 0.44162 m/s. The official
keyframes, full-video sheet, terminal-video sheet, and trajectory agree: the
weave and upright finish are visible, but the robot performs small corrective
steps through the terminal phase. The rendered showcase therefore remains
honestly incomplete despite the substantial batch-level improvement.

Training traces explain the remaining imbalance. Near PPO iteration 749,
`sculptor_terminal_stillness` contributed about 4.6 episodic reward while
full-strength linear and angular command tracking contributed about 110.7
combined. The strict dwell objective was phase-gated correctly and the command
was already zero after route completion, but its signal was too small relative
to the broad zero-command tracking kernels.

Terminal stillness weight is now computed generically from the compiled
command contract: it is at least the sum of the absolute live weights of the
authored command-tracking terms, with the existing 1.0 floor. For this tuple it
becomes 4.0, matching the 2.0 linear plus 2.0 angular supervision. The term
remains inactive before route completion, so it cannot trade navigation for
standing early. The calculation uses only installed command capabilities and
term weights—no robot, simulator task, channel, or prompt name.

Focused Mjlab adapter verification is 48/48 passing; scoped Ruff, compileall,
and `git diff --check` pass. Automatic reward v11 and its paired environment
draft remain diagnosis provenance only: both reward edits were
partition-gate flagged. The next safe run is one UI exact-promoted-tuple
continuation from selection v19 (reward v7/env v4), warm-starting iter 10
after this balanced-supervision slice is committed.

## Iter-11 balanced-hold recovery launched 2026-07-23 (Codex)

The next safe recovery is active from the UI as job `801b0549f5be7328`.
New Run used the exact 484-character behavior goal, Auto, one 750-PPO cycle,
1,024 environments, seed 42, two 1,000-step 1920x1080 episodes, `gen_003`
observe-only, and **Resume exact promoted tuple**.

The UI restored selection v19 and the runner pinned `selection_v20.json` for
iter 11 with the unchanged tuple
`de07325bab038d29fa6705148f795d201d8159c42d93b8ddd92c4ec41f2226db`,
reward v7, and env v4. Run context captured clean commit `356a188`. The worker
resolved and loaded `runs/iter_10/checkpoint.pt` (sha8 `11d72466`) for both
actor and critic. Live configuration proves goal-conditioned route traversal
with terminal braking, full linear/angular tracking weights of 2.0, entropy
coefficient 0.0075, and continuity-aware terminal stillness at balanced
weight 4, `hold_s=2`, continuity scale 2. PPO iteration 0 is active.

Do not run an intermediate GPU audit or edit watched core while iter 11 is
alive. After completion, preserve/promote its checkpoint and repeat the
strict actual-geometry, first-episode-safe trajectory, objective/fitness,
keyframe, and full-video audit. The full physical conjunction and rendered
100-frame hold remain the acceptance gate.

## Official iter-11 audit + whole-body continuity gate 2026-07-23 (Codex)

UI job `801b0549f5be7328` completed, preserved
`runs/iter_11/checkpoint.pt`, and atomically marked iter 11 complete. Frozen
`gen_003` reports fitness 0.11491, progress 0.83198, ordered-course evidence
in 61/64, success in 60/64, contact in 7/64, a fall in 1/64, terminal speed
0.07942 m/s, and its permissive completion gate in 14/64. The realism audit
remains clean, although joint-velocity p99 increased from 10.95 to 17.80.

The strict first-episode-safe audit found 60/64 actual ordered route-and-finish
traversals, 60/64 index-5 and success observations, 57/64 zero-contact
trajectories, and 61/64 full-length/no-sustained-fall trajectories. Twenty-seven
environments produced a literal uninterrupted 100-frame horizontal hold;
twenty-five (`2, 3, 4, 6, 7, 10, 12, 16, 18, 22, 24, 26, 38, 39, 41, 42,
47, 50, 51, 52, 53, 55, 56, 59, 60`) also satisfied the complete ordered
route, zero-contact, success, and no-fall conjunction. This is a small
batch-level improvement over iter 10's 23/64, but route reliability and falls
regressed.

Rendered environment 0 completed the actual disks at steps
121/275/423/561/671, reached index 5 at 756 and success at 856, and had zero
contact and no fall. It finished at 0.3372 m from center, with terminal mean
speed 0.09416 m/s and final speed 0.08116 m/s, but its longest literal quiet
run was only 82 frames. Keyframes, the complete video sheet, and the terminal
sheet all show a repeated wide-stance sway. The showcase remains incomplete.

The weight correction worked mechanically: terminal-stillness contribution
rose from about 4.6 to 20 episodic reward. It exposed the next generic loophole.
Continuity considered only horizontal base speed, so a policy could step in
place, rotate, and swing joints without resetting its nominal uninterrupted
streak. The rendered behavior and increased joint-velocity p99 are direct
evidence of that exploit.

Continuity now requires the same whole-body quiet contract used by the dense
stillness score: horizontal speed below 0.12 m/s, angular speed below 0.5
rad/s, and joint RMS velocity below 1.0 rad/s. Any translation, rotation, or
joint-driven stepping loses the accumulated streak potential. Dense shaping,
the exact horizontal task threshold, terminal phase gate, timestep correction,
balanced weight, frozen metric, and command contract are unchanged. The logic
uses robot state capabilities generically and has no embodiment, task,
channel, or prompt-name keying.

Focused Mjlab verification is 48/48 passing, including independent
horizontal, joint-only, and angular interruption regressions plus selective
reset. Ruff, compileall, and `git diff --check` pass. Iter 11 produced no
applicable reward revision: both LLM edits were rejected by validation. Its
environment draft remains unpromoted diagnosis provenance. The next safe run
is one UI exact-tuple continuation from selection v20, reward v7/env v4,
warm-starting iter 11 after this whole-body gate is committed.

## Iter-12 whole-body-hold recovery launched 2026-07-23 (Codex)

UI job `b12d93ae0b4b3029` is the active one-cycle exact-tuple recovery. New Run
retained the exact behavior goal and used Auto, 750 PPO iterations, 1,024
environments, seed 42, two 1,000-step 1920x1080 episodes, and `gen_003`
observe-only. The UI restored selection v20; iter 12 pinned
`selection_v21.json` with the unchanged promoted reward-v7/env-v4 tuple.

Run context captured clean commit `16d7c5e`. The worker resolved and loaded
`runs/iter_11/checkpoint.pt` (sha8 `e1833254`) for actor and critic. Live logs
prove goal-conditioned traversal with terminal braking, tracking weights 2.0,
entropy coefficient 0.0075, and the new whole-body continuity contract at
terminal weight 4, `hold_s=2`, continuity scale 2. PPO iteration 0 is active.

Do not run an intermediate GPU audit or edit watched core. After completion,
apply the same strict actual-geometry, first-episode-safe artifact, keyframe,
and full-video acceptance audit. The rendered uninterrupted 100-frame hold
remains mandatory.

## Official iter-12 audit + exact-contract continuation 2026-07-23 (Codex)

UI job `b12d93ae0b4b3029` completed, preserved
`runs/iter_12/checkpoint.pt`, and atomically marked iter 12 complete. Frozen
`gen_003` reports fitness 0.19524, progress 0.84841, ordered-course evidence
0.92183, finish evidence 0.79884, hold evidence 0.40625, stillness 0.41599,
uprightness 0.62220, its permissive completion gate in 24/64, contact in 6/64,
zero falls, and terminal speed 0.07633 m/s. The realism audit remains clean:
no joint-limit or torque saturation, no reset launch, and no naturalness
rejection; joint-velocity p99 improved to 9.77.

The independent first-episode-safe audit found 63/64 actual ordered
traversals of all four 0.45 m waypoint disks plus the 0.9 m finish disk,
63/64 index-5 and authored-success observations, 58/64 zero-contact
trajectories, and 64/64 full-length/no-sustained-fall trajectories.
Twenty-eight environments produced a literal uninterrupted 100-frame
horizontal hold after index 5 while inside the finish and upright; 26 also
satisfied the complete ordered-route, success, zero-contact, and no-fall
conjunction. Requiring the newly trained whole-body quiet contract as well
reduced those counts to 26 and 24 respectively. The horizontal longest-run
mean/median were 94.53/96.5 frames and the whole-body-qualified
mean/median were 89.95/89.5 frames. Sixty-one of 64 terminal mean speeds and
59/64 final speeds were below 0.12 m/s.

Rendered environment 0 completed the actual zones at steps
127/266/406/543/626, reached index 5 at 687 and authored success at 787, and
had zero forbidden contact and no fall. It finished 0.436 m from the finish
center with terminal mean speed 0.05849 m/s and final speed 0.01920 m/s. Its
longest literal horizontal hold was nevertheless only 92 frames, and its
longest whole-body-qualified hold was 81 frames. The official keyframes,
complete-video sheet, and terminal sheet agree: the route is real and clean,
but the robot still alternates small corrective steps and arm/torso motion
inside the finish. The showcase therefore remains honestly incomplete.

This is not a stalled supervision signal. The newly strict
`sculptor_terminal_stillness` episodic reward averaged 0.0 over PPO
iterations 0-99, then rose through 0.52, 8.74, 13.02, 15.41, 16.85, and
17.73 in successive 100-iteration blocks, reaching 19.54 over iterations
700-749. Route reliability stayed at 63/64 while terminal speed and
joint-velocity p99 improved. The highest-quality next action is therefore one
more exact-promoted-tuple recovery cycle from selection v21, warm-starting the
iter-12 actor and critic without changing the now-correct generic contract.
Do not train the automatic reward/environment diagnosis drafts; they remain
unpromoted provenance.

## Iter-13 exact-contract continuation launched 2026-07-23 (Codex)

The UI-launched recovery is active as job `7449bab7e0aa9fb9`. New Run retained
the exact 484-character behavior goal and used Auto, one 750-PPO cycle, 1,024
environments, seed 42, two 1,000-step 1920x1080 episodes, `gen_003`
observe-only, and **Resume exact promoted tuple**.

The UI emitted `promoted_tuple_restored` for selection v21 with tuple
`de07325bab038d29fa6705148f795d201d8159c42d93b8ddd92c4ec41f2226db`,
reward v7 SHA
`b6c65d349b9f23f5b36de68ec25eb5d48879ac5b84f1aa86a30949e5a4290df9`,
and env v4 SHA
`db049dafa3fb1fa0bc5ce590c485ec62c469b3bfa4aab757255b36adcadcbb39`.
Iter 13 pins `selection_v22.json` with the same tuple. Run context captured
clean commit `884cce054c0c6f46a61bf1681074d8858ba660af`.

The worker resolved and loaded actor plus critic from
`runs/iter_12/checkpoint.pt` (sha8 `40095ac5`). Its live configuration proves
goal-conditioned waypoint traversal with terminal braking, preserved
`track_linear_velocity` and `track_angular_velocity` weights of 2.0, entropy
coefficient 0.0075, and continuity-aware whole-body terminal stillness at
weight 4, `hold_s=2`, and continuity scale 2. PPO iteration 0 is active.

Do not run an intermediate GPU audit or edit reload-watched core while iter
13 is alive. Let it finish and preserve/promote its checkpoint, then repeat
the strict actual-geometry, first-episode-safe trajectory, objective/fitness,
keyframe, and full-video audit. The rendered uninterrupted 100-frame
post-completion hold remains the acceptance gate.

## Official iter-13 acceptance 2026-07-23 (Codex)

UI job `7449bab7e0aa9fb9` completed, preserved
`runs/iter_13/checkpoint.pt`, wrote the exact-tuple completion marker, and
closed normally. Frozen `gen_003` reports fitness 0.24734, progress 0.84141,
course 0.92196, finish 0.80196, hold 0.46406, stillness 0.42090,
uprightness 0.62443, its permissive completion gate in 29/64, contact in
6/64, zero falls, and terminal speed 0.08224 m/s. The realism audit is clean:
no torque or joint-limit saturation, no reset launch, no naturalness flag,
and joint-velocity p99 9.98.

The stricter first-episode-safe audit used 999 valid samples in every
environment and the actual authored horizontal geometry. It found 62/64
ordered traversals of the four 0.45 m waypoint disks plus the 0.9 m finish,
62/64 index-5 and authored-success observations, 58/64 zero-contact
trajectories, and 64/64 without a sustained fall. Twenty-five environments
produced a literal uninterrupted 100-frame horizontal hold and 23 satisfied
the complete horizontal physical conjunction. Twenty-one produced a
joint-qualified whole-body hold and 19 satisfied that full conjunction. This
batch conjunction count is lower than iter 12, so do not claim broad policy
robustness from this seed; the selected rendered rollout is the accepted
showcase evidence.

Rendered environment 0 is the first official clip to pass every showcase
gate. It entered the actual waypoint/finish disks at steps
122/273/405/547/640, reached waypoint index 5 at 691 and authored success at
791, had zero forbidden contact and no fall, and remained inside the finish.
From steps 887 through 998 it held continuously for 112 frames (2.24 s):
horizontal speed stayed below 0.12 m/s (max 0.11465, mean 0.04124, final
0.04408), joint RMS velocity stayed below 1.0 rad/s (max 0.94536, mean
0.40303, final 0.29952), projected-gravity z remained between -0.84656 and
-0.82860, base height remained 0.63114-0.63921 m, and finish distance remained
0.15674-0.17973 m.

The official keyframes, full 20-second video sheet, terminal sheet, and a
10-fps sheet covering the entire 2.24-second accepted window were inspected.
They visibly agree with the trajectory: the robot completes the alternating
course, enters the finish upright, and stays planted in essentially the same
pose throughout the verified window with no corrective step. The accepted
video is
`runs/iter_13/rollout/rollout.mp4`.

Automatic reward v13 and environment v12 are diagnosis provenance only. Two
reward edits were partition-gate flagged, and neither draft is part of the
accepted immutable selection v22 reward-v7/env-v4 tuple. Do not relaunch or
replace the accepted clip before the lab call. The complex UI workflow and
physical behavior are now proven for the selected official rollout.

## Selected-policy report provenance hardened 2026-07-23 (Codex)

The live Results tab correctly selected iter 13 and its official video, but
the generated showcase report exposed a presentation provenance bug: it
loaded the numerically latest reward draft (`rewards/v13.py`) and described an
older iteration as the ending behavior. Those drafts are diagnosis
provenance, not the immutable tuple that produced the accepted policy.

`sculptor/timelapse.py` now selects the highest-fitness completed policy
(newer iteration wins an exact tie), falls back safely for legacy projects,
and loads the reward module pinned by that iteration's `artifact_tuple.json`.
Report behavior, primary-metric endpoints, labels, and summary/edit deltas now
use the selected policy and list position, so sparse iteration numbering does
not corrupt provenance. Core selection remains robot/task-name independent.
A regression creates newer, lower-fitness completed policies plus a newer
reward draft and requires the report to select the best policy's pinned
reward.

Verification: focused timelapse tests 9/9 passed in 3.58 seconds; full
CPU-only suite passed with 2,231 tests and one optional-JAX skip in 281.65
seconds; scoped Ruff, compileall, and `git diff --check` passed. Rebuild the
Results report completed successfully in the live UI after commit `c129a72`;
its rendered markdown now says `Selected policy reward module:
rewards/v7.py` and `Selected (iter 13)`. Results still foregrounds iter 13 as
best with steering fitness 0.24734 and the accepted official rollout.

## Critical evidence correction + physical course fix 2026-07-24 (Codex)

The preceding "Official iter-13 acceptance" is **retracted**. Visual review
correctly exposed that the robot was not weaving around the rendered boxes.
The trajectory/metric route was expressed in the robot's per-environment
local frame, but the fixed mocap objects remained at unshifted global
coordinates. In iter 13 environment 0, the robot origin was approximately
`(7,-7)` while the four boxes stayed at global x positions 2/3.5/5/6.5.
The measured object-vs-manifest local-frame error is 9.899495 m for every box.
Local waypoint counts and the 112-frame hold therefore do not establish
physical slalom success.

Commit `11e664d` (`fix(world): align physical objects and add route RSI`)
repairs the general runtime contract:

- every authored fixed or floating object receives an explicit train/eval
  reset at `nominal local pose + env origin`;
- object-position train variations are composed as local deltas instead of
  overwriting the frame transform;
- generic train-only waypoint-route RSI splits episodes between the real
  entrance and positions just before later waypoints, facing the next target;
- evaluation resets are untouched;
- command, NumPy metric, and Torch reward consume the RSI route start
  consistently;
- a fixed-geometry alignment invariant blocks waypoint advancement/success
  whenever physical objects do not match the resolved manifest;
- all logic is capability/data driven, with no robot or task-name keying.

Focused physical-frame/RSI verification passed 35 tests plus compileall and
`git diff --check`.

Commit `55865b8` (`feat(results): reject misaligned physical evidence`) adds
`/physical-scene-audit` and a fail-closed Results presentation. The live UI
now says **Invalid evidence — physical scene mismatch · iter 13**, reports the
9.90 m maximum error, removes the "best" label, and disables policy export.
Backend audit/project tests passed 18 tests; TypeScript passed; the live
Results screen was visually inspected.

There is no corrected accepted rollout yet. A fresh UI-launched run under
`11e664d` or newer must pass the physical-scene audit and the original
ordered-route/contact/finish/hold gates before the showcase goal can be
declared complete.

## Normal-run reference-motion integration 2026-07-24 (Codex)

Commit `3d92602` (`feat(runs): add UI reference-motion priors`) makes the
existing reference-motion system available from the ordinary **New run**
workflow, not only mission stages/CLI:

- **Pre-existing motion → Choose motion** searches the target robot's
  reference library and shows previews, tier, license, and duration;
- launch sends an exact `(reference_robot, reference_clip_id)` pair and
  rejects incomplete/missing pairs before GPU work;
- the selected clip becomes the deterministic immutable phase-indexed
  tracking base already used by missions;
- the behavior prompt is authored only as a bounded residual, and the editor
  must preserve the reference target hash/composition;
- build finishes in a private staging directory before one immutable reward
  and one atomic authored tuple are created;
- exact resumes reuse only matching clip bytes, behavior goal, selected reward
  hash, and tuple hash;
- run events and `run_context.json` record the full motion provenance;
- route/object RSI remains independent, so motion style and physical
  exploration compose rather than substitute for one another.

Cross-robot use stays honest: a source motion must be retargeted/registered
into the target robot namespace through the existing GMR pipeline first.
There is no silent fallback to another embodiment. This works for any future
robot whose registered clip and adapter share the reward state contract;
core run logic has no robot/task-name branches.

Verification:

- reference-run + reference-tracking suites: 60 passed;
- sculpt orchestration suite: 45 passed;
- UI run/API suite: 48 passed;
- TypeScript, compileall, targeted Ruff on new core files, and
  `git diff --check` passed;
- a real G1 treadmill clip (24.825 s) compiled into a 19,481-byte tracking
  prior with recorded clip/target hashes;
- live UI picker/selection was exercised and visually inspected.

The current demo procedure is fully replaced by
`reward-sculptor-ui/docs/LAB_CALL_DEMO_RUNBOOK.md`. It explicitly treats iter
13 as invalid failure provenance and requires a fresh aligned rollout.

Final broad verification after both implementation slices:
`MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q
--ignore=tests/test_refs_preview.py` completed in 293.25 seconds with
**2,242 passed / 1 optional-JAX skip** (163 warnings, no failures).

## First aligned rollout diagnosis + contact-safe supervision 2026-07-24 (Codex)

Fresh UI job `job_22b38fd7ee6bfbb0` completed iter 14 under clean commit
`443a9c6`. It is the first result whose rendered geometry and abstract task
share the same frame:

- `/physical-scene-audit` reports `aligned`, maximum error `0.00 m`, all four
  boxes checked;
- every one of 64 evaluation environments entered all four actual waypoint
  disks in order and entered the finish;
- rendered env 0 reached waypoint index 5 at frame 733, ended inside the
  finish at 0 m/s, remained upright, and produced a 113-frame whole-body
  quiet hold;
- nevertheless, every environment recorded at least one forbidden box
  contact, only 37/64 met the no-sustained-fall proxy, and the full
  conjunction was 0/64;
- the full 20 s video/contact sheet visibly shows the robot and physical
  boxes co-located and a real alternating route, with contacts during the
  obstacle passes.

The root cause is generic. Reward v7's `box_disturbance` treated fixed-object
velocity as a contact proxy, but a fixed box remains at zero velocity even
when struck. At the same time, command tracking aimed at each waypoint-zone
center. The centers satisfy the abstract predicate but do not leave the
selected embodiment's full reach envelope clear of the obstacle.

Commit `0822e07` (`fix(world): supervise obstacle clearance from contacts`)
adds two capability/data-driven corrections:

- waypoint commands choose a safer subtarget inside the same authored disk
  when a forbidden object's horizontal bounding volume plus the selected
  robot capability's reach radius conflicts with the center target; the
  subtarget remains inside the task tolerance, and route RSI consumes those
  same safe targets;
- every authored forbidden contact pair installs direct binary supervision
  from its already-compiled simulator contact sensor, weighted to dominate
  the aggregate authored command-tracking income (`-8` for the current
  2+2 command weights).

There is no robot name, simulator task ID, object name, or prompt phrase in
the control logic. Box/sphere/cylinder/capsule/frame bounds and arbitrary
future robot capability geometry are handled through typed manifest data.

Verification for this slice: focused compiler/adapter suite **64 passed**;
compileall and `git diff --check` passed; scoped Ruff passed when excluding
three pre-existing F401 side-effect imports in the already-touched compiler
test surface. A new exact-tuple UI Resume must warm-start iter 14 and prove
zero contact plus the original route/finish/hold conjunction before this goal
is complete.

## Iter 15 improvement + premature-turn correction 2026-07-24 (Codex)

UI job `job_6988b3289674ac23` completed iter 15 under `0822e07` and clean
launch context `104eadf`. Direct contact supervision and the embodiment-aware
subtargets worked:

- scene audit: aligned, `0.00 m` maximum error;
- actual route + finish: 62/64;
- contact-free: **34/64**, up from 0/64;
- no sustained fall proxy: 51/64;
- horizontal 100-frame full conjunction: 9/64;
- whole-body 100-frame full conjunction: **6/64**.

The official rendered env 0 completed the route but contacted box 1 for three
frames and held only 51 frames. At those contact frames its local root moved
from approximately `(2.04, 0.54)` to `(2.07, 0.52)`. The nominal safe
subtarget was near `(2.0, 1.118)`, but the command's unchanged 0.35 m switch
radius declared that subtarget reached early and began the diagonal turn
toward the next negative-Y waypoint while still alongside box 1.

Commit `63dbc28` (`fix(world): reach clearance subtargets before turning`)
fixes the general set-versus-command distinction. When and only when a route
uses forbidden-object clearance subtargets, the command transition radius is
tightened to `max(0.08, 0.4 * task_tolerance)` (0.14 m here). The immutable
task predicate remains 0.35 m. Ordinary routes are byte-for-byte unchanged.
The adjusted command and route RSI share the same safe targets; no robot,
object, task, or simulator name controls the behavior.

Verification: focused compiler/adapter suite **64 passed**; scoped Ruff
(excluding the documented pre-existing F401 imports), compileall, and diff
check passed. Relaunch through UI exact recovery from iter 15 is required.

## Iter 16 route-stall diagnosis + clearance approach control 2026-07-24 (Codex)

UI job `job_76049ba5693da89f` completed iter 16 under `63dbc28` and clean
launch context `bf7772d`. The tighter clearance transition nearly solved
contact, but did not produce a valid policy:

- scene audit stayed exactly aligned (`0.00 m`);
- contact-free evaluation improved from 34/64 to **58/64**;
- only 1/64 completed the actual route and finish;
- rendered env 0 was contact-free but stopped after waypoint 3;
- no environment achieved the 100-frame hold or full conjunction.

The command required a 0.14 m target entry while retaining the ordinary
intermediate-route speed floor of 35% cruise (0.28 m/s). That floor was
designed for broad gates, but is too fast for a tight obstacle-clearance
subtarget: policies overshot/oscillated before the command could advance.
The frozen objective had already advanced at its unchanged 0.35 m predicate,
creating conflicting route and command phases. The automatic diagnosis also
confirmed a second, independent stop-early exploit in reward v7:
`finish_settle` paid strongly near the finish before ordered completion.

The existing UI-authored reward v15 already implements the evidence-grounded
reward correction: it gates settle income by the reward-visible route
completion proxy and tightens the finish crossfade, without exposing held-out
metric truth. The next run must therefore be a normal Resume with exact
promoted-tuple recovery **off**, so it consumes reward v15 instead of
restoring promoted v7.

Commit `0c4d000` (`fix(world): slow into clearance subtargets`) adds a
configurable intermediate approach floor. Ordinary routes preserve `0.35x`
cruise. Only routes with
clearance-adjusted subtargets use `0.10x`, allowing the distance ramp to slow
naturally into the tight transition while retaining non-zero crossing speed.
No robot, task, object, or simulator name participates. Focused
compiler/adapter tests pass **64/64**; scoped Ruff, compileall, and diff check
pass. Launch one UI Resume from iter 16 with reward v15, env v4, exact
recovery off.

That corrected run is now live as UI job `job_4ff8e2081df13d11`, iter 17,
under clean code commit `5b0c834`. It pins `selection_v26`, tuple
`0a7bc62b2fb1a1134de3a1a02b70e072d71e7a2aba9eb02d036327d6fdf868e6`,
reward v15 SHA
`3f66bdbeb08080e6c3585afb98e89295d19a21490eb914d617155222701af64c`,
and env v4 SHA
`db049dafa3fb1fa0bc5ce590c485ec62c469b3bfa4aab757255b36adcadcbb39`.
The live log proves the `0.10x` clearance approach floor, `0.140 m`
transition, `0.350 m` frozen predicate, route RSI, physical alignment,
four `-8` contact sensors, full command weights, and actor+critic warm start
from iter 16 checkpoint SHA8 `56d1d91a`. PPO iteration 0 is active. Do not
edit reload-watched core or run intermediate GPU audits until it finishes.

## Iter 17 safe-cap correction 2026-07-24 (Codex)

UI job `job_4ff8e2081df13d11` completed iter 17 cleanly with aligned physical
geometry, but it is not a success:

- physical-scene audit: aligned, maximum error `0.00 m`;
- contact-free: **61/64**, the best physical-contact result so far;
- actual route + finish: only 3/64;
- no 100-frame holds or full conjunctions;
- rendered env 0 was upright/contact-free but stopped after waypoint 3.

The first-episode trajectory makes the controller defect exact. In env 0,
the frozen raw-disk predicate advanced waypoint 3 when the local root entered
the 0.35 m authored disk. The command continued targeting waypoint 3's safe
point `(5.0, 1.118)` and converged to approximately `(4.844, 1.098)`—only
2 cm short of that steering target, safely inside the authored disk, but
outside the artificial 0.14 m target ball. At the `0.10x` velocity floor,
standing there nearly satisfies the base velocity-tracking reward, so the
command never advances to waypoint 4. This is a set/steering mismatch, not a
reward-only failure.

Commit `df8d2f5` (`fix(world): cross obstacle-safe waypoint caps`) replaces
the second point predicate with a geometric safe-cap transition:

- unadjusted waypoints still advance on the immutable authored disk;
- adjusted waypoints must be inside that same disk **and** cross the
  obstacle-clearance half-space derived from the typed shift vector;
- a 0.025 m transition slack consumes only half of the compiler's existing
  0.05 m safety margin, giving a moving policy a finite crossing surface;
- the ordinary `0.35x` intermediate speed floor is restored, because exact
  low-speed point capture is no longer required;
- command progression can no longer occur outside or redefine the frozen
  task predicate.

No robot, task, object, prompt, or simulator name controls the transition.
Focused compiler/adapter verification is **65 passed**; scoped Ruff,
compileall, and diff check pass.

The automatic iteration also produced reward v16 SHA
`8748914a188caec8e9389aeb974fdcd93816ad9e685eaf2edf55b016e1dd6616`
(sharper, bar-raising terminal phase gate plus dense waypoint-capture
kernel) and env v15 SHA
`08837c8d2f093bfe572cf60b012f997ccb3e2dffa2402592bdb8abe1a922c5f7`
(entropy scale restored to 1.0). The next UI Resume should keep exact
promoted-tuple recovery off so it consumes both evidence-authored drafts and
warm-starts iter 17.

That proof run is now live as UI job `job_99e23f1f888a44a5`, iter 18, under
clean launch commit `07ec6dc`. It pins selection v27 / tuple
`ca96cf73ebc66aecbd227e62d8c0217bf061492173ed2b82e86fd21d489f5c82`
with the exact reward v16 and env v15 hashes above. The live log proves the
safe-cap transition inside the frozen 0.350 m predicate, 0.025 m clearance
slack, four embodiment-derived safe targets, route RSI, local-frame physical
objects, direct contact supervision, entropy coefficient 0.01, and
actor+critic warm start from iter 17 checkpoint SHA8 `91fa1b84`. PPO
iteration 0 is active. Do not edit reload-watched core or run intermediate
GPU audits until it finishes.

## Iter 18 command/objective race + two-phase entry 2026-07-24 (Codex)

UI job `job_99e23f1f888a44a5` completed iter 18. The official artifact is
physically trustworthy but is not a successful demonstration:

- physical-scene audit: aligned, maximum error `0.00 m`;
- contact-free: 53/64;
- no sustained fall proxy: 51/64;
- actual route + finish: **0/64**;
- no 100-frame hold and no full conjunction;
- rendered env 0 stayed upright/contact-free and reached waypoint index 3,
  but visibly parked between boxes 3 and 4.

The full video/contact sheet and trajectory agree. Env 0 converged near local
`(4.925, 1.030)`, about 6 cm before the safe-cap plane. The frozen raw-disk
objective had already advanced to waypoint 4, while the command still targeted
waypoint 3's safe cap. Reward v16's dense active-waypoint capture then pulled
toward raw waypoint 4 while base command tracking pulled backward toward the
waypoint 3 cap. This contradictory equilibrium is a semantic race: any
command-only subset of a frozen success disk necessarily advances later than
the immutable objective.

Commit `b883447` (`fix(world): stage safe waypoint entries before disk
success`) removes the second post-success gate without weakening the frozen
predicate:

- each clearance-adjusted waypoint gets a finite-width command-only approach
  stage **outside** its authored disk, on the incoming and obstacle-safe side;
- the stage ray is constructed from typed obstacle-away clearance, authored
  tolerance, and route tangent, so entry at the original disk boundary has the
  required embodiment clearance;
- after crossing the outside stage, the command targets the immutable disk
  center and advances on the exact original disk-entry frame;
- any stochastic early disk entry also advances the command immediately,
  ensuring command and frozen objective can never remain split;
- train-only route RSI uses the same outside stages while preserving original
  logical waypoint indices;
- unadjusted routes retain ordinary authored disk entry, and no robot, task,
  object, prompt, or simulator name keys the behavior.

Focused compiler/adapter verification is **66 passed**. Scoped Ruff,
compileall, and `git diff --check` pass. The automatic diagnosis authored
reward v17, which independently removes the observed parking income by
speed-qualifying the capture kernel and finish-zone-qualifying settle income.

The next proof must be launched through **New run** in the UI with exact
promoted-tuple recovery off, so it consumes reward v17 and env v15 while
warm-starting iter 18 checkpoint SHA8 `ffe80ac9`. Use one 750-PPO cycle,
1,024 environments, seed 42, two 1,000-step 1920×1080 episodes, Auto,
`gen_003` observe-only. Require startup proof of outside-stage→frozen-disk
progression, original 0.350 m predicate, four direct `-8` contact sensors,
route RSI, physical local-frame placement, entropy 0.01, and actor+critic warm
start. Do not present iter 18 as success.

That corrected proof is now live as UI job `job_0825a00f4219404d`, iter 19.
It launched from clean code commit `cdbec1c` (controller commit `b883447`),
pins selection v28 / tuple
`b3574e18c09b2fda89467ac50ce0234e6ac744b624b6a7d068b91dd662378c07`,
reward v17 SHA
`aaaf2a20cf86532a5ba522f2e30b422133fb4927017906a07e19e0881b4a6e38`,
and env v15 SHA
`08837c8d2f093bfe572cf60b012f997ccb3e2dffa2402592bdb8abe1a922c5f7`.
The live startup log proves all four 0.268 m embodiment-clearance entries,
the outside approach stage with 0.100 m margin followed by the unchanged
0.350 m task disk, 0.025 m stage slack, 50/50 route RSI, local-frame objects,
four direct contact sensors at `-8`, full authored velocity-command weights,
whole-body terminal stillness, entropy 0.01, and actor+critic warm start from
iter 18 SHA8 `ffe80ac9`. PPO iteration 0 is active. Do not edit
reload-watched core or run intermediate GPU audits until iter 19 finishes.

## Iter 19 generated-reward conflict + clearance-stage firewall 2026-07-24 (Codex)

UI job `job_0825a00f4219404d` completed iter 19. The two-phase controller
removed the command/objective race and the official video now shows the robot
physically interacting with the aligned box course, but the policy still
stalls before completion. This is diagnostic evidence, not a showcase result:

- physical-scene audit: aligned, maximum error `0.00 m`;
- actual route + finish: 1/64;
- finish entered: 3/64;
- contact-free: 44/64;
- route + contact-free: 0/64;
- no 100-frame hold and no full conjunction;
- rendered env 0 reached only waypoint index 2, contacted for five frames,
  never entered finish, and ended at 0.1113 m/s.

The full first-episode index histogram was `[0, 0, 9, 33, 21, 1]` for maximum
indices 0 through 5. The largest groups converged immediately before the
outside approach stages for waypoints 3 and 4. That localization exposed the
remaining conflict: the base command reward correctly targets the temporary
outside stage, but generated reward v17 can only see the immutable raw
waypoint channel and simultaneously pulls toward the disk center. The policy
learned a compromise between two contradictory dense objectives. Scene
placement, rendering, and the frozen evaluator are not the cause.

Commit `3dfae11` (`fix: firewall generated reward during clearance staging`)
adds a generic per-environment training firewall. While an active command
term reports an incomplete typed clearance stage, `sculptor_primary` and its
recorded component values are scaled to zero. Full command tracking, direct
forbidden-contact penalties, survival/failure economics, terminal stillness,
and native realism priors remain active. Generated reward returns at full
scale as soon as the command enters the immutable predicate phase. The
firewall is capability/state driven and contains no robot, task, object,
prompt, or simulator-name keying; evaluator/catalog channels and task
predicates are untouched.

Verification for the code slice: focused compiler/adapter suite **67 passed**;
Ruff, compileall, and `git diff --check` passed. Iter 19's automatic reward
v18 generation failed both syntax attempts, so no reward v18 exists. It did
author env v16 with entropy scale 1.5. The next UI New run must therefore keep
exact promoted-tuple recovery off, use reward v17 plus env v16, warm-start
iter 19, and prove the startup firewall log before PPO. It remains one
750-iteration, 1,024-environment, seed-42 proof cycle with two 1,000-step
1920x1080 rollouts and `gen_003` observe-only.

That corrected proof is now live as UI job `job_86d707964503d576`, iter 20,
under clean captured code commit `b734809` (firewall implementation
`3dfae11`). It pins selection v29 / tuple
`1793447534bb34385f3fe43b7cb3ba796582583f6d8d8530f8359c4e51a12710`,
reward v17 SHA
`aaaf2a20cf86532a5ba522f2e30b422133fb4927017906a07e19e0881b4a6e38`,
and env v16 SHA
`46fa68a262828da90f551e8a81718a37fe6647df41bd12bba18513f617fda9b8`.
The UI-launched worker loaded actor and critic from iter 19 checkpoint SHA8
`182e00f5`, raised entropy coefficient from 0.01 to 0.015, emitted the exact
clearance-stage firewall proof line, and entered PPO iteration 1/750. Do not
edit reload-watched core or run intermediate GPU audits while iter 20 is
alive. After it stops, audit only the official trajectory, fitness/objective
artifacts, scene audit, keyframes, and full video against the full physical
acceptance conjunction.

## Iter 20 horizon impossibility + schedule correction 2026-07-24 (Codex)

UI job `job_86d707964503d576` completed iter 20. The reward firewall worked:
the official video visibly weaves through the co-located physical box course
instead of parking before an outside stage. The result is still diagnostic,
not showcase success:

- physical-scene audit: `aligned`, maximum error `0.00 m`;
- actual route + finish: 3/64;
- contact-free: 48/64;
- no sustained fall: 64/64;
- route + contact-free: 3/64;
- no 100-frame post-completion hold and no full conjunction;
- rendered env 0 was contact-free and upright but reached only waypoint 3.

The official first-episode trajectory makes the remaining failure
quantitative. All 64 environments reached waypoint 2, 60 reached waypoint 3,
30 reached waypoint 4, and only three reached waypoint 5. Median transition
times were 4.80 s, 9.64 s, 15.86 s, and 18.56 s; the three completions occurred
at 19.50–19.68 s. A 20 s episode requiring a two-second terminal hold cannot
accept a completion after 18 s, irrespective of policy quality. The fixed
0.8 m/s cruise schedule made the frozen objective temporally infeasible once
the real 12.740 m staged command path, turns, disk entries, and terminal
braking were included.

Commit `3039879` (`Make waypoint cruise horizon aware`) replaces that fixed
speed with a generic immutable-horizon schedule:

- command path length includes the aligned origin, every typed outside
  clearance stage, every original predicate center, and the finish;
- the schedule reserves the authored hold plus a bounded settle margin;
- a conservative 0.70 traversal-efficiency allowance accounts for ordinary
  stage capture, turning, disk entry, and braking;
- the resulting cruise is capped by the installed velocity-command domain;
- short routes retain the 0.8 m/s nominal command;
- task horizon, hold, tolerances, evaluator, reward firewall, and atomic tuple
  are untouched, with no robot/task/object/prompt/simulator-name keying.

For this exact project the compiler reports a 12.740 m command path, 16.000 s
traversal window, and 1.000 m/s cruise capped by the command domain. Focused
compiler/adapter verification is **68 passed**; scoped Ruff, compileall, and
`git diff --check` pass. The completed checkpoint is
`runs/iter_20/checkpoint.pt`, SHA8 `c1dbcce9`. Automatic reward v18 generation
failed both syntax attempts, so the next UI New run must use reward v17 plus
new env v17 (entropy scale 2.0), exact promoted-tuple recovery off, and warm
start iter 20. Keep the same one-cycle 750-PPO, 1,024-environment, seed-42,
two-episode 1920×1080 proof settings. Before PPO, require the new
`horizon-aware cruise 1.000 m/s ... 12.740 m ... 16.000 s` line in addition
to all alignment, staged-entry, RSI, contact, firewall, whole-body-stillness,
and actor+critic warm-start proofs.

That proof is now live as UI job `job_ac1eb30cafc3fdee`, iter 21, under clean
captured launch commit `6c541ff` (schedule implementation `3039879`). It pins
selection v30 / tuple
`d44cd6a28364eeb08ae086e8739ad220ed09fe0324d3d76ade04607ea4b7d978`,
reward v17 SHA
`aaaf2a20cf86532a5ba522f2e30b422133fb4927017906a07e19e0881b4a6e38`,
and env v17 SHA
`1d1655a8397c9df92291c9c62891b75c36298ffa361cf6933f28e0cbb92a4f63`.
The live log proves 1.000 m/s horizon-aware cruise for the 12.740 m staged
path and 16.000 s traversal window, 50/50 route RSI, local-pose physical
objects, four direct contact sensors at `-8`, full `track_linear_velocity`
and `track_angular_velocity` weights `2.0`, clearance-stage reward firewall,
whole-body terminal stillness at weight `4`, entropy coefficient `0.02`, and
actor+critic warm start from iter 20 SHA8 `c1dbcce9`. PPO is active on 1,024
environments. Do not edit reload-watched core or run intermediate GPU audits;
let iter 21 finish, then apply the full physical acceptance conjunction to its
official first-episode-safe artifacts and full video.

## Iter 21 route breakthrough + terminal-control correction 2026-07-24 (Codex)

UI job `job_ac1eb30cafc3fdee` completed iter 21 and preserved
`runs/iter_21/checkpoint.pt`. Horizon-aware scheduling produced a large,
physically real route improvement, but this remains diagnostic evidence:

- physical-scene audit: `aligned`, maximum error `0.00 m`;
- ordered route + finish: **40/64** (up from 3/64);
- contact-free: **30/64**; route + contact-free: **18/64**;
- generated success channel seen: 9/64, only 3/64 contact-free;
- forbidden-contact counts by box: `[6, 30, 2, 1]`;
- no 100-frame horizontal or whole-body hold; full conjunction 0/64;
- rendered env 0 entered all five predicates by 17.64 s, but touched box 2,
  achieved only a 35-frame horizontal hold / 30-frame joint-quiet lower-bound
  hold, and visibly kept correcting its stance.

The official full video and keyframes show the actual robot weaving around the
co-located orange boxes; this is no longer a detached-scene or parking failure.
Median ordered entries were 4.02, 8.54, 13.24, 16.41, and 18.60 seconds.
The remaining terminal failure is now causal: the controller commanded a
minimum `0.35 × cruise = 0.35 m/s` until the exact finish-disk entry frame and
then discontinuously switched to zero. In training, only half of resets were
mid-route and those were uniform over four suffix targets, so just 12.5% of
all resets deliberately exposed terminal approach/standing.

The next generic runtime slice fixes both sides without changing the frozen
objective:

- authored dwell routes use constant-deceleration
  `v ∝ sqrt(distance-to-predicate-boundary)` braking over the final 1.0 m;
- the command reaches the immutable boundary at at most 0.10 m/s, below the
  standing supervision's 0.12 m/s horizontal threshold, then becomes zero;
- train-only RSI remains 50% full-course, while the other half is split into
  25% interior recovery and 25% terminal-approach starts;
- non-dwell waypoint routes retain the previous controller and RSI behavior;
- official rollout trajectories now persist root body-frame linear and
  angular velocity, allowing the uninterrupted whole-body hold to be proven
  directly alongside joint velocity;
- task horizon, waypoints, tolerance, hold, metric firewall, contact sensors,
  evaluation freeze, and atomic tuple are unchanged; no robot/task/object/
  prompt/simulator-name keying is used.

Verification for this slice: focused compiler/adapter suite **68 passed**;
scoped Ruff, compileall, and `git diff --check` pass. Iter 21's automatic
reward-v18 edit again failed syntax validation, while env v18 was authored
with entropy scale 1.0. The next proof must therefore be a normal UI Resume
with exact promoted-tuple recovery **off**, reward v17 + env v18, and warm
start from iter 21. Require startup logs for the 0.10 m/s terminal-boundary
brake, 50/25/25 phase-balanced RSI, all existing physical alignment/contact/
firewall invariants, and actor+critic warm start before PPO.

That proof is now live as UI job `job_94a8897853269309`, iter 22, under clean
captured code commit `a642572`. Normal Resume kept exact promoted-tuple
recovery off and pinned selection v31 / tuple
`f9456723bfac3a1911042ad5adac1627ad35beeceecb525d043f2923c34503d8`,
reward v17 SHA
`aaaf2a20cf86532a5ba522f2e30b422133fb4927017906a07e19e0881b4a6e38`,
and env v18 SHA
`5388a2bb5d094aef865e6f0c69dc2af015400f43ed308f0d8ab1a8942d61e331`.
The worker command explicitly warm-starts iter 21 checkpoint SHA8 `675b6296`.
The live UI log proves the 1.000 m/s horizon schedule, `≤0.100 m/s` terminal
boundary brake, 50/25/25 phase-balanced RSI, local-frame objects, four direct
contact sensors at `-8`, full authored command weights, terminal stillness,
and clearance-stage firewall; PPO iteration 1/750 is active on 1,024
environments. Do not edit reload-watched core or run an intermediate GPU
audit. Preserve the checkpoint, then evaluate only official iter-22 evidence.

## Iter 22 batch success + honest evidence/posture hardening 2026-07-24 (Codex)

UI job `job_94a8897853269309` completed iter 22 and preserved its checkpoint.
The physical controller is now close, but rendered env 0 is not acceptable:

- ordered route + finish: **57/64**;
- contact-free: **52/64**; forbidden-contact counts by box `[2, 10, 1, 0]`;
- finish entry: **62/64**; no sustained falls in the frozen fitness artifact;
- five lanes achieved at least 100 uninterrupted frames below the horizontal,
  angular, and joint-velocity limits; four were also contact-free
  (`10, 11, 15, 50`);
- rendered env 0 was contact-free and entered all predicates, but ended at
  `0.443 m/s`, with `3.69 rad/s` root angular speed and deep crouched/flailing
  posture. Its full video is diagnostic, not showcase evidence.

The official trajectory also showed why velocity-only dwell credit was too
weak: terminal projected-gravity median was `-0.567`, terminal root height
collapsed to about `0.457 m`, and absolute joint-pose RMS rose to about
`1.03 rad`. A motionless collapse must not satisfy an authored upright hold.

The next generic runtime slice therefore:

- exposes reset-safe, embodiment-normalized `action_rate` and `joint_vel_rms`
  channels to every mjlab generated reward;
- makes terminal continuity require both projected-gravity uprightness and a
  bounded RMS deviation from the articulation's own default joint pose, while
  retaining fail-soft behavior for fixed-base/custom tasks lacking those
  fields;
- adds an Advanced New Run **Evidence environment** control from UI through
  backend, CLI, adapter, viewer, video, and `behavior.json`;
- requires that lane to be selected before rollout and records both its index
  and percentile, while `trajectory.npz` and all frozen metrics continue to
  cover the complete 64-lane batch. There is no automatic best-lane or
  post-hoc video selection.

This is robot/task-name independent and does not alter the evaluator, metric
firewall, task predicates, contact rules, evaluation reset, or atomic tuple.
Focused verification: core **140 passed**, backend run forwarding **48
passed**, frontend TypeScript, scoped Ruff, compileall, and diff check pass.

Reward v18 now exists and raises posture weight plus caps excessive foot swing;
env v19 lowers entropy to `0.75x`. The next proof is a normal UI **New run**
with exact promoted-tuple recovery **off**, reward v18 + env v19, warm start
from iter 22, Auto, one 750-PPO cycle, 1,024 environments, seed 42, two
1,000-step 1920x1080 episodes, `gen_003` observe-only, and precommitted
evidence environment **10**. The lane choice is based only on the preceding
frozen iter-22 batch and is disclosed in launch/evidence metadata; it does not
change training or batch scoring. Require startup proof for posture-aware
terminal stillness, full command/contact/controller invariants, and actor +
critic warm start before leaving the worker alone.

## Iter 23 precommitted showcase lane live 2026-07-24 (Codex)

The next proof was launched entirely through the ordinary **New run** UI as
job `job_0102595ce1cf9e61`, under clean code commit `6ebc857`. It is iter 23
and pins selection v32 / tuple
`014c62f4757b1e91d8689afcddd568cd85a8d699778154d22729c8b5a70397fd`,
reward v18 SHA
`48c34c86b939d332f5c543ca93756d33d21aca9e351e6d5d57c3bfb1d393498d`,
and env v19 SHA
`84c9a5ff58c12ffb91755d5afaadb60b67555da35b4b824cb174ad87520014c2`.
Exact promoted-tuple recovery was off. The worker loaded actor and critic from
`runs/iter_22/checkpoint.pt`, source SHA8 `e3c665ec`.

The UI launch record and live worker command prove one 750-PPO cycle, 1,024
environments on `cuda:0`, seed 42, two 1,000-step 1920x1080 episodes,
`gen_003` observe-only, and precommitted evidence environment **10**. Startup
also proves:

- 1.000 m/s horizon-aware cruise for the 12.740 m staged path and 16.000 s
  traversal window, plus terminal predicate-boundary braking;
- 0.268 m typed embodiment clearance, outside approach stages, and unchanged
  0.350 m frozen predicate disks;
- train-only 50% entrance / 25% collision-local interior / 25% terminal
  approach RSI;
- all four physical boxes at nominal local pose plus environment origin;
- four compiled forbidden-contact sensors at weight `-8`;
- full `track_linear_velocity` and `track_angular_velocity` weights `2.0`;
- clearance-stage reward firewall and terminal whole-body stillness weight
  `4`;
- entropy coefficient `0.0075` from env v19.

PPO is active. Do not edit reload-watched core or run an intermediate GPU
audit. Let iter 23 finish and preserve its checkpoint, then inspect the
official all-lane trajectory/fitness, precommitted lane-10 keyframes and full
video, `behavior.json` lane disclosure, and Results physical-scene audit.
Acceptance remains fully conjunctive: tight scene alignment, visible ordered
weaving around the actual boxes, every waypoint and finish entered, index 5,
zero forbidden contact, no sustained fall, upright/default-like posture,
terminal horizontal speed below `0.12 m/s`, and 100 uninterrupted
post-completion frames of horizontal, angular, joint, and posture quiet.

## Iter 23 regression + explicit UI checkpoint recovery 2026-07-24 (Codex)

UI job `job_0102595ce1cf9e61` completed and preserved iter 23, but it is
diagnostic failure, not showcase evidence. The official scene audit is still
`aligned` with maximum error `0.0 m`, the precommitted evidence lane is
honestly recorded as lane 10 (return percentile `0.453125`), and direct
contact supervision remained effective at **62/64 contact-free**. Physical
task performance nevertheless collapsed:

- waypoint-index maxima were `{0: 48, 1: 4, 2: 11, 3: 1}`;
- **0/64** reached index 5 or entered the finish;
- lane 10 stayed beside the first box, never contacted it, and ended about
  `6.34 m` from the finish;
- terminal median finish distance was `6.48 m`; no lane met the route/finish/
  hold conjunction;
- lane 10's final horizontal speed was low, but default-pose RMS error was
  about `0.786 rad`, so it was neither a completed route nor an acceptable
  upright hold.

The aligned world, staged controller, and contact penalty are not the
regression. Reward v18 doubled the ordinary posture coefficient from `0.15`
to `0.30` (and added a swing-speed cap), creating a strong stationary basin
near the first waypoint. The generated `gen_003` metric was observe-only, so
it must not silently select or roll back a checkpoint; doing so would violate
the metric firewall.

Commit `62f9a1b` adds a generic, explicit **Warm-start checkpoint** field to
the ordinary New Run UI. The user supplies only an iteration number. The
backend resolves it exclusively to this project's non-empty
`runs/iter_N/checkpoint.pt` or `.zip`, rejects missing/empty or symlink-escaped
files before GPU work, emits the source path plus full SHA-256, and forwards
it as `--init-policy`. This is policy-only recovery: it never changes the
reward, environment tuple, or objective-metric authority. The same commit
removes a duplicated reference-motion model declaration and clears existing
Ruff import-order/dead-import findings in the touched run modules.

Verification: backend run suite **51 passed**, frontend TypeScript, scoped
Ruff, compileall, and `git diff --check` pass.

The next proof must be launched entirely through the UI with exact
promoted-tuple recovery **off**, current reward v19 + env v19, explicit
**Warm-start checkpoint = 22**, Auto, one 750-PPO cycle, 1,024 environments,
seed 42, two 1,000-step 1920x1080 episodes, `gen_003` observe-only, and
precommitted evidence lane 10. Require the new
`warm_start_checkpoint_resolved` event and actor+critic load from iter 22
before leaving PPO alone. Never present iter 23 as success.

That proof is now live as UI job `job_0d2b45c89d3cf056`, iter 24, under
clean captured commit `9fda226`. The ordinary New Run UI kept exact promoted
recovery off, accepted Warm-start checkpoint 22, and pinned selection v33 /
tuple
`2262c85d4d823331ac510211688d81573cf80fd3218c0fa23ee1e5fa0cc5cfa9`,
reward v19 SHA
`ee9941bb5cb6177801dcb1dd63541988ef15bc7a4d982fad1507e327e682bc5b`,
and env v19 SHA
`84c9a5ff58c12ffb91755d5afaadb60b67555da35b4b824cb174ad87520014c2`.
The UI log emitted `warm_start_checkpoint_resolved` with full source SHA
`e3c665ecde508fa1ea9f3f12c519b285a5bc8e8116acdb83213fd1a2d8041c21`;
the worker then emitted `warm_start_loaded` with `load_cfg_keys:
["actor", "critic"]`. It also proves every requested launch setting and
physical invariant, including reward-v19 anti-parking recovery, entropy
`0.0075`, full command weights, terminal stillness, direct contact sensors,
reward firewall, aligned boxes, staged controller, and 50/25/25 RSI. PPO
iteration 0/750 is active on 1,024 environments. Do not edit reload-watched
core or run intermediate GPU audits until iter 24 finishes.

## Iter 24 route recovery + conjunctive posture correction 2026-07-24 (Codex)

UI job `job_0d2b45c89d3cf056` completed iter 24 and preserved
`runs/iter_24/checkpoint.pt`. Explicit policy-only recovery from iter 22
restored the physical task while keeping the current reward/environment
authority:

- physical-scene audit: `aligned`, maximum error `0.00 m`;
- ordered real waypoint disks plus finish: **63/64**;
- waypoint index 5: **63/64**;
- contact-free: **49/64**, with every failure isolated to box 2
  (`[0, 15, 0, 0]`);
- route plus contact-free: **49/64**;
- no gross fall: **64/64**;
- literal 100-frame horizontal hold: **18/64**;
- posture-qualified 100-frame whole-body hold: **0/64**;
- full physical conjunction: **0/64**.

The precommitted evidence lane 10 is honest diagnostic evidence. It entered
the four actual waypoint disks and finish at frames
`[159, 373, 582, 717, 795]`, had zero forbidden-contact frames, and achieved
106 uninterrupted horizontal-quiet frames. Its terminal horizontal speed was
`0.0079 m/s`, angular speed `0.196 rad/s`, and joint-velocity RMS
`0.304 rad/s`, but default-pose RMS error was `1.003 rad`. The full video and
keyframes visibly show a real alternating weave around the co-located orange
boxes followed by a deep squat with raised arms. This is not showcase success.

The causal runtime defect was generic: terminal kinematic stillness and
posture were combined additively, so a perfectly motionless collapsed pose
could retain roughly 80% of the terminal reward even though the continuity
streak correctly rejected it. The corrected runtime now multiplies the
kinematic stillness score by the geometric mean of every available generic
posture factor (projected-gravity uprightness and distance from the
articulation's own default joint pose). Honest upright stillness remains
fully rewarded; a quiet collapse has its credit withheld; missing posture
signals remain fail-soft for fixed-base/custom adapters. No robot, task,
object, simulator, or prompt name is used, and the frozen evaluator,
waypoints, tolerances, hold, contact rules, metric firewall, and atomic tuple
are unchanged.

Focused verification is **52 passed** in `tests/test_mjlab_adapter.py`;
scoped Ruff (`F,E9`), compileall, and `git diff --check` pass. Iter 24's
automatic reward v20 independently adds a dense base-height posture gate to
`finish_settle`, while env v20 uses the case-grounded `0.5x` entropy scale.
The next proof should use the ordinary UI New run with exact promoted-tuple
recovery off, reward v20 + env v20, explicit Warm-start checkpoint **24**,
the same one-cycle 750-PPO / 1,024-environment / seed-42 / two-episode
1920x1080 settings, precommitted evidence lane **10**, and `gen_003`
observe-only. Before leaving PPO alone, require actor+critic load from iter
24 and the new `multiplicative posture gate` startup line alongside every
existing physical alignment, staged-route, contact, command, RSI, firewall,
and horizon invariant.

## Iter 25 live posture-conjunction proof 2026-07-24 (Codex)

The next proof is now live, launched entirely through the ordinary UI as job
`job_65bfa68b72389283`, iter 25. The UI captured clean code commit
`8a2d451cc7ac3eeeb827be635e3199f504b7e2c9`, kept exact promoted-tuple
recovery off, selected current reward v20 and env v20, and explicitly
resolved iter 24 checkpoint SHA
`c9b0fee59f4898d889d853634535b97e96b0b8830d3d778843e7dc16c4bef238`.
Iter 25 pins selection v34 / tuple
`4d91af1045ed0afa4290b1a3c0b2284ae8e02394c9dce190bb876f9d44329a02`.

The visible UI log proves `warm_start_loaded` for both actor and critic from
iter 24 (SHA8 `c9b0fee5`), `entropy_coef 0.01 -> 0.005`, and active PPO
iteration 0/750 on 1,024 environments. It also proves the new terminal
whole-body stillness supervision with **multiplicative posture gate** at
weight 4, four forbidden-contact sensors at weight -8, full
`track_linear_velocity` and `track_angular_velocity` supervision, the
clearance-stage reward firewall, 1.000 m/s horizon-aware schedule, terminal
predicate-boundary braking, four 0.268 m typed clearance entries with
outside stages followed by the unchanged 0.350 m task disks, 50/25/25
train-only RSI, and all physical boxes at nominal local pose plus environment
origin. The UI settings are one cycle, seed 42, lane 10, two 1,000-step
1920x1080 episodes, Auto, and `gen_003` observe-only.

Do not edit reload-watched core or run an intermediate GPU audit while the
iter 25 worker is alive. After it finishes and preserves the checkpoint,
inspect the official all-lane trajectory/objective/fitness artifacts,
behavior evidence-lane disclosure, lane-10 keyframes and full video, and the
Results physical-scene audit. Success remains the full physical conjunction;
iter 24 and every older video remain diagnostic evidence only.

## Iter 25 audit + strict posture conjunction 2026-07-24 (Codex)

UI job `job_65bfa68b72389283` completed iter 25 and preserved checkpoint SHA
`d5c1f8552626c2cf4c3e5cffef11edc61ccace0c6564a0fbb433f5ea3702f51b`.
The Results physical-scene audit is `aligned` at `0.00 m`; `behavior.json`
proves requested and resolved evidence lane 10 with percentile `0.53125`.
The official first-episode trajectory shows:

- all five actual disks entered in order and index 5 reached: **60/64**;
- contact-free: **55/64**, with failures by box `[1, 5, 3, 1]`;
- route plus contact-free: **52/64**;
- no sustained fall: **64/64** (official `fell_frac=0.0`);
- 100-frame horizontal hold: **10/64**;
- 100-frame horizontal/angular/joint-velocity hold: **9/64**;
- posture-qualified whole-body hold: **0/64**;
- full physical conjunction: **0/64**.

Lane 10 is aligned, contact-free, and enters the disks at frames
`[166, 376, 562, 764, 844]`. It holds all three velocity channels quiet for
111 uninterrupted post-completion frames and terminates at `0.105 m/s`
horizontal speed, `0.178 rad/s` angular speed, and `0.329 rad/s` joint RMS.
It nevertheless finishes `0.763 rad` RMS from the articulation's own default
pose (`0.724 rad` mean over the last 100 frames), with projected-gravity z
`-0.770`. The keyframes and full 20-second video visibly show a real
alternating weave around the co-located boxes followed by a deep squat with
raised arms. Iter 25 is diagnostic, not showcase success.

The residual runtime issue is generic. Multiplying by the **geometric mean**
of uprightness and default-pose factors still takes a square root when both
signals are available, diluting one bad factor enough for a folded but quiet
body to retain a large terminal income. The corrected terminal term uses the
strict smooth **product** of all available posture factors. One failing
factor can no longer be hidden by a moderately good one; missing posture
signals remain fail-soft. No robot/task/object/prompt name, frozen evaluator,
world geometry, route, tolerance, contact rule, hold duration, or metric
channel changes.

## Iter 26 live strict-posture proof 2026-07-24 (Codex)

The strict-product recovery proof is live as UI job
`job_f61191b6d9080217`, iter 26, launched entirely through the ordinary New
run flow from clean captured commit
`bb8e085035e72110f8e18267227245b3c0b50050`. Exact promoted-tuple recovery
is off. The UI selected reward v20 and current environment v21, explicitly
resolved iter 25 checkpoint SHA
`d5c1f8552626c2cf4c3e5cffef11edc61ccace0c6564a0fbb433f5ea3702f51b`,
and pins selection v35 / tuple
`95afc97b6000593eab01c8e7b374b71dbf20f6e75e647008d322f8270b0b88c7`.

The persisted UI job log proves env v21 applied `entropy_coef 0.01 -> 0.01`,
`warm_start_loaded` restored both actor and critic from iter 25 (SHA8
`d5c1f855`), and PPO entered iteration 0/750. The visible startup and live
training log retain the strict multiplicative terminal-posture conjunction
at weight 4, four direct forbidden-contact sensors at -8, full
`track_linear_velocity` and `track_angular_velocity` supervision, the
clearance-stage reward firewall, 1.000 m/s horizon-aware scheduling,
predicate-boundary braking, four typed 0.268 m clearance entries with outside
stages before the unchanged 0.350 m task disks, 50/25/25 train-only RSI, and
all four physical boxes at nominal local pose plus environment origin.

Settings remain one cycle, 750 PPO iterations, 1,024 environments on
`cuda:0`, seed 42, precommitted evidence lane 10, two 1,000-step 1920x1080
episodes, Auto, and `gen_003` observe-only. Do not edit reload-watched core or
run an intermediate GPU audit while iter 26 is alive. After checkpoint
preservation, inspect the official all-lane artifacts and the disclosed
lane-10 keyframes/full video. Promotion still requires the complete physical
conjunction, including 100 uninterrupted post-completion frames of horizontal,
angular, joint, uprightness, and default-pose quiet inside finish.

## Iter 26 audit + clearance-preserving terminal recovery 2026-07-24 (Codex)

UI job `job_f61191b6d9080217` completed iter 26 and preserved
`runs/iter_26/checkpoint.pt` at SHA
`d5a35ae6c0a3f2ca8cc7cc6c5fce076fcb8499cae8e1351c7ed04bd864c54cea`.
The Results physical-scene audit is `aligned` at `0.00 m`; `behavior.json`
proves requested/resolved precommitted evidence lane 10 (percentile `0.625`).
The official first-episode trajectory shows:

- every actual disk entered in order: **64/64**;
- waypoint index 5 reached: **62/64**;
- contact-free: **48/64**, with failures by box `[4, 9, 4, 0]`;
- physical route + index 5 + contact-free: **46/64**;
- no sustained fall: **64/64**;
- literal 100-frame horizontal hold: **13/64**;
- horizontal/angular/joint-velocity hold: **9/64**;
- posture-qualified whole-body hold: **9/64**;
- full physical conjunction: **7/64**, lanes
  `[15, 18, 20, 36, 45, 52, 57]`.

The strict posture product therefore fixed the prior folded-pose basin:
posture-qualified holds improved from 0 to 9 and the frozen upright channel
rose from `0.34964` to `0.65013`. The UI fitness remains only `0.18120`
because it is the 64-lane frozen conjunction, not a score for the rendered
video alone: completion gate is `20/64`, contact-free fraction `48/64`, and
only seven lanes satisfy every physical requirement simultaneously.

Lane 10 visibly performs the correct alternating weave around the co-located
boxes and stops upright-ish inside finish. It enters the real disks at frames
`[141, 290, 499, 697, 768]` and reaches index 5 at frame `805`. It is still
diagnostic rather than promotable:

- box 2 contact occurs at frames `389-390`; box 3 contact occurs at frame
  `590`;
- root-to-box center distance at those contacts is only `0.396-0.427 m`,
  proving a physical corner cut rather than detached scene evidence;
- its uninterrupted posture-qualified terminal streak is frames `927-999`,
  **73 frames / 1.46 s**, with the last horizontal-speed violation at frame
  `926`;
- terminal horizontal speed is `0.0160 m/s`, angular speed `0.175 rad/s`,
  joint RMS `0.204 rad/s`, projected-gravity z `-0.834`, and default-pose RMS
  `0.522 rad`.

The residual controller defects are generic and now corrected without
changing the frozen objective. First, after the outside clearance stage the
command had discarded its already-computed obstacle-away point and aimed at
the obstacle-adjacent disk center, inviting a tracking-lag corner cut. It now
continues toward that safe point **inside the same authored disk**; the raw
disk-entry predicate remains the sole advancement authority, so no second
success condition is introduced. Second, terminal constant-deceleration
braking now spans `2.0 m` instead of `1.0 m` and caps predicate-boundary entry
command at `0.05 m/s` instead of `0.10 m/s`, giving the learned body time to
settle before the frozen two-second dwell window. Both rules derive only from
typed capability geometry, authored forbidden objects, disk predicates, and
hold semantics; there is no robot, task, object-name, or prompt branch.

Focused verification is **18 passed** in
`tests/test_world_compiler_gates.py` and **52 passed** in
`tests/test_mjlab_adapter.py`. Before the next launch, also require scoped
Ruff (`F,E9`), compileall, and `git diff --check`. The next proof must be an
ordinary UI New run with exact promoted recovery off, current reward v20 and
environment v21, explicit Warm-start checkpoint **26**, the same one-cycle
750-PPO / 1,024-environment / seed-42 / two-episode 1920x1080 settings,
precommitted evidence lane **10**, Auto, and `gen_003` observe-only. Require
actor+critic load from the iter 26 SHA plus startup lines proving
clearance-preserving in-disk targets, `2.0 m`/`0.05 m/s` terminal braking,
aligned physical boxes, direct contact supervision, full command weights,
50/25/25 RSI, and the reward firewall before leaving PPO alone.

## Iter 27 live clearance-preserving recovery 2026-07-24 (Codex)

The next proof is live as ordinary UI New-run job
`job_0b8d143e516a7920`, iter 27. The UI captured clean code commit
`7e394526a95a7f17fc8bd8caf907af1de57e9c5c`, kept exact promoted-tuple
recovery off, and explicitly resolved iter 26 checkpoint SHA
`d5a35ae6c0a3f2ca8cc7cc6c5fce076fcb8499cae8e1351c7ed04bd864c54cea`.
Iter 27 pins selection v36 / tuple
`95afc97b6000593eab01c8e7b374b71dbf20f6e75e647008d322f8270b0b88c7`,
reward v20 SHA
`b05bd4fa1ec200ff6e2330d32d0ca8f2bfb095a3878e50fb7433439e32db55e9`,
and env v21 SHA
`a56ea1a4baaa66ce5190d2cd29c41a79471ab4f497d81ce866e932082eba134c`.

The persisted startup log proves actor and critic loaded from iter 26 (SHA8
`d5a35ae6`) and PPO entered learning iteration 0/750 on 1,024 environments.
It also proves:

- four typed `0.268 m` obstacle-away in-disk clearance targets and the
  outside-stage controller under the clean commit that preserves those
  targets through disk entry;
- terminal boundary braking at at most `0.050 m/s` across a `2.000 m`
  constant-deceleration span;
- 1.000 m/s horizon-aware cruise for the 12.740 m staged path and 16 s
  traversal window;
- 50% entrance / 25% collision-local / 25% terminal-approach train-only RSI;
- all four boxes at nominal local pose plus environment origin;
- four forbidden-contact sensors at weight -8, full linear/angular command
  weights, the clearance-stage reward firewall, entropy `0.01`, and strict
  product whole-body terminal stillness at weight 4.

The UI settings are Auto, one 750-PPO cycle, 1,024 environments on `cuda:0`,
seed 42, two 1,000-step 1920x1080 episodes, precommitted evidence lane 10,
and `gen_003` observe-only with no reference-motion override. Do not run an
intermediate GPU audit or edit reload-watched core while iter 27 is alive.
After checkpoint preservation, audit only the official trajectory,
fitness/objective artifacts, behavior disclosure, keyframes/full video, and
Results physical-scene audit against the complete physical conjunction.

## Iter 27 audit + obstacle-safe disk traversal recovery 2026-07-25 (Codex)

UI job `job_0b8d143e516a7920` completed iter 27 and preserved
`runs/iter_27/checkpoint.pt` at SHA
`15d2a8434fe6b9f332760e8a71bb61e261994f41de66650ebcd2fc1f406084b6`.
The Results physical-scene audit is `aligned` at `0.00 m`, and
`behavior.json` proves requested/resolved precommitted evidence lane 10
(percentile `0.03125`). Realism is `ok`, 63/64 lanes are contact-free, the
only batch contact is box 2, and `fell_frac` is zero. The recovery nevertheless
regressed route execution:

- ordered physical route completion: **0/64**;
- waypoint index 5: **0/64**;
- maximum-index distribution: `{0: 41, 1: 9, 2: 13, 3: 1}`;
- terminal-speed aggregate: `0.20274 m/s`;
- objective fitness: `0.00000`, progress `0.24263`.

Lane 10 is contact-free and remains upright enough to avoid a fall, but the
official trajectory enters only waypoint zones 1–3 at frames
`[135, 545, 998]`. The frozen `0.350 m` command predicate advances only the
first two at frames `[339, 787]`; waypoint 3 is still `0.450 m` away on the
last frame, so waypoint 4 and finish are never approached. The full video
shows an alternating but slow, high-sway weave among the first boxes rather
than a completed task. It ends `3.535 m` from finish at index 2, horizontal
speed `0.178 m/s`, angular speed `0.434 rad/s`, joint RMS velocity
`0.466 rad/s`, projected-gravity z `-0.825`, and default-pose RMS
`0.758 rad`. There is no post-completion hold because completion never occurs.
Never present iter 27 as success.

The immutable evidence identifies a generic controller basin. After each
outside stage, the prior correction aimed only at the `0.268 m`
obstacle-away radial point near the disk boundary. Lane 10 then spent 212 and
258 frames creeping from the first two stages into their predicates; it
reached waypoint 3's stage only at frame 974. The new controller preserves
the same typed radial clearance but adds the route's outgoing tangent
component, producing a straight obstacle-safe chord from the outside stage
through the authored disk. Its steering target is `0.025 m` inside the frozen
predicate; the unchanged raw disk entry remains the sole advancement
authority and fires before the target itself must be reached. No robot, task,
object-name, or prompt branch was added.

Focused compiler + adapter verification is **70 passed**. Scoped Ruff
(`F,E9`), compileall, and diff check pass. The next proof must be launched
through the UI only after this slice is committed, warm-start iter 27, retain
the current physical-scene/contact/RSI/firewall/posture invariants, and prove
the same disclosed lane's full physical conjunction before promotion.

## Iter 28 live obstacle-safe disk traversal recovery 2026-07-25 (Codex)

UI New Run job `job_e71a1d16d5100f1b` is running iter 28 from clean captured
commit `9195e554b3d15c5ed73c26414d28f22a562eff65`. The UI emitted
`promoted_tuple_restored` before training, so selection v37 pins the exact
promoted tuple
`95afc97b6000593eab01c8e7b374b71dbf20f6e75e647008d322f8270b0b88c7`,
reward v20 SHA
`b05bd4fa1ec200ff6e2330d32d0ca8f2bfb095a3878e50fb7433439e32db55e9`,
and env v21 SHA
`a56ea1a4baaa66ce5190d2cd29c41a79471ab4f497d81ce866e932082eba134c`.
The worker explicitly loaded actor and critic from
`runs/iter_27/checkpoint.pt` SHA8 `15d2a843`, and PPO iteration 0/750 is
active.

Startup evidence proves the generic controller correction is installed:
each typed `0.268 m` obstacle-away entry uses its existing `0.100 m` outside
stage, then commands through the unchanged `0.350 m` authored disk on the
same clearance-preserving chord with a `0.025 m` inside margin. The frozen
disk predicate remains the only route advancement authority. The run also
retains `1.000 m/s` horizon-aware cruise for the `12.740 m` staged path,
`2.000 m` terminal braking to at most `0.050 m/s`, 50/25/25 train-only RSI,
local-pose-plus-environment-origin boxes, four forbidden-contact sensors at
weight -8, full linear/angular command weights 2, the clearance-stage reward
firewall, strict-product whole-body terminal stillness at weight 4, and
entropy `0.01`.

Launch settings are Auto, one 750-PPO cycle, 1,024 environments on `cuda:0`,
seed 42, two 1,000-step 1920x1080 episodes, precommitted evidence lane 10,
`gen_003` observe-only, exact promoted-tuple recovery on, and no reference
motion. Do not run intermediate GPU audits or edit reload-watched core while
iter 28 is alive. After checkpoint preservation, audit only the official
all-lane trajectory and fitness/objective artifacts, disclosed lane-10
behavior/keyframes/full video, and Results physical-scene audit against the
complete physical conjunction before any promotion.

## Iter 28 audit + robust predicate-depth recovery 2026-07-25 (Codex)

UI job `job_e71a1d16d5100f1b` completed iter 28 and preserved
`runs/iter_28/checkpoint.pt` at SHA
`8032c012dff39a68b5054f81bf35037bc1a2b691ad466bd3f12f0e126e732edc`.
The Results physical-scene audit is `aligned` with `0.00 m` maximum error,
realism is `ok`, and `behavior.json` proves requested/resolved precommitted
lane 10 (percentile `0.0625`). The official first-episode-safe trajectory and
full 20-second video reject the policy:

- ordered actual-disk route completion: **0/64**;
- waypoint index 5 and authored success: **0/64**;
- maximum-index distribution: `{0: 50, 1: 1, 2: 12, 3: 1}`;
- forbidden-contact-free: **64/64**;
- sustained-fall-free: **64/64**;
- 100-frame horizontal, whole-velocity, posture-qualified, and full
  conjunction holds: **0/64**.

Lane 10 enters only actual waypoint disks 1 and 2 at frames `[137, 309]`,
advances their frozen predicates at frames 158 and 573, and never enters
waypoint 3, waypoint 4, or finish. It is contact-free but ends at index 2,
`3.306 m` from finish. The video shows a real but partial weave among the
co-located boxes followed by parking near waypoint 3, not the requested
weave-and-stop. Never present iter 28 as success.

The immutable trace gives a precise generic controller cause. At the first
two predicates, the robot settles at `0.3514 m` and `0.3547 m` while the
frozen tolerance is `0.3500 m`. The new safe chord targeted radius
`0.325 m`—only `0.025 m` inside—so ordinary closed-loop tracking lag of
roughly `0.026–0.030 m` still leaves a stable just-outside equilibrium.
The controller now keeps the same typed radial obstacle clearance and
through-disk chord but targets radius `0.300 m`, two transition slacks
(`0.050 m`) inside the immutable predicate. Raw disk entry remains the only
advancement condition; no robot, task, object, or prompt name is used.

Focused compiler + adapter verification remains **70 passed**. Scoped Ruff
(`F,E9`), compileall, and diff check pass. The next UI proof should recover
actor+critic from iter 26—the last checkpoint with real full-conjunction
lanes—rather than continue the iter 27→28 route regression. It must retain
exact promoted reward v20/env v21 and every physical-scene/contact/RSI/
firewall/posture invariant, then prove precommitted lane 10's complete
physical conjunction before promotion.

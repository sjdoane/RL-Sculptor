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

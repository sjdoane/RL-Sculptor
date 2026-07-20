# Latest handoff — generic temporal manipulation evaluator (2026-07-19)

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

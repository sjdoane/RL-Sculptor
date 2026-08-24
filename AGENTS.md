# RewardSculptor engineering rules

These rules apply to both `RewardSculptor/` and `reward-sculptor-ui/`.
Historical context lives in `HANDOFF.md`; this file is the durable operating
contract.

## Product truth

- Describe only capabilities the launch path actually consumes. A parsed,
  hashed, or staged asset is not an active controller or world.
- Keep policy initialization, reference motion, and training environment
  independently selectable. Show the effective choice and its provenance
  before a run starts.
- Prefer plain researcher language in the primary flow. Put hashes, contracts,
  and simulator details in an expandable receipt, never behind an ambiguous
  label or hidden fallback.
- A readiness warning and the primary action must use the same authority. If a
  card says a run is blocked, the launch action must be blocked for that reason.

## Scientific contracts

- Content-address immutable artifacts with SHA-256 and pin the exact bytes used
  by a run. A compound import identity includes its admitted manifest and
  component set; never infer identity from a filename, alias, or clip ID alone.
  Robot-scoped assets are addressed by `(robot, artifact_id)` at every API and
  UI boundary; an ID is never globally unique by implication.
- Compatibility is exact and structural: ordered joints, observation/action
  names and shapes, policy architecture, recurrent state, normalizers,
  controller timing, adapter/task/robot, and relevant software versions.
  Missing required evidence fails closed.
- OGMP support currently means a validated linear phase-window automaton.
  Training and evaluation must consume the same immutable execution manifest.
  Do not label predicate, branching, preference-conditioned, or learned-oracle
  behavior as implemented until it is wired and tested end to end.
- A reference-conditioned run must expose the exact immutable playback clock
  to both actor and critic and must prove that the worker installed and used
  that observation. A phase-indexed reward by itself is not policy
  conditioning. Freeze the complete non-authorable mode runtime (clock,
  windows, masks, dispatch, tracking targets, and compute entrypoints), not
  only the numerical tracking kernels.
- Preserve root-frame semantics through crops and compositions. A composite
  may inherit a convention only when every exact parent agrees and its ordered
  parent/evidence chain re-verifies; missing, mixed, or stale evidence fails
  closed.
- Keep objective metrics independent from generated rewards. Never relax a
  trust, calibration, physical-scene, contact, or naturalness gate to make a
  run pass.
- Record requested configuration separately from observed runtime facts.
  `INITIALIZED_FROM` is earned by a successful load event, not by resolving a
  path. `EXECUTES_IN` names the validated world tuple actually used.
- Treat Tier K as a kinematic candidate, never a dynamics-feasibility claim.
  A research training launch may rely on a reference only after the Tier-D
  clip/rollout hash chain re-verifies; a short explicit pipeline check may
  inspect Tier K while remaining clearly labeled non-certified.
- A Tier-D certificate covers exact motion bytes, embodiment, simulator
  evidence, and cadence. Do not retime a certified clip in place. Materialize
  the transformed trajectory under a new identity and certify it again, or
  preserve cadence and declare explicit pre/post/terminal holds.

## Upload and execution safety

- Treat every upload as hostile. Accept data-only, bounded archives; reject
  path traversal, links, duplicate members, decompression bombs, executable
  code, pickle, TorchScript, and raw `.pt` files.
- Safetensors may be inspected only after member and size validation. Validate
  tensor keys, dtypes, ranks, and shapes against the declared policy contract,
  then write a server-owned artifact. Never execute an uploaded controller.
- Do not call discarded bytes "staged" or imply that a digest can be activated.
  An artifact is staged only when its immutable bytes are retained and a
  separate review/promotion path exists.
- Portable imports are transfer artifacts, not exact optimizer resumes, unless
  a future trusted format proves and restores the complete training state.
- Tier-D dry-run/preflight is a CPU/data-only operation: it must not import the
  simulator stack, probe CUDA, load an adapter, or launch a subprocess. Live
  certification uses only the trusted local adapter and a fresh atomically
  claimed work directory outside the donor and retained reference library.

## Implementation discipline

- Put contracts and validation in small pure core modules; keep HTTP routes and
  React components as adapters. Avoid parallel sources of truth.
- Add a regression test for every corrected failure mode, including the
  negative path. Prefer fixture bundles and immutable manifests over mocks that
  bypass validation.
- In strict lineage, a contradictory authoritative worker event is fatal.
  Bind the requested iteration plan before launch and prove every completed
  iteration (or a specifically authorized early-stop reason); a successful
  process exit is not completion evidence.
- Preserve user work and unrelated dirty-tree changes. Use `apply_patch` for
  edits. Do not modify reload-watched training code while a live worker depends
  on it.
- Run focused tests first, then the relevant core/backend/frontend suites,
  Ruff/compile checks, TypeScript typecheck, production build, and browser QA.
- Commit each complete, reviewable slice. Update `HANDOFF.md` and the relevant
  runbook when behavior or an honest capability boundary changes.

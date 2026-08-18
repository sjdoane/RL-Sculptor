# Starting-point research workflow

This document defines the researcher-facing contract for adapting an existing
behavior in RewardSculptor. It is intentionally stricter than “upload a
checkpoint and hope”: policy weights, reference motion, and training world are
independent artifacts, and the launch receipt states which of them will
actually influence the run.

## The product model

A starting point has three independent axes:

1. **Starting policy** — new weights, a checkpoint from this project, or a
   validated portable skill.
2. **Starting motion** — no motion, or an immutable reference trajectory used
   as the tracking prior.
3. **Training environment** — the project's validated active world. A bundle
   may attest to a source-world digest, but upload never changes the active
   world.

This separation is load-bearing. A reference is not proof that policy weights
were loaded. A world packaged beside a policy is not proof that the run used
that world. A controller description is not executable code. Runtime lineage
is recorded only after the corresponding evidence boundary is observed.

## Persona review

### First-time user

The first decision is phrased in task language:

- **From scratch** for a clean baseline;
- **Project checkpoint** to continue a behavior already learned here;
- **Imported skill** to validate a portable research artifact.

The default plan is a brief pipeline check. Research-sized budgets are an
explicit choice. The primary launch action and the readiness cards share the
same blockers, so a screen cannot say “blocked” while still offering a launch
that will predictably fail.

### Researcher

The receipt answers the reproducibility questions before launch:

- exact policy and motion SHA-256 values;
- adapter, task, robot, ordered joint/action/observation contracts;
- actor/critic architecture and normalizer expectations;
- admitted initialization modes;
- whether a reference has a verified dynamics-tracking certificate;
- the effective world selection;
- anything that was excluded, recorded only as a digest, or left inactive.

The common transfer default is **actor only**. **Actor + critic** is exposed as
an advanced choice because a value function trained under a different reward
can bias early optimization. A portable upload is never described as a full
resume: optimizer state, counters, exploration state, and arbitrary serialized
Python objects are not admitted from an untrusted bundle.

### Project owner

The primary strategic rule is scientific honesty. The UI may be concise, but
the stored receipt and knowledge graph must distinguish requested configuration
from observed execution. A useful demo is not allowed to erase that boundary.

## Lokesh feedback as executable gates

The lab feedback in `docs/RESEARCH_DIRECTION.md` is represented by tests and
launch rules rather than copy alone:

| Feedback | Executable contract |
|---|---|
| Start from an explicit motion prior | A reference is content-addressed and passed as a separate launch input. A no-reference run remains a clean baseline. |
| Use a pretrained controller as a feasibility filter | **Partially implemented.** Kinematic-only (Tier K) motions are candidates, not training priors. Tier D is target-specific simulated tracking evidence produced by a freshly trained throwaway tracker; it is not yet Lokesh's proposed frozen pretrained whole-body mimicker. Research launch still requires the exact Tier-D certificate and boundary hashes, and Tier K cannot train, roll out, or publish a checkpoint. |
| Treat a task as behavior-set × variation specification | **Partially implemented.** One exact reference stays independent from the active world, RSI, physics randomization, and objective metric. A bounded multi-behavior manifest, immutable variation sampler, and per-episode behavior/variation receipt are not implemented, so the UI must not claim this full formulation. |
| Make modes first-class | Composed references produce a canonical mode graph and an exact execution manifest. Reward and evaluation must bind to the same graph, clip bytes, windows, robot, and project context. |
| Keep objective evidence independent | Generated reward never grants its own success. The objective metric, physical-scene audit, contact/fall checks, and terminal behavior remain separate evidence. The production mode panel can persist an exact-context readiness receipt, but it is explicitly `observe_only`; the generated/validated/calibrated per-mode objective gauntlet is not yet attached to training or a production fitness gate. |
| Make controllers environment-aware | **Safety boundary, not yet the requested controller.** Bundled controller/world declarations are validated and digest-recorded but discarded, never executed. The active project world and server-owned adapter remain runtime authority; mode identity is not yet a policy observation and no uploaded hierarchical controller can be promoted. |

Regression coverage must include the negative cases: stale bytes under the
same filename, same clip ID with changed content or provenance, actor loading
from a motion-only receipt, a Tier-K reference submitted directly to the live
API, path traversal, duplicate archive members, decompression bombs, pickle or
TorchScript payloads, and mismatched observation/action contracts.

## Portable `.rskill` bundle

The portable format is a bounded ZIP-compatible data archive with a required
`manifest.json`. Every member has an OCI-style descriptor containing its path,
byte count, and SHA-256 digest. The descriptor set must exactly match the
archive member set.

Typical members are:

```text
manifest.json
policy/weights.safetensors        # optional; canonical trainable policy data
motion/clip.npz                   # optional; bounded trajectory arrays
motion/provenance.json            # required with motion/clip.npz
controller/controller.json        # optional declaration; never uploaded code
world/manifest.json               # optional bounded source-world declaration
```

`policy/weights.safetensors` uses RewardSculptor's declared rsl_rl tensor
namespace. The importer validates the actual tensor inventory, dtype, rank,
shape, actor/critic layer topology, normalizers, and exact project interface
contract. It then reconstructs a server-owned checkpoint. Raw `.pt`/`.pth`,
pickle, TorchScript, ONNX, Python, shared libraries, links, nested archives,
special files, and every other non-allowlisted member are rejected before
tensor parsing or library mutation.

World and controller members are evidence about the source setup. Unless a
future reviewed promotion path explicitly says otherwise, their digests may be
recorded but they do not become the project's active world or executable
controller.

A future activation path must stay data-only: normalize into a supported
server-owned controller/world schema, preview the compiled diff, validate the
exact robot/timing/sensor/world tuple, and require an explicit Promote action.
The current importer intentionally stops before that boundary.

Generate this format with
`sculpt export --portable --robot <slug> --config <project>/config.toml`.
To share an existing registered motion without policy weights, use
`sculpt refs export-skill --robot <slug> --clip <clip-id> --out motion.rskill`.
That export remains a kinematic candidate at the receiving installation until
the target-specific Tier-D tracking chain is verified; exporting never carries
training authorization with it.
The command without `--portable` deliberately produces a different deployment
ZIP containing raw checkpoints, executable inference material, reward source,
and environment snapshots. Renaming that ZIP to `.rskill` does not make it
portable: the importer rejects its distinct artifact kind and members.

## Initialization modes

| Mode | Loads | Does not imply |
|---|---|---|
| `actor_only` | compatible actor weights | compatible critic, optimizer, or source world |
| `actor_critic` | compatible actor and critic weights | optimizer/full-state resume or matched downstream reward |
| `reference_only` | no policy weights; attaches the attested motion | dynamics feasibility unless the Tier-D certificate verifies |
| `full_resume` | reserved for a future exact trusted training-state format | availability for sanitized portable imports |

The launch pins the selected manifest digest. Re-resolving the same skill alias
to different bytes must fail before a worker starts. At runtime, an
`INITIALIZED_FROM` lineage edge is earned only by the runner's observed
`warm_start_loaded` event and its exact digest/load-role evidence.

## G1 research example: evolve a parkour prior

The recommended demonstration starts from something already difficult but
well-scoped: a G1 rough-terrain or parkour tracking policy plus its certified
reference. The source behavior might run over fixed stepping stones. The new
project keeps the G1 policy/motion but authors a different validated world and
objective—for example:

> Traverse an irregular course, clear a laterally offset gap, land without
> forbidden contact, then carry forward momentum into a precise terminal stop.

Use the workflow in increasing-risk order:

1. Import the portable skill and inspect the receipt.
2. Run the **pipeline check** to prove contracts, reference lookup, reward
   binding, and artifact lineage.
3. Run a short **rehearsal** with actor-only transfer and the certified motion.
4. Compare against a from-scratch/no-motion baseline under the same world,
   seed family, and objective metric.
5. Author residual task terms per phase while preserving the tracking
   backbone and objective-metric firewall.
6. Inspect digest-bound per-mode diagnostics, then promote only when the
   independent objective and physical trust gates agree; do not choose a
   visually attractive single lane after the fact. Per-mode diagnostics are
   explanatory evidence, not a calibrated fitness grant.

This is an evolution experiment, not a claim that an arbitrary uploaded flip
or cartwheel is hardware-ready. The source reference must match G1 joint and
timing contracts, and dynamics certification is still only simulator evidence.

## Current OGMP boundary

The implemented mode system is **OGMP-inspired**, not paper-faithful OGMP. It
currently provides:

- a validated linear mode graph derived from one composed reference;
- fixed phase windows driven by per-environment elapsed time at the
  reference's certified cadence, with any remaining episode budget recorded
  as an explicit terminal hold rather than an in-place retime;
- per-mode generated reward scope;
- an exact execution manifest and context binding;
- a content-addressed production readiness receipt that binds reward, clip,
  robot, graph, manifest, and world selection while representing the current
  absence of calibrated per-mode objective evidence honestly.

It does not currently provide the paper's closed-loop queried oracle,
receding-horizon oracle updates, rho-bounded state exploration,
preference-conditioned oracle selection, runtime predicate/branch transitions,
or a policy conditioned on a learned mode latent. The UI and API must keep
those absences visible. The next research milestone is not another label; it is
an end-to-end runtime implementation with exact observations, transitions,
training semantics, and objective evaluation.

The schema exposes `reward_terms` and `success_predicate` extension fields,
but the current composition/authoring path does not populate them and they are
not readiness evidence. Mode-specific reward bodies live in the bound reward
artifact; runtime handover remains fixed elapsed-time dispatch.

## Knowledge-graph claims

The lineage vocabulary is deliberately narrow:

- an import attestation **ATTESTS** immutable policy/motion bytes;
- an attestation **DECLARES_TARGET** from manifest metadata;
- a policy is **COMPATIBLE_WITH** an embodiment only after exact tensor and
  interface validation;
- a run **TRACKS** the effective reference and **EXECUTES_IN** only the
  validated active world/software environment;
- a run **USES_MODE_EXECUTION** only the independently re-derived immutable
  mode execution artifact admitted immediately before a real train call;
- a run **INITIALIZED_FROM** only an observed successful load;
- a run **PRODUCED** its written checkpoint, which may be **DERIVED_FROM** the
  observed input policy.

Co-location in a bundle never creates `TRACKS`, `EXECUTES_IN`, or
`INITIALIZED_FROM`. Replaying the same evidence is idempotent; contradictory
facts under the same content identity are rejected.

## Design references

- [Oracle Guided Multi-mode Policies (OGMP)](https://arxiv.org/abs/2403.04205)
  defines the queried-oracle, multi-mode exploration architecture this
  project's narrower phase-window implementation is compared against.
- [Preferenced OGMP](https://arxiv.org/abs/2410.01030) extends that line with
  preference-conditioned behavior; RewardSculptor does not currently claim
  this capability.
- [PyTorch's security policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
  treats untrusted models as untrusted programs, motivating the no-pickle,
  no-uploaded-code boundary.
- [safetensors](https://github.com/huggingface/safetensors/blob/main/README.md)
  provides the bounded data-only tensor container used at the upload boundary.
- [OCI descriptors](https://github.com/opencontainers/image-spec/blob/main/descriptor.md)
  provide the digest-plus-size descriptor pattern used for archive members.
- [rsl_rl's on-policy runner](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/runners/on_policy_runner.py)
  is the upstream load/runtime shape against which portable policy contracts
  are validated.

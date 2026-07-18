# Prompt-driven environment authoring and interaction

Status: implementation architecture, written 2026-07-18. The knowledge-graph
audit and research campaign that support it are recorded in
`RL_SCULPTOR_AUDIT.md` and
`RewardSculptor/kg_seeds_env_authoring_2026-07.yml`. The companion proof of
concept is `RewardSculptor/scripts/world_schema_poc.py`.

This design extends RewardSculptor from prompt-driven reward and training-knob
adaptation to prompt-driven world and task authoring. It is intentionally an
architecture that Sam can implement incrementally, not a claim that the full
system already exists.

## 1. Required outcomes and invariants

A prompt must produce a complete environment tuple:

- world geometry, terrain, obstacles, rigid objects, regions, and contacts;
- initial-state and goal distributions;
- observations, commands, terminations, and success predicates;
- reward-generation context and objective metric observables;
- a frozen evaluation world and a train-only variation/curriculum surface.

The three acceptance examples are:

1. “Stay stable and walk or jump on uneven terrain.”
2. “Learn parkour over a course of boxes.”
3. “Get the humanoid with a gripper to score a ball into a goal.”

Five invariants govern every component:

1. The language model emits a typed declarative specification, never arbitrary
   MuJoCo or mjlab code.
2. The artifact is parametric, composable, versioned, and human-diffable like a
   CAD feature tree.
3. Robot-specific names and dimensions come from capability data, not
   task-name conditionals in the compiler.
4. Every load-bearing ambiguity is presented to the user with a
   “system decides” option that discloses the default and its reason.
5. The metric firewall, evaluation freeze, calibration, and keep/revert
   semantics are never weakened.

## 2. Research foundation

The 89-paper campaign spans six domains: language-model environment
generation, procedural terrain, parkour, robot-object interaction,
unsupervised environment design and curricula, and simulator scene
construction. The structured seed file records a tier, domain tags, and an
applicability rationale for each source.

The architecture adopts these load-bearing findings:

- Holodeck and ScenicNL separate semantic constraints from deterministic
  coordinate solving. WorldSpec therefore stores relations, parameters, and
  bounds; the compiler owns coordinates.
- Eurekaverse, GenSim, RoboGen, Text2World, and SceneSmith support generated
  environments followed by executable validation and repair. RewardSculptor
  uses one bounded repair round after a complete violation report.
- Rudin et al., Orbit, mjlab, and the terrain-curriculum literature support
  typed subterrain generators driven by a normalized difficulty variable.
- Miki et al., Agarwal et al., and perceptive humanoid locomotion work show
  that authored terrain changes the observation contract. Height sensing is
  compiled with the terrain rather than left to the reward author.
- MuJoCo Playground and the installed mjlab/MuJoCo-Warp stack require compiled
  topology to remain fixed. Structural, model-field, and state variation are
  therefore distinct schema classes.
- ACCEL, Prioritized Level Replay, SFL, Robust PLR, and related work support
  focusing training on intermediate-success configurations and making small
  edits to high-learning-value worlds.
- Fetch/HER, DribbleBot, humanoid soccer, HumanoidBench, PhysHOI, RoboGen, and
  RoboCasa support a compact vocabulary of goal predicates, placement
  distributions, and desired/forbidden contact relations.
- EnvGen, SAMPLR, and held-out evaluations in generated-environment work
  require train/evaluation separation. RewardSculptor persists the resolved
  evaluation scene rather than relying on a nominal difficulty alone.

The corpus is deliberately broader than the initial implementation plan. S
tier is the hybrid extraction target; A and B tiers remain metadata-searchable
and can be extracted later without changing the architecture.

## 3. Existing seams

The design attaches to existing, verified surfaces:

- `sculptor/env_spec.py` already versions validated JSON, separates shared
  from train-only knobs, applies bounded edits, and keeps/reverts an
  environment version with the reward version.
- `sculptor/adapters/_mjlab_runner.py` loads an mjlab configuration, mutates
  a deep copy for train or rollout, injects generated reward computation, and
  records `trajectory.npz`.
- mjlab 1.3.0 provides `SceneCfg`, `TerrainEntityCfg`,
  `TerrainGeneratorCfg`, primitive and heightfield subterrain
  configurations, `EntityCfg`, contact sensors, ray/height sensors, reset
  events, commands, and terrain-origin curricula.
- Generated metrics currently use a static array allowlist and the
  `completion gate × minimum channels` contract. Validation, calibration,
  and the partition gate are already separate enforcement steps.
- The diagnoser already proposes bounded train-only environment edits and
  records their measured outcomes in shared run-case memory.

The current mjlab adapter still contains task-ID-based schema selection. The
capability layer in section 5 is therefore a prerequisite for claiming
robot/task generality.

## 4. Components and data flow

```
prompt
  │
  ├── capability resolver ── robot/simulator descriptors
  │
  ├── KG retrieval ───────── papers, techniques, prior run cases
  │
  ▼
author ── draft WorldSpec + TaskSpec + underspecification report
  │
  ▼
clarifier ── all load-bearing questions, each with system default
  │
  ▼
schema validator ── bounds, references, variation classes, capabilities
  │
  ▼
deterministic compiler
  ├── train scene + train variation plan
  ├── resolved evaluation manifest
  ├── channel catalog
  └── reward/metric context
  │
  ▼
gate subprocess ── build, budget, settle, placement, reachability
  │
  ▼
existing train → rollout → metric → diagnose → edit loop
```

Versioned project artifacts:

- `env/world_v<N>.json`: authored geometry and train variation surface.
- `env/task_v<N>.json`: authored task semantics.
- `env/resolved_eval_v<N>.json`: fully resolved evaluation seed, sampled
  geometry/placements, dependency versions, and hashes.
- `env/channel_catalog_v<N>.json`: the compiled tensor contract.
- `env/clarifications_v<N>.json`: questions, defaults, answers, and
  provenance.
- `env/v<N>.json`: existing reset, termination, randomization, and optimizer
  knobs; retained as a separate artifact.

The corresponding `*_current.json` files participate in one atomic selection
unit with the reward. A kept iteration points all current files to a coherent
version tuple. Revert restores the tuple, never one file in isolation.

## 5. Capability descriptors: the generality layer

No compiler code may infer a robot from a task-ID substring. Two registries
provide the data needed to compile semantic roles.

### 5.1 RobotCapability

```json
{
  "capability_id": "unitree_g1:base",
  "asset_hash": "...",
  "root_body": "pelvis",
  "body_roles": {
    "left_foot": ["left_ankle_roll_link"],
    "right_foot": ["right_ankle_roll_link"],
    "torso": ["torso_link"],
    "left_end_effector": ["left_hand_link"]
  },
  "variants": {
    "dual_gripper": {
      "asset_id": "unitree_g1:dual_gripper",
      "capabilities": ["grasp", "push", "kick"]
    }
  },
  "geometry": {
    "standing_height_m": 1.28,
    "leg_length_m": 0.70,
    "foot_length_m": 0.27,
    "reach_radius_m": 0.75
  },
  "supported_commands": ["base_velocity", "robot_to_region", "waypoint_heading"],
  "supported_observations": ["proprioception", "height_scan", "object_relative"],
  "contact_capacity": {"max_pairs": 24}
}
```

Body roles resolve to concrete model names at validation. Variant constraints
such as “with a gripper” are resolved by the registry before world compilation.
If no installed variant satisfies a required capability, authoring stops with
a precise capability error; it does not silently generate an impossible task.

### 5.2 SimulatorCapability

The mjlab descriptor names:

- available terrain configuration classes and their fields/bounds;
- supported entity primitive builders;
- observation-group and command adapters;
- sensor types and contact/ray budgets;
- which MuJoCo fields may vary per environment;
- compiler and dependency versions.

An adapter implements semantic operations such as
`add_height_observation(...)` or `bind_waypoint_command(...)`. The core
compiler never assumes group names such as `actor`, `critic`, or a command
name such as `twist`.

## 6. Normative WorldSpec

`world_spec_version: 2` has exactly three top-level sections:
`meta`, `shared`, and `train`. Unknown keys are rejected at every level.

```json
{
  "world_spec_version": 2,
  "meta": {
    "version": "v3",
    "parent": "v2",
    "source": "generated",
    "prompt": "walk on uneven terrain",
    "grounding": ["paper:2411.01775", "paper:2109.11978"],
    "parameter_provenance": {
      "rough_noise": "default",
      "slope_angle": "user"
    }
  },
  "shared": {
    "eval_seed": 1729,
    "robot": {
      "capability_id": "unitree_g1:base",
      "required_capabilities": []
    },
    "terrain": {
      "kind": "generator",
      "layout": {
        "mode": "curriculum_grid",
        "rows": 8,
        "tile_size_m": [10.0, 10.0],
        "border_width_m": 5.0
      },
      "evaluation_difficulty": 0.5,
      "sub_terrains": {
        "rough": {
          "type": "hf_random_uniform",
          "nominal": {"noise_range_m": [0.02, 0.06], "noise_step_m": 0.02}
        },
        "slope": {
          "type": "hf_pyramid_sloped",
          "nominal": {"slope_deg": 12.0, "inverted": false}
        }
      }
    },
    "obstacles": {"course": []},
    "objects": {},
    "zones": {}
  },
  "train": {
    "variations": [
      {
        "id": "rough_noise",
        "target": "/shared/terrain/sub_terrains/rough/nominal/noise_range_m/1",
        "class": "generator_parameter",
        "distribution": {"kind": "uniform", "low": 0.04, "high": 0.12},
        "curriculum": {"axis": "difficulty", "mapping": "linear"}
      }
    ],
    "curriculum": {
      "difficulty_range": [0.0, 1.0],
      "promotion": {
        "signal": "traversal_fraction",
        "promote_above": 0.75,
        "demote_below": 0.50
      }
    }
  }
}
```

### 6.1 Evaluation freeze

`shared` is the canonical evaluation design. Every physical value there is a
single nominal value, including object mass, friction, restitution, geometry,
and placement. Ranges live only in `train.variations`.

The compiler uses `shared.eval_seed` to resolve every stochastic generator
and persists the result in `resolved_eval_v<N>.json`. The manifest contains:

- hashes of WorldSpec, TaskSpec, compiler, robot asset, and capability files;
- mjlab, MuJoCo, and MuJoCo-Warp versions;
- terrain generator seeds and the resolved tile/primitive parameters;
- nominal object/zone poses and physics;
- the compiled model hash and channel-catalog hash.

Rollout loads this manifest. It does not regenerate terrain from
`evaluation_difficulty`. A shared-field or dependency-hash change creates a new
evaluation lineage and resets the fitness baseline.

### 6.2 Terrain semantics

The type vocabulary maps to installed mjlab configuration classes:
heightfield random uniform, pyramid slope, wave, discrete obstacles and Perlin
noise; box stairs, random grids/spreads, stepping stones, narrow beams, tilted
grids, nested rings, and flat tiles.

Two layout modes avoid an important mjlab ambiguity:

- `curriculum_grid`: `rows` is difficulty; columns are exactly one per
  declared subterrain type. The schema has no user-set `cols`, because mjlab
  ignores it in curriculum mode.
- `sampled_grid`: explicit rows/columns and subterrain proportions are used
  for a non-curriculum sampled atlas.

Heightfield texture count, resolution, and border size are budgeted before
build. Flat-patch sampling is generated automatically when an object or goal
requires a valid placement on uneven terrain.

### 6.3 Obstacles and course grammar

Course topology is structural and lives in `shared`:

```json
{
  "obstacles": {
    "course": [
      {
        "id": "platform_01",
        "element": "platform",
        "nominal": {"height_m": 0.35, "length_m": 1.2, "gap_after_m": 0.45}
      },
      {
        "id": "beam_01",
        "element": "beam",
        "nominal": {"width_m": 0.24, "length_m": 2.0}
      }
    ],
    "layout": "linear",
    "waypoints": "auto"
  }
}
```

The closed element vocabulary is `platform`, `gap`, `beam`, `wall`,
`stairs`, and `stepping_stones`. Train ranges target stable element IDs,
not array indices. For example:

```json
{
  "id": "platform_height",
  "target": "/shared/obstacles/course/@platform_01/nominal/height_m",
  "class": "model_field",
  "distribution": {"kind": "uniform", "low": 0.20, "high": 0.55}
}
```

The validator resolves the extended pointer syntax `@<stable-id>` and
checks the target against a schema-generated editable registry. This replaces
an impossible static Pydantic `Literal` over arbitrary course indices.

### 6.4 Objects and regions

Objects are named free or fixed rigid entities. Regions are non-colliding
geometric predicates and visualization sites.

```json
{
  "objects": {
    "ball": {
      "shape": "sphere",
      "nominal": {
        "radius_m": 0.11,
        "mass_kg": 0.45,
        "friction": 0.8,
        "restitution": 0.65,
        "pose": {"placement": "zone:center", "z_m": 0.11}
      }
    },
    "goal_frame": {
      "shape": "frame",
      "fixed": true,
      "nominal": {"opening_m": [1.8, 1.0], "post_radius_m": 0.05}
    }
  },
  "zones": {
    "center": {"kind": "disk", "center_m": [0.0, 0.0], "radius_m": 1.5},
    "goal_mouth": {
      "kind": "box",
      "center_m": [4.0, 0.0, 0.5],
      "size_m": [0.2, 1.8, 1.0],
      "attach": "goal_frame"
    }
  }
}
```

Train-only placement and physics distributions are separate variation entries.
Structural values such as shape, entity count, site existence, and sensor
existence cannot vary per environment.

### 6.5 Variation classes

The validator table assigns each editable field one class:

- `structural`: topology, entity/sensor existence, course element count,
  heightfield resolution. Fixed for a compiled run.
- `generator_parameter`: roughness, slope, stair height, and other parameters
  used to construct a fixed terrain/course atlas. A range creates multiple
  precompiled train tiles; it is not mutated per environment after compilation.
- `model_field`: sizes, masses, friction, restitution, and other fields the
  simulator descriptor confirms are expandable.
- `state`: poses and velocities sampled at reset.

A train variation must target an allowed non-structural field. The compiler
bakes `generator_parameter` ranges into the fixed train atlas and never treats
them as runtime model mutation. The validator rejects structural train
variation, unsupported runtime mutation, shared/train target duplication, and
distribution ranges outside robot and simulator limits.

## 7. Normative TaskSpec

TaskSpec uses the same `meta/shared/train` split.

```json
{
  "task_spec_version": 1,
  "meta": {"version": "v2", "parent": "v1"},
  "shared": {
    "control_mode": "goal_directed",
    "goal": {
      "id": "score",
      "type": "object_to_region",
      "subject": "ball",
      "region": "goal_mouth",
      "success": {
        "predicate": "inside",
        "hold_s": 0.10,
        "tolerance_m": 0.0
      }
    },
    "contacts": {
      "desired": [["robot:left_foot|right_foot", "object:ball"]],
      "forbidden": [["robot:torso", "object:ball"]],
      "terminate_on": [["robot:torso", "world:terrain"]]
    },
    "termination": {
      "fall": "capability_default",
      "out_of_bounds_m": 12.0,
      "success_ends_episode": false
    },
    "observations": {
      "proprioception": true,
      "height_scan": "auto",
      "object_relative": ["ball"],
      "region_relative": ["goal_mouth"]
    }
  },
  "train": {
    "goal_sampling": [
      {
        "id": "ball_start",
        "target": "object:ball.pose",
        "distribution": {"kind": "uniform_in_region", "region": "center"}
      }
    ]
  }
}
```

The goal vocabulary is:

- `object_to_region`;
- `object_velocity`;
- `robot_to_region`;
- `waypoint_sequence`;
- `configuration_distribution`.

Each goal compiles to reset distributions, required observations, a sparse
success predicate, and metric channels. It does not compile directly to a
dense reward. Dense shaping remains generated and reviewed by the existing
reward pipeline.

Contact selectors use semantic robot roles from RobotCapability and named
world entities. Every selector must resolve to at least one concrete
geom/body pair. Desired, forbidden, and terminating contacts compile to
sensor channels and task events without task-specific branches.

## 8. Compiler and admission gates

New modules:

- `sculptor/world/capabilities.py`;
- `sculptor/world/world_spec.py`;
- `sculptor/world/task_spec.py`;
- `sculptor/world/compiler.py`;
- `sculptor/world/channels.py`;
- `sculptor/world/gates.py`;
- `sculptor/world/author.py`.

The mjlab runner calls `apply_world_spec` before applying env-spec v1. Train
compilation overlays `train.variations`; rollout uses only the persisted
evaluation manifest.

The gate chain runs in an isolated subprocess and returns all violations:

1. Schema gate: types, bounds, references, stable IDs, shared/train rules.
2. Capability gate: body roles, robot variant, commands, observations,
   sensors, and mutable fields exist.
3. Budget gate: geoms, contacts, rays, constraints, and heightfield texels.
4. Build gate: attach entities and compile a real `MjSpec`/model on CPU.
5. Initial-penetration gate: reject intersections above tolerance.
6. Settle gate: finite state, no fall-through, and task-specific stationary
   objects within declared tolerances. Global quiescence is not required; a
   ball on rough terrain may legitimately roll.
7. Placement gate: enough flat patches, no object/obstacle overlap, regions
   inside world bounds.
8. Reachability gate: kinematic envelope checks for gap, climb, reach, and
   goal placement. This is a cheap necessary-condition check, not a claim that
   an untrained policy can solve the task.

One complete violation list is returned to the Author for one repair attempt.
A second failure preserves the rejected draft and stops before GPU training.

## 9. ChannelCatalog and the metric firewall

The existing generated-metric runtime has a static `ALLOWED_ARRAYS`; merely
writing new tensors into `trajectory.npz` would not make them usable.
ChannelCatalog is the concrete integration seam.

Each compiled channel declares:

```json
{
  "name": "object__ball__pos_w",
  "dtype": "float32",
  "shape": ["T", "N", 3],
  "producer": "entity_state",
  "access": "shared_shaping",
  "metric_role": "progress",
  "max_bytes_per_rollout": 2400000
}
```

Implementation threads the catalog through every metric path:

1. The recorder writes only catalogued channels and a catalog hash.
2. Metric generation receives the exact catalog.
3. Validation accepts `base ALLOWED_ARRAYS ∪ project catalog names`, checks
   shape/dtype/budget, and still rejects unknown array names.
4. Runtime loads only declared arrays after matching the catalog hash.
5. Calibration, best-of-N evaluation, detail rendering, and observable
   extraction use the same resolved allowlist.

Generated object channels include pose/velocity, contact-pair state,
object-to-region distance, inside-region state, and a compiled success event.

Firewall behavior is explicit:

- `metric_only` channels, including the authoritative success event, are
  omitted from the reward `info` dictionary. This is real access control.
- `shared_shaping` channels may be granted to reward generation when useful,
  such as a ball-to-goal distance.
- The existing partition gate remains an advisory warning when reward and
  metric use the same physical observable. It is not falsely described as a
  hard ban.
- The existing post-write hard gate still blocks lowering a same-named
  positive completion threshold. Metric validation and calibration remain
  unchanged.
- A reward may reconstruct a metric proxy from shared raw state; the advisory
  partition warning, anti-gaming replay screens, metric degenerates, and
  Goodhart stop remain necessary. The design does not claim perfect
  information isolation.

The TaskSpec success predicate supplies the required completion-gate semantics,
while generated metric code still composes and validates the objective score.

## 10. Reward, diagnoser, curriculum, and memory

The compiler emits one typed world-abstraction block for reward generation:
available entities, frames, semantic body roles, granted channels, terrain
summary, and contact relations. The reward generator cannot reference a
channel absent from this block.

Three adaptation rates coexist:

1. Within-run mjlab curriculum promotes or demotes environment origins using
   per-level traversal or waypoint success.
2. Per-iteration diagnosis edits only registered `train.variations` by
   stable parameter ID. The apply path resolves the pointer, validates the
   new value, and rolls back that individual edit on failure.
3. Between-run re-authoring may add/remove structural features, but it creates
   a new evaluation lineage and passes the full gate chain.

The diagnoser receives success and traversal statistics by difficulty level.
Intermediate-success configurations are prioritized as learning candidates,
but objective fitness and dense progress remain the only selection keys.

Run-case memory records:

- WorldSpec, TaskSpec, and env-spec versions;
- parameter ID, old/new distribution, and curriculum state;
- robot and project scope;
- objective verdict and behavior signature.

Keep/revert moves the reward, all three environment artifacts, and their
resolved manifests as one tuple.

## 11. Clarification interaction

The Author returns a draft plus an underspecification report. Every parameter
is tagged `prompt`, `user`, or `default`.

A schema table marks load-bearing parameters: success semantics, goal scale,
object count, required robot capability, incline, roughness, gap distance,
obstacle height, contact permission, and physics values that materially change
the task.

The deterministic Clarifier:

1. queues every load-bearing parameter that remains defaulted;
2. groups related dimensions into short pages of at most five questions, but
   never drops later questions;
3. offers two to four bounded physical choices plus, on every question,
   “System decides (default: X — reason)”;
4. records the selected value and source in
   `clarifications_v<N>.json`;
5. revalidates after each page.

Examples:

- “Maximum slope?” 8°, 15°, 25°, or “System decides (default: 15°, moderate
  for this robot envelope).”
- “Platform height range?” 0.15–0.30 m, 0.30–0.55 m, or “System decides
  (default: 0.20–0.45 m, bounded by leg length).”
- “May the humanoid use its hands on the ball?” yes, no, or “System decides
  (default: no, because the prompt says score rather than carry).”

CLI interactive mode pages through the same JSON payload. `--yes` and
headless operation select every disclosed system default. The UI job enters
`awaiting_clarification`; a configured timeout selects defaults and records
that fact. No ambiguity is silently omitted merely because more than five
questions exist.

## 12. Knowledge-graph integration

Authoring retrieval has two layers:

- `query_papers` searches title, abstract, applicability rationale, and tags,
  optionally restricted to campaign tier/domain. This includes unextracted
  A/B sources.
- existing technique/failure retrieval supplies extracted claims and evidence.

WorldSpec `meta.grounding` records the paper IDs actually used. Generated
parameter suggestions distinguish source claims from RewardSculptor design
inferences. Run-case retrieval contributes measured local experience, scoped
with project and robot metadata.

The Phase 0 graph rules remain mandatory: one shared DB path, staleness-aware
embeddings, corroborating evidence lists, source-aligned citations, and
`sculpt kg doctor` after each corpus expansion.

## 13. Acceptance flows

### 13.1 Uneven terrain

The prompt resolves a locomotion-capable robot, asks about roughness, slope,
and whether jumping is required, then authors a curriculum grid. Non-flat
terrain automatically requests height observations through the simulator
capability adapter. Evaluation terrain is sampled once from the persisted
seed. The task goal is velocity tracking or waypoint traversal; success is
stable progress without forbidden fall contacts.

### 13.2 Box parkour

The prompt authors a course feature list with stable IDs. Clarification covers
platform heights, gaps, and beam width. Robot geometry bounds reject
unreachable segments. Train variations change numeric dimensions, while
course topology stays fixed within a run. Waypoint completion drives both
curriculum statistics and the sparse objective gate.

### 13.3 Humanoid with gripper scores a ball

The required `grasp`/gripper capability selects a compatible humanoid
variant. The world contains a free sphere, a physical goal frame, and a
non-colliding goal-mouth region. TaskSpec declares an object-to-region goal,
contact permissions, ball and goal-relative observations, reset
distributions, and an inside-region hold predicate. The compiler produces
entity state, contact, distance, and success channels. Evaluation fixes ball
pose, goal pose, object physics, terrain, and seed.

Nothing in the compiler says “soccer” or “G1.” Soccer is this composition of
generic capabilities, entities, contacts, regions, and predicates.

## 14. Phased implementation plan

Each phase ends with focused tests, a real mjlab smoke/E2E where stated,
`sculpt kg doctor`, and an adversarial review against this document.

### P1: capability spine and terrain

- RobotCapability and SimulatorCapability registries.
- WorldSpec shared/train validator and versioning.
- flat/generator terrain compiler;
- curriculum-grid semantics and height-observation adapter;
- resolved evaluation manifest and build/penetration/settle gates;
- `sculpt world author|show|validate`.

E2E: G1 or another installed humanoid walks on a rough/slope curriculum while
rollout reuses a byte-identical evaluation manifest.

### P2: courses and world diagnosis

- stable-ID course grammar and assembler;
- robot-relative feasibility envelopes;
- waypoint command adapter and per-level statistics;
- train variation registry and diagnoser edits;
- atomic keep/revert of the complete version tuple.

E2E: quadruped box parkour, followed by the same schema on a humanoid.

### P3: objects, tasks, and channels

- primitive/free/fixed entities and placement events;
- regions, contact selectors, TaskSpec, and success predicates;
- ChannelCatalog threaded through recorder, validation, runtime, calibration,
  and reports;
- metric-only versus shared-shaping channel access;
- object-task degenerate tests such as contact flicker and region-edge
  camping.

E2E: object-to-region with a simple installed robot.

### P4: gripper humanoid acceptance case

- robot-variant asset and capability resolution;
- gripper/end-effector observation and contact roles;
- staged exploration support or reference initialization where required.

E2E: the requested humanoid-with-gripper ball-to-goal task. This phase is
explicit; it is not deferred to an unspecified future robot library.

### P5: clarifier and product integration

- complete underspecification report and paginated questions;
- CLI/UI state machine, timeout defaults, and provenance ledger;
- KG authoring retrieval and citations;
- scene preview/edit UI and lineage display.

## 15. Proof-of-concept scope

`RewardSculptor/scripts/world_schema_poc.py` is a scene smoke test, not the
implementation. It validates:

- a small WorldSpec-shaped `shared/train` dictionary;
- generated rough terrain through mjlab configuration;
- a free ball, a fixed box entity, and a visual goal region;
- actual `MjSpec` compilation on CPU;
- finite/no-sink settle behavior;
- rejection of an initial object/geometry overlap.

It does not validate the course assembler, a physical frame-shaped goal,
TaskSpec, contacts, observation adapters, ChannelCatalog, or clarification.
Those claims remain implementation work in P2–P5.

## 16. Proof-of-concept result

Verified on 2026-07-18 with mjlab 1.3.0 and MuJoCo 3.7.0:

- process exit code: 0;
- compiled bodies: 5;
- compiled geoms: 17;
- heightfields: 3;
- free ball and fixed goal entity: present;
- goal-region site: present;
- valid scene: finite and above the no-sink threshold after 300 steps;
- deliberately overlapping ball/entity scene: rejected at 13.5 cm initial
  penetration;
- final result: all smoke assertions passed.

This proves the schema-to-mjlab scene path is real for terrain, entities, and a
region. It does not erase the limitations listed in section 15.

## 17. Risks and implementation decisions

- MuJoCo-Warp field expansion must be confirmed per field; unsupported ranges
  become reset-time state sampling or fixed compiled values.
- Structural scene size can exhaust contact or constraint capacity. Budgets
  are admission gates, not warnings.
- Numeric ranges can be physically plausible but unlearnable. Robot envelopes,
  paper precedents, clarification, and optional probe rollouts mitigate this.
- Object metrics create correlated gaming surfaces. Metric-only channels,
  degenerate validation, replay screens, and Goodhart stopping remain active.
- Fitness cannot be compared across resolved evaluation hashes. A shared-field
  edit always starts a new baseline.
- Hard exploration in whole-body manipulation may require staged curricula,
  reference states, or a reusable low-level controller. The Author can select
  declared scaffolds, but cannot pretend a valid scene implies a trainable
  task.
- A fixed relative-vector goal observation is the simplest P3 starting point.
  The channel catalog should retain achieved/desired-goal compatibility so HER
  can be added without changing TaskSpec semantics.

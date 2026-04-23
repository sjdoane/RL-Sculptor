# Migration — legacy Gymnasium → mjlab

This guide covers migrating projects from the original Gymnasium + SB3
pipeline to the mjlab (MuJoCo-Warp) pipeline that shipped in M2 of the
mjlab pivot.

**TL;DR** — *never modify an existing project in place*. Migration is
always a **fork**: you create a new project on the mjlab path that
targets an equivalent robot, and leave the legacy project untouched
(so its git history, runs/, and reward iterations remain reproducible).

## When to migrate

Migrate when both are true:

1. Your project's behavior goal is feasible on a library-registered
   mjlab robot — e.g. the project trained Hopper to hop forward; the
   library has Unitree Go1 (quadruped locomotion; same physics signal,
   richer state space). The mjlab stack gives you 10-100x env-steps/sec
   on an RTX-class GPU.
2. You have a CUDA 12.4+ GPU with ≥ 6 GiB VRAM. Check [Settings →
   GPU & training](#) in the UI; the panel is green if so.

## When to stay on the legacy path

- Quick experiments on a CPU-only machine. gym_sb3 stays faster than
  mjlab at small num_envs on CPU.
- You're iterating on the reward contract itself (scalar-only) and
  don't want the additional `compute_reward_batched` discipline.
- The target robot has no mjlab task registered and lives in the
  `preview_only` bucket in the library. The fork would just be
  cosmetic — stay on gym.

## How the UI surfaces migration

- Project cards show an **amber warning banner** when the project's
  adapter class is no longer in `sculptor.adapters.ADAPTER_REGISTRY`.
  Projects created before M5 that reference `gym_sb3` / `mjlab`
  adapters continue to work — those adapters are still registered —
  so the warning appears only when someone removes an adapter from
  the registry.
- The CreateProjectDialog's adapter dropdown shows all registered
  adapters, with `⏳` badges for coming-soon ones (Isaac Lab / MJX /
  RLlib). Picking a coming-soon adapter lets the project scaffold but
  leaves training disabled with a link to the adoption guide.

## Fork workflow (manual)

1. In the running UI, go to **Library**, filter by `mjlab_ready` or
   the category matching your existing robot (Hopper → look under
   Other / biped; Go1 is under Quadruped).
2. Click the target robot card → `Create project with this robot`.
3. The CreateProjectDialog opens. Leave the adapter default (`mjlab`
   for mjlab-ready entries).
4. Pick a `num_envs` that fits your VRAM — the inline estimate shows
   green/amber/red for headroom.
5. Submit. The new project scaffolds with:
   - a fresh `rewards/v0.py` template (gym-shaped; extend to batched
     per the docstring).
   - `kg_seeds.yml` auto-populated with the library entry's paper
     references.
   - a fresh git repo inside the project dir.
6. The legacy project is **untouched**. Its `runs/`, `rewards/`, and
   git history remain on disk for reproducibility.

## What to carry over manually

- **Reward ideas.** Your legacy `rewards/v<N>.py` is still there.
  Port the interesting ideas (component weights, bonus/penalty terms)
  into the new project's `rewards/v0.py` — but re-validate against the
  new robot's state schema. `sculptor.adapters.mjlab._INFO_KEYS` and
  the per-task `state_schema` are the authoritative contract.
- **Behavior goal.** Copy the behavior_goal verbatim from the legacy
  project's config.toml into the new one. The behavior goal is
  stack-agnostic.
- **Research citations.** If you hand-added arxiv IDs to the legacy
  project's `kg_seeds.yml`, check whether they're also in the library
  entry's `references`. If not, add them to the new project's
  `kg_seeds.yml` so the KG gets the same coverage.

## What NOT to carry over

- Checkpoints. The legacy gym_sb3 `.zip` is not compatible with
  mjlab's rsl_rl `.pt`.
- `runs/iter_*/diagnosis.json`. Failure-mode diagnoses are conditioned
  on the training curve + rollout; they won't transfer.
- `config.toml`'s `[adapter]` table. The new project scaffolds with
  the correct [adapter] for the library entry; don't copy the old one.

## Backward compatibility guarantees

- The `gym_sb3` adapter stays in the tree and in `ADAPTER_REGISTRY`
  indefinitely. Every legacy project continues to `uv run sculpt run`
  as before.
- The reward-module contract is additive: old modules (scalar-only
  `compute_reward`) still validate. `supports_batched` defaults to
  False.
- The UI's existing upload flow (custom URDF) is unchanged.

See [README.md](../README.md) for the top-level adapter matrix.

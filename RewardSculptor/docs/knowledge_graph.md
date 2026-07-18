# Knowledge graph — how to grow it

The sculptor's knowledge graph is what gives Claude grounded literature
to cite when rewriting reward functions. The bigger and more domain-
relevant the KG, the better the edits. This doc covers:

1. Where the KG lives on disk (**M7 Phase 1: shared across all projects**).
2. What's in the KG by default, including the bundled pre-extracted seed.
3. How to regenerate the bundled seed when `kg_seeds_global.yml` changes.
4. How to add a single paper from the UI.
5. Proposed follow-up: prompt-time research ("if you don't know X,
   go find a paper about it and add it to the KG").

## Where the KG lives — shared by default (Phase 1)

As of M7 Phase 1 the KG lives at a **single user-wide path**:

```
~/.local/share/sculptor/kg/graph.db
```

All projects read and write from the same DB. Creating a new project no
longer triggers a fresh Claude extract — the new project sees the
existing 46 seed papers (plus any you added via "Research a topic") from
day one.

**Overrides** (highest precedence first):

- `$SCULPTOR_KG_PATH` — wins when both are set.
- `$RS_KG_PATH` — UI-facing override. Used by the backend test suite to
  redirect to a per-test tmp DB.
- Legacy per-project / per-directory DBs (`<cwd>/kg/graph.db`,
  `<project>/kg/graph.db`) are **no longer honored anywhere**
  (sculptor side removed 2026-07-03 loop 5a; UI backend aligned
  2026-07-18 Phase-0 hardening). A leftover legacy file only triggers a
  warning pointing at `sculpt kg merge <path>` — one graph, one path.

**First-start bootstrap**: when the backend boots and finds no shared
DB, it copies the bundled pre-extracted sqlite from
`RewardSculptor/examples/kg_preextracted.db` into place. Zero Claude
calls on first launch — you get 46 papers + ~400 entities instantly.
Bootstrap is a no-op when `RS_KG_PATH` or `SCULPTOR_KG_PATH` is set so
tests don't accidentally seed the tmp DB.

`GET /system/kg/stats` returns `{papers, techniques, failure_modes,
reward_components, environments, edges, embeddings, db_path,
db_size_bytes, last_modified}` — surfaced on the Settings page's
"Knowledge graph" card.

## Default content

When you create a project from a library robot (Library tab → Create),
the robot's `references` list is auto-written to `kg_seeds.yml` and —
if `ANTHROPIC_API_KEY` is set — an ingest + extract job fires
immediately. Per-robot seed counts as of M5:

| Library entry | Papers | Repos |
|---------------|:---:|:---:|
| Unitree G1 | 3 | 2 |
| Unitree Go1 | 3 | 2 |
| ANYmal C | 3 | 2 |
| Unitree Go2 | 2 | 1 |
| Booster T1 | 1 | 2 |
| Yam arm | 0 | 0 |
| Cartpole (toy) | 0 | 0 |

Legacy Gymnasium entries (Hopper / Ant / HalfCheetah / Walker2d /
Humanoid) start with empty references — they were the pre-M3 default
and aren't tied to any specific paper.

## Bulk-seeding from `kg_seeds_global.yml`

[`examples/kg_seeds_global.yml`](../examples/kg_seeds_global.yml) is a
curated superset of ~50 arxiv papers covering:

- **SEA / jumping robotics** (15 papers from the AME456 research log —
  quadruped jumping, series-elastic actuators, parkour, stage-wise
  reward shaping, Pinto, Stanford Doggo, etc.).
- **mjlab-ready robot platforms** (13 papers: G1, Go1, Go2, ANYmal C,
  T1, legged_gym, Walk These Ways, etc.).
- **Foundational RL** (9 papers: PPO, SAC, DDPG, GAE, DeepMind Control
  Suite, OpenAI Gym, "RL That Matters").
- **Reward shaping + curriculum + manipulation** (8+ papers).

Ingest into any existing project:

```bash
# From the project's directory:
cd ~/.local/share/reward-sculptor/projects/<slug>

# Call sculpt's CLI against the global seeds file:
uv run --project ~/projects/RewardSculptor sculpt kg ingest \
    ~/projects/RewardSculptor/examples/kg_seeds_global.yml

# Then extract entities (Techniques / FailureModes / RewardComponents
# / Environments) — requires ANTHROPIC_API_KEY:
uv run --project ~/projects/RewardSculptor sculpt kg extract --all
```

Expected: ~50 papers ingested (PDFs cached at `kg/pdfs/`), ~100+
Techniques / FailureModes / RewardComponents extracted. First ingest
is slow (~2-5 minutes on cold arxiv cache); subsequent runs are
idempotent.

## Regenerating the bundled pre-extracted DB

When you edit `examples/kg_seeds_global.yml` (adding or removing
papers), regenerate the committed `examples/kg_preextracted.db` so new
users / fresh installs get the updated content on first boot:

```bash
cd ~/projects/RewardSculptor
./scripts/regenerate_kg_preextracted.sh
# …2-5 min (arxiv PDFs) + 2-5 min (Claude extraction) + ~$2-3 tokens…
git add examples/kg_preextracted.db
git commit -m "kg: regenerate pre-extracted DB"
```

The script ingests + extracts into a `/tmp/` staging DB, then moves the
result over `examples/kg_preextracted.db` only on success — a failed
run leaves the previous committed DB untouched. Mark the binary in
`.gitattributes` so `git diff` doesn't try to print it:

```
examples/kg_preextracted.db binary
```

## Adding a single paper from the UI

Go to any project's **KG tab** → `Add seeds`. Paste an arxiv ID
(`2401.16337`) or a full URL (`https://arxiv.org/abs/2401.16337`).
Check `auto extract` to trigger extraction immediately after ingest.

## Integrity: `sculpt kg doctor`

The shared graph accretes from many writers (seed ingest, extraction,
run-case memory, UI research jobs, legacy merges). `sculpt kg doctor`
reports every silent-degradation class in one pass — dangling edges,
orphan embeddings, stub-titled papers, dead `full_text_path` sidecars,
missing/stale embeddings (embeddings carry a `text_hash` of the exact
embedded text since 2026-07-18, so a description enriched by a later
extraction re-embeds instead of serving stale geometry) — and
`--fix` repairs the mechanical ones (`--reembed-all` for a full
embedding rebuild, `--no-network` to skip the two arxiv-touching heals).
Read-only without `--fix`; exits 1 when unfixed issues remain.

## Prompt-time research — implemented

Implemented as `sculptor/kg/research.py` (`research_topic`) + the UI
backend's `POST /projects/{slug}/kg/research` job: Claude proposes
arxiv IDs for a topic, IDs are normalized + deduped against the KG,
each surviving ID's REAL title/abstract is fetched from arxiv and
embedding-checked against the topic (hallucinated-ID guard,
threshold 0.15), then ingest + extract run. The original design sketch
below is kept for history:

1. When the user types a behavior goal ("jump 30 cm vertically with a
   pogo-stick SEA"), sculptor's diagnoser queries the KG for relevant
   techniques. If the domain-filtered results are thin (e.g. fewer
   than N techniques match "jumping" + "series_elastic"), the run is
   paused and the UI shows a banner: *"KG has 3 jumping papers + 0
   SEA papers. Research this topic now?"*
2. Clicking **Research** fires a new endpoint `POST /kg/research` with
   `{ topic, project_slug }`. The backend:
   - Asks Claude Opus 4.7: "Give me 10 arxiv IDs for the following
     topic: ...". Claude returns the list.
   - Batch-ingests via `sculpt.kg.ingest.ingest_from_seeds`.
   - Runs `sculpt.kg.extract.extract_all` on the fresh papers.
   - Returns the updated KG stats.
3. The sculpt run then continues with the enriched KG.

**Why it's deferred:** the `POST /kg/research` endpoint is a moderate
lift (~200 lines + one new Anthropic system prompt). It also needs a
design call: do we ALWAYS research when coverage is thin, or only on
user confirmation? Recommend: user confirmation by default, with a
`auto_research=true` project setting to opt in.

Until that ships, the workaround is manual: ingest
`kg_seeds_global.yml` upfront, then add individual papers as new
topics arise via the KG tab's `Add seeds` dialog.

## MuJoCo physics editor — also deferred

The user asked for an in-UI physics-parameter editor (gravity,
timestep, joint damping, mass, friction). Design sketch:

- New **Physics** tab on ProjectDetail, next to Rewards / KG / Runs.
- Reads the project's MJCF file (`rewards/robot.xml` or the
  library's MJCF resolved via `robot_descriptions`).
- Parses with `mujoco.MjSpec` — exposes editable fields grouped by
  section:
  - `<option>`: timestep, gravity, integrator, solver.
  - `<worldbody>` joints: damping, frictionloss, stiffness, armature.
  - `<actuator>`: forcerange, gear, kv, kp.
  - `<geom>`: friction, mass / density.
- Write-back round-trips through `MjSpec.to_xml()` so the edit stays
  valid.
- Each edit creates a git commit in the project dir (`physics: bump
  joint_damping 0.01→0.05`).

**Estimated effort:** 6-10 hours (MJCF editor UI is the bulk). This is
a good M7 candidate. Not blocking any current workflow.

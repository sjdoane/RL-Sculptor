# Robot library — adding, editing, regenerating

The UI's Library browser is backed by a single YAML file at
[`reward-sculptor-ui/backend/data/robot_library.yml`](../../reward-sculptor-ui/backend/data/robot_library.yml).
This document covers:

- Adding a new robot entry.
- Regenerating the thumbnail PNG.
- How KG seeding works at project-creation time.

## Library entry shape

```yaml
- slug: unitree_go1                 # lowercase + underscores; unique
  display_name: "Unitree Go1"
  category: Quadruped               # one of the 8 enumerated categories
  description: "..."                # 1-2 sentences
  source: menagerie                 # menagerie | mjlab_builtin | gymnasium_builtin
  menagerie_package: go1_mj_description  # robot_descriptions loader name
  training_support: mjlab_ready     # mjlab_ready | preview_only | gymnasium_compatible
  is_smoke_test_target: false       # true only for Cartpole
  preconfigured_tasks:
    - task_id: Mjlab-Velocity-Flat-Unitree-Go1
      display_name: "Velocity tracking (flat)"
      recommended_num_envs: 2048
  references:                       # verified URLs only — see below
    - kind: paper
      url: https://arxiv.org/abs/...
      citation: "Author et al., Title, Venue Year"
    - kind: repo
      url: https://github.com/org/repo
      citation: "org / repo"
  thumbnail_path: robots/unitree_go1.webp  # relative to frontend/public/
```

## Adding a new robot

### 1. Find the `menagerie_package` name

The `robot_descriptions` package exposes each Menagerie model as a
module ending in `_mj_description`. Enumerate available loaders:

```bash
cd ~/projects/RewardSculptor
uv run python -c "
import os, robot_descriptions
p = robot_descriptions.__path__[0]
for f in sorted(os.listdir(p)):
    if f.endswith('_mj_description.py'):
        print(f.replace('.py', ''))
"
```

Pick the loader matching your robot (e.g. `g1_mj_description` for
Unitree G1). The library YAML stores this in `menagerie_package`.

### 2. Decide `training_support`

- **`mjlab_ready`** — only if `mjlab.tasks.registry.list_tasks()`
  contains one or more task IDs that target your robot. The library
  loader's D-guard (backend/services/robot_library.py) auto-demotes
  entries at startup when their declared task IDs aren't in the live
  registry, so it's safe to claim `mjlab_ready` optimistically.
- **`gymnasium_compatible`** — only for MuJoCo Gymnasium envs (Hopper
  / Ant / Walker2d / HalfCheetah / Humanoid); `source:
  gymnasium_builtin`.
- **`preview_only`** — default for every Menagerie robot without an
  mjlab task. Still loads in the preview renderer; training button is
  disabled in the UI.

### 3. References

**Every URL must be verified** with curl / HEAD before merging:

```bash
for url in "https://arxiv.org/abs/..." "https://github.com/..."; do
  curl -s -o /dev/null -w "%{http_code} %{url_effective}\n" -L "$url"
done
```

Reject any URL not returning `200`. `references: []` is a valid entry;
invented citations are not. See the existing entries for tone and
format.

### 4. Validate the YAML loads

```bash
cd ~/projects/reward-sculptor-ui
uv run pytest backend/tests/test_library.py -q
```

`test_library_loads_without_errors` fails loudly if the new entry's
slug / category / URL / required fields are malformed.

## Regenerating thumbnails

Thumbnails are committed to
`reward-sculptor-ui/frontend/public/robots/*.png`. Regenerate one or
all via the script at
[`scripts/generate_library_thumbnails.py`](../../reward-sculptor-ui/scripts/generate_library_thumbnails.py):

```bash
cd ~/projects/reward-sculptor-ui

# Render just one (fast; useful when adding a new entry):
uv run python scripts/generate_library_thumbnails.py --only-slug unitree_g1

# Render all (5 min on first run due to Menagerie clone; faster after):
uv run python scripts/generate_library_thumbnails.py

# Force re-render when the camera fix / size changes:
uv run python scripts/generate_library_thumbnails.py --force
```

Render params: 320×240 PNG, iso view, arms and grippers get 20% camera
pullback (applied automatically based on category).

**First-run caveat:** `robot_descriptions` clones the Menagerie repo
on first use (~700 MB). The clone is cached at
`~/.local/share/robot_descriptions/`. Subsequent script runs reuse it.

## KG seeding on project creation

When a user picks a library robot in the UI and creates a project,
the backend:

1. Writes the library entry's `kind: paper` references into
   `<project>/kg_seeds.yml` as `{arxiv_id, citation}` entries (arxiv
   URL → ID via regex).
2. If `$ANTHROPIC_API_KEY` is set, fires a background
   `kg_ingest_extract` job via the existing `kg_jobs.run_ingest_extract_job`
   runner. The job appears in the active-jobs list.
3. Stashes `kind: repo` references on
   `metadata.json::robot_source.related_repos` for the future
   "Related repos" KG panel. Not sqlite-ingested today.

A library entry with `references: []` is valid — the project
scaffolds normally, and the user can hand-add KG seeds via the KG
tab's AddSeedsDialog.

## Categories — when to pick which

| Category | When |
|----------|------|
| Quadruped | 4-legged locomotion. Cassie is placed here too (bipedal legged without torso/arms). |
| Humanoid | Full-body bipedal with torso + arms. |
| Arm | Stationary-base manipulator (6-7 DoF). |
| Gripper_Hand | End-effector only (2-finger grippers, anthropomorphic hands). |
| Mobile_Manipulator | Arm on a mobile base (Stretch, TIAGo). |
| Drone | Aerial. |
| Biomechanical | Insect / human / creature models intended for biomech research. |
| Other | Sensors, partial-robot models, Cartpole. |

The UI's default category filter is {Quadruped, Humanoid, Arm,
Gripper_Hand}. Cartpole under `Other` is opt-in.

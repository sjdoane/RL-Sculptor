# M7 plan — bootstrap for a new Claude Code window

Read this end-to-end before doing anything. It's self-contained: user
profile, system state, architecture, known bugs, and a prioritised task
list so you can pick up cold without reading the full CONTEXT.md
change log (though you should still skim it — CONTEXT.md is the
authoritative session history).

**Companion docs to read in order of priority:**
1. `~/projects/CONTEXT.md` (session history + working-style rules).
2. `~/projects/M7_PLAN.md` (this file — the plan).
3. `~/projects/RewardSculptor/HISTORY.md` (sculptor library build log).
4. `~/projects/MJLAB_PIVOT_DESIGN.md` (M0 design doc — architecture
   of every adapter + library + KG integration).

---

## Who you are helping

Sam Doane — USC ME undergrad, AME 456 capstone on a **quadruped
jumping robot with series-elastic actuators (SEAs)**. WSL2 Ubuntu
24.04 on an RTX 5070 Laptop (8 GiB VRAM, sm_120, CUDA 13.0). Python
3.13.5 via uv. `ANTHROPIC_API_KEY` already set in
`~/projects/RewardSculptor/.env` (auto-loaded by `sculptor/__init__.py`).

**Working style** (repeat it back to yourself every turn):
- Terse. File:line references + metrics over prose.
- Flag gotchas early; say the trade-off when you make a judgment call.
- Confirm before destructive actions.
- No features beyond what the task requires.
- Log every meaningful change to `CONTEXT.md` incrementally.

---

## The stack in one screen

```
~/projects/
├── AME456/                     # capstone project (NOT modified by sculptor)
│   └── files/
│       ├── docs/RESEARCH_LOG.md  # gold mine of arxiv refs for the KG
│       ├── quadruped_mjx_env.py  # MJX quadruped env; imports sculptor.reward
│       └── reference_repos/      # Curriculum-Quadruped-Jumping-DRL etc.
├── RewardSculptor/             # the core library
│   ├── sculptor/adapters/      # gym_sb3 (ready), mjlab (ready), isaac/mjx/rllib (stubs)
│   ├── sculptor/kg/            # schema, store (sqlite), ingest (arxiv), extract (Claude), query
│   ├── sculptor/{diagnose,edit,sculpt,timelapse}.py
│   ├── examples/kg_seeds_global.yml   # 50-paper curated superset (NEW in last session)
│   └── docs/                   # adapters.md + migration/robot_library/wsl_setup/e2e/perf/knowledge_graph
└── reward-sculptor-ui/         # FastAPI + React control panel
    ├── backend/                # routes/services/tests (129 backend tests passing)
    ├── frontend/               # React + Vite + shadcn (TypeScript clean)
    └── scripts/generate_library_thumbnails.py
```

**Test health (as of handoff):** 129 backend + 76 sculptor + 2 GPU
(cached) = **205 passing**. Frontend typecheck clean.

**What works end-to-end right now:**
- Legacy gym_sb3 (Hopper) sculpt-run dry-run: `uv run sculpt run
  --config examples/hopper/config.toml "run fast" --iterations 3
  --dry-run` completes in ~50 s.
- UI: Library browser, project creation (library_slug-driven), adapter
  dropdown with coming-soon badges, Dashboard with GPU widget,
  Settings GPU panel, KG tab with bulk-seed button.
- mjlab GPU smoke test: `pytest -m gpu` trains Go1 100 iters at
  num_envs=1024 in ~90 s.

**What's broken (must-fix in M7):** the end-to-end mjlab sculpt-run
flow. When a user clicks **New run** on a newly-created mjlab
project, the sculpt subprocess exits 1 at ~19 s. Root cause
diagnosed below.

---

## Known bug — `sculpt exited with code 1` on mjlab projects

**User symptom:** fresh mjlab project (e.g. Unitree G1 from library),
behavior goal typed, iterations=20, click Launch. Run errors after
~19 s with `Error: sculpt exited with code 1` and WS CLOSED,
0/20 iters completed, no metric data.

**CONFIRMED by second reproduction after KG extract completed**
(2026-04-21). User waited for the `extracting entities via Claude`
banner to clear, then retried a sculpt run on the same G1 project —
**identical ~19 s exit-1 failure**. This definitively rules out KG
extract concurrency as the cause. The bug is in the training path
itself, not in KG orchestration.

**Root cause:** the `rewards/v0.py` template written by
[`sculpt_init`](RewardSculptor/sculptor/sculpt.py:862) is
gym-shape — exports ONLY a scalar `compute_reward(state, action,
next_state, info)`. The mjlab runner at
[`_mjlab_runner.py:SculptorRewardTerm`](RewardSculptor/sculptor/adapters/_mjlab_runner.py)
calls `compute_reward_batched` on the loaded module and raises
`AttributeError: module … has no attribute 'compute_reward_batched';
required when training with MjlabAdapter`. The subprocess exits 1;
sculptor's RunManager surfaces it as the generic "exited with code 1".

**Timing fingerprint matches the hypothesis.** 19 s ≈ mjlab + torch
+ mujoco_warp cold-import (~15-20 s measured in M3 benchmarks) PLUS
immediate `AttributeError` on the first call to
`compute_reward_batched`. No training iteration begins — which
matches iters 0/20 completed + no metric data.

**Why it's not the KG extract concurrency (eliminated):** (a) the
user repro'd post-KG; (b) the extract job writes only to sqlite
via brief WAL transactions; the sculpt subprocess doesn't even
touch sqlite until Stage-2 diagnose (iteration 1+); (c) no write
lock contention would cause a sub-second failure after a 19 s cold
import.

**Diagnostic commands for the new-window assistant to confirm before
coding the fix:**

```bash
# 1. Find the project dir.
ls ~/.local/share/reward-sculptor/projects/

# 2. Inspect the scaffolded v0.py — should show only compute_reward,
#    no compute_reward_batched.
cat ~/.local/share/reward-sculptor/projects/<slug>/rewards/v0.py | grep "^def "

# 3. Confirm the hypothesis definitively.
cd ~/projects/RewardSculptor && uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('v0',
    '/home/samjd/.local/share/reward-sculptor/projects/<slug>/rewards/v0.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print('has compute_reward:         ', hasattr(mod, 'compute_reward'))
print('has compute_reward_batched: ', hasattr(mod, 'compute_reward_batched'))
print('REWARD_SPEC supports_batched:', mod.REWARD_SPEC.get('supports_batched', False))
"
# Expected output:
#   has compute_reward:          True
#   has compute_reward_batched:  False    ← the bug
#   REWARD_SPEC supports_batched: False

# 4. Also inspect the backend stderr (the terminal running ./run.sh)
#    for the actual Python traceback from the mjlab runner subprocess.
#    It should contain:
#    "AttributeError: reward module ... missing compute_reward_batched"
```

If step 3 shows `False` for `compute_reward_batched`, the diagnosis
is confirmed — proceed with fix #1 below. If it shows `True`, the
bug is somewhere else and you should re-read the backend stderr
+ subprocess stderr for the real traceback before coding.

**Three fixes, in order of preference:**

1. **Adapter-aware v0 template.** Extend
   [`sculpt_init`](RewardSculptor/sculptor/sculpt.py:862) to accept
   the adapter name + write a different `rewards/v0.py` when adapter
   is `mjlab`. The mjlab template exports BOTH:
   ```python
   def compute_reward(state, action, next_state, info):
       # Scalar path: called for validation + UI probe.
       return 0.0, {"alive_bonus": 1.0}

   def compute_reward_batched(state, action, next_state, info):
       # Batched path: called per-step during training.
       # state/next_state are dict[str, torch.Tensor] shape (N, *feat);
       # action is (N, action_dim); info is dict[str, (N,) Tensor].
       import torch
       n = action.shape[0]
       ones = torch.ones(n, device=action.device)
       return ones, {"alive_bonus": ones}
   ```
   Plus `REWARD_SPEC["supports_batched"] = True`.

2. **Runner fallback wrap.** In
   [`_mjlab_runner._build_sculptor_term_class`](RewardSculptor/sculptor/adapters/_mjlab_runner.py),
   if the loaded module exports only `compute_reward` (scalar),
   auto-wrap it: run compute_reward per-env in a Python loop. Slow
   (~10× slower) but keeps sculpt-run working. Warns on load.

3. **Surface the error better.** Regardless of #1/#2, when
   `MjlabAdapter.train()` raises, the UI should show the actual
   exception text (not just "exited with code 1"). Extend
   [`backend/services/run_manager.py`](reward-sculptor-ui/backend/services/run_manager.py)
   to parse the subprocess stderr for `AttributeError`, `ImportError`,
   `RewardValidationError` and surface them as specific error
   classifications (mirrors the CUDA-error classifier pattern from
   [`backend/services/cuda_errors.py`](reward-sculptor-ui/backend/services/cuda_errors.py)).

**Recommended:** do #1 (proper fix) + #3 (observability). Skip #2 —
the wrapper would silently convert a correctness bug into a
performance one.

---

## Architectural decision 1 — shared knowledge graph

User's stated preference: **one KG shared across all projects**,
seeded once, continuously grown via prompt-time research. Per-project
KGs get abandoned / re-seeded; the shared model amortises the cost.

**Current architecture** (per-project):
- [`backend/services/kg_store.project_kg_db_path`](reward-sculptor-ui/backend/services/kg_store.py)
  returns `<project_dir>/kg/graph.db`. Each project has its own sqlite.
- sculptor's `SculptorKG` (under `sculptor/kg/store.py`) defaults to
  `./kg/graph.db` (cwd-relative), overridable via `SCULPTOR_KG_PATH`
  env var.

**Proposed shared architecture:**
- Default KG path becomes `$HOME/.local/share/sculptor/kg/graph.db`.
- UI backend's `project_kg_db_path` ignores the `<project>` arg and
  returns the shared path (keep signature for backward compat).
- Each project's `config.toml` still has
  `[kg].environment_tag = "continuous_locomotion" | "manipulation" | …`.
- Diagnose queries tag by env + behavior_goal semantic
  similarity — filtering for relevance without discarding the rest.
- sculptor library: extend `SculptorKG.__init__` to honor
  `RS_KG_PATH` / `SCULPTOR_KG_PATH` / fall back to the shared
  default.

**Implementation** (est. 30 min):
1. Edit
   [`kg_store.project_kg_db_path`](reward-sculptor-ui/backend/services/kg_store.py)
   to return `Path.home() / ".local/share/sculptor/kg/graph.db"`
   unless `RS_KG_PATH` env var is set. Create the parent dir on
   first call.
2. `sculptor/kg/store.SculptorKG.__init__` — same default logic,
   consistent env var.
3. Preserve per-project isolation opt-in: if
   `<project_dir>/kg/graph.db` already exists, use it (legacy
   behavior for existing projects).
4. New backend endpoint: `GET /system/kg/stats` → shared-KG paper
   + entity counts. Settings page GPU panel gets a new row.
5. Documentation update in
   [`RewardSculptor/docs/knowledge_graph.md`](RewardSculptor/docs/knowledge_graph.md).

**Trade-off:** cross-domain contamination (a manipulation project
sees quadruped papers in the flat paper list). Mitigation: the KG tab
already filters by search string; add a second filter
"environment_tag" pulled from the project's `config.toml.kg` section.
In practice the semantic-search reranker handles this well — unrelated
papers simply don't match the behavior-goal embedding.

**Tests to add:**
- `test_shared_kg_path_default` — no `RS_KG_PATH` + no legacy
  project-local DB → returns shared path.
- `test_shared_kg_path_respects_env_var`.
- `test_shared_kg_path_preserves_legacy_project_local_db`.
- `test_kg_ingest_via_one_project_visible_from_another`.

---

## Architectural decision 2 — prompt-time research

**User's vision (verbatim):** "when I prompt the reward sculptor, I
should be able to indicate that I want it to do research and add to
the knowledge graph … if it can't find any information about a
specific request (i.e. doesn't know physics parameters for how to
model SEAs), it will tell me and it will do the research on its own
and add it."

Two modes:
1. **Explicit opt-in**: NewRunDialog gets a "Expand KG for this
   topic" checkbox. Before the run starts, fire a research pass.
2. **Implicit** (auto-research on gap): diagnose.py's Stage-1
   preliminary run returns a coverage estimate ("I see 3 papers on
   quadruped jumping, 0 on SEA dynamics"). When coverage < threshold
   on a topic derived from the behavior goal, pause + ask user for
   consent to research.

Mode #1 is a ~200-line feature. Mode #2 is another ~200 lines on top
of that.

**New endpoint:** `POST /kg/research` with body
`{topic: str, max_papers: int = 10}`. Backend:

```python
@router.post("/kg/research")
async def research_topic(body: ResearchRequest, request: Request):
    # 1. Ask Claude for arxiv IDs.
    anthropic = anthropic.Anthropic()
    prompt = load_prompt("research_topic")  # new prompt file
    resp = anthropic.messages.parse(
        model="claude-opus-4-7",
        system=prompt,
        messages=[{"role": "user", "content": body.topic}],
        response_format=ResearchResponse,  # pydantic: list[{arxiv_id, title, relevance}]
    )
    # 2. Dedupe against existing KG.
    kg = SculptorKG()
    new_ids = [p.arxiv_id for p in resp.papers if not kg.has_paper(p.arxiv_id)]
    # 3. Fire ingest+extract job (reuse kg_jobs.run_ingest_extract_job).
    job = jobs.submit(
        kind="kg_research",
        fn=run_research_job(topic=body.topic, arxiv_ids=new_ids, kg_path=shared_kg_path),
    )
    return job.to_summary()
```

**New system prompt** at `sculptor/prompts/research_topic.md`:

```
You are a research librarian for an RL reward-engineering system. A
user has asked for literature on a specific topic. Return 5-10 arxiv
IDs that are most directly relevant.

Rules:
- Only arxiv IDs (YYMM.NNNNN). No DOIs, no GitHub, no blog posts.
- Prefer recent (last 5 years) unless the paper is foundational.
- If the topic is outside your training cutoff, say so honestly.
- Return a relevance_score (0-1) per paper + a one-line
  justification that names the specific contribution.

Format: parsed by messages.parse with this pydantic schema:
{papers: list[{arxiv_id: str, title: str, relevance_score: float,
               justification: str}], coverage_note: str}

coverage_note: if you found fewer than 5 solid matches, explain what's
missing and what the user would need to research manually.
```

**UI:**
- KG tab: "Research a topic" button next to Add seeds + Bulk-seed.
- Click: opens a dialog with a `<textarea>` for topic, max-papers
  slider (1-20, default 10), fire button.
- On submit: POSTs to `/kg/research`, gets a job back, toasts
  "Researching 'SEA physics parameters'… ~2-5 min", watches the job.

**Wiring into the sculpt loop** (mode #2 — auto-research):
- Extend `sculptor/diagnose.py::_preliminary_diagnose` to return a
  `topics_needing_research: list[str]` field. Populate from
  Claude's Stage-1 output.
- Extend `sculpt_run` to check this field. If non-empty AND the
  `config.toml` has `[kg].auto_research = true`, pause before Stage-2
  and fire research via the new endpoint (called programmatically,
  not via HTTP). Wait for completion, then re-query KG, then proceed.
- User opt-in via NewRunDialog checkbox: "Let Claude research topics
  it doesn't know".

**Tests:**
- `test_research_endpoint_returns_job` (mocked Claude).
- `test_research_prompt_includes_topic_verbatim`.
- `test_auto_research_triggers_when_coverage_thin` (mock diagnose
  output with `topics_needing_research=["SEA dynamics"]`).

---

## Architectural decision 3 — GPU-appropriate run parameters

**User's complaint:** "it seems like the adjustable parameters are
more applicable to CPU". Correct — the current
[`NewRunDialog`](reward-sculptor-ui/frontend/src/components/NewRunDialog.tsx)
shows:

- Behavior goal (good).
- Iterations (1-100, "~3-5 min on CPU").
- `--no-kg` toggle (good).
- `--dry-run` toggle (good).

It's missing, for GPU/mjlab runs:
- **Training iterations per sculpt cycle** (rsl_rl `--max_iterations`).
  mjlab needs 1000+ for a reasonable signal; default should scale with
  `adapter_config.task_id` (humanoids need more than cartpole).
- **num_envs override** (per-run; default from project config).
- **Device override** (`cuda:0` / `cpu`).
- **num_sculpt_iterations** (the outer loop — how many
  train→diagnose→edit→commit cycles).

Rename: the current "Iterations" field is ambiguous. Make it
**Sculpt iterations (outer)** + add **Training iterations per cycle
(inner)** with sensible defaults per adapter:

| Adapter | Sculpt iters default | Training iters / cycle default | Budget per cycle |
|---------|:---:|:---:|---|
| gym_sb3 (Hopper, Ant, …) | 10-20 | `steps_per_iter=50000` | ~3-5 min CPU |
| mjlab (Cartpole) | 10-20 | 500 | ~30 s GPU |
| mjlab (Go1 velocity) | 10-15 | 1000 | ~90 s GPU |
| mjlab (G1 velocity / tracking) | 5-10 | 1500 | ~3 min GPU |

**UI restructure** (est. 45 min):

```tsx
// NewRunDialog.tsx
<Tabs defaultValue="basic">
  <TabsList>
    <TabsTrigger value="basic">Basic</TabsTrigger>
    <TabsTrigger value="advanced">Advanced</TabsTrigger>
  </TabsList>
  <TabsContent value="basic">
    {/* Behavior goal (hero), --dry-run toggle, that's it. */}
  </TabsContent>
  <TabsContent value="advanced">
    {/* Sculpt iterations (outer), training iters/cycle (inner),
        num_envs override, device override, --no-kg, --expand-kg */}
  </TabsContent>
</Tabs>
```

When the project is mjlab-adapter, the Advanced tab shows GPU-flavored
controls; when gym_sb3, it shows CPU-flavored. The `useProject` hook
already exposes `adapter_class` so the dialog can pick the right
shape.

**Backend:** `POST /projects/{slug}/runs` already accepts
`iterations`, `no_kg`, `dry_run`. Extend with `training_iterations`
(new), `num_envs_override` (new), `device_override` (new),
`expand_kg` (new — triggers pre-run research).

---

## MuJoCo physics editor (deferred; design complete)

Design sketch from
[`docs/knowledge_graph.md`](RewardSculptor/docs/knowledge_graph.md)
§ "MuJoCo physics editor — also deferred":

- New **Physics** tab on ProjectDetail, between Rewards + Runs.
- Parses the project's MJCF via `mujoco.MjSpec`. Two sources:
  - Library project: resolved from `menagerie_package` via
    `robot_descriptions`.
  - Upload project: `<project>/uploads/robot/<file>.xml`.
- Editable fields grouped by section:
  - `<option>`: timestep (default 0.002 s for mjlab), gravity,
    integrator (`euler` / `implicit` / `semi-implicit`), solver
    (`CG` / `Newton`).
  - `<worldbody>` joints: per-joint `damping`, `frictionloss`,
    `stiffness`, `armature`.
  - `<actuator>`: per-actuator `forcerange`, `gear`, `kv`, `kp`.
  - `<geom>`: `friction` (sliding / torsional / rolling),
    `mass` / `density`.
- Write-back via `MjSpec.to_xml()`. Each edit = git commit in the
  project's repo (`physics: bump knee_joint_damping 0.01→0.05`).
- Validate after every edit: re-parse with `MjModel.from_xml_string`;
  refuse the write + surface the MuJoCo error if it fails.
- **For SEA-specific modeling** (Sam's AME456 use case):
  - Parallel-elastic spring via `<tendon>` or custom `<spring>` geom.
  - Serial-elastic model via motor rotor + spring + joint coupling.
  - Reference: MuJoCo Discussion #226 (cited in AME456
    RESEARCH_LOG.md: `github.com/google-deepmind/mujoco/discussions/226`).
  - The KG's `2209.07171` (Raffin Learning to Exploit Elastic
    Actuators) + `2301.03509` (ANYmal PEA) are canonical references
    once bulk-seeded.

**Estimated effort:** 8-12 hours. Moderate complexity mostly in the
MJCF-form UI (60+ editable fields grouped sensibly). MuJoCo's MjSpec
round-trip is well-tested.

**Priority call:** do this AFTER M7's bugs + shared KG + prompt-time
research. User will want it when iterating on the SEA parameters —
that's the whole point of the AME456 capstone.

---

## Other improvements noticed this session

Things that would move the UX needle, roughly in priority order:

1. **Cancel button on Active Jobs.** When a KG extract or sculpt run
   is stuck, the user has no way to kill it from the UI — currently
   requires `curl -X POST /jobs/{id}/stop`. Add a stop button to
   [`ActiveJobsIndicator`](reward-sculptor-ui/frontend/src/components/ActiveJobsIndicator.tsx).

2. **Per-paper progress in the KG bulk-seed flow.** Today the UI just
   shows "extracting entities via Claude". Extend
   [`kg_jobs.run_ingest_extract_job`](reward-sculptor-ui/backend/services/kg_jobs.py)
   to update `job.message` per-paper (`"extracting 12 / 50: Walk These
   Ways"`). Uses the existing job progress field.

3. **Run-viewer GPU tab.** M4 §4 deferral. WebSocket `gpu_stats`
   events every 2 s during training, charted in the Runs tab. Useful
   for debugging OOM + throughput.

4. **Reward diff view.** When Claude writes v2.py from v1.py, surface
   the diff inline on the Rewards tab with the citing paper highlighted.
   Currently the user has to click back and forth.

5. **Project settings drawer.** A gear icon on the ProjectDetail
   header that opens a drawer with: environment_tag, adapter config,
   iteration defaults, KG auto-research toggle, danger-zone (delete
   project). Centralises everything currently scattered across
   config.toml.

6. **KG graph view — click-through.** The pyvis graph node tooltips
   already show description/authors. Make clicking a node open the
   PaperDetailModal in a side pane instead of a new modal.

7. **"Why this edit?" expandable.** On the Rewards tab, per-version
   row, expand to show the diagnosis.json that triggered the edit
   (failure modes, citations, rationale). Already captured on disk;
   just needs a UI surface.

---

## M7 task list (prioritised, concrete)

Execute in order. Log each to CONTEXT.md as you go.

### P0 — unblock mjlab sculpt runs (critical, user-facing bug)

**User has repro'd this bug twice — once with an empty KG, once with
the KG fully seeded.** The KG state is orthogonal. Don't touch KG
code for this fix.

- [ ] **Run the diagnostic commands in the "Known bug" section first.**
  Confirm the hypothesis (v0.py missing `compute_reward_batched`)
  before coding. If the hypothesis fails, re-read the backend +
  subprocess stderr for the actual traceback and update this plan
  before proceeding.
- [ ] Edit `sculpt_init` to accept adapter name + write
  adapter-appropriate `rewards/v0.py`. For mjlab, include
  `compute_reward_batched` + `supports_batched: True`. Reference
  template: `MJLAB_PIVOT_DESIGN.md` §1.4 state schema + §2.2 contract.
- [ ] Extend the UI's project-create route
  ([`backend/routes/projects.py::create_project`](reward-sculptor-ui/backend/routes/projects.py))
  to pass the adapter name through to `scaffold_project`.
- [ ] **Migration helper for already-scaffolded projects**: endpoint
  `POST /projects/{slug}/rewards/regenerate-template` that rewrites
  `rewards/v0.py` using the adapter-aware template. UI surfaces this
  as a "Regenerate reward template" button on the Rewards tab when
  the current v0.py is missing the required entry points for the
  project's adapter. Critical so Sam doesn't have to delete + recreate
  his existing G1 project to adopt the fix.
- [ ] Extend `run_manager.py` to classify subprocess failures: parse
  stderr for `AttributeError.*compute_reward_batched`, surface as
  `/problems/reward-contract-mismatch` with actionable remediation
  ("rewards/v0.py is scalar-only; mjlab needs batched. Click
  'Regenerate reward template' on the Rewards tab."). Mirror the
  pattern in [`cuda_errors.py`](reward-sculptor-ui/backend/services/cuda_errors.py).
- [ ] Test: `test_sculpt_init_mjlab_writes_batched_template`.
- [ ] Test: `test_regenerate_template_endpoint_rewrites_v0`.
- [ ] Test: `test_run_manager_parses_reward_contract_error`.
- [ ] Verify: (a) create a fresh mjlab G1 project from the UI,
  launch a 3-iter `--dry-run`, confirm 3 iter_* dirs appear + sculpt
  run completes without error. (b) Take Sam's EXISTING broken G1
  project, click Regenerate, launch a 3-iter dry-run, confirm same
  result.

### P1 — shared knowledge graph

- [ ] Change default KG path to
  `~/.local/share/sculptor/kg/graph.db` in both
  `sculptor/kg/store.py::SculptorKG` AND
  `backend/services/kg_store.project_kg_db_path`.
- [ ] Respect legacy per-project DB when it exists (don't silently
  migrate).
- [ ] `RS_KG_PATH` env var override.
- [ ] New endpoint `GET /system/kg/stats` returning shared-KG counts.
- [ ] Settings page: new "Knowledge graph" card showing paper count,
  entity counts, last-ingest timestamp, path.
- [ ] Tests as listed in "Architectural decision 1".
- [ ] Update [`docs/knowledge_graph.md`](RewardSculptor/docs/knowledge_graph.md)
  with the shared-default documented + migration note.

### P2 — prompt-time research

- [ ] `sculptor/prompts/research_topic.md` (new prompt file — the
  research librarian system prompt; inline draft in "Architectural
  decision 2").
- [ ] `sculptor/kg/research.py` (new module): `research_topic(topic:
  str, max_papers: int) -> list[PaperRef]` using Claude Opus 4.7 +
  `messages.parse`. Dedupe against existing KG via `has_paper`.
- [ ] `POST /kg/research` endpoint in
  [`backend/routes/kg.py`](reward-sculptor-ui/backend/routes/kg.py).
  Body `{topic, max_papers}`. Fires a background job, returns handle.
- [ ] New "Research a topic" button on the KG tab + dialog.
- [ ] `NewRunDialog`: add "Let Claude research gaps" checkbox
  (opt-in).
- [ ] Extend `sculptor/diagnose.py::_preliminary_diagnose` to return
  `topics_needing_research: list[str]` (Claude includes this in the
  Stage-1 output).
- [ ] In `sculpt_run`: if `auto_research=True` in config AND Stage-1
  returned non-empty topics, call research_topic for each before
  Stage-2.
- [ ] Tests listed under "Architectural decision 2".
- [ ] Update `docs/knowledge_graph.md`.

### P3 — GPU-appropriate run parameters

- [ ] Restructure `NewRunDialog` into Basic/Advanced tabs.
- [ ] Advanced fields: sculpt iterations (outer), training iters per
  cycle (inner), num_envs override, device override, no_kg,
  expand_kg (from P2).
- [ ] Per-adapter defaults:
  - gym_sb3: 20 sculpt, 50000 training steps/cycle.
  - mjlab cartpole: 15 sculpt, 500 training iters/cycle.
  - mjlab Go1: 12 sculpt, 1000 training iters/cycle.
  - mjlab G1: 8 sculpt, 1500 training iters/cycle.
- [ ] Extend `POST /projects/{slug}/runs` to accept the new fields
  and wire them into `sculpt_run`.
- [ ] Tests: test the defaults, test the form renders per-adapter
  field set, test backend accepts new fields.

### P4 — job cancellation UX

- [ ] Add stop button to
  [`ActiveJobsIndicator`](reward-sculptor-ui/frontend/src/components/ActiveJobsIndicator.tsx).
  Calls existing `POST /jobs/{id}/stop`.
- [ ] Per-paper progress in KG extract job (`job.message`
  "extracting 12 / 50: Walk These Ways").

### P5 — nice-to-haves (pick whichever the user asks for next)

- Run-viewer GPU tab with live gpu_stats WS events.
- Reward diff view (per-version on Rewards tab).
- Project settings drawer.
- KG graph click-through.
- "Why this edit?" expandable on Rewards rows.

### P6 — MuJoCo physics editor

Separate milestone. Don't start until M7 P0-P4 ship. ~10 hours.

---

## Verification gate per milestone

Each P-block ships green or not at all. Required for P-completion:

1. All backend tests green (`cd ~/projects/reward-sculptor-ui && uv run
   pytest backend/tests/ -q`). Target: 140+ tests by end of M7.
2. Frontend typecheck clean (`cd ~/projects/reward-sculptor-ui/frontend
   && PATH=$HOME/.local/share/pnpm:$PATH node_modules/.bin/tsc --noEmit`).
3. Sculptor tests unchanged (`cd ~/projects/RewardSculptor && uv run
   pytest tests/ -q --ignore=tests/test_mjlab_gpu.py`). Target:
   76 passing, 1 JAX-skipped.
4. Manual smoke per P-block:
   - **P0:** create mjlab G1 project → launch dry-run → 3 iters
     complete.
   - **P1:** seed KG from one project → open a second project → its
     KG tab shows the same papers.
   - **P2:** type "SEA physics parameters" in research dialog → 5+
     arxiv IDs fetched + extracted.
   - **P3:** mjlab run dialog shows num_envs + device fields; gym
     dialog doesn't.
   - **P4:** click stop on a running job → job state flips to
     "stopped" within 2 s.

---

## Working-rules reminder (user preferences)

- Answer questions in 2-3 sentences when they're exploratory.
- Don't implement until the user confirms the approach.
- Log every meaningful file change to `CONTEXT.md` with
  What / Why / How / Verified.
- Confirm before destructive actions.
- File:line over prose.
- Use Claude-style design (shadcn).
- No emojis unless asked.

---

## Quick commands

```bash
# Start the UI (user's main loop):
cd ~/projects/reward-sculptor-ui && ./run.sh
# Open http://localhost:5173 in Windows browser.

# Test sculptor library (no GPU):
cd ~/projects/RewardSculptor && uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py
# Expect: 76 passed, 1 skipped.

# Test backend:
cd ~/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q
# Expect: 129 passed.

# Frontend typecheck (node lives in ~/.local/share/pnpm/):
cd ~/projects/reward-sculptor-ui/frontend && PATH=$HOME/.local/share/pnpm:$PATH node_modules/.bin/tsc --noEmit
# Expect: empty output.

# GPU smoke (cached fixture):
cd ~/projects/RewardSculptor && uv run pytest -m gpu -q
# Expect: 2 passed in ~3 s.

# Global-seed a project's KG from CLI (50 papers):
cd ~/.local/share/reward-sculptor/projects/<slug>
uv run --project ~/projects/RewardSculptor sculpt kg ingest \
    ~/projects/RewardSculptor/examples/kg_seeds_global.yml
uv run --project ~/projects/RewardSculptor sculpt kg extract --all

# Or one-click from UI: Project → Knowledge Graph tab → Bulk-seed library button.
```

---

*End of plan. Go in priority order. Log as you go. Ask before risky
changes.*

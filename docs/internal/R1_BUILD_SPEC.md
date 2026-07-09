# R1 Build Spec — Reference Library + Retrieval (LOCKED)

2026-07-09. Implements phase R1 of `REFERENCE_TRAJECTORY_PLAN.md`. Every decision
below is LOCKED by recon (dataset recon + code recon, both verified against the
real datasets and real code). Workers: build EXACTLY this; report (do not
improvise) if reality contradicts the spec.

## Verified ground truth (do not re-derive)

- **Datasets (both ungated, plain `curl -sL https://huggingface.co/datasets/{repo}/resolve/main/{path}`):**
  - `lvhaidong/LAFAN1_Retargeting_Dataset` `g1/*.csv` — 40 clips, 90.5 MB total,
    30 fps, no header; 36 cols/row = root pos xyz (0:3) + root quat **xyzw** (3:7)
    + 29 joint angles rad (7:36). Get-up clips: `fallAndGetUp1_subject1.csv`,
    `fallAndGetUp1_subject4.csv`, `fallAndGetUp1_subject5.csv`,
    `fallAndGetUp2_subject2.csv`, `fallAndGetUp2_subject3.csv`,
    `fallAndGetUp3_subject1.csv` (long multi-repetition recordings, e.g. 5047
    frames; root z verified 0.05→0.80 m). Also dance1-2, walk1-4, run1-2,
    sprint1, jumps1, fight1, fightAndSports1, aiming*, ground*, obstacles*, push*.
    License: **CC BY-NC-ND 4.0** (record in provenance).
  - `fleaven/Retargeted_AMASS_for_robotics` `g1/...` — one `.npy` per clip,
    shape (T, 36) float64, SAME column layout (root xyz + quat xyzw + 29 joints);
    fps encoded in filename `..._poses_{fps}_jpos.npy` (120 or 250; NOT uniform).
    Path = `g1/{AMASS_set}/{subject}/{motion}_poses_{fps}_jpos.npy`. License
    cc-by-4.0 (+ per-subset attribution in `g1/license.txt`). v1 slice only:
    `g1/ACCAD/` clips matching `*lie to crouch*`, `*crouch to lie*`,
    `*stand to walk*`, plus a handful of `g1/Eyes_Japan_Dataset/*/sitdown*` /
    `*standup*` clips.
- **Canonical G1 joint order** (both datasets follow it, per their READMEs — but
  VERIFY, don't trust): `sculptor/eval/robot_manifest.py:25-37` tuple `G1_29`,
  29 names ending `_joint` (left_hip_pitch_joint … right_wrist_yaw_joint).
- **Clip container**: `sculptor/reference.py` — dict → `np.savez_compressed`;
  `validate_clip` is the only entry point; optional keys checked only-if-present;
  `meta` dict rides as JSON bytes under `meta_json`; `_with_velocity` backfills
  `root_vel_z`. `load_clip` re-validates.
- **Joint resolver**: `sculptor/eval/joint_resolver.py:230` —
  `resolve_joint_roles(names, roles, *, lenient=False)`; also
  `assert_name_axis_contract(names, n_joints)`.
- **Env-root pattern**: copy `sculptor/archive.py:111` `saved_root()` exactly →
  `references_root()`: `RS_REFERENCE_ROOT` override else
  `~/.local/share/reward-sculptor/references`.
- **LLM registry**: `sculptor/llm.py` `ROLE_DEFAULTS` dict + `model_for(role)`.
  There is NO offline stub anywhere — every call site hard-fails without a key.
- **CLI**: typer sub-app pattern (`kg_app`/`eval_app` in `sculptor/cli.py`),
  lazy imports inside command bodies.
- **Stage attachment**: extend the `Stage` dataclass (`sculptor/mission.py`, same
  pattern as `steering_metric`; `to_dict`=asdict, `from_dict` filters unknown —
  back-compat automatic). `needs_reference_rsi: bool` already exists on Stage.
- **Preview primitives**: backend `services/preview_renderer.py` (offscreen
  `mujoco.Renderer`, `_build_camera` bbox-fit, PNG encode) and the keyframe-strip
  pattern in `_mjlab_runner.py:1848` (12 frames via np.linspace).
- **Backend router precedent**: `routes/saved.py` (top-level disk-truth library:
  regex id guard, slim-row listing, FileResponse with resolve/relative_to
  traversal guard, media-type map).
- Backend is currently DOWN (nothing to disturb); tests run via TestClient.
- Python for sculptor work: `~/projects/RewardSculptor/.venv` (numpy, torch,
  mujoco available). Backend venv: `~/projects/reward-sculptor-ui/.venv`.

## Locked design decisions

1. **Quaternion convention**: clips store `root_quat_wxyz (T,4)` — MuJoCo order,
   named in the key. Ingest converts dataset xyzw→wxyz. Any worker touching quats
   must unit-test the conversion with a known rotation (e.g. 90° about z).
2. **New optional clip keys** (flat, npz-friendly): `root_pos_xy (T,2)`,
   `root_quat_wxyz (T,4)` (unit-norm within 1e-3 when present), `joint_vel (T,J)`
   (finite-diff backfill like `_with_velocity` when `joint_pos` present),
   `contact_left_foot (T,)`, `contact_right_foot (T,)` (float 0/1; R1 ingest does
   NOT infer contacts — leave absent; the keys are schema-reserved only).
   `validate_clip` gains only-if-present shape/finite checks for each.
3. **Provenance**: REQUIRED for library clips. Stored twice, same content:
   (a) inside `meta["provenance"]` (travels with the npz), (b) as
   `provenance.json` beside the npz (cheap backend reads). Schema:
   ```json
   {"schema": 1, "clip_id": "...", "robot": "g1",
    "source": {"kind": "hf_dataset", "repo": "...", "path": "...", "url": "..."},
    "license": "CC BY-NC-ND 4.0", "attribution": "...",
    "retarget": {"tool": "dataset-provided", "notes": "..."},
    "tier": "K", "fps_source": 30.0,
    "parent_clip_id": null, "frame_range": null,
    "joint_mapping": {"identity": true} ,
    "content_sha256": "<hash of clip.npz bytes>",
    "labels": ["fall", "get", "up", "subject1"], "text": "fall and get up (subject 1)",
    "qc": {"duration_s": 168.2, "root_z_range": [0.05, 0.80], "checks": ["..."]},
    "ingested_at": "ISO8601"}
   ```
4. **Library layout** (`references_root()`):
   `<robot>/<clip_id>/clip.npz + provenance.json + preview.png` and a root
   `index.jsonl` (one row per clip: `clip_id, robot, text, labels, tier, license,
   n_frames, fps, duration_s, root_z_range, has_preview`). `clip_id` =
   slugified source name (+ `--segNN` suffix for segments), regex
   `^[a-z0-9][a-z0-9_-]{0,95}$`. Index is REBUILDABLE from provenance.json files
   (`sculpt refs index` rescans) — index is a cache, provenance is truth.
5. **Ingest QC gates (hard-fail the clip, never the batch; failures logged to
   `index_rejects.jsonl` with reasons)**:
   - column count exactly 36; finite; T ≥ 30 frames;
   - joint mapping: dataset joint order documented-as-G1_29 must be VERIFIED by
     `assert_name_axis_contract` + `resolve_joint_roles(G1_29, canonical roles)`;
     joint_names stored = `G1_29` verbatim; if a dataset README order differs
     from G1_29, build an explicit index-mapping table — never reorder silently;
   - plausibility: |joint angles| ≤ 2π; root z in (0, 2.5) m; per-frame joint
     delta ≤ 0.5 rad @30fps-equivalent (flag, don't fail, if exceeded — record in qc);
   - **motion-class content checks** (keyed by label heuristics):
     `fall`/`getup` clips must have min(root_z) < 0.35 AND max(root_z) > 0.6;
     `walk/run/sprint` must have horizontal displacement > 1 m. Failing a class
     check = reject with reason (catches mislabeled data).
6. **Segmentation** (`refs/segment.py`): deterministic root_z hysteresis for
   multi-rep clips: standing when z > 0.60 sustained ≥ 1.0 s; down when z < 0.35
   sustained ≥ 0.5 s; a segment = down-interval → next sustained-standing, padded
   ±0.5 s, min length 2 s. Each segment saved as derived clip
   (`parent_clip_id`, `frame_range` in provenance; labels inherited + `segment`).
   Applied at ingest to clips whose labels match fall/getup. Unit-test on a
   synthetic z-profile AND assert ≥ 2 segments come out of
   `fallAndGetUp1_subject1` at data-run time.
7. **Retrieval** (`refs/retrieve.py`) — two layers:
   - **Deterministic (always on, zero API)**: normalize labels at ingest
     (camelCase/underscore/digit split → tokens) + query tokens; score = weighted
     token overlap (IDF-ish weighting by token rarity across index) + a small
     built-in SYNONYM map (`get up`≈`getup`≈`stand up`≈`rise`≈`fall and get up`;
     `jump`≈`leap`≈`hop`; `walk`≈`locomotion`≈`gait`; `run`≈`sprint`≈`jog`;
     `lie`≈`lying`≈`supine`≈`ground`; `crouch`≈`squat`; `kick`; `dance`;
     `fight`≈`boxing`≈`sports`; `push`≈`shove`; extensible dict). MUST rank the
     fallAndGetUp segments top for the LITERAL acceptance query
     **"get up off the ground"** with no LLM — this is a unit test.
   - **LLM rerank (optional)**: `ROLE_DEFAULTS["reference_rerank"] =
     "claude-sonnet-5"` in llm.py; rerank top-20 with goal_text; returns ranked
     ids + `match_confidence` (0-1) + one-line reason. Wrapped in
     try/except at CLIENT-INIT and CALL level → on ANY failure (no key, network,
     parse) fall back to deterministic scores with `match_confidence=None` and a
     `rerank: "deterministic-fallback"` flag in the response. NEVER raises.
   - API: `search(query: str, robot: str, k: int = 10, use_llm: bool = True) ->
     list[RefMatch]` where RefMatch = dataclass(clip_id, text, score,
     match_confidence, reason, tier, license, n_frames, fps, duration_s).
8. **Preview** (`refs/preview.py`, sculptor-side): keyframe-strip PNG (12 frames,
   np.linspace over T) rendered offscreen with mujoco: resolve the G1 MJCF from
   the mjlab package (find it programmatically; if resolution fails → skip
   preview, log, clip still valid). Set root free-joint qpos from
   root_pos_xy+root_pos_z+root_quat_wxyz; set joint angles BY NAME via
   `model.joint(name).qposadr` (NEVER by index). Strip = horizontal concat,
   ~180 px per tile. Saved as `preview.png` per clip. Must run headless on WSL2
   (EGL/osmesa — copy whatever `preview_renderer.py`/mjlab already relies on;
   if a GL context can't be created in the sculptor venv, fall back to calling
   ingest with previews disabled and note it — preview generation must never
   block ingest).
9. **CLI** (`refs_app` sub-app of sculpt): `sculpt refs ingest --source
   lafan1-g1|fleaven-g1 [--filter GLOB] [--no-preview] [--limit N]`,
   `sculpt refs index` (rebuild index.jsonl from provenance files),
   `sculpt refs search "query" [--robot g1] [--k 10] [--no-llm]`,
   `sculpt refs list`. Downloads via plain HTTPS (urllib/requests — whatever the
   repo already depends on; stream to temp, verify readable, then move in).
   Idempotent: re-ingest skips clips whose content_sha256 already indexed.
10. **Stage fields** (`sculptor/mission.py` Stage): `reference_clip_id:
    Optional[str] = None`, `reference_tier: Optional[str] = None`,
    `reference_match_confidence: Optional[float] = None`. Mirror into backend
    StageSchema + frontend types (same drill as steering_metric).
11. **Backend** (`routes/references.py`, precedent saved.py):
    - `GET /references?robot=&q=&k=` — q present → retrieve.search (use_llm from
      query param `llm=0/1`, default 0 for the UI's as-you-type path); q absent →
      slim index listing.
    - `GET /references/{clip_id}` — provenance.json content + index row.
    - `GET /references/{clip_id}/preview` — FileResponse preview.png (404 clean).
    - `GET /references/{clip_id}/file/clip.npz` — download (traversal-guarded).
    - `POST /projects/{slug}/missions/{ms}/stages/{stage}/reference`
      body `{clip_id}` → validates clip exists + stage exists (mission.json
      stage list — NOT the training dir; see the regenerate-metric bug 8b0bfa3),
      sets the three Stage fields (+tier from provenance), atomic-ish
      save_mission; 409 while mission-scoped jobs live (same guard as
      regenerate). DELETE same path → clears fields.
    - clip_id path guard: the regex from decision 4. Robot inferred from clip_id
      lookup in index (v1 = g1 only; no robot path segment in v1 API).
12. **Frontend**: StageCard (MissionDetailDialog) gains a Reference row/chip after
    StageMetricChip: shows clip text + tier badge when attached; "Pick reference"
    ghost button opens a small dialog: search input (deterministic endpoint),
    result rows (text, duration, score, license), preview.png shown for the
    selected row, Attach/Detach buttons wired to the POST/DELETE. Types+api.ts
    per saved-missions conventions. Keep it modest — this is the v1 approval
    surface (plan §4.3), not a full library page.

## Worker plan (sequential where dependent)

- **W1 (sculptor data layer)**: reference.py schema ext; refs/{__init__,library,
  ingest,segment}.py; CLI ingest/index/list; tests (schema round-trip incl. new
  keys + quat conversion; QC gates incl. motion-class checks with synthetic
  arrays; segmentation on synthetic profile; ingest of a TINY in-repo fixture
  CSV/npy — construct 50-frame fixtures in-test, do NOT download in tests;
  idempotency). Gate: full sculptor pytest.
- **W2 (sculptor consumer layer)**: refs/retrieve.py (+llm.py role);
  refs/preview.py; CLI search; Stage fields; tests (deterministic ranking incl.
  THE acceptance query against a fixture index; LLM-fallback path by
  monkeypatching client init to raise; preview smoke — skip gracefully if no GL;
  Stage round-trip). Gate: full sculptor pytest.
- **W5 (data run, after W1, parallel with W2)**: download LAFAN1 g1 (40 CSVs) +
  fleaven ACCAD slice; run real ingest; verify: index rows ≥ 45, ≥2 getup
  segments from fallAndGetUp1_subject1, rejects file empty-or-explained,
  spot-load 3 clips (validate_clip, root_z ranges), previews rendered (or
  documented GL fallback).
- **W3 (backend, after W2)**: routes/references.py + StageSchema mirror + tests
  (TestClient with tmp references root: listing/search/detail/preview 404s/
  traversal/attach+detach incl. 409-guard + pending-stage-no-training-dir case).
  Gate: full backend pytest.
- **W4 (frontend, after W3 contract confirmed — may draft against this spec in
  parallel)**: types/api/StageCard row/picker dialog. Gates: pnpm typecheck+build.
- **Verify**: fresh-context adversarial verifier per W1/W2/W3 batch (opus for the
  final E2E); E2E = relaunch servers (setsid nohup per project_ui_persistent_launch),
  browser: open a mission stage → Pick reference → search "get up off the
  ground" → fallAndGetUp segment top, preview visible, attach persists in
  mission.json. Orchestrator commits each verified increment.

## Hard rules for all workers
- No subagents. No commits (orchestrator commits). Shell = Git-Bash: use
  `wsl bash <<'EOF' ... EOF`. Verify every Edit landed (grep) — this UNC path
  drops edits silently sometimes.
- Never download datasets inside pytest. Never require an API key in any test.
- If reality contradicts this spec (dataset column mismatch, README joint order
  differs from G1_29, GL unavailable), STOP that item, document exactly what you
  found, and continue with other items — do not improvise schema changes.

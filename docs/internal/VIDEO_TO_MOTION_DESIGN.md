# Video-to-Motion Design Note — GVHMR → GMR (future direction, no implementation)

2026-07-11. Companion to `REFERENCE_TRAJECTORY_PLAN.md` §R4 (`generate.py` reserves
the adapter slot). Scope: design only. Nothing in this note is scheduled; it exists
so the eventual increment starts from decisions, not archaeology.

## Purpose

Complete the reference-acquisition ladder for arbitrary motions:
retrieval (mocap library, shipped R1) → text2motion (MoMask, planned R4) →
**video2motion** (this note). A phone video of a human performing the target
motion becomes a Tier-K reference clip for any supported robot, feeding the
same four consumers (validation positive, calibration ladder, generation
grounding, RSI reset derivation) with zero new consumer formats.

## Pipeline

```
video.mp4
  → GVHMR (monocular HMR)          world-grounded SMPL-X pose sequence
  → floor/scale calibration        put feet on z=0, fix metric scale
  → GMR (existing refs/retarget.py wrapper, R3)   per-robot joint trajectory
  → canonical clip + provenance    Tier K, source.kind="video"
  → (optional, cost-gated) track.py Tier-D certification for steer rights
```

**Why GVHMR** (SIGGRAPH Asia 2024, world-grounded human motion recovery):
it recovers motion in a **gravity-aligned world frame** rather than a
camera-relative one. Every derivation we run keys off absolute root z and
projected gravity (archetype classing, RSI reset derivation, segment QC,
eval-reset verification) — a camera-frame recovery would poison all of it.
GVHMR also produces usable foot grounding, which our kinematic-tier contact
inference needs.

**Why GMR stays the retarget layer**: already wrapped (R3), 18 humanoids
including G1 and Booster T1, joint-role resolution proven for both. GVHMR's
SMPL-X output is exactly GMR's input class — the seam is one file format.

## Integration decisions (locked now, cheap to decide early)

1. **Process isolation.** GVHMR's dependency stack (its own detector/tracker
   checkpoints, pytorch3d-class deps) never imports into sculptor. It runs as
   a pinned standalone venv invoked by subprocess CLI, same isolation rule as
   GMR. Output = SMPL-X `.npz` handoff file.
2. **Provenance.** `source: {kind: "video", path/url, sha256, consent: "user-provided"}`,
   plus `recovery: {tool: "GVHMR", version, checkpoint-hash}` and the
   scale-calibration factor applied. No consent field → library guard refuses.
3. **Trust ceiling.** Per plan §10 unchanged: video-sourced clips are
   Tier K = observe-only grounding/validation; **steer rights require Tier-D
   tracking** (`track.py` feasibility certificate). Video adds no new trust path.
4. **fps.** From the video container (probed, recorded in provenance), not
   assumed. Segment/derivation code already consumes per-clip fps.
5. **QC gates at ingest** (same hard-fail-the-clip discipline as R1):
   - limb-length variance over time ≤ threshold (monocular scale wobble);
   - foot-skate + floor-penetration stats from the calibrated world frame;
   - per-frame joint delta bound (occlusion glitches);
   - motion-class content check keyed by the user's label (reuses R1 rules).
6. **Reset derivation compatibility.** The whole point for get-up-class
   missions: a video of "person stands up from lying" must yield a clip that
   passes the settled-start segment QC (start z<0.35, end z>0.55) and derives
   a lying reset through the SAME `derive_rsi_train_keys` path — no
   video-specific derivation code.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Monocular scale error (root z off by 10-20%) | scale calibration against subject height input or standing-segment anchor; factor recorded in provenance; QC bound |
| Occlusion artifacts (limbs teleporting) | per-frame delta QC; reject segments, not whole clips |
| Plausible-looking but dynamically infeasible poses | unchanged rule: Tier-D tracking mandatory before steer; failure = "infeasible-for-robot" verdict |
| Privacy/consent | user-provided videos only; consent tag required in provenance; no scraping path, ever |
| Dependency rot (GVHMR checkpoints) | pinned env + checkpoint hash in provenance; adapter is optional at runtime |

## Cost envelope

GVHMR inference is single-GPU, seconds-per-frame class on the RTX 5070
laptop — cheap relative to track.py. The expensive step remains Tier-D
tracking, which is already cost-gated by standing agreement. Video ingestion
itself needs no new spend gate.

## Non-goals

- No implementation in this sprint.
- Not a motion-editing tool: one video → one clip; blending/compositing stays out.
- MoMask (text2motion) is a sibling adapter with its own note when scheduled.

## Acceptance sketch (for whenever this is scheduled)

A phone video of a person lying supine then standing up →
`sculpt refs ingest --source video --file getup.mp4 --label "get up from lying"`
→ Tier-K clip that (a) passes segment QC, (b) retrieves for "get up off the
ground", (c) derives a lying reset with `derive_rsi_train_keys`, (d) grounds a
metric that passes the reference gates on the standing-test mission — with no
code changes outside `refs/generate.py` + the ingest CLI.

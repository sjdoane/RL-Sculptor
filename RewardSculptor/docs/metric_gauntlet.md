# Metric gauntlet and blind human-anchor studies

RewardSculptor's generated metrics and trust gates are hypotheses until they
are calibrated against evidence outside the optimizer. The metric gauntlet
builds a blinded pairwise study from archived rollout videos and analyzes
where an evaluator agrees with humans, where it rewards known gaming, and
where reliability changes across tasks, robots, embodiments, and motion
families.

The tooling is embodiment-agnostic. Humanoids, quadrupeds, arms, grippers,
mobile manipulators, and future adapters use the same schema. Robot-specific
semantics live in the dataset labels and task rubric, not in the analysis
code.

## 1. Prepare the private source manifest

Use one item per distinct rollout video. `competence_rank` is an ordinal
dataset label: larger means more task-competent. It determines the expected
preference used to test both the evaluator and the human-label protocol. It is
not silently treated as human authority—every discrepancy remains visible in
the pair-level output.

Items are compared only within their `comparison_group`. A group must describe
one coherent question and therefore cannot mix `task_id` or `task_prompt`.
Behavior class names are open-ended; use specific exploit families such as
`metric_gaming_contact_spam`, `proxy_only_motion`, or `early_termination`, not
only a generic `failure` label.

```json
{
  "schema_version": 1,
  "study_id": "arm-push-metric-anchor-v1",
  "rubric": "Prefer reliable object-to-region completion, then stability and collision avoidance. Abstain if the video does not show enough evidence.",
  "evaluator": {
    "evaluator_id": "generated_metric_arm_push",
    "evaluator_version": "sha256:0123456789abcdef",
    "score_semantics": "higher means predicted task competence",
    "higher_is_better": true
  },
  "items": [
    {
      "item_id": "push_heldout_001",
      "comparison_group": "arm_push_red_block_v1",
      "task_id": "arm_push_red_block",
      "task_prompt": "Push the red block fully into the marked region without knocking it over.",
      "robot_id": "franka_panda",
      "embodiment_family": "robot_arm_with_gripper",
      "motion_family": "nonprehensile_push",
      "behavior_class": "competent",
      "competence_rank": 3,
      "artifact_path": "rollouts/push_heldout_001.mp4",
      "evaluator_score": 0.82
    },
    {
      "item_id": "push_proxy_007",
      "comparison_group": "arm_push_red_block_v1",
      "task_id": "arm_push_red_block",
      "task_prompt": "Push the red block fully into the marked region without knocking it over.",
      "robot_id": "franka_panda",
      "embodiment_family": "robot_arm_with_gripper",
      "motion_family": "nonprehensile_push",
      "behavior_class": "metric_gaming_end_effector_motion",
      "competence_rank": 0,
      "artifact_path": "rollouts/push_proxy_007.mp4",
      "evaluator_score": 0.91
    }
  ]
}
```

Every `item_id` and video must be unique. Byte-identical videos are rejected
because counting the same evidence twice creates pseudo-replication. Evaluator
scores may be omitted, in which case human-label statistics are still
computed and evaluator comparisons are reported as not estimable. If any
score is present, the evaluator identity, immutable version/hash, score
semantics, and direction are required so the analysis cannot silently change
what a larger score means.

## 2. Freeze and build the study

```bash
sculpt eval gauntlet build source_manifest.json \
  --out studies/arm-push-v1 \
  --seed 20260719 \
  --forms 2 \
  --max-pairs-per-group 50 \
  --reliability-repeats 5 \
  --evaluator-tie-band 0.03
```

The builder:

- generates only unequal-rank comparisons;
- balances sampling across behavior-class pair strata so common failures do
  not erase rare exploit classes;
- randomizes A/B placement in Form A and exactly reverses it in Form B;
- optionally inserts opaque repeated comparisons for rater self-consistency;
- remuxes videos through ffmpeg, removes audio, metadata, chapters, subtitles,
  and source timestamps;
- replaces item and source names with opaque asset identifiers;
- refuses to overwrite a non-empty directory, preventing silent
  re-randomization;
- hashes the source manifest, public packets, private key, source videos, and
  sanitized public assets.

The output has this shape:

```text
study/
  assets/form_A/...
  assets/form_B/...
  study_packet_form_A.json
  study_packet_form_B.json
  response_template_form_A.jsonl
  response_template_form_B.jsonl
  study_key.json
  build_summary.json
```

Keep `study_key.json`, `build_summary.json`, the original manifest, and the
other form private while labeling is in progress. Give each rater exactly one
form. Public packets contain the task prompt, rubric, opaque media paths, and
allowed choices; they omit condition, iteration, reward source, evaluator
score, competence label, and expected answer.

Do not “fix” labels or regenerate a packet after inspecting responses. Create
a new study ID and output directory for a revised design.

## 3. Collect responses

Each non-empty JSONL line is one judgment:

```json
{"study_id":"arm-push-metric-anchor-v1","form_id":"A","pair_id":"p_...","rater_id":"rater_014","choice":"B","confidence":0.8,"duration_seconds":13.2,"notes":"Block crosses the boundary only in B."}
```

`choice` is one of `A`, `B`, `tie`, or `abstain`. `tie` means the behaviors are
genuinely indistinguishable under the task rubric. `abstain` means the
evidence is inadequate, ambiguous, occluded, or technically invalid. These
are scientifically different and are analyzed separately.

Rater IDs should be pseudonymous and stable. A rater cannot appear in both
forms, and duplicate `(rater, form, pair)` responses are rejected. Consent,
demographic data, compensation, and institutional review requirements remain
the research team's responsibility and should not be stored in the public
packet.

## 4. Analyze the frozen responses

Concatenate completed JSONL rows from all raters, verify that collection is
closed, then run:

```bash
sculpt eval gauntlet analyze \
  studies/arm-push-v1/study_key.json \
  studies/arm-push-v1/responses.jsonl \
  --out studies/arm-push-v1/analysis
```

The analysis produces `gauntlet_analysis.json` and
`gauntlet_analysis.md`, both bound to the study-key and response-file hashes.
Primary outputs include:

- human individual and pair-majority accuracy against expected ranks;
- evaluator accuracy against expected ranks and blind human majorities;
- false-competence rate by the lower-ranked behavior/exploit class;
- human pair consensus and Krippendorff's nominal alpha;
- repeated-pair self-consistency;
- displayed-A choice rates by counterbalanced form with Wilson intervals;
- task, robot, embodiment-family, and motion-family breakdowns;
- pair-level decisions, including ties and inconclusive human majorities.

Bootstrap confidence intervals use primary comparison pairs as the analysis
units. Repeated rater responses are not misrepresented as independent RL
seeds or independent behaviors. Publish the source-manifest schema, frozen
pairing design, key hash, response hash, exclusions, and pair-level results
alongside aggregate claims.

## 5. Recommended first dataset

Start with a compact but adversarial dataset before scaling:

- at least one humanoid, one quadruped, and one robot arm;
- one manipulation task where a gripper is required and one where it is not;
- competent, partial, still, fallen, unstable, early-terminated, proxy-only,
  and task-specific metric-gaming rollouts;
- held-out clips never used for metric generation, trust calibration, reward
  selection, or iteration stopping;
- multiple seeds and worlds within each important class;
- p10, median, and p90 episodes rather than only best-return videos.

No single gauntlet certifies an evaluator as universally safe. Its claim is
bounded by the recorded robots, tasks, worlds, exploit families, and attack
budget. New counterexamples should become new dataset items and trigger a new
versioned study rather than disappearing into an anecdotal bug fix.

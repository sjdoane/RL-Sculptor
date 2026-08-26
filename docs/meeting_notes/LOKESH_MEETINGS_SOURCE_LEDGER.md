# Lokesh meeting source and decision ledger

This file preserves provenance for the living research synthesis without
committing the noisy verbatim transcript or turning uncertain ASR into fact.
The canonical interpretation is `docs/GUIDING_RESEARCH_CONTEXT.md`.

## Source hierarchy

1. Direct decisions confirmed with Lokesh.
2. Public primary papers, official project pages, and released code/docs.
3. Transcript reconstruction, explicitly marked when uncertain.
4. Working research proposals, which remain hypotheses until agreed.

## Meeting 1 — date not recorded in the repository

Durable outcomes:

- Move RewardSculptor from an impressive hobby/tool demonstration toward a
  precise research contribution.
- Use motion priors and pretrained tracking rather than asking a scalar reward
  to express an entire time-varying whole-body behavior.
- Treat modes/stages, objective evaluation, variations/domain randomization,
  and environment awareness as central structure.
- Preserve the objective metric/trust loop as a likely RewardSculptor
  differentiator.
- Explore a formal lab collaboration, directed-research credit, and eventual
  hardware work, while leaving hours, ownership, authorship, IP, and schedule
  unresolved.
- Lokesh encouraged Sam to develop “technical trench time” and formulate exact
  failure mechanisms instead of describing tasks as merely hard.

Historical interpretation: `docs/RESEARCH_DIRECTION.md`. It is not current
authority where the second meeting or code audit supersedes it.

## Meeting 2 — transcript supplied 2026-08-24

Local source at review time:

`C:\Users\SamJD\.codex\attachments\1f682538-f7c2-4d7f-b3e8-15f50c10700d\pasted-text.txt`

SHA-256:

`b6f6bae76656039f1645515f108998ddd0dda28a6ce9e9d124c74e9713edb2c6`

The raw transcript is intentionally not committed because it is noisy,
contains informal personal conversation, and is not itself an authoritative
technical source.

Confirmed outcomes:

- The architecture is split into behavior generation and policy training.
- The input contract requires a reference behavior dataset plus a rough task
  reward.
- SONIC is the pretrained low-level controller foundation, not a VLM and not a
  separate controller per mode.
- Lokesh's visual behavior-adaptation work starts from SONIC but is an
  unpublished project distinct from the public SONIC paper.
- The lab/Sachin behavior-generation agent is a separate upstream branch; its
  publication/release identity was not established in the transcript.
- RewardSculptor fits most naturally as an agentic policy-training harness;
  reward generation is one possible component, not the entire role.
- Static references fail when a task leaves their local neighborhood. The box
  slipping away motivates explicit reacquisition, transition, and recovery
  coverage.
- An OGMP-like structure should be re-instantiated around reference datasets,
  but the desired system still uses one low-level controller rather than one
  controller per mode.
- The immediate deliverable is a precise current-system boundary, paper
  reading, architecture reconstruction, missing-gap analysis, and proposal
  draft—not a new GPU or hardware result.
- Sam requested AME 590, preferably two credits; approval and deadline remain
  unconfirmed.

High-confidence ASR repairs:

| Transcript token | Intended term |
|---|---|
| Sony | SONIC / GEAR-SONIC |
| tablara rasa | tabula rasa |
| blood/network/task word function | reward/task reward function |
| next open prediction | next-token prediction |
| portboard emotion tracking | whole-body motion tracking |
| BLM / VNM | VLM |
| agent hardness | agent harness |
| RS carpet / RS scapter | RewardSculptor / RL Sculptor |
| close move controller | closed-loop controller |
| question fiction | coefficient of friction |
| GCap | Google Calendar invitation |

Low-confidence garbled paper/field/workspace names were not reconstructed.

## 2026-08-24 literature and code audit

Audited from repository commit
`f6e7035964cba6324646aa1b2a12738037a620c4` plus the then-current dirty tree.
Unrelated dirty same-lane acceptance work was not treated as a stable
capability and was not modified.

The audit established:

- public SONIC, its VLA/token interface, Lokesh's unpublished visual system,
  and the lab behavior-generation agent are four separate components;
- current RewardSculptor has strong immutable experiment orchestration,
  reference admission, reward iteration, objective evaluation, and provenance;
- it has no SONIC execution, G1 visual policy input, online OGMP oracle,
  branch/recovery runtime, or validated box-reacquisition benchmark;
- current OGMP support is a fixed linear phase-window scaffold with an exact
  clock and manifest;
- the active rail-hop project is paused integration evidence, not the new
  adaptive-recovery benchmark.

## Update rule

Add a dated entry after every Lokesh meeting or scope-changing experiment.
Record confirmed decisions, superseded assumptions, unresolved questions,
source hashes/versions, and exact implementation commits. Never silently
promote an inference to a decision.

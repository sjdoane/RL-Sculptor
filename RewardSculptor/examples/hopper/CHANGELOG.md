# Sculpt Changelog

Auto-appended by `sculpt run`. Each entry records the iteration outcome, the diagnosed failure modes, and the edits applied to produce the next reward version.

## Iteration 0

- **Reward before**: `current.py`
- **Reward after**:  `v1.py`
- **Primary metric** (`mean_return`): +84.3856 (Δ —)
- **Behavior metrics**: `max_episode_length`=52, `mean_forward_velocity`=0.667, `fall_rate`=1
- **Failure modes**: component_imbalance
- **Diagnosis confidence**: 0.50
- **Evidence**: Dry-run stub diagnosis. mean_return=84.38563079999999, behavior={'fall_rate': 1.0, 'max_episode_length': 52, 'mean_episode_length': 51.5, 'mean_forward_velocity': 0.6669808030128479, 'mean_return': 85.85094261169434, 'n_episodes': 4, 'termination_reason_counts': {'terminated': 4, 'truncated': 0}}.
- **Edits**:
  - [increase] `alive_bonus` → `bumped`
    - *rationale*: novel. Dry-run canned edit — bumps alive_bonus by 0.5 so every iteration produces a distinct reward file.
    - *paper_refs*: (novel)

## Iteration 1

- **Reward before**: `current.py`
- **Reward after**:  `v2.py`
- **Primary metric** (`mean_return`): +105.3281 (Δ +20.9425 vs prev)
- **Behavior metrics**: `max_episode_length`=52, `mean_forward_velocity`=0.5611, `fall_rate`=1
- **Failure modes**: component_imbalance
- **Diagnosis confidence**: 0.50
- **Evidence**: Dry-run stub diagnosis. mean_return=105.32814739999999, behavior={'fall_rate': 1.0, 'max_episode_length': 52, 'mean_episode_length': 51.25, 'mean_forward_velocity': 0.5611203610897064, 'mean_return': 80.01090240478516, 'n_episodes': 4, 'termination_reason_counts': {'terminated': 4, 'truncated': 0}}.
- **Edits**:
  - [increase] `alive_bonus` → `bumped`
    - *rationale*: novel. Dry-run canned edit — bumps alive_bonus by 0.5 so every iteration produces a distinct reward file.
    - *paper_refs*: (novel)

## Iteration 2

- **Reward before**: `current.py`
- **Reward after**:  `v3.py`
- **Primary metric** (`mean_return`): +141.6864 (Δ +36.3582 vs prev)
- **Behavior metrics**: `max_episode_length`=58, `mean_forward_velocity`=0.5068, `fall_rate`=1
- **Failure modes**: component_imbalance
- **Diagnosis confidence**: 0.50
- **Evidence**: Dry-run stub diagnosis. mean_return=141.6863572, behavior={'fall_rate': 1.0, 'max_episode_length': 58, 'mean_episode_length': 57.0, 'mean_forward_velocity': 0.5067795664072037, 'mean_return': 85.88846206665039, 'n_episodes': 4, 'termination_reason_counts': {'terminated': 4, 'truncated': 0}}.
- **Edits**:
  - [increase] `alive_bonus` → `bumped`
    - *rationale*: novel. Dry-run canned edit — bumps alive_bonus by 0.5 so every iteration produces a distinct reward file.
    - *paper_refs*: (novel)


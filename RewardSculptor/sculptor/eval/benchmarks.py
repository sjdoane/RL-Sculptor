"""Benchmark task suite (§Ship 26 / E1).

Four tasks spanning difficulty, each pairing an mjlab task_id + a
natural-language behavior goal (what the sculptor pipeline gets) with a
HAND-AUTHORED spec metric (what the evaluation believes — see
spec_metrics.py). The E2 harness iterates this table; E3 baselines and
E4 ablation conditions run the same tasks so comparisons are paired.

Why these four:
  * cartpole_balance — cheap sanity task (256 envs, minutes/run): the
    harness self-test target and the high-seed-count task for variance
    estimation.
  * g1_floss / g1_kick — the two behaviors Sam has run real missions on
    (g1-flossing, g1-kick-v2/3 projects), so spec metrics can be
    validated against existing labeled-by-outcome recordings.
  * go1_trot — quadruped locomotion; gait rather than spin because the
    rollout does not persist yaw (projected gravity is yaw-invariant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    """One benchmark entry. `behavior_goal` is the EXACT NL string fed
    to the pipeline (decompose/diagnose see only this); `spec_metric`
    names the ground-truth function in spec_metrics.py."""

    name: str
    task_id: str
    behavior_goal: str
    spec_metric: str
    adapter: str = "mjlab"
    #: Default adapter-config overrides for eval runs (envs sized for
    #: throughput on the eval GPU; the harness may override).
    adapter_config: dict[str, Any] = field(default_factory=dict)
    #: Why this task / what the spec measures — surfaced in reports.
    notes: str = ""


BENCHMARKS: dict[str, BenchmarkTask] = {
    b.name: b
    for b in (
        BenchmarkTask(
            name="cartpole_balance",
            task_id="Mjlab-Cartpole-Balance",
            behavior_goal="balance the pole upright and keep the cart centered",
            spec_metric="cartpole_balance",
            adapter_config={"num_envs": 256},
            notes=(
                "Sanity task: spec = normalized mean episode length. "
                "Cheap enough for high seed counts."
            ),
        ),
        BenchmarkTask(
            name="g1_floss",
            task_id="Mjlab-Velocity-Flat-Unitree-G1",
            behavior_goal=(
                "perform a continuous flossing dance: swing the hips "
                "side to side while the arms swing in opposition"
            ),
            spec_metric="g1_floss",
            adapter_config={"num_envs": 4096},
            notes=(
                "Spec = joint-space periodicity (dominant-period power "
                "ratio over the most-moving joints) x uprightness."
            ),
        ),
        BenchmarkTask(
            name="g1_kick",
            task_id="Mjlab-Velocity-Flat-Unitree-G1",
            behavior_goal=(
                "repeatedly kick forward with one leg while keeping "
                "balance on the other"
            ),
            spec_metric="g1_kick",
            adapter_config={"num_envs": 4096},
            notes=(
                "Spec = leg-speed burst intensity x burst-vs-hum ratio "
                "x uprightness."
            ),
        ),
        BenchmarkTask(
            name="go1_trot",
            task_id="Mjlab-Velocity-Flat-Unitree-Go1",
            behavior_goal="trot forward in a straight line at a steady pace",
            spec_metric="go1_trot",
            adapter_config={"num_envs": 4096},
            notes=(
                "Spec = saturating horizontal speed x straightness x "
                "uprightness. Gait, not spin: yaw is not persisted."
            ),
        ),
    )
}


def get_benchmark(name: str) -> BenchmarkTask:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise KeyError(
            f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}"
        ) from None

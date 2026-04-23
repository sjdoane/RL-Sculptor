"""Adapters bridge Sculptor to a specific RL stack + environment.

Built-in ready-to-train:
  - `GymSB3Adapter` — Gymnasium + Stable-Baselines3 (CPU-friendly).
  - `MjlabAdapter` — mjlab (MuJoCo-Warp), primary GPU target.

Scaffolded (NotImplementedError, adoption guides in docs/adapters/):
  - `IsaacLabAdapter`, `MjxAdapter`, `RllibAdapter`.

`ADAPTER_REGISTRY` is the single source of truth for the UI's
`GET /library/adapters` endpoint and for the short-name → dotted-class
resolution used by `SculptorAdapter.load_adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sculptor.adapters.base import (
    ComponentProbe,
    RewardContract,
    RolloutResult,
    SculptorAdapter,
    TrainResult,
)


AdapterStatus = Literal["ready", "coming_soon"]


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    display_name: str
    class_path: str
    status: AdapterStatus
    supported_robot_categories: tuple[str, ...]
    adoption_guide_url: str
    estimated_effort: str = ""


ADAPTER_REGISTRY: dict[str, AdapterInfo] = {
    "gym_sb3": AdapterInfo(
        name="gym_sb3",
        display_name="Gymnasium + SB3",
        class_path="sculptor.adapters.gym_sb3.GymSB3Adapter",
        status="ready",
        supported_robot_categories=("Quadruped", "Humanoid", "Other"),
        adoption_guide_url="docs/adapters.md#gymnasium--sb3",
    ),
    "mjlab": AdapterInfo(
        name="mjlab",
        display_name="mjlab (MuJoCo-Warp)",
        class_path="sculptor.adapters.mjlab.MjlabAdapter",
        status="ready",
        supported_robot_categories=("Quadruped", "Humanoid", "Arm"),
        adoption_guide_url="MJLAB_PIVOT_DESIGN.md",
    ),
    "isaac": AdapterInfo(
        name="isaac",
        display_name="Isaac Lab",
        class_path="sculptor.adapters.isaac_lab.IsaacLabAdapter",
        status="coming_soon",
        supported_robot_categories=(
            "Humanoid", "Quadruped", "Arm", "Mobile_Manipulator",
        ),
        adoption_guide_url="docs/adapters/isaac.md",
        estimated_effort="4-8 hours for a senior Isaac Lab user",
    ),
    "mjx": AdapterInfo(
        name="mjx",
        display_name="Brax / MJX",
        class_path="sculptor.adapters.mjx.MjxAdapter",
        status="coming_soon",
        supported_robot_categories=("Quadruped", "Humanoid"),
        adoption_guide_url="docs/adapters/mjx.md",
        estimated_effort="4-6 hours for a JAX-familiar contributor",
    ),
    "rllib": AdapterInfo(
        name="rllib",
        display_name="Ray RLlib",
        class_path="sculptor.adapters.rllib.RllibAdapter",
        status="coming_soon",
        supported_robot_categories=("Quadruped", "Humanoid", "Other"),
        adoption_guide_url="docs/adapters/rllib.md",
        estimated_effort="4-8 hours (worker coordination + checkpoint shape)",
    ),
}


def get_adapter_info(name: str) -> AdapterInfo:
    """Look up registry metadata by short name. Raises KeyError on miss."""
    return ADAPTER_REGISTRY[name]


def resolve_class_path(name: str) -> str:
    """Short-name → dotted class path, for use with `load_adapter`."""
    return ADAPTER_REGISTRY[name].class_path


__all__ = [
    "ComponentProbe",
    "RewardContract",
    "RolloutResult",
    "SculptorAdapter",
    "TrainResult",
    "AdapterInfo",
    "AdapterStatus",
    "ADAPTER_REGISTRY",
    "get_adapter_info",
    "resolve_class_path",
]

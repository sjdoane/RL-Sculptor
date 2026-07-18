"""Prompt-driven world and task authoring.

The package deliberately exposes declarative JSON artifacts. Heavy mjlab and
MuJoCo imports stay inside compiler/gate functions so schema, CLI, and UI
operations remain cheap.
"""

from sculptor.world.artifacts import WorldArtifactStore
from sculptor.world.capabilities import (
    CapabilityError, RobotCapability, SimulatorCapability,
    robot_capabilities, simulator_capability,
)
from sculptor.world.task_spec import (
    TASK_SPEC_VERSION, load_task_spec, validate_task_spec,
)
from sculptor.world.world_spec import (
    WORLD_SPEC_VERSION, load_world_spec, validate_world_spec,
)

__all__ = [
    "CapabilityError", "RobotCapability", "SimulatorCapability",
    "TASK_SPEC_VERSION", "WORLD_SPEC_VERSION", "WorldArtifactStore",
    "load_task_spec", "load_world_spec", "robot_capabilities",
    "simulator_capability", "validate_task_spec", "validate_world_spec",
]

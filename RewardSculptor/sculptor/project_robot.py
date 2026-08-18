"""Canonical project robot-namespace resolution.

Robot-catalog asset slugs (``unitree_g1``) identify model sources. Reference
artifacts and portable policy contracts use embodiment namespaces (``g1``).
The mapping is persisted in project metadata and is never inferred from task
ids or tokenized names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ROBOT_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

# Read bridge for metadata created before ``reference_robot`` was persisted.
# This is intentionally finite: adding a catalog robot requires an explicit
# catalog mapping, not a generic naming heuristic.
LEGACY_REFERENCE_ROBOT_BY_LIBRARY_SLUG: dict[str, str] = {
    "ability_hand": "ability_hand",
    "agilex_piper": "agilex_piper",
    "agility_cassie": "agility_cassie",
    "ant": "ant",
    "anybotics_anymal_b": "anybotics_anymal_b",
    "anybotics_anymal_c": "anybotics_anymal_c",
    "apptronik_apollo": "apptronik_apollo",
    "arx_l5": "arx_l5",
    "bitcraze_crazyflie_2": "bitcraze_crazyflie_2",
    "booster_t1": "booster_t1",
    "boston_dynamics_spot": "boston_dynamics_spot",
    "cartpole_mjlab": "cartpole_mjlab",
    "dynamixel_2r": "dynamixel_2r",
    "elf2_humanoid": "elf2_humanoid",
    "fourier_n1": "fourier_n1",
    "franka_emika_panda": "franka_emika_panda",
    "franka_fr3": "franka_fr3",
    "franka_fr3_v2": "franka_fr3_v2",
    "g1": "g1",
    "go1": "go1",
    "half_cheetah": "half_cheetah",
    "halfcheetah": "halfcheetah",
    "hello_robot_stretch": "hello_robot_stretch",
    "hello_robot_stretch_3": "hello_robot_stretch_3",
    "hopper": "hopper",
    "humanoid": "humanoid",
    "i2rt_yam": "i2rt_yam",
    "jvrc_humanoid": "jvrc_humanoid",
    "kinova_gen3": "kinova_gen3",
    "kuka_iiwa_14": "kuka_iiwa_14",
    "leap_hand": "leap_hand",
    "low_cost_robot_arm": "low_cost_robot_arm",
    "mujoco_humanoid_builtin": "mujoco_humanoid_builtin",
    "openarm_v1": "openarm_v1",
    "pal_talos": "pal_talos",
    "pal_tiago_dual": "pal_tiago_dual",
    "pndbotics_adam_lite": "pndbotics_adam_lite",
    "rainy_rby1": "rainy_rby1",
    "rethink_robotics_sawyer": "rethink_robotics_sawyer",
    "robot_soccer_kit": "robot_soccer_kit",
    "robotis_op3": "robotis_op3",
    "robotstudio_so101": "robotstudio_so101",
    "robotiq_2f85": "robotiq_2f85",
    "robotiq_2f85_v4": "robotiq_2f85_v4",
    "shadow_dexee": "shadow_dexee",
    "shadow_hand": "shadow_hand",
    "skydio_x2": "skydio_x2",
    "tetheria_aero_hand_open": "tetheria_aero_hand_open",
    "toddlerbot_2xc": "toddlerbot_2xc",
    "toddlerbot_2xm": "toddlerbot_2xm",
    "trossen_vx300s": "trossen_vx300s",
    "trossen_wx250s": "trossen_wx250s",
    "trs_so_arm100": "trs_so_arm100",
    "ufactory_xarm7": "ufactory_xarm7",
    "unitree_a1": "unitree_a1",
    "unitree_aliengo": "unitree_aliengo",
    "unitree_g1": "g1",
    "unitree_go1": "go1",
    "unitree_go2": "unitree_go2",
    "unitree_h1": "unitree_h1",
    "unitree_h1_2": "unitree_h1_2",
    "unitree_z1": "unitree_z1",
    "universal_robots_ur10e": "universal_robots_ur10e",
    "universal_robots_ur5e": "universal_robots_ur5e",
    "walker2d": "walker2d",
    "wonik_allegro": "wonik_allegro",
}


def _namespace(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"project {field} is missing")
    namespace = value.strip()
    if _ROBOT_NAMESPACE_RE.fullmatch(namespace) is None:
        raise ValueError(f"project {field} is malformed")
    return namespace


def resolve_project_reference_robot(project_dir: Path) -> str:
    """Return the exact policy/reference namespace for ``project_dir``.

    Explicit metadata is authoritative. For a legacy allowlisted catalog
    slug, an explicit value must also agree with the allowlist; contradictory
    metadata fails closed before a policy or motion can influence training.
    """

    metadata_path = Path(project_dir) / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            "project exact robot identity is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    robot_source = metadata.get("robot_source")
    if robot_source is None:
        raise ValueError("project robot_source metadata is missing")
    if not isinstance(robot_source, dict):
        raise TypeError("project robot_source metadata must be an object")

    raw_library_slug = robot_source.get("library_slug")
    library_slug = (
        _namespace(raw_library_slug, field="robot library slug")
        if raw_library_slug is not None
        else None
    )
    raw_library_name = robot_source.get("library_name")
    library_name = (
        _namespace(raw_library_name, field="legacy robot library name")
        if raw_library_name is not None
        else None
    )
    if (
        library_slug is not None
        and library_name is not None
        and library_slug != library_name
    ):
        raise ValueError(
            "project robot library slug contradicts its legacy library name"
        )
    library_slug = library_slug or library_name
    raw_reference_robot = robot_source.get("reference_robot")
    explicit = (
        _namespace(raw_reference_robot, field="reference robot namespace")
        if raw_reference_robot is not None
        else None
    )

    legacy_expected = (
        LEGACY_REFERENCE_ROBOT_BY_LIBRARY_SLUG.get(library_slug)
        if library_slug is not None
        else None
    )
    if explicit is not None:
        if legacy_expected is not None and explicit != legacy_expected:
            raise ValueError(
                "project reference robot namespace contradicts its robot "
                f"library slug: {explicit!r} != {legacy_expected!r}"
            )
        return explicit

    if legacy_expected is not None:
        return legacy_expected

    if library_slug is None:
        raise ValueError(
            "project exact robot identity is missing; select a robot from "
            "the robot library before using a policy or motion starting point"
        )
    raise ValueError(
        f"project robot library slug {library_slug!r} has no explicit "
        "reference robot namespace"
    )


__all__ = [
    "LEGACY_REFERENCE_ROBOT_BY_LIBRARY_SLUG",
    "resolve_project_reference_robot",
]

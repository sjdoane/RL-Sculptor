"""Robot-agnostic prompt authoring and deterministic clarification.

The author emits only strict ``WorldSpec``/``TaskSpec`` dictionaries.  It does
not emit simulator code and it never infers robot asset names from a task ID.
An optional language-model implementation can be injected through
``AuthorModel``; the built-in author is a deterministic, offline baseline for
the acceptance task families in the environment-authoring architecture.

Clarification is deliberately independent from authoring.  Its questions are
derived from a complete parameter provenance report, are paginated without
truncation, and carry both draft and question-set hashes.  Applying stale
answers therefore fails before either specification is changed.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from sculptor.world.capabilities import (
    CapabilityError,
    RobotCapability,
    resolve_robot_capability,
    robot_capabilities,
)
from sculptor.world.task_spec import validate_task_spec
from sculptor.world.world_spec import validate_world_spec

AUTHORING_CONTRACT_VERSION = 1
CLARIFICATION_VERSION = 1
MAX_QUESTIONS_PER_PAGE = 5
_PROVENANCE = {"prompt", "user", "default", "timeout_default", "repair"}


class AuthoringError(ValueError):
    """The prompt or model output cannot produce an admissible draft."""


class StaleClarificationError(AuthoringError):
    """Clarification answers refer to a different immutable draft/question set."""


class AuthorModel(Protocol):
    """Minimal, mockable interface for an optional language-model author.

    The returned mapping has exact keys ``world_spec``, ``task_spec``, and an
    optional ``parameter_provenance`` mapping of combined JSON pointers to one
    of the provenance values accepted by WorldSpec.  Validation and capability
    enforcement remain local and deterministic.
    """

    def generate_authoring(
        self, request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return a declarative authoring candidate."""


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ParameterRecord:
    path: str
    value: Any
    provenance: str
    load_bearing: bool
    reason: str | None = None
    question_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class UnderspecificationReport:
    version: int
    parameters: tuple[ParameterRecord, ...]
    defaulted_load_bearing_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class ClarificationChoice:
    choice_id: str
    label: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class ClarificationQuestion:
    question_id: str
    parameter_path: str
    prompt: str
    choices: tuple[ClarificationChoice, ...]
    default_choice_id: str
    default_reason: str

    @property
    def system_default_label(self) -> str:
        default = next(
            choice for choice in self.choices
            if choice.choice_id == self.default_choice_id
        )
        return (
            f"System decides (default: {default.label} — "
            f"{self.default_reason})"
        )

    def to_dict(self) -> dict[str, Any]:
        data = _jsonable(self)
        data["system_default"] = {
            "choice_id": "system_default",
            "resolves_to": self.default_choice_id,
            "label": self.system_default_label,
            "reason": self.default_reason,
        }
        return data


@dataclass(frozen=True)
class ClarificationPage:
    page: int
    questions: tuple[ClarificationQuestion, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "questions": [question.to_dict() for question in self.questions],
        }


@dataclass(frozen=True)
class ClarificationPlan:
    version: int
    draft_hash: str
    question_set_hash: str
    pages: tuple[ClarificationPage, ...]

    @property
    def questions(self) -> tuple[ClarificationQuestion, ...]:
        return tuple(question for page in self.pages for question in page.questions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "draft_hash": self.draft_hash,
            "question_set_hash": self.question_set_hash,
            "pages": [page.to_dict() for page in self.pages],
            "question_count": len(self.questions),
        }


@dataclass(frozen=True)
class AuthoringDraft:
    version: int
    prompt: str
    capability_id: str
    world_spec: dict[str, Any]
    task_spec: dict[str, Any]
    underspecification_report: UnderspecificationReport
    clarification_plan: ClarificationPlan
    draft_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prompt": self.prompt,
            "capability_id": self.capability_id,
            "world_spec": copy.deepcopy(self.world_spec),
            "task_spec": copy.deepcopy(self.task_spec),
            "underspecification_report": self.underspecification_report.to_dict(),
            "clarification_plan": self.clarification_plan.to_dict(),
            "draft_hash": self.draft_hash,
        }


@dataclass(frozen=True)
class ClarificationAnswer:
    question_id: str
    choice_id: str
    source: str = "user"

    def __post_init__(self) -> None:
        if self.source not in {"user", "default", "timeout_default"}:
            raise ValueError(
                "clarification answer source must be user, default, or "
                "timeout_default"
            )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class ClarificationSubmission:
    version: int
    draft_hash: str
    question_set_hash: str
    answers: tuple[ClarificationAnswer, ...]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class AppliedAuthoring:
    version: int
    initial_draft_hash: str
    result_hash: str
    world_spec: dict[str, Any]
    task_spec: dict[str, Any]
    underspecification_report: UnderspecificationReport
    clarification_ledger: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "initial_draft_hash": self.initial_draft_hash,
            "result_hash": self.result_hash,
            "world_spec": copy.deepcopy(self.world_spec),
            "task_spec": copy.deepcopy(self.task_spec),
            "underspecification_report": self.underspecification_report.to_dict(),
            "clarification_ledger": copy.deepcopy(self.clarification_ledger),
        }


def _descriptor_for(
    capability_id: str, paths: Sequence[Path | str],
) -> str | None:
    for raw_path in paths:
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("capability_id") == capability_id:
            return str(path.resolve())
    return None


def _required_capabilities(prompt: str) -> frozenset[str]:
    text = prompt.casefold()
    required: set[str] = set()
    compact_low_rails = _requests_compact_low_rail_course(prompt)
    if compact_low_rails or any(token in text for token in (
            "walk", "run", "terrain", "parkour", "jump", "humanoid")):
        required.add("locomotion")
    if compact_low_rails or any(
        token in text for token in ("jump", "parkour", "vault", "hurdle")
    ):
        required.add("jump")
    grasp_intent = (
        any(token in text for token in (
            "gripper", "grasp", "pick up", "pickup", "carry", "lift"))
        or re.search(r"\bpick(?:ing|ed)?\b.{0,32}\b(?:and\s+)?place\b", text)
        or re.search(r"\bpick\b.{0,24}\bup\b", text)
    )
    if grasp_intent:
        required.add("grasp")
    elif any(token in text for token in (
            "ball", "object", "push", "score", "move into")):
        required.add("push")
    if "humanoid" in text:
        # The installed descriptor must represent the requested embodiment;
        # an arm with a gripper is not silently relabelled as a humanoid.
        required.add("whole_body")
    return frozenset(required)


def _candidate_score(cap: RobotCapability, required: frozenset[str]) -> tuple:
    if "locomotion" in required:
        physical = (
            cap.geometry.max_gap_m + cap.geometry.max_step_height_m,
            cap.geometry.leg_length_m,
        )
    elif "grasp" in required or "push" in required:
        physical = (cap.geometry.reach_radius_m, float(cap.contact_capacity))
    else:
        physical = (cap.geometry.reach_radius_m, cap.geometry.standing_height_m)
    return (*physical, cap.capability_id)


def _resolve_capability(
    prompt: str, capability_id: str | None,
    descriptor_paths: Sequence[Path | str],
) -> tuple[RobotCapability, tuple[RobotCapability, ...], str]:
    required = _required_capabilities(prompt)
    if capability_id is not None:
        cap = resolve_robot_capability(
            capability_id, required=required, extra_paths=descriptor_paths,
        )
        return cap, (cap,), "user"

    installed = robot_capabilities(descriptor_paths)
    eligible = [cap for cap in installed.values()
                if required <= cap.capabilities]
    if not eligible:
        inventory = "; ".join(
            f"{name}={sorted(cap.capabilities)}"
            for name, cap in sorted(installed.items())
        )
        raise CapabilityError(
            f"no installed robot satisfies required capabilities "
            f"{sorted(required)}; installed: {inventory}"
        )
    eligible.sort(key=lambda cap: _candidate_score(cap, required), reverse=True)
    # A prompt constraint that has one installed solution grounds the choice.
    # Multiple viable embodiments remain a disclosed system default.
    source = "prompt" if required and len(eligible) == 1 else "default"
    return eligible[0], tuple(eligible[:4]), source


def _portable_offline_capability(
    prompt: str,
    candidates: Sequence[RobotCapability],
    *,
    version: str,
    parent: str | None,
    grounding: Sequence[str],
    descriptor_paths: Sequence[Path | str],
) -> RobotCapability:
    """Choose the candidate whose authored semantics fit the broadest cohort.

    Capability IDs are never treated as interchangeable.  For an ambiguous
    auto-selection, each candidate first authors its own task; that task is
    then schema/capability-validated against the other eligible descriptors.
    This prefers portable semantic roles (for example a shared ``torso``
    push role) while still preserving every robot-specific constraint.
    """
    ranked: list[tuple[int, tuple, RobotCapability]] = []
    required = _required_capabilities(prompt)
    for candidate in candidates:
        world, task, _ = _offline_author(
            prompt,
            candidate,
            version=version,
            parent=parent,
            grounding=grounding,
            descriptor_path=_descriptor_for(
                candidate.capability_id, descriptor_paths),
        )
        compatible = 0
        for alternative in candidates:
            alternate_world = copy.deepcopy(world)
            robot = alternate_world["shared"]["robot"]
            robot["capability_id"] = alternative.capability_id
            descriptor = _descriptor_for(
                alternative.capability_id, descriptor_paths)
            if descriptor is None:
                robot.pop("descriptor_path", None)
            else:
                robot["descriptor_path"] = descriptor
            if (not validate_world_spec(alternate_world)
                    and not validate_task_spec(task, world=alternate_world)):
                compatible += 1
        ranked.append((
            compatible,
            _candidate_score(candidate, required),
            candidate,
        ))
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


# Deliberately NO "a"/"an": they are ARTICLES ("a box course" = a parkour
# course, not 1 box), so treating them as a count mis-reads "a course with 8
# steps" as 1. Explicit "single"/"one" still count.
_NUMBER_WORDS = {
    "single": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_COURSE_NOUNS = (
    r"(?:boxes|box|platforms?|steps?|obstacles?|blocks?|stairs?|hurdles?|rails?)"
)
_COURSE_COUNT_MODIFIER = (
    r"(?:progressively|more|high-friction|low-friction|taller|higher|lower|"
    r"shorter|wider|narrower|tall|high|low|raised|staggered|ascending|"
    r"descending|identical|fixed|small|large)"
)
_COUNT_RE = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\b\s+"
    r"(?:" + _COURSE_COUNT_MODIFIER + r"\s*,?\s*)*"
    + _COURSE_NOUNS + r"\b")


def _parse_count(prompt: str, *, default: int, lo: int = 1, hi: int = 12) -> int:
    """Extract a requested course-element COUNT from free text, e.g. "4 boxes",
    "climb five platforms". Clamped to [lo, hi]; falls back to `default` when no
    count is stated (so an unquantified "parkour course" keeps its nominal length).
    Fixes the "prompt for 4 boxes, get 3" bug — the template never read the number."""
    text = prompt.casefold()
    m = _COUNT_RE.search(text)
    if not m:
        return default
    token = m.group(1)
    n = int(token) if token.isdigit() else _NUMBER_WORDS.get(token, default)
    return max(lo, min(hi, n))


def _requests_post_route_jump(prompt: str) -> bool:
    """Whether the prompt orders one jump after the navigation route."""
    text = prompt.casefold()
    requests_jump = bool(re.search(
        r"\b(?:jump|leap|hop)(?:s|ed|ing)?\b", text
    ))
    return requests_jump and bool(re.search(
        r"\b(?:then|after|afterward|once|followed\s+by|at\s+(?:the\s+)?"
        r"(?:finish|end))\b",
        text,
    ))


def _requests_compact_low_rail_course(prompt: str) -> bool:
    """Recognize the named, deterministic low-rail course profile.

    Keep this cue narrow so ordinary references to rails or generic hurdle
    courses retain their existing parkour semantics.  Both forms below are
    natural researcher-facing ways to request the profile in the UI.
    """
    text = prompt.casefold()
    return bool(
        re.search(r"\bcompact\s+low[- ]rail(?:\s+course)?\b", text)
        or re.search(r"\blow[- ]rails?\b", text)
    )


def _intent(prompt: str) -> str:
    text = prompt.casefold()
    is_object = (
        any(token in text for token in
            ("ball", "object", "cube", "block", "gripper", "grasp"))
        and any(token in text for token in
                ("goal", "region", "score", "into", "onto", "place", "move",
                 "push", "put", "drop", "kick", "throw", "roll", "carry")))
    # A compact low-rail course is a deterministic reference-motion evolution
    # profile, not a generic platform course. Route it before the broad
    # parkour/hurdle vocabulary so the offline path preserves its exact short
    # horizon geometry and observation interface.
    if _requests_compact_low_rail_course(prompt):
        return "compact_low_rails"
    # Planar obstacle-navigation cues are not parkour.  Route these before the
    # generic box/obstacle checks below so "slalom around four boxes" cannot be
    # silently rewritten as "climb three platforms" by the offline fallback.
    if any(token in text for token in (
            "slalom", "weave", "weaving", "zig-zag", "zigzag")):
        return "slalom"
    # Specific parkour cues win outright.
    if any(token in text for token in (
            "parkour", "course of boxes", "box course", "obstacle course",
            "vault", "hurdle")):
        return "parkour"
    # An object-manipulation prompt is object_to_region EVEN IF it mentions a box
    # or platform ("push the block onto the platform") — the generic parkour nouns
    # below must not steal it.
    if is_object:
        return "object_to_region"
    # Generic climb / box / platform cues (no object task) → parkour. This is what
    # lets "generate 4 boxes" / "climb two boxes then jump off" ground at all.
    if any(token in text for token in (
            "boxes", "platform", "platforms",
            "climb", "climbs", "climbing")):
        return "parkour"
    if any(token in text for token in
           ("uneven", "rough terrain", "rough ground", "slope", "terrain")):
        return "uneven_terrain"
    if any(token in text for token in ("walk", "go to", "move to", "traverse")):
        return "robot_to_region"
    raise AuthoringError(
        "offline author cannot ground this prompt safely; inject an AuthorModel "
        "or use an uneven-terrain, parkour, robot-to-region, or "
        "object-to-region prompt"
    )


def _base_world(
    prompt: str, cap: RobotCapability, required: frozenset[str], *,
    version: str, parent: str | None, grounding: Sequence[str],
    descriptor_path: str | None,
) -> dict[str, Any]:
    robot: dict[str, Any] = {
        "capability_id": cap.capability_id,
        "required_capabilities": sorted(required),
    }
    if descriptor_path:
        robot["descriptor_path"] = descriptor_path
    return {
        "world_spec_version": 2,
        "meta": {
            "version": version,
            "parent": parent,
            "source": "generated",
            "prompt": prompt,
            "grounding": list(grounding),
            "parameter_provenance": {},
        },
        "shared": {
            "eval_seed": 1729,
            "robot": robot,
            "terrain": {"kind": "plane"},
            "obstacles": {
                "layout": "linear", "waypoints": "auto",
                "start_offset_m": 0.0, "course": [],
            },
            "objects": {},
            "zones": {},
        },
        "train": {"variations": [], "curriculum": {}},
    }


def _base_task(
    prompt: str, *, version: str, parent: str | None,
    grounding: Sequence[str],
) -> dict[str, Any]:
    return {
        "task_spec_version": 1,
        "meta": {
            "version": version,
            "parent": parent,
            "source": "generated",
            "prompt": prompt,
            "grounding": list(grounding),
        },
        "shared": {
            "control_mode": "goal_directed",
            "goal": {},
            "contacts": {"desired": [], "forbidden": [], "terminate_on": []},
            "termination": {
                "fall": "capability_default",
                "out_of_bounds_m": 12.0,
                "success_ends_episode": False,
                "episode_length_s": 20.0,
            },
            "observations": {
                "proprioception": True,
                "height_scan": False,
                "object_relative": [],
                "region_relative": [],
            },
        },
        "train": {"goal_sampling": [], "scaffolds": []},
    }


def _author_uneven(world: dict[str, Any], task: dict[str, Any]) -> None:
    world["shared"]["terrain"] = {
        "kind": "generator",
        "layout": {
            "mode": "curriculum_grid", "rows": 6,
            "tile_size_m": [8.0, 8.0], "border_width_m": 2.0,
        },
        "evaluation_difficulty": 0.45,
        "sub_terrains": {
            "rough": {
                "type": "hf_random_uniform",
                "nominal": {
                    "noise_range_m": [0.015, 0.06],
                    "noise_step_m": 0.015,
                },
            },
            "slope": {
                "type": "hf_pyramid_sloped",
                "nominal": {"slope_deg": 12.0, "inverted": False},
            },
        },
    }
    world["shared"]["zones"] = {
        "finish": {
            "kind": "box", "center_m": [6.0, 0.0, 0.5],
            "size_m": [1.0, 4.0, 1.0],
        },
    }
    world["train"] = {
        "variations": [
            {
                "id": "roughness",
                "target": (
                    "/shared/terrain/sub_terrains/rough/nominal/"
                    "noise_range_m/1"
                ),
                "class": "generator_parameter",
                "distribution": {"kind": "uniform", "low": 0.04, "high": 0.12},
                "curriculum": {"axis": "difficulty", "mapping": "linear"},
            },
            {
                "id": "slope_angle",
                "target": (
                    "/shared/terrain/sub_terrains/slope/nominal/slope_deg"
                ),
                "class": "generator_parameter",
                "distribution": {"kind": "uniform", "low": 5.0, "high": 20.0},
                "curriculum": {"axis": "difficulty", "mapping": "linear"},
            },
        ],
        "curriculum": {
            "difficulty_range": [0.0, 1.0],
            "promotion": {
                "signal": "traversal_fraction",
                "promote_above": 0.75,
                "demote_below": 0.50,
            },
        },
    }
    task["shared"]["goal"] = {
        "id": "reach_finish", "type": "robot_to_region", "region": "finish",
        "success": {
            "predicate": "distance_below", "hold_s": 0.5,
            "tolerance_m": 0.25,
        },
    }
    task["shared"]["observations"].update({
        "height_scan": True, "region_relative": ["finish"],
    })


def _author_parkour(
    world: dict[str, Any], task: dict[str, Any], cap: RobotCapability,
    *, count: int = 3,
) -> None:
    step = max(0.12, min(0.35, cap.geometry.max_step_height_m * 0.60))
    gap = max(0.15, min(0.45, cap.geometry.max_gap_m * 0.55))
    # Keep a robot-neutral launch pad before the first obstacle.  Starting a
    # default stance with its feet/calf already intersecting box_01 invalidates
    # admission and teaches no useful parkour behavior.
    lead_in = round(max(0.6, cap.geometry.foot_length_m * 3.0), 4)
    world["shared"]["obstacles"]["start_offset_m"] = lead_in

    # Build `count` platforms separated by gaps.  Heights ramp gently across the
    # course (each within the robot's max step height) and widths grow slightly so
    # later, higher platforms stay landable — the shape the fixed 3-box template
    # encoded, generalized to any authored count.
    n = max(1, min(12, int(count)))
    course: list[dict[str, Any]] = []
    height_ceiling = max(step, cap.geometry.max_step_height_m)
    for i in range(n):
        frac = i / max(1, n - 1)                        # 0 → 1 across the course
        height = min(height_ceiling, step * (0.70 + 0.45 * frac))
        course.append({
            "id": f"box_{i + 1:02d}", "element": "platform", "nominal": {
                "height_m": round(height, 4),
                "length_m": round(1.0 + 0.1 * i, 4), "width_m": 1.2,
            }})
        if i < n - 1:
            course.append({
                "id": f"gap_{i + 1:02d}", "element": "gap", "nominal": {
                    "length_m": round(gap * (1.0 + 0.15 * frac), 4),
                    "width_m": 1.2, "depth_m": 0.5,
                }})
    world["shared"]["obstacles"]["course"] = course

    # Domain-randomize a representative MIDDLE platform's height and (when the
    # course has ≥2 platforms) a gap length, so training sees a distribution of
    # layouts rather than one frozen course (sim-to-real robustness).
    mid_box = course[2 * (n // 2)]["id"]                # a central platform id
    variations: list[dict[str, Any]] = [
        {
            "id": "middle_box_height",
            "target": f"/shared/obstacles/course/@{mid_box}/nominal/height_m",
            "class": "model_field",
            "distribution": {
                "kind": "uniform", "low": round(step * 0.55, 4),
                "high": round(min(cap.geometry.max_step_height_m, step * 1.25), 4),
            },
        },
    ]
    if n >= 2:
        mid_gap = f"gap_{max(1, n // 2):02d}"
        variations.append({
            "id": "second_gap_length",
            "target": f"/shared/obstacles/course/@{mid_gap}/nominal/length_m",
            "class": "model_field",
            "distribution": {
                "kind": "uniform", "low": round(gap * 0.70, 4),
                "high": round(min(cap.geometry.max_gap_m, gap * 1.30), 4),
            },
        })
    world["train"] = {
        "variations": variations,
        "curriculum": {
            "difficulty_range": [0.0, 1.0],
            "promotion": {
                "signal": "waypoint_success", "promote_above": 0.75,
                "demote_below": 0.50,
            },
        },
    }
    task["shared"]["control_mode"] = "waypoint_following"
    task["shared"]["goal"] = {
        "id": "complete_course", "type": "waypoint_sequence",
        "waypoints": "auto",
        "success": {
            "predicate": "sequence_complete", "hold_s": 0.15,
            "ordered": True,
        },
    }
    task["shared"]["observations"]["height_scan"] = True


_COMPACT_RAIL_FIRST_X_M = 0.35
_COMPACT_RAIL_SPACING_M = 0.50
_COMPACT_RAIL_SIZE_M = (0.10, 0.60, 0.06)
_COMPACT_RAIL_FIRST_LANDING_X_M = 0.65
_COMPACT_RAIL_LANDING_RADIUS_M = 0.30
_COMPACT_RAIL_FINISH_OFFSET_M = 0.40
_COMPACT_RAIL_FINISH_RADIUS_M = 0.45
_COMPACT_RAIL_EPISODE_S = 8.0
_COMPACT_RAIL_HOLD_S = 2.0


def _compact_low_rail_geometry(
    count: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Return the immutable geometry/topology of the compact rail profile.

    The profile is deliberately expressed in ordinary fixed objects and disk
    zones.  It does not add an event-phase observation or a second success
    authority, so a compatible policy keeps the same height-scan plus ordered
    region observation interface it already uses for waypoint locomotion.
    """
    n = max(1, min(12, int(count)))
    objects: dict[str, Any] = {}
    zones: dict[str, Any] = {}
    waypoints: list[str] = []
    rail_height = _COMPACT_RAIL_SIZE_M[2]
    for index in range(n):
        number = index + 1
        rail_id = f"rail_{number:02d}"
        waypoint_id = f"waypoint_{number:02d}"
        rail_x = round(
            _COMPACT_RAIL_FIRST_X_M + index * _COMPACT_RAIL_SPACING_M,
            4,
        )
        landing_x = round(
            _COMPACT_RAIL_FIRST_LANDING_X_M
            + index * _COMPACT_RAIL_SPACING_M,
            4,
        )
        objects[rail_id] = {
            "shape": "box",
            "fixed": True,
            "route_semantics": "traverse_over",
            "nominal": {
                "size_m": list(_COMPACT_RAIL_SIZE_M),
                "rgba": [0.95, 0.18, 0.04, 1.0],
                "pose": {
                    "position_m": [rail_x, 0.0, rail_height / 2.0],
                },
            },
        }
        zones[waypoint_id] = {
            "kind": "disk",
            "center_m": [landing_x, 0.0],
            "radius_m": _COMPACT_RAIL_LANDING_RADIUS_M,
        }
        waypoints.append(waypoint_id)

    finish_x = round(
        _COMPACT_RAIL_FIRST_LANDING_X_M
        + (n - 1) * _COMPACT_RAIL_SPACING_M
        + _COMPACT_RAIL_FINISH_OFFSET_M,
        4,
    )
    zones["finish"] = {
        "kind": "disk",
        "center_m": [finish_x, 0.0],
        "radius_m": _COMPACT_RAIL_FINISH_RADIUS_M,
    }
    waypoints.append("finish")
    return objects, zones, waypoints


def _author_compact_low_rails(
    world: dict[str, Any],
    task: dict[str, Any],
    *,
    count: int = 4,
) -> None:
    """Author a short deterministic sequence of low transverse rails.

    This is a reusable course profile rather than a robot- or policy-specific
    environment.  The selected capability is validated elsewhere; here the
    physical task remains ordinary ordered disk traversal with literal
    forbidden object contacts and terminal dwell.
    """
    objects, zones, waypoints = _compact_low_rail_geometry(count)
    world["shared"]["objects"] = objects
    world["shared"]["zones"] = zones
    world["train"] = {
        "variations": [],
        "curriculum": {
            "difficulty_range": [0.0, 1.0],
            "promotion": {
                "signal": "waypoint_success",
                "promote_above": 0.75,
                "demote_below": 0.50,
            },
        },
    }

    task["shared"]["control_mode"] = "waypoint_following"
    task["shared"]["goal"] = {
        "id": "complete_compact_rail_course",
        "type": "waypoint_sequence",
        "waypoints": list(waypoints),
        "success": {
            "predicate": "sequence_complete",
            "hold_s": _COMPACT_RAIL_HOLD_S,
            "tolerance_m": _COMPACT_RAIL_LANDING_RADIUS_M,
            "ordered": True,
        },
    }
    task["shared"]["contacts"]["forbidden"] = [
        ["robot:any", f"object:rail_{index + 1:02d}"]
        for index in range(len(objects))
    ]
    task["shared"]["termination"].update({
        "success_ends_episode": False,
        "episode_length_s": _COMPACT_RAIL_EPISODE_S,
    })
    task["shared"]["observations"].update({
        "height_scan": True,
        "object_relative": [],
        "region_relative": list(waypoints),
    })


def _author_slalom(
    world: dict[str, Any],
    task: dict[str, Any],
    *,
    count: int = 4,
    compound_jump: bool = False,
    cap: RobotCapability | None = None,
) -> None:
    """Author a planar weave without coupling to a robot or task name.

    The obstacles are ordinary fixed objects because WorldSpec course elements
    are intentionally one-dimensional.  Explicit waypoint zones encode the
    alternating free-space path, while the final zone remains the last ordered
    waypoint long enough for the runtime hold predicate to enforce dwell.
    """
    n = max(1, min(12, int(count)))
    spacing = 1.5
    first_x = 2.0
    obstacle_offset = 0.0
    waypoint_offset = 0.85
    box_size = [0.45, 0.45, 0.75]

    objects: dict[str, Any] = {}
    zones: dict[str, Any] = {}
    waypoints: list[str] = []
    variations: list[dict[str, Any]] = []
    for index in range(n):
        number = index + 1
        x = round(first_x + index * spacing, 4)
        box_id = f"slalom_box_{number:02d}"
        waypoint_id = f"waypoint_{number:02d}"
        side = waypoint_offset if index % 2 == 0 else -waypoint_offset
        objects[box_id] = {
            "shape": "box", "fixed": True,
            "nominal": {
                "size_m": box_size,
                "rgba": [1.0, 0.28, 0.02, 1.0],
                "pose": {"position_m": [x, obstacle_offset, box_size[2] / 2]},
            },
        }
        zones[waypoint_id] = {
            "kind": "disk", "center_m": [x, side], "radius_m": 0.45,
        }
        waypoints.append(waypoint_id)
        variations.append({
            "id": f"{box_id}_lateral_position",
            "target": (
                f"/shared/objects/{box_id}/nominal/pose/position_m/1"
            ),
            "class": "state",
            "distribution": {"kind": "uniform", "low": -0.08, "high": 0.08},
        })

    finish_x = round(first_x + n * spacing, 4)
    zones["finish"] = {
        "kind": "disk", "center_m": [finish_x, 0.0], "radius_m": 0.9,
    }
    waypoints.append("finish")
    world["shared"]["objects"] = objects
    world["shared"]["zones"] = zones
    world["train"] = {
        "variations": variations,
        "curriculum": {
            "difficulty_range": [0.0, 1.0],
            "promotion": {
                "signal": "waypoint_success", "promote_above": 0.75,
                "demote_below": 0.50,
            },
        },
    }
    task["shared"]["control_mode"] = "waypoint_following"
    task["shared"]["goal"] = {
        "id": "complete_slalom_and_stop", "type": "waypoint_sequence",
        "waypoints": waypoints,
        "success": {
            "predicate": "sequence_complete", "hold_s": 2.0,
            "tolerance_m": 0.35, "ordered": True,
        },
    }
    task["shared"]["contacts"]["forbidden"] = [
        ["robot:any", f"object:slalom_box_{index + 1:02d}"]
        for index in range(n)
    ]
    task["shared"]["observations"].update({
        "height_scan": True,
        "region_relative": waypoints,
    })
    if compound_jump:
        if cap is None:
            raise AuthoringError(
                "post-route jump authoring requires a robot capability"
            )
        try:
            cap.resolve_role("left_foot")
            cap.resolve_role("right_foot")
        except CapabilityError as exc:
            raise AuthoringError(
                f"{cap.capability_id}: post-route jump requires declared "
                "left_foot and right_foot support roles"
            ) from exc
        # The compound showcase freezes every route predicate, including the
        # finish, at the explicitly authored 0.35 m radius.  Do not let the
        # broad route-only fallback disks silently change task truth.
        for zone in zones.values():
            zone["radius_m"] = 0.35
        task["shared"]["goal"]["success"]["hold_s"] = 0.0
        task["shared"]["termination"]["episode_length_s"] = max(
            24.0,
            float(task["shared"]["termination"]["episode_length_s"]),
        )
        task["shared"]["event_sequence"] = {
            "id": "route_jump_hold",
            "phases": [
                {"id": "route", "until": {"event": "goal_complete"}},
                {
                    "id": "jump",
                    "until": {
                        "event": "bilateral_support_cycle",
                        "support_contacts": [
                            ["robot:left_foot", "world:terrain"],
                            ["robot:right_foot", "world:terrain"],
                        ],
                        "min_air_time_s": 0.06,
                        "min_height_delta_m": 0.18,
                    },
                },
                {
                    "id": "hold",
                    "terminal": True,
                    "minimum_hold_s": 2.0,
                },
            ],
        }
        task["train"]["event_phase_sampling"] = {
            "route": 0.5,
            "jump": 0.4,
            "hold": 0.1,
        }


def _grasp_contact_role(cap: RobotCapability) -> str:
    if "gripper" in cap.body_roles:
        return "gripper"
    if "left_finger" in cap.body_roles and "right_finger" in cap.body_roles:
        return "left_finger|right_finger"
    finger_roles = sorted(role for role in cap.body_roles
                          if "finger" in role or "gripper" in role)
    if finger_roles:
        return "|".join(finger_roles)
    raise CapabilityError(
        f"{cap.capability_id}: advertises grasp but has no contact-capable "
        "gripper/finger body role"
    )


def _push_contact_role(cap: RobotCapability) -> str:
    for role in ("end_effector", "gripper", "left_end_effector",
                 "right_end_effector", "wrist", "torso", "base"):
        if role in cap.body_roles:
            return role
    if cap.body_roles:
        return sorted(cap.body_roles)[0]
    raise CapabilityError(
        f"{cap.capability_id}: has no contact-capable body role for push"
    )


def _author_object_to_region(
    world: dict[str, Any], task: dict[str, Any], cap: RobotCapability,
    required: frozenset[str], *, prompt: str = "",
) -> None:
    radius = round(max(0.035, min(0.11, cap.geometry.reach_radius_m * 0.08)), 4)
    reach = max(0.25, cap.geometry.reach_radius_m)
    goal_x = round(min(4.0, reach * 0.65), 4)
    world["shared"]["zones"] = {
        "object_start": {
            "kind": "disk", "center_m": [0.0, 0.0],
            "radius_m": round(max(0.12, min(0.8, reach * 0.25)), 4),
        },
        "goal_region": {
            "kind": "box", "center_m": [goal_x, 0.0, radius * 2.0],
            "size_m": [radius * 3.0, radius * 5.0, radius * 4.0],
        },
    }
    world["shared"]["objects"] = {
        "target_object": {
            "shape": "sphere", "fixed": False,
            "nominal": {
                "radius_m": radius, "mass_kg": 0.20,
                "friction": 0.8, "restitution": 0.15,
                "pose": {"placement": "zone:object_start", "z_m": radius},
            },
        },
    }
    # A "soccer goal" / "net" prompt gets a physical goal FRAME (posts +
    # crossbar) standing on the ground at the target region — so "a ball and a
    # soccer goal" authors a visible goal, not just an invisible reward zone.
    # (The `frame` primitive was in the schema + compiler all along; no template
    # ever emitted one.)
    if any(word in prompt.casefold() for word in ("goal", "net", "soccer")):
        post_r = round(max(0.02, radius * 0.5), 4)
        opening_w = round(min(2.0, radius * 12.0), 4)
        opening_h = round(min(1.5, radius * 9.0), 4)
        world["shared"]["objects"]["target_goal"] = {
            "shape": "frame", "fixed": True,
            "nominal": {
                "opening_m": [opening_w, opening_h],
                "post_radius_m": post_r, "depth_m": 0.3,
                # Body raised so the posts' lower ends rest on the z=0 ground.
                "pose": {"position_m": [
                    goal_x, 0.0, round(opening_h / 2 + post_r, 4)]},
            },
        }
    role = (_grasp_contact_role(cap) if "grasp" in required
            else _push_contact_role(cap))
    task["shared"]["goal"] = {
        "id": "place_object", "type": "object_to_region",
        "subject": "target_object", "region": "goal_region",
        "success": {
            "predicate": "inside", "hold_s": 0.25, "tolerance_m": 0.0,
        },
    }
    task["shared"]["contacts"]["desired"] = [
        [f"robot:{role}", "object:target_object"],
    ]
    task["shared"]["termination"]["fall"] = "disabled"
    task["shared"]["observations"].update({
        "object_relative": ["target_object"],
        "region_relative": ["goal_region"],
    })
    if "end_effector_relative" in cap.supported_observations:
        observation_role = next(
            (candidate for candidate in
             ("grasp", "tool_center", "end_effector", "gripper", "wrist")
             if candidate in cap.site_roles or candidate in cap.body_roles),
            None,
        )
        if observation_role is None:
            raise CapabilityError(
                f"{cap.capability_id}: advertises end_effector_relative but "
                "has no semantic end-effector site/body role"
            )
        task["shared"]["observations"]["end_effector_relative"] = [
            observation_role,
        ]
    task["train"]["goal_sampling"] = [
        {
            "id": "object_start", "target": "object:target_object.pose",
            "distribution": {
                "kind": "uniform_in_region", "region": "object_start",
            },
        },
    ]
    world["train"]["variations"] = [
        {
            "id": "object_mass", "target": (
                "/shared/objects/target_object/nominal/mass_kg"
            ),
            "class": "model_field",
            "distribution": {"kind": "uniform", "low": 0.12, "high": 0.32},
        },
        {
            "id": "object_friction", "target": (
                "/shared/objects/target_object/nominal/friction"
            ),
            "class": "model_field",
            "distribution": {"kind": "uniform", "low": 0.5, "high": 1.1},
        },
    ]
    world["train"]["curriculum"] = {
        "difficulty_range": [0.0, 1.0],
        "promotion": {
            "signal": "task_success", "promote_above": 0.75,
            "demote_below": 0.50,
        },
    }


def _author_robot_to_region(world: dict[str, Any], task: dict[str, Any]) -> None:
    world["shared"]["zones"] = {
        "finish": {
            "kind": "disk", "center_m": [4.0, 0.0], "radius_m": 0.75,
        },
    }
    task["shared"]["goal"] = {
        "id": "reach_finish", "type": "robot_to_region", "region": "finish",
        "success": {
            "predicate": "distance_below", "hold_s": 0.5,
            "tolerance_m": 0.25,
        },
    }
    task["shared"]["observations"]["region_relative"] = ["finish"]


def _offline_author(
    prompt: str, cap: RobotCapability, *, version: str,
    parent: str | None, grounding: Sequence[str],
    descriptor_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    intent = _intent(prompt)
    required = _required_capabilities(prompt)
    world = _base_world(
        prompt, cap, required, version=version, parent=parent,
        grounding=grounding, descriptor_path=descriptor_path,
    )
    task = _base_task(
        prompt, version=version, parent=parent, grounding=grounding,
    )
    if intent == "uneven_terrain":
        _author_uneven(world, task)
    elif intent == "compact_low_rails":
        _author_compact_low_rails(
            world,
            task,
            count=_parse_count(prompt, default=4),
        )
    elif intent == "slalom":
        _author_slalom(
            world,
            task,
            count=_parse_count(prompt, default=4),
            compound_jump=_requests_post_route_jump(prompt),
            cap=cap,
        )
    elif intent == "parkour":
        _author_parkour(world, task, cap, count=_parse_count(prompt, default=3))
    elif intent == "object_to_region":
        _author_object_to_region(world, task, cap, required, prompt=prompt)
    else:
        _author_robot_to_region(world, task)

    # These fields are directly grounded by the requested semantics.  All
    # other authored parameters are explicitly reported as defaults unless
    # the caller supplied the robot selection.
    provenance: dict[str, str] = {
        "/world/shared/robot/required_capabilities": "prompt",
        "/world/shared/terrain/kind": "prompt",
        "/task/shared/control_mode": "prompt",
        "/task/shared/goal/id": "prompt",
        "/task/shared/goal/type": "prompt",
        "/task/shared/goal/subject": "prompt",
        "/task/shared/goal/region": "prompt",
        "/task/shared/goal/waypoints": "prompt",
        "/task/shared/goal/success/predicate": "prompt",
    }
    if intent == "compact_low_rails":
        # The named profile and parsed cardinality determine topology. Exact
        # geometry remains a disclosed default unless the researcher includes
        # the values in the prompt; that keeps clarification provenance honest.
        provenance["/world/shared/objects"] = "prompt"
        text = prompt.casefold()
        explicit_size = all(
            token in text for token in ("0.10", "0.60", "0.06")
        )
        explicit_positions = all(
            token in text
            for token in ("0.35", "0.85", "1.35", "1.85")
        )
        explicit_landings = all(
            token in text
            for token in ("0.65", "1.15", "1.65", "2.15")
        )
        explicit_finish = "2.55" in text and "finish" in text
        explicit_radii = "0.30" in text and "0.45" in text
        explicit_dwell = bool(re.search(
            r"\b(?:2|two)(?:\.0)?\s*(?:s|sec|second)", text
        ))
        explicit_collision_avoidance = any(token in text for token in (
            "without touching", "avoid contact", "without contact",
            "forbidden contact", "zero contact",
        ))
        for name in world["shared"]["objects"]:
            prefix = f"/world/shared/objects/{name}"
            if explicit_size:
                provenance[f"{prefix}/nominal/size_m"] = "prompt"
            if explicit_positions:
                provenance[f"{prefix}/nominal/pose/position_m"] = "prompt"
        for name in world["shared"]["zones"]:
            prefix = f"/world/shared/zones/{name}"
            if ((name == "finish" and explicit_finish)
                    or (name != "finish" and explicit_landings)):
                provenance[f"{prefix}/center_m"] = "prompt"
            if explicit_radii:
                provenance[f"{prefix}/radius_m"] = "prompt"
        if explicit_dwell:
            provenance["/task/shared/goal/success/hold_s"] = "prompt"
        if explicit_radii:
            provenance[
                "/task/shared/goal/success/tolerance_m"
            ] = "prompt"
        if explicit_collision_avoidance:
            provenance["/task/shared/contacts/forbidden"] = "prompt"
            provenance["/task/shared/contacts/desired"] = "prompt"
            provenance["/task/shared/contacts/terminate_on"] = "prompt"
        if "ordered" in text:
            provenance["/task/shared/goal/success/ordered"] = "prompt"
        if "8" in text and any(token in text for token in (
            "episode", "horizon", "seconds", "second", "sec",
        )):
            provenance[
                "/task/shared/termination/episode_length_s"
            ] = "prompt"
    if intent == "slalom":
        # The slalom cue and parsed cardinality directly determine object
        # topology.  Treating the collection itself as an omitted default
        # would ask whether to delete all obstacles, which necessarily makes
        # the requested waypoint/contact task invalid and leaves no valid
        # clarification alternative.
        provenance["/world/shared/objects"] = "prompt"
        if _requests_post_route_jump(prompt):
            provenance["/task/shared/event_sequence"] = "prompt"
            provenance["/task/train/event_phase_sampling"] = "prompt"
        text = prompt.casefold()
        explicit_size = "0.45" in text and "0.75" in text
        explicit_positions = (
            "x=" in text or "approximately at" in text
            or "roughly at" in text
        )
        explicit_color = "orange" in text
        explicit_waypoints = "waypoint" in text
        explicit_generous = "generous" in text
        explicit_finish = "finish zone" in text
        explicit_collision_avoidance = any(token in text for token in (
            "without touching", "avoid contact", "without contact",
        ))
        explicit_lateral_randomization = (
            "random" in text and "0.08" in text and "lateral" in text
        )
        explicit_dwell = bool(re.search(
            r"\b(?:2|two)(?:\.0)?\s*(?:s|sec|second)", text))
        for name in world["shared"]["objects"]:
            prefix = f"/world/shared/objects/{name}"
            if explicit_size:
                provenance[f"{prefix}/nominal/size_m"] = "prompt"
            if explicit_positions:
                provenance[f"{prefix}/nominal/pose/position_m"] = "prompt"
            if explicit_color:
                provenance[f"{prefix}/nominal/rgba"] = "prompt"
        for name in world["shared"]["zones"]:
            prefix = f"/world/shared/zones/{name}"
            if explicit_positions or explicit_waypoints:
                provenance[f"{prefix}/center_m"] = "prompt"
            if ((name == "finish" and explicit_finish)
                    or (name != "finish" and explicit_generous)):
                provenance[f"{prefix}/radius_m"] = "prompt"
        if explicit_generous:
            provenance[
                "/task/shared/goal/success/tolerance_m"
            ] = "prompt"
        if explicit_dwell:
            provenance["/task/shared/goal/success/hold_s"] = "prompt"
        if explicit_collision_avoidance:
            provenance["/task/shared/contacts/forbidden"] = "prompt"
            provenance["/task/shared/contacts/desired"] = "prompt"
            provenance["/task/shared/contacts/terminate_on"] = "prompt"
        if "ordered" in text:
            provenance["/task/shared/goal/success/ordered"] = "prompt"
        if explicit_lateral_randomization:
            for index in range(len(world["train"]["variations"])):
                prefix = f"/world/train/variations/{index}/distribution"
                provenance[f"{prefix}/low"] = "prompt"
                provenance[f"{prefix}/high"] = "prompt"
    for index, item in enumerate(world["shared"]["obstacles"]["course"]):
        provenance[f"/world/shared/obstacles/course/{index}/id"] = "prompt"
        provenance[f"/world/shared/obstacles/course/{index}/element"] = "prompt"
    for name in world["shared"]["objects"]:
        provenance[f"/world/shared/objects/{name}/shape"] = "prompt"
        provenance[f"/world/shared/objects/{name}/fixed"] = "prompt"
        provenance[
            f"/world/shared/objects/{name}/nominal/pose/placement"
        ] = "prompt"
    for name, entry in world["shared"]["terrain"].get(
            "sub_terrains", {}).items():
        provenance[
            f"/world/shared/terrain/sub_terrains/{name}/type"
        ] = "prompt"
    if intent == "object_to_region":
        provenance["/task/shared/contacts/desired"] = "prompt"
    return world, task, provenance


def _flatten_parameters(value: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            if prefix == "/world/meta" and key == "parameter_provenance":
                continue
            yield from _flatten_parameters(value[key], f"{prefix}/{key}")
        return
    if isinstance(value, list):
        # Vectors, distributions, capability sets, and contact permissions are
        # semantic parameters.  Structural lists such as a course are walked.
        if (not value or all(not isinstance(item, (dict, list)) for item in value)
                or prefix.startswith("/task/shared/contacts/")):
            yield prefix, copy.deepcopy(value)
            return
        for index, item in enumerate(value):
            yield from _flatten_parameters(item, f"{prefix}/{index}")
        return
    yield prefix, copy.deepcopy(value)


def _source_for(path: str, provenance: Mapping[str, str]) -> str:
    if path in provenance:
        return provenance[path]
    return "default"


def _load_bearing_reason(path: str) -> str | None:
    if path == "/world/shared/robot/capability_id":
        return "the embodiment and its reachable task envelope depend on this choice"
    if path == "/world/shared/robot/required_capabilities":
        return "missing capabilities make the authored interaction impossible"
    if path.startswith("/task/shared/goal/success/"):
        return "this changes the physical success semantics"
    if path.startswith("/task/shared/contacts/"):
        return "contact permission changes which physical strategies are legal"
    if "/world/shared/terrain/sub_terrains/" in path and "/nominal/" in path:
        return "terrain severity materially changes feasibility and observations"
    if "/world/shared/obstacles/course/" in path and "/nominal/" in path:
        return "course geometry materially changes the required motion"
    if path.startswith("/world/shared/objects/") and any(
            token in path for token in (
                "/mass_kg", "/friction", "/restitution", "/radius_m",
                "/size_m", "/height_m", "/opening_m", "/z_m")):
        return "object scale or physics materially changes interaction dynamics"
    if path.startswith("/world/shared/zones/") and any(
            path.endswith(suffix) for suffix in
            ("/center_m", "/size_m", "/radius_m")):
        return "region placement and scale define task reach and precision"
    if (path.startswith("/world/train/variations/")
            and "/distribution/" in path
            and path.rsplit("/", 1)[-1] in {"low", "high", "std", "clip"}):
        return "the training range materially changes the learned robustness envelope"
    return None


def _parameter_records(
    world: dict[str, Any], task: dict[str, Any], provenance: Mapping[str, str],
) -> list[ParameterRecord]:
    raw = list(_flatten_parameters(world, "/world"))
    raw.extend(_flatten_parameters(task, "/task"))
    records: list[ParameterRecord] = []
    object_path = "/world/shared/objects"
    object_reason = (
        "object count changes task topology, sensing, and contact budgets"
    )
    object_source = _source_for(object_path, provenance)
    records.append(ParameterRecord(
        path=object_path,
        value=sorted(world.get("shared", {}).get("objects", {})),
        provenance=object_source,
        load_bearing=bool(world.get("shared", {}).get("objects")),
        reason=(object_reason
                if world.get("shared", {}).get("objects") else None),
    ))
    for path, value in raw:
        # Schema/meta identity is versioning metadata, not an authored physical
        # parameter.  It is intentionally outside the underspecification set.
        if (path in {"/world/world_spec_version", "/task/task_spec_version"}
                or path.startswith("/world/meta/")
                or path.startswith("/task/meta/")):
            continue
        source = _source_for(path, provenance)
        if source not in _PROVENANCE:
            raise AuthoringError(f"invalid provenance {source!r} for {path}")
        reason = _load_bearing_reason(path)
        if (path.startswith("/task/shared/contacts/")
                and not world.get("shared", {}).get("objects")):
            # Contact permission is load-bearing for interaction tasks.  An
            # empty contact list in a pure locomotion task carries no such
            # ambiguity and must not manufacture an impossible question.
            reason = None
        records.append(ParameterRecord(
            path=path, value=value, provenance=source,
            load_bearing=reason is not None, reason=reason,
        ))
    return records


def _get_combined(
    world: dict[str, Any], task: dict[str, Any], path: str,
) -> Any:
    if path.startswith("/world/"):
        current: Any = world
        parts = path.split("/")[2:]
    elif path.startswith("/task/"):
        current = task
        parts = path.split("/")[2:]
    else:
        raise KeyError(f"combined path must begin /world or /task: {path}")
    for part in parts:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _set_combined(
    world: dict[str, Any], task: dict[str, Any], path: str, value: Any,
) -> None:
    if path.startswith("/world/"):
        current: Any = world
        parts = path.split("/")[2:]
    elif path.startswith("/task/"):
        current = task
        parts = path.split("/")[2:]
    else:
        raise KeyError(f"combined path must begin /world or /task: {path}")
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    final = parts[-1]
    if isinstance(current, list):
        current[int(final)] = copy.deepcopy(value)
    else:
        current[final] = copy.deepcopy(value)


def _physical_bounds(path: str, cap: RobotCapability) -> tuple[float, float]:
    if "slope_deg" in path:
        return 0.0, 45.0
    if "noise_range_m" in path:
        return 0.0, 0.35
    if "noise_step_m" in path:
        return 0.001, 0.10
    if path.endswith("/restitution"):
        return 0.0, 1.0
    if path.endswith("/friction"):
        return 0.0, 5.0
    if path.endswith("/mass_kg"):
        return 0.001, 500.0
    if path.endswith("/hold_s"):
        return 0.0, 10.0
    if path.endswith("/tolerance_m"):
        return 0.0, 5.0
    if "gap" in path and path.endswith("/length_m"):
        return 0.02, max(0.02, cap.geometry.max_gap_m)
    if any(token in path for token in ("height_m", "step_height_m")):
        physical = cap.geometry.max_step_height_m
        return 0.0, max(0.05, physical if physical > 0 else 2.0)
    if path.endswith(("/num_steps", "/count")):
        return 1.0, 100.0
    return 0.0, 20.0


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _distinct(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[bytes] = set()
    for value in values:
        marker = _canonical_bytes(value)
        if marker not in seen:
            result.append(value)
            seen.add(marker)
    return result


def _choice_values(
    record: ParameterRecord, cap: RobotCapability,
    capability_choices: Sequence[RobotCapability],
    world: dict[str, Any], task: dict[str, Any],
) -> list[Any]:
    path, value = record.path, record.value
    if path == "/world/shared/robot/capability_id":
        return [candidate.capability_id for candidate in capability_choices]
    if path.startswith("/task/shared/contacts/"):
        values: list[Any] = [copy.deepcopy(value), []]
        if not value:
            objects = sorted(world["shared"].get("objects", {}))
            cap_roles = sorted(cap.body_roles)
            if objects and cap_roles:
                values.append([[f"robot:{cap_roles[0]}", f"object:{objects[0]}"]])
        return _distinct(values)
    if isinstance(value, bool):
        return [False, True]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        bounds = _physical_bounds(path, cap)
        number = float(value)
        if number == 0:
            candidates = [0.0, min(bounds[1], 0.05), min(bounds[1], 0.15)]
        else:
            candidates = [
                _clamp(number * 0.65, bounds), number,
                _clamp(number * 1.35, bounds),
            ]
        if isinstance(value, int):
            candidates = [int(round(item)) for item in candidates]
        else:
            candidates = [round(item, 6) for item in candidates]
        return _distinct(candidates)
    if isinstance(value, list) and value and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value):
        if all(float(item) == 0.0 for item in value):
            offset = 0.1
            positive = list(value)
            negative = list(value)
            positive[0] = offset
            negative[0] = -offset
            return [negative, copy.deepcopy(value), positive]
        if "noise_range_m" in path:
            bounds = _physical_bounds(path, cap)
        else:
            bounds = (0.0, 100.0)
        candidates = []
        for scale in (0.75, 1.0, 1.25):
            candidates.append(
                copy.deepcopy(value) if scale == 1.0 else [
                    round(_clamp(float(item) * scale, bounds), 6)
                    for item in value
                ]
            )
        return _distinct(candidates)
    return [copy.deepcopy(value)]


def _format_value(value: Any, path: str) -> str:
    unit = ""
    if path.endswith("_m") or "_m/" in path or path.endswith("/tolerance_m"):
        unit = " m"
    elif "_kg" in path:
        unit = " kg"
    elif "deg" in path:
        unit = "°"
    elif path.endswith("_s"):
        unit = " s"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:g}{unit}"
    if isinstance(value, list) and value and all(isinstance(item, (int, float))
                                                 for item in value):
        if len(value) == 2:
            return f"{value[0]:g}–{value[1]:g}{unit}"
        return " × ".join(f"{item:g}" for item in value) + unit
    if value == []:
        return "no declared contact restriction"
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _question_prompt(path: str) -> str:
    leaf = path.rsplit("/", 1)[-1].replace("_", " ")
    if path == "/world/shared/robot/capability_id":
        return "Which installed robot capability should be used?"
    if path.endswith("/hold_s"):
        return "How long must success remain true?"
    if path.endswith("/tolerance_m"):
        return "What positional tolerance should count as success?"
    if path.startswith("/task/shared/contacts/desired"):
        return "Which robot-object contact should the task encourage?"
    if path.startswith("/task/shared/contacts/forbidden"):
        return "Which robot-object contact should be forbidden?"
    if path.startswith("/task/shared/contacts/terminate_on"):
        return "Which contact should terminate the episode?"
    return f"Choose {leaf}."


def _valid_choice(
    world: dict[str, Any], task: dict[str, Any], path: str, value: Any,
) -> bool:
    candidate_world = copy.deepcopy(world)
    candidate_task = copy.deepcopy(task)
    _set_combined(candidate_world, candidate_task, path, value)
    return (not validate_world_spec(candidate_world)
            and not validate_task_spec(candidate_task, world=candidate_world))


def _make_questions(
    records: Sequence[ParameterRecord], cap: RobotCapability,
    capability_choices: Sequence[RobotCapability],
    world: dict[str, Any], task: dict[str, Any],
) -> tuple[list[ParameterRecord], list[ClarificationQuestion]]:
    questions: list[ClarificationQuestion] = []
    updated: list[ParameterRecord] = []
    for record in records:
        if not record.load_bearing or record.provenance != "default":
            updated.append(record)
            continue
        raw_values = _choice_values(
            record, cap, capability_choices, world, task,
        )
        values = [value for value in raw_values
                  if _valid_choice(world, task, record.path, value)]
        values = _distinct(values)[:4]
        if len(values) < 2:
            raise AuthoringError(
                f"load-bearing default {record.path} has fewer than two "
                "bounded valid choices; authoring cannot omit the ambiguity"
            )
        question_id = "q_" + _hash({
            "path": record.path, "draft_value": record.value,
        })[:16]
        choices = tuple(ClarificationChoice(
            choice_id="c_" + _hash({"path": record.path, "value": value})[:12],
            label=_format_value(value, record.path), value=value,
        ) for value in values)
        default_marker = _canonical_bytes(record.value)
        try:
            default_choice = next(
                choice for choice in choices
                if _canonical_bytes(choice.value) == default_marker
            )
        except StopIteration as exc:
            raise AuthoringError(
                f"load-bearing default {record.path} is not one of its "
                "bounded choices"
            ) from exc
        question = ClarificationQuestion(
            question_id=question_id,
            parameter_path=record.path,
            prompt=_question_prompt(record.path),
            choices=choices,
            default_choice_id=default_choice.choice_id,
            default_reason=record.reason or "validated project default",
        )
        questions.append(question)
        updated.append(dataclasses.replace(record, question_id=question_id))
    return updated, questions


def _build_draft(
    prompt: str, cap: RobotCapability,
    capability_choices: Sequence[RobotCapability],
    world: dict[str, Any], task: dict[str, Any],
    provenance: Mapping[str, str], capability_source: str,
) -> AuthoringDraft:
    provenance = dict(provenance)
    prompt_text = prompt.casefold()
    if (world.get("shared", {}).get("objects")
            and any(token in prompt_text for token in
                    ("object", "ball", "cube", "block", "item"))):
        provenance.setdefault("/world/shared/objects", "prompt")
    provenance["/world/shared/robot/capability_id"] = capability_source
    if "descriptor_path" in world.get("shared", {}).get("robot", {}):
        provenance["/world/shared/robot/descriptor_path"] = "user"
    records = _parameter_records(world, task, provenance)
    # Persist one complete provenance map for both artifacts in WorldSpec's
    # normative ledger (TaskSpec intentionally has no provenance meta key).
    world["meta"]["parameter_provenance"] = {
        record.path: record.provenance for record in records
    }
    _raise_validation(world, task)
    draft_hash = _hash({
        "contract_version": AUTHORING_CONTRACT_VERSION,
        "capability": cap.to_dict(),
        "world_spec": world,
        "task_spec": task,
    })
    records, questions = _make_questions(
        records, cap, capability_choices, world, task,
    )
    missing_questions = {
        record.path for record in records
        if record.load_bearing and record.provenance == "default"
        and record.question_id is None
    }
    if missing_questions:
        raise AuthoringError(
            "load-bearing defaults were not queued: "
            f"{sorted(missing_questions)}"
        )
    pages = tuple(
        ClarificationPage(
            page=index // MAX_QUESTIONS_PER_PAGE + 1,
            questions=tuple(questions[index:index + MAX_QUESTIONS_PER_PAGE]),
        )
        for index in range(0, len(questions), MAX_QUESTIONS_PER_PAGE)
    )
    question_set_hash = _hash({
        "version": CLARIFICATION_VERSION,
        "draft_hash": draft_hash,
        "questions": [question.to_dict() for question in questions],
    })
    plan = ClarificationPlan(
        version=CLARIFICATION_VERSION, draft_hash=draft_hash,
        question_set_hash=question_set_hash, pages=pages,
    )
    report = UnderspecificationReport(
        version=CLARIFICATION_VERSION,
        parameters=tuple(records),
        defaulted_load_bearing_paths=tuple(
            record.path for record in records
            if record.load_bearing and record.provenance == "default"
        ),
    )
    return AuthoringDraft(
        version=AUTHORING_CONTRACT_VERSION, prompt=prompt,
        capability_id=cap.capability_id,
        world_spec=copy.deepcopy(world), task_spec=copy.deepcopy(task),
        underspecification_report=report, clarification_plan=plan,
        draft_hash=draft_hash,
    )


def _raise_validation(world: dict[str, Any], task: dict[str, Any]) -> None:
    errors = [f"WorldSpec: {error}" for error in validate_world_spec(world)]
    errors.extend(
        f"TaskSpec: {error}"
        for error in validate_task_spec(task, world=world)
    )
    if errors:
        raise AuthoringError("invalid authored environment:\n- " + "\n- ".join(errors))


def _semantic_float_matches(actual: Any, expected: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and abs(float(actual) - float(expected)) <= 1e-9
    )


def _semantic_vector_matches(actual: Any, expected: Sequence[float]) -> bool:
    return (
        isinstance(actual, (list, tuple))
        and len(actual) == len(expected)
        and all(
            _semantic_float_matches(item, target)
            for item, target in zip(actual, expected, strict=True)
        )
    )


def _raise_compact_low_rail_drift(
    world: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    requested: int,
) -> None:
    """Protect the named compact course from schema-valid semantic drift."""
    expected_objects, expected_zones, waypoints = (
        _compact_low_rail_geometry(requested)
    )
    shared = world.get("shared")
    objects = shared.get("objects") if isinstance(shared, Mapping) else None
    zones = shared.get("zones") if isinstance(shared, Mapping) else None
    if not isinstance(objects, Mapping) or set(objects) != set(expected_objects):
        raise AuthoringError(
            "author model changed compact low-rail object cardinality or IDs"
        )
    if not isinstance(zones, Mapping) or set(zones) != set(expected_zones):
        raise AuthoringError(
            "author model changed compact low-rail landing/finish topology"
        )

    for name, expected in expected_objects.items():
        actual = objects.get(name)
        nominal = actual.get("nominal") if isinstance(actual, Mapping) else None
        pose = nominal.get("pose") if isinstance(nominal, Mapping) else None
        if (
            not isinstance(actual, Mapping)
            or actual.get("shape") != "box"
            or actual.get("fixed") is not True
            or actual.get("route_semantics") != "traverse_over"
            or not isinstance(nominal, Mapping)
            or not _semantic_vector_matches(
                nominal.get("size_m"), expected["nominal"]["size_m"]
            )
            or not isinstance(pose, Mapping)
            or not _semantic_vector_matches(
                pose.get("position_m"),
                expected["nominal"]["pose"]["position_m"],
            )
        ):
            raise AuthoringError(
                f"author model changed compact low-rail geometry for {name!r}"
            )

    for name, expected in expected_zones.items():
        actual = zones.get(name)
        if (
            not isinstance(actual, Mapping)
            or actual.get("kind") != "disk"
            or not _semantic_vector_matches(
                actual.get("center_m"), expected["center_m"]
            )
            or not _semantic_float_matches(
                actual.get("radius_m"), expected["radius_m"]
            )
        ):
            raise AuthoringError(
                f"author model changed compact low-rail zone {name!r}"
            )

    world_train = world.get("train")
    variations = (
        world_train.get("variations")
        if isinstance(world_train, Mapping)
        else None
    )
    if variations != []:
        raise AuthoringError(
            "author model made the deterministic compact low-rail geometry vary"
        )

    task_shared = task.get("shared")
    if not isinstance(task_shared, Mapping):
        raise AuthoringError("author model dropped compact low-rail task semantics")
    goal = task_shared.get("goal")
    success = goal.get("success") if isinstance(goal, Mapping) else None
    if (
        not isinstance(goal, Mapping)
        or goal.get("type") != "waypoint_sequence"
        or goal.get("waypoints") != waypoints
        or not isinstance(success, Mapping)
        or success.get("predicate") != "sequence_complete"
        or success.get("ordered") is not True
        or not _semantic_float_matches(
            success.get("tolerance_m"), _COMPACT_RAIL_LANDING_RADIUS_M
        )
        or not _semantic_float_matches(
            success.get("hold_s"), _COMPACT_RAIL_HOLD_S
        )
    ):
        raise AuthoringError(
            "author model changed compact low-rail ordered route authority"
        )

    contacts = task_shared.get("contacts")
    expected_forbidden = [
        ["robot:any", f"object:rail_{index + 1:02d}"]
        for index in range(requested)
    ]
    if (
        not isinstance(contacts, Mapping)
        or contacts.get("forbidden") != expected_forbidden
        or contacts.get("desired") != []
        or contacts.get("terminate_on") != []
    ):
        raise AuthoringError(
            "author model changed compact low-rail contact authority"
        )

    termination = task_shared.get("termination")
    if (
        not isinstance(termination, Mapping)
        or termination.get("success_ends_episode") is not False
        or not _semantic_float_matches(
            termination.get("episode_length_s"), _COMPACT_RAIL_EPISODE_S
        )
    ):
        raise AuthoringError(
            "author model changed compact low-rail episode/dwell horizon"
        )

    observations = task_shared.get("observations")
    if (
        not isinstance(observations, Mapping)
        or observations.get("proprioception") is not True
        or observations.get("height_scan") is not True
        or observations.get("object_relative") != []
        or observations.get("region_relative") != waypoints
        or observations.get("end_effector_relative", []) != []
        or "event_sequence" in task_shared
    ):
        raise AuthoringError(
            "author model changed compact low-rail policy observation interface"
        )


def _raise_prompt_semantic_drift(
    prompt: str, world: Mapping[str, Any], task: Mapping[str, Any],
) -> None:
    """Reject schema-valid drafts that contradict explicit prompt facts.

    Structural validators can prove that three platforms are safe and
    buildable; they cannot prove that three satisfies an explicit request for
    four.  Keep this gate deliberately narrow and deterministic: explicit
    obstacle cardinality is enforced for parkour and planar slalom prompts,
    and an explicitly ordered post-route jump must retain its authored event
    program.  That gives the hybrid author a clean rejection signal and lets
    it fall back instead of silently promoting drift.
    """
    try:
        intent = _intent(prompt)
    except AuthoringError:
        # Novel model-only intents have no offline semantic oracle yet.  Their
        # schema/capability/build gates remain authoritative.
        return
    if intent not in {"parkour", "slalom", "compact_low_rails"}:
        return
    requested = _parse_count(
        prompt,
        default=4 if intent == "compact_low_rails" else 0,
    )
    if requested <= 0:
        return
    if intent == "compact_low_rails":
        _raise_compact_low_rail_drift(
            world,
            task,
            requested=requested,
        )
        return
    shared = world.get("shared")
    if intent == "slalom":
        objects = shared.get("objects") if isinstance(shared, Mapping) else None
        actual = sum(
            1 for name, item in objects.items()
            if isinstance(name, str)
            and not any(token in name for token in ("finish", "zone", "marker"))
            and isinstance(item, Mapping) and item.get("shape") == "box"
        ) if isinstance(objects, Mapping) else 0
        # A slalom is not merely N obstacles: it needs an ordered planar path.
        # The task validator checks the IDs later; this narrow oracle makes an
        # LLM draft fall back when it loses the action semantics entirely.
        if actual != requested:
            raise AuthoringError(
                "author model contradicted explicit prompt cardinality: "
                f"requested {requested} slalom obstacle(s), authored {actual}"
            )
        task_shared = task.get("shared")
        goal = task_shared.get("goal") \
            if isinstance(task_shared, Mapping) else None
        success = goal.get("success") if isinstance(goal, Mapping) else None
        waypoints = goal.get("waypoints") if isinstance(goal, Mapping) else None
        ordered_path = (
            isinstance(goal, Mapping)
            and goal.get("type") == "waypoint_sequence"
            and isinstance(waypoints, list)
            and len(waypoints) == requested + 1
            and isinstance(success, Mapping)
            and success.get("ordered") is True
        )
        if not ordered_path:
            raise AuthoringError(
                "author model dropped the requested ordered slalom path or "
                "terminal waypoint"
            )
        prompt_text = prompt.casefold()
        requests_jump = bool(re.search(
            r"\b(?:jump|leap|hop)(?:s|ed|ing)?\b", prompt_text
        ))
        orders_jump_after_route = requests_jump and bool(re.search(
            r"\b(?:then|after|afterward|once|followed\s+by|at\s+(?:the\s+)?"
            r"(?:finish|end))\b",
            prompt_text,
        ))
        event_sequence = task_shared.get("event_sequence") \
            if isinstance(task_shared, Mapping) else None
        if orders_jump_after_route:
            phases = event_sequence.get("phases") \
                if isinstance(event_sequence, Mapping) else None
            event_program_preserved = (
                isinstance(phases, list)
                and [
                    phase.get("id") if isinstance(phase, Mapping) else None
                    for phase in phases
                ] == ["route", "jump", "hold"]
                and isinstance(phases[0].get("until"), Mapping)
                and phases[0]["until"].get("event") == "goal_complete"
                and isinstance(phases[1].get("until"), Mapping)
                and phases[1]["until"].get("event")
                == "bilateral_support_cycle"
                and isinstance(
                    phases[1]["until"].get("min_height_delta_m"),
                    (int, float),
                )
                and not isinstance(
                    phases[1]["until"].get("min_height_delta_m"), bool
                )
                and phases[1]["until"]["min_height_delta_m"] >= 0.18
                and phases[2].get("terminal") is True
            )
            if not event_program_preserved:
                raise AuthoringError(
                    "author model dropped the requested post-route jump "
                    "event sequence"
                )
        dwell = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|second)", prompt_text)
        authored_hold_s = float(success.get("hold_s", 0.0))
        if orders_jump_after_route and isinstance(event_sequence, Mapping):
            try:
                authored_hold_s = float(
                    event_sequence["phases"][2]["minimum_hold_s"]
                )
            except (KeyError, IndexError, TypeError, ValueError):
                authored_hold_s = 0.0
        if dwell and authored_hold_s < float(dwell.group(1)):
            raise AuthoringError(
                "author model shortened the requested terminal dwell"
            )
        if any(token in prompt_text for token in (
                "without touching", "avoid contact", "without contact")):
            contacts = task_shared.get("contacts") \
                if isinstance(task_shared, Mapping) else None
            forbidden = contacts.get("forbidden") \
                if isinstance(contacts, Mapping) else None
            if not isinstance(forbidden, list) or len(forbidden) < requested:
                raise AuthoringError(
                    "author model dropped requested obstacle-contact exclusions"
                )
        return
    obstacles = shared.get("obstacles") if isinstance(shared, Mapping) else None
    course = obstacles.get("course") if isinstance(obstacles, Mapping) else None
    actual = 0
    if isinstance(course, list):
        actual = sum(
            1 for item in course
            if isinstance(item, Mapping) and item.get("element") == "platform"
        )
    if actual != requested:
        raise AuthoringError(
            "author model contradicted explicit prompt cardinality: "
            f"requested {requested} parkour platform(s), authored {actual}"
        )


class WorldAuthor:
    """Produce strict environment drafts with an optional injected model."""

    def __init__(self, model: AuthorModel | None = None) -> None:
        self._model = model

    def author(
        self, prompt: str, *, robot_capability_id: str | None = None,
        robot_descriptor_paths: Iterable[Path | str] = (),
        version: str = "v1", parent: str | None = None,
        grounding: Iterable[str] = (),
        grounding_context: Iterable[Mapping[str, Any]] = (),
    ) -> AuthoringDraft:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AuthoringError("prompt must be a non-empty string")
        prompt = prompt.strip()
        paths = tuple(robot_descriptor_paths)
        grounding_items = tuple(grounding)
        grounding_evidence = tuple(dict(entry) for entry in grounding_context)
        cap, candidates, capability_source = _resolve_capability(
            prompt, robot_capability_id, paths,
        )
        if (robot_capability_id is None and self._model is None
                and len(candidates) > 1):
            cap = _portable_offline_capability(
                prompt,
                candidates,
                version=version,
                parent=parent,
                grounding=grounding_items,
                descriptor_paths=paths,
            )
        descriptor_path = _descriptor_for(cap.capability_id, paths)
        if self._model is None:
            world, task, provenance = _offline_author(
                prompt, cap, version=version, parent=parent,
                grounding=grounding_items, descriptor_path=descriptor_path,
            )
        else:
            request = {
                "contract_version": AUTHORING_CONTRACT_VERSION,
                "prompt": prompt,
                "selected_robot": cap.to_dict(),
                "required_capabilities": sorted(_required_capabilities(prompt)),
                "simulator_contract": {
                    "world_spec_version": 2, "task_spec_version": 1,
                    "output_keys": [
                        "world_spec", "task_spec", "parameter_provenance",
                    ],
                },
            }
            if grounding_evidence:
                # Retrieval-only KG evidence (technique/failure-mode/paper
                # summaries with node IDs). Input context only: the model's
                # OUTPUT key set stays closed, and the meta.grounding ledger
                # is written locally below from the retrieved node IDs.
                request["kg_grounding"] = list(grounding_evidence)
            try:
                generated = self._model.generate_authoring(request)
            except Exception as exc:
                raise AuthoringError(f"author model failed: {exc}") from exc
            if not isinstance(generated, Mapping):
                raise AuthoringError("author model output must be an object")
            unknown = set(generated) - {
                "world_spec", "task_spec", "parameter_provenance",
            }
            if unknown:
                raise AuthoringError(
                    f"author model output has unknown keys {sorted(unknown)}"
                )
            world = copy.deepcopy(generated.get("world_spec"))
            task = copy.deepcopy(generated.get("task_spec"))
            raw_provenance = generated.get("parameter_provenance", {})
            if not isinstance(raw_provenance, Mapping):
                raise AuthoringError(
                    "author model parameter_provenance must be an object"
                )
            provenance = dict(raw_provenance)
            if not isinstance(world, dict) or not isinstance(task, dict):
                raise AuthoringError(
                    "author model must return world_spec and task_spec objects"
                )
            if grounding_items:
                # The local retriever, not the language model, is authoritative
                # for provenance. This also guarantees that explicit
                # ``paper:<id>`` pins verified by the KG survive even when the
                # model emits a non-empty but unrelated grounding list.
                for spec in (world, task):
                    meta = spec.get("meta")
                    if isinstance(meta, dict):
                        meta["grounding"] = list(grounding_items)
            robot = world.get("shared", {}).get("robot", {})
            if robot.get("capability_id") != cap.capability_id:
                raise AuthoringError(
                    "author model changed the resolved robot capability from "
                    f"{cap.capability_id!r} to {robot.get('capability_id')!r}"
                )
            declared_required = robot.get("required_capabilities", [])
            if not isinstance(declared_required, list):
                raise AuthoringError(
                    "author model robot.required_capabilities must be a list"
                )
            omitted = sorted(
                _required_capabilities(prompt) - set(declared_required)
            )
            if omitted:
                raise AuthoringError(
                    "author model omitted prompt-required robot capabilities "
                    f"{omitted}"
                )
            try:
                cap.require(_required_capabilities(prompt))
            except CapabilityError as exc:
                raise AuthoringError(str(exc)) from exc
        _raise_prompt_semantic_drift(prompt, world, task)
        return _build_draft(
            prompt, cap, candidates, world, task, provenance,
            capability_source,
        )


def author_environment(
    prompt: str, *, model: AuthorModel | None = None,
    robot_capability_id: str | None = None,
    robot_descriptor_paths: Iterable[Path | str] = (),
    version: str = "v1", parent: str | None = None,
    grounding: Iterable[str] = (),
    grounding_context: Iterable[Mapping[str, Any]] = (),
) -> AuthoringDraft:
    """Convenience wrapper around :class:`WorldAuthor`."""
    return WorldAuthor(model).author(
        prompt, robot_capability_id=robot_capability_id,
        robot_descriptor_paths=robot_descriptor_paths, version=version,
        parent=parent, grounding=grounding,
        grounding_context=grounding_context,
    )


def default_clarification_submission(
    draft: AuthoringDraft, *, timeout: bool,
) -> ClarificationSubmission:
    """Select every disclosed default for headless/timeout operation."""
    source = "timeout_default" if timeout else "default"
    return ClarificationSubmission(
        version=CLARIFICATION_VERSION,
        draft_hash=draft.draft_hash,
        question_set_hash=draft.clarification_plan.question_set_hash,
        answers=tuple(
            ClarificationAnswer(
                question_id=question.question_id,
                choice_id="system_default", source=source,
            )
            for question in draft.clarification_plan.questions
        ),
    )


def apply_clarifications(
    draft: AuthoringDraft, submission: ClarificationSubmission,
) -> AppliedAuthoring:
    """Apply answers transactionally, revalidating after every page."""
    plan = draft.clarification_plan
    if submission.version != CLARIFICATION_VERSION:
        raise StaleClarificationError(
            f"clarification version {submission.version} is stale; "
            f"expected {CLARIFICATION_VERSION}"
        )
    if submission.draft_hash != draft.draft_hash:
        raise StaleClarificationError(
            "clarification draft hash does not match the current draft"
        )
    if submission.question_set_hash != plan.question_set_hash:
        raise StaleClarificationError(
            "clarification question-set hash does not match the current draft"
        )
    by_id = {question.question_id: question for question in plan.questions}
    answers: dict[str, ClarificationAnswer] = {}
    for answer in submission.answers:
        if answer.question_id not in by_id:
            raise AuthoringError(
                f"unknown clarification question {answer.question_id!r}"
            )
        if answer.question_id in answers:
            raise AuthoringError(
                f"duplicate clarification answer {answer.question_id!r}"
            )
        if (answer.choice_id == "system_default"
                and answer.source not in {"default", "timeout_default"}):
            raise AuthoringError(
                "system_default answers must record default or "
                "timeout_default provenance"
            )
        if (answer.choice_id != "system_default"
                and answer.source != "user"):
            raise AuthoringError(
                "a non-default choice must record user provenance"
            )
        answers[answer.question_id] = answer

    missing = sorted(set(by_id) - set(answers))
    if missing:
        raise AuthoringError(
            "clarification submission is incomplete; missing answers for "
            f"{missing}"
        )

    world = copy.deepcopy(draft.world_spec)
    task = copy.deepcopy(draft.task_spec)
    selected_records: list[dict[str, Any]] = []
    selected_sources: dict[str, str] = {}
    for page in plan.pages:
        for question in page.questions:
            answer = answers[question.question_id]
            choice_id = (question.default_choice_id
                         if answer.choice_id == "system_default"
                         else answer.choice_id)
            choice = next(
                (candidate for candidate in question.choices
                 if candidate.choice_id == choice_id),
                None,
            )
            if choice is None:
                raise AuthoringError(
                    f"question {question.question_id!r} has no choice "
                    f"{answer.choice_id!r}"
                )
            _set_combined(
                world, task, question.parameter_path, choice.value,
            )
            selected_sources[question.parameter_path] = answer.source
            selected_records.append({
                "question_id": question.question_id,
                "parameter_path": question.parameter_path,
                "selected_choice_id": answer.choice_id,
                "resolved_choice_id": choice.choice_id,
                "selected_value": copy.deepcopy(choice.value),
                "source": answer.source,
                "page": page.page,
                "default_reason": question.default_reason,
            })
        # This is deliberately page-scoped: a bad early answer never waits for
        # later pages before surfacing its complete schema violation list.
        _raise_validation(world, task)

    provenance = dict(world["meta"].get("parameter_provenance", {}))
    provenance.update(selected_sources)
    world["meta"]["parameter_provenance"] = provenance
    _raise_validation(world, task)
    updated_records = tuple(
        dataclasses.replace(
            record,
            value=(sorted(world.get("shared", {}).get("objects", {}))
                   if record.path == "/world/shared/objects"
                   else copy.deepcopy(
                       _get_combined(world, task, record.path)
                   )),
            provenance=selected_sources.get(record.path, record.provenance),
        )
        for record in draft.underspecification_report.parameters
    )
    answered = set(answers)
    report = UnderspecificationReport(
        version=CLARIFICATION_VERSION,
        parameters=updated_records,
        defaulted_load_bearing_paths=tuple(
            record.path for record in updated_records
            if record.load_bearing and record.provenance == "default"
            and record.question_id not in answered
        ),
    )
    result_hash = _hash({
        "contract_version": AUTHORING_CONTRACT_VERSION,
        "world_spec": world, "task_spec": task,
    })
    ledger = {
        "clarification_version": CLARIFICATION_VERSION,
        "draft_hash": draft.draft_hash,
        "question_set_hash": plan.question_set_hash,
        "result_hash": result_hash,
        "pages": [page.to_dict() for page in plan.pages],
        "answers": selected_records,
        "unanswered_question_ids": [
            question.question_id for question in plan.questions
            if question.question_id not in answered
        ],
    }
    return AppliedAuthoring(
        version=AUTHORING_CONTRACT_VERSION,
        initial_draft_hash=draft.draft_hash, result_hash=result_hash,
        world_spec=world, task_spec=task,
        underspecification_report=report,
        clarification_ledger=ledger,
    )


__all__ = [
    "AUTHORING_CONTRACT_VERSION", "CLARIFICATION_VERSION",
    "MAX_QUESTIONS_PER_PAGE", "AppliedAuthoring", "AuthorModel",
    "AuthoringDraft", "AuthoringError", "ClarificationAnswer",
    "ClarificationChoice", "ClarificationPage", "ClarificationPlan",
    "ClarificationQuestion", "ClarificationSubmission", "ParameterRecord",
    "StaleClarificationError", "UnderspecificationReport", "WorldAuthor",
    "apply_clarifications", "author_environment",
    "default_clarification_submission",
]

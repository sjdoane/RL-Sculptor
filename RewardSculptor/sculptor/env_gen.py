"""Goal-conditioned env-spec generation (§RL_SCULPTOR_AUDIT, env
generalization 2/4).

At a project's first run, the behavior goal is turned into a validated
env spec — the same task-env-alignment audit a human runs before
training (commands vs in-place skills, terminations vs the target
behavior, episode length vs skill timescale, RSI + its required
early-termination pairing, pushes, friction, exploration). Mirrors the
loop-4c seed-reward discipline: structured LLM output (pydantic-
constrained via ``messages.parse``), the full ``validate_env_spec``
gate on the result, exactly one retry carrying the complete violation
list, and every failure path falling back to task defaults — the run
never blocks on generation.

Cost bound: at most 2 LLM calls, once per project lifetime (only when
the project has no env spec and no explicit profile)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from sculptor.env_spec import (
    ENV_SPEC_VERSION,
    _PUSH_INTERVAL_ENVELOPE,
    _PUSH_ANGULAR_MAX,
    _PUSH_LINEAR_MAX,
    _SHARED_SCALARS,
    _TRAIN_RANGES,
    _TRAIN_SCALARS,
    validate_env_spec,
)
from sculptor.llm import log_llm_call, model_for, response_text_blocks

MODEL_ID = model_for("env_gen")
MAX_TOKENS = 8192

_PROMPT_PATH = Path(__file__).parent / "prompts" / "gen_env_spec.md"


# ── Pydantic shapes (what the LLM is constrained to emit) ──────────────────
class _PushEventsModel(BaseModel):
    enabled: bool
    interval_s: Optional[list[float]] = Field(
        default=None, description="[lo, hi] seconds between pushes")
    linear_mps: Optional[float] = Field(
        default=None, description="max |linear push velocity| per axis")
    angular_radps: Optional[float] = Field(
        default=None, description="max |angular push velocity| per axis")


class _SharedModel(BaseModel):
    """The evaluated task. Omitted field == keep the task default."""

    zero_velocity_commands: Optional[bool] = None
    orientation_termination_deg: Optional[float] = None
    episode_length_s: Optional[float] = None
    push_events: Optional[_PushEventsModel] = None


class _TrainModel(BaseModel):
    """Train-only curricula — never applied to evaluation rollouts."""

    reset_height_offset_m: Optional[list[float]] = None
    reset_vertical_velocity_mps: Optional[list[float]] = None
    reset_horizontal_velocity_mps: Optional[list[float]] = None
    reset_joint_position_offset_rad: Optional[list[float]] = None
    reset_joint_velocity_radps: Optional[list[float]] = None
    # Root-orientation reset offsets (radians) — §REFERENCE_TRAJECTORY_PLAN
    # §8 part 1. Not part of the original goal-conditioned generation
    # prompt (a get-up/lying-start curriculum is reference-clip-derived,
    # not something the LLM invents from a behavior goal alone), but the
    # field must exist for schema field-parity
    # (test_generator_models_cover_the_schema_exactly) and there is no
    # harm in leaving it generator-reachable (omitted by the LLM in
    # practice unless the prompt is later extended to mention it).
    reset_pitch_offset_rad: Optional[list[float]] = None
    reset_roll_offset_rad: Optional[list[float]] = None
    # Per-joint reference-posture reset target + its noise magnitude —
    # same rationale: schema field-parity only, reference-clip-derived
    # in practice (see sculptor/reference.py), not LLM-generated today.
    reset_joint_pos_target: Optional[list[float]] = None
    reset_joint_pos_noise_rad: Optional[float] = None
    # §DeepMimic phase RSI (arXiv 1804.02717) — downsampled reference [K][J]
    # trajectories. Schema field-parity only: DERIVED by sculptor/reference.py
    # (a clip's actual motion), never invented by the LLM from a goal alone.
    reset_joint_pos_trajectory: Optional[list[list[float]]] = None
    reset_joint_vel_trajectory: Optional[list[list[float]]] = None
    # §sim2real physics domain randomization (arXiv 1710.06537 / 2107.04034 /
    # 2212.03238) — multiplicative scales about the nominal model value (friction
    # is an absolute coefficient). A crash-safe subset (mass/damping/armature) is
    # applied to EVERY train run by default; these let the generator widen or add
    # axes for goals that need more robustness (keep MODERATE — 2508.08241).
    body_mass_scale_range: Optional[list[float]] = None
    com_offset_m: Optional[float] = None
    pd_kp_scale_range: Optional[list[float]] = None
    pd_kd_scale_range: Optional[list[float]] = None
    motor_strength_scale_range: Optional[list[float]] = None
    joint_damping_scale_range: Optional[list[float]] = None
    joint_armature_scale_range: Optional[list[float]] = None
    body_friction_range: Optional[list[float]] = None
    # Disables the task's standard fell-over/bad-orientation termination
    # for TRAIN only (§ get-up RSI fix, 2026-07-09) — a lying-start
    # reset trips that termination on itself, killing get-up training
    # before it can begin. Schema field-parity only (like the orientation
    # keys above): reference-clip-derived in practice (see
    # sculptor/reference.py's get-up branch), not something the LLM
    # invents from a behavior goal alone.
    fell_over_termination: Optional[bool] = None
    min_base_height_termination_m: Optional[float] = None
    friction_range: Optional[list[float]] = None
    entropy_coef_scale: Optional[float] = None
    push_events: Optional[_PushEventsModel] = None


class _EnvSpecModel(BaseModel):
    reasoning: str = Field(
        description="short paragraph: which checklist items drove each "
                    "deviation from the task defaults")
    shared: _SharedModel = Field(default_factory=_SharedModel)
    train: _TrainModel = Field(default_factory=_TrainModel)


def _model_to_spec(m: _EnvSpecModel, *, behavior_goal: str) -> dict:
    """Emitted model → env-spec dict. Omitted (None) fields are dropped
    so the spec only records deliberate deviations."""
    def _clean(section: BaseModel) -> dict:
        out = {}
        for k, v in section.model_dump().items():
            if v is None:
                continue
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if vv is not None}
            out[k] = v
        return out

    return {
        "env_spec_version": ENV_SPEC_VERSION,
        "meta": {
            "source": "generated",
            "behavior_goal": " ".join(str(behavior_goal).split())[:900],
            "reasoning": str(m.reasoning)[:2000],
        },
        "shared": _clean(m.shared),
        "train": _clean(m.train),
    }


def _bounds_block() -> str:
    """Hard bounds rendered from the validator's own tables — the single
    source of truth, so prompt and gate can never drift."""
    lines = ["HARD BOUNDS (a value outside these fails validation):"]
    for k, (lo, hi) in sorted(_SHARED_SCALARS.items()):
        lines.append(f"  shared.{k}: [{lo:g}, {hi:g}]")
    for k, (lo, hi) in sorted(_TRAIN_SCALARS.items()):
        lines.append(f"  train.{k}: [{lo:g}, {hi:g}]")
    for k, (lo, hi) in sorted(_TRAIN_RANGES.items()):
        lines.append(
            f"  train.{k}: [lo, hi] pair, both within [{lo:g}, {hi:g}]")
    lines.append(
        f"  push_events.interval_s within "
        f"[{_PUSH_INTERVAL_ENVELOPE[0]:g}, {_PUSH_INTERVAL_ENVELOPE[1]:g}]; "
        f"linear_mps <= {_PUSH_LINEAR_MAX:g}; "
        f"angular_radps <= {_PUSH_ANGULAR_MAX:g}")
    return "\n".join(lines)


def _user_content(behavior_goal: str, task_id: str) -> str:
    goal = " ".join(str(behavior_goal).split())[:900]
    return (
        f"BEHAVIOR GOAL: {goal}\n"
        f"TASK ID: {task_id} (the robot is named in the id; scale "
        f"height thresholds to it)\n\n"
        f"{_bounds_block()}\n\n"
        "Emit the env spec JSON. Omit every field where the task "
        "default is right for this goal."
    )


def generate_env_spec(
    *,
    behavior_goal: str,
    task_id: str,
    client=None,
) -> dict:
    """Generate + validate an env spec from the behavior goal. Raises
    on any failure (no client, parse failure twice, validation failure
    twice) — the CALLER decides that failure means task defaults."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=2, timeout=240.0)
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    user = _user_content(behavior_goal, task_id)

    def _one_call(content: str) -> dict:
        resp = client.messages.parse(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
            output_format=_EnvSpecModel,
        )
        log_llm_call(
            "env_gen", MODEL_ID, system=system_prompt, user=content,
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None))
        return _model_to_spec(resp.parsed_output, behavior_goal=behavior_goal)

    # Attempt 1 → full-violation feedback → attempt 2 → raise.
    call_failed = False
    try:
        spec = _one_call(user)
        errors = validate_env_spec(spec)
    except Exception as e:  # noqa: BLE001 — parse/API failure counts as attempt 1
        spec, errors, call_failed = None, [f"{type(e).__name__}: {e}"], True
    if not errors:
        return spec
    preamble = (
        "The previous attempt failed before producing a parseable spec"
        if call_failed else
        "Your previous spec failed validation — fix ALL of these")
    retry = (
        f"{user}\n\n{preamble} and re-emit the full spec:\n  - "
        + "\n  - ".join(errors)
    )
    spec = _one_call(retry)
    errors = validate_env_spec(spec)
    if errors:
        raise ValueError(
            "generated env spec failed validation twice:\n  - "
            + "\n  - ".join(errors))
    return spec

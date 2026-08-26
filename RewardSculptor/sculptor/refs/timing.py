"""Simulation and control timestep policy, with the literature behind it.

These two numbers are distinct and are routinely conflated:

  * the **physics timestep** — how finely the integrator resolves contact and
    actuator dynamics;
  * the **control timestep** — how often the policy emits an action. In mjlab
    this is `physics_dt * decimation`, exposed at runtime as `env.step_dt`.

They were invisible in this codebase until a Tier-D certification went wrong
because of them, twice: once by clocking a reference off the TRAINING BUDGET
(2000 PPO updates read as env steps), and once by re-deriving the control rate
from a training statistic that turned out to be mean per-env progress rather
than episode duration. Hence this module — the rates are stated, sourced, and
checkable rather than assumed at each call site.

What the literature does
------------------------
**Control at ~50 Hz is the sim2real convention for legged/humanoid RL.**
MuJoCo Playground issues control at a base frequency of 50 Hz specifically so
a policy runs identically in sim and on hardware, and Booster Gym's humanoid
locomotion stack matches it. This is the rate our mjlab G1 task uses.

**Physics an order of magnitude finer, via decimation.** 200-500 Hz is the
common band: mjlab's velocity task uses `timestep=0.005` (200 Hz) with
`decimation=4`; Genesis uses 1/200 s; MuJoCo whole-body MPPI work runs
dt=0.002 (500 Hz). Decimation, not a slower integrator, is what buys the
cheap policy rate — contact is still resolved finely.

**On hardware the loop underneath is faster still.** The low-level PD
controller typically runs at ~1 kHz while the RL policy runs at 50 Hz. The
policy emits joint TARGETS, not torques, so it does not need to run at the
torque loop's rate.

**Stiff elements move the floor a lot.** Series-elastic actuators are the
standard counterexample: the spring adds a fast mode, and an explicit
integrator needs `dt << 1/omega_n` to stay stable, so SEA force-control
studies discretize far finer than a rigid-actuator humanoid does (one uses a
1e-5 s fixed step). This is also why Brax's spring/positional backends need
smaller steps than MuJoCo: MuJoCo's soft-constraint solver is implicit in the
velocity update and tolerates steps that would blow up an explicit spring
model. A rigid-actuator G1 in MuJoCo at 200 Hz is NOT evidence that 200 Hz is
enough for an SEA model of the same robot.

Practical consequence for this project: the current 200 Hz / 50 Hz split is
the literature default and is appropriate for the rigid-actuator G1 we
simulate. If an actuator model with series elasticity is ever added, the
physics rate has to be revisited per-task — `validate_timing` flags that
rather than letting it pass silently.

Sources
-------
  MuJoCo Playground (50 Hz unified sim/hardware control)
    https://arxiv.org/html/2503.04613v1
  Booster Gym: End-to-End RL for Humanoid Locomotion (50 Hz policy, 1 kHz PD)
    https://arxiv.org/pdf/2506.15132
  TARC: Time-Adaptive Robotic Control (fixed-frequency 50 Hz baseline)
    https://arxiv.org/pdf/2510.23176
  Force control of SEAs via sliding mode (1e-5 s fixed integration step)
    https://www.degruyterbrill.com/document/doi/10.1515/eng-2025-0147/html
  Pratt & Williamson, Series Elastic Actuators (why the fast mode exists)
    https://fab.cba.mit.edu/classes/865.15/people/rebecca.kleinberger/assets/papers/SEA_Pratt.pdf
"""
from __future__ import annotations

from dataclasses import dataclass

#: The sim2real convention for legged/humanoid RL policy rate (Hz).
CONVENTIONAL_CONTROL_HZ = 50.0

#: Physics rates seen across current legged-RL stacks (Hz). Below the low end
#: contact resolution degrades; above the high end you are paying for detail a
#: rigid-actuator model does not need.
PHYSICS_HZ_BAND = (200.0, 500.0)

#: Control rates that still deploy sanely. Under ~20 Hz the policy cannot
#: reject disturbances between actions; over ~200 Hz it is doing the low-level
#: controller's job and will not transfer to a 50 Hz hardware loop.
CONTROL_HZ_BAND = (20.0, 200.0)

#: Physics rate below which a SERIES-ELASTIC actuator model is unsafe to trust.
#: The spring's natural mode needs dt << 1/omega_n; SEA force-control studies
#: run far finer than rigid-actuator humanoids. Advisory, not a hard gate — the
#: real number depends on the spring constant and reflected inertia.
SEA_MIN_PHYSICS_HZ = 1000.0


@dataclass(frozen=True)
class SimTiming:
    """A task's physics/control rates, derived the one canonical way."""

    physics_dt: float
    decimation: int

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def physics_hz(self) -> float:
        return 1.0 / self.physics_dt if self.physics_dt > 0 else 0.0

    @property
    def control_hz(self) -> float:
        return 1.0 / self.control_dt if self.control_dt > 0 else 0.0

    def steps_for(self, duration_s: float) -> int:
        """Control steps spanning `duration_s` — the count a phase clock or an
        `episode_length_s` cap has to agree with."""
        return max(1, round(duration_s * self.control_hz))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "physics_dt": self.physics_dt,
            "physics_hz": round(self.physics_hz, 3),
            "decimation": self.decimation,
            "control_dt": round(self.control_dt, 6),
            "control_hz": round(self.control_hz, 3),
        }


#: mjlab's G1 velocity task, read from
#: `mjlab/tasks/velocity/velocity_env_cfg.py` (`MujocoCfg.timestep=0.005`,
#: `decimation=4`) rather than inferred from a training statistic.
MJLAB_G1_VELOCITY = SimTiming(physics_dt=0.005, decimation=4)


def timing_for_task(task_id: str) -> SimTiming | None:
    """Resolve a task's REAL rates from its mjlab env cfg.

    This is the authority, and it is not the MJCF. mjlab sets
    `MujocoCfg.timestep` on the compiled model, so a robot XML that declares no
    `<option timestep=...>` (the Unitree G1 declares none, and therefore
    compiles at MuJoCo's 0.002 s default) still trains at whatever the task
    says — 0.005 s for the G1 velocity task. Reading the XML and reporting that
    as "the physics timestep" is off by 2.5x.

    Returns None when mjlab is unavailable or the task does not declare both
    rates; callers should treat that as "unknown", never as a default.
    """
    if not task_id:
        return None
    try:
        from mjlab.tasks.registry import load_env_cfg

        ec = load_env_cfg(task_id)
        sim = getattr(ec, "sim", None)
        mj = getattr(sim, "mujoco", None) if sim is not None else None
        physics_dt = float(getattr(mj, "timestep", 0.0) or 0.0)
        decimation = int(getattr(ec, "decimation", 0) or 0)
    except Exception:  # noqa: BLE001 — mjlab missing or task unknown
        return None
    if physics_dt <= 0 or decimation < 1:
        return None
    return SimTiming(physics_dt=physics_dt, decimation=decimation)


def validate_timing(
    timing: SimTiming,
    *,
    reference_fps: float | None = None,
    reference_duration_s: float | None = None,
    n_phase_targets: int | None = None,
    series_elastic: bool = False,
) -> list[str]:
    """Advisory findings about a task's timing. Empty list == nothing to say.

    Returns strings rather than raising: timing is a design choice, and a task
    that deliberately runs off-convention should be told so, not blocked."""
    out: list[str] = []
    if timing.physics_dt <= 0 or timing.decimation < 1:
        out.append(
            f"invalid timing: physics_dt={timing.physics_dt}, "
            f"decimation={timing.decimation}")
        return out

    lo, hi = PHYSICS_HZ_BAND
    if timing.physics_hz < lo:
        out.append(
            f"physics {timing.physics_hz:.0f} Hz is below the {lo:.0f}-{hi:.0f} Hz "
            "band current legged-RL stacks use; contact resolution degrades")
    if series_elastic and timing.physics_hz < SEA_MIN_PHYSICS_HZ:
        out.append(
            f"series-elastic actuators need a far finer step than "
            f"{timing.physics_hz:.0f} Hz (the spring's fast mode needs "
            f"dt << 1/omega_n; SEA studies run >= {SEA_MIN_PHYSICS_HZ:.0f} Hz). "
            "MuJoCo's implicit solver does not rescue an explicit spring model")

    clo, chi = CONTROL_HZ_BAND
    if not (clo <= timing.control_hz <= chi):
        out.append(
            f"control {timing.control_hz:.0f} Hz is outside the deployable "
            f"{clo:.0f}-{chi:.0f} Hz band (convention is "
            f"{CONVENTIONAL_CONTROL_HZ:.0f} Hz)")

    # A reference can only be tracked at the rate the policy actually acts.
    if reference_duration_s and n_phase_targets:
        steps = timing.steps_for(reference_duration_s)
        if n_phase_targets > steps:
            out.append(
                f"{n_phase_targets} phase targets over {reference_duration_s:.2f} s "
                f"but only {steps} control steps fit — some targets are never "
                "visited; reduce n_phase_targets or raise the control rate")
    if reference_fps and timing.control_hz < reference_fps / 2.0:
        out.append(
            f"control {timing.control_hz:.0f} Hz cannot represent a "
            f"{reference_fps:.0f} fps reference above "
            f"{timing.control_hz / 2:.0f} Hz (Nyquist); fast transients such "
            "as a kick or foot strike will alias")
    return out

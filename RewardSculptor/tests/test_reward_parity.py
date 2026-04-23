"""tests/test_reward_parity.py — bit-level parity check for the legacy
sculptor/reward.py (AME456 quadruped). Skipped in the standalone venv (which
has no jax/mjx/brax) via `importorskip`. To run, execute with a Python that
has the AME456 deps installed and REWARD_SCULPTOR_PATH pointing at this repo.


Design
------
1. The env now delegates reward computation to `sculptor.reward.compute_reward`.
2. This test also defines `_reference_reward_v0_preRefactor` — a literal,
   standalone transcription of the in-env reward block as it existed at
   quadruped_mjx_env.py:579-699 BEFORE the sculptor refactor.
3. We run the same 100-step trajectory (fixed seed, fixed k_s) twice:
     (a) with the env calling `sculptor.reward.compute_reward` (v0)
     (b) with the env calling `_reference_reward_v0_preRefactor`
   The two reward functions take identical `(state, action, next_state, info)`
   inputs from the env, so any divergence is attributable to the port.
4. We assert |reward_v0[i] - reward_ref[i]| < 1e-6 for all 100 steps.

The reference function and v0 both compute the reward with IEEE-754 float32
operations in the same order, so parity is bit-exact in practice — we keep
the 1e-6 tolerance that the user requested.

Runnable standalone: `python tests/test_reward_parity.py`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Skip this whole module if the AME456 quadruped dependency stack is not
# installed in the current interpreter. Keeps `uv run pytest tests/` green
# in the standalone sculptor venv (which has no jax/mjx/brax).
jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")


# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_RS_ROOT = _HERE.parent
_AME_FILES = (_RS_ROOT.parent.parent / "AME456" / "files").resolve()

if not _AME_FILES.is_dir():
    pytest.skip(
        f"AME456 quadruped source not found at {_AME_FILES}; skipping quadruped "
        "parity test.",
        allow_module_level=True,
    )

if str(_AME_FILES) not in sys.path:
    sys.path.insert(0, str(_AME_FILES))
if str(_RS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RS_ROOT))
os.environ.setdefault("REWARD_SCULPTOR_PATH", str(_RS_ROOT))

import quadruped_mjx_env as env_mod  # noqa: E402
from sculptor.reward import compute_reward as _compute_reward_v0  # noqa: E402


# ── Literal pre-refactor reward (verbatim from quadruped_mjx_env.py:579–699) ─
def _reference_reward_v0_preRefactor(state, action, next_state, info):
    """Bit-identical transcription of the pre-refactor in-env reward block.

    Operations, constants, and accumulation order are preserved so IEEE-754
    float32 rounding matches the pre-refactor env bit-for-bit.
    """
    del action  # unused in v0
    tilt                   = info["tilt"]
    roll_rate              = info["roll_rate"]
    pitch_rate             = info["pitch_rate"]
    vx                     = info["vx"]
    vy                     = info["vy"]
    body_vz                = info["body_vz"]
    n_feet_on              = info["n_feet_on"]
    all_feet_off           = info["all_feet_off"]
    height_gain            = info["height_gain"]
    peak_gain              = info["peak_gain"]
    current_cycle_max_tilt = info["current_cycle_max_tilt"]
    cycle_completed        = info["cycle_completed"]
    action_rate            = info["action_rate"]
    xy_drift               = info["xy_drift"]
    tau_motor              = info["tau_motor"]
    terminated             = info["terminated"]
    done                   = info["done"]

    ps = state
    new_max_cycle_peak_gain = next_state.max_cycle_peak_gain
    JUMP_REF_HEIGHT = 0.03  # matches env module constant

    reward = jp.float32(0.0)

    # 1. tilt²
    reward -= 2.0 * tilt * tilt
    # 2. angvel damping
    reward -= 0.2 * (roll_rate + pitch_rate)
    # 2b. horizontal vel²
    reward -= 2.0 * (vx ** 2 + vy ** 2)
    # 3. grounded + level upward vel
    reward += jp.where(
        (n_feet_on >= 3) & (tilt < jp.float32(0.524)),
        5.0 * jp.maximum(0.0, body_vz), 0.0)
    # 4. quadratic flight height
    height_gain_capped = jp.minimum(jp.maximum(0.0, height_gain), 0.4)
    reward += jp.where(
        all_feet_off, 150.0 * height_gain_capped ** 2, 0.0)
    # 5. cycle completion bonus
    height_scale = jp.clip(
        (peak_gain / JUMP_REF_HEIGHT) ** 2, 0.0, 16.0)
    cycle_upright = jp.maximum(
        0.0, 1.0 - current_cycle_max_tilt / jp.float32(0.436))
    reward += jp.where(
        cycle_completed, 100.0 * height_scale * cycle_upright, 0.0)
    # 6. xy drift dead-zone
    drift_excess = jp.maximum(0.0, xy_drift - 0.15)
    reward -= 3.0 * drift_excess ** 2
    # 7. action smoothness
    reward -= 0.05 * action_rate
    # 8. torque cost
    reward -= 0.0001 * jp.sum(tau_motor ** 2)
    # termination clamp
    reward = jp.where(terminated, jp.float32(-1000.0), reward)
    # 9. end-of-episode max peak
    reward += jp.where(
        done & ~terminated, 3000.0 * new_max_cycle_peak_gain, 0.0)
    # 9b. end-of-episode successive jumps
    reward += jp.where(
        done & ~terminated,
        100.0 * jp.minimum(ps.jump_count.astype(jp.float32), 5.0),
        0.0)
    # 9c. end-of-episode mean peak
    mean_peak = jp.where(
        ps.jump_count > 0,
        ps.cycle_peak_sum / jp.maximum(
            ps.jump_count.astype(jp.float32), 1.0),
        0.0)
    reward += jp.where(
        done & ~terminated, 500.0 * mean_peak, 0.0)
    # 10. end-of-episode never-jumped penalty
    reward += jp.where(
        done & ~terminated
        & (new_max_cycle_peak_gain <= jp.float32(0.001)),
        -500.0, 0.0)

    return reward, {}


# ── Trajectory runner ───────────────────────────────────────────────────────
def _run_trajectory(
    reward_fn,
    n_steps: int = 100,
    action_seed: int = 42,
    reset_seed: int = 0,
    fixed_ks: float = 20.0,
):
    """Run a fixed trajectory with `reward_fn` installed in env_mod.

    Returns a list of per-step scalar rewards (as Python floats).

    Swaps `env_mod._compute_reward` before constructing the env so the JAX
    jit trace captures the desired function. A fresh env + fresh jit is
    used on each call so the trace is not reused across reward_fn swaps.
    """
    env_mod._compute_reward = reward_fn  # type: ignore[attr-defined]

    env = env_mod.QuadrupedJumpMJXEnv(ks_range=(fixed_ks, fixed_ks))
    step_jit = jax.jit(env.step)

    state = env.reset(jax.random.PRNGKey(reset_seed))
    rng = jax.random.PRNGKey(action_seed)

    rewards = []
    for _ in range(n_steps):
        rng, a_rng = jax.random.split(rng)
        action = jax.random.uniform(a_rng, (5,), minval=-1.0, maxval=1.0)
        state = step_jit(state, action)
        rewards.append(float(state.reward))
    return rewards


# ── Test entry ──────────────────────────────────────────────────────────────
def test_reward_parity_v0():
    """100-step parity test: sculptor v0 == pre-refactor reference, tol 1e-6."""
    n_steps = 100
    tol = 1e-6

    rewards_v0  = _run_trajectory(_compute_reward_v0,                  n_steps=n_steps)
    rewards_ref = _run_trajectory(_reference_reward_v0_preRefactor,    n_steps=n_steps)

    diffs = [abs(a - b) for a, b in zip(rewards_v0, rewards_ref)]
    max_diff = max(diffs)
    worst_idx = diffs.index(max_diff)

    # Restore default (sculptor v0) in case the test module is reused.
    env_mod._compute_reward = _compute_reward_v0  # type: ignore[attr-defined]

    # Log a sample of steps + the worst offender.
    shown = sorted({0, 1, 2, 3, 4, 25, 50, 75, 98, 99, worst_idx})
    print(f"\n  {'step':>4}  {'v0':>16}  {'reference':>16}  {'|diff|':>10}")
    for i in shown:
        print(
            f"  {i:>4}  {rewards_v0[i]:>+16.10f}  "
            f"{rewards_ref[i]:>+16.10f}  {diffs[i]:>10.2e}"
        )
    print(f"\n  max |diff| over {n_steps} steps = {max_diff:.3e}  "
          f"(tolerance {tol:.0e})")

    assert max_diff < tol, (
        f"Parity failed: max |diff| = {max_diff:.3e} at step {worst_idx}. "
        f"v0={rewards_v0[worst_idx]} ref={rewards_ref[worst_idx]}"
    )


def main() -> int:
    print("sculptor/reward.py v0 parity test")
    print("=" * 70)
    try:
        test_reward_parity_v0()
    except AssertionError as e:
        print("\nPARITY TEST FAILED")
        print(f"  {e}")
        return 1
    print("\nPARITY TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

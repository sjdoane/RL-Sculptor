"""Spec-metric monotonicity audit (GPU-free).

Implements the reward-hacking literature's prescription (Skalse et al.
2022, arXiv:2209.13085; "Goodhart's Law in RL", ICLR 2024): a task
success metric must be MONOTONIC in true competence — no
competence-neutral or competence-positive change to a policy may
DECREASE its score. A metric that fails this rewards degeneracy.

This builds synthetic "competence ladders" (faithful to
spec_metrics.py's array contract) and reports whether each benchmark
spec is monotonic. It surfaces the E4 campaign confound: g1_kick's
peak/median burst-ratio is an extremal-Goodhart proxy — a policy that
kicks identically but ALSO moves its other legs (walking/stepping)
scores LOWER, exactly the inversion seen in the real data
(plain_ppo 600-iter ≈ 0.43 vs plain_ppo_matched 2400-iter ≈ 0.08).

Run:  uv run python scripts/audit_spec_metric_monotonicity.py
"""
from __future__ import annotations

import numpy as np

from sculptor.eval.spec_metrics import spec_g1_kick, spec_go1_trot, spec_g1_floss

T, E = 500, 8
G1_NAMES = [
    "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_pitch", "right_knee", "right_ankle",
    "left_shoulder_pitch", "left_elbow",
    "right_shoulder_pitch", "right_elbow",
]
UPRIGHT = np.tile(np.array([0.0, 0.0, -1.0]), (T, E, 1))  # gravity down → upright


def _kick_train(left_leg_idx, baseline_amp: float, J: int = 10) -> np.ndarray:
    """A FIXED genuine kick (periodic strong left-leg extensions, ~every
    100 frames, sustained 6 frames at 8 rad/s) PLUS a competence-neutral
    continuous oscillation of amplitude `baseline_amp` on every other
    joint (the policy also stepping/moving). More baseline motion is NOT
    less kick-competent — a monotone metric must not punish it."""
    jv = np.zeros((T, E, J))
    t = np.arange(T)
    # competence-neutral baseline activity on all joints
    if baseline_amp > 0:
        jv += baseline_amp * np.sin(2 * np.pi * t / 17.0)[:, None, None]
    # the genuine, FIXED kick on the left leg (overwrite, sustained)
    for start in range(80, T, 100):
        for j in left_leg_idx:
            jv[start:start + 6, :, j] = 8.0
    return jv


def _kick_events_score(jv, gravity, leg_idx, *, thresh=5.0, smooth=5):
    """PROPOSED monotone replacement: saturating count of discrete kick
    EVENTS = sustained leg speed crossing an ABSOLUTE threshold from
    upright (refractory-gated). Bounded, goal-grounded, and invariant to
    competence-neutral baseline motion below threshold."""
    from sculptor.eval.spec_metrics import _sliding_mean, upright_mask
    jv = jv[:, :, leg_idx]
    sm = _sliding_mean(jv, smooth)
    speed = np.abs(sm).max(axis=2)                       # (T',E)
    up = _sliding_mean(upright_mask(gravity).astype(float), smooth) > 0.999
    hot = (speed >= thresh) & up
    events = 0
    for e in range(hot.shape[1]):
        prev = False
        for t in range(hot.shape[0]):
            if hot[t, e] and not prev:
                events += 1
            prev = hot[t, e]
    events_per_env = events / hot.shape[1]
    return 1.0 - float(np.exp(-events_per_env / 3.0))   # saturate ~3 kicks


def audit_kick():
    # §Ship 34: the clear Ship-33 bug was the ratio guard scoring a CLEAN
    # kicker (still between kicks, median~0) as 0.0 — the IDEAL kicker
    # read as "no kick". The median floor fixes that. Validate two
    # properties: (1) a clean kicker scores well, and (2) the score is
    # monotone in kick STRENGTH (true competence).
    print("\n=== g1_kick (post Ship-34 floor): clean kicker + strength monotonicity ===")
    # (1) clean kicker: 8 rad/frame leg bursts from a dead-still stance.
    clean = _kick_train([0, 1, 2], 0.0)
    clean_score = spec_g1_kick(
        {"joint_vel": clean, "projected_gravity_b": UPRIGHT}, {},
        {"joint_names": G1_NAMES})["spec_score"]
    clean_ok = clean_score > 0.5
    print(f"  clean kicker (zero background, median~0): spec={clean_score:.3f}  "
          f"-> {'FIXED (was 0.000 pre-floor)' if clean_ok else 'STILL BROKEN'}")
    # (2) monotone in kick strength.
    print(f"  {'kick_strength':>14} {'spec_score':>12} {'burst_p95':>11}")
    scores = []
    for strength in [0.0, 1.0, 2.0, 4.0, 8.0]:
        jv = np.zeros((T, E, 10))
        for start in range(80, T, 100):
            for j in (0, 1, 2):
                jv[start:start + 6, :, j] = strength
        m = spec_g1_kick({"joint_vel": jv, "projected_gravity_b": UPRIGHT}, {},
                         {"joint_names": G1_NAMES})
        scores.append(m["spec_score"])
        print(f"  {strength:>14.1f} {m['spec_score']:>12.3f} {m['burst_p95']:>11.2f}")
    mono = all(scores[i] <= scores[i + 1] + 1e-9 for i in range(len(scores) - 1))
    print(f"  monotone-non-decreasing in kick strength? {mono}")
    return clean_ok and mono


def audit_trot():
    print("\n=== go1_trot: score vs TRUE forward displacement (control) ===")
    print(f"{'fwd_speed/frame':>16} {'spec_score':>12} {'net_disp':>10}")
    scores = []
    for v in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]:
        pos = np.zeros((T, E, 3))
        pos[..., 0] = (np.arange(T)[:, None] * v)          # straight forward
        pos[..., 2] = 0.30                                   # stance height
        m = spec_go1_trot({"root_link_pos_w": pos, "projected_gravity_b": UPRIGHT}, {})
        scores.append(m["spec_score"])
        print(f"{v:>16.3f} {m['spec_score']:>12.3f} {m['net_displacement']:>10.2f}")
    mono = all(scores[i] <= scores[i + 1] + 1e-9 for i in range(len(scores) - 1))
    print(f"  monotone-non-decreasing in forward speed? {mono}")
    return mono


def audit_floss():
    print("\n=== g1_floss: all-or-nothing multiplicative gate ===")
    t = np.arange(T)
    period = 30.0
    print(f"{'hip-arm phase':>16} {'spec_score':>12} {'structure':>10} {'periodicity':>12}")
    for label, arm_phase in [("in-phase 0", 0.0), ("quarter pi/2", np.pi / 2),
                             ("anti-phase pi", np.pi)]:
        jp = np.zeros((T, E, 10))
        hip = 0.4 * np.sin(2 * np.pi * t / period)
        arm = 0.4 * np.sin(2 * np.pi * t / period + arm_phase)
        for j in [0, 3]:               # hips
            jp[:, :, j] = hip[:, None]
        for j in [6, 8]:               # shoulders
            jp[:, :, j] = arm[:, None]
        m = spec_g1_floss({"joint_pos": jp, "projected_gravity_b": UPRIGHT}, {},
                          {"joint_names": G1_NAMES})
        print(f"{label:>16} {m['spec_score']:>12.3f} {m.get('structure',0):>10.3f} "
              f"{m['periodicity']:>12.3f}")


if __name__ == "__main__":
    print("SPEC-METRIC MONOTONICITY AUDIT (synthetic competence ladders)")
    kick_ok = audit_kick()
    trot_ok = audit_trot()
    audit_floss()
    print("\n--- VERDICT ---")
    print(f"g1_kick clean-kicker fixed AND monotone in strength: {kick_ok}")
    print(f"go1_trot monotone in forward speed: {trot_ok}")
    print("Note: a fuller discrete-kick-event redesign (absolute thresholds)"
          " is deferred — it needs calibration on REAL rollouts (GPU).")

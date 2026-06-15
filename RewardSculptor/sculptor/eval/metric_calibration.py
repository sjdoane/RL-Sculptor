"""Calibration: does a generated metric RANK policies like the
hand-authored ground-truth metric? (§Ship 35 — the circularity firewall.)

A generated metric is an LLM-authored proxy. Before it may STEER a run
(drive selection/early-stop) it must agree with a hand-authored ground
truth on tasks that ground truth covers: Spearman rank-correlation ≥ a
threshold over a competence ladder. Until then it runs OBSERVE-ONLY. The
4 hand-authored metrics in spec_metrics.py never retire — they are the
permanent calibration fence + regression set.

This offline check uses SYNTHETIC competence ladders (graded rollouts the
ground-truth metric is known to order correctly). Real-policy calibration
(Spearman over a pool of trained policies) is the stronger check and runs
when a GPU is available; this is the no-GPU proxy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sculptor.eval.generated_metric import (
    inject_joint_roles,
    load_generated_metric,
    read_required_roles,
)
from sculptor.eval.spec_metrics import _SPEC_FNS

T, E, J = 120, 4, 12
_NAMES_12 = [
    "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_shoulder_pitch", "right_shoulder_pitch",
    "left_elbow", "right_elbow", "torso", "neck",
]
_BEHAVIOR = {"max_episode_steps": T, "rollout_num_envs": E, "step_dt": 0.02}


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rho via Pearson on ranks (numpy only). 0.0 if EITHER side
    has no variation — a constant metric carries no rank information and
    must NOT spuriously correlate (argsort-of-constant yields sequential
    ranks, so guard on the RAW std, not the ranks)."""
    av, bv = np.asarray(a, float), np.asarray(b, float)
    # §Ship 41 review: spec_score is contractually [0,1]; round to 6 decimals
    # BEFORE the variation guard so sub-resolution drift cannot manufacture a
    # monotone rank. A degenerate metric reading joint_pos magnitude scored
    # ~1e-7 across the (Ship-41-enriched, cumsum-joint_pos) ladder; its 1e-7
    # std cleared the 1e-12 guard and argsort then gave a spurious rho=1.0.
    av, bv = np.round(av, 6), np.round(bv, 6)
    # §Ship 35 review: epsilon guard (exact == 0 can miss tiny-but-nonzero
    # std from float noise, spuriously correlating a near-constant metric).
    if av.size < 2 or av.std() < 1e-12 or bv.std() < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(av)).astype(float)
    rb = np.argsort(np.argsort(bv)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _upright_g(t: int = T) -> np.ndarray:
    g = np.zeros((t, E, 3)); g[..., 2] = -1.0
    return g


def _ladder(builtin_name: str) -> list[tuple[dict, dict, dict]]:
    """Graded competence ladder (low→high) for a built-in metric: rollouts
    the ground-truth metric should score in increasing order."""
    t = np.arange(T)
    meta = {"joint_names": _NAMES_12}

    def loco(speed: float):
        root = np.zeros((T, E, 3)); root[..., 2] = 0.30
        root[..., 0] = (t * speed)[:, None]
        return ({"root_link_pos_w": root, "projected_gravity_b": _upright_g()},
                _BEHAVIOR, meta)

    if builtin_name in ("go1_trot",):
        return [loco(s) for s in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)]

    if builtin_name == "cartpole_balance":
        out = []
        for ln in (50, 150, 250, 350, 450, 500):
            out.append(({}, {**_BEHAVIOR, "mean_episode_length": ln,
                             "max_episode_steps": 500}, meta))
        return out

    dt = _BEHAVIOR["step_dt"]

    if builtin_name == "g1_kick":
        # §Ship 41: populate ALL physical arrays (stationary, upright, standing
        # height) — the spec needs only joint_vel+gravity, but a generated
        # metric that gates on a stationary base / standing height returns 0.0
        # when root_link_pos_w/joint_pos are absent, so it could NEVER calibrate
        # (Spearman 0). The added arrays don't change the spec's rank order.
        out = []
        root = np.zeros((T, E, 3)); root[..., 2] = 0.7   # stationary, standing
        for strength in (0.0, 1.0, 2.0, 4.0, 8.0):
            jv = np.zeros((T, E, J))
            for start in range(20, T, 40):       # discrete leg bursts
                for jdx in (0, 2, 4):            # left hip/knee/ankle
                    jv[start:start + 5, :, jdx] = strength
            jp = np.cumsum(jv, axis=0) * dt      # consistent integrated position
            out.append(({"joint_vel": jv, "joint_pos": jp,
                         "projected_gravity_b": _upright_g(),
                         "root_link_pos_w": root},
                        _BEHAVIOR, meta))
        return out

    if builtin_name == "g1_floss":
        out = []
        root = np.zeros((T, E, 3)); root[..., 2] = 0.7
        for amp in (0.0, 0.1, 0.2, 0.4):
            jp = np.zeros((T, E, J))
            hip = amp * np.sin(2 * np.pi * t / 25)
            arm = amp * np.sin(2 * np.pi * t / 25 + np.pi)   # anti-phase
            for jdx in (0, 1):
                jp[:, :, jdx] = hip[:, None]
            for jdx in (6, 7, 8, 9):
                jp[:, :, jdx] = arm[:, None]
            jv = np.gradient(jp, axis=0)
            out.append(({"joint_pos": jp, "joint_vel": jv,
                         "projected_gravity_b": _upright_g(),
                         "root_link_pos_w": root},
                        _BEHAVIOR, meta))
        return out

    if builtin_name == "g1_jump":
        # §Ship 41: graded vertical hops (crouch→launch→apex→land) with knee
        # extension bursts, upright, no horizontal travel.
        out = []
        for height in (0.0, 0.1, 0.2, 0.35, 0.5):
            z = np.full(T, 0.55)
            jv = np.zeros((T, E, J))
            for start in range(15, T, 35):
                for k in range(20):
                    if start + k < T:
                        z[start + k] = 0.55 + height * np.sin(np.pi * k / 20)
                        if k < 6:                # launch: knees extend
                            for jdx in (2, 3):
                                jv[start + k, :, jdx] = height * 12.0
            jp = np.cumsum(jv, axis=0) * dt
            root = np.zeros((T, E, 3)); root[..., 2] = z[:, None]
            out.append(({"root_link_pos_w": root, "joint_vel": jv,
                         "joint_pos": jp, "projected_gravity_b": _upright_g()},
                        _BEHAVIOR, meta))
        return out

    raise KeyError(f"no calibration ladder for built-in {builtin_name!r}")


def calibrate_metric(
    generated_module_path: Path | str,
    builtin_name: str,
    *,
    threshold: float = 0.7,
) -> dict[str, Any]:
    """Compute Spearman rank-correlation between a generated metric and a
    built-in ground-truth metric over the latter's competence ladder.
    `ok=True` (steer-rights earned) iff rho ≥ threshold. Never raises."""
    if builtin_name not in _SPEC_FNS:
        raise KeyError(f"unknown built-in metric {builtin_name!r}")
    try:
        gen_fn = load_generated_metric(generated_module_path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "spearman": 0.0, "threshold": threshold,
                "builtin": builtin_name, "error": f"{type(e).__name__}: {e}"}
    # §Ship 49: resolve the metric's declared joint roles against the
    # synthetic biped names the ladder carries, so a role-based metric reads
    # the right columns (lenient — the 12-joint body has no roll/yaw axes).
    roles = read_required_roles(generated_module_path)
    builtin_fn = _SPEC_FNS[builtin_name]
    ladder = _ladder(builtin_name)
    gen_scores, builtin_scores = [], []
    for arrays, behavior, meta in ladder:
        inject_joint_roles(meta, roles, lenient=True)
        try:
            gen_scores.append(float(gen_fn(arrays, behavior, meta).get("spec_score", 0.0)))
        except Exception:  # noqa: BLE001 — a crash on a ladder point = 0
            gen_scores.append(0.0)
        builtin_scores.append(float(builtin_fn(arrays, behavior, meta).get("spec_score", 0.0)))
    rho = spearman(gen_scores, builtin_scores)
    return {
        "ok": bool(rho >= threshold),
        "spearman": round(rho, 4),
        "threshold": threshold,
        "builtin": builtin_name,
        "n": len(ladder),
        "gen_scores": [round(s, 4) for s in gen_scores],
        "builtin_scores": [round(s, 4) for s in builtin_scores],
    }

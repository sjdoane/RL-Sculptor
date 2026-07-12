#!/usr/bin/env python3
"""§start-pose physical verification probe (standalone, read-only w.r.t.
sculptor/ package code — writes only to reports/start_pose_probe/).

Physically verifies that the reference-derived LYING resets used by the
`standing-test` project's "starting-from-lying-flat-on-its" get-up mission
are FULLY lying (not partially propped, not intersecting the floor) by
actually resetting a real 2-env CUDA mjlab G1 env for each of the four
mission stages (torso_righting, feet_under_crouch, drive_to_stand,
getup_and_hold), stepping ~25 frames with zero actions to let physics
settle, and rendering a frame at reset (frame-0) and after settling —
a human can eyeball both PNGs per stage.

Env construction mirrors `tests/test_reference_getup_gpu_smoke.py`'s
`test_getup_eval_reset_env_reset_is_lying_and_deterministic` (the D9/D17
GPU-verification harness already in this repo): build the task's env
cfg, apply the shared-only eval baseline via `_apply_env_spec(...,
train=False, ...)`, then layer the stage's `eval_reset.json` payload on
top via `_apply_eval_reset(...)` — the SAME two functions the real
rollout path (`_cmd_rollout`) calls, imported directly (no side effects
on import) rather than reimplemented. Falls back to the stage's
`env/v0.json` train-RSI section at range midpoints (noise 0) if a stage
has no `eval_reset.json` (none of the four current stages hit this path
— all four ship one — but the fallback is exercised defensively).

Rendering reuses mjlab's OWN offscreen renderer (`render_mode="rgb_array"`
on `ManagerBasedRlEnv`, backed by `mjlab.viewer.offscreen_renderer
.OffscreenRenderer`) rather than a hand-built standalone MuJoCo model —
this renders the REAL compiled task scene (flat-terrain floor included),
tracking env_idx=0 (the same env index the numeric readings below come
from), so the PNG and the numbers describe the same physical state.
Needs `MUJOCO_GL=egl` set before mujoco is imported (this WSL2 box's
established headless-GL pattern — see `sculptor/adapters/mjlab.py`'s
`_remote_device_env` docstring and `reward-sculptor-ui/backend/services
/preview_renderer.py`).

Invocation:
    cd ~/projects/RewardSculptor
    MUJOCO_GL=egl .venv/bin/python scripts/probe_start_pose.py

Outputs (this script's only writes):
    reports/start_pose_probe/<stage>_frame0.png
    reports/start_pose_probe/<stage>_settled.png
    reports/start_pose_probe/probe_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402 — after MUJOCO_GL is set (harmless either way)

MISSION_DIR = Path(
    "/home/samjd/.local/share/reward-sculptor/projects/standing-test"
    "/.missions/starting-from-lying-flat-on-its"
)
STAGES = ["torso_righting", "feet_under_crouch", "drive_to_stand", "getup_and_hold"]
TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reports" / "start_pose_probe"
N_SETTLE_STEPS = 25
NUM_ENVS = 2
RENDER_ENV_IDX = 0

# Acceptance thresholds (§task spec).
Z_LYING_MAX = 0.35
PG_Z_ABS_UPRIGHT_MAX = 0.5
Z_DROP_TRANSIENT_MAX = 0.08
Z_FLOOR_PENETRATION_MIN = 0.10


def _load_eval_payload(stage_dir: Path) -> tuple[dict, str]:
    """Return (payload, source_label). Prefers env/eval_reset.json;
    falls back to env/v0.json's train RSI section at range midpoints
    with noise forced to 0 (matches `sculptor.reference.derive_eval_reset`'s
    own midpoint convention)."""
    eval_path = stage_dir / "env" / "eval_reset.json"
    if eval_path.is_file():
        return json.loads(eval_path.read_text(encoding="utf-8")), "eval_reset.json"

    v0_path = stage_dir / "env" / "v0.json"
    spec = json.loads(v0_path.read_text(encoding="utf-8"))
    train = spec.get("train") or {}

    def _mid(key: str):
        v = train.get(key)
        if v is None:
            return None
        return (float(v[0]) + float(v[1])) / 2.0

    payload = {
        "fell_over_termination": train.get("fell_over_termination"),
        "reset_height_offset_m": _mid("reset_height_offset_m"),
        "reset_joint_pos_noise_rad": 0.0,
        "reset_joint_pos_target": train.get("reset_joint_pos_target"),
        "reset_pitch_offset_rad": _mid("reset_pitch_offset_rad"),
        "reset_roll_offset_rad": _mid("reset_roll_offset_rad"),
        "reset_vertical_velocity_mps": _mid("reset_vertical_velocity_mps"),
    }
    return payload, "v0.json(train RSI midpoint, noise=0)"


def _quat_wxyz_to_pitch_rad(q: np.ndarray) -> float:
    """Rough single-axis pitch estimate (mirrors the GPU smoke test's own
    helper). NOTE: asin saturates at +-pi/2 — for the large (>90deg)
    pitch/roll offsets these lying resets actually use, this number is
    only a rough diagnostic, NOT the authoritative "how lying is this"
    measure. `pg_z` (projected_gravity_b's z component) below is the
    authoritative one: it is exactly what this codebase's own
    `fell_over` termination uses (`terminations.py`:
    `torch.acos(-projected_gravity[:, 2]).abs() > limit_angle`)."""
    w, x, y, z = q
    sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(np.arcsin(sinp))


def _upright_angle_deg(pg_z: float) -> float:
    """Angle (deg) between the robot's body-frame "up" axis and world
    up, DERIVED THE SAME WAY the codebase's own fell_over termination
    computes it: acos(-projected_gravity_b_z). 0 deg = perfectly
    upright/standing; 180 deg = perfectly inverted; ~90 deg = body's up
    axis is horizontal (lying on front/back/side)."""
    return float(np.degrees(np.arccos(np.clip(-pg_z, -1.0, 1.0))))


def _verdict(z0: float, z_end: float, pg_z_end: float, z_drop: float) -> str:
    bits = []
    if z_end >= Z_LYING_MAX:
        bits.append(f"NOT_LYING(z_settled={z_end:.3f}>={Z_LYING_MAX})")
    if abs(pg_z_end) >= PG_Z_ABS_UPRIGHT_MAX:
        bits.append(f"UPRIGHT(|pg_z_settled|={abs(pg_z_end):.3f}>={PG_Z_ABS_UPRIGHT_MAX})")
    if abs(z_drop) >= Z_DROP_TRANSIENT_MAX:
        bits.append(f"PROPPED_OR_UNSTABLE(|z_drop|={abs(z_drop):.3f}>={Z_DROP_TRANSIENT_MAX})")
    if z0 < Z_FLOOR_PENETRATION_MIN:
        bits.append(f"FLOOR_PENETRATION(z0={z0:.3f}<{Z_FLOOR_PENETRATION_MIN})")
    return "FULLY_LYING" if not bits else "; ".join(bits)


def _run_stage(stage: str, torch_mod, ManagerBasedRlEnv, load_env_cfg,
               ViewerConfig, _apply_env_spec, _apply_eval_reset, Image) -> dict:
    stage_dir = MISSION_DIR / "stages" / stage
    payload, source = _load_eval_payload(stage_dir)
    print(f"=== {stage} (reset source={source}) ===", file=sys.stderr, flush=True)

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.viewer = ViewerConfig(
        height=480, width=640, env_idx=RENDER_ENV_IDX, max_extra_envs=0,
        distance=3.0, elevation=-20.0, azimuth=120.0,
    )
    # Shared-only baseline — exactly what `_cmd_rollout` applies before an
    # eval-reset override (spec=None -> task defaults, standing-start,
    # nothing get-up-shaped).
    _apply_env_spec(env_cfg, None, train=False, task_id=TASK_ID)
    _apply_eval_reset(env_cfg, payload, task_id=TASK_ID)
    assert "fell_over" not in env_cfg.terminations, (
        f"{stage}: fell_over termination still present after eval-reset "
        "override — the lying start would trip it at reset")

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
    try:
        env.reset()
        asset = env.scene["robot"]

        root0 = asset.data.root_link_pose_w[RENDER_ENV_IDX].detach().cpu().numpy()
        z0 = float(root0[2])
        pg0 = asset.data.projected_gravity_b[RENDER_ENV_IDX].detach().cpu().numpy()
        pg_z0 = float(pg0[2])
        pitch0 = _quat_wxyz_to_pitch_rad(root0[3:7])

        frame0 = env.render()
        frame0_path = OUT_DIR / f"{stage}_frame0.png"
        Image.fromarray(frame0).save(frame0_path)

        n_actions = env.action_manager.total_action_dim
        zero_action = torch_mod.zeros(
            (env_cfg.scene.num_envs, n_actions), device=env.device)

        z_trace = [z0]
        terminated_any_during_settle = False
        for _ in range(N_SETTLE_STEPS):
            _obs, _rew, terminated, _truncated, _extras = env.step(zero_action)
            if bool(terminated.any().item()):
                terminated_any_during_settle = True
            root_i = asset.data.root_link_pose_w[RENDER_ENV_IDX].detach().cpu().numpy()
            z_trace.append(float(root_i[2]))

        root_end = asset.data.root_link_pose_w[RENDER_ENV_IDX].detach().cpu().numpy()
        z_end = float(root_end[2])
        pg_end = asset.data.projected_gravity_b[RENDER_ENV_IDX].detach().cpu().numpy()
        pg_z_end = float(pg_end[2])
        pitch_end = _quat_wxyz_to_pitch_rad(root_end[3:7])
        qvel_end = asset.data.joint_vel[RENDER_ENV_IDX].detach().cpu().numpy()
        max_qvel_end = float(np.max(np.abs(qvel_end)))

        frame_end = env.render()
        settled_path = OUT_DIR / f"{stage}_settled.png"
        Image.fromarray(frame_end).save(settled_path)

        z_drop = z0 - z_end
        z_min_during_settle = float(min(z_trace))
        verdict = _verdict(z0, z_end, pg_z_end, z_drop)

        result = {
            "reset_source": source,
            "z0": z0, "pitch0_rad_ROUGH": pitch0, "pg_z0": pg_z0,
            "upright_angle0_deg": _upright_angle_deg(pg_z0),
            "z_settled": z_end, "pitch_settled_rad_ROUGH": pitch_end,
            "pg_z_settled": pg_z_end,
            "upright_angle_settled_deg": _upright_angle_deg(pg_z_end),
            "z_drop_m": z_drop,
            "z_min_during_settle": z_min_during_settle,
            "max_abs_joint_vel_settled_radps": max_qvel_end,
            "terminated_any_env_during_settle": terminated_any_during_settle,
            "step_dt_s": float(env.step_dt),
            "z_trace": z_trace,
            "verdict": verdict,
            "frame0_png": str(frame0_path),
            "settled_png": str(settled_path),
        }
        print(
            f"{stage}: z0={z0:.3f} pg_z0={pg_z0:.3f} "
            f"(upright_angle0={result['upright_angle0_deg']:.1f}deg) | "
            f"z_end={z_end:.3f} pg_z_end={pg_z_end:.3f} "
            f"(upright_angle_end={result['upright_angle_settled_deg']:.1f}deg) "
            f"z_drop={z_drop:.3f} max|qvel|={max_qvel_end:.3f} "
            f"verdict={verdict}",
            file=sys.stderr, flush=True,
        )
        return result
    finally:
        env.close()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    import torch  # noqa: E402
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: E402
    from mjlab.tasks.registry import load_env_cfg  # noqa: E402
    from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402
    from PIL import Image  # noqa: E402

    sys.path.insert(0, str(REPO_ROOT))
    from sculptor.adapters._mjlab_runner import (  # noqa: E402
        _apply_env_spec, _apply_eval_reset,
    )

    summary: dict = {}
    for stage in STAGES:
        stage_t0 = time.time()
        try:
            summary[stage] = _run_stage(
                stage, torch, ManagerBasedRlEnv, load_env_cfg, ViewerConfig,
                _apply_env_spec, _apply_eval_reset, Image,
            )
            summary[stage]["wall_s"] = time.time() - stage_t0
        except Exception as e:  # noqa: BLE001 — report, don't fake, don't abort other stages
            tb = traceback.format_exc()
            print(f"=== {stage}: FAILED ===\n{tb}", file=sys.stderr, flush=True)
            summary[stage] = {
                "ERROR": f"{type(e).__name__}: {e}",
                "traceback": tb,
                "wall_s": time.time() - stage_t0,
            }

    elapsed = time.time() - t_start
    summary["_meta"] = {
        "elapsed_s": elapsed, "task_id": TASK_ID, "num_envs": NUM_ENVS,
        "n_settle_steps": N_SETTLE_STEPS, "render_env_idx": RENDER_ENV_IDX,
        "mission_dir": str(MISSION_DIR),
        "thresholds": {
            "z_lying_max_m": Z_LYING_MAX,
            "pg_z_abs_upright_max": PG_Z_ABS_UPRIGHT_MAX,
            "z_drop_transient_max_m": Z_DROP_TRANSIENT_MAX,
            "z_floor_penetration_min_m": Z_FLOOR_PENETRATION_MIN,
        },
    }
    out_json = OUT_DIR / "probe_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ndone in {elapsed:.1f}s total. summary -> {out_json}",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

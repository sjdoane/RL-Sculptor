"""Reference-clip -> battery-array conversion (§REFERENCE_TRAJECTORY_PLAN §5, §2.2, §7).

`clip_to_arrays` is THE integration trick the plan calls out: a converted
clip becomes just another entry in the synthetic battery
(`sculptor.eval.metric_validate._score(fn, arrays, meta)` works unchanged),
so a reference clip can serve as a nondegeneracy POSITIVE, a calibration
ladder rung, and a generation-grounding source from one canonical
conversion path.

Two public entry points:

  * `clip_to_arrays(clip, n_envs=4) -> (arrays, meta)` — converts a
    `sculptor.reference` clip dict into the `(T, E, J)` / `(T, E, 3)` /
    `(T, E)` array dict `ALLOWED_ARRAYS` names, tiled across `n_envs`
    (E-dimension broadcast — the plan's §14 open question, resolved by
    tiling: cheap and keeps every downstream array shape-consistent with
    the fixed battery). Missing SOURCE keys simply omit the derived array
    (metrics already `arrays.get()`-guard by contract).
  * `kinematic_signature(clip) -> dict` — a compact, numeric, JSON-
    serializable summary for LLM prompts (generation grounding, §7):
    duration/fps, root-height extrema + when they occur, phase
    segmentation, velocity ranges, orientation summary, contact schedule,
    per-phase joint-motion energy.

FK-derived foot positions/contacts (`_infer_foot_contacts`) are an OPTIONAL
enhancement gated on the G1 MJCF being resolvable via
`sculptor.refs.preview.resolve_g1_mjcf` — kinematics only (`mj_forward`,
never `mj_step`), so it never requires a physics-consistent clip. Any
failure (mjcf/mujoco unavailable, clip missing the fields needed to pose
the robot) degrades silently: the arrays are simply absent, exactly like
any other missing source key. `clip_to_arrays` NEVER raises because FK
failed.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: The keys `clip_to_arrays` may emit — a subset of
#: `sculptor.eval.generated_metric.ALLOWED_ARRAYS` (everything that key
#: family supports is either directly sourced from the clip or FK-derived;
#: `left_foot_contact`/`right_foot_contact` are separate keys, not part of
#: `ALLOWED_ARRAYS`'s foot-pos naming, so listed explicitly here too).
CONVERTED_ARRAY_KEYS = (
    "joint_pos",
    "joint_vel",
    "projected_gravity_b",
    "root_link_pos_w",
    "left_foot_contact",
    "right_foot_contact",
    "left_foot_pos_b",
    "right_foot_pos_b",
)

#: FK-inferred-contact heuristic thresholds (standard kinematic proxy —
#: §REFERENCE_TRAJECTORY_PLAN §2.2 "foot below height threshold AND |foot
#: vel| below threshold").
_CONTACT_HEIGHT_M = 0.06
_CONTACT_SPEED_MPS = 0.25


# ── orientation: world gravity -> body frame ─────────────────────────────
def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """`(..., 4)` unit quaternions (w, x, y, z) -> `(..., 3, 3)` rotation
    matrices, R such that `R @ v_world = v_body` is NOT what this returns —
    this returns the world-from-body rotation (standard quat-to-matrix);
    the caller applies the TRANSPOSE to rotate a world vector into the
    body frame (§below)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = w * w + x * x + y * y + z * z
    n = np.where(n < 1e-12, 1.0, n)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    r00 = 1.0 - (yy + zz); r01 = xy - wz;         r02 = xz + wy
    r10 = xy + wz;         r11 = 1.0 - (xx + zz); r12 = yz - wx
    r20 = xz - wy;         r21 = yz + wx;         r22 = 1.0 - (xx + yy)
    out = np.stack([
        np.stack([r00, r01, r02], axis=-1),
        np.stack([r10, r11, r12], axis=-1),
        np.stack([r20, r21, r22], axis=-1),
    ], axis=-2)
    return out


def _projected_gravity_b(root_quat_wxyz: np.ndarray) -> np.ndarray:
    """World gravity direction `(0, 0, -1)` rotated into the body frame,
    per-frame: `g_b = R_world_from_body^T @ g_world`. Identity quaternion
    (`w=1, x=y=z=0`) must return exactly `(0, 0, -1)` (unit-tested); a 90-deg
    rotation about a horizontal axis must tip gravity into a horizontal
    body axis (also unit-tested)."""
    R = _quat_wxyz_to_rotmat(root_quat_wxyz)          # (T, 3, 3)
    g_world = np.array([0.0, 0.0, -1.0])
    # body = R^T @ world  <=>  einsum contracting the FIRST matrix axis
    g_b = np.einsum("tij,j->ti", np.swapaxes(R, -1, -2), g_world)
    return g_b


# ── FK (kinematics only — mj_forward, never mj_step) ─────────────────────
def _infer_foot_contacts(clip: dict) -> Optional[dict[str, np.ndarray]]:
    """Best-effort FK pass over every keyframe to derive foot positions
    (pelvis/body frame) and inferred ground contacts. Returns None (never
    raises) when the MJCF/mujoco isn't available or the clip lacks the
    fields needed to pose the robot — `clip_to_arrays` treats that exactly
    like any other absent source key."""
    required = ("root_pos_xy", "root_quat_wxyz", "joint_pos", "joint_names")
    if any(clip.get(k) is None for k in required):
        return None
    try:
        import mujoco

        from sculptor.refs.preview import build_qpos_frame, resolve_g1_mjcf

        xml_path = resolve_g1_mjcf()
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)

        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        if left_id < 0 or right_id < 0:
            # No named foot sites on this MJCF — nothing we can FK to.
            return None

        z = clip["root_pos_z"]
        xy = clip["root_pos_xy"]
        quat = clip["root_quat_wxyz"]
        jp = clip["joint_pos"]
        names = list(clip["joint_names"])
        T = int(z.shape[0])

        left_world = np.zeros((T, 3))
        right_world = np.zeros((T, 3))
        pelvis_world = np.zeros((T, 3))
        for i in range(T):
            qpos = build_qpos_frame(
                model, root_pos_xy=xy[i], root_pos_z=float(z[i]),
                root_quat_wxyz=quat[i], joint_names=names, joint_pos=jp[i])
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)             # kinematics only — no dynamics
            left_world[i] = data.site_xpos[left_id]
            right_world[i] = data.site_xpos[right_id]
            pelvis_world[i] = qpos[0:3]

        R = _quat_wxyz_to_rotmat(quat)                  # (T, 3, 3)
        RT = np.swapaxes(R, -1, -2)
        left_b = np.einsum("tij,tj->ti", RT, left_world - pelvis_world)
        right_b = np.einsum("tij,tj->ti", RT, right_world - pelvis_world)

        fps = float(clip["fps"])
        left_vel = np.gradient(left_world, axis=0) * fps
        right_vel = np.gradient(right_world, axis=0) * fps
        left_speed = np.linalg.norm(left_vel, axis=-1)
        right_speed = np.linalg.norm(right_vel, axis=-1)
        left_contact = (
            (left_world[:, 2] < _CONTACT_HEIGHT_M)
            & (left_speed < _CONTACT_SPEED_MPS)).astype(np.float64)
        right_contact = (
            (right_world[:, 2] < _CONTACT_HEIGHT_M)
            & (right_speed < _CONTACT_SPEED_MPS)).astype(np.float64)

        return {
            "left_foot_pos_b": left_b,
            "right_foot_pos_b": right_b,
            "left_foot_contact": left_contact,
            "right_foot_contact": right_contact,
        }
    except Exception:  # noqa: BLE001 — FK is a best-effort enhancement
        return None


def _tile_env(arr: np.ndarray, n_envs: int) -> np.ndarray:
    """`(T, ...)` -> `(T, E, ...)` by tiling along a new axis-1."""
    return np.repeat(arr[:, None, ...], n_envs, axis=1)


def clip_to_arrays(
    clip: dict, *, n_envs: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert a validated `sculptor.reference` clip into the battery's
    `(T, E, J)` / `(T, E, 3)` / `(T, E)` array-dict format + a `meta` dict
    a generated metric can be scored against via
    `sculptor.eval.metric_validate._score(fn, arrays, meta)`.

    Missing SOURCE keys simply omit the derived array — metrics already
    must `arrays.get(...)`-guard by contract, so a clip lacking, say,
    `root_quat_wxyz` yields no `projected_gravity_b` rather than a
    fabricated one. `joint_vel` is finite-differenced from `joint_pos`
    when the clip doesn't carry it (mirrors `sculptor.reference._with_velocity`).
    FK-derived foot arrays are attempted opportunistically and never raise
    on failure (see `_infer_foot_contacts`).
    """
    from sculptor.reference import _with_velocity, validate_clip

    errors = validate_clip(clip)
    if errors:
        raise ValueError(
            "refusing to convert invalid reference clip:\n  - "
            + "\n  - ".join(errors))
    clip = _with_velocity(clip)

    z = clip["root_pos_z"]
    T = int(z.shape[0])
    fps = float(clip["fps"])
    arrays: dict[str, np.ndarray] = {}

    jp = clip.get("joint_pos")
    if jp is not None:
        arrays["joint_pos"] = _tile_env(jp, n_envs)
        jv = clip.get("joint_vel")
        if jv is None:
            jv = np.gradient(jp, axis=0) * fps
        arrays["joint_vel"] = _tile_env(jv, n_envs)

    quat = clip.get("root_quat_wxyz")
    if quat is not None:
        arrays["projected_gravity_b"] = _tile_env(
            _projected_gravity_b(quat), n_envs)

    xy = clip.get("root_pos_xy")
    root_xy = xy if xy is not None else np.zeros((T, 2))
    root_pos_w = np.concatenate([root_xy, z[:, None]], axis=1)
    arrays["root_link_pos_w"] = _tile_env(root_pos_w, n_envs)

    for key, src_key in (
        ("left_foot_contact", "contact_left_foot"),
        ("right_foot_contact", "contact_right_foot"),
    ):
        c = clip.get(src_key)
        if c is not None:
            arrays[key] = _tile_env(c, n_envs)

    inferred = False
    fk = _infer_foot_contacts(clip)
    if fk is not None:
        for key, val in fk.items():
            # Measured contacts (if present on the clip) take precedence
            # over FK-inferred ones; FK still supplies foot POSITIONS,
            # which are never sourced any other way.
            if key in arrays:
                continue
            arrays[key] = _tile_env(val, n_envs)
            if key in ("left_foot_contact", "right_foot_contact"):
                inferred = True

    meta: dict[str, Any] = {
        "joint_names": list(clip.get("joint_names") or []),
        "max_episode_steps": T,
        "rollout_num_envs": int(n_envs),
        "step_dt": 1.0 / fps,
        "reference": {"inferred_contacts": inferred},
    }
    return arrays, meta


# ── kinematic signature (§7 generation grounding) ─────────────────────────
def _smoothed_velocity(z: np.ndarray, fps: float, window: int = 5) -> np.ndarray:
    vz = np.gradient(z) * fps
    if window <= 1 or z.size < window:
        return vz
    kernel = np.ones(window) / window
    return np.convolve(vz, kernel, mode="same")


#: §D24 F1: number of evenly-spaced orientation knots recorded in
#: `kinematic_signature`'s `orientation.timeline` (below). ~12 is enough
#: resolution for an LLM to locate a phase-boundary-adjacent orientation
#: change without ballooning the prompt payload.
_ORIENTATION_TIMELINE_KNOTS = 12


def _orientation_timeline(clip: dict) -> list[dict[str, Any]]:
    """`~_ORIENTATION_TIMELINE_KNOTS` evenly spaced `{t, z, g_z}` knots
    spanning `clip` start to end (frame 0 and the last frame are ALWAYS
    included — `np.linspace(..., endpoint=True)` guarantees both
    endpoints exactly). Requires `root_quat_wxyz`; callers must guard on
    its presence (mirrors every other orientation-derived value in this
    module). `g_z` is the same body-frame projected-gravity-z convention
    as `orientation.initial_gravity_z_b`/`final_gravity_z_b` (near -1.0 =
    upright; near 0 or positive = lying/inverted).

    §D24 (docs/internal/REFERENCE_BUILD_LOG.md D23/D24): this timeline
    is what lets `sculptor.refs.spans` locate goal-aligned sub-span
    boundaries and measure RELATIVE orientation change within a
    candidate span — absolute g_z values are retarget-convention
    dependent (D23 found a real clip's "lying" g_z at -0.65, not the ~0
    a different pipeline might emit), so only the CHANGE across knots is
    ever used as a motion signal, never an absolute threshold.
    """
    quat = clip["root_quat_wxyz"]
    z = clip["root_pos_z"]
    fps = float(clip["fps"])
    T = int(np.asarray(z).shape[0])
    g_b = _projected_gravity_b(quat)
    n_knots = max(2, min(_ORIENTATION_TIMELINE_KNOTS, T))
    idx = np.unique(np.round(np.linspace(0, T - 1, n_knots)).astype(int))
    return [
        {"t": round(float(i) / fps, 3),
         "z": round(float(z[i]), 3),
         "g_z": round(float(g_b[i, 2]), 3)}
        for i in idx
    ]


def _phase_segments(z: np.ndarray, fps: float) -> list[dict]:
    """Rising/falling/still phase segmentation from smoothed root_z
    velocity — a compact motion outline for prompts."""
    vz = _smoothed_velocity(z, fps)
    still_eps = max(0.02, 0.05 * float(np.std(vz) or 1.0))
    labels = np.where(vz > still_eps, "rising",
                       np.where(vz < -still_eps, "falling", "still"))
    segments: list[dict] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segments.append({
                "phase": str(labels[start]),
                "t_start": round(start / fps, 3),
                "t_end": round(i / fps, 3),
                "z_start": round(float(z[start]), 3),
                "z_end": round(float(z[i - 1]), 3),
            })
            start = i
    return segments


def kinematic_signature(clip: dict) -> dict[str, Any]:
    """Compact, numeric, JSON-serializable summary of a reference clip for
    LLM prompts (§REFERENCE_TRAJECTORY_PLAN §7 generation grounding):
    duration, root-height extrema + WHEN they occur, phase segmentation,
    root velocity ranges, an orientation summary (lying vs upright) when
    orientation is available, a contact-schedule summary when contacts are
    available, and per-phase joint-motion energy. Floats rounded to 3
    decimals; every key is a stable, serializable primitive/container."""
    from sculptor.reference import _with_velocity, validate_clip

    errors = validate_clip(clip)
    if errors:
        raise ValueError(
            "refusing to summarize invalid reference clip:\n  - "
            + "\n  - ".join(errors))
    clip = _with_velocity(clip)
    z = clip["root_pos_z"]
    fps = float(clip["fps"])
    T = int(z.shape[0])
    vz = _smoothed_velocity(z, fps)

    i_min = int(np.argmin(z))
    i_max = int(np.argmax(z))
    sig: dict[str, Any] = {
        "duration_s": round(T / fps, 3),
        "fps": round(fps, 3),
        "n_frames": T,
        "root_z": {
            "start": round(float(z[0]), 3),
            "end": round(float(z[-1]), 3),
            "min": round(float(z[i_min]), 3),
            "min_t": round(i_min / fps, 3),
            "max": round(float(z[i_max]), 3),
            "max_t": round(i_max / fps, 3),
        },
        "root_velocity_mps": {
            "min": round(float(vz.min()), 3),
            "max": round(float(vz.max()), 3),
        },
        "phases": _phase_segments(z, fps),
    }

    quat = clip.get("root_quat_wxyz")
    if quat is not None:
        g_b = _projected_gravity_b(quat)
        sig["orientation"] = {
            "initial_gravity_z_b": round(float(g_b[0, 2]), 3),
            "final_gravity_z_b": round(float(g_b[-1, 2]), 3),
            "note": (
                "gravity_z_b near -1.0 = upright; near 0 or positive = "
                "lying/inverted"),
            # §D24 F1: additive-only key (schema-compatible — readers
            # ignore unknown keys, see reference_context.py's docstring).
            # Lets a stage-goal-aligned sub-span be located/measured
            # (sculptor.refs.spans) without a second full-clip pass.
            "timeline": _orientation_timeline(clip),
        }

    contacts: dict[str, np.ndarray] = {}
    for key, src_key in (
        ("left_foot", "contact_left_foot"), ("right_foot", "contact_right_foot"),
    ):
        c = clip.get(src_key)
        if c is not None:
            contacts[key] = c
    if not contacts:
        fk = _infer_foot_contacts(clip)
        if fk is not None:
            contacts = {
                "left_foot": fk["left_foot_contact"],
                "right_foot": fk["right_foot_contact"],
            }
            sig["contact_inferred"] = True
    if contacts:
        sched: dict[str, Any] = {}
        for foot, c in contacts.items():
            frac = float(np.mean(c > 0.5))
            sched[foot] = {"contact_fraction": round(frac, 3)}
        sig["contact_schedule"] = sched

    jp = clip.get("joint_pos")
    if jp is not None:
        jv = clip.get("joint_vel")
        if jv is None:
            jv = np.gradient(jp, axis=0) * fps
        energy_by_phase = []
        for seg in sig["phases"]:
            s = int(round(seg["t_start"] * fps))
            e = max(s + 1, int(round(seg["t_end"] * fps)))
            e = min(e, T)
            energy = float(np.mean(np.abs(jv[s:e]))) if e > s else 0.0
            energy_by_phase.append({
                "phase": seg["phase"], "t_start": seg["t_start"],
                "t_end": seg["t_end"], "mean_abs_joint_vel": round(energy, 3),
            })
        sig["joint_motion_energy_by_phase"] = energy_by_phase

    return sig

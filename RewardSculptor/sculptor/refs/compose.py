"""sculptor/refs/compose.py — novel motions composed from solved clips.

The reference library is a pile of *already-solved* motion: real, retargeted,
physically-plausible trajectories. Until now every consumer was
**single-clip**: retrieve one clip (`refs.retrieve`), crop it to the
goal-aligned sub-span (`refs.spans.select_reference_span`), track it
(`refs.track`). A goal whose motion is not substantially contained in some
single clip therefore had no reference at all, and fell through to either
`refs.synth` (one LLM sketch of root height/orientation over time — coarse,
and explicitly a last resort) or to running blind.

That is the ceiling this module lifts. A motion nobody has recorded is very
often a *re-sequencing of phases that have each been recorded separately* —
crouch from one clip, the launch from another, a landing absorb from a third.
Composing those spans yields a kinematic prior for a genuinely novel motion
while every frame remains real solved data rather than an LLM's guess at
numbers.

What this module does and does not claim:

- It is a **kinematic** composer. Concatenating dynamically-valid spans does
  NOT yield a dynamically-valid whole: momentum is not conserved across a
  seam, and a blend window is interpolation, not physics. The composite is a
  *candidate*, and `refs.track`'s Tier-D physics certification is the
  admission filter that decides whether it is trackable. Compose then track;
  never compose and trust.
- Seams are the risk, so they are measured, not assumed. `seam_report`
  returns the per-seam position discontinuity and the composite's peak joint
  velocity, and `compose_reference` refuses a composite whose seams exceed
  the configured tolerance rather than emitting a clip that looks valid and
  tracks catastrophically.
- Provenance is exact. Every composite records which clip and which frame
  range produced every output frame, plus the SE(2) transform applied to it,
  so a result can always be traced back to real source data.

Continuity handling, in order:

1. **Crop** each requested span with `refs.spans.crop_span` (shared frame
   handling; no bespoke indexing here).
2. **Resample** to one target fps — linear for vectors, slerp for the root
   quaternion, nearest for contact booleans.
3. **SE(2)-align** each segment onto its predecessor: rotate about world Z so
   headings agree, then translate so the root XY is continuous. Without this
   the composite teleports at every seam, which is the single largest source
   of untrackable composites.
4. **Cross-fade** over a blend window with a smoothstep weight (C1 at both
   ends, so the blend does not inject a velocity step of its own).
5. **Recompute velocities** from the blended positions. Inherited velocities
   are wrong the moment positions are blended or resampled, and a wrong
   `joint_vel` silently corrupts both RSI and DeepMimic-style tracking terms.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

#: Contact channels are boolean-valued; they are resampled/blended by
#: nearest-neighbour, never averaged (a 0.5 contact is not a state).
_CONTACT_KEYS = ("contact_left_foot", "contact_right_foot")

#: Default seam tolerances. Deliberately conservative: a composite that trips
#: these is far more likely to waste a Tier-D tracking run than to be the one
#: good novel motion.
DEFAULT_MAX_SEAM_JOINT_JUMP_RAD = 0.35
DEFAULT_MAX_JOINT_VEL_RAD_S = 30.0
DEFAULT_BLEND_S = 0.20


class ComposeError(ValueError):
    """Raised when segments cannot be composed into a valid clip."""


# ── quaternion helpers (wxyz, the clip schema's convention) ─────────────
def _quat_canonical(q: np.ndarray) -> np.ndarray:
    """Force consecutive quaternions onto one hemisphere.

    q and -q are the same rotation, but a sign flip mid-clip makes linear
    interpolation and finite-difference velocity swing through the long way.
    Source clips concatenated from different recordings routinely disagree on
    sign, so this must run before any interpolation.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


def _quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw (rotation about world Z) in radians from wxyz quaternions."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz, broadcasting over leading axes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    dtype=np.float64)


def _slerp(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp. Falls back to normalized lerp when the pair is
    nearly parallel (where slerp's sin division is numerically unstable)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = (sin_theta < 1e-6).ravel()
    out = np.empty_like(q0)
    if np.any(~near):
        idx = ~near
        t = theta[idx]
        s = sin_theta[idx]
        out[idx] = (np.sin((1.0 - w[idx]) * t) / s * q0[idx]
                    + np.sin(w[idx] * t) / s * q1[idx])
    if np.any(near):
        idx = near
        out[idx] = (1.0 - w[idx]) * q0[idx] + w[idx] * q1[idx]
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)


def _smoothstep(n: int) -> np.ndarray:
    """Blend weights 0->1 with zero derivative at both ends (3t^2 - 2t^3).

    A linear ramp is C0 only: it steps the velocity at both ends of the blend
    window, which is exactly the artifact a tracking reward punishes hardest.
    """
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return 3.0 * t ** 2 - 2.0 * t ** 3


# ── resampling ──────────────────────────────────────────────────────────
def _resample(clip: dict, target_fps: float) -> dict:
    """Resample every time-indexed channel to `target_fps`."""
    from sculptor.refs.perturb import _PASSTHROUGH_KEYS, _TIME_KEYS, _time_len

    src_fps = float(clip["fps"])
    n = _time_len(clip)
    if abs(src_fps - target_fps) < 1e-9:
        out = {k: clip[k] for k in _PASSTHROUGH_KEYS if k in clip}
        for k in _TIME_KEYS:
            if clip.get(k) is not None:
                out[k] = np.asarray(clip[k])
        return out

    duration = n / src_fps
    m = max(2, int(round(duration * target_fps)))
    src_t = np.arange(n, dtype=np.float64) / src_fps
    dst_t = np.minimum(np.arange(m, dtype=np.float64) / target_fps, src_t[-1])

    out: dict[str, Any] = {k: clip[k] for k in _PASSTHROUGH_KEYS if k in clip}
    out["fps"] = float(target_fps)
    for key in _TIME_KEYS:
        value = clip.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if key == "root_quat_wxyz":
            q = _quat_canonical(arr)
            idx = np.clip(np.searchsorted(src_t, dst_t, side="right") - 1,
                          0, n - 2)
            span = np.clip(src_t[idx + 1] - src_t[idx], 1e-12, None)
            w = (dst_t - src_t[idx]) / span
            out[key] = _slerp(q[idx], q[idx + 1], w)
        elif key in _CONTACT_KEYS:
            idx = np.clip(np.round(dst_t * src_fps).astype(int), 0, n - 1)
            out[key] = arr[idx]
        elif arr.ndim == 1:
            out[key] = np.interp(dst_t, src_t, arr)
        else:
            out[key] = np.stack(
                [np.interp(dst_t, src_t, arr[:, j])
                 for j in range(arr.shape[1])], axis=1)
    return out


# ── joint-set alignment ─────────────────────────────────────────────────
def _align_joints(segments: list[dict], names: Sequence[str]) -> None:
    """Reorder each segment's joint channels onto `names`, in place.

    Clips from different source recordings of the same robot frequently agree
    on the joint SET but not its ORDER. Concatenating those by position
    silently swaps limbs — a corruption that validates cleanly and only shows
    up as an inexplicably untrackable composite, so it is rejected loudly.
    """
    target = list(names)
    for seg_index, seg in enumerate(segments):
        seg_names = list(seg.get("joint_names") or [])
        if not seg_names:
            continue
        if seg_names == target:
            continue
        if sorted(seg_names) != sorted(target):
            missing = sorted(set(target) - set(seg_names))
            extra = sorted(set(seg_names) - set(target))
            raise ComposeError(
                f"segment {seg_index} joint set does not match segment 0; "
                f"missing={missing} unexpected={extra}. Compose requires one "
                "robot's joint set across every segment — retarget first "
                "(sculptor.refs.retarget) rather than composing across "
                "embodiments.")
        order = [seg_names.index(n) for n in target]
        for key in ("joint_pos", "joint_vel"):
            if seg.get(key) is not None:
                seg[key] = np.asarray(seg[key])[:, order]
        seg["joint_names"] = target


# ── SE(2) continuity ────────────────────────────────────────────────────
def _se2_align(seg: dict, anchor_xy: np.ndarray, anchor_yaw: float) -> dict:
    """Rotate/translate `seg` so its first frame starts at the anchor pose.

    Root XY and heading are the only channels with an absolute world frame;
    joint angles and root height are already relative. Without this the
    composite jumps to wherever the donor recording happened to be standing.
    """
    out = dict(seg)
    quat = seg.get("root_quat_wxyz")
    xy = seg.get("root_pos_xy")
    seg_yaw0 = float(_quat_yaw(np.asarray(quat)[0])) if quat is not None else 0.0
    d_yaw = float(anchor_yaw - seg_yaw0)

    if quat is not None:
        rot = _yaw_quat(d_yaw)
        out["root_quat_wxyz"] = _quat_canonical(
            _quat_mul(rot[None, :], _quat_canonical(np.asarray(quat))))
    if xy is not None:
        xy = np.asarray(xy, dtype=np.float64)
        c, s = np.cos(d_yaw), np.sin(d_yaw)
        rotated = np.stack([c * xy[:, 0] - s * xy[:, 1],
                            s * xy[:, 0] + c * xy[:, 1]], axis=1)
        out["root_pos_xy"] = rotated - rotated[0] + np.asarray(
            anchor_xy, dtype=np.float64)
    out["_se2"] = {"d_yaw_rad": d_yaw,
                   "anchor_xy": [float(v) for v in np.asarray(anchor_xy)]}
    return out


# ── blending ────────────────────────────────────────────────────────────
def _blend(a: dict, b: dict, n_blend: int) -> dict:
    """Cross-fade `a`'s tail into `b`'s head over `n_blend` frames."""
    from sculptor.refs.perturb import _TIME_KEYS

    n_a = int(np.asarray(a["root_pos_z"]).shape[0])
    n_b = int(np.asarray(b["root_pos_z"]).shape[0])
    n_blend = int(max(0, min(n_blend, n_a - 1, n_b - 1)))
    out: dict[str, Any] = {k: a[k] for k in ("fps", "joint_names", "meta")
                           if k in a}
    if n_blend == 0:
        for key in _TIME_KEYS:
            if a.get(key) is None or b.get(key) is None:
                continue
            out[key] = np.concatenate(
                [np.asarray(a[key]), np.asarray(b[key])], axis=0)
        return out

    w = _smoothstep(n_blend)
    for key in _TIME_KEYS:
        if a.get(key) is None or b.get(key) is None:
            continue
        av = np.asarray(a[key], dtype=np.float64)
        bv = np.asarray(b[key], dtype=np.float64)
        head, a_tail = av[:n_a - n_blend], av[n_a - n_blend:]
        b_head, tail = bv[:n_blend], bv[n_blend:]
        if key == "root_quat_wxyz":
            mid = _slerp(a_tail, b_head, w)
        elif key in _CONTACT_KEYS:
            mid = np.where(w < 0.5, a_tail, b_head)
        else:
            wr = w.reshape(-1, *([1] * (av.ndim - 1)))
            mid = (1.0 - wr) * a_tail + wr * b_head
        out[key] = np.concatenate([head, mid, tail], axis=0)
    return out


# ── derived velocities ──────────────────────────────────────────────────
def _recompute_velocities(clip: dict) -> dict:
    """Replace velocity channels with finite differences of the composed
    positions. Blending and resampling both invalidate inherited velocities."""
    fps = float(clip["fps"])
    n = int(np.asarray(clip["root_pos_z"]).shape[0])
    if n < 2:
        return clip
    out = dict(clip)
    z = np.asarray(clip["root_pos_z"], dtype=np.float64)
    out["root_vel_z"] = np.gradient(z, 1.0 / fps)
    jp = clip.get("joint_pos")
    if jp is not None:
        jp = np.asarray(jp, dtype=np.float64)
        out["joint_vel"] = np.gradient(jp, 1.0 / fps, axis=0)
    return out


# ── QC ──────────────────────────────────────────────────────────────────
def seam_report(clip: dict, seam_frames: Sequence[int]) -> dict[str, Any]:
    """Measured discontinuity at each seam plus the composite's peak speeds.

    This is the honest read on whether a composite is worth a Tier-D tracking
    run. It reports magnitudes rather than a bare pass/fail so a caller can
    see *how close* a rejected composite was.
    """
    jp = clip.get("joint_pos")
    fps = float(clip["fps"])
    seams: list[dict[str, Any]] = []
    for frame in seam_frames:
        entry: dict[str, Any] = {"frame": int(frame),
                                 "time_s": round(float(frame) / fps, 4)}
        if jp is not None and 0 < frame < np.asarray(jp).shape[0]:
            arr = np.asarray(jp, dtype=np.float64)
            jump = np.abs(arr[frame] - arr[frame - 1])
            entry["max_joint_jump_rad"] = round(float(np.max(jump)), 5)
            entry["mean_joint_jump_rad"] = round(float(np.mean(jump)), 5)
        z = np.asarray(clip["root_pos_z"], dtype=np.float64)
        if 0 < frame < z.shape[0]:
            entry["root_z_jump_m"] = round(
                float(abs(z[frame] - z[frame - 1])), 5)
        seams.append(entry)

    jv = clip.get("joint_vel")
    peak_joint_vel = (round(float(np.max(np.abs(np.asarray(jv)))), 4)
                      if jv is not None else None)
    return {
        "seams": seams,
        "peak_joint_vel_rad_s": peak_joint_vel,
        "peak_root_vel_z_m_s": round(
            float(np.max(np.abs(np.asarray(clip["root_vel_z"])))), 4)
        if clip.get("root_vel_z") is not None else None,
        "duration_s": round(
            float(np.asarray(clip["root_pos_z"]).shape[0]) / fps, 4),
    }


# ── public API ──────────────────────────────────────────────────────────
def compose_reference(
    segments: Iterable[Mapping[str, Any]],
    *,
    target_fps: Optional[float] = None,
    blend_s: float = DEFAULT_BLEND_S,
    max_seam_joint_jump_rad: float = DEFAULT_MAX_SEAM_JOINT_JUMP_RAD,
    max_joint_vel_rad_s: float = DEFAULT_MAX_JOINT_VEL_RAD_S,
    strict: bool = True,
) -> dict:
    """Compose one candidate reference clip from spans of several solved clips.

    `segments` is an ordered sequence of mappings:

        {"clip": <clip dict>,            # required, validate_clip-clean
         "t_start_s": float,             # optional, defaults to 0
         "t_end_s": float,               # optional, defaults to clip end
         "label": str,                   # optional, for provenance
         "source_id": str}               # optional, for provenance

    Returns a `validate_clip`-clean clip whose `meta["composition"]` records
    every source, frame range, and SE(2) transform. Raises `ComposeError`
    when segments are incompatible, or (when `strict`) when a seam exceeds
    `max_seam_joint_jump_rad` or the composite exceeds
    `max_joint_vel_rad_s` — thresholds that exist because an
    over-discontinuous composite burns a full Tier-D tracking run to tell you
    what a cheap kinematic check already knew.

    The result is a CANDIDATE. Certify it with `sculptor.refs.track` before
    any reward is authored against it.
    """
    from sculptor.reference import validate_clip
    from sculptor.refs.spans import crop_span

    raw = list(segments)
    if len(raw) < 2:
        raise ComposeError(
            f"compose_reference needs >= 2 segments, got {len(raw)}; a single "
            "span is already handled by refs.spans.crop_span")

    # 1. crop
    cropped: list[dict] = []
    provenance: list[dict[str, Any]] = []
    for i, spec in enumerate(raw):
        clip = spec.get("clip")
        if not isinstance(clip, Mapping):
            raise ComposeError(f"segment {i} has no 'clip' mapping")
        clip = dict(clip)
        errors = validate_clip(clip)
        if errors:
            raise ComposeError(
                f"segment {i} source clip is invalid:\n  - "
                + "\n  - ".join(errors))
        n = int(np.asarray(clip["root_pos_z"]).shape[0])
        fps = float(clip["fps"])
        t0 = float(spec.get("t_start_s", 0.0) or 0.0)
        t1 = float(spec.get("t_end_s") if spec.get("t_end_s") is not None
                   else n / fps)
        piece = crop_span(clip, t0, t1) if (t0 > 0.0 or t1 < n / fps) else clip
        cropped.append(dict(piece))
        provenance.append({
            "index": i,
            "label": str(spec.get("label") or f"segment_{i}"),
            "source_id": str(spec.get("source_id")
                             or (clip.get("meta") or {}).get("clip_id")
                             or "unknown"),
            "source_fps": fps,
            "source_frames": [int(round(t0 * fps)), int(round(t1 * fps))],
            "source_span_s": [round(t0, 4), round(t1, 4)],
        })

    # 2. one fps for everything
    fps_out = float(target_fps if target_fps
                    else max(float(c["fps"]) for c in cropped))
    resampled = [_resample(c, fps_out) for c in cropped]

    # 3. one joint ordering
    base_names = list(resampled[0].get("joint_names") or [])
    if base_names:
        _align_joints(resampled, base_names)

    # 4. SE(2)-align + blend, tracking where each seam lands
    n_blend = int(round(max(0.0, blend_s) * fps_out))
    acc = resampled[0]
    provenance[0]["se2"] = {"d_yaw_rad": 0.0, "anchor_xy": [0.0, 0.0]}
    seam_frames: list[int] = []
    for i in range(1, len(resampled)):
        quat = acc.get("root_quat_wxyz")
        xy = acc.get("root_pos_xy")
        anchor_yaw = (float(_quat_yaw(np.asarray(quat)[-1]))
                      if quat is not None else 0.0)
        anchor_xy = (np.asarray(xy, dtype=np.float64)[-1]
                     if xy is not None else np.zeros(2))
        aligned = _se2_align(resampled[i], anchor_xy, anchor_yaw)
        provenance[i]["se2"] = aligned.pop("_se2", {})
        n_acc = int(np.asarray(acc["root_pos_z"]).shape[0])
        seam_frames.append(max(0, n_acc - n_blend))
        acc = _blend(acc, aligned, n_blend)

    # 5. velocities follow the composed positions, never the sources
    acc = _recompute_velocities(acc)
    acc.pop("_se2", None)

    report = seam_report(acc, seam_frames)
    acc["meta"] = {
        **(acc.get("meta") or {}),
        "composition": {
            "schema_version": 1,
            "segments": provenance,
            "target_fps": fps_out,
            "blend_s": round(float(n_blend) / fps_out, 4),
            "blend_frames": n_blend,
            "seam_frames": seam_frames,
            "seam_report": report,
            "certified": False,
            "note": (
                "Kinematic composite of solved spans. Dynamically UNVERIFIED: "
                "momentum is not conserved across a seam. Certify with "
                "sculptor.refs.track before authoring a reward against it."),
        },
    }

    errors = validate_clip(acc)
    if errors:
        raise ComposeError(
            "composed clip is invalid:\n  - " + "\n  - ".join(errors))

    if strict:
        worst = max((s.get("max_joint_jump_rad", 0.0) for s in report["seams"]),
                    default=0.0)
        if worst > max_seam_joint_jump_rad:
            raise ComposeError(
                f"seam discontinuity {worst:.3f} rad exceeds "
                f"{max_seam_joint_jump_rad:.3f} rad. The spans do not meet "
                "kinematically — widen blend_s, pick spans whose boundary "
                "poses agree, or reorder them.")
        peak = report.get("peak_joint_vel_rad_s")
        if peak is not None and peak > max_joint_vel_rad_s:
            raise ComposeError(
                f"composite peak joint velocity {peak:.2f} rad/s exceeds "
                f"{max_joint_vel_rad_s:.2f} rad/s; the blend is too short for "
                "how far the poses are apart.")
    return acc

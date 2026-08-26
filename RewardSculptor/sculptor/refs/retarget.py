"""Cross-robot retargeting wrapper around GMR (`~/tools/GMR`, MIT license)
— §REFERENCE_TRAJECTORY_PLAN R3.

GMR is a separate Python package with its own venv (py3.10, `~/tools/GMR/
.venv`) and a dependency set (`mink`, a pinned `mujoco`, `smplx`, `torch`)
that does not co-install with the sculptor package (py3.13). This module
NEVER imports `general_motion_retargeting` directly — it shells out to
`sculptor/refs/_gmr_driver.py` running under GMR's own interpreter,
exchanging arguments/results as JSON files (§R3 design: "cross-venv
execution... JSON-over-files").

Public API: `retarget_clip` runs the driver and returns a result dict;
`retarget_and_register` additionally converts the result into OUR clip
container (`sculptor.reference` schema) and registers it in the reference
library (`sculptor.refs.library`) with retarget provenance.

Robot-name mapping: GMR's robot identifiers (`general_motion_retargeting.
params.ROBOT_XML_DICT` keys, e.g. `"unitree_g1"`, `"booster_t1_29dof"`) are
NOT the same strings as our library's `<robot>` slug (`"g1"`, `"t1"`, ...).
`GMR_ROBOT_IDS` is the verified map (read directly off GMR's own
`params.py` during this build — see module docstring in
`REFERENCE_TRAJECTORY_PLAN.md` §11 R3 for the acceptance bar this maps
toward). `unitree_h1` (plain H1) is INTENTIONALLY ABSENT: GMR ships no BVH
IK config for it (`bvh_lafan1`/`bvh_xsens` source-format dicts in GMR's
`params.py` have no `"unitree_h1"` entry — only `smplx_to_h1.json`
exists), and SMPL-X retargeting requires an auth-gated SMPL-X body model
download (smpl-x.is.tue.mpg.de account) this project's hard rules forbid
obtaining. `unitree_h1_2` (H1 gen-2, a real distinct 27-DOF Unitree
morphology) DOES have a shipped, pre-tuned `bvh_xsens` IK config, but the
`bvh_xsens` bone vocabulary differs from `bvh_lafan1`'s (verified: `Hips`/
`LeftUpLeg`/`LeftArm`/... vs `Hips`/`LeftHip`/`LeftShoulder`/...) so the
SAME LAFAN1 clip cannot drive it without hand-authoring new IK-calibration
constants (out of this module's scope — see the plan's "extend NOTHING
silently" rule). The cross-robot proof therefore uses G1 + Booster T1
(29-DOF) — both driven by the IDENTICAL, GMR-shipped `bvh_lafan1` IK
configs and human-bone vocabulary, zero new calibration authored.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sculptor.reference import save_clip, validate_clip
from sculptor.refs import library

#: Our library robot slug -> GMR's `tgt_robot` identifier
#: (`general_motion_retargeting.params.ROBOT_XML_DICT` key). Verified by
#: reading `~/tools/GMR/general_motion_retargeting/params.py` directly.
GMR_ROBOT_IDS: dict[str, str] = {
    "g1": "unitree_g1",
    "g1_hands": "unitree_g1_with_hands",
    "h1_2": "unitree_h1_2",
    "t1": "booster_t1_29dof",
}

#: `robot` slugs whose GMR id has a `bvh_lafan1` IK config, i.e. can be
#: driven by a plain LAFAN1-format BVH file with no new IK-config
#: authoring. Verified against `IK_CONFIG_DICT["bvh_lafan1"]` in GMR's
#: `params.py` during this build.
BVH_LAFAN1_CAPABLE_ROBOTS: frozenset[str] = frozenset({"g1", "g1_hands", "t1"})

DEFAULT_GMR_ROOT = Path.home() / "tools" / "GMR"
_DRIVER_PATH = Path(__file__).parent / "_gmr_driver.py"


class RetargetError(RuntimeError):
    """The GMR driver subprocess failed or returned ok=False."""


def default_gmr_python(gmr_root: Optional[Path] = None) -> Path:
    root = gmr_root or DEFAULT_GMR_ROOT
    candidate = root / ".venv" / "bin" / "python"
    if not candidate.is_file():
        # Windows-hosted venv layout, in case this ever runs off WSL.
        candidate = root / ".venv" / "Scripts" / "python.exe"
    return candidate


def gmr_robot_id(robot: str) -> str:
    try:
        return GMR_ROBOT_IDS[robot]
    except KeyError as e:
        raise ValueError(
            f"no GMR robot id mapped for library robot slug {robot!r}; "
            f"known slugs: {sorted(GMR_ROBOT_IDS)}") from e


def retarget_clip(
    source_path: Path,
    source_format: str,
    robot: str,
    out_dir: Path,
    *,
    gmr_python: Optional[Path] = None,
    bvh_format: str = "lafan1",
    fps: float = 30.0,
    smplx_body_model_dir: Optional[Path] = None,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Run ONE clip through GMR, in GMR's own venv, via a subprocess.

    Returns `{"ok", "robot", "out_npz", "stats", "tool_version", "error"}`
    — see `_gmr_driver.py`'s module docstring for the exact contract.
    Never raises on a driver-reported failure (`ok=False`); raises
    `RetargetError` only for a process-launch failure (interpreter
    missing, timeout, or the driver produced no result file at all —
    something that prevented the driver from even reporting an error).
    """
    if source_format not in ("bvh", "smplx"):
        raise ValueError(f"source_format must be 'bvh' or 'smplx', got {source_format!r}")
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"source clip not found: {source_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = Path(gmr_python) if gmr_python is not None else default_gmr_python()
    if not py.is_file():
        raise RetargetError(
            f"GMR python interpreter not found at {py} — pass "
            f"gmr_python= explicitly or install GMR's venv at "
            f"{DEFAULT_GMR_ROOT}/.venv")

    gmr_robot = gmr_robot_id(robot)
    out_npz = out_dir / f"{robot}_retarget_raw.npz"

    with tempfile.TemporaryDirectory(prefix="rs_retarget_") as tmp:
        tmp_path = Path(tmp)
        args_path = tmp_path / "args.json"
        result_path = tmp_path / "result.json"
        args: dict[str, Any] = {
            "source_path": str(source_path),
            "source_format": source_format,
            "gmr_robot": gmr_robot,
            "out_npz": str(out_npz),
            "fps": fps,
        }
        if source_format == "bvh":
            args["bvh_format"] = bvh_format
        else:
            if smplx_body_model_dir is None:
                raise ValueError(
                    "smplx_body_model_dir is required when source_format='smplx'")
            args["smplx_body_model_dir"] = str(smplx_body_model_dir)
        args_path.write_text(json.dumps(args), encoding="utf-8")

        proc = subprocess.run(
            [str(py), str(_DRIVER_PATH), str(args_path), str(result_path)],
            cwd=str(py.parent.parent),  # GMR repo root (ik_configs are relative in params.py)
            capture_output=True, text=True, timeout=timeout_s)

        if not result_path.is_file():
            raise RetargetError(
                f"GMR driver produced no result file (exit={proc.returncode}); "
                f"stdout tail:\n{proc.stdout[-2000:]}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}")
        result = json.loads(result_path.read_text(encoding="utf-8"))

        # Copy the driver's npz out of the tempdir-adjacent location (it
        # was already written to `out_dir`, not the tempdir, so nothing
        # to move — `out_npz` is already the final path). Kept as an
        # explicit check so a driver bug that ignores `out_npz` is caught
        # here rather than silently returning a stale/missing file.
        if result.get("ok") and not out_npz.is_file():
            result["ok"] = False
            result["error"] = (
                (result.get("error") or "")
                + f"; driver reported ok=True but {out_npz} does not exist")
        return result


# ── GMR raw output -> our canonical clip container ─────────────────────
def gmr_quat_wxyz_to_container(quat_wxyz: np.ndarray) -> np.ndarray:
    """Identity pass-through, kept as an explicit, unit-tested seam.

    GMR's `retarget()` returns MuJoCo qpos, whose orientation slice
    `qpos[3:7]` is documented (README §"Human/Robot Motion Data
    Formulation": "quaternion (with wxyz order by default, to align with
    mujoco)") and independently confirmed by `motion_retarget.py`'s own
    `R.from_quat(..., scalar_first=True)` calls and the example scripts'
    "save from wxyz to xyzw" comment before their `[[1,2,3,0]]` reindex.
    Our container (`sculptor.reference`) is ALSO wxyz
    (`root_quat_wxyz`) — so this conversion is a no-op BY VALUE, not by
    assumption. Kept as a real function (not inlined) so a future GMR
    version that changes convention has one seam to patch, and so the
    "deliberate conversion" the plan calls for is visible and testable
    rather than implicit."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"expected (T, 4) wxyz array, got shape {quat_wxyz.shape}")
    norms = np.linalg.norm(quat_wxyz, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise ValueError(
            f"GMR quaternion output is not unit-norm (min={norms.min():.4f}, "
            f"max={norms.max():.4f}) — refusing to trust the wxyz-passthrough "
            f"assumption on non-normalized input")
    return quat_wxyz / norms[:, None]


#: Same retargeting-noise band `sculptor.refs.ingest._clamp_root_z_noise`
#: clamps for real fleaven floor-contact clips (see that module's
#: docstring — a lying/floor-contact IK solve can settle a hair below
#: z=0). Verified reproduced here on a REAL GMR output: the LAFAN1
#: fallAndGetUp1_subject1 clip retargeted to booster_t1_29dof dips to
#: -0.016 m (226/5047 frames), well inside this band — not a T1-specific
#: bug, the same class of artifact, same fix, same constant.
_ROOT_Z_CLAMP_NOISE_FLOOR_M = -0.05
_ROOT_Z_CLAMP_TARGET_M = 1e-4


def _clamp_root_z_noise(z: np.ndarray) -> Optional[dict[str, Any]]:
    z_min = float(z.min())
    if not (_ROOT_Z_CLAMP_NOISE_FLOOR_M <= z_min <= 0.0):
        return None
    mask = z < _ROOT_Z_CLAMP_TARGET_M
    n_frames = int(mask.sum())
    z[:] = np.maximum(z, _ROOT_Z_CLAMP_TARGET_M)
    return {"n_frames": n_frames, "min_before": round(z_min, 6)}


def gmr_result_to_clip(out_npz: Path, *, source_meta: dict[str, Any]) -> dict[str, Any]:
    """Convert the driver's raw npz (`root_pos`, `root_quat_wxyz`,
    `joint_pos`, `joint_names`, `fps`) into a `sculptor.reference`-shaped
    clip dict, ready for `validate_clip`/`save_clip`. A root_z minimum in
    `[-0.05, 0]` m (retargeting float noise during floor contact — same
    band `ingest.py` clamps for real dataset clips) is clamped to a tiny
    positive epsilon; anything below that band is a real below-ground
    pose and is left for `validate_clip` to reject."""
    with np.load(out_npz, allow_pickle=False) as z:
        root_pos = z["root_pos"].astype(np.float64)
        quat_wxyz = z["root_quat_wxyz"].astype(np.float64)
        joint_pos = z["joint_pos"].astype(np.float64)
        joint_names = [str(n) for n in z["joint_names"]]
        fps = float(z["fps"])

    quat_wxyz = gmr_quat_wxyz_to_container(quat_wxyz)
    root_z_clamped = _clamp_root_z_noise(root_pos[:, 2])
    meta = {"source": "retarget:gmr", **source_meta}
    if root_z_clamped is not None:
        meta["root_z_clamped"] = root_z_clamped
    clip: dict[str, Any] = {
        "root_pos_z": root_pos[:, 2].copy(),
        "root_pos_xy": root_pos[:, :2].copy(),
        "root_quat_wxyz": quat_wxyz,
        "joint_pos": joint_pos,
        "joint_names": joint_names,
        "fps": fps,
        "meta": meta,
    }
    errors = validate_clip(clip)
    if errors:
        raise RetargetError(
            "GMR retarget output failed clip schema validation: "
            + "; ".join(errors))
    return clip


def retarget_and_register(
    source_path: Path,
    source_format: str,
    robot: str,
    *,
    clip_id: Optional[str] = None,
    license_: str,
    attribution: str,
    text: str = "",
    labels: Optional[list[str]] = None,
    gmr_python: Optional[Path] = None,
    bvh_format: str = "lafan1",
    fps: float = 30.0,
    root: Optional[Path] = None,
    tool_version_fallback: str = "unknown",
) -> "library.LibraryClip":
    """Run `retarget_clip`, convert the result, and register it in the
    reference library with retarget provenance (§2.3/§2.4). Raises
    `RetargetError` on any driver failure — never partially registers a
    clip (provenance is written last, after `save_clip` succeeds, mirroring
    `ingest.py`'s ordering)."""
    source_path = Path(source_path)
    work_dir = Path(tempfile.mkdtemp(prefix="rs_retarget_out_"))
    result = retarget_clip(
        source_path, source_format, robot, work_dir,
        gmr_python=gmr_python, bvh_format=bvh_format, fps=fps)
    if not result.get("ok"):
        raise RetargetError(
            f"GMR retarget failed for robot={robot!r}: {result.get('error')}")

    stats = result["stats"]
    source_meta = {
        "source_path": str(source_path),
        "source_format": source_format,
        "gmr_robot": result["robot"],
        "stats": stats,
    }
    clip = gmr_result_to_clip(Path(result["out_npz"]), source_meta=source_meta)

    cid = clip_id or (library.slugify(source_path.stem) + f"--{robot}")
    library.validate_clip_id(cid)
    source_sha = library.content_sha256(source_path.read_bytes())
    d = library.clip_dir(robot, cid, root=root)
    clip_path = save_clip(d / library.CLIP_FILENAME, clip)
    artifact_sha = library.content_sha256(clip_path.read_bytes())
    from sculptor.reference_clock import reference_playback_duration_s

    prov = library.make_provenance(
        clip_id=cid, robot=robot,
        source={
            "kind": "retarget", "source_path": str(source_path),
            "source_format": source_format,
        },
        license=license_, attribution=attribution,
        content_sha256_=artifact_sha,
        source_content_sha256_=source_sha,
        retarget={
            "tool": "GMR", "version": result.get("tool_version") or tool_version_fallback,
            "gmr_robot": result["robot"], "source_format": source_format,
            "bvh_format": bvh_format if source_format == "bvh" else None,
            "notes": "GMR (github.com/YanjieZe/GMR, MIT) IK retargeting via "
                     "the shipped, pre-tuned IK config for this "
                     "(source_format, robot) pair — no new calibration "
                     "authored by this pipeline.",
        },
        tier="K", fps_source=stats.get("fps"),
        joint_mapping={"identity": False, "source": "gmr_model_dof_names"},
        labels=list(labels or []), text=text,
        qc={
            "n_frames": stats.get("n_frames"),
            "duration_s": round(reference_playback_duration_s(
                frame_count=int(stats.get("n_frames", 0)),
                fps=float(stats.get("fps", 0.0))), 4),
            # §library._index_row_from_provenance reads qc["root_z_range"]
            # (the index-row contract `ingest.py` also writes to) — this is
            # the POST-clamp range (what's actually on disk in clip.npz);
            # `root_z_range_raw` keeps the pre-clamp GMR output for audit.
            "root_z_range": [
                round(float(clip["root_pos_z"].min()), 4),
                round(float(clip["root_pos_z"].max()), 4),
            ],
            "root_z_range_raw": stats.get("root_z_range"),
            "root_z_clamped": clip["meta"].get("root_z_clamped"),
        },
    )

    prov_path = library.write_provenance(robot, cid, prov, root=root)
    return library.LibraryClip(
        robot=robot, clip_id=cid, clip_path=d / library.CLIP_FILENAME,
        provenance_path=prov_path, provenance=prov)


def attach_role_resolution_qc(
    robot: str, clip_id: str, required_roles: list[str], *, root: Optional[Path] = None,
) -> dict[str, Any]:
    """Run `resolve_joint_roles` against a registered clip's own
    `joint_names` and persist the result under `provenance.qc.
    joint_role_resolution` (§mission: "store resolved role names in
    provenance qc"). Returns the resolution summary dict; does not raise
    on an unresolved role (that is itself the finding to report) — only
    on a missing clip/provenance."""
    from sculptor.eval.joint_resolver import resolve_joint_roles
    from sculptor.reference import load_clip

    clip_path = library.clip_dir(robot, clip_id, root=root) / library.CLIP_FILENAME
    clip = load_clip(clip_path)
    joint_names = clip.get("joint_names") or []
    res = resolve_joint_roles(joint_names, required_roles)
    summary = {
        "required_roles": list(required_roles),
        "resolved": {k: joint_names[i] for k, i in res.resolved.items()},
        "missing": list(res.missing),
        "ambiguous": {k: [joint_names[i] for i in v] for k, v in res.ambiguous.items()},
        "ok": res.ok,
    }
    prov = library.read_provenance(robot, clip_id, root=root)
    qc = dict(prov.get("qc") or {})
    qc["joint_role_resolution"] = summary
    prov["qc"] = qc
    library.write_provenance(robot, clip_id, prov, root=root)
    return summary

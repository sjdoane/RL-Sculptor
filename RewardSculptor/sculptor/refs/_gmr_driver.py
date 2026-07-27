"""Driver script executed INSIDE GMR's own venv (py3.10, `~/tools/GMR/
.venv`) — never imported by the sculptor package (py3.13; GMR's `mink`/
`mujoco`-pinned dependency set does not co-install there). `retarget.py`
shells out to this script via `gmr_python _gmr_driver.py <args_json>
<result_json>` (JSON-over-files, per §REFERENCE_TRAJECTORY_PLAN R3 design)
so the two venvs never need to share an import boundary.

Contract (all paths absolute, written by the caller):
  args_json:   {"source_path": str, "source_format": "bvh"|"smplx",
                "bvh_format": "lafan1"|"nokov"|"xsens" (bvh only),
                "gmr_robot": str, "out_npz": str}
  result_json: written by THIS script on both success and failure —
      {"ok": bool, "robot": str, "out_npz": str|None,
       "stats": {...} | None, "tool_version": str|None,
       "error": str|None}
  Exit code 0 always (errors are reported via result_json, not stderr/exit
  code) so the sculptor-side caller has one parsing path.

Output npz keys (consumed by `retarget.py._gmr_result_to_clip`):
  root_pos (T,3), root_quat_wxyz (T,4), joint_pos (T,J) [radians],
  joint_names (J,) [str], fps (scalar).
"""
from __future__ import annotations

import json
import sys
import traceback


def _robot_joint_names(model) -> list[str]:
    """Every hinge joint name in qpos[7:] order, robot-agnostic.

    Joint 0 is always the robot's floating-base free joint (`jnt_type ==
    mjtJoint.mjJNT_FREE`) regardless of robot — every GMR asset XML is
    built this way (verified against `unitree_g1` and `booster_t1_29dof`
    during this build: `jnt_type[0] == 0`, every subsequent joint type
    `== 3` i.e. hinge). Reads `model.joint(i).name` directly rather than
    trusting `GeneralMotionRetargeting.robot_dof_names` (that dict's key
    for the free joint is the ROOT BODY name on some robots and `None` on
    others — verified different between unitree_g1 ('pelvis') and
    booster_t1_29dof (None) in this build — so it is not a safe
    robot-agnostic root-skip signal)."""
    import mujoco

    names: list[str] = []
    for j in range(model.njnt):
        jtype = model.jnt_type[j]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if j == 0:
            if jtype != mujoco.mjtJoint.mjJNT_FREE:
                raise RuntimeError(
                    f"expected joint 0 to be the free-base joint "
                    f"(mjJNT_FREE), got type {jtype!r} name {name!r}")
            continue
        if jtype != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(
                f"unexpected non-hinge joint {name!r} (type {jtype!r}) "
                f"at index {j} — driver only knows how to read a "
                f"free-base + all-hinge-joints skeleton")
        names.append(name or f"joint_{j}")
    return names


def _gmr_tool_version(gmr_pkg) -> str:
    """GMR ships no `__version__` attribute and (in this environment) no
    installed package metadata (`importlib.metadata.version` raises) —
    verified during this build. Fall back to parsing the `version=`
    kwarg out of the repo's own `setup.py` (matches the README's version
    badge, "0.2.0" at build time), then to a literal "unknown" if even
    that file is missing/unparsable — never raises."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("general_motion_retargeting")
        except PackageNotFoundError:
            pass
    except ImportError:  # pragma: no cover — py<3.8 only
        pass
    try:
        import pathlib
        import re

        pkg_dir = pathlib.Path(gmr_pkg.__file__).resolve().parent
        setup_py = pkg_dir.parent / "setup.py"
        text = setup_py.read_text(encoding="utf-8")
        m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', text)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "unknown"


def _run(args: dict) -> dict:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    import general_motion_retargeting as gmr_pkg
    import numpy as np

    source_format = args["source_format"]
    gmr_robot = args["gmr_robot"]
    source_path = args["source_path"]

    if source_format == "bvh":
        bvh_format = args.get("bvh_format", "lafan1")
        if bvh_format not in ("lafan1", "nokov"):
            raise ValueError(
                f"driver only supports bvh_format lafan1|nokov "
                f"(xsens uses a different loader/CLI), got {bvh_format!r}")
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        frames, human_height = load_bvh_file(source_path, format=bvh_format)
        src_human = f"bvh_{bvh_format}"
        fps = float(args.get("fps", 30.0))
    elif source_format == "smplx":
        from general_motion_retargeting.utils.smpl import (
            get_smplx_data_offline_fast, load_smplx_file)

        smplx_folder = args["smplx_body_model_dir"]
        smplx_data, body_model, smplx_output, human_height = load_smplx_file(
            source_path, smplx_folder)
        tgt_fps = float(args.get("fps", 30.0))
        frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        src_human = "smplx"
        fps = float(aligned_fps)
    else:
        raise ValueError(f"unknown source_format {source_format!r}")

    retargeter = GMR(
        src_human=src_human, tgt_robot=gmr_robot,
        actual_human_height=human_height, verbose=False)
    joint_names = _robot_joint_names(retargeter.model)

    root_pos = np.zeros((len(frames), 3), dtype=np.float64)
    root_quat_wxyz = np.zeros((len(frames), 4), dtype=np.float64)
    joint_pos = np.zeros((len(frames), len(joint_names)), dtype=np.float64)
    for i, frame in enumerate(frames):
        qpos = retargeter.retarget(frame)
        root_pos[i] = qpos[0:3]
        root_quat_wxyz[i] = qpos[3:7]          # GMR/mujoco convention: wxyz
        joint_pos[i] = qpos[7:7 + len(joint_names)]

    out_npz = args["out_npz"]
    np.savez_compressed(
        out_npz,
        root_pos=root_pos.astype(np.float32),
        root_quat_wxyz=root_quat_wxyz.astype(np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_names=np.array(joint_names),
        fps=np.float32(fps),
    )

    tool_version = _gmr_tool_version(gmr_pkg)
    return {
        "ok": True,
        "robot": gmr_robot,
        "out_npz": out_npz,
        "stats": {
            "n_frames": int(len(frames)),
            "fps": fps,
            "n_joints": len(joint_names),
            "joint_names": joint_names,
            "src_human": src_human,
            "actual_human_height": float(human_height),
            "root_z_range": [
                round(float(root_pos[:, 2].min()), 4),
                round(float(root_pos[:, 2].max()), 4),
            ],
        },
        "tool_version": tool_version,
        "error": None,
    }


def main() -> None:
    args_path, result_path = sys.argv[1], sys.argv[2]
    with open(args_path, encoding="utf-8") as f:
        args = json.load(f)
    try:
        result = _run(args)
    except Exception as e:  # noqa: BLE001 — always report via result_json
        result = {
            "ok": False,
            "robot": args.get("gmr_robot"),
            "out_npz": None,
            "stats": None,
            "tool_version": None,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

"""Render 320×240 iso-view thumbnails for every robot_library.yml entry.

Usage:
    cd ~/projects/reward-sculptor-ui
    uv run python scripts/generate_library_thumbnails.py

Outputs `frontend/public/robots/<slug>.png` — one per library entry.
The library YAML declares `thumbnail_path: robots/<slug>.webp`, but the
GET /library/robots/{slug}/thumbnail route falls back to `.png` when
`.webp` is absent, so `.png` is the source-of-truth format and is
committed to git.

CPU-only rendering (M3 spec §6). MuJoCo picks the GL backend
automatically; WSLg provides glfw via Wayland and works out of the box.
If rendering fails on this machine specifically, try setting
MUJOCO_GL=osmesa (requires `apt install libosmesa6-dev`).

Camera framing: reuses `backend.services.preview_renderer._build_camera`
(bbox excluding plane geoms, COM-lifted center). Arms and hands get a
20% camera-distance reduction so small end-effectors don't frame too
tightly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from backend.services.preview_renderer import (  # noqa: E402
    _build_camera,
    resolve_library_xml,
)
from backend.services.robot_library import RobotEntry, load_library  # noqa: E402


THUMB_W = 320
THUMB_H = 240
ARM_LIKE = ("Arm", "Gripper_Hand")
OUT_DIR = _ROOT / "frontend" / "public" / "robots"


def _resolve_xml_path(entry: RobotEntry) -> Optional[Path]:
    """Find the MJCF for this entry, or None if it can't be rendered."""
    if entry.source == "menagerie":
        if not entry.menagerie_package:
            return None
        # robot_descriptions submodules are lazy — must be imported by
        # dotted name, not attribute-accessed on the top-level package.
        import importlib
        try:
            mod = importlib.import_module(
                f"robot_descriptions.{entry.menagerie_package}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"   import robot_descriptions.{entry.menagerie_package}"
                  f" failed: {type(e).__name__}: {e}")
            return None
        mjcf = getattr(mod, "MJCF_PATH", None)
        if mjcf:
            return Path(mjcf)
        return None

    if entry.source == "gymnasium_builtin":
        if not entry.preconfigured_tasks:
            return None
        env_id = entry.preconfigured_tasks[0].task_id
        return resolve_library_xml(env_id)

    if entry.source == "mjlab_builtin":
        try:
            import mjlab
            mjlab_dir = Path(mjlab.__file__).resolve().parent
            candidate = mjlab_dir / "tasks" / "cartpole" / "cartpole.xml"
            if candidate.is_file():
                return candidate
        except Exception as e:  # noqa: BLE001
            print(f"   mjlab import error: {type(e).__name__}: {e}")
        return None

    return None


def _render_entry(entry: RobotEntry, xml_path: Path, out_path: Path) -> None:
    """Render one thumbnail. Raises on any MuJoCo failure."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = _build_camera(model, data, "iso")
    if entry.category in ARM_LIKE:
        cam.distance *= 0.8

    with mujoco.Renderer(model, height=THUMB_H, width=THUMB_W) as renderer:
        renderer.update_scene(data, cam)
        frame = renderer.render()

    frame = np.asarray(frame, dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out_path, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-slug", default=None,
        help="Render just this one slug (useful for debugging).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-render even if the output PNG already exists.",
    )
    args = parser.parse_args()

    library_path = _ROOT / "backend" / "data" / "robot_library.yml"
    lib = load_library(library_path)

    rendered: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    entries = sorted(lib.entries_by_slug.values(), key=lambda e: e.slug)
    for entry in entries:
        if args.only_slug and entry.slug != args.only_slug:
            continue

        out_path = OUT_DIR / f"{entry.slug}.png"
        if out_path.exists() and not args.force:
            print(f"[=]  {entry.slug} (already rendered)")
            rendered.append(entry.slug)
            continue

        print(f"[*]  {entry.slug} ({entry.category})")
        xml_path = _resolve_xml_path(entry)
        if xml_path is None:
            skipped.append((entry.slug, "no MJCF available"))
            print(f"     SKIP — no MJCF")
            continue

        try:
            _render_entry(entry, xml_path, out_path)
            size_kb = out_path.stat().st_size / 1024
            rendered.append(entry.slug)
            print(f"     OK -> {out_path.relative_to(_ROOT)} ({size_kb:.1f} KB)")
        except Exception as e:  # noqa: BLE001
            failed.append((entry.slug, f"{type(e).__name__}: {e}"))
            print(f"     FAIL — {type(e).__name__}: {e}")

    print()
    print(f"Rendered: {len(rendered)}")
    print(f"Skipped:  {len(skipped)}")
    for slug, reason in skipped:
        print(f"  - {slug}: {reason}")
    print(f"Failed:   {len(failed)}")
    for slug, reason in failed:
        print(f"  - {slug}: {reason}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

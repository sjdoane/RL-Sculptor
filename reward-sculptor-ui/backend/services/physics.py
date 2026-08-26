"""MJCF physics-editor service (M7 Phase 5).

Resolves the active MJCF for a project, parses it for a user-facing
summary (option + joint defaults + actuators + geom friction), and
provides a Claude-backed `apply_prompt_edit` that rewrites the full
XML from a natural-language request.

Rules:
  - Library projects whose MJCF lives in the Menagerie package are
    read from the menagerie path UNTIL the first physics edit, at
    which point the library file is copied into
    `<project>/uploads/robot/base.xml` so the git history captures
    every change. Subsequent edits touch the project-local copy.
  - Upload projects always edit their existing
    `<project>/uploads/robot/<file>.xml`.
  - Every edit is validated via `mujoco.MjModel.from_xml_string(new_xml)`
    before the write is committed to disk. Failed validation leaves
    the previous file untouched.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional

log = logging.getLogger(__name__)

# Where an upload-kind project keeps its committed MJCF.
UPLOAD_ROBOT_DIR = "uploads/robot"


@dataclass
class MjcfSummary:
    """A human-digestible snapshot of a MuJoCo model's editable fields.

    Consumed by the UI's read-only "current parameters" panel (the
    right column of the Physics tab). Always rounded to 6 significant
    figures to keep the JSON payload compact.
    """

    timestep: Optional[float] = None
    gravity: Optional[list[float]] = None  # length-3
    integrator: Optional[str] = None
    solver: Optional[str] = None
    joint_defaults: dict[str, Any] = None  # type: ignore[assignment]
    joints: list[dict[str, Any]] = None  # type: ignore[assignment]
    actuators: list[dict[str, Any]] = None  # type: ignore[assignment]
    geoms: list[dict[str, Any]] = None  # type: ignore[assignment]
    # Populated when mujoco refuses the MJCF (missing meshes, malformed
    # XML, etc.). The frontend surfaces this as an inline warning with
    # a "Re-materialize" button — silent fallback to empty fields was
    # the Phase 5 bug that hid the missing-assets symptom.
    parse_error: Optional[str] = None
    # The rates TRAINING actually runs at, plus advisory findings. Distinct
    # from `timestep` above, which is only what the MJCF compiles to — see
    # `resolve_timing`. None when the task can't be resolved (never a guess).
    timing: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "gravity": self.gravity,
            "integrator": self.integrator,
            "solver": self.solver,
            "joint_defaults": self.joint_defaults or {},
            "joints": self.joints or [],
            "actuators": self.actuators or [],
            "geoms": self.geoms or [],
            "parse_error": self.parse_error,
            "timing": self.timing,
        }


@dataclass
class MjcfLoadResult:
    xml_source: str
    summary: MjcfSummary
    mjcf_source_kind: str  # "library" | "upload"
    mjcf_path: Path
    materialized: bool  # True when reading from <project>/uploads/robot/*


# ── path resolution ──────────────────────────────────────────────────
def resolve_mjcf_path(project_dir: Path) -> tuple[Optional[Path], str]:
    """Return (path, kind) where `kind` is `"upload"` | `"library"` |
    `"none"`. `path` is None when no MJCF is available.

    Precedence: if the project has a materialized robot at
    `<project>/uploads/robot/*.xml`, use it. Otherwise resolve from the
    project's metadata.robot_source (library projects → menagerie path).
    """
    # Upload kind — prefer a locally-committed XML.
    upload_dir = project_dir / UPLOAD_ROBOT_DIR
    if upload_dir.is_dir():
        xmls = sorted(upload_dir.glob("*.xml"))
        if xmls:
            return xmls[0], "upload"

    # Library kind — resolve via menagerie_package.
    lib = _resolve_library_mjcf(project_dir)
    if lib is not None:
        return lib, "library"
    return None, "none"


def _resolve_library_mjcf(project_dir: Path) -> Optional[Path]:
    """Return the original MJCF path for a library project, independent
    of whether it's been materialized locally. Dispatches on the
    library entry's `source` — handles `menagerie` (via
    robot_descriptions) and `mjlab_builtin` (via the mjlab package
    itself). Returns None for upload-only projects, unresolvable
    library slugs, or `gymnasium_builtin` (Gym envs don't expose
    MJCFs)."""
    meta = _read_metadata(project_dir)
    if meta is None:
        return None
    rs = meta.get("robot_source") or {}
    library_slug = rs.get("library_slug") or rs.get("library_name")
    if not library_slug:
        return None
    try:
        from backend.services.robot_library import get_library

        lib = get_library()
        entry = lib.get_robot(library_slug)
        if entry is None:
            log.warning(
                "_resolve_library_mjcf: no library entry for slug %r",
                library_slug,
            )
            return None
        if entry.source == "menagerie":
            return lib.resolve_menagerie_path(library_slug)
        if entry.source == "mjlab_builtin":
            return lib.resolve_mjlab_builtin_path(library_slug)
        # gymnasium_builtin et al. — not a MuJoCo robot with an MJCF on
        # disk. Physics tab is legitimately unavailable for these; the
        # route surfaces a more specific error via MJCF_UNAVAILABLE_REASON.
        log.info(
            "_resolve_library_mjcf: slug %r has source %r; no MJCF path",
            library_slug, entry.source,
        )
        return None
    except Exception as e:  # noqa: BLE001
        log.warning(
            "_resolve_library_mjcf: lookup failed for %r: %s",
            library_slug, e,
        )
        return None


def mjcf_unavailable_reason(project_dir: Path) -> str:
    """Human-readable reason string for the 404 when `load_mjcf` returns
    None. Callers (routes/physics.py) use this as the `detail` of the
    `/problems/mjcf-unavailable` response. Distinguishes: no project
    metadata, no library_slug configured, `gymnasium_builtin` slug
    (Gym envs have no MJCF), and `mjlab_builtin` mapping missing."""
    upload_dir = project_dir / UPLOAD_ROBOT_DIR
    if upload_dir.is_dir() and any(upload_dir.glob("*.xml")):
        # load_mjcf returned None but there's an xml on disk — parse fail.
        return (
            "An uploaded MJCF exists at uploads/robot/*.xml but could "
            "not be parsed. Re-upload or edit directly."
        )
    meta = _read_metadata(project_dir)
    if meta is None:
        return (
            "Pick a library robot OR upload a URDF/MJCF first — this "
            "project has no metadata.json."
        )
    rs = meta.get("robot_source") or {}
    library_slug = rs.get("library_slug") or rs.get("library_name")
    if not library_slug:
        return (
            "Pick a library robot OR upload a URDF/MJCF first — this "
            "project's metadata.json has no `robot_source.library_slug`."
        )
    try:
        from backend.services.robot_library import get_library

        entry = get_library().get_robot(library_slug)
    except Exception:  # noqa: BLE001
        entry = None
    if entry is None:
        return (
            f"Library slug {library_slug!r} is not in the catalog. "
            "Delete the project + re-create with a valid library robot, "
            "or upload an MJCF directly."
        )
    if entry.source == "gymnasium_builtin":
        return (
            f"Robot {library_slug!r} has source `gymnasium_builtin`. "
            "Gym envs don't expose MJCFs; the Physics tab is only "
            "available for `menagerie` or `mjlab_builtin` robots."
        )
    if entry.source == "mjlab_builtin":
        return (
            f"Could not find MJCF for mjlab_builtin slug {library_slug!r} "
            "in the installed mjlab package. Either mjlab's layout "
            "changed or the slug is missing from "
            "RobotLibrary._MJLAB_BUILTIN_MJCF."
        )
    if entry.source == "menagerie":
        return (
            f"Menagerie MJCF lookup failed for {library_slug!r}. Check "
            "that `robot_descriptions` is installed and the package "
            f"{entry.menagerie_package!r} resolves."
        )
    return f"Unknown library source {entry.source!r} for slug {library_slug!r}."


def _read_metadata(project_dir: Path) -> Optional[dict[str, Any]]:
    import json

    meta_path = project_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ── timing ───────────────────────────────────────────────────────────
def _adapter_task_id(project_dir: Path) -> Optional[str]:
    """The project's mjlab task id from config.toml, or None."""
    import tomllib

    try:
        with (project_dir / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
    except Exception:  # noqa: BLE001 — no config is a normal state
        return None
    adapter_cfg = (cfg.get("adapter") or {}).get("config") or {}
    tid = adapter_cfg.get("task_id") or adapter_cfg.get("env_id")
    return str(tid) if isinstance(tid, str) and tid else None


def resolve_timing(
    project_dir: Path, mjcf_timestep: Optional[float]
) -> Optional[dict[str, Any]]:
    """The physics/control rates TRAINING runs at, for the Physics tab.

    Two distinct numbers were being conflated, invisibly:

      * `mjcf_timestep` — what the robot XML compiles to. The Unitree G1
        declares no `<option timestep=...>`, so it compiles at MuJoCo's 0.002 s
        default and the tab reported **500 Hz**.
      * the task's `MujocoCfg.timestep` — what mjlab actually sets, 0.005 s,
        i.e. **200 Hz**. A 2.5x discrepancy the UI never showed.

    `mjcf_overridden` marks that case so the panel can say which number is
    real. The control rate (physics x decimation) was not surfaced anywhere at
    all; it is the rate that has to match hardware on deployment.

    Returns None when the task can't be resolved — "unknown" is honest, a
    default would re-introduce the same guessing this exists to remove.
    """
    try:
        from sculptor.refs.timing import timing_for_task, validate_timing
    except Exception:  # noqa: BLE001 — sculptor optional in some deployments
        return None

    task_id = _adapter_task_id(project_dir)
    timing = timing_for_task(task_id) if task_id else None
    if timing is None:
        return None

    payload: dict[str, Any] = timing.to_dict()
    payload["task_id"] = task_id
    payload["mjcf_timestep"] = mjcf_timestep
    payload["mjcf_overridden"] = bool(
        mjcf_timestep is not None
        and abs(float(mjcf_timestep) - timing.physics_dt) > 1e-9
    )
    payload["findings"] = validate_timing(timing)
    return payload


# ── load + summarize ─────────────────────────────────────────────────
def load_mjcf(project_dir: Path) -> Optional[MjcfLoadResult]:
    """Read the project's active MJCF + build a summary for the UI.

    Returns None when no MJCF is available (unconfigured upload
    project, library lookup failure). Callers surface a 404 in that
    case.
    """
    path, kind = resolve_mjcf_path(project_dir)
    if path is None:
        return None

    xml_source = path.read_text(encoding="utf-8", errors="replace")
    summary = _summarize_mjcf(path)
    summary.timing = resolve_timing(project_dir, summary.timestep)
    materialized = (
        kind == "upload"
        or path.is_relative_to(project_dir)
    )
    return MjcfLoadResult(
        xml_source=xml_source,
        summary=summary,
        mjcf_source_kind=kind,
        mjcf_path=path,
        materialized=materialized,
    )


def _summarize_mjcf(xml_path: Path) -> MjcfSummary:
    """Parse `xml_path` via mujoco + collect editable fields for the
    Physics tab's read-only panel. Falls back to a minimal summary if
    mujoco can't parse (surface a warning; do NOT crash the endpoint)."""
    try:
        import mujoco
    except ImportError:
        log.warning("_summarize_mjcf: mujoco unavailable")
        return MjcfSummary(parse_error="mujoco module is not installed")

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:  # noqa: BLE001
        log.warning("_summarize_mjcf: mjcf parse failed for %s: %s", xml_path, e)
        return MjcfSummary(parse_error=f"{type(e).__name__}: {e}")

    summary = MjcfSummary()
    summary.timestep = float(model.opt.timestep)
    summary.gravity = [
        round(float(x), 6) for x in (model.opt.gravity[0], model.opt.gravity[1], model.opt.gravity[2])
    ]
    integrator_map = {
        getattr(mujoco.mjtIntegrator, k): k.replace("mjINT_", "").lower()
        for k in dir(mujoco.mjtIntegrator)
        if k.startswith("mjINT_")
    }
    solver_map = {
        getattr(mujoco.mjtSolver, k): k.replace("mjSOL_", "").lower()
        for k in dir(mujoco.mjtSolver)
        if k.startswith("mjSOL_")
    }
    summary.integrator = integrator_map.get(int(model.opt.integrator))
    summary.solver = solver_map.get(int(model.opt.solver))

    # Joints — per-joint damping / armature / frictionloss / stiffness.
    joints: list[dict[str, Any]] = []
    for j in range(model.njnt):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            or f"joint_{j}"
        )
        joints.append({
            "name": name,
            "damping": round(float(model.dof_damping[model.jnt_dofadr[j]]), 6)
                if model.jnt_dofadr[j] < model.nv else None,
            "armature": round(float(model.dof_armature[model.jnt_dofadr[j]]), 6)
                if model.jnt_dofadr[j] < model.nv else None,
            "frictionloss": round(float(model.dof_frictionloss[model.jnt_dofadr[j]]), 6)
                if model.jnt_dofadr[j] < model.nv else None,
            "stiffness": round(float(model.jnt_stiffness[j]), 6),
            "range": (
                [round(float(model.jnt_range[j][0]), 6),
                 round(float(model.jnt_range[j][1]), 6)]
                if bool(model.jnt_limited[j])
                else None
            ),
        })
    summary.joints = joints

    # Actuators — forcerange, gear, kv/kp if they appear as gainprm.
    actuators: list[dict[str, Any]] = []
    for a in range(model.nu):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            or f"actuator_{a}"
        )
        forcerange = None
        if bool(model.actuator_forcelimited[a]):
            forcerange = [
                round(float(model.actuator_forcerange[a][0]), 6),
                round(float(model.actuator_forcerange[a][1]), 6),
            ]
        actuators.append({
            "name": name,
            "forcerange": forcerange,
            "gear": round(float(model.actuator_gear[a][0]), 6),
            # gainprm / biasprm usually carry kp / kv for position /
            # velocity servos — surface them verbatim for the UI.
            "gainprm": [round(float(x), 6) for x in model.actuator_gainprm[a][:3]],
            "biasprm": [round(float(x), 6) for x in model.actuator_biasprm[a][:3]],
        })
    summary.actuators = actuators

    # Geoms — friction tuple (sliding, torsional, rolling) + mass.
    geoms: list[dict[str, Any]] = []
    for g in range(model.ngeom):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            or f"geom_{g}"
        )
        body_id = int(model.geom_bodyid[g])
        geoms.append({
            "name": name,
            "body_mass": round(float(model.body_mass[body_id]), 6),
            "friction": [
                round(float(model.geom_friction[g][0]), 6),
                round(float(model.geom_friction[g][1]), 6),
                round(float(model.geom_friction[g][2]), 6),
            ],
        })
    summary.geoms = geoms[:50]  # cap — some models have hundreds of visual geoms

    return summary


# ── prompt edit ───────────────────────────────────────────────────────
def materialize_mjcf(project_dir: Path) -> Optional[Path]:
    """Copy a library-backed MJCF into `<project>/uploads/robot/base.xml`
    so subsequent edits can be tracked in git, and bring along every
    asset it references (meshes, textures, includes). Idempotent: a
    second call preserves any user / Claude edits to the XML while
    topping up missing assets.

    Returns the path to the local (now editable) MJCF, or None if
    neither a local nor a library MJCF is available.
    """
    local_dir = project_dir / UPLOAD_ROBOT_DIR
    lib_src = _resolve_library_mjcf(project_dir)

    # If a local XML already exists, respect any edits to it but top
    # up missing assets from the library source when registered.
    if local_dir.is_dir():
        xmls = sorted(local_dir.glob("*.xml"))
        if xmls:
            local = xmls[0]
            if lib_src is not None:
                warnings = _copy_referenced_assets(
                    lib_src, local_dir, skip_existing=True
                )
                for w in warnings:
                    log.warning("materialize_mjcf top-up: %s", w)
            return local

    # First materialization — requires a library source.
    if lib_src is None:
        return None
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / "base.xml"
    shutil.copy2(lib_src, local)
    warnings = _copy_referenced_assets(lib_src, local_dir)
    for w in warnings:
        log.warning("materialize_mjcf: %s", w)
    return local


def rematerialize_mjcf(project_dir: Path) -> Optional[Path]:
    """Wipe `<project>/uploads/robot/` and re-materialize from the
    library source. Used to heal broken states where the initial
    materialize dropped the assets/ dir. Returns None if no library
    source is registered (nothing to fall back to)."""
    if _resolve_library_mjcf(project_dir) is None:
        return None
    local_dir = project_dir / UPLOAD_ROBOT_DIR
    if local_dir.exists():
        shutil.rmtree(local_dir)
    return materialize_mjcf(project_dir)


# Filenames that mujoco typically stores under the XML's parent dir.
# Copied unconditionally when they exist — covers both the default
# `assets/` convention and the less-common `meshes/` + `textures/`
# split used by a minority of menagerie packages.
_DEFAULT_ASSET_DIRS: tuple[str, ...] = ("assets", "meshes", "textures")


def _copy_referenced_assets(
    src_xml: Path,
    dst_dir: Path,
    *,
    skip_existing: bool = False,
) -> list[str]:
    """Copy every mesh / texture / hfield / include file that
    `src_xml` references into `dst_dir`, preserving the relative
    directory structure so mujoco can parse the copy standalone.

    Strategy:
      1. Copy the `meshdir`, `texturedir`, and conventional `assets/`,
         `meshes/`, `textures/` subdirs under `src_xml.parent` when
         they exist (belt-and-braces against XMLs that reference
         files we didn't spot via element-scan).
      2. Parse the XML + chase `<mesh file=>`, `<hfield file=>`,
         `<texture file=>`, and `<include file=>` attrs. Resolve
         against the top-level compiler's meshdir / texturedir.
      3. Recurse into `<include file=>` for transitive assets.

    Returns a list of human-readable warnings for references we
    skipped (absolute paths, `..` traversal, missing files). Never
    raises — callers always get back a (possibly empty) warnings
    list so first-call failure doesn't block subsequent top-ups.
    """
    if skip_existing:
        def _copy_file(src: Path, dst: Path) -> None:
            if dst.exists():
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    else:
        def _copy_file(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def _tree_copy_fn(s: str, d: str) -> str:
        if skip_existing and Path(d).exists():
            return d
        shutil.copy2(s, d)
        return d

    warnings: list[str] = []
    src_parent = src_xml.parent

    def _rel_safe(p: str | None) -> Optional[PurePosixPath]:
        """Return a safe POSIX-style relative path, or None + a
        warning for absolute / traversal attempts."""
        if not p:
            return None
        pp = PurePosixPath(p.replace("\\", "/"))
        if pp.is_absolute():
            warnings.append(f"skipped absolute path: {p!r}")
            return None
        if any(part == ".." for part in pp.parts):
            warnings.append(f"skipped path with parent traversal: {p!r}")
            return None
        return pp

    def _copy_dir_if_exists(rel_dir: str) -> None:
        safe = _rel_safe(rel_dir)
        if safe is None or str(safe) in ("", "."):
            return
        src = src_parent / Path(str(safe))
        if src.is_dir():
            dst = dst_dir / Path(str(safe))
            shutil.copytree(
                src, dst, dirs_exist_ok=True, copy_function=_tree_copy_fn,
            )

    def _copy_referenced_file(rel: PurePosixPath) -> None:
        src = src_parent / Path(str(rel))
        if not src.is_file():
            warnings.append(f"referenced file not found: {str(rel)!r}")
            return
        dst = dst_dir / Path(str(rel))
        _copy_file(src, dst)

    def _join_under(dir_prefix: str, file_ref: str) -> Optional[PurePosixPath]:
        safe_file = _rel_safe(file_ref)
        if safe_file is None:
            return None
        if not dir_prefix:
            return safe_file
        safe_dir = _rel_safe(dir_prefix)
        if safe_dir is None:
            return None
        return safe_dir / safe_file

    try:
        top_tree = ET.parse(src_xml)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"could not parse source XML: {type(e).__name__}: {e}")
        return warnings
    top_root = top_tree.getroot()

    compiler = top_root.find("compiler")
    meshdir = (compiler.get("meshdir") if compiler is not None else "") or ""
    texturedir = (compiler.get("texturedir") if compiler is not None else "") or ""

    # Step 1 — copy conventional + declared asset dirs wholesale.
    dirs_to_copy = {meshdir, texturedir, *_DEFAULT_ASSET_DIRS}
    for d in dirs_to_copy:
        if d:
            _copy_dir_if_exists(d)

    # Step 2+3 — element scan + include recursion.
    visited: set[Path] = set()
    MAX_INCLUDE_DEPTH = 8

    def _process(xml_path: Path, depth: int = 0) -> None:
        if depth > MAX_INCLUDE_DEPTH or xml_path in visited:
            return
        visited.add(xml_path)
        try:
            tree = ET.parse(xml_path)
        except Exception as e:  # noqa: BLE001
            warnings.append(
                f"could not parse {xml_path.name}: {type(e).__name__}: {e}"
            )
            return
        root = tree.getroot()

        for tag, dir_prefix in (
            ("mesh", meshdir),
            ("hfield", meshdir),
            ("texture", texturedir),
        ):
            for el in root.iter(tag):
                f = el.get("file")
                if not f:
                    continue
                rel = _join_under(dir_prefix, f)
                if rel is not None:
                    _copy_referenced_file(rel)

        # Includes resolve against THIS xml's parent (not src_parent).
        try:
            xml_parent_rel = xml_path.parent.relative_to(src_parent)
        except ValueError:
            xml_parent_rel = Path(".")
        for el in root.iter("include"):
            f = el.get("file")
            if not f:
                continue
            safe = _rel_safe(f)
            if safe is None:
                continue
            inc_rel = xml_parent_rel / Path(str(safe))
            inc_src = src_parent / inc_rel
            if not inc_src.is_file():
                warnings.append(f"include not found: {f!r}")
                continue
            inc_dst = dst_dir / inc_rel
            _copy_file(inc_src, inc_dst)
            _process(inc_src, depth + 1)

    _process(src_xml)
    return warnings


# §Ship-8b: parser lives in `sculptor.adapters.mjcf_editor` now.
# Re-export the private alias so any external caller of
# `_parse_claude_output` (none in-tree as of 2026-04-23 but the symbol
# was public-looking) keeps working.
from sculptor.adapters.mjcf_editor import parse_claude_output as _parse_claude_output  # noqa: F401


def apply_prompt_edit(
    project_dir: Path,
    prompt: str,
    *,
    client=None,
    kg_store=None,
) -> dict[str, Any]:
    """Claude rewrites the full MJCF per `prompt`. Validates + commits.

    §Ship-8b refactor: the Claude-orchestration body moved into
    `sculptor.adapters.mjcf_editor.apply_mjcf_edit` so sculpt.py's
    auto-apply path (§7.4) can call the same code without importing
    backend. This function now handles backend-only concerns:
      * materialize library → project-local MJCF
      * git-commit after a successful write
      * map the editor's `committed=True` result to `commit_sha`

    The return shape is unchanged so existing callers (routes,
    reward_jobs) keep working.
    """
    from sculptor.adapters.mjcf_editor import apply_mjcf_edit

    # Load + materialize. Returns None if the project has no MJCF
    # source (unconfigured robot).
    local_path = materialize_mjcf(project_dir)
    if local_path is None:
        raise RuntimeError(
            "no MJCF available for this project — pick a library robot "
            "or upload a URDF/MJCF first."
        )
    loaded = load_mjcf(project_dir)
    summary_digest = loaded.summary.to_dict() if loaded else {}

    adapter_hint = _adapter_hint(project_dir)

    # Delegate the Claude call + parse + validate + write.
    result = apply_mjcf_edit(
        local_path, prompt,
        adapter_hint=adapter_hint,
        summary_digest=summary_digest,
        client=client,
        kg_store=kg_store,
        write=True,
    )

    # Backend-only post-processing: on success, git-commit the file.
    if result.get("committed"):
        commit_sha = _git_commit(
            project_dir, f"physics: {result.get('summary', 'edit')}"
        )
        result["commit_sha"] = commit_sha
    return result


def _adapter_hint(project_dir: Path) -> str:
    """Build a short line like 'mjlab (task=Mjlab-Velocity-Flat-
    Unitree-G1)' for the Claude prompt."""
    import tomllib

    try:
        with (project_dir / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
    except Exception:  # noqa: BLE001
        return "unknown"
    adapter = (cfg.get("adapter") or {}).get("class") or "unknown"
    adapter_cfg = (cfg.get("adapter") or {}).get("config") or {}
    task_id = adapter_cfg.get("task_id") or adapter_cfg.get("env_id")
    if task_id:
        return f"{adapter} (task={task_id})"
    return str(adapter)


def _git_commit(project_dir: Path, message: str) -> Optional[str]:
    """Commit any staged changes in `project_dir`. Best-effort — if
    git is missing or the dir isn't a repo, returns None."""
    try:
        subprocess.run(
            ["git", "-C", str(project_dir), "add", str(UPLOAD_ROBOT_DIR)],
            capture_output=True, check=False,
        )
        res = subprocess.run(
            ["git", "-C", str(project_dir), "commit", "-m", message],
            capture_output=True, check=False,
        )
        if res.returncode != 0:
            return None
        rev = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return rev.stdout.strip() or None
    except FileNotFoundError:
        return None

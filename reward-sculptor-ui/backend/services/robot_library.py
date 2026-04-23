"""Robot-library loader — the authoritative catalog of robots exposed to
the UI and the project-creation flow.

Data file: backend/data/robot_library.yml (committed; schema documented
inline at the top of the file).

Loader lifecycle:
  1. `get_library()` is a FastAPI dependency. First call parses the YAML,
     validates every entry, and caches a `RobotLibrary` singleton.
  2. On first request that filters by `training_support` or resolves a
     `mjlab_ready` entry's task list, the loader subprocesses
     `mjlab.tasks.registry.list_tasks()` to get the ground truth and
     *demotes* any `mjlab_ready` entry whose `preconfigured_tasks`
     reference task_ids not actually registered in the running mjlab.
     Demotions are in-memory only — the YAML file on disk is untouched.
  3. If `find_spec("mjlab")` is False, all `mjlab_ready` entries are
     demoted at load time with a "mjlab not installed" note.

Why lazy instead of eager-at-startup: the mjlab import triggers
mujoco_warp + CUDA kernel compilation, which takes 15-25 s on a cold
sculptor install. Doing that work at FastAPI startup would blow the 3 s
cold-start budget (MJLAB_PIVOT_DESIGN §7). First call to `list_robots()`
with a mjlab_ready filter pays the cost once; subsequent calls hit the
cached registry.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import yaml


log = logging.getLogger("reward-sculptor-ui.robot_library")


# ── schema ──────────────────────────────────────────────────────────────────
Category = Literal[
    "Quadruped",
    "Humanoid",
    "Arm",
    "Gripper_Hand",
    "Mobile_Manipulator",
    "Drone",
    "Biomechanical",
    "Other",
]

CATEGORIES: list[str] = list(
    Category.__args__  # type: ignore[attr-defined]
)

TrainingSupport = Literal[
    "mjlab_ready", "preview_only", "gymnasium_compatible"
]

Source = Literal["menagerie", "mjlab_builtin", "gymnasium_builtin"]


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,62}[a-z0-9]$")
_URL_RE = re.compile(r"^https?://[^\s]+$")


@dataclass
class PreconfiguredTask:
    task_id: str
    display_name: str
    recommended_num_envs: int


@dataclass
class Reference:
    kind: Literal["paper", "repo"]
    url: str
    citation: str


@dataclass
class RobotEntry:
    slug: str
    display_name: str
    category: str
    description: str
    source: str
    menagerie_package: Optional[str]
    training_support: str
    is_smoke_test_target: bool
    preconfigured_tasks: list[PreconfiguredTask]
    references: list[Reference]
    thumbnail_path: str
    # Populated by the D-guard if demoted at runtime.
    demote_note: Optional[str] = None


@dataclass
class RobotLibrary:
    """In-memory catalog. The loader builds one instance on first access
    and mutates it when the D-guard runs (see `validate_against_mjlab`).
    """

    entries_by_slug: dict[str, RobotEntry] = field(default_factory=dict)
    _d_guard_applied: bool = False
    _d_guard_lock: Any = field(default_factory=threading.Lock)
    _path_cache: dict[str, Path] = field(default_factory=dict)

    # ── public API ──────────────────────────────────────────────────────────
    def list_robots(
        self,
        category: Optional[str] = None,
        training_support: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list[RobotEntry]:
        """Ensure D-guard has run, then filter."""
        self._apply_d_guard()
        out: list[RobotEntry] = list(self.entries_by_slug.values())
        if category:
            out = [r for r in out if r.category == category]
        if training_support:
            out = [r for r in out if r.training_support == training_support]
        if search:
            needle = search.lower()
            out = [
                r for r in out
                if needle in r.slug.lower()
                or needle in r.display_name.lower()
                or needle in (r.description or "").lower()
            ]
        out.sort(key=lambda r: (r.category, r.display_name))
        return out

    def get_robot(self, slug: str) -> Optional[RobotEntry]:
        self._apply_d_guard()
        return self.entries_by_slug.get(slug)

    # Slug → (importable_package, filename) for `source: mjlab_builtin`
    # robots. mjlab ships these MJCFs inside its own site-packages tree
    # (e.g. `mjlab.tasks.cartpole/cartpole.xml`), NOT via menagerie /
    # robot_descriptions. Extend this dict when a new mjlab-only robot
    # is added to robot_library.yml with `source: mjlab_builtin`.
    # ClassVar keeps dataclass from treating this as an instance field.
    _MJLAB_BUILTIN_MJCF: ClassVar[dict[str, tuple[str, str]]] = {
        "cartpole_mjlab": ("mjlab.tasks.cartpole", "cartpole.xml"),
    }

    def resolve_mjlab_builtin_path(self, slug: str) -> Optional[Path]:
        """Return the MJCF path for an `mjlab_builtin`-sourced robot,
        read directly from the installed mjlab package via
        `importlib.resources`. None for slugs that aren't mjlab_builtin
        or aren't in the known mapping."""
        cached = self._path_cache.get(slug)
        if cached is not None:
            return cached
        entry = self.entries_by_slug.get(slug)
        if entry is None or entry.source != "mjlab_builtin":
            return None
        mapping = self._MJLAB_BUILTIN_MJCF.get(slug)
        if mapping is None:
            log.warning(
                "resolve_mjlab_builtin_path: no MJCF mapping for %r; "
                "add to RobotLibrary._MJLAB_BUILTIN_MJCF", slug,
            )
            return None
        pkg, filename = mapping
        try:
            from importlib import resources

            p = resources.files(pkg) / filename  # type: ignore[arg-type]
            resolved = Path(str(p))
            if not resolved.is_file():
                log.warning(
                    "resolve_mjlab_builtin_path: %s/%s does not exist "
                    "for robot %r", pkg, filename, slug,
                )
                return None
            self._path_cache[slug] = resolved
            return resolved
        except Exception as e:  # noqa: BLE001
            log.warning(
                "resolve_mjlab_builtin_path: exception for %r: %s: %s",
                slug, type(e).__name__, e,
            )
            return None

    def resolve_menagerie_path(self, slug: str) -> Optional[Path]:
        """Materialize a Menagerie robot's MJCF via robot_descriptions.
        Cached per-process. Returns None if the robot isn't Menagerie-
        sourced or the loader fails (e.g. missing mesh package)."""
        if slug in self._path_cache:
            return self._path_cache[slug]
        entry = self.entries_by_slug.get(slug)
        if entry is None or entry.source != "menagerie" or not entry.menagerie_package:
            return None
        try:
            import importlib
            mod = importlib.import_module(
                f"robot_descriptions.{entry.menagerie_package}"
            )
            mjcf_path = getattr(mod, "MJCF_PATH", None)
            if mjcf_path is None:
                return None
            resolved = Path(mjcf_path)
            if not resolved.is_file():
                log.warning(
                    "resolve_menagerie_path: MJCF_PATH %s does not exist "
                    "for robot %r", resolved, slug,
                )
                return None
            self._path_cache[slug] = resolved
            return resolved
        except Exception as e:  # noqa: BLE001
            log.warning(
                "resolve_menagerie_path: exception for %r: %s: %s",
                slug, type(e).__name__, e,
            )
            return None

    # ── D-guard: cross-reference with mjlab registry ────────────────────────
    def _apply_d_guard(self) -> None:
        """Demote any mjlab_ready entry whose preconfigured_tasks include
        a task_id not registered in the running mjlab. Runs exactly once
        per process (thread-safe)."""
        if self._d_guard_applied:
            return
        with self._d_guard_lock:
            if self._d_guard_applied:
                return
            registered = _fetch_mjlab_tasks()
            demoted = 0
            for entry in self.entries_by_slug.values():
                if entry.training_support != "mjlab_ready":
                    continue
                if registered is None:
                    entry.training_support = "preview_only"
                    entry.demote_note = (
                        "mjlab task registry unavailable; treat as preview-only"
                    )
                    entry.preconfigured_tasks = []
                    demoted += 1
                    continue
                # Keep tasks whose IDs are registered.
                valid = [
                    t for t in entry.preconfigured_tasks
                    if t.task_id in registered
                ]
                if not valid:
                    entry.training_support = "preview_only"
                    entry.demote_note = (
                        "No preconfigured_tasks for this robot are registered "
                        "in the running mjlab install; demoted to preview_only."
                    )
                    entry.preconfigured_tasks = []
                    demoted += 1
                elif len(valid) != len(entry.preconfigured_tasks):
                    dropped = [
                        t.task_id for t in entry.preconfigured_tasks
                        if t.task_id not in registered
                    ]
                    entry.demote_note = (
                        f"Dropped unregistered tasks: {dropped}"
                    )
                    entry.preconfigured_tasks = valid
            if demoted:
                log.warning(
                    "robot_library D-guard: %d mjlab_ready robot(s) demoted "
                    "to preview_only (mjlab registry mismatch)", demoted,
                )
            self._d_guard_applied = True


def _fetch_mjlab_tasks() -> Optional[set[str]]:
    """Subprocess-query `mjlab.tasks.registry.list_tasks()`. Returns None
    on import failure / timeout / mjlab missing. Used by the D-guard to
    detect tasks that are declared in the YAML but not present in the
    running mjlab install.

    Subprocess-isolated to preserve the sculptor_bridge lazy-import rule
    (MJLAB_PIVOT_DESIGN §7) — importing mjlab in-process here would be
    a 15-25 s startup hit on a cold install.
    """
    if importlib.util.find_spec("mjlab") is None:
        return None
    script = (
        "from mjlab.tasks.registry import list_tasks; "
        "import json; print(json.dumps(sorted(list_tasks())))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60.0,
        )
    except subprocess.TimeoutExpired:
        log.warning("D-guard: mjlab task fetch timed out after 60 s")
        return None
    if result.returncode != 0:
        log.warning(
            "D-guard: mjlab task fetch exited %d: %s",
            result.returncode, (result.stderr or "")[-400:],
        )
        return None
    try:
        tasks = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        log.warning("D-guard: could not parse mjlab task list: %s", e)
        return None
    return set(tasks) if isinstance(tasks, list) else None


# ── validation (eager, at load time) ────────────────────────────────────────
class LibraryValidationError(RuntimeError):
    """Malformed robot_library.yml detected at load time."""


def _validate_entry_dict(raw: dict, idx: int) -> RobotEntry:
    """Parse + validate a single entry from the YAML dict form."""
    def _err(msg: str) -> LibraryValidationError:
        slug = raw.get("slug", "<unknown>")
        return LibraryValidationError(f"robot_library.yml entry[{idx}] ({slug!r}): {msg}")

    slug = raw.get("slug", "")
    if not isinstance(slug, str) or not _SLUG_RE.match(slug):
        raise _err(f"invalid slug {slug!r} (must match {_SLUG_RE.pattern})")

    category = raw.get("category", "")
    if category not in CATEGORIES:
        raise _err(
            f"category={category!r} is not in {CATEGORIES}"
        )

    training_support = raw.get("training_support", "")
    if training_support not in (
        "mjlab_ready", "preview_only", "gymnasium_compatible",
    ):
        raise _err(f"training_support={training_support!r} is invalid")

    source = raw.get("source", "")
    if source not in ("menagerie", "mjlab_builtin", "gymnasium_builtin"):
        raise _err(f"source={source!r} is invalid")

    display_name = raw.get("display_name", "")
    if not isinstance(display_name, str) or not display_name.strip():
        raise _err("display_name is required")

    description = raw.get("description", "") or ""
    menagerie_package = raw.get("menagerie_package")
    if menagerie_package is not None and not isinstance(menagerie_package, str):
        raise _err(f"menagerie_package must be a string or null, got {type(menagerie_package).__name__}")

    tasks: list[PreconfiguredTask] = []
    for t_idx, t_raw in enumerate(raw.get("preconfigured_tasks") or []):
        if not isinstance(t_raw, dict):
            raise _err(f"preconfigured_tasks[{t_idx}] must be a dict")
        tid = t_raw.get("task_id")
        if not isinstance(tid, str) or not tid:
            raise _err(f"preconfigured_tasks[{t_idx}].task_id is required")
        tasks.append(PreconfiguredTask(
            task_id=tid,
            display_name=str(t_raw.get("display_name") or tid),
            recommended_num_envs=int(t_raw.get("recommended_num_envs") or 1024),
        ))

    refs: list[Reference] = []
    for r_idx, r_raw in enumerate(raw.get("references") or []):
        if not isinstance(r_raw, dict):
            raise _err(f"references[{r_idx}] must be a dict")
        kind = r_raw.get("kind")
        if kind not in ("paper", "repo"):
            raise _err(f"references[{r_idx}].kind must be 'paper' or 'repo'")
        url = r_raw.get("url", "")
        if not isinstance(url, str) or not _URL_RE.match(url):
            raise _err(
                f"references[{r_idx}].url must be an http(s) URL; got {url!r}"
            )
        refs.append(Reference(
            kind=kind,  # type: ignore[arg-type]
            url=url,
            citation=str(r_raw.get("citation") or ""),
        ))

    return RobotEntry(
        slug=slug,
        display_name=display_name,
        category=category,
        description=description,
        source=source,
        menagerie_package=menagerie_package,
        training_support=training_support,
        is_smoke_test_target=bool(raw.get("is_smoke_test_target", False)),
        preconfigured_tasks=tasks,
        references=refs,
        thumbnail_path=str(raw.get("thumbnail_path") or f"robots/{slug}.webp"),
    )


def load_library(path: Path) -> RobotLibrary:
    """Load + validate robot_library.yml. Raises LibraryValidationError
    on malformed YAML. D-guard is lazy; call `list_robots` or
    `get_robot` to trigger it."""
    if not path.is_file():
        raise LibraryValidationError(f"robot_library.yml not found at {path}")
    try:
        raw_doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise LibraryValidationError(f"robot_library.yml is not valid YAML: {e}") from e

    if not isinstance(raw_doc, dict):
        raise LibraryValidationError("robot_library.yml must contain a top-level mapping")
    entries_raw = raw_doc.get("robots") or []
    if not isinstance(entries_raw, list):
        raise LibraryValidationError("robot_library.yml: 'robots' must be a list")

    seen: set[str] = set()
    library = RobotLibrary()
    for idx, raw in enumerate(entries_raw):
        if not isinstance(raw, dict):
            raise LibraryValidationError(
                f"robot_library.yml entry[{idx}] must be a dict"
            )
        entry = _validate_entry_dict(raw, idx)
        if entry.slug in seen:
            raise LibraryValidationError(
                f"robot_library.yml: duplicate slug {entry.slug!r}"
            )
        seen.add(entry.slug)
        library.entries_by_slug[entry.slug] = entry
    log.info(
        "robot_library: loaded %d entries from %s",
        len(library.entries_by_slug), path,
    )
    return library


# ── FastAPI dependency singleton ────────────────────────────────────────────
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "robot_library.yml"
_singleton: Optional[RobotLibrary] = None
_singleton_lock = threading.Lock()


def get_library(path: Path = _DEFAULT_PATH) -> RobotLibrary:
    """Return the process-wide library singleton. Thread-safe lazy init."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = load_library(path)
    return _singleton


def reset_library_singleton() -> None:
    """Test utility — forces the next `get_library()` to re-read YAML."""
    global _singleton
    with _singleton_lock:
        _singleton = None

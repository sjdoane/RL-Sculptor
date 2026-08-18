"""Filesystem-backed project registry.

No database. `$RS_PROJECTS_ROOT` is the source of truth — each
subdirectory with a `metadata.json` file is a project. The sidecar
holds UI-only metadata (display_name, description, creation time,
robot source, last-run bookkeeping). Everything else (adapter class,
env_id, iteration settings) is read from `config.toml` on demand.

Per-project locking is via the `filelock` package, using
`<project>/.sculptor-ui.lock`. Lock is acquired only for write
operations (create/delete/…); reads never block. Lock acquisition
timeout is 5 s, after which `BusyError` is raised.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from filelock import FileLock, Timeout

from backend.models.project import (
    SLUG_PATTERN,
    ProjectDetail,
    ProjectStatus,
    ProjectSummary,
)
from backend.services import sculptor_bridge


# Same shape the API layer enforces via SlugStr; checked here too so
# store methods are safe against raw path-shaped input.
_SLUG_RE = re.compile(SLUG_PATTERN)

METADATA_FILE = "metadata.json"
LOCK_FILE = ".sculptor-ui.lock"
LOCK_TIMEOUT_S = 5
SCHEMA_VERSION = 1

# Extra dirs the UI creates on top of what `sculpt init` produces.
_UI_SUBDIRS = ("kg", "uploads")


class BusyError(RuntimeError):
    """Raised when a per-project lock can't be acquired within the timeout."""


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── slug helpers ─────────────────────────────────────────────────────
    @staticmethod
    def slugify(name: str) -> str:
        """Lowercase alnum + hyphens, stripped, capped at 62 chars."""
        base = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if not base:
            base = "project"
        return base[:62]

    def _project_dir(self, slug: str) -> Path:
        return self.root / slug

    def _metadata_path(self, slug: str) -> Path:
        return self._project_dir(slug) / METADATA_FILE

    def _lock_path(self, slug: str) -> Path:
        return self._project_dir(slug) / LOCK_FILE

    def _ensure_unique_slug(self, base: str) -> str:
        slug = base
        i = 2
        while self._project_dir(slug).exists():
            suffix = f"-{i}"
            slug = (base[: 62 - len(suffix)] + suffix) if len(base) + len(suffix) > 62 else base + suffix
            i += 1
            if i > 99:
                raise ValueError(f"could not generate a unique slug from {base!r}")
        return slug

    # ── locking ──────────────────────────────────────────────────────────
    def acquire_lock(self, slug: str, timeout: float = LOCK_TIMEOUT_S) -> FileLock:
        """Acquire the per-project write lock. Caller releases via
        `lock.release()` or by exiting a `with lock:` block."""
        lock_path = self._lock_path(slug)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path), timeout=timeout)
        try:
            lock.acquire()
        except Timeout as e:
            raise BusyError(f"project {slug} is busy") from e
        return lock

    # ── metadata IO ──────────────────────────────────────────────────────
    def _write_metadata(self, slug: str, data: dict) -> None:
        path = self._metadata_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _read_metadata(self, slug: str) -> Optional[dict]:
        path = self._metadata_path(slug)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    # ── discovery ────────────────────────────────────────────────────────
    def list_slugs(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / METADATA_FILE).is_file():
                yield d.name

    # ── create ───────────────────────────────────────────────────────────
    def create(
        self,
        *,
        display_name: str,
        adapter: str,
        description: str = "",
        iteration_budget: int = 20,
        behavior_goal: str = "",
    ) -> ProjectDetail:
        """Create a new project. Scaffolds via sculptor_bridge, writes the
        UI sidecar, creates ui-only subdirs, returns the detail view."""
        slug = self._ensure_unique_slug(self.slugify(display_name))
        project_dir = self._project_dir(slug)
        if project_dir.exists():
            # Defensive — _ensure_unique_slug should have prevented this.
            raise FileExistsError(f"project dir {project_dir} already exists")

        # Scaffold via sculptor (creates the dir + contents + commits).
        sculptor_bridge.scaffold_project(project_dir, adapter=adapter)

        # UI-only subdirs that sculpt_init doesn't create.
        for sub in _UI_SUBDIRS:
            (project_dir / sub).mkdir(exist_ok=True)

        # Acquire lock AFTER the dir exists (filelock needs the parent).
        lock = self.acquire_lock(slug)
        try:
            created_at = datetime.now(tz=timezone.utc)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "slug": slug,
                "display_name": display_name,
                "description": description,
                "created_at": created_at.isoformat(),
                "last_run_id": None,
                "last_run_status": None,
                "last_run_at": None,
                "robot_source": {
                    "kind": "library",
                    "library_name": None,
                    "upload_manifest": None,
                },
                "initial_reward_notes": "",
                "iteration_budget": int(iteration_budget),
                "behavior_goal": behavior_goal,
                "ui_state": {},
            }
            self._write_metadata(slug, metadata)
        finally:
            lock.release()

        detail = self.get(slug)
        if detail is None:
            raise RuntimeError(
                f"project scaffold succeeded but detail read failed for {slug}"
            )
        return detail

    # ── read ─────────────────────────────────────────────────────────────
    def get(self, slug: str) -> Optional[ProjectDetail]:
        # Reject before touching the filesystem: a non-conforming slug
        # ("..", "a/b") can never name a project, and building paths from
        # it may land outside root — e.g. slug ".." hitting a parseable
        # metadata.json at the parent used to raise a pydantic
        # ValidationError (HTTP 500) instead of 404.
        if not _SLUG_RE.fullmatch(slug):
            return None
        metadata = self._read_metadata(slug)
        if metadata is None:
            return None
        project_dir = self._project_dir(slug)

        adapter_class, adapter_config = _parse_adapter(project_dir / "config.toml")
        env_id = adapter_config.get("env_id")

        runs_dir = project_dir / "runs"
        n_iters = 0
        if runs_dir.is_dir():
            n_iters = sum(
                1
                for d in runs_dir.iterdir()
                if d.is_dir() and (d / "diagnosis.json").is_file()
            )

        status = _compute_status(project_dir, metadata, n_iters)

        library_slug = None
        reference_robot = None
        ready_to_train = True
        adapter_unavailable = False
        robot_source = metadata.get("robot_source") or {}
        if isinstance(robot_source, dict):
            ls = robot_source.get("library_slug") or robot_source.get("library_name")
            if isinstance(ls, str):
                library_slug = ls
            ts = robot_source.get("training_support")
            if ts == "preview_only":
                ready_to_train = False
            if robot_source.get("adapter_unavailable"):
                adapter_unavailable = True
                ready_to_train = False

        try:
            from backend.services.project_robot import (
                resolve_project_reference_robot,
            )

            reference_robot = resolve_project_reference_robot(project_dir)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Detail reads remain available for incomplete/legacy projects;
            # launch and attach boundaries use the same resolver and fail
            # closed when this field is unresolved or contradictory.
            reference_robot = None

        migration_warning = _compute_migration_warning(adapter_class)

        return ProjectDetail(
            slug=slug,
            display_name=metadata.get("display_name", slug),
            description=metadata.get("description", ""),
            status=status,
            created_at=_parse_iso(metadata.get("created_at")),
            env_id=env_id,
            n_iterations_completed=n_iters,
            project_dir=str(project_dir),
            adapter_class=adapter_class,
            adapter_config=adapter_config,
            ready_to_train=ready_to_train,
            library_slug=library_slug,
            reference_robot=reference_robot,
            adapter_unavailable=adapter_unavailable,
            migration_warning=migration_warning,
        )

    def list(self) -> list[ProjectSummary]:
        out: list[ProjectSummary] = []
        for slug in self.list_slugs():
            detail = self.get(slug)
            if detail is None:
                continue
            cfg = detail.adapter_config or {}
            num_envs = cfg.get("num_envs")
            device = cfg.get("device")
            out.append(
                ProjectSummary(
                    slug=detail.slug,
                    display_name=detail.display_name,
                    description=detail.description,
                    status=detail.status,
                    created_at=detail.created_at,
                    env_id=detail.env_id,
                    n_iterations_completed=detail.n_iterations_completed,
                    migration_warning=detail.migration_warning,
                    # §Ship 22b: free enrichment — `detail` already carries
                    # the parsed config; surface the card-facing bits.
                    adapter_class=detail.adapter_class,
                    library_slug=detail.library_slug,
                    reference_robot=detail.reference_robot,
                    num_envs=num_envs if isinstance(num_envs, int) else None,
                    device=device if isinstance(device, str) else None,
                )
            )
        out.sort(key=lambda p: p.created_at, reverse=True)
        return out

    # ── robot source + config.toml ───────────────────────────────────────
    def read_robot_source(self, slug: str) -> dict:
        meta = self._read_metadata(slug) or {}
        rs = meta.get("robot_source") or {}
        if not isinstance(rs, dict):
            return {"kind": "none"}
        return rs

    def write_robot_source(self, slug: str, robot_source: dict) -> None:
        """Replace the sidecar's `robot_source` block. Caller must hold
        the per-project lock."""
        meta = self._read_metadata(slug)
        if meta is None:
            raise FileNotFoundError(f"no metadata for {slug!r}")
        meta["robot_source"] = robot_source
        self._write_metadata(slug, meta)

    #: What the user is currently building, carried across the steps that
    #: build it. Composing a motion, authoring per-mode rewards for it and
    #: launching a run that trains it were three disconnected screens: the
    #: composed clip id existed only in whichever dialog had just produced
    #: it, so the same id had to be re-found by hand in the ~6000-clip
    #: library twice. This is deliberately a draft, not configuration — it
    #: records intent, and every step still reads the authoritative artifact
    #: (the selection file, the reward chain, the run params) for truth.
    _DRAFT_FIELDS = ("behavior_goal", "reference_clip_id", "reference_robot",
                     "mode_reward_filename")

    def read_behavior_draft(self, slug: str) -> dict:
        meta = self._read_metadata(slug) or {}
        draft = meta.get("behavior_draft")
        if not isinstance(draft, dict):
            return {}
        return {k: draft[k] for k in self._DRAFT_FIELDS if k in draft}

    def write_behavior_draft(self, slug: str, patch: dict) -> dict:
        """Merge `patch` into the draft. A field set to None is cleared.

        Merge rather than replace: the steps that write here run at
        different times and each knows only its own field.
        """
        meta = self._read_metadata(slug)
        if meta is None:
            raise FileNotFoundError(f"no metadata for {slug!r}")
        draft = meta.get("behavior_draft")
        draft = dict(draft) if isinstance(draft, dict) else {}
        for key in self._DRAFT_FIELDS:
            if key not in patch:
                continue
            value = patch[key]
            if value is None or (isinstance(value, str) and not value.strip()):
                draft.pop(key, None)
            else:
                draft[key] = value.strip() if isinstance(value, str) else value
        meta["behavior_draft"] = draft
        self._write_metadata(slug, meta)
        return draft

    def read_env_id(self, slug: str) -> Optional[str]:
        _, cfg = _parse_adapter(self._project_dir(slug) / "config.toml")
        env_id = cfg.get("env_id")
        return env_id if isinstance(env_id, str) else None

    def set_adapter_section(
        self,
        slug: str,
        adapter_class: str,
        config: dict[str, Any],
    ) -> None:
        """Rewrite config.toml's [adapter] section. Overwrites both
        `class` and the inline `config = {...}` table.

        Used by the mjlab-project flow: sculpt_init scaffolds with the
        default gym_sb3 adapter, then this rewrites the [adapter]
        section to point at MjlabAdapter + task_id. Caller must hold
        the per-project lock.
        """
        path = self._project_dir(slug) / "config.toml"
        if not path.is_file():
            raise FileNotFoundError(f"no config.toml in project {slug!r}")
        text = path.read_text(encoding="utf-8")

        # Serialise the inline config table. Only primitive types and
        # lists of primitives are emitted — matches sculpt_init's
        # template style.
        def _toml_val(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str):
                return '"' + v.replace('\\', '\\\\').replace('"', '\\"') + '"'
            if isinstance(v, list):
                return "[" + ", ".join(_toml_val(x) for x in v) + "]"
            if v is None:
                return '""'  # TOML has no null; empty-string is the
                             # project's convention
            return _toml_val(str(v))

        config_inline = (
            "{ " + ", ".join(f"{k} = {_toml_val(v)}" for k, v in config.items()) + " }"
            if config else "{}"
        )
        new_block = (
            f"[adapter]\n"
            f'class = "{adapter_class}"\n'
            f"config = {config_inline}\n"
        )

        # Replace the entire [adapter] section. The existing section
        # runs from `[adapter]` up to the next blank line OR next section
        # header.
        pattern = re.compile(
            r'^\[adapter\].*?(?=^\[[^\]]+\]|\Z)',
            re.MULTILINE | re.DOTALL,
        )
        new_text, n = pattern.subn(new_block + "\n", text, count=1)
        if n == 0:
            # No [adapter] block yet — append.
            new_text = text.rstrip() + "\n\n" + new_block
        path.write_text(new_text, encoding="utf-8")

    def set_env_id(self, slug: str, env_id: str) -> None:
        """Rewrite config.toml so `[adapter].config.env_id` equals `env_id`.

        Uses a targeted regex against the inline-table form written by
        `sculpt init`. If the template changes to a multi-line table
        this needs replacing with a real TOML writer."""
        path = self._project_dir(slug) / "config.toml"
        if not path.is_file():
            raise FileNotFoundError(f"no config.toml in project {slug!r}")
        text = path.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r'env_id\s*=\s*"[^"]*"',
            f'env_id = "{env_id}"',
            text, count=1,
        )
        if n == 0:
            raise RuntimeError(
                f"could not locate env_id key in {path}; template may have changed"
            )
        path.write_text(new_text, encoding="utf-8")

    def read_metric_history(self, slug: str) -> list[float | None]:
        """Load `<project>/reports/metric_history.json` as a list.
        Empty list if the file is missing or malformed."""
        path = self._project_dir(slug) / "reports" / "metric_history.json"
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        history = payload.get("history") if isinstance(payload, dict) else None
        if not isinstance(history, list):
            return []
        out: list[float | None] = []
        for v in history:
            if v is None:
                out.append(None)
            else:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    out.append(None)
        return out

    def read_seeds_yaml(self, slug: str) -> Path:
        return self._project_dir(slug) / "kg_seeds.yml"

    def append_seeds(
        self, slug: str, seeds: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Append seeds to kg_seeds.yml (idempotent on arxiv_id). Returns
        the full updated seeds list. Caller holds the per-project lock.
        """
        import yaml  # transitive dep via sculptor

        path = self.read_seeds_yaml(slug)
        if path.is_file():
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            doc = {}
        existing = doc.get("papers") or []
        if not isinstance(existing, list):
            existing = []
        seen = {
            (e.get("arxiv_id") if isinstance(e, dict) else None)
            for e in existing
        }
        for s in seeds:
            aid = s.get("arxiv_id") if isinstance(s, dict) else None
            if not aid or aid in seen:
                continue
            existing.append({k: v for k, v in s.items() if v is not None})
            seen.add(aid)
        doc["papers"] = existing
        path.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return list(existing)

    def pending_seeds(self, slug: str) -> list[dict[str, Any]]:
        """Return the contents of kg_seeds.yml as a plain list of dicts."""
        import yaml

        path = self.read_seeds_yaml(slug)
        if not path.is_file():
            return []
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return []
        seeds = doc.get("papers") or []
        return [s for s in seeds if isinstance(s, dict)]

    def invalidate_preview(self, slug: str) -> None:
        """Remove every cached preview PNG so the next GET regenerates.
        One file per camera angle so we have to glob."""
        uploads = self._project_dir(slug) / "uploads"
        if not uploads.is_dir():
            return
        for p in uploads.glob("preview_*.png"):
            try:
                p.unlink()
            except OSError:
                pass

    # ── delete ───────────────────────────────────────────────────────────
    def delete(self, slug: str) -> bool:
        """Move a project directory into the trash (chunk A1 — no
        longer a hard delete; see `backend.services.trash`). Acquires
        the per-project lock briefly to block concurrent writes, then
        releases BEFORE the move (Windows can't relocate a directory
        with open file handles)."""
        project_dir = self._project_dir(slug)
        if not project_dir.is_dir():
            return False
        lock = self.acquire_lock(slug)
        try:
            # Holding the lock here asserts no other writer is active.
            pass
        finally:
            lock.release()
        from backend.services import trash

        trash.move_to_trash(
            project_dir, kind="project", slug=slug, origin=project_dir,
        )
        return True


# ── module helpers ─────────────────────────────────────────────────────
def _compute_migration_warning(adapter_class: str) -> Optional[str]:
    """M6: return a human-readable warning when the project's adapter
    class_path is not in the current `ADAPTER_REGISTRY`. Returns None
    when the adapter is still valid (including coming-soon adapters —
    those are not legacy, they're forthcoming).
    """
    if not adapter_class:
        return None
    try:
        from sculptor.adapters import ADAPTER_REGISTRY  # type: ignore
    except ImportError:
        return None
    known = {info.class_path for info in ADAPTER_REGISTRY.values()}
    if adapter_class in known:
        return None
    return (
        f"Adapter {adapter_class!r} is no longer registered in "
        f"sculptor.adapters.ADAPTER_REGISTRY. The project will load but "
        f"training may fail. Fork to a new project using a current "
        f"adapter if you want to keep iterating on this robot."
    )


def _parse_adapter(config_path: Path) -> tuple[str, dict]:
    if not config_path.is_file():
        return "", {}
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        with config_path.open("rb") as f:
            cfg = tomllib.load(f)
    except Exception:  # noqa: BLE001
        return "", {}
    adapter_section = cfg.get("adapter", {}) or {}
    return (
        str(adapter_section.get("class", "")),
        dict(adapter_section.get("config", {}) or {}),
    )


def _compute_status(project_dir: Path, metadata: dict, n_iters: int) -> ProjectStatus:
    """Compute the project's status from filesystem state + sidecar.

    Rules (in order):
      - draft: config.toml has CHANGE_ME markers
      - configured: no kg/graph.db
      - ready: iterations count == 0
      - errored: last run recorded as errored in sidecar
      - completed: otherwise

    `running` is a transient state owned by the job manager (not
    implemented in this prompt); it never returns from this function.
    """
    config_path = project_dir / "config.toml"
    if config_path.is_file():
        txt = config_path.read_text(encoding="utf-8", errors="replace")
        if "CHANGE_ME" in txt:
            return "draft"
    # The KG moved to one user-wide shared store (M7 Phase 1); the legacy
    # per-project kg/graph.db is never created anymore, which pinned every
    # project at "configured" forever. Resolve through the same helper the
    # KG routes use (it always returns the shared path) and fall back to
    # the legacy file only if resolution itself fails.
    try:
        from backend.services.kg_store import project_kg_db_path

        kg_db = project_kg_db_path(project_dir)
    except Exception:  # noqa: BLE001 — status must never raise
        kg_db = project_dir / "kg" / "graph.db"
    if not kg_db.is_file():
        return "configured"
    if n_iters == 0:
        return "ready"
    if metadata.get("last_run_status") == "errored":
        return "errored"
    return "completed"


def _parse_iso(s: Optional[str]) -> datetime:
    """Parse an RFC 3339 string; fall back to epoch if malformed."""
    if not s:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)

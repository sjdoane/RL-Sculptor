"""sculptor/adapters/mjcf_editor.py — pure Claude-driven MJCF rewrite.

§Ship-8b factoring pass: lifts the Claude-orchestration half of
`backend/services/physics.py::apply_prompt_edit` into sculptor so
`sculptor.sculpt._run_one_iter` can apply a physics edit inline
(between rollout audit and next-iter train) WITHOUT creating a
`sculptor → backend` import cycle.

What stays in backend/physics.py:
  - materialize (library → project-local copy)
  - git commit
  - route-layer response wrapping (diff, classification)

What moves here:
  - Claude prompt build + LLM call
  - `<?rs-summary ... ?>` parser + `<mujoco>` body extractor
  - KG-context rendering (reuses `sculptor.diagnose._render_kg_context`)
  - mujoco XML validation via tempfile sibling

Design rules (locked by the Ship-8b critique pass):
  - **No filesystem mutation of the project dir** unless `write=True`.
    When False, the function returns the new XML + validation result
    and the caller decides what to do. Makes this easy to drive from
    unit tests that don't want to touch disk.
  - **No git / no materialize**. Both live in the backend because
    sculpt-side callers already own a materialized XML and have their
    own git helper (`sculptor.sculpt._git_add_commit`).
  - **Retry: one.** Same pattern as `sculptor.edit.apply_edits` — if
    the parse / validate fails, one retry with the error appended,
    then surface the failure.
"""

from __future__ import annotations

import difflib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


_MODEL_ID = "claude-opus-4-7"
_MAX_TOKENS = 16_000

_SUMMARY_PI_RE = re.compile(r"<\?rs-summary\s+(.*?)\s*\?>", re.DOTALL)


def parse_claude_output(output: str) -> tuple[str, str]:
    """Extract `(commit_summary, xml)` from a Claude response. Raises
    `ValueError` when the response violates the contract in
    `sculptor/prompts/physics_editor.md`.

    Contract:
      1. response opens with `<?rs-summary ...?>` processing instruction
      2. followed by `<mujoco ...>...</mujoco>` root element

    §Ship-8c hotfix (critique 1): only accept a PI that appears BEFORE
    the first `<mujoco`. Without this anchor, a Claude response that
    quoted back the original MJCF verbatim could contain an embedded
    `<?rs-summary ?>` comment from a prior reject — the parser would
    pick that token instead of Claude's real summary.
    """
    xml_start = output.find("<mujoco")
    xml_end = output.rfind("</mujoco>")
    if xml_start < 0 or xml_end < 0:
        raise ValueError(
            "physics edit response missing <mujoco>...</mujoco> body"
        )
    preface = output[:xml_start]
    m = _SUMMARY_PI_RE.search(preface)
    if m is None:
        raise ValueError(
            "physics edit response missing `<?rs-summary ... ?>` PI "
            "before <mujoco> body"
        )
    summary = m.group(1).strip()
    xml = output[xml_start:xml_end + len("</mujoco>")]
    return summary, xml


_MJCF_WRITE_LOCK_TIMEOUT_S = 30.0


def _write_with_lock(mjcf_path: Path, new_xml: str) -> None:
    """Write `new_xml` to `mjcf_path` under a file lock so two concurrent
    editors (sculpt auto-apply + backend Physics route) can't interleave
    a half-written XML. Lock file is `<mjcf_path>.lock` sibling — cleaned
    up by `filelock` on release.

    Uses `filelock` if available (it's a backend dep, also installed in
    sculptor's venv). Falls back to a plain write with a warning if the
    package is missing — degrades gracefully rather than crashing the
    sculpt loop.
    """
    try:
        from filelock import FileLock
    except ImportError:
        log.warning(
            "filelock not installed — writing MJCF without lock; "
            "concurrent editors may race."
        )
        mjcf_path.write_text(new_xml, encoding="utf-8")
        return
    lock_path = str(mjcf_path) + ".lock"
    with FileLock(lock_path, timeout=_MJCF_WRITE_LOCK_TIMEOUT_S):
        mjcf_path.write_text(new_xml, encoding="utf-8")


def _diff_lines(a: str, b: str, *, max_lines: int = 400) -> list[str]:
    return list(difflib.unified_diff(
        a.splitlines(keepends=True), b.splitlines(keepends=True),
        fromfile="before.xml", tofile="after.xml", n=3,
    ))[:max_lines]


def _validate_xml(new_xml: str, sibling_dir: Path) -> Optional[str]:
    """Validate `new_xml` by writing to a unique sibling tempfile and
    loading via `mujoco.MjModel.from_xml_path`. Returns None on success
    or a reason string on failure.

    §Ship-8b hotfix (critique critical-2): use `NamedTemporaryFile` with
    `delete=False` so two concurrent `apply_mjcf_edit` calls in the same
    project dir don't clobber each other's validation file. The old
    fixed `.__rs_validate.xml` name meant one caller's `finally: unlink`
    could delete the other's in-flight file → false rejections.

    Sibling-path approach matches the parent MJCF's `meshdir` /
    `texturedir` references — `from_xml_string` fails on mesh lookups.
    """
    import mujoco
    import tempfile
    import os as _os

    # Prefix starts with `.` so the file is hidden from `ls` + skipped
    # by the `*.xml` glob in `_find_materialized_mjcf`. `delete=False`
    # means the handle releases Windows file locks before mujoco opens
    # the path; we `os.unlink` in `finally` regardless of success.
    fd, path_str = tempfile.mkstemp(
        prefix=".__rs_validate_", suffix=".xml", dir=str(sibling_dir),
    )
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_xml)
        try:
            mujoco.MjModel.from_xml_path(path_str)
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        return None
    finally:
        try:
            _os.unlink(path_str)
        except OSError:
            pass


def _render_kg_block(kg_store, user_prompt: str) -> tuple[str, list[dict]]:
    """Query the KG for `user_prompt`, render a LITERATURE CONTEXT
    block plus a list[dict] of citations. Both empty when `kg_store`
    is None / query fails.
    """
    if kg_store is None:
        return "# LITERATURE CONTEXT\n(no KG matches)\n", []
    try:
        from sculptor.diagnose import _render_kg_context
        from sculptor.kg.query import query_semantic

        matches = query_semantic(user_prompt, top_k=5, store=kg_store)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "mjcf_editor: KG consultation failed: %s: %s",
            type(e).__name__, e,
        )
        return "# LITERATURE CONTEXT\n(no KG matches)\n", []
    if not matches:
        return "# LITERATURE CONTEXT\n(no KG matches)\n", []
    block = _render_kg_context(matches) + "\n"
    citations = [
        {
            "technique": m.technique.name,
            "paper_citation": m.paper_citation,
            "relevance_score": round(float(m.relevance_score), 3),
            "source_paper_ids": list(m.source_paper_ids or []),
        }
        for m in matches
    ]
    return block, citations


def _build_user_msg(
    *, user_prompt: str, adapter_hint: str, old_xml: str,
    summary_digest: dict, kg_block: str,
) -> str:
    return (
        f"USER REQUEST:\n{user_prompt}\n\n"
        f"ADAPTER: {adapter_hint}\n\n"
        f"{kg_block}\n"
        f"CURRENT DIGEST:\n"
        f"  timestep={summary_digest.get('timestep')}, "
        f"gravity={summary_digest.get('gravity')}, "
        f"integrator={summary_digest.get('integrator')}, "
        f"n_joints={len(summary_digest.get('joints') or [])}, "
        f"n_actuators={len(summary_digest.get('actuators') or [])}\n\n"
        f"CURRENT MJCF:\n{old_xml}"
    )


def _call_claude(client, system_prompt: str, user_msg: str) -> str:
    resp = client.messages.create(
        model=_MODEL_ID,
        max_tokens=_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(
        block.text for block in resp.content
        if getattr(block, "type", None) == "text"
    )


def apply_mjcf_edit(
    mjcf_path: Path | str,
    user_prompt: str,
    *,
    adapter_hint: str = "unknown",
    summary_digest: Optional[dict] = None,
    client=None,
    kg_store=None,
    write: bool = True,
) -> dict:
    """Claude rewrites the MJCF at `mjcf_path` per `user_prompt`.

    Returns the same shape as `backend/services/physics.py::apply_prompt_edit`
    sans git + materialize:

        {
          "new_xml": str,
          "old_xml": str,
          "summary": str,                # Claude's 1-line summary
          "committed": bool,             # True iff validated + written
          "rejected_reason": str | None,
          "rejected_at": "parse" | "claude_rejected" | "mujoco_validate" | None,
          "diff_lines": list[str] | None,
          "claude_output_raw": str | None,
          "kg_citations": list[dict],
        }

    `write=False` returns the validated XML in `new_xml` without
    touching `mjcf_path` — useful for tests + dry-run flows.

    `client=None` constructs a fresh `anthropic.Anthropic(max_retries=6)`.
    Caller is responsible for ensuring `ANTHROPIC_API_KEY` is set.

    Does NOT perform git commits or MJCF materialization — those are
    backend-side concerns that live in `backend/services/physics.py`.
    """
    mjcf_path = Path(mjcf_path).resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    user_prompt = (user_prompt or "").strip()
    if len(user_prompt) < 3 or len(user_prompt) > 2000:
        raise ValueError("prompt must be 3-2000 chars")

    old_xml = mjcf_path.read_text(encoding="utf-8", errors="replace")

    if client is None:
        import os
        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            raise RuntimeError(
                "apply_mjcf_edit requires ANTHROPIC_API_KEY in the environment"
            )
        import anthropic
        client = anthropic.Anthropic(max_retries=6)

    from sculptor.prompts import load_prompt
    system = load_prompt("physics_editor")

    kg_block, kg_citations = _render_kg_block(kg_store, user_prompt)
    user_msg = _build_user_msg(
        user_prompt=user_prompt,
        adapter_hint=adapter_hint,
        old_xml=old_xml,
        summary_digest=summary_digest or {},
        kg_block=kg_block,
    )

    output_text = _call_claude(client, system, user_msg)

    try:
        summary, new_xml = parse_claude_output(output_text)
    except ValueError as e:
        return {
            "new_xml": old_xml,
            "old_xml": old_xml,
            "summary": f"REJECTED: {e}",
            "committed": False,
            "rejected_reason": str(e),
            "rejected_at": "parse",
            "diff_lines": None,
            "claude_output_raw": output_text,
            "kg_citations": kg_citations,
        }

    if summary.upper().startswith("REJECTED:"):
        return {
            "new_xml": new_xml,
            "old_xml": old_xml,
            "summary": summary,
            "committed": False,
            "rejected_reason": summary[len("REJECTED:"):].strip(),
            "rejected_at": "claude_rejected",
            "diff_lines": _diff_lines(old_xml, new_xml),
            "claude_output_raw": None,
            "kg_citations": kg_citations,
        }

    validate_reason = _validate_xml(new_xml, mjcf_path.parent)
    if validate_reason is not None:
        return {
            "new_xml": new_xml,
            "old_xml": old_xml,
            "summary": f"REJECTED: mujoco rejected the new MJCF: {validate_reason}",
            "committed": False,
            "rejected_reason": f"mujoco rejected the new MJCF: {validate_reason}",
            "rejected_at": "mujoco_validate",
            "diff_lines": _diff_lines(old_xml, new_xml),
            "claude_output_raw": None,
            "kg_citations": kg_citations,
        }

    if write:
        # §Ship-8b hotfix (critique critical-1): concurrent writers
        # (sculpt auto-apply + backend Physics-tab edit) could otherwise
        # produce a torn MJCF. Serialize via a file lock sibling to the
        # MJCF; every writer acquires before the final write_text.
        _write_with_lock(mjcf_path, new_xml)

    return {
        "new_xml": new_xml,
        "old_xml": old_xml,
        "summary": summary,
        "committed": True,
        "rejected_reason": None,
        "rejected_at": None,
        "diff_lines": _diff_lines(old_xml, new_xml),
        "claude_output_raw": None,
        "kg_citations": kg_citations,
    }

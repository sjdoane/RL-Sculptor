"""tests/test_mjcf_editor.py — §Ship-8b pure Claude-driven MJCF rewrite.

No live LLM calls. Stubs Claude responses + mujoco validation so the
parse / validate / write / rejection paths are all exercised CPU-only.
Mirrors the existing backend `test_physics.py::test_apply_prompt_edit_*`
tests but for the factored-out module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.adapters.mjcf_editor import (
    apply_mjcf_edit,
    parse_claude_output,
)


_MINIMAL_MJCF = """<?xml version="1.0"?>
<mujoco model="tiny">
  <worldbody>
    <body name="b1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


# ── Stub Claude client ────────────────────────────────────────────────────
class _StubBlock:
    type = "text"
    def __init__(self, text: str):
        self.text = text


class _StubResp:
    def __init__(self, text: str):
        self.content = [_StubBlock(text)]


class _StubMessages:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("stub ran out of responses")
        return _StubResp(self._responses.pop(0))


class _StubClient:
    def __init__(self, *responses: str):
        self.messages = _StubMessages(*responses)


# ── parse_claude_output ───────────────────────────────────────────────────
def test_parse_claude_output_extracts_summary_and_xml() -> None:
    output = (
        "<?rs-summary reduced forcerange on knee joints?>\n"
        "<mujoco>\n  <body/>\n</mujoco>"
    )
    summary, xml = parse_claude_output(output)
    assert summary == "reduced forcerange on knee joints"
    assert xml.startswith("<mujoco>")
    assert xml.endswith("</mujoco>")


def test_parse_claude_output_missing_pi_raises() -> None:
    with pytest.raises(ValueError, match="rs-summary"):
        parse_claude_output("<mujoco/></mujoco>")


def test_parse_claude_output_missing_mujoco_body_raises() -> None:
    with pytest.raises(ValueError, match="mujoco"):
        parse_claude_output("<?rs-summary x?>")


def test_parse_claude_output_ignores_pi_inside_quoted_old_xml() -> None:
    """§Ship-8c hotfix (critique 1): if Claude echoes the old MJCF
    verbatim and the old MJCF happened to contain a `<?rs-summary ?>`
    token (e.g. pasted from a prior rejection), the parser must use
    Claude's OWN PI that comes BEFORE the <mujoco> body, not the
    embedded one."""
    output = (
        "<?rs-summary Claude-owned summary?>\n"
        "<mujoco model='x'>"
        # Embedded PI that looks like ours but is inside the XML body.
        "  <!-- user put this in: <?rs-summary EMBEDDED old stub?> -->"
        "</mujoco>"
    )
    summary, xml = parse_claude_output(output)
    assert summary == "Claude-owned summary"
    # Sanity: the embedded PI is still in the XML (parser shouldn't strip it).
    assert "EMBEDDED" in xml


def test_parse_claude_output_pi_after_mujoco_is_invalid() -> None:
    """If the only PI in the response is AFTER the `<mujoco>` opening
    tag, that's not Claude's summary — reject."""
    output = (
        "<mujoco model='x'></mujoco>"
        "<?rs-summary too late?>"
    )
    with pytest.raises(ValueError, match="before <mujoco>"):
        parse_claude_output(output)


# ── apply_mjcf_edit: happy path ───────────────────────────────────────────
@pytest.fixture
def mjcf_path(tmp_path: Path) -> Path:
    p = tmp_path / "robot" / "base.xml"
    p.parent.mkdir(parents=True)
    p.write_text(_MINIMAL_MJCF, encoding="utf-8")
    return p


def test_apply_mjcf_edit_happy_path_writes_new_xml(mjcf_path: Path) -> None:
    new_body = _MINIMAL_MJCF.replace("0.1 0.1 0.1", "0.2 0.2 0.2")
    client = _StubClient(f"<?rs-summary bumped geom size?>\n{new_body}")
    result = apply_mjcf_edit(
        mjcf_path, "make it bigger",
        adapter_hint="test", client=client, kg_store=None,
    )
    assert result["committed"] is True
    assert result["rejected_reason"] is None
    assert result["summary"] == "bumped geom size"
    assert "0.2 0.2 0.2" in result["new_xml"]
    # Disk was written.
    assert "0.2 0.2 0.2" in mjcf_path.read_text(encoding="utf-8")
    assert result["diff_lines"] is not None and len(result["diff_lines"]) > 0


def test_apply_mjcf_edit_write_false_does_not_touch_disk(mjcf_path: Path) -> None:
    original = mjcf_path.read_text(encoding="utf-8")
    new_body = _MINIMAL_MJCF.replace("0.1 0.1 0.1", "0.5 0.5 0.5")
    client = _StubClient(f"<?rs-summary test dry-run?>\n{new_body}")
    result = apply_mjcf_edit(
        mjcf_path, "dry run",
        adapter_hint="test", client=client, kg_store=None,
        write=False,
    )
    assert result["committed"] is True
    # Returned XML has the change...
    assert "0.5 0.5 0.5" in result["new_xml"]
    # ...but disk does not.
    assert mjcf_path.read_text(encoding="utf-8") == original


# ── Rejection paths ───────────────────────────────────────────────────────
def test_apply_mjcf_edit_rejects_parse_failure(mjcf_path: Path) -> None:
    original = mjcf_path.read_text(encoding="utf-8")
    client = _StubClient("no summary PI, no mujoco body — just text")
    result = apply_mjcf_edit(
        mjcf_path, "please adjust",
        adapter_hint="test", client=client,
    )
    assert result["committed"] is False
    assert result["rejected_at"] == "parse"
    assert result["claude_output_raw"] is not None
    # Disk unchanged.
    assert mjcf_path.read_text(encoding="utf-8") == original


def test_apply_mjcf_edit_rejects_claude_refusal(mjcf_path: Path) -> None:
    """Claude can self-reject by prefixing its summary with 'REJECTED:'.
    `apply_mjcf_edit` must NOT write that MJCF to disk even if the XML
    otherwise parses."""
    original = mjcf_path.read_text(encoding="utf-8")
    client = _StubClient(
        f"<?rs-summary REJECTED: request is unsafe?>\n{_MINIMAL_MJCF}"
    )
    result = apply_mjcf_edit(
        mjcf_path, "do something bad", adapter_hint="test", client=client,
    )
    assert result["committed"] is False
    assert result["rejected_at"] == "claude_rejected"
    assert "unsafe" in result["rejected_reason"].lower()
    # Disk unchanged.
    assert mjcf_path.read_text(encoding="utf-8") == original


def test_apply_mjcf_edit_rejects_invalid_mujoco_xml(mjcf_path: Path) -> None:
    original = mjcf_path.read_text(encoding="utf-8")
    # MuJoCo parses the XML structure but rejects non-sensical fields.
    bad_mjcf = (
        '<mujoco model="broken">'
        '  <worldbody>'
        '    <body name="x">'
        '      <joint name="j1" type="NOT_A_REAL_TYPE"/>'
        '    </body>'
        '  </worldbody>'
        '</mujoco>'
    )
    client = _StubClient(f"<?rs-summary breaks mujoco?>\n{bad_mjcf}")
    result = apply_mjcf_edit(
        mjcf_path, "please adjust",
        adapter_hint="test", client=client,
    )
    assert result["committed"] is False
    assert result["rejected_at"] == "mujoco_validate"
    assert result["diff_lines"] is not None
    # Disk unchanged — validation failed before write.
    assert mjcf_path.read_text(encoding="utf-8") == original


def test_apply_mjcf_edit_cleans_up_validation_tempfile(
    mjcf_path: Path,
) -> None:
    """Whether the edit is accepted or rejected, the
    `.__rs_validate.xml` sibling must not be left on disk."""
    client = _StubClient(
        f"<?rs-summary t?>\n{_MINIMAL_MJCF.replace('0.1 0.1 0.1', '0.3 0.3 0.3')}"
    )
    apply_mjcf_edit(
        mjcf_path, "please adjust",
        adapter_hint="test", client=client,
    )
    assert not (mjcf_path.parent / ".__rs_validate.xml").exists()


# ── Input validation ──────────────────────────────────────────────────────
def test_apply_mjcf_edit_prompt_too_short(mjcf_path: Path) -> None:
    with pytest.raises(ValueError, match="3-2000"):
        apply_mjcf_edit(mjcf_path, "ok")  # 2 chars — below min


def test_apply_mjcf_edit_prompt_too_long(mjcf_path: Path) -> None:
    with pytest.raises(ValueError, match="3-2000"):
        apply_mjcf_edit(mjcf_path, "a" * 3000)


def test_apply_mjcf_edit_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        apply_mjcf_edit(tmp_path / "nope.xml", "hello world")


def test_apply_mjcf_edit_missing_api_key_raises(
    mjcf_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        apply_mjcf_edit(mjcf_path, "hello world")  # no client, no key


# ── §Ship-8b hotfix: tempfile uniqueness + write lock ─────────────────────
def test_validate_tempfile_is_unique_per_call(mjcf_path: Path) -> None:
    """Two `apply_mjcf_edit` calls in the same dir must not collide on
    the validation tempfile — old fixed name `.__rs_validate.xml`
    meant thread A's `finally: unlink` could kill thread B's in-flight
    file and cause a spurious reject."""
    import threading
    results: list[dict] = []

    def _run(i: int) -> None:
        # Sizes strictly positive (0.1, 0.2, 0.3, ...) so mujoco accepts.
        s = f"0.{i + 1}"
        new_body = _MINIMAL_MJCF.replace("0.1 0.1 0.1", f"{s} {s} {s}")
        client = _StubClient(f"<?rs-summary t{i}?>\n{new_body}")
        # write=False so we don't race on the real file — we're only
        # testing the validation-tempfile race.
        r = apply_mjcf_edit(
            mjcf_path, f"edit number {i}",
            adapter_hint="t", client=client, write=False,
        )
        results.append(r)

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    # All three should have committed (in-memory); none should have
    # failed validation due to a stolen tempfile.
    assert all(r["committed"] for r in results), [
        (r["committed"], r.get("rejected_at"), r.get("rejected_reason"))
        for r in results
    ]
    # And no stale tempfile must remain.
    leftover = list(mjcf_path.parent.glob(".__rs_validate_*.xml"))
    assert leftover == [], leftover


def test_apply_mjcf_edit_write_is_lock_protected(mjcf_path: Path) -> None:
    """Concurrent `apply_mjcf_edit(write=True)` calls on the same
    `mjcf_path` must serialize — the resulting file must match one of
    the parsed outputs, not be a torn mix of two."""
    import threading

    # Two distinct MJCFs that differ only in a geom size. Both have
    # their `<?xml ?>` header stripped by `parse_claude_output` before
    # being written, so we compare against the parsed form.
    xml_a_raw = _MINIMAL_MJCF.replace("0.1 0.1 0.1", "0.5 0.5 0.5")
    xml_b_raw = _MINIMAL_MJCF.replace("0.1 0.1 0.1", "0.9 0.9 0.9")
    _, xml_a_parsed = parse_claude_output(f"<?rs-summary a?>\n{xml_a_raw}")
    _, xml_b_parsed = parse_claude_output(f"<?rs-summary b?>\n{xml_b_raw}")

    def _edit(xml: str) -> None:
        client = _StubClient(f"<?rs-summary concurrent?>\n{xml}")
        apply_mjcf_edit(
            mjcf_path, "concurrent edit",
            adapter_hint="t", client=client, write=True,
        )

    threads = [
        threading.Thread(target=_edit, args=(xml_a_raw,)),
        threading.Thread(target=_edit, args=(xml_b_raw,)),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    final = mjcf_path.read_text(encoding="utf-8")
    assert final == xml_a_parsed or final == xml_b_parsed, (
        "torn write detected — final file matches neither input"
    )


# ── §Ship-8b: sculpt-side auto-apply wiring (unit-level) ──────────────────
def test_find_materialized_mjcf_happy_path(tmp_path: Path) -> None:
    from sculptor.sculpt import _find_materialized_mjcf
    p = tmp_path / "uploads" / "robot" / "base.xml"
    p.parent.mkdir(parents=True)
    p.write_text(_MINIMAL_MJCF, encoding="utf-8")
    assert _find_materialized_mjcf(tmp_path) == p


def test_find_materialized_mjcf_none_when_missing(tmp_path: Path) -> None:
    from sculptor.sculpt import _find_materialized_mjcf
    assert _find_materialized_mjcf(tmp_path) is None
    (tmp_path / "uploads" / "robot").mkdir(parents=True)
    # Dir exists but no xml.
    assert _find_materialized_mjcf(tmp_path) is None


def test_scoped_git_commit_only_stages_requested_paths(tmp_path: Path) -> None:
    """`_scoped_git_commit(paths=['uploads/robot'])` must NOT bundle
    in-flight reward / run artifacts into an auto-physics commit."""
    import subprocess
    from sculptor.sculpt import _scoped_git_commit, _git_add_commit

    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(project), "commit", "-m", "init", "--allow-empty"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "config", "user.email", "t@t"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "config", "user.name", "t"],
        check=True,
    )

    # Create dirty files in multiple subdirs.
    (project / "uploads" / "robot").mkdir(parents=True)
    (project / "uploads" / "robot" / "base.xml").write_text("<mujoco/>")
    (project / "runs").mkdir()
    (project / "runs" / "dirty.txt").write_text("should NOT be committed")
    (project / "reports").mkdir()
    (project / "reports" / "other.txt").write_text("also not")

    ok = _scoped_git_commit(
        project, paths=["uploads/robot"], message="scoped test",
    )
    assert ok is True

    # Confirm only uploads/robot landed in the commit.
    show = subprocess.run(
        ["git", "-C", str(project), "show", "--name-only",
         "--pretty=format:", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    committed_files = [ln.strip() for ln in show.stdout.splitlines() if ln.strip()]
    assert "uploads/robot/base.xml" in committed_files
    assert "runs/dirty.txt" not in committed_files
    assert "reports/other.txt" not in committed_files


def test_scoped_git_commit_refuses_absolute_path(tmp_path: Path) -> None:
    """§Ship-8c hotfix (critique 4): absolute path → refuse with False,
    never call `git add`."""
    import subprocess
    from sculptor.sculpt import _scoped_git_commit

    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)

    ok = _scoped_git_commit(
        project, paths=["/etc/passwd"], message="evil",
    )
    assert ok is False


def test_scoped_git_commit_refuses_parent_traversal(tmp_path: Path) -> None:
    """§Ship-8c hotfix (critique 4): `../` escape → refuse."""
    import subprocess
    from sculptor.sculpt import _scoped_git_commit

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("stolen")
    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)

    ok = _scoped_git_commit(
        project, paths=["../outside/leak.txt"], message="try traversal",
    )
    assert ok is False


def test_scoped_git_commit_strips_newlines_from_message(tmp_path: Path) -> None:
    """§Ship-8c hotfix (critique minor-11): multi-line commit message
    becomes a single clean first line ≤200 chars."""
    import subprocess
    from sculptor.sculpt import _scoped_git_commit

    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(project), "commit", "-m", "init", "--allow-empty"],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(project), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.name", "t"], check=True)

    (project / "file.txt").write_text("content")
    ok = _scoped_git_commit(
        project, paths=["file.txt"],
        message="first line\nsecond line\nthird with embedded \"quotes\"",
    )
    assert ok is True
    log = subprocess.run(
        ["git", "-C", str(project), "log", "-1", "--pretty=%s"],
        capture_output=True, text=True, check=True,
    )
    subject = log.stdout.strip()
    assert subject == "first line"

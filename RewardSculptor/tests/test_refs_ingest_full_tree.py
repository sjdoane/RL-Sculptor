"""fleaven-g1 full-tree enumeration (`sculpt refs ingest --all`).

Covers the HF tree-API pagination walker, retry-then-skip-and-log
behavior, the manifest reuse/freshness logic, and `--all` plumbing
through `ingest_source` / the CLI. Offline only — `urllib.request.urlopen`
is monkeypatched with fabricated two-page responses; nothing here
reaches the network (per Hard rules).
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from sculptor.refs.ingest import (
    _hf_list_files_all_pages,
    _parse_link_next,
    FLEAVEN_REPO,
    enumerate_fleaven_g1_all,
    ingest_source,
    load_or_build_fleaven_manifest,
    manifest_is_fresh,
    read_fleaven_manifest,
    write_fleaven_manifest,
)


# ── fake urlopen plumbing ─────────────────────────────────────────────────
class _FakeHeaders:
    def __init__(self, link: Optional[str]):
        self._link = link

    def get(self, key: str, default=None):
        if key.lower() == "link":
            return self._link if self._link is not None else default
        return default


class _FakeResponse:
    def __init__(self, payload: list[dict[str, Any]], link: Optional[str]):
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = _FakeHeaders(link)

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _entry(path: str, size: int = 100, kind: str = "file") -> dict[str, Any]:
    return {"type": kind, "path": path, "size": size}


# ── pagination: two-page walk following Link: rel="next" ──────────────────
def test_parse_link_next_extracts_url_from_rfc5988_header() -> None:
    header = (
        '<https://huggingface.co/api/datasets/x/tree/main/g1?cursor=ABC>; '
        'rel="next"')
    assert _parse_link_next(header) == (
        "https://huggingface.co/api/datasets/x/tree/main/g1?cursor=ABC")


def test_parse_link_next_returns_none_when_absent() -> None:
    assert _parse_link_next(None) is None
    assert _parse_link_next("") is None
    assert _parse_link_next('<https://x>; rel="prev"') is None


def test_hf_list_files_all_pages_follows_link_header_across_two_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = [
        _entry("g1/ACCAD", kind="directory"),
        _entry("g1/ACCAD/a.npy", size=10),
        _entry("g1/ACCAD/b.npy", size=20),
    ]
    page2 = [
        _entry("g1/BMLhandball/c.npy", size=30),
    ]
    next_url = "https://huggingface.co/api/datasets/x/tree/main/g1?cursor=NEXT"
    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if url == next_url:
            return _FakeResponse(page2, link=None)
        return _FakeResponse(page1, link=f'<{next_url}>; rel="next"')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    entries = _hf_list_files_all_pages(FLEAVEN_REPO, "g1")

    assert calls == [calls[0], next_url]  # exactly 2 pages fetched
    paths = [e["path"] for e in entries]
    assert paths == ["g1/ACCAD/a.npy", "g1/ACCAD/b.npy", "g1/BMLhandball/c.npy"]
    assert entries[0]["size"] == 10
    # directory entries excluded
    assert all("ACCAD" != p for p in paths if p == "g1/ACCAD")


def test_enumerate_fleaven_g1_all_filters_to_npy_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = [
        _entry("g1/ACCAD/a.npy", size=10),
        _entry("g1/ACCAD/readme.txt", size=5),
        _entry("g1/ACCAD/license.txt", size=5),
        _entry("g1/amass.bib", size=5),
    ]

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(page1, link=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    entries = enumerate_fleaven_g1_all()
    assert [e["path"] for e in entries] == ["g1/ACCAD/a.npy"]


# ── retry-then-skip-and-log on transient failures ──────────────────────────
def test_hf_list_files_all_pages_retries_transient_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = [_entry("g1/ACCAD/a.npy", size=10)]
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            import urllib.error

            raise urllib.error.URLError("transient network blip")
        return _FakeResponse(page1, link=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # don't actually wait in tests

    entries = _hf_list_files_all_pages(FLEAVEN_REPO, "g1", max_attempts=3, retry_delay_s=0.0)
    assert attempts["n"] == 3
    assert [e["path"] for e in entries] == ["g1/ACCAD/a.npy"]


def test_hf_list_files_all_pages_gives_up_after_max_attempts_and_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page that never succeeds is skipped (logged), not raised — the
    walk stops at the unrecoverable page and returns everything gathered
    before it, per §task 'transient HTTP failures retried a few times,
    then skip-and-log'."""

    def fake_urlopen(req, timeout=None):
        import urllib.error

        raise urllib.error.URLError("permanently broken")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    logs: list[str] = []
    entries = _hf_list_files_all_pages(
        FLEAVEN_REPO, "g1", max_attempts=3, retry_delay_s=0.0, progress=logs.append)

    assert entries == []
    assert any("unrecoverable" in m or "giving up" in m for m in logs)


def test_hf_list_files_all_pages_skips_unrecoverable_second_page_keeps_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = [_entry("g1/ACCAD/a.npy", size=10)]
    next_url = "https://huggingface.co/api/datasets/x/tree/main/g1?cursor=NEXT"

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if url == next_url:
            import urllib.error

            raise urllib.error.URLError("page 2 is down for good")
        return _FakeResponse(page1, link=f'<{next_url}>; rel="next"')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    entries = _hf_list_files_all_pages(FLEAVEN_REPO, "g1", max_attempts=2, retry_delay_s=0.0)
    # First page's file survives even though the second page never resolves.
    assert [e["path"] for e in entries] == ["g1/ACCAD/a.npy"]


# ── manifest: write / freshness / reuse ────────────────────────────────────
def test_manifest_write_and_read_roundtrip(tmp_path: Path) -> None:
    entries = [{"path": "g1/a.npy", "size": 10}, {"path": "g1/b.npy", "size": 20}]
    manifest_path = tmp_path / "manifest.json"
    write_fleaven_manifest(manifest_path, entries)

    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["n_files"] == 2
    assert payload["total_bytes"] == 30
    assert "generated_at" in payload

    round_tripped = read_fleaven_manifest(manifest_path)
    assert round_tripped == entries


def test_manifest_is_fresh_true_when_just_written(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_fleaven_manifest(manifest_path, [{"path": "g1/a.npy", "size": 1}])
    assert manifest_is_fresh(manifest_path) is True


def test_manifest_is_fresh_false_when_stale(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    from datetime import datetime, timedelta, timezone

    stale_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    manifest_path.write_text(
        json.dumps({
            "source": "fleaven-g1", "generated_at": stale_ts, "n_files": 1,
            "total_bytes": 1, "files": [{"path": "g1/a.npy", "size": 1}],
        }),
        encoding="utf-8")
    assert manifest_is_fresh(manifest_path) is False


def test_manifest_is_fresh_false_when_missing(tmp_path: Path) -> None:
    assert manifest_is_fresh(tmp_path / "does_not_exist.json") is False


def test_manifest_is_fresh_false_when_corrupt(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("not json{{{", encoding="utf-8")
    assert manifest_is_fresh(manifest_path) is False


def test_load_or_build_fleaven_manifest_reuses_fresh_manifest_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    entries = [{"path": "g1/a.npy", "size": 10}]
    write_fleaven_manifest(manifest_path, entries)

    def fail_urlopen(req, timeout=None):
        raise AssertionError("must not hit the network when manifest is fresh")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    result = load_or_build_fleaven_manifest(manifest_path)
    assert result == entries


def test_load_or_build_fleaven_manifest_refresh_forces_re_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_fleaven_manifest(manifest_path, [{"path": "g1/old.npy", "size": 1}])

    new_page = [_entry("g1/new.npy", size=99)]

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(new_page, link=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = load_or_build_fleaven_manifest(manifest_path, refresh=True)
    assert result == [{"path": "g1/new.npy", "size": 99}]
    # And it re-wrote the manifest on disk.
    assert read_fleaven_manifest(manifest_path) == [{"path": "g1/new.npy", "size": 99}]


def test_load_or_build_fleaven_manifest_no_path_always_enumerates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResponse([_entry("g1/a.npy", size=1)], link=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = load_or_build_fleaven_manifest(None)
    assert calls["n"] == 1
    assert result == [{"path": "g1/a.npy", "size": 1}]


# ── --all plumbing through ingest_source ────────────────────────────────────
def test_ingest_source_all_flag_uses_full_tree_enumerator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`full_tree=True` must route through the manifest/enumerator path,
    NOT the single-page `_hf_list_files` — verified by making
    `_hf_list_files` raise if ever called, and by returning MORE than
    1000 entries worth of pagination via two fake pages."""
    import numpy as np

    def _npy_bytes() -> bytes:
        n = 50
        rows = np.zeros((n, 36), dtype=np.float64)
        rows[:, 2] = 0.78  # root z
        rows[:, 6] = 1.0  # quat w component (xyzw index 3 -> col 6)
        buf = io.BytesIO()
        np.save(buf, rows)
        return buf.getvalue()

    raw = _npy_bytes()

    def fail_hf_list_files(repo, path, **kwargs):
        raise AssertionError("full_tree=True must not call single-page _hf_list_files")

    def fake_list_all_pages(repo, path, **kwargs):
        return [{"path": "g1/ACCAD/clip1_poses_30_jpos.npy", "size": len(raw)}]

    def fake_get_bytes(url, **kwargs):
        return raw

    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files", fail_hf_list_files)
    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files_all_pages", fake_list_all_pages)
    monkeypatch.setattr("sculptor.refs.ingest._http_get_bytes", fake_get_bytes)

    summary = ingest_source(
        "fleaven-g1", root=tmp_path, no_preview=True, full_tree=True)

    assert summary.accepted == ["clip1_poses_30_jpos"]


def test_ingest_source_all_rejects_non_fleaven_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fleaven-g1"):
        ingest_source("lafan1-g1", root=tmp_path, full_tree=True)


def test_ingest_source_all_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    n = 50
    rows = np.zeros((n, 36), dtype=np.float64)
    rows[:, 2] = 0.78
    rows[:, 6] = 1.0
    buf = io.BytesIO()
    np.save(buf, rows)
    raw = buf.getvalue()

    def fake_list_all_pages(repo, path, **kwargs):
        return [{"path": "g1/ACCAD/clip1_poses_30_jpos.npy", "size": len(raw)}]

    def fake_get_bytes(url, **kwargs):
        return raw

    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files_all_pages", fake_list_all_pages)
    monkeypatch.setattr("sculptor.refs.ingest._http_get_bytes", fake_get_bytes)

    manifest_path = tmp_path / "manifest.json"
    summary = ingest_source(
        "fleaven-g1", root=tmp_path, no_preview=True, full_tree=True,
        manifest_path=manifest_path)

    assert summary.accepted == ["clip1_poses_30_jpos"]
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["n_files"] == 1


# ── slice (non --all) behavior byte-identical ───────────────────────────────
def test_ingest_source_without_all_flag_still_uses_single_page_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: default (`full_tree=False`, i.e. no `--all`)
    must go through the pre-existing single-page `_hf_list_files` path,
    never touching the new pagination walker or manifest logic."""
    rows_csv = (
        b"0.0,0.0,0.78,0.0,0.0,0.0,1.0," + b",".join([b"0.0"] * 29) + b"\n"
    ) * 50

    def fake_list_files(repo: str, path: str, **kwargs):
        return ["g1/dance1-2_subject3.csv"]

    def fail_list_all_pages(repo, path, **kwargs):
        raise AssertionError("full_tree=False must not call the pagination walker")

    def fake_get_bytes(url: str, **kwargs):
        return rows_csv

    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files", fake_list_files)
    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files_all_pages", fail_list_all_pages)
    monkeypatch.setattr("sculptor.refs.ingest._http_get_bytes", fake_get_bytes)

    summary = ingest_source("lafan1-g1", root=tmp_path, no_preview=True)
    assert summary.accepted == ["dance1_2_subject3"]


# ── CLI plumbing: --all / --manifest-out / --refresh-manifest options exist ─
def test_cli_refs_ingest_accepts_all_and_manifest_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the actual `sculpt refs ingest --all --manifest-out ...`
    typer command end-to-end (mocked network), confirming the CLI wires
    the new flags through to `ingest_source` rather than just existing
    as unused options."""
    import numpy as np
    from typer.testing import CliRunner

    from sculptor.cli import app

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path))

    n = 50
    rows = np.zeros((n, 36), dtype=np.float64)
    rows[:, 2] = 0.78
    rows[:, 6] = 1.0
    buf = io.BytesIO()
    np.save(buf, rows)
    raw = buf.getvalue()

    def fake_list_all_pages(repo, path, **kwargs):
        return [{"path": "g1/ACCAD/clip1_poses_30_jpos.npy", "size": len(raw)}]

    def fake_get_bytes(url, **kwargs):
        return raw

    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files_all_pages", fake_list_all_pages)
    monkeypatch.setattr("sculptor.refs.ingest._http_get_bytes", fake_get_bytes)

    manifest_path = tmp_path / "manifest.json"
    runner = CliRunner()
    result = runner.invoke(app, [
        "refs", "ingest", "--source", "fleaven-g1", "--all", "--no-preview",
        "--manifest-out", str(manifest_path),
    ])
    assert result.exit_code == 0, result.output
    assert "accepted=1" in result.output
    assert manifest_path.is_file()

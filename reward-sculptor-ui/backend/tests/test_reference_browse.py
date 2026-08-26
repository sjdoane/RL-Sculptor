"""The clip library has to be browsable, not just searchable.

`GET /references` defaults to `k=10` against a ~6000-clip library and takes no
offset, so the picker showed the ten alphabetically-first clips and a freshly
composed motion was findable only by typing its own id back into a semantic
search box. `GET /references/browse` is the paginated, faceted counterpart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def library_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small index in registration order: corpus first, composite last."""
    root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    from sculptor.refs import library

    path = library.index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"clip_id": "balance_on_beam03_poses_100_jpos", "robot": "g1",
         "text": "balance on a beam", "labels": ["poses", "100", "jpos"],
         "tier": "K", "duration_s": 4.0, "fps": 100.0, "n_frames": 400},
        {"clip_id": "run03_poses_100_jpos", "robot": "g1",
         "text": "running", "labels": ["poses", "100", "jpos"],
         "tier": "K", "duration_s": 1.02, "fps": 100.0, "n_frames": 102},
        {"clip_id": "go1_trot", "robot": "go1", "text": "trotting",
         "labels": ["gait"], "tier": "A", "duration_s": 2.0, "fps": 50.0,
         "n_frames": 100},
        {"clip_id": "platform-ascent--g1", "robot": "g1",
         "text": "platform ascent", "labels": ["novel", "composed"],
         "tier": "K", "duration_s": 6.92, "fps": 100.0, "n_frames": 692},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def test_browse_returns_a_total_not_just_a_page(
    client: TestClient, library_index: Path,
) -> None:
    d = client.get("/references/browse",
                   params={"robot": "g1", "limit": 1}).json()
    assert d["total"] == 3, "3 g1 clips, page size 1"
    assert len(d["rows"]) == 1
    assert d["facets"]["total"] == 3
    assert d["facets"]["composed"] == 1


def test_browse_is_scoped_to_the_robot(
    client: TestClient, library_index: Path,
) -> None:
    """A go1 project must never be offered g1 motion."""
    ids = [r["clip_id"] for r in
           client.get("/references/browse",
                      params={"robot": "go1"}).json()["rows"]]
    assert ids == ["go1_trot"]


def test_the_composite_leads_the_default_view(
    client: TestClient, library_index: Path,
) -> None:
    """The clip you just made is the one you are looking for."""
    rows = client.get("/references/browse", params={"robot": "g1"}).json()["rows"]
    assert rows[0]["clip_id"] == "platform-ascent--g1"
    assert rows[0]["composed"] is True


def test_offset_pages_without_repeating_or_skipping(
    client: TestClient, library_index: Path,
) -> None:
    first = client.get("/references/browse",
                       params={"robot": "g1", "limit": 2}).json()["rows"]
    second = client.get("/references/browse",
                        params={"robot": "g1", "limit": 2,
                                "offset": 2}).json()["rows"]
    seen = [r["clip_id"] for r in first + second]
    assert len(seen) == 3
    assert len(set(seen)) == 3


def test_multi_word_queries_match_across_underscores(
    client: TestClient, library_index: Path,
) -> None:
    """`balance beam` used to return nothing: the ids use underscores."""
    d = client.get("/references/browse",
                   params={"robot": "g1", "q": "balance beam"}).json()
    assert [r["clip_id"] for r in d["rows"]] == \
        ["balance_on_beam03_poses_100_jpos"]


def test_every_query_token_must_match(
    client: TestClient, library_index: Path,
) -> None:
    """AND, not OR — otherwise a two-word query widens the result set."""
    d = client.get("/references/browse",
                   params={"robot": "g1", "q": "balance running"}).json()
    assert d["total"] == 0


def test_composed_filter_isolates_composites(
    client: TestClient, library_index: Path,
) -> None:
    only = client.get("/references/browse",
                      params={"robot": "g1", "composed": "true"}).json()
    assert [r["clip_id"] for r in only["rows"]] == ["platform-ascent--g1"]
    rest = client.get("/references/browse",
                      params={"robot": "g1", "composed": "false"}).json()
    assert only["total"] + rest["total"] == 3


def test_duration_sort_and_bounds(
    client: TestClient, library_index: Path,
) -> None:
    d = client.get("/references/browse",
                   params={"robot": "g1", "sort": "duration"}).json()
    assert [r["duration_s"] for r in d["rows"]] == [6.92, 4.0, 1.02]
    short = client.get("/references/browse",
                       params={"robot": "g1", "max_duration_s": 2.0}).json()
    assert [r["clip_id"] for r in short["rows"]] == ["run03_poses_100_jpos"]


def test_browse_does_not_shadow_the_clip_detail_route(
    client: TestClient, library_index: Path,
) -> None:
    """`/references/browse` is registered before `/references/{clip_id}`."""
    assert client.get("/references/browse",
                      params={"robot": "g1"}).status_code == 200


def test_the_legacy_list_endpoint_still_returns_a_bare_list(
    client: TestClient, library_index: Path,
) -> None:
    """Typeahead callers depend on the list shape; only the cap moved."""
    body = client.get("/references", params={"robot": "g1", "k": 2}).json()
    assert isinstance(body, list) and len(body) == 2
    paged = client.get("/references",
                       params={"robot": "g1", "k": 2, "offset": 2}).json()
    assert isinstance(paged, list) and len(paged) == 1

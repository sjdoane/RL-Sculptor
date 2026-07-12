"""tests/test_reference_context.py — offline tests for
`sculptor.reference_context`, the shared reader that threads a stage's
reference kinematic signature into the diagnose/edit LLM context.

No API calls, no reference-clip fixtures needed — `load_reference_signature`
and `render_reference_signature_block` operate purely on the LOCKED JSON
schema the mission scaffold writes, which we construct by hand here (per
the schema documented in `sculptor/refs/convert.py::kinematic_signature`).
"""

from __future__ import annotations

import json
from pathlib import Path

from sculptor.reference_context import (
    REFERENCE_SIGNATURE_FILENAME,
    load_reference_signature,
    render_reference_signature_block,
)

_VALID_PAYLOAD = {
    "schema": 1,
    "clip_id": "g1_jump_ref_01",
    "robot": "g1",
    "tier": "K",
    "text": "Reference standing long jump.",
    "signature": {
        "duration_s": 1.8,
        "fps": 30.0,
        "n_frames": 54,
        "root_z": {
            "start": 0.1, "end": 0.72, "min": 0.1, "min_t": 0.0,
            "max": 0.72, "max_t": 1.5,
        },
        "root_velocity_mps": {"min": -0.2, "max": 1.4},
        "phases": [
            {"phase": "rising", "t_start": 0.0, "t_end": 1.5,
             "z_start": 0.1, "z_end": 0.72},
        ],
    },
}


def _write(dirpath: Path, payload: dict | str) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (dirpath / REFERENCE_SIGNATURE_FILENAME).write_text(text, encoding="utf-8")


# ── load_reference_signature ───────────────────────────────────────────────
def test_load_reference_signature_present_and_valid(tmp_path: Path):
    _write(tmp_path, _VALID_PAYLOAD)
    out = load_reference_signature(tmp_path)
    assert out is not None
    assert out["clip_id"] == "g1_jump_ref_01"
    assert out["signature"]["root_z"]["max"] == 0.72


def test_load_reference_signature_accepts_a_file_path_and_uses_its_parent(
    tmp_path: Path,
):
    """diagnose.py knows `config.toml`, not the stage dir directly — the
    reader must resolve a file path to its parent."""
    _write(tmp_path, _VALID_PAYLOAD)
    config_path = tmp_path / "config.toml"
    config_path.write_text("[target]\nname='x'\n", encoding="utf-8")
    out = load_reference_signature(config_path)
    assert out is not None
    assert out["clip_id"] == "g1_jump_ref_01"


def test_load_reference_signature_absent_returns_none(tmp_path: Path):
    # Deliberately do not write the file.
    assert load_reference_signature(tmp_path) is None


def test_load_reference_signature_absent_dir_returns_none(tmp_path: Path):
    assert load_reference_signature(tmp_path / "does_not_exist") is None


def test_load_reference_signature_corrupt_json_returns_none(tmp_path: Path):
    _write(tmp_path, "{not valid json")
    assert load_reference_signature(tmp_path) is None


def test_load_reference_signature_non_dict_payload_returns_none(tmp_path: Path):
    _write(tmp_path, "[1, 2, 3]")
    assert load_reference_signature(tmp_path) is None


def test_load_reference_signature_wrong_schema_version_returns_none(
    tmp_path: Path,
):
    bad = dict(_VALID_PAYLOAD, schema=2)
    _write(tmp_path, bad)
    assert load_reference_signature(tmp_path) is None


def test_load_reference_signature_missing_schema_key_returns_none(
    tmp_path: Path,
):
    bad = {k: v for k, v in _VALID_PAYLOAD.items() if k != "schema"}
    _write(tmp_path, bad)
    assert load_reference_signature(tmp_path) is None


def test_load_reference_signature_missing_signature_key_returns_none(
    tmp_path: Path,
):
    bad = {k: v for k, v in _VALID_PAYLOAD.items() if k != "signature"}
    _write(tmp_path, bad)
    assert load_reference_signature(tmp_path) is None


def test_load_reference_signature_non_dict_signature_returns_none(
    tmp_path: Path,
):
    bad = dict(_VALID_PAYLOAD, signature="not a dict")
    _write(tmp_path, bad)
    assert load_reference_signature(tmp_path) is None


# ── render_reference_signature_block ───────────────────────────────────────
def test_render_reference_signature_block_includes_header_and_numbers():
    block = render_reference_signature_block(_VALID_PAYLOAD)
    assert block.startswith("# REFERENCE MOTION SIGNATURE")
    assert "g1_jump_ref_01" in block
    assert "g1" in block
    assert "tier: K" in block
    assert "Reference standing long jump." in block
    # The numeric signature is present verbatim (JSON-embedded).
    assert "0.72" in block
    assert "root_z" in block


def test_render_reference_signature_block_omits_optional_fields_gracefully():
    minimal = {
        "schema": 1,
        "clip_id": "clip_x",
        "robot": None,
        "tier": None,
        "text": None,
        "signature": {"duration_s": 1.0},
    }
    block = render_reference_signature_block(minimal)
    assert "clip_x" in block
    assert "tier:" not in block
    assert "description:" not in block


def test_render_reference_signature_block_handles_malformed_input():
    assert render_reference_signature_block(None) == ""  # type: ignore[arg-type]
    assert render_reference_signature_block({}) == ""
    assert render_reference_signature_block({"signature": "not a dict"}) == ""


def test_render_reference_signature_block_matches_load_round_trip(
    tmp_path: Path,
):
    """End-to-end: write -> load -> render produces a non-empty block
    containing the clip id, exactly what diagnose.py/edit.py rely on."""
    _write(tmp_path, _VALID_PAYLOAD)
    loaded = load_reference_signature(tmp_path)
    assert loaded is not None
    block = render_reference_signature_block(loaded)
    assert "g1_jump_ref_01" in block

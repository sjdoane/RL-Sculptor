"""§Ship 24 (R1): reproducibility capture tests.

Unit-level: capture shape, determinism modulo VOLATILE_KEYS, config
hashing, prompt hashing, mission provenance lifecycle (init / resume /
stage records / corrupt tolerance).
Integration: a dry-run sculpt_run writes reports/run_context.json and
emits run_context_captured with the base seed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sculptor.run_context import (
    VOLATILE_KEYS,
    capture_run_context,
    init_mission_provenance,
    mission_provenance_path,
    record_stage_in_provenance,
    write_json_atomic,
    write_run_context,
)
from tests.test_sculpt import _write_minimal_project


def _strip_volatile(ctx: dict) -> dict:
    return {k: v for k, v in ctx.items() if k not in VOLATILE_KEYS}


# ── capture_run_context ──────────────────────────────────────────────


def test_capture_has_all_required_sections(tmp_path: Path) -> None:
    proj = _write_minimal_project(tmp_path)
    ctx = capture_run_context(
        proj, proj / "config.toml",
        parsed_config={"iteration": {"seed": 7}},
        behavior_goal="hop", base_seed=7, iterations=3, start_iter=0,
    )
    for key in (
        "schema", "captured_at", "behavior_goal", "code_git", "project_git",
        "python", "packages", "prompts", "llm", "seeds", "env", "config",
    ):
        assert key in ctx, f"missing {key}"
    assert ctx["behavior_goal"] == "hop"
    assert ctx["seeds"]["base_seed"] == 7
    assert ctx["iterations"] == 3
    # The minimal project IS a git repo — sha must resolve.
    assert ctx["project_git"]["available"] is True
    assert len(ctx["project_git"]["sha"]) == 40
    # sculptor's own repo too (this checkout).
    assert ctx["code_git"]["available"] is True
    # numpy is a hard dep of the test env.
    assert "numpy" in ctx["packages"]


def test_capture_config_hash_matches_file(tmp_path: Path) -> None:
    proj = _write_minimal_project(tmp_path)
    cfg_path = proj / "config.toml"
    ctx = capture_run_context(proj, cfg_path)
    assert ctx["config"]["sha256"] == hashlib.sha256(
        cfg_path.read_bytes()).hexdigest()
    # Effective (post-override) config recorded verbatim when provided.
    ctx2 = capture_run_context(
        proj, cfg_path, parsed_config={"iteration": {"steps_per_iter": 5}})
    assert ctx2["config"]["effective"] == {"iteration": {"steps_per_iter": 5}}


def test_capture_deterministic_modulo_volatile(tmp_path: Path) -> None:
    """Two captures of identical state differ ONLY in VOLATILE_KEYS —
    this is the property that makes run_context.json diffable across
    reruns."""
    proj = _write_minimal_project(tmp_path)
    a = capture_run_context(proj, proj / "config.toml", base_seed=1)
    b = capture_run_context(proj, proj / "config.toml", base_seed=1)
    assert _strip_volatile(a) == _strip_volatile(b)


def test_dirty_capture_hashes_patch_and_untracked_bytes_deterministically(
    tmp_path: Path,
) -> None:
    proj = _write_minimal_project(tmp_path)
    config = proj / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + "\n# local experiment\n",
        encoding="utf-8",
    )
    untracked = proj / "research-note.bin"
    untracked.write_bytes(b"candidate-a\x00")

    first = capture_run_context(proj)["project_git"]
    second = capture_run_context(proj)["project_git"]
    assert first["dirty"] is True
    assert len(first["diff_sha256"]) == 64
    assert first["diff_sha256"] == second["diff_sha256"]
    assert first["untracked_count"] == 1

    untracked.write_bytes(b"candidate-b\x00")
    changed = capture_run_context(proj)["project_git"]
    assert changed["diff_sha256"] != first["diff_sha256"]


def test_prompt_hashes_present_and_stable(tmp_path: Path) -> None:
    proj = _write_minimal_project(tmp_path)
    a = capture_run_context(proj)["prompts"]
    b = capture_run_context(proj)["prompts"]
    assert "sha256" in a, a
    assert a["sha256"], "no prompts hashed"
    # Known pipeline prompts must be covered.
    for name in ("decompose_task", "diagnose_grounded", "edit_rewriter"):
        assert name in a["sha256"], f"prompt {name} not hashed"
        assert len(a["sha256"][name]) == 64
    assert a == b


def test_llm_specs_have_model_ids(tmp_path: Path) -> None:
    proj = _write_minimal_project(tmp_path)
    llm = capture_run_context(proj)["llm"]
    for key in ("decompose", "diagnose", "edit"):
        assert key in llm
        assert llm[key].get("model_id", "").startswith("claude"), llm[key]
        assert llm[key]["temperature"] == "sdk_default"


def test_no_secrets_in_env_section(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SCULPTOR_REMOTE_KEY_PATH", "/home/x/.ssh/id_secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    proj = _write_minimal_project(tmp_path)
    blob = json.dumps(capture_run_context(proj, proj / "config.toml"))
    assert "sk-ant-secret-value" not in blob
    assert "id_secret" not in blob
    # ...but the fact that remote vars were SET is recorded.
    ctx = capture_run_context(proj)
    assert "SCULPTOR_REMOTE_KEY_PATH" in ctx["env"]["sculptor_vars_set"]


def test_capture_persists_full_warm_start_policy_contract_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    receipt = {
        "schema": 1,
        "source": {"contract_sha256": "a" * 64},
        "target": {"contract_sha256": "b" * 64},
        "compatibility": {"type": "exact_policy_contract"},
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    monkeypatch.setenv(
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON",
        encoded,
    )
    proj = _write_minimal_project(tmp_path)

    captured = capture_run_context(proj)

    admitted = captured["warm_start_policy_contract"]
    assert admitted["receipt"] == receipt
    assert admitted["receipt_sha256"] == hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()


def test_write_run_context_round_trip(tmp_path: Path) -> None:
    proj = _write_minimal_project(tmp_path)
    ctx = capture_run_context(proj, proj / "config.toml", base_seed=42)
    out = write_run_context(proj / "reports", ctx)
    assert out == proj / "reports" / "run_context.json"
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["seeds"]["base_seed"] == 42


def test_write_json_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "x" / "ctx.json", {"a": 1})
    leftovers = [p for p in (tmp_path / "x").iterdir() if p.suffix == ".tmp"]
    assert not leftovers
    assert json.loads((tmp_path / "x" / "ctx.json").read_text()) == {"a": 1}


def test_prompts_dir_override_changes_hashes(tmp_path: Path, monkeypatch) -> None:
    """SCULPTOR_PROMPTS_DIR points capture at the OVERRIDE prompts — a
    tuned prompt set is a different experiment and must hash differently."""
    override = tmp_path / "prompts"
    override.mkdir()
    (override / "custom_only.md").write_text("tuned prompt", encoding="utf-8")
    monkeypatch.setenv("SCULPTOR_PROMPTS_DIR", str(override))
    proj = _write_minimal_project(tmp_path)
    prompts = capture_run_context(proj)["prompts"]
    assert prompts["dir"] == str(override.resolve())
    assert set(prompts["sha256"]) == {"custom_only"}
    assert prompts["sha256"]["custom_only"] == hashlib.sha256(
        b"tuned prompt").hexdigest()


def test_non_repo_project_degrades(tmp_path: Path) -> None:
    bare = tmp_path / "not_a_repo"
    bare.mkdir()
    # Use the dir itself as "project" — and neutralize repo discovery
    # walking up into THIS checkout's repo by checking only the shape.
    ctx = capture_run_context(bare)
    assert "available" in ctx["project_git"]
    # dirty must be True/False/None — never fabricated from an error.
    if ctx["project_git"].get("available"):
        assert ctx["project_git"]["dirty"] in (True, False, None)


def test_datetime_in_config_serializes(tmp_path: Path) -> None:
    """tomllib parses TOML dates into datetime objects — default=str in
    write_json_atomic is what keeps capture from crashing on them."""
    import datetime as dt

    proj = _write_minimal_project(tmp_path)
    ctx = capture_run_context(
        proj, proj / "config.toml",
        parsed_config={"meta": {"created": dt.datetime(2026, 6, 10, 12, 0)}},
    )
    out = write_run_context(proj / "reports", ctx)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "2026-06-10" in loaded["config"]["effective"]["meta"]["created"]


# ── mission provenance ───────────────────────────────────────────────


def test_mission_provenance_init_and_stage_records(tmp_path: Path) -> None:
    md = tmp_path / "mission"
    md.mkdir()
    path = init_mission_provenance(md, goal="do a flip", n_stages=3, seed=5)
    assert path == mission_provenance_path(md)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["goal"] == "do a flip"
    assert doc["n_stages_initial"] == 3
    assert len(doc["contexts"]) == 1
    assert doc["contexts"][0]["seeds"]["base_seed"] == 5
    assert doc["stages"] == []

    record_stage_in_provenance(md, {
        "stage_name": "crouch", "status": "succeeded",
        "iterations_used": 4, "criterion_satisfied": True,
    })
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["stages"]) == 1
    assert doc["stages"][0]["stage_name"] == "crouch"
    assert "recorded_at" in doc["stages"][0]


def test_mission_provenance_resume_appends_context(tmp_path: Path) -> None:
    """A resumed mission may run DIFFERENT code — every resume appends
    its own context while stage records survive."""
    md = tmp_path / "mission"
    md.mkdir()
    init_mission_provenance(md, goal="g", n_stages=2, seed=1)
    record_stage_in_provenance(md, {"stage_name": "s0", "status": "succeeded"})
    init_mission_provenance(md, goal="g", n_stages=2, seed=1)  # resume
    doc = json.loads(mission_provenance_path(md).read_text(encoding="utf-8"))
    assert len(doc["contexts"]) == 2
    assert len(doc["stages"]) == 1, "stage records must survive resume"


def test_mission_provenance_corrupt_file_recovers(tmp_path: Path) -> None:
    md = tmp_path / "mission"
    md.mkdir()
    mission_provenance_path(md).write_text("{broken", encoding="utf-8")
    # init recovers with a fresh doc...
    init_mission_provenance(md, goal="g", n_stages=1, seed=None)
    doc = json.loads(mission_provenance_path(md).read_text(encoding="utf-8"))
    assert doc["goal"] == "g"
    # ...and record on a corrupt file is a silent no-op, never a raise.
    mission_provenance_path(md).write_text("{broken", encoding="utf-8")
    assert record_stage_in_provenance(md, {"stage_name": "x"}) is None


# ── integration: sculpt_run writes the snapshot ──────────────────────


def test_sculpt_run_writes_run_context(tmp_path: Path, capsys) -> None:
    import tests.test_sculpt as ts

    ts._SCHEDULE = [1.0, 2.0]
    proj = _write_minimal_project(tmp_path)
    from sculptor.sculpt import sculpt_run

    result = sculpt_run(
        config_path=proj / "config.toml", behavior_goal="dummy goal",
        iterations=1, resume=False, no_kg=True, dry_run=True, seed=123,
    )
    assert result.iterations_run == 1

    rc_path = proj / "reports" / "run_context.json"
    assert rc_path.is_file(), "run_context.json not written"
    rc = json.loads(rc_path.read_text(encoding="utf-8"))
    assert rc["seeds"]["base_seed"] == 123          # CLI seed threaded
    assert rc["behavior_goal"] == "dummy goal"
    assert rc["config"]["sha256"]
    assert rc["config"]["effective"]["iteration"]["seed"] == 123
    assert rc["project_git"]["available"] is True

    out = capsys.readouterr().out
    events = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines()
              if l.startswith("[SCULPT-EVENT] ")]
    started = [e for e in events if e["type"] == "run_started"]
    assert started and started[0]["base_seed"] == 123
    captured = [e for e in events if e["type"] == "run_context_captured"]
    assert captured, "run_context_captured event missing"
    assert captured[0]["base_seed"] == 123
    assert captured[0]["config_sha256"] == rc["config"]["sha256"]
    # The citation-provenance file (reports/provenance.json) keeps its
    # own writer + shape — run-context keys must never leak into it.
    cite = proj / "reports" / "provenance.json"
    if cite.exists():
        cite_doc = json.loads(cite.read_text(encoding="utf-8"))
        assert "code_git" not in cite_doc
        assert "schema" not in cite_doc

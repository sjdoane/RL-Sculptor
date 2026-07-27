"""Metric-gauntlet study packets and human-anchor analysis (no GPU/LLM)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sculptor.eval.gauntlet import (
    GauntletError,
    analyze_blind_study,
    build_blind_study,
    load_and_verify_study_key,
)


def _manifest(tmp_path: Path) -> Path:
    videos = tmp_path / "private_videos"
    videos.mkdir()
    definitions = [
        ("competent_secret", "competent", 3, 0.80),
        # Deliberately metric-gamed: lower human/ground-truth competence but
        # the evaluator scores it above the competent rollout.
        ("gamed_secret", "metric_gaming", 1, 0.90),
        ("partial_secret", "partial", 2, 0.50),
        ("fallen_secret", "safety_failure", 0, 0.10),
    ]
    items = []
    for index, (item_id, behavior_class, rank, score) in enumerate(definitions):
        video = videos / f"condition_{item_id}.mp4"
        video.write_bytes(f"synthetic-video-{index}".encode())
        items.append({
            "item_id": item_id,
            "comparison_group": "arm_reach_block_v1",
            "task_id": "arm_reach_block",
            "task_prompt": "Reach the red block and hold without collision.",
            "robot_id": "generic_arm_7dof",
            "embodiment_family": "robot_arm",
            "motion_family": "reach",
            "behavior_class": behavior_class,
            "competence_rank": rank,
            "artifact_path": str(video),
            "evaluator_score": score,
            "private_condition": f"condition_{index}",
        })
    manifest = {
        "schema_version": 1,
        "study_id": "arm-metric-anchor-v1",
        "rubric": "Prefer task completion, stability, and collision avoidance.",
        "evaluator": {
            "evaluator_id": "generated_metric_arm_reach",
            "evaluator_version": "sha256:test-only",
            "score_semantics": "higher means predicted task competence",
            "higher_is_better": True,
        },
        "items": items,
    }
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def built_study(tmp_path: Path, monkeypatch):
    # Unit tests exercise blinding, pairing, hashing, and statistics. Real
    # packet builds use the default ffmpeg metadata-stripping remux.
    monkeypatch.setattr(
        "sculptor.eval.gauntlet._sanitize_video",
        lambda source, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            shutil.copyfile(source, destination),
        ),
    )
    manifest = _manifest(tmp_path)
    out = tmp_path / "study"
    summary = build_blind_study(
        manifest,
        out,
        seed=20260719,
        forms=2,
        max_pairs_per_group=20,
        reliability_repeats=1,
        evaluator_tie_band=0.01,
    )
    return out, summary


def test_build_is_blinded_hashed_and_counterbalanced(built_study) -> None:
    out, summary = built_study
    assert summary["n_primary_pairs"] == 6  # all C(4, 2) unequal-rank pairs
    assert summary["presentations_per_form"] == 7

    key = load_and_verify_study_key(out / "study_key.json")
    packet_a = json.loads(
        (out / "study_packet_form_A.json").read_text(encoding="utf-8")
    )
    packet_b = json.loads(
        (out / "study_packet_form_B.json").read_text(encoding="utf-8")
    )
    public_text = json.dumps([packet_a, packet_b])
    for secret in (
        "competent_secret", "gamed_secret", "partial_secret", "fallen_secret",
        "private_condition", "condition_gamed_secret", "evaluator_score",
        "competence_rank", "metric_gaming",
    ):
        assert secret not in public_text
    assert all(pair["artifact_a"].startswith("assets/form_A/")
               for pair in packet_a["pairs"])

    for pair in key["pairs"].values():
        assert pair["presentations"]["A"]["a_item_id"] == (
            pair["presentations"]["B"]["b_item_id"]
        )
        assert pair["presentations"]["A"]["b_item_id"] == (
            pair["presentations"]["B"]["a_item_id"]
        )


def _perfect_responses(out: Path) -> Path:
    key = load_and_verify_study_key(out / "study_key.json")
    rows = []
    for form, rater in (("A", "rater_001"), ("B", "rater_002")):
        for pair in key["pairs"].values():
            presentation = pair["presentations"][form]
            choice = (
                "A" if presentation["a_item_id"] == pair["expected_item_id"]
                else "B"
            )
            rows.append({
                "study_id": key["study_id"],
                "form_id": form,
                "pair_id": pair["pair_id"],
                "rater_id": rater,
                "choice": choice,
                "confidence": 0.9,
                "duration_seconds": 8.0,
                "notes": "",
            })
    path = out / "responses.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    return path


def test_analysis_measures_human_anchor_false_competence_and_bias(
    built_study, tmp_path: Path,
) -> None:
    out, _ = built_study
    responses = _perfect_responses(out)
    analysis = analyze_blind_study(
        out / "study_key.json", responses, tmp_path / "analysis",
    )

    assert analysis["counts"]["raters"] == 2
    assert analysis["human"]["pair_majority_accuracy_vs_expected"]["point"] == 1.0
    assert analysis["human"]["repeat_self_consistency"]["point"] == 1.0
    assert analysis["human"]["krippendorff_alpha_nominal"] == 1.0
    assert analysis["evaluator"]["pair_accuracy_vs_human_majority"]["point"] < 1.0
    false_gaming = analysis["evaluator"][
        "false_competence_by_lower_behavior_class"
    ]["metric_gaming"]
    assert false_gaming["rate"] == 1.0
    # Counterbalancing makes identical preferences choose display A exactly
    # half the time across the two forms.
    assert analysis["order_bias_display_a_choice_rate"]["overall"]["rate"] == 0.5
    assert "robot_arm" in analysis["dimensions"]["embodiment_family"]
    assert len(analysis["analysis_sha256"]) == 64
    assert (tmp_path / "analysis" / "gauntlet_analysis.json").is_file()
    assert (tmp_path / "analysis" / "gauntlet_analysis.md").is_file()


def test_tampered_key_and_duplicate_responses_are_rejected(
    built_study, tmp_path: Path,
) -> None:
    out, _ = built_study
    key_path = out / "study_key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    key["design"]["seed"] += 1
    tampered = tmp_path / "tampered_key.json"
    tampered.write_text(json.dumps(key), encoding="utf-8")
    with pytest.raises(GauntletError, match="hash mismatch"):
        load_and_verify_study_key(tampered)

    responses = _perfect_responses(out)
    first = responses.read_text(encoding="utf-8").splitlines()[0]
    with responses.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")
    with pytest.raises(GauntletError, match="duplicate response"):
        analyze_blind_study(key_path, responses, tmp_path / "analysis")


def test_build_rejects_nonempty_output_and_duplicate_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sculptor.eval.gauntlet._sanitize_video",
        lambda source, destination: shutil.copyfile(source, destination),
    )
    manifest = _manifest(tmp_path)
    out = tmp_path / "not_empty"
    out.mkdir()
    (out / "old_packet.json").write_text("{}")
    with pytest.raises(GauntletError, match="not empty"):
        build_blind_study(manifest, out)

    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["items"][1]["artifact_path"] = doc["items"][0]["artifact_path"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(GauntletError, match="pseudo-replication"):
        build_blind_study(duplicate, tmp_path / "duplicate_out")
